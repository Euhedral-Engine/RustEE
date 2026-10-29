//! Execute in any order, conclude in offer order (Java's `Sequencer`).
//!
//! Entries are offered in the order they must conclude and execute in any order. An entry that completes marks itself
//! [ready](Entry::ready) and calls [`drain`](Sequencer::drain). Whichever thread finds the head ready concludes the
//! ready prefix, in offer order, through the sequencer's `drained` function. A work-in-progress count makes exactly
//! one thread drain at a time, without a lock, and a completion that arrives during a drain makes that drain look
//! again instead of waiting.
//!
//! - **Any thread may offer**: the order is a growing multi-producer queue, so the order is the order of the offers,
//!   and an offer is never refused.
//! - **The entry is the caller's own object** and carries its readiness, so the sequencer allocates nothing for it
//!   (the queue recycles its chunks). A caller that pools its entries makes the whole path allocation-free.
//! - [`Progress`] is the readiness word an entry embeds: pending, ready or abandoned. An abandoned entry is ready
//!   (it concludes with nothing). Whoever owns an entry abandons it when it dies, or the order waits for it: only a
//!   [`Slot`] does that by itself, when it is dropped.
//! - [`Ticket`] is an entry that carries a value and [`Slot`] the holder's handle to one, for callers that do not need
//!   their own entry type.

#![allow(
    unsafe_code,
    reason = "the thread that wins the work-in-progress count owns the order's consumer side; a ticket's value is written before its progress word and read after it"
)]

use crate::sync::{AtomicU8, AtomicUsize, Ordering, UnsafeCell};
use euhedral_data_structures::atomics::CachePadded;
use euhedral_data_structures::queues::{Consumer, MpscQueue, Multi, Producer, Single};
use euhedral_data_structures::sync::HotScope;
use std::fmt;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::sync::Arc;

/// Something concluded in order.
pub trait Entry {
    /// Whether the entry completed and may conclude once everything offered before it concluded.
    fn ready(&self) -> bool;
}

const PENDING: u8 = 0;
const READY: u8 = 1;
const ABANDONED: u8 = 2;
/// A ready ticket whose value was taken.
const TAKEN: u8 = 3;

/// The readiness word of an entry: pending until it finishes or is abandoned, and ready from then on.
///
/// The finishing thread stores it with release ordering after it wrote whatever the entry carries, and the thread
/// that concludes reads it with acquire ordering, so a ready entry's data is visible to the conclusion.
pub struct Progress {
    state: AtomicU8,
}

impl Progress {
    /// A pending entry.
    pub fn new() -> Self {
        Self {
            state: AtomicU8::new(PENDING),
        }
    }

    /// Whether the entry finished or was abandoned.
    pub fn ready(&self) -> bool {
        self.state.load(Ordering::Acquire) != PENDING
    }

    /// Whether the entry was abandoned: ready, with nothing to conclude.
    pub fn is_abandoned(&self) -> bool {
        self.state.load(Ordering::Acquire) == ABANDONED
    }

    /// Marks the entry finished. Write what the entry carries first.
    pub fn finish(&self) {
        self.state.store(READY, Ordering::Release);
    }

    /// Marks the entry abandoned: it concludes with nothing.
    pub fn abandon(&self) {
        self.state.store(ABANDONED, Ordering::Release);
    }

    /// Makes a concluded entry pending again, for a caller that pools its entries. Only after the entry concluded and
    /// before it is offered again.
    pub fn reset(&self) {
        self.state.store(PENDING, Ordering::Release);
    }
}

impl Default for Progress {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for Progress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Progress")
            .field("ready", &self.ready())
            .finish()
    }
}

/// Concludes entries in the order they were offered, whatever order they complete in.
///
/// - [`offer`](Self::offer) appends an entry to the order, from any thread.
/// - An entry that became ready calls [`drain`](Self::drain). One drainer at a time (a work-in-progress counter:
///   whoever finds it busy leaves its work to the drainer) concludes the prefix of ready entries in order through
///   `drained`, which runs on the draining thread and must not wait.
///
/// A panic in `drained` is the entry's own: the entries behind it still conclude, and the panic resumes on the
/// draining thread once the drain ended (the sequencer is never left claimed).
pub struct Sequencer<E: Entry + Send + 'static> {
    producer: Producer<E, Multi>,
    /// The order's consumer side: used by the thread that holds the work-in-progress count only.
    consumer: UnsafeCell<Consumer<E, Single>>,
    /// The work-in-progress count: the thread that raises it from 0 drains; the others leave their work to it.
    wip: CachePadded<AtomicUsize>,
    drained: Box<dyn Fn(&E) + Send + Sync>,
}

// SAFETY: the consumer is used only by the thread that raised the work-in-progress count from zero (acquire on the
// raise, release on the lowering hand it from one such thread to the next); the producer is `Sync` and `E: Send`.
unsafe impl<E: Entry + Send + 'static> Send for Sequencer<E> {}
// SAFETY: as above.
unsafe impl<E: Entry + Send + 'static> Sync for Sequencer<E> {}

impl<E: Entry + Send + 'static> fmt::Debug for Sequencer<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Sequencer")
            .field("len", &self.len())
            .finish_non_exhaustive()
    }
}

/// The chunk size of the order, and the chunks it keeps for reuse.
const ORDER_CHUNK: usize = 16;
const ORDER_POOLED: usize = 2;

impl<E: Entry + Send + 'static> Sequencer<E> {
    /// A sequencer that concludes each entry through `drained`, after every entry offered before it.
    pub fn new(drained: impl Fn(&E) + Send + Sync + 'static) -> Arc<Self> {
        let order = MpscQueue::<E>::with_pool(ORDER_CHUNK, ORDER_POOLED);
        Arc::new(Self {
            producer: order.producer(),
            consumer: UnsafeCell::new(order.consumer()),
            wip: CachePadded::new(AtomicUsize::new(0)),
            drained: Box::new(drained),
        })
    }

    /// Appends `entry` to the order. An entry that is already ready when it is offered concludes at the next
    /// [`drain`](Self::drain): offering does not drain, so call it after the offer.
    pub fn offer(&self, entry: E) {
        if self.producer.offer(entry).is_err() {
            // The order is unbounded: this does not happen.
            panic!("the sequencer refused an entry");
        }
    }

    /// Entries offered and not yet concluded.
    pub fn len(&self) -> usize {
        usize::try_from(self.producer.size_long()).unwrap_or(usize::MAX)
    }

    /// Whether every entry offered so far has concluded.
    pub fn is_empty(&self) -> bool {
        self.producer.is_empty()
    }

    /// Java's `drain`: concludes the ready prefix, unless another thread is draining: that one then concludes whatever
    /// became ready, this call's entry included.
    pub fn drain(&self) {
        let _hot = HotScope::enter();
        if self.wip.fetch_add(1, Ordering::AcqRel) != 0 {
            return;
        }
        // The count went from 0 to 1 on this thread, so no other thread is in this function until the count returns
        // to 0 below; the consumer belongs to this thread meanwhile. Everything the entries run (`ready`, the conclusion,
        // the drop of an entry) is inside `catch_unwind`: a panic there is the entry's own and must not leave the count
        // raised, which would stop every later drain.
        let mut panic = None;
        let mut missed = 1;
        loop {
            // SAFETY: as above; the access ends before the count is lowered.
            self.consumer.with_mut(|consumer| {
                // SAFETY: as above.
                let consumer = unsafe { &mut *consumer };
                loop {
                    let head_ready = catch_unwind(AssertUnwindSafe(|| {
                        consumer.peek().map(|head| head.ready())
                    }));
                    let ready = match head_ready {
                        Ok(ready) => ready.unwrap_or(false),
                        Err(payload) => {
                            // An entry whose `ready` panics is concluded and removed: it would stall the order for good.
                            panic.get_or_insert(payload);
                            true
                        }
                    };
                    if !ready {
                        break;
                    }
                    // The entry stays in the order while it concludes, as Java's `finally` polls it afterwards.
                    if let Err(payload) = catch_unwind(AssertUnwindSafe(|| {
                        if let Some(head) = consumer.peek() {
                            (self.drained)(head);
                        }
                    })) {
                        panic.get_or_insert(payload);
                    }
                    if let Err(payload) = catch_unwind(AssertUnwindSafe(|| drop(consumer.poll()))) {
                        panic.get_or_insert(payload);
                    }
                }
            });
            missed = self.wip.fetch_sub(missed, Ordering::AcqRel) - missed;
            if missed == 0 {
                break;
            }
        }
        if let Some(payload) = panic {
            if std::thread::panicking() {
                // A `Slot` dropped while its thread unwinds drains here, and a second panic would abort the process.
                log::error!("an entry failed to conclude while the thread was already panicking");
            } else {
                resume_unwind(payload);
            }
        }
    }
}

/// An entry that carries a value from the thread that completes it to the function that concludes it.
pub struct Ticket<T> {
    state: AtomicU8,
    /// Written once by the holder before `state` is stored `READY`, taken once by the concluding thread after it
    /// loads `READY`.
    value: UnsafeCell<Option<T>>,
}

// SAFETY: the value is written before the state word (release) and taken after it (acquire), by one thread each,
// and taken once (the take changes the state with a compare-and-set).
unsafe impl<T: Send> Send for Ticket<T> {}
// SAFETY: as above.
unsafe impl<T: Send> Sync for Ticket<T> {}

impl<T> fmt::Debug for Ticket<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Ticket").finish_non_exhaustive()
    }
}

impl<T> Ticket<T> {
    /// A pending ticket.
    pub fn new() -> Self {
        Self {
            state: AtomicU8::new(PENDING),
            value: UnsafeCell::new(None),
        }
    }

    /// Completes the ticket with `value`. Only the ticket's holder does, once.
    fn complete(&self, value: T) {
        self.value.with_mut(|slot| {
            // SAFETY: the holder is the only writer, and the concluding thread reads only after it sees `READY`.
            unsafe { *slot = Some(value) }
        });
        self.state.store(READY, Ordering::Release);
    }

    fn abandon(&self) {
        self.state.store(ABANDONED, Ordering::Release);
    }

    /// The value of a completed ticket, once; `None` for an abandoned or pending one, or after the take.
    pub fn take(&self) -> Option<T> {
        self.state
            .compare_exchange(READY, TAKEN, Ordering::Acquire, Ordering::Relaxed)
            .ok()?;
        // SAFETY: the value was written before `READY` (acquired above), and the compare-and-set let one caller in.
        self.value.with_mut(|slot| unsafe { (*slot).take() })
    }
}

impl<T> Default for Ticket<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Entry for Ticket<T> {
    fn ready(&self) -> bool {
        self.state.load(Ordering::Acquire) != PENDING
    }
}

/// A shared entry is an entry: callers pool their entries behind an `Arc` and offer clones of the handle.
impl<E: Entry + ?Sized> Entry for Arc<E> {
    fn ready(&self) -> bool {
        (**self).ready()
    }
}

impl<T: Send + 'static> Sequencer<Arc<Ticket<T>>> {
    /// A sequencer of tickets that hands each completed value to `emit`, in offer order; abandoned tickets pass.
    pub fn ordered(emit: impl Fn(T) + Send + Sync + 'static) -> Arc<Self> {
        Self::new(move |ticket: &Arc<Ticket<T>>| {
            if let Some(value) = ticket.take() {
                emit(value);
            }
        })
    }

    /// Offers a new ticket and returns its holder's handle. A caller that pools its entries offers them itself.
    pub fn register(self: &Arc<Self>) -> Slot<T> {
        let ticket = Arc::new(Ticket::new());
        self.offer(ticket.clone());
        Slot {
            held: Some(Held {
                ticket,
                sequencer: self.clone(),
            }),
        }
    }
}

/// A sequencer of tickets.
type TicketOrder<T> = Sequencer<Arc<Ticket<T>>>;

/// The holder's handle to a [`Ticket`] in a sequencer's order. Complete it or abandon it; dropping it abandons it, so
/// a frame that dies holding one does not stall the order (an entry of the caller's own type has no such guarantee).
///
/// A slot keeps its sequencer alive (an `Arc`): without a collector, a completion on another thread must not outlive
/// the order it writes to.
pub struct Slot<T: Send + 'static> {
    held: Option<Held<T>>,
}

/// What a slot holds until it is completed or abandoned.
struct Held<T: Send + 'static> {
    ticket: Arc<Ticket<T>>,
    sequencer: Arc<TicketOrder<T>>,
}

impl<T: Send + 'static> fmt::Debug for Slot<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Slot").finish_non_exhaustive()
    }
}

impl<T: Send + 'static> Slot<T> {
    /// The value for this position; concluded once every earlier entry concluded.
    pub fn complete(mut self, value: T) {
        if let Some(Held { ticket, sequencer }) = self.held.take() {
            ticket.complete(value);
            sequencer.drain();
        }
    }

    /// This position has nothing to conclude (its frame was killed).
    pub fn abandon(mut self) {
        self.finish_abandoned();
    }

    fn finish_abandoned(&mut self) {
        if let Some(Held { ticket, sequencer }) = self.held.take() {
            ticket.abandon();
            sequencer.drain();
        }
    }
}

impl<T: Send + 'static> Drop for Slot<T> {
    fn drop(&mut self) {
        self.finish_abandoned();
    }
}

/// Ports of the reference tests of the in-order conclusion: results concluded in offer order after out-of-order
/// execution, a completion waiting for the lowest outstanding one, killed frames passed, a drain that never recurses,
/// a failing conclusion that does not stop the drain, and completions from many threads.
#[cfg(all(test, not(loom)))]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize as StdAtomicUsize, Ordering as StdOrdering};

    type Recording = (Arc<Sequencer<Arc<Ticket<u64>>>>, Arc<Mutex<Vec<u64>>>);

    fn recording() -> Recording {
        let out = Arc::new(Mutex::new(Vec::new()));
        let sink = out.clone();
        let sequencer = Sequencer::ordered(move |value| sink.lock().unwrap().push(value));
        (sequencer, out)
    }

    /// FrameSequencerTest.emitsResultsInInputOrderAfterOutOfOrderExecution
    #[test]
    fn values_conclude_in_offer_order_after_out_of_order_completion() {
        let (sequencer, out) = recording();
        let first = sequencer.register();
        let second = sequencer.register();
        let third = sequencer.register();
        third.complete(3);
        second.complete(2);
        assert!(out.lock().unwrap().is_empty(), "the head is not ready");
        assert_eq!(sequencer.len(), 3);
        first.complete(1);
        assert_eq!(*out.lock().unwrap(), [1, 2, 3]);
        assert!(sequencer.is_empty());
    }

    /// OffsetCollectorTest.partitionCollectorWaitsForTheLowestOutstandingOffset
    #[test]
    fn a_completion_waits_for_the_lowest_outstanding_entry() {
        let (sequencer, out) = recording();
        let lower = sequencer.register();
        let higher = sequencer.register();
        higher.complete(11);
        assert!(out.lock().unwrap().is_empty());
        lower.complete(10);
        assert_eq!(*out.lock().unwrap(), [10, 11]);
        assert!(sequencer.is_empty());
    }

    /// OffsetCollectorTest.firstRegisteredFrameCommitsOnceAfterSuccessfulAcknowledgement
    #[test]
    fn the_first_entry_concludes_once_when_it_completes() {
        let (sequencer, out) = recording();
        sequencer.register().complete(4);
        assert_eq!(*out.lock().unwrap(), [4]);
        assert!(sequencer.is_empty());
    }

    /// FrameSequencerTest.sourceErrorsTerminateOutputAndCancelPendingFrames: killed frames abandon their slots
    /// (explicitly or by dropping them), and the in-order conclusion passes them.
    #[test]
    fn abandoned_and_dropped_slots_are_passed() {
        let (sequencer, out) = recording();
        let a = sequencer.register();
        let b = sequencer.register();
        let c = sequencer.register();
        let d = sequencer.register();
        d.complete(4);
        b.abandon();
        a.complete(1);
        assert_eq!(*out.lock().unwrap(), [1]);
        drop(c);
        assert_eq!(*out.lock().unwrap(), [1, 4]);
        assert!(sequencer.is_empty());
    }

    /// SequencerTest.anEntryOfferedAndCompletedWhileDrainingIsDrainedByTheSameLoop: a completion during a drain never
    /// recurses into the drain; the running drain takes it.
    #[test]
    fn an_entry_offered_and_completed_while_draining_is_concluded_by_the_same_drain() {
        let out = Arc::new(Mutex::new(Vec::new()));
        let pending: Arc<Mutex<Option<Slot<u64>>>> = Arc::new(Mutex::new(None));
        let (sink, next) = (out.clone(), pending.clone());
        let sequencer = Sequencer::ordered(move |value: u64| {
            sink.lock().unwrap().push(value);
            let slot = next.lock().unwrap().take();
            if let Some(slot) = slot {
                slot.complete(value + 1);
                assert_eq!(
                    sink.lock().unwrap().len(),
                    1,
                    "a drain never recurses into itself"
                );
            }
        });
        let first = sequencer.register();
        *pending.lock().unwrap() = Some(sequencer.register());
        first.complete(0);
        assert_eq!(*out.lock().unwrap(), [0, 1]);
        assert!(sequencer.is_empty());
    }

    /// SequencerTest.aFailingEntryDoesNotStopTheDrain
    #[test]
    fn a_panicking_conclusion_does_not_stop_the_drain_or_leave_it_claimed() {
        let out = Arc::new(Mutex::new(Vec::new()));
        let sink = out.clone();
        let sequencer = Sequencer::ordered(move |value: u64| {
            sink.lock().unwrap().push(value);
            assert_ne!(value, 0, "injected");
        });
        let first = sequencer.register();
        let second = sequencer.register();
        second.complete(1);
        let panicked = catch_unwind(AssertUnwindSafe(|| first.complete(0)));
        assert!(panicked.is_err(), "the panic resumes after the drain");
        assert_eq!(*out.lock().unwrap(), [0, 1]);
        sequencer.register().complete(2);
        assert_eq!(
            *out.lock().unwrap(),
            [0, 1, 2],
            "the sequencer is not left claimed"
        );
        assert!(sequencer.is_empty());
    }

    /// SequencerTest.entriesCompletingOnManyThreadsDrainOnceEachInOfferOrderOnOneThreadAtATime
    #[test]
    #[allow(clippy::disallowed_methods)] // The test's own threads complete the slots, as the Java test's pool does.
    fn slots_completed_on_many_threads_conclude_once_each_in_order_on_one_thread_at_a_time() {
        let rounds = if cfg!(miri) { 2 } else { 20 };
        for round in 0..rounds as u64 {
            let out = Arc::new(Mutex::new(Vec::new()));
            let draining = Arc::new(StdAtomicUsize::new(0));
            let overlaps = Arc::new(StdAtomicUsize::new(0));
            let (sink, busy, overlap) = (out.clone(), draining.clone(), overlaps.clone());
            let sequencer = Sequencer::ordered(move |value: u64| {
                if busy.fetch_add(1, StdOrdering::AcqRel) != 0 {
                    overlap.fetch_add(1, StdOrdering::Relaxed);
                }
                sink.lock().unwrap().push(value);
                busy.fetch_sub(1, StdOrdering::AcqRel);
            });
            let count = if cfg!(miri) { 100 } else { 2_000u64 };
            let mut slots: Vec<(u64, Slot<u64>)> =
                (0..count).map(|i| (i, sequencer.register())).collect();
            // A deterministic shuffle (xorshift), so a failure reproduces.
            let mut state = 0x9e37_79b9_7f4a_7c15 ^ round;
            for i in (1..slots.len()).rev() {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                slots.swap(i, (state % (i as u64 + 1)) as usize);
            }
            let work = Arc::new(Mutex::new(slots));
            let threads: Vec<_> = (0..8)
                .map(|_| {
                    let work = work.clone();
                    std::thread::spawn(move || {
                        loop {
                            let next = work.lock().unwrap().pop();
                            let Some((value, slot)) = next else { return };
                            slot.complete(value);
                        }
                    })
                })
                .collect();
            for thread in threads {
                thread.join().unwrap();
            }
            assert_eq!(*out.lock().unwrap(), (0..count).collect::<Vec<_>>());
            assert_eq!(
                overlaps.load(StdOrdering::Relaxed),
                0,
                "two threads concluded at once"
            );
            assert!(sequencer.is_empty());
        }
    }

    /// A conclusion that panics does not stop the entries behind it, from a drain another thread left to this one.
    #[test]
    fn a_conclusion_that_panics_does_not_stop_the_ones_behind_it() {
        let out = Arc::new(Mutex::new(Vec::new()));
        let sink = out.clone();
        let sequencer = Sequencer::ordered(move |value: u64| {
            sink.lock().unwrap().push(value);
            assert_ne!(value % 2, 0, "injected");
        });
        let slots: Vec<_> = (0..6).map(|_| sequencer.register()).collect();
        let mut panics = 0;
        for (value, slot) in slots.into_iter().enumerate().rev() {
            if catch_unwind(AssertUnwindSafe(|| slot.complete(value as u64))).is_err() {
                panics += 1;
            }
        }
        assert_eq!(
            panics, 1,
            "the panic of the draining completion resumes once"
        );
        assert_eq!(*out.lock().unwrap(), [0, 1, 2, 3, 4, 5]);
        assert!(sequencer.is_empty());
    }

    /// A `Slot` dropped while its thread unwinds drains, and a conclusion that panics must not resume then: a second
    /// panic would abort the process.
    #[test]
    fn a_slot_dropped_during_a_panic_whose_conclusion_panics_does_not_abort() {
        let sequencer = Sequencer::new(|_: &Arc<Ticket<u64>>| panic!("injected"));
        let outer = catch_unwind(AssertUnwindSafe(|| {
            let _slot = sequencer.register();
            panic!("outer");
        }));
        assert!(outer.is_err(), "the outer panic is the one that propagates");
        assert!(
            sequencer.is_empty(),
            "the entry concluded and left the order"
        );
    }

    /// An entry whose `ready` panics is concluded and removed, and the count is not left raised.
    #[test]
    fn a_ready_that_panics_does_not_leave_the_sequencer_claimed() {
        struct Flaky(StdAtomicUsize);
        impl Entry for Flaky {
            fn ready(&self) -> bool {
                if self.0.fetch_add(1, StdOrdering::Relaxed) == 0 {
                    panic!("injected");
                }
                true
            }
        }
        let concluded = Arc::new(StdAtomicUsize::new(0));
        let counted = concluded.clone();
        let sequencer = Sequencer::new(move |_: &Flaky| {
            counted.fetch_add(1, StdOrdering::Relaxed);
        });
        sequencer.offer(Flaky(StdAtomicUsize::new(0)));
        assert!(catch_unwind(AssertUnwindSafe(|| sequencer.drain())).is_err());
        assert!(sequencer.is_empty());
        sequencer.offer(Flaky(StdAtomicUsize::new(1)));
        sequencer.drain();
        assert!(sequencer.is_empty(), "a later drain still runs");
        assert_eq!(concluded.load(StdOrdering::Relaxed), 2);
    }

    /// Java's order queue grows: there is no capacity, and an offer is never refused.
    #[test]
    fn the_order_grows_without_a_capacity() {
        let (sequencer, out) = recording();
        let total: u64 = if cfg!(miri) { 2_000 } else { 100_000 };
        let slots: Vec<_> = (0..total).map(|_| sequencer.register()).collect();
        assert_eq!(sequencer.len() as u64, total);
        for (value, slot) in slots.into_iter().enumerate().rev() {
            slot.complete(value as u64);
        }
        let out = out.lock().unwrap();
        assert!(out.iter().copied().eq(0..total));
        assert!(sequencer.is_empty());
    }

    /// Any thread may offer: the order is the order of the offers.
    #[test]
    #[allow(clippy::disallowed_methods)] // The test's own threads offer.
    fn offers_from_many_threads_conclude_in_the_order_they_were_offered() {
        let (sequencer, out) = recording();
        let offered = Arc::new(Mutex::new(Vec::new()));
        let per_thread: u64 = if cfg!(miri) { 20 } else { 500 };
        let threads: Vec<_> = (0..4u64)
            .map(|thread| {
                let (sequencer, offered) = (sequencer.clone(), offered.clone());
                std::thread::spawn(move || {
                    for i in 0..per_thread {
                        // The lock makes the offer order observable; the queue itself needs none.
                        let mut offered = offered.lock().unwrap();
                        offered.push(thread * 1000 + i);
                        let slot = sequencer.register();
                        drop(offered);
                        slot.complete(thread * 1000 + i);
                    }
                })
            })
            .collect();
        for thread in threads {
            thread.join().unwrap();
        }
        assert_eq!(*out.lock().unwrap(), *offered.lock().unwrap());
        assert!(sequencer.is_empty());
    }

    /// The caller's own entry type, with a [`Progress`] it embeds.
    struct Mine {
        progress: Progress,
        id: u64,
    }

    impl Entry for Mine {
        fn ready(&self) -> bool {
            self.progress.ready()
        }
    }

    #[test]
    fn a_callers_own_entries_conclude_in_order_and_an_abandoned_one_passes() {
        let out = Arc::new(Mutex::new(Vec::new()));
        let sink = out.clone();
        let sequencer = Sequencer::new(move |entry: &Arc<Mine>| {
            if !entry.progress.is_abandoned() {
                sink.lock().unwrap().push(entry.id);
            }
        });
        let entries: Vec<_> = (0..4)
            .map(|id| {
                Arc::new(Mine {
                    progress: Progress::new(),
                    id,
                })
            })
            .collect();
        for entry in &entries {
            sequencer.offer(entry.clone());
        }
        entries[2].progress.finish();
        sequencer.drain();
        entries[1].progress.abandon();
        sequencer.drain();
        assert!(out.lock().unwrap().is_empty(), "the head is pending");
        entries[0].progress.finish();
        sequencer.drain();
        assert_eq!(*out.lock().unwrap(), [0, 2]);
        entries[3].progress.finish();
        sequencer.drain();
        assert_eq!(*out.lock().unwrap(), [0, 2, 3]);
        assert!(sequencer.is_empty());
    }
}

/// Offers racing a drain, and completions racing each other, under every interleaving the memory model allows. The
/// order queue is not modelled: its core (`euhedral-data-structures` `queues/core.rs`) uses `std` atomics, so an offer or a
/// poll runs as one step between the model's own atomics, and the model assumes the queue is a linearizable
/// multi-producer queue (its own tests cover that).
#[cfg(all(loom, test))]
mod loom_tests {
    use super::*;
    use loom::sync::Mutex;
    use loom::thread;

    fn model_sequencer() -> (Arc<Sequencer<Arc<Ticket<u32>>>>, Arc<Mutex<Vec<u32>>>) {
        let out = Arc::new(Mutex::new(Vec::new()));
        let sink = out.clone();
        let sequencer = Sequencer::ordered(move |value| sink.lock().unwrap().push(value));
        (sequencer, out)
    }

    // The holder of the first entry completes it while another thread offers a second entry and completes that: the
    // offer and the two drains race, and the entries conclude once each, in offer order, with nothing left behind.
    //
    // The tickets are made before the threads start: the order queue is not modelled, so a ticket made on one thread
    // and found in the queue by another would look unpublished to the checker. Each ticket's value and state, which
    // are modelled, are written by the thread that completes it and read by the one that concludes it.
    #[test]
    fn loom_an_offer_racing_a_drain_concludes_every_entry_once_in_order() {
        loom::model(|| {
            let (sequencer, out) = model_sequencer();
            let (first, second) = (Arc::new(Ticket::new()), Arc::new(Ticket::new()));
            sequencer.offer(first.clone());
            let other = {
                let (sequencer, second) = (sequencer.clone(), second.clone());
                thread::spawn(move || {
                    sequencer.offer(second.clone());
                    second.complete(2);
                    sequencer.drain();
                })
            };
            first.complete(1);
            sequencer.drain();
            other.join().unwrap();
            assert_eq!(*out.lock().unwrap(), [1, 2]);
            assert!(sequencer.is_empty(), "no entry is left behind a drain");
        });
    }

    // Two completions race on entries offered in order: whichever drains concludes both, in order, once.
    #[test]
    fn loom_two_completions_conclude_in_order_whoever_drains() {
        loom::model(|| {
            let (sequencer, out) = model_sequencer();
            let (a, b) = (sequencer.register(), sequencer.register());
            let t = thread::spawn(move || b.complete(2));
            a.complete(1);
            t.join().unwrap();
            assert_eq!(*out.lock().unwrap(), [1, 2]);
            assert!(sequencer.is_empty());
        });
    }

    // An abandoned slot (dropped) races a completion behind it.
    #[test]
    fn loom_a_dropped_slot_never_stalls_the_order() {
        loom::model(|| {
            let (sequencer, out) = model_sequencer();
            let (a, b) = (sequencer.register(), sequencer.register());
            let t = thread::spawn(move || drop(a));
            b.complete(2);
            t.join().unwrap();
            assert_eq!(*out.lock().unwrap(), [2]);
            assert!(sequencer.is_empty());
        });
    }
}
