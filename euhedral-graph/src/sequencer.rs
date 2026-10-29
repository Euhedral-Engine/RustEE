//! Execute in any order, conclude in admission order.
//!
//! A ring of cells indexed by sequence number holds the values in the order their slots were registered. Each cell has
//! a state word (pending, ready, abandoned) that the finishing thread stores with release ordering after writing the
//! value and the releaser loads with acquire ordering. A work-in-progress count makes the thread that raises it from
//! zero conclude the ready prefix. A completion that arrives while a drain runs raises the count, and the drain looks
//! again before it returns.
//!
//! One thread at a time registers, through the [`Registrar`] that [`Sequencer::new`] returns with the sequencer (it is
//! not `Clone`, and registering takes `&mut self`). The order of registration is therefore the order of the sequence
//! numbers, and a registration is two plain stores and a release store, with no allocation and no read-modify-write.

#![allow(
    unsafe_code,
    reason = "the thread that wins the work-in-progress count owns the ring's consumer side; a cell's value is written before its state word and read after it"
)]

use euhedral_data_structures::atomics::CachePadded;
use euhedral_data_structures::sync::HotScope;
use std::cell::UnsafeCell;
use std::fmt;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, AtomicU32, AtomicU64, Ordering};

/// The cell is free, or its slot is registered and not finished.
const PENDING: u8 = 0;
/// The slot finished with a value in the cell.
const READY: u8 = 1;
/// The slot finished with nothing to release.
const ABANDONED: u8 = 2;

/// Releases values in the order their slots were registered, whatever order they complete in.
///
/// - [`Registrar::register`] takes a slot, in the order of the calls. The ring has a fixed capacity: registering fails
///   while that many slots are registered and not released.
/// - A slot is completed with a value, or abandoned (a killed frame), from any thread. Dropping a
///   slot abandons it, so a frame that dies holding one can never stall the sequence.
/// - One releaser at a time (a work-in-progress counter: whoever finds it busy leaves its work to
///   the releaser) releases the prefix of finished slots in order: each completed value goes to
///   `emit`, abandoned ones are passed. `emit` runs on the releasing thread and must not wait (it
///   offers the next frame).
///
/// A panic in `emit` is its own: the slots behind it are still released, and the panic resumes on
/// the releasing thread once the release ended (the releaser is never left claimed).
pub struct Sequencer<T> {
    cells: Box<[Cell<T>]>,
    mask: u64,
    /// The next sequence number to register. Written by the one registrar (a plain store: its only writer), read by
    /// the releaser.
    tail: CachePadded<AtomicU64>,
    /// The next sequence number to release. Written by the thread that holds `wip` (a plain store: its only writer
    /// at a time), read by the registrar to see whether the ring is full.
    head: CachePadded<AtomicU64>,
    /// The work-in-progress count: the thread that raises it from 0 releases; the others leave
    /// their slot to it.
    wip: CachePadded<AtomicU32>,
    emit: Box<dyn Fn(T) + Send + Sync>,
}

/// One position of the ring.
struct Cell<T> {
    /// [`PENDING`], [`READY`] or [`ABANDONED`]: the finishing thread stores it (release) after writing `value`; the
    /// releaser loads it (acquire) and resets it when the cell is free again.
    state: AtomicU8,
    /// Written once by the slot's holder before `state` is stored, taken by the releaser after it loads `state`.
    value: UnsafeCell<Option<T>>,
}

// SAFETY: the value is written before the state word (release) and read after it (acquire), by one thread each; the
// emitter is `Send + Sync`.
unsafe impl<T: Send> Send for Sequencer<T> {}
// SAFETY: as above.
unsafe impl<T: Send> Sync for Sequencer<T> {}

impl<T> fmt::Debug for Sequencer<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Sequencer")
            .field("capacity", &self.cells.len())
            .finish_non_exhaustive()
    }
}

impl<T: Send + 'static> Sequencer<T> {
    /// A sequencer with room for `capacity` unreleased slots (rounded up to a power of two, at least two), and the
    /// registrar that takes its slots.
    pub fn new(
        capacity: usize,
        emit: impl Fn(T) + Send + Sync + 'static,
    ) -> (Arc<Self>, Registrar<T>) {
        let capacity = capacity.max(2).next_power_of_two();
        let sequencer = Arc::new(Self {
            cells: (0..capacity)
                .map(|_| Cell {
                    state: AtomicU8::new(PENDING),
                    value: UnsafeCell::new(None),
                })
                .collect(),
            mask: capacity as u64 - 1,
            tail: CachePadded::new(AtomicU64::new(0)),
            head: CachePadded::new(AtomicU64::new(0)),
            wip: CachePadded::new(AtomicU32::new(0)),
            emit: Box::new(emit),
        });
        let registrar = Registrar {
            sequencer: sequencer.clone(),
        };
        (sequencer, registrar)
    }

    /// Slots registered and not yet released.
    pub fn len(&self) -> usize {
        let tail = self.tail.load(Ordering::Acquire);
        let head = self.head.load(Ordering::Acquire);
        usize::try_from(tail.saturating_sub(head)).unwrap_or(usize::MAX)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The most unreleased slots the ring holds.
    pub fn capacity(&self) -> usize {
        self.cells.len()
    }

    fn finish(&self, seq: u64, value: Option<T>) {
        let _hot = HotScope::enter();
        let cell = &self.cells[(seq & self.mask) as usize];
        let state = if value.is_some() { READY } else { ABANDONED };
        // SAFETY: the slot's holder is the only writer, and the releaser reads the value only after it sees the state.
        unsafe { *cell.value.get() = value };
        cell.state.store(state, Ordering::Release);
        self.drain();
    }

    /// Java's `drain`: releases the ready prefix, unless another thread is releasing: that one then releases this
    /// slot too.
    fn drain(&self) {
        if self.wip.fetch_add(1, Ordering::AcqRel) != 0 {
            return;
        }
        // The count went from 0 to 1 on this thread, so no other thread is in this function until the count returns
        // to 0 below; the head belongs to this thread meanwhile.
        let mut panic = None;
        let mut missed = 1;
        loop {
            loop {
                let head = self.head.load(Ordering::Relaxed);
                if head == self.tail.load(Ordering::Acquire) {
                    break;
                }
                let cell = &self.cells[(head & self.mask) as usize];
                let state = cell.state.load(Ordering::Acquire);
                if state == PENDING {
                    break;
                }
                // SAFETY: the state was stored after the value was written, and nothing writes the cell again until
                // the head moves past it.
                let value = unsafe { (*cell.value.get()).take() };
                if let Some(value) = value
                    && let Err(payload) = catch_unwind(AssertUnwindSafe(|| (self.emit)(value)))
                {
                    panic.get_or_insert(payload);
                }
                // Freed after the emit, as Java's `finally`: the position stays in the order while it concludes.
                cell.state.store(PENDING, Ordering::Relaxed);
                self.head.store(head + 1, Ordering::Release);
            }
            missed = self.wip.fetch_sub(missed, Ordering::AcqRel) - missed;
            if missed == 0 {
                break;
            }
        }
        if let Some(payload) = panic {
            resume_unwind(payload);
        }
    }
}

/// The one handle that registers slots of a [`Sequencer`]. It is not `Clone`, and registering takes `&mut self`, so
/// registrations are ordered by the caller's own sequence.
pub struct Registrar<T: Send + 'static> {
    sequencer: Arc<Sequencer<T>>,
}

impl<T: Send + 'static> fmt::Debug for Registrar<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Registrar").finish_non_exhaustive()
    }
}

impl<T: Send + 'static> Registrar<T> {
    /// The next slot in admission order, or `None` while the ring holds [`capacity`](Sequencer::capacity) slots that are
    /// not released.
    pub fn register(&mut self) -> Option<Slot<T>> {
        let _hot = HotScope::enter();
        let sequencer = &*self.sequencer;
        let tail = sequencer.tail.load(Ordering::Relaxed);
        // Pairs with the releaser's release store of the head: the cell is free and its value taken.
        if tail - sequencer.head.load(Ordering::Acquire) == sequencer.cells.len() as u64 {
            return None;
        }
        // The cell's state is `PENDING` already (the releaser reset it); the tail publishes the registration.
        sequencer.tail.store(tail + 1, Ordering::Release);
        Some(Slot {
            sequencer: Some(self.sequencer.clone()),
            seq: tail,
        })
    }

    /// The sequencer this registrar takes slots of.
    pub fn sequencer(&self) -> &Arc<Sequencer<T>> {
        &self.sequencer
    }
}

/// One position in a [`Sequencer`]. Complete it or abandon it; dropping it abandons it.
///
/// A slot keeps its sequencer alive (an `Arc`): without a collector, a completion on another thread must not outlive
/// the ring it writes to.
pub struct Slot<T: Send + 'static> {
    sequencer: Option<Arc<Sequencer<T>>>,
    seq: u64,
}

impl<T: Send + 'static> fmt::Debug for Slot<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Slot")
            .field("seq", &self.seq)
            .finish_non_exhaustive()
    }
}

impl<T: Send + 'static> Slot<T> {
    /// The value for this position; released once every earlier slot finished.
    pub fn complete(mut self, value: T) {
        if let Some(sequencer) = self.sequencer.take() {
            sequencer.finish(self.seq, Some(value));
        }
    }

    /// This position has nothing to release (its frame was killed).
    pub fn abandon(mut self) {
        if let Some(sequencer) = self.sequencer.take() {
            sequencer.finish(self.seq, None);
        }
    }
}

impl<T: Send + 'static> Drop for Slot<T> {
    fn drop(&mut self) {
        if let Some(sequencer) = self.sequencer.take() {
            sequencer.finish(self.seq, None);
        }
    }
}

/// Ports of the reference tests of the in-order release: results emitted in input order after out-of-order
/// execution, a completion waiting for the lowest outstanding one, killed frames passed, a release that never
/// recurses, a failing emit that does not stop the release, and completions from many threads.
#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use std::sync::atomic::AtomicUsize;

    type Recording = (Arc<Sequencer<u64>>, Registrar<u64>, Arc<Mutex<Vec<u64>>>);

    fn recording() -> Recording {
        let out = Arc::new(Mutex::new(Vec::new()));
        let sink = out.clone();
        let (sequencer, registrar) =
            Sequencer::new(4096, move |value| sink.lock().unwrap().push(value));
        (sequencer, registrar, out)
    }

    fn take(registrar: &mut Registrar<u64>) -> Slot<u64> {
        registrar.register().expect("the ring has room")
    }

    /// FrameSequencerTest.emitsResultsInInputOrderAfterOutOfOrderExecution
    #[test]
    fn values_are_released_in_registration_order_after_out_of_order_completion() {
        let (sequencer, mut registrar, out) = recording();
        let first = take(&mut registrar);
        let second = take(&mut registrar);
        let third = take(&mut registrar);
        third.complete(3);
        second.complete(2);
        assert!(out.lock().unwrap().is_empty(), "the head is not ready");
        assert_eq!(sequencer.len(), 3);
        first.complete(1);
        assert_eq!(*out.lock().unwrap(), [1, 2, 3]);
        assert!(sequencer.is_empty());
    }

    /// OffsetCollectorTest.partitionCollectorWaitsForTheLowestOutstandingOffset: a later completion
    /// waits for the lowest outstanding one; when that one completes, everything up to the highest
    /// ready releases at once.
    #[test]
    fn a_completion_waits_for_the_lowest_outstanding_slot() {
        let (sequencer, mut registrar, out) = recording();
        let lower = take(&mut registrar);
        let higher = take(&mut registrar);
        higher.complete(11);
        assert!(out.lock().unwrap().is_empty());
        lower.complete(10);
        assert_eq!(*out.lock().unwrap(), [10, 11]);
        assert!(sequencer.is_empty());
    }

    /// OffsetCollectorTest.firstRegisteredFrameCommitsOnceAfterSuccessfulAcknowledgement
    #[test]
    fn the_first_slot_releases_once_when_it_completes() {
        let (sequencer, mut registrar, out) = recording();
        take(&mut registrar).complete(4);
        assert_eq!(*out.lock().unwrap(), [4]);
        assert!(sequencer.is_empty());
    }

    /// FrameSequencerTest.sourceErrorsTerminateOutputAndCancelPendingFrames, as a pipeline does it:
    /// killed frames abandon their slots (explicitly or by dropping them), and the in-order release
    /// passes them.
    #[test]
    fn abandoned_and_dropped_slots_are_passed() {
        let (sequencer, mut registrar, out) = recording();
        let a = take(&mut registrar);
        let b = take(&mut registrar);
        let c = take(&mut registrar);
        let d = take(&mut registrar);
        d.complete(4);
        b.abandon();
        a.complete(1);
        assert_eq!(*out.lock().unwrap(), [1]);
        drop(c);
        assert_eq!(*out.lock().unwrap(), [1, 4]);
        assert!(sequencer.is_empty());
    }

    /// SequencerTest.anEntryOfferedAndCompletedWhileDrainingIsDrainedByTheSameLoop: a completion
    /// during a release never recurses into the release; the running release takes it.
    #[test]
    fn a_slot_registered_and_completed_while_releasing_is_released_by_the_same_release() {
        let out = Arc::new(Mutex::new(Vec::new()));
        let pending: Arc<Mutex<Option<Slot<u64>>>> = Arc::new(Mutex::new(None));
        let (sink, next) = (out.clone(), pending.clone());
        let (sequencer, mut registrar) = Sequencer::new(8, move |value: u64| {
            sink.lock().unwrap().push(value);
            let slot = next.lock().unwrap().take();
            if let Some(slot) = slot {
                slot.complete(value + 1);
                assert_eq!(
                    sink.lock().unwrap().len(),
                    1,
                    "a release never recurses into itself"
                );
            }
        });
        let first = take(&mut registrar);
        *pending.lock().unwrap() = Some(take(&mut registrar));
        first.complete(0);
        assert_eq!(*out.lock().unwrap(), [0, 1]);
        assert!(sequencer.is_empty());
    }

    /// SequencerTest.aFailingEntryDoesNotStopTheDrain
    #[test]
    fn a_panicking_emit_does_not_stop_the_release_or_leave_it_claimed() {
        let out = Arc::new(Mutex::new(Vec::new()));
        let sink = out.clone();
        let (sequencer, mut registrar) = Sequencer::new(8, move |value: u64| {
            sink.lock().unwrap().push(value);
            assert_ne!(value, 0, "injected");
        });
        let first = take(&mut registrar);
        let second = take(&mut registrar);
        second.complete(1);
        let panicked = catch_unwind(AssertUnwindSafe(|| first.complete(0)));
        assert!(panicked.is_err(), "the panic resumes after the release");
        assert_eq!(*out.lock().unwrap(), [0, 1]);
        take(&mut registrar).complete(2);
        assert_eq!(
            *out.lock().unwrap(),
            [0, 1, 2],
            "the releaser is not left claimed"
        );
        assert!(sequencer.is_empty());
    }

    /// SequencerTest.entriesCompletingOnManyThreadsDrainOnceEachInOfferOrderOnOneThreadAtATime
    #[test]
    #[allow(clippy::disallowed_methods)] // The test's own threads complete the slots, as the Java test's pool does.
    fn slots_completed_on_many_threads_release_once_each_in_order_on_one_thread_at_a_time() {
        for round in 0..20u64 {
            let out = Arc::new(Mutex::new(Vec::new()));
            let releasing = Arc::new(AtomicUsize::new(0));
            let overlaps = Arc::new(AtomicUsize::new(0));
            let (sink, busy, overlap) = (out.clone(), releasing.clone(), overlaps.clone());
            let (sequencer, mut registrar) = Sequencer::new(4096, move |value: u64| {
                if busy.fetch_add(1, Ordering::AcqRel) != 0 {
                    overlap.fetch_add(1, Ordering::Relaxed);
                }
                sink.lock().unwrap().push(value);
                busy.fetch_sub(1, Ordering::AcqRel);
            });
            let count = 2_000u64;
            let mut slots: Vec<(u64, Slot<u64>)> =
                (0..count).map(|i| (i, take(&mut registrar))).collect();
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
                overlaps.load(Ordering::Relaxed),
                0,
                "two threads released at once"
            );
            assert!(sequencer.is_empty());
        }
    }

    /// An emit that panics does not stop the slots behind it, from a drain another thread left to this one.
    #[test]
    fn an_emit_that_panics_does_not_stop_the_ones_behind_it() {
        let out = Arc::new(Mutex::new(Vec::new()));
        let sink = out.clone();
        let (sequencer, mut registrar) = Sequencer::new(8, move |value: u64| {
            sink.lock().unwrap().push(value);
            assert_ne!(value % 2, 0, "injected");
        });
        let slots: Vec<_> = (0..6).map(|_| take(&mut registrar)).collect();
        let mut panics = 0;
        for (value, slot) in slots.into_iter().enumerate().rev() {
            if catch_unwind(AssertUnwindSafe(|| slot.complete(value as u64))).is_err() {
                panics += 1;
            }
        }
        assert_eq!(
            panics, 1,
            "the panic of the releasing completion resumes once"
        );
        assert_eq!(*out.lock().unwrap(), [0, 1, 2, 3, 4, 5]);
        assert!(sequencer.is_empty());
    }

    #[test]
    fn a_full_ring_refuses_registration_until_a_slot_is_released() {
        let (sequencer, mut registrar, out) = recording();
        let (small, mut small_registrar) = Sequencer::new(3, |_: u64| {});
        assert_eq!(small.capacity(), 4, "rounded up to a power of two");
        let slots: Vec<_> = (0..4)
            .map(|_| small_registrar.register().unwrap())
            .collect();
        assert!(small_registrar.register().is_none());
        let mut slots = slots.into_iter();
        slots.next().unwrap().complete(0);
        assert!(
            small_registrar.register().is_some(),
            "the head slot was released"
        );
        drop(slots);
        assert!(sequencer.is_empty() && out.lock().unwrap().is_empty());
        let _ = take(&mut registrar);
    }

    #[test]
    fn the_ring_wraps_many_times_in_order() {
        let out = Arc::new(Mutex::new(Vec::new()));
        let sink = out.clone();
        let (sequencer, mut registrar) =
            Sequencer::new(4, move |value: u64| sink.lock().unwrap().push(value));
        let mut next = 0u64;
        for _ in 0..1000 {
            let slots: Vec<_> = (0..3).map(|_| registrar.register().unwrap()).collect();
            for slot in slots.into_iter().rev() {
                slot.complete(next);
                next += 1;
            }
        }
        // Each round completed its slots newest first: three values per round, the round's last number first.
        let emitted = out.lock().unwrap().clone();
        assert_eq!(emitted.len(), 3000);
        assert!(emitted.chunks(3).all(|c| c[0] > c[1] && c[1] > c[2]));
        assert!(sequencer.is_empty());
    }
}
