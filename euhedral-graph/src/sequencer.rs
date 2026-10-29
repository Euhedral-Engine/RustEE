//! Execute in any order, conclude in admission order (Euhedral-Execution `FrameSequencer` /
//! `OffsetCollector`; Euhedral-Inference `Sequencer`).
//!
//! Java's design, without a lock: a multi-producer queue holds the entries in the order they were offered, each entry
//! has a `ready` flag, and a work-in-progress count makes the thread that raises it from zero conclude the ready
//! prefix. A completion that arrives while a drain runs raises the count, and the drain looks again before it
//! returns.

#![allow(
    unsafe_code,
    reason = "the thread that wins the work-in-progress count owns the order's consumer; an entry's value is written before its ready flag and read after it"
)]

use euhedral_data_structures::atomics::CachePadded;
use euhedral_data_structures::queues::{Consumer, MpscQueue, Multi, Producer, Single};
use euhedral_data_structures::sync::HotScope;
use std::cell::UnsafeCell;
use std::fmt;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

/// Releases values in the order their slots were registered, whatever order they complete in.
///
/// - [`register`](Self::register) takes a slot, in the order the registrations reach the order queue. Any thread may
///   register; the order is the order of the offers, so the owner of the order registers from one chain of work.
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
    /// The entries in offer order (Java's `order`).
    order: Producer<Arc<Entry<T>>, Multi>,
    /// Peeked and polled by the thread that holds `wip`, and by no other.
    consumer: UnsafeCell<Consumer<Arc<Entry<T>>, Single>>,
    /// The work-in-progress count: the thread that raises it from 0 releases; the others leave
    /// their slot to it.
    wip: CachePadded<AtomicU32>,
    /// The order queue, for its length: slots registered and not yet released, with no counter of its own to update.
    queue: MpscQueue<Arc<Entry<T>>>,
    emit: Box<dyn Fn(T) + Send + Sync>,
}

// SAFETY: the consumer is used only under the work-in-progress count, which admits one thread at a time; the
// emitter is `Send + Sync`; the entries are `Send` when `T` is.
unsafe impl<T: Send> Send for Sequencer<T> {}
// SAFETY: as above.
unsafe impl<T: Send> Sync for Sequencer<T> {}

/// One position of the order.
struct Entry<T> {
    /// Java's `ready()`: the finishing thread stores it (release) after writing `state`; the releaser loads it (acquire).
    ready: AtomicBool,
    /// Written once by the slot's holder before `ready` is stored, taken by the releaser after it loads `ready`.
    state: UnsafeCell<State<T>>,
}

// SAFETY: `state` is written before the ready flag (sequentially consistent) and read after it, by one thread each.
unsafe impl<T: Send> Send for Entry<T> {}
// SAFETY: as above.
unsafe impl<T: Send> Sync for Entry<T> {}

enum State<T> {
    Pending,
    Ready(T),
    Abandoned,
}

impl<T> fmt::Debug for Sequencer<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Sequencer").finish_non_exhaustive()
    }
}

impl<T: Send + 'static> Sequencer<T> {
    pub fn new(emit: impl Fn(T) + Send + Sync + 'static) -> Arc<Self> {
        let queue = MpscQueue::new(16);
        Arc::new(Self {
            order: queue.producer(),
            consumer: UnsafeCell::new(queue.consumer()),
            wip: CachePadded::new(AtomicU32::new(0)),
            queue,
            emit: Box::new(emit),
        })
    }

    /// The next slot in admission order.
    pub fn register(self: &Arc<Self>) -> Slot<T> {
        let _hot = HotScope::enter();
        let entry = Arc::new(Entry {
            ready: AtomicBool::new(false),
            state: UnsafeCell::new(State::Pending),
        });
        if self.order.offer(entry.clone()).is_err() {
            unreachable!("the order queue is unbounded");
        }
        Slot {
            sequencer: Some(self.clone()),
            entry,
        }
    }

    /// Slots registered and not yet released.
    pub fn len(&self) -> usize {
        usize::try_from(self.queue.size_long()).unwrap_or(usize::MAX)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn finish(&self, entry: &Entry<T>, state: State<T>) {
        let _hot = HotScope::enter();
        // SAFETY: the slot's holder is the only writer, and the releaser reads the state only after it sees `ready`.
        unsafe { entry.state.get().write(state) };
        entry.ready.store(true, Ordering::Release);
        self.drain();
    }

    /// Java's `drain`: releases the ready prefix, unless another thread is releasing: that one then releases this
    /// slot too.
    fn drain(&self) {
        if self.wip.fetch_add(1, Ordering::AcqRel) != 0 {
            return;
        }
        // SAFETY: the count went from 0 to 1 on this thread, so no other thread is in this function until the
        // count returns to 0 below; the consumer belongs to this thread meanwhile.
        let consumer = unsafe { &mut *self.consumer.get() };
        let mut panic = None;
        let mut missed = 1;
        loop {
            loop {
                let state = match consumer.peek() {
                    Some(head) if head.ready.load(Ordering::Acquire) => {
                        // SAFETY: `ready` was stored after the state was written, and nothing writes it again.
                        unsafe { std::mem::replace(&mut *head.state.get(), State::Abandoned) }
                    }
                    _ => break,
                };
                if let State::Ready(value) = state
                    && let Err(payload) = catch_unwind(AssertUnwindSafe(|| (self.emit)(value)))
                {
                    panic.get_or_insert(payload);
                }
                // Polled after the emit, as Java's `finally`: the entry stays in the order while it concludes.
                drop(consumer.poll());
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

/// One position in a [`Sequencer`]. Complete it or abandon it; dropping it abandons it.
pub struct Slot<T: Send + 'static> {
    sequencer: Option<Arc<Sequencer<T>>>,
    entry: Arc<Entry<T>>,
}

impl<T: Send + 'static> fmt::Debug for Slot<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Slot").finish_non_exhaustive()
    }
}

impl<T: Send + 'static> Slot<T> {
    /// The value for this position; released once every earlier slot finished.
    pub fn complete(mut self, value: T) {
        if let Some(sequencer) = self.sequencer.take() {
            sequencer.finish(&self.entry, State::Ready(value));
        }
    }

    /// This position has nothing to release (its frame was killed).
    pub fn abandon(mut self) {
        if let Some(sequencer) = self.sequencer.take() {
            sequencer.finish(&self.entry, State::Abandoned);
        }
    }
}

impl<T: Send + 'static> Drop for Slot<T> {
    fn drop(&mut self) {
        if let Some(sequencer) = self.sequencer.take() {
            sequencer.finish(&self.entry, State::Abandoned);
        }
    }
}

/// The ports of Euhedral-Execution's `FrameSequencerTest` and `OffsetCollectorTest`, and of
/// Euhedral-Inference's `SequencerTest` (the same pattern). The Kafka-specific parts (commit
/// policy, commit retries, passwords) have no counterpart here.
#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use std::sync::atomic::AtomicUsize;

    fn recording() -> (Arc<Sequencer<u64>>, Arc<Mutex<Vec<u64>>>) {
        let out = Arc::new(Mutex::new(Vec::new()));
        let sink = out.clone();
        (
            Sequencer::new(move |value| sink.lock().unwrap().push(value)),
            out,
        )
    }

    /// FrameSequencerTest.emitsResultsInInputOrderAfterOutOfOrderExecution
    #[test]
    fn values_are_released_in_registration_order_after_out_of_order_completion() {
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

    /// OffsetCollectorTest.partitionCollectorWaitsForTheLowestOutstandingOffset: a later completion
    /// waits for the lowest outstanding one; when that one completes, everything up to the highest
    /// ready releases at once.
    #[test]
    fn a_completion_waits_for_the_lowest_outstanding_slot() {
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
    fn the_first_slot_releases_once_when_it_completes() {
        let (sequencer, out) = recording();
        sequencer.register().complete(4);
        assert_eq!(*out.lock().unwrap(), [4]);
        assert!(sequencer.is_empty());
    }

    /// FrameSequencerTest.sourceErrorsTerminateOutputAndCancelPendingFrames, as a pipeline does it:
    /// killed frames abandon their slots (explicitly or by dropping them), and the in-order release
    /// passes them.
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

    /// SequencerTest.anEntryOfferedAndCompletedWhileDrainingIsDrainedByTheSameLoop: a completion
    /// during a release never recurses into the release; the running release takes it.
    #[test]
    fn a_slot_registered_and_completed_while_releasing_is_released_by_the_same_release() {
        let out = Arc::new(Mutex::new(Vec::new()));
        let pending: Arc<Mutex<Option<Slot<u64>>>> = Arc::new(Mutex::new(None));
        let (sink, next) = (out.clone(), pending.clone());
        let sequencer = Sequencer::new(move |value: u64| {
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
        let first = sequencer.register();
        *pending.lock().unwrap() = Some(sequencer.register());
        first.complete(0);
        assert_eq!(*out.lock().unwrap(), [0, 1]);
        assert!(sequencer.is_empty());
    }

    /// SequencerTest.aFailingEntryDoesNotStopTheDrain
    #[test]
    fn a_panicking_emit_does_not_stop_the_release_or_leave_it_claimed() {
        let out = Arc::new(Mutex::new(Vec::new()));
        let sink = out.clone();
        let sequencer = Sequencer::new(move |value: u64| {
            sink.lock().unwrap().push(value);
            assert_ne!(value, 0, "injected");
        });
        let first = sequencer.register();
        let second = sequencer.register();
        second.complete(1);
        let panicked = catch_unwind(AssertUnwindSafe(|| first.complete(0)));
        assert!(panicked.is_err(), "the panic resumes after the release");
        assert_eq!(*out.lock().unwrap(), [0, 1]);
        sequencer.register().complete(2);
        assert_eq!(
            *out.lock().unwrap(),
            [0, 1, 2],
            "the releaser is not left claimed"
        );
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
            let sequencer = Sequencer::new(move |value: u64| {
                if busy.fetch_add(1, Ordering::AcqRel) != 0 {
                    overlap.fetch_add(1, Ordering::Relaxed);
                }
                sink.lock().unwrap().push(value);
                busy.fetch_sub(1, Ordering::AcqRel);
            });
            let count = 2_000u64;
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
                overlaps.load(Ordering::Relaxed),
                0,
                "two threads released at once"
            );
            assert!(sequencer.is_empty());
        }
    }

    /// Registrations from 4 threads, each in order, completed in reverse from 4 threads: whatever the interleaving of
    /// the registrations, each thread's values are emitted in the order that thread registered them, each once, and
    /// one thread at a time.
    #[test]
    fn registrations_from_4_threads_complete_in_reverse_and_emit_in_order() {
        #[cfg(not(miri))]
        const PER_THREAD: u64 = 250;
        #[cfg(miri)]
        const PER_THREAD: u64 = 6;
        #[cfg(not(miri))]
        const ROUNDS: u64 = 20;
        #[cfg(miri)]
        const ROUNDS: u64 = 1;
        for _ in 0..ROUNDS {
            let out = Arc::new(Mutex::new(Vec::new()));
            let releasing = Arc::new(AtomicUsize::new(0));
            let overlaps = Arc::new(AtomicUsize::new(0));
            let (sink, busy, overlap) = (out.clone(), releasing.clone(), overlaps.clone());
            let sequencer = Sequencer::new(move |value: u64| {
                if busy.fetch_add(1, Ordering::SeqCst) != 0 {
                    overlap.fetch_add(1, Ordering::SeqCst);
                }
                sink.lock().unwrap().push(value);
                busy.fetch_sub(1, Ordering::SeqCst);
            });
            let barrier = Arc::new(std::sync::Barrier::new(4));
            let registrants: Vec<_> = (0..4u64)
                .map(|thread| {
                    let (sequencer, barrier) = (sequencer.clone(), barrier.clone());
                    std::thread::spawn(move || {
                        barrier.wait();
                        (0..PER_THREAD)
                            .map(|index| (thread * PER_THREAD + index, sequencer.register()))
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
            let mut batches: Vec<Vec<(u64, Slot<u64>)>> = registrants
                .into_iter()
                .map(|registrant| registrant.join().unwrap())
                .collect();
            // Complete each thread's slots newest first, from its own thread.
            let completers: Vec<_> = batches
                .drain(..)
                .map(|batch| {
                    std::thread::spawn(move || {
                        for (value, slot) in batch.into_iter().rev() {
                            slot.complete(value);
                        }
                    })
                })
                .collect();
            for completer in completers {
                completer.join().unwrap();
            }
            let emitted = out.lock().unwrap().clone();
            assert_eq!(emitted.len() as u64, 4 * PER_THREAD, "each once");
            for thread in 0..4u64 {
                let own: Vec<u64> = emitted
                    .iter()
                    .copied()
                    .filter(|value| value / PER_THREAD == thread)
                    .collect();
                assert_eq!(
                    own,
                    (thread * PER_THREAD..(thread + 1) * PER_THREAD).collect::<Vec<_>>(),
                    "thread {thread}'s registration order"
                );
            }
            assert_eq!(overlaps.load(Ordering::SeqCst), 0, "two releasers at once");
            assert!(sequencer.is_empty());
        }
    }

    /// An emit that panics does not stop the slots behind it, from a drain another thread left to this one.
    #[test]
    fn an_emit_that_panics_does_not_stop_the_ones_behind_it() {
        let out = Arc::new(Mutex::new(Vec::new()));
        let sink = out.clone();
        let sequencer = Sequencer::new(move |value: u64| {
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
            "the panic of the releasing completion resumes once"
        );
        assert_eq!(*out.lock().unwrap(), [0, 1, 2, 3, 4, 5]);
        assert!(sequencer.is_empty());
    }
}
