//! Execute in any order, conclude in admission order (Euhedral-Execution `FrameSequencer` /
//! `OffsetCollector`).

use std::collections::VecDeque;
use std::fmt;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

/// Releases values in the order their slots were registered, whatever order they complete in.
///
/// - [`register`](Self::register) takes a slot, in admission order. Registrations must be
///   serialized by the caller (one admitting source at a time, as a `LatticeSource` is polled).
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
    order: Mutex<Order<T>>,
    /// The work-in-progress count: the thread that raises it from 0 releases; the others leave
    /// their slot to it.
    wip: AtomicUsize,
    emit: Box<dyn Fn(T) + Send + Sync>,
}

/// The slots not yet released, the oldest first.
struct Order<T> {
    /// The sequence number of `slots[0]`.
    head: u64,
    slots: VecDeque<State<T>>,
}

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
        Arc::new(Self {
            order: Mutex::new(Order {
                head: 0,
                slots: VecDeque::new(),
            }),
            wip: AtomicUsize::new(0),
            emit: Box::new(emit),
        })
    }

    /// The next slot in admission order.
    pub fn register(self: &Arc<Self>) -> Slot<T> {
        let mut order = self.lock();
        let seq = order.head + order.slots.len() as u64;
        order.slots.push_back(State::Pending);
        Slot {
            sequencer: Some(self.clone()),
            seq,
        }
    }

    /// Slots registered and not yet released.
    pub fn len(&self) -> usize {
        self.lock().slots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn lock(&self) -> MutexGuard<'_, Order<T>> {
        self.order.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn finish(&self, seq: u64, state: State<T>) {
        {
            let mut order = self.lock();
            let index = usize::try_from(seq - order.head).expect("a slot within the order");
            debug_assert!(
                matches!(order.slots[index], State::Pending),
                "a slot finishes once"
            );
            order.slots[index] = state;
        }
        self.release();
    }

    /// Releases the finished prefix, unless another thread is releasing: that one then releases
    /// this slot too.
    fn release(&self) {
        if self.wip.fetch_add(1, Ordering::AcqRel) != 0 {
            return;
        }
        let mut panic = None;
        let mut missed = 1;
        loop {
            loop {
                let next = {
                    let mut order = self.lock();
                    match order.slots.front() {
                        Some(State::Ready(_) | State::Abandoned) => {
                            order.head += 1;
                            order.slots.pop_front()
                        }
                        _ => None,
                    }
                };
                match next {
                    Some(State::Ready(value)) => {
                        if let Err(payload) = catch_unwind(AssertUnwindSafe(|| (self.emit)(value)))
                        {
                            panic.get_or_insert(payload);
                        }
                    }
                    Some(_) => {}
                    None => break,
                }
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
            sequencer.finish(self.seq, State::Ready(value));
        }
    }

    /// This position has nothing to release (its frame was killed).
    pub fn abandon(mut self) {
        if let Some(sequencer) = self.sequencer.take() {
            sequencer.finish(self.seq, State::Abandoned);
        }
    }
}

impl<T: Send + 'static> Drop for Slot<T> {
    fn drop(&mut self) {
        if let Some(sequencer) = self.sequencer.take() {
            sequencer.finish(self.seq, State::Abandoned);
        }
    }
}

/// The ports of Euhedral-Execution's `FrameSequencerTest` and `OffsetCollectorTest`, and of
/// Euhedral-Inference's `SequencerTest` (the same pattern). The Kafka-specific parts (commit
/// policy, commit retries, passwords) have no counterpart here.
#[cfg(test)]
mod tests {
    use super::*;

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
}
