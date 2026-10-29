//! A source of one frame.
//!
//! Java's `SingleUseSource` decides who takes the frame with `COMPLETE.compareAndSet(false, true)`;
//! the frame is read only by the winner, so it sits in a plain cell.

// The compare-and-set winner is the frame cell's only user.
#![allow(unsafe_code, reason = "the compare-and-set winner owns the frame")]

use crate::frames::{Frame, FrameError};
use crate::generics::{DownstreamSlot, LatticeReceiver, LatticeSource};
use std::cell::UnsafeCell;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering, fence};

/// A source of one frame. It completes once the frame is taken; completed before that, it cancels
/// the frame.
///
/// `taken` is Java's `COMPLETE`: `pull`, `request` and `complete` each try to flip it, and the one
/// that does owns the frame. A `pull` whose stop condition matches the frame gives it back and
/// leaves the source untaken.
pub struct SingleUseSource {
    taken: AtomicBool,
    /// Read and written only by the holder of `taken`, or through `&mut self`.
    frame: UnsafeCell<Option<Box<dyn Frame>>>,
    downstream: DownstreamSlot,
}

// SAFETY: the frame is reached only by the thread that flipped `taken` from false to true (it
// stays true until that thread hands the frame back or finishes with it), or through `&mut self`.
// The frame is `Send`, and the cell never shares a reference to it between threads.
unsafe impl Sync for SingleUseSource {}

impl SingleUseSource {
    pub fn wrap(frame: Box<dyn Frame>) -> Arc<Self> {
        Arc::new(Self {
            taken: AtomicBool::new(false),
            frame: UnsafeCell::new(Some(frame)),
            downstream: DownstreamSlot::new(),
        })
    }

    /// Flips `taken`; the caller that does so owns the frame.
    fn try_take(&self) -> bool {
        self.taken
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
    }

    /// The frame, for the holder of `taken`.
    #[allow(
        clippy::mut_from_ref,
        reason = "the `taken` flag, not the borrow, makes the access exclusive"
    )]
    fn frame_mut(&self) -> &mut Option<Box<dyn Frame>> {
        // SAFETY: the caller holds `taken`, so no other thread reads or writes the cell.
        unsafe { &mut *self.frame.get() }
    }

    /// Cancels the frame if nobody has taken it.
    fn cancel_untaken(&self) {
        if self.try_take()
            && let Some(frame) = self.frame_mut().take()
        {
            frame.do_finally_with_error(FrameError::Cancelled);
        }
    }
}

impl Drop for SingleUseSource {
    /// A frame nobody took is cancelled.
    fn drop(&mut self) {
        if let Some(frame) = self.frame.get_mut().take() {
            frame.do_finally_with_error(FrameError::Cancelled);
        }
    }
}

impl LatticeSource for SingleUseSource {
    fn add_downstream(self: Arc<Self>, downstream: Arc<dyn LatticeReceiver>) {
        self.downstream.attach(self.clone(), downstream);
    }

    /// Hands the frame to `consumer` unless `stop` matches it, then completes.
    fn pull(
        &self,
        consumer: &mut dyn FnMut(Box<dyn Frame>),
        stop: &mut dyn FnMut(&dyn Frame) -> bool,
        demand: u64,
    ) -> u64 {
        if demand == 0 || !self.try_take() {
            return 0;
        }
        let Some(frame) = self.frame_mut().take() else {
            return 0;
        };
        if stop(&*frame) {
            *self.frame_mut() = Some(frame);
            self.taken.store(false, Ordering::SeqCst);
            // A `complete` that found the frame held has left it to us: it set the slot's
            // completion first, and the fence orders our store before this read.
            fence(Ordering::SeqCst);
            if self.downstream.is_complete() {
                self.cancel_untaken();
            }
            return 0;
        }
        consumer(frame);
        self.complete();
        1
    }

    /// Pushes the frame to the attached receiver, then completes.
    fn request(&self, demand: u64) {
        if demand == 0 {
            return;
        }
        let Some(receiver) = self.downstream.receiver() else {
            return;
        };
        if self.try_take()
            && let Some(frame) = self.frame_mut().take()
        {
            receiver.push(frame);
            self.complete();
        }
    }

    /// Completes the source, cancelling the frame if it was not taken.
    fn complete(&self) {
        self.downstream.complete();
        self.cancel_untaken();
    }

    fn is_complete(&self) -> bool {
        self.downstream.is_complete()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flow_control::testing::{Fate, Fates, TestReceiver, Tracked, frame};

    #[test]
    fn a_request_pushes_the_frame_once_and_completes() {
        let source = SingleUseSource::wrap(frame(7));
        let receiver = TestReceiver::new();
        source.request(1);
        source.clone().add_downstream(receiver.clone());
        assert!(receiver.upstream.lock().unwrap().is_some());
        source.request(0);
        assert!(receiver.received().is_empty());
        source.request(1);
        source.request(1);
        assert_eq!(receiver.received(), [7]);
        assert!(source.is_complete());
        assert!(receiver.is_completed());
    }

    #[test]
    fn a_pull_takes_the_frame_unless_stopped() {
        let source = SingleUseSource::wrap(frame(7));
        let mut pulled = Vec::new();
        assert_eq!(source.pull(&mut |f| pulled.push(f), &mut |_| false, 0), 0);
        assert_eq!(source.pull(&mut |f| pulled.push(f), &mut |_| true, 1), 0);
        assert!(!source.is_complete());
        assert_eq!(source.pull(&mut |f| pulled.push(f), &mut |_| false, 1), 1);
        assert_eq!(source.pull(&mut |f| pulled.push(f), &mut |_| false, 1), 0);
        assert_eq!(pulled.len(), 1);
        assert!(source.is_complete());
    }

    #[test]
    fn completing_first_cancels_the_frame_and_completes_late_receivers() {
        let fates = Fates::default();
        let source = SingleUseSource::wrap(Tracked::new(3, &fates));
        source.complete();
        assert!(source.is_complete());
        assert_eq!(
            *fates.lock().unwrap(),
            [(3, Fate::Failed(FrameError::Cancelled.to_string()))]
        );
        let receiver = TestReceiver::new();
        source.clone().add_downstream(receiver.clone());
        assert!(receiver.is_completed());
        source.request(1);
        assert!(receiver.received().is_empty());
    }

    #[test]
    fn one_downstream_only() {
        let source = SingleUseSource::wrap(frame(1));
        source.clone().add_downstream(TestReceiver::new());
        let second = TestReceiver::new();
        source.clone().add_downstream(second.clone());
        assert_eq!(second.errors().len(), 1);
    }

    #[test]
    fn a_frame_is_delivered_or_cancelled_exactly_once() {
        use std::sync::Barrier;
        const ROUNDS: u64 = if cfg!(miri) { 20 } else { 10_000 };
        let fates = Fates::default();
        let barrier = Barrier::new(2);
        let sources: Vec<_> = (0..ROUNDS)
            .map(|id| {
                let source = SingleUseSource::wrap(Tracked::new(id, &fates));
                let receiver = TestReceiver::new();
                source.clone().add_downstream(receiver.clone());
                (source, receiver)
            })
            .collect();
        std::thread::scope(|scope| {
            scope.spawn(|| {
                barrier.wait();
                for (source, _) in &sources {
                    source.complete();
                }
            });
            barrier.wait();
            for (source, _) in &sources {
                source.request(1);
            }
        });
        let mut cancelled = 0;
        for (id, (source, receiver)) in sources.iter().enumerate() {
            assert!(source.is_complete());
            assert!(receiver.is_completed());
            let delivered = receiver.received().len();
            let fated = fates
                .lock()
                .unwrap()
                .iter()
                .filter(|(fated, _)| *fated == id as u64)
                .count();
            // Delivered frames are the receiver's to finish: `TestReceiver` drops them unfinished.
            assert_eq!(delivered + fated, 1, "round {id}");
            cancelled += fated;
        }
        assert!(cancelled <= ROUNDS as usize);
    }

    #[test]
    fn a_completed_source_delivers_nothing_to_a_later_pull() {
        let source = SingleUseSource::wrap(frame(7));
        source.complete();
        let mut pulled = Vec::new();
        assert_eq!(source.pull(&mut |f| pulled.push(f), &mut |_| false, 1), 0);
        assert!(pulled.is_empty());
        let receiver = TestReceiver::new();
        source.clone().add_downstream(receiver.clone());
        source.request(1);
        assert!(receiver.received().is_empty());
    }

    #[test]
    fn a_pull_that_stops_gives_the_frame_back_and_a_racing_complete_still_cancels_it() {
        use std::sync::Barrier;
        const ROUNDS: u64 = if cfg!(miri) { 10 } else { 5_000 };
        for id in 0..ROUNDS {
            let fates = Fates::default();
            let source = SingleUseSource::wrap(Tracked::new(id, &fates));
            let barrier = Barrier::new(2);
            std::thread::scope(|scope| {
                scope.spawn(|| {
                    barrier.wait();
                    source.complete();
                });
                barrier.wait();
                // The stop condition always matches, so the pull never delivers.
                assert_eq!(source.pull(&mut |_| unreachable!(), &mut |_| true, 1), 0);
            });
            assert_eq!(
                *fates.lock().unwrap(),
                [(id, Fate::Failed(FrameError::Cancelled.to_string()))],
                "round {id}: cancelled exactly once"
            );
        }
    }

    #[test]
    fn a_source_dropped_with_its_frame_cancels_it() {
        let fates = Fates::default();
        drop(SingleUseSource::wrap(Tracked::new(5, &fates)));
        assert_eq!(
            *fates.lock().unwrap(),
            [(5, Fate::Failed(FrameError::Cancelled.to_string()))]
        );
    }
}
