use super::downstream::Downstream;
use crate::frames::{Frame, FrameError};
use crate::generics::{LatticeReceiver, LatticeSource};
#[allow(clippy::disallowed_types, reason = "HOT-FIX(C13)")]
use std::sync::{Arc, Mutex, PoisonError};

/// A source of one frame. It completes once the frame is taken; completed before that, it cancels
/// the frame.
pub struct SingleUseSource {
    #[allow(clippy::disallowed_types, reason = "HOT-FIX(C13)")]
    frame: Mutex<Option<Box<dyn Frame>>>,
    downstream: Downstream,
}

impl SingleUseSource {
    #[allow(clippy::disallowed_types, reason = "HOT-FIX(C13)")]
    pub fn wrap(frame: Box<dyn Frame>) -> Arc<Self> {
        Arc::new(Self {
            frame: Mutex::new(Some(frame)),
            downstream: Downstream::default(),
        })
    }

    /// The frame, unless it was taken or `stop` matches it.
    fn take(&self, stop: impl FnOnce(&dyn Frame) -> bool) -> Option<Box<dyn Frame>> {
        let mut slot = self.frame.lock().unwrap_or_else(PoisonError::into_inner);
        if slot.as_deref().is_some_and(stop) {
            return None;
        }
        slot.take()
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
        if demand == 0 {
            return 0;
        }
        let Some(frame) = self.take(|frame| stop(frame)) else {
            return 0;
        };
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
        if let Some(frame) = self.take(|_| false) {
            receiver.push(frame);
            self.complete();
        }
    }

    /// Completes the source, cancelling the frame if it was not taken.
    fn complete(&self) {
        if let Some(frame) = self.take(|_| false) {
            frame.do_finally_with_error(FrameError::Cancelled);
        }
        self.downstream.complete();
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
}
