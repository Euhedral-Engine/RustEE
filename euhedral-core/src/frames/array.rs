use super::{Frame, FrameError, FrameHeader, execute_all};
use crate::impls::Recycler;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

/// A frame that runs a fixed batch of frames inline, in order.
///
/// Each member that is alive is executed; members are not finalized, and the first member error
/// ends the batch and becomes the batch's error.
pub struct ArrayFrame {
    header: FrameHeader,
    frames: Box<[Box<dyn Frame>]>,
    recycler: Option<Recycler<ArrayFrame>>,
}

impl ArrayFrame {
    /// An ordered batch that dies with `kill_switch` and is dropped when it finishes.
    pub fn new(
        id_hash: u64,
        frames: impl Into<Box<[Box<dyn Frame>]>>,
        kill_switch: Option<Arc<AtomicBool>>,
    ) -> Self {
        Self::with_recycler(id_hash, frames, None, kill_switch)
    }

    /// A batch that returns to `recycler` when it finishes.
    pub fn with_recycler(
        id_hash: u64,
        frames: impl Into<Box<[Box<dyn Frame>]>>,
        recycler: Option<Recycler<ArrayFrame>>,
        kill_switch: Option<Arc<AtomicBool>>,
    ) -> Self {
        Self {
            header: FrameHeader::with_kill_switch(id_hash, kill_switch),
            frames: frames.into(),
            recycler,
        }
    }

    pub fn frames(&self) -> &[Box<dyn Frame>] {
        &self.frames
    }

    /// Swaps in a new batch and hands back the old one.
    pub fn replace(&mut self, frames: impl Into<Box<[Box<dyn Frame>]>>) -> Box<[Box<dyn Frame>]> {
        std::mem::replace(&mut self.frames, frames.into())
    }
}

impl Frame for ArrayFrame {
    fn header(&self) -> &FrameHeader {
        &self.header
    }

    fn header_mut(&mut self) -> &mut FrameHeader {
        &mut self.header
    }

    fn execute(&mut self) -> Result<(), FrameError> {
        execute_all(self.frames.iter_mut())
    }

    fn recycle(self: Box<Self>) {
        Recycler::recycle_owned(self, |frame| frame.recycler.as_ref());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frames::testing::{Recording, run};
    use crate::impls::FrameManager;
    use std::sync::Mutex;
    use std::sync::atomic::Ordering;

    fn log() -> Arc<Mutex<Vec<u64>>> {
        Arc::new(Mutex::new(Vec::new()))
    }

    #[test]
    fn executes_every_member_in_order() {
        let log = log();
        let members: Vec<Box<dyn Frame>> =
            (1..=3).map(|id| Recording::new(id, &log).boxed()).collect();
        let mut batch = ArrayFrame::new(9, members, None);
        assert_eq!(batch.frames().len(), 3);
        batch.execute().unwrap();
        assert_eq!(*log.lock().unwrap(), [1, 2, 3]);
    }

    #[test]
    fn skips_members_that_are_not_alive() {
        let log = log();
        let dead = Arc::new(AtomicBool::new(true));
        let mut skipped = Recording::new(2, &log);
        *skipped.header_mut() = FrameHeader::with_kill_switch(2, Some(dead));
        let mut batch = ArrayFrame::new(
            9,
            vec![
                Recording::new(1, &log).boxed(),
                skipped.boxed(),
                Recording::new(3, &log).boxed(),
            ],
            None,
        );
        batch.execute().unwrap();
        assert_eq!(*log.lock().unwrap(), [1, 3]);
    }

    #[test]
    fn the_first_member_error_ends_the_batch() {
        let log = log();
        let mut batch = ArrayFrame::new(
            9,
            vec![
                Recording::new(1, &log).boxed(),
                Recording::failing(2, &log, || FrameError::failed("member failed")).boxed(),
                Recording::new(3, &log).boxed(),
            ],
            None,
        );
        let error = batch.execute().unwrap_err();
        assert_eq!(error.to_string(), "frame failed: member failed");
        assert_eq!(*log.lock().unwrap(), [1, 2]);
    }

    #[test]
    fn a_member_cancellation_cancels_the_batch() {
        let log = log();
        let mut batch = ArrayFrame::new(
            9,
            vec![
                Recording::failing(1, &log, || FrameError::Cancelled).boxed(),
                Recording::new(2, &log).boxed(),
            ],
            None,
        );
        assert!(batch.execute().unwrap_err().is_cancelled());
        assert_eq!(*log.lock().unwrap(), [1]);
    }

    #[test]
    fn an_empty_batch_succeeds() {
        assert!(ArrayFrame::new(9, Vec::new(), None).execute().is_ok());
    }

    #[test]
    fn replace_swaps_the_batch() {
        let log = log();
        let mut batch = ArrayFrame::new(9, vec![Recording::new(1, &log).boxed()], None);
        let old = batch.replace(vec![
            Recording::new(2, &log).boxed(),
            Recording::new(3, &log).boxed(),
        ]);
        assert_eq!(old.len(), 1);
        assert_eq!(old[0].header().id_hash(), 1);
        batch.execute().unwrap();
        assert_eq!(*log.lock().unwrap(), [2, 3]);
    }

    #[test]
    fn a_pooled_batch_returns_when_finished_or_killed() {
        let log = log();
        let manager = FrameManager::<Vec<Box<dyn Frame>>, ArrayFrame>::new();
        let kill_switch = Arc::new(AtomicBool::new(false));
        let batch = ArrayFrame::with_recycler(
            9,
            vec![Recording::new(1, &log).boxed()],
            Some(manager.recycler()),
            Some(kill_switch.clone()),
        );
        run(Box::new(batch));
        assert_eq!(manager.recycle_queue_size(), 1);

        let killed = ArrayFrame::with_recycler(
            9,
            vec![Recording::new(2, &log).boxed()],
            Some(manager.recycler()),
            Some(kill_switch.clone()),
        );
        killed.kill();
        assert!(kill_switch.load(Ordering::Acquire));
        run(Box::new(killed));
        assert_eq!(manager.recycle_queue_size(), 2);
        assert_eq!(*log.lock().unwrap(), [1]);
    }

    #[test]
    fn a_failed_batch_is_still_returned() {
        let log = log();
        let manager = FrameManager::<Vec<Box<dyn Frame>>, ArrayFrame>::new();
        let batch = ArrayFrame::with_recycler(
            9,
            vec![Recording::failing(1, &log, || FrameError::failed("boom")).boxed()],
            Some(manager.recycler()),
            None,
        );
        run(Box::new(batch));
        assert_eq!(manager.recycle_queue_size(), 1);
    }
}
