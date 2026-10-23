use super::{Frame, FrameError, FrameHeader};
use crate::impls::Recycler;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

/// The task a [`RunnableFrame`] runs, shareable between the frames of one pool.
pub type Task = Arc<dyn Fn() + Send + Sync>;

/// A frame that runs a task.
pub struct RunnableFrame {
    header: FrameHeader,
    task: Task,
    recycler: Option<Recycler<RunnableFrame>>,
}

impl RunnableFrame {
    /// An ordered frame that runs `task` and is dropped when it finishes.
    pub fn new(id_hash: u64, task: Task) -> Self {
        Self::with_recycler(id_hash, task, None, None)
    }

    /// A frame that returns to `recycler` when it finishes and dies with `kill_switch`.
    pub fn with_recycler(
        id_hash: u64,
        task: Task,
        recycler: Option<Recycler<RunnableFrame>>,
        kill_switch: Option<Arc<AtomicBool>>,
    ) -> Self {
        Self {
            header: FrameHeader::with_kill_switch(id_hash, kill_switch),
            task,
            recycler,
        }
    }
}

impl Frame for RunnableFrame {
    fn header(&self) -> &FrameHeader {
        &self.header
    }

    fn header_mut(&mut self) -> &mut FrameHeader {
        &mut self.header
    }

    fn execute(&mut self) -> Result<(), FrameError> {
        (self.task)();
        Ok(())
    }

    fn recycle(self: Box<Self>) {
        Recycler::recycle_owned(self, |frame| frame.recycler.as_ref());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frames::testing::run;
    use crate::impls::{FrameFactory, FrameManager};
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn counter() -> (Arc<AtomicUsize>, Task) {
        let count = Arc::new(AtomicUsize::new(0));
        let task_count = count.clone();
        let task: Task = Arc::new(move || {
            task_count.fetch_add(1, Ordering::Relaxed);
        });
        (count, task)
    }

    #[test]
    fn executing_runs_the_task_every_time() {
        let (count, task) = counter();
        let mut frame = RunnableFrame::new(1, task);
        assert!(frame.header().is_ordered());
        frame.execute().unwrap();
        frame.execute().unwrap();
        assert_eq!(count.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn a_pooled_frame_returns_after_running() {
        let (count, task) = counter();
        let mut manager = FrameManager::<(), RunnableFrame>::new();
        let recycler = manager.recycler();
        manager
            .set_factory(FrameFactory::new(
                move |id, ()| {
                    RunnableFrame::with_recycler(id, task.clone(), Some(recycler.clone()), None)
                },
                |(), _| {},
            ))
            .unwrap();

        let frame = manager.get_or_create(());
        let first = &*frame as *const RunnableFrame;
        run(frame);
        assert_eq!(manager.recycle_queue_size(), 1);
        let again = manager.get_or_create(());
        assert_eq!(&*again as *const RunnableFrame, first);
        run(again);
        assert_eq!(count.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn a_killed_frame_is_skipped_but_still_returned() {
        let (count, task) = counter();
        let manager = FrameManager::<(), RunnableFrame>::new();
        let kill_switch = Arc::new(AtomicBool::new(false));
        let frame = RunnableFrame::with_recycler(
            1,
            task,
            Some(manager.recycler()),
            Some(kill_switch.clone()),
        );
        frame.kill();
        assert!(kill_switch.load(Ordering::Acquire));

        run(Box::new(frame));
        assert_eq!(count.load(Ordering::Relaxed), 0);
        assert_eq!(manager.recycle_queue_size(), 1);
    }
}
