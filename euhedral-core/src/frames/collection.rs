use super::{Frame, FrameError, FrameHeader, execute_all};
use crate::impls::Recycler;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

/// A frame that runs the frames of any collection inline, in iteration order.
///
/// Behaves like [`ArrayFrame`](super::ArrayFrame) for collections that can grow or shrink between
/// runs, such as a `Vec` or `VecDeque` the owner refills.
pub struct CollectionFrame<C> {
    header: FrameHeader,
    frames: C,
    recycler: Option<Recycler<CollectionFrame<C>>>,
}

impl<C> CollectionFrame<C>
where
    for<'a> &'a mut C: IntoIterator<Item = &'a mut Box<dyn Frame>>,
{
    /// An ordered batch that dies with `kill_switch` and is dropped when it finishes.
    pub fn new(id_hash: u64, frames: C, kill_switch: Option<Arc<AtomicBool>>) -> Self {
        Self::with_recycler(id_hash, frames, None, kill_switch)
    }

    /// A batch that returns to `recycler` when it finishes.
    pub fn with_recycler(
        id_hash: u64,
        frames: C,
        recycler: Option<Recycler<CollectionFrame<C>>>,
        kill_switch: Option<Arc<AtomicBool>>,
    ) -> Self {
        Self {
            header: FrameHeader::with_kill_switch(id_hash, kill_switch),
            frames,
            recycler,
        }
    }

    pub fn frames(&self) -> &C {
        &self.frames
    }

    pub fn frames_mut(&mut self) -> &mut C {
        &mut self.frames
    }

    /// Swaps in a new collection and hands back the old one.
    pub fn replace(&mut self, frames: C) -> C {
        std::mem::replace(&mut self.frames, frames)
    }
}

impl<C> Frame for CollectionFrame<C>
where
    C: Send + 'static,
    for<'a> &'a mut C: IntoIterator<Item = &'a mut Box<dyn Frame>>,
{
    fn header(&self) -> &FrameHeader {
        &self.header
    }

    fn header_mut(&mut self) -> &mut FrameHeader {
        &mut self.header
    }

    fn execute(&mut self) -> Result<(), FrameError> {
        execute_all(&mut self.frames)
    }

    fn recycle(self: Box<Self>) {
        Recycler::recycle_owned(self, |frame| frame.recycler.as_ref());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frames::testing::{Recording, run};
    use crate::impls::{FrameFactory, FrameManager};
    use std::collections::{LinkedList, VecDeque};
    use std::sync::Mutex;

    type Batch = Vec<Box<dyn Frame>>;

    fn log() -> Arc<Mutex<Vec<u64>>> {
        Arc::new(Mutex::new(Vec::new()))
    }

    #[test]
    fn runs_a_vec_deque_and_a_linked_list_in_order() {
        let log = log();
        let deque: VecDeque<Box<dyn Frame>> =
            (1..=2).map(|id| Recording::new(id, &log).boxed()).collect();
        CollectionFrame::new(9, deque, None).execute().unwrap();
        let list: LinkedList<Box<dyn Frame>> =
            (3..=4).map(|id| Recording::new(id, &log).boxed()).collect();
        CollectionFrame::new(9, list, None).execute().unwrap();
        assert_eq!(*log.lock().unwrap(), [1, 2, 3, 4]);
    }

    #[test]
    fn skips_dead_members_and_stops_at_the_first_error() {
        let log = log();
        let mut dead = Recording::new(2, &log);
        *dead.header_mut() =
            FrameHeader::with_kill_switch(2, Some(Arc::new(AtomicBool::new(true))));
        let batch: Batch = vec![
            Recording::new(1, &log).boxed(),
            dead.boxed(),
            Recording::failing(3, &log, || FrameError::failed("member failed")).boxed(),
            Recording::new(4, &log).boxed(),
        ];
        let mut frame = CollectionFrame::new(9, batch, None);
        assert!(!frame.execute().unwrap_err().is_cancelled());
        assert_eq!(*log.lock().unwrap(), [1, 3]);
    }

    #[test]
    fn the_collection_can_be_refilled_in_place_or_replaced() {
        let log = log();
        let mut frame = CollectionFrame::new(9, Batch::new(), None);
        frame.execute().unwrap();
        frame.frames_mut().push(Recording::new(1, &log).boxed());
        frame.execute().unwrap();
        let old = frame.replace(vec![Recording::new(2, &log).boxed()]);
        assert_eq!(old.len(), 1);
        assert_eq!(frame.frames().len(), 1);
        frame.execute().unwrap();
        assert_eq!(*log.lock().unwrap(), [1, 2]);
    }

    #[test]
    fn a_pooled_batch_returns_to_a_pool_of_its_own_type() {
        let log = log();
        let mut manager = FrameManager::<Batch, CollectionFrame<Batch>>::new();
        let recycler = manager.recycler();
        manager
            .set_factory(FrameFactory::new(
                move |id, batch| {
                    CollectionFrame::with_recycler(id, batch, Some(recycler.clone()), None)
                },
                |batch, frame: &mut CollectionFrame<Batch>| {
                    frame.replace(batch);
                },
            ))
            .unwrap();

        let frame = manager.get_or_create(vec![Recording::new(1, &log).boxed()]);
        run(frame);
        let reused = manager.get_or_create(vec![Recording::new(2, &log).boxed()]);
        assert_eq!(manager.total_recycled(), 1);
        run(reused);
        assert_eq!(*log.lock().unwrap(), [1, 2]);
        assert_eq!(manager.recycle_queue_size(), 1);
    }
}
