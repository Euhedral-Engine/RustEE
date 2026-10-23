use super::FrameFactory;
use crate::frames::Frame;
use euhedral_data_structures::queues::{Consumer, MpscQueue, Multi, Producer, Single};
use std::fmt;
use std::sync::{Arc, Weak};

/// The default number of frames a [`FrameManager`] holds for reuse.
pub const DEFAULT_CAPACITY: usize = 8_192;

/// The smallest batch a [`FrameManager`] moves out of its recycle queue at once.
const MIN_BATCH: usize = 256;

/// A pool of reusable frames of type `F`, refilled with data of type `D`.
///
/// Frames come back through [`Recycler`] handles, from any thread, into a bounded MPSC queue. The
/// manager itself is the queue's single consumer: checking frames out takes `&mut self`, so only its
/// owner can do it. When the pool is empty, [`get_or_create`](Self::get_or_create) asks the
/// [`FrameFactory`] for a new frame; when the queue is full, a returned frame is dropped.
///
/// Frames only hold a weak reference to the pool, so frames still in the queue never keep it alive,
/// and frames returned after the manager is dropped are simply dropped.
pub struct FrameManager<D, F> {
    consumer: Consumer<Box<F>, Single>,
    producer: Arc<Producer<Box<F>, Multi>>,
    buffer: Vec<Box<F>>,
    batch: usize,
    capacity: usize,
    factory: Option<FrameFactory<D, F>>,
    total_recycled: u64,
}

impl<D, F> FrameManager<D, F> {
    /// A pool holding up to [`DEFAULT_CAPACITY`] returned frames.
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_CAPACITY)
    }

    /// A pool holding up to `capacity` returned frames, rounded up to a power of two (at least one).
    ///
    /// # Panics
    ///
    /// Panics if the rounded capacity does not fit in a `usize`.
    pub fn with_capacity(capacity: usize) -> Self {
        let capacity = capacity
            .max(1)
            .checked_next_power_of_two()
            .expect("frame pool capacity overflows usize");
        let queue = MpscQueue::bounded(capacity);
        let batch = capacity.max(MIN_BATCH);
        Self {
            consumer: queue.consumer(),
            producer: Arc::new(queue.producer()),
            buffer: Vec::with_capacity(batch),
            batch,
            capacity,
            factory: None,
            total_recycled: 0,
        }
    }

    /// How many returned frames the recycle queue holds before it starts dropping them.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Frames moved from the recycle queue into the checkout buffer so far.
    pub fn total_recycled(&self) -> u64 {
        self.total_recycled
    }

    /// Returned frames waiting in the recycle queue.
    pub fn recycle_queue_size(&self) -> u64 {
        self.consumer.size_long()
    }

    /// Returned frames already moved into the checkout buffer.
    pub fn buffered(&self) -> usize {
        self.buffer.len()
    }

    /// A handle that returns frames to this pool, for the frames themselves to hold.
    pub fn recycler(&self) -> Recycler<F> {
        Recycler {
            producer: Arc::downgrade(&self.producer),
        }
    }

    /// Returns `frame` to the pool. False, dropping the frame, when the recycle queue is full.
    pub fn recycle(&self, frame: Box<F>) -> bool {
        self.producer.offer(frame).is_ok()
    }

    pub fn factory(&self) -> Option<&FrameFactory<D, F>> {
        self.factory.as_ref()
    }

    /// Installs the factory [`get_or_create`](Self::get_or_create) uses. A pool takes one factory
    /// for life: if it already has one, `factory` is handed back.
    pub fn set_factory(&mut self, factory: FrameFactory<D, F>) -> Result<(), FrameFactory<D, F>> {
        if self.factory.is_some() {
            return Err(factory);
        }
        self.factory = Some(factory);
        Ok(())
    }

    /// A returned frame, if there is one.
    pub fn get(&mut self) -> Option<Box<F>> {
        if self.buffer.is_empty() {
            let buffer = &mut self.buffer;
            let drained = self
                .consumer
                .drain(|frame| buffer.push(frame), self.batch as u64);
            self.total_recycled += drained;
        }
        self.buffer.pop()
    }

    /// Discards up to `max` returned frames, buffered ones first, and reports how many it dropped.
    pub fn dump(&mut self, max: u64) -> u64 {
        let from_buffer = (self.buffer.len() as u64).min(max);
        self.buffer
            .truncate(self.buffer.len() - from_buffer as usize);
        let mut total = from_buffer;
        while total < max {
            let drained = self.consumer.drain(drop, max - total);
            if drained == 0 {
                break;
            }
            total += drained;
        }
        total
    }

    /// Discards every returned frame. The pool stays usable.
    pub fn close(&mut self) {
        self.buffer.clear();
        self.consumer.clear();
    }

    /// An empty pool with the same capacity and no factory.
    pub fn copy_empty(&self) -> Self {
        Self::with_capacity(self.capacity)
    }
}

impl<D, F: Frame> FrameManager<D, F> {
    /// A returned frame refilled with `data`, or a new frame made from it.
    ///
    /// # Panics
    ///
    /// Panics if no factory has been [set](Self::set_factory).
    pub fn get_or_create(&mut self, data: D) -> Box<F> {
        let recycled = self.get();
        let factory = self
            .factory
            .as_mut()
            .expect("FrameManager::get_or_create needs a FrameFactory");
        match recycled {
            Some(mut frame) => {
                factory.replace(data, &mut frame);
                frame
            }
            None => factory.create(data),
        }
    }
}

impl<D, F> Default for FrameManager<D, F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<D, F> fmt::Debug for FrameManager<D, F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FrameManager")
            .field("capacity", &self.capacity)
            .field("buffered", &self.buffer.len())
            .field("queued", &self.consumer.size_long())
            .field("total_recycled", &self.total_recycled)
            .field("factory", &self.factory)
            .finish()
    }
}

/// Returns frames to a [`FrameManager`] from any thread.
///
/// A frame that belongs to a pool keeps its recycler and returns itself with
/// [`recycle_owned`](Self::recycle_owned).
pub struct Recycler<F> {
    producer: Weak<Producer<Box<F>, Multi>>,
}

impl<F> Recycler<F> {
    /// Returns `frame` to the pool. False, dropping the frame, when the queue is full or the pool is
    /// gone.
    pub fn recycle(&self, frame: Box<F>) -> bool {
        match self.producer.upgrade() {
            Some(producer) => producer.offer(frame).is_ok(),
            None => false,
        }
    }

    /// Returns `frame` to the pool named by the recycler `recycler` finds inside it, if any.
    ///
    /// ```
    /// # use euhedral_core::frames::{Frame, FrameHeader};
    /// # use euhedral_core::impls::{FrameManager, Recycler};
    /// struct Pooled {
    ///     header: FrameHeader,
    ///     recycler: Option<Recycler<Pooled>>,
    /// }
    ///
    /// impl Frame for Pooled {
    ///     fn header(&self) -> &FrameHeader { &self.header }
    ///     fn header_mut(&mut self) -> &mut FrameHeader { &mut self.header }
    ///     fn recycle(self: Box<Self>) {
    ///         Recycler::recycle_owned(self, |frame| frame.recycler.as_ref());
    ///     }
    /// }
    ///
    /// let mut manager = FrameManager::<(), Pooled>::new();
    /// let frame = Box::new(Pooled { header: FrameHeader::new(1), recycler: Some(manager.recycler()) });
    /// frame.do_finally();
    /// assert!(manager.get().is_some());
    /// ```
    pub fn recycle_owned(frame: Box<F>, recycler: impl FnOnce(&F) -> Option<&Recycler<F>>) -> bool {
        let Some(producer) = recycler(&frame).and_then(|recycler| recycler.producer.upgrade())
        else {
            return false;
        };
        producer.offer(frame).is_ok()
    }

    /// Whether this and `other` return frames to the same pool.
    pub fn same_pool(&self, other: &Self) -> bool {
        Weak::ptr_eq(&self.producer, &other.producer)
    }

    /// Whether this returns frames to `manager`.
    pub fn belongs_to<D>(&self, manager: &FrameManager<D, F>) -> bool {
        std::ptr::eq(self.producer.as_ptr(), Arc::as_ptr(&manager.producer))
    }
}

impl<F> Clone for Recycler<F> {
    fn clone(&self) -> Self {
        Self {
            producer: Weak::clone(&self.producer),
        }
    }
}

impl<F> fmt::Debug for Recycler<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Recycler")
            .field("live", &(self.producer.strong_count() > 0))
            .finish()
    }
}

#[cfg(test)]
mod tests;
