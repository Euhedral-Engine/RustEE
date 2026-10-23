use super::IngestSink;
use super::downstream::Downstream;
use crate::frames::{Frame, FrameError};
use crate::generics::{FrameSink, LatticeReceiver, LatticeSource};
use euhedral_data_structures::queues::partitioned::{
    PartitionedConsumer, PartitionedMpscQueue, PartitionedProducer,
};
use euhedral_data_structures::queues::queue::{Multi, Single};
use std::num::NonZeroUsize;
use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, Ordering, fence};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError, TryLockError, Weak};

type Consumer = PartitionedConsumer<Box<dyn Frame>, Single>;

/// The chunk size of the default queue.
pub(super) const DEFAULT_CHUNK: usize = 8192;

/// When a source that was told to finish completes.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Ending {
    /// When a drain finds nothing to deliver and the queue is empty.
    Gracefully,
    /// As soon as a drain leaves the queue empty.
    Exhausted,
}

/// What an owner layered over the queue does when it completes and after each drain.
pub(super) trait Owner: Send + Sync {
    /// Completes the owner, as the lattice or a finished drain asks.
    fn complete(&self, source: &QueueSource);

    /// Runs after each drain lets go of the queue, and after a publication that raced the close.
    /// Must not block.
    fn drain_finished(&self, source: &QueueSource);
}

/// The lattice-facing side of a queue sink: a [`LatticeSource`] that delivers queued frames to
/// demand, and a [`FrameSink`] that queues them.
///
/// One drain holds the queue at a time. A drain blocked out by another leaves the delivery to it:
/// the holder runs again for requests that arrived while it held the queue.
pub(super) struct QueueSource {
    downstream: Downstream,
    queue: PartitionedMpscQueue<Box<dyn Frame>>,
    producer: PartitionedProducer<Box<dyn Frame>, Multi>,
    consumer: Mutex<Consumer>,
    /// A request waiting for a drain: a wakeup, not outstanding demand, so an empty queue never
    /// spins.
    request_pending: AtomicBool,
    finish: AtomicBool,
    publication_closed: AtomicBool,
    ending: Ending,
    owner: OnceLock<Weak<dyn Owner>>,
}

impl QueueSource {
    pub(super) fn new(partitions: NonZeroUsize, chunk_size: usize, ending: Ending) -> Arc<Self> {
        let queue = PartitionedMpscQueue::with_partitions(partitions.get(), chunk_size);
        Arc::new(Self {
            downstream: Downstream::default(),
            producer: queue.producer(),
            consumer: Mutex::new(queue.consumer()),
            queue,
            request_pending: AtomicBool::new(false),
            finish: AtomicBool::new(false),
            publication_closed: AtomicBool::new(false),
            ending,
            owner: OnceLock::new(),
        })
    }

    /// Hands completion and drain notifications to `owner`.
    pub(super) fn set_owner(&self, owner: Weak<dyn Owner>) {
        assert!(
            self.owner.set(owner).is_ok(),
            "the queue already has an owner"
        );
    }

    fn owner(&self) -> Option<Arc<dyn Owner>> {
        self.owner.get().and_then(Weak::upgrade)
    }

    pub(super) fn partitions(&self) -> usize {
        self.queue.partitions()
    }

    /// Queues `frame` with `offer`, unless publication has closed.
    fn publish(
        &self,
        frame: Box<dyn Frame>,
        offer: impl FnOnce(
            &PartitionedProducer<Box<dyn Frame>, Multi>,
            Box<dyn Frame>,
        ) -> Result<(), Box<dyn Frame>>,
    ) -> Result<(), Box<dyn Frame>> {
        if self.publication_closed.load(Ordering::Acquire) {
            return Err(frame);
        }
        offer(&self.producer, frame)?;
        // Closing does not wait for publishers: a frame queued as the sink closes is seen either by
        // the drain after the close, or by its publisher here.
        fence(Ordering::SeqCst);
        if self.publication_closed.load(Ordering::Acquire)
            && let Some(owner) = self.owner()
        {
            owner.drain_finished(self);
        }
        Ok(())
    }

    pub(super) fn offer(&self, frame: Box<dyn Frame>) -> Result<(), Box<dyn Frame>> {
        self.publish(frame, |producer, frame| producer.offer(frame))
    }

    pub(super) fn offer_seeded(
        &self,
        seed: u64,
        frame: Box<dyn Frame>,
    ) -> Result<(), Box<dyn Frame>> {
        self.publish(frame, |producer, frame| producer.offer_seeded(seed, frame))
    }

    /// # Panics
    ///
    /// If `partition` is not below the partition count.
    pub(super) fn offer_to(
        &self,
        partition: usize,
        frame: Box<dyn Frame>,
    ) -> Result<(), Box<dyn Frame>> {
        assert!(
            partition < self.partitions(),
            "partition {partition} is out of range for {} partitions",
            self.partitions()
        );
        self.publish(frame, |producer, frame| producer.offer_to(partition, frame))
    }

    pub(super) fn size(&self) -> u64 {
        self.queue.size_long()
    }

    pub(super) fn demand(&self) -> u64 {
        self.downstream.demand()
    }

    /// Closes publication and completes the source, leaving what is queued.
    pub(super) fn close(&self) {
        self.publication_closed.store(true, Ordering::Release);
        fence(Ordering::SeqCst);
        self.downstream.complete();
    }

    /// Completes the source the way its owner does.
    fn complete_owner(&self) {
        match self.owner() {
            Some(owner) => owner.complete(self),
            None => self.close(),
        }
    }

    fn drain_finished(&self) {
        if let Some(owner) = self.owner() {
            owner.drain_finished(self);
        }
    }

    /// Completes the source once the queue has drained, as its [`Ending`] says.
    pub(super) fn finish(&self) {
        self.finish.store(true, Ordering::Release);
    }

    fn should_end(&self, delivered: u64) -> bool {
        self.finish.load(Ordering::Acquire)
            && (delivered == 0 || self.ending == Ending::Exhausted)
            && self.queue.is_empty()
    }

    /// The queue, unless a drain holds it.
    fn try_hold(&self) -> Option<MutexGuard<'_, Consumer>> {
        match self.consumer.try_lock() {
            Ok(consumer) => Some(consumer),
            Err(TryLockError::Poisoned(poisoned)) => Some(poisoned.into_inner()),
            Err(TryLockError::WouldBlock) => None,
        }
    }

    /// Hands every queued frame to `consume`, unless a drain holds the queue: that drain hands the
    /// frames queued behind it on when it lets go.
    pub(super) fn discard_queued(&self, mut consume: impl FnMut(Box<dyn Frame>)) {
        loop {
            let Some(mut consumer) = self.try_hold() else {
                return;
            };
            consumer.drain_all(&mut consume);
            drop(consumer);
            if self.queue.is_empty() {
                return;
            }
        }
    }

    /// Cancels every queued frame, waiting for a running drain.
    pub(super) fn clear(&self) {
        let mut consumer = self.consumer.lock().unwrap_or_else(PoisonError::into_inner);
        consumer.drain_all(|frame| frame.do_finally_with_error(FrameError::Cancelled));
    }

    /// Delivers queued frames to the receiver while requests are waiting and no other drain holds
    /// the queue.
    fn drain_requests(&self) {
        while self.request_pending.load(Ordering::Acquire) {
            let Some(mut consumer) = self.try_hold() else {
                return;
            };
            self.request_pending.store(false, Ordering::Release);
            let receiver = self.downstream.receiver();
            let delivered = match &receiver {
                Some(receiver) if !self.downstream.is_complete() => {
                    let drained = panic::catch_unwind(AssertUnwindSafe(|| {
                        consumer.drain_while(
                            |frame| receiver.push(frame),
                            |_| self.downstream.is_complete(),
                            self.downstream.demand(),
                        )
                    }));
                    drop(consumer);
                    match drained {
                        Ok(delivered) => {
                            self.downstream.deliver(delivered);
                            Some(delivered)
                        }
                        Err(panic) => {
                            self.drain_finished();
                            panic::resume_unwind(panic);
                        }
                    }
                }
                _ => {
                    drop(consumer);
                    None
                }
            };
            self.drain_finished();
            match delivered {
                Some(delivered) if self.should_end(delivered) => self.complete_owner(),
                Some(_) => {}
                None => return,
            }
        }
    }
}

impl LatticeSource for QueueSource {
    fn add_downstream(self: Arc<Self>, downstream: Arc<dyn LatticeReceiver>) {
        self.downstream.attach(self.clone(), downstream);
    }

    /// Hands queued frames to `consumer`, unless another drain holds the queue. Requests that
    /// arrive meanwhile are delivered once the pull lets go. A drain stops once the source
    /// completes.
    fn pull(
        &self,
        consumer: &mut dyn FnMut(Box<dyn Frame>),
        stop: &mut dyn FnMut(&dyn Frame) -> bool,
        demand: u64,
    ) -> u64 {
        if demand == 0 || self.downstream.is_complete() || self.downstream.receiver().is_none() {
            return 0;
        }
        let Some(mut queue) = self.try_hold() else {
            return 0;
        };
        let pulled = panic::catch_unwind(AssertUnwindSafe(|| {
            queue.drain_while(
                &mut *consumer,
                |frame| self.downstream.is_complete() || stop(&**frame),
                demand,
            )
        }));
        drop(queue);
        self.drain_finished();
        self.drain_requests();
        let pulled = pulled.unwrap_or_else(|panic| panic::resume_unwind(panic));
        if self.should_end(pulled) {
            self.complete_owner();
        }
        pulled
    }

    /// Adds `demand` and delivers queued frames up to the total demand.
    fn request(&self, demand: u64) {
        if demand == 0 || self.downstream.is_complete() || self.downstream.receiver().is_none() {
            return;
        }
        self.downstream.add_demand(demand);
        self.request_pending.store(true, Ordering::Release);
        self.drain_requests();
    }

    /// Completes the source the way its owner does.
    fn complete(&self) {
        self.complete_owner();
    }

    fn is_complete(&self) -> bool {
        self.downstream.is_complete()
    }
}

impl FrameSink for QueueSource {
    /// Queues `frame` in a random partition, or hands it back once publication has closed.
    fn offer(&self, frame: Box<dyn Frame>) -> Result<(), Box<dyn Frame>> {
        QueueSource::offer(self, frame)
    }
}

/// A partitioned queue that any thread publishes frames to, delivered to the lattice as it asks.
///
/// Frames in one partition are delivered in order; partitions have no order between them. A pull
/// stops before the first frame its caller's stop condition matches, leaving it queued.
pub struct QueueIngestSink {
    source: Arc<QueueSource>,
}

impl Default for QueueIngestSink {
    /// One partition.
    fn default() -> Self {
        Self::new()
    }
}

impl QueueIngestSink {
    /// One partition.
    pub fn new() -> Self {
        Self::with_partitions(NonZeroUsize::MIN)
    }

    pub fn with_partitions(partitions: NonZeroUsize) -> Self {
        Self {
            source: QueueSource::new(partitions, DEFAULT_CHUNK, Ending::Gracefully),
        }
    }

    pub fn partitions(&self) -> usize {
        self.source.partitions()
    }

    /// Queues `frame` in a random partition, or hands it back once the sink has completed.
    pub fn offer(&self, frame: Box<dyn Frame>) -> Result<(), Box<dyn Frame>> {
        self.source.offer(frame)
    }

    /// Queues `frame` in the partition `seed` picks: the same seed always picks the same one.
    pub fn offer_seeded(&self, seed: u64, frame: Box<dyn Frame>) -> Result<(), Box<dyn Frame>> {
        self.source.offer_seeded(seed, frame)
    }

    /// Queues `frame` in `partition`.
    ///
    /// # Panics
    ///
    /// If `partition` is not below the partition count.
    pub fn offer_to(&self, partition: usize, frame: Box<dyn Frame>) -> Result<(), Box<dyn Frame>> {
        self.source.offer_to(partition, frame)
    }

    /// Cancels every queued frame, waiting for a running drain. Do not call it from a drain.
    pub fn clear(&self) {
        self.source.clear();
    }

    /// Frames queued, give or take those being published or drained.
    pub fn size(&self) -> u64 {
        self.source.size()
    }

    /// Demand asked for and not yet delivered.
    pub fn demand(&self) -> u64 {
        self.source.demand()
    }

    /// Completes the sink once the lattice has drained the queue. Frames may still be published
    /// until then.
    pub fn complete_gracefully(&self) {
        self.source.finish();
    }
}

impl IngestSink for QueueIngestSink {
    fn delegate(&self) -> Arc<dyn LatticeSource> {
        self.source.clone()
    }

    /// Stops publication and detaches from the lattice; queued frames stay until cleared.
    fn complete(&self) {
        self.source.close();
    }

    fn is_complete(&self) -> bool {
        self.source.is_complete()
    }
}

#[cfg(test)]
mod tests;
