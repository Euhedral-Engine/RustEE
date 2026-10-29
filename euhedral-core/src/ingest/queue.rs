use super::IngestSink;
use crate::frames::{Frame, FrameError};
use crate::generics::{DownstreamSlot, FrameSink, LatticeReceiver, LatticeSource};
use euhedral_data_structures::atomics::CachePadded;
use euhedral_data_structures::queues::partitioned::{
    PartitionedConsumer, PartitionedMpscQueue, PartitionedProducer,
};
use euhedral_data_structures::queues::queue::{Multi, Single};
use euhedral_data_structures::sync::{ClaimCell, ClaimGuard, HotScope};
use std::any::Any;
use std::cell::Cell;
use std::num::NonZeroUsize;
use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering, fence};
use std::sync::{Arc, OnceLock, Weak};

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
/// One drain holds the queue at a time: the `consumer` claim is Java's `draining` flag. A drain
/// blocked out by another leaves the delivery to it: the holder runs again for requests that
/// arrived while it held the queue, and nothing ever waits.
///
/// Lifecycle calls set atomics and try the claim once: `close` sets `publication_closed` and
/// completes the downstream slot, a kill sets the owner's kill switch, and `clear` sets
/// `clear_requested`. A holder applies them as it lets go (`drain_finished`), and the next
/// `request` or `pull` otherwise. `Drop` finishes what is still queued.
pub(super) struct QueueSource {
    downstream: DownstreamSlot,
    /// Demand asked for and not yet delivered (Java's padded `demand`).
    demand: CachePadded<AtomicU64>,
    queue: PartitionedMpscQueue<Box<dyn Frame>>,
    producer: PartitionedProducer<Box<dyn Frame>, Multi>,
    consumer: ClaimCell<Consumer>,
    /// A request waiting for a drain: a wakeup, not outstanding demand, so an empty queue never
    /// spins (Java's `requestPending`).
    request_pending: AtomicBool,
    /// A `clear` waiting for whoever holds the queue.
    clear_requested: AtomicBool,
    finish: AtomicBool,
    publication_closed: AtomicBool,
    ending: Ending,
    owner: OnceLock<Weak<dyn Owner>>,
}

impl QueueSource {
    pub(super) fn new(partitions: NonZeroUsize, chunk_size: usize, ending: Ending) -> Arc<Self> {
        let queue = PartitionedMpscQueue::with_partitions(partitions.get(), chunk_size);
        Arc::new(Self {
            downstream: DownstreamSlot::new(),
            demand: CachePadded::new(AtomicU64::new(0)),
            producer: queue.producer(),
            consumer: ClaimCell::new(queue.consumer()),
            queue,
            request_pending: AtomicBool::new(false),
            clear_requested: AtomicBool::new(false),
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
        let _hot = HotScope::enter();
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
        self.demand.load(Ordering::Acquire)
    }

    /// Adds `demand`, stopping at the largest value, and returns the total (Java's `accumulate`
    /// with a positive step, a compare-and-set loop).
    fn add_demand(&self, demand: u64) -> u64 {
        self.demand
            .try_update(Ordering::SeqCst, Ordering::SeqCst, |current| {
                Some(current.saturating_add(demand))
            })
            .map_or_else(
                |current| current,
                |previous| previous.saturating_add(demand),
            )
    }

    /// Takes `delivered` off the demand, stopping at zero (Java's `accumulate` with a negative
    /// step), and returns what is left.
    fn take_demand(&self, delivered: u64) -> u64 {
        self.demand
            .try_update(Ordering::SeqCst, Ordering::SeqCst, |current| {
                Some(current.saturating_sub(delivered))
            })
            .map_or_else(
                |current| current,
                |previous| previous.saturating_sub(delivered),
            )
    }

    /// Closes publication and completes the source, leaving what is queued.
    pub(super) fn close(&self) {
        self.publication_closed.store(true, Ordering::SeqCst);
        fence(Ordering::SeqCst);
        // Java forgets the demand (`lazySet(0)`) before it tells the receiver.
        self.demand.store(0, Ordering::Release);
        self.downstream.complete();
    }

    /// Completes the source the way its owner does.
    fn complete_owner(&self) {
        match self.owner() {
            Some(owner) => owner.complete(self),
            None => self.close(),
        }
    }

    /// What runs after a drain lets go of the queue (Java's `onDrainFinished`): a waiting `clear`,
    /// then the owner's. A panic in either is raised after both have run.
    fn drain_finished(&self) {
        let cleared = self.apply_clear();
        let owned = panic::catch_unwind(AssertUnwindSafe(|| {
            if let Some(owner) = self.owner() {
                owner.drain_finished(self);
            }
        }));
        if let Some(panic) = cleared.or(owned.err()) {
            panic::resume_unwind(panic);
        }
    }

    /// The tail of every release of the queue: the waiting `clear`, the owner's hook, and the
    /// requests that were refused the queue while it was held.
    fn after_release(&self) {
        let finished = panic::catch_unwind(AssertUnwindSafe(|| self.drain_finished()));
        self.drain_requests();
        if let Err(panic) = finished {
            panic::resume_unwind(panic);
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
    fn try_hold(&self) -> Option<ClaimGuard<'_, Consumer>> {
        self.consumer.try_claim()
    }

    /// Hands every queued frame to `consume`, unless a drain holds the queue: that drain hands the
    /// frames queued behind it on when it lets go (Java's `discardQueued`). A panic in `consume` is
    /// raised after the rest are handed over and the release's tail has run.
    pub(super) fn discard_queued(&self, mut consume: impl FnMut(Box<dyn Frame>)) {
        let mut panicked = None;
        loop {
            let Some(mut consumer) = self.try_hold() else {
                break;
            };
            Self::drain_catching(&mut consumer, &mut consume, &mut panicked);
            // The release is ordered before the emptiness check, so a frame published behind the
            // drain is seen here, or its publisher takes the claim.
            consumer.release_seq_cst();
            if self.queue.is_empty() {
                break;
            }
        }
        panicked = self.apply_clear().or(panicked);
        self.drain_requests();
        if let Some(panic) = panicked {
            panic::resume_unwind(panic);
        }
    }

    /// Hands every queued frame to `consume`, keeping the first panic and going on.
    fn drain_catching(
        consumer: &mut Consumer,
        mut consume: impl FnMut(Box<dyn Frame>),
        panicked: &mut Option<Box<dyn Any + Send>>,
    ) {
        consumer.drain_all(|frame| {
            if let Err(panic) = panic::catch_unwind(AssertUnwindSafe(|| consume(frame))) {
                panicked.get_or_insert(panic);
            }
        });
    }

    /// Cancels every queued frame. A drain that holds the queue cancels them when it lets go, so the
    /// call never waits.
    pub(super) fn clear(&self) {
        self.clear_requested.store(true, Ordering::SeqCst);
        let panicked = self.apply_clear();
        self.drain_requests();
        if let Some(panic) = panicked {
            panic::resume_unwind(panic);
        }
    }

    /// Cancels the queued frames while a `clear` is waiting, if the queue can be had. Only a holder
    /// takes the request off, so a `clear` that finds the queue held is answered by the holder.
    /// Returns the first panic of a cancellation; the rest were cancelled all the same.
    fn apply_clear(&self) -> Option<Box<dyn Any + Send>> {
        let mut panicked = None;
        while self.clear_requested.load(Ordering::SeqCst) {
            let Some(mut consumer) = self.try_hold() else {
                break;
            };
            if self.clear_requested.swap(false, Ordering::SeqCst) {
                Self::drain_catching(
                    &mut consumer,
                    |frame| frame.do_finally_with_error(FrameError::Cancelled),
                    &mut panicked,
                );
            }
            consumer.release_seq_cst();
        }
        panicked
    }

    /// Delivers queued frames to the receiver while requests are waiting and no other drain holds
    /// the queue (Java's `drainRequests`).
    fn drain_requests(&self) {
        // The wakeup is published before the claim is tried: the releasing holder either sees it,
        // or the requester takes the claim itself.
        while self.request_pending.load(Ordering::SeqCst) {
            let Some(mut consumer) = self.try_hold() else {
                return;
            };
            // One read-modify-write takes the wakeup and synchronizes with the store that set it,
            // so the demand banked before that store is seen below (a plain store of `false`
            // could overwrite a racing request's and leave this holder reading older demand).
            if !self.request_pending.swap(false, Ordering::SeqCst) {
                consumer.release_seq_cst();
                continue;
            }
            let delivered = match self.downstream.receiver() {
                Some(receiver) if !self.downstream.is_complete() => {
                    // Counted before each push, so a push that panics still counts: the frame
                    // left the queue.
                    let pushed = Cell::new(0_u64);
                    let drained = panic::catch_unwind(AssertUnwindSafe(|| {
                        consumer.drain_while(
                            |frame| {
                                pushed.set(pushed.get() + 1);
                                receiver.push(frame);
                            },
                            |_| self.downstream.is_complete(),
                            self.demand.load(Ordering::SeqCst),
                        )
                    }));
                    match drained {
                        Ok(delivered) => {
                            // Accounted before the claim is released, so the next holder cannot
                            // over-deliver.
                            self.take_demand(delivered);
                            consumer.release_seq_cst();
                            Some(delivered)
                        }
                        Err(panic) => {
                            // What was pushed is off the demand, so a later holder pushes no more
                            // than was asked for.
                            self.take_demand(pushed.get());
                            consumer.release_seq_cst();
                            let _ = panic::catch_unwind(AssertUnwindSafe(|| self.drain_finished()));
                            panic::resume_unwind(panic);
                        }
                    }
                }
                _ => {
                    consumer.release_seq_cst();
                    None
                }
            };
            self.drain_finished();
            // A request refused while the queue was held (or while `drain_finished` held it) is
            // seen by the loop's condition.
            if let Some(delivered) = delivered
                && self.should_end(delivered)
            {
                self.complete_owner();
            }
        }
    }
}

impl Drop for QueueSource {
    /// Finishes every frame still queued: a frame is owned by whoever holds its `Box`, and nothing
    /// is left to deliver it.
    fn drop(&mut self) {
        self.consumer
            .get_mut()
            .drain_all(|frame| frame.do_finally_with_error(FrameError::Cancelled));
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
        queue.release_seq_cst();
        let finished = panic::catch_unwind(AssertUnwindSafe(|| self.after_release()));
        let pulled = pulled.unwrap_or_else(|panic| panic::resume_unwind(panic));
        if let Err(panic) = finished {
            panic::resume_unwind(panic);
        }
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
        self.add_demand(demand);
        self.request_pending.store(true, Ordering::SeqCst);
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

    /// Cancels every queued frame. If a drain holds the queue it cancels them when it lets go, so the
    /// call never waits, and may be made from a drain.
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
        // Completing is lifecycle work, also when a frame does it.
        let _cold = HotScope::leave();
        self.source.close();
    }

    fn is_complete(&self) -> bool {
        self.source.is_complete()
    }
}

#[cfg(test)]
mod tests;
