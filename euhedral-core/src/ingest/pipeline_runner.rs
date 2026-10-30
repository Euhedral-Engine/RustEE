use super::IngestSink;
use super::queue::{DEFAULT_CHUNK, Ending, Owner, QueueSource};
use crate::frames::{Completion, Frame, FrameError, Outcome, PipelineBuilder, PipelineFrame};
use crate::generics::{FrameSink, LatticeSource};
use crate::impls::FrameManager;
use std::error::Error;
use std::fmt;
use std::num::NonZeroUsize;
use std::panic::{self, AssertUnwindSafe};
use std::ptr;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, Ordering};
use std::sync::{Arc, Weak};

/// Set in the admission word once admission closes; the rest counts accepted chains that have not
/// finished.
const ADMISSION_CLOSED: u64 = 1 << 63;

/// Input refused because the runner no longer admits chains; the input is handed back.
pub struct AdmissionClosed<I>(pub I);

impl<I> fmt::Debug for AdmissionClosed<I> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("AdmissionClosed(..)")
    }
}

impl<I> fmt::Display for AdmissionClosed<I> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("pipeline admission is closed")
    }
}

impl<I> Error for AdmissionClosed<I> {}

type Observer = Box<dyn FnOnce(Outcome) + Send>;

/// How a submitted chain ended, as the submitter reads it: a boxed [`Outcome`] behind an `AtomicPtr`, published once with release
/// ordering by the chain's last frame and taken with acquire ordering by whoever checks (Java's `CompletableFuture` the chain
/// completes, with one completer and one reader). Nothing waits: [`take`](Self::take) is `None` until the chain has ended.
///
/// A chain that is dropped without ending (a lattice closed under it) publishes [`Outcome::Cancelled`], so a reader never has to
/// infer that nothing is coming.
pub struct OutcomeSlot {
    outcome: AtomicPtr<Outcome>,
}

impl OutcomeSlot {
    fn publish(&self, outcome: Outcome) {
        let published = self
            .outcome
            .swap(Box::into_raw(Box::new(outcome)), Ordering::Release);
        debug_assert!(published.is_null(), "a chain ends once");
        if !published.is_null() {
            // SAFETY: a pointer from `Box::into_raw` in an earlier `publish`, taken out of the slot by this swap.
            drop(unsafe { Box::from_raw(published) });
        }
    }

    /// Whether the chain has ended and its outcome has not been taken.
    pub fn is_set(&self) -> bool {
        !self.outcome.load(Ordering::Acquire).is_null()
    }

    /// The outcome, once the chain has ended; `None` before, and after it was taken.
    pub fn take(&self) -> Option<Outcome> {
        let outcome = self.outcome.swap(ptr::null_mut(), Ordering::Acquire);
        // SAFETY: a pointer from `Box::into_raw` in `publish`; the swap handed its ownership to this call alone, and the acquire
        // makes the chain's writes to the outcome visible.
        (!outcome.is_null()).then(|| *unsafe { Box::from_raw(outcome) })
    }
}

impl Drop for OutcomeSlot {
    fn drop(&mut self) {
        let outcome = *self.outcome.get_mut();
        if !outcome.is_null() {
            // SAFETY: as in `take`; `&mut self` rules out any other access.
            drop(unsafe { Box::from_raw(outcome) });
        }
    }
}

impl fmt::Debug for OutcomeSlot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OutcomeSlot")
            .field("set", &self.is_set())
            .finish()
    }
}

/// The chain's end of an [`OutcomeSlot`]: it tells the outcome, or publishes `Cancelled` when the observer is dropped untold.
struct Report {
    slot: Arc<OutcomeSlot>,
    told: bool,
}

impl Report {
    fn tell(mut self, outcome: Outcome) {
        self.slot.publish(outcome);
        self.told = true;
    }
}

impl Drop for Report {
    fn drop(&mut self) {
        if !self.told {
            self.slot.publish(Outcome::Cancelled);
        }
    }
}

/// The slot that hears how a chain ended, and the observer that tells it.
fn observed() -> (Arc<OutcomeSlot>, Observer) {
    let slot = Arc::new(OutcomeSlot {
        outcome: AtomicPtr::new(ptr::null_mut()),
    });
    let report = Report {
        slot: slot.clone(),
        told: false,
    };
    let observer = Box::new(move |result| report.tell(result));
    (slot, observer)
}

/// What a runner shares with its chains and with whoever closes it.
struct Shared {
    source: Arc<QueueSource>,
    kill_switch: Arc<AtomicBool>,
    admission: AtomicU64,
}

impl Shared {
    /// Counts in one more chain, unless admission is closed.
    fn admit(&self) -> bool {
        self.admission
            .try_update(Ordering::AcqRel, Ordering::Acquire, |state| {
                (state & ADMISSION_CLOSED == 0).then_some(state + 1)
            })
            .is_ok()
    }

    /// Counts a chain out; the last one after admission closed completes the source.
    fn chain_finished(&self) {
        if self.admission.fetch_sub(1, Ordering::AcqRel) - 1 == ADMISSION_CLOSED {
            self.source.close();
        }
    }

    /// Closes admission; returns whether no accepted chain is left.
    fn close_admission(&self) -> bool {
        self.admission.fetch_or(ADMISSION_CLOSED, Ordering::AcqRel) & !ADMISSION_CLOSED == 0
    }

    fn complete_gracefully(&self) {
        if self.close_admission() {
            self.source.close();
        }
    }

    /// Kills every chain, closes admission and the source, then cancels whatever is queued.
    fn kill(&self) {
        // Sequentially consistent, as a volatile write is: a drain that holds the queue reads it
        // after letting go, and the kill's own claim attempt must not both miss it.
        self.kill_switch.store(true, Ordering::SeqCst);
        self.close_admission();
        let closed = panic::catch_unwind(AssertUnwindSafe(|| self.source.close()));
        self.cancel_queued(&self.source);
        if let Err(panic) = closed {
            panic::resume_unwind(panic);
        }
    }

    fn cancel_queued(&self, source: &QueueSource) {
        if self.kill_switch.load(Ordering::SeqCst) {
            source.discard_queued(|frame| frame.do_finally_with_error(FrameError::Cancelled));
        }
    }
}

impl Owner for Shared {
    /// The lattice completing the source kills the runner.
    fn complete(&self, _source: &QueueSource) {
        self.kill();
    }

    fn drain_finished(&self, source: &QueueSource) {
        self.cancel_queued(source);
    }
}

/// Runs typed [`PipelineFrame`] chains through the lattice.
///
/// Each input checks a chain out of the runner's pool and queues it; the lattice runs the chain's
/// stages, each re-queued here, until its consumer takes the output. One owner submits, through
/// `&mut self`; chains finish on whichever threads run them.
///
/// [`complete_gracefully`](Self::complete_gracefully) stops admission and completes the runner
/// once every accepted chain has finished. [`complete`](IngestSink::complete), or the lattice
/// completing the delegate, kills the chains instead: queued ones end cancelled, and running ones
/// end cancelled when their stage returns. Complete a runner before dropping it: queued chains
/// keep its queue alive.
pub struct PipelineRunner<I> {
    manager: FrameManager<I, PipelineFrame<I>>,
    shared: Arc<Shared>,
    completion: Completion,
}

impl<I: Send + 'static> PipelineRunner<I> {
    /// A runner of `builder`'s pipeline with one partition. `consumer` takes each output, in
    /// parallel across chains if `consume_in_parallel`, in order otherwise.
    pub fn new<O: Send + 'static>(
        builder: &PipelineBuilder<I, O>,
        consumer: impl Fn(O) + Send + Sync + 'static,
        consume_in_parallel: bool,
    ) -> Self {
        Self::with_partitions(builder, consumer, consume_in_parallel, NonZeroUsize::MIN)
    }

    /// Like [`new`](Self::new), with `partitions` queue partitions.
    pub fn with_partitions<O: Send + 'static>(
        builder: &PipelineBuilder<I, O>,
        consumer: impl Fn(O) + Send + Sync + 'static,
        consume_in_parallel: bool,
        partitions: NonZeroUsize,
    ) -> Self {
        let source = QueueSource::new(partitions, DEFAULT_CHUNK, Ending::Gracefully);
        let shared = Arc::new(Shared {
            source: source.clone(),
            kill_switch: Arc::new(AtomicBool::new(false)),
            admission: AtomicU64::new(0),
        });
        let owner: Arc<dyn Owner> = shared.clone();
        source.set_owner(Arc::downgrade(&owner));
        let sink: Arc<dyn FrameSink> = source;
        let kill_switch = shared.kill_switch.clone();
        let manager = if consume_in_parallel {
            builder.compose_fanned_out(sink, consumer, kill_switch)
        } else {
            builder.compose_fanned_in(sink, consumer, kill_switch)
        };
        let finished = Arc::downgrade(&shared);
        let completion: Completion = Arc::new(move || {
            if let Some(shared) = Weak::upgrade(&finished) {
                shared.chain_finished();
            }
        });
        Self {
            manager,
            shared,
            completion,
        }
    }

    pub fn partitions(&self) -> usize {
        self.shared.source.partitions()
    }

    /// Runs `data` through the pipeline from a random partition.
    pub fn run(&mut self, data: I) -> Result<(), AdmissionClosed<I>> {
        let root = self.checkout(data, None)?;
        self.publish(root, |source, root| source.offer(root));
        Ok(())
    }

    /// Runs `data` from the partition `seed` picks: the same seed always picks the same one.
    pub fn run_seeded(&mut self, seed: u64, data: I) -> Result<(), AdmissionClosed<I>> {
        let root = self.checkout(data, None)?;
        self.publish(root, |source, root| source.offer_seeded(seed, root));
        Ok(())
    }

    /// Runs `data` from `partition`.
    ///
    /// # Panics
    ///
    /// If `partition` is not below the partition count; no chain is checked out then.
    pub fn run_on(&mut self, partition: usize, data: I) -> Result<(), AdmissionClosed<I>> {
        self.check_partition(partition);
        let root = self.checkout(data, None)?;
        self.publish(root, |source, root| source.offer_to(partition, root));
        Ok(())
    }

    /// Like [`run`](Self::run), and reports how the chain ended.
    pub fn submit(&mut self, data: I) -> Result<Arc<OutcomeSlot>, AdmissionClosed<I>> {
        let (outcome, observer) = observed();
        let root = self.checkout(data, Some(observer))?;
        self.publish(root, |source, root| source.offer(root));
        Ok(outcome)
    }

    /// Like [`run_seeded`](Self::run_seeded), and reports how the chain ended.
    pub fn submit_seeded(
        &mut self,
        seed: u64,
        data: I,
    ) -> Result<Arc<OutcomeSlot>, AdmissionClosed<I>> {
        let (outcome, observer) = observed();
        let root = self.checkout(data, Some(observer))?;
        self.publish(root, |source, root| source.offer_seeded(seed, root));
        Ok(outcome)
    }

    /// Like [`run_on`](Self::run_on), and reports how the chain ended.
    ///
    /// # Panics
    ///
    /// If `partition` is not below the partition count; no chain is checked out then.
    pub fn submit_on(
        &mut self,
        partition: usize,
        data: I,
    ) -> Result<Arc<OutcomeSlot>, AdmissionClosed<I>> {
        self.check_partition(partition);
        let (outcome, observer) = observed();
        let root = self.checkout(data, Some(observer))?;
        self.publish(root, |source, root| source.offer_to(partition, root));
        Ok(outcome)
    }

    fn check_partition(&self, partition: usize) {
        assert!(
            partition < self.partitions(),
            "partition {partition} is out of range for {} partitions",
            self.partitions()
        );
    }

    /// A chain for `data`, counted in, with its completion and `observer` installed.
    fn checkout(
        &mut self,
        data: I,
        observer: Option<Observer>,
    ) -> Result<Box<PipelineFrame<I>>, AdmissionClosed<I>> {
        if !self.shared.admit() {
            return Err(AdmissionClosed(data));
        }
        let mut root = self.manager.get_or_create(data);
        root.on_completion(self.completion.clone());
        if let Some(observer) = observer {
            root.observe(observer);
        }
        Ok(root)
    }

    /// Queues `root` with `offer`, cancelling it if the runner has completed.
    fn publish(
        &self,
        root: Box<PipelineFrame<I>>,
        offer: impl FnOnce(&QueueSource, Box<dyn Frame>) -> Result<(), Box<dyn Frame>>,
    ) {
        if let Err(root) = offer(&self.shared.source, root) {
            root.do_finally_with_error(FrameError::Cancelled);
        }
    }
}

impl<I> PipelineRunner<I> {
    /// A handle that completes this runner from any thread.
    pub fn control(&self) -> PipelineControl {
        PipelineControl {
            shared: self.shared.clone(),
        }
    }

    /// Stops admission, and completes the runner once every accepted chain has finished.
    pub fn complete_gracefully(&self) {
        self.shared.complete_gracefully();
    }

    /// Queued chains, give or take those being published or drained.
    pub fn size(&self) -> u64 {
        self.shared.source.size()
    }

    /// Demand asked for and not yet delivered.
    pub fn demand(&self) -> u64 {
        self.shared.source.demand()
    }
}

impl<I> IngestSink for PipelineRunner<I> {
    fn delegate(&self) -> Arc<dyn LatticeSource> {
        self.shared.source.clone()
    }

    /// Kills every chain: queued ones end cancelled at once, running ones when their stage
    /// returns. Admission closes.
    fn complete(&self) {
        self.shared.kill();
    }

    fn is_complete(&self) -> bool {
        self.shared.source.is_complete()
    }
}

/// Completes a [`PipelineRunner`] from any thread.
#[derive(Clone)]
pub struct PipelineControl {
    shared: Arc<Shared>,
}

impl PipelineControl {
    /// See [`PipelineRunner::complete_gracefully`].
    pub fn complete_gracefully(&self) {
        self.shared.complete_gracefully();
    }

    /// See [`IngestSink::complete`] on [`PipelineRunner`].
    pub fn complete(&self) {
        self.shared.kill();
    }

    pub fn is_complete(&self) -> bool {
        self.shared.source.is_complete()
    }
}

#[cfg(test)]
mod tests;
