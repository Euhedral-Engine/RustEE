use super::IngestSink;
use super::queue::{DEFAULT_CHUNK, Ending, Owner, QueueSource};
use crate::frames::{Completion, Frame, FrameError, Outcome, PipelineBuilder, PipelineFrame};
use crate::generics::{FrameSink, LatticeSource};
use crate::impls::FrameManager;
use std::error::Error;
use std::fmt;
use std::num::NonZeroUsize;
use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
#[allow(clippy::disallowed_types, reason = "HOT-FIX(C14)")]
use std::sync::mpsc::{self, Receiver};
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

/// A channel that hears how a chain ended, and the observer that tells it.
#[allow(clippy::disallowed_types, reason = "HOT-FIX(C14)")]
fn observed() -> (Receiver<Outcome>, Observer) {
    let (sender, outcome) = mpsc::sync_channel(1);
    let observer = Box::new(move |result| {
        let _ = sender.send(result);
    });
    (outcome, observer)
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
    #[allow(clippy::disallowed_types, reason = "HOT-FIX(C14)")]
    pub fn submit(&mut self, data: I) -> Result<Receiver<Outcome>, AdmissionClosed<I>> {
        let (outcome, observer) = observed();
        let root = self.checkout(data, Some(observer))?;
        self.publish(root, |source, root| source.offer(root));
        Ok(outcome)
    }

    /// Like [`run_seeded`](Self::run_seeded), and reports how the chain ended.
    #[allow(clippy::disallowed_types, reason = "HOT-FIX(C14)")]
    pub fn submit_seeded(
        &mut self,
        seed: u64,
        data: I,
    ) -> Result<Receiver<Outcome>, AdmissionClosed<I>> {
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
    #[allow(clippy::disallowed_types, reason = "HOT-FIX(C14)")]
    pub fn submit_on(
        &mut self,
        partition: usize,
        data: I,
    ) -> Result<Receiver<Outcome>, AdmissionClosed<I>> {
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
