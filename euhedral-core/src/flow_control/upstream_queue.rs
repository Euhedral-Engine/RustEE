use crate::frames::Frame;
use crate::utils::math::unsigned_multiply_high;
use crate::utils::{ACQUIRE_CONTENTION_SCALE, AverageFlow, FlowContext};
use euhedral_data_structures::atomics::PaddedAtomicLong;
use euhedral_data_structures::queues::{Consumer, MpscQueue, Multi, Producer, Single};
use euhedral_hashing::hasher_api;
use std::cell::Cell;
use std::f64::consts::LN_2;
use std::num::NonZeroU64;
use std::sync::Arc;
use std::time::{Duration, Instant};

thread_local! {
    static ROUTED_PUSHES: Cell<u64> = const { Cell::new(0) };
}

/// Records that a frame was routed downstream on this thread, so a request serviced on this
/// thread can tell that its source produced work synchronously.
pub(crate) fn note_routed_push() {
    ROUTED_PUSHES.with(|pushes| pushes.set(pushes.get().wrapping_add(1)));
}

/// What a request can observe about its source producing work on this thread: the frames
/// delivered against the thread's [`FlowContext`] request, and the frames routed on the thread.
fn request_evidence() -> (u64, u64) {
    let satisfied =
        FlowContext::with(|context| context.map_or(0, |context| context.satisfied_request()));
    (satisfied, ROUTED_PUSHES.with(Cell::get))
}

/// An upstream source as an [`UpstreamQueue`] sees it.
///
/// One handle sits in the queue of every worker that may service it. A worker services a handle
/// only while it holds the handle's lock, so a handle is serviced by one worker at a time.
pub trait UpstreamHandle: Send + Sync {
    /// The handle's registration order, which [`UpstreamQueue::reset_for_next_trial`] restores.
    fn sequence(&self) -> u64;

    /// Hands up to `demand` frames to `consumer`, stopping before the first frame `stop` matches,
    /// and returns how many. `None` means the handle did not look at its source, so the pull says
    /// nothing about whether the source has work.
    fn pull(
        &self,
        consumer: &mut dyn FnMut(Box<dyn Frame>),
        stop: &mut dyn FnMut(&dyn Frame) -> bool,
        demand: u64,
    ) -> Option<u64>;

    /// Passes demand for `demand` frames on to the source.
    fn request(&self, demand: u64);

    /// A complete handle is dropped from every queue that comes across it.
    fn is_complete(&self) -> bool;

    /// Takes the handle's lock, or returns false if another worker holds it. The default never
    /// contends.
    fn try_acquire(&self) -> bool {
        true
    }

    /// Releases the lock taken by [`try_acquire`](Self::try_acquire).
    fn release(&self) {}
}

/// A handle in one worker's queue, with that worker's latest view of whether the handle produces
/// work. A handle starts out productive until the worker services it.
pub(crate) struct Entry {
    handle: Arc<dyn UpstreamHandle>,
    productive: bool,
}

impl Entry {
    pub(crate) fn new(handle: Arc<dyn UpstreamHandle>) -> Self {
        Self {
            handle,
            productive: true,
        }
    }
}

/// How [`UpstreamQueue`] turns a demand into a number of handles to split it across.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum PullBucketDivisionMode {
    /// `demand / target` buckets.
    #[default]
    Floor,
    /// `ceil(demand / target)` buckets.
    Ceil,
}

/// One worker's view of the upstream handles feeding a lattice, and the scheduler that spreads
/// the worker's demand across them.
///
/// Each worker owns its queue, so scheduling needs no coordination beyond each handle's lock. A
/// demand is split into buckets, each bucket goes to the next handle round-robin, and handles
/// return to the queue in a shuffled order so that workers do not all visit them in step.
///
/// The queue also keeps the worker's acquisition contention: the share of handles it found locked
/// by another worker, as a fixed-point EWMA scaled by [`ACQUIRE_CONTENTION_SCALE`].
pub struct UpstreamQueue {
    cpu: Option<usize>,
    handles: Consumer<Entry, Single>,
    requeue: Producer<Entry, Multi>,
    shuffle: Vec<Entry>,
    shuffle_window: usize,
    seed: u64,
    /// A handle whose last pull stopped at a frame that has to be routed. It is kept out of the
    /// queue and serviced first by the next request, so the frame does not wait behind every other
    /// handle.
    pending_request: Option<Entry>,

    upstream_count: Arc<PaddedAtomicLong>,
    cached_up_count: u64,
    nonproductive_count: u64,
    pull_bucket_target: NonZeroU64,
    pull_bucket_division_mode: PullBucketDivisionMode,

    acquire_contention: AverageFlow,
    contention_evidence_count: u64,
    applied_evidence_count: u64,
    last_contention_evidence: Option<Instant>,
    decay_exposure: f64,
    decay_timestamp: Option<Instant>,
    active_half_life: Option<Duration>,

    diagnostics_enabled: bool,
    contention_observation_count: u64,
    last_raw_contention: Option<u64>,
    last_contention_observation: Option<Instant>,
    successful_acquisition_count: u64,
    failed_acquisition_count: u64,
    total_acquisition_attempts: u64,
}

impl UpstreamQueue {
    /// The pull bucket target workers start with.
    pub const DEFAULT_PULL_BUCKET_TARGET: NonZeroU64 = NonZeroU64::new(2048).unwrap();

    /// A queue consuming `handles`, whose live handle count is `upstream_count`. Handles are
    /// shuffled back into the queue in batches of up to `shuffle_window`.
    ///
    /// # Panics
    ///
    /// Panics if the consumer of `handles` is already taken.
    pub(crate) fn new(
        cpu: Option<usize>,
        handles: &MpscQueue<Entry>,
        upstream_count: Arc<PaddedAtomicLong>,
        shuffle_window: usize,
    ) -> Self {
        let shuffle_window = shuffle_window.max(1);
        Self {
            cpu,
            handles: handles.consumer(),
            requeue: handles.producer(),
            shuffle: Vec::with_capacity(shuffle_window),
            shuffle_window,
            seed: crate::internal::random::next_u64(),
            pending_request: None,
            upstream_count,
            cached_up_count: 0,
            nonproductive_count: 0,
            pull_bucket_target: Self::DEFAULT_PULL_BUCKET_TARGET,
            pull_bucket_division_mode: PullBucketDivisionMode::Floor,
            acquire_contention: AverageFlow::new(),
            contention_evidence_count: 0,
            applied_evidence_count: 0,
            last_contention_evidence: None,
            decay_exposure: 0.0,
            decay_timestamp: None,
            active_half_life: None,
            diagnostics_enabled: false,
            contention_observation_count: 0,
            last_raw_contention: None,
            last_contention_observation: None,
            successful_acquisition_count: 0,
            failed_acquisition_count: 0,
            total_acquisition_attempts: 0,
        }
    }

    /// The CPU the worker was on when the queue was made, if known.
    pub fn cpu(&self) -> Option<usize> {
        self.cpu
    }

    /// The live handle count as of the last refresh, refreshing it first if it was zero.
    pub fn cached_upstream_count(&mut self) -> u64 {
        if self.cached_up_count == 0 {
            return self.true_upstream_count();
        }
        self.cached_up_count
    }

    /// Refreshes and returns the live handle count.
    pub fn true_upstream_count(&mut self) -> u64 {
        self.cached_up_count = self.upstream_count.get_acquire().max(0) as u64;
        self.cached_up_count
    }

    /// The acquisition-contention EWMA, or `None` before the first pull that tried to acquire a
    /// handle.
    pub fn acquire_contention(&self) -> Option<u64> {
        self.acquire_contention.value()
    }

    /// [`acquire_contention`](Self::acquire_contention) as a fraction in `0.0..=1.0`, for reporting.
    pub fn normalized_acquire_contention(&self) -> Option<f64> {
        self.acquire_contention
            .value()
            .map(|value| value as f64 / ACQUIRE_CONTENTION_SCALE as f64)
    }

    /// The contention EWMA decayed by the age of its evidence: halved for every `half_life` since
    /// the queue first saw the latest evidence. Zero without evidence.
    ///
    /// Evidence is dated when a query first sees it, not when the pull recorded it.
    ///
    /// # Panics
    ///
    /// Panics if `half_life` is zero.
    pub fn effective_contention(&mut self, now: Instant, half_life: Duration) -> u64 {
        assert_half_life(half_life);
        if self.contention_evidence_count == 0 {
            return 0;
        }
        self.date_new_evidence(now);
        let evidence = self.last_contention_evidence.unwrap_or(now);
        decay_contention(
            self.acquire_contention.value().unwrap_or(0),
            now.saturating_duration_since(evidence),
            half_life,
        )
    }

    /// Like [`effective_contention`](Self::effective_contention), but with a half-life that may
    /// change over time: each stretch of time decays the estimate by the half-life in force during
    /// it. `initial_half_life` applies until
    /// [`install_contention_half_life`](Self::install_contention_half_life) replaces it, and again
    /// after a reset.
    ///
    /// # Panics
    ///
    /// Panics if `initial_half_life` is zero.
    pub fn adaptive_contention(&mut self, now: Instant, initial_half_life: Duration) -> u64 {
        assert_half_life(initial_half_life);
        let half_life = *self.active_half_life.get_or_insert(initial_half_life);
        if self.contention_evidence_count == 0 {
            return 0;
        }
        if !self.date_new_evidence(now) {
            let since = self.decay_timestamp.unwrap_or(now);
            if let Some(elapsed) = now.checked_duration_since(since)
                && !elapsed.is_zero()
            {
                self.decay_exposure += elapsed.as_nanos() as f64 / half_life.as_nanos() as f64;
                self.decay_timestamp = Some(now);
            }
        }
        let stored = self
            .acquire_contention
            .value()
            .unwrap_or(0)
            .min(ACQUIRE_CONTENTION_SCALE);
        (stored as f64 * (-LN_2 * self.decay_exposure).exp()).round() as u64
    }

    /// Switches the [adaptive](Self::adaptive_contention) half-life to `half_life` from `now` on,
    /// after decaying the time so far under the half-life it replaces.
    ///
    /// # Panics
    ///
    /// Panics if either half-life is zero.
    pub fn install_contention_half_life(
        &mut self,
        now: Instant,
        half_life: Duration,
        initial_half_life: Duration,
    ) {
        assert_half_life(half_life);
        self.adaptive_contention(now, initial_half_life);
        self.active_half_life = Some(half_life);
    }

    /// Dates evidence recorded since the last query at `now`, restarting its decay. Returns
    /// whether there was any.
    fn date_new_evidence(&mut self, now: Instant) -> bool {
        if self.contention_evidence_count == self.applied_evidence_count {
            return false;
        }
        self.applied_evidence_count = self.contention_evidence_count;
        self.last_contention_evidence = Some(now);
        self.decay_timestamp = Some(now);
        self.decay_exposure = 0.0;
        true
    }

    /// Clears the contention history and diagnostics.
    pub fn reset_acquire_contention(&mut self) {
        self.acquire_contention.reset();
        self.decay_exposure = 0.0;
        self.decay_timestamp = None;
        self.active_half_life = None;
        self.contention_evidence_count = 0;
        self.applied_evidence_count = 0;
        self.last_contention_evidence = None;
        self.contention_observation_count = 0;
        self.last_raw_contention = None;
        self.last_contention_observation = None;
        self.successful_acquisition_count = 0;
        self.failed_acquisition_count = 0;
        self.total_acquisition_attempts = 0;
    }

    /// Restores the state a fresh worker would have, for the next run of a benchmark: clears the
    /// contention history, forgets which handles were unproductive, and puts the live handles back
    /// in registration order. Meant for a drained lattice.
    pub fn reset_for_next_trial(&mut self) {
        self.reset_acquire_contention();
        let mut live = Vec::new();
        live.append(&mut self.shuffle);
        live.extend(self.pending_request.take());
        for _ in 0..self.handles.size_long() {
            live.extend(self.handles.poll());
        }
        live.retain(|entry| !entry.handle.is_complete());
        live.sort_by_key(|entry| entry.handle.sequence());
        for mut entry in live {
            entry.productive = true;
            self.offer(entry);
        }
        self.true_upstream_count();
        self.nonproductive_count = 0;
    }

    /// Turns acquisition diagnostics on or off, clearing the contention history either way.
    pub fn set_acquire_diagnostics_enabled(&mut self, enabled: bool) {
        self.diagnostics_enabled = enabled;
        self.reset_acquire_contention();
    }

    /// Live handles minus those this worker last found without work. Handles the worker has not
    /// serviced yet count as productive.
    pub fn productive_handle_count(&mut self) -> u64 {
        self.true_upstream_count();
        self.remove_completed_handles();
        self.cached_up_count - self.cached_up_count.min(self.nonproductive_count)
    }

    /// Sets how demand is split into buckets. Meant for a drained lattice.
    pub fn set_pull_bucket_treatment(&mut self, target: NonZeroU64, mode: PullBucketDivisionMode) {
        self.pull_bucket_target = target;
        self.pull_bucket_division_mode = mode;
    }

    /// Passes demand for `demand` frames on to the handles, which push what they produce.
    pub fn request(&mut self, demand: u64) {
        self.service(None, demand);
    }

    /// Takes up to `demand` frames straight from the handles into `consumer`, without requesting
    /// more, and returns how many. Each handle's pull stops before the first frame `stop` matches;
    /// the next [`request`](Self::request) services that handle first, so the frame is routed.
    pub fn pull(
        &mut self,
        consumer: &mut dyn FnMut(Box<dyn Frame>),
        stop: &mut dyn FnMut(&dyn Frame) -> bool,
        demand: u64,
    ) -> u64 {
        self.service(Some(Pull { consumer, stop }), demand)
    }

    fn service(&mut self, mut pull: Option<Pull<'_>>, demand: u64) -> u64 {
        self.true_upstream_count();
        if demand == 0 || self.cached_up_count == 0 {
            return 0;
        }

        let bucket = self.pull_buckets(demand);
        let mut attempts = 0;
        let mut failed = 0;
        let mut total = 0;
        let mut limit = demand;
        let mut cycles = 0;
        while cycles < self.cached_up_count && limit > 0 {
            let preferred = if pull.is_none() {
                self.pending_request.take()
            } else {
                None
            };
            let Some(entry) = preferred.or_else(|| self.handles.poll()) else {
                if self.shuffle.is_empty() {
                    cycles += 1;
                } else {
                    self.flush_shuffle();
                }
                continue;
            };
            if entry.handle.is_complete() {
                self.observe_removal(&entry);
                continue;
            }

            attempts += 1;
            if !entry.handle.try_acquire() {
                failed += 1;
                self.buffer_for_requeue(entry);
                cycles += 1;
                continue;
            }

            let request = limit.min(bucket);
            limit -= request;
            total += Acquired::new(self, entry).service(pull.as_mut(), request);
            cycles = 0;
        }
        self.record_acquisition_cycle(attempts, failed);
        self.flush_shuffle();
        total
    }

    fn record_acquisition_cycle(&mut self, attempts: u64, failed: u64) {
        if attempts == 0 {
            return;
        }
        let raw = scale_acquire_contention(failed, attempts);
        self.acquire_contention
            .record(raw)
            .expect("a scaled fraction is within the contention scale");
        self.contention_evidence_count += 1;
        if !self.diagnostics_enabled {
            return;
        }
        self.contention_observation_count += 1;
        self.last_raw_contention = Some(raw);
        self.last_contention_observation = Some(Instant::now());
        self.successful_acquisition_count += attempts - failed;
        self.failed_acquisition_count += failed;
        self.total_acquisition_attempts += attempts;
    }

    /// Sets the productivity this worker observed for `entry`, keeping the unproductive count in
    /// step.
    fn reclassify(&mut self, entry: &mut Entry, productive: bool) {
        if entry.productive && !productive {
            self.nonproductive_count += 1;
        } else if !entry.productive && productive {
            self.nonproductive_count = self.nonproductive_count.saturating_sub(1);
        }
        entry.productive = productive;
    }

    /// Holds `entry` back for the next request, returning any handle held before to the queue.
    fn hold_for_request(&mut self, entry: Entry) {
        if let Some(previous) = self.pending_request.replace(entry) {
            self.buffer_for_requeue(previous);
        }
    }

    /// Returns a serviced or contended handle to the queue through the shuffle window.
    fn buffer_for_requeue(&mut self, entry: Entry) {
        if self.shuffle.len() == self.shuffle_window {
            self.flush_shuffle();
        }
        self.shuffle.push(entry);
    }

    fn flush_shuffle(&mut self) {
        while self.shuffle.len() > 1 {
            self.seed = hasher_api::mix(self.seed.wrapping_add(1));
            let index = unsigned_multiply_high(self.seed, self.shuffle.len() as u64) as usize;
            let entry = self.shuffle.swap_remove(index);
            self.offer(entry);
        }
        if let Some(entry) = self.shuffle.pop() {
            self.offer(entry);
        }
    }

    fn offer(&self, entry: Entry) {
        if self.requeue.offer(entry).is_err() {
            unreachable!("upstream handle queues are unbounded");
        }
    }

    /// Drops complete handles when the queue holds more entries than there are live handles.
    fn remove_completed_handles(&mut self) {
        self.flush_shuffle();
        if let Some(entry) = self
            .pending_request
            .take_if(|entry| entry.handle.is_complete())
        {
            self.observe_removal(&entry);
        }
        let held = u64::from(self.pending_request.is_some());
        let mut queued = self.handles.size_long();
        let mut surplus = (queued + held).saturating_sub(self.cached_up_count);
        while queued > 0 && surplus > 0 {
            let Some(entry) = self.handles.poll() else {
                return;
            };
            queued -= 1;
            if entry.handle.is_complete() {
                self.observe_removal(&entry);
                surplus -= 1;
            } else {
                self.offer(entry);
            }
        }
    }

    fn observe_removal(&mut self, entry: &Entry) {
        if !entry.productive {
            self.nonproductive_count = self.nonproductive_count.saturating_sub(1);
        }
    }

    /// The share of `demand` each handle is asked for: the demand split evenly into the number of
    /// buckets the division mode gives, kept between one and the number of handles.
    fn pull_buckets(&self, demand: u64) -> u64 {
        if demand == 0 || self.cached_up_count < 2 {
            return demand;
        }
        let target = self.pull_bucket_target.get();
        let needed = match self.pull_bucket_division_mode {
            PullBucketDivisionMode::Floor => demand / target,
            PullBucketDivisionMode::Ceil => 1 + (demand - 1) / target,
        };
        let buckets = needed.clamp(1, self.cached_up_count);
        1 + (demand - 1) / buckets
    }

    /// How many pulls recorded acquisition contention since the last reset.
    pub fn contention_evidence_count(&self) -> u64 {
        self.contention_evidence_count
    }

    /// When a contention query first saw the latest evidence.
    pub fn last_contention_evidence(&self) -> Option<Instant> {
        self.last_contention_evidence
    }

    /// Pulls recorded while diagnostics were on.
    pub fn contention_observation_count(&self) -> u64 {
        self.contention_observation_count
    }

    /// The last pull's contention sample, while diagnostics are on.
    pub fn last_raw_contention(&self) -> Option<u64> {
        self.last_raw_contention
    }

    /// When the last pull's contention was recorded, while diagnostics are on.
    pub fn last_contention_observation(&self) -> Option<Instant> {
        self.last_contention_observation
    }

    pub fn successful_acquisition_count(&self) -> u64 {
        self.successful_acquisition_count
    }

    pub fn failed_acquisition_count(&self) -> u64 {
        self.failed_acquisition_count
    }

    pub fn total_acquisition_attempts(&self) -> u64 {
        self.total_acquisition_attempts
    }
}

/// Where a pull hands its frames, and where each handle's pull stops.
struct Pull<'a> {
    consumer: &'a mut dyn FnMut(Box<dyn Frame>),
    stop: &'a mut dyn FnMut(&dyn Frame) -> bool,
}

/// A handle whose lock this worker holds. Dropping it releases the lock and returns the handle to
/// the queue, even if servicing the handle panicked.
struct Acquired<'a> {
    queue: &'a mut UpstreamQueue,
    entry: Option<Entry>,
    hold_for_request: bool,
}

impl<'a> Acquired<'a> {
    fn new(queue: &'a mut UpstreamQueue, entry: Entry) -> Self {
        Self {
            queue,
            entry: Some(entry),
            hold_for_request: false,
        }
    }

    /// Pulls or requests up to `demand` frames from the handle and records what that showed about
    /// its source.
    fn service(mut self, pull: Option<&mut Pull<'_>>, demand: u64) -> u64 {
        let Self {
            queue,
            entry,
            hold_for_request,
        } = &mut self;
        let entry = entry.as_mut().expect("the entry is held until dropped");
        let was_productive = entry.productive;
        let (pulled, productive) = match pull {
            Some(Pull { consumer, stop }) => {
                let mut stopped = false;
                let mut watch = |frame: &dyn Frame| {
                    let matched = stop(frame);
                    stopped |= matched;
                    matched
                };
                let result = entry.handle.pull(&mut **consumer, &mut watch, demand);
                *hold_for_request = stopped;
                let productive = match result {
                    Some(0) if !stopped => false,
                    Some(0) | None => was_productive,
                    Some(_) => true,
                };
                (result.unwrap_or(0), productive)
            }
            None => {
                let before = request_evidence();
                entry.handle.request(demand);
                (0, was_productive || request_evidence() != before)
            }
        };
        FlowContext::with(|context| {
            if let Some(context) = context {
                context.add_satisfied_pull(pulled);
            }
        });
        queue.reclassify(entry, productive);
        pulled
    }
}

impl Drop for Acquired<'_> {
    fn drop(&mut self) {
        let Some(entry) = self.entry.take() else {
            return;
        };
        entry.handle.release();
        if self.hold_for_request {
            self.queue.hold_for_request(entry);
        } else {
            self.queue.buffer_for_requeue(entry);
        }
    }
}

fn assert_half_life(half_life: Duration) {
    assert!(
        !half_life.is_zero(),
        "contention half-life must be positive"
    );
}

/// Decays a contention estimate by its age: halves it for every `half_life` of `age`. The estimate
/// is first clamped to [`ACQUIRE_CONTENTION_SCALE`].
///
/// # Panics
///
/// Panics if `half_life` is zero.
pub fn decay_contention(stored: u64, age: Duration, half_life: Duration) -> u64 {
    assert_half_life(half_life);
    let bounded = stored.min(ACQUIRE_CONTENTION_SCALE);
    if age.is_zero() || bounded == 0 {
        return bounded;
    }
    let multiplier = (-LN_2 * (age.as_nanos() as f64 / half_life.as_nanos() as f64)).exp();
    (bounded as f64 * multiplier).round() as u64
}

/// `failed / attempts` scaled by [`ACQUIRE_CONTENTION_SCALE`], rounded down.
fn scale_acquire_contention(failed: u64, attempts: u64) -> u64 {
    assert!(
        attempts > 0 && failed <= attempts,
        "acquisition counts need 0 <= failures <= attempts and attempts > 0"
    );
    (u128::from(failed) * u128::from(ACQUIRE_CONTENTION_SCALE) / u128::from(attempts)) as u64
}

#[cfg(test)]
mod tests;
