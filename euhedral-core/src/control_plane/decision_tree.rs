//! The fragment worker's policy: whether to sit out a cycle, whether to run frames straight from
//! upstream or stage them in the cache first, and how large a batch to run.

use super::participation;
use crate::config::IdlePolicy;
use crate::flow_control::UpstreamQueue;
use crate::utils::ACQUIRE_CONTENTION_SCALE;
use crate::utils::calibration::benchmark;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Contention, scaled by [`ACQUIRE_CONTENTION_SCALE`], above which costly bodies are staged: 85%.
const CONTENTION_THRESHOLD: u64 = 850_000;

/// [`benchmark`] cycles as costly as the cheapest body worth confirming as expensive. A tuning knob.
pub(super) const EXPENSIVE_BODY_WEIGHT: u32 = 288;
/// [`benchmark`] cycles as costly as the dearest body still run straight from upstream under
/// contention. A tuning knob.
pub(super) const DIRECT_EXECUTION_BODY_WEIGHT: u32 = 272;

/// The work a direct batch aims for.
const DIRECT_BATCH_WORK_TARGET_NS: f64 = 250_000.0;
/// The work a staged batch aims for.
const STAGED_BATCH_WORK_TARGET_NS: f64 = 8_000_000.0;

/// Body-cost samples per estimate.
const WINDOW: usize = 32;
/// Consecutive windows that must find bodies expensive before the estimate says so.
const EXPENSIVE_CONFIRMATION_WINDOWS: u32 = 2;

/// How a fragment worker serves frames in a cycle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ExecutionPath {
    /// Sit the cycle out and park.
    Idle,
    /// Run frames straight from upstream.
    Direct,
    /// Stage frames in the cache, then run them from there.
    Staged,
}

/// An estimate of what one frame body costs, from timed samples.
///
/// Each window of 32 samples yields its second-smallest, which replaces the estimate at once if it
/// is cheap, and only after two windows in a row if it is expensive. The output hooks record
/// samples on the worker's thread; the policy reads the estimate.
///
/// The window is plain state of that one worker, held in relaxed atomics so the hooks and the
/// policy can share it through an `Arc`: `record`, `calibrate` and `reset` run on the worker, in
/// turn. The count and the estimate are what other threads may read, and keep their release
/// stores.
pub(super) struct BodyCost {
    /// The window's samples, as `f64` bits.
    samples: [AtomicU64; WINDOW],
    /// The cost, as `f64` bits, from which a body counts as expensive.
    expensive_threshold_ns: AtomicU64,
    confirmations: AtomicU32,
    estimate_ns: AtomicU64,
    recorded: AtomicU64,
}

impl BodyCost {
    /// An empty estimate that takes no body as expensive until [`calibrate`](Self::calibrate)d.
    pub(super) fn new() -> Self {
        Self {
            samples: std::array::from_fn(|_| AtomicU64::new(0f64.to_bits())),
            expensive_threshold_ns: AtomicU64::new(f64::INFINITY.to_bits()),
            confirmations: AtomicU32::new(0),
            estimate_ns: AtomicU64::new(0f64.to_bits()),
            recorded: AtomicU64::new(0),
        }
    }

    /// Takes bodies costing at least `expensive_threshold` as expensive, and starts over.
    pub(super) fn calibrate(&self, expensive_threshold: Duration) {
        self.expensive_threshold_ns.store(
            (expensive_threshold.as_nanos() as f64).to_bits(),
            Ordering::Relaxed,
        );
        self.reset();
    }

    /// Records one timed body.
    pub(super) fn record(&self, elapsed: Duration) {
        if elapsed.is_zero() {
            return;
        }
        let recorded = self.recorded.load(Ordering::Relaxed);
        let index = (recorded % WINDOW as u64) as usize;
        self.samples[index].store((elapsed.as_nanos() as f64).to_bits(), Ordering::Relaxed);
        self.recorded
            .store(recorded.saturating_add(1), Ordering::Release);
        if index == WINDOW - 1 {
            self.update();
        }
    }

    /// The estimated body cost in nanoseconds; zero before the first full window.
    pub(super) fn estimate_ns(&self) -> f64 {
        f64::from_bits(self.estimate_ns.load(Ordering::Acquire))
    }

    /// Whether a full window has been recorded.
    pub(super) fn has_history(&self) -> bool {
        self.recorded.load(Ordering::Acquire) >= WINDOW as u64
    }

    /// Forgets every sample and the estimate, keeping the calibration.
    pub(super) fn reset(&self) {
        self.confirmations.store(0, Ordering::Relaxed);
        self.recorded.store(0, Ordering::Release);
        self.estimate_ns.store(0f64.to_bits(), Ordering::Release);
    }

    fn update(&self) {
        let (mut minimum, mut second) = (f64::INFINITY, f64::INFINITY);
        for sample in &self.samples {
            let sample = f64::from_bits(sample.load(Ordering::Relaxed));
            if sample < minimum {
                second = minimum;
                minimum = sample;
            } else if sample < second {
                second = sample;
            }
        }
        let threshold = f64::from_bits(self.expensive_threshold_ns.load(Ordering::Relaxed));
        if second >= threshold {
            let confirmations = (self.confirmations.load(Ordering::Relaxed) + 1)
                .min(EXPENSIVE_CONFIRMATION_WINDOWS);
            self.confirmations.store(confirmations, Ordering::Relaxed);
            if confirmations < EXPENSIVE_CONFIRMATION_WINDOWS {
                return;
            }
        } else {
            self.confirmations.store(0, Ordering::Relaxed);
        }
        self.estimate_ns.store(second.to_bits(), Ordering::Release);
    }
}

/// The policy one fragment worker runs by. Only that worker uses it.
pub(super) struct DecisionTree {
    idle_policy: IdlePolicy,
    direct_threshold_ns: f64,
    body_cost: Arc<BodyCost>,
    path: ExecutionPath,
    batch_size: u64,
    service_time_ns: f64,
}

impl DecisionTree {
    /// A policy with thresholds measured on the calling thread's CPU: call it from the worker.
    pub(super) fn calibrated(idle_policy: IdlePolicy, body_cost: Arc<BodyCost>) -> Self {
        let expensive = benchmark(EXPENSIVE_BODY_WEIGHT);
        let direct = benchmark(DIRECT_EXECUTION_BODY_WEIGHT);
        Self::with_thresholds(idle_policy, body_cost, expensive, direct)
    }

    /// A policy that takes bodies costing at least `expensive` as expensive, and runs bodies costing
    /// at most `direct` straight from upstream under contention. Resets `body_cost`.
    pub(super) fn with_thresholds(
        idle_policy: IdlePolicy,
        body_cost: Arc<BodyCost>,
        expensive: Duration,
        direct: Duration,
    ) -> Self {
        body_cost.calibrate(expensive);
        Self {
            idle_policy,
            direct_threshold_ns: direct.as_nanos() as f64,
            body_cost,
            path: ExecutionPath::Direct,
            batch_size: 2,
            service_time_ns: 0.0,
        }
    }

    /// Whether the worker of `rank` should sit this cycle out. Never before a body-cost estimate,
    /// for the first-ranked or only worker, or while there is a productive handle per worker;
    /// always when no handle is productive; otherwise as the participation model says.
    pub(super) fn should_idle(
        &self,
        contention: u64,
        productive_handles: u64,
        registered_workers: usize,
        rank: Option<usize>,
    ) -> bool {
        let Some(rank) = rank else {
            return false;
        };
        if rank <= 1
            || registered_workers <= 1
            || !self.body_cost.has_history()
            || plentiful(productive_handles, registered_workers)
        {
            return false;
        }
        if productive_handles == 0 {
            return true;
        }
        participation::should_idle(
            rank,
            productive_handles,
            registered_workers,
            self.body_cost.estimate_ns(),
            contention as f64 / ACQUIRE_CONTENTION_SCALE as f64,
        )
    }

    /// How long an idle worker parks, as the idle policy's function picks it from the queue's
    /// contention at `now`. Also installs the half-life the function picks in the queue.
    pub(super) fn idle_park(
        &self,
        queue: &mut UpstreamQueue,
        now: Instant,
        registered_workers: usize,
        productive_handles: u64,
    ) -> Duration {
        let initial = self.idle_policy.contention_half_life();
        let contention =
            queue.adaptive_contention(now, initial) as f64 / ACQUIRE_CONTENTION_SCALE as f64;
        let productive_ratio = if registered_workers > 0 {
            productive_handles as f64 / registered_workers as f64
        } else {
            f64::NAN
        };
        let body = self.body_cost.estimate_ns();
        let function = self.idle_policy.function();
        let park = function.park(
            contention,
            productive_ratio,
            body,
            self.idle_policy.idle_park(),
        );
        let half_life = function.half_life(contention, productive_ratio, body, initial);
        let half_life = if half_life.is_zero() {
            initial
        } else {
            half_life
        };
        queue.install_contention_half_life(now, half_life, initial);
        park
    }

    /// Runs frames straight from upstream when there is nothing to stage from, before a body-cost
    /// estimate, for a lone worker, while there is a productive handle per worker, or when cheap
    /// bodies meet moderate contention. Stages them otherwise.
    pub(super) fn select_path(
        &mut self,
        productive_handles: u64,
        upstream_handles: u64,
        registered_workers: usize,
        contention: u64,
    ) -> ExecutionPath {
        let direct = upstream_handles == 0
            || registered_workers <= 1
            || !self.body_cost.has_history()
            || plentiful(productive_handles, registered_workers)
            || (contention <= CONTENTION_THRESHOLD
                && self.body_cost.estimate_ns() <= self.direct_threshold_ns);
        self.path = if direct {
            ExecutionPath::Direct
        } else {
            ExecutionPath::Staged
        };
        self.path
    }

    /// Records that `frames` frames took `elapsed` to run, into the service-time average.
    pub(super) fn record_execution(&mut self, elapsed: Duration, frames: u64) {
        if elapsed.is_zero() || frames == 0 {
            return;
        }
        let sample = elapsed.as_nanos() as f64 / frames as f64;
        if self.service_time_ns == 0.0 {
            self.service_time_ns = sample;
        } else {
            self.service_time_ns += (sample - self.service_time_ns) / 8.0;
        }
    }

    /// Ends a batch and returns the next batch size: the size that fills the current path's work
    /// target at the measured service time, as a power of two, moved at most a halving or a
    /// doubling from the current size, and within `[2, eligible_cap]`.
    pub(super) fn complete_batch(&mut self, eligible_cap: u64) -> u64 {
        let cap = eligible_cap.max(2);
        let desired = self.desired_batch_size().min(cap);
        let minimum = self.batch_size.div_ceil(2);
        let maximum = self.batch_size.saturating_mul(2);
        self.batch_size = desired.clamp(minimum, maximum).min(cap);
        self.batch_size
    }

    fn desired_batch_size(&self) -> u64 {
        if self.service_time_ns <= 0.0 {
            return self.batch_size;
        }
        let target = match self.path {
            ExecutionPath::Direct | ExecutionPath::Idle => DIRECT_BATCH_WORK_TARGET_NS,
            ExecutionPath::Staged => STAGED_BATCH_WORK_TARGET_NS,
        };
        let raw = (target / self.service_time_ns.max(1.0)).floor() as u64;
        1 << raw.max(2).ilog2()
    }

    /// Back to the direct path, a batch of two, and no timing history.
    pub(super) fn reset(&mut self) {
        self.path = ExecutionPath::Direct;
        self.batch_size = 2;
        self.service_time_ns = 0.0;
        self.body_cost.reset();
    }

    #[cfg(test)]
    pub(super) fn batch_size(&self) -> u64 {
        self.batch_size
    }

    /// The service-time average in nanoseconds per frame; zero before the first sample.
    #[cfg(test)]
    pub(super) fn service_time_ns(&self) -> f64 {
        self.service_time_ns
    }

    pub(super) fn idle_policy(&self) -> &IdlePolicy {
        &self.idle_policy
    }
}

/// Whether there is a productive handle per worker.
fn plentiful(productive_handles: u64, registered_workers: usize) -> bool {
    productive_handles >= registered_workers as u64
}

#[cfg(test)]
mod tests;
