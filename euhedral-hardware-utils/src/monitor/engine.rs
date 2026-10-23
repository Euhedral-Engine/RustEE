//! Turns raw provider samples into resolved interval samples (upstream `SampleStateEngine` and
//! `SlowSampleCache`).
//!
//! Owned by one monitor and used under its evaluation lock, so nothing here is synchronized.

use super::sampling::{
    CounterDelta, CounterSignal, CpuIntervalSignals, CpuSlowIntervalSignals, FastHardwareSample,
    IntervalHardwareSample, IoIntervalSignals, LatencyInterval, MemoryIntervalSignals, Resolved,
    Signal, SignalResolution, SignalValidity, SignalValue, SlowHardwareSample,
    SystemSlowIntervalSignals, ThermalSeverity,
};
use crate::util::CpuSet;

/// Slow samples are attempted on an anchored 5 s grid.
pub(crate) const SLOW_PERIOD_NS: i64 = 5_000_000_000;
/// A slow sample stays usable for 15 s.
pub(crate) const SLOW_TTL_NS: i64 = 15_000_000_000;

/// The anchored slow-sample grid and the last good slow sample.
#[derive(Debug, Default)]
pub(crate) struct SlowSampleCache {
    next_attempt_ns: Option<i64>,
    last: Option<SlowHardwareSample>,
    refreshed: bool,
}

impl SlowSampleCache {
    /// Due on the first call and once `poll_start_ns` reaches the next grid point. A failed attempt
    /// does not move the grid, so a failure is retried on the next tick.
    pub(crate) fn is_due(&self, poll_start_ns: i64) -> bool {
        self.next_attempt_ns
            .is_none_or(|next| poll_start_ns.wrapping_sub(next) >= 0)
    }

    /// Stores a successful sample and moves the grid to the first point strictly after
    /// `poll_start_ns`, skipping missed points instead of catching up.
    pub(crate) fn anchor_and_store(&mut self, poll_start_ns: i64, sample: SlowHardwareSample) {
        self.next_attempt_ns = Some(match self.next_attempt_ns {
            None => poll_start_ns.wrapping_add(SLOW_PERIOD_NS),
            Some(next) => {
                let behind = poll_start_ns.wrapping_sub(next);
                if behind >= 0 {
                    next.wrapping_add((behind / SLOW_PERIOD_NS + 1) * SLOW_PERIOD_NS)
                } else {
                    next
                }
            }
        });
        self.last = Some(sample);
        self.refreshed = true;
    }

    /// The cached sample when its age is within the TTL, with `Current` for the first resolution
    /// after it was stored and `Cached` after that; an expired or future-dated sample is dropped.
    pub(crate) fn resolve(
        &mut self,
        evaluation_ns: i64,
    ) -> Option<(&SlowHardwareSample, SignalResolution)> {
        let age = evaluation_ns.wrapping_sub(self.last.as_ref()?.observed_at_ns);
        if !(0..=SLOW_TTL_NS).contains(&age) {
            self.last = None;
        }
        let resolution = if std::mem::take(&mut self.refreshed) {
            SignalResolution::Current
        } else {
            SignalResolution::Cached
        };
        self.last.as_ref().map(|sample| (sample, resolution))
    }

    pub(crate) fn clear(&mut self) {
        *self = Self::default();
    }
}

/// A cumulative counter's baseline and its last interval.
#[derive(Clone, Copy, Debug, Default)]
struct CounterState {
    baseline: Option<(u64, i64)>,
    last_interval: Option<(u64, i64)>,
}

impl CounterState {
    /// - no baseline: store it, `Baseline`;
    /// - a reading not newer than the baseline, or smaller (reset or wrap): re-baseline, `Baseline`;
    /// - otherwise the increment, `Current`;
    /// - unsupported: forget everything, `Unavailable`;
    /// - a transient failure: keep the baseline for the TTL and repeat the last interval as `Cached`,
    ///   so a single failed read holds pressure;
    /// - a reading dated after the sample that carries it is inconsistent: `Unavailable`.
    fn evaluate(
        &mut self,
        signal: &CounterSignal,
        outer_observed_at_ns: i64,
        evaluation_ns: i64,
        ttl_ns: i64,
    ) -> CounterDelta {
        match signal.validity() {
            SignalValidity::Unsupported => {
                *self = Self::default();
                return CounterDelta::unavailable();
            }
            SignalValidity::TransientFailure => {
                let Some((_, at)) = self.baseline else {
                    return CounterDelta::unavailable();
                };
                if !(0..=ttl_ns).contains(&evaluation_ns.wrapping_sub(at)) {
                    *self = Self::default();
                    return CounterDelta::unavailable();
                }
                return match self.last_interval {
                    Some((delta, elapsed)) => {
                        CounterDelta::interval(delta, elapsed, at, SignalResolution::Cached)
                            .unwrap_or_else(CounterDelta::unavailable)
                    }
                    None => CounterDelta::unavailable(),
                };
            }
            SignalValidity::Valid => {}
        }
        let (value, at) = (signal.value(), signal.observed_at_ns());
        if at.wrapping_sub(outer_observed_at_ns) > 0 {
            return CounterDelta::unavailable();
        }
        let previous = self.baseline.replace((value, at));
        match previous {
            Some((prior, prior_at)) if at.wrapping_sub(prior_at) > 0 && value >= prior => {
                let (delta, elapsed) = (value - prior, at.wrapping_sub(prior_at));
                self.last_interval = Some((delta, elapsed));
                CounterDelta::interval(delta, elapsed, at, SignalResolution::Current)
                    .expect("elapsed is positive")
            }
            _ => {
                self.last_interval = None;
                CounterDelta::baseline(at)
            }
        }
    }

    /// Forgets the baseline so the next reading starts a fresh interval.
    fn rebase(&mut self) {
        *self = Self::default();
    }
}

/// A gauge's last valid value.
#[derive(Clone, Copy, Debug)]
struct GaugeState<T> {
    value: Option<(T, i64)>,
    refreshed: bool,
}

impl<T> Default for GaugeState<T> {
    fn default() -> Self {
        Self {
            value: None,
            refreshed: false,
        }
    }
}

impl<T: SignalValue> GaugeState<T> {
    /// Only a strictly newer valid reading replaces the value; a transient failure keeps it and an
    /// unsupported signal clears it.
    fn update(&mut self, signal: &Signal<T>) {
        self.refreshed = false;
        match signal.validity() {
            SignalValidity::Unsupported => self.value = None,
            SignalValidity::TransientFailure => {}
            SignalValidity::Valid => {
                let newer = self
                    .value
                    .is_none_or(|(_, at)| signal.observed_at_ns().wrapping_sub(at) > 0);
                if newer {
                    self.value = Some((signal.value(), signal.observed_at_ns()));
                    self.refreshed = true;
                }
            }
        }
    }

    /// `Current` when refreshed by this evaluation's sample, `Cached` while within the TTL, then
    /// `Unavailable` (and forgotten).
    fn resolve(&mut self, evaluation_ns: i64, ttl_ns: i64) -> Resolved<T> {
        let Some((value, at)) = self.value else {
            return Resolved::unavailable();
        };
        if !(0..=ttl_ns).contains(&evaluation_ns.wrapping_sub(at)) {
            self.value = None;
            return Resolved::unavailable();
        }
        let resolution = if self.refreshed {
            SignalResolution::Current
        } else {
            SignalResolution::Cached
        };
        Resolved::new(value, at, resolution)
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct CpuState {
    wait: CounterState,
    psi: CounterState,
    reported: GaugeState<f64>,
    throttle: CounterState,
    steal: CounterState,
    external: GaugeState<f64>,
    runnable: GaugeState<f64>,
}

#[derive(Debug, Default)]
struct ScopeState {
    productive: CounterState,
    throttled: CounterState,
    wait: CounterState,
    psi: CounterState,
    reported: GaugeState<f64>,
    quota_capacity: GaugeState<f64>,
    quota_period: GaugeState<u64>,
    mem_hard: GaugeState<u64>,
    mem_high: GaugeState<u64>,
    mem_usage: GaugeState<u64>,
    mem_inactive: GaugeState<u64>,
    mem_reclaim: CounterState,
    mem_stall: CounterState,
    io_bytes: CounterState,
    io_stall: CounterState,
    io_latency: CounterState,
    io_ops: CounterState,
    io_queue: GaugeState<f64>,
}

/// Per-monitor sample state (upstream `SampleStateEngine`).
#[derive(Debug)]
pub(crate) struct SampleStateEngine {
    logical_span: usize,
    fast_ttl_ns: i64,
    scope: ScopeState,
    cpus: Vec<CpuState>,
    slow: SlowSampleCache,
    last_evaluation_ns: Option<i64>,
    membership: Option<(CpuSet, i64)>,
}

impl SampleStateEngine {
    /// `logical_span` must be positive. The fast TTL is `5 * fast_period_ns`, kept within
    /// `[1 s, 30 s]`.
    pub(crate) fn new(logical_span: usize, fast_period_ns: i64) -> Self {
        assert!(logical_span > 0, "logical_span must be positive");
        Self {
            logical_span,
            fast_ttl_ns: fast_period_ns
                .saturating_mul(5)
                .clamp(1_000_000_000, 30_000_000_000),
            scope: ScopeState::default(),
            cpus: vec![CpuState::default(); logical_span],
            slow: SlowSampleCache::default(),
            last_evaluation_ns: None,
            membership: None,
        }
    }

    #[cfg(test)]
    pub(crate) fn fast_ttl_ns(&self) -> i64 {
        self.fast_ttl_ns
    }

    pub(crate) fn is_slow_due(&self, poll_start_ns: i64) -> bool {
        self.slow.is_due(poll_start_ns)
    }

    /// Stores a successful slow sample; a malformed or missing one is ignored and retried.
    pub(crate) fn process_slow(&mut self, poll_start_ns: i64, sample: Option<SlowHardwareSample>) {
        let Some(sample) = sample else { return };
        match sample.check_shape() {
            Ok(()) if sample.logical_span == self.logical_span => {
                self.slow.anchor_and_store(poll_start_ns, sample)
            }
            Ok(()) => log::debug!(
                "discarding slow sample spanning {} CPUs; the monitor spans {}",
                sample.logical_span,
                self.logical_span
            ),
            Err(error) => log::debug!("discarding malformed slow sample: {error}"),
        }
    }

    /// Forgets every counter baseline and gauge but keeps the slow cache and membership. The monitor
    /// calls this when it restarts after a stop, so the first interval after a restart does not span
    /// the stopped period.
    pub(crate) fn reset_counter_state(&mut self) {
        self.last_evaluation_ns = None;
        self.scope = ScopeState::default();
        self.cpus.fill(CpuState::default());
    }

    /// Forgets everything, including the slow cache and membership.
    pub(crate) fn clear_all(&mut self) {
        self.reset_counter_state();
        self.membership = None;
        self.slow.clear();
    }

    /// Resolves one fast sample at `evaluation_ns`, or returns `None` when nothing should be
    /// published:
    /// - `evaluation_ns` repeats or precedes the previous evaluation (everything is cleared);
    /// - the sample is `None` (a complete failure; membership and baselines are kept);
    /// - the sample is dated after `evaluation_ns` or is malformed.
    ///
    /// The first sample sets the membership; only a strictly newer sample replaces it. A CPU leaving
    /// the membership loses its per-CPU state immediately.
    pub(crate) fn process_fast(
        &mut self,
        evaluation_ns: i64,
        sample: Option<&FastHardwareSample>,
    ) -> Option<IntervalHardwareSample> {
        if let Some(previous) = self.last_evaluation_ns
            && evaluation_ns.wrapping_sub(previous) <= 0
        {
            self.clear_all();
            return None;
        }
        self.last_evaluation_ns = Some(evaluation_ns);

        let sample = sample?;
        if let Err(error) = sample.check_shape() {
            log::debug!("discarding malformed fast sample: {error}");
            return None;
        }
        if sample.logical_span != self.logical_span {
            log::debug!(
                "discarding fast sample spanning {} CPUs; the monitor spans {}",
                sample.logical_span,
                self.logical_span
            );
            return None;
        }
        let outer = sample.observed_at_ns;
        if outer.wrapping_sub(evaluation_ns) > 0 {
            return None;
        }
        let newer = self
            .membership
            .as_ref()
            .is_none_or(|(_, at)| outer.wrapping_sub(*at) > 0);
        if newer {
            let span = CpuSet::range(0, self.logical_span);
            self.membership = Some((sample.effective_cpus.intersection(&span), outer));
        }
        let membership = self
            .membership
            .as_ref()
            .expect("membership is set")
            .0
            .clone();
        let ttl = self.fast_ttl_ns;
        let scope = &mut self.scope;
        let counter = |state: &mut CounterState, signal: &CounterSignal| {
            state.evaluate(signal, outer, evaluation_ns, ttl)
        };

        scope.quota_capacity.update(&sample.quota_capacity_cpus);
        scope.quota_period.update(&sample.quota_period_ns);
        let productive = counter(&mut scope.productive, &sample.productive_cpu_ns);
        let throttled = counter(&mut scope.throttled, &sample.scope_quota_throttled_ns);
        let wait = counter(&mut scope.wait, &sample.scope_scheduler_wait_ns);
        let psi = counter(&mut scope.psi, &sample.scope_psi_stall_ns);
        scope
            .reported
            .update(&sample.scope_reported_scheduler_stall_ratio.ratio());

        let memory = &sample.memory_signals;
        scope.mem_hard.update(&memory.hard_limit_bytes);
        scope.mem_high.update(&memory.high_limit_bytes);
        scope.mem_usage.update(&memory.usage_bytes);
        scope.mem_inactive.update(&memory.inactive_file_bytes);
        let reclaim = counter(&mut scope.mem_reclaim, &memory.cumulative_reclaim_bytes);
        let mem_stall = counter(&mut scope.mem_stall, &memory.memory_stall_ns);

        let io = &sample.io_signals;
        let io_bytes = counter(&mut scope.io_bytes, &io.productive_bytes);
        let io_stall = counter(&mut scope.io_stall, &io.stall_ns);
        let io_latency = counter(&mut scope.io_latency, &io.operation_latency_ns);
        let io_ops = counter(&mut scope.io_ops, &io.completed_operations);
        scope.io_queue.update(&io.maximum_queue_depth);
        // When the pair is unusable and the two baselines have diverged, re-baseline both so neither
        // carries a baseline the other lacks. Two fresh baselines taken together are kept.
        let latency = LatencyInterval::new(io_latency, io_ops).unwrap_or_else(|| {
            let at = |state: &CounterState| state.baseline.map(|(_, at)| at);
            if at(&scope.io_latency) != at(&scope.io_ops) {
                scope.io_latency.rebase();
                scope.io_ops.rebase();
            }
            LatencyInterval::unavailable()
        });

        let mut cpu_signals = Vec::with_capacity(self.logical_span);
        for (cpu, state) in self.cpus.iter_mut().enumerate() {
            let fast = sample
                .cpu_signals
                .get(cpu)
                .filter(|_| membership.contains(cpu));
            let Some(fast) = fast else {
                *state = CpuState::default();
                cpu_signals.push(CpuIntervalSignals::unavailable());
                continue;
            };
            let scheduler_wait = counter(&mut state.wait, &fast.scheduler_wait);
            let psi_stall = counter(&mut state.psi, &fast.psi_stall);
            state
                .reported
                .update(&fast.reported_scheduler_stall_ratio.ratio());
            let quota_throttle = counter(&mut state.throttle, &fast.quota_throttle);
            let steal = counter(&mut state.steal, &fast.steal);
            state
                .external
                .update(&fast.external_contention_ratio.ratio());
            state.runnable.update(&fast.runnable_per_capacity);
            cpu_signals.push(CpuIntervalSignals {
                scheduler_wait,
                psi_stall,
                reported_scheduler_stall_ratio: state.reported.resolve(evaluation_ns, ttl),
                quota_throttle,
                steal,
                external_contention_ratio: state.external.resolve(evaluation_ns, ttl),
                runnable_per_capacity: state.runnable.resolve(evaluation_ns, ttl),
            });
        }

        let (cpu_slow_signals, system_slow_signals) = self.resolve_slow(evaluation_ns, &membership);
        let scope = &mut self.scope;
        Some(IntervalHardwareSample {
            observed_at_ns: evaluation_ns,
            logical_span: self.logical_span,
            effective_cpus: membership,
            quota_capacity_cpus: scope.quota_capacity.resolve(evaluation_ns, ttl),
            quota_period_ns: scope.quota_period.resolve(evaluation_ns, ttl),
            productive_cpu_ns: productive,
            scope_quota_throttled_ns: throttled,
            scope_scheduler_wait_ns: wait,
            scope_psi_stall_ns: psi,
            scope_reported_scheduler_stall_ratio: scope.reported.resolve(evaluation_ns, ttl),
            cpu_signals,
            memory_signals: MemoryIntervalSignals {
                hard_limit_bytes: scope.mem_hard.resolve(evaluation_ns, ttl),
                high_limit_bytes: scope.mem_high.resolve(evaluation_ns, ttl),
                usage_bytes: scope.mem_usage.resolve(evaluation_ns, ttl),
                inactive_file_bytes: scope.mem_inactive.resolve(evaluation_ns, ttl),
                cumulative_reclaim_bytes: reclaim,
                memory_stall_ns: mem_stall,
            },
            io_signals: IoIntervalSignals {
                productive_bytes: io_bytes,
                stall_ns: io_stall,
                operations_latency: latency,
                maximum_queue_depth: scope.io_queue.resolve(evaluation_ns, ttl),
            },
            cpu_slow_signals,
            system_slow_signals,
        })
    }

    fn resolve_slow(
        &mut self,
        evaluation_ns: i64,
        membership: &CpuSet,
    ) -> (Vec<CpuSlowIntervalSignals>, SystemSlowIntervalSignals) {
        let span = self.logical_span;
        let Some((slow, resolution)) = self.slow.resolve(evaluation_ns) else {
            return (
                vec![CpuSlowIntervalSignals::unavailable(); span],
                SystemSlowIntervalSignals::unavailable(),
            );
        };
        let at = slow.observed_at_ns;
        let double = |signal: &Signal<f64>| resolve_slow_signal(signal, at, resolution);
        let long = |signal: &Signal<u64>| resolve_slow_signal(signal, at, resolution);
        let thermal = |signal: &Signal<ThermalSeverity>| signal.get().unwrap_or_default();
        let low_power = |signal: &Signal<bool>| signal.get().unwrap_or(false);

        let system = &slow.system_signals;
        let system_signals = SystemSlowIntervalSignals {
            available_capacity_units: double(&system.available_capacity_units),
            nominal_capacity_units: double(&system.nominal_capacity_units),
            thermal_severity: thermal(&system.thermal_severity),
            low_power_mode: low_power(&system.low_power_mode),
            observed_at_ns: at,
            resolution,
        };
        let cpu_signals = (0..span)
            .map(|cpu| {
                match slow
                    .cpu_signals
                    .get(cpu)
                    .filter(|_| membership.contains(cpu))
                {
                    Some(signals) => CpuSlowIntervalSignals {
                        available_capacity_units: double(&signals.available_capacity_units),
                        nominal_capacity_units: double(&signals.nominal_capacity_units),
                        constrained_frequency_hz: long(&signals.constrained_frequency_hz),
                        nominal_frequency_hz: long(&signals.nominal_frequency_hz),
                        thermal_severity: thermal(&signals.thermal_severity),
                        low_power_mode: low_power(&signals.low_power_mode),
                        observed_at_ns: at,
                        resolution,
                    },
                    None => CpuSlowIntervalSignals::unavailable(),
                }
            })
            .collect();
        (cpu_signals, system_signals)
    }
}

fn resolve_slow_signal<T: SignalValue>(
    signal: &Signal<T>,
    at: i64,
    resolution: SignalResolution,
) -> Resolved<T> {
    if signal.is_valid() {
        Resolved::new(signal.value(), at, resolution)
    } else {
        Resolved::unavailable()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::monitor::sampling::{CpuSlowSignals, SystemSlowSignals};

    const PERIOD: i64 = 100_000_000;

    fn fast(
        at: i64,
        effective: &[usize],
        cpu_usage: u64,
        mem_usage: u64,
        io_ops: u64,
        io_latency: u64,
    ) -> FastHardwareSample {
        let mut sample =
            FastHardwareSample::unsupported(at, 2, effective.iter().copied().collect());
        sample.quota_capacity_cpus = Signal::valid(4.0, at);
        sample.productive_cpu_ns = Signal::valid(cpu_usage, at);
        sample.memory_signals.usage_bytes = Signal::valid(mem_usage, at);
        sample.io_signals.operation_latency_ns = Signal::valid(io_latency, at);
        sample.io_signals.completed_operations = Signal::valid(io_ops, at);
        sample
    }

    fn slow(at: i64) -> SlowHardwareSample {
        SlowHardwareSample::unsupported(at, 1)
    }

    #[test]
    fn slow_grid_anchors_and_skips_to_the_first_future_point() {
        let mut cache = SlowSampleCache::default();
        assert!(cache.is_due(1_000));
        cache.anchor_and_store(1_000, slow(1_000));
        assert!(!cache.is_due(1_000 + 4_999_999_999));
        assert!(cache.is_due(1_000 + 5_000_000_000));
        cache.anchor_and_store(1_000 + 6_000_000_000, slow(1_000 + 6_000_000_000));
        assert!(!cache.is_due(1_000 + 9_999_999_999));
        assert!(cache.is_due(1_000 + 10_000_000_000));
    }

    #[test]
    fn slow_samples_expire_after_the_ttl_or_when_dated_in_the_future() {
        let mut cache = SlowSampleCache::default();
        cache.anchor_and_store(1_000, slow(1_000));
        assert!(cache.resolve(1_000 + SLOW_TTL_NS).is_some());
        assert!(cache.resolve(1_000 + SLOW_TTL_NS + 1).is_none());
        cache.anchor_and_store(1_000, slow(1_000));
        assert!(cache.resolve(999).is_none());
        cache.anchor_and_store(1_000, slow(1_000));
        cache.clear();
        assert!(cache.resolve(1_000).is_none());
        assert!(cache.is_due(5_000));
    }

    #[test]
    fn a_grid_anchored_at_zero_stays_anchored() {
        let mut cache = SlowSampleCache::default();
        cache.anchor_and_store(-SLOW_PERIOD_NS, slow(-SLOW_PERIOD_NS));
        assert!(!cache.is_due(-1));
        assert!(cache.is_due(0));
    }

    #[test]
    fn counters_baseline_then_measure_then_rebaseline_on_reset() {
        let mut engine = SampleStateEngine::new(2, PERIOD);
        let first = engine
            .process_fast(1_100, Some(&fast(1_000, &[0], 100, 500, 10, 50)))
            .unwrap();
        assert_eq!(
            first.productive_cpu_ns.resolution(),
            SignalResolution::Baseline
        );
        let second = engine
            .process_fast(2_100, Some(&fast(2_000, &[0], 150, 500, 15, 70)))
            .unwrap();
        assert_eq!(
            second.productive_cpu_ns.resolution(),
            SignalResolution::Current
        );
        assert_eq!(
            (
                second.productive_cpu_ns.delta(),
                second.productive_cpu_ns.elapsed_ns()
            ),
            (50, 1_000)
        );
        assert_eq!(
            second.io_signals.operations_latency.mean_latency_ns(),
            Some(4.0)
        );
        let third = engine
            .process_fast(3_100, Some(&fast(3_000, &[0], 120, 500, 18, 80)))
            .unwrap();
        assert_eq!(
            third.productive_cpu_ns.resolution(),
            SignalResolution::Baseline
        );
    }

    #[test]
    fn gauges_refresh_then_expire_after_the_fast_ttl() {
        let mut engine = SampleStateEngine::new(2, PERIOD);
        let ttl = engine.fast_ttl_ns();
        assert_eq!(ttl, 1_000_000_000);
        let first = engine
            .process_fast(1_100, Some(&fast(1_000, &[0], 100, 500, 10, 50)))
            .unwrap();
        assert_eq!(
            first.memory_signals.usage_bytes.resolution(),
            SignalResolution::Current
        );
        assert_eq!(first.memory_signals.usage_bytes.value(), 500);
        assert_eq!(first.quota_capacity_cpus.get(), Some(4.0));

        assert!(engine.process_fast(1_000 + ttl + 200, None).is_none());

        let mut failing = fast(1_000 + ttl + 300, &[0], 0, 0, 0, 0);
        failing.memory_signals.usage_bytes = Signal::transient_failure(1_000 + ttl + 300);
        let expired = engine
            .process_fast(1_000 + ttl + 400, Some(&failing))
            .unwrap();
        assert_eq!(
            expired.memory_signals.usage_bytes.resolution(),
            SignalResolution::Unavailable
        );
    }

    #[test]
    fn a_transient_failure_holds_the_gauge_as_cached() {
        let mut engine = SampleStateEngine::new(2, PERIOD);
        engine
            .process_fast(1_100, Some(&fast(1_000, &[0], 100, 500, 10, 50)))
            .unwrap();
        let mut failing = fast(2_000, &[0], 150, 0, 15, 70);
        failing.memory_signals.usage_bytes = Signal::transient_failure(2_000);
        let held = engine.process_fast(2_100, Some(&failing)).unwrap();
        assert_eq!(
            held.memory_signals.usage_bytes.resolution(),
            SignalResolution::Cached
        );
        assert_eq!(held.memory_signals.usage_bytes.value(), 500);
    }

    #[test]
    fn a_transient_counter_failure_repeats_the_last_interval() {
        let mut engine = SampleStateEngine::new(2, PERIOD);
        engine
            .process_fast(1_100, Some(&fast(1_000, &[0], 100, 0, 0, 0)))
            .unwrap();
        engine
            .process_fast(2_100, Some(&fast(2_000, &[0], 150, 0, 0, 0)))
            .unwrap();
        let mut failing = fast(3_000, &[0], 0, 0, 0, 0);
        failing.productive_cpu_ns = Signal::transient_failure(3_000);
        let held = engine.process_fast(3_100, Some(&failing)).unwrap();
        assert_eq!(
            held.productive_cpu_ns.resolution(),
            SignalResolution::Cached
        );
        assert_eq!(
            (
                held.productive_cpu_ns.delta(),
                held.productive_cpu_ns.elapsed_ns()
            ),
            (50, 1_000)
        );
        // The baseline survived, so recovery measures from the last good reading.
        let recovered = engine
            .process_fast(4_100, Some(&fast(4_000, &[0], 250, 0, 0, 0)))
            .unwrap();
        assert_eq!(
            (
                recovered.productive_cpu_ns.delta(),
                recovered.productive_cpu_ns.elapsed_ns()
            ),
            (100, 2_000)
        );
        let mut unsupported = fast(5_000, &[0], 0, 0, 0, 0);
        unsupported.productive_cpu_ns = Signal::unsupported(5_000);
        let gone = engine.process_fast(5_100, Some(&unsupported)).unwrap();
        assert_eq!(
            gone.productive_cpu_ns.resolution(),
            SignalResolution::Unavailable
        );
    }

    #[test]
    fn leaving_the_membership_clears_per_cpu_state() {
        let mut engine = SampleStateEngine::new(2, PERIOD);
        let mut both = fast(1_000, &[0, 1], 100, 500, 10, 50);
        both.cpu_signals[1].scheduler_wait = Signal::valid(5, 1_000);
        engine.process_fast(1_010, Some(&both));
        let mut one = fast(2_000, &[0], 150, 500, 15, 70);
        one.cpu_signals[1].scheduler_wait = Signal::valid(9, 2_000);
        let result = engine.process_fast(2_010, Some(&one)).unwrap();
        assert_eq!(
            result.cpu_signals[1].scheduler_wait.resolution(),
            SignalResolution::Unavailable
        );
        assert_eq!(result.effective_cpus, [0].into_iter().collect());
        // Back in the membership, CPU 1 starts from a fresh baseline.
        let mut again = fast(3_000, &[0, 1], 150, 500, 15, 70);
        again.cpu_signals[1].scheduler_wait = Signal::valid(12, 3_000);
        let result = engine.process_fast(3_010, Some(&again)).unwrap();
        assert_eq!(
            result.cpu_signals[1].scheduler_wait.resolution(),
            SignalResolution::Baseline
        );
    }

    #[test]
    fn stale_samples_do_not_remap_membership() {
        let mut engine = SampleStateEngine::new(2, PERIOD);
        engine.process_fast(2_010, Some(&fast(2_000, &[0, 1], 0, 0, 0, 0)));
        let stale = engine
            .process_fast(3_010, Some(&fast(1_000, &[0], 0, 0, 0, 0)))
            .unwrap();
        assert_eq!(stale.effective_cpus, CpuSet::range(0, 2));
    }

    #[test]
    fn regressing_time_and_future_samples_publish_nothing() {
        let mut engine = SampleStateEngine::new(2, PERIOD);
        let sample = fast(1_000, &[0], 100, 500, 10, 50);
        assert!(engine.process_fast(1_100, Some(&sample)).is_some());
        assert!(engine.process_fast(1_050, Some(&sample)).is_none());
        let future = fast(5_000, &[0], 100, 500, 10, 50);
        assert!(engine.process_fast(4_000, Some(&future)).is_none());
        let mut malformed = fast(6_000, &[0], 0, 0, 0, 0);
        malformed.cpu_signals.pop();
        assert!(engine.process_fast(6_100, Some(&malformed)).is_none());
        let wider = FastHardwareSample::unsupported(7_000, 3, CpuSet::range(0, 3));
        assert!(engine.process_fast(7_100, Some(&wider)).is_none());
    }

    #[test]
    fn reset_counter_state_rebaselines_but_keeps_the_slow_sample() {
        let mut engine = SampleStateEngine::new(1, PERIOD);
        engine.process_slow(
            1_000,
            Some(SlowHardwareSample {
                observed_at_ns: 1_000,
                logical_span: 1,
                cpu_signals: vec![CpuSlowSignals {
                    constrained_frequency_hz: Signal::valid(2_000, 1_000),
                    nominal_frequency_hz: Signal::valid(4_000, 1_000),
                    ..CpuSlowSignals::unsupported(1_000)
                }],
                system_signals: SystemSlowSignals {
                    thermal_severity: Signal::valid(ThermalSeverity::Serious, 1_000),
                    ..SystemSlowSignals::unsupported(1_000)
                },
            }),
        );
        let mut sample = FastHardwareSample::unsupported(1_000, 1, CpuSet::range(0, 1));
        sample.productive_cpu_ns = Signal::valid(1, 1_000);
        engine.process_fast(1_100, Some(&sample)).unwrap();
        sample.observed_at_ns = 2_000;
        sample.productive_cpu_ns = Signal::valid(5, 2_000);
        assert_eq!(
            engine
                .process_fast(2_100, Some(&sample))
                .unwrap()
                .productive_cpu_ns
                .delta(),
            4
        );

        engine.reset_counter_state();
        sample.observed_at_ns = 3_000;
        sample.productive_cpu_ns = Signal::valid(9, 3_000);
        let after = engine.process_fast(3_100, Some(&sample)).unwrap();
        assert_eq!(
            after.productive_cpu_ns.resolution(),
            SignalResolution::Baseline
        );
        assert_eq!(
            after.cpu_slow_signals[0].nominal_frequency_hz.get(),
            Some(4_000)
        );
        assert_eq!(
            after.system_slow_signals.thermal_severity,
            ThermalSeverity::Serious
        );
        // Time may now restart below the old evaluation clock without being called a regression.
        assert!(engine.process_fast(3_200, Some(&sample)).is_some());

        engine.clear_all();
        assert!(engine.is_slow_due(3_300));
    }
}
