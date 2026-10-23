//! Typed hardware signals and the provider interfaces (upstream `monitor.sampling` and its `enums`,
//! `primitives`, `samples` and `signals` packages).
//!
//! A provider reports raw *signals*: a cumulative counter or a gauge, a timestamp, and a
//! [`SignalValidity`]. The monitor's state engine turns consecutive samples into *resolved* values
//! (counter deltas over a measured interval, gauges with a freshness TTL), each with a
//! [`SignalResolution`]. A failed or unsupported read is never a zero: it is a validity, and it
//! propagates as an unavailable resolution.
//!
//! Units are canonical: CPU and stall times in nanoseconds, memory and I/O in bytes, frequencies in
//! hertz, timestamps in monotonic nanoseconds (see [`crate::monitor::MonotonicClock`]).

use crate::util::CpuSet;

/// Whether a signal was read.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum SignalValidity {
    #[default]
    Valid,
    /// This read failed; a later read may succeed.
    TransientFailure,
    /// The platform cannot supply this signal. Set only by the provider, never inferred from one
    /// failure.
    Unsupported,
}

/// How fresh a resolved value is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SignalResolution {
    /// From a valid reading in this evaluation.
    Current,
    /// Retained from an earlier valid reading that is still within its TTL.
    Cached,
    /// A counter's first, reset or regressed reading: a new baseline, no interval yet.
    Baseline,
    /// Unsupported, expired or inconsistent; the value is the type's zero.
    Unavailable,
}

impl SignalResolution {
    /// `Current` or `Cached`.
    pub fn is_usable(self) -> bool {
        matches!(self, SignalResolution::Current | SignalResolution::Cached)
    }
}

/// Thermal state, from nominal to emergency.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ThermalSeverity {
    #[default]
    Nominal,
    Fair,
    Serious,
    Critical,
    Emergency,
}

/// A payload a [`Signal`] can carry.
pub trait SignalValue: Copy + Default + PartialEq {
    /// The canonical form of a reading, or `None` when the reading is not valid for the type (a
    /// negative, NaN or infinite gauge).
    fn canonical(self) -> Option<Self>;
}

impl SignalValue for u64 {
    fn canonical(self) -> Option<u64> {
        Some(self)
    }
}

impl SignalValue for f64 {
    fn canonical(self) -> Option<f64> {
        // `x + 0.0` turns -0.0 into +0.0.
        (self.is_finite() && self >= 0.0).then_some(self + 0.0)
    }
}

impl SignalValue for bool {
    fn canonical(self) -> Option<bool> {
        Some(self)
    }
}

impl SignalValue for ThermalSeverity {
    fn canonical(self) -> Option<ThermalSeverity> {
        Some(self)
    }
}

/// One raw reading (upstream `CounterSignal`, `LongGaugeSignal`, `DoubleGaugeSignal`,
/// `BooleanSignal` and `ThermalSignal`).
///
/// A signal that is not valid always carries the type's zero. A "valid" gauge that is negative or not
/// finite is recorded as a transient failure.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Signal<T: SignalValue> {
    value: T,
    observed_at_ns: i64,
    validity: SignalValidity,
}

/// A cumulative, never-decreasing counter (a decrease is read as a reset).
pub type CounterSignal = Signal<u64>;
/// An integral gauge such as bytes or hertz.
pub type LongGaugeSignal = Signal<u64>;
/// A real-valued, non-negative gauge such as a ratio or a CPU count.
pub type DoubleGaugeSignal = Signal<f64>;
pub type BooleanSignal = Signal<bool>;
pub type ThermalSignal = Signal<ThermalSeverity>;

impl<T: SignalValue> Signal<T> {
    pub fn new(value: T, observed_at_ns: i64, validity: SignalValidity) -> Self {
        match (validity, value.canonical()) {
            (SignalValidity::Valid, Some(value)) => Self {
                value,
                observed_at_ns,
                validity,
            },
            (SignalValidity::Valid, None) => Self::transient_failure(observed_at_ns),
            (validity, _) => Self {
                value: T::default(),
                observed_at_ns,
                validity,
            },
        }
    }

    pub fn valid(value: T, observed_at_ns: i64) -> Self {
        Self::new(value, observed_at_ns, SignalValidity::Valid)
    }

    pub fn transient_failure(observed_at_ns: i64) -> Self {
        Self {
            value: T::default(),
            observed_at_ns,
            validity: SignalValidity::TransientFailure,
        }
    }

    pub fn unsupported(observed_at_ns: i64) -> Self {
        Self {
            value: T::default(),
            observed_at_ns,
            validity: SignalValidity::Unsupported,
        }
    }

    /// The payload; the type's zero unless the signal is valid.
    pub fn value(&self) -> T {
        self.value
    }

    /// The payload when the signal is valid.
    pub fn get(&self) -> Option<T> {
        (self.validity == SignalValidity::Valid).then_some(self.value)
    }

    pub fn observed_at_ns(&self) -> i64 {
        self.observed_at_ns
    }

    pub fn validity(&self) -> SignalValidity {
        self.validity
    }

    pub fn is_valid(&self) -> bool {
        self.validity == SignalValidity::Valid
    }
}

impl Signal<f64> {
    /// A ratio that must lie in `[0, 1]`: a larger reading is a transient failure.
    pub(crate) fn ratio(self) -> Self {
        if self.is_valid() && self.value > 1.0 {
            Self::transient_failure(self.observed_at_ns)
        } else {
            self
        }
    }
}

/// A gauge resolved for one evaluation (upstream `ResolvedLong`, `ResolvedDouble`). The value is the
/// type's zero when unavailable. `Baseline` is not used for gauges.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Resolved<T> {
    value: T,
    observed_at_ns: i64,
    resolution: SignalResolution,
}

pub type ResolvedLong = Resolved<u64>;
pub type ResolvedDouble = Resolved<f64>;

impl<T: Copy + Default> Resolved<T> {
    pub fn new(value: T, observed_at_ns: i64, resolution: SignalResolution) -> Self {
        if resolution == SignalResolution::Unavailable {
            Self::unavailable()
        } else {
            Self {
                value,
                observed_at_ns,
                resolution,
            }
        }
    }

    pub fn unavailable() -> Self {
        Self {
            value: T::default(),
            observed_at_ns: 0,
            resolution: SignalResolution::Unavailable,
        }
    }

    pub fn value(&self) -> T {
        self.value
    }

    /// The value unless unavailable.
    pub fn get(&self) -> Option<T> {
        (self.resolution != SignalResolution::Unavailable).then_some(self.value)
    }

    pub fn observed_at_ns(&self) -> i64 {
        self.observed_at_ns
    }

    pub fn resolution(&self) -> SignalResolution {
        self.resolution
    }
}

/// A counter's increment over one measured interval.
///
/// - `Current`: the difference of two consecutive valid readings; `elapsed_ns > 0`.
/// - `Cached`: the last `Current` interval, repeated while the counter is transiently unreadable
///   within its TTL, so a single failed read holds pressure instead of dropping it.
/// - `Baseline`, `Unavailable`: no interval; delta and elapsed are zero.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CounterDelta {
    delta: u64,
    elapsed_ns: i64,
    observed_at_ns: i64,
    resolution: SignalResolution,
}

impl CounterDelta {
    /// `None` when `elapsed_ns` is not positive.
    pub fn interval(
        delta: u64,
        elapsed_ns: i64,
        observed_at_ns: i64,
        resolution: SignalResolution,
    ) -> Option<Self> {
        if !resolution.is_usable() {
            return Some(Self::without_interval(observed_at_ns, resolution));
        }
        (elapsed_ns > 0).then_some(Self {
            delta,
            elapsed_ns,
            observed_at_ns,
            resolution,
        })
    }

    pub fn baseline(observed_at_ns: i64) -> Self {
        Self::without_interval(observed_at_ns, SignalResolution::Baseline)
    }

    pub fn unavailable() -> Self {
        Self::without_interval(0, SignalResolution::Unavailable)
    }

    fn without_interval(observed_at_ns: i64, resolution: SignalResolution) -> Self {
        Self {
            delta: 0,
            elapsed_ns: 0,
            observed_at_ns,
            resolution,
        }
    }

    pub fn delta(&self) -> u64 {
        self.delta
    }

    pub fn elapsed_ns(&self) -> i64 {
        self.elapsed_ns
    }

    pub fn observed_at_ns(&self) -> i64 {
        self.observed_at_ns
    }

    pub fn resolution(&self) -> SignalResolution {
        self.resolution
    }

    /// `delta / elapsed` for a usable interval, otherwise `None`.
    pub fn rate(&self) -> Option<f64> {
        (self.resolution.is_usable() && self.elapsed_ns > 0)
            .then(|| self.delta as f64 / self.elapsed_ns as f64)
    }
}

/// Cumulative latency and completed operations over one shared interval. Usable only when both
/// counters produced an interval of the same length; otherwise both are re-baselined.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LatencyInterval {
    latency_delta_ns: u64,
    operations_delta: u64,
    elapsed_ns: i64,
    observed_at_ns: i64,
    resolution: SignalResolution,
}

impl LatencyInterval {
    pub fn new(latency: CounterDelta, operations: CounterDelta) -> Option<Self> {
        let paired = latency.resolution().is_usable()
            && operations.resolution().is_usable()
            && latency.elapsed_ns() == operations.elapsed_ns();
        paired.then_some(Self {
            latency_delta_ns: latency.delta(),
            operations_delta: operations.delta(),
            elapsed_ns: latency.elapsed_ns(),
            observed_at_ns: latency.observed_at_ns(),
            resolution: if latency.resolution() == SignalResolution::Current
                && operations.resolution() == SignalResolution::Current
            {
                SignalResolution::Current
            } else {
                SignalResolution::Cached
            },
        })
    }

    pub fn unavailable() -> Self {
        Self {
            latency_delta_ns: 0,
            operations_delta: 0,
            elapsed_ns: 0,
            observed_at_ns: 0,
            resolution: SignalResolution::Unavailable,
        }
    }

    pub fn latency_delta_ns(&self) -> u64 {
        self.latency_delta_ns
    }

    pub fn operations_delta(&self) -> u64 {
        self.operations_delta
    }

    pub fn elapsed_ns(&self) -> i64 {
        self.elapsed_ns
    }

    pub fn observed_at_ns(&self) -> i64 {
        self.observed_at_ns
    }

    pub fn resolution(&self) -> SignalResolution {
        self.resolution
    }

    /// Mean latency per completed operation, when usable and at least one operation completed.
    pub fn mean_latency_ns(&self) -> Option<f64> {
        (self.resolution.is_usable() && self.operations_delta > 0)
            .then(|| self.latency_delta_ns as f64 / self.operations_delta as f64)
    }
}

// ----- Fast signals -----

/// Per-logical-CPU fast signals.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CpuFastSignals {
    /// Cumulative time tasks spent runnable but waiting for this CPU, ns.
    pub scheduler_wait: CounterSignal,
    /// Cumulative PSI stall attributed to this CPU, ns.
    pub psi_stall: CounterSignal,
    /// A provider-computed stall ratio in `[0, 1]`.
    pub reported_scheduler_stall_ratio: DoubleGaugeSignal,
    /// Cumulative quota throttling attributed to this CPU, ns.
    pub quota_throttle: CounterSignal,
    /// Cumulative time stolen by a hypervisor, ns.
    pub steal: CounterSignal,
    /// Contention from outside the scope, ratio in `[0, 1]`.
    pub external_contention_ratio: DoubleGaugeSignal,
    /// Runnable tasks per unit of capacity (may exceed 1).
    pub runnable_per_capacity: DoubleGaugeSignal,
}

impl CpuFastSignals {
    /// Every signal unsupported.
    pub fn unsupported(at_ns: i64) -> Self {
        Self {
            scheduler_wait: Signal::unsupported(at_ns),
            psi_stall: Signal::unsupported(at_ns),
            reported_scheduler_stall_ratio: Signal::unsupported(at_ns),
            quota_throttle: Signal::unsupported(at_ns),
            steal: Signal::unsupported(at_ns),
            external_contention_ratio: Signal::unsupported(at_ns),
            runnable_per_capacity: Signal::unsupported(at_ns),
        }
    }
}

/// Memory fast signals for the sampling scope (cgroup, job object or host).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MemoryFastSignals {
    pub hard_limit_bytes: LongGaugeSignal,
    /// A soft limit (cgroup `memory.high`) at which the kernel starts reclaiming.
    pub high_limit_bytes: LongGaugeSignal,
    pub usage_bytes: LongGaugeSignal,
    /// Reclaimable page cache included in `usage_bytes`; the working set is `usage - inactive`.
    pub inactive_file_bytes: LongGaugeSignal,
    pub cumulative_reclaim_bytes: CounterSignal,
    /// Cumulative memory PSI stall, ns.
    pub memory_stall_ns: CounterSignal,
}

impl MemoryFastSignals {
    pub fn unsupported(at_ns: i64) -> Self {
        Self {
            hard_limit_bytes: Signal::unsupported(at_ns),
            high_limit_bytes: Signal::unsupported(at_ns),
            usage_bytes: Signal::unsupported(at_ns),
            inactive_file_bytes: Signal::unsupported(at_ns),
            cumulative_reclaim_bytes: Signal::unsupported(at_ns),
            memory_stall_ns: Signal::unsupported(at_ns),
        }
    }
}

/// I/O fast signals.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IoFastSignals {
    /// Cumulative bytes transferred; telemetry, never pressure on its own.
    pub productive_bytes: CounterSignal,
    /// Cumulative I/O PSI stall, ns.
    pub stall_ns: CounterSignal,
    /// Cumulative time spent on completed operations, ns.
    pub operation_latency_ns: CounterSignal,
    pub completed_operations: CounterSignal,
    /// The deepest in-flight queue among devices.
    pub maximum_queue_depth: DoubleGaugeSignal,
}

impl IoFastSignals {
    pub fn unsupported(at_ns: i64) -> Self {
        Self {
            productive_bytes: Signal::unsupported(at_ns),
            stall_ns: Signal::unsupported(at_ns),
            operation_latency_ns: Signal::unsupported(at_ns),
            completed_operations: Signal::unsupported(at_ns),
            maximum_queue_depth: Signal::unsupported(at_ns),
        }
    }
}

/// One fast-cadence reading of the whole scope.
///
/// `cpu_signals` has exactly `logical_span` entries, indexed by logical CPU id; entries for ids that
/// are not effective are ignored.
#[derive(Clone, Debug, PartialEq)]
pub struct FastHardwareSample {
    pub observed_at_ns: i64,
    pub logical_span: usize,
    pub effective_cpus: CpuSet,
    /// CPUs of capacity the scope may use. Fractional: a quota of half a CPU is 0.5.
    pub quota_capacity_cpus: DoubleGaugeSignal,
    pub quota_period_ns: LongGaugeSignal,
    /// Cumulative CPU time consumed by the scope, ns.
    pub productive_cpu_ns: CounterSignal,
    pub scope_quota_throttled_ns: CounterSignal,
    pub scope_scheduler_wait_ns: CounterSignal,
    /// Cumulative CPU PSI stall of the scope, ns.
    pub scope_psi_stall_ns: CounterSignal,
    pub scope_reported_scheduler_stall_ratio: DoubleGaugeSignal,
    pub cpu_signals: Vec<CpuFastSignals>,
    pub memory_signals: MemoryFastSignals,
    pub io_signals: IoFastSignals,
}

/// Why a provider sample was discarded.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SampleShapeError {
    ZeroSpan,
    EffectiveCpuOutOfSpan { cpu: usize, span: usize },
    CpuSignalCount { expected: usize, actual: usize },
}

impl std::fmt::Display for SampleShapeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SampleShapeError::ZeroSpan => f.write_str("logical span must be positive"),
            SampleShapeError::EffectiveCpuOutOfSpan { cpu, span } => {
                write!(f, "effective CPU {cpu} is outside the logical span {span}")
            }
            SampleShapeError::CpuSignalCount { expected, actual } => {
                write!(
                    f,
                    "expected {expected} per-CPU signal entries, got {actual}"
                )
            }
        }
    }
}

impl std::error::Error for SampleShapeError {}

impl FastHardwareSample {
    /// A sample whose per-CPU, memory and I/O signals are all unsupported.
    pub fn unsupported(observed_at_ns: i64, logical_span: usize, effective_cpus: CpuSet) -> Self {
        Self {
            observed_at_ns,
            logical_span,
            effective_cpus,
            quota_capacity_cpus: Signal::unsupported(observed_at_ns),
            quota_period_ns: Signal::unsupported(observed_at_ns),
            productive_cpu_ns: Signal::unsupported(observed_at_ns),
            scope_quota_throttled_ns: Signal::unsupported(observed_at_ns),
            scope_scheduler_wait_ns: Signal::unsupported(observed_at_ns),
            scope_psi_stall_ns: Signal::unsupported(observed_at_ns),
            scope_reported_scheduler_stall_ratio: Signal::unsupported(observed_at_ns),
            cpu_signals: vec![CpuFastSignals::unsupported(observed_at_ns); logical_span],
            memory_signals: MemoryFastSignals::unsupported(observed_at_ns),
            io_signals: IoFastSignals::unsupported(observed_at_ns),
        }
    }

    pub fn check_shape(&self) -> Result<(), SampleShapeError> {
        check_span(self.logical_span, self.cpu_signals.len())?;
        if let Some(cpu) = self
            .effective_cpus
            .last()
            .filter(|&cpu| cpu >= self.logical_span)
        {
            return Err(SampleShapeError::EffectiveCpuOutOfSpan {
                cpu,
                span: self.logical_span,
            });
        }
        Ok(())
    }
}

fn check_span(span: usize, entries: usize) -> Result<(), SampleShapeError> {
    if span == 0 {
        return Err(SampleShapeError::ZeroSpan);
    }
    if entries != span {
        return Err(SampleShapeError::CpuSignalCount {
            expected: span,
            actual: entries,
        });
    }
    Ok(())
}

// ----- Slow signals -----

/// Per-logical-CPU slow signals (capacity, frequency, thermal, power).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CpuSlowSignals {
    pub available_capacity_units: DoubleGaugeSignal,
    pub nominal_capacity_units: DoubleGaugeSignal,
    /// The highest frequency the CPU may currently run at (a policy or thermal cap), Hz.
    pub constrained_frequency_hz: LongGaugeSignal,
    pub nominal_frequency_hz: LongGaugeSignal,
    pub thermal_severity: ThermalSignal,
    pub low_power_mode: BooleanSignal,
}

impl CpuSlowSignals {
    pub fn unsupported(at_ns: i64) -> Self {
        Self {
            available_capacity_units: Signal::unsupported(at_ns),
            nominal_capacity_units: Signal::unsupported(at_ns),
            constrained_frequency_hz: Signal::unsupported(at_ns),
            nominal_frequency_hz: Signal::unsupported(at_ns),
            thermal_severity: Signal::unsupported(at_ns),
            low_power_mode: Signal::unsupported(at_ns),
        }
    }
}

/// System-wide slow signals.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SystemSlowSignals {
    pub available_capacity_units: DoubleGaugeSignal,
    pub nominal_capacity_units: DoubleGaugeSignal,
    pub thermal_severity: ThermalSignal,
    pub low_power_mode: BooleanSignal,
}

impl SystemSlowSignals {
    pub fn unsupported(at_ns: i64) -> Self {
        Self {
            available_capacity_units: Signal::unsupported(at_ns),
            nominal_capacity_units: Signal::unsupported(at_ns),
            thermal_severity: Signal::unsupported(at_ns),
            low_power_mode: Signal::unsupported(at_ns),
        }
    }
}

/// One slow-cadence reading. `cpu_signals` has exactly `logical_span` entries.
#[derive(Clone, Debug, PartialEq)]
pub struct SlowHardwareSample {
    pub observed_at_ns: i64,
    pub logical_span: usize,
    pub cpu_signals: Vec<CpuSlowSignals>,
    pub system_signals: SystemSlowSignals,
}

impl SlowHardwareSample {
    pub fn unsupported(observed_at_ns: i64, logical_span: usize) -> Self {
        Self {
            observed_at_ns,
            logical_span,
            cpu_signals: vec![CpuSlowSignals::unsupported(observed_at_ns); logical_span],
            system_signals: SystemSlowSignals::unsupported(observed_at_ns),
        }
    }

    pub fn check_shape(&self) -> Result<(), SampleShapeError> {
        check_span(self.logical_span, self.cpu_signals.len())
    }
}

// ----- Interval signals -----

/// Per-CPU fast signals resolved for one evaluation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CpuIntervalSignals {
    pub scheduler_wait: CounterDelta,
    pub psi_stall: CounterDelta,
    pub reported_scheduler_stall_ratio: ResolvedDouble,
    pub quota_throttle: CounterDelta,
    pub steal: CounterDelta,
    pub external_contention_ratio: ResolvedDouble,
    pub runnable_per_capacity: ResolvedDouble,
}

impl CpuIntervalSignals {
    pub fn unavailable() -> Self {
        Self {
            scheduler_wait: CounterDelta::unavailable(),
            psi_stall: CounterDelta::unavailable(),
            reported_scheduler_stall_ratio: Resolved::unavailable(),
            quota_throttle: CounterDelta::unavailable(),
            steal: CounterDelta::unavailable(),
            external_contention_ratio: Resolved::unavailable(),
            runnable_per_capacity: Resolved::unavailable(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MemoryIntervalSignals {
    pub hard_limit_bytes: ResolvedLong,
    pub high_limit_bytes: ResolvedLong,
    pub usage_bytes: ResolvedLong,
    pub inactive_file_bytes: ResolvedLong,
    pub cumulative_reclaim_bytes: CounterDelta,
    pub memory_stall_ns: CounterDelta,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IoIntervalSignals {
    pub productive_bytes: CounterDelta,
    pub stall_ns: CounterDelta,
    pub operations_latency: LatencyInterval,
    pub maximum_queue_depth: ResolvedDouble,
}

/// Per-CPU slow signals resolved for one evaluation. Thermal severity and low-power mode collapse to
/// `Nominal` / `false` when not valid.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CpuSlowIntervalSignals {
    pub available_capacity_units: ResolvedDouble,
    pub nominal_capacity_units: ResolvedDouble,
    pub constrained_frequency_hz: ResolvedLong,
    pub nominal_frequency_hz: ResolvedLong,
    pub thermal_severity: ThermalSeverity,
    pub low_power_mode: bool,
    pub observed_at_ns: i64,
    pub resolution: SignalResolution,
}

impl CpuSlowIntervalSignals {
    pub fn unavailable() -> Self {
        Self {
            available_capacity_units: Resolved::unavailable(),
            nominal_capacity_units: Resolved::unavailable(),
            constrained_frequency_hz: Resolved::unavailable(),
            nominal_frequency_hz: Resolved::unavailable(),
            thermal_severity: ThermalSeverity::Nominal,
            low_power_mode: false,
            observed_at_ns: 0,
            resolution: SignalResolution::Unavailable,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SystemSlowIntervalSignals {
    pub available_capacity_units: ResolvedDouble,
    pub nominal_capacity_units: ResolvedDouble,
    pub thermal_severity: ThermalSeverity,
    pub low_power_mode: bool,
    pub observed_at_ns: i64,
    pub resolution: SignalResolution,
}

impl SystemSlowIntervalSignals {
    pub fn unavailable() -> Self {
        Self {
            available_capacity_units: Resolved::unavailable(),
            nominal_capacity_units: Resolved::unavailable(),
            thermal_severity: ThermalSeverity::Nominal,
            low_power_mode: false,
            observed_at_ns: 0,
            resolution: SignalResolution::Unavailable,
        }
    }
}

/// Everything the pressure model needs for one evaluation.
#[derive(Clone, Debug, PartialEq)]
pub struct IntervalHardwareSample {
    pub observed_at_ns: i64,
    pub logical_span: usize,
    pub effective_cpus: CpuSet,
    pub quota_capacity_cpus: ResolvedDouble,
    pub quota_period_ns: ResolvedLong,
    pub productive_cpu_ns: CounterDelta,
    pub scope_quota_throttled_ns: CounterDelta,
    pub scope_scheduler_wait_ns: CounterDelta,
    pub scope_psi_stall_ns: CounterDelta,
    pub scope_reported_scheduler_stall_ratio: ResolvedDouble,
    pub cpu_signals: Vec<CpuIntervalSignals>,
    pub memory_signals: MemoryIntervalSignals,
    pub io_signals: IoIntervalSignals,
    pub cpu_slow_signals: Vec<CpuSlowIntervalSignals>,
    pub system_slow_signals: SystemSlowIntervalSignals,
}

// ----- Providers -----

/// A source of legacy [`SystemSnapshot`](crate::monitor::SystemSnapshot)s (upstream
/// `SystemSnapshotProvider`). The monitor wraps one in a compatibility adapter.
pub trait SystemSnapshotProvider: Send {
    /// `None` when this read failed.
    fn snapshot(&mut self) -> Option<crate::monitor::SystemSnapshot>;
}

/// A source of detailed samples (upstream `DetailedSystemSnapshotProvider`).
///
/// Independent of [`SystemSnapshotProvider`]: a platform provider that also offers a legacy snapshot
/// keeps that delta state apart.
pub trait DetailedSnapshotProvider: Send {
    /// One fast-cadence reading, normally at `requested_at_ns`. `None` is a complete failure; the
    /// monitor keeps its last membership and baselines.
    fn sample_fast(&mut self, requested_at_ns: i64) -> Option<FastHardwareSample>;

    /// One slow-cadence reading; `None` is a failure and is retried on the next tick.
    fn sample_slow(&mut self, requested_at_ns: i64) -> Option<SlowHardwareSample>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_signals_carry_the_canonical_zero() {
        let failed = CounterSignal::new(7, 3, SignalValidity::TransientFailure);
        assert_eq!(
            (failed.value(), failed.get(), failed.observed_at_ns()),
            (0, None, 3)
        );
        assert_eq!(CounterSignal::valid(7, 3).get(), Some(7));
        assert!(!BooleanSignal::new(true, 1, SignalValidity::Unsupported).value());
        assert_eq!(
            ThermalSignal::new(ThermalSeverity::Critical, 1, SignalValidity::Unsupported).value(),
            ThermalSeverity::Nominal
        );
    }

    #[test]
    fn invalid_gauge_readings_become_transient_failures() {
        for bad in [-1.0, f64::NAN, f64::INFINITY] {
            let signal = DoubleGaugeSignal::valid(bad, 9);
            assert_eq!(signal.validity(), SignalValidity::TransientFailure, "{bad}");
            assert_eq!(signal.value(), 0.0);
        }
        assert!(DoubleGaugeSignal::valid(-0.0, 0).value().is_sign_positive());
        assert_eq!(
            DoubleGaugeSignal::valid(1.5, 0).ratio().validity(),
            SignalValidity::TransientFailure
        );
        assert_eq!(DoubleGaugeSignal::valid(1.0, 0).ratio().get(), Some(1.0));
    }

    #[test]
    fn counter_deltas_need_a_positive_interval_only_when_usable() {
        assert!(CounterDelta::interval(5, 0, 1, SignalResolution::Current).is_none());
        let delta = CounterDelta::interval(5, 10, 1, SignalResolution::Cached).unwrap();
        assert_eq!(delta.rate(), Some(0.5));
        let baseline = CounterDelta::interval(5, 10, 1, SignalResolution::Baseline).unwrap();
        assert_eq!(
            (baseline.delta(), baseline.elapsed_ns(), baseline.rate()),
            (0, 0, None)
        );
    }

    #[test]
    fn latency_pairs_only_matching_intervals() {
        let latency = CounterDelta::interval(4_000, 100, 1, SignalResolution::Current).unwrap();
        let ops = CounterDelta::interval(2, 100, 1, SignalResolution::Current).unwrap();
        assert_eq!(
            LatencyInterval::new(latency, ops)
                .unwrap()
                .mean_latency_ns(),
            Some(2_000.0)
        );
        let other = CounterDelta::interval(2, 99, 1, SignalResolution::Current).unwrap();
        assert!(LatencyInterval::new(latency, other).is_none());
        assert!(LatencyInterval::new(latency, CounterDelta::baseline(1)).is_none());
    }

    #[test]
    fn shape_checks_catch_malformed_samples() {
        let mut sample = FastHardwareSample::unsupported(0, 2, CpuSet::range(0, 2));
        assert_eq!(sample.check_shape(), Ok(()));
        sample.effective_cpus = CpuSet::range(0, 3);
        assert!(matches!(
            sample.check_shape(),
            Err(SampleShapeError::EffectiveCpuOutOfSpan { cpu: 2, .. })
        ));
        sample.effective_cpus = CpuSet::new();
        sample.cpu_signals.pop();
        assert!(matches!(
            sample.check_shape(),
            Err(SampleShapeError::CpuSignalCount { .. })
        ));
        assert_eq!(
            SlowHardwareSample::unsupported(0, 0).check_shape(),
            Err(SampleShapeError::ZeroSpan)
        );
    }
}
