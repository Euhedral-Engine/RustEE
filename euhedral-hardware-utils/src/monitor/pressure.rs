//! Turns a resolved interval sample into a published [`HardwareUtilization`] (upstream
//! `PressureEvaluator`, `PressureState`, `PressureConstants` and `PressureProjection`).
//!
//! Each input becomes a raw ratio in `[0, 1]`, is smoothed by an asymmetric EWMA (fast attack, slow
//! release), and the smoothed cells combine by `max` within and across the CPU, memory and I/O
//! domains.

use super::sampling::{
    CounterDelta, IntervalHardwareSample, Resolved, SignalResolution, ThermalSeverity,
};
use super::utilization::{HardwareUtilization, HardwareUtilizationParts};

const ATTACK_TAU_SECONDS: f64 = 0.8962840235449102;
const RELEASE_TAU_SECONDS: f64 = 3.8991451492447347;
const HEADROOM_ONSET: f64 = 0.80;
const HEADROOM_RANGE: f64 = 0.20;
/// Reclaiming 2% of the limit per second is full reclaim pressure.
const RECLAIM_FULL_FRACTION: f64 = 0.02;
const RUN_QUEUE_ONSET: f64 = 1.0;
const RUN_QUEUE_RANGE: f64 = 3.0;
const IO_LATENCY_ONSET_NS: f64 = 1_000_000.0;
const IO_LATENCY_RANGE_NS: f64 = 49_000_000.0;
const IO_QUEUE_ONSET: f64 = 1.0;
const IO_QUEUE_RANGE: f64 = 7.0;
const LOW_POWER_LOSS: f64 = 0.15;

/// Clamps to `[0, 1]`; NaN becomes 0.
fn unit(x: f64) -> f64 {
    if x > 0.0 { x.min(1.0) } else { 0.0 }
}

fn thermal_loss(severity: ThermalSeverity) -> f64 {
    match severity {
        ThermalSeverity::Nominal => 0.0,
        ThermalSeverity::Fair => 0.15,
        ThermalSeverity::Serious => 0.35,
        ThermalSeverity::Critical => 0.65,
        ThermalSeverity::Emergency => 1.0,
    }
}

fn low_power_loss(low_power: bool) -> f64 {
    if low_power { LOW_POWER_LOSS } else { 0.0 }
}

fn max(values: &[f64]) -> f64 {
    values
        .iter()
        .fold(0.0, |max, &v| if v > max { v } else { max })
}

/// The fraction of the interval a counter spent stalled; 0 without a usable interval.
fn rate(delta: &CounterDelta) -> f64 {
    delta.rate().map_or(0.0, unit)
}

fn gauge_ratio(gauge: &Resolved<f64>) -> f64 {
    gauge.get().map_or(0.0, unit)
}

/// Whether both members of a ratio pair are available; `Current` stands for the pair.
fn pair(a: SignalResolution, b: SignalResolution) -> SignalResolution {
    if a == SignalResolution::Unavailable || b == SignalResolution::Unavailable {
        SignalResolution::Unavailable
    } else {
        SignalResolution::Current
    }
}

const CELL_SCOPE_WAIT: usize = 0;
const CELL_SCOPE_PSI: usize = 1;
const CELL_SCOPE_REPORTED: usize = 2;
const CELL_GLOBAL_THROTTLE: usize = 3;
const CELL_SYSTEM_THERMAL: usize = 4;
const CELL_SYSTEM_LOW_POWER: usize = 5;
const CELL_HEADROOM: usize = 6;
const CELL_RECLAIM: usize = 7;
const CELL_MEMORY_STALL: usize = 8;
const CELL_IO_STALL: usize = 9;
const CELL_IO_LATENCY: usize = 10;
const CELL_IO_QUEUE: usize = 11;
const GLOBAL_CELL_COUNT: usize = 12;

const PC_WAIT: usize = 0;
const PC_PSI: usize = 1;
const PC_REPORTED: usize = 2;
const PC_RUN_QUEUE: usize = 3;
const PC_CPU_THROTTLE: usize = 4;
const PC_STEAL: usize = 5;
const PC_EXTERNAL: usize = 6;
const PC_CAPACITY_LOSS: usize = 7;
const PC_FREQUENCY_LOSS: usize = 8;
const PC_THERMAL: usize = 9;
const PC_LOW_POWER: usize = 10;
const PER_CPU_CELL_COUNT: usize = 11;

/// One smoothed value and the evaluation that last moved it.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Cell {
    value: f64,
    at_ns: i64,
}

/// The smoother's memory: one cell per global input and per input of each logical CPU.
#[derive(Clone, Debug)]
pub(crate) struct PressureState {
    logical_span: usize,
    cells: Vec<Option<Cell>>,
}

impl PressureState {
    /// `logical_span` must be positive.
    pub(crate) fn new(logical_span: usize) -> Self {
        assert!(logical_span > 0, "logical_span must be positive");
        Self {
            logical_span,
            cells: vec![None; GLOBAL_CELL_COUNT + logical_span * PER_CPU_CELL_COUNT],
        }
    }

    pub(crate) fn clear(&mut self) {
        self.cells.fill(None);
    }

    fn cpu_cell(cpu: usize, offset: usize) -> usize {
        GLOBAL_CELL_COUNT + cpu * PER_CPU_CELL_COUNT + offset
    }

    fn clear_cpu(&mut self, cpu: usize) {
        let base = Self::cpu_cell(cpu, 0);
        self.cells[base..base + PER_CPU_CELL_COUNT].fill(None);
    }

    /// Smooths `input` into a cell. An unavailable input forgets the cell and reads 0; the first input
    /// is taken as is; a repeated or earlier timestamp returns the cell unchanged; otherwise the cell
    /// moves toward the input with `alpha = 1 - exp(-dt / tau)`, `tau` the attack constant when
    /// rising and the release constant when falling.
    fn smooth(
        &mut self,
        index: usize,
        input: f64,
        resolution: SignalResolution,
        evaluation_ns: i64,
    ) -> f64 {
        let cell = &mut self.cells[index];
        if resolution == SignalResolution::Unavailable {
            *cell = None;
            return 0.0;
        }
        let input = unit(input);
        let Some(previous) = *cell else {
            *cell = Some(Cell {
                value: input,
                at_ns: evaluation_ns,
            });
            return input;
        };
        let dt_ns = evaluation_ns.wrapping_sub(previous.at_ns);
        if dt_ns <= 0 {
            return previous.value;
        }
        let tau = if input >= previous.value {
            ATTACK_TAU_SECONDS
        } else {
            RELEASE_TAU_SECONDS
        };
        let alpha = unit(-(-(dt_ns as f64 / 1e9) / tau).exp_m1());
        let value = unit(previous.value + (input - previous.value) * alpha);
        *cell = Some(Cell {
            value,
            at_ns: evaluation_ns,
        });
        value
    }

    #[cfg(test)]
    fn values(&self) -> impl Iterator<Item = f64> + '_ {
        self.cells.iter().flatten().map(|cell| cell.value)
    }
}

/// Evaluates `sample` at `evaluation_ns`, updating `state`.
pub(crate) fn evaluate(
    sample: &IntervalHardwareSample,
    state: &mut PressureState,
    evaluation_ns: i64,
) -> HardwareUtilization {
    debug_assert_eq!(state.logical_span, sample.logical_span);
    let span = sample.logical_span.min(state.logical_span);
    let effective = &sample.effective_cpus;
    let effective_count = effective.len();

    // Without a quota the scope may use every effective CPU.
    let quota_cpus = sample
        .quota_capacity_cpus
        .get()
        .filter(|quota| *quota > 0.0)
        .map_or(effective_count as f64, |quota| {
            quota.min(effective_count as f64)
        });
    let quota_cpu_usage = sample
        .productive_cpu_ns
        .rate()
        .filter(|_| quota_cpus > 0.0)
        .map_or(0.0, |rate| unit(rate / quota_cpus));

    let scope_wait_raw = rate(&sample.scope_scheduler_wait_ns);
    let scope_psi_raw = rate(&sample.scope_psi_stall_ns);
    let scope_reported_raw = gauge_ratio(&sample.scope_reported_scheduler_stall_ratio);
    let global_throttle_raw = rate(&sample.scope_quota_throttled_ns);
    let system_slow = &sample.system_slow_signals;

    let at = evaluation_ns;
    let scope_wait = state.smooth(
        CELL_SCOPE_WAIT,
        scope_wait_raw,
        sample.scope_scheduler_wait_ns.resolution(),
        at,
    );
    let scope_psi = state.smooth(
        CELL_SCOPE_PSI,
        scope_psi_raw,
        sample.scope_psi_stall_ns.resolution(),
        at,
    );
    let scope_reported = state.smooth(
        CELL_SCOPE_REPORTED,
        scope_reported_raw,
        sample.scope_reported_scheduler_stall_ratio.resolution(),
        at,
    );
    let global_throttle = state.smooth(
        CELL_GLOBAL_THROTTLE,
        global_throttle_raw,
        sample.scope_quota_throttled_ns.resolution(),
        at,
    );
    let system_thermal = state.smooth(
        CELL_SYSTEM_THERMAL,
        thermal_loss(system_slow.thermal_severity),
        system_slow.resolution,
        at,
    );
    let system_low_power = state.smooth(
        CELL_SYSTEM_LOW_POWER,
        low_power_loss(system_slow.low_power_mode),
        system_slow.resolution,
        at,
    );

    let mut per_cpu_pressure = vec![0.0; sample.logical_span];
    let mut per_cpu_throttle = vec![0.0; sample.logical_span];
    let mut per_cpu_stall = vec![0.0; sample.logical_span];

    for cpu in 0..span {
        let (Some(signals), Some(slow)) = (
            sample.cpu_signals.get(cpu),
            sample.cpu_slow_signals.get(cpu),
        ) else {
            state.clear_cpu(cpu);
            continue;
        };
        if !effective.contains(cpu) {
            state.clear_cpu(cpu);
            continue;
        }
        let cell = |offset| PressureState::cpu_cell(cpu, offset);

        let wait_raw = rate(&signals.scheduler_wait);
        let psi_raw = rate(&signals.psi_stall);
        let reported_raw = gauge_ratio(&signals.reported_scheduler_stall_ratio);
        let run_queue_raw = signals.runnable_per_capacity.get().map_or(0.0, |runnable| {
            unit((runnable - RUN_QUEUE_ONSET) / RUN_QUEUE_RANGE)
        });
        per_cpu_stall[cpu] = max(&[
            scope_wait_raw,
            scope_psi_raw,
            scope_reported_raw,
            wait_raw,
            psi_raw,
            reported_raw,
            run_queue_raw,
        ]);

        let scheduler = max(&[
            scope_wait,
            scope_psi,
            scope_reported,
            state.smooth(
                cell(PC_WAIT),
                wait_raw,
                signals.scheduler_wait.resolution(),
                at,
            ),
            state.smooth(cell(PC_PSI), psi_raw, signals.psi_stall.resolution(), at),
            state.smooth(
                cell(PC_REPORTED),
                reported_raw,
                signals.reported_scheduler_stall_ratio.resolution(),
                at,
            ),
            state.smooth(
                cell(PC_RUN_QUEUE),
                run_queue_raw,
                signals.runnable_per_capacity.resolution(),
                at,
            ),
        ]);

        let cpu_throttle = state.smooth(
            cell(PC_CPU_THROTTLE),
            rate(&signals.quota_throttle),
            signals.quota_throttle.resolution(),
            at,
        );
        let throttle = max(&[global_throttle, cpu_throttle]);
        per_cpu_throttle[cpu] = throttle;

        let external = max(&[
            state.smooth(
                cell(PC_STEAL),
                rate(&signals.steal),
                signals.steal.resolution(),
                at,
            ),
            state.smooth(
                cell(PC_EXTERNAL),
                gauge_ratio(&signals.external_contention_ratio),
                signals.external_contention_ratio.resolution(),
                at,
            ),
        ]);

        let capacity_raw = match (
            slow.available_capacity_units.get(),
            slow.nominal_capacity_units.get(),
        ) {
            (Some(available), Some(nominal)) if nominal > 0.0 => unit(1.0 - available / nominal),
            _ => 0.0,
        };
        let frequency_raw = match (
            slow.constrained_frequency_hz.get(),
            slow.nominal_frequency_hz.get(),
        ) {
            (Some(frequency), Some(nominal)) if nominal > 0 => {
                unit(1.0 - frequency as f64 / nominal as f64)
            }
            _ => 0.0,
        };
        let capacity = max(&[
            state.smooth(
                cell(PC_CAPACITY_LOSS),
                capacity_raw,
                pair(
                    slow.available_capacity_units.resolution(),
                    slow.nominal_capacity_units.resolution(),
                ),
                at,
            ),
            state.smooth(
                cell(PC_FREQUENCY_LOSS),
                frequency_raw,
                pair(
                    slow.constrained_frequency_hz.resolution(),
                    slow.nominal_frequency_hz.resolution(),
                ),
                at,
            ),
            state.smooth(
                cell(PC_THERMAL),
                thermal_loss(slow.thermal_severity),
                slow.resolution,
                at,
            ),
            system_thermal,
            state.smooth(
                cell(PC_LOW_POWER),
                low_power_loss(slow.low_power_mode),
                slow.resolution,
                at,
            ),
            system_low_power,
        ]);

        per_cpu_pressure[cpu] = max(&[scheduler, throttle, external, capacity]);
    }
    for cpu in span..state.logical_span {
        state.clear_cpu(cpu);
    }

    // Memory: the hard limit, else the high limit; `u64::MAX` reads as no limit.
    let memory = &sample.memory_signals;
    let limit = memory
        .hard_limit_bytes
        .get()
        .or_else(|| memory.high_limit_bytes.get())
        .filter(|limit| *limit != u64::MAX);
    let usage = memory.usage_bytes.get().unwrap_or(0);
    let inactive = memory.inactive_file_bytes.get().unwrap_or(0);
    let working_set = usage.saturating_sub(inactive);
    let memory_utilization = match limit {
        Some(0) => 1.0,
        Some(limit) => unit(working_set as f64 / limit as f64),
        None => 0.0,
    };
    let headroom = match limit {
        Some(_) => state.smooth(
            CELL_HEADROOM,
            (memory_utilization - HEADROOM_ONSET) / HEADROOM_RANGE,
            SignalResolution::Current,
            at,
        ),
        None => state.smooth(CELL_HEADROOM, 0.0, SignalResolution::Unavailable, at),
    };
    // In f64 throughout, so large limits over long intervals cannot overflow.
    let reclaim = &memory.cumulative_reclaim_bytes;
    let reclaim_raw = match (limit, reclaim.rate()) {
        (Some(limit), Some(bytes_per_ns)) if limit > 0 => {
            unit(bytes_per_ns * 1e9 / limit as f64 / RECLAIM_FULL_FRACTION)
        }
        _ => 0.0,
    };
    let reclaim_resolution = match limit {
        Some(limit) if limit > 0 => reclaim.resolution(),
        _ => SignalResolution::Unavailable,
    };
    let memory_domain = max(&[
        headroom,
        state.smooth(CELL_RECLAIM, reclaim_raw, reclaim_resolution, at),
        state.smooth(
            CELL_MEMORY_STALL,
            rate(&memory.memory_stall_ns),
            memory.memory_stall_ns.resolution(),
            at,
        ),
    ]);

    let io = &sample.io_signals;
    let latency_raw = io.operations_latency.mean_latency_ns().map_or(0.0, |mean| {
        unit((mean - IO_LATENCY_ONSET_NS) / IO_LATENCY_RANGE_NS)
    });
    let queue_raw = io
        .maximum_queue_depth
        .get()
        .map_or(0.0, |depth| unit((depth - IO_QUEUE_ONSET) / IO_QUEUE_RANGE));
    let io_domain = max(&[
        state.smooth(
            CELL_IO_STALL,
            rate(&io.stall_ns),
            io.stall_ns.resolution(),
            at,
        ),
        state.smooth(
            CELL_IO_LATENCY,
            latency_raw,
            io.operations_latency.resolution(),
            at,
        ),
        state.smooth(
            CELL_IO_QUEUE,
            queue_raw,
            io.maximum_queue_depth.resolution(),
            at,
        ),
    ]);
    let disk_io_bytes_per_second = io.productive_bytes.rate().map_or(0.0, |rate| rate * 1e9);

    for cpu in effective.iter().filter(|&cpu| cpu < span) {
        per_cpu_pressure[cpu] = max(&[per_cpu_pressure[cpu], memory_domain, io_domain]);
    }

    let cpu_throttle_ratio =
        if sample.scope_quota_throttled_ns.resolution() == SignalResolution::Unavailable {
            effective
                .iter()
                .filter_map(|cpu| per_cpu_throttle.get(cpu).copied())
                .fold(0.0, |max, v| if v > max { v } else { max })
        } else if effective.is_empty() {
            0.0
        } else {
            global_throttle
        };

    let per_cpu = |bytes: u64| match effective_count {
        0 => 0,
        count => bytes / count as u64,
    };
    let global_memory_pool = limit.unwrap_or(u64::MAX);
    HardwareUtilization::new(HardwareUtilizationParts {
        timestamp_ns: evaluation_ns,
        quota_cpus,
        quota_cpu_usage,
        period_ns: sample.quota_period_ns.get().unwrap_or(0),
        global_effective_cpus: effective.clone(),
        cpu_throttle_ratio,
        per_quota_cpu_throttle_ratio: per_cpu_throttle,
        per_quota_cpu_pressure: per_cpu_pressure,
        per_cpu_scheduler_stall: per_cpu_stall,
        global_memory_pool,
        per_cpu_memory_pool: if limit.is_some() {
            per_cpu(global_memory_pool)
        } else {
            u64::MAX
        },
        total_memory_utilization: memory_utilization,
        mem_per_cpu_usage_bytes: per_cpu(working_set),
        memory_usage_bytes: usage,
        inactive_file_bytes: inactive,
        disk_io_bytes_per_second,
        disk_io_pressure: io_domain,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::monitor::engine::SampleStateEngine;
    use crate::monitor::sampling::{
        CpuIntervalSignals, CpuSlowIntervalSignals, FastHardwareSample, IoIntervalSignals,
        LatencyInterval, MemoryIntervalSignals, Signal, SystemSlowIntervalSignals,
    };
    use crate::util::CpuSet;

    const NOW: i64 = 1_000;

    fn current<T: Copy + Default>(value: T) -> Resolved<T> {
        Resolved::new(value, NOW, SignalResolution::Current)
    }

    fn delta(delta: u64, elapsed: i64) -> CounterDelta {
        CounterDelta::interval(delta, elapsed, NOW, SignalResolution::Current).unwrap()
    }

    fn quiet_sample(span: usize, effective: CpuSet) -> IntervalHardwareSample {
        IntervalHardwareSample {
            observed_at_ns: NOW,
            logical_span: span,
            effective_cpus: effective,
            quota_capacity_cpus: Resolved::unavailable(),
            quota_period_ns: Resolved::unavailable(),
            productive_cpu_ns: CounterDelta::unavailable(),
            scope_quota_throttled_ns: CounterDelta::unavailable(),
            scope_scheduler_wait_ns: CounterDelta::unavailable(),
            scope_psi_stall_ns: CounterDelta::unavailable(),
            scope_reported_scheduler_stall_ratio: Resolved::unavailable(),
            cpu_signals: vec![CpuIntervalSignals::unavailable(); span],
            memory_signals: MemoryIntervalSignals {
                hard_limit_bytes: Resolved::unavailable(),
                high_limit_bytes: Resolved::unavailable(),
                usage_bytes: Resolved::unavailable(),
                inactive_file_bytes: Resolved::unavailable(),
                cumulative_reclaim_bytes: CounterDelta::unavailable(),
                memory_stall_ns: CounterDelta::unavailable(),
            },
            io_signals: IoIntervalSignals {
                productive_bytes: CounterDelta::unavailable(),
                stall_ns: CounterDelta::unavailable(),
                operations_latency: LatencyInterval::unavailable(),
                maximum_queue_depth: Resolved::unavailable(),
            },
            cpu_slow_signals: vec![CpuSlowIntervalSignals::unavailable(); span],
            system_slow_signals: SystemSlowIntervalSignals::unavailable(),
        }
    }

    fn close(actual: f64, expected: f64) {
        assert!((actual - expected).abs() < 1e-3, "{actual} != {expected}");
    }

    #[test]
    fn composition_golden() {
        let mut sample = quiet_sample(1, CpuSet::range(0, 1));
        sample.quota_capacity_cpus = current(1000.0);
        sample.quota_period_ns = current(1000);
        sample.cpu_signals[0] = CpuIntervalSignals {
            scheduler_wait: delta(25, 100),
            psi_stall: delta(40, 100),
            quota_throttle: delta(30, 100),
            steal: delta(20, 100),
            runnable_per_capacity: current(2.5),
            ..CpuIntervalSignals::unavailable()
        };
        sample.cpu_slow_signals[0] = CpuSlowIntervalSignals {
            available_capacity_units: current(90.0),
            nominal_capacity_units: current(100.0),
            resolution: SignalResolution::Current,
            observed_at_ns: NOW,
            ..CpuSlowIntervalSignals::unavailable()
        };
        sample.memory_signals.hard_limit_bytes = current(1000);
        sample.memory_signals.usage_bytes = current(900);
        sample.memory_signals.cumulative_reclaim_bytes = delta(10, 1_000_000_000);
        sample.memory_signals.memory_stall_ns = delta(20, 100);
        sample.io_signals.stall_ns = delta(40, 100);
        sample.io_signals.operations_latency =
            LatencyInterval::new(delta(25_500_000, 100), delta(1, 100)).unwrap();
        sample.system_slow_signals.resolution = SignalResolution::Current;

        let util = evaluate(&sample, &mut PressureState::new(1), NOW);
        close(util.pressure(), 0.50);
        close(util.per_quota_cpu_pressure[0], 0.50);
        close(util.cpu_throttle_ratio, 0.30);
        close(util.total_memory_utilization, 0.90);
        close(util.disk_io_pressure, 0.50);
        assert_eq!(util.quota_cpus, 1.0);
        assert_eq!(
            (util.per_cpu_memory_pool, util.mem_per_cpu_usage_bytes),
            (1000, 900)
        );
    }

    #[test]
    fn smoothing_attacks_fast_and_releases_slowly() {
        let mut state = PressureState::new(1);
        let current = SignalResolution::Current;
        assert_eq!(state.smooth(0, 0.0, current, 0), 0.0);
        let up = state.smooth(0, 1.0, current, 1_000_000_000);
        close(up, 1.0 - (-1.0 / ATTACK_TAU_SECONDS).exp());
        let down = state.smooth(0, 0.0, current, 2_000_000_000);
        close(down, up * (-1.0 / RELEASE_TAU_SECONDS).exp());
        assert_eq!(state.smooth(0, 1.0, current, 2_000_000_000), down);
        assert!(!state.smooth(0, f64::NAN, current, 3_000_000_000).is_nan());
        assert_eq!(
            state.smooth(0, 1.0, SignalResolution::Unavailable, 4_000_000_000),
            0.0
        );
        assert_eq!(state.smooth(0, 0.7, current, 5_000_000_000), 0.7);
    }

    #[test]
    fn usage_is_against_the_effective_cpus_without_a_quota() {
        let mut sample = quiet_sample(2, CpuSet::range(0, 2));
        sample.productive_cpu_ns = delta(100, 100);
        let util = evaluate(&sample, &mut PressureState::new(2), NOW);
        assert_eq!(util.quota_cpus, 2.0);
        close(util.quota_cpu_usage, 0.5);
        sample.quota_capacity_cpus = current(0.5);
        let util = evaluate(&sample, &mut PressureState::new(2), NOW);
        assert_eq!(util.quota_cpus, 0.5);
        close(util.quota_cpu_usage, 1.0);
    }

    #[test]
    fn reclaim_over_a_large_limit_does_not_overflow() {
        let mut sample = quiet_sample(1, CpuSet::range(0, 1));
        let limit = 64 << 30;
        sample.memory_signals.hard_limit_bytes = current(limit);
        // 1% of the limit per second for 10 s: half of full reclaim pressure.
        sample.memory_signals.cumulative_reclaim_bytes = delta(limit / 10, 10_000_000_000);
        let util = evaluate(&sample, &mut PressureState::new(1), NOW);
        close(util.pressure(), 0.5);
        assert_eq!(util.global_memory_pool, limit);
    }

    #[test]
    fn unknown_memory_limit_publishes_unlimited_pools() {
        let mut sample = quiet_sample(2, CpuSet::range(0, 2));
        sample.memory_signals.usage_bytes = current(600);
        sample.memory_signals.inactive_file_bytes = current(200);
        let util = evaluate(&sample, &mut PressureState::new(2), NOW);
        assert_eq!(
            (util.global_memory_pool, util.per_cpu_memory_pool),
            (u64::MAX, u64::MAX)
        );
        assert_eq!(util.mem_per_cpu_usage_bytes, 200);
        assert_eq!(
            (util.memory_usage_bytes, util.inactive_file_bytes),
            (600, 200)
        );
        assert_eq!(util.total_memory_utilization, 0.0);
    }

    #[test]
    fn non_effective_cpus_are_cleared_and_read_zero() {
        let mut state = PressureState::new(2);
        let mut sample = quiet_sample(2, CpuSet::range(0, 2));
        sample.cpu_signals[1].scheduler_wait = delta(50, 100);
        let util = evaluate(&sample, &mut state, NOW);
        close(util.per_quota_cpu_pressure[1], 0.5);
        let mut sample = quiet_sample(2, CpuSet::range(0, 1));
        sample.cpu_signals[1].scheduler_wait = delta(50, 100);
        let util = evaluate(&sample, &mut state, NOW + 1);
        assert_eq!(util.per_quota_cpu_pressure[1], 0.0);
        assert!(
            state.cells[PressureState::cpu_cell(1, 0)..]
                .iter()
                .all(Option::is_none)
        );
    }

    #[test]
    fn throttle_falls_back_to_the_per_cpu_maximum() {
        let mut sample = quiet_sample(2, CpuSet::range(0, 2));
        sample.cpu_signals[0].quota_throttle = delta(10, 100);
        sample.cpu_signals[1].quota_throttle = delta(40, 100);
        close(
            evaluate(&sample, &mut PressureState::new(2), NOW).cpu_throttle_ratio,
            0.4,
        );
        sample.scope_quota_throttled_ns = delta(20, 100);
        close(
            evaluate(&sample, &mut PressureState::new(2), NOW).cpu_throttle_ratio,
            0.2,
        );
    }

    /// A transient counter failure holds pressure, and recovery does not drop it.
    #[test]
    fn transient_failure_holds_prior_pressure() {
        const SPAN: usize = 2;
        const PERIOD: i64 = 100_000_000;
        let mut engine = SampleStateEngine::new(SPAN, PERIOD);
        let mut state = PressureState::new(SPAN);
        let mut step = |at: i64, wait| {
            let mut sample = FastHardwareSample::unsupported(at, SPAN, CpuSet::range(0, 1));
            sample.scope_scheduler_wait_ns = wait;
            let interval = engine
                .process_fast(at, Some(&sample))
                .expect("engine published");
            evaluate(&interval, &mut state, at).per_quota_cpu_pressure[0]
        };
        let t1 = 1_000_000_000;
        let (t2, t3, t4) = (t1 + PERIOD, t1 + 2 * PERIOD, t1 + 3 * PERIOD);
        step(t1, Signal::valid(0, t1));
        let valid = step(t2, Signal::valid(50_000_000, t2));
        assert!(valid > 0.0);
        let failed = step(t3, Signal::transient_failure(t3));
        assert!(failed >= valid - 1e-9, "{failed} fell below {valid}");
        let recovered = step(t4, Signal::valid(100_000_000, t4));
        assert!(
            recovered >= failed - 1e-9,
            "{recovered} fell below {failed}"
        );
    }

    /// A small deterministic generator, so the property tests need no dependency.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }

        fn chance(&mut self) -> bool {
            self.next() & 1 == 0
        }

        fn resolution(&mut self) -> SignalResolution {
            [
                SignalResolution::Current,
                SignalResolution::Cached,
                SignalResolution::Baseline,
                SignalResolution::Unavailable,
            ][self.below(4) as usize]
        }

        fn counter(&mut self) -> CounterDelta {
            let elapsed = [1, 100, 1_000_000_000, i64::MAX][self.below(4) as usize];
            let delta = [0, 1, self.next() >> 34, u64::MAX][self.below(4) as usize];
            CounterDelta::interval(delta, elapsed, NOW, self.resolution()).unwrap()
        }

        fn double(&mut self) -> Resolved<f64> {
            let value = [0.0, 0.5, 1.0, 3.0, 1e300][self.below(5) as usize];
            Resolved::new(value, NOW, self.resolution())
        }

        fn long(&mut self) -> Resolved<u64> {
            let value = [0, 1, self.next() >> 20, u64::MAX][self.below(4) as usize];
            Resolved::new(value, NOW, self.resolution())
        }

        fn thermal(&mut self) -> ThermalSeverity {
            [
                ThermalSeverity::Nominal,
                ThermalSeverity::Fair,
                ThermalSeverity::Serious,
                ThermalSeverity::Critical,
                ThermalSeverity::Emergency,
            ][self.below(5) as usize]
        }

        fn sample(&mut self, span: usize) -> IntervalHardwareSample {
            let effective = (0..span).filter(|_| self.chance()).collect();
            let mut sample = quiet_sample(span, effective);
            sample.quota_capacity_cpus = self.double();
            sample.quota_period_ns = self.long();
            sample.productive_cpu_ns = self.counter();
            sample.scope_quota_throttled_ns = self.counter();
            sample.scope_scheduler_wait_ns = self.counter();
            sample.scope_psi_stall_ns = self.counter();
            sample.scope_reported_scheduler_stall_ratio = self.double();
            for cpu in 0..span {
                sample.cpu_signals[cpu] = CpuIntervalSignals {
                    scheduler_wait: self.counter(),
                    psi_stall: self.counter(),
                    reported_scheduler_stall_ratio: self.double(),
                    quota_throttle: self.counter(),
                    steal: self.counter(),
                    external_contention_ratio: self.double(),
                    runnable_per_capacity: self.double(),
                };
                sample.cpu_slow_signals[cpu] = CpuSlowIntervalSignals {
                    available_capacity_units: self.double(),
                    nominal_capacity_units: self.double(),
                    constrained_frequency_hz: self.long(),
                    nominal_frequency_hz: self.long(),
                    thermal_severity: self.thermal(),
                    low_power_mode: self.chance(),
                    observed_at_ns: NOW,
                    resolution: self.resolution(),
                };
            }
            sample.memory_signals = MemoryIntervalSignals {
                hard_limit_bytes: self.long(),
                high_limit_bytes: self.long(),
                usage_bytes: self.long(),
                inactive_file_bytes: self.long(),
                cumulative_reclaim_bytes: self.counter(),
                memory_stall_ns: self.counter(),
            };
            sample.io_signals = IoIntervalSignals {
                productive_bytes: self.counter(),
                stall_ns: self.counter(),
                operations_latency: LatencyInterval::new(self.counter(), self.counter())
                    .unwrap_or_else(LatencyInterval::unavailable),
                maximum_queue_depth: self.double(),
            };
            sample.system_slow_signals.thermal_severity = self.thermal();
            sample.system_slow_signals.low_power_mode = self.chance();
            sample.system_slow_signals.resolution = self.resolution();
            sample
        }
    }

    fn assert_unit(value: f64, what: &str) {
        assert!(
            value.is_finite() && (0.0..=1.0).contains(&value),
            "{what} = {value}"
        );
    }

    fn assert_outputs_in_unit(util: &HardwareUtilization, context: &str) {
        assert_unit(util.cpu_throttle_ratio, context);
        assert_unit(util.total_memory_utilization, context);
        assert_unit(util.disk_io_pressure, context);
        assert_unit(util.quota_cpu_usage, context);
        for values in [
            &util.per_quota_cpu_pressure,
            &util.per_quota_cpu_throttle_ratio,
            &util.per_cpu_scheduler_stall,
        ] {
            values.iter().for_each(|&value| assert_unit(value, context));
        }
        assert!(util.disk_io_bytes_per_second.is_finite() && util.disk_io_bytes_per_second >= 0.0);
    }

    #[test]
    fn every_cell_and_output_is_finite_and_in_the_unit_interval() {
        let mut rng = Rng(0x5eed_1234_abcd_ef01);
        let mut state = PressureState::new(3);
        let mut at = 1_000_000;
        for step in 0..5_000 {
            at += match rng.below(8) {
                0 => 0,
                1 => 1,
                2 => 1_000_000_000_000,
                _ => 1 + rng.below(2_000_000_000) as i64,
            };
            let util = evaluate(&rng.sample(3), &mut state, at);
            let context = format!("step {step}");
            state
                .values()
                .for_each(|value| assert_unit(value, &context));
            assert_outputs_in_unit(&util, &context);
        }
    }

    type Knob = fn(&mut IntervalHardwareSample, &mut Rng);

    /// The generator seed a knob sees in its prepare call.
    const PREPARE: u64 = 0x2545_f491_4f6c_dd1d;

    /// The prepare call makes the counter live with its current value; the raise call increases it.
    fn raise_counter(counter: &mut CounterDelta, rng: &mut Rng) {
        let elapsed = if counter.resolution().is_usable() {
            counter.elapsed_ns()
        } else {
            1_000_000
        };
        let delta = match rng.0 {
            PREPARE => counter.delta(),
            _ => counter.delta().max(rng.below(elapsed as u64 + 1)),
        };
        *counter = CounterDelta::interval(delta, elapsed, NOW, SignalResolution::Current).unwrap();
    }

    fn raise_double(gauge: &mut Resolved<f64>, rng: &mut Rng) {
        let value = match rng.0 {
            PREPARE => gauge.value(),
            _ => gauge.value().max(rng.below(1_000) as f64 / 100.0),
        };
        *gauge = Resolved::new(value, NOW, SignalResolution::Current);
    }

    /// Each knob makes one input live and worse, and must never lower an output.
    fn knobs() -> Vec<(&'static str, Knob)> {
        vec![
            ("scope.wait", |s, r| {
                raise_counter(&mut s.scope_scheduler_wait_ns, r)
            }),
            ("scope.psi", |s, r| {
                raise_counter(&mut s.scope_psi_stall_ns, r)
            }),
            ("scope.throttle", |s, r| {
                raise_counter(&mut s.scope_quota_throttled_ns, r)
            }),
            ("scope.reported", |s, r| {
                raise_double(&mut s.scope_reported_scheduler_stall_ratio, r)
            }),
            ("cpu0.wait", |s, r| {
                raise_counter(&mut s.cpu_signals[0].scheduler_wait, r)
            }),
            ("cpu0.psi", |s, r| {
                raise_counter(&mut s.cpu_signals[0].psi_stall, r)
            }),
            ("cpu0.throttle", |s, r| {
                raise_counter(&mut s.cpu_signals[0].quota_throttle, r)
            }),
            ("cpu0.steal", |s, r| {
                raise_counter(&mut s.cpu_signals[0].steal, r)
            }),
            ("cpu0.reported", |s, r| {
                raise_double(&mut s.cpu_signals[0].reported_scheduler_stall_ratio, r)
            }),
            ("cpu0.external", |s, r| {
                raise_double(&mut s.cpu_signals[0].external_contention_ratio, r)
            }),
            ("cpu0.run_queue", |s, r| {
                raise_double(&mut s.cpu_signals[0].runnable_per_capacity, r)
            }),
            ("cpu0.thermal", |s, r| {
                let slow = &mut s.cpu_slow_signals[0];
                slow.resolution = SignalResolution::Current;
                if r.0 != PREPARE {
                    slow.thermal_severity = slow.thermal_severity.max(r.thermal());
                }
            }),
            ("system.low_power", |s, r| {
                // Called once to prepare (live, off) and once to raise (on).
                let system = &mut s.system_slow_signals;
                system.low_power_mode = r.0 != PREPARE;
                system.resolution = SignalResolution::Current;
            }),
            ("memory.stall", |s, r| {
                raise_counter(&mut s.memory_signals.memory_stall_ns, r)
            }),
            ("io.stall", |s, r| {
                raise_counter(&mut s.io_signals.stall_ns, r)
            }),
            ("io.queue", |s, r| {
                raise_double(&mut s.io_signals.maximum_queue_depth, r)
            }),
        ]
    }

    #[test]
    fn raising_one_input_never_lowers_pressure() {
        const SPAN: usize = 2;
        for (name, raise) in knobs() {
            let mut rng = Rng(0x9e37_79b9_7f4a_7c15
                ^ name.len() as u64
                ^ (name.as_bytes()[name.len() - 1] as u64) << 40);
            let mut strictly_raised = 0;
            for trial in 0..300 {
                let history: Vec<_> = (0..rng.below(4)).map(|_| rng.sample(SPAN)).collect();
                let mut base = if trial % 2 == 0 {
                    quiet_sample(SPAN, CpuSet::new())
                } else {
                    rng.sample(SPAN)
                };
                base.effective_cpus = base.effective_cpus.with(0);
                // Make the input live in both runs, so only its value differs.
                raise(&mut base, &mut Rng(PREPARE));
                let mut raised = base.clone();
                raise(&mut raised, &mut rng);
                let run = |last: &IntervalHardwareSample| {
                    let mut state = PressureState::new(SPAN);
                    let mut at = 1_000_000;
                    for sample in &history {
                        at += 100_000_000;
                        evaluate(sample, &mut state, at);
                    }
                    evaluate(last, &mut state, at + 100_000_000)
                };
                let (low, high) = (run(&base), run(&raised));
                let context = format!("{name} trial {trial}");
                assert!(
                    high.cpu_throttle_ratio >= low.cpu_throttle_ratio - 1e-12,
                    "{context}"
                );
                assert!(
                    high.disk_io_pressure >= low.disk_io_pressure - 1e-12,
                    "{context}"
                );
                for cpu in 0..SPAN {
                    assert!(
                        high.per_quota_cpu_pressure[cpu] >= low.per_quota_cpu_pressure[cpu] - 1e-12,
                        "{context} cpu {cpu}"
                    );
                    assert!(
                        high.per_cpu_scheduler_stall[cpu]
                            >= low.per_cpu_scheduler_stall[cpu] - 1e-12,
                        "{context} cpu {cpu}"
                    );
                }
                if high.per_quota_cpu_pressure.iter().sum::<f64>()
                    > low.per_quota_cpu_pressure.iter().sum::<f64>()
                {
                    strictly_raised += 1;
                }
            }
            assert!(
                strictly_raised > 0,
                "{name} never raised pressure; the property would be vacuous"
            );
        }
    }
}
