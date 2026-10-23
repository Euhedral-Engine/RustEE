//! Published monitor values (upstream `SystemUtilization` and its nested records).
//!
//! Every type here is immutable once built. The constructors sanitize their inputs (ratios clamped to `[0, 1]`, NaN and negatives to zero), so a consumer
//! never sees a value outside its documented range. The structs are `#[non_exhaustive]`: other crates
//! read the fields but build values only through the constructors.

use crate::util::CpuSet;
use std::error::Error;
use std::fmt;

/// A malformed snapshot or a malformed socket/core query.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum UtilizationError {
    ZeroCpus,
    PressureLength { expected: usize, actual: usize },
    EffectiveCpuOutOfSpan { cpu: usize, span: usize },
    CpuInTwoCores { cpu: usize },
    CoreHasNonEffectiveCpu { cpu: usize },
}

impl fmt::Display for UtilizationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UtilizationError::ZeroCpus => f.write_str("total_cpus must be positive"),
            UtilizationError::PressureLength { expected, actual } => {
                write!(
                    f,
                    "pressure_per_cpu has {actual} entries; expected {expected}"
                )
            }
            UtilizationError::EffectiveCpuOutOfSpan { cpu, span } => {
                write!(f, "effective CPU {cpu} is outside the span {span}")
            }
            UtilizationError::CpuInTwoCores { cpu } => {
                write!(f, "CPU {cpu} belongs to more than one core")
            }
            UtilizationError::CoreHasNonEffectiveCpu { cpu } => {
                write!(f, "core contains non-effective CPU {cpu}")
            }
        }
    }
}

impl Error for UtilizationError {}

/// `[0, 1]`, with NaN, negatives and infinities at 0.
pub(crate) fn sanitize_ratio(value: f64) -> f64 {
    if value.is_finite() && value > 0.0 {
        value.min(1.0)
    } else {
        0.0
    }
}

/// Non-negative and finite: NaN and negatives at 0, `+inf` at `f64::MAX`.
pub(crate) fn sanitize_telemetry(value: f64) -> f64 {
    if value.is_nan() || value <= 0.0 {
        0.0
    } else {
        value.min(f64::MAX)
    }
}

fn saturating_mul(value: u64, count: usize) -> u64 {
    value.saturating_mul(count as u64)
}

fn saturating_product(pool: u64, ratio: f64) -> u64 {
    let product = pool as f64 * ratio;
    if product.is_nan() || product <= 0.0 {
        0
    } else if product >= u64::MAX as f64 {
        u64::MAX
    } else {
        product as u64
    }
}

fn utilization_of(usage: u64, limit: u64) -> f64 {
    usage as f64 / limit.max(1) as f64
}

/// A legacy whole-scope reading (upstream `SystemSnapshot`), as produced by a
/// [`SystemSnapshotProvider`](super::SystemSnapshotProvider). The period is in nanoseconds.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct SystemSnapshot {
    pub time_ns: i64,
    /// The logical CPU span (highest id + 1).
    pub total_cpus: usize,
    /// Clamped to `[0, effective CPU count]`.
    pub quota_cpus: f64,
    pub period_ns: u64,
    /// Cumulative CPU time of the scope, ns.
    pub cpu_usage_ns: u64,
    /// Cumulative quota throttling of the scope, ns.
    pub cpu_throttle_ns: u64,
    pub effective_cpus: CpuSet,
    /// Per-CPU pressure ratios, `total_cpus` entries clamped to `[0, 1]`.
    pub pressure_per_cpu: Box<[f64]>,
    /// `u64::MAX` when unlimited.
    pub memory_limit_bytes: u64,
    pub memory_usage_bytes: u64,
    pub inactive_file_bytes: u64,
    /// Cumulative I/O bytes.
    pub disk_io_bytes: u64,
}

impl SystemSnapshot {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        time_ns: i64,
        total_cpus: usize,
        quota_cpus: f64,
        period_ns: u64,
        cpu_usage_ns: u64,
        cpu_throttle_ns: u64,
        effective_cpus: CpuSet,
        pressure_per_cpu: &[f64],
        memory_limit_bytes: u64,
        memory_usage_bytes: u64,
        inactive_file_bytes: u64,
        disk_io_bytes: u64,
    ) -> Result<Self, UtilizationError> {
        if total_cpus == 0 {
            return Err(UtilizationError::ZeroCpus);
        }
        if pressure_per_cpu.len() != total_cpus {
            return Err(UtilizationError::PressureLength {
                expected: total_cpus,
                actual: pressure_per_cpu.len(),
            });
        }
        if let Some(cpu) = effective_cpus.last().filter(|&cpu| cpu >= total_cpus) {
            return Err(UtilizationError::EffectiveCpuOutOfSpan {
                cpu,
                span: total_cpus,
            });
        }
        let quota_cpus = if quota_cpus.is_finite() && quota_cpus > 0.0 {
            quota_cpus.min(effective_cpus.len() as f64)
        } else {
            0.0
        };
        Ok(Self {
            time_ns,
            total_cpus,
            quota_cpus,
            period_ns,
            cpu_usage_ns,
            cpu_throttle_ns,
            effective_cpus,
            pressure_per_cpu: pressure_per_cpu
                .iter()
                .copied()
                .map(sanitize_ratio)
                .collect(),
            memory_limit_bytes,
            memory_usage_bytes,
            inactive_file_bytes,
            disk_io_bytes,
        })
    }
}

/// One logical CPU's share of a [`HardwareUtilization`].
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct CpuSnapshot {
    pub cpu_id: usize,
    /// This CPU's share of the quota pool it was queried with.
    pub quota_cpus: f64,
    pub period_ns: u64,
    pub global_cpu_count: usize,
    pub global_memory_limit: u64,
    pub global_bytes_used: u64,
    pub memory_limit: u64,
    pub memory_utilization: f64,
    /// Raw scheduler stall on this CPU.
    pub stall_ratio: f64,
    pub throttle_ratio: f64,
    /// Composite smoothed pressure, `[0, 1]`.
    pub pressure: f64,
    pub last_usage_ns: i64,
}

/// One core's share of a [`HardwareUtilization`].
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct CoreSnapshot {
    pub core_id: usize,
    pub quota_cpus: f64,
    pub period_ns: u64,
    pub global_cpu_count: usize,
    pub global_memory_limit: u64,
    pub global_bytes_used: u64,
    pub memory_limit: u64,
    pub memory_utilization: f64,
    pub effective_cpus: CpuSet,
    /// Indexed by logical CPU id (`effective_cpus.span()` entries); `None` for ids not in this core.
    pub cpu_snapshots: Vec<Option<CpuSnapshot>>,
}

/// One socket's share of a [`HardwareUtilization`].
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct SocketSnapshot {
    pub socket_id: usize,
    pub effective_cores: CpuSet,
    pub global_memory_limit: u64,
    pub global_bytes_used: u64,
    pub memory_limit: u64,
    pub memory_utilization: f64,
    /// Indexed by core id; `None` for cores with no effective CPU here.
    pub core_snapshots: Vec<Option<CoreSnapshot>>,
    pub last_usage_ns: i64,
}

/// The inputs of a [`HardwareUtilization`], before sanitizing.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct HardwareUtilizationParts {
    pub timestamp_ns: i64,
    pub quota_cpus: f64,
    pub quota_cpu_usage: f64,
    pub period_ns: u64,
    pub global_effective_cpus: CpuSet,
    pub cpu_throttle_ratio: f64,
    pub per_quota_cpu_throttle_ratio: Vec<f64>,
    pub per_quota_cpu_pressure: Vec<f64>,
    pub per_cpu_scheduler_stall: Vec<f64>,
    pub global_memory_pool: u64,
    pub per_cpu_memory_pool: u64,
    pub total_memory_utilization: f64,
    pub mem_per_cpu_usage_bytes: u64,
    pub memory_usage_bytes: u64,
    pub inactive_file_bytes: u64,
    pub disk_io_bytes_per_second: f64,
    pub disk_io_pressure: f64,
}

/// One published evaluation (upstream `HardwareUtilization`).
///
/// Per-CPU arrays are indexed by logical CPU id over the logical span.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct HardwareUtilization {
    pub timestamp_ns: i64,
    /// CPUs of quota, clamped to `[0, effective CPU count]`; fractional.
    pub quota_cpus: f64,
    /// Fraction of the quota used over the interval, `[0, 1]`.
    pub quota_cpu_usage: f64,
    pub period_ns: u64,
    pub global_effective_cpus: CpuSet,
    pub cpu_throttle_ratio: f64,
    pub per_quota_cpu_throttle_ratio: Box<[f64]>,
    /// Composite smoothed pressure per CPU, `[0, 1]`.
    pub per_quota_cpu_pressure: Box<[f64]>,
    /// Raw (unsmoothed) scheduler stall per CPU, `[0, 1]`.
    pub per_cpu_scheduler_stall: Box<[f64]>,
    /// The scope's memory limit, `u64::MAX` when unknown or unlimited.
    pub global_memory_pool: u64,
    /// The memory limit divided evenly over the effective CPUs.
    pub per_cpu_memory_pool: u64,
    /// Working set over limit, `[0, 1]`.
    pub total_memory_utilization: f64,
    /// The working set divided evenly over the effective CPUs.
    pub mem_per_cpu_usage_bytes: u64,
    pub memory_usage_bytes: u64,
    pub inactive_file_bytes: u64,
    /// I/O throughput; telemetry, not pressure.
    pub disk_io_bytes_per_second: f64,
    pub disk_io_pressure: f64,
}

impl HardwareUtilization {
    pub fn new(parts: HardwareUtilizationParts) -> Self {
        let effective = parts.global_effective_cpus.len() as f64;
        let quota_cpus = if parts.quota_cpus.is_finite() && parts.quota_cpus > 0.0 {
            parts.quota_cpus.min(effective)
        } else {
            0.0
        };
        let ratios = |values: Vec<f64>| {
            values
                .into_iter()
                .map(sanitize_ratio)
                .collect::<Box<[f64]>>()
        };
        Self {
            timestamp_ns: parts.timestamp_ns,
            quota_cpus,
            quota_cpu_usage: sanitize_ratio(parts.quota_cpu_usage),
            period_ns: parts.period_ns,
            global_effective_cpus: parts.global_effective_cpus,
            cpu_throttle_ratio: sanitize_ratio(parts.cpu_throttle_ratio),
            per_quota_cpu_throttle_ratio: ratios(parts.per_quota_cpu_throttle_ratio),
            per_quota_cpu_pressure: ratios(parts.per_quota_cpu_pressure),
            per_cpu_scheduler_stall: ratios(parts.per_cpu_scheduler_stall),
            global_memory_pool: parts.global_memory_pool,
            per_cpu_memory_pool: parts.per_cpu_memory_pool,
            total_memory_utilization: sanitize_ratio(parts.total_memory_utilization),
            mem_per_cpu_usage_bytes: parts.mem_per_cpu_usage_bytes,
            memory_usage_bytes: parts.memory_usage_bytes,
            inactive_file_bytes: parts.inactive_file_bytes,
            disk_io_bytes_per_second: sanitize_telemetry(parts.disk_io_bytes_per_second),
            disk_io_pressure: sanitize_ratio(parts.disk_io_pressure),
        }
    }

    /// The highest pressure over the effective CPUs; 1.0 when no CPU is effective.
    pub fn pressure(&self) -> f64 {
        if self.global_effective_cpus.is_empty() {
            return 1.0;
        }
        self.global_effective_cpus
            .iter()
            .filter_map(|cpu| self.per_quota_cpu_pressure.get(cpu))
            .fold(0.0, |max, &p| if p > max { p } else { max })
    }

    fn global_bytes_used(&self) -> u64 {
        saturating_product(self.global_memory_pool, self.total_memory_utilization)
    }

    /// A socket's view, from its effective core-to-CPU map (as published by
    /// [`EffectiveSocketTopology`](crate::topology::EffectiveSocketTopology)) and the quota it gets.
    ///
    /// `Ok(None)` when no core has an effective CPU or the quota is not a finite non-negative number;
    /// an error when a CPU is listed under two cores or is not effective.
    pub fn socket_snapshot(
        &self,
        socket_id: usize,
        effective_core_to_cpu: &[Option<CpuSet>],
        cpu_quota_pool: f64,
    ) -> Result<Option<SocketSnapshot>, UtilizationError> {
        if !cpu_quota_pool.is_finite() || cpu_quota_pool < 0.0 {
            return Ok(None);
        }
        let mut socket_cpus = CpuSet::new();
        let mut cores = Vec::new();
        for (core, cpus) in effective_core_to_cpu.iter().enumerate() {
            let Some(cpus) = cpus.as_ref().filter(|cpus| !cpus.is_empty()) else {
                continue;
            };
            if let Some(cpu) = socket_cpus.intersection(cpus).first() {
                return Err(UtilizationError::CpuInTwoCores { cpu });
            }
            if let Some(cpu) = cpus.difference(&self.global_effective_cpus).first() {
                return Err(UtilizationError::CoreHasNonEffectiveCpu { cpu });
            }
            socket_cpus = socket_cpus.union(cpus);
            cores.push(core);
        }
        if cores.is_empty() {
            return Ok(None);
        }
        let per_core_quota = cpu_quota_pool / cores.len() as f64;
        let mut core_snapshots = vec![None; effective_core_to_cpu.len()];
        for &core in &cores {
            let cpus = effective_core_to_cpu[core]
                .as_ref()
                .expect("selected cores have CPUs");
            core_snapshots[core] = Some(self.core_snapshot(core, cpus, per_core_quota));
        }
        let cpus = socket_cpus.len();
        let memory_limit = saturating_mul(self.per_cpu_memory_pool, cpus);
        let memory_usage = saturating_mul(self.mem_per_cpu_usage_bytes, cpus);
        Ok(Some(SocketSnapshot {
            socket_id,
            effective_cores: cores.into_iter().collect(),
            global_memory_limit: self.global_memory_pool,
            global_bytes_used: self.global_bytes_used(),
            memory_limit,
            memory_utilization: utilization_of(memory_usage, memory_limit),
            core_snapshots,
            last_usage_ns: self.timestamp_ns,
        }))
    }

    /// A core's view; `cpu_quota_pool` is split evenly over its CPUs.
    pub fn core_snapshot(
        &self,
        core_id: usize,
        effective_cpus: &CpuSet,
        cpu_quota_pool: f64,
    ) -> CoreSnapshot {
        let count = effective_cpus.len();
        let per_cpu_quota = cpu_quota_pool / count.max(1) as f64;
        let mut cpu_snapshots = vec![None; effective_cpus.span()];
        for cpu in effective_cpus {
            cpu_snapshots[cpu] = self.cpu_snapshot(cpu, per_cpu_quota);
        }
        let memory_limit = saturating_mul(self.per_cpu_memory_pool, count);
        let memory_usage = saturating_mul(self.mem_per_cpu_usage_bytes, count);
        CoreSnapshot {
            core_id,
            quota_cpus: cpu_quota_pool,
            period_ns: self.period_ns,
            global_cpu_count: self.global_effective_cpus.len(),
            global_memory_limit: self.global_memory_pool,
            global_bytes_used: self.global_bytes_used(),
            memory_limit,
            memory_utilization: utilization_of(memory_usage, memory_limit),
            effective_cpus: effective_cpus.clone(),
            cpu_snapshots,
        }
    }

    /// A CPU's view, or `None` when the CPU is not effective.
    pub fn cpu_snapshot(&self, cpu_id: usize, cpu_quota: f64) -> Option<CpuSnapshot> {
        if !self.global_effective_cpus.contains(cpu_id) {
            return None;
        }
        let at = |values: &[f64]| values.get(cpu_id).copied().unwrap_or(0.0);
        Some(CpuSnapshot {
            cpu_id,
            quota_cpus: cpu_quota,
            period_ns: self.period_ns,
            global_cpu_count: self.global_effective_cpus.len(),
            global_memory_limit: self.global_memory_pool,
            global_bytes_used: self.global_bytes_used(),
            memory_limit: self.per_cpu_memory_pool,
            memory_utilization: utilization_of(
                self.mem_per_cpu_usage_bytes,
                self.per_cpu_memory_pool,
            ),
            stall_ratio: at(&self.per_cpu_scheduler_stall),
            throttle_ratio: at(&self.per_quota_cpu_throttle_ratio),
            pressure: at(&self.per_quota_cpu_pressure),
            last_usage_ns: self.timestamp_ns,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn utilization(cpus: &[usize], pressure: Vec<f64>) -> HardwareUtilization {
        HardwareUtilization::new(HardwareUtilizationParts {
            timestamp_ns: 17,
            quota_cpus: cpus.len() as f64,
            period_ns: 100,
            global_effective_cpus: cpus.iter().copied().collect(),
            per_quota_cpu_throttle_ratio: vec![0.0; pressure.len()],
            per_cpu_scheduler_stall: vec![0.1; pressure.len()],
            per_quota_cpu_pressure: pressure,
            global_memory_pool: 1_000,
            per_cpu_memory_pool: 100,
            total_memory_utilization: 0.25,
            mem_per_cpu_usage_bytes: 25,
            ..Default::default()
        })
    }

    #[test]
    fn sanitizes_ratios_quota_and_telemetry() {
        let value = HardwareUtilization::new(HardwareUtilizationParts {
            quota_cpus: 9.0,
            quota_cpu_usage: 3.0,
            global_effective_cpus: CpuSet::range(0, 2),
            cpu_throttle_ratio: f64::NAN,
            per_quota_cpu_pressure: vec![-1.0, 2.0, f64::INFINITY],
            total_memory_utilization: -0.0,
            disk_io_bytes_per_second: f64::INFINITY,
            disk_io_pressure: 0.5,
            ..Default::default()
        });
        assert_eq!(value.quota_cpus, 2.0);
        assert_eq!(value.quota_cpu_usage, 1.0);
        assert_eq!(value.cpu_throttle_ratio, 0.0);
        assert_eq!(&*value.per_quota_cpu_pressure, &[0.0, 1.0, 0.0]);
        assert!(value.total_memory_utilization.is_sign_positive());
        assert_eq!(value.disk_io_bytes_per_second, f64::MAX);
        assert_eq!(value.disk_io_pressure, 0.5);
        let fractional = HardwareUtilization::new(HardwareUtilizationParts {
            quota_cpus: 0.5,
            global_effective_cpus: CpuSet::range(0, 4),
            ..Default::default()
        });
        assert_eq!(fractional.quota_cpus, 0.5);
    }

    #[test]
    fn pressure_is_the_effective_maximum() {
        assert_eq!(utilization(&[0, 2], vec![0.3, 0.9, 0.5]).pressure(), 0.5);
        assert_eq!(utilization(&[], vec![0.3]).pressure(), 1.0);
    }

    #[test]
    fn socket_snapshots_split_quota_and_memory_by_membership() {
        let value = utilization(&[0, 1, 3], vec![0.1, 0.2, 0.0, 0.4]);
        let map = vec![
            Some([0, 1].into_iter().collect()),
            None,
            Some([3].into_iter().collect()),
        ];
        let socket = value.socket_snapshot(4, &map, 3.0).unwrap().unwrap();
        assert_eq!(socket.effective_cores, [0, 2].into_iter().collect());
        assert_eq!((socket.memory_limit, socket.global_bytes_used), (300, 250));
        assert!((socket.memory_utilization - 0.25).abs() < 1e-12);
        let core0 = socket.core_snapshots[0].as_ref().unwrap();
        assert_eq!(core0.quota_cpus, 1.5);
        let cpu1 = core0.cpu_snapshots[1].unwrap();
        assert_eq!(
            (cpu1.quota_cpus, cpu1.pressure, cpu1.stall_ratio),
            (0.75, 0.2, 0.1)
        );
        assert!(socket.core_snapshots[1].is_none());
        assert_eq!(socket.core_snapshots[2].as_ref().unwrap().quota_cpus, 1.5);
    }

    #[test]
    fn socket_snapshots_reject_overlap_and_non_effective_members() {
        let value = utilization(&[0, 1], vec![0.0; 2]);
        let overlap = vec![Some(CpuSet::range(0, 2)), Some(CpuSet::range(1, 2))];
        assert_eq!(
            value.socket_snapshot(0, &overlap, 1.0),
            Err(UtilizationError::CpuInTwoCores { cpu: 1 })
        );
        let outside = vec![Some(CpuSet::range(1, 3))];
        assert_eq!(
            value.socket_snapshot(0, &outside, 1.0),
            Err(UtilizationError::CoreHasNonEffectiveCpu { cpu: 2 })
        );
        assert_eq!(value.socket_snapshot(0, &[None], 1.0), Ok(None));
        assert_eq!(value.socket_snapshot(0, &overlap, f64::NAN), Ok(None));
        assert!(value.cpu_snapshot(5, 1.0).is_none());
    }

    #[test]
    fn legacy_snapshots_validate_shape_and_clamp_values() {
        let snapshot = SystemSnapshot::new(
            1,
            4,
            99.0,
            100_000,
            5,
            0,
            CpuSet::range(0, 2),
            &[2.0, -1.0, 0.5, f64::NAN],
            10,
            5,
            1,
            0,
        )
        .unwrap();
        assert_eq!(snapshot.quota_cpus, 2.0);
        assert_eq!(&*snapshot.pressure_per_cpu, &[1.0, 0.0, 0.5, 0.0]);
        assert_eq!(
            SystemSnapshot::new(1, 0, 1.0, 0, 0, 0, CpuSet::new(), &[], 0, 0, 0, 0),
            Err(UtilizationError::ZeroCpus)
        );
        assert!(
            SystemSnapshot::new(1, 2, 1.0, 0, 0, 0, CpuSet::new(), &[0.0], 0, 0, 0, 0).is_err()
        );
        assert!(
            SystemSnapshot::new(
                1,
                2,
                1.0,
                0,
                0,
                0,
                CpuSet::range(0, 3),
                &[0.0; 2],
                0,
                0,
                0,
                0
            )
            .is_err()
        );
    }
}
