//! macOS resources (upstream `MacosResources`).
//!
//! The process's CPU time and disk I/O come from `proc_pid_rusage`, host memory from
//! `host_statistics64`, and the thermal state and Low Power Mode from `NSProcessInfo`. macOS has no
//! CPU quota, per-CPU wait or stall accounting, so those signals are unsupported.

use crate::monitor::sampling::{
    DetailedSnapshotProvider, FastHardwareSample, Signal, SlowHardwareSample,
    SystemSnapshotProvider, ThermalSeverity,
};
use crate::monitor::{SystemSnapshot, monotonic_now_ns};
use crate::topology::Topology;
use crate::util::CpuSet;

/// Cumulative usage of this process.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ProcessUsage {
    pub(crate) cpu_ns: u64,
    /// Bytes read and written by disk I/O, when the source reports them.
    pub(crate) io_bytes: Option<u64>,
}

/// Physical memory of the host.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct HostMemory {
    pub(crate) total_bytes: u64,
    /// Everything but free pages.
    pub(crate) used_bytes: u64,
    /// The part of `used_bytes` the kernel can reclaim without paging: inactive and speculative
    /// pages.
    pub(crate) reclaimable_bytes: u64,
}

impl HostMemory {
    /// Host memory from `vm_statistics64` page counts.
    pub(crate) fn from_pages(
        total_bytes: u64,
        page_bytes: u64,
        free: u64,
        inactive: u64,
        speculative: u64,
    ) -> Self {
        let used_bytes = total_bytes.saturating_sub(free.saturating_mul(page_bytes));
        Self {
            total_bytes,
            used_bytes,
            reclaimable_bytes: inactive
                .saturating_add(speculative)
                .saturating_mul(page_bytes)
                .min(used_bytes),
        }
    }
}

/// Mach absolute-time ticks to nanoseconds, saturating.
pub(crate) fn ticks_to_ns(ticks: u64, numer: u32, denom: u32) -> u64 {
    let nanos = u128::from(ticks) * u128::from(numer) / u128::from(denom.max(1));
    u64::try_from(nanos).unwrap_or(u64::MAX)
}

/// The system reads behind [`MacosResources`] (upstream `MacosResourceProbe`).
pub(crate) trait Probe: Send {
    /// `None` when the read failed.
    fn process_usage(&mut self) -> Option<ProcessUsage>;

    /// `None` when the read failed.
    fn host_memory(&mut self) -> Option<HostMemory>;

    /// `NSProcessInfo.thermalState`, `0` (nominal) to `3` (critical); `None` when unavailable.
    fn thermal_state(&mut self) -> Option<i64>;

    /// `NSProcessInfo.isLowPowerModeEnabled`; `None` when unavailable.
    fn low_power_mode(&mut self) -> Option<bool>;
}

fn thermal_severity(state: i64) -> Option<ThermalSeverity> {
    Some(match state {
        0 => ThermalSeverity::Nominal,
        1 => ThermalSeverity::Fair,
        2 => ThermalSeverity::Serious,
        3 => ThermalSeverity::Critical,
        _ => return None,
    })
}

/// Detailed and legacy samples of this process on a macOS host.
pub(crate) struct MacosResources<P> {
    probe: P,
    span: usize,
    cpus: CpuSet,
}

#[cfg(target_os = "macos")]
impl MacosResources<super::system::SystemProbe> {
    pub(crate) fn new(topology: &Topology) -> Self {
        Self::with_probe(super::system::SystemProbe::new(), topology)
    }
}

impl<P: Probe> MacosResources<P> {
    pub(crate) fn with_probe(probe: P, topology: &Topology) -> Self {
        Self {
            probe,
            span: topology.cpu_span(),
            cpus: topology.cpu_set().clone(),
        }
    }

    fn sample_fast_at(&mut self, at: i64) -> FastHardwareSample {
        let mut sample = FastHardwareSample::unsupported(at, self.span, self.cpus.clone());
        match self.probe.process_usage() {
            Some(usage) => {
                sample.productive_cpu_ns = Signal::valid(usage.cpu_ns, at);
                sample.io_signals.productive_bytes = usage.io_bytes.map_or_else(
                    || Signal::transient_failure(at),
                    |bytes| Signal::valid(bytes, at),
                );
            }
            None => {
                sample.productive_cpu_ns = Signal::transient_failure(at);
                sample.io_signals.productive_bytes = Signal::transient_failure(at);
            }
        }
        let memory = &mut sample.memory_signals;
        match self.probe.host_memory() {
            Some(host) => {
                memory.hard_limit_bytes = Signal::valid(host.total_bytes, at);
                memory.usage_bytes = Signal::valid(host.used_bytes, at);
                memory.inactive_file_bytes = Signal::valid(host.reclaimable_bytes, at);
            }
            None => {
                memory.hard_limit_bytes = Signal::transient_failure(at);
                memory.usage_bytes = Signal::transient_failure(at);
                memory.inactive_file_bytes = Signal::transient_failure(at);
            }
        }
        sample
    }

    fn sample_slow_at(&mut self, at: i64) -> SlowHardwareSample {
        let mut sample = SlowHardwareSample::unsupported(at, self.span);
        let system = &mut sample.system_signals;
        system.thermal_severity = match self.probe.thermal_state() {
            Some(state) => thermal_severity(state).map_or_else(
                || Signal::transient_failure(at),
                |severity| Signal::valid(severity, at),
            ),
            None => Signal::unsupported(at),
        };
        system.low_power_mode = self.probe.low_power_mode().map_or_else(
            || Signal::unsupported(at),
            |enabled| Signal::valid(enabled, at),
        );
        sample
    }
}

impl<P: Probe> DetailedSnapshotProvider for MacosResources<P> {
    fn sample_fast(&mut self, requested_at_ns: i64) -> Option<FastHardwareSample> {
        Some(self.sample_fast_at(requested_at_ns))
    }

    fn sample_slow(&mut self, requested_at_ns: i64) -> Option<SlowHardwareSample> {
        Some(self.sample_slow_at(requested_at_ns))
    }
}

/// Snapshots carry cumulative values only, so they need no interval state of their own.
impl<P: Probe> SystemSnapshotProvider for MacosResources<P> {
    fn snapshot(&mut self) -> Option<SystemSnapshot> {
        let sample = self.sample_fast_at(monotonic_now_ns());
        crate::platform::legacy_snapshot(&sample, &vec![0.0; sample.logical_span])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::monitor::sampling::SignalValidity;
    use crate::topology::source;

    #[derive(Default)]
    struct Fake {
        usage: Option<ProcessUsage>,
        memory: Option<HostMemory>,
        thermal: Option<i64>,
        low_power: Option<bool>,
    }

    impl Probe for Fake {
        fn process_usage(&mut self) -> Option<ProcessUsage> {
            self.usage
        }
        fn host_memory(&mut self) -> Option<HostMemory> {
            self.memory
        }
        fn thermal_state(&mut self) -> Option<i64> {
            self.thermal
        }
        fn low_power_mode(&mut self) -> Option<bool> {
            self.low_power
        }
    }

    fn resources(fake: Fake) -> MacosResources<Fake> {
        MacosResources::with_probe(fake, &source::fallback(4))
    }

    #[test]
    fn ticks_convert_through_the_timebase() {
        assert_eq!(ticks_to_ns(3, 125, 3), 125);
        assert_eq!(ticks_to_ns(1_000_000_000, 1, 1), 1_000_000_000);
        assert_eq!(ticks_to_ns(u64::MAX, 125, 3), u64::MAX);
        assert_eq!(ticks_to_ns(7, 1, 0), 7);
    }

    #[test]
    fn host_memory_counts_inactive_pages_as_reclaimable() {
        let memory = HostMemory::from_pages(100 * 16_384, 16_384, 10, 20, 5);
        assert_eq!(memory.used_bytes, 90 * 16_384);
        assert_eq!(memory.reclaimable_bytes, 25 * 16_384);
        let overcounted = HostMemory::from_pages(10, 1, 8, 9, 9);
        assert_eq!(
            (overcounted.used_bytes, overcounted.reclaimable_bytes),
            (2, 2)
        );
    }

    #[test]
    fn fast_samples_carry_cpu_io_and_host_memory() {
        let mut resources = resources(Fake {
            usage: Some(ProcessUsage {
                cpu_ns: 1_000_000_000,
                io_bytes: Some(4_096),
            }),
            memory: Some(HostMemory::from_pages(1_000, 1, 100, 200, 0)),
            ..Fake::default()
        });
        let sample = resources.sample_fast(7).unwrap();
        assert_eq!(sample.check_shape(), Ok(()));
        assert_eq!(sample.effective_cpus, CpuSet::range(0, 4));
        assert_eq!(sample.productive_cpu_ns.get(), Some(1_000_000_000));
        assert_eq!(sample.io_signals.productive_bytes.get(), Some(4_096));
        assert_eq!(sample.memory_signals.hard_limit_bytes.get(), Some(1_000));
        assert_eq!(sample.memory_signals.usage_bytes.get(), Some(900));
        assert_eq!(sample.memory_signals.inactive_file_bytes.get(), Some(200));
        assert_eq!(
            sample.quota_capacity_cpus.validity(),
            SignalValidity::Unsupported
        );
        assert_eq!(
            sample.scope_quota_throttled_ns.validity(),
            SignalValidity::Unsupported
        );

        let snapshot = resources.snapshot().unwrap();
        assert_eq!(snapshot.quota_cpus, 4.0);
        assert_eq!(
            (snapshot.cpu_usage_ns, snapshot.disk_io_bytes),
            (1_000_000_000, 4_096)
        );
        assert_eq!(snapshot.memory_limit_bytes, 1_000);
    }

    #[test]
    fn failed_reads_are_transient_failures_and_fail_the_snapshot() {
        let mut resources = resources(Fake {
            usage: Some(ProcessUsage {
                cpu_ns: 5,
                io_bytes: None,
            }),
            ..Fake::default()
        });
        let sample = resources.sample_fast(1).unwrap();
        assert_eq!(sample.productive_cpu_ns.get(), Some(5));
        assert_eq!(
            sample.io_signals.productive_bytes.validity(),
            SignalValidity::TransientFailure
        );
        assert_eq!(
            sample.memory_signals.usage_bytes.validity(),
            SignalValidity::TransientFailure
        );
        assert!(resources.snapshot().is_none());
    }

    #[test]
    fn slow_samples_map_the_thermal_state_and_low_power_mode() {
        let mut resources = resources(Fake {
            thermal: Some(2),
            low_power: Some(true),
            ..Fake::default()
        });
        let sample = resources.sample_slow(3).unwrap();
        assert_eq!(sample.check_shape(), Ok(()));
        assert_eq!(
            sample.system_signals.thermal_severity.get(),
            Some(ThermalSeverity::Serious)
        );
        assert_eq!(sample.system_signals.low_power_mode.get(), Some(true));

        resources.probe.thermal = Some(9);
        resources.probe.low_power = None;
        let sample = resources.sample_slow(4).unwrap();
        assert_eq!(
            sample.system_signals.thermal_severity.validity(),
            SignalValidity::TransientFailure
        );
        assert_eq!(
            sample.system_signals.low_power_mode.validity(),
            SignalValidity::Unsupported
        );
        resources.probe.thermal = None;
        let thermal = resources
            .sample_slow(5)
            .unwrap()
            .system_signals
            .thermal_severity;
        assert_eq!(thermal.validity(), SignalValidity::Unsupported);
    }
}
