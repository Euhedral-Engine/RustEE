//! Windows resources (upstream `WindowsResources`).
//!
//! Inside a job object the scope is the job: its CPU rate cap is the quota, its accounting gives the
//! CPU time, and its memory limit is the hard limit. Outside one the scope is this process for CPU
//! and I/O and the host for memory. The process's I/O transfer counters are the productive I/O; they
//! include network and device I/O as well as disk. Windows reports no scheduler wait, stall,
//! throttling or reclaim, so those signals are unsupported; the power slow signal is Battery Saver.

use crate::monitor::sampling::{
    DetailedSnapshotProvider, FastHardwareSample, Signal, SignalValue, SlowHardwareSample,
    SystemSnapshotProvider,
};
use crate::monitor::{SystemSnapshot, monotonic_now_ns};
use crate::topology::Topology;
use crate::util::CpuSet;

/// One read of a value the host may not provide.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Reading<T> {
    Value(T),
    /// The host does not have it, such as a CPU cap outside a job.
    Unsupported,
    Failed,
}

impl<T: SignalValue> Reading<T> {
    fn signal(self, at: i64) -> Signal<T> {
        match self {
            Reading::Value(value) => Signal::valid(value, at),
            Reading::Unsupported => Signal::unsupported(at),
            Reading::Failed => Signal::transient_failure(at),
        }
    }
}

/// The memory limit of the scope and the usage it applies to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Memory {
    pub(crate) limit_bytes: u64,
    pub(crate) usage_bytes: u64,
}

/// The system reads behind [`WindowsResources`].
pub(crate) trait Probe: Send {
    /// The job's hard CPU rate cap, as a fraction of every processor in the system.
    fn cpu_rate_cap(&mut self) -> Reading<f64>;

    /// Cumulative CPU time of the scope, ns.
    fn cpu_time_ns(&mut self) -> Option<u64>;

    fn memory(&mut self) -> Option<Memory>;

    /// Cumulative bytes transferred by the process's I/O.
    fn io_bytes(&mut self) -> Option<u64>;

    /// The CPUs the process may run on, as logical CPU ids.
    fn process_cpus(&mut self) -> Option<CpuSet>;

    /// Whether Battery Saver is on.
    fn battery_saver(&mut self) -> Reading<bool>;
}

/// Detailed and legacy samples of this process or its job.
pub(crate) struct WindowsResources<P> {
    probe: P,
    span: usize,
    cpus: CpuSet,
}

#[cfg(windows)]
impl WindowsResources<super::system::SystemProbe> {
    pub(crate) fn new(topology: &Topology) -> Self {
        Self::with_probe(super::system::SystemProbe::new(), topology)
    }
}

impl<P: Probe> WindowsResources<P> {
    pub(crate) fn with_probe(probe: P, topology: &Topology) -> Self {
        Self {
            probe,
            span: topology.cpu_span(),
            cpus: topology.cpu_set().clone(),
        }
    }

    fn sample_fast_at(&mut self, at: i64) -> FastHardwareSample {
        let effective = self
            .probe
            .process_cpus()
            .map(|cpus| cpus.intersection(&self.cpus))
            .filter(|cpus| !cpus.is_empty())
            .unwrap_or_else(|| self.cpus.clone());
        let mut sample = FastHardwareSample::unsupported(at, self.span, effective);
        let processors = self.cpus.len() as f64;
        sample.quota_capacity_cpus = match self.probe.cpu_rate_cap() {
            Reading::Value(fraction) => Reading::Value(fraction * processors),
            other => other,
        }
        .signal(at);
        let counter = |value: Option<u64>| {
            value.map_or_else(|| Signal::transient_failure(at), |v| Signal::valid(v, at))
        };
        sample.productive_cpu_ns = counter(self.probe.cpu_time_ns());
        sample.io_signals.productive_bytes = counter(self.probe.io_bytes());
        let memory = self.probe.memory();
        sample.memory_signals.hard_limit_bytes = counter(memory.map(|memory| memory.limit_bytes));
        sample.memory_signals.usage_bytes = counter(memory.map(|memory| memory.usage_bytes));
        sample
    }

    fn sample_slow_at(&mut self, at: i64) -> SlowHardwareSample {
        let mut sample = SlowHardwareSample::unsupported(at, self.span);
        sample.system_signals.low_power_mode = self.probe.battery_saver().signal(at);
        sample
    }
}

impl<P: Probe> DetailedSnapshotProvider for WindowsResources<P> {
    fn sample_fast(&mut self, requested_at_ns: i64) -> Option<FastHardwareSample> {
        Some(self.sample_fast_at(requested_at_ns))
    }

    fn sample_slow(&mut self, requested_at_ns: i64) -> Option<SlowHardwareSample> {
        Some(self.sample_slow_at(requested_at_ns))
    }
}

/// Snapshots carry cumulative values only, so they need no interval state of their own.
impl<P: Probe> SystemSnapshotProvider for WindowsResources<P> {
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

    struct Fake {
        cap: Reading<f64>,
        cpu: Option<u64>,
        memory: Option<Memory>,
        io: Option<u64>,
        cpus: Option<CpuSet>,
        saver: Reading<bool>,
    }

    impl Default for Fake {
        fn default() -> Self {
            Self {
                cap: Reading::Unsupported,
                cpu: Some(0),
                memory: Some(Memory {
                    limit_bytes: 0,
                    usage_bytes: 0,
                }),
                io: Some(0),
                cpus: None,
                saver: Reading::Unsupported,
            }
        }
    }

    impl Probe for Fake {
        fn cpu_rate_cap(&mut self) -> Reading<f64> {
            self.cap
        }
        fn cpu_time_ns(&mut self) -> Option<u64> {
            self.cpu
        }
        fn memory(&mut self) -> Option<Memory> {
            self.memory
        }
        fn io_bytes(&mut self) -> Option<u64> {
            self.io
        }
        fn process_cpus(&mut self) -> Option<CpuSet> {
            self.cpus.clone()
        }
        fn battery_saver(&mut self) -> Reading<bool> {
            self.saver
        }
    }

    fn resources(fake: Fake) -> WindowsResources<Fake> {
        WindowsResources::with_probe(fake, &source::fallback(8))
    }

    #[test]
    fn a_job_cap_is_a_fractional_quota_of_every_processor() {
        let mut resources = resources(Fake {
            cap: Reading::Value(0.0625),
            cpu: Some(2_000),
            memory: Some(Memory {
                limit_bytes: 1 << 30,
                usage_bytes: 1 << 29,
            }),
            io: Some(512),
            cpus: Some(CpuSet::range(2, 12)),
            ..Fake::default()
        });
        let sample = resources.sample_fast(5).unwrap();
        assert_eq!(sample.check_shape(), Ok(()));
        assert_eq!(sample.quota_capacity_cpus.get(), Some(0.5));
        assert_eq!(sample.effective_cpus, CpuSet::range(2, 8));
        assert_eq!(sample.productive_cpu_ns.get(), Some(2_000));
        assert_eq!(sample.io_signals.productive_bytes.get(), Some(512));
        assert_eq!(sample.memory_signals.hard_limit_bytes.get(), Some(1 << 30));
        assert_eq!(sample.memory_signals.usage_bytes.get(), Some(1 << 29));
        assert_eq!(
            sample.scope_quota_throttled_ns.validity(),
            SignalValidity::Unsupported
        );
        assert_eq!(
            sample.quota_period_ns.validity(),
            SignalValidity::Unsupported
        );
        assert!(sample.cpu_signals.iter().all(
            |cpu| cpu.reported_scheduler_stall_ratio.validity() == SignalValidity::Unsupported
        ));

        let snapshot = resources.snapshot().unwrap();
        assert_eq!((snapshot.quota_cpus, snapshot.cpu_usage_ns), (0.5, 2_000));
    }

    #[test]
    fn without_a_job_the_quota_is_unsupported_and_cpus_default_to_the_topology() {
        let mut resources = resources(Fake::default());
        let sample = resources.sample_fast(1).unwrap();
        assert_eq!(
            sample.quota_capacity_cpus.validity(),
            SignalValidity::Unsupported
        );
        assert_eq!(sample.effective_cpus, CpuSet::range(0, 8));
        assert_eq!(resources.snapshot().unwrap().quota_cpus, 8.0);
    }

    #[test]
    fn failed_reads_are_transient_failures() {
        let mut resources = resources(Fake {
            cap: Reading::Failed,
            cpu: None,
            memory: None,
            io: None,
            ..Fake::default()
        });
        let sample = resources.sample_fast(1).unwrap();
        for validity in [
            sample.quota_capacity_cpus.validity(),
            sample.productive_cpu_ns.validity(),
            sample.io_signals.productive_bytes.validity(),
            sample.memory_signals.hard_limit_bytes.validity(),
            sample.memory_signals.usage_bytes.validity(),
        ] {
            assert_eq!(validity, SignalValidity::TransientFailure);
        }
        assert!(resources.snapshot().is_none());
    }

    #[test]
    fn battery_saver_is_the_low_power_signal() {
        let mut resources = resources(Fake {
            saver: Reading::Value(true),
            ..Fake::default()
        });
        let sample = resources.sample_slow(2).unwrap();
        assert_eq!(sample.check_shape(), Ok(()));
        assert_eq!(sample.system_signals.low_power_mode.get(), Some(true));
        assert_eq!(
            sample.system_signals.thermal_severity.validity(),
            SignalValidity::Unsupported
        );
    }
}
