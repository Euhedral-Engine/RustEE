//! Adapts a legacy [`SystemSnapshotProvider`] to the detailed sampling interface (upstream
//! `SystemSnapshotCompatibilityAdapter`).
//!
//! Snapshots are read in the canonical form: periods are nanoseconds, counters are cumulative nanoseconds and bytes, and the
//! per-CPU pressure is a reported scheduler-stall ratio.

use super::sampling::{
    CpuFastSignals, DetailedSnapshotProvider, FastHardwareSample, Signal, SlowHardwareSample,
    SystemSnapshotProvider,
};

pub(crate) struct LegacyAdapter<P> {
    delegate: P,
}

impl<P: SystemSnapshotProvider> LegacyAdapter<P> {
    pub(crate) fn new(delegate: P) -> Self {
        Self { delegate }
    }
}

impl<P: SystemSnapshotProvider> DetailedSnapshotProvider for LegacyAdapter<P> {
    /// A failed snapshot is a failed sample, so the engine keeps its membership and baselines.
    fn sample_fast(&mut self, _requested_at_ns: i64) -> Option<FastHardwareSample> {
        let snapshot = self.delegate.snapshot()?;
        let at = snapshot.time_ns;
        let mut sample = FastHardwareSample::unsupported(
            at,
            snapshot.total_cpus,
            snapshot.effective_cpus.clone(),
        );
        sample.quota_capacity_cpus = Signal::valid(snapshot.quota_cpus, at);
        sample.quota_period_ns = match snapshot.period_ns {
            0 => Signal::unsupported(at),
            period => Signal::valid(period, at),
        };
        sample.productive_cpu_ns = Signal::valid(snapshot.cpu_usage_ns, at);
        sample.scope_quota_throttled_ns = Signal::valid(snapshot.cpu_throttle_ns, at);
        sample.cpu_signals = snapshot
            .pressure_per_cpu
            .iter()
            .map(|&pressure| CpuFastSignals {
                reported_scheduler_stall_ratio: Signal::valid(pressure, at),
                ..CpuFastSignals::unsupported(at)
            })
            .collect();
        let memory = &mut sample.memory_signals;
        memory.hard_limit_bytes = match snapshot.memory_limit_bytes {
            u64::MAX => Signal::unsupported(at),
            limit => Signal::valid(limit, at),
        };
        memory.usage_bytes = Signal::valid(snapshot.memory_usage_bytes, at);
        memory.inactive_file_bytes = Signal::valid(snapshot.inactive_file_bytes, at);
        sample.io_signals.productive_bytes = Signal::valid(snapshot.disk_io_bytes, at);
        Some(sample)
    }

    /// Legacy providers have no slow signals; returning `None` keeps the slow cache empty without
    /// calling the delegate.
    fn sample_slow(&mut self, _requested_at_ns: i64) -> Option<SlowHardwareSample> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::monitor::SystemSnapshot;
    use crate::util::CpuSet;

    struct Scripted(Vec<Option<SystemSnapshot>>);

    impl SystemSnapshotProvider for Scripted {
        fn snapshot(&mut self) -> Option<SystemSnapshot> {
            self.0.remove(0)
        }
    }

    fn snapshot(memory_limit: u64, period: u64) -> SystemSnapshot {
        SystemSnapshot::new(
            1_000,
            2,
            1.5,
            period,
            300,
            40,
            CpuSet::range(0, 2),
            &[0.25, 0.5],
            memory_limit,
            700,
            100,
            9_000,
        )
        .unwrap()
    }

    #[test]
    fn maps_the_canonical_profile() {
        let mut adapter = LegacyAdapter::new(Scripted(vec![Some(snapshot(4_096, 100_000))]));
        let sample = adapter.sample_fast(5).unwrap();
        assert_eq!(sample.check_shape(), Ok(()));
        assert_eq!((sample.observed_at_ns, sample.logical_span), (1_000, 2));
        assert_eq!(sample.quota_capacity_cpus.get(), Some(1.5));
        assert_eq!(sample.quota_period_ns.get(), Some(100_000));
        assert_eq!(sample.productive_cpu_ns.get(), Some(300));
        assert_eq!(sample.scope_quota_throttled_ns.get(), Some(40));
        assert_eq!(
            sample.cpu_signals[1].reported_scheduler_stall_ratio.get(),
            Some(0.5)
        );
        assert_eq!(sample.cpu_signals[1].scheduler_wait.get(), None);
        assert_eq!(sample.memory_signals.hard_limit_bytes.get(), Some(4_096));
        assert_eq!(sample.memory_signals.usage_bytes.get(), Some(700));
        assert_eq!(sample.memory_signals.inactive_file_bytes.get(), Some(100));
        assert_eq!(sample.io_signals.productive_bytes.get(), Some(9_000));
        assert!(adapter.sample_slow(5).is_none());
    }

    #[test]
    fn unlimited_memory_and_zero_period_are_unsupported() {
        let mut adapter = LegacyAdapter::new(Scripted(vec![Some(snapshot(u64::MAX, 0))]));
        let sample = adapter.sample_fast(5).unwrap();
        assert_eq!(sample.memory_signals.hard_limit_bytes.get(), None);
        assert_eq!(sample.quota_period_ns.get(), None);
    }

    #[test]
    fn a_failed_snapshot_is_a_failed_sample() {
        let mut adapter = LegacyAdapter::new(Scripted(vec![None]));
        assert!(adapter.sample_fast(5).is_none());
    }
}
