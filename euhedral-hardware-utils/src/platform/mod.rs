//! Platform back ends, selected at compile time.

#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
pub(crate) mod linux;

#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(crate) mod macos;

#[cfg_attr(not(windows), allow(dead_code))]
pub(crate) mod windows;

use crate::affinity::AffinityProvider;
use crate::monitor::SystemSnapshot;
use crate::monitor::sampling::{
    DetailedSnapshotProvider, FastHardwareSample, Signal, SignalValidity, SignalValue,
    SystemSnapshotProvider,
};
use crate::topology::{Topology, source};

fn processor_count() -> usize {
    std::thread::available_parallelism().map_or(1, |n| n.get())
}

/// The legacy snapshot of a fast sample, with `pressure` as its per-CPU pressure.
///
/// A snapshot cannot mark one value as failed, and a zero in place of a cumulative counter would
/// reset its baseline, so a read in which any carried signal failed gives no snapshot. Unsupported
/// signals take their neutral value.
pub(crate) fn legacy_snapshot(
    sample: &FastHardwareSample,
    pressure: &[f64],
) -> Option<SystemSnapshot> {
    fn read<T: SignalValue>(signal: Signal<T>, unsupported: T) -> Option<T> {
        match signal.validity() {
            SignalValidity::Valid => Some(signal.value()),
            SignalValidity::Unsupported => Some(unsupported),
            SignalValidity::TransientFailure => None,
        }
    }
    let memory = &sample.memory_signals;
    SystemSnapshot::new(
        sample.observed_at_ns,
        sample.logical_span,
        read(
            sample.quota_capacity_cpus,
            sample.effective_cpus.len() as f64,
        )?,
        read(sample.quota_period_ns, 0)?,
        read(sample.productive_cpu_ns, 0)?,
        read(sample.scope_quota_throttled_ns, 0)?,
        sample.effective_cpus.clone(),
        pressure,
        read(memory.hard_limit_bytes, u64::MAX)?,
        read(memory.usage_bytes, 0)?,
        read(memory.inactive_file_bytes, 0)?,
        read(sample.io_signals.productive_bytes, 0)?,
    )
    .ok()
}

pub(crate) fn discover_topology() -> Topology {
    #[cfg(target_os = "linux")]
    return linux::layout::discover(std::path::Path::new("/sys"));
    #[cfg(target_os = "macos")]
    return macos::layout::discover(&macos::system::SystemSysctl, processor_count());
    #[cfg(windows)]
    return windows::layout::discover(
        windows::system::logical_processor_information().as_deref(),
        processor_count(),
    );
    #[allow(unreachable_code)]
    source::fallback(processor_count())
}

pub(crate) fn detailed_provider(topology: &Topology) -> Option<Box<dyn DetailedSnapshotProvider>> {
    #[cfg(target_os = "linux")]
    return Some(Box::new(linux::resources::LinuxResources::new(topology)));
    #[cfg(target_os = "macos")]
    return Some(Box::new(macos::resources::MacosResources::new(topology)));
    #[cfg(windows)]
    return Some(Box::new(windows::resources::WindowsResources::new(
        topology,
    )));
    #[allow(unreachable_code)]
    {
        let _ = topology;
        None
    }
}

pub(crate) fn snapshot_provider() -> Option<Box<dyn SystemSnapshotProvider>> {
    #[cfg(target_os = "linux")]
    return Some(Box::new(linux::resources::LinuxSnapshots::new(
        linux::resources::LinuxResources::new(Topology::system()),
    )));
    #[cfg(target_os = "macos")]
    return Some(Box::new(macos::resources::MacosResources::new(
        Topology::system(),
    )));
    #[cfg(windows)]
    return Some(Box::new(windows::resources::WindowsResources::new(
        Topology::system(),
    )));
    #[allow(unreachable_code)]
    None
}

pub(crate) fn affinity_provider() -> Option<Box<dyn AffinityProvider>> {
    #[cfg(target_os = "linux")]
    return Some(Box::new(linux::affinity::LinuxAffinity::new(
        Topology::system().cpu_span(),
    )));
    #[cfg(target_os = "macos")]
    return Some(Box::new(macos::affinity::MacosAffinity));
    #[cfg(windows)]
    return Some(Box::new(windows::affinity::WindowsAffinity::new()));
    #[allow(unreachable_code)]
    None
}

#[cfg(test)]
pub(crate) mod testing {
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A fresh directory removed on drop.
    pub(crate) struct TempDir(PathBuf);

    impl TempDir {
        pub(crate) fn new(name: &str) -> Self {
            static NEXT: AtomicUsize = AtomicUsize::new(0);
            let unique = NEXT.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "euhedral-hw-{name}-{}-{unique}",
                std::process::id()
            ));
            let _ = std::fs::remove_dir_all(&path);
            std::fs::create_dir_all(&path).unwrap();
            Self(path)
        }

        pub(crate) fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// Writes `content` to `root/relative`, creating parent directories.
    pub(crate) fn write(root: &Path, relative: &str, content: &str) {
        let path = root.join(relative);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, content).unwrap();
    }
}
