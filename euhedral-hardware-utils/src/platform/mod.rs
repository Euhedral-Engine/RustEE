//! Platform back ends, selected at compile time.

use crate::affinity::AffinityProvider;
use crate::monitor::sampling::{DetailedSnapshotProvider, SystemSnapshotProvider};
use crate::topology::{Topology, source};

fn processor_count() -> usize {
    std::thread::available_parallelism().map_or(1, |n| n.get())
}

pub(crate) fn discover_topology() -> Topology {
    source::fallback(processor_count())
}

pub(crate) fn detailed_provider(topology: &Topology) -> Option<Box<dyn DetailedSnapshotProvider>> {
    let _ = topology;
    None
}

pub(crate) fn snapshot_provider() -> Option<Box<dyn SystemSnapshotProvider>> {
    None
}

pub(crate) fn affinity_provider() -> Option<Box<dyn AffinityProvider>> {
    None
}
