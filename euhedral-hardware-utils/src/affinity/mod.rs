//! Thread affinity and pinned executors (upstream `affinity` package, `ThreadTools` and
//! `ThreadPinner`).
//!
//! The free functions here act on the calling thread through one process-wide
//! [`AffinityController`] built lazily over [`Topology::system`] and this platform's provider.

mod controller;
mod executor;
pub mod masks;

pub use controller::{AffinityController, ControllerError, ManagedCpuBinding};
pub use executor::{ExecutorError, PinnedThreadExecutor, Rejected, is_interrupted};

use crate::topology::{CpuInfo, Topology};
use crate::util::CpuSet;
use std::sync::OnceLock;

/// What the affinity path can apply and later undo honestly.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AffinityCapability {
    /// Applies the complete requested mask and restores the thread's original mask.
    Exact,
    /// Applies a scheduler placement preference only, with no promise of an exact CPU.
    LocalityHint,
    /// Changes nothing; requests return `false`.
    Unsupported,
}

/// Platform affinity operations on the calling thread (upstream `AffinityProvider`).
///
/// Masks are little-endian words whose bit indexes are logical CPU ids; the controller validates them
/// before a provider sees them.
pub trait AffinityProvider: Send + Sync {
    fn capability(&self) -> AffinityCapability;

    /// The calling thread's logical CPU, when it can be determined exactly.
    fn current_cpu(&self) -> Option<usize> {
        None
    }

    /// The calling thread's exact mask.
    fn capture_affinity(&self) -> Option<Vec<u64>> {
        None
    }

    /// Applies every CPU of `mask` at once.
    fn apply_exact(&self, _mask: &[u64]) -> bool {
        false
    }

    /// Restores a mask returned by [`capture_affinity`](Self::capture_affinity), unvalidated.
    fn restore_exact(&self, _mask: &[u64]) -> bool {
        false
    }

    /// The positive scheduler-locality tag for a logical CPU.
    fn locality_for_cpu(&self, _cpu: usize) -> Option<u32> {
        None
    }

    fn apply_locality(&self, _locality: u32) -> bool {
        false
    }

    /// Clears the calling thread's locality preference.
    fn release_locality(&self) -> bool {
        false
    }

    /// Requests a timer granularity, in nanoseconds; `false` when the platform cannot.
    fn set_timer_resolution(&self, nanos: u64) -> bool;
}

/// The process-wide controller.
pub fn controller() -> &'static AffinityController {
    static CONTROLLER: OnceLock<AffinityController> = OnceLock::new();
    CONTROLLER.get_or_init(|| {
        let topology = Topology::system();
        let span = topology.cpu_span().max(1);
        let provider = crate::platform::affinity_provider();
        AffinityController::new(provider, topology.cpu_set(), span)
            .or_else(|error| {
                log::debug!("affinity disabled: {error}");
                AffinityController::new(None, &CpuSet::range(0, span), span)
            })
            .expect("a dense supported set is valid")
    })
}

/// The affinity behaviour available here (upstream `ThreadTools.getAffinityCapability`).
pub fn capability() -> AffinityCapability {
    controller().capability()
}

/// The thread's mask when the controller was built, or the topology's CPUs (upstream
/// `ThreadTools.BASE_MASK`).
pub fn base_mask() -> &'static CpuSet {
    controller().base_mask()
}

/// The calling thread's logical CPU (upstream `getCpu`).
pub fn current_cpu() -> Option<usize> {
    controller().current_cpu()
}

/// The topology record of the calling thread's CPU.
pub fn current_cpu_info() -> Option<&'static CpuInfo> {
    current_cpu().and_then(|cpu| Topology::system().cpu_info(cpu))
}

/// Pins the calling thread to the CPU it runs on now.
pub fn set_affinity_current() -> bool {
    controller().set_affinity_current()
}

pub fn set_affinity(cpu: usize) -> bool {
    controller().set_affinity(cpu)
}

pub fn set_affinity_cpus(cpus: &[usize]) -> bool {
    controller().set_affinity_cpus(cpus)
}

pub fn set_affinity_set(cpus: &CpuSet) -> bool {
    controller().set_affinity_set(cpus)
}

pub fn set_affinity_mask(mask: &[u64]) -> bool {
    controller().set_affinity_mask(mask)
}

/// Restores the calling thread's binding from before its first request.
pub fn release_affinity() {
    controller().release_affinity();
}

/// Requests a timer granularity: Linux timer slack for the calling thread, the process timer
/// resolution on Windows. macOS has no equivalent and returns `false`.
pub fn set_timer_resolution(nanos: u64) -> bool {
    controller().set_timer_resolution(nanos)
}
