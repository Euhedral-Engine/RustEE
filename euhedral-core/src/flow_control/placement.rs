//! A placement policy on a vertex: one callback that runs on every frame before the vertex routes it.
//!
//! The policy sees the frame and the CPUs the lattice routes with now, and says where the frame goes by setting its
//! origin and routing policy; the routing that follows honours them. It is one callback, not a registry, and it is
//! set while nothing routes.
//!
//! The policy runs on the frame path, from every thread that routes at once. It must be re-entrant, take no lock,
//! not allocate and not block; whatever it reads is the caller's to make safe to read from many threads.
//!
//! The policy and the CPU set are cells of the lattice's registry, so a thread reads them from its view with no lock
//! and no reference-count update (a light reader pays the reference-count pair a view read costs). A vertex without a
//! placement pays a branch; one with a placement and no policy set pays a branch and a relaxed load.

use super::registry::UpstreamRegistry;
use super::views::Cell;
use crate::frames::Frame;
use euhedral_hardware_utils::util::CpuSet;
use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::atomic::{AtomicBool, Ordering};

/// What a placement policy is: it may change the frame's origin and routing policy, given the CPUs the lattice routes
/// with. Called concurrently from many threads: `Fn`, never `FnMut`.
pub type PlacementPolicy = dyn Fn(&mut dyn Frame, &CpuSet) + Send + Sync + 'static;

/// A policy in a cell.
struct PolicyBox(Box<PlacementPolicy>);

static NO_CPUS: LazyLock<CpuSet> = LazyLock::new(CpuSet::new);

/// The placement of one vertex: its policy and the CPUs it is given.
pub struct Placement {
    /// Whether a policy is set: the only thing a frame reads when there is none.
    active: AtomicBool,
    policy: Cell<PolicyBox>,
    cpus: Cell<CpuSet>,
}

impl Placement {
    /// A placement with no policy.
    pub fn new(registry: &UpstreamRegistry) -> Arc<Self> {
        Arc::new(Self {
            active: AtomicBool::new(false),
            policy: registry.cell(),
            cpus: registry.cell(),
        })
    }

    /// Sets the policy. Lifecycle work, for the caller to do while nothing routes.
    pub fn set_policy(
        &self,
        registry: &UpstreamRegistry,
        policy: impl Fn(&mut dyn Frame, &CpuSet) + Send + Sync + 'static,
    ) {
        registry.set_cell(&self.policy, Some(Arc::new(PolicyBox(Box::new(policy)))));
        self.active.store(true, Ordering::Release);
    }

    /// Publishes the CPUs the policy is given from now on.
    pub fn publish_cpus(&self, registry: &UpstreamRegistry, cpus: CpuSet) {
        registry.set_cell(&self.cpus, Some(Arc::new(cpus)));
    }

    /// Runs the policy on `frame`, if one is set.
    #[inline]
    pub fn apply(&self, registry: &Arc<UpstreamRegistry>, frame: &mut dyn Frame) {
        if !self.active.load(Ordering::Relaxed) {
            return;
        }
        registry.with_view(|view| {
            if let Some(policy) = view.get(&self.policy) {
                (policy.0)(frame, view.get(&self.cpus).unwrap_or(&NO_CPUS));
            }
        });
    }
}

impl std::fmt::Debug for Placement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Placement")
            .field("active", &self.active.load(Ordering::Relaxed))
            .finish_non_exhaustive()
    }
}
