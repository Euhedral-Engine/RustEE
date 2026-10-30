//! A placement policy on a vertex: one callback that runs on every frame before the vertex routes it.
//!
//! The policy sees the frame and the CPUs the lattice routes with now, and says where the frame goes by setting its
//! origin and routing policy; the routing that follows honours them. It is one callback, not a registry, and it is
//! set while nothing routes.
//!
//! The policy runs on the frame path, from every thread that routes at once. It must be re-entrant, take no lock,
//! not allocate and not block; whatever it reads is the caller's to make safe to read from many threads.
//!
//! The policy and the CPU set are cells of the lattice's registry, so a thread reads them from its view with no lock.
//! A worker (a full reader) pays nothing for either. A light reader (a producer thread that is not registered) upgrades
//! one weak reference per cell, so with a policy set it pays two extra reference-count pairs per frame, one for the
//! policy and one for the CPU set. A vertex without a policy set pays one relaxed load of its own flag per frame.

use super::registry::UpstreamRegistry;
use super::views::{Cell, View};
use crate::frames::Frame;
use euhedral_hardware_utils::util::CpuSet;
use std::sync::Arc;
use std::sync::LazyLock;

/// What a placement policy is: it may change the frame's origin and routing policy, given the CPUs the lattice routes
/// with. Called concurrently from many threads: `Fn`, never `FnMut`.
pub type PlacementPolicy = dyn Fn(&mut dyn Frame, &CpuSet) + Send + Sync + 'static;

/// A policy in a cell.
struct PolicyBox(Box<PlacementPolicy>);

static NO_CPUS: LazyLock<CpuSet> = LazyLock::new(CpuSet::new);

/// The placement of one vertex: its policy and the CPUs it is given. Whether a policy is set is a flag of the vertex
/// (next to the fields a push reads anyway), so that a frame without a policy reads nothing here.
pub struct Placement {
    policy: Cell<PolicyBox>,
    cpus: Cell<CpuSet>,
}

impl Placement {
    /// A placement with no policy.
    pub fn new(registry: &UpstreamRegistry) -> Arc<Self> {
        Arc::new(Self {
            policy: registry.cell(),
            cpus: registry.cell(),
        })
    }

    /// Sets the policy. Lifecycle work, for the caller to do while nothing routes. A vertex runs it only once its
    /// [`set_placement_policy`](super::LatticeVertex::set_placement_policy), which calls this, has raised its flag.
    pub fn set_policy(
        &self,
        registry: &UpstreamRegistry,
        policy: impl Fn(&mut dyn Frame, &CpuSet) + Send + Sync + 'static,
    ) {
        registry.set_cell(&self.policy, Some(Arc::new(PolicyBox(Box::new(policy)))));
    }

    /// Publishes the CPUs the policy is given from now on.
    pub fn publish_cpus(&self, registry: &UpstreamRegistry, cpus: CpuSet) {
        registry.set_cell(&self.cpus, Some(Arc::new(cpus)));
    }

    /// Runs the policy on `frame`, reading it from `view`, the read the caller is already in (a vertex routes a frame
    /// inside a read of its routing generation).
    #[inline]
    pub fn apply_in(&self, view: &View<'_>, frame: &mut dyn Frame) {
        if let Some(policy) = view.get(&self.policy) {
            (policy.0)(frame, view.get(&self.cpus).unwrap_or(&NO_CPUS));
        }
    }

    /// Runs the policy on `frame` in a read of its own.
    pub fn apply(&self, registry: &Arc<UpstreamRegistry>, frame: &mut dyn Frame) {
        registry.with_view(|view| self.apply_in(view, frame));
    }
}

impl std::fmt::Debug for Placement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Placement").finish_non_exhaustive()
    }
}
