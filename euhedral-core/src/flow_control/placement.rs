//! A placement policy on a vertex: one callback that runs on every frame before the vertex routes it.
//!
//! The policy sees the frame and the CPUs the lattice routes with now, and says where the frame goes by setting its
//! origin and routing policy, returning `true` when it did (placed the frame). The routing that follows honours the
//! origin and routing policy. It is one callback, not a registry. A placement always has one: until the owner sets its
//! own, it is a policy that places nothing and returns `false`.
//!
//! The policy runs on the frame path, from every thread that routes at once. It must be re-entrant, take no lock,
//! not allocate and not block; whatever it reads is the caller's to make safe to read from many threads.
//!
//! **The policy is a plain field.** It is set only while nothing routes and nothing pulls (the owner sets it before
//! the vertex has any source, see [`set_policy`](Placement::set_policy)) and is only read after that, so a frame reads
//! it with no atomic and no view. What makes the write visible to the threads that read it is whatever hands them the
//! vertex's sources afterwards (the owner's start, which synchronises with every later ingest).
//!
//! **The CPU set changes at run time**, so it is a cell of the lattice's registry, read from the thread's view with no
//! lock. A worker (a full reader) reads it with no read-modify-write on anything shared: a pull holds one read of the
//! view for all the frames it places ([`within`](Placement::within)), and a push reads the view it is already in. A
//! light reader (a producer thread that is not registered) upgrades one weak reference to read it, so a push costs it one
//! reference-count pair and a pull costs it one for the whole pull.

use super::registry::UpstreamRegistry;
use super::views::{Cell, View};
use crate::frames::Frame;
use euhedral_hardware_utils::util::CpuSet;
use std::cell::UnsafeCell;
use std::sync::Arc;
use std::sync::LazyLock;

/// What a placement policy is: it may change the frame's origin and routing policy, given the CPUs the lattice routes
/// with, and returns whether it placed the frame. Called concurrently from many threads: `Fn`, never `FnMut`.
pub type PlacementPolicy = dyn Fn(&mut dyn Frame, &CpuSet) -> bool + Send + Sync + 'static;

static NO_CPUS: LazyLock<CpuSet> = LazyLock::new(CpuSet::new);

/// The policy a placement has until its owner sets one: places nothing.
fn no_placement(_frame: &mut dyn Frame, _cpus: &CpuSet) -> bool {
    false
}

/// The placement of one vertex: its policy and the CPUs it is given.
pub struct Placement {
    /// Written only by [`set_policy`](Self::set_policy), while nothing reads it.
    policy: UnsafeCell<Box<PlacementPolicy>>,
    cpus: Cell<CpuSet>,
}

// SAFETY: the policy is `Send + Sync`, and the field is written only through `set_policy`, whose contract is that no
// other thread reads the placement while it runs; the CPU set is a cell of the registry, which is shared by design.
unsafe impl Sync for Placement {}

impl Placement {
    /// A placement whose policy places nothing.
    pub fn new(registry: &UpstreamRegistry) -> Arc<Self> {
        Arc::new(Self {
            policy: UnsafeCell::new(Box::new(no_placement)),
            cpus: registry.cell(),
        })
    }

    /// Sets the policy.
    ///
    /// # Safety
    ///
    /// No thread may be reading the placement (routing a frame, or pulling through the vertex) during this call, and
    /// every thread that reads it afterwards must have synchronised with the call: the vertex has no source yet, and a
    /// source is ingested only by an operation that happens after this call.
    pub unsafe fn set_policy(
        &self,
        policy: impl Fn(&mut dyn Frame, &CpuSet) -> bool + Send + Sync + 'static,
    ) {
        // SAFETY: the caller's contract: no reader exists, so this is the only access to the field.
        unsafe { *self.policy.get() = Box::new(policy) };
    }

    /// Publishes the CPUs the policy is given from now on.
    pub fn publish_cpus(&self, registry: &UpstreamRegistry, cpus: CpuSet) {
        registry.set_cell(&self.cpus, Some(Arc::new(cpus)));
    }

    #[inline]
    fn policy(&self) -> &PlacementPolicy {
        // SAFETY: the field is written only by `set_policy`, whose caller guarantees nothing reads while it runs.
        unsafe { &**self.policy.get() }
    }

    /// Runs the policy on `frame`, reading the CPUs from `view`, the read the caller is already in (a vertex routes a
    /// frame inside a read of its routing generation). Returns whether the policy placed the frame.
    #[inline]
    pub fn apply_in(&self, view: &View<'_>, frame: &mut dyn Frame) -> bool {
        self.policy()(frame, view.get(&self.cpus).unwrap_or(&NO_CPUS))
    }

    /// Runs `f` inside one read of the view, with the policy and the CPUs as that read sees them, for a stretch of work
    /// that places many frames (a pull): the view is read once, not for every frame. The read stays open while `f`
    /// runs, so a worker does not take in a newer CPU set until the next read; nothing `f` does through the registry is
    /// held up by it.
    pub fn within<R>(
        &self,
        registry: &Arc<UpstreamRegistry>,
        f: impl FnOnce(&PlacementView<'_>) -> R,
    ) -> R {
        registry.with_view(|view| {
            f(&PlacementView {
                policy: self.policy(),
                cpus: view.get(&self.cpus).unwrap_or(&NO_CPUS),
            })
        })
    }

    /// Runs the policy on `frame` in a read of its own; returns whether it placed the frame.
    pub fn apply(&self, registry: &Arc<UpstreamRegistry>, frame: &mut dyn Frame) -> bool {
        registry.with_view(|view| self.apply_in(view, frame))
    }
}

/// The policy and the CPUs of a read of the view, for the length of [`Placement::within`].
pub struct PlacementView<'a> {
    policy: &'a PlacementPolicy,
    cpus: &'a CpuSet,
}

impl PlacementView<'_> {
    /// Runs the policy on `frame`: no view is read. Returns whether the policy placed the frame.
    #[inline]
    pub fn apply(&self, frame: &mut dyn Frame) -> bool {
        (self.policy)(frame, self.cpus)
    }
}

impl std::fmt::Debug for Placement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Placement").finish_non_exhaustive()
    }
}
