//! A location-agnostic memory and residency planner for the RustEE control plane.
//!
//! - [`Planner`]: locations (a GPU's memory, pinned host memory, pageable host memory, others) with budgets; owners
//!   with priorities and lifetimes ([`Scope`]) that **claim** and **free** bytes against them; firm claims, elastic
//!   holdings that grow into free memory and give back on order; and a leak check (an ended owner holds nothing; a
//!   baseline comes back).
//! - [`admission`]: a plan's whole working set is placed before it allocates: spill to another location before
//!   cutting a consumer, then cut the lowest-priority optional consumers, else refuse. Never discovered by an
//!   allocation failing in the steady state.
//! - [`arena`]: plan once, allocate once, reuse. A session's working set is allocated once and handed out as slots
//!   ([`SlotArena`]) and scratch pieces ([`ScratchArena`]) through lock-free free lists; in the steady state no
//!   allocation or free reaches a backend ([`BackendCalls`] counts the ones that do).
//! - [`backend`]: the allocation side. The planner accounts and decides; a [`Backend`] allocates and frees, and a
//!   [`Reaper`] queues frees behind the device work that last used the memory, and [`Reaper::reap`] runs the ones that
//!   may block on the thread that calls it (a lattice worker, if a worker calls it: the `HOT-FIX` id M1).
//! - [`deferred`]: blocking driver calls (frees, unloads) held by a [`DeferredCalls`] lattice source until no lane has work
//!   in flight (or a bound forces them), instead of made where the memory is dropped: a call on another thread still makes
//!   every stream's enqueue wait for the running kernels.
//! - `cuda` (feature `cuda`): the CUDA driver backends: device memory (`cuMemAlloc`), the stream-ordered pool
//!   (`cuMemAllocAsync`), VMM allocations exportable as POSIX file descriptors (`cuMemCreate`, imported by Vulkan as
//!   external memory) and pinned host memory with a capacity cache.
//!
//! Nothing here owns a thread or takes a lock, and the planner waits for nothing: budgets and owner totals are
//! atomics, and owner slots, give-back orders, the reaper and the pinned cache go through RustEE's lock-free queues.
//! The one place that blocks is a driver free that waits for the device: [`Reaper::reap`] makes it on the calling
//! thread (the `HOT-FIX` id M1). Give-back orders reach their owners
//! as frames of a `LatticeSource` ([`GiveBacks`]). Residency is planned, not discovered: every consumer's room is
//! reserved, what spills to pinned host memory is decided up front, and what cannot fit is refused; released pinned
//! buffers stay in a capacity cache.

#![cfg_attr(
    test,
    allow(
        clippy::disallowed_types,
        clippy::disallowed_methods,
        reason = "test fixtures lock their records and run peer threads"
    )
)]

pub mod admission;
pub mod arena;
pub mod backend;
#[cfg(feature = "cuda")]
pub mod cuda;
pub mod deferred;
pub mod planner;
pub mod tiers;

pub use admission::{Admission, Placed, PlanItem, PlanRequest, Reservation, Tiers};
pub use arena::{BackendCalls, SCRATCH_MIN, ScratchArena, ScratchLease, SlotArena, SlotLease};
pub use backend::{AllocError, Allocation, Backend, Block, Reaper, Retired};
pub use deferred::{Bounds, Deferral, DeferredCalls, Place, PoolAttached, Ticket, WeakDeferral};
pub use planner::{
    Claim, ClaimResult, DEFAULT_OWNERS, Elastic, GiveBack, GiveBackHandler, GiveBacks, Leak,
    LeakReport, LocationId, LocationKind, LocationSpec, LocationUse, MAX_LOCATIONS, OwnerId,
    OwnerSpec, Planner, Priority, Scope, Shortfall, Snapshot,
};
pub use tiers::{
    Copier, CopyError, HostCopier, MarkerRetired, MoveDone, MoveError, MoveHandler, Mover, Moves,
    PitchedCopy, ResidentId, TierPool,
};

/// Compiles and runs the examples in the crate README as doctests.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;
