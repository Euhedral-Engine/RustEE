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
//!   [`Reaper`] runs frees that may block off the lattice's workers.
//! - [`cuda`] (feature `cuda`): the CUDA driver backends: device memory (`cuMemAlloc`), the stream-ordered pool
//!   (`cuMemAllocAsync`), VMM allocations exportable as POSIX file descriptors (`cuMemCreate`, imported by Vulkan as
//!   external memory) and pinned host memory with a capacity cache.
//!
//! Nothing here owns a thread, takes a lock or waits: budgets and owner totals are atomics, and owner slots, give-back
//! orders, the reaper and the pinned cache go through RustEE's lock-free queues. Give-back orders reach their owners
//! as frames of a `LatticeSource` ([`GiveBacks`]). Residency is planned, not discovered: every consumer's room is
//! reserved, what spills to pinned host memory is decided up front, and what cannot fit is refused; released pinned
//! buffers stay in a capacity cache.

pub mod admission;
pub mod arena;
pub mod backend;
#[cfg(feature = "cuda")]
pub mod cuda;
pub mod planner;
pub mod tiers;

pub use admission::{Admission, Placed, PlanItem, PlanRequest, Reservation, Tiers};
pub use arena::{BackendCalls, SCRATCH_MIN, ScratchArena, ScratchLease, SlotArena, SlotLease};
pub use backend::{AllocError, Allocation, Backend, Block, Reaper, Retired};
pub use planner::{
    Claim, ClaimResult, DEFAULT_OWNERS, Elastic, GiveBack, GiveBackHandler, GiveBacks, Leak,
    LeakReport, LocationId, LocationKind, LocationSpec, LocationUse, MAX_LOCATIONS, OwnerId,
    OwnerSpec, Planner, Priority, Scope, Shortfall, Snapshot,
};
pub use tiers::{
    Copier, CopyError, HostCopier, MarkerRetired, MoveDone, MoveError, MoveHandler, Mover, Moves,
    ResidentId, TierPool,
};

/// Compiles and runs the examples in the crate README as doctests.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;
