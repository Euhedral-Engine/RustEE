# euhedral-memory

A location-agnostic memory and residency planner for the RustEE control plane, and CUDA allocation backends. It:
- reserve every consumer's room up front;
- fit or refuse;
- spill to pinned host memory by plan, not by failure;
- keep released pinned buffers in a capacity cache.

**Where it sits.** It builds on [`euhedral-core`](../euhedral-core) (`LatticeSource`, `RunnableFrame`: give-back orders are
frames), [`euhedral-data-structures`](../euhedral-data-structures) (lock-free queues) and
[`euhedral-hashing`](../euhedral-hashing). Nothing in the workspace depends on it. Feature `cuda` adds the CUDA backends,
with `libcuda.so.1` loaded by `dlopen` at run time (no toolkit, no link-time dependency).

## Modules

| Module | Main types |
|---|---|
| [`planner`](src/planner.rs) | `Planner`, `LocationSpec` / `LocationKind` / `LocationId`, `OwnerSpec` / `OwnerId`, `Priority`, `Scope`, `Claim`, `Elastic`, `ClaimResult`, `GiveBack` and the `GiveBacks` lattice source, `Snapshot`, `LeakReport`. |
| [`admission`](src/admission.rs) | `PlanRequest`, `PlanItem`, `Admission`, `Placed`, `Reservation`. |
| [`arena`](src/arena.rs) | `SlotArena` / `SlotLease`, `ScratchArena` / `ScratchLease`, `BackendCalls`. |
| [`backend`](src/backend.rs) | The `Backend` trait, `Block`, `Allocation`, `Reaper` and `Retired`. |
| [`cuda`](src/cuda/mod.rs) (feature `cuda`) | `Driver`, `Device` (primary context, `import_fd` for Vulkan-exported memory), `DeviceMemory`, `StreamPool`, `Vmm` (POSIX-fd export), `PinnedHost`. |

## Key concepts

- **Plan once, allocate once, reuse.**
  - A session's working set (pools of equal slots, scratch, staging, pinned host slots) is admitted and allocated
    once.
  - It is handed out as arena leases through lock-free free lists. In the steady state nothing reaches a backend;
    `Planner::calls` counts what does.
  - Real allocations and frees happen at a session's start and end, at a size change and at idle release.
- **Locations are alike.** A device, pinned host and pageable host are all budgets the same API claims against. Each has
  a capacity (what this process may hold) and a reserve (always left free).
- **Owners have lifetimes.**
  - An owner is a static name, a priority, a `Scope` (`Process`, `Epoch`, `Plan`, `Lease`) and an optional parent.
  - `end_owner` and `end_scope` end a tree of owners and report what it still holds: the leak check.
  - `check_baseline` compares with an earlier `Snapshot`.
- **Claims are values.** A `Claim` gives its bytes back when dropped, so it travels with the memory it accounts for and
  dies with it.
- **Elastic holdings give back on order.** `claim_or_reclaim` takes room from lower-priority elastic owners: lowest
  priority first, then the biggest holder. The orders are frames of the planner's `GiveBacks` source and run the owner's
  handler. The planner calls nobody and waits for nothing.
- **Admission before allocation.** `admit` places a plan's items at their preferred locations, then spills the spillable
  ones, then cuts optional ones, both lowest priority first. Otherwise it refuses with a `Shortfall`. An admitted plan's
  `Reservation` gives its allocations their claims.
- **No locks.** Budgets are CAS loops on atomics. Owner slots, give-back orders, the reaper and the caches are lock-free
  queues.

```rust
use euhedral_memory::{
    LocationKind, LocationSpec, OwnerSpec, Placed, PlanItem, PlanRequest, Planner, Priority, Scope, SlotArena,
};

const GIB: u64 = 1 << 30;
let planner = Planner::new([
    LocationSpec { kind: LocationKind::Device { ordinal: 0 }, capacity: 16 * GIB, reserve: 700 << 20 },
    LocationSpec { kind: LocationKind::HostPinned, capacity: 32 * GIB, reserve: 0 },
]);
let dev = planner.location(LocationKind::Device { ordinal: 0 }).unwrap();
let pinned = planner.location(LocationKind::HostPinned).unwrap();

// Something else holds 9 GiB: the spillable buffer moves to pinned host memory before any optional consumer is cut.
let other = planner.open_owner(OwnerSpec::new("other process", Priority(200), Scope::Lease(1)));
let _lease = planner.claim(other, dev, 9 * GIB, "lease").unwrap();

let plan = planner.open_owner(OwnerSpec::new("plan", Priority(100), Scope::Plan(1)));
let request = PlanRequest {
    owner: plan,
    items: vec![
        PlanItem::required("slots", dev, 3 * GIB),
        PlanItem::optional("consumer a", Priority(80), dev, GIB),
        PlanItem::optional("consumer b", Priority(20), dev, 2 * GIB),
        PlanItem::required("spillable buffer", dev, 4 * GIB).spills_to(vec![(dev, GIB / 2), (pinned, 4 * GIB)]),
    ],
};
let admitted = planner.admit(&request).unwrap();
assert_eq!(admitted.placed, [Placed::Preferred, Placed::Preferred, Placed::Dropped, Placed::Spilled]);

// The plan allocates once, from its reservation, and reuses slots for the session's life.
let mut reservation = admitted.reservation.unwrap();
let claim = reservation.take(plan, dev, 3 * GIB, "slots").unwrap();
let slots = SlotArena::new(0, GIB / 8, 24, claim);
let slot = slots.lease().unwrap(); // no allocation: a free-list pop
drop(slot);
drop((slots, reservation));
assert!(planner.end_owner(plan).is_clean());
```

## Build and test

```sh
mise exec -- cargo test -p euhedral-memory                       # planner, admission, arenas (no device needed)
mise exec -- cargo test -p euhedral-memory --features cuda       # + the CUDA backends (skipped without a device)
mise exec -- cargo run --release -p euhedral-memory --features cuda --example alloc_sweep   # CUDA against Vulkan
```
