# Changelog

Changes to the lattice's execution model, newest first. Each entry says what changed, why, how to use it, its exact
semantics and the tests that pin it.

## Greedy, not drain, for holding a partition while its frames run (2026-10-23)

**What.** `ControlPlaneCache::drain` and `drain_while` are renamed `greedy` and `greedy_while`; the mode they make is
**greedy mode**, against **poll mode**. The tests and docs that named it "drain" follow.

**Why.** In RustEE, **drain** means stop accepting work and finish what is in flight (`set_drain_mode`, a draining
edge or vertex, `is_drained`, a shard draining its clones). **Greedy** means consume as fast as possible by holding
locks longer, which is what this take does: it holds a partition while it runs the frames it takes from it. One word
for both was confusing.

**How to use it.** Replace `cache.drain(..)` with `cache.greedy(..)` and `cache.drain_while(..)` with
`cache.greedy_while(..)`. Behaviour is unchanged. The queues in `euhedral-data-structures` keep their `drain*`
methods: there the word means taking items out of a container.

## Routing policy honoured by stealing and direct pulls (939ccfb, 2026-10-23)

**What.** Work stealing and a worker's direct pull from its upstreams now follow a frame's
[`RoutingPolicy`](euhedral-core/src/flow_control/mod.rs). Until this change, stealing stopped only at ordered frames.
So:
- an idle worker on any core could steal an unordered `CacheLocal` frame from the core it was routed to;
- a `SocketLocal` frame could be stolen across sockets;
- the direct path could run a `CacheLocal` frame bound to another core.

**Why.** The routing policy and the ordered flag are both binding. A frame's policy says how close to its origin it
must run. Routing already placed frames by it:
- the lattice's `route_to_socket` sends `SocketLocal` and tighter frames to the origin's socket;
- a shard's `route_to_core` sends `CacheLocal` frames to the origin's core.

Stealing and direct pulls must not undo that placement.

**How to use it.** Set the policy and the origin on the frame's header before it is published:
```rust
let mut header = FrameHeader::new(id_hash);
header.set_origin(euhedral_hardware_utils::affinity::current_cpu_info().copied()); // or the CpuInfo to stay near
header.set_routing_policy(RoutingPolicy::CacheLocal);
```
A frame with no origin, or with `Anywhere` (the default), goes anywhere, as before.

**Semantics.** `locality_allows(frame, here, queued)` in `control_plane/cache.rs` decides whether a worker on core
`here.0` of socket `here.1` may take a frame that another cache holds (`queued`), or that is still in an upstream
(`queued = None`). The cache that holds a frame may always run it.

| Policy | Who may take it from elsewhere |
|---|---|
| `Anywhere`, or no origin | any worker |
| `CacheLocal` | a worker on the origin's core only: its SMT threads, where `route_to_core` places it |
| `SocketLocal` | a worker on the origin's socket only |

**Fallbacks keep a frame stealable:**
- **`CacheLocal`, origin core inactive:** routing falls back to a hashed core. The frame then keeps to the origin's socket, so it is never left to one worker.
- **`SocketLocal`, origin socket inactive:** routing falls back to another socket. The frame then goes anywhere.

**Stealing:**
- `work_steal` and `work_steal_polled` stop, in each partition, at the first frame that is ordered or that its policy keeps where it is.
- `upstream_pull` (the direct path) stops before the first ordered frame and the first frame bound to another core, leaving both to the lattice's routing.
- Ordered frames are unchanged: only the owning cache's CPU runs them (see poll mode below).

**Tests** (`control_plane/cache/tests.rs`, `impls/base_cloneable/tests.rs`):
- `locality_allows_what_each_routing_policy_lets_leave`: the predicate for every policy and fallback.
- `stealing_leaves_cache_local_frames_to_their_core_in_greedy_and_poll_mode`.
- `a_cache_local_frame_behind_a_blocking_one_waits_for_its_own_core`.
- `socket_local_frames_are_stolen_only_on_their_socket_and_anywhere_frames_by_all`.
- `a_cache_local_frame_that_fell_back_to_another_core_stays_stealable_on_its_socket`.
- `upstream_pull_leaves_a_frame_bound_to_another_core_to_routing`, `upstream_pull_runs_a_frame_bound_to_its_own_core`.
- `cache_local_frames_behind_a_blocking_one_stay_on_their_core_in_both_modes`: on a real lattice of two cores, 64 `CacheLocal` frames queued behind a blocking one run only on their core, in greedy and in poll mode. Before the change, the other core ran them.

## Poll mode (8cb789d, 845d5e9, 2026-10-23)

**What.** A worker whose max batch is at most `FragmentConfig::POLL_MODE_MAX_BATCH` (2) **polls** its cache instead
of taking from it greedily.
- **A poll takes one frame at a time,** holding the frame's partition only while it takes the frame out.
- **It steals the same way.**
- **So a frame that blocks no longer holds up the unordered frames queued behind it:** idle workers steal them.

**Why.** The two modes exist for two kinds of work:
- **Greedy mode is for short frames.** A greedy take holds a partition while it runs the frames it takes, so the consumer consumes them as fast as it can without touching the producer side. If one of those frames blocks, though, it holds up every frame queued behind it in its partition, and work stealing passes the held partition over.
- **Poll mode is for lattices that run slow or blocking work.** Each frame costs the partition hold once, and in exchange no partition is held while a frame runs.

**How to use it.** Give the lattice's fragments a max batch of 2 or less:
```rust
let fragment = FragmentConfig::default()
    .with_max_batch_size(NonZeroU64::new(FragmentConfig::POLL_MODE_MAX_BATCH).unwrap());
assert!(fragment.polls());
let config = LatticeConfig::new(Arc::new(BaseCloneableObject::new(fragment, DefaultExecutor::new())));
```
- **`FragmentConfig::polls()`** says whether a config's workers poll. The fragment's cycle uses it to choose `poll` or `greedy` for its own cache, and `work_steal_polled` or `work_steal` for stealing.
- **A process can run lattices in both modes,** for example one greedy lattice for short frames and one polling lattice for blocking frames, each on its own CPUs (`LatticeConfig::allowed_cpus`).

**Semantics:**
- **`ControlPlaneCache::poll(consume, limit)`** moves up to `limit` frames into `consume` one at a time. While `consume` runs a frame, that frame's partition is free.
- **`poll_while(consume, stop, limit)`** does the same, but leaves each partition at its first frame that `stop` matches. Its walk starts one partition further on each call, so a busy partition does not starve the others.
- **`work_steal_polled`** is `work_steal` over `poll_while`: a stolen frame that blocks does not hold up the victim's partition.
- **Ordered frames keep their order with the hold released:**
  - only the cache's owner takes them;
  - stealers stop at an ordered head, and an SMT buddy takes from its own cache;
  - the owner runs them one after another.
  - So an ordered stream runs on its owning CPU, one frame at a time and in sequence, while unordered frames are stolen around it.
- **The routing policy limits polls and stolen polls** just as it limits greedy takes (see the entry above).

**Tests:**
- **`config/fragment_config.rs`:**
  - `workers_poll_at_a_max_batch_of_two_or_less`.
- **`control_plane/cache/tests.rs`:**
  - `a_poll_frees_the_partition_while_its_frame_runs`, `a_greedy_take_holds_the_partition_while_its_frames_run`;
  - `polls_take_one_frame_at_a_time_in_order_and_leave_ordered_heads_to_the_owner`;
  - `polls_walk_every_partition`.
- **`impls/base_cloneable/tests.rs`**, on a real SMT core:
  - `a_polling_lattice_runs_the_frames_queued_behind_one_that_blocks`: in poll mode the 512 unordered frames queued behind a 2 s blocker all run within a second. In greedy mode, 21 of them wait for it.
  - `a_polling_lattice_runs_ordered_frames_on_their_own_cpu_in_sequence_while_others_steal` (845d5e9): two ordered streams, one per SMT thread. Each runs only on its own CPU, one frame at a time and in sequence, while both workers steal the 20 µs unordered frames queued between them. Stealers take only unordered work.
