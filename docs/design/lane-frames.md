# Lanes whose items are frames

Status: design, no code. Base: `origin/main` at 47b957f.

## The change

Today `LaneSource::serve` runs each item inside the source's own `request` (`serve` → `admit` → `run_one`,
`euhedral-graph/src/lane_source/source.rs:428-533`), and only the item's successors are frames. In this design:
- the source only assembles;
- an item is released once it is ready, as a pooled **item frame** whose `id_hash == routing_hash ==` the lane's hash;
- the core runs a lane's item frames in order on one worker and owns their lifecycle;
- the lane's submitter and state move from the source's `SourceCell` into a **lane cell** that only the lane's frames enter.

The source keeps three jobs: it wraps intake work into frames, it holds work that is not ready, and it hands out the frames
the lane makes.

Why: workers learn the shape of the workload only from frames they bring back and push through their executor
(`fragment.rs:498-515`). That timing sets the batch size, idling and participation. Work run inside a source's `request` or
`pull` is invisible to them. §11 lists every source that does this.

## 1. The ordering guarantee

Frames of one hash with `id_hash == routing_hash` run one after another on one thread, in push order.

| Step | RustEE | Java |
|---|---|---|
| The socket, core and partition come from the routing hash (SMT: an odd hash goes to the buddy) | `shard.rs:691-701`, `cache.rs:620-633` | same |
| Only the owner takes ordered frames: stealers stop at an ordered head, and direct pulls stop before one. Workers may still route them by hand | `cache.rs:395, 545-549`, `vertex.rs:505-547` | `ControlPlaneCache.java:197, 252` |
| The owner runs each frame inline, greedy or poll | `fragment.rs:396-398, 507`, `executor.rs:87-89` | same |
| A source's pushes are serialized by the interceptor's claim, so they enter the partition queue in push order | `vertex.rs:1036-1141` | same |

**Rebalances.** The existing behaviour is enough, and nothing in the core changes:
- A worker that is inside a source's push or pull when a drain starts finishes it freely.
- After that, the drain keeps workers out of the handle until the rebalance is over (`vertex.rs:974-981`;
  `LatticeVertex.java:589-593, 621-625`).
- No external thread pushes lane frames: only workers inside the source's `request` do.

RustEE routes by each worker's own view, refreshed once per cycle (`vertex.rs:333`, `fragment.rs:373`). Java routes by one
mapping. The difference shows only while a routing generation changes, which happens only in a rebalance, and a stale view
lasts only until that worker's push or pull ends. It falls under the rebalance case above, so it needs no fix.

`ControlPlaneCache::pull` (`cache.rs:360`; Java `ControlPlaneCache.java:185`, `NO_STOP`) fills the caller's own cache, ordered
frames included. Nothing in production calls it. It is a trap.

## 2. The lane's state

- `Arc<Lane>` holds the lane cell, `LaneCell<Inner>`, with the submitter, the state and the armed retirements. Only the lane's
  item frames and its poll frame (§5) enter it, from `execute`, `do_finally` or `do_finally_with_error`.
- By §1 those frames run one after another on one thread, so the fields are plain.
- The cell keeps `SourceCell`'s checked claim (one uncontended CAS on entry, one store on exit, on a line that stays in the
  lane core's L1). A refused entry panics, as `SourceCell::enter` does.
  - A refusal can come only from a rebalance overlap, which §1 accepts.
  - The claim is not debug-only, because without it an overlap in a release build would be two `&mut`.
- `LaneCx` keeps its methods. `ready` now offers to the source's ready queue (MPSC: frames produce into it, and the source
  drains it in `request`/`pull`).
- The frame holds its `Arc<Lane>` from the moment it is made (§7), so no reference count moves per item.
- `Drop for Lane` does what `Inner::drop` does today (`source.rs:77-113`), wherever the last reference drops, as today. It
  happens once, at shutdown or release.

## 3. Hashes

| Work | id = routing hash |
|---|---|
| Every item of a lane's own intake, ordered or not | the lane's hash: it needs the lane cell |
| An item of a group's shared intake | the hash of the lane whose source took it (as today, `source.rs:498-507`) |
| The lane's poll frame | the lane's hash |
| Successors that touch no lane state | the item's choice (`randomize_hash` for parallel work) |

The caller gives each lane its hash (`LaneSource::new(.., hash)`) and never randomizes it. A dependent that wants its lanes apart picks hashes
that map to different cores, and accepts that a rebalance may put two lanes on one core. Item frames are ordered, so they leave
only through `request` + push. Pulls hand over unordered successors.

## 4. Waits: not released until ready

- **Every wait is a gate.** The source checks it before release, and nothing waits once released. An item says what it needs
  with `LaneWork::needs(&self) -> Option<Wait>`. The source's `LaneGate` answers `met` from what it can read without the
  cell:
  - markers (`DeviceQueue` is `Sync`);
  - clocks;
  - counters the lane's frames publish with single-writer release stores: free submission-ring and marker-ring slots,
    freed buffers, locks run, the hardware probe's progress.

  The source keeps its own plain "taken" counts, so the room left is `freed − taken`. Those atomics are shared between two
  parties, and none of them is a read-modify-write.
- `Outcome::Wait` goes away. A submission refused after release (`mark` finds no room despite the gate) is a failure:
  `failed`.
- **Held items.** An unmet item waits in the source's plain `held` queue, with the items of its order key behind it.
  Unordered items pass it. So order keys now act only at the source.
- **Dead held items.** They need one look per pass (one acquire load of `is_alive`). The source then emits a dead item at
  once, the core skips it, and its `do_finally` runs `skipped` in the cell (`executor.rs:43-46`). There is no other end
  path. A released item needs no check from the lane at all.
- **A value another thread writes.** The source holds the item on a gate that reads the value with one acquire load.
  - The writer does not call `request`, which is the downstream's demand.
  - The writer's release store is the signal. The source rereads its gates on every pass, which is every worker cycle while
    the lattice is busy, and at least once per probe interval (1 ms) when workers sit out (CHANGELOG, 2026-11-03). That
    makes a timer backstop redundant.
  - For less than 1 ms, the writer's frame delivers a wake item into the lane's intake from its `do_finally`.

## 5. Retirements and completions

- **Retirements are observed in the lane cell.** `poll_retired` is an owner call: it settles the ticket, releases what the
  stream kept and probes for faults (`euhedral-gpu/src/lanes.rs:168-189`).
  - Every lane frame drains retirements on entry, as `serve` does today (`source.rs:485`).
  - When retirements are outstanding and nothing is released, the source emits the lane's **poll frame**. It is made once,
    and the source emits it when the published `armed > retired` and the frame is back in the pool, so at most one is in
    flight.
- **Completions that need order run in the cell.** `retired` changes lane state, and retirements arrive in arm order.
- **Completions that do not need order are spawned.** What `retired`, `then` or `failed` make for others is a frame put into
  a source: the lane's ready queue, or `Completions::after_marker`.

## 6. Lifecycle

| Today | Item frame |
|---|---|
| `admit`/`run_waiting` → `run_one` | `execute`: enter the cell, drain retirements, `X::check`, run |
| `then`, `failed(S::Error)` | inside `execute` |
| `retired` | in the cell, in the lane frame that drains it (usually the poll frame) |
| `skipped` (draft PR #47) | `do_finally` of a frame skipped as dead: enter the cell, `skipped(work, cx)` |
| — | `do_finally_with_error` (a panic in `run`, unroutable at a closed lattice, a push to a retired cache): enter the cell, `skipped` |

`LaneFrame::is_alive` delegates to `LaneWork::is_alive`, so the header carries no kill switch.

Every production path finishes a frame:
- `close` keeps what the cache holds for its owner to drain (`cache.rs:607`);
- a push to a retired cache gets `do_finally_with_error` (`cache.rs:626`; Java `ControlPlaneCache.java:396, 414`).

The one exception is `reset_for_next_trial` (`cache.rs:595`; Java `clearLocalCacheOnOwnerThread`,
`ControlPlaneCache.java:307`), which drops queued frames without finishing them. It is a benchmark trial boundary that
runs once ingest is frozen, so lane frames must not run across one.

## 7. Costs and recycling (the `GrpcFrame` pattern)

In Java's gRPC transport (`EuhedralGrpcServerHandler`, `GrpcFrame`, `FrameManager`):
- each source owns a `FrameManager`;
- each frame is made once with the source's fixed parts;
- `replace(message)` swaps the payload and calls `resetHash()`;
- `recycle()` offers the frame to a bounded MPSC queue that the owner drains 256 at a time.

RustEE has the same pieces: `impls::FrameManager`, `FrameFactory`, `Recycler`, `Frame::recycle`.

| Java | Lane item frame |
|---|---|
| `FrameManager` owned by the handler | owned by the `LaneSource`, consumed with `&mut` in `request` |
| `create`: callbacks, kill switch, recycler | `create`: `Arc<Lane>`, `FrameHeader::new(lane hash)`, the `Recycler` |
| `replace(message)` + `resetHash()` | `replace(work)`: the item and its delivery stamp (the hash never changes) |
| `doFinally` → `recycle()` | `do_finally` → (`skipped` if dead) → `Recycler::recycle_owned` |

**Per item, beyond today's intake offer and poll (estimates, not measured):**

| Cost | Detail |
|---|---|
| Allocation | none after warm-up |
| New read-modify-writes | the recycle offer, and the `Weak::upgrade` in `Recycler::recycle`, which a strong producer handle would remove: +1 to +2. The partition offer and the cache count: +2 or +3. The cell's CAS: +1 (today one per pass) |
| Lock | none |
| CPU | about 0.2-0.5 µs per item |
| Latency | up to one owner park (15 µs) when the lane's core is idle. Successors go out on the source's next pass, one poll later: the source pushes, the Euhedral way |

Compare with today's `item_ns` per kind once step 1 lands.

## 8. Metrics

These are kept:
- `service_ns`, for the source's passes;
- `item_ns[kind]`, timed around `run` with `record_owned`;
- `intake_to_run_ns`;
- `ready_depth`;
- `stranded`.

These are new:
- `skipped[kind]`;
- `held_depth`;
- `release_to_run_ns`.

## 9. Migration

Each step can be merged on its own.

1. **euhedral-graph.** Add `Lane`, `LaneCell`, `LaneFrame`, the poll frame, the pool and the ready queue, and swap the
   `LaneSource` internals.
   - `LaneSource::new` takes the lane hash.
   - `LaneWork` gains `is_alive`/`skipped` (PR #47's API, without its source-side check of intake items) and `needs`, and
     loses `Outcome::Wait`. `LaneState` gains its `LaneGate`.
   - Re-pin `tests/lane_source*.rs` and `lane_intake_alloc.rs`.
2. **euhedral-gpu.**
   - `LaneSubmit` (`src/lanes.rs`) is unchanged. The `Stream` calls enter the context themselves, which matters now that
     `poll_retired` runs on the lane's core.
   - Re-pin `open_lane_sources` and the fake and CUDA lane tests.
   - In euhedral-benchmarks: `bin/idle_polls.rs`.
3. **euhedral-memory.** `Moves` as in §11 (it does not use `lane_source`).
4. **A dependent that adopts this** repins, then moves each item kind's liveness check from `run` to `is_alive`/`skipped`,
   turns each wait into a gate on what its lane publishes, and passes its lane hashes. Draft PR #47 is superseded by step 1.

## 10. Tests

- **core:** `ordered_frames_of_one_hash_never_overlap` (per-hash in-flight counters on a real lattice with greedy, poll,
  stealing and SMT; at most 1 in flight, push order kept).
- **graph:**
  - `items_of_a_lane_run_in_release_order_one_at_a_time`;
  - `an_item_is_not_released_until_its_gate_is_met`;
  - `a_held_item_holds_its_keys_later_items_and_not_others`;
  - `a_dead_held_item_is_emitted_and_skipped_without_its_gate`;
  - `a_dead_released_item_is_skipped_by_the_core_and_skipped_runs_in_the_lane`;
  - `retirements_reach_retired_in_arm_order`;
  - `a_lane_with_only_armed_retirements_still_polls`;
  - `steady_state_allocates_nothing`.
- **gpu:** `cuda_lanes` and `cuda_lanes_fault`, with retirements polled on the lane's core.

## 11. Migration inventory: sources that execute work in `request`/`pull`

Each replacement below follows the same pattern:
- ordered frames with `id_hash == routing_hash`;
- work held until it is ready;
- frames recycled as `GrpcFrame` is;
- completions spawned from `do_finally` into a source.

The rows run from hottest down.

| # | Site | What runs in the pass | Replacement |
|---|---|---|---|
| 1 | `LaneSource::serve` (`euhedral-graph/src/lane_source/source.rs:478-533`) | every ready item's `run`: the device submissions and whatever else the item does | this design |
| 2 | `Moves::emit` → `pump` (`euhedral-memory/src/tiers.rs:1155-1164, 866-906`) | finishes moves, starts second-hop copies (driver calls), frees slots, and allocates a `Vec` per pass | the source emits one recycled pump frame per tier, ordered on the tier's hash, when moves wait. Retired moves are spawned from its `do_finally` |
| 3 | `AsyncIo` (`euhedral-io/src/lib.rs:917-957`, `uring.rs:623-651`) | reaping (fine), plus an SQE flush (`io_uring_enter`) and `EAGAIN` requeues | the flush moves to the submitting frame, or to a frame emitted when entries wait. Continuations only build frames (`lib.rs:54-55`) |
| 4 | `IdleWatch::poll` (`euhedral-graph/src/lake.rs:562-570`) | a detach under a lock, once per idle period | a lifecycle frame (low priority) |
| — | `QueueSource`, `SingleUseSource`, `LatticeHotSource`; `Completions` (queries only); `DeferredCalls` (pushes one key frame); `GiveBacks` (wraps orders, though its frames are boxed per order and should be recycled) | — | fine |

**Order.** 1 (§9 step 1), then 2, 3 and 4.
