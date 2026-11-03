# Lanes whose items are frames

Status: design, no code. Base: `origin/main` at 47b957f.

## The change in one paragraph

Today `LaneSource::serve` runs each item inside the source's own `request` (`serve` → `admit` → `run_one`,
`euhedral-graph/src/lane_source/source.rs:428-533`), and only the item's successors are frames. In this design the source only
assembles. Each item it may release becomes a pooled **item frame** whose `id_hash == routing_hash ==` the lane's hash. The
source pushes these frames within its demand, and the core runs them in order on one worker and owns their lifecycle. The
lane's submitter and state move from the source's `SourceCell` into a **lane cell** that only the lane's frames enter. The
source keeps three jobs: wrapping intake work into frames, holding back work that is not ready, and handing out the frames
that the lane's frames make.

Why: workers learn the shape of the workload only from frames they bring back and push through their executor
(`fragment.rs:498-515`). That timing sets the batch size, idling and participation. Work run inside a source's `request` or
`pull` is invisible to them. §11 lists every source that does this.

## 1. The ordering guarantee

**What holds.** Frames whose hashes are equal, with `id_hash == routing_hash`, run strictly one after another on one thread,
never two at once, and in the order they were pushed, as long as the routing generation stays the same.

| Step | Code | Java |
|---|---|---|
| The socket, core and partition are picked from the routing hash (SMT: an odd hash always goes to the buddy) | `shard.rs:691-701`, `cache.rs:620-633` | same |
| Only the cache's owner thread takes ordered frames. Stealers stop at an ordered head, and direct pulls stop before one | `cache.rs:395`, `cache.rs:545-549` | `ControlPlaneCache.java:197, 252` |
| The owner runs each frame inline (`output.accept` → `ExecutionTerminal::push` → `run`), so greedy and poll both run one after another | `fragment.rs:396-398, 507`, `executor.rs:87-89` | same |
| A source's pushes are serialized by the interceptor's claim, so its frames enter the partition queue in push order | `vertex.rs:1036-1141` | same |
| Manual routing (`route_frame`, the placement policy's `CacheLocal` origin) is a pure function of the frame and the generation, so it keeps order | `vertex.rs:505-547`, `placement.rs:48-69` | same |

`ControlPlaneCache::pull` (`cache.rs:360`) fills the caller's own cache with no stop, ordered frames included. Java does the
same (`ControlPlaneCache.java:185`, `NO_STOP`), and nothing in production calls it. It is a trap, not drift.

**Where it does not hold today (both settled: fix them, §9 step 1):**

- **Rebalance window.** The author's rule: a rebalance closes global ingest first, gives the workers up to a minute to drain
  fully, and only then lets work back in. Java has every piece of this, but its code at 7e76a814 reopens early, and RustEE
  follows Java line for line. Here is the Java sequence, with RustEE's equivalent in brackets:
  1. Global ingest closes: `ControlPlaneLattice.java:326` (`lattice.rs:457`).
  2. Each shard drains its core distributor, `prepareIngestController`, `ControlPlaneShard.java:185, 211-222`
     (`shard.rs:262`).
  3. Each shard then publishes the new core mapping at once, `publishIngestController`, `ControlPlaneShard.java:189, 224-230`
     (`shard.rs:329`). This happens **before** the old clones drain.
  4. It sets each clone's drain mode and waits, asynchronously, until the clone is drained or `shutdownTimeout` has passed,
     `ControlPlaneShard.java:285-317` (`shard.rs:388-449`). The bound is one minute in both
     (`LatticeConfig.java:87`, `lattice_config.rs:41, 92`).
  5. Once every clone has drained, it reopens its distributor, `tryRestartIngest`, `ControlPlaneShard.java:376-397`
     (`shard.rs:487-503`).
  6. Global ingest reopens as soon as the shards' `update` calls return, while their drains (step 4) are still running:
     `ControlPlaneLattice.java:377, 406` (`lattice.rs:503`). Only retired sockets are waited for
     (`ControlPlaneLattice.java:384-400`, `lattice.rs:513-530`).

  The interceptor gates `request`/`pull` on the *global* drain only (`LatticeVertex.java:589-593, 621-625`,
  `vertex.rs:974-981`), and a push is never gated (`LatticeVertex.java:359-373`, `vertex.rs:505-547`). So between step 3 and
  step 5, sources push again, and frames of hash *h* route by the new mapping to a new core while the old core still holds or
  runs earlier *h* frames. A clone that misses the deadline is replaced (`ControlPlaneShard.java:322-330`,
  `shard.rs:458-468`) while its worker may still be inside a frame.

  The bound is the same in both, so this is not drift between the two. It is a gap against the author's rule in both, and
  it is fixed in RustEE first:
  - global ingest stays closed until every shard's drain completes (a countdown, as for retired shards);
  - the new core mapping is published in `restart_ingest`, which already republishes it, and no longer at `shard.rs:329`;
  - a replaced clone starts only after the old worker has left its frame (the liveness await).

  The same fix belongs in Java.
- **View skew (drift).** Java routes every push by one volatile mapping (`acquireRoutingState`,
  `LatticeVertex.java:359-373`), so a push that follows another never routes by an older generation. RustEE's `route_frame`
  reads the calling worker's view (`vertex.rs:333`), which changes only when that worker calls `refresh_view` at the start of
  its cycle (`fragment.rs:373`, `views.rs:442-444`). The source's claim orders pass *N* on worker A before pass *N+1* on
  worker B, but it does not order their views, so B can route by an older generation than A did.

  Fix: the interceptor refreshes the calling worker's view when it takes a source's claim. That is one load when nothing
  changed, and the claim's acquire then makes the views monotonic across a source's passes.

With both fixed, frames of one hash never overlap. Lanes keep the checked claim of §2 as a guard all the same.

## 2. The lane's state

- `Arc<Lane>` holds `cell: LaneCell<Inner>` (submitter, state, blocked keys, armed retirements). `LaneCell` is
  `SourceCell`'s claim (one `SeqCst` CAS on entry, one store on exit; `source_cell.rs:26-80`) under a name that does not say
  "source".
- Only the lane's item frames and its poll frame enter the cell, from `execute` or `do_finally`. By §1 they run one after
  another on one thread, so the claim is uncontended and its line stays in that core's L1. The fields inside are plain.
- The claim is checked in every build because of the windows in §1. If entry is refused (an overlap), the frame does not
  wait. It hands its item back to the source (§4) and counts `overlaps`. Aliasing is impossible, and order is as good as the
  core makes it.
- `LaneCx` keeps its shape: `submitter`, `state`, `parts`, `now`, `bridges`, `deliver`. Its `ready` writes to the source's
  ready queue, an MPSC queue the frames produce into and the source drains in `request`/`pull`, instead of a `VecDeque` in
  the source.
- The frame holds its `Arc<Lane>` from the moment it is made (§7), so no reference count moves per item.
- `Drop for Lane` does what `Inner::drop` does today (`source.rs:77-113`). It runs wherever the last reference goes, so the
  source's `complete` must hand the close to lifecycle work (Q6).

## 3. Hashes

| Work | id = routing hash | Why |
|---|---|---|
| Any item from a lane's own intake, with or without `order()` | the lane's hash | It needs the lane's submitter and state, so it runs in the lane cell. `order()` no longer routes. It only says which items a wait holds back (§4). |
| An item from a group's shared intake | the hash of the lane whose source took it | It is bound to the lane the moment it is taken, as it is today (`source.rs:498-507`). |
| The lane's poll frame | the lane's hash | It drains retirements (§5). |
| Successors that touch no lane state | the item's choice; `randomize_hash` for parallel work | They are ordinary frames. |

The lane's hash is `hasher_api::mix(lane id ^ pool id)` and is never randomized. Item frames are ordered, so they leave the
source only through `request` + push, never through `pull` (`cache.rs:395`). Pulls hand over unordered successors.

## 4. Waits: not released until ready

- **Before release.** An item says what it needs with `LaneWork::needs(&self) -> Option<Wait>`. The source owns a gate,
  `LaneGate: Send`, which answers `met` from what the source can read without the cell:
  - markers (`DeviceQueue` is `Sync`);
  - clocks;
  - single-writer counters that the lane's frames publish with a release store (freed buffers, locks run, the probe's
    progress). The source keeps its own plain "taken" count, so room is `freed − taken`.
  
  These atomics are shared between two parties, so no owned state becomes atomic, and none of them is a read-modify-write.
  The source holds an item that is not met, together with the items of its key behind it, in its own `held` queue and
  emits nothing for them.
- **Found while running.** Some needs show up only when the item runs (the hardware queue is full). Then
  `Outcome::Wait(w)` makes the frame hand the item back through the lane's return queue (MPSC, lane → source). The cell
  marks the key blocked, and every later frame of that key hands itself back too, in run order, until the source re-releases
  the key's head with a `resumed` mark. The return queue's order is the order the frames ran, which is the order they were
  released, so the source's `held` queue for the key stays in delivery order. See Q4.
- **A dead held item.** It still needs one look, in one place. Each pass the source reads `is_alive()` on its held items
  (one acquire load each) and emits a dead one at once, wait or no wait. The core then skips it and calls `do_finally`
  (`executor.rs:43-46`), which runs `skipped` in the lane cell. There is no source-level end path: the end is the frame's
  own lifecycle. The items of its key behind it go on, as in 6a. A released item needs no check from the lane at all.
- **Waiting for a value the host writes.** Example: a step whose look needs a grade the host has not stored yet. Its need
  is a gate on a host value, read with one acquire load (the `host_reached` pattern), so the source holds the item.
  - The writer does not call `request`. That call is the downstream's demand, and a writer calling it would run the
    source's pass on its own thread, the very thing this design removes.
  - The writer's own release store is the signal. The source rereads the gate on every pass, and
    workers enter it every cycle while the lattice is busy, and at least once per `IdlePolicy::probe_interval` (1 ms) when
    they sit out (CHANGELOG, 2026-11-03).
  - A timer backstop is then redundant: a held item's gate is reread on every pass, so no rerun timer is needed.
  - If 1 ms is too slow, the writer's frame can, in its `do_finally`, deliver a small wake item into the lane's intake.
    Delivery only enqueues, and the source's next pass releases the held step together with it. This is the "spawn into a
    source" rule of §5.

## 5. Retirements and completions

- **Who observes completion.** `poll_retired` stays an owner call. CUDA's version settles the ticket, releases what the
  stream kept, and probes for a fault (`euhedral-gpu/src/lanes.rs:168-189`), so it runs in the lane cell.
  - Every lane frame drains retirements on entry, as `serve` does today (`source.rs:485`).
  - A lane with retirements outstanding and nothing released gets the **poll frame**: one per lane, made once.
  - The lane publishes `armed` and `retired` totals (single-writer stores). The source emits the poll frame when
    `armed > retired` and the frame is back in its pool, so there is at most one in flight, and it needs no flag.
- **Completions that need order run in the lane.** The `retired` callbacks and the backend's settlement change lane state
  (freed bitstream buffers, released kept memory) and arrive in arm order, so they run in the cell inside whichever lane
  frame drains them.
- **Completions that do not need order are spawned.** What `retired`/`then`/`failed` make for others (an output for the next
  stage, a sink notification) is a frame put into a source: the lane's ready queue, or `Completions::after_marker` when only
  the marker matters. The workers handle it from there. This is the "spawn a frame in `do_finally` and put it in a source"
  rule.

## 6. Lifecycle

| Today | Item frame |
|---|---|
| `admit`/`run_waiting` → `run_one` | `execute`: enter the cell, drain retirements, `X::check`, run the item |
| `then` (after mark or `Finished`) | inside `execute` |
| `retired` | in the cell, in whichever lane frame drains it (usually the poll frame) |
| `failed(S::Error)` | inside `execute` |
| `skipped` (draft 6a, PR #47) | `do_finally` of a frame the core skipped as dead: enter the cell, then `skipped(work, cx)` |
| — | `do_finally_with_error` (a panic in `run`, unroutable at push, the lattice closed): hand the item back; the source ends it with `skipped` on the next lane pass, or the lane's close does |
| — | `Drop` of a frame that still holds its item (a cache `reset_for_next_trial`/close drops frames without `do_finally`, `cache.rs:595-603`): the same hand-back (Q5) |

`LaneFrame::is_alive` delegates to `LaneWork::is_alive`, so the header needs no kill switch and the item's own switch is
read once, by the executor.

## 7. Costs and recycling (the `GrpcFrame` pattern)

The Java gRPC transport (`EuhedralGrpcServerHandler`, `GrpcFrame`, `FrameManager`) makes per-message frames free in steady
state:
- each source owns one `FrameManager` (single consumer, password-guarded);
- each frame is made once with the source's fixed parts: callbacks, kill switch, recycler;
- `replace(message)` swaps the payload and calls `resetHash()`;
- `doFinally` → `recycle()` offers the frame to a bounded MPSC queue, which the owner drains 256 at a time.

RustEE already has the same pieces: `impls::FrameManager`, `FrameFactory`, `Recycler`, and `Frame::recycle`
(`impls/frame_manager.rs`).

| Java | Lane item frame |
|---|---|
| `FrameManager` owned by the handler, consumed under its ingest password | owned by the `LaneSource`'s cell, consumed with `&mut` in `request` |
| `create(id, message)`: callbacks, kill switch, recycler fixed | `create`: `Arc<Lane>`, `FrameHeader::new(lane hash)`, the `Recycler` |
| `replace(message)` + `resetHash()` | `replace(work)`: the item and its delivery stamp. The hash is never changed, so nothing to reset |
| `doFinally` → `recycle()` | `do_finally` → (`skipped` if dead) → `Recycler::recycle_owned` |
| a full recycle queue drops the frame | the same, and a drop that still holds an item hands it back (§6) |

The poll frame and the return path use the same pool.

**Per item, beyond today's intake offer and poll (estimates, not measured):**

| Cost | Detail |
|---|---|
| Allocation | none after warm-up: the bound sizes the pool, as `with_bound` sizes the queues today |
| New read-modify-writes | the recycle offer (+1, plus the `Weak::upgrade` in `Recycler::recycle`, +1, which a strong producer handle would remove); the partition offer and the cache count `fetch_add`/`fetch_sub` (+2 or +3, the subtraction batched under greedy); the cell's CAS (+1; today one per pass) |
| Other new atomics | a successor's ready-queue offer (+1) instead of a `VecDeque` push |
| No lock | none on the path |
| CPU | about 0.2-0.5 µs per item: two queue hops, two routing lookups, the item's lines moving from the releasing core to the lane's |
| Latency | the lane's core must take the frame. That costs nothing when its worker is busy, and up to one park (15 µs, `EMPTY_PARK` / `IdlePolicy::DEFAULT_IDLE_PARK`) when it is idle. Successors go out on the source's next pass, one poll hop (Q3). |

Compare with today's `item_ns` per kind once step 2 lands.

## 8. Metrics

| Metric | Fate |
|---|---|
| `service_ns` | kept: the source's `request`/`pull` |
| `item_ns[kind]` | kept: timed around `run` in `execute`, with `record_owned` (the cell claim hands it from thread to thread) |
| `intake_to_run_ns` | kept: the delivery stamp travels in the frame |
| `ready_depth` | kept: the ready queue's depth at the end of each pass |
| `stranded` | kept |
| `skipped[kind]` | from 6a: counted in the cell |
| new | `held_depth`, `handed_back`, `overlaps` (refused cell entries, which should be zero outside a rebalance), `release_to_run_ns` |

## 9. Migration

Each step can be merged on its own.

1. **Core, one PR each.** The two settled drift fixes from §1:
   - the interceptor refreshes the worker's view when it takes the claim;
   - a rebalance keeps global ingest closed until every shard has drained (bounded by `shutdown_timeout`, 60 s), publishes
     the new core mapping only at the restart, and starts a replaced clone only after its old worker has left.

   Then, if Q5 is answered yes, a cache clear finishes its frames. Only `euhedral-core` and its dependents are tested.
2. **euhedral-graph:** `Lane`, `LaneCell`, `LaneFrame`, the poll frame, the pool, the ready queue and the return queue. The
   `LaneSource` internals are swapped. `LaneWork` gains `is_alive`/`skipped` (6a's API, minus PR #47's source-side check of
   intake items) and `needs`, and `LaneState` gains its `LaneGate`. `LaneCx` keeps its methods. `tests/lane_source*.rs`
   and `lane_intake_alloc.rs` are re-pinned.
3. **euhedral-gpu:** the `LaneSubmit` impls (`src/lanes.rs`) are unchanged. The `Stream` calls already enter the context
   themselves, which matters now that `poll_retired` runs on the lane's core. `open_lane_sources` and the fake and CUDA lane
   tests are re-pinned. **euhedral-benchmarks:** `bin/idle_polls.rs`. **euhedral-memory** does not use `lane_source`.
4. **MySphere** (repin first):
   - `crates/lanes/src/dec.rs`: `Packet`'s kill check (C1, dec.rs:585-593) moves to `is_alive`, and `skipped` drops the
     buffer.
   - `enc.rs`: `Encode`'s check (C2, enc.rs:350-354) moves to `is_alive`, and `skipped` calls
     `sink.rejected(job, Cancelled)`.
   - The dec waits that the gate can read become `needs` (`Room` and `Released`); `HardwareAfter` stays as an in-run
     `Wait`. The same split applies to enc's (`Buffer`, `AfterLocks`, `ClockPassed`, `KeyFrame`).
   - The engine's lanes (`crates/engine/src/lanes.rs:615-733`) choose where lane frames run (Q3).
   - `docs/design/epoch-kill-switch.md` step 6b becomes "C1/C2 move to the frame". Its tests stay, plus a dead held encode
     that is rejected without its `Buffer`. Step 6a (draft PR #47) is superseded by step 2.

## 10. Tests

- **core** (each written to fail first where it pins a fix):
  - `ordered_frames_of_one_hash_never_overlap`: per-hash in-flight counters on a real lattice with greedy, poll, stealing
    and SMT. Asserts at most 1 in flight and push order kept.
  - `pushes_after_a_claim_never_route_by_an_older_generation` (view-skew fix).
  - `a_shard_rebalance_never_runs_one_hash_on_two_cores` (rebalance fix).
  - `a_cleared_cache_finishes_its_frames` (Q5).
- **graph**:
  - `items_of_a_lane_run_in_release_order_one_at_a_time`;
  - `an_item_is_not_released_until_its_gate_is_met`;
  - `a_dead_held_item_is_emitted_and_skipped_without_its_wait`;
  - `a_dead_released_item_is_skipped_by_the_core_and_skipped_runs_in_the_lane`;
  - `an_item_handed_back_mid_run_keeps_its_keys_order`;
  - `retirements_reach_retired_in_arm_order`;
  - `a_lane_with_only_armed_retirements_still_polls`;
  - `a_refused_cell_entry_hands_the_item_back`;
  - `a_frame_dropped_by_a_cache_reset_still_ends_its_item`;
  - `steady_state_allocates_nothing` (counting allocator, as `lane_intake_alloc.rs`).
- **gpu**: `cuda_lanes` and `cuda_lanes_fault`, re-run with retirements polled on the lane's core.

## 11. Migration inventory: sources that execute work in `request`/`pull`

The rule is that a source assembles frames and the workers execute them. Each replacement below follows the same pattern:
- ordered frames with `id_hash == routing_hash` where order matters;
- work held until it is ready;
- frames recycled as `GrpcFrame` is;
- completions spawned from `do_finally` into a source.

Costs are from the docs cited, not measured here. The list runs from highest cost or hottest to lowest. MySphere sites are
at `arch/next-4` e4a5984b.

| # | Site | What runs in the pass | Frame-based replacement |
|---|---|---|---|
| 1 | MySphere `dec` lanes ×2 (`crates/engine/src/lanes.rs:669-739` → RustEE `source.rs:478-533`) | `cuvidDecodePicture` in the item: p50 3.83 ms in a burst (cuda-move.md:638), and a decoder tail of 1-3 ms (R-S2) | this design: a dec item frame per packet; `Room` as a gate; a dead packet skipped by the core |
| 2 | MySphere `enc`, `enc-vp`, `enc-aux` lanes (same sites) | `nvEncEncodePicture` (µs) and the lock item: 0.01-0.1 ms after the clock, 6-12 ms if early (cuda-move.md:661, 841) | this design: `Buffer`/`AfterLocks`/`ClockPassed` as gates; a dead encode's `skipped` calls `rejected(Cancelled)` |
| 3 | MySphere compute lanes and `ofa` (lanes.rs:669-739) | kernel launches and graph replays per stage, several per picture (µs each; a graph instantiation on a miss is long, R-G3); `nvOFExecute` about 20 µs (cuda-move.md:1506-1512) | this design; graph builds stay in the lane cell's frame |
| 4 | MySphere `infer` lane (lanes.rs:669-739) | the TensorRT enqueue (µs) or ORT (about 0.3 ms) (cuda-move.md:714) | this design |
| 5 | MySphere `LinkSource::request` → `deliver` → `make` (`crates/engine/src/admission.rs:706, 665, 415-664`) | per admitted picture: pool slots, the lease, an analysis record decoded inline (`on_admit`, :615, under 0.1 ms), and a `Task` boxed per picture | `make` keeps the gate (pools, credit) and binds slots, which is assembly. The record decode moves into the admitted frame (`SOURCE_LANE`, already ordered). The `Task` frames come from a `FrameManager` |
| 6 | MySphere `xfer` lane (lanes.rs:669-739) | copies (µs) | this design |
| 7 | RustEE `euhedral-memory` `Moves::emit` → `pump` (`tiers.rs:1155-1164`, `866-906`) | finishes moves, starts second-hop copies (driver calls), frees slots, and allocates a `Vec` per pass | the source only emits a pump frame when the moves queue has entries: one per tier, recycled, ordered on the tier's hash. Retired moves are spawned from that frame's `do_finally` |
| 8 | MySphere offline analyser lanes (`crates/bins/analyze/src/run.rs`, `ana.rs`) | decode and inference items, as 1-4 | this design (offline: it moves with the repin) |
| 9 | RustEE `AsyncIo` (`euhedral-io/src/lib.rs:917-957`, `uring.rs:623-651`) | reaping completions (fine), plus flushing waiting SQEs (one `io_uring_enter`) and requeueing `EAGAIN` entries inside the pass | the reap stays. The flush moves to the submitting frame, or to a frame the source emits when entries wait. Continuations must only build a frame (`lib.rs:54-55`); MySphere's browser chain already follows that (`chain.rs:4-5`) |
| 10 | RustEE `IdleWatch::poll` (`lake.rs:562-570`) | detaching a busy period's sinks (a lock; lifecycle work), once per idle period | a lifecycle frame the watch emits (rare; low priority) |
| — | RustEE `QueueSource`, `SingleUseSource`, `LatticeHotSource` (`ingest/queue.rs:393`, `single_use.rs:79`, `hot_source.rs:187`) | only hand out frames | fine |
| — | RustEE `Completions` (`completions.rs:436, 514`); MySphere `Ready` impls (tensorrt `lib.rs:144`, `core.rs:474`, analyze `source.rs:134-146`, nvdec `decoder.rs:1885`) | one query per pending operation, then emit | fine. nvdec's `Token` takes a `try_lock` in the query (`decoder.rs:1900`): a lock-free-rule item, not this rule's |
| — | RustEE `DeferredCalls` (`deferred.rs:576-603`, `poll` :410-417) | decides, then pushes one key frame; the calls run in that frame | fine |
| — | RustEE `GiveBacks` (`planner.rs:1241-1285`) | wraps orders into frames | fine as to execution. Its frames are `Box`ed per order (recycle them), and `wake` pushes from outside `request` (HOT-FIX S1, as `Moves` S2) |
| — | MySphere `ClaimCell`/`SourceCell` state: viewport lane, link lane, rings, the CUDA restore and interp books (`viewport.rs:642`, `core.rs:219`, `ring.rs:440`, `restore.rs:324`, `interp.rs:214`) | none: entered by frames or lane items, never by a source pass | already the lane-cell pattern. The books move with their lanes in 1-4 |

**Order.**
- Sites 1-4, 6 and 8 all move at once with step 2 of §9, because the `LaneSource` change moves every lane. MySphere then
  adopts the gates and placement in table order: dec, enc, compute/ofa, infer, xfer.
- Site 5 is MySphere's own and can move in parallel.
- Site 7 is next in RustEE, then sites 9 and 10.

## Questions for the user

- **Q1 and Q2: settled by the author** (fixed in §9 step 1). The rebalance closes global ingest, drains for up to a minute,
  then reopens; this is Java's mechanism, and its code reopens early, as RustEE's does. The view skew is drift and is fixed
  to Java's single-mapping guarantee.
- **Q3. Where lane frames run.** Hashed onto the lattice's cores, one lane can share a core with other lanes and other
  ordered work, and a submission blocked on a driver lock stalls that core's owner.
  - *Recommendation:* the lane source takes an optional origin per lane (a CPU with `CacheLocal`), and the engine places
    each lane on its own CPU.
- **Q4. Waits found while running.** Keep the hand-back for needs that only `run` can find, or require every wait to be a
  gate before release?
  - *Recommendation:* keep the hand-back as the fallback (the full hardware queue is found only by trying) and make the
    common waits gates.
- **Q5. Frames a cache drops.** `reset_for_next_trial` and a closed cache drop their frames without `do_finally`. Should
  the core finish them?
  - *Recommendation:* yes, with `do_finally`. Lane frames keep the `Drop` hand-back until then.
- **Q6. Successor latency and the lane's close.** Successors go out on the source's next pass (one poll hop, as with a
  `Lake`), and the lane closes where its last reference drops.
  - *Recommendation:* accept the hop, and have `complete` hand the close (quiesce and state close) to a lifecycle thread.
