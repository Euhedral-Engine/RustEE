# Lane sources Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A device lane runs as one `LatticeSource`. It owns the lane's submitter and per-lane state, takes work from its
own intake and its lane group's shared intake, and pushes ready frames within the demand it has banked. The plan also
builds the cell that makes that source's state lock-free, the re-armable fan-in cell for pooled units, and the CUDA binding.

**Architecture:**
- `euhedral-data-structures` gains `SetOnce`.
- `euhedral-core` gains:
  - `SourceCell`, the state behind the lattice's single-entry guarantee;
  - `DownstreamSlot`, a source's one receiver;
  - `LatticeReceiver::serialize`, the interceptor's own hand-off for releasing the source it holds. A source's
    lifecycle calls set atomics, and the source applies them in its next `request` or `pull`.
- `euhedral-graph` gains:
  - the backend-neutral `LaneSubmit` trait;
  - `LaneSource`;
  - the intakes and their delivery;
  - `LanePoolBuilder`;
  - the re-armable `Join`.
- `euhedral-gpu` implements `LaneSubmit` for its streams and opens lane sources in one call.
- Every concurrent field mirrors its counterpart in the upstream Java runtime (§"Java counterparts").

**Tech Stack:**
- Rust 1.99.1 (edition 2024, `mise.toml`).
- The workspace crates. Nothing new at run time.
- `loom` as a `cfg(loom)` dev-dependency of `euhedral-data-structures` and `euhedral-graph`; the lock audit approved it.

**Specs** (read them before any task):
- the lock-free hot-path audit, `docs/superpowers/specs/2026-10-28-lock-free-hot-path-audit.md` on branch `lock-audit`
  (§2.0, §3.0, §3.5 and chunks 2 and 5 of §4);
- the GPU-side design's RustEE requirements RE-1 to RE-6. The GPU-side design is the application's document; its
  requirements are restated in full below, so no task depends on reading it;
- `docs/superpowers/specs/2026-10-28-euhedral-gpu-design.md` (merged).

The requirements, restated:

| # | Crate | Requirement |
|---|---|---|
| RE-1 | core | `SourceCell<T>` in the audit's shape. `unsafe fn enter(&self) -> SourceGuard<'_, T>`. In debug builds `enter` panics on a second thread or a nested entry. A source's lifecycle calls set atomics only, and the state they concern is applied inside the source's next `request` or `pull`, where it is single-consumer. |
| RE-2 | graph | `LaneSubmit`: the trait a lane source drives. It has `lane`, `queue`, `await_marker` and `mark`, arms a retirement, has `poll_retired`, and lends the queue's raw handle to an item. |
| RE-3 | graph | `LaneSource<S, X, W>`, a `LatticeSource` per lane. **Each `request(demand)`:**<br>1. banks the demand;<br>2. drains retirements and runs their continuations;<br>3. calls `X::check`, one read of each hardware state;<br>4. runs the released waiting items, then its intake, then the shared intake. It runs every item whose dependencies are met, with no budget. It awaits input markers, and hands another backend's markers to the item. On "done" it marks, delivers successors, queues frames and arms a retirement if asked. "Waits for X" keeps the item in the source, and same-key items wait behind it;<br>5. pushes at most the banked demand of ready frames.<br>**`pull(demand)`** hands over at most `demand` ready frames, never banks demand and never submits work. **Metrics:** service time per call and per item kind, intake-to-run latency, ready-queue depth. |
| RE-4 | graph | Delivery into intakes at the top. A lane's own intake is MPSC; a lane group's shared intake is MPMC, lock-free, with pooled chunks. There is no lane choice: unbound work runs on whichever source of its group a worker enters first. Delivery is not a lattice push and spends no demand. A pool may hold lanes of several backends. |
| RE-5 | graph | A fan-in cell for recycled units: the audit's `Join` (G4), able to live inside a pooled object and be re-armed with no allocation. Each part ends with a value or abandons; the last arrival runs the continuation on the values. |
| RE-6 | gpu | `Stream` implements RE-2. `open_lane_sources(names, options per lane, state per lane)` returns the pool with its sources. `StreamOptions` gains a priority: the CUDA stream priority, a device property. |

## Decisions made by the planner

1. **RE-1 is the audit's chunk-2 primitives, nothing more.** Part 1 builds the four pieces chunk 2 "Adds":
   - `SetOnce`;
   - `SourceCell`;
   - `DownstreamSlot`;
   - `LatticeReceiver::serialize` with its implementation on `UpstreamInterceptor`, used by the interceptor itself.
   The conversions of the existing sources stay in the audit's own chunk 2, which consumes these primitives: ingest
   (C11-C13), `Completions` (G1-G3), `AsyncIo` (I1, I2, I5), C15, S4 and S5. Nothing in this plan changes them.
2. **The audit's blocking-work facility is not built.** The user corrected the audit: "A worker is a cpu… long calls run
   on the worker". A long call in an item or a frame runs on the worker that reaches it. `LaneSource` has no class of
   blocking work and no thread.
3. **Java mirrors win over the audit's sketches where they differ** (the user's rule, given 2026-10-28). Each case is
   listed under "Java counterparts":
   - **`DownstreamSlot`** follows `AbstractIngestSink.Delegate`: the flags `attached`, `complete` and `live`. It does
     not use the audit's four-state `AtomicU8`.
   - **The interceptor's claim** stays Java's `wip` counter. The audit's owner token is not needed: a `serialize` from
     the holder's own thread loses the claim like any other caller, and the holder runs the action when it takes the
     claim back after its release.
   - **A source's lifecycle calls touch atomics only**, as Java's `complete`, `clear` and `kill` do: the source applies
     them on its next `request` or `pull`, under the lattice's serialization. `DownstreamSlot` therefore has no
     `serialize`.
   - **`LaneSource`'s banked demand** is Java's `demand` atomic, not a plain field in the cell.
4. **Delivery lives on a new `LaneIntakes<W>`, not on `LanePool`.** `LanePool` stays non-generic, because
   `Completions`, `StageGraph`, `euhedral-memory` and `euhedral-gpu`'s `Lanes` use it. `LaneIntakes::deliver(work, to)`
   is the requirement's `LanePool::deliver`.
5. **Lanes of several backends share one pool through `LanePoolBuilder`.** A backend opens its queues at lane ids the
   builder reserved (`Gpu::open_lanes_at`).
   - A backend's `await_marker` refuses another backend's marker.
   - `LaneSubmit::accepts` tells the source which markers to hand to the item instead.
6. **The `check` hook only refreshes state.** Work that follows a hardware state is an item that answers
   `Outcome::Wait(X::Wait)`. The source asks `X::met` once per request for each waiting head.
   - The design describes the hook as doing the map/copy/mark itself. The same work runs as an item, so it goes through
     one code path, the item's.
   - `X` needs no access to the submitter, which keeps `LaneState` free of the other type parameters.
7. **There is no built-in retirement wait.** A retirement's continuation delivers what follows it, or updates the
   per-lane state that an item's `met` reads. The design's "waits for a retirement" is one of those two.
8. **`Join` keeps one value slot per part.**
   - Java's `Join` carries only the first failure. The successor of a join needs every branch's marker, and Java gets
     those from its `StageFrame` predecessors' plain `marker` fields, read after `arrive()`'s count.
   - A per-part slot written before the count's read-modify-write is that same pattern in one cell.
   - The double-end check is debug-only, because Java has none.
9. **Out of scope:**
   - RE-7 to RE-12. RE-9 (`raw`) is in, as part of RE-2/RE-6, because the design's `LaneSubmit` lends the raw handle.
   - The rest of chunk 5: `OneShot`, `Latch`, `Sequencer` and the lake.
   - `Completions`, which the GPU-side design stops using for device work.
10. **Four draft PRs, stacked where they depend on each other:**
    - Part 1 (`lane-source`, this branch) and Part 2 (`lane-source-join`, from `origin/main`) are independent.
    - Part 3 (`lane-source-lanes`) stacks on Part 1.
    - Part 4 (`lane-source-gpu`) stacks on Part 3.
    - PRs are opened with `gh pr create --draft`. No agent marks a PR ready.

## Java counterparts

Two Java code bases are the reference for every concurrent field this plan touches. Mirror their fields, atomics and
orderings, and deviate only for a genuine Java bug or a missing Rust equivalent. Each deviation is listed with its reason.
- **`euhedral-core` and `euhedral-data-structures` places** (sources, the upstream queue and interceptor, the hot source
  and receiver, the queues): the `core` package of the Java execution runtime's `euhedral-core` module.
- **`euhedral-graph` places** (lanes and `LanePool`, completions, join, latch, sequencer, stage graph, lake, and the new
  lane source and delivery): the Java runtime that `euhedral-graph` is derived from. That means its `runtime/graph`
  package (`LanePool`, `StageFrame`, `Join`, `Sequencer`, `FrameLake` and its lake implementation), its `gpu` package (`GpuStream`
  and its CUDA implementation) and its architecture notes 03 (execution engine), 07 (CUDA execution) and 08
  (concurrency and ownership).

**How that runtime drives lanes, which `LaneSource` and delivery start from:**
- A ready stage (`StageFrame`) is a frame published into the lake (the `FrameLake` implementation's `publish`).
  - The lake is several queue ingest sinks, each its own lattice source.
  - The frame's routing hash picks the sink, and "a sink is like a GPU stream": one producer partition keeps publish
    order, so equal hashes keep one sink's order.
- Whichever worker runs the stage picks its lane from the shared `LanePool` (`final GpuStream[] lanes`). In
  `StageFrame.submit` it then:
  - awaits each predecessor's `marker` that ran on another lane, skipping same-lane and host predecessors;
  - submits on the stream;
  - `mark`s its reusable marker.
  A host-only stage records no marker.
- Arrivals are `ARRIVALS.getAndAdd` (volatile). The last arrival publishes the successor, which reads its predecessors'
  plain `marker`/`lane` fields.
- A deferred stage satisfies its edges when its asynchronous completion arrives. A failure goes to `owner.fail`, the
  frame publishes no successor, and `doFinally` drops its live count.
- Retirement is `GpuStream.notifyRetired`. Its listener on the driver thread only enqueues (`publishFromCallback`), and
  `confirmRetired` runs on an ordinary thread.
- The lake counts admitted units (`accepted`) and completes once they have all terminated.

**Mapping of access modes, used throughout:**

| Java | Rust |
|---|---|
| plain field | a plain field (owned, or in a `SourceCell`) |
| `getOpaque` / `setOpaque` | `Relaxed` |
| `getAcquire` | `Acquire` |
| `setRelease`, `lazySet` | `Release` |
| volatile access: `get`, `set`, `getVolatile`, `setVolatile`, `compareAndSet`, `getAndSet`, `getAndIncrement`, `getAndAdd`, `decrementAndGet`, `accumulateAndGet` | `SeqCst` |
| `VarHandle.fullFence()` | `fence(SeqCst)` |

| RustEE place (task) | Java counterpart | What is mirrored | Deviation and why |
|---|---|---|---|
| `SetOnce<T>` (1.1) | a reference field set by `VarHandle.compareAndSet(null, v)` and read by `getAcquire` (`AbstractIngestSink.Delegate.getDownstream`) | set: CAS null → value, `SeqCst`; get: `Acquire` | The value is boxed and freed at drop. Rust has no GC. |
| `SourceCell<T>` (1.2) | the plain fields a Java source touches only while its interceptor's `wip` is held (`QueueIngestSink.Delegate`'s consumer side) | plain access, with no atomics | A debug-only entry check, which Java has no counterpart for. Rust needs `UnsafeCell` to express "plain, guarded by the claim". |
| `DownstreamSlot` (1.3) | `AbstractIngestSink.Delegate`: `attached` (`AtomicBoolean`), `COMPLETE`, `DOWNSTREAM`, `addDownstream`, `complete`, `getDownstream` | `attached.compareAndSet(false, true)`; `setVolatile(terminal)` then `getVolatile(COMPLETE)` and `compareAndSet(terminal, null)`; complete is `COMPLETE.compareAndSet(false, true)` then `getAndSet(null)`; `getAcquire` reads; every method touches atomics only | "Downstream non-null" becomes a `live` flag. The receiver `Arc` stays in a `SetOnce` until the slot drops. Clearing it while a worker may be reading it needs reclamation, and Rust has no GC. `attach` also calls `receiver.add_upstream(source)`, because Rust's `LatticeSource` contract has a source announce itself. **No `serialize` on the slot** (it had one in the first draft): Java's lifecycle calls never run source-state work outside `request`/`pull`, and a slot-level hand-off raced an attach and needed a queue with a spin. |
| `UpstreamInterceptor` claim (1.4) | `LatticeVertex.UpstreamInterceptor.wip` (`PaddedAtomicLong`): `acquireLock` is `wip.getAndIncrement() == 0`, `releaseLock` is `wip.setRelease(0)`, and service checks `wip.getOpaque() == 0` | `CachePadded<AtomicU64>`: `try_acquire` is `fetch_add(1, SeqCst) == 0`; the service check is `Relaxed`. Today's Rust `swap(true, Acquire)` flag is replaced | **Release is `wip.swap(0, AcqRel)`, where Java does `setRelease(0)`.** The swap's result is the information Java gets from its separate re-check: 1 means no caller tried the claim while it was held; more means one did, and the holder drains. This keeps an uncontended service to two read-modify-writes (`fetch_add` and the swap) with no fence. **A `Claim` token** (`try_acquire` returns `Option<Claim>`; `pull` and `request` borrow it; `release` consumes it) replaces Java's convention that only a holder calls `pull`/`request`/`releaseLock`: Rust's safe `pub` trait would otherwise let any caller release a claim it does not hold or `pull` while an action runs, and the `unsafe` entry into the source's state would be unsound. The token is not constructible outside the crate and names its handle by a unique id. |
| `LatticeReceiver::serialize` on the interceptor (1.4) | `QueueIngestSink.Delegate.drainRequests` and `QueueIngestSink.discardQueued`: publish a pending flag, then try the claim; the releasing owner re-checks | actions go on an `MpscQueue`; `pending.store(true, SeqCst)`; the claim is tried with `wip.fetch_add(1, SeqCst)`; the holder clears `pending` with `pending.swap(false, SeqCst)` and drains; the release swap (above) says whether to look at `pending` and take the claim back | New method. Java has no lifecycle hand-off to a source, and its nearest structure is the one mirrored here. **`pending` is cleared with one swap, where Java reads it and then sets it false:** Java volatiles have one total order; Rust `SeqCst` loads and stores of separate operations give no such guarantee on every architecture, so the read-and-clear must be one read-modify-write. The release's proof (a serializer's `fetch_add` before the swap makes it return more than 1 and synchronizes with the serializer's earlier `pending` store; one after it finds zero and runs its own action) is a comment on `release_claim`. |
| interceptor `complete` / `on_complete` / `on_error` (1.4) | `UpstreamInterceptor.COMPLETE.compareAndSet(false, true)`, then `upstream.complete()` and `removeUpstream()` | the same | The interceptor also drops its source, through its own `serialize`, when it finishes. Rust has no GC to collect the source/interceptor cycle. The source is installed (compare-and-set, once) before the interceptor is registered, as Java's `addUpstream` sets `upstream` before registering, and `complete` is checked again after the install. A source is identified by a `Weak` (its allocation outlives it while the interceptor lives), so a reused address never matches: Java compares references, which the GC keeps unique. |
| `UpstreamQueue` (none) | `UpstreamQueue` | — | Not touched. |
| `Join` (2.1) | `Join` (Java graph package): `pending` (`AtomicInteger`), `failure` (`AtomicReference`, first wins), a fixed continuation; `expect(n)` is `failure.set(null)` then `pending.set(n)`; `arrive` is `decrementAndGet() == 0`; per-part values as in `StageFrame.arrive` (`ARRIVALS.getAndAdd`, then plain reads of the predecessors' `marker`) | `pending: AtomicUsize`; `reset` stores `SeqCst`; arrivals `fetch_sub(1, SeqCst)`; the last runs the continuation. Values are written plainly before the read-modify-write and read plainly by the last arrival. | Per-part value slots instead of one failure (decision 8). A debug-only `ended` flag per part, and a debug-only "round complete" check on `reset`. The continuation is a stored `Fn` that delivers work, where Java's is a frame published to the lake: same effect, with no allocation per round. |
| `Sequencer`, `Latch`, `Completions`, `StageGraph`, `Lake` (none) | `Sequencer` (`wip` drain over an `MpscQueue`), `FrameLake` and its lake implementation, `StageGraph` | — | Not touched. They are the audit's chunks 5-6 and G1-G3. |
| `LanePool`, `LanePoolBuilder` (3.1) | `LanePool`: `final GpuStream[] lanes`, immutable after construction; `withTransferLane` composes the array before construction | The pool is immutable after `build()`. The builder is plain and single-threaded, before the pool is shared. | `LanePoolBuilder` is new; its closest counterpart is `withTransferLane`'s array composition. `choose` and `nextHome` (lane choice) are not used by lane sources: in the approved design a worker takes work from whichever source it enters. |
| `LaneSubmit` (3.2) | `GpuStream`: `submit`, `openMarker`/`mark`/`await`, `notifyRetired` (listener only enqueues on the driver thread), `confirmRetired` (ordinary thread, returns the device failure) | `await_marker` ≈ `await`, `mark` ≈ `mark`, `arm_retirement` + `poll_retired` ≈ `notifyRetired` + `confirmRetired` (the owner drains, the callback only enqueues), `raw` ≈ the stream handle a `submit` selects | `mark` returns a new marker from the merged `euhedral-gpu` event ring (`Ok(None)` when the ring is full), where Java re-records a reusable per-stage marker. That is the merged crate's design, which has no busy case in Java. Methods take `&mut self` (one owner). |
| `LaneIntakes`: own and shared intakes, `deliver` (3.2) | the lake's `publish` (`FrameLake`): a sink per routing hash, each a `QueueIngestSink` with one producer partition ("a sink is like a GPU stream"), so equal hashes keep publish order; publishing only enqueues (`publishFromCallback` is the same call) | A lane's own intake is one `MpscQueue` (`Multi` producers, `Single` consumer), so delivery order is kept per lane. A group's shared intake is one `MpmcQueue` that every source of the group takes from (the `Multi` consumer's CAS claim, the port of the Java queue's). `deliver` only offers. An order key is the hash that binds work to one lane's intake. | Delivery targets a lane or a group by id, not by hash modulo the sink count: a lane's state (a codec session, a recording) is bound to one queue. |
| item run inside `LaneSource::request` (3.4) | `StageFrame.submit` / `doFinally` | Markers of the item's own lane are skipped (Java's `predecessor.lane == lane`); other lanes' markers are awaited before the work; `mark` after it; a CPU-only item (`Outcome::Finished`) records no marker (Java's `host()`); `retire: true` defers the successors to the retirement, as a deferred stage does; `failed` publishes no successor, as `owner.fail` | The item runs inside its lane's source, not as a frame of its own. A marker of another backend is handed to the item (no Java counterpart: one backend there). |
| `LaneSource` request/pull/demand/complete (3.3-3.6) | demand and guards: `AbstractIngestSink.Delegate` + `QueueIngestSink.Delegate` (`demand` `PaddedAtomicLong` with `accumulateAndGet(accumulate)`, `COMPLETE`, `DOWNSTREAM`; `request` returns when complete, without a receiver or with `demand <= 0`, drains within `demand.getAcquire()` and subtracts what it pushed; `pull` has the same guards and hands over at most its own demand; `complete` is CAS, then `demand.lazySet(0)`, then `getAndSet(null)` → `onComplete`) | `demand: CachePadded<AtomicU64>` with the same `accumulate` (saturating; a negative step floors at 0) in a `fetch_update(SeqCst, SeqCst)` CAS loop; the same guards; `complete` as Java, with `demand.store(0, Release)` | **The lane is a source** (the user's approved design): its submitter has one owner, so one source per lane runs that lane's items under the interceptor's claim. Java lets any worker submit to any `GpuStream` from a `StageFrame`, relying on the driver's per-call thread safety. The merged `euhedral-gpu` submitter is `&mut` (exclusivity by the borrow checker), so no Rust equivalent of "any thread submits" exists without a lock. The lane's state (submitter, waiting items, ready frames, armed continuations) is plain in a `SourceCell`, as `QueueIngestSink` keeps its consumer side under `draining`. `request` runs device work before it pushes (RE-3). `complete` does not drain queued work, as Java's does not ("Does not clear the queue"): what remains is dropped with the source. |
| retirement continuations (3.4) | `CudaGpuMemory.Completion` (`announced` `AtomicBoolean`, `volatile callbackStatus`); listener → `publishFromCallback` | `euhedral-gpu`'s retirement queue: the callback only enqueues and the owner drains (merged PR #10) | Not reworked; `LaneSource` pops its continuations from a plain queue in arm order, as the stream's own `PlainQueue` does. |
| `euhedral-gpu` streams (4.x) | `GpuStream`'s CUDA implementation | Unchanged except the lane array and the priority | The priority has no Java counterpart (Java streams are all normal). |

## Global Constraints

- Layering, no cycles: utilities (`euhedral-data-structures`, `euhedral-hashing`, `euhedral-hardware-utils`) → `euhedral-core` → `euhedral-graph`, `euhedral-io` → `euhedral-memory` → `euhedral-gpu`. `euhedral-gpu/tests/layering.rs` enforces it. No dev-dependency or feature may point upward.
- The hot path is lock-free. Locks are for lifecycle only (start, stop, rebalance). Nothing in Parts 1-4 takes a `Mutex`, `RwLock` or `Condvar` on a per-request, per-pull, per-item or per-delivery path. The fakes in `testing` are test code.
- The lattice serializes a source's `request` and `pull` (single consumer). `SourceCell` relies on it.
- **`request` is always called and pushes all ready work within the banked demand. `pull` may never be called and never banks demand.** No work is reachable only through `pull`.
- Readiness is selective admission: a source with nothing ready gives nothing. There are no "has work" flags and no wake paths.
- Long calls run on the worker that needs them. There is no blocking facility and no thread of ours.
- Text in code, docs, comments, READMEs, CHANGELOG and GLOSSARY names no other repository and no application domain, and makes no negative self-descriptions ("builds on X", never "does not depend on Y"). Commit messages may name repositories.
- Every `unsafe` block has a `// SAFETY:` comment. `euhedral-graph` stays `#![deny(unsafe_code)]`; the modules that need `unsafe` carry a reasoned module-level `allow`.
- Gate before every commit: `mise exec -- cargo fmt --all --check` and `mise exec -- cargo clippy --workspace --all-targets -- -D warnings`. From Part 4 on, also `mise exec -- cargo clippy -p euhedral-gpu --all-targets --features cuda -- -D warnings`.
- Set `CARGO_TARGET_DIR` to the machine's build directory before any command. GPU tests run under the machine's GPU scheduler.
- Each part ends with:
  - a CHANGELOG entry (What, Why, How, Tests), as the file's existing entries;
  - GLOSSARY terms for every new public concept;
  - the crate README's module table.
- Commits end with the session line the coordinator gives. PRs are drafts.

## Review Focus

1. **A `serialize` racing a worker's release must never strand an action.** Expected: every action runs exactly once, never while a `pull` or `request` of the source is in progress. Test `release_races_serialize_and_no_action_is_stranded` (Task 1.4).
2. **A source whose `request` is the only entry still delivers everything.** Expected: every frame is pushed, never more than the cumulative demand, in ready order, even when demand arrives before the work. Test `a_source_driven_only_by_request_delivers_every_frame_within_demand` (Task 3.4).
3. **A long item on one source must not hold group work.** Expected: the group's other sources take the shared intake while one worker is inside a long item, and the shared intake's claim is never held across an item. Test `a_source_inside_a_long_item_leaves_group_work_to_the_others` (Task 3.5).
4. **A full marker ring must stall a lane without losing an item.** Expected: the item that could not be marked is marked first at the next entry, and nothing else runs on that lane before it. Test `a_busy_mark_stalls_the_lane_until_the_ring_frees` (Task 3.4).
5. **A join re-armed in its own continuation must start a clean round.** Expected: no value of the previous round is seen and no arrival is lost. Tests `reset_from_inside_the_continuation_arms_the_next_round` (Task 2.1) and the loom model (Task 2.2).

## File map

| File | Part | Responsibility |
|---|---|---|
| `euhedral-data-structures/src/sync/mod.rs`, `sync/set_once.rs`, `sync/loom.rs` | 1 | `SetOnce`; the loom shim for the atomics |
| `euhedral-data-structures/Cargo.toml`, `src/lib.rs` | 1 | `pub mod sync`; `[target.'cfg(loom)'.dev-dependencies] loom = "0.7"`; `check-cfg` for `loom` |
| `euhedral-core/src/generics/source_cell.rs` | 1 | `SourceCell`, `SourceGuard` |
| `euhedral-core/src/generics/downstream_slot.rs` | 1 | `DownstreamSlot` |
| `euhedral-core/src/generics/lattice.rs` | 1 | `LatticeReceiver::serialize` (default: run at once), used by the interceptor |
| `euhedral-core/src/flow_control/vertex.rs`, `vertex/tests.rs` | 1 | the `Claim` token, the `wip` claim, action queue, `serialize`, installing and dropping its source |
| `euhedral-graph/src/join.rs`, `src/sync.rs`, `tests/join_alloc.rs` | 2 | the re-armable `Join`; the loom shim; the allocation count |
| `euhedral-graph/src/lanes.rs` | 3 | `LanePoolBuilder` |
| `euhedral-graph/src/lane_source/mod.rs` | 3 | `LaneSubmit`, `Retired`, `LaneState`, `LaneWork`, `Outcome`, `LaneCx` |
| `euhedral-graph/src/lane_source/intake.rs` | 3 | `LaneIntakes`, `LaneInputs`, `Deliver`, `GroupId` |
| `euhedral-graph/src/lane_source/source.rs` | 3 | `LaneSource` |
| `euhedral-graph/src/lane_source/metrics.rs` | 3 | `LaneMetrics` |
| `euhedral-graph/src/testing.rs` (+ `testing/lanes.rs`) | 3 | `FakeLanes`, `FakeLane`, `FakeOp`: a stepped fake `LaneSubmit` |
| `euhedral-graph/tests/lane_source.rs`, `tests/lane_source_lattice.rs`, `tests/lane_intake_alloc.rs` | 3 | the fake suite, a real lattice, the allocation count |
| `euhedral-gpu/src/api.rs` | 4 | `StreamPriority`, `StreamOptions::priority`, `Gpu::open_lanes_at` |
| `euhedral-gpu/src/lanes.rs` | 4 | `LaneSpec`, `OpenedLanes`, `open_lane_sources` |
| `euhedral-gpu/src/cuda/{stream.rs,fns.rs,mod.rs}`, `src/fake/{stream.rs,mod.rs}` | 4 | `LaneSubmit` for `CudaStream` and `FakeStream`; lane arrays with holes; priority; `raw` |
| `euhedral-gpu/tests/fake_lanes.rs`, `tests/cuda_lanes.rs` | 4 | the fake suite; the CUDA integration test (feature `cuda`) |

---

## Part 1 (draft PR 1, branch `lane-source`): the source invariant's primitives (RE-1)

### Task 1.1: `SetOnce<T>`

**Files:**
- Create: `euhedral-data-structures/src/sync/mod.rs`, `src/sync/set_once.rs`, `src/sync/loom.rs`
- Modify: `euhedral-data-structures/src/lib.rs` (`pub mod sync;`), `euhedral-data-structures/Cargo.toml`
- Test: unit tests in `set_once.rs`; the loom model in `set_once.rs` under `#[cfg(loom)]`

**Interfaces:**
- Produces:

```rust
pub struct SetOnce<T> { /* AtomicPtr<T> */ }
impl<T> SetOnce<T> {
    pub const fn new() -> Self;
    /// Installs `value` unless one is set: CAS null → Box::into_raw (SeqCst, as VarHandle.compareAndSet). On failure the
    /// box is rebuilt and the value handed back.
    pub fn set(&self, value: T) -> Result<(), T>;
    /// Acquire load (as getAcquire).
    pub fn get(&self) -> Option<&T>;
    pub fn get_mut(&mut self) -> Option<&mut T>;
    pub fn into_inner(self) -> Option<T>;
}
// Drop frees the value once. Send if T: Send; Sync if T: Send + Sync. Default, Debug (shows Some/None only).
```

- `sync/loom.rs`: under `cfg(loom)`, `pub(crate) use loom::sync::atomic::{AtomicPtr, AtomicBool, AtomicUsize, Ordering};`; otherwise the same names from `std`.

- [ ] **Step 1: Write the failing tests.**
  - `set_hands_the_value_back_when_one_is_set`: `set(1)` is `Ok`; `set(2)` is `Err(2)`; `get()` is `Some(&1)`.
  - `get_is_none_until_set`.
  - `a_set_value_is_dropped_exactly_once`: drop counter; also after a refused `set`, whose value is dropped by the caller.
  - `racing_setters_install_exactly_one`: 8 threads × 1000 rounds; exactly one `Ok` per round, and every thread then reads the winner's value.
  - `into_inner_and_get_mut_return_the_value`.
- [ ] **Step 2: Run them and see them fail.** `mise exec -- cargo test -p euhedral-data-structures sync::set_once`. Expected: FAIL (no module).
- [ ] **Step 3: Implement `SetOnce`** with the orderings above.
- [ ] **Step 4: Add the loom model** `loom_one_setter_wins_and_every_reader_sees_it`. Two threads `set`, a third `get`s. With `loom::model`, exactly one set succeeds, and a `Some` read equals the winner's value.
- [ ] **Step 5: Run the tests.**
  - `mise exec -- cargo test -p euhedral-data-structures sync::`: PASS.
  - `RUSTFLAGS="--cfg loom" mise exec -- cargo test -p euhedral-data-structures --release --lib sync::set_once::loom`: PASS.
- [ ] **Step 6: Commit.** `git commit -m "euhedral-data-structures: SetOnce, a lock-free set-once cell"`

### Task 1.2: `SourceCell<T>`

**Files:**
- Create: `euhedral-core/src/generics/source_cell.rs`
- Modify: `euhedral-core/src/generics/mod.rs` (`pub use source_cell::{SourceCell, SourceGuard};`)

**Interfaces:**
- Produces, in the audit's shape:

```rust
/// State that only a source's `pull` and `request` (and actions `serialize` runs) touch.
pub struct SourceCell<T> { value: UnsafeCell<T>, #[cfg(debug_assertions)] inside: AtomicUsize /* holder's thread token, 0 free */ }
impl<T> SourceCell<T> {
    pub const fn new(value: T) -> Self;
    /// # Safety
    /// The caller is inside this source's `pull` or `request`, or in an action `serialize` runs, and holds no other
    /// guard of this cell.
    pub unsafe fn enter(&self) -> SourceGuard<'_, T>;
    pub fn get_mut(&mut self) -> &mut T;
    pub fn into_inner(self) -> T;
}
pub struct SourceGuard<'a, T>; // Deref + DerefMut to T; Drop clears `inside` (debug)
unsafe impl<T: Send> Sync for SourceCell<T> {}
```

- The debug token is the address of a thread-local, so it is non-zero and unique per live thread.
- `enter` does `inside.compare_exchange(0, token, AcqRel, Acquire)`. On failure it panics:
  - `"SourceCell entered again on the same thread (a nested entry)"` when the holder is this thread;
  - `"SourceCell entered from another thread while one is inside"` otherwise.
- The guard's drop stores 0 (`Release`).
- Release builds have no field and no check.

- [ ] **Step 1: Write the failing tests.**
  - `a_guard_reads_and_writes_the_value`.
  - `#[cfg(debug_assertions)] #[should_panic(expected = "nested entry")] a_nested_enter_panics_in_debug`.
  - **`#[cfg(debug_assertions)] a_second_thread_entering_while_one_is_inside_panics_in_debug`** (the plan's required test that the debug check catches a second entry):
    - thread A enters, then waits on `barrier1` and `barrier2`;
    - thread B, after `barrier1`, runs `catch_unwind(|| unsafe { cell.enter() })`. The result is `Err`, and its message contains `"from another thread"`;
    - B passes `barrier2`, and A drops its guard;
    - B then enters successfully, so the check never sticks.
  - **`#[cfg(debug_assertions)] a_direct_request_racing_the_lattice_is_caught_in_debug`:** a test `LatticeSource` whose `request` enters its `SourceCell<u64>` and blocks on a barrier inside. A second thread calls `request` directly. That call panics with `"from another thread"`. This is design risk R-S3, new code that calls `request` directly.
  - `entries_in_turn_from_many_threads_are_accepted`: 8 threads take turns through a test `Mutex<()>`, enter and increment. The final value is 8000, with no panic.
  - `get_mut_and_into_inner_need_no_entry`.
- [ ] **Step 2: Run them and see them fail.** `mise exec -- cargo test -p euhedral-core generics::source_cell`.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the tests in debug and in release.**
  - `mise exec -- cargo test -p euhedral-core generics::source_cell`: PASS.
  - `mise exec -- cargo test -p euhedral-core --release generics::source_cell`: PASS, with the debug-only tests compiled out.
- [ ] **Step 5: Commit.** `git commit -m "euhedral-core: SourceCell, a source's state under the lattice's single-entry guarantee"`

### Task 1.3: `DownstreamSlot` and `LatticeReceiver::serialize`'s default (as built: no `serialize` on the slot)

**Files:**
- Create: `euhedral-core/src/generics/downstream_slot.rs`
- Modify: `euhedral-core/src/generics/lattice.rs`, `generics/mod.rs`

**Interfaces:**
- Consumes: `SetOnce` (Task 1.1).
- Produces:

```rust
// lattice.rs, on LatticeReceiver:
/// Runs `action` when no `pull` or `request` of the source this receiver serves is in progress, without waiting. The
/// default runs it at once (a receiver that is no lattice: the caller is the only driver).
fn serialize(&self, action: Box<dyn FnOnce() + Send>) { action() }

pub struct DownstreamSlot { attached: AtomicBool, complete: AtomicBool, live: AtomicBool,
                            receiver: SetOnce<Arc<dyn LatticeReceiver>> }
impl DownstreamSlot {
    pub const fn new() -> Self;
    /// Java `addDownstream`: a second attach is told `on_error`; a completed slot tells the receiver `on_complete`;
    /// otherwise the receiver is installed, `live` set (SeqCst), and if `complete` (SeqCst) then reads true, the
    /// `live` CAS true→false that succeeds delivers `on_complete`. On success calls `receiver.add_upstream(source)`.
    pub fn attach(&self, source: Arc<dyn LatticeSource>, receiver: Arc<dyn LatticeReceiver>);
    /// Java `getDownstream` (getAcquire): the receiver while live; never an `Arc` clone.
    pub fn receiver(&self) -> Option<&Arc<dyn LatticeReceiver>>;
    /// Java `complete`: CAS complete false→true (SeqCst); the call that did it swaps `live` to false (SeqCst) and, if
    /// it was live, tells the receiver `on_complete`. Returns whether this call completed the slot.
    pub fn complete(&self) -> bool;
    pub fn is_complete(&self) -> bool; // Acquire
}
// No `serialize` on the slot: a source's lifecycle calls set atomics, and the source applies them in its next
// `request` or `pull`.
```

- [ ] **Step 1: Write the failing tests.**
  - `attach_then_complete_tells_the_receiver_once`.
  - `complete_before_attach_tells_the_late_receiver_once`.
  - `a_second_attach_is_refused_with_an_error`: the second receiver gets `on_error`, and the first is unaffected.
  - `attach_racing_complete_tells_the_receiver_exactly_once`: 2 threads × 10 000 rounds; `on_complete` count == 1 per round.
  - `receiver_is_none_after_complete_and_the_arc_lives_until_the_slot_drops`: `Arc::strong_count` before and after.
- [ ] **Step 2: Run them and see them fail.** `mise exec -- cargo test -p euhedral-core generics::downstream_slot`.
- [ ] **Step 3: Implement** with the orderings in the interface block.
- [ ] **Step 4: Run them.** PASS. Also run `mise exec -- cargo test -p euhedral-core`: every existing `LatticeReceiver` implementation still compiles, because of the default.
- [ ] **Step 5: Commit.** `git commit -m "euhedral-core: DownstreamSlot and LatticeReceiver::serialize"`

### Task 1.4: The interceptor's `wip` claim, `serialize`, and the source it lets go

**Files:**
- Modify: `euhedral-core/src/flow_control/vertex.rs` (`UpstreamInterceptor`), `flow_control/upstream_queue.rs` (`Claim`; `UpstreamHandle` takes it), `vertex/tests.rs`

**Interfaces:**
- Consumes: `SourceCell` (1.2), `SetOnce` (1.1), `LatticeReceiver::serialize` (1.3), `MpscQueue`.
- Produces:
  - **The claim** (Java `wip`): field `wip: CachePadded<AtomicU64>`. It replaces `lock`.
    - `try_acquire` is `wip.fetch_add(1, SeqCst) == 0`.
    - `try_acquire` returns `Option<Claim>`; `pull`/`request` take `&mut Claim`; `release` takes the `Claim`. A claim carries the unique id of its handle and a handle refuses another's.
    - `release` is `while wip.swap(0, AcqRel) != 1 { if !pending.load(SeqCst) || wip.fetch_add(1, SeqCst) != 0 { return } run_actions() }`: only a swap that returns more than 1 looks at `pending` and drains.
  - **Actions:**
    - fields `actions: Producer<Box<dyn FnOnce() + Send>, Multi>`, `drain: SourceCell<Consumer<.., Single>>` and `pending: AtomicBool`;
    - `serialize(action)`:
      1. offer the action;
      2. `pending.store(true, SeqCst)`;
      3. if `wip.fetch_add(1, SeqCst) == 0`, it holds the claim: run the actions, then release as above;
      4. otherwise return. The holder runs the action when its release swap returns more than 1.
    - `run_actions` is `if pending.swap(false, SeqCst) { drain and run every action }`, run by a `serialize` that took the claim and by a holder that took it back.
  - **The upstream:**
    - `upstream`: an install-once pointer slot, installed by compare-and-set in `attach` before the handle is registered (so a worker that finds the handle finds the source); `get` and `take` need the claim. `complete` is checked again after the install;
    - `identity: SetOnce<Weak<dyn LatticeSource>>`: the source's identity, never dereferenced. `pub fn has_upstream(&self, source: &Arc<dyn LatticeSource>) -> bool` replaces `upstream()`, whose only users are two tests;
    - when the interceptor finishes (`complete`, `on_complete`, `on_error`, after the existing `COMPLETE` CAS), it `serialize`s an action that takes the source out of the slot. For `complete`, the action also calls `source.complete()` before the drop.

- [ ] **Step 1: Write the failing tests** in `vertex/tests.rs`.
  - `serialize_runs_at_once_when_no_worker_holds_the_source`.
  - `serialize_runs_at_release_when_another_worker_holds_the_source`:
    1. thread A `try_acquire`s;
    2. thread B `serialize`s, and the action has not run when B returns;
    3. A `release`s, and the action has run, on A's thread (recorded thread id).
  - `serialize_from_inside_the_sources_own_pull_runs_before_release`: a source whose `pull` calls its receiver's `serialize`. The action does not run during `pull`, and it has run when the `UpstreamHandle::release` call returns.
  - `an_action_never_runs_while_a_pull_or_request_is_in_progress`:
    - 4 threads loop 10 000 times: `try_acquire`, `request` (which sets and clears an `inside` flag), `release`;
    - 1 thread `serialize`s 10 000 actions, each asserting `!inside`;
    - every action ran.
  - **`release_races_serialize_and_no_action_is_stranded`:** as above with empty `request`s; the action count == 10 000 after all threads join.
  - `the_interceptor_drops_its_source_when_it_finishes`: a drop-counting source; `on_complete` while a worker holds the claim → not dropped yet; `release` → dropped once.
  - `complete_completes_the_source_through_serialize`.
  - `lifecycle_calls_from_another_thread_touch_only_atomics_and_take_effect_on_the_next_request`: a source with a `SourceCell` and a `DownstreamSlot`; a `clear()` and a `complete()` from another thread while a worker is inside `request` do not enter the cell, a clear is applied by the next `request`, and the source is dropped only after the worker releases.
  - The existing tests at `vertex/tests.rs:373` and `:420` switch to `has_upstream`.
- [ ] **Step 2: Run them and see them fail.** `mise exec -- cargo test -p euhedral-core flow_control::vertex`.
- [ ] **Step 3: Implement.** Keep `catch_unwind` around `pull`/`request` as today. A panicking action is caught, logged and dropped (`log::error!`), and the drain continues.
- [ ] **Step 4: Run the tests.**
  - `mise exec -- cargo test -p euhedral-core`: PASS.
  - `mise exec -- cargo test -p euhedral-core --release flow_control`: PASS.
  - `mise exec -- cargo test --workspace`: PASS. The graph, io and memory lattice suites run on the changed claim.
- [ ] **Step 5: Commit.** `git commit -m "euhedral-core: the interceptor's wip claim serializes lifecycle actions and lets its source go"`

### Task 1.5: Part 1 docs and the draft PR

**Files:** `CHANGELOG.md`, `GLOSSARY.md` (sources section: **SourceCell**, **source invariant**, **DownstreamSlot**, **serialize**), `euhedral-core/README.md`, `euhedral-data-structures/README.md`.

- [ ] **Step 1: Write the CHANGELOG entry.** "The source invariant's primitives (2026-10-29)": What / Why / How / Tests. List the `wip` claim and `upstream()` → `has_upstream` as breaking for code that called `upstream()`.
- [ ] **Step 2: Run the gate:** `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace`. Expected: clean.
- [ ] **Step 3: Commit, push and open the draft PR.**
  - `git push -u origin lane-source`.
  - `gh pr create --draft --base main --title "Source invariant primitives: SetOnce, SourceCell, DownstreamSlot, serialize"`. The body ends with the session line.

---

## Part 2 (draft PR 2, branch `lane-source-join` from `origin/main`): the re-armable fan-in cell (RE-5)

### Task 2.1: `Join` with a fixed capacity, a stored continuation and `reset`

**Files:**
- Modify: `euhedral-graph/src/join.rs` (module-level `#![allow(unsafe_code, reason = "parts own their slots until the count publishes them")]`), `euhedral-graph/src/lib.rs` (doc line on where `unsafe` lives)
- Create: `euhedral-graph/src/sync.rs` (the loom shim, as 1.1), `euhedral-graph/tests/join_alloc.rs`
- Modify: `euhedral-graph/Cargo.toml` (`[target.'cfg(loom)'.dev-dependencies] loom = "0.7"`; `check-cfg`)

**Interfaces:**
- Produces:

```rust
pub struct Join<T> {
    pending: AtomicUsize,               // Java Join.pending
    parts: AtomicUsize,                 // this round's part count (published by `pending`)
    slots: Box<[Part<T>]>,              // capacity; Part { value: UnsafeCell<Option<T>>, #[cfg(debug_assertions)] ended: AtomicBool }
    then: Then<T>,                      // Each(Box<dyn Fn(&mut [Option<T>]) + Send + Sync>) | Once(UnsafeCell<Option<Box<dyn FnOnce(Vec<Option<T>>) + Send>>>)
    #[cfg(debug_assertions)] phase: AtomicU8, // Idle | Armed | Running
}
impl<T: Send + 'static> Join<T> {
    /// One round, unchanged API: armed with `parts`.
    pub fn new(parts: usize, then: impl FnOnce(Vec<Option<T>>) + Send + 'static) -> Arc<Self>;
    /// A cell for a pooled unit: `capacity` slots, the continuation stored once, not armed until `reset`.
    pub fn with_capacity(capacity: usize, then: impl Fn(&mut [Option<T>]) + Send + Sync + 'static) -> Self;
    /// Java `expect`: clears this round's values and arms `parts` (1..=capacity): `pending.store(parts, SeqCst)`.
    /// Valid when the previous round's continuation returned, or from inside it (debug-asserted).
    pub fn reset(&self, parts: usize);
    pub fn arrive(&self, part: usize, value: T);
    pub fn abandon(&self, part: usize);
    pub fn capacity(&self) -> usize;
}
```

- An arrival writes its slot, then `pending.fetch_sub(1, SeqCst)`. Java uses `decrementAndGet`, and the
  `StageFrame.arrive` pattern publishes values written before the read-modify-write.
- The arrival that sees 1 runs the continuation on `&mut slots[..parts]`. `// SAFETY:` every other part's write
  happened-before its read-modify-write, which precedes this one.
- The continuation must not wait. It delivers work, as Java's `Join` publishes to its lake.
- A part index ≥ `parts` panics. Debug builds also panic on an arrival when not armed, and on a part ending twice.

- [ ] **Step 1: Write the failing tests** in `join.rs`. The existing five tests stay; `a_part_ending_twice_or_out_of_range_is_refused` becomes `#[cfg(debug_assertions)]` for the twice case.
  - `a_join_of_64_parts_from_8_threads_runs_once_per_round`: `with_capacity(64)`, 1000 rounds re-armed by the test after each continuation (a counter `== round + 1`). Every value is `Some(i)`.
  - `reset_from_inside_the_continuation_arms_the_next_round`: the continuation counts and calls `reset(2)` until round 100; the slots it sees are only the current round's values (the test writes `round * 10 + part`).
  - `fewer_parts_than_capacity_use_only_the_first_slots`: capacity 8, `reset(3)`, the continuation gets a slice of length 3.
  - `an_abandoned_part_reaches_the_continuation_as_none`.
  - `#[cfg(debug_assertions)] #[should_panic(expected = "round")] reset_before_the_round_completes_panics_in_debug`.
  - `#[cfg(debug_assertions)] #[should_panic(expected = "not armed")] an_arrival_on_a_join_that_is_not_armed_panics_in_debug`.
- [ ] **Step 2: Run them and see them fail.** `mise exec -- cargo test -p euhedral-graph join::`.
- [ ] **Step 3: Implement.** Keep `Join::new`'s behaviour. It arms at construction and runs the `FnOnce` with a `Vec` taken from the slots. That allocation is its only one per round, as today.
- [ ] **Step 4: Run them.** PASS.
- [ ] **Step 5: Commit.** `git commit -m "euhedral-graph: Join is a re-armable fan-in cell with a fixed capacity"`

### Task 2.2: No allocation per round, and the loom model

**Files:** `euhedral-graph/tests/join_alloc.rs`; the `#[cfg(loom)] mod loom` in `join.rs`.

- [ ] **Step 1: Write `a_join_rearmed_10_000_times_allocates_nothing_after_construction`.**
  - Use the counting global allocator pattern of `euhedral-gpu/tests/cuda_alloc.rs`.
  - Make the join with `with_capacity(4)`. Its continuation re-arms and counts.
  - Each round has 4 arrivals from the test thread. Assert the allocations counted after construction == 0.
- [ ] **Step 2: Write the loom model `loom_two_parts_run_the_continuation_once_and_rearm`.** 2 threads arrive with values. The continuation runs exactly once, sees both values, and `reset(2)`s. A second round on the same threads also runs once.
- [ ] **Step 3: Run them.**
  - `mise exec -- cargo test -p euhedral-graph --test join_alloc`: PASS.
  - `RUSTFLAGS="--cfg loom" mise exec -- cargo test -p euhedral-graph --release --lib join::loom`: PASS.
- [ ] **Step 4: Write the docs:** CHANGELOG "Re-armable `Join` (2026-10-29)"; GLOSSARY **Join, fan-in cell**; `euhedral-graph/README.md` table row. Run the gate.
- [ ] **Step 5: Commit and open the draft PR.**
  - `git commit -m "euhedral-graph: Join allocates nothing per round; loom model"`.
  - `git push -u origin lane-source-join`.
  - `gh pr create --draft --base main --title "Re-armable Join for pooled units"`.

---

## Part 3 (draft PR 3, branch `lane-source-lanes`, stacked on Part 1): lane sources (RE-2, RE-3, RE-4)

### Task 3.1: `LanePoolBuilder`

**Files:** `euhedral-graph/src/lanes.rs` (unit tests there).

**Interfaces:**
- Produces:

```rust
#[derive(Debug, Default)]
pub struct LanePoolBuilder { /* Vec<Option<Arc<dyn DeviceQueue>>> */ }
impl LanePoolBuilder {
    pub fn new() -> Self;
    /// The first of `count` consecutive lane ids, reserved for one backend's queues. Panics past 64 lanes.
    pub fn reserve(&mut self, count: usize) -> LaneId;
    /// Puts `queue` at a reserved `lane`. Panics if the lane was not reserved or is already set.
    pub fn set(&mut self, lane: LaneId, queue: Arc<dyn DeviceQueue>);
    /// The pool: lane i is the queue set at i. Panics if a reserved lane is unset.
    pub fn build(self) -> Arc<LanePool>;
}
```

- [ ] **Step 1: Write the failing tests.**
  - `a_built_pool_holds_each_reserved_lane_at_its_id`.
  - `a_pool_of_two_backends_reaches_markers_of_both`: two `Idle`-style queue types, one reserved range each; `reached` is true for each queue's own marker at its lane.
  - `#[should_panic] building_with_an_unset_lane_panics`.
  - `#[should_panic] setting_a_lane_twice_panics`.
  - `#[should_panic] reserving_past_64_lanes_panics`.
- [ ] **Step 2: Run them and see them fail; implement; run them and see them pass.** `mise exec -- cargo test -p euhedral-graph lanes::`.
- [ ] **Step 3: Commit.** `git commit -m "euhedral-graph: LanePoolBuilder for pools of several backends"`

### Task 3.2: The traits, the intakes and the stepped fake

**Files:**
- Create: `euhedral-graph/src/lane_source/{mod.rs,intake.rs}`, `src/testing/lanes.rs` (re-exported from `testing`)
- Modify: `euhedral-graph/src/lib.rs` (`pub mod lane_source;` and the re-exports)

**Interfaces:**
- Produces (`lane_source/mod.rs`):

```rust
pub trait LaneSubmit: Send + 'static {
    type Error: std::error::Error + Send + Sync + 'static;
    type Raw: Copy;
    fn lane(&self) -> LaneId;
    fn id(&self) -> QueueId;
    fn queue(&self) -> Arc<dyn DeviceQueue>;
    /// Whether `await_marker` can order this lane behind `marker` on the device (a lane of this backend in this pool).
    fn accepts(&self, marker: Marker) -> bool;
    fn await_marker(&mut self, marker: Marker) -> Result<(), Self::Error>;
    /// A marker after the work submitted so far; `Ok(None)` when none can be made now (its ring is full). Never waits.
    fn mark(&mut self) -> Result<Option<Marker>, Self::Error>;
    /// Arms a retirement after the work submitted so far; they come back from `poll_retired` in arm order.
    fn arm_retirement(&mut self) -> Result<(), Self::Error>;
    /// The next retirement that arrived, settled (the backend releases what it kept for it).
    fn poll_retired(&mut self) -> Option<Retired<Self::Error>>;
    /// # Safety
    /// The caller enqueues on the handle only from inside an item of this lane, and `mark` afterwards covers it.
    unsafe fn raw(&mut self) -> Self::Raw;
}
pub struct Retired<E> { pub seq: u64, pub failure: Option<E> }

pub trait LaneState: Send + 'static {
    type Wait: Send + 'static;
    /// Step 3 of each request: one read of each hardware state; never a loop or a wait.
    fn check(&mut self, now: Instant) { let _ = now; }
    /// Whether `wait` is met; asked once per request per waiting head.
    fn met(&mut self, wait: &Self::Wait, now: Instant) -> bool;
}
impl LaneState for () { type Wait = std::convert::Infallible; /* met: match *wait {} */ }

pub enum Outcome<A> { Submitted { retire: bool }, Finished, Wait(A) }

pub trait LaneWork<S: LaneSubmit, X: LaneState>: Send + Sized + 'static {
    const KINDS: &'static [&'static str];
    fn kind(&self) -> usize;
    fn waits(&self) -> &[Marker];
    fn order(&self) -> Option<u64> { None }
    fn run(&mut self, cx: &mut LaneCx<'_, S, X, Self>) -> Outcome<X::Wait>;
    fn then(&mut self, marker: Option<Marker>, cx: &mut LaneCx<'_, S, X, Self>) { let _ = (marker, cx); }
    fn retired(self, marker: Marker, failure: Option<S::Error>, cx: &mut LaneCx<'_, S, X, Self>) { let _ = (marker, failure, cx); }
    fn failed(self, error: S::Error, cx: &mut LaneCx<'_, S, X, Self>);
}

pub struct LaneCx<'a, S, X, W> { /* &mut S, &mut X, &mut VecDeque<Box<dyn Frame>>, &Arc<LaneIntakes<W>>, now, lane */ }
impl<S: LaneSubmit, X, W: Send + 'static> LaneCx<'_, S, X, W> {
    pub fn lane(&self) -> LaneId;
    pub fn submitter(&mut self) -> &mut S;
    pub fn state(&mut self) -> &mut X;
    pub fn parts(&mut self) -> (&mut S, &mut X);
    pub fn now(&self) -> Instant;                 // the entry's one clock read
    /// True for a marker the source did not await: another backend's, for the item to bridge.
    pub fn bridges(&self, marker: Marker) -> bool; // marker.queue != own id && !submitter.accepts(marker)
    pub fn deliver(&self, work: W, to: Deliver);
    pub fn ready(&mut self, frame: Box<dyn Frame>);
    pub fn intakes(&self) -> &Arc<LaneIntakes<W>>;
}
```

- Produces (`lane_source/intake.rs`):

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)] pub struct GroupId(pub u16);
#[derive(Clone, Copy, Debug, PartialEq, Eq)] pub enum Deliver { Lane(LaneId), Group(GroupId) }
pub struct LaneIntakes<W> { /* pool, own: Box<[Producer<Envelope<W>, Multi>]>, shared: Box<[Producer<..>]>, group_of */ }
impl<W: Send + 'static> LaneIntakes<W> {
    /// One MPSC intake per lane (`MpscQueue::with_pool(256, 4)`), one MPMC intake per group
    /// (`MpmcQueue::with_pool(256, 4)`); `groups[g]` lists group g's lanes; a lane is in at most one group (panics).
    pub fn new(pool: Arc<LanePool>, groups: &[&[LaneId]]) -> (Arc<Self>, Vec<LaneInputs<W>>);
    /// Never runs the work and never waits. Panics on a lane or group the pool lacks.
    pub fn deliver(&self, work: W, to: Deliver);
    pub fn pool(&self) -> &Arc<LanePool>;
    pub fn group_of(&self, lane: LaneId) -> Option<GroupId>;
}
pub struct LaneInputs<W> { /* lane, own: Consumer<Envelope<W>, Single>, shared: Option<Consumer<Envelope<W>, Multi>> */ }
impl<W> LaneInputs<W> { pub fn lane(&self) -> LaneId; }
// Envelope<W> (private): { work: W, delivered: Instant }
```

- Produces (`testing/lanes.rs`), a stepped fake with no wall clock:

```rust
pub struct FakeLanes; // control side, Clone
pub struct FakeLane;  // LaneSubmit<Error = FakeLaneError, Raw = u64>
#[derive(Clone, Debug, PartialEq, Eq)] pub enum FakeOp { Await(Marker), Mark(Marker), Arm(u64), Raw }
impl FakeLanes {
    /// `count` lanes at `first` (from a LanePoolBuilder), with their queues to `set` into the builder.
    pub fn open(first: LaneId, count: usize) -> (FakeLanes, Vec<FakeLane>, Vec<Arc<dyn DeviceQueue>>);
    pub fn complete_through(&self, lane: LaneId, value: u64); // the queue's completed value
    pub fn retire_all(&self);                                  // every armed retirement arrives, ok
    pub fn fail_retirements(&self, lane: LaneId);              // the next arrivals carry a failure
    pub fn busy_marks(&self, lane: LaneId, n: usize);          // the next n marks return Ok(None)
    pub fn log(&self, lane: LaneId) -> Vec<FakeOp>;            // empty when made with `quiet`
    pub fn quiet(self) -> Self;                                // no log: no allocation per op after warm-up
}
```

- `accepts` is true for markers of `FakeLane`s of the same `open` call.

- [ ] **Step 1: Write the failing intake tests** in `tests/lane_source.rs`.
  - `deliver_to_a_lane_lands_in_that_lanes_own_intake_only`.
  - `deliver_to_a_group_lands_in_its_shared_intake`.
  - `#[should_panic] a_lane_in_two_groups_is_refused`.
  - `fake_lane_marks_awaits_and_retires_in_order`.
- [ ] **Step 2: Run them and see them fail.** `mise exec -- cargo test -p euhedral-graph --test lane_source`.
- [ ] **Step 3: Implement the traits, `LaneIntakes` and `FakeLanes`.** `FakeLanes` may use a `Mutex` internally, because it is a test fixture, as `FakeDevice` is.
- [ ] **Step 4: Run them.** PASS.
- [ ] **Step 5: Commit.** `git commit -m "euhedral-graph: LaneSubmit, LaneWork, LaneState and the lane intakes; a stepped fake lane"`

### Task 3.3: `LaneSource`: construction, attach, `pull`, `complete`, metrics

**Files:** create `euhedral-graph/src/lane_source/{source.rs,metrics.rs}`, with module-level `allow(unsafe_code)` for `SourceCell::enter`.

**Interfaces:**
- Consumes: `SourceCell`, `DownstreamSlot` (Part 1); Task 3.2.
- Produces:

```rust
pub struct LaneSource<S: LaneSubmit, X: LaneState, W> {
    lane: LaneId,
    complete: AtomicBool,                 // Java COMPLETE
    demand: CachePadded<AtomicU64>,       // Java PaddedAtomicLong demand
    downstream: DownstreamSlot,           // Java DOWNSTREAM
    ready_len: AtomicUsize,               // published depth (Relaxed), for queries
    intakes: Arc<LaneIntakes<W>>,
    metrics: LaneMetrics,
    cell: SourceCell<Inner<S, X, W>>,     // plain, under the interceptor's claim
}
// Inner: submitter, state, own: Consumer<Single>, shared: Option<Consumer<Multi>>, ready: VecDeque<Box<dyn Frame>>,
//        waiting: VecDeque<Waiting<W, X::Wait>>, armed: VecDeque<(W, Marker)>, unmarked: Option<(W, bool)>,
//        blocked_keys: Vec<u64> (reused; no allocation after warm-up)
impl<S: LaneSubmit, X: LaneState, W: LaneWork<S, X>> LaneSource<S, X, W> {
    /// Panics if `inputs.lane() != submitter.lane()`.
    pub fn new(submitter: S, state: X, inputs: LaneInputs<W>, intakes: Arc<LaneIntakes<W>>) -> Arc<Self>;
    pub fn lane(&self) -> LaneId;
    pub fn banked(&self) -> u64;      // demand.load(Acquire), Java getDemand
    pub fn ready_len(&self) -> usize;
    pub fn metrics(&self) -> &LaneMetrics;
}
impl LatticeSource for LaneSource<..> { /* add_downstream, pull, request, complete, is_complete */ }

pub struct LaneMetrics {
    pub service_ns: Histogram,          // per request and per pull call
    pub item_ns: Box<[Histogram]>,      // per W::KINDS entry
    pub intake_to_run_ns: Histogram,    // delivery to first run
    pub ready_depth: Histogram,         // sampled at the end of each request
}
```

- `add_downstream` is `downstream.attach(self, receiver)`.
- `complete` (Java): CAS `complete`, then `demand.store(0, Release)`, then `downstream.complete()`. It drains nothing.
- `is_complete` is an `Acquire` load.
- `pull` (Java guards): return 0 when complete, with no receiver, or with `demand == 0`. Otherwise enter the cell and
  hand over up to `demand` frames from `ready`, stopping before the first frame `stop` matches. It never touches
  `self.demand`, never runs an item and never drains retirements.
- `Drop`, which has `&mut self` (no entry needed), runs in this order:
  1. drop the submitter first, a shutdown call that waits for its device work;
  2. then the armed, waiting, unmarked and queued items (their `Drop` ends what they hold);
  3. then the ready frames, each finished with `do_finally_with_error` ("the source was dropped"), as `Completions` cancels late work.

- [ ] **Step 1: Write the failing tests** in `tests/lane_source.rs`.
  - `pull_hands_over_ready_frames_up_to_its_demand_and_leaves_the_bank_alone`: 5 ready frames, `pull(3)` → 3, `banked()` unchanged; `pull` with a `stop` on frame 2 → 1.
  - `pull_never_runs_items_or_drains_retirements`: queued items stay queued (`FakeOp` log empty) after any number of `pull`s.
  - `nothing_is_pulled_before_a_receiver_is_attached`.
  - `complete_tells_the_receiver_once_and_zeroes_the_bank`.
  - `dropping_the_source_ends_every_item_it_still_holds`: item `Drop` counter == items delivered.
  - `metrics_record_service_items_and_intake_latency`.
- [ ] **Step 2: Run them and see them fail; implement; run them and see them pass.**
- [ ] **Step 3: Commit.** `git commit -m "euhedral-graph: LaneSource attaches, pulls ready frames, completes as an ingest source does"`

### Task 3.4: `request`: bank, retire, check, run, push

**Files:** `euhedral-graph/src/lane_source/source.rs`; tests in `tests/lane_source.rs`.

**Interfaces:**
- **`request(d)`**, in Java's order and with its guards.
  - It returns at once when complete, with no receiver, or with `d == 0`.
  - Otherwise `demand.fetch_update(SeqCst, SeqCst, accumulate(+d))`. It enters the cell, takes one clock read `now`, and then:
    1. **retirements:** `while let Some(r) = submitter.poll_retired()`, pop the front of `armed` (debug-assert seq order) and call `item.retired(marker, r.failure, cx)`;
    2. **check:** `state.check(now)`;
    3. **mark retry:** if `unmarked` is `Some`, retry `mark`. If it is still `None`, go to step 7 (push) and stop.
    4. **waiting:** in order, for each waiting item:
       - skip it if its key is blocked;
       - if its `wait` is not `met`, block its key (if it has one);
       - otherwise run it.
    5. **own intake:** poll each envelope. If its key is blocked, park it in `waiting` with no wait of its own (it waits behind the key). Otherwise run it.
    6. **shared intake:** `poll()` one envelope at a time, releasing the queue's claim before the item runs. Debug-assert `order().is_none()`. Run it.
    7. **push:** `n = min(demand.load(Acquire), ready.len())`; pop `n` frames and push them to the receiver; then `fetch_update(accumulate(-n))`. Store `ready_len` (`Relaxed`) and record `service_ns` and `ready_depth`.
  - Steps 4-6 stop early when an item leaves `unmarked` set (the ring is busy).
- **Running one item:**
  1. On its first run, await each wait marker:
     - skip a marker of this lane's own queue;
     - `await_marker` it if `accepts`;
     - otherwise leave it for the item to bridge.
     - A failed await calls `item.failed(e)`.
     Then record `intake_to_run_ns`.
  2. Call `run` and record `item_ns[kind]`.
  3. Act on the outcome:
     - `Submitted { retire }`: `mark`.
       - `Ok(Some(m))`: call `then(Some(m))`. If `retire`, call `arm_retirement` and push `(item, m)` to `armed`; otherwise drop the item.
       - `Ok(None)`: store the item in `unmarked`.
       - `Err(e)`: call `failed(e)`.
     - `Finished`: call `then(None)` and drop the item.
     - `Wait(a)`: push the item to the back of `waiting`, marked as awaited.
- `accumulate(curr, next)` mirrors Java exactly: a negative step floors at 0; a positive step saturates at `u64::MAX`, which means unbounded and is never decremented.

- [ ] **Step 1: Write the failing tests.**
  - **`a_source_driven_only_by_request_delivers_every_frame_within_demand`** (the required test):
    - deliver 1000 items in batches of 37, each `Finished` with one frame tagged by its index;
    - never call `pull`;
    - call `request(d)` with `d` cycling through `[1, 3, 0 (skipped), 7, 2]` until 1000 frames have arrived;
    - after every call, assert `received <= sum of demand so far` and the received tags are `0..received` in order;
    - at the end, `received == 1000` and `banked() == sum - 1000`.
  - `demand_banked_before_the_work_is_ready_is_spent_when_it_becomes_ready`: `request(5)` with nothing queued pushes 0; deliver 3; `request(1)` pushes 3; `banked() == 3`.
  - `nothing_ready_pushes_nothing`: an item that answers `Wait`, then `request(10)`; 0 pushed, no frame and no flag.
  - `request_runs_every_ready_item_with_no_budget`: 500 queued `Submitted` items without frames, then one `request(1)` → the log has 500 marks.
  - `delivery_is_not_a_push_and_spends_no_demand`: items that make no frames; `banked()` unchanged after they run.
  - `input_markers_of_the_own_lane_are_not_awaited` (no `FakeOp::Await` for them).
  - `input_markers_of_another_lane_are_awaited_on_the_device`.
  - `a_foreign_marker_is_handed_to_the_item_not_awaited`: two `FakeLanes::open` calls make two backends in one pool; inside `run`, `cx.bridges(m)` is true and the log has no `Await`.
  - `a_waiting_item_holds_back_its_order_key_and_lets_others_through`: key 7 waits; a later key-7 item does not run; key 8 and unkeyed items do; after `met`, both key-7 items run in delivery order.
  - `a_waiting_item_is_released_when_its_state_is_met`: `X::Wait = u64` deadline; `met` compares with a counter the test advances; released at the next `request`.
  - `a_retirement_runs_the_items_continuation_in_arm_order` (3 items, `retire_all`, then `request(1)`).
  - `a_successor_delivered_by_a_continuation_runs_in_the_same_request`.
  - **`a_busy_mark_stalls_the_lane_until_the_ring_frees`**: `busy_marks(lane, 2)`, 3 items. The first `request`: 1 run, 0 marks, items 2 and 3 not run. The second: still busy, nothing new runs. The third: item 1 is marked first, then items 2 and 3 run and are marked, in that order.
  - `a_failed_retirement_reaches_the_item_as_a_failure` (`fail_retirements`).
  - `a_failed_await_ends_the_item_with_failed`.
  - `#[cfg(debug_assertions)] #[should_panic(expected = "nested entry")] an_item_that_requests_its_own_source_panics_in_debug`.
- [ ] **Step 2: Run them and see them fail.** `mise exec -- cargo test -p euhedral-graph --test lane_source`.
- [ ] **Step 3: Implement `request` as specified.**
- [ ] **Step 4: Run them.** PASS, in debug and with `--release`.
- [ ] **Step 5: Commit.** `git commit -m "euhedral-graph: LaneSource::request banks demand, runs every ready item, pushes within the bank"`

### Task 3.5: Groups on real threads, a real lattice, and steady-state allocation

**Files:** `euhedral-graph/tests/lane_source.rs`, `tests/lane_source_lattice.rs`, `tests/lane_intake_alloc.rs`.

- [ ] **Step 1: Write the failing tests.**
  - `group_work_runs_on_the_first_source_entered`: 3 sources in one group; 10 group items; `request` on source 1 only → source 1's log has 10 marks, and the others have none.
  - **`a_source_inside_a_long_item_leaves_group_work_to_the_others`** (test threads with `#[allow(clippy::disallowed_methods, reason = "test threads stand in for workers")]`):
    1. the first group item blocks on `barrier_a` and `barrier_b`;
    2. thread A requests source 0, which takes that item and blocks;
    3. after `barrier_a`, thread B requests source 1, which runs items 2-10;
    4. B passes `barrier_b`.
    Assert source 1's log has 9 marks before A returns, and source 0's has 1.
  - `the_shared_intake_is_free_while_an_item_runs`: inside an item on source 0, `shared_consumer_probe.poll()` from the item's own thread returns the next envelope (the claim is not held). The test reaches the shared queue through a test-only accessor on `LaneIntakes` behind `cfg(feature = "testing")`.
  - `lane_sources_on_a_real_lattice_deliver_every_frame_once` (`tests/lane_source_lattice.rs`; lattice helper as in `tests/credit_graph.rs`):
    - 4 workers; 3 group sources and 1 own-lane source attached with `add_upstream`;
    - 20 000 items, a third with retirements (a test thread calls `retire_all` until done, with a 30 s bound);
    - each frame increments a per-index counter;
    - every counter == 1 and no worker panicked;
    - in debug builds the `SourceCell` check is armed throughout.
  - `steady_state_delivery_and_requests_allocate_nothing` (`tests/lane_intake_alloc.rs`, counting allocator):
    - `FakeLanes::open(..).quiet()`; items with no frames;
    - warm up 2000 deliveries and requests;
    - then 20 000 more, half to a group and half own, with retirements;
    - allocations == 0.
- [ ] **Step 2: Run them and see them fail; fix the implementation where they find a fault; run them and see them pass.**
  - `mise exec -- cargo test -p euhedral-graph --tests`.
  - `mise exec -- cargo test -p euhedral-graph --release --tests`.
- [ ] **Step 3: Commit.** `git commit -m "euhedral-graph: lane groups on real threads and a real lattice; no allocation in steady state"`

### Task 3.6: Part 3 docs and the draft PR

**Files:** `CHANGELOG.md` ("Lane sources (2026-10-29)"); `GLOSSARY.md` (**lane source**, **intake**, **shared intake**, **lane group**, **banked demand**, **lane item**, **waits for**); `euhedral-graph/README.md` (module row and a short "A lane is a source" section with the fake example compiled as a doctest).

- [ ] **Step 1: Write the docs.** Run the gate and `mise exec -- cargo test --workspace`.
- [ ] **Step 2: Commit, push and open the draft PR.**
  - `git push -u origin lane-source-lanes`.
  - `gh pr create --draft --base lane-source --title "Lane sources: LaneSubmit, LaneSource, intakes, LanePoolBuilder"`.

---

## Part 4 (draft PR 4, branch `lane-source-gpu`, stacked on Part 3): the CUDA binding (RE-6, with RE-9's `raw`)

### Task 4.1: Lanes at reserved ids, and the priority option

**Files:** `euhedral-gpu/src/api.rs`, `src/cuda/{stream.rs,mod.rs,fns.rs}`, `src/fake/{stream.rs,mod.rs}`; tests in `tests/fake_streams.rs`.

**Interfaces:**
- Produces:

```rust
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum StreamPriority { #[default] Normal, High }
pub struct StreamOptions { pub markers: u32, pub priority: StreamPriority } // Default: 256, Normal
// on Gpu:
/// Streams for lanes `first..first + names.len()` of a pool built elsewhere (LanePoolBuilder::reserve), one options
/// entry per name. Their markers name those lanes; a marker of a lane not opened by this call is refused by
/// `await_marker` (`Err(Invalid)`).
fn open_lanes_at(self: &Arc<Self>, first: LaneId, names: &[&str], options: &[StreamOptions])
    -> Result<Vec<Self::Stream>, GpuError>;
// open_lanes keeps its signature: open_lanes_at(LaneId(0), ..) and a LanePool of those queues.
impl CudaStream {
    /// The stream's priority as the driver reports it (cuStreamGetPriority).
    pub fn priority(&self) -> Result<i32, GpuError>;
}
```

- CUDA: the lane array becomes `Arc<[Option<Arc<LaneRing>>]>`, of length `first + n`.
- `High` maps to the greatest priority from `cuCtxGetStreamPriorityRange`; `Normal` maps to 0. Streams are made with
  `cuStreamCreateWithPriority(.., CU_STREAM_NON_BLOCKING, p)`. Add the three entry points to `GpuFns`.
- The fake records the priority and answers `FakeStream::priority()`.

- [ ] **Step 1: Write the failing tests.**
  - `lanes_opened_at_reserved_ids_mark_with_those_ids` (fake).
  - `a_marker_of_a_lane_this_call_did_not_open_is_refused` (fake).
  - `the_priority_option_reaches_the_stream` (fake).
- [ ] **Step 2: Run them and see them fail; implement both backends; run them and see them pass.**
  - `mise exec -- cargo test -p euhedral-gpu`.
  - `mise exec -- cargo clippy -p euhedral-gpu --all-targets --features cuda -- -D warnings`.
- [ ] **Step 3: Commit.** `git commit -m "euhedral-gpu: lanes at reserved pool ids; stream priority"`

### Task 4.2: `LaneSubmit` for the streams, and `open_lane_sources`

**Files:**
- Create: `euhedral-gpu/src/lanes.rs`
- Modify: `src/cuda/stream.rs`, `src/fake/stream.rs`, `src/lib.rs`
- Test: `euhedral-gpu/tests/fake_lanes.rs` (dev-dependency on `euhedral-graph` with `testing`, which the layering allows)

**Interfaces:**
- `impl LaneSubmit for CudaStream` and `impl LaneSubmit for FakeStream`, with `Error = GpuError`:
  - the methods delegate with fully qualified `Stream::` calls;
  - `mark` maps `Err(GpuError::Busy)` → `Ok(None)`;
  - `poll_retired` takes the ticket, then `confirm_retired`, giving `Retired { seq: ticket.seq(), failure }`;
  - `accepts` is true when the marker's lane is opened by this backend's call and its ring id equals `marker.queue`;
  - `Raw` is `*mut c_void`, the `CUstream`, for CUDA, and the queue id as `u64` for the fake.
- Also produces:

```rust
pub struct LaneSpec<X> { pub name: String, pub options: StreamOptions, pub state: X }
pub struct OpenedLanes<S: LaneSubmit, X: LaneState, W> {
    pub pool: Arc<LanePool>, pub intakes: Arc<LaneIntakes<W>>, pub sources: Vec<Arc<LaneSource<S, X, W>>>,
}
/// One pool of this device's lanes, its intakes and a source per lane (lane i is `lanes[i]`). Lanes whose states
/// differ in type share an enum `X`; a pool of several backends is built with LanePoolBuilder, open_lanes_at,
/// LaneIntakes::new and LaneSource::new instead.
pub fn open_lane_sources<G, X, W>(gpu: &Arc<G>, lanes: Vec<LaneSpec<X>>, groups: &[&[LaneId]])
    -> Result<OpenedLanes<G::Stream, X, W>, GpuError>
where G: Gpu, G::Stream: LaneSubmit<Error = GpuError>, X: LaneState, W: LaneWork<G::Stream, X>;
```

- [ ] **Step 1: Write the failing fake tests.**
  - `a_chain_of_kernel_items_across_lanes_runs_in_order`: lanes `a`, `b`, `c` in one group plus `x` on its own. An `add1` item on a group lane delivers a successor with its marker. The successor must run behind the marker on the device. After `gpu.run_until_idle()` and `request`s, the word == chain length.
  - `a_retirement_continuation_reads_back_the_result`: a copy-to-host item with `retire: true`; `retired` pushes a frame carrying the value; `request(1)` pushes it.
  - `a_marker_of_a_foreign_lane_in_the_pool_is_bridged`: a `FakeLanes` lane from `euhedral-graph::testing` in the same `LanePoolBuilder`; the CUDA-side (fake) item sees `cx.bridges(m)`.
  - `a_busy_marker_ring_stalls_then_resumes`: `markers: 2`; 4 items before any completion; then step the device; all 4 are marked in order.
  - `the_high_priority_lane_is_opened_with_high_priority`.
- [ ] **Step 2: Run them and see them fail; implement; run them and see them pass.** `mise exec -- cargo test -p euhedral-gpu --test fake_lanes`.
- [ ] **Step 3: Commit.** `git commit -m "euhedral-gpu: streams are LaneSubmit; open_lane_sources"`

### Task 4.3: The CUDA integration test

**Files:** create `euhedral-gpu/tests/cuda_lanes.rs` (`#![cfg(feature = "cuda")]`). Without a device every test skips, passing, as `tests/cuda.rs` does. Kernels come from the prebuilt `tests/kernels/ops.ptx`, so NVRTC is not needed.

- [ ] **Step 1: Write the tests.**
  - `lane_sources_on_the_device_run_a_chain_and_retire_it`: `open_lane_sources` with `k0..k2` (group) and `kv` (`High`). A 300-item chain alternates group and `kv`, then a copy to pinned host with `retire: true`. A test loop calls `request(u64::MAX)` on every source until the frame arrives (bound 10 s). The value equals the chain length.
  - `the_high_priority_lane_has_the_devices_greatest_priority`: `CudaStream::priority()` equals the greatest value of the context's range; the `Normal` lanes are 0.
  - `work_enqueued_on_the_raw_handle_is_covered_by_the_next_mark`:
    1. an item calls `cuMemsetD8Async` through `Driver::symbol` on `unsafe { cx.submitter().raw() }` and answers `Submitted`;
    2. a dependent item on another lane copies the bytes out behind that marker and retires;
    3. the bytes equal the set value.
  - `a_foreign_lane_in_the_pool_is_bridged_not_awaited`: a `FakeLanes` lane (`euhedral-graph` testing) beside CUDA lanes in one `LanePoolBuilder`; the CUDA item sees `bridges`, and nothing waits on the device for it.
  - `a_poisoned_device_fails_armed_items`: in its own test binary (`tests/cuda_lanes_fault.rs`), as `cuda_fault.rs`. A faulting kernel item with `retire: true`; `retired` gets `Some(Poisoned(..))`, and later items get `failed`.
- [ ] **Step 2: Run them on the GPU.**
  - `mise exec -- cargo test -p euhedral-gpu --features cuda --test cuda_lanes -- --test-threads 1`.
  - `mise exec -- cargo test -p euhedral-gpu --features cuda --test cuda_lanes_fault`.
  - Expected: PASS on the device; skip-pass without one.
- [ ] **Step 3: Commit.** `git commit -m "euhedral-gpu: CUDA integration test for lane sources"`

### Task 4.4: Part 4 docs and the draft PR

**Files:** `euhedral-gpu/README.md`, `CHANGELOG.md`, `GLOSSARY.md`.
- In the README:
  - replace "The intended shape: a lane is a source" (today a sketch) with `open_lane_sources` and the composite-pool recipe;
  - change "Build-time calls … or on a lattice set aside for blocking work" to "… run them in the frame or item that needs them; a worker is a CPU". `api.rs`'s `Module` doc gets the same change.
- The CHANGELOG entry is "GPU lane sources (2026-10-29)". GLOSSARY gets **stream priority**.

- [ ] **Step 1: Write the docs.** Run the full gate:
  - `cargo fmt --all --check`;
  - `cargo clippy --workspace --all-targets -- -D warnings`;
  - `cargo clippy -p euhedral-gpu --all-targets --features cuda -- -D warnings`;
  - `cargo test --workspace`;
  - `cargo test -p euhedral-gpu --features cuda` on the GPU.
- [ ] **Step 2: Commit, push and open the draft PR.**
  - `git push -u origin lane-source-gpu`.
  - `gh pr create --draft --base lane-source-lanes --title "euhedral-gpu: lane sources on streams, stream priority"`.

## Questions for the user

None. Every choice the requirements left open is settled above. Each place the plan departs from the audit's sketch is
the user's own rule, "mirror the Java core", and is listed in "Java counterparts".
