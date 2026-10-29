# Lock-free hot path: audit and fix design (2026-10-28)

**Rule.** The hot path is lock-free. Locks are allowed only for lifecycle work: start, stop (shutdown, drop,
complete, close) and rebalance (attach and detach of sources and receivers, lattice resizing, shard moves).

**Locking** here means `Mutex`, `RwLock`, `Condvar`, `Barrier`, a contended `OnceLock`/`LazyLock` initialisation
reachable after start, a spin-wait or back-off loop that waits on another thread, and a blocking system call or park,
on a path that runs per frame, per pull, per offer, per completion or per request.

**Execution model (from RustEE's author; this revision applies it).**

1. The lattice serialises access to every `LatticeSource` attached to it: only one worker is ever inside a source's
   `request` or `pull`, so whatever those two touch is single-consumer state. The workers drive the system themselves,
   through the events of a frame or of a source; one lattice is enough.
2. A spin inside a queue is fine when it is a CAS loop (as in the upstream Java queues).
3. The hot path is lock-free; locks only for start, stop and rebalance.
4. **`request` is guaranteed to be called; `pull` is not.** All ready work is pushed in `request`, within its demand:
   never more than the demand. Unmet demand accumulates and is served by a later `request`. `pull` hands over at most
   its own demand and never banks demand. So no work may be reachable only through `pull`.
5. **Readiness is selective admission.** A source with nothing ready gives nothing. There are no separate "has work"
   flags or wake-ups; a flag or wake path that only exists to say "there is work" is redundant.
6. **Blocking calls: measure, then place.** If a worker needs something done, it does it; handing work off has no
   point by itself. A blocking call that is short (under about 100 µs), or a few hundred microseconds and rare, runs
   on the worker, on the hot path. A long call, or one that waits for device-wide work, goes to the blocking-work
   facility (or runs at lifecycle time). The CUDA frees in §3.8 are the first application.

**Decisions taken on the first two revisions' questions.** The tick paths count as hot and become lock-free, last.
`PipelineRunner::submit*` returns a lock-free one-shot. A lake's sinks stay attached for the lake's lifetime. CUDA
frees follow principle 6. A `loom` dev-dependency and Miri on a nightly toolchain are approved for the new primitives.
`AsyncIo`'s kernel submission moves into its source (single consumer). `SourceCell` is approved.

**Scope.** `main` at `b33258c`: `euhedral-core`, `euhedral-data-structures`, `euhedral-hashing`,
`euhedral-hardware-utils`, `euhedral-graph`, `euhedral-io`, `euhedral-memory`. `euhedral-gpu` is out of scope (it is
being reworked separately). Test code (`tests.rs`, `testing.rs`, `#[cfg(test)]` modules, `tests/`, `examples/`) is
not counted. Paths are relative to the repository root; line numbers are those of `b33258c`.

**Classes** (where the lock is taken today):

- **HOT**: runs per frame, pull, offer, completion or request in steady state.
- **LIFECYCLE**: start, stop or rebalance only.
- **MIXED**: both; each row says which paths are which. Rows tagged **tick** are taken in steady state by the
  periodic control loop (resource monitor poller, its dispatcher, the lattice's and shards' utilization updates).
  Those never run on a lattice worker and never contend with one, but they are not lifecycle work either, so they are
  counted as MIXED. Decided: they count as hot and are made lock-free, last.

**Routes** (how each HOT or MIXED lock goes away; §2.1 gives every finding's route and reason):

- **SOURCE**: the state is only touched inside one source's `pull`/`request`, so it becomes plain single-consumer
  state in a `SourceCell` (§2.0); lifecycle calls reach it through the source's serialiser, never concurrently.
- **SPLIT**: producers on any thread feed it, and only the source's `pull`/`request` drains it: a lock-free queue
  (or atomics) on the producer side, the queue's `Single` consumer role (or a `SourceCell`) on the consumer side.
- **OWNER**: one thread owns it by construction (the worker that owns its output); it moves onto that thread.
- **SNAPSHOT**: read on hot paths by many threads, written only at rebalance or close: a published snapshot read
  without a lock or a reference count, retired by the writer after a grace period.
- **CONST**: written once at construction and never again, so a plain field.
- **FAN-IN**: written by many producers (completions and arrivals from any worker): atomics, or ownership proved by
  the counters that already order the phases.
- **QUEUE-MC**: multi-consumer by design (work stealing): the existing queue's `Multi` consumer role.
- **TICK**: the periodic control loop; lock-free, last.
- **BLOCKING**: a blocking call, placed by principle 6.

## 1. Summary

| Crate | HOT | MIXED | of which tick | LIFECYCLE |
|---|---:|---:|---:|---:|
| `euhedral-core` | 4 | 16 | 5 | 14 |
| `euhedral-data-structures` | 0 | 0 | 0 | 1 |
| `euhedral-hashing` | 0 | 0 | 0 | 0 |
| `euhedral-hardware-utils` | 0 | 7 | 7 | 4 |
| `euhedral-graph` | 10 | 3 | 0 | 2 |
| `euhedral-io` | 10 | 5 | 0 | 1 |
| `euhedral-memory` | 0 | 1 | 0 | 2 |
| **Total** | **24** | **32** | **12** | **24** |

| Route | Findings | Count |
|---|---|---:|
| SOURCE | C11, C12, C13, G1, G2, G3, I1, I2, I4, I5, I10, I14 | 12 |
| SPLIT | C15, I3, I6, I7, I8, I9, I13, I15 | 8 |
| OWNER | C1, C2, C3, C10 | 4 |
| SNAPSHOT | C5, C6, C7, C8, C9, I11, I12 | 7 |
| CONST | G7 | 1 |
| FAN-IN | C14, G4, G5, G6, G10, G11, G12, G13, G14, G15 | 10 |
| QUEUE-MC | C4 | 1 |
| TICK | C16-C20, H1-H7 | 12 |
| BLOCKING | M1 | 1 |

One finding is one lock (a field or a static) or one blocking call site, with all its acquisitions.

**Source-contract findings** (§2.5): the `pull`/demand audit of every source found 6 places where work is reachable
only through `pull`, a push ignores demand, or a wake path or readiness flag duplicates selective admission (S1-S6, all
SOURCE route, in `euhedral-core`: 2, `euhedral-graph`: 1, `euhedral-io`: 1, `euhedral-memory`: 2). They are not
locks, so they are not in the counts above.

**Changes from the earlier revisions.** D1 (multi-producer queue offers spinning on the tail's `RESIZING` flag while
another producer links a chunk, `queues/core.rs:371-378`) is a CAS loop inside the queue and is fine by point 2; it
moves to the "not locks" list, and the chunk that redesigned it is dropped. Twelve findings now take the SOURCE route,
which needs no new concurrent structure at all, and the `Claim<T>` primitive of the first revision is gone: every
claim it stood for was either a source's single consumer or the queue's own `Multi` consumer. The third revision
applies the decisions and principles 4-6: G7 becomes CONST, M1 BLOCKING, and §2.5 adds the source-contract findings.

**Where the hot locks are, by path:**

- **Every frame a worker runs** passes `LatticeHotSource::accept`: one `RwLock` read, two hook `Mutex`es and the
  `StopWatch` `Mutex` twice (C1, C2, C10), plus the body-cost `Mutex` on sampled frames (C3).
- **Every frame pushed through the lattice** (a source's `request` pushing through the global distributor, socket edge,
  core distributor, core edge, cache) takes two vertex routing `RwLock`s and two edge downstream `RwLock`s, each with
  an `Arc` clone on a line every worker shares (C6, C8).
- **Every cache take and steal** takes a partition `Mutex` (`try_lock`), and every steal attempt a directory
  `RwLock` (C4, C5).
- **Every pull and request of a stock source** takes the ingest `Downstream` `RwLock` and the queue consumer `Mutex`
  (C11, C12, C13); `Completions` and `AsyncIo` take two or three `Mutex`es per poll (G1-G3, I1-I6). All of these sit
  inside the lattice's serialisation already.
- **Every completion** in `euhedral-graph` takes a `Mutex` (`Join`, `Sequencer`, `Latch`) and a stage graph takes
  five `Mutex`es and an `RwLock` per stage (G4-G6, G10-G15).
- **Every Windows I/O operation** takes the slab `Mutex` two to four times and one or two `HashSet` `Mutex`es (I7-I15).

**Not locks, and left as they are** (listed so the next audit does not re-raise them):

- Multi-producer queue offers spinning on `RESIZING` while another producer links a chunk
  (`euhedral-data-structures/src/queues/core.rs:371-378`, formerly D1): a CAS loop inside the queue (point 2).
- The worker's idle parks: `control_plane/fragment.rs:359` (no upstreams and an empty cache), `:430`/`:432` (Idle
  path), `:439`/`:442` (unproductive park or yield), and the `spin_loop` hint at `:454`: the `IdlePolicy` at work,
  only when the worker found no work, waiting on no other thread.
- The lattice's own source claim, `UpstreamInterceptor::try_acquire` (`flow_control/vertex.rs:594`), which is what
  serialises sources (§2.0); `ChunkPool::try_lock` (`queues/chunk.rs:133`) and the multi-consumer queue's
  `try_lock_consumers` (`queues/core.rs:176`): CAS claims inside the queues that fail instead of waiting.
- The seqlock read of the flow meters (`metrics/flow.rs:88-115`): bounded retries, and the writer never waits.
- CAS retry loops that wait on nobody: the atomics in `euhedral-data-structures/src/atomics`, the planner, arena and
  tier moves in `euhedral-memory` (`planner.rs:293,391,756,1163`, `arena.rs:207`, `tiers.rs:610,928,966`).
- `GetQueuedCompletionStatusEx` with a zero timeout (`euhedral-io/src/win/real.rs:384`) and io_uring's submit without
  a wait (`uring.rs:658`): system calls that never block.
- `OnceLock::get` on an initialised cell is one acquire load. Only the initialisation can block; where it is a
  lifecycle step, the row is LIFECYCLE.

## 2. Findings

### 2.0 The source invariant

**What the lattice guarantees.** A source joins a lattice through `LatticeVertex::ingest`
(`flow_control/vertex.rs:369`), which wraps it in one `UpstreamInterceptor` and adds that interceptor to the
`UpstreamRegistry` as an `UpstreamHandle`. A worker's `UpstreamQueue` services a handle only between
`handle.try_acquire()` (`upstream_queue.rs:405`; for the interceptor a `swap(true, Acquire)` on its `lock`) and
`handle.release()` (`:642`, a `store(false, Release)`), and the interceptor reaches the source's `pull` and `request`
only from there (`serviceable` checks the lock, `vertex.rs:547`). A source accepts one downstream, so it has one live
interceptor. Hence:

> **Invariant.** For a source attached to a lattice, at most one thread at a time executes its `pull` or `request`,
> and each executor of them happens-after the previous one (release on `release`, acquire on `try_acquire`).

The glossary already states the contract for every source: "Only one thread at a time is inside a source's `pull` or
`request`". A source driven without a lattice (a test, or `AsyncIo` driven by hand) relies on its caller for the same
guarantee, as today.

**Under the invariant:** `LatticeSource::pull`, `LatticeSource::request`, and everything they run synchronously on the
same source: `QueueSource::drain_requests`, `Completions::emit`, `Front::reap` and the engines' `poll`, the
`IdleWatch` poll, and the source's own push to its receiver. Also the interceptor's own `serviceable` state.

**Not under it:**

- `add_downstream` (attach). It runs before the interceptor is registered, so it happens-before every `pull` and
  `request`; a slot it writes once is safe to read under the invariant without a lock.
- `complete`, `clear`, kill: lifecycle calls from any thread, including from a frame the source's own `pull` is running
  (the direct path executes pulled frames inside `pull`).
- Producers: `QueueIngestSink::offer*`, `Completions::{after, after_marker, at, register}`, `AsyncIo` submissions,
  `Lake::offer`, `LatticeHotSource::accept`, `LatticeHotReceiver::push`. Any thread.
- `is_complete`, `pending`, `demand`, `size`: queries from any thread.
- Everything a source's `request` pushes into (vertices, edges, caches): shared by all sources, so many workers are in
  it at once, each under a different source's serialisation.
- Cache partitions: a cache is a lattice node, not a source; its owner and the stealers take from it concurrently.
- `Drop`: `&mut self`.

**How lifecycle calls reach source state** (`serialize`): a lifecycle call that must touch a source's consumer state
(cancel queued frames, drop the receiver, drop the lane pool) hands the work to whoever holds the source's
serialisation, and never waits:

- `LatticeReceiver` gains `fn serialize(&self, action: Box<dyn FnOnce() + Send>)`. The default runs the action at
  once (a receiver that is not a lattice: the caller is the only driver).
- `UpstreamInterceptor` implements it on its existing claim. The `lock` becomes an owner token (`AtomicU64`, 0 for
  free) and the interceptor gets an `MpscQueue` of pending actions and a `pending` flag. `serialize` acquires the
  claim if it is free, runs the action and any pending ones, and releases. If the calling thread holds the claim (an
  action from a frame that the source's own `pull` runs) or another worker does, the action is queued; the holder runs
  pending actions before it releases and looks again after (the `waiting`-flag protocol of
  `euhedral-io/src/uring.rs:280-376`). Nobody waits, and an action never runs while a `pull` or `request` of the source
  is in progress.
- After the interceptor completes, workers stop servicing it, so `serialize` finds the claim free and runs at once.

**`SourceCell<T>`** (new, `euhedral-core::generics`, next to `LatticeSource`; **approved**) encodes the invariant
once:

```rust
/// State that only a source's `pull` and `request` (and actions run by `serialize`) touch. The lattice serialises
/// those calls (see the source invariant), so it needs no lock and no atomics.
pub struct SourceCell<T> {
    value: UnsafeCell<T>,
    #[cfg(debug_assertions)]
    inside: AtomicBool, // debug only: catches a second thread, or a nested `enter`
}

impl<T> SourceCell<T> {
    pub const fn new(value: T) -> Self;
    /// # Safety
    /// The caller is inside this source's `pull` or `request`, or in an action `serialize` runs, and holds no other
    /// guard of this cell.
    pub unsafe fn enter(&self) -> SourceGuard<'_, T>; // DerefMut to T; debug: asserts and sets `inside`
    pub fn get_mut(&mut self) -> &mut T;              // construction, Drop
    pub fn into_inner(self) -> T;
}
unsafe impl<T: Send> Sync for SourceCell<T> {}
```

The name is meant for reuse by any component whose state lives behind a source: it says what serialises it. The
GPU-side design's RustEE requirements list the same primitive under the same name (their item G-11); it is built
once, here, in chunk 2.

**What the contract adds to the invariant** (principles 4 and 5). Every source keeps its banked demand in its
`SourceCell` (a plain `u64`, saturating, `u64::MAX` meaning unbounded), or in an atomic only where a public query reads
it. `request(d)` adds `d` and pushes ready work until the bank is spent or nothing is ready; `pull(d)` hands over at
most `d` and leaves the bank alone. Work that becomes ready between requests waits for the next `request`; no producer
pushes it, and nothing signals it.

**`DownstreamSlot`** (core) is the receiver of a source: `receiver: SetOnce<Arc<dyn LatticeReceiver>>` and
`state: AtomicU8` (empty, attaching, attached, complete).

- `attach` claims the slot with a CAS (empty → attaching), sets the receiver, and moves to attached; if `complete` raced
  in between, it delivers `on_complete` itself.
- `complete` swaps in complete and calls `on_complete` if the previous state was attached.
- Under the invariant, `receiver()` is a state load and a pointer load: no lock, no `Arc` clone.
- The receiver is never cleared. The source ↔ interceptor cycle that `complete` breaks today by dropping the receiver
  is broken on the other side: the interceptor keeps its source in a `SourceCell<Option<Arc<dyn LatticeSource>>>` and
  drops it, through its own `serialize`, when it finishes.

### 2.1 Route and reason, per HOT and MIXED finding

| ID | Class | Route | Reason |
|---|---|---|---|
| C1 | HOT | OWNER | `LatticeHotSource::accept` is a push by the fragment's one worker, not a `pull`/`request`; the hooks are only called by that worker. |
| C2 | HOT | OWNER | The stop watch lives in C1's hooks: same single caller. |
| C3 | MIXED | OWNER | `BodyCost` is recorded from C1's hooks and read, calibrated and reset by the same worker's `DecisionTree`. |
| C10 | MIXED | OWNER | Single pusher by the hot source's contract ("one thread is expected to do the pushing"); its downstream is attached once at clone build and becomes a `DownstreamSlot`. |
| C11 | MIXED | SOURCE | The ingest receiver is read only in `pull`/`request`/`drain_requests`; written at attach (before registration) and at `complete` (state flag): `DownstreamSlot`. |
| C12 | MIXED | SOURCE | The queue's consumer is used only by `pull` and `drain_requests` (from `request`); `clear`, kill's `discard_queued` and the after-close drain from `publish` go through `serialize`. The queue's `Single` consumer role in a `SourceCell`. |
| C13 | MIXED | SOURCE | The one frame is taken only by `pull`/`request`; `complete` cancels it through `serialize`. |
| G1 | HOT | SOURCE | `Completions::emit` runs only from `pull`/`request`; registrations arrive through the existing MPSC queue (producers); the `add`-side `sweep` goes (the next poll or `Drop` already guarantees it). |
| G2 | MIXED | SOURCE | The lane pool is read only by `emit`; it moves into the `Poll` state and `complete` drops it through `serialize`. |
| G3 | MIXED | SOURCE | Receiver read only by `request`: `DownstreamSlot`. |
| I1 | HOT | SOURCE | `Front::reap` runs only from `AsyncIo::pull`/`request` (its doc already says "only the polling thread touches it"). |
| I2 | MIXED | SOURCE | Receiver read only by `request`: `DownstreamSlot`. |
| I4 | HOT | SOURCE | The ring's submission side is written today by the submitting thread and by the poll; moving all kernel submission into the poll (submitters only offer to the existing lock-free `incoming` queue) makes it single-consumer (decided). |
| I5 | HOT | SOURCE | Reaping the completion queue happens only in the poll; the lock "makes it a fact" (`uring.rs:67`): the invariant already does. |
| I10 | HOT | SOURCE | The inbox is touched only by the poll. |
| I14 | MIXED | SOURCE | Like I4 for an IoRing: build and submit move into the poll; `break_ring` goes through `serialize`. |
| C15 | HOT | SPLIT | Deferred frames are produced by pushers (any thread); retrying them becomes the `pull`/`request` of a retry source the lattice polls, instead of a back-off loop. |
| I3 | MIXED | SPLIT | Rings are appended by submitting threads (first submit) and walked only by the poll: an append-only array of `SetOnce` slots and a length. |
| I6 | MIXED | SPLIT | An op is inserted by its ring's owner thread and reaped only by the poll: per-slot generations, and freed slot indexes go back over an SPSC queue (producer: the poll; consumer: the owner thread). |
| I7 | HOT | SPLIT | Ops begin on any submitting thread and end in the poll: per-slot state atomics; freed indexes go back over an SPMC queue (producer: the poll; consumers: submitters). |
| I8 | HOT | SPLIT | Failed calls are produced by submitters and drained by the poll: `MpscQueue`, `Single` consumer used under the invariant. |
| I9 | HOT | SPLIT | Pool-job results are produced by pool threads and drained by the poll: `MpscQueue`. |
| I13 | MIXED | SPLIT | As I3, for Windows rings. |
| I15 | HOT | SPLIT | Entries are produced by `launch` on submitting threads and drained by the poll: `MpscQueue`. |
| C5 | MIXED | SNAPSHOT | The directory is read by every stealing worker and written at cache build and close. |
| C6 | MIXED | SNAPSHOT | The routing generation is read by every worker whose source `request` pushes (many sources at once), written at rebalance and close. |
| C7 | MIXED | SNAPSHOT | `Book` is locked on a worker only because the last pusher retires a generation; with the writer retiring it after the grace period, `Book` is lifecycle-only. |
| C8 | MIXED | SNAPSHOT | Edge downstreams are read by all pushing workers, written at attach and close. |
| C9 | MIXED | SNAPSHOT | Edge parents are read by every worker that climbs edges, written at attach and remap. |
| G7 | MIXED | CONST | The lake's sinks stay attached for its lifetime (decided), so its generation is built once in `Lake::new` and never replaced: a plain field, read by `Lake::offer` from any thread with no lock and no `Arc` clone. |
| I11 | HOT | SNAPSHOT | Handle routes are read per submit on any thread; a handle's first use is its attach. |
| I12 | HOT | SNAPSHOT | As I11. |
| C14 | HOT | FAN-IN | The outcome is set by a chain's last finalizer on any worker and read by the submitter: a one-shot cell (decided). |
| G4 | HOT | FAN-IN | `Join` parts arrive from any worker; the fix is the re-armable fan-in cell both plans need (§3.11). |
| G5 | HOT | FAN-IN | `Latch` is set from any worker, waited on by the main thread. |
| G6 | HOT | FAN-IN | `Sequencer` slots are registered and completed from any thread; release is already serialised by its `wip` counter. |
| G10 | HOT | FAN-IN | The quantum is read by stage frames on any worker; the run's `holds` count proves `conclude` owns it. |
| G11 | HOT | FAN-IN | First outcome from any stage frame: set-once. |
| G12 | HOT | FAN-IN | Inputs merge from every predecessor's frame: per-lane atomic maximum; `pending` reaching zero proves the stage's frame owns them. |
| G13 | HOT | FAN-IN | Outputs published by one stage frame, subscribed by other quanta's frames: a set-once value plus a closable waiter stack. |
| G14 | HOT | FAN-IN | The frontier is covered by many frames: per-lane atomic maximum plus a take-all stack of host completions. |
| G15 | HOT | FAN-IN | Pins pushed at registration, taken once per stage and at quiesce: take-all stacks. |
| C4 | MIXED | QUEUE-MC | The owner and stealing workers consume concurrently by design; the queue's `Multi` consumer role (a CAS claim inside the queue) replaces the `Mutex`. |
| C16-C20, H1-H7 | MIXED | TICK | The periodic control loop; decided hot, made lock-free last. |
| M1 | MIXED | BLOCKING | By measurement (principle 6): `cuMemFreeAsync` (0.2 µs), VMM unmap and destroy (20-52 µs, 65 µs busy) and sub-allocation frees (0) stay on the worker; `cuMemFree` and `cuMemFreeHost` wait for all queued device work (about 350 ms when busy) and never run on a worker. |


### 2.2 HOT (where the lock is taken today)

| ID | Site | Protects | Hot paths |
|---|---|---|---|
| C1 | `euhedral-core/src/flow_control/hot_source.rs:21-22` (`:56`, `:60`) | `before_each` / `after_each` hooks (`Mutex<Box<dyn FnMut>>`) | Worker: `run_cycle` → `timed` → `output.accept` (`fragment.rs:467`) → `LatticeHotSource::accept` → both hooks, for every frame it runs (cache greedy/poll, direct pull, steal). |
| C2 | `euhedral-core/src/control_plane/fragment.rs:748-753` | the sampling `StopWatch` (`Arc<Mutex<StopWatch>>`) | Same path as C1: `start` in `before_each`, `stop` in `after_each`, every frame. |
| C14 | `euhedral-core/src/ingest/pipeline_runner.rs:39-43` | the outcome hand-off: `mpsc::sync_channel(1)` per submitted chain | Per request: `PipelineRunner::submit*` makes the channel; the chain's last finalizer `send`s it on a worker, which takes the channel waker's internal `Mutex` and wakes the waiting receiver. |
| C15 | `euhedral-core/src/flow_control/hot_receiver.rs:46` | back-off (`spin_wait::await_while`: spin, yield, then 10 µs parks) until the `PushHook` takes the frame | Per push: `LatticeHotReceiver::push` waits on whatever frees the hook. Public API; nothing in the workspace uses it. |
| G1 | `euhedral-graph/src/completions.rs:70` (`:310`, `:315`; `try_lock` at `:246`) | `Poll`: the consumer of incoming registrations and the waiting list | Per pull and request: worker → `UpstreamInterceptor::pull`/`request` → `Completions::pull`/`request` → `emit` (blocking `lock`); the lock is held while `consumer` runs the emitted frames on the direct path. `sweep` per `add` racing `complete`. |
| G4 | `euhedral-graph/src/join.rs:10` (`:56`) | `JoinState` (values, ended flags, remaining, continuation) | Per completion: `Join::arrive` / `abandon` from each branch's frame. |
| G5 | `euhedral-graph/src/latch.rs:27` (`:79`, `:97`, `:135`) | the value, abandoned flag and parked waiter | Per completion: `LatchSetter::set` / its `Drop` on a worker. The waiter's `thread::park` (`:147`) is on the main thread, by design. |
| G6 | `euhedral-graph/src/sequencer.rs:24` (`:64`, `:75`, `:88`, `:110`) | `Order`: head sequence and slot states | Per admission (`register`), per completion (`Slot::complete`/`abandon`/`Drop` → `finish`), and once per released slot in the `release` drain loop. |
| G10 | `euhedral-graph/src/stage_graph.rs:663` (`:869`, `:1059`, `:1092`) | the quantum (`RwLock<Option<Q>>`) | Per stage: `StageFrame::execute` → `Run::run_stage` (read); `quiesce` (read); `conclude` (write) per quantum. |
| G11 | `euhedral-graph/src/stage_graph.rs:674` (`:771`, `:1096`) | the first outcome (`Mutex<Option<QuantumOutcome>>`) | Per failing or killed stage (`note`, `fail`); `conclude` per quantum. |
| G12 | `euhedral-graph/src/stage_graph.rs:686` (`:820`, `:864`, `:910`) | each stage's merged input markers | Per arrival (`arrive`), per stage run (`run_stage`), per pass-through submit (`submitted`). |
| G13 | `euhedral-graph/src/stage_graph.rs:676` (`:981`, `:1043`, `:1110`) | `Externals`: published outputs, waiters, swept flag | Per stage submit (`notify_externals`), per external subscription (`subscribe`), per quantum (`quiesce`). |
| G14 | `euhedral-graph/src/stage_graph.rs:677` (`:913`, `:1047`, `:1072`, `:1114`, `:1126`) | `BoundarySet`: frontier markers and host completions | Per stage submit (`submitted`), per pinned downstream (`cover`), per quantum (`quiesce`, `arm_boundary`). |
| G15 | `euhedral-graph/src/stage_graph.rs:679` (`:964`, `:1002`, `:1053`) | pins of upstream quanta, by stage | Per external edge (`register_external`), per stage submit (`release_pins`), per quantum (`quiesce`). |
| I1 | `euhedral-io/src/lib.rs:506` (`:541`) | continuations held back by a `stop` | Per pull and request: `AsyncIo::pull`/`request` → `Front::reap`; held across the engine poll and `consumer`. |
| I4 | `euhedral-io/src/uring.rs:58` (`try_lock` at `:355`) | a ring's submission side (`Queue`: incoming consumer, backlog) | Per submit (`Ring::queue` → `flush_waiting`) and per poll (before and after reaping). Non-blocking already. |
| I5 | `euhedral-io/src/uring.rs:68` (`try_lock` at `:613`) | reaping a ring's completion queue | Per poll. Non-blocking already. |
| I7 | `euhedral-io/src/win/mod.rs:363` (`:451`, `:491`, `:503`, `:508`, `:517`, `:560`, `:592`, `:617`, `:776`) | the operation slab (`Table`: slots, generations, free list, `by_addr` map) | Per operation: `submit` → `begin` → `launch` (`set_state`); per completion: `take_packet`/`take_token` → `release`; per resubmitted piece: `rebegin`. |
| I8 | `euhedral-io/src/win/mod.rs:365` (`:604`, `:635`, `:800`, `:1298`) | operations whose call failed at once | Per failed call; per poll (`poll_failed`). |
| I9 | `euhedral-io/src/win/mod.rs:234` (`:258`, `:1416`) | results of thread-pool open and rename jobs | Per open/rename completion (`Delivery::drop` on a pool thread); per poll. |
| I10 | `euhedral-io/src/win/mod.rs:371` (`:1262`, `:1287`) | packets taken from the port but not handled yet | Per poll (`poll_port`). |
| I11 | `euhedral-io/src/win/mod.rs:373` (`:835`, `:839`, `:856`) | handles joined to the completion port (`HashSet`) | Per send, receive, accept, connect (`join_port`, which also calls `associate` every time) and per file read/write (`file_route`). |
| I12 | `euhedral-io/src/win/mod.rs:378` (`:858`, `:861`) | handles a ring serves (`HashSet`) | Per file read/write (`file_route`). |
| I15 | `euhedral-io/src/win/mod.rs:324` (`:696`, `:702`, `:730`, `:736`, `:809`, `:952`) | a ring's backlog of entries | Per ring submit (`launch`) and per flush (`flush_ring`, three acquisitions per entry). |

### 2.3 MIXED

| ID | Site | Protects | Hot paths | Lifecycle paths |
|---|---|---|---|---|
| C3 | `euhedral-core/src/control_plane/decision_tree.rs:49` | `BodyCost` window: samples, threshold, confirmations | `record` (`:85`) per sampled frame from `after_each` (C1). | `calibrate` (`:76`) at worker start; `reset` (`:108`) on a lattice reset. All on the worker thread. |
| C4 | `euhedral-core/src/control_plane/cache.rs:130` | each partition's single consumer (`Mutex<Consumer<_, Single>>`) | `try_hold` (`:436`, `:673`, `:718`): `greedy_while` (held while the worker runs the frames), `poll_while`/`take_one`, `work_steal`, `work_steal_polled`; every cycle with cached work and every steal. | `hold` (blocking): `first_touch` (`:307`), `reset_for_next_trial` (`:579`). |
| C5 | `euhedral-core/src/control_plane/cache.rs:33` | the lattice's cache directory: one `RwLock<Weak<ControlPlaneCache>>` per CPU | `cache_at` (`:71`, read + `Weak::upgrade`) per steal attempt (`work_steal`, `work_steal_polled`), every unproductive cycle. | `claim` (`:78`) at cache construction; `withdraw` (`:85`) at cache close. |
| C6 | `euhedral-core/src/flow_control/vertex.rs:138` | the current routing generation (`RwLock<Arc<RoutingState>>`) | `routing_state` (`:202`, read + `Arc` clone) per `try_push` (`:350`): every frame a source pushes, at both the global distributor and the shard's core distributor; `on_error`. | `set_downstream_mapping` (`:272`) on rebalance; `close_after_routes_retire` (`:338`) on close. |
| C7 | `euhedral-core/src/flow_control/vertex.rs:139` | `Book`: downstream reference counts per generation, retirements | `release` (`:286`) runs from `RoutingState::drop` (`:102-108`) on whichever thread drops the last `Arc` of a retired generation: after a remap or close, a worker mid-push (the comment at `control_plane/lattice.rs:576` relies on it). Once per generation, on a hot thread. | `set_downstream_mapping` (`:258`), `close_after_routes_retire` (`:325`), `defer_child_retirement` (`:436`). |
| C8 | `euhedral-core/src/flow_control/edge.rs:63` | the edge's downstream (`RwLock<Option<Downstream>>`) | `downstream` (`:233`, read + `Arc` clone) per `push_downstream` and `error_downstream`: every pushed frame at the socket edge and the core edge. | `add_downstream` (`:173`), `add_downstream_node` (`:197`) at attach; `close_edge` (`:242`). |
| C9 | `euhedral-core/src/flow_control/edge.rs:62` | the parent link (`RwLock<Option<Weak<dyn LatticeNode>>>`) | `parent` (`:109`, read + `Weak::upgrade`) per hop of `LatticeEdge::request` and `pull`: per pull of the public `ControlPlaneCache::pull` (`cache.rs:350`). The stock worker loop does not climb edges (it uses the registry queue directly). | `set_parent` (`:125`) at attach and on every remap; `defer_retirement` (`:252`). |
| C10 | `euhedral-core/src/flow_control/hot_source.rs:23` | the hot source's downstream and completion | `accept` (`:49`, read + `Arc` clone) per frame (C1 path); `is_complete` (`:109`). | `add_downstream` (`:70`), `complete` (`:99`). |
| C11 | `euhedral-core/src/ingest/downstream.rs:10` | an ingest source's receiver | `receiver` (`:46`, read + `Arc` clone) per pull and request: `QueueSource::pull` (`queue.rs:270`), `request` (`:295`), `drain_requests` (`:219`); `SingleUseSource::request`. | `attach` (`:31`), `complete` (`:58`). |
| C12 | `euhedral-core/src/ingest/queue.rs:47` | the ingest queue's consumer | `try_hold` (`:183`) per pull (`:273`) and per request drain (`:215`); `discard_queued` (`:194`) after a kill. | `clear` (`:207`, blocking). |
| C13 | `euhedral-core/src/ingest/single_use.rs:9` | the one frame | `take` (`:23`) per pull and request until taken. | `complete` (cancels the frame). |
| C16 | `euhedral-core/src/control_plane/lattice.rs:150` | the close gate (`RwLock<()>`) | **tick**: `update` (`:304`) per utilization, and each socket's update task (`:344`). | `close` (`:562`, write). |
| C17 | `euhedral-core/src/control_plane/lattice.rs:144` | serialising updates (`Mutex<()>`) | **tick**: `update` (`:308`) per utilization. | Rebalances, which run inside `update`. |
| C18 | `euhedral-core/src/control_plane/lattice.rs:142` | `State`: primed, rebalancing, version, handles, active sockets | **tick**: `update` (`:312`) reads version and active sockets per utilization. | `ready` (`:278`), `handle_system_topology_change` (`:361`, `:403`), `finish_rebalance` (`:438`), `is_rebalancing` (`:531`). |
| C19 | `euhedral-core/src/control_plane/shard.rs:77` | the shard's `State` (clones, active cores, distributor, pool) | **tick**: `update` (`:207`) per socket per utilization, which also allocates a `Vec` of clone `Arc`s. | Start, rebalance, drain, shut-down, close, reset, and the queries `is_started`, `is_drained`, `active_cores`, `is_rebalancing`. |
| C20 | `euhedral-core/src/internal/task_pool.rs:13`, `:20` | the task channel's sender and the shared `Receiver` (`mpsc` + two `Mutex`es) | **tick**: `execute_or_run` per socket per utilization (`lattice.rs:343`); the pool threads lock the receiver (`:69`) per task. | Shard drain and shut-down tasks (`shard.rs:293`, `:430`, `:537`), `shutdown`. |
| G2 | `euhedral-graph/src/completions.rs:63` | the lane pool (`Mutex<Option<Arc<LanePool>>>`) | `emit` (`:322`, lock + `Arc` clone) per pull and request. | `complete` (`:472`) drops it. |
| G3 | `euhedral-graph/src/completions.rs:73` | the downstream receiver | `request` (`:422`, lock + `Arc` clone) per request. | `add_downstream` (`:395`), `complete` (`:473`). |
| G7 | `euhedral-graph/src/lake.rs:165` | the attached generation of sinks (`RwLock<Option<Arc<Generation>>>`) | `attach` read (`:317`, + `Arc` clone) per `offer` and per `admit`. | `attach` write (`:326`) at the start of a busy period, `discard` (`:399`), `detach` (`:416`), `complete` (`:441`). |
| I2 | `euhedral-io/src/lib.rs:509` | the downstream receiver | `request` (`:851`, lock + `Arc` clone) per request. | `add_downstream` (`:829`), `complete` (`:874`). |
| I3 | `euhedral-io/src/uring.rs:39` | the rings of all submitting threads | `poll` (`:610`): lock, clone the `Vec<Arc<Ring>>` (allocation plus one `Arc` clone per ring) per poll. | `ring` (`:513`): a thread's first submit creates its ring. |
| I6 | `euhedral-io/src/uring.rs:69` | the ring's operations in flight (`Ops` slab) | `submit` (`:594`) per operation on the submitting thread; `reap_one` (`:377`) per completion on the polling worker: the two contend. | `Drop for Ring` (`:420`). |
| I13 | `euhedral-io/src/win/mod.rs:379` | the rings of all submitting threads | `poll` (`:1432`): lock and clone per poll. | `ring` (`:668`): a thread's first ring. |
| I14 | `euhedral-io/src/win/mod.rs:321` | a ring's build-and-submit side (`RingInner`) | `try_lock` per flush (`:691`) and per pop (`:750`). | `break_ring` (`:808`, blocking) when the ring fails for good. |
| H1 | `euhedral-hardware-utils/src/monitor/mod.rs:159-160` | monitor `Lifecycle` (`Mutex` + `Condvar`) | **tick**: the poller's `wait_for_tick` (`:652`, `wait_timeout`) and `begin_evaluation` per tick. | `start`, `stop`, `close_within` (`:483`, `:495`), `try_finish_close`. |
| H2 | `euhedral-hardware-utils/src/monitor/mod.rs:161` | evaluation state | **tick**: `run_evaluation` (`:250`) per tick. | Reset on start. |
| H3 | `euhedral-hardware-utils/src/monitor/mod.rs:162` | the last `HardwareUtilization` | **tick**: write per evaluation (`:298`). | `utilization` (`:523`, `:532`): read at lattice start. |
| H4 | `euhedral-hardware-utils/src/monitor/mod.rs:66` | the process-wide snapshot provider (`static OnceLock<Option<Mutex<..>>>`) | **tick**: `system_snapshot` (`:71`) per tick (it is also public). | First use initialises the `OnceLock`. |
| H5 | `euhedral-hardware-utils/src/monitor/dispatcher.rs:29-30` | the listener list and the latest value (`Mutex` + `Condvar`) | **tick**: `offer` (`:99`) per utilization; `dispatch_loop` (`:139`, `wait`) per delivery. | `add_listener`, `remove_listener` (attach), `begin_close`, `Drop`. |
| H6 | `euhedral-hardware-utils/src/topology/mapper.rs:51` | per-socket topology versions | **tick**: `update` (`:125`) via `update_from` from `lattice.update` per utilization. | — |
| H7 | `euhedral-hardware-utils/src/topology/mapper.rs:52` | the effective topology (`RwLock<Arc<..>>`) | **tick**: `effective_topology` read + `Arc` clone (`:95`) and write (`:184`) per utilization. | Lattice construction reads it. |
| M1 | `euhedral-memory/src/cuda/backends.rs:49`, `:63`, `:444`, `:478` (and `free_vmm` `:319`) | blocking driver calls: `cuMemAlloc`, `cuMemHostAlloc`, `cuMemFree`, `cuMemFreeHost` (the frees synchronise the device) | Wherever the application runs `Reaper::reap` (`backend.rs:140`): its doc says "on the owner's lane", which is a lattice worker, per frame that reaps. | Arena and pool construction and teardown; the module doc (`backend.rs:5-7`) says a blocking free never runs on a worker. The two docs disagree; §3.8 settles it. |

### 2.4 LIFECYCLE

| ID | Site | Protects | Paths |
|---|---|---|---|
| C21 | `euhedral-core/src/flow_control/registry.rs:70` | workers and the handles each worker receives | `register_with` (`:121`) at worker start; `unregister` (`:232`) at worker exit; `add_handle` (`:175`) per attached source. |
| C22 | `euhedral-core/src/control_plane/fragment.rs:93` | the worker task's interrupt | `start` (`:545`), `interrupt_task` (`:272`) on close and `dump_locks`. |
| C23 | `euhedral-core/src/control_plane/fragment.rs:96` | serialising start, reset, close | `start` (`:520`), `reset` (`:637`), `close` (`:689`). |
| C24 | `euhedral-core/src/control_plane/fragment.rs:103` | the worker's `Thread` for unparking | `run_worker` (`:301`), `WorkerExit::drop` (`:911`), `unpark_worker` (`:278`) from stop and reset. |
| C25 | `euhedral-core/src/control_plane/fragment.rs:636-654` (`park_timeout` at `:645`) | waiting for the worker to finish a reset (`park_timeout` loop) | `reset`, from `ControlPlaneLattice::clear`. |
| C26 | `euhedral-core/src/control_plane/lattice.rs:146` | whether the lattice started | `start` (`:248`), which every `add_upstream` calls; `clear` (`:496`). |
| C27 | `euhedral-core/src/control_plane/lattice.rs:566-591`, `:578`, `:623` | close: the shard-closing task (`Mutex<Option<FnOnce>>`) and an `mpsc` channel waited with `recv_timeout` | `close`. |
| C28 | `euhedral-core/src/control_plane/countdown.rs:9` | the rebalance continuation | `tick` (`:31`) as each retired shard shuts down. |
| C29 | `euhedral-core/src/config/liveness.rs:76` (and `thread::sleep` at `:55`) | a worker slot's `Thread`; polling for termination | `enter`/`exit` at worker start and exit; `await_termination` on clone close. |
| C30 | `euhedral-core/src/metrics/registry.rs:26` | registered meter sources | `register`/`remove` at worker and cache build and close; `snapshot` and `len` are observer queries, never taken by a worker. |
| C31 | `euhedral-core/src/flow_control/edge.rs:164` | `add_upstream_handle` spin-waits (`await_while`) while the edge drains | Attaching a source during a rebalance. Was reached from `Lake::offer` when a busy period started; with lake sinks attached for the lake's lifetime it is reached only from attach calls. |
| C32 | `euhedral-core/src/control_plane/shard.rs:445`, `:539`, `:605` | waiting for clones to drain; waiting for a rebalance before a reset | Rebalance, shut-down, reset (task-pool threads and the resetting thread). |
| C33 | `euhedral-core/src/control_plane/lattice.rs:264`, `:508` | waiting for readiness; waiting for a rebalance before a reset | `start`, `clear`. |
| C34 | `euhedral-core/src/control_plane/cache.rs:142`, `ingest/queue.rs:54`, `flow_control/vertex.rs:480` | set-once cells (`OnceLock`): SMT sibling, ingest owner, interceptor upstream | Set once at build or attach by one setter; `get` on hot paths is a lock-free load. |
| D2 | `euhedral-data-structures/src/queues/core.rs:192-202` | `lock_consumers`: spins until the multi-consumer lock is free | Only `Consumer<_, Multi>::clear` (`queue.rs:434`). |
| H8 | `euhedral-hardware-utils/src/affinity/executor.rs:170-171` | a pinned executor's control state (`Mutex` + `Condvar`) | `start`, `submit*` (worker start, clone build), `shutdown*`, `close`, `await_termination`, handle clone and drop. |
| H9 | `euhedral-hardware-utils/src/affinity/executor.rs:245` | the per-CPU executor registry | `get_or_create`, `get`, `close_all`, unregister. |
| H10 | `euhedral-hardware-utils/src/monitor/mod.rs:319` | the poller's `JoinHandle` | `start`, `close_within`. |
| H11 | `affinity/executor.rs:286`, `affinity/mod.rs:76`, `topology/mod.rs:242`, `monitor/mod.rs:55` (all in `euhedral-hardware-utils/src`) | process-wide `OnceLock` statics: executor registry, affinity controller, system topology, monotonic epoch | First use, which lattice construction and worker start reach before any frame. |
| G8 | `euhedral-graph/src/lake.rs:169` | generations being retired | `retire` (`:294`) on detach and complete; `is_attached` (`:286`) is a query. Goes with S6: one generation per lake, retired at `complete`. |
| G9 | `euhedral-graph/src/lake.rs:507` | the idle watch's downstream | `add_downstream` (`:532`), `complete` (`:558`). Goes with S6: no idle watch. |
| I16 | `euhedral-io/src/win/real.rs:274`, `:278` | `OnceLock` statics: the IoRing API and the socket extension functions | Loaded when an engine opens. |
| M2 | `euhedral-memory/src/cuda/copier.rs:41`, `cuda/backends.rs:175` | `cuStreamSynchronize` | `Drop` of the copier and of the stream pool. |
| M3 | `euhedral-memory/src/planner.rs:1206`, `tiers.rs:1063` | set-once downstream (`OnceLock`) | Set at attach; `get` per request is a lock-free load. |

`cuCtxSynchronize` and `cuEventSynchronize` are loaded (`euhedral-memory/src/cuda/mod.rs:145`, `:177`) but the
library calls neither.


### 2.5 Source-contract findings (`pull`/demand audit)

Every `LatticeSource` in scope, checked against principles 4 and 5:

| Source | Work only via `pull`? | `request` pushes more than demand? | `pull` banks demand? | Redundant readiness or wake path? |
|---|---|---|---|---|
| `QueueSource` (`QueueIngestSink`, `ArrayIngestSink`, `PipelineRunner`, lake sinks) | no | no | no, but `pull` also pushes banked request demand | yes (S4) |
| `SingleUseSource` | no | no (one frame, demand ≥ 1) | no | no |
| `LatticeHotSource` | no | yes (S3) | n/a | no |
| `Completions` | no | no | no | yes (S5) |
| `AsyncIo` (io_uring, Windows) | no | no (`room` checked per completion) | no | yes (S6) |
| `IdleWatch` (lake) | n/a (pushes nothing) | n/a | n/a | the whole source (S6) |
| `GiveBacks` (memory planner) | no | no, but producers push outside `request` | no | yes (S1) |
| `Moves` (memory tiers) | yes (S2) | no, but producers push outside `request` | no | yes (S2) |

| ID | Site | What | Fix (route SOURCE) |
|---|---|---|---|
| S1 | `euhedral-memory/src/planner.rs:733` → `GiveBacks::wake` (`:1266`) | A producer (the thread that orders a give-back) pushes give-back frames to the downstream itself, outside `request`, running `emit` concurrently with the source's own `request`/`pull`. That is a wake path, and it breaks the single-consumer invariant. | Delete `wake` and its call. `request` adds to the banked demand and pushes ready orders within it; `pull` hands over its own demand. The order queue's consumer becomes the `Single` role in a `SourceCell`, and banked demand a plain field in it. |
| S2 | `euhedral-memory/src/tiers.rs:914` → `Moves::wake` (`:1119`); `pump` only in `Moves::emit` (`:1102`) | Same producer-side wake when a move finishes. Also, `Mover::pump`, which advances copies (retires a hop, starts the next, finishes moves), runs only on the `pull` path: a lattice that only requests never advances a move. | Delete `wake`. `request` runs `pump`, then pushes finished moves within the banked demand; `pull` pumps and hands over at most its own demand. The done-queue consumer goes into a `SourceCell`. |
| S3 | `euhedral-core/src/flow_control/hot_source.rs:94` (`request` ignores demand), `:48-58` (`accept` pushes regardless) | Pushes without demand. Harmless for the fragment's output, whose downstream (the executor's terminal) asks for unbounded demand once, but wrong for the public type. | The hot source banks demand. While the bank is unbounded (the stock case), `accept` pushes at once. Otherwise `accept` pushes within the bank and parks frames beyond it in an SPSC queue (producer: the one pusher; consumer: `request` under the invariant); `request` pushes them within its demand. |
| S4 | `euhedral-core/src/ingest/queue.rs:50` (`request_pending`), `:214-253` (`drain_requests` retry loop), `:285` (`pull` calls `drain_requests`), `:108`, `:166` (producers trigger `drain_finished`) | `request_pending` and the retry loop exist so a second thread's request is not lost while another holds the consumer: impossible under the invariant. `pull` delivers banked request demand. Producers run consumer-side work after close. | `request` alone pushes banked demand; `pull` hands over its own demand only; the flag and loop go. The after-close cancellation goes through `serialize` (§2.0). |
| S5 | `euhedral-graph/src/completions.rs:194` (`add` → `sweep`) | A producer polls the source to cancel registrations that raced `complete`: a wake path into consumer state. | Delete; the next `request`/`pull`, or `Drop`, cancels them (part of G1's fix). |
| S6 | `euhedral-io/src/uring.rs:66` (`waiting`), `:280-376` (`queue`/`flush_waiting` hand-off), `euhedral-io/src/lib.rs:536` (`pending() == 0` gate); `euhedral-graph/src/lake.rs:502-569` (`IdleWatch`), `:75`, `:107`, `:117` (`idle_since`, `idle_for`) | The io_uring `waiting` flag and its fence protocol only tell the `sq` holder that a submitter queued work, and the `pending` gate in `Front::reap` is a "has work" check before the poll. The lake's idle watch is a source that pushes nothing; it exists to detach idle sinks. | io: with kernel submission moved into the poll (I4), `waiting` and `flush_waiting` go; the poll drains `incoming` itself. The `pending` gate goes (the poll finds nothing on its own); `pending()` stays as the public count that `complete` documents. The same applies to the Windows ring's `inner` hand-off (I14). Lake: sinks stay attached, so `IdleWatch`, `detach`, `idle_since`/`idle_for`, the per-busy-period re-attach and the `retiring` list (G8) go. |

The `euhedral-io` and lake parts of S6 count once, under `euhedral-io` (lake removal is G7's chunk). Compliant and
unchanged: `SingleUseSource`, `ArrayIngestSink` (a `QueueSource` that ends when exhausted), `AsyncIo`'s demand
handling (`request` banks and reaps within it; Windows `poll` checks `room` before every completion).

## 3. Replacement designs

### 3.0 Building blocks

In the preferred order: single ownership (including the source invariant, which is single ownership the lattice
already provides); the queues of `euhedral-data-structures` (`Queue<T, P, C>` with `Single`/`Multi` roles,
`PartitionedQueue`); atomics and published snapshots written only at lifecycle time. New pieces, each because nothing
in the workspace offers it:

- **In `euhedral-core`** (§2.0): `SourceCell<T>`, `DownstreamSlot`, and `LatticeReceiver::serialize` with its
  implementation on `UpstreamInterceptor` (owner-token claim plus a pending-action `MpscQueue`).
- **In `euhedral-data-structures`** (a new `sync` module, in the queues' style of role handles and drop safety):
  1. **`SetOnce<T>`**: an `AtomicPtr<T>` installed by CAS from a `Box`; `get` is an acquire load returning `&T`; `set`
     hands the value back if already set. No blocking init path (unlike `OnceLock::set`, which can block a racing
     `get_or_init`).
  2. **`Published<T>`**: an atomically replaceable `Option<Arc<T>>` with lock-free, count-free reads and
     lifecycle-only writes, reclaimed by epochs. The domain is the lattice's: `UpstreamRegistry` already keeps one
     queue per thread and now keeps one padded epoch slot per thread too.
     - A **worker** announces a quiescent point once per cycle (`run_cycle` start: one load of the global epoch and
       one store to its own slot) and marks itself idle before it parks. Inside the cycle, `load_in_cycle()` is a plain
       acquire load of the pointer: no read-modify-write and no reference count per push. That is what C5, C6, C8 and
       C9 need: their readers are workers (a source's `request` and the steals run only on workers; a thread that
       never registers has an empty upstream queue, so it never services a source).
     - **Any other thread** (`Lake::offer` and submissions from outside the lattice) uses `load()`, which marks its
       slot active at the current epoch for the duration of the guard (one store and one fence, its own line).
     - `store(new)` swaps the pointer, advances the epoch, and returns a `Retired<T>`. `wait(deadline)` (lifecycle
       threads; spins) drops the old value once every slot is idle or past the new epoch. `defer` keeps it in the
       cell's single pending slot until a later `store`, `try_reclaim` or `Drop` finds it reclaimable. A store with a
       still-pending retirement waits (lifecycle) or, as `try_store`, hands the value back. A debug assertion catches a
       writer that waits from inside its own read.
  3. **`OneShot<T>`**: `state: AtomicU8` (empty, waiting, set, abandoned), the value, and a `SetOnce<Thread>` waiter.
     The setter writes, swaps the state, and unparks a waiter it finds (a wake, not a wait). `try_recv` and a parking
     `recv` for a thread outside the lattice.
  4. **`TakeAllStack<T>`**: CAS push; the consumer takes the whole list with one swap, optionally closing it (later
     pushes hand the value back). Whole-list swaps avoid ABA. For the stage graph's per-stage lists, where a queue's
     chunk per list would be too heavy.

### 3.1 SOURCE route

**Ingest (C11, C12, C13).**

- `Downstream` becomes `DownstreamSlot` plus the `demand` atomic. `demand` stays atomic because it is also a public
  query; `request_pending` and the `try_hold` retry loop in `drain_requests` go, because under the invariant no
  second thread can hold the consumer (S4).
- Demand (S4): `request` alone pushes, within the banked demand; `pull` hands over at most its own demand and no
  longer calls `drain_requests`.
- `QueueSource.consumer: SourceCell<PartitionedConsumer<_, Single>>`, entered by `pull` and `request`. The
  three paths that touch it from outside go through `self.downstream.serialize(..)`: `clear` (lifecycle), kill's
  `discard_queued` (lifecycle), and the drain that `publish` triggers after the sink closed (a producer, once). An
  unattached sink (no lattice) runs them at once.
- `SingleUseSource.frame: SourceCell<Option<Box<dyn Frame>>>`. `pull`/`request` test `stop` and take the frame;
  `complete` cancels it in a serialised action, then completes the slot. Exactly one of them finishes the frame.

**`Completions` (G1, G2, G3).** `poll: SourceCell<Poll>` with the lane pool moved inside it. `emit` enters the
cell; registrations still arrive through the `incoming` MPSC queue. `complete` sets the flag, offers the `Closed`
marker, completes the `DownstreamSlot`, and serialises an action that drops the lanes and cancels late registrations.
The `sweep` that `add` runs when it races `complete` goes (S5): the next poll, or `Drop` with `&mut self`, cancels
them, as the code's own comment already guarantees.

**`AsyncIo` (I1, I2, I4, I5, I10, I14).**

- `Front.held: SourceCell<VecDeque<..>>`; `Front.downstream: DownstreamSlot`.
- io_uring: `cq` loses its lock (reaping is only in the poll). The submission side becomes single-consumer: a
  submitting thread only offers its entry to the ring's existing lock-free `incoming` queue, and the poll (under the
  invariant) drains `incoming` into the kernel's submission queue, keeps the backlog, and enters the kernel. `sq`,
  `waiting` and `flush_waiting` go (S6). Decided: an operation reaches the kernel at the next poll of the `AsyncIo`
  source instead of during `submit`.
- The `pending() == 0` gate in `Front::reap` goes (S6): the poll finds nothing on its own. `pending()` stays as the
  public count that `complete`'s doc refers to.
- Windows: the inbox (I10) and a ring's `inner` (I14) become `SourceCell`s in the poll; `break_ring` from a submitter
  serialises its cleanup.

### 3.2 SPLIT route

- **C15 `LatticeHotReceiver`**: a frame the hook defers goes to an owned `MpscQueue<Box<F>>`; the receiver is also a
  `LatticeSource` whose `pull`/`request` (under the invariant, so a `Single` consumer) offers queued frames to the hook
  again, in order, within the request's banked demand; a new `push` while frames are queued queues behind them.
  Nothing spins: the lattice's requests retry.
- **I3, I13 ring lists**: rings are never removed while the engine lives. `rings: Box<[SetOnce<Arc<Ring>>]>` of
  `MAX_RINGS` slots and `len: AtomicUsize`; a thread's first submit reserves an index (`fetch_add`) and sets it; the
  poll walks `0..len`. No clone or allocation per poll.
- **I6 io_uring ops slab**: fixed segments of 256 slots (an append-only array of `SetOnce` segment pointers), each slot
  an `UnsafeCell<Option<Op>>` with a `generation: AtomicU32`. The owner thread (the only submitter to its ring, by the
  `MINE` thread-local) takes a free index, writes the op, stores the generation with release; the poll checks it with
  acquire, updates `moved`, and on the end takes the op and returns the index over a `Queue<u32, Single, Single>`
  (producer: the poll; consumer: the owner thread).
- **I7 Windows slab**: the slot state machine (`Idle`, `Issuing`, `InFlight`, `Busy`) in an `AtomicU8` per slot,
  changed by CAS; freed indexes return over an `SpmcQueue<u32>` (producer: the poll; consumers: submitters). The
  `by_addr` map goes: `Rec` is `#[repr(C)]` with the `OVERLAPPED` first and the slot index, generation and an engine
  tag beside it, so a packet's pointer leads to its slot by a cast, and the tag rejects strays.
- **I8, I9, I15**: `MpscQueue`s (producers: failing submitters, pool threads, `launch`); the poll uses the `Single`
  consumer under the invariant.

### 3.3 OWNER route (C1, C2, C3, C10)

- `LatticeHotSource` gains a single-role pusher, as the queues do: `pusher(&self) -> HotPusher` (panics if one is
  live), holding the hooks by value (`FnMut`) and pushing with `&mut self`. The shared part keeps only a
  `DownstreamSlot`.
- `ControlPlaneFragment` keeps the pusher and the `StopWatch` in `Cycle` (the worker's own state); `timed` takes
  `&mut Cycle`.
- `BodyCost` becomes a plain struct owned by `DecisionTree`; `record`, `calibrate` and `reset` already run only on the
  worker.
- Demand (S3): the hot source banks the demand its receiver requests. While the bank is unbounded (the stock case:
  the executor's terminal requests `u64::MAX` once), `accept` pushes at once. Otherwise it pushes within the bank and
  parks the rest in an SPSC queue (producer: the pusher; consumer: `request` under the invariant), which `request`
  pushes within its demand.
- Result: no shared write per frame on the output path besides the executor's own.

### 3.4 SNAPSHOT route

- **C6, C7 vertex routing**: `routing: Published<RoutingState>`. `try_push` uses `load_in_cycle` and holds the
  reference across the downstream push. `set_downstream_mapping` and `close_after_routes_retire` store, `wait` for the
  grace period (deadline: the lattice's shutdown timeout), then call `release(old)` themselves. `RoutingState` loses
  its `Drop` hook, so `Book` is locked only by these lifecycle calls. A writer that reaches the deadline (one called
  from inside a push) defers, and `release` runs at the next lifecycle step or at drop.
- **C8, C9 edges**: `downstream` becomes a `DownstreamSlot`-like cell for the `Downstream` enum (attached once, then
  detached at `close_edge` through `Published` with a deferred retirement); `parent` becomes
  `Published<Weak<dyn LatticeNode>>`, still upgraded on the climb (off the stock worker loop), and `set_parent` skips
  the store when unchanged.
- **C5 cache directory**: `slots: Box<[Published<ControlPlaneCache>]>` with strong `Arc`s; a steal reads with
  `load_in_cycle`; `withdraw` stores `None` and defers (a stealer may be running stolen frames).
- **I11, I12 handle routes**: `Published<HashMap<Handle, Route>>`, copied on write when a handle is first seen (its
  attach, under a lifecycle lock), read per operation. `join_port` then calls `associate` once per handle, not on every
  send and receive.

### 3.5 FAN-IN route

- **C14 `PipelineRunner`**: `observed()` returns a `OneShot<Outcome>` receiver instead of `mpsc::Receiver`, recycled
  through an `MpscQueue` pool so steady state allocates nothing (decided).
- **G4 `Join`, re-armable**: built with a fixed capacity and a continuation stored once
  (`Fn(&mut [Option<T>]) + Send + Sync`), so a round allocates nothing.
  - State: `remaining: AtomicUsize`, `parts: usize`, and per part `ended: AtomicBool` and
    `value: UnsafeCell<Option<T>>`.
  - `end` asserts `!ended.swap(true)`, writes the value, then `fetch_sub(1, AcqRel)`. The arrival that reaches zero
    runs the continuation on the values in place.
  - `reset(parts)` re-arms it for the next round. It is valid once the previous round's continuation has returned
    (debug-asserted), and may be called from that continuation: it clears the flags and stores `remaining` with
    release.
  - `Join::new(parts, FnOnce(Vec<Option<T>>))` stays as a one-round convenience.
  - This is the re-armable fan-in cell the GPU-side design asks for (§3.11).
- **G5 `Latch`**: a wrapper of `OneShot`.
- **G6 `Sequencer`**: admission order is the order of an `MpscQueue<Arc<SeqCell<T>>>` (cells with an `AtomicU8` state
  and the value, recycled through a second queue). `register` offers a cell; `finish` writes and stores the state;
  `release` keeps its `wip` counter, and its winner owns the order queue's `Single` consumer, popping and emitting
  while the front is not pending.
- **G10-G15 stage graph**: the counts (`pending`, `live`, `holds`) already order a run's phases with `AcqRel`:
  - quantum (G10): `UnsafeCell<Option<Q>>`, read while `holds > 0`, taken by `conclude` after it reached zero;
  - cause (G11): `SetOnce<QuantumOutcome>`;
  - inputs (G12): `Box<[AtomicU64]>` per stage, one per lane, `fetch_max(value + 1)`, read after `pending` reached zero;
  - boundary (G14): frontier markers the same way; host completions on a `TakeAllStack`, taken by `arm_boundary`;
  - externals (G13): per stage a `SetOnce<Vec<Marker>>` output and a closable `TakeAllStack` of waiters; `quiesce`
    closes the stacks and marks the run swept, and a late subscriber reads the boundary atomics;
  - pins (G15): a `TakeAllStack<QuantumRef>` per stage.

### 3.6 QUEUE-MC route (C4)

Build each cache's queue as a `PartitionedMpmcQueue` with one `Consumer<_, Multi>` per partition:

- `greedy_while`: `drain_while` with `max_consume_batch = u64::MAX`, so one claim covers the take, as the held
  `MutexGuard` does;
- `poll_while`/`take_one`: `drain_while(consume, stop, 1)`, which peeks, tests and takes under one claim, so a
  stealer never takes an ordered head;
- `first_touch` and `reset_for_next_trial`: `Consumer<_, Multi>::clear` (lifecycle).

### 3.7 TICK route (decided: hot, fixed last)

- The dispatcher thread becomes the only steady-state caller of `ControlPlaneLattice::update`, so `updating` goes.
- The steady branch reads atomics and a `Published<ActiveSet>` and calls each shard's update directly, so the
  per-tick `TaskPool` hop goes.
- `ControlPlaneShard::update` reads a `Published` list of clones.
- In the monitor, the poller owns its evaluation state and a per-monitor provider. The last utilization is
  `Published`, the dispatcher takes the latest value from an `AtomicPtr` mailbox and is unparked, and listeners are a
  copy-on-write `Published<Vec<..>>`.
- In the mapper, versions are owned by the single updater and `current` is `Published`.
- Lifecycle `Mutex`es and `Condvar`s stay for start, stop and close.

### 3.8 BLOCKING route (M1)

Principle 6 decides by measurement. The measurements come from the GPU-side memory planner design:

| Call | Cost | Where it runs |
|---|---|---|
| `cuMemFreeAsync` (stream-ordered pool, `backends.rs:160`) | 0.2 µs | the worker, hot path |
| VMM unmap and destroy (`free_vmm`, `backends.rs:319`) | 20-52 µs (65 µs busy) | the worker, hot path |
| sub-allocation free (arena slots, scratch classes) | 0 | the worker, hot path |
| `cuMemFree` (`backends.rs:63`), `cuMemFreeHost` (`:444`) | waits for all queued device work: about 350 ms when busy | never on a worker: lifecycle, or the blocking-work facility |
| `cuMemAlloc` (`:49`), `cuMemHostAlloc` (`:478`) | not measured yet | lifecycle (arena and pool construction) until measured; then by the same rule |

Changes (chunk 9):

- `Backend::may_block()` is true exactly for the backends whose `free` calls `cuMemFree` or `cuMemFreeHost`.
- `Reaper::reap` on a reaper whose backend may block is submitted as a job to the blocking-work facility (§3.11), or
  run at lifecycle time. A reaper whose backend does not block is reaped on the worker, inline, where the memory was
  released. Nothing is handed off just to move it.
- A debug assertion in the blocking backends' `free`: the thread is not in a hot scope.
- `Reaper::reap` stops allocating a keep-`Vec` per call and stops re-offering, which reorders retirements: it stops at
  the first unretired item instead (retirements on one stream finish in order).
- Doc text, replacing `backend.rs:5-7`: "A free that waits for device-wide work (`cuMemFree` and `cuMemFreeHost`
  synchronise the device) never runs on a lattice worker: its backend reports `may_block`, and its `Reaper` is reaped
  on the blocking-work facility or at shutdown. A short free (`cuMemFreeAsync`, a VMM unmap, a sub-allocation) runs on
  the worker that releases the memory."
- Doc text, replacing the `Reaper` doc at `:95-97`: "Frees queued behind the device work that last used them. `push`
  never waits; `reap` frees what has retired. Reap a non-blocking backend's reaper on the worker, as ordinary work; a
  reaper whose backend `may_block` is reaped on the blocking-work facility or at shutdown, never on a worker."

### 3.9 Lifecycle rows

Keep. Two adjustments remove lifecycle waits from hot entry points:

- `add_upstream` checks an `AtomicBool started` before locking `started` (C26).
- `Lake::offer` no longer reaches attach (C31): a lake attaches its sinks once, in `Lake::new` (decided).

### 3.10 CONST route and the source-contract fixes

- **G7 and the lake part of S6**: `Lake::new` builds its one `Generation` and attaches its sinks. `generation` is a
  plain field, so `offer` and `admit` read it with no lock and no `Arc` clone. `attach`, `detach`, `discard`,
  `IdleWatch`, `idle_since`/`idle_for` and the `retiring` list go; `complete` completes the sinks gracefully once the
  admitted units reach zero, as today. `idle_release` and `attachments` leave the public API.
- **S1 `GiveBacks`**: no producer-side `wake`. `orders` keeps its multi-producer side; its consumer becomes the
  `Single` role in a `SourceCell`, with the banked demand. `request` adds demand and pushes ready orders within it;
  `pull` hands over at most its own demand. Requeue on `stop` stays (it is under the invariant now).
- **S2 `Moves`**: no producer-side `wake`. `request` runs `Mover::pump`, then pushes finished moves within the banked
  demand; `pull` pumps and hands over at most its own demand. Moves now advance whenever the lattice requests the
  source, which it does every cycle that has demand; nothing depends on `pull` any more.
- **S3** is in §3.3; **S4**, **S5** and the io part of **S6** are in §3.1.

### 3.11 Overlaps with the GPU-side design's RustEE requirements

The GPU-side design (its own repository) lists three RustEE primitives that this plan also needs. Each is built once,
in RustEE, by whichever plan reaches it first, and the other plan consumes it:

| Primitive | This plan | GPU-side design | Where it lives |
|---|---|---|---|
| `SourceCell<T>` (approved) | §2.0; chunk 2 | its G-11 | `euhedral-core::generics` |
| Re-armable fan-in cell (`Join` with `reset(parts)`, no allocation per round) | G4, §3.5; chunk 5 | its fan-in requirement | `euhedral-graph::join` |
| Blocking-work facility | M1, §3.8; chunk 9 | its blocking-work requirement | `euhedral-core` |

**Blocking-work facility** (shape proposed here; to be agreed with the GPU-side design before either builds it):

- `BlockingWork::new(threads)` owns a few threads outside the lattice. They are lifecycle-created, so they may park
  when idle.
- `submit(job)` is lock-free: an `MpscQueue` offer, plus an unpark only when a thread is parked. A job is
  `FnOnce() -> Option<Box<dyn Frame>>`: it runs the blocking call and may return a continuation frame.
- The facility is a `LatticeSource`: continuation frames go to an `MpscQueue`, and the facility's `request` pushes them
  within its demand (principles 4 and 5). The lattice runs them like any other frame; no wake path.
- For long or device-wide-waiting calls only (principle 6). The core's internal `TaskPool` (lifecycle tasks) can
  become its first client later; that is not needed for this plan.

## 4. Fix plan

Each chunk is one PR, in this order, and runs the workspace gates (`cargo fmt`, `cargo clippy --workspace
--all-targets`, `cargo test --workspace`; the Windows workflow for `euhedral-io`) plus the tests named.

Verification (approved): `loom` models for `SetOnce`, `Published`, `OneShot`, `TakeAllStack`, the re-armable `Join`
and the io slabs, as a dev-dependency behind `cfg(loom)`; Miri on a nightly toolchain beside the pinned stable one
for the same types' unit tests.

**Chunk 1: guard rails (no behaviour change).**

- `LifecycleMutex<T>` / `LifecycleRwLock<T>` in `euhedral-data-structures::sync`, the only code allowed to name
  `std::sync::{Mutex, RwLock}`; their lock methods `debug_assert!` that the thread is not in a hot scope.
- `HotScope`, a debug-only thread-local depth counter, entered by:
  - `run_cycle`, `UpstreamQueue::{pull, request}`, `LatticeVertex::try_push`, `ControlPlaneCache::{push,
    greedy_while, poll_while, work_steal*}`;
  - the producers `QueueSource::offer*`, `Completions::add`, `Join::end`, `Sequencer::{register, finish}`, the stage
    graph's `Run::{arrive, run_stage, submitted}`, `AsyncIo` submissions.

  Lifecycle calls made from inside a frame leave the scope explicitly.
- A workspace-root `clippy.toml`, with the same entries added to `euhedral-graph/clippy.toml` and
  `euhedral-io/clippy.toml`:
  - `disallowed-types`: `std::sync::{Mutex, RwLock, Condvar, Barrier}`, `std::sync::mpsc::{Sender, SyncSender,
    Receiver}`;
  - `disallowed-methods`: `std::thread::{park, park_timeout, sleep, yield_now}`, `std::hint::spin_loop` (queues' CAS
    loops and the idle policy carry `#[allow(.., reason = ..)]`), `Condvar::wait*`, `spin_wait::await_while`,
    `Consumer::clear` (multi role).
- Every current HOT and MIXED site gets `#[allow(.., reason = "HOT-LOCK(<id>)")]` and a hot-scope exemption.
- Tests:
  - `a_lifecycle_lock_inside_a_hot_scope_panics` (`should_panic`, debug);
  - `hot_lock_ratchet`, which counts `HOT-LOCK(` tags and fails above a constant that each later PR lowers.

**Chunk 2: the source invariant (SOURCE route plus C15).**

- Adds `SetOnce` (data-structures), and `SourceCell`, `DownstreamSlot`, `LatticeReceiver::serialize` and the
  interceptor's owner-token claim and action queue (core).
- Converts C11, C12, C13 (ingest), G1, G2, G3 (`Completions`), I1, I2, I5 (`AsyncIo` front and io_uring reaping) and
  C15 (retry source).
- Source contract: S4 (`QueueSource` pushes only in `request`, within banked demand; `request_pending` and the retry
  loop go) and S5 (no add-side sweep).
- Tests:
  - the lattice test suites with `SourceCell`'s debug check armed: any second thread inside a source panics;
  - `serialize` runs at once when the claim is free, at release when another worker holds it, and at release when
    called from a frame the source's own `pull` runs;
  - complete exactly once under attach/complete races;
  - the interceptor drops its source when it finishes (a drop counter: no source ↔ interceptor leak);
  - `QueueIngestSink::clear` and kill while a worker pulls: every frame is finished exactly once;
  - `LatticeHotReceiver` with a hook that defers N times delivers in order, and `push` returns at once;
  - demand: a source driven only by `request` (no `pull` calls) delivers every frame; it never pushes more than the
    banked demand; `pull(d)` hands over at most `d` and leaves the bank unchanged.
- Ratchet: -10.

**Chunk 3: worker output path (C1, C2, C3, C10) and S3.**

- S3: the hot source banks demand; beyond a bounded bank, frames wait in its SPSC queue for `request`.
- Tests: existing `hot_source`, `decision_tree` and fragment tests; a debug lattice run with those exemptions removed.
- Paired benchmark: frames per second and p99 service time on short frames (profile, hypothesis, paired gate, keep or
  discard).
- Ratchet: -4.

**Chunk 4: snapshots and cache partitions (C4, C5, C6, C7, C8, C9).**

- Adds `Published` and the registry's epoch slots.
- Tests:
  - `Published` drop counting (a value never freed while read; `wait` never returns early; `try_store` refuses while
    a retirement is pending);
  - remap while 8 workers push (no frame lost; every dropped downstream closed once, after its last push);
  - close while pushes are in flight;
  - stealers racing a greedy owner never take an ordered or `CacheLocal` frame;
  - a cache closed while stolen from is freed after the stealer's cycle.
- Paired benchmark as in chunk 3.
- Ratchet: -6.

**Chunk 5: fan-in and lake (C14, G4, G5, G6, G7, and the lake part of S6).**

- Adds `OneShot` and the re-armable `Join` (shared with the GPU-side design, §3.11).
- The lake attaches once in `Lake::new`; `generation` becomes a plain field; `IdleWatch`, idle tracking, detach,
  re-attach and `retiring` go (G7, G8, G9).
- Tests: existing `pipeline_runner`, `join`, `sequencer`, `latch`, `lake` tests; `Join` with 64 parts from 8
  threads; a `Join` re-armed 10 000 times with no allocation after construction (counting allocator) and a debug
  panic on `reset` before the round completes; `Sequencer` registering from 4 threads and completing in reverse; lake
  offers from 8 threads racing `complete` (every admitted unit finishes; sinks complete once).
- Ratchet: -5.

**Chunk 6: stage graph (G10-G15).**

- Adds `TakeAllStack`.
- Tests: existing `stage_graph` tests and `tests/credit_graph.rs`; a diamond graph with concurrent arrivals on 8
  workers (merged inputs equal the per-lane maximum); external subscriptions before, during and after publication and
  after quiesce.
- Ratchet: -6.

**Chunk 7: io_uring rest (I3, I4, I6) and the io part of S6.** Kernel submission moves into the `AsyncIo` source
(decided); `waiting`, `flush_waiting` and the `pending() == 0` gate go.

- Tests: existing `uring.rs` tests and `tests/io_owned.rs`; 8 submitting threads and one poller (no operation lost,
  slots reused); a poll racing `complete`.
- Ratchet: -3.

**Chunk 8: Windows engine (I7-I15).**

- Tests: the simulated-sys tests (`win/tests.rs`, every platform) and the Windows workflow's live tests; strays
  rejected by the record tag; a ring break raised by a submitter while the poll runs.
- Ratchet: -9.

**Chunk 9: memory (M1, S1, S2) and the blocking-work facility.**

- Adds `BlockingWork` in `euhedral-core` (shared with the GPU-side design, §3.11), unless that plan has already built
  it.
- M1 per §3.8: `may_block` reapers go to the facility; the debug assertion in the blocking frees; the doc text.
- S1 and S2 per §3.10: no producer-side `wake`; `request` pumps and pushes within banked demand.
- Tests: a blocking-free backend reaped inside a hot scope panics in debug; a stream-pool reaper reaped on a worker
  does not; `BlockingWork` delivers each continuation once, through `request`, within demand; give-backs and moves
  complete in a lattice that never calls `pull`.
- Ratchet: -1.

**Chunk 10: tick path (C16-C20, H1-H7).** Decided: hot, fixed last. Existing monitor, dispatcher, mapper and
lattice tests; 10 000 synthetic ticks with the hot-scope assertion armed on the dispatcher thread. Ratchet: -12.

**Chunk 11: lock the door.** The ratchet test becomes "no `HOT-LOCK(` tag exists". A new `Mutex` in hot code then
fails clippy, a lifecycle lock reached from a hot path fails the debug suite, and a second thread inside a source
fails it too.

## 5. Risks

- **The invariant becomes load-bearing for memory safety.** `SourceCell::enter` is `unsafe`; a source attached
  outside a lattice and driven from two threads at once is undefined behaviour in release builds. Mitigations:
  - the glossary contract already forbids it;
  - the debug check in `SourceCell` turns any violation into a panic in every debug test;
  - the interceptor's claim, which provides the invariant, is the only lattice-side change.
- **Lifecycle calls on a source take effect when its worker leaves.** `clear`, kill and `complete` return before their
  effect if a worker is inside the source; the effect runs before that worker releases the source. Waiting instead
  would deadlock when the call comes from a frame the source's own `pull` runs.
- **I/O submission latency**: an operation reaches the kernel at the next poll of the `AsyncIo` source instead of
  during `submit` (decided).
- **Demand banking changes delivery timing.** Sources that pushed from producers (S1, S2, S4) now deliver only when
  the lattice requests them, which it does every cycle that has demand; a lattice that stops requesting stops
  delivery, as the contract intends.
- **Reclamation bugs** in `Published` and the io slabs. Mitigations: drop-counting tests, `loom`, Miri, one
  pending retirement per cell, retirements either waited for on lifecycle threads or deferred to a known point.
- **Grace periods follow worker cycles.** A remap waits for each busy worker to finish its current cycle, which
  includes the frames it runs; a long frame lengthens the rebalance. The deadline then defers the retirement.
- **Retirement moves threads (C6, C7).** Dropped downstreams and close retirements run on the remapping or closing
  thread instead of the last pushing worker.
- **Public API changes**: `PipelineRunner::submit*` returns a one-shot (C14), `Lake` loses
  `attach`/`detach`/`idle_release`/`attachments` (S6), `LatticeHotSource` hooks moving to its pusher (C1),
  `LatticeHotReceiver` retrying through the lattice (C15), `LatticeReceiver::serialize` (a new defaulted method).
- **Windows slab rewrite (I7)**, with the `OVERLAPPED`-to-record cast: the simulated tests run on every platform, the
  live ones only on the Windows workflow.
- **Divergence from the upstream Java runtime's structure**; behaviour stays.
- **Performance**: chunks 3 and 4 carry paired benchmarks; a primitive that does not beat what it replaces on the
  per-frame path is redone, not merged.

## 6. Decisions

The questions of the earlier revisions are answered, and the answers are applied above:

1. Tick paths count as hot; they are fixed last (chunk 10).
2. `PipelineRunner::submit*` returns a lock-free one-shot (C14).
3. A lake keeps its sinks attached for its lifetime (G7, S6).
4. A blocking call is placed by its measured cost (principle 6, §3.8).
5. `loom` and Miri on nightly are approved (§4).
6. Kernel submission moves into the `AsyncIo` source (I4, chunk 7).

No new questions.
