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
6. **Long calls run on the worker that needs them.** A worker is a CPU, in the lattice or not, so handing work to
   another thread gains nothing: there is no blocking-work facility. While one worker is inside a long call, the others
   keep serving other sources and steal unordered work. Long calls are kept off per-frame paths by design (reuse
   pools, stream-ordered frees), as the Java runtime's pinned pool keeps `cuMemFreeHost` off them.
7. **Mirror the Java runtime.** Every place this plan touches takes its counterpart's fields, atomics and orderings:
   the upstream Java runtime's `core` package (and its data-structures and hardware-utils modules) for
   `euhedral-core`, `euhedral-data-structures` and `euhedral-hardware-utils`; the Java runtime `euhedral-graph` is
   derived from (its `runtime/graph` and `gpu` packages) for `euhedral-graph`, `euhedral-io` and `euhedral-memory`.
   A deviation needs a genuine Java bug (cited) or a missing Rust equivalent (usually: Rust has no garbage
   collector), and §3.12 records it.

**Decisions taken.** The tick paths count as hot and become lock-free, last. `PipelineRunner::submit*` returns a
lock-free one-shot. A lake has two modes, attached for life and idle detach (detach when idle, re-attach on the next
offer), as Java's lake. Long calls run on the worker that needs them
(principle 6). A `loom` dev-dependency and Miri on a nightly toolchain are approved. `SourceCell` is approved. Every
place mirrors its Java counterpart (principle 7); the hot source and receiver are Java's. The source primitives and the
re-armable `Join` come from the lane-source plan (§3.11). Decision 6 (I/O submission in the poll) is superseded, as the user settled, by
Java's per-thread ring (§6).

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

**Routes** (how each HOT or MIXED lock goes away; §2.1 gives every finding's route and reason, §3.12 its Java
counterpart):

- **SOURCE**: the state is only touched inside one source's `pull`/`request`, so it becomes plain single-consumer
  state in a `SourceCell`, and the receiver a `DownstreamSlot` (both from the lane-source plan, §2.0).
- **CLAIM**: single-consumer state that Java guards with a try-claim (a CAS that fails instead of waiting), because a
  second thread may reach it (a kill, a late publisher, a stealer): a `ClaimCell`.
- **PUBLISH**: a reference Java publishes with a release store and later replaces or nulls while readers may still
  hold it. Rust has no garbage collector, so the old value is kept alive by a stand-in: a **cell** of the lattice's
  registry, whose replacements travel to each reader through a queue (§2, primitive 2), a recycled allocation, or
  keeping it until its owner drops.
- **CONST**: written once (at construction or at first use) and never again: a plain field or a `SetOnce`.
- **ATOMIC**: a Java atomic or CAS that stays one in Rust (the `COMPLETE` compare-and-set of a one-use source).
- **OWNED**: Java plain fields touched by one thread. In Rust they are plain fields of the one thread's state, changed
  through `&mut self`, with no atomic: another thread reads only a snapshot the owner publishes once per window or cycle,
  never one per frame.
- **FAN-IN**: written by many producers (completions and arrivals from any worker): Java's counts, set-once cells,
  one-shots and CAS stacks.
- **THREAD**: state Java confines to one thread (a ring's submission side belongs to its submitting thread).
- **SPLIT**: producers on any thread, one consumer (the source): a lock-free queue or tag-indexed slots.
- **MOVE**: the work moves to lifecycle time (association at open) or is rebalance work Java does under its
  rebalance lock.
- **TICK**: the periodic control loop; lock-free, last.
- **WORKER**: a long call; it runs on the worker that needs it and is kept off per-frame paths (principle 6).
- **KEPT**: Java's own design, already mirrored (the hot receiver's back-off).
- **LANE-SOURCE**: fixed by the lane-source plan itself.

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
| SOURCE | C11, G3, I1, I2, I5, I10 | 6 |
| CLAIM | C4, C12, G1 | 3 |
| PUBLISH | C5, C6, C8, C9, C10, G7 | 6 |
| CONST | C1, G2 | 2 |
| ATOMIC | C13 | 1 |
| OWNED | C2, C3 | 2 |
| FAN-IN | C14, G5, G6, G10, G11, G12, G13, G14, G15, G16, G17 | 11 |
| THREAD | I4, I14, I15 | 3 |
| SPLIT | I3, I6, I7, I8, I9, I13 | 6 |
| MOVE | C7, I11, I12 | 3 |
| TICK | C16-C20, H1-H7 | 12 |
| WORKER | M1 | 1 |
| KEPT | C15 | 1 |
| LANE-SOURCE | G4 | 1 |

One finding is one lock (a field or a static) or one blocking call site, with all its acquisitions.

**Source-contract findings** (§2.5): the `pull`/demand audit of every source, re-checked against Java, leaves 3
findings (S1, S2 in `euhedral-memory`; S6a in `euhedral-io`). S3, S4, S5 and S6b (the lake's idle watch) are Java's own
design and are dropped. They are not locks, so they are not in the counts above.

**Changes from the earlier revisions.** D1 (multi-producer queue offers spinning on the tail's `RESIZING` flag while
another producer links a chunk, `queues/core.rs:371-378`) is a CAS loop inside the queue and is fine by point 2; it
moves to the "not locks" list, and the chunk that redesigned it is dropped. Twelve findings now take the SOURCE route,
which needs no new concurrent structure at all, and the `Claim<T>` primitive of the first revision is gone: every
claim it stood for was either a source's single consumer or the queue's own `Multi` consumer. The third revision
applied the decisions and principles 4-6 and added the source-contract findings. The fourth revision re-checks every
design against its Java counterpart (principle 7): routes CLAIM, ATOMIC, THREAD, MOVE, WORKER and KEPT replace
OWNER, QUEUE-MC and BLOCKING; the blocking-work facility is dropped; the source primitives and `Join` come from the
lane-source plan.

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

**The primitives that encode the invariant** are built by the lane-source plan (Part 1), in the Java runtime's shapes;
this plan consumes them and does not redefine them:

- **`SourceCell<T>`** (approved; `euhedral-core::generics`): state that only a source's `pull` and `request` (and its
  `Drop`) touch. Java keeps the same state in plain fields. `unsafe fn enter` returns a guard; debug
  builds panic on a second thread or a nested entry.
- **`DownstreamSlot`** (`euhedral-core::generics`): Java `AbstractIngestSink.Delegate`'s `attached`, `complete` and
  `live` flags around a `SetOnce` receiver. `receiver()` is an acquire load with no lock and no `Arc` clone. The
  receiver stays allocated until the slot drops, because a worker may still hold it (Rust has no garbage collector).
- **The interceptor's `wip` claim** (Java `UpstreamInterceptor.wip`): `fetch_add(1) == 0` acquires, `store(0,
  Release)` releases. It replaces today's `swap(true, Acquire)`.
- **Lifecycle calls set atomics** (lane-source Part 1 as merged, PR #11 at `29c97af`, the Java way). A source's
  `complete`, `clear` and kill touch only atomic state (and, where Java does, a try-claim that fails instead of
  waiting); the source applies that state in its next `request` or `pull`, where it is the only user, and its `Drop`
  (`&mut self`) applies whatever is left. `DownstreamSlot` has no `serialize`, and no source uses one.
- **The interceptor's `Claim`**: `try_acquire` returns a `Claim` token that the interceptor's `pull`, `request` and
  `release` require; `release` is `wip.swap(0, AcqRel)` and drains only when the swap returns more than 1.
- **`LatticeReceiver::serialize`** stays, for the interceptor's own finish only: it drops its source when it finishes
  (Java's garbage collector breaks the source ↔ interceptor cycle; Rust has to).
- **`SetOnce<T>`** (`euhedral-data-structures::sync`): a reference installed once by CAS and read with acquire.

**What the contract adds to the invariant** (principles 4 and 5). Banked demand is Java's `demand` padded atomic
(`AbstractIngestSink.Delegate`), changed with `accumulate` (saturating; a negative step floors at 0). `request(d)`
returns when the source is complete, has no receiver or `d == 0`; otherwise it banks `d` and pushes ready work within
the bank, subtracting what it pushed. `pull(d)` has the same guards and hands over at most `d`. A source's own
request-side drain may also run after its `pull` releases the queue, as Java's `hookOnPull` does through
`drainRequests`; it pushes within the banked demand only.

### 2.1 Route and reason, per HOT and MIXED finding

| ID | Class | Route | Reason |
|---|---|---|---|
| C1 | HOT | CONST | Java `LatticeHotSource` keeps its hooks in final fields and calls them without a lock; Rust hooks become `Fn + Send + Sync`, set at construction. |
| C2 | HOT | OWNED | Java `StopWatch` is plain fields used by the fragment's one worker; in Rust the worker times its own pushes (`BodyCost::timed`), so the watch is a plain field and `start` and `stop` take `&mut self`. |
| C3 | MIXED | OWNED | Java `FragmentDecisionTree`'s body-cost window is plain fields on the worker; the decision tree owns it. Other threads read a snapshot (`BodyCostView`) published once per completed window. |
| C10 | MIXED | PUBLISH | Java reads the downstream with `getOpaque` per push and nulls it at `complete`; Rust keeps a replaced receiver until the source drops. |
| C11 | MIXED | SOURCE | Java `AbstractIngestSink.Delegate`'s downstream: `DownstreamSlot`. |
| C12 | MIXED | CLAIM | Java `QueueIngestSink.Delegate.draining`: the queue's consumer is claimed by `pull`, the request drain and `discardQueued` (a kill, a late publisher), each with a CAS that fails instead of waiting. |
| C13 | MIXED | ATOMIC | Java `SingleUseSource`: the `COMPLETE` CAS decides who takes the frame. |
| G1 | HOT | CLAIM | No Java counterpart; the nearest, `QueueIngestSink`: the poll state is claimed by `emit` (from `pull`/`request`) and by a registration that lands after `complete` (Java's `onPublishedAfterClose`). |
| G2 | MIXED | CONST | Java holds its `LanePool` in a final field (`final GpuStream[] lanes`, immutable). |
| G3 | MIXED | SOURCE | `DownstreamSlot`. |
| I1 | HOT | SOURCE | Java `AsyncReads` reaps "one thread at a time" under the lattice; its reap state is plain. |
| I2 | MIXED | SOURCE | Java `AsyncReads.Delegate` is an `AbstractIngestSink.Delegate`: `DownstreamSlot`. |
| I5 | HOT | SOURCE | Java reaps a queue's completions only from the source; the lock "makes it a fact" (`uring.rs:67`): the invariant already does. |
| I10 | HOT | SOURCE | Java `CompletionPortBackend.Port`'s taken entries are plain, used only by the reaping thread. |
| C4 | MIXED | CLAIM | Java `ControlPlaneCache.pLocks`: one CAS claim per partition over a single-consumer partition; a stealer that loses it moves on. |
| C5 | MIXED | PUBLISH | Java's `WORK_STEAL` array: `setRelease` on build, `getAcquire` per steal, CAS to null on retire. |
| C6 | MIXED | PUBLISH | Java `LatticeVertex`: the routing state is published with `setRelease`; each push takes a reader count on it by CAS and re-checks the pointer. |
| C8 | MIXED | PUBLISH | Java `LatticeEdge.DOWNSTREAM`: CAS at attach, `getOpaque` per push, nulled at close. |
| C9 | MIXED | PUBLISH | Java `LatticeEdge.PARENT`: `setRelease`, `getOpaque`. |
| G7 | MIXED | PUBLISH | Java's lake (the `FrameLake` implementation): the generation is a volatile field read once per offer; attach (first use, or the first offer after an idle detach) and detach set it under the attach lock, which is rebalance work. |
| C14 | HOT | FAN-IN | Java `PipelineRunner.submit` returns a `CompletableFuture` the chain completes: a one-shot. |
| G4 | HOT | LANE-SOURCE | The lane-source plan's Part 2 rebuilds `Join` on Java's `Join` (`pending` count, per-part slots). |
| G5 | HOT | FAN-IN | No Java graph counterpart; Java waits for a result with a `CompletableFuture`: a one-shot. |
| G6 | HOT | FAN-IN | Java `Sequencer`: an MPSC order queue and a `wip` counter; whoever wins `wip` drains the ready prefix. |
| G16 | BARRIER | FAN-IN | A lane source's `request` banks demand and sets its wake flag with `SeqCst` operations on every call, once per source per lattice cycle: three full barriers (the bank's compare-and-set, the flag's exchange, the claim). Added after the audit, with the lane sources. Java's `QueueIngestSink.hookOnRequest` banks with `accumulateAndGet` and `requestPending.set(true)` (volatile: a barrier too); the rule for a frame's path is stricter than Java's. |
| G17 | BARRIER | FAN-IN | A `Join` arrival costs a `swap` on the part's closed flag plus a `SeqCst` `fetch_sub`; `reset` a `SeqCst` compare-and-set, a `SeqCst` store and a release store per part. Java's `Join.arrive` is one `decrementAndGet` and `expect` two plain volatile stores. |
| G10 | HOT | FAN-IN | Java `StageGraph.quantum` is a plain field ordered by the live and arrival counts. |
| G11 | HOT | FAN-IN | Java `AbstractQuantum.failure`: `compareAndSet(null, cause)`, the first wins. |
| G12 | HOT | FAN-IN | Java: a successor reads its predecessors' plain `marker` and `lane` after its final arrival (`ARRIVALS.getAndAdd`). |
| G13 | HOT | FAN-IN | Java `StageFrame.externals`: a CAS stack closed by `getAndSet(sentinel)`. |
| G14 | HOT | FAN-IN | Java `StageGraph.usedLanes`: a lane bit mask; the home lane joins the used lanes at quiesce. |
| G15 | HOT | FAN-IN | Java registers external edges at binding into fixed await slots, each holding a live count. |
| I4 | HOT | THREAD | Java: a ring's thread is its only submitter and enters the kernel itself (`ReadBackend.Queue.submit`); the poll only reaps. |
| I14 | MIXED | THREAD | Java `IoRingBackend.Ring`: built and submitted by its thread; a failed submit retires it (`volatile retired`). |
| I15 | HOT | THREAD | Java keeps no backlog: a submission the ring cannot take now is refused. |
| I3 | MIXED | SPLIT | Java `AsyncReads.rings`: `AtomicReferenceArray[MAX_RINGS]` and `ringCount.getAndIncrement()`. |
| I6 | MIXED | SPLIT | Java `AsyncReads.pending`: slots by tag, CAS empty → read by the submitter, emptied by the reaper. |
| I7 | HOT | SPLIT | Java `CompletionPortBackend`: one `OVERLAPPED` per tag in one array; a packet's pointer gives its tag. |
| I8 | HOT | SPLIT | Java posts a failed start's error to the port as its completion (`refused` per tag). |
| I9 | HOT | SPLIT | No Java counterpart (Java only reads); producers (pool threads) only enqueue, as Java's `publishFromCallback`. |
| I13 | MIXED | SPLIT | As I3, for Windows rings. |
| C7 | MIXED | MOVE | Java runs the retired generation's release on the last reader's thread under `synchronized(routingUpdateLock)`: rebalance work under the rebalance lock, which the rule allows. |
| I11 | HOT | MOVE | Java associates a file with the completion port when it opens it, never per operation. |
| I12 | HOT | MOVE | Java's I/O ring entries carry a raw handle reference; no handle set exists. |
| C16-C20, H1-H7 | MIXED | TICK | The periodic control loop; decided hot. Java is lock-free there except the dispatcher's spin lock, the evaluation monitor and the JDK executor's queue, which this plan replaces (§3.9). |
| M1 | MIXED | WORKER | Long frees run on the worker that needs them (principle 6); Java's pinned pool keeps `cuMemFreeHost` off per-frame paths, and per-frame device frees are stream-ordered or VMM. |
| C15 | HOT | KEPT | Java `LatticeHotReceiver` retries with the same back-off (`onSpinWait` × 128, `yield` to 512, then `parkNanos(10 µs)`); the Rust receiver already mirrors it and takes no lock. |


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
| C31 | `euhedral-core/src/flow_control/edge.rs:164` | `add_upstream_handle` spin-waits (`await_while`) while the edge drains | Attaching a source during a rebalance. Reached from `Lake::offer` only when no set is attached (the first unit, or the first offer after an idle detach): rebalance work. |
| C32 | `euhedral-core/src/control_plane/shard.rs:445`, `:539`, `:605` | waiting for clones to drain; waiting for a rebalance before a reset | Rebalance, shut-down, reset (task-pool threads and the resetting thread). |
| C33 | `euhedral-core/src/control_plane/lattice.rs:264`, `:508` | waiting for readiness; waiting for a rebalance before a reset | `start`, `clear`. |
| C34 | `euhedral-core/src/control_plane/cache.rs:142`, `ingest/queue.rs:54`, `flow_control/vertex.rs:480` | set-once cells (`OnceLock`): SMT sibling, ingest owner, interceptor upstream | Set once at build or attach by one setter; `get` on hot paths is a lock-free load. |
| D2 | `euhedral-data-structures/src/queues/core.rs:192-202` | `lock_consumers`: spins until the multi-consumer lock is free | Only `Consumer<_, Multi>::clear` (`queue.rs:434`). |
| H8 | `euhedral-hardware-utils/src/affinity/executor.rs:170-171` | a pinned executor's control state (`Mutex` + `Condvar`) | `start`, `submit*` (worker start, clone build), `shutdown*`, `close`, `await_termination`, handle clone and drop. |
| H9 | `euhedral-hardware-utils/src/affinity/executor.rs:245` | the per-CPU executor registry | `get_or_create`, `get`, `close_all`, unregister. |
| H10 | `euhedral-hardware-utils/src/monitor/mod.rs:319` | the poller's `JoinHandle` | `start`, `close_within`. |
| H11 | `affinity/executor.rs:286`, `affinity/mod.rs:76`, `topology/mod.rs:242`, `monitor/mod.rs:55` (all in `euhedral-hardware-utils/src`) | process-wide `OnceLock` statics: executor registry, affinity controller, system topology, monotonic epoch | First use, which lattice construction and worker start reach before any frame. |
| G8 | `euhedral-graph/src/lake.rs:169` | generations being retired | `retire` (`:294`) on detach and complete; `is_attached` (`:286`) is a query. Stays (rebalance): it holds a detached generation's retirement until its sinks completed. |
| G9 | `euhedral-graph/src/lake.rs:507` | the idle watch's downstream | `add_downstream` (`:532`), `complete` (`:558`). Becomes a `DownstreamSlot` with the watch (chunk 5). |
| I16 | `euhedral-io/src/win/real.rs:274`, `:278` | `OnceLock` statics: the IoRing API and the socket extension functions | Loaded when an engine opens. |
| M2 | `euhedral-memory/src/cuda/copier.rs:41`, `cuda/backends.rs:175` | `cuStreamSynchronize` | `Drop` of the copier and of the stream pool. |
| M3 | `euhedral-memory/src/planner.rs:1206`, `tiers.rs:1063` | set-once downstream (`OnceLock`) | Set at attach; `get` per request is a lock-free load. |
| load | `euhedral-gpu/src/cuda/nvrtc.rs`, `euhedral-gpu/src/module_cache.rs` (`euhedral-gpu` is outside the audit's scope) | blocking file reads, directory listings and path resolution of kernel module loading | Build-time calls (compiler and cache loading), not reached per frame; their `allow`s say `LIFECYCLE(load)`. |

`cuCtxSynchronize` and `cuEventSynchronize` are loaded (`euhedral-memory/src/cuda/mod.rs:145`, `:177`) but the
library calls neither.


### 2.5 Source-contract findings (`pull`/demand audit)

Every `LatticeSource` in scope, checked against principles 4 and 5 and then against its Java counterpart. Three
findings of the third revision are Java's own design and are dropped (S3, S4, S5); three stand (S1, S2, S6).

| Source | Java counterpart | Work only via `pull`? | Pushes beyond demand? | Redundant readiness or wake path? |
|---|---|---|---|---|
| `QueueSource` (`QueueIngestSink`, `ArrayIngestSink`, `PipelineRunner`, lake sinks) | `QueueIngestSink` | no | no | no: `request_pending` is Java's `requestPending` |
| `SingleUseSource` | `SingleUseSource` | no | no | no |
| `LatticeHotSource` | `LatticeHotSource` | no | yes, by design ("Pushes without demand") | no |
| `Completions` | none (nearest `QueueIngestSink`) | no | no | no: the add-side sweep is Java's `onPublishedAfterClose` |
| `AsyncIo` (io_uring, Windows) | `AsyncReads` | no | no | yes (S6a) |
| `IdleWatch` (lake) | the Java lake's watch | n/a (pushes nothing) | n/a | no: Java's watch polls atomics in `request`/`pull` |
| `GiveBacks` (memory planner) | none (nearest `QueueIngestSink`) | no | producers push outside `request` | yes (S1) |
| `Moves` (memory tiers) | none (nearest `QueueIngestSink`) | yes (S2) | producers push outside `request` | yes (S2) |

| ID | Site | What | Fix |
|---|---|---|---|
| S1 | `euhedral-memory/src/planner.rs:733` → `GiveBacks::wake` (`:1266`) | A producer pushes give-back frames to the downstream itself, outside `request`, running `emit` concurrently with the source's own `request`/`pull`: a wake path that breaks the single-consumer invariant. Java publishers only enqueue (`QueueIngestSink.offer`, `publishFromCallback`). | Delete `wake`. `request` banks demand (Java's `demand` atomic) and pushes ready orders within it; `pull` hands over its own demand. The order consumer goes into a `SourceCell`. |
| S2 | `euhedral-memory/src/tiers.rs:914` → `Moves::wake` (`:1119`); `pump` only in `Moves::emit` (`:1102`) | The same producer wake. `Mover::pump` runs only on the `pull` path, so a lattice that only requests never advances a move. | Delete `wake`. `request` pumps, then pushes finished moves within the banked demand; `pull` pumps and hands over at most its own demand. |
| S6a | `euhedral-io/src/uring.rs:66` (`waiting`), `:280-376` (`queue`/`flush_waiting`), `euhedral-io/src/lib.rs:536` (`pending() == 0` gate); the Windows ring's `inner` hand-off | The `waiting` flag tells the `sq` holder that another thread queued entries; Java has one submitter per ring, so there is nothing to hand over. Java's `reap` has no "anything pending" gate. | With the ring's thread as its only submitter (I4), `incoming`, `waiting`, `flush_waiting` and the backlog go. The gate goes; `pending()` stays as the public count. |

Dropped, with their Java reason:

- **S3** (the hot source ignores demand): Java `LatticeHotSource.request` is a no-op ("Pushes without demand") and its
  `pull` returns 0.
- **S4** (`request_pending`, the drain retry loop, `pull` calling `drain_requests`, producers draining after close):
  all four are Java `QueueIngestSink`'s (`requestPending`, `drainRequests`, `hookOnPull`'s `finally`,
  `onPublishedAfterClose`). They stay, with the claim made lock-free (C12).
- **S6b** (the lake's idle watch): Java's lake detaches idle sinks the same way, with a watch source whose
  `request`/`pull` read the idle state; the user keeps the idle-detach mode to cut idle CPU. It stays (G7, G9).
- **S5** (`Completions::add` sweeping after `complete`): Java's late publisher discards through the same try-claim
  (`onPublishedAfterClose` → `discardQueued`). It stays, through the `ClaimCell` (G1).

Compliant and unchanged: `SingleUseSource`, `ArrayIngestSink`, `AsyncIo`'s demand handling (`request` banks and reaps
within it; the Windows poll checks `room` before every completion).

## 3. Replacement designs

Each design names its Java counterpart (principle 7) and states any deviation; §3.12 collects them. Java access modes
map to Rust as the lane-source plan's table does: plain → plain or `Relaxed`, `getOpaque`/`setOpaque` → `Relaxed`,
`getAcquire` → `Acquire`, `setRelease`/`lazySet` → `Release`, volatile access and every `compareAndSet`,
`getAndSet`, `getAndAdd` → `SeqCst`, `VarHandle.fullFence()` → `fence(SeqCst)`.

### 3.0 Building blocks

**Consumed from the lane-source plan** (§2.0, §3.11): `SetOnce<T>`, `SourceCell<T>`, `DownstreamSlot` and the
interceptor's `wip` claim with its `Claim` token (Part 1), and the re-armable `Join` (Part 2). No source here uses
`serialize`: lifecycle calls set atomics and the source applies them.

**Built here**, in `euhedral-data-structures::sync` (the module Part 1 creates), each because nothing in the
workspace offers it:

1. **`ClaimCell<T>`** (chunk 2): Java's try-claim over plain single-consumer state (`QueueIngestSink.Delegate.draining`,
   `ControlPlaneCache.pLocks`, `TopologyMapper.drainOwner`).
   - `claimed: AtomicBool`, `value: UnsafeCell<T>`.
   - `try_claim(&self) -> Option<ClaimGuard<'_, T>>` is `compare_exchange(false, true, SeqCst, SeqCst)`, Java's
     `compareAndSet`. It never waits.
   - The guard's drop is `store(false, Release)` (Java `pLocks.setRelease(i, 0)`). `ClaimGuard::release_seq_cst` is
     `store(false, SeqCst)`, for the sites where Java releases with a volatile `set(false)` and then re-reads a flag
     (`draining.set(false)` before `drainRequests`, `drainOwner.set(false)` before `pending.get()`).
   - `is_claimed` (`Relaxed`) for tests and metrics; `get_mut` and `into_inner` with `&mut self`.
   - A lifecycle caller that must have the value (a cache's first touch or reset) loops on `try_claim` inside its
     lifecycle step.
2. **Cells and views** (chunk 4; `euhedral-core::flow_control::views`): the stand-in for Java's garbage collector where
   Java publishes a reference with `setRelease` and replaces or nulls it while readers may still hold the old one. The
   rule it follows: wherever threads meet to hand something off, a queue carries it.
   - A **cell** is a slot of lattice state of one type, made by `UpstreamRegistry::cell`. The writer (lifecycle work:
     start, stop, rebalance, attach) keeps the current value under a `LifecycleMutex` and, in the same critical section,
     offers it to the mailbox of every registered reader: one `MpscQueue` per reader, many writers, one consumer.
   - A **reader** (a worker, or any thread on its first read) owns a mailbox and a **view**: a thread-local vector of
     the values it last drained. It drains the mailbox in one batch at the start of each cycle (`refresh_view`) and at
     the start of an outermost read, and a read (`with_view`) is a load from the view: no lock, no atomic
     read-modify-write, no barrier instruction and no reference count. A nested read does not drain, so a reference
     handed out by an outer read stays valid.
   - Reclamation is `Arc`: the writer and each view hold a clone, and the last holder to let go drops the value (a
     routing generation releases its downstreams from `Drop`). A thread that exits withdraws its mailbox and drops its
     view. There is no epoch, no announcement and no barrier.
   - A reader that stops reading keeps the values it last saw until it reads again, exits, or calls `remove_reader`;
     a worker drains at every cycle start and its parks are timed.
   - Deviation (missing Rust equivalent): Java's volatile read of a field the collector keeps alive. Later chunks that
     planned a `Published` cell (the shard's arrays, the monitor and the mapper) use a cell
     where they have a registry and a queue of updates where they have not.
3. **`OneShot<T>`** (chunk 5): Java's `CompletableFuture` as `PipelineRunner.submit` uses it. `state: AtomicU8`
   (empty, waiting, set, abandoned), the value in an `UnsafeCell`, and a `SetOnce<Thread>` waiter. The setter writes,
   swaps the state (`SeqCst`) and unparks a waiter it finds (a wake, never a wait). `try_recv`, and a parking `recv`
   for a thread outside the lattice.
4. **`TakeAllStack<T>`** (chunk 6): Java `StageFrame.externals`. CAS push on an `AtomicPtr` head; `close(sentinel)` is
   one `swap` (Java `getAndSet(sentinel)`) that returns the whole list; a push that finds a sentinel hands its value
   back with the sentinel, so the pusher acts at once (Java: the edge arrives with `source.marker`).
5. **`LifecycleMutex<T>` / `LifecycleRwLock<T>`** (chunk 1): the only code allowed to name `std::sync::{Mutex,
   RwLock}`; their lock methods assert in debug builds that the thread is not in a hot scope.

### 3.1 SOURCE and CLAIM: ingest, `Completions`, the `AsyncIo` front (chunk 2)

**Ingest** (Java `AbstractIngestSink`, `QueueIngestSink`, `SingleUseSource`, `PipelineRunner`).

Lifecycle calls set atomics, as Java's and as lane-source Part 1 now requires: `complete` sets `publication_closed` and
completes the `DownstreamSlot`; kill sets the kill switch; `clear` sets `clear_requested`. Each then tries Java's
`draining` claim once (`discard_queued`); when a worker holds it, the holder applies the flags in `on_drain_finished`,
and the next `request` or `pull` otherwise. `Drop` (`&mut self`) finishes whatever is still queued. No call goes
through `serialize`.

- **C11**: `Downstream` becomes `DownstreamSlot`. `demand` stays Java's padded atomic with `accumulate`, a
  `fetch_update(SeqCst, SeqCst)` loop.
- **C12**: `consumer: ClaimCell<PartitionedConsumer<_, Single>>` is Java's `draining` flag and the queue's consumer
  side. Every path is Java's, with the claim lock-free:
  - `pull` (Java `hookOnPull`): claim or return 0; drain within its own demand; release with `release_seq_cst`; run
    `on_drain_finished`; run `drain_requests`; complete the owner when finishing and empty.
  - `request` (Java `hookOnRequest`): `request_pending.store(true, SeqCst)`, then `drain_requests`.
  - `drain_requests` (Java's loop): while `request_pending` and the claim succeeds, clear the flag, push within
    `demand.load(Acquire)`, subtract what was pushed before releasing (`release_seq_cst`), `on_drain_finished`.
  - `discard_queued` (Java's): claim or return; drain everything; release; repeat while not empty.
  - `publish` after close (Java `afterPublish`): `fence(SeqCst)`, then `on_published_after_close` if closed.
  - **`clear`** deviates (a genuine Java bug): Java `QueueIngestSink.clear()` calls the queue's single-consumer
    `clear` (`MpscQueue.clear` → `scDrain`) from the calling thread without the `draining` claim, racing a worker's
    drain. Rust `clear` claims like `discard_queued`; when the claim is held, it sets `clear_requested`, and the
    holder's `on_drain_finished` discards (`PipelineRunner`'s kill-switch pattern).
- **C13**: `SingleUseSource` takes its frame with Java's `COMPLETE.compareAndSet(false, true)`: `taken: AtomicBool`
  and `frame: UnsafeCell<Option<Box<dyn Frame>>>`, read only by the CAS winner. Deviation (a genuine Java bug):
  Java's `complete()` never sets `COMPLETE`, so a completed source still hands its frame to a later `pull`; Rust's
  `complete` also tries the CAS and, if it wins, cancels the frame (a Rust frame must be finished).

**`Completions`** (G1, G2, G3). No Java counterpart: the Java graph runtime retires device work through
`GpuStream.notifyRetired` and enqueues what follows, with no polling source. The nearest Java structure is
`QueueIngestSink`, mirrored:

- **G1**: `poll: ClaimCell<Poll>`, claimed by `emit` (from `pull`/`request`) and by a registration that lands after
  `complete` (today's add-side `sweep`; Java's `onPublishedAfterClose` → `discardQueued`). `complete` sets the closed
  flag and completes the `DownstreamSlot`; the next poll, the sweep or `Drop` cancels what is left. The losing side relies on
  the holder, which re-checks after it releases. Registrations still arrive through the `incoming` MPSC queue.
- **G2**: the lane pool is an immutable `Arc<LanePool>` field, as Java's `final LanePool`; it is dropped with the
  source.
- **G3**: `DownstreamSlot`.

**The `AsyncIo` front** (I1, I2; Java `AsyncReads.Delegate`, an `AbstractIngestSink.Delegate`).

- **I1**: `held` is plain reap state in a `SourceCell`, as Java's `completion` field ("reaped by one thread at a time").
- **I2**: `DownstreamSlot`; demand as Java's.
- The `pending() == 0` gate in `Front::reap` goes (S6a): Java's `reap` walks the rings without one.

### 3.2 The hot source and the fragment's output (chunk 3)

Java `LatticeHotSource`, `ControlPlaneFragment`'s output, `StopWatch`, `FragmentDecisionTree`.

- **C1**: the hooks are final fields, `Option<Box<dyn Fn(&dyn Frame) + Send + Sync>>` and `Option<Box<dyn Fn() + Send
  + Sync>>`, called without a lock. The builder methods take `Fn` instead of `FnMut` (a public API change).
- **C10**: one word, `downstream: AtomicPtr<Link>` (a link holds the receiver's `Arc`): null is open with no receiver,
  a link is attached, and a `COMPLETE` tag is completed (Java's two words, folded; see §3.12).
  - `accept`: load (`Acquire`); return a failed frame when it is null or `COMPLETE`; before-hook; load again; push;
    after-hook.
  - `add_downstream`: a `SeqCst` CAS loop that installs over null or `COMPLETE` and refuses (`on_error`) over a link;
    then `receiver.add_upstream`.
  - `complete`: `downstream.swap(COMPLETE, SeqCst)`; a removed link is retired, and its receiver told `on_complete`.
  - Deviations: the pointer load is `Acquire`, not opaque (Java's final-field publication has no Rust equivalent).
    A swapped-out receiver is kept on a retired list (an intrusive push-only stack of links, touched only by
    `complete`) until the source drops, because the pusher may still hold the pointer (no garbage collector).
    `complete` tells the old receiver `on_complete`, which Java does not: Rust's interceptor drops its source on
    `on_complete` (lane-source Part 1) to break the cycle Java's collector breaks. `accept` without a downstream
    finishes the frame with an error (a genuine Java bug: Java's `accept` returns without finishing the frame, so its
    completion never runs).
  - `request` stays a no-op and `pull` returns 0 (S3 dropped).
- **C2**: `StopWatch` keeps Java's fields and logic as plain fields, `start` and `stop` taking `&mut self`. The worker
  owns it (inside `BodyCost`) and times its own pushes with `BodyCost::timed`, a guard that also stops the watch when
  the push panics, so the two output hooks no longer share it.
- **C3**: `BodyCost`'s window is plain fields owned by the worker's decision tree: `record`, `calibrate` and `reset`
  take `&mut self`, as Java's plain fields. Another thread reads a `BodyCostView` (the estimate and whether a window
  has been recorded), which the owner publishes once per completed window and on reset (`ControlPlaneFragment::
  body_cost_ns`).

### 3.3 Cache, vertex and edges (chunk 4)

- **C4** (Java `pLocks`): `partitions: Box<[CachePadded<ClaimCell<Partition>>]>`. `greedy_while`, `poll_while`,
  `work_steal` and `work_steal_polled` `try_claim` each partition and skip it on failure (Java `continue`); the guard
  releases with `Release`. `first_touch` and `reset_for_next_trial` loop on `try_claim` (lifecycle). Deviation: Java
  skips the claim when work stealing is off (`pLocks == null`); Rust always claims (one uncontended CAS), because a
  plain consumer shared behind `&self` has no safe Rust form.
- **C5** (Java `WORK_STEAL`): `slots: Box<[Cell<CacheRef>]>`, one cell per CPU. Build stores (Java `setRelease`); a steal
  reads the slot from the thread's view (Java's `getAcquire`) and upgrades the weak reference, then holds the cache for
  the steal; `withdraw` stores `None` only if the slot still holds this cache (Java's CAS to null, `set_cell_if`, decided
  under the lock). The cache's `retired` flag refuses pushes, as Java's. Deviation: a cell stands in for the collector,
  and the slot holds a weak reference, so a directory that owned its caches strongly would form a cycle with the
  caches' directory handle; a cache lives as long as its owner holds it. The upgrade is one reference-count update per
  steal that finds a cache; a borrowed reference would need a proof that the cache outlives the steal, which the
  view cannot give.
- **C6** (Java `LatticeVertex.ROUTING_STATE`, `RoutingState.admission`). The routing generation is a cell of
  `Arc<RoutingState>`.
  - `try_push` reads the state from the worker's view (Java's `getAcquire` of `ROUTING_STATE`), checks that the vertex
    is not closed (`Relaxed`, Java opaque), routes and pushes. It takes no count: Java's `admission` reader count
    (a compare-and-set and a decrement on one line every pusher shares per frame) is not needed, because a generation's
    downstreams are released when the last holder of its `Arc` lets go, and a view holds it until the reader drains.
  - The writer (`set_downstream_mapping`, `close_after_routes_retire`, under the routing-update lifecycle lock) builds
    the next state, retains its handles, and stores it in the cell; the old generation's `Arc` drops from the store
    and, as each reader drains, from the views.
  - `UpstreamInterceptor::request` names its vertex for the call, so the pushes of the source inside it take no
    reference of their own (Java's interceptor holds its vertex in a field).
  - Deviation (a Java bug against the lock-free rule): Java retires the old state first, so a reader that loaded it
    spins until the writer publishes, and a writer descheduled inside `synchronized` stalls every pusher; here the new
    generation is stored before the old one is let go, and no reader waits.
  - Deviation (missing Rust equivalent): the release runs from `Drop` of the last holder instead of from the reader
    count reaching the retired value. A reader that stops reading holds its generation until it reads again or exits.
    The release therefore waits for every worker's next cycle or read (Java: for the last in-flight push); bounded by
    the cycle and the timed parks.
  - Deviation (Java bug against the frame-path rule): Java runs a deferred retirement on the last reader, and a
    retirement that closes a clone waits for its workers (`closeClone` → `awaitTermination`): for the clone's own worker
    that is a wait on itself until the shutdown timeout, and for any worker a stall of its frames. Here the retirements
    go to a thread of their own when the last holder is a frame-path thread.
- **C7** (Java `releaseHandlesWhenQuiescent` under `synchronized(routingUpdateLock)`): rebalance work, run by
  whichever thread lets go of the last `Arc` of a retired generation, as in Java it is run by the last reader. The
  releasing thread leaves its hot scope for the lifecycle lock (once per retired generation).
- **C8** (Java `LatticeEdge.DOWNSTREAM`): `downstream: SetOnce<Downstream>` installed by CAS (Java
  `compareAndExchange`: the loser gets the existing value and attaches below it, as Java). A push reads the edge's
  `closed` flag (`Relaxed`, Java opaque) and the slot (`Acquire`). `close_edge` swaps `closed`, then tells a receiver
  `on_complete`. Deviation: the downstream stays allocated until the edge drops (an edge closes for good, so nothing
  re-attaches).
- **C9** (Java `LatticeEdge.PARENT`): `parent: Cell<ParentRef>` (a weak reference). The climb reads it from the view
  and upgrades; `set_parent` stores, skipping an unchanged parent. Deviation: a cell stands in for the collector; the
  upgrade is one reference-count update per hop of a request or pull.

### 3.4 Fan-in and the lake (chunk 5)

- **C14** (Java `PipelineRunner.submit` → `CompletableFuture`, `root.observe(future)`): `submit*` returns a
  `OneShot<Outcome>` receiver; one is allocated per submission, as Java allocates its future.
- **G4**: rebuilt by the lane-source plan's Part 2 on Java's `Join`; nothing left here.
- **G5** `Latch`: removed. The main thread checks a `OneShot` periodically instead of blocking on it.
- **G6** `Sequencer` (Java `Sequencer`):
  - `order`: an `MpscQueue` of `Arc<Entry>`, offered by `register` in order. An entry holds `ready: AtomicBool`
    (Java's entry `ready()`) and its value in an `UnsafeCell`.
  - `finish` writes the value, stores `ready` (`SeqCst`), and calls `drain`.
  - `drain` is Java's: `if wip.fetch_add(1, SeqCst) != 0 { return }`; `missed = 1`; loop: while the head is ready,
    emit it and pop it; `missed = wip.fetch_sub(missed, SeqCst) - missed`; return at 0. The order consumer is owned
    by the `wip` winner (an `UnsafeCell`, `// SAFETY:` the `wip` count), as Java's `peek`/`poll`.
  - A failing emit is logged and the drain continues (Java's catch).
- **G16** lane source `request` (Java `QueueIngestSink.hookOnRequest`): bank the demand with one `fetch_add(Release)`
  (saturating: a sum that wrapped is stored back as unbounded), only for a non-zero demand; set the wake flag with a
  plain release store; the claim attempt that follows is the requester's one full barrier, and the holder reads the flag
  after its sequentially consistent release of the claim and takes it with `swap(false, SeqCst)`. Unbounded demand
  is never spent, so serving it takes no read-modify-write; spending is one `AcqRel` compare-and-set.
  Deviation (Java bug against the frame-path rule): Java's volatile bank and volatile flag store are full barriers.
- **G17** `Join` (Java `Join`): an arrival writes its slot and does one `fetch_sub(AcqRel)`. Java has no check that a part
  ends once; here a part is a **move-only token** (`Part`, handed out by `reset`, `Round::rearm`, `Join::once` and
  `arm_owned`), consumed by `arrive` or `abandon`, and dropping it abandons. A double arrival does not compile, there is
  no range check and no debug-only flag, and a killed unit drops its tokens so its round completes with `None`s.
  Deviation (missing Rust equivalent): Java's `Join` is a shared object whose parts are bare indexes, safe only by the
  caller's contract; in Rust that contract is undefined behaviour from safe code, so ending is by ownership, and a
  part holds the join (an `Arc`) because there is no collector to keep it alive. `Join::arm_owned` is Java's `expect`
  for a cell whose owner proved the previous round over: a release store of the count after the values a cut-short
  round left are dropped; `unsafe`, because only the caller knows that no token of the old round will end.
  `Join::discard_round(&mut self)` is the safe form for an owner with exclusive access (`Arc::get_mut`): no token
  exists then. `is_idle` says whether a round is armed. `reset` keeps its check (one `AcqRel` compare-and-set).
- **G7, G8, G9: the lake's two modes** (Java's lake, the `FrameLake` implementation). `idle_release` stays:
  `Duration::ZERO` keeps the sinks attached for life (Java `idleDetachNanos <= 0`); a positive value detaches them once
  no unit has been active for that long, and the next offer re-attaches a fresh set. The hot side is Java's:
  - `generation: Cell<Generation>` (Java `volatile Generation generation`). `offer` and `admit` read it from the
    thread's view: no lock and no `Arc` clone for a registered reader. Deviation: a cell stands in for the collector,
    because an offering thread may hold a set that is being detached; the set is released when the last view that
    holds it drains.
  - `offer` (Java `publish`): read the set; offer to the sink its routing hash picks; a detached set's sinks refuse
    (publication closed), and the offer then calls `attach` and tries the set attached since, as Java's loop; a lake
    that finished, or an `attach` that returns the same set, refuses.
  - `admit` (Java `admit`): count first (`accepted` CAS, the `CLOSED` sign bit as Java), then `attach`, so a watch that
    finds the lake idle afterwards cannot detach the set it returns.
  - `attach` and `detach` (Java's, under `synchronized(attachedSources)`): rebalance work under `attach_lock:
    LifecycleMutex<()>`, taken after `HotScope::leave()`. `attach` returns the current set without the lock when one is
    attached (the per-offer fast path); otherwise it builds and attaches a set and its watch, counts `attachments`, and
    stores it. `detach(from)` returns unless `from` is current, the lake is not finished and no unit is active; it
    stores `None`, hands the retirement to `retiring` (G8, lifecycle) and finishes `from`'s sinks gracefully.
  - Idleness (Java `idleSince`): `idle_since_ns: AtomicU64`, stored (`SeqCst`) by the `terminated` that brings the
    active count to zero; `idle_for` reads it.
  - The watch (Java's `Generation.watch`, an ingest delegate): its `request` and `pull` poll: return when idle detach
    is off, a unit is active, or the lake has been idle for less than `idle_release`; otherwise `detach(owner)`. Its
    downstream is a `DownstreamSlot` (G9). As Java, the watch is attached in both modes.
  - `complete_gracefully`, `terminated` and `finished` as Java's.
  - Public API unchanged: `attach`, `detach`, `idle_release`, `attachments`, `is_attached` stay.

### 3.5 Stage graph (chunk 6)

Java `StageGraph`, `StageFrame`, `AbstractQuantum`.

- **G10** quantum: Java's plain `quantum` field, set by `start` before the roots are published and cleared after
  retirement. Rust: `UnsafeCell<Option<Q>>`, read while `holds > 0`, taken by `conclude` after it reached zero
  (`// SAFETY:` the counts).
- **G11** cause: Java `failure.compareAndSet(null, cause)`: `SetOnce<QuantumOutcome>`.
- **G12** inputs: Java's successor reads its predecessors' plain `marker` and `lane` after its final arrival. Rust: one
  slot per incoming edge (`UnsafeCell<Option<Arrival>>`), written by the arrival before `pending.fetch_sub(1,
  SeqCst)`; the arrival that reaches zero reads all slots. The per-stage `Mutex<Vec<Marker>>` merge goes.
- **G13** externals: per stage a `TakeAllStack` of waiters, closed with Java's two sentinels: `SUBMITTED` (later
  subscribers arrive at once with the stage's output markers) and `SWEPT` (they arrive with the graph's joined
  marker). The output markers are a plain slot written before the closing `swap`, which publishes them.
- **G14** boundary: Java `usedLanes`: a lane bit mask (`load`, then `fetch_or` when the bit is missing). At quiesce the
  home lane joins every other used lane (mark there, await on home), so the per-lane frontier list goes. A deferred
  host completion holds one `live` count until it completes (Java `deferral`/`completeDeferred`), so the boundary
  needs no host list. Deviation: a pool wider than 64 lanes uses `Box<[AtomicU64]>` (Java's mask is one `long`).
- **G15** pins: Java registers external edges at binding into fixed await slots (`externalIn`, `firstSlot`), each
  holding a `live` count. Rust: `Box<[UnsafeCell<Option<QuantumRef>>]>` by slot, written at registration (before
  `start` publishes the roots), taken by the stage after its final arrival or by `quiesce` once `live` is zero.

### 3.6 io_uring (chunk 7)

Java `AsyncReads` and `UringBackend` (its `ReadBackend.Queue`).

- **I4, S6a**: one ring per submitting thread, and **the thread is its ring's only submitter**: `submit` builds the
  entry and enters the kernel itself (Java `Queue.submit`). The submission side is thread-confined state (an
  `UnsafeCell` reached only through the thread's `MINE` entry; a debug check compares the owner's thread id). `sq`,
  `incoming`, `waiting`, `flush_waiting` and the backlog go. This replaces decision 6's "submission in the poll":
  Java's design gives the same single-owner submission side without the poll's added latency.
- A submission the ring cannot take now (no room, the ring's in-flight count at its completion-queue bound, or a busy
  tag slot) is refused: `submit*` hands the request back (Java returns false and the caller acts synchronously). A
  public API change.
- Short pieces: the reaping thread resubmits the rest on **its own** ring (Java `reap` → `ring()`); a rest that
  cannot be submitted ends the operation with an error, as Java's.
- **I5**: the completion queue is reaped only by the source (`SourceCell`).
- **I3** (Java `rings`, `ringCount`): `Box<[SetOnce<Arc<Ring>>]>` of `MAX_RINGS` and an `AtomicUsize`; a thread's first
  submit takes an index with `fetch_add` and sets the slot; the reaper walks `0..min(count, MAX_RINGS)` and skips unset
  slots. A thread past `MAX_RINGS` is refused (Java returns null).
- **I6** (Java `pending`, `tags`): `pending: Box<[Slot]>` indexed by `tag & (PENDING - 1)`, tags from
  `tags.fetch_add(1)`. The submitter CASes the slot's state empty → busy, writes the operation, and stores full
  (`Release`); the reaper reads it (`Acquire`), takes it, and stores empty. Deviation: Java installs a frame reference
  in one CAS; Rust writes the operation in place (no allocation per operation), so it reserves first.
- `in_flight` (Java `inFlight`): `fetch_add`/`fetch_sub`.

### 3.7 Windows engine (chunk 8)

Java `CompletionPortBackend` (one shared port) and `IoRingBackend` (one ring per thread).

- **I7**: tag-indexed slots as I6. One `OVERLAPPED` per tag lives in one array, so a packet's `OVERLAPPED` pointer
  gives its tag (`(ptr - base) / size`), as Java's `reap`. The `by_addr` map goes; an engine tag in the record rejects
  strays.
- **I8**: a call that fails to start posts its error to the port as its completion (Java `refused` per tag and
  `PostQueuedCompletionStatus`): `refused: Box<[AtomicU32]>`. The failed list goes; if the post fails, `submit`
  refuses.
- **I9**: pool-job results (open, rename) have no Java counterpart (Java only reads). Pool threads only enqueue on an
  `MpscQueue` (as Java's `publishFromCallback`), and the source drains it.
- **I10**: the taken entries are plain reap state in a `SourceCell` (Java `Port.entries`/`taken`/`next`).
- **I11**: association with the port happens when the engine opens a file (Java `open`), and for a socket made
  outside the engine at an explicit `register` (lifecycle). No per-operation lookup. Deviation: Java has no sockets;
  `register` is the Rust form of its `open`.
- **I12**: I/O ring entries carry a raw handle reference (Java `IORING_REF_RAW`); the handle set goes.
- **I13**: as I3.
- **I14, I15** (Java `IoRingBackend.Ring`): the ring is built into and submitted by its thread only; an in-flight bound
  refuses beyond the ring's entries; a failed submit sets `retired: AtomicBool` (`Release`; Java `volatile retired`)
  and the thread makes a new ring next time. The reaper still reaps a retired ring until it is empty. The backlog and
  `break_ring`'s lock go.

### 3.8 Memory (chunk 9)

- **M1** (principle 6; Java `CudaGpuMemory`'s pinned pool):
  - `cuMemFreeAsync` (0.2 µs), VMM unmap and destroy (20-52 µs, 65 µs busy) and sub-allocation frees run where the
    memory is released, on per-frame paths.
  - `cuMemFree` and `cuMemFreeHost` (they wait for the device's queued work, about 350 ms when busy) run on the worker
    that needs them, and are kept off per-frame paths by design:
    - device memory on per-frame paths comes from the stream pool or VMM, whose frees are short;
    - the pinned host backend keeps released blocks for reuse, as Java's pinned pool: power-of-two size classes of
      at least 64 KiB, one `MpmcQueue` per class, `pool_bytes: AtomicU64`, a byte limit (default 1/256 of the
      machine's memory, at least 64 MiB). A release beyond the limit goes to an excess queue; `trim()` frees it on
      whichever worker calls it, and `drain` frees everything at close (Java `pinnedExcess`, `trimPinnedPool`,
      `drainPinned`).
  - `Reaper::reap` runs on the worker as ordinary work; `may_block` routing is not added. Its per-call keep-`Vec` goes:
    it stops at the first unretired item (retirements on one stream finish in order).
  - Deviation: Java's comment says a worker must never wait for a pinned free; the user's rule (long calls run on the
    worker that needs them) replaces that, and the pool structure is kept.
  - Doc text, replacing `backend.rs:5-7`: "A free that waits for the device (`cuMemFree`, `cuMemFreeHost`) runs on the
    worker that needs it. Per-frame paths use the stream-ordered pool, VMM or sub-allocation, whose frees are short;
    pinned host blocks are pooled for reuse, so their frees happen only when the pool trims or closes."
  - Doc text, replacing the `Reaper` doc at `:95-97`: "Frees queued behind the device work that last used them. `push`
    never waits; `reap` frees what has retired, on the worker that calls it, as ordinary work."
- **S1, S2** (nearest Java: `QueueIngestSink`, whose publishers only enqueue): `GiveBacks::wake` and `Moves::wake` go.
  `request` banks demand (Java's `demand` atomic), pumps (moves), and pushes within it; `pull` pumps and hands over at
  most its own demand. The order and done-queue consumers go into `SourceCell`s.

### 3.9 TICK (chunk 10)

Java `ControlPlaneLattice`, `ControlPlaneShard`, `ResourceMonitor`, `LatestValueDispatcher`, `TopologyMapper`.

- **C16**: Java `closed` and `resetting` (`AtomicBoolean`, `getAcquire`): the close `RwLock` goes.
- **C17**: Java has no update lock: `update` runs only on the dispatcher thread, and topology changes are guarded by
  `rebalancing.compareAndSet`. The `updating` `Mutex` goes.
- **C18**: Java `currentGlobalVersion` (volatile), `activeShardIds` (`AtomicReference<int[]>`, `getOpaque`), `primed`,
  `rebalancing`, `ready`: atomics, and `active_shards: Published<Box<[usize]>>`.
- **C19**: Java `clones` and `activeCoreIds` (`AtomicReference`, `setRelease`/`getOpaque`), `currentVersion`,
  `started`, `primed`, `rebalancing`; `lifecycleLock` only for start and shut-down. Rust: the same atomics, the clone
  list `Published`; the per-tick `Vec` of clone `Arc`s goes.
- **C20**: Java hops each socket's update to `controlPlaneExecutor` (`CompletableFuture.runAsync`); the hop stays.
  Deviation: the JDK's fixed pool queues on a `LinkedBlockingQueue` (a `ReentrantLock`), and ticks are hot
  (decided), so `TaskPool` gets a lock-free `MpmcQueue` of jobs; idle pool threads park, and a submitter unparks one
  (pool threads are lifecycle-created; the submitter never waits).
- **H1**: Java `ResourceMonitor.state` (volatile int, CAS transitions) and park/unpark of the polling thread; the
  poller's wait for its next tick (`DeadlineWaiter`, `parkNanos`) is its timer and waits on no other thread. Rust:
  `state: AtomicU8`, the poller parks until its deadline, `stop`/`close` CAS then unpark. `close_within` keeps its
  spin and park (lifecycle).
- **H2**: Java guards evaluation with `synchronized(evaluationLock)`, plus the `EVAL_ACTIVE`/`PUB_CLAIMED` Dekker flags
  against close. Deviation (ticks are hot, decided): the evaluation state is a `ClaimCell` the poller takes; an
  on-demand evaluation from another thread that loses it returns the last published utilization (Java's
  `coalescedReadPending` coalescing). The Dekker flags stay, `SeqCst`.
- **H3**: Java `volatile lastUtilization`: `Published<HardwareUtilization>`.
- **H4**: Java's snapshot provider is a field of the monitor, used inside evaluation; Rust: a per-monitor provider
  under H2's claim. The process-wide `system_snapshot` stays for callers outside the monitor (not on the tick path).
- **H5**: Java `LatestValueDispatcher` keeps `pendingUtilization` and the listener array under a CAS spin lock with a
  park back-off. Deviation (ticks hot): `offer` swaps an `AtomicPtr<HardwareUtilization>` mailbox (latest wins) and
  unparks the dispatch thread; listeners are a copy-on-write `Published<Box<[Listener]>>` changed at attach and
  detach under a lifecycle lock; `closing` is `SeqCst`, then unpark. The dispatch thread's park is its idle wait.
- **H6, H7**: Java `TopologyMapper`: `pending` (`AtomicReference`, install the greatest by CAS), `drainOwner` (claim,
  release with `set(false)`, then re-check `pending`), `socketVersions` plain under the claim, `volatile
  effectiveTopology`. Rust: `pending: AtomicPtr`, `ClaimCell<SocketVersions>` released with `release_seq_cst`,
  `effective: Published<EffectiveTopology>`.

### 3.10 Lifecycle rows

Keep. Two adjustments remove lifecycle waits from hot entry points:

- `add_upstream` checks an `AtomicBool started` before locking `started` (C26), as Java's `started` atomic.
- `Lake::offer` reaches attach only for the lake's first unit (C31): the sinks stay attached afterwards.

### 3.11 Overlaps with the lane-source plan

The lane-source plan (`docs/superpowers/plans/2026-10-29-lane-source.md`, being built) builds these primitives in the
Java shapes, and this plan consumes them; neither plan builds one twice:

| Primitive | Lane-source part | Java counterpart | Consumed here by |
|---|---|---|---|
| `SetOnce<T>` | Part 1 (Task 1.1) | a reference set by `compareAndSet(null, v)`, read by `getAcquire` | C8, G11, I3, I13 |
| `SourceCell<T>` (approved) | Part 1 (Task 1.2) | plain fields under the interceptor's `wip` | I1, I5, I10, S1, S2 |
| `DownstreamSlot` | Part 1 (Task 1.3) | `AbstractIngestSink.Delegate` (`attached`, `complete`, `live`) | C11, G3, I2 |
| `wip` claim and `Claim` token; `serialize` for the interceptor's own finish | Part 1 (Task 1.4, PR #11 at `29c97af`) | `UpstreamInterceptor.wip` | every source, through the lattice; no source calls `serialize` |
| Re-armable `Join` | Part 2 (Tasks 2.1-2.2) | `Join` (graph runtime) | G4 (done there) |

The blocking-work facility of the third revision is dropped (principle 6); neither plan builds it.

### 3.12 Java counterparts and deviations

"Core" is the upstream Java runtime's `core` package (with its data-structures and hardware-utils modules); "graph
runtime" is the Java runtime `euhedral-graph` is derived from (its `runtime/graph` and `gpu` packages).

| Place | Java counterpart | Deviation and why |
|---|---|---|
| C1, C10 hot source | core `flow_control/LatticeHotSource` | **One word, not Java's two.** `downstream` is null (open, no receiver), a link, or a `COMPLETE` tag, so `complete` is one `swap(COMPLETE)` and `add_downstream` one CAS loop that installs over null or `COMPLETE` and refuses over a link. Java sets `complete` and `downstream` in two steps; racing `complete` and `add_downstream` can then leave an attached receiver never told (the attach lands between the two steps) or tell a live receiver the source completed (a reopen between them). Here Rust has `on_complete` and no GC, and a receiver that is not told leaks the interceptor's source cycle, so the one word makes `complete` linearizable, as the lock the base held made it. Acquire pointer load (no final-field rule). Retired receivers are kept until drop (no GC), on an intrusive push-only stack of links rather than a lifecycle-locked list, so `add_downstream` and `complete` set atomics only; a source attached `k` times holds `k` retired receivers until it drops (a bounded scheme, counting the pushes in flight and freeing a retired link at zero, is possible if a reattach-heavy user appears; not built). `on_complete` at complete (the interceptor drops its source on it: no GC). `accept` finishes an unsent frame (Java bug: it drops it unfinished), finishes the frame when the before-hook panics, and runs the after-hook after every before-hook that returned, also when the push panics. |
| C2 stop watch | core `utils/StopWatch` | **Plain fields, not atomics.** Java's watch is plain state of one thread, and the Rust worker times its own pushes, so no closure shares it. The first Rust design shared it between the output's two hooks as relaxed atomics (deviation category: missing Rust equivalent of a plain field shared by two closures); owning it on the worker removes that need. |
| C3 body cost | core `control_plane/FragmentDecisionTree` | Plain fields owned by the decision tree, as C2; a snapshot for other threads, published per window. |
| C4 cache partitions | core `ControlPlaneCache.pLocks` | Always claimed (missing Rust equivalent: `Consumer<_, Single>` behind `&self` needs proven exclusive access). |
| C5 cache directory | core `ControlPlaneCache.WORK_STEAL` | Cells of weak references (no GC); a stealer holds the cache it upgraded for the steal. Known cost: each steal attempt that finds a cache does one reference-count update on it (`Arc<Weak>` plus `upgrade`), which Java's plain reference does not. |
| C6 routing | core `LatticeVertex.ROUTING_STATE`, `RoutingState.admission` | No reader count: the generation is a cell, released by the last holder of its `Arc` (missing Rust equivalent of the collector); the next state is stored before the old one is let go, so no reader waits for the writer (Java bug against the lock-free rule). |
| C7 routing book | core `LatticeVertex.releaseHandlesWhenQuiescent` | None. |
| C8, C9 edges | core `LatticeEdge.DOWNSTREAM`, `PARENT` | Downstream kept until drop; a receiver added to a closed edge is told complete once, taken or refused (a Java race: `close` nulls `DOWNSTREAM`, so a later compare-and-set attaches a receiver that is never told); parent a cell (no GC). |
| C11, C12 ingest | core `ingest/AbstractIngestSink`, `QueueIngestSink` | `clear` claims (Java bug: unclaimed single-consumer drain); demand is unsigned and saturates at the top (`add_demand`, `take_demand`) where Java's `accumulate` takes a signed step; the queue is finished when its source drops (no GC); `clear` cancels the frames (`do_finally_with_error`) where Java's `clear` drops them silently; `pull`'s stop also matches once the source has completed. |
| C13 single use | core `ingest/SingleUseSource` | `complete` cancels an untaken frame (Java bug: `complete` leaves it takeable); a `pull` whose stop condition matches gives the frame back (Java ignores the stop condition); a frame nobody took is cancelled when the source drops. |
| C14 pipeline outcome | core `ingest/PipelineRunner`, `CompletableFuture` | One reader that polls, not Java's many waiters and callbacks; a chain dropped without ending abandons the receiver where Java's future would stay incomplete (missing Rust equivalent: no collector notices a dropped future). |
| C15 hot receiver | core `flow_control/LatticeHotReceiver`, `utils/SpinWait` | None (already mirrored). |
| C16-C19 lattice, shard | core `ControlPlaneLattice`, `ControlPlaneShard` | `Published` for the arrays (no GC). |
| C20 task pool | core `controlPlaneExecutor` (JDK fixed pool) | Lock-free job queue: the JDK queue locks, and ticks are hot. |
| H1-H4 monitor | hardware-utils `ResourceMonitor`, `SystemSnapshotProvider` | H2: claim instead of `synchronized` (ticks hot); H3 `Published` (no GC). |
| H5 dispatcher | hardware-utils `LatestValueDispatcher` | Mailbox swap and copy-on-write listeners instead of its spin lock (ticks hot). |
| H6, H7 mapper | hardware-utils `TopologyMapper` | `Published` (no GC). |
| G1-G3 completions | none; nearest core `QueueIngestSink` | A request that finds the poll held leaves a wakeup for the holder (`request_pending`, as `QueueIngestSink`'s); the lane pool lives with the source (a marker is reached when its lane says so, also after `complete`, as in Java); the poll state is a `SourceCell`, the state a holder mutates; a registration whose marker is still unreached is leaked and counted instead of finished when it is cancelled (at drop, by a sweep, or refused after `complete`), because the device may still use what it holds. |
| G4 join | graph runtime `Join` | In the lane-source plan. |
| G5 latch | none; `CompletableFuture` | Removed with `Latch` (user decision: a blocking wait for the lattice's work is not needed; the main thread checks periodically with a one-shot). It was the last blocking call in `euhedral-graph`. |
| G6 sequencer | graph runtime `Sequencer` | A panic in `emit` resumes on the releasing thread once the drain ended; Java logs it (the Rust caller of `complete` sees it). Entries are `Arc`s (no GC); `len` is the order queue's length. `ready` is release/acquire and the work-in-progress count is the one read-modify-write (the claim). |
| G16 lane source request | core `QueueIngestSink.hookOnRequest` | A release-ordered bank and flag; the claim attempt is the barrier (Java: volatile bank and flag). |
| G17 join | graph runtime `Join` | Debug-only double-arrival check (Java: none); `arm_owned` is Java's `expect` and `unsafe` (no collector); `is_idle`. |
| G7-G9 lake | graph runtime `FrameLake` implementation (both modes) | A cell for the generation (no GC), with the attach lock's own strong reference to the set for the lifecycle side; retirements held in `retiring`. The terminating unit stores `idle_since` before it lowers the count and the watch reads the count before the time (Java stores after the decrement, so a watch that sees the count at zero can read the previous idle time and detach a set that just went idle: a Java bug). The watch's downstream is a `DownstreamSlot`. |
| G10-G15 stage graph | graph runtime `StageGraph`, `StageFrame`, `AbstractQuantum` | Wider-than-64-lane masks. |
| I1-I6 io_uring | graph runtime `AsyncReads`, `UringBackend` | Operations written in place behind a reserve step (no allocation per operation). I1, I2: a request that finds the front entered leaves a wakeup for the holder (`request_pending`, as `QueueIngestSink`'s). |
| I7-I15 Windows | graph runtime `CompletionPortBackend`, `IoRingBackend` | `register` for sockets (Java reads files only); pool-job results have no counterpart. |
| M1 frees | graph runtime `CudaGpuMemory` pinned pool | Trim on a worker (the user's rule over Java's comment). |
| S1, S2 memory sources | none; nearest core `QueueIngestSink` | — |

## 4. Fix plan

The implementation plan is `docs/superpowers/plans/2026-10-29-lock-free-hot-path.md` (same branch): bite-sized TDD
tasks, one draft PR per chunk. In outline:

| Chunk | Fixes | Ratchet | Waits for |
|---|---|---:|---|
| 1. Guard rails | `LifecycleMutex`, `HotScope`, clippy config, `HOT-FIX(<id>)` tags, the ratchet test | sets 57 | lane-source Parts 1 and 2 merged |
| 2. Sources and claims | C11, C12, C13, G1, G2, G3, I1, I2; adds `ClaimCell` | -8 | chunk 1 |
| 3. Hot source | C1, C2, C3, C10 | -4 | chunk 1 |
| 4. Cache, vertex, edges | C4-C9; adds cells and views | -6 | chunk 2 (`ClaimCell`) |
| 5. Fan-in and lake | C14, G5, G6, G7; adds `OneShot` | -4 | chunk 4 (cells) |
| 6. Stage graph | G10-G15; adds `TakeAllStack` | -6 | chunk 1 |
| 7. io_uring | I3, I4, I5, I6, S6a | -5 | chunk 2 (the front) |
| 8. Windows engine | I7-I15 | -9 | chunk 7 (the shared slots and refusal API) |
| 9. Memory | M1, S1, S2 | -3 | chunk 1 |
| 10. Tick path | C16-C20, H1-H7 | -12 | chunk 4 (cells, `ClaimCell`) |
| 11. Lock the door | the ratchet becomes "no `HOT-FIX(` tag" | to 0 | chunks 2-10 |

The 57 tags are the 54 HOT and MIXED findings left once lane-source Part 2 has fixed G4 and without C15 (Java's
back-off, kept; M1 is tagged by a comment), plus three source-contract markers: S1, S2, S6a.

**Lane-source ordering.** Lane-source Part 1 changes `euhedral-data-structures` (`sync/`, `lib.rs`, `Cargo.toml`),
`euhedral-core/src/generics/` and `flow_control/vertex.rs`; Part 2 changes `euhedral-graph/src/join.rs`, `src/sync.rs`,
`lib.rs` and `Cargo.toml`. Chunk 1 touches all of those, so every chunk starts after both parts have merged. Parts 3
and 4 change `euhedral-graph/src/lanes.rs`, `src/lane_source/`, `src/testing.rs`, `src/lib.rs`, `tests/lane_*` and
`euhedral-gpu`; no chunk edits those files. Every part also edits `CHANGELOG.md`, `GLOSSARY.md` and crate READMEs; a
chunk's last task (docs) runs only once no open lane-source PR edits the same file.

**Gates per chunk**: `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test
--workspace`, the loom models of the chunk's new primitives (`RUSTFLAGS="--cfg loom"`), Miri on nightly for their unit
tests, and the Windows workflow for `euhedral-io`. Chunks 3 and 4 carry paired benchmarks (profile, hypothesis, paired
gate, keep or discard).

## 5. Risks

- **The invariant is load-bearing for memory safety.** `SourceCell::enter` is `unsafe`; a source driven from two
  threads at once without a lattice is undefined behaviour in release builds. Mitigations: the glossary contract, the
  debug check in `SourceCell`, the interceptor's `wip` claim as the only lattice-side mechanism.
- **Lifecycle calls on a source take effect when its holder releases.** `clear`, kill and `complete` set atomics and
  may return before their effect when a worker holds the claim; the holder applies them before it lets go, or the next
  `request`/`pull` does, or `Drop` (Java's `drainRequests` and `onDrainFinished` protocol). Waiting instead would deadlock when the call comes from a frame the source's own `pull` runs.
- **Stale views (C6).** A reader that stops reading holds the generation it last saw, so a retired generation's
  downstreams are released late. Mitigations: workers drain at each cycle start and their parks are timed; threads that are not workers hold weak references only;
  a thread that exits drops its view; drop-counted handles in the remap tests.
- **Reclamation bugs** in the cells and the tag slots. Mitigations: drop-counting tests, Miri, one owner per mailbox.
- **I/O submission can now be refused** (I4, the Java mirror): `submit*` hands a request back when its ring is full, its
  tag slot is busy or the thread has no ring; callers handle it synchronously, as Java's do. Short-read rests that
  cannot be resubmitted end the operation with an error.
- **Long frees on a worker (M1)** hold that worker for up to about 350 ms when the device is busy; the pinned pool and
  per-frame stream or VMM frees keep them off per-frame paths, and the other workers keep serving.
- **Public API changes**: hot-source hooks take `Fn` (C1); `PipelineRunner::submit*` returns a one-shot (C14);
  `AsyncIo::submit*` may refuse (I4); sockets made outside the Windows engine need `register` (I11).
- **Windows slab rewrite (I7)** with the `OVERLAPPED`-to-tag arithmetic: the simulated tests run on every platform,
  the live ones only on the Windows workflow.
- **Performance**: chunks 3 and 4 carry paired benchmarks; a change that does not beat what it replaces on the
  per-frame path is redone, not merged.

## 6. Decisions and questions

Answers applied: ticks are hot and fixed last (chunk 10); `PipelineRunner::submit*` returns a lock-free one-shot
(C14); a lake keeps both of Java's modes, attached for life and idle detach (G7); long calls run on the worker
that needs them, with no blocking-work facility (principle 6, M1); `loom` and Miri on nightly are approved; every
place mirrors its Java counterpart (principle 7, §3.12); the hot source and receiver are Java's (S3 dropped, C15
kept); the source primitives and `Join` come from the lane-source plan (§3.11).

Settled by the user (2026-10-28): **I/O submission follows Java** (decision 6 superseded). Each thread's ring has
that thread as its only submitter and `submit` enters the kernel itself, so the submission side is single-owner without
the poll's added latency (I4, chunk 7). There are no open questions.
