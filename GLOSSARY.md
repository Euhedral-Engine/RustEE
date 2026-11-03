# RustEE glossary

For people and agents who build on RustEE. It says what RustEE does for you, how to use the lifecycles,
what the invariants give you, and what each term is, from the lowest layer up. Implementation-level terms
(internal types, mechanisms, counters, tuning constants) are in one glossary per crate, listed below. Paths
are relative to the repository root.

## The layers

A crate builds only on lower layers, and `euhedral-gpu/tests/layering.rs` checks every manifest. The layers
come from the crates' `Cargo.toml` files. The terms below follow this order. Inside `euhedral-core` they run
frames, sources and receivers, ingest sources, edges, the lattice, metrics.

| Layer | Crate | What it gives you | Implementation terms |
|---|---|---|---|
| 0 | [`euhedral-data-structures`](euhedral-data-structures/README.md) | Lock-free queues with safe role handles, claim cells, padded atomics, the lifecycle locks that keep locks off the frame path. | [glossary](euhedral-data-structures/GLOSSARY.md) |
| 0 | [`euhedral-hashing`](euhedral-hashing/README.md) | xxHash64 hashes for ids, routing keys and fingerprints, equal to the Java port's. | [glossary](euhedral-hashing/GLOSSARY.md) |
| 1 | [`euhedral-hardware-utils`](euhedral-hardware-utils/README.md) | The machine's topology, CPU sets, thread pinning and a monitor of CPU pressure. | [glossary](euhedral-hardware-utils/GLOSSARY.md) |
| 2 | [`euhedral-core`](euhedral-core/README.md) | Frames and their lifecycle, sources and receivers, the lattice that runs each frame on one pinned worker, ingest sources, executors, metrics. | [glossary](euhedral-core/GLOSSARY.md) |
| 3 | [`euhedral-graph`](euhedral-graph/README.md) | No-wait dataflow on the lattice: completions, in-order conclusion, lakes, device lanes, stage graphs. | [glossary](euhedral-graph/GLOSSARY.md) |
| 3 | [`euhedral-io`](euhedral-io/README.md) | File and socket I/O whose completions are frames. | [glossary](euhedral-io/GLOSSARY.md) |
| 4 | [`euhedral-memory`](euhedral-memory/README.md) | Planned memory: budgets, owners, claims, admission, arenas, tiers, a leak check. | [glossary](euhedral-memory/GLOSSARY.md) |
| 5 | [`euhedral-gpu`](euhedral-gpu/README.md) | GPU streams as lanes, markers, retirement as frames, kernels loaded at run time, recorded graphs. | [glossary](euhedral-gpu/GLOSSARY.md) |
| 6 | [`euhedral-benchmarks`](euhedral-benchmarks/README.md) | End-to-end benchmarks of the control plane. | [glossary](euhedral-benchmarks/GLOSSARY.md) |

## What RustEE handles for you

Each row is something RustEE does for every frame, source or operation. Do not rebuild it in your own code.

| RustEE does | So you do not |
|---|---|
| Checks `is_alive()` before `execute` and finishes a dead frame with `do_finally`. | Check your kill switch at the top of `execute`. (A kill that lands during `execute` is not seen by RustEE: check `is_alive()` where a long `execute` can stop.) |
| Catches a panic in `execute`, logs it, and finishes the frame with `do_finally_with_error`. Calls one finalizer per frame it ran. | Wrap `execute` in `catch_unwind`, or track whether a frame finished. |
| Runs the ordered frames of one routing hash one at a time, on one CPU, in the order they were offered. | Add sequence numbers, per-key locks or per-key queues to get per-key order. |
| Routes each frame by its routing hash, routing policy and placement policy to a socket, a core, a cache partition and an SMT lane, and first touches a core's queues on its own CPU. | Pin threads, pick cores or lay out NUMA memory. |
| Steals unordered frames onto idle workers, sizes batches from the measured service time and parks idle workers. | Write a load balancer, or spin and sleep in frame code. |
| Lets one worker at a time into a lattice source's `pull` and `request`. | Lock state that only `pull` and `request` touch. |
| Asks each source for frames in batches sized to the core's cache and CPU pressure (`request`). | Push frames nobody asked for, or buffer them without bound. |
| Returns a finished frame to its pool when the finalizer is the default one. | Keep your own free list of frames. |
| Follows the machine: effective CPUs (cgroup limits included), pressure and topology changes. It rebalances the workers when the effective topology changes. | Poll `/proc`, re-pin threads or resize the worker set. |
| Finds completions (device lanes, timers, file and socket operations) in the workers' own polls and runs the continuation as a frame. | Start a completion thread or a timer thread, use a channel, or block on a device or an I/O call. |
| Concludes `Sequencer` entries in the order they were offered. | Keep a reorder buffer. |
| Keeps a memory budget by location and owner, orders lower-priority holders to give memory back, and reports leaks. | Count bytes by hand. |
| Settles a device retirement and hands it to you as a frame. | Synchronize a stream or wait on an event. |

What you still do:
- Synchronize state that is shared outside one frame or one source (see the invariants).
- Handle `Err(frame)` from an `offer`: the sink did not take the frame, so you own it again.
- Complete your sources and wait for `is_drained()` before `ControlPlaneLattice::close`. A frame still in a
  worker's cache when its clone closes is dropped, and no finalizer runs for it.
- Drop a `QuantumRef` when nothing will depend on its quantum, and a `UnitGuard` when its unit ends.
- Keep a finalizer short. It runs on the worker that ran the frame.
- Take a lifecycle lock only for start, stop and rebalance state. A debug build aborts if one is taken
  inside a hot scope.

## Invariants

Each is kept by the code. What it gives you follows it.

1. **One thread handles a frame while it is in the lattice.** A frame is owned by whoever holds its `Box`.
   Queues, executors and pools move it and never share it, and the worker that runs `execute` runs the
   finalizer. *So frame-owned state is plain:* the fields of a frame need no atomics and no lock. Only state
   shared outside the frame needs synchronization: an `Arc` several frames hold, a kill switch, a counter
   that other threads read.
2. **One thread at a time is inside a lattice source's `pull` or `request`.** The interceptor's claim
   serializes them and orders one worker's exit before the next worker's entry. *So source-owned state is
   plain:* keep it in a [`SourceCell`](euhedral-core/src/generics/source_cell.rs), with no atomics and no
   lock. A lifecycle call from another thread (`complete`, a clear, a kill) sets an atomic, and the source
   applies it in its next `request` or `pull`.
3. **`request` is always called; `pull` may never be.** A source pushes all ready work in `request`, never
   more than the demand it has been asked for. *So no work may be reachable only through `pull`.*
4. **Ordered frames of one routing hash run one at a time, in offer order, on one CPU, and are never
   stolen.** *So state keyed by an id hash needs no lock* while only frames of that hash touch it.
5. **A routing policy and the ordered flag are binding.** Routing, work stealing and a worker's direct pull
   take a frame only where both let it run.
6. **The frame path of `euhedral-core` takes no lock.** Queues hand frames over; locks are for start, stop
   and rebalance. A pooled frame returns to its pool, so steady-state execution of pooled frames allocates
   nothing. A lock the lock audit lists on a hot path of another crate, and that is not replaced yet,
   carries a `HOT-FIX(<id>)` tag. A test counts the tags, and the count only goes down. See the
   [lock policy](euhedral-data-structures/GLOSSARY.md#lock-policy).
7. **No completion thread, no timer thread, no channel.** `Completions`, `AsyncIo` and the lane sources are
   `LatticeSource`s that the lattice's own polls drive, and a poll does not block. A long call runs on the
   worker that needs it, and the other workers keep serving other sources and steal unordered work. Calls
   that block are named: module loading and dropping a stream or a graph in `euhedral-gpu`, and open and
   rename on Windows, which run on the OS thread pool.
8. **A frame is finished by exactly one finalizer, except in the cases the frame lifecycle lists as
   dropped.** See [the frame lifecycle](#frame).

## Building blocks that replace hand-rolled synchronization

Compose these, and the structure enforces order or exclusivity.

| You need | Use | What it guarantees |
|---|---|---|
| Plain state inside a source | `SourceCell` | One `&mut` at a time, checked in every build. `try_enter` hands a refusal back, `enter` panics. |
| One receiver per source, told of completion once | `DownstreamSlot` | A second attach hears `on_error`; the receiver hears `on_complete` exactly once however attach and complete race; reads take no lock. |
| A claim that losers leave a note on | `ClaimCell` | One thread holds it. A thread that finds it held writes what it wants and leaves; the holder looks after it lets go. |
| Per-key order | An ordered frame (one id hash); one partition of a `QueueIngestSink`; a `Lake` sink picked by routing hash | Frames of one hash run in offer order. |
| Conclude in order what finished out of order | `Sequencer` | One drainer at a time concludes the ready prefix, on its own thread. |
| Join branches | The fan-in count | One `fetch_sub(1, AcqRel)` per branch; the branch that reads 1 runs the next step and sees every other branch's writes. |
| Hand work to the next stage | Offer the frame to a `FrameSink` from `do_finally` | The finalizer is the frame's last owner, so no one else touches the frame. |
| Hand a result back without waiting | `OutcomeSlot`, or an `AtomicPtr` to a boxed value | Published once with release, taken with acquire. |
| Shape a graph of nodes | `LatticeEdge`, `LatticeNode`, `LatticeVertex` | One downstream per edge, set once; a vertex sends each routing hash to one downstream. |
| Many producers, one consumer | `MpscQueue`, `QueueIngestSink` | A `Single` consumer role has one live handle. |
| A queue only one thread uses | `PlainQueue` | The queue API with no synchronization. |
| State only a lane's owner touches | A stream's submitter | `&mut` methods: the borrow checker is the lock. |
| Wait for a device, a timer or I/O | `Completions`, `LaneSource`, `AsyncIo` | The lattice's polls find the completion and emit a continuation frame. |
| Start and stop a group of work | A `Lake` unit, a kill switch | `complete_gracefully` ends once every unit ended; the kill switch stops every frame that shares it. |
| Lifecycle state | `LifecycleMutex`, `LifecycleRwLock` | A debug build aborts if one is taken on the frame path. |

## Lifecycles

### Frame

1. **Build.** `FrameHeader::new(id_hash)` makes an ordered frame that is always alive.
   `FrameHeader::with_kill_switch` shares a kill switch with the other frames of one piece of work.
   `randomize_hash(seed)` makes the frame unordered; `set_origin` and `set_routing_policy` bind it near a
   CPU.
2. **Offer.** Hand the `Box<dyn Frame>` to a source or sink (`QueueIngestSink::offer`, `Lake::offer`,
   `LatticeHotSource::accept`). A sink that cannot take it (completed, closed, a lake with no unit
   admitted) returns the same frame in `Err`. `LatticeHotSource::accept` has no `Err`: it finishes a frame
   it cannot push with `do_finally_with_error`.
3. **Route.** RustEE picks the socket, core, partition and SMT lane. Ordered frames of one routing hash
   queue in one partition, in offer order. Unordered frames may be stolen.
4. **Run.** One worker takes the frame, and its executor
   ([`AbstractExecutor::run`](euhedral-core/src/generics/executor.rs)):
   - skips `execute` for a frame that is not alive, and calls `do_finally`;
   - calls `execute`, then `do_finally` when it returned `Ok` or `FrameError::Cancelled`;
   - calls `do_finally_with_error(error)` when `execute` returned `FrameError::Failed` or panicked. The
     panic becomes `FrameError::Failed` with its message, and is logged.

   The liveness check is made once, before `execute`. `do_finally` cannot tell a dead frame from a
   finished one: ask `is_alive()`.
5. **Finish.** The finalizer takes `self: Box<Self>` on the same thread. It is the frame's last owner on
   that executor. Hand the frame on (offer it to a sink: that is how frames chain into a graph), `recycle`
   it (the default), or drop it. A finalizer that panics is logged, and the frame is gone.
6. **Never run.** A frame that is not delivered still gets `do_finally_with_error`, and its `execute` never
   ran:
   - a queue sink that is cleared, completed or dropped with frames queued, a `SingleUseSource` and a
     `PipelineRunner` give `FrameError::Cancelled`;
   - a hop that cannot deliver gives `FrameError::Failed`: an edge with no downstream, a vertex with no
     active downstream, a retired cache, a `LatticeHotSource` with no receiver.

   A `Completions` registration that comes too late gets `do_finally`. A frame is dropped with no
   finalizer when it is in a worker's cache as `ControlPlaneLattice::clear` runs (`clear` returns how many)
   or as its clone closes.

### Source

1. **Attach.** `add_downstream(receiver)` gives the source its one receiver. A lattice takes a source with
   `ControlPlaneLattice::add_upstream`, which starts the lattice and wraps the source in an
   `UpstreamInterceptor`.
2. **Run.** Workers call `request(demand)` (always) and `pull(consumer, stop, demand)` (sometimes), one
   worker at a time. The source pushes at most `demand` frames to the receiver, and `pull` stops before the
   first frame `stop` matches.
3. **Complete.** `complete()` may come from any thread. A source that uses a `DownstreamSlot` tells its
   receiver `on_complete` once. A completed handle is dropped from every worker's queue. A panic in `pull`
   or `request` is logged and completes the interceptor.
4. **Drop.** A source dropped with frames queued finishes them (see the frame lifecycle, step 6).

Your code keeps state that `pull` and `request` touch in a `SourceCell`, applies lifecycle calls there,
pushes no more than the demand, and pushes all ready work in `request`.

### Lattice

1. **Build.** `ControlPlaneLattice::new(config)` from a `LatticeConfig`: name, `allowed_cpus`, startup and
   shutdown timeouts (60 s by default) and the `CloneableObject` cloned onto each core.
2. **Placement.** `set_placement_policy` works only before the lattice starts: it fails with
   `LatticeError::Running` after, and `Closed` once closed.
3. **Start.** The first `add_upstream` (or `start`) starts it. It waits until the shard of every active
   socket is ready, or fails with `StartTimedOut`, then follows the monitor.
4. **Run.** When the effective topology changes, the lattice stops ingest, starts or rebalances the shards
   of active sockets, shuts down those of inactive sockets, then reopens ingest. A shard drains a clone
   before it remaps; a clone that does not drain within the shutdown timeout is closed and replaced, and
   what its cache holds is dropped. `is_drained()` says every shard has finished its work.
5. **Clear.** `clear(timeout, f)` stops ingest, drops what every worker cache holds, runs `f`, then reopens
   ingest. Pause your sources first.
6. **Close.** `close()`, also run on drop, is idempotent. It stops following the monitor, closes the shards
   once no frame can still route to them, and waits up to the shutdown timeout.

### Device lane

A lane is a `LaneSource`, and the source owns the lane's submitter.
1. **Deliver.** Any thread calls `LaneIntakes::deliver`, which only enqueues. A lane that completed or was
   dropped hands the work back in `Err(LaneClosed)`; end it yourself.
2. **Request.** Each `request(demand)` banks the demand, runs the continuations of the retirements that
   arrived, reads the lane's hardware state once, runs every ready item (released waiting items, then the
   lane's own intake, then the group's shared intake; there is no budget) and pushes at most the banked
   demand.
3. **Item.** `run` answers `Submitted { retire }`, `Finished`, `Wait(w)` or `Failed(e)`. After the lane
   marked the work, `then` delivers successors and queues frames; a retirement arrives in `retired`;
   `failed` ends an item the lane could not order, mark or arm. An item that is not alive (`is_alive`,
   asked when the item leaves an intake and before a waiting item's wait is asked about) is never run: the
   lane ends it with `skipped`.
4. **Pull.** `pull` hands over ready frames and submits nothing.
5. **Drop.** The device work ends first (`quiesce`), then the state closes, then the submitter goes.

### Stage graph and lake

1. **Admit.** `Lake::admit()` counts a unit in and returns a `UnitGuard`. It fails with `LakeClosed` once
   admission is closed. The first unit of a busy period attaches the lake's sinks.
2. **Start.** `StageGraph::start(quantum, kill, external)` binds the quantum and publishes its roots once
   its external edges arrived. A stage runs in `execute` and hands on to its successors in `do_finally`;
   the last arrival at a stage publishes it.
3. **Run.** Stages submit and return `Submitted`; they never wait. `StageError::Retry` runs the stage
   again.
4. **Kill.** Set the `kill` switch you passed to `start`. The quantum's remaining stage frames are skipped
   and its stages get `Stage::abandoned`. Device work already submitted runs to its end.
5. **Retire.** Once no frame of the quantum can run and its retirement boundary is reached, `retire`
   receives the quantum and its `QuantumOutcome` (`Completed`, `Failed` or `Killed`). Drop the `UnitGuard`
   there.
6. **Close.** `Lake::complete_gracefully()` closes admission. Once every unit ended, the sinks drain,
   complete and detach. A running unit may still `fork` another unit.

### Completions, `AsyncIo`, `Sequencer`

- **Completions.** `after`, `after_marker`, `at` and `register` may be called from any thread. A poll emits
  the continuations of what completed, in registration order. A continuation whose frame is dead is still
  emitted, so `pending()` drains to zero. Call `complete` only once `pending()` is zero.
- **`AsyncIo`.** A submission returns at once. It returns `Err` only when nothing was submitted, and the
  continuation is then dropped without running. Otherwise the continuation runs exactly once, with the
  buffer back, whether the operation succeeded or failed. Dropping an `AsyncIo` with operations in flight
  leaks their buffers.
- **`Sequencer`.** `offer` an entry in the order it must conclude, execute in any order, and have the
  thread that finishes an entry call `drain`. Abandon an entry that dies, or the order waits for it (a
  `Slot` abandons on drop).

### Memory and GPU

- **Memory owner.** `open_owner` gives an owner a priority and a scope. The owner claims bytes (`claim`) or
  has a plan admitted (`admit`, then `Reservation::take`). A `Claim` returns its bytes when dropped.
  `end_owner` and `end_scope` report what the owner still holds; a clean report is a clean end.
- **GPU stream.** `open_lanes` gives one submitter per stream, with one owner. `submit` enqueues. `mark` and
  `await_marker` order work across streams on the device. A retirement arrives as a frame (`retire_into`)
  or through `notify_retired` and `confirm_retired`, and the submitter travels with its continuation. A
  fault poisons the device for the process. Dropping a submitter waits for the device: drop it in the
  owning source's stop path, never in a frame.

## Terms, by layer

### Layer 0: queues, claims, locks, hashing

**queue** ([`queues::queue`](euhedral-data-structures/src/queues/queue.rs)): the lock-free FIFO
`Queue<T, P, C>` (`SpscQueue`, `MpscQueue`, `SpmcQueue`, `MpmcQueue`). It hands out role handles: a
`Single` role has one live handle taking `&mut self`, a `Multi` role's handle is `Clone`. Details:
[`euhedral-data-structures` glossary](euhedral-data-structures/GLOSSARY.md#queues).

**partitioned queue** ([`queues::partitioned`](euhedral-data-structures/src/queues/partitioned.rs)):
independent queues behind one handle pair, each on its own cache lines. Producers pick a partition (the
same seed always picks the same one); there is no ordering between partitions.

**ClaimCell, ClaimGuard** ([`ClaimCell`](euhedral-data-structures/src/sync/claim_cell.rs)): state that one
thread at a time claims with a compare-and-set that returns `None` instead of waiting. It is the lock-free
form of a queue sink's `draining` flag: a thread that finds the claim held records what it wants (a pending
request, a pending sweep, a pending clear) before it tries, and the holder looks for those records after it
releases (`release_seq_cst`, a sequentially consistent store and fence). The ingest queue's consumer and
`Completions`'s poll state are claims. Use a `SourceCell` where the claim guards state the holder mutates
inside a source.

**lifecycle lock** ([`LifecycleMutex`, `LifecycleRwLock`](euhedral-data-structures/src/sync/lifecycle.rs)):
the lock types for start, stop and rebalance state. They ignore poison, and in a debug build taking one
inside a hot scope prints its location and aborts the process. The workspace lint refuses the standard
lock types, `Condvar`, channels, parks, sleeps and spins elsewhere; the exceptions carry a reason tag (see
the [lock policy](euhedral-data-structures/GLOSSARY.md#lock-policy)).

**hot scope** ([`HotScope`](euhedral-data-structures/src/sync/lifecycle.rs)): a per-thread depth that marks
the code a worker or a producer runs per frame, pull, request, offer or completion. `enter()` opens a scope
for its guard's lifetime; `leave()` suspends the scopes for a lifecycle step a frame reaches. It does
nothing in a release build. Tests that provoke a violation on purpose install
`HotScope::install_violation_hook`.

**hashing** ([`euhedral-hashing`](euhedral-hashing/src/lib.rs)): xxHash64 for routing keys, ordered-chain
ids and vectors, identical to the Java hashers. Strings hash as UTF-16, `_pair` functions chain multi-part
keys, and `mix_with_zero_check` never returns zero, which is what `randomize_hash` mixes a seed with.
Details: [`euhedral-hashing` glossary](euhedral-hashing/GLOSSARY.md).

### Layer 1: the machine

**topology** ([`Topology`](euhedral-hardware-utils/src/topology/mod.rs)): the machine's logical CPUs,
cores, sockets and caches. Logical CPU ids are the OS ids and may be sparse; `cpu_span()` is the highest id
plus one.

**effective topology** ([`TopologyMapper`](euhedral-hardware-utils/src/topology/mapper.rs),
[`EffectiveSystemTopology`](euhedral-hardware-utils/src/topology/mapper.rs)): the part of the topology
usable now: the CPUs the monitor reports as effective, the topology's CPUs and the allowed CPUs,
intersected. A socket or core with no effective CPU is inactive, and nothing routes to it. Pinning and the
monitor: [`euhedral-hardware-utils` glossary](euhedral-hardware-utils/GLOSSARY.md).

**CPU set** ([`CpuSet`](euhedral-hardware-utils/src/util/cpu_set.rs)): an immutable set of logical CPU ids.
`LatticeConfig::allowed_cpus` is one.

### Layer 2: `euhedral-core`

#### Frames

**Frame** ([`Frame`](euhedral-core/src/frames/mod.rs)): a unit of work, like a kernel. It carries a
`FrameHeader`, does its work in `execute`, and is owned by whoever holds its `Box`, so it is never touched
by two threads at once. Stock frames: `RunnableFrame`, `CallbackFrame`, `ArrayFrame`, `CollectionFrame`,
`DummyFrame`, `BenchmarkFrame`.

**FrameHeader** ([`FrameHeader`](euhedral-core/src/frames/mod.rs)): the state every frame carries: its id
hash, routing hash, origin, routing policy and optional kill switch.

**id hash** ([`FrameHeader::id_hash`](euhedral-core/src/frames/mod.rs)): the frame's identity. A new
header's routing hash equals it, which makes the frame ordered. Give related frames one id hash.

**routing hash** ([`FrameHeader::routing_hash`](euhedral-core/src/frames/mod.rs)): the hash the lattice
routes a frame by: it picks the socket, the core, the cache partition and the SMT lane.
`randomize_hash(seed)` sets it to the id hash mixed with a seed, which makes the frame unordered for every
seed; `reset_hash` sets it back.

**ordered frame, ordered stream** ([`FrameHeader::is_ordered`](euhedral-core/src/frames/mod.rs)): a frame
whose routing hash equals its id hash. Ordered frames with one routing hash land in one partition of one
cache (the odd-hash SMT split included) and run in the order they were offered, on that cache's owning CPU.
Stealers never take them.

**unordered frame**: a frame whose routing hash was randomized. Unordered frames spread over the cores, and
idle workers may steal them, as far as their routing policy allows.

**origin** ([`FrameHeader::origin`](euhedral-core/src/frames/mod.rs)): the `CpuInfo` of the CPU a frame was
made on, or should stay near. A frame without an origin goes anywhere, whatever its routing policy.

**RoutingPolicy** ([`RoutingPolicy`](euhedral-core/src/flow_control/mod.rs)): how close to its origin a
frame must run: `Anywhere` (the default), `SocketLocal` (a CPU of the origin's socket) or `CacheLocal` (a
CPU of the origin's core, its SMT threads). It is binding, like the ordered flag: routing places a frame by
it, and work stealing and direct pulls take a frame only where it may run. If the origin's core is
inactive, `CacheLocal` falls back to a hashed core on the origin's socket; if the origin's socket is
inactive, `SocketLocal` falls back to anywhere.

**kill switch** ([`FrameHeader::with_kill_switch`](euhedral-core/src/frames/mod.rs)): an `Arc<AtomicBool>`
shared by the frames of one piece of work. `kill()` sets it; a frame whose switch is set is no longer
alive, and executors skip it and call `do_finally`.

**lifecycle events** ([`Frame`](euhedral-core/src/frames/mod.rs),
[`AbstractExecutor::run`](euhedral-core/src/generics/executor.rs)): what an executor calls on a frame:
`execute` if the frame is alive, then exactly one finalizer, `do_finally` or `do_finally_with_error`. Frames
chain into a graph through these events: a finalizer can hand a follow-up frame to a source. See
[the frame lifecycle](#frame).

**`do_finally`** ([`Frame::do_finally`](euhedral-core/src/frames/mod.rs)): the finalizer after `execute`
returned `Ok` or `FrameError::Cancelled`, or after a dead frame was skipped. It takes the frame by value
and decides where it goes next; the default calls `recycle`.

**`do_finally_with_error`** ([`Frame::do_finally_with_error`](euhedral-core/src/frames/mod.rs)): the
finalizer after `execute` failed or panicked, and for a frame that was never executed: one a sink cancelled
(`FrameError::Cancelled`) or a hop could not deliver (`FrameError::Failed`). The default calls `recycle`.

**FrameError** ([`FrameError`](euhedral-core/src/frames/mod.rs)): why `execute` stopped early, or why a
frame never ran: `Cancelled` or `Failed`. From `execute`, `Cancelled` finishes with `do_finally` and
`Failed` with `do_finally_with_error`.

**recycle, FrameManager** ([`FrameManager`](euhedral-core/src/impls/frame_manager.rs),
[`FrameFactory`](euhedral-core/src/impls/frame_factory.rs)): `recycle` returns a frame to its pool; the
default drops it. A `FrameManager` keeps finished frames for reuse (8192 by default; a frame returned to a
full or dropped pool is dropped) and refills them through its `FrameFactory`, so steady-state execution
allocates nothing. Only the manager's owner checks frames out (`&mut self`).

**PipelineFrame, fan out, fan in** ([`PipelineFrame`](euhedral-core/src/frames/pipeline.rs)): one run of a
typed, multi-stage pipeline, as a single frame that steps through its stages and offers itself to a sink
after each one. A `fan_out` stage gets its own routing hash and may run in parallel across chains; a
`fan_in` stage keeps the chain's id hash and stays ordered with the pipeline's other chains. The chain ends
as `Success`, `Filtered`, `Cancelled` or `Failed`.

#### Sources and receivers

**LatticeSource** ([`LatticeSource`](euhedral-core/src/generics/lattice.rs)): where frames come from: a
programmable stream the lattice's workers pull from or request from. Only one thread at a time is inside a
source's `pull` or `request`.

**`request`** ([`LatticeSource::request`](euhedral-core/src/generics/lattice.rs)): signals demand for more
frames, which the source pushes to its downstream receiver now or later. Demand accumulates; `u64::MAX`
means unbounded and `0` asks for nothing.

**`pull`** ([`LatticeSource::pull`](euhedral-core/src/generics/lattice.rs)): hands up to `demand` frames
straight to a consumer, without pushing, and stops before the first frame a `stop` predicate matches. Its
demand does not accumulate.

**demand**: how many frames a receiver has asked a source for. It flows up through the lattice while
frames flow down.

**LatticeReceiver, LatticeTerminal, LatticeInterceptor**
([`generics::lattice`](euhedral-core/src/generics/lattice.rs)): a receiver takes the frames a source pushes
and hears `on_complete` and `on_error`; a terminal is the bottom of a chain that attaches to a source; an
interceptor is a stage that is both a source and a receiver. A receiver's `serialize` hands an action to
whoever holds its source: [`euhedral-core` glossary](euhedral-core/GLOSSARY.md#sources).

**source invariant**: the lattice serializes a source's `pull` and `request`. One worker is inside either
at a time, and the interceptor's claim orders one worker's exit before the next worker's entry. State that
only those calls touch is plain, in a `SourceCell`, with no atomics and no lock. `request` is always
called; `pull` may never be.

**SourceCell, SourceGuard** ([`SourceCell`](euhedral-core/src/generics/source_cell.rs)): a source's plain
state under the source invariant. Entry takes a claim with one compare-and-set in every build, so two
guards never exist: `try_enter` hands a refusal back (a nested entry, or a direct call racing the lattice),
`enter` panics. Under the lattice the claim is always free. A source's lifecycle calls from other threads
set atomics, and the source applies them in `pull` or `request`.

**DownstreamSlot** ([`DownstreamSlot`](euhedral-core/src/generics/downstream_slot.rs)): the one receiver a
source pushes to, read without a lock. A second attach is told `on_error`; the receiver hears
`on_complete` exactly once however attach and completion race; the receiver's `Arc` stays in the slot until
the slot drops, and a `live` flag ends its use. Every method touches atomics only.

#### Ingest sources

**ingest sink** ([`IngestSink`](euhedral-core/src/ingest/mod.rs)): a ready-made `LatticeSource` that feeds
a lattice from outside it. Hand its `delegate()` to `ControlPlaneLattice::add_upstream`; `complete()`
finishes it at once and detaches it, and what is queued stays until cleared or dropped.

**QueueIngestSink** ([`QueueIngestSink`](euhedral-core/src/ingest/queue.rs)): an ingest sink over a
partitioned queue that any thread offers frames to. Frames of one partition are delivered in order. `offer`
hands the frame back once the sink has completed. `complete_gracefully` completes it once the lattice has
drained the queue; `clear` cancels the queued frames.

**ArrayIngestSink** ([`ArrayIngestSink`](euhedral-core/src/ingest/array.rs)): a fixed batch of frames,
delivered in order. It completes as soon as its last frame is taken.

**SingleUseSource** ([`SingleUseSource`](euhedral-core/src/ingest/single_use.rs)): a source of one frame.
`pull`, `request` and `complete` each try one compare-and-set, and the winner owns the frame. It completes
once the frame is taken; completed before that, it cancels the frame.

**PipelineRunner** ([`PipelineRunner`](euhedral-core/src/ingest/pipeline_runner.rs)): an ingest sink that
runs typed `PipelineFrame` chains. `run` and `submit` admit an input, and hand it back in `AdmissionClosed`
once admission is closed; `complete_gracefully` stops admission and completes once every accepted chain has
finished; `complete` kills the chains.

**OutcomeSlot** ([`OutcomeSlot`](euhedral-core/src/ingest/pipeline_runner.rs)): how a submitted chain
ended, as the submitter reads it. `PipelineRunner::submit` returns one. Nothing waits on it: the owner
checks `is_set` or `take` periodically. `take` is `None` until the chain ended, then the `Outcome` once. A
chain dropped unended publishes `Cancelled`.

**LatticeHotSource** ([`LatticeHotSource`](euhedral-core/src/flow_control/hot_source.rs)): a source its
owner pushes frames into by hand with `accept`, without waiting for demand. A frame accepted with no
receiver, or after completion, is finished with `do_finally_with_error`. Its optional hooks are `Fn`; it
reads its downstream with atomics and keeps every receiver it installed until it drops, so a pusher never
holds a freed one.

#### Edges

**LatticeEdge, LatticeNode, LatticeVertex** ([`edge`](euhedral-core/src/flow_control/edge.rs),
[`vertex`](euhedral-core/src/flow_control/vertex.rs)): the links of a lattice. Frames flow down an edge and
demand climbs up it; every node is built on an edge. An edge has at most one downstream, a receiver or a
node, set once. A second receiver hears `on_error`; one that attaches to a closed edge hears
`on_complete`. A vertex is a node that routes each frame to one of its downstreams (by default by routing
hash) and ingests sources. A frame that cannot be routed is finished with `do_finally_with_error`.
Internals: [`euhedral-core` glossary](euhedral-core/GLOSSARY.md#flow-control-what-workers-read).

#### The lattice

**lattice** ([`ControlPlaneLattice`](euhedral-core/src/control_plane/lattice.rs)): where frames enter the
control plane. It routes each frame to the shard of one active socket and keeps the shards in line with the
effective topology; it starts on its first `add_upstream`. See [the lattice lifecycle](#lattice). Shards,
fragments, workers and caches: [`euhedral-core` glossary](euhedral-core/GLOSSARY.md#the-control-plane).

**CloneableObject** ([`CloneableObject`](euhedral-core/src/generics/cloneable.rs)): what a lattice clones
onto each core. The object in the `LatticeConfig` is a template; the stock one is
[`BaseCloneableObject`](euhedral-core/src/impls/base_cloneable.rs), a fragment feeding an executor.

**executor** ([`AbstractExecutor`](euhedral-core/src/generics/executor.rs),
[`DefaultExecutor`](euhedral-core/src/impls/default_executor.rs)): the terminal that runs each frame
through its lifecycle. Attached to a source with `input`, it requests unbounded demand.

**placement policy, placement picker** ([`ControlPlaneLattice::set_placement_policy`](euhedral-core/src/control_plane/lattice.rs),
[`PlacementPicker`](euhedral-core/src/control_plane/placement.rs)): one callback that runs first on every
frame the lattice routes or a worker pulls, with the frame and the CPUs the lattice routes with, and sets
the frame's origin (cache-local); the routing then honours it. It can be set only before the lattice
starts. It runs from many threads at once: it is `Fn`, and must be re-entrant, take no lock, not allocate
and not block. The picker chooses between CPU sets A and B by a predicate and picks the nth CPU of the
chosen set within the lattice's by the frame's routing hash.

**drain** ([`LatticeVertex::set_drain`](euhedral-core/src/flow_control/vertex.rs),
[`ControlPlaneLattice::is_drained`](euhedral-core/src/control_plane/lattice.rs)): stop accepting work and
finish what is in flight. A draining edge or vertex ignores demand and makes new upstreams wait;
`is_drained` says a cache, clone, shard or lattice has finished its work. Taking items out of a queue
(`drain*`) is a different thing.

**work stealing** ([`CacheConfig::work_steal`](euhedral-core/src/config/cache_config.rs)): a worker that
ran nothing takes unordered frames from the front of another cache's partitions, visiting the lattice's
caches in turn. In each partition it stops at the first frame that is ordered or that its routing policy
keeps where it is. On by default.

**greedy mode** ([`FragmentConfig::polls`](euhedral-core/src/config/fragment_config.rs)): a fragment with a
max batch above 2 takes from its cache greedily, so it consumes without touching the producer side: a bias
to low latency, for short frames. A frame that blocks holds up the frames queued behind it in its
partition, and stealers pass the held partition over.

**poll mode** ([`FragmentConfig::POLL_MODE_MAX_BATCH`](euhedral-core/src/config/fragment_config.rs)): a
fragment with a max batch of 2 or less polls its cache: it takes one frame at a time and holds the
partition only while it takes the frame out. For slow or blocking work: idle workers steal the unordered
frames behind a blocked one. Ordered frames still run only on their owning CPU, one at a time and in order.

**max batch size** ([`FragmentConfig::max_batch_size`](euhedral-core/src/config/fragment_config.rs)): the
largest batch a worker grows to, 4096 by default. It also selects greedy or poll mode. The worker cycle,
idle parks and batch caps: [`euhedral-core` glossary](euhedral-core/GLOSSARY.md#the-worker-cycle).

**productive source** ([`euhedral-core` glossary](euhedral-core/GLOSSARY.md#flow-control-what-workers-read)): a source that, the last time a
worker took from it, gave that worker frames. A worker that finds every source empty parks for the unproductive park. A worker
may also park while some source looks productive to it: the Idle path parks the workers that the participation model says
are not needed, given the productive sources per worker; a worker that sits out still enters one handle every probe
interval (1 ms by default), which is the longest the sources go unentered while the first-ranked worker is held in a frame. A pull that finds nothing shows a source empty. A request that
pushes nothing shows nothing either way, so a source whose frames only arrive through `request` and whose `pull` hands
over nothing looks productive to a worker that never pulls it.

#### Metrics

**metrics, MetricsRegistry** ([`metrics`](euhedral-core/src/metrics/mod.rs),
[`MetricsRegistry`](euhedral-core/src/metrics/registry.rs)): what workers and caches measure about
themselves. A config with a registry (`FragmentConfig::with_metrics`, `LatticeConfig::with_metrics`) makes
each worker and cache register its meters there; without one nothing is recorded. The registry holds them
weakly; closing a fragment or cache takes its meters out. Meters, snapshots and histograms:
[`euhedral-core` glossary](euhedral-core/GLOSSARY.md#metrics).

**service time** ([`time_execute`](euhedral-core/src/metrics/mod.rs)): how long a frame's `execute` takes
to return: the body of the work, not `do_finally` and not the time from ingest to a CPU.
`AbstractExecutor::run` times every `execute` on a metered worker's thread into its
`.execution.serviceTime` summary; an executor that overrides `run` wraps its `execute` call in
`time_execute` to opt in. Upstream named a smoothed per-cycle form `.execution.latency`; RustEE does not
use that name.

### Layer 3: dataflow

#### Completions and I/O

**completion token, Ready, Pending** ([`Ready`](euhedral-graph/src/completions.rs),
[`Pending`](euhedral-graph/src/completions.rs)): what says an operation has completed. A `Ready` is one
cheap query (`is_ready`) that never blocks; a `Pending` is polled until it yields the continuation frame,
for a completion that carries a result. A device lane's marker and a timer's deadline are completion tokens
too.

**continuation** ([`Continuation`](euhedral-io/src/lib.rs), [`IoThen`](euhedral-io/src/lib.rs)): the frame
(or the closure that builds it) that carries on the work once an operation completed. The source emits it,
and the lattice runs it like any other frame.

**Completions** ([`Completions`](euhedral-graph/src/completions.rs)): a `LatticeSource` of continuation
frames behind completion tokens. The lattice's polls query each pending token once and emit the
continuations of those that completed, in registration order; nothing waits and there is no timer thread.
Not `Completion` of a pipeline chain (the owner notification in
[`frames::pipeline`](euhedral-core/src/frames/pipeline.rs)): that is an event on one chain, this is a
source of frames.

**AsyncIo** ([`AsyncIo`](euhedral-io/src/lib.rs)): a `LatticeSource` of continuations behind file and
socket operations. A submission returns at once; the lattice's polls reap the completions. io_uring on
Linux and Android; IoRing and completion ports on Windows. Engines:
[`euhedral-io` glossary](euhedral-io/GLOSSARY.md).

#### Ordering and joining

**Sequencer, entry, ticket, slot** ([`Sequencer`](euhedral-graph/src/sequencer.rs),
[`Entry`](euhedral-graph/src/sequencer.rs), [`Ticket`](euhedral-graph/src/sequencer.rs),
[`Slot`](euhedral-graph/src/sequencer.rs)): execute in any order, conclude in offer order. Entries are the
caller's own objects, offered from any thread to a growing queue and never refused; an entry that finished
calls `drain`, and one thread at a time concludes the prefix of ready entries, in order, on its own thread,
through a function that must not wait. A `Ticket` is an entry that carries a value, and a `Slot` its
holder's handle (completing it or dropping it). The readiness word and the drainer:
[`euhedral-graph` glossary](euhedral-graph/GLOSSARY.md#completions-and-sequencing).

**fan-in**: branches that meet. No type of its own: the object the branches share carries a `pending`
count and plain result fields, armed with one store of the branch count before the branches are published;
each branch writes its field and ends with one `fetch_sub(1, AcqRel)` where its frame ends (`do_finally`,
which runs once whether the frame finished or was killed), and the branch that reads 1 runs the next step.

#### Device lanes

**device lane, DeviceQueue** ([`DeviceQueue`](euhedral-graph/src/lanes.rs)): a device timeline that runs
its submissions in order and whose progress is one cheap read (`completed`). A Vulkan queue with a timeline
semaphore, a CUDA stream with events, a copy engine, or a fake
([`FakeDevice`](euhedral-graph/src/testing.rs)).

**marker** ([`Marker`](euhedral-graph/src/lanes.rs)): a queue ([`QueueId`](euhedral-graph/src/lanes.rs), an
opaque identity unique in the process), a lane and a value on its timeline: the lane's work up to that
value. A marker of another queue is never reached by a `LanePool` and is refused by a backend's await.
Copyable; querying it never blocks. A successor on another lane orders itself behind a marker on the
device.

**LanePool** ([`LanePool`](euhedral-graph/src/lanes.rs), [`LaneId`](euhedral-graph/src/lanes.rs)): the
device lanes shared by every graph of a process. `reached` says whether a marker's work has completed;
`choose` picks a stage's lane, continuing its predecessor's when allowed.

**lane source** ([`LaneSource`](euhedral-graph/src/lane_source/source.rs)): a device lane as one
`LatticeSource`. It owns the lane's submitter and per-lane state in a `SourceCell`, takes work from its own
intake and its lane group's shared intake, and pushes the frames its items made within the demand it has
banked. See [the device lane lifecycle](#device-lane) and the
[`euhedral-graph` glossary](euhedral-graph/GLOSSARY.md#lanes).

**lane item** ([`LaneWork`](euhedral-graph/src/lane_source/mod.rs)): one unit of work on a lane. It runs on
the lane's submitter, answers `Submitted`, `Finished`, `Wait` or `Failed`, delivers its successors after
the lane marked it, and may ask for a retirement whose continuation runs when the device is done. An item
can carry an order key. It may also say when it is no longer alive (`is_alive`): the lane then ends it with
`skipped` instead of running it.

#### Lakes and stage graphs

**lake, sink, unit, UnitGuard** ([`Lake`](euhedral-graph/src/lake.rs),
[`UnitGuard`](euhedral-graph/src/lake.rs)): the pool of ready work of one kind (stage frames, host work,
control), as several `QueueIngestSink`s of one partition each, every one an upstream source of the
lattice. `offer` puts a frame in the sink its routing hash picks, so frames of one hash keep their order
and different hashes spread over the sinks; a `Lake` is a `FrameSink`, which is what a `GraphRuntime`
publishes through. A lake accepts frames only while a **unit** (a quantum, a host task) is admitted:
`admit` counts one in and returns its `UnitGuard`, and dropping the guard terminates the unit.
`complete_gracefully` closes admission; once every unit terminated, the sinks drain what they hold,
complete and detach. A running unit may `fork` another unit while the lake drains. Idle detach and busy
periods: [`euhedral-graph` glossary](euhedral-graph/GLOSSARY.md#lakes).

**submission edge, completion edge, external edge**
([`EdgeKind`](euhedral-graph/src/stage_graph.rs), [`ExternalEdge`](euhedral-graph/src/stage_graph.rs)):
when a stage's successor may run. A submission edge is satisfied when the predecessor submitted (the
successor waits on the device); a completion edge when the predecessor's work completed, found by a
`Completions` poll. An external edge makes a stage of a new quantum depend on a stage of an earlier
quantum.

**stage** ([`Stage`](euhedral-graph/src/stage_graph.rs), [`StageSpec`](euhedral-graph/src/stage_graph.rs)):
one step of a stage graph, run once per quantum. It computes or submits and returns a
[`Submitted`](euhedral-graph/src/stage_graph.rs); it never waits. It may be ordered (its frames run in the
order they are published) and may name the lanes able to run its work.

**quantum** ([`StageGraph`](euhedral-graph/src/stage_graph.rs)): the unit of work a stage graph is started
for, carrying whatever the caller's stages need. It has a kill switch of its own and ends with a
[`QuantumOutcome`](euhedral-graph/src/stage_graph.rs): `Completed`, `Failed` or `Killed`. Retirement and
the retry backstop: [`euhedral-graph` glossary](euhedral-graph/GLOSSARY.md#stage-graphs).

**QuantumRef pin** ([`QuantumRef`](euhedral-graph/src/stage_graph.rs)): a handle from `start` that keeps a
quantum from retiring while any clone lives, so a later quantum's external edge can still use its results.
Drop it when nothing will depend on the quantum, and drop held refs on a kill or at the end of the stream,
or that quantum never retires.

### Layer 4: memory

**planner** ([`Planner`](euhedral-memory/src/planner.rs)): one process's accounting of memory by location
and owner. It decides; it never allocates and never waits. Budgets and owner totals are atomics, owner
slots are recycled through a lock-free queue, and nothing takes a lock.

**location** ([`LocationId`, `LocationKind`, `LocationSpec`](euhedral-memory/src/planner.rs)): a memory
space with a budget: a device's memory, pinned host memory, pageable host memory or another. Its capacity
is what this process may hold; its reserve is always left free.

**owner** ([`OwnerSpec`, `OwnerId`](euhedral-memory/src/planner.rs)): who holds memory: a static name, a
priority, a scope and an optional parent. An owner id is a slot and its generation, so a recycled slot is
a different owner.

**scope** ([`Scope`](euhedral-memory/src/planner.rs)): an owner's lifetime: `Process`, `Epoch`, `Plan` or
`Lease`. `end_scope` ends every owner of one at once.

**claim** ([`Claim`](euhedral-memory/src/planner.rs)): bytes accounted to an owner at a location. It is a
value: dropping it gives the bytes back, so it travels with the memory it accounts for.

**elastic holding** ([`Elastic`](euhedral-memory/src/planner.rs)): a claim that grows into free memory and
gives back on order: a cache, a buffer of slots that can spill, a lease.

**give-back order** ([`GiveBack`](euhedral-memory/src/planner.rs)): the planner's order to a lower-priority
elastic owner to free bytes for a higher-priority claim (`claim_or_reclaim`: lowest priority first, then
the biggest holder). Orders reach their owners as frames of the planner's `GiveBacks` `LatticeSource`,
which run the owner's handler. The planner calls nobody and waits for nothing.

**admission** ([`Planner::admit`](euhedral-memory/src/admission.rs)): placing a plan's whole working set
before it allocates: every item at its preferred location if it fits, else spilled items (lowest priority
first), else optional items cut (lowest priority first), else refused with a `Shortfall`. An admitted plan
holds a `Reservation`, from which its allocations `take` their claims. Tiered items, tiers, backends and
reapers: [`euhedral-memory` glossary](euhedral-memory/GLOSSARY.md).

**leak check** ([`Planner::end_owner`, `Planner::check_baseline`](euhedral-memory/src/planner.rs)): an
ended owner's live claims are leaks; a snapshot taken at a baseline must come back.

**arena** ([`SlotArena`, `ScratchArena`](euhedral-memory/src/arena.rs)): plan once, allocate once, reuse.
One planned allocation cut into slots or power-of-two pieces, leased and returned through lock-free free
lists, so the steady state makes no allocation or free. `BackendCalls` counts the ones that do.

**resident** ([`ResidentId`, `Mover`](euhedral-memory/src/tiers.rs)): one slot's worth of an owner's data,
kept in the fastest tier its priority earns and moved between tiers by copies. `with` gives its address
while it is not moving.

**move** ([`Mover::move_to`](euhedral-memory/src/tiers.rs), [`Copier`](euhedral-memory/src/tiers.rs)): an
asynchronous copy of a resident to another tier: the destination slot is claimed first, the source
released once the copy retired. Spills (to make room or on a give-back order) and bring-backs are moves the
mover plans itself. Completions are frames of the `Moves` source.

### Layer 5: the GPU

**GPU stream** ([`Stream`](euhedral-gpu/src/api.rs), [`CudaStream`](euhedral-gpu/src/cuda/stream.rs)): an
ordered queue of device work and a lane of a `LanePool`, split in two. Its **queue** (`DeviceQueue`, shared
and `Sync`) is what the pool and `Completions` hold. Its **submitter** (`Stream`) is `Send`, not `Sync`,
and has one owner: `submit`, `mark`, `await_marker`, `notify_retired`, `confirm_retired` and a graph's
`replay` take `&mut`, so ownership replaces locking. `submit` runs a closure that enqueues launches,
copies and memsets; one submission's operations are contiguous on the stream. Work for a stream from
another thread goes to its owner, or to another lane. Marker rings, spans and kept values:
[`euhedral-gpu` glossary](euhedral-gpu/GLOSSARY.md).

**retirement** ([`notify_retired`, `confirm_retired`, `RetireTicket`](euhedral-gpu/src/api.rs),
[`retire_into`](euhedral-gpu/src/retire.rs)): the device reaching a point after the work submitted so far.
The owning source learns of it in `poll_retired` (in its `request` or `pull`) and tells the listeners, in
order; `confirm_retired` settles the ticket (a submitter method) and returns the device failure, or none;
`retire_into` makes the retirement a frame of a `Completions`, and the submitter travels with its
continuation. A stage finishing its submission is never completion. The callback and its queue:
[`euhedral-gpu` glossary](euhedral-gpu/GLOSSARY.md).

**poisoned device** ([`Poison`](euhedral-gpu/src/poison.rs), [`GpuError::Poisoned`](euhedral-gpu/src/error.rs)):
a device that faulted (an illegal address, a launch failure). The first fault is kept; later submissions
fail fast with it, every marker reads as reached so no continuation is stranded, and every retirement that
ends in a device fault is confirmed as `Poisoned(cause)`, with `cause` equal to `Gpu::poisoned()`. A fault
is sticky for the process.

**module** ([`Module`](euhedral-gpu/src/api.rs), [`ModuleCache`](euhedral-gpu/src/module_cache.rs)):
kernels the application brings, loaded at run time by `from_source` (compiled with the run-time compiler,
cached under a hash of the source, options, headers, include paths, target and compiler version) or
`from_image` (PTX, cubin or fatbin). Loading blocks: it is a build-time call.

**kernel arguments** ([`Args`, `KernelArg`](euhedral-gpu/src/args.rs)): a typed builder that packs a
launch's arguments, each at its own alignment. Where the driver reports the kernel's parameter sizes, a
launch whose arguments do not fit is refused before the driver sees it.

**sequence** ([`Sequence`](euhedral-gpu/src/api.rs)): a recurring set of submissions over fixed streams,
written once as a closure and run in three passes: record (for real, and captured), probe (its fingerprint
only) and update (new argument values, launch *n* to launch *n*).

**graph** ([`Graph`](euhedral-gpu/src/api.rs), [`CudaGraph`](euhedral-gpu/src/cuda/graph.rs)): a recorded
sequence. `replay` launches it as one submission on any stream; `Gpu::update` sets new argument values in
place; `Gpu::rerecord` records it again, in place when the driver can update the executable and otherwise
as a new one. A sequence of another shape is `GpuError::ShapeChanged`.

**stream priority** ([`StreamPriority`](euhedral-gpu/src/api.rs)): how the device schedules a stream's
waiting work against other streams'. `Normal` is the driver's default (0); `High` is the context's greatest
priority (numerically the lowest value, from `cuCtxGetStreamPriorityRange`). A device property: the driver
decides what it does with the hint. Set per lane in `StreamOptions`.
