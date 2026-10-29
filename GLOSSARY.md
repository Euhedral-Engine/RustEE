# RustEE glossary

The terms a user of RustEE meets, as the code names them, grouped by topic. Each links to the type
or module that defines it. Paths are relative to the repository root.

## Frames

**Frame** ([`Frame`](euhedral-core/src/frames/mod.rs)): a unit of work, like a kernel. It carries a
`FrameHeader`, does its work in `execute`, and is owned by whoever holds its `Box`, so it is never
touched by two threads at once.

**FrameHeader** ([`FrameHeader`](euhedral-core/src/frames/mod.rs)): the state every frame carries:
its id hash, routing hash, origin, routing policy and optional kill switch.

**id hash** ([`FrameHeader::id_hash`](euhedral-core/src/frames/mod.rs)): the frame's identity. A new
header's routing hash equals it, which makes the frame ordered.

**routing hash** ([`FrameHeader::routing_hash`](euhedral-core/src/frames/mod.rs)): the hash the
lattice routes a frame by: it picks the socket, the core, the cache partition and the SMT lane.
`randomize_hash(seed)` sets it to the id hash mixed with a seed; `reset_hash` sets it back.

**ordered frame, ordered stream** ([`FrameHeader::is_ordered`](euhedral-core/src/frames/mod.rs)): a
frame whose routing hash equals its id hash. Ordered frames with one routing hash land in one
partition of one cache (the odd-hash SMT split included) and run in the order they were offered, on
that cache's owning CPU. Stealers never take them.

**unordered frame**: a frame whose routing hash was randomized. Unordered frames spread over the
cores, and idle workers may steal them, as far as their routing policy allows.

**origin** ([`FrameHeader::origin`](euhedral-core/src/frames/mod.rs)): the `CpuInfo` of the CPU a
frame was made on, or should stay near. A frame without an origin goes anywhere, whatever its
routing policy.

**RoutingPolicy** ([`RoutingPolicy`](euhedral-core/src/flow_control/mod.rs)): how close to its
origin a frame must run: `Anywhere` (the default), `SocketLocal` (a CPU of the origin's socket) or
`CacheLocal` (a CPU of the origin's core, its SMT threads). It is binding, like the ordered flag:
routing places a frame by it, and work stealing and direct pulls take a frame only where it may run.
If the origin's core or socket is inactive, `CacheLocal` falls back to the origin's socket and
`SocketLocal` to anywhere.

**kill switch** ([`FrameHeader::with_kill_switch`](euhedral-core/src/frames/mod.rs)): an
`Arc<AtomicBool>` shared by the frames of one piece of work. `kill()` sets it; a frame whose switch
is set is no longer alive, and executors skip it and call `do_finally`.

**lifecycle events** ([`Frame`](euhedral-core/src/frames/mod.rs),
[`AbstractExecutor::run`](euhedral-core/src/generics/executor.rs)): what an executor calls on a
frame: `execute` if the frame is alive, then exactly one finalizer, `do_finally` or
`do_finally_with_error`. Frames chain into a graph through these events: a finalizer can hand a
follow-up frame to a source.

**`do_finally`** ([`Frame::do_finally`](euhedral-core/src/frames/mod.rs)): the finalizer after
`execute` returned `Ok` or `FrameError::Cancelled`, or after a dead frame was skipped. It takes the
frame by value and decides where it goes next; the default calls `recycle`.

**`do_finally_with_error`** ([`Frame::do_finally_with_error`](euhedral-core/src/frames/mod.rs)): the
finalizer after `execute` failed or panicked. The default calls `recycle`.

**FrameError** ([`FrameError`](euhedral-core/src/frames/mod.rs)): why `execute` stopped early:
`Cancelled` (finished with `do_finally`) or `Failed` (finished with `do_finally_with_error`).

**recycle, FrameManager** ([`FrameManager`](euhedral-core/src/impls/frame_manager.rs),
[`FrameFactory`](euhedral-core/src/impls/frame_factory.rs)): `recycle` returns a frame to its pool.
A `FrameManager` keeps finished frames for reuse and refills them through its `FrameFactory`, so
steady-state execution allocates nothing.

**PipelineFrame, fan out, fan in** ([`PipelineFrame`](euhedral-core/src/frames/pipeline.rs)): one
run of a typed, multi-stage pipeline, as a single frame that steps through its stages and offers
itself to a sink after each one. A `fan_out` stage gets its own routing hash and may run in parallel
across chains; a `fan_in` stage keeps the chain's id hash and stays ordered with the pipeline's
other chains.

## Sources and ingest

**LatticeSource** ([`LatticeSource`](euhedral-core/src/generics/lattice.rs)): where frames come
from: a programmable stream the lattice's workers pull from or request from. Only one thread at a
time is inside a source's `pull` or `request`.

**`request`** ([`LatticeSource::request`](euhedral-core/src/generics/lattice.rs)): signals demand for
more frames, which the source pushes to its downstream receiver now or later. Demand accumulates;
`u64::MAX` means unbounded and `0` asks for nothing.

**`pull`** ([`LatticeSource::pull`](euhedral-core/src/generics/lattice.rs)): hands up to `demand`
frames straight to a consumer, without pushing, and stops before the first frame a `stop`
predicate matches. Its demand does not accumulate.

**demand**: how many frames a receiver has asked a source for. It flows up through the lattice while
frames flow down.

**LatticeReceiver, LatticeTerminal, LatticeInterceptor**
([`generics::lattice`](euhedral-core/src/generics/lattice.rs)): a receiver takes the frames a source
pushes and hears `on_complete` and `on_error`; a terminal is the bottom of a chain that attaches to a
source; an interceptor is a stage that is both a source and a receiver.

**source invariant**: the lattice serializes a source's `pull` and `request`. One worker is inside
either at a time, and the interceptor's claim orders one worker's exit before the next worker's entry.
State that only those calls touch is plain, in a `SourceCell`, with no atomics and no lock. `request`
is always called; `pull` may never be.

**SourceCell, SourceGuard** ([`SourceCell`](euhedral-core/src/generics/source_cell.rs)): a source's
plain state under the source invariant. `enter` is `unsafe`, because the compiler cannot see the
lattice's guarantee; the caller is inside `pull` or `request`, or in an action `serialize` runs. In
debug builds a second entry, nested or from another thread, panics. Release builds carry no check.

**`serialize`** ([`LatticeReceiver::serialize`](euhedral-core/src/generics/lattice.rs)): runs an action
when no `pull` or `request` of the receiver's source is in progress, without waiting: at once if none
is, otherwise on the worker that holds the source, before that worker releases it. Actions run one at
a time, in order, each exactly once. A source's lifecycle calls (completion, release of what it
holds) reach its `SourceCell` through it. The default runs the action at once.

**DownstreamSlot** ([`DownstreamSlot`](euhedral-core/src/generics/downstream_slot.rs)): the one receiver
a source pushes to, read without a lock. A second attach is told `on_error`; the receiver hears
`on_complete` exactly once however attach and completion race; the receiver's `Arc` stays in the slot
until the slot drops, and a `live` flag ends its use. `serialize` goes through the attached receiver.

**SetOnce** ([`SetOnce`](euhedral-data-structures/src/sync/set_once.rs)): a cell installed at most once
by a compare-and-set and read with an acquire load; the value lives until the cell drops.

**ingest sink** ([`IngestSink`](euhedral-core/src/ingest/mod.rs)): a ready-made `LatticeSource` that
feeds a lattice from outside it. Hand its `delegate()` to `ControlPlaneLattice::add_upstream`;
`complete()` finishes it at once.

**QueueIngestSink** ([`QueueIngestSink`](euhedral-core/src/ingest/queue.rs)): an ingest sink over a
partitioned queue that any thread offers frames to. Frames of one partition are delivered in order;
`complete_gracefully` completes it once the lattice has drained the queue.

**ArrayIngestSink** ([`ArrayIngestSink`](euhedral-core/src/ingest/array.rs)): a fixed batch of
frames, delivered in order. It completes as soon as its last frame is taken.

**SingleUseSource** ([`SingleUseSource`](euhedral-core/src/ingest/single_use.rs)): a source of one
frame. It completes once the frame is taken; completed before that, it cancels the frame.

**Downstream** ([`ingest::downstream`](euhedral-core/src/ingest/downstream.rs), private): the one
receiver an ingest source pushes to, the demand it asked for, and the source's completion, which the
receiver hears of exactly once.

**PipelineRunner** ([`PipelineRunner`](euhedral-core/src/ingest/pipeline_runner.rs)): an ingest
sink that runs typed `PipelineFrame` chains. `run` and `submit` admit an input; `complete_gracefully`
stops admission and completes once every accepted chain has finished; `complete` kills the chains.

**LatticeHotSource** ([`LatticeHotSource`](euhedral-core/src/flow_control/hot_source.rs)): a source
its owner pushes frames into by hand with `accept`, without waiting for demand.

## The lattice

**lattice** ([`ControlPlaneLattice`](euhedral-core/src/control_plane/lattice.rs)): where frames
enter the control plane. It routes each frame to the shard of one active socket and keeps the
shards in line with the effective topology; it starts on its first `add_upstream`.

**shard** ([`ControlPlaneShard`](euhedral-core/src/control_plane/shard.rs)): the control plane of one
socket. It keeps a clone of the lattice's `CloneableObject` on each effective core of the socket and
routes the socket's frames among them by hash (`CacheLocal` frames to their origin's core).

**CloneableObject** ([`CloneableObject`](euhedral-core/src/generics/cloneable.rs)): what a lattice
clones onto each core. The object in the `LatticeConfig` is a template; the stock one is
[`BaseCloneableObject`](euhedral-core/src/impls/base_cloneable.rs), a fragment feeding an executor.

**fragment** ([`ControlPlaneFragment`](euhedral-core/src/control_plane/fragment.rs)): the worker
that runs one core's frames, on a thread pinned to the core's first CPU. It takes frames from its
cache and its upstreams and pushes them to its executor.

**worker**: a fragment's pinned thread (or its SMT buddy's). It registers with the lattice's
`UpstreamRegistry` and runs cycles until stopped.

**executor** ([`AbstractExecutor`](euhedral-core/src/generics/executor.rs),
[`DefaultExecutor`](euhedral-core/src/impls/default_executor.rs)): the terminal that runs each frame
through its lifecycle. Attached to a source with `input`, it requests unbounded demand.

**ControlPlaneCache** ([`ControlPlaneCache`](euhedral-core/src/control_plane/cache.rs)): a worker's
core-local queue of frames and the demand that keeps it filled. It is split into partitions by
routing hash, sized to a share of the core's L1 and L2, and first touched on its owner's CPU so its
memory lands on that CPU's NUMA node.

**partition** ([`ControlPlaneCache`](euhedral-core/src/control_plane/cache.rs),
[`CacheConfig::partitions`](euhedral-core/src/config/cache_config.rs)): one of a cache's independent
queues (8 by default). Frames of one ordered stream stay in one partition, in order. Each logical
CPU owns all the partitions of its cache; one consumer holds a partition at a time, and the others
pass it over.

**SMT buddy, sibling cache** ([`ControlPlaneFragment::smt_buddy`](euhedral-core/src/control_plane/fragment.rs),
[`ControlPlaneCache::pair_with`](euhedral-core/src/control_plane/cache.rs)): with SMT enabled and two
effective CPUs on a core, the fragment runs a second worker, its SMT buddy, on the second CPU, with
its own cache (the sibling cache). The core's frames reach the first cache, which hands those with
an odd routing hash to the sibling.

**CacheDirectory** ([`CacheDirectory`](euhedral-core/src/control_plane/cache.rs)): a lattice's caches
by CPU, which work stealing walks, with the upstream registry and topology they share.

**first touch** ([`CloneableObject::first_touch`](euhedral-core/src/generics/cloneable.rs),
[`ControlPlaneCache::first_touch`](euhedral-core/src/control_plane/cache.rs)): filling a clone's
queues once from its own CPU before it receives work, then clearing them, so Linux places those
pages on that CPU's NUMA node.

**rebalance**: what the lattice and its shards do when the effective topology changes: stop ingest,
start, drain or close clones, remap routing, then reopen ingest. A clone is closed only once no
routing generation can still send it frames.

**routing generation** ([`RoutingState`](euhedral-core/src/flow_control/vertex.rs)): one immutable
version of a vertex's routing table. A push routes with the generation current when it began, and
holding a generation keeps its downstreams open.

**LatticeEdge, LatticeNode, LatticeVertex** ([`edge`](euhedral-core/src/flow_control/edge.rs),
[`vertex`](euhedral-core/src/flow_control/vertex.rs)): the links of a lattice. Frames flow down an
edge and demand climbs up it; a vertex is a node that routes each frame to one of its downstreams
(by default by routing hash) and ingests sources.

**upstream, UpstreamHandle, UpstreamInterceptor**
([`UpstreamHandle`](euhedral-core/src/flow_control/upstream_queue.rs),
[`UpstreamInterceptor`](euhedral-core/src/flow_control/vertex.rs)): a source fed into the lattice,
as the workers see it. A worker services a handle only while it holds the handle's claim, a counter
(`wip`) moved from zero by `fetch_add`, so the source sees one worker at a time. `try_acquire` returns
a `Claim`, the proof that `pull`, `request` and `release` require. The interceptor installs its source
before it is registered and lets it go, through `serialize`, when it finishes.

**UpstreamRegistry, UpstreamQueue** ([`registry`](euhedral-core/src/flow_control/registry.rs),
[`upstream_queue`](euhedral-core/src/flow_control/upstream_queue.rs)): the registry holds a lattice's
upstream handles and its registered workers. Each worker has its own queue of handles and splits its
demand across them round-robin, in a shuffled order.

**productive handle**: an upstream handle that produced work the last time this worker serviced it.
The decision tree and idle timing use the count of productive handles per worker.

**contention** ([`UpstreamQueue`](euhedral-core/src/flow_control/upstream_queue.rs),
[`ACQUIRE_CONTENTION_SCALE`](euhedral-core/src/utils/average_flow.rs)): the share of handles a worker
found locked by another worker, as a decaying average scaled to `1_000_000`.

## Execution

**drain** ([`ControlPlaneCache::set_drain_mode`](euhedral-core/src/control_plane/cache.rs),
[`CloneableObject::set_drain_mode`](euhedral-core/src/generics/cloneable.rs),
[`LatticeVertex::set_drain`](euhedral-core/src/flow_control/vertex.rs),
[`ControlPlaneLattice::is_drained`](euhedral-core/src/control_plane/lattice.rs)): stop accepting
work and finish what is in flight. A draining edge or vertex ignores demand and makes new upstreams
wait; `is_drained` says a cache, clone, shard or lattice has finished its work; a shard drains its
clones in a rebalance. The queues' `drain*` methods in `euhedral-data-structures` are a different
thing: taking items out of a container.

**greedy** ([`ControlPlaneCache::greedy`](euhedral-core/src/control_plane/cache.rs)): consume as
fast as possible by holding locks longer. A greedy take holds a cache partition while it runs the
frames it takes from it.

**greedy mode** ([`ControlPlaneCache::greedy`](euhedral-core/src/control_plane/cache.rs),
[`FragmentConfig::polls`](euhedral-core/src/config/fragment_config.rs)): a fragment with a max batch
above 2 takes from its cache greedily, so it consumes without touching the producer side: a bias to
low latency, for short frames. A frame that blocks holds up the frames queued behind it in its
partition, and stealers pass the held partition over.

**poll mode** ([`ControlPlaneCache::poll`](euhedral-core/src/control_plane/cache.rs),
[`FragmentConfig::POLL_MODE_MAX_BATCH`](euhedral-core/src/config/fragment_config.rs)): a fragment
with a max batch of 2 or less polls its cache: it takes one frame at a time and holds the partition
only while it takes the frame out. For slow or blocking work: idle workers steal the unordered
frames behind a blocked one.

**max batch size** ([`FragmentConfig::max_batch_size`](euhedral-core/src/config/fragment_config.rs)):
the largest batch a worker grows to, 4096 by default. It also selects greedy or poll mode.

**batch cap** ([`ControlPlaneFragment::batch_cap`](euhedral-core/src/control_plane/fragment.rs)): the
largest batch a worker currently runs: the max batch size, capped by its cache's capacity, at least
two, and lowered as its CPU's pressure rises.

**cap factor** ([`ControlPlaneCache::cap_factor`](euhedral-core/src/control_plane/cache.rs)): the
share of its frame quota a cache currently asks its upstreams for, in `[0.15, 1]`. It moves quickly
down and slowly up as the core's CPU pressure rises and falls.

**work stealing** ([`ControlPlaneCache::work_steal`](euhedral-core/src/control_plane/cache.rs),
[`CacheConfig::work_steal`](euhedral-core/src/config/cache_config.rs)): a worker that ran nothing
takes unordered frames from the front of another cache's partitions, visiting the lattice's caches in
turn. In each partition it stops at the first frame that is ordered or that its routing policy keeps
where it is. `work_steal_polled` is the poll-mode form. On by default.

**direct pull** ([`ControlPlaneCache::upstream_pull`](euhedral-core/src/control_plane/cache.rs)): a
worker taking frames straight from its upstreams, bypassing the caches. It stops before the first
ordered frame and the first frame bound to another core, leaving those to the lattice's routing.

**decision tree** ([`control_plane::decision_tree`](euhedral-core/src/control_plane/decision_tree.rs),
private): each worker's policy for a cycle: whether to sit it out, which execution path to take, and
how large the next batch is. It times a sample of frame bodies to estimate their cost.

**execution paths: Direct, Staged, Idle**
([`ExecutionPath`](euhedral-core/src/control_plane/decision_tree.rs), private): Direct runs frames
straight from upstream; Staged requests a batch into the cache and runs it from there; Idle sits the
cycle out and parks. Direct is taken for a lone worker, before the body cost is known, while
there is a productive handle per worker, or for cheap bodies at moderate contention; Staged
otherwise. Idle is never chosen for the first-ranked or only worker; for the others, when no handle is
productive or when a fitted participation model says so.

**park, idle park, unproductive park** ([`IdlePolicy`](euhedral-core/src/config/idle_policy.rs)): a
worker with nothing to do parks its thread. The idle park is how long it parks on the Idle path; the
unproductive park is how long it parks when none of its upstream handles produced work (zero
disables it). Both default to 15 µs and are tunable.

**idle timing, adaptive vs fixed** ([`IdlePolicy`](euhedral-core/src/config/idle_policy.rs),
[`IdleTimingFunction`](euhedral-core/src/config/idle_timing.rs),
[`FixedTiming`](euhedral-core/src/config/idle_timing.rs)): what picks the idle park and the
contention half-life. The default is adaptive: a fitted model (`cache_local_bounded`) of the
contention, productive handles per worker and body cost. `IdlePolicy::fixed` always uses the same
park; prefer it for graphics loops, because the adaptive timing is not built for predictable timing.

## Dataflow

The terms of [`euhedral-graph`](euhedral-graph/README.md) and of [`euhedral-io`](euhedral-io/README.md)
(`AsyncIo`, ring, completion port, IoRing): work that is submitted, never waited for, and continues on a
completion.

**completion token, Ready, Pending** ([`Ready`](euhedral-graph/src/completions.rs),
[`Pending`](euhedral-graph/src/completions.rs)): what says an operation has completed. A `Ready` is
one cheap query (`is_ready`) that never blocks; a `Pending` is polled until it yields the
continuation frame, for a completion that carries a result. A device lane's marker and a timer's
deadline are completion tokens too.

**continuation** ([`Continuation`](euhedral-io/src/lib.rs),
[`IoThen`](euhedral-io/src/lib.rs)): the frame (or the closure that builds it) that carries
on the work once an operation completed. The source emits it, and the lattice runs it like any
other frame.

**Completions** ([`Completions`](euhedral-graph/src/completions.rs)): a `LatticeSource` of
continuation frames behind completion tokens. The lattice's polls query each pending token once and
emit the continuations of those that completed, in registration order; nothing waits and there is
no timer thread. Not `Completion` of a pipeline chain (the owner notification in
[`frames::pipeline`](euhedral-core/src/frames/pipeline.rs)): that is an event on one chain, this is
a source of frames.

**AsyncIo** ([`AsyncIo`](euhedral-io/src/lib.rs)): a `LatticeSource` of continuations behind
file and socket operations. A submission returns at once; the lattice's polls reap the completions.
io_uring on Linux and Android; IoRing and completion ports on Windows.

**ring** ([`uring`](euhedral-io/src/uring.rs), [`win`](euhedral-io/src/win/mod.rs)): an
io_uring (or Windows IoRing) submission and completion queue pair. Each submitting thread has its
own, so submitting never contends; an operation belongs to the ring of the thread that submitted it.

**completion port, IoRing** ([`win`](euhedral-io/src/win/mod.rs)): the two engines of
`AsyncIo` on Windows. A completion port collects the packets of overlapped calls and is polled with
a zero timeout; IoRing is Windows' kernel ring for file reads and writes (Windows 11), used where
the OS has one that is not emulated in user mode. Open and rename run on the OS thread pool and
post to the port.

**device lane, DeviceQueue** ([`DeviceQueue`](euhedral-graph/src/lanes.rs)): a device timeline that
runs its submissions in order and whose progress is one cheap read (`completed`). A Vulkan queue
with a timeline semaphore, a CUDA stream with events, a hardware codec engine, or a fake
([`FakeDevice`](euhedral-graph/src/testing.rs)).

**marker** ([`Marker`](euhedral-graph/src/lanes.rs)): a queue ([`QueueId`](euhedral-graph/src/lanes.rs), an opaque
identity unique in the process), a lane and a value on its timeline: the lane's work up to that value. A marker of
another queue is never reached by a `LanePool` and is refused by a backend's await. Copyable; querying it never blocks. A successor on another lane orders itself
behind a marker on the device.

**LanePool** ([`LanePool`](euhedral-graph/src/lanes.rs), [`LaneId`](euhedral-graph/src/lanes.rs)):
the device lanes shared by every graph of a process. `reached` says whether a marker's work has
completed; `choose` picks a stage's lane, continuing its predecessor's when allowed.

**submission edge, completion edge, external edge**
([`EdgeKind`](euhedral-graph/src/stage_graph.rs), [`ExternalEdge`](euhedral-graph/src/stage_graph.rs)): when a
stage's successor may run. A submission edge is satisfied when the predecessor submitted (the
successor waits on the device); a completion edge when the predecessor's work completed, found by a
`Completions` poll. An external edge makes a stage of a new quantum depend on a stage of an earlier
quantum.

**stage** ([`Stage`](euhedral-graph/src/stage_graph.rs), [`StageSpec`](euhedral-graph/src/stage_graph.rs)): one
step of a stage graph, run once per quantum. It computes or submits and returns a
[`Submitted`](euhedral-graph/src/stage_graph.rs); it never waits. It may be ordered (its frames run in
the order they are published) and may name the lanes able to run its work.

**quantum** ([`StageGraph`](euhedral-graph/src/stage_graph.rs)): the unit of work a stage graph is started
for, carrying whatever the caller's stages need. It has a kill switch of its own and ends with a
[`QuantumOutcome`](euhedral-graph/src/stage_graph.rs): `Completed`, `Failed` or `Killed`.

**live count** ([`StageGraph::live`](euhedral-graph/src/stage_graph.rs)): the quanta started and not yet
retired. Each quantum also counts its own published frames and armed completion edges; at zero no
stage can run again.

**retirement boundary, retirement** ([`StageGraph`](euhedral-graph/src/stage_graph.rs)): the latest marker
per lane of everything a quantum submitted, and every host completion no completion edge consumed.
A quantum retires, handing itself and its outcome to the graph's `retire` callback, once none of its
frames can run again and the boundary is reached. The boundary is armed as a `Completions` edge.

**QuantumRef pin** ([`QuantumRef`](euhedral-graph/src/stage_graph.rs)): a handle from `start` that keeps a
quantum from retiring while any clone lives, so a later quantum's external edge can still use its
results. Drop it when nothing will depend on the quantum.

**retry backstop** ([`GraphRuntime::retry_backstop`](euhedral-graph/src/stage_graph.rs),
[`DEFAULT_RETRY_BACKSTOP`](euhedral-graph/src/stage_graph.rs)): a stage that answers `StageError::Retry`
runs again when any lane of the pool made progress, or after this deadline (20 ms by default) for a
resource outside the pool. Set it longer than any one pass on your devices.

**Sequencer, slot, releaser** ([`Sequencer`](euhedral-graph/src/sequencer.rs),
[`Slot`](euhedral-graph/src/sequencer.rs)): execute in any order, release in admission order.
`register` takes a slot in admission order; a slot is completed with a value or abandoned (dropping
it abandons it). The **releaser** is the one thread at a time (a work-in-progress counter) that
passes the prefix of finished slots to `emit`, in order, on its own thread.

**Join, join part** ([`Join`](euhedral-graph/src/join.rs)): a fan-in that waits for no one. Each
part arrives with a value or is abandoned, from any thread, and the last arrival runs the
continuation once with every part's value (`None` for an abandoned part).

**lake, sink, unit, UnitGuard** ([`Lake`](euhedral-graph/src/lake.rs),
[`UnitGuard`](euhedral-graph/src/lake.rs)): the pool of ready work of one kind (stage frames, host
work, control), as several `QueueIngestSink`s of one partition each, every one an upstream source
of the lattice. `offer` puts a frame in the sink its routing hash picks, so frames of one hash keep
their order and different hashes spread over the sinks; a `Lake` is a `FrameSink`, which is what a
`GraphRuntime` publishes through. A lake accepts frames only while a **unit** (a quantum, a host
task) is admitted: `admit` counts one in and returns its `UnitGuard`, and dropping the guard
terminates the unit. `complete_gracefully` closes admission; once every unit terminated, the sinks
drain what they hold, complete and detach. A running unit may `fork` another unit while the lake
drains.

**busy period, idle watch** ([`Lake`](euhedral-graph/src/lake.rs)): the sinks of a lake are
attached to the lattice only while it has units. Beside the sinks a lake attaches an idle watch, a
source the workers poll that holds no frames; when no unit has been admitted for the lake's idle
release time, the watch detaches the sinks and itself, so the lattice has nothing left to poll.
The next `admit` attaches a fresh set. An idle release of `Duration::ZERO` keeps the sinks attached.

**latch** ([`Latch`](euhedral-graph/src/latch.rs), [`LatchSetter`](euhedral-graph/src/latch.rs),
[`LatchWaiter`](euhedral-graph/src/latch.rs)): a one-shot hand-off from a frame to the main thread.
`Latch::new()` makes the two ends; a frame sets the `LatchSetter`, and the main thread parks in
`LatchWaiter::wait` until it does. `wait` is the one blocking call in the crate, for `main` (or a
test) only; a setter dropped without setting makes the wait panic instead of hang.

## Metrics

**metrics, MetricsRegistry** ([`metrics`](euhedral-core/src/metrics/mod.rs),
[`MetricsRegistry`](euhedral-core/src/metrics/registry.rs)): what workers and caches measure about
themselves. A config with a registry (`FragmentConfig::with_metrics`,
`LatticeConfig::with_metrics`) makes each worker and cache register its meters there; without one
nothing is recorded. The registry holds them weakly; closing a fragment or cache takes its meters
out.

**meter, metric name, prefix, tags** ([`Meter`](euhedral-core/src/metrics/registry.rs),
[`metric_name`](euhedral-core/src/metrics/mod.rs)): one measured value of one worker or cache. Its
name is a prefix (`euhedral` by default, a trailing `.` dropped) and a suffix such as
`.execution.serviceTime`; its tags are `core` (the clone's physical core) and `cpu` (the worker's
CPU). Meters are per CPU: a core's two SMT workers each have their own.

**gauge, counter, summary, flow** ([`MeterValue`](euhedral-core/src/metrics/registry.rs)): a gauge
reads a value now (`.execution.inProgress`, `.cache.backlog`); a counter counts since the meter
was made (`.execution.frames`); a summary is a cumulative histogram of recorded values
(`.execution.serviceTime`, `.execution.throughput`, `.cache.capFactor`); a flow is a
`FlowRecorder`'s current statistics, named after the summary it sits beside
(`.execution.serviceTime.flow`, `.execution.throughput.flow`).

**histogram, log2 buckets, percentile** ([`Histogram`](euhedral-core/src/metrics/histogram.rs)):
atomics counting, summing and keeping the maximum of recorded values, and counting them in
buckets: each octave `[2^e, 2^(e+1))` from `2^-16` to `2^48` split in four. A `HistogramSnapshot`
gives the mean, percentiles within a bucket's width, and with `since` the values recorded between
two snapshots.

**service time** ([`time_execute`](euhedral-core/src/metrics/mod.rs)): how long a frame's `execute`
takes to return: the body of the work, not `do_finally` and not the time from ingest to a CPU.
`AbstractExecutor::run` times every `execute` on a metered worker's thread into its
`.execution.serviceTime` summary; an executor that overrides `run` wraps its `execute` call in
`time_execute` to opt in. Upstream named a smoothed per-cycle form `.execution.latency`; RustEE does
not use that name, at the user's direction.

**throughput** ([`ExecutionFlow`](euhedral-core/src/control_plane/fragment.rs), private): a worker's
smoothed rate of frames run (frames/s), from a `FlowRecorder` of the frames of each productive
cycle. The throughput summary samples it at the end of each batch; a second recorder of each
cycle's mean `execute` time gives the service-time flow.

**FlowStats** ([`FlowStats`](euhedral-core/src/metrics/flow.rs)): a `FlowRecorder`'s average,
standard deviation, trend and recent minimum and maximum, which a worker publishes through a
sequence lock: it never waits, and readers never see a half-written set.

**snapshot** ([`MetricsSnapshot`](euhedral-core/src/metrics/registry.rs)): every registered meter,
read without stopping the workers (`MetricsRegistry::snapshot`, `ControlPlaneLattice::metrics`),
with lookups by name, worker and suffix.

## Hardware

**topology** ([`Topology`](euhedral-hardware-utils/src/topology/mod.rs)): the machine's logical
CPUs, cores, sockets and caches. Logical CPU ids are the OS ids and may be sparse; `cpu_span()` is
the highest id plus one.

**effective topology** ([`TopologyMapper`](euhedral-hardware-utils/src/topology/mapper.rs),
[`EffectiveSystemTopology`](euhedral-hardware-utils/src/topology/mapper.rs)): the part of the
topology usable now: the CPUs the monitor reports as effective, the topology's CPUs and the allowed
CPUs, intersected. A socket or core with no effective CPU is inactive, and nothing routes to it.

**CPU set** ([`CpuSet`](euhedral-hardware-utils/src/util/cpu_set.rs)): an immutable set of logical
CPU ids. `LatticeConfig::allowed_cpus` is one.

**P-core, E-core** ([`CoreInfo::p_core`](euhedral-hardware-utils/src/topology/mod.rs)): performance
and efficiency cores. A core is a P-core unless the platform classified it as an efficiency core.

**affinity, pinning** ([`affinity`](euhedral-hardware-utils/src/affinity/mod.rs)): binding a thread
to CPUs. `AffinityCapability` says what the platform can do: `Exact` masks (Linux, Windows), a
`LocalityHint` (macOS), or `Unsupported`.

**PinnedThreadExecutor** ([`PinnedThreadExecutor`](euhedral-hardware-utils/src/affinity/executor.rs)):
runs each command on a fresh thread pinned to one CPU, with at most one live executor per CPU. Each
fragment worker runs on one. Every handle is one use of the CPU's shared executor: it closes when
its last use ends (the last handle is dropped or `release`d), so fragments of different lattices can
share a CPU and each closes only its own worker task (`TaskInterrupt`).

**monitor** ([`ResourceMonitor`](euhedral-hardware-utils/src/monitor/mod.rs)): a background sampler
that polls CPU, memory, I/O and power signals (every 200 ms by default), publishes a smoothed
`HardwareUtilization`, and keeps a `TopologyMapper` in step. The lattice follows one.

**pressure** ([`HardwareUtilization`](euhedral-hardware-utils/src/monitor/utilization.rs),
[`monitor::pressure`](euhedral-hardware-utils/src/monitor/pressure.rs)): a per-CPU value in
`[0, 1]` combining scheduler stall, throttling, contention, memory, I/O and thermal signals, each
smoothed with a fast attack and a slow release. It lowers batch caps and cap factors.

## Data structures

**queue** ([`queues::queue`](euhedral-data-structures/src/queues/queue.rs)): the lock-free FIFO
`Queue<T, P, C>` (`SpscQueue`, `MpscQueue`, `SpmcQueue`, `MpmcQueue`). It hands out role handles: a
`Single` role has one live handle taking `&mut self`, a `Multi` role's handle is `Clone`.

**chunk** ([`queues::chunk`](euhedral-data-structures/src/queues/chunk.rs)): a power-of-two ring of
slots that a queue keeps its elements in. An unbounded queue links a fresh chunk when its ring is
full, and can pool retired chunks for reuse.

**partitioned queue** ([`queues::partitioned`](euhedral-data-structures/src/queues/partitioned.rs)):
independent queues behind one handle pair, each on its own cache lines. Producers pick a partition
(the same seed always picks the same one); there is no ordering between partitions.

**plain queue** ([`PlainQueue`](euhedral-data-structures/src/queues/plain_queue.rs)): a FIFO for one
thread with the same batch API as the concurrent queues and no synchronisation.

**padded atomics** ([`atomics`](euhedral-data-structures/src/atomics/mod.rs)): atomics, arrays and
striped adders with each value alone on a 128-byte block (`CachePadded`), so neighbours never share
a cache line.

## Hashing

**hashing** ([`euhedral-hashing`](euhedral-hashing/src/lib.rs)): xxHash64 for routing keys,
ordered-chain ids and vectors, identical to the Java hashers. Strings hash as UTF-16, `_pair`
functions chain multi-part keys, and `mix_with_zero_check` never returns zero, which is what
`randomize_hash` mixes a seed with.

## Memory

**planner** ([`Planner`](euhedral-memory/src/planner.rs)): one process's accounting of memory by
location and owner. It decides; it never allocates and never waits. Budgets and owner totals are
atomics, owner slots are recycled through a lock-free queue, and nothing takes a lock.

**location** ([`LocationId`, `LocationKind`, `LocationSpec`](euhedral-memory/src/planner.rs)): a
memory space with a budget: a device's memory, pinned host memory, pageable host memory or another.
Its capacity is what this process may hold; its reserve is always left free.

**owner** ([`OwnerSpec`, `OwnerId`](euhedral-memory/src/planner.rs)): who holds memory: a static
name, a priority, a scope and an optional parent. An owner id is a slot and its generation, so a
recycled slot is a different owner.

**scope** ([`Scope`](euhedral-memory/src/planner.rs)): an owner's lifetime: `Process`, `Epoch`,
`Plan` or `Lease`. `end_scope` ends every owner of one at once.

**claim** ([`Claim`](euhedral-memory/src/planner.rs)): bytes accounted to an owner at a location.
It is a value: dropping it gives the bytes back, so it travels with the memory it accounts for.

**elastic holding** ([`Elastic`](euhedral-memory/src/planner.rs)): a claim that grows into free
memory and gives back on order: a cache, a buffer of slots that can spill, a lease.

**give-back order** ([`GiveBack`](euhedral-memory/src/planner.rs)): the planner's order to a
lower-priority elastic owner to free bytes for a higher-priority claim
(`claim_or_reclaim`). Orders reach their owners as frames of the planner's `GiveBacks`
`LatticeSource`, which run the owner's handler.

**admission** ([`Planner::admit`](euhedral-memory/src/admission.rs)): placing a plan's whole
working set before it allocates: every item at its preferred location if it fits, else spilled
items (lowest priority first), else optional items cut (lowest priority first), else refused. An
admitted plan holds a `Reservation`, from which its allocations `take` their claims.

**tiered item** ([`PlanItem::tiered`](euhedral-memory/src/admission.rs)): an admission item of
equal units over an ordered list of locations, fastest first: its required units go to the first,
the rest to the first location with free room, then the next; units that fit nowhere are left out.

**leak check** ([`Planner::end_owner`, `Planner::check_baseline`](euhedral-memory/src/planner.rs)):
an ended owner's live claims are leaks; a snapshot taken at a baseline must come back.

**arena** ([`SlotArena`, `ScratchArena`](euhedral-memory/src/arena.rs)): plan once, allocate once,
reuse. One planned allocation cut into slots or power-of-two pieces, leased and returned through
lock-free free lists, so the steady state makes no allocation or free. `BackendCalls` counts the
ones that do.

**tier** ([`TierPool`](euhedral-memory/src/tiers.rs)): a location with one planned allocation cut
into equal slots. Free slots are an elastic holding of the pool's owner; a retired slot's budget is
back with the planner until `restore` takes it again.

**resident** ([`ResidentId`, `Mover`](euhedral-memory/src/tiers.rs)): one slot's worth of an
owner's data, kept in the fastest tier its priority earns and moved between tiers by copies. `with`
gives its address while it is not moving.

**move** ([`Mover::move_to`](euhedral-memory/src/tiers.rs), [`Copier`](euhedral-memory/src/tiers.rs)):
an asynchronous copy of a resident to another tier: the destination slot is claimed first, the
source released once the copy retired. Spills (to make room or on a give-back order) and
bring-backs are moves the mover plans itself. Completions are frames of the `Moves` source.

**backend** ([`Backend`](euhedral-memory/src/backend.rs), [`cuda`](euhedral-memory/src/cuda/mod.rs)):
what allocates at a location. The CUDA backends (feature `cuda`, the driver loaded at run time) are
`DeviceMemory`, the stream-ordered `StreamPool`, `Vmm` (exportable as a POSIX fd for Vulkan) and
`PinnedHost` (a capacity cache by size class).

**reaper** ([`Reaper`](euhedral-memory/src/backend.rs)): frees queued behind the device work that
last used the memory (a `Retired` test, such as a lane's marker), freed by `reap` on the owner's lane.

## GPU

**GPU stream** ([`Stream`](euhedral-gpu/src/api.rs), [`CudaStream`](euhedral-gpu/src/cuda/stream.rs)): an ordered
queue of device work and a lane of a `LanePool`, split in two. Its **queue** (`DeviceQueue`, shared and `Sync`) is what
the pool and `Completions` hold. Its **submitter** (`Stream`) is `Send`, not `Sync`, and has one owner: `submit`,
`mark`, `await_marker`, `notify_retired`, `confirm_retired` and a graph's `replay` take `&mut`, so ownership replaces
locking. `submit` runs a closure that enqueues launches, copies and memsets; one submission's operations are contiguous
on the stream. `mark` and `await_marker` order work across streams on the device. Work for a stream from another thread
goes to its owner, or to another lane.

**retirement** ([`notify_retired`, `confirm_retired`, `RetireTicket`](euhedral-gpu/src/api.rs),
[`retire_into`](euhedral-gpu/src/retire.rs)): the device reaching a point after the work submitted so far. The
driver's callback only pushes the outcome onto the stream's unbounded lock-free MPSC queue (no allocation, free or
lock); the owning source drains it with `poll_retired` (in its `request` or `pull`) and tells the listeners, in order;
`confirm_retired` settles the ticket (a submitter method) and returns the device failure, or none; `retire_into` makes the retirement a frame of a
`Completions`, and the submitter travels with its continuation. A stage finishing its submission is never completion.

**poisoned device** ([`Poison`](euhedral-gpu/src/poison.rs), [`GpuError::Poisoned`](euhedral-gpu/src/error.rs)): a
device that faulted (an illegal address, a launch failure). The first fault is kept; later submissions fail fast with
it, and every marker reads as reached so no continuation is stranded.

**module** ([`Module`](euhedral-gpu/src/api.rs), [`ModuleCache`](euhedral-gpu/src/module_cache.rs)): kernels the
application brings, loaded at run time by `from_source` (compiled with the run-time compiler, cached under a hash of
the source, options, headers, include paths, target and compiler version) or `from_image` (PTX, cubin or fatbin).
Loading blocks: it is a build-time call.

**kernel arguments** ([`Args`, `KernelArg`](euhedral-gpu/src/args.rs)): a typed builder that packs a launch's
arguments, each at its own alignment. A launch whose arguments do not fit the kernel's parameter sizes is refused
before the driver sees it.

**sequence** ([`Sequence`](euhedral-gpu/src/api.rs)): a recurring set of submissions over fixed streams, written once
as a closure and run in three passes: record (for real, and captured), probe (its fingerprint only) and update (new
argument values, launch *n* to launch *n*).

**graph** ([`Graph`](euhedral-gpu/src/api.rs), [`CudaGraph`](euhedral-gpu/src/cuda/graph.rs)): a recorded sequence.
`replay` launches it as one submission on any stream; `Gpu::update` sets new argument values in place;
`Gpu::rerecord` records it again, in place when the driver can update the executable and otherwise as a new one.

**shape and fingerprint** ([`ShapeHasher`, `Fingerprint`](euhedral-gpu/src/shape.rs)): the shape of a sequence is its
operations in order with their streams, order edges, kernels, geometry, argument layouts and copy parameters; the
fingerprint is 128 bits of it. A changed shape is `GpuError::ShapeChanged`.

**shadow stream** ([`CudaSequence`](euhedral-gpu/src/cuda/graph.rs)): the stream a recording captures a stream's work
on, so the real stream is never in capture mode and a failed capture never disturbs the work.
