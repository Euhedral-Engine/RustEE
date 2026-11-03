# euhedral-core glossary

The implementation-level terms of [`euhedral-core`](README.md): internal types, mechanisms, counters and
policies. What a user builds on, and how to use it, is in the [top-level glossary](../GLOSSARY.md). Paths
are relative to this crate. Entries run from the lowest layer of the crate to the highest: what a source
meets, what workers read, the control plane, the worker cycle, metrics.

## Sources

**serialize** ([`LatticeReceiver::serialize`](src/generics/lattice.rs)): runs an action when no `pull` or
`request` of the receiver's source is in progress, without waiting: at once if none is, otherwise on the
worker that holds the source, once that worker has let go of its `pull` or `request`. Actions run one at a
time, in order, each exactly once. The interceptor uses it to release the source it holds. The default
runs the action at once. A source's own lifecycle calls (`complete`, a clear, a kill) do not use it: they
set atomics, and the source applies them in its next `request` or `pull`.

## Flow control: what workers read

**cell, view** ([`Cell`](src/flow_control/views.rs)): a slot of lattice state that changes only when the
lattice is rebuilt (a vertex's routing generation, an edge's parent, a CPU's cache). The writer keeps the
current value under a lifecycle lock and sends it, through a queue, to the mailbox of every registered
reader; a reader drains its mailbox in one batch at the start of each cycle and applies the values to a
view only it reads. A frame reads the view: no lock, no atomic read-modify-write, no reference count. The
last holder of a replaced value drops it. A worker (or a thread that registers) holds the values
themselves; any other thread that reads gets a light mailbox on its first read and holds a weak reference,
which it upgrades for the length of a call, so an idle thread never keeps a replaced value alive.

**routing generation** ([`RoutingState`](src/flow_control/vertex.rs)): one immutable version of a vertex's
routing table. A push routes with the generation its worker's view holds. A remap publishes the next
generation and lets go of the old one; the last holder of a generation (the store or a worker's view)
drops it, and the drop closes the downstreams no later generation routes to. `with_routing_state` reads
the generation the calling thread's view holds.

**upstream, UpstreamHandle, UpstreamInterceptor**
([`UpstreamHandle`](src/flow_control/upstream_queue.rs),
[`UpstreamInterceptor`](src/flow_control/vertex.rs)): a source fed into the lattice, as the workers see it.
A worker services a handle only while it holds the handle's claim, a counter (`wip`) moved from zero by
`fetch_add`, so the source sees one worker at a time. `try_acquire` returns a `Claim`, the proof that
`pull`, `request` and `release` require. The interceptor installs its source before it is registered and
lets it go, through `serialize`, when it finishes. A panic in the source's `pull` or `request` is logged
and completes the interceptor. A complete handle is dropped from every queue that comes across it.

**UpstreamRegistry, UpstreamQueue** ([`registry`](src/flow_control/registry.rs),
[`upstream_queue`](src/flow_control/upstream_queue.rs)): the registry holds a lattice's upstream handles
and its registered workers. Each worker has its own queue of handles and splits its demand across them
round-robin, in a shuffled order.

**productive handle, worker's view**: whether an upstream handle produced work the last time this worker
serviced it. Each worker has its own view of each handle, plain state in the worker's queue, and one service
of a handle (one acquisition) changes it by what that service showed:
- a new handle is productive until the worker finds it empty;
- a **pull** that finds frames shows it productive; a pull that finds none shows it unproductive, unless the last frame
  pushed through the handle during the call was routed (a frame that finds no downstream shows nothing);
- a **request** whose last push through the handle was routed (or that satisfies the thread's request) shows it productive.
  A request that routes nothing shows nothing, because a request has no empty result: the view stays as it was;
- a pull that stopped at a frame, and a service that could not be made (the handle was complete, had no
  demand, its lattice was closed or was draining, or its source panicked), leave the view as it was.

A frame routed through some other handle, or by another thread, is no evidence about this one. The decision tree
and idle timing use the count of productive handles per worker. A handle that a worker only requests, never
pulls (a worker on the Staged path), stays productive for that worker.

**contention** ([`UpstreamQueue`](src/flow_control/upstream_queue.rs),
[`ACQUIRE_CONTENTION_SCALE`](src/utils/average_flow.rs)): the share of handles a worker found locked by
another worker, as a decaying average scaled to `1_000_000`.

## The control plane

**shard** ([`ControlPlaneShard`](src/control_plane/shard.rs)): the control plane of one socket. It keeps a
clone of the lattice's `CloneableObject` on each effective core of the socket and routes the socket's
frames among them by hash (`CacheLocal` frames to their origin's core).

**fragment** ([`ControlPlaneFragment`](src/control_plane/fragment.rs)): the worker that runs one core's
frames, on a thread pinned to the core's first CPU. It takes frames from its cache and its upstreams and
pushes them to its executor.

**worker**: a fragment's pinned thread (or its SMT buddy's). It registers with the lattice's
`UpstreamRegistry` and runs cycles until stopped.

**ControlPlaneCache** ([`ControlPlaneCache`](src/control_plane/cache.rs)): a worker's core-local queue of
frames and the demand that keeps it filled. It is split into partitions by routing hash, sized to a share
of the core's L1 and L2, and first touched on its owner's CPU so its memory lands on that CPU's NUMA node.
A frame offered to a retired cache is finished with `do_finally_with_error`. What a cache still holds when
its clone closes, or when `ControlPlaneLattice::clear` runs, is dropped: no finalizer runs for it.

**partition** ([`ControlPlaneCache`](src/control_plane/cache.rs),
[`CacheConfig::partitions`](src/config/cache_config.rs)): one of a cache's independent queues (8 by
default). Frames of one ordered stream stay in one partition, in order. Each logical CPU owns all the
partitions of its cache; one consumer holds a partition at a time, and the others pass it over.

**SMT buddy, sibling cache** ([`ControlPlaneFragment::smt_buddy`](src/control_plane/fragment.rs),
[`ControlPlaneCache::pair_with`](src/control_plane/cache.rs)): with SMT enabled and two effective CPUs on a
core, the fragment runs a second worker, its SMT buddy, on the second CPU, with its own cache (the sibling
cache). The core's frames reach the first cache, which hands those with an odd routing hash to the sibling.

**CacheDirectory** ([`CacheDirectory`](src/control_plane/cache.rs)): a lattice's caches by CPU, which work
stealing walks, with the upstream registry and topology they share. Each slot is a cell holding a weak
reference: a cache publishes itself when built and withdraws when it closes.

**first touch** ([`CloneableObject::first_touch`](src/generics/cloneable.rs),
[`ControlPlaneCache::first_touch`](src/control_plane/cache.rs)): filling a clone's queues once from its own
CPU before it receives work (each partition three chunks over, with dummy frames), then clearing them, so
Linux places those pages on that CPU's NUMA node.

**rebalance**: what the lattice and its shards do when the effective topology changes: stop ingest, start,
drain or close clones, remap routing, then reopen ingest. A clone is closed only once no routing generation
can still send it frames.

## The worker cycle

**greedy** ([`ControlPlaneCache::greedy`](src/control_plane/cache.rs)): consume as fast as possible by
holding locks longer. A greedy take holds a cache partition while it runs the frames it takes from it.

**direct pull** ([`ControlPlaneCache::upstream_pull`](src/control_plane/cache.rs)): a worker taking frames
straight from its upstreams, bypassing the caches. It stops before the first ordered frame and the first
frame bound to another core, leaving those to the lattice's routing.

**decision tree** ([`control_plane::decision_tree`](src/control_plane/decision_tree.rs), private): each
worker's policy for a cycle: whether to sit it out, which execution path to take, and how large the next
batch is. It times a sample of frame bodies to estimate their cost: each window of 32 samples yields its
second-smallest, which replaces the estimate at once if it is cheap and only after two windows in a row if
it is expensive. A batch is sized to fill the path's work target (250 µs Direct, 8 ms Staged) at the
measured service time, as a power of two, moved at most a halving or a doubling from the last batch, within
`[2, batch cap]`.

**execution paths: Direct, Staged, Idle** ([`ExecutionPath`](src/control_plane/decision_tree.rs), private):
Direct runs frames straight from upstream; Staged requests a batch into the cache and runs it from there;
Idle sits the cycle out and parks. Direct is taken when the worker has no upstream handle, for a lone
worker, before the body cost is known, while there is a productive handle per worker, or for cheap bodies
at moderate contention (up to 85%); Staged otherwise. Idle is never chosen for the first-ranked or only
worker, before the body cost is known, or while there is a productive handle per worker; for the others,
it is chosen when no handle is productive or when a fitted participation model says so. A worker on the Idle path
enters no handle, so its view of the handles cannot change by itself: once per **probe interval** (an
[`IdlePolicy`](src/config/idle_policy.rs) setting, 1 ms by default) its cycle takes the Direct path instead, so it enters a
handle and sees whether it has work. Without the probe, the first-ranked worker, which never sits out, would be the only
one to enter any handle, and a frame that holds it would leave every source unserved while the other workers sat idle.

**probe interval** ([`IdlePolicy::probe_interval`](src/config/idle_policy.rs)): the longest a worker sits out the Idle
path without entering a handle. When it has passed, the worker's next cycle takes the Direct path (a pull of one handle, then a
request of the next if the pull ran nothing) and the clock starts again. Zero makes every cycle enter a handle; `Duration::MAX`
sits out for good.

**park, idle park, unproductive park** ([`IdlePolicy`](src/config/idle_policy.rs)): a worker with nothing
to do parks its thread. The idle park is how long it parks on the Idle path; the unproductive park is how
long it parks when none of its upstream handles produced work (zero disables it). Both default to 15 µs
and are tunable. A worker with no upstreams and an empty cache parks for a fixed 15 µs that is not
configurable.

**idle timing, adaptive vs fixed** ([`IdlePolicy`](src/config/idle_policy.rs),
[`IdleTimingFunction`](src/config/idle_timing.rs), [`FixedTiming`](src/config/idle_timing.rs)): what picks
the idle park and the contention half-life (1 ms by default). The default is adaptive: a fitted model
(`cache_local_bounded`) of the contention, productive handles per worker and body cost.
`IdlePolicy::fixed` always uses the same park. Use it for loops that need predictable timing, because the
adaptive timing is not built for that.

**batch cap** ([`ControlPlaneFragment::batch_cap`](src/control_plane/fragment.rs)): the largest batch a
worker currently runs. At no pressure it is the max batch size capped by its cache's frame quota (what the
cache's first chunks hold), at least two; it falls linearly toward two as its CPU's pressure rises to 1.

**cap factor** ([`ControlPlaneCache::cap_factor`](src/control_plane/cache.rs)): the share of its frame
quota a cache currently asks its upstreams for, in `[0.15, 1]`. It moves quickly down and slowly up as the
core's CPU pressure rises and falls.

## Metrics

**meter, metric name, prefix, tags** ([`Meter`](src/metrics/registry.rs),
[`metric_name`](src/metrics/mod.rs)): one measured value of one worker or cache. Its name is a prefix
(`euhedral` by default, a trailing `.` dropped, blank means the default) and a suffix such as
`.execution.serviceTime`; its tags are `core` (the clone's physical core) and `cpu` (the worker's CPU).
Meters are per CPU: a core's two SMT workers each have their own.

**gauge, counter, summary, flow** ([`MeterValue`](src/metrics/registry.rs)): a gauge reads a value now
(`.execution.inProgress`, `.cache.backlog`); a counter counts since the meter was made
(`.execution.frames`); a summary is a cumulative histogram of recorded values (`.execution.serviceTime`,
`.execution.throughput`, `.cache.capFactor`); a flow is a `FlowRecorder`'s current statistics, named after
the summary it sits beside (`.execution.serviceTime.flow`, `.execution.throughput.flow`).

**histogram, log2 buckets, percentile** ([`Histogram`](src/metrics/histogram.rs)): atomics counting,
summing and keeping the maximum of recorded values, and counting them in buckets: each octave
`[2^e, 2^(e+1))` from `2^-16` to `2^48` split in four. A `HistogramSnapshot` gives the mean, percentiles
within a bucket's width, and with `since` the values recorded between two snapshots.

**throughput** ([`ExecutionFlow`](src/control_plane/fragment.rs), private): a worker's smoothed rate of
frames run (frames/s), from a `FlowRecorder` of the frames of each productive cycle. The throughput summary
samples it at the end of each batch; a second recorder of each cycle's mean `execute` time gives the
service-time flow.

**FlowStats** ([`FlowStats`](src/metrics/flow.rs)): a `FlowRecorder`'s average, standard deviation, trend
and recent minimum and maximum, which a worker publishes through a sequence lock: it never waits, and
readers never see a half-written set.

**snapshot** ([`MetricsSnapshot`](src/metrics/registry.rs)): every registered meter, read without stopping
the workers (`MetricsRegistry::snapshot`, `ControlPlaneLattice::metrics`), with lookups by name, worker and
suffix.
