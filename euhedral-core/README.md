# euhedral-core

The RustEE runtime: frames, the lattice that runs them, and the sources that feed it. It is a port
of the upstream Java `euhedral-core` (Euhedral-Execution). See [`GLOSSARY.md`](../GLOSSARY.md) for
every term used here.

**Where it sits.** It depends on the other three crates of the workspace and on `log`:

- [`euhedral-hardware-utils`](../euhedral-hardware-utils) for the topology, pinned executors and
  the resource monitor;
- [`euhedral-data-structures`](../euhedral-data-structures) for the queues behind caches, upstream
  queues, ingest sinks and frame pools;
- [`euhedral-hashing`](../euhedral-hashing) for routing hashes.

Nothing in the workspace depends on it; applications do.

## How work moves

```text
Frames:  LatticeSource -> lattice -> shard (per socket) -> fragment (per core) -> executor -> Frame::execute
Demand:  LatticeSource <- lattice <- shard              <- fragment
```

A **frame** is a unit of work, like a kernel. A **`LatticeSource`** is a programmable stream of
frames: the lattice's workers pull frames from it or request them. Frames chain into a graph through
their **lifecycle events**: a frame's `do_finally` can hand the next frame to a source. The
**lattice** places a worker on each effective core, routes frames to cores by their routing hash
and routing policy, and lets idle workers steal unordered work.

## Modules

| Module | Main types |
|---|---|
| [`frames`](src/frames/mod.rs) | `Frame`, `FrameHeader`, `FrameError`; stock frames: `RunnableFrame`, `CallbackFrame`, `ArrayFrame`, `CollectionFrame`, `DummyFrame`, `BenchmarkFrame`; typed pipelines: `PipelineFrame`, `PipelineBuilder`, `Outcome`, `Status`. |
| [`generics`](src/generics/mod.rs) | The interfaces frames flow through: `LatticeSource`, `LatticeReceiver` (with `serialize`), `LatticeTerminal`, `LatticeInterceptor`, `FrameSink`, `FramePusher`; `SourceCell` and `SourceGuard` (a source's plain state), `DownstreamSlot` (a source's one receiver); `CloneableObject` (what a lattice clones onto each core) and `AbstractExecutor`. |
| [`ingest`](src/ingest/mod.rs) | Ready-made `LatticeSource`s behind the `IngestSink` trait: `QueueIngestSink`, `ArrayIngestSink`, `PipelineRunner` (with `PipelineControl`, `AdmissionClosed`); and `SingleUseSource`. |
| [`control_plane`](src/control_plane/mod.rs) | `ControlPlaneLattice` (with `LatticeError`, `UtilizationSource`), `ControlPlaneShard`, `ControlPlaneFragment`, `ControlPlaneCache`, `CacheDirectory`. |
| [`config`](src/config/mod.rs) | `LatticeConfig`, `FragmentConfig`, `CacheConfig`, `IdlePolicy` with `FixedTiming` and `IdleTimingFunction` (`TimingProvider`, `TimingModel`), `CloneConfig`, `CloneLivenessRegistry`. |
| [`flow_control`](src/flow_control/mod.rs) | `RoutingPolicy`; `LatticeEdge` and `LatticeNode`; `LatticeVertex` and its `RoutingState` generations (read under a count, `routing_state()` returns a `RouteGuard`); `UpstreamRegistry`, `UpstreamHandle`, `UpstreamQueue`; `LatticeHotSource`, `LatticeHotReceiver`. |
| [`impls`](src/impls/mod.rs) | `BaseCloneableObject` (the stock per-core object: a fragment feeding an executor), `DefaultExecutor`, and frame pooling: `FrameManager`, `FrameFactory`, `Recycler`. |
| [`metrics`](src/metrics/mod.rs) | `time_execute` (times an executor's `execute` call); `MetricsRegistry` and its `MetricsSnapshot` of `Meter`s (`MeterValue`: gauge, counter, summary, flow); `Histogram` and `HistogramSnapshot` (log2 buckets, percentiles); `FlowStats`; the metric names (`metric_name`, `DEFAULT_PREFIX`, the `*_SUFFIX` constants, `CORE_TAG`, `CPU_TAG`). |
| [`utils`](src/utils/mod.rs) | `math`, `spin_wait`, `StopWatch`, `AverageFlow`, `FlowRecorder`, `FlowContext`, `calibration`. |

## Key concepts

### Frames and their lifecycle

A frame implements [`Frame`](src/frames/mod.rs): a `FrameHeader` and usually `execute`. A frame is
owned by whoever holds its `Box`. Queues, executors and pools move it rather than share it, so a
frame needs no locking and is never touched by two threads at once.

An executor that receives a frame:

1. skips a frame that is not alive (its **kill switch** is set) and calls `do_finally`;
2. otherwise calls `execute`, then `do_finally` when it returns `Ok` or `FrameError::Cancelled`,
   and `do_finally_with_error` for any other error or a panic.

The finalizer takes the frame by value: it is the frame's last owner on that executor and decides
where it goes next. By default it calls `recycle`, which returns a pooled frame to its
`FrameManager` and drops any other frame. A finalizer that hands a follow-up frame to a source is
how frames chain into a graph.

The ingest sinks are `LatticeSource`s, so any executor can be attached to one without a lattice.
This runs a batch on the calling thread:

```rust
use euhedral_core::frames::{Frame, FrameError, FrameHeader};
use euhedral_core::generics::CloneableObject;
use euhedral_core::impls::DefaultExecutor;
use euhedral_core::ingest::{ArrayIngestSink, IngestSink};
use std::sync::{Arc, Mutex};

struct Square {
    header: FrameHeader,
    value: u64,
    results: Arc<Mutex<Vec<u64>>>,
}

impl Frame for Square {
    fn header(&self) -> &FrameHeader {
        &self.header
    }

    fn header_mut(&mut self) -> &mut FrameHeader {
        &mut self.header
    }

    fn execute(&mut self) -> Result<(), FrameError> {
        self.value *= self.value;
        Ok(())
    }

    // The lifecycle event after a successful execute: the frame's last owner decides what's next.
    fn do_finally(self: Box<Self>) {
        self.results.lock().unwrap().push(self.value);
    }
}

let results = Arc::new(Mutex::new(Vec::new()));
let sink = ArrayIngestSink::new((1..=3).map(|value| {
    Box::new(Square { header: FrameHeader::new(7), value, results: results.clone() })
        as Box<dyn Frame>
}));
// The executor's terminal asks for unbounded demand, and the sink pushes every frame to it.
Arc::new(DefaultExecutor::new()).input(sink.delegate());
assert_eq!(*results.lock().unwrap(), [1, 4, 9]);
assert!(sink.is_complete());
```

### Ordered and unordered frames, routing policy

Every header has an **id hash** and a **routing hash**. A new header's routing hash is its id hash,
which makes the frame **ordered**. Ordered frames with one routing hash land in one partition of one
core's cache and run in the order they were offered, on that cache's owning CPU; stealers never take
them. `randomize_hash(seed)` mixes a seed into the routing hash, which makes the frame unordered, so
frames spread over the cores and idle workers may steal them. `reset_hash` makes it ordered again.

The **routing policy** says how close to its **origin** (the `CpuInfo` it was made on) a frame must
run: `Anywhere` (the default), `SocketLocal` or `CacheLocal`. The policy and the ordered flag are
both binding. Routing places a frame by them, and stealing and a worker's direct pull take a frame
only where both let it run. A frame without an origin goes anywhere.

```rust
use euhedral_core::flow_control::RoutingPolicy;
use euhedral_core::frames::FrameHeader;
use euhedral_hardware_utils::topology::Topology;

// Ordered: frames sharing an id hash keep their order.
let ordered = FrameHeader::new(42);
assert!(ordered.is_ordered());

// Unordered: a seed spreads the frames out.
let mut unordered = FrameHeader::new(42);
unordered.randomize_hash(1);
assert!(!unordered.is_ordered());

// Bound to the core of CPU 0 (its SMT threads), wherever it was made.
let mut pinned = FrameHeader::new(42);
pinned.randomize_hash(2);
pinned.set_origin(Topology::system().cpu_info(0).copied());
pinned.set_routing_policy(RoutingPolicy::CacheLocal);
```

`CacheLocal` falls back to a hashed core on the origin's socket when the origin's core is
inactive, and `SocketLocal` to any socket when the origin's socket is inactive, so a frame is never
left without a worker. See the [changelog](../CHANGELOG.md) for the exact rules.

### Build and start a lattice, attach an ingest sink

A `ControlPlaneLattice` is built from a `LatticeConfig`: a name, the CPUs it may use
(`allowed_cpus`), startup and shutdown timeouts, and the `CloneableObject` it clones onto every
effective core (by default a `BaseCloneableObject`: a fragment feeding a `DefaultExecutor`). It
starts on its first `add_upstream` (or `start`): it waits until every active socket's shard is
ready, then follows the resource monitor and rebalances when the effective topology changes.

```rust,no_run
use euhedral_core::config::LatticeConfig;
use euhedral_core::control_plane::ControlPlaneLattice;
use euhedral_core::frames::{Frame, RunnableFrame};
use euhedral_core::impls::DefaultExecutor;
use euhedral_core::ingest::{IngestSink, QueueIngestSink};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

let mut config = LatticeConfig::with_executor(DefaultExecutor::new());
config.name = "example".into();
let lattice = ControlPlaneLattice::new(config).unwrap();

let sink = QueueIngestSink::new();
lattice.add_upstream(sink.delegate()).unwrap(); // starts the lattice

let done = Arc::new(AtomicU64::new(0));
for id in 0..1000 {
    let counter = done.clone();
    let mut frame = RunnableFrame::new(id, Arc::new(move || {
        counter.fetch_add(1, Ordering::Relaxed);
    }));
    frame.header_mut().randomize_hash(id + 1); // unordered: any worker may run it
    assert!(sink.offer(Box::new(frame)).is_ok());
}
while done.load(Ordering::Relaxed) < 1000 {
    std::thread::sleep(Duration::from_millis(1));
}
sink.complete_gracefully(); // completes once the lattice has drained the queue
lattice.close();
```

The ingest sinks, all ready-made `LatticeSource`s:

- **`QueueIngestSink`**: a partitioned queue any thread offers frames to (`offer`, `offer_seeded`,
  `offer_to`); frames of one partition are delivered in order. `complete_gracefully` completes it
  once the lattice has drained it; `complete` stops it at once and leaves what is queued. One drain
  holds the queue at a time, and none waits: a `request` that finds it held banks its demand and the
  holder pushes within it as it lets go. `clear` cancels the queued frames, through the holder if one
  is running.
- **`ArrayIngestSink`**: a fixed batch, delivered in order, complete once its last frame is taken.
- **`SingleUseSource`**: one frame.
- **`PipelineRunner`**: runs typed `PipelineFrame` chains (below).

`LatticeHotSource` is a simpler source its owner pushes into by hand, without waiting for demand. Its hooks are `Fn + Send + Sync`, and it takes no lock: the downstream is an atomic pointer to a receiver the source keeps until it drops.

### Graphs: frames chained through lifecycle events

A frame's finalizer may offer a follow-up frame to a sink, so each step is scheduled on its own:

```rust,no_run
use euhedral_core::config::LatticeConfig;
use euhedral_core::control_plane::ControlPlaneLattice;
use euhedral_core::frames::{Frame, FrameError, FrameHeader};
use euhedral_core::ingest::{IngestSink, QueueIngestSink};
use std::sync::Arc;
use std::sync::mpsc::{Sender, channel};

struct Step {
    header: FrameHeader,
    value: u64,
    remaining: u32,
    sink: Arc<QueueIngestSink>,
    done: Sender<u64>,
}

impl Frame for Step {
    fn header(&self) -> &FrameHeader {
        &self.header
    }

    fn header_mut(&mut self) -> &mut FrameHeader {
        &mut self.header
    }

    fn execute(&mut self) -> Result<(), FrameError> {
        self.value += 1;
        Ok(())
    }

    fn do_finally(mut self: Box<Self>) {
        if self.remaining == 0 {
            let _ = self.done.send(self.value);
            return;
        }
        // The edge of the graph: hand this frame on as the next step. It keeps its id hash, so
        // the steps of one chain stay ordered.
        self.remaining -= 1;
        let sink = self.sink.clone();
        if sink.offer(self).is_err() {
            // The sink has completed; the frame is dropped.
        }
    }
}

let lattice = ControlPlaneLattice::new(LatticeConfig::default()).unwrap();
let sink = Arc::new(QueueIngestSink::new());
lattice.add_upstream(sink.delegate()).unwrap();
let (done, results) = channel();
for chain in 0..4 {
    let step = Step { header: FrameHeader::new(chain), value: 0, remaining: 9, sink: sink.clone(), done: done.clone() };
    assert!(sink.offer(Box::new(step)).is_ok());
}
let finished: Vec<u64> = results.iter().take(4).collect();
assert_eq!(finished, [10; 4]);
sink.complete_gracefully();
lattice.close();
```

`PipelineRunner` packages this pattern for typed stages. A `fan_out` stage may run in parallel
across chains; a `fan_in` stage keeps the chain's id hash and stays ordered with the other chains
of its pipeline:

```rust,no_run
use euhedral_core::config::LatticeConfig;
use euhedral_core::control_plane::ControlPlaneLattice;
use euhedral_core::frames::{PipelineFrame, Status};
use euhedral_core::ingest::{IngestSink, PipelineRunner};

let lattice = ControlPlaneLattice::new(LatticeConfig::default()).unwrap();
let builder = PipelineFrame::<u32>::builder()
    .fan_out(|value| value * 2)
    .fan_in(|value| format!("result={value}"));
let mut runner = PipelineRunner::new(&builder, |text: String| println!("{text}"), false);
lattice.add_upstream(runner.delegate()).unwrap();

let Ok(outcome) = runner.submit(21) else { panic!("admission closed") };
assert_eq!(outcome.recv().unwrap().status(), Status::Success);
runner.complete_gracefully(); // stops admission; completes once every accepted chain finished
lattice.close();
```

### Inside the lattice: shards, fragments and caches

- The lattice routes each frame to the **shard** of one active socket (`SocketLocal` and tighter
  frames to their origin's socket, the rest by routing hash).
- A shard keeps a clone of the `CloneableObject` on each effective core of its socket and routes
  frames to a core by hash (`CacheLocal` frames to their origin's core).
- On each core, a **fragment** runs a worker thread pinned to the core's first CPU, and with SMT a
  second worker, its **SMT buddy**, on the second CPU. Each worker owns a **`ControlPlaneCache`**:
  a queue split into **partitions** by routing hash. The core's frames reach the first cache, which
  hands those with an odd routing hash to the buddy's cache.
- Clones are built, and their caches first touched, on their own CPU, so the queues' memory lands
  on that CPU's NUMA node.

### The worker cycle: execution paths, work stealing, idle timing

Each cycle a worker:

1. runs frames from its own cache, up to its batch;
2. picks an **execution path** with its decision tree: **Direct** (run frames straight from its
   upstreams), **Staged** (request a batch into the cache and run it from there next cycle) or
   **Idle** (sit the cycle out);
3. if that ran nothing, **steals** unordered frames that their policy lets leave, from the next
   cache of the lattice in turn;
4. if still nothing, parks: for the **idle park** on the Idle path, for the **unproductive park**
   when none of its upstreams produced work, and otherwise it yields.

Batches grow and shrink with the frames' measured service time, toward a work target per path,
within a **batch cap** that falls as the CPU's pressure rises. The cache's **cap factor** shrinks
what it asks its upstreams for in the same way.

### Greedy mode and poll mode

- **Greedy mode** (a max batch above 2, the default 4096) is for short frames. A greedy take
  (`ControlPlaneCache::greedy`) holds a partition while it runs the frames it takes, so the worker
  consumes them as fast as it can without touching the producer side: a bias to low latency. A
  frame that blocks holds up every frame queued behind it in its partition, and stealers pass the
  held partition over.
- **Poll mode** (`FragmentConfig::polls()`: a max batch of `POLL_MODE_MAX_BATCH`, 2, or less) is for
  slow or blocking work. A poll takes one frame at a time and holds its partition only while it
  takes the frame out, so idle workers can steal the unordered frames behind a blocked one. Ordered
  frames still run only on their owning CPU, one at a time and in order.

In RustEE, **drain** means something else: stop accepting work and finish what is in flight
(`set_drain_mode`, a draining edge or vertex, `is_drained`, a shard draining its clones during a
rebalance).

A process can run one lattice of each kind, each on its own CPUs. The idle and park timings are
tunable. The default `IdlePolicy` is adaptive: a fitted `IdleTimingFunction` picks the park from
the contention, the productive handles per worker and the frame cost. `IdlePolicy::fixed` always
parks for the same time; use a fixed idle timing for graphics loops, because the adaptive one is not
built for predictable timing.

```rust
use euhedral_core::config::{FragmentConfig, IdlePolicy, LatticeConfig};
use euhedral_core::impls::{BaseCloneableObject, DefaultExecutor};
use std::num::NonZeroU64;
use std::sync::Arc;
use std::time::Duration;

// A lattice for blocking work: poll mode, a fixed 15 µs idle park, on CPUs 2 and 3.
let fragment = FragmentConfig::default()
    .with_max_batch_size(NonZeroU64::new(FragmentConfig::POLL_MODE_MAX_BATCH).unwrap())
    .with_idle_policy(
        IdlePolicy::fixed(Duration::from_micros(15), IdlePolicy::DEFAULT_CONTENTION_HALF_LIFE)
            .unwrap(),
    );
assert!(fragment.polls());
let mut config = LatticeConfig::new(Arc::new(BaseCloneableObject::new(fragment, DefaultExecutor::new())));
config.name = "blocking".into();
config.allowed_cpus = [2, 3].into_iter().collect();
```

`CacheConfig` sets each cache's memory budget (a share of the core's L1 and L2, 70% by default),
partition count (8), chunk pooling, ring-walk reset threshold, and whether work stealing is on.

### Metrics

With a `MetricsRegistry` in the fragment config (or `LatticeConfig::with_metrics`), each worker
and each cache registers its meters, tagged with its `core` and `cpu`, named after a prefix
(`euhedral` by default), as upstream's Micrometer meters are. The meters are per CPU: a core's
two SMT workers and their caches each have their own (`HistogramSnapshot::merge` combines them).

| Name | Kind | What |
|---|---|---|
| `.execution.serviceTime` | summary, ns | each frame's body: how long its `execute` took to return (not `do_finally`) |
| `.execution.serviceTime.flow` | flow, ns | the current smoothed mean `execute` time of the worker's cycles |
| `.execution.throughput` | summary, frames/s | each batch's smoothed rate of frames run |
| `.execution.throughput.flow` | flow, frames/s | the current smoothed rate of frames run |
| `.execution.inProgress` | gauge | 1 while the worker runs frames, else 0 |
| `.execution.frames` | counter | frames the worker has run |
| `.cache.backlog` | gauge, frames | frames queued in the cache |
| `.cache.capFactor` | summary | the cache's cap factor at each utilization update |

**Service time, not latency.** At the user's direction RustEE departs from upstream here: upstream's
`.execution.latency` sampled a smoothed per-cycle service time at each batch end; RustEE has no
`latency` meter, and `.execution.serviceTime` records how long every frame's `execute` took to
return, so its percentiles are the frames'. It is `execute` only: not `do_finally`, and not the time
from ingest to a CPU. `AbstractExecutor::run` times it on a metered worker's thread; an executor that
overrides `run` wraps its `execute` call in `metrics::time_execute` to opt in.

`frames` and the two `.flow` meters are RustEE's own too; the rest are upstream's names. Each
worker keeps two `FlowRecorder`s (the mean `execute` time and the frames of each productive cycle) and
publishes their statistics after each productive cycle; the throughput summary samples the rate at
each batch end. Summaries are cumulative histograms with log2 buckets, so a snapshot gives means
and percentiles, and `HistogramSnapshot::since` a window's. Recording takes no lock and allocates
nothing; without a registry nothing is recorded and a cycle pays one untaken branch.

```rust,no_run
use euhedral_core::config::LatticeConfig;
use euhedral_core::control_plane::ControlPlaneLattice;
use euhedral_core::metrics::MetricsRegistry;

let registry = MetricsRegistry::new();
let lattice = ControlPlaneLattice::new(LatticeConfig::with_metrics("app", registry.clone())).unwrap();
// ... add upstreams and run frames ...
let snapshot = lattice.metrics().unwrap(); // the same as registry.snapshot()
for meter in snapshot.service_time_summaries() {
    let service_time = meter.summary().unwrap();
    println!("core {} cpu {}: mean {:.0} ns, p99 {:?} ns", meter.core, meter.cpu,
        service_time.mean(), service_time.percentile(0.99));
}
let frames: u64 = snapshot.frames_counters().filter_map(|meter| meter.counter()).sum();
println!("{frames} frames run");
lattice.close();
```

## Build and test

```sh
mise exec -- cargo test -p euhedral-core
```

Many tests start real lattices on pinned threads. They take turns (one test with pinned workers at
a time) and wait at most 10 s for workers, so a failure does not hang the suite. Some need
particular hardware and pass without checking anything when the host lacks it:

- the SMT and poll-mode tests need a core with two CPUs this process may run on (an SMT core);
- the `CacheLocal` lattice test needs two whole cores the process may run on.

Run them on an SMT machine with no CPU restrictions to exercise everything. In this README, the
three examples that start no lattice are compiled and run as doctests; the four that start one
are compiled only (`no_run`), because they start a worker on every allowed CPU.

Two measurements are `#[ignore]`d tests; run them in release:

```sh
mise exec -- cargo test --release -p euhedral-core --lib -- --ignored --nocapture metrics_
```

`metrics_cost_per_cycle` times a worker cycle's bookkeeping with metrics off and on;
`metrics_overhead_per_frame` runs empty frames on an SMT core's two workers with metrics off and
on, in greedy and in poll mode.
