# Changelog

Changes to the lattice's execution model, newest first. Each entry says what changed, why, how to use it, its exact
semantics and the tests that pin it.

## `AsyncIo` on Windows (2026-10-26)

**What.** `euhedral-graph`'s `AsyncIo` gets a Windows engine: IoRing for file reads and writes where the OS has a kernel
ring, and I/O completion ports for everything else (sockets, accept, connect, and files without a ring). Linux and
Android keep io_uring. The portable half of `io` (the public API, the `LatticeSource` behaviour, the frames a pull's
`stop` holds back) moved to `io/mod.rs`; the Linux engine and its tests are otherwise unchanged. The default feature is
now `async-io` (it was `io-uring`).

**Why.** MySphere's players run on Windows too, and `AsyncIo` was Linux-only.

**API additions, on every platform.**
- `RawFile`, `RawSock`, `OwnedFile` and `OwnedSock`: the platform's handles (descriptors on Linux and Android, `HANDLE` and
  `SOCKET` on Windows). Linux signatures are unchanged.
- `AsyncIo::open_path(path, OpenSpec, then)` with `OpenSpec { read, write, create: Create, unix_mode }`. `AsyncIo::open()`
  stays the constructor, so the plan's "`AsyncIo::open(path, ..)`" is `open_path`; `open_file(path, flags, mode, then)`
  stays on Linux and Android.
- `AsyncIoOptions { windows_ioring }` with `AsyncIo::open_with_options`; the option is ignored off Windows.

**Semantics on Windows.**
- Each operation ends through a poll: the "skip the port on success" mode is not set. An overlapped call that fails at
  once (no packet will come) ends with that error at the next poll.
- File reads and writes use rings only if the probe says the OS has a kernel ring (not `IORING_FEATURE_UM_EMULATION`) with
  both operations supported (`IORING_VERSION_3`, Windows 11 22H2 and later); otherwise both go through the port. A file keeps
  its route: IoRing refuses a handle that belongs to a completion port (`ERROR_INVALID_PARAMETER`, found by a live test), so
  a handle that joined the port stays on it and a handle a ring serves is never joined. The IoRing functions are resolved with `LoadLibraryExW` and `GetProcAddress`, never linked, so the
  library loads on Windows without them.
- Open and rename run on the OS thread pool (`TrySubmitThreadpoolCallback`) and post their result to the port, so they
  come out of the same poll. This is the one place the crate uses threads it did not create.
- A handle joins the engine's port when its first operation is submitted and must not belong to another port. A file handle
  opened without `FILE_FLAG_OVERLAPPED` is refused with `InvalidInput`.
- A full ring submission queue goes to the ring's backlog and is handed over by a later submission or poll, as on Linux.
  A flush builds at most 16 entries before it submits, and a submission never waits behind a submit.
- **Operation identity.** An operation's record lives in a slab and is never freed while the engine lives, so a late
  packet naming its `OVERLAPPED` can be read safely. A slot is idle, issuing, in flight or busy; its generation changes at
  every issue, and a ring entry carries slot and generation. A port packet carries only an address, so it is accepted for a
  slot that is in flight on the port with an `Internal` status that is not `STATUS_PENDING`. A poll leaves the packet of a
  slot whose call is still returning for the next poll.
- **Warning statuses.** A call that reports failure with a warning status (`ERROR_MORE_DATA`, `WSAEMSGSIZE`) still queues a
  packet; the real layer reads the `OVERLAPPED`'s status after such a call and treats a warning as "a packet follows".
- Results are decoded from the packet's `NTSTATUS` and byte count, not through the handle (which the caller may close). Socket
  statuses map to Winsock errors for the common cases (refused, reset, aborted, unreachable, timed out, message too long).
- A handle that skips the port on success, or a socket opened without overlapped I/O, is refused. A handle already on
  another port is refused too (`ERROR_INVALID_PARAMETER` for a handle this engine did not join): a handle serves one
  `AsyncIo`, ever.
- A slot stays `Issuing` until its call is made (or its ring entry or pool job is handed over): nothing that ends or fails
  operations touches a record being filled. A ring that cannot be read ends its operations and the entries behind it, and an
  operation sent to a broken ring ends with an error instead of waiting. A poll whose inbox holds only packets of returning
  calls also looks at the port. A packet is taken only if its count agrees with the `OVERLAPPED`'s `InternalHigh`; status and
  bytes are read from the `OVERLAPPED` as atomics. A read or receive that fails with a warning status keeps the bytes that
  arrived: the result is the error (`ERROR_MORE_DATA`, `WSAEMSGSIZE`) and `IoBuf::filled` is their count.
- Open and rename leave their result in a queue the poll drains (a job that panics or is dropped still ends the operation
  with an error). Writes go in pieces of at most 256 KiB, since a write that extends a file is synchronous on NTFS. Paths
  near the Win32 limit go as verbatim paths. A ring whose completions cannot be read ends its operations with the error.

**Tests.** 44 tests run on every platform against a simulated `Sys` (every operation's success, error and short transfers,
the ring and its fallbacks, a full submission queue, the order across ring and port, a pull's `stop`, late and warning packets,
slot reuse, routes and generations, handle modes, pool failures). 16 run on Windows
only (`io/win/live.rs`: files, loopback TCP and UDP, accept, connect, open and rename, threads each with a ring), each twice:
with IoRing and forced onto the port. `.github/workflows/windows.yml` runs the crate's tests on a `windows-2025` runner on
pushes to `graph`.

## Dataflow sources: `euhedral-graph` (2026-10-26)

**What.** A new crate, [`euhedral-graph`](euhedral-graph/README.md), adds two kinds of `LatticeSource` that the
lattice's own polls drive, and the structures built on them: `Completions` (continuation frames behind markers,
timers and completion tokens), `AsyncIo` (io_uring file and socket operations whose completions are frames; Linux and
Android), `Sequencer`, `Join`, `LanePool` with `DeviceQueue` and `Marker`, and `StageGraph`. They are ports of
Euhedral-Inference's `runtime.graph` package (`StageGraph`, `LanePool`, `Join`, `AsyncReads`) and
`quantization.lattice` (`GpuCompletions`), and of Euhedral-Execution's `FrameSequencer`. `euhedral-core` is unchanged.

**Why.** A frame must not wait: a worker is a core. Work that finishes elsewhere (a device queue, a file, a socket, a
deadline) was left to the application to poll by hand. These sources make the completion itself a frame: the lattice
polls the source like any other, and the continuation runs on whichever worker it is routed to.

**How to use it.**
```rust
let (lanes, devices) = FakeDevice::pool(&["compute"]); // or your own DeviceQueue lanes
let completions = Completions::new(lanes.clone());
lattice.add_upstream(completions.clone())?;
let marker = devices[0].submit(cost, &[]);            // submit; this returns at once
completions.after_marker(marker, Box::new(next_frame)); // next_frame runs once the marker is reached
```
- A `StageGraph<Q>` runs a static DAG of stages per quantum `Q`, with submission, completion and external edges, a
  kill switch per quantum and a `retire` callback that receives the `QuantumOutcome`. Its stage frames go to a
  `QueueIngestSink`; ordered stages use the id hash (`StageSpec::ordered`), unordered ones `randomize_hash`. Frames keep
  the default routing policy.
- `GraphRuntime::retry_backstop` (default `DEFAULT_RETRY_BACKSTOP`, 20 ms) bounds how long a retried stage waits for
  a lane's progress.
- Features: `io-uring` (default; `AsyncIo`, Linux and Android only) and `testing` (`testing::FakeDevice`). Windows and
  macOS get everything but `AsyncIo` for now.

**Semantics.**
- `Completions` queries each pending token once per poll, emits continuations in registration order and honours the
  pull's `stop` (ordered frames go out through `request`'s pushes and the lattice's routing, so ordered lanes keep
  registration order). A frame whose kill switch is set is emitted anyway: the executor skips it and runs its
  `do_finally`, so `pending()` drains to zero. A registration after `complete` finishes its frame through `do_finally`
  and fails a debug assertion.
- `Completions::lateness_us` reports `(p50, p99, max)` in µs, each rounded up, from a `metrics::Histogram`: the maximum
  is exact, the percentiles are interpolated within a bucket (a quarter of the value wide).
- A `StageGraph` never scans for ready work: the last arrival at a stage publishes it. A quantum retires once none of
  its frames can run again and every lane it used reached its last marker, and not before every `QuantumRef` clone was
  dropped.
- `QuantumOutcome` is how a quantum ended; it is not `frames::Outcome` (how a pipeline chain ended).
- `AsyncIo` has one submission ring per submitting thread; an operation belongs to the thread that submitted it.

**Tests.** 75 unit tests at that point (`Completions`, `AsyncIo`, `Sequencer`, `Join`, `LanePool`, `StageGraph`, `FakeDevice`), the
README doctests, and `tests/credit_graph.rs`: a 4-worker lattice running a per-item stage graph with a `Sequencer`, a
`Join`, credit-bounded admission, a seek and a pause, which fails if a frame callback runs over 200 µs without the OS
preempting it.

## Metrics restored (2026-10-24)

**What.** `euhedral-core` gets a [`metrics`](euhedral-core/src/metrics/mod.rs) module, the port of upstream's
`metrics` package (`ExecutionMetrics`, `CacheMetrics`, `MetricsAggregator`), which the port had dropped. Each fragment
worker and each cache registers meters in a `MetricsRegistry`; `MetricsRegistry::snapshot` and
`ControlPlaneLattice::metrics` read them all without stopping the workers. The fragment's cycle state has upstream's
two `FlowRecorder`s back, which feed the throughput summary and the two flow meters.

**Why.** Upstream reports latency, throughput, in-progress, backlog and cap-factor meters through Micrometer; the
port had the raw state (`in_progress`, `cap_factor`, the cache's count) but no way to read it as metrics.

**Departure from upstream: service time, not latency.** At the user's direction, RustEE has no `.execution.latency`.
Upstream's was a batch-end sample of a smoothed per-cycle service time (cycle time over frames). RustEE's
`.execution.serviceTime` summary records every frame's own body time: how long its `execute` takes to return. It is
`execute` only, not `do_finally` or the rest of the frame's lifecycle, and not the time a frame takes to get from
ingest to a CPU. Its percentiles are the frames'.

**How to use it.**
```rust
let registry = MetricsRegistry::new();
let config = LatticeConfig::with_metrics("app", registry.clone()); // or:
let fragment = FragmentConfig::default().with_metrics("app", registry.clone()); // fragments and their caches
// ... run the lattice ...
let snapshot = registry.snapshot(); // or lattice.metrics()
let p99 = snapshot.get("app.execution.serviceTime", cpu).and_then(|m| m.summary()).and_then(|s| s.percentile(0.99));
```
- `FragmentConfig` and `CacheConfig` have upstream's `metric_prefix()` and `registry()`, with `with_metric_prefix`
  and `with_registry` (`None` turns metrics off). `FragmentConfig::with_metrics` sets both its own and its cache
  config's, like `FragmentConfig.ofDefaults(metricPrefix, registry)`; `LatticeConfig::with_metrics` is
  `LatticeConfig.ofDefaults(.., metricPrefix, registry)`.
- `CloneableObject::metrics_registry` (default `None`) lets a lattice find its object's registry.

**Semantics.**

| Meter | Kind | Recorded | Upstream |
|---|---|---|---|
| `<prefix>.execution.serviceTime` | summary, ns | every frame, around its `execute` call in `AbstractExecutor::run` (`time_execute`) | replaces `.execution.latency` (`reportLatency` in `reportMetrics`) |
| `<prefix>.execution.serviceTime.flow` | flow, ns | after each productive cycle: the `FlowStats` of a recorder of the cycle's mean `execute` time | none |
| `<prefix>.execution.throughput` | summary, frames/s | at each batch end: the frame recorder's rate × 10^9 | `reportThroughput` in `reportMetrics` |
| `<prefix>.execution.throughput.flow` | flow, frames/s | after each productive cycle: the frame recorder's rate `FlowStats` | none |
| `<prefix>.execution.inProgress` | gauge | 1 while the worker is inside a cache take, a direct pull or a steal (`timed`) | `executionStarted`/`executionFinished` around `drainLocalCache`, `pullAvailableUpstream`, `stealWork` |
| `<prefix>.execution.frames` | counter | after each productive cycle | none |
| `<prefix>.cache.backlog` | gauge, frames | read at snapshot: the cache's count | `CacheMetrics` gauge on `TOTAL_COUNT` |
| `<prefix>.cache.capFactor` | summary | in `ControlPlaneCache::update`, the cap factor it set | `recordCapFactor` in `update` |

- **Execute timing.** `AbstractExecutor::run` wraps its `execute` call in `metrics::time_execute`. A metered worker
  installs its meters in a thread-local for its loop, so on that thread `time_execute` reads the clock before and
  after `execute` and records the difference; the finalizer is outside it. A frame that is not alive is not executed,
  so not timed; an `execute` that panics is not recorded. An executor that overrides `run` opts in by wrapping its own
  `execute` call in `time_execute`, and nothing else. An executor that runs frames on another thread is not timed.
  The decision tree's body cost keeps its own sampled timing of the push (one in 256), unchanged.
- **The recorders** are recorded where upstream's `recordProgress` records them, after each productive cycle and at
  the cycle's end time: the mean `execute` time of the cycle's frames (from the service-time summary's count and sum,
  so no further clock read) and the frames run. A worker reset starts them over, as `CycleState.reset()` does; the summaries
  stay cumulative.
- **Prefix:** `euhedral` unless set; blank means the default, and a trailing `.` is dropped (`metricName`).
- **Tags, per CPU:** `core` (the clone's core id) and `cpu`. Upstream has only `core`, so a core's two SMT workers
  share one Micrometer meter per name; here each worker and each cache has its own, which is what the user wants.
  `HistogramSnapshot::merge` combines them if a reader needs a core's total.
- **Summaries** are cumulative histograms of atomics: count, sum, max, and log2 buckets (each octave from `2^-16` to
  `2^48` split in four), so percentiles are within a quarter-octave. Negative and `NaN` values are ignored, as
  Micrometer ignores negative amounts.
- **Flows** are published by the worker through a sequence lock: the worker never waits, readers retry a read a write
  overlapped.
- **Lifecycle:** a fragment or cache registers when built and leaves its registry when closed or dropped (upstream's
  caches never removed their meters).
- **Cost.** No lock and no allocation on a worker's path; each worker owns its meters on their own cache lines.
  Without a registry a cycle pays one untaken branch, and each frame's `run` one relaxed load of a process-wide flag
  (and, once any metered worker has started in the process, a thread-local lookup) before calling `execute`. With
  one, each frame pays two clock reads and a histogram record, and each productive cycle up to about 30 ns, mostly the
  two `FlowRecorder` updates (`metrics_cost_per_cycle`, which records no `execute`: 21 to 22 ns over 4 to 8 ns off).
  Measured on an i9-14900K, in release, one SMT core's two workers running empty frames from an array sink
  (`metrics_overhead_per_frame`, medians of two runs of seven rounds, off and on alternating in one process): greedy
  mode 156 ns/frame off, 175 to 176 on (+19 to +20 ns, +12 to +13%); poll mode, with a cycle per one or two frames,
  244 to 246 off, 302 to 303 on (+55 to +59 ns, +22 to +24%). The user's frames run for milliseconds, so this is noise
  for them.
- `MetricsAggregator.getInProgressGauges(registry)` matched the latency suffix upstream; `in_progress_gauges` matches
  the in-progress suffix.
- No exporter: the `metrics` crate facade is push-based, so a bridge would either record through a global recorder on
  the hot path or poll snapshots on a thread of its own; an application can write the latter in a few lines.

**Tests:**
- `metrics/histogram.rs`: buckets and their bounds, recording, percentiles, `since` and `merge`, concurrent writers.
- `metrics/flow.rs`: publishing, no torn reads under a concurrent writer, `FlowStats` from a recorder.
- `metrics/tests.rs`: the names (upstream's, and service time) and prefix rules, nothing built without a registry,
  meters and tags, the aggregator lookups, sources leaving the registry, snapshots while four writers record.
- `config/*`: the prefix and registry settings and defaults.
- `control_plane/cache/tests.rs`: no meters without a registry; backlog and cap factor tagged with the core; closing or
  dropping a cache removes its meters.
- `metrics/tests.rs` also: `run` times `execute` alone on a thread with a sink, not a 3 ms `do_finally` and not a dead
  frame, and nothing without a sink; `time_execute` returns what `execute` returns.
- `control_plane/fragment/tests.rs`: no meters without a registry; both SMT workers and caches register, tagged;
  running workers record frames, throughput and flows while a reader snapshots; a cycle counts frames, feeds the
  service-time flow from the timed `execute` calls and reports throughput at batch ends.
- `impls/base_cloneable/tests.rs`: a metered lattice reports every worker's meters, every frame's `execute` timed,
  until closed; `the_service_time_is_execute_alone_not_the_finalizer`: 50 µs `execute`s with 2 ms finalizers record
  at least 50 µs each and under 1 ms.
  Ignored measurements: `metrics_cost_per_cycle` (fragment tests) and `metrics_overhead_per_frame`.

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
