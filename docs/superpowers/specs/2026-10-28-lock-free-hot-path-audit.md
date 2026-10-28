# Lock-free hot path: audit and fix design (2026-10-28)

**Rule.** The hot path is lock-free. Locks are allowed only for lifecycle work: start, stop (shutdown, drop,
complete, close) and rebalance (attach and detach of sources and receivers, lattice resizing, shard moves).

**Locking** here means `Mutex`, `RwLock`, `Condvar`, `Barrier`, a contended `OnceLock`/`LazyLock` initialisation
reachable after start, a spin-wait or back-off loop that waits on another thread, and a blocking system call or park,
on a path that runs per frame, per pull, per offer, per completion or per request.

**Scope.** `main` at `b33258c`: `euhedral-core`, `euhedral-data-structures`, `euhedral-hashing`,
`euhedral-hardware-utils`, `euhedral-graph`, `euhedral-io`, `euhedral-memory`. `euhedral-gpu` is out of scope (it is
being reworked separately). Test code (`tests.rs`, `testing.rs`, `#[cfg(test)]` modules, `tests/`, `examples/`) is
not counted. Paths are relative to the repository root; line numbers are those of `b33258c`.

**Classes.**

- **HOT**: runs per frame, pull, offer, completion or request in steady state.
- **LIFECYCLE**: start, stop or rebalance only.
- **MIXED**: both; each row says which paths are which. Rows tagged **tick** are taken in steady state by the
  periodic control loop (resource monitor poller, its dispatcher, the lattice's and shards' utilization updates).
  Those never run on a lattice worker and never contend with one, but they are not lifecycle work either, so they are
  counted as MIXED and fixed last (see question 1).

## 1. Summary

| Crate | HOT | MIXED | of which tick | LIFECYCLE |
|---|---:|---:|---:|---:|
| `euhedral-core` | 4 | 16 | 5 | 14 |
| `euhedral-data-structures` | 1 | 0 | 0 | 1 |
| `euhedral-hashing` | 0 | 0 | 0 | 0 |
| `euhedral-hardware-utils` | 0 | 7 | 7 | 4 |
| `euhedral-graph` | 10 | 3 | 0 | 2 |
| `euhedral-io` | 10 | 5 | 0 | 1 |
| `euhedral-memory` | 0 | 1 | 0 | 2 |
| **Total** | **25** | **32** | **12** | **24** |

One finding is one lock (a field or a static) or one blocking call site, with all its acquisitions.

**Where the hot locks are, by path:**

- **Every frame a worker runs** passes `LatticeHotSource::accept`: one `RwLock` read, two hook `Mutex`es and the
  `StopWatch` `Mutex` twice (C1, C2, C10), plus the body-cost `Mutex` on sampled frames (C3).
- **Every frame pushed through the lattice** (the request path: source push, global distributor, socket edge, core
  distributor, core edge, cache) takes two vertex routing `RwLock`s and two edge downstream `RwLock`s, each with an
  `Arc` clone on a line every worker shares (C6, C8).
- **Every cache take and steal** takes a partition `Mutex` (`try_lock`), and every steal attempt a directory
  `RwLock` (C4, C5).
- **Every pull and request of a stock source** takes the ingest `Downstream` `RwLock` and the queue consumer `Mutex`
  (C11, C12, C13); `Completions` and `AsyncIo` take two or three `Mutex`es per poll (G1-G3, I1-I6).
- **Every completion** in `euhedral-graph` takes a `Mutex` (`Join`, `Sequencer`, `Latch`) and a stage graph takes
  five `Mutex`es and an `RwLock` per stage (G4-G6, G10-G15).
- **Every Windows I/O operation** takes the slab `Mutex` two to four times and one or two `HashSet` `Mutex`es (I7-I15).
- **Multi-producer queue offers** spin while another producer installs a new chunk (D1).

**Not locks, and left as they are** (listed so the next audit does not re-raise them):

- The worker's idle parks: `control_plane/fragment.rs:359` (no upstreams and an empty cache), `:430`/`:432` (Idle
  path), `:439`/`:442` (unproductive park or yield), and the `spin_loop` hint at `:454`. They run only when the
  worker found no work, and they wait on no other thread (question 3).
- Non-blocking claims, which fail instead of waiting: `UpstreamInterceptor::try_acquire`
  (`flow_control/vertex.rs:594`), `ChunkPool::try_lock` (`queues/chunk.rs:133`), the multi-consumer queue's
  `try_lock_consumers` (`queues/core.rs:176`). Several fixes below reuse this pattern (question 2).
- The seqlock read of the flow meters (`metrics/flow.rs:88-115`): bounded retries, and the writer never waits.
- CAS retry loops that wait on nobody: the atomics in `euhedral-data-structures/src/atomics`, the planner, arena and
  tier moves in `euhedral-memory` (`planner.rs:293,391,756,1163`, `arena.rs:207`, `tiers.rs:610,928,966`).
- `GetQueuedCompletionStatusEx` with a zero timeout (`euhedral-io/src/win/real.rs:384`) and io_uring's submit without
  a wait (`uring.rs:658`): system calls that never block.
- `OnceLock::get` on an initialised cell is one acquire load. Only the initialisation can block; where it is a
  lifecycle step, the row is LIFECYCLE.

## 2. Findings

### 2.1 HOT

| ID | Site | Protects | Hot paths |
|---|---|---|---|
| C1 | `euhedral-core/src/flow_control/hot_source.rs:21-22` (`:56`, `:60`) | `before_each` / `after_each` hooks (`Mutex<Box<dyn FnMut>>`) | Worker: `run_cycle` → `timed` → `output.accept` (`fragment.rs:467`) → `LatticeHotSource::accept` → both hooks, for every frame it runs (cache greedy/poll, direct pull, steal). |
| C2 | `euhedral-core/src/control_plane/fragment.rs:748-753` | the sampling `StopWatch` (`Arc<Mutex<StopWatch>>`) | Same path as C1: `start` in `before_each`, `stop` in `after_each`, every frame. |
| C14 | `euhedral-core/src/ingest/pipeline_runner.rs:39-43` | the outcome hand-off: `mpsc::sync_channel(1)` per submitted chain | Per request: `PipelineRunner::submit*` makes the channel; the chain's last finalizer `send`s it on a worker, which takes the channel waker's internal `Mutex` and wakes the waiting receiver. |
| C15 | `euhedral-core/src/flow_control/hot_receiver.rs:46` | back-off (`spin_wait::await_while`: spin, yield, then 10 µs parks) until the `PushHook` takes the frame | Per push: `LatticeHotReceiver::push` waits on whatever frees the hook. Public API; nothing in the workspace uses it. |
| D1 | `euhedral-data-structures/src/queues/core.rs:371-378` | the tail's `RESIZING` flag in `mp_reserve` | Per offer of every multi-producer queue at a chunk boundary: a producer spins while another allocates and links the next chunk (`mp_resize`, `:428`), allocator call included. Reached by `QueueIngestSink::offer`, every cache `offer`, `Completions::add`, the io_uring incoming queue, registry handle queues. |
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

### 2.2 MIXED

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
| M1 | `euhedral-memory/src/cuda/backends.rs:49`, `:63`, `:444`, `:478` (and `free_vmm` `:319`) | blocking driver calls: `cuMemAlloc`, `cuMemHostAlloc`, `cuMemFree`, `cuMemFreeHost` (the frees synchronise the device) | Wherever the application runs `Reaper::reap` (`backend.rs:140`): its doc says "on the owner's lane", which is a lattice worker, per frame that reaps. | Arena and pool construction and teardown; the module doc (`backend.rs:5-7`) says a blocking free never runs on a worker. See question 8. |

### 2.3 LIFECYCLE

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
| C31 | `euhedral-core/src/flow_control/edge.rs:164` | `add_upstream_handle` spin-waits (`await_while`) while the edge drains | Attaching a source during a rebalance. Reached from `Lake::offer` when a busy period starts on a worker (question 7). |
| C32 | `euhedral-core/src/control_plane/shard.rs:445`, `:539`, `:605` | waiting for clones to drain; waiting for a rebalance before a reset | Rebalance, shut-down, reset (task-pool threads and the resetting thread). |
| C33 | `euhedral-core/src/control_plane/lattice.rs:264`, `:508` | waiting for readiness; waiting for a rebalance before a reset | `start`, `clear`. |
| C34 | `euhedral-core/src/control_plane/cache.rs:142`, `ingest/queue.rs:54`, `flow_control/vertex.rs:480` | set-once cells (`OnceLock`): SMT sibling, ingest owner, interceptor upstream | Set once at build or attach by one setter; `get` on hot paths is a lock-free load. |
| D2 | `euhedral-data-structures/src/queues/core.rs:192-202` | `lock_consumers`: spins until the multi-consumer lock is free | Only `Consumer<_, Multi>::clear` (`queue.rs:434`). |
| H8 | `euhedral-hardware-utils/src/affinity/executor.rs:170-171` | a pinned executor's control state (`Mutex` + `Condvar`) | `start`, `submit*` (worker start, clone build), `shutdown*`, `close`, `await_termination`, handle clone and drop. |
| H9 | `euhedral-hardware-utils/src/affinity/executor.rs:245` | the per-CPU executor registry | `get_or_create`, `get`, `close_all`, unregister. |
| H10 | `euhedral-hardware-utils/src/monitor/mod.rs:319` | the poller's `JoinHandle` | `start`, `close_within`. |
| H11 | `affinity/executor.rs:286`, `affinity/mod.rs:76`, `topology/mod.rs:242`, `monitor/mod.rs:55` (all in `euhedral-hardware-utils/src`) | process-wide `OnceLock` statics: executor registry, affinity controller, system topology, monotonic epoch | First use, which lattice construction and worker start reach before any frame. |
| G8 | `euhedral-graph/src/lake.rs:169` | generations being retired | `retire` (`:294`) on detach and complete; `is_attached` (`:286`) is a query. |
| G9 | `euhedral-graph/src/lake.rs:507` | the idle watch's downstream | `add_downstream` (`:532`), `complete` (`:558`). |
| I16 | `euhedral-io/src/win/real.rs:274`, `:278` | `OnceLock` statics: the IoRing API and the socket extension functions | Loaded when an engine opens. |
| M2 | `euhedral-memory/src/cuda/copier.rs:41`, `cuda/backends.rs:175` | `cuStreamSynchronize` | `Drop` of the copier and of the stream pool. |
| M3 | `euhedral-memory/src/planner.rs:1206`, `tiers.rs:1063` | set-once downstream (`OnceLock`) | Set at attach; `get` per request is a lock-free load. |

`cuCtxSynchronize` and `cuEventSynchronize` are loaded (`euhedral-memory/src/cuda/mod.rs:145`, `:177`) but the
library calls neither.

## 3. Replacement designs

### 3.0 Building blocks

The designs below use, in the preferred order: single ownership; the queues of `euhedral-data-structures`
(`Queue<T, P, C>` with `Single`/`Multi` handles, `PartitionedQueue`); atomics and copy-on-write snapshots written
only at lifecycle time. Five small primitives are new, because nothing in the workspace offers them. They go in
`euhedral-data-structures` (a new `sync` module), next to the queues, with the queues' style of role handles and
drop safety:

1. **`Claim<T>`**: a value behind an `AtomicBool`, the pattern `ChunkPool` already uses (`queues/chunk.rs:112`).
   `try_claim(&self) -> Option<ClaimGuard<'_, T>>` never waits; the guard gives `&mut T` and releases on drop,
   unwinding included (no poisoning: a panicking holder leaves the value as it was, as `try_hold` does today with
   `PoisonError::into_inner`). `claim_spin(&self, deadline)` exists for lifecycle callers only and is listed in the
   lint as a disallowed method outside lifecycle code. Why new: `Consumer<_, Multi>` gives the same claim for queue
   consumers (and is used for those below), but several claimed values are not queues (`VecDeque`s, a slab, the
   completions' waiting list).
2. **`SetOnce<T>`**: `AtomicPtr<T>` installed by CAS from a `Box`; `get` is an acquire load returning `&T`; `set`
   hands the value back if already set; no blocking init path. Why new: `OnceLock::set` can block a racing `get_or_init`
   and its intent (lazy init) differs; here the cell is written once at attach or by the first of several writers
   (`get_or_insert` semantics).
3. **`Published<T>`**: an atomically replaceable `Option<Arc<T>>` whose reads are lock-free and whose writes are
   lifecycle-only. Why new: there is no atomic `Arc` swap in the workspace, and `Arc` clones on the shared count are
   themselves the contention C6 and C8 suffer from. Design (a two-parity reader count, a brief form of RCU):
   - state: `current: AtomicPtr<T>` (from `Arc::into_raw`, null for `None`); `parity: AtomicUsize`;
     `readers: [Box<[CachePadded<AtomicUsize>]>; 2]`, one padded counter per stripe. The stripe is the caller's CPU for
     pinned workers (each worker writes only its own line) and a thread-local index for any other thread.
   - `load(&self) -> Guard<'_, T>`: `p = parity.load`; `readers[p & 1][stripe].fetch_add(1)`; if
     `parity.load != p`, undo and retry (only a concurrent writer causes a retry); then `current.load(Acquire)`. The
     guard derefs to `Option<&T>` and decrements on drop. Cost: two read-modify-writes on the caller's own line and
     two loads, against today's `RwLock` read (a CAS and a release on a line shared by every worker) plus an `Arc`
     clone and drop (two more on a shared line). `Guard::to_arc()` clones when a caller must keep the value.
   - `store(&self, new: Option<Arc<T>>) -> Retired<T>`: callers are serialised by a lifecycle lock or by
     construction. `old = current.swap(new)`; `parity.fetch_add(1)`. `Retired` holds `old` and the parity it was read
     under. It drops `old` once that parity's stripes sum to zero: `retired.wait(deadline)` (spins; lifecycle threads
     only), or `cell.defer(retired)` (never waits: keeps it in the cell's single pending slot and frees it at the next
     `store`, `try_reclaim` or drop that finds the stripes empty).
   - Invariant: at most one retirement is pending. A `store` that finds the previous one still held by readers either
     waits (lifecycle threads) or returns `Err(new)` to a caller that must not wait (`try_store`), which retries at its
     next lifecycle step. A store whose `old` is `None` needs no grace period.
   - A debug assertion fails a `store(...).wait()` called while the same thread holds a guard of that cell.
4. **`OneShot<T>`**: one value from a setter to a receiver: `state: AtomicU8` (empty, waiting, set, abandoned),
   `value: UnsafeCell<MaybeUninit<T>>`, `waiter: SetOnce<Thread>`. The setter writes the value, swaps the state to
   set, and unparks the waiter if the previous state was waiting (a wake, never a wait). The receiver has
   `try_recv` and `recv` (parks; for a thread outside the lattice). Why new: `std::sync::mpsc` and `Latch` both take a
   lock on the setter's side.
5. **`TakeAllStack<T>`**: a push-only stack (`AtomicPtr` head, CAS push) whose consumer takes the whole list with one
   swap, optionally closing it (`take_and_close`: later pushes fail and hand the value back). Swapping the whole list
   avoids the ABA problem of single pops. Why new: the queues allocate a chunk per queue, too heavy for the per-stage,
   per-quantum lists of the stage graph, which hold a handful of entries and are drained once.

And one composite in `euhedral-core` (public, since `euhedral-graph`, `euhedral-io` and `euhedral-memory` sources use
it):

6. **`DownstreamSlot<R: ?Sized>`**: a source's one receiver, attached once and detached once on completion, which
   hears of completion exactly once. `cell: Published<R>`, `state: AtomicU8` (empty, attaching, attached, complete).
   - `attach(r)`: CAS empty → attaching (otherwise: complete → `r.on_complete()`, attached → `r.on_error`); store
     `Some(r)` (old is `None`: nothing to retire); CAS attaching → attached; if `complete` raced in between, take it
     back and call `r.on_complete()`.
   - `complete()`: `prev = state.swap(complete)`; if `prev == attached`: `store(None)`, `on_complete()` on the old
     receiver, `defer` the retirement (no wait, so `complete` stays callable from a frame). If `prev == attaching`, the
     attacher delivers the completion.
   - `with(|r| ..)`: if attached, run `f` under a `load` guard. One store after attach, so the pending-retirement rule
     never bites.

### 3.1 `euhedral-core`

**C1, C2, C3, C10: the worker's output path (single ownership).** The hooks, the stop watch and the body-cost window
are only ever touched by the one worker that owns the output, so they move onto that worker's stack.

- `LatticeHotSource` gains a single-role pusher, as the queues do: `pusher(&self) -> HotPusher` (panics if one is
  live). `HotPusher` holds the hooks by value (`FnMut`, no `Mutex`) and pushes with `&mut self`. The shared part keeps
  only the downstream, in a `DownstreamSlot` (C10): `accept` reads it under a guard, with no `Arc` clone per frame.
  `before_each`/`after_each` move from the source builder to the pusher builder.
- `ControlPlaneFragment` builds the pusher at start and keeps it in `Cycle` (the worker's own state), with the
  `StopWatch` as a plain field. `timed` takes `&mut Cycle` and calls `cycle.output.push(frame)`.
- `BodyCost` becomes a plain struct owned by `DecisionTree` (`&mut self` everywhere): `record`, `calibrate` and
  `reset` already run only on the worker. The atomics `estimate_ns` and `samples` become plain fields. If anything
  off the worker needs the estimate later, publish it with one relaxed `AtomicU64` store per window.
- Result: zero shared writes per frame on the output path, besides the executor's own.

**C4: cache partitions (existing queue role).** Build each cache's queue as a `PartitionedMpmcQueue` and hold one
`Consumer<_, Multi>` per partition; its built-in non-blocking consumer claim (`try_lock_consumers`) replaces the
`Mutex`.

- `greedy_while`: `consumer.drain_while(consume, stop, remaining)` with the queue's `max_consume_batch` set to
  `u64::MAX` at construction, so one claim covers the whole take, as the held `MutexGuard` does today (stealers pass
  over a claimed partition).
- `poll_while`/`take_one`: `consumer.drain_while(consume, stop, 1)`, which peeks, tests `stop` and takes under one
  claim, so a stealer never takes an ordered head.
- `first_touch` and `reset_for_next_trial`: `Consumer<_, Multi>::clear` (spins on the claim: lifecycle only).
- Producer side is unchanged (already `Multi`). The claim is released on unwind, which is what the poison takeover
  does now.

**C5: cache directory.** `slots: Box<[Published<ControlPlaneCache>]>` holding strong `Arc`s (the directory already
lives no longer than the lattice; `withdraw` at close clears the slot, so no cycle survives close). `cache_at` returns
a guard: no `Weak::upgrade` per steal. `claim` stores (old is `None`); `withdraw` stores `None` and `defer`s, because
a stealer may hold the guard while it runs stolen frames, which can take long.

**C6, C7: vertex routing (copy-on-write generation, writer retires).** `routing: Published<RoutingState>`.

- `try_push` and `on_error` route under a guard held across the downstream push, as the cloned `Arc` is held today.
- `set_downstream_mapping` and `close_after_routes_retire` store the new generation, then wait for the old one's
  readers (`retired.wait(deadline)`, deadline from the lattice's shutdown timeout) and call `release(old)` themselves.
  `RoutingState` loses its `Drop` hook, so `Book` (C7) is only ever locked by these lifecycle callers: it stays a
  lifecycle `Mutex`.
- A push in flight finishes on the generation it began with, and a dropped downstream is closed only after every such
  push left, as today. What changes is the thread: retirement runs on the remapping or closing thread instead of the
  last pushing worker (question 4). `ControlPlaneLattice::close` then no longer needs the spawned closing thread for the
  "last route retires on a worker" case; it keeps it only if a wait past the deadline must not block the caller.
- If the wait reaches its deadline (a writer called from inside a push), `defer` the retirement and run `release` at
  the next lifecycle step or at drop.

**C8, C9: edge links.** `downstream` becomes a `DownstreamSlot`-like cell for the `Downstream` enum (attach once,
cleared at `close_edge`). `push_downstream` reads it under a guard; no `Arc` clone per hop. `parent` becomes
`Published<Weak<dyn LatticeNode>>`; the climb upgrades the `Weak` as today (it is off the stock worker loop) but takes
no lock. `set_parent` skips the store when the parent is unchanged, which is every remap after the first.

**C11, C12, C13: stock ingest sources.**

- `Downstream` (C11) becomes `DownstreamSlot<dyn LatticeReceiver>` plus the existing `demand` atomic.
- `QueueSource` (C12) builds a `PartitionedMpmcQueue` and keeps a `PartitionedConsumer<_, Multi>`: `pull` and
  `drain_requests` use its non-blocking claim (`try_hold` returns `None` today in the same cases); `clear` uses the
  spinning `clear` (lifecycle). `discard_queued` keeps its retry loop over the non-blocking claim.
- `SingleUseSource` (C13): `frame: Claim<Option<Box<dyn Frame>>>` plus a `cancelled: AtomicBool`. `pull`/`request`:
  `try_claim`; test `stop` on the frame; take it or leave it; on release, if `cancelled` is set and the frame is still
  there, cancel it. `complete`: set `cancelled`, then `try_claim` and cancel the frame; if a pull holds the claim, that
  pull cancels it on release. The frame is finished exactly once.

**C14: `PipelineRunner` outcomes.** `observed()` returns a `OneShot<Outcome>` receiver instead of an `mpsc::Receiver`;
the chain's finalizer sets it (no lock, at most one unpark). Allocation drops from a channel plus a boxed closure to
one `Arc<OneShot>`, which a pool of `OneShot`s recycled through an `MpscQueue` can also remove. This changes the
public return type of `submit*` (question 6).

**C15: `LatticeHotReceiver`.** Never wait on the hook. Recommended (question 5): a frame the hook defers goes to an
owned retry queue (`MpscQueue<Box<F>>`); each later `push` first offers the retry queue's frames in order, and the
receiver is also a `LatticeSource` whose `pull`/`request` offers them again, so the lattice's own polls retry them and
nothing spins. Order: a deferred frame stays ahead of later pushes because they queue behind it while the queue is not
empty.

**C16-C20 (tick).** Only if question 1 says ticks must be lock-free:

- The dispatcher thread is the only caller of `ControlPlaneLattice::update` in steady state; `start` posts its first
  update through the same path, so `updating` (C17) has one caller and goes.
- The steady-state branch reads `closed` and `resetting` (atomics) and a `Published<ActiveSet>` (version and active
  sockets) written only by rebalance; it then calls each shard's update directly (it is now atomics per clone), so the
  per-tick `TaskPool` hop (C20) goes. `gate` (C16) stays as a lifecycle lock around the rebalance branch and close.
- `ControlPlaneShard::update` reads a `Published<Vec<(core, clone)>>` written at rebalance; `State` (C19) stays for
  lifecycle.
- `TaskPool` keeps only lifecycle tasks; its `mpsc` + `Mutex<Receiver>` can become an `MpmcQueue` with park/unpark of
  idle pool threads if lifecycle tasks should not lock either (they may).

**Lifecycle rows (C21-C34).** Keep. Two improvements that remove waits reachable from hot entry points:

- C31: `add_upstream_handle` stops spinning while the edge drains. Handles that arrive during a drain go to an
  `MpscQueue` on the edge; the code that clears the drain flag (`finish_rebalance`, `Reopen::drop`, the end of
  `handle_system_topology_change`) flushes them into the registry (question 9).
- C26: `add_upstream` calls `start`, which locks `started` on every attach; check an `AtomicBool started` first and lock
  only when not started.

### 3.2 `euhedral-data-structures`

**D1: lock-free chunk install for multi-producer queues.** Two steps, each a PR on its own:

1. Allocate before claiming: a producer that finds the ring full takes a chunk from the pool or allocates one
   *before* it tries to set `RESIZING`; the loser of the race returns its chunk to the pool. The flag's window shrinks
   to a few stores, with no allocator call inside.
2. Helping: publish the resize as a descriptor (old chunk, new chunk, tail) in an `AtomicPtr` on the producer state.
   A producer that sees `RESIZING` completes the install itself (CAS the old chunk's `next`, CAS `tail_chunk`, store
   the epoch, clear the flag with a CAS on `tail`) instead of spinning; every step is idempotent, so the producer that
   began the resize and any helpers converge. Then no producer waits on another, and a producer preempted mid-resize
   stops nobody.

The consumer side already never waits. A slot reserved and not yet published stops the consumer at that slot until its
producer publishes; that is a progress dependency of every array queue, not a wait (question 12).

D2 stays (lifecycle only); the lint below keeps `lock_consumers` and `Consumer<_, Multi>::clear` out of hot code.

### 3.3 `euhedral-graph`

**G1, G2, G3: `Completions`.**

- `poll: Claim<Poll>`. `emit` uses `try_claim` and returns 0 if another poll holds it (the lattice's interceptor
  already lets one worker at a time poll a source; a direct caller polling from two threads gets 0 on one of them).
  `sweep` already does this.
- `lanes` (G2) moves into `Poll`, so the claim holder owns it: `complete` sets the flag and, if it gets the claim,
  drops the lanes; otherwise the next claim holder drops them when it sees the flag. `emit` reads lanes without a
  lock or an `Arc` clone.
- `downstream` (G3) becomes a `DownstreamSlot`.

**G4: `Join`.** `remaining: AtomicUsize`; per part `ended: AtomicBool` and `value: UnsafeCell<Option<T>>`;
`then: UnsafeCell<Option<Then<T>>>`. `end(part, v)`: `assert!(!ended[part].swap(true))` (one writer per part from
here on), write the value, then `remaining.fetch_sub(1, AcqRel)`; the arrival that brings it to zero has seen every
other part's write and takes the values and the continuation. Wait-free, no allocation per arrival.

**G5: `Latch`.** Becomes a thin wrapper of `OneShot` (the setter's `Drop` sets abandoned). The waiter still parks on
the main thread; the setter no longer locks.

**G6: `Sequencer`.** Admission order becomes the order of an `MpscQueue<Arc<SeqCell<T>>>`, whose cells hold an
`AtomicU8` state (pending, ready, abandoned) and the value.

- `register`: take a cell (recycled through a second `MpscQueue`, so steady state allocates nothing), offer a clone
  to the order queue, return the `Slot` holding the cell.
- `finish`: write the value, store the state with release, then `release()`.
- `release()` keeps its `wip` counter, which already lets one thread at a time drain; the drainer owns the order
  queue's `Consumer<_, Single>` (in an `UnsafeCell` reached only by the `wip` winner), peeks the front, and pops and
  emits while the front is not pending.
- `len` becomes the queue's size.

**G7: `Lake`.** `generation: Published<Generation>`. `offer` and `admit` read under a guard. `attach` (first offer of
a busy period, a lifecycle step) keeps a lifecycle lock serialising writers and stores `Some` (nothing to retire).
`detach` and `complete` store `None` and `defer`; `detach` uses `try_store` and, if a previous retirement is still
held, stays attached until the next idle poll (detaching is already best effort). `retiring` (G8) stays.

**G10-G15: stage graph runs.** A run's shared state is written by several arrivals and read once a count reaches zero.
The counts (`pending`, `live`, `holds`) already order those phases with `AcqRel` read-modify-writes, so each piece
becomes either atomics or single ownership proved by a count:

- **G10 quantum**: `UnsafeCell<Option<Q>>`. Stage runs and `quiesce` read it while `holds > 0`; `conclude` runs once,
  from the retirement frame, after `holds` reached zero, so it owns the cell. A debug counter asserts no reader is
  active at `conclude`.
- **G11 cause**: `SetOnce<QuantumOutcome>` (first outcome wins, as `get_or_insert` does); `conclude` takes it.
- **G12 inputs**: markers merge as a per-lane maximum, so each stage's inputs become `Box<[AtomicU64]>`, one per lane
  of the pool, updated with `fetch_max(value + 1)` (0 means none). `run_stage` and pass-through `submitted` read them
  after `pending` reached zero. Wait-free, and the per-arrival `Vec` work goes.
- **G14 boundary**: frontier markers the same way (per-lane `AtomicU64`, `fetch_max`), so `cover` is one atomic per
  marker; host completions go to a `TakeAllStack<Box<dyn Ready>>` that `arm_boundary` takes after `holds` reached zero.
- **G13 externals**: per stage, a one-shot event: `SetOnce<Vec<Marker>>` for the stage's output plus a
  `TakeAllStack<ArriveWith>` of waiters. `subscribe`: if the output is set, call now; else push the waiter, and if the
  push finds the stack closed, call with the published value. `notify_externals`: set the output, `take_and_close`
  the waiters, call them. `quiesce` (swept) closes every stage's stack and marks it swept; a subscriber that finds it
  swept reads the boundary atomics at that moment, as it reads the locked boundary now.
- **G15 pins**: per stage, a `TakeAllStack<QuantumRef>`: `register_external` pushes; `release_pins(stage)` takes that
  stage's list; `quiesce` takes every list.

### 3.4 `euhedral-io`

**I1, I2: `Front`.** `held: Claim<VecDeque<Box<dyn Frame>>>` (one poll at a time is the documented contract; a
second concurrent poll returns 0 instead of waiting). `downstream: DownstreamSlot`.

**I3, I13: ring lists (append-only).** Rings are never removed while the engine lives. Replace the `Mutex<Vec>` with
`rings: Box<[SetOnce<Arc<Ring>>]>` of `MAX_RINGS` entries and `len: AtomicUsize`. A thread's first submit reserves an
index (`fetch_add`), sets the slot, and the poll walks `0..len`, skipping slots not yet set. No clone, no allocation per
poll.

**I4, I5, I14: ring claims.** `sq: Claim<Queue>`, `cq: Claim<()>`, Windows `inner: Claim<RingInner>`, keeping the
existing `waiting` flag and fence protocol, which already makes them non-blocking. `break_ring` sets `broken` and uses
`try_claim`; if the claim is busy, the holder clears the backlog when it sees `broken` on release.

**I6: io_uring operation slab.** Each ring belongs to its submitting thread (the `MINE` thread-local guarantees it),
and one poller at a time reaps it (the `cq` claim), so:

- slots live in fixed segments of 256 (an append-only array of `SetOnce` segment pointers), each slot an
  `UnsafeCell<Option<Op>>` with a `generation: AtomicU32`;
- `submit` (owner thread) takes a free index, writes the `Op`, then stores the generation with release;
- `reap_one` (the `cq` holder) checks the generation with acquire, reads and updates `moved`, and on the end takes
  the `Op` and returns the index through an `MpscQueue<u32>` free list whose consumer is the owner thread;
- `Drop` has `&mut self`.

**I7: Windows slab.** Same shape, with the state machine (`Idle`, `Issuing`, `InFlight`, `Busy`) in an `AtomicU8` per
slot changed by CAS, and the free list an `MpmcQueue<u32>` (any thread submits). The `by_addr` map goes: `Rec` is
`#[repr(C)]` with the `OVERLAPPED` first and the slot index and generation next to it, so a packet's `OVERLAPPED`
pointer leads to its slot by a cast. Records sit in segments the engine allocates and never frees while it lives; a
per-engine tag in each record rejects foreign pointers (strays).

**I8, I9, I15: Windows queues.** `ready` and `pool_done` become `MpscQueue`s (producers: failing submitters, pool
threads; consumer: the poller, through the `Front` claim). A ring's `backlog` becomes an `MpscQueue<Entry>` whose
consumer is used by the `inner` claim holder; `clear` (break) is done by that holder.

**I10: inbox.** Owned by the poller: `Claim<VecDeque<Packet>>` (taken once per poll alongside the `Front` claim), or
folded into the `Front` claim's state.

**I11, I12: handle routes.** A handle's first use is its attach. Keep a
`Published<HashMap<Handle, Route>>` copied on write when a new handle joins (writers serialised by a lifecycle lock)
and read lock-free per operation. This also removes the `associate` system call that `join_port` makes on every send
and receive today: it is made once, when the handle is first seen.

### 3.5 `euhedral-hardware-utils` (tick; question 1)

- **H1-H4 monitor**: the poller thread owns the evaluation state (H2) and the snapshot provider (H4: a provider per
  monitor, moved into the poller, instead of a process-wide `Mutex`). The last utilization (H3) becomes a
  `Published<HardwareUtilization>`. The poller's wait between ticks becomes `park_timeout` on its own thread, which
  `stop`/`close` unpark; `Lifecycle` (H1) stays a `Mutex` + `Condvar` used only by start, stop and close.
- **H5 dispatcher**: the latest value goes into a coalescing mailbox (`AtomicPtr` swap of a boxed value; the dispatcher
  takes it with a swap to null), and `offer` unparks the dispatcher thread. Listener add and remove become a
  `Published<Vec<Listener>>` copied on write (attach).
- **H6, H7 mapper**: per-socket versions are written only by the single updater (the dispatcher thread), so they become
  plain fields behind `&mut` or `AtomicU64`s; `current` becomes a `Published<EffectiveSystemTopology>`.

### 3.6 `euhedral-memory`

**M1**: depends on question 8. If blocking frees must never run on a worker: enforce it, with a debug assertion in
the CUDA backends' `free` (and in `Reaper::reap` when its backend `may_block`) that the calling thread is not inside a
hot scope (§4, chunk 1), and document that reaping belongs on a lattice set aside for blocking work. If reaping on the
owner's lane is intended, it stays, but it is then a blocking call per frame, which the rule forbids; the
alternative is a reaper that issues the asynchronous frees (`cuMemFreeAsync` on the owning stream, which the stream
pool already uses at `backends.rs:160`) and never synchronises.

## 4. Fix plan

Each chunk is one PR, in this order. Every PR runs the workspace gates (`cargo fmt`, `cargo clippy --workspace
--all-targets`, `cargo test --workspace`; the Windows workflow for `euhedral-io`) plus the tests named.

**Chunk 1: guard rails (no behaviour change).**

- `LifecycleMutex<T>` and `LifecycleRwLock<T>` (thin wrappers of the std types, in
  `euhedral-data-structures::sync`), the only code allowed to name `std::sync::{Mutex, RwLock}`. Their lock methods
  `debug_assert!` that the thread is not in a hot scope.
- `HotScope`: a debug-only thread-local depth counter (`cfg(debug_assertions)`; nothing in release). Entered by
  `ControlPlaneFragment::run_cycle`, `UpstreamQueue::pull`/`request`, `LatticeVertex::try_push`,
  `ControlPlaneCache::{push, greedy_while, poll_while, work_steal, work_steal_polled}`, `QueueSource::{offer, pull,
  request}`, `Completions::{emit, add}`, `Join::end`, `Sequencer::{register, finish}`, the stage graph's
  `Run::{arrive, run_stage, submitted}`, `AsyncIo`'s `submit` and `reap`. A lifecycle call made from inside a frame
  (attach from `Lake::offer`, `complete` from a finalizer) leaves the scope explicitly with `HotScope::lifecycle(|| ..)`.
- Lint: a workspace-root `clippy.toml` (and the same entries added to `euhedral-graph/clippy.toml` and
  `euhedral-io/clippy.toml`): `disallowed-types` = `std::sync::{Mutex, RwLock, Condvar, Barrier}`,
  `std::sync::mpsc::{Sender, SyncSender, Receiver}`; `disallowed-methods` = `std::thread::{park, park_timeout, sleep,
  yield_now}`, `std::hint::spin_loop`, `std::sync::Condvar::wait*`, `euhedral_core::utils::spin_wait::await_while`,
  `Claim::claim_spin`, `Consumer::clear` (multi role). Allowed sites carry `#[allow(..., reason = "lifecycle: ..")]`
  or `reason = "idle policy"` (the worker's parks).
- Migration: every current HOT and MIXED site gets `#[allow(..., reason = "HOT-LOCK(<id>)")]`, and its lock calls a
  `HotScope::lifecycle` exemption, so the lint and the assertions pass on day one.
- Tests: `a_lifecycle_lock_inside_a_hot_scope_panics` (`should_panic`, debug); `hot_lock_ratchet` (an integration
  test that counts `HOT-LOCK(` tags under `*/src` and fails above a constant each later PR lowers). The existing
  lattice, graph and io tests run in debug, so any lifecycle lock reached from a hot scope fails them.

**Chunk 2: primitives.** `Claim`, `SetOnce`, `Published`, `OneShot`, `TakeAllStack` in
`euhedral-data-structures::sync`; `DownstreamSlot` in `euhedral-core::flow_control`.

- Tests, in the crate's style (threaded, under deadlines):
  - drop counting: every value dropped exactly once, none while a guard is held (a guard that sleeps while
    `store().wait()` runs, which must not return before the guard drops);
  - `try_store` refuses while a retirement is pending;
  - `defer` frees at the next store;
  - a writer waiting while its own thread holds a guard trips the debug assertion;
  - `OneShot`: set-before-wait and wait-before-set, abandon, many pairs in parallel;
  - `Claim` released on unwind;
  - `TakeAllStack` close races (push after close hands the value back; no value lost under 8 pushers);
  - `DownstreamSlot` exactly-once completion under attach/complete races.
- Miri over these modules (question 10).

**Chunk 3: queue chunk install (D1).** Step 1 (allocate before claiming) and step 2 (helping) as two commits.

- Tests: the existing model check against `VecDeque`; a stress test with chunk size 2 and 8 producers; a test hook
  that pauses a producer after it sets `RESIZING`, during which the other producers' offers must complete (they spin
  forever today).
- Paired benchmark of offer throughput with the existing queue benches.

**Chunk 4: worker output path (C1, C2, C3, C10).**

- Tests: existing `hot_source`, `decision_tree` and fragment tests; a new test that runs a lattice in debug with the
  hot-scope assertion armed for the C1/C2/C3/C10 sites (their `HOT-LOCK` exemptions removed).
- Ratchet: -4. Paired benchmark of a lattice of short frames (frames per second, p99 service time).

**Chunk 5: caches (C4, C5).**

- Tests: existing cache tests (ordered frames keep their order under stealing; in poll mode a blocked frame leaves
  the unordered frames behind it to stealers); new: stealers racing a greedy owner never take an ordered or
  `CacheLocal` frame; a cache closed while a stealer holds its directory guard is freed after the stealer leaves
  (drop counter).
- Ratchet: -2.

**Chunk 6: routing and ingest (C6, C7, C8, C9, C11, C12, C13; C31 and C26 improvements).**

- Tests: existing vertex retirement tests; new: remap while 8 threads push (no frame lost, every dropped downstream
  closed exactly once, after its last push); `close_after_routes_retire` while pushes are in flight; ingest
  complete-exactly-once under attach and complete races; a handle added during a drain is registered when the drain
  ends without the adder waiting.
- Ratchet: -7.

**Chunk 7: completions and receivers in core (C14, C15).** After questions 5 and 6.

- Tests: existing pipeline-runner tests ported to the new receiver; a hook that defers N times then accepts gets its
  frames in order with no thread parked or spinning (the test checks the pusher returns at once).
- Ratchet: -2.

**Chunk 8: graph sources and fan-in (G1-G7).**

- Tests: existing `completions`, `join`, `sequencer`, `latch` and `lake` tests (including the multi-threaded ones);
  new: `Join` with 64 parts arriving from 8 threads; `Sequencer` registering from 4 threads and completing in
  reverse; `Lake` offers racing detach and re-attach (no frame lost, no generation freed while offered to).
- Ratchet: -9.

**Chunk 9: stage graph (G10-G15).**

- Tests: existing `stage_graph` tests and `tests/credit_graph.rs`; new: a diamond graph with concurrent arrivals
  across 8 workers checking merged inputs equal the per-lane maximum; external edges subscribing before, during and
  after publication and after quiesce.
- Ratchet: -6.

**Chunk 10: io_uring (I1-I6).**

- Tests: existing `uring.rs` tests and `tests/io_owned.rs`; new: 8 submitting threads with one poller (no
  operation lost, slots reused); a poll racing `complete`.
- Ratchet: -6.

**Chunk 11: Windows engine (I7-I15).**

- Tests: the simulated-sys tests (`win/tests.rs`, which run on every platform) and the Windows workflow's live tests;
  new simulated tests: strays rejected by the record tag; a ring break while another thread holds the claim.
- Ratchet: -9.

**Chunk 12: memory (M1).** After question 8: the debug assertion, or the asynchronous reaper. Tests: a blocking-free
backend reaped inside a hot scope panics in debug.

**Chunk 13: tick path (C16-C20, H1-H7).** Only if question 1 says so.

- Tests: existing monitor, dispatcher, mapper and lattice tests (including coalescing to the latest value and
  rebalance on topology change); new: 10 000 synthetic ticks with the hot-scope assertion armed on the dispatcher
  thread.
- Ratchet: -12.

**Chunk 14: lock the door.** Ratchet constant at zero; the test becomes "no `HOT-LOCK(` tag exists". From here a new
`Mutex` in hot code fails clippy, and a lifecycle lock reached from a hot path fails the debug test suite.

## 5. Risks

- **Reclamation bugs.** `Published` and the slabs free memory that other threads may read. Mitigation: the
  drop-counting tests, Miri, and keeping every retirement either waited (lifecycle) or deferred to a known point; one
  pending retirement per cell.
- **Writer waits.** A lifecycle writer waits for readers. A writer running inside a reader of the same cell would wait
  for itself: the debug assertion catches it, and the wait has a deadline after which the retirement is deferred.
- **Retirement moves threads (C6, C7).** Downstreams and close retirements run on the remapping or closing thread
  instead of the last pushing worker. Code that relied on running on a worker (none found) would change; lattice
  close must keep its shard-closing thread if waiting there is unwanted.
- **Claims return 0 instead of waiting.** `Completions`, `AsyncIo` and the ingest queue polled by two threads at once
  outside a lattice: one gets 0. The lattice never does this.
- **Public API changes**: `PipelineRunner::submit*` return type (C14), `LatticeHotSource` hooks moving to the pusher
  (C1), `LatticeHotReceiver` retry semantics (C15).
- **Behaviour under rebalance**: handles added during a drain register when it ends instead of after a spin (C31);
  the same ordering, different thread.
- **Windows slab rewrite (I7)**: largest single change, with the `OVERLAPPED`-to-record cast; the simulated tests run on
  every platform, the live ones only on the Windows workflow.
- **Stripe memory.** One padded counter per CPU per parity per `Published`: 2 × 128 B × CPUs. Hundreds of cells
  (per-edge downstreams, one per source) would cost hundreds of kilobytes on a large host; stripes can be shared by
  cells of one lattice (one stripe set per lattice, each cell a parity word) if that matters.
- **Divergence from the upstream Java runtime.** The fixes change structure the port kept (generation `Arc`s released
  by the last holder, `try_lock` partitions). Behaviour stays; the code reads less like upstream.
- **Performance.** The new primitives must beat what they replace on the per-frame path; chunks 3-6 carry paired
  benchmarks (profile, hypothesis, paired gate, keep or discard), and a chunk whose primitive does not win is redone,
  not merged.

## 6. Questions for the user

1. **Ticks.** Do the periodic control paths count as hot: the resource monitor's poller and dispatcher, the lattice's
   and shards' utilization updates, the task pool they use (C16-C20, H1-H7)? They never run on a worker and never
   contend with one. If they may lock, chunk 13 is dropped and those 12 rows become LIFECYCLE.
2. **Non-blocking claims.** Is a claim that fails instead of waiting acceptable on a hot path (an `AtomicBool` CAS;
   whoever finds it taken skips the work and leaves it to the holder or the next poll)? RustEE already does this in
   `UpstreamInterceptor`, `ChunkPool` and the multi-consumer queue, and the designs for C4, C12, C13, G1, I1, I4, I5,
   I10 and I14 rely on it.
3. **Idle parks.** The worker's `park_timeout` and `yield_now` when it found no work (`fragment.rs:359-454`) are the
   idle policy, not locks. Confirm they stay.
4. **Who retires a routing generation.** OK to move closing dropped downstreams and running close retirements from the
   last worker that pushed on the old generation to the thread that remaps or closes, which then waits (with a
   deadline) for pushes in flight?
5. **`LatticeHotReceiver`.** Its `push` backs off until the hook takes the frame. Which lock-free behaviour do you
   want: (a) deferred frames go to a retry queue that later pushes and the lattice's polls drain, in order
   (recommended), or (b) `push` hands a deferred frame back to its caller?
6. **`PipelineRunner::submit*`** returns `std::sync::mpsc::Receiver<Outcome>`. OK to return a lock-free one-shot
   receiver type instead (same `recv`/`try_recv` shape)?
7. **Attach from `Lake::offer`.** The first offer of a busy period attaches the lake's sinks (`add_upstream`: the
   `started` lock, a spin while a rebalance drains, the registry lock), on the offering thread, often a worker. Attach
   is lifecycle by the rule; is it acceptable that a per-offer API reaches it, or should a lake keep its sinks attached
   for its lifetime and only stop pulling from them when idle?
8. **CUDA frees.** `euhedral-memory/src/backend.rs:5-7` says a blocking free never runs on a lattice worker, and
   `Reaper`'s doc (`:95-97`) says to run `reap` on the owner's lane with its blocking time counted as service time.
   Which is intended? The second puts `cuMemFree` (which synchronises the device) on a worker.
9. **Attach during a drain.** OK for `add_upstream_handle` to queue a handle added while the edge drains and register
   it when the drain ends, instead of spinning until then (C31)?
10. **Verification tools.** For the new primitives, is a `loom` dev-dependency acceptable for model checking, and may
    the plan run Miri (a nightly toolchain beside the pinned stable one)? Without them the tests are threaded stress
    tests under deadlines, the crates' current style.
11. **Strict scope of "lock".** The cache partitions and io rings use `Mutex::try_lock`, which never waits. Should these
    still go (they are a `Mutex` type, and chunk 14's lint bans the type), or may a `try_lock`-only `Mutex` stay where a
    claim would do the same?
12. **Array-queue progress.** In the chunked queues, a producer preempted between reserving a slot and publishing it
    holds up the consumer at that slot (the consumer returns empty, it does not wait). Acceptable, or should the
    consumer skip unpublished slots, which breaks per-partition FIFO?
