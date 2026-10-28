# Changelog

Changes to the lattice's execution model, newest first. Each entry says what changed, why, how to use it, its exact
semantics and the tests that pin it.

## `AsyncIo` moves to its own crate, `euhedral-io` (2026-10-28)

**What.** `euhedral-graph`'s `io` module (the `AsyncIo` API, the io_uring engine, the Windows IoRing and completion
port engines and their simulated tests), its integration test `tests/io_owned.rs`, its `async-io` feature and its
`io-uring`, `windows-sys` and (for the engines) `libc` dependencies move into the new workspace member `euhedral-io`.
`Continuation<T>` (what turns a completed operation's result into the next frame) moves with it: only the I/O used it.
`euhedral-graph` loses the `async-io` feature and the `io-uring` and `windows-sys` dependencies; `Completions`,
`ThreadUsage` and the rest stay.

**Why.** Asynchronous I/O does not use anything else in `euhedral-graph` (completions, lanes, stage graphs), and the
graph does not use it, so the two were joined for no reason. `euhedral-io` depends on `euhedral-core` and
`euhedral-data-structures` only, not on `euhedral-graph`.

**Use.** `euhedral_graph::{AsyncIo, AsyncIoOptions, Continuation, Create, IoBuf, IoDone, IoDoneOwned, IoMemory, IoThen,
OpenSpec, OwnedFile, OwnedSock, RawFile, RawSock}` become `euhedral_io::{...}`; add `euhedral-io` to the dependencies.
Behaviour is unchanged: the code, its docs and its tests moved as they were, with their paths adjusted.

**Tests.** The same tests, now `cargo test -p euhedral-io`; the Windows workflow runs that package.

## Tiered admission and the copy rate: `euhedral-memory` (2026-10-28)

**What.**
- `PlanItem::tiered(consumer, priority, required, unit, units, min_first, order)`: a consumer of equal units that may
  live in any of `order`'s locations, fastest first. `Planner::admit` places its `min_first` units in `order[0]` with
  the plan's other items (spill, cut, refuse as before); once those fit, the other units fill the first location of
  `order` with free room, then the next, highest priority first. `Admission::tiers` gives the units per location;
  units that fit nowhere are left out (the consumer runs with fewer). Extra units use free room only, never ordering a
  give-back.
- Example `copy_rate` (feature `cuda`): pinned host memory to and from the device, by how the host memory is pinned
  (the driver's `cuMemHostAlloc`, a registered mapping on huge pages, on small pages), copy size and stream count.

**Why.** Residency in tiers: as much as possible in the fastest location, the rest streamed from the next, refused only
when nothing can hold it. And a measured answer to how to saturate the link.

**Measured** (RTX 5070 Ti, PCIe 5, 1 GiB per direction): registered huge pages 50-52 GB/s host to device, 57 device to
host, 58 both ways; registered small pages and `cuMemHostAlloc` 24-27 one way, 31 both. Copy size (2-128 MiB) and
stream count (1-4) change nothing: pin on huge pages.

**Tests.** `tests/planner.rs`: `a_tiered_consumer_fills_the_device_then_the_slower_tiers_in_order`,
`tiered_consumers_take_the_fast_tier_by_priority_and_what_fits_nowhere_is_left_out`,
`a_tiered_consumer_whose_required_units_do_not_fit_is_refused_or_dropped`.

## Tier moves: `euhedral-memory::tiers` (2026-10-28)

**What.** Planned movement of claimed memory between locations (device, pinned host, pageable host, any modeled one):
- `TierPool`: one planned allocation per tier, cut into slots, leased and returned without allocating;
- `Mover`: residents (one slot of an owner's data each) placed by priority, moved by asynchronous copies, spilled
  under budget pressure and brought back as room returns;
- `Copier`: the copy engine per pair of locations: `HostCopier` (`memcpy`), `cuda::StreamCopier` (`cuMemcpy*Async` on
  one stream, a ring of events made once), or any copier whose completion is a lane's `Marker` (`MarkerRetired`);
- `Moves`: a `LatticeSource` whose frames deliver completions to the residents' owners and drive the moves;
- planner: `Claim::transfer` (a claim changes owner without leaving its location's total), `Claim::is_elastic`,
  `Elastic::into_claim`.

**Why.** Planning where memory lives is half the job; the other half is moving it when the plan changes: a
higher-priority claim arrives, a consumer turns off, a budget shrinks. Each consumer copying its own buffers to host
memory on its own repeats the race the planner removed. One mover does every move under the planner's accounting.

**How to use it.** Allocate each tier once and give it to a `TierPool` under one owner. Make a `Mover` with the pools
(fastest first), the copiers and an optional staging tier. `place` residents with a priority and a completion handler;
`move_to` moves one; `with` reads or writes it while it is still; `release` gives it up (and cancels a move). Call
`pump` from any thread, or add `mover.moves()` upstream of a lattice. Install `give_back_handler()` as the pools'
owner's give-back handler and the handler of residents' owners, and call `restore` when budget frees up.

**Semantics.**
- No locks: residents are cells of atomics (free, resident, moving, busy); moves, free slots and completions go
  through lock-free queues. Nothing waits: a copy's completion is a test (`Retired`) checked by `pump`.
- Exact accounting at every point: a move leases the destination slot (its bytes become the resident owner's) before
  the copy starts and returns the source slot after it retires. A location's total never changes during moves.
- A resident placed in a full tier evicts that tier's lowest-priority resident below it to the next tier with room.
  Spilled residents come back home, highest priority first, whenever their home tier has a free slot.
- Slots of residents that can spill (not in the slowest tier) are elastic holdings of their owner. A give-back order
  at a tier retires free slots first, then spills the lowest-priority residents; a spill for an order retires its
  source slot rather than freeing it, so bring-back cannot refill it. `restore` takes retired slots back when the
  location has room.
- A pair no copier joins goes through the staging tier in two hops; a move that cannot continue leaves the resident
  in the staging slot.
- Releasing a resident mid-move marks it cancelled: when the copy retires, both slots go back and the owner hears a
  cancelled `MoveDone`.

**Tests.**
- 5 tests without a device (a copier whose copies retire a few checks late): moves between every pair with the bytes
  intact and the accounts exact while in flight and after; a move released midway; eviction by priority and
  bring-back; a give-back order that retires a free slot, spills a resident and gets its budget back; eight owners
  placing, filling and moving at once with no lock.
- 1 test on a device (feature `cuda`): one resident from pageable host to device to pinned host and back through stream
  copies and two-hop staging, bytes intact, accounts exact at each step.

## Memory planner: `euhedral-memory` (2026-10-28)

**What.** A new crate, [`euhedral-memory`](euhedral-memory/README.md):
- a location-agnostic `Planner`: budgets per location, owners with priorities and scopes, firm claims, elastic
  holdings, give-back orders, and a leak check;
- admission of a plan's whole working set before it allocates (`admit`: spill, then cut, then refuse);
- arenas (`SlotArena`, `ScratchArena`) that hand a session's planned memory out and back without the driver, with
  `BackendCalls` to prove the steady state allocates nothing;
- CUDA allocation backends (feature `cuda`): `DeviceMemory`, `StreamPool`, `Vmm` with POSIX-fd export, `PinnedHost` with
  a capacity cache, and `Device::import_fd`.

The rest of the workspace is unchanged.

**Why.** Consumers that size themselves from the free memory, each on its own, race: the last to allocate fails, and a
leak shows only as an out-of-memory error somewhere else. One planner per process accounts every location and owner,
admits a plan before it allocates and orders give-back by priority: "a pool where you can claim or free memory on a
device or the host".

**How to use it.** Make one `Planner` with the process's locations. Open an owner per component and lifetime.
- `admit` a plan before building it, and allocate from its `Reservation` once.
- Lease slots and scratch from arenas per unit of work.
- Drop the arenas (and so the claims) at the session's end, then `end_scope`.
- Add `planner.give_backs()` upstream of a lattice so give-back orders run as frames.

**Semantics.**
- No locks.
  - Claims are CAS loops on a location's `used`, which never exceeds capacity less reserve.
  - Owner slots are recycled through a lock-free queue once an ended owner's last claim drops. The slot's generation
    changes, so a stale `OwnerId` is refused.
  - A claim pins its slot; an order reads its owner's handler only under a pin.
- Give-back takes only from strictly lower priorities: lowest first, then the biggest holder. An elastic holding does not
  grow while an order to its owner is open. Freed elastic bytes count against the open order first.
- `end_owner` ends the owner's live descendants too. What they still hold is reported, and stays accounted until those
  claims drop (the memory is still allocated).
- Admission places an item's bytes per location. It keeps everything preferred if that fits in free plus reclaimable.
  Otherwise it spills items lowest priority first, each only if the spill location has room, then drops optional items
  lowest priority first, else returns `Shortfall`.
  - With give-back needed, it returns the orders and no reservation.
  - Otherwise a `Reservation` holds firm claims per location, and `take` moves bytes from it to an allocation's owner.

**Tests.**
- 14 tests without a device: claims, reserve, elastic give-back and its frames through the `GiveBacks` source, priority,
  leaks, baseline, admission fit / spill / cut / refuse / reclaim, the reaper, slot recycling under 1000 owners, eight
  threads claiming at once, and arenas over 10,000 units of work with no backend call.
- 3 tests on a device (feature `cuda`): every backend at 256 B to 100 MiB, VMM fd export, and pinned cache reuse.
- The `alloc_sweep` example measures CUDA against Vulkan allocation, import and copy on the local GPU.
## `Completions` and the io_uring ring take registrations through `MpscQueue` (2026-10-26)

**What.** In `euhedral-graph`, `Completions`' registrations (`after`, `after_marker`, `at`, `register`) go into a
`euhedral_data_structures::queues::MpscQueue` instead of a `Mutex<Vec>`, and an `AsyncIo` io_uring ring's entries that were
queued while another thread held its `sq` lock go into one instead of a `Mutex<VecDeque>`. The crate now depends on
`euhedral-data-structures`. `held`, `Completions`' waiting list, `Sequencer`, `Join`, the retry counts and the Windows
engine's backlog are unchanged.

**Why.** A thread registering a completion or a ring entry no longer takes a lock the polling thread also takes; the
poll no longer waits behind a registrant, and the registrant never waits behind a poll.

**Semantics that changed.** A registration that races `complete`. Before, `complete` set its flag under the registration
lock, so a registration either landed or was refused on the spot. Now `complete` sets the flag and appends one marker to the
queue. A registration that sees the flag is refused on the spot, as before (`do_finally`, a panic in debug builds). One that
does not is queued, and its position against the marker decides: ahead of it, it is a registration like any other and is
emitted; behind it, the next poll (`pull` or `request`, also of a completed source) cancels it through `do_finally` without
running it, and so does the last drop of the source if no poll comes. It ends exactly one way, never both and never neither;
that cancellation does not panic. The argument is on `Completions::complete`.

**The ring.** The `sq` holder drains the queue before it lets go, and looks again after it does; `flush_waiting` has the
argument that an entry pushed just after the holder's last drain is submitted by its pusher taking `sq` or by that re-check.
The flag that says "something was queued" stays, set after the entry is published: a queue's size counts reserved slots,
so it cannot tell a producer between reservation and publish from a hidden entry, and the crate does not spin.

**Tests.** `Completions`: a registration ahead of the marker is emitted and one behind it is cancelled; a `request` of a
completed source and the last drop cancel what is behind the marker; a registration racing `complete` (1500 rounds) ends
exactly once; 8 threads x 10 000 registrations against a hand-driven poller are emitted once each, in per-thread order.
Ring: entries queued while another thread holds `sq` are never stranded (1000 rounds of 4 threads against a slow enter,
the last pushes together so no later push can rescue one). The existing backlog and refused-enter tests pass unchanged.

## The branches of a fan-out run on different device lanes (2026-10-26)

**What.** A stage graph now places stages on lanes by Euhedral-Inference's PATH placement
(`StageGraph.markPaths`): each stage's lane is continued by exactly one successor, its critical
successor, and every other successor of a fan-out takes a lane in turn. `Topology` computes this once,
when the graph is built, as `path_predecessor`.

**Why.** `path_predecessor` was the predecessor on a stage's longest path from a root, so in a fan-out
A -> {B, C} both B and C continued A's lane. The branches never spread over lanes and ran one after the
other on the device, although the docs say a fan-out's other branches take a lane in turn.

**How to use.** Nothing changes in the graph's API. Give the branches of a fan-out several allowed
lanes in `StageSpec::lanes`; with a pool of two or more lanes they now run on different ones, and a join
waits (device side, in `StageCx::waits`) for the markers of the predecessors whose lane it does not
continue.

**Semantics.**
- Only submission edges place lanes; a completion edge never carries one.
- A stage's critical successor is its successor of greatest height (the stages on its longest
  submission path to a sink). Of equal heights, the first listed (in edge order) wins.
- A stage continues the lane of the predecessor it is the critical successor of. Stages are visited
  sinks first in a first-in first-out order (sinks by stage index; a stage is queued when its last
  successor was visited, and its predecessors are taken in stage index order), so a join that is the
  critical successor of several predecessors continues the one visited last. A join that is the
  critical successor of none continues no lane.
- The Java `transfers()` rule (a transfer stage never continues a compute chain) has no counterpart: a
  stage's allowed lanes already say which lanes it may use, and a continued lane outside them is
  ignored.
- Frame routing, id hashes and sinks are unchanged; this concerns the device lane a stage submits to.

**Tests.** In `stage_graph`: `fan_out_runs_both_branches_and_fan_in_runs_the_join_once_behind_the_other_lane`,
`a_fan_out_spreads_its_branches_over_the_lanes_and_the_join_waits_for_the_other_branch`,
`a_chain_keeps_one_lane`, `a_wider_fan_out_uses_every_lane`,
`a_longer_branch_continues_the_lane_whatever_its_place_in_the_edge_order`,
`of_equal_heights_the_first_listed_successor_continues`,
`a_strictly_greater_height_beats_an_earlier_successor`,
`a_join_continues_the_last_visited_of_the_predecessors_it_is_critical_for`,
`a_join_continues_only_the_predecessor_it_is_the_critical_successor_of` and
`a_completion_edge_never_carries_a_lane`.

## A fragment closes only its own use of its CPU's executor (2026-10-26)

**What.** `PinnedThreadExecutor` handles are now counted uses of the CPU's one shared executor, and
`ControlPlaneFragment::close` and `dump_locks` end only their own fragment's use: they interrupt the
fragment's worker task and `release` the handle. The executor closes when its last use ends. Fragments
of different lattices on one CPU, and a shard's replacement clone on a core whose old clone is
closing, no longer interrupt each other's workers.

**Why.** `close` and `dump_locks` called `PinnedThreadExecutor::close`, which closes the executor for
every user for good and interrupts every task on it. A second lattice on the same CPU lost its worker
to the first one's close, and its next `start` panicked ("a closed executor cannot restart"). In a
shard, a clone that failed to drain in time was replaced on the same core; the replacement shared the
running executor, and the deferred retirement of the old clone then closed it. The replacement's worker
left on `is_interrupted()` while its fragment still counted as running, so the core stopped without a
sound and its frames hung.

**API.**
- `PinnedThreadExecutor::release(&self)`: ends this handle's use. Idempotent; dropping the handle
  does it too.
- `PinnedThreadExecutor::submit_interruptible(task)`: `submit`, plus a `TaskInterrupt` whose
  `interrupt()` sets the `is_interrupted()` flag of that task alone and whose `is_finished()` says
  the task has ended and left the executor.
- `PinnedThreadExecutor::close` keeps its meaning: closed for every user. A user that only wants to
  stop its own work does not call it.

**Semantics.**
- A use is a handle. `get_or_create` and `get` each return a new use of the live executor; `Clone`
  adds one. A shut-down executor is restarted by a `get_or_create` as before. A handle joins an
  executor in one step with the check that it is not closed, so a use never lands on an executor that
  has just closed.
- When the last use ends the executor closes as `close` does (interrupting any task still on it) and
  its registry slot is freed once no task is left.
- A fragment's worker task has its own `TaskInterrupt`, set on `close` and on `dump_locks`. The
  fragment's other users of the CPU keep running, and `start` on a fragment sharing the executor finds
  it running.
- When the last use ends while its worker is still finishing, the executor is closed with a task left, and
  its slot stays taken until that task ends. A re-placement on that CPU in the meantime gets
  `ClosedWithActiveTasks`, and `BaseCloneableObject::place` panics on it. This limit of the registry is
  unchanged; closing a lattice waits for its workers (which leave the registry just before the task
  ends), so only a quick re-placement from another thread can meet it.
- `BaseCloneableObject::place` no longer shuts the executor down after the build: that interrupted
  every task on a shared executor, and a build racing another lattice's start could stop its worker.
  The executor is created under the fragment's worker name and left as found.

**Tests.** `executor_closes_after_last_user`, `releasing_a_user_leaves_the_other_users_tasks_running`
and `a_task_interrupt_reaches_its_own_task_only` (`euhedral-hardware-utils`);
`replacement_clone_on_same_core_keeps_running` and `dumping_locks_interrupts_only_the_fragments_own_worker`
(`fragment`); `two_lattices_on_one_cpu_close_independently` (`base_cloneable`, three lattices on one CPU
closing in turn).

## Lakes, latches and stage graphs over any frame sink (2026-10-26)

**What.** `euhedral-graph` gets `Lake` (with `UnitGuard` and `LakeClosed`) and `Latch` (with `LatchSetter` and `LatchWaiter`).
`GraphRuntime::stages` is now an `Arc<dyn FrameSink>` instead of an `Arc<QueueIngestSink>`, so a stage graph publishes into
a lake (or any other sink) through `offer`. The workspace gets one CI workflow, `.github/workflows/ci.yml`.

**Why.** A program has several kinds of ready work (stage frames, host work, control), each of which wants its own pool, spread
over several sources so producers and workers stay off one cache line, and attached to the lattice only while there is work,
because a worker polls every attached source for as long as it is attached. And the main thread needs one place to wait for
the lattice to hand it a result.

**Usage.**
- `Lake::new(lattice, sinks, idle_release)` makes a lake of `sinks` queue ingest sinks of one partition each.
  `Duration::ZERO` as `idle_release` never detaches them.
- `lake.admit()` counts a unit in (a quantum, a host task), attaches the sinks if they are detached, and returns a `UnitGuard`;
  dropping the guard terminates the unit. `UnitGuard::fork()` admits another unit even while the lake drains.
- `lake.offer(frame)` (also `FrameSink::offer`) queues the frame in the sink `routing_hash % sinks` picks, or hands it back.
- `lake.complete_gracefully()` closes admission.
- `Latch::new()` returns a `LatchSetter` to move into a frame and a `LatchWaiter` for the main thread; `wait` parks until
  `set`.
- `GraphRuntime { stages: lake.clone(), .. }`. The lake's units are the graph's quanta: admit before `StageGraph::start` and
  drop the guard in `retire`, or hold one for as long as the graph runs.

**Semantics.**
- A frame goes into one sink, by `routing_hash` taken as unsigned modulo the sink count. Frames of one hash therefore keep their
  offer order, and a sink has one partition, so a sink delivers in offer order; an ordered frame still runs in its own chain's
  order, and two chains in one sink may run at the same time.
- A frame is accepted only while a unit is admitted: with none, `offer` hands the frame back and attaches nothing. A refused
  frame is the caller's to end (the stage graph ends it in place).
- Admission counts units in one word with a closed bit. `complete_gracefully` sets the bit; admission then fails with
  `LakeClosed`, units already admitted may still offer frames and fork units, and when the last one terminates every sink
  completes gracefully (it delivers what it holds, then completes and detaches) and the idle watch ends. A lake with no unit
  when it is closed completes at once. After that `offer` hands every frame back.
- The sinks attach to the lattice at the first `admit`, not at `Lake::new`. The attachment is a *generation*: the sinks and
  their idle watch. The watch is a `LatticeSource` that holds no frames and checks the lake whenever a worker pulls from it or
  requests from it (two atomic loads while a unit is admitted). When no unit has been admitted for `idle_release`, it takes
  the generation out of the lake, completes its sinks gracefully and completes itself, which detaches all of them. The next
  `admit` makes a new generation (`Lake::attachments` counts them). `admit` counts the unit before it asks for a generation,
  and the watch checks the count again under the lock that takes the generation, so a unit never loses its sinks.
- A frame that races a detach (its sink completed first) goes to the generation attached since, as long as a unit is admitted;
  an offer re-checks the unit count under the lock before it makes a generation, so a unit that ended meanwhile leaves no sinks
  attached. `Lake::is_attached` is true until every sink, including those of a detached set that is still draining, has completed.
- Attaching registers the new sources with the lattice outside the lake's lock (registration asks the source for demand, which
  polls the watch). A lattice that refuses (closed) fails `admit` with `LakeClosed::Lattice` and the half-attached generation
  cancels what it holds.
- A lake dropped before it completed lets its sinks drain and complete.
- `Latch::wait` is the one blocking call: it parks the thread until the value is set, returning at once if it was set before.
  A `LatchSetter` dropped without setting makes `wait` panic instead of hanging.

**Tests.** `tests/lake.rs`, on a lattice of one CPU (the first the process may run on, so the tests assume no second core):
`frames_of_one_hash_keep_order`, `sinks_detach_after_idle_and_reattach_on_admit`, `a_lake_with_no_idle_time_never_detaches`,
`offer_without_admitted_unit_is_refused`, `complete_gracefully_drains_then_completes` (with a fork), `latch_set_before_wait_returns`,
`latch_set_from_frame_wakes_waiter`, and a latch whose setter was dropped. They take turns (two lattices of one CPU running
together made a rebalance panic with "cannot restart the executor"). `tests/credit_graph.rs` publishes its stage frames through
a lake of two sinks. The stage graph's unit tests publish through a queue sink behind a small `FrameSink` adapter.

## `AsyncIo::read_into`: reads into caller-owned memory (2026-10-26)

**What.** `AsyncIo::read_into(file, offset, buf, then)` reads into memory the caller owns, such as a pinned host
arena, instead of an `IoBuf`'s `Vec`. `buf` is any `IoMemory`; `then` receives an `IoDoneOwned { buf, result }`. The
engines (io_uring, IoRing and the completion port) work on a view of the region, so nothing is copied.

**Why.** The inference engine reads weights from disk straight into host memory it allocates and pins itself.

**API.** `pub unsafe trait IoMemory: Send + 'static { as_mut_ptr, len, is_empty }`, implemented for `Vec<u8>`;
`IoDoneOwned<B> { buf: Option<B>, result, filled }`. `read` is unchanged.

**Semantics.**
- The result is a `read`'s: `Ok(n)` with `n` the region's length, or less if the file ended after some bytes (the first
  `n` bytes are the file's, the rest untouched); `UnexpectedEof` when no byte was read. An empty region is refused at
  submission, with nothing started.
- The region is cut into pieces of at most 256 KiB, as a `read`'s buffer is; each piece lands at its own offset and
  the engines' short-transfer and slot-identity rules apply unchanged.
- `then` always runs. The value comes back in `IoDoneOwned::buf` after a success and after an error alike. `buf` is
  `None` only for an operation the engine cannot prove finished with the kernel, which already leaked its buffer on
  Windows (a ring that broke, a ring entry that may have reached the kernel): the value is leaked (never dropped) and
  `result` is the ring's failure. A submission that returns `Err` started nothing and drops the value.
- `filled` is the count of bytes at the start of the region that arrived: `n` for `Ok(n)`; after an error, the bytes of a
  Windows read that failed with a warning status (`ERROR_MORE_DATA`) and moved bytes anyway, as `IoBuf::filled` reports.
- `IoMemory` is the only new `unsafe` surface; its contract (initialized, stable when moved
  without claiming unique ownership, exclusive until returned) is on the trait.

**Tests.** `tests/io_owned.rs` (`read_into_fills_owned_region`, `read_into_split_pieces_preserve_offsets`,
`read_into_short_read_reports_count`, `owned_buffer_returned_on_error`); in `io/uring.rs`,
`reads_into_parts_of_one_arena_land_in_place`; in `io/win/tests.rs`, the owned reads through the simulated ring and port
and the leak of a broken ring's region (the continuation runs with `buf` `None`); in `io/win/live.rs`, a read into owned memory on a real Windows.

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

**Tests.** 45 tests run on every platform against a simulated `Sys` (every operation's success, error and short transfers,
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
