# euhedral-graph glossary

The implementation-level terms of [`euhedral-graph`](README.md): the mechanisms under completions, lanes,
lakes and stage graphs. What a user composes (`Completions`, `Sequencer`, `Lake`, `LaneSource`, stage
graphs) and the lifecycles are in the [top-level glossary](../GLOSSARY.md). Paths are relative to this
crate. Entries run from the lowest layer up: completions and sequencing, lanes and their sources, lakes,
stage graphs.

## Completions and sequencing

**late registration** ([`Completions`](src/completions.rs)): a registration that lands behind the marker
`complete` appended to the registration queue. The next poll (`Completions::reap`, the poll frame's work, or a late
registration's own sweep) finishes its frame through `do_finally` without running it, and so does the last drop of
the source if no poll comes. It does
not panic: only the polling thread sees it, and it cannot tell a misuse from a poll that was late.

**leaked registration** ([`Completions::leaked_unreached`](src/completions.rs)): a registration that waits
for a lane's marker the lane has still not reached, and is refused, late or abandoned. It is leaked
(`mem::forget`) and counted, because the device may still use what its frame holds, and finishing the frame
would release it.

**progress** ([`Progress`](src/sequencer.rs)): an entry's readiness word: pending, ready or abandoned. The
finishing thread stores it with release ordering after it wrote whatever the entry carries, and the thread
that concludes reads it with acquire ordering. An abandoned entry is ready and concludes with nothing; the
owner of an entry abandons it when it dies, or the order waits for it, and only a `Slot` does that by
itself.

**drainer** ([`Sequencer::drain`](src/sequencer.rs)): the one thread at a time that concludes the prefix of
ready entries, in offer order, on its own thread. A work-in-progress counter admits it; a thread that finds
the counter busy leaves its work to the drainer, and a completion that arrives during a drain makes that
drain look again.

## Lanes

**LanePoolBuilder** ([`LanePoolBuilder`](src/lanes.rs)): composes a `LanePool` from the queues of several
backends. A backend `reserve`s a range of lane ids, opens its queues at them and `set`s each one; `build`
makes the immutable pool. A pool has 1 to 64 lanes.

**order key** ([`LaneWork::order`](src/lane_source/mod.rs)): items of one order key run in delivery order,
even across a wait.

**intake, shared intake** ([`LaneIntakes`](src/lane_source/intake.rs)): where delivered work waits. A lane's
own intake is an MPSC queue, so delivery order is kept per lane. A lane group's shared intake is an MPMC
queue that every source of the group takes from, one item at a time, so unbound work runs on whichever
source of the group a worker enters first. Delivery only enqueues: it spends no demand and runs nothing. A
lane that completed or was dropped refuses the work and hands it back.

**lane group** ([`GroupId`](src/lane_source/intake.rs)): lanes whose sources share one intake. A lane is in
at most one group, and a group may hold lanes of several backends.

**banked demand** ([`LaneSource::banked`](src/lane_source/source.rs)): the demand a lane source has been
asked for and has not yet spent on a push. `request` always adds to it, even with nothing ready, and a push
subtracts what it pushed; `pull` never touches it. `u64::MAX` saturates the bank; a spend floors at zero.

**waits for** ([`Outcome::Wait`](src/lane_source/mod.rs), [`LaneState::met`](src/lane_source/mod.rs)): an
item that cannot go on until the lane's own hardware state allows it answers `Wait`. It stays in the
source, the source asks `met` once per request, and items with the same order key wait behind it.

## Lakes

**idle detach** ([`Lake`](src/lake.rs)): a lake's sinks stay attached while a unit is admitted and for the
idle time after the last one ended; then the watch detaches them, and the next unit attaches a fresh set.
`Duration::ZERO` keeps them attached for life. Reading the attached set takes no lock; attaching and
detaching are rebalance work.

**busy period, idle watch** ([`Lake`](src/lake.rs)): the sinks of a lake are attached to the lattice only
while it has units. Beside the sinks a lake attaches an idle watch, a source the workers poll that holds no
frames; when no unit has been admitted for the lake's idle release time, the watch pushes one frame (its key
frame) and the worker that runs it detaches the sinks and the watch, so the lattice has nothing left to poll. The next `admit` attaches a fresh set. An idle release of
`Duration::ZERO` keeps the sinks attached.

## Stage graphs

**live count** ([`StageGraph::live`](src/stage_graph.rs)): the quanta started and not yet retired. Each
quantum also counts its own published frames, armed completion edges, host completions no edge consumed,
external edges not yet arrived and admission's hold; at zero no stage can run again.

**retirement boundary, retirement** ([`StageGraph`](src/stage_graph.rs)): the latest marker per lane of
everything a quantum submitted (and of later quanta's stages that used it through external edges, and of
external edges into stages that never ran). A quantum retires, handing itself and its outcome to the graph's
`retire` callback, once none of its frames can run again and the boundary is reached. The boundary is armed
as one `Completions` edge per lane used; a host completion no completion edge consumed holds the quantum live
until it is ready, so the quantum retires after it.

**typed lane pool, lane handle** ([`LanePool`](src/lanes.rs), [`StageCx::lane_handle`](src/stage_graph.rs)): a pool is typed
over its lane, as Java's `LanePool` holds `GpuStream`s. The default, `dyn DeviceQueue`, is the query side only; with a
backend's shared lane type, a stage gets the lane it was given from its context and submits to it from whichever
worker runs it. `LanePool::as_queues` is the query view that `Completions` polls.

**placement, home lane** ([`Placement`](src/lanes.rs), [`LanePool::next_home`](src/lanes.rs)): how a stage that continues
no lane picks one of its allowed lanes: at random from a per-thread generator (Java's `ThreadLocalRandom`; no shared
counter), or by a function of the stage (`Fixed`, Java's `fixed`: it overrides the continued lane too and is deterministic under any number of workers). A root
continues its run's home lane, taken in turn over the pool's compute lanes (not the reserved ones) when the run is made
(Java `nextHome`), so a graph's critical chain keeps one lane in every quantum.

**lane control, retire cell, quiesce** ([`LaneControl`, `RetireCell`](src/lanes.rs), [`StageGraph`](src/stage_graph.rs)):
what a graph asks of a lane it joins and retires: `await_marker` (later work on this lane waits on the device for a
marker of another lane), `notify_retired(cell)` (the backend's callback stores the work's status in that cell and does
nothing else) and `confirm_retired(status)`. `RetireCell` is one atomic: the status, or pending. The graph's boundary
(Java `quiesce`) has the other lanes a quantum used join its home lane, arms one retirement there with the quantum's own
cell, and registers `Completions::after_retired(cell, frame)`; the frame settles the status, runs the `retired` hooks and
retires the quantum. Each arm names its own cell, so parties arming on one lane never read each other's retirement, and
nothing is claimed.

**stage gate** ([`Stage::pass_gate`, `Gate`](src/stage_graph.rs)): Java `StageFrame.passGate`/`releaseGate`. When a stage's last incoming edge arrives it is asked whether it may be published now; a stage that holds itself takes a `Gate` and the thing it waits for releases it, once, from any thread that may publish frames. The held stage keeps the quantum live; a gate that fails, panics or is dropped unreleased fails the quantum and drops the count.

**hold from run** ([`StageCx::hold`, `StageError::Held`](src/stage_graph.rs)): a stage holds itself at its gate from inside `run`, when it finds there that what it needs is not ready. It takes the gate with `cx.hold()` and answers `Held`; the release runs it again exactly once. A release that comes before `run` returns publishes the stage when `run` returns, never beside it. It registers nothing with the completions, so it is no retry and allocates nothing.

**stage harness** ([`testing::StageHarness`](src/testing/stage.rs)): calls one custom stage's hooks (`run`, `pass_gate`, `abandoned`) on the calling thread with a context the test states (stage id, lane, waits, pool), and returns what the stage answered (and what a graph would make of it), whether it took a gate and how the gate was spent. No graph, lattice or worker; a released gate runs nothing. Feature `testing`.

**abandoned hook** ([`Stage::abandoned`](src/stage_graph.rs)): Java `abandoned(home)`. Once per stage that never submitted, in stage order, at the quantum's quiesce: after the other lanes joined the home lane, before later quanta waiting on its stages are released and before the retirement is armed on home, with the home lane.

**retired hook** ([`Stage::retired`](src/stage_graph.rs)): Java `StageFrame.retired`. Once per stage that was attempted,
in stage order, after the quantum's device work ended and before the quantum is handed on: the place to release what
the stage held for its work. `committed` is true only when the whole quantum succeeded and the stage submitted.

**pooled quantum, run** ([`StageGraph`](src/stage_graph.rs)): the state of one quantum (its counts, its
markers, its first cause, its external-edge records) kept in a bounded pool and rebound to each quantum in
turn. `start` takes one and `retire` gives it back, as the graph's frames (stage, arrival, boundary and
retirement frames, one pooled type) go back to a pool of their own. A pool that is full drops what comes
back. A take retries the queue's try-claim for as long as the pool holds anything, up to 65536 polls, and
makes a new object only when the pool is empty (or stays unpollable that long): the pool grows to the
objects in flight at once, and no further.

**TakeAllStack** ([`take_all`](src/take_all.rs)): a stack any thread pushes to and one thread takes whole and
seals. The nodes are the caller's own, pooled objects (each holds its link), so a push allocates nothing.
The seal names why the stack closed (`Submitted`: the stage submitted; `Swept`: its quantum quiesced
without it), and a push after it is refused with that reason. Every node ends in exactly one place: in the
list a seal returned, or refused. A stage graph keeps the external edges waiting for a stage on one.

**retry backstop** ([`GraphRuntime::retry_backstop`](src/stage_graph.rs),
[`DEFAULT_RETRY_BACKSTOP`](src/stage_graph.rs)): a stage that answers `StageError::Retry` runs again when
any lane of the pool made progress, or after this deadline (20 ms by default) for a resource outside the
pool. The deadline is checked by a `Completions` poll. Set it longer than any one pass on your devices, so
that it fires only when no lane completed anything for that long.
