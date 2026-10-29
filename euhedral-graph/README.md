# euhedral-graph

No-wait dataflow on the RustEE lattice: completions, in-order release, joins,
device lanes and per-item stage graphs. Nothing in it waits and nothing in it owns a thread. See
[`GLOSSARY.md`](../GLOSSARY.md) for every term used here.

Asynchronous I/O (`AsyncIo`) is in [`euhedral-io`](../euhedral-io/README.md).

**Where it sits.** It depends on [`euhedral-core`](../euhedral-core), on
[`euhedral-data-structures`](../euhedral-data-structures) and, on Linux and Android, on `libc`.
Nothing in the workspace depends on it; applications do. It adds
`LatticeSource`s that the lattice's own polls drive, and plain structures that frame callbacks run,
so it needs no thread of its own. It does not re-export `euhedral-core`: import `Frame`,
`LatticeSource` and the rest from there.

## Modules

| Module | Main types |
|---|---|
| [`completions`](src/completions.rs) | `Completions`, a `LatticeSource` of continuation frames behind completions: markers, timers, `Ready` tokens and `Pending` operations. |
| [`sequencer`](src/sequencer.rs) | `Sequencer`, `Entry`, `Progress`, `Ticket` and `Slot`: execute in any order, conclude in offer order (Java's order queue and work-in-progress drain; any thread offers, no lock, no refusal). |
| [`join`](src/join.rs) | `Join`, `Round`: a re-armable fan-in cell; the continuation runs when its last part arrives. |
| [`lake`](src/lake.rs) | `Lake`, `UnitGuard` and `LakeClosed`: the pool of ready work of one kind, as several ingest sinks that attach to the lattice for a busy period; offering reads the attached set with no lock. A `FrameSink`. |
| [`lanes`](src/lanes.rs) | `LanePool`, `LanePoolBuilder`, `LaneId`, `Marker` and the trait `DeviceQueue`: device timelines that are queried, never waited on. |
| [`lane_source`](src/lane_source/mod.rs) | `LaneSource`, a `LatticeSource` per device lane; the traits it drives (`LaneSubmit`, `LaneState`, `LaneWork` with `Outcome` and `LaneCx`); `LaneIntakes`, `LaneInputs`, `Deliver` and `GroupId` for delivery; `LaneMetrics`. |
| [`stage_graph`](src/stage_graph.rs) | `StageGraph` and its spec (`StageGraphSpec`, `StageSpec`, `StageId`, `EdgeKind`), the `Stage` trait with `StageCx`, `Submitted` and `StageError`, `QuantumOutcome`, `ExternalEdge`, `QuantumRef`, `GraphRuntime`, `RetryCounts`, `GraphError` and `DEFAULT_RETRY_BACKSTOP`. |
| [`testing`](src/testing.rs) | `FakeDevice`, an in-order fake `DeviceQueue`, and `FakeLanes`, a stepped fake `LaneSubmit`. Feature `testing`. |
| crate root | Every type above, re-exported; `ThreadUsage` (the calling thread's CPU time and involuntary context switches, to tell a preempted callback from a slow one; measured on Linux and Android). |

## Key concepts

### Nothing waits

Work is **submitted** and the call returns at once. What follows it starts on one of two events:

- the **submission**: a device successor orders itself behind its predecessor's `Marker` on the
  device (a wait inside its own submission), so the host never waits;
- the **completion**, found when a lattice worker polls a source. The continuation is a frame the
  source emits, and the lattice runs it like any other.

`Completions` and `euhedral-io`'s `AsyncIo` are `LatticeSource`s, so they attach to a lattice with `add_upstream`
and the workers' own polls (`pull` and `request`) query each pending operation once. There is no
timer thread, no completion thread and no channel. A poll never blocks: a lane's completed value
is one atomic read (`DeviceQueue::completed`), and an io_uring completion queue is read without
entering the kernel.

The crate's clippy bans (see [`clippy.toml`](clippy.toml)) keep it that way: no threads, sleeps,
channels, `Condvar`, blocking `std::fs` or `std::net` calls.

### Completions

A `Completions` holds frames to emit once something completes:

- `after_marker(marker, frame)`: a device lane reached a value;
- `at(deadline, frame)`: a timer, which is just a deadline token;
- `after(ready, frame)`: a `Ready` token (`is_ready`, one cheap query) said so;
- `register(pending)`: a `Pending` operation is polled until it yields its continuation frame, for
  a completion that carries a result.

Continuations are emitted in registration order, so frames on one ordered lane keep that order. A
frame whose kill switch is set is still emitted: the executor skips it and runs its `do_finally`,
which ends what it holds, so `pending()` drains to zero on its own. `lateness_us()` reports how
late emissions were: the cost of finding completions by polls.

One poll runs at a time and none waits. A `request` that finds a poll running banks its demand and leaves a wakeup,
and the running poll pushes within the demand as it lets go; the same holds for the sweep that cancels a
registration made after `complete`.

This runs one by hand, with a `Sequencer` that concludes in offer order whatever order the
items finished in:

```rust
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use euhedral_core::frames::{Frame, RunnableFrame};
use euhedral_core::generics::LatticeSource;
use euhedral_graph::testing::FakeDevice;
use euhedral_graph::{Completions, Sequencer};

// The pool's lanes are only polled for markers; a timer needs none.
let (lanes, _devices) = FakeDevice::pool(&["compute"]);
let completions = Completions::new(lanes);

let released = Arc::new(Mutex::new(Vec::new()));
let sequencer = Sequencer::ordered({
    let released = released.clone();
    move |seq: u32| released.lock().unwrap().push(seq)
});

// Items 0, 1, 2 are admitted in that order; their timers fire in the order 1, 2, 0.
let fired = Arc::new(Mutex::new(Vec::new()));
let now = Instant::now();
for (seq, after_ms) in [(0u32, 6), (1, 0), (2, 3)] {
    let slot = Mutex::new(Some(sequencer.register()));
    let fired = fired.clone();
    let task = Arc::new(move || {
        fired.lock().unwrap().push(seq);
        if let Some(slot) = slot.lock().unwrap().take() {
            slot.complete(seq); // the sequencer concludes it once every earlier entry did
        }
    });
    let continuation = Box::new(RunnableFrame::new(7, task));
    completions.at(now + Duration::from_millis(after_ms), continuation);
}

// A lattice worker does this on its own polls; here the loop is the worker.
while completions.pending() > 0 {
    completions.pull(
        &mut |mut frame| {
            frame.execute().unwrap();
            frame.do_finally();
        },
        &mut |_| false, // never stop before a frame
        u64::MAX,
    );
    std::thread::sleep(Duration::from_millis(1));
}

assert_eq!(*fired.lock().unwrap(), [1, 2, 0]); // finished out of order
assert_eq!(*released.lock().unwrap(), [0, 1, 2]); // concluded in offer order
```

### Device lanes and markers

A `DeviceQueue` is a device timeline that runs submissions in order: a Vulkan queue with a timeline
semaphore, a CUDA stream with events, a hardware codec engine, or a fake. What is generic is the
query, `completed()`, one cheap read. A `Marker` names the queue that made it (`QueueId`, from `DeviceQueue::id`), a lane and a value on it;
a `LanePool` holds the lanes and answers `reached(marker)`, which is false for a marker of another queue. Submitting is technology-specific and lives with whoever
owns the queue: it takes the markers its work must wait for on the device and returns the marker
of the new work.

### A lane is a source

A device lane's submitter has one owner, so a lane runs as one `LatticeSource`, a `LaneSource`. Work is
**delivered** to a lane's own intake or to its group's shared intake (`LaneIntakes::deliver`), which only
enqueues. The lattice enters the source, and each `request(demand)`:

1. banks the demand;
2. runs the continuations of the retirements that arrived;
3. reads the lane's hardware state once (`LaneState::check`);
4. runs every item that is ready, the released waiting items first, then the lane's own intake, then the group's
   shared intake. There is no budget: a long item runs on the worker that needs it;
5. pushes at most the banked demand of the frames the items made.

An item that cannot go on answers `Outcome::Wait`; it stays in the source and items of its order key wait behind it. An item that could not do its work (a submission the device refused) answers `Outcome::Failed(error)`: the source calls `failed` and delivers no successor.
`pull` hands over ready frames and never submits work. The example drives one lane by hand with the stepped fake:

```rust
use std::sync::Arc;
use euhedral_core::frames::{Failure, Frame, FrameHeader};
use euhedral_core::generics::{LatticeReceiver, LatticeSource, LatticeTerminal};
use euhedral_graph::testing::{FakeLane, FakeLaneError, FakeLanes};
use euhedral_graph::{
    Deliver, LaneCx, LaneId, LaneIntakes, LanePoolBuilder, LaneSource, LaneWork, Marker, Outcome,
};

/// An item that submits, then queues one frame.
struct Item(u64);

impl LaneWork<FakeLane, ()> for Item {
    const KINDS: &'static [&'static str] = &["item"];
    fn kind(&self) -> usize { 0 }
    fn waits(&self) -> &[Marker] { &[] }
    fn run(&mut self, _cx: &mut LaneCx<'_, FakeLane, (), Self>) -> Outcome<std::convert::Infallible, FakeLaneError> {
        Outcome::Submitted { retire: false }
    }
    fn then(&mut self, _marker: Option<Marker>, cx: &mut LaneCx<'_, FakeLane, (), Self>) {
        cx.ready(Box::new(Out(FrameHeader::new(self.0))));
    }
    fn failed(self, _error: FakeLaneError, _cx: &mut LaneCx<'_, FakeLane, (), Self>) {}
}

struct Out(FrameHeader);
impl Frame for Out {
    fn header(&self) -> &FrameHeader { &self.0 }
    fn header_mut(&mut self) -> &mut FrameHeader { &mut self.0 }
}

/// Counts what it is pushed.
#[derive(Default)]
struct Count(std::sync::atomic::AtomicUsize);
impl LatticeTerminal for Count {
    fn add_upstream(&self, _upstream: Arc<dyn LatticeSource>) {}
}
impl LatticeReceiver for Count {
    fn push(&self, _frame: Box<dyn Frame>) { self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst); }
    fn on_complete(&self) {}
    fn on_error(&self, _error: Failure) {}
}

let mut builder = LanePoolBuilder::new();
let first = builder.reserve(1);
let (_fake, mut lanes, queues) = FakeLanes::open(first, 1);
builder.set(first, queues.into_iter().next().unwrap());
let (intakes, mut inputs) = LaneIntakes::<Item>::new(builder.build(), &[]);
let source = LaneSource::new(lanes.remove(0), (), inputs.remove(0), intakes.clone());
let out = Arc::new(Count::default());
source.clone().add_downstream(out.clone());

intakes.deliver(Item(1), Deliver::Lane(LaneId(0)));
intakes.deliver(Item(2), Deliver::Lane(LaneId(0)));
source.request(1); // demand for one frame: both items run, one frame is pushed
assert_eq!(out.0.load(std::sync::atomic::Ordering::SeqCst), 1);
source.request(1); // the second frame goes out with the demand banked since
assert_eq!(out.0.load(std::sync::atomic::Ordering::SeqCst), 2);
```

### Stage graphs

A `StageGraph<Q>` is a static DAG of stages run once per **quantum** `Q`: the unit of work the
graph is started for. Edges say when a stage's successor may run:

- a **submission** edge: when the predecessor submitted (the successor waits on the device);
- a **completion** edge: when the predecessor's work completed (its marker was reached, or its
  host completion was ready), found by a `Completions` poll, for successors that need the result;
- an **external** edge (`ExternalEdge`): a stage of a new quantum depending on a stage of an
  earlier one, held through a `QuantumRef`.

A stage's `run` computes or submits and returns a `Submitted` (its marker, perhaps a host
completion); it never waits. A stage that finds its resources busy answers `StageError::Retry` and
runs again when a lane of the pool made progress, or after `GraphRuntime::retry_backstop` (default
`DEFAULT_RETRY_BACKSTOP`, 20 ms: set it longer than any one pass on your devices). A stage can be
**ordered** (`StageSpec::ordered`): its frames then run in the order they were published, on one
lane of the lattice. The graph's **kill switch** goes on every stage frame of a quantum; a killed
quantum's unrun stages are `abandoned`.

A quantum **retires** when none of its frames can run again and every lane it used has reached its
last marker (its retirement boundary, armed as a `Completions` edge). `retire` then receives the
quantum and its `QuantumOutcome`: `Completed`, `Failed` or `Killed`. `live()` counts quanta not
retired yet.

### Lakes

A `Lake` is where ready work waits for the lattice: `sinks` queue ingest sinks of one partition
each, every one a source the workers poll. `offer` takes a frame and puts it in the sink its
routing hash picks, so the frames of one lane stay in order in one sink and the rest spread over
all of them. A lake is a `FrameSink`, and it is what a `GraphRuntime` publishes stage frames
through (`GraphRuntime::stages` is any `Arc<dyn FrameSink>`). Use one lake per kind of work.

A lake takes frames only while a **unit** is admitted. `admit()` counts a unit in and returns a
`UnitGuard`; the unit ends when the guard drops. For a stage graph the unit is the quantum: admit
before `start`, drop the guard in `retire`. The first unit attaches the sinks to the lattice, and
once no unit has been admitted for the idle release time, an idle watch (a source of no frames
that the workers poll) detaches them, so an idle lake costs the lattice nothing to poll; the next
unit attaches a fresh set. `complete_gracefully` closes admission, and the sinks drain and complete
once the last unit ends.

```rust,no_run
use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use euhedral_core::config::LatticeConfig;
use euhedral_core::control_plane::ControlPlaneLattice;
use euhedral_core::frames::RunnableFrame;
use euhedral_data_structures::sync::one_shot;
use euhedral_graph::Lake;

let lattice = ControlPlaneLattice::new(LatticeConfig::default()).unwrap();
let lake = Lake::new(lattice.clone(), NonZeroUsize::new(2).unwrap(), Duration::from_secs(1));

// The main thread starts the work and checks on it; nothing blocks.
let (done, mut finished) = one_shot();
let unit = lake.admit().unwrap(); // attaches the sinks
let done = Mutex::new(Some(done));
let frame = RunnableFrame::new(7, Arc::new(move || {
    if let Some(done) = done.lock().unwrap().take() {
        done.send("ran");
    }
}));
assert!(lake.offer(Box::new(frame)).is_ok());
while !finished.is_set() {
    std::thread::sleep(Duration::from_millis(1));
}
assert_eq!(finished.try_take(), Ok(Some("ran")));

drop(unit);
lake.complete_gracefully();
lattice.close();
```

A frame hands a result back to the main thread with a `one_shot` (`euhedral_data_structures::sync`): the frame
`send`s, and the main thread checks `is_set` or `try_take` periodically. Nothing in the crate blocks.

### Sequencer and Join

A `Sequencer<E: Entry>` (Java's `Sequencer`) concludes entries in the order they were offered, whatever order they
complete in. `offer(entry)` appends to the order from any thread (the order is a growing multi-producer queue, so an
offer is never refused); an entry is the caller's own object, and `Entry::ready` says whether it finished. The thread
that finishes an entry calls `drain()`: one drainer at a time (a work-in-progress counter) concludes the prefix of
ready entries, on its own thread, through the function given to `Sequencer::new`, which must not wait (it offers the
next frame). An entry embeds a `Progress` (pending, ready or abandoned; an abandoned entry concludes with nothing, so
one that is killed never stalls the order), and a caller that pools its entries allocates nothing on the whole path.
For callers without an entry type of their own, `Sequencer::ordered(emit)` makes a sequencer of `Ticket`s that carry a
value, and `register()` offers a ticket and returns its `Slot`: complete it with a value or abandon it (dropping it
abandons it), from any thread. `register` allocates the ticket.

A `Join<T>` is a continuation that runs on the thread of its last arrival, after `parts` parts
arrived or were abandoned. Neither is a queue, and neither blocks. `Join::once` makes a join of one
round and hands out its `Part`s. A `Part` is a token moved to its branch: `arrive(value)` or `abandon()` consumes it, so a part ends once (a second arrival does not compile), and dropping it abandons. `Join::with_capacity` makes a cell (an `Arc<Join>`) for a pooled unit: `reset(parts)` arms a round and returns its parts, and the
continuation receives a `Round` of the values and calls `rearm(parts)` for the next one (it returns the next round's parts). Arming and
arriving take no lock and allocate nothing.

## Build and test

```sh
mise exec -- cargo test -p euhedral-graph
```

`.github/workflows/ci.yml` runs formatting, Clippy and the tests of the whole workspace on Ubuntu and
Windows.

The tests are unit tests in each module (the stage graph's drives its sources by hand), the fake
device's, and `tests/credit_graph.rs`: one lattice of 4 workers, with a per-item
stage graph, a `Sequencer`, a `Join` as the pool lease, credit-bounded admission, a seek and a
pause, and a timing executor that fails the test if any frame callback runs over 200 µs without
the OS preempting it. It takes a few seconds and runs on the CPUs in `EUHEDRAL_TEST_CPUS` (default
`0,1,2,3`); as with `euhedral-core`'s tests that need particular hardware, it passes without
checking anything when the process may not run on all of them. It needs Linux.
