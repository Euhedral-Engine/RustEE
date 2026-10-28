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
| [`sequencer`](src/sequencer.rs) | `Sequencer` and `Slot`: execute in any order, release in admission order. |
| [`join`](src/join.rs) | `Join`: a continuation that runs when its last part arrives. |
| [`lake`](src/lake.rs) | `Lake`, `UnitGuard` and `LakeClosed`: the pool of ready work of one kind, as several ingest sinks that attach to the lattice for a busy period. A `FrameSink`. |
| [`latch`](src/latch.rs) | `Latch`, `LatchSetter` and `LatchWaiter`: a frame hands one value to the main thread, which parks until it arrives. |
| [`lanes`](src/lanes.rs) | `LanePool`, `LaneId`, `Marker` and the trait `DeviceQueue`: device timelines that are queried, never waited on. |
| [`stage_graph`](src/stage_graph.rs) | `StageGraph` and its spec (`StageGraphSpec`, `StageSpec`, `StageId`, `EdgeKind`), the `Stage` trait with `StageCx`, `Submitted` and `StageError`, `QuantumOutcome`, `ExternalEdge`, `QuantumRef`, `GraphRuntime`, `RetryCounts`, `GraphError` and `DEFAULT_RETRY_BACKSTOP`. |
| [`testing`](src/testing.rs) | `FakeDevice`, an in-order fake `DeviceQueue`. Feature `testing`. |
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

This runs one by hand, with a `Sequencer` that releases in admission order whatever order the
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
let sequencer = Sequencer::new({
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
            slot.complete(seq); // the sequencer releases it once every earlier slot did
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
assert_eq!(*released.lock().unwrap(), [0, 1, 2]); // released in admission order
```

### Device lanes and markers

A `DeviceQueue` is a device timeline that runs submissions in order: a Vulkan queue with a timeline
semaphore, a CUDA stream with events, a hardware codec engine, or a fake. What is generic is the
query, `completed()`, one cheap read. A `Marker` names a lane and a value on it; a `LanePool` holds
the lanes and answers `reached(marker)`. Submitting is technology-specific and lives with whoever
owns the queue: it takes the markers its work must wait for on the device and returns the marker
of the new work.

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

### Lakes and latches

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
use euhedral_graph::{Lake, Latch};

let lattice = ControlPlaneLattice::new(LatticeConfig::default()).unwrap();
let lake = Lake::new(lattice.clone(), NonZeroUsize::new(2).unwrap(), Duration::from_secs(1));

// The main thread starts the work and waits for it; nothing else blocks.
let (done, finished) = Latch::new();
let unit = lake.admit().unwrap(); // attaches the sinks
let done = Mutex::new(Some(done));
let frame = RunnableFrame::new(7, Arc::new(move || {
    if let Some(done) = done.lock().unwrap().take() {
        done.set("ran");
    }
}));
assert!(lake.offer(Box::new(frame)).is_ok());
assert_eq!(finished.wait(), "ran");

drop(unit);
lake.complete_gracefully();
lattice.close();
```

A `Latch` is the hand-off back to the main thread: `Latch::new()` returns a `LatchSetter` to move
into a frame and a `LatchWaiter` for `main`. `LatchWaiter::wait` parks the thread until the value
is set (at once if it already was) and is the one call in the crate that blocks; nothing on the
lattice may call it. A setter dropped without setting makes `wait` panic.

### Sequencer and Join

A `Sequencer<T>` hands values out in admission order. `register()` takes a slot, in the order the
caller admits work; the slot is completed with a value, or abandoned (dropping it abandons it),
from any thread. One releaser at a time (a work-in-progress counter) releases the prefix of
finished slots to `emit`, on its own thread, and `emit` must not wait: it offers the next frame.

A `Join<T>` is a continuation that runs on the thread of its last arrival, after `parts` parts
arrived or were abandoned. Neither is a queue, and neither blocks.

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
