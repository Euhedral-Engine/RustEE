# euhedral-graph

No-wait dataflow on the RustEE lattice: completions, asynchronous I/O, in-order release, joins,
device lanes and per-item stage graphs. Nothing in it waits and nothing in it owns a thread. See
[`GLOSSARY.md`](../GLOSSARY.md) for every term used here.

The pieces are ports of Euhedral-Inference's `runtime.graph` package (`StageGraph`, `LanePool`,
`Join`, `AsyncReads`) and `quantization.lattice` (`GpuCompletions`), and of Euhedral-Execution's
`FrameSequencer` (and the Kafka `OffsetCollector`). `AsyncIo` goes beyond `AsyncReads`: writes,
sends, receives, accepts and connects.

**Where it sits.** It depends on [`euhedral-core`](../euhedral-core), and, for `AsyncIo`, on
`io-uring` and `libc` (Linux and Android) or `windows-sys` (Windows). Nothing in the workspace depends on it; applications do. It adds
`LatticeSource`s that the lattice's own polls drive, and plain structures that frame callbacks run,
so it needs no thread of its own. It does not re-export `euhedral-core`: import `Frame`,
`LatticeSource` and the rest from there.

## Modules

| Module | Main types |
|---|---|
| [`completions`](src/completions.rs) | `Completions`, a `LatticeSource` of continuation frames behind completions: markers, timers, `Ready` tokens and `Pending` operations. |
| [`io`](src/io/mod.rs) | `AsyncIo`, a `LatticeSource` of continuations behind file and socket operations, with `IoBuf`, `IoDone`, `IoThen`, `OpenSpec` (`Create`), `AsyncIoOptions` and the handle aliases `RawFile`, `RawSock`, `OwnedFile` and `OwnedSock`. io_uring on Linux and Android ([`uring`](src/io/uring.rs)); IoRing and completion ports on Windows ([`win`](src/io/win/mod.rs)); none on macOS yet. Feature `async-io` (default). |
| [`sequencer`](src/sequencer.rs) | `Sequencer` and `Slot`: execute in any order, release in admission order. |
| [`join`](src/join.rs) | `Join`: a continuation that runs when its last part arrives. |
| [`lanes`](src/lanes.rs) | `LanePool`, `LaneId`, `Marker` and the trait `DeviceQueue`: device timelines that are queried, never waited on. |
| [`stage_graph`](src/stage_graph.rs) | `StageGraph` and its spec (`StageGraphSpec`, `StageSpec`, `StageId`, `EdgeKind`), the `Stage` trait with `StageCx`, `Submitted` and `StageError`, `QuantumOutcome`, `ExternalEdge`, `QuantumRef`, `GraphRuntime`, `RetryCounts`, `GraphError` and `DEFAULT_RETRY_BACKSTOP`. |
| [`testing`](src/testing.rs) | `FakeDevice`, an in-order fake `DeviceQueue`. Feature `testing`. |
| crate root | Every type above, re-exported; `Continuation` (what turns a completed operation's result into the next frame); `ThreadUsage` (the calling thread's CPU time and involuntary context switches, to tell a preempted callback from a slow one; measured on Linux and Android). |

## Key concepts

### Nothing waits

Work is **submitted** and the call returns at once. What follows it starts on one of two events:

- the **submission**: a device successor orders itself behind its predecessor's `Marker` on the
  device (a wait inside its own submission), so the host never waits;
- the **completion**, found when a lattice worker polls a source. The continuation is a frame the
  source emits, and the lattice runs it like any other.

`Completions` and `AsyncIo` are `LatticeSource`s, so they attach to a lattice with `add_upstream`
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

### AsyncIo

`AsyncIo` does file and socket I/O that holds no worker: `read`, `write`, `send`, `recv`, `accept`,
`open_path` (and `open_file` with raw flags on Linux), `rename`, `send_to` and `connect` each
return at once, and the operation's `IoDone` (its buffer back, and the bytes moved or the error)
reaches the `IoThen` continuation, which builds the frame the source emits. Handles are the
platform's: descriptors on Linux and Android, `HANDLE`s and `SOCKET`s on Windows (the aliases
`RawFile`, `RawSock`, `OwnedFile` and `OwnedSock`). A read is cut into pieces so that no single
completion holds a worker for long. The `AsyncIo` type is the same everywhere; an engine under it
starts operations and finds their ends, and `io/mod.rs` does the rest.

- **Linux and Android:** io_uring, with one ring per submitting thread. An operation belongs to the
  thread that submitted it, and lattice workers live as long as the lattice.
- **Windows:** one I/O completion port per `AsyncIo`, polled with a zero timeout, for every
  operation, and one IoRing per submitting thread for file reads and writes where the OS has a
  kernel ring (Windows 11; writes from 22H2; not one emulated in user mode). The ring functions are
  looked up at run time, so the library still loads where they are missing, and each operation
  falls back to the port on its own. `AsyncIoOptions { windows_ioring: false }` forces the port.
  Windows has no asynchronous open or rename, so those run on the OS thread pool and post their
  result to the port: the one place the crate uses threads it did not create. Handles must be
  opened for overlapped I/O (`open_path` does, and so do Rust's sockets) and must not skip the port
  on success; a handle that would block or never complete is refused. A handle joins the engine's
  port and can serve one `AsyncIo`, ever. Writes go in pieces of at most 256 KiB, since a write
  that extends a file is synchronous on NTFS. Tested on a real Windows by
  [CI](../.github/workflows/windows.yml) on every push to the `graph` branch.
- **macOS:** no engine yet.

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
mise exec -- cargo test -p euhedral-graph --no-default-features   # without AsyncIo
```

The Windows engine's state machines run on every platform against a simulated `Sys` (the seam over
the Windows calls, in `io/win/sim.rs`). Its real calls (`io/win/real.rs`) compile on Windows only;
the workflow `.github/workflows/windows.yml` runs `cargo test -p euhedral-graph` on a
`windows-2025` runner, where `io/win/live.rs` exercises files, loopback sockets, open and rename,
each twice: with IoRing and forced onto the completion port.

The tests are unit tests in each module (the stage graph's and `AsyncIo`'s drive their sources by
hand), the fake device's, and `tests/credit_graph.rs`: one lattice of 4 workers, with a per-item
stage graph, a `Sequencer`, a `Join` as the pool lease, credit-bounded admission, a seek and a
pause, and a timing executor that fails the test if any frame callback runs over 200 µs without
the OS preempting it. It takes a few seconds and runs on the CPUs in `EUHEDRAL_TEST_CPUS` (default
`0,1,2,3`); as with `euhedral-core`'s tests that need particular hardware, it passes without
checking anything when the process may not run on all of them. It needs Linux.

`AsyncIo`'s tests use a scratch directory beside the test binary, under `target/`.
