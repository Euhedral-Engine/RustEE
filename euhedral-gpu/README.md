# euhedral-gpu

Asynchronous, backend-neutral GPU access for the RustEE control plane. A lattice worker can:
- submit kernels and copies to a stream, without ever waiting for the device;
- order work across streams on the device with markers;
- learn that work retired as an ordinary frame;
- load kernels the application brings with it, from source or from a prebuilt image;
- record a recurring sequence of submissions once and replay it.

**Where it sits.** It is the top layer of the workspace. It builds on [`euhedral-core`](../euhedral-core) (frames),
[`euhedral-graph`](../euhedral-graph) (`DeviceQueue`, `Marker`, `LanePool` and `Completions`),
[`euhedral-memory`](../euhedral-memory) (the CUDA driver, device and allocations) and
[`euhedral-hashing`](../euhedral-hashing) (cache keys and fingerprints). The crates form layers (see the
[workspace README](../README.md#crates)); a crate builds only on lower layers, and this one is the highest.

## Modules

| Module | Main types |
|---|---|
| [`api`](src/api.rs) | The traits: `Gpu`, `Stream`, `Enqueue`, `Module`, `Kernel`, `Sequence`, `Graph`, `RetireListener`; `RetireTicket`, `StreamOptions`, `CompileOptions`, `Rerecorded`. |
| [`args`](src/args.rs) | `KernelArg`, `Args` (the typed argument builder), `Launch`, `Dim3`. |
| [`error`](src/error.rs) | `GpuError`, the one error type. |
| [`poison`](src/poison.rs) | `Poison`: a device's first fault, set once. |
| [`kept`](src/kept.rs) | The values a submitter keeps until a retirement is confirmed (a `PlainQueue`), and the frozen values a recording shares. |
| [`shape`](src/shape.rs) | `Fingerprint`, `ShapeHasher`, `CopyKind`. |
| [`retire`](src/retire.rs) | `retire_into`: a retirement that continues as a frame of a `Completions`. |
| [`module_cache`](src/module_cache.rs) | `ModuleCache`, `CacheKey`, `cache_key`: compiled modules keyed by a content hash. |
| [`fake`](src/fake/mod.rs) (feature `fake`) | `FakeGpu`, a stepped device with no wall clock, and its streams, modules and graphs. |
| [`cuda`](src/cuda/mod.rs) (feature `cuda`) | `CudaGpu`, `CudaStream`, `CudaModule`, `CudaKernel`, `CudaGraph`, `Nvrtc`. |

## Key concepts

- **Streams are lanes, and ownership replaces locking.**
  - `Gpu::open_lanes` opens streams as the lanes of a new `LanePool` and returns one *submitter* per lane. A stream is
    split in two. Its **queue** (`DeviceQueue`: `id`, `completed`, `name`) is shared and `Sync`: the pool and
    `Completions` hold it, so a stage graph's lane choice, `Marker` and `Completions::after_marker` work with it as
    they are. Its **submitter** (`Stream`) is `Send` and not `Sync`, with exactly one owner.
  - Everything that enqueues takes `&mut`: `submit`, `mark`, `await_marker`, `notify_retired`, `confirm_retired`, and
    `Graph::replay`. Exclusivity is the borrow checker's, so there is no lock to take or wait on, the operations of one
    submission are contiguous on the stream because nothing else can enqueue on it, and a call on a stream from inside
    its own `submit` does not compile (a `compile_fail` doctest and `tests/ownership.rs` pin this).
  - Work for a stream from another thread goes to the thread that owns it (one of RustEE's lock-free queues, for
    instance an `MpscQueue` the owner drains, or the submitter handed over inside a frame, which is what `retire_into`
    does), or runs on another lane. There is no routing layer.
  - Enqueue operations (`launch`, the copies, `memset`) are `unsafe fn`s, as `euhedral_memory::Copier::start` is: a
    kernel or a copy can touch any device-visible memory.
- **Markers across streams.** `mark()` records a marker after the work submitted so far; `await_marker(m)` orders later
  work behind it on the device, and the host does not wait. A marker carries the identity of the queue that made it (`QueueId`, from `DeviceQueue::id`): `await_marker`
  refuses a marker of a queue that is not in the stream's pool with `Err(Invalid)`, whatever its lane and value, and a
  `LanePool` never reaches one. Each CUDA stream has a ring of `StreamOptions::markers`
  events (default 256). `mark()` returns `Err(Busy)` rather than reuse a slot whose marker has not completed; an
  await on a marker that already completed adds no device wait.
- **Retirement.**
  - A retirement is a completion point after the work submitted so far. **The driver's callback only queues its
    outcome**: each stream owns one unbounded, lock-free MPSC queue (`Queue<_, Multi, Single>` of
    `euhedral-data-structures`, chunk-growing, with pooled chunks so steady state recycles them), and the callback,
    whose user data is a pointer to the stream's end of that queue, pushes the status of the work before it. It makes
    no driver call, takes no lock, frees nothing, and allocates nothing in steady state (`tests/cuda_alloc.rs` counts
    every allocation of the process over 1600 retirements: zero). A listener is never touched by the callback.
  - The retirement's number is not sent: a stream's callbacks run in the order they were enqueued, and each armed
    retirement enqueues one, so the n-th status that arrives belongs to the n-th retirement armed. The owner keeps what
    to do with each (a `PlainQueue` by number), and `poll_retired()` drains the queue and dispatches: `notify_retired(
    listener)` listeners are told, on the owner's thread, in order; the first retirement armed with
    `arm_retirement()` returns its ticket. Listeners are made and dropped on the owner's thread only.
  - If the device is poisoned, `poll_retired` reports the retirements still outstanding with `POISONED_STATUS` and
    drops what arrives afterwards (it could no longer say whose it is).
  - `confirm_retired(ticket)` is a method of the submitter and returns the device failure behind the ticket, or `None`.
    The submitter owns its kept values (a `PlainQueue`), so releasing them needs no lock and no queue. It releases the values kept with `Enqueue::keep`, so the host buffers a copy reads live until the retirement is
    confirmed. A ticket carries the identity of the stream that armed it (its fields are private, made only by a
    backend); settling it on another stream is `Some(Invalid)` and releases nothing.
  - `retire_into(stream, &completions, |stream, ticket| frame)` arms a retirement whose continuation is a frame
    emitted by the lattice's own poll. The submitter **travels with its continuation**: it is taken by value and
    handed back to the closure with the ticket, so the frame that continues the work owns the stream and its kept
    values. `Completions` polls the stream's retirement queue through `Stream::poll_retired`; there is no
    per-retirement cell. A submission finishing never counts as completion. If the device is poisoned before the
    callback runs, the continuation is emitted at once with `POISONED_STATUS`, so nothing is stranded.
- **Kernels from outside.** A `Module` loads from `from_source` or from `from_image`; the crate enumerates no kernels.
  - `from_source` compiles CUDA C with NVRTC for the device's exact architecture (or PTX for the newest architecture
    NVRTC knows, which the driver compiles forward), with the options, include paths and in-memory headers of
    `CompileOptions`. With `cache_dir` set, the result is cached under a 128-bit hash of the source, the options, the
    headers, the include paths, the target and NVRTC's version. A cache entry that will not load is rebuilt, and writes
    are atomic renames.
  - `from_image` takes PTX, a cubin or a fatbin that the application produced in its own build. No NVRTC is needed.
  - NVRTC is loaded with `dlopen`. The search order is the explicit `CompileOptions::compiler`, then `EUHEDRAL_NVRTC`,
    then the loader's own search, then `$CUDA_HOME/lib64`, `$CUDA_PATH/lib64`, `/usr/local/cuda/lib64` and
    `/usr/local/cuda-*/lib64`.
  - `Module::kernel(name)` looks a kernel up by its unmangled symbol (declare kernels `extern "C"`). `Args` packs the
    arguments (`Args::new().device(addr).arg(2.5f32)`); where the driver reports the kernel's parameter sizes, a launch
    with arguments that do not fit is `Err(Args)` before the driver sees it.
- **Recurring sequences.**
  - `Gpu::record(&mut [&mut a, &mut b], |q| ...)` runs the closure once for real and records it. The same code path then serves
    `fingerprint` (a probe: nothing reaches the device) and `update` (new argument values: launch *n* of the update
    is launch *n* of the recording).
  - The **shape** is the operations in order, their stream indexes, the `order` edges, kernel identities, launch
    geometry, argument layouts, and the parameters of copies and memsets. A sequence of another shape is
    `Err(ShapeChanged)` and applies nothing: call `rerecord`, which updates the executable in place when the driver
    can, else instantiates a new one (`Rerecorded`).
  - Recording uses a shadow stream per stream and captures there, so a real stream is never in capture mode. A failed
    capture is `Err(Capture)`, and the work itself ran regardless.
  - A replay is one submission on the stream it is launched on; its lane marker is that stream's `mark()` afterwards.
    A replay with the arguments of an update does not touch a replay in flight.
  - Host memory that a recorded copy reads is read again by every replay: use pinned memory, and hand it to the
    recording with `Enqueue::keep`. Every pointer a recording names must stay valid for every replay until the graph
    drops. A replay hands the recording's kept values to its stream, which holds them until the next retirement armed
    after the replay is confirmed, so a replay in flight survives the graph dropping; arm one per replay. `replay` takes
    `&mut` the graph, so a replay has its executable to itself without a lock: the owner of the graph decides which
    stream replays it, and a graph that several lanes need is one graph per lane. The recording's kept values are
    frozen into an immutable `Arc<[..]>` that streams and replays share. A re-record replaces the old
    recording's kept values. The recording's own run is a real run too: whatever the closure kept is handed to every
    stream it ran on, so a failed recording, or a graph dropped right after `record`, leaves the run its buffers.
  - Recording forks and joins the real streams with events of its own, and a panic in the recording closure ends the
    capture.
- **Failures are values.** A device fault (an illegal address, a launch failure) surfaces through `confirm_retired`
  and poisons the device: later calls return `Err(Poisoned(cause))` at once, and every marker reads as reached. The
  driver's other unrecoverable errors (an event query that fails for any reason but "not ready") poison it the same
  way. A fault is sticky for the process, so recovery is a new process. Nothing panics on the lattice.
- **Build-time calls.** Module loading (compile, file reads, the driver's load), dropping a stream and dropping a
  graph block. Make them at start-up or shutdown, or on a lattice set aside for blocking work.

```rust
use std::sync::{Arc, Mutex};

use euhedral_core::frames::RunnableFrame;
use euhedral_core::generics::LatticeSource;
use euhedral_gpu::fake::{FakeGpu, FakeModule};
use euhedral_gpu::{
    Args, CompileOptions, Enqueue, Gpu, Launch, Module, Stream, StreamOptions, retire_into,
};
use euhedral_graph::Completions;

// The fake device stands in for a real one; a kernel is a host closure registered by name.
let gpu = FakeGpu::new();
gpu.define("add1", |memory, _launch, args| {
    let word = memory.bytes_mut(args.read::<u64>(0).unwrap(), 4)?;
    let value = u32::from_le_bytes(word.try_into().unwrap()) + 1;
    word.copy_from_slice(&value.to_le_bytes());
    Ok(())
});
let module = FakeModule::from_source(&gpu, "// the kernels' source", &CompileOptions::default()).unwrap();
let add1 = module.kernel("add1").unwrap();
let (lanes, mut streams) = gpu.open_lanes(&["compute", "copy"], StreamOptions::default()).unwrap();
let buffer = gpu.alloc(4);
let mut out = [0u8; 4];

// Lane 0 launches and marks; lane 1 copies the result out behind that marker, on the device.
streams[0]
    .submit(|q| {
        // SAFETY: the kernel adds one to the word at `buffer`, a live allocation nothing else uses.
        unsafe { q.launch(&add1, Launch::linear(1, 1), &Args::new().device(buffer)) }
    })
    .unwrap();
let done = streams[0].mark().unwrap();
streams[1].await_marker(done).unwrap();
streams[1]
    .submit(|q| {
        // SAFETY: `out` lives until the run below ends.
        unsafe { q.copy_to_host(out.as_mut_ptr(), buffer, 4) }
    })
    .unwrap();

// The copy's retirement continues as a frame, emitted by the lattice's own poll. The submitter travels with it.
let completions = Completions::new(lanes);
let confirmed = Arc::new(Mutex::new(None));
let copy_lane = streams.remove(1);
retire_into(copy_lane, &completions, {
    let confirmed = confirmed.clone();
    move |mut stream, ticket| {
        let failure = stream.confirm_retired(ticket);
        Box::new(RunnableFrame::new(
            1,
            Arc::new(move || *confirmed.lock().unwrap() = Some(failure.clone())),
        ))
    }
})
.unwrap_or_else(|e| panic!("{}", e.error));

// A fake device runs when it is stepped; a real one runs by itself. Here the loop is the worker.
gpu.run_until_idle();
while completions.pending() > 0 {
    completions.pull(
        &mut |mut frame| {
            frame.execute().unwrap();
            frame.do_finally();
        },
        &mut |_| false,
        u64::MAX,
    );
}
assert_eq!(*confirmed.lock().unwrap(), Some(None)); // retired, no failure
assert_eq!(u32::from_le_bytes(out), 1);
```

## Features

| Feature | What it adds |
|---|---|
| `fake` | `FakeGpu`: a deterministic device for tests of code built on this crate. The crate's own tests enable it through a dev-dependency on itself. |
| `cuda` | The CUDA backend, on the driver and primary context `euhedral-memory` loads (`Driver::shared`): one library, one context. Linux. |

The traits and the fake compile on every platform the workspace builds on.

## Build and test

```sh
mise exec -- cargo test -p euhedral-gpu                      # traits, fake backend, layering check: no device needed
mise exec -- cargo test -p euhedral-gpu --features cuda      # + the CUDA backend (its tests skip, passing, without a device)
CUDA_HOME=/path/to/toolkit mise exec -- cargo test -p euhedral-gpu --features cuda   # + kernels compiled from source
mise exec -- cargo run -p euhedral-gpu --features cuda --example graph_replay        # the acceptance program
```

NVRTC is found through `EUHEDRAL_NVRTC` (a library file) or `CUDA_HOME` (a toolkit); without it, tests that compile from
source skip and prebuilt images still load. The test `tests/cuda_fault.rs` is a process of its own, because a device
fault is sticky.
