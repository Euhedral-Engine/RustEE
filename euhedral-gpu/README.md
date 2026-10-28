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
| [`stream_lock`](src/stream_lock.rs) | `StreamLock`: the mutual exclusion of one stream's submissions. |
| [`shape`](src/shape.rs) | `Fingerprint`, `ShapeHasher`, `CopyKind`. |
| [`retire`](src/retire.rs) | `retire_into`: a retirement that continues as a frame of a `Completions`. |
| [`module_cache`](src/module_cache.rs) | `ModuleCache`, `CacheKey`, `cache_key`: compiled modules keyed by a content hash. |
| [`fake`](src/fake/mod.rs) (feature `fake`) | `FakeGpu`, a stepped device with no wall clock, and its streams, modules and graphs. |
| [`cuda`](src/cuda/mod.rs) (feature `cuda`) | `CudaGpu`, `CudaStream`, `CudaModule`, `CudaKernel`, `CudaGraph`, `Nvrtc`. |

## Key concepts

- **Streams are lanes.**
  - `Gpu::open_lanes` opens streams as the lanes of a new `LanePool`. A stream is a `DeviceQueue`, so a stage graph's
    lane choice, `Marker` and `Completions::after_marker` work with it as they are.
  - `Stream::submit(|q| ...)` runs on the calling thread with the stream selected for every operation. The operations
    of one submission are contiguous on the stream, even with other workers submitting at the same moment: a
    per-stream lock is held only while the closure enqueues, which is a few non-blocking driver calls. A call on a
    stream from inside its own `submit` is `Err(Invalid)`, never a deadlock.
  - Enqueue operations (`launch`, the copies, `memset`) are `unsafe fn`s, as `euhedral_memory::Copier::start` is: a
    kernel or a copy can touch any device-visible memory.
- **Markers across streams.** `mark()` records a marker after the work submitted so far; `await_marker(m)` orders later
  work behind it on the device, and the host does not wait. A marker belongs to the pool of the stream that made it; a
  marker of another pool is not detected. Each CUDA stream has a ring of `StreamOptions::markers`
  events (default 256). `mark()` returns `Err(Busy)` rather than reuse a slot whose marker has not completed; an
  await on a marker that already completed adds no device wait.
- **Retirement.**
  - `notify_retired(listener)` arms a completion point after the work submitted so far. The listener runs once, on the
    backend's callback thread, and may only enqueue: no driver calls, no blocking, no frames built or dropped. A panic
    in a listener is caught.
  - `confirm_retired(ticket)` runs on an ordinary thread and returns the device failure behind the ticket, or `None`.
    It releases the values kept with `Enqueue::keep`, so the host buffers a copy reads live until the retirement is
    confirmed. A ticket carries the identity of the stream that armed it (its fields are private, made only by a
    backend); settling it on another stream is `Some(Invalid)` and releases nothing.
  - A listener is dropped on the backend's callback thread, so it must not own the last handle to a stream, a graph or
    a device.
  - `retire_into(&stream, &completions, |ticket| frame)` arms a retirement whose continuation is a frame emitted by the
    lattice's own poll. A submission finishing never counts as completion. If the device is poisoned before the
    callback runs, the continuation is emitted at once, so nothing is stranded.
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
  - `Gpu::record(&streams, |q| ...)` runs the closure once for real and records it. The same code path then serves
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
    after the replay is confirmed, so a replay in flight survives the graph dropping; arm one per replay. Launches of
    one graph are serialized inside it, so replays from several workers are safe. A re-record replaces the old
    recording's kept values.
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
let (lanes, streams) = gpu.open_lanes(&["compute", "copy"], StreamOptions::default()).unwrap();
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

// The copy's retirement continues as a frame, emitted by the lattice's own poll.
let completions = Completions::new(lanes);
let confirmed = Arc::new(Mutex::new(None));
retire_into(&streams[1], &completions, {
    let (stream, confirmed) = (streams[1].clone(), confirmed.clone());
    move |ticket| {
        Box::new(RunnableFrame::new(
            1,
            Arc::new(move || *confirmed.lock().unwrap() = Some(stream.confirm_retired(ticket))),
        ))
    }
})
.unwrap();

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
