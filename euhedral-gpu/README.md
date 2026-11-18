# euhedral-gpu

Asynchronous, backend-neutral GPU access for the RustEE control plane. A lattice worker can:
- submit kernels and copies to a stream, without ever waiting for the device;
- order work across streams on the device with markers;
- learn that work retired as an ordinary frame;
- load kernels the application brings with it, from source or from a prebuilt image;
- record a recurring sequence of submissions once and replay it.

**Terms.** The [top-level glossary](../GLOSSARY.md) covers the terms a user meets; this crate's [`GLOSSARY.md`](GLOSSARY.md) covers its implementation-level terms.

**Where it sits.** It is the top layer of the workspace. It builds on [`euhedral-core`](../euhedral-core) (frames),
[`euhedral-graph`](../euhedral-graph) (`DeviceQueue`, `Marker`, `LanePool` and `Completions`),
[`euhedral-memory`](../euhedral-memory) (the CUDA driver, device and allocations) and
[`euhedral-hashing`](../euhedral-hashing) (cache keys and fingerprints). The crates form layers (see the
[workspace README](../README.md#crates)); a crate builds only on lower layers, and this one is the highest.

## Modules

| Module | Main types |
|---|---|
| [`api`](src/api.rs) | The traits: `Gpu`, `Stream`, `Lane`, `LaneEnqueue`, `Enqueue`, `Module`, `Kernel`, `Sequence`, `Graph`, `RetireListener`; `RetireTicket`, `DeviceAttributes` (`Gpu::attributes`), `StreamOptions`, `StreamPriority`, `CompileOptions`, `KernelOptions`, `Rerecorded`, `Span` (timed spans, `Stream::span_*`). |
| [`lanes`](src/lanes.rs) | `LaneSubmit` for the backends' streams, and `open_lane_sources`: a device's lanes as a pool, its intakes and a `LaneSource` per lane (`LaneSpec`, `OpenedLanes`); `open_shared_lanes`: a device's lanes as a typed pool that any worker submits to (`SharedLanes`). |
| [`ring`](src/ring.rs) | The lock-free marker ring behind a lane's markers: numbers claimed by compare-and-set, completion found by an in-order scan, the awaiters' pins. No driver; loom-checked. |
| [`cuda::texture`](src/cuda/texture.rs) | `TexObject`, `View`: texture objects over pitched device memory (filter, address and coordinate modes). |
| [`tensor_map`](src/tensor_map.rs), [`cuda::tensor_map`](src/cuda/tensor_map.rs) | `TensorMap`, `TiledMap`: the 128-byte, 64-aligned `CUtensorMap` a kernel takes by value for the Tensor Memory Accelerator (compute capability 9.0 and newer); `TensorMap::tiled` encodes one with `cuTensorMapEncodeTiled`, `TensorMap::fake` (feature `fake`) stands in without a driver. |
| [`graph_cache`](src/graph_cache.rs) | `GraphCache`: an owned lane's graphs by key (a hit updates and replays, a miss records, a warm-up only records), `SharedGraphCache`: the same through a shared lane for any number of workers (instances by key), `GraphBody`, `Ran`. |
| [`hold`](src/hold.rs) | `GraphHold`: what the work of a run on a shared lane reads, held by the caller until the work ended. |
| [`args`](src/args.rs) | `KernelArg`, `Args` (the typed argument builder, rewritable in place), `Launch` (geometry and the programmatic-launch flag), `Dim3`. |
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
  - `Gpu::open_lanes` opens streams as the lanes of a new `LanePool` and returns one *submitter* per lane. `Gpu::open_lanes_at` opens them at lane ids a `LanePoolBuilder` reserved, with options per lane (the marker ring, and the **stream priority**). A stream is
    split in two. Its **queue** (`DeviceQueue`: `id`, `completed`, `name`) is shared and `Sync`: the pool and
    `Completions` hold it, so a stage graph's lane choice, `Marker` and `Completions::after_marker` work with it as
    they are. Its **submitter** (`Stream`) is `Send` and not `Sync`, with exactly one owner.
  - Everything that enqueues takes `&mut`: `submit`, `mark`, `await_marker`, `notify_retired`, `confirm_retired`, and
    `Graph::replay`. Exclusivity is the borrow checker's, so there is no lock to take or wait on, the operations of one
    submission are contiguous on the stream because nothing else can enqueue on it, and a call on a stream from inside
    its own `submit` does not compile (a `compile_fail` doctest and `tests/ownership.rs` pin this).
  - In RustEE's model the owner is a `LatticeSource`: the lattice serializes its `request` and `pull`, so whichever
    worker runs them is the submitter's only user. Work for a lane from other frames reaches its source through a
    lock-free queue (an `MpscQueue` the source drains), or the submitter is handed from frame to frame inside a chain,
    which is what `retire_into` does, or the work runs on another lane. There is no routing layer.
  - Enqueue operations (`launch`, the copies, `memset`) are `unsafe fn`s, as `euhedral_memory::Copier::start` is: a
    kernel or a copy can touch any device-visible memory.
- **Shared lanes: any worker submits.** A stage of a stage graph submits to the lane it was given, from whichever worker
  runs it, as in Java, where a stage calls `stream.submit` directly. `Stream::shared()` is the lane's shared handle
  (`Lane`: `CudaQueue`, `FakeQueue`): the queue side above, and `submit`, `submit_marked` and `mark` through `&self`, plus
  `LaneControl` (`await_marker`, `notify_retired`, `confirm_retired`), which is what a stage graph uses to join and retire
  lanes. `open_shared_lanes` opens a device's lanes as a `LanePool` typed over the lane, which a `StageGraph<Q, L>` takes
  as its runtime's lanes, so a stage reaches its lane with `cx.lane_handle()` and returns the marker of its work in
  `Submitted::marker`.
  - **Submit** is the driver's: it serializes the launches it gets on one stream, and a submission's operations are
    contiguous in one call's driver calls (another worker's may come between two calls). Nothing is locked.
  - **Mark** records after all the work submitted so far, which may include another worker's: that only over-waits.
    `submit_marked` claims the marker's number before the work and records it after (Java opens the marker before the
    stage submits), so a full ring (`Err(Busy)`) is learned before any work is in and a retried stage submitted nothing.
    The numbers are handed out by compare-and-set, and the scan that finds markers complete walks them in order and stops
    at a number whose record is not stored yet, so a slot that still holds an older record is never mistaken for the new
    marker (`src/ring.rs`; its loom models check the claim and the scan). A claim whose work failed is given up as a hole
    that completes without an event.
  - **Retirement** keeps the callback that only queues, and each arm names its own sink: `notify_retired(cell)` makes the
    driver's callback store the status of the work before it in that `RetireCell` and nothing else, so parties arming on
    one lane never read each other's retirement and nothing is claimed or waited for. The cell is refcounted (the callback
    holds a clone), so it outlives a lane that is dropped first. The continuation that settles the status is a frame
    (`Completions::after_retired`).
  - **Library calls.** `LaneStream::raw_stream()` on the closure's enqueue handle lends the lane's driver stream (a `RawStream<'_, R>`: it
    borrows the handle and is `!Send`, so it stays inside the closure and on the calling thread), for library calls that
    enqueue on a stream of their own accord (any library that enqueues on a stream): Java's `GpuStream.submit(Runnable)`. What the
    library does with the stream is the caller's contract (see `LaneStream`); the marker claim still precedes the closure.
  - **Held values.** A shared lane keeps none. What a submission's work reads (a host buffer a copy reads) is held by the
    stage, in its quantum, and released in the stage's `retired` hook, which runs once the quantum's device work ended: a
    retirement armed by another party cannot release it early.
  - The owned `Stream` is unchanged and still what `LaneSource` consumes. A stream and a shared handle over it share the
    marker ring and the driver stream; the owner keeps its own retirement channel and kept values.
- **Markers across streams.** `mark()` records a marker after the work submitted so far; `await_marker(m)` orders later
  work behind it on the device, and the host does not wait. A marker carries the identity of the queue that made it (`QueueId`, from `DeviceQueue::id`): `await_marker`
  refuses a marker of a queue that is not in the stream's pool with `Err(Invalid)`, whatever its lane and value, and a
  `LanePool` never reaches one. Each CUDA stream has a ring of `StreamOptions::markers`
  events (default 256). `mark()` returns `Err(Busy)` rather than reuse a slot whose marker has not completed; an
  await on a marker that already completed adds no device wait. An await that races the owner re-recording the marker's
  slot is made safe without a lock: the awaiter raises a per-slot pin before it enqueues its wait, and the owner's
  `mark` returns `Busy` (it never waits) if the slot it would reuse is pinned.
- **Retirement.**
  - A retirement is a completion point after the work submitted so far. **The driver's callback only queues its
    outcome**: each stream owns one unbounded, lock-free MPSC queue (`Queue<_, Multi, Single>` of
    `euhedral-data-structures`, chunk-growing, with pooled chunks so steady state recycles them), and the callback,
    whose user data is a pointer to the stream's end of that queue, pushes the status of the work before it. It makes
    no driver call, takes no lock, frees nothing, and allocates nothing while fewer than `UNPOLLED_RETIREMENTS` (252)
    statuses wait unpolled: the queue's chunk pool is seeded when the stream is made, and the owner's polling returns
    chunks to it (`tests/cuda_alloc.rs` counts every allocation of the process over 5600 retirements, in batches of 8
    and of 100: zero). Past the bound the queue grows by allocating a chunk, which is what "unbounded" accepts. The pool
    guards its stack with a try-lock inside `euhedral-data-structures`, which the separate lock audit tracks. A listener
    is never touched by the callback.
  - The retirement's number is not sent: a stream's callbacks run in the order they were enqueued, and each armed
    retirement enqueues one, so the n-th status that arrives belongs to the n-th retirement armed. The owner keeps what
    to do with each (a `PlainQueue` by number), and `poll_retired()` drains the queue and dispatches: `notify_retired(
    listener)` listeners are told, inside the owning source's call, in order; the first retirement armed with
    `arm_retirement()` returns its ticket. Listeners are made and dropped inside the owner's calls only.
  - If the device is poisoned, `poll_retired` reports the retirements still outstanding with `POISONED_STATUS` and
    drops what arrives afterwards (it could no longer say whose it is). A status that arrives with no retirement armed
    is never dropped quietly: it poisons the device with `Invalid`.
  - Retirement uses the stream callback (`cuStreamAddCallback`), which reports the status of the work before it exactly
    once, even after a device fault, so a fault arrives as the retirement's status. The entry point is required when
    the CUDA backend loads (every CUDA 13 driver exports it); a driver without it is a `GpuError::Load` ("libcuda: no
    cuStreamAddCallback"), not a panic. There is no `cuLaunchHostFunc` fallback: a host function reports success
    whatever became of the work before it.
  - `retire_into` refuses (the stream comes back in the error) a stream that has a listener-free retirement
    outstanding (`outstanding_tickets()`): its ticket would reach the continuation in place of its own.
    `retire_into_with` carries extra state with the stream and returns it on failure.
  - Dropping a submitter is a shutdown call (it waits for the device and its callbacks): a submitter is dropped in the
    owning source's lifecycle stop path, and a continuation hands the stream back to that source instead of dropping it. If the wait cannot complete (a lost context or a sticky fault) while
    retirements are outstanding, the drop leaks one reference to the stream's shared side, so a callback still to come
    never pushes into freed memory.
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
    NVRTC knows, which the driver compiles forward; `CompileOptions::arch_specific` asks for the architecture-specific
    `sm_XYa` machine code instead, and is an error naming the architecture when NVRTC does not list it, with no
    fallback to the generic target or to PTX), with the options, include paths and in-memory headers of
    `CompileOptions`. With `cache_dir` set, the result is cached under a 128-bit hash of the source, the options, the
    headers, the include paths, the target and NVRTC's version. A cache entry that will not load is rebuilt, and writes
    are atomic renames.
  - `from_image` takes PTX, a cubin or a fatbin that the application produced in its own build. No NVRTC is needed.
  - NVRTC is loaded at run time (`dlopen` on Linux, `LoadLibrary` on Windows). The search order on Linux is the
    explicit `CompileOptions::compiler`, then `EUHEDRAL_NVRTC`, then the loader's own search, then `$CUDA_HOME/lib64`,
    `$CUDA_PATH/lib64`, `/usr/local/cuda/lib64` and `/usr/local/cuda-*/lib64`. On Windows it is the explicit path,
    `EUHEDRAL_NVRTC`, `nvrtc64_130_0.dll` and `nvrtc64_120_0.dll` found by walking the absolute entries of `PATH`
    (where a toolkit's installer adds its `bin`; the working directory is never searched), then `bin\x64` and `bin` of
    `%CUDA_HOME%` and `%CUDA_PATH%`. An absolute file path is loaded with its own directory searched first, so NVRTC
    finds its builtins library beside it.
  - `Module::kernel(name)` looks a kernel up by its unmangled symbol (declare kernels `extern "C"`). `Args` packs the
    arguments (`Args::new().device(addr).arg(2.5f32)`); where the driver reports the kernel's parameter sizes, a launch
    with arguments that do not fit is `Err(Args)` before the driver sees it.
  - `Module::kernel_with(name, KernelOptions { max_dynamic_shared_bytes })` opts the kernel in, once at the lookup, to
    more dynamic shared memory than the driver's default; a launch above the kernel's limit is `Err(Invalid)` naming
    the kernel, the bytes and the limit. Repeated lookups only raise the limit.
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
  as `Poisoned(cause)` and poisons the device: later calls return `Err(Poisoned(cause))` at once, and every marker
  reads as reached. The
  driver's other unrecoverable errors (an event query that fails for any reason but "not ready") poison it the same
  way. A fault is sticky for the process, so recovery is a new process. Nothing panics on the lattice.
- **Build-time calls.** Module loading (compile, file reads, the driver's load), dropping a stream and dropping a
  graph block. Run them in the frame or item that needs them; a worker is a CPU. Keep them out of paths that run per item.
  Module loads and device frees also wait for the context's running kernels, and a `cuEventRecord` on another thread's
  stream measured behind them for as long as they waited, so they belong before the work they would stall starts, not
  beside it.

```rust
use std::sync::{Arc, Mutex};

use euhedral_core::generics::LatticeSource;
use euhedral_gpu::fake::{FakeGpu, FakeModule};
use euhedral_gpu::{
    Args, CompileOptions, Enqueue, LaneEnqueue, Gpu, Launch, Module, OnceFrame, Stream, StreamOptions, retire_into,
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

// The copy's retirement continues as a frame, emitted by the lattice's own poll. The submitter travels with it,
// and the frame hands it back to its source: dropping a submitter waits for the device, so a frame never drops one.
let completions = Completions::new(&lanes);
let returned = Arc::new(Mutex::new(None));
let copy_lane = streams.remove(1);
retire_into(copy_lane, &completions, {
    let returned = returned.clone();
    move |mut stream, ticket| {
        let failure = stream.confirm_retired(ticket);
        Box::new(OnceFrame::new(1, move || {
            *returned.lock().unwrap() = Some((failure, stream));
        }))
    }
})
.unwrap_or_else(|e| panic!("{}", e.error));

// A fake device runs when it is stepped; a real one runs by itself. Here the loop is the worker (`reap` is what the
// source's poll frame does).
gpu.run_until_idle();
while completions.pending() > 0 {
    completions.reap();
    completions.pull(
        &mut |mut frame| {
            frame.execute().unwrap();
            frame.do_finally();
        },
        &mut |_| false,
        u64::MAX,
    );
}
let (failure, copy_lane) = returned.lock().unwrap().take().expect("the continuation ran");
assert_eq!(failure, None); // retired, no failure
assert_eq!(u32::from_le_bytes(out), 1);
drop(copy_lane); // a shutdown call: made here, in the source's stop path, never in a frame
```

## Lanes as sources

A lane's work belongs to a `LatticeSource`: [`open_lane_sources`](src/lanes.rs) opens a device's streams as one pool, makes
the intakes, and returns a `euhedral_graph::LaneSource` per lane. The lattice serializes a source's `request` and
`pull`, so the submitter it owns is only used by one call at a time, on whichever worker runs it, and nothing is locked.
Every stream is a `euhedral_graph::LaneSubmit`: `mark` returns `Ok(None)` for a full marker ring (the source stalls the
lane and marks the item at its next entry), `arm_retirement` returns the retirement's number, `poll_retired` settles the
ticket and reports the device failure, and `raw` lends the stream's handle (the `CUstream`; the queue id for the fake)
to an item that enqueues work of its own, with the context made current by the item (`Device::enter`) and covered by the
next mark.

```rust,no_run
use std::convert::Infallible;

use euhedral_gpu::fake::{FakeGpu, FakeStream};
use euhedral_gpu::{Enqueue, LaneEnqueue, GpuError, LaneSpec, Stream, StreamOptions, open_lane_sources};
use euhedral_graph::{Deliver, LaneCx, LaneId, LaneWork, Marker, Outcome};

/// An item that sets a device word on its lane.
struct Fill {
    at: u64,
    waits: Vec<Marker>,
}

impl LaneWork<FakeStream, ()> for Fill {
    const KINDS: &'static [&'static str] = &["fill"];

    fn kind(&self) -> usize {
        0
    }

    fn waits(&self) -> &[Marker] {
        &self.waits
    }

    fn run(&mut self, cx: &mut LaneCx<'_, FakeStream, (), Self>) -> Outcome<Infallible, GpuError> {
        let at = self.at;
        // SAFETY: `at` is a live allocation of 4 bytes that only this item touches.
        cx.submitter().submit(|q| unsafe { q.memset(at, 7, 4) }).unwrap();
        Outcome::Submitted { retire: false }
    }

    fn failed(self, error: GpuError, _cx: &mut LaneCx<'_, FakeStream, (), Self>) {
        panic!("{error}");
    }
}

let gpu = FakeGpu::new();
let word = gpu.alloc(4);
let lanes = vec![LaneSpec {
    name: "compute".to_owned(),
    options: StreamOptions::default(),
    state: (),
}];
let opened = open_lane_sources::<_, (), Fill>(&gpu, lanes, &[]).unwrap();
// Attach a receiver to each source (`LatticeSource::add_downstream`); a worker's `request` then runs the item.
opened
    .intakes
    .deliver(Fill { at: word, waits: Vec::new() }, Deliver::Lane(LaneId(0)))
    .unwrap();
```

A pool of several backends is built by hand: `LanePoolBuilder::reserve` a range per backend, `Gpu::open_lanes_at` at the
range's first lane, `LanePoolBuilder::set` each stream's `queue()`, then `LaneIntakes::new` and one `LaneSource::new`
per lane. A stream refuses (`Err(Invalid)`) a marker of a lane its own opening call did not make; the lane source hands
such a marker to the item (`LaneCx::bridges`) instead of awaiting it. So lanes of two `open_lanes_at` calls on one
device cannot order on the device: open the lanes that must order each other in one call.

A retirement armed behind a lane that faulted may never get its callback. `poll_retired` therefore looks at the device
once when it finds nothing and retirements are outstanding (one event query, at most every millisecond), which poisons
the device and has the next drain report the rest as poisoned.

The value of a failed retirement does not depend on which of the two (the driver's callback or that look) saw the fault
first. Every retirement that ends because of a device fault, the faulting work's own included, is confirmed as
`Poisoned(cause)`, where `cause` is exactly `Gpu::poisoned()`. A callback that reports a fault status poisons the device
first, if the look has not. The cause is route-independent too: a `Fault` with `call` `"device work"` (the cause says
what failed, not which call noticed), the driver's `code` and its `name`. A backend with no driver call to name records
its own label. A status that is not a device fault (the work failed, the device is intact) is confirmed as `Call`.

Code that imports both `Stream` and `LaneSubmit` names the one it means with a path (`Stream::mark(&mut s)`), as both
have `mark`, `id` and `await_marker`.

`retire_into` is the same shape for a chain of frames: the stream travels inside the continuation instead of living in a
source, and the last frame hands it back to the source that stops it.

## Device attributes

`Gpu::attributes()` returns a `DeviceAttributes`: the multiprocessor count and the per-block shared-memory opt-in limit in
bytes (`CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT` and `CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN`), what a launch
planner needs to size a grid and a dynamic shared-memory request. It is a small typed struct, as `CudaGpu::arch` and
`driver_version` are typed accessors, and not a general attribute call: the CUDA backend reads both once when the device
opens, so the query is a copy that makes no driver call and cannot fail (a failed read, or a value that is not positive, fails the open with `Invalid` naming the attribute; opening needs a driver that knows attribute 97, CUDA 9 or later). The fake reports
`fake::DEFAULT_ATTRIBUTES` until `FakeGpu::set_attributes` sets others; the fake module refuses a kernel opt-in above its opt-in limit. `tests/cuda_device_attributes.rs` reads them from a
real card; `tests/fake_device_attributes.rs` covers the fake.

## Releasing and re-opening a device

A program can release a device completely and open it again in the same process, any number of times. After every holder
below is dropped, the driver's primary context is gone (`Driver::primary_context_active(ordinal)` reports `false`), the
process holds no memory on the card, and the next `CudaGpu::open` (or `Device::open`) makes a new context that works as the
first did. The driver library and NVRTC stay loaded for the process (`Driver::shared`, `Nvrtc::shared`): they hold no
context, and nothing in this crate or `euhedral-memory` keeps a cache or a thread-local keyed by a context. `tests/cuda_reopen.rs`
does ten rounds on a real card; `tests/fake_reopen.rs` does the same on the fake, where `FakeContext` keeps the retains
and releases across opens.

**What holds the context.** One `Arc<Device>` is the primary-context retain (`cuDevicePrimaryCtxRetain` in `Device::open`);
its drop is the only release, and the context is released when the last `Arc<Device>` goes. Strong holders, each directly or
through another:
- `CudaGpu` (it holds the device), and so everything that holds a `CudaGpu`: every `CudaStream` and shared `CudaQueue`
  (a lane), every module and kernel (a kernel holds its module), every graph and so a `GraphCache`'s entries, the shared
  lane handles, and the retirement continuations of `retire_into`.
- `LanePool` and `SharedLanes` (the pool holds its lanes, the lanes hold the gpu): a pool kept alive keeps the context.
- Lane sources and their intakes from `open_lane_sources` (they own the streams), and a stage graph's frames that own a
  pool or a stream.
- `DeviceMemory`, `StreamPool`, `Vmm`, `PinnedHost`, `StreamCopier` and texture objects hold an `Arc<Device>` directly,
  not the gpu: they can outlive it, and then they alone keep the context.
- A `Mover` (its copiers), and the completion tokens of work in flight: an `EventRetired` keeps its copier, a
  `MarkerRetired` keeps a lane's queue, in a `Reaper` or a move task until the work retires.
- A call queued on a device's `DeferredCalls` (its closure holds the device until it runs). A `DeferredCalls` attached to
  a lattice is held by the lattice until it is completed.
Not holders: `Completions` (its pool is weak), a `Block` (a plain address and size), a marker, and the loaded driver
and NVRTC (process-wide, no context).

**The order.** Drop outward in:

1. Emit every marker registration: poll the `Completions` until `pending() == 0`.
2. Complete the source (`LatticeSource::complete`).
3. Drop the lane pool and everything holding lanes: the streams and submitters, the `Mover`, and the lane sources and
   their intakes (complete each lane source that is attached to a lattice, then drop it: completing closes its intake
   and downstream but does not drop its submitter, and the lattice lets go of the source afterwards). Dropping a stream
   waits for its device work.
4. Return the blocks: free device and pinned blocks to their backends, then drop the backends (`PinnedHost`'s drop frees
   what it cached, which waits for the device; `DeviceMemory`, `StreamPool`, `Vmm`, `StreamCopier`).
5. Drop the kernels, modules, graphs and caches of graphs, and any deferral's queue: complete a `DeferredCalls` that is
   attached to a lattice, then drop it (dropping it makes what is queued).
6. Drop the `CudaGpu` and every other `Arc<Device>`. The last drop releases the context, on the thread that drops it, and
   waits for the driver to tear it down.

**Out of order.** Nothing crashes; what is wrong is what stays.
- A holder that outlives the rest keeps the context: the device stays active until that holder drops (a kernel or
  module kept past the gpu keeps its gpu; a `PinnedHost` kept past it keeps the device; a pool keeps its lanes). The
  last holder's drop releases it, wherever that is, except in the two cases below.
- A stream dropped after the pool still waits for its own work, and a marker of a dropped pool is never reached: pending
  `Completions` registrations are leaked and counted (`leaked_unreached`), not emitted (see the `euhedral-graph` README).
- A pinned or device block not returned to its backend before the backend drops is not freed by the backend (a pinned
  backend frees only what it cached); the driver destroys what the context holds when the last `Arc<Device>` goes, so
  nothing outlives the context, but a block's address is a plain number: using it after its backend and device are gone is
  the caller's error, which the crate cannot detect.

**When the context is never released.** Two cases in these crates keep it for the life of the process; `Driver::primary_context_active`
reports `true` and a re-open gets the same context.
- A stream dropped after a sticky fault or a lost context, while retirements it armed are outstanding, deliberately leaks
  its shared side (a callback may still arrive), so its gpu and the device stay. The fault is sticky for the context, so
  the re-open meets the same fault; a program cannot recover in-process.
- A `Completions` marker registration left when its pool went is leaked and forgotten (`leaked_unreached`). If its frame
  owns a holder of the device (a stream through `retire_into`, the typed pool of a `SharedLanes` whose `queues` view went,
  a module, a backend), that holder is never dropped and keeps the context for ever. (A frame that owns the pool itself
  keeps the pool alive, so its registration is not left this way.) The order above avoids it: emit before dropping the pool.

**What the fake shows and does not.** `FakeContext` counts retains and releases across opens. Its streams, modules and
pool hold the gpu, but the real backends of `euhedral-memory` hold the device directly, so the fake has no "gpu dropped, pinned
host alive" except through `FakeContext::hold`, which stands for such a holder. Two opens of a `FakeContext` alive at once
get separate memory, where a second retain of a live primary context shares the first's.

**Timings** on the development card (one round of the test: open the device, compile a module
from source, 32 MiB of device buffers, a pinned loan pair, a copy and a kernel): `CudaGpu::open` about 50 ms (the context
create), the release (the drops above, the last of which destroys the context) about 45 ms; free memory read after each
open was constant over ten rounds.

## The fake device in residency and failure tests

`FakeGpu` can stand in for a card in tests of code that loads, holds and gives up device memory. Nothing was added to the
`Gpu` trait: device memory, a free-memory query and the context belong to `euhedral-memory` and the driver on the real
backend, and `FakeContext` (open, release, open again) already stands for the context. These are fake-only controls and
records, beside `FakeGpu::alloc` and `free`:

- **Bytes.** `bytes_held()` and `peak_bytes()` count what was allocated and not freed, and the most held at once.
  `set_capacity(Some(n))` gives the device a size: `try_alloc` then fails with `Err(Call)` named `OUT_OF_MEMORY` when the
  allocation would hold more, and holds nothing. `free_bytes()` answers as a free-memory query would. `alloc` is
  `try_alloc` that panics.
- **Injected failures.** `fail_allocs(skip, count)` lets `skip` allocations through, then fails the next `count`;
  `fail_module_loads(skip, count)` does the same for `Module::from_source` and `from_image`, which then return
  `Err(Load)`. A call that fails changes nothing on the device.
- **Who called.** `calls()` is `log()` with a `FakeCaller` for each operation: the thread, its name, the CPU the
  submission ran on (where the platform tells) and a label. A frame that wants to be named wraps its work in
  `let _label = FakeGpu::label_caller("name")`; the label is thread-local, nests, and ends with the guard. `memory_log()`
  lists each allocation and free in order with its caller. `log()` is unchanged.

The caller is the one who put the operation on its lane, not the thread that stepped the device.

## Features

| Feature | What it adds |
|---|---|
| `fake` | `FakeGpu`: a deterministic device for tests of code built on this crate; `FakeContext` is its context, opened, fully released and opened again, with the retains and releases counted, and with bytes held, a capacity, injected allocation and module-load failures and the caller of each operation (see "The fake device in residency and failure tests"). The crate's own tests enable it through a dev-dependency on itself. |
| `cuda` | The CUDA backend, on the driver and primary context `euhedral-memory` loads (`Driver::shared`): one library, one context. Linux and Windows. |

The traits and the fake compile on every platform the workspace builds on.

## Build and test

```sh
mise exec -- cargo test -p euhedral-gpu                      # traits, fake backend, layering check: no device needed
mise exec -- cargo test -p euhedral-gpu --features cuda      # + the CUDA backend (its tests skip, passing, without a device)
CUDA_HOME=/path/to/toolkit mise exec -- cargo test -p euhedral-gpu --features cuda   # + kernels compiled from source
mise exec -- cargo run -p euhedral-gpu --features cuda --example graph_replay        # the acceptance program
```

NVRTC is found through `EUHEDRAL_NVRTC` (a library file) or `CUDA_HOME` (a toolkit); without it, tests that compile from
source skip and prebuilt images still load. The tests `tests/cuda_fault.rs` and `tests/cuda_lanes_fault.rs` are processes of their own, because a device
fault is sticky. `tests/cuda_reopen.rs` and `tests/fake_reopen.rs` release a device and open it again ten times. `tests/cuda_lanes.rs` is the lane-source suite on the device, and `tests/fake_lanes.rs` the same on the fake.
