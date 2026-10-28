# euhedral-gpu: asynchronous, backend-neutral GPU access (design)

Date: 2026-10-28. Status: draft for the user's approval.

## Why

Programs on the RustEE lattice need to drive a GPU without ever blocking a frame:
- submit kernels and copies;
- order work across device queues without the host waiting;
- learn about completion as an ordinary lattice event;
- record repeating work once and replay it.

Today RustEE has the generic half: `LanePool`, `Marker`, `DeviceQueue`, `Completions` and `StageGraph` in `euhedral-graph`, plus a CUDA driver, device and copiers in `euhedral-memory`. There is nothing yet behind the device queue that launches work. Each application ends up writing its own submission, completion and interop layer.

The user's direction (2026-10-28):
- a generalized, asynchronous abstraction for accessing the GPU, in RustEE;
- the interface is backend-neutral, with CUDA as the first backend;
- it records and replays CUDA graphs;
- other repositories bring their own CUDA kernels without modifying RustEE.

## What success looks like

An application:
1. Loads its own kernel module, either CUDA C source or a prebuilt image.
2. Opens streams from a shared device and gets them into its `LanePool`.
3. Submits launches and copies from any lattice worker. The host never waits.
4. Lets a frame follow a stream's completion through `Completions`.
5. Records a recurring sequence of submissions once and replays it each time.

It does all of this without a line of RustEE changed, and with one driver context shared with `euhedral-memory`.

## Goals

1. **Backend-neutral traits** for the device, stream, marker, module, kernel, launch, retirement and graph. A CUDA backend implements them first. A fake backend implements them for tests, as `FakeDevice` does today.
2. **Streams as ordered device queues.**
   - `submit(f)` runs launches, copies and memsets on the calling thread with the stream selected for every operation.
   - Stream order carries ordinary dependencies.
   - Each stream implements `DeviceQueue`, so `LanePool`, `Marker` and `Completions::after_marker` work unchanged.
3. **Markers across streams on the device.** `mark` records a marker after the work submitted so far. `await_marker` orders later work behind a marker, on the device; the host does not wait. Markers are reusable and closed when unused.
4. **Retirement without blocking.**
   - `notify_retired(listener)` arms a completion point after the work submitted so far.
   - The listener runs exactly once. On the driver's callback thread it may only enqueue work: no driver calls, no blocking, no frames, no frees.
   - `confirm_retired(ticket)` runs on an ordinary thread and returns the device failure, or none. It never throws.
   - A stage finishing its submission never counts as completion. Host buffers that a copy reads stay alive until retirement is confirmed.
5. **Kernels from outside RustEE.**
   - A `Module` loads from either:
     - backend source (CUDA C, compiled at run time with NVRTC with caller-given options and include paths, and cached by content hash), or
     - a prebuilt image (PTX, cubin or fatbin) the application produced in its own build.
   - A `Kernel` is looked up by symbol name.
   - Arguments are packed through a typed, safe argument builder. Raw pointers to device memory come from `euhedral-memory` handles.
   - Launch geometry (grid, block, dynamic shared memory) is per call.
   - Nothing in RustEE enumerates an application's kernels.
6. **Graph record and replay.**
   - Record a sequence of submissions on a stream into a graph, instantiate it, and replay it on any stream.
   - Arguments can be updated between replays without re-recording, as long as the shape (the topology and the kernels) is unchanged.
   - A fingerprint tells when a recording no longer matches and must be redone.
   - Recording runs beside normal submission: work goes to the stream and to the capture at once, so a failed capture never disturbs the stream's own work.
7. **One driver per process.** The device and its context are shared with `euhedral-memory`'s CUDA backend: one loaded driver and one primary context, not two. Allocation, pinned host memory and stream-ordered alloc and free stay in `euhedral-memory`, and the GPU crate uses them.
8. **Failures are values.**
   - A device fault, such as an illegal address or a launch failure, surfaces through `confirm_retired` and marks the device poisoned. Later submissions fail fast with that cause.
   - Nothing panics on the lattice.

## Non-goals

- Porting any application's kernels, or centralising an application's GPU work. Applications do that on top.
- A second backend in this step. The traits must not assume CUDA-only concepts in their signatures, but only CUDA and the fake are built.
- Multi-GPU scheduling. A device has an ordinal; picking devices is the application's job.
- Graphics and presentation.

## Anti-goals

- A frame that blocks on the device, or a completion found by a host-side wait.
- Any change to RustEE being needed to add a kernel.
- Driver calls from inside the driver's callback.
- A second driver context beside `euhedral-memory`'s.

## Constraints

- RustEE style and rules apply:
  - nothing owns a thread;
  - sources are `LatticeSource`s driven by the lattice's polls;
  - no blocking on the lattice;
  - generic language only. Code, docs and comments name no other repository and no application domain.
- The CUDA driver is loaded at run time, as `euhedral-memory` does today, and so is NVRTC: there is no link-time dependency. A machine without NVRTC can still load prebuilt images.
- Linux first. Windows must compile the traits and the fake backend; the CUDA backend on Windows follows the existing `euhedral-memory` pattern.

## How it fits together (decided)

- **New crate `euhedral-gpu`** (coordinator's call).
  - It holds the traits, the fake backend, and the CUDA backend behind a `cuda` feature.
  - It depends on `euhedral-core`, `euhedral-graph` (for `DeviceQueue`, `Marker`, `LanePool`, `Completions`) and `euhedral-memory` (for the driver, device and allocation).
- **Shared driver:** `euhedral-memory`'s `cuda::Driver` and `Device` grow what the GPU crate needs (streams, events, modules, launches, graphs, host callbacks), or the GPU crate loads the extra entry points against the same driver handle. Either way there is one `dlopen` and one context.
- **Completion path:** retirement listeners feed `Completions` (`register` or `after_marker`), so a continuation frame is emitted by the lattice's own poll. The driver callback only enqueues.
- **Graph capture:** stream capture (begin, end, instantiate, launch, exec update).
  - Recording uses a separate capture stream that receives the same submissions as the real one, so the real stream is never in capture mode.
  - Argument updates use per-node kernel parameter updates. A topology change triggers a re-record, detected by the fingerprint.
- **Kernel sources:**
  - `Module::from_source(src, options)` goes through NVRTC, with a cache under a caller-given directory, keyed by the hash of the source, the options and the device architecture.
  - `Module::from_image(bytes)` loads PTX, cubin or fatbin.
- **Tests:**
  - The fake backend covers ordering, markers, retirement and graph fingerprint logic without a GPU.
  - CUDA integration tests behind the feature use a small kernel supplied by the test itself, both as source and as PTX. This proves that kernels come from outside the crate.
  - The integration tests also cover: cross-stream markers; retirement order; a deliberate device fault surfacing through `confirm_retired` and poisoning the device; and graph replay with updated arguments matching direct submission bit for bit.

## Left to the builder

- Exact trait shapes and error types, and whether the argument builder is a derive or a builder API.
- How the GPU crate obtains the shared driver: extending `euhedral-memory::cuda`, or a small shared driver crate both use.
- Event-ring sizing behind `DeviceQueue::completed` for CUDA streams.
- How graph replay maps onto `LanePool` lane choice.

## Acceptance

- An example program outside the crate's source tree:
  1. compiles its own CUDA C kernel through `Module::from_source`;
  2. runs it on two streams joined by a marker;
  3. records the sequence as a graph and replays it 100 times with changing arguments;
  4. lets a lattice frame follow each replay's completion.

  The results must match a CPU reference, with no host waits on the lattice.
- All existing RustEE tests pass. Clippy `-D warnings` and fmt are clean.
- The fake backend's tests run on a machine without a GPU.
