# euhedral-gpu Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A new workspace crate, `euhedral-gpu`: backend-neutral, asynchronous GPU access on the lattice (streams that are `DeviceQueue`s, cross-stream markers, retirement into `Completions`, kernels from outside RustEE, graph record and replay), with a deterministic fake backend and a CUDA backend that shares `euhedral-memory`'s driver and primary context.

**Architecture:** Traits with associated types (`Gpu`, `Stream`, `Enqueue`, `Module`, `Kernel`, `Sequence`, `Graph`, `RetireListener`) live in the crate root modules. `fake` (feature `fake`) implements them on a stepped, wall-clock-free fake device. `cuda` (feature `cuda`) implements them on the CUDA driver: it takes `euhedral-memory`'s shared `Driver` and `Device`, and looks up its extra entry points in the same loaded library. A recurring sequence is a closure that one code path runs in three passes: record (run for real and capture on shadow streams), probe (fingerprint only) and update (new argument values, positional).

**Tech Stack:** Rust 1.99.1 (edition 2024, `mise.toml`), workspace crates `euhedral-core`, `euhedral-graph`, `euhedral-memory`, `euhedral-hashing`; `libc` (dlopen) for the CUDA backend; CUDA driver API and NVRTC, both loaded at run time.

**Spec:** `docs/superpowers/specs/2026-10-28-euhedral-gpu-design.md` (approved). Read it before any task; this plan argues from it and does not reopen it.

## Decisions made by the planner

The spec left these to the builder; they are settled here.

1. **Shared driver: extend `euhedral-memory::cuda` minimally; the GPU crate loads its own entry points against the same handle.** `Driver` gains `Driver::shared()` (one process-wide `dlopen` + `cuInit`, the same `Arc` every call) and `Driver::symbol(name)` (`dlsym` on the retained handle). `euhedral-gpu` keeps its own table (`cuda::fns::GpuFns`: streams' waits, callbacks, modules, launches, capture, graphs) filled through `Driver::symbol`. It reuses `Driver::fns` for what memory already loads (streams, events, copies, memset). Why:
   - one `dlopen` and one primary context (`Device::open` retains the device's primary context, which is per process);
   - `euhedral-memory` stays about memory: it gains two small functions, not launch or graph semantics;
   - no new crate in the layer table and no move of public types that memory's users import;
   - a lower shared driver crate would mean moving `Driver`, `Device` and the FFI types out of memory and re-exporting them: churn with no gain, because `euhedral-gpu` already sits above memory.
2. **Trait shapes.** Associated types and generic associated types, not trait objects. `Stream::submit(f)` takes a closure that gets `&mut Self::Enqueue<'_>`. A recurring sequence is a closure over `&mut impl Sequence`, run by `Gpu::record`, `Gpu::fingerprint` and `Gpu::update`. Updates are positional: launch n of the update pass sets launch n of the recording. So no launch ids exist.
3. **Errors.** One enum, `GpuError` (Task 2), `Clone + PartialEq`. Device faults are a variant of their own, so poisoning can tell them from call errors.
4. **Argument builder: a builder API, not a derive.** `Args::new().arg(x).device(addr)`. Values are `unsafe trait KernelArg` types: primitives and arrays are built in, and an application implements it for its own `#[repr(C)]` structs. CUDA checks the argument count and sizes against `cuFuncGetParamCount`/`cuFuncGetParamInfo` when the driver has them (it does here: driver 615.71.09).
5. **Enqueue operations are `unsafe fn`s,** like `euhedral_memory::Copier::start`. A kernel or a copy can touch any device-visible memory, pinned host memory included. The builder and the geometry are safe.
6. **Event-ring sizing behind `DeviceQueue::completed`.** Each CUDA stream has a ring of `StreamOptions::markers` events (default **256**, `CU_EVENT_DISABLE_TIMING`). Marker values count from 1. A slot is reused only once its previous value has completed. `mark()` on a full ring first rescans; if the ring is still full it returns `GpuError::Busy`, which a stage maps to `StageError::Retry`. `completed()` scans upward from the last completed value with `cuEventQuery` and stops at the first one not ready. If an await races a ring wrap, it may wait for *later* work on the marked stream. It never waits for earlier work and never forms a cycle: that later work was enqueued before the wait. After a device fault, `completed()` reports every marked value. No continuation is stranded, and each one learns the fault from `confirm_retired` or `Gpu::poisoned`.
7. **Graph replay and `LanePool` lane choice.** A replay is one submission on the stream it is launched on. Its lane marker is that stream's `mark()` after `Graph::replay`. Lane choice is unchanged: a stage chooses its lane as before and replays on that lane's stream. The graph's inner branches are invisible to the pool. `record` runs the sequence for real once, exactly as a replay on `streams[0]` would run: the other streams fork from `streams[0]` at the start and join it at the end.
8. **Retirement uses `cuStreamAddCallback`, not `cuLaunchHostFunc`.** The driver documents that the callback runs exactly once and receives the device error. `cuLaunchHostFunc` is skipped when the context has an error, which would strand a continuation. Retirement is never captured into a graph, so `cuStreamAddCallback`'s capture restriction does not matter. If a future driver drops the symbol, `GpuFns` falls back to `cuLaunchHostFunc`. `retire_into` also emits a continuation once the device is poisoned, so even then nothing is stranded.
9. **Stream exclusivity is a per-stream `Mutex`, held only while enqueueing.** It gives each submission contiguous ops on the stream. The critical section is a few non-blocking driver enqueue calls, measured in microseconds. A call on a stream from inside its own `submit` closure returns `GpuError::Invalid("re-entered")`, detected by a thread token, and never deadlocks. `record` locks its streams in ascending lane order for the closure's duration, so nothing foreign is captured.
10. **Shape and fingerprint.** The shape is the op kinds in order, each op's stream index, `order` edges, kernel identity (module key + symbol), launch geometry, argument *layout* (count, sizes, alignments), and copy and memset parameters (addresses, sizes, value). Only kernel argument *values* update in place. Any other change is `GpuError::ShapeChanged` and calls for `rerecord`. The fingerprint is 128 bits: two seeded `euhedral_hashing::hasher_api::hash_bytes_with_seed` over one shape byte string.
11. **Capture.** `cuStreamBeginCapture_v2` with `CU_STREAM_CAPTURE_MODE_RELAXED` (2), so other lattice workers' legal driver calls are never refused during a capture. The launch node is read right after each shadow launch with `cuStreamGetCaptureInfo_v3`, falling back to `_v2`. The single current dependency is that node.
12. **Re-record.** `Gpu::rerecord` captures a new graph and tries `cuGraphExecUpdate_v2` on the existing executable. On success it keeps the *original* `CUgraph` and its node handles: parameter updates name nodes of the graph the executable was instantiated from. The new graph is destroyed. On any update result other than success, it instantiates the new graph and replaces both. It returns `Rerecorded::Updated` or `Rerecorded::Instantiated`.
13. **NVRTC.**
    - Loaded with `dlopen` (no link-time dependency) through a search order (Task 12). The first success serves the process.
    - It compiles to **cubin for the device's exact architecture** (`--gpu-architecture=sm_XY`) when NVRTC lists that architecture (`nvrtcGetSupportedArchs`). Otherwise it emits **PTX for the newest `compute_XY` it supports**, which the driver JITs forward.
    - The cache key is a 128-bit hash of: format tag, source, options in order, in-memory headers, include-path strings, target string, NVRTC version.
    - Cache files are written atomically (temp file, then rename). A file that fails to load is recompiled and overwritten.
14. **Module loading is a build-time call,** like `euhedral_memory::Backend::alloc`. It blocks: compile, file reads, `cuModuleLoadDataEx`. Call it off the lattice's workers: at start-up, or on a lattice set aside for blocking work. Cache file IO therefore uses `std::fs` under a reasoned `#[allow(clippy::disallowed_methods)]`, not `euhedral-io`. Routing the reads through `AsyncIo` would not make loading non-blocking, because NVRTC and the driver's module load block anyway.
15. **Fake backend.** Feature `fake`, enabled for the crate's own tests through a self dev-dependency, as `euhedral-graph`'s `testing` is. It is a stepped device with no wall clock: `FakeGpu::step()` runs the oldest runnable op of any stream. Fake kernels are host closures registered by name and run on a bounds-checked fake memory. A kernel fault poisons the device. Tests are deterministic and never time anything.
16. **The acceptance example is `euhedral-gpu/examples/graph_replay.rs`** (`required-features = ["cuda"]`), with its kernel in `euhedral-gpu/examples/kernels/pipeline.cu`. That is outside the crate's `src` tree. It is not a separate workspace crate because of feature unification: a member that enables `euhedral-gpu/cuda` turns on `euhedral-memory/cuda` under `cargo test --workspace`, and that does not compile on Windows.
17. **Layering check: an integration test, `euhedral-gpu/tests/layering.rs`.** It reads every workspace member's `Cargo.toml` with a small line parser that covers all dependency tables, dev, build, target and dotted forms. It asserts that every member has a layer and that every `euhedral-*` dependency is strictly lower. It needs no new dependency, and it runs in CI on both operating systems.
18. **Sticky device faults poison the CUDA context for the whole process,** so the fault test is its own test binary (`tests/cuda_fault.rs`) with one test.
19. **CI** gains a Linux-only step that runs Clippy and the tests of `euhedral-gpu` with `--features cuda`. CUDA tests skip, passing, with no driver, so the CUDA code is at least compiled and linted on every push.
20. **Dependencies.** The library uses `euhedral-core`, `euhedral-graph`, `euhedral-memory`, `euhedral-hashing`, and `libc` (with `cuda`). `euhedral-hardware-utils` is a dev-dependency (the example picks a CPU for its lattice). The library does not need `euhedral-data-structures` or `euhedral-io`. They stay free to add when a need appears.

**Phase order note:** the backend-neutral `Args` builder and `Launch` geometry move into Phase 1, because the traits and the fake backend need them. Phase 4 is the CUDA side of launching: kernel lookup, the parameter check and `cuLaunchKernel`.

## Global Constraints

- Layering, no cycles (user): `euhedral-data-structures`, `euhedral-hashing`, `euhedral-hardware-utils` → `euhedral-core` → `euhedral-graph`, `euhedral-io` → `euhedral-memory` → `euhedral-gpu`. Nothing below `euhedral-gpu` depends on it, not even through a dev-dependency or a feature. Task 1's test enforces it.
- No dev-dependency of any crate may enable `euhedral-memory/cuda` or `euhedral-gpu/cuda`. Feature unification would break the Windows `cargo test --workspace`. CUDA tests and examples use `#![cfg(feature = "cuda")]` / `required-features = ["cuda"]`.
- Language: code, docs, comments and READMEs name no other repository and no application domain. No negative self-descriptions (write "builds on X", never "does not depend on Y"). Commit messages may name repositories.
- Nothing owns a thread. Nothing blocks the lattice: submission, marking, awaiting, replay and updates are non-blocking driver calls. Module loading and stream/graph drops are build-time or shutdown calls, documented as such.
- Driver callbacks only enqueue: no driver calls, no blocking, no frame built or dropped. A panic inside a listener is caught (`catch_unwind`) and never crosses the FFI boundary.
- Failures are values. Nothing panics on the lattice. A device fault poisons the device, and later submissions fail fast with `GpuError::Poisoned(cause)`.
- Capture happens on a shadow stream per recorded stream; the real stream is never in capture mode. Parameter updates go through `cuGraphExecKernelNodeSetParams_v2`. The fingerprint detects shape changes.
- Kernels come from outside RustEE: `from_source` (NVRTC, cache under a caller-given dir) or `from_image` (PTX, cubin, fatbin). Nothing in RustEE enumerates kernels.
- The fake backend's tests run with no GPU. No test asserts timing.
- Crate lints as `euhedral-memory`: `unsafe_op_in_unsafe_fn = "deny"`, `missing_debug_implementations = "warn"`, clippy `undocumented_unsafe_blocks`, `dbg_macro`, `todo` = "warn"; every `unsafe` block has a `// SAFETY:` comment. `euhedral-gpu/clippy.toml` copies `euhedral-graph/clippy.toml` with "euhedral-io's AsyncIo" kept in the reasons.
- Gate before every commit: `cargo fmt --all --check` and `cargo clippy --workspace --all-targets -- -D warnings`. Once `cuda` exists (Task 9), also run `cargo clippy -p euhedral-gpu --all-targets --features cuda -- -D warnings`.
- Commands tagged **[GPU]** touch the device. On the development machine, run them through the machine's GPU queue, as the executor's brief says. Use the cargo target directory the brief gives.
- Where NVRTC lives on the development machine: it is not on the loader path. A CUDA 13.1 toolkit is at `/home/brandon/qwen38-work/cuda-13.1` (`lib64/libnvrtc.so.13`, with `libnvrtc-builtins.so.13.1` beside it). [GPU] commands that compile from source set `CUDA_HOME=/home/brandon/qwen38-work/cuda-13.1`. The driver is 615.71.09 (CUDA 13.4 user-mode driver); the device is an RTX 5070 Ti, 16 GB, compute capability 12.0 (sm_120).

## Review Focus

1. **Two lattice workers submitting to one stream at once.** Each submission's ops must stay contiguous on the stream, and neither call may deadlock or return an error. Test: `concurrent_submissions_to_one_stream_stay_contiguous` (Task 5).
2. **More marks outstanding than the ring holds, and awaits on old markers.** `mark()` returns `Busy` rather than reusing a live slot. An await on a marker that has already completed adds no device wait. Test: `a_full_marker_ring_is_busy_and_old_markers_need_no_wait` (Task 16, ring of 4).
3. **A retirement whose ticket is never confirmed, and a device that faults before the callback.** Kept host buffers are released when the stream drops. A `retire_into` continuation is emitted once the device is poisoned. Tests: `kept_values_live_until_confirmed_or_the_stream_drops` and `a_poisoned_device_still_emits_retire_into_continuations` (Tasks 5 and 6).
4. **A corrupt or truncated cache file, and two processes writing one key.** The file is recompiled and overwritten, not reported as an error. Writes are atomic renames. Tests: `a_corrupt_cache_entry_is_rebuilt` (Task 14) and `stores_are_atomic_renames` (Task 13).
5. **Argument updates while an earlier replay is still running.** Replay i must read its own arguments, not those set for i+1. Test: `updates_between_replays_do_not_touch_a_replay_in_flight` (Task 19).

---

## File Structure

```
euhedral-gpu/
  Cargo.toml            features: fake, cuda; lints; [[example]] graph_replay (required-features cuda)
  clippy.toml           the control plane's no-thread, no-wait bans
  README.md             (Task 23)
  src/lib.rs            module list, re-exports, README doctest hook (Task 23)
  src/error.rs          GpuError
  src/poison.rs         Poison: a device's first fault, set once
  src/stream_lock.rs    StreamLock<T>: a Mutex that reports re-entry from its own holder
  src/shape.rs          Fingerprint, ShapeHasher, CopyKind
  src/args.rs           KernelArg, Args, Dim3, Launch
  src/api.rs            the traits, StreamOptions, CompileOptions, RetireTicket, RetireListener, Rerecorded
  src/retire.rs         retire_into (into Completions), Kept (values held until a retirement is confirmed)
  src/module_cache.rs   CacheKey, KeyMaterial, ModuleCache (backend-neutral, no GPU needed to test)
  src/fake/mod.rs       FakeGpu, FakeMemory, FakeFault, FakeOp log
  src/fake/stream.rs    FakeStream, FakeEnqueue
  src/fake/module.rs    FakeModule, FakeKernel
  src/fake/graph.rs     FakeSequence, FakeGraph
  src/cuda/mod.rs       CudaGpu, fault classification
  src/cuda/fns.rs       GpuFns: entry points looked up through Driver::symbol; FFI structs
  src/cuda/stream.rs    CudaStream, lane rings, CudaEnqueue
  src/cuda/retire.rs    the callback trampoline
  src/cuda/nvrtc.rs     Nvrtc: search, load, compile
  src/cuda/module.rs    CudaModule, CudaKernel
  src/cuda/graph.rs     CudaSequence, CudaGraph
  tests/layering.rs     the workspace layer check
  tests/fake_streams.rs ordering, markers, retirement, poisoning on the fake
  tests/fake_graph.rs   record, fingerprint, update, rerecord on the fake
  tests/cuda.rs         CUDA streams, retirement, modules, launches, cross-stream (skips with no driver)
  tests/cuda_graph.rs   CUDA record/replay/update/rerecord (skips with no driver)
  tests/cuda_fault.rs   one deliberate fault (its own process)
  tests/kernels/ops.cu  the tests' kernels; ops.ptx generated from it and committed
  examples/graph_replay.rs, examples/kernels/pipeline.cu   the acceptance program
euhedral-memory/src/cuda/mod.rs   Driver::shared, Driver::symbol (Task 8)
Cargo.toml (workspace members), .github/workflows/ci.yml (Task 9), README.md, CHANGELOG.md, GLOSSARY.md, euhedral-memory/README.md (Task 23)
```

---

## Phase 1: skeleton, layering check, traits, fake backend

Phase gate: `cargo test -p euhedral-gpu` green on a machine with no GPU; fmt and workspace clippy clean.

### Task 1: Crate skeleton and the layering test

**Files:**
- Create: `euhedral-gpu/Cargo.toml`, `euhedral-gpu/clippy.toml`, `euhedral-gpu/src/lib.rs`, `euhedral-gpu/tests/layering.rs`
- Modify: `Cargo.toml` (append `"euhedral-gpu"` to `members`)

**Interfaces:**
- Produces: the package `euhedral-gpu` with features `fake = []` and `cuda = ["euhedral-memory/cuda", "dep:libc"]`, default none.
  - Dependencies: `euhedral-core`, `euhedral-graph` (`default-features = false`), `euhedral-memory` (`default-features = false`), `euhedral-hashing`, and `libc = { version = "0.2.174", optional = true }`.
  - Dev-dependencies: `euhedral-gpu = { path = ".", default-features = false, features = ["fake"] }` and `euhedral-hardware-utils`.

- [ ] **Step 1: Write the failing test** `euhedral-gpu/tests/layering.rs` (file-level `#![allow(clippy::disallowed_methods, reason = "the test reads the workspace's manifests")]`):

```rust
const LAYERS: &[(&str, u8)] = &[
    ("euhedral-data-structures", 0), ("euhedral-hashing", 0), ("euhedral-hardware-utils", 0),
    ("euhedral-core", 1), ("euhedral-graph", 2), ("euhedral-io", 2),
    ("euhedral-memory", 3), ("euhedral-gpu", 4),
];

/// Every `euhedral-*` name declared in any dependency table of `manifest`.
fn euhedral_deps(manifest: &str) -> Vec<String>;

#[test]
fn the_parser_finds_every_kind_of_dependency_table() {
    let m = "[package]\nname = \"x\"\n[dependencies]\neuhedral-core = { path = \"../c\" }\nlog = \"1\"\n\
             [dev-dependencies]\neuhedral-graph.workspace = true\n\
             [target.'cfg(unix)'.dependencies]\neuhedral-io = { path = \"../io\" }\n\
             [build-dependencies]\neuhedral-hashing = \"*\"\n[dependencies.euhedral-memory]\npath = \"../m\"\n\
             [features]\neuhedral-gpu = []\n";
    let mut got = euhedral_deps(m);
    got.sort();
    assert_eq!(got, ["euhedral-core", "euhedral-graph", "euhedral-hashing", "euhedral-io", "euhedral-memory"]);
}

#[test]
fn every_member_has_a_layer() { /* members from ../Cargo.toml `members = [...]`; each in LAYERS, message "add <name> to LAYERS" */ }

#[test]
fn crates_depend_only_on_lower_layers() {
    // for each member: for each dep in euhedral_deps(its Cargo.toml), dep != member => layer(dep) < layer(member),
    // failure message "<member> (layer a) depends on <dep> (layer b)"
}
```

Parser rule: a table header is a dependency table if, without its brackets, it ends with `dependencies` or contains `dependencies.`. In a dotted header the name follows `dependencies.`. Inside a dependency table, a key is the text before the first `=` or `.`, trimmed. Paths come from `env!("CARGO_MANIFEST_DIR")` + `/..`.

- [ ] **Step 2: Run it to see it fail.** Run `cargo test -p euhedral-gpu --test layering`. Expected: a compile error, because the package does not exist yet.
- [ ] **Step 3: Create the package.**
  - `Cargo.toml`: features, dependencies and lints as above. Use the `[lints]` blocks copied from `euhedral-memory/Cargo.toml`, `description = "Asynchronous, backend-neutral GPU access on the Euhedral control plane: streams as device lanes, markers, retirement as frames, kernels loaded at run time, graph record and replay"`, `license = "Apache-2.0"` and `readme = "README.md"`.
  - `clippy.toml`: copy `euhedral-graph/clippy.toml`.
  - `src/lib.rs`: a one-paragraph crate doc comment.
  - Add the crate to the workspace `members`.
- [ ] **Step 4: Run the tests.** Run `cargo test -p euhedral-gpu --test layering`. Expected: 3 passed. Then run `cargo clippy --workspace --all-targets -- -D warnings`. Expected: clean.
- [ ] **Step 5: Commit.** `git add Cargo.toml Cargo.lock euhedral-gpu && git commit -m "euhedral-gpu: crate skeleton and the workspace layering test"`

### Task 2: Errors, poison, the stream lock, shape fingerprints

**Files:**
- Create: `euhedral-gpu/src/error.rs`, `src/poison.rs`, `src/stream_lock.rs`, `src/shape.rs`
- Modify: `src/lib.rs`

**Interfaces:**
- Produces:
```rust
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GpuError {
    Call { call: &'static str, code: i32, name: String },   // a backend call failed
    Fault { call: &'static str, code: i32, name: String },  // a device fault: the device is poisoned
    Poisoned(Box<GpuError>),                                // fail-fast after a fault; the cause
    Busy,                                                   // a resource is full (marker ring): retry
    Invalid(String),                                        // misuse: re-entry, a bad stream index, a foreign marker
    Compile { log: String },
    Load(String),                                           // a library or an image that will not load
    NotFound(String),                                       // a kernel symbol
    Args(String),                                           // arguments that do not fit the kernel
    Capture(String),                                        // a recording failed; the real work was unaffected
    ShapeChanged { recorded: Fingerprint, now: Fingerprint },
}
impl Display, std::error::Error for GpuError

pub struct Poison { /* OnceLock<GpuError> */ }
impl Poison { pub fn new() -> Self; pub fn set(&self, cause: GpuError) /* first wins */; pub fn get(&self) -> Option<GpuError>;
              pub fn check(&self) -> Result<(), GpuError> /* Err(Poisoned(Box::new(cause))) */ }

pub struct StreamLock<T> { /* Mutex<T> + AtomicU64 holder token */ }
impl<T> StreamLock<T> { pub fn new(v: T) -> Self; pub fn lock(&self) -> Result<StreamGuard<'_, T>, GpuError> }
// StreamGuard: Deref/DerefMut; Err(Invalid("re-entered")) when the calling thread already holds it.
// Thread token: address of a thread_local! u8, as u64. A poisoned Mutex is recovered (PoisonError::into_inner).

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)] pub struct Fingerprint(pub u64, pub u64);
#[derive(Clone, Copy, Debug, PartialEq, Eq)] pub enum CopyKind { ToDevice, ToHost, OnDevice }
#[derive(Debug, Default)] pub struct ShapeHasher { /* Vec<u8> */ }
impl ShapeHasher {
    pub fn launch(&mut self, stream: usize, kernel: u64, launch: &Launch, layout: u64);
    pub fn copy(&mut self, stream: usize, kind: CopyKind, dst: u64, src: u64, bytes: u64);
    pub fn memset(&mut self, stream: usize, dst: u64, value: u8, bytes: u64);
    pub fn order(&mut self, after: usize, before: usize);
    pub fn finish(&self) -> Fingerprint; // seeds 0x6770_7573_6861_7065 and 0x6670_7269_6e74_0001
}
```
(`Launch` comes from Task 3. Write `shape.rs` against it, and land Tasks 2 and 3 in one commit if it is easier to compile them together.)

- [ ] **Step 1: Write failing unit tests** (in each file's `#[cfg(test)] mod tests`):
  - `poison_keeps_the_first_cause`: set A, then B; `get() == Some(A)`; `check() == Err(Poisoned(Box::new(A)))`.
  - `a_stream_lock_reports_reentry_and_serves_other_threads`: a nested `lock()` on the same thread is `Err(Invalid(_))`. After the guard drops, another thread (`std::thread::scope` under `#[allow(clippy::disallowed_methods)]`) locks it.
  - `fingerprints_differ_by_any_shape_field_and_not_by_nothing`: two identical sequences give equal fingerprints. Each of these gives a different one: launch geometry, layout, kernel id, stream index, copy address, an extra `order`.
- [ ] **Step 2: Run** `cargo test -p euhedral-gpu --lib`. Expected: FAIL (the items are not defined).
- [ ] **Step 3: Implement the types.** `ShapeHasher` appends a tag byte per op kind, then the fields as little-endian bytes.
- [ ] **Step 4: Run** `cargo test -p euhedral-gpu --lib`. Expected: PASS.
- [ ] **Step 5: Commit.** `git commit -m "euhedral-gpu: errors, poison, the stream lock and shape fingerprints"`

### Task 3: The typed argument builder and launch geometry

**Files:**
- Create: `euhedral-gpu/src/args.rs`
- Modify: `src/lib.rs`

**Interfaces:**
- Produces:
```rust
/// # Safety
/// Plain data: `Copy`, no pointers into host memory the device cannot see, every byte initialized, the device's
/// layout of the type equals Rust's (`#[repr(C)]` for structs).
pub unsafe trait KernelArg: Copy + Send + Sync + 'static {}
// built in: u8 u16 u32 u64 i8 i16 i32 i64 f32 f64 usize isize, [T; N] where T: KernelArg

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Args { /* storage: Vec<u128> (16-byte aligned), slots: Vec<(u32 offset, u16 size, u16 align)> */ }
impl Args {
    pub fn new() -> Self;
    pub fn arg<T: KernelArg>(mut self, value: T) -> Self;   // align_of::<T>() <= 16, else panics at build: a build-time misuse
    pub fn device(self, addr: u64) -> Self;                   // a device address (from euhedral_memory::Block::addr)
    pub fn len(&self) -> usize; pub fn is_empty(&self) -> bool;
    pub fn sizes(&self) -> impl Iterator<Item = usize> + '_;
    pub fn layout(&self) -> u64;                              // hash of (size, align) per slot
    pub fn read<T: KernelArg>(&self, index: usize) -> Option<T>; // None on a size mismatch
    pub(crate) fn pointers(&self) -> Vec<*mut std::ffi::c_void>; // one pointer per slot into storage
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)] pub struct Dim3 { pub x: u32, pub y: u32, pub z: u32 }
impl Dim3 { pub const fn x(x: u32) -> Self; pub const fn new(x: u32, y: u32, z: u32) -> Self }
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Launch { pub grid: Dim3, pub block: Dim3, pub shared_bytes: u32 }
impl Launch { /// ceil(n / block) blocks of `block` threads in x.
              pub fn linear(n: u32, block: u32) -> Self }
```

- [ ] **Step 1: Write failing tests:**
  - `arguments_read_back_by_index_and_type`: `Args::new().arg(3u32).arg(2.5f32).device(0x1000).arg([1u16; 3])`. `read::<u32>(0) == Some(3)`, `read::<f32>(1) == Some(2.5)`, `read::<u64>(2) == Some(0x1000)`, `read::<u64>(0) == None`, `len() == 4`.
  - `pointers_are_aligned_to_each_argument`: each `pointers()[i] as usize % align == 0`.
  - `layout_ignores_values_and_sees_types`: equal `layout()` for `arg(1u32)` and `arg(2u32)`; different for `arg(1u64)`.
  - `linear_launch_rounds_up`: `Launch::linear(1000, 256).grid == Dim3::x(4)`.
- [ ] **Step 2: Run** `cargo test -p euhedral-gpu --lib args`. Expected: FAIL.
- [ ] **Step 3: Implement** in `src/args.rs`. Each slot starts at the next multiple of its alignment; values are written with `ptr::copy_nonoverlapping`.
- [ ] **Step 4: Run** the same command. Expected: PASS.
- [ ] **Step 5: Commit.** `git commit -m "euhedral-gpu: typed kernel arguments and launch geometry"`

### Task 4: The backend-neutral traits and `retire_into`

**Files:**
- Create: `euhedral-gpu/src/api.rs`, `src/retire.rs`
- Modify: `src/lib.rs` (re-export everything public at the root)

**Interfaces:**
- Consumes: `euhedral_graph::{DeviceQueue, LaneId, LanePool, Marker, Completions, Pending}` and `euhedral_core::frames::Frame`.
- Produces:
```rust
pub trait Gpu: Send + Sync + fmt::Debug + Sized + 'static {
    type Stream: Stream<Gpu = Self>;
    type Module: Module<Gpu = Self>;
    type Kernel: Kernel;
    type Graph: Graph<Gpu = Self>;
    type Sequence<'a>: Sequence<Kernel = Self::Kernel> where Self: 'a;
    /// Streams `names[i]` as lane i of a new pool (1 to 64 names).
    fn open_lanes(self: &Arc<Self>, names: &[&str], options: StreamOptions)
        -> Result<(Arc<LanePool>, Vec<Arc<Self::Stream>>), GpuError>;
    fn poisoned(&self) -> Option<GpuError>;
    /// Runs `f` once for real (as a replay on `streams[0]` would) and records it.
    fn record<F>(&self, streams: &[&Arc<Self::Stream>], f: F) -> Result<Self::Graph, GpuError>
        where F: FnOnce(&mut Self::Sequence<'_>) -> Result<(), GpuError>;
    fn rerecord<F>(&self, graph: &mut Self::Graph, streams: &[&Arc<Self::Stream>], f: F) -> Result<Rerecorded, GpuError>
        where F: FnOnce(&mut Self::Sequence<'_>) -> Result<(), GpuError>;
    /// The fingerprint `f` would record, over `streams` streams; nothing reaches the device.
    fn fingerprint<F>(&self, streams: usize, f: F) -> Result<Fingerprint, GpuError>
        where F: FnOnce(&mut Self::Sequence<'_>) -> Result<(), GpuError>;
    /// New argument values, launch n of `f` to launch n of the recording; Err(ShapeChanged) applies nothing.
    fn update<F>(&self, graph: &mut Self::Graph, f: F) -> Result<(), GpuError>
        where F: FnOnce(&mut Self::Sequence<'_>) -> Result<(), GpuError>;
}
pub trait Stream: DeviceQueue + Sized + 'static {
    type Gpu: Gpu<Stream = Self>;
    type Enqueue<'a>: Enqueue<Kernel = <Self::Gpu as Gpu>::Kernel> where Self: 'a;
    fn gpu(&self) -> &Arc<Self::Gpu>;
    fn lane(&self) -> LaneId;
    fn submit<R>(&self, f: impl FnOnce(&mut Self::Enqueue<'_>) -> Result<R, GpuError>) -> Result<R, GpuError>;
    fn mark(&self) -> Result<Marker, GpuError>;
    fn await_marker(&self, marker: Marker) -> Result<(), GpuError>;
    fn notify_retired(&self, listener: Box<dyn RetireListener>) -> Result<(), GpuError>;
    fn confirm_retired(&self, ticket: RetireTicket) -> Option<GpuError>;
}
pub trait Enqueue {
    type Kernel: Kernel;
    /// # Safety: `args` match the kernel's parameters; every address it touches is valid for that access until retired.
    unsafe fn launch(&mut self, kernel: &Self::Kernel, launch: Launch, args: &Args) -> Result<(), GpuError>;
    /// # Safety (all copies and memset): the ranges are valid, unaliased by other work, and untouched until retired.
    unsafe fn copy_to_device(&mut self, dst: u64, src: *const u8, bytes: usize) -> Result<(), GpuError>;
    unsafe fn copy_to_host(&mut self, dst: *mut u8, src: u64, bytes: usize) -> Result<(), GpuError>;
    unsafe fn copy_on_device(&mut self, dst: u64, src: u64, bytes: usize) -> Result<(), GpuError>;
    unsafe fn memset(&mut self, dst: u64, value: u8, bytes: usize) -> Result<(), GpuError>;
    /// Keeps `value` alive until the next retirement armed on this stream is confirmed (in a recording: until the graph drops).
    fn keep(&mut self, value: Box<dyn Any + Send>);
}
pub trait Kernel: Clone + Send + Sync + fmt::Debug { fn name(&self) -> &str; fn id(&self) -> u64; }
pub trait Module: Send + Sync + fmt::Debug + Sized {
    type Gpu: Gpu<Module = Self>;
    fn from_source(gpu: &Arc<Self::Gpu>, source: &str, options: &CompileOptions) -> Result<Self, GpuError>;
    fn from_image(gpu: &Arc<Self::Gpu>, image: &[u8]) -> Result<Self, GpuError>;
    fn kernel(&self, name: &str) -> Result<<Self::Gpu as Gpu>::Kernel, GpuError>;
}
pub trait Sequence {
    type Kernel: Kernel;
    type Enqueue<'a>: Enqueue<Kernel = Self::Kernel> where Self: 'a;
    fn submit(&mut self, stream: usize, f: impl FnOnce(&mut Self::Enqueue<'_>) -> Result<(), GpuError>) -> Result<(), GpuError>;
    /// Later work on `before` waits for the work submitted so far on `after`.
    fn order(&mut self, after: usize, before: usize) -> Result<(), GpuError>;
}
pub trait Graph: Send + Sync + fmt::Debug {
    type Gpu: Gpu<Graph = Self>;
    fn fingerprint(&self) -> Fingerprint;
    fn launches(&self) -> usize;
    fn replay(&self, stream: &<Self::Gpu as Gpu>::Stream) -> Result<(), GpuError>;
}
pub trait RetireListener: Send + 'static {
    /// Once, on the backend's callback thread: enqueue only (no driver calls, no blocking, no frames).
    fn retired(self: Box<Self>, ticket: RetireTicket);
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RetireTicket { pub lane: LaneId, pub seq: u64, pub status: i32 } // status 0: the work before it succeeded
impl RetireTicket { pub fn ok(&self) -> bool }
#[derive(Clone, Copy, Debug)] pub struct StreamOptions { pub markers: u32 } // Default: 256
#[derive(Clone, Debug, Default)]
pub struct CompileOptions {
    pub name: String,                    // the program's name in logs ("module.cu" when empty)
    pub options: Vec<String>,            // passed as given; the backend adds the architecture option
    pub include_paths: Vec<PathBuf>,     // files under them are not part of the cache key
    pub headers: Vec<(String, String)>,  // (include name, contents): part of the cache key
    pub cache_dir: Option<PathBuf>,      // None: no cache
    pub compiler: Option<PathBuf>,       // the run-time compiler library to load, else the search order
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)] pub enum Rerecorded { Updated, Instantiated }

// src/retire.rs
/// Arms a retirement on `stream` whose continuation `then(ticket)` is emitted by `completions`; emitted at once
/// if the device is poisoned before the callback runs.
pub fn retire_into<S: Stream>(stream: &Arc<S>, completions: &Completions,
    then: impl FnOnce(RetireTicket) -> Box<dyn Frame> + Send + 'static) -> Result<(), GpuError>;
/// Values held until a retirement is confirmed: (retirement seq, value), released in order.
pub(crate) struct Kept { /* Mutex<VecDeque<(u64, Box<dyn Any + Send>)>> */ }
impl Kept { pub fn push(&self, seq: u64, v: Box<dyn Any + Send>); pub fn release_through(&self, seq: u64); pub fn clear(&self) }
```
`retire_into` mechanics:
- An `Arc<RetireCell { done: AtomicBool, seq: AtomicU64, status: AtomicI32 }>` serves as the listener. It stores `seq` and `status`, then `done` (Release).
- A `Pending` registered on `completions` polls the cell: when it is done, `then(ticket)`. If not, and `stream.gpu().poisoned()` is set, `then(RetireTicket { status: POISONED_STATUS, .. })` with `pub const POISONED_STATUS: i32 = -1`.
- `confirm_retired` maps `POISONED_STATUS` to `Some(Poisoned(cause))`.

- [ ] **Step 1: Write failing unit tests** in `src/retire.rs`:
  - `kept_values_release_in_order_through_a_seq`: push (1,a), (1,b), (2,c); `release_through(1)` drops a and b (count with an `Arc` strong count), not c.
  - `retire_cell_ticket_round_trips`: a listener stores (seq 7, status 0); the pending yields `then`'s frame with `ticket.seq == 7 && ticket.ok()`. The pending takes its poison check as a closure (`Box<dyn Fn() -> bool + Send>`; `retire_into` passes one that reads `stream.gpu().poisoned()`), so this test needs no backend. A second case: with the closure returning true and the cell not done, the frame comes at once with `status == POISONED_STATUS`.
- [ ] **Step 2: Run** `cargo test -p euhedral-gpu --lib retire`. Expected: FAIL.
- [ ] **Step 3: Implement** `api.rs` and `retire.rs` as above. `RetireCell`'s pending is a private type in `retire.rs`.
- [ ] **Step 4: Run** `cargo test -p euhedral-gpu --lib` and `cargo clippy -p euhedral-gpu --all-targets -- -D warnings`. Expected: PASS, clean.
- [ ] **Step 5: Commit.** `git commit -m "euhedral-gpu: backend-neutral traits and retirement into Completions"`

### Task 5: Fake device, streams, markers and retirement

**Files:**
- Create: `euhedral-gpu/src/fake/mod.rs`, `src/fake/stream.rs`, `src/fake/module.rs`, `euhedral-gpu/tests/fake_streams.rs`
- Modify: `src/lib.rs` (`#[cfg(feature = "fake")] pub mod fake;`)

**Interfaces:**
- Consumes: Tasks 2–4.
- Produces:
```rust
pub type FakeKernelFn = Arc<dyn Fn(&mut FakeMemory, &Launch, &Args) -> Result<(), FakeFault> + Send + Sync>;
#[derive(Clone, Copy, Debug, PartialEq, Eq)] pub enum FakeFault { IllegalAddress /*code 700*/, Trap /*code 710*/ }
pub struct FakeMemory { /* BTreeMap<u64 base, Vec<u8>> */ }
impl FakeMemory { pub fn bytes(&self, addr: u64, len: usize) -> Result<&[u8], FakeFault>;
                  pub fn bytes_mut(&mut self, addr: u64, len: usize) -> Result<&mut [u8], FakeFault>; } // a range inside one allocation, else IllegalAddress
#[derive(Clone, Debug, PartialEq, Eq)] pub enum FakeOp { Launch { lane: u16, kernel: String }, Copy { lane: u16 }, Memset { lane: u16 }, Mark { lane: u16, value: u64 }, Retire { lane: u16, seq: u64 } }
pub struct FakeGpu { /* Mutex<state>, Poison, kernels: Mutex<HashMap<String, FakeKernelFn>> */ }
impl FakeGpu {
    pub fn new() -> Arc<Self>;
    pub fn define(&self, name: &str, f: impl Fn(&mut FakeMemory, &Launch, &Args) -> Result<(), FakeFault> + Send + Sync + 'static);
    pub fn alloc(&self, bytes: usize) -> u64;            // zeroed; bases 0x1000_0000 upward, 256-aligned, never 0
    pub fn write(&self, addr: u64, bytes: &[u8]);        // host-side, for tests (panics outside an allocation)
    pub fn read(&self, addr: u64, len: usize) -> Vec<u8>;
    pub fn step(&self) -> bool;                          // runs the oldest runnable op of any stream
    pub fn run_until_idle(&self) -> usize;
    pub fn log(&self) -> Vec<FakeOp>;                    // ops in the order they ran
}
impl Gpu for FakeGpu { type Stream = FakeStream; type Module = FakeModule; type Kernel = FakeKernel; type Graph = FakeGraph; type Sequence<'a> = FakeSequence<'a>; ... }
pub struct FakeStream; impl Stream for FakeStream; impl DeviceQueue for FakeStream
pub struct FakeEnqueue<'a>; impl Enqueue for FakeEnqueue<'_>
pub struct FakeModule; impl Module for FakeModule   // from_source/from_image: module id = hash of the text/bytes; kernels resolve by name in FakeGpu's registry
#[derive(Clone, Debug)] pub struct FakeKernel { name: Arc<str>, id: u64, f: FakeKernelFn } impl Kernel
```
Execution model:
- Every op gets a global sequence number at enqueue.
- `step()` picks, among each stream's head op, the runnable one with the lowest number. An await is runnable once its marker's lane has run that `Mark`.
- `step()` runs it with the state lock released while a retire listener runs.
- Copies to and from host pointers copy at run time (the `unsafe` contract keeps the host memory alive).
- `completed()` is the number of `Mark` ops run on that lane.

- [ ] **Step 1: Write failing tests** in `tests/fake_streams.rs`. Kernels are defined by each test, e.g. `"add1"`: `u32 += 1` at `args.read::<u64>(0)`.
  - `ops_on_one_stream_run_in_submission_order`: memset 0, launch add1, copy to host; `log()` order matches and the host value is 1.
  - `an_await_orders_work_behind_another_streams_marker`: lane 0: add1 then `mark`; lane 1: `await_marker(m)`, then copy out. The copy's result is 1, and `log()` shows lane 0's `Mark` before lane 1's `Copy`.
  - `markers_complete_as_the_device_reaches_them`: `!pool.reached(m)` before `run_until_idle()`, `pool.reached(m)` after.
  - `retirement_runs_each_listener_once_in_order`: three `notify_retired` on one stream record seqs `[1, 2, 3]`, each once, all `ok()`.
  - `kept_values_live_until_confirmed_or_the_stream_drops` (Review Focus 3): an `Arc` kept through `keep` survives until `confirm_retired(ticket)`. A second kept `Arc`, with no confirm, is released when the stream drops.
  - `concurrent_submissions_to_one_stream_stay_contiguous` (Review Focus 1): two threads each submit 100 submissions of 3 launches (kernel names `a`/`b`). In `log()` every run of 3 consecutive launches has a single name. File-level `#![allow(clippy::disallowed_methods, reason = "peer threads stand in for lattice workers")]`.
  - `a_call_inside_its_own_submit_is_refused`: `s.submit(|_| s.mark().map(|_| ()))` is `Err(Invalid(_))`.
  - `unknown_kernels_are_not_found`: `module.kernel("nope")` is `Err(NotFound(_))`.
- [ ] **Step 2: Run** `cargo test -p euhedral-gpu --test fake_streams`. Expected: FAIL (the module is missing).
- [ ] **Step 3: Implement** `src/fake/{mod,stream,module}.rs`. The `Gpu::record`/`rerecord`/`fingerprint`/`update` bodies may return `Err(Invalid("not yet"))` until Task 7. The `Graph` impl is a stub type until then.
- [ ] **Step 4: Run** `cargo test -p euhedral-gpu --test fake_streams`. Expected: 8 passed.
- [ ] **Step 5: Commit.** `git commit -m "euhedral-gpu: fake backend streams, markers and retirement"`

### Task 6: Fake device faults and poisoning

**Files:**
- Modify: `euhedral-gpu/src/fake/mod.rs`, `src/fake/stream.rs`, `tests/fake_streams.rs`

**Interfaces:**
- Produces: a fault in a fake kernel does three things.
  - It sets the device's `Poison` to `GpuError::Fault { call: "fake kernel", code: 700 | 710, name: "ILLEGAL_ADDRESS" | "ASSERT" }`.
  - Later queued ops do not run, but their retire listeners do, with `status = code`.
  - New `submit`/`mark`/`await_marker`/`notify_retired` calls return `Err(Poisoned(cause))`, and `completed()` returns the lane's marked count.

- [ ] **Step 1: Write failing tests:**
  - `a_fault_surfaces_through_confirm_retired_and_poisons_the_device`: a kernel returning `Err(IllegalAddress)`, then `notify_retired`. The ticket has `!ok()`. `confirm_retired(t) == Some(Fault { code: 700, .. })`, `gpu.poisoned().is_some()`, and `s.submit(..) == Err(Poisoned(_))`.
  - `work_queued_behind_a_fault_never_runs_but_its_markers_are_reached`: after the fault, a queued launch is absent from `log()`, and `pool.reached(m)` holds for a marker made before the fault.
  - `a_poisoned_device_still_emits_retire_into_continuations` (Review Focus 3): `retire_into` armed after the faulting op on another lane, whose listener cannot run because that lane waits on the faulted lane's marker. A `Completions::pull` loop (as in `euhedral-graph`'s README) yields the frame. Its ticket confirms to `Some(Poisoned(_))`.
- [ ] **Step 2: Run** `cargo test -p euhedral-gpu --test fake_streams`. Expected: the new tests FAIL.
- [ ] **Step 3: Implement** fault handling in `step()`.
- [ ] **Step 4: Run** the same command. Expected: 11 passed.
- [ ] **Step 5: Commit.** `git commit -m "euhedral-gpu: fake device faults poison the device"`

### Task 7: Fake graph record, fingerprint, update, rerecord, replay

**Files:**
- Create: `euhedral-gpu/src/fake/graph.rs`, `euhedral-gpu/tests/fake_graph.rs`
- Modify: `src/fake/mod.rs`

**Interfaces:**
- Produces: `FakeSequence<'a>` (a pass: `Record { streams } | Probe | Update { graph }`, plus a `ShapeHasher` and a launch counter) and `FakeGraph { ops: Vec<RecordedOp>, args: Vec<Args>, fingerprint, kept: Vec<Box<dyn Any + Send>> }`.
  - `record` locks the streams in ascending lane order through their `StreamLock`s and runs the ops for real. On the real streams it adds the fork and join that make the run equal to a replay on `streams[0]`.
  - `replay(stream)` enqueues the recorded ops, linearized in record order, onto `stream` with the current `args`.
  - `update` stages new args and applies them only if the shape equals the recording's.
  - `rerecord` replaces `ops` and `fingerprint` and returns `Rerecorded::Updated` when the old shape and the new one have equal launch counts and kernels in the same order. Otherwise it returns `Rerecorded::Instantiated`. This mirrors CUDA's update-or-instantiate outcome.

- [ ] **Step 1: Write failing tests** in `tests/fake_graph.rs`. The sequence under test, `seq(a)`: stream 0 runs `"scale"(dst, src, a)`, `order(0, 1)`, stream 1 runs `"add1"(dst)`, `order(1, 0)`, and stream 0 copies out.
  - `record_runs_the_sequence_once_for_real`: after `record` and `run_until_idle`, the output equals the CPU reference for `a = 2`.
  - `replay_with_updated_arguments_matches_direct_submission`: `update(seq(3))` then `replay(s0)` gives the same bytes as submitting `seq(3)` directly on fresh buffers.
  - `the_fingerprint_matches_a_probe_of_the_same_sequence`: `gpu.fingerprint(2, |q| seq_into(q, 5))? == graph.fingerprint()`. A sequence with one more launch differs.
  - `an_update_of_another_shape_applies_nothing`: `update` with an extra launch is `Err(ShapeChanged { .. })`, and a following replay still uses the old arguments.
  - `rerecord_reports_update_or_instantiate`: the same kernels with another copy address give `Updated`; another kernel order gives `Instantiated`. Both replay correctly.
  - `a_sequence_names_only_its_own_streams`: with two streams, `submit(2, ..)` and `order(0, 2)` are `Err(Invalid(_))`; `order(1, 1)` is a no-op and `Ok`.
- [ ] **Step 2: Run** `cargo test -p euhedral-gpu --test fake_graph`. Expected: FAIL.
- [ ] **Step 3: Implement** `src/fake/graph.rs`.
- [ ] **Step 4: Run** `cargo test -p euhedral-gpu`. Expected: every test passes (unit, `layering`, `fake_streams`, `fake_graph`).
- [ ] **Step 5: Commit.** `git commit -m "euhedral-gpu: fake graph record, fingerprint, update and replay"`

**Phase 1 gate:** `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings && cargo test -p euhedral-gpu`.

---

## Phase 2: the shared driver, CUDA streams and retirement

Phase gate: `cargo test -p euhedral-gpu --features cuda` passes with no GPU (CUDA tests skip), and [GPU] passes on the device.

### Task 8: `euhedral-memory`: `Driver::shared` and `Driver::symbol`

**Files:**
- Modify: `euhedral-memory/src/cuda/mod.rs`, `euhedral-memory/tests/cuda.rs`

**Interfaces:**
- Produces:
```rust
impl Driver {
    /// The process's driver: loaded and initialized on first use; every call returns the same `Arc` (or the same error).
    pub fn shared() -> Result<Arc<Self>, String>;   // static OnceLock<Result<Arc<Driver>, String>>
    /// An entry point beyond `Fns`, looked up in the loaded library; null when the driver lacks it.
    pub fn symbol(&self, name: &CStr) -> *mut c_void;
}
```
`Driver` stores the `dlopen` handle (as `usize`, so `Driver` stays `Send + Sync` with no new `unsafe impl`). `load()` keeps working as before.

- [ ] **Step 1: Write failing tests** in `euhedral-memory/tests/cuda.rs`, skipping with no driver like the others:
  - `the_shared_driver_is_loaded_once`: `Arc::ptr_eq(&Driver::shared()?, &Driver::shared()?)`.
  - `symbols_beyond_the_table_are_found`: `!d.symbol(c"cuLaunchKernel").is_null()` and `d.symbol(c"cuNoSuchCall").is_null()`.
- [ ] **Step 2: Run** [GPU] `cargo test -p euhedral-memory --features cuda --test cuda`. Expected: FAIL (no such functions).
- [ ] **Step 3: Implement** both functions.
- [ ] **Step 4: Run** [GPU] the same command. Expected: PASS. Run `cargo clippy -p euhedral-memory --all-targets --features cuda -- -D warnings`. Expected: clean.
- [ ] **Step 5: Commit.** `git commit -m "euhedral-memory: one shared driver per process, and look-ups beyond its table"`

### Task 9: `GpuFns`, `CudaGpu` and the CI step

**Files:**
- Create: `euhedral-gpu/src/cuda/mod.rs`, `src/cuda/fns.rs`, `euhedral-gpu/tests/cuda.rs`
- Modify: `src/lib.rs` (`#[cfg(feature = "cuda")] pub mod cuda;`), `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: `euhedral_memory::cuda::{Driver, Device, CuStream, CuEvent, CudaError}`.
- Produces:
```rust
// fns.rs: loaded through Driver::symbol; a missing required symbol is GpuError::Load("libcuda: no <name>").
pub struct GpuFns {
    pub stream_wait_event: unsafe extern "C" fn(CuStream, CuEvent, c_uint) -> CuResult,          // cuStreamWaitEvent
    pub stream_query: unsafe extern "C" fn(CuStream) -> CuResult,                                   // cuStreamQuery
    pub stream_add_callback: Option<unsafe extern "C" fn(CuStream, StreamCallback, *mut c_void, c_uint) -> CuResult>, // cuStreamAddCallback
    pub launch_host_func: unsafe extern "C" fn(CuStream, HostFn, *mut c_void) -> CuResult,          // cuLaunchHostFunc (fallback)
    pub module_load_data_ex: unsafe extern "C" fn(*mut CuModule, *const c_void, c_uint, *mut c_int, *mut *mut c_void) -> CuResult,
    pub module_get_function: unsafe extern "C" fn(*mut CuFunction, CuModule, *const c_char) -> CuResult,
    pub module_unload: unsafe extern "C" fn(CuModule) -> CuResult,
    pub func_get_param_count: Option<unsafe extern "C" fn(CuFunction, *mut usize) -> CuResult>,
    pub func_get_param_info: Option<unsafe extern "C" fn(CuFunction, usize, *mut usize, *mut usize) -> CuResult>,
    pub launch_kernel: unsafe extern "C" fn(CuFunction, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, CuStream, *mut *mut c_void, *mut *mut c_void) -> CuResult,
    pub device_get_attribute: unsafe extern "C" fn(*mut c_int, c_int, c_int) -> CuResult,
    pub driver_get_version: unsafe extern "C" fn(*mut c_int) -> CuResult,
    pub stream_begin_capture: unsafe extern "C" fn(CuStream, c_int) -> CuResult,                    // cuStreamBeginCapture_v2
    pub stream_end_capture: unsafe extern "C" fn(CuStream, *mut CuGraph) -> CuResult,
    pub capture_info: CaptureInfo,     // enum { V3(fn 7 args), V2(fn 6 args) }: cuStreamGetCaptureInfo_v3, else _v2
    pub graph_instantiate: unsafe extern "C" fn(*mut CuGraphExec, CuGraph, u64) -> CuResult,      // cuGraphInstantiateWithFlags
    pub graph_launch: unsafe extern "C" fn(CuGraphExec, CuStream) -> CuResult,
    pub graph_exec_update: unsafe extern "C" fn(CuGraphExec, CuGraph, *mut ExecUpdateResultInfo) -> CuResult, // _v2
    pub graph_exec_kernel_node_set_params: unsafe extern "C" fn(CuGraphExec, CuGraphNode, *const KernelNodeParams) -> CuResult, // _v2
    pub graph_destroy: unsafe extern "C" fn(CuGraph) -> CuResult,
    pub graph_exec_destroy: unsafe extern "C" fn(CuGraphExec) -> CuResult,
}
#[repr(C)] pub struct KernelNodeParams { func, grid_x, grid_y, grid_z, block_x, block_y, block_z, shared_bytes: c_uint,
    kernel_params: *mut *mut c_void, extra: *mut *mut c_void, kern: *mut c_void, ctx: CuContext } // CUDA_KERNEL_NODE_PARAMS_v2
#[repr(C)] pub struct ExecUpdateResultInfo { result: c_int, error_node: CuGraphNode, error_from_node: CuGraphNode }
pub const CAPTURE_MODE_RELAXED: c_int = 2; pub const ATTR_CC_MAJOR: c_int = 75; pub const ATTR_CC_MINOR: c_int = 76;

// mod.rs
pub struct CudaGpu { device: Arc<Device>, fns: GpuFns, poison: Poison }
impl CudaGpu {
    pub fn open(ordinal: i32) -> Result<Arc<Self>, GpuError>;            // Driver::shared + Device::open
    pub fn from_device(device: Arc<Device>) -> Result<Arc<Self>, GpuError>; // share memory's exact Device
    pub fn device(&self) -> &Arc<Device>;
    pub fn arch(&self) -> (i32, i32);                                      // compute capability
    pub fn driver_version(&self) -> i32;                                    // e.g. 13040
    pub(crate) fn check(&self, call: &'static str, code: i32) -> Result<(), GpuError>; // poisons on a fault code
}
pub(crate) fn is_fault(code: i32) -> bool; // 700, 702, 710, 714, 715, 716, 717, 718, 719
```

- [ ] **Step 1: Write failing tests.**
  - Unit test `fault_codes_are_classified` (no GPU): `is_fault(700) && is_fault(719) && !is_fault(1) && !is_fault(600)`. Also `check` on 700 sets the poison, and on 1 does not; test it against a `Poison` directly by factoring the classification into `fn classify(call, code, name) -> GpuError`.
  - `tests/cuda.rs` (`#![cfg(feature = "cuda")]`, a `gpu()` helper returning `None` with no driver, which makes each test `return` after `eprintln!("no CUDA device: skipped")`):
    - `the_gpu_shares_memorys_driver_and_reports_its_architecture`: `Arc::ptr_eq(&gpu.device().driver, &Driver::shared()?)` and `gpu.arch().0 >= 7`.
- [ ] **Step 2: Run** `cargo test -p euhedral-gpu --features cuda --lib` and [GPU] `cargo test -p euhedral-gpu --features cuda --test cuda`. Expected: FAIL.
- [ ] **Step 3: Implement** `fns.rs` and `mod.rs`.
  - Every required symbol must be present (else `Load`). `stream_add_callback`, `func_get_param_count` and `func_get_param_info` are optional.
  - Add a CI step to `.github/workflows/ci.yml`, after `Test`, with `if: runner.os == 'Linux'`, named `CUDA backend (built and linted; its tests skip with no device)`. It runs `cargo clippy -p euhedral-gpu --all-targets --features cuda -- -D warnings` and then `cargo test -p euhedral-gpu --features cuda`.
- [ ] **Step 4: Run** the same commands. Expected: PASS.
- [ ] **Step 5: Commit.** `git commit -m "euhedral-gpu: CUDA entry points on memory's shared driver; CI builds the cuda feature"`

### Task 10: CUDA streams: lanes, the marker ring, awaits, copies

**Files:**
- Create: `euhedral-gpu/src/cuda/stream.rs`
- Modify: `src/cuda/mod.rs`, `tests/cuda.rs`

**Interfaces:**
- Produces:
```rust
pub struct CudaStream { gpu: Arc<CudaGpu>, lane: LaneId, name: String, raw: CuStream,
    ring: Arc<LaneRing>, lanes: Arc<[Arc<LaneRing>]>, state: StreamLock<StreamState>, kept: Kept }
struct LaneRing { device: Arc<Device>, events: Vec<CuEvent>, marked: AtomicU64, done: AtomicU64 } // Drop destroys events
struct StreamState { shadow: Option<CuStream>, next_retire: u64 }
impl Stream for CudaStream { type Gpu = CudaGpu; type Enqueue<'a> = CudaEnqueue<'a>; ... }
impl DeviceQueue for CudaStream  // completed(): scan as Decision 6; name()
pub struct CudaEnqueue<'a> { /* targets: real stream, plus the shadow while recording; a pass (Task 18) */ }
```
- `open_lanes` creates each stream with `STREAM_NON_BLOCKING` and its ring of `markers` events (`CU_EVENT_DISABLE_TIMING` = 2). Each stream holds the shared `lanes` array, so it can resolve another lane's marker.
- `mark()`:
  1. Lock the stream.
  2. `v = marked + 1`.
  3. If `v > N`, the slot's previous value `v - N` must be `<= done` after a rescan; else return `Err(Busy)`.
  4. `cuEventRecord(events[v % N], raw)`.
  5. Store `marked = v`.
- `await_marker(m)`:
  - `m.lane == self.lane`: no-op.
  - Unknown lane, or `m.value > marked`: `Invalid`.
  - `m.value <= done` after a rescan: no-op.
  - Otherwise `cuStreamWaitEvent(raw, other.events[m.value % N], 0)`.
- Every call checks `gpu.poison` first, and enters the context (`device.enter()`) around driver calls.
- Copies map to `memcpy_htod_async`, `memcpy_dtoh_async`, `memcpy_dtod_async` and `memset_d8_async` from `Driver::fns`.
- `Drop` (a shutdown call: it may block) runs `cuStreamSynchronize` on the real and shadow streams, then `kept.clear()`, then destroys the streams.

- [ ] **Step 1: Write failing tests** in `tests/cuda.rs`. Memory comes from `euhedral_memory::cuda::{DeviceMemory, PinnedHost}` through the `Backend` trait.
  - `a_stream_is_a_lane_whose_markers_complete`: memset 4 MiB, then `mark`; poll `pool.reached(m)` with a 5 s deadline (the test thread is not the lattice).
  - `copies_round_trip_through_the_device`: pinned 1 MiB pattern → device → second pinned buffer; the bytes are equal after the marker is reached.
  - `a_marker_from_an_unknown_lane_is_invalid`: `await_marker(Marker { lane: LaneId(9), value: 1 })` is `Err(Invalid(_))`.
- [ ] **Step 2: Run** [GPU] `cargo test -p euhedral-gpu --features cuda --test cuda`. Expected: FAIL.
- [ ] **Step 3: Implement** `stream.rs`.
- [ ] **Step 4: Run** [GPU] the same command. Expected: PASS. Run `cargo test -p euhedral-gpu --features cuda`. Expected: PASS with no GPU (skips).
- [ ] **Step 5: Commit.** `git commit -m "euhedral-gpu: CUDA streams as device lanes with an event ring"`

### Task 11: CUDA retirement through the driver callback

**Files:**
- Create: `euhedral-gpu/src/cuda/retire.rs`
- Modify: `src/cuda/stream.rs`, `tests/cuda.rs`

**Interfaces:**
- Produces:
  - `notify_retired`: lock the stream; `seq = next_retire` (then increment); box `CallbackRecord { listener, lane, seq }` into a raw pointer; then `cuStreamAddCallback(raw, trampoline, ptr, 0)`. Use `cuLaunchHostFunc` with `host_trampoline` (status 0) if the former is absent. If the enqueue fails, reclaim the box and return the error.
  - `extern "C" fn trampoline(_: CuStream, status: CuResult, user: *mut c_void)`: `Box::from_raw`, then `catch_unwind(AssertUnwindSafe(|| listener.retired(ticket)))`. It makes no driver call.
  - `confirm_retired(t)`:
    - `t.status == POISONED_STATUS` → `Some(Poisoned(cause))`.
    - Otherwise `t.status != 0` → classify with the driver's error name (an ordinary thread may call the driver). A fault poisons the device; return `Some(err)`.
    - Otherwise, if the device is poisoned, return `Some(Poisoned(cause))`.
    - Otherwise `None`.
    - In every case, run `kept.release_through(t.seq)`.
  - `keep(v)` pushes `(next_retire, v)`: it belongs to the next armed retirement.

- [ ] **Step 1: Write failing tests:**
  - `retirement_listeners_run_once_in_order`: on one stream, 5 × (memset; `notify_retired(rec)`), where the listener pushes `ticket.seq` into an `Arc<Mutex<Vec<u64>>>` (enqueue only). After a 5 s deadline poll, the list is `[0, 1, 2, 3, 4]` and every `confirm_retired` is `None`.
  - `retire_into_emits_a_continuation_frame`: a `Completions` over the stream's pool, `retire_into(&s, &completions, |t| frame_that_confirms(t))`, and a pull loop until `pending() == 0`. The frame ran once and saw `None`.
  - `kept_host_buffers_outlive_the_copy_until_confirmed`: a pinned buffer kept in the submission; its `Arc` strong count drops to 1 only after `confirm_retired`.
- [ ] **Step 2: Run** [GPU] `cargo test -p euhedral-gpu --features cuda --test cuda`. Expected: FAIL.
- [ ] **Step 3: Implement** `retire.rs` and wire it into the stream.
- [ ] **Step 4: Run** [GPU] the same command. Expected: PASS.
- [ ] **Step 5: Commit.** `git commit -m "euhedral-gpu: CUDA retirement through stream callbacks"`

**Phase 2 gate:** the global gate; `cargo test -p euhedral-gpu --features cuda` (no GPU: skips); [GPU] `cargo test -p euhedral-gpu --features cuda --test cuda`.

---

## Phase 3: NVRTC and the module cache

### Task 12: The NVRTC loader and compiler

**Files:**
- Create: `euhedral-gpu/src/cuda/nvrtc.rs`
- Modify: `src/cuda/mod.rs`

**Interfaces:**
- Produces:
```rust
pub struct Nvrtc { /* fns: create_program, compile_program, get_program_log_size, get_program_log, get_cubin_size, get_cubin,
                      get_ptx_size, get_ptx, destroy_program, version, get_num_supported_archs, get_supported_archs, get_error_string */ }
pub enum Target { Cubin(u32 /* sm */), Ptx(u32 /* compute */) }
impl Nvrtc {
    /// The process's NVRTC: the first successful load, by `explicit` or the search order.
    pub fn shared(explicit: Option<&Path>) -> Result<Arc<Self>, GpuError>;
    pub fn version(&self) -> (i32, i32);
    pub fn target_for(&self, arch: (i32, i32)) -> Target; // sm_XY if supported, else newest compute_XY
    pub fn compile(&self, source: &str, options: &CompileOptions, target: Target) -> Result<Vec<u8>, GpuError>;
        // PTX output gets a trailing NUL; options get "--gpu-architecture=sm_XY|compute_XY" and "--include-path=<p>" appended
}
/// Library paths to try, in order; pure, for tests.
pub(crate) fn candidates(explicit: Option<&Path>, env: &dyn Fn(&str) -> Option<OsString>,
                         list: &dyn Fn(&Path) -> Vec<PathBuf>) -> Vec<Candidate>;
pub(crate) enum Candidate { Soname(&'static str), File(PathBuf) }
```
Search order:
1. `explicit` (only it).
2. Env `EUHEDRAL_NVRTC` (a file).
3. Sonames through the loader: `libnvrtc.so.13`, `libnvrtc.so.12`, `libnvrtc.so`.
4. `$CUDA_HOME/lib64`, `$CUDA_PATH/lib64`, `/usr/local/cuda/lib64`, then `/usr/local/cuda-*/lib64`, names sorted descending. Each directory contributes its `libnvrtc.so.<N>` files, newest first, then `libnvrtc.so`.

For a `File`, first `dlopen` with `RTLD_NOW | RTLD_GLOBAL` every `libnvrtc-builtins.so.*` in the same directory, so NVRTC's own by-soname load of its builtins finds them. A compile error returns `Compile { log }`. The log names the program (`options.name`).

- [ ] **Step 1: Write failing tests.**
  - Unit test `the_search_order_is_explicit_env_sonames_then_toolkits` (no GPU, runs in CI with `--features cuda`): a fake `env` and `list` over the directories `/x/cuda/lib64` (files `libnvrtc.so.13`, `libnvrtc.so.12`, `libnvrtc-builtins.so.13.1`) and `/usr/local/cuda-12.4/lib64`. Assert the exact candidate vector. With `explicit = Some(p)`, assert it is exactly `[File(p)]`.
  - `tests/cuda.rs`: `nvrtc_compiles_for_this_device`. Skip with no NVRTC (`Nvrtc::shared(None)` is an `Err`); `target_for(gpu.arch())` is `Cubin(120)` on sm_120 with NVRTC 13.x; compiling `tests/kernels/ops.cu` gives bytes starting with `\x7fELF`.
  - `a_compile_error_returns_the_log`: source `"extern \"C\" __global__ void k() { nope; }"` gives `Err(Compile { log })` with `log.contains("nope")`.
- [ ] **Step 2: Create `euhedral-gpu/tests/kernels/ops.cu`:**
```c
// The integration tests' kernels; ops.ptx is generated from this file (see its header).
extern "C" __global__ void axpb(float* y, const float* x, float a, float b, unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) y[i] = a * x[i] + b; }
extern "C" __global__ void add_u32(unsigned* z, const unsigned* x, unsigned c, unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) z[i] = x[i] + c; }
extern "C" __global__ void spin(unsigned long long cycles) {
    long long t = clock64(); while ((unsigned long long)(clock64() - t) < cycles) {} }
extern "C" __global__ void poke(int* p) { *p = 1; }
```
- [ ] **Step 3: Run** `cargo test -p euhedral-gpu --features cuda --lib nvrtc` and [GPU] `CUDA_HOME=/home/brandon/qwen38-work/cuda-13.1 cargo test -p euhedral-gpu --features cuda --test cuda nvrtc`. Expected: FAIL.
- [ ] **Step 4: Implement** `nvrtc.rs`.
- [ ] **Step 5: Run** the same commands. Expected: PASS.
- [ ] **Step 6: Commit.** `git commit -m "euhedral-gpu: NVRTC loaded at run time, with a search order"`

### Task 13: The module cache

**Files:**
- Create: `euhedral-gpu/src/module_cache.rs`
- Modify: `src/lib.rs`

**Interfaces:**
- Produces (backend-neutral, compiled without features):
```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)] pub struct CacheKey(pub u64, pub u64);
pub struct KeyMaterial<'a> { pub source: &'a str, pub options: &'a CompileOptions, pub target: &'a str, pub compiler_version: (i32, i32) }
pub fn cache_key(m: &KeyMaterial<'_>) -> CacheKey; // "euhedral-gpu-module-v1\0" + source + each option + headers + include paths (as strings) + target + version
pub struct ModuleCache { dir: PathBuf }
impl ModuleCache {
    pub fn new(dir: impl Into<PathBuf>) -> Self;
    pub fn path(&self, key: CacheKey, ext: &str) -> PathBuf;                      // "{:016x}{:016x}.{ext}"
    pub fn load(&self, key: CacheKey, ext: &str) -> Option<Vec<u8>>;
    pub fn store(&self, key: CacheKey, ext: &str, bytes: &[u8]) -> std::io::Result<()>; // create_dir_all; write "<name>.tmp-<pid>-<n>"; rename
}
```
`std::fs` sits under `#[allow(clippy::disallowed_methods, reason = "module loading is a build-time call; it blocks by nature (compile, driver load)")]`.

- [ ] **Step 1: Write failing unit tests** (dirs under `env!("CARGO_TARGET_TMPDIR")` via a test helper; never `/tmp`):
  - `the_key_changes_with_each_input`: change each of source, one option, option order, a header's contents, an include path, the target or the version, and the key differs; identical inputs give equal keys.
  - `stores_are_atomic_renames` (Review Focus 4): after `store`, the directory holds exactly the final file, with no `.tmp-` left. `load` returns the bytes, and a second `store` of other bytes replaces them.
  - `a_missing_entry_loads_as_none`.
- [ ] **Step 2: Run** `cargo test -p euhedral-gpu --lib module_cache`. Expected: FAIL.
- [ ] **Step 3: Implement** `module_cache.rs`.
- [ ] **Step 4: Run** the same command. Expected: PASS.
- [ ] **Step 5: Commit.** `git commit -m "euhedral-gpu: a module cache keyed by content hash"`

### Task 14: `CudaModule` and `CudaKernel`

**Files:**
- Create: `euhedral-gpu/src/cuda/module.rs`, `euhedral-gpu/tests/kernels/ops.ptx`
- Modify: `src/cuda/mod.rs`, `tests/cuda.rs`

**Interfaces:**
- Produces:
```rust
pub struct CudaModule { inner: Arc<ModuleInner> }   // ModuleInner { gpu: Arc<CudaGpu>, raw: CuModule, key: u64 } ; Drop: cuModuleUnload
#[derive(Clone, Debug)] pub struct CudaKernel { func: CuFunction, module: Arc<ModuleInner>, name: Arc<str>, id: u64, params: Option<Arc<[u32]>> }
impl Module for CudaModule; impl Kernel for CudaKernel
```
- `from_source`:
  - `nvrtc = Nvrtc::shared(options.compiler.as_deref())?`; `target = nvrtc.target_for(gpu.arch())`; `ext` is `"cubin"` or `"ptx"`.
  - With `cache_dir`, try `cache.load`. If the image fails `cuModuleLoadDataEx`, recompile and `store` (overwrite).
  - Otherwise compile, `store`, and load.
  - `key` = `CacheKey.0`.
- `from_image`: an image starting with `\x7fELF` (cubin) or the fatbin magic `0xBA55ED50` (little-endian) is passed as is. Anything else is PTX text, NUL-terminated if needed. JIT options `CU_JIT_ERROR_LOG_BUFFER` (5) and `CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES` (6) take an 8 KiB buffer. A failure is `Load("<name>: <log>")`. `key` = `hash_bytes(image)`.
- `kernel(name)`:
  - `cuModuleGetFunction`; `CUDA_ERROR_NOT_FOUND` (500) is `NotFound("<name> (declare kernels extern \"C\" to look them up by their source name)")`.
  - `params`: through `func_get_param_count`/`func_get_param_info` when both exist.
  - `id = hash(module key, name)`.

- [ ] **Step 1: Generate the PTX fixture** with the toolkit: `$CUDA_HOME/bin/nvcc --ptx -arch=compute_75 -o euhedral-gpu/tests/kernels/ops.ptx euhedral-gpu/tests/kernels/ops.cu`. Then prepend `// Generated from ops.cu: nvcc --ptx -arch=compute_75 (CUDA 13.1).` and commit it.
- [ ] **Step 2: Write failing tests** in `tests/cuda.rs`:
  - `modules_load_from_source_and_from_an_image`: `from_source(ops.cu)` (skip the source half with no NVRTC) and `from_image(include_bytes!("kernels/ops.ptx"))`. Skip the image half when `gpu.driver_version() < 13010`. Both give an `axpb` kernel. On this driver `axpb.params` is `Some([8, 8, 4, 4, 4])`.
  - `missing_kernels_are_not_found`: `kernel("axpbx")` is `Err(NotFound(_))`.
  - `a_corrupt_cache_entry_is_rebuilt` (Review Focus 4): compile with `cache_dir`, overwrite the cached file with 16 garbage bytes, and `from_source` again succeeds and rewrites the file (the size differs from 16).
  - `a_cache_hit_skips_the_compiler`: the second `from_source` with the same inputs leaves the cache file's modified time unchanged.
- [ ] **Step 3: Run** [GPU] `CUDA_HOME=/home/brandon/qwen38-work/cuda-13.1 cargo test -p euhedral-gpu --features cuda --test cuda module`. Expected: FAIL.
- [ ] **Step 4: Implement** `module.rs`.
- [ ] **Step 5: Run** the same command. Expected: PASS.
- [ ] **Step 6: Commit.** `git commit -m "euhedral-gpu: CUDA modules from source (cached) and from images"`

**Phase 3 gate:** the global gate; [GPU] `CUDA_HOME=… cargo test -p euhedral-gpu --features cuda --test cuda`.

---

## Phase 4: launching kernels

### Task 15: `CudaEnqueue::launch` with the parameter check

**Files:**
- Modify: `euhedral-gpu/src/cuda/stream.rs`, `tests/cuda.rs`

**Interfaces:**
- Produces: `launch(kernel, launch, args)`.
  - If `kernel.params` is `Some(sizes)` and `args.sizes()` differs in count or any size: `Err(Args("<name>: expected sizes [..], got [..]"))`, before any driver call.
  - Otherwise `cuLaunchKernel(func, grid, block, shared_bytes, raw, args.pointers().as_mut_ptr(), null)`.
  - Launch errors go through `gpu.check`.

- [ ] **Step 1: Write failing tests:**
  - `axpb_from_source_and_from_ptx_match_the_cpu_bit_for_bit`: n = 1 << 20, x[i] = i as f32 * 0.25, a = 1.5, b = -2.0. Run the source module and the PTX module on two output buffers. Both outputs equal `a.mul_add(x, b)` per element: CUDA contracts `a*x+b` into an FMA, so the reference uses `f32::mul_add`. They are also bit-equal to each other.
  - `arguments_that_do_not_fit_are_refused_before_launch`: `axpb` with 4 arguments gives `Err(Args(_))`, and `gpu.poisoned()` is `None`.
- [ ] **Step 2: Run** [GPU] `CUDA_HOME=… cargo test -p euhedral-gpu --features cuda --test cuda launch`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the same command. Expected: PASS.
- [ ] **Step 5: Commit.** `git commit -m "euhedral-gpu: kernel launches with typed, checked arguments"`

---

## Phase 5: cross-stream markers, poisoning, the fault test

### Task 16: Cross-stream markers and retirement order on the device

**Files:**
- Modify: `euhedral-gpu/tests/cuda.rs` (plus any fixes in `src/cuda/stream.rs` they expose)

- [ ] **Step 1: Write the tests:**
  - `a_marker_orders_another_streams_work_on_the_device`: s0 runs `spin(≈50 ms of cycles: 2.5e9 * 0.05)`, then `add_u32(z, x, 7)`, then `mark → m`. s1 runs `await_marker(m)` and copies z to pinned. After s1's marker, the host has `x + 7` everywhere (z starts zeroed, so a copy that ran before the kernel would show zeros).
  - `retirements_on_two_streams_follow_their_markers`: s1 awaits s0's marker, then both arm a retirement. s0's ticket is recorded before s1's.
  - `a_full_marker_ring_is_busy_and_old_markers_need_no_wait` (Review Focus 2): `StreamOptions { markers: 4 }`. After a long `spin`, 4 marks succeed and the 5th is `Err(Busy)`. Once reached, a 5th mark succeeds. `await_marker` of the first (completed) marker on another stream adds no device wait: the other stream's next marker is reached while s0 is still spinning in a later spin.
- [ ] **Step 2: Run** [GPU] `CUDA_HOME=… cargo test -p euhedral-gpu --features cuda --test cuda marker`. Expected: PASS. If one fails, fix `stream.rs` with superpowers:systematic-debugging; do not loosen the test.
- [ ] **Step 3: Commit.** `git commit -m "euhedral-gpu: cross-stream markers and retirement order on the device"`

### Task 17: A deliberate device fault (its own process)

**Files:**
- Create: `euhedral-gpu/tests/cuda_fault.rs`

- [ ] **Step 1: Write the test** `a_device_fault_surfaces_through_confirm_retired_and_poisons_the_device`. It is the only test in the file: a sticky fault poisons the context for the whole process.
  1. Two lanes. s0 runs `poke(0x10 as *mut i32)` and then `notify_retired(rec)`.
  2. Poll the record for 5 s. Exactly one ticket, with `!ok()` and `status == 700`.
  3. `s0.confirm_retired(t) == Some(Fault { code: 700, .. })`, and `gpu.poisoned()` is a `Fault`.
  4. `s1.submit(..)`, `s1.mark()` and `s0.notify_retired(..)` are `Err(Poisoned(_))`.
  5. `s0.completed()` equals its marked count (nothing stranded).
  6. A `retire_into` armed on s1 behind `await_marker` of an s0 marker made after the poke yields its frame in a pull loop, and that frame's `confirm_retired` is `Some(_)`.
- [ ] **Step 2: Run** [GPU] `cargo test -p euhedral-gpu --features cuda --test cuda_fault`. Expected: PASS. If the callback is never called on this driver after a sticky fault (see Risks), make `retire_into`'s poisoned path cover it, and assert that path instead of the listener's ticket.
- [ ] **Step 3: Commit.** `git commit -m "euhedral-gpu: a device fault surfaces as a value and poisons the device"`

---

## Phase 6: graph capture, replay, update, fingerprint

### Task 18: Record on shadow streams; instantiate; replay

**Files:**
- Create: `euhedral-gpu/src/cuda/graph.rs`, `euhedral-gpu/tests/cuda_graph.rs`
- Modify: `src/cuda/stream.rs` (`CudaEnqueue` gets a pass and the shadow target), `src/cuda/mod.rs` (the `Gpu` graph methods)

**Interfaces:**
- Produces:
```rust
pub struct CudaSequence<'a> { pass: Pass<'a>, shape: ShapeHasher, launches: u32, /* record: streams' guards, shadows, rec events, nodes, failed: Option<GpuError>, kept */ }
pub struct CudaGraph { gpu: Arc<CudaGpu>, graph: CuGraph, exec: CuGraphExec, nodes: Vec<CuGraphNode>,
                       launches: Vec<(CudaKernel, Launch, Args)>, fingerprint: Fingerprint, kept: Vec<Box<dyn Any + Send>> }
// Drop: cuGraphExecDestroy (freed by the driver once in-flight replays end), cuGraphDestroy.
```
`record(streams, f)`, step by step:
1. Check the poison. Lock the streams in ascending lane order. Each stream creates its shadow (`STREAM_NON_BLOCKING`) on first use and keeps it.
2. Create the recording's events (`DISABLE_TIMING`), destroyed at the end.
3. `begin_capture(shadow0, CAPTURE_MODE_RELAXED)`. Record a fork event on shadow0, and every other shadow waits on it.
4. Fork the real streams too: `mark` streams[0] (ring), and every other real stream awaits it.
5. Run `f` with `Pass::Record`. Each op goes to the real stream first, then to the shadow (unless the capture has already failed).
   - A real error returns at once; the guard of step 7 still runs.
   - A shadow error is stored in `failed`.
   - After each shadow launch, `capture_info(shadow)` must report status active and exactly one dependency: that node goes into `nodes`. Anything else stores `failed`.
   - `order(a, b)`: real `mark(a)` + `await_marker(b)`; shadow records a recording event on shadow_a, and shadow_b waits on it.
   - `keep` goes to the graph, not to the stream.
6. Join: every other shadow records a join event and shadow0 waits on all of them. The real streams do the same with ring markers.
7. A guard always calls `end_capture(shadow0)`. If `failed` is set, or end or instantiate fails, destroy what exists and return `Err(Capture(..))`. The real work was submitted regardless.
8. `instantiate(&exec, graph, 0)`.

`replay(stream)`: lock the stream, check the poison, then `cuGraphLaunch(exec, stream.raw)`.

- [ ] **Step 1: Write failing tests** in `tests/cuda_graph.rs` (skip with no driver; the kernels come from `kernels/ops.ptx` via `include_bytes!("kernels/ops.ptx")`, so NVRTC is not needed). `seq(q, bufs, a, b)`:
  - stream 0: `axpb(y, x, a, b)`; `order(0, 1)`;
  - stream 1: `add_u32` over y's bits into z with c = 3; `order(1, 0)`;
  - stream 0: copy z into pinned.

  Tests:
  - `record_runs_the_sequence_once_and_replay_repeats_it`: after `record`, poll the s0 marker; pinned equals the CPU reference. Zero z, `replay(&s0)`, `mark`, poll; equal again.
  - `replay_matches_direct_submission_bit_for_bit`: the same sequence submitted directly on fresh buffers gives identical bytes.
  - `a_replay_runs_on_any_stream`: replay on s1 gives the same result.
  - `submissions_from_other_threads_wait_out_a_recording`: `record`'s closure sets `inside = true`, submits its launches, sleeps 20 ms and sets `inside = false`. A peer thread waits for `inside`, then calls `s0.submit`, whose closure notes `inside`. The note is `false` (the peer waited the recording out), and `graph.launches()` is exactly the closure's count. File-level allow for threads and sleeps (a test fixture).
- [ ] **Step 2: Run** [GPU] `cargo test -p euhedral-gpu --features cuda --test cuda_graph`. Expected: FAIL.
- [ ] **Step 3: Implement** `graph.rs`, `Pass::Record` in `CudaEnqueue`, and `CudaGpu::record`.
- [ ] **Step 4: Run** the same command. Expected: PASS.
- [ ] **Step 5: Commit.** `git commit -m "euhedral-gpu: CUDA graph capture on shadow streams, and replay"`

### Task 19: Argument updates and the fingerprint probe

**Files:**
- Modify: `euhedral-gpu/src/cuda/graph.rs`, `src/cuda/stream.rs`, `tests/cuda_graph.rs`

**Interfaces:**
- Produces:
  - `Pass::Probe`: ops only feed the `ShapeHasher`.
  - `Pass::Update(&mut CudaGraph)`: ops feed the hasher, and launch n stages `(n, args.clone())`.
  - After `f`, a fingerprint different from the graph's returns `ShapeChanged { recorded, now }` and nothing is applied.
  - Otherwise, for each staged launch whose `args` differ from the stored ones, call `graph_exec_kernel_node_set_params(exec, nodes[n], &KernelNodeParams { func, geometry, kernel_params: staged.pointers(), extra: null, kern: null, ctx: null })` and store the args.
  - Errors go through `gpu.check`.

- [ ] **Step 1: Write failing tests:**
  - `replays_with_changing_arguments_match_direct_submission`: for `a` in `[0.5, 1.0, 2.0, 3.5]`, `update(seq(a))` + replay equals direct `seq(a)` bit for bit.
  - `updates_between_replays_do_not_touch_a_replay_in_flight` (Review Focus 5):
    1. Record a sequence that begins with `spin` (≈30 ms) and writes into `out[k]`, with `k` an argument picked through a device pointer argument.
    2. Replay with `(a=1, out0)`, then immediately `update(a=2, out1)` and replay. Do not wait between them.
    3. `out0` holds the a=1 result and `out1` the a=2 result.
  - `the_probe_fingerprint_equals_the_recordings`: `gpu.fingerprint(2, |q| seq(q, .., 9.0, ..))` equals `graph.fingerprint()`.
  - `an_update_of_another_shape_applies_nothing`: an update with another `Launch` geometry is `Err(ShapeChanged { .. })`, and the next replay still gives the previous arguments' result.
- [ ] **Step 2: Run** [GPU] `cargo test -p euhedral-gpu --features cuda --test cuda_graph`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the same command. Expected: PASS.
- [ ] **Step 5: Commit.** `git commit -m "euhedral-gpu: graph argument updates in place, and the fingerprint probe"`

### Task 20: Re-record through executable update, else re-instantiate

**Files:**
- Modify: `euhedral-gpu/src/cuda/graph.rs`, `src/cuda/mod.rs`, `tests/cuda_graph.rs`

**Interfaces:**
- Produces: `rerecord(graph, streams, f)`.
  1. Run record steps 1–7 into a new `CuGraph`, with fresh `nodes` and `launches`. The real run happens.
  2. Call `graph_exec_update(exec, new, &mut info)`.
     - Success (`info.result == 0`): destroy `new`, keep `graph.graph` and the old `nodes` (Decision 12). Replace `launches`, `fingerprint` and the stored args with the new recording's. **Append** the new kept values. Re-apply the new recording's args to the old nodes with `set_params`, so the executable carries them. Return `Updated`.
     - Otherwise: instantiate `new`; destroy the old executable and graph; replace everything (kept values appended); return `Instantiated`.

- [ ] **Step 1: Write failing tests:**
  - `a_rerecord_of_the_same_topology_updates_in_place`: the same sequence with the copy into another pinned buffer returns `Updated`. A replay writes the new buffer. A following `update` with new args still takes effect, which checks that the old node handles stay valid.
  - `a_rerecord_of_another_topology_instantiates_anew`: an extra launch returns `Instantiated`, and the replay matches the direct result.
- [ ] **Step 2: Run** [GPU] `cargo test -p euhedral-gpu --features cuda --test cuda_graph rerecord`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the same command. Expected: PASS.
- [ ] **Step 5: Commit.** `git commit -m "euhedral-gpu: re-record through executable update, else re-instantiate"`

### Task 21: A failed capture leaves the real work intact

**Files:**
- Modify: `euhedral-gpu/tests/cuda_graph.rs` (plus fixes in `src/cuda/graph.rs`)

- [ ] **Step 1: Write the test** `a_failed_capture_never_disturbs_the_real_work`. Record a sequence whose stream-1 op is `copy_to_device` from a *pageable* `Vec<u8>`; the driver refuses pageable copies inside a capture.
  - `record` returns `Err(Capture(_))`, and the device result of the real run is correct.
  - A later direct submission on both streams works, which checks that the shadows have left capture mode.
  - If this driver captures the pageable copy without error, the test must still be meaningful. Inject the failure with `CudaSequence::fail_capture(&mut self)` instead: `#[doc(hidden)] pub`, documented as a test hook that marks the capture failed. Assert the same things.
- [ ] **Step 2: Run** [GPU] `cargo test -p euhedral-gpu --features cuda --test cuda_graph failed_capture`. Expected: PASS after any fix.
- [ ] **Step 3: Commit.** `git commit -m "euhedral-gpu: a failed capture leaves the stream's own work intact"`

**Phase 6 gate:** the global gate; [GPU] `cargo test -p euhedral-gpu --features cuda --test cuda_graph`; the fake tests still pass.

---

## Phase 7: the acceptance program

### Task 22: `examples/graph_replay.rs`

**Files:**
- Create: `euhedral-gpu/examples/graph_replay.rs`, `euhedral-gpu/examples/kernels/pipeline.cu`
- Modify: `euhedral-gpu/Cargo.toml` (`[[example]] name = "graph_replay", required-features = ["cuda"]`)

**The program.** Integer kernels keep the CPU reference exact.

`pipeline.cu` (the example's own kernels):
- `extern "C" __global__ void mix(unsigned* y, const unsigned* x, unsigned a, unsigned n)`: `y[i] = x[i] * a + i`.
- `extern "C" __global__ void fold(unsigned* z, const unsigned* y, unsigned b, unsigned n)`: `z[i] = (y[i] ^ b) + (y[i] >> 3)`.

`graph_replay.rs`:
1. Get the device with `CudaGpu::open(0)`. Load the kernels with `CudaModule::from_source(include_str!("kernels/pipeline.cu"), CompileOptions { cache_dir: Some(dir), .. })`. `dir` is `$EUHEDRAL_GPU_CACHE` when set, else a `gpu-module-cache` directory beside the example's executable (`std::env::current_exe()`'s parent, inside the cargo target directory). The file header documents both.
2. Open two lanes, `compute-a` and `compute-b`. Allocate x, y, z (device, through `euhedral_memory::cuda::DeviceMemory`) and a pinned output ring of 2. Fill x on the host.
3. The sequence `seq(q, i)`:
   - stream 0: `mix(y, x, a_i)`; `order(0, 1)`;
   - stream 1: `fold(z, y, b_i)`; `order(1, 0)`;
   - stream 0: copy z into `pinned[i % 2]`.
   - `a_i = 3 + i` and `b_i = 0x9e37_79b9 ^ i`.
4. `record` with i = 0. This is iteration 0's real run.
5. Build a lattice: `LatticeConfig::with_executor(DefaultExecutor::new())` on the first allowed CPU (`euhedral_hardware_utils::affinity::base_mask`). Add a `Completions` over the pool with `add_upstream`.
6. Chain: a continuation frame for iteration i runs on the lattice. It:
   1. confirms the retirement ticket (`None`, or the run fails);
   2. compares `pinned[i % 2]` with the CPU reference for i;
   3. if i < 100, runs `gpu.update(graph, seq(.., i+1))`, `graph.replay(&s0)`, then `retire_into(&s0, &completions, next frame)`.

   The graph sits in a `Mutex` in the shared state: only one continuation runs at a time. The main thread arms iteration 0's retirement after `record`.
7. The main thread waits on a `euhedral_graph::Latch` (main thread, not the lattice). It prints `100 replays matched the CPU reference; lateness p50/p99/max µs: …` from `completions.lateness_us()`. Then it completes the source and closes the lattice.
8. Any mismatch or `GpuError` sets the latch with `Err`, and `main` exits non-zero.

- [ ] **Step 1: Write the example and its kernel** as above. Comments speak generically: "a recurring sequence", "replays".
- [ ] **Step 2: Build** `cargo build -p euhedral-gpu --features cuda --example graph_replay`. Expected: OK.
- [ ] **Step 3: Run** [GPU] `CUDA_HOME=/home/brandon/qwen38-work/cuda-13.1 cargo run -p euhedral-gpu --features cuda --example graph_replay`. Expected: exit 0 and the line `100 replays matched the CPU reference; …`. Run it a second time: the cache hit makes start-up faster (observe, do not assert).
- [ ] **Step 4: Check that no lattice code waits on the host:** `grep -nE 'synchronize|thread::sleep|\.wait\(' euhedral-gpu/examples/graph_replay.rs`. Expected: only the main thread's `Latch` wait.
- [ ] **Step 5: Commit.** `git commit -m "euhedral-gpu: acceptance example: own kernels, two streams, a graph replayed 100 times on the lattice"`

---

## Phase 8: documentation

### Task 23: README, crate docs, CHANGELOG, GLOSSARY, workspace docs

**Files:**
- Create: `euhedral-gpu/README.md`
- Modify:
  - `euhedral-gpu/src/lib.rs`: crate docs; the README doctest hook `#[cfg(doctest)] #[doc = include_str!("../README.md")] struct ReadmeDoctests;`.
  - `CHANGELOG.md`: a new top entry.
  - `GLOSSARY.md`: a new `## GPU` section after `## Memory`.
  - `README.md` (root): a crates table row; replace the dependency paragraph with the layer list.
  - `euhedral-memory/README.md`: "Where it sits", since `euhedral-gpu` builds on it now; mention `Driver::shared` and `Driver::symbol` in the `cuda` row.

**Content (decided):**
- `euhedral-gpu/README.md`, in the structure of `euhedral-memory/README.md`:
  - A summary.
  - **Where it sits**: builds on `euhedral-core`, `euhedral-graph`, `euhedral-memory`, `euhedral-hashing`; it is the top layer.
  - A modules table.
  - Key concepts, one short section each:
    - streams are lanes;
    - markers across streams;
    - retirement (listener rules; `retire_into`; confirm; kept values);
    - kernels from outside RustEE (`from_source` with the NVRTC search order and the cache; `from_image`);
    - recurring sequences (record, probe, update, rerecord; the shape; replay and lane choice);
    - failures are values (faults poison; a poisoned device stays poisoned for the process);
    - build-time calls (module loading, drops).
  - Features: `fake`, `cuda`.
  - Build and test: the `--features cuda` commands, and `CUDA_HOME` or `EUHEDRAL_NVRTC` to find NVRTC.
  - One doctest on the fake backend: two lanes, a marker, `retire_into`, and a pull loop, as in `euhedral-graph`'s README.
- CHANGELOG entry `## GPU access: euhedral-gpu (2026-10-28)`, with **What / Why / Use / Tests** like its neighbours. List each test file and the key test names. Add a line on `euhedral-memory`'s `Driver::shared`/`Driver::symbol`.
- GLOSSARY entries:
  - **GPU stream**
  - **retirement** (`notify_retired`, `confirm_retired`, `RetireTicket`, `retire_into`)
  - **poisoned device**
  - **module** (`from_source`, `from_image`, module cache)
  - **kernel arguments** (`Args`, `KernelArg`)
  - **sequence** (record, probe, update passes)
  - **graph** (replay, `rerecord`)
  - **shape and fingerprint**
  - **shadow stream**

  Each links to its file.
- Root README: a row `| [euhedral-gpu](euhedral-gpu/README.md) | Asynchronous, backend-neutral GPU access: streams as device lanes, markers, retirement as frames, kernels loaded at run time, graph record and replay (CUDA backend, feature cuda). |`. The layer list is the Global Constraints line.

- [ ] **Step 1: Write the README with its doctest,** then run `cargo test -p euhedral-gpu --doc`. Expected: PASS (fake doctest, no GPU).
- [ ] **Step 2: Write the other documents.** Then check the language rule: grep `euhedral-gpu`, `CHANGELOG.md`, `GLOSSARY.md`, `README.md` and `euhedral-memory/README.md` (case-insensitive) for application-domain words and other repositories' names; the executor's brief gives the list. Expected: no new hits. The repository's existing port notes elsewhere are out of scope. Also `grep -rniE 'does not depend|doesn.t depend|no dependency on' euhedral-gpu`. Expected: no hits.
- [ ] **Step 3: Run the final gate:** `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings && cargo clippy -p euhedral-gpu --all-targets --features cuda -- -D warnings && cargo test --workspace && cargo test -p euhedral-gpu --features cuda`. Then run [GPU] `CUDA_HOME=… cargo test -p euhedral-gpu --features cuda -- --test-threads=2` (the `cuda_fault` binary runs in its own process anyway) and [GPU] the example. Expected: all green.
- [ ] **Step 4: Commit.** `git commit -m "euhedral-gpu: README, CHANGELOG and GLOSSARY entries; workspace layering in the root README"`

---

## Risks

- **sm_120 and NVRTC versions.**
  - NVRTC older than 12.8 does not know `sm_120`. `target_for` then falls back to the newest `compute_XY` PTX, which the driver JITs (first load is slow, then the driver's own cache serves it).
  - PTX whose ISA is newer than the driver fails with `CUDA_ERROR_UNSUPPORTED_PTX_VERSION` (222), and `from_image` reports it as `Load` with the JIT log.
  - The committed `ops.ptx` comes from CUDA 13.1 and needs a 13.1+ driver. Its test skips below `driver_version() < 13010`.
- **NVRTC's builtins library.** `libnvrtc.so.13` loads `libnvrtc-builtins.so.13.1` by soname, and toolkit copies carry no RPATH. The loader preloads the builtins from the same directory with `RTLD_GLOBAL` (Task 12). A pip-style layout works the same way.
- **`cuStreamAddCallback` vs `cuLaunchHostFunc`.**
  - The former is "slated for eventual deprecation" but is the only one documented to run on device errors. The latter is the fallback when the symbol disappears; `retire_into`'s poisoned path then prevents stranded continuations.
  - Task 17 confirms the callback's behaviour after a *sticky* fault on this driver. If it is not called, the poisoned path is what the test asserts.
  - A listener panic is caught in the trampoline.
- **Callback record free.** Each callback's boxed record (the listener) is freed on the driver's callback thread after the listener returns. That is a host heap free, not a driver free; no frame is built or dropped there.
- **Capture-info ABI.** `cuStreamGetCaptureInfo_v2` (6 args) and `_v3` (7 args, adds edge data) are both exported by driver 615. `GpuFns` prefers `_v3`. A launch whose capture shows more than one current dependency fails the capture rather than guessing.
- **Kernel node params layout.** `CUDA_KERNEL_NODE_PARAMS_v2` (CUDA 12.0+) adds `kern` and `ctx` after `extra`. A wrong layout would mis-set parameters, and Task 19's bit-for-bit tests catch it.
- **Executable update failure modes.** `cuGraphExecUpdate_v2` can return:
  - `CU_GRAPH_EXEC_UPDATE_ERROR` (1), `TOPOLOGY_CHANGED` (2), `NODE_TYPE_CHANGED` (3), `FUNCTION_CHANGED` (4), `PARAMETERS_CHANGED` (5), `NOT_SUPPORTED` (6), `UNSUPPORTED_FUNCTION_CHANGE` (7) or `ATTRIBUTES_CHANGED` (8);
  - or an error code.

  Every one of these leads to re-instantiation, never to a half-updated executable that is kept. Which node handles stay valid after a successful update is settled by keeping the original graph (Decision 12); Task 20's second assertion checks it.
- **Sticky faults.** An illegal address poisons the primary context for the process: every later CUDA call fails. The device's poison makes that a fast, typed failure, and recovery means a new process (documented). The fault test stays in its own binary.
- **Pageable host memory in recordings.** Captured copies keep their host address and read it again at every replay. Use pinned memory (`euhedral_memory::cuda::PinnedHost`), kept through `Enqueue::keep` for the graph's life. A graph dropped while a replay is in flight frees kept host memory early; dropping a graph is documented as a call made after its last replay retired.
- **Locks.** The per-stream `Mutex` is held across the submit closure. A slow closure delays other submitters to that stream, so the docs say to keep closures to enqueueing. A recording holds its streams for the whole closure; record at setup or between uses.
- **Marker ring wrap** may over-wait (never under-wait), as Decision 6 says. Raise `StreamOptions::markers` if `Busy` shows up under load.
- **Relaxed capture mode** does not stop other threads from calling unsafe APIs during a capture. The crate makes no such calls on shadow streams. An application calling `cuMemAlloc` meanwhile is legal in relaxed mode.

## Questions for the user

None. Every builder-left item is settled above within the approved design.
