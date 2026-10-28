//! Asynchronous, backend-neutral GPU access on the Euhedral control plane.
//!
//! Streams are device lanes, markers order work across streams on the device, retirement is a completion that
//! continues as a frame, kernels are loaded at run time, and a recurring sequence of submissions is recorded once and
//! replayed. Nothing here waits on the device from a lattice worker.
//!
//! | Piece | Module | What it is |
//! |---|---|---|
//! | [`Gpu`], [`Stream`], [`Enqueue`] | `api` | a device, an ordered queue of device work, and what a submission enqueues |
//! | [`Module`], [`Kernel`], [`Args`] | `api`, `args` | code loaded at run time and its typed arguments |
//! | [`Sequence`], [`Graph`] | `api` | a recurring sequence of submissions, recorded once and replayed |
//! | [`retire_into`], [`RetireListener`] | `retire`, `api` | retirement: the callback queues, the owner drains and dispatches, and it continues as a frame |
//! | [`ModuleCache`] | `module_cache` | compiled modules keyed by a content hash |
//! | `fake::FakeGpu` (feature `fake`) | `fake` | a stepped device for tests, with no wall clock |
//! | `cuda::CudaGpu` (feature `cuda`) | `cuda` | the CUDA backend on `euhedral-memory`'s driver and context |
//!
//! Every public type is also re-exported at the crate root. The README has the key concepts and a runnable example.

mod api;
mod args;
#[cfg(feature = "cuda")]
pub mod cuda;
mod error;
#[cfg(feature = "fake")]
pub mod fake;
#[cfg(any(feature = "fake", feature = "cuda"))]
mod kept;
mod module_cache;
mod poison;
mod retire;
mod shape;

pub use api::{
    CompileOptions, Enqueue, Gpu, Graph, Kernel, Lanes, Module, POISONED_STATUS, Rerecorded,
    RetireListener, RetireTicket, Sequence, Stream, StreamOptions,
};
pub use args::{Args, Dim3, KernelArg, Launch};
pub use error::GpuError;
pub use module_cache::{CacheKey, KeyMaterial, ModuleCache, cache_key};
pub use poison::Poison;
pub use retire::{OnceFrame, RetireError, retire_into};
pub use shape::{CopyKind, Fingerprint, ShapeHasher};

/// Compiles and runs the examples in the crate README as doctests.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;
