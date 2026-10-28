//! Asynchronous, backend-neutral GPU access on the Euhedral control plane.
//!
//! Streams are device lanes, markers order work across streams on the device, retirement is a completion that
//! continues as a frame, kernels are loaded at run time, and a recurring sequence of submissions is recorded once and
//! replayed. Nothing here waits on the device from a lattice worker.

mod api;
mod args;
mod error;
mod poison;
mod retire;
mod shape;
mod stream_lock;

pub use api::{
    CompileOptions, Enqueue, Gpu, Graph, Kernel, Lanes, Module, POISONED_STATUS, Rerecorded,
    RetireListener, RetireTicket, Sequence, Stream, StreamOptions,
};
pub use args::{Args, Dim3, KernelArg, Launch};
pub use error::GpuError;
pub use poison::Poison;
pub use retire::retire_into;
pub use shape::{CopyKind, Fingerprint, ShapeHasher};
pub use stream_lock::{StreamGuard, StreamLock};
