//! Frames, frame pools and pipelines executed by the Euhedral control plane.
//!
//! - [`frames`]: the [`Frame`](frames::Frame) trait every unit of work implements, and the stock
//!   frames.
//! - [`generics`]: the interfaces frames flow through, and executors.
//! - [`impls`]: pools that recycle finished frames, and the default executor (upstream package
//!   `impl`).
//! - [`config`]: where each clone of a control-plane object runs.
//! - [`flow_control`]: the lattice frames flow down through and demand flows up through: edges,
//!   routing vertices, upstream handles and each worker's scheduler over them.
//! - [`utils`]: math, spin waiting, timing, flow statistics and calibration for the control plane.
//!
//! A frame is owned by whoever holds its `Box`: queues, executors and pools move it rather than
//! share it, so a frame needs no locking and is never touched by two threads at once.

pub mod config;
pub mod flow_control;
pub mod frames;
pub mod generics;
pub mod impls;
pub mod utils;

mod internal;
