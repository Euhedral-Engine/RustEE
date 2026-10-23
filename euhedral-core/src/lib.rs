//! Frames, frame pools and pipelines executed by the Euhedral control plane.
//!
//! - [`frames`]: the [`Frame`](frames::Frame) trait every unit of work implements, and the stock
//!   frames.
//! - [`generics`]: the interfaces frames flow through, and executors.
//! - [`impls`]: pools that recycle finished frames (upstream package `impl`).
//! - [`config`]: where each clone of a control-plane object runs.
//! - [`flow_control`]: how far a frame may travel from where it was made.
//!
//! A frame is owned by whoever holds its `Box`: queues, executors and pools move it rather than
//! share it, so a frame needs no locking and is never touched by two threads at once.

pub mod config;
pub mod flow_control;
pub mod frames;
pub mod generics;
pub mod impls;

mod internal;
