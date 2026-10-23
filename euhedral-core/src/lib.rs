//! Frames, frame pools and pipelines executed by the Euhedral control plane.
//!
//! - [`frames`]: the [`Frame`](frames::Frame) trait every unit of work implements, and the stock
//!   frames.
//! - [`generics`]: the interfaces frames flow through, and executors.
//! - [`impls`]: pools that recycle finished frames, the default executor, and the default object a
//!   lattice clones onto each core: a fragment feeding an executor (upstream package `impl`).
//! - [`ingest`]: sinks that feed frames into a lattice from outside it: a queue producers offer
//!   to, a fixed batch, a single frame, and a runner of typed pipelines.
//! - [`config`]: how a lattice, its fragments, caches and idle workers are set up, and where each
//!   clone of a control-plane object runs.
//! - [`control_plane`]: the lattice where frames enter, and the shard per socket that places a
//!   clone of an object on each effective core and routes frames among them; the fragment that
//!   runs a core's frames on pinned workers, and the core-local cache that holds them.
//! - [`flow_control`]: the lattice frames flow down through and demand flows up through: edges,
//!   routing vertices, upstream handles and each worker's scheduler over them.
//! - [`utils`]: math, spin waiting, timing, flow statistics and calibration for the control plane.
//!
//! A frame is owned by whoever holds its `Box`: queues, executors and pools move it rather than
//! share it, so a frame needs no locking and is never touched by two threads at once.

pub mod config;
pub mod control_plane;
pub mod flow_control;
pub mod frames;
pub mod generics;
pub mod impls;
pub mod ingest;
pub mod utils;

mod internal;
