//! Configuration handed to the objects the control plane clones onto each core.

mod clone_config;
mod liveness;

pub use clone_config::{CloneConfig, RegistryMismatch};
pub use liveness::{CloneLivenessRegistry, Worker, WorkerSlot};
