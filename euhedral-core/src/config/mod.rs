//! Configuration of a control-plane lattice and of the objects it clones onto each core.

mod clone_config;
mod lattice_config;
mod liveness;

pub use clone_config::{CloneConfig, RegistryMismatch};
pub use lattice_config::LatticeConfig;
pub use liveness::{CloneLivenessRegistry, Worker, WorkerSlot};
