//! Configuration of a control-plane lattice, its core-local caches, and the objects it clones onto
//! each core.

mod cache_config;
mod clone_config;
mod lattice_config;
mod liveness;

pub use cache_config::{CacheConfig, InvalidMemoryBudget};
pub use clone_config::{CloneConfig, RegistryMismatch};
pub use lattice_config::LatticeConfig;
pub use liveness::{CloneLivenessRegistry, Worker, WorkerSlot};
