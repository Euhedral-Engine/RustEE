//! Configuration of a control-plane lattice, its core-local caches, and the objects it clones onto
//! each core.

mod cache_config;
mod clone_config;
mod fragment_config;
mod idle_policy;
mod idle_timing;
mod lattice_config;
mod liveness;

pub use cache_config::{CacheConfig, InvalidMemoryBudget};
pub use clone_config::{CloneConfig, RegistryMismatch};
pub use fragment_config::FragmentConfig;
pub use idle_policy::{IdlePolicy, ZeroHalfLife};
pub use idle_timing::{
    FixedTiming, IdleTimingFunction, InvalidTimingModel, TimingCoefficients, TimingModel,
    TimingProvider,
};
pub use lattice_config::LatticeConfig;
pub use liveness::{CloneLivenessRegistry, Worker, WorkerSlot};
