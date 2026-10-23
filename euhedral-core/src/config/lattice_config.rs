use crate::generics::{AbstractExecutor, CloneableObject};
use crate::impls::BaseCloneableObject;
use euhedral_hardware_utils::topology::Topology;
use euhedral_hardware_utils::util::CpuSet;
use std::sync::Arc;
use std::time::Duration;

/// How a [`ControlPlaneLattice`](crate::control_plane::ControlPlaneLattice) is named, where it may
/// run, how long it waits, and what it clones onto each core.
#[derive(Clone)]
pub struct LatticeConfig {
    /// The lattice's name, used in logs and thread names; a blank name becomes
    /// `ControlPlaneLattice`.
    pub name: String,
    /// The name of each socket's shard, after the lattice's name and before the socket id.
    pub shard_name: String,
    /// The CPUs the lattice may place clones on.
    pub allowed_cpus: CpuSet,
    /// How long shards and clones get to drain before they are closed.
    pub shutdown_timeout: Duration,
    /// How long starting waits for every shard to become ready.
    pub startup_timeout: Duration,
    /// The object cloned onto each effective core.
    pub cloneable: Arc<dyn CloneableObject>,
}

impl LatticeConfig {
    pub const DEFAULT_NAME: &str = "EuhedralLattice";
    pub const DEFAULT_SHARD_NAME: &str = "EuhedralShard";
    pub const DEFAULT_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(60);
    pub const DEFAULT_STARTUP_TIMEOUT: Duration = Duration::from_secs(60);

    /// The default names and timeouts, clones of `cloneable` on every CPU of the host.
    pub fn new(cloneable: Arc<dyn CloneableObject>) -> Self {
        Self {
            name: Self::DEFAULT_NAME.to_string(),
            shard_name: Self::DEFAULT_SHARD_NAME.to_string(),
            allowed_cpus: Topology::system().cpu_set().clone(),
            shutdown_timeout: Self::DEFAULT_SHUTDOWN_TIMEOUT,
            startup_timeout: Self::DEFAULT_STARTUP_TIMEOUT,
            cloneable,
        }
    }

    /// The defaults, with a [`BaseCloneableObject`] of default fragments running frames with
    /// `executor`.
    pub fn with_executor<E: AbstractExecutor>(executor: E) -> Self {
        Self::new(Arc::new(BaseCloneableObject::with_executor(executor)))
    }
}

impl Default for LatticeConfig {
    /// The default names and timeouts, and the default [`BaseCloneableObject`] on every CPU of the
    /// host.
    fn default() -> Self {
        Self::new(Arc::new(BaseCloneableObject::default()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::CloneConfig;

    struct Unit;

    impl CloneableObject for Unit {
        fn clone_object(&self, _config: &CloneConfig) -> Arc<dyn CloneableObject> {
            Arc::new(Unit)
        }
    }

    #[test]
    fn defaults_name_the_lattice_and_allow_every_host_cpu() {
        let config = LatticeConfig::new(Arc::new(Unit));
        assert_eq!(config.name, "EuhedralLattice");
        assert_eq!(config.shard_name, "EuhedralShard");
        assert_eq!(&config.allowed_cpus, Topology::system().cpu_set());
        assert_eq!(config.shutdown_timeout, Duration::from_secs(60));
        assert_eq!(config.startup_timeout, Duration::from_secs(60));
    }
}
