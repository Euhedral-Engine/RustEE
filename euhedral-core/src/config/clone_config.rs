use super::CloneLivenessRegistry;
use crate::control_plane::CacheDirectory;
use euhedral_hardware_utils::util::CpuSet;
use std::error::Error;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

/// Where one clone of a [`CloneableObject`](crate::generics::CloneableObject) runs.
///
/// The control plane builds these as it places clones on cores. Equality and hashing cover the
/// clone's identity (shard, core and CPUs) and ignore the liveness registry and cache directory,
/// which are runtime wiring.
#[derive(Clone, Debug)]
pub struct CloneConfig {
    shard_name: String,
    core_id: usize,
    effective_cpus: CpuSet,
    liveness_registry: Arc<CloneLivenessRegistry>,
    cache_directory: Option<Arc<CacheDirectory>>,
}

impl CloneConfig {
    /// A config with a fresh liveness registry for `effective_cpus`.
    pub fn new(shard_name: impl Into<String>, core_id: usize, effective_cpus: CpuSet) -> Self {
        let liveness_registry = Arc::new(CloneLivenessRegistry::new(&effective_cpus));
        Self {
            shard_name: shard_name.into(),
            core_id,
            effective_cpus,
            liveness_registry,
            cache_directory: None,
        }
    }

    /// A config sharing `liveness_registry`, which must cover exactly `effective_cpus`.
    pub fn with_registry(
        shard_name: impl Into<String>,
        core_id: usize,
        effective_cpus: CpuSet,
        liveness_registry: Arc<CloneLivenessRegistry>,
    ) -> Result<Self, RegistryMismatch> {
        if liveness_registry.cpus() != &effective_cpus {
            return Err(RegistryMismatch {
                effective_cpus,
                registry_cpus: liveness_registry.cpus().clone(),
            });
        }
        Ok(Self {
            shard_name: shard_name.into(),
            core_id,
            effective_cpus,
            liveness_registry,
            cache_directory: None,
        })
    }

    /// This config, placing the clone in `directory`'s lattice.
    pub fn with_cache_directory(self, directory: Arc<CacheDirectory>) -> Self {
        Self {
            cache_directory: Some(directory),
            ..self
        }
    }

    /// The shard managing the clone.
    pub fn shard_name(&self) -> &str {
        &self.shard_name
    }

    /// The physical core the clone is assigned to.
    pub fn core_id(&self) -> usize {
        self.core_id
    }

    /// The logical CPUs of that core the clone may use.
    pub fn effective_cpus(&self) -> &CpuSet {
        &self.effective_cpus
    }

    /// The effective CPUs in ascending order.
    pub fn cpus(&self) -> Vec<usize> {
        self.effective_cpus.iter().collect()
    }

    /// Where the clone's workers, and its SMT buddy's, publish their liveness.
    pub fn liveness_registry(&self) -> &Arc<CloneLivenessRegistry> {
        &self.liveness_registry
    }

    /// The caches, upstream registry and topology of the lattice the clone runs in. Every config a
    /// lattice's shards build carries it.
    pub fn cache_directory(&self) -> Option<&Arc<CacheDirectory>> {
        self.cache_directory.as_ref()
    }
}

impl PartialEq for CloneConfig {
    fn eq(&self, other: &Self) -> bool {
        self.core_id == other.core_id
            && self.shard_name == other.shard_name
            && self.effective_cpus == other.effective_cpus
    }
}

impl Eq for CloneConfig {}

impl Hash for CloneConfig {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.shard_name.hash(state);
        self.core_id.hash(state);
        self.effective_cpus.hash(state);
    }
}

/// A liveness registry that does not cover exactly a config's effective CPUs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RegistryMismatch {
    pub effective_cpus: CpuSet,
    pub registry_cpus: CpuSet,
}

impl fmt::Display for RegistryMismatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "liveness registry covers CPUs {} but the effective CPUs are {}",
            self.registry_cpus, self.effective_cpus
        )
    }
}

impl Error for RegistryMismatch {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::hash_map::DefaultHasher;

    fn cpus(ids: &[usize]) -> CpuSet {
        ids.iter().copied().collect()
    }

    fn hash(config: &CloneConfig) -> u64 {
        let mut hasher = DefaultHasher::new();
        config.hash(&mut hasher);
        hasher.finish()
    }

    #[test]
    fn new_builds_a_registry_for_the_effective_cpus() {
        let config = CloneConfig::new("shard", 3, cpus(&[6, 7]));
        assert_eq!(config.shard_name(), "shard");
        assert_eq!(config.core_id(), 3);
        assert_eq!(config.effective_cpus(), &cpus(&[6, 7]));
        assert_eq!(config.cpus(), [6, 7]);
        assert_eq!(config.liveness_registry().cpus(), &cpus(&[6, 7]));
    }

    #[test]
    fn a_shared_registry_must_match_the_effective_cpus() {
        let registry = Arc::new(CloneLivenessRegistry::new(&cpus(&[0, 1])));
        let config =
            CloneConfig::with_registry("shard", 0, cpus(&[0, 1]), registry.clone()).unwrap();
        assert!(Arc::ptr_eq(config.liveness_registry(), &registry));

        let mismatch = CloneConfig::with_registry("shard", 0, cpus(&[0]), registry).unwrap_err();
        assert_eq!(mismatch.effective_cpus, cpus(&[0]));
        assert_eq!(mismatch.registry_cpus, cpus(&[0, 1]));
        assert!(mismatch.to_string().contains("liveness registry"));
    }

    #[test]
    fn equality_ignores_the_registry() {
        let first = CloneConfig::new("shard", 1, cpus(&[2, 3]));
        let second = CloneConfig::new("shard", 1, cpus(&[2, 3]));
        assert!(!Arc::ptr_eq(
            first.liveness_registry(),
            second.liveness_registry()
        ));
        assert_eq!(first, second);
        assert_eq!(hash(&first), hash(&second));
    }

    #[test]
    fn equality_covers_shard_core_and_cpus() {
        let base = CloneConfig::new("shard", 1, cpus(&[2, 3]));
        assert_ne!(base, CloneConfig::new("other", 1, cpus(&[2, 3])));
        assert_ne!(base, CloneConfig::new("shard", 2, cpus(&[2, 3])));
        assert_ne!(base, CloneConfig::new("shard", 1, cpus(&[2])));
    }

    #[test]
    fn a_config_can_carry_a_cache_directory_without_changing_identity() {
        use crate::flow_control::UpstreamRegistry;
        use euhedral_hardware_utils::topology::Topology;

        let plain = CloneConfig::new("shard", 1, cpus(&[2, 3]));
        assert!(plain.cache_directory().is_none());
        let directory = CacheDirectory::new(UpstreamRegistry::new(), Topology::system().clone());
        let placed = plain.clone().with_cache_directory(directory.clone());
        assert!(Arc::ptr_eq(placed.cache_directory().unwrap(), &directory));
        assert_eq!(placed, plain);
        assert_eq!(hash(&placed), hash(&plain));
    }

    #[test]
    fn an_empty_cpu_set_is_allowed() {
        let config = CloneConfig::new("", 0, CpuSet::new());
        assert!(config.cpus().is_empty());
        assert!(config.liveness_registry().workers().is_empty());
    }
}
