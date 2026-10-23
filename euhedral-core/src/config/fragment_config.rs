use super::{CacheConfig, IdlePolicy};
use std::num::NonZeroU64;

/// How a [`ControlPlaneFragment`](crate::control_plane::ControlPlaneFragment) caches, batches and
/// idles.
///
/// The config is also the fragments' template: cloned onto a core, it builds a fragment there.
#[derive(Clone, Debug)]
pub struct FragmentConfig {
    cache_config: CacheConfig,
    max_batch_size: NonZeroU64,
    smt_enabled: bool,
    idle_policy: IdlePolicy,
}

impl Default for FragmentConfig {
    /// The default cache and idle policy, batches of up to 4096 frames, and a worker on each SMT
    /// thread of a core.
    fn default() -> Self {
        Self {
            cache_config: CacheConfig::default(),
            max_batch_size: Self::DEFAULT_MAX_BATCH_SIZE,
            smt_enabled: true,
            idle_policy: IdlePolicy::default(),
        }
    }
}

impl FragmentConfig {
    pub const DEFAULT_MAX_BATCH_SIZE: NonZeroU64 = NonZeroU64::new(4096).unwrap();

    /// The config of each fragment's core-local cache.
    pub fn cache_config(&self) -> &CacheConfig {
        &self.cache_config
    }

    pub fn with_cache_config(self, cache_config: CacheConfig) -> Self {
        Self {
            cache_config,
            ..self
        }
    }

    /// The largest batch a worker grows to; its cache's capacity and the CPU's pressure may cap it
    /// lower.
    pub fn max_batch_size(&self) -> NonZeroU64 {
        self.max_batch_size
    }

    pub fn with_max_batch_size(self, max_batch_size: NonZeroU64) -> Self {
        Self {
            max_batch_size,
            ..self
        }
    }

    /// Whether a core with two effective CPUs runs a worker on each, splitting the core's frames
    /// and its L1 and L2 between them.
    pub fn smt_enabled(&self) -> bool {
        self.smt_enabled
    }

    pub fn with_smt_enabled(self, smt_enabled: bool) -> Self {
        Self {
            smt_enabled,
            ..self
        }
    }

    /// How workers without work park.
    pub fn idle_policy(&self) -> &IdlePolicy {
        &self.idle_policy
    }

    pub fn with_idle_policy(self, idle_policy: IdlePolicy) -> Self {
        Self {
            idle_policy,
            ..self
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn defaults() {
        let config = FragmentConfig::default();
        assert_eq!(config.cache_config(), &CacheConfig::default());
        assert_eq!(config.max_batch_size().get(), 4096);
        assert!(config.smt_enabled());
        assert_eq!(
            config.idle_policy().idle_park(),
            IdlePolicy::DEFAULT_IDLE_PARK
        );
    }

    #[test]
    fn builders_set_their_field() {
        let cache = CacheConfig::default().with_work_steal(false);
        let idle = IdlePolicy::fixed(Duration::from_micros(1), Duration::from_micros(2)).unwrap();
        let config = FragmentConfig::default()
            .with_cache_config(cache.clone())
            .with_max_batch_size(NonZeroU64::new(64).unwrap())
            .with_smt_enabled(false)
            .with_idle_policy(idle);
        assert_eq!(config.cache_config(), &cache);
        assert_eq!(config.max_batch_size().get(), 64);
        assert!(!config.smt_enabled());
        assert_eq!(config.idle_policy().idle_park(), Duration::from_micros(1));
    }
}
