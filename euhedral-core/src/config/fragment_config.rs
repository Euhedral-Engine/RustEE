use super::{CacheConfig, IdlePolicy};
use crate::metrics::{MetricsConfig, MetricsRegistry};
use std::num::NonZeroU64;
use std::sync::Arc;

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
    metrics: MetricsConfig,
}

impl Default for FragmentConfig {
    /// The default cache and idle policy, batches of up to 4096 frames, and a worker on each SMT
    /// thread of a core, and no metrics.
    fn default() -> Self {
        Self {
            cache_config: CacheConfig::default(),
            max_batch_size: Self::DEFAULT_MAX_BATCH_SIZE,
            smt_enabled: true,
            idle_policy: IdlePolicy::default(),
            metrics: MetricsConfig::default(),
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

    /// The max batch at or below which workers poll their caches instead of taking from them greedily
    /// ([`polls`](Self::polls)).
    pub const POLL_MODE_MAX_BATCH: u64 = 2;

    /// Whether the workers poll their caches, taking frames one at a time and holding a partition
    /// only while they take a frame out ([`ControlPlaneCache::poll`](crate::control_plane::ControlPlaneCache::poll)):
    /// with a max batch of [`POLL_MODE_MAX_BATCH`](Self::POLL_MODE_MAX_BATCH) or less, for frames
    /// that may block. A blocking frame then holds up no frame queued behind it that another worker
    /// could run.
    pub fn polls(&self) -> bool {
        self.max_batch_size.get() <= Self::POLL_MODE_MAX_BATCH
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

    /// Metrics for the fragments and their caches: sets this config's and its cache config's
    /// prefix and registry (`FragmentConfig.ofDefaults(metricPrefix, registry)` upstream).
    pub fn with_metrics(self, prefix: impl Into<String>, registry: Arc<MetricsRegistry>) -> Self {
        let prefix = prefix.into();
        let cache_config = self
            .cache_config
            .clone()
            .with_metric_prefix(prefix.clone())
            .with_registry(registry.clone());
        Self {
            cache_config,
            ..self.with_metric_prefix(prefix).with_registry(registry)
        }
    }

    /// The prefix of the workers' metric names: [`DEFAULT_PREFIX`](crate::metrics::DEFAULT_PREFIX)
    /// unless another was set.
    pub fn metric_prefix(&self) -> &str {
        self.metrics.prefix()
    }

    /// Sets the workers' metric prefix, not the cache config's; a blank one means the default.
    pub fn with_metric_prefix(self, prefix: impl Into<String>) -> Self {
        Self {
            metrics: self.metrics.with_prefix(prefix),
            ..self
        }
    }

    /// Where the workers register their meters; `None`, the default, records nothing.
    pub fn registry(&self) -> Option<&Arc<MetricsRegistry>> {
        self.metrics.registry()
    }

    /// Sets the workers' registry, not the cache config's, or with `None` turns the workers'
    /// metrics off.
    pub fn with_registry(self, registry: impl Into<Option<Arc<MetricsRegistry>>>) -> Self {
        Self {
            metrics: self.metrics.with_registry(registry.into()),
            ..self
        }
    }

    pub(crate) fn metrics(&self) -> &MetricsConfig {
        &self.metrics
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
        assert_eq!(config.metric_prefix(), "euhedral");
        assert!(config.registry().is_none());
    }

    #[test]
    fn with_metrics_sets_the_fragment_and_its_cache() {
        let registry = MetricsRegistry::new();
        let config = FragmentConfig::default().with_metrics("app.", registry.clone());
        assert_eq!(config.metric_prefix(), "app.");
        assert_eq!(config.cache_config().metric_prefix(), "app.");
        assert!(Arc::ptr_eq(config.registry().unwrap(), &registry));
        assert!(Arc::ptr_eq(
            config.cache_config().registry().unwrap(),
            &registry
        ));

        let fragment_only = FragmentConfig::default().with_registry(registry);
        assert!(fragment_only.registry().is_some());
        assert!(fragment_only.cache_config().registry().is_none());
        assert!(fragment_only.with_registry(None).registry().is_none());
    }

    #[test]
    fn workers_poll_at_a_max_batch_of_two_or_less() {
        let with =
            |batch| FragmentConfig::default().with_max_batch_size(NonZeroU64::new(batch).unwrap());
        assert!(!FragmentConfig::default().polls());
        assert!(with(1).polls());
        assert!(with(2).polls());
        assert!(!with(3).polls());
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
