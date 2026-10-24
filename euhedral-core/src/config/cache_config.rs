use crate::metrics::{MetricsConfig, MetricsRegistry};
use std::error::Error;
use std::fmt;
use std::num::NonZeroUsize;
use std::sync::Arc;

/// How a [`ControlPlaneCache`](crate::control_plane::ControlPlaneCache) sizes, splits and shares
/// its local queue.
#[derive(Clone, Debug, PartialEq)]
pub struct CacheConfig {
    memory_budget: f64,
    partitions: NonZeroUsize,
    max_pooled_chunks: usize,
    ring_walk_reset_threshold: u64,
    work_steal: bool,
    metrics: MetricsConfig,
}

/// A memory budget outside `(0, 1]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InvalidMemoryBudget(pub f64);

impl fmt::Display for InvalidMemoryBudget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "the memory budget must be in (0, 1]; got {}", self.0)
    }
}

impl Error for InvalidMemoryBudget {}

impl Default for CacheConfig {
    /// A budget of 70% of the core's L1 and L2, eight partitions, no chunk pooling, a ring-walk
    /// reset threshold of four frames, work stealing, and no metrics.
    fn default() -> Self {
        Self {
            memory_budget: 0.7,
            partitions: NonZeroUsize::new(8).expect("eight is not zero"),
            max_pooled_chunks: 0,
            ring_walk_reset_threshold: 4,
            work_steal: true,
            metrics: MetricsConfig::default(),
        }
    }
}

impl CacheConfig {
    /// The share of the core's L1 and L2 the queue's chunks are sized to fill, in `(0, 1]`.
    pub fn memory_budget(&self) -> f64 {
        self.memory_budget
    }

    pub fn with_memory_budget(self, memory_budget: f64) -> Result<Self, InvalidMemoryBudget> {
        if !(memory_budget > 0.0 && memory_budget <= 1.0) {
            return Err(InvalidMemoryBudget(memory_budget));
        }
        Ok(Self {
            memory_budget,
            ..self
        })
    }

    /// The number of independent partitions in the queue.
    pub fn partitions(&self) -> NonZeroUsize {
        self.partitions
    }

    pub fn with_partitions(self, partitions: NonZeroUsize) -> Self {
        Self { partitions, ..self }
    }

    /// Retired chunks each partition keeps for reuse once the queue has grown.
    pub fn max_pooled_chunks(&self) -> usize {
        self.max_pooled_chunks
    }

    pub fn with_max_pooled_chunks(self, max_pooled_chunks: usize) -> Self {
        Self {
            max_pooled_chunks,
            ..self
        }
    }

    /// The frames one walk over the partitions must collect for a greedy take to walk them again:
    /// higher values end greedy takes sooner, lower ones keep them going longer.
    pub fn ring_walk_reset_threshold(&self) -> u64 {
        self.ring_walk_reset_threshold
    }

    pub fn with_ring_walk_reset_threshold(self, ring_walk_reset_threshold: u64) -> Self {
        Self {
            ring_walk_reset_threshold,
            ..self
        }
    }

    /// Whether other caches may take unordered frames from this one, and this one from them.
    pub fn work_steal(&self) -> bool {
        self.work_steal
    }

    pub fn with_work_steal(self, work_steal: bool) -> Self {
        Self { work_steal, ..self }
    }

    /// The prefix of the caches' metric names: [`DEFAULT_PREFIX`](crate::metrics::DEFAULT_PREFIX)
    /// unless another was set.
    pub fn metric_prefix(&self) -> &str {
        self.metrics.prefix()
    }

    /// Sets the metric prefix; a blank one means the default.
    pub fn with_metric_prefix(self, prefix: impl Into<String>) -> Self {
        Self {
            metrics: self.metrics.with_prefix(prefix),
            ..self
        }
    }

    /// Where the caches register their meters; `None`, the default, records nothing.
    pub fn registry(&self) -> Option<&Arc<MetricsRegistry>> {
        self.metrics.registry()
    }

    /// Sets the registry, or with `None` turns the caches' metrics off.
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

    #[test]
    fn defaults() {
        let config = CacheConfig::default();
        assert_eq!(config.memory_budget(), 0.7);
        assert_eq!(config.partitions().get(), 8);
        assert_eq!(config.max_pooled_chunks(), 0);
        assert_eq!(config.ring_walk_reset_threshold(), 4);
        assert!(config.work_steal());
        assert_eq!(config.metric_prefix(), "euhedral");
        assert!(config.registry().is_none());
    }

    #[test]
    fn metrics_take_a_prefix_and_a_registry() {
        let registry = MetricsRegistry::new();
        let config = CacheConfig::default()
            .with_metric_prefix("app")
            .with_registry(registry.clone());
        assert_eq!(config.metric_prefix(), "app");
        assert!(Arc::ptr_eq(config.registry().unwrap(), &registry));
        assert_ne!(config, CacheConfig::default().with_metric_prefix("app"));
        assert_ne!(
            config,
            CacheConfig::default()
                .with_metric_prefix("app")
                .with_registry(MetricsRegistry::new())
        );
        assert_eq!(config.clone(), config);
        assert_eq!(config.with_metric_prefix(" ").metric_prefix(), "euhedral");
        assert!(
            CacheConfig::default()
                .with_registry(None)
                .registry()
                .is_none()
        );
    }

    #[test]
    fn the_memory_budget_must_be_in_the_unit_interval() {
        for budget in [0.0, -0.5, 1.5, f64::NAN, f64::INFINITY] {
            assert!(CacheConfig::default().with_memory_budget(budget).is_err());
        }
        let config = CacheConfig::default().with_memory_budget(1.0).unwrap();
        assert_eq!(config.memory_budget(), 1.0);
        assert_eq!(
            InvalidMemoryBudget(2.0).to_string(),
            "the memory budget must be in (0, 1]; got 2"
        );
    }

    #[test]
    fn builders_set_their_field() {
        let config = CacheConfig::default()
            .with_partitions(NonZeroUsize::new(3).unwrap())
            .with_max_pooled_chunks(5)
            .with_ring_walk_reset_threshold(9)
            .with_work_steal(false);
        assert_eq!(config.partitions().get(), 3);
        assert_eq!(config.max_pooled_chunks(), 5);
        assert_eq!(config.ring_walk_reset_threshold(), 9);
        assert!(!config.work_steal());
    }
}
