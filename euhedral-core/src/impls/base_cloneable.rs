use super::DefaultExecutor;
use crate::config::{CloneConfig, CloneLivenessRegistry, FragmentConfig};
use crate::control_plane::ControlPlaneFragment;
use crate::flow_control::LatticeEdge;
use crate::generics::{AbstractExecutor, CloneableObject};
use crate::metrics::MetricsRegistry;
use euhedral_hardware_utils::affinity::PinnedThreadExecutor;
use euhedral_hardware_utils::monitor::CoreSnapshot;
use std::panic;
use std::sync::Arc;
use std::time::Instant;

/// The stock [`CloneableObject`]: on each core, a [`ControlPlaneFragment`] whose workers hand
/// their frames to an [`AbstractExecutor`].
///
/// The object passed to the lattice is a template. Each clone is built on its core's first CPU,
/// fragment and executors both, and its caches are first touched there, so their memory lands on
/// that CPU's NUMA node. The fragment's output, and its SMT buddy's, each feed a clone of the
/// executor bound to the worker's CPU, which runs the frames on that worker's thread. Lifecycle
/// calls go to the fragment; executors have none of their own.
pub struct BaseCloneableObject<E: AbstractExecutor = DefaultExecutor> {
    name: String,
    config: Option<CloneConfig>,
    fragment_config: FragmentConfig,
    fragment: Option<Arc<ControlPlaneFragment>>,
    executors: Vec<Arc<E>>,
}

impl Default for BaseCloneableObject {
    /// The default fragment, running frames with a [`DefaultExecutor`].
    fn default() -> Self {
        Self::new(FragmentConfig::default(), DefaultExecutor::new())
    }
}

impl<E: AbstractExecutor> BaseCloneableObject<E> {
    /// A template whose clones run fragments of `fragment_config` and clones of `executor`.
    pub fn new(fragment_config: FragmentConfig, executor: E) -> Self {
        Self {
            name: "BaseCloneableObject".to_string(),
            config: None,
            fragment_config,
            fragment: None,
            executors: vec![Arc::new(executor)],
        }
    }

    /// A template whose clones run default fragments and clones of `executor`.
    pub fn with_executor(executor: E) -> Self {
        Self::new(FragmentConfig::default(), executor)
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn fragment_config(&self) -> &FragmentConfig {
        &self.fragment_config
    }

    /// The clone's fragment; `None` for the template.
    pub fn fragment(&self) -> Option<&Arc<ControlPlaneFragment>> {
        self.fragment.as_ref()
    }

    /// The executor fed by the fragment, then the one fed by its SMT buddy, if any. The template
    /// holds just the executor it clones.
    pub fn executors(&self) -> &[Arc<E>] {
        &self.executors
    }

    /// The clone placed by `config`; see [`clone_object`](CloneableObject::clone_object).
    fn place(&self, config: &CloneConfig) -> Arc<Self> {
        let cpu = config
            .effective_cpus()
            .first()
            .unwrap_or_else(|| panic!("core {} has no effective CPU", config.core_id()));
        let pinned = PinnedThreadExecutor::get_or_create(
            cpu,
            &format!("{}-Worker-{}", config.shard_name(), config.core_id()),
        )
        .unwrap_or_else(|error| panic!("no pinned executor for CPU {cpu}: {error}"));
        let fragment_config = self.fragment_config.clone();
        let executor = self.executors[0].clone();
        let placement = config.clone();
        let built = pinned
            .submit(move || Self::build(fragment_config, &executor, placement))
            .unwrap_or_else(|rejected| {
                panic!("CPU {cpu}'s pinned executor rejected the build: {rejected}")
            })
            .join();
        match built {
            Ok(clone) => Arc::new(clone),
            Err(panic) => panic::resume_unwind(panic),
        }
    }

    /// Builds the clone placed by `config` on the calling thread, wires its outputs and first
    /// touches its caches.
    fn build(fragment_config: FragmentConfig, executor: &E, config: CloneConfig) -> Self {
        let fragment = match ControlPlaneFragment::new(&fragment_config, &config) {
            Ok(fragment) => fragment,
            Err(error) => panic!(
                "cannot build the fragment of core {}: {error}",
                config.core_id()
            ),
        };
        let workers = std::iter::once(&fragment).chain(fragment.smt_buddy());
        let executors = workers
            .map(|worker| {
                let executor = Arc::new(executor.hook_on_clone(Some(worker.cpu())));
                if let Some(output) = worker.output() {
                    executor.clone().input(output);
                }
                executor
            })
            .collect();
        fragment.first_touch();
        Self {
            name: format!("{}-pipeline-{}", config.shard_name(), config.core_id()),
            config: Some(config),
            fragment_config,
            fragment: Some(fragment),
            executors,
        }
    }
}

impl<E: AbstractExecutor> CloneableObject for BaseCloneableObject<E> {
    /// A clone placed by `config`, built on its first effective CPU's pinned executor. The executor
    /// is shared with every other user of that CPU and is left as it was found: the build is one task
    /// on it, and releasing the executor ends only this call's use of it.
    ///
    /// # Panics
    ///
    /// If `config` has no effective CPU, the CPU has no pinned executor, or building the fragment
    /// fails; see [`ControlPlaneFragment::new`].
    fn clone_object(&self, config: &CloneConfig) -> Arc<dyn CloneableObject> {
        self.place(config)
    }

    fn start(&self) {
        if let Some(fragment) = &self.fragment {
            fragment.start();
        }
    }

    fn is_started(&self) -> bool {
        self.fragment
            .as_ref()
            .is_none_or(|fragment| fragment.is_started())
    }

    /// Whether the fragment's workers have registered, so routed frames will be run.
    fn ready(&self) -> bool {
        self.fragment
            .as_ref()
            .is_none_or(|fragment| fragment.ready())
    }

    fn update(&self, snapshot: &CoreSnapshot) {
        if let Some(fragment) = &self.fragment {
            fragment.update(snapshot);
        }
    }

    /// Attaches the fragment's cache below `edge`.
    fn input_edge(self: Arc<Self>, edge: Arc<LatticeEdge>) {
        if let Some(fragment) = &self.fragment {
            fragment.clone().input_edge(edge);
        }
    }

    fn is_drained(&self) -> bool {
        self.fragment
            .as_ref()
            .is_none_or(|fragment| fragment.is_drained())
    }

    fn set_drain_mode(&self, drain: bool) {
        if let Some(fragment) = &self.fragment {
            fragment.set_drain_mode(drain);
        }
    }

    fn reset(&self, deadline: Instant) -> u64 {
        self.fragment
            .as_ref()
            .map_or(0, |fragment| fragment.reset(deadline))
    }

    fn dump_locks(&self) {
        if let Some(fragment) = &self.fragment {
            fragment.dump_locks();
        }
    }

    fn core(&self) -> Option<usize> {
        self.config.as_ref().map(CloneConfig::core_id)
    }

    /// The fragment config's registry, for the template and its clones alike.
    fn metrics_registry(&self) -> Option<Arc<MetricsRegistry>> {
        self.fragment_config.metrics_registry()
    }

    fn liveness_registry(&self) -> Option<Arc<CloneLivenessRegistry>> {
        self.config
            .as_ref()
            .map(|config| config.liveness_registry().clone())
    }

    fn close(&self) {
        if let Some(fragment) = &self.fragment {
            fragment.close();
        }
    }
}

#[cfg(test)]
mod tests;
