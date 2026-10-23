use super::LatticeSource;
use crate::config::{CloneConfig, CloneLivenessRegistry};
use crate::flow_control::LatticeEdge;
use euhedral_hardware_utils::affinity::PinnedThreadExecutor;
use euhedral_hardware_utils::monitor::CoreSnapshot;
use std::sync::Arc;
use std::time::Instant;

/// Anything the control plane clones onto each core it places work on.
///
/// A [`ControlPlaneShard`](crate::control_plane::ControlPlaneShard) calls, in order:
/// [`clone_object`](Self::clone_object), [`input_edge`](Self::input_edge) to attach the clone below
/// its core's edge, [`set_drain_mode`](Self::set_drain_mode), [`start`](Self::start), then
/// [`update`](Self::update) with each new utilization of the core. [`input`](Self::input) and
/// [`output`](Self::output) wire clones to other sources. Every other method has a default suited
/// to an object without that concern.
pub trait CloneableObject: Send + Sync + 'static {
    /// A new instance placed by `config`.
    fn clone_object(&self, config: &CloneConfig) -> Arc<dyn CloneableObject>;

    /// Like [`clone_object`](Self::clone_object), for a clone that runs on `executor`. The default
    /// ignores the executor.
    fn clone_object_on(
        &self,
        config: &CloneConfig,
        executor: &PinnedThreadExecutor,
    ) -> Arc<dyn CloneableObject> {
        let _ = executor;
        self.clone_object(config)
    }

    fn start(&self) {}

    /// Called once before [`start`](Self::start) on the clone's own CPU. Objects that allocate
    /// should fill their queues and touch their state here, then reset it, so the memory lands on
    /// the NUMA node of the CPU that uses it.
    fn first_touch(&self) {}

    fn is_started(&self) -> bool {
        true
    }

    fn ready(&self) -> bool {
        true
    }

    /// The latest utilization of the clone's core.
    fn update(&self, snapshot: &CoreSnapshot) {
        let _ = snapshot;
    }

    /// Attaches `stream` as a source of frames for this clone.
    fn input(self: Arc<Self>, stream: Arc<dyn LatticeSource>) {
        let _ = stream;
    }

    /// Attaches this clone below `edge`, the edge its core's frames arrive through: typically with
    /// [`LatticeEdge::add_downstream`] or [`LatticeEdge::add_downstream_node`]. The default ignores
    /// it.
    fn input_edge(self: Arc<Self>, edge: Arc<LatticeEdge>) {
        let _ = edge;
    }

    /// The source this clone emits frames through, if any.
    fn output(&self) -> Option<Arc<dyn LatticeSource>> {
        None
    }

    fn is_drained(&self) -> bool {
        true
    }

    fn set_drain_mode(&self, drain: bool) {
        let _ = drain;
    }

    /// Clears trial-specific buffered work and controller state before another policy is
    /// activated, and reports roughly how many buffered frames it dropped. An object that owns
    /// single-consumer queues must reset them on their owning thread and acknowledge before
    /// `deadline`.
    fn reset(&self, deadline: Instant) -> u64 {
        let _ = deadline;
        0
    }

    fn dump_locks(&self) {}

    /// The physical core this clone is bound to.
    fn core(&self) -> Option<usize> {
        None
    }

    /// Where this clone's workers publish their liveness.
    fn liveness_registry(&self) -> Option<Arc<CloneLivenessRegistry>> {
        None
    }

    fn close(&self) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use euhedral_hardware_utils::affinity;
    use euhedral_hardware_utils::util::CpuSet;
    use std::sync::Mutex;

    /// Implements only what a clone must, and logs every placement it is cloned to.
    #[derive(Default)]
    struct Minimal {
        placements: Arc<Mutex<Vec<CloneConfig>>>,
    }

    impl CloneableObject for Minimal {
        fn clone_object(&self, config: &CloneConfig) -> Arc<dyn CloneableObject> {
            self.placements.lock().unwrap().push(config.clone());
            Arc::new(Minimal {
                placements: self.placements.clone(),
            })
        }
    }

    #[test]
    fn the_defaults_describe_an_object_without_lifecycle_concerns() {
        let object: Arc<dyn CloneableObject> = Arc::new(Minimal::default());
        object.first_touch();
        object.start();
        assert!(object.is_started());
        assert!(object.ready());
        assert!(object.output().is_none());
        assert!(object.is_drained());
        object.set_drain_mode(true);
        assert_eq!(object.reset(Instant::now()), 0);
        object.dump_locks();
        assert_eq!(object.core(), None);
        assert!(object.liveness_registry().is_none());
        object.close();
    }

    #[test]
    fn clones_are_new_objects_placed_by_their_config() {
        let config = CloneConfig::new("shard", 2, CpuSet::from_iter([4, 5]));
        let original: Arc<dyn CloneableObject> = Arc::new(Minimal::default());
        let placements = Arc::new(Mutex::new(Vec::new()));
        let original_with_log = Minimal {
            placements: placements.clone(),
        };
        let clone = original_with_log.clone_object(&config);
        assert!(!Arc::ptr_eq(&original, &clone));
        assert_eq!(*placements.lock().unwrap(), [config]);
    }

    #[test]
    fn clone_object_on_defaults_to_clone_object() {
        let Some(cpu) = affinity::base_mask().first() else {
            return;
        };
        let executor = PinnedThreadExecutor::get_or_create(cpu, "clone-test").unwrap();
        let config = CloneConfig::new("shard", 0, CpuSet::from_iter([cpu]));
        let original = Minimal::default();
        original.clone_object_on(&config, &executor);
        assert_eq!(*original.placements.lock().unwrap(), [config]);
    }
}
