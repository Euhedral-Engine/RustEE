//! A cloneable object that records what the control plane does to it.

use crate::config::CloneConfig;
use crate::flow_control::LatticeEdge;
use crate::flow_control::testing::TestReceiver;
use crate::generics::CloneableObject;
use euhedral_hardware_utils::monitor::CoreSnapshot;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

pub(crate) type Hook = Box<dyn FnOnce() + Send>;

/// Records what the control plane does to it. The template's clones share its `spawned` list and copy its
/// knobs at the time they are cloned.
pub(crate) struct TestClone {
    pub(crate) config: Option<CloneConfig>,
    pub(crate) spawned: Arc<Mutex<Vec<Arc<TestClone>>>>,
    pub(crate) on_clone: Arc<Mutex<Option<Hook>>>,
    pub(crate) receiver: Arc<TestReceiver>,
    pub(crate) edge: Mutex<Option<Arc<LatticeEdge>>>,
    pub(crate) starts: AtomicUsize,
    pub(crate) started: AtomicBool,
    pub(crate) closed: AtomicBool,
    pub(crate) drain_modes: Mutex<Vec<bool>>,
    pub(crate) updates: Mutex<Vec<usize>>,
    pub(crate) quotas: Mutex<Vec<f64>>,
    pub(crate) drained: AtomicBool,
    pub(crate) ready: AtomicBool,
    pub(crate) buffered: AtomicU64,
    pub(crate) panic_on_reset: AtomicBool,
    pub(crate) panic_on_dump: AtomicBool,
}

impl TestClone {
    pub(crate) fn template() -> Arc<Self> {
        Arc::new(Self::new(None, Default::default(), Default::default()))
    }

    fn new(
        config: Option<CloneConfig>,
        spawned: Arc<Mutex<Vec<Arc<TestClone>>>>,
        on_clone: Arc<Mutex<Option<Hook>>>,
    ) -> Self {
        Self {
            config,
            spawned,
            on_clone,
            receiver: TestReceiver::new(),
            edge: Mutex::new(None),
            starts: AtomicUsize::new(0),
            started: AtomicBool::new(false),
            closed: AtomicBool::new(false),
            drain_modes: Mutex::new(Vec::new()),
            updates: Mutex::new(Vec::new()),
            quotas: Mutex::new(Vec::new()),
            drained: AtomicBool::new(true),
            ready: AtomicBool::new(true),
            buffered: AtomicU64::new(0),
            panic_on_reset: AtomicBool::new(false),
            panic_on_dump: AtomicBool::new(false),
        }
    }

    /// Every clone made from this template, in the order they were made.
    pub(crate) fn spawned(&self) -> Vec<Arc<TestClone>> {
        self.spawned.lock().unwrap().clone()
    }

    pub(crate) fn config(&self) -> &CloneConfig {
        self.config.as_ref().expect("a clone has a config")
    }

    pub(crate) fn drain_modes(&self) -> Vec<bool> {
        self.drain_modes.lock().unwrap().clone()
    }

    pub(crate) fn in_drain_mode(&self) -> bool {
        self.drain_modes().last().copied().unwrap_or(false)
    }

    pub(crate) fn updates(&self) -> Vec<usize> {
        self.updates.lock().unwrap().clone()
    }

    pub(crate) fn is_closed(&self) -> bool {
        self.closed.load(Ordering::SeqCst)
    }

    pub(crate) fn edge(&self) -> Arc<LatticeEdge> {
        self.edge
            .lock()
            .unwrap()
            .clone()
            .expect("the clone has an input edge")
    }
}

impl CloneableObject for TestClone {
    fn clone_object(&self, config: &CloneConfig) -> Arc<dyn CloneableObject> {
        if let Some(hook) = self.on_clone.lock().unwrap().take() {
            hook();
        }
        let clone = Arc::new(Self::new(
            Some(config.clone()),
            self.spawned.clone(),
            self.on_clone.clone(),
        ));
        clone
            .drained
            .store(self.drained.load(Ordering::SeqCst), Ordering::SeqCst);
        clone
            .ready
            .store(self.ready.load(Ordering::SeqCst), Ordering::SeqCst);
        self.spawned.lock().unwrap().push(clone.clone());
        clone
    }

    fn start(&self) {
        self.starts.fetch_add(1, Ordering::SeqCst);
        self.started.store(true, Ordering::SeqCst);
    }

    fn is_started(&self) -> bool {
        self.started.load(Ordering::SeqCst)
    }

    fn ready(&self) -> bool {
        self.ready.load(Ordering::SeqCst)
    }

    fn update(&self, snapshot: &CoreSnapshot) {
        self.updates.lock().unwrap().push(snapshot.core_id);
        self.quotas.lock().unwrap().push(snapshot.quota_cpus);
    }

    fn input_edge(self: Arc<Self>, edge: Arc<LatticeEdge>) {
        edge.add_downstream(self.receiver.clone());
        *self.edge.lock().unwrap() = Some(edge);
    }

    fn is_drained(&self) -> bool {
        self.drained.load(Ordering::SeqCst)
    }

    fn set_drain_mode(&self, drain: bool) {
        self.drain_modes.lock().unwrap().push(drain);
    }

    fn reset(&self, _deadline: Instant) -> u64 {
        assert!(!self.panic_on_reset.load(Ordering::SeqCst), "reset failed");
        self.buffered.swap(0, Ordering::SeqCst)
    }

    fn dump_locks(&self) {
        assert!(!self.panic_on_dump.load(Ordering::SeqCst), "dump failed");
    }

    fn core(&self) -> Option<usize> {
        self.config.as_ref().map(CloneConfig::core_id)
    }

    fn close(&self) {
        self.closed.store(true, Ordering::SeqCst);
    }
}
