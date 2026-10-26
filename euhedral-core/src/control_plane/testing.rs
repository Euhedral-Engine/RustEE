//! A cloneable object that records what the control plane does to it, and helpers for tests that
//! run workers on the host's CPUs.

use crate::config::CloneConfig;
use crate::flow_control::LatticeEdge;
use crate::flow_control::testing::TestReceiver;
use crate::generics::CloneableObject;
use euhedral_hardware_utils::affinity;
use euhedral_hardware_utils::monitor::CoreSnapshot;
use euhedral_hardware_utils::topology::Topology;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock, Mutex, MutexGuard, PoisonError};
use std::thread;
use std::time::{Duration, Instant};

/// How long tests wait for workers.
pub(crate) const PATIENCE: Duration = Duration::from_secs(10);

/// Tests that start workers pin them to the same few host CPUs and assert on those CPUs' executors
/// (that they are open, closed or gone), so they take turns.
pub(crate) fn pinned_turn() -> MutexGuard<'static, ()> {
    static PINNED: Mutex<()> = Mutex::new(());
    PINNED.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Waits for `condition`, failing the test after [`PATIENCE`].
pub(crate) fn wait_until(what: &str, mut condition: impl FnMut() -> bool) {
    let deadline = Instant::now() + PATIENCE;
    while !condition() {
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        thread::sleep(Duration::from_millis(1));
    }
}

/// The host's last core with at least two CPUs this process may run on, and those CPUs.
pub(crate) static SMT_CORE: LazyLock<Option<(usize, Vec<usize>)>> = LazyLock::new(|| {
    let allowed = affinity::base_mask();
    Topology::system()
        .core_infos()
        .iter()
        .rev()
        .map(|core| (core.core, core.cpus.iter().collect::<Vec<_>>()))
        .find(|(_, cpus)| cpus.len() >= 2 && cpus.iter().all(|&cpu| allowed.contains(cpu)))
});

/// The host's last core this process may run on and its first CPU.
pub(crate) static SINGLE_CORE: LazyLock<Option<(usize, usize)>> = LazyLock::new(|| {
    let allowed = affinity::base_mask();
    Topology::system()
        .core_infos()
        .iter()
        .rev()
        .find_map(|core| {
            let cpu = core.cpus.iter().find(|&cpu| allowed.contains(cpu))?;
            Some((core.core, cpu))
        })
});

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
