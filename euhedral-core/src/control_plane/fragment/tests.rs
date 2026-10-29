use super::*;
use crate::config::CacheConfig;
use crate::control_plane::testing::{PATIENCE, SINGLE_CORE, SMT_CORE, pinned_turn, wait_until};
use crate::flow_control::testing::{TestReceiver, frame};
use crate::flow_control::{Claim, LatticeNode, UpstreamHandle};
use crate::frames::Failure;
use crate::generics::{LatticeReceiver, LatticeTerminal};
use crate::metrics::CAP_FACTOR_SUFFIX;
use crate::utils::StopWatch;
use euhedral_data_structures::sync::HotScope;
use euhedral_hardware_utils::monitor::{HardwareUtilization, HardwareUtilizationParts};
use euhedral_hardware_utils::topology::Topology;
use euhedral_hardware_utils::topology::source::{CoreKind, LogicalCpu, TopologyInput, normalize};
use euhedral_hardware_utils::util::CpuSet;
use std::collections::VecDeque;
use std::sync::Condvar;
use std::sync::mpsc;
use std::sync::{Mutex, PoisonError};

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

fn directory() -> Arc<CacheDirectory> {
    CacheDirectory::new(UpstreamRegistry::new(), Topology::system().clone())
}

/// A fragment under test, closed with its workers gone when dropped.
struct Fixture {
    directory: Arc<CacheDirectory>,
    edge: Arc<LatticeEdge>,
    fragment: Arc<ControlPlaneFragment>,
    receiver: Arc<Gate>,
    buddy_receiver: Arc<Gate>,
    _serial: MutexGuard<'static, ()>,
}

impl Fixture {
    /// A fragment of `config` on `cpus` of `core`, below an edge, its outputs into gates.
    fn new(config: &FragmentConfig, core: usize, cpus: &[usize]) -> Self {
        let serial = pinned_turn();
        let directory = directory();
        let clone = CloneConfig::new("test", core, cpus.iter().copied().collect())
            .with_cache_directory(directory.clone());
        let fragment = ControlPlaneFragment::new(config, &clone).unwrap();
        let edge = LatticeEdge::new(directory.registry(), Default::default());
        fragment.clone().input_edge(edge.clone());
        let receiver = Gate::new();
        fragment.output().unwrap().add_downstream(receiver.clone());
        let buddy_receiver = Gate::new();
        if let Some(buddy) = fragment.smt_buddy() {
            buddy
                .output()
                .unwrap()
                .add_downstream(buddy_receiver.clone());
        }
        Self {
            directory,
            edge,
            fragment,
            receiver,
            buddy_receiver,
            _serial: serial,
        }
    }

    /// A fragment with an SMT buddy, or `None` on a host without two CPUs on a core.
    fn smt() -> Option<Self> {
        let (core, cpus) = SMT_CORE.as_ref()?;
        Some(Self::new(&FragmentConfig::default(), *core, &cpus[..2]))
    }

    /// A fragment on one CPU, or `None` on a host this process cannot pin to.
    fn single() -> Option<Self> {
        let (core, cpu) = (*SINGLE_CORE)?;
        Some(Self::new(&FragmentConfig::default(), core, &[cpu]))
    }

    fn started(self) -> Self {
        self.fragment.start();
        wait_until("the workers to get ready", || self.fragment.ready());
        self
    }

    fn registry(&self) -> &Arc<UpstreamRegistry> {
        self.directory.registry()
    }

    /// Every frame the outputs received.
    fn received(&self) -> Vec<u64> {
        let mut received = self.receiver.received();
        received.extend(self.buddy_receiver.received());
        received.sort_unstable();
        received
    }

    /// A snapshot of the fragment's core at `timestamp_ns` with `pressure` on every CPU.
    fn snapshot(&self, timestamp_ns: i64, pressure: f64) -> CoreSnapshot {
        let topology = Topology::system();
        let utilization = HardwareUtilization::new(HardwareUtilizationParts {
            timestamp_ns,
            quota_cpus: topology.cpu_count() as f64,
            global_effective_cpus: topology.cpu_set().clone(),
            per_quota_cpu_pressure: vec![pressure; topology.cpu_span()],
            ..Default::default()
        });
        let core = self.fragment.core;
        let cpus = &topology.core_info(core).unwrap().cpus;
        utilization.core_snapshot(core, cpus, cpus.len() as f64)
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        self.receiver.open();
        self.buddy_receiver.open();
        self.fragment.close();
        let deadline = Instant::now() + PATIENCE;
        let stopped = self.fragment.liveness.await_termination(deadline);
        let terminated = self.fragment.await_executors(PATIENCE);
        if !thread::panicking() {
            assert!(stopped && terminated, "the workers did not stop");
        }
    }
}

/// Records what reaches it, holding the pusher at a frame it was told to stop at until opened.
#[derive(Default)]
struct Gate {
    received: Mutex<Vec<u64>>,
    stop_at: Mutex<Option<u64>>,
    holding: Condvar,
    held: AtomicBool,
    /// The frames pushed while the pushing thread was inside a hot scope.
    pushed_hot: AtomicU64,
}

impl Gate {
    fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    fn received(&self) -> Vec<u64> {
        lock(&self.received).clone()
    }

    fn stop_at(&self, id: u64) {
        *lock(&self.stop_at) = Some(id);
    }

    fn open(&self) {
        *lock(&self.stop_at) = None;
        self.holding.notify_all();
    }

    fn is_holding(&self) -> bool {
        self.held.load(Ordering::Acquire)
    }
}

impl LatticeTerminal for Gate {
    fn add_upstream(&self, _upstream: Arc<dyn LatticeSource>) {}
}

impl LatticeReceiver for Gate {
    fn push(&self, frame: Box<dyn Frame>) {
        let id = frame.header().id_hash();
        if HotScope::active() {
            self.pushed_hot.fetch_add(1, Ordering::AcqRel);
        }
        let mut stop_at = lock(&self.stop_at);
        while *stop_at == Some(id) {
            self.held.store(true, Ordering::Release);
            stop_at = self
                .holding
                .wait(stop_at)
                .unwrap_or_else(PoisonError::into_inner);
        }
        self.held.store(false, Ordering::Release);
        drop(stop_at);
        lock(&self.received).push(id);
    }

    fn on_complete(&self) {}

    fn on_error(&self, _error: Failure) {}
}

/// Hands out queued frames to whichever worker pulls.
#[derive(Default)]
struct Source {
    frames: Mutex<VecDeque<Box<dyn Frame>>>,
    requested: AtomicU64,
}

impl Source {
    fn with(frames: impl IntoIterator<Item = Box<dyn Frame>>) -> Arc<Self> {
        Arc::new(Self {
            frames: Mutex::new(frames.into_iter().collect()),
            ..Default::default()
        })
    }
}

impl UpstreamHandle for Source {
    fn sequence(&self) -> u64 {
        0
    }

    fn pull(
        &self,
        _claim: &mut Claim,
        consumer: &mut dyn FnMut(Box<dyn Frame>),
        stop: &mut dyn FnMut(&dyn Frame) -> bool,
        demand: u64,
    ) -> Option<u64> {
        let mut pulled = 0;
        while pulled < demand {
            let next = {
                let mut frames = lock(&self.frames);
                match frames.front() {
                    Some(frame) if !stop(&**frame) => frames.pop_front(),
                    _ => None,
                }
            };
            let Some(frame) = next else {
                break;
            };
            consumer(frame);
            pulled += 1;
        }
        Some(pulled)
    }

    fn request(&self, _claim: &mut Claim, demand: u64) {
        self.requested.fetch_add(demand, Ordering::AcqRel);
    }

    fn is_complete(&self) -> bool {
        false
    }
}

/// An unordered frame with routing hash `hash`.
fn unordered(id: u64, hash: u64) -> Box<dyn Frame> {
    let mut frame = frame(id);
    frame.header_mut().set_routing_hash(hash);
    frame
}

/// A topology of one CPU, `cpu`, that this host does not have.
fn foreign_topology(cpu: usize) -> Topology {
    normalize(TopologyInput {
        provider: "linux".to_string(),
        cpus: vec![LogicalCpu {
            id: cpu,
            socket_key: "linux:package:0000000000".to_string(),
            die_key: "linux:die:0000000000".to_string(),
            core_key: "linux:core:0000000000".to_string(),
            kind: CoreKind::Unknown,
        }],
        caches: Vec::new(),
    })
    .unwrap()
}

#[test]
fn a_fragment_needs_a_lattice_a_cpu_and_an_executor() {
    let config = FragmentConfig::default();
    let new = |clone: &CloneConfig| ControlPlaneFragment::new(&config, clone).err();

    let unplaced = CloneConfig::new("test", 0, CpuSet::from_iter([0]));
    assert_eq!(new(&unplaced), Some(FragmentError::NoCacheDirectory));
    let placed = |cpus: &[usize], directory: &Arc<CacheDirectory>| {
        CloneConfig::new("test", 0, cpus.iter().copied().collect())
            .with_cache_directory(directory.clone())
    };
    assert_eq!(new(&placed(&[], &directory())), Some(FragmentError::NoCpu));
    let absent = Topology::system().cpu_span() + 1000;
    assert_eq!(
        new(&placed(&[absent], &directory())),
        Some(FragmentError::UnknownCpu(absent))
    );
    let foreign = CacheDirectory::new(UpstreamRegistry::new(), foreign_topology(absent));
    assert!(matches!(
        new(&placed(&[absent], &foreign)),
        Some(FragmentError::Executor(_))
    ));

    assert_eq!(
        FragmentError::NoCacheDirectory.to_string(),
        "the clone config names no cache directory"
    );
    assert_eq!(
        FragmentError::NoCpu.to_string(),
        "the clone config has no effective CPU"
    );
    assert_eq!(
        FragmentError::UnknownCpu(7).to_string(),
        "CPU 7 is not in the lattice's topology"
    );
    let error = new(&placed(&[absent], &foreign)).unwrap().to_string();
    assert!(error.starts_with("no pinned executor: "), "{error}");
}

#[test]
#[should_panic(expected = "cannot build the fragment of core 3: the clone config names no cache")]
fn cloning_the_config_without_a_lattice_panics() {
    FragmentConfig::default().clone_object(&CloneConfig::new("test", 3, CpuSet::from_iter([0])));
}

#[test]
fn a_fragment_is_placed_by_its_clone() {
    let Some(fixture) = Fixture::single() else {
        return;
    };
    let fragment = &fixture.fragment;
    let (core, cpu) = SINGLE_CORE.unwrap();
    let topology = Topology::system();
    assert_eq!(fragment.name(), format!("test-Worker-{core}"));
    assert_eq!(fragment.cpu(), cpu);
    assert_eq!(CloneableObject::core(&**fragment), Some(core));
    assert_eq!(fragment.socket(), topology.cpu_info(cpu).unwrap().socket);
    assert_eq!(
        fragment.is_p_core(),
        topology.core_info(core).unwrap().p_core
    );
    assert!(fragment.smt_buddy().is_none());
    assert!(fragment.cache().sibling().is_none());
    assert_eq!(fragment.cache().cpu(), cpu);
    assert_eq!(fragment.config().max_batch_size().get(), 4096);
    assert_eq!(
        fragment.batch_cap(),
        fragment.cache().frame_quota().clamp(2, 4096)
    );
    assert_eq!(
        fragment.liveness_registry().unwrap().cpus(),
        &CpuSet::from_iter([cpu])
    );
    assert!(fragment.cache().edge().parent().is_some());
    assert!(!fragment.is_started());
    assert!(!fragment.ready());
    assert!(fragment.is_drained());
}

#[test]
fn smt_gives_the_fragment_a_buddy_on_the_second_cpu() {
    let Some(fixture) = Fixture::smt() else {
        return;
    };
    let fragment = &fixture.fragment;
    let (_, cpus) = SMT_CORE.as_ref().unwrap();
    let buddy = fragment.smt_buddy().unwrap();
    assert_eq!(fragment.cpu(), cpus[0]);
    assert_eq!(buddy.cpu(), cpus[1]);
    assert!(buddy.smt_buddy().is_none());
    assert!(Arc::ptr_eq(
        fragment.cache().sibling().unwrap(),
        buddy.cache()
    ));
    assert!(buddy.cache().edge().parent().is_none());
    assert!(!Arc::ptr_eq(
        &fragment.output().unwrap(),
        &buddy.output().unwrap()
    ));
}

#[test]
fn without_smt_a_two_cpu_core_runs_one_worker() {
    let Some((core, cpus)) = SMT_CORE.as_ref() else {
        return;
    };
    let config = FragmentConfig::default().with_smt_enabled(false);
    let fixture = Fixture::new(&config, *core, &cpus[..2]);
    assert!(fixture.fragment.smt_buddy().is_none());
    assert_eq!(fixture.fragment.cpu(), cpus[0]);
}

#[test]
fn started_workers_register_and_run_the_cores_frames() {
    let Some(fixture) = Fixture::smt() else {
        return;
    };
    let fixture = fixture.started();
    let fragment = &fixture.fragment;
    let buddy = fragment.smt_buddy().unwrap();
    assert!(fragment.is_started());
    assert_eq!(fixture.registry().thread_count(), 2);
    assert!(
        fragment
            .liveness
            .workers()
            .iter()
            .all(|worker| worker.running)
    );

    for id in 0..200 {
        fixture.edge.push_downstream(unordered(id, id * 7 + 1000));
    }

    wait_until("every frame to run", || fixture.received().len() == 200);
    assert_eq!(fixture.received(), (0..200).collect::<Vec<_>>());
    // The odd lane ran on the buddy.
    let buddy_ran = fixture.buddy_receiver.received();
    assert!(!buddy_ran.is_empty());
    assert!(buddy_ran.iter().all(|id| (id * 7 + 1000) % 2 == 1));
    wait_until("the fragment to drain", || fragment.is_drained());
    assert!(buddy.is_drained());
}

#[cfg(debug_assertions)]
#[test]
fn a_worker_cycle_runs_inside_a_hot_scope() {
    let Some(fixture) = Fixture::single() else {
        return;
    };
    let fixture = fixture.started();
    for id in 0..50 {
        fixture.edge.push_downstream(unordered(id, id + 1));
    }
    wait_until("every frame to run", || {
        fixture.receiver.received().len() == 50
    });
    // The output receiver is reached on the worker, from inside the cycle.
    assert_eq!(fixture.receiver.pushed_hot.load(Ordering::Acquire), 50);
    // This thread is no worker.
    assert!(!HotScope::active());
}

#[test]
fn workers_run_frames_from_their_upstreams() {
    let Some(fixture) = Fixture::single() else {
        return;
    };
    let fixture = fixture.started();
    let source = Source::with((0..500).map(|id| unordered(id, id + 1)));
    fixture.registry().add_handle(source.clone());

    wait_until("every upstream frame to run", || {
        fixture.receiver.received().len() == 500
    });
    // With nothing run, a direct cycle asks its upstreams for a batch.
    assert!(source.requested.load(Ordering::Acquire) > 0);
    let mut received = fixture.receiver.received();
    received.sort_unstable();
    assert_eq!(received, (0..500).collect::<Vec<_>>());
}

#[test]
fn closing_stops_the_workers_and_retires_the_caches() {
    let Some(fixture) = Fixture::smt() else {
        return;
    };
    let fixture = fixture.started();
    let fragment = &fixture.fragment;
    let buddy = fragment.smt_buddy().unwrap().clone();

    fragment.close();

    assert!(
        fragment
            .liveness
            .await_termination(Instant::now() + PATIENCE)
    );
    assert!(!fragment.is_started() && !buddy.is_started());
    assert!(!fragment.ready());
    assert!(fragment.cache().is_closed() && buddy.cache().is_closed());
    assert!(fragment.executor.is_closed() && buddy.executor.is_closed());
    // A worker leaves the registry before it reports that it stopped.
    assert_eq!(fixture.registry().thread_count(), 0);
    let restarted = panic::catch_unwind(AssertUnwindSafe(|| fragment.start()));
    let message = restarted.unwrap_err();
    let message = message.downcast_ref::<String>().unwrap();
    assert!(
        message.contains("cannot start a closed fragment"),
        "{message}"
    );
}

#[test]
fn starting_twice_runs_one_worker() {
    let Some(fixture) = Fixture::single() else {
        return;
    };
    let fixture = fixture.started();
    fixture.fragment.start();
    thread::sleep(Duration::from_millis(20));
    assert_eq!(fixture.registry().thread_count(), 1);
}

#[test]
fn dumping_locks_interrupts_the_workers() {
    let Some(fixture) = Fixture::smt() else {
        return;
    };
    let fixture = fixture.started();
    let fragment = &fixture.fragment;
    fragment.dump_locks();
    assert!(
        fragment
            .liveness
            .await_termination(Instant::now() + PATIENCE)
    );
    assert!(!fragment.is_started());
    assert!(fragment.smt_buddy().unwrap().executor.is_closed());
}

/// A started fragment of its own on `core`'s `cpu`, below its own edge, its output into a gate.
fn started_on(core: usize, cpu: usize) -> (Arc<ControlPlaneFragment>, Arc<LatticeEdge>, Arc<Gate>) {
    let directory = directory();
    let clone = CloneConfig::new("test", core, CpuSet::from_iter([cpu]))
        .with_cache_directory(directory.clone());
    let fragment = ControlPlaneFragment::new(&FragmentConfig::default(), &clone).unwrap();
    let edge = LatticeEdge::new(directory.registry(), Default::default());
    fragment.clone().input_edge(edge.clone());
    let gate = Gate::new();
    fragment.output().unwrap().add_downstream(gate.clone());
    fragment.start();
    wait_until("the worker to get ready", || fragment.ready());
    (fragment, edge, gate)
}

/// Pushes `count` frames from `first` into `edge` and waits for `gate` to receive them all.
fn run_frames(edge: &LatticeEdge, gate: &Gate, first: u64, count: u64) {
    for id in first..first + count {
        edge.push_downstream(unordered(id, id + 1));
    }
    wait_until("the frames to run", || {
        let received = gate.received();
        (first..first + count).all(|id| received.contains(&id))
    });
}

#[test]
fn replacement_clone_on_same_core_keeps_running() {
    let Some((core, cpu)) = *SINGLE_CORE else {
        return;
    };
    let _serial = pinned_turn();
    // A shard that cannot drain a clone in time starts a replacement on the same core while the old
    // clone's worker is still on the CPU's executor, then closes the old clone.
    let (old, old_edge, old_gate) = started_on(core, cpu);
    run_frames(&old_edge, &old_gate, 0, 10);
    let (replacement, edge, gate) = started_on(core, cpu);

    old.close();
    old.dump_locks();
    assert!(old.liveness.await_termination(Instant::now() + PATIENCE));
    assert!(!old.is_started());

    assert!(replacement.is_started() && replacement.ready());
    assert!(!replacement.executor.is_closed());
    run_frames(&edge, &gate, 100, 50);
    assert!(replacement.is_started(), "the replacement's worker stays");

    replacement.close();
    assert!(
        replacement
            .liveness
            .await_termination(Instant::now() + PATIENCE)
    );
    // The replacement was the executor's last user.
    assert!(replacement.await_executors(PATIENCE));
    assert!(replacement.executor.is_closed());
}

#[test]
fn dumping_locks_interrupts_only_the_fragments_own_worker() {
    let Some((core, cpu)) = *SINGLE_CORE else {
        return;
    };
    let _serial = pinned_turn();
    let (first, _, _) = started_on(core, cpu);
    let (second, edge, gate) = started_on(core, cpu);

    first.dump_locks();
    assert!(first.liveness.await_termination(Instant::now() + PATIENCE));
    assert!(second.is_started() && !second.executor.is_closed());
    run_frames(&edge, &gate, 0, 20);

    first.close();
    second.close();
    assert!(second.liveness.await_termination(Instant::now() + PATIENCE));
    assert!(second.await_executors(PATIENCE));
}

#[test]
fn a_reset_before_start_clears_the_caches_and_the_cap() {
    let Some(fixture) = Fixture::smt() else {
        return;
    };
    let fragment = &fixture.fragment;
    for id in 0..10 {
        fixture.edge.push_downstream(unordered(id, id + 100));
    }
    assert_eq!(fragment.cache().local_cache_count(), 5);
    let full = fragment.batch_cap();
    fragment.update(&fixture.snapshot(10, 1.0));
    assert_eq!(fragment.batch_cap(), 2);

    assert_eq!(fragment.reset(Instant::now()), 10);

    assert!(fragment.is_drained());
    assert_eq!(fragment.batch_cap(), full);
    // The last snapshot is forgotten too: an older one is followed again.
    fragment.update(&fixture.snapshot(5, 1.0));
    assert_eq!(fragment.batch_cap(), 2);
}

#[test]
fn a_running_worker_resets_itself() {
    let Some(fixture) = Fixture::single() else {
        return;
    };
    let fixture = fixture.started();
    let fragment = fixture.fragment.clone();
    fixture.receiver.stop_at(0);
    fixture.edge.push_downstream(unordered(0, 2));
    wait_until("the worker to hold frame 0", || {
        fixture.receiver.is_holding()
    });
    assert!(!fragment.is_drained());
    for id in 1..50 {
        fixture.edge.push_downstream(unordered(id, 2));
    }

    let (cleared_tx, cleared) = mpsc::channel();
    let resetting = fragment.clone();
    let reset = thread::spawn(move || {
        cleared_tx
            .send(resetting.reset(Instant::now() + PATIENCE))
            .unwrap();
    });
    thread::sleep(Duration::from_millis(20));
    fixture.receiver.open();
    reset.join().unwrap();
    let cleared = cleared.recv().unwrap();

    wait_until("the fragment to drain", || fragment.is_drained());
    assert!(cleared > 0);
    assert_eq!(fixture.receiver.received().len() as u64 + cleared, 50);
}

#[test]
fn a_worker_that_cannot_reset_in_time_fails_the_reset() {
    let Some(fixture) = Fixture::single() else {
        return;
    };
    let fixture = fixture.started();
    fixture.receiver.stop_at(0);
    fixture.edge.push_downstream(unordered(0, 2));
    wait_until("the worker to hold frame 0", || {
        fixture.receiver.is_holding()
    });

    let deadline = Instant::now() + Duration::from_millis(20);
    let reset = panic::catch_unwind(AssertUnwindSafe(|| fixture.fragment.reset(deadline)));

    let message = reset.unwrap_err();
    let message = message.downcast_ref::<String>().unwrap();
    assert!(message.contains("timed out resetting"), "{message}");
}

#[test]
fn pressure_lowers_the_batch_cap_of_both_workers() {
    let Some(fixture) = Fixture::smt() else {
        return;
    };
    let fragment = &fixture.fragment;
    let buddy = fragment.smt_buddy().unwrap();
    let max = fragment.batch_cap();
    assert!(max > 2);

    fragment.update(&fixture.snapshot(10, 0.5));
    let half = (max as f64 - 0.5 * (max - 2) as f64).round() as u64;
    assert_eq!(fragment.batch_cap(), half);
    assert_eq!(buddy.batch_cap(), half);
    assert!(fragment.cache().cap_factor() < 1.0);

    // Stale snapshots are ignored, the cache's cap included.
    let factor = fragment.cache().cap_factor();
    fragment.update(&fixture.snapshot(10, 0.0));
    fragment.update(&fixture.snapshot(9, 0.0));
    assert_eq!(fragment.batch_cap(), half);
    assert_eq!(fragment.cache().cap_factor(), factor);

    fragment.update(&fixture.snapshot(11, 0.0));
    assert_eq!(fragment.batch_cap(), max);
    fragment.update(&fixture.snapshot(12, 1.0));
    assert_eq!(fragment.batch_cap(), 2);
}

#[test]
fn a_snapshot_of_another_core_changes_nothing() {
    let Some(fixture) = Fixture::single() else {
        return;
    };
    let fragment = &fixture.fragment;
    let max = fragment.batch_cap();
    let utilization = HardwareUtilization::new(HardwareUtilizationParts {
        timestamp_ns: 10,
        per_quota_cpu_pressure: vec![1.0; Topology::system().cpu_span()],
        ..Default::default()
    });
    fragment.update(&utilization.core_snapshot(0, &CpuSet::new(), 1.0));
    assert_eq!(fragment.batch_cap(), max);
    assert_eq!(fragment.cache().cap_factor(), 1.0);
}

#[test]
fn a_small_cache_or_batch_size_caps_the_batch() {
    let Some((core, cpu)) = *SINGLE_CORE else {
        return;
    };
    let config =
        FragmentConfig::default().with_max_batch_size(std::num::NonZeroU64::new(1).unwrap());
    let fixture = Fixture::new(&config, core, &[cpu]);
    assert_eq!(fixture.fragment.batch_cap(), 2);
    drop(fixture);

    let config =
        FragmentConfig::default().with_max_batch_size(std::num::NonZeroU64::new(u64::MAX).unwrap());
    let fixture = Fixture::new(&config, core, &[cpu]);
    assert_eq!(
        fixture.fragment.batch_cap(),
        fixture.fragment.cache().frame_quota()
    );
}

#[test]
fn drain_mode_and_first_touch_reach_both_workers() {
    let Some((core, cpus)) = SMT_CORE.as_ref() else {
        return;
    };
    let config = FragmentConfig::default()
        .with_cache_config(CacheConfig::default().with_max_pooled_chunks(8));
    let fixture = Fixture::new(&config, *core, &cpus[..2]);
    let fragment = &fixture.fragment;
    let buddy = fragment.smt_buddy().unwrap();

    fragment.set_drain_mode(true);
    assert!(fragment.cache().edge().drain_flag().load(Ordering::Acquire));
    assert!(buddy.cache().edge().drain_flag().load(Ordering::Acquire));
    fragment.set_drain_mode(false);
    assert!(!buddy.cache().edge().drain_flag().load(Ordering::Acquire));

    fragment.first_touch();
    assert!(fragment.cache().pooled_chunks() > 0);
    assert!(buddy.cache().pooled_chunks() > 0);
}

#[test]
fn the_policy_times_a_sample_of_bodies() {
    let view = Arc::new(BodyCostView::default());
    let mut cost = BodyCost::publishing_to(view.clone());
    let output = Arc::new(LatticeHotSource::new());
    let receiver = TestReceiver::new();
    output.clone().add_downstream(receiver.clone());
    // The first push of every interval is timed: the 32nd sample is push 31 · interval.
    let interval = StopWatch::DEFAULT_INTERVAL.get();
    for id in 0..31 * interval {
        cost.timed(|| output.accept(frame(id)));
    }
    assert!(!view.has_history());
    cost.timed(|| output.accept(frame(0)));
    assert!(view.has_history());
    assert_eq!(receiver.received().len() as u64, 31 * interval + 1);
}

#[test]
fn the_config_clones_into_fragments() {
    let Some((core, cpu)) = *SINGLE_CORE else {
        return;
    };
    let _serial = pinned_turn();
    let clone =
        CloneConfig::new("test", core, CpuSet::from_iter([cpu])).with_cache_directory(directory());
    let config = FragmentConfig::default().with_smt_enabled(false);
    let made = config.clone_object(&clone);
    assert_eq!(made.core(), Some(core));
    assert!(made.output().is_some());
    made.close();

    let fragment = ControlPlaneFragment::new(&config, &clone).unwrap();
    let again = fragment.clone_object(&clone);
    assert_eq!(again.core(), Some(core));
    again.close();
    fragment.close();
}

#[test]
fn frames_on_the_buddys_lane_keep_the_pair_undrained() {
    let Some(fixture) = Fixture::smt() else {
        return;
    };
    fixture.edge.push_downstream(unordered(1, 3));
    assert!(fixture.fragment.cache().is_drained());
    assert!(!fixture.fragment.is_drained());
}

#[test]
fn a_stopped_buddy_stops_the_pair() {
    let Some(fixture) = Fixture::smt() else {
        return;
    };
    let fixture = fixture.started();
    let buddy = fixture.fragment.smt_buddy().unwrap();
    buddy.close();
    assert!(!fixture.fragment.is_started());
    assert!(!fixture.fragment.ready());
}

#[test]
fn a_frame_in_a_workers_hands_keeps_it_undrained_and_closing_does_not_wait() {
    let Some(fixture) = Fixture::single() else {
        return;
    };
    let fixture = fixture.started();
    let fragment = &fixture.fragment;
    fixture.receiver.stop_at(0);
    fixture
        .registry()
        .add_handle(Source::with([unordered(0, 2)]));
    wait_until("the worker to hold frame 0", || {
        fixture.receiver.is_holding()
    });
    assert!(fragment.cache().is_drained());
    assert!(!fragment.is_drained());

    fragment.close();
    assert!(!fragment.is_started());
    fixture.receiver.open();
    wait_until("the fragment to drain", || fragment.is_drained());
}

#[test]
fn an_idle_worker_steals_from_the_other_caches() {
    let Some(fixture) = Fixture::single() else {
        return;
    };
    let fixture = fixture.started();
    let span = fixture.directory.topology().cpu_span();
    let slot = (fixture.fragment.cpu() + 3) % span;
    let victim = ControlPlaneCache::new(
        &CacheConfig::default(),
        &CloneConfig::new("victim", 0, CpuSet::from_iter([slot])),
        slot,
        false,
        &fixture.directory,
    );
    for id in 0..20 {
        victim.push(unordered(id, id + 100));
    }
    // An upstream keeps the worker cycling instead of parking.
    fixture.registry().add_handle(Source::with([]));

    wait_until("the worker to steal every frame", || victim.is_drained());
    wait_until("the stolen frames to run", || {
        fixture.receiver.received().len() == 20
    });
    victim.close();
}

#[test]
fn a_running_reset_restarts_the_policy() {
    let Some(fixture) = Fixture::single() else {
        return;
    };
    let fixture = fixture.started();
    let fragment = &fixture.fragment;
    let frames = 32 * StopWatch::DEFAULT_INTERVAL.get();
    fixture
        .registry()
        .add_handle(Source::with((0..frames).map(|id| unordered(id, id + 1))));
    wait_until("a body-cost history", || fragment.body_cost_ns().is_some());

    fragment.reset(Instant::now() + PATIENCE);

    assert!(!fragment.body_cost_ns().is_some());
}

#[test]
fn cycles_size_the_next_batch_at_the_end_of_each() {
    let mut policy = DecisionTree::with_thresholds(
        crate::config::IdlePolicy::default(),
        BodyCost::new(),
        Duration::from_micros(1),
        Duration::from_micros(1),
    );
    let mut cycle = Cycle::new(0, None);
    let one = Run {
        processed: 1,
        elapsed: Duration::from_nanos(100),
    };
    cycle.complete(&one, &mut policy, 4096, None, || unreachable!("mid-batch"));
    assert_eq!((cycle.completed, cycle.batch_size), (1, 2));
    cycle.complete(&one, &mut policy, 4096, None, || 3);
    assert_eq!((cycle.completed, cycle.batch_size), (0, 4));
    assert_eq!(cycle.registered_workers, 3);
    assert_eq!(policy.service_time_ns(), 100.0);
    let many = Run {
        processed: 4,
        elapsed: Duration::from_nanos(400),
    };
    cycle.complete(&many, &mut policy, 6, None, || 5);
    assert_eq!((cycle.completed, cycle.batch_size), (0, 6));
    assert_eq!(cycle.registered_workers, 5);
}

fn metered(registry: &Arc<MetricsRegistry>) -> FragmentConfig {
    FragmentConfig::default().with_metrics("test", registry.clone())
}

#[test]
fn a_fragment_without_a_registry_has_no_meters() {
    let Some(fixture) = Fixture::single() else {
        return;
    };
    let fixture = fixture.started();
    assert!(fixture.fragment.metrics.is_none());
    assert!(fixture.fragment.metrics_registry().is_none());
    for id in 0..100 {
        fixture.edge.push_downstream(unordered(id, id + 1));
    }
    wait_until("every frame to run", || fixture.received().len() == 100);
    let mut meters = Vec::new();
    MeterSource::collect(&*fixture.fragment, &mut meters);
    MeterSource::collect(&**fixture.fragment.cache(), &mut meters);
    assert!(meters.is_empty());
}

#[test]
fn both_workers_and_their_caches_register_meters_tagged_with_core_and_cpu() {
    use crate::metrics::{
        CACHE_BACKLOG_SUFFIX, FRAMES_SUFFIX, IN_PROGRESS_SUFFIX, SERVICE_TIME_SUFFIX,
        THROUGHPUT_SUMMARY_SUFFIX, metric_name,
    };

    let Some((core, cpus)) = SMT_CORE.as_ref() else {
        return;
    };
    let registry = MetricsRegistry::new();
    let fixture = Fixture::new(&metered(&registry), *core, &cpus[..2]);
    let fragment = &fixture.fragment;
    assert!(Arc::ptr_eq(
        &fragment.metrics_registry().unwrap(),
        &registry
    ));
    // Two workers and two caches.
    assert_eq!(registry.len(), 4);

    let snapshot = registry.snapshot();
    for suffix in [
        SERVICE_TIME_SUFFIX,
        THROUGHPUT_SUMMARY_SUFFIX,
        IN_PROGRESS_SUFFIX,
        FRAMES_SUFFIX,
        CACHE_BACKLOG_SUFFIX,
        CAP_FACTOR_SUFFIX,
    ] {
        let name = metric_name("test", suffix);
        let mut tagged: Vec<_> = snapshot
            .named(&name)
            .map(|meter| (meter.core, meter.cpu))
            .collect();
        tagged.sort_unstable();
        assert_eq!(tagged, [(*core, cpus[0]), (*core, cpus[1])], "{name}");
    }
    // Six meters per worker and two per cache.
    assert_eq!(snapshot.meters().len(), 16);
    // Flows are unmeasured before the workers run.
    assert!(
        snapshot
            .service_time_flows()
            .all(|meter| meter.flow().is_none())
    );
    let frames = snapshot.get("test.execution.frames", cpus[0]).unwrap();
    assert_eq!(frames.counter(), Some(0));
    assert!(
        snapshot
            .in_progress_gauges()
            .all(|meter| meter.gauge() == Some(0.0))
    );

    // Frames queued before the workers start show as the backlog of their lanes.
    for id in 0..10 {
        fixture.edge.push_downstream(unordered(id, id));
    }
    fragment.update(&fixture.snapshot(10, 0.5));
    let snapshot = registry.snapshot();
    let backlog: f64 = snapshot
        .cache_backlog_gauges()
        .map(|meter| meter.gauge().unwrap())
        .sum();
    assert_eq!(backlog, 10.0);
    assert!(
        snapshot
            .cap_factor_summaries()
            .all(|meter| meter.summary().unwrap().count == 1)
    );

    fragment.close();
    assert!(registry.is_empty());
}

#[test]
fn running_workers_record_frames_and_throughput_while_snapshots_are_taken() {
    let Some((core, cpu)) = *SINGLE_CORE else {
        return;
    };
    let registry = MetricsRegistry::new();
    let fixture = Fixture::new(&metered(&registry), core, &[cpu]).started();

    let stop = AtomicBool::new(false);
    let rounds = 40u64;
    let per_round = 250u64;
    let snapshots = thread::scope(|scope| {
        let reader = scope.spawn(|| {
            // Counts only grow, read while the worker records.
            let mut last = (0u64, 0u64);
            let mut taken = 0u64;
            while !stop.load(Ordering::Acquire) {
                let snapshot = registry.snapshot();
                let read = |name| snapshot.get(name, cpu).unwrap();
                let frames = read("test.execution.frames").counter().unwrap();
                let throughput = read("test.execution.throughput").summary().unwrap().count;
                assert!(frames >= last.0 && throughput >= last.1);
                last = (frames, throughput);
                taken += 1;
                thread::yield_now();
            }
            taken
        });
        for round in 0..rounds {
            for id in round * per_round..(round + 1) * per_round {
                fixture.edge.push_downstream(unordered(id, id * 2 + 1));
            }
            thread::sleep(Duration::from_micros(500));
        }
        wait_until("every frame to run", || {
            fixture.received().len() as u64 == rounds * per_round
        });
        wait_until("a throughput sample", || {
            registry
                .snapshot()
                .get("test.execution.throughput", cpu)
                .and_then(|meter| meter.summary())
                .is_some_and(|summary| summary.count > 0)
        });
        stop.store(true, Ordering::Release);
        reader.join().unwrap()
    });
    assert!(snapshots > 0);

    let snapshot = registry.snapshot();
    let frames = snapshot.get("test.execution.frames", cpu).unwrap();
    assert_eq!(frames.counter(), Some(rounds * per_round));
    // These outputs feed no executor, so no `execute` is timed.
    let service_time = snapshot
        .get("test.execution.serviceTime", cpu)
        .unwrap()
        .summary()
        .unwrap();
    assert_eq!(service_time.count, 0);
    let throughput = snapshot
        .get("test.execution.throughput", cpu)
        .unwrap()
        .summary()
        .unwrap();
    // Frames per second.
    assert!(throughput.mean() > 1.0, "{throughput:?}");
    assert!(
        snapshot
            .get("test.execution.serviceTime.flow", cpu)
            .unwrap()
            .flow()
            .is_none()
    );
    let rate = snapshot
        .get("test.execution.throughput.flow", cpu)
        .and_then(|meter| meter.flow())
        .unwrap();
    assert!(rate.average > 0.0, "{rate:?}");
    wait_until("the worker to drain", || fixture.fragment.is_drained());
    let idle = registry.snapshot();
    assert_eq!(
        idle.get("test.execution.inProgress", cpu).unwrap().gauge(),
        Some(0.0)
    );
}

#[test]
fn cycles_count_frames_and_report_the_flow_at_the_end_of_each_batch() {
    let registry = MetricsRegistry::new();
    let config = crate::metrics::MetricsConfig::default().with_registry(Some(registry));
    let metrics = ExecutionMetrics::new(&config, 1, 2).unwrap();
    let mut policy = DecisionTree::with_thresholds(
        crate::config::IdlePolicy::default(),
        BodyCost::new(),
        Duration::from_micros(1),
        Duration::from_micros(1),
    );
    let mut cycle = Cycle::new(0, Some(&metrics));
    let run = Run {
        processed: 1,
        elapsed: Duration::from_nanos(300),
    };
    let summaries = |metrics: &ExecutionMetrics| {
        let mut meters = Vec::new();
        metrics.collect(0, &mut meters);
        let count = |name: &str| {
            meters
                .iter()
                .find(|meter| meter.name.ends_with(name))
                .unwrap()
                .clone()
        };
        (
            count(".execution.frames").counter().unwrap(),
            count(".execution.serviceTime").summary().unwrap().clone(),
            count(".execution.throughput").summary().unwrap().clone(),
        )
    };
    let start = Instant::now();
    for step in 1..=12u32 {
        // The executor times each frame's execute as the worker pushes it.
        metrics.record_service_time(Duration::from_nanos(200));
        cycle.now = start + Duration::from_micros(u64::from(step) * 10);
        cycle.complete(&run, &mut policy, 4096, Some(&metrics), || 1);
    }
    let (frames, service_time, throughput) = summaries(&metrics);
    assert_eq!(frames, 12);
    // Every execute, not the cycle's elapsed time.
    assert_eq!((service_time.count, service_time.sum), (12, 2400.0));
    assert_eq!(service_time.max, 200.0);
    // Batches of 2, then as the policy sizes them: each batch end after the flow is measured
    // records a throughput sample, and none is recorded mid-batch.
    assert!(
        throughput.count > 0 && throughput.count < 12,
        "{throughput:?}"
    );
    // One frame each 10 µs: 100,000 frames per second.
    assert!((throughput.max - 100_000.0).abs() < 1.0, "{throughput:?}");

    // The flow meters carry the recorders' current statistics.
    let mut meters = Vec::new();
    metrics.collect(0, &mut meters);
    let flow = |suffix: &str| {
        *meters
            .iter()
            .find(|meter| meter.name.ends_with(suffix))
            .unwrap()
            .flow()
            .unwrap()
    };
    // The service-time flow is the mean execute time of each cycle's frames.
    let service_time = flow(".execution.serviceTime.flow");
    assert_eq!((service_time.average, service_time.max), (200.0, 200.0));
    assert_eq!(service_time.standard_deviation, 0.0);
    let rate = flow(".execution.throughput.flow");
    assert!((rate.average - 100_000.0).abs() < 1.0, "{rate:?}");

    // Without metrics a cycle keeps no flow.
    assert!(Cycle::new(0, None).flow.is_none());
}

/// The cost of a cycle's bookkeeping with metrics off and on, on one thread, without the per-frame
/// body timing (which `metrics_overhead_per_frame` includes): what the worker
/// pays per productive cycle, whatever the frames.
#[test]
#[ignore = "a measurement: run with --ignored --nocapture"]
fn metrics_cost_per_cycle() {
    const CYCLES: u32 = 2_000_000;
    let registry = MetricsRegistry::new();
    let config = crate::metrics::MetricsConfig::default().with_registry(Some(registry));
    let metrics = ExecutionMetrics::new(&config, 0, 0).unwrap();
    for processed in [1, 64] {
        let mut costs = Vec::new();
        for metered in [false, true, false, true] {
            let mut policy = DecisionTree::with_thresholds(
                crate::config::IdlePolicy::default(),
                BodyCost::new(),
                Duration::from_micros(1),
                Duration::from_micros(1),
            );
            let meters = metered.then_some(&**metrics);
            let mut cycle = Cycle::new(0, meters);
            let run = Run {
                processed,
                elapsed: Duration::from_nanos(100 * processed),
            };
            let start = Instant::now();
            for _ in 0..CYCLES {
                cycle.now += Duration::from_micros(1);
                cycle.complete(&run, &mut policy, 4096, std::hint::black_box(meters), || 1);
            }
            costs.push((
                metered,
                start.elapsed().as_nanos() as f64 / f64::from(CYCLES),
            ));
        }
        let mean = |on: bool| {
            let picked: Vec<f64> = costs.iter().filter(|c| c.0 == on).map(|c| c.1).collect();
            picked.iter().sum::<f64>() / picked.len() as f64
        };
        let (off, on) = (mean(false), mean(true));
        eprintln!(
            "{processed} frames/cycle: {off:.1} ns/cycle off, {on:.1} ns/cycle on: {:+.1} ns/cycle, {:+.2} ns/frame",
            on - off,
            (on - off) / processed as f64
        );
    }
}

#[test]
fn a_parked_worker_does_not_hold_back_a_retirement() {
    let Some(fixture) = Fixture::single() else {
        return;
    };
    let fixture = fixture.started();
    // The worker has nothing to run: it parks between short cycles.
    thread::sleep(Duration::from_millis(50));
    let epochs = fixture.registry().epochs().clone();
    assert!(epochs.slot_count() >= 1);
    let cell = euhedral_data_structures::sync::Published::new(epochs, Some(Arc::new(7_u32)));
    let started = Instant::now();
    cell.store(None)
        .wait(Instant::now() + Duration::from_secs(1))
        .unwrap_or_else(|_| panic!("an idle worker holds nothing back"));
    // Timing is asserted on a quiet machine; a shared CI runner reports it.
    if std::env::var_os("CI").is_some() {
        eprintln!(
            "the retirement waited {:?} for a parked worker",
            started.elapsed()
        );
    } else {
        assert!(
            started.elapsed() < Duration::from_millis(250),
            "the retirement waited {:?} for a parked worker",
            started.elapsed()
        );
    }
}

#[test]
fn a_stopped_worker_withdraws_its_announcement() {
    let Some(fixture) = Fixture::single() else {
        return;
    };
    let epochs = fixture.registry().epochs().clone();
    let fixture = fixture.started();
    fixture.fragment.close();
    wait_until("the worker to stop", || !fixture.fragment.ready());
    let cell = euhedral_data_structures::sync::Published::new(epochs, Some(Arc::new(7_u32)));
    cell.store(None)
        .wait(Instant::now() + Duration::from_secs(1))
        .unwrap_or_else(|_| panic!("a stopped worker holds nothing back"));
}
