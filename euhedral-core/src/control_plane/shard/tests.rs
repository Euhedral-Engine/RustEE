use super::*;
use crate::control_plane::testing::TestClone;
use crate::flow_control::UpstreamRegistry;
use crate::flow_control::testing::{Fate, Fates, Tracked, frame};
use euhedral_hardware_utils::monitor::{HardwareUtilization, HardwareUtilizationParts};
use euhedral_hardware_utils::topology::{CpuInfo, Topology};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;

const SPAN: usize = 4;

/// Core `c` of the test socket has CPUs `2c` and `2c + 1`.
fn cpus_of(core: usize) -> CpuSet {
    CpuSet::range(2 * core, 2 * core + 2)
}

fn topology(version: u64, cores: &[usize]) -> EffectiveSocketTopology {
    let mut effective_core_to_cpu = vec![None; SPAN];
    for &core in cores {
        effective_core_to_cpu[core] = Some(cpus_of(core));
    }
    EffectiveSocketTopology {
        version,
        socket_id: 0,
        effective_cores: cores.iter().copied().collect(),
        effective_cpus: cores
            .iter()
            .flat_map(|&core| cpus_of(core).iter().collect::<Vec<_>>())
            .collect(),
        effective_core_to_cpu,
    }
}

/// A snapshot with every core of the socket.
fn snapshot() -> SocketSnapshot {
    let utilization = HardwareUtilization::new(HardwareUtilizationParts {
        quota_cpus: (2 * SPAN) as f64,
        global_effective_cpus: CpuSet::range(0, 2 * SPAN),
        ..Default::default()
    });
    let all = topology(0, &(0..SPAN).collect::<Vec<_>>());
    utilization
        .socket_snapshot(0, &all.effective_core_to_cpu, (2 * SPAN) as f64)
        .unwrap()
        .unwrap()
}

fn eventually(what: &str, condition: impl Fn() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while !condition() {
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        thread::sleep(Duration::from_millis(1));
    }
}

struct Fixture {
    registry: Arc<UpstreamRegistry>,
    directory: Arc<CacheDirectory>,
    upstream: Arc<LatticeEdge>,
    template: Arc<TestClone>,
    shard: Arc<ControlPlaneShard>,
}

impl Fixture {
    fn new(shutdown_timeout: Duration) -> Self {
        let registry = UpstreamRegistry::new();
        let directory = CacheDirectory::new(registry.clone(), Topology::system().clone());
        let upstream = LatticeEdge::new(&registry, Default::default());
        let template = TestClone::template();
        let shard = ControlPlaneShard::new(
            &directory,
            0,
            "shard",
            template.clone(),
            shutdown_timeout,
            SPAN,
        );
        Self {
            registry,
            directory,
            upstream,
            template,
            shard,
        }
    }

    fn started(cores: &[usize]) -> Self {
        let fixture = Self::new(Duration::from_secs(10));
        fixture.start(cores);
        fixture
    }

    fn start(&self, cores: &[usize]) {
        self.shard.start(
            Some(&snapshot()),
            &topology(1, cores),
            self.upstream.clone(),
        );
    }

    fn update(&self, version: u64, cores: &[usize]) {
        self.shard
            .update(Some(&snapshot()), &topology(version, cores));
    }

    fn rebalance(&self, version: u64, cores: &[usize]) {
        self.update(version, cores);
        self.settle();
    }

    fn settle(&self) {
        eventually("the rebalance", || !self.shard.is_rebalancing());
    }

    fn spawned(&self) -> Vec<Arc<TestClone>> {
        self.template.spawned()
    }

    /// The latest clone spawned on `core`.
    fn clone_on(&self, core: usize) -> Arc<TestClone> {
        self.spawned()
            .into_iter()
            .rev()
            .find(|clone| clone.config().core_id() == core)
            .expect("a clone was spawned on the core")
    }

    fn push(&self, frame: Box<dyn Frame>) {
        self.upstream.push_downstream(frame);
    }
}

fn hashed(id: u64, routing_hash: u64) -> Box<dyn Frame> {
    let mut frame = frame(id);
    frame.header_mut().set_routing_hash(routing_hash);
    frame
}

fn from_core(id: u64, policy: RoutingPolicy, core: usize) -> Box<dyn Frame> {
    let mut frame = frame(id);
    let header = frame.header_mut();
    header.set_routing_policy(policy);
    header.set_origin(Some(CpuInfo {
        cpu: 2 * core,
        core,
        socket: 0,
    }));
    frame
}

#[test]
fn start_primes_a_started_clone_on_each_effective_core() {
    let fixture = Fixture::started(&[0, 2]);
    let shard = &fixture.shard;

    let spawned = fixture.spawned();
    assert_eq!(spawned.len(), 2);
    for (clone, core) in spawned.iter().zip([0, 2]) {
        assert_eq!(clone.config().core_id(), core);
        assert_eq!(clone.config().shard_name(), "shard");
        assert_eq!(clone.config().effective_cpus(), &cpus_of(core));
        assert!(Arc::ptr_eq(
            clone.config().cache_directory().unwrap(),
            &fixture.directory
        ));
        assert_eq!(clone.starts.load(Ordering::SeqCst), 1);
        assert_eq!(clone.updates(), [core]);
        assert_eq!(clone.drain_modes(), [true, false]);
        assert!(!clone.edge().drain_flag().load(Ordering::SeqCst));
    }
    assert_eq!(shard.active_cores(), 2);
    assert!(shard.is_started());
    assert!(!shard.is_rebalancing());
    assert!(shard.is_drained());
    assert_eq!(shard.socket(), 0);
    assert_eq!(shard.name(), "shard");
}

#[test]
fn starting_a_running_shard_does_nothing() {
    let fixture = Fixture::started(&[0]);
    fixture.start(&[0, 1]);
    assert_eq!(fixture.spawned().len(), 1);
    assert_eq!(fixture.shard.active_cores(), 1);
}

#[test]
fn a_shard_that_is_not_started_ignores_updates() {
    let fixture = Fixture::new(Duration::from_secs(10));
    fixture.update(1, &[0]);
    assert!(fixture.spawned().is_empty());
    assert!(!fixture.shard.is_started());
    assert!(fixture.shard.is_drained());
    assert_eq!(fixture.shard.active_cores(), 0);
}

#[test]
fn frames_route_by_their_rotated_hash() {
    let fixture = Fixture::started(&[1, 3]);
    // Rotated left by 31 bits, the first hash lands in the upper half and the second in the lower.
    fixture.push(hashed(1, 1 << 32));
    fixture.push(hashed(2, 1 << 63));
    assert_eq!(fixture.clone_on(1).receiver.received(), [2]);
    assert_eq!(fixture.clone_on(3).receiver.received(), [1]);
}

#[test]
fn cache_local_frames_go_to_their_origin_core_while_it_is_active() {
    let fixture = Fixture::started(&[0, 2]);
    for id in 0..8 {
        fixture.push(from_core(id, RoutingPolicy::CacheLocal, 2));
    }
    assert_eq!(
        fixture.clone_on(2).receiver.received(),
        (0..8).collect::<Vec<_>>()
    );
    assert!(fixture.clone_on(0).receiver.received().is_empty());
}

#[test]
fn other_frames_route_by_hash_whatever_their_origin() {
    let fixture = Fixture::started(&[0, 2]);
    // Hash 0 routes to the first active core; neither frame may follow its origin to core 2.
    let mut socket_local = from_core(1, RoutingPolicy::SocketLocal, 2);
    socket_local.header_mut().set_routing_hash(0);
    let mut inactive_origin = from_core(2, RoutingPolicy::CacheLocal, 1);
    inactive_origin.header_mut().set_routing_hash(0);
    fixture.push(socket_local);
    fixture.push(inactive_origin);
    assert_eq!(fixture.clone_on(0).receiver.received(), [1, 2]);
}

#[test]
fn an_update_of_the_same_version_updates_each_clone() {
    let fixture = Fixture::started(&[0, 2]);
    fixture.update(1, &[0, 2]);
    assert_eq!(fixture.spawned().len(), 2);
    assert_eq!(fixture.clone_on(0).updates(), [0, 0]);
    assert_eq!(fixture.clone_on(2).updates(), [2, 2]);
    assert!(!fixture.shard.is_rebalancing());
}

#[test]
fn an_update_restarts_a_clone_that_stopped() {
    let fixture = Fixture::started(&[0]);
    let clone = fixture.clone_on(0);
    clone.started.store(false, Ordering::SeqCst);
    fixture.update(1, &[0]);
    assert_eq!(clone.starts.load(Ordering::SeqCst), 2);
    assert!(clone.is_started());
}

#[test]
fn removing_a_core_drains_every_clone_and_closes_the_removed_one() {
    let fixture = Fixture::started(&[0, 2]);
    let (kept, removed) = (fixture.clone_on(0), fixture.clone_on(2));

    fixture.rebalance(2, &[0]);

    eventually("the removed clone to close", || removed.is_closed());
    assert!(removed.receiver.is_completed());
    assert!(!kept.is_closed());
    assert_eq!(kept.drain_modes(), [true, false, true, false]);
    eventually("the kept clone's update", || kept.updates() == [0, 0]);
    assert_eq!(fixture.spawned().len(), 2);
    assert_eq!(fixture.shard.active_cores(), 1);

    fixture.push(from_core(7, RoutingPolicy::CacheLocal, 2));
    fixture.push(hashed(8, u64::MAX));
    assert_eq!(kept.receiver.received(), [7, 8]);
    assert!(!kept.edge().drain_flag().load(Ordering::SeqCst));
}

#[test]
fn adding_a_core_spawns_a_clone_and_keeps_the_others() {
    let fixture = Fixture::started(&[0]);
    let kept = fixture.clone_on(0);

    fixture.rebalance(2, &[0, 3]);

    let added = fixture.clone_on(3);
    assert_eq!(fixture.spawned().len(), 2);
    assert_eq!(added.drain_modes(), [true, false]);
    assert_eq!(added.updates(), [3]);
    assert!(!kept.is_closed());
    assert_eq!(fixture.shard.active_cores(), 2);
    fixture.push(from_core(5, RoutingPolicy::CacheLocal, 3));
    assert_eq!(added.receiver.received(), [5]);
}

#[test]
fn a_clone_that_does_not_drain_in_time_is_replaced() {
    let fixture = Fixture::new(Duration::from_millis(20));
    fixture.start(&[0, 1]);
    let stuck = fixture.clone_on(0);
    let healthy = fixture.clone_on(1);
    stuck.drained.store(false, Ordering::SeqCst);

    fixture.rebalance(2, &[0, 1]);

    let replacement = fixture.clone_on(0);
    assert!(!Arc::ptr_eq(&replacement, &stuck));
    eventually("the stuck clone to close", || stuck.is_closed());
    assert!(stuck.receiver.is_completed());
    assert!(!healthy.is_closed());
    assert_eq!(fixture.spawned().len(), 3);
    assert_eq!(replacement.config().effective_cpus(), &cpus_of(0));
    assert_eq!(replacement.drain_modes(), [true, false]);
    assert_eq!(replacement.updates(), [0]);

    fixture.push(from_core(9, RoutingPolicy::CacheLocal, 0));
    assert_eq!(replacement.receiver.received(), [9]);
    assert!(stuck.receiver.received().is_empty());
    assert!(fixture.shard.is_drained());
}

#[test]
fn a_rebalance_with_no_previous_clones_reopens_ingest() {
    let fixture = Fixture::started(&[]);
    assert!(fixture.spawned().is_empty());
    assert!(!fixture.shard.is_rebalancing());

    fixture.update(2, &[1]);

    assert!(!fixture.shard.is_rebalancing());
    let clone = fixture.clone_on(1);
    assert_eq!(clone.drain_modes(), [true, false]);
    assert!(!clone.edge().drain_flag().load(Ordering::SeqCst));
    fixture.push(hashed(4, 0));
    assert_eq!(clone.receiver.received(), [4]);
}

#[test]
fn a_shard_without_cores_primes_and_fails_the_frames_it_gets() {
    let fixture = Fixture::started(&[]);
    assert!(fixture.shard.is_started());
    assert!(fixture.shard.is_drained());
    assert!(!fixture.shard.is_rebalancing());
    assert_eq!(fixture.shard.active_cores(), 0);

    let fates = Fates::default();
    fixture.push(Tracked::new(1, &fates));
    let mut cache_local = Tracked::new(2, &fates);
    cache_local
        .header_mut()
        .set_routing_policy(RoutingPolicy::CacheLocal);
    cache_local.header_mut().set_origin(Some(CpuInfo {
        cpu: 0,
        core: 0,
        socket: 0,
    }));
    fixture.push(cache_local);

    let fates = fates.lock().unwrap().clone();
    assert_eq!(fates.len(), 2);
    assert!(
        fates
            .iter()
            .all(|(_, fate)| matches!(fate, Fate::Failed(_)))
    );
    assert_eq!(fixture.shard.reset_for_next_trial(Instant::now()), Ok(0));
}

#[test]
fn a_shard_without_cores_keeps_ingest_drained_until_a_core_appears() {
    let fixture = Fixture::started(&[]);
    let distributor = fixture.shard.lock().distributor.clone().unwrap();
    assert!(distributor.edge().drain_flag().load(Ordering::SeqCst));

    fixture.rebalance(2, &[0]);
    assert!(!distributor.edge().drain_flag().load(Ordering::SeqCst));
}

#[test]
fn removing_every_core_leaves_ingest_drained() {
    let fixture = Fixture::started(&[0]);
    let clone = fixture.clone_on(0);
    let drain = clone.edge().drain_flag().clone();

    fixture.rebalance(2, &[]);

    eventually("the clone to close", || clone.is_closed());
    assert!(drain.load(Ordering::SeqCst));
    assert_eq!(fixture.shard.active_cores(), 0);
}

#[test]
fn updates_during_a_rebalance_are_ignored() {
    let fixture = Fixture::started(&[0]);
    let clone = fixture.clone_on(0);
    clone.drained.store(false, Ordering::SeqCst);

    fixture.update(2, &[0, 1]);
    assert!(fixture.shard.is_rebalancing());
    assert!(!fixture.shard.is_drained());
    assert!(clone.edge().drain_flag().load(Ordering::SeqCst));
    fixture.update(3, &[0, 1, 2]);
    assert_eq!(fixture.spawned().len(), 2);

    clone.drained.store(true, Ordering::SeqCst);
    fixture.settle();
    assert!(!clone.is_closed());
    assert_eq!(fixture.shard.active_cores(), 2);
    fixture.rebalance(3, &[0, 1, 2]);
    assert_eq!(fixture.shard.active_cores(), 3);
}

#[test]
fn is_started_waits_for_every_clone_to_be_ready() {
    let fixture = Fixture::started(&[0, 1]);
    fixture.clone_on(1).ready.store(false, Ordering::SeqCst);
    assert!(!fixture.shard.is_started());
    fixture.clone_on(1).ready.store(true, Ordering::SeqCst);
    assert!(fixture.shard.is_started());
}

#[test]
fn is_drained_waits_for_every_clone() {
    let fixture = Fixture::started(&[0, 1]);
    fixture.clone_on(1).drained.store(false, Ordering::SeqCst);
    assert!(!fixture.shard.is_drained());
    fixture.clone_on(1).drained.store(true, Ordering::SeqCst);
    assert!(fixture.shard.is_drained());
}

#[test]
fn reset_drains_resets_every_clone_and_reopens_ingest() {
    let fixture = Fixture::started(&[0, 1]);
    fixture.clone_on(0).buffered.store(3, Ordering::SeqCst);
    fixture.clone_on(1).buffered.store(4, Ordering::SeqCst);

    let deadline = Instant::now() + Duration::from_secs(1);
    assert_eq!(fixture.shard.reset_for_next_trial(deadline), Ok(7));

    for core in [0, 1] {
        let clone = fixture.clone_on(core);
        assert_eq!(clone.drain_modes(), [true, false, true, false]);
        assert!(!clone.edge().drain_flag().load(Ordering::SeqCst));
    }
    assert_eq!(fixture.shard.reset_for_next_trial(deadline), Ok(0));
}

#[test]
fn a_failed_reset_still_reopens_ingest() {
    let fixture = Fixture::started(&[0]);
    let clone = fixture.clone_on(0);
    clone.panic_on_reset.store(true, Ordering::SeqCst);

    let deadline = Instant::now() + Duration::from_secs(1);
    let reset = panic::catch_unwind(AssertUnwindSafe(|| {
        fixture.shard.reset_for_next_trial(deadline)
    }));

    assert!(reset.is_err());
    assert!(!clone.in_drain_mode());
    assert!(!clone.edge().drain_flag().load(Ordering::SeqCst));
}

#[test]
fn reset_of_a_shard_that_is_not_started_clears_nothing() {
    let fixture = Fixture::new(Duration::from_secs(10));
    assert_eq!(fixture.shard.reset_for_next_trial(Instant::now()), Ok(0));
}

#[test]
fn reset_fails_if_the_rebalance_outlasts_the_deadline() {
    let fixture = Fixture::started(&[0]);
    let clone = fixture.clone_on(0);
    clone.drained.store(false, Ordering::SeqCst);
    clone.buffered.store(5, Ordering::SeqCst);
    fixture.update(2, &[0]);

    let deadline = Instant::now() + Duration::from_millis(5);
    assert_eq!(
        fixture.shard.reset_for_next_trial(deadline),
        Err(RebalanceTimedOut {
            name: "shard".into()
        })
    );
    assert_eq!(clone.buffered.load(Ordering::SeqCst), 5);
    assert_eq!(
        RebalanceTimedOut {
            name: "shard".into()
        }
        .to_string(),
        "timed out waiting for shard to finish rebalancing"
    );

    clone.drained.store(true, Ordering::SeqCst);
    fixture.settle();
}

fn counter() -> (Arc<AtomicUsize>, impl FnOnce() + Send + 'static) {
    let count = Arc::new(AtomicUsize::new(0));
    let counted = count.clone();
    (count, move || {
        counted.fetch_add(1, Ordering::SeqCst);
    })
}

#[test]
fn shut_down_waits_for_the_upstream_to_retire() {
    let fixture = Fixture::started(&[0, 1]);
    let parent = parented(&fixture);

    let (calls, done) = counter();
    fixture.shard.shut_down(done);
    thread::sleep(Duration::from_millis(10));
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert!(fixture.shard.is_started());

    assert!(parent.set_downstream_mapping(&[], &[None]));
    eventually("the shutdown", || calls.load(Ordering::SeqCst) == 1);
    for clone in fixture.spawned() {
        assert!(clone.is_closed());
        assert!(clone.in_drain_mode());
        assert!(clone.receiver.is_completed());
    }
    assert!(!fixture.shard.is_started());
    assert_eq!(fixture.shard.active_cores(), 0);
}

#[test]
fn shut_down_finishes_once() {
    let fixture = Fixture::started(&[0]);
    let (calls, done) = counter();
    let (repeated, again) = counter();
    fixture.shard.shut_down(done);
    eventually("the shutdown", || calls.load(Ordering::SeqCst) == 1);
    fixture.shard.shut_down(again);
    assert_eq!(repeated.load(Ordering::SeqCst), 1);
    thread::sleep(Duration::from_millis(10));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[test]
fn shutting_down_a_shard_that_is_not_started_finishes_at_once() {
    let fixture = Fixture::new(Duration::from_secs(10));
    let (calls, done) = counter();
    fixture.shard.shut_down(done);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[test]
fn shutting_down_a_shard_without_clones_finishes() {
    let fixture = Fixture::started(&[]);
    let (calls, done) = counter();
    fixture.shard.shut_down(done);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert!(!fixture.shard.is_started());
}

#[test]
fn shut_down_forces_out_a_clone_that_does_not_drain() {
    let fixture = Fixture::new(Duration::from_millis(10));
    fixture.start(&[0]);
    let clone = fixture.clone_on(0);
    clone.drained.store(false, Ordering::SeqCst);
    let (calls, done) = counter();
    fixture.shard.shut_down(done);
    eventually("the shutdown", || calls.load(Ordering::SeqCst) == 1);
    assert!(clone.is_closed());
}

#[test]
fn shut_down_finishes_even_if_a_clone_fails_to_close() {
    let fixture = Fixture::started(&[0, 1]);
    fixture
        .clone_on(0)
        .panic_on_dump
        .store(true, Ordering::SeqCst);
    let (calls, done) = counter();
    fixture.shard.shut_down(done);
    eventually("the shutdown", || calls.load(Ordering::SeqCst) == 1);
}

#[test]
fn close_closes_every_clone_at_once() {
    let fixture = Fixture::started(&[0, 1]);
    fixture.shard.close();
    for clone in fixture.spawned() {
        assert!(clone.is_closed());
        assert!(clone.receiver.is_completed());
    }
    assert!(!fixture.shard.is_started());
    assert!(fixture.shard.is_drained());
    assert_eq!(fixture.shard.active_cores(), 0);
    fixture.shard.close();

    let (calls, done) = counter();
    fixture.shard.shut_down(done);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[test]
fn a_closed_shard_starts_again_with_new_clones() {
    let fixture = Fixture::started(&[0]);
    fixture.shard.close();
    let upstream = LatticeEdge::new(&fixture.registry, Default::default());
    fixture
        .shard
        .start(Some(&snapshot()), &topology(1, &[0]), upstream.clone());

    assert_eq!(fixture.spawned().len(), 2);
    let clone = fixture.clone_on(0);
    assert!(!clone.is_closed());
    upstream.push_downstream(hashed(3, 0));
    assert_eq!(clone.receiver.received(), [3]);
}

#[test]
fn closing_during_a_rebalance_still_closes_the_draining_clones() {
    let fixture = Fixture::new(Duration::from_millis(20));
    fixture.start(&[0, 1]);
    let removed = fixture.clone_on(1);
    removed.drained.store(false, Ordering::SeqCst);

    fixture.update(2, &[0]);
    assert!(fixture.shard.is_rebalancing());
    fixture.shard.close();

    eventually("the removed clone to close", || removed.is_closed());
    assert!(fixture.clone_on(0).is_closed());
    assert_eq!(fixture.spawned().len(), 2);
    assert!(!fixture.shard.is_rebalancing());
}

#[test]
fn closing_while_clones_spawn_closes_the_new_clones() {
    let fixture = Fixture::started(&[0]);
    let shard = fixture.shard.clone();
    *fixture.template.on_clone.lock().unwrap() = Some(Box::new(move || shard.close()));

    fixture.update(2, &[0, 1]);

    assert!(fixture.clone_on(1).is_closed());
    assert!(fixture.clone_on(0).is_closed());
    assert!(!fixture.shard.is_started());
    assert_eq!(fixture.shard.active_cores(), 0);
}

#[test]
fn a_removed_core_that_returns_gets_a_new_edge() {
    let fixture = Fixture::started(&[0, 2]);
    let removed = fixture.clone_on(2);
    fixture.rebalance(2, &[0]);
    eventually("the removed clone to close", || removed.is_closed());

    fixture.rebalance(3, &[0, 2]);

    let returned = fixture.clone_on(2);
    assert!(!Arc::ptr_eq(&returned, &removed));
    assert!(!Arc::ptr_eq(&returned.edge(), &removed.edge()));
    fixture.push(from_core(6, RoutingPolicy::CacheLocal, 2));
    assert_eq!(returned.receiver.received(), [6]);
}

#[test]
fn a_removed_clone_closes_only_after_the_routes_to_it_return() {
    let fixture = Fixture::started(&[0, 2]);
    let removed = fixture.clone_on(2);
    let distributor = fixture.shard.lock().distributor.clone().unwrap();
    let held = distributor.hold_routing();

    fixture.rebalance(2, &[0]);
    thread::sleep(Duration::from_millis(10));
    assert!(!removed.is_closed());

    drop(held);
    eventually("the removed clone to close", || removed.is_closed());
}

/// A clone removed by a rebalance leaves the shard's clones; the retirement that closes it waits for the routes to it. If
/// the shard stops and the distributor drops while a route still holds the generation, the retirement must not be lost
/// with the vertex: no one else would close that clone, and its worker would live for the life of the process.
#[test]
fn a_removed_clone_is_closed_even_if_the_distributor_drops_before_the_routes_to_it_return() {
    let fixture = Fixture::started(&[0, 2]);
    let removed = fixture.clone_on(2);
    let distributor = fixture.shard.lock().distributor.clone().unwrap();
    let held = distributor.hold_routing();
    fixture.rebalance(2, &[0]);
    thread::sleep(Duration::from_millis(10));
    assert!(!removed.is_closed(), "a held generation still routes to it");

    let weak = Arc::downgrade(&distributor);
    drop(distributor);
    fixture.shard.close();
    // The upstream edge holds the distributor as its downstream node, as the lattice's own vertex does in production.
    let Fixture { upstream, .. } = fixture;
    drop(upstream);
    assert!(
        weak.upgrade().is_none(),
        "the distributor dropped with its upstream"
    );
    // The shard's close closes the clones it still has; this one it no longer has.
    drop(held);
    eventually("the removed clone to close", || removed.is_closed());
}

#[test]
fn a_shard_draining_a_removed_clone_is_not_drained() {
    let fixture = Fixture::started(&[0, 1]);
    let removed = fixture.clone_on(1);
    removed.drained.store(false, Ordering::SeqCst);

    fixture.update(2, &[0]);
    assert!(!fixture.shard.is_drained());

    removed.drained.store(true, Ordering::SeqCst);
    fixture.settle();
    assert!(fixture.shard.is_drained());
}

#[test]
fn an_update_of_the_new_version_during_a_rebalance_is_ignored() {
    let fixture = Fixture::started(&[0]);
    let clone = fixture.clone_on(0);
    clone.drained.store(false, Ordering::SeqCst);

    fixture.update(2, &[0, 1]);
    fixture.update(2, &[0, 1]);
    assert_eq!(fixture.clone_on(1).updates(), [1]);

    clone.drained.store(true, Ordering::SeqCst);
    fixture.settle();
}

#[test]
fn clone_updates_are_dropped_once_the_shard_closes() {
    let fixture = Fixture::started(&[1]);
    let kept = fixture.clone_on(1);
    let shard = fixture.shard.clone();
    *fixture.template.on_clone.lock().unwrap() = Some(Box::new(move || shard.close()));

    // Core 0 spawns, and closes the shard, before core 1's update is queued.
    fixture.update(2, &[0, 1]);

    assert_eq!(kept.updates(), [1]);
    assert!(kept.is_closed());
}

#[test]
fn reset_waits_out_a_rebalance() {
    let fixture = Fixture::started(&[0]);
    let clone = fixture.clone_on(0);
    clone.drained.store(false, Ordering::SeqCst);
    clone.buffered.store(2, Ordering::SeqCst);
    fixture.update(2, &[0]);

    let unstick = clone.clone();
    let releaser = thread::spawn(move || {
        thread::sleep(Duration::from_millis(20));
        unstick.drained.store(true, Ordering::SeqCst);
    });
    let deadline = Instant::now() + Duration::from_secs(10);
    assert_eq!(fixture.shard.reset_for_next_trial(deadline), Ok(2));
    releaser.join().unwrap();
}

/// Maps the fixture's upstream below a parent vertex, so retiring it waits on the parent.
fn parented(fixture: &Fixture) -> Arc<LatticeVertex> {
    let parent = LatticeVertex::new(&fixture.registry, "parent", 1);
    parent.set_drain(true);
    let upstream: Arc<dyn LatticeNode> = fixture.upstream.clone();
    assert!(parent.set_downstream_mapping(&[0], &[Some(upstream)]));
    parent
}

#[test]
fn a_second_shut_down_finishes_at_once() {
    let fixture = Fixture::started(&[0]);
    let parent = parented(&fixture);
    let (first, done) = counter();
    let (second, again) = counter();

    fixture.shard.shut_down(done);
    fixture.shard.shut_down(again);
    assert_eq!(first.load(Ordering::SeqCst), 0);
    assert_eq!(second.load(Ordering::SeqCst), 1);

    assert!(parent.set_downstream_mapping(&[], &[None]));
    eventually("the shutdown", || first.load(Ordering::SeqCst) == 1);
}

#[test]
fn a_shut_down_overtaken_by_a_restart_leaves_the_new_run_alone() {
    let fixture = Fixture::started(&[0]);
    let parent = parented(&fixture);
    let (calls, done) = counter();
    fixture.shard.shut_down(done);

    fixture.shard.close();
    let upstream = LatticeEdge::new(&fixture.registry, Default::default());
    fixture
        .shard
        .start(Some(&snapshot()), &topology(1, &[0]), upstream);
    let restarted = fixture.clone_on(0);

    assert!(parent.set_downstream_mapping(&[], &[None]));
    eventually("the stale shutdown", || calls.load(Ordering::SeqCst) == 1);
    assert!(fixture.shard.is_started());
    assert!(!restarted.is_closed());
}

#[test]
fn a_shard_without_a_snapshot_places_clones_by_its_topology() {
    let fixture = Fixture::new(Duration::from_secs(10));
    fixture
        .shard
        .start(None, &topology(1, &[0, 3]), fixture.upstream.clone());

    for core in [0, 3] {
        let clone = fixture.clone_on(core);
        assert_eq!(clone.config().effective_cpus(), &cpus_of(core));
        assert!(clone.is_started());
        assert!(clone.updates().is_empty());
    }
    fixture.shard.update(None, &topology(1, &[0, 3]));
    assert!(fixture.clone_on(0).updates().is_empty());
}
