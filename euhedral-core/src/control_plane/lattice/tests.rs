use super::*;
use crate::control_plane::testing::TestClone;
use crate::flow_control::LatticeHotSource;
use crate::flow_control::testing::{Fate, Fates, Tracked, frame};
use euhedral_hardware_utils::monitor::HardwareUtilizationParts;
use euhedral_hardware_utils::topology::CpuInfo;
use euhedral_hardware_utils::topology::source::{CoreKind, LogicalCpu, TopologyInput, normalize};
use euhedral_hardware_utils::util::CpuSet;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Two sockets of two cores each; core `c` has CPUs `2c` and `2c + 1` and sits on socket `c / 2`.
fn two_sockets() -> euhedral_hardware_utils::topology::Topology {
    let cpus = (0..8)
        .map(|cpu| LogicalCpu {
            id: cpu,
            socket_key: format!("linux:package:{:010}", cpu / 4),
            die_key: "linux:die:0000000000".to_string(),
            core_key: format!("linux:core:{:010}", cpu / 2),
            kind: CoreKind::Unknown,
        })
        .collect();
    normalize(TopologyInput {
        provider: "linux".to_string(),
        cpus,
        caches: Vec::new(),
    })
    .unwrap()
}

const ALL: [usize; 8] = [0, 1, 2, 3, 4, 5, 6, 7];
const SOCKET_0: [usize; 4] = [0, 1, 2, 3];

fn utilization(cpus: &[usize], quota_cpus: f64) -> Arc<HardwareUtilization> {
    Arc::new(HardwareUtilization::new(HardwareUtilizationParts {
        quota_cpus,
        global_effective_cpus: cpus.iter().copied().collect(),
        ..Default::default()
    }))
}

/// Publishes the utilizations a test hands it, on the test's thread.
#[derive(Default)]
struct ManualSource {
    latest: Mutex<Option<Arc<HardwareUtilization>>>,
    listeners: Mutex<Vec<UtilizationListener>>,
    starts: AtomicUsize,
    closed: AtomicBool,
}

impl ManualSource {
    /// Makes `utilization` the latest, without publishing it.
    fn set(&self, utilization: Arc<HardwareUtilization>) {
        *self.latest.lock().unwrap() = Some(utilization);
    }

    /// Makes `utilization` the latest and publishes it if started.
    fn publish(&self, utilization: Arc<HardwareUtilization>) {
        self.set(utilization.clone());
        if self.starts.load(Ordering::SeqCst) > 0 && !self.closed.load(Ordering::SeqCst) {
            for listener in self.listeners.lock().unwrap().iter() {
                listener(&utilization);
            }
        }
    }
}

impl UtilizationSource for ManualSource {
    fn utilization(&self) -> Option<Arc<HardwareUtilization>> {
        self.latest.lock().unwrap().clone()
    }

    fn subscribe(&self, listener: UtilizationListener) {
        self.listeners.lock().unwrap().push(listener);
    }

    fn start(&self) -> Result<(), MonitorError> {
        self.starts.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    fn close(&self) {
        self.closed.store(true, Ordering::SeqCst);
    }
}

fn eventually(what: &str, condition: impl Fn() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while !condition() {
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        thread::sleep(Duration::from_millis(1));
    }
}

struct Fixture {
    source: Arc<ManualSource>,
    template: Arc<TestClone>,
    lattice: Arc<ControlPlaneLattice>,
}

impl Fixture {
    fn new(allowed: &[usize], configure: impl FnOnce(&mut LatticeConfig)) -> Self {
        let source = Arc::new(ManualSource::default());
        let template = TestClone::template();
        let mut config = LatticeConfig::new(template.clone());
        config.name = "lattice".into();
        config.shard_name = "shard".into();
        config.startup_timeout = Duration::from_secs(10);
        config.shutdown_timeout = Duration::from_secs(10);
        configure(&mut config);
        let mapper = Arc::new(TopologyMapper::with_topology(
            two_sockets(),
            &allowed.iter().copied().collect(),
        ));
        let lattice = ControlPlaneLattice::with_source(config, mapper, source.clone());
        Self {
            source,
            template,
            lattice,
        }
    }

    /// A started lattice over both sockets, with `cpus` effective.
    fn started(cpus: &[usize]) -> Self {
        let fixture = Self::new(&ALL, |_| {});
        fixture.source.set(utilization(cpus, cpus.len() as f64));
        fixture.lattice.start().unwrap();
        fixture
    }

    fn publish(&self, cpus: &[usize]) {
        self.source.publish(utilization(cpus, cpus.len() as f64));
        self.settle();
    }

    fn settle(&self) {
        eventually("the rebalance", || {
            !self.lattice.is_rebalancing()
                && self
                    .lattice
                    .shards
                    .iter()
                    .all(|shard| !shard.is_rebalancing())
        });
    }

    /// The latest clone spawned on `core`.
    fn clone_on(&self, core: usize) -> Arc<TestClone> {
        self.template
            .spawned()
            .into_iter()
            .rev()
            .find(|clone| clone.config().core_id() == core)
            .expect("a clone was spawned on the core")
    }

    fn received_on(&self, cores: &[usize]) -> Vec<u64> {
        let mut received: Vec<u64> = cores
            .iter()
            .flat_map(|&core| self.clone_on(core).receiver.received())
            .collect();
        received.sort_unstable();
        received
    }

    fn push(&self, frame: Box<dyn Frame>) {
        self.lattice.controller.push(frame);
    }

    fn drained(&self) -> bool {
        self.lattice
            .controller
            .edge()
            .drain_flag()
            .load(Ordering::SeqCst)
    }
}

fn from_socket(id: u64, policy: RoutingPolicy, socket: usize) -> Box<dyn Frame> {
    let mut frame = frame(id);
    let header = frame.header_mut();
    header.set_routing_policy(policy);
    header.set_origin(Some(CpuInfo {
        cpu: 4 * socket,
        core: 2 * socket,
        socket,
    }));
    frame
}

fn hashed(id: u64, routing_hash: u64) -> Box<dyn Frame> {
    let mut frame = frame(id);
    frame.header_mut().set_routing_hash(routing_hash);
    frame
}

#[test]
fn the_test_topology_numbers_cores_by_cpu_pairs() {
    let topology = two_sockets();
    assert_eq!(topology.socket_count(), 2);
    assert_eq!(topology.core_count(), 4);
    for cpu in ALL {
        let info = topology.cpu_info(cpu).unwrap();
        assert_eq!((info.core, info.socket), (cpu / 2, cpu / 4));
    }
}

#[test]
fn start_places_a_clone_on_every_effective_core() {
    let fixture = Fixture::started(&ALL);
    let lattice = &fixture.lattice;

    assert_eq!(lattice.active_workers(), 4);
    assert!(lattice.shards.iter().all(|shard| shard.is_started()));
    for core in 0..4 {
        let clone = fixture.clone_on(core);
        assert_eq!(
            clone.config().shard_name(),
            format!("lattice-shard-{}", core / 2)
        );
        assert_eq!(
            clone.config().effective_cpus(),
            &CpuSet::range(2 * core, 2 * core + 2)
        );
        assert_eq!(clone.updates(), [core]);
    }
    let directory = fixture
        .clone_on(0)
        .config()
        .cache_directory()
        .unwrap()
        .clone();
    for core in 0..4 {
        let clone = fixture.clone_on(core);
        assert!(Arc::ptr_eq(
            clone.config().cache_directory().unwrap(),
            &directory
        ));
        assert!(Arc::ptr_eq(clone.edge().registry(), directory.registry()));
    }
    assert_eq!(directory.topology().socket_count(), 2);
    assert_eq!(fixture.source.starts.load(Ordering::SeqCst), 1);
    assert!(!fixture.drained());
    assert!(lattice.is_drained());
    assert_eq!(lattice.name(), "lattice");
    assert_eq!(lattice.shards[1].name(), "lattice-shard-1");

    lattice.start().unwrap();
    assert_eq!(fixture.template.spawned().len(), 4);
    assert_eq!(fixture.source.starts.load(Ordering::SeqCst), 1);
}

#[test]
fn a_blank_name_becomes_the_type_name() {
    let fixture = Fixture::new(&ALL, |config| config.name = " ".into());
    assert_eq!(fixture.lattice.name(), "ControlPlaneLattice");
    assert_eq!(fixture.lattice.config().shard_name, "shard");
}

#[test]
fn frames_route_to_their_origin_socket_or_by_hash() {
    let fixture = Fixture::started(&ALL);
    fixture.push(from_socket(1, RoutingPolicy::SocketLocal, 1));
    fixture.push(from_socket(2, RoutingPolicy::CacheLocal, 1));
    fixture.push(from_socket(3, RoutingPolicy::SocketLocal, 0));
    // Hash routing: the upper half of the hash space goes to the second socket.
    fixture.push(hashed(4, 0));
    fixture.push(hashed(5, u64::MAX));
    let mut anywhere = from_socket(6, RoutingPolicy::Anywhere, 0);
    anywhere.header_mut().set_routing_hash(u64::MAX);
    fixture.push(anywhere);

    assert_eq!(fixture.received_on(&[0, 1]), [3, 4]);
    assert_eq!(fixture.received_on(&[2, 3]), [1, 2, 5, 6]);
}

#[test]
fn quota_is_split_by_effective_cpus() {
    let fixture = Fixture::new(&ALL, |_| {});
    // Socket 0 has three effective CPUs, socket 1 two: five CPUs share a quota of five.
    fixture.source.set(utilization(&[0, 1, 2, 4, 5], 5.0));
    fixture.lattice.start().unwrap();

    assert_eq!(*fixture.clone_on(0).quotas.lock().unwrap(), [1.5]);
    assert_eq!(*fixture.clone_on(1).quotas.lock().unwrap(), [1.5]);
    assert_eq!(*fixture.clone_on(2).quotas.lock().unwrap(), [2.0]);
}

#[test]
fn a_socket_with_every_cpu_disallowed_neither_blocks_startup_nor_takes_frames() {
    let fixture = Fixture::new(&SOCKET_0, |config| {
        config.startup_timeout = Duration::from_secs(5);
    });
    fixture.source.set(utilization(&ALL, 8.0));

    fixture.lattice.start().unwrap();

    let lattice = &fixture.lattice;
    assert_eq!(lattice.active_workers(), 2);
    assert!(lattice.shards[0].is_started());
    assert!(!lattice.shards[1].is_started());
    assert!(
        fixture
            .template
            .spawned()
            .iter()
            .all(|clone| clone.config().core_id() < 2)
    );

    let fates = Fates::default();
    for (id, policy) in [RoutingPolicy::SocketLocal, RoutingPolicy::CacheLocal]
        .into_iter()
        .enumerate()
    {
        fixture.push(from_socket(id as u64, policy, 1));
    }
    fixture.push(hashed(2, u64::MAX));
    fixture.push(Tracked::new(3, &fates));
    assert_eq!(fixture.received_on(&[0, 1]), [0, 1, 2, 3]);
    assert!(fates.lock().unwrap().is_empty());

    assert!(lattice.is_drained());
    let reset = lattice.clear(Duration::from_secs(1), || {}).unwrap();
    assert_eq!(reset.active_workers, 2);
}

#[test]
fn a_socket_without_effective_cpus_neither_blocks_startup_nor_takes_frames() {
    let fixture = Fixture::new(&ALL, |config| {
        config.startup_timeout = Duration::from_secs(5);
    });
    fixture.source.set(utilization(&SOCKET_0, 4.0));

    fixture.lattice.start().unwrap();

    assert_eq!(fixture.lattice.active_workers(), 2);
    assert!(!fixture.lattice.shards[1].is_started());
    fixture.push(from_socket(1, RoutingPolicy::SocketLocal, 1));
    fixture.push(hashed(2, u64::MAX));
    assert_eq!(fixture.received_on(&[0, 1]), [1, 2]);
}

#[test]
fn a_lattice_without_any_effective_cpu_starts_and_fails_frames_cleanly() {
    let fixture = Fixture::new(&ALL, |_| {});
    fixture.source.set(utilization(&[], 0.0));

    fixture.lattice.start().unwrap();

    assert_eq!(fixture.lattice.active_workers(), 0);
    assert!(!fixture.drained());
    let fates = Fates::default();
    let source = Arc::new(LatticeHotSource::new());
    fixture.lattice.add_upstream(source.clone()).unwrap();
    source.accept(Tracked::new(1, &fates));
    let fates = fates.lock().unwrap().clone();
    assert_eq!(fates.len(), 1);
    assert!(matches!(fates[0].1, Fate::Failed(_)));
}

#[test]
fn a_socket_that_loses_every_cpu_is_shut_down_and_frames_move_on() {
    let fixture = Fixture::started(&ALL);
    let retired = [fixture.clone_on(2), fixture.clone_on(3)];

    fixture.publish(&SOCKET_0);

    eventually("the retired clones to close", || {
        retired.iter().all(|clone| clone.is_closed())
    });
    assert!(!fixture.lattice.shards[1].is_started());
    assert!(fixture.lattice.lock().handles[1].is_none());
    assert!(!fixture.clone_on(0).is_closed());
    assert_eq!(fixture.lattice.active_workers(), 2);
    assert!(!fixture.drained());

    fixture.push(from_socket(1, RoutingPolicy::SocketLocal, 1));
    fixture.push(hashed(2, u64::MAX));
    assert_eq!(fixture.received_on(&[0, 1]), [1, 2]);

    fixture.publish(&ALL);
    eventually("the returning shard", || {
        fixture.lattice.shards[1].is_started()
    });
    assert_eq!(fixture.lattice.active_workers(), 4);
    let returned = fixture.clone_on(2);
    assert!(!Arc::ptr_eq(&returned, &retired[0]));
    fixture.push(from_socket(3, RoutingPolicy::CacheLocal, 1));
    assert_eq!(returned.receiver.received(), [3]);
}

#[test]
fn a_core_change_rebalances_only_its_shard() {
    let fixture = Fixture::started(&ALL);
    let removed = fixture.clone_on(3);

    fixture.publish(&[0, 1, 2, 3, 4, 5]);

    eventually("the removed clone to close", || removed.is_closed());
    assert_eq!(fixture.lattice.active_workers(), 3);
    assert!(fixture.lattice.shards[1].is_started());
    assert_eq!(fixture.template.spawned().len(), 4);
    assert!(!fixture.drained());
}

#[test]
fn an_update_of_the_same_topology_updates_every_shard() {
    let fixture = Fixture::started(&ALL);
    fixture.publish(&ALL);
    for core in 0..4 {
        eventually("the update", || {
            fixture.clone_on(core).updates() == [core, core]
        });
    }
    assert_eq!(fixture.template.spawned().len(), 4);
}

#[test]
fn start_needs_a_utilization_and_can_be_retried() {
    let fixture = Fixture::new(&ALL, |_| {});
    assert!(matches!(
        fixture.lattice.start(),
        Err(LatticeError::NoUtilization)
    ));
    assert_eq!(fixture.source.starts.load(Ordering::SeqCst), 0);

    fixture.source.set(utilization(&ALL, 8.0));
    fixture.lattice.start().unwrap();
    assert_eq!(fixture.lattice.active_workers(), 4);
}

#[test]
fn start_times_out_if_a_shard_is_never_ready() {
    let fixture = Fixture::new(&ALL, |config| {
        config.startup_timeout = Duration::from_millis(20);
    });
    fixture.template.ready.store(false, Ordering::SeqCst);
    fixture.source.set(utilization(&ALL, 8.0));

    assert!(matches!(
        fixture.lattice.start(),
        Err(LatticeError::StartTimedOut)
    ));
    assert_eq!(fixture.source.starts.load(Ordering::SeqCst), 0);

    for clone in fixture.template.spawned() {
        clone.ready.store(true, Ordering::SeqCst);
    }
    fixture.lattice.start().unwrap();
    assert_eq!(fixture.source.starts.load(Ordering::SeqCst), 1);
}

#[test]
fn add_upstream_starts_the_lattice_and_feeds_it() {
    let fixture = Fixture::new(&ALL, |_| {});
    fixture.source.set(utilization(&ALL, 8.0));
    let source = Arc::new(LatticeHotSource::new());

    fixture.lattice.add_upstream(source.clone()).unwrap();

    assert_eq!(fixture.lattice.active_workers(), 4);
    source.accept(from_socket(1, RoutingPolicy::CacheLocal, 1));
    source.accept(hashed(2, 0));
    assert_eq!(fixture.received_on(&[2, 3]), [1]);
    assert_eq!(fixture.received_on(&[0, 1]), [2]);
}

#[test]
fn clear_resets_every_shard_while_ingest_is_drained() {
    let fixture = Fixture::started(&ALL);
    for core in 0..4 {
        fixture
            .clone_on(core)
            .buffered
            .store(core as u64 + 1, Ordering::SeqCst);
    }
    let drained_during = AtomicBool::new(false);

    let reset = fixture
        .lattice
        .clear(Duration::from_secs(1), || {
            drained_during.store(fixture.drained(), Ordering::SeqCst);
        })
        .unwrap();

    assert_eq!(
        reset,
        CacheReset {
            cleared_frames: 10,
            active_workers: 4
        }
    );
    assert!(drained_during.load(Ordering::SeqCst));
    assert!(!fixture.drained());
    assert!(!fixture.lattice.resetting.load(Ordering::SeqCst));
}

#[test]
fn clear_rejects_a_zero_timeout_and_a_second_reset() {
    let fixture = Fixture::started(&ALL);
    assert!(matches!(
        fixture.lattice.clear(Duration::ZERO, || {}),
        Err(LatticeError::ZeroTimeout)
    ));

    let nested = Mutex::new(None);
    fixture
        .lattice
        .clear(Duration::from_secs(1), || {
            *nested.lock().unwrap() = Some(fixture.lattice.clear(Duration::from_secs(1), || {}));
        })
        .unwrap();
    assert!(matches!(
        nested.lock().unwrap().take(),
        Some(Err(LatticeError::ResetInProgress))
    ));
}

#[test]
fn clear_before_start_clears_nothing() {
    let fixture = Fixture::new(&ALL, |_| {});
    assert_eq!(
        fixture
            .lattice
            .clear(Duration::from_secs(1), || panic!("not reached"))
            .unwrap(),
        CacheReset {
            cleared_frames: 0,
            active_workers: 0
        }
    );
}

#[test]
fn updates_during_a_clear_are_ignored() {
    let fixture = Fixture::started(&ALL);
    fixture
        .lattice
        .clear(Duration::from_secs(1), || {
            fixture
                .source
                .publish(utilization(&SOCKET_0, SOCKET_0.len() as f64));
        })
        .unwrap();
    assert_eq!(fixture.lattice.active_workers(), 4);
    assert!(!fixture.lattice.is_rebalancing());
}

#[test]
fn clear_that_outlasts_a_rebalance_leaves_ingest_drained() {
    let fixture = Fixture::started(&ALL);
    let stuck = fixture.clone_on(2);
    stuck.drained.store(false, Ordering::SeqCst);
    fixture
        .source
        .publish(utilization(&SOCKET_0, SOCKET_0.len() as f64));
    assert!(fixture.lattice.is_rebalancing());

    let error = fixture
        .lattice
        .clear(Duration::from_millis(5), || panic!("not reached"))
        .unwrap_err();
    assert!(matches!(error, LatticeError::RebalanceTimedOut(_)));
    assert_eq!(
        error.to_string(),
        "timed out waiting for lattice to finish rebalancing"
    );
    assert!(fixture.drained());

    stuck.drained.store(true, Ordering::SeqCst);
    fixture.settle();
    assert!(!fixture.drained());
}

#[test]
fn a_shard_rebalance_that_outlasts_a_clear_fails_it_and_reopens_ingest() {
    let fixture = Fixture::started(&ALL);
    let stuck = fixture.clone_on(3);
    stuck.drained.store(false, Ordering::SeqCst);
    fixture
        .source
        .publish(utilization(&[0, 1, 2, 3, 4, 5], 6.0));
    assert!(!fixture.lattice.is_rebalancing());
    assert!(fixture.lattice.shards[1].is_rebalancing());

    let error = fixture
        .lattice
        .clear(Duration::from_millis(5), || panic!("not reached"))
        .unwrap_err();
    assert!(matches!(error, LatticeError::RebalanceTimedOut(_)));
    assert!(!fixture.drained());

    stuck.drained.store(true, Ordering::SeqCst);
    fixture.settle();
}

#[test]
fn is_drained_waits_for_every_shard() {
    let fixture = Fixture::started(&ALL);
    fixture.clone_on(3).drained.store(false, Ordering::SeqCst);
    assert!(!fixture.lattice.is_drained());
    fixture.clone_on(3).drained.store(true, Ordering::SeqCst);
    assert!(fixture.lattice.is_drained());
}

#[test]
fn close_shuts_everything_down_for_good() {
    let fixture = Fixture::started(&ALL);
    fixture.lattice.close();

    assert!(fixture.source.closed.load(Ordering::SeqCst));
    assert!(
        fixture
            .template
            .spawned()
            .iter()
            .all(|clone| clone.is_closed())
    );
    assert!(
        fixture
            .lattice
            .shards
            .iter()
            .all(|shard| !shard.is_started())
    );
    assert_eq!(fixture.lattice.active_workers(), 0);

    assert!(matches!(fixture.lattice.start(), Err(LatticeError::Closed)));
    let source = Arc::new(LatticeHotSource::new());
    assert!(matches!(
        fixture.lattice.add_upstream(source),
        Err(LatticeError::Closed)
    ));
    assert!(matches!(
        fixture.lattice.clear(Duration::from_secs(1), || {}),
        Err(LatticeError::Closed)
    ));
    fixture.lattice.update(&utilization(&SOCKET_0, 4.0));
    assert_eq!(fixture.template.spawned().len(), 4);
    fixture.lattice.close();
}

#[test]
fn close_waits_for_held_routes_before_closing_shards() {
    let fixture = Fixture::new(&ALL, |config| {
        config.shutdown_timeout = Duration::from_millis(10);
    });
    fixture.source.set(utilization(&ALL, 8.0));
    fixture.lattice.start().unwrap();
    let held = fixture.lattice.controller.hold_routing();

    fixture.lattice.close();
    assert!(
        fixture
            .template
            .spawned()
            .iter()
            .all(|clone| !clone.is_closed())
    );

    drop(held);
    eventually("the shards to close", || {
        fixture
            .template
            .spawned()
            .iter()
            .all(|clone| clone.is_closed())
    });
}

#[test]
fn dropping_the_lattice_closes_it() {
    let fixture = Fixture::started(&ALL);
    let Fixture {
        source,
        template,
        lattice,
    } = fixture;
    drop(lattice);
    assert!(source.closed.load(Ordering::SeqCst));
    assert!(template.spawned().iter().all(|clone| clone.is_closed()));
}

#[test]
fn errors_describe_themselves() {
    assert_eq!(LatticeError::Closed.to_string(), "the lattice is closed");
    assert_eq!(
        LatticeError::NoUtilization.to_string(),
        "no utilization to start the lattice from"
    );
    assert_eq!(
        LatticeError::StartTimedOut.to_string(),
        "timed out waiting for the shards to start"
    );
    assert_eq!(
        LatticeError::Monitor(MonitorError::Closed).to_string(),
        "the utilization source failed: the monitor is closed"
    );
    assert!(
        LatticeError::Monitor(MonitorError::Closed)
            .source()
            .is_some()
    );
    assert_eq!(
        LatticeError::ResetInProgress.to_string(),
        "a lattice reset is already in progress"
    );
    assert_eq!(
        LatticeError::ZeroTimeout.to_string(),
        "a reset needs a positive timeout"
    );
    assert!(LatticeError::Closed.source().is_none());
}

#[test]
fn start_waits_for_shards_that_become_ready_late() {
    let fixture = Fixture::new(&ALL, |_| {});
    fixture.template.ready.store(false, Ordering::SeqCst);
    fixture.source.set(utilization(&ALL, 8.0));
    let template = fixture.template.clone();
    let readier = thread::spawn(move || {
        eventually("every clone", || template.spawned().len() == 4);
        thread::sleep(Duration::from_millis(20));
        for clone in template.spawned() {
            clone.ready.store(true, Ordering::SeqCst);
        }
    });

    fixture.lattice.start().unwrap();
    readier.join().unwrap();
    assert_eq!(fixture.source.starts.load(Ordering::SeqCst), 1);
}

#[test]
fn a_topology_change_during_a_rebalance_waits_for_the_next_update() {
    let fixture = Fixture::started(&ALL);
    let stuck = fixture.clone_on(2);
    stuck.drained.store(false, Ordering::SeqCst);
    fixture
        .source
        .publish(utilization(&SOCKET_0, SOCKET_0.len() as f64));
    assert!(fixture.lattice.is_rebalancing());

    fixture.source.publish(utilization(&[0, 1], 2.0));
    assert_eq!(fixture.lattice.shards[0].active_cores(), 2);

    stuck.drained.store(true, Ordering::SeqCst);
    fixture.settle();
    assert_eq!(fixture.lattice.shards[0].active_cores(), 2);
    fixture.publish(&[0, 1]);
    assert_eq!(fixture.lattice.shards[0].active_cores(), 1);
}

#[test]
fn clear_waits_out_a_rebalance() {
    let fixture = Fixture::started(&ALL);
    let stuck = fixture.clone_on(2);
    stuck.drained.store(false, Ordering::SeqCst);
    fixture
        .source
        .publish(utilization(&SOCKET_0, SOCKET_0.len() as f64));
    let releaser = thread::spawn(move || {
        thread::sleep(Duration::from_millis(20));
        stuck.drained.store(true, Ordering::SeqCst);
    });

    let reset = fixture
        .lattice
        .clear(Duration::from_secs(10), || {})
        .unwrap();
    releaser.join().unwrap();
    assert_eq!(reset.active_workers, 2);
}
