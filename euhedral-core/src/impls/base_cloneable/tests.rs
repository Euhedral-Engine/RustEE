use super::*;
use crate::config::FragmentConfig;
use crate::config::{CacheConfig, LatticeConfig};
use crate::control_plane::testing::{
    PATIENCE, PinnedTurn, SINGLE_CORE, SMT_CORE, pinned_turn, wait_until,
};
use crate::control_plane::{CacheDirectory, ControlPlaneLattice};
use crate::flow_control::testing::{Fate, Fates, Tracked, frame};
use crate::flow_control::{LatticeHotSource, LatticeNode, UpstreamRegistry};
use crate::frames::{Frame, FrameError, FrameHeader};
use crate::impls::DefaultExecutor;
use crate::ingest::{ArrayIngestSink, IngestSink};
use crate::metrics::{HistogramSnapshot, MetricsRegistry};
use euhedral_hardware_utils::affinity;
use euhedral_hardware_utils::monitor::{HardwareUtilization, HardwareUtilizationParts};
use euhedral_hardware_utils::topology::Topology;
use euhedral_hardware_utils::util::CpuSet;
use std::num::NonZeroU64;
use std::panic::AssertUnwindSafe;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant};

/// The executor's CPU, the CPU that ran the frame, and the frame.
type Run = (Option<usize>, Option<usize>, u64);

/// What [`Recording`] executors did, shared by a template and its clones.
#[derive(Default)]
struct Log {
    /// The CPU each clone was made for, and the CPU that made it.
    hooks: Mutex<Vec<(Option<usize>, Option<usize>)>>,
    runs: Mutex<Vec<Run>>,
}

/// Records where it was cloned and where it ran each frame.
#[derive(Default)]
struct Recording {
    cpu: Option<usize>,
    log: Arc<Log>,
}

impl AbstractExecutor for Recording {
    fn cpu(&self) -> Option<usize> {
        self.cpu
    }

    fn execute(&self, frame: &mut dyn Frame) -> Result<(), FrameError> {
        let id = frame.header().id_hash();
        self.log
            .runs
            .lock()
            .unwrap()
            .push((self.cpu, affinity::current_cpu(), id));
        Ok(())
    }

    fn hook_on_clone(&self, cpu: Option<usize>) -> Self {
        self.log
            .hooks
            .lock()
            .unwrap()
            .push((cpu, affinity::current_cpu()));
        Self {
            cpu,
            log: self.log.clone(),
        }
    }
}

fn placed(core: usize, cpus: &[usize]) -> CloneConfig {
    let directory = CacheDirectory::new(UpstreamRegistry::new(), Topology::system().clone());
    CloneConfig::new("test", core, cpus.iter().copied().collect()).with_cache_directory(directory)
}

/// Closes `clone` and waits for its workers and their executors.
fn close(clone: &BaseCloneableObject<impl AbstractExecutor>) {
    clone.close();
    let fragment = clone.fragment().unwrap();
    assert!(
        clone
            .liveness_registry()
            .unwrap()
            .await_termination(Instant::now() + PATIENCE)
    );
    assert!(fragment.await_executors(PATIENCE));
}

/// A clone and the template it came from, closed when dropped.
struct Placed {
    log: Arc<Log>,
    clone: Arc<BaseCloneableObject<Recording>>,
    _turn: PinnedTurn,
}

impl Placed {
    fn on(core: usize, cpus: &[usize]) -> Self {
        let turn = pinned_turn();
        let log = Arc::new(Log::default());
        let fragment_config = FragmentConfig::default()
            .with_cache_config(CacheConfig::default().with_max_pooled_chunks(8));
        let template = BaseCloneableObject::new(
            fragment_config,
            Recording {
                cpu: None,
                log: log.clone(),
            },
        );
        let clone = template.place(&placed(core, cpus));
        Self {
            log,
            clone,
            _turn: turn,
        }
    }

    fn smt() -> Option<Self> {
        let (core, cpus) = SMT_CORE.as_ref()?;
        Some(Self::on(*core, &cpus[..2]))
    }

    fn fragment(&self) -> &Arc<ControlPlaneFragment> {
        self.clone.fragment().unwrap()
    }

    fn runs(&self) -> Vec<Run> {
        self.log.runs.lock().unwrap().clone()
    }
}

impl Drop for Placed {
    fn drop(&mut self) {
        if thread::panicking() {
            self.clone.close();
        } else {
            close(&self.clone);
        }
    }
}

#[test]
fn the_template_has_no_fragment_and_nothing_to_run() {
    let template = BaseCloneableObject::default();
    assert_eq!(template.name(), "BaseCloneableObject");
    assert!(template.fragment().is_none());
    assert_eq!(template.executors().len(), 1);
    assert_eq!(template.executors()[0].cpu(), None);
    assert_eq!(template.fragment_config().max_batch_size().get(), 4096);
    template.start();
    assert!(template.is_started());
    assert!(template.ready());
    assert!(template.is_drained());
    template.set_drain_mode(true);
    assert_eq!(template.reset(Instant::now()), 0);
    template.dump_locks();
    assert_eq!(CloneableObject::core(&template), None);
    assert!(template.liveness_registry().is_none());
    assert!(template.output().is_none());
    template.close();
    let template = BaseCloneableObject::with_executor(DefaultExecutor::on_cpu(2));
    assert_eq!(template.executors()[0].cpu(), Some(2));
}

#[test]
fn a_clone_is_built_and_first_touched_on_its_core() {
    let Some(placed) = Placed::smt() else {
        return;
    };
    let (core, cpus) = SMT_CORE.as_ref().unwrap();
    let clone = &placed.clone;
    let fragment = placed.fragment();
    let buddy = fragment.smt_buddy().unwrap();

    assert_eq!(clone.name(), format!("test-pipeline-{core}"));
    assert_eq!(CloneableObject::core(&**clone), Some(*core));
    assert_eq!(
        clone.liveness_registry().unwrap().cpus(),
        &cpus[..2].iter().copied().collect::<CpuSet>()
    );
    assert_eq!((fragment.cpu(), buddy.cpu()), (cpus[0], cpus[1]));
    let executor_cpus: Vec<_> = clone.executors().iter().map(|e| e.cpu()).collect();
    assert_eq!(executor_cpus, [Some(cpus[0]), Some(cpus[1])]);
    // Both executors were cloned on the core's first CPU.
    assert_eq!(
        *placed.log.hooks.lock().unwrap(),
        [
            (Some(cpus[0]), Some(cpus[0])),
            (Some(cpus[1]), Some(cpus[0]))
        ]
    );
    assert!(fragment.cache().pooled_chunks() > 0);
    assert!(buddy.cache().pooled_chunks() > 0);
    assert!(clone.output().is_none());
    // The build leaves the executor to the fragment, which holds it without a task until it starts.
    assert!(PinnedThreadExecutor::get(cpus[0]).unwrap().is_some());
    assert!(!clone.is_started());
    assert!(!clone.ready());
    // Closing the clone ends the last use of the CPU's executor, so no use leaked from the build.
    close(clone);
    wait_until("the CPU's executor to close", || {
        PinnedThreadExecutor::get(cpus[0]).unwrap().is_none()
    });
}

#[test]
fn a_started_clone_runs_each_workers_frames_on_its_executor() {
    let Some(placed) = Placed::smt() else {
        return;
    };
    let (_, cpus) = SMT_CORE.as_ref().unwrap();
    let clone = &placed.clone;
    let directory = placed.fragment().cache().edge().registry().clone();
    let edge = LatticeEdge::new(&directory, Default::default());
    clone.clone().input_edge(edge.clone());
    clone.start();
    wait_until("the clone to get ready", || clone.ready());
    assert!(clone.is_started());
    // A later start leaves the running clone as it is.
    clone.start();

    for id in 0..100u64 {
        let mut frame = frame(id);
        frame.header_mut().set_routing_hash(id + 1000);
        edge.push_downstream(frame);
    }

    wait_until("every frame to run", || placed.runs().len() == 100);
    for (executor, ran_on, id) in placed.runs() {
        let lane = if (id + 1000) % 2 == 1 {
            cpus[1]
        } else {
            cpus[0]
        };
        assert_eq!(executor, Some(lane), "frame {id}");
        assert_eq!(ran_on, Some(lane), "frame {id}");
    }
    wait_until("the clone to drain", || clone.is_drained());
}

#[test]
fn lifecycle_calls_reach_the_fragment() {
    let Some(placed) = Placed::smt() else {
        return;
    };
    let clone = &placed.clone;
    let fragment = placed.fragment();
    let directory = fragment.cache().edge().registry().clone();
    let edge = LatticeEdge::new(&directory, Default::default());
    clone.clone().input_edge(edge.clone());
    assert!(fragment.cache().edge().parent().is_some());

    clone.set_drain_mode(true);
    assert!(fragment.cache().edge().drain_flag().load(Ordering::Acquire));
    clone.set_drain_mode(false);
    assert!(!fragment.cache().edge().drain_flag().load(Ordering::Acquire));

    let full = fragment.batch_cap();
    let topology = Topology::system();
    let utilization = HardwareUtilization::new(HardwareUtilizationParts {
        timestamp_ns: 1,
        quota_cpus: topology.cpu_count() as f64,
        global_effective_cpus: topology.cpu_set().clone(),
        per_quota_cpu_pressure: vec![1.0; topology.cpu_span()],
        ..Default::default()
    });
    let core = fragment.cache().core();
    let cpus = &topology.core_info(core).unwrap().cpus;
    clone.update(&utilization.core_snapshot(core, cpus, cpus.len() as f64));
    assert_eq!(fragment.batch_cap(), 2);

    edge.push_downstream(frame(7));
    assert!(!clone.is_drained());
    assert_eq!(clone.reset(Instant::now()), 1);
    assert!(clone.is_drained());
    assert_eq!(fragment.batch_cap(), full);

    clone.start();
    wait_until("the clone to get ready", || clone.ready());
    clone.dump_locks();
    assert!(
        clone
            .liveness_registry()
            .unwrap()
            .await_termination(Instant::now() + PATIENCE)
    );
    assert!(!clone.is_started());
}

#[test]
fn closing_a_clone_closes_its_fragment() {
    let Some(placed) = Placed::smt() else {
        return;
    };
    let clone = &placed.clone;
    clone.start();
    wait_until("the clone to get ready", || clone.ready());
    close(clone);
    assert!(!clone.is_started());
    assert!(placed.fragment().cache().is_closed());
}

#[test]
fn a_clone_without_a_cpu_cannot_be_built() {
    let template = BaseCloneableObject::default();
    let config = CloneConfig::new("test", 4, CpuSet::new());
    let built = panic::catch_unwind(AssertUnwindSafe(|| template.clone_object(&config)));
    let message = built.err().unwrap();
    let message = message.downcast_ref::<String>().unwrap();
    assert_eq!(message, "core 4 has no effective CPU");
}

#[test]
fn a_failed_build_panics_and_releases_its_executor() {
    let Some((core, cpu)) = *SINGLE_CORE else {
        return;
    };
    let _turn = pinned_turn();
    let template = BaseCloneableObject::default();
    let unplaced = CloneConfig::new("test", core, CpuSet::from_iter([cpu]));
    let built = panic::catch_unwind(AssertUnwindSafe(|| template.clone_object(&unplaced)));
    let message = built.err().unwrap();
    let message = message.downcast_ref::<String>().unwrap();
    assert!(
        message.starts_with(&format!("cannot build the fragment of core {core}")),
        "{message}"
    );
    assert!(PinnedThreadExecutor::get(cpu).unwrap().is_none());
}

#[test]
fn a_running_pinned_executor_is_reused_and_left_running() {
    let Some((core, cpu)) = *SINGLE_CORE else {
        return;
    };
    let _turn = pinned_turn();
    let running = PinnedThreadExecutor::get_or_create(cpu, "already-running").unwrap();
    let clone = BaseCloneableObject::default().place(&placed(core, &[cpu]));
    assert!(!running.is_shutdown());
    running.close();
    close(&clone);
}

#[test]
fn the_default_lattice_config_runs_base_cloneable_objects() {
    let Some((core, cpu)) = *SINGLE_CORE else {
        return;
    };
    let _turn = pinned_turn();
    let template = LatticeConfig::default().cloneable;
    let clone = template.clone_object(&placed(core, &[cpu]));
    assert_eq!(clone.core(), Some(core));
    assert!(!clone.is_started());
    assert!(clone.liveness_registry().is_some());
    clone.close();
}

#[test]
fn a_default_lattice_runs_frames_end_to_end() {
    let Some((_, cpus)) = SMT_CORE.as_ref() else {
        return;
    };
    let _turn = pinned_turn();
    let log = Arc::new(Log::default());
    let mut config = LatticeConfig::with_executor(Recording {
        cpu: None,
        log: log.clone(),
    });
    config.name = "end-to-end".into();
    config.allowed_cpus = cpus.iter().copied().collect();
    config.startup_timeout = PATIENCE;
    config.shutdown_timeout = PATIENCE;
    let lattice = ControlPlaneLattice::new(config).unwrap();
    let source = Arc::new(LatticeHotSource::new());
    lattice.add_upstream(source.clone()).unwrap();
    if lattice.active_workers() == 0 {
        lattice.close();
        return;
    }

    let fates = Fates::default();
    for id in 0..200 {
        source.accept(Tracked::new(id, &fates));
    }

    wait_until("every frame to finish", || {
        fates.lock().unwrap().len() == 200
    });
    assert!(
        fates
            .lock()
            .unwrap()
            .iter()
            .all(|(_, fate)| *fate == Fate::Finished)
    );
    let runs = log.runs.lock().unwrap().clone();
    assert_eq!(runs.len(), 200);
    assert!(
        runs.iter()
            .all(|(executor, ran_on, _)| executor == ran_on && cpus.contains(&ran_on.unwrap()))
    );
    lattice.close();
}

/// Waits in `execute` until released (or 2 s pass), like a frame whose source stalls.
struct Blocking {
    header: FrameHeader,
    started: Arc<AtomicBool>,
    release: Arc<AtomicBool>,
}

impl Frame for Blocking {
    fn header(&self) -> &FrameHeader {
        &self.header
    }

    fn header_mut(&mut self) -> &mut FrameHeader {
        &mut self.header
    }

    fn execute(&mut self) -> Result<(), FrameError> {
        self.started.store(true, Ordering::Release);
        let deadline = Instant::now() + Duration::from_secs(2);
        while !self.release.load(Ordering::Acquire) && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(1));
        }
        Ok(())
    }
}

/// Counts itself when run; with `stream`, also checks that the frames of its id run one at a
/// time and in sequence, and records the CPUs they ran on. Without, it works `work` first.
struct Counted {
    header: FrameHeader,
    seq: u64,
    stream: Option<Arc<Stream>>,
    work: Duration,
    done: Arc<AtomicU64>,
}

#[derive(Default)]
struct Stream {
    running: AtomicBool,
    next: AtomicU64,
    faults: AtomicU64,
    cpus: Mutex<Vec<Option<usize>>>,
}

impl Frame for Counted {
    fn header(&self) -> &FrameHeader {
        &self.header
    }

    fn header_mut(&mut self) -> &mut FrameHeader {
        &mut self.header
    }

    fn execute(&mut self) -> Result<(), FrameError> {
        if let Some(stream) = &self.stream {
            let overlapped = stream.running.swap(true, Ordering::AcqRel);
            let in_turn = stream.next.load(Ordering::Acquire) == self.seq;
            let spin = Instant::now();
            while spin.elapsed() < Duration::from_micros(5) {
                std::hint::spin_loop();
            }
            if overlapped || !in_turn {
                stream.faults.fetch_add(1, Ordering::AcqRel);
            }
            stream.next.store(self.seq + 1, Ordering::Release);
            stream.running.store(false, Ordering::Release);
            let cpu = affinity::current_cpu();
            let mut cpus = stream.cpus.lock().unwrap();
            if !cpus.contains(&cpu) {
                cpus.push(cpu);
            }
        } else {
            let spin = Instant::now();
            while spin.elapsed() < self.work {
                std::hint::spin_loop();
            }
        }
        self.done.fetch_add(1, Ordering::AcqRel);
        Ok(())
    }
}

/// A started lattice of `fragment` workers on `cpus` and its source, or `None` if no worker started.
fn running_lattice(
    fragment: FragmentConfig,
    cpus: &[usize],
) -> Option<(Arc<ControlPlaneLattice>, Arc<LatticeHotSource>)> {
    let mut config = LatticeConfig::new(Arc::new(BaseCloneableObject::new(
        fragment,
        DefaultExecutor::new(),
    )));
    config.name = "poll-mode".into();
    config.allowed_cpus = cpus.iter().copied().collect();
    config.startup_timeout = PATIENCE;
    config.shutdown_timeout = PATIENCE;
    let lattice = ControlPlaneLattice::new(config).unwrap();
    let source = Arc::new(LatticeHotSource::new());
    lattice.add_upstream(source.clone()).unwrap();
    if lattice.active_workers() == 0 {
        lattice.close();
        return None;
    }
    Some((lattice, source))
}

/// Sends `count` unordered frames through `source` and waits until they have all run.
fn run_counted(source: &LatticeHotSource, count: u64) {
    let done = Arc::new(AtomicU64::new(0));
    for seq in 0..count {
        let mut header = FrameHeader::new(0x70);
        header.randomize_hash(seq + 1);
        source.accept(Box::new(Counted {
            header,
            seq,
            stream: None,
            work: Duration::ZERO,
            done: done.clone(),
        }));
    }
    wait_until("the frames to run", || {
        done.load(Ordering::Acquire) == count
    });
}

#[test]
fn two_lattices_on_one_cpu_close_independently() {
    let Some((_, cpu)) = *SINGLE_CORE else {
        return;
    };
    let _turn = pinned_turn();
    let Some((first, first_source)) = running_lattice(FragmentConfig::default(), &[cpu]) else {
        return;
    };
    let Some((second, second_source)) = running_lattice(FragmentConfig::default(), &[cpu]) else {
        first.close();
        return;
    };
    run_counted(&first_source, 50);
    run_counted(&second_source, 50);

    // Closing one lattice leaves the other's worker on the shared executor running.
    first.close();
    assert_eq!(second.active_workers(), 1);
    run_counted(&second_source, 50);

    // A lattice that starts while the other runs gets a working executor too.
    let third = running_lattice(FragmentConfig::default(), &[cpu]);
    if let Some((third, third_source)) = &third {
        run_counted(third_source, 50);
        run_counted(&second_source, 50);
        third.close();
    }
    run_counted(&second_source, 50);
    second.close();

    // The last lattice to close took the executor with it.
    wait_until("the CPU's executor to close", || {
        PinnedThreadExecutor::get(cpu).unwrap().is_none()
    });
}

fn polling() -> FragmentConfig {
    FragmentConfig::default()
        .with_max_batch_size(NonZeroU64::new(FragmentConfig::POLL_MODE_MAX_BATCH).unwrap())
}

#[test]
fn a_polling_lattice_runs_the_frames_queued_behind_one_that_blocks() {
    let Some((_, cpus)) = SMT_CORE.as_ref() else {
        return;
    };
    let _turn = pinned_turn();
    let Some((lattice, source)) = running_lattice(polling(), cpus) else {
        return;
    };
    let started = Arc::new(AtomicBool::new(false));
    let release = Arc::new(AtomicBool::new(false));
    // Ordered with an even hash: the core's first worker takes it and blocks.
    source.accept(Box::new(Blocking {
        header: FrameHeader::new(0x10),
        started: started.clone(),
        release: release.clone(),
    }));
    wait_until("the blocker to start", || started.load(Ordering::Acquire));

    let done = Arc::new(AtomicU64::new(0));
    let sent = Instant::now();
    for seq in 0..512 {
        let mut header = FrameHeader::new(0x20);
        header.randomize_hash(seq + 1);
        source.accept(Box::new(Counted {
            header,
            seq,
            stream: None,
            work: Duration::ZERO,
            done: done.clone(),
        }));
    }
    let deadline = sent + Duration::from_secs(1);
    while done.load(Ordering::Acquire) < 512 && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(1));
    }
    let (ran, took) = (done.load(Ordering::Acquire), sent.elapsed());
    release.store(true, Ordering::Release);
    lattice.close();
    assert_eq!(
        ran, 512,
        "{ran} of 512 frames ran within {took:?} of a 2 s blocker"
    );
}

#[test]
fn a_polling_lattice_runs_ordered_frames_on_their_own_cpu_in_sequence_while_others_steal() {
    let Some((_, cpus)) = SMT_CORE.as_ref() else {
        return;
    };
    let _turn = pinned_turn();
    let Some((lattice, source)) = running_lattice(polling(), cpus) else {
        return;
    };
    // An even id belongs to the core's first CPU, an odd one to its SMT buddy. Between them come
    // unordered frames of 20 us that keep both workers stealing from each other.
    let streams = [Arc::new(Stream::default()), Arc::new(Stream::default())];
    let done = Arc::new(AtomicU64::new(0));
    let per_stream = 2000;
    for seq in 0..per_stream {
        for (id, stream) in [0x40, 0x41].into_iter().zip(&streams) {
            source.accept(Box::new(Counted {
                header: FrameHeader::new(id),
                seq,
                stream: Some(stream.clone()),
                work: Duration::ZERO,
                done: done.clone(),
            }));
        }
        for k in 0..2 {
            let mut header = FrameHeader::new(0x50);
            header.randomize_hash(2 * seq + k + 1);
            source.accept(Box::new(Counted {
                header,
                seq,
                stream: None,
                work: Duration::from_micros(20),
                done: done.clone(),
            }));
        }
    }
    wait_until("every frame to run", || {
        done.load(Ordering::Acquire) == 4 * per_stream
    });
    lattice.close();
    for (stream, &owner) in streams.iter().zip(cpus) {
        assert_eq!(
            stream.faults.load(Ordering::Acquire),
            0,
            "out of turn or overlapping"
        );
        assert_eq!(stream.next.load(Ordering::Acquire), per_stream);
        assert_eq!(
            *stream.cpus.lock().unwrap(),
            [Some(owner)],
            "ran off its owner's CPU"
        );
    }
}

/// The CPUs of the host's last two cores whose CPUs this process may all run on.
fn two_cores() -> Option<[(usize, Vec<usize>); 2]> {
    let allowed = affinity::base_mask();
    let mut cores = Topology::system()
        .core_infos()
        .iter()
        .rev()
        .filter_map(|core| {
            let cpus: Vec<usize> = core.cpus.iter().collect();
            cpus.iter()
                .all(|&cpu| allowed.contains(cpu))
                .then_some((core.core, cpus))
        });
    Some([cores.next()?, cores.next()?])
}

/// Records the CPU it ran on.
struct Located {
    header: FrameHeader,
    ran_on: Arc<Mutex<Vec<Option<usize>>>>,
}

impl Frame for Located {
    fn header(&self) -> &FrameHeader {
        &self.header
    }

    fn header_mut(&mut self) -> &mut FrameHeader {
        &mut self.header
    }

    fn execute(&mut self) -> Result<(), FrameError> {
        self.ran_on.lock().unwrap().push(affinity::current_cpu());
        Ok(())
    }
}

/// An unordered frame bound to `cpu`'s core.
fn cache_local_header(id: u64, seed: u64, cpu: usize) -> FrameHeader {
    let mut header = FrameHeader::new(id);
    header.randomize_hash(seed);
    header.set_origin(Topology::system().cpu_info(cpu).copied());
    header.set_routing_policy(crate::flow_control::RoutingPolicy::CacheLocal);
    header
}

#[test]
fn cache_local_frames_behind_a_blocking_one_stay_on_their_core_in_both_modes() {
    let Some([(_, home), (_, other)]) = two_cores() else {
        return;
    };
    let cpus: Vec<usize> = home.iter().chain(&other).copied().collect();
    for fragment in [FragmentConfig::default(), polling()] {
        let _turn = pinned_turn();
        let Some((lattice, source)) = running_lattice(fragment, &cpus) else {
            return;
        };
        // The test is about frames whose core is active. A core joins the routing when its clone is ready, which can
        // follow the start of the lattice; until then its cache-local frames fall back to a hashed core, as they
        // should. Send nothing before both cores are in, and skip a host where they never are.
        let deadline = Instant::now() + PATIENCE;
        while lattice.active_workers() < 2 && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(1));
        }
        if lattice.active_workers() < 2 {
            lattice.close();
            return;
        }
        let started = Arc::new(AtomicBool::new(false));
        let release = Arc::new(AtomicBool::new(false));
        source.accept(Box::new(Blocking {
            header: cache_local_header(0x10, 1, home[0]),
            started: started.clone(),
            release: release.clone(),
        }));
        wait_until("the blocker to start", || started.load(Ordering::Acquire));
        let ran_on = Arc::new(Mutex::new(Vec::new()));
        for seq in 0..64 {
            source.accept(Box::new(Located {
                header: cache_local_header(0x20, seq + 2, home[0]),
                ran_on: ran_on.clone(),
            }));
        }
        // The other core idles and steals what it may; give it the time.
        thread::sleep(Duration::from_millis(100));
        release.store(true, Ordering::Release);
        wait_until("every frame to run", || ran_on.lock().unwrap().len() == 64);
        lattice.close();
        let ran_on = ran_on.lock().unwrap().clone();
        assert!(
            ran_on
                .iter()
                .all(|cpu| cpu.is_some_and(|cpu| home.contains(&cpu))),
            "ran off core {home:?}: {ran_on:?}"
        );
    }
}

fn counted(seq: u64, done: &Arc<AtomicU64>) -> Box<dyn Frame> {
    let mut header = FrameHeader::new(0x30);
    header.randomize_hash(seq + 1);
    Box::new(Counted {
        header,
        seq,
        stream: None,
        work: Duration::ZERO,
        done: done.clone(),
    })
}

#[test]
fn a_metered_lattice_reports_every_workers_meters_until_closed() {
    let Some((core, cpus)) = SMT_CORE.as_ref() else {
        return;
    };
    let _turn = pinned_turn();
    let registry = MetricsRegistry::new();
    let fragment = FragmentConfig::default().with_metrics("lattice", registry.clone());
    let Some((lattice, source)) = running_lattice(fragment, &cpus[..2]) else {
        return;
    };
    assert!(Arc::ptr_eq(&lattice.metrics_registry().unwrap(), &registry));
    let done = Arc::new(AtomicU64::new(0));
    for seq in 0..1000 {
        source.accept(counted(seq, &done));
    }
    wait_until("every frame to run", || {
        done.load(Ordering::Acquire) == 1000
    });

    let snapshot = lattice.metrics().unwrap();
    let frames: u64 = snapshot
        .frames_counters()
        .map(|meter| meter.counter().unwrap())
        .sum();
    assert_eq!(frames, 1000);
    let timed: u64 = snapshot
        .service_time_summaries()
        .map(|meter| meter.summary().unwrap().count)
        .sum();
    assert_eq!(timed, 1000);
    // Both workers of the core, and both caches.
    for meters in [
        snapshot.service_time_summaries().collect::<Vec<_>>(),
        snapshot.throughput_summaries().collect(),
        snapshot.in_progress_gauges().collect(),
        snapshot.cache_backlog_gauges().collect(),
        snapshot.cap_factor_summaries().collect(),
    ] {
        let mut tagged: Vec<_> = meters.iter().map(|meter| (meter.core, meter.cpu)).collect();
        tagged.sort_unstable();
        assert_eq!(tagged, [(*core, cpus[0]), (*core, cpus[1])]);
        assert!(
            meters
                .iter()
                .all(|meter| meter.name.starts_with("lattice."))
        );
    }

    lattice.close();
    assert!(registry.is_empty());
    assert!(lattice.metrics().unwrap().meters().is_empty());
}

/// Spins for `execute` in its body and for `finally` in its finalizer.
struct SlowFinalizer {
    header: FrameHeader,
    execute: Duration,
    finally: Duration,
    done: Arc<AtomicU64>,
}

fn spin(duration: Duration) {
    let start = Instant::now();
    while start.elapsed() < duration {
        std::hint::spin_loop();
    }
}

impl Frame for SlowFinalizer {
    fn header(&self) -> &FrameHeader {
        &self.header
    }

    fn header_mut(&mut self) -> &mut FrameHeader {
        &mut self.header
    }

    fn execute(&mut self) -> Result<(), FrameError> {
        spin(self.execute);
        Ok(())
    }

    fn do_finally(self: Box<Self>) {
        spin(self.finally);
        self.done.fetch_add(1, Ordering::AcqRel);
    }
}

#[test]
fn the_service_time_is_execute_alone_not_the_finalizer() {
    let Some((_, cpus)) = SMT_CORE.as_ref() else {
        return;
    };
    let _turn = pinned_turn();
    let registry = MetricsRegistry::new();
    let fragment = FragmentConfig::default().with_metrics("body", registry.clone());
    let Some((lattice, source)) = running_lattice(fragment, &cpus[..2]) else {
        return;
    };
    let done = Arc::new(AtomicU64::new(0));
    for seq in 0..20 {
        let mut header = FrameHeader::new(0x40);
        header.randomize_hash(seq + 1);
        source.accept(Box::new(SlowFinalizer {
            header,
            execute: Duration::from_micros(50),
            finally: Duration::from_millis(2),
            done: done.clone(),
        }));
    }
    wait_until("every frame to finish", || {
        done.load(Ordering::Acquire) == 20
    });
    let snapshot = lattice.metrics().unwrap();
    let mut service_time = HistogramSnapshot::empty();
    for meter in snapshot.service_time_summaries() {
        service_time.merge(meter.summary().unwrap());
    }
    assert_eq!(service_time.count, 20);
    // Each execute spun 50 µs; the 2 ms finalizers are not in it. The mean, not the max, carries the claim: one
    // preempted sample on a shared runner can exceed 1 ms, while finalizers in every sample would put the mean past 2 ms.
    assert!(service_time.sum / 20.0 >= 50_000.0, "{service_time:?}");
    assert!(service_time.sum / 20.0 < 1_000_000.0, "{service_time:?}");
    lattice.close();
}

/// The median of `samples`.
fn median(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(f64::total_cmp);
    samples[samples.len() / 2]
}

/// The wall time per frame of an SMT core's two workers running empty frames from an array
/// sink, with metrics off and on, in greedy and in poll mode. Rounds alternate off and on.
#[test]
#[ignore = "a measurement: run with --ignored --nocapture"]
fn metrics_overhead_per_frame() {
    const FRAMES: u64 = 200_000;
    const ROUNDS: usize = 7;
    let Some((_, cpus)) = SMT_CORE.as_ref() else {
        return;
    };
    let _turn = pinned_turn();
    for (mode, base) in [("greedy", FragmentConfig::default()), ("poll", polling())] {
        let (mut off, mut on) = (Vec::new(), Vec::new());
        for round in 0..ROUNDS * 2 {
            let metered = round % 2 == 1;
            let fragment = if metered {
                base.clone().with_metrics("bench", MetricsRegistry::new())
            } else {
                base.clone()
            };
            let Some((lattice, _source)) = running_lattice(fragment, &cpus[..2]) else {
                return;
            };
            let done = Arc::new(AtomicU64::new(0));
            let frames: Vec<_> = (0..FRAMES).map(|seq| counted(seq, &done)).collect();
            let sink = ArrayIngestSink::new(frames);
            let start = Instant::now();
            lattice.add_upstream(sink.delegate()).unwrap();
            let deadline = start + PATIENCE;
            while done.load(Ordering::Acquire) < FRAMES && Instant::now() < deadline {
                thread::sleep(Duration::from_micros(100));
            }
            let per_frame = start.elapsed().as_nanos() as f64 / FRAMES as f64;
            assert_eq!(done.load(Ordering::Acquire), FRAMES, "timed out");
            if metered {
                let snapshot = lattice.metrics().unwrap();
                let ran: u64 = snapshot
                    .frames_counters()
                    .map(|meter| meter.counter().unwrap())
                    .sum();
                assert_eq!(ran, FRAMES);
                on.push(per_frame);
            } else {
                off.push(per_frame);
            }
            lattice.close();
        }
        let (off, on) = (median(off), median(on));
        eprintln!(
            "{mode}: {off:.1} ns/frame off, {on:.1} ns/frame on, {:+.1} ns ({:+.1}%)",
            on - off,
            (on - off) / off * 100.0
        );
    }
}

/// Records whether it was a long call (unordered), the CPU it ran on (`affinity::current_cpu`: the OS's processor number when the
/// platform reports one the topology knows, else the CPU the worker was bound to) and the name of the thread that ran it.
type Ran = Arc<Mutex<Vec<(bool, Option<usize>, String)>>>;

struct Classified {
    header: FrameHeader,
    ran: Ran,
}

impl Frame for Classified {
    fn header(&self) -> &FrameHeader {
        &self.header
    }

    fn header_mut(&mut self) -> &mut FrameHeader {
        &mut self.header
    }

    fn execute(&mut self) -> Result<(), FrameError> {
        let long = !self.header.is_ordered();
        self.ran.lock().unwrap().push((
            long,
            affinity::current_cpu(),
            thread::current().name().unwrap_or("?").to_owned(),
        ));
        // Work to do, so that every core that can take a frame is busy and the others look for more.
        let until = Instant::now() + Duration::from_micros(20);
        while Instant::now() < until {
            std::hint::spin_loop();
        }
        Ok(())
    }
}

/// A worker's pull from an ingest sink takes frames straight from the source, without the lattice's routing, so the
/// interceptor's pull places them: a long call (an unordered frame) belongs on the cores of set A, an ordered frame on
/// those of set B. The sink is full of long calls first, which are what the cores pull directly. Every core also steals,
/// and in poll mode a stealer looks at one frame at a time while others are being queued: it must leave every frame
/// placed on another core, also one that arrives as it looks.
#[test]
fn a_placement_policy_also_places_the_frames_a_worker_pulls_from_a_sink() {
    use crate::control_plane::PlacementPicker;
    use crate::ingest::QueueIngestSink;

    const FRAMES: u64 = if cfg!(miri) { 40 } else { 6_000 };
    let Some([(_, set_a), (_, set_b)]) = two_cores() else {
        return;
    };
    let cpus: Vec<usize> = set_a.iter().chain(&set_b).copied().collect();
    for fragment in [FragmentConfig::default(), polling()] {
        let _turn = pinned_turn();
        let fragment_label = format!("{fragment:?}");
        let mut config = LatticeConfig::new(Arc::new(BaseCloneableObject::new(
            fragment,
            DefaultExecutor::new(),
        )));
        config.name = "placed".into();
        config.allowed_cpus = cpus.iter().copied().collect();
        config.startup_timeout = PATIENCE;
        config.shutdown_timeout = PATIENCE;
        let lattice = ControlPlaneLattice::new(config).unwrap();
        lattice
            .set_placement_policy(
                PlacementPicker::new(
                    Topology::system().clone(),
                    |frame: &dyn Frame| !frame.header().is_ordered(),
                    set_a.iter().copied().collect(),
                    set_b.iter().copied().collect(),
                )
                .into_policy(),
            )
            .unwrap();
        let sink = QueueIngestSink::new();
        let ran = Ran::default();
        lattice.add_upstream(sink.delegate()).unwrap();
        // The frames are offered once both cores are active: a frame placed on a core the routing does not have yet goes to a
        // core it has (a cache-local frame's fallback), and this test asserts where every frame ran.
        let deadline = Instant::now() + PATIENCE;
        while lattice.active_workers() < 2 && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(1));
        }
        if lattice.active_workers() < 2 {
            lattice.close();
            return;
        }
        for id in 0..FRAMES {
            let mut header = FrameHeader::new(id + 1);
            // The long calls first: they are what the cores pull directly.
            if id < FRAMES / 2 {
                header.randomize_hash(id + 7);
            }
            sink.offer(Box::new(Classified {
                header,
                ran: ran.clone(),
            }))
            .ok()
            .expect("the sink takes the frame");
        }
        wait_until("every frame to run", || {
            ran.lock().unwrap().len() == FRAMES as usize
        });
        let active = lattice.active_workers();
        lattice.close();
        let ran = ran.lock().unwrap().clone();
        // What a failure prints: the sets, the lattice and the platform, and where each kind of frame ran, by CPU and thread.
        let mut seen = std::collections::BTreeMap::new();
        for (long, cpu, thread) in &ran {
            *seen.entry((*long, *cpu, thread.clone())).or_insert(0_usize) += 1;
        }
        let topology = Topology::system();
        let dump = format!(
            "fragment {fragment_label}\nset A {set_a:?}, set B {set_b:?}, active workers {active}\naffinity {:?}, base mask {:?}, topology: {} cpus, {} cores\n(long call, cpu, thread) -> frames: {seen:#?}",
            affinity::capability(),
            affinity::base_mask(),
            topology.cpu_count(),
            topology.core_count(),
        );
        let on = |set: &[usize], cpu: Option<usize>| cpu.is_some_and(|cpu| set.contains(&cpu));
        assert_eq!(
            ran.iter().filter(|(long, _, _)| *long).count(),
            FRAMES as usize / 2
        );
        // Placement is exact: not one frame runs off the set the policy placed it on.
        let misplaced = |kind: bool, set: &[usize]| {
            ran.iter()
                .filter(|(long, cpu, _)| *long == kind && !on(set, *cpu))
                .count()
        };
        assert_eq!(
            (misplaced(true, &set_a), misplaced(false, &set_b)),
            (0, 0),
            "(long calls off set A, ordered frames off set B)\n{dump}"
        );
    }
}
