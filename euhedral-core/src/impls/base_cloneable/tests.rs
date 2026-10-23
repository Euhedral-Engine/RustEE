use super::*;
use crate::config::{CacheConfig, LatticeConfig};
use crate::control_plane::testing::{PATIENCE, SINGLE_CORE, SMT_CORE, pinned_turn, wait_until};
use crate::control_plane::{CacheDirectory, ControlPlaneLattice};
use crate::flow_control::testing::{Fate, Fates, Tracked, frame};
use crate::flow_control::{LatticeHotSource, LatticeNode, UpstreamRegistry};
use crate::frames::{Frame, FrameError};
use euhedral_hardware_utils::affinity;
use euhedral_hardware_utils::monitor::{HardwareUtilization, HardwareUtilizationParts};
use euhedral_hardware_utils::topology::Topology;
use euhedral_hardware_utils::util::CpuSet;
use std::panic::AssertUnwindSafe;
use std::sync::Mutex;
use std::sync::atomic::Ordering;
use std::thread;

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
    _turn: std::sync::MutexGuard<'static, ()>,
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
    // The executor made for the build is shut down until the fragment starts.
    assert!(PinnedThreadExecutor::get(cpus[0]).unwrap().is_none());
    assert!(!clone.is_started());
    assert!(!clone.ready());
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
fn a_failed_build_panics_and_shuts_its_executor_down() {
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
