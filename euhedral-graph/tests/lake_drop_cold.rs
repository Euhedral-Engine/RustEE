//! A frame that drops the last reference to a lake on a lattice worker takes no lifecycle lock inside the hot scope.
//!
//! The violation hook is process-wide, so this test has a file, and so a process, of its own.

// The test thread is the harness, not a frame: it polls for what the lattice did.
#![cfg(debug_assertions)]
#![allow(
    clippy::disallowed_methods,
    clippy::disallowed_types,
    reason = "test fixtures: the test thread polls for the lattice and locks its records"
)]

use std::num::NonZeroUsize;
use std::panic::Location;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

use euhedral_core::config::LatticeConfig;
use euhedral_core::control_plane::ControlPlaneLattice;
use euhedral_core::frames::{Frame, RunnableFrame};
use euhedral_core::impls::DefaultExecutor;
use euhedral_data_structures::sync::HotScope;
use euhedral_graph::Lake;
use euhedral_hardware_utils::affinity;
use euhedral_hardware_utils::util::CpuSet;

fn one_cpu_lattice() -> Arc<ControlPlaneLattice> {
    let cpu = affinity::base_mask()
        .first()
        .expect("the process may run on a CPU");
    let mut config = LatticeConfig::with_executor(DefaultExecutor::new());
    config.name = "lake-drop-cold".into();
    config.shard_name = "lake-drop-cold".into();
    config.allowed_cpus = CpuSet::range(cpu, cpu + 1);
    config.startup_timeout = Duration::from_secs(10);
    config.shutdown_timeout = Duration::from_secs(5);
    ControlPlaneLattice::new(config).expect("a lattice")
}

static PLACES: Mutex<Vec<String>> = Mutex::new(Vec::new());

/// Keeps every violation: none is expected.
fn record(at: &'static Location<'static>) {
    PLACES
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .push(at.to_string());
}

#[test]
fn the_last_reference_to_a_lake_dropped_by_a_frame_takes_no_lock_inside_the_hot_scope() {
    let lattice = one_cpu_lattice();
    let lake = Lake::new(lattice.clone(), NonZeroUsize::MIN, Duration::ZERO);
    // The unit holds the lake: the frame that owns the unit owns the last reference once the test lets go.
    let unit = Mutex::new(Some(lake.admit().expect("a unit")));
    let released = Arc::new(AtomicBool::new(false));
    let done = Arc::new(AtomicU64::new(0));
    let frame: Box<dyn Frame> = {
        let (released, done) = (released.clone(), done.clone());
        Box::new(RunnableFrame::new(
            1,
            Arc::new(move || {
                let deadline = Instant::now() + Duration::from_secs(20);
                while !released.load(Ordering::Acquire) {
                    assert!(Instant::now() < deadline, "the test never released");
                    std::thread::sleep(Duration::from_millis(1));
                }
                drop(unit.lock().unwrap_or_else(PoisonError::into_inner).take());
                done.fetch_add(1, Ordering::AcqRel);
            }),
        ))
    };
    let _hook = HotScope::install_violation_hook(record);
    assert!(lake.offer(frame).is_ok());
    drop(lake);
    released.store(true, Ordering::Release);
    let deadline = Instant::now() + Duration::from_secs(20);
    while done.load(Ordering::Acquire) != 1 {
        assert!(Instant::now() < deadline, "timed out waiting for the frame");
        std::thread::sleep(Duration::from_millis(1));
    }
    lattice.close();
    let places = PLACES.lock().unwrap_or_else(PoisonError::into_inner);
    assert!(places.is_empty(), "{places:?}");
}
