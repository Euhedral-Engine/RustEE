//! A lifecycle lock taken by a frame that runs through a lake reaches the violation hook.
//!
//! The hook is process-wide and receives the violations of every thread, so this test has a file, and so a process,
//! of its own.

// The test thread is the harness, not a frame: it polls for what the lattice did.
#![cfg(debug_assertions)]
#![allow(
    clippy::disallowed_methods,
    clippy::disallowed_types,
    reason = "test fixtures: the test thread polls for the lattice and locks its records"
)]

use std::num::NonZeroUsize;
use std::panic::Location;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

use euhedral_core::config::LatticeConfig;
use euhedral_core::control_plane::ControlPlaneLattice;
use euhedral_core::frames::{Frame, RunnableFrame};
use euhedral_core::impls::DefaultExecutor;
use euhedral_data_structures::sync::{HotScope, LifecycleMutex};
use euhedral_graph::Lake;
use euhedral_hardware_utils::affinity;
use euhedral_hardware_utils::util::CpuSet;

/// A lattice of the first CPU this process may run on.
fn one_cpu_lattice() -> Arc<ControlPlaneLattice> {
    let cpu = affinity::base_mask()
        .first()
        .expect("the process may run on a CPU");
    let mut config = LatticeConfig::with_executor(DefaultExecutor::new());
    config.name = "lake-violation".into();
    config.shard_name = "lake-violation".into();
    config.allowed_cpus = CpuSet::range(cpu, cpu + 1);
    config.startup_timeout = Duration::from_secs(10);
    config.shutdown_timeout = Duration::from_secs(5);
    ControlPlaneLattice::new(config).expect("a lattice")
}

fn eventually(what: &str, condition: impl Fn() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(20);
    while !condition() {
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(1));
    }
}

static PLACES: Mutex<Vec<String>> = Mutex::new(Vec::new());

/// Keeps the locations in this file; any other violation is a real one and ends the run.
fn record(at: &'static Location<'static>) {
    if at.file() == file!() {
        PLACES
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(at.to_string());
    } else {
        HotScope::report_default(at);
    }
}

#[test]
fn a_frame_taking_a_lifecycle_lock_through_a_lake_reaches_the_violation_hook() {
    let lattice = one_cpu_lattice();
    let lake = Lake::new(lattice.clone(), NonZeroUsize::MIN, Duration::ZERO);
    let unit = lake.admit().expect("a unit");
    let lifecycle = Arc::new(LifecycleMutex::new(0_u32));
    let done = Arc::new(AtomicU64::new(0));
    let frame: Box<dyn Frame> = {
        let (lifecycle, done) = (lifecycle.clone(), done.clone());
        Box::new(RunnableFrame::new(
            1,
            Arc::new(move || {
                *lifecycle.lock() += 1;
                done.fetch_add(1, Ordering::AcqRel);
            }),
        ))
    };
    {
        let _hook = HotScope::install_violation_hook(record);
        assert!(lake.offer(frame).is_ok());
        eventually("the frame to run", || done.load(Ordering::Acquire) == 1);
    }

    let places = PLACES.lock().unwrap_or_else(PoisonError::into_inner);
    assert_eq!(places.len(), 1, "{places:?}");
    assert!(
        places[0].contains("tests/lake_violation.rs"),
        "{}",
        places[0]
    );
    assert_eq!(*lifecycle.lock(), 1, "the lock proceeds after the report");
    drop(places);
    drop(unit);
    lake.complete_gracefully();
    lattice.close();
}
