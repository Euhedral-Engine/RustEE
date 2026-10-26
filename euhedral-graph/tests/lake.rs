//! Lakes and latches on a real lattice of one CPU.
//!
//! CI runners have two cores, so nothing here assumes more than one: the lattice is given a single
//! CPU, one the process may run on, and the tests assert structure (order, attachment, counts), not
//! simultaneity.

// The test thread is the harness, not a frame: it polls for what the lattice did, and waits on a
// latch, with sleeps no frame may make.
#![allow(clippy::disallowed_methods)]

use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

use euhedral_core::config::LatticeConfig;
use euhedral_core::control_plane::ControlPlaneLattice;
use euhedral_core::frames::{Frame, FrameError, FrameHeader, RunnableFrame};
use euhedral_core::generics::FrameSink;
use euhedral_core::impls::DefaultExecutor;
use euhedral_graph::{Lake, Latch, LatchSetter};
use euhedral_hardware_utils::affinity;
use euhedral_hardware_utils::util::CpuSet;

/// Lattices of one CPU that run at the same time rebalance each other (a rebalance panicked with
/// "cannot restart the executor" when the tests ran in parallel), so the tests that start one take
/// turns.
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

fn take_turn() -> MutexGuard<'static, ()> {
    ONE_AT_A_TIME.lock().unwrap_or_else(PoisonError::into_inner)
}

/// A lattice of the first CPU this process may run on.
fn one_cpu_lattice() -> Arc<ControlPlaneLattice> {
    let cpu = affinity::base_mask()
        .first()
        .expect("the process may run on a CPU");
    let mut config = LatticeConfig::with_executor(DefaultExecutor::new());
    config.name = "lake-test".into();
    config.shard_name = "lake-test".into();
    config.allowed_cpus = CpuSet::range(cpu, cpu + 1);
    config.startup_timeout = Duration::from_secs(10);
    config.shutdown_timeout = Duration::from_secs(5);
    ControlPlaneLattice::new(config).expect("a lattice")
}

fn sinks(count: usize) -> NonZeroUsize {
    NonZeroUsize::new(count).expect("at least one sink")
}

/// Polls `condition` until it holds; the lattice is running it on another thread.
fn eventually(what: &str, condition: impl Fn() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(20);
    while !condition() {
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(1));
    }
}

/// A frame of ordered lane `hash` that counts itself in `ran` and notes `value` in `seen`.
fn noting(
    hash: u64,
    value: u64,
    seen: &Arc<Mutex<Vec<u64>>>,
    ran: &Arc<AtomicU64>,
) -> Box<dyn Frame> {
    let (seen, ran) = (seen.clone(), ran.clone());
    Box::new(RunnableFrame::new(
        hash,
        Arc::new(move || {
            seen.lock()
                .unwrap_or_else(PoisonError::into_inner)
                .push(value);
            ran.fetch_add(1, Ordering::AcqRel);
        }),
    ))
}

fn counting(hash: u64, ran: &Arc<AtomicU64>) -> Box<dyn Frame> {
    let ran = ran.clone();
    Box::new(RunnableFrame::new(
        hash,
        Arc::new(move || {
            ran.fetch_add(1, Ordering::AcqRel);
        }),
    ))
}

#[test]
fn frames_of_one_hash_keep_order() {
    let _turn = take_turn();
    const LANE: u64 = 0x1a4e;
    const FRAMES: u64 = 500;
    let lattice = one_cpu_lattice();
    let lake = Lake::new(lattice.clone(), sinks(3), Duration::ZERO);
    let unit = lake.admit().expect("a unit");

    let seen = Arc::new(Mutex::new(Vec::new()));
    let ran = Arc::new(AtomicU64::new(0));
    let noise = Arc::new(Mutex::new(Vec::new()));
    for value in 0..FRAMES {
        assert!(lake.offer(noting(LANE, value, &seen, &ran)).is_ok());
        // Other lanes, other sinks, interleaved with the one under test.
        assert!(
            lake.offer(noting(LANE + 1 + value % 5, value, &noise, &ran))
                .is_ok()
        );
    }
    eventually("every frame to run", || {
        ran.load(Ordering::Acquire) == 2 * FRAMES
    });

    let seen = seen.lock().unwrap_or_else(PoisonError::into_inner);
    assert_eq!(*seen, (0..FRAMES).collect::<Vec<_>>());
    drop(unit);
    lake.complete_gracefully();
    lattice.close();
}

#[test]
fn sinks_detach_after_idle_and_reattach_on_admit() {
    let _turn = take_turn();
    const IDLE: Duration = Duration::from_millis(30);
    let lattice = one_cpu_lattice();
    let lake = Lake::new(lattice.clone(), sinks(2), IDLE);
    assert!(
        !lake.is_attached(),
        "nothing attaches before the first unit"
    );
    assert_eq!(lake.attachments(), 0);

    let ran = Arc::new(AtomicU64::new(0));
    let unit = lake.admit().expect("a unit");
    assert!(lake.is_attached());
    assert_eq!(lake.attachments(), 1);
    assert!(lake.offer(counting(7, &ran)).is_ok());
    eventually("the first frame to run", || {
        ran.load(Ordering::Acquire) == 1
    });

    // While a unit is admitted the sinks stay, however long it takes.
    std::thread::sleep(IDLE * 4);
    assert!(lake.is_attached(), "a unit is admitted");
    assert_eq!(lake.attachments(), 1);

    // The last unit ends: after the idle time the watch detaches the set.
    drop(unit);
    eventually("the sinks to detach", || !lake.is_attached());
    assert_eq!(lake.active(), 0);
    assert!(!lake.is_complete(), "a detached lake can take units again");

    // The next unit attaches a fresh set, which runs frames.
    let unit = lake.admit().expect("a unit after the detach");
    assert!(lake.is_attached());
    assert_eq!(lake.attachments(), 2);
    assert!(lake.offer(counting(7, &ran)).is_ok());
    eventually("the second frame to run", || {
        ran.load(Ordering::Acquire) == 2
    });

    drop(unit);
    lake.complete_gracefully();
    lattice.close();
}

#[test]
fn a_lake_with_no_idle_time_never_detaches() {
    let _turn = take_turn();
    let lattice = one_cpu_lattice();
    let lake = Lake::new(lattice.clone(), sinks(1), Duration::ZERO);
    drop(lake.admit().expect("a unit"));
    std::thread::sleep(Duration::from_millis(100));
    assert!(lake.is_attached());
    assert_eq!(lake.attachments(), 1);
    lake.complete_gracefully();
    lattice.close();
}

#[test]
fn offer_without_admitted_unit_is_refused() {
    let _turn = take_turn();
    let lattice = one_cpu_lattice();
    let lake = Lake::new(lattice.clone(), sinks(2), Duration::ZERO);
    let ran = Arc::new(AtomicU64::new(0));

    // Never admitted: the same frame comes back, and nothing attached.
    let refused = lake
        .offer(counting(11, &ran))
        .expect_err("no unit is admitted");
    assert_eq!(refused.header().id_hash(), 11);
    assert!(!lake.is_attached());
    assert_eq!(lake.attachments(), 0);

    // Admitted and ended: refused again, through the `FrameSink` the stage graphs use.
    drop(lake.admit().expect("a unit"));
    let sink: &dyn FrameSink = &*lake;
    let refused = sink.offer(counting(12, &ran)).expect_err("the unit ended");
    assert_eq!(refused.header().id_hash(), 12);

    // A refused frame was not queued anywhere.
    std::thread::sleep(Duration::from_millis(50));
    assert_eq!(ran.load(Ordering::Acquire), 0);
    lake.complete_gracefully();
    lattice.close();
}

#[test]
fn complete_gracefully_drains_then_completes() {
    let _turn = take_turn();
    const FRAMES: u64 = 200;
    let lattice = one_cpu_lattice();
    let lake = Lake::new(lattice.clone(), sinks(2), Duration::ZERO);
    let unit = lake.admit().expect("a unit");
    let ran = Arc::new(AtomicU64::new(0));
    for index in 0..FRAMES {
        assert!(lake.offer(counting(index, &ran)).is_ok());
    }

    // Admission closes; the unit still holds the lake open and may fork more work.
    lake.complete_gracefully();
    assert!(lake.admit().is_err(), "admission is closed");
    assert!(!lake.is_complete(), "a unit is still admitted");
    let forked = unit.fork();
    assert_eq!(lake.active(), 2);
    assert!(
        lake.offer(counting(1000, &ran)).is_ok(),
        "a unit may still offer"
    );

    drop(unit);
    assert!(!lake.is_complete(), "the forked unit is still admitted");
    assert!(lake.offer(counting(1001, &ran)).is_ok());
    drop(forked);
    assert!(lake.is_complete(), "the last unit ended");

    // The sinks drain what they hold, then complete and detach.
    eventually("the sinks to drain", || {
        ran.load(Ordering::Acquire) == FRAMES + 2
    });
    eventually("the sinks to complete", || !lake.is_attached());
    assert!(
        lake.offer(counting(1002, &ran)).is_err(),
        "a completed lake takes nothing"
    );
    assert_eq!(ran.load(Ordering::Acquire), FRAMES + 2);
    lattice.close();
}

/// A frame that sets a latch when it runs.
struct SetLatch {
    header: FrameHeader,
    setter: Option<LatchSetter<u64>>,
    value: u64,
}

impl Frame for SetLatch {
    fn header(&self) -> &FrameHeader {
        &self.header
    }

    fn header_mut(&mut self) -> &mut FrameHeader {
        &mut self.header
    }

    fn execute(&mut self) -> Result<(), FrameError> {
        if let Some(setter) = self.setter.take() {
            setter.set(self.value);
        }
        Ok(())
    }
}

#[test]
fn latch_set_before_wait_returns() {
    let (setter, waiter) = Latch::new();
    setter.set(41_u64 + 1);
    assert_eq!(waiter.wait(), 42);
}

#[test]
#[should_panic(expected = "dropped without setting")]
fn a_latch_whose_setter_was_dropped_does_not_park_forever() {
    let (setter, waiter) = Latch::<u64>::new();
    drop(setter);
    waiter.wait();
}

#[test]
fn latch_set_from_frame_wakes_waiter() {
    let _turn = take_turn();
    let lattice = one_cpu_lattice();
    let lake = Lake::new(lattice.clone(), sinks(1), Duration::ZERO);
    let unit = lake.admit().expect("a unit");
    let (setter, waiter) = Latch::new();
    let frame = SetLatch {
        header: FrameHeader::new(3),
        setter: Some(setter),
        value: 1234,
    };
    assert!(lake.offer(Box::new(frame)).is_ok());
    // The test thread parks here until the frame runs on the lattice's worker.
    assert_eq!(waiter.wait(), 1234);
    drop(unit);
    lake.complete_gracefully();
    lattice.close();
}
