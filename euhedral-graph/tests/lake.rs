//! Lakes on a real lattice of one CPU.
//!
//! CI runners have two cores, so nothing here assumes more than one: the lattice is given a single
//! CPU, one the process may run on, and the tests assert structure (order, attachment, counts), not
//! simultaneity.
#![allow(
    clippy::disallowed_types,
    reason = "test fixtures: the test thread locks its records"
)]
// The test thread is the harness, not a frame: it polls for what the lattice did, and waits on a
// one-shot, with sleeps no frame may make.
#![allow(clippy::disallowed_methods)]

use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

use euhedral_core::config::LatticeConfig;
use euhedral_core::control_plane::ControlPlaneLattice;
use euhedral_core::frames::{Frame, FrameError, FrameHeader, RunnableFrame};
use euhedral_core::generics::FrameSink;
use euhedral_core::impls::DefaultExecutor;
use euhedral_data_structures::sync::HotScope;
use euhedral_data_structures::sync::{OneShotSender, one_shot};
use euhedral_graph::Lake;
use euhedral_hardware_utils::affinity;
use euhedral_hardware_utils::util::CpuSet;

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

    // The sinks drain what they hold, then complete and detach: they are attached until then.
    eventually("the sinks to complete", || !lake.is_attached());
    assert_eq!(
        ran.load(Ordering::Acquire),
        FRAMES + 2,
        "the sinks completed only after the drain"
    );
    assert!(
        lake.offer(counting(1002, &ran)).is_err(),
        "a completed lake takes nothing"
    );
    assert_eq!(ran.load(Ordering::Acquire), FRAMES + 2);
    lattice.close();
}

#[test]
fn a_frame_that_completes_its_sink_does_not_panic() {
    let lattice = one_cpu_lattice();
    let lake = Lake::new(lattice.clone(), sinks(2), Duration::ZERO);
    let unit = Arc::new(Mutex::new(Some(lake.admit().expect("a unit"))));
    let done = Arc::new(AtomicU64::new(0));

    // The frame ends the lake's last unit on a worker: the sinks complete from inside a frame, which is
    // lifecycle work done on the hot path's thread.
    let frame: Box<dyn Frame> = {
        let (lake, unit, done) = (lake.clone(), unit.clone(), done.clone());
        Box::new(RunnableFrame::new(
            1,
            Arc::new(move || {
                lake.complete_gracefully();
                drop(unit.lock().unwrap_or_else(PoisonError::into_inner).take());
                done.fetch_add(1, Ordering::AcqRel);
            }),
        ))
    };
    assert!(lake.offer(frame).is_ok());
    eventually("the frame to run", || done.load(Ordering::Acquire) == 1);
    eventually("the lake to complete", || lake.is_complete());
    eventually("the sinks to complete", || !lake.is_attached());
    lattice.close();
}

#[test]
fn offers_racing_an_idle_detach_land_in_exactly_one_set() {
    const THREADS: u64 = 8;
    const BURSTS: u64 = 25;
    const PER_BURST: u64 = 20;
    let lattice = one_cpu_lattice();
    let lake = Lake::new(lattice.clone(), sinks(3), Duration::from_millis(1));
    let ran = Arc::new(AtomicU64::new(0));
    let offered = Arc::new(AtomicU64::new(0));
    let threads: Vec<_> = (0..THREADS)
        .map(|thread| {
            let (lake, ran, offered) = (lake.clone(), ran.clone(), offered.clone());
            std::thread::spawn(move || {
                for burst in 0..BURSTS {
                    let unit = lake.admit().expect("a unit");
                    for frame in 0..PER_BURST {
                        let lane = thread * 1000 + burst * 31 + frame;
                        if lake.offer(counting(lane, &ran)).is_ok() {
                            offered.fetch_add(1, Ordering::AcqRel);
                        }
                    }
                    drop(unit);
                    // An idle gap longer than the idle time, so the watch detaches between bursts.
                    std::thread::sleep(Duration::from_millis(3));
                }
            })
        })
        .collect();
    for thread in threads {
        thread.join().unwrap();
    }
    let offered = offered.load(Ordering::Acquire);
    assert_eq!(
        offered,
        THREADS * BURSTS * PER_BURST,
        "a unit holds the lake open"
    );
    eventually("every offered frame to run", || {
        ran.load(Ordering::Acquire) == offered
    });
    assert!(
        lake.attachments() > 1,
        "the idle gaps detached the set and the next burst attached another ({} attachments)",
        lake.attachments()
    );
    std::thread::sleep(Duration::from_millis(20));
    assert_eq!(ran.load(Ordering::Acquire), offered, "each frame ran once");
    lake.complete_gracefully();
    lattice.close();
}

#[test]
fn offers_on_an_attached_set_take_no_lock() {
    let lattice = one_cpu_lattice();
    let lake = Lake::new(lattice.clone(), sinks(2), Duration::ZERO);
    let unit = lake.admit().expect("a unit");
    let ran = Arc::new(AtomicU64::new(0));
    {
        // A lifecycle lock taken in here ends a debug run.
        let _hot = HotScope::enter();
        for index in 0..10_000 {
            assert!(lake.offer(counting(index, &ran)).is_ok());
        }
    }
    eventually("the frames to run", || {
        ran.load(Ordering::Acquire) == 10_000
    });
    drop(unit);
    lake.complete_gracefully();
    lattice.close();
}

#[test]
fn an_attached_for_life_lake_never_detaches() {
    let lattice = one_cpu_lattice();
    let lake = Lake::new(lattice.clone(), sinks(2), Duration::ZERO);
    drop(lake.admit().expect("a unit"));
    std::thread::sleep(Duration::from_millis(50));
    assert!(lake.is_attached());
    assert_eq!(lake.attachments(), 1);
    lake.complete_gracefully();
    lattice.close();
}

#[test]
fn offers_from_8_threads_racing_complete_finish_every_admitted_unit() {
    const THREADS: u64 = 8;
    for _ in 0..5 {
        let lattice = one_cpu_lattice();
        let lake = Lake::new(lattice.clone(), sinks(2), Duration::from_millis(1));
        let ran = Arc::new(AtomicU64::new(0));
        let accepted = Arc::new(AtomicU64::new(0));
        let units = Arc::new(AtomicU64::new(0));
        let threads: Vec<_> = (0..THREADS)
            .map(|thread| {
                let (lake, ran, accepted, units) =
                    (lake.clone(), ran.clone(), accepted.clone(), units.clone());
                std::thread::spawn(move || {
                    for round in 0..200 {
                        let Ok(unit) = lake.admit() else { return };
                        units.fetch_add(1, Ordering::AcqRel);
                        for frame in 0..4 {
                            if lake
                                .offer(counting(thread * 100_000 + round * 10 + frame, &ran))
                                .is_ok()
                            {
                                accepted.fetch_add(1, Ordering::AcqRel);
                            }
                        }
                        // A unit may fork while the lake drains.
                        let forked = unit.fork();
                        drop(unit);
                        drop(forked);
                    }
                })
            })
            .collect();
        std::thread::sleep(Duration::from_millis(5));
        lake.complete_gracefully();
        for thread in threads {
            thread.join().unwrap();
        }
        assert!(
            units.load(Ordering::Acquire) > 0,
            "some units were admitted"
        );
        eventually("the lake to complete", || lake.is_complete());
        assert_eq!(lake.active(), 0, "every admitted unit terminated");
        let accepted = accepted.load(Ordering::Acquire);
        eventually("every accepted frame to run", || {
            ran.load(Ordering::Acquire) == accepted
        });
        eventually("the sinks to complete", || !lake.is_attached());
        assert_eq!(ran.load(Ordering::Acquire), accepted, "each ran once");
        lattice.close();
    }
}

/// A frame that sends a value through a one-shot when it runs.
struct SendOnce {
    header: FrameHeader,
    sender: Option<OneShotSender<u64>>,
    value: u64,
}

impl Frame for SendOnce {
    fn header(&self) -> &FrameHeader {
        &self.header
    }

    fn header_mut(&mut self) -> &mut FrameHeader {
        &mut self.header
    }

    fn execute(&mut self) -> Result<(), FrameError> {
        if let Some(sender) = self.sender.take() {
            sender.send(self.value);
        }
        Ok(())
    }
}

#[test]
fn a_value_sent_before_the_check_is_there() {
    let (sender, mut receiver) = one_shot();
    sender.send(41_u64 + 1);
    assert!(receiver.is_set());
    assert_eq!(receiver.try_take(), Ok(Some(42)));
}

#[test]
fn a_one_shot_whose_sender_was_dropped_reports_it_instead_of_staying_empty() {
    let (sender, mut receiver) = one_shot::<u64>();
    drop(sender);
    assert!(receiver.try_take().is_err());
}

#[test]
fn a_value_sent_from_a_frame_arrives_for_a_main_thread_that_checks() {
    let lattice = one_cpu_lattice();
    let lake = Lake::new(lattice.clone(), sinks(1), Duration::ZERO);
    let unit = lake.admit().expect("a unit");
    let (sender, mut receiver) = one_shot();
    let frame = SendOnce {
        header: FrameHeader::new(3),
        sender: Some(sender),
        value: 1234,
    };
    assert!(lake.offer(Box::new(frame)).is_ok());
    // The test thread checks until the frame has run on the lattice's worker.
    eventually("the frame to send", || receiver.is_set());
    assert_eq!(receiver.try_take(), Ok(Some(1234)));
    drop(unit);
    lake.complete_gracefully();
    lattice.close();
}
