//! Shared lanes on a real device: many workers submit to one lane through `&self`, the stages of a graph submit to their
//! lanes through their context, and retirement is read and settled in a frame. Every test skips, passing, when no
//! driver or device is present.
#![cfg(feature = "cuda")]
#![allow(
    clippy::disallowed_types,
    clippy::disallowed_methods,
    reason = "test fixtures: peer threads stand in for the lattice's workers and the test sleeps between looks"
)]

mod common;

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use common::cuda_support::{Pinned, device_word, gpu, module, wait_until};
use euhedral_core::frames::{DummyFrame, Frame, FrameError};
use euhedral_core::generics::{FrameSink, LatticeSource};
use euhedral_data_structures::queues::{Consumer, MpmcQueue, Multi, Producer};
use euhedral_gpu::cuda::{CudaKernel, CudaQueue};
use euhedral_gpu::{
    Args, GpuError, Lane, LaneEnqueue, Launch, Module, SharedLanes, StreamOptions,
    open_shared_lanes,
};
use euhedral_graph::{
    Completions, DEFAULT_RETRY_BACKSTOP, EdgeKind, GraphRuntime, LaneControl, LaneId,
    QuantumOutcome, RetireCell, Stage, StageCx, StageError, StageGraph, StageGraphSpec, StageSpec,
    Submitted,
};

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

fn lanes(gpu: &Arc<euhedral_gpu::cuda::CudaGpu>, count: usize) -> SharedLanes<CudaQueue> {
    let names = ["a", "b", "c", "d"];
    let specs: Vec<(&str, StreamOptions)> = names[..count]
        .iter()
        .map(|name| (*name, StreamOptions::default()))
        .collect();
    open_shared_lanes(gpu, &specs).unwrap()
}

/// Sixteen workers submit increments and marks to one lane at once. Every marker is distinct, the numbers are
/// contiguous, each worker's own markers rise, the device completes them all, and every increment ran.
#[test]
fn many_workers_submit_to_one_lane_and_every_marker_completes_in_order() {
    let Some(gpu) = gpu() else { return };
    let Some(module) = module(&gpu) else { return };
    let (_mem, block) = device_word(&gpu);
    let pinned = Pinned::new(&gpu);
    let add = module.kernel("add_u32").unwrap();
    let lanes = lanes(&gpu, 1);
    let lane = lanes.pool.lane(LaneId(0)).clone();
    const WORKERS: usize = 16;
    const EACH: usize = 200;
    let markers = Mutex::new(Vec::new());
    std::thread::scope(|scope| {
        for _ in 0..WORKERS {
            scope.spawn(|| {
                let mut mine = Vec::with_capacity(EACH);
                for _ in 0..EACH {
                    // A refusal comes before any work: the launch is entered once per marker made.
                    let marker = loop {
                        let made = lane.submit_marked(|q| {
                            // SAFETY: the device word is live and 256 bytes; the kernel adds 1 to its first element.
                            unsafe {
                                q.launch(
                                    &add,
                                    Launch::linear(1, 1),
                                    &Args::new()
                                        .device(block.addr)
                                        .device(block.addr)
                                        .arg(1u32)
                                        .arg(1u32),
                                )
                            }
                        });
                        match made {
                            Ok(((), marker)) => break marker,
                            Err(GpuError::Busy) => std::thread::yield_now(),
                            Err(other) => panic!("{other}"),
                        }
                    };
                    mine.push(marker.value);
                }
                assert!(
                    mine.windows(2).all(|w| w[0] < w[1]),
                    "a worker's own markers rise"
                );
                lock(&markers).extend(mine);
            });
        }
    });
    let mut values = std::mem::take(&mut *lock(&markers));
    values.sort_unstable();
    assert_eq!(
        values,
        (1..=(WORKERS * EACH) as u64).collect::<Vec<_>>(),
        "distinct and contiguous"
    );
    assert!(
        wait_until(Duration::from_secs(20), || DeviceQueueExt::done(&*lane)
            >= (WORKERS * EACH) as u64),
        "the device completes every marker"
    );
    // Read the word back on the same lane, behind everything.
    let host = pinned.word();
    lane.submit(|q| {
        // SAFETY: the pinned word is 4 bytes and lives until the lanes are gone; the device word is live.
        unsafe { q.copy_to_host(host.bytes(), block.addr, 4) }
    })
    .unwrap();
    let last = loop {
        match lane.mark() {
            Ok(marker) => break marker,
            Err(GpuError::Busy) => std::thread::yield_now(),
            Err(other) => panic!("{other}"),
        }
    };
    assert!(wait_until(Duration::from_secs(20), || lanes
        .queues
        .reached(last)));
    assert_eq!(host.load(), (WORKERS * EACH) as u32, "every launch ran");
}

/// `completed()` of a lane, by its trait.
trait DeviceQueueExt {
    fn done(&self) -> u64;
}

impl DeviceQueueExt for CudaQueue {
    fn done(&self) -> u64 {
        euhedral_graph::DeviceQueue::completed(self)
    }
}

struct Sink(Producer<Box<dyn Frame>, Multi>);

impl FrameSink for Sink {
    fn offer(&self, frame: Box<dyn Frame>) -> Result<(), Box<dyn Frame>> {
        self.0.offer(frame)
    }
}

fn run_frame(mut frame: Box<dyn Frame>) {
    if !frame.is_alive() {
        frame.do_finally();
        return;
    }
    match frame.execute() {
        Ok(()) | Err(FrameError::Cancelled) => frame.do_finally(),
        Err(error) => frame.do_finally_with_error(error),
    }
}

/// What a quantum works on: a device word, and the host word its last stage copies it to.
struct Work {
    word: u64,
    host: common::HostWord,
}

/// Stage one: sets the word to 3. Stage two, on another lane: doubles it and copies it to the host. In the wrong order
/// the host sees 12 (doubled before it was set, the first time 0, then the last quantum's 6).
struct Steps {
    kernel: CudaKernel,
    /// Burns about a millisecond on the device, so that a second stage that did not wait would overtake the first.
    spin: CudaKernel,
    last: bool,
}

impl Stage<Work, CudaQueue> for Steps {
    fn run(&self, work: &Work, cx: &StageCx<'_, CudaQueue>) -> Result<Submitted, StageError> {
        let lane = cx.lane_handle().expect("a stage with a lane");
        for marker in cx.waits {
            lane.await_marker(*marker)
                .map_err(|e| StageError::Corrupt(e.to_string()))?;
        }
        let (a, b) = if self.last {
            (2.0f32, 0.0f32)
        } else {
            (0.0, 3.0)
        };
        let made = lane.submit_marked(|q| {
            // SAFETY: the device word is live; the kernel reads and writes its first float. The host word is pinned and
            // outlives the quantum.
            unsafe {
                if !self.last {
                    q.launch(
                        &self.spin,
                        Launch::linear(1, 1),
                        &Args::new().arg(2_000_000u64),
                    )?;
                }
                q.launch(
                    &self.kernel,
                    Launch::linear(1, 1),
                    &Args::new()
                        .device(work.word)
                        .device(work.word)
                        .arg(a)
                        .arg(b)
                        .arg(1u32),
                )?;
                if self.last {
                    q.copy_to_host(work.host.bytes(), work.word, 4)?;
                }
                Ok(())
            }
        });
        match made {
            Ok(((), marker)) => Ok(Submitted::marker(marker)),
            Err(GpuError::Busy) => Err(StageError::Retry),
            Err(e) => Err(StageError::Corrupt(e.to_string())),
        }
    }
}

/// Four workers run quanta of a two-stage graph, each stage on its own lane, at once: the second stage awaits the first's
/// marker on the device (the host reads 6, not 12), and a quantum retires only when both lanes reached their markers.
#[test]
fn stages_submit_on_their_lanes_and_the_boundary_covers_both() {
    let Some(gpu) = gpu() else { return };
    let Some(module) = module(&gpu) else { return };
    let (_mem, block) = device_word(&gpu);
    let pinned = Pinned::new(&gpu);
    let axpb = module.kernel("axpb").unwrap();
    let spin = module.kernel("spin").unwrap();
    let lanes = lanes(&gpu, 2);
    let completions = Completions::new(lanes.queues.clone());
    let queue = MpmcQueue::<Box<dyn Frame>>::bounded(1024);
    let frames: Consumer<Box<dyn Frame>, Multi> = queue.consumer();
    let mut spec = StageGraphSpec::new();
    let first = spec.stage(StageSpec {
        name: "set",
        ordered: None,
        lanes: vec![LaneId(0)],
    });
    let second = spec.stage(StageSpec {
        name: "double",
        ordered: None,
        lanes: vec![LaneId(1)],
    });
    spec.edge(first, second, EdgeKind::Submission);
    const QUANTA: usize = 64;
    let outcomes = Arc::new(Mutex::new(Vec::new()));
    let log = outcomes.clone();
    let graph = StageGraph::new(
        spec,
        vec![
            Arc::new(Steps {
                kernel: axpb.clone(),
                spin: spin.clone(),
                last: false,
            }) as Arc<dyn Stage<Work, CudaQueue>>,
            Arc::new(Steps {
                kernel: axpb,
                spin,
                last: true,
            }),
        ],
        GraphRuntime {
            stages: Arc::new(Sink(queue.producer())),
            completions: completions.clone(),
            lanes: lanes.pool.clone(),
            retries: Arc::default(),
            retry_backstop: DEFAULT_RETRY_BACKSTOP,
        },
        move |_work, outcome| lock(&log).push(outcome),
    )
    .unwrap();
    let done = Arc::new(AtomicBool::new(false));
    let workers: Vec<_> = (0..4)
        .map(|_| {
            let (frames, completions, done) = (frames.clone(), completions.clone(), done.clone());
            std::thread::spawn(move || {
                while !done.load(Ordering::Acquire) {
                    if let Some(frame) = frames.poll() {
                        run_frame(frame);
                        continue;
                    }
                    let mut frame = None;
                    completions.pull(&mut |f| frame = Some(f), &mut |_| false, 1);
                    match frame {
                        Some(frame) => run_frame(frame),
                        None => std::thread::yield_now(),
                    }
                }
            })
        })
        .collect();
    let kill = Arc::new(AtomicBool::new(false));
    // Each quantum has its own device word and host word, so the quanta overlap on the device.
    for i in 0..QUANTA {
        // SAFETY: word `i` of the pinned block is 4 bytes, aligned, and outlives the quanta.
        let host =
            unsafe { common::HostWord::from_raw((pinned.word().bytes() as *mut u32).add(i)) };
        let work = Work {
            word: block.addr + 4 * i as u64,
            host,
        };
        let _ = graph.start(work, kill.clone(), None::<euhedral_graph::ExternalEdge>);
    }
    assert!(
        wait_until(Duration::from_secs(60), || lock(&outcomes).len() == QUANTA),
        "every quantum retired"
    );
    done.store(true, Ordering::Release);
    for worker in workers {
        worker.join().unwrap();
    }
    assert!(
        lock(&outcomes)
            .iter()
            .all(|o| *o == QuantumOutcome::Completed)
    );
    for i in 0..QUANTA {
        // SAFETY: as above; the retirement covered the copy.
        let seen = unsafe { ((pinned.word().bytes() as *const u32).add(i)).read_volatile() };
        assert_eq!(
            f32::from_bits(seen),
            6.0,
            "quantum {i}: the second stage ran after the first"
        );
    }
    assert_eq!(graph.live(), 0);
}

/// Sixteen threads arm a retirement each on one lane while others keep submitting: every party reads the status of its
/// own retirement (a cell it named), none waits for another's, and all arrive.
#[test]
fn parties_arming_retirements_on_one_lane_each_get_their_own() {
    let Some(gpu) = gpu() else { return };
    let Some(module) = module(&gpu) else { return };
    let spin = module.kernel("spin").unwrap();
    let lanes = lanes(&gpu, 1);
    let lane = lanes.pool.lane(LaneId(0)).clone();
    const PARTIES: usize = 16;
    const EACH: usize = 50;
    let cells: Vec<Arc<RetireCell>> = (0..PARTIES * EACH)
        .map(|_| Arc::new(RetireCell::new()))
        .collect();
    std::thread::scope(|scope| {
        for party in 0..PARTIES {
            let (lane, cells, spin) = (&lane, &cells, &spin);
            scope.spawn(move || {
                for round in 0..EACH {
                    lane.submit(|q| {
                        // SAFETY: the kernel only spins.
                        unsafe { q.launch(spin, Launch::linear(1, 1), &Args::new().arg(1_000u64)) }
                    })
                    .unwrap();
                    lane.notify_retired(&cells[party * EACH + round]).unwrap();
                }
            });
        }
    });
    assert!(
        wait_until(Duration::from_secs(30), || cells
            .iter()
            .all(|c| c.status().is_some())),
        "every retirement arrived"
    );
    assert!(cells.iter().all(|c| c.status() == Some(0)));
    assert_eq!(lane.confirm_retired(0).map(|e| e.to_string()), None);
}

struct Counted(Arc<AtomicUsize>);

impl Drop for Counted {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

/// A frame settles a retirement: the continuation of a cell is a frame, emitted when the driver's callback filled it.
#[test]
fn a_retirement_cell_is_settled_in_a_frame_after_the_work_before_it() {
    let Some(gpu) = gpu() else { return };
    let Some(module) = module(&gpu) else { return };
    let spin = module.kernel("spin").unwrap();
    let lanes = lanes(&gpu, 1);
    let lane = lanes.pool.lane(LaneId(0)).clone();
    lane.submit(|q| {
        // SAFETY: the kernel only spins.
        unsafe { q.launch(&spin, Launch::linear(1, 1), &Args::new().arg(1_000_000u64)) }
    })
    .unwrap();
    let cell = Arc::new(RetireCell::new());
    lane.notify_retired(&cell).unwrap();
    let completions = Completions::new(lanes.queues.clone());
    completions.after_retired(cell.clone(), Box::new(DummyFrame::INSTANCE));
    let mut frames = Vec::new();
    assert!(wait_until(Duration::from_secs(20), || {
        completions.pull(&mut |f| frames.push(f), &mut |_| false, 8);
        !frames.is_empty()
    }));
    assert_eq!(frames.len(), 1, "the continuation is a frame");
    assert_eq!(cell.status(), Some(0));
    assert_eq!(lane.confirm_retired(0).map(|e| e.to_string()), None);
    completions.pull(&mut |f| frames.push(f), &mut |_| false, 8);
    assert_eq!(frames.len(), 1, "once");
}

/// A stage holds a value for its work and releases it in its `retired` hook, which runs once the quantum's device work
/// ended: every quantum's hold is released, each once, with four workers running the frames.
#[test]
fn what_stages_hold_is_released_by_their_hooks_once_per_quantum() {
    let Some(gpu) = gpu() else { return };
    let Some(module) = module(&gpu) else { return };
    let spin = module.kernel("spin").unwrap();
    let lanes = lanes(&gpu, 2);
    let completions = Completions::new(lanes.queues.clone());
    let queue = MpmcQueue::<Box<dyn Frame>>::bounded(1024);
    let frames: Consumer<Box<dyn Frame>, Multi> = queue.consumer();
    let mut spec = StageGraphSpec::new();
    let first = spec.stage(StageSpec {
        name: "one",
        ordered: None,
        lanes: vec![LaneId(0)],
    });
    let second = spec.stage(StageSpec {
        name: "two",
        ordered: None,
        lanes: vec![LaneId(1)],
    });
    spec.edge(first, second, EdgeKind::Submission);
    struct Holder {
        spin: CudaKernel,
        dropped: Arc<AtomicUsize>,
    }
    struct Held(Mutex<Vec<Counted>>);
    impl Stage<Held, CudaQueue> for Holder {
        fn run(&self, held: &Held, cx: &StageCx<'_, CudaQueue>) -> Result<Submitted, StageError> {
            let lane = cx.lane_handle().expect("a stage with a lane");
            for marker in cx.waits {
                lane.await_marker(*marker)
                    .map_err(|e| StageError::Corrupt(e.to_string()))?;
            }
            let made = lane.submit_marked(|q| {
                // SAFETY: the kernel only spins.
                unsafe {
                    q.launch(
                        &self.spin,
                        Launch::linear(1, 1),
                        &Args::new().arg(10_000u64),
                    )
                }
            });
            match made {
                Ok(((), marker)) => {
                    lock(&held.0).push(Counted(self.dropped.clone()));
                    Ok(Submitted::marker(marker))
                }
                Err(GpuError::Busy) => Err(StageError::Retry),
                Err(e) => Err(StageError::Corrupt(e.to_string())),
            }
        }

        fn retired(&self, held: &Held, committed: bool) {
            assert!(committed);
            lock(&held.0).clear();
        }
    }
    const QUANTA: usize = 64;
    let dropped = Arc::new(AtomicUsize::new(0));
    let outcomes = Arc::new(Mutex::new(Vec::new()));
    let log = outcomes.clone();
    let stages: Vec<Arc<dyn Stage<Held, CudaQueue>>> = (0..2)
        .map(|_| {
            Arc::new(Holder {
                spin: spin.clone(),
                dropped: dropped.clone(),
            }) as Arc<dyn Stage<Held, CudaQueue>>
        })
        .collect();
    let graph = StageGraph::new(
        spec,
        stages,
        GraphRuntime {
            stages: Arc::new(Sink(queue.producer())),
            completions: completions.clone(),
            lanes: lanes.pool.clone(),
            retries: Arc::default(),
            retry_backstop: DEFAULT_RETRY_BACKSTOP,
        },
        move |_held, outcome| lock(&log).push(outcome),
    )
    .unwrap();
    let done = Arc::new(AtomicBool::new(false));
    let workers: Vec<_> = (0..4)
        .map(|_| {
            let (frames, completions, done) = (frames.clone(), completions.clone(), done.clone());
            std::thread::spawn(move || {
                while !done.load(Ordering::Acquire) {
                    if let Some(frame) = frames.poll() {
                        run_frame(frame);
                        continue;
                    }
                    let mut frame = None;
                    completions.pull(&mut |f| frame = Some(f), &mut |_| false, 1);
                    match frame {
                        Some(frame) => run_frame(frame),
                        None => std::thread::yield_now(),
                    }
                }
            })
        })
        .collect();
    let kill = Arc::new(AtomicBool::new(false));
    for _ in 0..QUANTA {
        let _ = graph.start(
            Held(Mutex::default()),
            kill.clone(),
            None::<euhedral_graph::ExternalEdge>,
        );
    }
    assert!(
        wait_until(Duration::from_secs(60), || lock(&outcomes).len() == QUANTA),
        "every quantum retired"
    );
    done.store(true, Ordering::Release);
    for worker in workers {
        worker.join().unwrap();
    }
    assert!(
        lock(&outcomes)
            .iter()
            .all(|o| *o == QuantumOutcome::Completed)
    );
    assert_eq!(
        dropped.load(Ordering::SeqCst),
        2 * QUANTA,
        "each hold released, once"
    );
    assert_eq!(graph.live(), 0);
}
