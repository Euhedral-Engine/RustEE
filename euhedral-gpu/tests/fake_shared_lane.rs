//! Shared lanes on the fake backend: any thread submits to a lane through `&self`, a stage of a graph submits to its
//! lane through its context, and retirement is read and settled in a frame. No GPU, no wall clock.
#![cfg(feature = "fake")]
#![allow(
    clippy::disallowed_types,
    clippy::disallowed_methods,
    reason = "test fixtures: the test locks its records and runs peer threads that stand in for workers"
)]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use euhedral_core::frames::{Frame, FrameError};
use euhedral_core::generics::{FrameSink, LatticeSource};
use euhedral_data_structures::queues::{Consumer, MpmcQueue, Multi, Producer};
use euhedral_gpu::fake::{FakeGpu, FakeKernel, FakeModule, FakeOp, FakeQueue};
use euhedral_gpu::{
    Args, CompileOptions, Enqueue, GpuError, Lane, Launch, Module, SharedLanes, StreamOptions,
    open_shared_lanes,
};
use euhedral_graph::{
    Completions, DEFAULT_RETRY_BACKSTOP, EdgeKind, GraphRuntime, LaneId, Pending, QuantumOutcome,
    Stage, StageCx, StageError, StageGraph, StageGraphSpec, StageSpec, Submitted,
};

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

fn gpu() -> Arc<FakeGpu> {
    let gpu = FakeGpu::new();
    gpu.define("add1", |mem, _launch, args| {
        let at = args.read::<u64>(0).unwrap();
        let word = mem.bytes_mut(at, 4)?;
        let v = u32::from_le_bytes(word.try_into().unwrap()) + 1;
        word.copy_from_slice(&v.to_le_bytes());
        Ok(())
    });
    gpu
}

fn kernel(gpu: &Arc<FakeGpu>) -> FakeKernel {
    FakeModule::from_source(gpu, "// fake source", &CompileOptions::default())
        .unwrap()
        .kernel("add1")
        .unwrap()
}

fn open(gpu: &Arc<FakeGpu>, lanes: usize) -> SharedLanes<FakeQueue> {
    let names = ["a", "b", "c", "d"];
    let specs: Vec<(&str, StreamOptions)> = names[..lanes]
        .iter()
        .map(|name| {
            (
                *name,
                StreamOptions {
                    markers: 1024,
                    ..StreamOptions::default()
                },
            )
        })
        .collect();
    open_shared_lanes(gpu, &specs).unwrap()
}

/// Eight threads submit a launch and make a marker on one lane, at once: every marker is distinct, the numbers are
/// contiguous, and the device completes all of them.
#[test]
fn many_workers_submit_to_one_lane_and_every_marker_completes() {
    let gpu = gpu();
    let kernel = kernel(&gpu);
    let word = gpu.alloc(4);
    let lanes = open(&gpu, 1);
    let lane = lanes.pool.lane(LaneId(0)).clone();
    let markers = Mutex::new(Vec::new());
    let (workers, each) = if cfg!(miri) { (3, 4) } else { (8, 50) };
    std::thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| {
                for _ in 0..each {
                    lane.submit(|q| {
                        // SAFETY: a live fake allocation.
                        unsafe {
                            q.launch(&kernel, Launch::linear(1, 1), &Args::new().device(word))
                        }
                    })
                    .unwrap();
                    let marker = loop {
                        match lane.mark() {
                            Ok(marker) => break marker,
                            Err(GpuError::Busy) => {}
                            Err(other) => panic!("{other}"),
                        }
                    };
                    lock(&markers).push(marker);
                }
            });
        }
    });
    let mut values: Vec<u64> = lock(&markers).iter().map(|m| m.value).collect();
    values.sort_unstable();
    let total = (workers * each) as u64;
    assert_eq!(
        values,
        (1..=total).collect::<Vec<u64>>(),
        "distinct and contiguous"
    );
    gpu.run_until_idle();
    assert!(lock(&markers).iter().all(|m| lanes.queues.reached(*m)));
    assert_eq!(
        u64::from(u32::from_le_bytes(gpu.read(word, 4).try_into().unwrap())),
        total,
        "every launch ran"
    );
}

/// The retirements a frame settled: the ticket's number and the failure behind it.
type Settled = Arc<Mutex<Vec<(u64, Option<GpuError>)>>>;

/// A frame polling a lane for a retirement, as `retire_into` polls a stream: the ticket is read and settled here, not
/// by the callback.
struct Settle {
    lane: Arc<FakeQueue>,
    settled: Settled,
}

impl Pending for Settle {
    fn poll(&mut self) -> Option<Box<dyn Frame>> {
        let ticket = self.lane.poll_retired()?;
        let failure = self.lane.confirm_retired(ticket);
        lock(&self.settled).push((ticket.seq(), failure));
        Some(Box::new(euhedral_core::frames::DummyFrame::INSTANCE))
    }
}

struct Counted(Arc<AtomicUsize>);

impl Drop for Counted {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

/// The device's retirement callback only queues; the ticket is read and settled in a frame, and the values kept for it
/// are released there, once.
#[test]
fn retirement_is_settled_in_a_frame_and_releases_kept_values_once() {
    let gpu = gpu();
    let kernel = kernel(&gpu);
    let word = gpu.alloc(4);
    let lanes = open(&gpu, 1);
    let lane = lanes.pool.lane(LaneId(0)).clone();
    let dropped = Arc::new(AtomicUsize::new(0));
    lane.submit(|q| {
        // SAFETY: a live fake allocation.
        unsafe { q.launch(&kernel, Launch::linear(1, 1), &Args::new().device(word))? };
        q.keep(Box::new(Counted(dropped.clone())));
        Ok(())
    })
    .unwrap();
    lane.arm_retirement().unwrap();
    let completions = Completions::new(lanes.queues.clone());
    let settled = Arc::new(Mutex::new(Vec::new()));
    completions.register(Box::new(Settle {
        lane: lane.clone(),
        settled: settled.clone(),
    }));
    let mut frames = Vec::new();
    completions.pull(&mut |f| frames.push(f), &mut |_| false, 8);
    assert!(
        frames.is_empty() && lock(&settled).is_empty(),
        "the device has not run"
    );
    assert_eq!(dropped.load(Ordering::SeqCst), 0);
    gpu.run_until_idle();
    completions.pull(&mut |f| frames.push(f), &mut |_| false, 8);
    assert_eq!(frames.len(), 1, "the continuation is a frame");
    assert_eq!(*lock(&settled), [(0, None)]);
    assert_eq!(dropped.load(Ordering::SeqCst), 1, "released once");
    completions.pull(&mut |f| frames.push(f), &mut |_| false, 8);
    assert_eq!(dropped.load(Ordering::SeqCst), 1);
}

// ---- a stage graph whose stages submit to their lanes ----

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

/// A stage that orders itself behind the markers it was given, launches on its lane, and returns the lane's marker.
struct Launches {
    kernel: FakeKernel,
    word: u64,
}

impl Stage<u64, FakeQueue> for Launches {
    fn run(&self, _: &u64, cx: &StageCx<'_, FakeQueue>) -> Result<Submitted, StageError> {
        let lane = cx.lane_handle().expect("a stage with a lane");
        for marker in cx.waits {
            lane.await_marker(*marker)
                .map_err(|e| StageError::Corrupt(e.to_string()))?;
        }
        lane.submit(|q| {
            // SAFETY: a live fake allocation.
            unsafe {
                q.launch(
                    &self.kernel,
                    Launch::linear(1, 1),
                    &Args::new().device(self.word),
                )
            }
        })
        .map_err(|e| StageError::Corrupt(e.to_string()))?;
        match lane.mark() {
            Ok(marker) => Ok(Submitted::marker(marker)),
            Err(GpuError::Busy) => Err(StageError::Retry),
            Err(e) => Err(StageError::Corrupt(e.to_string())),
        }
    }
}

#[test]
fn a_stage_submits_on_its_lane_and_a_successor_on_another_lane_awaits_its_marker() {
    let gpu = gpu();
    let kernel = kernel(&gpu);
    let word = gpu.alloc(4);
    let lanes = open(&gpu, 2);
    let completions = Completions::new(lanes.queues.clone());
    let queue = MpmcQueue::<Box<dyn Frame>>::bounded(64);
    let frames: Consumer<Box<dyn Frame>, Multi> = queue.consumer();
    let mut spec = StageGraphSpec::new();
    let a = spec.stage(StageSpec {
        name: "first",
        ordered: None,
        lanes: vec![LaneId(0)],
    });
    let b = spec.stage(StageSpec {
        name: "second",
        ordered: None,
        lanes: vec![LaneId(1)],
    });
    spec.edge(a, b, EdgeKind::Submission);
    let retired = Arc::new(Mutex::new(Vec::new()));
    let log = retired.clone();
    let graph = StageGraph::new(
        spec,
        vec![
            Arc::new(Launches {
                kernel: kernel.clone(),
                word,
            }) as Arc<dyn Stage<u64, FakeQueue>>,
            Arc::new(Launches {
                kernel: kernel.clone(),
                word,
            }),
        ],
        GraphRuntime {
            stages: Arc::new(Sink(queue.producer())),
            completions: completions.clone(),
            lanes: lanes.pool.clone(),
            retries: Arc::default(),
            retry_backstop: DEFAULT_RETRY_BACKSTOP,
        },
        move |q, outcome| lock(&log).push((q, outcome)),
    )
    .unwrap();
    let _ = graph.start(7, Arc::new(std::sync::atomic::AtomicBool::new(false)), None);
    let pump = || {
        loop {
            while let Some(frame) = frames.poll() {
                run_frame(frame);
            }
            let mut frame = None;
            completions.pull(&mut |f| frame = Some(f), &mut |_| false, 1);
            match frame {
                Some(frame) => run_frame(frame),
                None => return,
            }
        }
    };
    pump();
    assert!(
        lock(&retired).is_empty(),
        "the device has not run: the boundary is ahead"
    );
    // The first stage's marker (lane 0, value 1) is what the second awaited on its own lane.
    let log = gpu.log();
    assert_eq!(log, [], "nothing ran on the device yet");
    gpu.run_until_idle();
    pump();
    assert_eq!(
        *lock(&retired),
        [(7, QuantumOutcome::Completed)],
        "the boundary covers both lanes"
    );
    assert_eq!(u32::from_le_bytes(gpu.read(word, 4).try_into().unwrap()), 2);
    let log = gpu.log();
    let mark = log
        .iter()
        .position(|op| *op == FakeOp::Mark { lane: 0, value: 1 })
        .expect("the first stage's marker");
    let second = log
        .iter()
        .position(|op| matches!(op, FakeOp::Launch { lane: 1, .. }))
        .expect("the second stage's launch");
    let wait = log
        .iter()
        .position(|op| *op == FakeOp::Await { lane: 1, value: 1 })
        .expect("the second lane waited for the marker");
    assert!(mark < wait && wait < second, "{log:?}");
}
