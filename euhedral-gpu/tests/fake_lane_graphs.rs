//! Recurring sequences recorded and replayed through shared lanes on the fake backend: a stage of a stage graph runs a
//! cached graph on its lane, the replay composes with the lane's marker, and what the work reads is held by the quantum
//! until the stage's `retired` hook. No GPU, no wall clock.
#![cfg(feature = "fake")]
#![allow(
    clippy::disallowed_types,
    clippy::disallowed_methods,
    reason = "test fixtures: the test locks its records"
)]

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use euhedral_core::frames::{Frame, FrameError};
use euhedral_core::generics::{FrameSink, LatticeSource};
use euhedral_data_structures::queues::{Consumer, MpmcQueue, Multi, Producer};
use euhedral_gpu::fake::{FakeGpu, FakeKernel, FakeModule, FakeOp, FakeQueue};
use euhedral_gpu::{
    Args, CompileOptions, Enqueue, GpuError, GraphBody, GraphHold, Lane, LaneEnqueue, Launch,
    Module, Ran, SharedGraphCache, SharedLanes, StreamOptions, open_shared_lanes,
};
use euhedral_graph::{
    Completions, DEFAULT_RETRY_BACKSTOP, EdgeKind, GraphRuntime, LaneControl, LaneId,
    QuantumOutcome, Stage, StageCx, StageError, StageGraph, StageGraphSpec, StageSpec, Submitted,
};

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

fn gpu() -> Arc<FakeGpu> {
    let gpu = FakeGpu::new();
    gpu.define("addv", |mem, _launch, args| {
        let at = args.read::<u64>(0).unwrap();
        let by = args.read::<u32>(1).unwrap();
        let word = mem.bytes_mut(at, 4)?;
        let v = u32::from_le_bytes(word.try_into().unwrap()) + by;
        word.copy_from_slice(&v.to_le_bytes());
        Ok(())
    });
    gpu
}

fn word(gpu: &FakeGpu, at: u64) -> u32 {
    u32::from_le_bytes(gpu.read(at, 4).try_into().unwrap())
}

struct Counted(Arc<AtomicUsize>);

impl Drop for Counted {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

/// One kernel body for every pass and every way to submit: it is written against `LaneEnqueue` only.
fn add_by<E: LaneEnqueue<Kernel = FakeKernel>>(
    enq: &mut E,
    kernel: &FakeKernel,
    at: u64,
    by: u32,
) -> Result<(), GpuError> {
    // SAFETY: a live fake allocation.
    unsafe {
        enq.launch(
            kernel,
            Launch::linear(1, 1),
            &Args::new().device(at).arg(by),
        )
    }
}

struct Body {
    kernel: FakeKernel,
    word: u64,
    by: u32,
    /// What the body keeps for its work (the host buffers a copy would read).
    keeps: Option<Arc<AtomicUsize>>,
}

impl GraphBody<FakeGpu> for Body {
    fn enqueue<E: Enqueue<Kernel = FakeKernel>>(&mut self, enq: &mut E) -> Result<(), GpuError> {
        add_by(enq, &self.kernel, self.word, self.by)?;
        if let Some(count) = &self.keeps {
            enq.keep(Box::new(Counted(count.clone())));
        }
        Ok(())
    }
}

fn lanes(gpu: &Arc<FakeGpu>, count: usize) -> SharedLanes<FakeQueue> {
    let names = ["a", "b", "c", "d"];
    let specs: Vec<(&str, StreamOptions)> = names[..count]
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

fn kernel(gpu: &Arc<FakeGpu>) -> FakeKernel {
    FakeModule::from_source(gpu, "addv", &CompileOptions::default())
        .unwrap()
        .kernel("addv")
        .unwrap()
}

/// One kernel body serves a direct submission on a lane (`submit_marked`), the lane's held submission, and the passes of
/// a recording, an update and a replay: it is not written twice.
#[test]
fn one_kernel_body_serves_every_way_to_submit_and_every_pass() {
    let gpu = gpu();
    let kernel = kernel(&gpu);
    let at = gpu.alloc(4);
    let lanes = lanes(&gpu, 1);
    let lane = lanes.pool.lane(LaneId(0)).clone();
    // A shared lane's plain submission.
    lane.submit_marked(|e| add_by(e, &kernel, at, 1)).unwrap();
    // The same body in a held submission and in the cache's passes.
    let mut hold = GraphHold::new();
    lane.submit_held(&mut hold, |e| add_by(e, &kernel, at, 10))
        .unwrap();
    let cache = SharedGraphCache::<FakeGpu, _>::new();
    let mut body = Body {
        kernel: kernel.clone(),
        word: at,
        by: 100,
        keeps: None,
    };
    assert_eq!(
        cache.run(&gpu, &lane, 1u32, &mut body, &mut hold).unwrap(),
        Ran::Direct,
        "first sighting"
    );
    assert_eq!(
        cache.run(&gpu, &lane, 1u32, &mut body, &mut hold).unwrap(),
        Ran::Recorded,
        "second sighting"
    );
    body.by = 1000;
    assert_eq!(
        cache.run(&gpu, &lane, 1u32, &mut body, &mut hold).unwrap(),
        Ran::Replayed
    );
    gpu.run_until_idle();
    assert_eq!(word(&gpu, at), 1 + 10 + 100 + 100 + 1000);
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

/// A quantum: its number, and what its stages hold for their work.
struct Batch {
    id: u64,
    hold: Mutex<GraphHold>,
}

/// A stage that runs a cached graph on its lane, inside the lane's marker, and holds what the work reads.
struct Replays {
    cache: Arc<SharedGraphCache<FakeGpu, u32>>,
    gpu: Arc<FakeGpu>,
    kernel: FakeKernel,
    word: u64,
    key: u32,
    by: u32,
    keeps: Option<Arc<AtomicUsize>>,
    hooks: Arc<AtomicUsize>,
}

impl Stage<Batch, FakeQueue> for Replays {
    fn run(&self, q: &Batch, cx: &StageCx<'_, FakeQueue>) -> Result<Submitted, StageError> {
        let lane = cx.lane_handle().expect("a stage with a lane");
        let mut body = Body {
            kernel: self.kernel.clone(),
            word: self.word,
            by: self.by,
            keeps: self.keeps.clone(),
        };
        let mut hold = lock(&q.hold);
        // The marker is claimed before the run: a refusal (`Busy`) comes before any recording or replay.
        let made = lane.submit_marked(|_| {
            self.cache
                .run_after(&self.gpu, lane, cx.waits, self.key, &mut body, &mut hold)
        });
        match made {
            Ok((_, marker)) => Ok(Submitted::marker(marker)),
            Err(GpuError::Busy) => Err(StageError::Retry),
            Err(e) => Err(StageError::Corrupt(e.to_string())),
        }
    }

    fn retired(&self, q: &Batch, _committed: bool) {
        self.hooks.fetch_add(1, Ordering::SeqCst);
        lock(&q.hold).release();
    }
}

struct Fixture {
    gpu: Arc<FakeGpu>,
    graph: Arc<StageGraph<Batch, FakeQueue>>,
    frames: Consumer<Box<dyn Frame>, Multi>,
    completions: Arc<Completions>,
    retired: Arc<Mutex<Vec<(u64, QuantumOutcome)>>>,
    cache: Arc<SharedGraphCache<FakeGpu, u32>>,
    lanes: SharedLanes<FakeQueue>,
    word: u64,
    hooks: Arc<AtomicUsize>,
    kept: Arc<AtomicUsize>,
}

impl Fixture {
    fn settle(&self) {
        loop {
            loop {
                while let Some(frame) = self.frames.poll() {
                    run_frame(frame);
                }
                let mut frame = None;
                self.completions
                    .pull(&mut |f| frame = Some(f), &mut |_| false, 1);
                match frame {
                    Some(frame) => run_frame(frame),
                    None => break,
                }
            }
            if self.gpu.run_until_idle() == 0 {
                return;
            }
        }
    }

    fn start(&self, id: u64) {
        let _ = self.graph.start(
            Batch {
                id,
                hold: Mutex::default(),
            },
            Arc::new(AtomicBool::new(false)),
            None,
        );
    }
}

/// Stage 0 on lane 0 and stage 1 on lane 1 (after it), each replaying a graph of its own key.
fn fixture() -> Fixture {
    let gpu = gpu();
    let kernel = kernel(&gpu);
    let at = gpu.alloc(4);
    let lanes = lanes(&gpu, 2);
    let completions = Completions::new(lanes.queues.clone());
    let queue = MpmcQueue::<Box<dyn Frame>>::bounded(256);
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
    let cache = Arc::new(SharedGraphCache::<FakeGpu, u32>::new());
    let hooks = Arc::new(AtomicUsize::new(0));
    let kept = Arc::new(AtomicUsize::new(0));
    let stage = |key: u32, by: u32| {
        Arc::new(Replays {
            cache: cache.clone(),
            gpu: gpu.clone(),
            kernel: kernel.clone(),
            word: at,
            key,
            by,
            keeps: Some(kept.clone()),
            hooks: hooks.clone(),
        }) as Arc<dyn Stage<Batch, FakeQueue>>
    };
    let graph = StageGraph::new(
        spec,
        vec![stage(1, 1), stage(2, 10)],
        GraphRuntime {
            stages: Arc::new(Sink(queue.producer())),
            completions: completions.clone(),
            lanes: lanes.pool.clone(),
            retries: Arc::default(),
            retry_backstop: DEFAULT_RETRY_BACKSTOP,
        },
        move |q: Batch, outcome| lock(&log).push((q.id, outcome)),
    )
    .unwrap();
    Fixture {
        gpu,
        graph,
        frames,
        completions,
        retired,
        cache,
        lanes,
        word: at,
        hooks,
        kept,
    }
}

/// A stage replays a cached graph on its shared lane inside a stage graph quantum, and the successor on another lane
/// awaits its marker on the device.
#[test]
fn a_stage_replays_a_cached_graph_on_its_lane_and_the_successor_awaits_its_marker() {
    let f = fixture();
    f.start(1);
    f.settle();
    assert_eq!(*lock(&f.retired), [(1, QuantumOutcome::Completed)]);
    assert_eq!(
        f.cache.stats().directs,
        2,
        "the first sighting ran directly"
    );
    f.start(2);
    f.settle();
    assert_eq!(f.cache.stats().records, 2, "the second recorded both");
    for q in 3..=6 {
        f.start(q);
        f.settle();
    }
    assert_eq!(lock(&f.retired).len(), 6);
    assert_eq!(f.cache.stats().replays, 8, "four quanta replayed both");
    assert_eq!(f.cache.len(), 2);
    assert_eq!(word(&f.gpu, f.word), 6 * (1 + 10));
    // The second stage's launches waited for the first's marker on lane 1.
    let log = f.gpu.log();
    let wait = log
        .iter()
        .position(|op| matches!(op, FakeOp::Await { lane: 1, .. }))
        .expect("the second lane waited for the first's marker");
    let launch = log
        .iter()
        .position(|op| matches!(op, FakeOp::Launch { lane: 1, .. }))
        .expect("the second stage's launch");
    assert!(wait < launch, "{log:?}");
}

impl Fixture {
    /// Runs the frames, not the device.
    fn pump(&self) {
        loop {
            while let Some(frame) = self.frames.poll() {
                run_frame(frame);
            }
            let mut frame = None;
            self.completions
                .pull(&mut |f| frame = Some(f), &mut |_| false, 1);
            match frame {
                Some(frame) => run_frame(frame),
                None => return,
            }
        }
    }
}

/// What the work reads is held by the quantum and released only by the stages' `retired` hooks, after the work: not when
/// the run is enqueued, and not by a retirement another party arms on the lane while the work is queued. (With caching
/// off the values are the quantum's alone.)
#[test]
fn what_a_run_reads_is_released_by_the_hook_after_the_work_and_by_nothing_before() {
    let f = fixture();
    f.cache.set_enabled(false);
    // Another party arms a retirement on lane 0 ahead of the quantum's work: it fires while that work is queued.
    let other = Arc::new(euhedral_graph::RetireCell::new());
    f.lanes.pool.lane(LaneId(0)).notify_retired(&other).unwrap();
    f.start(1);
    f.pump();
    while other.status().is_none() {
        assert!(f.gpu.step());
        f.pump();
    }
    assert_eq!(
        word(&f.gpu, f.word),
        0,
        "the quantum's work is still queued"
    );
    assert_eq!(f.kept.load(Ordering::SeqCst), 0, "nothing is released yet");
    assert_eq!(f.hooks.load(Ordering::SeqCst), 0);
    f.settle();
    assert_eq!(word(&f.gpu, f.word), 11);
    assert_eq!(f.hooks.load(Ordering::SeqCst), 2);
    assert_eq!(
        f.kept.load(Ordering::SeqCst),
        2,
        "released by the hooks, once each"
    );
}

/// A quantum run with the device held back has not retired: no hook has run.
#[test]
fn a_quantum_whose_work_is_queued_has_not_retired_and_no_hook_has_run() {
    let f = fixture();
    f.start(1);
    f.pump();
    assert_eq!(f.hooks.load(Ordering::SeqCst), 0);
    assert!(lock(&f.retired).is_empty());
    f.settle();
    assert_eq!(f.hooks.load(Ordering::SeqCst), 2);
}
