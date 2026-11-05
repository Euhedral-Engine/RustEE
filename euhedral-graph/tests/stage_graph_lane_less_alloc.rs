//! A quantum whose stages name no lane allocates nothing after warm-up and costs exactly the frames of its stages
//! (run, kill and abandon alike): it concludes in the call that ends it. This binary counts every allocation of the
//! process, so it holds one test and runs alone.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use euhedral_core::frames::{Frame, FrameError};
use euhedral_core::generics::FrameSink;
use euhedral_data_structures::queues::{Consumer, MpscQueue, Multi, Producer, Single};
use euhedral_graph::{
    Completions, DEFAULT_RETRY_BACKSTOP, DeviceQueue, EdgeKind, GraphRuntime, LanePool,
    QuantumOutcome, QueueId, Stage, StageCx, StageError, StageGraph, StageGraphSpec, StageId,
    StageSpec, Submitted,
};

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

struct Counting;

// SAFETY: every call is forwarded unchanged to the system allocator; the counter is an atomic.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: the caller's contract, forwarded.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: the caller's contract, forwarded.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: the caller's contract, forwarded.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static COUNTING: Counting = Counting;

/// A query-only lane: the pool's home lane for the quanta.
#[derive(Debug)]
struct IdleLane(QueueId);

impl DeviceQueue for IdleLane {
    fn id(&self) -> QueueId {
        self.0
    }

    fn completed(&self) -> u64 {
        0
    }

    fn name(&self) -> &str {
        "idle"
    }
}

/// Host work; the quantum number 2 mod 3 fails (with an empty message: building one allocates nothing).
struct HostStage;

impl Stage<u64> for HostStage {
    fn run(&self, quantum: &u64, _cx: &StageCx<'_>) -> Result<Submitted, StageError> {
        if quantum % 3 == 2 {
            return Err(StageError::Corrupt(String::new()));
        }
        Ok(Submitted::done())
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

/// Runs the frames queued, and counts them.
fn drain(frames: &mut Consumer<Box<dyn Frame>, Single>) -> usize {
    let mut ran = 0;
    while let Some(frame) = frames.poll() {
        run_frame(frame);
        ran += 1;
    }
    ran
}

const WARM_UP: u64 = 300;
const MEASURED: u64 = 3_000;

#[test]
fn a_quantum_with_no_lane_allocates_nothing_after_warm_up_and_runs_only_its_stages_frames() {
    let probe = ALLOCATIONS.load(Ordering::Relaxed);
    drop(std::hint::black_box(Box::new(7u64)));
    assert!(
        ALLOCATIONS.load(Ordering::Relaxed) > probe,
        "the counting allocator is live"
    );

    let pool = Arc::new(LanePool::new(vec![
        Arc::new(IdleLane(QueueId::next())) as Arc<dyn DeviceQueue>
    ]));
    let queue = MpscQueue::<Box<dyn Frame>>::bounded(1024);
    let mut frames = queue.consumer();
    let mut spec = StageGraphSpec::new();
    for _ in 0..2 {
        spec.stage(StageSpec {
            name: "host",
            ordered: None,
            lanes: Vec::new(),
        });
    }
    spec.edge(StageId(0), StageId(1), EdgeKind::Submission);
    let retired = Arc::new(AtomicUsize::new(0));
    let counted = retired.clone();
    let graph = StageGraph::new(
        spec,
        vec![
            Arc::new(HostStage) as Arc<dyn Stage<u64>>,
            Arc::new(HostStage),
        ],
        GraphRuntime {
            stages: Arc::new(Sink(queue.producer())),
            completions: Completions::new(pool.clone()),
            lanes: pool,
            retries: Arc::default(),
            retry_backstop: DEFAULT_RETRY_BACKSTOP,
        },
        move |quantum, outcome| {
            let as_expected = match quantum % 3 {
                0 => outcome == QuantumOutcome::Completed,
                1 => outcome == QuantumOutcome::Killed,
                _ => matches!(outcome, QuantumOutcome::Failed { .. }),
            };
            assert!(as_expected, "quantum {quantum}: {outcome:?}");
            counted.fetch_add(1, Ordering::Relaxed);
        },
    )
    .expect("a valid graph");

    let killed = Arc::new(AtomicBool::new(true));
    let alive = Arc::new(AtomicBool::new(false));
    // Per quantum number mod 3: run (2 stages, 2 frames), kill (the root's frame ends it: 1), abandon (the root fails:
    // 1 frame).
    let round = |frames: &mut Consumer<Box<dyn Frame>, Single>, q: u64| {
        let kill = if q % 3 == 1 { &killed } else { &alive };
        let _ = graph.start(q, kill.clone(), Vec::new());
        let ran = drain(frames);
        let expected = if q.is_multiple_of(3) { 2 } else { 1 };
        assert_eq!(ran, expected, "frames of quantum {q}");
    };
    for q in 0..WARM_UP {
        round(&mut frames, q);
    }
    let before = ALLOCATIONS.load(Ordering::Relaxed);
    for q in WARM_UP..WARM_UP + MEASURED {
        round(&mut frames, q);
    }
    let allocated = ALLOCATIONS.load(Ordering::Relaxed) - before;
    assert_eq!(
        retired.load(Ordering::Relaxed) as u64,
        WARM_UP + MEASURED,
        "every quantum retired in its own frames"
    );
    assert_eq!(graph.live(), 0);
    assert_eq!(
        allocated, 0,
        "{allocated} allocations over {MEASURED} quanta with no lane"
    );
}
