//! A stage that answers `StageError::Retry` allocates nothing when it runs again, after warm-up: its wake-up is pooled
//! with its quantum's run, as the frames are. `RetryCounts` stay exact. This binary counts every allocation of the
//! process, so it holds one test and runs alone.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::time::Duration;

use euhedral_core::frames::{Frame, FrameError};
use euhedral_core::generics::{FrameSink, LatticeSource};
use euhedral_data_structures::queues::{Consumer, MpscQueue, Multi, Producer, Single};
use euhedral_graph::{
    Completions, DeviceQueue, GraphRuntime, LanePool, QuantumOutcome, QueueId, RetryCounts, Stage,
    StageCx, StageError, StageGraph, StageGraphSpec, StageSpec, Submitted,
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

/// A lane whose progress the test makes.
#[derive(Debug)]
struct HandLane {
    id: QueueId,
    completed: AtomicU64,
}

impl DeviceQueue for HandLane {
    fn id(&self) -> QueueId {
        self.id
    }

    fn completed(&self) -> u64 {
        self.completed.load(Ordering::Acquire)
    }

    fn name(&self) -> &str {
        "hand"
    }
}

/// Answers `Retry` as many times as its quantum says, then finishes.
struct Flaky {
    runs: AtomicU32,
}

impl Stage<u64> for Flaky {
    fn run(&self, retries: &u64, _cx: &StageCx<'_>) -> Result<Submitted, StageError> {
        if u64::from(self.runs.fetch_add(1, Ordering::Relaxed)) < *retries {
            return Err(StageError::Retry);
        }
        self.runs.store(0, Ordering::Relaxed);
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

struct World {
    hand: Arc<HandLane>,
    completions: Arc<Completions>,
    frames: Consumer<Box<dyn Frame>, Single>,
    graph: Arc<StageGraph<u64>>,
    counts: Arc<RetryCounts>,
    retired: Arc<AtomicUsize>,
    /// Whether a quiet moment makes the lane progress (a retried stage wakes on it) or the backstop does.
    progress: bool,
}

impl World {
    /// With `progress`, the lane advances whenever nothing else can run and the backstop is an hour away; without it, the
    /// lane never moves and the backstop has passed at once.
    fn new(progress: bool) -> Self {
        let hand = Arc::new(HandLane {
            id: QueueId::next(),
            completed: AtomicU64::new(0),
        });
        let pool = Arc::new(LanePool::new(vec![hand.clone() as Arc<dyn DeviceQueue>]));
        let completions = Completions::new(pool.clone());
        let queue = MpscQueue::<Box<dyn Frame>>::bounded(1024);
        let frames = queue.consumer();
        let mut spec = StageGraphSpec::new();
        spec.stage(StageSpec {
            name: "flaky",
            ordered: None,
            lanes: vec![],
        });
        let counts = Arc::new(RetryCounts::default());
        let retired = Arc::new(AtomicUsize::new(0));
        let counted = retired.clone();
        let graph = StageGraph::new(
            spec,
            vec![Arc::new(Flaky {
                runs: AtomicU32::new(0),
            }) as Arc<dyn Stage<u64>>],
            GraphRuntime {
                stages: Arc::new(Sink(queue.producer())),
                completions: completions.clone(),
                lanes: pool,
                retries: counts.clone(),
                retry_backstop: if progress {
                    Duration::from_secs(3600)
                } else {
                    Duration::ZERO
                },
            },
            move |_, outcome| {
                assert_eq!(outcome, QuantumOutcome::Completed);
                counted.fetch_add(1, Ordering::Relaxed);
            },
        )
        .expect("a valid graph");
        Self {
            hand,
            completions,
            frames,
            graph,
            counts,
            retired,
            progress,
        }
    }

    /// Runs frames until nothing is left to do.
    fn settle(&mut self) {
        loop {
            while let Some(frame) = self.frames.poll() {
                run_frame(frame);
            }
            let mut frame = None;
            self.completions
                .pull(&mut |f| frame = Some(f), &mut |_| false, 1);
            if let Some(frame) = frame {
                run_frame(frame);
                continue;
            }
            if self.progress && self.completions.pending() > 0 {
                self.hand.completed.fetch_add(1, Ordering::AcqRel);
                continue;
            }
            return;
        }
    }
}

const RETRIES: u64 = 3;
const WARM_UP: u64 = 50;
const MEASURED: u64 = 500;

/// `WARM_UP + MEASURED` quanta of `RETRIES` retries each; the measured ones allocate nothing.
fn retries_allocate_nothing(progress: bool) {
    let mut world = World::new(progress);
    let kill = Arc::new(AtomicBool::new(false));
    let round = |world: &mut World| {
        let _ = world.graph.start(RETRIES, kill.clone(), None);
        world.settle();
    };
    let before_warm_up = ALLOCATIONS.load(Ordering::Relaxed);
    for _ in 0..WARM_UP {
        round(&mut world);
    }
    assert!(
        ALLOCATIONS.load(Ordering::Relaxed) > before_warm_up,
        "warming up allocates the pooled runs and frames"
    );

    let before = ALLOCATIONS.load(Ordering::Relaxed);
    for _ in 0..MEASURED {
        round(&mut world);
    }
    let allocated = ALLOCATIONS.load(Ordering::Relaxed) - before;

    let total = WARM_UP + MEASURED;
    assert_eq!(world.retired.load(Ordering::Relaxed) as u64, total);
    assert_eq!(world.graph.live(), 0);
    assert_eq!(world.completions.pending(), 0);
    let counts = &world.counts;
    assert_eq!(counts.retries(), RETRIES * total, "every Retry counted");
    let (progressed, backstopped) = if progress {
        (RETRIES * total, 0)
    } else {
        (0, RETRIES * total)
    };
    assert_eq!(counts.progress(), progressed, "wakes by a lane's progress");
    assert_eq!(counts.backstops(), backstopped, "wakes by the backstop");
    assert_eq!(
        allocated, 0,
        "{allocated} allocations over {MEASURED} quanta of {RETRIES} retries each (progress wakes: {progress})"
    );
}

#[test]
fn a_retried_stage_allocates_nothing_when_it_runs_again() {
    // The counter sees allocations at all.
    let probe = ALLOCATIONS.load(Ordering::Relaxed);
    drop(std::hint::black_box(Box::new(7u64)));
    assert!(
        ALLOCATIONS.load(Ordering::Relaxed) > probe,
        "the counting allocator is live"
    );
    retries_allocate_nothing(true);
    retries_allocate_nothing(false);
}
