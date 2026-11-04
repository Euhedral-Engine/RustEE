//! A stage graph's quantum allocates nothing after warm-up: its run, its frames, its markers and its external edges
//! are pooled. This binary counts every allocation of the process, so it holds one test and runs alone.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

use euhedral_core::frames::{Frame, FrameError};
use euhedral_core::generics::{FrameSink, LatticeSource};
use euhedral_data_structures::queues::{Consumer, MpscQueue, Multi, Producer, Single};
use euhedral_graph::{
    Completions, DEFAULT_RETRY_BACKSTOP, DeviceQueue, EdgeKind, ExternalEdge, GraphRuntime, LaneId,
    LanePool, Marker, QuantumOutcome, QuantumRef, QueueId, Stage, StageCx, StageError, StageGraph,
    StageGraphSpec, StageId, StageSpec, Submitted,
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

/// A lane whose work completes when the test says so.
#[derive(Debug)]
struct HandLane {
    id: QueueId,
    submitted: AtomicU64,
    completed: AtomicU64,
}

impl HandLane {
    fn new() -> Self {
        Self {
            id: QueueId::next(),
            submitted: AtomicU64::new(0),
            completed: AtomicU64::new(0),
        }
    }

    fn pending(&self) -> bool {
        self.completed.load(Ordering::Acquire) < self.submitted.load(Ordering::Acquire)
    }

    fn finish(&self) {
        self.completed
            .store(self.submitted.load(Ordering::Acquire), Ordering::Release);
    }
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

/// Submits to its lane, or does CPU work.
struct Probe {
    hands: Vec<Arc<HandLane>>,
}

impl Stage<u64> for Probe {
    fn run(&self, _quantum: &u64, cx: &StageCx<'_>) -> Result<Submitted, StageError> {
        Ok(match cx.lane {
            Some(lane) => {
                let hand = &self.hands[usize::from(lane.0)];
                let value = hand.submitted.fetch_add(1, Ordering::AcqRel) + 1;
                Submitted::marker(Marker::new(hand.id, lane, value))
            }
            None => Submitted::done(),
        })
    }
}

/// Where the graph's stage frames queue: a bounded queue the test drains.
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
    hands: Vec<Arc<HandLane>>,
    completions: Arc<Completions>,
    frames: Consumer<Box<dyn Frame>, Single>,
    graph: Arc<StageGraph<u64>>,
    retired: Arc<AtomicUsize>,
}

impl World {
    fn new() -> Self {
        let hands: Vec<Arc<HandLane>> = (0..2).map(|_| Arc::new(HandLane::new())).collect();
        let pool = Arc::new(LanePool::new(
            hands
                .iter()
                .map(|h| h.clone() as Arc<dyn DeviceQueue>)
                .collect(),
        ));
        let completions = Completions::new(pool.clone());
        let queue = MpscQueue::<Box<dyn Frame>>::bounded(1024);
        let frames = queue.consumer();
        let sink = Arc::new(Sink(queue.producer()));
        // Stage 0 on lane 0 feeds stage 1 on lane 1 (a device wait) and, through a completion edge, a CPU stage 2;
        // the join, stage 3 on lane 0, follows both.
        let mut spec = StageGraphSpec::new();
        let lanes: [&[u16]; 4] = [&[0], &[1], &[], &[0]];
        for lanes in lanes {
            spec.stage(StageSpec {
                name: "probe",
                ordered: None,
                lanes: lanes.iter().map(|&l| LaneId(l)).collect(),
            });
        }
        spec.edge(StageId(0), StageId(1), EdgeKind::Submission)
            .edge(StageId(0), StageId(2), EdgeKind::Completion)
            .edge(StageId(1), StageId(3), EdgeKind::Submission)
            .edge(StageId(2), StageId(3), EdgeKind::Submission);
        let retired = Arc::new(AtomicUsize::new(0));
        let counted = retired.clone();
        let stages: Vec<Arc<dyn Stage<u64>>> = (0..4)
            .map(|_| {
                Arc::new(Probe {
                    hands: hands.clone(),
                }) as Arc<dyn Stage<u64>>
            })
            .collect();
        let graph = StageGraph::new(
            spec,
            stages,
            GraphRuntime {
                stages: sink,
                completions: completions.clone(),
                lanes: pool,
                retries: Arc::default(),
                retry_backstop: DEFAULT_RETRY_BACKSTOP,
            },
            move |_, outcome| {
                assert_eq!(outcome, QuantumOutcome::Completed);
                counted.fetch_add(1, Ordering::Relaxed);
            },
        )
        .expect("a valid graph");
        Self {
            hands,
            completions,
            frames,
            graph,
            retired,
        }
    }

    /// Runs frames, and completes the lanes' work, until nothing is left to do.
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
            if self.hands.iter().any(|h| h.pending()) {
                self.hands.iter().for_each(|h| h.finish());
                continue;
            }
            return;
        }
    }
}

const WARM_UP: u64 = 300;
const MEASURED: u64 = 3_000;

#[test]
fn a_quantum_allocates_nothing_after_warm_up() {
    // The counter sees allocations at all.
    let probe = ALLOCATIONS.load(Ordering::Relaxed);
    drop(std::hint::black_box(Box::new(7u64)));
    assert!(
        ALLOCATIONS.load(Ordering::Relaxed) > probe,
        "the counting allocator is live"
    );

    let mut world = World::new();
    let kill = Arc::new(AtomicBool::new(false));
    // Each quantum names the stage 3 of the one before it: a submission edge into its root on even rounds, a
    // completion edge on odd ones. The work is settled after every second quantum, so that an edge is satisfied
    // either at once (the stage has submitted) or later (the edge waits on the stage's stack).
    let mut previous: Option<QuantumRef> = None;
    let mut round = |world: &mut World, q: u64| {
        let edge = previous.take().map(|from| ExternalEdge {
            from,
            from_stage: StageId(3),
            to_stage: StageId(0),
            kind: if q.is_multiple_of(2) {
                EdgeKind::Submission
            } else {
                EdgeKind::Completion
            },
        });
        previous = Some(world.graph.start(q, kill.clone(), edge));
        if !q.is_multiple_of(2) {
            world.settle();
        }
    };

    let before_warm_up = ALLOCATIONS.load(Ordering::Relaxed);
    for q in 0..WARM_UP {
        round(&mut world, q);
    }
    assert!(
        ALLOCATIONS.load(Ordering::Relaxed) > before_warm_up,
        "warming up allocates the pooled runs and frames"
    );

    let before = ALLOCATIONS.load(Ordering::Relaxed);
    for q in WARM_UP..WARM_UP + MEASURED {
        round(&mut world, q);
    }
    let allocated = ALLOCATIONS.load(Ordering::Relaxed) - before;
    drop(previous);
    world.settle();

    assert_eq!(
        world.retired.load(Ordering::Relaxed) as u64,
        WARM_UP + MEASURED,
        "every quantum retired"
    );
    assert_eq!(world.graph.live(), 0);
    assert_eq!(
        allocated, 0,
        "{allocated} allocations over {MEASURED} quanta of 4 stages, each with an external edge"
    );
}
