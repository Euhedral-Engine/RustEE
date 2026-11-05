//! Quanta started on one thread and run by 8 worker threads: the pools of runs and frames serve the takes, also when
//! takers race. This binary counts every allocation of the process, so it holds one test and runs alone.
//!
//! A take makes a new frame only when the pool is empty, so the frames ever made are at most the frames live at once
//! (or in the pool): after warm-up the pool still grows to the highest number of frames in flight at once, which depends
//! on how the workers interleave, and never beyond it. The test asserts that bound. A pool whose takers lose races to
//! each other makes a frame per lost race instead (about one take in eight with 8 workers), far past the bound.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

use euhedral_core::frames::{Frame, FrameError};
use euhedral_core::generics::{FrameSink, LatticeSource};
use euhedral_data_structures::queues::{Consumer, MpmcQueue, Multi, Producer};
use euhedral_graph::{
    Completions, DEFAULT_RETRY_BACKSTOP, DeviceQueue, EdgeKind, ExternalEdge, GraphRuntime, LaneId,
    LanePool, Marker, QuantumOutcome, QueueId, Stage, StageCx, StageError, StageGraph,
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

#[derive(Debug)]
struct HandLane {
    id: QueueId,
    submitted: AtomicU64,
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

struct Probe {
    hands: Vec<Arc<HandLane>>,
}

impl Stage<u64> for Probe {
    fn run(&self, _quantum: &u64, cx: &StageCx<'_>) -> Result<Submitted, StageError> {
        let lane = cx.lane.expect("every stage has a lane");
        let hand = &self.hands[usize::from(lane.0)];
        let value = hand.submitted.fetch_add(1, Ordering::AcqRel) + 1;
        Ok(Submitted::marker(Marker::new(hand.id, lane, value)))
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

const WORKERS: usize = 8;
const BATCH: u64 = 32;
const WARM_UP: u64 = 100;
const MEASURED: u64 = 400;

#[test]
#[allow(clippy::disallowed_methods)] // The test's own threads run frames, as lattice workers do.
fn quanta_run_by_8_workers_allocate_nothing_after_warm_up() {
    let probe = ALLOCATIONS.load(Ordering::Relaxed);
    drop(std::hint::black_box(Box::new(7u64)));
    assert!(
        ALLOCATIONS.load(Ordering::Relaxed) > probe,
        "the counting allocator is live"
    );

    let hands: Vec<Arc<HandLane>> = (0..4)
        .map(|_| {
            Arc::new(HandLane {
                id: QueueId::next(),
                submitted: AtomicU64::new(0),
                completed: AtomicU64::new(0),
            })
        })
        .collect();
    let pool = Arc::new(LanePool::new(
        hands
            .iter()
            .map(|h| h.clone() as Arc<dyn DeviceQueue>)
            .collect(),
    ));
    let completions = Completions::new(pool.clone());
    let queue = MpmcQueue::<Box<dyn Frame>>::bounded(4096);
    let frames: Consumer<Box<dyn Frame>, Multi> = queue.consumer();
    let sink = Arc::new(Sink(queue.producer()));
    // A diamond over four lanes: the root feeds two branches on other lanes, and the join follows both.
    let mut spec = StageGraphSpec::new();
    for lane in 0..4u16 {
        spec.stage(StageSpec {
            name: "probe",
            ordered: None,
            lanes: vec![LaneId(lane)],
        });
    }
    spec.edge(StageId(0), StageId(1), EdgeKind::Submission)
        .edge(StageId(0), StageId(2), EdgeKind::Submission)
        .edge(StageId(1), StageId(3), EdgeKind::Submission)
        .edge(StageId(2), StageId(3), EdgeKind::Submission);
    let retired = Arc::new(AtomicU64::new(0));
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
            counted.fetch_add(1, Ordering::Release);
        },
    )
    .expect("a valid graph");

    let done = Arc::new(AtomicBool::new(false));
    let workers: Vec<_> = (0..WORKERS)
        .map(|_| {
            let (frames, completions, hands, done) = (
                frames.clone(),
                completions.clone(),
                hands.clone(),
                done.clone(),
            );
            std::thread::spawn(move || {
                while !done.load(Ordering::Acquire) {
                    if let Some(frame) = frames.poll() {
                        run_frame(frame);
                        continue;
                    }
                    let mut frame = None;
                    completions.reap();
                    completions.pull(&mut |f| frame = Some(f), &mut |_| false, 1);
                    if let Some(frame) = frame {
                        run_frame(frame);
                        continue;
                    }
                    for hand in &hands {
                        hand.completed
                            .store(hand.submitted.load(Ordering::Acquire), Ordering::Release);
                    }
                    std::thread::yield_now();
                }
            })
        })
        .collect();

    let kill = Arc::new(AtomicBool::new(false));
    let mut started = 0u64;
    let mut batch = |count: u64| {
        for _ in 0..count {
            let _ = graph.start(started, kill.clone(), None::<ExternalEdge>);
            started += 1;
        }
        while retired.load(Ordering::Acquire) < started {
            std::thread::yield_now();
        }
    };
    for _ in 0..WARM_UP {
        batch(BATCH);
    }
    let before = ALLOCATIONS.load(Ordering::Relaxed);
    for _ in 0..MEASURED {
        batch(BATCH);
    }
    let allocated = ALLOCATIONS.load(Ordering::Relaxed) - before;
    done.store(true, Ordering::Release);
    for worker in workers {
        worker.join().unwrap();
    }
    assert_eq!(graph.live(), 0);
    // A frame, with the two marker lists it grows on first use. A quantum of this graph has at most 8 frames live at
    // once (4 stages, 4 lanes' boundary), and `BATCH` quanta are in flight.
    let bound = 3 * 8 * BATCH as usize;
    assert!(
        allocated <= bound,
        "{allocated} allocations over {} quanta of 4 stages on {WORKERS} workers, above the {bound} of a pool that \
         only grows to the frames in flight",
        MEASURED * BATCH
    );
}
