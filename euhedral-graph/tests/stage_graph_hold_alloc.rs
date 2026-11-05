//! A stage that holds itself at a gate from inside `run` allocates nothing, after warm-up, whoever releases the gate:
//! the driver once the stage is parked, the stage's own run before it returns, or a thread of its own. Nothing is
//! retried: no registration with the completions, and `RetryCounts` stay at zero. This binary counts every allocation of
//! the process, so it holds one test and runs alone.
#![allow(
    clippy::disallowed_types,
    reason = "test fixture: the slot the parked gate waits in is a standard lock, taken by the driver and the releaser"
)]
#![allow(
    clippy::disallowed_methods,
    reason = "test fixture: the releaser is the test's own thread, started before anything is measured; it parks until told, and the driver spins until it is done"
)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use euhedral_core::frames::{Frame, FrameError};
use euhedral_core::generics::FrameSink;
use euhedral_data_structures::queues::{Consumer, MpscQueue, Multi, Producer, Single};
use euhedral_graph::{
    Completions, DEFAULT_RETRY_BACKSTOP, DeviceQueue, EdgeKind, Gate, GraphRuntime, LanePool,
    QuantumOutcome, QueueId, RetryCounts, Stage, StageCx, StageError, StageGraph, StageGraphSpec,
    StageId, StageSpec, Submitted,
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

/// Who releases the gate.
#[derive(Clone, Copy, Debug)]
enum Release {
    /// The driver, once the stage is parked.
    Driver,
    /// The stage itself, before its run returns.
    BeforeReturn,
    /// A thread started before the measurement, once the stage is parked.
    Thread,
}

/// Finds its input not ready on the first of every two runs, and holds; finishes on the second.
struct Waiter {
    runs: AtomicU32,
    release: Release,
    parked: Arc<Mutex<Option<Gate>>>,
}

impl Stage<u64> for Waiter {
    fn run(&self, _quantum: &u64, cx: &StageCx<'_>) -> Result<Submitted, StageError> {
        if self.runs.fetch_add(1, Ordering::AcqRel).is_multiple_of(2) {
            let gate = cx.hold();
            match self.release {
                Release::BeforeReturn => gate.release(),
                Release::Driver | Release::Thread => {
                    *self.parked.lock().unwrap() = Some(gate);
                }
            }
            return Err(StageError::Held);
        }
        Ok(Submitted::done())
    }
}

struct Follower;

impl Stage<u64> for Follower {
    fn run(&self, _quantum: &u64, _cx: &StageCx<'_>) -> Result<Submitted, StageError> {
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

/// A thread that releases the gate in `parked` when told to. It is started before anything is measured, and tells
/// the driver it is done through a flag: nothing it does allocates.
struct Releaser {
    go: Arc<AtomicBool>,
    done: Arc<AtomicBool>,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl Releaser {
    fn start(parked: Arc<Mutex<Option<Gate>>>) -> Self {
        let go = Arc::new(AtomicBool::new(false));
        let done = Arc::new(AtomicBool::new(false));
        let stop = Arc::new(AtomicBool::new(false));
        let thread = {
            let (go, done, stop) = (go.clone(), done.clone(), stop.clone());
            thread::spawn(move || {
                loop {
                    while !go.swap(false, Ordering::Acquire) {
                        if stop.load(Ordering::Acquire) {
                            return;
                        }
                        thread::park();
                    }
                    let gate = parked.lock().unwrap().take();
                    gate.expect("a parked gate").release();
                    done.store(true, Ordering::Release);
                }
            })
        };
        Self {
            go,
            done,
            stop,
            thread: Some(thread),
        }
    }

    /// Has the thread release the parked gate, and returns once it has.
    fn release(&self) {
        let handle = self.thread.as_ref().expect("running");
        self.done.store(false, Ordering::Relaxed);
        self.go.store(true, Ordering::Release);
        handle.thread().unpark();
        // A releaser that panicked ends its thread: the test fails instead of waiting for ever.
        let deadline = Instant::now() + Duration::from_secs(30);
        while !self.done.load(Ordering::Acquire) {
            assert!(!handle.is_finished(), "the releaser thread ended");
            assert!(
                Instant::now() < deadline,
                "the releaser did not release in time"
            );
            std::hint::spin_loop();
        }
    }
}

impl Drop for Releaser {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(thread) = self.thread.take() {
            thread.thread().unpark();
            thread.join().expect("the releaser ended");
        }
    }
}

struct World {
    completions: Arc<Completions>,
    frames: Consumer<Box<dyn Frame>, Single>,
    graph: Arc<StageGraph<u64>>,
    counts: Arc<RetryCounts>,
    retired: Arc<AtomicUsize>,
    waiter: Arc<Waiter>,
    parked: Arc<Mutex<Option<Gate>>>,
    releaser: Option<Releaser>,
}

impl World {
    fn new(release: Release) -> Self {
        let pool = Arc::new(LanePool::new(vec![
            Arc::new(IdleLane(QueueId::next())) as Arc<dyn DeviceQueue>
        ]));
        let completions = Completions::new(pool.clone());
        let queue = MpscQueue::<Box<dyn Frame>>::bounded(1024);
        let frames = queue.consumer();
        let mut spec = StageGraphSpec::new();
        for name in ["waiter", "follower"] {
            spec.stage(StageSpec {
                name,
                ordered: None,
                lanes: vec![],
            });
        }
        spec.edge(StageId(0), StageId(1), EdgeKind::Submission);
        let parked: Arc<Mutex<Option<Gate>>> = Arc::default();
        let waiter = Arc::new(Waiter {
            runs: AtomicU32::new(0),
            release,
            parked: parked.clone(),
        });
        let counts = Arc::new(RetryCounts::default());
        let retired = Arc::new(AtomicUsize::new(0));
        let counted = retired.clone();
        let graph = StageGraph::new(
            spec,
            vec![
                waiter.clone() as Arc<dyn Stage<u64>>,
                Arc::new(Follower) as Arc<dyn Stage<u64>>,
            ],
            GraphRuntime {
                stages: Arc::new(Sink(queue.producer())),
                completions: completions.clone(),
                lanes: pool,
                retries: counts.clone(),
                retry_backstop: DEFAULT_RETRY_BACKSTOP,
            },
            move |_, outcome| {
                assert_eq!(outcome, QuantumOutcome::Completed);
                counted.fetch_add(1, Ordering::Relaxed);
            },
        )
        .expect("a valid graph");
        let releaser = matches!(release, Release::Thread).then(|| Releaser::start(parked.clone()));
        Self {
            completions,
            frames,
            graph,
            counts,
            retired,
            waiter,
            parked,
            releaser,
        }
    }

    /// Runs frames, and releases the parked gate, until nothing is left to do.
    fn settle(&mut self) {
        loop {
            while let Some(frame) = self.frames.poll() {
                run_frame(frame);
            }
            assert_eq!(self.completions.pending(), 0, "nothing is registered");
            if self.parked.lock().unwrap().is_none() {
                return;
            }
            match &self.releaser {
                Some(releaser) => releaser.release(),
                None => self
                    .parked
                    .lock()
                    .unwrap()
                    .take()
                    .expect("a parked gate")
                    .release(),
            }
        }
    }
}

const WARM_UP: u64 = 50;
const MEASURED: u64 = 500;

fn holds_allocate_nothing(release: Release) {
    let mut world = World::new(release);
    let kill = Arc::new(AtomicBool::new(false));
    let round = |world: &mut World, q: u64| {
        let _ = world.graph.start(q, kill.clone(), None);
        world.settle();
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

    let total = WARM_UP + MEASURED;
    assert_eq!(world.retired.load(Ordering::Relaxed) as u64, total);
    assert_eq!(world.graph.live(), 0);
    assert_eq!(
        u64::from(world.waiter.runs.load(Ordering::Relaxed)),
        2 * total,
        "each stage ran exactly twice per quantum: held, then released"
    );
    let counts = &world.counts;
    assert_eq!(
        (counts.retries(), counts.progress(), counts.backstops()),
        (0, 0, 0),
        "no retry"
    );
    assert_eq!(
        allocated, 0,
        "{allocated} allocations over {MEASURED} quanta that each held a stage once ({release:?})"
    );
}

#[test]
fn a_stage_held_from_inside_run_allocates_nothing() {
    // The counter sees allocations at all.
    let probe = ALLOCATIONS.load(Ordering::Relaxed);
    drop(std::hint::black_box(Box::new(7u64)));
    assert!(
        ALLOCATIONS.load(Ordering::Relaxed) > probe,
        "the counting allocator is live"
    );
    for release in [Release::Driver, Release::BeforeReturn, Release::Thread] {
        holds_allocate_nothing(release);
    }
}
