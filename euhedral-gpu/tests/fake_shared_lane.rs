//! Shared lanes on the fake backend: any thread submits to a lane through `&self`, a stage of a graph submits to its
//! lane through its context, each retirement names the cell its callback fills, and what a stage holds for its work
//! is released by its `retired` hook once the device work ended. No GPU, no wall clock.
#![cfg(feature = "fake")]
#![allow(
    clippy::disallowed_types,
    clippy::disallowed_methods,
    reason = "test fixtures: the test locks its records and runs peer threads that stand in for workers"
)]

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use euhedral_core::frames::{DummyFrame, Frame, FrameError};
use euhedral_core::generics::{FrameSink, LatticeSource};
use euhedral_data_structures::queues::{Consumer, MpmcQueue, Multi, Producer};
use euhedral_gpu::fake::{FakeFault, FakeGpu, FakeKernel, FakeModule, FakeOp, FakeQueue};
use euhedral_gpu::{
    Args, CompileOptions, GpuError, Lane, LaneEnqueue, LaneStream, Launch, Module, SharedLanes,
    StreamOptions, open_shared_lanes,
};
use euhedral_graph::{
    Completions, DEFAULT_RETRY_BACKSTOP, EdgeKind, GraphRuntime, LaneControl, LaneId,
    QuantumOutcome, RetireCell, Stage, StageCx, StageError, StageGraph, StageGraphSpec, StageSpec,
    Submitted,
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
    gpu.define("boom", |_, _, _| Err(FakeFault::IllegalAddress));
    gpu
}

fn kernel(gpu: &Arc<FakeGpu>) -> FakeKernel {
    FakeModule::from_source(gpu, "// fake source", &CompileOptions::default())
        .unwrap()
        .kernel("add1")
        .unwrap()
}

fn boom(gpu: &Arc<FakeGpu>) -> FakeKernel {
    FakeModule::from_source(gpu, "// fake source", &CompileOptions::default())
        .unwrap()
        .kernel("boom")
        .unwrap()
}

fn word(gpu: &FakeGpu, at: u64) -> u32 {
    u32::from_le_bytes(gpu.read(at, 4).try_into().unwrap())
}

fn open(gpu: &Arc<FakeGpu>, lanes: usize) -> SharedLanes<FakeQueue> {
    open_with(gpu, lanes, 1024)
}

fn open_with(gpu: &Arc<FakeGpu>, lanes: usize, markers: u32) -> SharedLanes<FakeQueue> {
    let names = ["a", "b", "c", "d"];
    let specs: Vec<(&str, StreamOptions)> = names[..lanes]
        .iter()
        .map(|name| {
            (
                *name,
                StreamOptions {
                    markers,
                    ..StreamOptions::default()
                },
            )
        })
        .collect();
    open_shared_lanes(gpu, &specs).unwrap()
}

/// Eight threads submit a launch and make its marker on one lane, at once: every marker is distinct, the numbers are
/// contiguous, each launch ran once, and the device completes all of them.
#[test]
fn many_workers_submit_to_one_lane_and_every_marker_completes() {
    let gpu = gpu();
    let kernel = kernel(&gpu);
    let word_at = gpu.alloc(4);
    let lanes = open(&gpu, 1);
    let lane = lanes.pool.lane(LaneId(0)).clone();
    let markers = Mutex::new(Vec::new());
    let (workers, each) = if cfg!(miri) { (3, 4) } else { (8, 50) };
    std::thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| {
                for _ in 0..each {
                    let marker = loop {
                        let made = lane.submit_marked(|q| {
                            // SAFETY: a live fake allocation.
                            unsafe {
                                q.launch(
                                    &kernel,
                                    Launch::linear(1, 1),
                                    &Args::new().device(word_at),
                                )
                            }
                        });
                        match made {
                            Ok(((), marker)) => break marker,
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
        u64::from(word(&gpu, word_at)),
        total,
        "every launch ran once (a refused submission ran no work)"
    );
}

/// A submission that cannot be marked is refused before any of its work: the closure does not run.
#[test]
fn a_full_marker_ring_refuses_a_submission_before_its_work() {
    let gpu = gpu();
    let lanes = open_with(&gpu, 1, 1);
    let lane = lanes.pool.lane(LaneId(0)).clone();
    let ran = AtomicUsize::new(0);
    lane.submit_marked(|_| {
        ran.fetch_add(1, Ordering::SeqCst);
        Ok(())
    })
    .unwrap();
    let refused = lane.submit_marked(|_| {
        ran.fetch_add(1, Ordering::SeqCst);
        Ok(())
    });
    assert!(matches!(refused, Err(GpuError::Busy)));
    assert_eq!(ran.load(Ordering::SeqCst), 1, "the work was never entered");
    gpu.run_until_idle();
    lane.submit_marked(|_| {
        ran.fetch_add(1, Ordering::SeqCst);
        Ok(())
    })
    .unwrap();
    assert_eq!(ran.load(Ordering::SeqCst), 2, "the slot is free again");
}

/// A submission whose work fails leaves no marker and holds no slot.
#[test]
fn a_failed_submission_gives_its_marker_slot_back() {
    let gpu = gpu();
    let lanes = open_with(&gpu, 1, 1);
    let lane = lanes.pool.lane(LaneId(0)).clone();
    for _ in 0..3 {
        let failed: Result<((), _), _> =
            lane.submit_marked(|_| Err(GpuError::Invalid("the work was refused".into())));
        assert!(matches!(failed, Err(GpuError::Invalid(_))));
    }
    lane.submit_marked(|_| Ok(())).unwrap();
}

/// Waits for a retirement cell to be filled: a frame emitted when the callback has run.
fn after_retired(completions: &Arc<Completions>, cell: &Arc<RetireCell>) {
    completions.after_retired(cell.clone(), Box::new(DummyFrame::INSTANCE));
}

fn pull(completions: &Arc<Completions>) -> usize {
    let mut frames = Vec::new();
    completions.pull(&mut |f| frames.push(f), &mut |_| false, 8);
    frames.len()
}

/// The device's retirement callback only fills the cell its arm named; a frame emitted when it is filled settles it.
#[test]
fn a_retirement_fills_the_cell_it_was_armed_with_and_a_frame_settles_it() {
    let gpu = gpu();
    let kernel = kernel(&gpu);
    let word_at = gpu.alloc(4);
    let lanes = open(&gpu, 1);
    let lane = lanes.pool.lane(LaneId(0)).clone();
    lane.submit(|q| {
        // SAFETY: a live fake allocation.
        unsafe { q.launch(&kernel, Launch::linear(1, 1), &Args::new().device(word_at)) }
    })
    .unwrap();
    let cell = Arc::new(RetireCell::new());
    lane.notify_retired(&cell).unwrap();
    let completions = Completions::new(lanes.queues.clone());
    after_retired(&completions, &cell);
    assert_eq!(pull(&completions), 0, "the device has not run");
    assert_eq!(cell.status(), None);
    gpu.run_until_idle();
    assert_eq!(cell.status(), Some(0));
    assert_eq!(pull(&completions), 1, "the continuation is a frame");
    assert_eq!(lane.confirm_retired(0).map(|e| e.to_string()), None);
    assert_eq!(pull(&completions), 0, "once");
}

/// Two parties arm retirements on one lane: each reads the status of its own, however their arms and the device's
/// callbacks interleave, and nothing is claimed.
#[test]
fn parties_arming_on_one_lane_each_read_their_own_retirement() {
    let gpu = gpu();
    let kernel = kernel(&gpu);
    let word_at = gpu.alloc(4);
    let lanes = open(&gpu, 1);
    let lane = lanes.pool.lane(LaneId(0)).clone();
    let first = Arc::new(RetireCell::new());
    let second = Arc::new(RetireCell::new());
    lane.notify_retired(&first).unwrap();
    lane.submit(|q| {
        // SAFETY: a live fake allocation.
        unsafe { q.launch(&kernel, Launch::linear(1, 1), &Args::new().device(word_at)) }
    })
    .unwrap();
    lane.notify_retired(&second).unwrap();
    // The device reaches the first retirement before the launch and the second after it.
    while first.status().is_none() {
        assert!(gpu.step());
    }
    assert_eq!(word(&gpu, word_at), 0, "the launch is behind the first");
    assert_eq!(second.status(), None, "the second is not the first's");
    gpu.run_until_idle();
    assert_eq!(second.status(), Some(0));
    assert_eq!(word(&gpu, word_at), 1);
}

#[test]
fn many_parties_arming_at_once_each_see_exactly_their_own_retirement() {
    let gpu = gpu();
    let lanes = open(&gpu, 1);
    let lane = lanes.pool.lane(LaneId(0)).clone();
    let (parties, rounds) = if cfg!(miri) { (3, 3) } else { (8, 40) };
    let cells: Vec<Arc<RetireCell>> = (0..parties * rounds)
        .map(|_| Arc::new(RetireCell::new()))
        .collect();
    std::thread::scope(|scope| {
        for party in 0..parties {
            let (lane, cells) = (&lane, &cells);
            scope.spawn(move || {
                for round in 0..rounds {
                    lane.notify_retired(&cells[party * rounds + round]).unwrap();
                }
            });
        }
    });
    gpu.run_until_idle();
    assert!(cells.iter().all(|c| c.status() == Some(0)), "each its own");
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

struct Counted(Arc<AtomicUsize>);

impl Drop for Counted {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

/// A quantum of the tests: its number, and what its stages hold for their work (the host buffers a copy reads).
struct Batch {
    id: u64,
    held: Mutex<Vec<Counted>>,
}

impl Batch {
    fn new(id: u64) -> Self {
        Self {
            id,
            held: Mutex::default(),
        }
    }
}

/// A stage that orders itself behind the markers it was given, submits a launch on its lane with its marker (any `Busy`
/// comes before the work), holds a value for the launch, and releases it in its retirement hook.
struct Launches {
    kernel: FakeKernel,
    word: u64,
    holds: Option<Arc<AtomicUsize>>,
    runs: Arc<AtomicUsize>,
    hooks: Arc<AtomicUsize>,
}

impl Stage<Batch, FakeQueue> for Launches {
    fn run(&self, q: &Batch, cx: &StageCx<'_, FakeQueue>) -> Result<Submitted, StageError> {
        self.runs.fetch_add(1, Ordering::SeqCst);
        let lane = cx.lane_handle().expect("a stage with a lane");
        for marker in cx.waits {
            lane.await_marker(*marker)
                .map_err(|e| StageError::Corrupt(e.to_string()))?;
        }
        let made = lane.submit_marked(|e| {
            // SAFETY: a live fake allocation.
            unsafe {
                e.launch(
                    &self.kernel,
                    Launch::linear(1, 1),
                    &Args::new().device(self.word),
                )
            }
        });
        match made {
            Ok(((), marker)) => {
                if let Some(count) = &self.holds {
                    lock(&q.held).push(Counted(count.clone()));
                }
                Ok(Submitted::marker(marker))
            }
            Err(GpuError::Busy) => Err(StageError::Retry),
            Err(e) => Err(StageError::Corrupt(e.to_string())),
        }
    }

    fn retired(&self, q: &Batch, _committed: bool) {
        self.hooks.fetch_add(1, Ordering::SeqCst);
        lock(&q.held).clear();
    }
}

type Retired = Arc<Mutex<Vec<(u64, QuantumOutcome)>>>;

struct Fixture {
    gpu: Arc<FakeGpu>,
    lanes: SharedLanes<FakeQueue>,
    graph: Arc<StageGraph<Batch, FakeQueue>>,
    frames: Consumer<Box<dyn Frame>, Multi>,
    completions: Arc<Completions>,
    retired: Retired,
    word: u64,
    runs: Arc<AtomicUsize>,
    hooks: Arc<AtomicUsize>,
    held: Arc<AtomicUsize>,
}

impl Fixture {
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

    /// Runs device and frames until nothing moves.
    fn settle(&self) {
        loop {
            self.pump();
            if self.gpu.run_until_idle() == 0 {
                self.pump();
                return;
            }
        }
    }

    fn start(&self, id: u64) {
        let _ = self
            .graph
            .start(Batch::new(id), Arc::new(AtomicBool::new(false)), None);
    }
}

fn fixture(
    markers: u32,
    stages: &[Vec<LaneId>],
    edges: &[(usize, usize)],
    kernel_of: impl Fn(&Arc<FakeGpu>) -> FakeKernel,
) -> Fixture {
    let gpu = gpu();
    let kernel = kernel_of(&gpu);
    let at = gpu.alloc(4);
    let lanes = open_with(&gpu, 2, markers);
    let completions = Completions::new(lanes.queues.clone());
    let queue = MpmcQueue::<Box<dyn Frame>>::bounded(256);
    let frames: Consumer<Box<dyn Frame>, Multi> = queue.consumer();
    let mut spec = StageGraphSpec::new();
    let ids: Vec<_> = stages
        .iter()
        .map(|allowed| {
            spec.stage(StageSpec {
                name: "s",
                ordered: None,
                lanes: allowed.clone(),
            })
        })
        .collect();
    for &(a, b) in edges {
        spec.edge(ids[a], ids[b], EdgeKind::Submission);
    }
    let retired: Retired = Arc::new(Mutex::new(Vec::new()));
    let log = retired.clone();
    let runs = Arc::new(AtomicUsize::new(0));
    let hooks = Arc::new(AtomicUsize::new(0));
    let held = Arc::new(AtomicUsize::new(0));
    let impls = stages
        .iter()
        .map(|_| {
            Arc::new(Launches {
                kernel: kernel.clone(),
                word: at,
                holds: Some(held.clone()),
                runs: runs.clone(),
                hooks: hooks.clone(),
            }) as Arc<dyn Stage<Batch, FakeQueue>>
        })
        .collect();
    let graph = StageGraph::new(
        spec,
        impls,
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
        lanes,
        graph,
        frames,
        completions,
        retired,
        word: at,
        runs,
        hooks,
        held,
    }
}

#[test]
fn a_stage_submits_on_its_lane_and_a_successor_on_another_lane_awaits_its_marker() {
    let f = fixture(1024, &[vec![LaneId(0)], vec![LaneId(1)]], &[(0, 1)], kernel);
    f.start(7);
    f.pump();
    assert!(
        lock(&f.retired).is_empty(),
        "the device has not run: the boundary is ahead"
    );
    assert_eq!(f.gpu.log(), [], "nothing ran on the device yet");
    f.settle();
    assert_eq!(
        *lock(&f.retired),
        [(7, QuantumOutcome::Completed)],
        "the boundary covers both lanes"
    );
    assert_eq!(word(&f.gpu, f.word), 2);
    let log = f.gpu.log();
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

/// Java `quiesce`: the lanes a quantum used join its home lane on the device, and the one retirement armed there ends
/// the quantum: it does not end before the work of every lane did.
#[test]
fn a_quantum_does_not_retire_before_the_work_of_every_lane_it_used_ran() {
    let f = fixture(
        1024,
        &[vec![LaneId(0)], vec![LaneId(1)], vec![LaneId(0), LaneId(1)]],
        &[(0, 2), (1, 2)],
        kernel,
    );
    f.start(1);
    f.pump();
    // Run the device one operation at a time: the quantum retires only on the last.
    while f.gpu.step() {
        f.pump();
        if word(&f.gpu, f.word) < 3 {
            assert!(
                lock(&f.retired).is_empty(),
                "retired with only {} of 3 launches run",
                word(&f.gpu, f.word)
            );
        }
    }
    f.pump();
    assert_eq!(word(&f.gpu, f.word), 3);
    assert_eq!(*lock(&f.retired), [(1, QuantumOutcome::Completed)]);
}

/// What a stage holds for its work is released by its hook once the device work ended, and by nothing earlier: not by
/// a retirement another party armed on the lane while the stage's work was queued.
#[test]
fn what_a_stage_holds_is_released_by_its_hook_after_the_work_and_never_before() {
    let f = fixture(1024, &[vec![LaneId(0)]], &[], kernel);
    // Another party arms a retirement on the lane, then the stage submits behind it: the party's fires first, with the
    // stage's launch still queued.
    let other = Arc::new(RetireCell::new());
    f.lanes.pool.lane(LaneId(0)).notify_retired(&other).unwrap();
    f.start(1);
    f.pump();
    assert_eq!(f.held.load(Ordering::SeqCst), 0, "held, not yet dropped");
    while other.status().is_none() {
        assert!(f.gpu.step());
        f.pump();
    }
    assert_eq!(
        word(&f.gpu, f.word),
        0,
        "the stage's launch is still queued"
    );
    assert_eq!(
        f.held.load(Ordering::SeqCst),
        0,
        "another party's retirement releases nothing of the stage's"
    );
    f.settle();
    assert_eq!(word(&f.gpu, f.word), 1);
    assert_eq!(f.hooks.load(Ordering::SeqCst), 1);
    assert_eq!(
        f.held.load(Ordering::SeqCst),
        1,
        "released once, by the hook"
    );
    assert_eq!(*lock(&f.retired), [(1, QuantumOutcome::Completed)]);
}

/// Quanta one after another do not pile up what their stages hold: each is released as its quantum retires.
#[test]
fn values_held_by_stages_are_released_once_their_quanta_retire() {
    const QUANTA: u64 = 50;
    let f = fixture(1024, &[vec![LaneId(0)]], &[], kernel);
    for q in 1..=QUANTA {
        f.start(q);
        f.settle();
    }
    assert_eq!(
        lock(&f.retired).len(),
        QUANTA as usize,
        "every quantum retired"
    );
    assert!(
        lock(&f.retired)
            .iter()
            .all(|(_, o)| *o == QuantumOutcome::Completed)
    );
    assert_eq!(f.hooks.load(Ordering::SeqCst), QUANTA as usize);
    assert_eq!(f.held.load(Ordering::SeqCst), QUANTA as usize);
}

/// A stage whose mark is refused is retried before it submitted anything: a ring of one marker, two stages on one
/// lane, and each stage's work runs once.
#[test]
fn a_busy_marker_ring_makes_a_stage_retry_without_running_its_work_twice() {
    let f = fixture(1, &[vec![LaneId(0)], vec![LaneId(0)]], &[(0, 1)], kernel);
    f.start(1);
    f.settle();
    assert_eq!(*lock(&f.retired), [(1, QuantumOutcome::Completed)]);
    assert_eq!(
        word(&f.gpu, f.word),
        2,
        "two stages each launch once, though one was retried"
    );
    assert!(
        f.runs.load(Ordering::SeqCst) > 2,
        "the second stage was retried at least once"
    );
}

/// A device fault surfaces on the quantum's retirement as a failure of the quantum, and its hooks run, uncommitted.
#[test]
fn a_device_fault_fails_the_quantum_at_its_retirement_and_the_hooks_still_run() {
    let f = fixture(1024, &[vec![LaneId(0)]], &[], boom);
    f.start(1);
    f.settle();
    let retired = lock(&f.retired).clone();
    assert!(
        matches!(&retired[..], [(1, QuantumOutcome::DeviceFailed { .. })]),
        "{retired:?}"
    );
    assert_eq!(f.hooks.load(Ordering::SeqCst), 1);
    assert_eq!(f.held.load(Ordering::SeqCst), 1);
}

/// The closure of a marked submission is given the lane\'s raw stream for library calls; a refused marker comes before the
/// closure, so no library call is made.
#[test]
fn the_raw_stream_of_a_lane_is_given_inside_the_marked_submission_and_only_after_its_marker_is_claimed()
 {
    let gpu = gpu();
    let lanes = open_with(&gpu, 2, 1);
    let lane = lanes.pool.lane(LaneId(1)).clone();
    let seen = Mutex::new(Vec::new());
    lane.submit_marked(|q| {
        // SAFETY: used inside this closure only.
        lock(&seen).push(unsafe { q.raw_stream() });
        Ok(())
    })
    .unwrap();
    let raw = lock(&seen)[0];
    assert_eq!(raw.lane, LaneId(1), "the lane\'s own stream");
    // The ring of one marker is full: the next submission is refused before its closure runs.
    let refused = lane.submit_marked(|q| {
        // SAFETY: as above.
        lock(&seen).push(unsafe { q.raw_stream() });
        Ok(())
    });
    assert!(matches!(refused, Err(GpuError::Busy)));
    assert_eq!(
        lock(&seen).len(),
        1,
        "no library call is made behind a refused marker"
    );
}
