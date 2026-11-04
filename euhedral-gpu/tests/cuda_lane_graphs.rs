//! Recurring sequences recorded and replayed through shared lanes on a real device: a stage of a stage graph replays a
//! cached graph on its lane, workers that run one key at once each get an instance of their own, and what the work reads
//! is held by the quantum until the stage's hook. Every test skips, passing, when no driver or device is present.
#![cfg(feature = "cuda")]
#![allow(
    clippy::disallowed_types,
    clippy::disallowed_methods,
    reason = "test fixtures: peer threads stand in for the lattice's workers and the test waits for the device"
)]

mod common;

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Barrier, Mutex, PoisonError};
use std::time::Duration;

use common::cuda_support::{Pinned, device_word, gpu, module, wait_until};
use euhedral_core::frames::{Frame, FrameError};
use euhedral_core::generics::{FrameSink, LatticeSource};
use euhedral_data_structures::queues::{Consumer, MpmcQueue, Multi, Producer};
use euhedral_gpu::cuda::{CudaGpu, CudaKernel, CudaQueue};
use euhedral_gpu::{
    Args, Enqueue, GpuError, GraphBody, GraphHold, Lane, LaneEnqueue, Launch, Module, Ran,
    SharedGraphCache, SharedLanes, StreamOptions, open_shared_lanes,
};
use euhedral_graph::{
    Completions, DEFAULT_RETRY_BACKSTOP, EdgeKind, GraphRuntime, LaneControl, LaneId,
    QuantumOutcome, Stage, StageCx, StageError, StageGraph, StageGraphSpec, StageSpec, Submitted,
};

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

fn lanes(gpu: &Arc<CudaGpu>, count: usize) -> SharedLanes<CudaQueue> {
    let names = ["a", "b", "c", "d"];
    let specs: Vec<(&str, StreamOptions)> = names[..count]
        .iter()
        .map(|name| (*name, StreamOptions::default()))
        .collect();
    open_shared_lanes(gpu, &specs).unwrap()
}

struct Counted(Arc<AtomicUsize>);

impl Drop for Counted {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

/// Stage one sets the word to 3 (behind a spin, so that a second stage that did not wait would overtake it); stage two
/// doubles it. In the wrong order the word ends at 3, not 6.
struct Steps {
    axpb: CudaKernel,
    spin: CudaKernel,
    word: u64,
    last: bool,
    /// Called inside every pass, to hold threads together.
    inside: Option<Arc<Barrier>>,
    keeps: Option<Arc<AtomicUsize>>,
}

impl GraphBody<CudaGpu> for Steps {
    fn enqueue<E: Enqueue<Kernel = CudaKernel>>(&mut self, enq: &mut E) -> Result<(), GpuError> {
        if let Some(barrier) = &self.inside {
            barrier.wait();
        }
        let (a, b) = if self.last {
            (2.0f32, 0.0f32)
        } else {
            (0.0, 3.0)
        };
        // SAFETY: the device word is live; the kernel reads and writes its first float.
        unsafe {
            if !self.last {
                enq.launch(
                    &self.spin,
                    Launch::linear(1, 1),
                    &Args::new().arg(2_000_000u64),
                )?;
            }
            enq.launch(
                &self.axpb,
                Launch::linear(1, 1),
                &Args::new()
                    .device(self.word)
                    .device(self.word)
                    .arg(a)
                    .arg(b)
                    .arg(1u32),
            )?;
        }
        if let Some(count) = &self.keeps {
            enq.keep(Box::new(Counted(count.clone())));
        }
        Ok(())
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

/// A quantum: the device word its stages work on, and what they hold for their work.
struct Work {
    word: u64,
    hold: Mutex<GraphHold>,
}

struct Replays {
    cache: Arc<SharedGraphCache<CudaGpu, u32>>,
    gpu: Arc<CudaGpu>,
    axpb: CudaKernel,
    spin: CudaKernel,
    key: u32,
    last: bool,
    keeps: Arc<AtomicUsize>,
    hooks: Arc<AtomicUsize>,
}

impl Stage<Work, CudaQueue> for Replays {
    fn run(&self, work: &Work, cx: &StageCx<'_, CudaQueue>) -> Result<Submitted, StageError> {
        let lane = cx.lane_handle().expect("a stage with a lane");
        for marker in cx.waits {
            lane.await_marker(*marker)
                .map_err(|e| StageError::Corrupt(e.to_string()))?;
        }
        let mut body = Steps {
            axpb: self.axpb.clone(),
            spin: self.spin.clone(),
            word: work.word,
            last: self.last,
            inside: None,
            keeps: Some(self.keeps.clone()),
        };
        let mut hold = lock(&work.hold);
        let made = lane.submit_marked(|_| {
            self.cache
                .run(&self.gpu, lane, self.key, &mut body, &mut hold)
        });
        match made {
            Ok((_, marker)) => Ok(Submitted::marker(marker)),
            Err(GpuError::Busy) => Err(StageError::Retry),
            Err(e) => Err(StageError::Corrupt(e.to_string())),
        }
    }

    fn retired(&self, work: &Work, committed: bool) {
        assert!(committed);
        self.hooks.fetch_add(1, Ordering::SeqCst);
        lock(&work.hold).release();
    }
}

/// Four workers run quanta of a two-stage graph, each stage replaying a cached graph on its own lane, at once: the
/// second stage awaits the first\'s marker on the device (the word ends at 6, not 3), a quantum retires only when both
/// lanes reached their markers, and the stages\' hooks release what the work read.
#[test]
fn stages_replay_cached_graphs_on_their_lanes_and_the_boundary_covers_both() {
    let Some(gpu) = gpu() else { return };
    let Some(module) = module(&gpu) else { return };
    let (_mem, block) = device_word(&gpu);
    let pinned = Pinned::new(&gpu);
    let (axpb, spin) = (
        module.kernel("axpb").unwrap(),
        module.kernel("spin").unwrap(),
    );
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
    let cache = Arc::new(SharedGraphCache::<CudaGpu, u32>::new());
    let (keeps, hooks) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
    let stage = |key: u32, last: bool| {
        Arc::new(Replays {
            cache: cache.clone(),
            gpu: gpu.clone(),
            axpb: axpb.clone(),
            spin: spin.clone(),
            key,
            last,
            keeps: keeps.clone(),
            hooks: hooks.clone(),
        }) as Arc<dyn Stage<Work, CudaQueue>>
    };
    let outcomes = Arc::new(Mutex::new(Vec::new()));
    let log = outcomes.clone();
    let graph = StageGraph::new(
        spec,
        vec![stage(1, false), stage(2, true)],
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
    // Each quantum has its own device word, so the quanta overlap on the device and take their own arguments.
    for i in 0..QUANTA {
        let work = Work {
            word: block.addr + 4 * i as u64,
            hold: Mutex::default(),
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
    assert_eq!(hooks.load(Ordering::SeqCst), 2 * QUANTA);
    // Read the words back on lane 0, behind everything.
    let lane = lanes.pool.lane(LaneId(0)).clone();
    let host = pinned.word();
    for i in 0..QUANTA {
        let (_, marker) = lane
            .submit_marked(|q| {
                // SAFETY: the pinned block holds the words; the device words are live.
                unsafe { q.copy_to_host(host.bytes().add(4 * i), block.addr + 4 * i as u64, 4) }
            })
            .unwrap();
        assert!(wait_until(Duration::from_secs(10), || lanes
            .queues
            .reached(marker)));
    }
    for i in 0..QUANTA {
        // SAFETY: the copy ended (its marker was reached).
        let seen = unsafe { ((host.bytes() as *const u32).add(i)).read_volatile() };
        assert_eq!(
            f32::from_bits(seen),
            6.0,
            "quantum {i}: the second stage ran after the first"
        );
    }
    let stats = cache.stats();
    assert!(stats.records >= 2 && stats.replays + stats.records == 2 * QUANTA as u64);
    assert_eq!(graph.live(), 0);
    // The cache holds the recordings\' buffers; the quanta\'s holds are gone, and every update pass\'s value too.
    drop(cache);
    assert!(keeps.load(Ordering::SeqCst) >= 2 * QUANTA - 8);
}

/// Workers that run one key at once each get an instance of their own on a real device: they are held together inside the
/// passes, so a wait on a taken instance would never end, and every replay runs.
#[test]
fn workers_running_one_key_at_once_each_get_an_instance_and_every_replay_runs() {
    const N: usize = 6;
    let Some(gpu) = gpu() else { return };
    let Some(module) = module(&gpu) else { return };
    let (_mem, block) = device_word(&gpu);
    let pinned = Pinned::new(&gpu);
    let add = module.kernel("add_u32").unwrap();
    let lanes = lanes(&gpu, 1);
    let lane = lanes.pool.lane(LaneId(0)).clone();
    let cache = SharedGraphCache::<CudaGpu, u32>::with_capacity(32);
    let barrier = Barrier::new(N);

    struct Add {
        kernel: CudaKernel,
        word: u64,
        by: u32,
        barrier: *const Barrier,
    }
    // SAFETY: the barrier outlives the scoped threads that hold the pointer.
    unsafe impl Send for Add {}
    impl GraphBody<CudaGpu> for Add {
        fn enqueue<E: Enqueue<Kernel = CudaKernel>>(
            &mut self,
            enq: &mut E,
        ) -> Result<(), GpuError> {
            // SAFETY: see the `Send` impl.
            unsafe { &*self.barrier }.wait();
            // SAFETY: the device word is live.
            unsafe {
                enq.launch(
                    &self.kernel,
                    Launch::linear(1, 1),
                    &Args::new()
                        .device(self.word)
                        .device(self.word)
                        .arg(self.by)
                        .arg(1u32),
                )
            }
        }
    }

    let round = |expect: Ran| {
        let ran = std::thread::scope(|scope| {
            let threads: Vec<_> = (0..N)
                .map(|t| {
                    let (gpu, lane, cache, add, barrier) = (&gpu, &lane, &cache, &add, &barrier);
                    scope.spawn(move || {
                        let mut hold = GraphHold::new();
                        let mut body = Add {
                            kernel: add.clone(),
                            word: block.addr + 4 * t as u64,
                            by: 1,
                            barrier,
                        };
                        let ran = cache.run(gpu, lane, 7, &mut body, &mut hold).unwrap();
                        (ran, hold)
                    })
                })
                .collect();
            threads
                .into_iter()
                .map(|t| t.join().unwrap())
                .collect::<Vec<_>>()
        });
        assert!(ran.iter().all(|(r, _)| *r == expect));
    };
    round(Ran::Recorded);
    assert_eq!(cache.len(), N);
    round(Ran::Replayed);
    assert_eq!(
        cache.len(),
        N,
        "each worker had an instance: none was added"
    );
    let host = pinned.word();
    for t in 0..N {
        let (_, marker) = lane
            .submit_marked(|q| {
                // SAFETY: the pinned block holds the words; the device words are live.
                unsafe { q.copy_to_host(host.bytes().add(4 * t), block.addr + 4 * t as u64, 4) }
            })
            .unwrap();
        assert!(wait_until(Duration::from_secs(10), || lanes
            .queues
            .reached(marker)));
    }
    for t in 0..N {
        // SAFETY: the copy ended.
        let seen = unsafe { ((host.bytes() as *const u32).add(t)).read_volatile() };
        assert_eq!(
            seen, 2,
            "worker {t}: its recording ran once and its replay ran once"
        );
    }
}
