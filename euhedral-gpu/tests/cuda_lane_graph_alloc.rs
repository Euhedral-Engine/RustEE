//! A stage that replays a cached graph on its shared lane allocates nothing per run or per quantum once warm: the
//! update of the instance, the replay, the marker, the wait on another lane's marker, the hold of what the work reads, the
//! graph's frames and its boundary. This binary counts every
//! allocation of the process, so it holds the one test that counts and runs alone; it skips, passing, when no driver or
//! device is present.
#![cfg(feature = "cuda")]
#![allow(
    clippy::disallowed_methods,
    clippy::disallowed_types,
    reason = "the test thread is not a lattice worker: it pumps frames and yields while the device runs, and a quantum's \
              held values sit behind a lock the test owns"
)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use euhedral_core::frames::{Frame, FrameError};
use euhedral_core::generics::{FrameSink, LatticeSource};
use euhedral_data_structures::queues::{Consumer, MpscQueue, Multi, Producer, Single};
use euhedral_gpu::cuda::{CudaGpu, CudaKernel, CudaModule, CudaQueue};
use euhedral_gpu::{
    Args, Enqueue, GpuError, GraphBody, GraphHold, Lane, Launch, Module, SharedGraphCache,
    StreamOptions, open_shared_lanes,
};
use euhedral_graph::{
    Completions, DEFAULT_RETRY_BACKSTOP, EdgeKind, GraphRuntime, LaneControl, LaneId, Stage,
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

const OPS: &str = include_str!("kernels/ops.cu");
const OPS_PTX: &[u8] = include_bytes!("kernels/ops.ptx");

fn module(gpu: &Arc<CudaGpu>) -> Option<CudaModule> {
    if gpu.driver_version() >= 13010 {
        CudaModule::from_image(gpu, OPS_PTX).ok()
    } else {
        CudaModule::from_source(gpu, OPS, &Default::default()).ok()
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

thread_local! {
    static ARGS: std::cell::RefCell<Args> = std::cell::RefCell::new(Args::new());
}

struct Counted(Arc<AtomicUsize>);

impl Drop for Counted {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }
}

/// What a quantum is: the device word its stages add to, and what they hold for their work until the hooks.
struct Work {
    word: u64,
    /// One hold per stage: a second run into one hold would allocate its list.
    holds: [Mutex<GraphHold>; 2],
}

/// The sequence: one launch with this unit\'s arguments, and a value kept for it (made only when it is kept).
struct Add {
    kernel: CudaKernel,
    word: u64,
    freed: Arc<AtomicUsize>,
}

impl GraphBody<CudaGpu> for Add {
    fn enqueue<E: Enqueue<Kernel = CudaKernel>>(&mut self, enq: &mut E) -> Result<(), GpuError> {
        // The arguments are rewritten in place in a container this worker keeps: a pass allocates nothing.
        ARGS.with_borrow_mut(|args| {
            args.clear();
            args.push_device(self.word)
                .push_device(self.word)
                .push(1u32)
                .push(1u32);
            // SAFETY: matching arguments over a live device word.
            unsafe { enq.launch(&self.kernel, Launch::linear(1, 1), args) }
        })?;
        enq.keep_with(|| Counted(self.freed.clone()));
        Ok(())
    }
}

/// A stage that waits for the markers it was given, runs a cached graph on its lane inside its marker (a refusal comes
/// before the run), and releases what the work reads in its hook.
struct Replays {
    cache: Arc<SharedGraphCache<CudaGpu, u32>>,
    gpu: Arc<CudaGpu>,
    kernel: CudaKernel,
    key: u32,
    slot: usize,
    freed: Arc<AtomicUsize>,
    hooks: Arc<AtomicUsize>,
}

impl Stage<Work, CudaQueue> for Replays {
    fn run(&self, work: &Work, cx: &StageCx<'_, CudaQueue>) -> Result<Submitted, StageError> {
        let lane = cx.lane_handle().expect("a stage with a lane");
        for marker in cx.waits {
            lane.await_marker(*marker)
                .map_err(|e| StageError::Corrupt(e.to_string()))?;
        }
        let mut body = Add {
            kernel: self.kernel.clone(),
            word: work.word,
            freed: self.freed.clone(),
        };
        let mut hold = work.holds[self.slot]
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
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
        work.holds[self.slot]
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .release();
        self.hooks.fetch_add(1, Ordering::Release);
    }
}

#[test]
fn warm_stages_replay_cached_graphs_and_quanta_run_without_allocating() {
    let Ok(gpu) = CudaGpu::open(0) else {
        eprintln!("no CUDA device: skipped");
        return;
    };
    let Some(module) = module(&gpu) else {
        eprintln!("no kernel module: skipped");
        return;
    };
    let mem = euhedral_memory::cuda::DeviceMemory {
        device: gpu.device().clone(),
        location: euhedral_memory::LocationId(0),
    };
    let buf = euhedral_memory::Backend::alloc(&mem, 4096).unwrap();
    let add = module.kernel("add_u32").unwrap();
    let specs = [
        ("a", StreamOptions::default()),
        ("b", StreamOptions::default()),
    ];
    let lanes = open_shared_lanes(&gpu, &specs).unwrap();
    let completions = Completions::new(lanes.queues.clone());
    let queue = MpscQueue::<Box<dyn Frame>>::bounded(1024);
    let mut frames: Consumer<Box<dyn Frame>, Single> = queue.consumer();
    let mut spec = StageGraphSpec::new();
    let first = spec.stage(StageSpec {
        name: "first",
        ordered: None,
        lanes: vec![LaneId(0)],
    });
    let second = spec.stage(StageSpec {
        name: "second",
        ordered: None,
        lanes: vec![LaneId(1)],
    });
    spec.edge(first, second, EdgeKind::Submission);
    let retired = Arc::new(AtomicUsize::new(0));
    let counted = retired.clone();
    let cache = Arc::new(SharedGraphCache::<CudaGpu, u32>::new());
    let freed = Arc::new(AtomicUsize::new(0));
    let hooks = Arc::new(AtomicUsize::new(0));
    let stage = |key: u32, slot: usize| {
        Arc::new(Replays {
            cache: cache.clone(),
            gpu: gpu.clone(),
            kernel: add.clone(),
            key,
            slot,
            freed: freed.clone(),
            hooks: hooks.clone(),
        }) as Arc<dyn Stage<Work, CudaQueue>>
    };
    let graph = StageGraph::new(
        spec,
        vec![stage(1, 0), stage(2, 1)],
        GraphRuntime {
            stages: Arc::new(Sink(queue.producer())),
            completions: completions.clone(),
            lanes: lanes.pool.clone(),
            retries: Arc::default(),
            retry_backstop: DEFAULT_RETRY_BACKSTOP,
        },
        move |_, _| {
            counted.fetch_add(1, Ordering::Release);
        },
    )
    .unwrap();
    let kill = Arc::new(AtomicBool::new(false));
    let mut started = 0usize;
    let mut quantum = || {
        let work = Work {
            word: buf.addr,
            holds: [Mutex::default(), Mutex::default()],
        };
        let _ = graph.start(work, kill.clone(), None::<euhedral_graph::ExternalEdge>);
        started += 1;
        while retired.load(Ordering::Acquire) < started {
            while let Some(frame) = frames.poll() {
                run_frame(frame);
            }
            let mut frame = None;
            completions.pull(&mut |f| frame = Some(f), &mut |_| false, 1);
            if let Some(frame) = frame {
                run_frame(frame);
            }
        }
    };
    for _ in 0..200 {
        quantum();
    }
    let before = ALLOCATIONS.load(Ordering::Relaxed);
    for _ in 0..2_000 {
        quantum();
    }
    let allocated = ALLOCATIONS.load(Ordering::Relaxed) - before;
    assert_eq!(
        allocated, 0,
        "{allocated} allocations over 2000 quanta of two stages that replay a cached graph, mark, hold what the work \
         reads and release it in their hook"
    );
    assert_eq!(
        hooks.load(Ordering::Acquire),
        2 * 2_200,
        "every stage\'s hook ran once"
    );
    let stats = cache.stats();
    assert_eq!(
        stats.records, 2,
        "one instance per key: the quanta never overlapped"
    );
    assert_eq!(stats.replays, 2 * 2_200 - 2);
    assert_eq!(
        freed.load(Ordering::Relaxed),
        0,
        "the instances still own the recordings\' values"
    );
    drop(graph);
    drop(cache);
    assert_eq!(
        freed.load(Ordering::Relaxed),
        2,
        "the values were dropped with the instances, once, and no hold kept one"
    );
}
