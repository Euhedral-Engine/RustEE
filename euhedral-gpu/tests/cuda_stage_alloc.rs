//! A stage that submits to its lane through its context allocates nothing per submit or per quantum once warm: the
//! launch, the marker, the wait on another lane's marker, the graph's frames and its boundary. This binary counts every
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
    Args, GpuError, Lane, LaneEnqueue, Launch, Module, StreamOptions, open_shared_lanes,
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

/// What a quantum is: the device word its stages add to, and what each stage holds for its work until its hook.
struct Work {
    word: u64,
    held: [Mutex<Option<Arc<AtomicUsize>>>; 2],
}

/// A stage that waits for the markers it was given, launches on its lane with its marker (a refusal comes before the
/// work), holds a value for the launch, and releases it in its retirement hook.
struct Adds {
    kernel: CudaKernel,
    /// The value a stage holds (a clone of this) and the count of hooks that released one.
    value: Arc<AtomicUsize>,
    released: Arc<AtomicUsize>,
    slot: usize,
}

impl Stage<Work, CudaQueue> for Adds {
    fn run(&self, work: &Work, cx: &StageCx<'_, CudaQueue>) -> Result<Submitted, StageError> {
        let lane = cx.lane_handle().expect("a stage with a lane");
        for marker in cx.waits {
            lane.await_marker(*marker)
                .map_err(|e| StageError::Corrupt(e.to_string()))?;
        }
        // The arguments are rewritten in place in a container this worker keeps: a launch allocates nothing.
        let made = ARGS.with_borrow_mut(|args| {
            args.clear();
            args.push_device(work.word)
                .push_device(work.word)
                .push(1u32)
                .push(1u32);
            lane.submit_marked(|q| {
                // SAFETY: matching arguments over a live device word.
                unsafe { q.launch(&self.kernel, Launch::linear(1, 1), args) }
            })
        });
        match made {
            Ok(((), marker)) => {
                *work.held[self.slot]
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner) = Some(self.value.clone());
                Ok(Submitted::marker(marker))
            }
            Err(GpuError::Busy) => Err(StageError::Retry),
            Err(e) => Err(StageError::Corrupt(e.to_string())),
        }
    }

    fn retired(&self, work: &Work, committed: bool) {
        assert!(committed);
        let held = work.held[self.slot]
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take();
        if held.is_some() {
            self.released.fetch_add(1, Ordering::Release);
        }
    }
}

#[test]
fn warm_stages_submit_and_quanta_run_without_allocating() {
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
    let value = Arc::new(AtomicUsize::new(0));
    let released = Arc::new(AtomicUsize::new(0));
    let graph = StageGraph::new(
        spec,
        vec![
            Arc::new(Adds {
                kernel: add.clone(),
                value: value.clone(),
                released: released.clone(),
                slot: 0,
            }) as Arc<dyn Stage<Work, CudaQueue>>,
            Arc::new(Adds {
                kernel: add,
                value: value.clone(),
                released: released.clone(),
                slot: 1,
            }),
        ],
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
            held: [Mutex::new(None), Mutex::new(None)],
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
        "{allocated} allocations over 2000 quanta of two stages that launch, wait on a marker, mark, hold a value and \
         release it in their hook"
    );
    assert_eq!(
        released.load(Ordering::Acquire),
        2 * 2_200,
        "every value a stage held was released by its hook, once"
    );
    assert_eq!(
        Arc::strong_count(&value),
        3,
        "this handle and the two stages': no hold outlives its quantum"
    );
}
