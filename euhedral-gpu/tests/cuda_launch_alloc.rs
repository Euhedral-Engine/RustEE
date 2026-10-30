//! A repeated launch, a graph update and a replay allocate nothing once warm, and a timed span reads its duration without waiting.
//! This binary counts every allocation of the process, so it holds the one test that counts and runs alone; it skips, passing,
//! when no driver or device is present.
#![cfg(feature = "cuda")]
#![allow(
    clippy::disallowed_methods,
    reason = "the test thread is not a lattice worker: it polls and sleeps"
)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use euhedral_gpu::cuda::{CudaGpu, CudaModule};
use euhedral_gpu::{
    Args, Enqueue, Gpu, GpuError, Graph, Launch, Module, Sequence, Stream, StreamOptions,
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

fn module(gpu: &std::sync::Arc<CudaGpu>) -> Option<CudaModule> {
    if gpu.driver_version() >= 13010 {
        CudaModule::from_image(gpu, OPS_PTX).ok()
    } else {
        CudaModule::from_source(gpu, OPS, &Default::default()).ok()
    }
}

fn allocations() -> usize {
    ALLOCATIONS.load(Ordering::Relaxed)
}

#[test]
fn warm_launches_updates_and_replays_allocate_nothing_and_a_span_reads_without_waiting() {
    let Ok(gpu) = CudaGpu::open(0) else {
        eprintln!("no CUDA device: skipped");
        return;
    };
    let Some(module) = module(&gpu) else {
        eprintln!("no kernel module: skipped");
        return;
    };
    let (pool, mut streams) = gpu
        .open_lanes(
            &["a"],
            StreamOptions {
                spans: 4,
                ..StreamOptions::default()
            },
        )
        .unwrap();
    let mut stream = streams.remove(0);
    let add = module.kernel("add_u32").unwrap();
    let spin = module.kernel("spin").unwrap();
    let launch = Launch::linear(256, 128);
    let mem = euhedral_memory::cuda::DeviceMemory {
        device: gpu.device().clone(),
        location: euhedral_memory::LocationId(0),
    };
    let buf = euhedral_memory::Backend::alloc(&mem, 4096).unwrap();

    // A reused `Args`: rewritten in place for each launch.
    let mut args = Args::new();
    let mut direct =
        |stream: &mut euhedral_gpu::cuda::CudaStream, round: u32| -> Result<(), GpuError> {
            args.clear();
            args.push_device(buf.addr)
                .push_device(buf.addr)
                .push(round)
                .push(256u32);
            // SAFETY: matching arguments over a live block of 4 KiB, waited for below.
            stream.submit(|q| unsafe { q.launch(&add, launch, &args) })
        };
    for round in 0..8 {
        direct(&mut stream, round).unwrap();
    }
    let before = allocations();
    for round in 0..200 {
        direct(&mut stream, round).unwrap();
    }
    let direct_allocs = allocations() - before;

    // A recorded graph of three launches, updated and replayed with the same reused arguments.
    let mut a = [Args::new(), Args::new(), Args::new()];
    let fill = |a: &mut [Args; 3], round: u32| {
        for (i, args) in a.iter_mut().enumerate() {
            args.clear();
            args.push_device(buf.addr)
                .push_device(buf.addr)
                .push(round + i as u32)
                .push(256u32);
        }
    };
    fill(&mut a, 0);
    let body =
        |q: &mut euhedral_gpu::cuda::CudaSequence<'_>, a: &[Args; 3]| -> Result<(), GpuError> {
            q.submit(0, |e| {
                for args in a {
                    // SAFETY: as above.
                    unsafe { e.launch(&add, launch, args)? };
                }
                Ok(())
            })
        };
    let mut graph = gpu.record(&mut [&mut stream], |q| body(q, &a)).unwrap();
    for round in 1..8 {
        fill(&mut a, round);
        gpu.update(&mut graph, |q| body(q, &a)).unwrap();
        graph.replay(&mut stream).unwrap();
    }
    let before = allocations();
    for round in 8..208 {
        fill(&mut a, round);
        gpu.update(&mut graph, |q| body(q, &a)).unwrap();
        graph.replay(&mut stream).unwrap();
    }
    let graph_allocs = allocations() - before;

    // The same through a lane's graph cache: a warm hit (update and replay) allocates nothing either.
    struct Triple<'a> {
        add: &'a euhedral_gpu::cuda::CudaKernel,
        args: [Args; 3],
    }
    impl euhedral_gpu::GraphBody<CudaGpu> for Triple<'_> {
        fn enqueue<E: Enqueue<Kernel = euhedral_gpu::cuda::CudaKernel>>(
            &mut self,
            enq: &mut E,
        ) -> Result<(), GpuError> {
            for args in &self.args {
                // SAFETY: matching arguments over a live block of 4 KiB, waited for below.
                unsafe { enq.launch(self.add, Launch::linear(256, 128), args)? };
            }
            Ok(())
        }
    }
    let mut cache = euhedral_gpu::GraphCache::<CudaGpu, u32>::new();
    let mut triple = Triple {
        add: &add,
        args: a.clone(),
    };
    assert_eq!(
        cache.warm(&gpu, &mut stream, 7, &mut triple).unwrap(),
        euhedral_gpu::Ran::Recorded
    );
    for round in 0..8 {
        for (i, args) in triple.args.iter_mut().enumerate() {
            args.set(2, round + i as u32).unwrap();
        }
        assert_eq!(
            cache.run(&gpu, &mut stream, 7, &mut triple).unwrap(),
            euhedral_gpu::Ran::Replayed
        );
    }
    let before = allocations();
    for round in 8..208 {
        for (i, args) in triple.args.iter_mut().enumerate() {
            args.set(2, round + i as u32).unwrap();
        }
        cache.run(&gpu, &mut stream, 7, &mut triple).unwrap();
    }
    let cache_allocs = allocations() - before;

    let marker = stream.mark().unwrap();
    let start = Instant::now();
    while !pool.reached(marker) {
        assert!(start.elapsed() < Duration::from_secs(20), "the work ends");
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(
        direct_allocs, 0,
        "200 direct launches of a reused Args allocated"
    );
    assert_eq!(graph_allocs, 0, "200 updates and replays allocated");
    assert_eq!(
        cache_allocs, 0,
        "200 warm hits of the graph cache allocated"
    );

    // A span around a long kernel: not ready at once, then its duration.
    let span = stream.span_begin().unwrap();
    // SAFETY: the spin kernel takes its cycle count only.
    stream
        .submit(|q| unsafe {
            q.launch(
                &spin,
                Launch::linear(1, 1),
                &Args::new().arg(2_000_000_000u64),
            )
        })
        .unwrap();
    stream.span_end(span).unwrap();
    let polled_at = Instant::now();
    assert_eq!(
        stream.span_poll(span).unwrap(),
        None,
        "the end is not reached: nothing is waited for"
    );
    assert!(
        polled_at.elapsed() < Duration::from_millis(50),
        "a poll does not wait for the kernel"
    );
    let duration = loop {
        if let Some(d) = stream.span_poll(span).unwrap() {
            break d;
        }
        assert!(
            polled_at.elapsed() < Duration::from_secs(30),
            "the span ends"
        );
        std::thread::sleep(Duration::from_millis(2));
    };
    assert!(
        duration > Duration::from_millis(100),
        "{duration:?} for 2e9 cycles"
    );
    assert!(
        stream.span_poll(span).is_err(),
        "a read span is freed: its handle is stale"
    );

    // ... and the ring is bounded: four open spans fill it, a fifth is busy, a cancel frees one.
    let open: Vec<_> = (0..4).map(|_| stream.span_begin().unwrap()).collect();
    assert_eq!(stream.span_begin(), Err(GpuError::Busy));
    stream.span_cancel(open[0]).unwrap();
    assert!(stream.span_begin().is_ok());
}
