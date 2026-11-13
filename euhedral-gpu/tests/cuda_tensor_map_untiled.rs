//! A kernel given a map that was never encoded (all zeros) must fail fast, not hang: the device faults or the kernel's
//! bounded wait traps, and the launch's retirement reports a failure within a bounded time. A fault poisons the context
//! for the whole process, so this file holds one test and runs as its own process. Skips, passing, when no driver, device
//! or run-time compiler is present or the device is older than compute capability 9.0.
#![cfg(feature = "cuda")]
#![allow(
    clippy::disallowed_types,
    reason = "test fixtures: the test thread locks its records"
)]
#![allow(
    clippy::disallowed_methods,
    reason = "the test thread is not a lattice worker: it polls and sleeps"
)]

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use euhedral_gpu::cuda::{CudaGpu, CudaModule};
use euhedral_gpu::{
    Args, CompileOptions, Gpu, LaneEnqueue, Launch, Module, RetireListener, RetireTicket, Stream,
    StreamOptions, TensorMap,
};
use euhedral_memory::cuda::DeviceMemory;
use euhedral_memory::{Backend, LocationId};

struct Record(Arc<Mutex<Vec<RetireTicket>>>);

impl RetireListener for Record {
    fn retired(self: Box<Self>, ticket: RetireTicket) {
        self.0.lock().unwrap().push(ticket);
    }
}

const SOURCE: &str = r#"
struct alignas(64) TMap { unsigned long long opaque[16]; };

__device__ __forceinline__ unsigned smem_address(const void* p) {
    return static_cast<unsigned>(__cvta_generic_to_shared(p));
}

extern "C" __global__ void tile_copy(unsigned* out, int c0, int c1, const __grid_constant__ TMap map) {
    __shared__ __align__(128) unsigned tile[128];
    __shared__ unsigned long long bar;
    if (threadIdx.x == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_address(&bar)), "r"(1));
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                     ::"r"(smem_address(&bar)), "r"(128u * 4u) : "memory");
        asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
                     ::"r"(smem_address(tile)), "l"(&map), "r"(c0), "r"(c1), "r"(smem_address(&bar)) : "memory");
    }
    // Bounded, so a descriptor that does not arrive fails the launch fast instead of hanging it.
    asm volatile(
        "{\n.reg .pred p;\n.reg .u32 n;\nmov.u32 n, 0;\nWAIT_%=:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
        "@p bra DONE_%=;\n"
        "add.u32 n, n, 1;\n"
        "setp.lt.u32 p, n, 100000;\n"
        "@p bra WAIT_%=;\n"
        "trap;\n"
        "DONE_%=:\n}" ::"r"(smem_address(&bar)), "r"(0u) : "memory");
    out[threadIdx.x] = tile[threadIdx.x];
}
"#;

#[test]
fn a_map_that_was_never_encoded_fails_the_launch_fast() {
    let Ok(gpu) = CudaGpu::open(0) else {
        eprintln!("no CUDA device: skipped");
        return;
    };
    if gpu.arch().0 < 9 {
        eprintln!("compute capability below 9.0: skipped");
        return;
    }
    let Ok(module) = CudaModule::from_source(&gpu, SOURCE, &CompileOptions::default()) else {
        eprintln!("no way to compile the test kernels: skipped");
        return;
    };
    let (_pool, mut s) = gpu.open_lanes(&["a"], StreamOptions::default()).unwrap();
    let mem = DeviceMemory {
        device: gpu.device().clone(),
        location: LocationId(0),
    };
    let out = mem.alloc(128 * 4).unwrap();
    let kernel = module.kernel("tile_copy").unwrap();
    // SAFETY: `TensorMap` is 128 bytes of integers; zeros are the map nobody encoded.
    let untiled: TensorMap = unsafe { std::mem::zeroed() };
    let seen = Arc::new(Mutex::new(Vec::new()));
    let started = Instant::now();
    s[0].submit(|q| {
        // SAFETY: matching arguments; the kernel writes only the live block. The map is the fault under test.
        unsafe {
            q.launch(
                &kernel,
                Launch::linear(128, 128),
                &Args::new()
                    .device(out.addr)
                    .arg(0i32)
                    .arg(0i32)
                    .arg(untiled),
            )
        }
    })
    .unwrap();
    s[0].notify_retired(Box::new(Record(seen.clone()))).unwrap();
    while seen.lock().unwrap().is_empty() {
        s[0].poll_retired();
        assert!(
            started.elapsed() < Duration::from_secs(20),
            "the launch with an untiled map hung"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
    let ticket = seen.lock().unwrap()[0];
    eprintln!(
        "untiled map: retired after {:?} with status {}",
        started.elapsed(),
        ticket.status()
    );
    assert!(!ticket.ok(), "an untiled map must not complete cleanly");
}
