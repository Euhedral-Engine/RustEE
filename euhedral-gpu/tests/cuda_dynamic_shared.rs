//! A kernel opted in to more than the default dynamic shared memory. The test skips, passing, when no driver,
//! device or run-time compiler is present.
#![cfg(feature = "cuda")]
#![allow(
    clippy::disallowed_methods,
    reason = "the test thread is not a lattice worker: it polls and sleeps"
)]

use std::time::{Duration, Instant};

use euhedral_gpu::cuda::{CudaGpu, CudaModule};
use euhedral_gpu::{
    Args, CompileOptions, Gpu, GpuError, KernelOptions, LaneEnqueue, Launch, Module, Sequence,
    Stream, StreamOptions,
};
use euhedral_memory::cuda::{DeviceMemory, PinnedHost};
use euhedral_memory::{Backend, LocationId};

const WORDS: usize = 16 * 1024;
const BYTES: usize = WORDS * 4;
const SHARED: u32 = 64 * 1024;

/// Fills the dynamic shared memory with a function of the index, then reads it back from the other end.
const SOURCE: &str = r#"
extern "C" __global__ void reverse(unsigned* out, unsigned n) {
    extern __shared__ unsigned smem[];
    for (unsigned i = threadIdx.x; i < n; i += blockDim.x) smem[i] = i * 3u + 1u;
    __syncthreads();
    for (unsigned i = threadIdx.x; i < n; i += blockDim.x) out[i] = smem[n - 1u - i];
}
"#;

#[test]
fn a_launch_with_64_kib_of_dynamic_shared_memory_runs_after_opt_in() {
    let gpu = match CudaGpu::open(0) {
        Ok(gpu) => gpu,
        Err(e) => {
            eprintln!("no CUDA device: skipped ({e})");
            return;
        }
    };
    let module = match CudaModule::from_source(&gpu, SOURCE, &CompileOptions::default()) {
        Ok(module) => module,
        Err(e) => {
            eprintln!("no way to compile the test kernels: skipped ({e})");
            return;
        }
    };
    let (pool, mut s) = gpu.open_lanes(&["a"], StreamOptions::default()).unwrap();
    let mem = DeviceMemory {
        device: gpu.device().clone(),
        location: LocationId(0),
    };
    let pinned = PinnedHost::new(gpu.device().clone(), LocationId(1), 0);
    let dev = mem.alloc(BYTES as u64).unwrap();
    let host = pinned.alloc(BYTES as u64).unwrap();
    let launch = Launch {
        shared_bytes: SHARED,
        ..Launch::linear(256, 256)
    };
    let args = Args::new().device(dev.addr).arg(WORDS as u32);

    // Without the opt-in, the launch is refused before the driver sees it, naming the kernel and the numbers.
    let plain = module.kernel("reverse").unwrap();
    let default_limit = plain
        .max_dynamic_shared_bytes()
        .expect("the driver reports its default");
    assert!(
        default_limit < SHARED,
        "the default limit is {default_limit}"
    );
    let refused = s[0].submit(|e| {
        // SAFETY: matching arguments over a live block; nothing reaches the device.
        unsafe { e.launch(&plain, launch, &args) }
    });
    let Err(GpuError::Invalid(text)) = refused else {
        panic!("expected a refusal, got {refused:?}")
    };
    assert!(
        text.contains("reverse") && text.contains(&SHARED.to_string()),
        "{text}"
    );
    assert!(text.contains(&default_limit.to_string()), "{text}");

    // The same refusal on the eager and the probing paths, before anything is recorded or issued.
    let eager = gpu.record(&mut [&mut s[0]], |q| {
        q.submit(0, |e| {
            // SAFETY: as above.
            unsafe { e.launch(&plain, launch, &args) }
        })
    });
    assert!(
        matches!(eager, Err(GpuError::Invalid(_))),
        "{:?}",
        eager.err()
    );
    let probed = gpu.fingerprint(1, |q| {
        q.submit(0, |e| {
            // SAFETY: as above.
            unsafe { e.launch(&plain, launch, &args) }
        })
    });
    assert!(matches!(probed, Err(GpuError::Invalid(_))), "{probed:?}");

    // With it, the launch runs and the data comes back reversed.
    let kernel = module
        .kernel_with(
            "reverse",
            KernelOptions {
                max_dynamic_shared_bytes: SHARED,
            },
        )
        .unwrap();
    assert_eq!(kernel.max_dynamic_shared_bytes(), Some(SHARED));
    // A later lookup with less does not lower it, for the handle that was there before either.
    let _smaller = module
        .kernel_with(
            "reverse",
            KernelOptions {
                max_dynamic_shared_bytes: 1024,
            },
        )
        .unwrap();
    assert_eq!(plain.max_dynamic_shared_bytes(), Some(SHARED));
    s[0].submit(|e| {
        // SAFETY: matching arguments over live blocks that the test keeps until the work ends.
        unsafe {
            e.launch(&kernel, launch, &args)?;
            e.copy_to_host(host.addr as *mut u8, dev.addr, BYTES)
        }
    })
    .unwrap();
    let m = s[0].mark().unwrap();
    let start = Instant::now();
    while !pool.reached(m) {
        assert!(start.elapsed() < Duration::from_secs(10), "the work ends");
        std::thread::sleep(Duration::from_millis(1));
    }
    // SAFETY: a pinned block of WORDS words that the work has finished writing.
    let out = unsafe { std::slice::from_raw_parts(host.addr as *const u32, WORDS) };
    let expected: Vec<u32> = (0..WORDS as u32)
        .map(|i| (WORDS as u32 - 1 - i) * 3 + 1)
        .collect();
    assert_eq!(out, expected.as_slice());
}
