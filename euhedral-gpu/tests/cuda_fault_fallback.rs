//! The `cuLaunchHostFunc` fallback of retirement (a driver without `cuStreamAddCallback`), forced by a test switch,
//! through a deliberate fault. A host function enqueued after a fault is skipped by the driver, so a retirement may
//! never be called back: polling must ask the stream (without waiting) and report the rest as poisoned. A sticky fault
//! is for the whole process, so this file holds one test. Skips, passing, with no device.
#![cfg(feature = "cuda")]
#![allow(
    clippy::disallowed_methods,
    reason = "the test thread is not a lattice worker: it polls until the retirements are reported"
)]

use std::time::{Duration, Instant};

use euhedral_gpu::cuda::{CudaGpu, CudaModule};
use euhedral_gpu::{
    Args, CompileOptions, Enqueue, Gpu, GpuError, Launch, Module, Stream, StreamOptions,
};

const OPS: &str = include_str!("kernels/ops.cu");
const OPS_PTX: &[u8] = include_bytes!("kernels/ops.ptx");

#[test]
fn a_fault_with_host_function_retirements_still_reports_every_retirement() {
    let Ok(gpu) = CudaGpu::open(0) else {
        eprintln!("no CUDA device: skipped");
        return;
    };
    let module = if gpu.driver_version() >= 13010 {
        CudaModule::from_image(&gpu, OPS_PTX)
    } else {
        CudaModule::from_source(&gpu, OPS, &CompileOptions::default())
    };
    let Ok(module) = module else {
        eprintln!("no way to load the test kernels: skipped");
        return;
    };
    gpu.force_host_func_fallback(true);
    let (spin, poke) = (
        module.kernel("spin").unwrap(),
        module.kernel("poke").unwrap(),
    );
    let (_pool, mut streams) = gpu.open_lanes(&["a"], StreamOptions::default()).unwrap();
    let mut stream = streams.remove(0);
    stream
        .submit(|q| {
            // SAFETY: `spin` touches no memory. `poke` writes through an address that is not mapped: the deliberate
            // fault of this test.
            unsafe {
                q.launch(
                    &spin,
                    Launch::linear(1, 1),
                    &Args::new().arg(125_000_000u64),
                )?;
                q.launch(&poke, Launch::linear(1, 1), &Args::new().device(0x10))
            }
        })
        .unwrap();
    // Armed behind the fault, while it has not happened yet.
    for _ in 0..3 {
        stream.arm_retirement().unwrap();
    }
    assert_eq!(stream.outstanding_tickets(), 3);

    let deadline = Instant::now() + Duration::from_secs(10);
    let mut tickets = Vec::new();
    while tickets.len() < 3 {
        assert!(
            Instant::now() < deadline,
            "every retirement is reported, even if the driver skipped its host function: got {}",
            tickets.len()
        );
        if let Some(ticket) = stream.poll_retired() {
            tickets.push(ticket);
        }
    }
    assert_eq!(
        tickets.iter().map(|t| t.seq()).collect::<Vec<_>>(),
        [0, 1, 2]
    );
    assert_eq!(stream.outstanding_tickets(), 0);
    // Whichever way the driver treated the host functions (this one runs them with status 0, an older one may skip
    // them), every retirement was reported and the fault is real: the stream refuses new work with it.
    let refused = stream.mark();
    assert!(
        matches!(refused, Err(GpuError::Fault { .. } | GpuError::Poisoned(_))),
        "{refused:?}; tickets {:?}",
        tickets.iter().map(|t| t.status()).collect::<Vec<_>>()
    );
    assert!(gpu.poisoned().is_some());
}
