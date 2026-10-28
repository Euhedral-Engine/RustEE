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
    // A host function reports success whatever became of the work before it, and this driver runs them after the fault.
    // The stream's query shows the fault a little after the first host function ran, so the first ticket can still say
    // success when it is handed out; what is guaranteed is that no retirement behind the fault *confirms* as a success.
    // Using the stream observes the fault if the polls did not, and every ticket is then settled.
    let refused = stream.mark();
    assert!(
        matches!(refused, Err(GpuError::Fault { .. } | GpuError::Poisoned(_))),
        "{refused:?}"
    );
    assert!(gpu.poisoned().is_some());
    for ticket in &tickets {
        let failure = stream.confirm_retired(*ticket);
        assert!(
            matches!(
                failure,
                Some(GpuError::Fault { .. } | GpuError::Poisoned(_))
            ),
            "retirement {} (status {}) was armed behind the fault: {failure:?}",
            ticket.seq(),
            ticket.status()
        );
    }
    // Tickets handed out after the fault was visible are already poisoned.
    assert_eq!(
        tickets.last().map(|t| t.status()),
        Some(euhedral_gpu::POISONED_STATUS)
    );
}
