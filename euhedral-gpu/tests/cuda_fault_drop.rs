//! Dropping the pool, then a stream whose retirements are still to be delivered behind a fault. The driver delivers
//! the callbacks after the drop; each pushes into the stream's retirement queue, which must still be alive. A sticky
//! fault is for the whole process, so this file holds one test. Skips, passing, with no device.
#![cfg(feature = "cuda")]
#![allow(
    clippy::disallowed_methods,
    reason = "the test thread is not a lattice worker: it waits for the driver's callbacks after the drop"
)]

use std::time::Duration;

use euhedral_gpu::cuda::{CudaGpu, CudaModule, leaked_shared_sides};
use euhedral_gpu::{Args, CompileOptions, Enqueue, Gpu, Launch, Module, Stream, StreamOptions};

const OPS: &str = include_str!("kernels/ops.cu");
const OPS_PTX: &[u8] = include_bytes!("kernels/ops.ptx");

#[test]
fn a_stream_dropped_after_its_pool_is_safe_with_callbacks_still_to_come() {
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
    let (spin, poke) = (
        module.kernel("spin").unwrap(),
        module.kernel("poke").unwrap(),
    );
    let (pool, mut streams) = gpu.open_lanes(&["a"], StreamOptions::default()).unwrap();
    let mut stream = streams.remove(0);
    stream
        .submit(|q| {
            // SAFETY: `spin` touches no memory. `poke` writes through an address that is not mapped: the deliberate
            // fault of this test, after the retirements below are armed.
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
    for _ in 0..4 {
        stream.arm_retirement().unwrap();
    }
    // The device is observed poisoned before the driver delivers any callback (a lattice worker polling right after a
    // fault is noticed): polling reports the four retirements as poisoned, and nothing is armed any more, yet all four
    // callbacks are still to come.
    gpu.poison_for_test("a fault noticed before the callbacks");
    let mut reported = 0;
    while let Some(ticket) = stream.poll_retired() {
        assert_eq!(ticket.status(), euhedral_gpu::POISONED_STATUS);
        reported += 1;
    }
    assert_eq!(reported, 4);
    let leaked_before = leaked_shared_sides();
    // The pool (the only other holder of the stream's shared side) goes first; then the stream, whose wait ends with
    // the fault while the callbacks are still to be delivered.
    drop(pool);
    drop(stream);
    assert_eq!(
        leaked_shared_sides() - leaked_before,
        1,
        "the shared side is kept alive for the callbacks still to come"
    );
    // The driver delivers whatever it still has; a callback that pushed into freed memory would crash here or later.
    std::thread::sleep(Duration::from_millis(500));
}
