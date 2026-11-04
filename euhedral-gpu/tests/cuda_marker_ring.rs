//! A full marker ring is `Busy`, and an old marker needs no wait. The test times itself against kernels that spin for
//! 300 ms and 1500 ms, so it runs as its own process.
//!
//! Why: in `tests/cuda.rs` it failed in about one run in ten, with `Stream::mark` taking as long as the rest of the
//! spin. The time is inside the driver's `cuEventRecord` (the only driver call that does work on the mark path: the
//! events are made when the stream opens, and the mark makes, frees and loads nothing). Timing the other tests' calls
//! showed what holds it: `cuModuleLoadDataEx`, `cuModuleUnload`, `cuMemFree` and `cuMemFreeHost`, called by other
//! tests of the same process, each ran for exactly the stall (for example `cuModuleLoadDataEx` 249.834 ms against
//! `cuEventRecord` 249.838 ms). Those calls wait for the context's running kernels, here this test's spin, and the
//! `cuEventRecord` on the other stream returned only when they did. That is the driver's behavior, and these are the
//! build-time calls the README keeps off the item path. A process that runs only this
//! test has none of them while the spin runs. Skips, passing, when no driver or device is present.
#![cfg(feature = "cuda")]
#![allow(
    clippy::disallowed_methods,
    reason = "the test thread is not a lattice worker: it polls and sleeps"
)]

mod common;

use std::time::Duration;

use common::cuda_support::{gpu, module, wait_until};
use euhedral_gpu::{Args, Gpu, GpuError, LaneEnqueue, Launch, Module, Stream, StreamOptions};

/// Cycles of the `spin` kernel for about `ms` milliseconds.
fn cycles(ms: u64) -> u64 {
    ms * 2_500_000
}

#[test]
fn a_full_marker_ring_is_busy_and_old_markers_need_no_wait() {
    let Some(gpu) = gpu() else { return };
    let Some(ops) = module(&gpu) else { return };
    let spin = ops.kernel("spin").unwrap();
    let (pool, mut s) = gpu
        .open_lanes(
            &["a", "b"],
            StreamOptions {
                markers: 4,
                ..StreamOptions::default()
            },
        )
        .unwrap();
    let reached_within = |marker, limit: Duration| wait_until(limit, || pool.reached(marker));
    let spin_for = |stream: &mut euhedral_gpu::cuda::CudaStream, ms: u64| {
        stream
            .submit(|q| {
                // SAFETY: `spin` takes one 64-bit argument and touches no memory.
                unsafe { q.launch(&spin, Launch::linear(1, 1), &Args::new().arg(cycles(ms))) }
            })
            .unwrap();
    };
    spin_for(&mut s[0], 300);
    let first = s[0].mark().unwrap();
    let mut last = first;
    for _ in 1..4 {
        last = s[0].mark().unwrap();
    }
    assert_eq!(
        s[0].mark(),
        Err(GpuError::Busy),
        "four markers are outstanding"
    );
    assert!(reached_within(last, Duration::from_secs(10)));
    // A fifth marker reuses the first slot, behind a long spin.
    spin_for(&mut s[0], 1500);
    let fifth = s[0].mark().unwrap();
    assert_eq!(fifth.value, 5);
    // The first marker completed long ago: waiting for it adds no device wait, though its slot now holds the
    // fifth marker, which is behind the spin.
    s[1].await_marker(first).unwrap();
    let other = s[1].mark().unwrap();
    assert!(reached_within(other, Duration::from_secs(10)));
    assert!(!pool.reached(fifth), "stream a is still spinning");
    assert!(reached_within(fifth, Duration::from_secs(30)));
    let _ = s[0].queue().completed();
    drop((pool, s));
}
