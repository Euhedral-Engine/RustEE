//! Blocking frees and unloads wait in the deferred list while a kernel runs, and run at a quiet moment.
//!
//! The driver's `cuMemFree`, `cuMemFreeHost` and `cuModuleUnload` (like `cuModuleLoadDataEx`) hold up a `cuEventRecord` on
//! every other stream until the running kernels end (`tests/cuda_marker_ring.rs` says what was measured). With deferral
//! enabled, the backends queue those calls on `euhedral_memory::deferred()`: a thread that drops memory or a module while a
//! kernel spins returns at once, a marker on another stream is made in microseconds, and the queue runs when the owner says
//! nothing is in flight. The test times itself against a 600 ms spin, so it runs as its own process (the deferred list is
//! process-wide). Skips, passing, when no driver or device is present.
#![cfg(feature = "cuda")]
#![allow(
    clippy::disallowed_methods,
    reason = "the test thread is not a lattice worker: it polls and sleeps"
)]

mod common;

use std::time::{Duration, Instant};

use common::cuda_support::{device_word, gpu, module, wait_until};
use euhedral_gpu::{Args, Enqueue, Gpu, Launch, Module, Stream, StreamOptions};
use euhedral_memory::cuda::PinnedHost;
use euhedral_memory::{Backend, LocationId};

/// Cycles of the `spin` kernel for about `ms` milliseconds.
fn cycles(ms: u64) -> u64 {
    ms * 2_500_000
}

#[test]
fn deferred_frees_and_unloads_do_not_hold_up_a_marker_and_run_when_asked() {
    let Some(gpu) = gpu() else { return };
    let Some(ops) = module(&gpu) else { return };
    let Some(extra) = module(&gpu) else { return };
    let spin = ops.kernel("spin").unwrap();
    let list = euhedral_memory::deferred();
    list.enable(true);
    let (mem, block) = device_word(&gpu);
    let pinned = PinnedHost::new(gpu.device().clone(), LocationId(1), 0);
    let host = pinned.alloc(1 << 20).unwrap();
    let (pool, mut s) = gpu
        .open_lanes(&["a", "b"], StreamOptions::default())
        .unwrap();
    s[0].submit(|q| {
        // SAFETY: `spin` takes one 64-bit argument and touches no memory.
        unsafe { q.launch(&spin, Launch::linear(1, 1), &Args::new().arg(cycles(600))) }
    })
    .unwrap();
    let spinning = Stream::mark(&mut s[0]).unwrap();
    std::thread::sleep(Duration::from_millis(50));
    let started = Instant::now();
    // Another thread drops what blocks: it returns at once, the calls wait in the list.
    std::thread::scope(|scope| {
        scope.spawn(|| {
            // SAFETY: the block came from `mem` and nothing uses it.
            unsafe { mem.free(block) };
            // SAFETY: the block came from `pinned` and nothing uses it.
            unsafe { pinned.free(host) };
            pinned.trim(true);
            drop(extra);
        });
    });
    let queued = started.elapsed();
    let mark_started = Instant::now();
    let other = Stream::mark(&mut s[1]).unwrap();
    let marked = mark_started.elapsed();
    assert!(
        queued < Duration::from_millis(100),
        "dropping returned in {queued:?}: the calls were not deferred"
    );
    assert!(
        marked < Duration::from_millis(100),
        "a marker took {marked:?} behind a deferred free"
    );
    assert!(wait_until(Duration::from_secs(5), || pool.reached(other)));
    assert!(!pool.reached(spinning), "stream a is still spinning");
    assert_eq!(
        list.pending(),
        3,
        "the device free, the pinned free and the unload"
    );
    // The kernel ends; the owner is at a quiet moment.
    assert!(wait_until(Duration::from_secs(30), || pool.reached(spinning)));
    assert_eq!(list.run_all(), 3);
    assert_eq!(list.pending(), 0);
    list.enable(false);
    drop((pool, s));
}
