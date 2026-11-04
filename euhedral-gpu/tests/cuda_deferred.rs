//! A device's blocking frees and unloads wait in its `DeferredCalls` while a kernel runs, and are made once its lanes have
//! nothing in flight.
//!
//! The driver's `cuMemFree`, `cuMemFreeHost` and `cuModuleLoadDataEx` hold up a `cuEventRecord` on every other stream until the
//! running kernels end (`tests/cuda_marker_ring.rs` says what was measured); `cuModuleUnload` makes its caller wait. With a
//! deferral attached to the device, a thread that drops memory or a module while a kernel spins returns at once, another
//! stream's marker is made in microseconds, and the calls are made when `LanePool::idle` says no lane has marked work in
//! flight. The test times itself against a 600 ms spin, so it runs as its own process. Skips, passing, when no driver or
//! device is present.
#![cfg(feature = "cuda")]
#![allow(
    clippy::disallowed_methods,
    reason = "the test thread is not a lattice worker: it polls and sleeps"
)]

mod common;

use std::time::{Duration, Instant};

use common::cuda_support::{device_word, gpu, module, wait_until};
use euhedral_gpu::{Args, Gpu, LaneEnqueue, Launch, Module, Stream, StreamOptions};
use euhedral_memory::cuda::{DeviceMemory, PinnedHost};
use euhedral_memory::{Backend, Bounds, DeferredCalls, LocationId, Place};

/// The tests share the device: the second one's frees would hold up the first one's marker.
#[allow(
    clippy::disallowed_types,
    reason = "a test fixture: the tests take turns on the device"
)]
static ONE_AT_A_TIME: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[allow(
    clippy::disallowed_types,
    reason = "a test fixture: the tests take turns on the device"
)]
fn serial() -> std::sync::MutexGuard<'static, ()> {
    ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Cycles of the `spin` kernel for about `ms` milliseconds.
fn cycles(ms: u64) -> u64 {
    ms * 2_500_000
}

#[test]
fn deferred_frees_and_unloads_leave_a_marker_alone_and_are_made_when_the_lanes_are_idle() {
    let _serial = serial();
    let Some(gpu) = gpu() else { return };
    let Some(ops) = module(&gpu) else { return };
    let Some(extra) = module(&gpu) else { return };
    let spin = ops.kernel("spin").unwrap();
    let calls = DeferredCalls::new(Bounds::default());
    gpu.device()
        .attach_deferral(calls.handle())
        .expect("the first deferral");
    let (mem, block) = device_word(&gpu);
    assert!(!mem.free_blocks(), "a free no longer waits for the device");
    let pinned = PinnedHost::new(gpu.device().clone(), LocationId(1), 0);
    let host = pinned.alloc(1 << 20).unwrap();
    let (pool, mut s) = gpu
        .open_lanes(&["a", "b"], StreamOptions::default())
        .unwrap();
    calls.attach_pool(&pool).expect("the pool");
    s[0].submit(|q| {
        // SAFETY: `spin` takes one 64-bit argument and touches no memory.
        unsafe { q.launch(&spin, Launch::linear(1, 1), &Args::new().arg(cycles(600))) }
    })
    .unwrap();
    let spinning = Stream::mark(&mut s[0]).unwrap();
    std::thread::sleep(Duration::from_millis(50));
    let started = Instant::now();
    // Another thread drops what blocks: it returns at once, the calls wait.
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
    assert!(!pool.idle(), "and a lane has marked work in flight");
    assert_eq!(
        calls.pending(),
        3,
        "the device free, the pinned free, the unload"
    );
    assert!(calls.pending_bytes(Place::Device) >= 256);
    assert!(calls.pending_bytes(Place::Host) >= 1 << 20);
    // The kernel ends: the lanes are idle, the calls are made.
    assert!(wait_until(Duration::from_secs(30), || pool.idle()));
    assert_eq!(calls.drain(), 3);
    assert_eq!(calls.pending(), 0);
    drop((pool, s));
}

#[test]
fn an_allocation_that_runs_out_of_memory_makes_the_deferred_frees_and_tries_again() {
    let _serial = serial();
    let Some(gpu) = gpu() else { return };
    let device = gpu.device().clone();
    let (free, _) = device.mem_info().unwrap();
    // Two blocks of 60 % of the free memory cannot both exist.
    let bytes = free / 10 * 6;
    let mem = DeviceMemory {
        device: device.clone(),
        location: LocationId(0),
    };
    let first = mem.alloc(bytes).unwrap();
    assert!(
        mem.alloc(bytes).is_err(),
        "without a deferral the memory is not there"
    );
    let calls = DeferredCalls::new(Bounds::default());
    device
        .attach_deferral(calls.handle())
        .expect("the first deferral");
    // SAFETY: the block came from `mem` and nothing uses it.
    unsafe { mem.free(first) };
    assert_eq!(calls.pending(), 1);
    let second = mem
        .alloc(bytes)
        .expect("the deferred free was made and the allocation retried");
    assert_eq!(calls.pending(), 0);
    // A second failure is the caller's: nothing is waiting this time.
    assert!(mem.alloc(bytes).is_err());
    // SAFETY: as above.
    unsafe { mem.free(second) };
    calls.drain();
}

#[test]
fn a_device_does_not_keep_its_deferral_alive_and_the_owner_letting_go_makes_the_calls() {
    let _serial = serial();
    let Some(gpu) = gpu() else { return };
    let device = gpu.device().clone();
    let calls = DeferredCalls::new(Bounds::default());
    device
        .attach_deferral(calls.handle())
        .expect("the first deferral");
    let weak = std::sync::Arc::downgrade(&calls);
    // A queued call owns the device, as the backends' calls do.
    let (d2, made) = (
        device.clone(),
        std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
    );
    let m = made.clone();
    device.defer_or_run(Place::Device, 0, move |_| {
        let _owns = &d2;
        m.store(true, std::sync::atomic::Ordering::Release);
    });
    assert_eq!(calls.pending(), 1);
    assert!(device.deferral().is_some());
    drop(calls);
    assert!(
        made.load(std::sync::atomic::Ordering::Acquire),
        "the source's drop made the call"
    );
    assert!(weak.upgrade().is_none(), "and the source is gone");
    assert!(
        device.deferral().is_none(),
        "so the device defers nothing more"
    );
    // A drop now is made where it is dropped.
    let ran = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let r = ran.clone();
    device.defer_or_run(Place::Device, 0, move |_| {
        r.store(true, std::sync::atomic::Ordering::Release);
    });
    assert!(ran.load(std::sync::atomic::Ordering::Acquire));
}
