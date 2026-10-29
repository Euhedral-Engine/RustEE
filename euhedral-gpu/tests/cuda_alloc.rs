//! The retirement path allocates nothing in steady state. This binary counts every allocation of the process, so it
//! holds one test and runs alone: the driver's callbacks (on threads the driver owns), the push onto the stream's
//! retirement queue, and the owner's drain are all inside the counted window. Skips, passing, with no device.
#![cfg(feature = "cuda")]
#![allow(
    clippy::disallowed_methods,
    reason = "the test thread is not a lattice worker: it polls for the callbacks"
)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use euhedral_gpu::cuda::{CudaGpu, CudaStream};
use euhedral_gpu::{Gpu, Stream, StreamOptions, UNPOLLED_RETIREMENTS};
use euhedral_graph::DeviceQueue;

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

/// Arms `batch` retirements, waits (from the device, with a marker after them) until the driver has run every
/// callback, then drains the tickets. Draining only after the callbacks finished keeps the owner's side of the queue
/// from running beside the callbacks' side.
fn round(stream: &mut CudaStream, queue: &Arc<dyn DeviceQueue>, batch: usize) {
    for _ in 0..batch {
        stream.arm_retirement().unwrap();
    }
    let marker = stream.mark().unwrap();
    let deadline = Instant::now() + Duration::from_secs(10);
    while queue.completed() < marker.value {
        assert!(Instant::now() < deadline, "the callbacks run");
    }
    let mut arrived = 0;
    while let Some(ticket) = stream.poll_retired() {
        assert!(ticket.ok());
        arrived += 1;
    }
    assert_eq!(arrived, batch);
}

#[test]
fn the_retirement_path_allocates_nothing_in_steady_state() {
    let Ok(gpu) = CudaGpu::open(0) else {
        eprintln!("no CUDA device: skipped");
        return;
    };
    // The counter sees allocations at all.
    let probe = ALLOCATIONS.load(Ordering::Relaxed);
    drop(std::hint::black_box(Box::new(7u64)));
    assert!(
        ALLOCATIONS.load(Ordering::Relaxed) > probe,
        "the counting allocator is live"
    );
    let (_pool, mut streams) = gpu.open_lanes(&["a"], StreamOptions::default()).unwrap();
    let stream = &mut streams[0];
    let queue = stream.queue();
    // The queue's chunk pool is seeded when the stream is made, so the callbacks need no warm-up. Batches of 8
    // cross a 63-status chunk now and then; a batch of 100 leaves more than a chunk unpolled at once. Both stay under
    // the bound the callback is documented to allocate nothing below.
    const { assert!(100 > 64 && 100 < UNPOLLED_RETIREMENTS) };
    // The owner's own list of armed retirements grows once, to the largest batch (that is the owner's side, and not
    // the callback's).
    round(stream, &queue, 100);
    let before = ALLOCATIONS.load(Ordering::Relaxed);
    for _ in 0..200 {
        round(stream, &queue, 8);
    }
    for _ in 0..40 {
        round(stream, &queue, 100);
    }
    let after = ALLOCATIONS.load(Ordering::Relaxed);
    assert_eq!(
        after - before,
        0,
        "5600 retirements armed, queued by the driver's callbacks and drained: no allocation"
    );
}
