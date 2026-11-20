//! Dropping the pool, then a stream whose retirements are still to be delivered behind a fault. The driver delivers
//! the callbacks after the drop; each pushes into the stream's retirement queue, which must still be alive. A sticky
//! fault is for the whole process, so this file holds one test. Skips, passing, with no device.
//!
//! The stream under test starts with a host callback that the test holds, and the four retirements and a last callback
//! follow it in the stream; a callback blocks the later work of its stream until it returns. The fault is made on
//! another lane, so the callbacks are still to come at the drop whatever the speed of any thread, and the test lets
//! the hold go only after the drop.
#![cfg(feature = "cuda")]
#![allow(
    clippy::disallowed_methods,
    reason = "the test thread is not a lattice worker: it waits for the driver's callbacks after the drop"
)]

use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use euhedral_gpu::cuda::fns::{GpuFns, StreamCallback};
use euhedral_gpu::cuda::{CudaGpu, CudaModule, leaked_shared_sides};
use euhedral_gpu::{Args, CompileOptions, Gpu, LaneEnqueue, Launch, Module, Stream, StreamOptions};

/// The longest any wait of this test may take: a fail-fast guard, never a margin.
const GUARD: Duration = Duration::from_secs(60);

/// Set by the holding callback once it runs: the driver's callback thread is inside it.
static HOLDING: AtomicBool = AtomicBool::new(false);
/// Set by the test after the drop: lets the holding callback return.
static RELEASED: AtomicBool = AtomicBool::new(false);
/// Set by the callback queued behind the holding one.
static AFTER: AtomicBool = AtomicBool::new(false);

/// Waits until `flag` is set, or the guard runs out (the test then fails on its own assertions).
fn wait_for(flag: &AtomicBool) -> bool {
    let start = Instant::now();
    while !flag.load(Ordering::Acquire) {
        if start.elapsed() > GUARD {
            return false;
        }
        std::thread::sleep(Duration::from_micros(100));
    }
    true
}

/// Lets the holding callback return when it goes out of scope, a panic included.
struct Release;

impl Drop for Release {
    fn drop(&mut self) {
        RELEASED.store(true, Ordering::Release);
    }
}

unsafe extern "system" fn hold(_: *mut c_void, _: i32, _: *mut c_void) {
    HOLDING.store(true, Ordering::Release);
    wait_for(&RELEASED);
}

unsafe extern "system" fn after(_: *mut c_void, _: i32, _: *mut c_void) {
    AFTER.store(true, Ordering::Release);
}

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
    let poke = module.kernel("poke").unwrap();
    let fns = GpuFns::load(&gpu.device().driver).unwrap();
    let (pool, mut streams) = gpu
        .open_lanes(&["a", "b"], StreamOptions::default())
        .unwrap();
    let mut faulter = streams.remove(1);
    let mut stream = streams.remove(0);
    // Whatever happens next, the callback thread goes free when the test ends.
    let _release = Release;
    // The head of the stream under test: a callback that holds the stream, so everything queued behind it in the
    // stream (the four retirements, then `after`) is delivered only after the test lets go.
    {
        let _current = gpu.device().enter().unwrap();
        let raw = stream.raw_stream_for_tests();
        let callback: StreamCallback = hold;
        // SAFETY: the context is current; the stream is alive; the callbacks are plain functions.
        unsafe {
            assert_eq!(
                (fns.stream_add_callback)(raw, callback, std::ptr::null_mut(), 0),
                0
            );
        }
    }
    assert!(
        wait_for(&HOLDING),
        "the callback thread never took the hold"
    );
    for _ in 0..4 {
        stream.arm_retirement().unwrap();
    }
    {
        let _current = gpu.device().enter().unwrap();
        // SAFETY: as above.
        unsafe {
            assert_eq!(
                (fns.stream_add_callback)(
                    stream.raw_stream_for_tests(),
                    after,
                    std::ptr::null_mut(),
                    0
                ),
                0
            );
        }
    }
    // The fault comes from another lane, after the arming above (a launch after a fault could not arm). Dropping that
    // lane's stream waits for its work, so the fault is in place when this returns, whatever the speed of this thread.
    faulter
        .submit(|q| {
            // SAFETY: `poke` writes through an address that is not mapped: the deliberate fault of this test.
            unsafe { q.launch(&poke, Launch::linear(1, 1), &Args::new().device(0x10)) }
        })
        .unwrap();
    drop(faulter);
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
    // Let the stream go: the driver delivers its four callbacks, then the one queued behind them.
    // A callback that pushed into freed memory would crash here or later.
    RELEASED.store(true, Ordering::Release);
    assert!(wait_for(&AFTER), "the driver never delivered the callbacks");
}
