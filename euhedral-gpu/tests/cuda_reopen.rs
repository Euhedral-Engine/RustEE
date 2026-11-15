//! A device released completely and opened again in one process, on a real card: after everything that holds the device
//! is dropped in the documented order the driver reports the primary context inactive, a later open computes the same
//! result, and ten rounds in a row leave the card's free memory where it was. Skips, passing, with no driver, device or
//! kernel compiler.
#![cfg(feature = "cuda")]
#![allow(
    clippy::disallowed_methods,
    reason = "the test thread is not a lattice worker: it polls the source and times the calls"
)]

mod common;

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

use common::cuda_support::wait_until;
use euhedral_core::frames::RunnableFrame;
use euhedral_core::generics::LatticeSource;
use euhedral_gpu::cuda::{CudaGpu, CudaModule, Nvrtc};
use euhedral_gpu::{Args, CompileOptions, Gpu, LaneEnqueue, Launch, Module, Stream, StreamOptions};
use euhedral_graph::Completions;
use euhedral_memory::cuda::{DeviceMemory, Driver, PinnedHost};
use euhedral_memory::{Backend, LocationId};

const OPS: &str = include_str!("kernels/ops.cu");
const OPS_PTX: &[u8] = include_bytes!("kernels/ops.ptx");
const ROUNDS: usize = 10;
/// Floats per buffer: 16 MiB each, so a round that leaked its device buffers would show within a few rounds.
const N: usize = 4 << 20;
/// How far the free memory read after an open may fall below the first open's. Other programs share the card; a leak of
/// the round's two device buffers (32 MiB) per round passes this within four rounds.
const TOLERANCE: u64 = 128 << 20;

/// The test kernels, from source when the compiler is present, else from the prebuilt image.
fn load_module(gpu: &Arc<CudaGpu>) -> Option<(CudaModule, &'static str)> {
    if Nvrtc::shared(None).is_ok() {
        let module = CudaModule::from_source(gpu, OPS, &CompileOptions::default()).unwrap();
        return Some((module, "source"));
    }
    if gpu.driver_version() >= 13010 {
        return Some((CudaModule::from_image(gpu, OPS_PTX).unwrap(), "image"));
    }
    None
}

struct Opened {
    gpu: Arc<CudaGpu>,
    free: u64,
    open: Duration,
}

fn open() -> Option<Opened> {
    let start = Instant::now();
    let gpu = match CudaGpu::open(0) {
        Ok(gpu) => gpu,
        Err(e) => {
            eprintln!("no CUDA device: skipped ({e})");
            return None;
        }
    };
    let (free, _) = gpu.device().mem_info().unwrap();
    Some(Opened {
        gpu,
        free,
        open: start.elapsed(),
    })
}

/// One round's work, then the release in the documented order. Returns the time the release took.
fn round(gpu: Arc<CudaGpu>, module: CudaModule, driver: &Driver) -> Duration {
    let (pool, mut streams) = gpu
        .open_lanes(&["compute", "copy"], StreamOptions::default())
        .unwrap();
    let mem = DeviceMemory {
        device: gpu.device().clone(),
        location: LocationId(0),
    };
    let pinned = PinnedHost::new(gpu.device().clone(), LocationId(1), 64 << 20);
    let bytes = N * 4;
    let (x_dev, y_dev) = (
        mem.alloc(bytes as u64).unwrap(),
        mem.alloc(bytes as u64).unwrap(),
    );
    // The pinned loans: one to read from, one to write to.
    let (x_host, y_host) = (
        pinned.alloc(bytes as u64).unwrap(),
        pinned.alloc(bytes as u64).unwrap(),
    );
    let (a, b) = (1.5f32, -2.0f32);
    // SAFETY: a pinned block of `bytes` bytes, owned here and untouched by the device until the first copy runs.
    let x = unsafe { std::slice::from_raw_parts_mut(x_host.addr as *mut f32, N) };
    for (i, v) in x.iter_mut().enumerate() {
        *v = i as f32 * 0.25;
    }
    let axpb = module.kernel("axpb").unwrap();
    let args = Args::new()
        .device(y_dev.addr)
        .device(x_dev.addr)
        .arg(a)
        .arg(b)
        .arg(N as u32);
    streams[0]
        .submit(|q| {
            // SAFETY: valid blocks and arguments that match `axpb`; the buffers live until the marker below is reached.
            unsafe {
                q.copy_to_device(x_dev.addr, x_host.addr as *const u8, bytes)?;
                q.launch(&axpb, Launch::linear(N as u32, 256), &args)
            }
        })
        .unwrap();
    let produced = streams[0].mark().unwrap();
    streams[1].await_marker(produced).unwrap();
    streams[1]
        .submit(|q| {
            // SAFETY: valid blocks, live until the marker below is reached.
            unsafe { q.copy_to_host(y_host.addr as *mut u8, y_dev.addr, bytes) }
        })
        .unwrap();
    let copied = streams[1].mark().unwrap();
    let completions = Completions::new(&pool);
    let emitted = Arc::new(AtomicU32::new(0));
    completions.after_marker(
        copied,
        Box::new(RunnableFrame::new(1, {
            let emitted = emitted.clone();
            Arc::new(move || {
                emitted.fetch_add(1, Ordering::AcqRel);
            })
        })),
    );
    // The test thread is the worker: it polls until the marker is reached and its frame has run.
    assert!(wait_until(Duration::from_secs(30), || {
        completions.reap();
        completions.pull(
            &mut |mut frame| {
                frame.execute().unwrap();
                frame.do_finally();
            },
            &mut |_| false,
            u64::MAX,
        );
        completions.pending() == 0
    }));
    assert_eq!(emitted.load(Ordering::Acquire), 1);
    let x_ref: &[f32] = x;
    // SAFETY: the copy that wrote this block is done (the marker was reached).
    let y = unsafe { std::slice::from_raw_parts(y_host.addr as *const f32, N) };
    assert!(
        x_ref
            .iter()
            .zip(y)
            .all(|(&x, &y)| y.to_bits() == a.mul_add(x, b).to_bits()),
        "the kernel's output after an open is the CPU's, bit for bit"
    );

    // Release, in the documented order. Registrations are emitted (`pending() == 0`); complete the source; drop the
    // pool; then everything else that holds the device; the gpu last.
    let start = Instant::now();
    LatticeSource::complete(&*completions);
    drop(streams);
    drop(pool);
    drop(completions);
    // SAFETY: the copies that used the blocks ended (the marker was reached) and the streams are gone.
    unsafe {
        pinned.free(x_host);
        pinned.free(y_host);
        mem.free(x_dev);
        mem.free(y_dev);
    }
    drop(pinned);
    drop(mem);
    drop(axpb);
    drop(module);
    assert!(
        driver.primary_context_active(0).unwrap(),
        "the gpu itself still holds the context"
    );
    drop(gpu);
    start.elapsed()
}

#[test]
fn a_device_released_completely_opens_again_ten_times() {
    let Ok(driver) = Driver::shared() else {
        eprintln!("no CUDA driver: skipped");
        return;
    };
    assert!(
        !driver.primary_context_active(0).unwrap_or(false),
        "nothing in this process has opened the device yet"
    );
    let mut first_free = None;
    for n in 0..ROUNDS {
        let Some(Opened { gpu, free, open }) = open() else {
            return;
        };
        assert!(driver.primary_context_active(0).unwrap());
        let Some((module, how)) = load_module(&gpu) else {
            eprintln!("neither NVRTC nor a 13.1 driver: skipped");
            return;
        };
        let release = round(gpu, module, &driver);
        let inactive = !driver.primary_context_active(0).unwrap();
        eprintln!(
            "round {n}: module from {how}; open {open:?}; free after open {} MiB; release {release:?}; primary context inactive after release: {inactive}",
            free >> 20
        );
        assert!(inactive, "round {n}: the driver's context is gone");
        let first = *first_free.get_or_insert(free);
        assert!(
            free + TOLERANCE >= first,
            "round {n}: free memory after an open fell from {} to {} MiB: the process holds memory on the card",
            first >> 20,
            free >> 20
        );
    }
}
