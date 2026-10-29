//! CUDA fixtures of the lane-source tests: the device and its test kernels, device and pinned words, and the fault
//! scenario that two test binaries run (a fault is sticky for the process, so each binary runs it once).

#![allow(dead_code, reason = "each test binary uses part of the fixtures")]
#![allow(
    clippy::disallowed_methods,
    reason = "the test thread is not a lattice worker: it requests the sources and sleeps between rounds"
)]

use std::marker::PhantomData;
use std::sync::Arc;
use std::time::{Duration, Instant};

use super::{HostWord, Item, Plan, Probe, collect_all, lock, pump_from};
use euhedral_gpu::cuda::{CudaGpu, CudaModule, CudaStream};
use euhedral_gpu::{
    Args, CompileOptions, Enqueue, Gpu, GpuError, LaneSpec, Module, Stream, StreamOptions,
    open_lane_sources,
};
use euhedral_graph::{Deliver, LaneId};
use euhedral_memory::cuda::{DeviceMemory, PinnedHost};
use euhedral_memory::{Backend, Block, LocationId};

const OPS: &str = include_str!("../kernels/ops.cu");
const OPS_PTX: &[u8] = include_bytes!("../kernels/ops.ptx");

pub fn gpu() -> Option<Arc<CudaGpu>> {
    match CudaGpu::open(0) {
        Ok(gpu) => Some(gpu),
        Err(e) => {
            eprintln!("no CUDA device: skipped ({e})");
            None
        }
    }
}

/// The test kernels (`spin`, `axpb`, `add_u32`, `poke`), from the prebuilt image or compiled.
pub fn module(gpu: &Arc<CudaGpu>) -> Option<CudaModule> {
    let module = if gpu.driver_version() >= 13010 {
        CudaModule::from_image(gpu, OPS_PTX)
    } else {
        CudaModule::from_source(gpu, OPS, &CompileOptions::default())
    };
    match module {
        Ok(module) => Some(module),
        Err(e) => {
            eprintln!("no way to load the test kernels: skipped ({e})");
            None
        }
    }
}

pub fn wait_until(limit: Duration, mut condition: impl FnMut() -> bool) -> bool {
    let start = Instant::now();
    while !condition() {
        if start.elapsed() > limit {
            return false;
        }
        std::thread::sleep(Duration::from_micros(100));
    }
    true
}

/// Zeroes `bytes` at `addr` on a stream of its own and waits for it from the test thread.
pub fn zero(gpu: &Arc<CudaGpu>, addr: u64, bytes: usize) {
    let (pool, mut s) = gpu.open_lanes(&["zero"], StreamOptions::default()).unwrap();
    s[0].submit(|q| {
        // SAFETY: a live device allocation that nothing else uses yet.
        unsafe { q.memset(addr, 0, bytes) }
    })
    .unwrap();
    let marker = Stream::mark(&mut s[0]).unwrap();
    assert!(wait_until(Duration::from_secs(5), || pool.reached(marker)));
}

/// A zeroed device allocation of 256 bytes.
pub fn device_word(gpu: &Arc<CudaGpu>) -> (DeviceMemory, Block) {
    let mem = DeviceMemory {
        device: gpu.device().clone(),
        location: LocationId(0),
    };
    let block = mem.alloc(256).unwrap();
    zero(gpu, block.addr, 256);
    (mem, block)
}

/// A pinned host word, for a copy out of the device. Freed on drop: drop it after the lanes that copy into it.
pub struct Pinned {
    backend: PinnedHost,
    block: Block,
}

impl Pinned {
    pub fn new(gpu: &Arc<CudaGpu>) -> Self {
        let backend = PinnedHost::new(gpu.device().clone(), LocationId(1), 0);
        let block = backend.alloc(4096).unwrap();
        // SAFETY: a pinned block of 4096 bytes, owned by this value.
        unsafe { std::ptr::write_bytes(block.addr as *mut u8, 0, 4096) };
        Self { backend, block }
    }

    pub fn word(&self) -> HostWord {
        // SAFETY: the block is page aligned, 4096 bytes, and lives until this value drops, which the tests do after
        // the lanes that use the word.
        unsafe { HostWord::from_raw(self.block.addr as *mut u32) }
    }
}

impl Drop for Pinned {
    fn drop(&mut self) {
        // SAFETY: the block came from this backend; the lanes that copied into it are gone.
        unsafe { self.backend.free(self.block) };
    }
}

/// Lane a faults on the device; lane b's item waits for lane a's marker and asks for a retirement. With
/// `retire_steps` the faulting item asks for one as well; without, the item behind the fault is the only retirement
/// and nothing but a look at the device can find that its callback will never come.
///
/// Run it once per process: the fault is sticky.
pub fn fault_run(retire_steps: bool) {
    let Some(gpu) = gpu() else { return };
    let Some(module) = module(&gpu) else { return };
    let (_mem, block) = device_word(&gpu);
    let pinned = Pinned::new(&gpu);
    let lanes = ["a", "b"]
        .iter()
        .map(|name| LaneSpec {
            name: (*name).to_owned(),
            options: StreamOptions::default(),
            state: (),
        })
        .collect();
    let opened = open_lane_sources::<_, (), Item<CudaStream>>(&gpu, lanes, &[]).unwrap();
    let _collect = collect_all(&opened.sources);
    let (spin, poke) = (
        module.kernel("spin").unwrap(),
        module.kernel("poke").unwrap(),
    );
    // Step 0 (lane a) spins, then writes through an unmapped address: the deliberate fault. Step 1 (lane b) copies the
    // word out behind step 0's marker and asks for a retirement.
    let plan = Arc::new(Plan::<CudaStream> {
        steps: 1,
        route: Box::new(|_| Deliver::Lane(LaneId(1))),
        launches: Box::new(move |_| {
            vec![
                (spin.clone(), Args::new().arg(125_000_000u64)),
                (poke.clone(), Args::new().device(0x10)),
            ]
        }),
        device_word: block.addr,
        host_word: pinned.word(),
        probe: Probe::new(),
        retire_steps,
        _stream: PhantomData,
    });
    opened
        .intakes
        .deliver(Item::first(&plan), Deliver::Lane(LaneId(0)))
        .unwrap();
    let expected = if retire_steps { 2 } else { 1 };
    let mut round = 0;
    assert!(
        wait_until(Duration::from_secs(30), || {
            pump_from(&opened.sources, round % 2);
            round += 1;
            lock(&plan.probe.retired).len() >= expected
        }),
        "a retirement was stranded: {:?}",
        lock(&plan.probe.retired)
    );
    let retired = lock(&plan.probe.retired).clone();
    let step = |n: u32| retired.iter().find(|(_, step, _)| *step == n);
    let Some((_, _, Some(read))) = step(1) else {
        panic!("step 1's retirement carried no failure: {retired:?}");
    };
    assert!(
        matches!(read, GpuError::Poisoned(_) | GpuError::Fault { .. }),
        "{read:?}"
    );
    if retire_steps {
        let Some((_, _, Some(first))) = step(0) else {
            panic!("step 0's retirement carried no failure: {retired:?}");
        };
        assert!(
            matches!(first, GpuError::Fault { code: 700, .. }),
            "{first:?}"
        );
    } else {
        assert!(step(0).is_none(), "{retired:?}");
    }
    assert!(gpu.poisoned().is_some(), "the failure poisoned the device");
    assert!(
        lock(&plan.probe.failed).is_empty(),
        "no submission was refused before the fault: {:?}",
        lock(&plan.probe.failed)
    );

    // Later work fails fast: an item's submission on the poisoned device ends it with `failed`.
    let mut late = Item::first(&plan);
    late.tag = 9;
    opened
        .intakes
        .deliver(late, Deliver::Lane(LaneId(0)))
        .unwrap();
    assert!(wait_until(Duration::from_secs(10), || {
        pump_from(&opened.sources, 0);
        !lock(&plan.probe.failed).is_empty()
    }));
    let failed = lock(&plan.probe.failed).clone();
    assert_eq!(failed.len(), 1, "{failed:?}");
    assert_eq!((failed[0].0, failed[0].1), (9, 0));
    assert!(
        matches!(failed[0].2, GpuError::Poisoned(_)),
        "{:?}",
        failed[0].2
    );
    // The device is lost: nothing frees its memory or waits for it again in this process.
    std::mem::forget(pinned);
    std::mem::forget(opened);
}
