//! Lane sources on the CUDA backend, on a real device. Every test skips, passing, when no driver or device is present.
#![cfg(feature = "cuda")]
#![allow(
    clippy::disallowed_methods,
    reason = "the test thread is not a lattice worker: it requests the sources and sleeps between rounds"
)]

mod common;

use std::convert::Infallible;
use std::marker::PhantomData;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

use common::{Item, Plan, Probe, Value, collect_all, lock, pump};
use euhedral_gpu::cuda::{CudaGpu, CudaKernel, CudaModule, CudaStream};
use euhedral_gpu::{
    Args, CompileOptions, Enqueue, Gpu, GpuError, LaneSpec, Launch, Module, Stream, StreamOptions,
    StreamPriority, open_lane_sources,
};
use euhedral_graph::testing::FakeLanes;
use euhedral_graph::{
    Deliver, GroupId, LaneCx, LaneId, LaneIntakes, LanePoolBuilder, LaneSource, LaneSubmit,
    LaneWork, Marker, Outcome,
};
use euhedral_memory::cuda::DeviceMemory;
use euhedral_memory::{Backend, LocationId};

const OPS: &str = include_str!("kernels/ops.cu");
const OPS_PTX: &[u8] = include_bytes!("kernels/ops.ptx");

type CudaItem = Item<CudaStream>;

fn gpu() -> Option<Arc<CudaGpu>> {
    match CudaGpu::open(0) {
        Ok(gpu) => Some(gpu),
        Err(e) => {
            eprintln!("no CUDA device: skipped ({e})");
            None
        }
    }
}

fn module(gpu: &Arc<CudaGpu>) -> Option<CudaModule> {
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

fn wait_until(limit: Duration, mut condition: impl FnMut() -> bool) -> bool {
    let start = Instant::now();
    while !condition() {
        if start.elapsed() > limit {
            return false;
        }
        std::thread::sleep(Duration::from_micros(100));
    }
    true
}

fn specs(names: &[(&str, StreamPriority)]) -> Vec<LaneSpec<()>> {
    names
        .iter()
        .map(|(name, priority)| LaneSpec {
            name: (*name).to_owned(),
            options: StreamOptions {
                priority: *priority,
                ..StreamOptions::default()
            },
            state: (),
        })
        .collect()
}

/// Zeroes `bytes` at `addr` on a stream of its own and waits for it from the test thread.
fn zero(gpu: &Arc<CudaGpu>, addr: u64, bytes: usize) {
    let (pool, mut s) = gpu.open_lanes(&["zero"], StreamOptions::default()).unwrap();
    s[0].submit(|q| {
        // SAFETY: a live device allocation that nothing else uses yet.
        unsafe { q.memset(addr, 0, bytes) }
    })
    .unwrap();
    let marker = Stream::mark(&mut s[0]).unwrap();
    assert!(wait_until(Duration::from_secs(5), || pool.reached(marker)));
}

fn device_word(gpu: &Arc<CudaGpu>) -> (DeviceMemory, euhedral_memory::Block) {
    let mem = DeviceMemory {
        device: gpu.device().clone(),
        location: LocationId(0),
    };
    let block = mem.alloc(256).unwrap();
    zero(gpu, block.addr, 256);
    (mem, block)
}

/// A chain of `steps` fold items: each spins a little, then sets `x = 0.5 * x + c`, an order-sensitive fold of a float
/// word; the host word receives its bits.
fn fold_plan(
    module: &CudaModule,
    word: u64,
    steps: u32,
    route: impl Fn(u32) -> Deliver + Send + Sync + 'static,
) -> (Arc<Plan<CudaStream>>, f32) {
    let spin = module.kernel("spin").unwrap();
    let axpb = module.kernel("axpb").unwrap();
    let c = |step: u32| ((step % 7) + 1) as f32;
    let expected = (0..steps).fold(0.0f32, |x, step| 0.5 * x + c(step));
    let plan = Arc::new(Plan {
        steps,
        route: Box::new(route),
        launches: Box::new(move |step| {
            vec![
                (spin.clone(), Args::new().arg(200_000u64)),
                (
                    axpb.clone(),
                    Args::new()
                        .device(word)
                        .device(word)
                        .arg(0.5f32)
                        .arg(c(step))
                        .arg(1u32),
                ),
            ]
        }),
        device_word: word,
        host_word: Arc::new(AtomicU32::new(0)),
        probe: Probe::new(),
        retire_steps: false,
        _stream: PhantomData,
    });
    (plan, expected)
}

#[test]
fn lane_sources_on_the_device_run_a_chain_and_retire_it() {
    let Some(gpu) = gpu() else { return };
    let Some(module) = module(&gpu) else { return };
    let (mem, block) = device_word(&gpu);
    let opened = open_lane_sources::<_, (), CudaItem>(
        &gpu,
        specs(&[
            ("k0", StreamPriority::Normal),
            ("k1", StreamPriority::Normal),
            ("k2", StreamPriority::Normal),
            ("kv", StreamPriority::High),
        ]),
        &[&[LaneId(0), LaneId(1), LaneId(2)]],
    )
    .unwrap();
    // 300 items alternate between the group and the high-priority lane, then one copies the word out and retires.
    let steps = 300;
    let (plan, expected) = fold_plan(&module, block.addr, steps, |step| {
        if step % 2 == 0 {
            Deliver::Group(GroupId(0))
        } else {
            Deliver::Lane(LaneId(3))
        }
    });
    let collect = collect_all(&opened.sources);
    opened
        .intakes
        .deliver(CudaItem::first(&plan), Deliver::Group(GroupId(0)))
        .unwrap();
    assert!(
        wait_until(Duration::from_secs(10), || {
            pump(&opened.sources);
            !collect.received().is_empty()
        }),
        "the chain did not finish in time: {} items ran",
        plan.probe.ran().len()
    );
    assert_eq!(collect.received().len(), 1);
    let got = f32::from_bits(u32::try_from(collect.received()[0]).unwrap());
    assert_eq!(got, expected, "the fold ran out of order");
    assert!(lock(&plan.probe.failed).is_empty());
    assert!(lock(&plan.probe.submit_errors).is_empty());
    let retired = lock(&plan.probe.retired).clone();
    assert_eq!(retired.len(), 1);
    assert!(retired[0].1.is_none(), "{retired:?}");
    assert_eq!(plan.probe.ran().len(), steps as usize + 1);
    drop((opened, collect));
    // SAFETY: the lanes are gone (their streams waited for their work); nothing uses the block.
    unsafe { mem.free(block) };
}

#[test]
fn the_high_priority_lane_has_the_devices_greatest_priority() {
    let Some(gpu) = gpu() else { return };
    let options = [
        StreamOptions {
            priority: StreamPriority::High,
            ..StreamOptions::default()
        },
        StreamOptions::default(),
    ];
    let s = gpu
        .open_lanes_at(LaneId(0), &["high", "normal"], &options)
        .unwrap();
    let range = gpu.device().driver.symbol(c"cuCtxGetStreamPriorityRange");
    assert!(!range.is_null());
    // SAFETY: the driver exports cuCtxGetStreamPriorityRange(int *least, int *greatest) -> CUresult.
    let range: unsafe extern "C" fn(*mut i32, *mut i32) -> i32 =
        unsafe { std::mem::transmute(range) };
    let (mut least, mut greatest) = (0, 0);
    {
        let _c = gpu.device().enter().unwrap();
        // SAFETY: out pointers; the context is current.
        assert_eq!(unsafe { range(&mut least, &mut greatest) }, 0);
    }
    assert_eq!(s[0].priority().unwrap(), greatest);
    assert_eq!(s[1].priority().unwrap(), 0);
    assert!(greatest <= 0 && least >= 0);
}

/// Sets the buffer with a driver call on the lane's raw handle, behind a spin so that a missing order shows.
struct RawMemset {
    gpu: Arc<CudaGpu>,
    spin: CudaKernel,
    dst: u64,
    host: Arc<AtomicU32>,
    probe: Arc<Probe>,
    stage: u8,
    waits: Vec<Marker>,
}

impl LaneWork<CudaStream, ()> for RawMemset {
    const KINDS: &'static [&'static str] = &["memset", "read"];

    fn kind(&self) -> usize {
        usize::from(self.stage)
    }

    fn waits(&self) -> &[Marker] {
        &self.waits
    }

    fn run(&mut self, cx: &mut LaneCx<'_, CudaStream, (), Self>) -> Outcome<Infallible> {
        lock(&self.probe.ran).push(u64::from(self.stage));
        if self.stage == 1 {
            let (dst, host) = (self.dst, self.host.as_ptr().cast::<u8>());
            let read = cx
                .submitter()
                .submit(|q| {
                    // SAFETY: the host word lives in this item, kept until the retirement; the device word is a live
                    // allocation behind the memset.
                    unsafe { q.copy_to_host(host, dst, 4) }
                })
                .is_ok();
            assert!(read);
            return Outcome::Submitted { retire: true };
        }
        let spin = self.spin.clone();
        cx.submitter()
            .submit(|q| {
                // SAFETY: the spin touches no memory.
                unsafe { q.launch(&spin, Launch::linear(1, 1), &Args::new().arg(20_000_000u64)) }
            })
            .unwrap();
        let symbol = self.gpu.device().driver.symbol(c"cuMemsetD8Async");
        assert!(!symbol.is_null());
        // SAFETY: the driver exports cuMemsetD8Async(CUdeviceptr, unsigned char, size_t, CUstream) -> CUresult.
        let memset: unsafe extern "C" fn(u64, u8, usize, *mut std::ffi::c_void) -> i32 =
            unsafe { std::mem::transmute(symbol) };
        let _context = self.gpu.device().enter().unwrap();
        // SAFETY: the lane's handle, used from inside an item of this lane; the next mark covers the work. The
        // destination is a live allocation of at least 4 bytes that only this chain touches.
        let code = unsafe { memset(self.dst, 0xAB, 4, cx.submitter().raw()) };
        assert_eq!(code, 0);
        Outcome::Submitted { retire: false }
    }

    fn then(&mut self, marker: Option<Marker>, cx: &mut LaneCx<'_, CudaStream, (), Self>) {
        if self.stage == 1 {
            return;
        }
        let next = RawMemset {
            gpu: self.gpu.clone(),
            spin: self.spin.clone(),
            dst: self.dst,
            host: self.host.clone(),
            probe: self.probe.clone(),
            stage: 1,
            waits: marker.into_iter().collect(),
        };
        cx.deliver(next, Deliver::Lane(LaneId(1))).unwrap();
    }

    fn retired(
        self,
        _marker: Marker,
        failure: Option<GpuError>,
        cx: &mut LaneCx<'_, CudaStream, (), Self>,
    ) {
        assert!(failure.is_none(), "{failure:?}");
        cx.ready(Value::frame(u64::from(self.host.load(Ordering::SeqCst))));
    }

    fn failed(self, error: GpuError, _cx: &mut LaneCx<'_, CudaStream, (), Self>) {
        lock(&self.probe.failed).push((u64::from(self.stage), error));
    }
}

#[test]
fn work_enqueued_on_the_raw_handle_is_covered_by_the_next_mark() {
    let Some(gpu) = gpu() else { return };
    let Some(module) = module(&gpu) else { return };
    let (mem, block) = device_word(&gpu);
    let opened = open_lane_sources::<_, (), RawMemset>(
        &gpu,
        specs(&[("a", StreamPriority::Normal), ("b", StreamPriority::Normal)]),
        &[],
    )
    .unwrap();
    let collect = collect_all(&opened.sources);
    let probe = Probe::new();
    let first = RawMemset {
        gpu: gpu.clone(),
        spin: module.kernel("spin").unwrap(),
        dst: block.addr,
        host: Arc::new(AtomicU32::new(0)),
        probe: probe.clone(),
        stage: 0,
        waits: Vec::new(),
    };
    opened
        .intakes
        .deliver(first, Deliver::Lane(LaneId(0)))
        .unwrap();
    assert!(wait_until(Duration::from_secs(10), || {
        pump(&opened.sources);
        !collect.received().is_empty()
    }));
    // The dependent item on lane b copied the bytes out behind the marker that covers the raw memset.
    assert_eq!(collect.received(), vec![0xABAB_ABAB]);
    assert!(lock(&probe.failed).is_empty());
    drop((opened, collect));
    // SAFETY: the lanes are gone; nothing uses the block.
    unsafe { mem.free(block) };
}

#[test]
fn a_foreign_lane_in_the_pool_is_bridged_not_awaited() {
    let Some(gpu) = gpu() else { return };
    let Some(module) = module(&gpu) else { return };
    let (mem, block) = device_word(&gpu);
    // One pool of two backends: CUDA lanes 0 and 1, and a foreign lane 2.
    let mut builder = LanePoolBuilder::new();
    let first = builder.reserve(2);
    let foreign_first = builder.reserve(1);
    let streams = gpu
        .open_lanes_at(first, &["a", "b"], &[StreamOptions::default(); 2])
        .unwrap();
    for stream in &streams {
        builder.set(Stream::lane(stream), Stream::queue(stream));
    }
    let (_fake, mut foreign, queues) = FakeLanes::open(foreign_first, 1);
    builder.set(foreign_first, queues.into_iter().next().unwrap());
    let pool = builder.build();
    let marker = LaneSubmit::mark(&mut foreign[0]).unwrap().unwrap();
    assert!(!LaneSubmit::accepts(&streams[0], marker));
    let (intakes, inputs) = LaneIntakes::<CudaItem>::new(pool, &[]);
    let sources: Vec<_> = streams
        .into_iter()
        .zip(inputs)
        .map(|(stream, input)| LaneSource::new(stream, (), input, intakes.clone()))
        .collect();
    let collect = collect_all(&sources);
    // A chain of one step, then the read; the first item names the foreign marker among its inputs.
    let (plan, expected) = fold_plan(&module, block.addr, 1, |_| Deliver::Lane(LaneId(1)));
    let mut item = CudaItem::first(&plan);
    item.waits = vec![marker];
    intakes.deliver(item, Deliver::Lane(LaneId(0))).unwrap();
    assert!(wait_until(Duration::from_secs(10), || {
        pump(&sources);
        !collect.received().is_empty()
    }));
    // The foreign marker is for the item to bridge; the lane never tried to wait for it on the device (which would
    // have been refused as a marker of no stream of this call).
    assert_eq!(*lock(&plan.probe.bridged), vec![(0, marker)]);
    assert!(lock(&plan.probe.failed).is_empty());
    assert_eq!(
        f32::from_bits(u32::try_from(collect.received()[0]).unwrap()),
        expected
    );
    drop((sources, intakes, collect));
    // SAFETY: the lanes are gone; nothing uses the block.
    unsafe { mem.free(block) };
}
