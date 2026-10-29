//! A deliberate device fault under lane sources. A sticky fault poisons the context for the whole process, so this
//! file holds one test and runs as its own process. Skips, passing, when no driver or device is present.
#![cfg(feature = "cuda")]
#![allow(
    clippy::disallowed_methods,
    reason = "the test thread is not a lattice worker: it requests the sources and sleeps between rounds"
)]

mod common;

use std::marker::PhantomData;
use std::sync::Arc;
use std::sync::atomic::AtomicU32;
use std::time::{Duration, Instant};

use common::{Item, Plan, Probe, collect_all, lock, pump};
use euhedral_gpu::cuda::{CudaGpu, CudaModule, CudaStream};
use euhedral_gpu::{
    Args, CompileOptions, Gpu, GpuError, LaneSpec, Module, StreamOptions, open_lane_sources,
};
use euhedral_graph::{Deliver, LaneId, Marker};

const OPS: &str = include_str!("kernels/ops.cu");
const OPS_PTX: &[u8] = include_bytes!("kernels/ops.ptx");

type CudaItem = Item<CudaStream>;

fn wait_until(limit: Duration, mut condition: impl FnMut() -> bool) -> bool {
    let start = Instant::now();
    while !condition() {
        if start.elapsed() > limit {
            return false;
        }
        std::thread::sleep(Duration::from_micros(200));
    }
    true
}

#[test]
fn a_poisoned_device_fails_armed_items() {
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
    let lanes = ["a", "b"]
        .iter()
        .map(|name| LaneSpec {
            name: (*name).to_owned(),
            options: StreamOptions::default(),
            state: (),
        })
        .collect();
    let opened = open_lane_sources::<_, (), CudaItem>(&gpu, lanes, &[]).unwrap();
    let _collect = collect_all(&opened.sources);
    // Step 0 (lane a) asks for a retirement, spins, then writes through an unmapped address: the deliberate fault. Step 1 (lane b) is the
    // item that copies out behind step 0's marker and asks for a retirement.
    let plan = Arc::new(Plan::<CudaStream> {
        steps: 1,
        route: Box::new(|_| Deliver::Lane(LaneId(1))),
        launches: Box::new(move |_| {
            vec![
                (spin.clone(), Args::new().arg(125_000_000u64)),
                (poke.clone(), Args::new().device(0x10)),
            ]
        }),
        device_word: 0x20,
        host_word: Arc::new(AtomicU32::new(0)),
        probe: Probe::new(),
        retire_steps: true,
        _stream: PhantomData,
    });
    opened
        .intakes
        .deliver(CudaItem::first(&plan), Deliver::Lane(LaneId(0)))
        .unwrap();
    // The faulting item asks for a retirement as well: its callback reports the device's failure.
    assert!(wait_until(Duration::from_secs(20), || {
        pump(&opened.sources);
        !lock(&plan.probe.retired).is_empty()
    }));
    let retired = lock(&plan.probe.retired).clone();
    assert!(
        retired
            .iter()
            .any(|(tag, failure)| *tag == 0 && failure.is_some()),
        "the retirement carried no failure: {retired:?}"
    );
    assert!(gpu.poisoned().is_some(), "the failure poisoned the device");

    // Later items fail fast: ordering behind a marker of the other lane is refused with the cause.
    let marker_of_a = Marker::new(opened.pool.lane(LaneId(0)).id(), LaneId(0), 1);
    let mut later = CudaItem::first(&plan);
    later.tag = 7;
    later.waits = vec![marker_of_a];
    opened
        .intakes
        .deliver(later, Deliver::Lane(LaneId(1)))
        .unwrap();
    assert!(wait_until(Duration::from_secs(10), || {
        pump(&opened.sources);
        !lock(&plan.probe.failed).is_empty()
    }));
    let failed = lock(&plan.probe.failed).clone();
    assert_eq!(failed.len(), 1);
    assert_eq!(failed[0].0, 7);
    assert!(
        matches!(failed[0].1, GpuError::Poisoned(_)),
        "{:?}",
        failed[0].1
    );
}
