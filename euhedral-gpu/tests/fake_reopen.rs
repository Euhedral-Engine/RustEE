//! A device released completely and opened again in one process, on the fake backend: the context is released when the
//! last holder goes, a later open works as the first did, and the sequence repeats. No GPU, no wall clock.
#![cfg(feature = "fake")]

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use euhedral_core::frames::RunnableFrame;
use euhedral_core::generics::LatticeSource;
use euhedral_gpu::fake::{FakeContext, FakeGpu, FakeModule};
use euhedral_gpu::{Args, CompileOptions, Gpu, LaneEnqueue, Launch, Module, Stream, StreamOptions};
use euhedral_graph::Completions;

const ROUNDS: u32 = 10;

fn open(device: &FakeContext) -> Arc<FakeGpu> {
    let gpu = device.open();
    gpu.define("add_word", |mem, _launch, args| {
        let at = args.read::<u64>(0).unwrap();
        let c = args.read::<u32>(1).unwrap();
        let word = mem.bytes_mut(at, 4)?;
        let v = u32::from_le_bytes(word.try_into().unwrap()) + c;
        word.copy_from_slice(&v.to_le_bytes());
        Ok(())
    });
    gpu
}

/// Everything a program holds on a device, in one round's order of making: lanes, a module, allocations, a host buffer
/// the copy writes, and a completions source with a marker registered.
struct Round {
    gpu: Arc<FakeGpu>,
    pool: Arc<euhedral_graph::LanePool>,
    streams: Vec<euhedral_gpu::fake::FakeStream>,
    module: FakeModule,
    word: u64,
    scratch: u64,
    completions: Arc<Completions>,
    emitted: Arc<AtomicU32>,
}

fn start(device: &FakeContext, add: u32) -> (Round, Vec<u8>) {
    let gpu = open(device);
    assert!(device.active(), "an open retains the context");
    let (pool, mut streams) = gpu
        .open_lanes(&["compute", "copy"], StreamOptions::default())
        .unwrap();
    let module =
        FakeModule::from_source(&gpu, "// fake source", &CompileOptions::default()).unwrap();
    let kernel = module.kernel("add_word").unwrap();
    let (word, scratch) = (gpu.alloc(4), gpu.alloc(4096));
    let mut host = vec![0u8; 4];
    streams[0]
        .submit(|q| {
            // SAFETY: `word` is a live fake allocation and the arguments match the fake kernel.
            unsafe {
                q.launch(
                    &kernel,
                    Launch::linear(1, 1),
                    &Args::new().device(word).arg(add),
                )
            }
        })
        .unwrap();
    let produced = streams[0].mark().unwrap();
    streams[1].await_marker(produced).unwrap();
    streams[1]
        .submit(|q| {
            // SAFETY: `host` outlives the run below, which the test drives before it reads it.
            unsafe { q.copy_to_host(host.as_mut_ptr(), word, 4) }
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
    gpu.run_until_idle();
    let round = Round {
        gpu,
        pool,
        streams,
        module,
        word,
        scratch,
        completions,
        emitted,
    };
    (round, host)
}

/// Drives the source as a worker would until nothing is pending.
fn emit_all(completions: &Completions) {
    completions.reap();
    completions.pull(
        &mut |mut frame| {
            frame.execute().unwrap();
            frame.do_finally();
        },
        &mut |_| false,
        u64::MAX,
    );
}

/// The documented order: marker registrations emitted, the source completed, the pool dropped, then the rest of what
/// holds the device, except the module, which is handed back to the caller.
fn release_keeping_module(round: Round) -> FakeModule {
    let Round {
        gpu,
        pool,
        streams,
        module,
        word,
        scratch,
        completions,
        emitted,
    } = round;
    emit_all(&completions);
    assert_eq!(completions.pending(), 0);
    assert_eq!(emitted.load(Ordering::Acquire), 1);
    LatticeSource::complete(&*completions);
    drop(streams);
    drop(pool);
    drop(completions);
    gpu.free(word);
    gpu.free(scratch);
    assert_eq!(gpu.live_allocations(), 0, "no allocation is outstanding");
    assert!(
        gpu.context().active(),
        "the gpu itself still holds the context"
    );
    drop(gpu);
    module
}

fn release(round: Round) {
    drop(release_keeping_module(round));
}

#[test]
fn a_device_released_completely_opens_again_ten_times() {
    let device = FakeContext::new();
    assert!(!device.active());
    for round in 0..ROUNDS {
        let (held, host) = start(&device, round + 1);
        assert_eq!(
            u32::from_le_bytes(host.try_into().unwrap()),
            round + 1,
            "work on a reopened device gives the first open's result"
        );
        assert_eq!(
            (device.retained(), device.released()),
            (u64::from(round) + 1, u64::from(round))
        );
        release(held);
        assert!(!device.active(), "round {round}: the context is released");
        assert_eq!(device.retained(), device.released());
    }
    assert_eq!(device.retained(), u64::from(ROUNDS));
}

#[test]
fn a_reopened_device_starts_empty() {
    let device = FakeContext::new();
    let (held, _) = start(&device, 5);
    let stale = held.word;
    release(held);
    let gpu = open(&device);
    assert_eq!(
        gpu.live_allocations(),
        0,
        "the old context's memory is gone"
    );
    assert!(gpu.log().is_empty());
    let fresh = gpu.alloc(4);
    assert_eq!(gpu.read(fresh, 4), [0; 4]);
    assert_ne!(stale, 0);
}

#[test]
fn a_module_kept_past_the_drop_keeps_the_context_until_it_goes() {
    let device = FakeContext::new();
    let (held, _) = start(&device, 1);
    let module = release_keeping_module(held);
    assert!(
        device.active(),
        "a module holds its device: the context stays until the module goes"
    );
    assert_eq!(device.released(), 0);
    drop(module);
    assert!(
        !device.active(),
        "the last holder's drop releases the context"
    );
    assert_eq!(device.retained(), device.released());
    // And the device opens again.
    let (held, host) = start(&device, 2);
    assert_eq!(u32::from_le_bytes(host.try_into().unwrap()), 2);
    release(held);
    assert!(!device.active());
}
