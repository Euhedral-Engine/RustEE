//! The fake device as a stand-in for a card in residency and failure tests: bytes held and their peak, a size,
//! injected allocation and module-load failures, and who submitted each operation. No GPU, no wall clock.
#![cfg(feature = "fake")]
#![allow(
    clippy::disallowed_methods,
    reason = "a peer thread stands in for the lattice's worker that submits"
)]

use std::thread;

use euhedral_gpu::fake::{FakeContext, FakeGpu, FakeMemoryEvent, FakeModule};
use euhedral_gpu::{CompileOptions, Gpu, GpuError, LaneEnqueue, Module, Stream, StreamOptions};

#[test]
fn bytes_held_follow_allocations_and_frees_and_the_peak_stays() {
    let gpu = FakeGpu::new();
    assert_eq!((gpu.bytes_held(), gpu.peak_bytes()), (0, 0));
    let a = gpu.alloc(1000);
    let b = gpu.alloc(24);
    assert_eq!(gpu.bytes_held(), 1024);
    gpu.free(a);
    assert_eq!((gpu.bytes_held(), gpu.peak_bytes()), (24, 1024));
    gpu.free(b);
    assert_eq!((gpu.bytes_held(), gpu.peak_bytes()), (0, 1024));
    assert_eq!(gpu.live_allocations(), 0);
}

#[test]
fn an_allocation_past_the_capacity_fails_and_holds_nothing() {
    let gpu = FakeGpu::new();
    assert_eq!(gpu.free_bytes(), None);
    gpu.set_capacity(Some(100));
    let a = gpu.try_alloc(60).unwrap();
    assert_eq!(gpu.free_bytes(), Some(40));
    let refused = gpu.try_alloc(41).unwrap_err();
    assert!(matches!(refused, GpuError::Call { name, .. } if name == "OUT_OF_MEMORY"));
    assert_eq!((gpu.bytes_held(), gpu.live_allocations()), (60, 1));
    gpu.free(a);
    assert!(
        gpu.try_alloc(100).is_ok(),
        "the freed bytes can be taken again"
    );
}

#[test]
fn injected_allocation_failures_skip_then_fail_then_stop() {
    let gpu = FakeGpu::new();
    gpu.fail_allocs(1, 2);
    let results: Vec<bool> = (0..5).map(|_| gpu.try_alloc(8).is_ok()).collect();
    assert_eq!(results, [true, false, false, true, true]);
    assert_eq!(gpu.bytes_held(), 24, "the failed calls held nothing");
}

#[test]
fn injected_module_load_failures_fail_both_routes() {
    let gpu = FakeGpu::new();
    gpu.fail_module_loads(1, 2);
    let options = CompileOptions::default();
    assert!(FakeModule::from_source(&gpu, "a", &options).is_ok());
    assert!(matches!(
        FakeModule::from_source(&gpu, "b", &options),
        Err(GpuError::Load(_))
    ));
    assert!(matches!(
        FakeModule::from_image(&gpu, b"c"),
        Err(GpuError::Load(_))
    ));
    assert!(FakeModule::from_image(&gpu, b"d").is_ok());
}

#[test]
fn each_operation_names_its_submitter_and_each_allocation_its_caller() {
    let gpu = FakeGpu::new();
    let (_pool, mut streams) = gpu
        .open_lanes(&["a", "b"], StreamOptions::default())
        .unwrap();
    let mut second = streams.pop().unwrap();
    let mut first = streams.pop().unwrap();
    let buf = gpu.alloc(4);
    {
        let _label = FakeGpu::label_caller("first");
        // SAFETY: `buf` is a live fake allocation.
        first.submit(|q| unsafe { q.memset(buf, 0, 4) }).unwrap();
        let _inner = FakeGpu::label_caller("inner");
        let _ = gpu.alloc(8);
    }
    let worker = thread::Builder::new()
        .name("submitter".into())
        .spawn(move || {
            let _label = FakeGpu::label_caller("second");
            // SAFETY: `buf` is a live fake allocation.
            second.submit(|q| unsafe { q.memset(buf, 1, 4) }).unwrap();
            thread::current().id()
        })
        .unwrap();
    let worker_thread = worker.join().unwrap();
    gpu.run_until_idle();

    let calls = gpu.calls();
    assert_eq!(calls.len(), 2);
    assert_eq!(
        calls.len(),
        gpu.log().len(),
        "one caller for each logged op"
    );
    assert_eq!(calls[0].op, gpu.log()[0]);
    assert_eq!(calls[0].caller.thread, thread::current().id());
    assert_eq!(calls[0].caller.label.as_deref(), Some("first"));
    assert_eq!(calls[1].caller.thread, worker_thread);
    assert_eq!(calls[1].caller.name.as_deref(), Some("submitter"));
    assert_eq!(calls[1].caller.label.as_deref(), Some("second"));

    let events = gpu.memory_log();
    let FakeMemoryEvent::Alloc { caller, bytes, .. } = &events[1] else {
        panic!("the second event is an allocation");
    };
    assert_eq!((*bytes, caller.label.as_deref()), (8, Some("inner")));
    let FakeMemoryEvent::Alloc { caller, .. } = &events[0] else {
        panic!("the first event is an allocation");
    };
    assert_eq!(
        caller.label, None,
        "the label of a later scope is not on an earlier call"
    );
}

#[test]
fn a_label_ends_with_its_guard() {
    let gpu = FakeGpu::new();
    {
        let _label = FakeGpu::label_caller("scoped");
        gpu.alloc(1);
    }
    gpu.alloc(1);
    let labels: Vec<_> = gpu
        .memory_log()
        .into_iter()
        .map(|event| match event {
            FakeMemoryEvent::Alloc { caller, .. } | FakeMemoryEvent::Free { caller, .. } => {
                caller.label
            }
        })
        .collect();
    assert_eq!(labels, [Some("scoped".to_owned()), None]);
}

fn label_now(gpu: &FakeGpu) -> Option<String> {
    gpu.alloc(1);
    match gpu.memory_log().pop().unwrap() {
        FakeMemoryEvent::Alloc { caller, .. } => caller.label,
        FakeMemoryEvent::Free { .. } => unreachable!("the last event is the allocation"),
    }
}

#[test]
fn dropping_an_inner_label_shows_the_outer_one_again() {
    let gpu = FakeGpu::new();
    let _outer = FakeGpu::label_caller("outer");
    let inner = FakeGpu::label_caller("inner");
    assert_eq!(label_now(&gpu).as_deref(), Some("inner"));
    drop(inner);
    assert_eq!(label_now(&gpu).as_deref(), Some("outer"));
}

#[test]
fn labels_dropped_out_of_order_leave_no_stale_label() {
    let gpu = FakeGpu::new();
    let a = FakeGpu::label_caller("a");
    let b = FakeGpu::label_caller("b");
    drop(a);
    assert_eq!(label_now(&gpu).as_deref(), Some("b"));
    drop(b);
    assert_eq!(label_now(&gpu), None);
}

#[test]
fn the_capacity_belongs_to_the_device_and_a_re_open_keeps_it() {
    let device = FakeContext::new();
    let first = device.open();
    first.set_capacity(Some(100));
    assert!(first.try_alloc(90).is_ok());
    drop(first);
    let second = device.open();
    assert_eq!(second.free_bytes(), Some(100), "a new open holds nothing");
    assert!(
        second.try_alloc(101).is_err(),
        "the size is still the device's"
    );
    assert!(second.try_alloc(100).is_ok());
}

#[test]
fn a_refusal_for_the_capacity_does_not_use_up_a_skip() {
    let gpu = FakeGpu::new();
    gpu.set_capacity(Some(10));
    gpu.fail_allocs(1, 1);
    assert!(gpu.try_alloc(11).is_err(), "too big: refused for size");
    assert!(
        gpu.try_alloc(1).is_ok(),
        "the one skipped call, still to come"
    );
    assert!(gpu.try_alloc(1).is_err(), "the injected failure");
    assert!(gpu.try_alloc(1).is_ok());
}
