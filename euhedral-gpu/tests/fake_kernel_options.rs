//! Kernel lookup options and the dynamic shared memory limit on the fake backend. No GPU.
#![cfg(feature = "fake")]

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use euhedral_gpu::fake::{
    DEFAULT_DYNAMIC_SHARED_BYTES, FakeGpu, FakeKernel, FakeModule, FakeOp, MAX_DYNAMIC_SHARED_BYTES,
};
use euhedral_gpu::{
    Args, CompileOptions, Gpu, GpuError, Kernel, KernelOptions, LaneEnqueue, Launch, Module,
    Sequence, Stream, StreamOptions,
};

/// Runs of the `noop` kernel; only the test that reads it launches it with a refusal in mind, and it reads a delta.
static RUNS: AtomicUsize = AtomicUsize::new(0);
const OPT_IN: u32 = 64 * 1024;

fn module() -> (Arc<FakeGpu>, FakeModule) {
    let gpu = FakeGpu::new();
    gpu.define("noop", |_m, _l, _a| {
        RUNS.fetch_add(1, Ordering::SeqCst);
        Ok(())
    });
    let module = FakeModule::from_source(&gpu, "// kernels", &CompileOptions::default()).unwrap();
    (gpu, module)
}

fn opted(module: &FakeModule, bytes: u32) -> FakeKernel {
    module
        .kernel_with(
            "noop",
            KernelOptions {
                max_dynamic_shared_bytes: bytes,
            },
        )
        .unwrap()
}

fn launch_with(gpu: &Arc<FakeGpu>, kernel: &FakeKernel, shared_bytes: u32) -> Result<(), GpuError> {
    let (_pool, mut streams) = gpu.open_lanes(&["a"], StreamOptions::default()).unwrap();
    let launch = Launch {
        shared_bytes,
        ..Launch::linear(1, 1)
    };
    let result = streams[0].submit(|e| {
        // SAFETY: the kernel takes no arguments.
        unsafe { e.launch(kernel, launch, &Args::new()) }
    });
    gpu.run_until_idle();
    result
}

#[test]
fn the_option_is_recorded_on_the_kernel() {
    let (_gpu, module) = module();
    assert_eq!(opted(&module, OPT_IN).max_dynamic_shared_bytes(), OPT_IN);
}

#[test]
fn a_default_lookup_is_a_lookup_with_default_options() {
    let (_gpu, module) = module();
    assert_eq!(KernelOptions::default().max_dynamic_shared_bytes, 0);
    let plain = module.kernel("noop").unwrap();
    let explicit = opted(&module, 0);
    assert_eq!(plain.id(), explicit.id());
    assert_eq!(
        plain.max_dynamic_shared_bytes(),
        DEFAULT_DYNAMIC_SHARED_BYTES
    );
    assert_eq!(
        explicit.max_dynamic_shared_bytes(),
        DEFAULT_DYNAMIC_SHARED_BYTES
    );
}

#[test]
fn a_launch_above_the_limit_is_an_error_that_names_the_kernel_the_bytes_and_the_limit() {
    let (gpu, module) = module();
    let kernel = module.kernel("noop").unwrap();
    launch_with(&gpu, &kernel, DEFAULT_DYNAMIC_SHARED_BYTES).unwrap();
    let error = launch_with(&gpu, &kernel, DEFAULT_DYNAMIC_SHARED_BYTES + 1).unwrap_err();
    let GpuError::Invalid(text) = &error else {
        panic!("{error:?}")
    };
    assert!(text.contains("noop"), "{text}");
    assert!(
        text.contains(&(DEFAULT_DYNAMIC_SHARED_BYTES + 1).to_string()),
        "{text}"
    );
    assert!(
        text.contains(&DEFAULT_DYNAMIC_SHARED_BYTES.to_string()),
        "{text}"
    );
}

#[test]
fn an_opted_in_kernel_launches_up_to_its_limit_and_not_past_it() {
    let (gpu, module) = module();
    let kernel = opted(&module, OPT_IN);
    launch_with(&gpu, &kernel, OPT_IN).unwrap();
    assert!(launch_with(&gpu, &kernel, OPT_IN + 1).is_err());
}

#[test]
fn a_second_lookup_raises_the_limit_for_every_handle_and_never_lowers_it() {
    let (gpu, module) = module();
    let early = module.kernel("noop").unwrap();
    let big = opted(&module, OPT_IN);
    assert_eq!(early.max_dynamic_shared_bytes(), OPT_IN);
    let small = opted(&module, 1024);
    let default = module.kernel("noop").unwrap();
    for kernel in [&early, &big, &small, &default] {
        assert_eq!(kernel.max_dynamic_shared_bytes(), OPT_IN);
    }
    launch_with(&gpu, &small, OPT_IN).unwrap();
}

fn launches(gpu: &FakeGpu) -> usize {
    gpu.log()
        .iter()
        .filter(|op| matches!(op, FakeOp::Launch { .. }))
        .count()
}

#[test]
fn a_refused_launch_enqueues_and_runs_nothing() {
    let (gpu, module) = module();
    let kernel = module.kernel("noop").unwrap();
    let (runs, ops) = (RUNS.load(Ordering::SeqCst), launches(&gpu));
    assert!(launch_with(&gpu, &kernel, DEFAULT_DYNAMIC_SHARED_BYTES + 1).is_err());
    assert_eq!(launches(&gpu), ops);
    assert_eq!(RUNS.load(Ordering::SeqCst), runs);
}

#[test]
fn an_eager_launch_over_the_limit_is_refused_in_a_recording_and_in_a_probe() {
    let (gpu, module) = module();
    let kernel = module.kernel("noop").unwrap();
    let (_pool, mut streams) = gpu.open_lanes(&["a"], StreamOptions::default()).unwrap();
    let launch = Launch {
        shared_bytes: DEFAULT_DYNAMIC_SHARED_BYTES + 1,
        ..Launch::linear(1, 1)
    };
    let body = |q: &mut euhedral_gpu::fake::FakeSequence<'_>| {
        q.submit(0, |e| {
            // SAFETY: the kernel takes no arguments.
            unsafe { e.launch(&kernel, launch, &Args::new()) }
        })
    };
    let recorded = gpu.record(&mut [&mut streams[0]], body);
    assert!(
        matches!(recorded, Err(GpuError::Invalid(_))),
        "{:?}",
        recorded.err()
    );
    let probed = gpu.fingerprint(1, body);
    assert!(matches!(probed, Err(GpuError::Invalid(_))), "{probed:?}");
}

#[test]
fn a_value_the_device_cannot_give_is_refused_at_the_lookup() {
    let (_gpu, module) = module();
    assert!(
        module
            .kernel_with(
                "noop",
                KernelOptions {
                    max_dynamic_shared_bytes: MAX_DYNAMIC_SHARED_BYTES
                }
            )
            .is_ok()
    );
    let refused = module.kernel_with(
        "noop",
        KernelOptions {
            max_dynamic_shared_bytes: u32::MAX,
        },
    );
    let Err(GpuError::Invalid(text)) = refused else {
        panic!("{refused:?}")
    };
    assert!(
        text.contains("noop") && text.contains(&u32::MAX.to_string()),
        "{text}"
    );
}
