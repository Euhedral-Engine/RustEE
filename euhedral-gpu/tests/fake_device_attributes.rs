//! The device's limits a launch planner sizes work by: the fake reports settable values.
#![cfg(feature = "fake")]

use euhedral_gpu::fake::{DEFAULT_ATTRIBUTES, FakeGpu, FakeModule};
use euhedral_gpu::{DeviceAttributes, Gpu};

#[test]
fn a_new_fake_reports_its_defaults() {
    let gpu = FakeGpu::new();
    assert_eq!(gpu.attributes(), DEFAULT_ATTRIBUTES);
}

#[test]
fn the_fake_reports_what_was_set() {
    let gpu = FakeGpu::new();
    let set = DeviceAttributes {
        multiprocessors: 132,
        max_shared_memory_per_block_optin: 227 * 1024,
    };
    gpu.set_attributes(set);
    assert_eq!(gpu.attributes(), set);
    assert_ne!(set, DEFAULT_ATTRIBUTES);
}

#[test]
fn a_kernel_opt_in_above_the_devices_limit_is_refused_and_one_at_it_succeeds() {
    use euhedral_gpu::{CompileOptions, GpuError, KernelOptions, Module};

    let gpu = FakeGpu::new();
    gpu.define("k", |_, _, _| Ok(()));
    gpu.set_attributes(DeviceAttributes {
        multiprocessors: 4,
        max_shared_memory_per_block_optin: 64 * 1024,
    });
    let module = FakeModule::from_source(&gpu, "", &CompileOptions::default()).unwrap();
    let at = KernelOptions {
        max_dynamic_shared_bytes: 64 * 1024,
    };
    assert!(module.kernel_with("k", at).is_ok());
    let above = KernelOptions {
        max_dynamic_shared_bytes: 64 * 1024 + 1,
    };
    assert!(matches!(
        module.kernel_with("k", above),
        Err(GpuError::Invalid(_))
    ));
}
