//! The device's limits a launch planner sizes work by: the fake reports settable values.
#![cfg(feature = "fake")]

use euhedral_gpu::fake::{DEFAULT_ATTRIBUTES, FakeGpu};
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
