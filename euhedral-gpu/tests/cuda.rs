//! The CUDA backend on a real device. Every test skips, passing, when no driver or device is present.
#![cfg(feature = "cuda")]

use std::sync::Arc;

use euhedral_gpu::cuda::CudaGpu;
use euhedral_memory::cuda::Driver;

fn gpu() -> Option<Arc<CudaGpu>> {
    match CudaGpu::open(0) {
        Ok(gpu) => Some(gpu),
        Err(e) => {
            eprintln!("no CUDA device: skipped ({e})");
            None
        }
    }
}

#[test]
fn the_gpu_shares_memorys_driver_and_reports_its_architecture() {
    let Some(gpu) = gpu() else { return };
    let shared = Driver::shared().unwrap();
    assert!(Arc::ptr_eq(&gpu.device().driver, &shared));
    assert!(gpu.arch().0 >= 7, "{:?}", gpu.arch());
    assert!(gpu.driver_version() >= 12000);
}
