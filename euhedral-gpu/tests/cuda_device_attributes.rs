//! The device's limits read from a real card: a multiprocessor count and a shared-memory opt-in limit that a card
//! has. Skips, passing, with no driver or device.
#![cfg(feature = "cuda")]

#[path = "common/skip.rs"]
mod skip;

use euhedral_gpu::Gpu;
use euhedral_gpu::cuda::CudaGpu;

#[test]
fn a_card_reports_its_multiprocessors_and_shared_memory_optin() {
    let gpu = match CudaGpu::open(0) {
        Ok(gpu) => gpu,
        Err(e) => {
            skip::skip(format_args!("no CUDA device: skipped ({e})"));
            return;
        }
    };
    let attributes = gpu.attributes();
    eprintln!("device attributes: {attributes:?}");
    assert!(attributes.multiprocessors >= 1);
    // Every card that opts in at all allows at least the 48 KiB a block gets without opting in.
    assert!(attributes.max_shared_memory_per_block_optin >= 48 * 1024);
}
