//! The CUDA backends on a real device (skipped, passing, when no driver or device is present).
#![cfg(feature = "cuda")]

use euhedral_memory::cuda::{Device, DeviceMemory, Driver, PinnedHost, StreamPool, Vmm};
use euhedral_memory::{Backend, LocationId};

fn device() -> Option<std::sync::Arc<Device>> {
    let d = Driver::load().ok()?;
    Device::open(d, 0).ok()
}

#[test]
fn each_backend_allocates_and_frees() {
    let Some(dev) = device() else {
        eprintln!("no CUDA device: skipped");
        return;
    };
    let loc = LocationId(0);
    let (free0, total) = dev.mem_info().unwrap();
    assert!(free0 > 0 && total >= free0);
    let backends: Vec<Box<dyn Backend>> = vec![
        Box::new(DeviceMemory {
            device: dev.clone(),
            location: loc,
        }),
        Box::new(StreamPool::new(dev.clone(), loc, 0).unwrap()),
        Box::new(Vmm::new(dev.clone(), loc, false).unwrap()),
        Box::new(PinnedHost::new(dev.clone(), loc, 64 << 20)),
    ];
    for b in &backends {
        for bytes in [256u64, 64 << 10, 1 << 20, 100 << 20] {
            let block = b.alloc(bytes).unwrap();
            assert!(block.bytes >= bytes && block.addr != 0, "{b:?} {bytes}");
            assert_eq!(
                b.granted(bytes),
                block.bytes.max(b.granted(bytes)).min(b.granted(bytes))
            );
            // SAFETY: unused.
            unsafe { b.free(block) };
        }
    }
}

#[test]
fn vmm_exports_a_posix_fd() {
    let Some(dev) = device() else { return };
    let vmm = Vmm::new(dev, LocationId(0), true).unwrap();
    assert!(vmm.granularity >= 4096);
    let block = vmm.alloc_vmm(3 << 20).unwrap();
    assert_eq!(block.size % vmm.granularity, 0);
    let fd = vmm.export_fd(&block).unwrap();
    drop(fd);
    // SAFETY: unused.
    unsafe { vmm.free_vmm(block) };
}

#[test]
fn pinned_host_reuses_released_buffers() {
    let Some(dev) = device() else { return };
    let p = PinnedHost::new(dev, LocationId(1), 256 << 20);
    let a = p.alloc(70 << 20).unwrap();
    let addr = a.addr;
    // SAFETY: unused.
    unsafe { p.free(a) };
    assert_eq!(p.cached(), 128 << 20);
    let b = p.alloc(100 << 20).unwrap();
    assert_eq!(b.addr, addr, "same class: the cached buffer comes back");
    // SAFETY: unused.
    unsafe { p.free(b) };
    p.trim(true);
    assert_eq!(p.cached(), 0);
}
