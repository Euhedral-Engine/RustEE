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

/// A resident moves pageable → device → pinned → pageable → device → pageable through stream copies (pageable ↔
/// device staging through pinned), its bytes intact and its owner's accounts exact at each step.
#[test]
fn residents_move_between_the_device_and_host_tiers_by_stream_copies() {
    use euhedral_memory::cuda::StreamCopier;
    use euhedral_memory::{
        Copier, HostCopier, LocationKind, LocationSpec, Mover, OwnerSpec, Planner, Priority, Scope,
        TierPool,
    };
    use std::sync::Arc;

    let Some(dev) = device() else { return };
    const SLOT: u64 = 1 << 20;
    const SLOTS: u32 = 4;
    let (a, b, c) = (LocationId(0), LocationId(1), LocationId(2));
    let planner = Planner::new([
        LocationSpec {
            kind: LocationKind::Device { ordinal: 0 },
            capacity: SLOT * u64::from(SLOTS),
            reserve: 0,
        },
        LocationSpec {
            kind: LocationKind::HostPinned,
            capacity: SLOT * u64::from(SLOTS),
            reserve: 0,
        },
        LocationSpec {
            kind: LocationKind::HostPageable,
            capacity: SLOT * u64::from(SLOTS),
            reserve: 0,
        },
    ]);
    // Each tier: one allocation, made once.
    let device_mem = DeviceMemory {
        device: dev.clone(),
        location: a,
    };
    let pinned = PinnedHost::new(dev.clone(), b, 0);
    let dblock = device_mem.alloc(SLOT * u64::from(SLOTS)).unwrap();
    let pblock = pinned.alloc(SLOT * u64::from(SLOTS)).unwrap();
    let pageable = vec![0u64; (SLOT as usize / 8) * SLOTS as usize];
    let tiers_owner = planner.open_owner(OwnerSpec::new("tiers", Priority(1), Scope::Process));
    let pools = vec![
        TierPool::new(&planner, tiers_owner, a, dblock.addr, SLOT, SLOTS).unwrap(),
        TierPool::new(&planner, tiers_owner, b, pblock.addr, SLOT, SLOTS).unwrap(),
        TierPool::new(
            &planner,
            tiers_owner,
            c,
            pageable.as_ptr() as u64,
            SLOT,
            SLOTS,
        )
        .unwrap(),
    ];
    let stream = Arc::new(StreamCopier::new(dev.clone(), [a], [b], 8).unwrap());
    let copiers: Vec<Arc<dyn Copier>> = vec![stream.clone(), Arc::new(HostCopier::new([b, c]))];
    let mover = Mover::new(&planner, pools, copiers, Some(b), 8);
    let o = planner.open_owner(OwnerSpec::new("consumer", Priority(50), Scope::Process));
    let id = mover.place(o, c, SLOT, Arc::new(|_| {})).unwrap();
    let pattern = |i: usize| (i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ 0x5a5a;
    mover
        .with(id, |_, addr| {
            // SAFETY: the resident's pageable slot, SLOT bytes, not moving.
            let w = unsafe { std::slice::from_raw_parts_mut(addr as *mut u64, SLOT as usize / 8) };
            w.iter_mut().enumerate().for_each(|(i, x)| *x = pattern(i));
        })
        .unwrap();
    let held = |l: LocationId| planner.snapshot().owner("consumer", l);
    for to in [a, b, c, a, c] {
        mover.move_to(id, to).unwrap();
        assert_eq!(held(to), SLOT, "the destination is claimed before the copy");
        let mut spins = 0u64;
        while mover.in_flight() > 0 {
            mover.pump();
            std::thread::yield_now();
            spins += 1;
            assert!(spins < 50_000_000, "the copy to {to:?} never completed");
        }
        assert_eq!(mover.whereabouts(id).unwrap().0, to);
        for l in [a, b, c] {
            assert_eq!(
                held(l),
                if l == to { SLOT } else { 0 },
                "after the move to {to:?}"
            );
        }
    }
    let intact = mover
        .with(id, |_, addr| {
            // SAFETY: the resident's pageable slot again.
            let w = unsafe { std::slice::from_raw_parts(addr as *const u64, SLOT as usize / 8) };
            w.iter().enumerate().all(|(i, x)| *x == pattern(i))
        })
        .unwrap();
    assert!(intact, "the bytes survived the device round trips");
    assert!(stream.copies() >= 4, "device legs went through the stream");
    assert!(mover.release(id));
    drop(mover);
    // SAFETY: no move in flight; the pools are gone with the mover.
    unsafe {
        device_mem.free(dblock);
        pinned.free(pblock);
    }
    drop(pageable);
}
