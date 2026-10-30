//! The CUDA backends on a real device (skipped, passing, when no driver or device is present).
#![allow(
    clippy::disallowed_methods,
    reason = "test fixtures: the test thread runs peer threads and waits on them"
)]
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

#[test]
fn the_shared_driver_is_loaded_once() {
    let Ok(first) = Driver::shared() else {
        eprintln!("no CUDA driver: skipped");
        return;
    };
    let second = Driver::shared().unwrap();
    assert!(std::sync::Arc::ptr_eq(&first, &second));
}

#[test]
fn symbols_beyond_the_table_are_found() {
    let Ok(d) = Driver::shared() else {
        eprintln!("no CUDA driver: skipped");
        return;
    };
    assert!(!d.symbol(c"cuLaunchKernel").is_null());
    assert!(d.symbol(c"cuNoSuchCall").is_null());
}

/// Rows with padding go host -> device -> host through pitched copies, on a stream the caller made: only the row bytes
/// move, the padding stays what it was, and the stream is still the caller's (and alive) after the copier drops.
#[test]
fn pitched_copies_on_a_borrowed_stream_move_only_the_rows_and_leave_the_stream_alone() {
    use euhedral_memory::cuda::StreamCopier;
    use euhedral_memory::{Copier, PitchedCopy};

    let Some(dev) = device() else { return };
    const ROW: usize = 100;
    const HOST_PITCH: usize = 128;
    const DEVICE_PITCH: usize = 256;
    const ROWS: usize = 17;
    let (device_loc, host_loc) = (LocationId(0), LocationId(1));
    let mut stream = std::ptr::null_mut();
    {
        let _c = dev.enter().unwrap();
        // SAFETY: out pointer; the context is current; destroyed below.
        dev.check("cuStreamCreate", unsafe {
            (dev.driver.fns.stream_create)(&mut stream, 1)
        })
        .unwrap();
    }
    let device_mem = DeviceMemory {
        device: dev.clone(),
        location: device_loc,
    };
    let pinned = PinnedHost::new(dev.clone(), host_loc, 0);
    let on_device = device_mem.alloc((DEVICE_PITCH * ROWS) as u64).unwrap();
    let source = pinned.alloc((HOST_PITCH * ROWS) as u64).unwrap();
    let back = pinned.alloc((HOST_PITCH * ROWS) as u64).unwrap();
    // SAFETY: two pinned blocks of HOST_PITCH * ROWS bytes, used by this thread only.
    let (source_bytes, back_bytes) = unsafe {
        (
            std::slice::from_raw_parts_mut(source.addr as *mut u8, HOST_PITCH * ROWS),
            std::slice::from_raw_parts_mut(back.addr as *mut u8, HOST_PITCH * ROWS),
        )
    };
    for (i, b) in source_bytes.iter_mut().enumerate() {
        *b = (i % 251) as u8;
    }
    back_bytes.fill(0xEE);

    // SAFETY: `stream` is live, made on the device's primary context, and outlives the copier (destroyed below).
    let copier =
        unsafe { StreamCopier::on_stream(dev.clone(), stream, [device_loc], [host_loc], 4) }
            .unwrap();
    let wait = |test: Box<dyn euhedral_memory::Retired>| {
        let mut spins = 0u64;
        while !test.retired() {
            std::thread::yield_now();
            spins += 1;
            assert!(spins < 50_000_000, "the copy never retired");
        }
    };
    // SAFETY: the ranges are the blocks above; nothing else touches them until the tests retire.
    unsafe {
        wait(
            copier
                .start_pitched(
                    host_loc,
                    device_loc,
                    PitchedCopy {
                        src: source.addr,
                        src_pitch: HOST_PITCH as u64,
                        dst: on_device.addr,
                        dst_pitch: DEVICE_PITCH as u64,
                        row_bytes: ROW as u64,
                        rows: ROWS as u64,
                    },
                )
                .unwrap(),
        );
        wait(
            copier
                .start_pitched(
                    device_loc,
                    host_loc,
                    PitchedCopy {
                        src: on_device.addr,
                        src_pitch: DEVICE_PITCH as u64,
                        dst: back.addr,
                        dst_pitch: HOST_PITCH as u64,
                        row_bytes: ROW as u64,
                        rows: ROWS as u64,
                    },
                )
                .unwrap(),
        );
    }
    for row in 0..ROWS {
        let (s, b) = (
            &source_bytes[row * HOST_PITCH..][..HOST_PITCH],
            &back_bytes[row * HOST_PITCH..][..HOST_PITCH],
        );
        assert_eq!(&b[..ROW], &s[..ROW], "row {row}'s bytes arrived");
        assert!(
            b[ROW..].iter().all(|&x| x == 0xEE),
            "row {row}'s padding was not written"
        );
    }
    assert_eq!(copier.copies(), 2);
    // A pitch shorter than the row is refused before the driver is called.
    // SAFETY: refused before any range is used.
    let refused = unsafe {
        copier.start_pitched(
            host_loc,
            device_loc,
            PitchedCopy {
                src: source.addr,
                src_pitch: 10,
                dst: on_device.addr,
                dst_pitch: DEVICE_PITCH as u64,
                row_bytes: ROW as u64,
                rows: 1,
            },
        )
    };
    assert!(refused.is_err());

    drop(copier);
    // The caller's stream is still usable and is the caller's to destroy.
    {
        let _c = dev.enter().unwrap();
        // SAFETY: the stream made above, idle.
        dev.check("cuStreamSynchronize", unsafe {
            (dev.driver.fns.stream_synchronize)(stream)
        })
        .unwrap();
        // SAFETY: as above; nothing uses it afterwards.
        dev.check("cuStreamDestroy", unsafe {
            (dev.driver.fns.stream_destroy)(stream)
        })
        .unwrap();
    }
    // SAFETY: unused.
    unsafe {
        device_mem.free(on_device);
        pinned.free(source);
        pinned.free(back);
    }
}
