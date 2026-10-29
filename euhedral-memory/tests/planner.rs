//! The planner's accounting, give-back, admission and leak check, without a device.
#![allow(
    clippy::disallowed_types,
    reason = "test fixtures: the test thread locks its records"
)]

use euhedral_memory::{
    ClaimResult, LocationKind, LocationSpec, OwnerSpec, Placed, PlanItem, PlanRequest, Planner,
    Priority, Reaper, Scope,
};

const MIB: u64 = 1 << 20;
const GIB: u64 = 1 << 30;

fn two_locations() -> std::sync::Arc<Planner> {
    Planner::new([
        LocationSpec {
            kind: LocationKind::Device { ordinal: 0 },
            capacity: 16 * GIB,
            reserve: 700 * MIB,
        },
        LocationSpec {
            kind: LocationKind::HostPinned,
            capacity: 8 * GIB,
            reserve: 0,
        },
    ])
}

#[test]
fn claims_account_and_drop_gives_back() {
    let p = two_locations();
    let dev = p.location(LocationKind::Device { ordinal: 0 }).unwrap();
    let o = p.open_owner(OwnerSpec::new("slots", Priority(10), Scope::Epoch(1)));
    let free0 = p.free(dev);
    assert_eq!(free0, 16 * GIB - 700 * MIB);
    let c = p.claim(o, dev, 3 * GIB, "slot block").unwrap();
    assert_eq!(p.free(dev), free0 - 3 * GIB);
    assert_eq!(p.snapshot().owner("slots", dev), 3 * GIB);
    drop(c);
    assert_eq!(p.free(dev), free0);
}

#[test]
fn a_claim_past_the_reserve_is_refused() {
    let p = two_locations();
    let dev = p.location(LocationKind::Device { ordinal: 0 }).unwrap();
    let o = p.open_owner(OwnerSpec::new("big", Priority(1), Scope::Process));
    let e = p.claim(o, dev, 16 * GIB - 600 * MIB, "x").unwrap_err();
    assert_eq!(e.asked, 16 * GIB - 600 * MIB);
    assert_eq!(e.free, 16 * GIB - 700 * MIB);
}

#[test]
fn elastic_takes_what_is_left_and_gives_back_on_order() {
    let p = two_locations();
    let dev = p.location(LocationKind::Device { ordinal: 0 }).unwrap();
    let buffer = p.open_owner(OwnerSpec::new(
        "spillable buffer",
        Priority(1),
        Scope::Epoch(1),
    ));
    let plan = p.open_owner(OwnerSpec::new("plan", Priority(100), Scope::Plan(1)));
    let _firm = p.claim(plan, dev, 10 * GIB, "plan").unwrap();
    let mut e = p
        .elastic(buffer, dev, 512 * MIB, 20 * GIB, "slots")
        .unwrap();
    assert_eq!(e.bytes(), 16 * GIB - 700 * MIB - 10 * GIB);
    assert_eq!(p.free(dev), 0);
    // A higher-priority owner needs 2 GiB: an order goes to the buffer (as a frame of the give-back source).
    let need = p.open_owner(OwnerSpec::new("next plan", Priority(100), Scope::Plan(2)));
    let orders = match p.claim_or_reclaim(need, dev, 2 * GIB, "build") {
        ClaimResult::Reclaiming(o) => o,
        other => panic!("{other:?}"),
    };
    assert_eq!(orders.len(), 1);
    assert_eq!(orders[0].bytes, 2 * GIB);
    assert_eq!(e.ordered(), 2 * GIB);
    // It may not grow while an order is open.
    assert_eq!(e.grow_to(30 * GIB), e.bytes());
    // The buffer frees its slots, then the claim is granted.
    let after = e.bytes() - 2 * GIB;
    e.shrink_to(after);
    assert_eq!(e.ordered(), 0);
    assert!(matches!(
        p.claim_or_reclaim(need, dev, 2 * GIB, "build"),
        ClaimResult::Granted(_)
    ));
}

#[test]
fn a_lower_priority_owner_cannot_reclaim_from_a_higher_one() {
    let p = two_locations();
    let dev = p.location(LocationKind::Device { ordinal: 0 }).unwrap();
    let hi = p.open_owner(OwnerSpec::new("cache a", Priority(90), Scope::Process));
    let lo = p.open_owner(OwnerSpec::new("consumer b", Priority(20), Scope::Process));
    let _e = p.elastic(hi, dev, 0, 100 * GIB, "cache").unwrap();
    assert!(matches!(
        p.claim_or_reclaim(lo, dev, MIB, "x"),
        ClaimResult::Refused(_)
    ));
}

#[test]
fn ending_an_owner_reports_what_its_tree_still_holds() {
    let p = two_locations();
    let dev = p.location(LocationKind::Device { ordinal: 0 }).unwrap();
    let pin = p.location(LocationKind::HostPinned).unwrap();
    let epoch = p.open_owner(OwnerSpec::new("session 3", Priority(50), Scope::Epoch(3)));
    let buffer =
        p.open_owner(OwnerSpec::new("buffer", Priority(1), Scope::Epoch(3)).child_of(epoch));
    let a = p.claim(epoch, dev, GIB, "slot block").unwrap();
    let b = p.claim(buffer, pin, 100 * MIB, "host slot").unwrap();
    drop(a);
    let report = p.end_owner(epoch);
    assert_eq!(report.leaks.len(), 1);
    assert_eq!(report.leaks[0].owner, "buffer");
    assert_eq!(report.leaks[0].bytes, 100 * MIB);
    drop(b);
}

#[test]
fn baseline_comes_back_after_switches() {
    let p = two_locations();
    let dev = p.location(LocationKind::Device { ordinal: 0 }).unwrap();
    let ctx = p.open_owner(OwnerSpec::new(
        "context",
        Priority(u16::MAX),
        Scope::Process,
    ));
    let _c = p.claim(ctx, dev, 400 * MIB, "context").unwrap();
    let base = p.snapshot();
    for session_id in 0..5u64 {
        let o = p.open_owner(OwnerSpec::new(
            "session",
            Priority(50),
            Scope::Epoch(session_id),
        ));
        let _x = p
            .claim(o, dev, (3 + session_id) * GIB, "working set")
            .unwrap();
        drop(_x);
        assert!(p.end_scope(Scope::Epoch(session_id)).is_clean());
    }
    assert!(p.check_baseline(&base).is_clean());
    // A leak: one claim kept past its owner's end.
    let o = p.open_owner(OwnerSpec::new("leaky", Priority(50), Scope::Epoch(9)));
    let kept = p.claim(o, dev, GIB, "forgotten").unwrap();
    let _ = p.end_scope(Scope::Epoch(9));
    let r = p.check_baseline(&base);
    assert_eq!(r.bytes(), GIB);
    drop(kept);
    assert!(p.check_baseline(&base).is_clean());
}

fn plan(
    p: &std::sync::Arc<Planner>,
    owner_prio: u16,
) -> (
    euhedral_memory::OwnerId,
    euhedral_memory::LocationId,
    euhedral_memory::LocationId,
) {
    let dev = p.location(LocationKind::Device { ordinal: 0 }).unwrap();
    let pin = p.location(LocationKind::HostPinned).unwrap();
    (
        p.open_owner(OwnerSpec::new("plan", Priority(owner_prio), Scope::Plan(1))),
        dev,
        pin,
    )
}

/// Two required consumers, six optional ones by priority, and a spillable buffer.
fn items(dev: euhedral_memory::LocationId, pin: euhedral_memory::LocationId) -> Vec<PlanItem> {
    vec![
        PlanItem::required("slots", dev, 3 * GIB),
        PlanItem::required("staging", dev, GIB / 2),
        PlanItem::optional("consumer a", Priority(80), dev, 600 * MIB),
        PlanItem::optional("consumer b", Priority(70), dev, 2 * GIB),
        PlanItem::optional("consumer c", Priority(60), dev, 300 * MIB),
        PlanItem::optional("consumer d", Priority(55), dev, GIB),
        PlanItem::optional("consumer e", Priority(50), dev, 100 * MIB),
        PlanItem::optional("consumer f", Priority(20), dev, 2 * GIB),
        // A spillable buffer: on the device, or most of its slots on pinned host memory.
        PlanItem::required("spillable buffer", dev, 4 * GIB)
            .spills_to(vec![(dev, 800 * MIB), (pin, 4 * GIB)]),
    ]
}

#[test]
fn admission_places_everything_when_it_fits() {
    let p = two_locations();
    let (o, dev, pin) = plan(&p, 100);
    let a = p
        .admit(&PlanRequest {
            owner: o,
            items: items(dev, pin),
        })
        .unwrap();
    assert!(a.placed.iter().all(|x| *x == Placed::Preferred));
    let r = a.reservation.unwrap();
    assert_eq!(r.left(dev), a.bytes[0]);
}

#[test]
fn admission_spills_before_cutting_then_cuts_by_priority() {
    let p = two_locations();
    let (o, dev, pin) = plan(&p, 100);
    // Something else holds 4 GiB firmly: 11.3 GiB left; the plan wants about 13.5 GiB.
    let other = p.open_owner(OwnerSpec::new(
        "other process",
        Priority(200),
        Scope::Lease(1),
    ));
    let _held = p.claim(other, dev, 4 * GIB, "lease").unwrap();
    let a = p
        .admit(&PlanRequest {
            owner: o,
            items: items(dev, pin),
        })
        .unwrap();
    assert_eq!(
        a.placed[8],
        Placed::Spilled,
        "the buffer spills to pinned host memory first"
    );
    assert!(
        a.placed[..8].iter().all(|x| *x == Placed::Preferred),
        "{:?}",
        a.placed
    );
    // Tighter: 7 GiB held. Spilling is not enough; consumer f goes, then consumer e, ...
    drop(a);
    drop(_held);
    let _held = p.claim(other, dev, 7 * GIB, "lease").unwrap();
    let a = p
        .admit(&PlanRequest {
            owner: o,
            items: items(dev, pin),
        })
        .unwrap();
    assert_eq!(a.placed[8], Placed::Spilled);
    assert_eq!(
        a.placed[7],
        Placed::Dropped,
        "consumer f is cut before anything above it"
    );
    assert_eq!(
        a.placed[6],
        Placed::Preferred,
        "consumer e still fits once consumer f is cut"
    );
    assert_eq!(a.placed[2], Placed::Preferred, "consumer a is kept");
    let mut r = a.reservation.unwrap();
    // The plan's allocations take their claims from the reservation; an unplanned one gets none.
    let slots = p.open_owner(OwnerSpec::new("slots", Priority(100), Scope::Plan(1)).child_of(o));
    let c = r.take(slots, dev, 3 * GIB, "slot block").unwrap();
    assert_eq!(c.bytes(), 3 * GIB);
    assert!(r.take(slots, dev, 100 * GIB, "unplanned").is_none());
}

#[test]
fn admission_refuses_when_the_required_part_cannot_fit() {
    let p = two_locations();
    let (o, dev, pin) = plan(&p, 100);
    let other = p.open_owner(OwnerSpec::new(
        "other process",
        Priority(200),
        Scope::Lease(1),
    ));
    let _held = p.claim(other, dev, 13 * GIB, "lease").unwrap();
    let e = p
        .admit(&PlanRequest {
            owner: o,
            items: items(dev, pin),
        })
        .unwrap_err();
    assert_eq!(e.location, dev);
}

#[test]
fn admission_orders_give_back_from_lower_priority_elastic_owners() {
    let p = two_locations();
    let (o, dev, pin) = plan(&p, 100);
    let buffer = p.open_owner(OwnerSpec::new("old buffer", Priority(1), Scope::Epoch(1)));
    let _e = p.elastic(buffer, dev, 0, 100 * GIB, "slots").unwrap();
    let a = p
        .admit(&PlanRequest {
            owner: o,
            items: items(dev, pin),
        })
        .unwrap();
    assert!(a.reservation.is_none());
    assert!(!a.orders.is_empty());
    assert_eq!(a.orders[0].owner, buffer);
}

#[test]
fn reaper_frees_only_what_retired() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    struct Freed(Arc<AtomicUsize>);
    impl Drop for Freed {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }
    let freed = Arc::new(AtomicUsize::new(0));
    let done = Arc::new(AtomicBool::new(false));
    let r = Reaper::new();
    let d = done.clone();
    r.push(move || d.load(Ordering::SeqCst), Freed(freed.clone()));
    r.push(|| true, Freed(freed.clone()));
    assert_eq!(r.reap(), 1);
    assert_eq!(freed.load(Ordering::SeqCst), 1);
    done.store(true, Ordering::SeqCst);
    assert_eq!(r.reap(), 1);
    assert_eq!(r.pending(), 0);
}

#[test]
fn give_back_orders_arrive_as_frames_of_the_lattice_source() {
    use euhedral_core::generics::LatticeSource;
    use std::sync::{Arc, Mutex};
    let p = two_locations();
    let dev = p.location(LocationKind::Device { ordinal: 0 }).unwrap();
    let got: Arc<Mutex<Vec<euhedral_memory::GiveBack>>> = Arc::default();
    let g = got.clone();
    let buffer = p.open_owner(
        OwnerSpec::new("buffer", Priority(1), Scope::Epoch(1))
            .on_give_back(Arc::new(move |o| g.lock().unwrap().push(o))),
    );
    let mut e = p.elastic(buffer, dev, 0, 100 * GIB, "slots").unwrap();
    let plan = p.open_owner(OwnerSpec::new("plan", Priority(100), Scope::Plan(1)));
    assert!(matches!(
        p.claim_or_reclaim(plan, dev, GIB, "x"),
        ClaimResult::Reclaiming(_)
    ));
    assert_eq!(p.orders_pending(), 1);
    // A lattice worker's poll: the order comes out as a frame, and the frame runs the buffer's handler.
    let source = p.give_backs();
    let n = source.pull(
        &mut |mut f| {
            f.execute().unwrap();
            f.do_finally();
        },
        &mut |_| false,
        8,
    );
    assert_eq!(n, 1);
    let orders = got.lock().unwrap().clone();
    assert_eq!(orders.len(), 1);
    assert_eq!(orders[0].bytes, GIB);
    // The buffer frees: the order is done and the claim is granted.
    let b = e.bytes();
    e.shrink_to(b - GIB);
    assert_eq!(e.ordered(), 0);
    assert!(matches!(
        p.claim_or_reclaim(plan, dev, GIB, "x"),
        ClaimResult::Granted(_)
    ));
}

#[test]
fn owner_slots_are_recycled_and_claims_race_free() {
    use std::sync::Arc;
    let p = Planner::with_owners(
        [LocationSpec {
            kind: LocationKind::Device { ordinal: 0 },
            capacity: 64 * GIB,
            reserve: 0,
        }],
        16,
    );
    let dev = euhedral_memory::LocationId(0);
    // Far more owners than slots over time: each ended, empty owner gives its slot back.
    for i in 0..1000u64 {
        let o = p.open_owner(OwnerSpec::new("plan", Priority(50), Scope::Plan(i)));
        let c = p.claim(o, dev, MIB, "x").unwrap();
        assert!(p.end_owner(o).leaks.len() == 1); // still held at the end: reported
        drop(c); // the last claim's drop recycles the slot
    }
    assert_eq!(p.snapshot().used(dev), 0);
    // Claims from many threads at once: the budget is never exceeded and comes back to zero.
    let o = p.open_owner(OwnerSpec::new("many", Priority(50), Scope::Process));
    let threads: Vec<_> = (0..8)
        .map(|_| {
            let p = Arc::clone(&p);
            std::thread::spawn(move || {
                let mut held = Vec::new();
                for _ in 0..2000 {
                    if let Ok(c) = p.claim(o, dev, 64 * MIB, "x") {
                        held.push(c);
                    }
                    if held.len() > 64 {
                        held.clear();
                    }
                    assert!(p.snapshot().used(dev) <= 64 * GIB);
                }
            })
        })
        .collect();
    for t in threads {
        t.join().unwrap();
    }
    assert_eq!(p.snapshot().used(dev), 0);
    assert!(p.end_owner(o).is_clean());
}

#[test]
fn arenas_hand_out_and_take_back_without_backend_calls() {
    use euhedral_memory::{ScratchArena, SlotArena};
    let p = two_locations();
    let dev = p.location(LocationKind::Device { ordinal: 0 }).unwrap();
    let plan = p.open_owner(OwnerSpec::new("plan", Priority(100), Scope::Plan(1)));
    // One planned allocation (one backend call) for 12 slots and one for 64 MiB of scratch.
    let slot_bytes = 100 * MIB;
    p.calls().alloc(12 * slot_bytes);
    let slots = SlotArena::new(
        0x1000_0000,
        slot_bytes,
        12,
        p.claim(plan, dev, 12 * slot_bytes, "slot pool").unwrap(),
    );
    p.calls().alloc(64 * MIB);
    let scratch = ScratchArena::new(
        0x9000_0000,
        64 * MIB,
        p.claim(plan, dev, 64 * MIB, "scratch").unwrap(),
    );
    let before = p.calls().get();
    // A session of 10,000 units of work: each leases slots and scratch and returns them.
    for _ in 0..10_000 {
        let a = slots.lease().unwrap();
        let b = slots.lease().unwrap();
        assert_ne!(a.address(), b.address());
        let s1 = scratch.alloc(256).unwrap();
        let s2 = scratch.alloc(64 << 10).unwrap();
        let s3 = scratch.alloc(1 << 20).unwrap();
        assert!(s1.offset() + s1.bytes() <= scratch.bytes());
        drop((a, b, s1, s2, s3));
    }
    assert_eq!(
        p.calls().get(),
        before,
        "no allocation or free in the steady state"
    );
    assert_eq!(slots.leased(), 0);
    assert_eq!(scratch.in_use(), 0);
    // Back-pressure, never an allocation: the 13th slot is refused.
    let held: Vec<_> = (0..12).map(|_| slots.lease().unwrap()).collect();
    assert!(slots.lease().is_none());
    drop(held);
    // The session ends: the arenas drop (the real frees) and the plan's owner is clean.
    drop((slots, scratch));
    assert!(p.end_owner(plan).is_clean());
}

#[test]
fn accounting_existing_memory_always_succeeds_and_admission_sees_it() {
    let p = two_locations();
    let dev = p.location(LocationKind::Device { ordinal: 0 }).unwrap();
    let o = p.open_owner(OwnerSpec::new("existing", Priority(100), Scope::Epoch(1)));
    let a = p.account(o, dev, 20 * GIB).unwrap();
    assert_eq!(p.free(dev), 0);
    assert!(p.claim(o, dev, 1, "x").is_err());
    drop(a);
    assert_eq!(p.free(dev), 16 * GIB - 700 * MIB);
    p.end_owner(o);
    assert!(
        p.account(o, dev, 1).is_none(),
        "an ended owner accounts nothing"
    );
}

/// A device, pinned host memory and a window onto a file.
fn three_tiers() -> std::sync::Arc<Planner> {
    Planner::new([
        LocationSpec {
            kind: LocationKind::Device { ordinal: 0 },
            capacity: 16 * GIB,
            reserve: 0,
        },
        LocationSpec {
            kind: LocationKind::HostPinned,
            capacity: 4 * GIB,
            reserve: 0,
        },
        LocationSpec {
            kind: LocationKind::Other("file window"),
            capacity: 2 * GIB,
            reserve: 0,
        },
    ])
}

#[test]
fn a_tiered_consumer_fills_the_device_then_the_slower_tiers_in_order() {
    let p = three_tiers();
    let dev = p.location(LocationKind::Device { ordinal: 0 }).unwrap();
    let pin = p.location(LocationKind::HostPinned).unwrap();
    let file = p.location(LocationKind::Other("file window")).unwrap();
    let o = p.open_owner(OwnerSpec::new("plan", Priority(100), Scope::Plan(1)));
    // The working set takes 12 GiB of the device; 100 units of 100 MiB, 8 required on the device.
    let items = vec![
        PlanItem::required("working set", dev, 12 * GIB),
        PlanItem::tiered(
            "buffer",
            Priority(10),
            true,
            100 * MIB,
            100,
            8,
            vec![dev, pin, file],
        ),
    ];
    let a = p.admit(&PlanRequest { owner: o, items }).unwrap();
    let t = &a.tiers[1];
    // 4 GiB left on the device: 40 units; 4 GiB pinned: 40; the file window takes the last 20.
    assert_eq!(t, &vec![(dev, 40), (pin, 40), (file, 20)], "{t:?}");
    assert!(a.tiers[0].is_empty());
    assert_eq!(a.bytes[0], 12 * GIB + 40 * 100 * MIB);
    assert_eq!(a.bytes[1], 40 * 100 * MIB);
    assert_eq!(a.bytes[2], 20 * 100 * MIB);
    let r = a.reservation.expect("reserved");
    assert_eq!(r.left(pin), 40 * 100 * MIB);
}

#[test]
fn tiered_consumers_take_the_fast_tier_by_priority_and_what_fits_nowhere_is_left_out() {
    let p = three_tiers();
    let dev = p.location(LocationKind::Device { ordinal: 0 }).unwrap();
    let pin = p.location(LocationKind::HostPinned).unwrap();
    let o = p.open_owner(OwnerSpec::new("plan", Priority(100), Scope::Plan(1)));
    let items = vec![
        PlanItem::required("working set", dev, 14 * GIB),
        // Listed first but lower: the higher one takes the device's room first.
        PlanItem::tiered("low", Priority(10), false, GIB, 6, 0, vec![dev, pin]),
        PlanItem::tiered("high", Priority(50), false, GIB, 3, 1, vec![dev, pin]),
    ];
    let a = p.admit(&PlanRequest { owner: o, items }).unwrap();
    // The device has 2 GiB after the working set: "high" 1 required + 1 more; then pinned 1. "low": the 4 GiB pinned
    // less "high"'s 1 GiB: 3 units; the other 3 fit nowhere.
    assert_eq!(a.tiers[2], vec![(dev, 2), (pin, 1)]);
    assert_eq!(a.tiers[1], vec![(dev, 0), (pin, 3)]);
    assert_eq!(a.bytes[1], 4 * GIB);
}

#[test]
fn a_tiered_consumer_whose_required_units_do_not_fit_is_refused_or_dropped() {
    let p = three_tiers();
    let dev = p.location(LocationKind::Device { ordinal: 0 }).unwrap();
    let pin = p.location(LocationKind::HostPinned).unwrap();
    let o = p.open_owner(OwnerSpec::new("plan", Priority(100), Scope::Plan(1)));
    let need = |required| {
        vec![
            PlanItem::required("working set", dev, 15 * GIB),
            PlanItem::tiered("buffer", Priority(10), required, GIB, 4, 2, vec![dev, pin]),
        ]
    };
    assert!(
        p.admit(&PlanRequest {
            owner: o,
            items: need(true)
        })
        .is_err(),
        "2 GiB required, 1 free"
    );
    let a = p
        .admit(&PlanRequest {
            owner: o,
            items: need(false),
        })
        .unwrap();
    assert_eq!(a.placed[1], Placed::Dropped);
    assert!(a.tiers[1].is_empty());
}
