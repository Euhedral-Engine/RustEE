//! Moves between tiers, without a device: three host-memory locations stand for a fast, a middle and a slow tier,
//! joined by a copier whose copies retire a few checks after they start (as an asynchronous copy does).
#![allow(
    clippy::disallowed_types,
    reason = "test fixtures: the test thread locks its records"
)]

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use euhedral_core::frames::{Failure, Frame};
use euhedral_core::generics::{LatticeReceiver, LatticeSource, LatticeTerminal};
use euhedral_memory::{
    ClaimResult, Copier, CopyError, LocationId, LocationKind, LocationSpec, MoveDone, MoveError,
    MoveHandler, Mover, OwnerId, OwnerSpec, Planner, Priority, Retired, Scope, TierPool,
};

const KIB: u64 = 1 << 10;
const SLOT: u64 = 64 * KIB;

/// Copies at once but reports completion only after `delay` checks; joins the listed pairs only.
#[derive(Debug)]
struct SlowCopier {
    pairs: Vec<(LocationId, LocationId)>,
    delay: u32,
    copies: AtomicU64,
}

struct Countdown(AtomicU32);

impl Retired for Countdown {
    fn retired(&self) -> bool {
        self.0
            .try_update(Ordering::AcqRel, Ordering::Acquire, |n| n.checked_sub(1))
            .is_err()
    }
}

impl Copier for SlowCopier {
    fn joins(&self, from: LocationId, to: LocationId) -> bool {
        self.pairs.contains(&(from, to))
    }

    unsafe fn start(
        &self,
        _from: LocationId,
        src: u64,
        _to: LocationId,
        dst: u64,
        bytes: u64,
    ) -> Result<Box<dyn Retired>, CopyError> {
        // SAFETY: the mover's contract: two leased slots of host memory, `bytes` each.
        unsafe { std::ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, bytes as usize) };
        self.copies.fetch_add(1, Ordering::Relaxed);
        Ok(Box::new(Countdown(AtomicU32::new(self.delay))))
    }
}

struct Rig {
    planner: Arc<Planner>,
    locations: [LocationId; 3],
    pools: Vec<Arc<TierPool>>,
    mover: Arc<Mover>,
    pool_owner: OwnerId,
    _memory: Vec<Vec<u64>>,
}

/// Three tiers of `slots` slots each; `fast` ↔ `slow` has no copier (moves between them stage through the middle).
fn rig(slots: [u32; 3], residents: u32) -> Rig {
    // Each location's budget is its tier's pool: the tiers hold all of it.
    let cap = |n: u32| u64::from(n) * SLOT;
    let planner = Planner::new([
        LocationSpec {
            kind: LocationKind::Device { ordinal: 0 },
            capacity: cap(slots[0]),
            reserve: 0,
        },
        LocationSpec {
            kind: LocationKind::HostPinned,
            capacity: cap(slots[1]),
            reserve: 0,
        },
        LocationSpec {
            kind: LocationKind::HostPageable,
            capacity: cap(slots[2]),
            reserve: 0,
        },
    ]);
    let locations = [LocationId(0), LocationId(1), LocationId(2)];
    let pool_owner = planner.open_owner(OwnerSpec::new("tiers", Priority(1), Scope::Process));
    let mut memory = Vec::new();
    let mut pools = Vec::new();
    for (l, n) in locations.iter().zip(slots) {
        let block = vec![0u64; (SLOT as usize / 8) * n as usize];
        let base = block.as_ptr() as u64;
        memory.push(block);
        pools.push(TierPool::new(&planner, pool_owner, *l, base, SLOT, n).unwrap());
    }
    let [a, b, c] = locations;
    let copier: Arc<dyn Copier> = Arc::new(SlowCopier {
        pairs: vec![(a, b), (b, a), (b, c), (c, b)],
        delay: 2,
        copies: AtomicU64::new(0),
    });
    let mover = Mover::new(&planner, pools.clone(), vec![copier], Some(b), residents);
    Rig {
        planner,
        locations,
        pools,
        mover,
        pool_owner,
        _memory: memory,
    }
}

/// Writes `seed`'s pattern into the resident's slot (no move meanwhile).
fn put(m: &Mover, id: euhedral_memory::ResidentId, seed: u64) -> bool {
    m.with(id, |_, addr| fill(addr, seed)).is_some()
}

/// Whether the resident's slot holds `seed`'s pattern, and where it is.
fn check(m: &Mover, id: euhedral_memory::ResidentId, seed: u64) -> Option<(LocationId, bool)> {
    m.with(id, |l, addr| (l, holds(addr, seed)))
}

fn fill(addr: u64, seed: u64) {
    // SAFETY: a slot of the rig's host memory, SLOT bytes.
    let words = unsafe { std::slice::from_raw_parts_mut(addr as *mut u64, (SLOT / 8) as usize) };
    for (i, w) in words.iter_mut().enumerate() {
        *w = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ i as u64;
    }
}

fn holds(addr: u64, seed: u64) -> bool {
    // SAFETY: as in `fill`.
    let words = unsafe { std::slice::from_raw_parts(addr as *const u64, (SLOT / 8) as usize) };
    words
        .iter()
        .enumerate()
        .all(|(i, w)| *w == seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ i as u64)
}

fn held(p: &Planner, name: &str, l: LocationId) -> u64 {
    p.snapshot().owner(name, l)
}

fn quiet() -> MoveHandler {
    Arc::new(|_| {})
}

fn settle(m: &Mover) {
    for _ in 0..100 {
        m.pump();
        if m.in_flight() == 0 {
            return;
        }
    }
    panic!("moves did not finish");
}

#[test]
fn moves_between_every_pair_keep_the_bytes_and_the_accounts() {
    let r = rig([2, 2, 2], 8);
    let o = r
        .planner
        .open_owner(OwnerSpec::new("consumer", Priority(50), Scope::Process));
    let [a, b, c] = r.locations;
    let id = r.mover.place(o, a, SLOT, quiet()).unwrap();
    assert_eq!(r.mover.whereabouts(id).unwrap().0, a);
    assert!(put(&r.mover, id, 7));
    assert_eq!(held(&r.planner, "consumer", a), SLOT);
    for to in [b, c, a, c, b, a] {
        let from = r.mover.whereabouts(id).unwrap().0;
        r.mover.move_to(id, to).unwrap();
        // In flight: the destination is the owner's already, the source still is.
        assert_eq!(
            held(&r.planner, "consumer", to),
            SLOT,
            "{from:?} -> {to:?}: the destination is claimed first"
        );
        assert_eq!(
            held(&r.planner, "consumer", from),
            SLOT,
            "{from:?} -> {to:?}: the source is kept while copying"
        );
        assert!(
            r.mover.whereabouts(id).is_none(),
            "moving: no address to read"
        );
        settle(&r.mover);
        let (now, intact) = check(&r.mover, id, 7).unwrap();
        assert_eq!(now, to);
        assert!(intact, "{from:?} -> {to:?}: the bytes came along");
        for l in r.locations {
            assert_eq!(
                held(&r.planner, "consumer", l),
                if l == to { SLOT } else { 0 }
            );
        }
    }
    // The location totals never moved: the pools' slots changed hands only.
    for (pool, l) in r.pools.iter().zip(r.locations) {
        assert_eq!(
            r.planner.snapshot().used(l),
            u64::from(pool.capacity()) * SLOT
        );
    }
    assert!(r.mover.release(id));
    for pool in &r.pools {
        assert_eq!(pool.free(), pool.capacity());
    }
    assert_eq!(held(&r.planner, "consumer", a), 0);
}

#[test]
fn a_move_released_midway_gives_both_slots_back() {
    let r = rig([2, 2, 2], 8);
    let o = r
        .planner
        .open_owner(OwnerSpec::new("consumer", Priority(50), Scope::Process));
    let [a, b, _] = r.locations;
    let seen: Arc<std::sync::Mutex<Vec<MoveDone>>> = Arc::default();
    let s = seen.clone();
    let id = r
        .mover
        .place(o, a, SLOT, Arc::new(move |d| s.lock().unwrap().push(d)))
        .unwrap();
    r.mover.move_to(id, b).unwrap();
    assert!(r.mover.release(id), "released while the copy runs");
    settle(&r.mover);
    // The completion, as a frame of the mover's source.
    let source = r.mover.moves();
    let n = source.pull(
        &mut |mut f| {
            f.execute().unwrap();
            f.do_finally();
        },
        &mut |_| false,
        8,
    );
    assert_eq!(n, 1);
    let done = seen.lock().unwrap().clone();
    assert!(done[0].cancelled && done[0].from == a && done[0].to == b);
    for l in r.locations {
        assert_eq!(held(&r.planner, "consumer", l), 0);
    }
    for pool in &r.pools {
        assert_eq!(pool.free(), pool.capacity());
    }
    assert!(r.mover.whereabouts(id).is_none());
    assert_eq!(r.mover.move_to(id, a), Err(MoveError::NotResident));
}

#[test]
fn a_higher_priority_resident_evicts_a_lower_one_and_spilled_ones_come_back() {
    let r = rig([2, 4, 4], 8);
    let [a, b, _] = r.locations;
    let lo = r
        .planner
        .open_owner(OwnerSpec::new("low", Priority(10), Scope::Process));
    let hi = r
        .planner
        .open_owner(OwnerSpec::new("high", Priority(90), Scope::Process));
    let l1 = r.mover.place(lo, a, SLOT, quiet()).unwrap();
    let l2 = r.mover.place(lo, a, SLOT, quiet()).unwrap();
    assert!(put(&r.mover, l1, 1) && put(&r.mover, l2, 2));
    // The fast tier is full: the high-priority resident starts one tier down and a low one is spilled for it.
    let h = r.mover.place(hi, a, SLOT, quiet()).unwrap();
    assert_eq!(r.mover.whereabouts(h).unwrap().0, b);
    assert!(put(&r.mover, h, 3));
    assert_eq!(r.mover.in_flight(), 1, "one eviction");
    settle(&r.mover);
    // Brought home as soon as the evicted slot was free.
    settle(&r.mover);
    assert_eq!(
        check(&r.mover, h, 3),
        Some((a, true)),
        "the high-priority resident is home, its bytes with it"
    );
    let places: Vec<LocationId> = [l1, l2]
        .iter()
        .map(|id| r.mover.whereabouts(*id).unwrap().0)
        .collect();
    assert_eq!(
        places.iter().filter(|l| **l == a).count(),
        1,
        "one low resident stays, one was spilled"
    );
    for (id, seed) in [(l1, 1), (l2, 2)] {
        assert!(check(&r.mover, id, seed).unwrap().1);
    }
    // The low one goes, and the spilled low one comes back.
    let home = if places[0] == a { l1 } else { l2 };
    let spilled = if places[0] == a { l2 } else { l1 };
    r.mover.release(home);
    r.mover.pump();
    settle(&r.mover);
    assert_eq!(r.mover.whereabouts(spilled).unwrap().0, a);
}

#[test]
fn a_give_back_order_to_the_tiers_spills_and_hands_the_budget_back() {
    let r = rig([4, 4, 4], 8);
    let [a, _, _] = r.locations;
    let o = r
        .planner
        .open_owner(OwnerSpec::new("consumer", Priority(50), Scope::Process));
    let ids: Vec<_> = (0..3)
        .map(|_| r.mover.place(o, a, SLOT, quiet()).unwrap())
        .collect();
    // The fast location is full (the pool holds all of it); something above everything needs two slots' worth.
    let urgent = r
        .planner
        .open_owner(OwnerSpec::new("urgent", Priority(1000), Scope::Process));
    let orders = match r.planner.claim_or_reclaim(urgent, a, 2 * SLOT, "x") {
        ClaimResult::Reclaiming(o) => o,
        other => panic!("{other:?}"),
    };
    // The pool's owner holds the free slot and the residents' owner its spillable slots, both elastically: the
    // orders go to them, and the mover carries them out.
    assert!(
        orders
            .iter()
            .all(|x| x.owner == r.pool_owner || x.owner == o)
    );
    assert!(orders.iter().any(|x| x.owner == r.pool_owner));
    let handler = r.mover.give_back_handler();
    for o in orders {
        handler(o);
    }
    settle(&r.mover);
    // One free slot went at once; one resident was spilled, and the next order takes its slot.
    let free_now = match r.planner.claim_or_reclaim(urgent, a, 2 * SLOT, "x") {
        ClaimResult::Granted(c) => Some(c),
        ClaimResult::Reclaiming(more) => {
            for o in more {
                handler(o);
            }
            settle(&r.mover);
            match r.planner.claim_or_reclaim(urgent, a, 2 * SLOT, "x") {
                ClaimResult::Granted(c) => Some(c),
                other => panic!("{other:?}"),
            }
        }
        ClaimResult::Refused(s) => panic!("{s:?}"),
    };
    assert!(free_now.is_some());
    assert_eq!(r.pools[0].retired(), 2);
    let at_a = ids
        .iter()
        .filter(|id| r.mover.whereabouts(**id).unwrap().0 == a)
        .count();
    assert_eq!(at_a, 2, "the residents that fit stay");
    // The urgent claim goes: the slots' budget comes back and the spilled resident with it.
    drop(free_now);
    assert_eq!(r.mover.restore(), 2);
    settle(&r.mover);
    settle(&r.mover);
    assert!(
        ids.iter()
            .all(|id| r.mover.whereabouts(*id).unwrap().0 == a)
    );
}

#[test]
fn many_owners_move_at_once_without_locks() {
    let r = Arc::new(rig([8, 16, 32], 64));
    let [a, b, c] = r.locations;
    let threads: Vec<_> = (0..8u64)
        .map(|t| {
            let r = r.clone();
            std::thread::spawn(move || {
                let name: &'static str = Box::leak(format!("owner {t}").into_boxed_str());
                let o = r.planner.open_owner(OwnerSpec::new(
                    name,
                    Priority(10 + t as u16),
                    Scope::Process,
                ));
                let mut mine = Vec::new();
                for k in 0..4 {
                    if let Ok(id) = r.mover.place(o, [a, b, c][k % 3], SLOT, quiet()) {
                        // Fill wherever it is now; a placement that moves later carries the bytes along.
                        while !put(&r.mover, id, t * 100 + k as u64) {
                            r.mover.pump();
                        }
                        mine.push((id, t * 100 + k as u64));
                    }
                }
                for round in 0..50 {
                    for (id, _) in &mine {
                        let _ = r
                            .mover
                            .move_to(*id, [a, b, c][(round + id.index as usize) % 3]);
                    }
                    r.mover.pump();
                }
                for _ in 0..1000 {
                    r.mover.pump();
                    if mine
                        .iter()
                        .all(|(id, _)| r.mover.whereabouts(*id).is_some())
                    {
                        break;
                    }
                }
                for (id, seed) in &mine {
                    let intact = loop {
                        match check(&r.mover, *id, *seed) {
                            Some((_, ok)) => break ok,
                            None => {
                                r.mover.pump();
                            }
                        }
                    };
                    assert!(intact, "owner {t}: bytes intact after the moves");
                    assert!(r.mover.release(*id));
                }
                for l in r.locations {
                    assert_eq!(
                        r.planner.snapshot().owner(name, l),
                        0,
                        "owner {t} holds nothing at the end"
                    );
                }
            })
        })
        .collect();
    for t in threads {
        t.join().unwrap();
    }
    settle(&r.mover);
    for pool in &r.pools {
        assert_eq!(pool.free(), pool.capacity(), "every slot came back");
    }
    assert!(r.mover.moves_started() > 100);
}

#[test]
fn a_host_copier_copies_rows_between_layouts_with_their_own_pitches() {
    use euhedral_memory::{HostCopier, PitchedCopy};
    const ROW: usize = 6;
    const ROWS: usize = 4;
    let source: Vec<u8> = (0..(10 * ROWS) as u8).collect();
    let mut target = vec![0xEEu8; 8 * ROWS];
    let l = LocationId(0);
    let copier = HostCopier::new([l]);
    // SAFETY: both layouts are live vectors long enough for every row, and they do not overlap.
    let test = unsafe {
        copier.start_pitched(
            l,
            l,
            PitchedCopy {
                src: source.as_ptr() as u64,
                src_pitch: 10,
                dst: target.as_mut_ptr() as u64,
                dst_pitch: 8,
                row_bytes: ROW as u64,
                rows: ROWS as u64,
            },
        )
    }
    .unwrap();
    assert!(test.retired(), "a host copy is done when it returns");
    for row in 0..ROWS {
        assert_eq!(
            &target[row * 8..row * 8 + ROW],
            &source[row * 10..row * 10 + ROW]
        );
        assert!(
            target[row * 8 + ROW..(row + 1) * 8]
                .iter()
                .all(|&b| b == 0xEE)
        );
    }
    // A pitch shorter than the row is refused.
    // SAFETY: refused before any range is used.
    let refused = unsafe {
        copier.start_pitched(
            l,
            l,
            PitchedCopy {
                src: 0,
                src_pitch: 2,
                dst: 0,
                dst_pitch: 8,
                row_bytes: ROW as u64,
                rows: 1,
            },
        )
    };
    assert!(refused.is_err());
}

// ---------------------------------------------------------------------------------------------------------------
// The source hands frames out; the moves are driven by a frame (a worker), not by the source's request or pull.

/// Counts how often a copy is asked whether it retired.
struct Queried {
    queries: Arc<AtomicU32>,
    left: AtomicU32,
}

impl Retired for Queried {
    fn retired(&self) -> bool {
        self.queries.fetch_add(1, Ordering::Relaxed);
        self.left
            .try_update(Ordering::AcqRel, Ordering::Acquire, |n| n.checked_sub(1))
            .is_err()
    }
}

#[derive(Debug)]
struct QueriedCopier {
    pairs: Vec<(LocationId, LocationId)>,
    queries: Arc<AtomicU32>,
    /// How many queries say "not yet" before a copy retires.
    left: u32,
}

impl Copier for QueriedCopier {
    fn joins(&self, from: LocationId, to: LocationId) -> bool {
        self.pairs.contains(&(from, to))
    }

    unsafe fn start(
        &self,
        _from: LocationId,
        src: u64,
        _to: LocationId,
        dst: u64,
        bytes: u64,
    ) -> Result<Box<dyn Retired>, CopyError> {
        // SAFETY: the mover's contract: two leased slots of host memory, `bytes` each.
        unsafe { std::ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, bytes as usize) };
        Ok(Box::new(Queried {
            queries: self.queries.clone(),
            left: AtomicU32::new(self.left),
        }))
    }
}

/// A receiver that keeps what the source pushes, to be run by the test as a worker would.
#[derive(Default)]
struct Keep {
    frames: std::sync::Mutex<Vec<Box<dyn Frame>>>,
}

impl Keep {
    fn take(&self) -> Vec<Box<dyn Frame>> {
        std::mem::take(&mut *self.frames.lock().unwrap())
    }

    /// Runs every frame kept, as a worker's executor does, and returns how many.
    fn run(&self) -> usize {
        let frames = self.take();
        let n = frames.len();
        for mut frame in frames {
            frame.execute().unwrap();
            frame.do_finally();
        }
        n
    }
}

impl LatticeTerminal for Keep {
    fn add_upstream(&self, _upstream: Arc<dyn LatticeSource>) {}
}

impl LatticeReceiver for Keep {
    fn push(&self, frame: Box<dyn Frame>) {
        self.frames.lock().unwrap().push(frame);
    }

    fn on_complete(&self) {}

    fn on_error(&self, _error: Failure) {}
}

/// A rig whose copies count how often they are asked whether they retired, with one resident moving from the
/// fast tier to the middle one.
fn moving_rig() -> (Rig, Arc<AtomicU32>, Arc<std::sync::Mutex<Vec<MoveDone>>>) {
    moving_rig_with(2)
}

/// As [`moving_rig`], with a copy that retires after `left` queries.
fn moving_rig_with(left: u32) -> (Rig, Arc<AtomicU32>, Arc<std::sync::Mutex<Vec<MoveDone>>>) {
    let mut r = rig([2, 2, 2], 8);
    let queries = Arc::new(AtomicU32::new(0));
    let [a, b, c] = r.locations;
    let copier: Arc<dyn Copier> = Arc::new(QueriedCopier {
        pairs: vec![(a, b), (b, a), (b, c), (c, b)],
        queries: queries.clone(),
        left,
    });
    r.mover = Mover::new(&r.planner, r.pools.clone(), vec![copier], Some(b), 8);
    let o = r
        .planner
        .open_owner(OwnerSpec::new("consumer", Priority(50), Scope::Process));
    let seen: Arc<std::sync::Mutex<Vec<MoveDone>>> = Arc::default();
    let s = seen.clone();
    let id = r
        .mover
        .place(o, a, SLOT, Arc::new(move |d| s.lock().unwrap().push(d)))
        .unwrap();
    r.mover.move_to(id, b).unwrap();
    assert_eq!(r.mover.in_flight(), 1);
    (r, queries, seen)
}

#[test]
fn a_request_or_pull_of_the_moves_source_drives_no_move_and_a_frame_does() {
    let (r, queries, seen) = moving_rig();
    let source = r.mover.moves();
    let keep = Arc::new(Keep::default());
    source.clone().add_downstream(keep.clone());

    // A pull hands out what is done and nothing else: it does not ask the copy whether it retired.
    assert_eq!(
        source.pull(&mut |_| panic!("nothing is done"), &mut |_| false, 8),
        0
    );
    assert_eq!(queries.load(Ordering::Relaxed), 0, "a pull polls no copy");

    // The pull pushed one frame to drive the moves (a request does the same); it polled no copy itself.
    source.request(8);
    assert_eq!(
        queries.load(Ordering::Relaxed),
        0,
        "a request polls no copy"
    );
    assert_eq!(keep.frames.lock().unwrap().len(), 1, "one frame to pump");
    // That frame is the only one out: a request while it is away pushes no second.
    source.request(8);
    assert_eq!(keep.frames.lock().unwrap().len(), 1, "one pump at a time");

    // The worker that runs the frame polls the copy, as many times as the requests that follow ask it to.
    for _ in 0..16 {
        keep.run();
        if r.mover.in_flight() == 0 {
            break;
        }
        source.request(8);
    }
    assert!(queries.load(Ordering::Relaxed) >= 2, "the frame polled it");
    assert_eq!(r.mover.in_flight(), 0, "and finished the move");

    // The completion is a frame of the source, run like any other (the handler is the owner's).
    assert!(
        seen.lock().unwrap().is_empty(),
        "the handler runs in a frame"
    );
    source.request(8);
    assert!(keep.run() >= 1);
    assert_eq!(seen.lock().unwrap().len(), 1);
}

#[test]
fn a_source_with_nothing_in_flight_pushes_no_pump_frame() {
    let r = rig([2, 2, 2], 8);
    let source = r.mover.moves();
    let keep = Arc::new(Keep::default());
    source.clone().add_downstream(keep.clone());
    for _ in 0..4 {
        source.request(8);
    }
    assert!(keep.take().is_empty(), "the absence of work is the signal");
}

#[test]
fn a_freed_slot_asks_for_one_pump_to_bring_a_spilled_resident_home() {
    let r = rig([1, 2, 2], 8);
    let [a, b, _] = r.locations;
    let lo = r
        .planner
        .open_owner(OwnerSpec::new("low", Priority(10), Scope::Process));
    let hi = r
        .planner
        .open_owner(OwnerSpec::new("high", Priority(90), Scope::Process));
    let low = r.mover.place(lo, a, SLOT, quiet()).unwrap();
    let high = r.mover.place(hi, a, SLOT, quiet()).unwrap();
    settle(&r.mover);
    // The high-priority resident took the fast tier's slot and the low one was spilled below it.
    assert_eq!(r.mover.whereabouts(high).unwrap().0, a);
    assert_eq!(r.mover.whereabouts(low).unwrap().0, b);
    let source = r.mover.moves();
    let keep = Arc::new(Keep::default());
    source.clone().add_downstream(keep.clone());
    // The eviction's completion goes out first, as a frame like any other.
    source.request(8);
    keep.run();
    source.request(8);
    assert!(
        keep.take().is_empty(),
        "nothing is due while the slot is taken"
    );
    // The high one leaves: its slot is free at home, and the spilled one is due to come back.
    r.mover.release(high);
    source.request(8);
    assert_eq!(keep.run(), 1, "one pump frame");
    settle(&r.mover);
    assert_eq!(r.mover.whereabouts(low).unwrap().0, a, "brought home");
    source.request(8);
    keep.run();
    source.request(8);
    assert!(keep.take().is_empty(), "and nothing is due again");
}

#[test]
fn a_copy_that_never_retires_is_polled_ever_less_often_and_the_source_reads_unproductive_between() {
    // A worker that finds nothing to do parks only if the handle it visits reads unproductive: a visit that
    // pushes a frame does not. So between polls, a visit must push nothing.
    let (r, queries, _seen) = moving_rig_with(u32::MAX);
    let source = r.mover.moves();
    let keep = Arc::new(Keep::default());
    source.clone().add_downstream(keep.clone());
    const VISITS: usize = 20_000;
    let mut pushes = 0;
    for visit in 0..VISITS {
        if visit % 2 == 0 {
            source.pull(&mut |_| panic!("nothing is done"), &mut |_| false, 8);
        } else {
            source.request(8);
        }
        let pushed = keep.run();
        pushes += pushed;
    }
    let polls = queries.load(Ordering::Relaxed) as usize;
    assert_eq!(
        polls, pushes,
        "one query per pump frame, and nothing else queries"
    );
    assert!(polls >= 8, "it is still polled ({polls})");
    assert!(
        polls <= 120,
        "{polls} polls in {VISITS} visits: the back-off did not grow"
    );
    // A new move is polled at the next visit, however far the back-off had grown.
    let before = queries.load(Ordering::Relaxed);
    let [a, b, _] = r.locations;
    let o = r
        .planner
        .open_owner(OwnerSpec::new("second", Priority(50), Scope::Process));
    let id = r.mover.place(o, a, SLOT, quiet()).unwrap();
    r.mover.move_to(id, b).unwrap();
    source.pull(&mut |_| panic!("nothing is done"), &mut |_| false, 8);
    assert_eq!(keep.run(), 1, "a new move asks for a pump at once");
    assert!(queries.load(Ordering::Relaxed) > before);
}

#[test]
fn a_freed_slot_asks_for_a_pump_whatever_the_back_off() {
    let (r, _queries, _seen) = moving_rig_with(u32::MAX);
    let source = r.mover.moves();
    let keep = Arc::new(Keep::default());
    source.clone().add_downstream(keep.clone());
    for visit in 0..2_000 {
        if visit % 2 == 0 {
            source.pull(&mut |_| panic!("nothing is done"), &mut |_| false, 8);
        } else {
            source.request(8);
        }
        keep.run();
    }
    // A slot frees: the next visit pushes the pump, back-off or not.
    let [a, ..] = r.locations;
    let o = r
        .planner
        .open_owner(OwnerSpec::new("other", Priority(50), Scope::Process));
    let id = r.mover.place(o, a, SLOT, quiet()).unwrap();
    keep.run();
    r.mover.release(id);
    source.request(8);
    assert_eq!(keep.run(), 1);
}

#[test]
fn a_resident_busy_when_its_home_frees_still_comes_home() {
    let r = rig([1, 2, 2], 8);
    let [a, b, _] = r.locations;
    let lo = r
        .planner
        .open_owner(OwnerSpec::new("low", Priority(10), Scope::Process));
    let hi = r
        .planner
        .open_owner(OwnerSpec::new("high", Priority(90), Scope::Process));
    let low = r.mover.place(lo, a, SLOT, quiet()).unwrap();
    let high = r.mover.place(hi, a, SLOT, quiet()).unwrap();
    settle(&r.mover);
    assert_eq!(r.mover.whereabouts(high).unwrap().0, a);
    assert_eq!(r.mover.whereabouts(low).unwrap().0, b);
    // The home slot frees, and a pump runs, while the spilled resident is busy (its owner is inside `with`).
    r.mover
        .with(low, |_, _| {
            assert!(r.mover.release(high));
            r.mover.pump();
        })
        .expect("resident and idle");
    assert!(
        r.mover.pump_due(),
        "the pump could not move the busy resident: another is due"
    );
    settle(&r.mover);
    assert_eq!(r.mover.whereabouts(low).unwrap().0, a, "brought home");
}
