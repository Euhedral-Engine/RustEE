//! Moving claimed memory between locations: tiers, residents and planned moves.
//!
//! A **tier** is a location with a [`TierPool`]: one planned allocation cut into equal slots, made once. Free slots are
//! an elastic holding of the pool's owner, so a higher-priority claim elsewhere can order them back. A **resident** is
//! one slot's worth of an owner's data, kept in the fastest tier its priority earns; the [`Mover`] moves residents
//! between tiers by copying their contents:
//! - **asynchronously**, through a [`Copier`] per pair of locations (a host memcpy for host memory, the CUDA backend's
//!   stream copies for a device); a pair no copier joins goes through a staging tier in two hops;
//! - **by plan**: a resident placed in a full tier evicts the lowest-priority resident there to the next tier;
//!   spilled residents come back, highest priority first, as slots free up; a give-back order to a pool's owner spills
//!   that tier's lowest-priority residents and hands the freed slots' budget back;
//! - **with exact accounting**: the destination slot is leased (its bytes move to the resident's owner) before the
//!   copy starts, and the source slot goes back to its pool only once the copy completed, so the owner holds both
//!   while the data is in flight and one afterwards;
//! - **without allocating**: every slot comes from a pool planned up front; a move only leases and returns slots.
//!
//! Completions reach the residents' owners as lattice frames of the mover's [`Moves`] source (routed by the owner's
//! hash, as give-back orders are); the same source drives the moves when the lattice asks it for frames. Nothing
//! takes a lock: residents are cells of atomics, moves and free slots go through lock-free queues.

use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU16, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock, Weak};

use euhedral_core::frames::{Frame, RunnableFrame};
use euhedral_core::generics::{LatticeReceiver, LatticeSource};
use euhedral_data_structures::queues::{Consumer, MpmcQueue, Multi, Producer};
use euhedral_data_structures::sync::HotScope;
use euhedral_graph::{DeviceQueue, Marker};

use crate::backend::Retired;
use crate::planner::{Claim, GiveBack, GiveBackHandler, LocationId, OwnerId, Planner, Priority};

/// A lock-free queue's two ends.
type Queue<T> = (Producer<T, Multi>, Consumer<T, Multi>);

// ---------------------------------------------------------------------------------------------------------------
// Copiers

/// Why a copy could not start.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CopyError {
    pub what: String,
}

impl fmt::Display for CopyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.what)
    }
}

impl std::error::Error for CopyError {}

/// A copy of `rows` rows of `row_bytes` bytes each between two layouts that have their own row pitches (the distance in
/// bytes between the starts of consecutive rows), such as a picture with padded rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PitchedCopy {
    /// The first byte of the first source row.
    pub src: u64,
    /// The source's row pitch in bytes, at least `row_bytes`.
    pub src_pitch: u64,
    /// The first byte of the first destination row.
    pub dst: u64,
    /// The destination's row pitch in bytes, at least `row_bytes`.
    pub dst_pitch: u64,
    /// The bytes copied from each row.
    pub row_bytes: u64,
    /// The number of rows.
    pub rows: u64,
}

/// Copies bytes between two locations without waiting: `start` queues the copy and returns the test of its
/// completion.
pub trait Copier: Send + Sync + fmt::Debug {
    /// Whether this copier copies from `from` to `to`.
    fn joins(&self, from: LocationId, to: LocationId) -> bool;

    /// Starts copying `bytes` from address `src` at `from` to address `dst` at `to`.
    ///
    /// # Safety
    /// Both ranges are valid for `bytes` at their locations, do not overlap, and nothing else writes `dst` or frees
    /// either until the returned test says the copy retired.
    unsafe fn start(
        &self,
        from: LocationId,
        src: u64,
        to: LocationId,
        dst: u64,
        bytes: u64,
    ) -> Result<Box<dyn Retired>, CopyError>;

    /// Starts a pitched copy (rows with their own pitches) from `from` to `to`. A copier that has no such copy refuses.
    ///
    /// # Safety
    /// Every row range of both layouts is valid at its location, the two layouts do not overlap, and nothing else
    /// writes the destination rows or frees either layout until the returned test says the copy retired.
    unsafe fn start_pitched(
        &self,
        from: LocationId,
        to: LocationId,
        copy: PitchedCopy,
    ) -> Result<Box<dyn Retired>, CopyError> {
        let _ = (from, to, copy);
        Err(CopyError {
            what: "pitched copies: not this copier's".into(),
        })
    }
}

/// Copies between host locations (pinned and pageable host memory) with `memcpy`, at once.
#[derive(Debug)]
pub struct HostCopier {
    locations: Vec<LocationId>,
}

impl HostCopier {
    /// A copier between any two of `locations`, all of them addressable by the host.
    pub fn new(locations: impl IntoIterator<Item = LocationId>) -> Self {
        HostCopier {
            locations: locations.into_iter().collect(),
        }
    }
}

impl Copier for HostCopier {
    fn joins(&self, from: LocationId, to: LocationId) -> bool {
        self.locations.contains(&from) && self.locations.contains(&to)
    }

    unsafe fn start(
        &self,
        _from: LocationId,
        src: u64,
        _to: LocationId,
        dst: u64,
        bytes: u64,
    ) -> Result<Box<dyn Retired>, CopyError> {
        // SAFETY: the caller's contract: valid, non-overlapping host ranges nobody else touches meanwhile.
        unsafe {
            std::ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, bytes as usize);
        }
        Ok(Box::new(|| true))
    }

    unsafe fn start_pitched(
        &self,
        _from: LocationId,
        _to: LocationId,
        copy: PitchedCopy,
    ) -> Result<Box<dyn Retired>, CopyError> {
        if copy.src_pitch < copy.row_bytes || copy.dst_pitch < copy.row_bytes {
            return Err(CopyError {
                what: "a row pitch is shorter than the row".into(),
            });
        }
        for row in 0..copy.rows {
            // SAFETY: the caller's contract: every row range is valid, the layouts do not overlap, and nobody else
            // touches the destination rows meanwhile.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    (copy.src + row * copy.src_pitch) as *const u8,
                    (copy.dst + row * copy.dst_pitch) as *mut u8,
                    copy.row_bytes as usize,
                );
            }
        }
        Ok(Box::new(|| true))
    }
}

/// A lane's marker as a completion test: retired once the lane's queue completed it. A marker of another queue is
/// never retired.
pub struct MarkerRetired {
    pub queue: Arc<dyn DeviceQueue>,
    pub marker: Marker,
}

impl fmt::Debug for MarkerRetired {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MarkerRetired")
            .field("marker", &self.marker)
            .finish()
    }
}

impl Retired for MarkerRetired {
    fn retired(&self) -> bool {
        self.queue.id() == self.marker.queue && self.queue.completed() >= self.marker.value
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Tier pools

/// One tier's slots: a planned allocation at `location`, cut into `count` slots of `slot_bytes` from `base`. Each
/// free slot carries its claim, an elastic holding of the pool's owner; leasing a slot moves its bytes to the
/// lessee's account, returning it moves them back.
pub struct TierPool {
    location: LocationId,
    owner: OwnerId,
    base: u64,
    slot_bytes: u64,
    count: u32,
    free: Queue<(u32, Claim)>,
    free_count: AtomicU32,
    /// Slots whose budget was given back on order: (index) without a claim, until restored.
    retired: Queue<u32>,
    retired_count: AtomicU32,
}

impl fmt::Debug for TierPool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TierPool")
            .field("location", &self.location)
            .field("slot_bytes", &self.slot_bytes)
            .field("count", &self.count)
            .field("free", &self.free())
            .field("retired", &self.retired())
            .finish()
    }
}

impl TierPool {
    /// A pool of `count` slots of `slot_bytes` from `base` (the address of a planned allocation at `location`),
    /// owned by `owner`. Claims the whole pool for the owner (elastic: free slots can be ordered back); `None` when
    /// the budget does not have it.
    pub fn new(
        planner: &Arc<Planner>,
        owner: OwnerId,
        location: LocationId,
        base: u64,
        slot_bytes: u64,
        count: u32,
    ) -> Option<Arc<Self>> {
        let n = (count as usize).next_power_of_two().max(16);
        let q = MpmcQueue::<(u32, Claim)>::new(n);
        let r = MpmcQueue::<u32>::new(n);
        let pool = TierPool {
            location,
            owner,
            base,
            slot_bytes,
            count,
            free: (q.producer(), q.consumer()),
            free_count: AtomicU32::new(0),
            retired: (r.producer(), r.consumer()),
            retired_count: AtomicU32::new(0),
        };
        for i in 0..count {
            let mut e = planner
                .elastic(owner, location, 0, slot_bytes, "tier slot")
                .ok()?;
            if e.grow_to(slot_bytes) < slot_bytes {
                return None;
            }
            pool.push_free(i, e.into_claim());
        }
        Some(Arc::new(pool))
    }

    fn push_free(&self, index: u32, claim: Claim) {
        self.free_count.fetch_add(1, Ordering::AcqRel);
        let _ = self.free.0.offer((index, claim));
    }

    pub fn location(&self) -> LocationId {
        self.location
    }

    pub fn owner(&self) -> OwnerId {
        self.owner
    }

    pub fn slot_bytes(&self) -> u64 {
        self.slot_bytes
    }

    pub fn capacity(&self) -> u32 {
        self.count
    }

    /// Free slots now.
    pub fn free(&self) -> u32 {
        self.free_count.load(Ordering::Acquire)
    }

    /// Slots whose budget is given back now.
    pub fn retired(&self) -> u32 {
        self.retired_count.load(Ordering::Acquire)
    }

    pub fn address(&self, index: u32) -> u64 {
        self.base + u64::from(index) * self.slot_bytes
    }

    /// A free slot for `owner` (its bytes now on `owner`'s account), or `None` when every slot is leased. `elastic`
    /// holdings can be ordered back from `owner` (a resident that can spill to a slower tier).
    pub fn lease(&self, owner: OwnerId, elastic: bool) -> Option<(u32, Claim)> {
        let (index, claim) = self.free.1.poll()?;
        self.free_count.fetch_sub(1, Ordering::AcqRel);
        match claim.transfer(owner, elastic) {
            Ok(c) => Some((index, c)),
            Err(c) => {
                self.push_free(index, c);
                None
            }
        }
    }

    /// Returns a slot (its bytes back on the pool owner's account, elastic).
    pub fn give_back(&self, index: u32, claim: Claim) {
        match claim.transfer(self.owner, true) {
            Ok(c) => self.push_free(index, c),
            // The pool's owner ended: the slot's budget goes with the claim.
            Err(c) => {
                drop(c);
                self.retire(index);
            }
        }
    }

    fn retire(&self, index: u32) {
        self.retired_count.fetch_add(1, Ordering::AcqRel);
        let _ = self.retired.0.offer(index);
    }

    /// Gives the budget of up to `n` free slots back (a give-back order to the pool's owner); returns how many.
    pub fn retire_free(&self, n: u32) -> u32 {
        let mut done = 0;
        while done < n {
            let Some((index, claim)) = self.free.1.poll() else {
                break;
            };
            self.free_count.fetch_sub(1, Ordering::AcqRel);
            drop(claim);
            self.retire(index);
            done += 1;
        }
        done
    }

    /// Claims the budget of up to `n` retired slots again, as far as the location's free memory allows; returns how
    /// many came back.
    pub fn restore(&self, planner: &Arc<Planner>, n: u32) -> u32 {
        let mut done = 0;
        while done < n {
            let Some(index) = self.retired.1.poll() else {
                break;
            };
            let claim = planner
                .elastic(self.owner, self.location, 0, self.slot_bytes, "tier slot")
                .ok()
                .and_then(|mut e| {
                    (e.grow_to(self.slot_bytes) == self.slot_bytes).then(|| e.into_claim())
                });
            match claim {
                Some(c) => {
                    self.retired_count.fetch_sub(1, Ordering::AcqRel);
                    self.push_free(index, c);
                    done += 1;
                }
                None => {
                    let _ = self.retired.0.offer(index);
                    break;
                }
            }
        }
        done
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Residents

/// A resident's handle: a cell of the mover and its generation (a reused cell is a different resident).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ResidentId {
    pub index: u32,
    pub generation: u32,
}

/// What a resident's owner hears when one of its moves ends.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MoveDone {
    pub resident: ResidentId,
    pub owner: OwnerId,
    pub from: LocationId,
    pub to: LocationId,
    /// The resident was released during the move: both slots went back and the resident is gone.
    pub cancelled: bool,
}

/// A resident owner's handler for its moves' completions (it runs as a lattice frame).
pub type MoveHandler = Arc<dyn Fn(MoveDone) + Send + Sync>;

const FREE: u32 = 0;
const RESIDENT: u32 = 1;
const MOVING: u32 = 2;
const BUSY: u32 = 3;

const NO_LOCATION: u16 = u16::MAX;

struct Cell {
    state: AtomicU32,
    generation: AtomicU32,
    owner: AtomicU64,
    priority: AtomicU16,
    home: AtomicU16,
    location: AtomicU16,
    slot: AtomicU32,
    claim: AtomicPtr<Claim>,
    handler: AtomicPtr<MoveHandler>,
    cancelled: AtomicBool,
}

impl Cell {
    fn new() -> Self {
        Cell {
            state: AtomicU32::new(FREE),
            generation: AtomicU32::new(0),
            owner: AtomicU64::new(0),
            priority: AtomicU16::new(0),
            home: AtomicU16::new(NO_LOCATION),
            location: AtomicU16::new(NO_LOCATION),
            slot: AtomicU32::new(0),
            claim: AtomicPtr::new(std::ptr::null_mut()),
            handler: AtomicPtr::new(std::ptr::null_mut()),
            cancelled: AtomicBool::new(false),
        }
    }

    fn owner(&self) -> OwnerId {
        let p = self.owner.load(Ordering::Acquire);
        OwnerId {
            slot: (p >> 32) as u32,
            generation: p as u32,
        }
    }

    fn take_claim(&self) -> Option<Claim> {
        let p = self.claim.swap(std::ptr::null_mut(), Ordering::AcqRel);
        // SAFETY: a pointer from `Box::into_raw` in `put_claim`, swapped out once.
        (!p.is_null()).then(|| *unsafe { Box::from_raw(p) })
    }

    fn put_claim(&self, claim: Claim) {
        let old = self
            .claim
            .swap(Box::into_raw(Box::new(claim)), Ordering::AcqRel);
        if !old.is_null() {
            // SAFETY: as in `take_claim`.
            drop(unsafe { Box::from_raw(old) });
        }
    }

    fn handler(&self) -> Option<MoveHandler> {
        let p = self.handler.load(Ordering::Acquire);
        // SAFETY: set at placement, freed only when the cell is freed (state FREE, no reader past the state check).
        (!p.is_null()).then(|| unsafe { (*p).clone() })
    }

    fn clear_handler(&self) {
        let p = self.handler.swap(std::ptr::null_mut(), Ordering::AcqRel);
        if !p.is_null() {
            // SAFETY: from `Box::into_raw` at placement.
            drop(unsafe { Box::from_raw(p) });
        }
    }
}

/// Why a resident could not be placed or moved.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MoveError {
    /// No tier had a slot for it (every tier full of residents of its priority or above).
    NoRoom,
    /// The resident is gone, or busy with another move.
    NotResident,
    /// The data is bigger than a slot of the tier.
    TooBig,
    /// No copier joins the two locations, directly or through the staging tier.
    NoCopier,
    Copy(CopyError),
}

// ---------------------------------------------------------------------------------------------------------------
// Moves

struct Hop {
    location: LocationId,
    slot: u32,
    claim: Claim,
}

struct MoveTask {
    resident: ResidentId,
    bytes: u64,
    from: LocationId,
    /// Where the data is copied from now (the resident's slot; the staging slot in a two-hop move's second hop).
    source: Option<Hop>,
    /// The staging slot of a two-hop move, while its first copy is in flight.
    stage: Option<Hop>,
    dest: Hop,
    retired: Box<dyn Retired>,
    /// A spill for a give-back order: the source slot's budget goes back to the planner (the slot retires) rather
    /// than staying free in its pool, where bring-back would refill it at once.
    give_back: bool,
}

/// The tiers, their residents and the moves between them.
pub struct Mover {
    planner: Arc<Planner>,
    /// The tiers, fastest first; `pools[i]` is `tiers[i]`'s pool.
    pools: Vec<Arc<TierPool>>,
    copiers: Vec<Arc<dyn Copier>>,
    /// The tier two-hop moves go through (a pinned host tier), by index into `pools`.
    staging: Option<usize>,
    cells: Box<[Cell]>,
    free_cells: Queue<u32>,
    moves: Queue<Box<MoveTask>>,
    in_flight: AtomicU32,
    done: Queue<(MoveHandler, MoveDone)>,
    source: Arc<Moves>,
    moves_started: AtomicU64,
}

impl fmt::Debug for Mover {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Mover")
            .field("pools", &self.pools)
            .field("in_flight", &self.in_flight())
            .finish_non_exhaustive()
    }
}

impl Mover {
    /// A mover over `pools` (the tiers, fastest first), copying with `copiers`; `staging` is the location two-hop
    /// moves pass through. Holds up to `residents` residents at once.
    pub fn new(
        planner: &Arc<Planner>,
        pools: Vec<Arc<TierPool>>,
        copiers: Vec<Arc<dyn Copier>>,
        staging: Option<LocationId>,
        residents: u32,
    ) -> Arc<Self> {
        let n = (residents as usize).next_power_of_two().max(16);
        let cq = MpmcQueue::<u32>::new(n);
        let free_cells = (cq.producer(), cq.consumer());
        for i in 0..residents {
            let _ = free_cells.0.offer(i);
        }
        let mq = MpmcQueue::<Box<MoveTask>>::new(n);
        let dq = MpmcQueue::<(MoveHandler, MoveDone)>::new(n);
        let staging = staging.and_then(|l| pools.iter().position(|p| p.location == l));
        Arc::new_cyclic(|weak: &Weak<Mover>| Mover {
            planner: planner.clone(),
            pools,
            copiers,
            staging,
            cells: (0..residents).map(|_| Cell::new()).collect(),
            free_cells,
            moves: (mq.producer(), mq.consumer()),
            in_flight: AtomicU32::new(0),
            done: (dq.producer(), dq.consumer()),
            source: Arc::new(Moves::new(weak.clone())),
            moves_started: AtomicU64::new(0),
        })
    }

    /// The source of completion frames: add it upstream of a lattice; the lattice's demand also drives the moves.
    pub fn moves(&self) -> Arc<Moves> {
        self.source.clone()
    }

    /// Moves in flight.
    pub fn in_flight(&self) -> u32 {
        self.in_flight.load(Ordering::Acquire)
    }

    /// Moves started so far.
    pub fn moves_started(&self) -> u64 {
        self.moves_started.load(Ordering::Acquire)
    }

    pub fn pool(&self, location: LocationId) -> Option<&Arc<TierPool>> {
        self.pools.iter().find(|p| p.location == location)
    }

    fn tier(&self, location: LocationId) -> Option<usize> {
        self.pools.iter().position(|p| p.location == location)
    }

    fn cell(&self, id: ResidentId) -> Option<&Cell> {
        let c = self.cells.get(id.index as usize)?;
        (c.generation.load(Ordering::Acquire) == id.generation).then_some(c)
    }

    /// Where a resident is now (its location and slot address), `None` while it moves or once it is gone.
    pub fn whereabouts(&self, id: ResidentId) -> Option<(LocationId, u64)> {
        let c = self.cell(id)?;
        if c.state.load(Ordering::Acquire) != RESIDENT {
            return None;
        }
        let loc = LocationId(c.location.load(Ordering::Acquire));
        let pool = self.pool(loc)?;
        let addr = pool.address(c.slot.load(Ordering::Acquire));
        // Still the same resident at the same place (no move started in between).
        (c.state.load(Ordering::Acquire) == RESIDENT
            && c.generation.load(Ordering::Acquire) == id.generation)
            .then_some((loc, addr))
    }

    /// Places a resident of `owner` whose home is `home` (the tier it earns): in a free slot of the first tier from
    /// `home` down that has one. A full home tier with a resident of lower priority evicts it (spilled to the next
    /// tier), and this one comes home once its slot is free. `on_moved` hears each of its moves.
    pub fn place(
        self: &Arc<Self>,
        owner: OwnerId,
        home: LocationId,
        bytes: u64,
        on_moved: MoveHandler,
    ) -> Result<ResidentId, MoveError> {
        let start = self.tier(home).ok_or(MoveError::NoRoom)?;
        if bytes > self.pools[start].slot_bytes {
            return Err(MoveError::TooBig);
        }
        let priority = self.planner.priority_of(owner);
        let mut placed = None;
        for t in start..self.pools.len() {
            if bytes > self.pools[t].slot_bytes {
                continue;
            }
            if let Some(lease) = self.pools[t].lease(owner, self.spillable(t)) {
                placed = Some((t, lease));
                break;
            }
            if t == start {
                // Make room at home for later: the lowest-priority resident there below this one moves down.
                self.evict(start, priority, false);
            }
        }
        let Some((t, (slot, claim))) = placed else {
            return Err(MoveError::NoRoom);
        };
        let Some(index) = self.free_cells.1.poll() else {
            self.pools[t].give_back(slot, claim);
            return Err(MoveError::NoRoom);
        };
        let c = &self.cells[index as usize];
        let generation = c.generation.load(Ordering::Acquire);
        c.owner.store(
            (u64::from(owner.slot) << 32) | u64::from(owner.generation),
            Ordering::Release,
        );
        c.priority.store(priority.0, Ordering::Release);
        c.home.store(home.0, Ordering::Release);
        c.location
            .store(self.pools[t].location.0, Ordering::Release);
        c.slot.store(slot, Ordering::Release);
        c.cancelled.store(false, Ordering::Release);
        c.put_claim(claim);
        c.clear_handler();
        c.handler
            .store(Box::into_raw(Box::new(on_moved)), Ordering::Release);
        c.state.store(RESIDENT, Ordering::Release);
        Ok(ResidentId { index, generation })
    }

    /// Releases a resident: its slot goes back to its pool now, or, while it moves, both slots go back once the copy
    /// completed (its owner hears a cancelled move).
    pub fn release(&self, id: ResidentId) -> bool {
        let Some(c) = self.cell(id) else { return false };
        loop {
            match c.state.load(Ordering::Acquire) {
                RESIDENT => {
                    if c.state
                        .compare_exchange(RESIDENT, BUSY, Ordering::AcqRel, Ordering::Acquire)
                        .is_ok()
                    {
                        let loc = LocationId(c.location.load(Ordering::Acquire));
                        if let (Some(pool), Some(claim)) = (self.pool(loc), c.take_claim()) {
                            pool.give_back(c.slot.load(Ordering::Acquire), claim);
                        }
                        self.free_cell(id.index);
                        return true;
                    }
                }
                MOVING | BUSY => {
                    c.cancelled.store(true, Ordering::Release);
                    // The move's end sees the flag (it reads it after leaving MOVING only through `finish`).
                    return true;
                }
                _ => return false,
            }
        }
    }

    fn free_cell(&self, index: u32) {
        let c = &self.cells[index as usize];
        c.clear_handler();
        c.location.store(NO_LOCATION, Ordering::Release);
        c.generation.fetch_add(1, Ordering::AcqRel);
        c.state.store(FREE, Ordering::Release);
        let _ = self.free_cells.0.offer(index);
    }

    /// Moves a resident to `to`, which becomes its home (an explicit move; planned ones come from placement,
    /// give-back and bring-back, and keep the home).
    pub fn move_to(&self, id: ResidentId, to: LocationId) -> Result<(), MoveError> {
        let c = self.cell(id).ok_or(MoveError::NotResident)?;
        if c.state
            .compare_exchange(RESIDENT, MOVING, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return Err(MoveError::NotResident);
        }
        let home = c.home.swap(to.0, Ordering::AcqRel);
        self.start_move(id, to, false).inspect_err(|_| {
            c.home.store(home, Ordering::Release);
            c.state.store(RESIDENT, Ordering::Release);
        })
    }

    /// Runs `f` with the resident's location and slot address while no move can start (the owner reads or writes
    /// its data there); `None` while it moves or once it is gone. Keep `f` short: planned moves skip it meanwhile.
    pub fn with<R>(&self, id: ResidentId, f: impl FnOnce(LocationId, u64) -> R) -> Option<R> {
        let c = self.cell(id)?;
        c.state
            .compare_exchange(RESIDENT, BUSY, Ordering::AcqRel, Ordering::Acquire)
            .ok()?;
        let r = (c.generation.load(Ordering::Acquire) == id.generation).then(|| {
            let loc = LocationId(c.location.load(Ordering::Acquire));
            let addr = self
                .pool(loc)
                .map_or(0, |p| p.address(c.slot.load(Ordering::Acquire)));
            f(loc, addr)
        });
        c.state.store(RESIDENT, Ordering::Release);
        // A release that came while it was busy.
        if c.cancelled.swap(false, Ordering::AcqRel) {
            self.release(id);
        }
        r
    }

    /// Starts moving a resident whose cell this thread set to MOVING.
    fn start_move(&self, id: ResidentId, to: LocationId, give_back: bool) -> Result<(), MoveError> {
        let c = &self.cells[id.index as usize];
        let from = LocationId(c.location.load(Ordering::Acquire));
        let ti = self.tier(to).ok_or(MoveError::NoRoom)?;
        let fi = self.tier(from).ok_or(MoveError::NotResident)?;
        let bytes = self.pools[fi].slot_bytes.min(self.pools[ti].slot_bytes);
        if from == to {
            return Err(MoveError::NotResident);
        }
        let direct = self.copier(from, to);
        let stage_tier = match direct {
            Some(_) => None,
            None => {
                let s = self.staging.ok_or(MoveError::NoCopier)?;
                let sl = self.pools[s].location;
                if self.copier(from, sl).is_none() || self.copier(sl, to).is_none() {
                    return Err(MoveError::NoCopier);
                }
                Some(s)
            }
        };
        let owner = c.owner();
        // The destination first: its bytes are the owner's before anything is copied.
        let (dslot, dclaim) = self.pools[ti]
            .lease(owner, self.spillable(ti))
            .ok_or(MoveError::NoRoom)?;
        let stage = match stage_tier {
            // A staging slot is held only for the copy: firm.
            Some(s) => match self.pools[s].lease(owner, false) {
                Some((slot, claim)) => Some(Hop {
                    location: self.pools[s].location,
                    slot,
                    claim,
                }),
                None => {
                    self.pools[ti].give_back(dslot, dclaim);
                    return Err(MoveError::NoRoom);
                }
            },
            None => None,
        };
        let Some(sclaim) = c.take_claim() else {
            self.pools[ti].give_back(dslot, dclaim);
            if let Some(h) = stage {
                self.pools[self.tier(h.location).unwrap_or(0)].give_back(h.slot, h.claim);
            }
            return Err(MoveError::NotResident);
        };
        let source = Hop {
            location: from,
            slot: c.slot.load(Ordering::Acquire),
            claim: sclaim,
        };
        let dest = Hop {
            location: to,
            slot: dslot,
            claim: dclaim,
        };
        let first_to = stage.as_ref().map_or((to, dslot), |h| (h.location, h.slot));
        let started = self.copy((source.location, source.slot), first_to, bytes);
        match started {
            Ok(retired) => {
                self.in_flight.fetch_add(1, Ordering::AcqRel);
                self.moves_started.fetch_add(1, Ordering::AcqRel);
                let task = Box::new(MoveTask {
                    resident: id,
                    bytes,
                    from,
                    source: Some(source),
                    stage,
                    dest,
                    retired,
                    give_back,
                });
                if let Err(task) = self.moves.0.offer(task) {
                    // An unbounded queue never refuses; finish at once if it did.
                    self.finish(*task);
                }
                Ok(())
            }
            Err(e) => {
                c.put_claim(source.claim);
                self.pools[ti].give_back(dest.slot, dest.claim);
                if let Some(h) = stage {
                    self.pools[self.tier(h.location).unwrap_or(0)].give_back(h.slot, h.claim);
                }
                Err(MoveError::Copy(e))
            }
        }
    }

    fn copier(&self, from: LocationId, to: LocationId) -> Option<&Arc<dyn Copier>> {
        self.copiers.iter().find(|c| c.joins(from, to))
    }

    fn copy(
        &self,
        (from, fslot): (LocationId, u32),
        (to, tslot): (LocationId, u32),
        bytes: u64,
    ) -> Result<Box<dyn Retired>, CopyError> {
        let copier = self.copier(from, to).ok_or_else(|| CopyError {
            what: "no copier".into(),
        })?;
        let (fp, tp) = (
            self.pool(from).ok_or_else(|| CopyError {
                what: "no pool".into(),
            })?,
            self.pool(to).ok_or_else(|| CopyError {
                what: "no pool".into(),
            })?,
        );
        // SAFETY: two leased slots of planned pools, `bytes` within each; the source slot is the resident's (nobody
        // writes it while it moves: the owner reads it only through `whereabouts`, which says it is moving), and the
        // destination slot was leased for this move.
        unsafe { copier.start(from, fp.address(fslot), to, tp.address(tslot), bytes) }
    }

    /// Drives the moves: finishes those whose copy retired (or starts their second hop), then brings spilled
    /// residents home where slots are free. Returns the moves finished. Never waits; any thread may call it.
    pub fn pump(&self) -> u32 {
        let mut finished = 0;
        let mut keep = Vec::new();
        let limit = self.moves.1.size_long();
        let mut n = 0;
        while n < limit {
            let Some(mut task) = self.moves.1.poll() else {
                break;
            };
            n += 1;
            if !task.retired.retired() {
                keep.push(task);
                continue;
            }
            if let Some(stage) = task.stage.take() {
                // The first hop is done: the source goes back, the staging slot is copied on to the destination.
                if let Some(source) = task.source.replace(stage) {
                    self.free_source(source, task.give_back);
                }
                let from = task.source.as_ref().map(|h| (h.location, h.slot));
                match from.map(|f| self.copy(f, (task.dest.location, task.dest.slot), task.bytes)) {
                    Some(Ok(r)) => {
                        task.retired = r;
                        keep.push(task);
                    }
                    _ => {
                        // Could not go on: the data stays in the staging slot, which becomes the resident's.
                        if let Some(staged) = task.source.take() {
                            let dest = std::mem::replace(&mut task.dest, staged);
                            self.return_slot(dest);
                        }
                        self.finish_task(*task);
                        finished += 1;
                    }
                }
                continue;
            }
            self.finish(*task);
            finished += 1;
        }
        for t in keep {
            if let Err(t) = self.moves.0.offer(t) {
                self.finish(*t);
            }
        }
        self.bring_back();
        finished
    }

    /// The resident's old slot after its data left: back to its pool, or retired for a give-back spill.
    fn free_source(&self, hop: Hop, give_back: bool) {
        match self.pool(hop.location) {
            Some(pool) if give_back => {
                drop(hop.claim);
                pool.retire(hop.slot);
            }
            _ => self.return_slot(hop),
        }
    }

    fn return_slot(&self, hop: Hop) {
        match self.pool(hop.location) {
            Some(pool) => pool.give_back(hop.slot, hop.claim),
            None => drop(hop.claim),
        }
    }

    /// The copy to the destination completed: the source goes back, the resident is at the destination.
    fn finish(&self, mut task: MoveTask) {
        if let Some(source) = task.source.take() {
            self.free_source(source, task.give_back);
        }
        if let Some(stage) = task.stage.take() {
            self.return_slot(stage);
        }
        self.finish_task(task);
    }

    /// Makes the resident live at `task.dest` (or gives the destination back if it was released meanwhile) and
    /// queues its owner's completion.
    fn finish_task(&self, task: MoveTask) {
        self.in_flight.fetch_sub(1, Ordering::AcqRel);
        let id = task.resident;
        let c = &self.cells[id.index as usize];
        let cancelled = c.cancelled.load(Ordering::Acquire);
        let dest = task.dest;
        let to = dest.location;
        let done = MoveDone {
            resident: id,
            owner: c.owner(),
            from: task.from,
            to,
            cancelled,
        };
        let handler = c.handler();
        if cancelled {
            self.return_slot(dest);
            self.free_cell(id.index);
        } else {
            c.location.store(to.0, Ordering::Release);
            c.slot.store(dest.slot, Ordering::Release);
            c.put_claim(dest.claim);
            c.state.store(RESIDENT, Ordering::Release);
            // A release that came between the flag's read and now: honour it.
            if c.cancelled.swap(false, Ordering::AcqRel) {
                self.release(id);
            }
        }
        if let Some(h) = handler {
            let _ = self.done.0.offer((h, done));
            self.source.wake();
        }
    }

    /// Whether a resident of tier `t` can spill (a slower tier exists): its slot is then an elastic holding, so a
    /// higher-priority claim sees it as reclaimable and the give-back order comes to the mover.
    fn spillable(&self, t: usize) -> bool {
        t + 1 < self.pools.len()
    }

    /// Spills the lowest-priority resident of tier `t` below `priority` to the next tier with a free slot. Returns
    /// whether a move started.
    fn evict(&self, t: usize, priority: Priority, give_back: bool) -> bool {
        let loc = self.pools[t].location;
        loop {
            let victim = self
                .cells
                .iter()
                .enumerate()
                .filter(|(_, c)| {
                    c.state.load(Ordering::Acquire) == RESIDENT
                        && c.location.load(Ordering::Acquire) == loc.0
                        && c.priority.load(Ordering::Acquire) < priority.0
                })
                .min_by_key(|(_, c)| c.priority.load(Ordering::Acquire))
                .map(|(i, c)| (i as u32, c.generation.load(Ordering::Acquire)));
            let Some((index, generation)) = victim else {
                return false;
            };
            let c = &self.cells[index as usize];
            if c.state
                .compare_exchange(RESIDENT, MOVING, Ordering::AcqRel, Ordering::Acquire)
                .is_err()
            {
                continue;
            }
            let id = ResidentId { index, generation };
            for down in t + 1..self.pools.len() {
                if self
                    .start_move(id, self.pools[down].location, give_back)
                    .is_ok()
                {
                    return true;
                }
            }
            c.state.store(RESIDENT, Ordering::Release);
            return false;
        }
    }

    /// Brings spilled residents home, highest priority first, while their home tiers have free slots.
    fn bring_back(&self) {
        loop {
            let candidate = self
                .cells
                .iter()
                .enumerate()
                .filter(|(_, c)| {
                    let home = c.home.load(Ordering::Acquire);
                    c.state.load(Ordering::Acquire) == RESIDENT
                        && c.location.load(Ordering::Acquire) != home
                        && self.pool(LocationId(home)).is_some_and(|p| p.free() > 0)
                        && self.tier(LocationId(home))
                            < self.tier(LocationId(c.location.load(Ordering::Acquire)))
                })
                .max_by_key(|(_, c)| c.priority.load(Ordering::Acquire))
                .map(|(i, c)| (i as u32, c.generation.load(Ordering::Acquire)));
            let Some((index, generation)) = candidate else {
                return;
            };
            let c = &self.cells[index as usize];
            if c.state
                .compare_exchange(RESIDENT, MOVING, Ordering::AcqRel, Ordering::Acquire)
                .is_err()
            {
                continue;
            }
            let home = LocationId(c.home.load(Ordering::Acquire));
            if self
                .start_move(ResidentId { index, generation }, home, false)
                .is_err()
            {
                c.state.store(RESIDENT, Ordering::Release);
                return;
            }
        }
    }

    /// The give-back handler for the pools' owner (`OwnerSpec::on_give_back`): an order at a tier spills that tier's
    /// residents, lowest priority first, into the tiers below, and gives the freed slots' budget back. The slots
    /// come back (`TierPool::restore`) when [`Mover::restore`] finds room.
    pub fn give_back_handler(self: &Arc<Self>) -> GiveBackHandler {
        let weak = Arc::downgrade(self);
        Arc::new(move |order: GiveBack| {
            if let Some(m) = weak.upgrade() {
                m.give_back(order);
            }
        })
    }

    fn give_back(&self, order: GiveBack) {
        let Some(t) = self.tier(order.location) else {
            return;
        };
        let pool = &self.pools[t];
        let slots = order.bytes.div_ceil(pool.slot_bytes.max(1)) as u32;
        // Free slots go first, whoever the order names (the pool's owner holds them; residents' owners hold theirs
        // elastically too, and a free slot is cheaper to give than a spill).
        let retired = pool.retire_free(slots);
        // What free slots did not cover: spill residents (any priority: the order comes from above all of them); their
        // slots retire as the moves finish, the budget going back to the planner for the asker's next claim.
        for _ in retired..slots {
            if !self.evict(t, Priority(u16::MAX), true) {
                break;
            }
        }
    }

    /// Takes back the budget of retired slots where the locations have room again (bring-back's first step).
    pub fn restore(&self) -> u32 {
        self.pools
            .iter()
            .map(|p| p.restore(&self.planner, p.retired()))
            .sum()
    }

    fn take_done(&self) -> Option<(MoveHandler, MoveDone)> {
        self.done.1.poll()
    }
}

impl Drop for Mover {
    fn drop(&mut self) {
        // Moves still in flight are dropped with their slots' claims (the copies were the caller's to finish).
        while self.moves.1.poll().is_some() {}
        for c in self.cells.iter() {
            drop(c.take_claim());
            c.clear_handler();
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Completions as frames

/// A `LatticeSource` of move completions: each runs its resident owner's handler as a frame routed by the owner's
/// hash. Pulling also drives the moves ([`Mover::pump`]), so a lattice that asks for frames keeps the copies going.
pub struct Moves {
    mover: Weak<Mover>,
    downstream: OnceLock<Arc<dyn LatticeReceiver>>,
    demand: AtomicU64,
    complete: AtomicBool,
}

impl fmt::Debug for Moves {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Moves")
            .field("demand", &self.demand.load(Ordering::Acquire))
            .finish_non_exhaustive()
    }
}

impl Moves {
    fn new(mover: Weak<Mover>) -> Self {
        Moves {
            mover,
            downstream: OnceLock::new(),
            demand: AtomicU64::new(0),
            complete: AtomicBool::new(false),
        }
    }

    fn frame(handler: MoveHandler, done: MoveDone) -> Box<dyn Frame> {
        let id = euhedral_hashing::hasher_api::mix_with_zero_check(
            (u64::from(done.owner.slot) << 32) | u64::from(done.owner.generation),
        );
        Box::new(RunnableFrame::new(id, Arc::new(move || handler(done))))
    }

    fn emit(
        &self,
        consumer: &mut dyn FnMut(Box<dyn Frame>),
        stop: &mut dyn FnMut(&dyn Frame) -> bool,
        demand: u64,
    ) -> u64 {
        let Some(m) = self.mover.upgrade() else {
            return 0;
        };
        m.pump();
        let mut n = 0;
        while n < demand {
            let Some((handler, done)) = m.take_done() else {
                break;
            };
            let frame = Self::frame(handler.clone(), done);
            if stop(&*frame) {
                let _ = m.done.0.offer((handler, done));
                break;
            }
            consumer(frame);
            n += 1;
        }
        n
    }

    // HOT-FIX(S2): a producer pushes to the downstream itself, outside `request`.
    fn wake(&self) {
        let _hot = HotScope::enter();
        let Some(downstream) = self.downstream.get() else {
            return;
        };
        let demand = self.demand.load(Ordering::Acquire);
        if demand == 0 {
            return;
        }
        let Some(m) = self.mover.upgrade() else {
            return;
        };
        let mut emitted = 0;
        while emitted < demand {
            let Some((handler, done)) = m.take_done() else {
                break;
            };
            downstream.push(Self::frame(handler, done));
            emitted += 1;
        }
        if emitted > 0 {
            let _ = self
                .demand
                .try_update(Ordering::AcqRel, Ordering::Acquire, |d| {
                    Some(d.saturating_sub(emitted))
                });
        }
    }
}

impl LatticeSource for Moves {
    fn add_downstream(self: Arc<Self>, downstream: Arc<dyn LatticeReceiver>) {
        if self.complete.load(Ordering::Acquire) {
            downstream.on_complete();
            return;
        }
        if self.downstream.set(downstream.clone()).is_ok() {
            downstream.add_upstream(self);
        }
    }

    fn pull(
        &self,
        consumer: &mut dyn FnMut(Box<dyn Frame>),
        stop: &mut dyn FnMut(&dyn Frame) -> bool,
        demand: u64,
    ) -> u64 {
        self.emit(consumer, stop, demand)
    }

    fn request(&self, demand: u64) {
        if demand == 0 || self.complete.load(Ordering::Acquire) {
            return;
        }
        self.demand.fetch_add(demand, Ordering::AcqRel);
        self.wake();
    }

    fn complete(&self) {
        if !self.complete.swap(true, Ordering::AcqRel)
            && let Some(d) = self.downstream.get()
        {
            d.on_complete();
        }
    }

    fn is_complete(&self) -> bool {
        self.complete.load(Ordering::Acquire)
    }
}
