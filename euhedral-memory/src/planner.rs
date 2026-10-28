//! The planner: locations with budgets, owners with lifetimes and priorities, firm and elastic claims, give-back
//! orders, and the leak check. Lock-free: budgets and owner totals are atomics, owner slots are recycled through a
//! lock-free queue, and give-back orders go out through a [`GiveBacks`] lattice source.
//!
//! The planner only accounts and decides. It never allocates and never waits. Allocation is a backend's job
//! ([`crate::Backend`]); an allocation holds the [`Claim`] that accounts for it, and dropping the claim gives the bytes
//! back to the location.

use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{
    AtomicBool, AtomicPtr, AtomicU16, AtomicU32, AtomicU64, AtomicUsize, Ordering,
};

use euhedral_data_structures::queues::{Consumer, MpmcQueue, Multi, Producer};

/// The most locations a planner has (a device or two, pinned and pageable host memory).
pub const MAX_LOCATIONS: usize = 8;

/// A memory space the planner budgets: a device's memory, pinned host memory, pageable host memory, or another.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LocationId(pub u16);

/// What a location is. The planner treats every kind alike; the kind is for backends and reports.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LocationKind {
    /// Device memory of GPU `ordinal` (VRAM).
    Device { ordinal: u32 },
    /// Page-locked host memory (DMA-able by a device).
    HostPinned,
    /// Ordinary host memory.
    HostPageable,
    /// Anything else (a second NUMA node's memory, a file cache).
    Other(&'static str),
}

/// A location's budget.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LocationSpec {
    pub kind: LocationKind,
    /// Bytes the planner may hand out at this location, before the reserve: for a device, what it has free when the
    /// planner starts plus what this process already holds; for pinned host memory, what the host may lock.
    pub capacity: u64,
    /// Bytes always left free: other programs' share (the desktop, the compositor), the driver's own growth.
    pub reserve: u64,
}

/// How much an owner's memory matters: higher keeps its memory when another owner needs room.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub struct Priority(pub u16);

/// The lifetime an owner is tied to. [`Planner::end_scope`] ends every owner of a scope at once.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Scope {
    /// Lives as long as the process (a context, a kernel's constants).
    Process,
    /// An epoch or a session: ends when it ends.
    Epoch(u64),
    /// A built plan: ends when the plan is dropped.
    Plan(u64),
    /// A lease handed to another component (another process's share of a device): ends when given back.
    Lease(u64),
}

impl Scope {
    fn pack(self) -> (u8, u64) {
        match self {
            Scope::Process => (0, 0),
            Scope::Epoch(n) => (1, n),
            Scope::Plan(n) => (2, n),
            Scope::Lease(n) => (3, n),
        }
    }
}

/// What an elastic owner runs when the planner orders it to give memory back: free what it can (on its own lane),
/// then shrink its [`Elastic`] holdings. It runs as a frame of the [`GiveBacks`] source.
pub type GiveBackHandler = Arc<dyn Fn(GiveBack) + Send + Sync>;

/// An owner of memory.
#[derive(Clone)]
pub struct OwnerSpec {
    /// A name for reports: the component (a static name; the scope tells the instance).
    pub name: &'static str,
    /// The owner this one belongs to: ending the parent ends it.
    pub parent: Option<OwnerId>,
    pub priority: Priority,
    pub scope: Scope,
    /// Where give-back orders to this owner go (elastic owners).
    pub give_back: Option<GiveBackHandler>,
}

impl fmt::Debug for OwnerSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OwnerSpec")
            .field("name", &self.name)
            .field("parent", &self.parent)
            .field("priority", &self.priority)
            .field("scope", &self.scope)
            .field("give_back", &self.give_back.is_some())
            .finish()
    }
}

impl OwnerSpec {
    pub fn new(name: &'static str, priority: Priority, scope: Scope) -> Self {
        OwnerSpec {
            name,
            parent: None,
            priority,
            scope,
            give_back: None,
        }
    }

    pub fn child_of(mut self, parent: OwnerId) -> Self {
        self.parent = Some(parent);
        self
    }

    pub fn on_give_back(mut self, handler: GiveBackHandler) -> Self {
        self.give_back = Some(handler);
        self
    }
}

/// An owner: its slot and the slot's generation (a recycled slot is a different owner).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct OwnerId {
    pub slot: u32,
    pub generation: u32,
}

/// What a claim could not get.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Shortfall {
    pub location: LocationId,
    /// Bytes asked.
    pub asked: u64,
    /// Bytes free at the location (past its reserve) when asked.
    pub free: u64,
    /// Bytes that lower-priority owners hold elastically there (what give-back could free).
    pub reclaimable: u64,
}

impl fmt::Display for Shortfall {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "location {}: asked {} MiB, {} MiB free, {} MiB reclaimable",
            self.location.0,
            self.asked >> 20,
            self.free >> 20,
            self.reclaimable >> 20
        )
    }
}

impl std::error::Error for Shortfall {}

/// An order to an elastic owner: give back `bytes` at `location`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GiveBack {
    pub owner: OwnerId,
    pub location: LocationId,
    pub bytes: u64,
}

/// The outcome of [`Planner::claim_or_reclaim`].
#[derive(Debug)]
pub enum ClaimResult {
    Granted(Claim),
    /// Not free now; these give-back orders went out (as frames of the [`GiveBacks`] source) to lower-priority owners.
    /// Claim again once they freed it: the planner never waits.
    Reclaiming(Vec<GiveBack>),
    Refused(Shortfall),
}

/// One owner's live claims, as the leak check reports them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Leak {
    pub owner: &'static str,
    pub scope: Scope,
    pub location: LocationId,
    pub bytes: u64,
}

/// What an ended owner (or a baseline check) found still held.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LeakReport {
    pub leaks: Vec<Leak>,
}

impl LeakReport {
    pub fn is_clean(&self) -> bool {
        self.leaks.is_empty()
    }

    pub fn bytes(&self) -> u64 {
        self.leaks.iter().map(|l| l.bytes).sum()
    }
}

impl fmt::Display for LeakReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} leaks, {} bytes", self.leaks.len(), self.bytes())?;
        for l in self.leaks.iter().take(16) {
            write!(
                f,
                "; {} {:?}: {} B at {}",
                l.owner, l.scope, l.bytes, l.location.0
            )?;
        }
        Ok(())
    }
}

/// A location's accounting at one moment.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LocationUse {
    pub kind: LocationKind,
    pub capacity: u64,
    pub reserve: u64,
    /// Firm claims and elastic holdings together.
    pub used: u64,
    /// Elastic holdings.
    pub elastic: u64,
    /// Give-back ordered and not yet freed.
    pub ordered: u64,
    /// The most ever used at once.
    pub peak: u64,
}

impl LocationUse {
    /// Bytes free past the reserve.
    pub fn free(&self) -> u64 {
        self.capacity
            .saturating_sub(self.reserve)
            .saturating_sub(self.used)
    }
}

/// The planner's accounting at one moment: per location, and per live owner and location.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Snapshot {
    pub locations: Vec<LocationUse>,
    /// (owner name, scope, location, bytes held) for owners alive at the snapshot that hold something.
    pub owners: Vec<(&'static str, Scope, LocationId, u64)>,
}

impl Snapshot {
    pub fn used(&self, location: LocationId) -> u64 {
        self.locations
            .get(usize::from(location.0))
            .map_or(0, |l| l.used)
    }

    /// Bytes held by owners named `name` at `location`.
    pub fn owner(&self, name: &str, location: LocationId) -> u64 {
        self.owners
            .iter()
            .filter(|o| o.0 == name && o.2 == location)
            .map(|o| o.3)
            .sum()
    }
}

// ---------------------------------------------------------------------------------------------------------------
// State

#[derive(Debug)]
struct Loc {
    kind: LocationKind,
    reserve: u64,
    capacity: AtomicU64,
    used: AtomicU64,
    elastic: AtomicU64,
    ordered: AtomicU64,
    peak: AtomicU64,
}

impl Loc {
    fn free(&self) -> u64 {
        self.capacity
            .load(Ordering::Acquire)
            .saturating_sub(self.reserve)
            .saturating_sub(self.used.load(Ordering::Acquire))
    }

    /// Adds `bytes` to `used` if it stays within capacity less reserve. Lock-free (a CAS loop).
    fn try_take(&self, bytes: u64) -> Result<(), u64> {
        let mut used = self.used.load(Ordering::Acquire);
        loop {
            let limit = self
                .capacity
                .load(Ordering::Acquire)
                .saturating_sub(self.reserve);
            if used.saturating_add(bytes) > limit {
                return Err(limit.saturating_sub(used));
            }
            match self.used.compare_exchange_weak(
                used,
                used + bytes,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    self.peak.fetch_max(used + bytes, Ordering::AcqRel);
                    return Ok(());
                }
                Err(now) => used = now,
            }
        }
    }
}

/// `state` bits: 1 = live (opened and not ended); the rest counts claims and pins (<< 1).
const LIVE: u64 = 1;
const ONE: u64 = 2;
const NO_PARENT: u64 = u64::MAX;

struct Slot {
    generation: AtomicU32,
    state: AtomicU64,
    priority: AtomicU16,
    scope_kind: AtomicU32,
    scope_id: AtomicU64,
    parent: AtomicU64,
    name_ptr: AtomicPtr<u8>,
    name_len: AtomicUsize,
    /// `Arc<dyn Fn(GiveBack)>` boxed, or null: set at open, cleared at recycle (when nothing can read it).
    handler: AtomicPtr<GiveBackHandler>,
    held: [AtomicU64; MAX_LOCATIONS],
    elastic: [AtomicU64; MAX_LOCATIONS],
    ordered: [AtomicU64; MAX_LOCATIONS],
}

impl Default for Slot {
    fn default() -> Self {
        Slot {
            generation: AtomicU32::new(0),
            state: AtomicU64::new(0),
            priority: AtomicU16::new(0),
            scope_kind: AtomicU32::new(0),
            scope_id: AtomicU64::new(0),
            parent: AtomicU64::new(NO_PARENT),
            name_ptr: AtomicPtr::new(std::ptr::null_mut()),
            name_len: AtomicUsize::new(0),
            handler: AtomicPtr::new(std::ptr::null_mut()),
            held: Default::default(),
            elastic: Default::default(),
            ordered: Default::default(),
        }
    }
}

impl Slot {
    fn name(&self) -> &'static str {
        let p = self.name_ptr.load(Ordering::Acquire);
        if p.is_null() {
            return "";
        }
        let len = self.name_len.load(Ordering::Acquire);
        // SAFETY: (ptr, len) always come from one `&'static str` (written before the slot is published, read after;
        // a recycled slot's next name is another `&'static str`, so either is valid UTF-8 of 'static lifetime). The
        // pair is written as a unit while the slot is not live and nobody pins it.
        unsafe { std::str::from_utf8_unchecked(std::slice::from_raw_parts(p, len)) }
    }

    fn scope(&self) -> Scope {
        let id = self.scope_id.load(Ordering::Acquire);
        match self.scope_kind.load(Ordering::Acquire) {
            1 => Scope::Epoch(id),
            2 => Scope::Plan(id),
            3 => Scope::Lease(id),
            _ => Scope::Process,
        }
    }

    fn is(&self, id: OwnerId) -> bool {
        self.generation.load(Ordering::Acquire) == id.generation
    }

    fn live(&self) -> bool {
        self.state.load(Ordering::Acquire) & LIVE != 0
    }

    /// Pins a live slot of generation `generation` (as a claim does): it cannot be recycled until unpinned.
    fn pin(&self, generation: u32) -> bool {
        let mut s = self.state.load(Ordering::Acquire);
        loop {
            if s & LIVE == 0 || self.generation.load(Ordering::Acquire) != generation {
                return false;
            }
            match self
                .state
                .compare_exchange_weak(s, s + ONE, Ordering::AcqRel, Ordering::Acquire)
            {
                Ok(_) => {
                    // A slot recycled and reopened between the check and the CAS is another owner: let it go.
                    if self.generation.load(Ordering::Acquire) == generation {
                        return true;
                    }
                    self.state.fetch_sub(ONE, Ordering::AcqRel);
                    return false;
                }
                Err(now) => s = now,
            }
        }
    }

    /// Drops a pin; true when the slot is now ended and unpinned (the caller recycles it, exactly once).
    fn unpin_raw(&self) -> bool {
        let before = self.state.fetch_sub(ONE, Ordering::AcqRel);
        before == ONE
    }
}

/// A lock-free queue's two ends.
type Ends<T> = (Producer<T, Multi>, Consumer<T, Multi>);

/// The planner of one process: shared by every component through an `Arc`.
pub struct Planner {
    locations: Box<[Loc]>,
    slots: Box<[Slot]>,
    fresh: AtomicU32,
    free_slots: (Producer<u32, Multi>, Consumer<u32, Multi>),
    orders: Ends<(GiveBackHandler, GiveBack)>,
    orders_pending: AtomicU64,
    calls: crate::arena::BackendCalls,
    source: Arc<GiveBacks>,
}

impl fmt::Debug for Planner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Planner")
            .field("locations", &self.locations)
            .finish_non_exhaustive()
    }
}

/// Owners a planner holds at once by default (slots are recycled when an ended owner holds nothing).
pub const DEFAULT_OWNERS: usize = 1024;

impl Planner {
    pub fn new(locations: impl IntoIterator<Item = LocationSpec>) -> Arc<Self> {
        Self::with_owners(locations, DEFAULT_OWNERS)
    }

    pub fn with_owners(
        locations: impl IntoIterator<Item = LocationSpec>,
        owners: usize,
    ) -> Arc<Self> {
        let locations: Box<[Loc]> = locations
            .into_iter()
            .map(|s| Loc {
                kind: s.kind,
                reserve: s.reserve,
                capacity: AtomicU64::new(s.capacity),
                used: AtomicU64::new(0),
                elastic: AtomicU64::new(0),
                ordered: AtomicU64::new(0),
                peak: AtomicU64::new(0),
            })
            .collect();
        assert!(
            locations.len() <= MAX_LOCATIONS,
            "at most {MAX_LOCATIONS} locations"
        );
        let slots = (0..owners).map(|_| Slot::default()).collect();
        let free = MpmcQueue::<u32>::new(256);
        let orders = MpmcQueue::<(GiveBackHandler, GiveBack)>::new(64);
        Arc::new_cyclic(|weak| Planner {
            locations,
            slots,
            fresh: AtomicU32::new(0),
            free_slots: (free.producer(), free.consumer()),
            orders: (orders.producer(), orders.consumer()),
            orders_pending: AtomicU64::new(0),
            calls: crate::arena::BackendCalls::default(),
            source: Arc::new(GiveBacks::new(weak.clone())),
        })
    }

    fn loc(&self, id: LocationId) -> &Loc {
        &self.locations[usize::from(id.0)]
    }

    fn slot(&self, id: OwnerId) -> &Slot {
        &self.slots[id.slot as usize]
    }

    /// The real allocations and frees (backend calls) so far: allocation backends and adapters count each one
    /// here, so the steady state can be checked to make none.
    pub fn calls(&self) -> &crate::arena::BackendCalls {
        &self.calls
    }

    /// The first location of `kind`.
    pub fn location(&self, kind: LocationKind) -> Option<LocationId> {
        self.locations
            .iter()
            .position(|l| l.kind == kind)
            .map(|i| LocationId(i as u16))
    }

    pub fn locations(&self) -> usize {
        self.locations.len()
    }

    /// The location's capacity changed outside the planner (another process took or gave back memory).
    pub fn set_capacity(&self, location: LocationId, capacity: u64) {
        self.loc(location)
            .capacity
            .store(capacity, Ordering::Release);
    }

    /// Bytes free at `location` past its reserve.
    pub fn free(&self, location: LocationId) -> u64 {
        self.loc(location).free()
    }

    /// Opens an owner. Panics when every slot is held (raise `with_owners`): owners are bounded so the planner never
    /// allocates on a claim path.
    pub fn open_owner(&self, spec: OwnerSpec) -> OwnerId {
        let index = match self.free_slots.1.poll() {
            Some(i) => i,
            None => {
                let i = self.fresh.fetch_add(1, Ordering::AcqRel);
                assert!(
                    (i as usize) < self.slots.len(),
                    "the planner's {} owner slots are all held",
                    self.slots.len()
                );
                i
            }
        };
        let s = &self.slots[index as usize];
        // The slot is not live and nobody pins it (it was fresh or recycled): write its fields, then publish.
        let generation = s.generation.load(Ordering::Acquire);
        s.priority.store(spec.priority.0, Ordering::Release);
        let (kind, id) = spec.scope.pack();
        s.scope_kind.store(u32::from(kind), Ordering::Release);
        s.scope_id.store(id, Ordering::Release);
        s.parent.store(
            spec.parent.map_or(NO_PARENT, |p| {
                (u64::from(p.slot) << 32) | u64::from(p.generation)
            }),
            Ordering::Release,
        );
        s.name_ptr
            .store(spec.name.as_ptr().cast_mut(), Ordering::Release);
        s.name_len.store(spec.name.len(), Ordering::Release);
        let handler = spec
            .give_back
            .map_or(std::ptr::null_mut(), |h| Box::into_raw(Box::new(h)));
        s.handler.store(handler, Ordering::Release);
        s.state.store(LIVE, Ordering::Release);
        OwnerId {
            slot: index,
            generation,
        }
    }

    fn priority(&self, owner: OwnerId) -> Priority {
        Priority(self.slot(owner).priority.load(Ordering::Acquire))
    }

    /// Elastic bytes at `location` held by live owners of priority below `priority`, less what is already ordered.
    fn reclaimable(&self, location: LocationId, priority: Priority) -> u64 {
        let l = usize::from(location.0);
        let n = self.fresh.load(Ordering::Acquire) as usize;
        self.slots[..n.min(self.slots.len())]
            .iter()
            .filter(|s| s.live() && s.priority.load(Ordering::Acquire) < priority.0)
            .map(|s| {
                s.elastic[l]
                    .load(Ordering::Acquire)
                    .saturating_sub(s.ordered[l].load(Ordering::Acquire))
            })
            .sum()
    }

    fn shortfall(&self, owner: OwnerId, location: LocationId, asked: u64, free: u64) -> Shortfall {
        Shortfall {
            location,
            asked,
            free,
            reclaimable: self.reclaimable(location, self.priority(owner)),
        }
    }

    /// Accounts `bytes` already taken from the location to `owner` (pinning its slot), as a claim.
    fn make_claim(
        self: &Arc<Self>,
        owner: OwnerId,
        location: LocationId,
        bytes: u64,
        elastic: bool,
    ) -> Option<Claim> {
        let s = self.slot(owner);
        if !s.pin(owner.generation) {
            return None;
        }
        let l = usize::from(location.0);
        s.held[l].fetch_add(bytes, Ordering::AcqRel);
        if elastic {
            s.elastic[l].fetch_add(bytes, Ordering::AcqRel);
            self.loc(location)
                .elastic
                .fetch_add(bytes, Ordering::AcqRel);
        }
        Some(Claim {
            planner: self.clone(),
            owner,
            location,
            bytes,
            elastic,
        })
    }

    /// Accounts memory that exists already (allocated outside a plan: a driver's opaque object, a component not yet
    /// planned): always granted, even past the budget, so the books match the device. Over-budget accounting shows as a
    /// location's `used` above its capacity, which the next admission sees.
    pub fn account(
        self: &Arc<Self>,
        owner: OwnerId,
        location: LocationId,
        bytes: u64,
    ) -> Option<Claim> {
        let loc = self.loc(location);
        let now = loc.used.fetch_add(bytes, Ordering::AcqRel) + bytes;
        loc.peak.fetch_max(now, Ordering::AcqRel);
        self.make_claim(owner, location, bytes, false).or_else(|| {
            loc.used.fetch_sub(bytes, Ordering::AcqRel);
            None
        })
    }

    /// A firm claim from what is free: never takes from another owner. `tag` names it for debugging.
    pub fn claim(
        self: &Arc<Self>,
        owner: OwnerId,
        location: LocationId,
        bytes: u64,
        _tag: &'static str,
    ) -> Result<Claim, Shortfall> {
        let loc = self.loc(location);
        if let Err(free) = loc.try_take(bytes) {
            return Err(self.shortfall(owner, location, bytes, free));
        }
        self.make_claim(owner, location, bytes, false)
            .ok_or_else(|| {
                loc.used.fetch_sub(bytes, Ordering::AcqRel);
                Shortfall {
                    location,
                    asked: bytes,
                    free: loc.free(),
                    reclaimable: 0,
                }
            })
    }

    /// A firm claim that may take from lower-priority elastic owners: granted if free, else give-back orders go out
    /// (lowest priority first, then the biggest holder) and the caller claims again once they freed it, or refused if
    /// even that would not cover it.
    pub fn claim_or_reclaim(
        self: &Arc<Self>,
        owner: OwnerId,
        location: LocationId,
        bytes: u64,
        tag: &'static str,
    ) -> ClaimResult {
        let free = match self.claim(owner, location, bytes, tag) {
            Ok(c) => return ClaimResult::Granted(c),
            Err(s) => s.free,
        };
        let loc = self.loc(location);
        let priority = self.priority(owner);
        let ordered_already = loc.ordered.load(Ordering::Acquire);
        let reclaimable = self.reclaimable(location, priority);
        if bytes > free + reclaimable + ordered_already {
            return ClaimResult::Refused(Shortfall {
                location,
                asked: bytes,
                free,
                reclaimable,
            });
        }
        let l = usize::from(location.0);
        let n = (self.fresh.load(Ordering::Acquire) as usize).min(self.slots.len());
        let mut holders: Vec<(u16, u64, u32)> = (0..n)
            .filter_map(|i| {
                let s = &self.slots[i];
                let p = s.priority.load(Ordering::Acquire);
                let left = s.elastic[l]
                    .load(Ordering::Acquire)
                    .saturating_sub(s.ordered[l].load(Ordering::Acquire));
                (s.live() && p < priority.0 && left > 0).then_some((p, left, i as u32))
            })
            .collect();
        holders.sort_by(|a, b| a.0.cmp(&b.0).then(b.1.cmp(&a.1)));
        let mut left = (bytes - free).saturating_sub(ordered_already);
        let mut orders = Vec::new();
        for (_, has, index) in holders {
            if left == 0 {
                break;
            }
            let s = &self.slots[index as usize];
            let generation = s.generation.load(Ordering::Acquire);
            // Pinned while its handler is read and cloned: the slot cannot be recycled (and its handler freed) meanwhile.
            if !s.pin(generation) {
                continue;
            }
            let take = has.min(left);
            s.ordered[l].fetch_add(take, Ordering::AcqRel);
            loc.ordered.fetch_add(take, Ordering::AcqRel);
            left -= take;
            let order = GiveBack {
                owner: OwnerId {
                    slot: index,
                    generation,
                },
                location,
                bytes: take,
            };
            let h = s.handler.load(Ordering::Acquire);
            if !h.is_null() {
                // SAFETY: the handler box lives until the slot is recycled, which our pin prevents.
                let handler = unsafe { (*h).clone() };
                if self.orders.0.offer((handler, order.clone())).is_ok() {
                    self.orders_pending.fetch_add(1, Ordering::AcqRel);
                    self.source.wake();
                }
            }
            self.unpin(OwnerId {
                slot: index,
                generation,
            });
            orders.push(order);
        }
        ClaimResult::Reclaiming(orders)
    }

    /// An elastic claim: at least `min` bytes (else refused), and as much more up to `max` as is free. It can grow
    /// later and gives back on order (its owner's give-back handler runs as a frame of [`Planner::give_backs`]).
    pub fn elastic(
        self: &Arc<Self>,
        owner: OwnerId,
        location: LocationId,
        min: u64,
        max: u64,
        _tag: &'static str,
    ) -> Result<Elastic, Shortfall> {
        let loc = self.loc(location);
        loop {
            let free = loc.free();
            if min > free {
                return Err(self.shortfall(owner, location, min, free));
            }
            let bytes = max.min(free).max(min);
            if loc.try_take(bytes).is_ok() {
                let claim = self
                    .make_claim(owner, location, bytes, true)
                    .ok_or_else(|| {
                        loc.used.fetch_sub(bytes, Ordering::AcqRel);
                        Shortfall {
                            location,
                            asked: min,
                            free,
                            reclaimable: 0,
                        }
                    })?;
                return Ok(Elastic { claim });
            }
        }
    }

    /// Give-back ordered from `owner` at `location` and not yet freed.
    pub fn ordered(&self, owner: OwnerId, location: LocationId) -> u64 {
        let s = self.slot(owner);
        if !s.is(owner) {
            return 0;
        }
        s.ordered[usize::from(location.0)].load(Ordering::Acquire)
    }

    /// The source of give-back frames: add it upstream of a lattice (`add_downstream`) and each order runs its
    /// owner's handler as a frame there.
    pub fn give_backs(&self) -> Arc<GiveBacks> {
        self.source.clone()
    }

    pub(crate) fn take_order(&self) -> Option<(GiveBackHandler, GiveBack)> {
        let o = self.orders.1.poll();
        if o.is_some() {
            self.orders_pending.fetch_sub(1, Ordering::AcqRel);
        }
        o
    }

    pub(crate) fn requeue_order(&self, o: (GiveBackHandler, GiveBack)) {
        if self.orders.0.offer(o).is_ok() {
            self.orders_pending.fetch_add(1, Ordering::AcqRel);
        }
    }

    /// Orders queued and not yet emitted as frames.
    pub fn orders_pending(&self) -> u64 {
        self.orders_pending.load(Ordering::Acquire)
    }

    /// Ends `owner` and every live descendant. Whatever they still hold is a leak: it is reported (and stays
    /// accounted until its claims drop, since the memory is still allocated); the slot is recycled after that.
    pub fn end_owner(&self, owner: OwnerId) -> LeakReport {
        let mut report = LeakReport::default();
        self.end_tree(owner, &mut report);
        report
    }

    fn end_tree(&self, owner: OwnerId, report: &mut LeakReport) {
        let s = self.slot(owner);
        if !s.is(owner) || !s.live() {
            return;
        }
        let packed = (u64::from(owner.slot) << 32) | u64::from(owner.generation);
        let n = (self.fresh.load(Ordering::Acquire) as usize).min(self.slots.len());
        for i in 0..n {
            let c = &self.slots[i];
            if c.live() && c.parent.load(Ordering::Acquire) == packed {
                let child = OwnerId {
                    slot: i as u32,
                    generation: c.generation.load(Ordering::Acquire),
                };
                self.end_tree(child, report);
            }
        }
        // Take the live bit; whoever drops the last claim (or we, if none) recycles the slot.
        let before = s.state.fetch_and(!LIVE, Ordering::AcqRel);
        if before & LIVE == 0 {
            return;
        }
        for (l, held) in s.held.iter().enumerate().take(self.locations.len()) {
            let b = held.load(Ordering::Acquire);
            if b > 0 {
                report.leaks.push(Leak {
                    owner: s.name(),
                    scope: s.scope(),
                    location: LocationId(l as u16),
                    bytes: b,
                });
            }
        }
        if before == LIVE {
            self.recycle(owner.slot);
        }
    }

    /// Ends every live owner of `scope` (and their descendants).
    pub fn end_scope(&self, scope: Scope) -> LeakReport {
        let mut report = LeakReport::default();
        let n = (self.fresh.load(Ordering::Acquire) as usize).min(self.slots.len());
        for i in 0..n {
            let s = &self.slots[i];
            if s.live() && s.scope() == scope {
                let id = OwnerId {
                    slot: i as u32,
                    generation: s.generation.load(Ordering::Acquire),
                };
                self.end_tree(id, &mut report);
            }
        }
        report
    }

    fn unpin(&self, owner: OwnerId) {
        if self.slot(owner).unpin_raw() {
            self.recycle(owner.slot);
        }
    }

    /// The slot is ended and unpinned: nobody reads it. Clear it and hand it back.
    fn recycle(&self, index: u32) {
        let s = &self.slots[index as usize];
        for l in 0..self.locations.len() {
            let o = s.ordered[l].swap(0, Ordering::AcqRel);
            if o > 0 {
                self.locations[l].ordered.fetch_sub(o, Ordering::AcqRel);
            }
        }
        let h = s.handler.swap(std::ptr::null_mut(), Ordering::AcqRel);
        if !h.is_null() {
            // SAFETY: the box made at open; no pin holds the slot, so nobody reads it.
            drop(unsafe { Box::from_raw(h) });
        }
        s.generation.fetch_add(1, Ordering::AcqRel);
        let _ = self.free_slots.0.offer(index);
    }

    pub fn snapshot(&self) -> Snapshot {
        let locations = self
            .locations
            .iter()
            .map(|l| LocationUse {
                kind: l.kind,
                capacity: l.capacity.load(Ordering::Acquire),
                reserve: l.reserve,
                used: l.used.load(Ordering::Acquire),
                elastic: l.elastic.load(Ordering::Acquire),
                ordered: l.ordered.load(Ordering::Acquire),
                peak: l.peak.load(Ordering::Acquire),
            })
            .collect();
        let mut owners = Vec::new();
        let n = (self.fresh.load(Ordering::Acquire) as usize).min(self.slots.len());
        for s in &self.slots[..n] {
            if !s.live() {
                continue;
            }
            for (l, held) in s.held.iter().enumerate().take(self.locations.len()) {
                let b = held.load(Ordering::Acquire);
                if b > 0 {
                    owners.push((s.name(), s.scope(), LocationId(l as u16), b));
                }
            }
        }
        Snapshot { locations, owners }
    }

    /// The leak check against an earlier snapshot (a baseline): every byte still held by an ended owner, and every
    /// location above its baseline use, is reported.
    pub fn check_baseline(&self, baseline: &Snapshot) -> LeakReport {
        let mut report = LeakReport::default();
        let n = (self.fresh.load(Ordering::Acquire) as usize).min(self.slots.len());
        for s in &self.slots[..n] {
            if s.live() || s.state.load(Ordering::Acquire) == 0 {
                continue;
            }
            for (l, held) in s.held.iter().enumerate().take(self.locations.len()) {
                let b = held.load(Ordering::Acquire);
                if b > 0 {
                    report.leaks.push(Leak {
                        owner: s.name(),
                        scope: s.scope(),
                        location: LocationId(l as u16),
                        bytes: b,
                    });
                }
            }
        }
        for (n, l) in self.locations.iter().enumerate() {
            let base = baseline.locations.get(n).map_or(0, |b| b.used);
            let now = l.used.load(Ordering::Acquire);
            let location = LocationId(n as u16);
            if now > base && report.leaks.iter().all(|x| x.location != location) {
                report.leaks.push(Leak {
                    owner: "(live owners above the baseline)",
                    scope: Scope::Process,
                    location,
                    bytes: now - base,
                });
            }
        }
        report
    }

    /// For admission: free bytes and reclaimable bytes per location for an owner of `priority`.
    pub(crate) fn room(&self, priority: Priority) -> Vec<(u64, u64)> {
        (0..self.locations.len())
            .map(|n| {
                (
                    self.locations[n].free(),
                    self.reclaimable(LocationId(n as u16), priority),
                )
            })
            .collect()
    }

    pub(crate) fn priority_of(&self, owner: OwnerId) -> Priority {
        self.priority(owner)
    }

    /// Takes `bytes[n]` at every location n at once, or nothing (each location by CAS; a failure rolls back).
    pub(crate) fn reserve_many(
        self: &Arc<Self>,
        owner: OwnerId,
        bytes: &[u64],
    ) -> Result<Vec<Option<Claim>>, Shortfall> {
        let mut taken = Vec::new();
        for (n, &b) in bytes.iter().enumerate() {
            let location = LocationId(n as u16);
            if b == 0 {
                taken.push(None);
                continue;
            }
            // On an error `taken` drops: the earlier locations are given back.
            taken.push(Some(self.claim(owner, location, b, "reservation")?));
        }
        Ok(taken)
    }

    fn release(&self, owner: OwnerId, location: LocationId, bytes: u64, elastic: bool) {
        self.unhold(owner, location, bytes, elastic);
        self.loc(location).used.fetch_sub(bytes, Ordering::AcqRel);
    }

    /// Takes `bytes` off `owner`'s account at `location` (and off its elastic holdings and open order if `elastic`),
    /// leaving the location's total as it is: the bytes are released, or moved to another owner's account.
    fn unhold(&self, owner: OwnerId, location: LocationId, bytes: u64, elastic: bool) {
        let l = usize::from(location.0);
        let s = self.slot(owner);
        let loc = self.loc(location);
        s.held[l].fetch_sub(bytes, Ordering::AcqRel);
        if elastic {
            s.elastic[l].fetch_sub(bytes, Ordering::AcqRel);
            loc.elastic.fetch_sub(bytes, Ordering::AcqRel);
            // Freed bytes count against an open order first.
            let mut o = s.ordered[l].load(Ordering::Acquire);
            while o > 0 {
                let done = o.min(bytes);
                match s.ordered[l].compare_exchange_weak(
                    o,
                    o - done,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                ) {
                    Ok(_) => {
                        loc.ordered.fetch_sub(done, Ordering::AcqRel);
                        break;
                    }
                    Err(now) => o = now,
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Claims

/// Bytes accounted to an owner at a location. Dropping it gives them back: keep it with the memory it accounts for
/// and drop both when the memory is really freed (after the device stopped reading it).
pub struct Claim {
    planner: Arc<Planner>,
    owner: OwnerId,
    location: LocationId,
    bytes: u64,
    elastic: bool,
}

impl fmt::Debug for Claim {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Claim")
            .field("owner", &self.owner)
            .field("location", &self.location)
            .field("bytes", &self.bytes)
            .finish()
    }
}

impl Claim {
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    pub fn location(&self) -> LocationId {
        self.location
    }

    pub fn owner(&self) -> OwnerId {
        self.owner
    }

    /// Moves `bytes` of this claim to a new claim of `owner` (a plan's reservation handing an allocation its share).
    /// `None` if this claim has fewer bytes or `owner` is not live.
    pub fn take(&mut self, owner: OwnerId, bytes: u64, _tag: &'static str) -> Option<Claim> {
        if bytes > self.bytes || self.elastic {
            return None;
        }
        // The location's total does not change: the bytes move from one owner to the other.
        let new = self
            .planner
            .make_claim(owner, self.location, bytes, false)?;
        let l = usize::from(self.location.0);
        self.planner.slot(self.owner).held[l].fetch_sub(bytes, Ordering::AcqRel);
        self.bytes -= bytes;
        Some(new)
    }

    /// Moves all of this claim's bytes to `to`'s account at the same location, as a firm or an elastic holding. The
    /// location's total does not change; `to`'s total grows before this owner's shrinks. `Err(self)` when `to` is not
    /// live.
    pub fn transfer(mut self, to: OwnerId, elastic: bool) -> Result<Claim, Claim> {
        let Some(new) = self
            .planner
            .make_claim(to, self.location, self.bytes, elastic)
        else {
            return Err(self);
        };
        self.planner
            .unhold(self.owner, self.location, self.bytes, self.elastic);
        self.bytes = 0;
        Ok(new)
    }

    /// Whether the bytes are an elastic holding (counted as reclaimable by higher priorities).
    pub fn is_elastic(&self) -> bool {
        self.elastic
    }

    /// Gives `bytes` of this claim back now (a reservation trimmed to what the plan really allocated).
    pub fn shrink(&mut self, bytes: u64) {
        let b = bytes.min(self.bytes);
        self.planner
            .release(self.owner, self.location, b, self.elastic);
        self.bytes -= b;
    }
}

impl Drop for Claim {
    fn drop(&mut self) {
        self.planner
            .release(self.owner, self.location, self.bytes, self.elastic);
        self.planner.unpin(self.owner);
    }
}

/// An elastic holding: grows into free memory, shrinks on order.
#[derive(Debug)]
pub struct Elastic {
    claim: Claim,
}

impl Elastic {
    pub fn bytes(&self) -> u64 {
        self.claim.bytes
    }

    pub fn location(&self) -> LocationId {
        self.claim.location
    }

    pub fn owner(&self) -> OwnerId {
        self.claim.owner
    }

    /// Give-back ordered from this holding's owner and not yet done.
    pub fn ordered(&self) -> u64 {
        self.claim
            .planner
            .ordered(self.claim.owner, self.claim.location)
    }

    /// Grows toward `target` bytes as far as the free memory allows; returns the new size. Never grows while a
    /// give-back is ordered from the owner.
    pub fn grow_to(&mut self, target: u64) -> u64 {
        if target <= self.claim.bytes || self.ordered() > 0 {
            return self.claim.bytes;
        }
        let p = self.claim.planner.clone();
        let loc = p.loc(self.claim.location);
        let want = target - self.claim.bytes;
        let got = loop {
            let free = loc.free();
            let take = want.min(free);
            if take == 0 || loc.try_take(take).is_ok() {
                break take;
            }
        };
        if got > 0 {
            let l = usize::from(self.claim.location.0);
            let s = p.slot(self.claim.owner);
            s.held[l].fetch_add(got, Ordering::AcqRel);
            s.elastic[l].fetch_add(got, Ordering::AcqRel);
            loc.elastic.fetch_add(got, Ordering::AcqRel);
            self.claim.bytes += got;
        }
        self.claim.bytes
    }

    /// Shrinks to `bytes` (the memory above it was freed); counts against an ordered give-back first.
    /// The holding as a plain claim (still elastic: reclaimable, and counted against its owner's orders when freed).
    pub fn into_claim(self) -> Claim {
        self.claim
    }

    pub fn shrink_to(&mut self, bytes: u64) {
        if bytes < self.claim.bytes {
            let b = self.claim.bytes - bytes;
            self.claim.shrink(b);
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Give-back orders as frames

use euhedral_core::frames::{Frame, RunnableFrame};
use euhedral_core::generics::{LatticeReceiver, LatticeSource};

/// A `LatticeSource` of give-back frames: each order runs its owner's handler on whichever worker the frame is
/// routed to (ordered per owner: the frame's id hash is the owner's). Nothing polls the planner: an order wakes the
/// downstream lattice with a push when it has demand.
pub struct GiveBacks {
    planner: std::sync::Weak<Planner>,
    downstream: std::sync::OnceLock<Arc<dyn LatticeReceiver>>,
    demand: AtomicU64,
    complete: AtomicBool,
}

impl fmt::Debug for GiveBacks {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GiveBacks")
            .field("demand", &self.demand.load(Ordering::Acquire))
            .finish_non_exhaustive()
    }
}

impl GiveBacks {
    fn new(planner: std::sync::Weak<Planner>) -> Self {
        GiveBacks {
            planner,
            downstream: std::sync::OnceLock::new(),
            demand: AtomicU64::new(0),
            complete: AtomicBool::new(false),
        }
    }

    fn frame(handler: GiveBackHandler, order: GiveBack) -> Box<dyn Frame> {
        let id = euhedral_hashing::hasher_api::mix_with_zero_check(
            (u64::from(order.owner.slot) << 32) | u64::from(order.owner.generation),
        );
        Box::new(RunnableFrame::new(
            id,
            Arc::new(move || handler(order.clone())),
        ))
    }

    /// Emits up to `demand` order frames into `consumer`, stopping before one `stop` refuses.
    fn emit(
        &self,
        consumer: &mut dyn FnMut(Box<dyn Frame>),
        stop: &mut dyn FnMut(&dyn Frame) -> bool,
        demand: u64,
    ) -> u64 {
        let Some(p) = self.planner.upgrade() else {
            return 0;
        };
        let mut n = 0;
        while n < demand {
            let Some((handler, order)) = p.take_order() else {
                break;
            };
            let frame = Self::frame(handler.clone(), order.clone());
            if stop(&*frame) {
                p.requeue_order((handler, order));
                break;
            }
            consumer(frame);
            n += 1;
        }
        n
    }

    /// An order was queued: push it downstream if the lattice asked for frames.
    fn wake(&self) {
        let Some(downstream) = self.downstream.get() else {
            return;
        };
        let demand = self.demand.load(Ordering::Acquire);
        if demand == 0 {
            return;
        }
        let emitted = self.emit(&mut |f| downstream.push(f), &mut |_| false, demand);
        if emitted > 0 {
            let _ = self
                .demand
                .try_update(Ordering::AcqRel, Ordering::Acquire, |d| {
                    Some(d.saturating_sub(emitted))
                });
        }
    }
}

impl LatticeSource for GiveBacks {
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
