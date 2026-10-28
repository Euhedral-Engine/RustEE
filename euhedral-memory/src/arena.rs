//! Plan once, allocate once, reuse: arenas cut from one planned allocation.
//!
//! At the start of a session (an epoch or an admitted plan) the working set is allocated once, by a backend, and
//! accounted by one [`Claim`]. Everything inside it is handed out and back without the driver:
//! - [`SlotArena`]: equal slots (pools of equal buffers, staging, pinned host slots), leased from a
//!   lock-free free list;
//! - [`ScratchArena`]: power-of-two sub-allocation (scratch, parameters, statistics, readback records), leased from
//!   per-class lock-free free lists over a bump pointer.
//!
//! A lease is a value: dropping it returns the slot. In the steady state no allocation or free reaches a backend; the
//! real frees happen when the arena itself drops (session end, size change, idle release), counted by [`BackendCalls`].

use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use euhedral_data_structures::queues::{Consumer, MpmcQueue, Multi, Producer};

use crate::planner::Claim;

/// Counts the real allocations and frees (backend calls), so a test can require none in the steady state.
#[derive(Debug, Default)]
pub struct BackendCalls {
    allocs: AtomicU64,
    frees: AtomicU64,
    alloc_bytes: AtomicU64,
}

impl BackendCalls {
    pub fn alloc(&self, bytes: u64) {
        self.allocs.fetch_add(1, Ordering::Relaxed);
        self.alloc_bytes.fetch_add(bytes, Ordering::Relaxed);
    }

    pub fn free(&self) {
        self.frees.fetch_add(1, Ordering::Relaxed);
    }

    /// (allocations, frees, bytes allocated) so far.
    pub fn get(&self) -> (u64, u64, u64) {
        (
            self.allocs.load(Ordering::Relaxed),
            self.frees.load(Ordering::Relaxed),
            self.alloc_bytes.load(Ordering::Relaxed),
        )
    }
}

/// Equal slots of one planned allocation.
pub struct SlotArena {
    base: u64,
    slot_bytes: u64,
    count: u32,
    leased: AtomicU32,
    free: (Producer<u32, Multi>, Consumer<u32, Multi>),
    _claim: Claim,
}

impl fmt::Debug for SlotArena {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SlotArena")
            .field("slot_bytes", &self.slot_bytes)
            .field("count", &self.count)
            .field("leased", &self.leased())
            .finish()
    }
}

impl SlotArena {
    /// `count` slots of `slot_bytes` from `base` (an address or an offset into the allocation `claim` accounts for).
    pub fn new(base: u64, slot_bytes: u64, count: u32, claim: Claim) -> Arc<Self> {
        let q = MpmcQueue::<u32>::new((count as usize).next_power_of_two().max(16));
        let free = (q.producer(), q.consumer());
        for i in 0..count {
            let _ = free.0.offer(i);
        }
        Arc::new(SlotArena {
            base,
            slot_bytes,
            count,
            leased: AtomicU32::new(0),
            free,
            _claim: claim,
        })
    }

    /// A free slot, or `None` when every slot is leased (the plan sized the pool: back-pressure, not an allocation).
    pub fn lease(self: &Arc<Self>) -> Option<SlotLease> {
        let index = self.free.1.poll()?;
        self.leased.fetch_add(1, Ordering::AcqRel);
        Some(SlotLease {
            arena: self.clone(),
            index,
        })
    }

    pub fn capacity(&self) -> u32 {
        self.count
    }

    pub fn leased(&self) -> u32 {
        self.leased.load(Ordering::Acquire)
    }

    pub fn slot_bytes(&self) -> u64 {
        self.slot_bytes
    }

    pub fn address(&self, index: u32) -> u64 {
        self.base + u64::from(index) * self.slot_bytes
    }
}

/// A leased slot; dropping it returns the slot.
pub struct SlotLease {
    arena: Arc<SlotArena>,
    index: u32,
}

impl fmt::Debug for SlotLease {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SlotLease")
            .field("index", &self.index)
            .finish()
    }
}

impl SlotLease {
    pub fn index(&self) -> u32 {
        self.index
    }

    pub fn address(&self) -> u64 {
        self.arena.address(self.index)
    }
}

impl Drop for SlotLease {
    fn drop(&mut self) {
        self.arena.leased.fetch_sub(1, Ordering::AcqRel);
        let _ = self.arena.free.0.offer(self.index);
    }
}

/// The smallest scratch class (256 B: a device allocation's alignment).
pub const SCRATCH_MIN: u64 = 256;
const SCRATCH_CLASSES: usize = 40;

/// Power-of-two sub-allocation from one planned allocation: per-class free lists over a bump pointer. Released
/// pieces are reused by their class; the block is never compacted (a plan's scratch is the same for every unit of work).
pub struct ScratchArena {
    base: u64,
    bytes: u64,
    top: AtomicU64,
    in_use: AtomicU64,
    classes: Vec<(Producer<u64, Multi>, Consumer<u64, Multi>)>,
    _claim: Claim,
}

impl fmt::Debug for ScratchArena {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ScratchArena")
            .field("bytes", &self.bytes)
            .field("top", &self.top.load(Ordering::Acquire))
            .field("in_use", &self.in_use())
            .finish()
    }
}

impl ScratchArena {
    pub fn new(base: u64, bytes: u64, claim: Claim) -> Arc<Self> {
        let classes = (0..SCRATCH_CLASSES)
            .map(|_| {
                let q = MpmcQueue::<u64>::new(16);
                (q.producer(), q.consumer())
            })
            .collect();
        Arc::new(ScratchArena {
            base,
            bytes,
            top: AtomicU64::new(0),
            in_use: AtomicU64::new(0),
            classes,
            _claim: claim,
        })
    }

    pub fn class_of(bytes: u64) -> u64 {
        bytes.max(SCRATCH_MIN).next_power_of_two()
    }

    fn index(class: u64) -> usize {
        (class.trailing_zeros() - SCRATCH_MIN.trailing_zeros()) as usize
    }

    /// A piece of at least `bytes`, or `None` when the block is used up (the plan undersized it: a bug to report).
    pub fn alloc(self: &Arc<Self>, bytes: u64) -> Option<ScratchLease> {
        let class = Self::class_of(bytes);
        let i = Self::index(class);
        if i >= SCRATCH_CLASSES {
            return None;
        }
        let offset = match self.classes[i].1.poll() {
            Some(o) => o,
            None => {
                let mut top = self.top.load(Ordering::Acquire);
                loop {
                    let at = top.next_multiple_of(class.min(1 << 16));
                    if at + class > self.bytes {
                        return None;
                    }
                    match self.top.compare_exchange_weak(
                        top,
                        at + class,
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    ) {
                        Ok(_) => break at,
                        Err(now) => top = now,
                    }
                }
            }
        };
        self.in_use.fetch_add(class, Ordering::AcqRel);
        Some(ScratchLease {
            arena: self.clone(),
            offset,
            class,
        })
    }

    pub fn in_use(&self) -> u64 {
        self.in_use.load(Ordering::Acquire)
    }

    pub fn bytes(&self) -> u64 {
        self.bytes
    }
}

/// A piece of a scratch arena; dropping it returns the piece to its class.
pub struct ScratchLease {
    arena: Arc<ScratchArena>,
    offset: u64,
    class: u64,
}

impl fmt::Debug for ScratchLease {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ScratchLease")
            .field("offset", &self.offset)
            .field("class", &self.class)
            .finish()
    }
}

impl ScratchLease {
    pub fn offset(&self) -> u64 {
        self.offset
    }

    pub fn address(&self) -> u64 {
        self.arena.base + self.offset
    }

    pub fn bytes(&self) -> u64 {
        self.class
    }
}

impl Drop for ScratchLease {
    fn drop(&mut self) {
        self.arena.in_use.fetch_sub(self.class, Ordering::AcqRel);
        let _ = self.arena.classes[ScratchArena::index(self.class)]
            .0
            .offer(self.offset);
    }
}
