//! Allocation backends and deferred frees.
//!
//! A backend allocates at one location and frees what it allocated. The planner never calls a backend: an owner
//! claims first, then allocates, and keeps the claim with the memory ([`Allocation`]). A free that may block (CUDA's
//! `cuMemFree` and `cuMemFreeHost` synchronize the device; Vulkan's adapters drain before a free) is queued in a
//! [`Reaper`] behind the device work that last used the memory, so it is not freed while the device may still use it.
//! [`Reaper::reap`] makes the frees of the retired prefix on the thread that calls it: long calls run on the worker
//! that needs them, and the blocking time counts as the service time of the work that called `reap`. A device with a
//! `DeferredCalls` attached (`Device::attach_deferral` in `euhedral-gpu`) keeps frees off busy moments. A caller whose
//! frees are long enough to starve other work reaps from a thread of its own, or from a lattice set aside for
//! blocking work.

use std::fmt;

use euhedral_data_structures::queues::{Consumer, MpscQueue, Multi, Producer, Single};
use euhedral_data_structures::sync::{ClaimCell, HotScope};

use crate::planner::{Claim, LocationId};

/// A block of memory a backend allocated.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Block {
    /// The address (a device pointer, a host pointer).
    pub addr: u64,
    /// The bytes allocated (at least what was asked: backends round up to their granularity).
    pub bytes: u64,
    /// The backend's own handle, if it has one (a VMM allocation handle), else 0.
    pub handle: u64,
}

/// Why an allocation failed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AllocError {
    pub what: String,
}

impl fmt::Display for AllocError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.what)
    }
}

impl std::error::Error for AllocError {}

/// An allocator of one location.
pub trait Backend: Send + Sync + fmt::Debug {
    fn location(&self) -> LocationId;
    /// The bytes a request of `bytes` really takes (the backend's granularity), for claims made before allocating.
    fn granted(&self, bytes: u64) -> u64;
    /// Allocates `bytes`. May block briefly (the driver's lock); it is a build-time call.
    fn alloc(&self, bytes: u64) -> Result<Block, AllocError>;
    /// Frees a block of this backend.
    ///
    /// # Safety
    /// The block came from this backend's `alloc`, is freed once, and nothing (host or device) uses it any more.
    unsafe fn free(&self, block: Block);
    /// Whether `free` may wait for the device (then queue the block in a [`Reaper`] so it is freed only after the
    /// device work that used it retired; `reap` runs the free on the thread that calls it).
    fn free_blocks(&self) -> bool;
}

/// Memory and the claim that accounts for it. Dropping it does not free the memory: hand it to a [`Reaper`] (or
/// free it where blocking is allowed) with [`Allocation::free_now`].
#[derive(Debug)]
pub struct Allocation {
    pub block: Block,
    claim: Claim,
}

impl Allocation {
    pub fn new(block: Block, claim: Claim) -> Self {
        Allocation { block, claim }
    }

    pub fn claim(&self) -> &Claim {
        &self.claim
    }

    /// Frees the memory now and gives the claim back.
    ///
    /// # Safety
    /// As [`Backend::free`]; `backend` is the one that allocated the block.
    pub unsafe fn free_now(self, backend: &dyn Backend) {
        // SAFETY: the caller's contract.
        unsafe { backend.free(self.block) };
        drop(self.claim);
    }
}

/// Something that says whether the device finished with a resource (a lane's marker reached, a fence, an event).
pub trait Retired: Send {
    fn retired(&self) -> bool;
}

impl<F: Fn() -> bool + Send> Retired for F {
    fn retired(&self) -> bool {
        self()
    }
}

/// A queued free: the check that its work retired, and the item to drop.
type Entry<T> = (Box<dyn Retired>, T);

/// Frees queued behind the device work that last used them. `push` never waits; `reap` frees the retired prefix of
/// the queue, in the order it was pushed, and stops at the first item whose work has not retired: that item stays at
/// the front, so nothing behind it is freed before it (one queue should hold one stream's frees; a queue that mixes
/// streams delays a retired item behind an unretired one of another stream).
///
/// Producers push from any thread. `reap` has one consumer at a time: a `reap` that finds another one running
/// returns 0 at once and never waits.
///
/// Where a long free runs. Dropping a retired item makes the driver's `cuMemFree` or `cuMemFreeHost`, which
/// synchronize the device, on the thread that called `reap`. Long calls run on the worker that needs them: when a
/// lattice worker reaps, the free counts as the service time of the work that called `reap`, and nothing here moves
/// it elsewhere. To keep frees off busy moments, attach a `DeferredCalls` to the device (`Device::attach_deferral`
/// in `euhedral-gpu`): the frees are then queued and made at a moment with nothing in flight. A caller whose frees
/// are long enough to starve other work reaps from a thread of its own, or from a lattice set aside for blocking
/// work.
pub struct Reaper<T: Send> {
    queue: MpscQueue<Entry<T>>,
    producer: Producer<Entry<T>, Multi>,
    consumer: ClaimCell<Consumer<Entry<T>, Single>>,
}

impl<T: Send> fmt::Debug for Reaper<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Reaper")
            .field("pending", &self.pending())
            .finish()
    }
}

impl<T: Send> Default for Reaper<T> {
    fn default() -> Self {
        let queue = MpscQueue::new(64);
        let producer = queue.producer();
        let consumer = ClaimCell::new(queue.consumer());
        Reaper {
            queue,
            producer,
            consumer,
        }
    }
}

impl<T: Send> Reaper<T> {
    pub fn new() -> Self {
        Self::default()
    }

    /// Queues `item` to be dropped (freed) once `retired` says so.
    pub fn push(&self, retired: impl Retired + 'static, item: T) {
        if let Err((_, item)) = self.producer.offer((Box::new(retired), item)) {
            // An unbounded queue never refuses; if it did, the item is dropped now rather than lost.
            drop(item);
        }
    }

    /// Drops the queued items from the front while their work has retired, in push order, and returns how many. The
    /// first item whose work has not retired stays at the front, with everything behind it. A `reap` that finds
    /// another one running (on another thread, or this thread from inside a free) returns 0 at once, never waits.
    pub fn reap(&self) -> usize {
        let _hot = HotScope::enter();
        let Some(mut consumer) = self.consumer.try_claim() else {
            return 0;
        };
        let limit = self.queue.size_long();
        consumer.drain_while(drop, |(r, _)| !r.retired(), limit) as usize
    }

    /// Drops everything queued, retired or not (shutdown, after the device was drained). Returns 0 and drops nothing
    /// if a `reap` is running.
    pub fn reap_all(&self) -> usize {
        let Some(mut consumer) = self.consumer.try_claim() else {
            return 0;
        };
        consumer.drain_all(drop) as usize
    }

    pub fn pending(&self) -> usize {
        self.queue.size_long() as usize
    }
}
