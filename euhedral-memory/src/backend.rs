//! Allocation backends and deferred frees.
//!
//! A backend allocates at one location and frees what it allocated. The planner never calls a backend: an owner
//! claims first, then allocates, and keeps the claim with the memory ([`Allocation`]). A free that may block (CUDA's
//! `cuMemFree` and `cuMemFreeHost` synchronize the device; Vulkan's adapters drain before a free) is queued in a
//! [`Reaper`] behind the device work that last used the memory, so it is not freed while the device may still use it.
//! [`Reaper::reap`] makes the frees on the thread that calls it: when that is a lattice worker, the blocking free runs
//! on that worker and counts as the service time of the work that called `reap` (the `HOT-FIX` id M1 marks it). A caller whose
//! frees are long enough to starve other work reaps from a thread of its own, or from a lattice set aside for
//! blocking work. A thread of its own does not stop the stall, though: on the CUDA driver a free that waits for the device also
//! makes every other stream's enqueue wait, whichever thread makes it. The backends' blocking frees can therefore be queued for
//! a quiet moment as well ([`crate::Deferred`]).

use std::fmt;

use euhedral_data_structures::queues::{Consumer, MpmcQueue, Multi, Producer};
use euhedral_data_structures::sync::HotScope;

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

/// Frees queued behind the device work that last used them. `push` never waits; `reap` frees what has retired. A
/// free is ordinary work: run `reap` on the owner's lane (its blocking time counts as that work's service time); only
/// frees measured to block for much longer than the lattice's ordinary work go to a lattice set aside for them.
// HOT-FIX(M1): `reap` makes the blocking driver frees (`cuMemFree`, `cuMemFreeHost`) on the worker that calls it.
pub struct Reaper<T: Send> {
    queue: MpmcQueue<(Box<dyn Retired>, T)>,
    producer: Producer<(Box<dyn Retired>, T), Multi>,
    consumer: Consumer<(Box<dyn Retired>, T), Multi>,
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
        let queue = MpmcQueue::new(64);
        let producer = queue.producer();
        let consumer = queue.consumer();
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

    /// Drops every queued item whose work retired and returns how many; the others stay queued (behind the newer
    /// ones). Another thread reaping at the same moment makes this return early, never wait.
    pub fn reap(&self) -> usize {
        let _hot = HotScope::enter();
        let mut n = 0;
        let mut keep = Vec::new();
        let limit = self.queue.size_long();
        self.consumer.drain(
            |(r, item)| {
                if r.retired() {
                    drop(item);
                    n += 1;
                } else {
                    keep.push((r, item));
                }
            },
            limit,
        );
        for k in keep {
            let _ = self.producer.offer(k);
        }
        n
    }

    /// Drops everything queued, retired or not (shutdown, after the device was drained).
    pub fn reap_all(&self) -> usize {
        let mut n = 0;
        self.consumer.drain_all(|_| n += 1);
        n
    }

    pub fn pending(&self) -> usize {
        self.queue.size_long() as usize
    }
}
