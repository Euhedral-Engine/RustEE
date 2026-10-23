//! The chunked ring-buffer queue behind every queue flavour (`BaseConcurrentQueue` upstream).
//!
//! Elements live in power-of-two ring chunks. Counters advance by [`INCREMENT`] per element and wrap
//! onto a chunk through `chunk_mask`. A producer may write up to `epoch`; when the ring is full an
//! unbounded queue links a fresh chunk by publishing a `LINK` slot, and the consumer follows it.
//!
//! Every method is `unsafe` because it relies on a role contract that the handles in `queue.rs`
//! enforce: one single-producer (`sp_*`) caller at a time, one single-consumer (`sc_*`) caller at a time.

use super::chunk::{Chunk, ChunkPool, FULL, LINK, Slot};
use super::queue_utils::{INCREMENT, MAX_SIZE, RESIZING, SHIFT, round_chunk_size};
use crate::atomics::CachePadded;
use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, AtomicUsize, Ordering};

/// State only the consumer writes; producers read `head` to learn how much room there is.
struct ConsumerState<T> {
    head: AtomicU64,
    head_chunk: UnsafeCell<*mut Chunk<T>>,
}

/// State only the producer (or, for multiple producers, the one resizing) writes.
struct ProducerState<T> {
    epoch: AtomicU64,
    tail_chunk: AtomicPtr<Chunk<T>>,
}

/// Fixed at construction and read by every operation.
struct Config {
    chunk_len: usize,
    chunk_mask: u64,
    /// How far past `head` a producer may write, in counter units.
    window: u64,
    bounded: bool,
    capacity: u64,
    /// The most a multi-consumer drain takes per hold of the consumer lock.
    max_consume_batch: u64,
}

/// Written only occasionally: the pool at chunk boundaries, the role flags when a handle is made, and
/// the statistics. Kept off the hot lines so that traffic never invalidates them.
pub(crate) struct Shared<T> {
    pool: ChunkPool<T>,
    chunks_allocated: AtomicUsize,
    pub(crate) producer_taken: AtomicBool,
    pub(crate) consumer_taken: AtomicBool,
}

/// Each group of fields below is padded to its own 128-byte blocks, like upstream's `TailPad`,
/// `MidPad1`, `MidPad2` and `StartPad`, so a producer, a consumer and a resizing thread never
/// contend for a line they do not both need.
pub(crate) struct Core<T> {
    tail: CachePadded<AtomicU64>,
    consumer: CachePadded<ConsumerState<T>>,
    producer: CachePadded<ProducerState<T>>,
    pub(crate) shared: CachePadded<Shared<T>>,
    config: CachePadded<Config>,
    /// Zero, or the token of the thread currently acting as the consumer of a multi-consumer queue.
    mc_lock: CachePadded<AtomicU64>,
}

unsafe impl<T: Send> Send for Core<T> {}
unsafe impl<T: Send> Sync for Core<T> {}

impl<T> Core<T> {
    /// An unbounded queue whose chunks hold `round_chunk_size(chunk_size)` ring slots.
    pub(crate) fn new(chunk_size: usize, max_pooled_chunks: usize) -> Self {
        let chunk_len = round_chunk_size(chunk_size.max(2) as u64) as usize;
        let window = ((chunk_len as u64) - 1) << SHIFT;
        Self::build(chunk_len, window, false, u64::MAX, max_pooled_chunks)
    }

    /// A queue that holds exactly `capacity` elements and never allocates another chunk.
    pub(crate) fn bounded(capacity: usize) -> Self {
        assert!(capacity > 0, "capacity must be positive");
        assert!(
            (capacity as u64) < MAX_SIZE,
            "capacity must be below {MAX_SIZE}"
        );
        // One spare slot lets a full ring be told apart from an empty one.
        let chunk_len = round_chunk_size(capacity as u64 + 1) as usize;
        Self::build(
            chunk_len,
            (capacity as u64) << SHIFT,
            true,
            capacity as u64,
            0,
        )
    }

    fn build(
        chunk_len: usize,
        window: u64,
        bounded: bool,
        capacity: u64,
        max_pooled_chunks: usize,
    ) -> Self {
        let first = Box::into_raw(Chunk::new(chunk_len));
        Self {
            tail: CachePadded::new(AtomicU64::new(0)),
            consumer: CachePadded::new(ConsumerState {
                head: AtomicU64::new(0),
                head_chunk: UnsafeCell::new(first),
            }),
            producer: CachePadded::new(ProducerState {
                epoch: AtomicU64::new(window),
                tail_chunk: AtomicPtr::new(first),
            }),
            shared: CachePadded::new(Shared {
                pool: ChunkPool::new(max_pooled_chunks),
                chunks_allocated: AtomicUsize::new(0),
                producer_taken: AtomicBool::new(false),
                consumer_taken: AtomicBool::new(false),
            }),
            config: CachePadded::new(Config {
                chunk_len,
                chunk_mask: ((chunk_len as u64) - 1) << SHIFT,
                window,
                bounded,
                capacity,
                max_consume_batch: u64::MAX,
            }),
            mc_lock: CachePadded::new(AtomicU64::new(0)),
        }
    }

    pub(crate) fn capacity(&self) -> u64 {
        self.config.capacity
    }

    #[cfg(test)]
    fn pool_address(&self) -> *const u8 {
        &self.shared.pool as *const ChunkPool<T> as *const u8
    }

    pub(crate) fn max_pooled_chunks(&self) -> usize {
        self.shared.pool.max()
    }

    pub(crate) fn pooled_chunks(&self) -> usize {
        self.shared.pool.len()
    }

    pub(crate) fn chunks_allocated(&self) -> usize {
        self.shared.chunks_allocated.load(Ordering::Relaxed)
    }

    pub(crate) fn max_consume_batch(&self) -> u64 {
        self.config.max_consume_batch
    }

    /// Sets the batch cap while the queue is still unshared.
    pub(crate) fn set_max_consume_batch(&mut self, batch: u64) {
        self.config.max_consume_batch = batch;
    }

    // ----- Consumer lock (multiple consumers) -----

    /// Becomes the active consumer unless another one is, without waiting.
    pub(crate) fn try_lock_consumers(&self) -> Option<ConsumerLock<'_>> {
        let token = thread_token();
        self.mc_lock
            .compare_exchange(0, token, Ordering::Acquire, Ordering::Relaxed)
            .is_ok()
            // Lazily: building the guard eagerly would drop it, and so unlock, when the attempt failed.
            .then(|| ConsumerLock {
                cell: &self.mc_lock,
            })
    }

    /// Becomes the active consumer, waiting for the current one to finish.
    ///
    /// # Panics
    ///
    /// Panics if the calling thread is already the active consumer, since it would wait for itself.
    pub(crate) fn lock_consumers(&self) -> ConsumerLock<'_> {
        loop {
            if let Some(lock) = self.try_lock_consumers() {
                return lock;
            }
            assert!(
                self.mc_lock.load(Ordering::Relaxed) != thread_token(),
                "clear called from inside a consumer callback of the same queue"
            );
            std::hint::spin_loop();
        }
    }

    /// Reserved positions, which can include slots a producer has claimed but not yet published.
    pub(crate) fn size_long(&self) -> u64 {
        // `head` first: it only grows, so a later `tail` is never behind it.
        let head = self.consumer.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Acquire);
        tail.saturating_sub(head) >> SHIFT
    }

    /// The ring slot a counter position maps to.
    fn index(&self, position: u64) -> usize {
        ((position & self.config.chunk_mask) >> SHIFT) as usize
    }

    fn allocate_chunk(&self) -> *mut Chunk<T> {
        let chunk = self.shared.pool.pop().unwrap_or_else(|| {
            self.shared.chunks_allocated.fetch_add(1, Ordering::Relaxed);
            Chunk::new(self.config.chunk_len)
        });
        Box::into_raw(chunk)
    }

    /// Hands a fully consumed chunk to the pool, or frees it.
    ///
    /// # Safety
    ///
    /// `chunk` must be a chunk of this queue whose slots are all `EMPTY` and whose `next` is null, and
    /// the caller must not touch it afterwards: a producer may reuse it at once.
    unsafe fn release_chunk(&self, chunk: *mut Chunk<T>) {
        self.shared.pool.push(unsafe { Box::from_raw(chunk) });
    }

    /// Ends `old` at the slot for `position` and continues in `next`.
    ///
    /// # Safety
    ///
    /// The caller must own that slot, which is free, and `next` must be a fresh chunk.
    unsafe fn link(&self, old: *mut Chunk<T>, next: *mut Chunk<T>, position: u64) {
        let old = unsafe { &*old };
        // Published by the release store on the slot just below.
        old.next.store(next, Ordering::Relaxed);
        unsafe { old.slots[self.index(position)].link() };
    }

    // ----- Single producer -----

    /// Appends `item`, or hands it back if a bounded queue is full.
    ///
    /// # Safety
    ///
    /// At most one thread may call the `sp_*` methods at a time, and none may call the `mp_*` ones.
    pub(crate) unsafe fn sp_offer(&self, item: T) -> Result<(), T> {
        let tail = self.tail.load(Ordering::Relaxed);
        while self.producer.epoch.load(Ordering::Relaxed) <= tail {
            if !unsafe { self.sp_make_room(tail) } {
                return Err(item);
            }
        }
        let chunk = unsafe { &*self.producer.tail_chunk.load(Ordering::Relaxed) };
        unsafe { chunk.slots[self.index(tail)].publish(item) };
        self.tail.store(tail + INCREMENT, Ordering::Release);
        Ok(())
    }

    /// Moves items from the front of `items` until the queue is full, returning how many.
    ///
    /// # Safety
    ///
    /// As for [`sp_offer`](Self::sp_offer).
    pub(crate) unsafe fn sp_fill(&self, items: &mut Vec<T>) -> usize {
        let mut total = 0;
        while !items.is_empty() {
            let tail = self.tail.load(Ordering::Relaxed);
            let epoch = self.producer.epoch.load(Ordering::Relaxed);
            if epoch <= tail {
                if !unsafe { self.sp_make_room(tail) } {
                    break;
                }
                continue;
            }

            let count = (((epoch - tail) >> SHIFT) as usize).min(items.len());
            let chunk = unsafe { &*self.producer.tail_chunk.load(Ordering::Relaxed) };
            let mut position = tail;
            for item in items.drain(..count) {
                unsafe { chunk.slots[self.index(position)].publish(item) };
                position += INCREMENT;
            }
            self.tail.store(position, Ordering::Release);
            total += count;
        }
        total
    }

    /// The producer has caught up with `epoch`. Raises it if the consumer freed room, otherwise starts
    /// a new chunk; returns false only for a full bounded queue.
    unsafe fn sp_make_room(&self, tail: u64) -> bool {
        let head = self.consumer.head.load(Ordering::Acquire);
        if head + self.config.window > tail {
            self.producer
                .epoch
                .store(head + self.config.window, Ordering::Relaxed);
            return true;
        }
        if self.config.bounded {
            return false;
        }

        let old = self.producer.tail_chunk.load(Ordering::Relaxed);
        let next = self.allocate_chunk();
        unsafe { self.link(old, next, tail) };
        self.producer.tail_chunk.store(next, Ordering::Relaxed);
        self.producer
            .epoch
            .store(tail + self.config.window, Ordering::Relaxed);
        true
    }

    // ----- Multiple producers -----

    /// Appends `item`, or hands it back if a bounded queue is full.
    ///
    /// # Safety
    ///
    /// The `mp_*` methods may run on any number of threads at once, but never on a queue that is also
    /// used through the `sp_*` methods.
    pub(crate) unsafe fn mp_offer(&self, item: T) -> Result<(), T> {
        match unsafe { self.mp_reserve(INCREMENT) } {
            Reservation::Reserved { chunk, tail, .. } => {
                unsafe { (*chunk).slots[self.index(tail)].publish(item) };
                Ok(())
            }
            Reservation::Full => Err(item),
        }
    }

    /// Moves items from the front of `items` until the queue is full, returning how many.
    ///
    /// # Safety
    ///
    /// As for [`mp_offer`](Self::mp_offer).
    pub(crate) unsafe fn mp_fill(&self, items: &mut Vec<T>) -> usize {
        let mut total = 0;
        while !items.is_empty() {
            let wanted = (items.len() as u64) << SHIFT;
            match unsafe { self.mp_reserve(wanted) } {
                Reservation::Full => break,
                Reservation::Reserved { chunk, tail, count } => {
                    let count = (count >> SHIFT) as usize;
                    // Write through the chunk the reservation came from, not whatever the tail chunk
                    // is by now: another producer may already have moved on to a newer one.
                    let chunk = unsafe { &*chunk };
                    for (i, item) in items.drain(..count).enumerate() {
                        let position = tail + (i as u64) * INCREMENT;
                        unsafe { chunk.slots[self.index(position)].publish(item) };
                    }
                    total += count;
                }
            }
        }
        total
    }

    /// Claims between one slot and `wanted` counter units of room at the tail, never more than the
    /// current epoch allows. Starts a new chunk when the ring is full and the queue is unbounded; that
    /// producer claims the first slot of the new chunk.
    unsafe fn mp_reserve(&self, wanted: u64) -> Reservation<T> {
        loop {
            let tail = self.tail.load(Ordering::Acquire);
            if tail & RESIZING != 0 {
                // Another producer is installing a chunk and will clear the flag in a moment.
                std::hint::spin_loop();
                continue;
            }
            let epoch = self.producer.epoch.load(Ordering::Acquire);
            let chunk = self.producer.tail_chunk.load(Ordering::Acquire);

            let mut limit = epoch;
            if epoch <= tail {
                // Out of room for this epoch: see whether the consumer has freed some.
                let head = self.consumer.head.load(Ordering::Acquire);
                let next_epoch = head + self.config.window;
                if next_epoch <= tail {
                    if self.config.bounded {
                        return Reservation::Full;
                    }
                    if self
                        .tail
                        .compare_exchange(
                            tail,
                            tail | RESIZING,
                            Ordering::AcqRel,
                            Ordering::Relaxed,
                        )
                        .is_ok()
                    {
                        return unsafe { self.mp_resize(chunk, tail) };
                    }
                    continue;
                }
                if self
                    .producer
                    .epoch
                    .compare_exchange(epoch, next_epoch, Ordering::AcqRel, Ordering::Relaxed)
                    .is_err()
                {
                    continue;
                }
                limit = next_epoch;
            }

            let count = wanted.min(limit - tail);
            if self
                .tail
                .compare_exchange(tail, tail + count, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
            {
                return Reservation::Reserved { chunk, tail, count };
            }
        }
    }

    /// Installs a new tail chunk. The caller has set the resizing flag in the tail, which keeps every
    /// other producer out, and `old` is the tail chunk: the tail has not moved since it was read.
    unsafe fn mp_resize(&self, old: *mut Chunk<T>, tail: u64) -> Reservation<T> {
        let next = self.allocate_chunk();
        self.producer.tail_chunk.store(next, Ordering::Relaxed);
        self.producer
            .epoch
            .store(tail + self.config.window, Ordering::Relaxed);
        unsafe { self.link(old, next, tail) };
        // Clears the flag and takes the first slot of `next` for this producer. The release makes the
        // new chunk, epoch and link visible to the producers waiting on the flag.
        self.tail.fetch_add(RESIZING, Ordering::Release);
        Reservation::Reserved {
            chunk: next,
            tail,
            count: INCREMENT,
        }
    }

    // ----- Single consumer -----

    /// Removes and returns the oldest published element.
    ///
    /// # Safety
    ///
    /// At most one thread may call the `sc_*` methods at a time, and the references they return must
    /// not be used after the next `sc_*` call.
    pub(crate) unsafe fn sc_poll(&self) -> Option<T> {
        let head = self.consumer.head.load(Ordering::Relaxed);
        let slot = unsafe { self.sc_slot(head) }?;
        let item = unsafe { slot.take() };
        self.consumer
            .head
            .store(head + INCREMENT, Ordering::Release);
        Some(item)
    }

    /// The oldest published element, left in place.
    ///
    /// # Safety
    ///
    /// As for [`sc_poll`](Self::sc_poll).
    pub(crate) unsafe fn sc_peek(&self) -> Option<&T> {
        let head = self.consumer.head.load(Ordering::Relaxed);
        let slot = unsafe { self.sc_slot(head) }?;
        Some(unsafe { slot.peek() })
    }

    /// Hands up to `limit` elements to `consume`, stopping before the first one `stop` accepts.
    ///
    /// The head is published when this returns or unwinds, so a panicking callback leaves the queue
    /// consistent: the element it was given is gone, the rest are still queued.
    ///
    /// # Safety
    ///
    /// As for [`sc_poll`](Self::sc_poll).
    pub(crate) unsafe fn sc_drain(
        &self,
        consume: &mut impl FnMut(T),
        stop: &mut impl FnMut(&T) -> bool,
        limit: u64,
    ) -> u64 {
        let mut commit = HeadCommit {
            cell: &self.consumer.head,
            head: self.consumer.head.load(Ordering::Relaxed),
        };
        let mut total = 0;
        while total < limit {
            let Some(slot) = (unsafe { self.sc_slot(commit.head) }) else {
                break;
            };
            if stop(unsafe { slot.peek() }) {
                break;
            }
            let item = unsafe { slot.take() };
            commit.head += INCREMENT;
            total += 1;
            consume(item);
        }
        total
    }

    /// The slot at `head` if it holds a published element, retiring finished chunks on the way.
    unsafe fn sc_slot(&self, head: u64) -> Option<&Slot<T>> {
        loop {
            let chunk_ptr = unsafe { *self.consumer.head_chunk.get() };
            let chunk = unsafe { &*chunk_ptr };
            let slot = &chunk.slots[self.index(head)];
            match slot.state() {
                FULL => return Some(slot),
                LINK => {
                    // `next` was stored before the slot was published as LINK.
                    let next = chunk.next.load(Ordering::Acquire);
                    unsafe { slot.clear_link() };
                    chunk.next.store(std::ptr::null_mut(), Ordering::Relaxed);
                    unsafe { *self.consumer.head_chunk.get() = next };
                    unsafe { self.release_chunk(chunk_ptr) };
                }
                // EMPTY: nothing has been published here yet.
                _ => return None,
            }
        }
    }
}

/// Held while a multi-consumer queue's consumer operation runs; released on drop, including by unwind.
pub(crate) struct ConsumerLock<'a> {
    cell: &'a AtomicU64,
}

impl Drop for ConsumerLock<'_> {
    fn drop(&mut self) {
        self.cell.store(0, Ordering::Release);
    }
}

/// A non-zero number unique to the calling thread.
fn thread_token() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    thread_local! {
        static TOKEN: u64 = NEXT.fetch_add(1, Ordering::Relaxed);
    }
    TOKEN.with(|token| *token)
}

/// What a producer got from [`Core::mp_reserve`].
enum Reservation<T> {
    /// `count` counter units starting at `tail`, all in `chunk`.
    Reserved {
        chunk: *mut Chunk<T>,
        tail: u64,
        count: u64,
    },
    /// A bounded queue is full.
    Full,
}

/// Publishes the consumer's head when a drain finishes or unwinds.
struct HeadCommit<'a> {
    cell: &'a AtomicU64,
    head: u64,
}

impl Drop for HeadCommit<'_> {
    fn drop(&mut self) {
        self.cell.store(self.head, Ordering::Release);
    }
}

impl<T> Drop for Core<T> {
    fn drop(&mut self) {
        // Only the chain from the head chunk is live; the chunk drop drops any elements still in it.
        let mut chunk = *self.consumer.head_chunk.get_mut();
        while !chunk.is_null() {
            let boxed = unsafe { Box::from_raw(chunk) };
            chunk = boxed.next.load(Ordering::Relaxed);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{align_of, size_of};
    use std::sync::Arc;

    fn block(address: *const u8) -> usize {
        address as usize / 128
    }

    #[test]
    fn every_group_of_hot_fields_sits_alone_on_its_own_128_byte_block() {
        let core = Core::<u64>::new(8, 2);
        // One field from each group: the producer's tail, the consumer's head, the producer's epoch, the
        // pool's lock and a configuration constant that every operation reads.
        let blocks = [
            block(&*core.tail as *const AtomicU64 as *const u8),
            block(&core.consumer.head as *const AtomicU64 as *const u8),
            block(&core.producer.epoch as *const AtomicU64 as *const u8),
            block(core.pool_address()),
            block(&core.config.chunk_mask as *const u64 as *const u8),
        ];

        for (i, a) in blocks.iter().enumerate() {
            for b in &blocks[i + 1..] {
                assert_ne!(a, b, "two groups share a cache line: {blocks:?}");
            }
        }
    }

    #[test]
    fn the_queue_is_aligned_and_sized_in_whole_128_byte_blocks() {
        assert_eq!(align_of::<Core<u64>>(), 128);
        assert_eq!(size_of::<Core<u64>>() % 128, 0);
        let shared = Arc::new(Core::<u64>::new(8, 0));
        assert_eq!(Arc::as_ptr(&shared) as usize % 128, 0);
    }
}
