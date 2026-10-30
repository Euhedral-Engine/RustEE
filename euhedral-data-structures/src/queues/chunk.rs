//! Ring-buffer chunks, their slots, and the pool that recycles them.

use crate::atomics::CachePadded;
use crate::sync::loom as shim;
use std::cell::UnsafeCell;
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicPtr, AtomicU8, Ordering};

/// Nothing published. A consumer that sees this stops.
pub(crate) const EMPTY: u8 = 0;
/// `value` is initialised and owned by whoever consumes the slot.
pub(crate) const FULL: u8 = 1;
/// Upstream's sentinel: the chunk ends here and `Chunk::next` continues at the same index.
pub(crate) const LINK: u8 = 2;

/// One ring position. A producer writes `value` and then release-stores `FULL`; a consumer
/// acquire-loads the state before reading `value`. Every producer and consumer path uses the same
/// pair, so a consumer that sees a slot also sees everything the producer wrote before publishing it.
pub(crate) struct Slot<T> {
    state: AtomicU8,
    value: UnsafeCell<MaybeUninit<T>>,
}

impl<T> Slot<T> {
    fn new() -> Self {
        Self {
            state: AtomicU8::new(EMPTY),
            value: UnsafeCell::new(MaybeUninit::uninit()),
        }
    }

    pub(crate) fn state(&self) -> u8 {
        self.state.load(Ordering::Acquire)
    }

    /// Writes `value` and publishes it.
    ///
    /// # Safety
    ///
    /// The caller must own this slot: it reserved the position and the slot is `EMPTY`.
    pub(crate) unsafe fn publish(&self, value: T) {
        unsafe { (*self.value.get()).write(value) };
        self.state.store(FULL, Ordering::Release);
    }

    /// Ends the chunk at this slot; the caller has already stored `Chunk::next`.
    ///
    /// # Safety
    ///
    /// The caller must own this slot, as for [`publish`](Self::publish).
    pub(crate) unsafe fn link(&self) {
        self.state.store(LINK, Ordering::Release);
    }

    /// The consumer's view of a slot it observed `FULL`.
    ///
    /// # Safety
    ///
    /// The slot must be `FULL` and the caller the sole consumer.
    pub(crate) unsafe fn peek(&self) -> &T {
        unsafe { (*self.value.get()).assume_init_ref() }
    }

    /// Moves the value out and empties the slot.
    ///
    /// # Safety
    ///
    /// The slot must be `FULL` and the caller the sole consumer.
    pub(crate) unsafe fn take(&self) -> T {
        let value = unsafe { (*self.value.get()).assume_init_read() };
        // Producers reuse the slot only after they acquire a head that the consumer has release-stored
        // after this, so a relaxed store is enough.
        self.state.store(EMPTY, Ordering::Relaxed);
        value
    }

    /// Resets a `LINK` slot of a chunk being retired.
    ///
    /// # Safety
    ///
    /// The caller must be the sole consumer.
    pub(crate) unsafe fn clear_link(&self) {
        self.state.store(EMPTY, Ordering::Relaxed);
    }
}

pub(crate) struct Chunk<T> {
    pub(crate) slots: Box<[Slot<T>]>,
    /// The chunk that continues this one; set before the `LINK` slot is published.
    pub(crate) next: AtomicPtr<Chunk<T>>,
}

impl<T> Chunk<T> {
    pub(crate) fn new(len: usize) -> Box<Self> {
        Box::new(Self {
            slots: (0..len).map(|_| Slot::new()).collect(),
            next: AtomicPtr::new(std::ptr::null_mut()),
        })
    }
}

impl<T> Drop for Chunk<T> {
    fn drop(&mut self) {
        for slot in self.slots.iter_mut() {
            if *slot.state.get_mut() == FULL {
                unsafe { slot.value.get_mut().assume_init_drop() };
            }
        }
    }
}

/// Retired chunks waiting to be reused: a wait-free single-producer single-consumer ring of chunk pointers
/// (`ChunkAllocator` upstream).
///
/// The consumer of the queue retires chunks into it ([`push`](Self::push)) and the producer that grows the queue takes
/// them out ([`pop`](Self::pop)); each side is one thread at a time (a multi-consumer queue serializes its consumers,
/// and a multi-producer queue lets one resizing producer at a time allocate), and the hand-over between threads
/// carries the ring's indexes with it. Neither side waits or takes a lock: a full ring frees the chunk, an empty one
/// allocates a fresh chunk.
pub(crate) struct ChunkPool<T> {
    max: usize,
    /// The ring, in an allocation of its own (as upstream's allocator is its own queue), so that its padded indexes
    /// do not widen the queue's shared block.
    ring: Box<Ring<T>>,
}

struct Ring<T> {
    /// The next cell [`pop`](Self::pop) takes. Written by the allocating side only.
    head: CachePadded<shim::AtomicUsize>,
    /// The next cell [`push`](Self::push) fills. Written by the freeing side only.
    tail: CachePadded<shim::AtomicUsize>,
    cells: Box<[shim::AtomicPtr<Chunk<T>>]>,
}

impl<T> ChunkPool<T> {
    pub(crate) fn new(max: usize) -> Self {
        Self {
            max,
            ring: Box::new(Ring {
                head: CachePadded::new(shim::AtomicUsize::new(0)),
                tail: CachePadded::new(shim::AtomicUsize::new(0)),
                cells: (0..max)
                    .map(|_| shim::AtomicPtr::new(std::ptr::null_mut()))
                    .collect(),
            }),
        }
    }

    pub(crate) fn max(&self) -> usize {
        self.max
    }

    /// A retired chunk, if one is waiting. The allocating side only.
    pub(crate) fn pop(&self) -> Option<Box<Chunk<T>>> {
        if self.max == 0 {
            return None;
        }
        let head = self.ring.head.load(Ordering::Relaxed);
        if head == self.ring.tail.load(Ordering::Acquire) {
            return None;
        }
        let chunk = self.ring.cells[head % self.max].load(Ordering::Relaxed);
        self.ring
            .head
            .store(head.wrapping_add(1), Ordering::Release);
        // SAFETY: the cell held a pointer from `Box::into_raw` in `push`, which this side alone takes out.
        Some(unsafe { Box::from_raw(chunk) })
    }

    /// Keeps `chunk` for reuse if there is room, otherwise drops it. All of the chunk's slots must be `EMPTY` and
    /// its `next` null. The freeing side only.
    pub(crate) fn push(&self, chunk: Box<Chunk<T>>) {
        if self.max == 0 {
            return;
        }
        let tail = self.ring.tail.load(Ordering::Relaxed);
        if tail.wrapping_sub(self.ring.head.load(Ordering::Acquire)) >= self.max {
            return;
        }
        self.ring.cells[tail % self.max].store(Box::into_raw(chunk), Ordering::Relaxed);
        self.ring
            .tail
            .store(tail.wrapping_add(1), Ordering::Release);
    }

    /// Chunks currently idle in the pool.
    pub(crate) fn len(&self) -> usize {
        let tail = self.ring.tail.load(Ordering::Acquire);
        tail.wrapping_sub(self.ring.head.load(Ordering::Acquire))
            .min(self.max)
    }
}

impl<T> Drop for ChunkPool<T> {
    fn drop(&mut self) {
        let ring = &mut *self.ring;
        let (head, tail) = (
            ring.head.load(Ordering::Relaxed),
            ring.tail.load(Ordering::Relaxed),
        );
        let mut at = head;
        while at != tail {
            let chunk = ring.cells[at % self.max].load(Ordering::Relaxed);
            // SAFETY: each cell between head and tail holds a pointer from `Box::into_raw` in `push`.
            drop(unsafe { Box::from_raw(chunk) });
            at = at.wrapping_add(1);
        }
    }
}

unsafe impl<T: Send> Send for ChunkPool<T> {}
unsafe impl<T: Send> Sync for ChunkPool<T> {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    fn marked(len: usize, mark: usize) -> Box<Chunk<u32>> {
        let chunk = Chunk::<u32>::new(len);
        chunk
            .next
            .store(std::ptr::null_mut(), std::sync::atomic::Ordering::Relaxed);
        // The slot count stands for the mark.
        assert_eq!(chunk.slots.len(), len);
        let _ = mark;
        chunk
    }

    #[test]
    fn a_pool_hands_chunks_back_in_the_order_it_took_them_and_drops_what_does_not_fit() {
        let pool = ChunkPool::<u32>::new(2);
        assert!(pool.pop().is_none());
        pool.push(marked(4, 1));
        pool.push(marked(8, 2));
        pool.push(marked(16, 3));
        assert_eq!(pool.len(), 2, "the third did not fit");
        assert_eq!(pool.pop().unwrap().slots.len(), 4);
        pool.push(marked(32, 4));
        assert_eq!(pool.pop().unwrap().slots.len(), 8);
        assert_eq!(pool.pop().unwrap().slots.len(), 32);
        assert!(pool.pop().is_none());
        assert_eq!(pool.len(), 0);
    }

    #[test]
    fn a_pool_of_none_keeps_nothing() {
        let pool = ChunkPool::<u32>::new(0);
        pool.push(marked(4, 0));
        assert!(pool.pop().is_none());
        assert_eq!(pool.len(), 0);
    }

    #[test]
    fn chunks_left_in_a_pool_are_freed_with_it() {
        let pool = ChunkPool::<Arc<()>>::new(4);
        let live = Arc::new(());
        let chunk = Chunk::<Arc<()>>::new(2);
        // SAFETY: the chunk is ours and nothing else reads the slot.
        unsafe { chunk.slots[0].publish(live.clone()) };
        // A chunk keeps a published value only until it is dropped.
        pool.push(chunk);
        assert_eq!(Arc::strong_count(&live), 2);
        drop(pool);
        assert_eq!(Arc::strong_count(&live), 1);
    }

    #[test]
    #[allow(clippy::disallowed_methods)] // The test's two threads stand for the queue's consumer and its resizing producer.
    fn one_thread_frees_while_another_allocates_and_every_chunk_is_used_once() {
        const CHUNKS: usize = if cfg!(miri) { 40 } else { 20_000 };
        let pool = Arc::new(ChunkPool::<u32>::new(4));
        let freeing = {
            let pool = pool.clone();
            thread::spawn(move || {
                for len in 2..CHUNKS + 2 {
                    pool.push(marked(len, len));
                }
            })
        };
        let mut last = 0;
        let mut taken = 0;
        while !freeing.is_finished() || pool.len() > 0 {
            if let Some(chunk) = pool.pop() {
                assert!(
                    chunk.slots.len() > last,
                    "chunks come back in order, none twice"
                );
                last = chunk.slots.len();
                taken += 1;
            }
        }
        freeing.join().unwrap();
        assert!(taken > 0);
    }
}

#[cfg(all(loom, test))]
mod loom_tests {
    use super::*;
    use loom::cell::UnsafeCell as LoomCell;
    use loom::sync::Arc;
    use loom::thread;

    /// One loom-tracked word per chunk, written by the freeing side before it retires the chunk and read by the
    /// allocating side after it takes the chunk out: a ring that did not pass the writes on shows up as a data race.
    struct Witness {
        words: Vec<LoomCell<u32>>,
        addresses: Vec<usize>,
    }

    // SAFETY: a word is written before its chunk is retired and read after the chunk is taken out, which is the
    // ordering under test.
    unsafe impl Sync for Witness {}
    unsafe impl Send for Witness {}

    // The freeing side retires two chunks into a ring of one while the allocating side takes: a chunk taken is a
    // chunk retired, no chunk comes out twice, one that finds the ring full is freed, and the retiring side's writes
    // are visible to the taker.
    #[test]
    fn loom_one_freeing_side_and_one_allocating_side_lose_and_duplicate_nothing() {
        loom::model(|| {
            let pool = Arc::new(ChunkPool::<u32>::new(1));
            let chunks = [Chunk::<u32>::new(1), Chunk::<u32>::new(1)];
            let witness = Arc::new(Witness {
                addresses: chunks
                    .iter()
                    .map(|c| &**c as *const Chunk<u32> as usize)
                    .collect(),
                words: (0..2).map(|_| LoomCell::new(0)).collect(),
            });
            let freeing = {
                let (pool, witness) = (pool.clone(), witness.clone());
                thread::spawn(move || {
                    for (i, chunk) in chunks.into_iter().enumerate() {
                        witness.words[i].with_mut(|w| unsafe { *w = 7 + i as u32 });
                        pool.push(chunk);
                    }
                })
            };
            let allocating = {
                let (pool, witness) = (pool.clone(), witness.clone());
                thread::spawn(move || {
                    let mut taken = Vec::new();
                    for _ in 0..2 {
                        if let Some(chunk) = pool.pop() {
                            let address = &*chunk as *const Chunk<u32> as usize;
                            let i = witness
                                .addresses
                                .iter()
                                .position(|&a| a == address)
                                .unwrap();
                            let seen = witness.words[i].with(|w| unsafe { *w });
                            assert_eq!(seen, 7 + i as u32, "the retiring side's write is visible");
                            taken.push(i);
                        }
                        thread::yield_now();
                    }
                    taken
                })
            };
            freeing.join().unwrap();
            let taken = allocating.join().unwrap();
            let mut distinct = taken.clone();
            distinct.sort_unstable();
            distinct.dedup();
            assert_eq!(distinct.len(), taken.len(), "no chunk comes out twice");
            assert!(
                taken.len() + pool.len() <= 2,
                "a chunk is taken, kept or freed"
            );
        });
    }
}
