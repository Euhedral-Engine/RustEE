//! Ring-buffer chunks, their slots, and the pool that recycles them.

use std::cell::UnsafeCell;
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU8, Ordering};

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

/// Retired chunks waiting to be reused. A try-lock guards the stack: when it is contended the chunk is
/// freed, or a fresh one allocated, instead of waiting, so it can never stall a producer or consumer.
pub(crate) struct ChunkPool<T> {
    busy: AtomicBool,
    max: usize,
    // Chunks travel as heap pointers; re-boxing a `Vec<Chunk>` element would allocate on every reuse.
    #[allow(clippy::vec_box)]
    stack: UnsafeCell<Vec<Box<Chunk<T>>>>,
}

impl<T> ChunkPool<T> {
    pub(crate) fn new(max: usize) -> Self {
        Self {
            busy: AtomicBool::new(false),
            max,
            stack: UnsafeCell::new(Vec::new()),
        }
    }

    pub(crate) fn max(&self) -> usize {
        self.max
    }

    fn try_lock(&self) -> bool {
        self.busy
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_ok()
    }

    fn unlock(&self) {
        self.busy.store(false, Ordering::Release);
    }

    pub(crate) fn pop(&self) -> Option<Box<Chunk<T>>> {
        if self.max == 0 || !self.try_lock() {
            return None;
        }
        let chunk = unsafe { (*self.stack.get()).pop() };
        self.unlock();
        chunk
    }

    /// Keeps `chunk` for reuse if there is room and the pool is free, otherwise drops it. All of the
    /// chunk's slots must be `EMPTY` and its `next` null.
    pub(crate) fn push(&self, chunk: Box<Chunk<T>>) {
        if self.max == 0 || !self.try_lock() {
            return;
        }
        let stack = unsafe { &mut *self.stack.get() };
        if stack.len() < self.max {
            stack.push(chunk);
        }
        self.unlock();
    }

    /// Chunks currently idle in the pool.
    pub(crate) fn len(&self) -> usize {
        if self.max == 0 || !self.try_lock() {
            return 0;
        }
        let len = unsafe { (*self.stack.get()).len() };
        self.unlock();
        len
    }
}

unsafe impl<T: Send> Send for ChunkPool<T> {}
unsafe impl<T: Send> Sync for ChunkPool<T> {}
