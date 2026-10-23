//! Queue flavours and their producer and consumer handles.
//!
//! Upstream lets any thread call any method and documents who may call what. Here a queue hands out
//! role handles instead: a *single* role has at most one live handle and takes `&mut self`; a *multi*
//! role can be cloned freely. That keeps the whole API safe.

use super::core::Core;
use std::marker::PhantomData;
use std::sync::Arc;
use std::sync::atomic::Ordering;

mod sealed {
    pub trait Sealed {}
}

/// Whether a queue side is used by one thread or many.
pub trait Role: sealed::Sealed {
    const MULTI: bool;
}

/// At most one handle exists at a time and its operations take `&mut self`.
pub struct Single;
/// Handles are cloneable and may be used from many threads at once.
pub struct Multi;

impl sealed::Sealed for Single {}
impl sealed::Sealed for Multi {}
impl Role for Single {
    const MULTI: bool = false;
}
impl Role for Multi {
    const MULTI: bool = true;
}

/// Single producer, single consumer.
pub type SpscQueue<T> = Queue<T, Single, Single>;
/// Multiple producers, single consumer.
pub type MpscQueue<T> = Queue<T, Multi, Single>;
/// Single producer, multiple consumers.
pub type SpmcQueue<T> = Queue<T, Single, Multi>;
/// Multiple producers, multiple consumers.
pub type MpmcQueue<T> = Queue<T, Multi, Multi>;

/// An unbounded or bounded FIFO queue, generic over its producer and consumer roles.
pub struct Queue<T, P: Role, C: Role> {
    core: Arc<Core<T>>,
    roles: PhantomData<(P, C)>,
}

impl<T, P: Role, C: Role> Queue<T, P, C> {
    /// An unbounded queue whose chunks hold `chunk_size` slots, rounded up to a power of two.
    pub fn new(chunk_size: usize) -> Self {
        Self::with_pool(chunk_size, 0)
    }

    /// Like [`new`](Self::new), keeping up to `max_pooled_chunks` retired chunks for reuse.
    pub fn with_pool(chunk_size: usize, max_pooled_chunks: usize) -> Self {
        Self::from_core(Core::new(chunk_size, max_pooled_chunks))
    }

    /// A queue that holds exactly `capacity` elements: `offer` fails when it is full.
    ///
    /// # Panics
    ///
    /// Panics if `capacity` is zero.
    pub fn bounded(capacity: usize) -> Self {
        Self::from_core(Core::bounded(capacity))
    }

    fn from_core(core: Core<T>) -> Self {
        Self {
            core: Arc::new(core),
            roles: PhantomData,
        }
    }

    /// The producer handle.
    ///
    /// # Panics
    ///
    /// For a [`Single`] producer, panics if its handle is still alive.
    pub fn producer(&self) -> Producer<T, P> {
        if !P::MULTI {
            assert!(
                self.core
                    .shared
                    .producer_taken
                    .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
                    .is_ok(),
                "the single producer handle is already in use"
            );
        }
        Producer {
            core: Arc::clone(&self.core),
            role: PhantomData,
        }
    }

    /// The consumer handle.
    ///
    /// # Panics
    ///
    /// For a [`Single`] consumer, panics if its handle is still alive.
    pub fn consumer(&self) -> Consumer<T, C> {
        if !C::MULTI {
            assert!(
                self.core
                    .shared
                    .consumer_taken
                    .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
                    .is_ok(),
                "the single consumer handle is already in use"
            );
        }
        Consumer {
            core: Arc::clone(&self.core),
            role: PhantomData,
        }
    }

    /// Reserved slots, including any a producer has claimed but not yet published.
    pub fn size_long(&self) -> u64 {
        self.core.size_long()
    }

    pub fn is_empty(&self) -> bool {
        self.size_long() == 0
    }

    /// `u64::MAX` for an unbounded queue.
    pub fn capacity(&self) -> u64 {
        self.core.capacity()
    }

    pub fn max_pooled_chunks(&self) -> usize {
        self.core.max_pooled_chunks()
    }

    /// Retired chunks currently idle in the pool.
    pub fn pooled_chunks(&self) -> usize {
        self.core.pooled_chunks()
    }

    /// Chunks allocated fresh over the queue's lifetime, not counting the first or reused ones.
    pub fn chunks_allocated(&self) -> usize {
        self.core.chunks_allocated()
    }
}

impl<T, P: Role> Queue<T, P, Multi> {
    /// Caps how many elements one consumer takes per lock hold. A `drain` above the cap releases the
    /// lock between batches so other consumers can interleave.
    ///
    /// # Panics
    ///
    /// Panics if `batch` is zero, or if a producer or consumer handle has already been created.
    pub fn with_max_consume_batch(mut self, batch: u64) -> Self {
        assert!(batch > 0, "max consume batch must be positive");
        Arc::get_mut(&mut self.core)
            .expect("configure the queue before creating its handles")
            .set_max_consume_batch(batch);
        self
    }

    pub fn max_consume_batch(&self) -> u64 {
        self.core.max_consume_batch()
    }
}

/// The producing side of a [`Queue`].
pub struct Producer<T, P: Role> {
    core: Arc<Core<T>>,
    role: PhantomData<P>,
}

impl<T, P: Role> Producer<T, P> {
    pub fn size_long(&self) -> u64 {
        self.core.size_long()
    }

    pub fn is_empty(&self) -> bool {
        self.size_long() == 0
    }

    pub fn capacity(&self) -> u64 {
        self.core.capacity()
    }
}

impl<T, P: Role> Drop for Producer<T, P> {
    fn drop(&mut self) {
        if !P::MULTI {
            self.core
                .shared
                .producer_taken
                .store(false, Ordering::Release);
        }
    }
}

impl<T> Producer<T, Single> {
    /// Appends `item`, or returns it if a bounded queue is full.
    pub fn offer(&mut self, item: T) -> Result<(), T> {
        // SAFETY: this is the only live single-producer handle, and `&mut self` excludes concurrent use.
        unsafe { self.core.sp_offer(item) }
    }

    /// Moves items from the front of `items` into the queue and returns how many it took. Items it
    /// could not take (a bounded queue filled up) stay in `items`, in order.
    pub fn fill(&mut self, items: &mut Vec<T>) -> usize {
        // SAFETY: as for `offer`.
        unsafe { self.core.sp_fill(items) }
    }
}

impl<T> Clone for Producer<T, Multi> {
    fn clone(&self) -> Self {
        Self {
            core: Arc::clone(&self.core),
            role: PhantomData,
        }
    }
}

impl<T> Producer<T, Multi> {
    /// Appends `item`, or returns it if a bounded queue is full.
    pub fn offer(&self, item: T) -> Result<(), T> {
        // SAFETY: a queue with multiple producers never uses the single-producer methods.
        unsafe { self.core.mp_offer(item) }
    }

    /// Moves items from the front of `items` into the queue and returns how many it took. Items it
    /// could not take (a bounded queue filled up) stay in `items`, in order.
    pub fn fill(&self, items: &mut Vec<T>) -> usize {
        // SAFETY: as for `offer`.
        unsafe { self.core.mp_fill(items) }
    }
}

/// The consuming side of a [`Queue`].
pub struct Consumer<T, C: Role> {
    core: Arc<Core<T>>,
    role: PhantomData<C>,
}

impl<T, C: Role> Consumer<T, C> {
    pub fn size_long(&self) -> u64 {
        self.core.size_long()
    }

    pub fn is_empty(&self) -> bool {
        self.size_long() == 0
    }

    pub fn capacity(&self) -> u64 {
        self.core.capacity()
    }
}

impl<T, C: Role> Drop for Consumer<T, C> {
    fn drop(&mut self) {
        if !C::MULTI {
            self.core
                .shared
                .consumer_taken
                .store(false, Ordering::Release);
        }
    }
}

impl<T> Consumer<T, Single> {
    /// Removes and returns the oldest published element.
    pub fn poll(&mut self) -> Option<T> {
        // SAFETY: this is the only live single-consumer handle, and `&mut self` excludes concurrent use.
        unsafe { self.core.sc_poll() }
    }

    /// The oldest published element, left in place. The borrow ends before the next `poll` or `drain`.
    pub fn peek(&mut self) -> Option<&T> {
        // SAFETY: as for `poll`; the returned borrow keeps `&mut self` alive, so nothing can consume it.
        unsafe { self.core.sc_peek() }
    }

    /// Hands up to `limit` elements to `consume`, oldest first, and returns how many.
    pub fn drain(&mut self, mut consume: impl FnMut(T), limit: u64) -> u64 {
        // SAFETY: as for `poll`.
        unsafe { self.core.sc_drain(&mut consume, &mut |_| false, limit) }
    }

    /// [`drain`](Self::drain) without a limit.
    pub fn drain_all(&mut self, consume: impl FnMut(T)) -> u64 {
        self.drain(consume, u64::MAX)
    }

    /// Like [`drain`](Self::drain), but stops before the first element for which `stop` returns true,
    /// leaving it queued.
    pub fn drain_while(
        &mut self,
        mut consume: impl FnMut(T),
        mut stop: impl FnMut(&T) -> bool,
        limit: u64,
    ) -> u64 {
        // SAFETY: as for `poll`.
        unsafe { self.core.sc_drain(&mut consume, &mut stop, limit) }
    }

    /// Removes every published element.
    pub fn clear(&mut self) {
        while self.drain_all(drop) > 0 {}
    }
}

impl<T> Clone for Consumer<T, Multi> {
    fn clone(&self) -> Self {
        Self {
            core: Arc::clone(&self.core),
            role: PhantomData,
        }
    }
}

impl<T> Consumer<T, Multi> {
    /// Removes and returns the oldest published element, or `None` if there is none or another
    /// consumer is active.
    pub fn poll(&self) -> Option<T> {
        let _lock = self.core.try_lock_consumers()?;
        // SAFETY: the lock makes this thread the only consumer until `_lock` drops.
        unsafe { self.core.sc_poll() }
    }

    /// Runs `look` on the oldest published element, leaving it queued. Returns `None` if there is none
    /// or another consumer is active.
    pub fn peek_with<R>(&self, look: impl FnOnce(&T) -> R) -> Option<R> {
        let _lock = self.core.try_lock_consumers()?;
        // SAFETY: as for `poll`; the borrow ends inside `look`, before the lock is released.
        unsafe { self.core.sc_peek() }.map(look)
    }

    /// Hands up to `limit` elements to `consume`, oldest first, and returns how many. It stops early if
    /// another consumer is active.
    pub fn drain(&self, consume: impl FnMut(T), limit: u64) -> u64 {
        self.drain_while(consume, |_| false, limit)
    }

    /// [`drain`](Self::drain) without a limit.
    pub fn drain_all(&self, consume: impl FnMut(T)) -> u64 {
        self.drain(consume, u64::MAX)
    }

    /// Like [`drain`](Self::drain), but stops before the first element for which `stop` returns true,
    /// leaving it queued.
    pub fn drain_while(
        &self,
        mut consume: impl FnMut(T),
        mut stop: impl FnMut(&T) -> bool,
        limit: u64,
    ) -> u64 {
        let mut total = 0;
        while total < limit {
            let Some(_lock) = self.core.try_lock_consumers() else {
                break;
            };
            let batch = (limit - total).min(self.core.max_consume_batch());
            // SAFETY: as for `poll`.
            let drained = unsafe { self.core.sc_drain(&mut consume, &mut stop, batch) };
            total += drained;
            if drained < batch {
                break;
            }
        }
        total
    }

    /// Removes every published element, waiting for any active consumer to finish first.
    ///
    /// # Panics
    ///
    /// Panics if called from inside a callback of this queue's own consumer, which would wait forever.
    pub fn clear(&self) {
        let _lock = self.core.lock_consumers();
        // SAFETY: as for `poll`.
        while unsafe { self.core.sc_drain(&mut drop, &mut |_| false, u64::MAX) } > 0 {}
    }
}
