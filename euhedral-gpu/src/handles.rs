#![cfg_attr(
    not(any(feature = "cuda", test)),
    allow(dead_code, reason = "the CUDA backend uses the pool")
)]

//! A lock-free stock of driver handles (private streams) that any thread takes one of, uses alone, and gives back.
//!
//! A slot holds a handle or 0. A thread takes a handle by swapping a slot to 0 (a compare-and-set that acquires what the
//! last user wrote) and gives it back by storing it into an empty slot (a release store).
//!
//! **No spurious misses or refusals.** The pool counts the handles that exist (`population`, told by the owner when it
//! makes one) and the handles stocked or on their way into a slot (`stocked`, raised *before* the store that lands a
//! handle and lowered after the swap that takes one):
//! - a `take` that scanned past a handle another thread moved retries while `stocked > 0`, so it misses only when no handle
//!   is stocked, and the owner then makes a new one. It waits for nothing but a giver between its count and its store, a
//!   few instructions.
//! - while the population is at most the slots, an empty slot exists for every handle in flight, so a `give` loops until
//!   its handle lands and is never refused.
//!
//! A handle beyond the slots (more handles than slots exist) goes to an overflow list, never destroyed on the thread that
//! gave it: the owner destroys what the list and the slots hold when it goes away. Nothing allocates after the table is
//! made, except the list's node of a handle that overflows.

#![allow(
    unsafe_code,
    reason = "the overflow list's nodes are boxes handed between threads as addresses, owned by whoever swapped them out"
)]

use crate::sync::{AtomicUsize, Ordering};

/// One step of the short wait for a peer between its count and its store (or its swap and its count).
fn backoff() {
    #[cfg(loom)]
    loom::thread::yield_now();
    #[cfg(not(loom))]
    #[allow(
        clippy::disallowed_methods,
        reason = "JAVA-BACKOFF: the retry waits only for a peer in a step of a few instructions, as Java's `SpinWait` waits for a peer's short step"
    )]
    std::hint::spin_loop();
}

/// A fixed table of handles, each a non-zero `usize` (a pointer).
pub(crate) struct HandlePool {
    slots: Box<[AtomicUsize]>,
    /// Handles stocked, or counted in on their way into a slot.
    stocked: AtomicUsize,
    /// Handles that exist, stocked or in use.
    population: AtomicUsize,
    /// Handles that found no slot: a push-only list (a Treiber stack that is never popped before the drain, so it has no
    /// ABA), drained when the owner goes away.
    overflow: crate::sync::AtomicUsize,
}

/// An overflow node: the handle and the node pushed before it.
struct Overflow {
    handle: usize,
    next: usize,
}

impl std::fmt::Debug for HandlePool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HandlePool")
            .field("slots", &self.slots.len())
            .finish()
    }
}

impl HandlePool {
    pub(crate) fn new(slots: usize) -> Self {
        Self {
            slots: (0..slots.max(1)).map(|_| AtomicUsize::new(0)).collect(),
            stocked: AtomicUsize::new(0),
            population: AtomicUsize::new(0),
            overflow: AtomicUsize::new(0),
        }
    }

    /// The owner made a handle (it is in use until given).
    pub(crate) fn created(&self) {
        self.population.fetch_add(1, Ordering::Relaxed);
    }

    /// Takes a handle. `None` only when no handle is stocked.
    pub(crate) fn take(&self) -> Option<usize> {
        loop {
            for slot in self.slots.iter() {
                let seen = slot.load(Ordering::Relaxed);
                if seen != 0
                    && slot
                        .compare_exchange(seen, 0, Ordering::Acquire, Ordering::Relaxed)
                        .is_ok()
                {
                    self.stocked.fetch_sub(1, Ordering::Release);
                    return Some(seen);
                }
            }
            if self.stocked.load(Ordering::Acquire) == 0 {
                return None;
            }
            // A handle is stocked or landing, and a scan missed it: look again.
            backoff();
        }
    }

    /// Stocks `handle`. `Err(handle)` only when more handles exist than slots and none is free: the caller pushes it with
    /// [`overflow`](Self::overflow).
    pub(crate) fn give(&self, handle: usize) -> Result<(), usize> {
        debug_assert_ne!(handle, 0);
        self.stocked.fetch_add(1, Ordering::Release);
        let must_land = self.population.load(Ordering::Relaxed) <= self.slots.len();
        loop {
            for slot in self.slots.iter() {
                if slot.load(Ordering::Relaxed) == 0
                    && slot
                        .compare_exchange(0, handle, Ordering::Release, Ordering::Relaxed)
                        .is_ok()
                {
                    return Ok(());
                }
            }
            if !must_land {
                self.stocked.fetch_sub(1, Ordering::Release);
                return Err(handle);
            }
            // The population fits the slots, so a slot is empty for this handle: one was emptied behind the scan.
            backoff();
        }
    }

    /// Keeps `handle`, which found no slot, until the owner goes away.
    pub(crate) fn overflow(&self, handle: usize) {
        let node = Box::into_raw(Box::new(Overflow { handle, next: 0 })) as usize;
        let mut head = self.overflow.load(Ordering::Relaxed);
        loop {
            // SAFETY: the node was made above and is this thread's until the exchange publishes it.
            unsafe { (*(node as *mut Overflow)).next = head };
            match self
                .overflow
                .compare_exchange(head, node, Ordering::Release, Ordering::Relaxed)
            {
                Ok(_) => return,
                Err(seen) => head = seen,
            }
        }
    }

    /// Every handle the pool still holds, in the slots and the overflow list (the owner is going away: no thread uses it).
    pub(crate) fn drain(&self) -> impl Iterator<Item = usize> + '_ {
        let mut node = self.overflow.swap(0, Ordering::Acquire);
        let overflowed = std::iter::from_fn(move || {
            if node == 0 {
                return None;
            }
            // SAFETY: the list was swapped out whole: this iterator owns every node, made by `overflow`.
            let boxed = unsafe { Box::from_raw(node as *mut Overflow) };
            node = boxed.next;
            Some(boxed.handle)
        });
        let stocked = self.slots.iter().filter_map(|slot| {
            let v = slot.swap(0, Ordering::Acquire);
            (v != 0).then_some(v)
        });
        stocked.chain(overflowed)
    }
}

#[cfg(all(test, not(loom)))]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize as Count;

    fn stocked(slots: usize, handles: &[usize]) -> HandlePool {
        let pool = HandlePool::new(slots);
        for &h in handles {
            pool.created();
            pool.give(h).unwrap();
        }
        pool
    }

    #[test]
    fn a_handle_is_taken_once_and_given_back_to_a_free_slot() {
        let pool = stocked(2, &[7, 9]);
        let a = pool.take().unwrap();
        let b = pool.take().unwrap();
        assert_eq!(
            {
                let mut v = [a, b];
                v.sort_unstable();
                v
            },
            [7, 9]
        );
        assert_eq!(pool.take(), None, "nothing stocked");
        pool.give(a).unwrap();
        assert_eq!(pool.take(), Some(a));
    }

    #[test]
    fn a_handle_beyond_the_slots_goes_to_the_overflow_list_and_is_drained_with_the_rest() {
        let pool = stocked(2, &[7, 9]);
        pool.created(); // a third handle exists: more than the slots
        assert_eq!(pool.give(11), Err(11), "no slot for it");
        pool.overflow(11);
        pool.created();
        assert_eq!(pool.give(13), Err(13));
        pool.overflow(13);
        let mut all: Vec<usize> = pool.drain().collect();
        all.sort_unstable();
        assert_eq!(all, [7, 9, 11, 13]);
        assert_eq!(pool.drain().count(), 0);
    }

    #[test]
    fn threads_never_hold_one_handle_at_once() {
        let pool = stocked(4, &[1, 2, 3, 4]);
        let in_use: Vec<Count> = (0..5).map(|_| Count::new(0)).collect();
        let (threads, rounds) = if cfg!(miri) { (3, 6) } else { (8, 3_000) };
        std::thread::scope(|scope| {
            for _ in 0..threads {
                let (pool, in_use) = (&pool, &in_use);
                scope.spawn(move || {
                    for _ in 0..rounds {
                        if let Some(h) = pool.take() {
                            let users = in_use[h].fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                            assert_eq!(users, 0, "two threads hold handle {h}");
                            in_use[h].fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
                            pool.give(h).unwrap();
                        }
                    }
                });
            }
        });
    }

    /// With as many slots as handles and fewer threads than handles, a handle is always stocked and a slot is always empty
    /// for a give: a take never misses and a give is never refused, however the scans interleave.
    #[test]
    fn a_take_never_misses_and_a_give_is_never_refused_while_a_handle_is_stocked_and_the_population_fits()
     {
        let pool = stocked(8, &[1, 2, 3, 4, 5, 6, 7, 8]);
        let (misses, refusals) = (Count::new(0), Count::new(0));
        let rounds = if cfg!(miri) { 20 } else { 3_000_000 };
        std::thread::scope(|scope| {
            for _ in 0..7 {
                let (pool, misses, refusals) = (&pool, &misses, &refusals);
                scope.spawn(move || {
                    for _ in 0..rounds {
                        let Some(mut h) = pool.take() else {
                            misses.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            continue;
                        };
                        while let Err(back) = pool.give(h) {
                            refusals.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            h = back;
                        }
                    }
                });
            }
        });
        assert_eq!((misses.into_inner(), refusals.into_inner()), (0, 0));
    }
}

/// The table's handshake under every interleaving the memory model allows.
#[cfg(all(loom, test))]
mod loom_tests {
    use super::*;
    use crate::sync::UnsafeCell;
    use loom::sync::Arc;
    use loom::thread;

    // Two threads take the one handle and use what it stands for (a cell): at most one has it, and the next user sees what
    // the last wrote.
    #[test]
    fn loom_one_handle_is_used_by_one_thread_and_its_contents_are_ordered() {
        loom::model(|| {
            let pool = Arc::new(HandlePool::new(1));
            let cell = Arc::new(UnsafeCell::new(0u32));
            pool.created();
            pool.give(1).unwrap();
            let user = |pool: Arc<HandlePool>, cell: Arc<UnsafeCell<u32>>| {
                move || {
                    if let Some(h) = pool.take() {
                        // SAFETY: the handle is this thread's alone until it is given back.
                        cell.with_mut(|c| unsafe { *c += 1 });
                        pool.give(h).unwrap();
                    }
                }
            };
            let a = thread::spawn(user(pool.clone(), cell.clone()));
            let b = thread::spawn(user(pool.clone(), cell.clone()));
            a.join().unwrap();
            b.join().unwrap();
            assert!(pool.take().is_some(), "the handle is back");
        });
    }
}
