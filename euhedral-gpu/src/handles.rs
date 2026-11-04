#![cfg_attr(
    not(any(feature = "cuda", test)),
    allow(dead_code, reason = "the CUDA backend uses the pool")
)]

//! A lock-free stock of driver handles (private streams) that any thread takes one of, uses alone, and gives back.
//!
//! A slot holds a handle or 0. A thread takes a handle by swapping a slot to 0 (a compare-and-set that acquires what the
//! last user wrote) and gives it back by storing it into an empty slot (a release store). Nothing waits and nothing
//! allocates after the table is made; a handle that finds no empty slot on its way back is the caller's to destroy.

use crate::sync::{AtomicUsize, Ordering};

/// A fixed table of handles, each a non-zero `usize` (a pointer).
pub(crate) struct HandlePool {
    slots: Box<[AtomicUsize]>,
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
        }
    }

    /// Takes a handle, if one is stocked. Never waits.
    pub(crate) fn take(&self) -> Option<usize> {
        for slot in self.slots.iter() {
            let seen = slot.load(Ordering::Relaxed);
            if seen != 0
                && slot
                    .compare_exchange(seen, 0, Ordering::Acquire, Ordering::Relaxed)
                    .is_ok()
            {
                return Some(seen);
            }
        }
        None
    }

    /// Stocks `handle`; `Err(handle)` when every slot holds one (a stock that other threads are emptying and filling as it
    /// is scanned may refuse a handle it had room for: the caller destroys it, and makes another when it needs one).
    pub(crate) fn give(&self, handle: usize) -> Result<(), usize> {
        debug_assert_ne!(handle, 0);
        // A slot another thread emptied behind the scan is found by the second pass; a stock that stays full is full.
        for _ in 0..2 {
            for slot in self.slots.iter() {
                if slot.load(Ordering::Relaxed) == 0
                    && slot
                        .compare_exchange(0, handle, Ordering::Release, Ordering::Relaxed)
                        .is_ok()
                {
                    return Ok(());
                }
            }
        }
        Err(handle)
    }

    /// Every handle still stocked (the pool is going away: no thread uses it).
    #[cfg_attr(
        not(feature = "cuda"),
        allow(dead_code, reason = "the CUDA queue's drop uses it")
    )]
    pub(crate) fn drain(&self) -> impl Iterator<Item = usize> + '_ {
        self.slots.iter().filter_map(|slot| {
            let v = slot.swap(0, Ordering::Acquire);
            (v != 0).then_some(v)
        })
    }
}

#[cfg(all(test, not(loom)))]
mod tests {
    use super::*;

    #[test]
    fn a_handle_is_taken_once_and_given_back_to_a_free_slot() {
        let pool = HandlePool::new(2);
        assert_eq!(pool.take(), None);
        pool.give(7).unwrap();
        pool.give(9).unwrap();
        assert_eq!(pool.give(11), Err(11), "full: the caller destroys it");
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
        assert_eq!(pool.take(), None);
    }

    #[test]
    fn threads_never_hold_one_handle_at_once() {
        let pool = HandlePool::new(4);
        for h in 1..=4 {
            pool.give(h).unwrap();
        }
        let in_use: Vec<std::sync::atomic::AtomicUsize> = (0..5)
            .map(|_| std::sync::atomic::AtomicUsize::new(0))
            .collect();
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
                            // A handle that finds no slot is dropped (the stock is a cache): exclusivity is the property.
                            let _ = pool.give(h);
                        }
                    }
                });
            }
        });
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
