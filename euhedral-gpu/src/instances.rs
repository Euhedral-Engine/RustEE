//! The instances of recorded graphs, pooled by key, for any number of threads to take, use and give back without a lock.
//!
//! A recorded graph is used by one thread at a time (its executable takes new arguments and launches), so a key that
//! several threads run at once needs several instances of its graph. The pool is a fixed table of slots; a thread that
//! wants the graph of a key *takes* a free instance of it by one compare-and-set, and a thread that finds none does not
//! wait for another's: it *reserves* a slot, records an instance of its own, and puts it in the pool for the next. Nothing
//! spins and nothing allocates after the table is made: a slot's contents move in and out of it.
//!
//! The state any thread may touch is one word per slot, `(tag << 2) | state`, and only that:
//! - **`EMPTY`** (the word is 0): no instance, free to reserve.
//! - **`FREE`**: an instance of the key whose hash is `tag` is in the slot, and nobody uses it.
//! - **`BUSY`**: a thread owns the slot (it uses the instance, or it is recording one), and its tag says which key. Only
//!   the owner touches the slot's cell, and it hands the slot on with a release store of the new word; a thread that
//!   takes a slot does so by a compare-and-set that acquires that store, so the cell's contents are ordered between
//!   owners.
//!
//! A tag is a hash of the key, so a tag match is confirmed against the key once the slot is taken; the slot is given back
//! at once when it does not match. A scan therefore never blocks another thread on a slot it does not want, except for
//! the instant between its two stores.

#![allow(
    unsafe_code,
    reason = "a slot's cell is accessed only by the thread that owns the slot, which the slot's word orders"
)]

use std::hash::{Hash, Hasher};

use crate::sync::{AtomicU64, Ordering, UnsafeCell};

const EMPTY: u64 = 0;
const FREE: u64 = 1;
const BUSY: u64 = 2;
const STATE: u64 = 3;

/// The tag of `key`: 62 bits of its hash.
pub(crate) fn tag_of<K: Hash>(key: &K) -> u64 {
    let mut hasher = std::hash::DefaultHasher::new();
    key.hash(&mut hasher);
    hasher.finish() >> 2
}

fn word(tag: u64, state: u64) -> u64 {
    (tag << 2) | state
}

struct Slot<K, T> {
    word: AtomicU64,
    cell: UnsafeCell<Option<(K, T)>>,
}

/// A fixed table of instances, by key.
pub(crate) struct InstancePool<K, T> {
    slots: Box<[Slot<K, T>]>,
}

// SAFETY: a slot's cell is accessed only by the thread that owns the slot (its word says BUSY, and the thread that made
// it so by compare-and-set), and ownership passes by release/acquire on the word; the contents move between threads, so
// they must be `Send`.
unsafe impl<K: Send, T: Send> Sync for InstancePool<K, T> {}

impl<K, T> std::fmt::Debug for InstancePool<K, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InstancePool")
            .field("slots", &self.slots.len())
            .finish()
    }
}

/// A slot a thread owns: it gives the slot up when it is dropped, empty.
struct Owned<'a, K, T> {
    pool: &'a InstancePool<K, T>,
    index: usize,
    tag: u64,
}

impl<K, T> Owned<'_, K, T> {
    fn slot(&self) -> &Slot<K, T> {
        &self.pool.slots[self.index]
    }

    /// Puts `contents` in the slot and frees it for the key `tag` names.
    fn give(self, key: K, value: T) {
        // SAFETY: this thread owns the slot.
        let put = |cell: *mut Option<(K, T)>| unsafe { *cell = Some((key, value)) };
        self.slot().cell.with_mut(put);
        let free = word(self.tag, FREE);
        self.slot().word.store(free, Ordering::Release);
        std::mem::forget(self);
    }
}

impl<K, T> Drop for Owned<'_, K, T> {
    fn drop(&mut self) {
        // SAFETY: this thread owns the slot.
        let old = self.slot().cell.with_mut(|cell| unsafe { (*cell).take() });
        self.slot().word.store(EMPTY, Ordering::Release);
        // Dropped after the slot is free: a driver call in the contents' drop does not hold the slot.
        drop(old);
    }
}

/// A taken instance. It goes back to the pool when it is dropped (so an early return keeps it), or is dropped for good by
/// [`discard`](Self::discard).
pub(crate) struct Lease<'a, K, T> {
    owned: Option<Owned<'a, K, T>>,
    key: Option<K>,
    value: Option<T>,
}

impl<K, T> Lease<'_, K, T> {
    pub(crate) fn get(&mut self) -> &mut T {
        self.value.as_mut().expect("a lease holds its instance")
    }

    /// Drops the instance instead of putting it back (it is not fit for another run).
    #[cfg(test)]
    pub(crate) fn discard(mut self) {
        self.value = None;
        self.key = None;
        // The owned slot drops empty.
        self.owned = None;
    }
}

impl<K, T> Drop for Lease<'_, K, T> {
    fn drop(&mut self) {
        if let (Some(owned), Some(key), Some(value)) =
            (self.owned.take(), self.key.take(), self.value.take())
        {
            owned.give(key, value);
        }
    }
}

/// A slot reserved to record an instance in: given back empty when dropped.
pub(crate) struct Reservation<'a, K, T> {
    owned: Owned<'a, K, T>,
}

impl<K, T> Reservation<'_, K, T> {
    /// Puts the recorded instance in the pool.
    pub(crate) fn fill(self, key: K, value: T) {
        self.owned.give(key, value);
    }
}

impl<K: Copy + Eq, T> InstancePool<K, T> {
    /// A pool of `slots` instances.
    pub(crate) fn new(slots: usize) -> Self {
        Self {
            slots: (0..slots.max(1))
                .map(|_| Slot {
                    word: AtomicU64::new(EMPTY),
                    cell: UnsafeCell::new(None),
                })
                .collect(),
        }
    }

    pub(crate) fn capacity(&self) -> usize {
        self.slots.len()
    }

    fn start(&self, tag: u64) -> usize {
        (tag % self.slots.len() as u64) as usize
    }

    /// Takes a free instance of `key`, if there is one. Never waits: a slot that is in use is passed over.
    pub(crate) fn take(&self, key: &K, tag: u64) -> Option<Lease<'_, K, T>> {
        let n = self.slots.len();
        let start = self.start(tag);
        for step in 0..n {
            let index = (start + step) % n;
            let slot = &self.slots[index];
            let free = word(tag, FREE);
            if slot.word.load(Ordering::Acquire) != free {
                continue;
            }
            if slot
                .word
                .compare_exchange(free, word(tag, BUSY), Ordering::AcqRel, Ordering::Relaxed)
                .is_err()
            {
                continue;
            }
            let owned = Owned {
                pool: self,
                index,
                tag,
            };
            // SAFETY: this thread owns the slot.
            let contents = slot.cell.with_mut(|cell| unsafe { (*cell).take() });
            match contents {
                Some((k, value)) if k == *key => {
                    return Some(Lease {
                        owned: Some(owned),
                        key: Some(k),
                        value: Some(value),
                    });
                }
                // The tag matched another key: the slot goes back as it was.
                Some((k, value)) => owned.give(k, value),
                None => {}
            }
        }
        None
    }

    /// Reserves a slot to record an instance of the key `tag` names: an empty one, else the first free instance (of any
    /// key), which goes. `None` when every slot is in use: the caller runs its recording without pooling it.
    pub(crate) fn reserve(&self, tag: u64) -> Option<Reservation<'_, K, T>> {
        let n = self.slots.len();
        let start = self.start(tag);
        for step in 0..n {
            let index = (start + step) % n;
            let slot = &self.slots[index];
            if slot.word.load(Ordering::Relaxed) == EMPTY
                && slot
                    .word
                    .compare_exchange(EMPTY, word(tag, BUSY), Ordering::AcqRel, Ordering::Relaxed)
                    .is_ok()
            {
                return Some(Reservation {
                    owned: Owned {
                        pool: self,
                        index,
                        tag,
                    },
                });
            }
        }
        // The pool is full: the bound evicts a free instance, any one.
        for step in 0..n {
            let index = (start + step) % n;
            let slot = &self.slots[index];
            let seen = slot.word.load(Ordering::Acquire);
            if seen & STATE != FREE {
                continue;
            }
            if slot
                .word
                .compare_exchange(seen, word(tag, BUSY), Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
            {
                // SAFETY: this thread owns the slot.
                let old = slot.cell.with_mut(|cell| unsafe { (*cell).take() });
                drop(old);
                return Some(Reservation {
                    owned: Owned {
                        pool: self,
                        index,
                        tag,
                    },
                });
            }
        }
        None
    }

    /// Whether an instance of the key `tag` names is in the pool, in use or not (a tag match: a hash collision between
    /// two keys is a false yes).
    pub(crate) fn contains(&self, tag: u64) -> bool {
        self.slots.iter().any(|s| {
            s.word.load(Ordering::Acquire) >> 2 == tag && s.word.load(Ordering::Relaxed) != EMPTY
        })
    }

    /// Drops the free instances whose key `keep` refuses; those in use stay for the next call. Returns how many went.
    pub(crate) fn retain(&self, keep: impl Fn(&K) -> bool) -> usize {
        let mut dropped = 0;
        for (index, slot) in self.slots.iter().enumerate() {
            let seen = slot.word.load(Ordering::Acquire);
            if seen & STATE != FREE {
                continue;
            }
            let tag = seen >> 2;
            if slot
                .word
                .compare_exchange(seen, word(tag, BUSY), Ordering::AcqRel, Ordering::Relaxed)
                .is_err()
            {
                continue;
            }
            let owned = Owned {
                pool: self,
                index,
                tag,
            };
            // SAFETY: this thread owns the slot.
            match slot.cell.with_mut(|cell| unsafe { (*cell).take() }) {
                Some((k, v)) if keep(&k) => owned.give(k, v),
                Some(_) => {
                    dropped += 1;
                    // The owned slot drops empty, and the instance with it.
                }
                None => {}
            }
        }
        dropped
    }

    /// Instances in the pool, in use or not.
    pub(crate) fn len(&self) -> usize {
        self.slots
            .iter()
            .filter(|s| s.word.load(Ordering::Acquire) != EMPTY)
            .count()
    }
}

#[cfg(all(test, not(loom)))]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;

    use super::*;

    fn tag(k: u32) -> u64 {
        tag_of(&k)
    }

    #[test]
    fn an_instance_is_taken_by_its_key_and_goes_back_when_dropped() {
        let pool = InstancePool::<u32, u32>::new(4);
        pool.reserve(tag(7)).unwrap().fill(7, 70);
        assert!(pool.take(&8, tag(8)).is_none(), "another key");
        let mut lease = pool.take(&7, tag(7)).expect("a free instance");
        assert_eq!(*lease.get(), 70);
        assert!(
            pool.take(&7, tag(7)).is_none(),
            "an instance in use is not taken twice"
        );
        *lease.get() = 71;
        drop(lease);
        assert_eq!(*pool.take(&7, tag(7)).unwrap().get(), 71);
        assert_eq!(pool.len(), 1);
    }

    #[test]
    fn a_second_instance_of_a_key_is_recorded_beside_the_first() {
        let pool = InstancePool::<u32, u32>::new(4);
        pool.reserve(tag(1)).unwrap().fill(1, 10);
        let first = pool.take(&1, tag(1)).unwrap();
        assert!(pool.take(&1, tag(1)).is_none());
        pool.reserve(tag(1)).unwrap().fill(1, 11);
        let mut second = pool.take(&1, tag(1)).expect("the second instance");
        assert_eq!(*second.get(), 11);
        drop((first, second));
        assert_eq!(pool.len(), 2);
    }

    #[test]
    fn a_full_pool_evicts_a_free_instance_and_refuses_when_all_are_in_use() {
        let pool = InstancePool::<u32, u32>::new(2);
        pool.reserve(tag(1)).unwrap().fill(1, 10);
        pool.reserve(tag(2)).unwrap().fill(2, 20);
        let busy = pool.take(&1, tag(1)).unwrap();
        pool.reserve(tag(3)).unwrap().fill(3, 30);
        assert!(pool.take(&2, tag(2)).is_none(), "the free one was evicted");
        let other = pool.take(&3, tag(3)).unwrap();
        assert!(
            pool.reserve(tag(4)).is_none(),
            "every slot is in use: nothing to evict"
        );
        drop((busy, other));
        assert_eq!(pool.len(), 2);
    }

    #[test]
    fn a_reservation_dropped_unfilled_frees_its_slot_and_a_discarded_lease_goes() {
        let pool = InstancePool::<u32, u32>::new(1);
        drop(pool.reserve(tag(1)).unwrap());
        assert_eq!(pool.len(), 0);
        pool.reserve(tag(1)).unwrap().fill(1, 10);
        pool.take(&1, tag(1)).unwrap().discard();
        assert_eq!(pool.len(), 0);
        assert!(!pool.contains(tag(1)));
    }

    #[test]
    fn retain_drops_the_free_instances_it_refuses_and_leaves_those_in_use() {
        let pool = InstancePool::<u32, u32>::new(4);
        for k in 1..=3u32 {
            pool.reserve(tag(k)).unwrap().fill(k, k * 10);
        }
        let in_use = pool.take(&2, tag(2)).unwrap();
        assert_eq!(
            pool.retain(|k| *k == 1),
            2 - 1,
            "key 3 goes; key 2 is in use"
        );
        assert!(pool.contains(tag(1)) && pool.contains(tag(2)) && !pool.contains(tag(3)));
        drop(in_use);
        assert_eq!(pool.retain(|k| *k == 1), 1);
        assert_eq!(pool.len(), 1);
    }

    struct Counted(Arc<AtomicUsize>);

    impl Drop for Counted {
        fn drop(&mut self) {
            self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
    }

    #[test]
    fn what_a_pool_drops_is_dropped_once() {
        let drops = Arc::new(AtomicUsize::new(0));
        {
            let pool = InstancePool::<u32, Counted>::new(2);
            pool.reserve(tag(1))
                .unwrap()
                .fill(1, Counted(drops.clone()));
            pool.reserve(tag(2))
                .unwrap()
                .fill(2, Counted(drops.clone()));
            pool.reserve(tag(3))
                .unwrap()
                .fill(3, Counted(drops.clone()));
            assert_eq!(
                drops.load(std::sync::atomic::Ordering::SeqCst),
                1,
                "evicted"
            );
            pool.take(&3, tag(3)).unwrap().discard();
            assert_eq!(drops.load(std::sync::atomic::Ordering::SeqCst), 2);
        }
        assert_eq!(
            drops.load(std::sync::atomic::Ordering::SeqCst),
            3,
            "the pool's drop"
        );
    }

    /// Threads run the same keys at once: no instance is ever used by two threads, none waits, and every use finds or
    /// makes an instance.
    #[test]
    fn many_threads_never_share_an_instance_and_never_wait() {
        let pool = InstancePool::<u32, AtomicUsize>::new(64);
        let made = AtomicUsize::new(0);
        let (threads, rounds) = if cfg!(miri) { (3, 6) } else { (8, 2_000) };
        std::thread::scope(|scope| {
            for t in 0..threads {
                let (pool, made) = (&pool, &made);
                scope.spawn(move || {
                    for r in 0..rounds {
                        let key = ((t + r) % 3) as u32;
                        if let Some(mut lease) = pool.take(&key, tag(key)) {
                            // Exclusive use: the instance counts users, and it must be one.
                            let users = lease
                                .get()
                                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                            assert_eq!(users, 0, "two threads in one instance");
                            lease
                                .get()
                                .fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
                        } else if let Some(slot) = pool.reserve(tag(key)) {
                            made.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            slot.fill(key, AtomicUsize::new(0));
                        }
                    }
                });
            }
        });
        assert!(pool.len() >= 3 && pool.len() <= 64);
        assert!(made.load(std::sync::atomic::Ordering::Relaxed) >= 3);
    }
}

/// The pool's handshake under every interleaving the memory model allows.
#[cfg(all(loom, test))]
mod loom_tests {
    use super::*;
    use loom::sync::Arc;
    use loom::sync::atomic::AtomicUsize;
    use loom::thread;

    // Two threads take the one instance of a key: at most one gets it, the instance's contents written by its last owner
    // are seen by the next, and what is not taken is still there after.
    #[test]
    fn loom_one_instance_is_taken_by_one_thread_and_its_contents_are_ordered() {
        loom::model(|| {
            let pool = Arc::new(InstancePool::<u32, Arc<AtomicUsize>>::new(2));
            let inner = Arc::new(AtomicUsize::new(0));
            pool.reserve(5).unwrap().fill(5, inner.clone());
            let taken = Arc::new(AtomicUsize::new(0));
            let worker = {
                let (pool, taken) = (pool.clone(), taken.clone());
                thread::spawn(move || {
                    if let Some(mut lease) = pool.take(&5, 5) {
                        taken.fetch_add(1, loom::sync::atomic::Ordering::SeqCst);
                        let before = lease
                            .get()
                            .fetch_add(1, loom::sync::atomic::Ordering::SeqCst);
                        assert!(before <= 1);
                    }
                })
            };
            if let Some(mut lease) = pool.take(&5, 5) {
                taken.fetch_add(1, loom::sync::atomic::Ordering::SeqCst);
                let before = lease
                    .get()
                    .fetch_add(1, loom::sync::atomic::Ordering::SeqCst);
                assert!(before <= 1);
            }
            worker.join().unwrap();
            // The instance was given back by whoever took it: it is takeable again, and the count shows the users.
            let mut again = pool.take(&5, 5).expect("the instance is back");
            let seen = again.get().load(loom::sync::atomic::Ordering::SeqCst);
            assert_eq!(seen, taken.load(loom::sync::atomic::Ordering::SeqCst));
        });
    }

    // A thread that finds the key's instance taken reserves a slot of its own and fills it: the two instances are
    // different slots, and the table never loses one.
    #[test]
    fn loom_a_thread_that_finds_the_instance_in_use_records_another() {
        loom::model(|| {
            let pool = Arc::new(InstancePool::<u32, u32>::new(2));
            pool.reserve(9).unwrap().fill(9, 1);
            let worker = {
                let pool = pool.clone();
                thread::spawn(move || {
                    if pool.take(&9, 9).is_none() {
                        if let Some(slot) = pool.reserve(9) {
                            slot.fill(9, 2);
                        }
                    }
                })
            };
            let mine = pool.take(&9, 9);
            drop(mine);
            worker.join().unwrap();
            assert!(pool.len() >= 1 && pool.len() <= 2);
            assert!(
                pool.take(&9, 9).is_some(),
                "an instance of the key is there"
            );
        });
    }
}
