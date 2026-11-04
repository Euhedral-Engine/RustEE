//! The instances of recorded graphs, pooled by key, for any number of threads to take, use and give back without a lock.
//!
//! A recorded graph is used by one thread at a time (its executable takes new arguments and launches), so a key that
//! several threads run at once needs several instances of its graph. The pool is a fixed table of slots; a thread that
//! wants the graph of a key *takes* a free instance of it by one compare-and-set, and a thread that finds none does not
//! wait for another's: it *reserves* a slot, records an instance of its own, and puts it in the pool for the next. Nothing
//! spins and nothing allocates after the table is made: a slot's contents move in and out of it.
//!
//! The state any thread may touch is one word per slot, `(tag << 2) | state`, and one stamp, and only that:
//! - **`EMPTY`** (the word is 0): no instance, free to reserve.
//! - **`FREE`**: an instance of the key whose hash is `tag` is in the slot, and nobody uses it.
//! - **`BUSY`**: a thread owns the slot and uses its instance (tag says which key).
//! - **`RECORDING`**: a thread owns the slot and is recording the instance that will be put in it. The key is not there
//!   yet: it does not count as present.
//!
//! Only the owner touches a slot's cell, and it hands the slot on with a release store of the new word; a thread that
//! takes a slot does so by a compare-and-set that acquires that store, so the cell's contents are ordered between owners.
//!
//! A tag is a hash of the key, so a tag match is confirmed against the key once the slot is taken; the slot is given back
//! at once when it does not match. A scan therefore never blocks another thread on a slot it does not want, except for
//! the instant between its two stores.
//!
//! **Eviction is approximately least recently used**, as Java's `MAX_CAPTURES` map is least recently used: each slot keeps
//! the tick of its last take, and a full pool evicts the free instance with the oldest one. The tick is a coarse clock that
//! advances only when an instance is filled, so a take costs a read and a store, nothing shared is read-modify-written
//! per run. An instance in use is never evicted.

#![allow(
    unsafe_code,
    reason = "a slot's cell is accessed only by the thread that owns the slot, which the slot's word orders"
)]

use std::hash::{Hash, Hasher};

use crate::sync::{AtomicU64, Ordering, UnsafeCell};

const EMPTY: u64 = 0;
const FREE: u64 = 1;
const BUSY: u64 = 2;
const RECORDING: u64 = 3;
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
    /// The tick of the last take or fill: the least recent free slot is the one a full pool evicts. Not an ordering: only
    /// a hint, so relaxed.
    stamp: AtomicU64,
    cell: UnsafeCell<Option<(K, T)>>,
}

/// A fixed table of instances, by key.
pub(crate) struct InstancePool<K, T> {
    slots: Box<[Slot<K, T>]>,
    clock: AtomicU64,
}

// SAFETY: a slot's cell is accessed only by the thread that owns the slot (its word says BUSY or RECORDING, and the
// thread that made it so by compare-and-set), and ownership passes by release/acquire on the word; the contents move
// between threads, so they must be `Send`.
unsafe impl<K: Send, T: Send> Sync for InstancePool<K, T> {}

impl<K, T> std::fmt::Debug for InstancePool<K, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InstancePool")
            .field("slots", &self.slots.len())
            .finish()
    }
}

/// A taken instance. It is the slot's owner: dropped, it frees the slot (also when the thread unwinds).
pub(crate) struct Lease<'a, K, T> {
    pool: &'a InstancePool<K, T>,
    index: usize,
    tag: u64,
}

impl<K, T> Lease<'_, K, T> {
    /// The instance.
    pub(crate) fn get(&mut self) -> &mut T {
        // SAFETY: this lease owns the slot, and holds the instance in it.
        self.pool.slots[self.index].cell.with_mut(|cell| unsafe {
            &mut (*cell)
                .as_mut()
                .expect("a leased slot holds its instance")
                .1
        })
    }

    /// Drops the instance instead of putting it back (it is not fit for another run).
    pub(crate) fn discard(self) {
        // SAFETY: this lease owns the slot.
        let old = self.pool.slots[self.index]
            .cell
            .with_mut(|cell| unsafe { (*cell).take() });
        self.pool.slots[self.index]
            .word
            .store(EMPTY, Ordering::Release);
        std::mem::forget(self);
        drop(old);
    }
}

impl<K, T> Drop for Lease<'_, K, T> {
    fn drop(&mut self) {
        self.pool.slots[self.index]
            .word
            .store(word(self.tag, FREE), Ordering::Release);
    }
}

/// A slot reserved to record an instance in: given back empty when dropped.
pub(crate) struct Reservation<'a, K, T> {
    pool: &'a InstancePool<K, T>,
    index: usize,
    tag: u64,
    filled: bool,
}

impl<K, T> Reservation<'_, K, T> {
    /// Puts the recorded instance in the pool, free.
    pub(crate) fn fill(mut self, key: K, value: T) {
        // SAFETY: this reservation owns the slot.
        let put = |cell: *mut Option<(K, T)>| unsafe { *cell = Some((key, value)) };
        self.pool.slots[self.index].cell.with_mut(put);
        self.filled = true;
        self.pool.advance(self.index);
        self.pool.slots[self.index]
            .word
            .store(word(self.tag, FREE), Ordering::Release);
    }
}

impl<K, T> Drop for Reservation<'_, K, T> {
    fn drop(&mut self) {
        if !self.filled {
            // SAFETY: this reservation owns the slot.
            let old = self.pool.slots[self.index]
                .cell
                .with_mut(|cell| unsafe { (*cell).take() });
            self.pool.slots[self.index]
                .word
                .store(EMPTY, Ordering::Release);
            drop(old);
        }
    }
}

impl<K, T> InstancePool<K, T> {
    /// A take marks its slot with the current tick: a read of the clock and a store to the slot, no read-modify-write on
    /// anything shared. The clock advances when an instance is filled (a recording, which is rare), so instances used since
    /// the last fill share a tick and ties go to the scan order: the choice of a victim is approximately least recently
    /// used, and a wrong choice only records an instance again.
    fn touch(&self, index: usize) {
        let tick = self.clock.load(Ordering::Relaxed);
        self.slots[index].stamp.store(tick, Ordering::Relaxed);
    }

    /// A fill advances the clock and marks its slot with the new tick.
    fn advance(&self, index: usize) {
        let tick = self.clock.fetch_add(1, Ordering::Relaxed) + 1;
        self.slots[index].stamp.store(tick, Ordering::Relaxed);
    }
}

impl<K: Copy + Eq, T> InstancePool<K, T> {
    /// A pool of `slots` instances.
    pub(crate) fn new(slots: usize) -> Self {
        Self {
            slots: (0..slots.max(1))
                .map(|_| Slot {
                    word: AtomicU64::new(EMPTY),
                    stamp: AtomicU64::new(0),
                    cell: UnsafeCell::new(None),
                })
                .collect(),
            clock: AtomicU64::new(0),
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
            let lease = Lease {
                pool: self,
                index,
                tag,
            };
            // SAFETY: this thread owns the slot.
            let same = slot
                .cell
                .with(|cell| unsafe { matches!(&*cell, Some((k, _)) if k == key) });
            // The tag matched another key: the slot goes back as it was (the lease frees it).
            if same {
                self.touch(index);
                return Some(lease);
            }
        }
        None
    }

    /// Reserves a slot to record an instance of the key `tag` names: an empty one, else the least recently used free
    /// instance (of any key), which goes. `None` when every slot is in use or being recorded: the caller runs without
    /// pooling.
    pub(crate) fn reserve(&self, tag: u64) -> Option<Reservation<'_, K, T>> {
        let n = self.slots.len();
        let start = self.start(tag);
        for step in 0..n {
            let index = (start + step) % n;
            let slot = &self.slots[index];
            if slot.word.load(Ordering::Relaxed) == EMPTY
                && slot
                    .word
                    .compare_exchange(
                        EMPTY,
                        word(tag, RECORDING),
                        Ordering::AcqRel,
                        Ordering::Relaxed,
                    )
                    .is_ok()
            {
                return Some(self.reservation(index, tag));
            }
        }
        // The pool is full: the bound evicts the least recently used free instance. Another thread may take or evict it
        // first; look again (bounded: each failed claim was another thread's progress).
        for _ in 0..n {
            let mut best: Option<(usize, u64, u64)> = None;
            for (index, slot) in self.slots.iter().enumerate() {
                let seen = slot.word.load(Ordering::Acquire);
                if seen & STATE != FREE {
                    continue;
                }
                let stamp = slot.stamp.load(Ordering::Relaxed);
                if best.is_none_or(|(_, _, oldest)| stamp < oldest) {
                    best = Some((index, seen, stamp));
                }
            }
            let (index, seen, _) = best?;
            let slot = &self.slots[index];
            if slot
                .word
                .compare_exchange(
                    seen,
                    word(tag, RECORDING),
                    Ordering::AcqRel,
                    Ordering::Relaxed,
                )
                .is_ok()
            {
                // SAFETY: this thread owns the slot.
                let old = slot.cell.with_mut(|cell| unsafe { (*cell).take() });
                drop(old);
                return Some(self.reservation(index, tag));
            }
        }
        None
    }

    fn reservation(&self, index: usize, tag: u64) -> Reservation<'_, K, T> {
        Reservation {
            pool: self,
            index,
            tag,
            filled: false,
        }
    }

    /// Whether an instance of the key `tag` names is in the pool, in use or not, and not still being recorded (a tag
    /// match: a hash collision between two keys is a false yes).
    pub(crate) fn contains(&self, tag: u64) -> bool {
        self.slots.iter().any(|s| {
            let seen = s.word.load(Ordering::Acquire);
            seen != EMPTY && seen & STATE != RECORDING && seen >> 2 == tag
        })
    }

    /// Drops the free instances whose key `keep` refuses; those in use stay for the next call. Returns how many went. A
    /// `keep` that panics leaves its slot free.
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
            // The lease frees the slot when `keep` unwinds as well as when it keeps the instance.
            let lease = Lease {
                pool: self,
                index,
                tag,
            };
            // SAFETY: this thread owns the slot.
            let kept = slot
                .cell
                .with(|cell| unsafe { matches!(&*cell, Some((k, _)) if keep(k)) });
            if kept {
                drop(lease);
            } else {
                lease.discard();
                dropped += 1;
            }
        }
        dropped
    }

    /// Instances in the pool, in use or not, and those being recorded.
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
    fn a_full_pool_evicts_the_free_instance_that_was_used_least_recently() {
        let pool = InstancePool::<u32, u32>::new(3);
        for k in 1..=3u32 {
            pool.reserve(tag(k)).unwrap().fill(k, k * 10);
        }
        // Use 1 and 3 again: 2 was not used since it was filled.
        drop(pool.take(&1, tag(1)).unwrap());
        drop(pool.take(&3, tag(3)).unwrap());
        pool.reserve(tag(4)).unwrap().fill(4, 40);
        assert!(
            pool.take(&2, tag(2)).is_none(),
            "the one not used since went"
        );
        assert!(pool.take(&1, tag(1)).is_some() && pool.take(&3, tag(3)).is_some());
    }

    #[test]
    fn a_key_still_being_recorded_is_not_present() {
        let pool = InstancePool::<u32, u32>::new(2);
        let recording = pool.reserve(tag(8)).unwrap();
        assert!(!pool.contains(tag(8)), "its recording has not ended");
        assert!(pool.take(&8, tag(8)).is_none());
        recording.fill(8, 80);
        assert!(pool.contains(tag(8)));
    }

    #[test]
    fn a_retain_that_panics_leaves_its_slot_free() {
        let pool = InstancePool::<u32, u32>::new(2);
        pool.reserve(tag(1)).unwrap().fill(1, 10);
        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            pool.retain(|_| panic!("the closure broke"))
        }));
        assert!(panicked.is_err());
        assert_eq!(*pool.take(&1, tag(1)).expect("still free").get(), 10);
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

    // Two threads take the one instance of a key: at most one gets it.
    #[test]
    fn loom_one_instance_is_taken_by_one_thread() {
        loom::model(|| {
            let pool = Arc::new(InstancePool::<u32, u32>::new(2));
            pool.reserve(5).unwrap().fill(5, 0);
            let taken = Arc::new(AtomicUsize::new(0));
            let worker = {
                let (pool, taken) = (pool.clone(), taken.clone());
                thread::spawn(move || {
                    if let Some(mut lease) = pool.take(&5, 5) {
                        taken.fetch_add(1, loom::sync::atomic::Ordering::SeqCst);
                        *lease.get() += 1;
                    }
                })
            };
            if let Some(mut lease) = pool.take(&5, 5) {
                taken.fetch_add(1, loom::sync::atomic::Ordering::SeqCst);
                *lease.get() += 1;
            }
            worker.join().unwrap();
            // Whoever took it gave it back: it is takeable again, and the count shows its users in turn.
            let mut again = pool.take(&5, 5).expect("the instance is back");
            assert_eq!(
                *again.get() as usize,
                taken.load(loom::sync::atomic::Ordering::SeqCst)
            );
        });
    }

    // The instance is used by one thread and given back (the lease drops): the next taker, whichever thread it is, sees
    // what the first user wrote, and never sees the instance before.
    #[test]
    fn loom_a_given_back_instance_hands_over_its_contents() {
        loom::model(|| {
            let pool = Arc::new(InstancePool::<u32, u32>::new(2));
            pool.reserve(7).unwrap().fill(7, 0);
            let user = {
                let pool = pool.clone();
                thread::spawn(move || {
                    let Some(mut lease) = pool.take(&7, 7) else {
                        return;
                    };
                    *lease.get() = 42;
                })
            };
            // Either the instance is in use, or it is back with the first user's write.
            if let Some(mut again) = pool.take(&7, 7) {
                let seen = *again.get();
                assert!(seen == 0 || seen == 42);
            }
            user.join().unwrap();
            assert!(matches!(
                *pool.take(&7, 7).expect("given back").get(),
                0 | 42
            ));
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
                    if pool.take(&9, 9).is_none()
                        && let Some(slot) = pool.reserve(9)
                    {
                        slot.fill(9, 2);
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
