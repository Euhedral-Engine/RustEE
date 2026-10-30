use super::loom::{AtomicBool, Ordering, fence};
use std::cell::UnsafeCell;
use std::fmt;
use std::marker::PhantomData;
use std::mem;
use std::ops::{Deref, DerefMut};

/// A claim that one thread at a time may hold, and that a thread that finds it held leaves alone.
///
/// A flag set by a compare-and-set and cleared by a store (Java's `draining` flag), over state the holder is
/// licensed to use. The cell holds that state and hands it out through a guard, so nothing outside a claim can
/// reach it.
///
/// A claimer that loses the compare-and-set never waits: it writes what it wants before it tries (a pending
/// request), and the holder looks for it after it lets go.
///
/// - [`try_claim`](Self::try_claim) is a compare-and-set (`SeqCst`), as a volatile one is.
/// - Dropping the guard stores `false` with release ordering, enough to hand the state to the next claimer.
/// - [`ClaimGuard::release_seq_cst`] stores `false` sequentially consistently and then fences. A `SeqCst` store
///   followed by a `SeqCst` load is ordered already; the fence orders the reads that are not `SeqCst` (a queue's
///   emptiness, a completion flag) after the release, which Java's volatile write does for the reads that follow
///   it.
pub struct ClaimCell<T> {
    claimed: AtomicBool,
    value: UnsafeCell<T>,
}

// SAFETY: the value is reached only through a guard, and the flag admits one guard at a time, so
// sharing the cell moves the `T` between threads without ever aliasing it.
unsafe impl<T: Send> Sync for ClaimCell<T> {}

impl<T> ClaimCell<T> {
    /// A free cell holding `value`.
    #[cfg(not(loom))]
    pub const fn new(value: T) -> Self {
        Self {
            claimed: AtomicBool::new(false),
            value: UnsafeCell::new(value),
        }
    }

    /// A free cell holding `value`.
    #[cfg(loom)]
    pub fn new(value: T) -> Self {
        Self {
            claimed: AtomicBool::new(false),
            value: UnsafeCell::new(value),
        }
    }

    /// Claims the state, or returns `None` when another claim (on this thread or another) is held.
    /// Never waits.
    pub fn try_claim(&self) -> Option<ClaimGuard<'_, T>> {
        self.claimed
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .ok()
            .map(|_| ClaimGuard {
                cell: self,
                marker: PhantomData,
            })
    }

    /// Whether a claim is held now; a hint, since it may change before the caller acts on it.
    pub fn is_claimed(&self) -> bool {
        self.claimed.load(Ordering::Relaxed)
    }

    /// The state, for the cell's exclusive owner.
    pub fn get_mut(&mut self) -> &mut T {
        self.value.get_mut()
    }

    /// The state, for the cell's owner.
    pub fn into_inner(self) -> T {
        self.value.into_inner()
    }
}

impl<T: Default> Default for ClaimCell<T> {
    fn default() -> Self {
        Self::new(T::default())
    }
}

impl<T> fmt::Debug for ClaimCell<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ClaimCell")
            .field("claimed", &self.is_claimed())
            .finish_non_exhaustive()
    }
}

/// The state of a [`ClaimCell`], held until the guard drops or [`release_seq_cst`](Self::release_seq_cst)
/// gives it back.
///
/// Like a mutex guard, it is `Send` when `T` is and `Sync` only when `T` is.
#[must_use = "dropping the guard at once releases the claim"]
pub struct ClaimGuard<'a, T> {
    cell: &'a ClaimCell<T>,
    /// Keeps the auto traits off; the impls below say exactly when the guard is `Send` and `Sync`.
    marker: PhantomData<*const ()>,
}

// SAFETY: the guard is the cell's one access to a `T`, so moving it moves `&mut T`, which is
// `Send` exactly when `T` is.
unsafe impl<T: Send> Send for ClaimGuard<'_, T> {}
// SAFETY: a shared reference to the guard gives `&T`, which is `Send` exactly when `T` is `Sync`.
unsafe impl<T: Sync> Sync for ClaimGuard<'_, T> {}

impl<T> ClaimGuard<'_, T> {
    /// Gives the claim back with a sequentially consistent store, then a sequentially consistent
    /// fence: the holder's next reads, `SeqCst` or not, are ordered after the release. A claimer that
    /// lost to this holder wrote its wakeup first, so the holder's re-check finds it or the claimer
    /// wins the next claim.
    pub fn release_seq_cst(self) {
        let cell = self.cell;
        // The guard has nothing else to drop.
        mem::forget(self);
        cell.claimed.store(false, Ordering::SeqCst);
        fence(Ordering::SeqCst);
    }
}

impl<T> Deref for ClaimGuard<'_, T> {
    type Target = T;

    fn deref(&self) -> &T {
        // SAFETY: the guard holds the cell's claim, so it is the only access to the value.
        unsafe { &*self.cell.value.get() }
    }
}

impl<T> DerefMut for ClaimGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        // SAFETY: as in `deref`, and `&mut self` makes the borrow exclusive.
        unsafe { &mut *self.cell.value.get() }
    }
}

impl<T> Drop for ClaimGuard<'_, T> {
    fn drop(&mut self) {
        self.cell.claimed.store(false, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool as StdAtomicBool, AtomicUsize, Ordering as StdOrdering};
    use std::sync::{Arc, Barrier};
    use std::thread;

    #[cfg(not(miri))]
    const ATTEMPTS: usize = 10_000;
    #[cfg(miri)]
    const ATTEMPTS: usize = 50;

    #[test]
    fn a_second_claim_fails_while_the_first_is_held() {
        let cell = ClaimCell::new(1u32);
        let first = cell.try_claim().expect("free");
        assert!(cell.is_claimed());
        assert!(
            cell.try_claim().is_none(),
            "nested: refused, never two guards"
        );
        let elsewhere = thread::scope(|scope| scope.spawn(|| cell.try_claim().is_none()).join());
        assert!(elsewhere.unwrap(), "another thread is refused too");
        assert_eq!(*first, 1);
    }

    #[test]
    fn a_released_claim_can_be_taken_again() {
        let cell = ClaimCell::new(String::from("a"));
        let mut guard = cell.try_claim().expect("free");
        guard.push('b');
        drop(guard);
        assert!(!cell.is_claimed());
        let mut guard = cell.try_claim().expect("freed by the drop");
        assert_eq!(*guard, "ab");
        guard.push('c');
        guard.release_seq_cst();
        assert!(!cell.is_claimed());
        assert_eq!(*cell.try_claim().expect("freed by release_seq_cst"), "abc");
    }

    #[test]
    fn claims_from_8_threads_never_overlap() {
        let cell = Arc::new(ClaimCell::new(()));
        let inside = Arc::new(StdAtomicBool::new(false));
        let claimed = Arc::new(AtomicUsize::new(0));
        let threads: Vec<_> = (0..8)
            .map(|_| {
                let (cell, inside, claimed) = (cell.clone(), inside.clone(), claimed.clone());
                thread::spawn(move || {
                    for _ in 0..ATTEMPTS {
                        if let Some(guard) = cell.try_claim() {
                            assert!(!inside.swap(true, StdOrdering::SeqCst), "two holders");
                            claimed.fetch_add(1, StdOrdering::Relaxed);
                            inside.store(false, StdOrdering::SeqCst);
                            drop(guard);
                        }
                    }
                })
            })
            .collect();
        for thread in threads {
            thread.join().unwrap();
        }
        assert!(claimed.load(StdOrdering::Relaxed) > 0);
    }

    #[test]
    fn writes_under_one_claim_are_seen_by_the_next() {
        let cell = Arc::new(ClaimCell::new(0u64));
        let barrier = Arc::new(Barrier::new(4));
        let each = ATTEMPTS / 10;
        let threads: Vec<_> = (0..4)
            .map(|id| {
                let (cell, barrier) = (cell.clone(), barrier.clone());
                thread::spawn(move || {
                    barrier.wait();
                    let mut done = 0;
                    while done < each {
                        if let Some(mut guard) = cell.try_claim() {
                            // A plain read-modify-write: a lost handoff loses an update.
                            let seen = *guard;
                            *guard = seen + 1;
                            done += 1;
                            if id % 2 == 0 {
                                guard.release_seq_cst();
                            }
                        }
                    }
                })
            })
            .collect();
        for thread in threads {
            thread.join().unwrap();
        }
        assert_eq!(*cell.try_claim().unwrap(), 4 * each as u64);
    }

    #[test]
    fn get_mut_and_into_inner_need_no_claim() {
        let mut cell = ClaimCell::new(String::from("a"));
        cell.get_mut().push('b');
        assert_eq!(cell.into_inner(), "ab");
        assert_eq!(*ClaimCell::<u8>::default().try_claim().unwrap(), 0);
        assert_eq!(
            format!("{:?}", ClaimCell::new(0)),
            "ClaimCell { claimed: false, .. }"
        );
    }

    /// The loser of a claim is told so, and the holder's release is what the loser's wakeup meets: a
    /// "pending" flag set before a failed claim is read by the holder after `release_seq_cst`.
    #[test]
    fn a_wakeup_set_before_a_failed_claim_is_seen_after_the_holders_release() {
        let rounds = ATTEMPTS / 10;
        for _ in 0..rounds {
            let cell = Arc::new(ClaimCell::new(()));
            let pending = Arc::new(StdAtomicBool::new(false));
            let served = Arc::new(AtomicUsize::new(0));
            let barrier = Arc::new(Barrier::new(2));
            let requester = {
                let (cell, pending, served, barrier) = (
                    cell.clone(),
                    pending.clone(),
                    served.clone(),
                    barrier.clone(),
                );
                thread::spawn(move || {
                    barrier.wait();
                    pending.store(true, StdOrdering::SeqCst);
                    if let Some(guard) = cell.try_claim() {
                        if pending.swap(false, StdOrdering::SeqCst) {
                            served.fetch_add(1, StdOrdering::SeqCst);
                        }
                        guard.release_seq_cst();
                        // The holder side of the same protocol.
                        while pending.load(StdOrdering::SeqCst) {
                            let Some(guard) = cell.try_claim() else {
                                return;
                            };
                            if pending.swap(false, StdOrdering::SeqCst) {
                                served.fetch_add(1, StdOrdering::SeqCst);
                            }
                            guard.release_seq_cst();
                        }
                    }
                })
            };
            let holder = {
                let (cell, pending, served, barrier) = (
                    cell.clone(),
                    pending.clone(),
                    served.clone(),
                    barrier.clone(),
                );
                thread::spawn(move || {
                    barrier.wait();
                    if let Some(guard) = cell.try_claim() {
                        guard.release_seq_cst();
                        while pending.load(StdOrdering::SeqCst) {
                            let Some(guard) = cell.try_claim() else {
                                return;
                            };
                            if pending.swap(false, StdOrdering::SeqCst) {
                                served.fetch_add(1, StdOrdering::SeqCst);
                            }
                            guard.release_seq_cst();
                        }
                    }
                })
            };
            requester.join().unwrap();
            holder.join().unwrap();
            assert_eq!(served.load(StdOrdering::SeqCst), 1, "the wakeup was lost");
        }
    }
}

#[cfg(all(loom, test))]
mod loom_tests {
    use super::*;
    use loom::cell::UnsafeCell as LoomCell;
    use loom::sync::Arc;
    use loom::thread;

    /// A value the claim protects. Loom tracks its plain accesses, so a claim that did not pass the
    /// previous holder's writes on shows up as a data race.
    struct Payload {
        value: LoomCell<u32>,
    }

    // SAFETY: the payload is reached only while a claim is held.
    unsafe impl Send for Payload {}

    // The wakeup handshake (a flag set before a failed claim, read after `release_seq_cst`) needs
    // sequentially consistent ordering across a store and a later load, which loom's model of
    // `SeqCst` (acquire-release) does not check; `a_wakeup_set_before_a_failed_claim_...` runs it on
    // real threads.
    #[test]
    fn loom_two_claimers_exclude_each_other_and_see_prior_writes() {
        loom::model(|| {
            let cell = Arc::new(ClaimCell::new(Payload {
                value: LoomCell::new(0),
            }));
            let claimers: Vec<_> = (0..2)
                .map(|_| {
                    let cell = cell.clone();
                    thread::spawn(move || match cell.try_claim() {
                        Some(guard) => {
                            // SAFETY: the claim is held.
                            guard.value.with_mut(|slot| unsafe { *slot += 1 });
                            true
                        }
                        None => false,
                    })
                })
                .collect();
            let wins = claimers
                .into_iter()
                .map(|t| t.join().unwrap())
                .filter(|&won| won)
                .count();
            assert!(wins >= 1, "a free cell is won by someone");
            let guard = cell.try_claim().expect("both claimers are done");
            // SAFETY: the claim is held.
            let total = guard.value.with(|slot| unsafe { *slot });
            assert_eq!(total as usize, wins);
        });
    }
}
