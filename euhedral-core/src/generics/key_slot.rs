//! The home of a source's key frame.
//!
//! A source never runs work in `request` or `pull`: workers execute frames, so that they can measure
//! what a frame costs and learn the shape of the workload. A source that has work to do (polling a
//! device or the operating system, finishing copies, detaching a set of sinks) therefore hands out a
//! *key frame* whose `execute` does it. The source owns one reusable frame, the key. It leaves the
//! source by a push from `request` or `pull`, and the work happens in `execute` on whichever worker
//! the frame was routed to. Once the frame has finished, its [`recycle`](crate::frames::Frame::recycle)
//! sends it home. While the key is out the source has no second one to push, so at most one worker is
//! ever in that work, and a `request` or `pull` that finds the key away has nothing to push.
//!
//! [`KeySlot`] is the home: one atomic reference to the key. Taking it is a swap with `Acquire`, homing
//! it a swap (`AcqRel`, so that a second key is found and dropped, never leaked), and a `request` or `pull` that finds the key away reads one flag and does nothing
//! more.
//!
//! A key that **polls** (asks a device or the operating system whether something ended) must not be pushed at
//! every visit while nothing ends: a pushed key reads as a productive handle and counts as work the worker ran, so
//! the worker would never park. [`KeyPacing`] is the back-off a polling source keeps beside its key.

use std::marker::PhantomData;
use std::ptr;
use std::sync::atomic::{AtomicPtr, AtomicU32, Ordering};

/// The home of one key: empty while the key is away.
///
/// The slot owns the key it holds and frees it when it drops. A key that is away is owned by whoever
/// took it, and its holder either homes it again or drops it.
pub struct KeySlot<K> {
    home: AtomicPtr<K>,
    /// The slot owns a `Box<K>` while it holds a key.
    owns: PhantomData<Box<K>>,
}

impl<K> KeySlot<K> {
    /// A slot with no key: the key is away from the start, or there is none.
    pub const fn empty() -> Self {
        Self {
            home: AtomicPtr::new(ptr::null_mut()),
            owns: PhantomData,
        }
    }

    /// A slot with `key` at home.
    pub fn with_key(key: Box<K>) -> Self {
        Self {
            home: AtomicPtr::new(Box::into_raw(key)),
            owns: PhantomData,
        }
    }

    /// Whether the key is at home now. A hint: it may leave or return right after the read.
    pub fn is_home(&self) -> bool {
        !self.home.load(Ordering::Relaxed).is_null()
    }

    /// Takes the key out, or `None` if it is away. Reads the flag first, so a key that is away costs
    /// no read-modify-write.
    pub fn take(&self) -> Option<Box<K>> {
        if !self.is_home() {
            return None;
        }
        let key = self.home.swap(ptr::null_mut(), Ordering::Acquire);
        // SAFETY: a non-null pointer in the slot was made by `Box::into_raw` in `with_key` or `home`,
        // and the swap took it out, so this call alone owns the allocation now.
        (!key.is_null()).then(|| unsafe { Box::from_raw(key) })
    }

    /// Sends the key home. The slot must be empty: there is one key, and only its holder homes it. (A second key,
    /// a bug, is dropped in place of the first in every build, and fails a debug build; it is never leaked.)
    pub fn home(&self, key: Box<K>) {
        let previous = self.home.swap(Box::into_raw(key), Ordering::AcqRel);
        debug_assert!(
            previous.is_null(),
            "a key was homed while the slot already held one"
        );
        if !previous.is_null() {
            // SAFETY: a non-null pointer in the slot is a `Box::into_raw` from `with_key` or `home`, and the swap
            // took it out, so this call alone owns the allocation now.
            drop(unsafe { Box::from_raw(previous) });
        }
    }
}

/// The back-off of a key that polls: after a poll that finds nothing the source skips visits before it pushes the key
/// again, twice as many as after the previous empty poll, up to [`MAX_WINDOW`](Self::MAX_WINDOW), and any find, or a
/// new thing to wait for, ends it.
///
/// A visit is a `request` or a `pull` of the source. The count is of visits, not of time, so it is deterministic and
/// costs a few relaxed loads and stores; nothing here is a clock or a timer. Between pushes the visit pushes nothing,
/// so the handle reads unproductive and the worker parks as it did when the poll ran in the source and found nothing.
/// The cost is latency: an operation that ends while the window is at its cap is found within that many visits.
///
/// The key's `execute` calls [`found`](Self::found) or [`empty`](Self::empty); the source's visits call
/// [`due`](Self::due), which the lattice serializes; whatever registers a new wait calls [`reset`](Self::reset).
/// The fields are atomics only because the key runs on another thread; each is written by the side that owns it and
/// a stale read costs one visit.
#[derive(Debug, Default)]
pub struct KeyPacing {
    /// Visits left to skip.
    skip: AtomicU32,
    /// Visits skipped after the next empty poll (0: none yet).
    window: AtomicU32,
}

impl KeyPacing {
    /// The most visits skipped between two pushes.
    pub const MAX_WINDOW: u32 = 256;

    pub const fn new() -> Self {
        Self {
            skip: AtomicU32::new(0),
            window: AtomicU32::new(0),
        }
    }

    /// A visit asks whether to push the key now: `true` once the skipped visits have been spent. Call it only when
    /// there is something to poll for.
    pub fn due(&self) -> bool {
        let skip = self.skip.load(Ordering::Relaxed);
        if skip == 0 {
            return true;
        }
        self.skip.store(skip - 1, Ordering::Relaxed);
        false
    }

    /// The poll found nothing: the next window is twice the last, up to the cap.
    pub fn empty(&self) {
        let window = self
            .window
            .load(Ordering::Relaxed)
            .saturating_mul(2)
            .clamp(1, Self::MAX_WINDOW);
        self.window.store(window, Ordering::Relaxed);
        self.skip.store(window, Ordering::Relaxed);
    }

    /// The poll found something: poll again at the next visit.
    pub fn found(&self) {
        self.reset();
    }

    /// There is something new to wait for: poll at the next visit.
    pub fn reset(&self) {
        if self.window.load(Ordering::Relaxed) != 0 || self.skip.load(Ordering::Relaxed) != 0 {
            self.window.store(0, Ordering::Relaxed);
            self.skip.store(0, Ordering::Relaxed);
        }
    }
}

impl<K> Drop for KeySlot<K> {
    fn drop(&mut self) {
        let key = *self.home.get_mut();
        if !key.is_null() {
            // SAFETY: the slot owns what it holds (see `take`), and `&mut self` means no other thread
            // can take it.
            drop(unsafe { Box::from_raw(key) });
        }
    }
}

impl<K> Default for KeySlot<K> {
    fn default() -> Self {
        Self::empty()
    }
}

impl<K> std::fmt::Debug for KeySlot<K> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KeySlot")
            .field("home", &self.is_home())
            .finish()
    }
}

// SAFETY: the slot hands a `Box<K>` from the thread that homes it to the thread that takes it, which
// `K: Send` allows; no shared reference to a `K` ever exists, so `K` needs no `Sync`.
unsafe impl<K: Send> Sync for KeySlot<K> {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};
    use std::thread;

    /// Counts its drops.
    struct Key(Arc<AtomicUsize>);

    impl Drop for Key {
        fn drop(&mut self) {
            self.0.fetch_add(1, SeqCst);
        }
    }

    #[test]
    fn the_key_is_taken_once_and_comes_back_when_homed() {
        let drops = Arc::new(AtomicUsize::new(0));
        let slot = KeySlot::with_key(Box::new(Key(drops.clone())));
        assert!(slot.is_home());
        let key = slot.take().expect("at home");
        assert!(!slot.is_home());
        assert!(slot.take().is_none(), "away: nothing to take");
        slot.home(key);
        assert!(slot.is_home());
        assert_eq!(drops.load(SeqCst), 0, "homing frees nothing");
    }

    #[test]
    fn an_empty_slot_has_no_key_and_takes_one_home() {
        let drops = Arc::new(AtomicUsize::new(0));
        let slot = KeySlot::<Key>::empty();
        assert!(slot.take().is_none());
        slot.home(Box::new(Key(drops.clone())));
        assert!(slot.take().is_some(), "the key was dropped by the take");
        assert_eq!(drops.load(SeqCst), 1);
    }

    #[test]
    fn the_slot_frees_the_key_it_holds_and_not_one_that_is_away() {
        let drops = Arc::new(AtomicUsize::new(0));
        let slot = KeySlot::with_key(Box::new(Key(drops.clone())));
        drop(slot);
        assert_eq!(drops.load(SeqCst), 1, "a key at home goes with the slot");
        let slot = KeySlot::with_key(Box::new(Key(drops.clone())));
        let away = slot.take().expect("at home");
        drop(slot);
        assert_eq!(drops.load(SeqCst), 1, "a key that is away is its holder's");
        drop(away);
        assert_eq!(drops.load(SeqCst), 2);
    }

    #[test]
    fn eight_threads_never_hold_the_key_at_once() {
        const ROUNDS: usize = if cfg!(miri) { 20 } else { 20_000 };
        // The key holds a counter that only its holder touches, as a source's state would be.
        let slot = Arc::new(KeySlot::with_key(Box::new(0_u64)));
        let threads: Vec<_> = (0..8)
            .map(|_| {
                let slot = slot.clone();
                thread::spawn(move || {
                    let mut held = 0_u64;
                    for _ in 0..ROUNDS {
                        if let Some(mut key) = slot.take() {
                            // A plain read-modify-write: a second holder would lose updates.
                            let v = *key;
                            *key = v + 1;
                            held += 1;
                            slot.home(key);
                        }
                    }
                    held
                })
            })
            .collect();
        let total: u64 = threads.into_iter().map(|t| t.join().unwrap()).sum();
        assert_eq!(*slot.take().expect("home at the end"), total);
    }

    #[test]
    fn the_back_off_doubles_after_each_empty_poll_up_to_the_cap_and_a_find_ends_it() {
        let pacing = KeyPacing::new();
        assert!(pacing.due(), "nothing was empty yet");
        let mut windows = Vec::new();
        for _ in 0..12 {
            pacing.empty();
            let mut skipped = 0;
            while !pacing.due() {
                skipped += 1;
            }
            windows.push(skipped);
        }
        assert_eq!(windows, [1, 2, 4, 8, 16, 32, 64, 128, 256, 256, 256, 256]);
        pacing.empty();
        assert!(!pacing.due());
        pacing.found();
        assert!(pacing.due(), "a find polls at the next visit");
        pacing.empty();
        let mut skipped = 0;
        while !pacing.due() {
            skipped += 1;
        }
        assert_eq!(skipped, 1, "and the window starts again");
        pacing.empty();
        pacing.empty();
        pacing.reset();
        assert!(
            pacing.due(),
            "a new thing to wait for polls at the next visit"
        );
    }

    #[test]
    #[cfg(not(debug_assertions))]
    fn a_second_key_is_dropped_not_leaked_in_a_release_build() {
        let drops = Arc::new(AtomicUsize::new(0));
        let slot = KeySlot::with_key(Box::new(Key(drops.clone())));
        slot.home(Box::new(Key(drops.clone())));
        assert_eq!(drops.load(SeqCst), 1, "the first was dropped");
        drop(slot);
        assert_eq!(drops.load(SeqCst), 2);
    }
}
