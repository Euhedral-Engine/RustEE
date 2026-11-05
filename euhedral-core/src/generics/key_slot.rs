//! The home of a source's key frame.
//!
//! A source never runs work in `request` or `pull`: workers execute frames, so that they can measure
//! what a frame costs and learn the shape of the workload. A source that has work to do (polling a
//! device or the operating system, finishing copies, detaching a set of sinks) therefore hands out a
//! *key frame* whose `execute` does it. The source owns one reusable frame, the key. It leaves the
//! source only through `request` and a push, and the work happens in `execute` on whichever worker
//! the frame was routed to. Once the frame has finished, its [`recycle`](crate::frames::Frame::recycle)
//! sends it home. While the key is out the source has no second one to push, so at most one worker is
//! ever in that work, and a `request` that finds the key away has nothing to push.
//!
//! [`KeySlot`] is the home: one atomic reference to the key. Taking it is a swap with `Acquire`, homing
//! it a store with `Release`, and a `request` that finds the key away reads one flag and does nothing
//! more.

use std::marker::PhantomData;
use std::ptr;
use std::sync::atomic::{AtomicPtr, Ordering};

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

    /// Sends the key home. The slot must be empty: there is one key, and only its holder homes it.
    pub fn home(&self, key: Box<K>) {
        debug_assert!(
            !self.is_home(),
            "a key was homed while the slot already held one"
        );
        self.home.store(Box::into_raw(key), Ordering::Release);
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
}
