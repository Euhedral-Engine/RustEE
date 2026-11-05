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
use std::sync::atomic::{AtomicPtr, AtomicU32, AtomicU64, Ordering};
use std::time::{Duration, Instant};

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

/// The back-off of a key that polls blind: one that asks a device or the operating system whether something ended, with
/// nothing to say when it does.
///
/// A key pushed at every visit while nothing ends would keep a worker busy for ever. After a poll that found nothing the
/// source pushes the key again only after a **time** window: [`BASE_WINDOW`](Self::BASE_WINDOW), twice as long after each
/// further empty poll, up to [`MAX_WINDOW`](Self::MAX_WINDOW). Between pushes a visit pushes nothing, so the handle reads
/// unproductive and the worker parks. The window is time, not a count of visits, because a visit's interval varies by
/// two orders of magnitude with the load: the latency a poll adds stays within the window, whatever the load. It is
/// small on purpose (tens of microseconds).
///
/// Two things end the wait at once, and a source must not leave them to the window:
/// - **Events.** The source keeps a counter of the things that give the key something new to look at (a registration, a
///   submission, a move, a bell that rang). The key records the count before it polls ([`polling`](Self::polling)); a
///   poll that found nothing backs off only if the count did not move since ([`empty`](Self::empty)), and a visit
///   finds the count moved ([`fresh`](Self::fresh)). A counter, not a flag that the registering thread resets, so a
///   registration that races a poll is never lost.
/// - **A known time.** A wait that has a deadline (a timer) needs no polling before it: the source compares the
///   deadline with the clock itself.
///
/// The key's `execute` calls [`polling`](Self::polling) first, then [`found`](Self::found) or [`empty`](Self::empty).
/// The source's visits call [`fresh`](Self::fresh) and [`ready`](Self::ready), which the lattice serializes. The fields
/// are atomics only because the key runs on another thread; each is written by the key alone and a stale read costs one
/// visit. Nothing here loops or takes a lock; the clock is read only while a window is open.
#[derive(Debug)]
pub struct KeyPacing {
    epoch: Instant,
    /// Nanoseconds since `epoch`, plus one, before which a blind poll is not due; 0 when none is waited for.
    not_before: AtomicU64,
    /// Empty polls in a row, capped where the window reaches the maximum.
    step: AtomicU32,
    /// The event count the last poll started from.
    polled: AtomicU64,
}

impl KeyPacing {
    /// The wait after the first empty poll.
    pub const BASE_WINDOW: Duration = Duration::from_micros(16);
    /// The longest wait between two blind polls.
    pub const MAX_WINDOW: Duration = Duration::from_micros(100);

    pub fn new() -> Self {
        Self {
            epoch: Instant::now(),
            not_before: AtomicU64::new(0),
            step: AtomicU32::new(0),
            polled: AtomicU64::new(0),
        }
    }

    fn nanos(&self, at: Instant) -> u64 {
        u64::try_from(at.saturating_duration_since(self.epoch).as_nanos()).unwrap_or(u64::MAX - 1)
    }

    /// The key starts a poll that will have seen `events`.
    pub fn polling(&self, events: u64) {
        self.polled.store(events, Ordering::Relaxed);
    }

    /// Whether `events` moved since the last poll started: something new to look at, whatever the window.
    pub fn fresh(&self, events: u64) -> bool {
        events != self.polled.load(Ordering::Relaxed)
    }

    /// Whether the window of the last empty poll is over. Reads the clock only while a window is open.
    pub fn ready(&self) -> bool {
        self.ready_at(Instant::now())
    }

    /// [`ready`](Self::ready) at `now`.
    pub fn ready_at(&self, now: Instant) -> bool {
        let not_before = self.not_before.load(Ordering::Relaxed);
        not_before == 0 || self.nanos(now) + 1 >= not_before
    }

    /// Whether a blind poll is due: something new, or the window is over.
    pub fn due(&self, events: u64) -> bool {
        self.fresh(events) || self.ready()
    }

    /// The poll found something: poll again at the next visit.
    pub fn found(&self) {
        if self.not_before.load(Ordering::Relaxed) != 0 || self.step.load(Ordering::Relaxed) != 0 {
            self.step.store(0, Ordering::Relaxed);
            self.not_before.store(0, Ordering::Relaxed);
        }
    }

    /// The poll found nothing, with `events` counted at its end: if the count moved since it started, something new
    /// arrived meanwhile and may not have been seen, so there is no back-off; otherwise the next window opens.
    pub fn empty(&self, events: u64) {
        self.empty_at(events, Instant::now());
    }

    /// [`empty`](Self::empty) at `now`.
    pub fn empty_at(&self, events: u64, now: Instant) {
        if self.fresh(events) {
            self.found();
            return;
        }
        let step = self.step.load(Ordering::Relaxed);
        let window = Self::BASE_WINDOW
            .saturating_mul(1u32.checked_shl(step).unwrap_or(u32::MAX))
            .min(Self::MAX_WINDOW);
        if window < Self::MAX_WINDOW {
            self.step.store(step + 1, Ordering::Relaxed);
        }
        let until = self.nanos(now) + u64::try_from(window.as_nanos()).unwrap_or(u64::MAX / 2) + 1;
        self.not_before.store(until, Ordering::Relaxed);
    }
}

impl Default for KeyPacing {
    fn default() -> Self {
        Self::new()
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
    fn the_window_doubles_after_each_empty_poll_up_to_the_cap_and_a_find_ends_it() {
        let pacing = KeyPacing::new();
        let t0 = Instant::now();
        assert!(pacing.ready_at(t0), "nothing was empty yet");
        let us = Duration::from_micros;
        let mut windows = Vec::new();
        for _ in 0..8 {
            pacing.polling(0);
            pacing.empty_at(0, t0);
            // The window is the time until the poll is due: probe it.
            let mut waited = 0;
            while !pacing.ready_at(t0 + us(waited)) {
                waited += 1;
            }
            windows.push(waited);
        }
        assert_eq!(windows, [16, 32, 64, 100, 100, 100, 100, 100]);
        pacing.empty_at(0, t0);
        assert!(!pacing.ready_at(t0 + us(99)));
        pacing.found();
        assert!(pacing.ready_at(t0), "a find polls at the next visit");
        pacing.empty_at(0, t0);
        assert!(!pacing.ready_at(t0 + us(15)));
        assert!(pacing.ready_at(t0 + us(16)), "and the window starts again");
    }

    #[test]
    fn an_event_that_races_a_poll_is_never_lost() {
        let pacing = KeyPacing::new();
        let t0 = Instant::now();
        // The poll starts having seen 3 events; a fourth arrives while it runs; it finds nothing.
        pacing.polling(3);
        pacing.empty_at(4, t0);
        assert!(
            pacing.ready_at(t0),
            "no back-off: the fourth may not have been seen"
        );
        assert!(pacing.fresh(4), "and the next visit sees it");
        // The next poll sees all 4 and finds nothing: now it backs off.
        pacing.polling(4);
        pacing.empty_at(4, t0);
        assert!(!pacing.fresh(4));
        assert!(!pacing.ready_at(t0));
        // A fifth event ends the wait at once, whatever the window.
        assert!(pacing.fresh(5));
        assert!(pacing.due(5));
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
