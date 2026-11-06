//! The home of a source's key frame.
//!
//! `request` and `pull` run at every visit of a worker, so they do no heavy or high-frequency work: a small
//! continuation (O(1) bookkeeping, handing out a result that is ready) may run there, but anything that costs more or
//! recurs at the rate of the visits would hide from the workers, which measure what a frame costs and learn the shape
//! of the workload only from the frames they execute. A source that has such work (polling a device or the operating
//! system, finishing copies, detaching a set of sinks) therefore hands out a *key frame* whose `execute` does it.
//!
//! The source owns one reusable frame, the key. A worker that is run the frame does the work, and the frame's
//! [`recycle`](crate::frames::Frame::recycle) sends it home. While the key is out the source has no second one, so at most
//! one worker is ever in that work.
//!
//! **A key leaves a source only through `request`, which pushes it** to the downstream; the downstream routes it to a
//! worker (it is unordered, so any worker). A `pull` never pushes, and it never hands the key to its consumer either: the
//! pulling worker would run the frame through its executor while it still holds the source's handle, so that nothing
//! else could take ready work from the source until the work is done, and the work behind a key is long (a poll of every
//! registration, a system call, a device query per copy, a lock and the completion of every sink).
//!
//! [`KeySlot`] is the home: one atomic reference to the key. Taking it ([`KeySlot::take`]) is one swap with `Acquire`
//! and a null result means the key is away (there is no flag to read first); homing it is a swap (`AcqRel`, so that a
//! key found at home is dropped and never leaked). A key that is dropped without having run is replaced by a new one
//! ([`KeySlot::home_replacement`]), which never displaces what is there.
//!
//! A key that **polls** (asks a device or the operating system whether something ended) must not be pushed at
//! every visit while nothing ends: a pushed key reads as a productive handle and counts as work the worker ran, so
//! the worker would never park. [`KeyPacing`] is the back-off a polling source keeps beside its key. A request that
//! pushes the key shows the handle productive for that service only; the next request, which pushes nothing, shows it
//! empty again, because a request's only evidence of productivity is that it issued frames.

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

    /// Takes the key out for a `request`, which pushes it to the downstream, or `None` if it is away. One swap, and
    /// nothing read before it: a null result is the key being away.
    pub fn take(&self) -> Option<Box<K>> {
        let key = self.home.swap(ptr::null_mut(), Ordering::Acquire);
        // SAFETY: a non-null pointer in the slot was made by `Box::into_raw` in `with_key`, `home` or
        // `home_replacement`, and the swap took it out, so this call alone owns the allocation now.
        (!key.is_null()).then(|| unsafe { Box::from_raw(key) })
    }

    /// Sends the key home. The slot must be empty: there is one key, and only its holder homes it. (A second key,
    /// a bug, displaces the key found there, which is dropped, in every build, and fails a debug build; nothing is
    /// leaked.)
    pub fn home(&self, key: Box<K>) {
        let previous = self.home.swap(Box::into_raw(key), Ordering::AcqRel);
        debug_assert!(
            previous.is_null(),
            "a key was homed while the slot already held one"
        );
        if !previous.is_null() {
            // SAFETY: a non-null pointer in the slot is a `Box::into_raw` from `with_key`, `home` or
            // `home_replacement`, and the swap took it out, so this call alone owns the allocation now.
            drop(unsafe { Box::from_raw(previous) });
        }
    }

    /// Homes the replacement of a key that was dropped without having run, only if the slot is empty (one
    /// compare-exchange; it is a cold path). If the slot already holds a key (the dropped one was a stray, or a second
    /// key was homed over it), the replacement is not wanted: `disarm` makes it inert, so that its own drop does not
    /// make another replacement, and it is dropped. A chain of replacements therefore ends at once.
    pub fn home_replacement(&self, key: Box<K>, disarm: impl FnOnce(&mut K)) {
        let raw = Box::into_raw(key);
        if self
            .home
            .compare_exchange(ptr::null_mut(), raw, Ordering::AcqRel, Ordering::Relaxed)
            .is_err()
        {
            // SAFETY: the exchange failed, so `raw` was not stored and this call still owns the allocation.
            let mut key = unsafe { Box::from_raw(raw) };
            disarm(&mut key);
            drop(key);
        }
    }
}

/// The back-off of a key that polls blind: one that asks a device or the operating system whether something ended, with
/// nothing to say when it does.
///
/// A key pushed at every visit while nothing ends would keep a worker busy for ever. After a poll that found nothing the
/// source pushes the key again only after a **time** window: [`BASE_WINDOW`](Self::BASE_WINDOW), twice as long after each
/// further empty poll, up to [`MAX_WINDOW`](Self::MAX_WINDOW). Between pushes a visit pushes nothing, so the request
/// issued no frame and the handle reads unproductive (a request that issues no frame shows its handle empty, as a pull
/// that finds none does), and the worker parks. The window is time, not a count of visits, because a visit's interval varies by
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
            // SAFETY: the slot owns what it holds (a `Box::into_raw` allocation), and `&mut self` means no other thread
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
            .field("home", &!self.home.load(Ordering::Relaxed).is_null())
            .finish()
    }
}

// SAFETY: the slot hands a `Box<K>` from the thread that homes it to the thread that takes it, which
// `K: Send` allows; no shared reference to a `K` ever exists, so `K` needs no `Sync`.
unsafe impl<K: Send> Sync for KeySlot<K> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frames::{Frame, FrameHeader};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};
    use std::thread;

    /// A key that counts its drops and what it was told.
    struct Key {
        drops: Arc<AtomicUsize>,
        /// Plain state of whoever holds the key.
        turns: u64,
    }

    impl Key {
        fn boxed(drops: &Arc<AtomicUsize>) -> Box<Self> {
            Box::new(Self {
                drops: drops.clone(),
                turns: 0,
            })
        }
    }

    impl Drop for Key {
        fn drop(&mut self) {
            self.drops.fetch_add(1, SeqCst);
        }
    }

    #[test]
    fn the_key_is_taken_once_and_comes_back_when_homed() {
        let drops = Arc::new(AtomicUsize::new(0));
        let slot = KeySlot::with_key(Key::boxed(&drops));
        let key = slot.take().expect("at home");
        assert!(slot.take().is_none(), "away: nothing to take");
        slot.home(key);
        assert!(slot.take().is_some(), "home again");
        assert_eq!(
            drops.load(SeqCst),
            1,
            "only the take that was dropped here freed it"
        );
    }

    #[test]
    fn an_empty_slot_has_no_key_and_takes_one_home() {
        let drops = Arc::new(AtomicUsize::new(0));
        let slot = KeySlot::<Key>::empty();
        assert!(slot.take().is_none());
        slot.home(Key::boxed(&drops));
        assert!(slot.take().is_some(), "the key was dropped by the take");
        assert_eq!(drops.load(SeqCst), 1);
    }

    #[test]
    fn the_slot_frees_the_key_it_holds_and_not_one_that_is_away() {
        let drops = Arc::new(AtomicUsize::new(0));
        let slot = KeySlot::with_key(Key::boxed(&drops));
        drop(slot);
        assert_eq!(drops.load(SeqCst), 1, "a key at home goes with the slot");
        let slot = KeySlot::with_key(Key::boxed(&drops));
        let away = slot.take().expect("at home");
        drop(slot);
        assert_eq!(drops.load(SeqCst), 1, "a key that is away is its holder's");
        drop(away);
        assert_eq!(drops.load(SeqCst), 2);
    }

    #[test]
    fn a_replacement_homes_only_into_an_empty_slot_and_a_chain_of_them_ends() {
        let drops = Arc::new(AtomicUsize::new(0));
        let slot = KeySlot::<Key>::empty();
        slot.home_replacement(Key::boxed(&drops), |_| panic!("homed: not disarmed"));
        assert_eq!(drops.load(SeqCst), 0, "the replacement is at home");
        // The slot holds a key: the replacement is disarmed and dropped, the key at home stays.
        let disarmed = Arc::new(AtomicUsize::new(0));
        slot.home_replacement(Key::boxed(&drops), |k| {
            disarmed.fetch_add(1, SeqCst);
            k.turns = 99;
        });
        assert_eq!(disarmed.load(SeqCst), 1);
        assert_eq!(drops.load(SeqCst), 1, "the refused replacement was dropped");
        assert_eq!(slot.take().expect("the first stays").turns, 0);
    }

    /// A key whose drop replaces it, as a source's keys do, and whose replacement is disarmed when it is refused.
    struct Replacing {
        header: FrameHeader,
        slot: std::sync::Weak<KeySlot<Replacing>>,
        made: Arc<AtomicUsize>,
    }

    impl Replacing {
        fn boxed(slot: &Arc<KeySlot<Replacing>>, made: &Arc<AtomicUsize>) -> Box<Self> {
            Box::new(Self {
                header: FrameHeader::new(1),
                slot: Arc::downgrade(slot),
                made: made.clone(),
            })
        }
    }

    impl Frame for Replacing {
        fn header(&self) -> &FrameHeader {
            &self.header
        }

        fn header_mut(&mut self) -> &mut FrameHeader {
            &mut self.header
        }
    }

    impl Drop for Replacing {
        fn drop(&mut self) {
            if let Some(slot) = self.slot.upgrade() {
                self.made.fetch_add(1, SeqCst);
                let replacement = Box::new(Replacing {
                    header: FrameHeader::new(1),
                    slot: self.slot.clone(),
                    made: self.made.clone(),
                });
                slot.home_replacement(replacement, |k| k.slot = std::sync::Weak::new());
            }
        }
    }

    #[test]
    fn a_stray_key_that_is_dropped_does_not_start_a_chain_of_replacements() {
        let made = Arc::new(AtomicUsize::new(0));
        let slot = Arc::new(KeySlot::<Replacing>::empty());
        slot.home(Replacing::boxed(&slot, &made));
        // A second key that is not in the slot (the bug the debug assertion of `home` catches) is dropped: it makes one
        // replacement, which the full slot refuses, and the refused one makes none: it ends there, not at a stack overflow.
        drop(Replacing::boxed(&slot, &made));
        assert_eq!(made.load(SeqCst), 1, "one replacement, refused, inert");
        // (A key taken out of the slot by the test is made inert before it is dropped, or its drop would replace it.)
        let take_inert = || {
            let mut key = slot.take().expect("a key at home");
            key.slot = std::sync::Weak::new();
        };
        take_inert();
        // Dropped with the slot empty: its replacement goes home.
        drop(Replacing::boxed(&slot, &made));
        assert_eq!(made.load(SeqCst), 2);
        take_inert();
        assert!(slot.take().is_none(), "and no chain left a third");
    }

    #[test]
    fn eight_threads_never_hold_the_key_at_once() {
        const ROUNDS: usize = if cfg!(miri) { 20 } else { 20_000 };
        // The key holds a counter that only its holder touches, as a source's state would be.
        let drops = Arc::new(AtomicUsize::new(0));
        let slot = Arc::new(KeySlot::with_key(Key::boxed(&drops)));
        let threads: Vec<_> = (0..8)
            .map(|_| {
                let slot = slot.clone();
                thread::spawn(move || {
                    let mut held = 0_u64;
                    for _ in 0..ROUNDS {
                        if let Some(mut key) = slot.take() {
                            // A plain read-modify-write: a second holder would lose updates.
                            let v = key.turns;
                            key.turns = v + 1;
                            held += 1;
                            slot.home(key);
                        }
                    }
                    held
                })
            })
            .collect();
        let total: u64 = threads.into_iter().map(|t| t.join().unwrap()).sum();
        assert_eq!(slot.take().expect("home at the end").turns, total);
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
        let slot = KeySlot::with_key(Key::boxed(&drops));
        slot.home(Key::boxed(&drops));
        assert_eq!(drops.load(SeqCst), 1, "the first was dropped");
        drop(slot);
        assert_eq!(drops.load(SeqCst), 2);
    }
}
