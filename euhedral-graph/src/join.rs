//! Run-time fan-in: the last arrival runs the continuation.
//!
//! A [`Join`] is a counted cell. A round is armed with the number of parts it waits for ([`Join::reset`]),
//! every part ends once with a value or by abandoning, and the arrival that brings the count to zero runs
//! the continuation on the values. It waits for no one, takes no lock, and a cell made with
//! [`Join::with_capacity`] is armed again for the next round without allocating, so it can live inside a
//! pooled object.
//!
//! # Memory ordering
//!
//! - `pending` is the round's count. [`Join::reset`] and [`Round::rearm`] store it with `SeqCst`; each
//!   arrival is one `fetch_sub(1, SeqCst)`, a single read-modify-write, so the arrival that reads 1 is the
//!   last and sees, through the chain of read-modify-writes, every earlier arrival's slot write.
//! - A part writes its slot plainly before that read-modify-write and the last arrival reads the slots plainly
//!   after it.
//! - `closed[part]` says whether a part may still end. An arrival takes it with one `swap(true, Acquire)`, so
//!   exactly one of two racing arrivals of a part proceeds; arming stores it with `Release`, which is what
//!   makes a slot's earlier readers happen before the new round's write to it.
//!
//! # The continuation and the next round
//!
//! The continuation runs with a [`Round`], which lends the slots. [`Round::rearm`] consumes it, so the
//! values cannot be read once the next round can write them, and the continuation may deliver the next
//! round's work after calling it. Dropping the round without re-arming leaves the join idle, ready for
//! [`Join::reset`]. [`Join::reset`] refuses while a round is armed or its continuation is running.

#![allow(
    unsafe_code,
    reason = "parts own their slots until the count publishes them"
)]

use std::fmt;
use std::mem;
use std::sync::Arc;

use crate::sync::{AtomicBool, AtomicUsize, Ordering, UnsafeCell};

/// No round is armed and no continuation is running.
const IDLE: usize = usize::MAX;
/// [`Join::reset`] is arming a round.
const RESETTING: usize = usize::MAX - 1;

/// Waits for `parts` branches without waiting: each branch arrives (or is abandoned) from any
/// thread, and the last of them runs the continuation, once per round, with every part's value
/// (`None` for an abandoned branch).
pub struct Join<T> {
    /// Parts of the round that have not ended; [`IDLE`] or [`RESETTING`] between rounds, 0 while the
    /// continuation runs.
    pending: AtomicUsize,
    /// This round's part count. Written before the round is published by `pending`.
    parts: AtomicUsize,
    slots: Box<[Part<T>]>,
    then: Then<T>,
}

struct Part<T> {
    /// Written by the part's one arrival, read by the last arrival.
    value: UnsafeCell<Option<T>>,
    /// The part has ended, or is not part of the round.
    closed: AtomicBool,
}

type EachRound<T> = Box<dyn Fn(Round<'_, T>) + Send + Sync>;
type OneRound<T> = Box<dyn FnOnce(Vec<Option<T>>) + Send>;

enum Then<T> {
    /// Runs every round.
    Each(EachRound<T>),
    /// Runs on the first round only.
    Once(UnsafeCell<Option<OneRound<T>>>),
}

// SAFETY: a slot is written by the one arrival that closed its part and read by the one arrival that
// brought the count to zero, after every write (see the module's ordering notes), so a `Join` shares
// values between threads exactly as a channel does: `T: Send`. `Then::Once`'s cell is taken only by that
// last arrival, once.
unsafe impl<T: Send> Sync for Join<T> {}

impl<T> fmt::Debug for Join<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Join")
            .field("capacity", &self.slots.len())
            .finish_non_exhaustive()
    }
}

/// The values of a finished round, lent to the continuation. [`Round::rearm`] starts the next round and
/// ends the loan.
pub struct Round<'a, T> {
    join: &'a Join<T>,
    parts: usize,
}

impl<T> fmt::Debug for Round<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Round")
            .field("parts", &self.parts)
            .finish_non_exhaustive()
    }
}

impl<T> Round<'_, T> {
    /// The number of parts this round had.
    pub fn len(&self) -> usize {
        self.parts
    }

    /// A round has at least one part.
    pub fn is_empty(&self) -> bool {
        self.parts == 0
    }

    /// Part `part`'s value, `None` when it was abandoned.
    ///
    /// # Panics
    /// When `part` is not a part of this round.
    pub fn get(&self, part: usize) -> Option<&T> {
        assert!(
            part < self.parts,
            "part {part} of a round of {}",
            self.parts
        );
        // SAFETY: every part of the round ended, and its write happened before the arrival that made this
        // `Round`, which no other arrival overlaps (module notes); `&self` forbids a concurrent `take`.
        self.join.slots[part]
            .value
            .with_mut(|slot| unsafe { (*slot).as_ref() })
    }

    /// Takes part `part`'s value, `None` when it was abandoned.
    ///
    /// # Panics
    /// When `part` is not a part of this round.
    pub fn take(&mut self, part: usize) -> Option<T> {
        assert!(
            part < self.parts,
            "part {part} of a round of {}",
            self.parts
        );
        // SAFETY: as `get`; `&mut self` makes this the only access.
        self.join.slots[part]
            .value
            .with_mut(|slot| unsafe { (*slot).take() })
    }

    /// Every part's value in part order.
    pub fn values(&self) -> impl Iterator<Item = Option<&T>> {
        (0..self.parts).map(|part| self.get(part))
    }

    /// Arms the next round of `parts` parts (1 to the capacity) and ends this loan: the next round's
    /// parts may arrive, on any thread, as soon as this returns. The join's usual call from its
    /// continuation, before it delivers the next round's work.
    ///
    /// # Panics
    /// When `parts` is 0 or above the capacity.
    pub fn rearm(self, parts: usize) {
        let join = self.join;
        join.check_parts(parts);
        // The loan ends here; dropping `self` would mark the join idle.
        mem::forget(self);
        join.arm(parts);
    }
}

impl<T> Drop for Round<'_, T> {
    /// A continuation that does not re-arm leaves the join idle, ready for [`Join::reset`]. This also runs
    /// when the continuation panics.
    fn drop(&mut self) {
        self.join.pending.store(IDLE, Ordering::SeqCst);
    }
}

impl<T: Send + 'static> Join<T> {
    /// A join of `parts` branches (at least one) that runs `then` on the last arrival's thread.
    /// `then` must not wait. One round only: `then` is consumed by it.
    pub fn new(parts: usize, then: impl FnOnce(Vec<Option<T>>) + Send + 'static) -> Arc<Self> {
        assert!(parts >= 1, "a join needs at least one part");
        let join = Self::build(parts, Then::Once(UnsafeCell::new(Some(Box::new(then)))));
        join.arm(parts);
        Arc::new(join)
    }

    /// A cell for a pooled unit: room for `capacity` parts and a continuation stored once, run by the
    /// last arrival of every round. It is not armed until [`Join::reset`]. The continuation receives the
    /// round's values and re-arms the join with [`Round::rearm`] if another round follows. It must not
    /// wait: it delivers work.
    pub fn with_capacity(
        capacity: usize,
        then: impl Fn(Round<'_, T>) + Send + Sync + 'static,
    ) -> Self {
        assert!(capacity >= 1, "a join needs at least one part");
        Self::build(capacity, Then::Each(Box::new(then)))
    }
}

impl<T> Join<T> {
    fn build(capacity: usize, then: Then<T>) -> Self {
        assert!(capacity < RESETTING, "a join of {capacity} parts");
        Self {
            pending: AtomicUsize::new(IDLE),
            parts: AtomicUsize::new(0),
            slots: (0..capacity)
                .map(|_| Part {
                    value: UnsafeCell::new(None),
                    closed: AtomicBool::new(true),
                })
                .collect(),
            then,
        }
    }

    /// The most parts a round can have.
    pub fn capacity(&self) -> usize {
        self.slots.len()
    }

    /// Arms a round of `parts` parts (1 to the capacity), before any of them is published. Valid when the
    /// join has never been armed or the previous round's continuation has returned (or re-armed with
    /// [`Round::rearm`], which is the form for inside it).
    ///
    /// # Panics
    /// When the previous round is not complete or its continuation still runs, when `parts` is 0 or above
    /// the capacity, or on a join made by [`Join::new`], which has one round.
    pub fn reset(&self, parts: usize) {
        self.check_parts(parts);
        assert!(
            matches!(self.then, Then::Each(_)),
            "a join made for one round cannot be re-armed"
        );
        // The one read-modify-write that takes the right to arm: it fails unless the previous round is
        // complete and its continuation has returned.
        if self
            .pending
            .compare_exchange(IDLE, RESETTING, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            panic!("the previous round of the join is not complete");
        }
        self.arm(parts);
    }

    fn check_parts(&self, parts: usize) {
        assert!(
            (1..=self.slots.len()).contains(&parts),
            "a round of {parts} parts in a join of capacity {}",
            self.slots.len()
        );
    }

    /// Opens the first `parts` parts and publishes the round. The caller owns the join's state: `pending`
    /// is `RESETTING`, or 0 inside a continuation that gave up its loan.
    fn arm(&self, parts: usize) {
        self.parts.store(parts, Ordering::Relaxed);
        for (index, slot) in self.slots.iter().enumerate() {
            slot.closed.store(index >= parts, Ordering::Release);
        }
        self.pending.store(parts, Ordering::SeqCst);
    }

    /// Branch `part` arrived with `value`. Each part arrives or is abandoned exactly once.
    ///
    /// # Panics
    /// When `part` is not a part of the armed round or has ended.
    pub fn arrive(&self, part: usize, value: T) {
        self.end(part, Some(value));
    }

    /// Branch `part` ended without a value.
    ///
    /// # Panics
    /// As [`Join::arrive`].
    pub fn abandon(&self, part: usize) {
        self.end(part, None);
    }

    fn end(&self, part: usize, value: Option<T>) {
        let Some(slot) = self.slots.get(part) else {
            panic!("part {part} of a join of capacity {}", self.slots.len());
        };
        // One read-modify-write takes the part: of two arrivals of the same part only one passes.
        if slot.closed.swap(true, Ordering::Acquire) {
            self.refuse(part);
        }
        // SAFETY: closing the part gave this arrival the slot alone. The round's last arrival reads it only
        // after the read-modify-write below, and arming (which reopens it) happens after that read.
        slot.value.with_mut(|cell| unsafe { *cell = value });
        // One read-modify-write, so the chain of arrivals carries every earlier slot write to the arrival
        // that sees 1.
        if self.pending.fetch_sub(1, Ordering::SeqCst) == 1 {
            self.finish();
        }
    }

    #[cold]
    #[inline(never)]
    fn refuse(&self, part: usize) -> ! {
        let parts = self.parts.load(Ordering::Relaxed);
        if parts == 0 {
            panic!("an arrival at part {part} of a join that is not armed");
        }
        if part >= parts {
            panic!("part {part} of a join of {parts}");
        }
        panic!("part {part} of a join ended twice");
    }

    /// The last arrival of a round.
    fn finish(&self) {
        let parts = self.parts.load(Ordering::Relaxed);
        let mut round = Round { join: self, parts };
        match &self.then {
            Then::Each(then) => then(round),
            Then::Once(once) => {
                // SAFETY: only the last arrival of the one round reaches this, once (`reset` refuses a
                // `Then::Once` join), so nothing else touches the cell.
                let then = once.with_mut(|cell| unsafe { (*cell).take() });
                let values: Vec<_> = (0..parts).map(|part| round.take(part)).collect();
                // The loan ends; the join is idle before the continuation runs.
                drop(round);
                if let Some(then) = then {
                    then(values);
                }
            }
        }
    }
}

#[cfg(all(test, not(loom)))]
mod tests {
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Mutex, OnceLock};

    use super::*;

    type Runs = Arc<Mutex<Vec<Vec<Option<&'static str>>>>>;

    fn recording(parts: usize) -> (Arc<Join<&'static str>>, Runs) {
        let runs = Arc::new(Mutex::new(Vec::new()));
        let sink = runs.clone();
        (
            Join::new(parts, move |values| sink.lock().unwrap().push(values)),
            runs,
        )
    }

    #[test]
    fn the_last_arrival_runs_the_continuation_once_with_every_part() {
        let (join, runs) = recording(3);
        join.arrive(2, "c");
        join.abandon(0);
        assert!(
            runs.lock().unwrap().is_empty(),
            "the join waits for every part"
        );
        join.arrive(1, "b");
        assert_eq!(*runs.lock().unwrap(), [vec![None, Some("b"), Some("c")]]);
    }

    #[test]
    fn a_join_of_one_runs_on_its_only_arrival() {
        let (join, runs) = recording(1);
        join.arrive(0, "only");
        assert_eq!(*runs.lock().unwrap(), [vec![Some("only")]]);
    }

    #[test]
    fn a_part_ending_twice_or_out_of_range_is_refused() {
        let (join, runs) = recording(2);
        join.arrive(0, "a");
        assert!(catch_unwind(AssertUnwindSafe(|| join.arrive(0, "again"))).is_err());
        assert!(catch_unwind(AssertUnwindSafe(|| join.abandon(2))).is_err());
        join.arrive(1, "b");
        assert_eq!(
            *runs.lock().unwrap(),
            [vec![Some("a"), Some("b")]],
            "a refused part changes nothing"
        );
    }

    #[test]
    #[should_panic(expected = "at least one part")]
    fn a_join_of_none_is_refused() {
        let _ = Join::<()>::new(0, |_| {});
    }

    #[test]
    #[allow(clippy::disallowed_methods)] // The test's own threads arrive, as branches on lattice workers would.
    fn parts_arriving_on_many_threads_run_the_continuation_exactly_once() {
        for _ in 0..if cfg!(miri) { 2 } else { 50 } {
            let runs = Arc::new(AtomicUsize::new(0));
            let counted = runs.clone();
            let join = Join::new(64, move |values: Vec<Option<usize>>| {
                assert!(values.iter().enumerate().all(|(i, v)| *v == Some(i)));
                counted.fetch_add(1, Ordering::AcqRel);
            });
            let threads: Vec<_> = (0..8)
                .map(|t| {
                    let join = join.clone();
                    std::thread::spawn(move || {
                        (t * 8..t * 8 + 8).for_each(|part| join.arrive(part, part))
                    })
                })
                .collect();
            for thread in threads {
                thread.join().unwrap();
            }
            assert_eq!(runs.load(Ordering::Acquire), 1);
        }
    }

    #[test]
    #[allow(clippy::disallowed_methods)] // The test's own threads arrive, as branches on lattice workers would.
    fn a_join_of_64_parts_from_8_threads_runs_once_per_round() {
        const ROUNDS: usize = if cfg!(miri) { 10 } else { 1000 };
        let rounds = Arc::new(AtomicUsize::new(0));
        let counted = rounds.clone();
        let join = Arc::new(Join::<usize>::with_capacity(64, move |mut round| {
            assert_eq!(round.len(), 64);
            assert!((0..64).all(|i| round.take(i) == Some(i)));
            counted.fetch_add(1, Ordering::AcqRel);
        }));
        for round in 0..ROUNDS {
            join.reset(64);
            let threads: Vec<_> = (0..8)
                .map(|t| {
                    let join = join.clone();
                    std::thread::spawn(move || {
                        (t * 8..t * 8 + 8).for_each(|part| join.arrive(part, part))
                    })
                })
                .collect();
            for thread in threads {
                thread.join().unwrap();
            }
            assert_eq!(rounds.load(Ordering::Acquire), round + 1);
        }
    }

    #[test]
    fn reset_from_inside_the_continuation_arms_the_next_round() {
        const ROUNDS: usize = 100;
        let seen = Arc::new(Mutex::new(Vec::new()));
        let sink = seen.clone();
        let join = Arc::new(Join::<usize>::with_capacity(2, move |mut round| {
            let mut seen = sink.lock().unwrap();
            let r = seen.len();
            seen.push((round.take(0), round.take(1)));
            if r + 1 < ROUNDS {
                round.rearm(2);
            }
        }));
        join.reset(2);
        for r in 0..ROUNDS {
            // The re-arm happened inside the continuation, before the arrival below returned.
            join.arrive(1, r * 10 + 1);
            join.arrive(0, r * 10);
        }
        let seen = seen.lock().unwrap();
        assert_eq!(seen.len(), ROUNDS);
        for (r, (a, b)) in seen.iter().enumerate() {
            assert_eq!((*a, *b), (Some(r * 10), Some(r * 10 + 1)), "round {r}");
        }
    }

    #[test]
    fn a_re_armed_round_never_sees_the_previous_rounds_values() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let sink = seen.clone();
        // The continuation leaves its values in place; the next round must overwrite them.
        let join = Join::<u32>::with_capacity(3, move |round| {
            sink.lock()
                .unwrap()
                .push(round.values().map(|v| v.copied()).collect::<Vec<_>>());
        });
        join.reset(3);
        join.arrive(0, 1);
        join.arrive(1, 2);
        join.arrive(2, 3);
        join.reset(3);
        join.abandon(0);
        join.arrive(1, 20);
        join.abandon(2);
        assert_eq!(
            *seen.lock().unwrap(),
            [vec![Some(1), Some(2), Some(3)], vec![None, Some(20), None]]
        );
    }

    #[test]
    fn fewer_parts_than_capacity_use_only_the_first_slots() {
        let lens = Arc::new(Mutex::new(Vec::new()));
        let sink = lens.clone();
        let join =
            Join::<u8>::with_capacity(8, move |round| sink.lock().unwrap().push(round.len()));
        join.reset(3);
        (0..3).for_each(|p| join.arrive(p, 0));
        join.reset(8);
        (0..8).for_each(|p| join.arrive(p, 0));
        join.reset(1);
        join.arrive(0, 0);
        assert_eq!(*lens.lock().unwrap(), [3, 8, 1]);
        assert_eq!(join.capacity(), 8);
        join.reset(3);
        assert!(
            catch_unwind(AssertUnwindSafe(|| join.arrive(3, 0))).is_err(),
            "a part beyond this round's count is refused"
        );
    }

    #[test]
    fn an_abandoned_part_reaches_the_continuation_as_none() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let sink = seen.clone();
        let join = Join::<&'static str>::with_capacity(2, move |round| {
            sink.lock()
                .unwrap()
                .push((round.get(0).copied(), round.get(1).copied()));
        });
        join.reset(2);
        join.abandon(0);
        join.arrive(1, "b");
        assert_eq!(*seen.lock().unwrap(), [(None, Some("b"))]);
    }

    #[test]
    #[should_panic(expected = "round")]
    fn reset_before_the_round_completes_panics() {
        let join = Join::<()>::with_capacity(2, |_| {});
        join.reset(2);
        join.arrive(0, ());
        join.reset(2);
    }

    #[test]
    #[should_panic(expected = "not armed")]
    fn an_arrival_on_a_join_that_is_not_armed_panics() {
        let join = Join::<()>::with_capacity(2, |_| {});
        join.arrive(0, ());
    }

    #[test]
    fn an_arrival_after_the_round_completed_is_refused() {
        let join = Join::<()>::with_capacity(1, |_| {});
        join.reset(1);
        join.arrive(0, ());
        assert!(catch_unwind(AssertUnwindSafe(|| join.arrive(0, ()))).is_err());
        join.reset(1);
        join.arrive(0, ());
    }

    #[test]
    fn reset_through_the_join_while_the_continuation_runs_is_refused() {
        let slot: Arc<OnceLock<Arc<Join<()>>>> = Arc::new(OnceLock::new());
        let weak = Arc::downgrade(&slot);
        let refused = Arc::new(AtomicUsize::new(0));
        let record = refused.clone();
        let join = Arc::new(Join::<()>::with_capacity(1, move |round| {
            let join = weak.upgrade().unwrap().get().unwrap().clone();
            // The round is still lent: the join cannot be armed behind its back.
            let result = catch_unwind(AssertUnwindSafe(|| join.reset(1)));
            record.fetch_add(usize::from(result.is_err()), Ordering::AcqRel);
            round.rearm(1);
        }));
        slot.set(join.clone()).unwrap();
        join.reset(1);
        join.arrive(0, ());
        assert_eq!(refused.load(Ordering::Acquire), 1);
        join.arrive(0, ()); // the re-armed round
        assert_eq!(refused.load(Ordering::Acquire), 2);
    }

    #[test]
    fn a_continuation_that_panics_leaves_the_join_ready_to_re_arm() {
        let join = Join::<()>::with_capacity(1, |_| panic!("continuation"));
        join.reset(1);
        assert!(catch_unwind(AssertUnwindSafe(|| join.arrive(0, ()))).is_err());
        join.reset(1);
    }

    #[test]
    fn a_join_made_for_one_round_cannot_be_re_armed() {
        let join = Join::new(1, |_| {});
        join.arrive(0, ());
        assert!(catch_unwind(AssertUnwindSafe(|| join.reset(1))).is_err());
    }

    #[test]
    fn values_a_round_leaves_behind_are_dropped_with_the_join() {
        let live = Arc::new(());
        let join = Join::<Arc<()>>::with_capacity(2, |_| {});
        join.reset(2);
        join.arrive(0, live.clone());
        join.arrive(1, live.clone());
        assert_eq!(Arc::strong_count(&live), 3);
        drop(join);
        assert_eq!(Arc::strong_count(&live), 1);
    }
}

/// Model checks of the arrival protocol: `RUSTFLAGS="--cfg loom" cargo test -p euhedral-graph --release --lib
/// join::loom`. Each explores every interleaving of the threads' atomic operations the memory model allows,
/// and `loom`'s cell reports a slot access that is not ordered after the write it reads.
#[cfg(all(test, loom))]
mod loom {
    use ::loom::sync::Arc;
    use ::loom::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use ::loom::thread;

    use super::*;

    /// The value part `part` of round `round` sends.
    fn value(round: usize, part: usize) -> usize {
        round * 10 + part
    }

    #[test]
    fn loom_two_parts_run_the_continuation_once_and_rearm() {
        ::loom::model(|| {
            let runs = Arc::new(AtomicUsize::new(0));
            let rearmed = Arc::new(AtomicBool::new(false));
            let (counted, flag) = (runs.clone(), rearmed.clone());
            let join = Arc::new(Join::<usize>::with_capacity(2, move |mut round| {
                let r = counted.fetch_add(1, Ordering::SeqCst);
                assert!(r < 2, "the continuation ran a third time");
                assert_eq!(round.len(), 2);
                assert_eq!(round.take(0), Some(value(r, 0)), "round {r} part 0");
                assert_eq!(round.take(1), Some(value(r, 1)), "round {r} part 1");
                if r == 0 {
                    round.rearm(2);
                    // Work that follows the re-arm, delivered from inside the continuation.
                    flag.store(true, Ordering::Release);
                }
            }));
            join.reset(2);
            let threads: Vec<_> = (0..2)
                .map(|part| {
                    let join = join.clone();
                    let rearmed = rearmed.clone();
                    thread::spawn(move || {
                        join.arrive(part, value(0, part));
                        while !rearmed.load(Ordering::Acquire) {
                            thread::yield_now();
                        }
                        join.arrive(part, value(1, part));
                    })
                })
                .collect();
            for thread in threads {
                thread.join().unwrap();
            }
            assert_eq!(runs.load(Ordering::SeqCst), 2);
        });
    }

    #[test]
    fn loom_an_abandoned_part_and_an_outside_reset_start_a_clean_round() {
        ::loom::model(|| {
            let runs = Arc::new(AtomicUsize::new(0));
            let counted = runs.clone();
            let join = Arc::new(Join::<usize>::with_capacity(2, move |round| {
                let r = counted.fetch_add(1, Ordering::SeqCst);
                // Round 0 abandons part 1; round 1 abandons part 0. Neither sees the other's value.
                let expected = if r == 0 {
                    [Some(value(0, 0)), None]
                } else {
                    [None, Some(value(1, 1))]
                };
                assert_eq!(
                    round.values().map(|v| v.copied()).collect::<Vec<_>>(),
                    expected
                );
            }));
            for round in 0..2 {
                join.reset(2);
                let threads: Vec<_> = (0..2)
                    .map(|part| {
                        let join = join.clone();
                        thread::spawn(move || {
                            if part == round {
                                join.arrive(part, value(round, part));
                            } else {
                                join.abandon(part);
                            }
                        })
                    })
                    .collect();
                for thread in threads {
                    thread.join().unwrap();
                }
                assert_eq!(runs.load(Ordering::SeqCst), round + 1);
            }
        });
    }

    #[test]
    fn loom_one_part_racing_itself_is_taken_once() {
        ::loom::model(|| {
            let join = Arc::new(Join::<usize>::with_capacity(1, |_| {}));
            join.reset(1);
            let refused = Arc::new(AtomicUsize::new(0));
            let threads: Vec<_> = (0..2)
                .map(|_| {
                    let join = join.clone();
                    let refused = refused.clone();
                    thread::spawn(move || {
                        // Two arrivals of one part: one passes, the other is refused (the panic is caught).
                        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            join.arrive(0, 1)
                        }));
                        if result.is_err() {
                            refused.fetch_add(1, Ordering::SeqCst);
                        }
                    })
                })
                .collect();
            for thread in threads {
                thread.join().unwrap();
            }
            assert_eq!(refused.load(Ordering::SeqCst), 1);
        });
    }
}
