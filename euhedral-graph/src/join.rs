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
//! A frame's path hits no full barrier, so an arrival is one read-modify-write and nothing else:
//!
//! - `pending` is the round's count. Each arrival is one `fetch_sub(1, AcqRel)`; the arrival that reads 1 is the
//!   last, and it sees, through the release sequence the chain of arrivals forms, every earlier arrival's slot
//!   write (the argument of a reference count's last release).
//! - A part writes its slot plainly before that read-modify-write and the last arrival reads the slots plainly
//!   after it. Arming stores the count with release ordering before any part is published, so a part's arrival
//!   happens after the arming it belongs to.
//! - **The contract** (Java's, which has no check either): a round is armed before any of its parts is
//!   published, and each part of it ends exactly once, by [`Join::arrive`] or [`Join::abandon`]. A debug build
//!   checks it (a part that ends twice, and one outside the round, are refused); a release build checks only
//!   that the part is inside the round, since the check that a part ends once is a second read-modify-write on
//!   the frame's path. A part that ends twice in a release build is a data race on its slot.
//!
//! # Arming
//!
//! [`Join::reset`] arms the next round and refuses while the previous one is armed or its continuation still
//! runs: it takes the right to arm with one compare-and-set. An owner that knows its cell is quiet (a pooled
//! unit it has just taken, a round it knows ended) arms with [`Join::arm_owned`], which is Java's `expect`: a
//! plain release store of the count, no check, no read-modify-write. It is the way to re-arm a cell whose
//! previous round was cut short, as a unit whose frames were killed: the parts that never arrived are not
//! waited for, and the values that did arrive are dropped. [`Join::discard_round`] is the safe form for an owner
//! with exclusive access (`&mut`), which proves that no part is outstanding; Java implies the unconditional form
//! (its `expect` is a plain store and a killed frame simply never arrives), not an obligation on the kill path to
//! `abandon` every part it cut off. [`Join::is_idle`] says whether a round is armed.
//!
//! # The continuation and the next round
//!
//! The continuation runs with a [`Round`], which lends the slots. Whatever it does not take, the round
//! drops when it ends, so a slot is empty whenever a part can next write it. The round ends in one of three
//! ways, each consuming it so the values cannot be read once the next round can write them:
//! - [`Round::rearm`] starts the next round; the continuation may deliver that round's work after it;
//! - [`Round::end`] (or dropping the round) leaves the join idle, ready for [`Join::reset`];
//! - a panic in the continuation drops the round, which is `end`.
//!
//! [`Join::reset`] refuses while a round is armed or its round is still held. A continuation that hands
//! the unit holding the join to another worker ends the round **first**:
//!
//! ```
//! # use euhedral_graph::Join;
//! # fn put_back(_: u32) {}
//! let join = Join::<u32>::with_capacity(2, |mut round| {
//!     let sum = round.take(0).unwrap_or(0) + round.take(1).unwrap_or(0);
//!     round.end(); // before the unit can be picked up and re-armed
//!     put_back(sum);
//! });
//! join.reset(2);
//! join.arrive(0, 1);
//! join.arrive(1, 2);
//! join.reset(2); // armed again
//! ```

#![allow(
    unsafe_code,
    reason = "parts own their slots until the count publishes them"
)]

use euhedral_data_structures::sync::HotScope;
use std::any::Any;
use std::cell::Cell;
use std::fmt;
use std::marker::PhantomData;
use std::mem;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::sync::Arc;

#[cfg(debug_assertions)]
use crate::sync::AtomicBool;
use crate::sync::{AtomicUsize, Ordering, UnsafeCell};

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
    /// The part has ended, or is not part of the round: the contract's check, in debug builds only.
    #[cfg(debug_assertions)]
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

/// The values of a finished round, lent to the continuation. The loan ends, and the join can be armed
/// again, when the round is dropped, [`Round::end`]ed or [`Round::rearm`]ed. A continuation that hands
/// the join's unit to anyone else (a pool, another worker) ends the round first: until then the join
/// refuses [`Join::reset`].
///
/// `&Round` lends `&T`, so a round is `Sync` only when `T` is, and `Send` when `T` is `Send`:
///
/// ```compile_fail
/// use std::cell::Cell;
/// fn is_sync<T: Sync>() {}
/// is_sync::<euhedral_graph::Round<'static, Cell<u8>>>();
/// ```
pub struct Round<'a, T> {
    join: &'a Join<T>,
    parts: usize,
    /// Makes the round `!Sync` for every `T`; the impl below restores `Sync` for `T: Sync`.
    _not_sync: PhantomData<Cell<()>>,
}

// SAFETY: `&Round` gives out `&T` only (`get`, `values`); `take` needs `&mut`. Sharing `&T` between threads
// is exactly `T: Sync`. `Send` follows from the fields: `&Join<T>` is `Send` when `T: Send`.
unsafe impl<T: Send + Sync> Sync for Round<'_, T> {}

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
    /// continuation, before it delivers the next round's work. Values the continuation did not take are
    /// dropped first.
    ///
    /// # Panics
    /// When `parts` is 0 or above the capacity. If a value's drop panics, the next round is armed and
    /// the panic is raised after.
    pub fn rearm(self, parts: usize) {
        let join = self.join;
        let round = self.parts;
        join.check_parts(parts);
        // The loan ends here; dropping `self` would mark the join idle.
        mem::forget(self);
        let panic = join.clear(round);
        join.arm(parts);
        if let Some(panic) = panic {
            resume_unwind(panic);
        }
    }

    /// Ends the round: drops the values the continuation did not take and leaves the join idle, ready
    /// for [`Join::reset`]. Dropping the round does the same; call this where the continuation hands the
    /// unit that holds the join to anyone else, before it does, because until the round ends the join
    /// refuses to be armed.
    pub fn end(self) {
        drop(self);
    }
}

impl<T> Drop for Round<'_, T> {
    /// A continuation that does not re-arm leaves the join idle, ready for [`Join::reset`]. This also runs
    /// when the continuation panics.
    fn drop(&mut self) {
        let panic = self.join.clear(self.parts);
        self.join.pending.store(IDLE, Ordering::Release);
        if let Some(panic) = panic
            && !std::thread::panicking()
        {
            resume_unwind(panic);
        }
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
                    #[cfg(debug_assertions)]
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
            .compare_exchange(IDLE, RESETTING, Ordering::AcqRel, Ordering::Acquire)
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

    /// Drops the values of a finished round of `parts` parts that its continuation left in the slots, so
    /// a slot is empty whenever a part can next write it: a smaller later round keeps nothing alive, and
    /// an arrival never runs a drop before its read-modify-write. A value whose drop panics does not stop
    /// the others; the first panic is returned. The caller owns the round.
    fn clear(&self, parts: usize) -> Option<Box<dyn Any + Send>> {
        let mut panic = None;
        for slot in &self.slots[..parts] {
            // SAFETY: the caller owns the finished round: every part ended, and none can write before the
            // join is armed.
            let value = slot.value.with_mut(|cell| unsafe { (*cell).take() });
            if let Err(payload) = catch_unwind(AssertUnwindSafe(|| drop(value))) {
                panic.get_or_insert(payload);
            }
        }
        panic
    }

    /// Publishes a round of `parts` parts. The caller owns the join's state: `pending` is `RESETTING`, or 0
    /// inside a continuation that gave up its loan.
    fn arm(&self, parts: usize) {
        self.parts.store(parts, Ordering::Relaxed);
        // In debug builds the parts reopen first (those of the round) and close (the rest), so that the count,
        // stored last, publishes them with it.
        #[cfg(debug_assertions)]
        for (index, slot) in self.slots.iter().enumerate() {
            slot.closed.store(index >= parts, Ordering::Relaxed);
        }
        self.pending.store(parts, Ordering::Release);
    }

    /// Whether no round is armed and no continuation runs: the state of a join that was never armed, or whose last
    /// round ended. An acquire load, a hint for an owner deciding how to arm.
    pub fn is_idle(&self) -> bool {
        self.pending.load(Ordering::Acquire) == IDLE
    }

    /// Discards an unfinished round: drops the values that arrived, forgets the parts that never did and leaves the
    /// join idle, ready for [`Join::reset`]. For an owner that has a unit whose frames were killed and dropped (a
    /// concluded picture): exclusive access proves that no part is outstanding, because a part's frame holds the
    /// join by shared reference. A round that completed (or never started) is left idle, as it is.
    ///
    /// # Panics
    /// If a value's drop panics, after the round is discarded.
    pub fn discard_round(&mut self) {
        let parts = self.parts.load(Ordering::Relaxed);
        let panic = self.clear(parts);
        self.pending.store(IDLE, Ordering::Release);
        if let Some(panic) = panic {
            resume_unwind(panic);
        }
    }

    /// Arms a round of `parts` parts (1 to the capacity) without asking whether the previous one ended: Java's
    /// `expect`. One release store of the count, no read-modify-write. Values the previous round left in the
    /// slots are dropped first, and parts of it that never arrived are not waited for.
    ///
    /// For a pooled cell whose owner knows it is quiet, including one whose previous round was cut short (the
    /// frames that held its parts were killed and dropped, so some never ended).
    ///
    /// # Safety
    ///
    /// No part of the previous round may arrive or be abandoned from now on, and no continuation of the join may
    /// be running or about to run: the caller is the cell's only user until its parts are published. A part of
    /// the previous round that arrives afterwards writes a slot of the new round.
    ///
    /// # Panics
    /// When `parts` is 0 or above the capacity, or on a join made by [`Join::new`], which has one round. If a
    /// value's drop panics, the round is armed and the panic is raised after.
    pub unsafe fn arm_owned(&self, parts: usize) {
        self.check_parts(parts);
        assert!(
            matches!(self.then, Then::Each(_)),
            "a join made for one round cannot be re-armed"
        );
        let previous = self.parts.load(Ordering::Relaxed);
        let panic = self.clear(previous);
        self.arm(parts);
        if let Some(panic) = panic {
            resume_unwind(panic);
        }
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
        let _hot = HotScope::enter();
        let Some(slot) = self.slots.get(part) else {
            panic!("part {part} of a join of capacity {}", self.slots.len());
        };
        // The part must belong to the round: one plain load, so a part outside it never writes a slot.
        let parts = self.parts.load(Ordering::Relaxed);
        if part >= parts {
            self.refuse(part);
        }
        // The contract's second half, in debug builds only: a part ends once.
        #[cfg(debug_assertions)]
        if slot.closed.swap(true, Ordering::Acquire) {
            self.refuse(part);
        }
        // SAFETY: by the contract this arrival is the part's only one, so it has the slot alone. The round's last
        // arrival reads it only after the read-modify-write below, and arming happens after that read.
        slot.value.with_mut(|cell| unsafe { *cell = value });
        // One read-modify-write, and nothing else on this path: the chain of arrivals carries every earlier slot
        // write to the arrival that sees 1.
        if self.pending.fetch_sub(1, Ordering::AcqRel) == 1 {
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
        let mut round = Round {
            join: self,
            parts,
            _not_sync: PhantomData,
        };
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

    /// The check that a part ends once is made in debug builds only: a release build trusts the contract.
    #[cfg(debug_assertions)]
    #[test]
    fn a_part_ending_twice_is_refused_in_a_debug_build() {
        let (join, runs) = recording(2);
        join.arrive(0, "a");
        assert!(catch_unwind(AssertUnwindSafe(|| join.arrive(0, "again"))).is_err());
        join.arrive(1, "b");
        assert_eq!(
            *runs.lock().unwrap(),
            [vec![Some("a"), Some("b")]],
            "a refused part changes nothing"
        );
    }

    #[test]
    fn a_part_out_of_range_is_refused_in_every_build() {
        let (join, runs) = recording(2);
        join.arrive(0, "a");
        assert!(catch_unwind(AssertUnwindSafe(|| join.abandon(2))).is_err());
        assert!(catch_unwind(AssertUnwindSafe(|| join.arrive(usize::MAX, "x"))).is_err());
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

    #[cfg(debug_assertions)]
    #[test]
    fn an_arrival_after_the_round_completed_is_refused_in_a_debug_build() {
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
    fn a_join_is_idle_until_armed_and_again_once_its_round_ended() {
        let join = Join::<u8>::with_capacity(2, |_| {});
        assert!(join.is_idle());
        join.reset(2);
        assert!(!join.is_idle());
        join.arrive(0, 1);
        assert!(!join.is_idle());
        join.abandon(1);
        assert!(join.is_idle(), "the continuation returned");
    }

    #[test]
    fn an_owner_arms_a_quiet_cell_without_the_check_and_a_cut_short_round_is_dropped() {
        let drops = Arc::new(AtomicUsize::new(0));
        struct Counted(Arc<AtomicUsize>);
        impl Drop for Counted {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::AcqRel);
            }
        }
        let ran = Arc::new(AtomicUsize::new(0));
        let counted = ran.clone();
        let join = Join::<Counted>::with_capacity(3, move |_| {
            counted.fetch_add(1, Ordering::AcqRel);
        });
        // A round that was cut short: two of three parts arrived, and the third's frame was killed and dropped.
        join.reset(3);
        join.arrive(0, Counted(drops.clone()));
        join.arrive(1, Counted(drops.clone()));
        assert!(!join.is_idle());
        assert!(catch_unwind(AssertUnwindSafe(|| join.reset(2))).is_err());
        // SAFETY: the owner knows the third part's frame is gone and no continuation runs.
        unsafe { join.arm_owned(2) };
        assert_eq!(
            drops.load(Ordering::Acquire),
            2,
            "the values that arrived were dropped"
        );
        join.arrive(1, Counted(drops.clone()));
        join.arrive(0, Counted(drops.clone()));
        assert_eq!(ran.load(Ordering::Acquire), 1, "the new round ran once");
        // SAFETY: the round ended and its continuation returned.
        unsafe { join.arm_owned(1) };
        join.arrive(0, Counted(drops.clone()));
        assert_eq!(ran.load(Ordering::Acquire), 2);
    }

    #[test]
    fn discard_round_drops_what_arrived_and_leaves_the_join_ready_for_reset() {
        let drops = Arc::new(AtomicUsize::new(0));
        struct Counted(Arc<AtomicUsize>);
        impl Drop for Counted {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::AcqRel);
            }
        }
        let mut join = Join::<Counted>::with_capacity(3, |_| panic!("the round never completes"));
        join.reset(3);
        join.arrive(0, Counted(drops.clone()));
        join.arrive(2, Counted(drops.clone()));
        assert!(catch_unwind(AssertUnwindSafe(|| join.reset(3))).is_err());
        join.discard_round();
        assert!(join.is_idle());
        assert_eq!(drops.load(Ordering::Acquire), 2);
        // A round that was never armed, and one already discarded, are left as they are.
        join.discard_round();
        join.reset(1);
        assert!(!join.is_idle());
        let mut fresh = Join::<u8>::with_capacity(1, |_| {});
        fresh.discard_round();
        assert!(fresh.is_idle());
    }

    #[test]
    fn arm_owned_refuses_a_bad_part_count_and_a_one_round_join() {
        let join = Join::<u8>::with_capacity(2, |_| {});
        // SAFETY: nothing is armed.
        assert!(catch_unwind(AssertUnwindSafe(|| unsafe { join.arm_owned(0) })).is_err());
        // SAFETY: as above.
        assert!(catch_unwind(AssertUnwindSafe(|| unsafe { join.arm_owned(3) })).is_err());
        let once = Join::<u8>::new(1, |_| {});
        // SAFETY: the call refuses before it touches anything.
        assert!(catch_unwind(AssertUnwindSafe(|| unsafe { once.arm_owned(1) })).is_err());
    }

    #[test]
    fn a_join_made_for_one_round_cannot_be_re_armed() {
        let join = Join::new(1, |_| {});
        join.arrive(0, ());
        assert!(catch_unwind(AssertUnwindSafe(|| join.reset(1))).is_err());
    }

    #[test]
    fn values_a_round_leaves_untaken_are_dropped_when_the_round_ends() {
        let live = Arc::new(());
        let join = Join::<Arc<()>>::with_capacity(2, |_| {});
        join.reset(2);
        join.arrive(0, live.clone());
        join.arrive(1, live.clone());
        // The continuation took nothing: the round's end drops what it left, a smaller later round
        // does not keep it alive.
        assert_eq!(Arc::strong_count(&live), 1);
        join.reset(1);
        join.abandon(0);
        assert_eq!(Arc::strong_count(&live), 1);
    }

    #[test]
    fn values_of_a_round_that_never_completes_are_dropped_with_the_join() {
        let live = Arc::new(());
        let join = Join::<Arc<()>>::with_capacity(2, |_| {});
        join.reset(2);
        join.arrive(0, live.clone());
        assert_eq!(Arc::strong_count(&live), 2);
        drop(join);
        assert_eq!(Arc::strong_count(&live), 1);
    }

    /// A value whose drop panics while the round is being released.
    struct PanicsOnDrop(Arc<AtomicUsize>);

    impl Drop for PanicsOnDrop {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::AcqRel);
            if !std::thread::panicking() {
                panic!("drop");
            }
        }
    }

    #[test]
    fn a_value_whose_drop_panics_does_not_strand_the_join() {
        let dropped = Arc::new(AtomicUsize::new(0));
        let join = Join::<PanicsOnDrop>::with_capacity(2, |_| {});
        join.reset(2);
        join.arrive(0, PanicsOnDrop(dropped.clone()));
        // The last arrival's round release drops both values; the first panic comes out of `arrive`,
        // after every value was dropped and the join was released.
        assert!(
            catch_unwind(AssertUnwindSafe(
                || join.arrive(1, PanicsOnDrop(dropped.clone()))
            ))
            .is_err()
        );
        assert_eq!(dropped.load(Ordering::Acquire), 2);
        join.reset(1);
        join.abandon(0);
    }

    /// The text of a panic.
    #[cfg(debug_assertions)]
    fn message(payload: Box<dyn std::any::Any + Send>) -> String {
        match payload.downcast::<String>() {
            Ok(text) => *text,
            Err(payload) => payload
                .downcast::<&str>()
                .map(|t| t.to_string())
                .unwrap_or_default(),
        }
    }

    #[cfg(debug_assertions)]
    #[test]
    fn a_duplicate_arrival_around_a_re_arm_is_refused_and_the_count_survives_in_a_debug_build() {
        let slot: Arc<OnceLock<Arc<Join<u8>>>> = Arc::new(OnceLock::new());
        let weak = Arc::downgrade(&slot);
        let refusals = Arc::new(Mutex::new(Vec::new()));
        let record = refusals.clone();
        let rounds = Arc::new(AtomicUsize::new(0));
        let counted = rounds.clone();
        let join = Arc::new(Join::<u8>::with_capacity(2, move |round| {
            let join = weak.upgrade().unwrap().get().unwrap().clone();
            // Before the re-arm the round's parts are all closed: a late duplicate is refused ...
            let late = catch_unwind(AssertUnwindSafe(|| join.arrive(0, 9)));
            record.lock().unwrap().push(message(late.unwrap_err()));
            if counted.fetch_add(1, Ordering::AcqRel) == 0 {
                round.rearm(2);
            }
        }));
        slot.set(join.clone()).unwrap();
        join.reset(2);
        join.arrive(0, 1);
        join.arrive(1, 2);
        // ... and the re-armed round still needs both of its parts.
        join.arrive(0, 3);
        assert_eq!(rounds.load(Ordering::Acquire), 1);
        let duplicate = catch_unwind(AssertUnwindSafe(|| join.arrive(0, 4)));
        assert!(message(duplicate.unwrap_err()).contains("ended twice"));
        join.arrive(1, 5);
        assert_eq!(rounds.load(Ordering::Acquire), 2);
        assert!(
            refusals
                .lock()
                .unwrap()
                .iter()
                .all(|m| m.contains("ended twice"))
        );
    }

    #[test]
    #[allow(clippy::disallowed_methods)] // The test's own threads stand for the workers that take a pooled unit.
    fn a_unit_handed_back_inside_the_continuation_is_re_armed_at_once_by_another_thread() {
        const ROUNDS: usize = if cfg!(miri) { 6 } else { 60 };
        type Handles = Arc<Mutex<Vec<std::thread::JoinHandle<()>>>>;
        let handles: Handles = Arc::new(Mutex::new(Vec::new()));
        let runs = Arc::new(AtomicUsize::new(0));
        let slot: Arc<OnceLock<Arc<Join<usize>>>> = Arc::new(OnceLock::new());
        let (weak, spawned, counted) = (Arc::downgrade(&slot), handles.clone(), runs.clone());
        let join = Arc::new(Join::<usize>::with_capacity(2, move |mut round| {
            let r = counted.fetch_add(1, Ordering::AcqRel);
            assert_eq!((round.take(0), round.take(1)), (Some(r), Some(r + 1)));
            // The unit goes back to the pool: the round ends before another worker can take it.
            round.end();
            if r + 1 < ROUNDS {
                let unit = weak.upgrade().unwrap().get().unwrap().clone();
                let next = std::thread::spawn(move || {
                    unit.reset(2); // must not find the previous round still held
                    unit.arrive(0, r + 1);
                    unit.arrive(1, r + 2);
                });
                spawned.lock().unwrap().push(next);
            }
        }));
        slot.set(join.clone()).unwrap();
        join.reset(2);
        join.arrive(0, 0);
        join.arrive(1, 1);
        loop {
            let next = handles.lock().unwrap().pop();
            let Some(next) = next else { break };
            next.join().unwrap();
        }
        assert_eq!(runs.load(Ordering::Acquire), ROUNDS);
    }

    /// `Round` lends `&T` across threads only when `T: Sync`.
    #[test]
    fn a_round_is_sync_exactly_when_its_values_are() {
        fn sync<T: Sync>() {}
        fn send<T: Send>() {}
        sync::<Round<'static, u8>>();
        send::<Round<'static, std::cell::Cell<u8>>>();
    }
}

/// Model checks of the arrival protocol: `RUSTFLAGS="--cfg loom" cargo test -p euhedral-graph --release --lib
/// join::loom`. Each explores every interleaving of the threads' atomic operations the memory model allows,
/// and `loom`'s cell reports a slot access that is not ordered after the write it reads. The models read
/// slots with `take`: a reference from `get` outlives the cell's closure, so a read through it is not
/// checked.
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
            let join = Arc::new(Join::<usize>::with_capacity(2, move |mut round| {
                let r = counted.fetch_add(1, Ordering::SeqCst);
                // Round 0 abandons part 1; round 1 abandons part 0. Neither sees the other's value.
                let expected = if r == 0 {
                    [Some(value(0, 0)), None]
                } else {
                    [None, Some(value(1, 1))]
                };
                assert_eq!([round.take(0), round.take(1)], expected);
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
    fn loom_a_unit_ended_and_handed_over_is_re_armed_by_another_thread() {
        ::loom::model(|| {
            let handed = Arc::new(AtomicBool::new(false));
            let flag = handed.clone();
            let runs = Arc::new(AtomicUsize::new(0));
            let counted = runs.clone();
            let join = Arc::new(Join::<usize>::with_capacity(2, move |mut round| {
                if counted.fetch_add(1, Ordering::SeqCst) == 0 {
                    assert_eq!(round.take(0), Some(value(0, 0)));
                    assert_eq!(round.take(1), Some(value(0, 1)));
                    // The unit goes to the pool: the round ends first.
                    round.end();
                    flag.store(true, Ordering::Release);
                } else {
                    // The taker's round of one abandoned part.
                    assert_eq!((round.len(), round.take(0)), (1, None));
                }
            }));
            join.reset(2);
            let first: Vec<_> = (0..2)
                .map(|part| {
                    let join = join.clone();
                    thread::spawn(move || join.arrive(part, value(0, part)))
                })
                .collect();
            let taker = {
                let join = join.clone();
                thread::spawn(move || {
                    while !handed.load(Ordering::Acquire) {
                        thread::yield_now();
                    }
                    // Another worker takes the unit and arms it at once.
                    join.reset(1);
                    join.abandon(0);
                })
            };
            for thread in first {
                thread.join().unwrap();
            }
            taker.join().unwrap();
            assert_eq!(runs.load(Ordering::SeqCst), 2);
        });
    }

    // An owner re-arms a round that was cut short (one part arrived, the other's frame was killed and dropped) with
    // `arm_owned`, a plain release store, after the arrival thread has been joined: the new round's two arrivals
    // run the continuation once, and no read of a slot is unordered with the write that fills it.
    #[test]
    fn loom_an_owner_re_arms_a_cut_short_round_and_the_next_one_runs_once() {
        ::loom::model(|| {
            let ran = Arc::new(AtomicUsize::new(0));
            let counted = ran.clone();
            let join = Arc::new(Join::<usize>::with_capacity(2, move |round| {
                assert_eq!(round.len(), 2);
                counted.fetch_add(1, Ordering::SeqCst);
            }));
            join.reset(2);
            let first = {
                let join = join.clone();
                thread::spawn(move || join.arrive(0, 1))
            };
            first.join().unwrap();
            // SAFETY: the only arrival of the cut-short round has been joined; no continuation runs.
            unsafe { join.arm_owned(2) };
            let threads: Vec<_> = (0..2)
                .map(|part| {
                    let join = join.clone();
                    thread::spawn(move || join.arrive(part, part + 10))
                })
                .collect();
            for thread in threads {
                thread.join().unwrap();
            }
            assert_eq!(ran.load(Ordering::SeqCst), 1);
        });
    }
}
