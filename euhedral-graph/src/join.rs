//! Run-time fan-in: the last arrival runs the continuation.
//!
//! A [`Join`] is a counted cell. A round is armed with the number of parts it waits for ([`Join::reset`]),
//! every part ends once with a value or by abandoning, and the arrival that brings the count to zero runs
//! the continuation on the values. It waits for no one, takes no lock, and a cell made with
//! [`Join::with_capacity`] is armed again for the next round without allocating, so it can live inside a
//! pooled object.
//!
//! # Parts are tokens
//!
//! Arming a round hands out one [`Part`] per branch ([`Join::reset`], [`Round::rearm`] and [`Join::once`] return them as
//! a [`Parts`] iterator). A part is moved to the branch it stands for and consumed by [`Part::arrive`] or
//! [`Part::abandon`]; it is neither `Copy` nor `Clone`, and dropping it abandons, so a branch that is killed and
//! dropped ends its part with no value and the round still completes. A part ends once because it is consumed: ending
//! it twice, or ending a part of a round that is over, does not compile. A part owns a reference to its join, so the join
//! outlives every part.
//!
//! # Memory ordering
//!
//! A frame's path hits no full barrier, so an arrival is one read-modify-write and nothing else:
//!
//! - `pending` is the round's count. Each arrival is one `fetch_sub(1, AcqRel)`; the arrival that reads 1 is the
//!   last, and it sees, through the release sequence the chain of arrivals forms, every earlier arrival's slot
//!   write (the argument of a reference count's last release).
//! - A part writes its slot plainly before that read-modify-write and the last arrival reads the slots plainly
//!   after it. Arming stores the count with release ordering before any part is handed out, so a part's arrival
//!   happens after the arming it belongs to. The slot a part writes is its own: a part exists once per slot per round,
//!   and the next round cannot be armed (by [`Join::reset`] or [`Round::rearm`]) before every part of this one ended.
//!
//! # Arming
//!
//! [`Join::reset`] arms the next round and refuses while the previous one is armed or its continuation still
//! runs: it takes the right to arm with one compare-and-set. An owner that knows its cell is quiet arms with
//! [`Join::arm_owned`], which is Java's `expect`: a plain release store of the count, no check, no read-modify-write
//! (unsafe: the caller promises that no part of the previous round is outstanding, which only the caller can know when
//! it holds parts it did not drop). [`Join::discard_round`] is the safe form for an owner with exclusive access
//! (`&mut`), which proves that no part exists. A killed unit needs neither: its parts drop and abandon. [`Join::is_idle`]
//! says whether a round is armed.
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
//! for part in join.reset(2) {
//!     part.arrive(1);
//! }
//! let _next = join.reset(2); // armed again
//! ```
//!
//! A part ends once; a second arrival does not compile:
//!
//! ```compile_fail
//! # use euhedral_graph::Join;
//! let mut parts = Join::<u32>::once(2, |_| {});
//! let first = parts.next().unwrap();
//! first.arrive(1);
//! first.arrive(2);
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
use std::mem::{self, ManuallyDrop};
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::sync::Arc;

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
    slots: Box<[Slot<T>]>,
    then: Then<T>,
}

struct Slot<T> {
    /// Written by the part's one arrival, read by the last arrival.
    value: UnsafeCell<Option<T>>,
}

type EachRound<T> = Box<dyn Fn(Round<'_, T>) + Send + Sync>;
type OneRound<T> = Box<dyn FnOnce(Vec<Option<T>>) + Send>;

enum Then<T> {
    /// Runs every round.
    Each(EachRound<T>),
    /// Runs on the first round only.
    Once(UnsafeCell<Option<OneRound<T>>>),
}

// SAFETY: a slot is written by the one arrival that consumed its part and read by the one arrival that
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
    join: &'a Arc<Join<T>>,
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
    /// parts may arrive, on any thread, as soon as they are handed out. The join's usual call from its
    /// continuation, before it delivers the next round's work with the returned parts. Values the continuation did
    /// not take are dropped first. Parts that are dropped unused abandon, so the round does not strand.
    ///
    /// # Panics
    /// When `parts` is 0 or above the capacity. If a value's drop panics, the next round is armed and
    /// the panic is raised after.
    pub fn rearm(self, parts: usize) -> Parts<T> {
        let join = self.join;
        let round = self.parts;
        join.check_parts(parts);
        // The loan ends here; dropping `self` would mark the join idle.
        mem::forget(self);
        let panic = join.clear(round);
        join.arm(parts);
        if let Some(panic) = panic {
            // The next round is armed; its parts end now, so the panic does not strand it.
            drop(Parts::new(join, parts));
            resume_unwind(panic);
        }
        Parts::new(join, parts)
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
    /// A join of `parts` branches (at least one) that runs `then` on the last arrival's thread, and the parts of
    /// its one round. `then` must not wait. One round only: `then` is consumed by it.
    pub fn once(parts: usize, then: impl FnOnce(Vec<Option<T>>) + Send + 'static) -> Parts<T> {
        assert!(parts >= 1, "a join needs at least one part");
        let join = Self::build(parts, Then::Once(UnsafeCell::new(Some(Box::new(then)))));
        join.arm(parts);
        Parts::new(&Arc::new(join), parts)
    }

    /// A cell for a pooled unit: room for `capacity` parts and a continuation stored once, run by the
    /// last arrival of every round. It is not armed until [`Join::reset`]. The continuation receives the
    /// round's values and re-arms the join with [`Round::rearm`] if another round follows. It must not
    /// wait: it delivers work.
    pub fn with_capacity(
        capacity: usize,
        then: impl Fn(Round<'_, T>) + Send + Sync + 'static,
    ) -> Arc<Self> {
        assert!(capacity >= 1, "a join needs at least one part");
        Arc::new(Self::build(capacity, Then::Each(Box::new(then))))
    }
}

impl<T> Join<T> {
    fn build(capacity: usize, then: Then<T>) -> Self {
        assert!(capacity < RESETTING, "a join of {capacity} parts");
        Self {
            pending: AtomicUsize::new(IDLE),
            parts: AtomicUsize::new(0),
            slots: (0..capacity)
                .map(|_| Slot {
                    value: UnsafeCell::new(None),
                })
                .collect(),
            then,
        }
    }

    /// The most parts a round can have.
    pub fn capacity(&self) -> usize {
        self.slots.len()
    }

    /// Arms a round of `parts` parts (1 to the capacity) and hands out its parts. Valid when the
    /// join has never been armed or the previous round's continuation has returned (or re-armed with
    /// [`Round::rearm`], which is the form for inside it).
    ///
    /// # Panics
    /// When the previous round is not complete or its continuation still runs, when `parts` is 0 or above
    /// the capacity, or on a join made by [`Join::once`], which has one round.
    pub fn reset(self: &Arc<Self>, parts: usize) -> Parts<T> {
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
        Parts::new(self, parts)
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
        self.pending.store(parts, Ordering::Release);
    }

    /// Whether no round is armed and no continuation runs: the state of a join that was never armed, or whose last
    /// round ended. An acquire load, a hint for an owner deciding how to arm.
    pub fn is_idle(&self) -> bool {
        self.pending.load(Ordering::Acquire) == IDLE
    }

    /// Discards an unfinished round: drops the values that arrived, forgets the parts that never did and leaves the
    /// join idle, ready for [`Join::reset`]. For an owner that has exclusive access to the cell (`Arc::get_mut`): that
    /// proves that no part exists, because every part holds the join. A round that completed (or never started) is
    /// left idle, as it is. A unit whose frames were killed needs no discard: their parts abandon when dropped.
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
    /// For a cell whose owner knows it is quiet, as one whose previous round it has just proved over by other means.
    /// Parts that still exist (not dropped, not consumed) are the one thing the owner must account for.
    ///
    /// # Safety
    ///
    /// No part of the previous round may arrive or be abandoned from now on (each must have been consumed or
    /// dropped, or be forgotten with [`mem::forget`] and never used), and no continuation of the join may be running
    /// or about to run: the caller is the cell's only user until it hands out the parts. A part of the previous round
    /// that arrives afterwards writes a slot of the new round.
    ///
    /// # Panics
    /// When `parts` is 0 or above the capacity, or on a join made by [`Join::once`], which has one round. If a
    /// value's drop panics, the round is armed and the panic is raised after.
    pub unsafe fn arm_owned(self: &Arc<Self>, parts: usize) -> Parts<T> {
        self.check_parts(parts);
        assert!(
            matches!(self.then, Then::Each(_)),
            "a join made for one round cannot be re-armed"
        );
        let previous = self.parts.load(Ordering::Relaxed);
        let panic = self.clear(previous);
        self.arm(parts);
        if let Some(panic) = panic {
            drop(Parts::new(self, parts));
            resume_unwind(panic);
        }
        Parts::new(self, parts)
    }

    fn end(this: &Arc<Self>, part: usize, value: Option<T>) {
        let _hot = HotScope::enter();
        // `part` is below the round's count, which is at most the capacity: `Parts` hands out indexes of the round only.
        // SAFETY: so the index is in range.
        let slot = unsafe { this.slots.get_unchecked(part) };
        // SAFETY: a part exists once per slot per round (`Parts` hands each index out once, and a round cannot be armed
        // again before every part of it ended), so this arrival has the slot alone. The round's last arrival reads it
        // only after the read-modify-write below, and arming happens after that read.
        slot.value.with_mut(|cell| unsafe { *cell = value });
        // One read-modify-write, and nothing else on this path: the chain of arrivals carries every earlier slot
        // write to the arrival that sees 1.
        if this.pending.fetch_sub(1, Ordering::AcqRel) == 1 {
            Self::finish(this);
        }
    }

    /// The last arrival of a round.
    fn finish(this: &Arc<Self>) {
        let parts = this.parts.load(Ordering::Relaxed);
        let mut round = Round {
            join: this,
            parts,
            _not_sync: PhantomData,
        };
        match &this.then {
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

/// One branch's right to end one part of one round, moved to the branch. [`arrive`](Self::arrive) ends it with a
/// value and [`abandon`](Self::abandon) without; dropping it abandons. It cannot be copied or cloned, and ending
/// consumes it, so a part ends once. It keeps its join alive.
///
/// ```compile_fail
/// # use euhedral_graph::Join;
/// let part = Join::<u8>::once(1, |_| {}).next().unwrap();
/// let copy = part.clone();
/// ```
#[must_use = "a part that is dropped unused abandons its branch"]
pub struct Part<T> {
    /// Moved out when the part ends, by `arrive` or by `drop`: never dropped as a field.
    join: ManuallyDrop<Arc<Join<T>>>,
    index: usize,
}

impl<T> Part<T> {
    /// The part's index in its round: where its value lands in the round.
    pub fn index(&self) -> usize {
        self.index
    }

    /// Ends the part with `value`. The last arrival of the round runs the continuation on this thread.
    pub fn arrive(self, value: T) {
        let (join, index) = self.into_parts();
        Join::end(&join, index, Some(value));
    }

    /// Ends the part without a value.
    pub fn abandon(self) {
        drop(self);
    }

    fn into_parts(self) -> (Arc<Join<T>>, usize) {
        let mut this = ManuallyDrop::new(self);
        // SAFETY: `this` is never used or dropped again, so the `Arc` is moved out exactly once.
        let join = unsafe { ManuallyDrop::take(&mut this.join) };
        (join, this.index)
    }
}

impl<T> Drop for Part<T> {
    fn drop(&mut self) {
        // SAFETY: `drop` runs once, and `self.join` is not used after it (a `ManuallyDrop` field is not dropped).
        let join = unsafe { ManuallyDrop::take(&mut self.join) };
        Join::end(&join, self.index, None);
    }
}

impl<T> fmt::Debug for Part<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Part")
            .field("index", &self.index)
            .finish_non_exhaustive()
    }
}

/// The parts of one armed round, in index order. Parts not taken from the iterator when it drops abandon, so a round
/// whose branches are never started still completes.
#[must_use = "a round whose parts are dropped unused completes with no values"]
pub struct Parts<T> {
    join: Arc<Join<T>>,
    next: usize,
    end: usize,
}

impl<T> Parts<T> {
    fn new(join: &Arc<Join<T>>, parts: usize) -> Self {
        Self {
            join: join.clone(),
            next: 0,
            end: parts,
        }
    }
}

impl<T> Iterator for Parts<T> {
    type Item = Part<T>;

    fn next(&mut self) -> Option<Part<T>> {
        if self.next == self.end {
            return None;
        }
        let index = self.next;
        self.next += 1;
        Some(Part {
            join: ManuallyDrop::new(self.join.clone()),
            index,
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let left = self.end - self.next;
        (left, Some(left))
    }
}

impl<T> ExactSizeIterator for Parts<T> {}

impl<T> Drop for Parts<T> {
    fn drop(&mut self) {
        // The parts not yet handed out end with no value.
        for index in self.next..self.end {
            Join::end(&self.join, index, None);
        }
    }
}

impl<T> fmt::Debug for Parts<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Parts")
            .field("left", &(self.end - self.next))
            .finish_non_exhaustive()
    }
}

#[cfg(all(test, not(loom)))]
mod tests {
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Mutex, OnceLock};

    use super::*;

    type Runs = Arc<Mutex<Vec<Vec<Option<&'static str>>>>>;

    fn recording(parts: usize) -> (Vec<Option<Part<&'static str>>>, Runs) {
        let runs = Arc::new(Mutex::new(Vec::new()));
        let sink = runs.clone();
        let parts = Join::once(parts, move |values| sink.lock().unwrap().push(values));
        (parts.map(Some).collect(), runs)
    }

    /// Part `index` of a round's parts, taken out of the list.
    fn part<T>(parts: &mut [Option<Part<T>>], index: usize) -> Part<T> {
        parts[index].take().expect("the part was not taken yet")
    }

    /// The parts of a fresh round, to take one by one.
    fn arm<T>(join: &Arc<Join<T>>, parts: usize) -> Vec<Option<Part<T>>> {
        join.reset(parts).map(Some).collect()
    }

    #[test]
    fn the_last_arrival_runs_the_continuation_once_with_every_part() {
        let (mut parts, runs) = recording(3);
        part(&mut parts, 2).arrive("c");
        part(&mut parts, 0).abandon();
        assert!(
            runs.lock().unwrap().is_empty(),
            "the join waits for every part"
        );
        part(&mut parts, 1).arrive("b");
        assert_eq!(*runs.lock().unwrap(), [vec![None, Some("b"), Some("c")]]);
    }

    #[test]
    fn a_join_of_one_runs_on_its_only_arrival() {
        let (mut parts, runs) = recording(1);
        part(&mut parts, 0).arrive("only");
        assert_eq!(*runs.lock().unwrap(), [vec![Some("only")]]);
    }

    #[test]
    fn a_part_that_is_dropped_abandons_and_the_round_completes() {
        let (mut parts, runs) = recording(3);
        part(&mut parts, 0).arrive("a");
        // The other two branches were killed and dropped.
        drop(parts);
        assert_eq!(*runs.lock().unwrap(), [vec![Some("a"), None, None]]);
    }

    #[test]
    fn parts_not_taken_from_the_iterator_abandon_when_it_drops() {
        let runs = Arc::new(Mutex::new(Vec::new()));
        let sink = runs.clone();
        let mut parts = Join::once(3, move |values: Vec<Option<u8>>| {
            sink.lock().unwrap().push(values)
        });
        assert_eq!(parts.len(), 3);
        parts.next().unwrap().arrive(7);
        assert_eq!(parts.len(), 2);
        assert!(runs.lock().unwrap().is_empty());
        drop(parts);
        assert_eq!(*runs.lock().unwrap(), [vec![Some(7), None, None]]);
    }

    #[test]
    fn a_part_knows_its_index() {
        let join = Join::<u8>::with_capacity(4, |_| {});
        let indexes: Vec<_> = join.reset(4).map(|part| part.index()).collect();
        assert_eq!(indexes, [0, 1, 2, 3]);
    }

    #[test]
    #[should_panic(expected = "at least one part")]
    fn a_join_of_none_is_refused() {
        let _ = Join::<()>::once(0, |_| {});
    }

    #[test]
    #[allow(clippy::disallowed_methods)] // The test's own threads arrive, as branches on lattice workers would.
    fn parts_arriving_on_many_threads_run_the_continuation_exactly_once() {
        for _ in 0..if cfg!(miri) { 2 } else { 50 } {
            let runs = Arc::new(AtomicUsize::new(0));
            let counted = runs.clone();
            let parts: Vec<_> = Join::once(64, move |values: Vec<Option<usize>>| {
                assert!(values.iter().enumerate().all(|(i, v)| *v == Some(i)));
                counted.fetch_add(1, Ordering::AcqRel);
            })
            .collect();
            let mut chunks: Vec<Vec<_>> = (0..8).map(|_| Vec::new()).collect();
            for part in parts {
                chunks[part.index() / 8].push(part);
            }
            let threads: Vec<_> = chunks
                .into_iter()
                .map(|chunk| {
                    std::thread::spawn(move || {
                        chunk.into_iter().for_each(|part| {
                            let value = part.index();
                            part.arrive(value)
                        })
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
        let join = Join::<usize>::with_capacity(64, move |mut round| {
            assert_eq!(round.len(), 64);
            assert!((0..64).all(|i| round.take(i) == Some(i)));
            counted.fetch_add(1, Ordering::AcqRel);
        });
        for round in 0..ROUNDS {
            let mut chunks: Vec<Vec<_>> = (0..8).map(|_| Vec::new()).collect();
            for part in join.reset(64) {
                chunks[part.index() / 8].push(part);
            }
            let threads: Vec<_> = chunks
                .into_iter()
                .map(|chunk| {
                    std::thread::spawn(move || {
                        chunk.into_iter().for_each(|part| {
                            let value = part.index();
                            part.arrive(value)
                        })
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
    fn rearm_from_inside_the_continuation_hands_out_the_next_round() {
        const ROUNDS: usize = 100;
        let seen = Arc::new(Mutex::new(Vec::new()));
        let next: Arc<Mutex<Vec<Part<usize>>>> = Arc::new(Mutex::new(Vec::new()));
        let (sink, stash) = (seen.clone(), next.clone());
        let join = Join::<usize>::with_capacity(2, move |mut round| {
            let mut seen = sink.lock().unwrap();
            let r = seen.len();
            seen.push((round.take(0), round.take(1)));
            if r + 1 < ROUNDS {
                stash.lock().unwrap().extend(round.rearm(2));
            }
        });
        next.lock().unwrap().extend(join.reset(2));
        for r in 0..ROUNDS {
            let mut parts = std::mem::take(&mut *next.lock().unwrap());
            // The re-arm happened inside the continuation, before the arrival below returned.
            parts.pop().unwrap().arrive(r * 10 + 1);
            parts.pop().unwrap().arrive(r * 10);
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
        for (part, value) in join.reset(3).zip([1, 2, 3]) {
            part.arrive(value);
        }
        let mut parts = arm(&join, 3);
        part(&mut parts, 0).abandon();
        part(&mut parts, 1).arrive(20);
        part(&mut parts, 2).abandon();
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
        join.reset(3).for_each(|p| p.arrive(0));
        join.reset(8).for_each(|p| p.arrive(0));
        join.reset(1).for_each(|p| p.arrive(0));
        assert_eq!(*lens.lock().unwrap(), [3, 8, 1]);
        assert_eq!(join.capacity(), 8);
        assert!(catch_unwind(AssertUnwindSafe(|| join.reset(9))).is_err());
        assert!(catch_unwind(AssertUnwindSafe(|| join.reset(0))).is_err());
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
        let mut parts = arm(&join, 2);
        part(&mut parts, 0).abandon();
        part(&mut parts, 1).arrive("b");
        assert_eq!(*seen.lock().unwrap(), [(None, Some("b"))]);
    }

    #[test]
    #[should_panic(expected = "round")]
    fn reset_before_the_round_completes_panics() {
        let join = Join::<()>::with_capacity(2, |_| {});
        let mut parts = arm(&join, 2);
        part(&mut parts, 0).arrive(());
        let _ = join.reset(2);
    }

    #[test]
    fn reset_through_the_join_while_the_continuation_runs_is_refused() {
        let slot: Arc<OnceLock<Arc<Join<()>>>> = Arc::new(OnceLock::new());
        let weak = Arc::downgrade(&slot);
        let refused = Arc::new(AtomicUsize::new(0));
        let next: Arc<Mutex<Vec<Part<()>>>> = Arc::new(Mutex::new(Vec::new()));
        let (record, stash) = (refused.clone(), next.clone());
        let join = Join::<()>::with_capacity(1, move |round| {
            let join = weak.upgrade().unwrap().get().unwrap().clone();
            // The round is still lent: the join cannot be armed behind its back.
            let result = catch_unwind(AssertUnwindSafe(|| join.reset(1)));
            record.fetch_add(usize::from(result.is_err()), Ordering::AcqRel);
            stash.lock().unwrap().extend(round.rearm(1));
        });
        slot.set(join.clone()).unwrap();
        join.reset(1).for_each(|part| part.arrive(()));
        assert_eq!(refused.load(Ordering::Acquire), 1);
        let again = std::mem::take(&mut *next.lock().unwrap());
        again.into_iter().for_each(|part| part.arrive(())); // the re-armed round
        assert_eq!(refused.load(Ordering::Acquire), 2);
    }

    #[test]
    fn a_continuation_that_panics_leaves_the_join_ready_to_re_arm() {
        let join = Join::<()>::with_capacity(1, |_| panic!("continuation"));
        let part = join.reset(1).next().unwrap();
        assert!(catch_unwind(AssertUnwindSafe(|| part.arrive(()))).is_err());
        assert!(
            join.is_idle(),
            "the round ended although its continuation panicked"
        );
        // Armed again; its part, dropped unused, abandons and runs the continuation, which panics again.
        let parts = join.reset(1);
        assert!(catch_unwind(AssertUnwindSafe(|| drop(parts))).is_err());
        assert!(join.is_idle());
    }

    #[test]
    fn a_join_is_idle_until_armed_and_again_once_its_round_ended() {
        let join = Join::<u8>::with_capacity(2, |_| {});
        assert!(join.is_idle());
        let mut parts = arm(&join, 2);
        assert!(!join.is_idle());
        part(&mut parts, 0).arrive(1);
        assert!(!join.is_idle());
        part(&mut parts, 1).abandon();
        assert!(join.is_idle(), "the continuation returned");
    }

    struct Counted(Arc<AtomicUsize>);

    impl Drop for Counted {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::AcqRel);
        }
    }

    #[test]
    fn an_owner_arms_a_cell_whose_round_it_proved_over_without_the_check() {
        let drops = Arc::new(AtomicUsize::new(0));
        let ran = Arc::new(AtomicUsize::new(0));
        let counted = ran.clone();
        let join = Join::<Counted>::with_capacity(3, move |_| {
            counted.fetch_add(1, Ordering::AcqRel);
        });
        // A round that was cut short: two of three parts arrived, and the third's branch was killed by a means that
        // did not drop its part (it is forgotten, and so is its hold on the join).
        let mut parts = arm(&join, 3);
        part(&mut parts, 0).arrive(Counted(drops.clone()));
        part(&mut parts, 1).arrive(Counted(drops.clone()));
        std::mem::forget(part(&mut parts, 2));
        assert!(!join.is_idle());
        assert!(catch_unwind(AssertUnwindSafe(|| join.reset(2))).is_err());
        // SAFETY: the third part is forgotten and will never arrive, and no continuation runs.
        let mut parts: Vec<_> = unsafe { join.arm_owned(2) }.map(Some).collect();
        assert_eq!(
            drops.load(Ordering::Acquire),
            2,
            "the values that arrived were dropped"
        );
        part(&mut parts, 1).arrive(Counted(drops.clone()));
        part(&mut parts, 0).arrive(Counted(drops.clone()));
        assert_eq!(ran.load(Ordering::Acquire), 1, "the new round ran once");
        // SAFETY: the round ended and its continuation returned.
        unsafe { join.arm_owned(1) }.for_each(|part| part.arrive(Counted(drops.clone())));
        assert_eq!(ran.load(Ordering::Acquire), 2);
    }

    #[test]
    fn discard_round_leaves_a_quiet_join_idle_and_ready_for_reset() {
        let mut join = Join::<u8>::with_capacity(1, |_| {});
        let cell = Arc::get_mut(&mut join).expect("no part exists");
        cell.discard_round();
        assert!(cell.is_idle());
        // A round that ended is left as it is.
        join.reset(1).for_each(|part| part.arrive(1));
        let cell = Arc::get_mut(&mut join).expect("the round ended and its parts are gone");
        cell.discard_round();
        assert!(cell.is_idle());
        let _ = join.reset(1);
    }

    #[test]
    fn arm_owned_refuses_a_bad_part_count_and_a_one_round_join() {
        let join = Join::<u8>::with_capacity(2, |_| {});
        // SAFETY: nothing is armed.
        assert!(catch_unwind(AssertUnwindSafe(|| unsafe { join.arm_owned(0) })).is_err());
        // SAFETY: as above.
        assert!(catch_unwind(AssertUnwindSafe(|| unsafe { join.arm_owned(3) })).is_err());
        let parts = Join::<u8>::once(1, |_| {});
        let once = parts.join.clone();
        // SAFETY: the call refuses before it touches anything.
        assert!(catch_unwind(AssertUnwindSafe(|| unsafe { once.arm_owned(1) })).is_err());
    }

    #[test]
    fn a_join_made_for_one_round_cannot_be_re_armed() {
        let parts = Join::once(1, |_| {});
        let join = parts.join.clone();
        parts.for_each(|part| part.arrive(()));
        assert!(catch_unwind(AssertUnwindSafe(|| join.reset(1))).is_err());
    }

    #[test]
    fn values_a_round_leaves_untaken_are_dropped_when_the_round_ends() {
        let live = Arc::new(());
        let join = Join::<Arc<()>>::with_capacity(2, |_| {});
        let mut parts = arm(&join, 2);
        part(&mut parts, 0).arrive(live.clone());
        assert_eq!(Arc::strong_count(&live), 2);
        part(&mut parts, 1).arrive(live.clone());
        // The continuation took nothing: the round's end drops what it left, a smaller later round
        // does not keep it alive.
        assert_eq!(Arc::strong_count(&live), 1);
        join.reset(1).for_each(Part::abandon);
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
        let mut parts = arm(&join, 2);
        part(&mut parts, 0).arrive(PanicsOnDrop(dropped.clone()));
        // The last arrival's round release drops both values; the first panic comes out of `arrive`,
        // after every value was dropped and the join was released.
        let last = part(&mut parts, 1);
        assert!(
            catch_unwind(AssertUnwindSafe(
                || last.arrive(PanicsOnDrop(dropped.clone()))
            ))
            .is_err()
        );
        assert_eq!(dropped.load(Ordering::Acquire), 2);
        join.reset(1).for_each(Part::abandon);
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
        let join = Join::<usize>::with_capacity(2, move |mut round| {
            let r = counted.fetch_add(1, Ordering::AcqRel);
            assert_eq!((round.take(0), round.take(1)), (Some(r), Some(r + 1)));
            // The unit goes back to the pool: the round ends before another worker can take it.
            round.end();
            if r + 1 < ROUNDS {
                let unit = weak.upgrade().unwrap().get().unwrap().clone();
                let next = std::thread::spawn(move || {
                    // must not find the previous round still held
                    let mut parts = unit.reset(2);
                    parts.next().unwrap().arrive(r + 1);
                    parts.next().unwrap().arrive(r + 2);
                });
                spawned.lock().unwrap().push(next);
            }
        });
        slot.set(join.clone()).unwrap();
        let mut parts = arm(&join, 2);
        part(&mut parts, 0).arrive(0);
        part(&mut parts, 1).arrive(1);
        loop {
            let next = handles.lock().unwrap().pop();
            let Some(next) = next else { break };
            next.join().unwrap();
        }
        assert_eq!(runs.load(Ordering::Acquire), ROUNDS);
    }

    /// `Round` lends `&T` across threads only when `T: Sync`; a part moves with its value.
    #[test]
    fn a_round_is_sync_exactly_when_its_values_are() {
        fn sync<T: Sync>() {}
        fn send<T: Send>() {}
        sync::<Round<'static, u8>>();
        send::<Round<'static, std::cell::Cell<u8>>>();
        send::<Part<std::cell::Cell<u8>>>();
        send::<Parts<std::cell::Cell<u8>>>();
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
            let next: Arc<::loom::sync::Mutex<Vec<Part<usize>>>> =
                Arc::new(::loom::sync::Mutex::new(Vec::new()));
            let (counted, flag, stash) = (runs.clone(), rearmed.clone(), next.clone());
            let join = Join::<usize>::with_capacity(2, move |mut round| {
                let r = counted.fetch_add(1, Ordering::SeqCst);
                assert!(r < 2, "the continuation ran a third time");
                assert_eq!(round.len(), 2);
                assert_eq!(round.take(0), Some(value(r, 0)), "round {r} part 0");
                assert_eq!(round.take(1), Some(value(r, 1)), "round {r} part 1");
                if r == 0 {
                    // Work that follows the re-arm, delivered from inside the continuation.
                    stash.lock().unwrap().extend(round.rearm(2));
                    flag.store(true, Ordering::Release);
                }
            });
            let threads: Vec<_> = join
                .reset(2)
                .map(|part| {
                    let rearmed = rearmed.clone();
                    let next = next.clone();
                    thread::spawn(move || {
                        let index = part.index();
                        part.arrive(value(0, index));
                        while !rearmed.load(Ordering::Acquire) {
                            thread::yield_now();
                        }
                        let second = {
                            let mut parts = next.lock().unwrap();
                            let at = parts.iter().position(|p| p.index() == index).unwrap();
                            parts.swap_remove(at)
                        };
                        second.arrive(value(1, index));
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
            let join = Join::<usize>::with_capacity(2, move |mut round| {
                let r = counted.fetch_add(1, Ordering::SeqCst);
                // Round 0 abandons part 1; round 1 abandons part 0. Neither sees the other's value.
                let expected = if r == 0 {
                    [Some(value(0, 0)), None]
                } else {
                    [None, Some(value(1, 1))]
                };
                assert_eq!([round.take(0), round.take(1)], expected);
            });
            for round in 0..2 {
                let threads: Vec<_> = join
                    .reset(2)
                    .map(|part| {
                        thread::spawn(move || {
                            let index = part.index();
                            if index == round {
                                part.arrive(value(round, index));
                            } else {
                                part.abandon();
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
            let join = Join::<usize>::with_capacity(2, move |mut round| {
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
            });
            let first: Vec<_> = join
                .reset(2)
                .map(|part| {
                    thread::spawn(move || {
                        let index = part.index();
                        part.arrive(value(0, index))
                    })
                })
                .collect();
            let taker = {
                let join = join.clone();
                thread::spawn(move || {
                    while !handed.load(Ordering::Acquire) {
                        thread::yield_now();
                    }
                    // Another worker takes the unit and arms it at once.
                    join.reset(1).for_each(Part::abandon);
                })
            };
            for thread in first {
                thread.join().unwrap();
            }
            taker.join().unwrap();
            assert_eq!(runs.load(Ordering::SeqCst), 2);
        });
    }

    // An owner re-arms a finished round with `arm_owned`, a plain release store, after the arrival threads have been
    // joined: the new round's two arrivals run the continuation once more, and no read of a slot is unordered with
    // the write that fills it. (A cut-short round is covered by the plain test; loom reports a part that is forgotten
    // as a leaked `Arc`.)
    #[test]
    fn loom_an_owner_re_arms_a_finished_round_and_the_next_one_runs_once() {
        ::loom::model(|| {
            let ran = Arc::new(AtomicUsize::new(0));
            let counted = ran.clone();
            let join = Join::<usize>::with_capacity(2, move |round| {
                assert_eq!(round.len(), 2);
                counted.fetch_add(1, Ordering::SeqCst);
            });
            let run = |parts: Parts<usize>| {
                let threads: Vec<_> = parts
                    .map(|part| {
                        thread::spawn(move || {
                            let index = part.index();
                            if index == 0 {
                                part.arrive(1)
                            } else {
                                part.abandon()
                            }
                        })
                    })
                    .collect();
                for thread in threads {
                    thread.join().unwrap();
                }
            };
            run(join.reset(2));
            // SAFETY: the round ended, every part of it ended, and no continuation runs.
            run(unsafe { join.arm_owned(2) });
            assert_eq!(ran.load(Ordering::SeqCst), 2);
        });
    }
}
