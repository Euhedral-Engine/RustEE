use super::PaddedAtomicLongArray;
use super::thread_random;
use std::ops::Deref;
use std::sync::atomic::{AtomicI64, Ordering};

/// A striped `i64` counter (`PaddedLongAdder` upstream): writers spread over cache-line padded slots
/// and readers combine them. All [`PaddedAtomicLongArray`] operations are reached through `Deref`.
///
/// Reads are not a snapshot: each slot is read once while writers may keep going.
#[derive(Debug)]
pub struct PaddedLongAdder {
    array: PaddedAtomicLongArray,
}

impl PaddedLongAdder {
    /// One slot per available CPU.
    pub fn new() -> Self {
        let cpus = std::thread::available_parallelism().map_or(1, |n| n.get());
        Self::with_length(cpus)
    }

    pub fn with_length(length: usize) -> Self {
        Self::with_options(length, true, false)
    }

    pub fn with_options(length: usize, bounds_check: bool, pad128: bool) -> Self {
        Self {
            array: PaddedAtomicLongArray::with_options(length, bounds_check, pad128),
        }
    }

    /// Adds to a randomly chosen slot.
    pub fn increment(&self) {
        self.add(1);
    }

    pub fn increment_at(&self, idx: usize) {
        self.add_at(idx, 1);
    }

    pub fn decrement(&self) {
        self.add(-1);
    }

    pub fn decrement_at(&self, idx: usize) {
        self.add_at(idx, -1);
    }

    pub fn add(&self, value: i64) {
        self.random_slot().fetch_add(value, Ordering::Release);
    }

    pub fn add_at(&self, idx: usize, value: i64) {
        self.array.get_and_add_release(idx, value);
    }

    /// The smallest slot, or `i64::MAX` when there are no slots.
    pub fn min(&self) -> i64 {
        self.slots().fold(i64::MAX, i64::min)
    }

    /// The largest slot, or `i64::MIN` when there are no slots.
    pub fn max(&self) -> i64 {
        self.slots().fold(i64::MIN, i64::max)
    }

    /// The exact mean of the slots, or `0.0` when there are none.
    pub fn mean(&self) -> f64 {
        let (sum, count) = self.slots().fold((0_i128, 0_u32), |(sum, count), v| {
            (sum + i128::from(v), count + 1)
        });
        if count == 0 {
            return 0.0;
        }
        sum as f64 / f64::from(count)
    }

    /// The wrapping sum of the slots.
    pub fn sum(&self) -> i64 {
        self.slots().fold(0, i64::wrapping_add)
    }

    pub fn sum_and_reset(&self) -> i64 {
        self.array.slots.logical_slots().fold(0, |sum, slot| {
            sum.wrapping_add(slot.swap(0, Ordering::SeqCst))
        })
    }

    pub fn reset(&self) {
        self.array.fill_release(0);
    }

    fn slots(&self) -> impl Iterator<Item = i64> + '_ {
        self.array
            .slots
            .logical_slots()
            .map(|slot| slot.load(Ordering::Acquire))
    }

    /// A uniformly chosen slot, reached by physical index so it works with or without bounds checks.
    fn random_slot(&self) -> &AtomicI64 {
        let slots = &self.array.slots;
        &slots.array[slots.physical_idx_for_raw(thread_random::next_u64() as i64)]
    }
}

impl Default for PaddedLongAdder {
    fn default() -> Self {
        Self::new()
    }
}

impl Deref for PaddedLongAdder {
    type Target = PaddedAtomicLongArray;

    fn deref(&self) -> &PaddedAtomicLongArray {
        &self.array
    }
}
