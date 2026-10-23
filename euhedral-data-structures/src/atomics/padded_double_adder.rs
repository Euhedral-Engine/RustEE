use super::thread_random;
use super::{AtomicDouble, PaddedAtomicDoubleArray};
use std::ops::Deref;

/// A striped `f64` counter (`PaddedDoubleAdder` upstream). Non-finite slots are ignored by every
/// reduction. All [`PaddedAtomicDoubleArray`] operations are reached through `Deref`.
#[derive(Debug)]
pub struct PaddedDoubleAdder {
    array: PaddedAtomicDoubleArray,
}

impl PaddedDoubleAdder {
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
            array: PaddedAtomicDoubleArray::with_options(length, bounds_check, pad128),
        }
    }

    /// Adds to a randomly chosen slot.
    pub fn increment(&self) {
        self.add(1.0);
    }

    pub fn increment_at(&self, idx: usize) {
        self.add_at(idx, 1.0);
    }

    pub fn decrement(&self) {
        self.add(-1.0);
    }

    pub fn decrement_at(&self, idx: usize) {
        self.add_at(idx, -1.0);
    }

    pub fn add(&self, value: f64) {
        self.random_slot().get_and_add_release(value);
    }

    pub fn add_at(&self, idx: usize, value: f64) {
        self.array.get_and_add_release(idx, value);
    }

    /// The smallest finite slot, or `f64::MAX` when there is none.
    pub fn min(&self) -> f64 {
        self.finite_slots().fold(f64::MAX, f64::min)
    }

    /// The largest finite slot, or `f64::MIN` when there is none.
    pub fn max(&self) -> f64 {
        self.finite_slots().fold(f64::MIN, f64::max)
    }

    /// The mean of the finite slots, or `0.0` when there are none.
    pub fn mean(&self) -> f64 {
        let (sum, count) = self
            .finite_slots()
            .fold((0.0, 0_u32), |(sum, count), v| (sum + v, count + 1));
        if count == 0 {
            return 0.0;
        }
        sum / f64::from(count)
    }

    /// The sum of the finite slots.
    pub fn sum(&self) -> f64 {
        // Not `Iterator::sum`, which starts from -0.0 and would return it when no slot is finite.
        self.finite_slots().fold(0.0, |sum, value| sum + value)
    }

    /// Returns the sum of the finite slots and zeroes every slot, finite or not.
    pub fn sum_and_reset(&self) -> f64 {
        self.array
            .slots
            .logical_slots()
            .map(|slot| slot.get_and_set(0.0))
            .filter(|value| value.is_finite())
            .fold(0.0, |sum, value| sum + value)
    }

    pub fn reset(&self) {
        self.array.fill_release(0.0);
    }

    fn finite_slots(&self) -> impl Iterator<Item = f64> + '_ {
        self.array
            .slots
            .logical_slots()
            .map(AtomicDouble::get_acquire)
            .filter(|value| value.is_finite())
    }

    /// A uniformly chosen slot, reached by physical index so it works with or without bounds checks.
    fn random_slot(&self) -> &AtomicDouble {
        let slots = &self.array.slots;
        &slots.array[slots.physical_idx_for_raw(thread_random::next_u64() as i64)]
    }
}

impl Default for PaddedDoubleAdder {
    fn default() -> Self {
        Self::new()
    }
}

impl Deref for PaddedDoubleAdder {
    type Target = PaddedAtomicDoubleArray;

    fn deref(&self) -> &PaddedAtomicDoubleArray {
        &self.array
    }
}
