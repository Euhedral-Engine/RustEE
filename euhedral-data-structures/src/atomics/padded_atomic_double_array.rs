use super::AtomicDouble;
use super::padded_slots::PaddedSlots;
use std::fmt;

/// A fixed-length array of `f64` atomics with each element on its own cache line
/// (`PaddedAtomicDoubleArray` upstream).
///
/// With `bounds_check` on, indexes are logical (`0..length`). With it off, indexes are physical slots,
/// as returned by [`physical_idx_for_raw`](Self::physical_idx_for_raw); the unused padding slots
/// between elements stay zero and are reachable. Compare-and-set compares raw bits.
pub struct PaddedAtomicDoubleArray {
    pub(crate) slots: PaddedSlots<AtomicDouble>,
}

impl PaddedAtomicDoubleArray {
    pub fn new(length: usize) -> Self {
        Self::with_options(length, true, false)
    }

    pub fn with_options(length: usize, bounds_check: bool, pad128: bool) -> Self {
        Self {
            slots: PaddedSlots::new(length, bounds_check, pad128),
        }
    }

    // ----- Get -----

    pub fn get(&self, idx: usize) -> f64 {
        self.slots.slot(idx).get()
    }

    pub fn get_acquire(&self, idx: usize) -> f64 {
        self.slots.slot(idx).get_acquire()
    }

    pub fn get_opaque(&self, idx: usize) -> f64 {
        self.slots.slot(idx).get_opaque()
    }

    pub fn get_plain(&self, idx: usize) -> f64 {
        self.slots.slot(idx).get_plain()
    }

    pub fn get_and_set(&self, idx: usize, val: f64) -> f64 {
        self.slots.slot(idx).get_and_set(val)
    }

    // ----- Set -----

    pub fn set(&self, idx: usize, val: f64) {
        self.slots.slot(idx).set(val);
    }

    pub fn lazy_set(&self, idx: usize, val: f64) {
        self.set_release(idx, val);
    }

    pub fn set_release(&self, idx: usize, val: f64) {
        self.slots.slot(idx).set_release(val);
    }

    pub fn set_opaque(&self, idx: usize, val: f64) {
        self.slots.slot(idx).set_opaque(val);
    }

    pub fn set_plain(&self, idx: usize, val: f64) {
        self.slots.slot(idx).set_plain(val);
    }

    pub fn fill(&self, val: f64) {
        self.slots.logical_slots().for_each(|slot| slot.set(val));
    }

    pub fn lazy_fill(&self, val: f64) {
        self.fill_release(val);
    }

    pub fn fill_release(&self, val: f64) {
        self.slots
            .logical_slots()
            .for_each(|slot| slot.set_release(val));
    }

    pub fn fill_opaque(&self, val: f64) {
        self.slots
            .logical_slots()
            .for_each(|slot| slot.set_opaque(val));
    }

    pub fn fill_plain(&self, val: f64) {
        self.slots
            .logical_slots()
            .for_each(|slot| slot.set_plain(val));
    }

    // ----- RMW -----

    pub fn get_and_increment(&self, idx: usize) -> f64 {
        self.get_and_add(idx, 1.0)
    }

    pub fn get_and_decrement(&self, idx: usize) -> f64 {
        self.get_and_add(idx, -1.0)
    }

    pub fn increment_and_get(&self, idx: usize) -> f64 {
        self.add_and_get(idx, 1.0)
    }

    pub fn decrement_and_get(&self, idx: usize) -> f64 {
        self.add_and_get(idx, -1.0)
    }

    pub fn get_and_add(&self, idx: usize, val: f64) -> f64 {
        self.slots.slot(idx).get_and_add(val)
    }

    pub fn add_and_get(&self, idx: usize, val: f64) -> f64 {
        self.get_and_add(idx, val) + val
    }

    pub fn get_and_add_release(&self, idx: usize, val: f64) -> f64 {
        self.slots.slot(idx).get_and_add_release(val)
    }

    pub fn add_release_and_get(&self, idx: usize, val: f64) -> f64 {
        self.get_and_add_release(idx, val) + val
    }

    // ----- CAS -----

    pub fn get_and_update(&self, idx: usize, mut update: impl FnMut(f64) -> f64) -> f64 {
        let slot = self.slots.slot(idx);
        loop {
            let prev = slot.get_acquire();
            if slot.weak_compare_and_set(prev, update(prev)) {
                return prev;
            }
        }
    }

    pub fn update_and_get(&self, idx: usize, mut update: impl FnMut(f64) -> f64) -> f64 {
        let slot = self.slots.slot(idx);
        loop {
            let prev = slot.get_acquire();
            let next = update(prev);
            if slot.weak_compare_and_set(prev, next) {
                return next;
            }
        }
    }

    pub fn get_and_accumulate(
        &self,
        idx: usize,
        val: f64,
        mut accumulator: impl FnMut(f64, f64) -> f64,
    ) -> f64 {
        let slot = self.slots.slot(idx);
        loop {
            let prev = slot.get_acquire();
            if slot.compare_and_set(prev, accumulator(prev, val)) {
                return prev;
            }
        }
    }

    pub fn accumulate_and_get(
        &self,
        idx: usize,
        val: f64,
        mut accumulator: impl FnMut(f64, f64) -> f64,
    ) -> f64 {
        let slot = self.slots.slot(idx);
        loop {
            let prev = slot.get_acquire();
            let next = accumulator(prev, val);
            if slot.compare_and_set(prev, next) {
                return next;
            }
        }
    }

    pub fn compare_and_set(&self, idx: usize, expect: f64, update: f64) -> bool {
        self.slots.slot(idx).compare_and_set(expect, update)
    }

    pub fn weak_compare_and_set(&self, idx: usize, expect: f64, update: f64) -> bool {
        self.slots.slot(idx).weak_compare_and_set(expect, update)
    }

    /// Returns the witness value: bitwise equal to `expect` exactly when the exchange happened.
    pub fn compare_and_exchange(&self, idx: usize, expect: f64, update: f64) -> f64 {
        self.slots.slot(idx).compare_and_exchange(expect, update)
    }

    /// Maps a hash or counter onto the physical slot of one of the elements.
    pub fn physical_idx_for_raw(&self, raw_idx: i64) -> usize {
        self.slots.physical_idx_for_raw(raw_idx)
    }

    pub fn length(&self) -> usize {
        self.slots.length()
    }
}

impl fmt::Debug for PaddedAtomicDoubleArray {
    /// Lists the logical elements, or every physical slot when bounds checks are off.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut list = f.debug_list();
        if self.slots.bounds_check {
            list.entries(self.slots.logical_slots().map(AtomicDouble::get_acquire));
        } else {
            list.entries(self.slots.array.iter().map(AtomicDouble::get_acquire));
        }
        list.finish()
    }
}
