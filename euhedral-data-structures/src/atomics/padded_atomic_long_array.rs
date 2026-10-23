use super::padded_slots::PaddedSlots;
use std::fmt;
use std::sync::atomic::{AtomicI64, Ordering};

/// A fixed-length array of `i64` atomics with each element on its own cache line
/// (`PaddedAtomicLongArray` upstream).
///
/// With `bounds_check` on, indexes are logical (`0..length`). With it off, indexes are physical slots,
/// as returned by [`physical_idx_for_raw`](Self::physical_idx_for_raw); the unused padding slots between elements stay
/// zero and are reachable.
pub struct PaddedAtomicLongArray {
    pub(crate) slots: PaddedSlots<AtomicI64>,
}

impl PaddedAtomicLongArray {
    pub fn new(length: usize) -> Self {
        Self::with_options(length, true, false)
    }

    pub fn with_options(length: usize, bounds_check: bool, pad128: bool) -> Self {
        Self {
            slots: PaddedSlots::new(length, bounds_check, pad128),
        }
    }

    // ----- Get -----

    pub fn get(&self, idx: usize) -> i64 {
        self.slots.slot(idx).load(Ordering::SeqCst)
    }

    pub fn get_acquire(&self, idx: usize) -> i64 {
        self.slots.slot(idx).load(Ordering::Acquire)
    }

    pub fn get_opaque(&self, idx: usize) -> i64 {
        self.slots.slot(idx).load(Ordering::Relaxed)
    }

    pub fn get_plain(&self, idx: usize) -> i64 {
        self.slots.slot(idx).load(Ordering::Relaxed)
    }

    pub fn get_and_set(&self, idx: usize, val: i64) -> i64 {
        self.slots.slot(idx).swap(val, Ordering::SeqCst)
    }

    // ----- Set -----

    pub fn set(&self, idx: usize, val: i64) {
        self.slots.slot(idx).store(val, Ordering::SeqCst);
    }

    pub fn lazy_set(&self, idx: usize, val: i64) {
        self.set_release(idx, val);
    }

    pub fn set_release(&self, idx: usize, val: i64) {
        self.slots.slot(idx).store(val, Ordering::Release);
    }

    pub fn set_opaque(&self, idx: usize, val: i64) {
        self.slots.slot(idx).store(val, Ordering::Relaxed);
    }

    pub fn set_plain(&self, idx: usize, val: i64) {
        self.slots.slot(idx).store(val, Ordering::Relaxed);
    }

    pub fn fill(&self, val: i64) {
        self.fill_with(val, Ordering::SeqCst);
    }

    pub fn lazy_fill(&self, val: i64) {
        self.fill_release(val);
    }

    pub fn fill_release(&self, val: i64) {
        self.fill_with(val, Ordering::Release);
    }

    pub fn fill_opaque(&self, val: i64) {
        self.fill_with(val, Ordering::Relaxed);
    }

    pub fn fill_plain(&self, val: i64) {
        self.fill_with(val, Ordering::Relaxed);
    }

    fn fill_with(&self, val: i64, order: Ordering) {
        for slot in self.slots.logical_slots() {
            slot.store(val, order);
        }
    }

    // ----- RMW -----

    pub fn get_and_increment(&self, idx: usize) -> i64 {
        self.get_and_add(idx, 1)
    }

    pub fn get_and_decrement(&self, idx: usize) -> i64 {
        self.get_and_add(idx, -1)
    }

    pub fn increment_and_get(&self, idx: usize) -> i64 {
        self.add_and_get(idx, 1)
    }

    pub fn decrement_and_get(&self, idx: usize) -> i64 {
        self.add_and_get(idx, -1)
    }

    pub fn get_and_add(&self, idx: usize, val: i64) -> i64 {
        self.slots.slot(idx).fetch_add(val, Ordering::SeqCst)
    }

    pub fn add_and_get(&self, idx: usize, val: i64) -> i64 {
        self.get_and_add(idx, val).wrapping_add(val)
    }

    pub fn get_and_add_release(&self, idx: usize, val: i64) -> i64 {
        self.slots.slot(idx).fetch_add(val, Ordering::Release)
    }

    pub fn add_release_and_get(&self, idx: usize, val: i64) -> i64 {
        self.get_and_add_release(idx, val).wrapping_add(val)
    }

    // ----- CAS -----

    pub fn get_and_update(&self, idx: usize, mut update: impl FnMut(i64) -> i64) -> i64 {
        let slot = self.slots.slot(idx);
        loop {
            let prev = slot.load(Ordering::Acquire);
            let next = update(prev);
            if slot
                .compare_exchange_weak(prev, next, Ordering::SeqCst, Ordering::Relaxed)
                .is_ok()
            {
                return prev;
            }
        }
    }

    pub fn update_and_get(&self, idx: usize, mut update: impl FnMut(i64) -> i64) -> i64 {
        let slot = self.slots.slot(idx);
        loop {
            let prev = slot.load(Ordering::Acquire);
            let next = update(prev);
            if slot
                .compare_exchange_weak(prev, next, Ordering::SeqCst, Ordering::Relaxed)
                .is_ok()
            {
                return next;
            }
        }
    }

    pub fn get_and_accumulate(
        &self,
        idx: usize,
        val: i64,
        mut accumulator: impl FnMut(i64, i64) -> i64,
    ) -> i64 {
        let slot = self.slots.slot(idx);
        loop {
            let prev = slot.load(Ordering::Acquire);
            let next = accumulator(prev, val);
            if slot
                .compare_exchange(prev, next, Ordering::SeqCst, Ordering::Relaxed)
                .is_ok()
            {
                return prev;
            }
        }
    }

    pub fn accumulate_and_get(
        &self,
        idx: usize,
        val: i64,
        mut accumulator: impl FnMut(i64, i64) -> i64,
    ) -> i64 {
        let slot = self.slots.slot(idx);
        loop {
            let prev = slot.load(Ordering::Acquire);
            let next = accumulator(prev, val);
            if slot
                .compare_exchange(prev, next, Ordering::SeqCst, Ordering::Relaxed)
                .is_ok()
            {
                return next;
            }
        }
    }

    pub fn compare_and_set(&self, idx: usize, expect: i64, update: i64) -> bool {
        self.slots
            .slot(idx)
            .compare_exchange(expect, update, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
    }

    pub fn weak_compare_and_set(&self, idx: usize, expect: i64, update: i64) -> bool {
        self.slots
            .slot(idx)
            .compare_exchange_weak(expect, update, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
    }

    /// Returns the witness value: equal to `expect` exactly when the exchange happened.
    pub fn compare_and_exchange(&self, idx: usize, expect: i64, update: i64) -> i64 {
        match self.slots.slot(idx).compare_exchange(
            expect,
            update,
            Ordering::SeqCst,
            Ordering::SeqCst,
        ) {
            Ok(witness) | Err(witness) => witness,
        }
    }

    /// Maps a hash or counter onto the physical slot of one of the elements.
    pub fn physical_idx_for_raw(&self, raw_idx: i64) -> usize {
        self.slots.physical_idx_for_raw(raw_idx)
    }

    pub fn length(&self) -> usize {
        self.slots.length()
    }
}

impl fmt::Debug for PaddedAtomicLongArray {
    /// Lists the logical elements, or every physical slot when bounds checks are off.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut list = f.debug_list();
        if self.slots.bounds_check {
            list.entries(
                self.slots
                    .logical_slots()
                    .map(|slot| slot.load(Ordering::Acquire)),
            );
        } else {
            list.entries(
                self.slots
                    .array
                    .iter()
                    .map(|slot| slot.load(Ordering::Acquire)),
            );
        }
        list.finish()
    }
}
