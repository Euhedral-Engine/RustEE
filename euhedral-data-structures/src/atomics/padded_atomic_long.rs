use super::CachePadded;
use std::sync::atomic::{AtomicI64, Ordering};

/// An `i64` atomic alone on its cache line (`PaddedAtomicLong` upstream).
///
/// Java's volatile access is sequentially consistent, `*Opaque`/`*Plain` map to relaxed, and
/// `*Acquire`/`*Release` map to the same orderings. Arithmetic wraps on overflow like Java `long`.
#[derive(Debug, Default)]
pub struct PaddedAtomicLong {
    value: CachePadded<AtomicI64>,
}

impl PaddedAtomicLong {
    pub fn new(value: i64) -> Self {
        Self {
            value: CachePadded::new(AtomicI64::new(value)),
        }
    }

    // ----- Get -----

    pub fn get(&self) -> i64 {
        self.value.load(Ordering::SeqCst)
    }

    pub fn get_acquire(&self) -> i64 {
        self.value.load(Ordering::Acquire)
    }

    pub fn get_opaque(&self) -> i64 {
        self.value.load(Ordering::Relaxed)
    }

    pub fn set_opaque(&self, val: i64) {
        self.value.store(val, Ordering::Relaxed);
    }

    pub fn get_plain(&self) -> i64 {
        self.value.load(Ordering::Relaxed)
    }

    // ----- Set -----

    pub fn set_plain(&self, val: i64) {
        self.value.store(val, Ordering::Relaxed);
    }

    pub fn get_and_set(&self, val: i64) -> i64 {
        self.value.swap(val, Ordering::SeqCst)
    }

    pub fn set(&self, val: i64) {
        self.value.store(val, Ordering::SeqCst);
    }

    pub fn lazy_set(&self, val: i64) {
        self.set_release(val);
    }

    pub fn set_release(&self, val: i64) {
        self.value.store(val, Ordering::Release);
    }

    // ----- RMW -----

    pub fn get_and_increment(&self) -> i64 {
        self.get_and_add(1)
    }

    pub fn get_and_decrement(&self) -> i64 {
        self.get_and_add(-1)
    }

    pub fn increment_and_get(&self) -> i64 {
        self.add_and_get(1)
    }

    pub fn decrement_and_get(&self) -> i64 {
        self.add_and_get(-1)
    }

    pub fn get_and_add(&self, val: i64) -> i64 {
        self.value.fetch_add(val, Ordering::SeqCst)
    }

    pub fn get_and_add_release(&self, val: i64) -> i64 {
        self.value.fetch_add(val, Ordering::Release)
    }

    pub fn add_and_get(&self, val: i64) -> i64 {
        self.get_and_add(val).wrapping_add(val)
    }

    // ----- CAS -----

    pub fn get_and_update(&self, mut update: impl FnMut(i64) -> i64) -> i64 {
        loop {
            let prev = self.get();
            if self.weak_compare_and_set(prev, update(prev)) {
                return prev;
            }
        }
    }

    pub fn update_and_get(&self, mut update: impl FnMut(i64) -> i64) -> i64 {
        loop {
            let prev = self.get();
            let next = update(prev);
            if self.weak_compare_and_set(prev, next) {
                return next;
            }
        }
    }

    pub fn get_and_accumulate(
        &self,
        val: i64,
        mut accumulator: impl FnMut(i64, i64) -> i64,
    ) -> i64 {
        loop {
            let prev = self.get();
            if self.compare_and_set(prev, accumulator(prev, val)) {
                return prev;
            }
        }
    }

    pub fn accumulate_and_get(
        &self,
        val: i64,
        mut accumulator: impl FnMut(i64, i64) -> i64,
    ) -> i64 {
        loop {
            let prev = self.get();
            let next = accumulator(prev, val);
            if self.compare_and_set(prev, next) {
                return next;
            }
        }
    }

    pub fn compare_and_set(&self, curr: i64, next: i64) -> bool {
        self.value
            .compare_exchange(curr, next, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
    }

    pub fn weak_compare_and_set(&self, curr: i64, next: i64) -> bool {
        self.value
            .compare_exchange_weak(curr, next, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
    }

    /// Returns the witness value: equal to `curr` exactly when the exchange happened.
    pub fn compare_and_exchange(&self, curr: i64, next: i64) -> i64 {
        match self
            .value
            .compare_exchange(curr, next, Ordering::SeqCst, Ordering::SeqCst)
        {
            Ok(witness) | Err(witness) => witness,
        }
    }
}
