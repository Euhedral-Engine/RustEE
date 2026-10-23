use std::sync::atomic::{AtomicU64, Ordering};

/// An atomic `f64` stored as its IEEE-754 bits (`AtomicDouble` upstream).
///
/// Like a Java `VarHandle` on a `double`, compare-and-set compares raw bits, so `NaN` matches itself
/// and `0.0` does not match `-0.0`. Add is a compare-and-set loop.
#[derive(Debug, Default)]
pub struct AtomicDouble {
    bits: AtomicU64,
}

impl AtomicDouble {
    pub fn new(value: f64) -> Self {
        Self {
            bits: AtomicU64::new(value.to_bits()),
        }
    }

    // ----- Get -----

    pub fn get(&self) -> f64 {
        f64::from_bits(self.bits.load(Ordering::SeqCst))
    }

    pub fn get_acquire(&self) -> f64 {
        f64::from_bits(self.bits.load(Ordering::Acquire))
    }

    pub fn get_opaque(&self) -> f64 {
        f64::from_bits(self.bits.load(Ordering::Relaxed))
    }

    pub fn set_opaque(&self, val: f64) {
        self.bits.store(val.to_bits(), Ordering::Relaxed);
    }

    pub fn get_plain(&self) -> f64 {
        f64::from_bits(self.bits.load(Ordering::Relaxed))
    }

    // ----- Set -----

    pub fn set_plain(&self, val: f64) {
        self.bits.store(val.to_bits(), Ordering::Relaxed);
    }

    pub fn get_and_set(&self, val: f64) -> f64 {
        f64::from_bits(self.bits.swap(val.to_bits(), Ordering::SeqCst))
    }

    pub fn set(&self, val: f64) {
        self.bits.store(val.to_bits(), Ordering::SeqCst);
    }

    pub fn lazy_set(&self, val: f64) {
        self.set_release(val);
    }

    pub fn set_release(&self, val: f64) {
        self.bits.store(val.to_bits(), Ordering::Release);
    }

    // ----- RMW -----

    pub fn get_and_increment(&self) -> f64 {
        self.get_and_add(1.0)
    }

    pub fn get_and_decrement(&self) -> f64 {
        self.get_and_add(-1.0)
    }

    pub fn increment_and_get(&self) -> f64 {
        self.add_and_get(1.0)
    }

    pub fn decrement_and_get(&self) -> f64 {
        self.add_and_get(-1.0)
    }

    pub fn get_and_add(&self, val: f64) -> f64 {
        self.add_with(val, Ordering::SeqCst)
    }

    /// Add with release ordering on success. Only the padded arrays and adders use it.
    pub(crate) fn get_and_add_release(&self, val: f64) -> f64 {
        self.add_with(val, Ordering::Release)
    }

    fn add_with(&self, val: f64, success: Ordering) -> f64 {
        let mut prev = self.bits.load(Ordering::Relaxed);
        loop {
            let next = (f64::from_bits(prev) + val).to_bits();
            match self
                .bits
                .compare_exchange_weak(prev, next, success, Ordering::Relaxed)
            {
                Ok(_) => return f64::from_bits(prev),
                Err(witness) => prev = witness,
            }
        }
    }

    pub fn add_and_get(&self, val: f64) -> f64 {
        self.get_and_add(val) + val
    }

    // ----- CAS -----

    pub fn get_and_update(&self, mut update: impl FnMut(f64) -> f64) -> f64 {
        loop {
            let prev = self.get();
            if self.weak_compare_and_set(prev, update(prev)) {
                return prev;
            }
        }
    }

    pub fn update_and_get(&self, mut update: impl FnMut(f64) -> f64) -> f64 {
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
        val: f64,
        mut accumulator: impl FnMut(f64, f64) -> f64,
    ) -> f64 {
        loop {
            let prev = self.get();
            if self.compare_and_set(prev, accumulator(prev, val)) {
                return prev;
            }
        }
    }

    pub fn accumulate_and_get(
        &self,
        val: f64,
        mut accumulator: impl FnMut(f64, f64) -> f64,
    ) -> f64 {
        loop {
            let prev = self.get();
            let next = accumulator(prev, val);
            if self.compare_and_set(prev, next) {
                return next;
            }
        }
    }

    pub fn compare_and_set(&self, curr: f64, next: f64) -> bool {
        self.bits
            .compare_exchange(
                curr.to_bits(),
                next.to_bits(),
                Ordering::SeqCst,
                Ordering::SeqCst,
            )
            .is_ok()
    }

    pub fn weak_compare_and_set(&self, curr: f64, next: f64) -> bool {
        self.bits
            .compare_exchange_weak(
                curr.to_bits(),
                next.to_bits(),
                Ordering::SeqCst,
                Ordering::SeqCst,
            )
            .is_ok()
    }

    /// Returns the witness value: bitwise equal to `curr` exactly when the exchange happened.
    pub fn compare_and_exchange(&self, curr: f64, next: f64) -> f64 {
        let witness = match self.bits.compare_exchange(
            curr.to_bits(),
            next.to_bits(),
            Ordering::SeqCst,
            Ordering::SeqCst,
        ) {
            Ok(witness) | Err(witness) => witness,
        };
        f64::from_bits(witness)
    }
}
