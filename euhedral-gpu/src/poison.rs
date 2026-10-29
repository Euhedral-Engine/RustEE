//! A device's first fault, set once, read without locking.

use std::sync::atomic::{AtomicPtr, Ordering};

use crate::GpuError;

/// The first fault of a device. Once set, later submissions fail fast with it.
///
/// Lock-free: the cause is published with one compare-and-swap, so the first setter wins and a reader never waits.
pub struct Poison {
    /// Null, or a `Box<GpuError>` made by `set` and freed only in `drop`.
    cause: AtomicPtr<GpuError>,
}

impl std::fmt::Debug for Poison {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Poison")
            .field("cause", &self.get())
            .finish()
    }
}

impl Default for Poison {
    fn default() -> Self {
        Self {
            cause: AtomicPtr::new(std::ptr::null_mut()),
        }
    }
}

impl Poison {
    pub fn new() -> Self {
        Self::default()
    }

    /// Records `cause` unless a cause is already recorded: the first fault wins.
    pub fn set(&self, cause: GpuError) {
        let boxed = Box::into_raw(Box::new(cause));
        if self
            .cause
            .compare_exchange(
                std::ptr::null_mut(),
                boxed,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_err()
        {
            // SAFETY: `boxed` was made just above and was not published: nothing else can see it.
            drop(unsafe { Box::from_raw(boxed) });
        }
    }

    pub fn get(&self) -> Option<GpuError> {
        let cause = self.cause.load(Ordering::Acquire);
        // SAFETY: a non-null pointer is a published `Box<GpuError>` that is neither changed nor freed before `drop`,
        // and `&self` keeps `drop` from running.
        unsafe { cause.as_ref() }.cloned()
    }

    /// `Err(Poisoned(cause))` once poisoned.
    pub fn check(&self) -> Result<(), GpuError> {
        match self.get() {
            Some(cause) => Err(GpuError::Poisoned(Box::new(cause))),
            None => Ok(()),
        }
    }
}

impl Drop for Poison {
    fn drop(&mut self) {
        let cause = *self.cause.get_mut();
        if !cause.is_null() {
            // SAFETY: the published box, freed once, with no other handle left (`&mut self`).
            drop(unsafe { Box::from_raw(cause) });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fault(code: i32) -> GpuError {
        GpuError::Fault {
            call: "test",
            code,
            name: "X".into(),
        }
    }

    #[test]
    fn poison_keeps_the_first_cause() {
        let p = Poison::new();
        assert_eq!(p.get(), None);
        assert_eq!(p.check(), Ok(()));
        p.set(fault(1));
        p.set(fault(2));
        assert_eq!(p.get(), Some(fault(1)));
        assert_eq!(p.check(), Err(GpuError::Poisoned(Box::new(fault(1)))));
    }
}
