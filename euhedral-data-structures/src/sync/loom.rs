//! The atomics and cells the cells in this module use. Under `--cfg loom` they are the model
//! checker's, so a loom test explores every interleaving; otherwise they are the standard library's.

#[cfg(loom)]
pub(crate) use loom::cell::UnsafeCell;
#[cfg(loom)]
pub(crate) use loom::sync::atomic::{AtomicBool, AtomicPtr, AtomicU8, AtomicU64, Ordering, fence};
#[cfg(not(loom))]
pub(crate) use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU8, AtomicU64, Ordering, fence};

/// `std`'s `UnsafeCell` with loom's closure-based access, so one body serves both builds.
#[cfg(not(loom))]
#[derive(Debug)]
pub(crate) struct UnsafeCell<T>(std::cell::UnsafeCell<T>);

#[cfg(not(loom))]
impl<T> UnsafeCell<T> {
    pub(crate) const fn new(value: T) -> Self {
        Self(std::cell::UnsafeCell::new(value))
    }

    pub(crate) fn with_mut<R>(&self, f: impl FnOnce(*mut T) -> R) -> R {
        f(self.0.get())
    }
}
