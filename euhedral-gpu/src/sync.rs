//! The atomics and cells the lock-free marker ring and the graph instance pool are built from. A build with `--cfg loom`
//! swaps them for the `loom` model checker's, which explores every interleaving the memory model allows.

#[cfg(loom)]
pub(crate) use loom::sync::atomic::{AtomicU32, AtomicU64, Ordering};
#[cfg(not(loom))]
pub(crate) use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

/// An `UnsafeCell` whose accesses are closures over a raw pointer, the shape `loom`'s cell has, so the model checker
/// sees each access and can report an unordered one.
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

#[cfg(loom)]
pub(crate) use loom::cell::UnsafeCell;
