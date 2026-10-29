//! The atomics and cells the crate's lock-free pieces are built from. A build with `--cfg loom` swaps
//! them for the `loom` model checker's, which explores every interleaving the memory model allows.

#[cfg(loom)]
pub(crate) use loom::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
#[cfg(not(loom))]
pub(crate) use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// An `UnsafeCell` whose accesses are closures over a raw pointer, the shape `loom`'s cell has, so the
/// model checker sees each access and can report an unordered one.
#[cfg(not(loom))]
#[derive(Debug)]
pub(crate) struct UnsafeCell<T>(std::cell::UnsafeCell<T>);

#[cfg(not(loom))]
impl<T> UnsafeCell<T> {
    pub(crate) const fn new(value: T) -> Self {
        Self(std::cell::UnsafeCell::new(value))
    }

    /// Runs `f` with a pointer to the value.
    #[inline]
    pub(crate) fn with_mut<R>(&self, f: impl FnOnce(*mut T) -> R) -> R {
        f(self.0.get())
    }
}

#[cfg(loom)]
pub(crate) use loom::cell::UnsafeCell;
