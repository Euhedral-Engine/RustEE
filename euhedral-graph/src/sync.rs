//! The atomics and cells the crate's lock-free pieces are built from. A build with `--cfg loom` swaps
//! them for the `loom` model checker's, which explores every interleaving the memory model allows.

#![allow(
    unsafe_code,
    reason = "`Owned` hands plain state from owner to owner under an ordering its callers state"
)]

#[cfg(loom)]
pub(crate) use loom::sync::atomic::{
    AtomicBool, AtomicPtr, AtomicU8, AtomicU64, AtomicUsize, Ordering,
};
#[cfg(not(loom))]
pub(crate) use std::sync::atomic::{
    AtomicBool, AtomicPtr, AtomicU8, AtomicU64, AtomicUsize, Ordering,
};

/// An `UnsafeCell` whose accesses are closures over a raw pointer, the shape `loom`'s cell has, so the
/// model checker sees each access and can report an unordered one. The checker tracks the closure's
/// duration only: a reference made from the pointer and kept after `f` returns is not checked, so code under
/// a model reads through an access that ends inside the closure (an owned `take`, a copy).
#[cfg(not(loom))]
#[derive(Debug)]
pub(crate) struct UnsafeCell<T>(std::cell::UnsafeCell<T>);

#[cfg(not(loom))]
impl<T> UnsafeCell<T> {
    pub(crate) const fn new(value: T) -> Self {
        Self(std::cell::UnsafeCell::new(value))
    }

    /// Runs `f` with a pointer to the value, for a read.
    #[inline]
    pub(crate) fn with<R>(&self, f: impl FnOnce(*const T) -> R) -> R {
        f(self.0.get())
    }

    /// Runs `f` with a pointer to the value, for a write.
    #[inline]
    pub(crate) fn with_mut<R>(&self, f: impl FnOnce(*mut T) -> R) -> R {
        f(self.0.get())
    }
}

#[cfg(loom)]
pub(crate) use loom::cell::UnsafeCell;

/// State that one thread or one stage owns at a time, handed from owner to owner by an ordering the code around it
/// establishes (a count, a queue, a closed stack). It is plain: no atomic, no lock. Every access is `unsafe`, and
/// states what ordered it.
///
/// The cell is `Sync` so that it can sit in state the threads share; the ordering that hands it over is what makes
/// each access sound.
pub(crate) struct Owned<T>(UnsafeCell<T>);

// SAFETY: the cell hands its value to one thread at a time through accesses whose safety contracts name the ordering
// that does so; the value moves between threads, so it must be `Send`.
unsafe impl<T: Send> Sync for Owned<T> {}
// SAFETY: the value is sent with the cell.
unsafe impl<T: Send> Send for Owned<T> {}

impl<T: std::fmt::Debug> std::fmt::Debug for Owned<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Owned").finish_non_exhaustive()
    }
}

impl<T> Owned<T> {
    pub(crate) fn new(value: T) -> Self {
        Self(UnsafeCell::new(value))
    }

    /// Reads the value with `f`.
    ///
    /// # Safety
    ///
    /// No thread writes the value while `f` runs, and every write that precedes it in the value's hand-over order
    /// happens before this call.
    #[inline]
    pub(crate) unsafe fn read<R>(&self, f: impl FnOnce(&T) -> R) -> R {
        // SAFETY: the caller's contract: nothing writes the value meanwhile.
        self.0.with(|p| f(unsafe { &*p }))
    }

    /// Changes the value with `f`.
    ///
    /// # Safety
    ///
    /// The calling thread owns the value: no other thread reads or writes it while `f` runs, and every access that
    /// precedes it in the value's hand-over order happens before this call.
    #[inline]
    pub(crate) unsafe fn write<R>(&self, f: impl FnOnce(&mut T) -> R) -> R {
        // SAFETY: the caller's contract: the calling thread is the only one to touch the value.
        self.0.with_mut(|p| f(unsafe { &mut *p }))
    }
}

impl<T: Copy> Owned<T> {
    /// A copy of the value.
    ///
    /// # Safety
    ///
    /// As [`read`](Self::read).
    #[inline]
    pub(crate) unsafe fn get(&self) -> T {
        // SAFETY: the caller's contract.
        unsafe { self.read(|v| *v) }
    }

    /// Replaces the value.
    ///
    /// # Safety
    ///
    /// As [`write`](Self::write).
    #[inline]
    pub(crate) unsafe fn set(&self, value: T) {
        // SAFETY: the caller's contract.
        unsafe { self.write(|v| *v = value) }
    }
}
