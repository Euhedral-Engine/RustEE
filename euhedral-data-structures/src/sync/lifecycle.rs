//! Locks for start, stop and rebalance, and the debug check that keeps them off the hot path.
//!
//! The hot path of the runtime takes no lock. Locks remain for lifecycle work only: start, stop (shutdown, drop,
//! complete, close) and rebalance (attaching and detaching sources and receivers, lattice resizing, shard moves).
//! [`LifecycleMutex`] and [`LifecycleRwLock`] are the only lock types the workspace lints allow, and each checks, in
//! debug builds, that it is not taken inside a [`HotScope`]. A worker's cycle and a producer's offer run inside one,
//! so a lifecycle lock that a frame reaches panics in the tests instead of costing a stall in production.
//!
//! Release builds carry none of it: the scope is a no-op and the locks are the standard ones. A poisoned lock is
//! used as it stands, as the runtime's other lifecycle state is.

// The wrappers are the one place a standard lock type is named.
#![allow(clippy::disallowed_types)]

use std::sync::{Mutex, MutexGuard, PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard, TryLockError};

#[cfg(debug_assertions)]
use std::cell::Cell;
use std::marker::PhantomData;

#[cfg(debug_assertions)]
thread_local! {
    /// How many hot scopes the current thread is inside.
    static DEPTH: Cell<u32> = const { Cell::new(0) };
}

/// Marks the code a thread runs per frame, per pull, per request, per offer or per completion.
///
/// [`enter`](Self::enter) opens a scope for the guard's lifetime; scopes nest. A lifecycle step that a hot path
/// legitimately reaches (the first attach of a lake, the retirement of a routing generation) calls
/// [`leave`](Self::leave) first, which suspends the scopes until its guard drops. Every method does nothing in a
/// release build.
pub struct HotScope;

/// Keeps one hot scope open; dropping it closes the scope.
#[must_use = "the scope closes when the guard drops"]
pub struct HotGuard {
    // The depth is the current thread's: the guard stays on it.
    _thread: PhantomData<*const ()>,
}

/// Suspends the hot scopes of the current thread; dropping it restores them.
#[must_use = "the scopes return when the guard drops"]
pub struct ColdGuard {
    #[cfg(debug_assertions)]
    saved: u32,
    // The depth is the current thread's: the guard stays on it.
    _thread: PhantomData<*const ()>,
}

impl HotScope {
    /// Opens a hot scope on this thread.
    pub fn enter() -> HotGuard {
        #[cfg(debug_assertions)]
        DEPTH.with(|depth| depth.set(depth.get() + 1));
        HotGuard { _thread: PhantomData }
    }

    /// Suspends every open hot scope of this thread for a lifecycle step.
    pub fn leave() -> ColdGuard {
        ColdGuard {
            #[cfg(debug_assertions)]
            saved: DEPTH.with(|depth| depth.replace(0)),
            _thread: PhantomData,
        }
    }

    /// Whether this thread is inside a hot scope. Always `false` in a release build.
    pub fn active() -> bool {
        #[cfg(debug_assertions)]
        {
            DEPTH.with(|depth| depth.get() > 0)
        }
        #[cfg(not(debug_assertions))]
        {
            false
        }
    }

    /// Panics when a lifecycle lock is taken inside a hot scope.
    #[inline]
    fn check_lifecycle_lock() {
        #[cfg(debug_assertions)]
        if Self::active() {
            panic!("lifecycle lock taken inside a hot scope");
        }
    }
}

impl Drop for HotGuard {
    fn drop(&mut self) {
        #[cfg(debug_assertions)]
        DEPTH.with(|depth| depth.set(depth.get().saturating_sub(1)));
    }
}

impl Drop for ColdGuard {
    fn drop(&mut self) {
        #[cfg(debug_assertions)]
        DEPTH.with(|depth| depth.set(self.saved));
    }
}

/// A mutex for lifecycle state. Poison is ignored, and in debug builds taking it inside a [`HotScope`] panics.
pub struct LifecycleMutex<T>(Mutex<T>);

impl<T> LifecycleMutex<T> {
    /// A mutex holding `value`.
    pub const fn new(value: T) -> Self {
        Self(Mutex::new(value))
    }

    /// Locks, waiting for the holder. Debug builds panic inside a hot scope.
    pub fn lock(&self) -> MutexGuard<'_, T> {
        HotScope::check_lifecycle_lock();
        self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Locks if the mutex is free. Debug builds panic inside a hot scope.
    pub fn try_lock(&self) -> Option<MutexGuard<'_, T>> {
        HotScope::check_lifecycle_lock();
        match self.0.try_lock() {
            Ok(guard) => Some(guard),
            Err(TryLockError::Poisoned(poisoned)) => Some(poisoned.into_inner()),
            Err(TryLockError::WouldBlock) => None,
        }
    }

    /// The value, for the mutex's exclusive owner.
    pub fn get_mut(&mut self) -> &mut T {
        self.0.get_mut().unwrap_or_else(PoisonError::into_inner)
    }

    /// Takes the value out of the mutex.
    pub fn into_inner(self) -> T {
        self.0.into_inner().unwrap_or_else(PoisonError::into_inner)
    }
}

impl<T: Default> Default for LifecycleMutex<T> {
    fn default() -> Self {
        Self::new(T::default())
    }
}

impl<T: std::fmt::Debug> std::fmt::Debug for LifecycleMutex<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

/// A read-write lock for lifecycle state. Poison is ignored, and in debug builds taking it inside a [`HotScope`]
/// panics.
pub struct LifecycleRwLock<T>(RwLock<T>);

impl<T> LifecycleRwLock<T> {
    /// A lock holding `value`.
    pub const fn new(value: T) -> Self {
        Self(RwLock::new(value))
    }

    /// Locks for reading, waiting for a writer. Debug builds panic inside a hot scope.
    pub fn read(&self) -> RwLockReadGuard<'_, T> {
        HotScope::check_lifecycle_lock();
        self.0.read().unwrap_or_else(PoisonError::into_inner)
    }

    /// Locks for writing, waiting for the holders. Debug builds panic inside a hot scope.
    pub fn write(&self) -> RwLockWriteGuard<'_, T> {
        HotScope::check_lifecycle_lock();
        self.0.write().unwrap_or_else(PoisonError::into_inner)
    }

    /// The value, for the lock's exclusive owner.
    pub fn get_mut(&mut self) -> &mut T {
        self.0.get_mut().unwrap_or_else(PoisonError::into_inner)
    }

    /// Takes the value out of the lock.
    pub fn into_inner(self) -> T {
        self.0.into_inner().unwrap_or_else(PoisonError::into_inner)
    }
}

impl<T: Default> Default for LifecycleRwLock<T> {
    fn default() -> Self {
        Self::new(T::default())
    }
}

impl<T: std::fmt::Debug> std::fmt::Debug for LifecycleRwLock<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

#[cfg(all(test, not(loom)))]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn a_lifecycle_lock_outside_a_hot_scope_locks() {
        let mutex = LifecycleMutex::new(1);
        *mutex.lock() += 1;
        assert_eq!(*mutex.lock(), 2);
        let rw = LifecycleRwLock::new(5);
        *rw.write() += 1;
        assert_eq!(*rw.read(), 6);
        assert!(!HotScope::active());
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "inside a hot scope")]
    fn a_lifecycle_lock_inside_a_hot_scope_panics() {
        let mutex = LifecycleMutex::new(());
        let _hot = HotScope::enter();
        drop(mutex.lock());
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "inside a hot scope")]
    fn a_lifecycle_read_write_lock_inside_a_hot_scope_panics() {
        let lock = LifecycleRwLock::new(());
        let _hot = HotScope::enter();
        drop(lock.read());
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "inside a hot scope")]
    fn a_try_lock_inside_a_hot_scope_panics() {
        let mutex = LifecycleMutex::new(());
        let _hot = HotScope::enter();
        drop(mutex.try_lock());
    }

    #[test]
    fn leaving_the_hot_scope_allows_the_lock_and_restores_the_depth() {
        let mutex = LifecycleMutex::new(0);
        let outer = HotScope::enter();
        let inner = HotScope::enter();
        {
            let _cold = HotScope::leave();
            assert!(!HotScope::active());
            *mutex.lock() += 1;
        }
        // Release builds never report a scope.
        assert_eq!(HotScope::active(), cfg!(debug_assertions));
        drop(inner);
        assert_eq!(HotScope::active(), cfg!(debug_assertions));
        drop(outer);
        assert!(!HotScope::active());
        assert_eq!(*mutex.lock(), 1);
    }

    #[test]
    fn a_scope_belongs_to_its_thread() {
        let _hot = HotScope::enter();
        let other = std::thread::spawn(HotScope::active);
        assert!(!other.join().unwrap());
    }

    #[test]
    fn a_poisoned_lifecycle_lock_still_locks() {
        let mutex = Arc::new(LifecycleMutex::new(7));
        let rw = Arc::new(LifecycleRwLock::new(8));
        let (m, r) = (Arc::clone(&mutex), Arc::clone(&rw));
        let _ = std::thread::spawn(move || {
            let _m = m.lock();
            let _r = r.write();
            panic!("poison both");
        })
        .join();
        assert_eq!(*mutex.lock(), 7);
        assert_eq!(*rw.read(), 8);
        assert!(mutex.try_lock().is_some());
    }

    #[test]
    fn try_lock_reports_a_held_mutex() {
        let mutex = LifecycleMutex::new(());
        let held = mutex.lock();
        assert!(mutex.try_lock().is_none());
        drop(held);
        assert!(mutex.try_lock().is_some());
    }
}
