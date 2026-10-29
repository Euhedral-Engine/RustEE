//! Locks for start, stop and rebalance, and the debug check that keeps them off the hot path.
//!
//! The hot path of the runtime takes no lock. Locks remain for lifecycle work only: start, stop (shutdown, drop,
//! complete, close) and rebalance (attaching and detaching sources and receivers, lattice resizing, shard moves).
//! [`LifecycleMutex`] and [`LifecycleRwLock`] are the lock types the workspace lints allow without a tag, and each
//! checks, in debug builds, that it is not taken inside a [`HotScope`]. A worker's cycle and a producer's offer run
//! inside one, so a lifecycle lock that a frame reaches ends the test run instead of costing a stall in production.
//!
//! A violation prints "lifecycle lock taken inside a hot scope" with the location of the lock call and aborts the
//! process. A panic would unwind into the frame runners, which catch panics and report them as a frame's failure, so
//! the violation could pass unseen. A test that provokes a violation on purpose installs a hook
//! ([`HotScope::install_violation_hook`], debug builds only); the hook receives the location, and the lock then
//! proceeds. A hook that is not meant for a location forwards it to [`HotScope::report_default`].
//!
//! Release builds carry none of it: the scope is a no-op and the locks are the standard ones. A poisoned lock is
//! used as it stands, as the runtime's other lifecycle state is.

// The wrappers are the one place a standard lock type is named.
#![allow(clippy::disallowed_types)]

use std::sync::{
    Mutex, MutexGuard, PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard, TryLockError,
};

#[cfg(debug_assertions)]
use std::cell::Cell;
#[cfg(debug_assertions)]
use std::io::Write;
use std::marker::PhantomData;
#[cfg(debug_assertions)]
use std::panic::Location;

#[cfg(debug_assertions)]
thread_local! {
    /// How many hot scopes the current thread is inside.
    static DEPTH: Cell<u32> = const { Cell::new(0) };
}

/// The installed violation hook, or `None` for the default (print and abort). Read only when a violation happens.
#[cfg(debug_assertions)]
static VIOLATION_HOOK: Mutex<Option<ViolationHook>> = Mutex::new(None);

/// What a debug build calls when a lifecycle lock is taken inside a hot scope, instead of aborting.
#[cfg(debug_assertions)]
pub type ViolationHook = fn(&'static Location<'static>);

/// Restores the previous violation hook when dropped.
#[cfg(debug_assertions)]
#[must_use = "the hook is removed when the guard drops"]
pub struct ViolationHookGuard {
    previous: Option<ViolationHook>,
}

#[cfg(debug_assertions)]
impl Drop for ViolationHookGuard {
    fn drop(&mut self) {
        *VIOLATION_HOOK
            .lock()
            .unwrap_or_else(PoisonError::into_inner) = self.previous;
    }
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
    #[inline]
    pub fn enter() -> HotGuard {
        #[cfg(debug_assertions)]
        DEPTH.with(|depth| depth.set(depth.get() + 1));
        HotGuard {
            _thread: PhantomData,
        }
    }

    /// Suspends every open hot scope of this thread for a lifecycle step.
    #[inline]
    pub fn leave() -> ColdGuard {
        ColdGuard {
            #[cfg(debug_assertions)]
            saved: DEPTH.with(|depth| depth.replace(0)),
            _thread: PhantomData,
        }
    }

    /// Whether this thread is inside a hot scope. Always `false` in a release build.
    #[inline]
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

    /// Installs `hook` to receive violations instead of the abort, until the returned guard drops (also when the
    /// test panics). The hook is process-wide: it receives the violations of every thread, so a hook keeps the
    /// locations it expects and passes the rest to [`report_default`](Self::report_default).
    #[cfg(debug_assertions)]
    pub fn install_violation_hook(hook: ViolationHook) -> ViolationHookGuard {
        let previous = VIOLATION_HOOK
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .replace(hook);
        ViolationHookGuard { previous }
    }

    /// The default report: prints the location straight to stderr (a test harness's output capture would swallow
    /// it, and the abort discards what the capture holds) and aborts the process.
    #[cfg(debug_assertions)]
    pub fn report_default(at: &'static Location<'static>) -> ! {
        let _ = writeln!(
            std::io::stderr(),
            "lifecycle lock taken inside a hot scope, at {at} (thread {:?})",
            std::thread::current().name()
        );
        std::process::abort();
    }

    /// Reports a lifecycle lock taken inside a hot scope: to the hook, or by printing and aborting.
    #[cfg(debug_assertions)]
    #[cold]
    #[track_caller]
    fn violation() {
        let at = Location::caller();
        let hook = *VIOLATION_HOOK
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        match hook {
            Some(hook) => hook(at),
            None => Self::report_default(at),
        }
    }

    /// Reports a violation when a lifecycle lock is taken inside a hot scope.
    #[inline]
    #[track_caller]
    fn check_lifecycle_lock() {
        #[cfg(debug_assertions)]
        if Self::active() {
            Self::violation();
        }
    }
}

impl Drop for HotGuard {
    #[inline]
    fn drop(&mut self) {
        #[cfg(debug_assertions)]
        DEPTH.with(|depth| {
            debug_assert!(depth.get() > 0, "a hot scope closed twice");
            depth.set(depth.get().saturating_sub(1));
        });
    }
}

impl Drop for ColdGuard {
    #[inline]
    fn drop(&mut self) {
        #[cfg(debug_assertions)]
        DEPTH.with(|depth| depth.set(self.saved));
    }
}

/// A mutex for lifecycle state. Poison is ignored, and in debug builds taking it inside a [`HotScope`] is a
/// violation.
pub struct LifecycleMutex<T>(Mutex<T>);

impl<T> LifecycleMutex<T> {
    /// A mutex holding `value`.
    pub const fn new(value: T) -> Self {
        Self(Mutex::new(value))
    }

    /// Locks, waiting for the holder. Debug builds report a violation inside a hot scope.
    #[track_caller]
    pub fn lock(&self) -> MutexGuard<'_, T> {
        HotScope::check_lifecycle_lock();
        self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Locks if the mutex is free. Debug builds report a violation inside a hot scope.
    #[track_caller]
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

/// A read-write lock for lifecycle state. Poison is ignored, and in debug builds taking it inside a [`HotScope`] is
/// a violation.
pub struct LifecycleRwLock<T>(RwLock<T>);

impl<T> LifecycleRwLock<T> {
    /// A lock holding `value`.
    pub const fn new(value: T) -> Self {
        Self(RwLock::new(value))
    }

    /// Locks for reading, waiting for a writer. Debug builds report a violation inside a hot scope.
    #[track_caller]
    pub fn read(&self) -> RwLockReadGuard<'_, T> {
        HotScope::check_lifecycle_lock();
        self.0.read().unwrap_or_else(PoisonError::into_inner)
    }

    /// Locks for writing, waiting for the holders. Debug builds report a violation inside a hot scope.
    #[track_caller]
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

    /// Violations a test provoked, and the places they were reported at.
    #[cfg(debug_assertions)]
    mod violations {
        use super::*;

        static PLACES: Mutex<Vec<String>> = Mutex::new(Vec::new());
        /// The hook is process-wide, so the tests that install it take turns.
        static TURN: Mutex<()> = Mutex::new(());

        fn places() -> std::sync::MutexGuard<'static, Vec<String>> {
            PLACES.lock().unwrap_or_else(PoisonError::into_inner)
        }

        /// Keeps the locations in this file; any other location is not a test's to swallow.
        fn record(at: &'static Location<'static>) {
            if at.file() == file!() {
                places().push(at.to_string());
            } else {
                HotScope::report_default(at);
            }
        }

        /// Runs `provoke` with the recorder installed and returns the places it was reported at.
        pub(super) fn recorded(provoke: impl FnOnce()) -> Vec<String> {
            let _turn = TURN.lock().unwrap_or_else(PoisonError::into_inner);
            places().clear();
            {
                let _hook = HotScope::install_violation_hook(record);
                provoke();
            }
            std::mem::take(&mut *places())
        }
    }

    #[cfg(debug_assertions)]
    #[test]
    fn a_lifecycle_lock_inside_a_hot_scope_reaches_the_hook_and_then_locks() {
        let mutex = LifecycleMutex::new(3);
        let places = violations::recorded(|| {
            let _hot = HotScope::enter();
            assert_eq!(*mutex.lock(), 3, "the lock proceeds after the report");
        });
        assert_eq!(places.len(), 1, "{places:?}");
        // The place is the lock call, in this file.
        assert!(places[0].contains("lifecycle.rs"), "{}", places[0]);
    }

    #[cfg(debug_assertions)]
    #[test]
    fn every_lock_call_reports_inside_a_hot_scope() {
        let mutex = LifecycleMutex::new(());
        let rw = LifecycleRwLock::new(());
        let places = violations::recorded(|| {
            let _hot = HotScope::enter();
            drop(rw.read());
            drop(rw.write());
            drop(mutex.try_lock());
        });
        assert_eq!(places.len(), 3, "{places:?}");
    }

    #[cfg(debug_assertions)]
    #[test]
    fn no_report_outside_a_hot_scope_or_while_left() {
        let mutex = LifecycleMutex::new(());
        let places = violations::recorded(|| {
            drop(mutex.lock());
            let _hot = HotScope::enter();
            let _cold = HotScope::leave();
            drop(mutex.lock());
        });
        assert!(places.is_empty(), "{places:?}");
    }

    #[cfg(debug_assertions)]
    #[test]
    fn the_hook_is_restored_when_its_guard_drops_even_in_a_panic() {
        let mutex = LifecycleMutex::new(());
        let places = violations::recorded(|| {
            // An inner hook for a test that fails: its guard drops while the panic unwinds.
            fn other(_: &'static Location<'static>) {}
            let unwound = std::panic::catch_unwind(|| {
                let _inner = HotScope::install_violation_hook(other);
                panic!("a test fails while its hook is installed");
            });
            assert!(unwound.is_err());
            // The recorder is back: this violation reaches it.
            let _hot = HotScope::enter();
            drop(mutex.lock());
        });
        assert_eq!(places.len(), 1, "{places:?}");
    }

    /// The default report cannot be caught: the frame runners catch panics, so a panic could pass unseen.
    #[cfg(all(debug_assertions, unix, not(miri)))]
    #[test]
    fn the_default_report_aborts_even_inside_catch_unwind() {
        use std::os::unix::process::ExitStatusExt;
        const CHILD: &str = "EUHEDRAL_LIFECYCLE_ABORT_CHILD";
        if std::env::var_os(CHILD).is_some() {
            let mutex = LifecycleMutex::new(());
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _hot = HotScope::enter();
                drop(mutex.lock());
            }));
            // Reached only when the report was caught.
            std::process::exit(0);
        }
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            // Under the harness's output capture, as an ordinary `cargo test` runs it.
            .args([
                "--exact",
                "sync::lifecycle::tests::the_default_report_aborts_even_inside_catch_unwind",
            ])
            .env(CHILD, "1")
            .output()
            .unwrap();
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert_eq!(output.status.signal(), Some(6), "{stderr}");
        assert!(
            stderr.contains("lifecycle lock taken inside a hot scope, at ")
                && stderr.contains("lifecycle.rs")
                && stderr.contains("(thread "),
            "{stderr}"
        );
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
