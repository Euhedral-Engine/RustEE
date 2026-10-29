//! A one-shot hand-off from a frame to the main thread.
//!
//! The main thread of a program starts the lattice and waits for it; it is the only thread allowed
//! to block. [`Latch::new`] makes the two ends: a [`LatchSetter`] that a frame (or anything else
//! on the lattice) moves in and sets, and a [`LatchWaiter`] that the main thread waits on.
//!
//! The wait parks the thread; it is the single blocking call of the crate and belongs in `main`
//! (or a test) only. Nothing on the lattice waits.

use std::fmt;
#[allow(clippy::disallowed_types, reason = "HOT-FIX(G5)")]
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::thread::{self, Thread};

/// What the two ends share.
struct State<T> {
    value: Option<T>,
    /// The setter was dropped without setting: the wait can never end.
    abandoned: bool,
    /// The thread parked in [`LatchWaiter::wait`], if one is.
    waiter: Option<Thread>,
}

/// A value that is set once, by one end, and waited for, by the other.
///
/// The type only makes the ends: `Latch::<T>::new()`.
pub struct Latch<T> {
    #[allow(clippy::disallowed_types, reason = "HOT-FIX(G5)")]
    state: Mutex<State<T>>,
}

impl<T> fmt::Debug for Latch<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Latch").finish_non_exhaustive()
    }
}

impl<T> Latch<T> {
    /// A latch with nothing set: the end that sets it and the end that waits for it.
    // A constructor of its two ends, not of a `Latch`: the shared state is private.
    #[allow(clippy::new_ret_no_self)]
    #[allow(clippy::disallowed_types, reason = "HOT-FIX(G5)")]
    pub fn new() -> (LatchSetter<T>, LatchWaiter<T>) {
        let latch = Arc::new(Self {
            state: Mutex::new(State {
                value: None,
                abandoned: false,
                waiter: None,
            }),
        });
        (
            LatchSetter {
                latch: Some(latch.clone()),
            },
            LatchWaiter { latch },
        )
    }

    fn lock(&self) -> MutexGuard<'_, State<T>> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// The end that sets the latch. Move it into a frame; setting wakes the waiter.
pub struct LatchSetter<T> {
    /// `None` once set.
    latch: Option<Arc<Latch<T>>>,
}

impl<T> fmt::Debug for LatchSetter<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LatchSetter").finish_non_exhaustive()
    }
}

impl<T> LatchSetter<T> {
    /// Sets the value and wakes the waiter, which may be waiting already or come later. Never
    /// blocks.
    pub fn set(mut self, value: T) {
        let latch = self.latch.take().expect("a setter sets once");
        let waiter = {
            let mut state = latch.lock();
            state.value = Some(value);
            state.waiter.take()
        };
        if let Some(waiter) = waiter {
            waiter.unpark();
        }
    }
}

impl<T> Drop for LatchSetter<T> {
    /// A setter dropped without setting (its frame was dropped, say) ends the wait with a panic
    /// instead of leaving the main thread parked forever.
    fn drop(&mut self) {
        let Some(latch) = self.latch.take() else {
            return;
        };
        let waiter = {
            let mut state = latch.lock();
            state.abandoned = true;
            state.waiter.take()
        };
        if let Some(waiter) = waiter {
            waiter.unpark();
        }
    }
}

/// The end that waits for the latch. Main thread only.
pub struct LatchWaiter<T> {
    latch: Arc<Latch<T>>,
}

impl<T> fmt::Debug for LatchWaiter<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LatchWaiter").finish_non_exhaustive()
    }
}

impl<T> LatchWaiter<T> {
    /// Parks the calling thread until the value is set, and returns it. It returns at once if the
    /// value was set before.
    ///
    /// This is the one call in the crate that blocks, and only the main thread (or a test) may
    /// make it: a lattice worker that waited would be a worker lost.
    ///
    /// # Panics
    ///
    /// If the setter was dropped without setting.
    #[allow(
        clippy::disallowed_methods,
        reason = "the main thread has nothing to run while the lattice works: this is where it waits"
    )]
    pub fn wait(self) -> T {
        loop {
            {
                let mut state = self.latch.lock();
                if let Some(value) = state.value.take() {
                    return value;
                }
                assert!(
                    !state.abandoned,
                    "the latch's setter was dropped without setting"
                );
                state.waiter = Some(thread::current());
            }
            // A wake-up that arrives before this call leaves the park token set, so the park
            // returns at once; a spurious return loops back to the check.
            thread::park();
        }
    }
}
