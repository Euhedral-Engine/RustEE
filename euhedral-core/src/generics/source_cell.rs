//! State that belongs to one lattice source.
//!
//! The lattice serializes a source's `pull` and `request`: one worker is inside either at a time,
//! and the interceptor's claim orders one worker's exit before the next worker's entry. State that
//! only those calls touch (and the actions [`LatticeReceiver::serialize`] runs in their place)
//! therefore needs no atomics and no lock. A [`SourceCell`] holds such state and says so in the
//! type: entering it is `unsafe`, because the compiler cannot see the lattice's guarantee.
//!
//! [`LatticeReceiver::serialize`]: super::LatticeReceiver::serialize

use std::cell::UnsafeCell;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
#[cfg(debug_assertions)]
use std::sync::atomic::{AtomicUsize, Ordering};

/// State that only a source's `pull` and `request` (and actions `serialize` runs) touch.
///
/// In debug builds [`enter`](Self::enter) panics if the cell is entered again before its guard
/// drops, from the same thread (a nested entry) or from another. Release builds carry no field
/// and make no check.
pub struct SourceCell<T> {
    value: UnsafeCell<T>,
    /// The token of the thread inside, or 0.
    #[cfg(debug_assertions)]
    inside: AtomicUsize,
}

// SAFETY: the value is reached only through `enter`, whose contract admits one thread at a time,
// so sharing the cell moves the `T` between threads without ever aliasing it.
unsafe impl<T: Send> Sync for SourceCell<T> {}

impl<T> SourceCell<T> {
    pub const fn new(value: T) -> Self {
        Self {
            value: UnsafeCell::new(value),
            #[cfg(debug_assertions)]
            inside: AtomicUsize::new(0),
        }
    }

    /// Takes the state for the duration of the guard.
    ///
    /// # Safety
    ///
    /// The caller is inside this source's `pull` or `request`, or in an action `serialize` runs,
    /// and holds no other guard of this cell.
    ///
    /// # Panics
    ///
    /// In debug builds, if the cell is already entered.
    #[track_caller]
    pub unsafe fn enter(&self) -> SourceGuard<'_, T> {
        #[cfg(debug_assertions)]
        {
            let token = thread_token();
            if let Err(holder) =
                self.inside
                    .compare_exchange(0, token, Ordering::AcqRel, Ordering::Acquire)
            {
                if holder == token {
                    panic!("SourceCell entered again on the same thread (a nested entry)");
                }
                panic!("SourceCell entered from another thread while one is inside");
            }
        }
        SourceGuard {
            cell: self,
            marker: PhantomData,
        }
    }

    /// The state, for the cell's exclusive owner.
    pub fn get_mut(&mut self) -> &mut T {
        self.value.get_mut()
    }

    /// The state, for the cell's owner.
    pub fn into_inner(self) -> T {
        self.value.into_inner()
    }
}

impl<T: Default> Default for SourceCell<T> {
    fn default() -> Self {
        Self::new(T::default())
    }
}

/// The state of a [`SourceCell`], borrowed until the guard drops.
///
/// Like a mutex guard, it is `Send` when `T` is and `Sync` only when `T` is: a shared reference to
/// the guard hands out `&T`, so it must not let two threads at a `!Sync` value such as a `Cell`.
///
/// ```compile_fail
/// use euhedral_core::generics::SourceCell;
/// use std::cell::Cell;
///
/// fn needs_sync<T: Sync>(_: &T) {}
///
/// let cell = SourceCell::new(Cell::new(0));
/// // SAFETY: a doctest on one thread.
/// let guard = unsafe { cell.enter() };
/// needs_sync(&guard); // a guard over a `Cell` is not `Sync`
/// ```
///
/// ```
/// use euhedral_core::generics::SourceCell;
///
/// fn needs_send_sync<T: Send + Sync>(_: &T) {}
///
/// let cell = SourceCell::new(0u64);
/// // SAFETY: a doctest on one thread.
/// let guard = unsafe { cell.enter() };
/// needs_send_sync(&guard);
/// ```
pub struct SourceGuard<'a, T> {
    cell: &'a SourceCell<T>,
    /// Keeps the auto traits off; the impls below say exactly when the guard is `Send` and `Sync`.
    marker: PhantomData<*const ()>,
}

// SAFETY: the guard is the cell's one access to a `T`, so moving it moves `&mut T`, which is
// `Send` exactly when `T` is.
unsafe impl<T: Send> Send for SourceGuard<'_, T> {}
// SAFETY: a shared reference to the guard gives `&T`, which is `Send` exactly when `T` is `Sync`.
unsafe impl<T: Sync> Sync for SourceGuard<'_, T> {}

impl<T> Deref for SourceGuard<'_, T> {
    type Target = T;

    fn deref(&self) -> &T {
        // SAFETY: `enter`'s contract makes this guard the only access to the value.
        unsafe { &*self.cell.value.get() }
    }
}

impl<T> DerefMut for SourceGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        // SAFETY: as in `deref`, and `&mut self` makes the borrow exclusive.
        unsafe { &mut *self.cell.value.get() }
    }
}

#[cfg(debug_assertions)]
impl<T> Drop for SourceGuard<'_, T> {
    fn drop(&mut self) {
        self.cell.inside.store(0, Ordering::Release);
    }
}

/// A non-zero value unique to the calling thread while it lives: the address of a thread-local.
#[cfg(debug_assertions)]
fn thread_token() -> usize {
    thread_local! {
        static TOKEN: u8 = const { 0 };
    }
    TOKEN.with(|token| std::ptr::from_ref(token) as usize)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(debug_assertions)]
    use crate::frames::Frame;
    #[cfg(debug_assertions)]
    use crate::generics::{LatticeReceiver, LatticeSource};
    #[cfg(debug_assertions)]
    use std::sync::Barrier;
    use std::sync::{Arc, Mutex};
    use std::thread;

    #[test]
    fn a_guard_reads_and_writes_the_value() {
        let cell = SourceCell::new(vec![1, 2]);
        {
            // SAFETY: the test is the only thread.
            let mut guard = unsafe { cell.enter() };
            guard.push(3);
            assert_eq!(guard.len(), 3);
        }
        // SAFETY: the test is the only thread, and the first guard dropped.
        assert_eq!(*unsafe { cell.enter() }, [1, 2, 3]);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "nested entry")]
    fn a_nested_enter_panics_in_debug() {
        let cell = SourceCell::new(0);
        // SAFETY: the test breaks the contract on purpose; the debug check is what it exercises.
        let _first = unsafe { cell.enter() };
        // SAFETY: as above.
        let _second = unsafe { cell.enter() };
    }

    #[cfg(debug_assertions)]
    #[test]
    fn a_second_thread_entering_while_one_is_inside_panics_in_debug() {
        let cell = Arc::new(SourceCell::new(0u32));
        let first = Arc::new(Barrier::new(2));
        let second = Arc::new(Barrier::new(2));
        let holder = {
            let (cell, first, second) = (cell.clone(), first.clone(), second.clone());
            thread::spawn(move || {
                // SAFETY: the only legitimate entry; the other thread's entry is the violation.
                let mut guard = unsafe { cell.enter() };
                *guard += 1;
                first.wait();
                second.wait();
            })
        };
        first.wait();
        let attempt = {
            let cell = cell.clone();
            thread::spawn(move || {
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    // SAFETY: the test breaks the contract on purpose.
                    drop(unsafe { cell.enter() });
                }))
            })
            .join()
            .unwrap()
        };
        let message = attempt.expect_err("the second entry is refused");
        let message = message
            .downcast_ref::<&str>()
            .map(|message| message.to_string())
            .or_else(|| message.downcast_ref::<String>().cloned())
            .unwrap();
        assert!(message.contains("from another thread"), "{message}");
        second.wait();
        holder.join().unwrap();

        // SAFETY: the holder has finished; the test is the only thread.
        let guard = unsafe { cell.enter() };
        assert_eq!(
            *guard, 1,
            "the refused entry left the state alone, and the check does not stick"
        );
    }

    /// A source whose `request` sits inside its state until the test lets it go.
    #[cfg(debug_assertions)]
    struct Blocking {
        state: SourceCell<u64>,
        inside: Barrier,
        release: Barrier,
    }

    #[cfg(debug_assertions)]
    impl LatticeSource for Blocking {
        fn add_downstream(self: Arc<Self>, _downstream: Arc<dyn LatticeReceiver>) {}

        fn pull(
            &self,
            _consumer: &mut dyn FnMut(Box<dyn Frame>),
            _stop: &mut dyn FnMut(&dyn Frame) -> bool,
            _demand: u64,
        ) -> u64 {
            0
        }

        fn request(&self, demand: u64) {
            // SAFETY: the lattice serializes `request`; the test below calls it directly from two
            // threads to see the debug check catch exactly that.
            let mut state = unsafe { self.state.enter() };
            *state += demand;
            self.inside.wait();
            self.release.wait();
        }

        fn complete(&self) {}

        fn is_complete(&self) -> bool {
            false
        }
    }

    #[cfg(debug_assertions)]
    #[test]
    fn a_direct_request_racing_the_lattice_is_caught_in_debug() {
        let source = Arc::new(Blocking {
            state: SourceCell::new(0),
            inside: Barrier::new(2),
            release: Barrier::new(2),
        });
        let lattice = {
            let source = source.clone();
            thread::spawn(move || source.request(1))
        };
        source.inside.wait();
        let direct = {
            let source = source.clone();
            thread::spawn(move || {
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| source.request(1)))
            })
            .join()
            .unwrap()
        };
        let message = direct.expect_err("a direct request while the lattice is inside panics");
        let message = message
            .downcast_ref::<&str>()
            .map(|message| message.to_string())
            .or_else(|| message.downcast_ref::<String>().cloned())
            .unwrap();
        assert!(message.contains("from another thread"), "{message}");
        source.release.wait();
        lattice.join().unwrap();
    }

    #[test]
    fn entries_in_turn_from_many_threads_are_accepted() {
        let cell = Arc::new(SourceCell::new(0u64));
        let turn = Arc::new(Mutex::new(()));
        let threads: Vec<_> = (0..8)
            .map(|_| {
                let (cell, turn) = (cell.clone(), turn.clone());
                thread::spawn(move || {
                    for _ in 0..1000 {
                        let _turn = turn.lock().unwrap();
                        // SAFETY: the mutex admits one thread at a time, as the lattice's claim does.
                        *unsafe { cell.enter() } += 1;
                    }
                })
            })
            .collect();
        for thread in threads {
            thread.join().unwrap();
        }
        // SAFETY: every thread has finished.
        assert_eq!(*unsafe { cell.enter() }, 8000);
    }

    #[test]
    fn get_mut_and_into_inner_need_no_entry() {
        let mut cell = SourceCell::new(String::from("a"));
        cell.get_mut().push('b');
        assert_eq!(cell.into_inner(), "ab");
    }
}
