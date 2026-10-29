//! State that belongs to one lattice source.
//!
//! The lattice serializes a source's `pull` and `request`: one worker is inside either at a time,
//! and the interceptor's claim orders one worker's exit before the next worker's entry. State that
//! only those calls touch (and the actions [`LatticeReceiver::serialize`] runs in their place)
//! therefore needs no atomics and no lock. A source's lifecycle calls, which may arrive from any
//! thread, set atomics instead, and the source applies them to its cell in its next `request` or
//! `pull`.
//!
//! A [`SourceCell`] holds such state and checks the guarantee itself, in every build: entering takes
//! a claim with one compare-and-set (uncontended under the lattice), so a nested entry, or a direct
//! call that races the lattice, can never produce two live `&mut` to the state. The source decides
//! what a refused entry means: [`try_enter`](SourceCell::try_enter) hands the refusal back, as the
//! ingest sinks' `draining` flag does; [`enter`](SourceCell::enter) panics.
//!
//! [`LatticeReceiver::serialize`]: super::LatticeReceiver::serialize

use std::cell::UnsafeCell;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicUsize, Ordering};

/// State that only a source's `pull` and `request` (and actions `serialize` runs) touch.
///
/// Entry takes a claim, a compare-and-set of the entering thread's token into `inside` (`SeqCst`),
/// and the guard's drop gives it back (`SeqCst` store of 0). The state is reached only through a
/// guard, so at most one `&mut T` exists at a time in every build.
pub struct SourceCell<T> {
    value: UnsafeCell<T>,
    /// The token of the thread inside, or 0.
    inside: AtomicUsize,
}

// SAFETY: the value is reached only through a guard, and the claim admits one guard at a time, so
// sharing the cell moves the `T` between threads without ever aliasing it.
unsafe impl<T: Send> Sync for SourceCell<T> {}

impl<T> SourceCell<T> {
    pub const fn new(value: T) -> Self {
        Self {
            value: UnsafeCell::new(value),
            inside: AtomicUsize::new(0),
        }
    }

    /// Takes the state for the duration of the guard, or hands back the refusal: `None` when the
    /// cell is already entered, from this thread (a nested entry) or from another. Never waits.
    pub fn try_enter(&self) -> Option<SourceGuard<'_, T>> {
        self.claim().ok()
    }

    /// Takes the state for the duration of the guard.
    ///
    /// # Panics
    ///
    /// If the cell is already entered: "again on the same thread (a nested entry)" or "from
    /// another thread while one is inside". The check is made in every build.
    #[track_caller]
    pub fn enter(&self) -> SourceGuard<'_, T> {
        match self.claim() {
            Ok(guard) => guard,
            Err(true) => panic!("SourceCell entered again on the same thread (a nested entry)"),
            Err(false) => panic!("SourceCell entered from another thread while one is inside"),
        }
    }

    /// The claim: `Err(true)` when this thread holds it, `Err(false)` when another does.
    fn claim(&self) -> Result<SourceGuard<'_, T>, bool> {
        let token = thread_token();
        match self
            .inside
            .compare_exchange(0, token, Ordering::SeqCst, Ordering::SeqCst)
        {
            Ok(_) => Ok(SourceGuard {
                cell: self,
                marker: PhantomData,
            }),
            Err(holder) => Err(holder == token),
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
/// let guard = cell.enter();
/// needs_sync(&guard); // a guard over a `Cell` is not `Sync`
/// ```
///
/// ```
/// use euhedral_core::generics::SourceCell;
///
/// fn needs_send_sync<T: Send + Sync>(_: &T) {}
///
/// let cell = SourceCell::new(0u64);
/// let guard = cell.enter();
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
        // SAFETY: the guard holds the cell's claim, so it is the only access to the value.
        unsafe { &*self.cell.value.get() }
    }
}

impl<T> DerefMut for SourceGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        // SAFETY: as in `deref`, and `&mut self` makes the borrow exclusive.
        unsafe { &mut *self.cell.value.get() }
    }
}

impl<T> Drop for SourceGuard<'_, T> {
    fn drop(&mut self) {
        self.cell.inside.store(0, Ordering::SeqCst);
    }
}

/// A non-zero value unique to the calling thread while it lives: the address of a thread-local.
fn thread_token() -> usize {
    thread_local! {
        static TOKEN: u8 = const { 0 };
    }
    TOKEN.with(|token| std::ptr::from_ref(token) as usize)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frames::Frame;
    use crate::generics::{LatticeReceiver, LatticeSource};
    use std::sync::Barrier;
    use std::sync::{Arc, Mutex};
    use std::thread;

    #[test]
    fn a_guard_reads_and_writes_the_value() {
        let cell = SourceCell::new(vec![1, 2]);
        {
            let mut guard = cell.enter();
            guard.push(3);
            assert_eq!(guard.len(), 3);
        }
        assert_eq!(*cell.enter(), [1, 2, 3]);
    }

    #[test]
    #[should_panic(expected = "nested entry")]
    fn a_nested_enter_panics_in_every_build() {
        let cell = SourceCell::new(0);
        let _first = cell.enter();
        let _second = cell.enter();
    }

    #[test]
    fn a_second_thread_entering_while_one_is_inside_panics_in_every_build() {
        let cell = Arc::new(SourceCell::new(0u32));
        let first = Arc::new(Barrier::new(2));
        let second = Arc::new(Barrier::new(2));
        let holder = {
            let (cell, first, second) = (cell.clone(), first.clone(), second.clone());
            thread::spawn(move || {
                let mut guard = cell.enter();
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
                    drop(cell.enter());
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
        let guard = cell.enter();
        assert_eq!(
            *guard, 1,
            "the refused entry left the state alone, and the check does not stick"
        );
    }

    /// A source whose `request` sits inside its state until the test lets it go.
    struct Blocking {
        state: SourceCell<u64>,
        inside: Barrier,
        release: Barrier,
    }

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
            let mut state = self.state.enter();
            *state += demand;
            self.inside.wait();
            self.release.wait();
        }

        fn complete(&self) {}

        fn is_complete(&self) -> bool {
            false
        }
    }

    #[test]
    fn a_direct_request_racing_the_lattice_is_refused_in_every_build() {
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
                        *cell.enter() += 1;
                    }
                })
            })
            .collect();
        for thread in threads {
            thread.join().unwrap();
        }
        assert_eq!(*cell.enter(), 8000);
    }

    #[test]
    fn get_mut_and_into_inner_need_no_entry() {
        let mut cell = SourceCell::new(String::from("a"));
        cell.get_mut().push('b');
        assert_eq!(cell.into_inner(), "ab");
    }

    #[test]
    fn try_enter_hands_back_the_refusal_and_the_claim_frees_on_drop() {
        let cell = SourceCell::new(5u32);
        let first = cell.try_enter().expect("free");
        assert!(
            cell.try_enter().is_none(),
            "nested: refused, never two guards"
        );
        drop(first);
        assert_eq!(*cell.try_enter().expect("freed by the drop"), 5);
    }

    #[test]
    fn a_racing_try_enter_never_gives_two_threads_the_state() {
        let cell = Arc::new(SourceCell::new(0u64));
        let threads: Vec<_> = (0..8)
            .map(|_| {
                let cell = cell.clone();
                thread::spawn(move || {
                    let mut entered = 0u64;
                    for _ in 0..20_000 {
                        if let Some(mut guard) = cell.try_enter() {
                            // A plain read-modify-write: a second guard would lose updates.
                            let v = *guard;
                            *guard = v + 1;
                            entered += 1;
                        }
                    }
                    entered
                })
            })
            .collect();
        let total: u64 = threads.into_iter().map(|t| t.join().unwrap()).sum();
        assert_eq!(*cell.enter(), total);
    }
}
