use super::loom::{AtomicPtr, Ordering};
use std::fmt;
use std::ptr;

/// A cell that holds a value installed at most once.
///
/// [`set`](Self::set) is a compare-and-set from empty to the value (sequentially consistent, as a
/// volatile compare-and-set is), and [`get`](Self::get) is an acquire load, so a reader that sees
/// the value sees everything its installer wrote before installing it. The value is boxed and lives
/// until the cell drops: it is never replaced or taken while the cell is shared, so a reference
/// from [`get`](Self::get) stays valid for as long as the cell is borrowed.
pub struct SetOnce<T> {
    value: AtomicPtr<T>,
}

impl<T> SetOnce<T> {
    /// An empty cell.
    #[cfg(not(loom))]
    pub const fn new() -> Self {
        Self {
            value: AtomicPtr::new(ptr::null_mut()),
        }
    }

    /// An empty cell.
    #[cfg(loom)]
    pub fn new() -> Self {
        Self {
            value: AtomicPtr::new(ptr::null_mut()),
        }
    }

    /// Installs `value` unless one is already set, in which case `value` is handed back.
    pub fn set(&self, value: T) -> Result<(), T> {
        let raw = Box::into_raw(Box::new(value));
        match self
            .value
            .compare_exchange(ptr::null_mut(), raw, Ordering::SeqCst, Ordering::SeqCst)
        {
            Ok(_) => Ok(()),
            Err(_) => {
                // SAFETY: `raw` came from `Box::into_raw` just above and was not published, so this
                // thread still owns it.
                Err(*unsafe { Box::from_raw(raw) })
            }
        }
    }

    /// The pointer, for the cell's exclusive owner; `replace` is written in its place.
    fn swap_exclusive(&mut self, replace: *mut T) -> *mut T {
        #[cfg(not(loom))]
        {
            std::mem::replace(self.value.get_mut(), replace)
        }
        #[cfg(loom)]
        {
            self.value.with_mut(|slot| std::mem::replace(slot, replace))
        }
    }

    /// The value, if one is set.
    pub fn get(&self) -> Option<&T> {
        let raw = self.value.load(Ordering::Acquire);
        // SAFETY: a non-null pointer was installed by `set` from a `Box`, is never replaced, and is
        // freed only when the cell drops, which the borrow of `self` rules out.
        unsafe { raw.as_ref() }
    }

    /// The value, if one is set, for the cell's exclusive owner.
    pub fn get_mut(&mut self) -> Option<&mut T> {
        let raw = self.swap_exclusive(ptr::null_mut());
        // Put the pointer back: the swap above only read it.
        self.swap_exclusive(raw);
        // SAFETY: as in `get`; `&mut self` makes the access exclusive.
        unsafe { raw.as_mut() }
    }

    /// Takes the value out of the cell.
    pub fn into_inner(mut self) -> Option<T> {
        let raw = self.swap_exclusive(ptr::null_mut());
        // SAFETY: a non-null pointer came from `Box::into_raw` in `set`; clearing it above leaves
        // `drop` nothing to free.
        (!raw.is_null()).then(|| *unsafe { Box::from_raw(raw) })
    }
}

impl<T> Default for SetOnce<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Drop for SetOnce<T> {
    fn drop(&mut self) {
        let raw = self.swap_exclusive(ptr::null_mut());
        if !raw.is_null() {
            // SAFETY: a non-null pointer came from `Box::into_raw` in `set` and nothing else frees
            // it; `&mut self` means no reference from `get` is alive.
            drop(unsafe { Box::from_raw(raw) });
        }
    }
}

// SAFETY: the cell owns a `T` and moves it between threads only whole, at `set` and `into_inner`.
unsafe impl<T: Send> Send for SetOnce<T> {}
// SAFETY: `set` moves a `T` in from any thread (`Send`) and `get` shares `&T` with every thread
// (`Sync`).
unsafe impl<T: Send + Sync> Sync for SetOnce<T> {}

impl<T> fmt::Debug for SetOnce<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(if self.get().is_some() {
            "SetOnce(Some(..))"
        } else {
            "SetOnce(None)"
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering as StdOrdering};
    use std::sync::{Arc, Barrier};
    use std::thread;

    struct Counted(Arc<AtomicUsize>, u32);

    impl Drop for Counted {
        fn drop(&mut self) {
            self.0.fetch_add(1, StdOrdering::SeqCst);
        }
    }

    #[test]
    fn set_hands_the_value_back_when_one_is_set() {
        let cell = SetOnce::new();
        assert_eq!(cell.set(1), Ok(()));
        assert_eq!(cell.set(2), Err(2));
        assert_eq!(cell.get(), Some(&1));
    }

    #[test]
    fn get_is_none_until_set() {
        let cell = SetOnce::<u32>::default();
        assert_eq!(cell.get(), None);
        assert_eq!(format!("{cell:?}"), "SetOnce(None)");
        cell.set(5).unwrap();
        assert_eq!(cell.get(), Some(&5));
        assert_eq!(format!("{cell:?}"), "SetOnce(Some(..))");
    }

    #[test]
    fn a_set_value_is_dropped_exactly_once() {
        let drops = Arc::new(AtomicUsize::new(0));
        let cell = SetOnce::new();
        assert!(cell.set(Counted(drops.clone(), 1)).is_ok());
        let refused = cell.set(Counted(drops.clone(), 2)).unwrap_err();
        assert_eq!(refused.1, 2);
        assert_eq!(
            drops.load(StdOrdering::SeqCst),
            0,
            "the refused value is the caller's"
        );
        drop(refused);
        assert_eq!(drops.load(StdOrdering::SeqCst), 1);
        drop(cell);
        assert_eq!(drops.load(StdOrdering::SeqCst), 2);

        let empty = SetOnce::<Counted>::new();
        drop(empty);
        assert_eq!(drops.load(StdOrdering::SeqCst), 2);
    }

    #[test]
    fn racing_setters_install_exactly_one() {
        const THREADS: usize = 8;
        const ROUNDS: usize = 1000;
        let cells: Arc<Vec<SetOnce<usize>>> =
            Arc::new((0..ROUNDS).map(|_| SetOnce::new()).collect());
        let wins: Arc<Vec<AtomicUsize>> =
            Arc::new((0..ROUNDS).map(|_| AtomicUsize::new(0)).collect());
        let barrier = Arc::new(Barrier::new(THREADS));
        let handles: Vec<_> = (0..THREADS)
            .map(|id| {
                let (cells, wins, barrier) = (cells.clone(), wins.clone(), barrier.clone());
                thread::spawn(move || {
                    barrier.wait();
                    for round in 0..ROUNDS {
                        if cells[round].set(id).is_ok() {
                            wins[round].fetch_add(1, StdOrdering::SeqCst);
                        }
                        let winner = *cells[round].get().expect("set by someone");
                        assert!(winner < THREADS);
                    }
                })
            })
            .collect();
        for handle in handles {
            handle.join().unwrap();
        }
        for round in 0..ROUNDS {
            assert_eq!(wins[round].load(StdOrdering::SeqCst), 1, "round {round}");
        }
    }

    #[test]
    fn into_inner_and_get_mut_return_the_value() {
        let mut cell = SetOnce::new();
        assert!(cell.get_mut().is_none());
        cell.set(String::from("a")).unwrap();
        cell.get_mut().unwrap().push('b');
        assert_eq!(cell.into_inner().as_deref(), Some("ab"));
        assert_eq!(SetOnce::<u8>::new().into_inner(), None);
    }
}

#[cfg(all(loom, test))]
mod loom_tests {
    use super::*;
    use loom::cell::UnsafeCell;
    use loom::sync::Arc;
    use loom::thread;

    /// A value written plainly before it is installed. Loom tracks the plain accesses, so a `get`
    /// that did not acquire what `set` released shows up as a data race on this cell.
    struct Payload {
        written: UnsafeCell<u32>,
    }

    // SAFETY: the payload is written once, before `set` publishes it, and only read after `get`.
    unsafe impl Send for Payload {}
    // SAFETY: as above; `SetOnce` hands out only shared references to a published payload.
    unsafe impl Sync for Payload {}

    impl Payload {
        fn new(value: u32) -> Self {
            let payload = Self {
                written: UnsafeCell::new(0),
            };
            // SAFETY: the payload is not shared yet.
            payload.written.with_mut(|slot| unsafe { *slot = value });
            payload
        }

        fn read(&self) -> u32 {
            // SAFETY: a published payload is never written again.
            self.written.with(|slot| unsafe { *slot })
        }
    }

    #[test]
    fn loom_one_setter_wins_and_every_reader_sees_it() {
        loom::model(|| {
            let cell = Arc::new(SetOnce::new());
            let setters: Vec<_> = [1u32, 2]
                .into_iter()
                .map(|value| {
                    let cell = cell.clone();
                    thread::spawn(move || cell.set(Payload::new(value)).is_ok())
                })
                .collect();
            let reader = {
                let cell = cell.clone();
                thread::spawn(move || cell.get().map(Payload::read))
            };
            let wins: Vec<bool> = setters.into_iter().map(|t| t.join().unwrap()).collect();
            let seen = reader.join().unwrap();
            assert_eq!(wins.iter().filter(|&&won| won).count(), 1);
            let winner = if wins[0] { 1 } else { 2 };
            assert_eq!(cell.get().map(Payload::read), Some(winner));
            if let Some(seen) = seen {
                assert_eq!(seen, winner);
            }
        });
    }
}
