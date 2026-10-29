use super::loom::{AtomicI64, Ordering};
use std::fmt;

/// A reader count with a retired bit: the admission a routing state is read under.
///
/// A reader [`try_acquire`](Self::try_acquire)s a count while the state is current, uses the state, and
/// [`release`](Self::release)s. A writer that replaces the state [`retire`](Self::retire)s it, which refuses new
/// readers and leaves the count as it is; the release that brings a retired state to zero readers says so, and that
/// reader (or the writer, if there was none) releases what the state holds.
///
/// The counter is one word: the readers in the low 63 bits and the retired bit as the sign bit, so "retired and
/// without readers" is exactly [`RETIRED`](Self::RETIRED), one read. This is Java's `RoutingState.admission`:
/// `tryAcquire` is a compare-and-set up while the word is not negative, `retire` a compare-and-set of the sign bit
/// in, `release` a decrement. A reader that loaded the address of a state just before the writer replaced it may
/// still count against it; the owner of the states keeps them allocated, and the reader's re-check against the
/// current state decides whether it may use it.
///
/// A state is rewritten for a new generation only while its word is exactly [`RETIRED`](Self::RETIRED) (no reader
/// holds a count and every `try_acquire` refuses); [`reopen`](Self::reopen) then admits readers again.
pub struct Admission {
    word: AtomicI64,
}

impl Admission {
    /// The word of a retired state with no readers.
    pub const RETIRED: i64 = i64::MIN;

    /// An admission that admits readers, with none counted.
    #[cfg(not(loom))]
    pub const fn new() -> Self {
        Self {
            word: AtomicI64::new(0),
        }
    }

    /// An admission that admits readers, with none counted.
    #[cfg(loom)]
    pub fn new() -> Self {
        Self {
            word: AtomicI64::new(0),
        }
    }

    /// Takes a count unless the state is retired. A compare-and-set, `SeqCst` as a volatile one is.
    ///
    /// # Panics
    ///
    /// If the count overflows.
    pub fn try_acquire(&self) -> bool {
        let mut current = self.word.load(Ordering::Acquire);
        while current >= 0 {
            assert!(current != i64::MAX, "reader count overflow");
            match self.word.compare_exchange(
                current,
                current + 1,
                Ordering::SeqCst,
                Ordering::SeqCst,
            ) {
                Ok(_) => return true,
                Err(actual) => current = actual,
            }
        }
        false
    }

    /// Refuses new readers from now on, leaving the count as it is. Does nothing if already retired.
    pub fn retire(&self) {
        let mut current = self.word.load(Ordering::Acquire);
        while current >= 0 {
            match self.word.compare_exchange(
                current,
                current | Self::RETIRED,
                Ordering::SeqCst,
                Ordering::SeqCst,
            ) {
                Ok(_) => return,
                Err(actual) => current = actual,
            }
        }
    }

    /// Gives a count back. Returns true when the state is retired and that was its last reader.
    pub fn release(&self) -> bool {
        self.word.fetch_sub(1, Ordering::SeqCst) - 1 == Self::RETIRED
    }

    /// Whether any reader holds a count.
    pub fn has_readers(&self) -> bool {
        self.word.load(Ordering::Acquire) & i64::MAX != 0
    }

    /// Whether the state is retired.
    pub fn is_retired(&self) -> bool {
        self.word.load(Ordering::Acquire) < 0
    }

    /// Whether the state is retired and no reader holds a count: the one moment it may be rewritten.
    pub fn is_retired_and_idle(&self) -> bool {
        self.word.load(Ordering::SeqCst) == Self::RETIRED
    }

    /// Admits readers again, after the state was rewritten. Only for a state that
    /// [`is_retired_and_idle`](Self::is_retired_and_idle): the store publishes the rewrite to the readers that
    /// count against it.
    pub fn reopen(&self) {
        debug_assert!(
            self.is_retired_and_idle(),
            "only an idle retired state reopens"
        );
        self.word.store(0, Ordering::SeqCst);
    }
}

#[cfg(not(loom))]
impl Default for Admission {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for Admission {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Admission")
            .field("retired", &self.is_retired())
            .field("readers", &(self.word.load(Ordering::Relaxed) & i64::MAX))
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn readers_count_up_and_the_last_release_of_a_retired_state_says_so() {
        let admission = Admission::new();
        assert!(admission.try_acquire());
        assert!(admission.try_acquire());
        assert!(admission.has_readers());
        admission.retire();
        assert!(admission.is_retired());
        assert!(!admission.try_acquire(), "a retired state refuses readers");
        assert!(!admission.release(), "one reader is left");
        assert!(admission.release(), "the last reader of a retired state");
        assert!(admission.is_retired_and_idle());
    }

    #[test]
    fn releasing_a_current_state_never_says_it_is_last() {
        let admission = Admission::new();
        assert!(admission.try_acquire());
        assert!(!admission.release());
        assert!(!admission.is_retired());
        assert!(!admission.has_readers());
    }

    #[test]
    fn retiring_twice_changes_nothing() {
        let admission = Admission::new();
        assert!(admission.try_acquire());
        admission.retire();
        admission.retire();
        assert!(admission.release());
    }

    #[test]
    fn an_idle_retired_state_reopens_for_readers() {
        let admission = Admission::new();
        admission.retire();
        assert!(admission.is_retired_and_idle());
        admission.reopen();
        assert!(!admission.is_retired());
        assert!(admission.try_acquire());
        assert!(!admission.release());
    }

    #[test]
    fn a_state_with_a_reader_is_not_idle() {
        let admission = Admission::new();
        assert!(admission.try_acquire());
        admission.retire();
        assert!(!admission.is_retired_and_idle());
    }

    #[test]
    #[should_panic(expected = "reader count overflow")]
    fn the_count_does_not_overflow_into_the_retired_bit() {
        let admission = Admission {
            word: AtomicI64::new(i64::MAX),
        };
        admission.try_acquire();
    }
}

#[cfg(loom)]
mod loom_tests {
    use super::*;
    use loom::cell::UnsafeCell;
    use loom::sync::atomic::AtomicPtr;
    use loom::thread;
    use std::sync::Arc;

    /// A routing state in miniature: the admission and the table readers use.
    struct State {
        admission: Admission,
        table: UnsafeCell<u32>,
    }

    // SAFETY: the table is read under a count and written only while the state is retired and idle.
    unsafe impl Sync for State {}
    // SAFETY: as above.
    unsafe impl Send for State {}

    fn state(table: u32) -> State {
        State {
            admission: Admission::new(),
            table: UnsafeCell::new(table),
        }
    }

    /// The reader's side of `tryAcquireRoutingState`: load the pointer, count, re-check; read the table.
    fn read(current: &AtomicPtr<State>) -> Option<u32> {
        let pointer = current.load(Ordering::Acquire);
        // SAFETY: states outlive the model's threads.
        let state = unsafe { &*pointer };
        if !state.admission.try_acquire() {
            return None;
        }
        let value = (pointer == current.load(Ordering::Acquire)).then(|| {
            // SAFETY: a count is held on a state that is not retired, so it is not being rewritten.
            state.table.with(|table| unsafe { *table })
        });
        state.admission.release();
        value
    }

    // One state is published; a reader loads it and may pause anywhere; the writer publishes the other, retires the
    // first, rewrites it once it is idle and publishes it again. A reader that counts against the recycled state
    // either re-checks successfully, and reads the table the writer wrote before reopening it, or gives the count
    // back; loom reports any read that races with the rewrite.
    #[test]
    fn loom_stale_reader_and_recycler() {
        loom::model(|| {
            let first = Arc::new(state(1));
            let second = Arc::new(state(2));
            let current = Arc::new(AtomicPtr::new(Arc::as_ptr(&first).cast_mut()));

            let reader = {
                let (first, second, current) = (first.clone(), second.clone(), current.clone());
                thread::spawn(move || {
                    let _keep = (first, second);
                    read(&current)
                })
            };
            let writer = {
                let (first, second, current) = (first.clone(), second.clone(), current.clone());
                thread::spawn(move || {
                    // Generation two.
                    current.store(Arc::as_ptr(&second).cast_mut(), Ordering::Release);
                    first.admission.retire();
                    // Generation three reuses the first state once its readers are gone.
                    if first.admission.is_retired_and_idle() {
                        // SAFETY: the word is exactly RETIRED: no reader, and every acquire refuses.
                        first.table.with_mut(|table| unsafe { *table = 3 });
                        first.admission.reopen();
                        current.store(Arc::as_ptr(&first).cast_mut(), Ordering::Release);
                    }
                })
            };
            let read = reader.join().unwrap();
            writer.join().unwrap();
            // Whatever the reader read was a generation that was current when it counted.
            assert!(matches!(read, None | Some(1..=3)), "read {read:?}");
        });
    }

    // The release that leaves a retired state without readers, and the writer's own check after it retires, race;
    // a once-only gate (Java's `handlesReleased`) lets exactly one of them release what the state holds, however
    // the retirement falls among the releases.
    #[test]
    fn loom_one_thread_releases_a_retired_state_whichever_side_sees_it_idle() {
        use loom::sync::atomic::AtomicBool;
        loom::model(|| {
            let admission = Arc::new(Admission::new());
            let gate = Arc::new(AtomicBool::new(false));
            assert!(admission.try_acquire());
            assert!(admission.try_acquire());
            let take = |gate: &AtomicBool| {
                gate.compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
                    .is_ok()
            };
            let readers: Vec<_> = (0..2)
                .map(|_| {
                    let (admission, gate) = (admission.clone(), gate.clone());
                    thread::spawn(move || {
                        admission.release() && admission.is_retired() && take(&gate)
                    })
                })
                .collect();
            let retirer = {
                let (admission, gate) = (admission.clone(), gate.clone());
                thread::spawn(move || {
                    admission.retire();
                    !admission.has_readers() && take(&gate)
                })
            };
            let mut released = usize::from(retirer.join().unwrap());
            for reader in readers {
                released += usize::from(reader.join().unwrap());
            }
            assert_eq!(released, 1, "the state's handles are released exactly once");
            assert!(admission.is_retired_and_idle());
        });
    }
}
