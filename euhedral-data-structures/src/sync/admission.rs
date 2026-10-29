use super::loom::{AtomicI64, Ordering};
use std::fmt;

/// A reader count with a retired bit: the admission that readers of a replaceable state take.
///
/// A reader [`try_acquire`](Self::try_acquire)s a count while the state is current, uses the state, and
/// [`release`](Self::release)s. A writer that replaces the state [`retire`](Self::retire)s it, which refuses new
/// readers and leaves the count as it is; the release that brings a retired state to zero readers says so, and that
/// reader (or the writer, if there was none) releases what the state holds.
///
/// The counter is one word: the readers in the low 63 bits and the retired bit as the sign bit, so "retired and
/// without readers" is exactly [`RETIRED`](Self::RETIRED), one read. This is Java's `admission` word:
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

#[cfg(all(loom, test))]
mod loom_tests {
    use super::*;
    use loom::cell::UnsafeCell;
    use loom::sync::atomic::{AtomicBool, AtomicPtr};
    use loom::thread;
    use std::sync::Arc;

    /// A replaceable state in miniature: the admission and the table readers use.
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

    /// The reader's side: load the pointer, count, re-check; read the table.
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

    /// The release protocol of a replaceable state in miniature: two states, a book (a lock) that serializes the
    /// rewriter's rewrite and every release, a `released` flag per state, and a table the release reads and clears.
    struct Model {
        states: [ModelState; 2],
        current: AtomicPtr<ModelState>,
        /// The generations released, in order; the book.
        book: loom::sync::Mutex<Vec<u32>>,
    }

    struct ModelState {
        admission: Admission,
        released: AtomicBool,
        /// The generation the state holds; 0 once released.
        table: UnsafeCell<u32>,
    }

    // SAFETY: the table is read under a count or under the book, and written under the book while the state is
    // retired, idle and released (a rewrite) or retired, idle and not released (the release).
    unsafe impl Sync for ModelState {}
    // SAFETY: as above.
    unsafe impl Send for ModelState {}

    impl Model {
        /// The first state publishes generation 1; the second is retired, idle and released: free for reuse.
        fn new() -> Self {
            let model = Self {
                states: [
                    ModelState {
                        admission: Admission::new(),
                        released: AtomicBool::new(false),
                        table: UnsafeCell::new(1),
                    },
                    ModelState {
                        admission: Admission::new(),
                        released: AtomicBool::new(true),
                        table: UnsafeCell::new(0),
                    },
                ],
                current: AtomicPtr::new(std::ptr::null_mut()),
                book: loom::sync::Mutex::new(Vec::new()),
            };
            model.states[1].admission.retire();
            model
        }

        fn publish_first(&self) {
            self.current
                .store(&self.states[0] as *const _ as *mut _, Ordering::Release);
        }

        /// Java's `tryAcquire`, then a read of the table under the count, then the release.
        fn read(&self) -> Option<u32> {
            let pointer = self.current.load(Ordering::Acquire);
            // SAFETY: the states live as long as the model.
            let state = unsafe { &*pointer };
            if !state.admission.try_acquire() {
                return None;
            }
            let generation = (pointer == self.current.load(Ordering::Acquire)).then(|| {
                // SAFETY: a count is held on a state that is not retired.
                state.table.with(|table| unsafe { *table })
            });
            if state.admission.release() {
                self.release_handles(state);
            }
            generation
        }

        /// The remap: under the book, rewrite a state that is retired, idle and released for the next generation,
        /// reopen it, publish it and retire the old one; then release the old one. Returns whether it remapped.
        fn remap(&self, generation: u32) -> bool {
            let retired = {
                let book = self.book.lock().unwrap();
                let Some(next) = self.states.iter().find(|state| {
                    state.admission.is_retired_and_idle() && state.released.load(Ordering::Acquire)
                }) else {
                    return false;
                };
                // SAFETY: retired, idle and released: no reader holds a count or can take one, and the release
                // has read the table under the book, which is held.
                next.table.with_mut(|table| unsafe { *table = generation });
                next.released.store(false, Ordering::Relaxed);
                next.admission.reopen();
                let old = self
                    .current
                    .swap(next as *const _ as *mut _, Ordering::AcqRel);
                // SAFETY: the states live as long as the model.
                let old = unsafe { &*old };
                old.admission.retire();
                drop(book);
                old
            };
            self.release_handles(retired);
            true
        }

        /// `release_handles_when_quiescent`: the checks before the lock only filter; the ones under it decide.
        fn release_handles(&self, state: &ModelState) {
            if !state.admission.is_retired() || state.admission.has_readers() {
                return;
            }
            let mut book = self.book.lock().unwrap();
            if state.released.load(Ordering::Acquire) || !state.admission.is_retired_and_idle() {
                return;
            }
            // SAFETY: retired and idle, not released, under the book: no one else reads or rewrites the table.
            let generation = state
                .table
                .with_mut(|table| unsafe { std::mem::take(&mut *table) });
            book.push(generation);
            state.released.store(true, Ordering::Release);
        }
    }

    // The rewriter replaces the table twice, so the first state is recycled for generation 3 when its readers are gone; one
    // reader reads and releases (and may be the last reader of a retired generation, late), the other is stale:
    // it loaded a pointer before the replacements and counts after them. Every generation the rewriter retired is
    // released exactly once, with its own table, never while a reader holds a count (loom reports a table read
    // or cleared concurrently with a rewrite), and a late release of a recycled state changes nothing.
    #[test]
    fn loom_the_release_protocol_releases_each_generation_once_whatever_the_recycling() {
        let mut builder = loom::model::Builder::new();
        builder.preemption_bound = Some(3);
        builder.check(|| {
            let model = Arc::new(Model::new());
            model.publish_first();
            let remapper = {
                let model = model.clone();
                thread::spawn(move || {
                    let first = model.remap(2);
                    let second = model.remap(3);
                    (first, second)
                })
            };
            let readers: Vec<_> = (0..2)
                .map(|_| {
                    let model = model.clone();
                    thread::spawn(move || model.read())
                })
                .collect();
            let (first, second) = remapper.join().unwrap();
            for reader in readers {
                let read = reader.join().unwrap();
                assert!(matches!(read, None | Some(1..=3)), "read {read:?}");
            }
            let book = model.book.lock().unwrap();
            let retired = usize::from(first) + usize::from(second);
            // Each retired generation was released, once; the last published one still routes.
            let mut released = book.clone();
            released.sort_unstable();
            released.dedup();
            assert_eq!(released.len(), book.len(), "released twice: {book:?}");
            assert_eq!(
                book.len(),
                retired,
                "a retired generation was not released: {book:?}"
            );
            let expected: Vec<u32> = match (first, second) {
                (true, true) => vec![1, 2],
                (true, false) => vec![1],
                _ => vec![],
            };
            assert_eq!(released, expected, "wrong generations released: {book:?}");
        });
    }
}
