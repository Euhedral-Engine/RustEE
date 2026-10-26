//! Run-time fan-in: the last arrival publishes the continuation (Euhedral-Inference `Join`).

use std::fmt;
use std::sync::{Arc, Mutex, PoisonError};

/// Waits for `parts` branches without waiting: each branch arrives (or is abandoned) from any
/// thread, and the last of them runs the continuation, once, with every part's value (`None` for an
/// abandoned branch).
pub struct Join<T> {
    state: Mutex<JoinState<T>>,
}

type Then<T> = Box<dyn FnOnce(Vec<Option<T>>) + Send>;

struct JoinState<T> {
    values: Vec<Option<T>>,
    /// Which parts arrived or were abandoned.
    ended: Vec<bool>,
    remaining: usize,
    then: Option<Then<T>>,
}

impl<T> fmt::Debug for Join<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Join").finish_non_exhaustive()
    }
}

impl<T: Send + 'static> Join<T> {
    /// A join of `parts` branches (at least one) that runs `then` on the last arrival's thread.
    /// `then` must not wait.
    pub fn new(parts: usize, then: impl FnOnce(Vec<Option<T>>) + Send + 'static) -> Arc<Self> {
        assert!(parts >= 1, "a join needs at least one part");
        Arc::new(Self {
            state: Mutex::new(JoinState {
                values: (0..parts).map(|_| None).collect(),
                ended: vec![false; parts],
                remaining: parts,
                then: Some(Box::new(then)),
            }),
        })
    }

    /// Branch `part` arrived with `value`. Each part arrives or is abandoned exactly once.
    pub fn arrive(&self, part: usize, value: T) {
        self.end(part, Some(value));
    }

    /// Branch `part` ended without a value.
    pub fn abandon(&self, part: usize) {
        self.end(part, None);
    }

    fn end(&self, part: usize, value: Option<T>) {
        let last = {
            let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
            let parts = state.ended.len();
            assert!(part < parts, "part {part} of a join of {parts}");
            assert!(!state.ended[part], "part {part} of a join ended twice");
            state.ended[part] = true;
            state.values[part] = value;
            state.remaining -= 1;
            (state.remaining == 0).then(|| (std::mem::take(&mut state.values), state.then.take()))
        };
        // The continuation runs outside the lock: it may start work that arrives at another join.
        if let Some((values, Some(then))) = last {
            then(values);
        }
    }
}

/// The port of Euhedral-Inference's `JoinTest` (whose join counts arrivals and carries the first
/// failure; this one carries every part's value, `None` for a failed or abandoned branch). The Java
/// join is reusable by rounds; this one is one round, made per use.
#[cfg(test)]
mod tests {
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    type Runs = Arc<Mutex<Vec<Vec<Option<&'static str>>>>>;

    fn recording(parts: usize) -> (Arc<Join<&'static str>>, Runs) {
        let runs = Arc::new(Mutex::new(Vec::new()));
        let sink = runs.clone();
        (
            Join::new(parts, move |values| sink.lock().unwrap().push(values)),
            runs,
        )
    }

    /// JoinTest.theLastArrivalPublishesTheContinuationOnceAndCarriesTheFirstFailure
    #[test]
    fn the_last_arrival_runs_the_continuation_once_with_every_part() {
        let (join, runs) = recording(3);
        join.arrive(2, "c");
        join.abandon(0);
        assert!(
            runs.lock().unwrap().is_empty(),
            "the join waits for every part"
        );
        join.arrive(1, "b");
        assert_eq!(*runs.lock().unwrap(), [vec![None, Some("b"), Some("c")]]);
    }

    #[test]
    fn a_join_of_one_runs_on_its_only_arrival() {
        let (join, runs) = recording(1);
        join.arrive(0, "only");
        assert_eq!(*runs.lock().unwrap(), [vec![Some("only")]]);
    }

    #[test]
    fn a_part_ending_twice_or_out_of_range_is_refused() {
        let (join, runs) = recording(2);
        join.arrive(0, "a");
        assert!(catch_unwind(AssertUnwindSafe(|| join.arrive(0, "again"))).is_err());
        assert!(catch_unwind(AssertUnwindSafe(|| join.abandon(2))).is_err());
        join.arrive(1, "b");
        assert_eq!(
            *runs.lock().unwrap(),
            [vec![Some("a"), Some("b")]],
            "a refused part changes nothing"
        );
    }

    #[test]
    #[should_panic(expected = "at least one part")]
    fn a_join_of_none_is_refused() {
        let _ = Join::<()>::new(0, |_| {});
    }

    #[test]
    #[allow(clippy::disallowed_methods)] // The test's own threads arrive, as branches on lattice workers would.
    fn parts_arriving_on_many_threads_run_the_continuation_exactly_once() {
        for _ in 0..50 {
            let runs = Arc::new(AtomicUsize::new(0));
            let counted = runs.clone();
            let join = Join::new(64, move |values: Vec<Option<usize>>| {
                assert!(values.iter().enumerate().all(|(i, v)| *v == Some(i)));
                counted.fetch_add(1, Ordering::AcqRel);
            });
            let threads: Vec<_> = (0..8)
                .map(|t| {
                    let join = join.clone();
                    std::thread::spawn(move || {
                        (t * 8..t * 8 + 8).for_each(|part| join.arrive(part, part))
                    })
                })
                .collect();
            for thread in threads {
                thread.join().unwrap();
            }
            assert_eq!(runs.load(Ordering::Acquire), 1);
        }
    }
}
