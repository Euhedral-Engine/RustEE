use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

pub(super) type Done = Box<dyn FnOnce() + Send>;

/// Calls its action once `count` ticks have arrived; at once for a count of zero.
pub(super) struct Countdown {
    remaining: AtomicUsize,
    done: Mutex<Option<Done>>,
}

impl Countdown {
    pub(super) fn new(count: usize, done: Done) -> Arc<Self> {
        if count == 0 {
            done();
            return Arc::new(Self {
                remaining: AtomicUsize::new(0),
                done: Mutex::new(None),
            });
        }
        Arc::new(Self {
            remaining: AtomicUsize::new(count),
            done: Mutex::new(Some(done)),
        })
    }

    pub(super) fn tick(&self) {
        if self.remaining.fetch_sub(1, Ordering::AcqRel) == 1
            && let Some(done) = self
                .done
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .take()
        {
            done();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn counted() -> (Arc<AtomicUsize>, Done) {
        let calls = Arc::new(AtomicUsize::new(0));
        let counter = calls.clone();
        (
            calls,
            Box::new(move || {
                counter.fetch_add(1, Ordering::SeqCst);
            }),
        )
    }

    #[test]
    fn calls_its_action_on_the_last_tick() {
        let (calls, done) = counted();
        let countdown = Countdown::new(2, done);
        countdown.tick();
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        countdown.tick();
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn a_count_of_zero_calls_its_action_at_once() {
        let (calls, done) = counted();
        let _countdown = Countdown::new(0, done);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }
}
