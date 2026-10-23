use std::cell::Cell;

thread_local! {
    static SLOT: Slot = const {
        Slot {
            active: Cell::new(false),
            context: FlowContext::new(),
        }
    };
}

struct Slot {
    active: Cell<bool>,
    context: FlowContext,
}

/// Per-thread bookkeeping of how much of a worker's demand its upstreams have satisfied.
///
/// A worker [initializes](Self::initialize) the context on its own thread, and code deep in the
/// call stack reaches it through [`with`](Self::with) without it being passed down. The counters are
/// `Cell`s, so nested calls on the same thread can all update the one context.
#[derive(Debug, Default)]
pub struct FlowContext {
    satisfied_request: Cell<u64>,
    original_request: Cell<u64>,
    satisfied_pull: Cell<u64>,
    original_pull: Cell<u64>,
}

impl FlowContext {
    const fn new() -> Self {
        Self {
            satisfied_request: Cell::new(0),
            original_request: Cell::new(0),
            satisfied_pull: Cell::new(0),
            original_pull: Cell::new(0),
        }
    }

    /// Runs `f` with this thread's context, or with `None` if the thread has not initialized one.
    pub fn with<R>(f: impl FnOnce(Option<&FlowContext>) -> R) -> R {
        SLOT.with(|slot| f(slot.active.get().then_some(&slot.context)))
    }

    /// Gives this thread a context, keeping the counters of one it already has.
    pub fn initialize() {
        SLOT.with(|slot| slot.active.set(true));
    }

    /// Removes this thread's context; a later [`initialize`](Self::initialize) starts from zero.
    pub fn clear() {
        SLOT.with(|slot| {
            slot.active.set(false);
            slot.context.clear_counters();
        });
    }

    pub fn clear_counters(&self) {
        self.satisfied_request.set(0);
        self.original_request.set(0);
        self.satisfied_pull.set(0);
        self.original_pull.set(0);
    }

    /// Frames delivered against the current request.
    pub fn satisfied_request(&self) -> u64 {
        self.satisfied_request.get()
    }

    pub fn add_satisfied_request(&self, frames: u64) {
        self.satisfied_request
            .set(self.satisfied_request.get().saturating_add(frames));
    }

    /// The demand the current request asked for.
    pub fn original_request(&self) -> u64 {
        self.original_request.get()
    }

    pub fn set_original_request(&self, demand: u64) {
        self.original_request.set(demand);
    }

    /// Frames handed over by the current pull.
    pub fn satisfied_pull(&self) -> u64 {
        self.satisfied_pull.get()
    }

    pub fn add_satisfied_pull(&self, frames: u64) {
        self.satisfied_pull
            .set(self.satisfied_pull.get().saturating_add(frames));
    }

    /// The demand the current pull asked for.
    pub fn original_pull(&self) -> u64 {
        self.original_pull.get()
    }

    pub fn set_original_pull(&self, demand: u64) {
        self.original_pull.set(demand);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    fn counters() -> Option<[u64; 4]> {
        FlowContext::with(|context| {
            context.map(|context| {
                [
                    context.satisfied_request(),
                    context.original_request(),
                    context.satisfied_pull(),
                    context.original_pull(),
                ]
            })
        })
    }

    #[test]
    fn a_thread_has_no_context_until_it_initializes_one() {
        thread::spawn(|| {
            assert_eq!(counters(), None);
            FlowContext::initialize();
            assert_eq!(counters(), Some([0; 4]));
        })
        .join()
        .unwrap();
    }

    #[test]
    fn counters_accumulate_and_clear() {
        thread::spawn(|| {
            FlowContext::initialize();
            FlowContext::with(|context| {
                let context = context.unwrap();
                context.set_original_request(10);
                context.add_satisfied_request(3);
                context.add_satisfied_request(4);
                context.set_original_pull(5);
                context.add_satisfied_pull(2);
            });
            assert_eq!(counters(), Some([7, 10, 2, 5]));

            FlowContext::with(|context| context.unwrap().clear_counters());
            assert_eq!(counters(), Some([0; 4]));
        })
        .join()
        .unwrap();
    }

    #[test]
    fn nested_calls_update_the_same_context() {
        thread::spawn(|| {
            FlowContext::initialize();
            FlowContext::with(|outer| {
                outer.unwrap().add_satisfied_pull(1);
                FlowContext::with(|inner| inner.unwrap().add_satisfied_pull(2));
                assert_eq!(outer.unwrap().satisfied_pull(), 3);
            });
        })
        .join()
        .unwrap();
    }

    #[test]
    fn reinitializing_keeps_the_counters_and_clearing_drops_them() {
        thread::spawn(|| {
            FlowContext::initialize();
            FlowContext::with(|context| context.unwrap().add_satisfied_request(4));
            FlowContext::initialize();
            assert_eq!(counters(), Some([4, 0, 0, 0]));

            FlowContext::clear();
            assert_eq!(counters(), None);
            FlowContext::initialize();
            assert_eq!(counters(), Some([0; 4]));
        })
        .join()
        .unwrap();
    }

    #[test]
    fn each_thread_has_its_own_context() {
        thread::spawn(|| {
            FlowContext::initialize();
            FlowContext::with(|context| context.unwrap().add_satisfied_request(9));
            thread::spawn(|| assert_eq!(counters(), None))
                .join()
                .unwrap();
            assert_eq!(counters(), Some([9, 0, 0, 0]));
        })
        .join()
        .unwrap();
    }

    #[test]
    fn counters_saturate() {
        thread::spawn(|| {
            FlowContext::initialize();
            FlowContext::with(|context| {
                let context = context.unwrap();
                context.add_satisfied_request(u64::MAX);
                context.add_satisfied_request(1);
                assert_eq!(context.satisfied_request(), u64::MAX);
            });
        })
        .join()
        .unwrap();
    }
}
