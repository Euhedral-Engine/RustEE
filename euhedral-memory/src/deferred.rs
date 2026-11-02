//! Frees that wait for a quiet moment.
//!
//! On the CUDA driver, a handful of calls make every other thread's enqueue on the same context wait for the kernels that
//! are running when the call is made: `cuModuleLoadDataEx`, `cuMemFree`, `cuMemFreeHost` and `cuMemHostUnregister` (and the
//! pinned allocations and registrations hold the driver's lock for their own length). A `cuEventRecord` on an unrelated
//! stream, a lane's marker, then takes as long as the longest running kernel (measured at 249 ms and 1.25 s on an RTX 5070 Ti,
//! driver 615.71). Running such a call on another thread does not help; it has to run when nothing is in flight.
//!
//! [`Deferred`] is a lock-free list of such calls. When it is [enabled](Deferred::enable), a free that may block is not made
//! where it is dropped: it is queued with [`Deferred::defer`], and the owner of a quiet moment (every lane drained, a stop, a
//! rebuild) runs the queue with [`Deferred::run_all`]. When it is not enabled, `defer` runs the work at once, so the default
//! behavior of every drop is unchanged.
//!
//! The CUDA backends route their blocking frees through the process-wide list ([`deferred`]): `DeviceMemory::free`,
//! `PinnedHost::trim`, dropping an `Imported` mapping, and euhedral-gpu's module unload. An allocation that fails for lack of
//! memory runs the list first and tries again (an allocation is a build-time call, and the memory it needs may be waiting
//! there). Memory in the list is not available to the planner until it runs; the owner that defers drains before it refuses.
//!
//! The queue is lock-free and a defer makes one allocation (the boxed call), never on a frame path: a free is lifecycle work.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use euhedral_data_structures::queues::{Consumer, MpmcQueue, Multi, Producer};
use euhedral_data_structures::sync::HotScope;

type Work = Box<dyn FnOnce() + Send>;

/// A list of blocking calls waiting for a quiet moment.
pub struct Deferred {
    // Held so the queue outlives its two ends.
    _queue: MpmcQueue<Work>,
    producer: Producer<Work, Multi>,
    consumer: Consumer<Work, Multi>,
    enabled: AtomicBool,
    pending: AtomicUsize,
}

impl std::fmt::Debug for Deferred {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Deferred")
            .field("enabled", &self.is_enabled())
            .field("pending", &self.pending())
            .finish()
    }
}

impl Default for Deferred {
    fn default() -> Self {
        Self::new()
    }
}

impl Deferred {
    /// An empty list that is not enabled: [`defer`](Self::defer) runs its work at once.
    pub fn new() -> Self {
        let queue = MpmcQueue::new(64);
        let producer = queue.producer();
        let consumer = queue.consumer();
        Deferred {
            _queue: queue,
            producer,
            consumer,
            enabled: AtomicBool::new(false),
            pending: AtomicUsize::new(0),
        }
    }

    /// Turns deferral on or off. Turning it off does not run what is queued: call [`run_all`](Self::run_all).
    pub fn enable(&self, on: bool) {
        self.enabled.store(on, Ordering::Release);
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Acquire)
    }

    /// Queues `work` for the next [`run_all`](Self::run_all), or runs it now when deferral is off. Never waits.
    pub fn defer(&self, work: impl FnOnce() + Send + 'static) {
        if !self.is_enabled() {
            work();
            return;
        }
        self.pending.fetch_add(1, Ordering::AcqRel);
        if let Err(work) = self.producer.offer(Box::new(work)) {
            // The queue never refuses; if it did, the work runs now rather than being lost.
            self.pending.fetch_sub(1, Ordering::AcqRel);
            work();
        }
    }

    /// Runs everything queued, in order, on the calling thread, and returns how many calls ran. The caller is at a quiet
    /// moment: the calls block. Another thread running the list at the same moment makes this return early.
    pub fn run_all(&self) -> usize {
        let _cold = HotScope::leave();
        let mut ran = 0;
        loop {
            let mut batch = Vec::new();
            self.consumer.drain_all(|work| batch.push(work));
            if batch.is_empty() {
                return ran;
            }
            for work in batch {
                work();
                self.pending.fetch_sub(1, Ordering::AcqRel);
                ran += 1;
            }
        }
    }

    /// How many calls wait.
    pub fn pending(&self) -> usize {
        self.pending.load(Ordering::Acquire)
    }
}

/// The process-wide list the CUDA backends use.
pub fn deferred() -> &'static Deferred {
    static LIST: OnceLock<Deferred> = OnceLock::new();
    LIST.get_or_init(Deferred::new)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    #[test]
    fn a_list_that_is_not_enabled_runs_the_work_at_once() {
        let list = Deferred::new();
        let ran = Arc::new(AtomicUsize::new(0));
        let r = ran.clone();
        list.defer(move || {
            r.fetch_add(1, Ordering::AcqRel);
        });
        assert_eq!(ran.load(Ordering::Acquire), 1);
        assert_eq!(list.pending(), 0);
        assert_eq!(list.run_all(), 0);
    }

    #[test]
    fn an_enabled_list_holds_the_work_until_run_all_and_runs_it_in_order() {
        let list = Deferred::new();
        list.enable(true);
        let order = Arc::new(std::sync::Mutex::new(Vec::new()));
        for i in 0..5 {
            let o = order.clone();
            list.defer(move || o.lock().unwrap().push(i));
        }
        assert_eq!(list.pending(), 5);
        assert!(order.lock().unwrap().is_empty(), "nothing ran while queued");
        assert_eq!(list.run_all(), 5);
        assert_eq!(*order.lock().unwrap(), vec![0, 1, 2, 3, 4]);
        assert_eq!(list.pending(), 0);
        assert_eq!(list.run_all(), 0);
    }

    #[test]
    fn work_deferred_from_many_threads_all_runs_once() {
        let list = Arc::new(Deferred::new());
        list.enable(true);
        let ran = Arc::new(AtomicUsize::new(0));
        std::thread::scope(|s| {
            for _ in 0..8 {
                let (list, ran) = (list.clone(), ran.clone());
                s.spawn(move || {
                    for _ in 0..100 {
                        let ran = ran.clone();
                        list.defer(move || {
                            ran.fetch_add(1, Ordering::AcqRel);
                        });
                    }
                });
            }
        });
        assert_eq!(list.pending(), 800);
        assert_eq!(list.run_all(), 800);
        assert_eq!(ran.load(Ordering::Acquire), 800);
    }

    #[test]
    fn turning_deferral_off_keeps_what_is_queued_until_it_runs() {
        let list = Deferred::new();
        list.enable(true);
        let ran = Arc::new(AtomicUsize::new(0));
        let r = ran.clone();
        list.defer(move || {
            r.fetch_add(1, Ordering::AcqRel);
        });
        list.enable(false);
        assert_eq!(ran.load(Ordering::Acquire), 0);
        let r = ran.clone();
        list.defer(move || {
            r.fetch_add(10, Ordering::AcqRel);
        });
        assert_eq!(
            ran.load(Ordering::Acquire),
            10,
            "off: the new work runs now"
        );
        assert_eq!(list.run_all(), 1);
        assert_eq!(ran.load(Ordering::Acquire), 11);
    }
}
