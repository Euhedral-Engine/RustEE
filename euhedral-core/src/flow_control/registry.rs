use super::upstream_queue::{Entry, UpstreamHandle, UpstreamQueue};
use euhedral_data_structures::atomics::PaddedAtomicLong;
use euhedral_data_structures::queues::{MpscQueue, Multi, Producer};
use euhedral_data_structures::sync::{EpochDomain, EpochGuard, EpochSlot, LifecycleMutex};
use euhedral_hardware_utils::affinity;
use euhedral_hardware_utils::topology::Topology;
use euhedral_hardware_utils::util::CpuSet;
use std::cell::RefCell;
use std::collections::BTreeSet;
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Weak};

/// Chunk size of each worker's handle queue.
const HANDLE_QUEUE_CHUNK: usize = 256;

static NEXT_ID: AtomicU64 = AtomicU64::new(0);

thread_local! {
    static THREAD_QUEUES: RefCell<Vec<ThreadQueue>> = const { RefCell::new(Vec::new()) };
    static THREAD_EPOCHS: RefCell<Vec<ThreadEpoch>> = const { RefCell::new(Vec::new()) };
}

/// This thread's queue for one registry. Dropping it, when the thread exits or
/// [`UpstreamRegistry::remove_thread`] runs, withdraws the thread from the registry.
struct ThreadQueue {
    registry: Weak<UpstreamRegistry>,
    registry_id: u64,
    queue_id: u64,
    cpu: Option<usize>,
    queue: Rc<RefCell<UpstreamQueue>>,
    producer: Producer<Entry, Multi>,
}

/// This thread's slot in one registry's epoch domain. It lives apart from the queue: a thread that only reads the
/// lattice's published values (a lifecycle thread, a test) gets a slot and no queue.
struct ThreadEpoch {
    registry: Weak<UpstreamRegistry>,
    registry_id: u64,
    slot: Rc<EpochSlot>,
}

impl Drop for ThreadQueue {
    fn drop(&mut self) {
        if let Some(registry) = self.registry.upgrade() {
            registry.unregister(self.queue_id);
        }
    }
}

/// A registered worker: where new handles are sent for it.
struct Worker {
    queue_id: u64,
    cpu: Option<usize>,
    producer: Producer<Entry, Multi>,
}

#[derive(Default)]
struct State {
    workers: Vec<Worker>,
    /// Handles added so far, so a worker registering later starts with all of them.
    handles: Vec<Arc<dyn UpstreamHandle>>,
}

/// The upstream handles feeding one lattice, and the workers servicing them.
///
/// Every thread that pulls or requests through the lattice has its own [`UpstreamQueue`]. A
/// thread that [registers](Self::register) is a worker: its queue receives every handle, those
/// added before it registered included. The queue of a thread that never registers stays empty.
///
/// The edges of one lattice share its registry.
///
/// The registry also owns the lattice's [`EpochDomain`]: every thread that uses it has one slot there, and a
/// worker announces a quiescent point once per cycle. A lattice's published values (the cache directory, an edge's
/// parent) are freed once every worker has passed the epoch that replaced them.
pub struct UpstreamRegistry {
    id: u64,
    epochs: Arc<EpochDomain>,
    upstream_count: Arc<PaddedAtomicLong>,
    thread_count: AtomicUsize,
    /// One past each CPU's rank among the workers' CPUs, or zero for a CPU without workers.
    ranks: Box<[AtomicUsize]>,
    p_cpus: CpuSet,
    shuffle_window: usize,
    state: LifecycleMutex<State>,
}

impl UpstreamRegistry {
    /// A registry ranking workers by this machine's topology.
    pub fn new() -> Arc<Self> {
        let topology = Topology::system();
        Self::with_topology(topology.cpu_span(), topology.p_cpu_set().clone())
    }

    /// A registry for CPU ids below `cpu_span`, ranking the CPUs in `p_cpus` first.
    pub fn with_topology(cpu_span: usize, p_cpus: CpuSet) -> Arc<Self> {
        Arc::new(Self {
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            epochs: EpochDomain::new(cpu_span),
            upstream_count: Arc::new(PaddedAtomicLong::new(0)),
            thread_count: AtomicUsize::new(0),
            ranks: (0..cpu_span).map(|_| AtomicUsize::new(0)).collect(),
            p_cpus,
            shuffle_window: cpu_span.max(1),
            state: LifecycleMutex::new(State::default()),
        })
    }

    /// Runs `f` with the calling thread's queue, creating the queue on first use.
    ///
    /// Returns `None` when the queue is already in use further up this thread's stack, as when a
    /// source pulled by the queue calls back into the lattice, or while the thread is exiting.
    pub fn with_thread_queue<R>(
        self: &Arc<Self>,
        f: impl FnOnce(&mut UpstreamQueue) -> R,
    ) -> Option<R> {
        let queue = self.thread_queue(affinity::current_cpu, |queue| queue.queue.clone())?;
        let mut queue = queue.try_borrow_mut().ok()?;
        Some(f(&mut queue))
    }

    /// The epoch domain of the lattice's published values.
    pub fn epochs(&self) -> &Arc<EpochDomain> {
        &self.epochs
    }

    /// The calling thread's slot in [`epochs`](Self::epochs), created on first use. A worker keeps
    /// it for its cycles ([`EpochSlot::quiescent`], [`EpochSlot::idle`]); [`with_pinned`](Self::with_pinned) uses
    /// the same slot. `None` while the thread is exiting.
    pub fn thread_epoch(self: &Arc<Self>) -> Option<Rc<EpochSlot>> {
        THREAD_EPOCHS
            .try_with(|entries| {
                let mut entries = entries.borrow_mut();
                if let Some(entry) = entries.iter().find(|entry| entry.registry_id == self.id) {
                    return entry.slot.clone();
                }
                entries.retain(|entry| entry.registry.strong_count() > 0);
                let slot = Rc::new(self.epochs.register());
                entries.push(ThreadEpoch {
                    registry: Arc::downgrade(self),
                    registry_id: self.id,
                    slot: slot.clone(),
                });
                slot
            })
            .ok()
    }

    /// Runs `f` with a guard that keeps the values the calling thread loads from the lattice's [`Published`]
    /// cells valid for the call.
    ///
    /// The cost is a thread-local lookup of the thread's slot (a borrow of the thread's list and a search for this
    /// registry), a reference-count update on the slot, and the pin: inside a worker's cycle the pin is one load
    /// of the slot, and on any other thread it announces the thread for the call (two stores, and a fence only where the process cannot issue the
    /// writer's heavy barrier) and withdraws it. Pin once around a series of loads, as the edges' climb does. A thread that is exiting, and
    /// has no slot of its own any more, uses a temporary one.
    ///
    /// [`Published`]: euhedral_data_structures::sync::Published
    pub fn with_pinned<R>(self: &Arc<Self>, f: impl FnOnce(&EpochGuard<'_>) -> R) -> R {
        match self.thread_epoch() {
            Some(slot) => f(&slot.pin()),
            None => f(&self.epochs.register().pin()),
        }
    }

    /// Makes the calling thread a worker. Does nothing if it already is one.
    pub fn register(self: &Arc<Self>) {
        self.register_with(affinity::current_cpu);
    }

    /// Like [`register`](Self::register), taking the thread's CPU from `cpu` if its queue is new.
    pub(crate) fn register_with(self: &Arc<Self>, cpu: impl FnOnce() -> Option<usize>) {
        let worker = self.thread_queue(cpu, |queue| Worker {
            queue_id: queue.queue_id,
            cpu: queue.cpu,
            producer: queue.producer.clone(),
        });
        let Some(worker) = worker else {
            return;
        };
        let mut state = self.state.lock();
        if state
            .workers
            .iter()
            .any(|known| known.queue_id == worker.queue_id)
        {
            return;
        }
        state.handles.retain(|handle| !handle.is_complete());
        for handle in &state.handles {
            offer(&worker.producer, handle);
        }
        state.workers.push(worker);
        self.thread_count.fetch_add(1, Ordering::Relaxed);
        self.rank_cpus(&state);
        log::trace!("registered a worker thread");
    }

    /// Drops the calling thread's queue and withdraws the thread as a worker. The thread gets a new,
    /// empty queue if it uses the lattice again.
    pub fn remove_thread(&self) {
        let removed: Vec<ThreadQueue> = THREAD_QUEUES
            .try_with(|queues| {
                let mut queues = queues.borrow_mut();
                let (removed, kept) = queues
                    .drain(..)
                    .partition(|queue| queue.registry_id == self.id);
                *queues = kept;
                removed
            })
            .unwrap_or_default();
        drop(removed);
        // The thread's slot goes back to the domain once nothing else holds it.
        let _ = THREAD_EPOCHS.try_with(|entries| {
            entries
                .borrow_mut()
                .retain(|entry| entry.registry_id != self.id);
        });
    }

    /// The number of live upstream handles.
    pub fn upstream_handle_count(&self) -> u64 {
        self.upstream_count.get_opaque().max(0) as u64
    }

    /// The number of registered worker threads.
    pub fn thread_count(&self) -> usize {
        self.thread_count.load(Ordering::Relaxed)
    }

    /// `cpu`'s rank among the CPUs of the workers, starting from 1: performance CPUs first, then the
    /// rest, each in id order. `None` for a CPU without workers.
    pub fn thread_rank(&self, cpu: usize) -> Option<usize> {
        let rank = self.ranks.get(cpu)?.load(Ordering::Acquire);
        (rank != 0).then_some(rank)
    }

    /// Adds `handle`: every worker's queue receives it, and it counts as live until
    /// [`remove_handle`](Self::remove_handle) reports it complete.
    pub fn add_handle(&self, handle: Arc<dyn UpstreamHandle>) {
        let mut state = self.state.lock();
        state.handles.retain(|handle| !handle.is_complete());
        for worker in &state.workers {
            offer(&worker.producer, &handle);
        }
        state.handles.push(handle);
        self.upstream_count.increment_and_get();
        log::trace!("added an upstream handle");
    }

    /// Stops counting a handle that completed; a handle reports its completion here exactly once.
    /// Queues drop the handle when they next come across it.
    pub fn remove_handle(&self) {
        self.upstream_count.decrement_and_get();
    }

    /// Runs `f` with the calling thread's queue, creating the queue on first use.
    fn thread_queue<R>(
        self: &Arc<Self>,
        cpu: impl FnOnce() -> Option<usize>,
        f: impl FnOnce(&ThreadQueue) -> R,
    ) -> Option<R> {
        THREAD_QUEUES
            .try_with(|queues| {
                let mut queues = queues.borrow_mut();
                let index = match queues.iter().position(|queue| queue.registry_id == self.id) {
                    Some(index) => index,
                    None => {
                        queues.retain(|queue| queue.registry.strong_count() > 0);
                        queues.push(self.new_thread_queue(cpu()));
                        queues.len() - 1
                    }
                };
                f(&queues[index])
            })
            .ok()
    }

    fn new_thread_queue(self: &Arc<Self>, cpu: Option<usize>) -> ThreadQueue {
        let handles = MpscQueue::new(HANDLE_QUEUE_CHUNK);
        let queue = UpstreamQueue::new(
            cpu,
            &handles,
            self.upstream_count.clone(),
            self.shuffle_window,
        );
        ThreadQueue {
            registry: Arc::downgrade(self),
            registry_id: self.id,
            queue_id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            cpu,
            queue: Rc::new(RefCell::new(queue)),
            producer: handles.producer(),
        }
    }

    fn unregister(&self, queue_id: u64) {
        let mut state = self.state.lock();
        let Some(index) = state
            .workers
            .iter()
            .position(|worker| worker.queue_id == queue_id)
        else {
            return;
        };
        state.workers.swap_remove(index);
        self.thread_count.fetch_sub(1, Ordering::Relaxed);
        self.rank_cpus(&state);
        log::trace!("removed a worker thread");
    }

    fn rank_cpus(&self, state: &State) {
        let cpus: BTreeSet<usize> = state
            .workers
            .iter()
            .filter_map(|worker| worker.cpu)
            .filter(|&cpu| cpu < self.ranks.len())
            .collect();
        let mut ranks = vec![0; self.ranks.len()];
        let performance = cpus.iter().filter(|&&cpu| self.p_cpus.contains(cpu));
        let efficiency = cpus.iter().filter(|&&cpu| !self.p_cpus.contains(cpu));
        for (rank, &cpu) in performance.chain(efficiency).enumerate() {
            ranks[cpu] = rank + 1;
        }
        for (slot, rank) in self.ranks.iter().zip(ranks) {
            slot.store(rank, Ordering::Release);
        }
    }
}

fn offer(producer: &Producer<Entry, Multi>, handle: &Arc<dyn UpstreamHandle>) {
    if producer.offer(Entry::new(handle.clone())).is_err() {
        unreachable!("upstream handle queues are unbounded");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flow_control::Claim;
    use crate::frames::Frame;
    use std::sync::atomic::AtomicBool;
    use std::thread;

    #[derive(Default)]
    struct Handle {
        sequence: u64,
        requested: AtomicU64,
        complete: AtomicBool,
    }

    impl Handle {
        fn new(sequence: u64) -> Arc<Self> {
            Arc::new(Self {
                sequence,
                ..Self::default()
            })
        }
    }

    impl UpstreamHandle for Handle {
        fn sequence(&self) -> u64 {
            self.sequence
        }

        fn pull(
            &self,
            _claim: &mut Claim,
            _consumer: &mut dyn FnMut(Box<dyn Frame>),
            _stop: &mut dyn FnMut(&dyn Frame) -> bool,
            _demand: u64,
        ) -> Option<u64> {
            Some(0)
        }

        fn request(&self, _claim: &mut Claim, demand: u64) {
            self.requested.fetch_add(demand, Ordering::Relaxed);
        }

        fn is_complete(&self) -> bool {
            self.complete.load(Ordering::Relaxed)
        }
    }

    fn registry() -> Arc<UpstreamRegistry> {
        UpstreamRegistry::with_topology(8, CpuSet::new())
    }

    fn queue_address(registry: &Arc<UpstreamRegistry>) -> usize {
        registry
            .with_thread_queue(|queue| queue as *mut UpstreamQueue as usize)
            .unwrap()
    }

    /// Requests `demand` frames through the calling thread's queue.
    fn request(registry: &Arc<UpstreamRegistry>, demand: u64) {
        registry.with_thread_queue(|queue| queue.request(demand));
    }

    #[test]
    fn a_thread_reuses_its_queue() {
        let registry = registry();
        assert_eq!(queue_address(&registry), queue_address(&registry));
        assert_eq!(
            registry.thread_count(),
            0,
            "using a queue does not register"
        );
    }

    #[test]
    fn threads_and_registries_have_separate_queues() {
        let first = registry();
        let second = registry();
        assert_ne!(queue_address(&first), queue_address(&second));

        let here = queue_address(&first);
        let registry = first.clone();
        let there = thread::spawn(move || queue_address(&registry))
            .join()
            .unwrap();
        assert_ne!(here, there);
    }

    #[test]
    fn the_queue_is_not_lent_twice() {
        let registry = registry();
        let nested = registry.with_thread_queue(|_| registry.with_thread_queue(|_| ()));
        assert_eq!(nested, Some(None));
    }

    #[test]
    fn registration_counts_each_thread_once() {
        let registry = registry();
        registry.register();
        registry.register();
        assert_eq!(registry.thread_count(), 1);

        let other = registry.clone();
        thread::spawn(move || {
            other.register();
            assert_eq!(other.thread_count(), 2);
            other.remove_thread();
        })
        .join()
        .unwrap();
        assert_eq!(registry.thread_count(), 1);
    }

    #[test]
    fn removing_a_thread_withdraws_it_and_gives_it_a_fresh_queue() {
        let registry = registry();
        registry.register();
        registry.add_handle(Handle::new(0));
        registry.remove_thread();
        assert_eq!(registry.thread_count(), 0);
        registry.remove_thread();
        assert_eq!(registry.thread_count(), 0);

        let handle = Handle::new(1);
        registry.add_handle(handle.clone());
        request(&registry, 5);
        assert_eq!(
            handle.requested.load(Ordering::Relaxed),
            0,
            "the fresh queue is empty"
        );
    }

    #[test]
    fn a_thread_that_exits_is_withdrawn() {
        let registry = registry();
        let worker = registry.clone();
        thread::spawn(move || worker.register()).join().unwrap();
        assert_eq!(registry.thread_count(), 0);
    }

    #[test]
    fn workers_receive_handles_added_before_and_after_they_register() {
        let registry = registry();
        let early = Handle::new(0);
        registry.add_handle(early.clone());
        registry.register();
        let late = Handle::new(1);
        registry.add_handle(late.clone());
        assert_eq!(registry.upstream_handle_count(), 2);

        request(&registry, 1);
        request(&registry, 1);

        assert_eq!(early.requested.load(Ordering::Relaxed), 1);
        assert_eq!(late.requested.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn a_thread_that_never_registers_sees_no_handles() {
        let registry = registry();
        let handle = Handle::new(0);
        registry.add_handle(handle.clone());

        request(&registry, 10);

        assert_eq!(handle.requested.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn complete_handles_are_not_handed_to_new_workers() {
        let registry = registry();
        let complete = Handle::new(0);
        registry.add_handle(complete.clone());
        complete.complete.store(true, Ordering::Relaxed);
        registry.remove_handle();

        registry.register();

        assert_eq!(registry.upstream_handle_count(), 0);
        assert_eq!(registry.thread_count(), 1);
        assert!(registry.state.lock().handles.is_empty());
        let queued = registry.with_thread_queue(|queue| queue.cached_upstream_count());
        assert_eq!(queued, Some(0));
    }

    #[test]
    fn an_unregistered_cpu_has_no_rank() {
        let registry = registry();
        assert_eq!(registry.thread_rank(0), None);
        assert_eq!(registry.thread_rank(99), None);
    }

    #[test]
    fn a_worker_ranks_its_cpu() {
        let registry = registry();
        registry.register_with(|| Some(3));
        assert_eq!(registry.thread_rank(3), Some(1));
        assert_eq!(registry.thread_rank(1), None);
        registry.remove_thread();
        assert_eq!(registry.thread_rank(3), None);
    }

    #[test]
    fn performance_cpus_rank_first() {
        let registry = UpstreamRegistry::with_topology(8, CpuSet::from_words(&[0b0110]));
        let workers: Vec<_> = [0, 2, 5, 1]
            .into_iter()
            .map(|cpu| {
                let registry = registry.clone();
                let (registered, done) = (std::sync::mpsc::channel(), std::sync::mpsc::channel());
                let worker = thread::spawn(move || {
                    registry.register_with(|| Some(cpu));
                    registered.0.send(()).unwrap();
                    done.1.recv().unwrap();
                });
                registered.1.recv().unwrap();
                (worker, done.0)
            })
            .collect();

        let ranks: Vec<_> = (0..8).map(|cpu| registry.thread_rank(cpu)).collect();
        assert_eq!(
            ranks,
            [Some(3), Some(1), Some(2), None, None, Some(4), None, None]
        );

        for (worker, done) in workers {
            done.send(()).unwrap();
            worker.join().unwrap();
        }
        assert!((0..8).all(|cpu| registry.thread_rank(cpu).is_none()));
    }

    #[test]
    fn workers_without_a_known_cpu_count_but_have_no_rank() {
        let registry = registry();
        registry.register_with(|| None);
        assert_eq!(registry.thread_count(), 1);
        assert!((0..8).all(|cpu| registry.thread_rank(cpu).is_none()));
    }

    #[test]
    fn a_dropped_registry_leaves_no_queue_behind() {
        let registry = registry();
        registry.register();
        drop(registry);
        let next = self::registry();
        next.register();
        let stale = THREAD_QUEUES.with(|queues| {
            queues
                .borrow()
                .iter()
                .filter(|queue| queue.registry.strong_count() == 0)
                .count()
        });
        assert_eq!(stale, 0);
    }
}
