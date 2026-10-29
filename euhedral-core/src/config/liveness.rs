use euhedral_data_structures::sync::LifecycleMutex;
use euhedral_hardware_utils::util::CpuSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::{self, Thread};
use std::time::{Duration, Instant};

/// The longest single wait while [`CloneLivenessRegistry::await_termination`] polls a worker.
const POLL: Duration = Duration::from_micros(5);

/// Publishes which worker threads are running for one clone and its optional SMT buddy, one slot per
/// participating logical CPU.
#[derive(Debug)]
pub struct CloneLivenessRegistry {
    cpus: CpuSet,
    /// Indexed by logical CPU id; `None` for ids outside `cpus`.
    slots: Box<[Option<WorkerSlot>]>,
}

impl CloneLivenessRegistry {
    pub fn new(effective_cpus: &CpuSet) -> Self {
        let mut slots: Vec<Option<WorkerSlot>> = (0..effective_cpus.span()).map(|_| None).collect();
        for cpu in effective_cpus.iter() {
            slots[cpu] = Some(WorkerSlot::default());
        }
        Self {
            cpus: effective_cpus.clone(),
            slots: slots.into(),
        }
    }

    /// The CPUs this registry has slots for.
    pub fn cpus(&self) -> &CpuSet {
        &self.cpus
    }

    /// The publication slot of one participating logical CPU.
    pub fn slot(&self, cpu: usize) -> Option<&WorkerSlot> {
        self.slots.get(cpu).and_then(Option::as_ref)
    }

    /// A consistent snapshot of every participating worker, in CPU order.
    pub fn workers(&self) -> Vec<Worker> {
        self.participants().map(WorkerSlot::snapshot).collect()
    }

    /// Waits until every participating worker has stopped running. False if `deadline` passes
    /// first.
    pub fn await_termination(&self, deadline: Instant) -> bool {
        for slot in self.participants() {
            while slot.running() {
                let now = Instant::now();
                if now >= deadline {
                    return false;
                }
                #[allow(clippy::disallowed_methods, reason = "LIFECYCLE(C29)")]
                thread::sleep(POLL.min(deadline - now));
            }
        }
        true
    }

    fn participants(&self) -> impl Iterator<Item = &WorkerSlot> {
        self.slots.iter().flatten()
    }
}

/// A worker as seen at one instant: its thread while it runs, and whether it runs.
#[derive(Clone, Debug)]
pub struct Worker {
    pub thread: Option<Thread>,
    pub running: bool,
}

/// Where one worker publishes its entry and exit.
#[derive(Debug, Default)]
pub struct WorkerSlot {
    thread: LifecycleMutex<Option<Thread>>,
    running: AtomicBool,
}

impl WorkerSlot {
    /// The running worker's thread.
    pub fn thread(&self) -> Option<Thread> {
        self.lock().clone()
    }

    pub fn running(&self) -> bool {
        self.running.load(Ordering::Acquire)
    }

    /// Publishes `worker` as running.
    pub fn enter(&self, worker: Thread) {
        let mut thread = self.lock();
        *thread = Some(worker);
        self.running.store(true, Ordering::Release);
    }

    /// Publishes that the worker has stopped.
    pub fn exit(&self) {
        let mut thread = self.lock();
        *thread = None;
        self.running.store(false, Ordering::Release);
    }

    fn snapshot(&self) -> Worker {
        let thread = self.lock();
        Worker {
            running: thread.is_some(),
            thread: thread.clone(),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Option<Thread>> {
        self.thread.lock()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Barrier};

    fn cpus(ids: &[usize]) -> CpuSet {
        ids.iter().copied().collect()
    }

    #[test]
    fn only_participating_cpus_have_slots() {
        let registry = CloneLivenessRegistry::new(&cpus(&[1, 3]));
        assert_eq!(registry.cpus(), &cpus(&[1, 3]));
        assert!(registry.slot(0).is_none());
        assert!(registry.slot(1).is_some());
        assert!(registry.slot(2).is_none());
        assert!(registry.slot(3).is_some());
        assert!(registry.slot(4).is_none());
        assert!(registry.slot(usize::MAX).is_none());
    }

    #[test]
    fn an_empty_registry_has_no_workers_and_is_terminated() {
        let registry = CloneLivenessRegistry::new(&CpuSet::new());
        assert!(registry.workers().is_empty());
        assert!(registry.await_termination(Instant::now()));
    }

    #[test]
    fn workers_publish_entry_and_exit() {
        let registry = CloneLivenessRegistry::new(&cpus(&[0, 2]));
        let workers = registry.workers();
        assert_eq!(workers.len(), 2);
        assert!(
            workers
                .iter()
                .all(|worker| !worker.running && worker.thread.is_none())
        );

        let slot = registry.slot(2).unwrap();
        slot.enter(thread::current());
        assert!(slot.running());
        assert_eq!(slot.thread().unwrap().id(), thread::current().id());
        let workers = registry.workers();
        assert!(!workers[0].running);
        assert!(workers[1].running);
        assert_eq!(
            workers[1].thread.as_ref().unwrap().id(),
            thread::current().id()
        );

        slot.exit();
        assert!(!slot.running());
        assert!(slot.thread().is_none());
    }

    #[test]
    fn await_termination_times_out_while_a_worker_runs() {
        let registry = CloneLivenessRegistry::new(&cpus(&[0]));
        registry.slot(0).unwrap().enter(thread::current());
        let start = Instant::now();
        assert!(!registry.await_termination(start + Duration::from_millis(20)));
        assert!(start.elapsed() >= Duration::from_millis(20));
        assert!(!registry.await_termination(start));
    }

    #[test]
    fn await_termination_returns_once_every_worker_exits() {
        let registry = Arc::new(CloneLivenessRegistry::new(&cpus(&[0, 1])));
        let entered = Arc::new(Barrier::new(3));
        let workers: Vec<_> = [0, 1]
            .into_iter()
            .map(|cpu| {
                let registry = registry.clone();
                let entered = entered.clone();
                thread::spawn(move || {
                    let slot = registry.slot(cpu).unwrap();
                    slot.enter(thread::current());
                    entered.wait();
                    thread::sleep(Duration::from_millis(10));
                    slot.exit();
                })
            })
            .collect();

        entered.wait();
        assert!(registry.workers().iter().all(|worker| worker.running));
        assert!(registry.await_termination(Instant::now() + Duration::from_secs(10)));
        assert!(registry.workers().iter().all(|worker| !worker.running));
        for worker in workers {
            worker.join().unwrap();
        }
    }

    #[test]
    fn a_snapshot_never_shows_a_running_worker_without_its_thread() {
        let registry = Arc::new(CloneLivenessRegistry::new(&cpus(&[0])));
        let flipper = {
            let registry = registry.clone();
            thread::spawn(move || {
                let slot = registry.slot(0).unwrap();
                for _ in 0..10_000 {
                    slot.enter(thread::current());
                    slot.exit();
                }
            })
        };
        while !flipper.is_finished() {
            for worker in registry.workers() {
                assert_eq!(worker.running, worker.thread.is_some());
            }
        }
        flipper.join().unwrap();
    }
}
