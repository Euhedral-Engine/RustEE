use std::panic::{self, AssertUnwindSafe};
#[allow(clippy::disallowed_types, reason = "HOT-FIX(C20)")]
use std::sync::mpsc::{self, Receiver, Sender};
#[allow(clippy::disallowed_types, reason = "HOT-FIX(C20)")]
use std::sync::{Arc, Mutex, PoisonError};
use std::thread;

type Task = Box<dyn FnOnce() + Send>;

/// A fixed set of named threads running submitted tasks in submission order.
///
/// A panicking task is logged and the thread moves on. [`shutdown`](Self::shutdown) stops taking
/// tasks and lets the queued ones run.
pub(crate) struct TaskPool {
    #[allow(clippy::disallowed_types, reason = "HOT-FIX(C20)")]
    sender: Mutex<Option<Sender<Task>>>,
}

impl TaskPool {
    /// A pool of `threads` threads (at least one) named `name`.
    #[allow(clippy::disallowed_types, reason = "HOT-FIX(C20)")]
    pub(crate) fn new(name: &str, threads: usize) -> Self {
        let (sender, receiver) = mpsc::channel::<Task>();
        let receiver = Arc::new(Mutex::new(receiver));
        for _ in 0..threads.max(1) {
            let receiver = receiver.clone();
            let spawned = thread::Builder::new()
                .name(name.to_string())
                .spawn(move || work(&receiver));
            if let Err(error) = spawned {
                log::error!("could not start a {name} thread: {error}");
            }
        }
        Self {
            sender: Mutex::new(Some(sender)),
        }
    }

    /// Queues `task`, or hands it back once the pool is shut down.
    pub(crate) fn execute(&self, task: impl FnOnce() + Send + 'static) -> Result<(), Task> {
        let sender = self.sender.lock().unwrap_or_else(PoisonError::into_inner);
        match sender.as_ref() {
            Some(sender) => sender.send(Box::new(task)).map_err(|rejected| rejected.0),
            None => Err(Box::new(task)),
        }
    }

    /// Runs `task` on the pool, or on the calling thread once the pool is shut down.
    pub(crate) fn execute_or_run(&self, task: impl FnOnce() + Send + 'static) {
        if let Err(task) = self.execute(task) {
            task();
        }
    }

    /// Stops taking tasks; the queued ones still run.
    pub(crate) fn shutdown(&self) {
        self.sender
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take();
    }
}

impl Drop for TaskPool {
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[allow(clippy::disallowed_types, reason = "HOT-FIX(C20)")]
fn work(receiver: &Mutex<Receiver<Task>>) {
    loop {
        let task = receiver
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .recv();
        let Ok(task) = task else {
            return;
        };
        if panic::catch_unwind(AssertUnwindSafe(task)).is_err() {
            log::error!("a pooled task panicked");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Barrier;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::time::Duration;

    #[test]
    fn runs_tasks_on_its_named_threads() {
        let pool = TaskPool::new("pool-test", 2);
        let (sender, receiver) = mpsc::channel();
        assert!(
            pool.execute(move || sender
                .send(thread::current().name().map(str::to_owned))
                .unwrap())
                .is_ok()
        );
        assert_eq!(receiver.recv().unwrap().as_deref(), Some("pool-test"));
    }

    #[test]
    fn runs_tasks_concurrently_up_to_its_size() {
        let pool = TaskPool::new("pool-test", 3);
        let barrier = Arc::new(Barrier::new(4));
        for _ in 0..3 {
            let barrier = barrier.clone();
            assert!(
                pool.execute(move || {
                    barrier.wait();
                })
                .is_ok()
            );
        }
        barrier.wait();
    }

    #[test]
    fn a_panicking_task_does_not_stop_the_pool() {
        let pool = TaskPool::new("pool-test", 1);
        assert!(pool.execute(|| panic!("task failed")).is_ok());
        let (sender, receiver) = mpsc::channel();
        assert!(pool.execute(move || sender.send(()).unwrap()).is_ok());
        receiver.recv_timeout(Duration::from_secs(5)).unwrap();
    }

    #[test]
    fn shutdown_runs_queued_tasks_and_rejects_new_ones() {
        let pool = TaskPool::new("pool-test", 1);
        let (gate, open) = mpsc::channel::<()>();
        let ran = Arc::new(AtomicUsize::new(0));
        assert!(pool.execute(move || open.recv().unwrap()).is_ok());
        let (counter, (done, finished)) = (ran.clone(), mpsc::channel());
        assert!(
            pool.execute(move || {
                counter.fetch_add(1, Ordering::SeqCst);
                done.send(()).unwrap();
            })
            .is_ok()
        );

        pool.shutdown();
        assert!(pool.execute(|| {}).is_err());
        gate.send(()).unwrap();
        finished.recv_timeout(Duration::from_secs(5)).unwrap();
        assert_eq!(ran.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn a_shut_down_pool_runs_tasks_inline() {
        let pool = TaskPool::new("pool-test", 1);
        pool.shutdown();
        let ran = Arc::new(AtomicBool::new(false));
        let flag = ran.clone();
        pool.execute_or_run(move || flag.store(true, Ordering::SeqCst));
        assert!(ran.load(Ordering::SeqCst));
    }
}
