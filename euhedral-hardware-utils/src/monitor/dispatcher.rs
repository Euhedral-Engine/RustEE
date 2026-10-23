//! Delivers published values to listeners on a dedicated thread (upstream `LatestValueDispatcher`).
//!
//! At most one value is pending: a newer offer replaces an undelivered one, so a slow listener sees
//! the latest value rather than a backlog. One value is delivered to every listener before the next
//! is taken.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::thread::{self, ThreadId};

/// Identifies a registered listener, for
/// [`ResourceMonitor::remove_listener`](super::ResourceMonitor::remove_listener).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ListenerId(u64);

type Listener<T> = Arc<dyn Fn(&T) + Send + Sync>;
type CloseHook = Box<dyn FnOnce() + Send>;

struct State<T> {
    listeners: Arc<[(ListenerId, Listener<T>)]>,
    next_id: u64,
    pending: Option<T>,
    closing: bool,
    closed: bool,
    close_hook: Option<CloseHook>,
}

struct Shared<T> {
    state: Mutex<State<T>>,
    changed: Condvar,
}

impl<T> Shared<T> {
    fn lock(&self) -> MutexGuard<'_, State<T>> {
        // Listeners run outside the lock, so a poisoned lock can only come from a panic in this
        // module's own bookkeeping, which leaves the state consistent.
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

pub(crate) struct LatestValueDispatcher<T> {
    shared: Arc<Shared<T>>,
    thread: ThreadId,
}

impl<T: Send + 'static> LatestValueDispatcher<T> {
    /// Starts the dispatch thread.
    pub(crate) fn new(name: &str) -> std::io::Result<Self> {
        let shared = Arc::new(Shared {
            state: Mutex::new(State {
                listeners: Arc::from([]),
                next_id: 0,
                pending: None,
                closing: false,
                closed: false,
                close_hook: None,
            }),
            changed: Condvar::new(),
        });
        let worker = Arc::clone(&shared);
        let handle = thread::Builder::new()
            .name(name.to_owned())
            .spawn(move || dispatch_loop(&worker))?;
        Ok(Self {
            shared,
            thread: handle.thread().id(),
        })
    }

    /// Registers a listener; ignored (but still given an id) once closing has begun.
    pub(crate) fn add_listener(&self, listener: impl Fn(&T) + Send + Sync + 'static) -> ListenerId {
        let mut state = self.shared.lock();
        let id = ListenerId(state.next_id);
        state.next_id += 1;
        if !state.closing {
            let mut listeners = state.listeners.to_vec();
            listeners.push((id, Arc::new(listener)));
            state.listeners = listeners.into();
        }
        id
    }

    /// Unregisters a listener. A delivery already in progress may still reach it once.
    pub(crate) fn remove_listener(&self, id: ListenerId) -> bool {
        let mut state = self.shared.lock();
        let before = state.listeners.len();
        let listeners: Vec<_> = state
            .listeners
            .iter()
            .filter(|(existing, _)| *existing != id)
            .cloned()
            .collect();
        state.listeners = listeners.into();
        state.listeners.len() != before
    }

    /// Replaces the pending value; dropped once closing has begun.
    pub(crate) fn offer(&self, value: T) {
        let mut state = self.shared.lock();
        if !state.closing {
            state.pending = Some(value);
            self.shared.changed.notify_all();
        }
    }

    /// Starts shutdown. The first call's hook runs on the dispatch thread after the last delivery,
    /// once [`is_closed`](Self::is_closed) reports true.
    pub(crate) fn begin_close(&self, hook: Option<CloseHook>) {
        let mut state = self.shared.lock();
        if !state.closing {
            state.closing = true;
            state.close_hook = hook;
            self.shared.changed.notify_all();
        }
    }

    /// Whether the dispatch thread has finished.
    pub(crate) fn is_closed(&self) -> bool {
        self.shared.lock().closed
    }

    /// Whether the caller is the dispatch thread, that is, inside a listener.
    pub(crate) fn is_dispatch_thread(&self) -> bool {
        thread::current().id() == self.thread
    }
}

impl<T> Drop for LatestValueDispatcher<T> {
    fn drop(&mut self) {
        let mut state = self.shared.lock();
        state.closing = true;
        self.shared.changed.notify_all();
    }
}

fn dispatch_loop<T>(shared: &Shared<T>) {
    loop {
        let (value, listeners) = {
            let mut state = shared.lock();
            while !state.closing && state.pending.is_none() {
                state = shared
                    .changed
                    .wait(state)
                    .unwrap_or_else(PoisonError::into_inner);
            }
            if state.closing {
                break;
            }
            (
                state.pending.take().expect("a value is pending"),
                Arc::clone(&state.listeners),
            )
        };
        for (_, listener) in listeners.iter() {
            // One listener's panic must not stop delivery to the others or kill the thread.
            if catch_unwind(AssertUnwindSafe(|| listener(&value))).is_err() {
                log::warn!("a resource monitor listener panicked");
            }
        }
    }
    let hook = {
        let mut state = shared.lock();
        state.listeners = Arc::from([]);
        state.pending = None;
        state.closed = true;
        shared.changed.notify_all();
        state.close_hook.take()
    };
    if let Some(hook) = hook
        && catch_unwind(AssertUnwindSafe(hook)).is_err()
    {
        log::warn!("a resource monitor close hook panicked");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::mpsc;
    use std::time::{Duration, Instant};

    fn wait_until(condition: impl Fn() -> bool) {
        let deadline = Instant::now() + Duration::from_secs(5);
        while !condition() {
            assert!(Instant::now() < deadline, "timed out");
            thread::sleep(Duration::from_millis(1));
        }
    }

    #[test]
    fn delivers_to_every_listener_and_isolates_panics() {
        let dispatcher = LatestValueDispatcher::new("test-dispatcher").unwrap();
        let (sender, receiver) = mpsc::channel();
        dispatcher.add_listener(|_: &u32| panic!("listener failure"));
        let sender = Mutex::new(sender);
        dispatcher.add_listener(move |value: &u32| sender.lock().unwrap().send(*value).unwrap());
        dispatcher.offer(7);
        assert_eq!(receiver.recv_timeout(Duration::from_secs(5)).unwrap(), 7);
        dispatcher.offer(8);
        assert_eq!(receiver.recv_timeout(Duration::from_secs(5)).unwrap(), 8);
    }

    #[test]
    fn coalesces_to_the_latest_value_while_a_listener_is_busy() {
        let dispatcher = LatestValueDispatcher::new("test-dispatcher").unwrap();
        let (entered, entered_rx) = mpsc::channel();
        let (release, release_rx) = mpsc::channel::<()>();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let recorded = Arc::clone(&seen);
        let (entered, release_rx) = (Mutex::new(entered), Mutex::new(release_rx));
        dispatcher.add_listener(move |value: &u32| {
            recorded.lock().unwrap().push(*value);
            if *value == 1 {
                entered.lock().unwrap().send(()).unwrap();
                release_rx.lock().unwrap().recv().unwrap();
            }
        });
        dispatcher.offer(1);
        entered_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        for value in 2..=5 {
            dispatcher.offer(value);
        }
        release.send(()).unwrap();
        wait_until(|| seen.lock().unwrap().len() == 2);
        thread::sleep(Duration::from_millis(20));
        assert_eq!(*seen.lock().unwrap(), [1, 5]);
    }

    #[test]
    fn removed_listeners_stop_receiving() {
        let dispatcher = LatestValueDispatcher::new("test-dispatcher").unwrap();
        let count = Arc::new(AtomicUsize::new(0));
        let counted = Arc::clone(&count);
        let id = dispatcher.add_listener(move |_: &u32| {
            counted.fetch_add(1, Ordering::SeqCst);
        });
        dispatcher.offer(1);
        wait_until(|| count.load(Ordering::SeqCst) == 1);
        assert!(dispatcher.remove_listener(id));
        assert!(!dispatcher.remove_listener(id));
        dispatcher.offer(2);
        thread::sleep(Duration::from_millis(20));
        assert_eq!(count.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn close_runs_the_hook_once_after_the_thread_is_done() {
        let dispatcher = Arc::new(LatestValueDispatcher::<u32>::new("test-dispatcher").unwrap());
        let ran = Arc::new(AtomicUsize::new(0));
        let closed_when_run = Arc::new(AtomicBool::new(false));
        let (hook_ran, hook_closed, observed) = (
            Arc::clone(&ran),
            Arc::clone(&closed_when_run),
            Arc::clone(&dispatcher),
        );
        dispatcher.begin_close(Some(Box::new(move || {
            hook_closed.store(observed.is_closed(), Ordering::SeqCst);
            hook_ran.fetch_add(1, Ordering::SeqCst);
        })));
        dispatcher.begin_close(Some(Box::new(|| panic!("second hook must not run"))));
        wait_until(|| ran.load(Ordering::SeqCst) == 1);
        assert!(dispatcher.is_closed());
        assert!(closed_when_run.load(Ordering::SeqCst));
        dispatcher.offer(1);
        assert!(!dispatcher.is_dispatch_thread());
    }

    #[test]
    fn listeners_can_tell_they_run_on_the_dispatch_thread() {
        let dispatcher = Arc::new(LatestValueDispatcher::new("test-dispatcher").unwrap());
        let inside = Arc::new(AtomicBool::new(false));
        let (seen, observed) = (Arc::clone(&inside), Arc::downgrade(&dispatcher));
        dispatcher.add_listener(move |_: &u32| {
            if let Some(dispatcher) = observed.upgrade() {
                seen.store(dispatcher.is_dispatch_thread(), Ordering::SeqCst);
            }
        });
        dispatcher.offer(1);
        wait_until(|| inside.load(Ordering::SeqCst));
    }
}
