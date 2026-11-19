//! Delivers published values to listeners on a dedicated thread (upstream `LatestValueDispatcher`).
//!
//! At most one value is pending: a newer offer replaces an undelivered one, so a slow listener sees
//! the latest value rather than a backlog. One value is delivered to every listener before the next
//! is taken.
//!
//! The offer path takes no lock and waits for nothing. A pointer cell holds the pending value: the
//! offerer swaps its box in and drops what it swapped out; the dispatch thread swaps the cell empty
//! and owns what it took, so every box has one owner at a time. The listener list is the dispatch
//! thread's own: registration edits a master list under a lifecycle lock (cold) and hands the new
//! list over through a second pointer cell, which the dispatch thread takes before each delivery.
//! An idle dispatch thread parks; an offer or a close unparks it, and the park token is sticky, so
//! a wake-up that comes before the park is not lost.

use euhedral_data_structures::sync::LifecycleMutex;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::ptr;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicPtr, Ordering};
use std::thread::{self, Thread, ThreadId};

/// Identifies a registered listener, for
/// [`ResourceMonitor::remove_listener`](super::ResourceMonitor::remove_listener).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ListenerId(u64);

type Listener<T> = Arc<dyn Fn(&T) + Send + Sync>;
type Listeners<T> = Arc<[(ListenerId, Listener<T>)]>;
type CloseHook = Box<dyn FnOnce() + Send>;

/// The cold state: the master listener list and the close hook.
struct Registry<T> {
    listeners: Listeners<T>,
    next_id: u64,
    close_hook: Option<CloseHook>,
}

struct Shared<T> {
    /// The pending value; a null pointer when there is none. Whoever swaps a pointer out owns it.
    mailbox: AtomicPtr<T>,
    /// The latest listener list the dispatch thread has not taken yet, or null.
    next_listeners: AtomicPtr<Listeners<T>>,
    closing: AtomicBool,
    closed: AtomicBool,
    registry: LifecycleMutex<Registry<T>>,
}

impl<T> Shared<T> {
    /// Takes the pending value, if any.
    fn take_value(&self) -> Option<Box<T>> {
        let raw = self.mailbox.swap(ptr::null_mut(), Ordering::AcqRel);
        // SAFETY: a non-null pointer in the cell came from `Box::into_raw` in `offer`, and the swap
        // made this thread its only owner.
        (!raw.is_null()).then(|| unsafe { Box::from_raw(raw) })
    }

    /// Takes the handed-over listener list, if any.
    fn take_listeners(&self) -> Option<Listeners<T>> {
        let raw = self.next_listeners.swap(ptr::null_mut(), Ordering::AcqRel);
        // SAFETY: as for `take_value`, with the pointer made in `publish_listeners`.
        (!raw.is_null()).then(|| *unsafe { Box::from_raw(raw) })
    }

    /// Hands a new list over, dropping one the dispatch thread never took: the new list holds
    /// everything the old one did. Called with the registry held, so lists arrive in order.
    fn publish_listeners(&self, listeners: Listeners<T>) {
        let raw = Box::into_raw(Box::new(listeners));
        let old = self.next_listeners.swap(raw, Ordering::AcqRel);
        if !old.is_null() {
            // SAFETY: the swap made this thread the owner of the superseded box.
            drop(unsafe { Box::from_raw(old) });
        }
    }
}

impl<T> Drop for Shared<T> {
    fn drop(&mut self) {
        drop(self.take_value());
        drop(self.take_listeners());
    }
}

pub(crate) struct LatestValueDispatcher<T> {
    shared: Arc<Shared<T>>,
    thread: Thread,
}

impl<T: Send + 'static> LatestValueDispatcher<T> {
    /// Starts the dispatch thread.
    pub(crate) fn new(name: &str) -> std::io::Result<Self> {
        let shared = Arc::new(Shared {
            mailbox: AtomicPtr::new(ptr::null_mut()),
            next_listeners: AtomicPtr::new(ptr::null_mut()),
            closing: AtomicBool::new(false),
            closed: AtomicBool::new(false),
            registry: LifecycleMutex::new(Registry {
                listeners: Arc::from([]),
                next_id: 0,
                close_hook: None,
            }),
        });
        let worker = Arc::clone(&shared);
        let handle = thread::Builder::new()
            .name(name.to_owned())
            .spawn(move || dispatch_loop(&worker))?;
        Ok(Self {
            shared,
            thread: handle.thread().clone(),
        })
    }

    /// Registers a listener; ignored (but still given an id) once closing has begun.
    pub(crate) fn add_listener(&self, listener: impl Fn(&T) + Send + Sync + 'static) -> ListenerId {
        let mut registry = self.shared.registry.lock();
        let id = ListenerId(registry.next_id);
        registry.next_id += 1;
        if !self.shared.closing.load(Ordering::SeqCst) {
            let mut listeners = registry.listeners.to_vec();
            listeners.push((id, Arc::new(listener)));
            registry.listeners = listeners.into();
            self.shared
                .publish_listeners(Arc::clone(&registry.listeners));
        }
        id
    }

    /// Unregisters a listener. A delivery already in progress may still reach it once.
    pub(crate) fn remove_listener(&self, id: ListenerId) -> bool {
        let mut registry = self.shared.registry.lock();
        let before = registry.listeners.len();
        let listeners: Vec<_> = registry
            .listeners
            .iter()
            .filter(|(existing, _)| *existing != id)
            .cloned()
            .collect();
        registry.listeners = listeners.into();
        let removed = registry.listeners.len() != before;
        if removed {
            self.shared
                .publish_listeners(Arc::clone(&registry.listeners));
        }
        removed
    }

    /// Replaces the pending value; dropped once closing has begun.
    pub(crate) fn offer(&self, value: T) {
        if self.shared.closing.load(Ordering::SeqCst) {
            return;
        }
        let raw = Box::into_raw(Box::new(value));
        let old = self.shared.mailbox.swap(raw, Ordering::AcqRel);
        if !old.is_null() {
            // SAFETY: the swap made this thread the owner of the replaced value.
            drop(unsafe { Box::from_raw(old) });
        }
        self.thread.unpark();
    }

    /// Starts shutdown. The first call's hook runs on the dispatch thread after the last delivery,
    /// once [`is_closed`](Self::is_closed) reports true.
    pub(crate) fn begin_close(&self, hook: Option<CloseHook>) {
        self.close(hook);
    }

    fn close(&self, hook: Option<CloseHook>) {
        {
            // The hook is stored before `closing` is set, so the dispatch thread finds it.
            let mut registry = self.shared.registry.lock();
            if self.shared.closing.load(Ordering::SeqCst) {
                return;
            }
            registry.close_hook = hook;
            self.shared.closing.store(true, Ordering::SeqCst);
        }
        self.thread.unpark();
    }

    /// Whether the dispatch thread has finished.
    pub(crate) fn is_closed(&self) -> bool {
        self.shared.closed.load(Ordering::SeqCst)
    }

    /// Whether the caller is the dispatch thread, that is, inside a listener.
    pub(crate) fn is_dispatch_thread(&self) -> bool {
        thread::current().id() == self.thread_id()
    }

    fn thread_id(&self) -> ThreadId {
        self.thread.id()
    }
}

impl<T> Drop for LatestValueDispatcher<T> {
    fn drop(&mut self) {
        {
            let mut registry = self.shared.registry.lock();
            if !self.shared.closing.load(Ordering::SeqCst) {
                registry.close_hook = None;
                self.shared.closing.store(true, Ordering::SeqCst);
            }
        }
        self.thread.unpark();
    }
}

fn dispatch_loop<T>(shared: &Shared<T>) {
    let mut listeners: Listeners<T> = Arc::from([]);
    loop {
        if shared.closing.load(Ordering::SeqCst) {
            break;
        }
        if let Some(value) = shared.take_value() {
            if let Some(next) = shared.take_listeners() {
                listeners = next;
            }
            for (_, listener) in listeners.iter() {
                // One listener's panic must not stop delivery to the others or kill the thread.
                if catch_unwind(AssertUnwindSafe(|| listener(&value))).is_err() {
                    log::warn!("a resource monitor listener panicked");
                }
            }
        } else {
            // Idle: an offer or a close unparks this thread, and the token is sticky, so a wake-up
            // between the take above and this park is not lost. A spurious return loops.
            #[allow(clippy::disallowed_methods, reason = "JAVA-BACKOFF")]
            thread::park();
        }
    }
    drop(shared.take_value());
    drop(shared.take_listeners());
    drop(listeners);
    let (hook, registered) = {
        let mut registry = shared.registry.lock();
        (
            registry.close_hook.take(),
            std::mem::replace(&mut registry.listeners, Arc::from([])),
        )
    };
    drop(registered);
    shared.closed.store(true, Ordering::SeqCst);
    if let Some(hook) = hook
        && catch_unwind(AssertUnwindSafe(hook)).is_err()
    {
        log::warn!("a resource monitor close hook panicked");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use euhedral_data_structures::sync::HotScope;
    #[allow(clippy::disallowed_types, reason = "test fixture")]
    use std::sync::Mutex;
    use std::sync::atomic::AtomicUsize;
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

    /// A value that counts its drops.
    struct Counted {
        id: u32,
        drops: Arc<AtomicUsize>,
    }

    impl Drop for Counted {
        fn drop(&mut self) {
            self.drops.fetch_add(1, Ordering::SeqCst);
        }
    }

    /// A listener that blocks inside its first delivery until released.
    fn gated_recorder(
        dispatcher: &LatestValueDispatcher<u32>,
    ) -> (Arc<Mutex<Vec<u32>>>, mpsc::Receiver<()>, mpsc::Sender<()>) {
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
        (seen, entered_rx, release)
    }

    #[test]
    fn offers_racing_close_deliver_the_latest_or_nothing_after_close() {
        let drops = Arc::new(AtomicUsize::new(0));
        let dispatcher = LatestValueDispatcher::new("test-dispatcher").unwrap();
        let (entered, entered_rx) = mpsc::channel();
        let (release, release_rx) = mpsc::channel::<()>();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let recorded = Arc::clone(&seen);
        let (entered, release_rx) = (Mutex::new(entered), Mutex::new(release_rx));
        dispatcher.add_listener(move |value: &Counted| {
            recorded.lock().unwrap().push(value.id);
            if value.id == 1 {
                entered.lock().unwrap().send(()).unwrap();
                release_rx.lock().unwrap().recv().unwrap();
            }
        });
        let value = |id| Counted {
            id,
            drops: Arc::clone(&drops),
        };
        dispatcher.offer(value(1));
        entered_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        dispatcher.offer(value(2));
        dispatcher.offer(value(3));
        dispatcher.begin_close(None);
        // Offered after the close: dropped by the offer itself.
        dispatcher.offer(value(4));
        assert_eq!(
            drops.load(Ordering::SeqCst),
            2,
            "values 2 and 4 are dropped"
        );
        release.send(()).unwrap();
        wait_until(|| dispatcher.is_closed());
        assert_eq!(*seen.lock().unwrap(), [1]);
        dispatcher.offer(value(5));
        assert_eq!(
            drops.load(Ordering::SeqCst),
            5,
            "every value is dropped once"
        );
    }

    #[test]
    fn a_burst_of_offers_delivers_the_last() {
        let drops = Arc::new(AtomicUsize::new(0));
        let dispatcher = LatestValueDispatcher::new("test-dispatcher").unwrap();
        let (entered, entered_rx) = mpsc::channel();
        let (release, release_rx) = mpsc::channel::<()>();
        let (seen_tx, seen_rx) = mpsc::channel();
        let (entered, release_rx, seen_tx) = (
            Mutex::new(entered),
            Mutex::new(release_rx),
            Mutex::new(seen_tx),
        );
        dispatcher.add_listener(move |value: &Counted| {
            seen_tx.lock().unwrap().send(value.id).unwrap();
            if value.id == 1 {
                entered.lock().unwrap().send(()).unwrap();
                release_rx.lock().unwrap().recv().unwrap();
            }
        });
        let value = |id| Counted {
            id,
            drops: Arc::clone(&drops),
        };
        dispatcher.offer(value(1));
        entered_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        for id in 2..=50 {
            dispatcher.offer(value(id));
        }
        assert_eq!(
            drops.load(Ordering::SeqCst),
            48,
            "each replaced value drops once"
        );
        release.send(()).unwrap();
        let wait = Duration::from_secs(5);
        assert_eq!(seen_rx.recv_timeout(wait).unwrap(), 1);
        assert_eq!(seen_rx.recv_timeout(wait).unwrap(), 50);
        wait_until(|| drops.load(Ordering::SeqCst) == 50);
        assert!(seen_rx.try_recv().is_err(), "nothing else is delivered");
    }

    #[test]
    fn listeners_added_during_dispatch_see_the_next_value() {
        let dispatcher = Arc::new(LatestValueDispatcher::new("test-dispatcher").unwrap());
        let (late_tx, late_rx) = mpsc::channel();
        let late_tx = Arc::new(Mutex::new(late_tx));
        let weak = Arc::downgrade(&dispatcher);
        let (done, done_rx) = mpsc::channel();
        let done = Mutex::new(done);
        dispatcher.add_listener(move |value: &u32| {
            if *value == 1 {
                let sender = Arc::clone(&late_tx);
                weak.upgrade().unwrap().add_listener(move |value: &u32| {
                    sender.lock().unwrap().send(*value).unwrap();
                });
            }
            done.lock().unwrap().send(*value).unwrap();
        });
        dispatcher.offer(1);
        assert_eq!(done_rx.recv_timeout(Duration::from_secs(5)).unwrap(), 1);
        assert!(
            late_rx.try_recv().is_err(),
            "the new listener missed value 1"
        );
        dispatcher.offer(2);
        assert_eq!(late_rx.recv_timeout(Duration::from_secs(5)).unwrap(), 2);
    }

    #[test]
    fn a_removed_listener_is_gone_for_the_next_value_and_the_list_is_freed() {
        let dispatcher = LatestValueDispatcher::<u32>::new("test-dispatcher").unwrap();
        let (seen, entered_rx, release) = gated_recorder(&dispatcher);
        let token = Arc::new(());
        let held = Arc::clone(&token);
        let id = dispatcher.add_listener(move |_: &u32| {
            let _ = &held;
        });
        dispatcher.offer(1);
        entered_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        // The list with the second listener is handed over but not yet taken.
        assert!(dispatcher.remove_listener(id));
        release.send(()).unwrap();
        dispatcher.offer(2);
        wait_until(|| seen.lock().unwrap().len() == 2);
        assert_eq!(
            Arc::strong_count(&token),
            1,
            "the removed listener is dropped"
        );
    }

    #[test]
    fn an_offer_takes_no_lock() {
        let dispatcher = LatestValueDispatcher::<u32>::new("test-dispatcher").unwrap();
        let _hot = HotScope::enter();
        dispatcher.offer(1);
        dispatcher.offer(2);
    }
}
