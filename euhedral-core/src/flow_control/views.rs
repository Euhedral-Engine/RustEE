//! Values the lattice's workers read on every frame, handed to them through queues.
//!
//! A *cell* is a slot of lattice-wide state that changes only when the lattice is rebuilt: a vertex's routing
//! generation, an edge's parent, a CPU's cache. The writer ([`set_cell`](UpstreamRegistry::set_cell), lifecycle work)
//! keeps the current value under a lifecycle lock and sends it, through a queue, to the mailbox of every registered
//! reader. A reader (a worker, or any thread that [`register_reader`](UpstreamRegistry::register_reader)s) drains its
//! mailbox in one batch at the start of each cycle and applies the values to a **view** only it reads. A frame reads the
//! view: no atomic, no shared line, no reference count. The value it replaced is dropped by the reader; the last
//! `Arc` to go ends the generation. Nothing is reclaimed by an epoch or a barrier.
//!
//! A thread that is not a registered reader reads the current value under the lock, once per call: a slow path
//! for threads that are not on a frame path.

use super::registry::UpstreamRegistry;
use euhedral_data_structures::queues::{Consumer, MpscQueue, Multi, Producer, Single};
use euhedral_data_structures::sync::{HotScope, LifecycleMutex};
use std::any::Any;
use std::cell::RefCell;
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Weak};

type Value = Arc<dyn Any + Send + Sync>;

/// A slot of lattice state of type `T`, made by [`UpstreamRegistry::cell`].
pub struct Cell<T> {
    id: usize,
    _marker: PhantomData<fn() -> T>,
}

impl<T> std::fmt::Debug for Cell<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Cell").field("id", &self.id).finish()
    }
}

/// One value sent to a reader.
struct Update {
    cell: usize,
    value: Option<Value>,
}

struct Reader {
    id: u64,
    mailbox: Producer<Update, Multi>,
}

/// The authoritative values, and the mailboxes to keep up to date. Changed under the lock only.
#[derive(Default)]
struct Store {
    values: Vec<Option<Value>>,
    readers: Vec<Reader>,
}

/// The lattice's cells and its readers.
pub(crate) struct Views {
    store: LifecycleMutex<Store>,
    next_reader: AtomicU64,
}

impl Views {
    pub(crate) fn new() -> Self {
        Self {
            store: LifecycleMutex::new(Store::default()),
            next_reader: AtomicU64::new(1),
        }
    }
}

/// This thread's view for one registry.
struct ThreadView {
    registry: Weak<UpstreamRegistry>,
    registry_id: u64,
    reader_id: u64,
    view: Rc<ViewState>,
}

struct ViewState {
    values: RefCell<Vec<Option<Value>>>,
    mailbox: RefCell<Consumer<Update, Single>>,
}

impl Drop for ThreadView {
    fn drop(&mut self) {
        if let Some(registry) = self.registry.upgrade() {
            let _cold = HotScope::leave();
            let mut store = registry.views().store.lock();
            store.readers.retain(|reader| reader.id != self.reader_id);
        }
    }
}

thread_local! {
    static THREAD_VIEWS: RefCell<Vec<ThreadView>> = const { RefCell::new(Vec::new()) };
}

impl ViewState {
    /// Applies what the mailbox holds. Does nothing when a read of this view is in progress further up the stack,
    /// which keeps a reference it handed out valid.
    fn drain(&self) {
        let mut replaced = Vec::new();
        {
            let Ok(mut values) = self.values.try_borrow_mut() else {
                return;
            };
            let Ok(mut mailbox) = self.mailbox.try_borrow_mut() else {
                return;
            };
            while let Some(update) = mailbox.poll() {
                if values.len() <= update.cell {
                    values.resize(update.cell + 1, None);
                }
                replaced.push(std::mem::replace(&mut values[update.cell], update.value));
            }
        }
        // The replaced values drop with no borrow held: the last holder of a generation ends it, and that may read
        // this view again.
        drop(replaced);
    }
}

impl UpstreamRegistry {
    /// A cell for a value of type `T`, holding nothing.
    pub fn cell<T: Send + Sync + 'static>(&self) -> Cell<T> {
        let _cold = HotScope::leave();
        let mut store = self.views().store.lock();
        store.values.push(None);
        Cell {
            id: store.values.len() - 1,
            _marker: PhantomData,
        }
    }

    /// Stores `value` in `cell` and sends it to every registered reader. Lifecycle work, off the frame path. A
    /// replaced value is dropped once the store and every reader have let go of it.
    pub fn set_cell<T: Send + Sync + 'static>(&self, cell: &Cell<T>, value: Option<Arc<T>>) {
        let _cold = HotScope::leave();
        let value: Option<Value> = value.map(|value| value as Value);
        let replaced = {
            let mut store = self.views().store.lock();
            let replaced = std::mem::replace(&mut store.values[cell.id], value.clone());
            for reader in &store.readers {
                let _ = reader.mailbox.offer(Update {
                    cell: cell.id,
                    value: value.clone(),
                });
            }
            replaced
        };
        // Outside the lock: dropping the last holder of a generation may take lifecycle locks of its own.
        drop(replaced);
    }

    /// Stores `value` in `cell` if `expected` accepts the value it holds now (decided under the lock), and sends it to
    /// the readers; returns whether it did. Java's compare-and-set of a reference to null, where the comparison is made
    /// on the value and not on an address that could be reused.
    pub fn set_cell_if<T: Send + Sync + 'static>(
        &self,
        cell: &Cell<T>,
        expected: impl FnOnce(Option<&T>) -> bool,
        value: Option<Arc<T>>,
    ) -> bool {
        let _cold = HotScope::leave();
        let value: Option<Value> = value.map(|value| value as Value);
        let replaced = {
            let mut store = self.views().store.lock();
            let held = store.values[cell.id]
                .as_ref()
                .and_then(|held| held.downcast_ref::<T>());
            if !expected(held) {
                return false;
            }
            let replaced = std::mem::replace(&mut store.values[cell.id], value.clone());
            for reader in &store.readers {
                let _ = reader.mailbox.offer(Update {
                    cell: cell.id,
                    value: value.clone(),
                });
            }
            replaced
        };
        drop(replaced);
        true
    }

    /// The current value of `cell`, read under the lock. For lifecycle code; a frame reads its view.
    pub fn cell_value<T: Send + Sync + 'static>(&self, cell: &Cell<T>) -> Option<Arc<T>> {
        let _cold = HotScope::leave();
        let store = self.views().store.lock();
        let value = store.values.get(cell.id)?.clone()?;
        value.downcast::<T>().ok()
    }

    /// Makes the calling thread a reader: it gets a mailbox and a view, and reads the cells without a lock.
    /// Does nothing if it already is one. A reader drains its mailbox when it starts a cycle
    /// ([`refresh_view`](Self::refresh_view)) and when it starts a read.
    pub fn register_reader(self: &Arc<Self>) {
        let _ = THREAD_VIEWS.try_with(|views| {
            let mut views = views.borrow_mut();
            if views.iter().any(|v| v.registry_id == self.id()) {
                return;
            }
            views.retain(|v| v.registry.strong_count() > 0);
            let _cold = HotScope::leave();
            let queue = MpscQueue::new(64);
            let reader_id = self.views().next_reader.fetch_add(1, Ordering::Relaxed);
            let mut store = self.views().store.lock();
            let values = store.values.clone();
            store.readers.push(Reader {
                id: reader_id,
                mailbox: queue.producer(),
            });
            drop(store);
            views.push(ThreadView {
                registry: Arc::downgrade(self),
                registry_id: self.id(),
                reader_id,
                view: Rc::new(ViewState {
                    values: RefCell::new(values),
                    mailbox: RefCell::new(queue.consumer()),
                }),
            });
        });
    }

    /// Withdraws the calling thread as a reader: its mailbox and view go, and the values it held are let go of.
    pub fn remove_reader(&self) {
        let removed: Vec<ThreadView> = THREAD_VIEWS
            .try_with(|views| {
                let mut views = views.borrow_mut();
                let (removed, kept) = views.drain(..).partition(|v| v.registry_id == self.id());
                *views = kept;
                removed
            })
            .unwrap_or_default();
        drop(removed);
    }

    /// Whether the calling thread is a reader of this registry.
    pub fn is_reader(&self) -> bool {
        THREAD_VIEWS
            .try_with(|views| views.borrow().iter().any(|v| v.registry_id == self.id()))
            .unwrap_or(false)
    }

    /// Applies what the calling reader's mailbox holds. A worker calls it at the start of each cycle, so a generation
    /// it no longer routes by is let go of promptly, also when it reads nothing.
    pub fn refresh_view(self: &Arc<Self>) {
        let _ = THREAD_VIEWS.try_with(|views| {
            if let Some(view) = views
                .borrow()
                .iter()
                .find(|v| v.registry_id == self.id())
                .map(|v| v.view.clone())
            {
                view.drain();
            }
        });
    }

    /// Drops the calling reader's view of every cell and takes the current values again at the next read. A worker
    /// about to park calls it so that it holds no generation while it sleeps.
    pub fn clear_view(self: &Arc<Self>) {
        let _ = THREAD_VIEWS.try_with(|views| {
            if let Some(view) = views
                .borrow()
                .iter()
                .find(|v| v.registry_id == self.id())
                .map(|v| v.view.clone())
            {
                view.drain();
                if let Ok(mut values) = view.values.try_borrow_mut() {
                    let old = std::mem::take(&mut *values);
                    drop(values);
                    drop(old);
                    // Re-fill from the store, so the next read needs no lock: take the current values.
                    let _cold = HotScope::leave();
                    let store = self.views().store.lock();
                    if let Ok(mut values) = view.values.try_borrow_mut() {
                        *values = store.values.clone();
                    }
                }
            }
        });
    }

    /// Runs `f` with the calling thread's view of the cells. A registered reader reads its own copies; any other
    /// thread reads under the lock, one cell at a time, on first use of each.
    pub fn with_view<R>(self: &Arc<Self>, f: impl FnOnce(&View<'_>) -> R) -> R {
        let local = THREAD_VIEWS
            .try_with(|views| {
                views
                    .borrow()
                    .iter()
                    .find(|v| v.registry_id == self.id())
                    .map(|v| v.view.clone())
            })
            .ok()
            .flatten();
        if let Some(state) = local {
            state.drain();
            let values = state.values.borrow();
            return f(&View {
                source: Source::Local(&values),
                registry: self,
            });
        }
        let kept = RefCell::new(Vec::new());
        f(&View {
            source: Source::Locked(&kept),
            registry: self,
        })
    }
}

enum Source<'a> {
    Local(&'a [Option<Value>]),
    /// Values taken under the lock, kept for the call.
    Locked(&'a RefCell<Vec<(usize, Value)>>),
}

/// The calling thread's view of the cells for one call of [`UpstreamRegistry::with_view`].
pub struct View<'a> {
    source: Source<'a>,
    registry: &'a Arc<UpstreamRegistry>,
}

impl View<'_> {
    /// The value in `cell` as of the reader's last drain, or as of the first read of it in this call for a thread that
    /// is not a reader.
    pub fn get<T: Send + Sync + 'static>(&self, cell: &Cell<T>) -> Option<&T> {
        match &self.source {
            Source::Local(values) => values.get(cell.id)?.as_ref()?.downcast_ref::<T>(),
            Source::Locked(kept) => {
                let existing = kept
                    .borrow()
                    .iter()
                    .find(|(id, _)| *id == cell.id)
                    .map(|(_, value)| Arc::as_ptr(value));
                let pointer = match existing {
                    Some(pointer) => pointer,
                    None => {
                        let _cold = HotScope::leave();
                        let store = self.registry.views().store.lock();
                        let value = store.values.get(cell.id)?.clone()?;
                        drop(store);
                        let pointer = Arc::as_ptr(&value);
                        kept.borrow_mut().push((cell.id, value));
                        pointer
                    }
                };
                // SAFETY: `kept` holds a clone of the `Arc` for the whole call, so the value lives as long as `self`;
                // the vector is only pushed to, and a value is an `Arc` allocation, which does not move.
                let value: &(dyn Any + Send + Sync) = unsafe { &*pointer };
                value.downcast_ref::<T>()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;
    use std::thread;

    fn registry() -> Arc<UpstreamRegistry> {
        UpstreamRegistry::with_topology(4, euhedral_hardware_utils::util::CpuSet::new())
    }

    /// A value that counts its drops.
    struct Counted(u64, Arc<AtomicUsize>);

    impl Drop for Counted {
        fn drop(&mut self) {
            self.1.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn an_unregistered_thread_reads_the_current_value_under_the_lock() {
        let registry = registry();
        let cell = registry.cell::<u64>();
        assert!(registry.with_view(|view| view.get(&cell).is_none()));
        registry.set_cell(&cell, Some(Arc::new(7)));
        assert_eq!(registry.with_view(|view| view.get(&cell).copied()), Some(7));
        registry.set_cell(&cell, None);
        assert!(registry.with_view(|view| view.get(&cell).is_none()));
    }

    #[test]
    fn a_reader_sees_a_change_at_its_next_read_and_the_replaced_value_is_dropped_by_the_last_holder()
     {
        let registry = registry();
        let drops = Arc::new(AtomicUsize::new(0));
        let cell = registry.cell::<Counted>();
        registry.set_cell(&cell, Some(Arc::new(Counted(1, drops.clone()))));
        registry.register_reader();
        assert_eq!(registry.with_view(|v| v.get(&cell).map(|c| c.0)), Some(1));

        registry.set_cell(&cell, Some(Arc::new(Counted(2, drops.clone()))));
        // The store let go of the first; the reader's view still holds it until it drains.
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        assert_eq!(registry.with_view(|v| v.get(&cell).map(|c| c.0)), Some(2));
        assert_eq!(
            drops.load(Ordering::SeqCst),
            1,
            "dropped by the reader that held it last"
        );

        registry.remove_reader();
        registry.set_cell(&cell, None);
        assert_eq!(drops.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn a_read_in_progress_keeps_its_reference_valid_when_an_update_arrives() {
        let registry = registry();
        let cell = registry.cell::<u64>();
        registry.set_cell(&cell, Some(Arc::new(1)));
        registry.register_reader();
        registry.with_view(|view| {
            let held = view.get(&cell).unwrap();
            registry.set_cell(&cell, Some(Arc::new(2)));
            // A nested read does not drain: `held` stays what it was.
            assert_eq!(
                registry.with_view(|inner| inner.get(&cell).copied()),
                Some(1)
            );
            assert_eq!(*held, 1);
        });
        assert_eq!(registry.with_view(|v| v.get(&cell).copied()), Some(2));
        registry.remove_reader();
    }

    #[test]
    fn set_cell_if_compares_under_the_lock() {
        let registry = registry();
        let cell = registry.cell::<u64>();
        registry.set_cell(&cell, Some(Arc::new(5)));
        assert!(!registry.set_cell_if(&cell, |held| held == Some(&6), None));
        assert_eq!(registry.cell_value(&cell).as_deref(), Some(&5));
        assert!(registry.set_cell_if(&cell, |held| held == Some(&5), Some(Arc::new(9))));
        assert_eq!(registry.cell_value(&cell).as_deref(), Some(&9));
    }

    #[test]
    #[allow(clippy::disallowed_methods)] // The test's own threads stand for workers and a remapper.
    fn readers_see_every_generation_in_order_and_each_is_dropped_once() {
        const READERS: usize = if cfg!(miri) { 2 } else { 6 };
        const GENERATIONS: u64 = if cfg!(miri) { 20 } else { 2_000 };
        let registry = registry();
        let drops = Arc::new(AtomicUsize::new(0));
        let cell = Arc::new(registry.cell::<Counted>());
        registry.set_cell(&cell, Some(Arc::new(Counted(0, drops.clone()))));
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let readers: Vec<_> = (0..READERS)
            .map(|_| {
                let (registry, cell, stop) = (registry.clone(), cell.clone(), stop.clone());
                thread::spawn(move || {
                    registry.register_reader();
                    let mut last = 0;
                    while !stop.load(Ordering::Acquire) {
                        let now = registry.with_view(|v| v.get(&cell).map(|c| c.0)).unwrap();
                        assert!(
                            now >= last,
                            "a reader never goes back to an older generation"
                        );
                        last = now;
                    }
                    last
                })
            })
            .collect();
        for generation in 1..=GENERATIONS {
            registry.set_cell(&cell, Some(Arc::new(Counted(generation, drops.clone()))));
        }
        stop.store(true, Ordering::Release);
        for reader in readers {
            reader.join().unwrap();
        }
        registry.set_cell(&cell, None);
        assert_eq!(
            drops.load(Ordering::SeqCst) as u64,
            GENERATIONS + 1,
            "every generation dropped exactly once"
        );
    }
}
