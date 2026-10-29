//! Values the lattice's workers read on every frame, handed to them through queues.
//!
//! A *cell* is a slot of lattice-wide state that changes only when the lattice is rebuilt: a vertex's routing
//! generation, an edge's parent, a CPU's cache. The writer ([`set_cell`](UpstreamRegistry::set_cell), lifecycle work)
//! keeps the current value under a lifecycle lock and sends it, through a queue, to the mailbox of every registered
//! reader. A reader drains its mailbox in one batch at the start of each cycle and applies the values to a **view**
//! only it reads. A frame reads the view: no lock, no atomic read-modify-write, no shared line, no reference count. The
//! value a view replaced is dropped by its reader; the last `Arc` to go ends the generation. Nothing is reclaimed by an
//! epoch or a barrier.
//!
//! There are two kinds of reader, and no reader ever takes the lock to read:
//! - A **worker** (or any thread that [`register_reader`](UpstreamRegistry::register_reader)s) is a *full* reader: its
//!   view holds `Arc`s, so a read costs nothing, and it keeps the values it last saw until it drains again. A worker
//!   drains at the start of every cycle, and its parks are timed, so a replaced generation waits at most one park.
//! - Any other thread that reads gets a *light* mailbox and view on its first read, through the same queue, and
//!   withdraws them when the thread exits. Its view holds `Weak`s: a read upgrades one for the length of the call (one
//!   reference-count update), so a thread that reads once and goes idle never keeps a replaced generation alive.

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
type WeakValue = Weak<dyn Any + Send + Sync>;

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

/// What a reader holds for one cell: nothing, the value, or (for a light reader) a weak reference to it.
#[derive(Clone)]
enum Slot {
    Empty,
    Full(Value),
    Light(WeakValue),
}

/// One value sent to a reader.
struct Update {
    cell: usize,
    slot: Slot,
}

struct Reader {
    id: u64,
    light: bool,
    mailbox: Producer<Update, Multi>,
}

/// The authoritative values, and the mailboxes to keep up to date. Changed under the lock only.
#[derive(Default)]
struct Store {
    values: Vec<Option<Value>>,
    readers: Vec<Reader>,
}

impl Store {
    /// What a reader of the given kind holds for `value`.
    fn slot(value: &Option<Value>, light: bool) -> Slot {
        match value {
            None => Slot::Empty,
            Some(value) if light => Slot::Light(Arc::downgrade(value)),
            Some(value) => Slot::Full(value.clone()),
        }
    }

    fn slots(&self, light: bool) -> Vec<Slot> {
        self.values
            .iter()
            .map(|value| Self::slot(value, light))
            .collect()
    }

    /// Stores `value` and sends it to every reader. Returns the value replaced.
    fn replace(&mut self, cell: usize, value: Option<Value>) -> Option<Value> {
        let replaced = std::mem::replace(&mut self.values[cell], value.clone());
        for reader in &self.readers {
            let _ = reader.mailbox.offer(Update {
                cell,
                slot: Self::slot(&value, reader.light),
            });
        }
        replaced
    }
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
    light: bool,
    view: Rc<ViewState>,
}

struct ViewState {
    values: RefCell<Vec<Slot>>,
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
                    values.resize(update.cell + 1, Slot::Empty);
                }
                replaced.push(std::mem::replace(&mut values[update.cell], update.slot));
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
    /// replaced value is dropped once the store and every full reader have let go of it.
    pub fn set_cell<T: Send + Sync + 'static>(&self, cell: &Cell<T>, value: Option<Arc<T>>) {
        let _cold = HotScope::leave();
        let value: Option<Value> = value.map(|value| value as Value);
        let replaced = self.views().store.lock().replace(cell.id, value);
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
            store.replace(cell.id, value)
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

    /// Makes the calling thread a full reader: it gets a mailbox and a view that holds the values, and reads the cells
    /// with no lock and no reference-count update. A thread that already reads as a light reader is made a full one.
    /// A reader drains its mailbox when it starts a cycle ([`refresh_view`](Self::refresh_view)) and when it starts a
    /// read.
    pub fn register_reader(self: &Arc<Self>) {
        self.register_as(false);
    }

    fn register_as(self: &Arc<Self>, light: bool) {
        // Views dropped here are dropped after the thread's list is released: their last values may read it.
        let mut doomed: Vec<ThreadView> = Vec::new();
        let _ = THREAD_VIEWS.try_with(|views| {
            let mut views = views.borrow_mut();
            let (dead, alive): (Vec<_>, Vec<_>) = views.drain(..).partition(|v| {
                v.registry.strong_count() == 0 || (v.registry_id == self.id() && v.light && !light)
            });
            *views = alive;
            doomed = dead;
            if views.iter().any(|v| v.registry_id == self.id()) {
                return;
            }
            let _cold = HotScope::leave();
            let queue = MpscQueue::new(64);
            let reader_id = self.views().next_reader.fetch_add(1, Ordering::Relaxed);
            let mut store = self.views().store.lock();
            let slots = store.slots(light);
            store.readers.push(Reader {
                id: reader_id,
                light,
                mailbox: queue.producer(),
            });
            drop(store);
            views.push(ThreadView {
                registry: Arc::downgrade(self),
                registry_id: self.id(),
                reader_id,
                light,
                view: Rc::new(ViewState {
                    values: RefCell::new(slots),
                    mailbox: RefCell::new(queue.consumer()),
                }),
            });
        });
        drop(doomed);
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
        self.local_view().is_some()
    }

    /// Applies what the calling reader's mailbox holds. A worker calls it at the start of each cycle, so a generation
    /// it no longer routes by is let go of promptly, also when it reads nothing.
    pub fn refresh_view(self: &Arc<Self>) {
        if let Some((view, false)) = self.local_view_kind() {
            view.drain();
        }
    }

    /// The calling thread's view and whether it is a light one, if it is a reader of this registry.
    fn local_view_kind(&self) -> Option<(Rc<ViewState>, bool)> {
        THREAD_VIEWS
            .try_with(|views| {
                views
                    .borrow()
                    .iter()
                    .find(|v| v.registry_id == self.id())
                    .map(|v| (v.view.clone(), v.light))
            })
            .ok()
            .flatten()
    }

    fn local_view(&self) -> Option<Rc<ViewState>> {
        self.local_view_kind().map(|(view, _)| view)
    }

    /// Runs `f` with the calling thread's view of the cells. A thread that is not yet a reader becomes a light one
    /// first (its first read takes the lock once to register; every later read is lock-free).
    pub fn with_view<R>(self: &Arc<Self>, f: impl FnOnce(&View<'_>) -> R) -> R {
        let local = self.local_view_kind().or_else(|| {
            self.register_as(true);
            self.local_view_kind()
        });
        match local {
            Some((state, false)) => {
                state.drain();
                let values = state.values.borrow();
                f(&View {
                    source: Source::Full(&values),
                })
            }
            Some((state, true)) => {
                state.drain();
                f(&View {
                    source: Source::Light {
                        state: &state,
                        held: RefCell::new(Vec::new()),
                    },
                })
            }
            None => {
                // The thread's local storage is already gone (the thread is exiting): read a snapshot, under the lock.
                let snapshot = {
                    let _cold = HotScope::leave();
                    self.views().store.lock().slots(false)
                };
                f(&View {
                    source: Source::Full(&snapshot),
                })
            }
        }
    }
}

enum Source<'a> {
    Full(&'a [Slot]),
    /// A light reader's view: values upgraded during this call, kept until it ends.
    Light {
        state: &'a ViewState,
        held: RefCell<Vec<(usize, Value)>>,
    },
}

/// The calling thread's view of the cells for one call of [`UpstreamRegistry::with_view`].
pub struct View<'a> {
    source: Source<'a>,
}

impl View<'_> {
    /// The value in `cell` as of the reader's last drain.
    pub fn get<T: Send + Sync + 'static>(&self, cell: &Cell<T>) -> Option<&T> {
        match &self.source {
            Source::Full(slots) => match slots.get(cell.id)? {
                Slot::Full(value) => value.downcast_ref::<T>(),
                Slot::Empty | Slot::Light(_) => None,
            },
            Source::Light { state, held } => {
                let existing = held
                    .borrow()
                    .iter()
                    .find(|(id, _)| *id == cell.id)
                    .map(|(_, value)| Arc::as_ptr(value));
                let pointer = match existing {
                    Some(pointer) => pointer,
                    None => {
                        let value = Self::upgrade(state, cell.id)?;
                        let pointer = Arc::as_ptr(&value);
                        held.borrow_mut().push((cell.id, value));
                        pointer
                    }
                };
                // SAFETY: `held` keeps a clone of the `Arc` for the whole call, so the value lives as long as `self`;
                // the vector is only pushed to, and a value is an `Arc` allocation, which does not move.
                let value: &(dyn Any + Send + Sync) = unsafe { &*pointer };
                value.downcast_ref::<T>()
            }
        }
    }

    /// Upgrades the light reader's reference to a cell's value. A reference that has died was replaced after the last
    /// drain: the update that replaced it was queued before it died, so draining again finds the newer value.
    fn upgrade(state: &ViewState, id: usize) -> Option<Value> {
        loop {
            let slot = state.values.borrow().get(id)?.clone();
            match slot {
                Slot::Empty | Slot::Full(_) => return None,
                Slot::Light(weak) => {
                    if let Some(value) = weak.upgrade() {
                        return Some(value);
                    }
                    state.drain();
                    let newer = state.values.borrow().get(id).cloned();
                    match newer {
                        Some(Slot::Light(now)) if !Weak::ptr_eq(&now, &weak) => {}
                        _ => return None,
                    }
                }
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
    fn a_thread_that_reads_without_registering_becomes_a_light_reader_on_its_first_read() {
        let registry = registry();
        let cell = registry.cell::<u64>();
        assert!(!registry.is_reader());
        assert!(registry.with_view(|view| view.get(&cell).is_none()));
        assert!(registry.is_reader(), "the first read registered the thread");
        registry.set_cell(&cell, Some(Arc::new(7)));
        assert_eq!(registry.with_view(|view| view.get(&cell).copied()), Some(7));
        registry.set_cell(&cell, None);
        assert!(registry.with_view(|view| view.get(&cell).is_none()));
        registry.remove_reader();
        assert!(!registry.is_reader());
    }

    #[test]
    #[allow(clippy::disallowed_methods)] // The test's own thread stands for a pusher that exits.
    fn a_thread_that_exits_withdraws_its_mailbox() {
        let registry = registry();
        let drops = Arc::new(AtomicUsize::new(0));
        let cell = Arc::new(registry.cell::<Counted>());
        registry.set_cell(&cell, Some(Arc::new(Counted(1, drops.clone()))));
        {
            let (registry, cell) = (registry.clone(), cell.clone());
            thread::spawn(move || {
                assert_eq!(registry.with_view(|v| v.get(&cell).map(|c| c.0)), Some(1));
            })
            .join()
            .unwrap();
        }
        registry.set_cell(&cell, None);
        assert_eq!(
            drops.load(Ordering::SeqCst),
            1,
            "the exited thread no longer holds the value"
        );
        assert_eq!(
            registry.views().store.lock().readers.len(),
            0,
            "its mailbox is gone"
        );
    }

    #[test]
    fn a_light_reader_keeps_a_generation_only_for_the_length_of_its_read() {
        let registry = registry();
        let drops = Arc::new(AtomicUsize::new(0));
        let cell = registry.cell::<Counted>();
        registry.set_cell(&cell, Some(Arc::new(Counted(1, drops.clone()))));
        registry.with_view(|v| {
            assert_eq!(v.get(&cell).map(|c| c.0), Some(1));
            // Replaced during the read: the read keeps what it began with.
            registry.set_cell(&cell, Some(Arc::new(Counted(2, drops.clone()))));
            assert_eq!(drops.load(Ordering::SeqCst), 0);
            assert_eq!(v.get(&cell).map(|c| c.0), Some(1));
        });
        assert_eq!(
            drops.load(Ordering::SeqCst),
            1,
            "let go when the read ended, with no further read by the thread"
        );
        assert_eq!(registry.with_view(|v| v.get(&cell).map(|c| c.0)), Some(2));
        registry.set_cell(&cell, None);
        assert_eq!(drops.load(Ordering::SeqCst), 2);
        assert!(registry.with_view(|v| v.get(&cell).is_none()));
    }

    #[test]
    fn registering_a_light_reader_makes_it_a_full_one() {
        let registry = registry();
        let drops = Arc::new(AtomicUsize::new(0));
        let cell = registry.cell::<Counted>();
        registry.set_cell(&cell, Some(Arc::new(Counted(1, drops.clone()))));
        assert_eq!(registry.with_view(|v| v.get(&cell).map(|c| c.0)), Some(1));
        registry.register_reader();
        assert_eq!(registry.with_view(|v| v.get(&cell).map(|c| c.0)), Some(1));
        registry.set_cell(&cell, Some(Arc::new(Counted(2, drops.clone()))));
        assert_eq!(drops.load(Ordering::SeqCst), 0, "a full reader holds it");
        registry.remove_reader();
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        registry.set_cell(&cell, None);
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
