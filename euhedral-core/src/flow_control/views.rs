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
use std::sync::atomic::{AtomicBool, Ordering, fence};
use std::sync::{Arc, Weak};

type Value = Arc<dyn Any + Send + Sync>;
type WeakValue = Weak<dyn Any + Send + Sync>;

/// A slot of lattice state of type `T`, made by [`UpstreamRegistry::cell`].
pub struct Cell<T> {
    id: usize,
    /// How many times the id was freed before this cell got it: a view whose entry for the id has another epoch holds
    /// a dead cell's value, not this one's.
    epoch: u64,
    views: Arc<Views>,
    _marker: PhantomData<fn() -> T>,
}

impl<T> Drop for Cell<T> {
    /// The cell's id goes back to the registry for the next cell, after every reader let go of what it held for it.
    fn drop(&mut self) {
        self.views.free(self.id);
    }
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
    epoch: u64,
    slot: Slot,
}

/// What a view holds for one cell id: the value, and the epoch of the cell that value belongs to.
#[derive(Clone)]
struct Entry {
    epoch: u64,
    slot: Slot,
}

impl Entry {
    const EMPTY: Entry = Entry {
        epoch: 0,
        slot: Slot::Empty,
    };
}

struct Reader {
    id: u64,
    light: bool,
    mailbox: Producer<Update, Multi>,
    /// Set by the writer when it drops a light reader whose mailbox backed up; the reader registers again at its next
    /// read.
    evicted: Arc<AtomicBool>,
}

/// How many updates a light reader's mailbox may hold before the writer drops the reader. A light reader that stops
/// reading (a thread that pushed once and went elsewhere) would otherwise collect an update for every change of the
/// lattice for as long as it lives.
const LIGHT_BACKLOG: u64 = 1024;

/// The authoritative values, and the mailboxes to keep up to date. Changed under the lock only.
#[derive(Default)]
struct Store {
    values: Vec<Option<Value>>,
    /// The epoch of each id: how many cells have freed it.
    epochs: Vec<u64>,
    /// Ids of the cells that were dropped, for the next cell.
    free: Vec<usize>,
    readers: Vec<Reader>,
    next_reader: u64,
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

    fn slots(&self, light: bool) -> Vec<Entry> {
        self.values
            .iter()
            .zip(&self.epochs)
            .map(|(value, &epoch)| Entry {
                epoch,
                slot: Self::slot(value, light),
            })
            .collect()
    }

    /// Stores `value` and sends it to every reader. Returns the value replaced.
    fn replace(&mut self, cell: usize, value: Option<Value>) -> Option<Value> {
        let replaced = std::mem::replace(&mut self.values[cell], value.clone());
        let epoch = self.epochs[cell];
        let mut evicted = false;
        for reader in &self.readers {
            if reader.light && reader.mailbox.size_long() > LIGHT_BACKLOG {
                reader.evicted.store(true, Ordering::Release);
                evicted = true;
                continue;
            }
            let _ = reader.mailbox.offer(Update {
                cell,
                epoch,
                slot: Self::slot(&value, reader.light),
            });
        }
        if evicted {
            self.readers
                .retain(|reader| !reader.evicted.load(Ordering::Relaxed));
        }
        replaced
    }
}

/// The lattice's cells and its readers.
pub(crate) struct Views {
    store: LifecycleMutex<Store>,
}

impl Views {
    pub(crate) fn new() -> Self {
        Self {
            store: LifecycleMutex::new(Store {
                next_reader: 1,
                ..Store::default()
            }),
        }
    }

    /// An id for a new cell, and its epoch: a dropped cell's id (its epoch is one more than the dead cell's), or a new
    /// one.
    fn allocate(&self) -> (usize, u64) {
        let _cold = HotScope::leave();
        let mut store = self.store.lock();
        match store.free.pop() {
            Some(id) => (id, store.epochs[id]),
            None => {
                store.values.push(None);
                store.epochs.push(0);
                let id = store.values.len() - 1;
                // Readers get an (empty) entry for the new id at their next drain, so that a read of a cell nobody has
                // set yet finds one; a read that began before the drain finds none and reads under the lock.
                drop(store.replace(id, None));
                (id, 0)
            }
        }
    }

    /// The current value of cell `id` if it is still the cell of `epoch`, read under the lock.
    fn value(&self, id: usize, epoch: u64) -> Option<Value> {
        let _cold = HotScope::leave();
        let store = self.store.lock();
        if store.epochs.get(id) != Some(&epoch) {
            return None;
        }
        store.values.get(id)?.clone()
    }

    /// Empties the cell `id` for every reader, and keeps the id for the next cell.
    fn free(&self, id: usize) {
        let _cold = HotScope::leave();
        let replaced = {
            let mut store = self.store.lock();
            // The next cell of this id has the next epoch: a view that still holds the dead cell's value for the id
            // (it cannot drain inside a read) sees that the epoch is not the new cell's.
            store.epochs[id] += 1;
            let replaced = store.replace(id, None);
            store.free.push(id);
            replaced
        };
        // Outside the lock: dropping the last holder of a generation may take lifecycle locks of its own.
        drop(replaced);
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
    values: RefCell<Vec<Entry>>,
    mailbox: RefCell<Consumer<Update, Single>>,
    /// Whether the writer dropped this (light) reader for a mailbox that backed up.
    evicted: Arc<AtomicBool>,
}

impl Drop for ThreadView {
    fn drop(&mut self) {
        // First stop the writer from offering to this mailbox.
        if let Some(registry) = self.registry.upgrade() {
            let _cold = HotScope::leave();
            let mut store = registry.views().store.lock();
            store.readers.retain(|reader| reader.id != self.reader_id);
        }
        // The thread's last holds on generations go here, as a reader: the values of the view and the updates still in
        // the mailbox. A retirement a release runs must not wait on this thread (this may be a worker's thread-local
        // destructor).
        let _reader = ReaderContext::enter();
        if let Ok(mut values) = self.view.values.try_borrow_mut() {
            let old = std::mem::take(&mut *values);
            drop(values);
            drop(old);
        }
        if let Ok(mut mailbox) = self.view.mailbox.try_borrow_mut() {
            while let Some(update) = mailbox.poll() {
                drop(update);
            }
        }
    }
}

thread_local! {
    static THREAD_VIEWS: RefCell<Vec<ThreadView>> = const { RefCell::new(Vec::new()) };
    static READER_DEPTH: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

/// Marks the code a reader runs while it lets go of values it held: the last holder of a replaced generation releases
/// it there, and that release must not wait.
struct ReaderContext;

impl ReaderContext {
    fn enter() -> Self {
        let _ = READER_DEPTH.try_with(|depth| depth.set(depth.get() + 1));
        Self
    }
}

impl Drop for ReaderContext {
    fn drop(&mut self) {
        let _ = READER_DEPTH.try_with(|depth| depth.set(depth.get() - 1));
    }
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
                    values.resize(update.cell + 1, Entry::EMPTY);
                }
                let entry = Entry {
                    epoch: update.epoch,
                    slot: update.slot,
                };
                replaced.push(std::mem::replace(&mut values[update.cell], entry).slot);
            }
        }
        // The replaced values drop with no borrow held: the last holder of a generation ends it, and that may read
        // this view again. It does so as a reader: whatever that release runs must not wait here.
        let _reader = ReaderContext::enter();
        drop(replaced);
    }
}

impl UpstreamRegistry {
    /// A cell for a value of type `T`, holding nothing.
    pub fn cell<T: Send + Sync + 'static>(&self) -> Cell<T> {
        let (id, epoch) = self.views().allocate();
        Cell {
            id,
            epoch,
            views: self.views().clone(),
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
            let evicted = Arc::new(AtomicBool::new(false));
            let mut store = self.views().store.lock();
            let reader_id = store.next_reader;
            store.next_reader += 1;
            let slots = store.slots(light);
            store.readers.push(Reader {
                id: reader_id,
                light,
                mailbox: queue.producer(),
                evicted: evicted.clone(),
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
                    evicted,
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

    /// Whether the calling thread is letting go of values it read as a reader: draining its mailbox, or ending a light
    /// read. A release that runs there is on a frame path (a worker's, or a producer's) and must not wait or run
    /// lifecycle work; the writer's own drop of a replaced value is not.
    pub(crate) fn in_reader_context() -> bool {
        READER_DEPTH
            .try_with(|depth| depth.get() > 0)
            .unwrap_or(false)
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
        let local = match self.local_view_kind() {
            Some((state, true)) if state.evicted.load(Ordering::Acquire) => {
                // The writer dropped this reader because it had stopped reading: register again, from the current
                // values.
                self.remove_reader();
                self.register_as(true);
                self.local_view_kind()
            }
            Some(local) => Some(local),
            None => {
                self.register_as(true);
                self.local_view_kind()
            }
        };
        match local {
            Some((state, false)) => {
                state.drain();
                let values = state.values.borrow();
                f(&View {
                    source: Source::Full(&values),
                    cold: std::cell::Cell::new(None),
                })
            }
            Some((state, true)) => {
                state.drain();
                // Declared first, so it ends after the view: the values this read upgraded are let go of in a reader
                // context.
                let _reader = ReaderContext::enter();
                f(&View {
                    source: Source::Light {
                        state: &state,
                        held: RefCell::new(Held::new()),
                    },
                    cold: std::cell::Cell::new(None),
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
                    cold: std::cell::Cell::new(None),
                })
            }
        }
    }
}

#[allow(
    clippy::large_enum_variant,
    reason = "a stack value made once per read; boxing the light variant would allocate on the frame path"
)]
enum Source<'a> {
    Full(&'a [Entry]),
    /// A light reader's view: values upgraded during this call, kept until it ends.
    Light {
        state: &'a ViewState,
        held: RefCell<Held>,
    },
}

/// The values a light read has upgraded: a handful, kept inline so that the read allocates nothing; a read of more
/// cells than that spills to a vector.
const HELD_INLINE: usize = 8;

struct Held {
    inline: [Option<(usize, Value)>; HELD_INLINE],
    used: usize,
    spill: Vec<(usize, Value)>,
}

impl Held {
    fn new() -> Self {
        Self {
            inline: [const { None }; HELD_INLINE],
            used: 0,
            spill: Vec::new(),
        }
    }

    fn find(&self, id: usize) -> Option<*const (dyn Any + Send + Sync)> {
        self.inline[..self.used]
            .iter()
            .flatten()
            .chain(self.spill.iter())
            .find(|(held, _)| *held == id)
            .map(|(_, value)| Arc::as_ptr(value))
    }

    fn push(&mut self, id: usize, value: Value) {
        if self.used < HELD_INLINE {
            self.inline[self.used] = Some((id, value));
            self.used += 1;
        } else {
            self.spill.push((id, value));
        }
    }
}

/// The calling thread's view of the cells for one call of [`UpstreamRegistry::with_view`].
pub struct View<'a> {
    source: Source<'a>,
    /// Values read under the lock for this call, where the view could not answer (see [`get`](Self::get)). One null
    /// word, and nothing allocated, on every read that stays on the view.
    cold: std::cell::Cell<Option<ColdValues>>,
}

/// Values a read took under the lock; boxed so that the view holds one word for them.
type ColdValues = Box<Vec<(usize, Value)>>;

/// What a light read found for a cell.
enum Found {
    Value(Value),
    /// The view cannot tell: read the cell under the lock.
    Cold,
    Nothing,
}

impl View<'_> {
    /// The value in `cell` as of the reader's last drain. Where the view cannot answer, the value is read under the
    /// lock instead: the id was freed and given to this cell after the view's last drain (a read in progress does not
    /// drain, and the view still holds the dead cell's value), or the writer dropped this light reader for a mailbox
    /// that backed up and skipped the offer that this read needs.
    pub fn get<T: Send + Sync + 'static>(&self, cell: &Cell<T>) -> Option<&T> {
        match &self.source {
            Source::Full(entries) => match entries.get(cell.id) {
                Some(entry) if entry.epoch == cell.epoch => match &entry.slot {
                    Slot::Full(value) => value.downcast_ref::<T>(),
                    Slot::Empty | Slot::Light(_) => None,
                },
                _ => self.cold_get(cell),
            },
            Source::Light { state, held } => {
                let existing = held.borrow().find(cell.id);
                let pointer = match existing {
                    Some(pointer) => pointer,
                    None => match Self::upgrade(state, cell) {
                        Found::Value(value) => {
                            let pointer = Arc::as_ptr(&value);
                            held.borrow_mut().push(cell.id, value);
                            pointer
                        }
                        Found::Cold => return self.cold_get(cell),
                        Found::Nothing => return None,
                    },
                };
                // SAFETY: `held` keeps a clone of the `Arc` for the whole call, so the value lives as long as `self`;
                // an entry is only ever added, and a value is an `Arc` allocation, which does not move.
                let value: &(dyn Any + Send + Sync) = unsafe { &*pointer };
                value.downcast_ref::<T>()
            }
        }
    }

    /// The value in `cell` as an `Arc` the caller keeps beyond this read (one reference-count pair, once), for a
    /// caller that would otherwise open a read for every use of it. Read like [`get`](Self::get).
    pub fn get_arc<T: Send + Sync + 'static>(&self, cell: &Cell<T>) -> Option<Arc<T>> {
        let value = match &self.source {
            Source::Full(entries) => match entries.get(cell.id) {
                Some(entry) if entry.epoch == cell.epoch => match &entry.slot {
                    Slot::Full(value) => Some(value.clone()),
                    Slot::Empty | Slot::Light(_) => None,
                },
                _ => cell.views.value(cell.id, cell.epoch),
            },
            Source::Light { state, .. } => match Self::upgrade(state, cell) {
                Found::Value(value) => Some(value),
                Found::Cold => cell.views.value(cell.id, cell.epoch),
                Found::Nothing => None,
            },
        }?;
        value.downcast::<T>().ok()
    }

    /// The cell's current value under the lock, kept for the rest of the call.
    fn cold_get<T: Send + Sync + 'static>(&self, cell: &Cell<T>) -> Option<&T> {
        let mut cold = self.cold.take().unwrap_or_default();
        let existing = cold
            .iter()
            .find(|(id, _)| *id == cell.id)
            .map(|(_, value)| Arc::as_ptr(value));
        let pointer = match existing {
            Some(pointer) => Some(pointer),
            None => cell.views.value(cell.id, cell.epoch).map(|value| {
                let pointer = Arc::as_ptr(&value);
                cold.push((cell.id, value));
                pointer
            }),
        };
        self.cold.set(Some(cold));
        // SAFETY: `cold` keeps a clone of the `Arc` for the whole call, so the value lives as long as `self`; an entry
        // is only ever added, and a value is an `Arc` allocation, which does not move (the vector may).
        let value: &(dyn Any + Send + Sync) = unsafe { &*pointer? };
        value.downcast_ref::<T>()
    }

    /// Upgrades the light reader's reference to a cell's value (one reference-count update, and no copy of the weak
    /// reference). A reference that has died was replaced after the last drain: the update that replaced it was queued
    /// before it died, so draining again finds the newer value.
    fn upgrade<T>(state: &ViewState, cell: &Cell<T>) -> Found {
        loop {
            let dead = match state.values.borrow().get(cell.id) {
                Some(entry) if entry.epoch == cell.epoch => match &entry.slot {
                    Slot::Empty | Slot::Full(_) => return Found::Nothing,
                    Slot::Light(weak) => match weak.upgrade() {
                        Some(value) => return Found::Value(value),
                        None => weak.clone(),
                    },
                },
                _ => return Found::Cold,
            };
            // A failed upgrade is a relaxed load of a count that reached zero. The thread that dropped the last
            // reference queued the replacement before it did; this fence makes that offer visible to the drain.
            fence(Ordering::Acquire);
            state.drain();
            let newer = state.values.borrow().get(cell.id).cloned();
            match newer {
                Some(Entry {
                    epoch,
                    slot: Slot::Light(now),
                }) if epoch == cell.epoch && !Weak::ptr_eq(&now, &dead) => {}
                _ if state.evicted.load(Ordering::Acquire) => {
                    // The writer dropped this reader and skipped the offer that replaced the value.
                    return Found::Cold;
                }
                Some(entry) if entry.epoch != cell.epoch => return Found::Cold,
                _ => return Found::Nothing,
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
    fn a_dropped_cell_gives_its_id_to_the_next_and_the_readers_forget_what_it_held() {
        let registry = registry();
        let first = registry.cell::<u64>();
        registry.set_cell(&first, Some(Arc::new(5)));
        registry.register_reader();
        assert_eq!(registry.with_view(|v| v.get(&first).copied()), Some(5));
        let id = first.id;
        drop(first);
        let second = registry.cell::<u64>();
        assert_eq!(second.id, id, "the id came back");
        assert_eq!(
            registry.with_view(|v| v.get(&second).copied()),
            None,
            "the new cell starts empty, not with the old one's value"
        );
        // Churn: cells made and dropped do not grow the store.
        for _ in 0..1_000 {
            let cell = registry.cell::<u64>();
            registry.set_cell(&cell, Some(Arc::new(1)));
        }
        assert!(registry.views().store.lock().values.len() <= 3);
        registry.remove_reader();
    }

    #[test]
    fn a_light_reader_that_stops_reading_is_dropped_when_its_mailbox_backs_up() {
        let registry = registry();
        let cell = registry.cell::<u64>();
        registry.set_cell(&cell, Some(Arc::new(0)));
        assert_eq!(registry.with_view(|v| v.get(&cell).copied()), Some(0));
        assert_eq!(registry.views().store.lock().readers.len(), 1);
        for value in 1..=(LIGHT_BACKLOG + 10) {
            registry.set_cell(&cell, Some(Arc::new(value)));
        }
        assert_eq!(
            registry.views().store.lock().readers.len(),
            0,
            "the writer stopped feeding a reader that never drains"
        );
        // The thread reads again: it registers again and sees the current value.
        assert_eq!(
            registry.with_view(|v| v.get(&cell).copied()),
            Some(LIGHT_BACKLOG + 10)
        );
        assert_eq!(registry.views().store.lock().readers.len(), 1);
        registry.remove_reader();
    }

    /// A read in progress does not drain, so its view still holds the value of a cell that was dropped; a cell that
    /// took the dropped one's id must not be given that value. Run by
    /// `a_full_readers_nested_read_does_not_see_a_dropped_cells_value_in_a_reused_id` and
    /// `a_light_readers_nested_read_does_not_see_a_dropped_cells_value_in_a_reused_id`.
    fn check_nested_read_of_a_reused_id(full: bool) {
        let registry = registry();
        let old = registry.cell::<u64>();
        registry.set_cell(&old, Some(Arc::new(7)));
        if full {
            registry.register_reader();
        }
        assert_eq!(registry.with_view(|v| v.get(&old).copied()), Some(7));
        registry.with_view(|outer| {
            let id = old.id;
            drop(old);
            let new = registry.cell::<u64>();
            assert_eq!(new.id, id, "the id was reused");
            registry.set_cell(&new, Some(Arc::new(9)));
            assert_eq!(
                registry.with_view(|nested| nested.get(&new).copied()),
                Some(9),
                "a nested read cannot drain, and must still not see the dead cell's 7"
            );
            assert_eq!(outer.get(&new).copied(), Some(9));
        });
        registry.remove_reader();
    }

    #[test]
    fn a_full_readers_nested_read_does_not_see_a_dropped_cells_value_in_a_reused_id() {
        check_nested_read_of_a_reused_id(true);
    }

    #[test]
    fn a_light_readers_nested_read_does_not_see_a_dropped_cells_value_in_a_reused_id() {
        check_nested_read_of_a_reused_id(false);
    }

    /// The writer drops a light reader whose mailbox backed up and skips the offers after that; a read that was
    /// already under way must still find the current value.
    #[test]
    fn a_light_read_that_is_evicted_while_it_runs_still_finds_the_current_value() {
        let registry = registry();
        let cell = registry.cell::<u64>();
        registry.set_cell(&cell, Some(Arc::new(0)));
        assert_eq!(registry.with_view(|v| v.get(&cell).copied()), Some(0));
        let seen = registry.with_view(|v| {
            for value in 1..=(LIGHT_BACKLOG + 10) {
                registry.set_cell(&cell, Some(Arc::new(value)));
            }
            v.get(&cell).copied()
        });
        assert_eq!(seen, Some(LIGHT_BACKLOG + 10));
        registry.remove_reader();
    }

    /// A cell made while a read is in progress is unknown to the view of that read: a nested read of it finds the value
    /// under the lock instead of nothing. Run by `a_full_readers_nested_read_of_a_new_cell_finds_its_value` and
    /// `a_light_readers_nested_read_of_a_new_cell_finds_its_value`.
    fn check_nested_read_of_a_new_cell(full: bool) {
        let registry = registry();
        if full {
            registry.register_reader();
        }
        registry.with_view(|_| {
            let cell = registry.cell::<u64>();
            registry.set_cell(&cell, Some(Arc::new(5)));
            assert_eq!(registry.with_view(|v| v.get(&cell).copied()), Some(5));
        });
        registry.remove_reader();
    }

    #[test]
    fn a_full_readers_nested_read_of_a_new_cell_finds_its_value() {
        check_nested_read_of_a_new_cell(true);
    }

    #[test]
    fn a_light_readers_nested_read_of_a_new_cell_finds_its_value() {
        check_nested_read_of_a_new_cell(false);
    }

    #[test]
    fn get_arc_keeps_a_value_beyond_the_read_for_full_and_light_readers() {
        for full in [true, false] {
            let registry = registry();
            let cell = registry.cell::<u64>();
            registry.set_cell(&cell, Some(Arc::new(11)));
            if full {
                registry.register_reader();
            }
            let kept = registry.with_view(|view| view.get_arc(&cell));
            assert_eq!(kept.as_deref(), Some(&11));
            registry.set_cell(&cell, Some(Arc::new(12)));
            assert_eq!(
                kept.as_deref(),
                Some(&11),
                "the Arc outlives the replacement"
            );
            assert_eq!(
                registry.with_view(|view| view.get_arc(&cell)).as_deref(),
                Some(&12)
            );
            let empty = registry.cell::<u64>();
            assert!(registry.with_view(|view| view.get_arc(&empty)).is_none());
            registry.remove_reader();
        }
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
