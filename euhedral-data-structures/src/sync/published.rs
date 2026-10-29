use super::barrier;
use super::loom::{AtomicBool, AtomicPtr, AtomicU64, Ordering, fence};
use crate::atomics::CachePadded;
use std::cell::Cell;
use std::fmt;
use std::mem::ManuallyDrop;
use std::ptr;
use std::sync::Arc;
use std::sync::atomic::compiler_fence;
use std::time::Instant;

/// A slot value: not inside a read section, so it holds no reference.
const IDLE: u64 = 0;

/// One thread's announcement: [`IDLE`], or the epoch at which the thread last announced a quiescent point or began
/// a read section. A value reachable only through a pointer swapped out before epoch `t` began is safe to free once
/// every announcement is [`IDLE`] or at least `t`.
struct SlotNode {
    state: CachePadded<AtomicU64>,
    taken: AtomicBool,
    /// The next node of the domain's push-only list. Written before the node is published, never after.
    next: *mut SlotNode,
}

/// The global epoch and the announcements of the threads that read [`Published`] values.
///
/// This is the stand-in for the garbage collector that frees what a replaced reference pointed to once no thread
/// can still read it. A reader thread that works in repeated passes announces a quiescent point once per
/// pass ([`EpochSlot::quiescent`]) and marks itself idle before it parks ([`EpochSlot::idle`]); any other thread
/// that reads a published value pins the domain for the length of the read ([`EpochSlot::pin`]).
///
/// The slots form a push-only list. A slot a thread gave back is reused by the next [`register`](Self::register);
/// nodes are freed when the domain drops.
///
/// **The barrier.** A thread that leaves idle announces itself and then loads a pointer, and a writer swaps the
/// pointer and then scans the announcements: each must see the other's store. When the process can issue heavy
/// barriers (`membarrier` on Linux, `FlushProcessWriteBuffers` on Windows; see the `barrier` module) the reader pays
/// none: a release store and a compiler fence, and the writer forces the barrier on every running thread before it
/// scans. Elsewhere the reader keeps the symmetric form, a sequentially consistent store and fence (the OS offers no
/// equivalent to ask for).
pub struct EpochDomain {
    epoch: CachePadded<AtomicU64>,
    head: AtomicPtr<SlotNode>,
    /// Whether readers skip their fence because the writer's scan forces a barrier.
    asymmetric: bool,
}

// SAFETY: the nodes are reached through atomics only; a node's `next` is written once before publication, and the
// raw pointers stay valid until the domain drops.
unsafe impl Send for EpochDomain {}
// SAFETY: as above.
unsafe impl Sync for EpochDomain {}

impl EpochDomain {
    /// A domain with room for `slots` threads at the start; it grows when more register.
    pub fn new(slots: usize) -> Arc<Self> {
        Self::with_barrier(slots, barrier::asymmetric())
    }

    /// Like [`new`](Self::new), with the barrier chosen: `asymmetric` is honoured only where the process can issue
    /// heavy barriers.
    pub fn with_barrier(slots: usize, asymmetric: bool) -> Arc<Self> {
        let domain = Arc::new(Self {
            epoch: CachePadded::new(AtomicU64::new(1)),
            head: AtomicPtr::new(ptr::null_mut()),
            asymmetric: asymmetric && barrier::asymmetric(),
        });
        for _ in 0..slots {
            domain.push_node(false);
        }
        domain
    }

    /// Pushes a new node, `taken` as given, and returns it.
    fn push_node(&self, taken: bool) -> *mut SlotNode {
        let node = Box::into_raw(Box::new(SlotNode {
            state: CachePadded::new(AtomicU64::new(IDLE)),
            taken: AtomicBool::new(taken),
            next: ptr::null_mut(),
        }));
        let mut head = self.head.load(Ordering::Acquire);
        loop {
            // SAFETY: `node` came from `Box::into_raw` just above and is not yet published.
            unsafe { (*node).next = head };
            match self
                .head
                .compare_exchange(head, node, Ordering::SeqCst, Ordering::SeqCst)
            {
                Ok(_) => return node,
                Err(current) => head = current,
            }
        }
    }

    /// A slot for the calling thread's use: a free one, or a new one.
    pub fn register(self: &Arc<Self>) -> EpochSlot {
        let mut node = self.head.load(Ordering::Acquire);
        // SAFETY: nodes live until the domain drops, and `self` is borrowed.
        while let Some(candidate) = unsafe { node.as_ref() } {
            if candidate
                .taken
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
            {
                return EpochSlot::new(self.clone(), node);
            }
            node = candidate.next;
        }
        EpochSlot::new(self.clone(), self.push_node(true))
    }

    /// The number of slots, taken or free.
    pub fn slot_count(&self) -> usize {
        let mut count = 0;
        let mut node = self.head.load(Ordering::Acquire);
        // SAFETY: as in `register`.
        while let Some(current) = unsafe { node.as_ref() } {
            count += 1;
            node = current.next;
        }
        count
    }

    /// Begins a new epoch and returns its number: every read section that starts after this call sees what was
    /// published before it.
    fn advance(&self) -> u64 {
        self.epoch.fetch_add(1, Ordering::SeqCst) + 1
    }

    /// The oldest announcement of a thread in a read section, or `None` when every thread is idle.
    fn floor(&self) -> Option<u64> {
        // Pairs with the barrier a thread issues when it leaves idle: either its announcement is seen here, or its
        // reads see what was published before this call. Readers that skip their fence are covered by the heavy
        // barrier, which also orders this thread's own earlier store (the pointer swap) before the loads below.
        fence(Ordering::SeqCst);
        if self.asymmetric {
            barrier::heavy();
        }
        let mut floor: Option<u64> = None;
        let mut node = self.head.load(Ordering::Acquire);
        // SAFETY: as in `register`.
        while let Some(current) = unsafe { node.as_ref() } {
            let state = current.state.load(Ordering::SeqCst);
            if state != IDLE {
                floor = Some(floor.map_or(state, |floor| floor.min(state)));
            }
            node = current.next;
        }
        floor
    }

    /// Whether every thread is idle or has announced epoch `target` or later.
    fn passed(&self, target: u64) -> bool {
        self.floor().is_none_or(|floor| floor >= target)
    }
}

impl Drop for EpochDomain {
    fn drop(&mut self) {
        #[cfg(not(loom))]
        let mut node = *self.head.get_mut();
        #[cfg(loom)]
        let mut node = self.head.with_mut(|head| *head);
        while !node.is_null() {
            // SAFETY: every node came from `Box::into_raw`; `&mut self` rules out any other access, and every slot
            // holds an `Arc` of the domain, so none is alive.
            let boxed = unsafe { Box::from_raw(node) };
            node = boxed.next;
        }
    }
}

impl fmt::Debug for EpochDomain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EpochDomain")
            .field("epoch", &self.epoch.load(Ordering::Relaxed))
            .field("slots", &self.slot_count())
            .finish()
    }
}

/// One thread's slot in an [`EpochDomain`]. The slot belongs to the thread that uses it: it is `Send`, so a thread
/// can be handed its slot, and not `Sync`.
///
/// - A thread that works in repeated passes calls [`quiescent`](Self::quiescent) at the start of each pass, which promises it holds no
///   reference to a published value from before the call, and [`idle`](Self::idle) before it parks. Between the
///   two, the references it loaded stay valid, and a [`pin`](Self::pin) costs one load.
/// - A thread that is not in a pass [`pin`](Self::pin)s for the length of a read.
///
/// Dropping the slot makes it idle and frees it for the next thread.
pub struct EpochSlot {
    domain: Arc<EpochDomain>,
    node: *const SlotNode,
    /// Whether the slot holds an announcement. Only the owning thread reads and writes it.
    active: Cell<bool>,
    /// The guards alive: while any is, the announcement only holds still or ages, so what the guards protect stays
    /// protected whatever the owner calls.
    pins: Cell<usize>,
    /// Whether the outermost guard announced the slot, and so withdraws it when the last guard drops.
    pinned_from_idle: Cell<bool>,
    /// Whether [`idle`](Self::idle) arrived while a guard was alive: the last guard withdraws the announcement.
    idle_requested: Cell<bool>,
}

// SAFETY: the node is reached through atomics and outlives the slot's `Arc` of the domain; the `Cell`s are used by
// the owning thread alone, which `!Sync` guarantees while `Send` moves the whole slot.
unsafe impl Send for EpochSlot {}

impl EpochSlot {
    fn new(domain: Arc<EpochDomain>, node: *const SlotNode) -> Self {
        Self {
            domain,
            node,
            active: Cell::new(false),
            pins: Cell::new(0),
            pinned_from_idle: Cell::new(false),
            idle_requested: Cell::new(false),
        }
    }

    fn state(&self) -> &AtomicU64 {
        // SAFETY: the node lives as long as the domain, which this slot holds.
        unsafe { &(*self.node).state }
    }

    /// The domain this slot belongs to.
    pub fn domain(&self) -> &Arc<EpochDomain> {
        &self.domain
    }

    /// Whether the slot holds an announcement now.
    pub fn is_active(&self) -> bool {
        self.active.get()
    }

    /// Announces the current epoch from idle. Where the writer forces a barrier on the readers (see
    /// [`EpochDomain`]) this is a release store and a compiler fence, with no barrier instruction; otherwise a
    /// sequentially consistent store and fence keep the loads that follow from passing the announcement.
    fn announce_from_idle(&self) {
        let epoch = self.domain.epoch.load(Ordering::Acquire);
        if self.domain.asymmetric {
            self.state().store(epoch, Ordering::Release);
            compiler_fence(Ordering::SeqCst);
        } else {
            self.state().store(epoch, Ordering::SeqCst);
            fence(Ordering::SeqCst);
        }
        self.active.set(true);
    }

    fn withdraw(&self) {
        self.state().store(IDLE, Ordering::Release);
        self.active.set(false);
    }

    /// Announces that this thread holds no reference to a published value from before this call, and that the
    /// values it loads from now on stay valid until the next call of [`quiescent`](Self::quiescent) or
    /// [`idle`](Self::idle).
    ///
    /// For an active slot this is a load of the epoch and a release store: a writer that reads the older
    /// announcement waits, which is safe. Leaving idle also orders the announcement before the loads that follow (see
    /// [`EpochDomain`]). While a guard is alive the call changes nothing: the guard's values were loaded before it.
    pub fn quiescent(&self) {
        if self.pins.get() > 0 {
            return;
        }
        if self.active.get() {
            let epoch = self.domain.epoch.load(Ordering::Acquire);
            self.state().store(epoch, Ordering::Release);
        } else {
            self.announce_from_idle();
        }
    }

    /// Withdraws the announcement: the thread holds no reference to a published value, and writers stop waiting
    /// for it. While a guard is alive the withdrawal waits for the last guard to drop.
    pub fn idle(&self) {
        if self.pins.get() > 0 {
            self.idle_requested.set(true);
            return;
        }
        self.withdraw();
    }

    /// Keeps the values loaded under the guard valid until it drops.
    ///
    /// An active slot is already announced at an epoch no later than now: the guard changes nothing and the values
    /// stay valid until the slot's next quiescent point after the guard drops. An idle slot announces the current
    /// epoch, with a fence, and the last guard withdraws it.
    pub fn pin(&self) -> EpochGuard<'_> {
        if self.pins.get() == 0 && !self.active.get() {
            self.announce_from_idle();
            self.pinned_from_idle.set(true);
        }
        self.pins.set(self.pins.get() + 1);
        EpochGuard { slot: self }
    }
}

impl Drop for EpochSlot {
    fn drop(&mut self) {
        self.state().store(IDLE, Ordering::Release);
        // SAFETY: as in `state`.
        unsafe { (*self.node).taken.store(false, Ordering::Release) };
    }
}

impl fmt::Debug for EpochSlot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EpochSlot")
            .field("active", &self.active.get())
            .finish_non_exhaustive()
    }
}

/// A read section of an [`EpochSlot`]: the [`Published`] values loaded under it stay valid while it lives.
#[must_use = "the values loaded under the guard are valid while it lives"]
pub struct EpochGuard<'a> {
    slot: &'a EpochSlot,
}

impl Drop for EpochGuard<'_> {
    fn drop(&mut self) {
        let slot = self.slot;
        let pins = slot.pins.get() - 1;
        slot.pins.set(pins);
        if pins == 0 && (slot.pinned_from_idle.replace(false) | slot.idle_requested.replace(false))
        {
            slot.withdraw();
        }
    }
}

/// A retired value waiting for its readers.
struct Deferred<T> {
    /// `Arc::into_raw` of the value.
    value: *const T,
    /// The epoch whose readers cannot hold the value.
    target: u64,
    next: *mut Deferred<T>,
}

/// An atomically replaceable `Option<Arc<T>>` with reads that take no lock and no reference count.
///
/// This is the stand-in for the garbage collector where a reference is published and later replaced or cleared
/// while readers may still hold the old one. A reader loads under an [`EpochGuard`]; a writer
/// ([`store`](Self::store), lifecycle work) swaps the pointer and gets a [`Retired`] for the old value, which it
/// frees by waiting for the readers ([`Retired::wait`]) or leaves to the cell ([`Retired::defer`]).
///
/// A store never waits and never fails: the retirements a cell has not yet been able to free chain on a list, and
/// each store, deferral and [`try_reclaim`](Self::try_reclaim) frees those the readers have passed. A value is
/// dropped on the thread that finds it free.
///
/// The destructor of `T` must not panic: a sweep takes the list whole and drops the freeable values one by one, so
/// a panic in one leaves the rest of that sweep's list unfreed. (A cell holding weak references or small
/// records has such destructors.)
pub struct Published<T> {
    domain: Arc<EpochDomain>,
    /// `Arc::into_raw` of the current value, or null.
    current: AtomicPtr<T>,
    /// The retirements not yet freed, most recent first.
    retired: AtomicPtr<Deferred<T>>,
}

// SAFETY: the cell owns `Arc<T>` references (moved between threads: `T: Send + Sync` for an `Arc`) and shares `&T`
// with every thread.
unsafe impl<T: Send + Sync> Send for Published<T> {}
// SAFETY: as above.
unsafe impl<T: Send + Sync> Sync for Published<T> {}

impl<T: Send + Sync> Published<T> {
    /// A cell in `domain` holding `value`.
    pub fn new(domain: Arc<EpochDomain>, value: Option<Arc<T>>) -> Self {
        Self {
            domain,
            current: AtomicPtr::new(into_raw(value)),
            retired: AtomicPtr::new(ptr::null_mut()),
        }
    }

    /// The domain readers of this cell pin.
    pub fn domain(&self) -> &Arc<EpochDomain> {
        &self.domain
    }

    /// The current value, valid while `guard` lives. An acquire load: no reference count, no read-modify-write.
    ///
    /// # Panics
    ///
    /// If `guard` belongs to another domain.
    pub fn load<'a>(&'a self, guard: &'a EpochGuard<'_>) -> Option<&'a T> {
        assert!(
            Arc::ptr_eq(&self.domain, guard.slot.domain()),
            "a guard protects the values of its own domain only"
        );
        let raw = self.current.load(Ordering::Acquire);
        // SAFETY: a non-null pointer came from `Arc::into_raw` and is freed only after every announcement has
        // passed the epoch that began after it was swapped out. The guard announced this thread before the load
        // (with a fence when it left idle), so a writer that swapped the pointer out either sees the announcement
        // and waits, or published its replacement before this load.
        unsafe { raw.as_ref() }
    }

    /// Publishes `value` and returns the retirement of the value it replaced.
    pub fn store(&self, value: Option<Arc<T>>) -> Retired<'_, T> {
        let old = self.current.swap(into_raw(value), Ordering::SeqCst);
        Retired {
            cell: self,
            old,
            target: self.domain.advance(),
        }
    }

    /// Publishes `value` if the cell still holds `expected` (`None`: still holds nothing), and returns the
    /// retirement of the old value; otherwise hands `value` back (Java's compare-and-set).
    ///
    /// `expected` is a reference, normally the one a [`load`](Self::load) under a guard returned: while the caller
    /// holds it the allocation cannot be freed and its address reused, so a match is a match of the value, not of
    /// a recycled address.
    pub fn store_if_current(
        &self,
        expected: Option<&T>,
        value: Option<Arc<T>>,
    ) -> Result<Retired<'_, T>, Option<Arc<T>>> {
        let new = into_raw(value);
        let expected = expected.map_or(ptr::null(), ptr::from_ref);
        match self.current.compare_exchange(
            expected.cast_mut(),
            new,
            Ordering::SeqCst,
            Ordering::SeqCst,
        ) {
            Ok(old) => Ok(Retired {
                cell: self,
                old,
                target: self.domain.advance(),
            }),
            // SAFETY: `new` came from `into_raw` just above and was not published.
            Err(_) => Err(unsafe { from_raw(new) }),
        }
    }

    /// Frees the retirements the readers have passed. Cheap when none is pending.
    pub fn try_reclaim(&self) {
        if self.retired.load(Ordering::Relaxed).is_null() {
            return;
        }
        let mut list = self.retired.swap(ptr::null_mut(), Ordering::AcqRel);
        if list.is_null() {
            return;
        }
        let floor = self.domain.floor();
        let mut kept: *mut Deferred<T> = ptr::null_mut();
        let mut kept_last: *mut Deferred<T> = ptr::null_mut();
        while !list.is_null() {
            // SAFETY: the swap above took the whole list; its nodes came from `Box::into_raw` and are ours alone.
            let node = unsafe { Box::from_raw(list) };
            list = node.next;
            if floor.is_none_or(|floor| floor >= node.target) {
                // SAFETY: `value` came from `Arc::into_raw`, and the readers have passed its epoch.
                drop(unsafe { Arc::from_raw(node.value) });
            } else {
                let node = Box::into_raw(node);
                // SAFETY: the node is ours until it is pushed back below.
                unsafe { (*node).next = kept };
                if kept_last.is_null() {
                    kept_last = node;
                }
                kept = node;
            }
        }
        if !kept.is_null() {
            self.push_chain(kept, kept_last);
        }
    }

    /// The number of retirements waiting; for tests and metrics. A reclaim that runs meanwhile holds its list
    /// away from the count, so the number is a lower bound.
    pub fn pending(&self) -> usize {
        let list = self.retired.swap(ptr::null_mut(), Ordering::AcqRel);
        let mut count = 0;
        let mut last = list;
        let mut node = list;
        // SAFETY: the swap took the whole list, so its nodes are ours until they are pushed back below.
        while let Some(current) = unsafe { node.as_ref() } {
            count += 1;
            last = node;
            node = current.next;
        }
        if !list.is_null() {
            self.push_chain(list, last);
        }
        count
    }

    /// Pushes the chain `first ..= last` on the retirement list.
    fn push_chain(&self, first: *mut Deferred<T>, last: *mut Deferred<T>) {
        let mut head = self.retired.load(Ordering::Acquire);
        loop {
            // SAFETY: the chain is ours until the compare-and-set publishes it.
            unsafe { (*last).next = head };
            match self
                .retired
                .compare_exchange(head, first, Ordering::AcqRel, Ordering::Acquire)
            {
                Ok(_) => return,
                Err(current) => head = current,
            }
        }
    }

    fn defer_raw(&self, old: *const T, target: u64) {
        if old.is_null() {
            return;
        }
        let node = Box::into_raw(Box::new(Deferred {
            value: old,
            target,
            next: ptr::null_mut(),
        }));
        self.push_chain(node, node);
        self.try_reclaim();
    }
}

impl<T> Drop for Published<T> {
    fn drop(&mut self) {
        #[cfg(not(loom))]
        let (current, mut list) = (*self.current.get_mut(), *self.retired.get_mut());
        #[cfg(loom)]
        let (current, mut list) = (
            self.current.with_mut(|current| *current),
            self.retired.with_mut(|retired| *retired),
        );
        if !current.is_null() {
            // SAFETY: `current` came from `Arc::into_raw`; `&mut self` means no reader holds a reference.
            drop(unsafe { Arc::from_raw(current) });
        }
        while !list.is_null() {
            // SAFETY: as above, and the nodes came from `Box::into_raw`.
            let node = unsafe { Box::from_raw(list) };
            list = node.next;
            // SAFETY: `value` came from `Arc::into_raw`.
            drop(unsafe { Arc::from_raw(node.value) });
        }
    }
}

impl<T> fmt::Debug for Published<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Published").finish_non_exhaustive()
    }
}

fn into_raw<T>(value: Option<Arc<T>>) -> *mut T {
    value.map_or(ptr::null_mut(), |value| Arc::into_raw(value).cast_mut())
}

/// # Safety
///
/// `raw` is null or the result of [`into_raw`], and not freed since.
unsafe fn from_raw<T>(raw: *mut T) -> Option<Arc<T>> {
    // SAFETY: the caller's contract.
    (!raw.is_null()).then(|| unsafe { Arc::from_raw(raw) })
}

/// The value a [`Published::store`] replaced, until the readers that may hold it have passed.
///
/// Free it by [`wait`](Self::wait)ing (lifecycle threads) or hand it to the cell with [`defer`](Self::defer).
/// Dropping a `Retired` defers it, so a replaced value is never lost or freed early.
#[must_use = "dropping a retirement defers it to the cell"]
pub struct Retired<'a, T: Send + Sync> {
    cell: &'a Published<T>,
    old: *const T,
    target: u64,
}

// SAFETY: the retirement owns an `Arc<T>` reference, which moves between threads when `T: Send + Sync`.
unsafe impl<T: Send + Sync> Send for Retired<'_, T> {}

impl<T: Send + Sync> Retired<'_, T> {
    /// Whether every reader that could hold the value has passed.
    pub fn is_free(&self) -> bool {
        self.old.is_null() || self.cell.domain.passed(self.target)
    }

    /// Frees the value once every reader has passed, spinning until `deadline`; hands the retirement back if the
    /// readers have not passed by then. For lifecycle threads that are not themselves readers in a pass: a reader
    /// that waits on its own announcement waits for the deadline.
    pub fn wait(self, deadline: Instant) -> Result<(), Self> {
        loop {
            if self.is_free() {
                let this = ManuallyDrop::new(self);
                if !this.old.is_null() {
                    // SAFETY: `old` came from `Arc::into_raw`, and the readers have passed its epoch.
                    drop(unsafe { Arc::from_raw(this.old) });
                }
                return Ok(());
            }
            if Instant::now() >= deadline {
                return Err(self);
            }
            back_off();
        }
    }

    /// Leaves the value to the cell, which frees it once the readers have passed: at the cell's next store,
    /// deferral or [`Published::try_reclaim`], or when the cell drops.
    pub fn defer(self) {
        // `Drop` does the deferral.
        drop(self);
    }
}

impl<T: Send + Sync> Drop for Retired<'_, T> {
    fn drop(&mut self) {
        self.cell.defer_raw(self.old, self.target);
    }
}

#[allow(clippy::disallowed_methods, reason = "LIFECYCLE(C5)")]
fn back_off() {
    std::thread::yield_now();
}

#[cfg(test)]
mod tests;
