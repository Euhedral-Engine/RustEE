//! Routing vertices, and the interceptor that feeds sources into them.
//!
//! A vertex publishes its routing table as a [`RoutingState`], one generation at a time, as Java's `LatticeVertex`
//! publishes its `routingState`. Each state carries an admission count: a reader takes a count by compare-and-set
//! while the state is not retired, re-checks that the state is still the published one, routes, and releases. A
//! remap builds the next generation under the rebalance lock, publishes it and retires the old one; the release
//! that leaves a retired state without readers closes the downstreams no generation routes to, on whichever thread
//! that is.
//!
//! Java's collector keeps a retired state alive for the readers that loaded its pointer just before the swap. Here
//! the vertex owns every state it allocated until it drops, and recycles a retired state for the next generation
//! once its readers are gone and its handles are released, so a stale reader's compare-and-set always lands on a
//! live `RoutingState`; the re-check against the published pointer keeps it from using a state that is no longer
//! current.

// States are allocated once, owned by the vertex's arena and freed when the vertex drops, so a reader that loaded
// a state's address may compare-and-set its admission count whatever the state has become since.
#![allow(
    unsafe_code,
    reason = "routing states live until the vertex drops, so a stale reader's pointer stays valid"
)]

use super::edge::{LatticeEdge, LatticeNode, Retirement};
use super::registry::UpstreamRegistry;
use super::upstream_queue::{Claim, UpstreamHandle, note_routed_push};
use crate::frames::{Failure, Frame, FrameError};
use crate::generics::{LatticeReceiver, LatticeSource, LatticeTerminal, SourceCell};
use crate::utils::math::unsigned_multiply_high;
use euhedral_data_structures::atomics::CachePadded;
use euhedral_data_structures::queues::{Consumer, MpscQueue, Multi, Producer, Single};
use euhedral_data_structures::sync::{Admission, HotScope, LifecycleMutex, SetOnce};
use std::cell::UnsafeCell;
use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::ops::Deref;
use std::panic::{self, AssertUnwindSafe};
use std::ptr::{self, NonNull};
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, Ordering};
use std::sync::{Arc, Weak};

/// Picks the active downstream for a frame: an index below `state.len()` into
/// [`RoutingState::mappings`].
pub type RoutingFunction = Arc<dyn Fn(&dyn Frame, &RoutingState) -> usize + Send + Sync>;

/// The default routing: spreads frames over `map_size` slots by their routing hash, so frames with
/// the same routing hash take the same slot.
pub fn route_by_hash(frame: &dyn Frame, map_size: usize) -> usize {
    unsigned_multiply_high(frame.header().routing_hash(), map_size as u64) as usize
}

/// Returned when a closed vertex is asked to take on an upstream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LatticeClosed;

impl fmt::Display for LatticeClosed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("the vertex is closed")
    }
}

impl Error for LatticeClosed {}

/// Identifies a downstream node by the address of its edge.
fn key(edge: &LatticeEdge) -> usize {
    edge as *const LatticeEdge as usize
}

/// What a generation holds: the routing table and the downstreams it routes to.
#[derive(Default)]
struct Table {
    mappings: Vec<usize>,
    active_indexes: Vec<Option<usize>>,
    handles: Vec<Option<Arc<dyn LatticeNode>>>,
}

impl Table {
    fn empty(downstream_count: usize) -> Self {
        Self {
            mappings: Vec::new(),
            active_indexes: vec![None; downstream_count],
            handles: vec![None; downstream_count],
        }
    }

    /// Each active downstream once, however many ids map to it.
    fn unique_handles(&self) -> impl Iterator<Item = &Arc<dyn LatticeNode>> {
        let mut seen = Vec::with_capacity(self.mappings.len());
        self.mappings.iter().filter_map(move |&id| {
            let handle = self.handles[id].as_ref()?;
            let key = key(handle.edge());
            (!seen.contains(&key)).then(|| {
                seen.push(key);
                handle
            })
        })
    }
}

/// One generation of a vertex's routing table: which downstreams are active, in what order.
///
/// A generation is immutable while it is published or read. Remapping publishes a new one, and a push routes with
/// the generation it took a count on. Holding a generation keeps its downstreams open: a downstream that a remap
/// drops is closed once no generation that routes to it is held any more.
///
/// A state is never freed before its vertex drops; once it is retired, has no readers and has released its
/// handles, a remap may rewrite it as the next generation (see the module documentation).
pub struct RoutingState {
    /// Java's `admission`: the readers' count, with the retired bit set once the state is no longer the one to read.
    admission: CachePadded<Admission>,
    /// Java's `handlesReleased`: taken by the one thread that releases the state's handles.
    handles_released: AtomicBool,
    /// Set, under the vertex's book, once the handles are released: the state may be rewritten.
    handles_done: AtomicBool,
    /// Rewritten only by a remap, while the state is retired and idle and `handles_done` is set, and read by a
    /// thread that holds a count or, to release the handles, by the one that took `handles_released`.
    table: UnsafeCell<Table>,
}

// SAFETY: `table` is written only while no thread can read it (see its documentation) and read only by threads
// that hold a count or the release; every other field is atomic.
unsafe impl Sync for RoutingState {}

impl RoutingState {
    fn new(table: Table) -> Self {
        Self {
            admission: CachePadded::new(Admission::new()),
            handles_released: AtomicBool::new(false),
            handles_done: AtomicBool::new(false),
            table: UnsafeCell::new(table),
        }
    }

    fn table(&self) -> &Table {
        // SAFETY: a `&RoutingState` reaches a reader only through a guard that holds a count, which keeps the
        // state from being rewritten; the release path reads it under its exclusive claim.
        unsafe { &*self.table.get() }
    }

    /// The number of active downstreams.
    pub fn len(&self) -> usize {
        self.table().mappings.len()
    }

    pub fn is_empty(&self) -> bool {
        self.table().mappings.is_empty()
    }

    /// The ids of the active downstreams, in ascending order: a routing index selects one.
    pub fn mappings(&self) -> &[usize] {
        &self.table().mappings
    }

    /// The routing index of downstream `downstream_id`, or `None` if it is inactive.
    pub fn active_index(&self, downstream_id: usize) -> Option<usize> {
        self.table()
            .active_indexes
            .get(downstream_id)
            .copied()
            .flatten()
    }

    /// Downstream `downstream_id`, if it is active.
    pub fn downstream(&self, downstream_id: usize) -> Option<&Arc<dyn LatticeNode>> {
        self.table().handles.get(downstream_id)?.as_ref()
    }
}

/// A reader's count on a [`RoutingState`], given back when dropped, also when the reader panics. Holding it keeps
/// the generation's downstreams open.
pub struct RouteGuard<'a> {
    vertex: &'a LatticeVertex,
    state: &'a RoutingState,
}

impl Deref for RouteGuard<'_> {
    type Target = RoutingState;

    fn deref(&self) -> &RoutingState {
        self.state
    }
}

impl Drop for RouteGuard<'_> {
    fn drop(&mut self) {
        self.vertex.release_route(self.state);
    }
}

/// The routing states a vertex allocated. They are freed when the vertex drops and not before: the address of one
/// may sit in a reader's hands for as long as the vertex lives.
#[derive(Default)]
struct Arena {
    states: Vec<NonNull<RoutingState>>,
}

// SAFETY: the states are `Sync`, and the arena frees them only when it drops, with exclusive access.
unsafe impl Send for Arena {}

impl Arena {
    fn alloc(&mut self, table: Table) -> NonNull<RoutingState> {
        let state = NonNull::from(Box::leak(Box::new(RoutingState::new(table))));
        self.states.push(state);
        state
    }

    /// A state that may be rewritten, if one is: retired, with no readers, and with its handles released.
    fn reusable(&self) -> Option<NonNull<RoutingState>> {
        self.states.iter().copied().find(|state| {
            // SAFETY: states live as long as the arena.
            let state = unsafe { state.as_ref() };
            state.admission.is_retired_and_idle() && state.handles_done.load(Ordering::Acquire)
        })
    }

    /// The next generation, closed to readers until it is published: a recycled state rewritten with `table`, or
    /// a new one. The state stays valid until the arena drops.
    fn next_state(&mut self, table: Table) -> NonNull<RoutingState> {
        let Some(found) = self.reusable() else {
            return self.alloc(table);
        };
        // SAFETY: states live as long as the arena.
        let state = unsafe { found.as_ref() };
        // SAFETY: the state is retired and idle (no reader, and every `try_acquire` refuses) and the handles are
        // released, so no thread reads the table; the caller holds the vertex's book, which keeps the other
        // writers out.
        unsafe { *state.table.get() = table };
        state.handles_released.store(false, Ordering::Relaxed);
        state.handles_done.store(false, Ordering::Relaxed);
        // Opens the state to readers: the rewrite above is visible to whoever acquires a count.
        state.admission.reopen();
        found
    }
}

impl Drop for Arena {
    fn drop(&mut self) {
        for state in self.states.drain(..) {
            // SAFETY: each state came from `Box::leak` in `alloc`, is freed only here, and the vertex that
            // owns the arena is gone, so no reader holds it.
            drop(unsafe { Box::from_raw(state.as_ptr()) });
        }
    }
}

/// A downstream some generation still routes to.
struct Reference {
    generations: usize,
    retirements: Vec<Retirement>,
}

/// The rebalance state, under the vertex's lifecycle lock (Java's `routingUpdateLock`).
struct Book {
    references: HashMap<usize, Reference>,
    routing_closed: bool,
    close_retirements: Vec<Retirement>,
    arena: Arena,
}

/// A node that routes each frame to exactly one of a fixed number of downstream nodes.
///
/// The downstreams are numbered `0..downstream_count`; a [mapping](Self::set_downstream_mapping)
/// makes some of them active, and the [routing function](RoutingFunction) picks among the active
/// ones. The default routes by hash, so frames with the same routing hash stay together while the
/// mapping holds. Mapping, closing and retirement all wait for pushes in flight on the previous
/// generation: see [`RoutingState`].
///
/// Sources join the vertex through [`ingest`](Self::ingest), which wraps each in an
/// [`UpstreamInterceptor`].
pub struct LatticeVertex {
    edge: LatticeEdge,
    name: String,
    downstream_count: usize,
    route: RoutingFunction,
    /// Java's `routingState`: the published generation, set with release ordering after it is built.
    routing: AtomicPtr<RoutingState>,
    /// Java's `routingUpdateLock` and the state it guards.
    book: LifecycleMutex<Book>,
    closed: AtomicBool,
    upstream_sequence: AtomicU64,
    this: Weak<LatticeVertex>,
}

impl LatticeVertex {
    /// A vertex in `registry` with `downstream_count` downstream slots, routing by hash. `name`
    /// is the log target.
    pub fn new(
        registry: &Arc<UpstreamRegistry>,
        name: impl Into<String>,
        downstream_count: usize,
    ) -> Arc<Self> {
        Self::with_routing(registry, name, downstream_count, |frame, state| {
            route_by_hash(frame, state.len())
        })
    }

    /// Like [`new`](Self::new), routing with `route`.
    pub fn with_routing(
        registry: &Arc<UpstreamRegistry>,
        name: impl Into<String>,
        downstream_count: usize,
        route: impl Fn(&dyn Frame, &RoutingState) -> usize + Send + Sync + 'static,
    ) -> Arc<Self> {
        let name = name.into();
        let mut arena = Arena::default();
        let first = arena.alloc(Table::empty(downstream_count));
        Arc::new_cyclic(|this: &Weak<Self>| Self {
            edge: LatticeEdge::for_node(registry, Arc::new(AtomicBool::new(false)), this.clone()),
            name,
            downstream_count,
            route: Arc::new(route),
            routing: AtomicPtr::new(first.as_ptr()),
            book: LifecycleMutex::new(Book {
                references: HashMap::new(),
                routing_closed: false,
                close_retirements: Vec::new(),
                arena,
            }),
            closed: AtomicBool::new(false),
            upstream_sequence: AtomicU64::new(0),
            this: this.clone(),
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn downstream_count(&self) -> usize {
        self.downstream_count
    }

    /// Sets or clears the drain flag. While it is set, demand is ignored, new upstreams wait, and
    /// the mapping may change.
    pub fn set_drain(&self, drain: bool) {
        self.edge.drain_flag().store(drain, Ordering::Release);
    }

    fn is_draining(&self) -> bool {
        self.edge.drain_flag().load(Ordering::Acquire)
    }

    /// The current routing generation, with a reader's count on it: holding the guard keeps the downstreams it
    /// routes to open. A closed vertex's generation routes nowhere.
    pub fn routing_state(&self) -> RouteGuard<'_> {
        loop {
            if let Some(guard) = self.try_acquire_current() {
                return guard;
            }
        }
    }

    /// The routing index of downstream `downstream_id` in the current generation.
    pub fn active_downstream_index(&self, downstream_id: usize) -> Option<usize> {
        self.routing_state().active_index(downstream_id)
    }

    /// Downstream `downstream_id`, if it is active in the current generation.
    pub fn downstream(&self, downstream_id: usize) -> Option<Arc<dyn LatticeNode>> {
        self.routing_state().downstream(downstream_id).cloned()
    }

    /// Java's `acquireRoutingState`: a count on the published generation, or `None` once the vertex is closed.
    /// Each retry follows a change of the published generation, so a reader never waits for the writer.
    fn acquire_routing(&self) -> Option<RouteGuard<'_>> {
        while !self.closed.load(Ordering::Acquire) {
            if let Some(guard) = self.try_acquire_current() {
                if self.closed.load(Ordering::Acquire) {
                    drop(guard);
                    return None;
                }
                return Some(guard);
            }
        }
        None
    }

    /// Java's `tryAcquireRoutingState`: loads the published pointer, takes a count, and checks the pointer again.
    /// A state that was recycled since the load is either the published one now, and right to use, or is not, and
    /// the count is given back.
    fn try_acquire_current(&self) -> Option<RouteGuard<'_>> {
        let pointer = self.routing.load(Ordering::Acquire);
        // SAFETY: states are freed only when the vertex drops, which the borrow of `self` rules out.
        let state = unsafe { &*pointer };
        if !state.admission.try_acquire() {
            return None;
        }
        let guard = RouteGuard {
            vertex: self,
            state,
        };
        // Dropping the guard of a state that is no longer published gives the count back.
        (ptr::eq(pointer, self.routing.load(Ordering::Acquire))).then_some(guard)
    }

    /// Java's `releaseRoute`: gives a count back; the last reader of a retired state releases its handles.
    fn release_route(&self, state: &RoutingState) {
        if state.admission.release() {
            self.release_handles_when_quiescent(state);
        }
    }

    /// Publishes a new routing generation in which the downstreams with ids in `active` are
    /// active, `handles[id]` being downstream `id`. The vertex becomes their parent. Downstreams
    /// the new generation drops are closed once the generations routing to them are released.
    ///
    /// Returns false, changing nothing, unless the vertex is draining and open, every active id is
    /// below [`downstream_count`](Self::downstream_count), and every active id has a handle. The
    /// caller must stop pushes for the drain to mean anything: setting the flag does not wait for
    /// pushes in flight.
    ///
    /// # Panics
    ///
    /// Panics if an active handle belongs to another registry.
    pub fn set_downstream_mapping(
        &self,
        active: &[usize],
        handles: &[Option<Arc<dyn LatticeNode>>],
    ) -> bool {
        let mut mappings = active.to_vec();
        mappings.sort_unstable();
        mappings.dedup();
        let valid = mappings
            .iter()
            .all(|&id| id < self.downstream_count && handles.get(id).is_some_and(Option::is_some));
        if !valid {
            return false;
        }
        let this: Arc<dyn LatticeNode> = self.this.upgrade().expect("a live vertex is shared");
        for &id in &mappings {
            let handle = handles[id].as_ref().expect("validated above");
            assert!(
                Arc::ptr_eq(self.edge.registry(), handle.edge().registry()),
                "linked edges must share one upstream registry"
            );
        }

        let mut table = Table::empty(self.downstream_count);
        for (index, &id) in mappings.iter().enumerate() {
            table.active_indexes[id] = Some(index);
            table.handles[id] = handles[id].clone();
        }
        table.mappings = mappings;

        let retired = {
            let mut book = self.book.lock();
            if !self.is_draining() || self.is_closed() {
                return false;
            }
            let next = book.arena.next_state(table);
            // SAFETY: states live until the vertex drops; the next one is closed to readers until it is
            // published below.
            let next_ref = unsafe { next.as_ref() };
            for handle in next_ref.table().unique_handles() {
                handle.edge().set_parent(Some(&this));
                book.references
                    .entry(key(handle.edge()))
                    .or_insert_with(|| Reference {
                        generations: 0,
                        retirements: Vec::new(),
                    })
                    .generations += 1;
            }
            self.publish(next)
        };
        self.release_handles_when_quiescent(retired);
        true
    }

    /// Publishes `next` and retires the generation it replaces, which it returns. The caller holds the book.
    ///
    /// Java retires the old state first and publishes second, so a reader that finds the retired state waits for
    /// the writer between the two. Here the new state is published first: a reader that loaded the old pointer
    /// either takes a count before the retirement and finds the pointer changed, or fails the count and loads the
    /// new pointer, and neither waits.
    fn publish(&self, next: NonNull<RoutingState>) -> &RoutingState {
        let old = self.routing.swap(next.as_ptr(), Ordering::AcqRel);
        // SAFETY: states live until the vertex drops.
        let old = unsafe { &*old };
        old.admission.retire();
        old
    }

    /// Java's `releaseHandlesWhenQuiescent`: once `state` is retired and its last reader has gone, closes the
    /// downstreams no other generation routes to and runs their retirements, then, once the vertex is closed and
    /// nothing routes anywhere, the close retirements. Rebalance work, run by whichever thread gets here first,
    /// once per generation.
    fn release_handles_when_quiescent(&self, state: &RoutingState) {
        if !state.admission.is_retired()
            || state.admission.has_readers()
            || state
                .handles_released
                .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
                .is_err()
        {
            return;
        }
        // The last reader of a retired generation retires it: rebalance work on whichever thread that is.
        let _cold = HotScope::leave();
        let mut closing = Vec::new();
        let mut retirements = Vec::new();
        let mut close_retirements = Vec::new();
        let finished;
        {
            let mut book = self.book.lock();
            // SAFETY: the state is retired with no readers, and this thread took `handles_released`, so no one
            // else reads or rewrites the table until `handles_done` is set below.
            let table = unsafe { &mut *state.table.get() };
            for handle in table.unique_handles() {
                let key = key(handle.edge());
                let Some(reference) = book.references.get_mut(&key) else {
                    continue;
                };
                reference.generations -= 1;
                if reference.generations == 0 {
                    let reference = book.references.remove(&key).expect("present above");
                    closing.push(handle.clone());
                    retirements.extend(reference.retirements);
                }
            }
            // The generation is over: let go of its handles (dropped below, outside the lock), and let a remap
            // reuse the state.
            finished = std::mem::take(table);
            state.handles_done.store(true, Ordering::Release);
            if self.is_closed() && book.references.is_empty() && !book.routing_closed {
                book.routing_closed = true;
                close_retirements = std::mem::take(&mut book.close_retirements);
            }
        }
        drop(finished);
        for handle in closing {
            if panic::catch_unwind(AssertUnwindSafe(|| handle.close())).is_err() {
                log::error!(target: &self.name, "failed to close a retired downstream");
            }
        }
        for action in retirements.into_iter().chain(close_retirements) {
            self.run_retirement(action);
        }
    }

    fn run_retirement(&self, action: Retirement) {
        if panic::catch_unwind(AssertUnwindSafe(action)).is_err() {
            log::error!(target: &self.name, "failed to retire a downstream owner");
        }
    }

    /// Closes the vertex to routing, then runs `after_routes_retire` once every generation has
    /// been released and every downstream closed; at once if that has already happened.
    pub fn close_after_routes_retire(&self, after_routes_retire: impl FnOnce() + Send + 'static) {
        let after_routes_retire: Retirement = Box::new(after_routes_retire);
        let retired = {
            let mut book = self.book.lock();
            if self.closed.load(Ordering::Acquire) {
                if !book.routing_closed {
                    book.close_retirements.push(after_routes_retire);
                    return;
                }
                drop(book);
                self.run_retirement(after_routes_retire);
                return;
            }
            self.closed.store(true, Ordering::SeqCst);
            book.close_retirements.push(after_routes_retire);
            let empty = book.arena.next_state(Table::empty(self.downstream_count));
            self.publish(empty)
        };
        self.edge.close_edge();
        self.release_handles_when_quiescent(retired);
    }

    /// Routes `frame` to one active downstream, or hands it back if none is active.
    ///
    /// # Panics
    ///
    /// Panics if the routing function returns an index outside the active downstreams.
    pub fn try_push(&self, frame: Box<dyn Frame>) -> Result<(), Box<dyn Frame>> {
        let _hot = HotScope::enter();
        let Some(state) = self.acquire_routing() else {
            return Err(frame);
        };
        if state.is_empty() {
            return Err(frame);
        }
        let index = (self.route)(&*frame, &state);
        assert!(
            index < state.len(),
            "the routing function chose index {index} of {} active downstreams",
            state.len()
        );
        let table = state.table();
        let downstream = table.handles[table.mappings[index]]
            .as_ref()
            .expect("an active downstream has a handle");
        downstream.push(frame);
        Ok(())
    }

    /// Wraps `source` in an [`UpstreamInterceptor`] that feeds this vertex, and adds it to the
    /// registry. A closed vertex completes `source` instead.
    pub fn ingest(
        &self,
        source: Arc<dyn LatticeSource>,
    ) -> Result<Arc<UpstreamInterceptor>, LatticeClosed> {
        if self.is_closed() {
            source.complete();
            return Err(LatticeClosed);
        }
        let interceptor = UpstreamInterceptor::new(self);
        source.clone().add_downstream(interceptor.clone());
        interceptor.attach(source);
        if interceptor.is_complete() && self.is_closed() {
            return Err(LatticeClosed);
        }
        Ok(interceptor)
    }

    /// Places this vertex below `upstream`, at the bottom of its branch.
    pub fn add_upstream_node(&self, upstream: &dyn LatticeNode) -> Result<(), LatticeClosed> {
        if self.is_closed() {
            return Err(LatticeClosed);
        }
        let this = self.this.upgrade().expect("a live vertex is shared");
        upstream.edge().add_downstream_node(this);
        Ok(())
    }
}

impl LatticeNode for LatticeVertex {
    fn edge(&self) -> &LatticeEdge {
        &self.edge
    }

    /// Routes `frame`, or finishes it with an error if no downstream is active.
    fn push(&self, frame: Box<dyn Frame>) {
        if let Err(frame) = self.try_push(frame) {
            frame.do_finally_with_error(unroutable());
        }
    }

    /// Passes `error` to every active downstream once.
    fn on_error(&self, error: Failure) {
        let Some(state) = self.acquire_routing() else {
            return;
        };
        let targets: Vec<_> = state.table().unique_handles().collect();
        match targets.as_slice() {
            [] => {}
            [only] => only.on_error(error),
            _ => {
                let shared: Arc<dyn Error + Send + Sync> = Arc::from(error);
                for target in targets {
                    target.on_error(Box::new(SharedFailure(shared.clone())));
                }
            }
        }
    }

    fn close(&self) {
        self.close_after_routes_retire(|| {});
    }

    fn is_closed(&self) -> bool {
        self.closed.load(Ordering::Acquire)
    }

    /// Runs `action` once no generation routes to `child`: at once if none does now.
    fn defer_child_retirement(&self, child: &LatticeEdge, action: Retirement) {
        let action = {
            let mut book = self.book.lock();
            match book.references.get_mut(&key(child)) {
                Some(reference) => {
                    reference.retirements.push(action);
                    return;
                }
                None => action,
            }
        };
        self.run_retirement(action);
    }
}

fn unroutable() -> FrameError {
    FrameError::failed("no active downstream to route the frame to")
}

/// One error handed to several downstreams.
#[derive(Debug)]
struct SharedFailure(Arc<dyn Error + Send + Sync>);

impl fmt::Display for SharedFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl Error for SharedFailure {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        self.0.source()
    }
}

/// An action [`LatticeReceiver::serialize`] hands to the interceptor.
type Action = Box<dyn FnOnce() + Send>;

/// Where the interceptor's claim ids come from. Zero is the id of a handle with no claim of its own.
static CLAIM_IDS: AtomicU64 = AtomicU64::new(1);

/// The source an interceptor feeds from: installed once, taken once.
///
/// Installing is a compare-and-set from empty, so it needs no claim: it is done before the
/// interceptor is registered, as the source is announced to the interceptor, and a worker that
/// finds the handle finds the source. Reading and taking happen only while the interceptor's claim
/// is held, which orders them against each other.
struct UpstreamSlot {
    source: AtomicPtr<Arc<dyn LatticeSource>>,
}

impl UpstreamSlot {
    const fn new() -> Self {
        Self {
            source: AtomicPtr::new(ptr::null_mut()),
        }
    }

    /// Installs `source` if the slot was never filled; returns whether it did.
    fn install(&self, source: Arc<dyn LatticeSource>) -> bool {
        let raw = Box::into_raw(Box::new(source));
        let installed = self
            .source
            .compare_exchange(ptr::null_mut(), raw, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok();
        if !installed {
            // SAFETY: `raw` came from `Box::into_raw` above and was not published.
            drop(unsafe { Box::from_raw(raw) });
        }
        installed
    }

    /// The source, if one is installed and not yet taken.
    ///
    /// # Safety
    ///
    /// The caller holds the interceptor's claim, which keeps `take` from running, and does not use
    /// the reference after it releases the claim.
    unsafe fn get(&self) -> Option<&Arc<dyn LatticeSource>> {
        // SAFETY: a non-null pointer was published by `install` from a `Box`, and is freed only by
        // `take` or by the drop of the slot. `take` runs under the claim, which the caller holds for
        // as long as it uses the reference, and the borrow of the slot rules out the drop.
        unsafe { self.source.load(Ordering::Acquire).as_ref() }
    }

    /// Takes the source out of the slot.
    ///
    /// # Safety
    ///
    /// The caller holds the interceptor's claim, which keeps `get` borrows from being alive.
    unsafe fn take(&self) -> Option<Arc<dyn LatticeSource>> {
        let raw = self.source.swap(ptr::null_mut(), Ordering::SeqCst);
        // SAFETY: a non-null pointer came from `Box::into_raw` in `install`, and the swap handed
        // its ownership to this call alone.
        (!raw.is_null()).then(|| *unsafe { Box::from_raw(raw) })
    }
}

impl Drop for UpstreamSlot {
    fn drop(&mut self) {
        let raw = *self.source.get_mut();
        if !raw.is_null() {
            // SAFETY: as in `take`; `&mut self` rules out any other access.
            drop(unsafe { Box::from_raw(raw) });
        }
    }
}

/// A source fed into a [`LatticeVertex`], as an [`UpstreamHandle`] workers service.
///
/// Frames the source pushes are routed by the vertex. A worker pulls or requests through the
/// interceptor only while it holds the interceptor's [`Claim`], so the source sees one worker at a
/// time. A panic in the source's pull or request is logged and completes the interceptor.
///
/// # The claim
///
/// The claim is a counter, `wip`. A caller that moves it from zero holds the claim; a caller that
/// finds it non-zero has lost and goes elsewhere. Nothing waits. [`try_acquire`] hands the winner a
/// [`Claim`], which `pull`, `request` and `release` require, so safe code cannot service the source
/// without holding the claim or give the claim back without having it.
///
/// [`try_acquire`]: UpstreamHandle::try_acquire
///
/// # Serialized actions
///
/// [`serialize`](LatticeReceiver::serialize) hands an action to whoever holds the claim, or runs it
/// itself when nobody does. An action never runs while a `pull` or `request` of the source is in
/// progress, runs exactly once, and runs in the order it was handed over.
///
/// The three steps that keep an action from being stranded:
///
/// 1. `serialize` offers the action to a queue, then publishes `pending`, then tries the claim
///    (`fetch_add`).
/// 2. A caller that takes the claim to run an action, and a holder that takes it back, run the
///    queued actions, clearing `pending` with one swap.
/// 3. A holder releases with one swap of `wip` to zero, and only a swap that returns more than 1
///    means a caller tried the claim meanwhile: then it takes the claim back and runs the actions.
///    The argument is on `release_claim`. A release nobody contended is the `fetch_add` and one
///    swap.
///
/// # The source
///
/// The source is installed before the interceptor is registered, and the interceptor lets it go
/// when it finishes, through `serialize`, so a source that holds the interceptor in turn is not
/// kept alive by the pair. The source's identity is a weak reference, whose allocation is not reused
/// while the interceptor lives, so a new source never compares equal to a dropped one.
pub struct UpstreamInterceptor {
    vertex: Weak<LatticeVertex>,
    registry: Arc<UpstreamRegistry>,
    log_target: String,
    sequence: u64,
    /// The id a [`Claim`] taken from this interceptor carries.
    claim_id: u64,
    wip: CachePadded<AtomicU64>,
    pending: AtomicBool,
    actions: Producer<Action, Multi>,
    drain: SourceCell<Consumer<Action, Single>>,
    upstream: UpstreamSlot,
    /// The source taken, for identity only.
    identity: SetOnce<Weak<dyn LatticeSource>>,
    complete: AtomicBool,
    this: Weak<UpstreamInterceptor>,
}

impl UpstreamInterceptor {
    fn new(vertex: &LatticeVertex) -> Arc<Self> {
        let actions = MpscQueue::new(16);
        Arc::new_cyclic(|this| Self {
            vertex: vertex.this.clone(),
            registry: vertex.edge.registry().clone(),
            log_target: vertex.name.clone(),
            sequence: vertex.upstream_sequence.fetch_add(1, Ordering::Relaxed),
            claim_id: CLAIM_IDS.fetch_add(1, Ordering::Relaxed),
            wip: CachePadded::new(AtomicU64::new(0)),
            pending: AtomicBool::new(false),
            actions: actions.producer(),
            drain: SourceCell::new(actions.consumer()),
            upstream: UpstreamSlot::new(),
            identity: SetOnce::new(),
            complete: AtomicBool::new(false),
            this: this.clone(),
        })
    }

    /// Whether `source` is the source this interceptor took. The interceptor lets its source go
    /// when it finishes, and still answers for it.
    pub fn has_upstream(&self, source: &Arc<dyn LatticeSource>) -> bool {
        self.identity.get().is_some_and(|attached| {
            attached.as_ptr().cast::<()>() == Arc::as_ptr(source).cast::<()>()
        })
    }

    /// Takes `source` as the upstream and adds the interceptor to the registry. Attaching the same
    /// source again does nothing; a different source is completed and left out.
    fn attach(&self, source: Arc<dyn LatticeSource>) {
        let weak = Arc::downgrade(&source);
        if let Err(other) = self.identity.set(weak) {
            let attached = self.identity.get().expect("set above");
            if !Weak::ptr_eq(attached, &other) {
                log::error!(target: &self.log_target, "an interceptor takes one source");
                source.complete();
            }
            return;
        }
        let this = self.this.upgrade().expect("a live interceptor");
        let handle: Arc<dyn UpstreamHandle> = this;
        // The source is in place before any worker can find the handle.
        let installed = self.upstream.install(source);
        debug_assert!(installed, "the identity admits one source");
        // The interceptor may have finished before the source arrived, with nothing to release.
        // Either the finisher's release sees the source installed above, or this check sees the
        // finish: both sides are sequentially consistent.
        if self.complete.load(Ordering::SeqCst) {
            self.let_go_of_the_source(true);
        }
        let vertex = self.vertex.upgrade();
        match &vertex {
            Some(vertex) => vertex.edge.add_upstream_handle(handle),
            None => self.registry.add_handle(handle),
        }
        if vertex.is_none_or(|vertex| vertex.is_closed()) {
            self.complete();
        }
    }

    /// Completes the interceptor and its source, and stops counting it as a live handle.
    pub fn complete(&self) {
        if self.finish() {
            log::trace!(target: &self.log_target, "closing an upstream interceptor");
            self.let_go_of_the_source(true);
        }
    }

    /// Marks the interceptor complete; returns whether this call did.
    fn finish(&self) -> bool {
        let finished = !self.complete.swap(true, Ordering::SeqCst);
        if finished {
            self.registry.remove_handle();
        }
        finished
    }

    /// Drops the source, after completing it if `complete_it`, once no `pull` or `request` of it
    /// is in progress.
    fn let_go_of_the_source(&self, complete_it: bool) {
        let Some(this) = self.this.upgrade() else {
            return;
        };
        self.serialize(Box::new(move || {
            // SAFETY: a serialized action runs under the claim.
            let source = unsafe { this.upstream.take() };
            if complete_it && let Some(source) = &source {
                source.complete();
            }
        }));
    }

    /// Whether a worker may service the source now.
    fn serviceable(&self, demand: u64) -> bool {
        if demand == 0 || self.is_complete() {
            return false;
        }
        self.vertex
            .upgrade()
            .is_some_and(|vertex| !vertex.is_closed() && !vertex.is_draining())
    }

    /// Checks that `claim` was taken from this interceptor.
    fn assert_own(&self, claim: &Claim) {
        assert_eq!(
            claim.owner(),
            self.claim_id,
            "a claim is serviced only by the handle it was taken from"
        );
    }

    /// Runs the queued actions. The caller holds the claim.
    fn run_actions(&self) {
        // One swap both reads and clears the flag, so an action offered after it keeps the flag set
        // for the next look.
        if !self.pending.swap(false, Ordering::SeqCst) {
            return;
        }
        // The caller holds the interceptor's claim, so no `pull` or `request` is in progress and
        // the cell's own claim is free.
        let mut drain = self.drain.enter();
        while let Some(action) = drain.poll() {
            if panic::catch_unwind(AssertUnwindSafe(action)).is_err() {
                log::error!(target: &self.log_target, "a serialized action panicked");
            }
        }
    }

    /// Gives the claim back, and runs the actions that arrived while it was held. The caller holds
    /// the claim.
    ///
    /// The release is one swap of `wip` to zero. If it returns 1, no caller tried the claim while it
    /// was held, so no `serialize` is waiting on it: a `serialize` whose `fetch_add` follows the
    /// swap finds zero and runs its own action. If it returns more, some caller's `fetch_add`
    /// preceded the swap, and the swap acquires from it. A `serialize` publishes `pending` before its
    /// `fetch_add`, so the holder now sees `pending` and takes the claim again to run the actions;
    /// a caller that only tried the claim (a worker that lost) leaves `pending` clear and the holder
    /// returns. When the holder cannot take the claim again, whoever holds it received that failed
    /// attempt in its own swap, sees more than 1, and runs the actions.
    fn release_claim(&self) {
        while self.wip.swap(0, Ordering::AcqRel) != 1 {
            if !self.pending.load(Ordering::SeqCst) || self.wip.fetch_add(1, Ordering::SeqCst) != 0
            {
                return;
            }
            self.run_actions();
        }
    }
}

impl UpstreamHandle for UpstreamInterceptor {
    fn sequence(&self) -> u64 {
        self.sequence
    }

    fn pull(
        &self,
        claim: &mut Claim,
        consumer: &mut dyn FnMut(Box<dyn Frame>),
        stop: &mut dyn FnMut(&dyn Frame) -> bool,
        demand: u64,
    ) -> Option<u64> {
        self.assert_own(claim);
        if !self.serviceable(demand) {
            return None;
        }
        // SAFETY: `claim` proves this caller holds the claim: it is the only one, so no action
        // runs and no other `pull` or `request` is in progress.
        let upstream = unsafe { self.upstream.get() }?;
        match panic::catch_unwind(AssertUnwindSafe(|| upstream.pull(consumer, stop, demand))) {
            Ok(pulled) => Some(pulled),
            Err(_) => {
                log::error!(target: &self.log_target, "an upstream panicked during a pull");
                self.complete();
                None
            }
        }
    }

    fn request(&self, claim: &mut Claim, demand: u64) {
        self.assert_own(claim);
        if !self.serviceable(demand) {
            return;
        }
        // SAFETY: as in `pull`.
        let Some(upstream) = (unsafe { self.upstream.get() }) else {
            return;
        };
        if panic::catch_unwind(AssertUnwindSafe(|| upstream.request(demand))).is_err() {
            log::error!(target: &self.log_target, "an upstream panicked during a request");
            self.complete();
        }
    }

    fn is_complete(&self) -> bool {
        self.complete.load(Ordering::Acquire)
    }

    fn try_acquire(&self) -> Option<Claim> {
        (self.wip.fetch_add(1, Ordering::SeqCst) == 0).then(|| Claim::new(self.claim_id))
    }

    fn release(&self, claim: Claim) {
        self.assert_own(&claim);
        self.release_claim();
    }
}

impl LatticeTerminal for UpstreamInterceptor {
    fn add_upstream(&self, upstream: Arc<dyn LatticeSource>) {
        self.attach(upstream);
    }
}

impl LatticeReceiver for UpstreamInterceptor {
    /// Routes `frame` through the vertex, or finishes it with an error if it cannot be routed.
    fn push(&self, frame: Box<dyn Frame>) {
        let routed = match self.vertex.upgrade() {
            Some(vertex) => vertex.try_push(frame),
            None => Err(frame),
        };
        match routed {
            Ok(()) => note_routed_push(),
            Err(frame) => frame.do_finally_with_error(unroutable()),
        }
    }

    fn on_complete(&self) {
        if self.finish() {
            log::trace!(target: &self.log_target, "an upstream completed");
            self.let_go_of_the_source(false);
        }
    }

    fn on_error(&self, error: Failure) {
        if self.finish() {
            log::error!(target: &self.log_target, "an upstream failed: {error}");
            self.let_go_of_the_source(false);
        }
    }

    /// Runs `action` at once if no worker holds the source, otherwise when the holder releases it.
    fn serialize(&self, action: Box<dyn FnOnce() + Send>) {
        if self.actions.offer(action).is_err() {
            unreachable!("the action queue is unbounded");
        }
        self.pending.store(true, Ordering::SeqCst);
        if self.wip.fetch_add(1, Ordering::SeqCst) == 0 {
            self.run_actions();
            self.release_claim();
        }
    }
}

#[cfg(test)]
mod tests;
