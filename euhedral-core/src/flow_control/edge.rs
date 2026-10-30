use super::registry::UpstreamRegistry;
use super::upstream_queue::UpstreamHandle;
use super::views::Cell;
use crate::frames::{Failure, Frame, FrameError};
use crate::generics::LatticeReceiver;
use crate::utils::spin_wait;
use std::ptr;
use std::sync::atomic::{AtomicBool, AtomicPtr, Ordering, fence};
use std::sync::{Arc, Weak};

/// Work to run once a retiring edge can no longer receive frames.
pub type Retirement = Box<dyn FnOnce() + Send>;

/// A stage of a lattice: frames flow down through it toward receivers, and demand flows up
/// through it toward the upstream handles.
///
/// Every node is built on a [`LatticeEdge`], which links it to its parent and its downstream.
pub trait LatticeNode: Send + Sync {
    /// The edge this node is built on.
    fn edge(&self) -> &LatticeEdge;

    /// Sends `frame` on toward the receivers. A frame that cannot be sent on is finished with an
    /// error.
    fn push(&self, frame: Box<dyn Frame>);

    /// Passes `error` on toward the receivers.
    fn on_error(&self, error: Failure);

    /// Closes the node for good.
    fn close(&self);

    fn is_closed(&self) -> bool;

    /// Called through `child`'s [`defer_retirement`](LatticeEdge::defer_retirement): runs
    /// `action` once `child` can no longer receive frames from this node. The default asks this
    /// node's own parent, as that parent's child.
    fn defer_child_retirement(&self, child: &LatticeEdge, action: Retirement) {
        let _ = child;
        self.edge().defer_retirement(action);
    }
}

/// Where an edge sends its frames.
enum Downstream {
    Receiver(Arc<dyn LatticeReceiver>),
    Node(Arc<dyn LatticeNode>),
}

/// The link every lattice node is built on, and the plain node that only links.
///
/// An edge has at most one downstream: a receiver, which is the floor of the edge's branch, or
/// another node, which extends the branch. Frames pushed into the edge go to the downstream. The
/// downstream is set once, by a compare-and-set (Java's `DOWNSTREAM`), and stays until the edge drops: an edge
/// closes for good, so nothing attaches again, and a push that read it just before the close may still be using it.
///
/// An edge's parent, as its cell holds it.
pub(crate) struct ParentRef(Weak<dyn LatticeNode>);

/// An edge may have a parent, the node above it. Demand from a request or pull climbs from parent
/// to parent, skipped by any closed or draining edge on the way, and the topmost edge serves it
/// from the calling thread's [`UpstreamQueue`](super::UpstreamQueue). Upstream handles are added
/// to the [`UpstreamRegistry`] all linked edges share, through any of them. The parent is a weak reference in a cell
/// of the registry (Java's `PARENT`): a worker reads it from its own view, which a change reaches through a queue.
pub struct LatticeEdge {
    registry: Arc<UpstreamRegistry>,
    drain: Arc<AtomicBool>,
    /// The node built on this edge, handed to children as their parent.
    node: Weak<dyn LatticeNode>,
    parent: Cell<ParentRef>,
    /// The downstream, a leaked `Box` installed once by a compare-and-set and freed when the edge drops (Java's
    /// `DOWNSTREAM`); null until then.
    downstream: AtomicPtr<Downstream>,
    closed: AtomicBool,
    /// Whether the receiver below was told the edge completed: the closing thread and an attach that finds the
    /// edge closed both try, and the first to take the flag tells.
    told: AtomicBool,
}

impl LatticeEdge {
    /// A plain edge in `registry`, ignoring demand while `drain` is set.
    pub fn new(registry: &Arc<UpstreamRegistry>, drain: Arc<AtomicBool>) -> Arc<Self> {
        Arc::new_cyclic(|this: &Weak<Self>| Self::for_node(registry, drain, this.clone()))
    }

    /// The edge of the node `node` will point to once constructed.
    pub(crate) fn for_node(
        registry: &Arc<UpstreamRegistry>,
        drain: Arc<AtomicBool>,
        node: Weak<dyn LatticeNode>,
    ) -> Self {
        Self {
            registry: registry.clone(),
            drain,
            node,
            parent: registry.cell(),
            downstream: AtomicPtr::new(ptr::null_mut()),
            closed: AtomicBool::new(false),
            told: AtomicBool::new(false),
        }
    }

    /// Installs `downstream` unless one is installed already, in which case it is handed back. Sequentially
    /// consistent, as Java's `compareAndSet` is: [`close_edge`](Self::close_edge) pairs with it.
    fn install_downstream(&self, downstream: Downstream) -> Result<(), Downstream> {
        let raw = Box::into_raw(Box::new(downstream));
        match self.downstream.compare_exchange(
            ptr::null_mut(),
            raw,
            Ordering::SeqCst,
            Ordering::SeqCst,
        ) {
            Ok(_) => Ok(()),
            // SAFETY: `raw` came from `Box::into_raw` just above and was not published.
            Err(_) => Err(*unsafe { Box::from_raw(raw) }),
        }
    }

    /// The downstream, if one is installed (Java's `getAcquire`). It lives as long as the edge: nothing replaces or
    /// frees it before the edge drops.
    fn downstream(&self) -> Option<&Downstream> {
        // SAFETY: a non-null pointer came from `Box::into_raw` in `install_downstream`, is never replaced, and is
        // freed only in `drop`, which the borrow of `self` rules out.
        unsafe { self.downstream.load(Ordering::Acquire).as_ref() }
    }

    pub fn registry(&self) -> &Arc<UpstreamRegistry> {
        &self.registry
    }

    /// The flag that, while set, makes the edge ignore demand and hold back new upstream handles.
    pub fn drain_flag(&self) -> &Arc<AtomicBool> {
        &self.drain
    }

    /// Java's `drain.getOpaque()`: a hint that decides whether to start a request or a pull; the flag's writer
    /// publishes nothing through it.
    fn is_draining(&self) -> bool {
        self.drain.load(Ordering::Relaxed)
    }

    /// Whether [`close_edge`](Self::close_edge) has run.
    pub fn is_edge_closed(&self) -> bool {
        self.closed.load(Ordering::Acquire)
    }

    /// Java's `CLOSED.getOpaque` on the request and pull path: a hint that decides whether to start.
    fn closed_hint(&self) -> bool {
        self.closed.load(Ordering::Relaxed)
    }

    /// The node above this edge, if it is still alive. Lifecycle code reads the cell under its lock: it is not on the
    /// frame path, and it should not become a reader (a reader's mailbox collects every change of the lattice).
    pub fn parent(&self) -> Option<Arc<dyn LatticeNode>> {
        self.registry.cell_value(&self.parent)?.0.upgrade()
    }

    /// Makes `parent` the node above this edge, or detaches the edge with `None`. The edge holds its
    /// parent weakly.
    ///
    /// # Panics
    ///
    /// Panics if `parent` belongs to another registry.
    pub fn set_parent(&self, parent: Option<&Arc<dyn LatticeNode>>) {
        if let Some(parent) = parent {
            assert!(
                Arc::ptr_eq(&self.registry, &parent.edge().registry),
                "linked edges must share one upstream registry"
            );
        }
        let current = self.registry.cell_value(&self.parent);
        let unchanged = match (&current, parent) {
            (None, None) => true,
            (Some(current), Some(parent)) => {
                current.0.as_ptr().cast::<()>() == Arc::as_ptr(parent).cast::<()>()
            }
            _ => false,
        };
        if !unchanged {
            self.registry.set_cell(
                &self.parent,
                parent.map(|parent| Arc::new(ParentRef(Arc::downgrade(parent)))),
            );
        }
    }

    /// The topmost edge a request or pull from this one reaches: `Some(None)` when this edge is the top,
    /// `Some(Some(node))` for the topmost node above it, and `None` when a closed or draining edge on the way
    /// stops it. One read of the worker's view covers the whole climb.
    fn climb(&self) -> Option<Option<Arc<dyn LatticeNode>>> {
        self.registry.with_view(|view| {
            let mut top: Option<Arc<dyn LatticeNode>> = None;
            let mut next = view.get(&self.parent).and_then(|p| p.0.upgrade());
            while let Some(node) = next {
                let edge = node.edge();
                if edge.closed_hint() || edge.is_draining() {
                    return None;
                }
                next = view.get(&edge.parent).and_then(|p| p.0.upgrade());
                top = Some(node);
            }
            Some(top)
        })
    }

    /// Signals demand for `demand` frames to the upstream handles, through the parents.
    pub fn request(&self, demand: u64) {
        if demand == 0 || self.closed_hint() || self.is_draining() {
            return;
        }
        if self.climb().is_some() {
            self.registry
                .with_thread_queue(|queue| queue.request(demand));
        }
    }

    /// Takes up to `demand` frames from the upstream handles straight into `consumer`, through the
    /// parents, and returns how many. See [`UpstreamQueue::pull`](super::UpstreamQueue::pull).
    pub fn pull(
        &self,
        consumer: &mut dyn FnMut(Box<dyn Frame>),
        stop: &mut dyn FnMut(&dyn Frame) -> bool,
        demand: u64,
    ) -> u64 {
        if demand == 0 || self.closed_hint() || self.is_draining() || self.climb().is_none() {
            return 0;
        }
        self.registry
            .with_thread_queue(|queue| queue.pull(consumer, stop, demand))
            .unwrap_or(0)
    }

    /// Adds `handle` to the registry, after waiting for any drain to end.
    pub fn add_upstream_handle(&self, handle: Arc<dyn UpstreamHandle>) {
        spin_wait::await_while(|| self.is_draining());
        self.registry.add_handle(handle);
    }

    /// Makes `receiver` the floor of this edge's branch: the downstream of this edge, or of the
    /// lowest node below it. If the branch already has a receiver, `receiver` is told of the error
    /// instead. A receiver added to an edge that has closed is told it completed, once, whether the edge took it or
    /// refused it.
    pub fn add_downstream(&self, receiver: Arc<dyn LatticeReceiver>) {
        let receiver = match self.install_downstream(Downstream::Receiver(receiver)) {
            Ok(()) => {
                // The close may have run before the receiver was installed and found nobody to tell.
                if self.closed.load(Ordering::SeqCst) {
                    self.tell_complete();
                }
                return;
            }
            Err(Downstream::Receiver(receiver)) => receiver,
            Err(Downstream::Node(_)) => unreachable!("the value handed back is the receiver set"),
        };
        match self.downstream() {
            Some(Downstream::Node(node)) => node.edge().add_downstream(receiver),
            // A refused receiver of a closed edge hears the edge completed, once: it was never installed, so
            // nothing else tells it.
            Some(Downstream::Receiver(_)) | None if self.closed.load(Ordering::SeqCst) => {
                receiver.on_complete();
            }
            Some(Downstream::Receiver(_)) | None => receiver.on_error(already_attached()),
        }
    }

    /// Extends this edge's branch with `node`: it becomes the downstream of this edge, or of the
    /// lowest node below it, and that edge becomes its parent. If the branch already ends in a
    /// receiver, `node` is told of the error instead.
    ///
    /// # Panics
    ///
    /// Panics if `node` belongs to another registry.
    pub fn add_downstream_node(&self, node: Arc<dyn LatticeNode>) {
        let node = match self.install_downstream(Downstream::Node(node)) {
            Ok(()) => {
                if let Some(Downstream::Node(node)) = self.downstream() {
                    let parent = self.node.upgrade().expect("a live node owns its edge");
                    node.edge().set_parent(Some(&parent));
                }
                return;
            }
            Err(Downstream::Node(node)) => node,
            Err(Downstream::Receiver(_)) => unreachable!("the value handed back is the node set"),
        };
        match self.downstream() {
            Some(Downstream::Node(below)) => below.edge().add_downstream_node(node),
            Some(Downstream::Receiver(_)) | None => node.on_error(already_attached()),
        }
    }

    /// Sends `frame` to the downstream, or finishes it with an error if there is none or the edge has closed.
    pub fn push_downstream(&self, frame: Box<dyn Frame>) {
        match self.live_downstream() {
            Some(Downstream::Receiver(receiver)) => receiver.push(frame),
            Some(Downstream::Node(node)) => node.push(frame),
            None => frame.do_finally_with_error(FrameError::failed("the edge has no downstream")),
        }
    }

    /// Passes `error` to the downstream, if any.
    pub fn error_downstream(&self, error: Failure) {
        match self.live_downstream() {
            Some(Downstream::Receiver(receiver)) => receiver.on_error(error),
            Some(Downstream::Node(node)) => node.on_error(error),
            None => {}
        }
    }

    /// The downstream, unless the edge has closed: a relaxed read of the flag (Java's opaque one) and an acquire
    /// load of the slot. No reference count: the slot's value lives as long as the edge.
    fn live_downstream(&self) -> Option<&Downstream> {
        if self.closed.load(Ordering::Relaxed) {
            return None;
        }
        self.downstream()
    }

    /// Tells the receiver below, if any, that the edge completed, unless something did already.
    fn tell_complete(&self) {
        if let Some(Downstream::Receiver(receiver)) = self.downstream()
            && !self.told.swap(true, Ordering::SeqCst)
        {
            receiver.on_complete();
        }
    }

    /// Closes the edge for good: pushes stop reaching its downstream, and a receiver is told it is complete.
    /// Returns whether this call closed it.
    pub fn close_edge(&self) -> bool {
        if self.closed.swap(true, Ordering::SeqCst) {
            return false;
        }
        // Pairs with the attach that sets the downstream and then reads the flag: either that attach sees the edge
        // closed, or this read of the downstream sees the attach.
        fence(Ordering::SeqCst);
        self.tell_complete();
        true
    }

    /// Runs `action` once this edge can no longer receive frames from its parent: right away
    /// without a parent, otherwise when the parent says so.
    pub fn defer_retirement(&self, action: Retirement) {
        match self.parent() {
            Some(parent) => parent.defer_child_retirement(self, action),
            None => action(),
        }
    }
}

impl Drop for LatticeEdge {
    fn drop(&mut self) {
        let raw = *self.downstream.get_mut();
        if !raw.is_null() {
            // SAFETY: as in `downstream`; `&mut self` rules out any reader.
            drop(unsafe { Box::from_raw(raw) });
        }
    }
}

impl LatticeNode for LatticeEdge {
    fn edge(&self) -> &LatticeEdge {
        self
    }

    fn push(&self, frame: Box<dyn Frame>) {
        self.push_downstream(frame);
    }

    fn on_error(&self, error: Failure) {
        self.error_downstream(error);
    }

    fn close(&self) {
        self.close_edge();
    }

    fn is_closed(&self) -> bool {
        self.is_edge_closed()
    }
}

fn already_attached() -> Failure {
    "the branch already ends in a receiver".into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flow_control::Claim;
    use crate::flow_control::testing::{Fate, Fates, TestReceiver, Tracked, frame};
    use euhedral_hardware_utils::util::CpuSet;
    use std::sync::Mutex;
    use std::sync::atomic::AtomicU64;
    use std::thread;
    use std::time::Duration;

    #[derive(Default)]
    struct Handle {
        requested: AtomicU64,
        pulled: AtomicU64,
    }

    impl UpstreamHandle for Handle {
        fn sequence(&self) -> u64 {
            0
        }

        fn pull(
            &self,
            _claim: &mut Claim,
            _consumer: &mut dyn FnMut(Box<dyn Frame>),
            _stop: &mut dyn FnMut(&dyn Frame) -> bool,
            demand: u64,
        ) -> Option<u64> {
            self.pulled.fetch_add(demand, Ordering::Relaxed);
            Some(demand)
        }

        fn request(&self, _claim: &mut Claim, demand: u64) {
            self.requested.fetch_add(demand, Ordering::Relaxed);
        }

        fn is_complete(&self) -> bool {
            false
        }
    }

    fn registry() -> Arc<UpstreamRegistry> {
        UpstreamRegistry::with_topology(4, CpuSet::new())
    }

    fn edge(registry: &Arc<UpstreamRegistry>) -> Arc<LatticeEdge> {
        LatticeEdge::new(registry, Arc::new(AtomicBool::new(false)))
    }

    /// An edge whose registry has a registered handle on this thread.
    fn served() -> (Arc<LatticeEdge>, Arc<Handle>) {
        let registry = registry();
        registry.register();
        let handle = Arc::new(Handle::default());
        let edge = edge(&registry);
        edge.add_upstream_handle(handle.clone());
        (edge, handle)
    }

    fn pull(edge: &LatticeEdge, demand: u64) -> u64 {
        edge.pull(&mut |_| {}, &mut |_| false, demand)
    }

    fn node(edge: &Arc<LatticeEdge>) -> Arc<dyn LatticeNode> {
        edge.clone()
    }

    #[test]
    fn pushes_and_errors_reach_the_receiver() {
        let edge = edge(&registry());
        let receiver = TestReceiver::new();
        edge.add_downstream(receiver.clone());

        edge.push(frame(1));
        edge.on_error("boom".into());

        assert_eq!(receiver.received(), [1]);
        assert_eq!(receiver.errors(), ["boom"]);
    }

    #[test]
    fn a_frame_without_a_downstream_fails() {
        let edge = edge(&registry());
        let fates = Fates::default();
        edge.push(Tracked::new(2, &fates));
        edge.on_error("ignored".into());
        assert_eq!(
            *fates.lock().unwrap(),
            [(
                2,
                Fate::Failed("frame failed: the edge has no downstream".into())
            )]
        );
    }

    #[test]
    fn closing_completes_and_detaches_the_receiver_once() {
        let edge = edge(&registry());
        let receiver = TestReceiver::new();
        edge.add_downstream(receiver.clone());

        assert!(edge.close_edge());
        assert!(receiver.is_completed());
        assert!(edge.is_closed());
        receiver.completed.store(false, Ordering::Release);
        edge.close();
        assert!(!receiver.is_completed());

        let fates = Fates::default();
        edge.push(Tracked::new(3, &fates));
        assert!(receiver.received().is_empty());
        assert_eq!(fates.lock().unwrap().len(), 1);
    }

    #[test]
    fn demand_reaches_the_upstream_handles() {
        let (edge, handle) = served();
        edge.request(5);
        assert_eq!(pull(&edge, 7), 7);
        assert_eq!(handle.requested.load(Ordering::Relaxed), 5);
        assert_eq!(handle.pulled.load(Ordering::Relaxed), 7);
    }

    #[test]
    fn zero_demand_is_ignored() {
        let (edge, handle) = served();
        edge.request(0);
        assert_eq!(pull(&edge, 0), 0);
        assert_eq!(handle.requested.load(Ordering::Relaxed), 0);
        assert_eq!(handle.pulled.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn a_draining_or_closed_edge_ignores_demand() {
        let (edge, handle) = served();
        edge.drain_flag().store(true, Ordering::Release);
        edge.request(5);
        assert_eq!(pull(&edge, 5), 0);

        edge.drain_flag().store(false, Ordering::Release);
        edge.close_edge();
        edge.request(5);
        assert_eq!(pull(&edge, 5), 0);

        assert_eq!(handle.requested.load(Ordering::Relaxed), 0);
        assert_eq!(handle.pulled.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn demand_climbs_through_the_parents() {
        let (parent, handle) = served();
        let child = edge(parent.registry());
        child.set_parent(Some(&node(&parent)));

        child.request(3);
        assert_eq!(pull(&child, 4), 4);
        assert_eq!(handle.requested.load(Ordering::Relaxed), 3);

        parent.drain_flag().store(true, Ordering::Release);
        child.request(3);
        assert_eq!(pull(&child, 4), 0, "a draining parent stops the demand");
        assert_eq!(handle.requested.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn parents_can_be_set_cleared_and_outlived() {
        let registry = registry();
        let child = edge(&registry);
        let parent = edge(&registry);
        child.set_parent(Some(&node(&parent)));
        assert!(Arc::ptr_eq(&child.parent().unwrap(), &node(&parent)));

        child.set_parent(None);
        assert!(child.parent().is_none());

        child.set_parent(Some(&node(&parent)));
        drop(parent);
        assert!(child.parent().is_none(), "the parent is held weakly");
    }

    #[test]
    #[should_panic(expected = "share one upstream registry")]
    fn edges_of_different_registries_cannot_link() {
        let child = edge(&registry());
        child.set_parent(Some(&node(&edge(&registry()))));
    }

    #[test]
    fn a_second_receiver_is_rejected() {
        let edge = edge(&registry());
        let first = TestReceiver::new();
        let second = TestReceiver::new();
        edge.add_downstream(first.clone());
        edge.add_downstream(second.clone());

        assert_eq!(second.errors(), ["the branch already ends in a receiver"]);
        edge.push(frame(1));
        assert_eq!(first.received(), [1]);
        assert!(second.received().is_empty());
    }

    #[test]
    fn nodes_extend_the_branch_and_receivers_land_at_its_floor() {
        let registry = registry();
        let top = edge(&registry);
        let middle = edge(&registry);
        let bottom = edge(&registry);
        top.add_downstream_node(middle.clone());
        top.add_downstream_node(bottom.clone());
        let receiver = TestReceiver::new();
        top.add_downstream(receiver.clone());

        assert!(Arc::ptr_eq(&middle.parent().unwrap(), &node(&top)));
        assert!(Arc::ptr_eq(&bottom.parent().unwrap(), &node(&middle)));
        assert!(receiver.errors().is_empty());
        top.push(frame(9));
        assert_eq!(receiver.received(), [9]);
    }

    #[test]
    fn a_node_below_a_receiver_is_told_of_the_error() {
        let registry = registry();
        let edge = edge(&registry);
        edge.add_downstream(TestReceiver::new());
        let late = self::edge(&registry);
        let receiver = TestReceiver::new();
        late.add_downstream(receiver.clone());

        edge.add_downstream_node(late.clone());

        assert_eq!(receiver.errors(), ["the branch already ends in a receiver"]);
        assert!(late.parent().is_none());
    }

    #[test]
    fn handles_count_until_removed_and_reach_linked_edges() {
        let registry = registry();
        registry.register();
        let parent = edge(&registry);
        let child = edge(&registry);
        let handle = Arc::new(Handle::default());
        child.add_upstream_handle(handle.clone());
        assert_eq!(registry.upstream_handle_count(), 1);

        child.set_parent(Some(&node(&parent)));
        parent.request(2);
        assert_eq!(handle.requested.load(Ordering::Relaxed), 2);

        registry.remove_handle();
        assert_eq!(registry.upstream_handle_count(), 0);
    }

    #[test]
    fn adding_a_handle_waits_for_the_drain_to_end() {
        let registry = registry();
        let edge = edge(&registry);
        edge.drain_flag().store(true, Ordering::Release);

        let adder = {
            let edge = edge.clone();
            thread::spawn(move || edge.add_upstream_handle(Arc::new(Handle::default())))
        };
        thread::sleep(Duration::from_millis(20));
        assert_eq!(registry.upstream_handle_count(), 0);

        edge.drain_flag().store(false, Ordering::Release);
        adder.join().unwrap();
        assert_eq!(registry.upstream_handle_count(), 1);
    }

    #[test]
    fn a_root_edge_retires_at_once() {
        let edge = edge(&registry());
        let ran = Arc::new(AtomicBool::new(false));
        let flag = ran.clone();
        edge.defer_retirement(Box::new(move || flag.store(true, Ordering::Release)));
        assert!(ran.load(Ordering::Acquire));
    }

    /// A node that records which children asked it to defer their retirement.
    struct Deferring {
        edge: LatticeEdge,
        children: Mutex<Vec<usize>>,
        deferred: Mutex<Vec<Retirement>>,
    }

    impl LatticeNode for Deferring {
        fn edge(&self) -> &LatticeEdge {
            &self.edge
        }
        fn push(&self, frame: Box<dyn Frame>) {
            self.edge.push_downstream(frame);
        }
        fn on_error(&self, error: Failure) {
            self.edge.error_downstream(error);
        }
        fn close(&self) {
            self.edge.close_edge();
        }
        fn is_closed(&self) -> bool {
            self.edge.is_edge_closed()
        }
        fn defer_child_retirement(&self, child: &LatticeEdge, action: Retirement) {
            self.children
                .lock()
                .unwrap()
                .push(child as *const LatticeEdge as usize);
            self.deferred.lock().unwrap().push(action);
        }
    }

    #[test]
    fn retirement_is_deferred_by_the_nearest_node_that_tracks_children() {
        let registry = registry();
        let tracker: Arc<Deferring> = Arc::new_cyclic(|this: &Weak<Deferring>| Deferring {
            edge: LatticeEdge::for_node(&registry, Arc::new(AtomicBool::new(false)), this.clone()),
            children: Mutex::new(Vec::new()),
            deferred: Mutex::new(Vec::new()),
        });
        let middle = edge(&registry);
        let leaf = edge(&registry);
        tracker.edge().add_downstream_node(middle.clone());
        middle.add_downstream_node(leaf.clone());

        let ran = Arc::new(AtomicBool::new(false));
        let flag = ran.clone();
        leaf.defer_retirement(Box::new(move || flag.store(true, Ordering::Release)));

        assert!(!ran.load(Ordering::Acquire));
        assert_eq!(
            *tracker.children.lock().unwrap(),
            [&*middle as *const LatticeEdge as usize],
            "a plain edge defers as its parent's child"
        );
        for action in tracker.deferred.lock().unwrap().drain(..) {
            action();
        }
        assert!(ran.load(Ordering::Acquire));
    }

    /// Counts the frames it receives and the times it is told the edge completed.
    #[derive(Default)]
    struct Told {
        pushed: std::sync::atomic::AtomicUsize,
        completions: std::sync::atomic::AtomicUsize,
    }

    impl crate::generics::LatticeTerminal for Told {
        fn add_upstream(&self, _upstream: Arc<dyn crate::generics::LatticeSource>) {}
    }

    impl LatticeReceiver for Told {
        fn push(&self, _frame: Box<dyn Frame>) {
            self.pushed.fetch_add(1, Ordering::SeqCst);
        }
        fn on_complete(&self) {
            self.completions.fetch_add(1, Ordering::SeqCst);
        }
        fn on_error(&self, _error: Failure) {}
    }

    #[cfg(not(miri))]
    const ROUNDS: usize = 2_000;
    #[cfg(miri)]
    const ROUNDS: usize = 20;

    #[test]
    fn close_while_pushing_finishes_or_delivers_every_frame() {
        for _ in 0..ROUNDS / 10 {
            let edge = edge(&registry());
            let told = Arc::new(Told::default());
            edge.add_downstream(told.clone());
            let fates = Fates::default();
            let pushers: Vec<_> = (0..3)
                .map(|pusher| {
                    let (edge, fates) = (edge.clone(), fates.clone());
                    thread::spawn(move || {
                        for id in 0..50_u64 {
                            edge.push(Tracked::new(pusher * 1_000 + id, &fates));
                        }
                    })
                })
                .collect();
            let closer = {
                let edge = edge.clone();
                thread::spawn(move || edge.close_edge())
            };
            for pusher in pushers {
                pusher.join().unwrap();
            }
            assert!(closer.join().unwrap());
            let failed = fates.lock().unwrap().len();
            assert_eq!(
                told.pushed.load(Ordering::SeqCst) + failed,
                150,
                "every frame is delivered or finished"
            );
            assert_eq!(told.completions.load(Ordering::SeqCst), 1);
        }
    }

    #[test]
    fn a_closed_edge_refuses_a_new_downstream() {
        let edge = edge(&registry());
        assert!(edge.close_edge());

        let told = Arc::new(Told::default());
        edge.add_downstream(told.clone());
        assert_eq!(
            told.completions.load(Ordering::SeqCst),
            1,
            "a receiver of a closed edge is told it is complete"
        );

        let fates = Fates::default();
        edge.push(Tracked::new(1, &fates));
        assert_eq!(told.pushed.load(Ordering::SeqCst), 0);
        assert_eq!(fates.lock().unwrap().len(), 1);
        edge.close_edge();
        assert_eq!(told.completions.load(Ordering::SeqCst), 1, "told once");
    }

    #[test]
    fn a_receiver_refused_by_a_closed_edge_that_has_one_is_completed_once() {
        let registry = registry();
        let edge = edge(&registry);
        let first = Arc::new(Told::default());
        edge.add_downstream(first.clone());
        assert!(edge.close_edge());
        assert_eq!(first.completions.load(Ordering::SeqCst), 1);

        let late = Arc::new(Told::default());
        edge.add_downstream(late.clone());
        assert_eq!(
            late.completions.load(Ordering::SeqCst),
            1,
            "a late receiver of a closed edge hears it completed, though the edge refused it"
        );
        let refused = TestReceiver::new();
        edge.add_downstream(refused.clone());
        assert!(refused.is_completed());
        assert!(refused.errors().is_empty());
        assert_eq!(first.completions.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn a_second_downstream_is_refused_or_attached_below() {
        let registry = registry();
        let top = edge(&registry);
        let below = edge(&registry);
        top.add_downstream_node(below.clone());
        let told = Arc::new(Told::default());
        top.add_downstream(told.clone());
        top.push(frame(1));
        assert_eq!(
            told.pushed.load(Ordering::SeqCst),
            1,
            "attached at the bottom"
        );

        let second = TestReceiver::new();
        top.add_downstream(second.clone());
        assert_eq!(second.errors().len(), 1, "a branch ends in one receiver");
    }

    #[test]
    fn attaches_racing_a_close_tell_the_receiver_exactly_once() {
        for _ in 0..ROUNDS {
            let edge = edge(&registry());
            let told = Arc::new(Told::default());
            let barrier = Arc::new(std::sync::Barrier::new(2));
            let attacher = {
                let (edge, told, barrier) = (edge.clone(), told.clone(), barrier.clone());
                thread::spawn(move || {
                    barrier.wait();
                    edge.add_downstream(told);
                })
            };
            barrier.wait();
            edge.close_edge();
            attacher.join().unwrap();
            assert_eq!(
                told.completions.load(Ordering::SeqCst),
                1,
                "a receiver of a closed edge is told once, whichever came first"
            );
        }
    }

    #[test]
    fn racing_attaches_install_one_downstream_and_refuse_the_rest() {
        for _ in 0..ROUNDS / 10 {
            let edge = edge(&registry());
            let receivers: Vec<_> = (0..4).map(|_| TestReceiver::new()).collect();
            let attachers: Vec<_> = receivers
                .iter()
                .map(|receiver| {
                    let (edge, receiver) = (edge.clone(), receiver.clone());
                    thread::spawn(move || edge.add_downstream(receiver))
                })
                .collect();
            for attacher in attachers {
                attacher.join().unwrap();
            }
            edge.push(frame(1));
            let delivered = receivers
                .iter()
                .filter(|r| !r.received().is_empty())
                .count();
            let refused = receivers.iter().filter(|r| !r.errors().is_empty()).count();
            assert_eq!((delivered, refused), (1, 3));
        }
    }

    #[test]
    fn set_parent_replaces_the_parent_a_worker_reads_at_its_next_read() {
        let registry = registry();
        let child = edge(&registry);
        let first: Arc<dyn LatticeNode> = edge(&registry);
        let second: Arc<dyn LatticeNode> = edge(&registry);
        child.set_parent(Some(&first));
        // The worker's view holds the first parent until it drains.
        registry.register_reader();
        assert!(Arc::ptr_eq(&child.parent().unwrap(), &first));
        for _ in 0..10 {
            child.set_parent(Some(&first));
        }
        child.set_parent(Some(&second));
        assert!(Arc::ptr_eq(&child.parent().unwrap(), &second));
        child.set_parent(None);
        assert!(child.parent().is_none());
        registry.remove_reader();
    }

    #[test]
    fn pushes_and_set_parent_race_without_a_lock() {
        let registry = registry();
        let child = edge(&registry);
        let told = Arc::new(Told::default());
        child.add_downstream(told.clone());
        let parents: Vec<Arc<dyn LatticeNode>> = (0..4).map(|_| edge(&registry) as _).collect();
        let stop = Arc::new(AtomicBool::new(false));
        let readers: Vec<_> = (0..3)
            .map(|_| {
                let (child, stop) = (child.clone(), stop.clone());
                thread::spawn(move || {
                    let mut seen = 0_u64;
                    while !stop.load(Ordering::Acquire) {
                        if let Some(parent) = child.parent() {
                            assert!(!parent.is_closed());
                            seen += 1;
                        }
                        child.push(frame(1));
                    }
                    seen
                })
            })
            .collect();
        for round in 0..ROUNDS {
            child.set_parent(Some(&parents[round % 4]));
        }
        stop.store(true, Ordering::Release);
        for reader in readers {
            reader.join().unwrap();
        }
        assert!(child.parent().is_some());
    }
}
