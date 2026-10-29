use super::registry::UpstreamRegistry;
use super::upstream_queue::UpstreamHandle;
use crate::frames::{Failure, Frame, FrameError};
use crate::generics::LatticeReceiver;
use crate::utils::spin_wait;
use std::sync::atomic::{AtomicBool, Ordering};
#[allow(clippy::disallowed_types, reason = "HOT-FIX(C9), HOT-FIX(C8)")]
use std::sync::{Arc, RwLock, Weak};

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
#[derive(Clone)]
enum Downstream {
    Receiver(Arc<dyn LatticeReceiver>),
    Node(Arc<dyn LatticeNode>),
}

/// The link every lattice node is built on, and the plain node that only links.
///
/// An edge has at most one downstream: a receiver, which is the floor of the edge's branch, or
/// another node, which extends the branch. Frames pushed into the edge go to the downstream.
///
/// An edge may have a parent, the node above it. Demand from a request or pull climbs from parent
/// to parent, skipped by any closed or draining edge on the way, and the topmost edge serves it
/// from the calling thread's [`UpstreamQueue`](super::UpstreamQueue). Upstream handles are added
/// to the [`UpstreamRegistry`] all linked edges share, through any of them.
pub struct LatticeEdge {
    registry: Arc<UpstreamRegistry>,
    drain: Arc<AtomicBool>,
    /// The node built on this edge, handed to children as their parent.
    node: Weak<dyn LatticeNode>,
    #[allow(clippy::disallowed_types, reason = "HOT-FIX(C9)")]
    parent: RwLock<Option<Weak<dyn LatticeNode>>>,
    #[allow(clippy::disallowed_types, reason = "HOT-FIX(C8)")]
    downstream: RwLock<Option<Downstream>>,
    closed: AtomicBool,
}

impl LatticeEdge {
    /// A plain edge in `registry`, ignoring demand while `drain` is set.
    pub fn new(registry: &Arc<UpstreamRegistry>, drain: Arc<AtomicBool>) -> Arc<Self> {
        Arc::new_cyclic(|this: &Weak<Self>| Self::for_node(registry, drain, this.clone()))
    }

    /// The edge of the node `node` will point to once constructed.
    #[allow(clippy::disallowed_types, reason = "HOT-FIX(C9), HOT-FIX(C8)")]
    pub(crate) fn for_node(
        registry: &Arc<UpstreamRegistry>,
        drain: Arc<AtomicBool>,
        node: Weak<dyn LatticeNode>,
    ) -> Self {
        Self {
            registry: registry.clone(),
            drain,
            node,
            parent: RwLock::new(None),
            downstream: RwLock::new(None),
            closed: AtomicBool::new(false),
        }
    }

    pub fn registry(&self) -> &Arc<UpstreamRegistry> {
        &self.registry
    }

    /// The flag that, while set, makes the edge ignore demand and hold back new upstream handles.
    pub fn drain_flag(&self) -> &Arc<AtomicBool> {
        &self.drain
    }

    fn is_draining(&self) -> bool {
        self.drain.load(Ordering::Acquire)
    }

    /// Whether [`close_edge`](Self::close_edge) has run.
    pub fn is_edge_closed(&self) -> bool {
        self.closed.load(Ordering::Acquire)
    }

    /// The node above this edge, if it is still alive.
    pub fn parent(&self) -> Option<Arc<dyn LatticeNode>> {
        self.parent.read().unwrap().as_ref()?.upgrade()
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
        *self.parent.write().unwrap() = parent.map(Arc::downgrade);
    }

    /// Signals demand for `demand` frames to the upstream handles, through the parents.
    pub fn request(&self, demand: u64) {
        if demand == 0 || self.is_edge_closed() || self.is_draining() {
            return;
        }
        match self.parent() {
            Some(parent) => parent.edge().request(demand),
            None => {
                self.registry
                    .with_thread_queue(|queue| queue.request(demand));
            }
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
        if demand == 0 || self.is_edge_closed() || self.is_draining() {
            return 0;
        }
        match self.parent() {
            Some(parent) => parent.edge().pull(consumer, stop, demand),
            None => self
                .registry
                .with_thread_queue(|queue| queue.pull(consumer, stop, demand))
                .unwrap_or(0),
        }
    }

    /// Adds `handle` to the registry, after waiting for any drain to end.
    pub fn add_upstream_handle(&self, handle: Arc<dyn UpstreamHandle>) {
        spin_wait::await_while(|| self.is_draining());
        self.registry.add_handle(handle);
    }

    /// Makes `receiver` the floor of this edge's branch: the downstream of this edge, or of the
    /// lowest node below it. If the branch already has a receiver, `receiver` is told of the error
    /// instead.
    pub fn add_downstream(&self, receiver: Arc<dyn LatticeReceiver>) {
        let existing = {
            let mut downstream = self.downstream.write().unwrap();
            match &*downstream {
                None => {
                    *downstream = Some(Downstream::Receiver(receiver));
                    return;
                }
                Some(existing) => existing.clone(),
            }
        };
        match existing {
            Downstream::Node(node) => node.edge().add_downstream(receiver),
            Downstream::Receiver(_) => receiver.on_error(already_attached()),
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
        let existing = {
            let mut downstream = self.downstream.write().unwrap();
            match &*downstream {
                None => {
                    let parent = self.node.upgrade().expect("a live node owns its edge");
                    node.edge().set_parent(Some(&parent));
                    *downstream = Some(Downstream::Node(node));
                    return;
                }
                Some(existing) => existing.clone(),
            }
        };
        match existing {
            Downstream::Node(below) => below.edge().add_downstream_node(node),
            Downstream::Receiver(_) => node.on_error(already_attached()),
        }
    }

    /// Sends `frame` to the downstream, or finishes it with an error if there is none.
    pub fn push_downstream(&self, frame: Box<dyn Frame>) {
        match self.downstream() {
            Some(Downstream::Receiver(receiver)) => receiver.push(frame),
            Some(Downstream::Node(node)) => node.push(frame),
            None => frame.do_finally_with_error(FrameError::failed("the edge has no downstream")),
        }
    }

    /// Passes `error` to the downstream, if any.
    pub fn error_downstream(&self, error: Failure) {
        match self.downstream() {
            Some(Downstream::Receiver(receiver)) => receiver.on_error(error),
            Some(Downstream::Node(node)) => node.on_error(error),
            None => {}
        }
    }

    fn downstream(&self) -> Option<Downstream> {
        self.downstream.read().unwrap().clone()
    }

    /// Closes the edge for good and detaches its downstream, telling a receiver it is complete.
    /// Returns whether this call closed it.
    pub fn close_edge(&self) -> bool {
        if self.closed.swap(true, Ordering::AcqRel) {
            return false;
        }
        let downstream = self.downstream.write().unwrap().take();
        if let Some(Downstream::Receiver(receiver)) = downstream {
            receiver.on_complete();
        }
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
}
