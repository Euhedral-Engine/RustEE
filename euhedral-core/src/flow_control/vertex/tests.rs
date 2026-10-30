use super::*;
use crate::flow_control::testing::{Fate, Fates, TestReceiver, Tracked, frame};
use crate::flow_control::{LatticeHotSource, RoutingPolicy};
use crate::frames::BenchmarkFrame;
use crate::generics::{DownstreamSlot, LatticeTerminal, SourceCell};
use euhedral_hardware_utils::topology::CpuInfo;
use euhedral_hardware_utils::util::CpuSet;
use std::sync::atomic::AtomicUsize;
use std::sync::mpsc;
use std::sync::{Barrier, Mutex};
use std::thread;
use std::time::{Duration, Instant};

fn registry() -> Arc<UpstreamRegistry> {
    UpstreamRegistry::with_topology(4, CpuSet::new())
}

fn receiving_edge(registry: &Arc<UpstreamRegistry>) -> (Arc<LatticeEdge>, Arc<TestReceiver>) {
    let edge = LatticeEdge::new(registry, Arc::new(AtomicBool::new(false)));
    let receiver = TestReceiver::new();
    edge.add_downstream(receiver.clone());
    (edge, receiver)
}

fn node(edge: &Arc<LatticeEdge>) -> Option<Arc<dyn LatticeNode>> {
    Some(edge.clone())
}

/// Maps `active` while draining, as the caller of a remap must.
fn map(vertex: &LatticeVertex, active: &[usize], handles: &[Option<Arc<dyn LatticeNode>>]) -> bool {
    vertex.set_drain(true);
    let mapped = vertex.set_downstream_mapping(active, handles);
    vertex.set_drain(false);
    mapped
}

/// A frame with `id` whose routing hash is `hash`.
fn routed(id: u64, hash: u64) -> Box<dyn Frame> {
    let mut frame = BenchmarkFrame::new(id, None);
    frame.header_mut().set_routing_hash(hash);
    Box::new(frame)
}

fn same(a: &Arc<dyn LatticeNode>, b: &Arc<LatticeEdge>) -> bool {
    std::ptr::eq(a.edge(), &**b)
}

/// A source that only pushes when the test tells it to.
#[derive(Default)]
struct DeferredSource {
    downstream: Mutex<Option<Arc<dyn LatticeReceiver>>>,
    requested: AtomicU64,
    complete: AtomicBool,
    panics: AtomicBool,
}

impl DeferredSource {
    fn emit(&self, frame: Box<dyn Frame>) {
        let downstream = self.downstream.lock().unwrap().clone().unwrap();
        downstream.push(frame);
    }
}

impl LatticeSource for DeferredSource {
    fn add_downstream(self: Arc<Self>, downstream: Arc<dyn LatticeReceiver>) {
        *self.downstream.lock().unwrap() = Some(downstream);
    }

    fn pull(
        &self,
        _consumer: &mut dyn FnMut(Box<dyn Frame>),
        _stop: &mut dyn FnMut(&dyn Frame) -> bool,
        _demand: u64,
    ) -> u64 {
        assert!(!self.panics.load(Ordering::Relaxed), "the source broke");
        0
    }

    fn request(&self, demand: u64) {
        assert!(!self.panics.load(Ordering::Relaxed), "the source broke");
        self.requested.fetch_add(demand, Ordering::Relaxed);
    }

    fn complete(&self) {
        self.complete.store(true, Ordering::Release);
    }

    fn is_complete(&self) -> bool {
        self.complete.load(Ordering::Acquire)
    }
}

#[test]
fn a_new_vertex_routes_nowhere() {
    let vertex = LatticeVertex::new(&registry(), "fresh", 4);
    assert_eq!(vertex.downstream_count(), 4);
    assert_eq!(vertex.name(), "fresh");
    assert!(vertex.hold_routing().is_empty());
    assert!((0..4).all(|id| vertex.active_downstream_index(id).is_none()));
}

#[test]
fn the_drain_flag_is_shared_with_the_edge() {
    let vertex = LatticeVertex::new(&registry(), "drain", 1);
    vertex.set_drain(true);
    assert!(vertex.edge().drain_flag().load(Ordering::Acquire));
    vertex.set_drain(false);
    assert!(!vertex.edge().drain_flag().load(Ordering::Acquire));
}

#[test]
fn a_mapping_activates_downstreams_and_adopts_them() {
    let registry = registry();
    let vertex = LatticeVertex::new(&registry, "map", 4);
    let (first, _) = receiving_edge(&registry);
    let (second, _) = receiving_edge(&registry);

    assert!(map(
        &vertex,
        &[2, 0],
        &[node(&first), None, node(&second), None]
    ));

    assert!(same(&vertex.downstream(0).unwrap(), &first));
    assert!(same(&vertex.downstream(2).unwrap(), &second));
    assert!(vertex.downstream(1).is_none());
    assert_eq!(vertex.hold_routing().mappings(), [0, 2]);
    assert_eq!(vertex.active_downstream_index(0), Some(0));
    assert_eq!(vertex.active_downstream_index(2), Some(1));
    assert_eq!(vertex.active_downstream_index(9), None);
    let parent = first.parent().unwrap();
    assert!(std::ptr::eq(parent.edge(), vertex.edge()));
    assert!(std::ptr::eq(second.parent().unwrap().edge(), vertex.edge()));
}

#[test]
fn mapping_needs_a_drained_open_vertex_and_valid_handles() {
    let registry = registry();
    let vertex = LatticeVertex::new(&registry, "invalid", 2);
    let (edge, _) = receiving_edge(&registry);

    assert!(
        !vertex.set_downstream_mapping(&[0], &[node(&edge)]),
        "not draining"
    );
    vertex.set_drain(true);
    assert!(
        !vertex.set_downstream_mapping(&[2], &[node(&edge), None, node(&edge)]),
        "past the slots"
    );
    assert!(
        !vertex.set_downstream_mapping(&[1], &[node(&edge)]),
        "no handle for the id"
    );
    assert!(
        !vertex.set_downstream_mapping(&[1], &[node(&edge), None]),
        "an empty handle"
    );
    assert!(vertex.hold_routing().is_empty());
    assert!(edge.parent().is_none(), "a refused mapping changes nothing");

    vertex.close();
    assert!(
        !vertex.set_downstream_mapping(&[0], &[node(&edge)]),
        "closed"
    );
}

#[test]
#[should_panic(expected = "share one upstream registry")]
fn mapping_a_handle_of_another_registry_panics() {
    let vertex = LatticeVertex::new(&registry(), "foreign", 1);
    let (edge, _) = receiving_edge(&registry());
    map(&vertex, &[0], &[node(&edge)]);
}

#[test]
fn a_remap_closes_downstreams_it_drops() {
    let registry = registry();
    let vertex = LatticeVertex::new(&registry, "remap", 4);
    let (kept, kept_receiver) = receiving_edge(&registry);
    let (dropped, dropped_receiver) = receiving_edge(&registry);
    assert!(map(&vertex, &[0, 1], &[node(&kept), node(&dropped)]));

    assert!(map(&vertex, &[0], &[node(&kept)]));

    assert!(dropped.is_closed());
    assert!(dropped_receiver.is_completed());
    assert!(!kept.is_closed());
    assert!(!kept_receiver.is_completed());
    assert!(vertex.downstream(1).is_none());

    assert!(map(&vertex, &[], &[]));
    assert!(kept.is_closed());
}

#[test]
fn frames_route_by_hash() {
    let registry = registry();
    let vertex = LatticeVertex::new(&registry, "route", 4);
    let (low, low_receiver) = receiving_edge(&registry);
    let (high, high_receiver) = receiving_edge(&registry);
    assert!(map(&vertex, &[0, 1], &[node(&low), node(&high)]));

    vertex.push(routed(1, 0));
    vertex.push(routed(2, u64::MAX));

    assert_eq!(low_receiver.received(), [1]);
    assert_eq!(high_receiver.received(), [2]);
    assert_eq!(route_by_hash(&*frame(7), 1), 0);
}

#[test]
fn a_custom_routing_function_sees_the_generation() {
    let registry = registry();
    let vertex = LatticeVertex::with_routing(&registry, "custom", 3, |frame, state| {
        let wanted = frame.header().id_hash() as usize;
        state.active_index(wanted).unwrap_or(0)
    });
    let (first, first_receiver) = receiving_edge(&registry);
    let (second, second_receiver) = receiving_edge(&registry);
    assert!(map(&vertex, &[0, 2], &[node(&first), None, node(&second)]));

    vertex.push(frame(2));
    vertex.push(frame(1));

    assert_eq!(second_receiver.received(), [2]);
    assert_eq!(first_receiver.received(), [1]);
}

#[test]
#[should_panic(expected = "the routing function chose index 5")]
fn an_out_of_range_route_panics() {
    let registry = registry();
    let vertex = LatticeVertex::with_routing(&registry, "broken", 1, |_, _| 5);
    let (edge, _) = receiving_edge(&registry);
    map(&vertex, &[0], &[node(&edge)]);
    vertex.push(frame(1));
}

#[test]
fn a_frame_without_an_active_downstream_is_refused() {
    let vertex = LatticeVertex::new(&registry(), "empty", 1);
    let fates = Fates::default();

    let refused = vertex.try_push(Tracked::new(1, &fates)).unwrap_err();
    assert_eq!(refused.header().id_hash(), 1);
    vertex.push(refused);

    assert_eq!(
        *fates.lock().unwrap(),
        [(
            1,
            Fate::Failed("frame failed: no active downstream to route the frame to".into())
        )]
    );
}

#[test]
fn errors_reach_every_active_downstream_once() {
    let registry = registry();
    let vertex = LatticeVertex::new(&registry, "errors", 3);
    let (first, first_receiver) = receiving_edge(&registry);
    let (second, second_receiver) = receiving_edge(&registry);
    assert!(map(
        &vertex,
        &[0, 1, 2],
        &[node(&first), node(&second), node(&first)]
    ));

    vertex.on_error("boom".into());

    assert_eq!(first_receiver.errors(), ["boom"]);
    assert_eq!(second_receiver.errors(), ["boom"]);

    assert!(map(&vertex, &[1], &[None, node(&second)]));
    vertex.on_error("alone".into());
    assert_eq!(second_receiver.errors(), ["boom", "alone"]);
}

#[test]
fn closing_closes_every_downstream_and_unmaps_them() {
    let registry = registry();
    let vertex = LatticeVertex::new(&registry, "close", 2);
    let (first, first_receiver) = receiving_edge(&registry);
    let (second, _) = receiving_edge(&registry);
    assert!(map(&vertex, &[0, 1], &[node(&first), node(&second)]));

    vertex.close();

    assert!(vertex.is_closed());
    assert!(vertex.edge().is_edge_closed());
    assert!(first.is_closed());
    assert!(second.is_closed());
    assert!(first_receiver.is_completed());
    assert_eq!(vertex.active_downstream_index(0), None);
    assert_eq!(vertex.active_downstream_index(1), None);
    let fates = Fates::default();
    vertex.push(Tracked::new(1, &fates));
    assert!(matches!(fates.lock().unwrap()[0], (1, Fate::Failed(_))));
    vertex.on_error("after close".into());
    assert!(first_receiver.errors().is_empty());
}

#[test]
fn a_shared_downstream_is_closed_once_no_generation_uses_it() {
    let registry = registry();
    let vertex = LatticeVertex::new(&registry, "shared", 2);
    let (shared, receiver) = receiving_edge(&registry);
    assert!(map(&vertex, &[0, 1], &[node(&shared), node(&shared)]));
    assert!(map(&vertex, &[1], &[None, node(&shared)]));
    assert!(!shared.is_closed());

    assert!(map(&vertex, &[], &[]));

    assert!(shared.is_closed());
    assert!(receiver.is_completed());
}

#[test]
fn a_vertex_goes_below_an_upstream_node() {
    let registry = registry();
    let upstream = LatticeEdge::new(&registry, Arc::new(AtomicBool::new(false)));
    let vertex = LatticeVertex::new(&registry, "below", 1);
    let (downstream, receiver) = receiving_edge(&registry);
    assert!(map(&vertex, &[0], &[node(&downstream)]));

    vertex.add_upstream_node(&*upstream).unwrap();

    assert!(std::ptr::eq(
        vertex.edge().parent().unwrap().edge(),
        &*upstream
    ));
    assert!(
        std::ptr::eq(downstream.parent().unwrap().edge(), vertex.edge()),
        "the vertex keeps its own downstreams"
    );
    upstream.push(frame(4));
    assert_eq!(receiver.received(), [4]);

    vertex.close();
    assert_eq!(vertex.add_upstream_node(&*upstream), Err(LatticeClosed));
}

#[test]
fn demand_through_a_vertex_climbs_to_its_parent() {
    let registry = registry();
    registry.register();
    let upstream = LatticeEdge::new(&registry, Arc::new(AtomicBool::new(false)));
    let vertex = LatticeVertex::new(&registry, "climb", 1);
    vertex.add_upstream_node(&*upstream).unwrap();
    let source = Arc::new(DeferredSource::default());
    vertex.ingest(source.clone()).unwrap();

    vertex.edge().request(3);
    assert_eq!(source.requested.load(Ordering::Relaxed), 3);

    upstream.drain_flag().store(true, Ordering::Release);
    vertex.edge().request(3);
    assert_eq!(vertex.edge().pull(&mut |_| {}, &mut |_| false, 3), 0);
    assert_eq!(
        source.requested.load(Ordering::Relaxed),
        3,
        "a draining parent stops it"
    );
}

#[test]
fn ingesting_attaches_the_source_once() {
    let registry = registry();
    let vertex = LatticeVertex::new(&registry, "ingest", 1);
    let source = Arc::new(LatticeHotSource::new());

    let interceptor = vertex.ingest(source.clone()).unwrap();

    let upstream: Arc<dyn LatticeSource> = source.clone();
    assert!(interceptor.has_upstream(&upstream));
    assert_eq!(
        registry.upstream_handle_count(),
        1,
        "the source attached itself too"
    );
    assert!(!interceptor.is_complete());

    source.complete();
    assert!(interceptor.is_complete());
    assert_eq!(registry.upstream_handle_count(), 0);
}

#[test]
fn ingested_sources_take_registration_order() {
    let vertex = LatticeVertex::new(&registry(), "order", 1);
    let first = vertex.ingest(Arc::new(DeferredSource::default())).unwrap();
    let second = vertex.ingest(Arc::new(DeferredSource::default())).unwrap();
    assert_eq!(first.sequence(), 0);
    assert_eq!(second.sequence(), 1);
}

#[test]
fn a_closed_vertex_completes_what_it_is_given() {
    let registry = registry();
    let vertex = LatticeVertex::new(&registry, "closed", 1);
    vertex.close();
    let source = Arc::new(DeferredSource::default());

    assert_eq!(vertex.ingest(source.clone()).err(), Some(LatticeClosed));

    assert!(source.is_complete());
    assert_eq!(registry.upstream_handle_count(), 0);
}

#[test]
fn an_interceptor_takes_one_source() {
    let vertex = LatticeVertex::new(&registry(), "one", 1);
    let first = Arc::new(DeferredSource::default());
    let interceptor = vertex.ingest(first.clone()).unwrap();
    let second = Arc::new(DeferredSource::default());

    interceptor.add_upstream(second.clone());

    assert!(second.is_complete());
    assert!(!first.is_complete());
    let upstream: Arc<dyn LatticeSource> = first;
    assert!(interceptor.has_upstream(&upstream));
    let other: Arc<dyn LatticeSource> = second;
    assert!(!interceptor.has_upstream(&other));
}

#[test]
fn an_interceptor_routes_what_its_source_pushes() {
    let registry = registry();
    let vertex = LatticeVertex::new(&registry, "push", 1);
    let (edge, receiver) = receiving_edge(&registry);
    assert!(map(&vertex, &[0], &[node(&edge)]));
    let source = Arc::new(DeferredSource::default());
    vertex.ingest(source.clone()).unwrap();

    source.emit(frame(6));

    assert_eq!(receiver.received(), [6]);
}

#[test]
fn completing_an_interceptor_completes_its_source() {
    let registry = registry();
    let vertex = LatticeVertex::new(&registry, "cancel", 1);
    let source = Arc::new(DeferredSource::default());
    let interceptor = vertex.ingest(source.clone()).unwrap();

    interceptor.complete();
    interceptor.complete();

    assert!(source.is_complete());
    assert!(interceptor.is_complete());
    assert_eq!(registry.upstream_handle_count(), 0);
}

#[test]
fn completion_and_errors_from_the_source_complete_the_interceptor() {
    let registry = registry();
    let vertex = LatticeVertex::new(&registry, "complete", 1);
    let completing = vertex.ingest(Arc::new(DeferredSource::default())).unwrap();
    let failing = vertex.ingest(Arc::new(DeferredSource::default())).unwrap();
    assert_eq!(registry.upstream_handle_count(), 2);

    completing.on_complete();
    failing.on_error("broken".into());
    failing.on_complete();

    assert!(completing.is_complete());
    assert!(failing.is_complete());
    assert_eq!(registry.upstream_handle_count(), 0);
}

#[test]
fn an_interceptor_is_serviced_only_under_its_lock() {
    let vertex = LatticeVertex::new(&registry(), "lock", 1);
    let source = Arc::new(DeferredSource::default());
    let interceptor = vertex.ingest(source.clone()).unwrap();
    let pull = |interceptor: &UpstreamInterceptor, claim: &mut Claim, demand| {
        interceptor.pull(claim, &mut |_| {}, &mut |_| false, demand)
    };

    let mut claim = interceptor.try_acquire().expect("the handle is free");
    assert!(interceptor.try_acquire().is_none(), "one claim at a time");
    assert_eq!(pull(&interceptor, &mut claim, 1), Some(0));
    assert_eq!(pull(&interceptor, &mut claim, 0), None);
    interceptor.request(&mut claim, 0);
    interceptor.request(&mut claim, 2);
    assert_eq!(source.requested.load(Ordering::Relaxed), 2);

    vertex.set_drain(true);
    assert_eq!(pull(&interceptor, &mut claim, 1), None);
    interceptor.request(&mut claim, 2);
    assert_eq!(source.requested.load(Ordering::Relaxed), 2);
    vertex.set_drain(false);

    interceptor.release(claim);
    assert!(given_back(&interceptor));
}

#[test]
fn a_panicking_source_completes_its_interceptor() {
    let registry = registry();
    let vertex = LatticeVertex::new(&registry, "panic", 1);
    let pulled = Arc::new(DeferredSource::default());
    let requested = Arc::new(DeferredSource::default());
    let pulling = vertex.ingest(pulled.clone()).unwrap();
    let requesting = vertex.ingest(requested.clone()).unwrap();
    pulled.panics.store(true, Ordering::Relaxed);
    requested.panics.store(true, Ordering::Relaxed);
    let mut pulling_claim = pulling.try_acquire().unwrap();
    let mut requesting_claim = requesting.try_acquire().unwrap();

    assert_eq!(
        pulling.pull(&mut pulling_claim, &mut |_| {}, &mut |_| false, 1),
        None
    );
    requesting.request(&mut requesting_claim, 1);

    // The source is completed once the worker lets the interceptor go.
    assert!(pulling.is_complete() && requesting.is_complete());
    assert!(!pulled.is_complete() && !requested.is_complete());
    pulling.release(pulling_claim);
    requesting.release(requesting_claim);

    assert!(pulling.is_complete() && pulled.is_complete());
    assert!(requesting.is_complete() && requested.is_complete());
    assert_eq!(registry.upstream_handle_count(), 0);
}

#[test]
fn workers_service_ingested_sources() {
    let registry = registry();
    registry.register();
    let vertex = LatticeVertex::new(&registry, "workers", 1);
    let source = Arc::new(DeferredSource::default());
    vertex.ingest(source.clone()).unwrap();

    vertex.edge().request(5);

    assert_eq!(source.requested.load(Ordering::Relaxed), 5);
}

#[test]
fn a_held_generation_keeps_its_downstreams_open() {
    let registry = registry();
    let vertex = LatticeVertex::new(&registry, "snapshot", 5);
    let edges: Vec<_> = (0..5).map(|_| receiving_edge(&registry).0).collect();
    let handles: Vec<_> = edges.iter().map(node).collect();
    assert!(map(&vertex, &[1, 4], &handles));
    let previous = vertex.hold_routing();
    assert_eq!(previous.mappings(), [1, 4]);

    assert!(map(&vertex, &[2, 4], &handles));
    let current = vertex.hold_routing();

    assert!(!ptr::eq(&*previous, &*current));
    assert_eq!(current.mappings(), [2, 4]);
    assert_eq!(current.active_index(1), None);
    assert_eq!(current.active_index(2), Some(0));
    assert_eq!(current.active_index(4), Some(1));
    assert_eq!(
        previous.mappings(),
        [1, 4],
        "a published generation never changes"
    );
    assert_eq!(previous.active_index(1), Some(0));
    assert!(
        !edges[1].is_closed(),
        "the held generation still routes to it"
    );

    drop(previous);
    assert!(edges[1].is_closed());
    assert!(!edges[4].is_closed());
}

#[test]
fn a_quiesced_remap_routes_later_frames_to_the_new_generation() {
    let registry = registry();
    let vertex = LatticeVertex::new(&registry, "handoff", 3);
    let (retired, retired_receiver) = receiving_edge(&registry);
    let (replacement, replacement_receiver) = receiving_edge(&registry);
    let handles = [None, node(&retired), node(&replacement)];
    assert!(map(&vertex, &[1], &handles));

    let (delivered, delivered_rx) = mpsc::channel();
    let (remapped, remapped_rx) = mpsc::channel::<()>();
    let producer = {
        let vertex = vertex.clone();
        thread::spawn(move || {
            vertex.push(frame(1));
            delivered.send(()).unwrap();
            remapped_rx.recv().unwrap();
            vertex.push(frame(2));
        })
    };
    delivered_rx.recv().unwrap();
    assert_eq!(retired_receiver.received(), [1]);

    assert!(map(&vertex, &[2], &handles));
    assert!(retired.is_closed());
    assert!(retired_receiver.is_completed());
    remapped.send(()).unwrap();
    producer.join().unwrap();

    assert_eq!(retired_receiver.received(), [1]);
    assert_eq!(replacement_receiver.received(), [2]);
}

#[test]
fn a_push_after_a_remap_uses_the_live_generation() {
    let registry = registry();
    let vertex = LatticeVertex::new(&registry, "asynchronous", 1);
    let (retired, retired_receiver) = receiving_edge(&registry);
    assert!(map(&vertex, &[0], &[node(&retired)]));
    let source = Arc::new(DeferredSource::default());
    vertex.ingest(source.clone()).unwrap();

    let (replacement, replacement_receiver) = receiving_edge(&registry);
    assert!(map(&vertex, &[0], &[node(&replacement)]));
    let emitter = source.clone();
    thread::spawn(move || emitter.emit(frame(3)))
        .join()
        .unwrap();

    assert!(retired.is_closed());
    assert!(retired_receiver.received().is_empty());
    assert_eq!(replacement_receiver.received(), [3]);
}

#[test]
fn a_push_after_an_empty_remap_fails_on_the_producer_thread() {
    let registry = registry();
    let vertex = LatticeVertex::new(&registry, "asynchronous-empty", 1);
    let (edge, receiver) = receiving_edge(&registry);
    assert!(map(&vertex, &[0], &[node(&edge)]));
    let source = Arc::new(DeferredSource::default());
    vertex.ingest(source.clone()).unwrap();
    assert!(map(&vertex, &[], &[node(&edge)]));

    let fates = Fates::default();
    let (emitter, tracked) = (source.clone(), Tracked::new(5, &fates));
    thread::spawn(move || emitter.emit(tracked)).join().unwrap();

    assert!(edge.is_closed());
    assert!(receiver.received().is_empty());
    assert_eq!(
        *fates.lock().unwrap(),
        [(
            5,
            Fate::Failed("frame failed: no active downstream to route the frame to".into())
        )]
    );
}

#[test]
fn pushes_racing_remaps_always_reach_an_open_downstream() {
    let registry = registry();
    let vertex = LatticeVertex::new(&registry, "race", 1);
    let generations: Vec<_> = (0..2).map(|_| receiving_edge(&registry)).collect();
    let fates = Fates::default();
    let stop = Arc::new(AtomicBool::new(false));
    let mut current = 0;
    assert!(map(&vertex, &[0], &[node(&generations[current].0)]));

    let producers: Vec<_> = (0..2)
        .map(|producer| {
            let (vertex, fates, stop) = (vertex.clone(), fates.clone(), stop.clone());
            thread::spawn(move || {
                // A pusher reads the routing from a view of its own.
                vertex.edge().registry().register_reader();
                let mut sent = 0;
                while !stop.load(Ordering::Acquire) {
                    vertex.push(Tracked::new(producer * 1_000_000 + sent, &fates));
                    sent += 1;
                }
                sent
            })
        })
        .collect();

    let mut remaps = Vec::new();
    for _ in 0..200 {
        let (edge, receiver) = receiving_edge(&registry);
        vertex.set_drain(true);
        assert!(vertex.set_downstream_mapping(&[0], &[node(&edge)]));
        vertex.set_drain(false);
        remaps.push((edge, receiver));
        current += 1;
    }
    stop.store(true, Ordering::Release);
    let sent: u64 = producers
        .into_iter()
        .map(|producer| producer.join().unwrap())
        .sum();
    let delivered: usize = generations
        .iter()
        .chain(&remaps)
        .map(|(_, receiver)| receiver.received().len())
        .sum();
    assert!(fates.lock().unwrap().is_empty(), "no frame was refused");
    assert_eq!(delivered as u64, sent);
    assert!(
        remaps[..remaps.len() - 1]
            .iter()
            .all(|(edge, _)| edge.is_closed())
    );
    assert!(current > 0);
}

/// Counts what reaches it, and what reaches it after it was told the edge above completed.
#[derive(Default)]
struct Counting {
    pushed: AtomicUsize,
    completions: AtomicUsize,
    late: AtomicUsize,
}

impl LatticeTerminal for Counting {
    fn add_upstream(&self, _upstream: Arc<dyn LatticeSource>) {}
}

impl LatticeReceiver for Counting {
    fn push(&self, _frame: Box<dyn Frame>) {
        if self.completions.load(Ordering::SeqCst) > 0 {
            self.late.fetch_add(1, Ordering::SeqCst);
        }
        self.pushed.fetch_add(1, Ordering::SeqCst);
    }

    fn on_complete(&self) {
        self.completions.fetch_add(1, Ordering::SeqCst);
    }

    fn on_error(&self, _error: Failure) {}
}

fn counting_edge(registry: &Arc<UpstreamRegistry>) -> (Arc<LatticeEdge>, Arc<Counting>) {
    let edge = LatticeEdge::new(registry, Arc::new(AtomicBool::new(false)));
    let receiver = Arc::new(Counting::default());
    edge.add_downstream(receiver.clone());
    (edge, receiver)
}

#[cfg(not(miri))]
const REMAPS: usize = 1_000;
#[cfg(miri)]
const REMAPS: usize = 12;
/// How many times the remap race runs in one test.
#[cfg(not(miri))]
const RACES: usize = 20;
#[cfg(miri)]
const RACES: usize = 1;
#[cfg(not(miri))]
const PUSHERS: u64 = 8;
#[cfg(miri)]
const PUSHERS: u64 = 3;

/// What `pushers` threads pushing through a two-way vertex saw while it was remapped `remaps` times, each time to
/// two fresh downstreams.
struct RemapRace {
    sent: u64,
    receivers: Vec<Arc<Counting>>,
    edges: Vec<Arc<LatticeEdge>>,
    fates: Fates,
}

fn race_remaps(pushers: u64, remaps: usize) -> RemapRace {
    let registry = registry();
    let vertex = LatticeVertex::new(&registry, "validated", 2);
    let fates = Fates::default();
    let stop = Arc::new(AtomicBool::new(false));
    let started = Arc::new(AtomicUsize::new(0));
    let mut receivers = Vec::new();
    let mut edges = Vec::new();
    let mut remap = |vertex: &LatticeVertex| {
        let pair: Vec<_> = (0..2).map(|_| counting_edge(&registry)).collect();
        let handles: Vec<_> = pair.iter().map(|(edge, _)| node(edge)).collect();
        assert!(map(vertex, &[0, 1], &handles));
        for (edge, receiver) in pair {
            edges.push(edge);
            receivers.push(receiver);
        }
    };
    remap(&vertex);

    let producers: Vec<_> = (0..pushers)
        .map(|producer| {
            let (vertex, fates, stop) = (vertex.clone(), fates.clone(), stop.clone());
            let started = started.clone();
            thread::spawn(move || {
                vertex.edge().registry().register_reader();
                let mut sent = 0_u64;
                while !stop.load(Ordering::Acquire) {
                    if sent == 1 {
                        started.fetch_add(1, Ordering::SeqCst);
                    }
                    let id = producer * 1_000_000 + sent;
                    let mut frame = Tracked::new(id, &fates);
                    frame
                        .header_mut()
                        .set_routing_hash(id.wrapping_mul(0x9E37_79B9_7F4A_7C15));
                    vertex.push(frame);
                    sent += 1;
                }
                sent
            })
        })
        .collect();
    // Remap only once every pusher is pushing, and keep pushing until the last remap is done.
    while started.load(Ordering::SeqCst) < pushers as usize {
        thread::yield_now();
    }
    for _ in 0..remaps {
        remap(&vertex);
    }
    stop.store(true, Ordering::Release);
    let sent = producers
        .into_iter()
        .map(|producer| producer.join().unwrap())
        .sum();
    RemapRace {
        sent,
        receivers,
        edges,
        fates,
    }
}

#[test]
fn pushes_racing_remaps_reach_only_validated_downstreams() {
    for _ in 0..RACES {
        check_pushes_racing_remaps_reach_only_validated_downstreams();
    }
}

/// One race of `pushes_racing_remaps_reach_only_validated_downstreams`.
fn check_pushes_racing_remaps_reach_only_validated_downstreams() {
    let race = race_remaps(PUSHERS, REMAPS);
    assert!(race.sent >= PUSHERS, "every pusher pushed");
    assert!(
        race.fates.lock().unwrap().is_empty(),
        "no frame was refused: {:?}",
        race.fates
            .lock()
            .unwrap()
            .iter()
            .take(3)
            .collect::<Vec<_>>()
    );
    let delivered: usize = race
        .receivers
        .iter()
        .map(|receiver| receiver.pushed.load(Ordering::SeqCst))
        .sum();
    assert_eq!(delivered as u64, race.sent, "no frame is lost or doubled");
    let late: usize = race
        .receivers
        .iter()
        .map(|receiver| receiver.late.load(Ordering::SeqCst))
        .sum();
    assert_eq!(late, 0, "a frame reached a downstream a remap had closed");
}

#[test]
fn the_last_reader_of_a_retired_generation_closes_its_dropped_downstreams_once() {
    for _ in 0..RACES {
        check_the_last_reader_closes_its_dropped_downstreams();
    }
}

/// One race of `the_last_reader_of_a_retired_generation_closes_its_dropped_downstreams_once`.
fn check_the_last_reader_closes_its_dropped_downstreams() {
    let race = race_remaps(PUSHERS, REMAPS);
    let (last, dropped) = race.receivers.split_at(race.receivers.len() - 2);
    for receiver in last {
        assert_eq!(
            receiver.completions.load(Ordering::SeqCst),
            1,
            "a dropped downstream is closed exactly once"
        );
    }
    for receiver in dropped {
        assert_eq!(receiver.completions.load(Ordering::SeqCst), 0);
    }
    assert!(
        race.edges[..race.edges.len() - 2]
            .iter()
            .all(|edge| edge.is_closed())
    );
    assert!(
        race.edges[race.edges.len() - 2..]
            .iter()
            .all(|edge| !edge.is_closed())
    );
}

#[test]
fn a_panicking_routing_function_gives_its_count_back() {
    let registry = registry();
    let vertex = LatticeVertex::with_routing(&registry, "route-panic", 1, |_, _| {
        panic!("the routing function broke")
    });
    let (edge, receiver) = receiving_edge(&registry);
    assert!(map(&vertex, &[0], &[node(&edge)]));

    let pushed = panic::catch_unwind(AssertUnwindSafe(|| vertex.try_push(frame(1))));
    assert!(pushed.is_err());
    assert!(receiver.received().is_empty());

    // A leaked count would keep the retired generation from releasing its downstream.
    assert!(map(&vertex, &[], &[]));
    assert!(
        edge.is_closed(),
        "the generation's last reader gave its count back"
    );
}

#[test]
fn releasing_the_last_reader_inside_a_hot_scope_takes_the_book_off_it() {
    let registry = registry();
    let vertex = LatticeVertex::new(&registry, "hot-release", 1);
    let (edge, receiver) = receiving_edge(&registry);
    assert!(map(&vertex, &[0], &[node(&edge)]));
    let held = vertex.hold_routing();
    assert!(map(&vertex, &[], &[]));
    assert!(!edge.is_closed());

    // The thread that drops the last count may be a worker mid-push; the book is lifecycle state, and a debug build
    // aborts when it is taken inside a hot scope.
    let _hot = HotScope::enter();
    drop(held);

    assert!(edge.is_closed());
    assert!(receiver.is_completed());
}

#[test]
fn closing_publishes_no_active_downstreams() {
    let registry = registry();
    let vertex = LatticeVertex::new(&registry, "closed-map", 2);
    let (edge, _) = receiving_edge(&registry);
    assert!(map(&vertex, &[1], &[None, node(&edge)]));

    vertex.close();

    assert_eq!(vertex.active_downstream_index(0), None);
    assert_eq!(vertex.active_downstream_index(1), None);
}

fn counter() -> (Arc<AtomicUsize>, impl FnOnce() + Send + 'static) {
    let count = Arc::new(AtomicUsize::new(0));
    let increment = count.clone();
    (count, move || {
        increment.fetch_add(1, Ordering::AcqRel);
    })
}

#[test]
fn retirement_waits_until_no_generation_routes_to_the_child() {
    let registry = registry();
    let vertex = LatticeVertex::new(&registry, "retire", 2);
    let (child, _) = receiving_edge(&registry);
    let (other, _) = receiving_edge(&registry);
    assert!(map(&vertex, &[0], &[node(&child)]));
    let held = vertex.hold_routing();

    let (retired, action) = counter();
    child.defer_retirement(Box::new(action));
    assert_eq!(retired.load(Ordering::Acquire), 0);

    assert!(map(&vertex, &[1], &[None, node(&other)]));
    assert_eq!(
        retired.load(Ordering::Acquire),
        0,
        "a held generation still routes to it"
    );
    assert!(!child.is_closed());

    drop(held);
    assert_eq!(retired.load(Ordering::Acquire), 1);
    assert!(child.is_closed());

    let (late, action) = counter();
    child.defer_retirement(Box::new(action));
    assert_eq!(
        late.load(Ordering::Acquire),
        1,
        "an unrouted child retires at once"
    );
}

/// A retirement can close a clone and wait for its workers, and the last holder of a retired generation may be one of
/// those workers (or any worker): it must not wait there. The retirement goes to a thread of its own, and the worker's
/// `refresh_view` returns at once.
#[test]
fn a_retirement_never_runs_on_the_worker_that_let_go_of_the_generation_last() {
    const WAIT: Duration = Duration::from_millis(400);
    let registry = registry();
    let vertex = Arc::new(LatticeVertex::new(&registry, "retire-worker", 2));
    let (child, _) = receiving_edge(&registry);
    let (other, _) = receiving_edge(&registry);
    assert!(map(&vertex, &[0], &[node(&child)]));
    let ran_on = Arc::new(Mutex::new(None));
    let done = Arc::new(AtomicBool::new(false));
    {
        let (ran_on, done) = (ran_on.clone(), done.clone());
        child.defer_retirement(Box::new(move || {
            *ran_on.lock().unwrap() = thread::current().name().map(str::to_owned);
            thread::sleep(WAIT);
            done.store(true, Ordering::Release);
        }));
    }
    let (ready_tx, ready_rx) = mpsc::channel();
    let (go_tx, go_rx) = mpsc::channel::<()>();
    let worker = {
        let (registry, vertex) = (registry.clone(), vertex.clone());
        thread::Builder::new()
            .name("worker-b".into())
            .spawn(move || {
                registry.register();
                // The worker's view holds the first generation.
                assert_eq!(vertex.with_routing_state(|state| state.len()), 1);
                ready_tx.send(()).unwrap();
                go_rx.recv().unwrap();
                let started = Instant::now();
                registry.refresh_view();
                started.elapsed()
            })
            .unwrap()
    };
    ready_rx.recv().unwrap();
    // The new generation no longer routes to the child; the worker's view is the first generation's last holder.
    assert!(map(&vertex, &[1], &[None, node(&other)]));
    assert!(!done.load(Ordering::Acquire));
    go_tx.send(()).unwrap();
    let refresh = worker.join().unwrap();
    assert!(
        refresh < WAIT / 4,
        "refresh_view took {refresh:?}: the retirement ran on the worker"
    );
    let deadline = Instant::now() + Duration::from_secs(10);
    while !done.load(Ordering::Acquire) {
        assert!(Instant::now() < deadline, "the retirement never ran");
        thread::sleep(Duration::from_millis(1));
    }
    let ran_on = ran_on.lock().unwrap().clone();
    assert_eq!(ran_on.as_deref(), Some("retire-worker-retire"));
    assert!(child.is_closed());
}

/// A light reader (a producer thread, another lattice's worker) can be the last holder too: the generation it upgraded
/// for a read outlives the writer's own hold, and its release at the end of the read must not run the retirement there.
#[test]
fn a_retirement_never_runs_on_a_light_reader_that_let_go_of_the_generation_last() {
    const WAIT: Duration = Duration::from_millis(400);
    let registry = registry();
    let vertex = Arc::new(LatticeVertex::new(&registry, "retire-light", 2));
    let (child, _) = receiving_edge(&registry);
    let (other, _) = receiving_edge(&registry);
    assert!(map(&vertex, &[0], &[node(&child)]));
    let ran_on = Arc::new(Mutex::new(None));
    let done = Arc::new(AtomicBool::new(false));
    {
        let (ran_on, done) = (ran_on.clone(), done.clone());
        child.defer_retirement(Box::new(move || {
            *ran_on.lock().unwrap() = thread::current().name().map(str::to_owned);
            thread::sleep(WAIT);
            done.store(true, Ordering::Release);
        }));
    }
    let (reading_tx, reading_rx) = mpsc::channel();
    let (go_tx, go_rx) = mpsc::channel::<()>();
    let reader = {
        let vertex = vertex.clone();
        thread::Builder::new()
            .name("producer".into())
            .spawn(move || {
                // Not a worker: a light reader. It holds the first generation for the length of this read.
                let started = vertex.with_routing_state(|state| {
                    assert_eq!(state.len(), 1);
                    reading_tx.send(()).unwrap();
                    go_rx.recv().unwrap();
                    Instant::now()
                });
                started.elapsed()
            })
            .unwrap()
    };
    reading_rx.recv().unwrap();
    // The remap replaces the generation; the reader's upgrade is now the last hold on it.
    assert!(map(&vertex, &[1], &[None, node(&other)]));
    assert!(!done.load(Ordering::Acquire));
    go_tx.send(()).unwrap();
    let leaving = reader.join().unwrap();
    assert!(
        leaving < WAIT / 4,
        "the end of the read took {leaving:?}: the retirement ran on the reader"
    );
    let deadline = Instant::now() + Duration::from_secs(10);
    while !done.load(Ordering::Acquire) {
        assert!(Instant::now() < deadline, "the retirement never ran");
        thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(
        ran_on.lock().unwrap().as_deref(),
        Some("retire-light-retire")
    );
}

/// The vertex's own drop can be the last hold on its generation, on a frame-path thread that held the last upgraded
/// `Weak`. The retirements still waiting for a generation to be released would never run (the release needs the vertex,
/// which is gone), so the drop hands them to a thread of their own, and waits for nothing.
#[test]
fn dropping_a_vertex_hands_its_pending_retirements_to_a_thread_and_waits_for_nothing() {
    const WAIT: Duration = Duration::from_millis(400);
    let registry = registry();
    let vertex = Arc::new(LatticeVertex::new(&registry, "retire-drop", 1));
    let (child, _) = receiving_edge(&registry);
    assert!(map(&vertex, &[0], &[node(&child)]));
    // A route holds the generation, so the retirements wait for it.
    let held = vertex.hold_routing();
    let ran_on = Arc::new(Mutex::new(None));
    let done = Arc::new(AtomicBool::new(false));
    {
        let (ran_on, done) = (ran_on.clone(), done.clone());
        child.defer_retirement(Box::new(move || {
            *ran_on.lock().unwrap() = thread::current().name().map(str::to_owned);
            thread::sleep(WAIT);
            done.store(true, Ordering::Release);
        }));
    }
    let closed = Arc::new(AtomicBool::new(false));
    {
        let closed = closed.clone();
        vertex.close_after_routes_retire(move || closed.store(true, Ordering::Release));
    }
    let worker = thread::Builder::new()
        .name("worker-d".into())
        .spawn({
            let registry = registry.clone();
            move || {
                registry.register();
                let started = Instant::now();
                drop(vertex);
                started.elapsed()
            }
        })
        .unwrap();
    let dropping = worker.join().unwrap();
    assert!(
        dropping < WAIT / 4,
        "dropping the vertex took {dropping:?}: a retirement ran on the dropping thread"
    );
    drop(held);
    let deadline = Instant::now() + Duration::from_secs(10);
    while !(done.load(Ordering::Acquire) && closed.load(Ordering::Acquire)) {
        assert!(Instant::now() < deadline, "the retirements never ran");
        thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(
        ran_on.lock().unwrap().as_deref(),
        Some("retire-drop-retire")
    );
}

#[test]
fn a_retirement_never_runs_in_the_exit_of_a_thread_whose_mailbox_held_the_last_generation() {
    const WAIT: Duration = Duration::from_millis(400);
    let registry = registry();
    let vertex = Arc::new(LatticeVertex::new(&registry, "retire-exit", 2));
    let (child, _) = receiving_edge(&registry);
    let (other, _) = receiving_edge(&registry);
    // The first generation routes to `other` only.
    assert!(map(&vertex, &[1], &[None, node(&other)]));
    let ran_on = Arc::new(Mutex::new(None));
    let done = Arc::new(AtomicBool::new(false));
    let (ready_tx, ready_rx) = mpsc::channel();
    let (go_tx, go_rx) = mpsc::channel::<()>();
    let reader = thread::Builder::new()
        .name("exiting".into())
        .spawn({
            let (registry, vertex) = (registry.clone(), vertex.clone());
            move || {
                registry.register_reader();
                assert_eq!(vertex.with_routing_state(|state| state.len()), 1);
                ready_tx.send(()).unwrap();
                go_rx.recv().unwrap();
                // Exits without draining: the later generations are updates in its mailbox.
            }
        })
        .unwrap();
    ready_rx.recv().unwrap();
    // Generation two routes to the child; generation three does not. Once the store lets go of two, the reader's
    // mailbox holds the last reference to it.
    assert!(map(&vertex, &[0], &[node(&child)]));
    {
        let (ran_on, done) = (ran_on.clone(), done.clone());
        child.defer_retirement(Box::new(move || {
            *ran_on.lock().unwrap() = thread::current().name().map(str::to_owned);
            thread::sleep(WAIT);
            done.store(true, Ordering::Release);
        }));
    }
    assert!(map(&vertex, &[1], &[None, node(&other)]));
    assert!(!done.load(Ordering::Acquire));
    let started = Instant::now();
    go_tx.send(()).unwrap();
    reader.join().unwrap();
    let exiting = started.elapsed();
    assert!(
        exiting < WAIT / 4,
        "the thread's exit took {exiting:?}: the retirement ran in its destructor"
    );
    let deadline = Instant::now() + Duration::from_secs(10);
    while !done.load(Ordering::Acquire) {
        assert!(Instant::now() < deadline, "the retirement never ran");
        thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(
        ran_on.lock().unwrap().as_deref(),
        Some("retire-exit-retire")
    );
}

#[test]
fn a_panicking_retirement_does_not_stop_the_rest() {
    let registry = registry();
    let vertex = LatticeVertex::new(&registry, "retire-panic", 1);
    let (child, _) = receiving_edge(&registry);
    assert!(map(&vertex, &[0], &[node(&child)]));
    child.defer_retirement(Box::new(|| panic!("retirement failed")));
    let (retired, action) = counter();
    child.defer_retirement(Box::new(action));

    assert!(map(&vertex, &[], &[]));

    assert_eq!(retired.load(Ordering::Acquire), 1);
    assert!(child.is_closed());
}

#[test]
fn close_retirements_wait_for_held_generations() {
    let registry = registry();
    let vertex = LatticeVertex::new(&registry, "close-retire", 1);
    let (child, _) = receiving_edge(&registry);
    assert!(map(&vertex, &[0], &[node(&child)]));
    let held = vertex.hold_routing();

    let (first, action) = counter();
    vertex.close_after_routes_retire(action);
    let (second, action) = counter();
    vertex.close_after_routes_retire(action);
    assert!(vertex.is_closed());
    assert_eq!(
        first.load(Ordering::Acquire) + second.load(Ordering::Acquire),
        0
    );
    assert!(!child.is_closed());

    drop(held);
    assert!(child.is_closed());
    assert_eq!(first.load(Ordering::Acquire), 1);
    assert_eq!(second.load(Ordering::Acquire), 1);

    let (late, action) = counter();
    vertex.close_after_routes_retire(action);
    assert_eq!(late.load(Ordering::Acquire), 1, "routing is already closed");
}

#[test]
fn closing_an_unmapped_vertex_retires_at_once() {
    let vertex = LatticeVertex::new(&registry(), "close-empty", 1);
    let (closed, action) = counter();
    vertex.close_after_routes_retire(action);
    assert_eq!(closed.load(Ordering::Acquire), 1);
}

#[test]
fn a_dropped_vertex_leaves_its_downstreams_alone() {
    let registry = registry();
    let vertex = LatticeVertex::new(&registry, "dropped", 1);
    let (child, receiver) = receiving_edge(&registry);
    assert!(map(&vertex, &[0], &[node(&child)]));

    drop(vertex);

    assert!(!child.is_closed());
    assert!(!receiver.is_completed());
    assert!(child.parent().is_none());
}

#[test]
fn close_retirements_wait_for_every_generation() {
    let registry = registry();
    let vertex = LatticeVertex::new(&registry, "close-generations", 1);
    let (first, _) = receiving_edge(&registry);
    let (second, _) = receiving_edge(&registry);
    assert!(map(&vertex, &[0], &[node(&first)]));
    let older = vertex.hold_routing();
    assert!(map(&vertex, &[0], &[node(&second)]));
    let newer = vertex.hold_routing();

    let (closed, action) = counter();
    vertex.close_after_routes_retire(action);
    drop(newer);
    assert!(second.is_closed());
    assert_eq!(
        closed.load(Ordering::Acquire),
        0,
        "the older generation still routes"
    );
    assert!(!first.is_closed());

    drop(older);
    assert!(first.is_closed());
    assert_eq!(closed.load(Ordering::Acquire), 1);
}

#[test]
fn a_push_during_a_request_counts_as_work_for_the_worker() {
    /// Pushes one queued frame per unit of demand, on the requesting thread.
    #[derive(Default)]
    struct Synchronous {
        downstream: Mutex<Option<Arc<dyn LatticeReceiver>>>,
        frames: Mutex<Vec<Box<dyn Frame>>>,
    }

    impl LatticeSource for Synchronous {
        fn add_downstream(self: Arc<Self>, downstream: Arc<dyn LatticeReceiver>) {
            *self.downstream.lock().unwrap() = Some(downstream);
        }
        fn pull(
            &self,
            _consumer: &mut dyn FnMut(Box<dyn Frame>),
            _stop: &mut dyn FnMut(&dyn Frame) -> bool,
            _demand: u64,
        ) -> u64 {
            0
        }
        fn request(&self, demand: u64) {
            let downstream = self.downstream.lock().unwrap().clone().unwrap();
            for _ in 0..demand {
                let Some(frame) = self.frames.lock().unwrap().pop() else {
                    return;
                };
                downstream.push(frame);
            }
        }
        fn complete(&self) {}
        fn is_complete(&self) -> bool {
            false
        }
    }

    let registry = registry();
    registry.register();
    let vertex = LatticeVertex::new(&registry, "evidence", 1);
    let (edge, receiver) = receiving_edge(&registry);
    assert!(map(&vertex, &[0], &[node(&edge)]));
    let source = Arc::new(Synchronous::default());
    vertex.ingest(source.clone()).unwrap();
    let productive = || registry.with_thread_queue(|queue| queue.productive_handle_count());

    assert_eq!(vertex.edge().pull(&mut |_| {}, &mut |_| false, 1), 0);
    assert_eq!(productive(), Some(0));

    source.frames.lock().unwrap().push(frame(8));
    vertex.edge().request(1);

    assert_eq!(receiver.received(), [8]);
    assert_eq!(productive(), Some(1));
}

#[test]
fn attaching_to_a_vertex_that_closed_meanwhile_completes_the_source() {
    let registry = registry();
    let vertex = LatticeVertex::new(&registry, "closed-meanwhile", 1);
    let interceptor = UpstreamInterceptor::new(&vertex);
    vertex.close();
    let source = Arc::new(DeferredSource::default());

    interceptor.add_upstream(source.clone());

    assert!(interceptor.is_complete());
    assert!(source.is_complete());
    assert_eq!(registry.upstream_handle_count(), 0);
}

#[test]
fn attaching_after_the_vertex_is_gone_completes_the_source() {
    let registry = registry();
    let vertex = LatticeVertex::new(&registry, "gone", 1);
    let interceptor = UpstreamInterceptor::new(&vertex);
    drop(vertex);
    let source = Arc::new(DeferredSource::default());

    interceptor.add_upstream(source.clone());

    assert!(interceptor.is_complete());
    assert!(source.is_complete());
    assert_eq!(registry.upstream_handle_count(), 0);
    let fates = Fates::default();
    interceptor.push(Tracked::new(1, &fates));
    assert!(matches!(fates.lock().unwrap()[0], (1, Fate::Failed(_))));
}

/// A source that records what it is doing, so a test can see actions interleave with it.
#[derive(Default)]
struct Watched {
    /// Inside a `pull` or `request`.
    busy: AtomicBool,
    /// Inside a serialized action.
    acting: AtomicBool,
    overlaps: AtomicUsize,
    complete: AtomicBool,
    drops: Mutex<Option<Arc<AtomicUsize>>>,
}

impl Watched {
    fn work(&self) {
        if self.acting.load(Ordering::SeqCst) {
            self.overlaps.fetch_add(1, Ordering::SeqCst);
        }
        self.busy.store(true, Ordering::SeqCst);
        std::hint::spin_loop();
        self.busy.store(false, Ordering::SeqCst);
    }
}

impl Drop for Watched {
    fn drop(&mut self) {
        if let Some(drops) = self.drops.lock().unwrap().as_ref() {
            drops.fetch_add(1, Ordering::SeqCst);
        }
    }
}

impl LatticeSource for Watched {
    fn add_downstream(self: Arc<Self>, _downstream: Arc<dyn LatticeReceiver>) {}

    fn pull(
        &self,
        _consumer: &mut dyn FnMut(Box<dyn Frame>),
        _stop: &mut dyn FnMut(&dyn Frame) -> bool,
        _demand: u64,
    ) -> u64 {
        self.work();
        0
    }

    fn request(&self, _demand: u64) {
        self.work();
    }

    fn complete(&self) {
        self.complete.store(true, Ordering::Release);
    }

    fn is_complete(&self) -> bool {
        self.complete.load(Ordering::Acquire)
    }
}

fn ingest_watched(name: &str) -> (Arc<LatticeVertex>, Arc<Watched>, Arc<UpstreamInterceptor>) {
    let vertex = LatticeVertex::new(&registry(), name, 1);
    let source = Arc::new(Watched::default());
    let interceptor = vertex.ingest(source.clone()).unwrap();
    (vertex, source, interceptor)
}

/// Whether the claim is free: takes it and gives it back.
fn given_back(interceptor: &UpstreamInterceptor) -> bool {
    interceptor
        .try_acquire()
        .map(|claim| interceptor.release(claim))
        .is_some()
}

/// An action that adds one to `counter`.
fn bump(counter: &Arc<AtomicUsize>) -> Box<dyn FnOnce() + Send> {
    let counter = counter.clone();
    Box::new(move || {
        counter.fetch_add(1, Ordering::SeqCst);
    })
}

#[test]
fn serialize_runs_at_once_when_no_worker_holds_the_source() {
    let (_vertex, _source, interceptor) = ingest_watched("serialize-now");
    let ran = Arc::new(AtomicUsize::new(0));

    interceptor.serialize(bump(&ran));

    assert_eq!(ran.load(Ordering::SeqCst), 1);
    assert!(given_back(&interceptor), "the claim was given back");
}

#[test]
fn serialize_runs_at_release_when_another_worker_holds_the_source() {
    let (_vertex, _source, interceptor) = ingest_watched("serialize-later");
    let (acquired_tx, acquired_rx) = mpsc::channel();
    let (go_tx, go_rx) = mpsc::channel::<()>();
    let worker = {
        let interceptor = interceptor.clone();
        thread::spawn(move || {
            let claim = interceptor.try_acquire().unwrap();
            acquired_tx.send(()).unwrap();
            go_rx.recv().unwrap();
            interceptor.release(claim);
            thread::current().id()
        })
    };
    acquired_rx.recv().unwrap();

    let ran_on = Arc::new(Mutex::new(None));
    let recorded = ran_on.clone();
    interceptor.serialize(Box::new(move || {
        *recorded.lock().unwrap() = Some(thread::current().id());
    }));
    assert!(
        ran_on.lock().unwrap().is_none(),
        "the action waits for the holder"
    );

    go_tx.send(()).unwrap();
    let worker_id = worker.join().unwrap();
    assert_eq!(*ran_on.lock().unwrap(), Some(worker_id));
}

/// A source whose `pull` serializes an action through its receiver.
struct SerializingSource {
    downstream: Mutex<Option<Arc<dyn LatticeReceiver>>>,
    in_pull: Arc<AtomicBool>,
    ran_during_pull: Arc<AtomicBool>,
    ran: Arc<AtomicBool>,
}

impl LatticeSource for SerializingSource {
    fn add_downstream(self: Arc<Self>, downstream: Arc<dyn LatticeReceiver>) {
        *self.downstream.lock().unwrap() = Some(downstream);
    }

    fn pull(
        &self,
        _consumer: &mut dyn FnMut(Box<dyn Frame>),
        _stop: &mut dyn FnMut(&dyn Frame) -> bool,
        _demand: u64,
    ) -> u64 {
        self.in_pull.store(true, Ordering::SeqCst);
        let downstream = self.downstream.lock().unwrap().clone().unwrap();
        let (in_pull, during, ran) = (
            self.in_pull.clone(),
            self.ran_during_pull.clone(),
            self.ran.clone(),
        );
        downstream.serialize(Box::new(move || {
            during.store(in_pull.load(Ordering::SeqCst), Ordering::SeqCst);
            ran.store(true, Ordering::SeqCst);
        }));
        self.in_pull.store(false, Ordering::SeqCst);
        0
    }

    fn request(&self, _demand: u64) {}

    fn complete(&self) {}

    fn is_complete(&self) -> bool {
        false
    }
}

#[test]
fn serialize_from_inside_the_sources_own_pull_runs_before_release() {
    let vertex = LatticeVertex::new(&registry(), "serialize-own-pull", 1);
    let (in_pull, during, ran) = (
        Arc::new(AtomicBool::new(false)),
        Arc::new(AtomicBool::new(false)),
        Arc::new(AtomicBool::new(false)),
    );
    let source = Arc::new(SerializingSource {
        downstream: Mutex::new(None),
        in_pull,
        ran_during_pull: during.clone(),
        ran: ran.clone(),
    });
    let interceptor = vertex.ingest(source).unwrap();
    let mut claim = interceptor.try_acquire().unwrap();

    assert_eq!(
        interceptor.pull(&mut claim, &mut |_| {}, &mut |_| false, 1),
        Some(0)
    );
    assert!(
        !ran.load(Ordering::SeqCst),
        "the action waits for the pull to end"
    );

    interceptor.release(claim);
    assert!(ran.load(Ordering::SeqCst));
    assert!(!during.load(Ordering::SeqCst), "and never overlapped it");
}

#[test]
fn an_action_never_runs_while_a_pull_or_request_is_in_progress() {
    const ACTIONS: usize = 10_000;
    let (_vertex, source, interceptor) = ingest_watched("never-overlaps");
    let ran = Arc::new(AtomicUsize::new(0));
    let workers: Vec<_> = (0..4)
        .map(|_| {
            let interceptor = interceptor.clone();
            thread::spawn(move || {
                for _ in 0..ACTIONS {
                    if let Some(mut claim) = interceptor.try_acquire() {
                        interceptor.request(&mut claim, 1);
                        interceptor.pull(&mut claim, &mut |_| {}, &mut |_| false, 1);
                        interceptor.release(claim);
                    }
                }
            })
        })
        .collect();
    let serializer = {
        let (interceptor, source, ran) = (interceptor.clone(), source.clone(), ran.clone());
        thread::spawn(move || {
            for _ in 0..ACTIONS {
                let (source, ran) = (source.clone(), ran.clone());
                interceptor.serialize(Box::new(move || {
                    source.acting.store(true, Ordering::SeqCst);
                    if source.busy.load(Ordering::SeqCst) {
                        source.overlaps.fetch_add(1, Ordering::SeqCst);
                    }
                    std::hint::spin_loop();
                    source.acting.store(false, Ordering::SeqCst);
                    ran.fetch_add(1, Ordering::SeqCst);
                }));
            }
        })
    };
    for worker in workers {
        worker.join().unwrap();
    }
    serializer.join().unwrap();

    assert_eq!(source.overlaps.load(Ordering::SeqCst), 0);
    assert_eq!(ran.load(Ordering::SeqCst), ACTIONS);
}

#[test]
fn release_races_serialize_and_no_action_is_stranded() {
    const ACTIONS: usize = 10_000;
    let (_vertex, _source, interceptor) = ingest_watched("no-strand");
    let ran = Arc::new(AtomicUsize::new(0));
    let workers: Vec<_> = (0..4)
        .map(|_| {
            let interceptor = interceptor.clone();
            thread::spawn(move || {
                for _ in 0..ACTIONS * 2 {
                    if let Some(claim) = interceptor.try_acquire() {
                        interceptor.release(claim);
                    }
                }
            })
        })
        .collect();
    let serializers: Vec<_> = (0..2)
        .map(|_| {
            let (interceptor, ran) = (interceptor.clone(), ran.clone());
            thread::spawn(move || {
                for _ in 0..ACTIONS / 2 {
                    interceptor.serialize(bump(&ran));
                }
            })
        })
        .collect();
    for thread in workers.into_iter().chain(serializers) {
        thread.join().unwrap();
    }

    assert_eq!(ran.load(Ordering::SeqCst), ACTIONS);
}

#[test]
fn actions_run_in_the_order_they_were_handed_over() {
    let (_vertex, _source, interceptor) = ingest_watched("order-of-actions");
    let claim = interceptor.try_acquire().unwrap();
    let seen = Arc::new(Mutex::new(Vec::new()));
    for id in 0..5 {
        let seen = seen.clone();
        interceptor.serialize(Box::new(move || seen.lock().unwrap().push(id)));
    }
    assert!(seen.lock().unwrap().is_empty());

    interceptor.release(claim);

    assert_eq!(*seen.lock().unwrap(), [0, 1, 2, 3, 4]);
}

#[test]
fn an_action_that_serializes_another_runs_it_in_the_same_drain() {
    let (_vertex, _source, interceptor) = ingest_watched("nested-serialize");
    let ran = Arc::new(AtomicUsize::new(0));
    let (inner, counter) = (interceptor.clone(), ran.clone());
    interceptor.serialize(Box::new(move || {
        inner.serialize(bump(&counter));
        assert_eq!(
            counter.load(Ordering::SeqCst),
            0,
            "queued behind this action"
        );
    }));
    assert_eq!(ran.load(Ordering::SeqCst), 1);
    assert!(given_back(&interceptor), "the claim was given back");
}

#[test]
fn a_panicking_action_does_not_stop_the_rest() {
    let (_vertex, _source, interceptor) = ingest_watched("panicking-action");
    let ran = Arc::new(AtomicUsize::new(0));
    let claim = interceptor.try_acquire().unwrap();
    interceptor.serialize(Box::new(|| panic!("an action broke")));
    interceptor.serialize(bump(&ran));

    interceptor.release(claim);

    assert_eq!(ran.load(Ordering::SeqCst), 1);
    assert!(given_back(&interceptor));
}

#[test]
fn the_interceptor_drops_its_source_when_it_finishes() {
    let vertex = LatticeVertex::new(&registry(), "drops-source", 1);
    let drops = Arc::new(AtomicUsize::new(0));
    let source = Arc::new(Watched::default());
    *source.drops.lock().unwrap() = Some(drops.clone());
    let interceptor = vertex.ingest(source).unwrap();
    let claim = interceptor.try_acquire().unwrap();

    interceptor.on_complete();
    assert_eq!(
        drops.load(Ordering::SeqCst),
        0,
        "a worker still holds the source"
    );

    interceptor.release(claim);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

#[test]
fn an_interceptor_does_not_keep_a_source_that_holds_it() {
    let vertex = LatticeVertex::new(&registry(), "no-cycle", 1);
    let source = Arc::new(Watched::default());
    let weak = Arc::downgrade(&source);
    let interceptor = vertex.ingest(source).unwrap();
    assert!(weak.upgrade().is_some());

    interceptor.on_error("broken".into());

    assert!(
        weak.upgrade().is_none(),
        "the source held the interceptor, and the interceptor let the source go"
    );
}

#[test]
fn a_source_that_arrives_after_the_interceptor_finished_is_completed_and_not_kept() {
    let vertex = LatticeVertex::new(&registry(), "late-source", 1);
    let interceptor = UpstreamInterceptor::new(&vertex);
    interceptor.on_complete();
    let source = Arc::new(Watched::default());
    let weak = Arc::downgrade(&source);

    interceptor.add_upstream(source.clone());
    assert!(source.is_complete());
    drop(source);

    assert!(weak.upgrade().is_none());
}

#[test]
fn complete_completes_the_source_through_serialize() {
    let (_vertex, source, interceptor) = ingest_watched("complete-serialized");
    let claim = interceptor.try_acquire().unwrap();

    interceptor.complete();
    assert!(interceptor.is_complete());
    assert!(!source.is_complete(), "the holder is inside the source");

    interceptor.release(claim);
    assert!(source.is_complete());
}

#[test]
fn a_release_and_a_serialize_in_the_same_instant_never_strand_the_action() {
    const ROUNDS: usize = 20_000;
    let (_vertex, _source, interceptor) = ingest_watched("same-instant");
    let ran = Arc::new(AtomicUsize::new(0));
    let start = Arc::new(AtomicUsize::new(0));
    let finished = Arc::new(AtomicUsize::new(0));
    let spin_until = |flag: &AtomicUsize, value: usize| {
        while flag.load(Ordering::Acquire) < value {
            std::hint::spin_loop();
        }
    };
    let worker = {
        let (interceptor, start, finished) = (interceptor.clone(), start.clone(), finished.clone());
        thread::spawn(move || {
            for round in 1..=ROUNDS {
                spin_until(&start, round);
                if let Some(claim) = interceptor.try_acquire() {
                    interceptor.release(claim);
                }
                finished.fetch_add(1, Ordering::AcqRel);
            }
        })
    };
    let serializer = {
        let (interceptor, ran, start, finished) = (
            interceptor.clone(),
            ran.clone(),
            start.clone(),
            finished.clone(),
        );
        thread::spawn(move || {
            for round in 1..=ROUNDS {
                spin_until(&start, round);
                interceptor.serialize(bump(&ran));
                finished.fetch_add(1, Ordering::AcqRel);
            }
        })
    };
    for round in 1..=ROUNDS {
        start.store(round, Ordering::Release);
        spin_until(&finished, round * 2);
        assert_eq!(
            ran.load(Ordering::SeqCst),
            round,
            "round {round} stranded its action"
        );
    }
    worker.join().unwrap();
    serializer.join().unwrap();
}

#[test]
fn two_serializers_and_a_release_in_the_same_instant_never_strand_an_action() {
    const ROUNDS: usize = 20_000;
    let (_vertex, _source, interceptor) = ingest_watched("two-serializers");
    let ran = Arc::new(AtomicUsize::new(0));
    let start = Arc::new(AtomicUsize::new(0));
    let finished = Arc::new(AtomicUsize::new(0));
    let spin_until = |flag: &AtomicUsize, value: usize| {
        while flag.load(Ordering::Acquire) < value {
            std::hint::spin_loop();
        }
    };
    let worker = {
        let (interceptor, start, finished) = (interceptor.clone(), start.clone(), finished.clone());
        thread::spawn(move || {
            for round in 1..=ROUNDS {
                spin_until(&start, round);
                if let Some(claim) = interceptor.try_acquire() {
                    interceptor.release(claim);
                }
                finished.fetch_add(1, Ordering::AcqRel);
            }
        })
    };
    let serializers: Vec<_> = (0..2)
        .map(|_| {
            let (interceptor, ran, start, finished) = (
                interceptor.clone(),
                ran.clone(),
                start.clone(),
                finished.clone(),
            );
            thread::spawn(move || {
                for round in 1..=ROUNDS {
                    spin_until(&start, round);
                    interceptor.serialize(bump(&ran));
                    finished.fetch_add(1, Ordering::AcqRel);
                }
            })
        })
        .collect();
    for round in 1..=ROUNDS {
        start.store(round, Ordering::Release);
        spin_until(&finished, round * 3);
        assert_eq!(
            ran.load(Ordering::SeqCst),
            round * 2,
            "round {round} stranded an action"
        );
    }
    worker.join().unwrap();
    for serializer in serializers {
        serializer.join().unwrap();
    }
}

#[test]
fn the_source_is_in_place_before_a_worker_can_find_the_handle() {
    for round in 0..300 {
        let registry = registry();
        let vertex = LatticeVertex::new(&registry, "installed-first", 1);
        let interceptor = UpstreamInterceptor::new(&vertex);
        let source = Arc::new(DeferredSource::default());
        let observer = {
            let (registry, interceptor, source) =
                (registry.clone(), interceptor.clone(), source.clone());
            thread::spawn(move || {
                while registry.upstream_handle_count() == 0 {
                    std::hint::spin_loop();
                }
                let mut claim = interceptor.try_acquire().expect("nothing else holds it");
                interceptor.request(&mut claim, 1);
                interceptor.release(claim);
                assert_eq!(
                    source.requested.load(Ordering::Relaxed),
                    1,
                    "round {round}: a request that found the handle reached the source"
                );
            })
        };
        interceptor.add_upstream(source);
        observer.join().unwrap();
    }
}

#[test]
fn a_new_source_is_never_taken_for_a_dropped_one_at_the_same_address() {
    let vertex = LatticeVertex::new(&registry(), "identity", 1);
    let source = Arc::new(Watched::default());
    let first: Arc<dyn LatticeSource> = source.clone();
    let interceptor = vertex.ingest(source).unwrap();
    assert!(interceptor.has_upstream(&first));
    interceptor.complete();
    drop(first);

    // Were the dropped source's allocation free, one of these would take it.
    let fresh: Vec<Arc<dyn LatticeSource>> = (0..256)
        .map(|_| Arc::new(Watched::default()) as Arc<dyn LatticeSource>)
        .collect();
    assert!(fresh.iter().all(|source| !interceptor.has_upstream(source)));
}

#[test]
fn a_claim_taken_from_another_handle_is_refused() {
    let vertex = LatticeVertex::new(&registry(), "foreign-claim", 1);
    let first = vertex.ingest(Arc::new(Watched::default())).unwrap();
    let second = vertex.ingest(Arc::new(Watched::default())).unwrap();
    let mut claim = first.try_acquire().unwrap();

    let refused = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        second.request(&mut claim, 1);
    }));

    assert!(refused.is_err());
    first.release(claim);
    assert!(given_back(&second), "the other handle was left alone");
}

/// A source whose plain state is in a `SourceCell`, and whose lifecycle calls set atomics only.
struct Stateful {
    slot: DownstreamSlot,
    state: SourceCell<Vec<u32>>,
    cleared: AtomicBool,
    /// Makes the next `request` stop inside its state until the test lets it go.
    block: AtomicBool,
    inside: Barrier,
    hold: Barrier,
    applied: AtomicUsize,
}

impl Stateful {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            slot: DownstreamSlot::new(),
            state: SourceCell::new(Vec::new()),
            cleared: AtomicBool::new(false),
            block: AtomicBool::new(false),
            inside: Barrier::new(2),
            hold: Barrier::new(2),
            applied: AtomicUsize::new(0),
        })
    }

    /// A lifecycle call from any thread: it sets an atomic and nothing else.
    fn clear(&self) {
        self.cleared.store(true, Ordering::SeqCst);
    }
}

impl LatticeSource for Stateful {
    fn add_downstream(self: Arc<Self>, downstream: Arc<dyn LatticeReceiver>) {
        let source: Arc<dyn LatticeSource> = self.clone();
        self.slot.attach(source, downstream);
    }

    fn pull(
        &self,
        _consumer: &mut dyn FnMut(Box<dyn Frame>),
        _stop: &mut dyn FnMut(&dyn Frame) -> bool,
        _demand: u64,
    ) -> u64 {
        0
    }

    fn request(&self, demand: u64) {
        let mut state = self.state.enter();
        if self.cleared.swap(false, Ordering::SeqCst) {
            self.applied.fetch_add(state.len(), Ordering::SeqCst);
            state.clear();
        }
        state.push(demand as u32);
        if self.block.swap(false, Ordering::SeqCst) {
            self.inside.wait();
            self.hold.wait();
        }
    }

    fn complete(&self) {
        self.slot.complete();
    }

    fn is_complete(&self) -> bool {
        self.slot.is_complete()
    }
}

#[test]
fn lifecycle_calls_from_another_thread_touch_only_atomics_and_take_effect_on_the_next_request() {
    let vertex = LatticeVertex::new(&registry(), "lifecycle", 1);
    let source = Stateful::new();
    let interceptor = vertex.ingest(source.clone()).unwrap();
    let worker = |interceptor: Arc<UpstreamInterceptor>, requests: usize| {
        thread::spawn(move || {
            let mut claim = interceptor.try_acquire().expect("the handle is free");
            for _ in 0..requests {
                interceptor.request(&mut claim, 1);
            }
            interceptor.release(claim);
        })
    };

    // A clear while a worker is inside `request` waits for the next `request`.
    source.block.store(true, Ordering::SeqCst);
    let first = worker(interceptor.clone(), 2);
    source.inside.wait();
    source.clear();
    assert_eq!(
        source.applied.load(Ordering::SeqCst),
        0,
        "not applied from this thread"
    );
    source.hold.wait();
    first.join().unwrap();
    assert_eq!(
        source.applied.load(Ordering::SeqCst),
        1,
        "applied by the next request"
    );

    // A completion while a worker is inside `request` reaches the receiver at once; the source is
    // let go only after the worker has released it.
    source.block.store(true, Ordering::SeqCst);
    let second = worker(interceptor.clone(), 1);
    source.inside.wait();
    source.complete();
    assert!(interceptor.is_complete());
    assert_eq!(
        Arc::strong_count(&source),
        2,
        "the interceptor still holds the source"
    );
    source.hold.wait();
    second.join().unwrap();
    assert_eq!(Arc::strong_count(&source), 1, "released with the claim");
}

// ---------------------------------------------------------------------------------------------
// The placement policy on a source's pull.

/// A source whose pull hands out the frames it was given, as an ingest sink does, and whose request pushes nothing.
#[derive(Default)]
struct PullSource {
    frames: Mutex<std::collections::VecDeque<Box<dyn Frame>>>,
}

impl LatticeSource for PullSource {
    fn add_downstream(self: Arc<Self>, _downstream: Arc<dyn LatticeReceiver>) {}

    fn pull(
        &self,
        consumer: &mut dyn FnMut(Box<dyn Frame>),
        stop: &mut dyn FnMut(&dyn Frame) -> bool,
        demand: u64,
    ) -> u64 {
        let mut frames = self.frames.lock().unwrap();
        let mut pulled = 0;
        while pulled < demand {
            match frames.front() {
                Some(frame) if !stop(&**frame) => {
                    consumer(frames.pop_front().unwrap());
                    pulled += 1;
                }
                _ => break,
            }
        }
        pulled
    }

    fn request(&self, _demand: u64) {}

    fn complete(&self) {}

    fn is_complete(&self) -> bool {
        false
    }
}

/// A vertex with a placement, one active downstream below `receiver`, and a policy that places the frames by id (see the
/// comment in the policy).
fn placing_vertex(
    registry: &Arc<UpstreamRegistry>,
    policy: bool,
    receiver: Arc<dyn LatticeReceiver>,
) -> Arc<LatticeVertex> {
    let vertex = LatticeVertex::with_placement(
        registry,
        "placing",
        1,
        Placement::new(registry),
        |frame, state| route_by_hash(frame, state.len()),
    );
    let edge = LatticeEdge::new(registry, Arc::new(AtomicBool::new(false)));
    edge.add_downstream(receiver);
    assert!(map(&vertex, &[0], &[node(&edge)]));
    if policy {
        // The even ids go to core 1, which the puller (core 0) is not on, and must stay there; the ids 1 and 5 go to the
        // puller's own core; the ids 3 and 9 go to core 1 but may run anywhere.
        // SAFETY: the vertex has no source yet.
        assert!(unsafe {
            vertex.set_placement_policy(|frame, _| {
                let id = frame.header().id_hash();
                let (core, policy) = if id.is_multiple_of(2) {
                    (1, RoutingPolicy::CacheLocal)
                } else if id.is_multiple_of(3) {
                    (1, RoutingPolicy::Anywhere)
                } else {
                    (0, RoutingPolicy::CacheLocal)
                };
                let header = frame.header_mut();
                header.set_origin(Some(CpuInfo {
                    cpu: core,
                    core,
                    socket: 0,
                }));
                header.set_routing_policy(policy);
                true
            })
        });
    }
    vertex
}

/// Keeps the frames that reach it, to hand them back.
#[derive(Default)]
struct Recycler {
    frames: Mutex<Vec<Box<dyn Frame>>>,
}

impl LatticeTerminal for Recycler {
    fn add_upstream(&self, _upstream: Arc<dyn LatticeSource>) {}
}

impl LatticeReceiver for Recycler {
    fn push(&self, frame: Box<dyn Frame>) {
        self.frames.lock().unwrap().push(frame);
    }

    fn on_complete(&self) {}

    fn on_error(&self, _error: Failure) {}
}

fn unordered_frame(id: u64) -> Box<dyn Frame> {
    let mut frame = BenchmarkFrame::new(id, None);
    frame.header_mut().randomize_hash(1);
    Box::new(frame)
}

#[test]
fn a_pull_runs_the_frames_placed_on_its_core_and_routes_the_ones_placed_elsewhere() {
    let registry = registry();
    registry.register();
    registry.set_worker_location(0, 0);
    let receiver = TestReceiver::new();
    let vertex = placing_vertex(&registry, true, receiver.clone());
    let source = Arc::new(PullSource::default());
    source
        .frames
        .lock()
        .unwrap()
        .extend((1..=6).map(unordered_frame));
    let interceptor = vertex.ingest(source.clone()).unwrap();
    let mut claim = interceptor.try_acquire().unwrap();
    let mut direct = Vec::new();
    let pulled = interceptor.pull(
        &mut claim,
        &mut |frame| direct.push(frame.header().id_hash()),
        &mut |_| false,
        10,
    );
    assert_eq!(pulled, Some(6));
    // The frames placed on the puller's core, and the one that may run anywhere, reached the fragment's consumer; the
    // even ones, placed on another core and kept there, went through the routing.
    assert_eq!(direct, [1, 3, 5]);
    let mut routed = receiver.received();
    routed.sort_unstable();
    assert_eq!(routed, [2, 4, 6]);
    interceptor.release(claim);
}

#[test]
fn a_pull_whose_thread_did_not_say_where_it_runs_routes_every_frame_that_has_an_origin() {
    let registry = registry();
    registry.register();
    let receiver = TestReceiver::new();
    let vertex = placing_vertex(&registry, true, receiver.clone());
    let source = Arc::new(PullSource::default());
    source
        .frames
        .lock()
        .unwrap()
        .extend((1..=6).map(unordered_frame));
    let interceptor = vertex.ingest(source).unwrap();
    let mut claim = interceptor.try_acquire().unwrap();
    let mut direct = Vec::new();
    interceptor.pull(
        &mut claim,
        &mut |frame| direct.push(frame.header().id_hash()),
        &mut |_| false,
        10,
    );
    assert!(direct.is_empty());
    let mut routed = receiver.received();
    routed.sort_unstable();
    assert_eq!(routed, [1, 2, 3, 4, 5, 6]);
    interceptor.release(claim);
}

#[test]
fn the_policys_return_decides_which_frames_a_pull_may_route() {
    let registry = registry();
    registry.register();
    registry.set_worker_location(0, 0);
    let receiver = TestReceiver::new();
    let vertex = placing_vertex(&registry, false, receiver.clone());
    // SAFETY: the vertex has no source yet.
    assert!(unsafe {
        vertex.set_placement_policy(|frame, _| {
            // Every frame gets an origin on another core, but the policy says it placed only the ones above 3.
            let header = frame.header_mut();
            header.set_origin(Some(CpuInfo {
                cpu: 1,
                core: 1,
                socket: 0,
            }));
            header.set_routing_policy(RoutingPolicy::CacheLocal);
            header.id_hash() > 3
        })
    });
    let source = Arc::new(PullSource::default());
    source
        .frames
        .lock()
        .unwrap()
        .extend((1..=6).map(unordered_frame));
    let interceptor = vertex.ingest(source).unwrap();
    let mut claim = interceptor.try_acquire().unwrap();
    let mut direct = Vec::new();
    interceptor.pull(
        &mut claim,
        &mut |frame| direct.push(frame.header().id_hash()),
        &mut |_| false,
        10,
    );
    assert_eq!(direct, [1, 2, 3]);
    let mut routed = receiver.received();
    routed.sort_unstable();
    assert_eq!(routed, [4, 5, 6]);
    interceptor.release(claim);
}

#[test]
fn without_a_policy_a_pull_hands_every_frame_to_the_fragments_consumer() {
    let registry = registry();
    registry.register();
    let receiver = TestReceiver::new();
    let vertex = placing_vertex(&registry, false, receiver.clone());
    let source = Arc::new(PullSource::default());
    source
        .frames
        .lock()
        .unwrap()
        .extend((1..=6).map(unordered_frame));
    let interceptor = vertex.ingest(source).unwrap();
    let mut claim = interceptor.try_acquire().unwrap();
    let mut direct = Vec::new();
    interceptor.pull(
        &mut claim,
        &mut |frame| direct.push(frame.header().id_hash()),
        &mut |_| false,
        10,
    );
    assert_eq!(direct, [1, 2, 3, 4, 5, 6]);
    assert!(receiver.received().is_empty());
    interceptor.release(claim);
}

/// Counts the allocations of the thread that asks, so that tests running beside it do not disturb the count.
mod counting {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::cell::Cell;

    thread_local! {
        static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
    }

    pub(super) struct Counting;

    // SAFETY: every call is forwarded unchanged to the system allocator; the counter is a thread-local cell.
    unsafe impl GlobalAlloc for Counting {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            let _ = ALLOCATIONS.try_with(|count| count.set(count.get() + 1));
            // SAFETY: the caller's contract, forwarded.
            unsafe { System.alloc(layout) }
        }

        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            // SAFETY: the caller's contract, forwarded.
            unsafe { System.dealloc(ptr, layout) }
        }

        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            let _ = ALLOCATIONS.try_with(|count| count.set(count.get() + 1));
            // SAFETY: the caller's contract, forwarded.
            unsafe { System.realloc(ptr, layout, new_size) }
        }
    }

    pub(super) fn allocations() -> usize {
        ALLOCATIONS.with(Cell::get)
    }
}

#[global_allocator]
static COUNTING: counting::Counting = counting::Counting;

/// A pull allocates what it did without a policy, which is nothing: placing and routing a pulled frame add no allocation.
#[test]
fn placing_a_pulled_frame_allocates_nothing() {
    fn allocations_of_pulls(policy: bool) -> usize {
        let registry = registry();
        registry.register();
        registry.set_worker_location(0, 0);
        let receiver = Arc::new(Recycler::default());
        receiver.frames.lock().unwrap().reserve(512);
        let vertex = placing_vertex(&registry, policy, receiver.clone());
        let source = Arc::new(PullSource::default());
        source.frames.lock().unwrap().reserve(512);
        let interceptor = vertex.ingest(source.clone()).unwrap();
        let mut pool: Vec<Box<dyn Frame>> = (1..=512).map(unordered_frame).collect();
        let mut direct: Vec<Box<dyn Frame>> = Vec::with_capacity(512);
        let mut claim = interceptor.try_acquire().unwrap();
        let mut round = |pool: &mut Vec<Box<dyn Frame>>, direct: &mut Vec<Box<dyn Frame>>| {
            source.frames.lock().unwrap().extend(pool.drain(..));
            interceptor.pull(
                &mut claim,
                &mut |frame| direct.push(frame),
                &mut |_| false,
                512,
            );
            // The frames come back: the ones the fragment got, and the ones the routing delivered.
            pool.append(direct);
            pool.append(&mut receiver.frames.lock().unwrap());
        };
        // Warm-up: the first pulls make the worker's queue, grow the buffers.
        for _ in 0..3 {
            round(&mut pool, &mut direct);
        }
        let before = counting::allocations();
        for _ in 0..30 {
            round(&mut pool, &mut direct);
        }
        counting::allocations() - before
    }
    // The counter sees allocations at all.
    let probe = counting::allocations();
    drop(std::hint::black_box(Box::new(7u64)));
    assert!(
        counting::allocations() > probe,
        "the counting allocator is live"
    );
    assert_eq!(allocations_of_pulls(false), 0, "a pull allocates nothing");
    assert_eq!(
        allocations_of_pulls(true),
        0,
        "and placing and routing the frames it pulls allocate nothing"
    );
}
