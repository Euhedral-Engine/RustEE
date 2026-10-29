use super::*;
use crate::control_plane::testing::PATIENCE;
use crate::flow_control::testing::{Fate, Fates, TestReceiver, Tracked, frame};
use crate::frames::Failure;
use crate::generics::LatticeTerminal;
use std::sync::atomic::AtomicUsize;
use std::thread;
use std::time::Instant;

type Hook = Box<dyn Fn(&TestReceiver) + Send + Sync>;

/// A [`TestReceiver`] that runs a hook after each push and on completion, and counts completions.
#[derive(Default)]
struct Hooked {
    inner: TestReceiver,
    on_push: Option<Hook>,
    on_complete: Option<Hook>,
    completions: AtomicUsize,
}

impl Hooked {
    fn on_push(hook: impl Fn(&TestReceiver) + Send + Sync + 'static) -> Arc<Self> {
        Arc::new(Self {
            on_push: Some(Box::new(hook)),
            ..Self::default()
        })
    }

    fn on_complete(hook: impl Fn(&TestReceiver) + Send + Sync + 'static) -> Arc<Self> {
        Arc::new(Self {
            on_complete: Some(Box::new(hook)),
            ..Self::default()
        })
    }

    fn completions(&self) -> usize {
        self.completions.load(Ordering::Acquire)
    }
}

impl LatticeTerminal for Hooked {
    fn add_upstream(&self, upstream: Arc<dyn LatticeSource>) {
        self.inner.add_upstream(upstream);
    }
}

impl LatticeReceiver for Hooked {
    fn push(&self, frame: Box<dyn Frame>) {
        self.inner.push(frame);
        if let Some(hook) = &self.on_push {
            hook(&self.inner);
        }
    }

    fn on_complete(&self) {
        self.completions.fetch_add(1, Ordering::AcqRel);
        self.inner.on_complete();
        if let Some(hook) = &self.on_complete {
            hook(&self.inner);
        }
    }

    fn on_error(&self, error: Failure) {
        self.inner.on_error(error);
    }
}

fn upstream(receiver: &TestReceiver) -> Arc<dyn LatticeSource> {
    receiver.upstream.lock().unwrap().clone().expect("attached")
}

fn attached() -> (QueueIngestSink, Arc<dyn LatticeSource>, Arc<TestReceiver>) {
    let sink = QueueIngestSink::new();
    let source = sink.delegate();
    let receiver = TestReceiver::new();
    source.clone().add_downstream(receiver.clone());
    (sink, source, receiver)
}

fn offer(sink: &QueueIngestSink, ids: impl IntoIterator<Item = u64>) {
    for id in ids {
        assert!(sink.offer(frame(id)).is_ok());
    }
}

#[test]
fn attaching_hands_the_receiver_its_upstream() {
    let (sink, _, receiver) = attached();
    let upstream = upstream(&receiver);
    offer(&sink, [4]);
    upstream.request(1);
    assert_eq!(receiver.received(), [4]);
}

#[test]
fn requests_deliver_up_to_the_outstanding_demand() {
    let (sink, source, receiver) = attached();
    source.request(2);
    assert_eq!(sink.demand(), 2);
    offer(&sink, 0..3);
    assert!(receiver.received().is_empty());
    source.request(1);
    assert_eq!(receiver.received(), [0, 1, 2]);
    assert_eq!(sink.demand(), 0);
    assert_eq!(sink.size(), 0);
    offer(&sink, [3]);
    source.request(0);
    assert_eq!(receiver.received().len(), 3);
    assert_eq!(sink.size(), 1);
}

#[test]
fn nothing_is_delivered_without_a_receiver() {
    let sink = QueueIngestSink::new();
    let source = sink.delegate();
    offer(&sink, [1]);
    source.request(1);
    assert_eq!(sink.demand(), 0);
    assert_eq!(source.pull(&mut drop, &mut |_| false, 1), 0);
    assert_eq!(sink.size(), 1);
}

#[test]
fn a_request_inside_a_pull_consumer_is_delivered_after_the_pull_lets_go() {
    let (sink, source, receiver) = attached();
    offer(&sink, 1..=3);
    let inner = source.clone();
    let mut pulled = Vec::new();
    let count = source.pull(
        &mut |frame| {
            pulled.push(frame.header().id_hash());
            inner.request(1);
            assert!(receiver.received().is_empty());
        },
        &mut |frame| frame.header().id_hash() == 2,
        3,
    );
    assert_eq!(count, 1);
    assert_eq!(pulled, [1]);
    assert_eq!(receiver.received(), [2]);
    assert_eq!(sink.size(), 1);
    assert_eq!(sink.demand(), 0);
}

#[test]
fn a_reentrant_request_drains_without_another_external_request() {
    let sink = QueueIngestSink::new();
    let source = sink.delegate();
    let receiver = Hooked::on_push(|inner| upstream(inner).request(1));
    source.clone().add_downstream(receiver.clone());
    offer(&sink, [1, 2]);
    source.request(1);
    assert_eq!(receiver.inner.received(), [1, 2]);
    assert_eq!(sink.size(), 0);
    assert_eq!(sink.demand(), 1);
}

#[test]
fn a_late_receiver_hears_of_completion_exactly_once() {
    let sink = QueueIngestSink::new();
    let source = sink.delegate();
    source.complete();
    let receiver = Arc::new(Hooked::default());
    source.clone().add_downstream(receiver.clone());
    source.complete();
    assert_eq!(receiver.completions(), 1);
    assert!(receiver.inner.upstream.lock().unwrap().is_none());
    let second = TestReceiver::new();
    source.clone().add_downstream(second.clone());
    assert_eq!(second.errors(), ["the source already has a downstream"]);
    assert!(!second.is_completed());
}

#[test]
fn publication_is_closed_before_the_receiver_hears_of_completion() {
    let sink = Arc::new(QueueIngestSink::new());
    let late = sink.clone();
    let receiver = Hooked::on_complete(move |_| {
        let late = late.clone();
        let refused = thread::spawn(move || late.offer(frame(9)).is_err())
            .join()
            .unwrap();
        assert!(refused);
    });
    sink.delegate().add_downstream(receiver.clone());
    sink.complete();
    assert_eq!(receiver.completions(), 1);
    assert_eq!(sink.size(), 0);
}

#[test]
fn attaching_while_completing_notifies_exactly_once() {
    for _ in 0..200 {
        let sink = QueueIngestSink::new();
        let source = sink.delegate();
        let receiver = Arc::new(Hooked::default());
        thread::scope(|scope| {
            let completing = source.clone();
            scope.spawn(move || completing.complete());
            source.clone().add_downstream(receiver.clone());
        });
        source.complete();
        assert_eq!(receiver.completions(), 1);
    }
}

#[test]
fn graceful_completion_keeps_work_a_pull_stopped_before() {
    let (sink, source, receiver) = attached();
    offer(&sink, [1]);
    sink.complete_gracefully();
    assert_eq!(source.pull(&mut drop, &mut |_| true, 1), 0);
    assert!(!sink.is_complete());
    assert!(sink.offer(frame(2)).is_ok());
    source.request(2);
    assert_eq!(receiver.received(), [1, 2]);
    assert!(!sink.is_complete());
    source.request(1);
    assert!(sink.is_complete());
    assert!(receiver.is_completed());
    assert!(sink.offer(frame(3)).is_err());
}

#[test]
fn graceful_completion_of_an_empty_queue_ends_on_the_next_pull() {
    let (sink, source, receiver) = attached();
    sink.complete_gracefully();
    assert!(!sink.is_complete());
    assert_eq!(source.pull(&mut drop, &mut |_| false, 1), 0);
    assert!(sink.is_complete());
    assert!(receiver.is_completed());
}

#[test]
fn completing_closes_publication_and_hands_frames_back() {
    let (sink, source, receiver) = attached();
    offer(&sink, [1]);
    sink.complete();
    assert!(sink.is_complete());
    assert!(receiver.is_completed());
    let refused = sink.offer(frame(7)).unwrap_err();
    assert_eq!(refused.header().id_hash(), 7);
    assert!(sink.offer_seeded(1, frame(8)).is_err());
    assert!(sink.offer_to(0, frame(9)).is_err());
    let sink_side: Arc<dyn FrameSink> = sink.source.clone();
    assert!(sink_side.offer(frame(10)).is_err());
    source.request(5);
    assert_eq!(source.pull(&mut drop, &mut |_| false, 5), 0);
    assert!(receiver.received().is_empty());
    assert_eq!(sink.size(), 1);
    assert_eq!(sink.demand(), 0);
}

#[test]
fn completing_during_a_request_leaves_no_demand() {
    let sink = Arc::new(QueueIngestSink::new());
    let completing = sink.clone();
    let receiver = Hooked::on_push(move |_| completing.complete());
    sink.delegate().add_downstream(receiver.clone());
    offer(&sink, [1, 2]);
    sink.delegate().request(5);
    assert_eq!(receiver.inner.received(), [1]);
    assert_eq!(receiver.completions(), 1);
    assert_eq!(sink.demand(), 0);
    assert_eq!(sink.size(), 1);
    assert!(sink.is_complete());
}

#[test]
fn completing_during_a_pull_stops_it() {
    let (sink, source, receiver) = attached();
    offer(&sink, [1, 2, 3]);
    let mut pulled = Vec::new();
    let count = source.pull(
        &mut |frame| {
            pulled.push(frame.header().id_hash());
            sink.complete();
        },
        &mut |_| false,
        3,
    );
    assert_eq!(count, 1);
    assert_eq!(pulled, [1]);
    assert!(receiver.is_completed());
    assert_eq!(sink.size(), 2);
}

#[test]
fn clearing_cancels_every_queued_frame() {
    let (sink, source, receiver) = attached();
    let fates = Fates::default();
    for id in 0..3 {
        assert!(sink.offer(Tracked::new(id, &fates)).is_ok());
    }
    sink.clear();
    assert_eq!(sink.size(), 0);
    let cancelled = Fate::Failed(FrameError::Cancelled.to_string());
    assert_eq!(
        *fates.lock().unwrap(),
        (0..3).map(|id| (id, cancelled.clone())).collect::<Vec<_>>()
    );
    source.request(3);
    assert!(receiver.received().is_empty());
    assert!(!sink.is_complete());
}

#[test]
fn seeded_offers_keep_their_order() {
    let sink = QueueIngestSink::with_partitions(NonZeroUsize::new(4).unwrap());
    assert_eq!(sink.partitions(), 4);
    let source = sink.delegate();
    let receiver = TestReceiver::new();
    source.clone().add_downstream(receiver.clone());
    for id in 0..32 {
        assert!(sink.offer_seeded(11, frame(id)).is_ok());
    }
    assert_eq!(sink.size(), 32);
    source.request(u64::MAX);
    assert_eq!(receiver.received(), (0..32).collect::<Vec<_>>());
}

#[test]
fn offers_to_a_partition_keep_their_order() {
    let sink = QueueIngestSink::with_partitions(NonZeroUsize::new(3).unwrap());
    let source = sink.delegate();
    let receiver = TestReceiver::new();
    source.clone().add_downstream(receiver.clone());
    for id in 0..12 {
        assert!(sink.offer_to(2, frame(id)).is_ok());
    }
    source.request(u64::MAX);
    assert_eq!(receiver.received(), (0..12).collect::<Vec<_>>());
}

#[test]
fn random_offers_reach_every_partition_and_all_arrive() {
    let sink = QueueIngestSink::with_partitions(NonZeroUsize::new(4).unwrap());
    let source = sink.delegate();
    let receiver = TestReceiver::new();
    source.clone().add_downstream(receiver.clone());
    offer(&sink, 0..256);
    source.request(u64::MAX);
    let mut received = receiver.received();
    received.sort_unstable();
    assert_eq!(received, (0..256).collect::<Vec<_>>());
}

#[test]
#[should_panic(expected = "partition 3 is out of range for 3 partitions")]
fn an_out_of_range_partition_panics() {
    let sink = QueueIngestSink::with_partitions(NonZeroUsize::new(3).unwrap());
    let _ = sink.offer_to(3, frame(0));
}

#[test]
fn a_panicking_receiver_does_not_wedge_the_queue() {
    let fail = Arc::new(AtomicBool::new(true));
    let failing = fail.clone();
    let receiver = Hooked::on_push(move |_| {
        if failing.swap(false, Ordering::AcqRel) {
            panic!("receiver failed");
        }
    });
    let sink = QueueIngestSink::new();
    let source = sink.delegate();
    source.clone().add_downstream(receiver.clone());
    offer(&sink, [1, 2, 3]);
    let request = source.clone();
    assert!(panic::catch_unwind(AssertUnwindSafe(|| request.request(1))).is_err());
    source.request(2);
    assert_eq!(receiver.inner.received(), [1, 2, 3]);
    let panicking = source.clone();
    offer(&sink, [4]);
    assert!(
        panic::catch_unwind(AssertUnwindSafe(|| {
            panicking.pull(&mut |_| panic!("consumer failed"), &mut |_| false, 1)
        }))
        .is_err()
    );
    offer(&sink, [5]);
    source.request(1);
    assert_eq!(receiver.inner.received(), [1, 2, 3, 5]);
}

#[test]
fn concurrent_publishers_and_requesters_deliver_every_frame_once() {
    const PUBLISHERS: u64 = 4;
    const EACH: u64 = 2_000;
    let sink = QueueIngestSink::with_partitions(NonZeroUsize::new(2).unwrap());
    let source = sink.delegate();
    let receiver = TestReceiver::new();
    source.clone().add_downstream(receiver.clone());
    thread::scope(|scope| {
        for publisher in 0..PUBLISHERS {
            let sink = &sink;
            scope.spawn(move || {
                for id in 0..EACH {
                    assert!(
                        sink.offer_seeded(publisher, frame(publisher * EACH + id))
                            .is_ok()
                    );
                }
            });
        }
        for _ in 0..2 {
            let source = source.clone();
            let receiver = &receiver;
            scope.spawn(move || {
                let deadline = Instant::now() + PATIENCE;
                while receiver.received.lock().unwrap().len() < (PUBLISHERS * EACH) as usize
                    && Instant::now() < deadline
                {
                    source.request(64);
                    thread::yield_now();
                }
            });
        }
    });
    let received = receiver.received();
    for publisher in 0..PUBLISHERS {
        let own: Vec<_> = received
            .iter()
            .copied()
            .filter(|id| id / EACH == publisher)
            .collect();
        assert_eq!(
            own,
            (publisher * EACH..(publisher + 1) * EACH).collect::<Vec<_>>()
        );
    }
    assert_eq!(sink.size(), 0);
}

#[test]
fn an_attached_receiver_hears_of_completion_exactly_once() {
    let sink = QueueIngestSink::new();
    let source = sink.delegate();
    let receiver = Arc::new(Hooked::default());
    source.clone().add_downstream(receiver.clone());
    sink.complete();
    source.complete();
    sink.complete();
    assert_eq!(receiver.completions(), 1);
}

#[test]
fn request_without_a_receiver_banks_nothing() {
    let sink = QueueIngestSink::new();
    sink.delegate().request(5);
    assert_eq!(sink.demand(), 0);
    offer(&sink, [1]);
    let receiver = TestReceiver::new();
    sink.delegate().add_downstream(receiver.clone());
    assert!(
        receiver.received().is_empty(),
        "the early request is not remembered"
    );
}

#[test]
fn demand_saturates_and_floors_at_zero() {
    let sink = QueueIngestSink::new();
    let source = &sink.source;
    assert_eq!(source.add_demand(u64::MAX), u64::MAX);
    assert_eq!(source.add_demand(u64::MAX), u64::MAX);
    assert_eq!(source.add_demand(1), u64::MAX);
    assert_eq!(source.take_demand(u64::MAX), 0);
    assert_eq!(source.take_demand(7), 0);
    assert_eq!(source.add_demand(3), 3);
    assert_eq!(source.take_demand(1), 2);
    assert_eq!(sink.demand(), 2);
}

#[test]
fn complete_before_attach_tells_the_late_receiver_once() {
    let sink = QueueIngestSink::new();
    sink.complete();
    let receiver = Arc::new(Hooked::default());
    sink.delegate().add_downstream(receiver.clone());
    sink.complete();
    assert_eq!(receiver.completions(), 1);
}

#[test]
fn a_request_while_a_pull_holds_the_queue_is_served_when_the_pull_releases() {
    use std::sync::Barrier;
    let (sink, source, receiver) = attached();
    offer(&sink, 0..5);
    let inside = Barrier::new(2);
    let release = Barrier::new(2);
    thread::scope(|scope| {
        let puller = {
            let source = source.clone();
            let (inside, release) = (&inside, &release);
            scope.spawn(move || {
                source.pull(
                    &mut |_frame| {
                        inside.wait();
                        release.wait();
                    },
                    &mut |_| false,
                    1,
                )
            })
        };
        inside.wait();
        source.request(4);
        assert!(
            receiver.received().is_empty(),
            "the request did not wait for the pull, and delivered nothing yet"
        );
        assert_eq!(sink.demand(), 4, "the refused request banked its demand");
        release.wait();
        assert_eq!(puller.join().unwrap(), 1);
    });
    assert_eq!(
        receiver.received().len(),
        4,
        "the pull served the request as it let go"
    );
    assert_eq!(sink.demand(), 0);
    assert_eq!(sink.size(), 0);
}

#[test]
fn request_pushes_no_more_than_the_banked_demand() {
    let (sink, source, receiver) = attached();
    offer(&sink, 0..10);
    source.request(3);
    assert_eq!(receiver.received(), [0, 1, 2]);
    assert_eq!(sink.demand(), 0);
    source.request(10);
    assert_eq!(receiver.received().len(), 10);
    assert_eq!(sink.demand(), 3);
}

#[test]
fn a_source_driven_only_by_request_delivers_every_frame() {
    const PUBLISHERS: u64 = 4;
    const EACH: u64 = if cfg!(miri) { 10 } else { 250 };
    let (sink, source, receiver) = attached();
    thread::scope(|scope| {
        for publisher in 0..PUBLISHERS {
            let sink = &sink;
            scope.spawn(move || {
                for id in 0..EACH {
                    assert!(sink.offer(frame(publisher * EACH + id)).is_ok());
                }
            });
        }
        let deadline = Instant::now() + PATIENCE;
        while receiver.received.lock().unwrap().len() < (PUBLISHERS * EACH) as usize
            && Instant::now() < deadline
        {
            source.request(1);
            thread::yield_now();
        }
    });
    let mut received = receiver.received();
    received.sort_unstable();
    assert_eq!(received, (0..PUBLISHERS * EACH).collect::<Vec<_>>());
}

/// An owner that discards what is queued after the source closes, as a killed runner does.
struct Discarding;

impl Owner for Discarding {
    fn complete(&self, source: &QueueSource) {
        source.close();
        source.discard_queued(|frame| frame.do_finally_with_error(FrameError::Cancelled));
    }

    fn drain_finished(&self, source: &QueueSource) {
        if source.is_complete() {
            source.discard_queued(|frame| frame.do_finally_with_error(FrameError::Cancelled));
        }
    }
}

#[test]
fn a_publish_after_close_is_discarded_by_someone() {
    const PUBLISHERS: u64 = 4;
    const EACH: u64 = if cfg!(miri) { 5 } else { 500 };
    for round in 0..if cfg!(miri) { 2 } else { 50 } {
        let fates = Fates::default();
        let source = QueueSource::new(NonZeroUsize::MIN, 64, Ending::Gracefully);
        let owner: Arc<dyn Owner> = Arc::new(Discarding);
        source.set_owner(Arc::downgrade(&owner));
        let receiver = TestReceiver::new();
        source.clone().add_downstream(receiver.clone());
        thread::scope(|scope| {
            for publisher in 0..PUBLISHERS {
                let (source, fates) = (&source, &fates);
                scope.spawn(move || {
                    for id in 0..EACH {
                        let tracked = Tracked::new(publisher * EACH + id, fates);
                        if let Err(refused) = source.offer(tracked) {
                            refused.do_finally_with_error(FrameError::Cancelled);
                        }
                    }
                });
            }
            let puller = {
                let source = source.clone();
                scope.spawn(move || {
                    for _ in 0..EACH {
                        source.pull(
                            &mut |frame| frame.do_finally_with_error(FrameError::Cancelled),
                            &mut |_| false,
                            8,
                        );
                        thread::yield_now();
                    }
                })
            };
            thread::yield_now();
            owner.complete(&source);
            puller.join().unwrap();
        });
        let mut ids: Vec<u64> = fates.lock().unwrap().iter().map(|(id, _)| *id).collect();
        ids.sort_unstable();
        assert_eq!(
            ids,
            (0..PUBLISHERS * EACH).collect::<Vec<_>>(),
            "round {round}: every frame finished exactly once"
        );
        assert_eq!(source.size(), 0, "round {round}: none left queued");
    }
}

#[test]
fn frames_queued_when_the_source_drops_are_finished() {
    let fates = Fates::default();
    let sink = QueueIngestSink::new();
    for id in 0..10 {
        assert!(sink.offer(Tracked::new(id, &fates)).is_ok());
    }
    sink.complete();
    assert!(
        fates.lock().unwrap().is_empty(),
        "a complete sink keeps its queue"
    );
    drop(sink);
    let mut ids: Vec<u64> = fates.lock().unwrap().iter().map(|(id, _)| *id).collect();
    ids.sort_unstable();
    assert_eq!(ids, (0..10).collect::<Vec<_>>());
}

#[test]
fn clear_racing_a_pull_discards_every_frame_once() {
    const FRAMES: u64 = if cfg!(miri) { 30 } else { 1_000 };
    for _ in 0..if cfg!(miri) { 2 } else { 20 } {
        let sink = QueueIngestSink::new();
        let source = sink.delegate();
        let receiver = TestReceiver::new();
        source.clone().add_downstream(receiver.clone());
        let fates = Fates::default();
        for id in 0..FRAMES {
            assert!(sink.offer(Tracked::new(id, &fates)).is_ok());
        }
        let delivered = Fates::default();
        thread::scope(|scope| {
            let puller = {
                let (source, delivered) = (source.clone(), &delivered);
                scope.spawn(move || {
                    while source.pull(
                        &mut |frame| {
                            let id = frame.header().id_hash();
                            delivered.lock().unwrap().push((id, Fate::Finished));
                        },
                        &mut |_| false,
                        1,
                    ) > 0
                    {}
                })
            };
            thread::yield_now();
            sink.clear();
            puller.join().unwrap();
        });
        // A clear that found the pull holding the queue is answered by the pull; one more pull
        // (any call that lets go of the queue) is not needed.
        assert_eq!(sink.size(), 0, "nothing left queued");
        let mut ids: Vec<u64> = fates
            .lock()
            .unwrap()
            .iter()
            .map(|(id, _)| *id)
            .chain(delivered.lock().unwrap().iter().map(|(id, _)| *id))
            .collect();
        ids.sort_unstable();
        assert_eq!(
            ids,
            (0..FRAMES).collect::<Vec<_>>(),
            "every frame delivered or cancelled, exactly once"
        );
    }
}
