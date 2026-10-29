use super::*;
use crate::frames::BenchmarkFrame;
use std::collections::VecDeque;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// A handle that records what it is asked for and can be made contended, empty or broken.
#[derive(Default)]
struct TestHandle {
    sequence: AtomicU64,
    requested: AtomicU64,
    pulled: AtomicU64,
    acquisition_attempts: AtomicU64,
    releases: AtomicU64,
    complete: AtomicBool,
    unavailable: AtomicBool,
    locked: AtomicBool,
    /// How many frames each pull finds; `None` finds the whole demand.
    pull_result: Mutex<Option<u64>>,
    pull_panics: AtomicBool,
}

impl TestHandle {
    fn requested(&self) -> u64 {
        self.requested.load(Ordering::Relaxed)
    }

    fn pulled(&self) -> u64 {
        self.pulled.load(Ordering::Relaxed)
    }

    fn attempts(&self) -> u64 {
        self.acquisition_attempts.load(Ordering::Relaxed)
    }

    fn set_available(&self, available: bool) {
        self.unavailable.store(!available, Ordering::Relaxed);
    }

    fn set_pull_result(&self, result: Option<u64>) {
        *self.pull_result.lock().unwrap() = result;
    }

    fn complete(&self, count: &PaddedAtomicLong) {
        if !self.complete.swap(true, Ordering::Relaxed) {
            count.decrement_and_get();
        }
    }
}

impl UpstreamHandle for TestHandle {
    fn sequence(&self) -> u64 {
        self.sequence.load(Ordering::Relaxed)
    }

    fn pull(
        &self,
        _claim: &mut Claim,
        _consumer: &mut dyn FnMut(Box<dyn Frame>),
        _stop: &mut dyn FnMut(&dyn Frame) -> bool,
        demand: u64,
    ) -> Option<u64> {
        assert!(!self.pull_panics.load(Ordering::Relaxed), "pull failed");
        self.pulled.fetch_add(demand, Ordering::Relaxed);
        let result = *self.pull_result.lock().unwrap();
        Some(result.map_or(demand, |result| result.min(demand)))
    }

    fn request(&self, _claim: &mut Claim, demand: u64) {
        self.requested.fetch_add(demand, Ordering::Relaxed);
    }

    fn is_complete(&self) -> bool {
        self.complete.load(Ordering::Relaxed)
    }

    fn try_acquire(&self) -> Option<Claim> {
        self.acquisition_attempts.fetch_add(1, Ordering::Relaxed);
        if self.unavailable.load(Ordering::Relaxed) {
            return None;
        }
        self.locked.store(true, Ordering::Relaxed);
        Some(Claim::new(0))
    }

    fn release(&self, _claim: Claim) {
        self.locked.store(false, Ordering::Relaxed);
        self.releases.fetch_add(1, Ordering::Relaxed);
    }
}

/// A queued source: pulls hand over frames up to the first one the stop condition matches, and
/// requests route frames on the requesting thread.
#[derive(Default)]
struct SourceHandle {
    frames: Mutex<VecDeque<Box<dyn Frame>>>,
    routed: Mutex<Vec<u64>>,
    lock: AtomicBool,
    complete: AtomicBool,
    pulls: AtomicU64,
    sequence: u64,
    /// Whether routed frames count against the requesting thread's [`FlowContext`] request.
    counts_against_request: bool,
    /// Whether routing goes unreported, as when the frame lands on a receiver that does not
    /// count it.
    silent: bool,
}

impl SourceHandle {
    fn with_frames(frames: impl IntoIterator<Item = Box<dyn Frame>>) -> Arc<Self> {
        Arc::new(Self {
            frames: Mutex::new(frames.into_iter().collect()),
            ..Self::default()
        })
    }

    fn len(&self) -> usize {
        self.frames.lock().unwrap().len()
    }

    fn routed(&self) -> Vec<u64> {
        self.routed.lock().unwrap().clone()
    }
}

impl UpstreamHandle for SourceHandle {
    fn sequence(&self) -> u64 {
        self.sequence
    }

    fn pull(
        &self,
        _claim: &mut Claim,
        consumer: &mut dyn FnMut(Box<dyn Frame>),
        stop: &mut dyn FnMut(&dyn Frame) -> bool,
        demand: u64,
    ) -> Option<u64> {
        self.pulls.fetch_add(1, Ordering::Relaxed);
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
        Some(pulled)
    }

    fn request(&self, _claim: &mut Claim, demand: u64) {
        for _ in 0..demand {
            let Some(frame) = self.frames.lock().unwrap().pop_front() else {
                return;
            };
            self.routed.lock().unwrap().push(frame.header().id_hash());
            if !self.silent {
                note_routed_push();
            }
            if self.counts_against_request {
                FlowContext::with(|context| context.unwrap().add_satisfied_request(1));
            }
        }
    }

    fn is_complete(&self) -> bool {
        self.complete.load(Ordering::Relaxed)
    }

    fn try_acquire(&self) -> Option<Claim> {
        (!self.lock.swap(true, Ordering::Acquire)).then(|| Claim::new(0))
    }

    fn release(&self, _claim: Claim) {
        self.lock.store(false, Ordering::Release);
    }
}

fn ordered(id: u64) -> Box<dyn Frame> {
    Box::new(BenchmarkFrame::new(id, None))
}

fn unordered(id: u64) -> Box<dyn Frame> {
    let mut frame = BenchmarkFrame::new(id, None);
    frame.header_mut().randomize_hash(1);
    Box::new(frame)
}

struct Fixture {
    queue: UpstreamQueue,
    handles: MpscQueue<Entry>,
    count: Arc<PaddedAtomicLong>,
}

impl Fixture {
    fn new() -> Self {
        let handles = MpscQueue::new(64);
        let count = Arc::new(PaddedAtomicLong::new(0));
        let queue = UpstreamQueue::new(Some(0), &handles, count.clone(), 8);
        Self {
            queue,
            handles,
            count,
        }
    }

    /// Queues `handle` and counts it as live.
    fn add<H: UpstreamHandle + 'static>(&self, handle: Arc<H>) -> Arc<H> {
        self.enqueue(handle.clone());
        self.count.increment_and_get();
        handle
    }

    fn add_test(&self) -> Arc<TestHandle> {
        self.add(Arc::new(TestHandle::default()))
    }

    /// Queues `handle` without counting it.
    fn enqueue(&self, handle: Arc<dyn UpstreamHandle>) {
        assert!(self.handles.producer().offer(Entry::new(handle)).is_ok());
    }

    /// Every handle the queue holds, wherever it is held.
    fn held(&self) -> u64 {
        self.handles.size_long()
            + self.queue.shuffle.len() as u64
            + u64::from(self.queue.pending_request.is_some())
    }

    fn pull(&mut self, demand: u64) -> u64 {
        self.queue.pull(&mut |_| {}, &mut |_| false, demand)
    }

    fn pull_until_ordered(&mut self, demand: u64) -> u64 {
        self.queue.pull(
            &mut |_| {},
            &mut |frame| frame.header().is_ordered(),
            demand,
        )
    }
}

fn at(base: Instant, nanos: u64) -> Instant {
    base + Duration::from_nanos(nanos)
}

fn nanos(nanos: u64) -> Duration {
    Duration::from_nanos(nanos)
}

#[test]
fn a_small_request_goes_to_one_handle() {
    let mut fixture = Fixture::new();
    let first = fixture.add_test();
    let second = fixture.add_test();

    fixture.queue.request(32);

    assert_eq!(first.requested(), 32);
    assert_eq!(second.requested(), 0);
}

#[test]
fn a_request_does_not_pull() {
    let mut fixture = Fixture::new();
    let upstream = fixture.add_test();

    fixture.queue.request(64);

    assert_eq!(upstream.requested(), 64);
    assert_eq!(upstream.pulled(), 0);
}

#[test]
fn a_pull_does_not_request() {
    let mut fixture = Fixture::new();
    let upstream = fixture.add_test();

    assert_eq!(fixture.pull(64), 64);

    assert_eq!(upstream.requested(), 0);
    assert_eq!(upstream.pulled(), 64);
}

#[test]
fn zero_demand_does_nothing() {
    let mut fixture = Fixture::new();
    let upstream = fixture.add_test();

    fixture.queue.request(0);
    assert_eq!(fixture.pull(0), 0);

    assert_eq!(upstream.requested(), 0);
    assert_eq!(upstream.pulled(), 0);
    assert_eq!(upstream.attempts(), 0);
}

#[test]
fn cached_and_true_counts_follow_the_live_handle_count() {
    let mut fixture = Fixture::new();
    assert_eq!(fixture.queue.cached_upstream_count(), 0);
    let upstream = fixture.add_test();

    assert_eq!(
        fixture.queue.cached_upstream_count(),
        1,
        "a zero cache refreshes"
    );
    upstream.set_pull_result(Some(0));
    fixture.pull(64);

    assert_eq!(fixture.queue.cached_upstream_count(), 1);
    assert_eq!(fixture.queue.true_upstream_count(), 1);
    assert_eq!(fixture.queue.productive_handle_count(), 0);

    fixture.add_test();
    assert_eq!(
        fixture.queue.cached_upstream_count(),
        1,
        "a nonzero cache is kept"
    );
    assert_eq!(fixture.queue.true_upstream_count(), 2);
}

#[test]
fn new_handles_count_as_productive() {
    let mut fixture = Fixture::new();
    fixture.add_test();
    fixture.add_test();

    assert_eq!(fixture.queue.productive_handle_count(), 2);
}

#[test]
fn productivity_returns_with_useful_work() {
    let mut fixture = Fixture::new();
    let upstream = fixture.add_test();
    upstream.set_pull_result(Some(0));

    assert_eq!(fixture.pull(64), 0);
    assert_eq!(fixture.queue.productive_handle_count(), 0);

    upstream.set_pull_result(None);

    assert_eq!(fixture.pull(64), 64);
    assert_eq!(fixture.queue.productive_handle_count(), 1);
    assert_eq!(fixture.queue.nonproductive_count, 0);
}

#[test]
fn a_request_is_not_evidence_of_an_empty_source() {
    let mut fixture = Fixture::new();
    let upstream = fixture.add_test();

    fixture.queue.request(64);
    assert_eq!(fixture.queue.productive_handle_count(), 1);

    upstream.set_pull_result(Some(0));
    fixture.pull(64);
    fixture.queue.request(64);

    assert_eq!(
        fixture.queue.productive_handle_count(),
        0,
        "a request keeps the worker's last view"
    );
}

#[test]
fn frames_delivered_against_the_request_restore_productivity() {
    FlowContext::initialize();
    let mut fixture = Fixture::new();
    let source = fixture.add(Arc::new(SourceHandle {
        counts_against_request: true,
        silent: true,
        ..SourceHandle::default()
    }));

    fixture.pull(64);
    assert_eq!(fixture.queue.productive_handle_count(), 0);

    source.frames.lock().unwrap().push_back(unordered(1));
    fixture.queue.request(64);

    assert_eq!(source.routed(), [1]);
    assert_eq!(
        FlowContext::with(|context| context.unwrap().satisfied_request()),
        1
    );
    assert_eq!(fixture.queue.productive_handle_count(), 1);
    FlowContext::clear();
}

#[test]
fn frames_routed_during_the_request_restore_productivity() {
    let mut fixture = Fixture::new();
    let source = fixture.add(Arc::new(SourceHandle::default()));

    fixture.pull(64);
    assert_eq!(fixture.queue.productive_handle_count(), 0);

    source.frames.lock().unwrap().push_back(unordered(1));
    fixture.queue.request(64);

    assert_eq!(source.routed(), [1]);
    assert_eq!(fixture.queue.productive_handle_count(), 1);
}

#[test]
fn a_request_without_evidence_keeps_the_handle_unproductive() {
    let mut fixture = Fixture::new();
    let source = fixture.add(Arc::new(SourceHandle {
        silent: true,
        ..SourceHandle::default()
    }));

    fixture.pull(64);
    source.frames.lock().unwrap().push_back(unordered(1));
    fixture.queue.request(64);

    assert_eq!(source.routed(), [1]);
    assert_eq!(fixture.queue.productive_handle_count(), 0);
}

#[test]
fn pulls_count_against_the_flow_context() {
    FlowContext::initialize();
    let mut fixture = Fixture::new();
    fixture.add_test();

    fixture.pull(5);

    assert_eq!(
        FlowContext::with(|context| context.unwrap().satisfied_pull()),
        5
    );
    FlowContext::clear();
}

#[test]
fn an_ordered_frame_progresses_when_pulls_and_requests_alternate_with_an_idle_source() {
    let mut fixture = Fixture::new();
    let ready = fixture.add(SourceHandle::with_frames([ordered(7)]));
    fixture.add(SourceHandle::with_frames([]));

    for _ in 0..4 {
        if !ready.routed().is_empty() {
            break;
        }
        let pulled = fixture.queue.pull(
            &mut |_| panic!("an ordered frame must be routed, not pulled"),
            &mut |frame| frame.header().is_ordered(),
            1,
        );
        assert_eq!(pulled, 0);
        fixture.queue.request(1);
    }

    assert_eq!(ready.routed(), [7]);
    assert_eq!(ready.len(), 0);
    assert_eq!(
        fixture.held(),
        2,
        "holding a handle back must not lose or copy it"
    );
}

#[test]
fn a_held_back_handle_honours_demand_and_its_lock() {
    let mut fixture = Fixture::new();
    let idle = fixture.add_test();
    let ready = fixture.add(SourceHandle::with_frames([ordered(1), ordered(2)]));

    // The first pull visits the idle handle; the second stops at the ordered frame.
    fixture.pull_until_ordered(1);
    fixture.pull_until_ordered(1);
    fixture.queue.request(0);
    assert_eq!(ready.len(), 2);

    let claim = ready.try_acquire().expect("the handle is free");
    fixture.queue.request(1);
    assert_eq!(ready.len(), 2, "another worker holds the source");
    assert_eq!(
        idle.requested(),
        1,
        "a contended handle lets the next one progress"
    );
    assert_eq!(fixture.held(), 2);
    ready.release(claim);

    fixture.pull_until_ordered(1);
    fixture.queue.request(1);
    assert_eq!(ready.len(), 1, "the held-back request keeps to its demand");
    assert_eq!(fixture.held(), 2);
}

#[test]
fn a_completed_held_back_handle_lets_demand_reach_another() {
    let mut fixture = Fixture::new();
    let ready = fixture.add(SourceHandle::with_frames([ordered(1)]));
    let live = fixture.add_test();
    fixture.pull_until_ordered(1);
    ready.complete.store(true, Ordering::Relaxed);
    fixture.count.decrement_and_get();

    fixture.queue.request(1);

    assert_eq!(live.requested(), 1);
    assert_eq!(fixture.queue.productive_handle_count(), 1);
    assert_eq!(fixture.held(), 1);
}

#[test]
fn a_trial_reset_forgets_the_held_back_handle() {
    let mut fixture = Fixture::new();
    let first = fixture.add(Arc::new(TestHandle::default()));
    let ready = fixture.add(Arc::new(SourceHandle {
        sequence: 1,
        ..SourceHandle::default()
    }));
    ready.frames.lock().unwrap().push_back(ordered(1));
    fixture.pull_until_ordered(1);
    fixture.pull_until_ordered(1);

    fixture.queue.reset_for_next_trial();
    fixture.queue.request(1);

    assert_eq!(
        first.requested(),
        1,
        "registration order comes back without the hint"
    );
    assert_eq!(ready.len(), 1);
    assert_eq!(fixture.held(), 2);
}

#[test]
fn a_trial_reset_restores_registration_order_and_productivity() {
    let mut fixture = Fixture::new();
    let handles: Vec<_> = [3, 1, 2]
        .into_iter()
        .map(|sequence| {
            let handle = fixture.add_test();
            handle.sequence.store(sequence, Ordering::Relaxed);
            handle.set_pull_result(Some(0));
            handle
        })
        .collect();
    fixture.pull(3 * 2048);
    assert_eq!(fixture.queue.productive_handle_count(), 0);
    let completed = fixture.add_test();
    completed.complete(&fixture.count);

    fixture.queue.reset_for_next_trial();

    assert_eq!(fixture.held(), 3, "the completed handle is gone");
    let mut order = Vec::new();
    while let Some(entry) = fixture.queue.handles.poll() {
        order.push(entry.handle.sequence());
        fixture.queue.offer(entry);
        if order.len() == 3 {
            break;
        }
    }
    assert_eq!(order, [1, 2, 3]);
    assert_eq!(fixture.queue.productive_handle_count(), 3);
    assert_eq!(fixture.queue.acquire_contention(), None);

    fixture.pull(3 * 2048);
    assert_eq!(
        fixture.queue.productive_handle_count(),
        0,
        "empty pulls after a reset count again"
    );
    drop(handles);
}

#[test]
fn a_contended_handle_is_kept_for_a_later_pull() {
    let mut fixture = Fixture::new();
    let upstream = fixture.add_test();
    upstream.set_available(false);

    assert_eq!(fixture.pull(64), 0);
    assert_eq!(fixture.held(), 1);
    assert_eq!(upstream.attempts(), 1);
    assert_eq!(
        fixture.queue.acquire_contention(),
        Some(ACQUIRE_CONTENTION_SCALE)
    );
    assert_eq!(fixture.queue.contention_evidence_count(), 1);

    upstream.set_available(true);

    assert_eq!(fixture.pull(64), 64);
    assert_eq!(fixture.held(), 1);
    assert_eq!(upstream.attempts(), 2);
    assert_eq!(fixture.queue.contention_evidence_count(), 2);
}

#[test]
fn a_panicking_handle_is_released_and_kept() {
    let mut fixture = Fixture::new();
    let upstream = fixture.add_test();
    upstream.set_pull_result(Some(0));
    fixture.pull(1);
    assert_eq!(fixture.queue.nonproductive_count, 1);
    upstream.pull_panics.store(true, Ordering::Relaxed);

    let outcome = catch_unwind(AssertUnwindSafe(|| fixture.pull(1)));

    assert!(outcome.is_err());
    assert!(!upstream.locked.load(Ordering::Relaxed));
    assert_eq!(upstream.releases.load(Ordering::Relaxed), 2);
    assert_eq!(
        fixture.queue.nonproductive_count, 1,
        "a panic is no evidence"
    );

    upstream.pull_panics.store(false, Ordering::Relaxed);
    upstream.set_pull_result(None);

    assert_eq!(fixture.pull(1), 1);
    assert_eq!(upstream.attempts(), 3);
    assert_eq!(upstream.releases.load(Ordering::Relaxed), 3);
    assert_eq!(fixture.held(), 1);
    assert_eq!(fixture.queue.nonproductive_count, 0);
}

#[test]
fn an_empty_service_uses_up_demand_but_a_failed_acquisition_does_not() {
    let mut fixture = Fixture::new();
    let empty = fixture.add_test();
    empty.set_pull_result(Some(0));
    let productive = fixture.add_test();

    assert_eq!(fixture.pull(1), 0);
    assert_eq!(empty.attempts(), 1);
    assert_eq!(productive.attempts(), 0);

    let mut retry = Fixture::new();
    let unavailable = retry.add_test();
    unavailable.set_available(false);
    let fallback = retry.add_test();

    assert_eq!(retry.pull(1), 1);
    assert_eq!(unavailable.attempts(), 1);
    assert_eq!(fallback.attempts(), 1);
}

#[test]
fn successful_acquisitions_record_zero_contention() {
    let mut fixture = Fixture::new();
    fixture.add_test();

    fixture.pull(64);

    assert_eq!(fixture.queue.acquire_contention(), Some(0));
    assert_eq!(fixture.queue.normalized_acquire_contention(), Some(0.0));
}

#[test]
fn a_mixed_cycle_records_one_fraction() {
    let mut fixture = Fixture::new();
    let unavailable = fixture.add_test();
    unavailable.set_available(false);
    fixture.add_test();

    fixture.pull(64);

    assert_eq!(fixture.queue.acquire_contention(), Some(500_000));
    assert_eq!(fixture.queue.normalized_acquire_contention(), Some(0.5));
    assert_eq!(unavailable.attempts(), 1);
}

#[test]
fn contention_decays_exponentially() {
    let full = ACQUIRE_CONTENTION_SCALE;
    let half_life = nanos(1_000);
    let close = |actual: u64, expected: u64, tolerance: u64| {
        assert!(
            actual.abs_diff(expected) <= tolerance,
            "{actual} vs {expected}"
        );
    };
    assert_eq!(decay_contention(full, Duration::ZERO, half_life), full);
    close(decay_contention(full, nanos(500), half_life), 707_107, 2);
    close(decay_contention(full, nanos(1_000), half_life), 500_000, 1);
    close(decay_contention(full, nanos(2_000), half_life), 250_000, 1);
    close(decay_contention(full, nanos(3_000), half_life), 125_000, 1);
    assert_eq!(decay_contention(full, nanos(100_000), half_life), 0);
    assert_eq!(decay_contention(0, nanos(10), half_life), 0);
}

#[test]
fn decay_clamps_the_estimate_to_the_scale() {
    assert_eq!(
        decay_contention(u64::MAX, Duration::ZERO, nanos(1_000)),
        ACQUIRE_CONTENTION_SCALE
    );
    assert_eq!(
        decay_contention(u64::MAX, nanos(1_000), nanos(1_000)),
        500_000
    );
}

#[test]
#[should_panic(expected = "half-life must be positive")]
fn decay_rejects_a_zero_half_life() {
    decay_contention(1, nanos(1), Duration::ZERO);
}

#[test]
fn effective_contention_ages_evidence_without_changing_the_estimate() {
    let base = Instant::now();
    let mut fixture = Fixture::new();
    fixture.add_test().set_available(false);
    fixture.pull(64);

    assert_eq!(
        fixture
            .queue
            .effective_contention(at(base, 10_000), nanos(1_000)),
        1_000_000
    );
    assert_eq!(
        fixture.queue.last_contention_evidence(),
        Some(at(base, 10_000))
    );
    let decayed = fixture
        .queue
        .effective_contention(at(base, 11_000), nanos(1_000));
    assert!(decayed.abs_diff(500_000) <= 1, "{decayed}");
    assert_eq!(fixture.queue.acquire_contention(), Some(1_000_000));
    assert_eq!(fixture.queue.contention_evidence_count(), 1);
    assert_eq!(
        fixture.queue.last_contention_evidence(),
        Some(at(base, 10_000))
    );
}

#[test]
fn new_evidence_restarts_the_decay() {
    let base = Instant::now();
    let mut fixture = Fixture::new();
    fixture.add_test().set_available(false);
    fixture.pull(64);
    fixture
        .queue
        .effective_contention(at(base, 10_000), nanos(1_000));

    fixture.pull(64);

    assert_eq!(
        fixture
            .queue
            .effective_contention(at(base, 11_000), nanos(1_000)),
        1_000_000
    );
    assert_eq!(fixture.queue.contention_evidence_count(), 2);
    assert_eq!(
        fixture.queue.last_contention_evidence(),
        Some(at(base, 11_000))
    );
}

#[test]
fn adaptive_contention_with_a_constant_half_life_matches_the_evidence_age() {
    let base = Instant::now();
    let mut fixture = Fixture::new();
    fixture.add_test().set_available(false);
    fixture.pull(64);
    assert_eq!(
        fixture
            .queue
            .adaptive_contention(at(base, 10_000), nanos(1_000)),
        1_000_000
    );
    for elapsed in [1, 50, 317, 500, 1_000, 7_777, 100_000] {
        assert_eq!(
            fixture
                .queue
                .adaptive_contention(at(base, 10_000 + elapsed), nanos(1_000)),
            decay_contention(1_000_000, nanos(elapsed), nanos(1_000)),
            "after {elapsed}ns"
        );
    }
    assert_eq!(
        fixture.queue.last_contention_evidence(),
        Some(at(base, 10_000))
    );
    assert_eq!(fixture.queue.acquire_contention(), Some(1_000_000));
}

#[test]
fn adaptive_contention_ignores_time_running_backwards() {
    let base = Instant::now();
    let mut fixture = Fixture::new();
    fixture.add_test().set_available(false);
    fixture.pull(64);
    fixture
        .queue
        .adaptive_contention(at(base, 10_000), nanos(1_000));

    assert_eq!(
        fixture
            .queue
            .adaptive_contention(at(base, 9_000), nanos(1_000)),
        1_000_000
    );
    assert_eq!(
        fixture
            .queue
            .adaptive_contention(at(base, 11_000), nanos(1_000)),
        500_000
    );
}

#[test]
fn half_life_changes_apply_to_later_time_and_fresh_evidence_restarts_the_decay() {
    let base = Instant::now();
    let initial = nanos(1_000);
    let mut fixture = Fixture::new();
    fixture.add_test().set_available(false);
    fixture.pull(64);
    let queue = &mut fixture.queue;
    queue.adaptive_contention(at(base, 10_000), initial);
    queue.install_contention_half_life(at(base, 11_000), nanos(2_000), initial);
    assert_eq!(
        queue.adaptive_contention(at(base, 11_000), initial),
        500_000
    );
    assert_eq!(
        queue.adaptive_contention(at(base, 13_000), initial),
        250_000
    );
    queue.install_contention_half_life(at(base, 13_000), nanos(500), initial);
    assert_eq!(
        queue.adaptive_contention(at(base, 13_500), initial),
        125_000
    );

    fixture.pull(64);
    let queue = &mut fixture.queue;
    assert_eq!(
        queue.adaptive_contention(at(base, 14_000), initial),
        1_000_000
    );
    assert_eq!(
        queue.adaptive_contention(at(base, 14_500), initial),
        500_000
    );

    queue.reset_acquire_contention();
    assert_eq!(queue.adaptive_contention(at(base, 20_000), initial), 0);
    fixture.pull(64);
    let queue = &mut fixture.queue;
    assert_eq!(
        queue.adaptive_contention(at(base, 20_000), initial),
        1_000_000
    );
    assert_eq!(
        queue.adaptive_contention(at(base, 21_000), initial),
        500_000,
        "a reset restores the initial half-life"
    );
}

#[test]
fn effective_and_adaptive_contention_agree_on_when_evidence_arrived() {
    let base = Instant::now();
    let half_life = nanos(1_000);
    let mut fixture = Fixture::new();
    fixture.add_test().set_available(false);
    fixture.pull(64);

    fixture
        .queue
        .effective_contention(at(base, 10_000), half_life);

    assert_eq!(
        fixture
            .queue
            .adaptive_contention(at(base, 11_000), half_life),
        500_000
    );
    assert_eq!(
        fixture
            .queue
            .effective_contention(at(base, 12_000), half_life),
        fixture
            .queue
            .adaptive_contention(at(base, 12_000), half_life)
    );
}

#[test]
fn contention_is_zero_without_evidence() {
    let mut fixture = Fixture::new();
    let now = Instant::now();
    assert_eq!(fixture.queue.effective_contention(now, nanos(1_000)), 0);
    assert_eq!(fixture.queue.adaptive_contention(now, nanos(1_000)), 0);
    assert_eq!(fixture.queue.last_contention_evidence(), None);
}

#[test]
#[should_panic(expected = "half-life must be positive")]
fn effective_contention_rejects_a_zero_half_life() {
    Fixture::new()
        .queue
        .effective_contention(Instant::now(), Duration::ZERO);
}

#[test]
#[should_panic(expected = "half-life must be positive")]
fn installing_a_zero_half_life_is_rejected() {
    Fixture::new()
        .queue
        .install_contention_half_life(Instant::now(), Duration::ZERO, nanos(1));
}

#[test]
fn diagnostics_are_tracked_only_when_enabled() {
    let mut fixture = Fixture::new();
    let unavailable = fixture.add_test();
    unavailable.set_available(false);
    let available = fixture.add_test();

    fixture.pull(64);

    assert_eq!(fixture.queue.contention_observation_count(), 0);
    assert_eq!(fixture.queue.last_raw_contention(), None);
    assert_eq!(fixture.queue.last_contention_observation(), None);
    assert_eq!(fixture.queue.total_acquisition_attempts(), 0);

    // The shuffle may have put the available handle first; restore the contended order.
    while fixture.queue.handles.poll().is_some() {}
    fixture.enqueue(unavailable);
    fixture.enqueue(available);
    fixture.queue.set_acquire_diagnostics_enabled(true);
    let before = Instant::now();
    fixture.pull(64);

    assert_eq!(fixture.queue.contention_observation_count(), 1);
    assert_eq!(fixture.queue.last_raw_contention(), Some(500_000));
    assert!(fixture.queue.last_contention_observation().unwrap() >= before);
    assert_eq!(fixture.queue.successful_acquisition_count(), 1);
    assert_eq!(fixture.queue.failed_acquisition_count(), 1);
    assert_eq!(fixture.queue.total_acquisition_attempts(), 2);

    fixture.queue.reset_acquire_contention();

    assert_eq!(fixture.queue.contention_observation_count(), 0);
    assert_eq!(fixture.queue.last_raw_contention(), None);
    assert_eq!(fixture.queue.last_contention_observation(), None);
    assert_eq!(fixture.queue.successful_acquisition_count(), 0);
    assert_eq!(fixture.queue.failed_acquisition_count(), 0);
    assert_eq!(fixture.queue.total_acquisition_attempts(), 0);
}

#[test]
fn contention_needs_an_acquisition_attempt() {
    let mut fixture = Fixture::new();
    assert_eq!(fixture.queue.acquire_contention(), None);
    assert_eq!(fixture.queue.normalized_acquire_contention(), None);
    assert_eq!(fixture.queue.contention_evidence_count(), 0);

    fixture.pull(0);
    assert_eq!(fixture.queue.acquire_contention(), None);

    fixture.add_test();
    fixture.pull(64);
    fixture.pull(0);
    assert_eq!(fixture.queue.acquire_contention(), Some(0));
    assert_eq!(fixture.queue.contention_evidence_count(), 1);
}

#[test]
fn completed_handles_are_not_acquisition_misses() {
    let mut fixture = Fixture::new();
    let completed = fixture.add_test();
    completed.complete.store(true, Ordering::Relaxed);

    fixture.pull(64);

    assert_eq!(completed.attempts(), 0);
    assert_eq!(fixture.queue.acquire_contention(), None);
    assert_eq!(fixture.held(), 0, "a completed handle leaves the queue");
}

#[test]
fn resetting_contention_clears_it() {
    let mut fixture = Fixture::new();
    fixture.add_test().set_available(false);
    fixture.pull(64);

    fixture.queue.reset_acquire_contention();

    assert_eq!(fixture.queue.acquire_contention(), None);
    assert_eq!(fixture.queue.contention_evidence_count(), 0);
    assert_eq!(fixture.queue.normalized_acquire_contention(), None);
}

#[test]
fn scaling_is_exact_at_realistic_and_extreme_counts() {
    assert_eq!(scale_acquire_contention(0, 32), 0);
    assert_eq!(scale_acquire_contention(32, 32), 1_000_000);
    assert_eq!(scale_acquire_contention(1, 3), 333_333);
    assert_eq!(scale_acquire_contention(2, 3), 666_666);
    assert_eq!(scale_acquire_contention(u64::MAX - 1, u64::MAX), 999_999);
    assert_eq!(scale_acquire_contention(u64::MAX, u64::MAX), 1_000_000);
}

#[test]
#[should_panic(expected = "acquisition counts")]
fn scaling_rejects_zero_attempts() {
    scale_acquire_contention(0, 0);
}

#[test]
#[should_panic(expected = "acquisition counts")]
fn scaling_rejects_more_failures_than_attempts() {
    scale_acquire_contention(2, 1);
}

#[test]
fn the_default_buckets_split_demand_in_chunks_of_2048() {
    let mut fixture = Fixture::new();
    fixture.queue.cached_up_count = 4;

    assert_eq!(fixture.queue.pull_buckets(2_048), 2_048);
    assert_eq!(fixture.queue.pull_buckets(2_049), 2_049);
    assert_eq!(fixture.queue.pull_buckets(4_095), 4_095);
    assert_eq!(fixture.queue.pull_buckets(4_096), 2_048);
    assert_eq!(
        fixture.queue.pull_buckets(1_000_000),
        250_000,
        "at most one bucket per handle"
    );
}

#[test]
fn floor_and_ceil_treatments_do_not_overflow() {
    let mut fixture = Fixture::new();
    fixture.queue.cached_up_count = 4;
    let target = |target| NonZeroU64::new(target).unwrap();

    fixture
        .queue
        .set_pull_bucket_treatment(target(2_048), PullBucketDivisionMode::Ceil);
    assert_eq!(fixture.queue.pull_buckets(2_049), 1_025);
    assert_eq!(fixture.queue.pull_buckets(3_000), 1_500);

    fixture
        .queue
        .set_pull_bucket_treatment(target(512), PullBucketDivisionMode::Floor);
    assert_eq!(fixture.queue.pull_buckets(4_096), 1_024);

    fixture
        .queue
        .set_pull_bucket_treatment(target(1), PullBucketDivisionMode::Ceil);
    assert_eq!(
        fixture.queue.pull_buckets(i64::MAX as u64),
        2_305_843_009_213_693_952
    );
    assert_eq!(fixture.queue.pull_buckets(u64::MAX), 1 + (u64::MAX - 1) / 4);
}

#[test]
fn a_single_handle_takes_the_whole_demand() {
    let mut fixture = Fixture::new();
    fixture.queue.cached_up_count = 1;
    assert_eq!(fixture.queue.pull_buckets(1_000_000), 1_000_000);
}

#[test]
fn unavailable_handles_get_one_attempt_per_pull_and_are_kept() {
    let mut fixture = Fixture::new();
    let first = fixture.add_test();
    let second = fixture.add_test();
    first.set_available(false);
    second.set_available(false);

    assert_eq!(fixture.pull(64), 0);

    assert_eq!(fixture.held(), 2);
    assert_eq!(first.attempts(), 1);
    assert_eq!(second.attempts(), 1);
}

#[test]
fn worker_queues_get_independent_shuffle_seeds() {
    let seeds: Vec<u64> = (0..8)
        .map(|cpu| {
            let handles = MpscQueue::new(64);
            UpstreamQueue::new(Some(cpu), &handles, Arc::new(PaddedAtomicLong::new(0)), 8).seed
        })
        .collect();
    for (i, seed) in seeds.iter().enumerate() {
        assert!(
            !seeds[i + 1..].contains(seed),
            "worker shuffle seeds must differ"
        );
    }
}

#[test]
fn the_shuffle_advances_when_several_serviced_handles_return() {
    let mut fixture = Fixture::new();
    let first = fixture.add_test();
    let second = fixture.add_test();
    let seed = fixture.queue.seed;

    fixture.pull(4_096);

    assert_ne!(fixture.queue.seed, seed);
    assert_eq!(first.attempts(), 1);
    assert_eq!(second.attempts(), 1);
    assert_eq!(fixture.held(), 2);
}

#[test]
fn the_shuffle_advances_when_several_contended_handles_return() {
    let mut fixture = Fixture::new();
    let first = fixture.add_test();
    let second = fixture.add_test();
    first.set_available(false);
    second.set_available(false);
    let seed = fixture.queue.seed;

    fixture.pull(64);

    assert_ne!(fixture.queue.seed, seed);
    assert_eq!(first.attempts(), 1);
    assert_eq!(second.attempts(), 1);
    assert_eq!(fixture.held(), 2);
}

#[test]
fn a_full_shuffle_window_flushes_before_taking_more() {
    let handles = MpscQueue::new(64);
    let count = Arc::new(PaddedAtomicLong::new(0));
    let mut queue = UpstreamQueue::new(None, &handles, count.clone(), 2);
    let added: Vec<Arc<TestHandle>> = (0..5)
        .map(|_| {
            let handle = Arc::new(TestHandle::default());
            handle.set_available(false);
            assert!(handles.producer().offer(Entry::new(handle.clone())).is_ok());
            count.increment_and_get();
            handle
        })
        .collect();

    queue.pull(&mut |_| {}, &mut |_| false, 64);

    assert!(queue.shuffle.is_empty());
    assert_eq!(handles.size_long(), 5);
    assert!(added.iter().all(|handle| handle.attempts() == 1));
}

#[test]
fn completed_handles_leave_the_queue() {
    let mut fixture = Fixture::new();
    let completed = fixture.add_test();
    let live = fixture.add_test();
    completed.complete(&fixture.count);

    fixture.queue.request(10);

    assert_eq!(fixture.queue.true_upstream_count(), 1);
    assert_eq!(live.requested(), 10);
    assert_eq!(completed.attempts(), 0);
    assert_eq!(fixture.held(), 1);
}

#[test]
fn a_completed_unproductive_handle_is_reconciled() {
    let mut fixture = Fixture::new();
    fixture.add_test();
    let nonproductive = fixture.add_test();
    nonproductive.set_pull_result(Some(0));
    fixture.pull(4_096);
    assert_eq!(fixture.queue.productive_handle_count(), 1);

    nonproductive.complete(&fixture.count);

    assert_eq!(fixture.queue.productive_handle_count(), 1);
    assert_eq!(fixture.queue.true_upstream_count(), 1);
    assert_eq!(fixture.queue.nonproductive_count, 0);
    assert_eq!(fixture.held(), 1);
}

#[test]
fn a_completed_held_back_handle_is_reconciled() {
    let mut fixture = Fixture::new();
    let ready = fixture.add(SourceHandle::with_frames([]));
    fixture.pull(1);
    assert_eq!(fixture.queue.nonproductive_count, 1);
    ready.frames.lock().unwrap().push_back(ordered(1));
    fixture.pull_until_ordered(1);
    assert!(fixture.queue.pending_request.is_some());

    ready.complete.store(true, Ordering::Relaxed);
    fixture.count.decrement_and_get();

    assert_eq!(fixture.queue.productive_handle_count(), 0);
    assert_eq!(fixture.queue.nonproductive_count, 0);
    assert_eq!(fixture.held(), 0);
}

#[test]
fn a_stopped_pull_keeps_the_worker_view() {
    FlowContext::initialize();
    let mut fixture = Fixture::new();
    let source = fixture.add(SourceHandle::with_frames([unordered(1)]));

    assert_eq!(fixture.queue.pull(&mut |_| {}, &mut |_| true, 64), 0);

    assert_eq!(
        FlowContext::with(|context| context.unwrap().satisfied_pull()),
        0
    );
    assert_eq!(source.len(), 1);
    assert_eq!(fixture.queue.productive_handle_count(), 1);
    FlowContext::clear();
}

#[test]
fn a_handle_that_did_not_look_keeps_the_worker_view() {
    struct Closed;
    impl UpstreamHandle for Closed {
        fn sequence(&self) -> u64 {
            0
        }
        fn pull(
            &self,
            _claim: &mut Claim,
            _consumer: &mut dyn FnMut(Box<dyn Frame>),
            _stop: &mut dyn FnMut(&dyn Frame) -> bool,
            _demand: u64,
        ) -> Option<u64> {
            None
        }
        fn request(&self, _claim: &mut Claim, _demand: u64) {}
        fn is_complete(&self) -> bool {
            false
        }
    }

    let mut fixture = Fixture::new();
    fixture.add(Arc::new(Closed));

    assert_eq!(fixture.pull(64), 0);
    assert_eq!(fixture.queue.productive_handle_count(), 1);
}

#[test]
fn an_empty_source_is_unproductive_beside_a_productive_one() {
    let mut fixture = Fixture::new();
    let empty = fixture.add(SourceHandle::with_frames([]));
    let productive = fixture.add(SourceHandle::with_frames([unordered(1)]));

    let mut pulled = Vec::new();
    let count = fixture.queue.pull(
        &mut |frame| pulled.push(frame.header().id_hash()),
        &mut |_| false,
        4_096,
    );

    assert_eq!(count, 1);
    assert_eq!(pulled, [1]);
    assert_eq!(fixture.queue.true_upstream_count(), 2);
    assert_eq!(fixture.queue.productive_handle_count(), 1);
    assert_eq!(empty.len(), 0);
    assert_eq!(productive.len(), 0);
}

#[test]
fn a_completed_unproductive_handle_met_by_a_pull_is_reconciled() {
    let mut fixture = Fixture::new();
    let nonproductive = fixture.add_test();
    nonproductive.set_pull_result(Some(0));
    fixture.pull(1);
    assert_eq!(fixture.queue.nonproductive_count, 1);
    let live = fixture.add_test();

    nonproductive.complete(&fixture.count);
    assert_eq!(fixture.pull(2), 2);

    assert_eq!(fixture.queue.nonproductive_count, 0);
    assert_eq!(live.attempts(), 1);
    assert_eq!(fixture.held(), 1);
}

#[test]
fn a_pull_skips_the_held_back_handle() {
    let mut fixture = Fixture::new();
    let ready = fixture.add(SourceHandle::with_frames([ordered(1)]));
    let other = fixture.add_test();
    fixture.pull_until_ordered(1);
    assert!(fixture.queue.pending_request.is_some());
    let pulls = ready.pulls.load(Ordering::Relaxed);

    assert_eq!(fixture.pull_until_ordered(1), 1);

    assert_eq!(
        ready.pulls.load(Ordering::Relaxed),
        pulls,
        "only a request routes the frame"
    );
    assert_eq!(other.pulled(), 1);
    assert_eq!(fixture.held(), 2);
}

#[test]
fn holding_back_a_second_handle_returns_the_first() {
    let mut fixture = Fixture::new();
    let first = fixture.add(SourceHandle::with_frames([ordered(1)]));
    let second = fixture.add(SourceHandle::with_frames([ordered(2)]));

    fixture.pull_until_ordered(1);
    fixture.pull_until_ordered(1);

    assert_eq!(fixture.held(), 2);
    fixture.queue.request(2);
    fixture.queue.request(2);
    assert_eq!(first.routed(), [1]);
    assert_eq!(second.routed(), [2]);
}

#[test]
fn a_held_back_handle_counts_when_reconciling_completed_handles() {
    let mut fixture = Fixture::new();
    let empty = fixture.add_test();
    empty.set_pull_result(Some(0));
    fixture.pull(1);
    let ready = fixture.add(SourceHandle::with_frames([ordered(1)]));
    fixture.pull_until_ordered(2 * 2048);
    assert!(fixture.queue.pending_request.is_some());
    assert_eq!(ready.len(), 1);

    empty.complete(&fixture.count);

    assert_eq!(fixture.queue.productive_handle_count(), 1);
    assert_eq!(fixture.queue.nonproductive_count, 0);
    assert_eq!(fixture.held(), 1);
}
