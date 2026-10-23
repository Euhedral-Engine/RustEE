use super::*;
use crate::config::TimingProvider;
use crate::flow_control::{UpstreamHandle, UpstreamRegistry};
use crate::frames::Frame;
use euhedral_hardware_utils::util::CpuSet;

fn nanos(nanos: u64) -> Duration {
    Duration::from_nanos(nanos)
}

/// A tree taking bodies of 1000 ns as expensive and running bodies of up to 500 ns direct.
fn tree_with(policy: IdlePolicy) -> (DecisionTree, Arc<BodyCost>) {
    let body_cost = Arc::new(BodyCost::new());
    let tree = DecisionTree::with_thresholds(policy, body_cost.clone(), nanos(1000), nanos(500));
    (tree, body_cost)
}

fn tree() -> (DecisionTree, Arc<BodyCost>) {
    tree_with(IdlePolicy::default())
}

fn record_all(body_cost: &BodyCost, samples: impl IntoIterator<Item = u64>) {
    for sample in samples {
        body_cost.record(nanos(sample));
    }
}

#[test]
fn batches_follow_the_work_target_by_halvings_and_doublings() {
    // Traced from the upstream policy.
    let (mut tree, body_cost) = tree();
    assert_eq!(tree.complete_batch(4096), 2);
    tree.record_execution(nanos(1000), 1);
    let batches: Vec<u64> = (0..7).map(|_| tree.complete_batch(4096)).collect();
    assert_eq!(batches, [4, 8, 16, 32, 64, 128, 128]);

    tree.record_execution(nanos(100_000), 10);
    assert_eq!(tree.service_time_ns(), 2125.0);
    assert_eq!(tree.complete_batch(4096), 64);

    record_all(&body_cost, 600..632);
    assert_eq!(body_cost.estimate_ns(), 601.0);
    assert_eq!(tree.select_path(0, 1, 2, 0), ExecutionPath::Staged);
    assert_eq!(tree.complete_batch(4096), 128);
    assert_eq!(tree.complete_batch(100), 100);
    assert_eq!(tree.complete_batch(1), 2);
    assert_eq!(tree.batch_size(), 2);
}

#[test]
fn a_huge_cap_or_a_tiny_service_time_never_overflows() {
    let (mut tree, _) = tree();
    tree.record_execution(nanos(1), 1_000_000);
    let mut batch = 0;
    for _ in 0..200 {
        batch = tree.complete_batch(u64::MAX);
    }
    // 250 µs over at least 1 ns per frame.
    assert_eq!(batch, 1 << 17);
}

#[test]
fn slow_frames_keep_the_batch_at_two() {
    let (mut tree, _) = tree();
    tree.record_execution(nanos(1_000_000), 1);
    assert_eq!(tree.complete_batch(4096), 2);
    tree.complete_batch(0);
    assert_eq!(tree.batch_size(), 2);
}

#[test]
fn an_odd_batch_halves_rounding_up() {
    let (mut tree, _) = tree();
    tree.record_execution(nanos(1), 1);
    for _ in 0..10 {
        tree.complete_batch(4096);
    }
    assert_eq!(tree.complete_batch(101), 101);
    tree.record_execution(nanos(100_000_000), 1);
    assert_eq!(tree.complete_batch(4096), 51);
}

#[test]
fn execution_samples_without_time_or_frames_are_ignored() {
    let (mut tree, _) = tree();
    tree.record_execution(Duration::ZERO, 10);
    tree.record_execution(nanos(10), 0);
    assert_eq!(tree.service_time_ns(), 0.0);
    tree.record_execution(nanos(800), 2);
    assert_eq!(tree.service_time_ns(), 400.0);
    tree.record_execution(nanos(1200), 1);
    assert_eq!(tree.service_time_ns(), 500.0);
}

#[test]
fn expensive_windows_need_confirming_and_cheap_ones_do_not() {
    // Traced from the upstream policy.
    let (_, body_cost) = tree();
    assert!(!body_cost.has_history());
    record_all(&body_cost, (0..32).map(|i| 100 + 10 * ((i * 7) % 32)));
    assert!(body_cost.has_history());
    assert_eq!(body_cost.estimate_ns(), 110.0);
    record_all(&body_cost, 2000..2032);
    assert_eq!(body_cost.estimate_ns(), 110.0);
    record_all(&body_cost, 3000..3031);
    assert_eq!(body_cost.estimate_ns(), 110.0);
    body_cost.record(nanos(5));
    assert_eq!(body_cost.estimate_ns(), 3000.0);
    record_all(&body_cost, (0..32).map(|i| 4000 - i));
    assert_eq!(body_cost.estimate_ns(), 3970.0);
    record_all(&body_cost, (0..32).map(|i| 4000 - i));
    assert_eq!(body_cost.estimate_ns(), 3970.0);
}

#[test]
fn a_body_at_the_expensive_threshold_is_expensive() {
    let (_, body_cost) = tree();
    record_all(&body_cost, [999; 32]);
    record_all(&body_cost, [1000; 32]);
    assert_eq!(body_cost.estimate_ns(), 999.0);
    record_all(&body_cost, [1000; 32]);
    assert_eq!(body_cost.estimate_ns(), 1000.0);
}

#[test]
fn a_cheap_window_cancels_a_pending_confirmation() {
    let (_, body_cost) = tree();
    record_all(&body_cost, [200; 32]);
    record_all(&body_cost, [5000; 32]);
    record_all(&body_cost, [300; 32]);
    record_all(&body_cost, [5000; 32]);
    assert_eq!(body_cost.estimate_ns(), 300.0);
    record_all(&body_cost, [5000; 32]);
    assert_eq!(body_cost.estimate_ns(), 5000.0);
}

#[test]
fn the_estimate_waits_for_a_full_window_and_skips_zero_samples() {
    let (_, body_cost) = tree();
    record_all(&body_cost, [100; 31]);
    body_cost.record(Duration::ZERO);
    assert!(!body_cost.has_history());
    assert_eq!(body_cost.estimate_ns(), 0.0);
    body_cost.record(nanos(100));
    assert_eq!(body_cost.estimate_ns(), 100.0);
}

#[test]
fn an_uncalibrated_estimate_takes_nothing_as_expensive() {
    let body_cost = BodyCost::new();
    record_all(&body_cost, [u32::MAX as u64; 32]);
    assert_eq!(body_cost.estimate_ns(), u32::MAX as f64);
}

#[test]
fn reset_forgets_the_history_and_the_batch() {
    let (mut tree, body_cost) = tree();
    record_all(&body_cost, [700; 32]);
    record_all(&body_cost, [5000; 32]);
    tree.record_execution(nanos(10), 1);
    tree.complete_batch(4096);
    assert_eq!(tree.select_path(0, 1, 2, 0), ExecutionPath::Staged);

    tree.reset();

    assert!(!body_cost.has_history());
    assert_eq!(body_cost.estimate_ns(), 0.0);
    assert_eq!(tree.batch_size(), 2);
    assert_eq!(tree.service_time_ns(), 0.0);
    // The pending confirmation is gone too: one expensive window is not enough.
    record_all(&body_cost, [5000; 32]);
    assert!(body_cost.has_history());
    assert_eq!(body_cost.estimate_ns(), 0.0);
    // And batches aim for the direct work target again.
    tree.record_execution(nanos(1000), 1);
    for _ in 0..10 {
        tree.complete_batch(4096);
    }
    assert_eq!(tree.batch_size(), 128);
}

#[test]
fn idle_decisions_follow_the_upstream_policy() {
    // Traced from the upstream policy.
    let (tree, body_cost) = tree();
    assert!(!tree.should_idle(950_000, 1, 8, Some(4)));
    record_all(&body_cost, [50; 32]);
    assert!(tree.should_idle(950_000, 2, 16, Some(8)));
    assert!(!tree.should_idle(50_000, 2, 16, Some(8)));
    assert!(tree.should_idle(50_000, 0, 16, Some(8)));
    assert!(!tree.should_idle(950_000, 16, 16, Some(8)));
    assert!(!tree.should_idle(950_000, 2, 16, Some(1)));
    assert!(!tree.should_idle(950_000, 0, 1, Some(8)));
    assert!(!tree.should_idle(950_000, 0, 16, None));
    assert!(tree.should_idle(950_000, 0, 16, Some(2)));
    assert!(!tree.should_idle(950_000, 0, 0, Some(2)));
}

#[test]
fn paths_stay_direct_until_staging_pays() {
    let (mut tree, body_cost) = tree();
    // No history yet.
    assert_eq!(tree.select_path(0, 4, 8, 900_000), ExecutionPath::Direct);
    record_all(&body_cost, [501; 32]);
    assert_eq!(tree.select_path(0, 4, 8, 900_000), ExecutionPath::Staged);
    // Nothing upstream, a lone worker, or a productive handle per worker.
    assert_eq!(tree.select_path(0, 0, 8, 900_000), ExecutionPath::Direct);
    assert_eq!(tree.select_path(0, 4, 1, 900_000), ExecutionPath::Direct);
    assert_eq!(tree.select_path(8, 4, 8, 900_000), ExecutionPath::Direct);
    assert_eq!(tree.select_path(7, 4, 8, 900_000), ExecutionPath::Staged);
    // Bodies above the direct threshold stage even under light contention.
    assert_eq!(tree.select_path(0, 4, 8, 0), ExecutionPath::Staged);

    record_all(&body_cost, [500; 32]);
    assert_eq!(tree.select_path(0, 4, 8, 850_000), ExecutionPath::Direct);
    assert_eq!(tree.select_path(0, 4, 8, 850_001), ExecutionPath::Staged);
}

#[test]
fn the_path_picks_the_batch_work_target() {
    let (mut tree, body_cost) = tree();
    record_all(&body_cost, [900; 32]);
    tree.record_execution(nanos(1000), 1);
    for _ in 0..20 {
        tree.complete_batch(u64::MAX);
    }
    assert_eq!(tree.batch_size(), 128);
    tree.select_path(0, 4, 8, 900_000);
    for _ in 0..20 {
        tree.complete_batch(u64::MAX);
    }
    assert_eq!(tree.batch_size(), 4096);
    tree.select_path(0, 0, 8, 900_000);
    tree.complete_batch(u64::MAX);
    assert_eq!(tree.batch_size(), 2048);
}

/// Answers with fixed timings and records what it was asked.
#[derive(Debug, Default)]
struct Recording {
    park: Duration,
    half_life: Duration,
    asked: Mutex<Vec<(f64, f64, f64, Duration)>>,
}

impl TimingProvider for Recording {
    fn park(&self, contention: f64, ratio: f64, body: f64, fallback: Duration) -> Duration {
        self.asked
            .lock()
            .unwrap()
            .push((contention, ratio, body, fallback));
        self.park
    }

    fn half_life(&self, contention: f64, ratio: f64, body: f64, fallback: Duration) -> Duration {
        self.asked
            .lock()
            .unwrap()
            .push((contention, ratio, body, fallback));
        self.half_life
    }
}

/// A handle that is always held by someone else, so every pull records full contention.
struct Contended;

impl UpstreamHandle for Contended {
    fn sequence(&self) -> u64 {
        0
    }

    fn pull(
        &self,
        _: &mut dyn FnMut(Box<dyn Frame>),
        _: &mut dyn FnMut(&dyn Frame) -> bool,
        _: u64,
    ) -> Option<u64> {
        Some(0)
    }

    fn request(&self, _: u64) {}

    fn is_complete(&self) -> bool {
        false
    }

    fn try_acquire(&self) -> bool {
        false
    }
}

/// Runs `f` with a worker queue on this thread that has seen full contention.
fn with_contended_queue<R>(f: impl FnOnce(&mut UpstreamQueue) -> R) -> R {
    let registry = UpstreamRegistry::with_topology(1, CpuSet::new());
    registry.register_with(|| Some(0));
    registry.add_handle(Arc::new(Contended));
    let result = registry
        .with_thread_queue(|queue| {
            queue.pull(&mut |_| {}, &mut |_| false, 1);
            f(queue)
        })
        .unwrap();
    registry.remove_thread();
    result
}

fn policy_with(function: Arc<Recording>) -> IdlePolicy {
    IdlePolicy::new(nanos(15_000), nanos(1_000_000), function).unwrap()
}

#[test]
fn the_idle_park_asks_the_function_and_installs_its_half_life() {
    let function = Arc::new(Recording {
        park: nanos(77_000),
        half_life: nanos(5_000),
        ..Default::default()
    });
    let (tree, body_cost) = tree_with(policy_with(function.clone()));
    record_all(&body_cost, [250; 32]);
    let now = Instant::now();

    let (park, later) = with_contended_queue(|queue| {
        let park = tree.idle_park(queue, now, 4, 1);
        (
            park,
            queue.adaptive_contention(now + nanos(5_000), nanos(1_000_000)),
        )
    });

    assert_eq!(park, nanos(77_000));
    assert_eq!(
        *function.asked.lock().unwrap(),
        [
            (1.0, 0.25, 250.0, nanos(15_000)),
            (1.0, 0.25, 250.0, nanos(1_000_000)),
        ]
    );
    // One installed half-life later, the contention has halved.
    assert_eq!(later, ACQUIRE_CONTENTION_SCALE / 2);
}

#[test]
fn a_zero_half_life_keeps_the_policys_and_no_workers_is_no_ratio() {
    let function = Arc::new(Recording::default());
    let (tree, _) = tree_with(policy_with(function.clone()));
    let now = Instant::now();

    let (park, later) = with_contended_queue(|queue| {
        let park = tree.idle_park(queue, now, 0, 3);
        (
            park,
            queue.adaptive_contention(now + nanos(1_000_000), nanos(1)),
        )
    });

    assert_eq!(park, Duration::ZERO);
    let asked = function.asked.lock().unwrap();
    assert!(asked[0].1.is_nan());
    assert_eq!(asked[0].2, 0.0);
    assert_eq!(later, ACQUIRE_CONTENTION_SCALE / 2);
}

#[test]
fn a_fixed_policy_parks_for_its_idle_park() {
    let policy = IdlePolicy::fixed(nanos(3_000), nanos(1_000)).unwrap();
    let (tree, _) = tree_with(policy);
    let park = with_contended_queue(|queue| tree.idle_park(queue, Instant::now(), 2, 1));
    assert_eq!(park, nanos(3_000));
    assert_eq!(tree.idle_policy().idle_park(), nanos(3_000));
}

#[test]
fn calibration_measures_both_thresholds() {
    let body_cost = Arc::new(BodyCost::new());
    let tree = DecisionTree::calibrated(IdlePolicy::default(), body_cost.clone());
    assert!(tree.direct_threshold_ns > 0.0);
    assert!(body_cost.window().expensive_threshold_ns > 0.0);
    assert!(body_cost.window().expensive_threshold_ns.is_finite());
}
