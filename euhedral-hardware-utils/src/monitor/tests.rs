use super::*;
use crate::monitor::sampling::{FastHardwareSample, SlowHardwareSample};
use crate::util::CpuSet;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicUsize};
use std::sync::{Weak, mpsc};

const SPAN: usize = 2;
const SECOND: i64 = 1_000_000_000;

/// Each snapshot is one second after the previous, with
/// half a CPU of usage and a tenth of throttling per second.
struct Incrementing {
    samples: Arc<AtomicI64>,
}

impl SystemSnapshotProvider for Incrementing {
    fn snapshot(&mut self) -> Option<SystemSnapshot> {
        let sample = self.samples.fetch_add(1, Ordering::SeqCst);
        let elapsed = u64::try_from(sample).unwrap() * SECOND as u64;
        SystemSnapshot::new(
            sample * SECOND,
            SPAN,
            1.0,
            100_000,
            elapsed / 2,
            elapsed / 10,
            CpuSet::range(0, SPAN),
            &[0.0; SPAN],
            1_000,
            600,
            100,
            u64::try_from(sample).unwrap() * 200,
        )
        .ok()
    }
}

/// A clock one second past the most recent snapshot.
struct SampleClock(Arc<AtomicI64>);

impl MonotonicClock for SampleClock {
    fn now_ns(&self) -> i64 {
        self.0.load(Ordering::SeqCst) * SECOND
    }
}

fn incrementing(update: TopologyUpdate) -> (ResourceMonitor, Arc<AtomicI64>) {
    let samples = Arc::new(AtomicI64::new(0));
    let provider = Box::new(LegacyAdapter::new(Incrementing {
        samples: Arc::clone(&samples),
    }));
    let clock = Arc::new(SampleClock(Arc::clone(&samples)));
    let monitor =
        ResourceMonitor::build(SPAN, DEFAULT_PERIOD, provider, clock, false, update).unwrap();
    (monitor, samples)
}

fn no_update() -> TopologyUpdate {
    Box::new(|_| {})
}

fn wait_until(what: &str, condition: impl Fn() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while !condition() {
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        thread::sleep(Duration::from_millis(1));
    }
}

/// A detailed provider driven by a closure.
struct FnProvider<F>(F);

impl<F: FnMut(i64) -> Option<FastHardwareSample> + Send> DetailedSnapshotProvider
    for FnProvider<F>
{
    fn sample_fast(&mut self, requested_at_ns: i64) -> Option<FastHardwareSample> {
        (self.0)(requested_at_ns)
    }

    fn sample_slow(&mut self, _requested_at_ns: i64) -> Option<SlowHardwareSample> {
        None
    }
}

fn quiet(at: i64) -> Option<FastHardwareSample> {
    Some(FastHardwareSample::unsupported(
        at,
        SPAN,
        CpuSet::range(0, SPAN),
    ))
}

fn with_fn_provider(
    provider: impl FnMut(i64) -> Option<FastHardwareSample> + Send + 'static,
    period: Duration,
) -> ResourceMonitor {
    ResourceMonitor::build(
        SPAN,
        period,
        Box::new(FnProvider(provider)),
        Arc::new(SystemClock),
        true,
        no_update(),
    )
    .unwrap()
}

#[test]
fn utilization_before_start_evaluates_once_and_caches() {
    let updates = Arc::new(AtomicUsize::new(0));
    let counted = Arc::clone(&updates);
    let (monitor, samples) = incrementing(Box::new(move |_| {
        counted.fetch_add(1, Ordering::SeqCst);
    }));
    let first = monitor.utilization().unwrap();
    let after_first = samples.load(Ordering::SeqCst);
    assert!(Arc::ptr_eq(&first, &monitor.utilization().unwrap()));
    assert_eq!(samples.load(Ordering::SeqCst), after_first);
    assert_eq!(updates.load(Ordering::SeqCst), 1);
    assert_eq!(monitor.state(), MonitorState::New);
}

#[test]
fn calculates_utilization_from_injected_snapshots() {
    let (monitor, _samples) = incrementing(no_update());
    let captured = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&captured);
    monitor.add_listener(move |utilization| sink.lock().unwrap().push(Arc::clone(utilization)));
    monitor.start().unwrap();
    wait_until("four publications", || captured.lock().unwrap().len() >= 4);
    monitor.close();

    let utilization = captured.lock().unwrap().last().cloned().unwrap();
    assert_eq!(utilization.quota_cpus, 1.0);
    assert!(
        (utilization.quota_cpu_usage - 0.5).abs() < 0.05,
        "{}",
        utilization.quota_cpu_usage
    );
    assert!(
        (utilization.cpu_throttle_ratio - 0.1).abs() < 0.01,
        "{}",
        utilization.cpu_throttle_ratio
    );
    assert_eq!(utilization.global_memory_pool, 1_000);
    assert_eq!(utilization.total_memory_utilization, 0.5);
    assert!(utilization.disk_io_bytes_per_second > 0.0);
    assert_eq!(utilization.global_effective_cpus, CpuSet::range(0, SPAN));
    assert_eq!(monitor.state(), MonitorState::Closed);
}

#[test]
fn a_restart_starts_counters_from_a_new_baseline() {
    // The topology update sees every evaluation; listeners may skip values.
    let captured = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&captured);
    let (monitor, samples) = incrementing(Box::new(move |utilization: &HardwareUtilization| {
        sink.lock()
            .unwrap()
            .push((utilization.timestamp_ns, utilization.quota_cpu_usage))
    }));
    monitor.start().unwrap();
    wait_until("steady usage", || {
        captured
            .lock()
            .unwrap()
            .iter()
            .filter(|&&(_, usage)| usage > 0.4)
            .count()
            >= 2
    });
    monitor.stop();
    wait_until("the reading after the stop", || {
        let life = monitor.shared.life();
        !life.coalesced_read && life.evaluating == 0
    });
    // The poller is parked, so the next snapshot handed out is the restart's first. Results are
    // stamped by the clock after their snapshot, so the reading after the stop is stamped exactly
    // here and the restart's first evaluation later.
    let restarted_at = samples.load(Ordering::SeqCst) * SECOND;
    monitor.start().unwrap();
    let first_after_restart = || {
        captured
            .lock()
            .unwrap()
            .iter()
            .find(|&&(at, _)| at > restarted_at)
            .map(|&(_, usage)| usage)
    };
    wait_until("an evaluation after restart", || {
        first_after_restart().is_some()
    });
    monitor.close();
    assert_eq!(
        first_after_restart(),
        Some(0.0),
        "the first interval after a restart must be a baseline"
    );
}

#[test]
fn stopped_monitor_parks_and_restarts() {
    let calls = Arc::new(AtomicUsize::new(0));
    let counted = Arc::clone(&calls);
    let monitor = with_fn_provider(
        move |at| {
            counted.fetch_add(1, Ordering::SeqCst);
            quiet(at)
        },
        Duration::from_millis(1),
    );
    monitor.start().unwrap();
    wait_until("sampling", || calls.load(Ordering::SeqCst) >= 3);
    monitor.stop();
    assert_eq!(monitor.state(), MonitorState::Stopped);
    thread::sleep(Duration::from_millis(20));
    let stopped = calls.load(Ordering::SeqCst);
    thread::sleep(Duration::from_millis(50));
    assert_eq!(
        calls.load(Ordering::SeqCst),
        stopped,
        "a stopped monitor kept sampling"
    );
    for _ in 0..3 {
        monitor.start().unwrap();
        let resumed = calls.load(Ordering::SeqCst);
        wait_until("sampling after restart", || {
            calls.load(Ordering::SeqCst) > resumed
        });
        monitor.stop();
    }
    assert!(monitor.close_timeout(Duration::from_secs(5)));
}

#[test]
fn close_from_a_listener_returns() {
    let monitor = Arc::new(with_fn_provider(quiet, Duration::from_millis(1)));
    let (sender, receiver) = mpsc::channel();
    let sender = Mutex::new(sender);
    let weak = Arc::downgrade(&monitor);
    monitor.add_listener(move |_| {
        if let Some(monitor) = weak.upgrade() {
            let closed = monitor.close_timeout(Duration::from_secs(5));
            let _ = sender.lock().unwrap().send(closed);
        }
    });
    monitor.start().unwrap();
    let closed_from_listener = receiver.recv_timeout(Duration::from_secs(10)).unwrap();
    assert!(
        !closed_from_listener,
        "a listener cannot see the monitor fully closed"
    );
    assert!(monitor.close_timeout(Duration::from_secs(5)));
    assert_eq!(monitor.state(), MonitorState::Closed);
}

#[test]
fn close_with_a_blocked_provider_times_out_then_completes() {
    let (entered, entered_rx) = mpsc::channel();
    let (release, release_rx) = mpsc::channel::<()>();
    let blocked = Arc::new(AtomicBool::new(true));
    let blocking = Arc::clone(&blocked);
    let monitor = with_fn_provider(
        move |at| {
            if blocking.swap(false, Ordering::SeqCst) {
                entered.send(()).unwrap();
                release_rx.recv().unwrap();
            }
            quiet(at)
        },
        Duration::from_millis(1),
    );
    monitor.start().unwrap();
    entered_rx.recv_timeout(Duration::from_secs(10)).unwrap();
    assert!(!monitor.close_timeout(Duration::from_millis(50)));
    assert_eq!(monitor.state(), MonitorState::Closing);
    release.send(()).unwrap();
    assert!(monitor.close_timeout(Duration::from_secs(5)));
    assert_eq!(monitor.state(), MonitorState::Closed);
}

#[test]
fn a_panicking_provider_does_not_kill_the_monitor() {
    let calls = Arc::new(AtomicUsize::new(0));
    let counted = Arc::clone(&calls);
    let monitor = with_fn_provider(
        move |at| {
            if counted.fetch_add(1, Ordering::SeqCst).is_multiple_of(2) {
                panic!("provider failure");
            }
            quiet(at)
        },
        Duration::from_millis(1),
    );
    let published = Arc::new(AtomicUsize::new(0));
    let sink = Arc::clone(&published);
    monitor.add_listener(move |_| {
        sink.fetch_add(1, Ordering::SeqCst);
    });
    monitor.start().unwrap();
    wait_until("publications despite panics", || {
        published.load(Ordering::SeqCst) >= 2
    });
    assert_eq!(monitor.state(), MonitorState::Running);
    assert!(monitor.close_timeout(Duration::from_secs(5)));
}

#[test]
fn close_from_inside_an_evaluation_does_not_deadlock() {
    let slot: Arc<OnceLock<Weak<ResourceMonitor>>> = Arc::new(OnceLock::new());
    let inner = Arc::clone(&slot);
    let monitor = Arc::new(with_fn_provider(
        move |at| {
            if let Some(monitor) = inner.get().and_then(Weak::upgrade) {
                assert!(!monitor.close_timeout(Duration::from_secs(5)));
                // A nested read returns the cached value instead of re-entering the evaluation.
                let _ = monitor.utilization();
            }
            quiet(at)
        },
        Duration::from_millis(1),
    ));
    slot.set(Arc::downgrade(&monitor)).unwrap();
    monitor.start().unwrap();
    wait_until("the self-close", || {
        monitor.state() != MonitorState::Running
    });
    assert!(monitor.close_timeout(Duration::from_secs(5)));
}

#[test]
fn closed_monitors_reject_start_and_publish_nothing() {
    let monitor = with_fn_provider(quiet, Duration::from_millis(1));
    monitor.close();
    assert!(matches!(monitor.start(), Err(MonitorError::Closed)));
    assert!(monitor.utilization().is_none());
    monitor.close();
}

#[test]
fn invalid_periods_are_rejected() {
    let build = |period| {
        ResourceMonitor::build(
            SPAN,
            period,
            Box::new(FnProvider(quiet)),
            Arc::new(SystemClock),
            true,
            no_update(),
        )
    };
    assert!(matches!(
        build(Duration::ZERO),
        Err(MonitorError::InvalidPeriod(_))
    ));
    assert!(matches!(
        build(Duration::MAX),
        Err(MonitorError::InvalidPeriod(_))
    ));
}

#[test]
fn the_poller_exits_when_dropped() {
    let monitor = with_fn_provider(quiet, Duration::from_millis(1));
    monitor.start().unwrap();
    let shared = Arc::clone(&monitor.shared);
    drop(monitor);
    let life = shared.life();
    assert_eq!(life.state, MonitorState::Closed);
    assert!(!life.poller_alive);
}

#[test]
fn a_regressing_clock_reanchors_instead_of_stalling() {
    struct Backwards(AtomicI64);
    impl MonotonicClock for Backwards {
        fn now_ns(&self) -> i64 {
            // Jumps back by an hour every 64 readings.
            let reading = self.0.fetch_add(1, Ordering::SeqCst);
            (reading % 64) * 10_000_000 - (reading / 64) * 3_600 * SECOND
        }
    }
    let calls = Arc::new(AtomicUsize::new(0));
    let counted = Arc::clone(&calls);
    let provider = FnProvider(move |at| {
        counted.fetch_add(1, Ordering::SeqCst);
        quiet(at)
    });
    let monitor = ResourceMonitor::build(
        SPAN,
        Duration::from_millis(1),
        Box::new(provider),
        Arc::new(Backwards(AtomicI64::new(0))),
        true,
        no_update(),
    )
    .unwrap();
    monitor.start().unwrap();
    wait_until("sampling across clock regressions", || {
        calls.load(Ordering::SeqCst) >= 20
    });
    assert!(monitor.close_timeout(Duration::from_secs(5)));
}
