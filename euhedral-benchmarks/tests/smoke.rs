//! A tiny run of the benchmark on a real lattice: it must complete its invocations and report sensible samples.

use euhedral_benchmarks::high_contention::{Options, run_fork};
use std::time::Duration;

#[test]
fn a_fork_with_a_tiny_target_completes_and_reports_every_iteration() {
    let options = Options {
        sources: 2,
        warmup_iterations: 1,
        warmup_time: Duration::from_millis(50),
        iterations: 2,
        iteration_time: Duration::from_millis(100),
        tasks: 20_000,
        // The test thread is the harness; pinning it would leave it pinned for the rest of the process.
        pin_harness: false,
        latency: true,
    };
    let mut announced = 0;
    let mut reported = 0;
    let outcome = run_fork(&options, |_| announced += 1, |_| reported += 1)
        .unwrap_or_else(|error| panic!("{} failed: {error}", file!()));
    let samples = &outcome.samples;
    assert_eq!(announced, 1);
    assert_eq!(reported, 3);
    assert_eq!(samples.len(), 3);
    assert_eq!(samples.iter().filter(|sample| sample.measured).count(), 2);
    for sample in samples {
        assert!(sample.invocations >= 1);
        assert!(sample.ops_per_second > 0.0 && sample.ns_per_op > 0.0);
    }
    // One frame in 64 is timed: some were, and the quantiles are ordered.
    let q = outcome
        .measured_latency
        .quantiles()
        .expect("sampled frames were executed");
    assert!(q.count > 0);
    assert!(q.p0 <= q.p25 && q.p25 <= q.p50 && q.p50 <= q.p90 && q.p90 <= q.p99 && q.p99 <= q.p100);
}

#[test]
fn a_fork_without_latency_reports_no_quantiles() {
    let options = Options {
        sources: 1,
        warmup_iterations: 0,
        warmup_time: Duration::ZERO,
        iterations: 1,
        iteration_time: Duration::from_millis(50),
        tasks: 5_000,
        pin_harness: false,
        latency: false,
    };
    let outcome = run_fork(&options, |_| {}, |_| {}).unwrap_or_else(|e| panic!("{} {e}", file!()));
    assert!(outcome.samples[0].latency.is_none());
    assert!(outcome.measured_latency.quantiles().is_none());
}
