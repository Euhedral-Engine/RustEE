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
    };
    let mut announced = 0;
    let mut reported = 0;
    let samples = run_fork(&options, |_| announced += 1, |_| reported += 1)
        .unwrap_or_else(|error| panic!("{} failed: {error}", file!()));
    assert_eq!(announced, 1);
    assert_eq!(reported, 3);
    assert_eq!(samples.len(), 3);
    assert_eq!(samples.iter().filter(|sample| sample.measured).count(), 2);
    for sample in &samples {
        assert!(sample.invocations >= 1);
        assert!(sample.ops_per_second > 0.0 && sample.ns_per_op > 0.0);
    }
}
