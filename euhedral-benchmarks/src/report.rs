//! The benchmark's output: one line per iteration, and a summary over every measured iteration of every fork.
//!
//! A fork prints `setup` and `sample` lines; the driver prints them under the fork's number and ends with one
//! `summary` line carrying the mean and standard deviation of the measured iterations' throughput and time per
//! operation, the sources and the worker CPUs.

use crate::high_contention::{Isolation, Sample, mean_and_deviation};
use crate::latency::{Histogram, Quantiles};
use std::time::Duration;

/// The line a fork prints for its setup.
pub fn setup_line(sources: usize, isolation: &Isolation) -> String {
    let core = isolation
        .harness_core
        .map_or_else(|| "none".to_owned(), |core| core.to_string());
    format!(
        "setup sources={sources} harness_core={core} workers={}",
        isolation.workers
    )
}

/// The quantiles as `latency_n=.. p0=..` fields, in nanoseconds.
pub fn quantile_fields(q: &Quantiles) -> String {
    format!(
        "latency_samples={} p0={} p25={} p50={} p90={} p99={} p100={}",
        q.count, q.p0, q.p25, q.p50, q.p90, q.p99, q.p100
    )
}

/// The line a fork prints for an iteration.
pub fn sample_line(sample: &Sample) -> String {
    let mut line = format!(
        "sample phase={} index={} invocations={} seconds={:.3} ops_per_s={:.1} ns_per_op={:.4}",
        if sample.measured { "measure" } else { "warmup" },
        sample.index,
        sample.invocations,
        sample.elapsed.as_secs_f64(),
        sample.ops_per_second,
        sample.ns_per_op
    );
    if let Some(latency) = &sample.latency {
        line.push(' ');
        line.push_str(&quantile_fields(latency));
    }
    line
}

/// The line a fork prints at its end: the merged latencies of its measured iterations.
pub fn histogram_line(measured: &Histogram) -> String {
    format!("histogram {}", measured.to_sparse())
}

/// The histogram a `histogram` line carries.
pub fn parse_histogram(line: &str) -> Option<Histogram> {
    Histogram::from_sparse(line.strip_prefix("histogram")?.trim())
}

/// The iteration a `sample` line describes.
pub fn parse_sample(line: &str) -> Option<Sample> {
    let mut fields = line.strip_prefix("sample ")?.split_whitespace();
    let mut value =
        |name: &str| -> Option<&str> { fields.next()?.strip_prefix(name)?.strip_prefix('=') };
    let measured = match value("phase")? {
        "measure" => true,
        "warmup" => false,
        _ => return None,
    };
    Some(Sample {
        measured,
        index: value("index")?.parse().ok()?,
        invocations: value("invocations")?.parse().ok()?,
        elapsed: Duration::from_secs_f64(value("seconds")?.parse().ok()?),
        ops_per_second: value("ops_per_s")?.parse().ok()?,
        ns_per_op: value("ns_per_op")?.parse().ok()?,
        latency: None,
    })
}

/// The one-line summary of the measured iterations of `forks` forks; `latency` holds every measured fork's latencies.
pub fn summary_line(forks: usize, setup: &str, samples: &[Sample], latency: &Histogram) -> String {
    let measured: Vec<&Sample> = samples.iter().filter(|sample| sample.measured).collect();
    let (ops, ops_deviation) = mean_and_deviation(
        &measured
            .iter()
            .map(|s| s.ops_per_second)
            .collect::<Vec<_>>(),
    );
    let (ns, ns_deviation) =
        mean_and_deviation(&measured.iter().map(|s| s.ns_per_op).collect::<Vec<_>>());
    let mut line = format!(
        "summary forks={forks} iterations={} {setup} throughput_ops_per_s={ops:.0} +- {ops_deviation:.0} \
         average_ns_per_op={ns:.4} +- {ns_deviation:.4}",
        measured.len()
    );
    match latency.quantiles() {
        Some(q) => {
            line.push(' ');
            line.push_str(&quantile_fields(&q));
        }
        None => line.push_str(" latency=off"),
    }
    line
}

#[cfg(test)]
mod tests {
    use super::*;
    use euhedral_hardware_utils::util::CpuSet;

    fn sample(measured: bool, ops: f64) -> Sample {
        Sample {
            measured,
            index: 2,
            invocations: 7,
            elapsed: Duration::from_millis(5_250),
            ops_per_second: ops,
            ns_per_op: 1e9 / ops,
            latency: None,
        }
    }

    #[test]
    fn a_sample_line_parses_back() {
        let line = sample_line(&sample(true, 40_000_000.0));
        let parsed = parse_sample(&line).expect("a sample");
        assert!(parsed.measured);
        assert_eq!((parsed.index, parsed.invocations), (2, 7));
        assert!((parsed.ops_per_second - 40_000_000.0).abs() < 1.0);
        assert!((parsed.ns_per_op - 25.0).abs() < 1e-3);
        assert!(parse_sample("setup sources=3").is_none());
        assert!(parse_sample("sample phase=other index=1").is_none());
    }

    #[test]
    fn the_summary_counts_the_measured_iterations_only() {
        let samples = [
            sample(false, 1.0),
            sample(true, 10_000_000.0),
            sample(true, 30_000_000.0),
        ];
        let mut latency = Histogram::new();
        for value in 1..=100 {
            latency.record(value);
        }
        let line = summary_line(2, "sources=3", &samples, &latency);
        assert!(
            line.starts_with("summary forks=2 iterations=2 sources=3 "),
            "{line}"
        );
        assert!(
            line.contains("throughput_ops_per_s=20000000 +- 14142136"),
            "{line}"
        );
    }

    #[test]
    fn a_sample_line_carries_its_quantiles_and_a_histogram_line_round_trips() {
        let mut with = sample(true, 1e7);
        let mut histogram = Histogram::new();
        for value in [100, 200, 300, 4_000] {
            histogram.record(value);
        }
        with.latency = histogram.quantiles();
        let line = sample_line(&with);
        assert!(line.contains("latency_samples=4 p0=100 "), "{line}");
        assert!(
            line.ends_with("p100=4000") || line.contains(" p100="),
            "{line}"
        );
        assert_eq!(
            parse_histogram(&histogram_line(&histogram)),
            Some(histogram)
        );
        assert!(parse_histogram("sample x").is_none());
    }

    #[test]
    fn the_setup_line_names_the_sources_the_core_and_the_workers() {
        let isolation = Isolation {
            harness_core: Some(7),
            workers: [0, 1, 2].into_iter().collect::<CpuSet>(),
        };
        let line = setup_line(3, &isolation);
        assert!(
            line.starts_with("setup sources=3 harness_core=7 workers="),
            "{line}"
        );
        let none = Isolation {
            harness_core: None,
            workers: CpuSet::default(),
        };
        assert!(setup_line(1, &none).contains("harness_core=none"));
    }
}
