//! End-to-end benchmarks of the control plane.
//!
//! - [`high_contention`]: the high-contention throughput benchmark. Many sources feed one lattice with no-op frames
//!   and the harness times how long a fixed number of completions takes, as average time and throughput.
//! - [`no_op`]: the benchmark's frame and repeating source.
//!
//! The binary `high_contention_throughput` runs it; see its `--help`.

pub mod high_contention;
pub mod latency;
pub mod no_op;
pub mod report;
