//! The high-contention throughput benchmark (`HighContentionThroughput` upstream).
//!
//! One [`RepeatingSource`] per source feeds no-op frames into one lattice; finishing a frame bumps a striped counter.
//! The harness thread is pinned to a core of its own and waits for the counter to advance by [`TASKS`]: one
//! invocation. An iteration repeats invocations for a fixed time and reports throughput; a fork builds a lattice,
//! runs the warm-up iterations, then the measured ones, and tears it down.

use crate::latency::{Histogram, Quantiles, Recorders};
use crate::no_op::{FRAMES_PER_SOURCE, RepeatingSource};
use euhedral_core::config::LatticeConfig;
use euhedral_core::control_plane::{ControlPlaneLattice, LatticeError};
use euhedral_core::impls::DefaultExecutor;
use euhedral_data_structures::atomics::PaddedLongAdder;
use euhedral_hardware_utils::affinity;
use euhedral_hardware_utils::topology::Topology;
use euhedral_hardware_utils::util::CpuSet;
use std::fmt;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// The completions one invocation waits for (`TASKS`).
pub const TASKS: u64 = 32_000_000;

/// The environment variable that overrides the number of sources (`euhedral.hct.sources` upstream).
pub const SOURCES_VARIABLE: &str = "EUHEDRAL_HCT_SOURCES";

/// How long one invocation may take (`TimeUnit.MINUTES.toNanos(1)`).
pub const INVOCATION_TIMEOUT: Duration = Duration::from_secs(60);

/// The sources when nothing overrides them: one fewer than the physical cores, at least one.
pub fn default_sources(physical_cores: usize) -> usize {
    physical_cores.saturating_sub(1).max(1)
}

/// The sources: `variable` if it holds a positive number, else [`default_sources`].
pub fn sources_from(variable: Option<&str>, physical_cores: usize) -> usize {
    variable
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|&sources| sources > 0)
        .unwrap_or_else(|| default_sources(physical_cores))
}

/// The next absolute completion target, or `None` if the count would wrap (`completionTarget`).
pub fn completion_target(completed: u64) -> Option<u64> {
    completed.checked_add(TASKS)
}

/// The count did not reach the target in time.
#[derive(Debug, PartialEq, Eq)]
pub struct AwaitTimeout {
    pub target: u64,
    pub observed: u64,
}

impl fmt::Display for AwaitTimeout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "timed out waiting for completion target {}; observed {}",
            self.target, self.observed
        )
    }
}

impl std::error::Error for AwaitTimeout {}

/// The sum of the counter, as the monotonic completion count.
pub fn completed(counters: &PaddedLongAdder) -> u64 {
    counters.sum() as u64
}

/// Waits until the completion count reaches `target`, or fails at `timeout` (`await`): spins, checks the sum every 32
/// spins, yields every 128, logs progress every 3 seconds.
#[allow(
    clippy::disallowed_methods,
    reason = "JAVA-BACKOFF: the harness thread waits for the lattice as the upstream harness does, spinning and yielding"
)]
pub fn await_completions(
    counters: &PaddedLongAdder,
    target: u64,
    timeout: Duration,
) -> Result<(), AwaitTimeout> {
    let start = Instant::now();
    let mut log = start;
    let mut spin: u32 = 0;
    let mut now = start;
    while now.duration_since(start) < timeout {
        if spin & 31 == 0 {
            let sum = completed(counters);
            if now.duration_since(log) >= Duration::from_secs(3) {
                eprintln!("progress: {sum}");
                log = now;
            }
            if sum >= target {
                return Ok(());
            }
        }
        spin = spin.wrapping_add(1);
        if spin & 127 == 0 {
            std::thread::yield_now();
        } else {
            std::hint::spin_loop();
        }
        now = Instant::now();
    }
    let observed = completed(counters);
    if observed >= target {
        Ok(())
    } else {
        Err(AwaitTimeout { target, observed })
    }
}

/// The core to reserve for the harness: the highest active efficiency core, else the highest active core
/// (`selectHarnessCore`). `None` with fewer than two active cores.
pub fn select_harness_core(e_cores: &CpuSet, active_cores: &CpuSet) -> Option<usize> {
    if active_cores.len() < 2 {
        return None;
    }
    e_cores
        .intersection(active_cores)
        .last()
        .or_else(|| active_cores.last())
}

/// No CPU was left for the workers.
#[derive(Debug, PartialEq, Eq)]
pub struct NoWorkerCpus;

impl fmt::Display for NoWorkerCpus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("reserving the benchmark core left no CPUs for fragment workers")
    }
}

impl std::error::Error for NoWorkerCpus {}

/// `active_cpus` without every logical CPU of the harness core (`workerCpuSet`).
pub fn worker_cpu_set(
    active_cpus: &CpuSet,
    harness_core_cpus: &CpuSet,
) -> Result<CpuSet, NoWorkerCpus> {
    let workers = active_cpus.difference(harness_core_cpus);
    if workers.is_empty() {
        Err(NoWorkerCpus)
    } else {
        Ok(workers)
    }
}

/// Where the harness runs and where the lattice may place its workers.
#[derive(Debug, PartialEq, Eq)]
pub struct Isolation {
    /// The reserved core, if one was.
    pub harness_core: Option<usize>,
    /// The CPUs the workers may use.
    pub workers: CpuSet,
}

/// Reserves a core for the harness and pins the calling thread to its first CPU. With fewer than two cores, or a
/// platform that will not pin, the harness stays unpinned and every CPU goes to the workers; either is logged.
pub fn isolate_harness_core(topology: &Topology, pin: bool) -> Result<Isolation, NoWorkerCpus> {
    let active_cpus = topology.cpu_set();
    let Some(core) = select_harness_core(topology.e_core_set(), topology.core_set()) else {
        eprintln!("benchmark-core isolation is unavailable with fewer than two physical cores");
        return Ok(Isolation {
            harness_core: None,
            workers: active_cpus.clone(),
        });
    };
    let Some(harness_cpus) = topology.core_info(core).map(|info| info.cpus.clone()) else {
        eprintln!("the topology does not describe core {core}; the harness is not pinned");
        return Ok(Isolation {
            harness_core: None,
            workers: active_cpus.clone(),
        });
    };
    let workers = worker_cpu_set(active_cpus, &harness_cpus)?;
    if pin {
        let pinned = harness_cpus.first().is_some_and(affinity::set_affinity);
        if !pinned {
            eprintln!("unable to pin the harness thread to core {core}; it runs unpinned");
            return Ok(Isolation {
                harness_core: None,
                workers: active_cpus.clone(),
            });
        }
    }
    Ok(Isolation {
        harness_core: Some(core),
        workers,
    })
}

/// How a fork is run.
#[derive(Clone, Debug)]
pub struct Options {
    pub sources: usize,
    pub warmup_iterations: u32,
    pub warmup_time: Duration,
    pub iterations: u32,
    pub iteration_time: Duration,
    /// Completions per invocation.
    pub tasks: u64,
    /// Pin the harness thread to its own core.
    pub pin_harness: bool,
    /// Sample the latency of one frame in 64.
    pub latency: bool,
}

impl Options {
    /// Upstream's annotations: 3 warm-up iterations of 3 seconds, 5 measured of 5, one invocation of [`TASKS`].
    pub fn defaults(sources: usize) -> Self {
        Self {
            sources,
            warmup_iterations: 3,
            warmup_time: Duration::from_secs(3),
            iterations: 5,
            iteration_time: Duration::from_secs(5),
            tasks: TASKS,
            pin_harness: true,
            latency: true,
        }
    }
}

/// One iteration's result.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sample {
    pub measured: bool,
    pub index: u32,
    pub invocations: u64,
    pub elapsed: Duration,
    /// Operations (completed frames) per second.
    pub ops_per_second: f64,
    /// Nanoseconds per operation.
    pub ns_per_op: f64,
    /// The latency quantiles of the frames sampled during the iteration, if latency is sampled.
    pub latency: Option<Quantiles>,
}

/// Why a fork failed.
#[derive(Debug)]
pub enum ForkError {
    Lattice(LatticeError),
    NoWorkers(NoWorkerCpus),
    Timeout(AwaitTimeout),
    Wrapped,
}

impl fmt::Display for ForkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Lattice(error) => write!(f, "the lattice failed: {error:?}"),
            Self::NoWorkers(error) => error.fmt(f),
            Self::Timeout(error) => error.fmt(f),
            Self::Wrapped => f.write_str("the completion count would wrap"),
        }
    }
}

impl std::error::Error for ForkError {}

/// One invocation: waits for `tasks` more completions.
fn invoke(counters: &PaddedLongAdder, tasks: u64) -> Result<(), ForkError> {
    let target = completed(counters)
        .checked_add(tasks)
        .ok_or(ForkError::Wrapped)?;
    await_completions(counters, target, INVOCATION_TIMEOUT).map_err(ForkError::Timeout)
}

/// One iteration: invocations for at least `time`, as the harness counts them.
fn iterate(
    counters: &PaddedLongAdder,
    recorders: Option<&Recorders>,
    tasks: u64,
    time: Duration,
    measured: bool,
    index: u32,
) -> Result<(Sample, Histogram), ForkError> {
    let before = recorders.map(Recorders::snapshot);
    let start = Instant::now();
    let mut invocations = 0_u64;
    while invocations == 0 || start.elapsed() < time {
        invoke(counters, tasks)?;
        invocations += 1;
    }
    let elapsed = start.elapsed();
    let ops_per_second = (invocations * tasks) as f64 / elapsed.as_secs_f64();
    let latency = match (recorders, before) {
        (Some(recorders), Some(before)) => recorders.snapshot().since(&before),
        _ => Histogram::new(),
    };
    let sample = Sample {
        measured,
        index,
        invocations,
        elapsed,
        ops_per_second,
        ns_per_op: 1e9 / ops_per_second,
        latency: latency.quantiles(),
    };
    Ok((sample, latency))
}

/// What a fork produced.
#[derive(Debug)]
pub struct ForkOutcome {
    pub samples: Vec<Sample>,
    /// The latencies of every measured iteration, merged.
    pub measured_latency: Histogram,
}

/// What a fork built, for the report.
#[derive(Debug)]
pub struct Setup {
    pub sources: usize,
    pub isolation: Isolation,
}

/// Runs one fork: builds a lattice of `options.sources` repeating sources, runs the warm-up and measured iterations
/// (calling `report` after each) and tears the lattice down. `Setup` is reported first.
pub fn run_fork(
    options: &Options,
    mut announce: impl FnMut(&Setup),
    mut report: impl FnMut(&Sample),
) -> Result<ForkOutcome, ForkError> {
    let topology = Topology::system();
    let isolation =
        isolate_harness_core(topology, options.pin_harness).map_err(ForkError::NoWorkers)?;
    announce(&Setup {
        sources: options.sources,
        isolation: Isolation {
            harness_core: isolation.harness_core,
            workers: isolation.workers.clone(),
        },
    });

    let slots = topology
        .cpu_span()
        .max(std::thread::available_parallelism().map_or(1, usize::from));
    let counters = Arc::new(PaddedLongAdder::with_options(slots, true, true));
    let recorders = options.latency.then(|| Arc::new(Recorders::new(slots)));
    let sources: Vec<_> = (0..options.sources)
        .map(|source| {
            RepeatingSource::new(
                id_hash(),
                FRAMES_PER_SOURCE,
                id_hash().wrapping_add((source * FRAMES_PER_SOURCE) as u64),
                &counters,
                slots,
                recorders.as_ref(),
            )
        })
        .collect();

    let mut config = LatticeConfig::with_executor(DefaultExecutor::new());
    config.allowed_cpus = isolation.workers;
    config.startup_timeout = Duration::from_secs(60);
    config.shutdown_timeout = Duration::from_secs(60);
    let lattice = ControlPlaneLattice::new(config).map_err(ForkError::Lattice)?;
    lattice.start().map_err(ForkError::Lattice)?;
    for source in &sources {
        lattice
            .add_upstream(source.clone())
            .map_err(ForkError::Lattice)?;
    }

    let result = (|| {
        let mut samples = Vec::new();
        let mut measured_latency = Histogram::new();
        for index in 1..=options.warmup_iterations {
            let (sample, _) = iterate(
                &counters,
                recorders.as_deref(),
                options.tasks,
                options.warmup_time,
                false,
                index,
            )?;
            report(&sample);
            samples.push(sample);
        }
        for index in 1..=options.iterations {
            let (sample, latency) = iterate(
                &counters,
                recorders.as_deref(),
                options.tasks,
                options.iteration_time,
                true,
                index,
            )?;
            measured_latency.merge(&latency);
            report(&sample);
            samples.push(sample);
        }
        Ok(ForkOutcome {
            samples,
            measured_latency,
        })
    })();

    // Stops all repeating sources and releases the lattice after the fork.
    for source in &sources {
        euhedral_core::generics::LatticeSource::complete(&**source);
    }
    lattice.close();
    result
}

/// A fixed starting point for the frames' hashes (`HasherApi.mix(BASE_SEED)` upstream): a splitmix64 step.
fn id_hash() -> u64 {
    let mut x = 0x9E37_79B9_7F4A_7C15_u64;
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// Mean and sample standard deviation.
pub fn mean_and_deviation(values: &[f64]) -> (f64, f64) {
    if values.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    if values.len() < 2 {
        return (mean, 0.0);
    }
    let variance =
        values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64;
    (mean, variance.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cpus(ids: &[usize]) -> CpuSet {
        ids.iter().copied().collect()
    }

    /// Verifies that a hybrid topology reserves its highest active E-core.
    #[test]
    fn prefers_the_highest_active_e_core() {
        assert_eq!(
            select_harness_core(&cpus(&[5, 8, 10]), &cpus(&[1, 3, 5, 8])),
            Some(8)
        );
    }

    /// Verifies homogeneous topologies reserve their highest active core.
    #[test]
    fn falls_back_to_the_highest_active_core() {
        assert_eq!(
            select_harness_core(&CpuSet::default(), &cpus(&[0, 2, 7])),
            Some(7)
        );
    }

    /// Verifies a one-core host keeps every CPU available to the lattice.
    #[test]
    fn declines_isolation_on_one_core() {
        assert_eq!(select_harness_core(&cpus(&[4]), &cpus(&[4])), None);
    }

    /// Verifies all logical siblings of the reserved core are removed.
    #[test]
    fn removes_the_whole_harness_core_from_the_worker_set() {
        assert_eq!(
            worker_cpu_set(&cpus(&[0, 1, 2, 3, 4]), &cpus(&[2, 3])),
            Ok(cpus(&[0, 1, 4]))
        );
    }

    /// Verifies isolation cannot silently leave the lattice without a worker CPU.
    #[test]
    fn rejects_an_empty_worker_set() {
        assert_eq!(
            worker_cpu_set(&cpus(&[2, 3]), &cpus(&[2, 3])),
            Err(NoWorkerCpus)
        );
    }

    /// Verifies each invocation advances the absolute target by the fixed operation count.
    #[test]
    fn advances_the_completion_target_and_refuses_to_wrap() {
        assert_eq!(completion_target(123), Some(32_000_123));
        assert_eq!(completion_target(u64::MAX), None);
    }

    /// Verifies waits succeed at the target and fail rather than returning partial work.
    #[test]
    fn the_completion_wait_is_bounded() {
        let counters = PaddedLongAdder::with_options(1, true, false);
        counters.add_at(0, 32_000_123);
        assert_eq!(
            await_completions(&counters, 32_000_123, Duration::from_millis(1)),
            Ok(())
        );
        let empty = PaddedLongAdder::with_options(1, true, false);
        assert_eq!(
            await_completions(&empty, 32_000_000, Duration::ZERO),
            Err(AwaitTimeout {
                target: 32_000_000,
                observed: 0
            })
        );
    }

    #[test]
    fn the_sources_default_to_one_fewer_than_the_cores_and_can_be_overridden() {
        assert_eq!(default_sources(8), 7);
        assert_eq!(default_sources(1), 1);
        assert_eq!(sources_from(None, 8), 7);
        assert_eq!(sources_from(Some("3"), 8), 3);
        assert_eq!(sources_from(Some(" 5 "), 8), 5);
        assert_eq!(sources_from(Some("0"), 8), 7);
        assert_eq!(sources_from(Some("many"), 8), 7);
    }

    #[test]
    fn mean_and_sample_deviation() {
        let (mean, deviation) = mean_and_deviation(&[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]);
        assert!((mean - 5.0).abs() < 1e-12);
        assert!((deviation - 2.138_089_935_299_395).abs() < 1e-9);
        assert_eq!(mean_and_deviation(&[3.0]), (3.0, 0.0));
        assert!(mean_and_deviation(&[]).0.is_nan());
    }
}
