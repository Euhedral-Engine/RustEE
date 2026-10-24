//! What the control plane's workers and caches measure about themselves, and how to read it (the
//! upstream `metrics` package: `ExecutionMetrics`, `CacheMetrics` and `MetricsAggregator`).
//!
//! Metrics are off unless a [`MetricsRegistry`] is configured: give one to
//! [`FragmentConfig::with_metrics`](crate::config::FragmentConfig::with_metrics), or build the
//! lattice from [`LatticeConfig::with_metrics`](crate::config::LatticeConfig::with_metrics). Then
//! each fragment worker and each cache registers its meters, tagged with its core and CPU, and
//! [`MetricsRegistry::snapshot`] (or
//! [`ControlPlaneLattice::metrics`](crate::control_plane::ControlPlaneLattice::metrics)) reads them
//! all without stopping the workers.
//!
//! The meters, after the prefix (`euhedral` by default):
//!
//! | Name | Kind | What |
//! |---|---|---|
//! | `.execution.serviceTime` | summary, ns | each frame's body: how long its `execute` took to return (not `do_finally`) |
//! | `.execution.serviceTime.flow` | flow, ns | the current smoothed mean `execute` time of the worker's cycles |
//! | `.execution.throughput` | summary, frames/s | each batch's smoothed rate of frames run |
//! | `.execution.throughput.flow` | flow, frames/s | the current smoothed rate of frames run |
//! | `.execution.inProgress` | gauge | 1 while the worker runs frames, else 0 |
//! | `.execution.frames` | counter | frames the worker has run |
//! | `.cache.backlog` | gauge, frames | frames queued in the cache |
//! | `.cache.capFactor` | summary | the cache's cap factor at each utilization update, in `[0.15, 1]` |
//!
//! Upstream's `.execution.latency` was a batch-end sample of a smoothed per-cycle service time;
//! RustEE replaces it with `.execution.serviceTime`, how long every frame's `execute` took
//! to return (not its `do_finally`), so its
//! percentiles are those of the frames. [`AbstractExecutor::run`](crate::generics::AbstractExecutor::run)
//! times each `execute` call ([`time_execute`]) when the worker running it has metrics; the
//! finalizer is not timed, and nothing is timed without metrics.
//!
//! Each worker keeps two [`FlowRecorder`](crate::utils::FlowRecorder)s in its own cycle state (as
//! upstream's `CycleState` does): one of the mean `execute` time of each productive cycle's frames, one
//! of the frames each ran, whose rate is the throughput. At the end of each productive cycle the
//! worker publishes their averages, deviations, trends and recent extremes as [`FlowStats`] (the
//! `.flow` meters), and at the end of each batch it records the rate in the throughput summary,
//! as upstream does.
//!
//! Summaries are cumulative [`Histogram`]s: count, sum, max and log2 buckets, so readers get means
//! and percentiles, and the difference of two snapshots ([`HistogramSnapshot::since`]) gives a
//! window's. Recording takes no lock and allocates nothing; each worker owns its meters, so
//! workers never write the same cache line. With no registry the hot path pays one untaken
//! branch per cycle.
//!
//! ```
//! use euhedral_core::metrics::{self, MetricsRegistry};
//! use euhedral_core::config::FragmentConfig;
//!
//! let registry = MetricsRegistry::new();
//! let fragment = FragmentConfig::default().with_metrics("app", registry.clone());
//! assert_eq!(fragment.metric_prefix(), "app");
//! // A lattice of these fragments registers each worker and cache; read them with:
//! let snapshot = registry.snapshot();
//! let name = metrics::metric_name("app", metrics::SERVICE_TIME_SUFFIX);
//! assert_eq!(name, "app.execution.serviceTime");
//! for meter in snapshot.named(&name) {
//!     let service_time = meter.summary().unwrap();
//!     println!("core {} cpu {}: p99 {:?} ns", meter.core, meter.cpu, service_time.percentile(0.99));
//! }
//! ```

mod flow;
mod histogram;
mod registry;

pub use flow::FlowStats;
pub(crate) use flow::PublishedFlow;

pub use histogram::{
    BUCKETS, Histogram, HistogramSnapshot, bucket_lower_bound, bucket_upper_bound,
};
pub use registry::{Meter, MeterValue, MetricsRegistry, MetricsSnapshot};

pub(crate) use registry::MeterSource;

use euhedral_data_structures::atomics::CachePadded;
use std::cell::RefCell;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// The prefix of every metric name when none is configured.
pub const DEFAULT_PREFIX: &str = "euhedral";

/// The tag naming a meter's physical core.
pub const CORE_TAG: &str = "core";
/// The tag naming a meter's CPU. Upstream has no such tag: there a core's two SMT workers share
/// their summaries, and only the first worker's gauges are kept.
pub const CPU_TAG: &str = "cpu";

pub const CACHE_BACKLOG_SUFFIX: &str = ".cache.backlog";
pub const CAP_FACTOR_SUFFIX: &str = ".cache.capFactor";

pub const IN_PROGRESS_SUFFIX: &str = ".execution.inProgress";
/// How long each frame's `execute` took, in nanoseconds: a summary. RustEE's replacement for upstream's
/// `.execution.latency`.
pub const SERVICE_TIME_SUFFIX: &str = ".execution.serviceTime";
/// A worker's current service time per frame, in nanoseconds: the [`FlowStats`] of its
/// service-time [`FlowRecorder`](crate::utils::FlowRecorder). Upstream has no such meter.
pub const SERVICE_TIME_FLOW_SUFFIX: &str = ".execution.serviceTime.flow";
pub const THROUGHPUT_SUMMARY_SUFFIX: &str = ".execution.throughput";
/// A worker's current rate, in frames per second: the [`FlowStats`] of the
/// [`FlowRecorder`](crate::utils::FlowRecorder) the throughput summary samples. Upstream has no
/// such meter.
pub const THROUGHPUT_FLOW_SUFFIX: &str = ".execution.throughput.flow";
/// The frames a worker has run. Upstream has no such counter.
pub const FRAMES_SUFFIX: &str = ".execution.frames";

/// `prefix` and `suffix` joined, without a trailing `.` on the prefix; a blank prefix is
/// [`DEFAULT_PREFIX`].
pub fn metric_name(prefix: &str, suffix: &str) -> String {
    let prefix = resolve_prefix(prefix);
    let prefix = prefix.strip_suffix('.').unwrap_or(prefix);
    format!("{prefix}{suffix}")
}

/// `prefix`, or [`DEFAULT_PREFIX`] if it is blank.
fn resolve_prefix(prefix: &str) -> &str {
    if prefix.trim().is_empty() {
        DEFAULT_PREFIX
    } else {
        prefix
    }
}

const NANOS_PER_SECOND: f64 = 1_000_000_000.0;

/// Whether any worker has installed a [`ServiceTimeSink`]: until one has, [`time_execute`] only
/// calls `execute`.
static SINKS_INSTALLED: AtomicBool = AtomicBool::new(false);

thread_local! {
    /// The meters of the lattice worker on this thread, while it runs with metrics.
    static SINK: RefCell<Option<Arc<CachePadded<ExecutionMetrics>>>> = const { RefCell::new(None) };
}

/// Runs `execute`, a frame's [`execute`](crate::generics::AbstractExecutor::execute) call, and on a
/// lattice worker with metrics records how long it took to return in the worker's
/// `.execution.serviceTime` summary. Elsewhere, or without metrics, it just runs `execute`.
///
/// [`AbstractExecutor::run`](crate::generics::AbstractExecutor::run) calls it; an executor that
/// overrides `run` calls it around its own call to `execute` to opt in, and around nothing else:
/// the service time is the body of the work, not the frame's finalizer. An `execute` that panics
/// is not recorded.
///
/// ```
/// use euhedral_core::frames::{Frame, FrameError};
/// use euhedral_core::metrics::time_execute;
///
/// fn run_body(frame: &mut dyn Frame) -> Result<(), FrameError> {
///     time_execute(|| frame.execute())
/// }
/// ```
#[inline]
pub fn time_execute<R>(execute: impl FnOnce() -> R) -> R {
    if !SINKS_INSTALLED.load(Ordering::Relaxed) {
        return execute();
    }
    SINK.with_borrow(|sink| match sink {
        None => execute(),
        Some(metrics) => {
            let start = Instant::now();
            let result = execute();
            metrics.record_service_time(start.elapsed());
            result
        }
    })
}

/// While held, [`time_execute`] on this thread records into a worker's meters.
pub(crate) struct ServiceTimeSink(());

impl ServiceTimeSink {
    /// Makes `metrics` this thread's sink until the guard drops.
    pub(crate) fn install(metrics: Arc<CachePadded<ExecutionMetrics>>) -> Self {
        SINKS_INSTALLED.store(true, Ordering::Relaxed);
        SINK.set(Some(metrics));
        Self(())
    }
}

impl Drop for ServiceTimeSink {
    fn drop(&mut self) {
        SINK.set(None);
    }
}

/// The metric prefix and registry of a fragment or cache config.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct MetricsConfig {
    prefix: Option<String>,
    registry: Option<Arc<MetricsRegistry>>,
}

impl MetricsConfig {
    pub(crate) fn prefix(&self) -> &str {
        resolve_prefix(self.prefix.as_deref().unwrap_or_default())
    }

    pub(crate) fn with_prefix(self, prefix: impl Into<String>) -> Self {
        Self {
            prefix: Some(prefix.into()),
            ..self
        }
    }

    pub(crate) fn registry(&self) -> Option<&Arc<MetricsRegistry>> {
        self.registry.as_ref()
    }

    pub(crate) fn with_registry(self, registry: Option<Arc<MetricsRegistry>>) -> Self {
        Self { registry, ..self }
    }
}

/// A fragment worker's meters (`ExecutionMetrics` upstream). The worker's thread is the only
/// writer. The in-progress gauge reads the fragment's own count, which it keeps with or without
/// metrics.
pub(crate) struct ExecutionMetrics {
    registry: Arc<MetricsRegistry>,
    core: usize,
    cpu: usize,
    service_time_name: Arc<str>,
    service_time_flow_name: Arc<str>,
    throughput_name: Arc<str>,
    throughput_flow_name: Arc<str>,
    in_progress_name: Arc<str>,
    frames_name: Arc<str>,
    frames: CachePadded<AtomicU64>,
    service_time_flow: PublishedFlow,
    throughput_flow: PublishedFlow,
    service_time: Histogram,
    throughput: Histogram,
}

impl ExecutionMetrics {
    /// The meters of the worker on `cpu` of `core`, or `None` without a registry.
    pub(crate) fn new(
        config: &MetricsConfig,
        core: usize,
        cpu: usize,
    ) -> Option<Arc<CachePadded<Self>>> {
        let registry = config.registry()?.clone();
        let prefix = config.prefix();
        let name = |suffix| Arc::<str>::from(metric_name(prefix, suffix));
        Some(Arc::new(CachePadded::new(Self {
            registry,
            core,
            cpu,
            service_time_name: name(SERVICE_TIME_SUFFIX),
            service_time_flow_name: name(SERVICE_TIME_FLOW_SUFFIX),
            throughput_name: name(THROUGHPUT_SUMMARY_SUFFIX),
            throughput_flow_name: name(THROUGHPUT_FLOW_SUFFIX),
            in_progress_name: name(IN_PROGRESS_SUFFIX),
            frames_name: name(FRAMES_SUFFIX),
            frames: CachePadded::new(AtomicU64::new(0)),
            service_time_flow: PublishedFlow::new(),
            throughput_flow: PublishedFlow::new(),
            service_time: Histogram::new(),
            throughput: Histogram::new(),
        })))
    }

    pub(crate) fn registry(&self) -> &Arc<MetricsRegistry> {
        &self.registry
    }

    /// Counts `frames` more frames run.
    #[inline]
    pub(crate) fn record_frames(&self, frames: u64) {
        self.frames.fetch_add(frames, Ordering::Relaxed);
    }

    /// Records how long one frame's `execute` took.
    #[inline]
    pub(crate) fn record_service_time(&self, elapsed: Duration) {
        self.service_time.record(elapsed.as_nanos() as f64);
    }

    /// The frames whose `execute` was timed, and the sum of their times in nanoseconds.
    #[inline]
    pub(crate) fn service_time_totals(&self) -> (u64, f64) {
        self.service_time.totals()
    }

    /// Publishes the current service time per frame, in nanoseconds.
    #[inline]
    pub(crate) fn publish_service_time(&self, stats: &FlowStats) {
        self.service_time_flow.publish(stats);
    }

    /// Publishes the current rate, in frames per second.
    #[inline]
    pub(crate) fn publish_throughput(&self, stats: &FlowStats) {
        self.throughput_flow.publish(stats);
    }

    /// Records a rate in frames per nanosecond, as frames per second.
    pub(crate) fn report_throughput(&self, frames_per_nano: f64) {
        self.throughput.record(frames_per_nano * NANOS_PER_SECOND);
    }

    /// Appends the worker's meters, with `in_progress` for its gauge.
    pub(crate) fn collect(&self, in_progress: u64, out: &mut Vec<Meter>) {
        let meter = |name: &Arc<str>, value| Meter {
            name: name.clone(),
            core: self.core,
            cpu: self.cpu,
            value,
        };
        out.push(meter(
            &self.service_time_name,
            MeterValue::Summary(self.service_time.snapshot()),
        ));
        out.push(meter(
            &self.service_time_flow_name,
            MeterValue::Flow(self.service_time_flow.read()),
        ));
        out.push(meter(
            &self.throughput_name,
            MeterValue::Summary(self.throughput.snapshot()),
        ));
        out.push(meter(
            &self.throughput_flow_name,
            MeterValue::Flow(self.throughput_flow.read()),
        ));
        out.push(meter(
            &self.in_progress_name,
            MeterValue::Gauge(in_progress.min(1) as f64),
        ));
        out.push(meter(
            &self.frames_name,
            MeterValue::Counter(self.frames.load(Ordering::Relaxed)),
        ));
    }
}

/// A cache's meters (`CacheMetrics` upstream). The backlog gauge reads the cache's own count.
pub(crate) struct CacheMetrics {
    registry: Arc<MetricsRegistry>,
    core: usize,
    cpu: usize,
    backlog_name: Arc<str>,
    cap_factor_name: Arc<str>,
    cap_factor: Histogram,
}

impl CacheMetrics {
    /// The meters of the cache on `cpu` of `core`, or `None` without a registry.
    pub(crate) fn new(
        config: &MetricsConfig,
        core: usize,
        cpu: usize,
    ) -> Option<Box<CachePadded<Self>>> {
        let registry = config.registry()?.clone();
        let prefix = config.prefix();
        Some(Box::new(CachePadded::new(Self {
            registry,
            core,
            cpu,
            backlog_name: metric_name(prefix, CACHE_BACKLOG_SUFFIX).into(),
            cap_factor_name: metric_name(prefix, CAP_FACTOR_SUFFIX).into(),
            cap_factor: Histogram::new(),
        })))
    }

    pub(crate) fn registry(&self) -> &Arc<MetricsRegistry> {
        &self.registry
    }

    pub(crate) fn record_cap_factor(&self, cap_factor: f64) {
        self.cap_factor.record(cap_factor);
    }

    /// Appends the cache's meters, with `backlog` frames queued.
    pub(crate) fn collect(&self, backlog: u64, out: &mut Vec<Meter>) {
        out.push(Meter {
            name: self.backlog_name.clone(),
            core: self.core,
            cpu: self.cpu,
            value: MeterValue::Gauge(backlog as f64),
        });
        out.push(Meter {
            name: self.cap_factor_name.clone(),
            core: self.core,
            cpu: self.cpu,
            value: MeterValue::Summary(self.cap_factor.snapshot()),
        });
    }
}

#[cfg(test)]
mod tests;
