use super::{FlowStats, HistogramSnapshot};
use std::fmt;
use std::sync::{Arc, Mutex, PoisonError, Weak};

/// Something whose meters a [`MetricsRegistry`] reads: a fragment or a cache.
pub(crate) trait MeterSource: Send + Sync {
    /// Appends the source's current meters to `out`.
    fn collect(&self, out: &mut Vec<Meter>);
}

/// Where fragments and caches register their meters, and where readers take a
/// [`MetricsSnapshot`] of them (Micrometer's `MeterRegistry` with `MetricsAggregator`'s lookups,
/// upstream).
///
/// Give one to [`FragmentConfig::with_metrics`](crate::config::FragmentConfig::with_metrics) (or
/// [`LatticeConfig::with_metrics`](crate::config::LatticeConfig::with_metrics)) and keep a
/// handle: each fragment and cache built from the config registers itself, and leaves when it is
/// closed or dropped. With no registry configured nothing registers and nothing is recorded.
///
/// The registry holds its sources weakly and is locked only to register, leave and list them; a
/// snapshot then reads each source's atomics without stopping its worker.
///
/// Two registries are equal only if they are the same registry.
#[derive(Default)]
pub struct MetricsRegistry {
    sources: Mutex<Vec<Weak<dyn MeterSource>>>,
}

impl MetricsRegistry {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    pub(crate) fn register(&self, source: Weak<dyn MeterSource>) {
        let mut sources = self.lock();
        sources.retain(|source| source.strong_count() > 0);
        sources.push(source);
    }

    /// Removes the source at `address`.
    pub(crate) fn remove(&self, address: *const ()) {
        self.lock()
            .retain(|source| source.strong_count() > 0 && source.as_ptr() as *const () != address);
    }

    /// The live sources registered.
    pub fn len(&self) -> usize {
        self.lock()
            .iter()
            .filter(|source| source.strong_count() > 0)
            .count()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Every registered fragment's and cache's meters, now.
    pub fn snapshot(&self) -> MetricsSnapshot {
        let live: Vec<Arc<dyn MeterSource>> = {
            let mut sources = self.lock();
            sources.retain(|source| source.strong_count() > 0);
            sources.iter().filter_map(Weak::upgrade).collect()
        };
        let mut meters = Vec::with_capacity(live.len() * 4);
        for source in &live {
            source.collect(&mut meters);
        }
        MetricsSnapshot { meters }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Vec<Weak<dyn MeterSource>>> {
        self.sources.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

impl PartialEq for MetricsRegistry {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self, other)
    }
}

impl fmt::Debug for MetricsRegistry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MetricsRegistry")
            .field("sources", &self.len())
            .finish()
    }
}

/// What a meter reads.
#[derive(Clone, Debug, PartialEq)]
pub enum MeterValue {
    /// A value now, such as the frames a cache holds.
    Gauge(f64),
    /// A count since the meter was made.
    Counter(u64),
    /// A cumulative distribution summary.
    Summary(HistogramSnapshot),
    /// A flow's current statistics; `None` until it has been measured.
    Flow(Option<FlowStats>),
}

/// One meter of one worker: its full name (the prefix and a suffix, such as
/// `euhedral.execution.latency`), its tags and what it reads.
#[derive(Clone, Debug, PartialEq)]
pub struct Meter {
    pub name: Arc<str>,
    /// The `core` tag: the physical core of the worker's clone.
    pub core: usize,
    /// The `cpu` tag: the CPU of the worker, which tells a core's two SMT workers apart.
    pub cpu: usize,
    pub value: MeterValue,
}

impl Meter {
    /// The value of tag `key` ([`CORE_TAG`](super::CORE_TAG) or [`CPU_TAG`](super::CPU_TAG)).
    pub fn tag(&self, key: &str) -> Option<usize> {
        match key {
            super::CORE_TAG => Some(self.core),
            super::CPU_TAG => Some(self.cpu),
            _ => None,
        }
    }

    pub fn gauge(&self) -> Option<f64> {
        match self.value {
            MeterValue::Gauge(value) => Some(value),
            _ => None,
        }
    }

    pub fn counter(&self) -> Option<u64> {
        match self.value {
            MeterValue::Counter(value) => Some(value),
            _ => None,
        }
    }

    /// The flow's statistics, if this is a flow meter and it has been measured.
    pub fn flow(&self) -> Option<&FlowStats> {
        match &self.value {
            MeterValue::Flow(stats) => stats.as_ref(),
            _ => None,
        }
    }

    pub fn summary(&self) -> Option<&HistogramSnapshot> {
        match &self.value {
            MeterValue::Summary(summary) => Some(summary),
            _ => None,
        }
    }
}

/// The meters of a [`MetricsRegistry`] at one moment, with the lookups of upstream's
/// `MetricsAggregator`.
///
/// Each meter was read on its own while workers kept running, so meters of different workers, or
/// a gauge and a summary of one worker, may be a few records apart.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MetricsSnapshot {
    meters: Vec<Meter>,
}

impl MetricsSnapshot {
    pub fn meters(&self) -> &[Meter] {
        &self.meters
    }

    /// The meters named exactly `name`: build it with [`metric_name`](super::metric_name).
    pub fn named<'a>(&'a self, name: &'a str) -> impl Iterator<Item = &'a Meter> + 'a {
        self.meters.iter().filter(move |meter| &*meter.name == name)
    }

    /// The meter named `name` of the worker on `cpu`.
    pub fn get(&self, name: &str, cpu: usize) -> Option<&Meter> {
        self.meters
            .iter()
            .find(|meter| &*meter.name == name && meter.cpu == cpu)
    }

    /// The meters whose name ends with `suffix`, whatever their prefix.
    pub fn ending_with<'a>(&'a self, suffix: &'a str) -> impl Iterator<Item = &'a Meter> + 'a {
        self.meters
            .iter()
            .filter(move |meter| meter.name.ends_with(suffix))
    }

    /// Every fragment's latency summary.
    pub fn latency_summaries(&self) -> impl Iterator<Item = &Meter> {
        self.ending_with(super::LATENCY_SUMMARY_SUFFIX)
    }

    /// Every fragment's throughput summary.
    pub fn throughput_summaries(&self) -> impl Iterator<Item = &Meter> {
        self.ending_with(super::THROUGHPUT_SUMMARY_SUFFIX)
    }

    /// Every fragment's in-progress gauge.
    pub fn in_progress_gauges(&self) -> impl Iterator<Item = &Meter> {
        self.ending_with(super::IN_PROGRESS_SUFFIX)
    }

    /// Every fragment's count of frames run.
    pub fn frames_counters(&self) -> impl Iterator<Item = &Meter> {
        self.ending_with(super::FRAMES_SUFFIX)
    }

    /// Every fragment's service-time flow.
    pub fn service_time_flows(&self) -> impl Iterator<Item = &Meter> {
        self.ending_with(super::SERVICE_TIME_SUFFIX)
    }

    /// Every fragment's rate flow.
    pub fn rate_flows(&self) -> impl Iterator<Item = &Meter> {
        self.ending_with(super::RATE_SUFFIX)
    }

    /// Every cache's backlog gauge.
    pub fn cache_backlog_gauges(&self) -> impl Iterator<Item = &Meter> {
        self.ending_with(super::CACHE_BACKLOG_SUFFIX)
    }

    /// Every cache's cap-factor summary.
    pub fn cap_factor_summaries(&self) -> impl Iterator<Item = &Meter> {
        self.ending_with(super::CAP_FACTOR_SUFFIX)
    }
}
