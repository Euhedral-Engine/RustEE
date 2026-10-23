//! Linux resource sampling from cgroups, `/proc` and sysfs (upstream `LinuxResourceProvider`).
//!
//! Scope signals come from the process's cgroup (v2 preferred, then v1); a process outside any cgroup
//! reads the host-wide `/proc` equivalents. Per-CPU steal and run-queue wait come from `/proc/stat`
//! and `/proc/schedstat`. Every signal is cumulative or a gauge as its type says; a source the host
//! lacks is unsupported, one that fails to read or parse is a transient failure.

use super::fs::{Read, Sources, read_trimmed, signal_of};
use super::parse::{self, StatLine};
use super::paths::{CgroupMode, LinuxPaths};
use crate::monitor::sampling::{
    CpuSlowSignals, DetailedSnapshotProvider, FastHardwareSample, Signal, SlowHardwareSample,
    SystemSlowSignals, SystemSnapshotProvider, ThermalSeverity,
};
use crate::monitor::{SystemSnapshot, monotonic_now_ns};
use crate::topology::Topology;
use crate::util::CpuSet;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

/// A v1 memory limit at or above this is the kernel's "unlimited" (`PAGE_COUNTER_MAX` pages).
const V1_UNLIMITED: u64 = 1 << 62;
const SECTOR_BYTES: u64 = 512;

fn clock_ticks_per_second() -> u64 {
    #[cfg(target_os = "linux")]
    {
        // SAFETY: sysconf has no preconditions.
        let ticks = unsafe { libc::sysconf(libc::_SC_CLK_TCK) };
        if ticks > 0 {
            return ticks as u64;
        }
    }
    100
}

fn page_size() -> u64 {
    #[cfg(target_os = "linux")]
    {
        // SAFETY: sysconf has no preconditions.
        let size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if size > 0 {
            return size as u64;
        }
    }
    4096
}

/// The files each signal reads, chosen once so a signal never switches between sources with
/// unrelated baselines.
#[derive(Debug, Default)]
struct Files {
    cpuset: Option<PathBuf>,
    cpu_max: Option<PathBuf>,
    cfs_quota: Option<PathBuf>,
    cfs_period: Option<PathBuf>,
    /// v2 `cpu.stat` (usage and throttling).
    cpu_stat: Option<PathBuf>,
    cpuacct_usage: Option<PathBuf>,
    v1_cpu_stat: Option<PathBuf>,
    cpu_pressure: Option<PathBuf>,
    memory_max: Option<PathBuf>,
    memory_v1_limit: Option<PathBuf>,
    memory_high: Option<PathBuf>,
    memory_current: Option<PathBuf>,
    memory_stat: Option<PathBuf>,
    memory_v1: bool,
    memory_pressure: Option<PathBuf>,
    vmstat: Option<PathBuf>,
    io_stat: Option<PathBuf>,
    io_pressure: Option<PathBuf>,
    meminfo: PathBuf,
    proc_stat: PathBuf,
    schedstat: PathBuf,
    diskstats: PathBuf,
    sys_block: PathBuf,
    cpu_root: PathBuf,
    thermal: PathBuf,
    platform_profile: PathBuf,
}

impl Files {
    fn resolve(paths: &LinuxPaths) -> Self {
        let bare = paths.mode() == CgroupMode::BareHost;
        let host = |file: &str| Some(paths.proc(file)).filter(|path| bare && path.exists());
        let cpu_stat = paths.v2("cpu.stat");
        let v1_cpu = cpu_stat.is_none();
        let memory_current = paths.v2("memory.current");
        let memory_v1 =
            memory_current.is_none() && paths.v1("memory", "memory.usage_in_bytes").is_some();
        Self {
            cpuset: paths
                .v2("cpuset.cpus.effective")
                .or_else(|| paths.v1("cpuset", "cpuset.effective_cpus"))
                .or_else(|| paths.v1("cpuset", "cpuset.cpus")),
            cpu_max: paths.v2("cpu.max"),
            cfs_quota: paths.v1("cpu", "cpu.cfs_quota_us").filter(|_| v1_cpu),
            cfs_period: paths.v1("cpu", "cpu.cfs_period_us").filter(|_| v1_cpu),
            cpu_stat,
            cpuacct_usage: paths.v1("cpuacct", "cpuacct.usage").filter(|_| v1_cpu),
            v1_cpu_stat: paths.v1("cpu", "cpu.stat").filter(|_| v1_cpu),
            cpu_pressure: paths.v2("cpu.pressure").or_else(|| host("pressure/cpu")),
            memory_max: paths.v2("memory.max"),
            memory_v1_limit: paths
                .v1("memory", "memory.limit_in_bytes")
                .filter(|_| memory_v1),
            memory_high: paths.v2("memory.high"),
            memory_stat: if memory_v1 {
                paths.v1("memory", "memory.stat")
            } else {
                paths.v2("memory.stat")
            },
            memory_current: memory_current.or_else(|| paths.v1("memory", "memory.usage_in_bytes")),
            memory_v1,
            memory_pressure: paths
                .v2("memory.pressure")
                .or_else(|| host("pressure/memory")),
            vmstat: host("vmstat"),
            io_stat: paths.v2("io.stat"),
            io_pressure: paths.v2("io.pressure").or_else(|| host("pressure/io")),
            meminfo: paths.proc("meminfo"),
            proc_stat: paths.proc("stat"),
            schedstat: paths.proc("schedstat"),
            diskstats: paths.proc("diskstats"),
            sys_block: paths.sys("block"),
            cpu_root: paths.sys("devices/system/cpu"),
            thermal: paths.sys("class/thermal"),
            platform_profile: paths.sys("firmware/acpi/platform_profile"),
        }
    }
}

/// The detailed Linux provider.
#[derive(Debug)]
pub(crate) struct LinuxResources {
    mode: CgroupMode,
    files: Files,
    sources: Sources,
    span: usize,
    topology_cpus: CpuSet,
    ns_per_tick: u64,
    page_size: u64,
    /// Counter keys read before, so one that disappears is a failure rather than unsupported.
    seen_keys: HashSet<&'static str>,
    /// Whole physical disks by diskstats name.
    whole_disks: HashMap<String, bool>,
    /// The deepest in-flight queue seen by the last diskstats read.
    max_queue: f64,
}

impl LinuxResources {
    pub(crate) fn new(topology: &Topology) -> Self {
        Self::with_root(Path::new("/"), topology)
    }

    pub(crate) fn with_root(root: &Path, topology: &Topology) -> Self {
        let paths = LinuxPaths::discover(root);
        Self {
            mode: paths.mode(),
            files: Files::resolve(&paths),
            sources: Sources::default(),
            span: topology.cpu_span().max(1),
            topology_cpus: topology.cpu_set().clone(),
            ns_per_tick: 1_000_000_000 / clock_ticks_per_second(),
            page_size: page_size(),
            seen_keys: HashSet::new(),
            whole_disks: HashMap::new(),
            max_queue: 0.0,
        }
    }

    fn read(&mut self, path: Option<PathBuf>) -> Read {
        self.sources.read(path.as_deref())
    }

    /// A keyed counter of an already-read file: a key the file lacks is unsupported until it has been
    /// seen once.
    fn keyed(&mut self, read: &Read, key: &'static str, scale: u64, at: i64) -> Signal<u64> {
        let Read::Text(text) = read else {
            return signal_of(read, at, |_| None);
        };
        let Some(rest) = parse::keyed_line(text, key) else {
            return if self.seen_keys.contains(key) {
                Signal::transient_failure(at)
            } else {
                Signal::unsupported(at)
            };
        };
        self.seen_keys.insert(key);
        rest.split_whitespace()
            .next()
            .and_then(parse::decimal)
            .and_then(|value| value.checked_mul(scale))
            .map_or_else(
                || Signal::transient_failure(at),
                |value| Signal::valid(value, at),
            )
    }

    fn effective_cpus(&mut self) -> CpuSet {
        let path = self.files.cpuset.clone();
        let cpus = match self.read(path) {
            Read::Text(text) => {
                parse::cpu_list(&text).map(|cpus| cpus.intersection(&self.topology_cpus))
            }
            _ => None,
        };
        cpus.filter(|cpus| !cpus.is_empty())
            .unwrap_or_else(|| self.topology_cpus.clone())
    }

    /// The quota in CPUs and the period in ns.
    fn quota(&mut self, at: i64) -> (Signal<f64>, Signal<u64>) {
        if let Some(path) = self.files.cpu_max.clone() {
            let read = self.read(Some(path));
            let parsed = match &read {
                Read::Text(text) => parse::cpu_max(text),
                _ => None,
            };
            return match (parsed, &read) {
                (Some(max), _) => (
                    max.quota_us.map_or_else(
                        || Signal::unsupported(at),
                        |quota| Signal::valid(quota as f64 / max.period_us as f64, at),
                    ),
                    max.period_us
                        .checked_mul(1_000)
                        .map_or_else(|| Signal::transient_failure(at), |ns| Signal::valid(ns, at)),
                ),
                (None, read) => (signal_of(read, at, |_| None), signal_of(read, at, |_| None)),
            };
        }
        if self.files.cfs_quota.is_none() {
            return (Signal::unsupported(at), Signal::unsupported(at));
        }
        let quota = self.read(self.files.cfs_quota.clone());
        let period = self.read(self.files.cfs_period.clone());
        let period = signal_of(&period, at, |text| parse::single(text).filter(|&p| p > 0));
        let quota_cpus = match (&quota, period.get()) {
            (Read::Text(text), _) if text.trim() == "-1" => Signal::unsupported(at),
            (Read::Text(text), Some(period)) => {
                parse::single(text).filter(|&quota| quota > 0).map_or_else(
                    || Signal::transient_failure(at),
                    |quota| Signal::valid(quota as f64 / period as f64, at),
                )
            }
            (Read::Text(_), None) => Signal::transient_failure(at),
            (read, _) => signal_of(read, at, |_| None),
        };
        let period_ns = match period.get() {
            Some(period) => period
                .checked_mul(1_000)
                .map_or_else(|| Signal::transient_failure(at), |ns| Signal::valid(ns, at)),
            None => Signal::new(0, at, period.validity()),
        };
        (quota_cpus, period_ns)
    }

    fn proc_stat(&mut self) -> Option<(Option<StatLine>, HashMap<usize, StatLine>)> {
        let path = self.files.proc_stat.clone();
        let Read::Text(text) = self.read(Some(path)) else {
            return None;
        };
        let mut aggregate = None;
        let mut cpus = HashMap::new();
        for line in text.lines().filter(|line| line.starts_with("cpu")) {
            match parse::stat_line(line)? {
                (None, stat) => aggregate = Some(stat),
                (Some(id), stat) => {
                    cpus.insert(id, stat);
                }
            }
        }
        Some((aggregate, cpus))
    }

    fn schedstat(&mut self) -> Option<HashMap<usize, u64>> {
        let path = self.files.schedstat.clone();
        let Read::Text(text) = self.read(Some(path)) else {
            return None;
        };
        Some(
            text.lines()
                .filter(|line| line.starts_with("cpu"))
                .filter_map(parse::schedstat_line)
                .collect(),
        )
    }

    fn is_whole_disk(&mut self, name: &str) -> bool {
        if let Some(&whole) = self.whole_disks.get(name) {
            return whole;
        }
        // Partitions, device-mapper, md, loop and ram devices have no backing `device` link; optical
        // drives are excluded explicitly.
        let whole = !name.starts_with("sr")
            && self
                .files
                .sys_block
                .join(name.replace('/', "!"))
                .join("device")
                .exists();
        self.whole_disks.insert(name.to_owned(), whole);
        whole
    }

    /// Whole-disk sector bytes, completion latency, completed operations and the deepest queue.
    fn disks(&mut self, at: i64) -> [Signal<u64>; 3] {
        let path = self.files.diskstats.clone();
        let read = self.read(Some(path));
        let Read::Text(text) = &read else {
            let failed = signal_of(&read, at, |_| None::<u64>);
            return [failed; 3];
        };
        let mut bytes = 0u64;
        let mut latency_ms = 0u64;
        let mut operations = 0u64;
        let mut valid = true;
        let mut queue = 0u64;
        for line in text.lines().filter(|line| !line.trim().is_empty()) {
            let Some(disk) = parse::diskstats_line(line) else {
                valid = false;
                continue;
            };
            if !self.is_whole_disk(&disk.name) {
                continue;
            }
            bytes = bytes.saturating_add(
                disk.sectors_read
                    .saturating_add(disk.sectors_written)
                    .saturating_mul(SECTOR_BYTES),
            );
            latency_ms = latency_ms.saturating_add(disk.read_ms.saturating_add(disk.write_ms));
            operations = operations.saturating_add(disk.reads.saturating_add(disk.writes));
            queue = queue.max(disk.in_flight);
        }
        self.max_queue = queue as f64;
        if !valid {
            return [Signal::transient_failure(at); 3];
        }
        [
            Signal::valid(bytes, at),
            Signal::valid(latency_ms.saturating_mul(1_000_000), at),
            Signal::valid(operations, at),
        ]
    }

    fn memory_reclaim_pages(&self, text: &str) -> Option<u64> {
        let keys = ["pgsteal_kswapd", "pgsteal_direct", "pgsteal_khugepaged"];
        let mut found = false;
        let mut total = 0u64;
        for key in keys {
            if let Some(value) = parse::keyed(text, key) {
                found = true;
                total = total.saturating_add(value);
            }
        }
        if !found {
            // Older kernels count per zone, as `pgsteal_kswapd_normal` and `pgsteal_direct_dma32`.
            for line in text.lines() {
                let mut tokens = line.split_whitespace();
                let (Some(key), Some(value)) = (tokens.next(), tokens.next()) else {
                    continue;
                };
                if key.starts_with("pgsteal_kswapd_") || key.starts_with("pgsteal_direct_") {
                    found = true;
                    total = total.saturating_add(parse::decimal(value)?);
                }
            }
        }
        found.then_some(total)
    }

    fn sample_fast_at(&mut self, at: i64) -> FastHardwareSample {
        let effective = self.effective_cpus();
        let mut sample = FastHardwareSample::unsupported(at, self.span, effective.clone());
        (sample.quota_capacity_cpus, sample.quota_period_ns) = self.quota(at);

        let stat = self.proc_stat();
        let stat_failed = |at| {
            if stat.is_none() {
                Signal::transient_failure(at)
            } else {
                Signal::unsupported(at)
            }
        };
        if let Some(path) = self.files.cpu_stat.clone() {
            let read = self.read(Some(path));
            sample.productive_cpu_ns = self.keyed(&read, "usage_usec", 1_000, at);
            sample.scope_quota_throttled_ns = self.keyed(&read, "throttled_usec", 1_000, at);
        } else if self.files.cpuacct_usage.is_some() || self.files.v1_cpu_stat.is_some() {
            let usage = self.files.cpuacct_usage.clone();
            sample.productive_cpu_ns = self.sources.signal(usage.as_deref(), at, parse::single);
            let read = self.read(self.files.v1_cpu_stat.clone());
            sample.scope_quota_throttled_ns = self.keyed(&read, "throttled_time", 1, at);
        } else if self.mode == CgroupMode::BareHost {
            sample.productive_cpu_ns = match stat.as_ref().map(|(aggregate, _)| *aggregate) {
                Some(Some(aggregate)) => {
                    Signal::valid(aggregate.busy().saturating_mul(self.ns_per_tick), at)
                }
                Some(None) => Signal::transient_failure(at),
                None => stat_failed(at),
            };
        }
        let cpu_pressure = self.files.cpu_pressure.clone();
        sample.scope_psi_stall_ns =
            self.sources
                .signal(cpu_pressure.as_deref(), at, parse::psi_some_total_ns);

        let run_delay = self.schedstat();
        for cpu in effective.iter().filter(|&cpu| cpu < self.span) {
            let signals = &mut sample.cpu_signals[cpu];
            signals.steal = match &stat {
                Some((_, cpus)) => cpus.get(&cpu).map_or_else(
                    || Signal::transient_failure(at),
                    |line| Signal::valid(line.steal.saturating_mul(self.ns_per_tick), at),
                ),
                None => Signal::transient_failure(at),
            };
            signals.scheduler_wait = match &run_delay {
                Some(delays) => delays.get(&cpu).map_or_else(
                    || Signal::transient_failure(at),
                    |&delay| Signal::valid(delay, at),
                ),
                None if self.files.schedstat.exists() => Signal::transient_failure(at),
                None => Signal::unsupported(at),
            };
        }
        self.sample_memory(&mut sample, at);
        self.sample_io(&mut sample, at);
        sample
    }

    fn sample_memory(&mut self, sample: &mut FastHardwareSample, at: i64) {
        let meminfo_path = self.files.meminfo.clone();
        let meminfo = self.read(Some(meminfo_path));
        let host_total = signal_of(&meminfo, at, |text| parse::meminfo_bytes(text, "MemTotal"));
        let memory = &mut sample.memory_signals;
        let cgroup_limit = if let Some(path) = self.files.memory_max.clone() {
            Some(
                self.sources
                    .signal(Some(&path), at, |text| match text.trim() {
                        "max" => Some(u64::MAX),
                        text => parse::decimal(text),
                    }),
            )
        } else {
            self.files.memory_v1_limit.clone().map(|path| {
                self.sources.signal(Some(&path), at, |text| {
                    parse::single(text).map(|v| if v >= V1_UNLIMITED { u64::MAX } else { v })
                })
            })
        };
        // An unlimited scope is bounded by the host's memory.
        memory.hard_limit_bytes = match cgroup_limit {
            Some(limit) if limit.get() == Some(u64::MAX) => host_total,
            Some(limit) => limit,
            None => host_total,
        };
        if let Some(path) = self.files.memory_high.clone() {
            let high = self
                .sources
                .signal(Some(&path), at, |text| match text.trim() {
                    "max" => Some(u64::MAX),
                    text => parse::decimal(text),
                });
            memory.high_limit_bytes = if high.get() == Some(u64::MAX) {
                Signal::unsupported(at)
            } else {
                high
            };
        }
        let stat = self.read(self.files.memory_stat.clone());
        let memory = &mut sample.memory_signals;
        if let Some(path) = self.files.memory_current.clone() {
            memory.usage_bytes = self.sources.signal(Some(&path), at, parse::single);
            let key = if self.files.memory_v1 {
                "total_inactive_file"
            } else {
                "inactive_file"
            };
            memory.inactive_file_bytes = signal_of(&stat, at, |text| parse::keyed(text, key));
        } else if self.mode == CgroupMode::BareHost {
            memory.usage_bytes = signal_of(&meminfo, at, |text| {
                parse::meminfo_bytes(text, "MemTotal")?
                    .checked_sub(parse::meminfo_bytes(text, "MemAvailable")?)
            });
        }
        let page = self.page_size;
        let reclaim_source = if self.files.memory_v1 {
            Read::Absent
        } else if self.files.memory_stat.is_some() {
            stat
        } else {
            self.read(self.files.vmstat.clone())
        };
        let reclaim = match &reclaim_source {
            Read::Text(text) => match self.memory_reclaim_pages(text) {
                Some(pages) => Signal::valid(pages.saturating_mul(page), at),
                None => Signal::unsupported(at),
            },
            read => signal_of(read, at, |_| None),
        };
        let pressure = self.files.memory_pressure.clone();
        let memory_stall = self
            .sources
            .signal(pressure.as_deref(), at, parse::psi_some_total_ns);
        let memory = &mut sample.memory_signals;
        memory.cumulative_reclaim_bytes = reclaim;
        memory.memory_stall_ns = memory_stall;
    }

    fn sample_io(&mut self, sample: &mut FastHardwareSample, at: i64) {
        let [disk_bytes, latency, operations] = self.disks(at);
        let io = &mut sample.io_signals;
        io.operation_latency_ns = latency;
        io.completed_operations = operations;
        io.maximum_queue_depth = if operations.is_valid() {
            Signal::valid(self.max_queue, at)
        } else {
            Signal::new(0.0, at, operations.validity())
        };
        let io_stat = self.files.io_stat.clone();
        let io_bytes = match io_stat {
            Some(path) => self.sources.signal(Some(&path), at, parse::io_stat_bytes),
            None => disk_bytes,
        };
        let pressure = self.files.io_pressure.clone();
        let io_stall = self
            .sources
            .signal(pressure.as_deref(), at, parse::psi_some_total_ns);
        let io = &mut sample.io_signals;
        io.productive_bytes = io_bytes;
        io.stall_ns = io_stall;
    }

    fn sample_slow_at(&mut self, at: i64) -> SlowHardwareSample {
        let mut sample = SlowHardwareSample::unsupported(at, self.span);
        let cpus = self.topology_cpus.clone();
        for cpu in cpus.iter().filter(|&cpu| cpu < self.span) {
            let dir = self.files.cpu_root.join(format!("cpu{cpu}/cpufreq"));
            let khz = |name: &str| {
                read_trimmed(&dir.join(name))
                    .and_then(|text| parse::decimal(&text))
                    .filter(|&khz| khz > 0)
                    .and_then(|khz| khz.checked_mul(1_000))
            };
            let (constrained, nominal) = (khz("scaling_max_freq"), khz("cpuinfo_max_freq"));
            if let (Some(constrained), Some(nominal)) = (constrained, nominal) {
                sample.cpu_signals[cpu] = CpuSlowSignals {
                    constrained_frequency_hz: Signal::valid(constrained, at),
                    nominal_frequency_hz: Signal::valid(nominal, at),
                    ..CpuSlowSignals::unsupported(at)
                };
            }
        }
        sample.system_signals = SystemSlowSignals {
            thermal_severity: thermal_severity(&self.files.thermal).map_or_else(
                || Signal::unsupported(at),
                |severity| Signal::valid(severity, at),
            ),
            low_power_mode: read_trimmed(&self.files.platform_profile).map_or_else(
                || Signal::unsupported(at),
                |profile| Signal::valid(profile == "low-power", at),
            ),
            ..SystemSlowSignals::unsupported(at)
        };
        sample
    }
}

/// The worst thermal state over the zones that declare a passive, hot or critical trip point.
/// Within 5 °C of the passive trip is fair, at the passive trip serious, at the hot trip critical,
/// and at the critical trip emergency.
fn thermal_severity(thermal: &Path) -> Option<ThermalSeverity> {
    let mut worst: Option<ThermalSeverity> = None;
    for entry in std::fs::read_dir(thermal).into_iter().flatten().flatten() {
        if !entry
            .file_name()
            .to_string_lossy()
            .starts_with("thermal_zone")
        {
            continue;
        }
        let zone = entry.path();
        let millidegrees =
            |name: &str| read_trimmed(&zone.join(name)).and_then(|text| text.parse::<i64>().ok());
        let Some(temperature) = millidegrees("temp") else {
            continue;
        };
        let (mut passive, mut hot, mut critical) = (None::<i64>, None::<i64>, None::<i64>);
        for index in 0.. {
            let Some(kind) = read_trimmed(&zone.join(format!("trip_point_{index}_type"))) else {
                break;
            };
            let Some(trip) = millidegrees(&format!("trip_point_{index}_temp")).filter(|&t| t > 0)
            else {
                continue;
            };
            let slot = match kind.as_str() {
                "passive" => &mut passive,
                "hot" => &mut hot,
                "critical" => &mut critical,
                _ => continue,
            };
            *slot = Some(slot.map_or(trip, |existing| existing.min(trip)));
        }
        if passive.is_none() && hot.is_none() && critical.is_none() {
            continue;
        }
        let reached = |trip: Option<i64>| trip.is_some_and(|trip| temperature >= trip);
        let severity = if reached(critical) {
            ThermalSeverity::Emergency
        } else if reached(hot) {
            ThermalSeverity::Critical
        } else if reached(passive) {
            ThermalSeverity::Serious
        } else if reached(passive.map(|trip| trip - 5_000)) {
            ThermalSeverity::Fair
        } else {
            ThermalSeverity::Nominal
        };
        worst = Some(worst.map_or(severity, |worst| worst.max(severity)));
    }
    worst
}

impl DetailedSnapshotProvider for LinuxResources {
    fn sample_fast(&mut self, requested_at_ns: i64) -> Option<FastHardwareSample> {
        Some(self.sample_fast_at(requested_at_ns))
    }

    fn sample_slow(&mut self, requested_at_ns: i64) -> Option<SlowHardwareSample> {
        Some(self.sample_slow_at(requested_at_ns))
    }
}

/// Legacy snapshots over a provider of their own, so legacy readers and monitors never share
/// interval state. The per-CPU pressure is the run-queue wait since the previous snapshot.
#[derive(Debug)]
pub(crate) struct LinuxSnapshots {
    resources: LinuxResources,
    previous: Option<(i64, Vec<Option<u64>>)>,
}

impl LinuxSnapshots {
    pub(crate) fn new(resources: LinuxResources) -> Self {
        Self {
            resources,
            previous: None,
        }
    }
}

impl SystemSnapshotProvider for LinuxSnapshots {
    fn snapshot(&mut self) -> Option<SystemSnapshot> {
        let at = monotonic_now_ns();
        let sample = self.resources.sample_fast_at(at);
        let waits: Vec<Option<u64>> = sample
            .cpu_signals
            .iter()
            .map(|cpu| cpu.scheduler_wait.get())
            .collect();
        let pressure: Vec<f64> = match &self.previous {
            Some((before, previous)) if at > *before => waits
                .iter()
                .zip(previous)
                .map(|(now, then)| match (now, then) {
                    (Some(now), Some(then)) if now >= then => {
                        (now - then) as f64 / (at - before) as f64
                    }
                    _ => 0.0,
                })
                .collect(),
            _ => vec![0.0; sample.logical_span],
        };
        self.previous = Some((at, waits));
        crate::platform::legacy_snapshot(&sample, &pressure)
    }
}

#[cfg(test)]
mod tests;
