//! An idle lattice: what its workers cost, and how often its sources are polled, when there is no work.
//!
//! ```text
//! idle_polls [--lanes N] [--push-only N] [--push-every N] [--seconds S] [--no-pin]
//! ```
//!
//! The lattice gets `--lanes` sources that have nothing to hand over (their `pull` finds 0 and their `request`
//! pushes nothing) and `--push-only` sources whose `pull` always finds 0 and whose `request` pushes a frame on every
//! `--push-every`th call (never, by default). Nothing else runs. For `--seconds` the process samples the CPU time of
//! every worker thread from `/proc/self/task/*/stat` once a second and reads each source's call counters at the end.
//!
//! The summary reports, over (worker thread, second) samples, the share of a CPU each worker used (mean and
//! P0, P25, P50, P90, P99, P100), and over the lane sources the polls per second (mean, throughput, same quantiles).
//! Run it alone on the machine.

use euhedral_benchmarks::high_contention::isolate_harness_core;
use euhedral_core::config::LatticeConfig;
use euhedral_core::control_plane::ControlPlaneLattice;
use euhedral_core::frames::{BenchmarkFrame, Frame};
use euhedral_core::generics::{DownstreamSlot, LatticeReceiver, LatticeSource};
use euhedral_core::impls::DefaultExecutor;
use euhedral_hardware_utils::topology::Topology;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Clock ticks per second of `/proc` CPU times.
const TICKS_PER_SECOND: f64 = 100.0;

/// A source that is polled and counted. A push-only one hands over nothing to a `pull` and pushes from `request`.
struct Counting {
    downstream: DownstreamSlot,
    pulls: AtomicU64,
    requests: AtomicU64,
    /// Push a frame on every this many requests; zero never pushes.
    push_every: u64,
    next_id: AtomicU64,
}

impl Counting {
    fn new(push_every: u64) -> Arc<Self> {
        Arc::new(Self {
            downstream: DownstreamSlot::new(),
            pulls: AtomicU64::new(0),
            requests: AtomicU64::new(0),
            push_every,
            next_id: AtomicU64::new(1),
        })
    }

    fn polls(&self) -> u64 {
        self.pulls.load(Ordering::Relaxed) + self.requests.load(Ordering::Relaxed)
    }
}

impl LatticeSource for Counting {
    fn add_downstream(self: Arc<Self>, downstream: Arc<dyn LatticeReceiver>) {
        let source: Arc<dyn LatticeSource> = self.clone();
        self.downstream.attach(source, downstream);
    }

    fn pull(
        &self,
        _consumer: &mut dyn FnMut(Box<dyn Frame>),
        _stop: &mut dyn FnMut(&dyn Frame) -> bool,
        _demand: u64,
    ) -> u64 {
        // One thread at a time serves a source, so the counters need no read-modify-write.
        let pulls = self.pulls.load(Ordering::Relaxed);
        self.pulls.store(pulls + 1, Ordering::Relaxed);
        0
    }

    fn request(&self, _demand: u64) {
        let requests = self.requests.load(Ordering::Relaxed) + 1;
        self.requests.store(requests, Ordering::Relaxed);
        if self.push_every != 0
            && requests.is_multiple_of(self.push_every)
            && let Some(receiver) = self.downstream.receiver()
        {
            let id = self.next_id.fetch_add(1, Ordering::Relaxed);
            let mut frame = BenchmarkFrame::new(id, None);
            frame.header_mut().randomize_hash(id);
            receiver.push(Box::new(frame));
        }
    }

    fn complete(&self) {
        self.downstream.complete();
    }

    fn is_complete(&self) -> bool {
        self.downstream.is_complete()
    }
}

/// The CPU ticks (user + system) of every thread of this process, by thread id.
fn thread_ticks() -> BTreeMap<u32, u64> {
    let mut ticks = BTreeMap::new();
    let Ok(entries) = std::fs::read_dir("/proc/self/task") else {
        return ticks;
    };
    for entry in entries.flatten() {
        let Some(tid) = entry
            .file_name()
            .to_str()
            .and_then(|name| name.parse().ok())
        else {
            continue;
        };
        let Ok(stat) = std::fs::read_to_string(entry.path().join("stat")) else {
            continue;
        };
        // The thread's name may hold spaces and parentheses; the fields start after its last `)`.
        let Some(rest) = stat.rfind(')').map(|end| &stat[end + 2..]) else {
            continue;
        };
        let fields: Vec<&str> = rest.split(' ').collect();
        // `rest` starts at field 3 (state); utime and stime are fields 14 and 15.
        let (Some(user), Some(system)) = (
            fields.get(11).and_then(|v| v.parse::<u64>().ok()),
            fields.get(12).and_then(|v| v.parse::<u64>().ok()),
        ) else {
            continue;
        };
        ticks.insert(tid, user + system);
    }
    ticks
}

/// The voluntary and involuntary context switches of every thread of this process, by thread id.
fn thread_switches() -> BTreeMap<u32, (u64, u64)> {
    let mut switches = BTreeMap::new();
    let Ok(entries) = std::fs::read_dir("/proc/self/task") else {
        return switches;
    };
    for entry in entries.flatten() {
        let Some(tid) = entry
            .file_name()
            .to_str()
            .and_then(|name| name.parse().ok())
        else {
            continue;
        };
        let Ok(status) = std::fs::read_to_string(entry.path().join("status")) else {
            continue;
        };
        let field = |name: &str| {
            status
                .lines()
                .find_map(|line| line.strip_prefix(name))
                .and_then(|value| value.trim().parse::<u64>().ok())
        };
        if let (Some(voluntary), Some(involuntary)) = (
            field("voluntary_ctxt_switches:"),
            field("nonvoluntary_ctxt_switches:"),
        ) {
            switches.insert(tid, (voluntary, involuntary));
        }
    }
    switches
}

struct Summary {
    mean: f64,
    quantiles: [f64; 6],
}

fn summarize(mut values: Vec<f64>) -> Summary {
    values.sort_by(f64::total_cmp);
    if values.is_empty() {
        return Summary {
            mean: f64::NAN,
            quantiles: [f64::NAN; 6],
        };
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let at = |q: f64| values[((values.len() - 1) as f64 * q).round() as usize];
    Summary {
        mean,
        quantiles: [at(0.0), at(0.25), at(0.5), at(0.9), at(0.99), at(1.0)],
    }
}

fn fields(summary: &Summary) -> String {
    let [p0, p25, p50, p90, p99, p100] = summary.quantiles;
    format!(
        "avg={:.4} p0={p0:.4} p25={p25:.4} p50={p50:.4} p90={p90:.4} p99={p99:.4} p100={p100:.4}",
        summary.mean
    )
}

fn number(arguments: &mut impl Iterator<Item = String>, name: &str) -> f64 {
    arguments
        .next()
        .and_then(|value| value.parse().ok())
        .unwrap_or_else(|| panic!("{name} takes a number"))
}

#[allow(
    clippy::disallowed_methods,
    reason = "JAVA-BACKOFF: the harness thread waits between samples of the workers' CPU time, outside any worker's cycle"
)]
fn main() {
    let (mut lanes, mut push_only, mut push_every, mut seconds, mut pin) =
        (16_usize, 0_usize, 0_u64, 60_u64, true);
    let mut arguments = std::env::args().skip(1);
    while let Some(flag) = arguments.next() {
        match flag.as_str() {
            "--lanes" => lanes = number(&mut arguments, "--lanes") as usize,
            "--push-only" => push_only = number(&mut arguments, "--push-only") as usize,
            "--push-every" => push_every = number(&mut arguments, "--push-every") as u64,
            "--seconds" => seconds = (number(&mut arguments, "--seconds") as u64).max(2),
            "--no-pin" => pin = false,
            other => panic!("unknown flag {other}"),
        }
    }

    let topology = Topology::system();
    let isolation = isolate_harness_core(topology, pin).expect("a CPU for the workers");
    let before_start = thread_ticks().into_keys().collect::<BTreeSet<_>>();
    let mut config = LatticeConfig::with_executor(DefaultExecutor::new());
    config.allowed_cpus = isolation.workers;
    config.startup_timeout = Duration::from_secs(60);
    config.shutdown_timeout = Duration::from_secs(60);
    let lattice = ControlPlaneLattice::new(config).expect("a lattice");
    lattice.start().expect("a started lattice");
    let lane_sources: Vec<_> = (0..lanes).map(|_| Counting::new(0)).collect();
    let push_sources: Vec<_> = (0..push_only).map(|_| Counting::new(push_every)).collect();
    for source in lane_sources.iter().chain(&push_sources) {
        lattice
            .add_upstream(source.clone())
            .expect("an attached source");
    }
    let workers: BTreeSet<u32> = thread_ticks()
        .into_keys()
        .filter(|tid| !before_start.contains(tid))
        .collect();

    // Let the workers settle, then take the window.
    std::thread::sleep(Duration::from_secs(2));
    let polls_before: Vec<u64> = lane_sources.iter().map(|source| source.polls()).collect();
    let push_before: Vec<u64> = push_sources.iter().map(|source| source.polls()).collect();
    let mut last = thread_ticks();
    let first = last.clone();
    let switches_first = thread_switches();
    let start = Instant::now();
    let mut last_at = start;
    let mut shares = Vec::new();
    for second in 1..=seconds {
        let wake = start + Duration::from_secs(second);
        std::thread::sleep(wake.saturating_duration_since(Instant::now()));
        let now = Instant::now();
        let ticks = thread_ticks();
        let elapsed = now.duration_since(last_at).as_secs_f64();
        for tid in &workers {
            let delta = ticks.get(tid).copied().unwrap_or(0) - last.get(tid).copied().unwrap_or(0);
            shares.push(delta as f64 / TICKS_PER_SECOND / elapsed);
        }
        last = ticks;
        last_at = now;
    }
    let window = start.elapsed().as_secs_f64();
    let switches_last = thread_switches();
    let per_worker_cpu_s: Vec<f64> = workers
        .iter()
        .map(|tid| {
            (last.get(tid).copied().unwrap_or(0) - first.get(tid).copied().unwrap_or(0)) as f64
                / TICKS_PER_SECOND
        })
        .collect();
    // The threads that did work: more than half a CPU-second over the window.
    let active: Vec<u32> = workers
        .iter()
        .zip(&per_worker_cpu_s)
        .filter(|(_, cpu)| **cpu > 0.5)
        .map(|(tid, _)| *tid)
        .collect();
    let active_cpu_s: Vec<f64> = per_worker_cpu_s
        .iter()
        .copied()
        .filter(|cpu| *cpu > 0.5)
        .collect();
    let (voluntary, involuntary) = workers.iter().fold((0, 0), |(v, i), tid| {
        let (v0, i0) = switches_first.get(tid).copied().unwrap_or_default();
        let (v1, i1) = switches_last.get(tid).copied().unwrap_or_default();
        (v + v1.saturating_sub(v0), i + i1.saturating_sub(i0))
    });
    let polls: Vec<f64> = lane_sources
        .iter()
        .zip(&polls_before)
        .map(|(source, before)| (source.polls() - before) as f64 / window)
        .collect();
    let push_polls: Vec<f64> = push_sources
        .iter()
        .zip(&push_before)
        .map(|(source, before)| (source.polls() - before) as f64 / window)
        .collect();

    println!(
        "idle_polls lanes={lanes} push_only={push_only} push_every={push_every} seconds={seconds} workers={} window_s={window:.2}",
        workers.len()
    );
    println!(
        "worker_cpu_share (CPUs per worker per second, over {} worker-second samples) {}",
        shares.len(),
        fields(&summarize(shares))
    );
    println!(
        "worker_cpu_total_s (CPU seconds per worker over the window) {} sum={:.3}",
        fields(&summarize(per_worker_cpu_s.clone())),
        per_worker_cpu_s.iter().sum::<f64>()
    );
    println!(
        "active_threads (CPU > 0.5 s) count={} cpu_total_s {} sum={:.3}",
        active.len(),
        fields(&summarize(active_cpu_s.clone())),
        active_cpu_s.iter().sum::<f64>()
    );
    println!(
        "context_switches_per_s (all new threads) voluntary={:.0} involuntary={:.0}",
        voluntary as f64 / window,
        involuntary as f64 / window
    );
    println!(
        "lane_polls_per_s (per lane source) {} throughput_total={:.0}",
        fields(&summarize(polls.clone())),
        polls.iter().sum::<f64>()
    );
    if !push_polls.is_empty() {
        println!(
            "push_only_polls_per_s {} throughput_total={:.0}",
            fields(&summarize(push_polls.clone())),
            push_polls.iter().sum::<f64>()
        );
    }

    for source in lane_sources.iter().chain(&push_sources) {
        source.complete();
    }
    lattice.close();
}
