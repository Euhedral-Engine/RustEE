//! A timing probe: how late a lone `Completions::at` timer is seen on a one-CPU lattice, idle and busy, with the poll
//! frame's pacing; and how often a handle is visited. It prints quantiles of the lateness (microseconds) and the visit
//! interval, and asserts only that each timer is seen. Run it with
//! `cargo test --release -p euhedral-graph --test timer_latency -- --ignored --nocapture`.
//!
//! The deterministic claims (a timer is not polled before its deadline and is polled at the first visit after it; a
//! blind wait is polled at most once per window) are unit tests in `completions.rs`; this one shows them on a lattice.
#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use euhedral_core::config::LatticeConfig;
use euhedral_core::control_plane::ControlPlaneLattice;
use euhedral_core::frames::{BenchmarkFrame, Frame, RunnableFrame};
use euhedral_core::generics::{DownstreamSlot, LatticeReceiver, LatticeSource};
use euhedral_core::impls::DefaultExecutor;
use euhedral_graph::Completions;
use euhedral_hardware_utils::affinity;
use euhedral_hardware_utils::util::CpuSet;

/// Idle source counting its visits.
struct Visits {
    downstream: DownstreamSlot,
    visits: AtomicU64,
}

impl LatticeSource for Visits {
    fn add_downstream(self: Arc<Self>, downstream: Arc<dyn LatticeReceiver>) {
        let s: Arc<dyn LatticeSource> = self.clone();
        self.downstream.attach(s, downstream);
    }
    fn pull(
        &self,
        _: &mut dyn FnMut(Box<dyn Frame>),
        _: &mut dyn FnMut(&dyn Frame) -> bool,
        _: u64,
    ) -> u64 {
        self.visits.fetch_add(1, Ordering::Relaxed);
        0
    }
    fn request(&self, _: u64) {
        self.visits.fetch_add(1, Ordering::Relaxed);
    }
    fn complete(&self) {
        self.downstream.complete();
    }
    fn is_complete(&self) -> bool {
        self.downstream.is_complete()
    }
}

/// Endless load: frames that spin `work` each, handed to pulls and pushed for requests.
struct Load {
    downstream: DownstreamSlot,
    on: AtomicBool,
    next: AtomicU64,
    work: Duration,
}

impl Load {
    fn frame(&self) -> Box<dyn Frame> {
        let id = self.next.fetch_add(1, Ordering::Relaxed);
        let work = self.work;
        let mut f = BenchmarkFrame::new(id, None);
        f.header_mut().randomize_hash(id);
        let _ = work;
        let mut r = RunnableFrame::new(
            id,
            Arc::new(move || {
                let end = Instant::now() + work;
                while Instant::now() < end {
                    std::hint::spin_loop();
                }
            }),
        );
        r.header_mut().randomize_hash(id);
        drop(f);
        Box::new(r)
    }
}

impl LatticeSource for Load {
    fn add_downstream(self: Arc<Self>, downstream: Arc<dyn LatticeReceiver>) {
        let s: Arc<dyn LatticeSource> = self.clone();
        self.downstream.attach(s, downstream);
    }
    fn pull(
        &self,
        consumer: &mut dyn FnMut(Box<dyn Frame>),
        _: &mut dyn FnMut(&dyn Frame) -> bool,
        demand: u64,
    ) -> u64 {
        if !self.on.load(Ordering::Relaxed) {
            return 0;
        }
        let n = demand.min(64);
        for _ in 0..n {
            consumer(self.frame());
        }
        n
    }
    fn request(&self, demand: u64) {
        if !self.on.load(Ordering::Relaxed) {
            return;
        }
        if let Some(r) = self.downstream.receiver() {
            for _ in 0..demand.min(64) {
                r.push(self.frame());
            }
        }
    }
    fn complete(&self) {
        self.downstream.complete();
    }
    fn is_complete(&self) -> bool {
        self.downstream.is_complete()
    }
}

fn q(mut v: Vec<f64>) -> String {
    v.sort_by(f64::total_cmp);
    let at = |p: f64| v[((v.len() - 1) as f64 * p).round() as usize];
    let avg = v.iter().sum::<f64>() / v.len() as f64;
    format!(
        "avg={avg:.0} p0={:.0} p25={:.0} p50={:.0} p90={:.0} p99={:.0} p100={:.0}",
        at(0.0),
        at(0.25),
        at(0.5),
        at(0.9),
        at(0.99),
        at(1.0)
    )
}

fn timers(label: &str, completions: &Arc<Completions>, dummies: &[Arc<Visits>]) {
    for d_us in [100u64, 400, 1600, 10_000] {
        let reps = if d_us >= 10_000 { 15 } else { 40 };
        let mut late = Vec::new();
        let v0: u64 = dummies
            .iter()
            .map(|d| d.visits.load(Ordering::Relaxed))
            .sum();
        let t0 = Instant::now();
        for i in 0..reps {
            let ran: Arc<Mutex<Option<Instant>>> = Arc::default();
            let r = ran.clone();
            let mut f = RunnableFrame::new(
                i,
                Arc::new(move || {
                    *r.lock().unwrap() = Some(Instant::now());
                }),
            );
            f.header_mut().randomize_hash(i + 1);
            let deadline = Instant::now() + Duration::from_micros(d_us);
            completions.at(deadline, Box::new(f));
            let give_up = deadline + Duration::from_secs(2);
            let at = loop {
                if let Some(at) = *ran.lock().unwrap() {
                    break at;
                }
                assert!(Instant::now() < give_up, "a timer was not seen in 2 s");
                std::thread::sleep(Duration::from_micros(50));
            };
            late.push(at.saturating_duration_since(deadline).as_micros() as f64);
        }
        let secs = t0.elapsed().as_secs_f64();
        let v1: u64 = dummies
            .iter()
            .map(|d| d.visits.load(Ordering::Relaxed))
            .sum();
        let per_handle = (v1 - v0) as f64 / dummies.len() as f64 / secs;
        eprintln!(
            "{label} timer {d_us} us: lateness_us {} | visits/s per idle handle {:.0} (interval {:.1} us)",
            q(late),
            per_handle,
            1e6 / per_handle.max(1.0)
        );
    }
    let (p50, p99, max) = completions.lateness_us();
    eprintln!("{label} Completions::lateness_us (cumulative) p50={p50} p99={p99} max={max}");
}

#[test]
#[ignore = "a timing probe: run it with --ignored --nocapture"]
fn timer_latency() {
    let cpu = affinity::base_mask().first().unwrap();
    let mut config = LatticeConfig::with_executor(DefaultExecutor::new());
    config.name = "timer-latency".into();
    config.shard_name = "timer-latency".into();
    config.allowed_cpus = CpuSet::range(cpu, cpu + 1);
    config.startup_timeout = Duration::from_secs(10);
    config.shutdown_timeout = Duration::from_secs(5);
    let lattice = ControlPlaneLattice::new(config).unwrap();
    lattice.start().unwrap();
    let (pool, _devices) = euhedral_graph::testing::FakeDevice::pool(&["a"]);
    let completions = Completions::new(pool);
    lattice.add_upstream(completions.clone()).unwrap();
    let dummies: Vec<Arc<Visits>> = (0..3)
        .map(|_| {
            Arc::new(Visits {
                downstream: DownstreamSlot::new(),
                visits: AtomicU64::new(0),
            })
        })
        .collect();
    for d in &dummies {
        lattice.add_upstream(d.clone()).unwrap();
    }
    let load = Arc::new(Load {
        downstream: DownstreamSlot::new(),
        on: AtomicBool::new(false),
        next: AtomicU64::new(1),
        work: Duration::from_micros(2),
    });
    lattice.add_upstream(load.clone()).unwrap();
    std::thread::sleep(Duration::from_millis(500));
    timers("idle", &completions, &dummies);
    load.on.store(true, Ordering::Relaxed);
    std::thread::sleep(Duration::from_millis(200));
    timers("busy(2us frames)", &completions, &dummies);
    load.on.store(false, Ordering::Relaxed);
    load.complete();
    for d in &dummies {
        d.complete();
    }
    lattice.close();
}
