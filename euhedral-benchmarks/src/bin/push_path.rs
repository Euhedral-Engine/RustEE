//! A microbenchmark of one push down the routing path: a vertex that routes to shard vertices, each of which routes to
//! worker edges that end in a receiver. Every frame is pushed from one thread and dropped by the receiver, so the number
//! is the cost of the hops (the loads of each node's routing view and downstream) plus one frame allocation, which
//! `--calibrate` measures alone.
//!
//! ```text
//! push_path [--shards N] [--workers N] [--samples N] [--ops N]
//! ```
//!
//! Each sample times `--ops` pushes (default 2,000,000) and the summary reports nanoseconds per push over the samples
//! (default 40): the mean and P0, P25, P50, P90, P99, P100. Run it alone on the machine.

use euhedral_core::flow_control::{LatticeEdge, LatticeNode, LatticeVertex, UpstreamRegistry};
use euhedral_core::frames::{BenchmarkFrame, Failure, Frame};
use euhedral_core::generics::{DownstreamSlot, LatticeReceiver, LatticeSource, LatticeTerminal};
use euhedral_hardware_utils::util::CpuSet;
use std::hint::black_box;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Instant;

/// Drops what it is pushed.
struct Sink;

impl LatticeTerminal for Sink {
    fn add_upstream(&self, _upstream: Arc<dyn LatticeSource>) {}
}

impl LatticeReceiver for Sink {
    fn push(&self, frame: Box<dyn Frame>) {
        black_box(&frame);
    }
    fn on_complete(&self) {}
    fn on_error(&self, _error: Failure) {}
}

/// A source that is only the identity a slot is attached with.
struct Nothing;

impl LatticeSource for Nothing {
    fn add_downstream(self: Arc<Self>, _downstream: Arc<dyn LatticeReceiver>) {}
    fn pull(
        &self,
        _consumer: &mut dyn FnMut(Box<dyn Frame>),
        _stop: &mut dyn FnMut(&dyn Frame) -> bool,
        _demand: u64,
    ) -> u64 {
        0
    }
    fn request(&self, _demand: u64) {}
    fn complete(&self) {}
    fn is_complete(&self) -> bool {
        false
    }
}

fn number(arguments: &mut impl Iterator<Item = String>, name: &str) -> usize {
    arguments
        .next()
        .and_then(|value| value.parse().ok())
        .unwrap_or_else(|| panic!("{name} takes a number"))
}

fn main() {
    let (mut shards, mut workers, mut samples, mut ops) =
        (4_usize, 4_usize, 40_usize, 2_000_000_u64);
    let mut calibrate = false;
    let mut arguments = std::env::args().skip(1);
    while let Some(flag) = arguments.next() {
        match flag.as_str() {
            "--shards" => shards = number(&mut arguments, "--shards").max(1),
            "--workers" => workers = number(&mut arguments, "--workers").max(1),
            "--samples" => samples = number(&mut arguments, "--samples").max(1),
            "--ops" => ops = number(&mut arguments, "--ops") as u64,
            "--calibrate" => calibrate = true,
            other => panic!("unknown flag {other}"),
        }
    }
    let registry = UpstreamRegistry::with_topology(4, CpuSet::new());
    let global = LatticeVertex::new(&registry, "global", shards);
    let mut shard_handles: Vec<Option<Arc<dyn LatticeNode>>> = Vec::new();
    for shard in 0..shards {
        let vertex = LatticeVertex::new(&registry, format!("shard-{shard}"), workers);
        let mut worker_handles: Vec<Option<Arc<dyn LatticeNode>>> = Vec::new();
        for _ in 0..workers {
            let edge = LatticeEdge::new(&registry, Arc::new(AtomicBool::new(false)));
            edge.add_downstream(Arc::new(Sink));
            worker_handles.push(Some(edge));
        }
        let active: Vec<usize> = (0..workers).collect();
        vertex.set_drain(true);
        assert!(vertex.set_downstream_mapping(&active, &worker_handles));
        vertex.set_drain(false);
        shard_handles.push(Some(vertex));
    }
    let active: Vec<usize> = (0..shards).collect();
    global.set_drain(true);
    assert!(global.set_downstream_mapping(&active, &shard_handles));
    global.set_drain(false);

    let slot = DownstreamSlot::new();
    slot.attach(Arc::new(Nothing), Arc::new(Sink));

    let mut next = 0_u64;
    let mut run = |calibrating: bool| -> f64 {
        let start = Instant::now();
        for _ in 0..ops {
            next = next.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let frame: Box<dyn Frame> = Box::new(BenchmarkFrame::new(next, None));
            if calibrating {
                drop(black_box(frame));
            } else {
                global.push(frame);
            }
        }
        start.elapsed().as_nanos() as f64 / ops as f64
    };
    for _ in 0..3 {
        run(calibrate);
    }
    let mut times: Vec<f64> = (0..samples).map(|_| run(calibrate)).collect();

    // The source's slot read alone: one acquire load, then the receiver.
    let start = Instant::now();
    let reads = ops * 10;
    for _ in 0..reads {
        black_box(slot.receiver());
    }
    let slot_read = start.elapsed().as_nanos() as f64 / reads as f64;

    times.sort_by(f64::total_cmp);
    let mean = times.iter().sum::<f64>() / times.len() as f64;
    let at = |q: f64| times[((times.len() - 1) as f64 * q).round() as usize];
    println!(
        "push_path shards={shards} workers={workers} {} samples={samples} ops={ops} mean_ns={mean:.3} throughput_ops_per_s={:.0} p0={:.3} p25={:.3} p50={:.3} p90={:.3} p99={:.3} p100={:.3} slot_read_ns={slot_read:.3}",
        if calibrate {
            "calibration(alloc+drop only)"
        } else {
            "path"
        },
        1e9 / mean,
        at(0.0),
        at(0.25),
        at(0.5),
        at(0.9),
        at(0.99),
        at(1.0),
    );
}
