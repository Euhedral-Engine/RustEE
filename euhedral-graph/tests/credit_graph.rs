//! A credit-graph on a real lattice with [`FakeDevice`] lanes: the proof that the crate's pieces
//! compose without anything waiting.
//!
//! - **One lattice in greedy mode** (4 workers, park 100 µs). Nothing in a frame waits: every stage
//!   submits to a fake device lane and returns.
//! - **Per item, a `StageGraph`:** fetch (ordered, disk) -completion-> parse (ordered, parse queue)
//!   -submission-> compute (unordered, either compute queue). The compute's submission completes
//!   the item's `Sequencer` slot, whose in-order release offers the pack (ordered), whose
//!   completion sends (ordered), whose completion puts the packet on the wire.
//! - **The pool lease** of an item is a `Join` of its retirement (boundary reached) and its
//!   sent-or-dropped end; its continuation frees the pool slot, which may admit the next item.
//! - **Admission** = credit horizon and pool (16), from the control lane (credits, seeks) and from
//!   lease releases.
//! - **The fake consumer** (the harness, on the test thread) plays 120 items a second, credits 1 s
//!   ahead, starts after 0.5 s, seeks at 1.5 s to seq 480 (a new generation, with a new kill
//!   switch), pauses 0.4 s and plays to the end of the 6 s stream.
//! - **Order at every hop** (fetch, parse, sequencer release, pack, send, sent) is checked within
//!   each generation.
//! - **The no-wait check:** a timing executor measures every frame callback (execute + finalizer)
//!   and counts the ones over 200 µs that the OS did not preempt (no involuntary context switch
//!   during it).
//!
//! Pass: 0 late, 0 out of order, 0 beyond credit, 0 left (items, slots, completions, quanta), made
//! = sent + dropped, in flight <= pool, no callback over 200 µs, shutdown under 100 ms.
//!
//! `EUHEDRAL_TEST_CPUS` (default 0,1,2,3) picks the lattice's CPUs. As in `euhedral-core`'s tests
//! that need particular hardware, the test passes without checking anything when this process may
//! not run on all of them. It needs Linux (per-thread context-switch counts).

#![cfg(target_os = "linux")]
// The fake consumer is test harness, not a frame: it looks at the wire every millisecond.
#![allow(clippy::disallowed_methods)]

use std::collections::{BTreeMap, VecDeque};
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, PoisonError, Weak};
use std::time::{Duration, Instant};

use euhedral_core::config::{FragmentConfig, IdlePolicy, LatticeConfig};
use euhedral_core::control_plane::ControlPlaneLattice;
use euhedral_core::frames::{Frame, FrameError, FrameHeader};
use euhedral_core::generics::AbstractExecutor;
use euhedral_core::generics::LatticeSource;
use euhedral_core::impls::BaseCloneableObject;
use euhedral_core::ingest::{IngestSink, QueueIngestSink};
use euhedral_graph::testing::FakeDevice;
use euhedral_graph::{
    Completions, DEFAULT_RETRY_BACKSTOP, EdgeKind, GraphRuntime, Join, Lake, LaneId, LanePool,
    Marker, QuantumOutcome, Sequencer, Slot, Stage, StageCx, StageError, StageGraph,
    StageGraphSpec, StageSpec, Submitted,
};
use euhedral_hardware_utils::affinity;

const FPS: u64 = 120;
const FRAME: Duration = Duration::from_nanos(1_000_000_000 / FPS);
const END: u64 = 6 * FPS;
const AHEAD: u64 = FPS;
const START_BUFFER: u64 = FPS / 2;
const SEEK_AT: Duration = Duration::from_millis(1500);
const SEEK_TO: u64 = 4 * FPS;
const PAUSE_AT_PTS: u64 = SEEK_TO + FPS / 2;
const PAUSE_FOR: Duration = Duration::from_millis(400);
const POOL: u64 = 16;
/// The sinks of the stage lake.
const STAGE_SINKS: NonZeroUsize = NonZeroUsize::new(2).unwrap();

const FETCH: Duration = Duration::from_micros(200);
const PARSE: Duration = Duration::from_micros(1500);
const COMPUTE: Duration = Duration::from_micros(6000);
const SEND: Duration = Duration::from_micros(300);

const SOURCE: u64 = 0x50e1;
const PARSER: u64 = 0xdec0;
const PACKER: u64 = 0xec0d;
const SOCKET: u64 = 0x50c4;
const CONTROL: u64 = 0xc7a1;

/// The lanes of the fake device.
const DISK: u16 = 0;
const PARSE_Q: u16 = 1;
const COMPUTE_A: u16 = 2;
const COMPUTE_B: u16 = 3;
const PACK_Q: u16 = 4;
const NIC: u16 = 5;

/// The longest frame callback the no-wait rule allows.
const SLOW: Duration = Duration::from_micros(200);

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

// ---------------------------------------------------------------------------------------------
// The timing executor: the stock lifecycle, with every callback measured.

#[derive(Default)]
struct CallbackStats {
    count: AtomicU64,
    max_ns: AtomicU64,
    /// Over 200 µs with no involuntary context switch: a frame that waited or worked too long.
    slow: AtomicU64,
    /// Over 200 µs, preempted by the OS (another process had the CPU): reported, not a defect of
    /// the graph.
    preempted: AtomicU64,
    /// Callbacks that blocked at all (a voluntary context switch inside them: a contended lock, a
    /// syscall wait).
    voluntary: AtomicU64,
}

fn stats() -> &'static CallbackStats {
    static STATS: OnceLock<CallbackStats> = OnceLock::new();
    STATS.get_or_init(CallbackStats::default)
}

/// (voluntary, involuntary) context switches of this thread so far.
fn switches() -> (i64, i64) {
    // SAFETY: getrusage writes one `rusage` we own; RUSAGE_THREAD is this thread.
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    // SAFETY: as above.
    unsafe { libc::getrusage(libc::RUSAGE_THREAD, &mut usage) };
    (usage.ru_nvcsw, usage.ru_nivcsw)
}

#[derive(Clone, Copy, Default)]
struct TimingExecutor {
    cpu: Option<usize>,
}

impl AbstractExecutor for TimingExecutor {
    fn cpu(&self) -> Option<usize> {
        self.cpu
    }

    fn execute(&self, frame: &mut dyn Frame) -> Result<(), FrameError> {
        frame.execute()
    }

    fn hook_on_clone(&self, cpu: Option<usize>) -> Self {
        Self { cpu }
    }

    fn run(&self, mut frame: Box<dyn Frame>) {
        let (v0, i0) = switches();
        let started = Instant::now();
        if !frame.is_alive() {
            frame.do_finally();
        } else {
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| frame.execute())) {
                Ok(Ok(())) | Ok(Err(FrameError::Cancelled)) => frame.do_finally(),
                Ok(Err(error)) => frame.do_finally_with_error(error),
                Err(_) => frame.do_finally_with_error(FrameError::failed("the frame panicked")),
            }
        }
        let took = started.elapsed();
        let (v1, i1) = switches();
        let s = stats();
        s.count.fetch_add(1, Ordering::Relaxed);
        s.max_ns.fetch_max(
            u64::try_from(took.as_nanos()).unwrap_or(u64::MAX),
            Ordering::Relaxed,
        );
        if v1 > v0 {
            s.voluntary.fetch_add(1, Ordering::Relaxed);
        }
        if took > SLOW {
            if i1 > i0 && v1 == v0 {
                &s.preempted
            } else {
                &s.slow
            }
            .fetch_add(1, Ordering::Relaxed);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Order checks at each hop (diagnostics: where an inversion first appears).

const HOP_NAMES: [&str; 6] = ["fetch", "parse", "release", "pack", "send", "sent"];

struct Hops {
    last: Mutex<[(u32, u64); 6]>,
    inversions: [AtomicU64; 6],
}

fn hops() -> &'static Hops {
    static HOPS: OnceLock<Hops> = OnceLock::new();
    HOPS.get_or_init(|| Hops {
        last: Mutex::new([(0, 0); 6]),
        inversions: Default::default(),
    })
}

fn hop(index: usize, generation: u32, seq: u64) {
    let hops = hops();
    let mut last = lock(&hops.last);
    let (e, p) = last[index];
    if e == generation && seq != p + 1 {
        hops.inversions[index].fetch_add(1, Ordering::Relaxed);
    }
    last[index] = (generation, seq);
}

// ---------------------------------------------------------------------------------------------
// The items and their graph.

/// What a released sequencer slot hands to the pack: the item and its compute's marker.
struct Staged {
    generation: u32,
    seq: u64,
    kill: Arc<AtomicBool>,
    marker: Marker,
    lease: Arc<Join<()>>,
}

/// The quantum: one item of one generation.
struct Item {
    generation: u32,
    seq: u64,
    kill: Arc<AtomicBool>,
    slot: Mutex<Option<Slot<Staged>>>,
    /// Part 0: the quantum retired. Part 1: the item was sent (a value) or dropped after admission
    /// (none).
    lease: Arc<Join<()>>,
}

impl Item {
    /// The item will never reach the sequencer: its slot is abandoned and its sent part ends empty.
    fn drop_slot(&self) {
        if let Some(slot) = lock(&self.slot).take() {
            slot.abandon();
            self.lease.abandon(1);
        }
    }
}

struct Device {
    lanes: Vec<Arc<FakeDevice>>,
}

impl Device {
    fn submit(&self, lane: u16, cost: Duration, waits: &[Marker]) -> Marker {
        self.lanes[usize::from(lane)].submit(cost, waits)
    }
}

struct Fetch(Arc<Device>);

impl Stage<Item> for Fetch {
    fn run(&self, item: &Item, cx: &StageCx<'_>) -> Result<Submitted, StageError> {
        hop(0, item.generation, item.seq);
        Ok(Submitted::marker(self.0.submit(DISK, FETCH, cx.waits)))
    }

    fn abandoned(&self, item: &Item) {
        item.drop_slot();
    }
}

struct Parse(Arc<Device>);

impl Stage<Item> for Parse {
    fn run(&self, item: &Item, cx: &StageCx<'_>) -> Result<Submitted, StageError> {
        hop(1, item.generation, item.seq);
        Ok(Submitted::marker(self.0.submit(PARSE_Q, PARSE, cx.waits)))
    }

    fn abandoned(&self, item: &Item) {
        item.drop_slot();
    }
}

struct Compute(Arc<Device>);

impl Stage<Item> for Compute {
    fn run(&self, item: &Item, cx: &StageCx<'_>) -> Result<Submitted, StageError> {
        let lane = cx.lane.expect("a compute lane").0;
        let marker = self.0.submit(lane, COMPUTE, cx.waits);
        if let Some(slot) = lock(&item.slot).take() {
            let Item {
                generation,
                seq,
                ref kill,
                ref lease,
                ..
            } = *item;
            slot.complete(Staged {
                generation,
                seq,
                kill: kill.clone(),
                marker,
                lease: lease.clone(),
            });
        }
        Ok(Submitted::marker(marker))
    }

    fn abandoned(&self, item: &Item) {
        item.drop_slot();
    }
}

// ---------------------------------------------------------------------------------------------
// The context, admission and the frames after the sequencer.

#[derive(Default)]
struct Counts {
    made: AtomicU64,
    sent: AtomicU64,
    dropped: AtomicU64,
    retired: AtomicU64,
    killed_outcomes: AtomicU64,
    failed_outcomes: AtomicU64,
    max_in_flight: AtomicU64,
}

#[derive(Debug)]
enum Msg {
    Packet { generation: u32, seq: u64 },
    End { generation: u32 },
}

struct Admission {
    generation: u32,
    kill: Arc<AtomicBool>,
    next: u64,
    horizon: u64,
    in_flight: u64,
    eos_sent: bool,
    closed: bool,
}

struct Ctx {
    /// The lake the stage graph publishes into; the scenario admits one unit for its whole run.
    stages: Arc<Lake>,
    control: QueueIngestSink,
    completions: Arc<Completions>,
    device: Arc<Device>,
    graph: OnceLock<Arc<StageGraph<Item>>>,
    sequencer: OnceLock<Arc<Sequencer<Staged>>>,
    link: Mutex<Admission>,
    delivering: AtomicBool,
    again: AtomicBool,
    wire: Mutex<VecDeque<Msg>>,
    counts: Counts,
    pack_cost: Duration,
    me: Weak<Ctx>,
}

impl Ctx {
    fn new(
        lattice: Arc<ControlPlaneLattice>,
        pack_cost: Duration,
        lanes: Arc<LanePool>,
        devices: Vec<Arc<FakeDevice>>,
    ) -> Arc<Self> {
        let pool = lanes.clone();
        let ctx = Arc::new_cyclic(|me| Ctx {
            stages: Lake::new(lattice, STAGE_SINKS, Duration::ZERO),
            control: QueueIngestSink::new(),
            completions: Completions::new(lanes),
            device: Arc::new(Device { lanes: devices }),
            graph: OnceLock::new(),
            sequencer: OnceLock::new(),
            link: Mutex::new(Admission {
                generation: 0,
                kill: Arc::new(AtomicBool::new(false)),
                next: 0,
                horizon: 0,
                in_flight: 0,
                eos_sent: false,
                closed: false,
            }),
            delivering: AtomicBool::new(false),
            again: AtomicBool::new(false),
            wire: Mutex::new(VecDeque::new()),
            counts: Counts::default(),
            pack_cost,
            me: me.clone(),
        });
        let me = ctx.me.clone();
        let sequencer = Sequencer::new(move |ready: Staged| {
            let Some(ctx) = me.upgrade() else { return };
            hop(2, ready.generation, ready.seq);
            let frame = Pack {
                header: FrameHeader::with_kill_switch(PACKER, Some(ready.kill.clone())),
                ctx: ctx.clone(),
                ready: Some(ready),
            };
            ctx.offer(Box::new(frame));
        });
        assert!(ctx.sequencer.set(sequencer).is_ok());
        let mut spec = StageGraphSpec::new();
        let fetch = spec.stage(StageSpec {
            name: "fetch",
            ordered: Some(SOURCE),
            lanes: vec![LaneId(DISK)],
        });
        let parse = spec.stage(StageSpec {
            name: "parse",
            ordered: Some(PARSER),
            lanes: vec![LaneId(PARSE_Q)],
        });
        let compute = spec.stage(StageSpec {
            name: "compute",
            ordered: None,
            lanes: vec![LaneId(COMPUTE_A), LaneId(COMPUTE_B)],
        });
        spec.edge(fetch, parse, EdgeKind::Completion)
            .edge(parse, compute, EdgeKind::Submission);
        let device = ctx.device.clone();
        let runtime = GraphRuntime {
            stages: ctx.stages.clone(),
            completions: ctx.completions.clone(),
            lanes: pool,
            retries: Default::default(),
            retry_backstop: DEFAULT_RETRY_BACKSTOP,
        };
        let me = ctx.me.clone();
        let graph = StageGraph::new(
            spec,
            vec![
                Arc::new(Fetch(device.clone())),
                Arc::new(Parse(device.clone())),
                Arc::new(Compute(device)),
            ],
            runtime,
            move |item: Item, outcome| {
                let Some(ctx) = me.upgrade() else { return };
                ctx.counts.retired.fetch_add(1, Ordering::Relaxed);
                match outcome {
                    QuantumOutcome::Completed => {}
                    QuantumOutcome::Killed => {
                        _ = ctx.counts.killed_outcomes.fetch_add(1, Ordering::Relaxed)
                    }
                    QuantumOutcome::Failed { .. } => {
                        _ = ctx.counts.failed_outcomes.fetch_add(1, Ordering::Relaxed)
                    }
                }
                item.drop_slot();
                item.lease.arrive(0, ());
            },
        )
        .expect("a valid graph");
        assert!(ctx.graph.set(graph).is_ok());
        ctx
    }

    fn offer(&self, frame: Box<dyn Frame>) {
        if let Err(frame) = self.stages.offer(frame) {
            frame.do_finally();
        }
    }

    /// Admits items while credit, the pool and the file allow: one admitting thread at a time, so
    /// sequencer slots are registered in seq order.
    fn deliver(&self) {
        loop {
            self.again.store(true, Ordering::Release);
            if self.delivering.swap(true, Ordering::AcqRel) {
                return;
            }
            while self.again.swap(false, Ordering::AcqRel) {
                while let Some((item, kill)) = self.make() {
                    let graph = self.graph.get().expect("the graph");
                    let _ = graph.start(item, kill, Vec::new());
                }
            }
            self.delivering.store(false, Ordering::Release);
            if !self.again.load(Ordering::Acquire) {
                return;
            }
        }
    }

    fn make(&self) -> Option<(Item, Arc<AtomicBool>)> {
        let mut link = lock(&self.link);
        if link.closed || link.next >= END.min(link.horizon) || link.in_flight >= POOL {
            return None;
        }
        let seq = link.next;
        link.next += 1;
        link.in_flight += 1;
        self.counts.made.fetch_add(1, Ordering::Relaxed);
        self.counts
            .max_in_flight
            .fetch_max(link.in_flight, Ordering::Relaxed);
        let me = self.me.clone();
        let lease = Join::new(2, move |parts: Vec<Option<()>>| {
            if let Some(ctx) = me.upgrade() {
                ctx.release(parts[1].is_some());
            }
        });
        let slot = self.sequencer.get().expect("the sequencer").register();
        let item = Item {
            generation: link.generation,
            seq,
            kill: link.kill.clone(),
            slot: Mutex::new(Some(slot)),
            lease,
        };
        Some((item, link.kill.clone()))
    }

    /// A item's lease ended (retired, and sent or dropped): its pool slot is free, which may admit
    /// the next.
    fn release(&self, sent: bool) {
        if sent {
            &self.counts.sent
        } else {
            &self.counts.dropped
        }
        .fetch_add(1, Ordering::Relaxed);
        let eos = {
            let mut link = lock(&self.link);
            link.in_flight -= 1;
            let eos = link.next >= END && link.in_flight == 0 && !link.eos_sent;
            link.eos_sent |= eos;
            eos.then_some(link.generation)
        };
        if let Some(generation) = eos {
            lock(&self.wire).push_back(Msg::End { generation });
        }
        self.deliver();
    }

    fn on_credit(&self, generation: u32, horizon: u64) {
        {
            let mut link = lock(&self.link);
            if generation != link.generation || horizon <= link.horizon {
                return;
            }
            link.horizon = horizon;
        }
        self.deliver();
    }

    fn seek(&self, generation: u32, to: u64, horizon: u64) {
        {
            let mut link = lock(&self.link);
            link.kill.store(true, Ordering::Release);
            link.kill = Arc::new(AtomicBool::new(false));
            link.generation = generation;
            link.next = to;
            link.horizon = horizon;
            link.eos_sent = false;
        }
        self.deliver();
    }

    fn in_flight(&self) -> u64 {
        lock(&self.link).in_flight
    }
}

macro_rules! header_impl {
    () => {
        fn header(&self) -> &FrameHeader {
            &self.header
        }
        fn header_mut(&mut self) -> &mut FrameHeader {
            &mut self.header
        }
    };
}

/// The pack's submission, in seq order (the sequencer's release), behind the compute's marker. The
/// send needs the result on the host, so it follows the pack's completion.
struct Pack {
    header: FrameHeader,
    ctx: Arc<Ctx>,
    ready: Option<Staged>,
}

impl Frame for Pack {
    header_impl!();

    fn do_finally(mut self: Box<Self>) {
        let ready = self.ready.take().expect("once");
        if !self.header.is_alive() {
            return ready.lease.abandon(1);
        }
        hop(3, ready.generation, ready.seq);
        let marker = self
            .ctx
            .device
            .submit(PACK_Q, self.ctx.pack_cost, &[ready.marker]);
        let next = Dispatch {
            header: FrameHeader::with_kill_switch(SOCKET, Some(ready.kill.clone())),
            ctx: self.ctx.clone(),
            ready: Some(ready),
        };
        self.ctx.completions.after_marker(marker, Box::new(next));
    }
}

/// The send: an ordered submission to the NIC; its completion puts the access unit on the wire.
struct Dispatch {
    header: FrameHeader,
    ctx: Arc<Ctx>,
    ready: Option<Staged>,
}

impl Frame for Dispatch {
    header_impl!();

    fn do_finally(mut self: Box<Self>) {
        let ready = self.ready.take().expect("once");
        if !self.header.is_alive() {
            return ready.lease.abandon(1);
        }
        hop(4, ready.generation, ready.seq);
        let marker = self.ctx.device.submit(NIC, SEND, &[]);
        let next = Sent {
            header: FrameHeader::new(0x5e47),
            ctx: self.ctx.clone(),
            ready: Some(ready),
        };
        self.ctx.completions.after_marker(marker, Box::new(next));
    }
}

struct Sent {
    header: FrameHeader,
    ctx: Arc<Ctx>,
    ready: Option<Staged>,
}

impl Frame for Sent {
    header_impl!();

    fn do_finally(mut self: Box<Self>) {
        let ready = self.ready.take().expect("once");
        hop(5, ready.generation, ready.seq);
        lock(&self.ctx.wire).push_back(Msg::Packet {
            generation: ready.generation,
            seq: ready.seq,
        });
        ready.lease.arrive(1, ());
    }
}

enum Control {
    Credit {
        generation: u32,
        horizon: u64,
    },
    Seek {
        generation: u32,
        to: u64,
        horizon: u64,
    },
}

struct ControlFrame {
    header: FrameHeader,
    ctx: Arc<Ctx>,
    msg: Option<Control>,
}

impl Frame for ControlFrame {
    header_impl!();

    fn execute(&mut self) -> Result<(), FrameError> {
        match self.msg.take() {
            Some(Control::Credit {
                generation,
                horizon,
            }) => self.ctx.on_credit(generation, horizon),
            Some(Control::Seek {
                generation,
                to,
                horizon,
            }) => self.ctx.seek(generation, to, horizon),
            None => {}
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------------------------
// The fake consumer (the harness).

#[derive(Default, Debug)]
struct ConsumerReport {
    shown: u64,
    late: u64,
    out_of_order: u64,
    stale_dropped: u64,
    over_credit: u64,
    start_latency: Option<Duration>,
    seek_latency: Option<Duration>,
    timed_out: bool,
}

fn consumer(ctx: &Arc<Ctx>) -> ConsumerReport {
    let mut report = ConsumerReport::default();
    let send = |msg: Control| {
        let frame = ControlFrame {
            header: FrameHeader::new(CONTROL),
            ctx: ctx.clone(),
            msg: Some(msg),
        };
        assert!(
            ctx.control.offer(Box::new(frame)).is_ok(),
            "control sink closed"
        );
    };
    let begun = Instant::now();
    let mut generation = 1u32;
    let mut buffer: BTreeMap<u64, ()> = BTreeMap::new();
    let mut last_rx: Option<u64> = None;
    let mut playhead = 0u64;
    let mut playing_since: Option<(Instant, u64)> = None;
    let mut asked_at = Instant::now();
    let mut seeked = false;
    let mut paused_until: Option<Instant> = None;
    let mut paused_done = false;
    let mut credited = AHEAD;
    let mut eos = false;
    send(Control::Seek {
        generation,
        to: 0,
        horizon: AHEAD,
    });
    loop {
        let msgs: Vec<Msg> = lock(&ctx.wire).drain(..).collect();
        for msg in msgs {
            match msg {
                Msg::Packet { generation: e, seq } if e == generation => {
                    if seq >= credited {
                        report.over_credit += 1;
                    }
                    if last_rx.is_some_and(|l| seq != l + 1) {
                        report.out_of_order += 1;
                    }
                    if last_rx.is_none() {
                        let latency = asked_at.elapsed();
                        if seeked {
                            report.seek_latency = Some(latency);
                        } else {
                            report.start_latency = Some(latency);
                        }
                    }
                    last_rx = Some(seq);
                    buffer.insert(seq, ());
                }
                Msg::Packet { .. } => report.stale_dropped += 1,
                Msg::End { generation: e } if e == generation => eos = true,
                Msg::End { .. } => {}
            }
        }
        let now = Instant::now();
        if !seeked && now.duration_since(begun) >= SEEK_AT {
            seeked = true;
            generation += 1;
            buffer.clear();
            last_rx = None;
            playing_since = None;
            playhead = SEEK_TO;
            asked_at = now;
            credited = SEEK_TO + AHEAD;
            send(Control::Seek {
                generation,
                to: SEEK_TO,
                horizon: credited,
            });
        }
        if let Some(until) = paused_until {
            if now >= until {
                paused_until = None;
                paused_done = true;
                playing_since = Some((now, playhead));
            }
        } else if !paused_done && seeked && playhead >= PAUSE_AT_PTS {
            paused_until = Some(now + PAUSE_FOR);
            playing_since = None;
        }
        if playing_since.is_none() && paused_until.is_none() {
            let have = buffer.range(playhead..).count() as u64;
            if have > 0 && have >= START_BUFFER.min(END - playhead) {
                playing_since = Some((now, playhead));
            }
        }
        if let Some((since, from)) = playing_since {
            let due = from + (now.duration_since(since).as_nanos() / FRAME.as_nanos()) as u64;
            while playhead < due.min(END) {
                if buffer.remove(&playhead).is_some() {
                    report.shown += 1;
                } else {
                    report.late += 1;
                }
                playhead += 1;
            }
        }
        let horizon = playhead + AHEAD;
        if horizon > credited {
            credited = horizon;
            send(Control::Credit {
                generation,
                horizon,
            });
        }
        if eos && playhead >= END {
            return report;
        }
        if begun.elapsed() > Duration::from_secs(30) {
            report.timed_out = true;
            return report;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
}

// ---------------------------------------------------------------------------------------------

fn cpus() -> Vec<usize> {
    std::env::var("EUHEDRAL_TEST_CPUS")
        .ok()
        .map(|v| {
            v.split(',')
                .filter_map(|c| c.trim().parse().ok())
                .collect::<Vec<usize>>()
        })
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| (0..4).collect())
}

fn lattice(cpus: &[usize]) -> Arc<ControlPlaneLattice> {
    let park = Duration::from_micros(100);
    let idle = IdlePolicy::fixed(park, IdlePolicy::DEFAULT_CONTENTION_HALF_LIFE)
        .unwrap()
        .with_unproductive_park(park);
    let fragment = FragmentConfig::default().with_idle_policy(idle);
    let mut config = LatticeConfig::new(Arc::new(BaseCloneableObject::new(
        fragment,
        TimingExecutor::default(),
    )));
    config.name = "credit-graph-test".into();
    config.shard_name = "credit-graph-test".into();
    config.allowed_cpus = cpus.iter().copied().collect();
    config.shutdown_timeout = Duration::from_secs(5);
    config.startup_timeout = Duration::from_secs(10);
    let lattice = ControlPlaneLattice::new(config).unwrap();
    lattice.start().unwrap();
    lattice
}

/// One scenario on its own lattice, asserting the pass criteria (the report is printed).
fn scenario(pack_cost: Duration) {
    let cpus = cpus();
    let allowed = affinity::base_mask();
    if !cpus.iter().all(|&cpu| allowed.contains(cpu)) {
        // This process may not run on every requested CPU: pass without checking, as
        // euhedral-core's tests do.
        return;
    }
    let lattice = lattice(&cpus);
    let (lanes, devices) =
        FakeDevice::pool(&["disk", "parse", "compute-a", "compute-b", "pack", "nic"]);
    let ctx = Ctx::new(lattice.clone(), pack_cost, lanes, devices);
    lattice.add_upstream(ctx.completions.clone()).unwrap();
    let unit = ctx.stages.admit().unwrap(); // attaches the lake's sinks
    lattice.add_upstream(ctx.control.delegate()).unwrap();
    let s = stats();
    for counter in [&s.count, &s.max_ns, &s.slow, &s.preempted, &s.voluntary] {
        counter.store(0, Ordering::Relaxed);
    }

    let report = consumer(&ctx);

    let closing = Instant::now();
    lock(&ctx.link).closed = true;
    let deadline = Instant::now() + Duration::from_secs(5);
    let graph = ctx.graph.get().unwrap().clone();
    while (ctx.in_flight() > 0 || graph.live() > 0) && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(1));
    }
    let left = ctx.in_flight();
    let quanta_left = graph.live();
    let slots_left = ctx.sequencer.get().unwrap().len();
    let pending_left = ctx.completions.pending();
    let drained_in = closing.elapsed();
    let closing = Instant::now();
    ctx.control.complete_gracefully();
    drop(unit);
    ctx.stages.complete_gracefully();
    LatticeSource::complete(&*ctx.completions);
    lattice.close();
    let closed_in = closing.elapsed();

    let c = &ctx.counts;
    let get = |a: &AtomicU64| a.load(Ordering::Relaxed);
    let (callbacks, slow) = (get(&s.count), get(&s.slow));
    let (p50, p99, max) = ctx.completions.lateness_us();
    println!("pack {pack_cost:?}, CPUs {cpus:?}: {report:?}");
    println!(
        "made {} sent {} dropped {} retired {} (killed {}, failed {}); max in flight {} (pool {POOL}); left: {left} \
         items, {quanta_left} quanta, {slots_left} slots, {pending_left} completions; drained in {drained_in:?}, \
         closed in {closed_in:?}",
        get(&c.made),
        get(&c.sent),
        get(&c.dropped),
        get(&c.retired),
        get(&c.killed_outcomes),
        get(&c.failed_outcomes),
        get(&c.max_in_flight),
    );
    println!(
        "callbacks: {callbacks}, max {:?}, over 200 µs: {slow} (+{} preempted by the OS), blocking at all: {}; \
         completion found after done (upper bound) p50 {p50} µs, p99 {p99} µs, max {max} µs",
        Duration::from_nanos(get(&s.max_ns)),
        get(&s.preempted),
        get(&s.voluntary),
    );
    let inversions: Vec<(&str, u64)> = HOP_NAMES
        .iter()
        .zip(&hops().inversions)
        .map(|(name, count)| (*name, count.swap(0, Ordering::Relaxed)))
        .collect();
    *lock(&hops().last) = [(0, 0); 6];
    println!("order breaks per hop, within a generation: {inversions:?}");
    assert!(!report.timed_out, "the consumer timed out");
    assert_eq!(report.late, 0, "late");
    assert_eq!(report.out_of_order, 0, "out of order");
    assert!(
        inversions.iter().all(|(_, n)| *n == 0),
        "order broke at a hop: {inversions:?}"
    );
    assert_eq!(report.over_credit, 0, "beyond credit");
    assert_eq!(
        (left, quanta_left, slots_left, pending_left),
        (0, 0, 0, 0),
        "left over"
    );
    assert_eq!(
        get(&c.made),
        get(&c.sent) + get(&c.dropped),
        "made = sent + dropped"
    );
    assert_eq!(get(&c.retired), get(&c.made), "every quantum retired once");
    assert_eq!(get(&c.failed_outcomes), 0);
    assert!(get(&c.max_in_flight) <= POOL);
    assert_eq!(slow, 0, "frame callbacks over 200 µs (not preempted)");
    assert!(
        closed_in < Duration::from_millis(100),
        "shutdown took {closed_in:?}"
    );
}

/// The scenarios share the callback statistics and the CPUs: one at a time.
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

#[test]
fn the_credit_graph_with_a_fast_pack() {
    let _one = lock(&ONE_AT_A_TIME);
    scenario(Duration::from_micros(2500));
}

/// A 7 ms pack: the pack is the bottleneck, the pool fills, and the seek kills a full pool.
#[test]
fn the_credit_graph_with_a_slow_pack() {
    let _one = lock(&ONE_AT_A_TIME);
    scenario(Duration::from_micros(7000));
}
