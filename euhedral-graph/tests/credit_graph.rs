//! A credit-graph on a real lattice with [`FakeDevice`] lanes: the proof that the crate's pieces
//! compose without anything waiting.
//!
//! - **One lattice in greedy mode** (4 workers, park 100 µs). Nothing in a frame waits: every stage
//!   submits to a fake device lane and returns.
//! - **Per item, a `StageGraph`:** fetch (ordered, disk) -completion-> parse (ordered, parse queue)
//!   -submission-> compute (unordered, either compute queue). The compute's submission completes
//!   the item's `Sequencer` slot, whose in-order release offers the pack (ordered), whose
//!   completion sends (ordered), whose completion puts the packet on the wire.
//! - **The pool lease** of an item is a count of two ends shared by the item and what it hands on:
//!   its retirement (boundary reached) and its sent-or-dropped end. The end that brings the count to
//!   zero frees the pool slot, which may admit the next item.
//! - **Admission** = credit horizon and pool (16), from the control lane (credits, jumps to a new generation) and from
//!   lease releases.
//! - **One stepped clock.** The fake consumer (the harness, on the test thread) and the `FakeDevice`s read one
//!   `StepClock` the harness owns. It advances to the next event (a timer of the ledger below, or the
//!   consumer's next frame, seek or pause end) only when nothing is owed, so "late" counts what the graph did,
//!   not how fast the machine ran.
//! - **An exact ledger** (`Ledger`): the test counts every callback of its own code the graph still owes (a stage
//!   run, a pack, a dispatch, a send, a retirement, a control frame) and every timer (a marker's completion that
//!   releases one). The harness waits for the count to reach zero, then moves what is due at the clock into the
//!   count, and only when neither holds does time move. Nothing is read from the lattice.
//! - **The fake consumer** takes 120 items a second, credits 1 s ahead, starts after 0.5 s, jumps at 1.5 s to
//!   seq 480 (a `Seek`: a new generation, with a new kill switch), pauses 0.4 s and takes items to the end of
//!   the 6 s stream, all in the scenario's time.
//! - **Order at every hop** (fetch, parse, sequencer release, pack, send, sent) is checked within
//!   each generation.
//! - **The no-wait check:** a timing executor measures every frame callback (execute + finalizer) on the
//!   wall clock and counts the ones over 200 µs that the OS did not preempt (no involuntary context switch
//!   during it).
//!
//! Pass: the numbers of each scenario exactly (`Pinned`: shown, late, made, sent, dropped, the start and seek
//! latencies in scenario time), 0 late (except the slow-pack scenario), 0 out of order, 0 beyond credit, 0 left (items, slots, completions, quanta), made
//! = sent + dropped, in flight <= pool, the stream finishing within its time. A pack slower than the consumer
//! must make items late. With `EUHEDRAL_TEST_TIMING` set (a quiet machine), also no callback over 200 µs and
//! shutdown under 100 ms; otherwise those two are printed, not asserted (run with `-- --nocapture` to see the
//! line; libtest hides a passing test's output).
//!
//! Real time decides only how long the harness waits for the count to reach zero, and a guard (60 s) against a
//! hang: a ledger that disagrees with the graph (a retirement expected early, a callback never made) hangs into
//! it and fails. No assertion is on a real-time bound.
//!
//! `EUHEDRAL_TEST_CPUS` (default 0,1,2,3) picks the lattice's CPUs. As in `euhedral-core`'s tests
//! that need particular hardware, the test passes without checking anything when this process may
//! not run on all of them. It needs Linux (per-thread context-switch counts).
#![allow(
    clippy::disallowed_types,
    reason = "test fixtures: the test thread locks its records"
)]
#![cfg(target_os = "linux")]
// The fake consumer is test harness, not a frame: it waits in real time for the ledger to reach zero.
#![allow(clippy::disallowed_methods)]

use std::collections::{BTreeMap, VecDeque};
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, PoisonError, Weak};
use std::time::{Duration, Instant};

use euhedral_core::config::{FragmentConfig, IdlePolicy, LatticeConfig};
use euhedral_core::control_plane::ControlPlaneLattice;
use euhedral_core::frames::{Frame, FrameError, FrameHeader};
use euhedral_core::generics::AbstractExecutor;
use euhedral_core::generics::LatticeSource;
use euhedral_core::impls::BaseCloneableObject;
use euhedral_core::ingest::{IngestSink, QueueIngestSink};
use euhedral_graph::testing::{FakeDevice, StepClock};
use euhedral_graph::{
    Completions, DEFAULT_RETRY_BACKSTOP, DeviceQueue, EdgeKind, GraphRuntime, Lake, LaneId,
    LanePool, Marker, QuantumOutcome, Sequencer, Slot, Stage, StageCx, StageError, StageGraph,
    StageGraphSpec, StageSpec, Submitted, Ticket,
};
use euhedral_hardware_utils::affinity;

const FPS: u64 = 120;
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
        s.count.fetch_add(1, Ordering::AcqRel);
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
    /// The item's lease. Dropping the staged item is its sent-or-dropped end.
    lease: Arc<Lease>,
}

impl Drop for Staged {
    fn drop(&mut self) {
        self.lease.end();
    }
}

/// The pool lease of an item: two ends, its retirement and its sent-or-dropped end. Whichever ends
/// last frees the pool slot.
struct Lease {
    pending: AtomicU32,
    /// Stored by the send before its end; read by the last end after its `fetch_sub`.
    sent: AtomicBool,
    ctx: Weak<Ctx>,
}

impl Lease {
    fn new(ctx: Weak<Ctx>) -> Arc<Self> {
        Arc::new(Self {
            pending: AtomicU32::new(2),
            sent: AtomicBool::new(false),
            ctx,
        })
    }

    fn end(&self) {
        if self.pending.fetch_sub(1, Ordering::AcqRel) == 1
            && let Some(ctx) = self.ctx.upgrade()
        {
            ctx.release(self.sent.load(Ordering::Relaxed));
        }
    }
}

/// The quantum: one item of one generation.
struct Item {
    generation: u32,
    seq: u64,
    kill: Arc<AtomicBool>,
    slot: Mutex<Option<Slot<Staged>>>,
    /// The lease. Dropping the item is its retirement end; while the item holds its slot, the
    /// sent-or-dropped end is the item's too.
    lease: Arc<Lease>,
    /// Per lane the item has submitted on, when its latest marker completes (the stage graph's boundary: see
    /// `Ledger`).
    lanes: Mutex<BTreeMap<u16, u64>>,
}

impl Item {
    /// Notes a submission on `lane` that completes at `done` and returns when the item's quantum retires if
    /// nothing else is submitted: the latest marker of each lane it used, the latest of those.
    fn submitted(&self, lane: u16, done: u64) -> u64 {
        let mut lanes = lock(&self.lanes);
        lanes.insert(lane, done);
        lanes.values().copied().max().unwrap_or(done)
    }

    /// The item will never reach the sequencer: its slot is abandoned and its sent-or-dropped end
    /// ends unsent.
    fn drop_slot(&self) {
        if let Some(slot) = lock(&self.slot).take() {
            slot.abandon();
            self.lease.end();
        }
    }
}

impl Drop for Item {
    fn drop(&mut self) {
        self.drop_slot();
        self.lease.end();
    }
}

struct Device {
    lanes: Vec<Arc<FakeDevice>>,
    ledger: Ledger,
}

impl Device {
    /// Submits on `lane`: the marker and when it completes, in the scenario's time.
    fn submit(&self, lane: u16, cost: Duration, waits: &[Marker]) -> (Marker, u64) {
        let device = &self.lanes[usize::from(lane)];
        let marker = device.submit(cost, waits);
        let done = device.done_at_ns(marker).expect("a submission has a time");
        (marker, done)
    }
}

struct Fetch(Arc<Device>);

impl Stage<Item> for Fetch {
    fn run(&self, item: &Item, cx: &StageCx<'_>) -> Result<Submitted, StageError> {
        hop(0, item.generation, item.seq);
        let (marker, done) = self.0.submit(DISK, FETCH, cx.waits);
        // The completion edge: the parse is released when this marker completes.
        let at = item.submitted(DISK, done);
        self.0.ledger.at(at, None);
        self.0.ledger.pay();
        Ok(Submitted::marker(marker))
    }

    fn abandoned(&self, item: &Item, _home: &dyn DeviceQueue) {
        item.drop_slot();
    }
}

struct Parse(Arc<Device>);

impl Stage<Item> for Parse {
    fn run(&self, item: &Item, cx: &StageCx<'_>) -> Result<Submitted, StageError> {
        hop(1, item.generation, item.seq);
        let (marker, done) = self.0.submit(PARSE_Q, PARSE, cx.waits);
        item.submitted(PARSE_Q, done);
        // The submission edge: the compute is published at once.
        self.0.ledger.owe(1);
        self.0.ledger.pay();
        Ok(Submitted::marker(marker))
    }

    fn abandoned(&self, item: &Item, _home: &dyn DeviceQueue) {
        item.drop_slot();
    }
}

struct Compute(Arc<Device>);

impl Stage<Item> for Compute {
    fn run(&self, item: &Item, cx: &StageCx<'_>) -> Result<Submitted, StageError> {
        // The lane is the item's, not the pool's pick (which is random per thread): the same items go to the
        // same lanes on every run.
        let lane = COMPUTE_A + (item.seq % 2) as u16;
        let (marker, done) = self.0.submit(lane, COMPUTE, cx.waits);
        let retire_at = item.submitted(lane, done);
        if let Some(slot) = lock(&item.slot).take() {
            let Item {
                generation,
                seq,
                ref kill,
                ..
            } = *item;
            slot.complete(Staged {
                generation,
                seq,
                kill: kill.clone(),
                marker,
                lease: item.lease.clone(),
            });
        }
        // The quantum retires when the latest marker of each lane it used is reached, then its retirement runs.
        self.0.ledger.at(retire_at, None);
        self.0.ledger.pay();
        Ok(Submitted::marker(marker))
    }

    fn abandoned(&self, item: &Item, _home: &dyn DeviceQueue) {
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
    sequencer: OnceLock<Arc<Sequencer<Arc<Ticket<Staged>>>>>,
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
            completions: Completions::new(&lanes),
            device: Arc::new(Device {
                lanes: devices,
                ledger: Ledger::default(),
            }),
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
        let sequencer = Sequencer::ordered(move |ready: Staged| {
            let Some(ctx) = me.upgrade() else { return };
            hop(2, ready.generation, ready.seq);
            // The release offers one pack, and its end pays.
            ctx.device.ledger.owe(1);
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
                    QuantumOutcome::Failed { .. } | QuantumOutcome::DeviceFailed { .. } => {
                        _ = ctx.counts.failed_outcomes.fetch_add(1, Ordering::Relaxed)
                    }
                }
                drop(item);
                // The item's last obligation, paid after everything the retirement set going.
                ctx.device.ledger.pay();
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
                    // The first stage's run is owed from the start.
                    self.device.ledger.owe(1);
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
        let slot = self.sequencer.get().expect("the sequencer").register();
        let item = Item {
            generation: link.generation,
            seq,
            kill: link.kill.clone(),
            slot: Mutex::new(Some(slot)),
            lease: Lease::new(self.me.clone()),
            lanes: Mutex::new(BTreeMap::new()),
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
            // The seek runs alone (the harness sent it with nothing owed), so no frame is between its kill check
            // and its run: the kill takes hold of every sleeping obligation at once.
            self.device.ledger.kill(link.generation);
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
        let ledger = &self.ctx.device.ledger;
        if !self.header.is_alive() {
            // Dropping the item ends its lease, which may admit the next: before this frame's end is paid.
            drop(ready);
            ledger.pay();
            return;
        }
        hop(3, ready.generation, ready.seq);
        let (marker, done) = self
            .ctx
            .device
            .submit(PACK_Q, self.ctx.pack_cost, &[ready.marker]);
        // The dispatch follows the pack's completion; a seek ends that entry at once.
        ledger.at(done, Some(ready.generation));
        let next = Dispatch {
            header: FrameHeader::with_kill_switch(SOCKET, Some(ready.kill.clone())),
            ctx: self.ctx.clone(),
            ready: Some(ready),
        };
        self.ctx.completions.after_marker(marker, Box::new(next));
        ledger.pay();
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
        let ledger = &self.ctx.device.ledger;
        if !self.header.is_alive() {
            drop(ready);
            ledger.pay();
            return;
        }
        hop(4, ready.generation, ready.seq);
        let (marker, done) = self.ctx.device.submit(NIC, SEND, &[]);
        ledger.at(done, None);
        let next = Sent {
            header: FrameHeader::new(0x5e47),
            ctx: self.ctx.clone(),
            ready: Some(ready),
        };
        self.ctx.completions.after_marker(marker, Box::new(next));
        ledger.pay();
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
        ready.lease.sent.store(true, Ordering::Relaxed);
        lock(&self.ctx.wire).push_back(Msg::Packet {
            generation: ready.generation,
            seq: ready.seq,
        });
        // The lease's end may admit the next item: before this frame's end is paid.
        drop(ready);
        self.ctx.device.ledger.pay();
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
        self.ctx.device.ledger.pay();
        Ok(())
    }
}

// ---------------------------------------------------------------------------------------------
// The stepped clock and the fake consumer (the harness).

/// A real-time guard that only catches a hang: the longest the harness waits for the obligations owed to be
/// paid. No assertion depends on it.
const HANG_GUARD: Duration = Duration::from_secs(60);
/// The bound on the scenario's own time (the stream is under 7 s): beyond it nothing is making progress.
const STREAM_TIME_LIMIT_NS: u64 = 30_000_000_000;

/// What the test knows the graph still has to do, counted by the test itself; the lattice's own queues and counters
/// are not read. An obligation is one callback of the test's code that will happen, with no further time needed (a
/// frame queued or running), or one thing waiting for a time (a timer). The harness moves scenario time only when
/// nothing is owed, and moves a timer that is due into what is owed, so the graph has done everything it can at
/// one time before the next time begins. Real time decides only how long the harness waits for a count to reach
/// zero.
///
/// The model, per item, in the order a handler works: register what follows (a timer, or a new obligation), then pay
/// its own. So the count never reaches zero while a handler's follow-up is unaccounted.
/// - Starting an item owes its fetch's run. The fetch registers a timer at its marker's completion, which releases
///   the parse (a completion edge) and pays. The parse owes the compute (a submission edge: published at once)
///   and pays; the compute registers the item's retirement timer, its slot's in-order release owes a pack, and
///   it pays.
/// - The retirement rule lives in `stage_graph.rs` (the module docs, `arm_boundary`, `hold_killed_work`): a
///   quantum retires on the latest marker of each lane it used, once its frames have ended, and a kill ends the
///   completion entries at their next visit and the quantum then waits for the same markers. The timer is
///   therefore the latest of the item's markers, one per lane (`Item::submitted`), whether or not the item
///   was killed on the way. When it is due the retirement owes one callback: the retirement closure, which
///   pays last, after the item's drop has ended its lease.
/// - The pack owes a dispatch timer at its marker (or, killed, only ends); the dispatch a send timer at the
///   NIC's marker; the send ends the item's lease. A dispatch whose frame carries a kill switch is an entry
///   the lattice ends at its next visit after the kill, so a seek makes those owed at once.
/// - A control message owes its frame's run.
#[derive(Default)]
struct Ledger {
    owed: AtomicU64,
    timers: Mutex<Vec<Timer>>,
}

struct Timer {
    /// Scenario time, ns.
    at: u64,
    /// A dispatch entry of this generation (ended at once by a seek of it).
    dispatch_of: Option<u32>,
}

impl Ledger {
    fn owe(&self, n: u64) {
        self.owed.fetch_add(n, Ordering::SeqCst);
    }

    fn pay(&self) {
        let was = self.owed.fetch_sub(1, Ordering::SeqCst);
        assert!(was > 0, "paid what was not owed");
    }

    fn owed(&self) -> u64 {
        self.owed.load(Ordering::SeqCst)
    }

    /// One thing is owed once scenario time reaches `at`.
    fn at(&self, at: u64, dispatch_of: Option<u32>) {
        lock(&self.timers).push(Timer { at, dispatch_of });
    }

    /// A seek of `generation`: its dispatch entries end at the lattice's next visit.
    fn kill(&self, generation: u32) {
        let mut timers = lock(&self.timers);
        let before = timers.len();
        timers.retain(|t| t.dispatch_of != Some(generation));
        self.owe((before - timers.len()) as u64);
    }

    /// Moves what is due at `now` into what is owed; how many.
    fn sweep(&self, now: u64) -> usize {
        let mut timers = lock(&self.timers);
        let before = timers.len();
        timers.retain(|t| t.at > now);
        let due = before - timers.len();
        self.owe(due as u64);
        due
    }

    fn next(&self) -> Option<u64> {
        lock(&self.timers).iter().map(|t| t.at).min()
    }

    fn waiting(&self) -> usize {
        lock(&self.timers).len()
    }
}

/// The test-owned time, and what advances it.
struct Stepper {
    clock: Arc<StepClock>,
    ctx: Arc<Ctx>,
    started: Instant,
}

impl Stepper {
    fn now_ns(&self) -> u64 {
        self.clock.now_ns()
    }

    fn ledger(&self) -> &Ledger {
        &self.ctx.device.ledger
    }

    /// Waits (in real time) until nothing is owed, then moves what is due into what is owed, until nothing is
    /// owed and nothing is due: the graph has done everything it can at this scenario time.
    fn settle(&self) {
        let ledger = self.ledger();
        loop {
            let waiting = Instant::now();
            while ledger.owed() > 0 {
                assert!(
                    waiting.elapsed() < HANG_GUARD && self.started.elapsed() < 20 * HANG_GUARD,
                    "the graph never paid {} obligations at {} ns ({} timers waiting)",
                    ledger.owed(),
                    self.now_ns(),
                    ledger.waiting()
                );
                std::thread::sleep(Duration::from_micros(20));
            }
            if ledger.sweep(self.now_ns()) == 0 {
                return;
            }
        }
    }

    /// The next time something is waiting for.
    fn next_timer(&self) -> Option<u64> {
        self.ledger().next()
    }

    /// Sets the clock to `next` and lets the graph do all it can at the new time.
    fn step_to(&self, next: u64) {
        self.clock.advance_to(next);
        self.settle();
    }
}

#[derive(Default, Debug)]
struct ConsumerReport {
    shown: u64,
    late: u64,
    out_of_order: u64,
    stale_dropped: u64,
    over_credit: u64,
    /// In the scenario's time, not the wall clock's.
    start_latency: Option<Duration>,
    seek_latency: Option<Duration>,
    timed_out: bool,
    steps: u64,
}

const FRAME_NS: u64 = 1_000_000_000 / FPS;

fn consumer(ctx: &Arc<Ctx>, stepper: &Stepper) -> ConsumerReport {
    let mut report = ConsumerReport::default();
    let send = |msg: Control| {
        // Owed before it is offered: the frame's end pays it.
        stepper.ledger().owe(1);
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
    let mut generation = 1u32;
    let mut buffer: BTreeMap<u64, ()> = BTreeMap::new();
    let mut last_rx: Option<u64> = None;
    let mut playhead = 0u64;
    // (when it began, the pts it began at), in ns of the scenario's time.
    let mut playing_since: Option<(u64, u64)> = None;
    let mut asked_at = 0u64;
    let mut seeked = false;
    let mut paused_until: Option<u64> = None;
    let mut paused_done = false;
    let mut credited = AHEAD;
    let mut eos = false;
    send(Control::Seek {
        generation,
        to: 0,
        horizon: AHEAD,
    });
    stepper.settle();
    loop {
        let now = stepper.now_ns();
        // What this pass did: if it did anything the lattice must go quiet at this time before the next pass.
        let mut acted = false;
        let msgs: Vec<Msg> = lock(&ctx.wire).drain(..).collect();
        acted |= !msgs.is_empty();
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
                        let latency = Duration::from_nanos(now - asked_at);
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
        if !seeked && now >= SEEK_AT.as_nanos() as u64 {
            seeked = true;
            acted = true;
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
                acted = true;
            }
        } else if !paused_done && seeked && playhead >= PAUSE_AT_PTS {
            paused_until = Some(now + PAUSE_FOR.as_nanos() as u64);
            playing_since = None;
            acted = true;
        }
        if playing_since.is_none() && paused_until.is_none() {
            let have = buffer.range(playhead..).count() as u64;
            if have > 0 && have >= START_BUFFER.min(END - playhead) {
                playing_since = Some((now, playhead));
                acted = true;
            }
        }
        if let Some((since, from)) = playing_since {
            let due = from + (now - since) / FRAME_NS;
            while playhead < due.min(END) {
                if buffer.remove(&playhead).is_some() {
                    report.shown += 1;
                } else {
                    report.late += 1;
                }
                playhead += 1;
                acted = true;
            }
        }
        let horizon = playhead + AHEAD;
        if horizon > credited {
            credited = horizon;
            acted = true;
            send(Control::Credit {
                generation,
                horizon,
            });
        }
        if eos && playhead >= END {
            return report;
        }
        if acted {
            stepper.settle();
            continue;
        }
        // Nothing more to do at this time: the next thing that can happen is a device submission
        // completing or one of the consumer's own times.
        let mut next = stepper.next_timer();
        let mut consider = |at: u64| {
            if at > now {
                next = Some(next.map_or(at, |n| n.min(at)));
            }
        };
        if !seeked {
            consider(SEEK_AT.as_nanos() as u64);
        }
        if let Some(until) = paused_until {
            consider(until);
        }
        if let Some((since, from)) = playing_since
            && playhead < END
        {
            consider(since + (playhead - from + 1) * FRAME_NS);
        }
        report.steps += 1;
        match next {
            Some(next) if next <= STREAM_TIME_LIMIT_NS && report.steps < 1_000_000 => {
                stepper.step_to(next);
            }
            // Nothing can happen any more, or the stream has outlived its time: it will not finish.
            _ => {
                report.timed_out = true;
                return report;
            }
        }
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

/// What one scenario did.
struct Outcome {
    report: ConsumerReport,
    inversions: Vec<(&'static str, u64)>,
    left: u64,
    quanta_left: usize,
    slots_left: usize,
    pending_left: usize,
    made: u64,
    sent: u64,
    dropped: u64,
    retired: u64,
    failed: u64,
    max_in_flight: u64,
    /// Callbacks over 200 µs of the wall clock, and the wall-clock time of the lattice's shutdown.
    slow: u64,
    closed_in: Duration,
}

/// One scenario on its own lattice, on a clock the harness steps. `None` when this process may not run on the
/// CPUs asked for.
fn run(pack_cost: Duration) -> Option<Outcome> {
    let cpus = cpus();
    let allowed = affinity::base_mask();
    if !cpus.iter().all(|&cpu| allowed.contains(cpu)) {
        // This process may not run on every requested CPU: pass without checking, as
        // euhedral-core's tests do.
        return None;
    }
    let lattice = lattice(&cpus);
    let clock = StepClock::new();
    let (lanes, devices) = FakeDevice::pool_stepped(
        &["disk", "parse", "compute-a", "compute-b", "pack", "nic"],
        &clock,
    );
    let ctx = Ctx::new(lattice.clone(), pack_cost, lanes, devices);
    lattice.add_upstream(ctx.completions.clone()).unwrap();
    let unit = ctx.stages.admit().unwrap(); // attaches the lake's sinks
    lattice.add_upstream(ctx.control.delegate()).unwrap();
    let s = stats();
    for counter in [&s.count, &s.max_ns, &s.slow, &s.preempted, &s.voluntary] {
        counter.store(0, Ordering::Relaxed);
    }
    let stepper = Stepper {
        clock,
        ctx: ctx.clone(),
        started: Instant::now(),
    };

    let report = consumer(&ctx, &stepper);

    // The drain: nothing more is admitted; time goes on until what is in flight has finished or no
    // submission is left to complete.
    lock(&ctx.link).closed = true;
    let graph = ctx.graph.get().unwrap().clone();
    let mut drain_steps = 0u64;
    while (ctx.in_flight() > 0 || graph.live() > 0) && drain_steps < 1_000_000 {
        let Some(next) = stepper.next_timer() else {
            break;
        };
        stepper.step_to(next);
        drain_steps += 1;
    }
    let left = ctx.in_flight();
    let quanta_left = graph.live();
    let slots_left = ctx.sequencer.get().unwrap().len();
    let pending_left = ctx.completions.pending();
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
    println!("pack {pack_cost:?}, CPUs {cpus:?}: {report:?}");
    println!(
        "made {} sent {} dropped {} retired {} (killed {}, failed {}); max in flight {} (pool {POOL}); left: {left} \
         items, {quanta_left} quanta, {slots_left} slots, {pending_left} completions; {drain_steps} drain steps",
        get(&c.made),
        get(&c.sent),
        get(&c.dropped),
        get(&c.retired),
        get(&c.killed_outcomes),
        get(&c.failed_outcomes),
        get(&c.max_in_flight),
    );
    println!(
        "wall-clock callbacks: {callbacks}, max {:?}, over 200 µs: {slow} (+{} preempted by the OS), blocking at all: {}; \
         shutdown {closed_in:?}",
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
    Some(Outcome {
        report,
        inversions,
        left,
        quanta_left,
        slots_left,
        pending_left,
        made: get(&c.made),
        sent: get(&c.sent),
        dropped: get(&c.dropped),
        retired: get(&c.retired),
        failed: get(&c.failed_outcomes),
        max_in_flight: get(&c.max_in_flight),
        slow,
        closed_in,
    })
}

/// What holds whatever the pack costs: order, credit, accounting, drain.
fn assert_graph_correct(o: &Outcome) {
    assert!(!o.report.timed_out, "the stream did not finish");
    assert!(
        o.report.start_latency.is_some(),
        "the first stream delivered nothing before the seek"
    );
    assert_eq!(o.report.out_of_order, 0, "out of order");
    assert!(
        o.inversions.iter().all(|(_, n)| *n == 0),
        "order broke at a hop: {:?}",
        o.inversions
    );
    assert_eq!(o.report.over_credit, 0, "beyond credit");
    assert_eq!(
        (o.left, o.quanta_left, o.slots_left, o.pending_left),
        (0, 0, 0, 0),
        "left over"
    );
    assert_eq!(o.made, o.sent + o.dropped, "made = sent + dropped");
    assert_eq!(o.retired, o.made, "every quantum retired once");
    assert_eq!(o.failed, 0);
    assert!(o.max_in_flight <= POOL);
}

/// What a scenario does, exactly: on the stepped clock with the ledger the same on every run and every machine.
struct Pinned {
    shown: u64,
    late: u64,
    made: u64,
    sent: u64,
    dropped: u64,
    /// Scenario time, ns.
    start_latency: u64,
    seek_latency: u64,
}

fn assert_pinned(o: &Outcome, pinned: &Pinned) {
    let got = Pinned {
        shown: o.report.shown,
        late: o.report.late,
        made: o.made,
        sent: o.sent,
        dropped: o.dropped,
        start_latency: o.report.start_latency.map_or(0, |d| d.as_nanos() as u64),
        seek_latency: o.report.seek_latency.map_or(0, |d| d.as_nanos() as u64),
    };
    assert_eq!(
        (
            got.shown,
            got.late,
            got.made,
            got.sent,
            got.dropped,
            got.start_latency,
            got.seek_latency
        ),
        (
            pinned.shown,
            pinned.late,
            pinned.made,
            pinned.sent,
            pinned.dropped,
            pinned.start_latency,
            pinned.seek_latency
        ),
        "(shown, late, made, sent, dropped, start latency, seek latency) of the scenario"
    );
}

/// A pack the consumer's rate can be met with: the graph is correct and no item is late.
fn scenario(pack_cost: Duration, pinned: &Pinned) {
    let Some(o) = run(pack_cost) else { return };
    assert_graph_correct(&o);
    assert_eq!(o.report.late, 0, "late");
    assert_pinned(&o, pinned);
    // Wall-clock timing is a report unless EUHEDRAL_TEST_TIMING is set: a loaded machine delays a callback
    // without an involuntary context switch (CPU quota, hypervisor steal, a sibling thread), which
    // the preemption count cannot see. Set it on a quiet machine to assert.
    if std::env::var_os("EUHEDRAL_TEST_TIMING").is_none() {
        println!(
            "timing (reported, not asserted; set EUHEDRAL_TEST_TIMING=1 to assert): {} callbacks over 200 µs, shutdown {:?}",
            o.slow, o.closed_in
        );
    } else {
        assert_eq!(o.slow, 0, "frame callbacks over 200 µs (not preempted)");
        assert!(
            o.closed_in < Duration::from_millis(100),
            "shutdown took {:?}",
            o.closed_in
        );
    }
}

/// The scenarios share the callback statistics and the CPUs: one at a time.
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

#[test]
fn the_credit_graph_with_a_fast_pack() {
    let _one = lock(&ONE_AT_A_TIME);
    scenario(
        Duration::from_micros(2500),
        &Pinned {
            shown: 397,
            late: 0,
            made: 517,
            sent: 516,
            dropped: 1,
            start_latency: 10_500_000,
            seek_latency: 11_833_281,
        },
    );
}

/// A 7 ms pack: the pack is the bottleneck, the pool fills, and the seek kills a full pool.
#[test]
fn the_credit_graph_with_a_slow_pack() {
    let _one = lock(&ONE_AT_A_TIME);
    scenario(
        Duration::from_micros(7000),
        &Pinned {
            shown: 368,
            late: 0,
            made: 469,
            sent: 453,
            dropped: 16,
            start_latency: 15_000_000,
            seek_latency: 118_000_000,
        },
    );
}

/// The stepped scenario counts lateness: a 16 ms pack (62 items a second) is slower than the consumer's 120 items a second, so the
/// stream runs dry and items are late, while the graph stays correct.
#[test]
fn a_pack_slower_than_the_consumer_makes_items_late() {
    let _one = lock(&ONE_AT_A_TIME);
    let Some(o) = run(Duration::from_micros(16000)) else {
        return;
    };
    assert_graph_correct(&o);
    assert!(o.report.late > 0, "a pack slower than the consumer is late");
    assert_pinned(
        &o,
        &Pinned {
            shown: 240,
            late: 63,
            made: 349,
            sent: 333,
            dropped: 16,
            start_latency: 24_000_000,
            seek_latency: 268_000_000,
        },
    );
}
