//! Shared test fixtures: a scripted lane item, a lane state with a gate, a receiver that records
//! what it is pushed, and a rig of fake lanes with their sources.

#![allow(dead_code, reason = "each test binary uses part of the fixtures")]
#![allow(
    clippy::disallowed_methods,
    clippy::disallowed_types,
    reason = "test threads stand in for workers; a long item holds one on a barrier"
)]

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use euhedral_core::frames::{Failure, Frame, FrameError, FrameHeader};
use euhedral_core::generics::{LatticeReceiver, LatticeSource, LatticeTerminal};
use euhedral_graph::LaneSubmit as _;
use euhedral_graph::testing::{FakeLane, FakeLaneError, FakeLanes};
use euhedral_graph::{
    Deliver, LaneCx, LaneId, LaneIntakes, LanePool, LanePoolBuilder, LaneSource, LaneState,
    LaneWork, Marker, Outcome,
};

pub fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// A frame that carries a tag in its id hash and counts how it ended.
pub struct Tag {
    header: FrameHeader,
    errored: Arc<AtomicUsize>,
}

impl Tag {
    pub fn frame(tag: u64, errored: &Arc<AtomicUsize>) -> Box<dyn Frame> {
        Box::new(Self {
            header: FrameHeader::new(tag),
            errored: errored.clone(),
        })
    }
}

impl Frame for Tag {
    fn header(&self) -> &FrameHeader {
        &self.header
    }

    fn header_mut(&mut self) -> &mut FrameHeader {
        &mut self.header
    }

    fn do_finally_with_error(self: Box<Self>, _error: FrameError) {
        self.errored.fetch_add(1, Ordering::SeqCst);
    }
}

/// What the items of a test record.
#[derive(Default)]
pub struct Probe {
    pub ran: Mutex<Vec<u64>>,
    pub retired: Mutex<Vec<(u64, Option<FakeLaneError>)>>,
    pub failed: Mutex<Vec<u64>>,
    pub bridged: Mutex<Vec<(u64, Marker)>>,
    pub dropped: AtomicUsize,
    /// Frames finished with an error (the source dropped them unpushed).
    pub errored: Arc<AtomicUsize>,
    /// Per-tag counts of frames made, for tests that count.
    pub counts: Mutex<Vec<u64>>,
    /// The fake lanes whose lane 0 submitter the drop-order counts look at.
    pub fake: Mutex<Option<FakeLanes>>,
    /// Items dropped before / after lane 0's submitter was dropped.
    pub dropped_before_submitter: AtomicUsize,
    pub dropped_after_submitter: AtomicUsize,
}

impl Probe {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    pub fn ran(&self) -> Vec<u64> {
        lock(&self.ran).clone()
    }
}

/// What a lane state's `close` saw.
pub type Closings = Arc<std::sync::Mutex<Vec<(bool, Option<u64>)>>>;
/// Where `close` looks to see whether the submitter is gone.
pub type Watch = Arc<std::sync::Mutex<Option<(euhedral_graph::testing::FakeLanes, LaneId)>>>;

/// The lane state: a gate the test opens, and a count of `check` calls.
#[derive(Clone, Default)]
pub struct Gate {
    pub value: Arc<AtomicU64>,
    pub checks: Arc<AtomicUsize>,
    /// What `close` saw: whether the lane's submitter was already dropped, and the raw handle it was given.
    pub closed: Closings,
    /// Where `close` looks to see whether the submitter is gone.
    pub watch: Watch,
}

impl LaneState for Gate {
    type Wait = u64;

    fn check(&mut self, _now: std::time::Instant) {
        self.checks.fetch_add(1, Ordering::SeqCst);
    }

    fn met(&mut self, wait: &u64, _now: std::time::Instant) -> bool {
        *wait <= self.value.load(Ordering::SeqCst)
    }

    fn close(&mut self, raw: &dyn std::any::Any) {
        let gone = lock(&self.watch)
            .as_ref()
            .is_some_and(|(fake, lane)| fake.dropped(*lane));
        lock(&self.closed).push((gone, raw.downcast_ref::<u64>().copied()));
    }
}

/// A scripted item.
pub struct Job {
    pub tag: u64,
    pub probe: Arc<Probe>,
    pub waits: Vec<Marker>,
    pub key: Option<u64>,
    /// `run` answers `Wait(n)` until the gate reaches it.
    pub need: Option<u64>,
    /// `run` answers `Submitted` (with this retire flag), or `Finished` when `None`.
    pub submit: Option<bool>,
    /// `then` queues a frame tagged with the item's tag.
    pub frame: bool,
    /// `then` delivers this item.
    pub then_deliver: Option<(Box<Job>, Deliver)>,
    /// `retired` delivers this item.
    pub retired_deliver: Option<(Box<Job>, Deliver)>,
    /// `run` requests this source (a re-entry the debug check must refuse).
    pub reenter: Option<Arc<dyn LatticeSource>>,
    /// `run` makes every armed retirement arrive and polls one away (a stolen retirement).
    pub steal: bool,
    /// `run` arms a retirement of its own through the submitter (a stray call).
    pub stray_arm: bool,
    /// `run` answers `Failed` (a submission the device refused).
    pub fail: bool,
    /// `run` blocks on these barriers (a long item).
    pub hold: Option<(Arc<std::sync::Barrier>, Arc<std::sync::Barrier>)>,
}

impl Job {
    pub fn new(probe: &Arc<Probe>, tag: u64) -> Self {
        Self {
            tag,
            probe: probe.clone(),
            waits: Vec::new(),
            key: None,
            need: None,
            submit: None,
            frame: false,
            then_deliver: None,
            retired_deliver: None,
            reenter: None,
            stray_arm: false,
            steal: false,
            fail: false,
            hold: None,
        }
    }

    pub fn waits(mut self, waits: &[Marker]) -> Self {
        self.waits = waits.to_vec();
        self
    }

    pub fn key(mut self, key: u64) -> Self {
        self.key = Some(key);
        self
    }

    pub fn need(mut self, need: u64) -> Self {
        self.need = Some(need);
        self
    }

    pub fn submit(mut self, retire: bool) -> Self {
        self.submit = Some(retire);
        self
    }

    pub fn frame(mut self) -> Self {
        self.frame = true;
        self
    }

    pub fn then_deliver(mut self, job: Job, to: Deliver) -> Self {
        self.then_deliver = Some((Box::new(job), to));
        self
    }

    pub fn retired_deliver(mut self, job: Job, to: Deliver) -> Self {
        self.retired_deliver = Some((Box::new(job), to));
        self
    }

    pub fn steal(mut self) -> Self {
        self.steal = true;
        self
    }

    pub fn stray_arm(mut self) -> Self {
        self.stray_arm = true;
        self
    }

    pub fn fail(mut self) -> Self {
        self.fail = true;
        self
    }

    pub fn hold(mut self, a: &Arc<std::sync::Barrier>, b: &Arc<std::sync::Barrier>) -> Self {
        self.hold = Some((a.clone(), b.clone()));
        self
    }
}

impl Drop for Job {
    fn drop(&mut self) {
        self.probe.dropped.fetch_add(1, Ordering::SeqCst);
        if let Some(fake) = lock(&self.probe.fake).as_ref() {
            let counter = if fake.dropped(LaneId(0)) {
                &self.probe.dropped_after_submitter
            } else {
                &self.probe.dropped_before_submitter
            };
            counter.fetch_add(1, Ordering::SeqCst);
        }
    }
}

pub type Source = LaneSource<FakeLane, Gate, Job>;
pub type Cx<'a> = LaneCx<'a, FakeLane, Gate, Job>;

impl LaneWork<FakeLane, Gate> for Job {
    const KINDS: &'static [&'static str] = &["job"];

    fn kind(&self) -> usize {
        0
    }

    fn waits(&self) -> &[Marker] {
        &self.waits
    }

    fn order(&self) -> Option<u64> {
        self.key
    }

    fn run(&mut self, cx: &mut Cx<'_>) -> Outcome<u64, FakeLaneError> {
        if let Some(need) = self.need
            && cx.state().value.load(Ordering::SeqCst) < need
        {
            return Outcome::Wait(need);
        }
        for &marker in &self.waits {
            if cx.bridges(marker) {
                lock(&self.probe.bridged).push((self.tag, marker));
            }
        }
        if self.steal {
            lock(&self.probe.fake).as_ref().expect("a rig").retire_all();
            let _ = cx.submitter().poll_retired();
        }
        if self.stray_arm {
            let _ = cx.submitter().arm_retirement().unwrap();
        }
        if let Some(source) = &self.reenter {
            // The claim is taken: a pull hands over nothing, a request banks its demand.
            assert_eq!(source.pull(&mut |_| {}, &mut |_| false, 5), 0);
            source.request(1);
        }
        if let Some((a, b)) = &self.hold {
            a.wait();
            b.wait();
        }
        lock(&self.probe.ran).push(self.tag);
        if self.fail {
            return Outcome::Failed(FakeLaneError::Refused);
        }
        match self.submit {
            Some(retire) => Outcome::Submitted { retire },
            None => Outcome::Finished,
        }
    }

    fn then(&mut self, _marker: Option<Marker>, cx: &mut Cx<'_>) {
        if self.frame {
            cx.ready(Tag::frame(self.tag, &self.probe.errored));
        }
        if let Some((job, to)) = self.then_deliver.take() {
            cx.deliver(*job, to).unwrap();
        }
    }

    fn retired(mut self, _marker: Marker, failure: Option<FakeLaneError>, cx: &mut Cx<'_>) {
        lock(&self.probe.retired).push((self.tag, failure));
        if let Some((job, to)) = self.retired_deliver.take() {
            cx.deliver(*job, to).unwrap();
        }
    }

    fn failed(self, _error: FakeLaneError, _cx: &mut Cx<'_>) {
        lock(&self.probe.failed).push(self.tag);
    }
}

/// Records the tags of the frames it is pushed.
#[derive(Default)]
pub struct Collect {
    pub received: Mutex<Vec<u64>>,
    pub completes: AtomicUsize,
    pub complete: AtomicBool,
}

impl Collect {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    pub fn received(&self) -> Vec<u64> {
        lock(&self.received).clone()
    }
}

impl LatticeTerminal for Collect {
    fn add_upstream(&self, _upstream: Arc<dyn LatticeSource>) {}
}

impl LatticeReceiver for Collect {
    fn push(&self, frame: Box<dyn Frame>) {
        lock(&self.received).push(frame.header().id_hash());
    }

    fn on_complete(&self) {
        self.completes.fetch_add(1, Ordering::SeqCst);
        self.complete.store(true, Ordering::SeqCst);
    }

    fn on_error(&self, _error: Failure) {}
}

/// Fake lanes of one backend, their pool, intakes and sources.
pub struct Rig {
    pub fake: FakeLanes,
    pub pool: Arc<LanePool>,
    pub intakes: Arc<LaneIntakes<Job>>,
    pub sources: Vec<Arc<Source>>,
    pub receivers: Vec<Arc<Collect>>,
    pub gates: Vec<Gate>,
    pub probe: Arc<Probe>,
}

impl Rig {
    /// `count` lanes; `groups` list lane indexes. Every source has a receiver.
    pub fn new(count: usize, groups: &[&[u16]]) -> Self {
        let mut builder = LanePoolBuilder::new();
        let first = builder.reserve(count);
        let (fake, lanes, queues) = FakeLanes::open(first, count);
        for (i, queue) in queues.into_iter().enumerate() {
            builder.set(LaneId(first.0 + i as u16), queue);
        }
        let pool = builder.build();
        let groups: Vec<Vec<LaneId>> = groups
            .iter()
            .map(|g| g.iter().map(|&l| LaneId(l)).collect())
            .collect();
        let group_refs: Vec<&[LaneId]> = groups.iter().map(Vec::as_slice).collect();
        let (intakes, inputs) = LaneIntakes::new(pool.clone(), &group_refs);
        let mut sources = Vec::new();
        let mut receivers = Vec::new();
        let mut gates = Vec::new();
        for (lane, input) in lanes.into_iter().zip(inputs) {
            let gate = Gate::default();
            let source = LaneSource::new(lane, gate.clone(), input, intakes.clone());
            let receiver = Collect::new();
            source.clone().add_downstream(receiver.clone());
            sources.push(source);
            receivers.push(receiver);
            gates.push(gate);
        }
        let probe = Probe::new();
        *lock(&probe.fake) = Some(fake.clone());
        Self {
            fake,
            pool,
            intakes,
            sources,
            receivers,
            gates,
            probe,
        }
    }

    pub fn job(&self, tag: u64) -> Job {
        Job::new(&self.probe, tag)
    }

    pub fn deliver(&self, job: Job, to: Deliver) {
        self.intakes.deliver(job, to).unwrap();
    }

    pub fn lane(&self, lane: u16) -> Deliver {
        Deliver::Lane(LaneId(lane))
    }

    /// A marker of lane `lane` of this rig, with `value`.
    pub fn marker(&self, lane: u16, value: u64) -> Marker {
        Marker::new(self.pool.lane(LaneId(lane)).id(), LaneId(lane), value)
    }

    pub fn received(&self, source: usize) -> Vec<u64> {
        self.receivers[source].received()
    }
}
