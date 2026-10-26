//! Static stage DAGs run per quantum, with submission, completion and external edges
//! (Euhedral-Inference `StageGraph`).

use std::fmt;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, RwLock};
use std::time::{Duration, Instant};

use euhedral_core::frames::{Frame, FrameError, FrameHeader};
use euhedral_core::generics::FrameSink;

use crate::completions::{Completions, Ready};
use crate::lanes::{LaneId, LanePool, Marker};

/// A stage's index in its [`StageGraphSpec`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct StageId(pub u16);

/// When an edge is satisfied.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EdgeKind {
    /// When the predecessor *submitted*. The successor orders itself behind the predecessor's
    /// marker on the device (a wait in its own submission), so the host never waits.
    Submission,
    /// When the predecessor's work *completed* (its marker reached, or its completion ready), found
    /// by a [`Completions`] poll. For successors that need the result on the host.
    Completion,
}

/// One stage of a graph.
#[derive(Clone, Debug)]
pub struct StageSpec {
    pub name: &'static str,
    /// `Some(id_hash)`: the stage's frames run on that ordered lane, in the order they are offered
    /// (a stateful codec, a socket). `None`: unordered.
    pub ordered: Option<u64>,
    /// The device lanes able to run the stage's work; empty for a CPU-only stage.
    pub lanes: Vec<LaneId>,
}

/// The shape of a graph: stages and edges, validated once (acyclic, ids in range) and reused for
/// every quantum.
#[derive(Clone, Debug, Default)]
pub struct StageGraphSpec {
    stages: Vec<StageSpec>,
    edges: Vec<(StageId, StageId, EdgeKind)>,
}

impl StageGraphSpec {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn stage(&mut self, spec: StageSpec) -> StageId {
        let id = StageId(u16::try_from(self.stages.len()).expect("at most 65536 stages"));
        self.stages.push(spec);
        id
    }

    pub fn edge(&mut self, from: StageId, to: StageId, kind: EdgeKind) -> &mut Self {
        self.edges.push((from, to, kind));
        self
    }

    pub fn stages(&self) -> &[StageSpec] {
        &self.stages
    }

    pub fn edges(&self) -> &[(StageId, StageId, EdgeKind)] {
        &self.edges
    }

    /// Checks the ids and that the graph is acyclic.
    pub fn validate(&self) -> Result<(), GraphError> {
        Topology::of(self).map(|_| ())
    }
}

/// What a stage's run submitted.
#[derive(Default)]
pub struct Submitted {
    /// The marker of the device work it queued, if any: successors on other lanes wait for it on
    /// the device; completion edges wait for it to be reached.
    pub marker: Option<Marker>,
    /// A host-side completion other than a marker (a driver call that has to be polled, a model
    /// request), if any: completion edges wait for it.
    pub completion: Option<Box<dyn Ready>>,
}

impl fmt::Debug for Submitted {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Submitted")
            .field("marker", &self.marker)
            .field("completion", &self.completion.is_some())
            .finish()
    }
}

impl Submitted {
    /// A CPU-only stage that finished: every outgoing edge is satisfied at once.
    pub fn done() -> Self {
        Self::default()
    }

    pub fn marker(marker: Marker) -> Self {
        Self {
            marker: Some(marker),
            completion: None,
        }
    }
}

/// Why a stage failed. The quantum then ends with [`QuantumOutcome::Failed`]: its later stages get
/// [`Stage::abandoned`], and it still retires.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StageError {
    /// This quantum's input was bad (a corrupt packet); the next quantum may be fine.
    Corrupt(String),
    /// The stage's resource is unusable until reconfigured.
    Fatal(String),
    /// The stage's resources are busy (a ring entry, a slot still in use on the device): it runs
    /// again once any lane of the pool completed more work, or after
    /// [`GraphRuntime::retry_backstop`] (a deadline the [`Completions`] poll checks, for a resource
    /// outside the pool). It never ends the quantum; a kill or a stop while it waits does.
    Retry,
}

/// The default for [`GraphRuntime::retry_backstop`]: how long a retried stage waits at most for a
/// lane's progress before it runs again.
///
/// Set it longer than any one pass on your devices, so that it fires only when no lane completed
/// anything for that long: a stalled lane, or a busy resource outside the pool. A backstop shorter
/// than a pass fires on every pass, and a firing then means nothing. Twenty milliseconds suits
/// passes of up to about ten.
pub const DEFAULT_RETRY_BACKSTOP: Duration = Duration::from_millis(20);

/// Retried stages, counted by how they ran again (all graphs of a runtime).
#[derive(Debug, Default)]
pub struct RetryCounts {
    retries: AtomicU64,
    progress: AtomicU64,
    backstops: AtomicU64,
}

impl RetryCounts {
    /// Stage runs that answered [`StageError::Retry`].
    pub fn retries(&self) -> u64 {
        self.retries.load(Ordering::Relaxed)
    }

    /// Retried stages woken by a lane's progress.
    pub fn progress(&self) -> u64 {
        self.progress.load(Ordering::Relaxed)
    }

    /// Retried stages woken by [`GraphRuntime::retry_backstop`] (no lane of the pool progressed:
    /// the busy resource is elsewhere).
    pub fn backstops(&self) -> u64 {
        self.backstops.load(Ordering::Relaxed)
    }
}

/// How a quantum ended.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum QuantumOutcome {
    Completed,
    Failed {
        stage: StageId,
        error: StageError,
    },
    /// Its kill switch was set (the work it belonged to was cancelled or superseded).
    Killed,
}

/// What a stage sees while it runs.
#[derive(Debug)]
pub struct StageCx<'a> {
    pub stage: StageId,
    /// The lane the stage submits to (`None` for a CPU-only stage).
    pub lane: Option<LaneId>,
    /// Markers its submission must wait for on the device: submission-edge predecessors on other
    /// lanes, and external edges. Empty when every predecessor ran on `lane`.
    pub waits: &'a [Marker],
    pub lanes: &'a LanePool,
}

/// One stage's work for a quantum `Q`. It computes or submits, and returns; it never waits.
pub trait Stage<Q>: Send + Sync {
    fn run(&self, quantum: &Q, cx: &StageCx<'_>) -> Result<Submitted, StageError>;

    /// The quantum ended (killed, or a predecessor failed) before this stage ran.
    fn abandoned(&self, quantum: &Q) {
        let _ = quantum;
    }
}

/// The RustEE side a graph publishes through.
#[derive(Clone)]
pub struct GraphRuntime {
    /// Where stage frames are offered: any sink of the lattice's frames, usually a
    /// [`Lake`](crate::Lake) (whose units are the graph's quanta, so the caller admits one before
    /// it starts a quantum and drops its guard when the quantum retires).
    pub stages: Arc<dyn FrameSink>,
    pub completions: Arc<Completions>,
    pub lanes: Arc<LanePool>,
    /// Shared by the runtime's graphs.
    pub retries: Arc<RetryCounts>,
    /// How long a retried stage waits at most for a lane's progress before it runs again; see
    /// [`DEFAULT_RETRY_BACKSTOP`].
    pub retry_backstop: Duration,
}

impl fmt::Debug for GraphRuntime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GraphRuntime")
            .field("lanes", &self.lanes.len())
            .finish_non_exhaustive()
    }
}

/// A started quantum, for later quanta's external edges.
///
/// **It pins the quantum:** a quantum retires only once its own stages are done *and* every clone
/// of its `QuantumRef` has been dropped, so a later quantum that names it in an [`ExternalEdge`]
/// can still use its results. Drop it at once when nothing will depend on the quantum (`let _ =
/// graph.start(..)`); to keep quantum n's results for n+1, hold n's ref until n+1's
/// [`StageGraph::start`] has taken its edge (the edge holds its own clone until its stage used it),
/// and drop held refs on a kill or at the end of the stream, or that quantum never retires.
#[derive(Clone)]
pub struct QuantumRef {
    pin: Arc<Pin>,
}

impl fmt::Debug for QuantumRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QuantumRef").finish_non_exhaustive()
    }
}

/// A dependency of a new quantum's stage `to_stage` on stage `from_stage` of an earlier quantum:
/// the temporal window of a stage on a window of earlier quanta, a pair of neighbours, carried
/// state. It holds a live count of the earlier quantum until it arrives, so the earlier quantum
/// cannot retire (release its resources) before the later one has used it.
///
/// Precisely: the edge arrives when `from_stage` submitted (a [`Submission`](EdgeKind::Submission)
/// edge: `to_stage` waits for its markers on the device) or once those markers are reached (a
/// [`Completion`](EdgeKind::Completion) edge; a host completion of `from_stage` is not shared). If
/// `from_stage` will never run (its quantum was killed or failed), the edge arrives anyway, with
/// every marker that quantum submitted. The earlier quantum stays pinned until `to_stage` has
/// submitted (its marker joins the earlier quantum's retirement boundary) or been abandoned.
#[derive(Clone, Debug)]
pub struct ExternalEdge {
    pub from: QuantumRef,
    pub from_stage: StageId,
    pub to_stage: StageId,
    pub kind: EdgeKind,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GraphError {
    UnknownStage(StageId),
    Cycle(Vec<StageId>),
    /// The number of stage implementations differs from the spec's stages.
    StageCount {
        spec: usize,
        given: usize,
    },
}

impl fmt::Display for GraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownStage(id) => write!(f, "unknown stage {}", id.0),
            Self::Cycle(ids) => write!(
                f,
                "cycle through stages {:?}",
                ids.iter().map(|s| s.0).collect::<Vec<_>>()
            ),
            Self::StageCount { spec, given } => {
                write!(f, "{given} stages given for a spec of {spec}")
            }
        }
    }
}

impl std::error::Error for GraphError {}

/// A validated graph whose quanta run on RustEE.
///
/// - [`start`](Self::start) binds a quantum and publishes its roots (once its external edges
///   arrived). The last arrival at a stage publishes it; the graph never scans for ready work.
/// - Each quantum counts its published frames and armed completion edges. At zero no stage can run
///   again; the quantum's retirement boundary (every lane it used reached) is armed with
///   [`Completions`], and the retirement frame hands the quantum and its [`QuantumOutcome`] to
///   `retire`.
/// - A killed quantum's frames are skipped by RustEE and their `do_finally` drops their counts; its
///   stages get [`Stage::abandoned`]. Device work already submitted runs to its end and is covered
///   by the boundary.
///
/// More precisely:
/// - **Lanes:** a stage with device lanes continues the lane its critical-path predecessor (the
///   predecessor on its longest path from a root) ran on, when allowed, and otherwise takes its
///   lanes in turn ([`LanePool::choose`]).
/// - **Waits:** a stage's [`StageCx::waits`] hold, per lane, the latest marker of its submission
///   predecessors and external edges, minus its own lane's. A stage that submitted no marker
///   (CPU-only) passes its inputs on, so device order is kept through it.
/// - **The boundary** is the latest marker per lane of everything the quantum submitted (plus the
///   markers of later quanta's stages that used it through external edges) and every host
///   completion no completion edge consumed.
/// - **The outcome** is the first of: a stage's failure (an `Err`, or a panic as
///   [`StageError::Fatal`]), or the kill switch found set before a stage or edge ran. Otherwise it
///   is [`QuantumOutcome::Completed`].
/// - **Stage frames** carry the quantum's kill switch, are ordered by [`StageSpec::ordered`] or
///   else unordered, and run the stage in `execute` (RustEE turns a panic there into
///   `do_finally_with_error`).
/// - **Order across quanta:** an ordered stage's frames run in the order they are published.
///   Successors through submission edges are published by the predecessor's frame; through
///   completion edges, by an arrival frame on the predecessor's ordered lane (completions are
///   emitted in registration order), so an ordered predecessor's order carries over to an ordered
///   successor (a fetch → parse chain).
pub struct StageGraph<Q> {
    spec: StageGraphSpec,
    stages: Vec<Arc<dyn Stage<Q>>>,
    topology: Topology,
    runtime: GraphRuntime,
    retire: Box<dyn Fn(Q, QuantumOutcome) + Send + Sync>,
    live: AtomicUsize,
    seeds: AtomicU64,
}

impl<Q> fmt::Debug for StageGraph<Q> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StageGraph")
            .field("stages", &self.spec.stages.len())
            .field("live", &self.live.load(Ordering::Relaxed))
            .finish_non_exhaustive()
    }
}

/// The id hash of a graph's unordered frames (their routing hash is randomized per frame).
const UNORDERED: u64 = 0x5747_e6a9_0f10_77a1;
const NO_LANE: u32 = u32::MAX;

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Adds `marker` to `markers`, keeping the latest per lane (a lane runs in order, so it covers the
/// earlier ones).
fn merge(markers: &mut Vec<Marker>, marker: Marker) {
    match markers.iter_mut().find(|m| m.lane == marker.lane) {
        Some(m) => m.value = m.value.max(marker.value),
        None => markers.push(marker),
    }
}

/// The edges of a validated spec, by stage.
#[derive(Debug)]
struct Topology {
    successors: Vec<Vec<(StageId, EdgeKind)>>,
    incoming: Vec<usize>,
    has_completion_successor: Vec<bool>,
    /// The predecessor on each stage's longest path from a root (the first listed on ties).
    path_predecessor: Vec<Option<StageId>>,
    roots: Vec<StageId>,
}

impl Topology {
    fn of(spec: &StageGraphSpec) -> Result<Self, GraphError> {
        let n = spec.stages.len();
        let mut successors = vec![Vec::new(); n];
        let mut predecessors: Vec<Vec<StageId>> = vec![Vec::new(); n];
        let mut incoming = vec![0; n];
        let mut has_completion_successor = vec![false; n];
        for &(from, to, kind) in &spec.edges {
            for id in [from, to] {
                if usize::from(id.0) >= n {
                    return Err(GraphError::UnknownStage(id));
                }
            }
            successors[usize::from(from.0)].push((to, kind));
            predecessors[usize::from(to.0)].push(from);
            incoming[usize::from(to.0)] += 1;
            has_completion_successor[usize::from(from.0)] |= kind == EdgeKind::Completion;
        }
        // Kahn's order, with each stage's depth (longest path from a root).
        let mut pending = incoming.clone();
        let mut ready: Vec<usize> = (0..n).filter(|&s| pending[s] == 0).collect();
        let roots = ready.iter().map(|&s| StageId(s as u16)).collect();
        let mut order = Vec::with_capacity(n);
        while let Some(stage) = ready.pop() {
            order.push(stage);
            for &(to, _) in &successors[stage] {
                let to = usize::from(to.0);
                pending[to] -= 1;
                if pending[to] == 0 {
                    ready.push(to);
                }
            }
        }
        if order.len() < n {
            return Err(GraphError::Cycle(Self::a_cycle(&predecessors, &pending)));
        }
        let mut depth = vec![0usize; n];
        let mut path_predecessor = vec![None; n];
        for &stage in &order {
            for &pred in &predecessors[stage] {
                let d = depth[usize::from(pred.0)] + 1;
                if path_predecessor[stage].is_none() || d > depth[stage] {
                    depth[stage] = d;
                    path_predecessor[stage] = Some(pred);
                }
            }
        }
        Ok(Self {
            successors,
            incoming,
            has_completion_successor,
            path_predecessor,
            roots,
        })
    }

    /// A cycle among the stages Kahn's order could not place, in edge order. Each of them waits for
    /// a predecessor that is also stuck, so walking predecessors must come back to a stage already
    /// seen.
    fn a_cycle(predecessors: &[Vec<StageId>], pending: &[usize]) -> Vec<StageId> {
        let stuck = |s: usize| pending[s] > 0;
        let mut at = (0..pending.len())
            .find(|&s| stuck(s))
            .expect("a stage on a cycle");
        let mut seen = vec![usize::MAX; pending.len()];
        let mut path = Vec::new();
        while seen[at] == usize::MAX {
            seen[at] = path.len();
            path.push(at);
            at = predecessors[at]
                .iter()
                .map(|p| usize::from(p.0))
                .find(|&p| stuck(p))
                .expect("a stuck predecessor");
        }
        let mut cycle: Vec<StageId> = path[seen[at]..]
            .iter()
            .map(|&s| StageId(s as u16))
            .collect();
        cycle.reverse();
        // Start the cycle at its lowest stage, for a stable report.
        let lowest = cycle
            .iter()
            .enumerate()
            .min_by_key(|(_, s)| s.0)
            .map_or(0, |(i, _)| i);
        cycle.rotate_left(lowest);
        cycle
    }
}

impl<Q: Send + Sync + 'static> StageGraph<Q> {
    /// `stages[i]` implements `spec.stages()[i]`. `retire` runs once per quantum, in a frame, after
    /// its boundary.
    pub fn new(
        spec: StageGraphSpec,
        stages: Vec<Arc<dyn Stage<Q>>>,
        runtime: GraphRuntime,
        retire: impl Fn(Q, QuantumOutcome) + Send + Sync + 'static,
    ) -> Result<Arc<Self>, GraphError> {
        if stages.len() != spec.stages.len() {
            return Err(GraphError::StageCount {
                spec: spec.stages.len(),
                given: stages.len(),
            });
        }
        let topology = Topology::of(&spec)?;
        for stage in &spec.stages {
            for lane in &stage.lanes {
                assert!(
                    usize::from(lane.0) < runtime.lanes.len(),
                    "stage {} names lane {} of a pool of {}",
                    stage.name,
                    lane.0,
                    runtime.lanes.len()
                );
            }
        }
        Ok(Arc::new(Self {
            spec,
            stages,
            topology,
            runtime,
            retire: Box::new(retire),
            live: AtomicUsize::new(0),
            seeds: AtomicU64::new(0),
        }))
    }

    /// Starts `quantum`; its frames share `kill`.
    pub fn start(
        self: &Arc<Self>,
        quantum: Q,
        kill: Arc<AtomicBool>,
        external: Vec<ExternalEdge>,
    ) -> QuantumRef {
        let n = self.spec.stages.len();
        for edge in &external {
            assert!(
                usize::from(edge.to_stage.0) < n,
                "an external edge to unknown stage {}",
                edge.to_stage.0
            );
        }
        let run = Arc::new(Run {
            graph: self.clone(),
            quantum: RwLock::new(Some(quantum)),
            kill,
            live: AtomicUsize::new(0),
            // The returned ref's pin, and the quantum's own token (dropped once its live count
            // reached 0).
            holds: AtomicUsize::new(2),
            stopped: AtomicBool::new(false),
            cause: Mutex::new(None),
            stages: (0..n)
                .map(|s| StageRun {
                    pending: AtomicUsize::new(self.topology.incoming[s]),
                    inputs: Mutex::new(Vec::new()),
                    lane: AtomicU32::new(NO_LANE),
                    ran: AtomicBool::new(false),
                })
                .collect(),
            externals: Mutex::new(Externals {
                swept: false,
                out: vec![None; n],
                waiters: Vec::new(),
            }),
            boundary: Mutex::new(BoundarySet::default()),
            pins: Mutex::new(Vec::new()),
        });
        self.live.fetch_add(1, Ordering::AcqRel);
        for edge in &external {
            run.stages[usize::from(edge.to_stage.0)]
                .pending
                .fetch_add(1, Ordering::AcqRel);
        }
        let roots: Vec<StageId> = self
            .topology
            .roots
            .iter()
            .copied()
            .filter(|s| run.stages[usize::from(s.0)].pending.load(Ordering::Acquire) == 0)
            .collect();
        // The roots, admission's hold (so fast roots cannot retire the quantum before every root
        // and edge is in), and one count per external edge until it arrives.
        run.live
            .store(roots.len() + 1 + external.len(), Ordering::Release);
        let reference = QuantumRef {
            pin: Arc::new(Pin { run: run.clone() }),
        };
        for root in roots {
            run.offer(root);
        }
        for edge in external {
            run.register_external(edge);
        }
        run.finish_one();
        reference
    }

    /// Quanta started and not yet retired.
    pub fn live(&self) -> usize {
        self.live.load(Ordering::Acquire)
    }

    fn next_seed(&self) -> u64 {
        self.seeds.fetch_add(1, Ordering::Relaxed)
    }
}

/// What a later quantum's external edges need of an earlier one, of any graph (type-erased).
/// What an external edge does when its earlier stage is satisfied: arrive with the stage's markers.
type ArriveWith = Box<dyn FnOnce(Vec<Marker>) + Send>;

trait Upstream: Send + Sync {
    /// Calls `arrive` with `stage`'s markers once it submitted, or with every marker of the quantum
    /// once it is known never to run; at once if either is already so.
    fn subscribe(&self, stage: StageId, arrive: ArriveWith);
    /// A later quantum's stage that used this one submitted `marker`: the retirement boundary
    /// covers it.
    fn cover(&self, marker: Marker);
    /// One pin less; the last lets the quantum retire.
    fn release_hold(self: Arc<Self>);
}

/// One pin of a quantum: the [`QuantumRef`] clones share one, and an external edge in use holds
/// one.
struct Pin {
    run: Arc<dyn Upstream>,
}

impl Drop for Pin {
    fn drop(&mut self) {
        self.run.clone().release_hold();
    }
}

/// A quantum bound to its graph.
struct Run<Q: Send + Sync + 'static> {
    graph: Arc<StageGraph<Q>>,
    /// Read while stages run (concurrently); taken by the retirement frame, after every stage is
    /// done.
    quantum: RwLock<Option<Q>>,
    kill: Arc<AtomicBool>,
    /// Published frames, armed completion edges, external edges not yet arrived, and admission's
    /// hold.
    live: AtomicUsize,
    /// The quantum's own token, its `QuantumRef` pin, and the external edges of later quanta that
    /// use it.
    holds: AtomicUsize,
    /// A stage failed, or a frame was refused: nothing more is published.
    stopped: AtomicBool,
    /// The first reason the quantum did not complete.
    cause: Mutex<Option<QuantumOutcome>>,
    stages: Box<[StageRun]>,
    externals: Mutex<Externals>,
    boundary: Mutex<BoundarySet>,
    /// The pins this quantum's external edges hold on earlier quanta, by the stage that uses them.
    pins: Mutex<Vec<(StageId, QuantumRef)>>,
}

struct StageRun {
    /// Incoming edges (and external edges) not yet arrived.
    pending: AtomicUsize,
    /// The markers that arrived with them (latest per lane).
    inputs: Mutex<Vec<Marker>>,
    /// The lane it ran on, or `NO_LANE`.
    lane: AtomicU32,
    ran: AtomicBool,
}

struct Externals {
    /// The quantum's stages can no longer run: every later subscription arrives at once.
    swept: bool,
    /// Each stage's output markers, once it submitted.
    out: Vec<Option<Vec<Marker>>>,
    waiters: Vec<(StageId, ArriveWith)>,
}

#[derive(Default)]
struct BoundarySet {
    markers: Vec<Marker>,
    hosts: Vec<Box<dyn Ready>>,
}

/// A retried stage's wake-up: any lane of the pool past where it was at the retry, or the
/// backstop's deadline.
struct Progress {
    lanes: Arc<LanePool>,
    seen: Vec<u64>,
    deadline: Instant,
    counts: Arc<RetryCounts>,
}

impl Progress {
    fn new(lanes: Arc<LanePool>, counts: Arc<RetryCounts>, backstop: Duration) -> Self {
        let seen = (0..lanes.len())
            .map(|i| lanes.lane(LaneId(i as u16)).completed())
            .collect();
        Progress {
            lanes,
            seen,
            deadline: Instant::now() + backstop,
            counts,
        }
    }
}

impl Ready for Progress {
    fn is_ready(&mut self) -> bool {
        let lanes = &self.lanes;
        if self
            .seen
            .iter()
            .enumerate()
            .any(|(i, &v)| lanes.lane(LaneId(i as u16)).completed() > v)
        {
            self.counts.progress.fetch_add(1, Ordering::Relaxed);
            return true;
        }
        if Instant::now() >= self.deadline {
            self.counts.backstops.fetch_add(1, Ordering::Relaxed);
            return true;
        }
        false
    }
}

/// Every marker reached and every host completion ready.
struct AllOf {
    lanes: Arc<LanePool>,
    markers: Vec<Marker>,
    hosts: Vec<Box<dyn Ready>>,
}

impl Ready for AllOf {
    fn is_ready(&mut self) -> bool {
        let lanes = &self.lanes;
        self.markers.retain(|m| !lanes.reached(*m));
        self.hosts.retain_mut(|host| !host.is_ready());
        self.markers.is_empty() && self.hosts.is_empty()
    }
}

impl<Q: Send + Sync + 'static> Run<Q> {
    fn halted(&self) -> bool {
        self.stopped.load(Ordering::Acquire) || self.kill.load(Ordering::Acquire)
    }

    fn note(&self, cause: QuantumOutcome) {
        lock(&self.cause).get_or_insert(cause);
    }

    fn fail(&self, stage: StageId, error: StageError) {
        self.note(QuantumOutcome::Failed { stage, error });
        self.stopped.store(true, Ordering::Release);
    }

    fn header(&self, ordered: Option<u64>, kill: bool) -> FrameHeader {
        let kill = kill.then(|| self.kill.clone());
        match ordered {
            Some(id) => FrameHeader::with_kill_switch(id, kill),
            None => {
                let mut header = FrameHeader::with_kill_switch(UNORDERED, kill);
                header.randomize_hash(self.graph.next_seed());
                header
            }
        }
    }

    /// Offers `frame`; a refused one (the sink completed) ends in place.
    fn offer_frame(&self, frame: Box<dyn Frame>) {
        if let Err(frame) = self.graph.runtime.stages.offer(frame) {
            frame.do_finally();
        }
    }

    /// Publishes `stage`, whose last incoming edge arrived (its live count is already taken).
    fn offer(self: &Arc<Self>, stage: StageId) {
        let ordered = self.graph.spec.stages[usize::from(stage.0)].ordered;
        let frame = StageFrame {
            header: self.header(ordered, true),
            run: self.clone(),
            stage,
            result: None,
        };
        if let Err(frame) = self.graph.runtime.stages.offer(Box::new(frame)) {
            self.fail(
                stage,
                StageError::Fatal("the stage sink no longer takes frames".into()),
            );
            frame.do_finally();
        }
    }

    /// An edge into `stage` arrived with `markers`; the last one publishes it.
    fn arrive(self: &Arc<Self>, stage: StageId, markers: &[Marker]) {
        let state = &self.stages[usize::from(stage.0)];
        if !markers.is_empty() {
            let mut inputs = lock(&state.inputs);
            for &marker in markers {
                merge(&mut inputs, marker);
            }
        }
        if state.pending.fetch_sub(1, Ordering::AcqRel) == 1 {
            if self.halted() {
                if self.kill.load(Ordering::Acquire) {
                    self.note(QuantumOutcome::Killed);
                }
                return;
            }
            self.live.fetch_add(1, Ordering::AcqRel);
            self.offer(stage);
        }
    }

    /// Drops one live count; the last one quiesces the quantum.
    fn finish_one(self: &Arc<Self>) {
        if self.live.fetch_sub(1, Ordering::AcqRel) == 1 {
            self.quiesce();
        }
    }

    /// Runs `stage` (in its frame's `execute`), unless the quantum stopped.
    fn run_stage(&self, stage: StageId) -> Option<Result<Submitted, StageError>> {
        if self.stopped.load(Ordering::Acquire) {
            return None;
        }
        let index = usize::from(stage.0);
        let state = &self.stages[index];
        state.ran.store(true, Ordering::Release);
        let spec = &self.graph.spec.stages[index];
        let lanes = &*self.graph.runtime.lanes;
        let lane = (!spec.lanes.is_empty()).then(|| {
            let continued = self.graph.topology.path_predecessor[index]
                .map(|p| self.stages[usize::from(p.0)].lane.load(Ordering::Acquire))
                .filter(|&l| l != NO_LANE)
                .map(|l| LaneId(l as u16));
            lanes.choose(&spec.lanes, continued)
        });
        if let Some(lane) = lane {
            state.lane.store(u32::from(lane.0), Ordering::Release);
        }
        let waits: Vec<Marker> = lock(&state.inputs)
            .iter()
            .copied()
            .filter(|m| Some(m.lane) != lane)
            .collect();
        let quantum = self.quantum.read().unwrap_or_else(PoisonError::into_inner);
        let quantum = quantum.as_ref().expect("a running quantum is bound");
        let cx = StageCx {
            stage,
            lane,
            waits: &waits,
            lanes,
        };
        Some(self.graph.stages[index].run(quantum, &cx))
    }

    /// `stage` answered [`StageError::Retry`]: it runs again on a lane's progress or at the
    /// backstop (its live count is kept).
    fn retry(self: &Arc<Self>, stage: StageId) {
        // It has not run: a kill while it waits abandons it.
        self.stages[usize::from(stage.0)]
            .ran
            .store(false, Ordering::Release);
        let runtime = &self.graph.runtime;
        runtime.retries.retries.fetch_add(1, Ordering::Relaxed);
        let ready = Progress::new(
            runtime.lanes.clone(),
            runtime.retries.clone(),
            runtime.retry_backstop,
        );
        let frame = Arrival {
            header: self.header(None, true),
            run: self.clone(),
            to: Target::Retry(stage),
        };
        runtime.completions.after(Box::new(ready), Box::new(frame));
    }

    /// `stage` submitted: its edges are satisfied (submission edges now, completion edges once its
    /// work completed).
    fn submitted(self: &Arc<Self>, stage: StageId, submitted: Submitted) {
        let index = usize::from(stage.0);
        let Submitted { marker, completion } = submitted;
        // Device order passes through a stage that queued nothing.
        let out = match marker {
            Some(marker) => vec![marker],
            None => lock(&self.stages[index].inputs).clone(),
        };
        let completion = {
            let mut boundary = lock(&self.boundary);
            if let Some(marker) = marker {
                merge(&mut boundary.markers, marker);
            }
            if self.graph.topology.has_completion_successor[index] {
                completion
            } else {
                boundary.hosts.extend(completion);
                None
            }
        };
        self.release_pins(stage, marker);
        self.notify_externals(stage, &out);
        if self.halted() {
            if self.kill.load(Ordering::Acquire) {
                self.note(QuantumOutcome::Killed);
            }
            return;
        }
        if self.graph.topology.has_completion_successor[index] {
            self.live.fetch_add(1, Ordering::AcqRel);
            let ready = AllOf {
                lanes: self.graph.runtime.lanes.clone(),
                markers: marker.into_iter().collect(),
                hosts: completion.into_iter().collect(),
            };
            // On the predecessor's ordered lane, if it has one: its completions (emitted in
            // registration order) then publish their successors in that order, so an ordered
            // successor keeps the predecessor's order.
            let ordered = self.graph.spec.stages[index].ordered;
            let frame = Arrival {
                header: self.header(ordered, true),
                run: self.clone(),
                to: Target::Completions(stage),
            };
            self.graph
                .runtime
                .completions
                .after(Box::new(ready), Box::new(frame));
        }
        for &(successor, kind) in &self.graph.topology.successors[index] {
            if kind == EdgeKind::Submission {
                self.arrive(successor, &out);
            }
        }
    }

    /// Ends the pins this quantum's external edges into `stage` hold, the used stage's `marker`
    /// joining the earlier quanta's boundaries.
    fn release_pins(&self, stage: StageId, marker: Option<Marker>) {
        let released: Vec<QuantumRef> = {
            let mut pins = lock(&self.pins);
            let (mine, rest) = std::mem::take(&mut *pins)
                .into_iter()
                .partition(|(to, _)| *to == stage);
            *pins = rest;
            mine.into_iter().map(|(_, pin)| pin).collect()
        };
        for pin in released {
            if let Some(marker) = marker {
                pin.pin.run.cover(marker);
            }
            drop(pin);
        }
    }

    fn notify_externals(&self, stage: StageId, out: &[Marker]) {
        let waiters: Vec<_> = {
            let mut externals = lock(&self.externals);
            externals.out[usize::from(stage.0)] = Some(out.to_vec());
            let (mine, rest) = std::mem::take(&mut externals.waiters)
                .into_iter()
                .partition(|(s, _)| *s == stage);
            externals.waiters = rest;
            mine
        };
        for (_, arrive) in waiters {
            arrive(out.to_vec());
        }
    }

    fn register_external(self: &Arc<Self>, edge: ExternalEdge) {
        let ExternalEdge {
            from,
            from_stage,
            to_stage,
            kind,
        } = edge;
        let upstream = from.pin.run.clone();
        lock(&self.pins).push((to_stage, from));
        let me = self.clone();
        upstream.subscribe(
            from_stage,
            Box::new(move |markers| me.external_arrived(to_stage, kind, markers)),
        );
    }

    fn external_arrived(self: &Arc<Self>, to: StageId, kind: EdgeKind, markers: Vec<Marker>) {
        if kind == EdgeKind::Completion && !markers.is_empty() && !self.halted() {
            // The edge's live count passes to the frame that arrives once the markers are reached.
            let ready = AllOf {
                lanes: self.graph.runtime.lanes.clone(),
                markers,
                hosts: Vec::new(),
            };
            let frame = Arrival {
                header: self.header(None, true),
                run: self.clone(),
                to: Target::Stage(to),
            };
            self.graph
                .runtime
                .completions
                .after(Box::new(ready), Box::new(frame));
            return;
        }
        let markers = if kind == EdgeKind::Completion {
            Vec::new()
        } else {
            markers
        };
        self.arrive(to, &markers);
        self.finish_one();
    }

    /// No stage of the quantum can run again: later quanta waiting on its stages are released, its
    /// unrun stages abandoned, and its own token dropped (the boundary is armed once no pin is
    /// left).
    fn quiesce(self: &Arc<Self>) {
        let (waiters, frontier) = {
            let mut externals = lock(&self.externals);
            externals.swept = true;
            (
                std::mem::take(&mut externals.waiters),
                lock(&self.boundary).markers.clone(),
            )
        };
        for (_, arrive) in waiters {
            arrive(frontier.clone());
        }
        let remaining: Vec<QuantumRef> = std::mem::take(&mut *lock(&self.pins))
            .into_iter()
            .map(|(_, p)| p)
            .collect();
        drop(remaining);
        {
            let quantum = self.quantum.read().unwrap_or_else(PoisonError::into_inner);
            let quantum = quantum.as_ref().expect("a quiescing quantum is bound");
            for (index, state) in self.stages.iter().enumerate() {
                if !state.ran.load(Ordering::Acquire) {
                    self.graph.stages[index].abandoned(quantum);
                }
            }
        }
        self.clone().release_hold();
    }

    /// Every pin is gone: arms the retirement boundary.
    fn arm_boundary(self: &Arc<Self>) {
        let BoundarySet { markers, hosts } = std::mem::take(&mut *lock(&self.boundary));
        let frame = Box::new(Retirement {
            header: self.header(None, false),
            run: self.clone(),
        });
        if markers.is_empty() && hosts.is_empty() {
            self.offer_frame(frame);
        } else {
            let ready = AllOf {
                lanes: self.graph.runtime.lanes.clone(),
                markers,
                hosts,
            };
            self.graph.runtime.completions.after(Box::new(ready), frame);
        }
    }

    fn conclude(&self) {
        let quantum = self
            .quantum
            .write()
            .unwrap_or_else(PoisonError::into_inner)
            .take();
        let Some(quantum) = quantum else { return };
        let outcome = lock(&self.cause)
            .take()
            .unwrap_or(QuantumOutcome::Completed);
        let retired = catch_unwind(AssertUnwindSafe(|| (self.graph.retire)(quantum, outcome)));
        self.graph.live.fetch_sub(1, Ordering::AcqRel);
        if let Err(panic) = retired {
            resume_unwind(panic);
        }
    }
}

impl<Q: Send + Sync + 'static> Upstream for Run<Q> {
    fn subscribe(&self, stage: StageId, arrive: ArriveWith) {
        let now = {
            let mut externals = lock(&self.externals);
            if let Some(out) = &externals.out[usize::from(stage.0)] {
                Some(out.clone())
            } else if externals.swept {
                Some(lock(&self.boundary).markers.clone())
            } else {
                externals.waiters.push((stage, arrive));
                return;
            }
        };
        if let Some(markers) = now {
            arrive(markers);
        }
    }

    fn cover(&self, marker: Marker) {
        merge(&mut lock(&self.boundary).markers, marker);
    }

    fn release_hold(self: Arc<Self>) {
        if self.holds.fetch_sub(1, Ordering::AcqRel) == 1 {
            self.arm_boundary();
        }
    }
}

/// A stage of a quantum: runs it in `execute`, hands on to its successors in `do_finally`.
struct StageFrame<Q: Send + Sync + 'static> {
    header: FrameHeader,
    run: Arc<Run<Q>>,
    stage: StageId,
    result: Option<Result<Submitted, StageError>>,
}

impl<Q: Send + Sync + 'static> Frame for StageFrame<Q> {
    fn header(&self) -> &FrameHeader {
        &self.header
    }

    fn header_mut(&mut self) -> &mut FrameHeader {
        &mut self.header
    }

    fn execute(&mut self) -> Result<(), FrameError> {
        self.result = self.run.run_stage(self.stage);
        Ok(())
    }

    fn do_finally(mut self: Box<Self>) {
        match self.result.take() {
            Some(Ok(submitted)) => self.run.submitted(self.stage, submitted),
            Some(Err(StageError::Retry)) if self.header.is_alive() => {
                // The live count passes to the retry, which offers the stage again.
                self.run.retry(self.stage);
                return;
            }
            Some(Err(StageError::Retry)) => {
                self.run.stages[usize::from(self.stage.0)]
                    .ran
                    .store(false, Ordering::Release);
                self.run.note(QuantumOutcome::Killed);
            }
            Some(Err(error)) => {
                self.run.fail(self.stage, error);
                self.run.release_pins(self.stage, None);
            }
            None if !self.header.is_alive() => self.run.note(QuantumOutcome::Killed),
            None => {}
        }
        self.run.finish_one();
    }

    fn do_finally_with_error(self: Box<Self>, error: FrameError) {
        let name = self.run.graph.spec.stages[usize::from(self.stage.0)].name;
        self.run.stages[usize::from(self.stage.0)]
            .ran
            .store(true, Ordering::Release);
        self.run.fail(
            self.stage,
            StageError::Fatal(format!("stage {name} failed: {error}")),
        );
        self.run.release_pins(self.stage, None);
        self.run.finish_one();
    }
}

enum Target {
    /// The completion-edge successors of this stage.
    Completions(StageId),
    /// One stage, the target of an external completion edge.
    Stage(StageId),
    /// A retried stage, run again (its live count is the arrival's).
    Retry(StageId),
}

/// A completion edge's arrival, emitted by [`Completions`] once the predecessor's work completed.
struct Arrival<Q: Send + Sync + 'static> {
    header: FrameHeader,
    run: Arc<Run<Q>>,
    to: Target,
}

impl<Q: Send + Sync + 'static> Frame for Arrival<Q> {
    fn header(&self) -> &FrameHeader {
        &self.header
    }

    fn header_mut(&mut self) -> &mut FrameHeader {
        &mut self.header
    }

    fn do_finally(self: Box<Self>) {
        if !self.header.is_alive() {
            self.run.note(QuantumOutcome::Killed);
        } else if !self.run.stopped.load(Ordering::Acquire) {
            match self.to {
                Target::Completions(stage) => {
                    for &(successor, kind) in
                        &self.run.graph.topology.successors[usize::from(stage.0)]
                    {
                        if kind == EdgeKind::Completion {
                            self.run.arrive(successor, &[]);
                        }
                    }
                }
                Target::Stage(stage) => self.run.arrive(stage, &[]),
                Target::Retry(stage) => {
                    self.run.offer(stage);
                    return;
                }
            }
        }
        self.run.finish_one();
    }

    fn do_finally_with_error(self: Box<Self>, _: FrameError) {
        self.do_finally();
    }
}

/// The quantum's end, after its boundary: hands it and its outcome to `retire`.
struct Retirement<Q: Send + Sync + 'static> {
    header: FrameHeader,
    run: Arc<Run<Q>>,
}

impl<Q: Send + Sync + 'static> Frame for Retirement<Q> {
    fn header(&self) -> &FrameHeader {
        &self.header
    }

    fn header_mut(&mut self) -> &mut FrameHeader {
        &mut self.header
    }

    fn do_finally(self: Box<Self>) {
        self.run.conclude();
    }

    fn do_finally_with_error(self: Box<Self>, _: FrameError) {
        self.run.conclude();
    }
}

/// The port of Euhedral-Inference's `StageGraphTest`, `StageGraphHostStageTest` and
/// `CrossGraphEdgesTest` (the parts that apply: CUDA capture, transfer lanes, workspaces and graph
/// recycling have no counterpart here). Frames are run by hand, the way an executor runs them, from
/// the stage sink and the completions.
#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicU64;

    use euhedral_core::generics::LatticeSource;
    use euhedral_core::ingest::{IngestSink, QueueIngestSink};

    use super::*;
    use crate::DeviceQueue;

    /// A lane whose work completes when the test says so.
    #[derive(Debug, Default)]
    struct HandLane {
        submitted: AtomicU64,
        completed: AtomicU64,
    }

    impl HandLane {
        fn submit(&self) -> u64 {
            self.submitted.fetch_add(1, Ordering::AcqRel) + 1
        }

        fn finish(&self) {
            self.completed
                .store(self.submitted.load(Ordering::Acquire), Ordering::Release);
        }
    }

    impl DeviceQueue for HandLane {
        fn completed(&self) -> u64 {
            self.completed.load(Ordering::Acquire)
        }

        fn name(&self) -> &str {
            "hand"
        }
    }

    #[derive(Clone, Debug, PartialEq, Eq)]
    enum Event {
        Ran {
            q: u64,
            stage: u16,
            lane: Option<u16>,
            waits: Vec<(u16, u64)>,
        },
        Abandoned {
            q: u64,
            stage: u16,
        },
        Retired {
            q: u64,
            outcome: QuantumOutcome,
        },
    }

    type Log = Arc<Mutex<Vec<Event>>>;
    /// The lane a stage ran on and its waits, as (lane, value).
    type RanWith = (Option<u16>, Vec<(u16, u64)>);

    #[derive(Clone, Default)]
    struct Token(Arc<AtomicBool>);

    impl Ready for Token {
        fn is_ready(&mut self) -> bool {
            self.0.load(Ordering::Acquire)
        }
    }

    #[derive(Clone)]
    enum Act {
        Submit,
        Cpu,
        Fail,
        Panic,
        Host(Token),
        /// Answers `Retry` while the count is above 0 (counting down), then submits.
        Retry(Arc<AtomicU32>),
    }

    struct Probe {
        act: Act,
        log: Log,
        hands: Vec<Arc<HandLane>>,
    }

    impl Stage<u64> for Probe {
        fn run(&self, q: &u64, cx: &StageCx<'_>) -> Result<Submitted, StageError> {
            let waits = cx.waits.iter().map(|m| (m.lane.0, m.value)).collect();
            lock(&self.log).push(Event::Ran {
                q: *q,
                stage: cx.stage.0,
                lane: cx.lane.map(|l| l.0),
                waits,
            });
            if let Act::Retry(left) = &self.act
                && left
                    .try_update(Ordering::AcqRel, Ordering::Acquire, |n| n.checked_sub(1))
                    .is_ok()
            {
                return Err(StageError::Retry);
            }
            let marker = cx.lane.map(|lane| Marker {
                lane,
                value: self.hands[usize::from(lane.0)].submit(),
            });
            match &self.act {
                Act::Submit | Act::Retry(_) => Ok(Submitted {
                    marker,
                    completion: None,
                }),
                Act::Cpu => Ok(Submitted::done()),
                Act::Fail => Err(StageError::Corrupt("a bad packet".into())),
                Act::Panic => panic!("the stage broke"),
                Act::Host(token) => Ok(Submitted {
                    marker,
                    completion: Some(Box::new(token.clone())),
                }),
            }
        }
    }

    /// A probe that knows its own id, for `abandoned`.
    struct Named(u16, Probe);

    impl Stage<u64> for Named {
        fn run(&self, q: &u64, cx: &StageCx<'_>) -> Result<Submitted, StageError> {
            self.1.run(q, cx)
        }

        fn abandoned(&self, q: &u64) {
            lock(&self.1.log).push(Event::Abandoned {
                q: *q,
                stage: self.0,
            });
        }
    }

    /// A receiver that never asks for frames: the harness pulls them.
    struct Unattended;

    impl euhedral_core::generics::LatticeTerminal for Unattended {
        fn add_upstream(&self, _: Arc<dyn LatticeSource>) {}
    }

    impl euhedral_core::generics::LatticeReceiver for Unattended {
        fn push(&self, frame: Box<dyn Frame>) {
            run_frame(frame);
        }

        fn on_complete(&self) {}

        fn on_error(&self, _: euhedral_core::frames::Failure) {}
    }

    /// The stage sink of the harness: a queue its tests pull from.
    struct QueueStages(Arc<QueueIngestSink>);

    impl FrameSink for QueueStages {
        fn offer(&self, frame: Box<dyn Frame>) -> Result<(), Box<dyn Frame>> {
            self.0.offer(frame)
        }
    }

    struct Harness {
        runtime: GraphRuntime,
        /// Where the runtime's stage frames queue.
        sink: Arc<QueueIngestSink>,
        hands: Vec<Arc<HandLane>>,
        log: Log,
        /// The id hashes and orderedness of the stage frames run, in order.
        headers: Mutex<Vec<(u64, bool)>>,
    }

    impl Harness {
        fn new(lanes: usize) -> Self {
            Self::with_backstop(lanes, DEFAULT_RETRY_BACKSTOP)
        }

        fn with_backstop(lanes: usize, retry_backstop: Duration) -> Self {
            let hands: Vec<Arc<HandLane>> =
                (0..lanes).map(|_| Arc::new(HandLane::default())).collect();
            let pool = Arc::new(LanePool::new(
                hands
                    .iter()
                    .map(|h| h.clone() as Arc<dyn DeviceQueue>)
                    .collect(),
            ));
            let stages = Arc::new(QueueIngestSink::new());
            // A queue ingest sink hands out frames only once something is attached; this one asks
            // for nothing.
            stages.delegate().add_downstream(Arc::new(Unattended));
            Self {
                runtime: GraphRuntime {
                    stages: Arc::new(QueueStages(stages.clone())),
                    completions: Completions::new(pool.clone()),
                    lanes: pool,
                    retries: Arc::default(),
                    retry_backstop,
                },
                sink: stages,
                hands,
                log: Arc::default(),
                headers: Mutex::default(),
            }
        }

        /// A graph of `stages` (name, ordered, lanes, act) and `edges`.
        fn graph(
            &self,
            stages: &[(Option<u64>, &[u16], Act)],
            edges: &[(u16, u16, EdgeKind)],
        ) -> Arc<StageGraph<u64>> {
            let mut spec = StageGraphSpec::new();
            let mut impls: Vec<Arc<dyn Stage<u64>>> = Vec::new();
            for (i, (ordered, lanes, act)) in stages.iter().enumerate() {
                spec.stage(StageSpec {
                    name: "probe",
                    ordered: *ordered,
                    lanes: lanes.iter().map(|&l| LaneId(l)).collect(),
                });
                let probe = Probe {
                    act: act.clone(),
                    log: self.log.clone(),
                    hands: self.hands.clone(),
                };
                impls.push(Arc::new(Named(i as u16, probe)));
            }
            for &(from, to, kind) in edges {
                spec.edge(StageId(from), StageId(to), kind);
            }
            let log = self.log.clone();
            StageGraph::new(spec, impls, self.runtime.clone(), move |q, outcome| {
                lock(&log).push(Event::Retired { q, outcome });
            })
            .unwrap()
        }

        /// Runs one frame, as an executor does, from the stage sink or else the completions.
        fn pump_once(&self) -> bool {
            let mut frame = None;
            self.sink
                .delegate()
                .pull(&mut |f| frame = Some(f), &mut |_| false, 1);
            if frame.is_none() {
                self.runtime
                    .completions
                    .pull(&mut |f| frame = Some(f), &mut |_| false, 1);
            }
            let Some(frame) = frame else { return false };
            let header = frame.header();
            lock(&self.headers).push((header.id_hash(), header.is_ordered()));
            run_frame(frame);
            true
        }

        fn pump(&self) -> usize {
            let mut ran = 0;
            while self.pump_once() {
                ran += 1;
            }
            ran
        }

        fn finish_lanes(&self) {
            self.hands.iter().for_each(|h| h.finish());
        }

        fn events(&self) -> Vec<Event> {
            lock(&self.log).clone()
        }

        fn ran(&self, q: u64) -> Vec<u16> {
            self.events()
                .into_iter()
                .filter_map(|e| match e {
                    Event::Ran { q: eq, stage, .. } if eq == q => Some(stage),
                    _ => None,
                })
                .collect()
        }

        fn ran_with(&self, q: u64, stage: u16) -> Option<RanWith> {
            self.events().into_iter().find_map(|e| match e {
                Event::Ran {
                    q: eq,
                    stage: s,
                    lane,
                    waits,
                } if eq == q && s == stage => Some((lane, waits)),
                _ => None,
            })
        }

        fn abandoned(&self, q: u64) -> Vec<u16> {
            self.events()
                .into_iter()
                .filter_map(|e| match e {
                    Event::Abandoned { q: eq, stage } if eq == q => Some(stage),
                    _ => None,
                })
                .collect()
        }

        fn retired(&self, q: u64) -> Vec<QuantumOutcome> {
            self.events()
                .into_iter()
                .filter_map(|e| match e {
                    Event::Retired { q: eq, outcome } if eq == q => Some(outcome),
                    _ => None,
                })
                .collect()
        }
    }

    fn run_frame(mut frame: Box<dyn Frame>) {
        if !frame.is_alive() {
            frame.do_finally();
            return;
        }
        match catch_unwind(AssertUnwindSafe(|| frame.execute())) {
            Ok(Ok(())) | Ok(Err(FrameError::Cancelled)) => frame.do_finally(),
            Ok(Err(error)) => frame.do_finally_with_error(error),
            Err(_) => frame.do_finally_with_error(FrameError::failed("the frame panicked")),
        }
    }

    fn alive() -> Arc<AtomicBool> {
        Arc::new(AtomicBool::new(false))
    }

    use EdgeKind::{Completion, Submission};

    #[test]
    fn the_topology_rejects_cycles_unknown_stages_and_mismatched_stage_counts() {
        let mut spec = StageGraphSpec::new();
        let ids: Vec<StageId> = (0..3)
            .map(|_| {
                spec.stage(StageSpec {
                    name: "s",
                    ordered: None,
                    lanes: vec![],
                })
            })
            .collect();
        spec.edge(ids[0], ids[1], Submission)
            .edge(ids[1], ids[2], Submission);
        assert_eq!(spec.validate(), Ok(()));
        let mut cyclic = spec.clone();
        cyclic.edge(ids[2], ids[1], Completion);
        assert_eq!(
            cyclic.validate(),
            Err(GraphError::Cycle(vec![ids[1], ids[2]]))
        );
        let mut looped = spec.clone();
        looped.edge(ids[0], ids[0], Submission);
        assert_eq!(looped.validate(), Err(GraphError::Cycle(vec![ids[0]])));
        let mut unknown = spec.clone();
        unknown.edge(ids[0], StageId(9), Submission);
        assert_eq!(
            unknown.validate(),
            Err(GraphError::UnknownStage(StageId(9)))
        );
        let harness = Harness::new(1);
        let result = StageGraph::<u64>::new(spec, Vec::new(), harness.runtime.clone(), |_, _| {});
        assert_eq!(
            result.err(),
            Some(GraphError::StageCount { spec: 3, given: 0 })
        );
    }

    /// StageGraphTest.linearGraphExposesOnlyTheRootAndEachSubmissionPublishesOnlyItsSuccessor
    #[test]
    fn a_linear_graph_publishes_only_its_root_and_each_submission_publishes_only_its_successor() {
        let h = Harness::new(1);
        let graph = h.graph(
            &[
                (None, &[0], Act::Submit),
                (None, &[0], Act::Submit),
                (None, &[0], Act::Submit),
            ],
            &[(0, 1, Submission), (1, 2, Submission)],
        );
        let _ = graph.start(7, alive(), Vec::new());
        assert_eq!(h.sink.size(), 1, "only the root is published");
        for expected in [vec![0], vec![0, 1], vec![0, 1, 2]] {
            assert!(h.pump_once());
            assert_eq!(h.ran(7), expected);
        }
        assert_eq!(
            h.ran_with(7, 1),
            Some((Some(0), vec![])),
            "one lane: no waits"
        );
        assert_eq!(graph.live(), 1);
        assert!(h.retired(7).is_empty(), "the boundary is not reached");
    }

    /// StageGraphTest.fanOutPublishesBothBranchesAndFanInPublishesTheJoinExactlyOnce and
    /// placementKeepsTheLongestChainOnTheHomeLaneWhicheverBranchRunsFirst
    #[test]
    fn fan_out_runs_both_branches_and_fan_in_runs_the_join_once_behind_the_other_lane() {
        let h = Harness::new(2);
        let both: &[u16] = &[0, 1];
        // 0 -> 1 -> 2 -> 4 (the long chain) and 0 -> 3 -> 4.
        let graph = h.graph(
            &[
                (None, both, Act::Submit),
                (None, both, Act::Submit),
                (None, both, Act::Submit),
                (None, both, Act::Submit),
                (None, both, Act::Submit),
            ],
            &[
                (0, 1, Submission),
                (1, 2, Submission),
                (2, 4, Submission),
                (0, 3, Submission),
                (3, 4, Submission),
            ],
        );
        let _ = graph.start(1, alive(), Vec::new());
        h.pump();
        let mut ran = h.ran(1);
        ran.sort_unstable();
        assert_eq!(ran, [0, 1, 2, 3, 4], "each stage ran once");
        let lane = |s| h.ran_with(1, s).unwrap().0.unwrap();
        assert_eq!(lane(1), lane(0), "the long chain keeps its root's lane");
        assert_eq!(lane(2), lane(0));
        assert_eq!(lane(4), lane(2), "the join continues the long chain");
        let (_, waits) = h.ran_with(1, 4).unwrap();
        if lane(3) != lane(4) {
            assert_eq!(
                waits.len(),
                1,
                "the join waits for the other branch's lane only"
            );
            assert_eq!(waits[0].0, lane(3));
        } else {
            assert!(waits.is_empty());
        }
    }

    /// StageGraphTest.crossLaneDependenciesAreOrderedByMarkersAndUsedLanesJoinTheHomeLaneBeforeRetirement
    #[test]
    fn cross_lane_dependencies_wait_on_markers_and_retirement_waits_for_every_lane_used() {
        let h = Harness::new(2);
        let graph = h.graph(
            &[
                (Some(0xdec0), &[0], Act::Submit),
                (None, &[1], Act::Submit),
                (None, &[1], Act::Submit),
            ],
            &[(0, 1, Submission), (1, 2, Submission)],
        );
        let _ = graph.start(3, alive(), Vec::new());
        h.pump();
        assert_eq!(
            h.ran_with(3, 1),
            Some((Some(1), vec![(0, 1)])),
            "the compute waits for the parse on the device"
        );
        assert_eq!(
            h.ran_with(3, 2),
            Some((Some(1), vec![])),
            "same lane: its order suffices"
        );
        assert!(h.retired(3).is_empty());
        h.hands[1].finish();
        h.pump();
        assert!(h.retired(3).is_empty(), "lane 0 has not reached the parse");
        h.hands[0].finish();
        h.pump();
        assert_eq!(h.retired(3), [QuantumOutcome::Completed]);
        assert_eq!(graph.live(), 0);
        assert!(
            h.abandoned(3).is_empty(),
            "a quantum whose stages all ran abandons nothing"
        );
    }

    /// StageGraphTest.submissionEdgeReleasesTheSuccessorBeforeTheProducersDeviceWorkRetires and
    /// deviceCompletionEdgeWaitsForTheProducersRetirement
    #[test]
    fn a_completion_edge_waits_for_the_producers_work_and_a_submission_edge_does_not() {
        let h = Harness::new(1);
        let graph = h.graph(
            &[
                (None, &[0], Act::Submit),
                (None, &[], Act::Cpu),
                (None, &[], Act::Cpu),
            ],
            &[(0, 1, Completion), (0, 2, Submission)],
        );
        let _ = graph.start(1, alive(), Vec::new());
        h.pump();
        assert_eq!(
            h.ran(1),
            [0, 2],
            "the submission successor ran; the completion successor waits"
        );
        h.finish_lanes();
        h.pump();
        assert_eq!(h.ran(1), [0, 2, 1]);
        assert_eq!(
            h.ran_with(1, 1),
            Some((None, vec![])),
            "a completed predecessor needs no device wait"
        );
        assert_eq!(h.retired(1), [QuantumOutcome::Completed]);
    }

    /// StageGraphHostStageTest.aDeferredStageReleasesItsSuccessorsOnlyWhenItsWorkCompletes
    #[test]
    fn a_host_completion_gates_its_completion_edges_and_otherwise_the_boundary() {
        let h = Harness::new(1);
        let (gated, unconsumed) = (Token::default(), Token::default());
        let graph = h.graph(
            &[
                (None, &[], Act::Host(gated.clone())),
                (None, &[], Act::Cpu),
                (None, &[], Act::Host(unconsumed.clone())),
            ],
            &[(0, 1, Completion)],
        );
        let _ = graph.start(1, alive(), Vec::new());
        h.pump();
        assert_eq!(h.ran(1), [0, 2]);
        gated.0.store(true, Ordering::Release);
        h.pump();
        assert_eq!(h.ran(1), [0, 2, 1]);
        assert!(
            h.retired(1).is_empty(),
            "the unconsumed host completion is in the boundary"
        );
        unconsumed.0.store(true, Ordering::Release);
        h.pump();
        assert_eq!(h.retired(1), [QuantumOutcome::Completed]);
    }

    /// StageGraphHostStageTest.aSkippedStageStillJoinsTheDeviceOrderOfItsPredecessors
    #[test]
    fn a_cpu_stage_passes_its_predecessors_device_order_on() {
        let h = Harness::new(2);
        let graph = h.graph(
            &[
                (None, &[0], Act::Submit),
                (None, &[], Act::Cpu),
                (None, &[1], Act::Submit),
            ],
            &[(0, 1, Submission), (1, 2, Submission)],
        );
        let _ = graph.start(1, alive(), Vec::new());
        h.pump();
        assert_eq!(h.ran_with(1, 1), Some((None, vec![(0, 1)])));
        assert_eq!(
            h.ran_with(1, 2),
            Some((Some(1), vec![(0, 1)])),
            "the kernel still waits for the parse"
        );
    }

    /// StageGraphTest.failedSubmissionReleasesNoDescendantAndRetiresSubmittedWorkOnce and
    /// stagesThatNeverSubmittedAreAbandonedInOrderBeforeTheBoundary
    #[test]
    fn a_failure_part_way_releases_no_descendant_and_retires_the_submitted_work_once() {
        let h = Harness::new(1);
        let graph = h.graph(
            &[
                (None, &[0], Act::Submit),
                (None, &[0], Act::Fail),
                (None, &[0], Act::Submit),
                (None, &[0], Act::Submit),
            ],
            &[(0, 1, Submission), (1, 2, Submission), (2, 3, Submission)],
        );
        let _ = graph.start(1, alive(), Vec::new());
        h.pump();
        assert_eq!(h.ran(1), [0, 1]);
        assert_eq!(
            h.abandoned(1),
            [2, 3],
            "the stages that never ran are abandoned, in order"
        );
        assert!(
            h.retired(1).is_empty(),
            "the root's submitted work is still covered by the boundary"
        );
        h.finish_lanes();
        h.pump();
        let failed = QuantumOutcome::Failed {
            stage: StageId(1),
            error: StageError::Corrupt("a bad packet".into()),
        };
        assert_eq!(h.retired(1), [failed]);
        assert_eq!(graph.live(), 0);
    }

    /// StageGraphTest.stageErrorIsRecordedAndNeverEscapesIntoEuhedral
    #[test]
    fn a_panicking_stage_is_a_fatal_failure_and_the_quantum_still_retires() {
        let h = Harness::new(1);
        let graph = h.graph(
            &[(None, &[], Act::Panic), (None, &[], Act::Cpu)],
            &[(0, 1, Submission)],
        );
        let _ = graph.start(1, alive(), Vec::new());
        h.pump();
        let retired = h.retired(1);
        assert!(
            matches!(
                &retired[..],
                [QuantumOutcome::Failed {
                    stage: StageId(0),
                    error: StageError::Fatal(_)
                }]
            ),
            "{retired:?}"
        );
        assert_eq!(
            h.abandoned(1),
            [1],
            "a stage that panicked ran; its successor did not"
        );
    }

    /// StageGraphTest.failedBranchKeepsTheJoinUnreadyEvenWhenTheOtherBranchSucceedsLater
    #[test]
    fn a_failed_branch_keeps_the_join_from_running_when_the_other_branch_succeeds() {
        let h = Harness::new(1);
        let graph = h.graph(
            &[
                (None, &[], Act::Cpu),
                (None, &[], Act::Fail),
                (None, &[], Act::Cpu),
                (None, &[], Act::Cpu),
            ],
            &[
                (0, 1, Submission),
                (0, 2, Submission),
                (1, 3, Submission),
                (2, 3, Submission),
            ],
        );
        let _ = graph.start(1, alive(), Vec::new());
        h.pump();
        assert!(!h.ran(1).contains(&3), "the join never runs");
        // The failure stops the quantum: the other branch, published but not yet run, is abandoned
        // too.
        assert_eq!(h.abandoned(1), [2, 3]);
        assert_eq!(h.retired(1).len(), 1);
    }

    /// Spins for `span` (test only: the library never waits).
    fn spin_for(span: Duration) {
        let until = Instant::now() + span;
        while Instant::now() < until {}
    }

    /// Spins past the default retry backstop.
    fn past_the_backstop() {
        spin_for(DEFAULT_RETRY_BACKSTOP + Duration::from_millis(1));
    }

    #[test]
    fn a_retry_runs_the_stage_again_on_a_lanes_progress_and_never_fails_the_quantum() {
        let h = Harness::new(1);
        let graph = h.graph(
            &[
                (None, &[0], Act::Submit),
                (None, &[0], Act::Retry(Arc::new(AtomicU32::new(2)))),
                (None, &[0], Act::Submit),
            ],
            &[(0, 1, Submission), (1, 2, Submission)],
        );
        let _ = graph.start(1, alive(), Vec::new());
        h.pump();
        assert_eq!(
            h.ran(1),
            [0, 1],
            "the retried stage waits; its successor is not released"
        );
        assert!(h.abandoned(1).is_empty());
        // The root's work completes: the retry wakes, answers Retry once more, and waits for the
        // next progress.
        h.finish_lanes();
        h.pump();
        assert_eq!(h.ran(1), [0, 1, 1]);
        assert_eq!(h.runtime.retries.retries(), 2);
        assert_eq!(h.runtime.retries.progress(), 1);
        // No lane progresses now: the backstop runs it again (the busy resource could be outside
        // the pool).
        past_the_backstop();
        h.pump();
        assert_eq!(h.ran(1), [0, 1, 1, 1, 2]);
        assert_eq!(h.runtime.retries.backstops(), 1);
        h.finish_lanes();
        h.pump();
        assert_eq!(
            h.retired(1),
            [QuantumOutcome::Completed],
            "a retry is never a failure"
        );
        assert!(h.abandoned(1).is_empty());
        assert_eq!(graph.live(), 0);
    }

    #[test]
    fn the_retry_backstop_setting_decides_when_a_retried_stage_runs_again() {
        // One retry, with the root's work never finishing, so only the backstop can wake the stage.
        let ran_after = |backstop: Duration, wait: Duration| {
            let h = Harness::with_backstop(1, backstop);
            let graph = h.graph(
                &[
                    (None, &[0], Act::Submit),
                    (None, &[0], Act::Retry(Arc::new(AtomicU32::new(1)))),
                ],
                &[(0, 1, Submission)],
            );
            let _ = graph.start(1, alive(), Vec::new());
            h.pump();
            assert_eq!(h.ran(1), [0, 1], "the retried stage waits");
            spin_for(wait);
            h.pump();
            (h.ran(1), h.runtime.retries.backstops())
        };
        // A short backstop fires well before the default of 20 ms.
        let (ran, backstops) = ran_after(Duration::from_millis(2), Duration::from_millis(4));
        assert_eq!(
            (ran, backstops),
            (vec![0, 1, 1], 1),
            "the configured backstop ran the stage again"
        );
        // A long one does not fire at the default's deadline: the setting is read, not the
        // constant.
        let (ran, backstops) = ran_after(
            Duration::from_millis(200),
            DEFAULT_RETRY_BACKSTOP + Duration::from_millis(5),
        );
        assert_eq!(
            (ran, backstops),
            (vec![0, 1], 0),
            "20 ms is not the backstop any more"
        );
    }

    #[test]
    fn a_kill_while_a_retry_waits_abandons_the_stage_and_retires_killed() {
        let h = Harness::new(1);
        let graph = h.graph(
            &[
                (None, &[0], Act::Submit),
                (None, &[0], Act::Retry(Arc::new(AtomicU32::new(5)))),
                (None, &[0], Act::Submit),
            ],
            &[(0, 1, Submission), (1, 2, Submission)],
        );
        let kill = alive();
        let _ = graph.start(1, kill.clone(), Vec::new());
        h.pump();
        assert_eq!(h.ran(1), [0, 1]);
        kill.store(true, Ordering::Release);
        h.finish_lanes();
        h.pump();
        assert_eq!(h.ran(1), [0, 1], "a killed retry does not run again");
        assert_eq!(
            h.abandoned(1),
            [1, 2],
            "the retried stage never ran to its end: abandoned with the rest"
        );
        assert_eq!(h.retired(1), [QuantumOutcome::Killed]);
        assert_eq!(graph.live(), 0);
    }

    /// StageGraphTest.cancellationStopsUnsentDescendantsButRetiresSubmittedWorkBeforeRecycling
    #[test]
    fn a_kill_stops_the_unrun_stages_but_retires_the_submitted_work() {
        let h = Harness::new(1);
        let graph = h.graph(
            &[
                (None, &[0], Act::Submit),
                (None, &[0], Act::Submit),
                (None, &[0], Act::Submit),
            ],
            &[(0, 1, Submission), (1, 2, Submission)],
        );
        let kill = alive();
        let _ = graph.start(1, kill.clone(), Vec::new());
        assert!(h.pump_once());
        kill.store(true, Ordering::Release);
        h.pump();
        assert_eq!(h.ran(1), [0]);
        assert_eq!(h.abandoned(1), [1, 2]);
        assert!(
            h.retired(1).is_empty(),
            "the root's device work runs to its end first"
        );
        h.finish_lanes();
        h.pump();
        assert_eq!(h.retired(1), [QuantumOutcome::Killed]);
        assert_eq!(graph.live(), 0);
    }

    #[test]
    fn a_quantum_killed_before_it_ran_abandons_every_stage_and_retires_killed() {
        let h = Harness::new(1);
        let graph = h.graph(
            &[(None, &[0], Act::Submit), (None, &[0], Act::Submit)],
            &[(0, 1, Submission)],
        );
        let _ = graph.start(1, Arc::new(AtomicBool::new(true)), Vec::new());
        h.pump();
        assert!(h.ran(1).is_empty());
        assert_eq!(h.abandoned(1), [0, 1]);
        assert_eq!(h.retired(1), [QuantumOutcome::Killed]);
    }

    /// StageGraphTest.rejectedStageFailsTheQuantumWithoutTouchingEuhedralsSharedRejection and
    /// rejectedRetirementFrameStillRetiresTheQuantumOnce
    #[test]
    fn a_refused_stage_frame_fails_the_quantum_and_it_still_retires_once() {
        let h = Harness::new(1);
        let graph = h.graph(&[(None, &[], Act::Cpu)], &[]);
        h.sink.complete();
        let _ = graph.start(1, alive(), Vec::new());
        let retired = h.retired(1);
        assert!(
            matches!(
                &retired[..],
                [QuantumOutcome::Failed {
                    stage: StageId(0),
                    error: StageError::Fatal(_)
                }]
            ),
            "{retired:?}"
        );
        assert_eq!(graph.live(), 0);
    }

    #[test]
    fn ordered_stages_run_on_their_lane_and_the_others_are_unordered() {
        let h = Harness::new(1);
        let graph = h.graph(
            &[(Some(0xec0d), &[], Act::Cpu), (None, &[], Act::Cpu)],
            &[(0, 1, Submission)],
        );
        let _ = graph.start(1, alive(), Vec::new());
        h.pump();
        let headers = lock(&h.headers).clone();
        assert_eq!(headers[0], (0xec0d, true));
        assert!(!headers[1].1, "an unordered stage's frame is stealable");
    }

    /// The order of an ordered stage carries over a completion edge: the arrival runs on the
    /// predecessor's lane.
    #[test]
    fn a_completion_edges_arrival_runs_on_its_predecessors_ordered_lane() {
        let h = Harness::new(1);
        let graph = h.graph(
            &[
                (Some(0x50e1), &[0], Act::Submit),
                (Some(0xdec0), &[], Act::Cpu),
            ],
            &[(0, 1, Completion)],
        );
        let _ = graph.start(1, alive(), Vec::new());
        h.finish_lanes();
        h.pump();
        h.finish_lanes();
        h.pump();
        assert_eq!(
            lock(&h.headers)[..3],
            [(0x50e1, true), (0x50e1, true), (0xdec0, true)]
        );
    }

    /// StageGraphTest.concurrentIncomingEdgesPublishTheJoinExactlyOnce
    #[test]
    #[allow(clippy::disallowed_methods)] // The test's own threads run frames, as lattice workers do.
    fn concurrent_incoming_edges_publish_the_join_exactly_once() {
        let h = Arc::new(Harness::new(4));
        let branches = 32u16;
        let mut stages: Vec<(Option<u64>, &[u16], Act)> = vec![(None, &[], Act::Cpu)];
        let lanes: &[u16] = &[0, 1, 2, 3];
        stages.extend((0..branches).map(|_| (None, lanes, Act::Submit)));
        stages.push((None, lanes, Act::Submit));
        let mut edges = Vec::new();
        for b in 1..=branches {
            edges.push((0, b, Submission));
            edges.push((b, branches + 1, Submission));
        }
        let graph = h.graph(&stages, &edges);
        let pull = Arc::new(Mutex::new(()));
        for round in 0..20 {
            let _ = graph.start(round, alive(), Vec::new());
            let threads: Vec<_> = (0..4)
                .map(|_| {
                    let (h, pull) = (h.clone(), pull.clone());
                    std::thread::spawn(move || {
                        loop {
                            let mut frame = None;
                            {
                                let _one = lock(&pull);
                                h.sink
                                    .delegate()
                                    .pull(&mut |f| frame = Some(f), &mut |_| false, 1);
                            }
                            let Some(frame) = frame else { return };
                            run_frame(frame);
                        }
                    })
                })
                .collect();
            for thread in threads {
                thread.join().unwrap();
            }
            h.pump();
            assert_eq!(
                h.ran(round).iter().filter(|&&s| s == branches + 1).count(),
                1
            );
            assert_eq!(h.ran(round).len(), usize::from(branches) + 2);
            let (_, waits) = h.ran_with(round, branches + 1).unwrap();
            assert!(
                waits.len() <= 3,
                "the latest marker per other lane: {waits:?}"
            );
        }
        h.finish_lanes();
        h.pump();
        assert_eq!(graph.live(), 0);
    }

    /// A graph of two stages on two lanes: stage 0 on lane 1, stage 1 on lane 0.
    fn crossed(h: &Harness, kind_out: Act) -> Arc<StageGraph<u64>> {
        h.graph(
            &[(None, &[1], Act::Submit), (None, &[0], kind_out)],
            &[(0, 1, Submission)],
        )
    }

    /// CrossGraphEdgesTest.aSecondGraphsFirstAccessorWaitsForTheFirstGraphsLastAccessor and
    /// aGraphCannotRetireBeforeItsExternalArrivals
    #[test]
    fn a_later_quantums_stage_waits_for_the_earlier_quantums_stage_on_the_device() {
        let h = Harness::new(2);
        let graph = crossed(&h, Act::Submit);
        let a = graph.start(1, alive(), Vec::new());
        let edge = ExternalEdge {
            from: a.clone(),
            from_stage: StageId(1),
            to_stage: StageId(0),
            kind: Submission,
        };
        let _ = graph.start(2, alive(), vec![edge]);
        assert_eq!(
            h.sink.size(),
            1,
            "only the earlier quantum's root is published"
        );
        drop(a);
        h.pump();
        assert_eq!(h.ran(2), [0, 1]);
        let (_, waits) = h.ran_with(2, 0).unwrap();
        assert_eq!(
            waits,
            [(0, 1)],
            "the later root waits for the earlier quantum's stage 1 (lane 0, value 1)"
        );
        h.finish_lanes();
        h.pump();
        assert_eq!(h.retired(1), [QuantumOutcome::Completed]);
        assert_eq!(h.retired(2), [QuantumOutcome::Completed]);
        assert_eq!(graph.live(), 0);
    }

    /// The pin: a held QuantumRef keeps its quantum from retiring, and an external edge keeps it
    /// until the later stage submitted, whose marker then joins the earlier quantum's boundary.
    #[test]
    fn an_earlier_quantum_retires_only_after_its_refs_are_dropped_and_the_later_stage_used_it() {
        let h = Harness::new(2);
        let graph = crossed(&h, Act::Submit);
        let a = graph.start(1, alive(), Vec::new());
        h.pump();
        h.finish_lanes();
        h.pump();
        assert!(h.retired(1).is_empty(), "its ref is held");
        // A later quantum whose root uses stage 1 of the first.
        let edge = ExternalEdge {
            from: a.clone(),
            from_stage: StageId(1),
            to_stage: StageId(0),
            kind: Submission,
        };
        drop(a);
        let b = graph.start(2, alive(), vec![edge]);
        drop(b);
        h.pump();
        assert_eq!(h.ran(2), [0, 1]);
        assert!(
            h.retired(1).is_empty(),
            "the later stage's work (lane 1) is in the earlier boundary"
        );
        h.hands[1].finish();
        h.pump();
        assert_eq!(
            h.retired(1),
            [QuantumOutcome::Completed],
            "the later stage's work reached: the earlier retires"
        );
        assert!(
            h.retired(2).is_empty(),
            "the later quantum's stage 1 (lane 0) is still running"
        );
        h.hands[0].finish();
        h.pump();
        assert_eq!(h.retired(2), [QuantumOutcome::Completed]);
        assert_eq!(graph.live(), 0);
    }

    #[test]
    fn the_boundary_covers_the_later_stage_that_used_the_quantum() {
        let h = Harness::new(2);
        // The earlier quantum submits on lane 0 only; the later one's root on lane 1.
        let earlier = h.graph(&[(None, &[0], Act::Submit)], &[]);
        let later = h.graph(&[(None, &[1], Act::Submit)], &[]);
        let a = earlier.start(1, alive(), Vec::new());
        let edge = ExternalEdge {
            from: a,
            from_stage: StageId(0),
            to_stage: StageId(0),
            kind: Submission,
        };
        let _ = later.start(2, alive(), vec![edge]);
        h.pump();
        h.hands[0].finish();
        h.pump();
        assert!(
            h.retired(1).is_empty(),
            "the later stage on lane 1 still reads the earlier results"
        );
        h.hands[1].finish();
        h.pump();
        assert_eq!(h.retired(1), [QuantumOutcome::Completed]);
    }

    /// CrossGraphEdgesTest.aCancelledPredecessorThatNeverRanItsStagesReleasesThemAtQuiesce and
    /// aFailedPredecessorStillReleasesItsBuffers
    #[test]
    fn an_edge_on_a_stage_that_never_runs_still_arrives_with_what_its_quantum_submitted() {
        let h = Harness::new(2);
        // The earlier quantum submits on lane 0, then fails; the later one's root runs on lane 1.
        let failing = h.graph(
            &[(None, &[0], Act::Submit), (None, &[0], Act::Fail)],
            &[(0, 1, Submission)],
        );
        let later = h.graph(&[(None, &[1], Act::Submit)], &[]);
        let a = failing.start(1, alive(), Vec::new());
        let edge = ExternalEdge {
            from: a,
            from_stage: StageId(1),
            to_stage: StageId(0),
            kind: Submission,
        };
        let _ = later.start(2, alive(), vec![edge]);
        h.pump();
        assert_eq!(h.ran(2), [0], "the later quantum still runs");
        assert_eq!(
            h.ran_with(2, 0).unwrap().1,
            [(0, 1)],
            "behind the earlier quantum's submitted work"
        );
        h.finish_lanes();
        h.pump();
        assert_eq!(h.retired(1).len(), 1);
        assert_eq!(h.retired(2), [QuantumOutcome::Completed]);
        let graph = crossed(&h, Act::Submit);

        // Killed before it ran anything: the edge arrives with nothing to wait for.
        let killed = graph.start(3, Arc::new(AtomicBool::new(true)), Vec::new());
        let edge = ExternalEdge {
            from: killed,
            from_stage: StageId(1),
            to_stage: StageId(0),
            kind: Submission,
        };
        let _ = graph.start(4, alive(), vec![edge]);
        h.pump();
        assert_eq!(h.ran_with(4, 0).unwrap().1, []);
        h.finish_lanes();
        h.pump();
        assert_eq!(h.retired(3), [QuantumOutcome::Killed]);
        assert_eq!(h.retired(4), [QuantumOutcome::Completed]);
        assert_eq!(graph.live(), 0);
    }

    #[test]
    fn an_external_completion_edge_waits_until_the_earlier_stages_work_completed() {
        let h = Harness::new(2);
        let graph = crossed(&h, Act::Submit);
        let a = graph.start(1, alive(), Vec::new());
        let edge = ExternalEdge {
            from: a,
            from_stage: StageId(1),
            to_stage: StageId(0),
            kind: Completion,
        };
        let _ = graph.start(2, alive(), vec![edge]);
        h.pump();
        assert!(
            h.ran(2).is_empty(),
            "lane 0 has not completed the earlier stage"
        );
        h.hands[0].finish();
        h.pump();
        assert_eq!(h.ran(2), [0, 1]);
        assert_eq!(
            h.ran_with(2, 0).unwrap().1,
            [],
            "completed work needs no device wait"
        );
        h.finish_lanes();
        h.pump();
        assert_eq!(graph.live(), 0);
    }

    #[test]
    fn live_counts_the_quanta_started_and_not_retired() {
        let h = Harness::new(1);
        let graph = h.graph(&[(None, &[0], Act::Submit)], &[]);
        for q in 0..5 {
            let _ = graph.start(q, alive(), Vec::new());
        }
        assert_eq!(graph.live(), 5);
        h.pump();
        assert_eq!(graph.live(), 5, "every boundary is still ahead");
        h.finish_lanes();
        h.pump();
        assert_eq!(graph.live(), 0);
        assert!((0..5).all(|q| h.retired(q) == [QuantumOutcome::Completed]));
        assert_eq!(h.runtime.completions.pending(), 0);
    }
}
