//! Static stage DAGs run per quantum, with submission, completion and external edges
//! (the upstream Java `StageGraph`).
//!
//! A started quantum is a pooled [`Run`]: the state of one quantum, recycled when the quantum retires, as the graph's
//! frames are recycled through their pool. After warm-up a quantum allocates nothing and takes no lock of its own (the
//! pools are queues whose consumer claim is a try-claim, retried and never waited on):
//!
//! - **The quantum, the cause, the markers** are plain fields of the run, handed from one stage to the next by the
//!   counts that order the run's phases (the Java fields written before a volatile count and read after it).
//! - **The first cause** of a quantum not completing is set by one compare-and-swap (Java `failure.compareAndSet`).
//! - **A stage's output markers** are written by the stage and read by its successors after their final arrival
//!   (Java's plain `marker` and `lane`); a stage merges its inputs from its predecessors when it runs.
//! - **External edges** wait on a closable stack per stage (Java `StageFrame.externals`), sealed `Submitted` when the
//!   stage submitted and `Swept` when its quantum quiesced without it.
//! - **The retirement boundary** is the latest marker per used lane (Java `usedLanes`). With lanes that can be controlled
//!   it is Java's `quiesce`: the other lanes join the home lane on the device and one retirement armed there ends the
//!   quantum (`RetireCell`, `Completions::after_retired`); over a query-only lane it is one completion per lane. A
//!   host completion that no edge consumes holds a live count until it is done (Java `deferral`).
//! - **The conclude rule.** A quantum *touches a lane* when one of its stages submitted a marker, a marker was recorded
//!   for it (an external edge brought one, or a later quantum's stage that used it submitted one), or the
//!   [`Stage::abandoned`] hook of a stage that names a lane ran. A quantum that touches a lane concludes in the frame
//!   of its retirement (or of its last lane's completion). A quantum that touches none has no device work to retire:
//!   it concludes in the call that drops its last live count (the frame that ends its last stage, its last host
//!   completion, or the frame that abandons its stages), and when a reference or a later quantum's stage still holds
//!   it, in the call that drops the last hold. Its frames are its stages' and its completions' frames.

#![allow(
    unsafe_code,
    reason = "a run's plain fields are handed from stage to stage by its counts, and external edges are the later quantum's own pooled nodes on an intrusive stack"
)]

use euhedral_data_structures::queues::{Consumer, MpmcQueue, Multi, Producer};
use euhedral_data_structures::sync::HotScope;
use std::collections::VecDeque;
use std::fmt;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::ptr::NonNull;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Weak};
use std::time::{Duration, Instant};

use euhedral_core::frames::{Frame, FrameError, FrameHeader};
use euhedral_core::generics::FrameSink;

use crate::completions::{Completions, Ready, Wake};
use crate::lanes::{DeviceQueue, LaneId, LanePool, Marker, RetireCell};
use crate::sync::{AtomicU64, AtomicUsize, Ordering, Owned};
use crate::take_all::{Link, Linked, Sealed, TakeAllStack};

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
    /// (a stateful device session, a socket). `None`: unordered.
    pub ordered: Option<u64>,
    /// The device lanes able to run the stage's work; empty for a CPU-only stage (Java `host()`): it submits nothing
    /// to a lane and records no marker, and a quantum of such stages alone has no device work to retire (the conclude
    /// rule in the module documentation).
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

/// What a stage's run answers when it does not submit. `Corrupt` and `Fatal` are failures: the quantum then ends with
/// [`QuantumOutcome::Failed`], its later stages get [`Stage::abandoned`], and it still retires. `Retry` and `Held` are
/// no failures: the stage waits and runs again.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StageError {
    /// This quantum's input was bad (a corrupt packet); the next quantum may be fine.
    Corrupt(String),
    /// The stage's resource is unusable until reconfigured.
    Fatal(String),
    /// The stage's resources are busy (a ring entry, a slot still in use on the device): it runs
    /// again once any lane of the pool completed more work, or after
    /// [`GraphRuntime::retry_backstop`] (a deadline the [`Completions`] poll checks, for a resource
    /// outside the pool). It never ends the quantum; a kill or a stop while it waits does. A stage that waits for
    /// something a thread of its own makes ready holds itself at a gate instead ([`Held`](Self::Held)): nothing polls.
    Retry,
    /// The stage found, while running, that what it needs does not exist yet, and holds itself at a [`Gate`] it took with
    /// [`StageCx::hold`]: it runs again, once, when the gate is released. Not a failure: answered without a gate taken,
    /// or a gate taken and another answer given, it fails the quantum with [`StageError::Fatal`], and so does a
    /// [`Gate::fail`] given `Held` (or `Retry`). It is never the error of a failed quantum. See [`Stage::run`].
    Held,
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
    /// The device work of the quantum failed at its retirement: the failure belongs to no one stage (Java's
    /// `retire(ticket)` fails the quantum without a stage). `cause` is the backend's description of the failure.
    DeviceFailed {
        cause: String,
    },
    /// Its kill switch was set (the work it belonged to was cancelled or superseded).
    Killed,
}

/// What a stage sees while it runs.
pub struct StageCx<'a, L: DeviceQueue + ?Sized = dyn DeviceQueue> {
    pub stage: StageId,
    /// The lane the stage submits to (`None` for a CPU-only stage).
    pub lane: Option<LaneId>,
    /// Markers its submission must wait for on the device: submission-edge predecessors on other
    /// lanes, and external edges. Empty when every predecessor ran on `lane`.
    pub waits: &'a [Marker],
    pub lanes: &'a LanePool<L>,
    /// Where [`hold`](Self::hold) takes the gate from.
    gates: &'a dyn GateSource,
    /// Set when the run took its gate: the frame's own flag (read by the thread that ends the frame, after the run).
    held: &'a AtomicBool,
}

impl<L: DeviceQueue + ?Sized> fmt::Debug for StageCx<'_, L> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StageCx")
            .field("stage", &self.stage)
            .field("lane", &self.lane)
            .field("waits", &self.waits)
            .finish_non_exhaustive()
    }
}

impl<L: DeviceQueue + ?Sized> StageCx<'_, L> {
    /// The lane the stage was given, as the pool holds it: for a pool of a backend's shared lane type, the handle the
    /// stage submits its device work to, from whichever worker runs it. `None` for a CPU-only stage.
    ///
    /// The stage orders itself behind [`waits`](Self::waits) on the lane, submits, and returns the marker of its work
    /// in [`Submitted::marker`]; the graph awaits nothing for it.
    pub fn lane_handle(&self) -> Option<&Arc<L>> {
        self.lane.map(|lane| self.lanes.lane(lane))
    }

    /// Holds the stage at a [`Gate`] from inside [`Stage::run`]: for a stage that finds, while it runs, that what it needs
    /// is not ready. The stage hands the gate to whoever makes that state ready, and answers
    /// [`Err(StageError::Held)`](StageError::Held). The contract is on [`Stage::run`].
    ///
    /// # Panics
    ///
    /// If the run already took its gate (a stage takes one gate per run). The quantum then fails with
    /// [`StageError::Fatal`], and the gate taken first stays valid.
    pub fn hold(&self) -> Gate {
        assert!(
            !self.held.swap(true, Ordering::Relaxed),
            "a stage takes one gate per run"
        );
        self.gates.hold_gate_in_run(self.stage)
    }
}

/// One stage's work for a quantum `Q`. It computes or submits, and returns; it never waits.
pub trait Stage<Q, L: DeviceQueue + ?Sized = dyn DeviceQueue>: Send + Sync {
    /// Runs the stage for `quantum`: it computes or submits, and returns; it never waits.
    ///
    /// # Holding from `run`
    ///
    /// A stage that finds, while running, that the state it needs is not ready (an input some other work is still
    /// producing) takes its [`Gate`] with [`StageCx::hold`] and answers [`Err(StageError::Held)`](StageError::Held). It
    /// hands the gate to whatever makes the state ready, which releases it ([`Gate::release`]) or fails it
    /// ([`Gate::fail`]). The stage then runs again exactly once per release, with nothing retried and nothing allocated.
    /// The contract, with no tolerance:
    ///
    /// - **Who may release.** The one holder of the [`Gate`], once, from any thread that may publish frames, including
    ///   the thread running the stage before `run` returns. The gate is moved, so it is spent at most once.
    /// - **A run holds the stage, it does not end it.** The answer `Held` submits nothing and satisfies no edge: no
    ///   marker, no completion, the stage's successors keep waiting, and its live count stays with the gate, so the
    ///   quantum cannot retire while a gate is out. The stage's `retired` hook runs at the end of the quantum
    ///   (`committed` false) as for any stage that ran without submitting, and its `abandoned` hook runs if it never
    ///   submits.
    /// - **Exactly once after a release.** Each release publishes the stage once and `run` is called once more, on a new
    ///   frame, with a new [`StageCx`] (its `waits` and lane are chosen as for any run). [`Stage::pass_gate`] is not
    ///   asked again. The second run may hold again.
    /// - **Release before `run` returns.** A gate released while the run that took it is still going does not publish
    ///   the stage beside it: the stage is published when `run` returns `Held`, and the run after that starts after
    ///   this one ended. Everything the releaser wrote before releasing is visible to that run.
    /// - **Kill and stop.** The gate keeps the quantum live as it keeps any stage, so a killed or failed quantum retires
    ///   only once the gate is spent. Released after a kill or a failure, the stage is published and ends without
    ///   running (the quantum is killed or failed already), and its count is dropped; it is abandoned exactly once, at the
    ///   quantum's boundary, like any stage that never submitted. A holder that learns of a kill releases the gate, fails
    ///   it or drops it; any of the three is harmless, and with the kill switch set each ends the quantum as
    ///   [`QuantumOutcome::Killed`] (the kill is noted before the failure of a `fail` or a drop, so it is the first
    ///   cause), unless the quantum had failed already: the first cause stands.
    /// - **Fail and drop.** [`Gate::fail`] fails the quantum with the given error, as for a gate taken in
    ///   [`pass_gate`](Self::pass_gate): the stage does not run again and the count is dropped. A gate dropped unspent
    ///   fails the quantum with [`StageError::Fatal`] the same way, and `Held` or `Retry` given to `fail` become
    ///   `Fatal` (they are no failures). Both before `run` returns drop the count when `run` returns.
    /// - **Retry untouched.** `Held` is no [`StageError::Retry`]: nothing is registered with the completions, and
    ///   [`RetryCounts`] do not move.
    /// - **A stage that breaks the contract fails the quantum.** `Held` answered with no gate taken, a gate taken and
    ///   another answer given (`Ok`, `Retry`; an error stands as the cause), or a second [`StageCx::hold`] in one run
    ///   (a panic): the quantum fails with [`StageError::Fatal`] (a panic in `run` after a gate was taken included), the
    ///   stage does not run again, and the gate it took is still held until its holder spends it: the count stays with
    ///   it, and spending it is harmless. An `Ok` answer's work counts all the same: its marker joins the quantum's
    ///   boundary and its host completion keeps the quantum live until it is ready, as for a stage that submitted in a
    ///   quantum that stopped meanwhile.
    fn run(&self, quantum: &Q, cx: &StageCx<'_, L>) -> Result<Submitted, StageError>;

    /// Called once per quantum, when the stage's last incoming edge arrived and before it is published (Java
    /// `StageFrame.passGate`): `true` lets it be published now; `false` holds it, and the stage must then have taken a
    /// [`Gate`] from `gate` ([`GateRef::hold`], which consumes it: a stage takes its gate once) and release it (from any
    /// thread that may publish frames, once) when whatever it waits for exists. A held stage keeps the quantum live. This
    /// is how a stage waits for host-side readiness without polling: the work that makes it ready releases the gate, so
    /// no stream ever waits for the host and nothing retries. A `pass_gate` that panics fails the quantum.
    ///
    /// Taking a gate means holding the stage whatever this returns (a `true` after [`GateRef::hold`] is ignored): the
    /// gate's holder publishes it.
    ///
    /// A wait that only running can find (the stage starts, and the state it needs is not ready) holds the stage from
    /// [`run`](Self::run) instead: [`StageCx::hold`] and [`StageError::Held`].
    fn pass_gate(&self, quantum: &Q, gate: GateRef<'_>) -> bool {
        let _ = (quantum, gate);
        true
    }

    /// The quantum ended without this stage submitting: it never ran (the quantum was killed, or a predecessor failed),
    /// or it ran and failed (Java `StageFrame.abandoned(home)`). Called once, at the quantum's boundary, in stage order,
    /// after the quantum's other lanes joined the `home` lane (when the lanes have control: a query-only lane cannot
    /// join, and the hook then runs without the join, and a hook given a query-only lane has nothing to enqueue on it) and
    /// before the retirement is armed on it. If that retirement cannot be armed on a home lane that has control, the quantum
    /// is noted as `DeviceFailed` (a quantum whose hook ran already has an earlier cause, which stands: the first cause
    /// wins), because what the hooks submitted is not covered, and the quantum still retires. So what the hook submits
    /// to `home` is ordered after the quantum's device work and covered by its retirement: a stage that enqueued part of
    /// its work and then failed releases its buffers here, behind work that may still read them.
    ///
    /// The retirement covers the hooks of stages that name a lane ([`StageSpec::lanes`]). The hook of a stage that names
    /// none is host work: it runs at the same place with the same arguments, and a quantum whose stages all name none
    /// concludes with it, so the work a retirement covers is that of the stages that name a lane.
    fn abandoned(&self, quantum: &Q, home: &L) {
        let _ = (quantum, home);
    }

    /// The quantum's device work ended (its boundary was reached and settled) and this stage ran: the place to release
    /// what the stage kept for its work, such as the host buffers a copy read, which its quantum holds (Java
    /// `StageFrame.retired`). Called once, on a worker in the retirement frame, in stage order, before the quantum
    /// is handed to `retire`. `committed` is true only when the whole quantum succeeded and this stage submitted:
    /// publish externally visible state then, and release temporary resources either way.
    fn retired(&self, quantum: &Q, committed: bool) {
        let _ = (quantum, committed);
    }
}

/// What a [`Gate`] releases into: a quantum's run, whatever its types.
trait GateTarget: Send + Sync {
    /// Publishes `stage`, held by its gate, once: a gate already spent (released or failed) does nothing. A lake
    /// that no longer takes frames ends the stage in place, so the held count is always dropped.
    fn release_gate(self: Arc<Self>, stage: StageId);

    /// Fails the quantum with `error` and drops `stage`'s held count: the stage never runs. Once, as `release_gate`.
    fn fail_gate(self: Arc<Self>, stage: StageId, error: StageError);
}

/// What a stage is given in [`Stage::pass_gate`] to take a [`Gate`] (inside [`Stage::run`] it is [`StageCx::hold`]). It is
/// moved by [`hold`](Self::hold), so a stage cannot take two gates:
///
/// ```compile_fail,E0382
/// use euhedral_graph::{GateRef, Stage, StageCx, StageError, Submitted};
///
/// struct Twice;
/// impl Stage<u64> for Twice {
///     fn run(&self, _: &u64, _: &StageCx<'_>) -> Result<Submitted, StageError> {
///         Ok(Submitted::done())
///     }
///     fn pass_gate(&self, _: &u64, gate: GateRef<'_>) -> bool {
///         let first = gate.hold();
///         let second = gate.hold(); // use of moved value: the gate is taken once
///         drop((first, second));
///         false
///     }
/// }
/// ```
pub struct GateRef<'a> {
    source: &'a dyn GateSource,
    stage: StageId,
    /// Set when the stage takes a gate: the arrival's own flag, on its stack.
    taken: &'a std::cell::Cell<bool>,
}

/// What a [`GateRef`] and a [`StageCx`] take a gate from.
trait GateSource: Sync {
    /// A gate for `stage`, which is not running: it is being asked whether it may be published.
    fn hold_gate(&self, stage: StageId) -> Gate;
    /// A gate for `stage` from inside its run.
    fn hold_gate_in_run(&self, stage: StageId) -> Gate;
}

impl GateRef<'_> {
    /// Holds the stage: it is published when the returned [`Gate`] is released. Consumes the reference, so it is called
    /// once, inside [`Stage::pass_gate`], which then returns `false`.
    pub fn hold(self) -> Gate {
        self.taken.set(true);
        self.source.hold_gate(self.stage)
    }
}

impl fmt::Debug for GateRef<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GateRef")
            .field("stage", &self.stage)
            .finish_non_exhaustive()
    }
}

/// A stage held at its gate (Java `StageFrame.releaseGate`). Release it once, from any thread that may publish frames (a
/// frame's `do_finally`, a [`Completions`] continuation): the stage is then published and runs. It is moved, so it is
/// released at most once. Dropping it unreleased fails the quantum (a stage nobody will publish would keep the quantum
/// live for ever).
///
/// A gate taken in [`Stage::pass_gate`] holds the stage before its first run; one taken in [`Stage::run`]
/// ([`StageCx::hold`]) holds it after a run that found its input not ready. The release publishes the stage in both,
/// exactly once; the contract of the second is on [`Stage::run`].
pub struct Gate {
    held: Option<(Arc<dyn GateTarget>, StageId)>,
}

impl Gate {
    /// Publishes the held stage. If the quantum ended its stages meanwhile (it was killed or failed), the stage is
    /// published all the same and ends without running, dropping its count. A gate taken in `run`, released before that
    /// run returned, publishes the stage when it returns.
    pub fn release(mut self) {
        if let Some((target, stage)) = self.held.take() {
            target.release_gate(stage);
        }
    }

    /// The gate cannot be released because what the stage waits for failed: the quantum fails with `error`, and the held
    /// stage's count is dropped. `Retry` and `Held` are no failures, so a gate failed with either fails the quantum with
    /// [`StageError::Fatal`]. If the quantum's kill switch is set, the quantum ends as [`QuantumOutcome::Killed`] instead,
    /// whichever way the gate is spent (release, `fail` or drop), unless the quantum had failed already: the first cause stands.
    pub fn fail(mut self, error: StageError) {
        if let Some((target, stage)) = self.held.take() {
            let error = match error {
                StageError::Retry | StageError::Held => StageError::Fatal(format!(
                    "a gate was failed with {error:?}, which is no failure"
                )),
                failure => failure,
            };
            target.fail_gate(stage, error);
        }
    }
}

impl Drop for Gate {
    fn drop(&mut self) {
        if let Some((target, stage)) = self.held.take() {
            target.fail_gate(
                stage,
                StageError::Fatal("a stage's gate was dropped without being released".into()),
            );
        }
    }
}

impl fmt::Debug for Gate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Gate")
            .field("stage", &self.held.as_ref().map(|h| h.1))
            .finish()
    }
}

/// The RustEE side a graph publishes through.
pub struct GraphRuntime<L: DeviceQueue + ?Sized = dyn DeviceQueue> {
    /// Where stage frames are offered: any sink of the lattice's frames, usually a
    /// [`Lake`](crate::Lake) (whose units are the graph's quanta, so the caller admits one before
    /// it starts a quantum and drops its guard when the quantum retires).
    pub stages: Arc<dyn FrameSink>,
    pub completions: Arc<Completions>,
    pub lanes: Arc<LanePool<L>>,
    /// Shared by the runtime's graphs.
    pub retries: Arc<RetryCounts>,
    /// How long a retried stage waits at most for a lane's progress before it runs again; see
    /// [`DEFAULT_RETRY_BACKSTOP`].
    pub retry_backstop: Duration,
}

impl<L: DeviceQueue + ?Sized> Clone for GraphRuntime<L> {
    fn clone(&self) -> Self {
        Self {
            stages: self.stages.clone(),
            completions: self.completions.clone(),
            lanes: self.lanes.clone(),
            retries: self.retries.clone(),
            retry_backstop: self.retry_backstop,
        }
    }
}

impl<L: DeviceQueue + ?Sized> fmt::Debug for GraphRuntime<L> {
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
pub struct QuantumRef {
    /// The pinned quantum; taken by the drop, which lets go of one hold.
    run: Option<Arc<dyn Upstream>>,
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
/// every marker that quantum submitted, and those its own external edges brought (so the order of the quanta before it
/// is kept). The earlier quantum stays pinned until `to_stage` has submitted (its marker joins the earlier quantum's
/// retirement boundary) or been abandoned.
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
/// - Each quantum counts its published frames, armed completion edges and host completions. At zero
///   no stage can run again; the quantum's retirement boundary is armed with [`Completions`] (Java `quiesce`: the
///   lanes it used join its home lane on the device and one retirement is armed there; lanes with no control
///   get one completion per lane). The retirement frame runs each attempted stage's [`Stage::retired`] hook and hands
///   the quantum and its [`QuantumOutcome`] to `retire`. A quantum that touched no lane (see the module's conclude
///   rule) has no boundary: that call runs the hooks and hands the quantum over itself, on its thread.
/// - A killed quantum's frames are skipped by RustEE and their `do_finally` drops their counts; its
///   stages get [`Stage::abandoned`]. Device work already submitted runs to its end and is covered
///   by the boundary.
/// - **A quantum is pooled.** `start` takes a retired quantum's state from the graph's pool (a
///   bounded queue) and `retire` gives it back, as frames go back to theirs; the stage, arrival,
///   boundary and retirement frames are pooled the same way. After warm-up, a quantum allocates
///   nothing and takes no lock of its own on its way (a take from a pool retries the queue's try-claim; it
///   never waits). A stage that returns a host completion allocates it, and
///   a completion that waits for both a marker and a host completion allocates its wake-up. A retried stage's
///   wake-up is pooled with its run: a retry allocates nothing.
///
/// More precisely:
/// - **Lanes:** each stage's lane is continued by exactly one successor, its critical successor
///   (the one on the longest submission path through it; the first listed on ties). A stage with
///   device lanes continues the lane of the predecessor it is the critical successor of, when
///   allowed; a root continues its run's home lane (taken in turn over the compute lanes), and every other branch of
///   a fan-out takes a lane by the pool's [`Placement`] ([`LanePool::choose`]). A join continues a predecessor's lane only if it is that
///   predecessor's critical successor, and waits on the markers of its other predecessors.
/// - **Waits:** a stage's [`StageCx::waits`] hold, per lane, the latest marker of its submission
///   predecessors and external edges, minus its own lane's. A stage that submitted no marker
///   (CPU-only) passes its inputs on, so device order is kept through it.
/// - **The boundary** is the latest marker per lane of everything the quantum submitted (plus the
///   markers of later quanta's stages that used it through external edges, and those of external
///   edges, whatever became of their stages). Every lane used is covered. A host
///   completion holds the quantum live until it is ready, so the quantum retires after it.
/// - **The outcome** is the first of: a stage's failure (an `Err`, or a panic as
///   [`StageError::Fatal`]), or the kill switch found set before a stage or edge ran. Otherwise it
///   is [`QuantumOutcome::Completed`].
/// - **Gates:** a stage may hold itself before it is published ([`Stage::pass_gate`]) or after it started, from inside
///   its run ([`StageCx::hold`], answering [`StageError::Held`]). A held stage keeps its live count, so the quantum
///   cannot retire while a gate is out; the release publishes the stage once. No retry is involved, and nothing is
///   allocated.
/// - **Stage frames** carry the quantum's kill switch, are ordered by [`StageSpec::ordered`] or
///   else unordered, and run the stage in `execute` (RustEE turns a panic there into
///   `do_finally_with_error`).
/// - **Order across quanta:** an ordered stage's frames run in the order they are published.
///   Successors through submission edges are published by the predecessor's frame; through
///   completion edges, by an arrival frame on the predecessor's ordered lane (completions are
///   emitted in registration order), so an ordered predecessor's order carries over to an ordered
///   successor (a fetch → parse chain).
pub struct StageGraph<Q, L: DeviceQueue + ?Sized = dyn DeviceQueue> {
    core: Arc<Core<Q, L>>,
    /// Retired runs, for the next quanta.
    runs: Pool<Arc<Run<Q, L>>>,
}

impl<Q, L: DeviceQueue + ?Sized> fmt::Debug for StageGraph<Q, L> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StageGraph")
            .field("stages", &self.core.spec.stages.len())
            .field("live", &self.core.live.load(Ordering::Relaxed))
            .finish_non_exhaustive()
    }
}

/// What every run of a graph shares: the shape, the stages, the runtime, and the frame pool. Immutable but for its
/// counters and the pool.
struct Core<Q, L: DeviceQueue + ?Sized> {
    spec: StageGraphSpec,
    stages: Vec<Arc<dyn Stage<Q, L>>>,
    topology: Topology,
    runtime: GraphRuntime<L>,
    retire: Box<dyn Fn(Q, QuantumOutcome) + Send + Sync>,
    live: AtomicUsize,
    seeds: AtomicU64,
    /// Finished frames, for the next stages, arrivals and boundaries (of any run of the graph).
    frames: Pool<Box<GraphFrame<Q, L>>>,
}

/// The id hash of a graph's unordered frames (their routing hash is randomized per frame).
const UNORDERED: u64 = 0x5747_e6a9_0f10_77a1;
const NO_LANE: u32 = u32::MAX;

/// Adds `marker` to `markers`, keeping the latest per queue and lane (a lane runs in order, so it covers
/// the earlier ones).
fn merge(markers: &mut Vec<Marker>, marker: Marker) {
    match markers
        .iter_mut()
        .find(|m| m.queue == marker.queue && m.lane == marker.lane)
    {
        Some(m) => m.value = m.value.max(marker.value),
        None => markers.push(marker),
    }
}

/// The edges of a validated spec, by stage.
#[derive(Debug)]
struct Topology {
    successors: Vec<Vec<(StageId, EdgeKind)>>,
    /// The stages whose submission edges lead into each stage: where it reads its input markers.
    submission_predecessors: Vec<Vec<usize>>,
    incoming: Vec<usize>,
    has_completion_successor: Vec<bool>,
    /// The predecessor whose lane each stage continues: the one for which the stage is the critical
    /// successor (see [`Topology::path_predecessors`]).
    path_predecessor: Vec<Option<StageId>>,
    roots: Vec<StageId>,
}

impl Topology {
    fn of(spec: &StageGraphSpec) -> Result<Self, GraphError> {
        let n = spec.stages.len();
        let mut successors = vec![Vec::new(); n];
        let mut submission_predecessors: Vec<Vec<usize>> = vec![Vec::new(); n];
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
            if kind == EdgeKind::Submission {
                submission_predecessors[usize::from(to.0)].push(usize::from(from.0));
            }
            incoming[usize::from(to.0)] += 1;
            has_completion_successor[usize::from(from.0)] |= kind == EdgeKind::Completion;
        }
        // Kahn's order, to reject cycles.
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
        let path_predecessor = Self::path_predecessors(&successors);
        Ok(Self {
            successors,
            submission_predecessors,
            incoming,
            has_completion_successor,
            path_predecessor,
            roots,
        })
    }

    /// Links each stage to the successor that continues its longest submission path to a sink, so
    /// lane placement keeps a graph's critical chain on one lane and every other branch of a
    /// fan-out takes a lane in turn. Entry `s` of the result is the stage whose lane `s` continues.
    ///
    /// Port of the upstream `StageGraph.markPaths`, with its tie-breaks:
    /// - only submission edges count (the Java `submittedSuccessors`); a completion edge never
    ///   carries a lane;
    /// - a stage's critical successor is the one with the greatest height (the number of stages on
    ///   its longest path to a sink); of equal heights, the first listed (in edge order) wins;
    /// - stages are visited sinks first, in a first-in first-out Kahn order over the reversed edges
    ///   (sinks by stage index; a stage joins the queue when its last successor was visited, its
    ///   predecessors being taken in the order of their stage indexes);
    /// - a stage that is the critical successor of several predecessors continues the one visited
    ///   last.
    fn path_predecessors(successors: &[Vec<(StageId, EdgeKind)>]) -> Vec<Option<StageId>> {
        let n = successors.len();
        let submitted: Vec<Vec<usize>> = successors
            .iter()
            .map(|edges| {
                edges
                    .iter()
                    .filter(|&&(_, kind)| kind == EdgeKind::Submission)
                    .map(|&(to, _)| usize::from(to.0))
                    .collect()
            })
            .collect();
        let mut predecessors: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (from, tos) in submitted.iter().enumerate() {
            for &to in tos {
                predecessors[to].push(from);
            }
        }
        let mut pending: Vec<usize> = submitted.iter().map(Vec::len).collect();
        let mut ready: VecDeque<usize> = (0..n).filter(|&s| pending[s] == 0).collect();
        let mut height = vec![0usize; n];
        let mut path_predecessor = vec![None; n];
        while let Some(stage) = ready.pop_front() {
            let mut critical: Option<usize> = None;
            for &successor in &submitted[stage] {
                if critical.is_none_or(|c| height[successor] > height[c]) {
                    critical = Some(successor);
                }
            }
            height[stage] = critical.map_or(1, |c| height[c] + 1);
            if let Some(critical) = critical {
                path_predecessor[critical] = Some(StageId(stage as u16));
            }
            for &predecessor in &predecessors[stage] {
                pending[predecessor] -= 1;
                if pending[predecessor] == 0 {
                    ready.push_back(predecessor);
                }
            }
        }
        path_predecessor
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

/// A pool of finished objects for the next ones, from any thread.
///
/// A bounded multi-producer, multi-consumer queue, as a [`FrameManager`](euhedral_core::impls::FrameManager) keeps its
/// returned frames (which cannot serve here: its checkout takes `&mut self`, and a quantum starts and a frame is made
/// on any thread). An object comes back through a weak handle, so objects in the pool never keep the pool alive; a
/// full pool drops what comes back.
///
/// A take polls the queue, whose consumer side is claimed by a try-lock that fails instead of waiting (and that also
/// finds nothing while the head is claimed by a producer and not yet published). The take polls again for as long as
/// the queue holds anything, up to [`TAKE_TRIES`] polls, so a lost race costs a retry, not a new object. A take makes
/// a new object only when the queue is empty, or stays unpollable for `TAKE_TRIES` polls in a row (a producer
/// descheduled between claiming a place and publishing into it).
struct Pool<T> {
    consumer: Consumer<T, Multi>,
    producer: Arc<Producer<T, Multi>>,
}

impl<T> Pool<T> {
    fn new(capacity: usize) -> Self {
        let queue = MpmcQueue::<T>::bounded(capacity);
        Self {
            consumer: queue.consumer(),
            producer: Arc::new(queue.producer()),
        }
    }

    fn take(&self) -> Option<T> {
        for _ in 0..TAKE_TRIES {
            if let Some(item) = self.consumer.poll() {
                return Some(item);
            }
            if self.consumer.is_empty() {
                return None;
            }
        }
        None
    }

    /// The handle objects come back through.
    fn returns(&self) -> Weak<Producer<T, Multi>> {
        Arc::downgrade(&self.producer)
    }
}

/// Returns `item` to the pool behind `to`, or drops it when the pool is gone or full.
fn give_back<T>(to: &Weak<Producer<T, Multi>>, item: T) {
    if let Some(pool) = to.upgrade() {
        let _ = pool.offer(item);
    }
}

/// The polls a take makes while the pool holds something it could not take yet.
const TAKE_TRIES: usize = 1 << 16;
/// Quanta kept for reuse.
const POOLED_RUNS: usize = 256;
/// Frames kept for reuse.
const POOLED_FRAMES: usize = 4096;

impl<Q: Send + Sync + 'static, L: DeviceQueue + ?Sized + 'static> StageGraph<Q, L> {
    /// `stages[i]` implements `spec.stages()[i]`. `retire` runs once per quantum, in a frame, after
    /// its boundary.
    pub fn new(
        spec: StageGraphSpec,
        stages: Vec<Arc<dyn Stage<Q, L>>>,
        runtime: GraphRuntime<L>,
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
            core: Arc::new(Core {
                spec,
                stages,
                topology,
                runtime,
                retire: Box::new(retire),
                live: AtomicUsize::new(0),
                seeds: AtomicU64::new(0),
                frames: Pool::new(POOLED_FRAMES),
            }),
            runs: Pool::new(POOLED_RUNS),
        }))
    }

    /// Starts `quantum`; its frames share `kill`.
    ///
    /// The quantum is a pooled run: after warm-up, a start allocates nothing (the external edges are taken from
    /// `external` one by one, and their records are pooled too).
    pub fn start(
        &self,
        quantum: Q,
        kill: Arc<AtomicBool>,
        external: impl IntoIterator<Item = ExternalEdge>,
    ) -> QuantumRef {
        let core = &self.core;
        let n = core.spec.stages.len();
        let run = self
            .runs
            .take()
            .unwrap_or_else(|| Run::new(core.clone(), self.runs.returns()));
        run.bind(quantum, kill);
        let mut edges = 0;
        for edge in external {
            assert!(
                usize::from(edge.to_stage.0) < n,
                "an external edge to unknown stage {}",
                edge.to_stage.0
            );
            run.add_external(edge);
            edges += 1;
        }
        core.live.fetch_add(1, Ordering::AcqRel);
        let roots = core
            .topology
            .roots
            .iter()
            .filter(|s| run.stages[usize::from(s.0)].pending.load(Ordering::Acquire) == 0)
            .count();
        // The roots, admission's hold (so fast roots cannot retire the quantum before every root
        // and edge is in), and one count per external edge until it arrives.
        run.live.store(roots + 1 + edges, Ordering::Release);
        let upstream: Arc<dyn Upstream> = run.clone();
        let reference = QuantumRef {
            run: Some(upstream),
        };
        for &root in &core.topology.roots {
            if run.stages[usize::from(root.0)]
                .pending
                .load(Ordering::Acquire)
                == 0
            {
                run.gate_and_offer(root);
            }
        }
        for index in 0..edges {
            run.register_external(index);
        }
        run.finish_one();
        reference
    }

    /// Quanta started and not yet retired.
    pub fn live(&self) -> usize {
        self.core.live.load(Ordering::Acquire)
    }
}

impl<Q: Send + Sync + 'static, L: DeviceQueue + ?Sized + 'static> Core<Q, L> {
    fn next_seed(&self) -> u64 {
        self.seeds.fetch_add(1, Ordering::Relaxed)
    }

    /// A frame for `run`: a finished one made over, or a new one.
    fn frame(
        &self,
        run: &Arc<Run<Q, L>>,
        kind: Kind,
        header: FrameHeader,
    ) -> Box<GraphFrame<Q, L>> {
        let mut frame = self
            .frames
            .take()
            .unwrap_or_else(|| Box::new(GraphFrame::new(self.frames.returns())));
        frame.run = Some(run.clone());
        frame.kind = kind;
        frame.header = header;
        frame
    }
}

/// What an external edge needs of an earlier quantum, of any graph (type-erased).
trait Upstream: Send + Sync {
    /// Delivers `node`'s edge with `stage`'s markers once it submitted, or with every marker of the quantum once it is
    /// known never to run; at once if either is already so. Otherwise the node waits on the stage's stack until then.
    fn subscribe(&self, stage: StageId, node: NonNull<ExtSlot>);
    /// A later quantum's stage that used this one submitted `marker`: the retirement boundary
    /// covers it.
    fn cover(&self, marker: Marker);
    /// One more pin.
    fn add_hold(&self);
    /// One pin less; the last lets the quantum retire.
    fn release_hold(self: Arc<Self>);
}

/// What an earlier quantum does to a later one whose edge it satisfies (type-erased).
trait Downstream: Send + Sync {
    /// The edge in slot `index` arrived, its markers already in the slot.
    fn external_arrived(self: Arc<Self>, index: usize);
}

impl Clone for QuantumRef {
    /// Another pin: the quantum retires only after this one is dropped too.
    fn clone(&self) -> Self {
        let run = self.run.as_ref().expect("a live ref holds its quantum");
        run.add_hold();
        Self {
            run: Some(run.clone()),
        }
    }
}

impl Drop for QuantumRef {
    fn drop(&mut self) {
        if let Some(run) = self.run.take() {
            run.release_hold();
        }
    }
}

impl QuantumRef {
    fn upstream(&self) -> &Arc<dyn Upstream> {
        self.run.as_ref().expect("a live ref holds its quantum")
    }
}

/// One external edge of a run: the record on the earlier quantum's stack, the pin it holds, and the markers it
/// brought. Pooled with its run.
struct ExtSlot {
    /// The link on the earlier quantum's stack.
    link: Link<ExtSlot>,
    /// This slot's place in its run.
    index: usize,
    to_stage: StageId,
    from_stage: StageId,
    kind: EdgeKind,
    /// The markers the earlier stage brought; written by the thread that satisfied the edge, read by the stage the
    /// edge leads to once it has arrived.
    markers: Owned<Vec<Marker>>,
    /// The earlier quantum, pinned until the stage used it or the quantum quiesced.
    pin: Owned<Option<QuantumRef>>,
    /// The run the edge leads to, until the edge arrives.
    target: Owned<Option<Arc<dyn Downstream>>>,
    /// Markers a completion edge still waits for.
    left: AtomicUsize,
}

// SAFETY: the link is this node's own and only the stack uses it.
unsafe impl Linked for ExtSlot {
    fn link(&self) -> &Link<ExtSlot> {
        &self.link
    }
}

impl ExtSlot {
    fn new() -> Self {
        Self {
            link: Link::new(),
            index: 0,
            to_stage: StageId(0),
            from_stage: StageId(0),
            kind: EdgeKind::Submission,
            markers: Owned::new(Vec::new()),
            pin: Owned::new(None),
            target: Owned::new(None),
            left: AtomicUsize::new(0),
        }
    }
}

/// The slots of a run's external edges: in use, and spare for the next quantum.
#[derive(Default)]
struct ExtSlots {
    active: Vec<Arc<ExtSlot>>,
    spare: Vec<Arc<ExtSlot>>,
}

struct StageRun {
    /// Incoming edges (and external edges) not yet arrived. The edge that brings it to zero publishes the stage, and
    /// sees everything the other edges' predecessors wrote before they arrived.
    pending: AtomicUsize,
    /// The lane it ran on, or `NO_LANE`: written by the stage, read by the successors that continue its lane after
    /// their final arrival.
    lane: Owned<u32>,
    /// Whether it submitted (a stage that did not, ran or not, is abandoned): written by the thread that owns its
    /// frame, read at quiesce.
    submitted: Owned<bool>,
    /// Whether it ran at all (Java `attempted`): its `retired` hook runs at the end of the quantum.
    attempted: Owned<bool>,
    /// The markers its successors and the edges on it wait for, once it submitted: the marker it queued, or else the
    /// markers it passed on.
    out: Owned<Vec<Marker>>,
    /// The external edges waiting for `out`.
    waiters: TakeAllStack<ExtSlot>,
    /// Where the retirement armed on its lane for its completion edges leaves its status (Java `RetiredEdge`), and
    /// whether one is armed this quantum.
    edge_cell: Arc<RetireCell>,
    edge_armed: Owned<bool>,
    /// The stage's gate (`GATE_*`): set when a gate is taken, and moved on by whoever spends it (release, fail or drop)
    /// and, for a gate taken in `run`, by the end of that run. Really shared: the thread that gates or runs the stage and
    /// the thread that spends the gate meet here.
    gate: crate::sync::AtomicU8,
}

/// No gate is out.
const GATE_IDLE: u8 = 0;
/// A gate is out and the stage is not running: it was taken in `pass_gate`, or the run that took it has ended with `Held`.
/// Whoever spends it publishes the stage, or fails the quantum and drops the stage's count.
const GATE_HELD: u8 = 1;
/// A gate taken in `run` is out, and that run has not ended.
const GATE_RUNNING: u8 = 2;
/// The gate taken in the running `run` was released: the end of the run publishes the stage.
const GATE_RELEASED: u8 = 3;
/// The gate taken in the running `run` was failed (the quantum is failed already): the end of the run drops the count.
const GATE_FAILED: u8 = 4;

/// What the party that spends a gate does next.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Spend {
    /// The stage is held: the spender publishes it (a release), or drops its count (a fail).
    Now,
    /// The run that took the gate has not ended: its end does it.
    AtRunEnd,
    /// No gate is out: nothing to do.
    Nothing,
}

/// What the end of a run that took a gate finds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RunEnd {
    /// The gate is out: the stage is held, and its count is the gate's.
    Parked,
    /// The gate was released while the run went on: the end publishes the stage.
    Released,
    /// The gate was failed while the run went on: the end drops the stage's count.
    Failed,
    /// No gate is out.
    NoGate,
}

impl StageRun {
    fn new() -> Self {
        Self {
            pending: AtomicUsize::new(0),
            lane: Owned::new(NO_LANE),
            submitted: Owned::new(false),
            attempted: Owned::new(false),
            out: Owned::new(Vec::new()),
            waiters: TakeAllStack::new(),
            edge_cell: Arc::new(RetireCell::new()),
            edge_armed: Owned::new(false),
            gate: crate::sync::AtomicU8::new(GATE_IDLE),
        }
    }

    /// The stage's gate is spent, to `to` (`GATE_RELEASED` or `GATE_FAILED`): a stage whose run is still going is left to
    /// that run's end, which then sees `to`.
    ///
    /// One exchange, which is the hand-over with [`run_ended_with_gate`](Self::run_ended_with_gate): of the end of the run
    /// and the spender, the second to arrive acts, and sees what the first wrote before it (release on the way in,
    /// acquire on the way out). A gate is spent once (it is moved), so no third party meets the state.
    fn spend_gate(&self, to: u8) -> Spend {
        match self.gate.swap(to, Ordering::AcqRel) {
            GATE_RUNNING => Spend::AtRunEnd,
            // Parked, or taken in `pass_gate`: nothing else acts on a spent gate, and the next to look is the run the stage
            // is published for.
            GATE_HELD => {
                self.gate.store(GATE_IDLE, Ordering::Relaxed);
                Spend::Now
            }
            _ => Spend::Nothing,
        }
    }

    /// A run that took a gate has ended: the stage is parked, or the gate was spent meanwhile.
    fn run_ended_with_gate(&self) -> RunEnd {
        match self.gate.compare_exchange(
            GATE_RUNNING,
            GATE_HELD,
            Ordering::Release,
            Ordering::Acquire,
        ) {
            Ok(_) => RunEnd::Parked,
            Err(spent @ (GATE_RELEASED | GATE_FAILED)) => {
                self.gate.store(GATE_IDLE, Ordering::Relaxed);
                if spent == GATE_RELEASED {
                    RunEnd::Released
                } else {
                    RunEnd::Failed
                }
            }
            Err(_) => RunEnd::NoGate,
        }
    }

    /// An incoming edge arrived. True for the last: the caller publishes the stage. Whatever a predecessor wrote
    /// before it arrived is visible to the caller that sees `true` (Java `StageFrame.arrive`: `ARRIVALS.getAndAdd`).
    fn arrive(&self) -> bool {
        self.pending.fetch_sub(1, Ordering::AcqRel) == 1
    }
}

/// The first cause of a quantum not completing: one claim wins and writes it (Java
/// `AbstractQuantum.failure.compareAndSet(null, cause)`).
struct FirstCause<T> {
    claimed: crate::sync::AtomicBool,
    value: Owned<Option<T>>,
}

impl<T> FirstCause<T> {
    fn new() -> Self {
        Self {
            claimed: crate::sync::AtomicBool::new(false),
            value: Owned::new(None),
        }
    }

    /// Whether a cause was claimed.
    fn is_claimed(&self) -> bool {
        self.claimed.load(Ordering::Acquire)
    }

    /// Forgets the cause.
    ///
    /// # Safety
    ///
    /// No thread claims or takes meanwhile, and every earlier claim happens before this call.
    unsafe fn reset(&self) {
        // SAFETY: the caller's contract.
        unsafe { self.value.write(|value| *value = None) };
        self.claimed.store(false, Ordering::Relaxed);
    }

    /// Claims the cause if no one did. True for the winner.
    fn claim(&self, cause: T) -> bool {
        if self.claimed.load(Ordering::Acquire)
            || self
                .claimed
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                .is_err()
        {
            return false;
        }
        // SAFETY: the compare-and-swap made this thread the only writer; the reader (`take`) runs after every
        // claimant is done, by the caller's contract.
        unsafe { self.value.write(|value| *value = Some(cause)) };
        true
    }

    /// The cause, if one was claimed.
    ///
    /// # Safety
    ///
    /// Every claimant is done, and what each did happens before this call.
    unsafe fn take(&self) -> Option<T> {
        // SAFETY: the caller's contract.
        unsafe { self.value.write(Option::take) }
    }
}

/// The lanes a quantum used, and the latest marker value on each (Java `usedLanes`, and the lane tails).
struct LaneTable {
    /// A bit per lane used.
    used: AtomicU64,
    values: Box<[AtomicU64]>,
}

impl LaneTable {
    fn new(lanes: usize) -> Self {
        Self {
            used: AtomicU64::new(0),
            values: (0..lanes).map(|_| AtomicU64::new(0)).collect(),
        }
    }

    /// `value` on `lane` joins the table: the latest of a lane covers the earlier ones (a lane runs in order). The
    /// value is in before the bit, so a reader that sees the bit sees at least this value.
    fn record(&self, lane: usize, value: u64) {
        self.values[lane].fetch_max(value, Ordering::AcqRel);
        let bit = 1u64 << lane;
        if self.used.load(Ordering::Acquire) & bit == 0 {
            self.used.fetch_or(bit, Ordering::AcqRel);
        }
    }

    /// The lanes used, a bit each.
    fn used(&self) -> u64 {
        self.used.load(Ordering::Acquire)
    }

    fn value(&self, lane: usize) -> u64 {
        self.values[lane].load(Ordering::Acquire)
    }

    /// Empties the table. Only while no thread uses it.
    fn clear(&self) {
        let mut used = self.used.swap(0, Ordering::AcqRel);
        while used != 0 {
            let lane = used.trailing_zeros() as usize;
            used &= used - 1;
            self.values[lane].store(0, Ordering::Relaxed);
        }
    }
}

/// A quantum bound to its graph: pooled, and rebound to each quantum in turn.
struct Run<Q, L: DeviceQueue + ?Sized> {
    core: Arc<Core<Q, L>>,
    /// Where the run goes when its quantum retires.
    home: Weak<Producer<Arc<Run<Q, L>>, Multi>>,
    /// The lane its roots continue in every quantum (Java `StageGraph.home`).
    home_lane: LaneId,
    /// Read while stages run (concurrently); taken by the retirement, after every stage is done.
    quantum: Owned<Option<Q>>,
    kill: Owned<Option<Arc<AtomicBool>>>,
    /// Published frames, armed completion edges and host completions, external edges not yet arrived, and
    /// admission's hold.
    live: AtomicUsize,
    /// The quantum's own token, its `QuantumRef` pin, and the external edges of later quanta that
    /// use it.
    holds: AtomicUsize,
    /// A stage failed, or a frame was refused: nothing more is published.
    stopped: AtomicBool,
    /// The first reason the quantum did not complete; read by the retirement, after every frame is done.
    cause: FirstCause<QuantumOutcome>,
    stages: Box<[StageRun]>,
    /// Each stage's wake-up for when it answers `Retry`.
    retry_wakes: Box<[Arc<RetryWake<L>>]>,
    /// The lanes the quantum submitted to, and the latest marker value on each.
    lanes: LaneTable,
    /// Boundary completions still to come, and one for the arming.
    boundary_left: AtomicUsize,
    /// Where the retirement armed on the home lane for the boundary leaves its status.
    retire_cell: Arc<RetireCell>,
    /// Whether the `abandoned` hook of a stage that names a lane ran this quantum (written by `quiesce`, read by the
    /// boundary).
    abandoned_on_lanes: Owned<bool>,
    /// The latest marker value of each lane already awaited on the home lane (0: none), so the boundary joins only what
    /// later quanta added since `quiesce` joined.
    joined_values: Owned<Vec<u64>>,
    ext: Owned<ExtSlots>,
}

impl<Q: Send + Sync + 'static, L: DeviceQueue + ?Sized + 'static> Run<Q, L> {
    fn new(core: Arc<Core<Q, L>>, home: Weak<Producer<Arc<Run<Q, L>>, Multi>>) -> Arc<Self> {
        let home_lane = core.runtime.lanes.next_home();
        let stages = (0..core.spec.stages.len())
            .map(|_| StageRun::new())
            .collect();
        let runtime = &core.runtime;
        let retry_wakes = (0..core.spec.stages.len())
            .map(|_| {
                Arc::new(RetryWake::new(
                    runtime.lanes.clone(),
                    runtime.retries.clone(),
                    runtime.retry_backstop,
                ))
            })
            .collect();
        let lanes = LaneTable::new(core.runtime.lanes.len());
        Arc::new(Self {
            core,
            home,
            home_lane,
            quantum: Owned::new(None),
            kill: Owned::new(None),
            live: AtomicUsize::new(0),
            holds: AtomicUsize::new(0),
            stopped: AtomicBool::new(false),
            cause: FirstCause::new(),
            stages,
            retry_wakes,
            lanes,
            boundary_left: AtomicUsize::new(0),
            retire_cell: Arc::new(RetireCell::new()),
            abandoned_on_lanes: Owned::new(false),
            joined_values: Owned::new(Vec::new()),
            ext: Owned::new(ExtSlots::default()),
        })
    }

    /// Binds `quantum`.
    fn bind(&self, quantum: Q, kill: Arc<AtomicBool>) {
        // SAFETY: this thread took the run out of the pool (or made it): nothing else holds it, and the pool's queue
        // handed over every earlier access.
        unsafe {
            self.quantum.write(|q| *q = Some(quantum));
            self.kill.write(|k| *k = Some(kill));
        }
        // SAFETY: as above: this thread owns the run until it publishes a frame or a reference.
        unsafe { self.cause.reset() };
        // SAFETY: as above.
        unsafe {
            self.abandoned_on_lanes.set(false);
            self.joined_values.write(|v| {
                v.clear();
                v.resize(64, 0);
            });
        }
        // The returned ref's pin, and the quantum's own token (dropped once its live count reached 0).
        self.holds.store(2, Ordering::Relaxed);
        self.stopped.store(false, Ordering::Relaxed);
        for (index, state) in self.stages.iter().enumerate() {
            state
                .pending
                .store(self.core.topology.incoming[index], Ordering::Relaxed);
            // SAFETY: as above.
            unsafe {
                state.lane.set(NO_LANE);
                state.submitted.set(false);
                state.attempted.set(false);
                state.edge_armed.set(false);
                state.out.write(Vec::clear);
            }
            state.waiters.reopen();
            state.gate.store(GATE_IDLE, Ordering::Relaxed);
        }
    }

    /// Ends the binding: the quantum is gone, and the run goes back to the pool.
    fn unbind(self: &Arc<Self>) {
        // SAFETY: the retirement is the last thing that touches a quantum's run: every frame, pin and external edge
        // of it is done.
        unsafe {
            self.kill.write(|k| *k = None);
            self.ext.write(|ext| {
                while let Some(slot) = ext.active.pop() {
                    slot.markers.write(Vec::clear);
                    slot.pin.write(|pin| *pin = None);
                    slot.target.write(|target| *target = None);
                    ext.spare.push(slot);
                }
            });
        }
        self.lanes.clear();
        give_back(&self.home, self.clone());
    }

    /// Takes the record of `edge`; the stage it leads to waits for it. Before any frame of the quantum is published.
    fn add_external(self: &Arc<Self>, edge: ExternalEdge) {
        let ExternalEdge {
            from,
            from_stage,
            to_stage,
            kind,
        } = edge;
        // SAFETY: bound but not yet published: this thread owns the slots.
        unsafe {
            self.ext.write(|ext| {
                let mut slot = ext.spare.pop().unwrap_or_else(|| Arc::new(ExtSlot::new()));
                {
                    let fresh = Arc::get_mut(&mut slot).expect("a spare slot is alone");
                    fresh.index = ext.active.len();
                    fresh.to_stage = to_stage;
                    fresh.from_stage = from_stage;
                    fresh.kind = kind;
                }
                slot.pin.write(|pin| *pin = Some(from));
                ext.active.push(slot);
            });
        }
        self.stages[usize::from(to_stage.0)]
            .pending
            .fetch_add(1, Ordering::AcqRel);
    }

    /// Puts edge `index` on its earlier quantum.
    fn register_external(self: &Arc<Self>, index: usize) {
        // SAFETY: the slots do not change while the quantum runs, and the pin is held until the edge arrives.
        let (upstream, from_stage, node) = unsafe {
            self.ext.read(|ext| {
                let slot = &ext.active[index];
                // The edge holds its run until it arrives (a record the earlier quantum's stack leads back to).
                let target: Arc<dyn Downstream> = self.clone();
                slot.target.write(|t| *t = Some(target));
                let upstream = slot.pin.read(|pin| {
                    pin.as_ref()
                        .expect("an edge pins its quantum until it arrives")
                        .upstream()
                        .clone()
                });
                (upstream, slot.from_stage, NonNull::from(&**slot))
            })
        };
        upstream.subscribe(from_stage, node);
    }

    fn halted(&self) -> bool {
        self.stopped.load(Ordering::Acquire) || self.kill_set()
    }

    fn kill_set(&self) -> bool {
        // SAFETY: the flag is set by `bind`, before any frame or ref exists, and cleared by `unbind`, after the last
        // of them; in between it is only read.
        unsafe {
            self.kill
                .read(|k| k.as_ref().is_some_and(|k| k.load(Ordering::Acquire)))
        }
    }

    /// Claims the cause if no one did.
    fn note(&self, cause: QuantumOutcome) {
        self.cause.claim(cause);
    }

    fn fail(&self, stage: StageId, error: StageError) {
        self.note(QuantumOutcome::Failed { stage, error });
        self.stopped.store(true, Ordering::Release);
    }

    fn header(&self, ordered: Option<u64>, kill: bool) -> FrameHeader {
        let kill = kill.then(|| {
            // SAFETY: as `kill_set`.
            unsafe {
                self.kill
                    .read(|k| k.clone().expect("a bound run has its kill flag"))
            }
        });
        match ordered {
            Some(id) => FrameHeader::with_kill_switch(id, kill),
            None => {
                let mut header = FrameHeader::with_kill_switch(UNORDERED, kill);
                header.randomize_hash(self.core.next_seed());
                header
            }
        }
    }

    /// Offers `frame`; a refused one (the sink completed) ends in place.
    fn offer_frame(&self, frame: Box<dyn Frame>) {
        if let Err(frame) = self.core.runtime.stages.offer(frame) {
            frame.do_finally();
        }
    }

    /// Publishes `stage`, whose last incoming edge arrived (its live count is already taken).
    fn offer(self: &Arc<Self>, stage: StageId) {
        let ordered = self.core.spec.stages[usize::from(stage.0)].ordered;
        let frame = self
            .core
            .frame(self, Kind::Stage(stage), self.header(ordered, true));
        if let Err(frame) = self.core.runtime.stages.offer(frame) {
            self.fail(
                stage,
                StageError::Fatal("the stage sink no longer takes frames".into()),
            );
            frame.do_finally();
        }
    }

    /// Java `publishOrRun`, for a stage a gate held: offers its frame, and a sink that no longer takes frames has the
    /// stage run right here, on the thread that released the gate, so its held count is always dropped.
    ///
    /// Only an unordered stage can: an ordered stage's frames must run in the order they were published and one at a time
    /// on their lane, which only the sink enforces, so running one beside the sink could overtake or overlap an earlier
    /// frame. A refused ordered stage ends in place as any refused frame does (it fails the quantum).
    fn publish_or_run(self: &Arc<Self>, stage: StageId) {
        let ordered = self.core.spec.stages[usize::from(stage.0)].ordered;
        if ordered.is_some() {
            self.offer(stage);
            return;
        }
        let frame = self
            .core
            .frame(self, Kind::Stage(stage), self.header(None, true));
        if let Err(mut frame) = self.core.runtime.stages.offer(frame) {
            if !frame.is_alive() {
                frame.do_finally();
                return;
            }
            match frame.execute() {
                Ok(()) | Err(FrameError::Cancelled) => frame.do_finally(),
                Err(error) => frame.do_finally_with_error(error),
            }
        }
    }

    /// An edge into `stage` arrived; the last one publishes it.
    fn arrive(self: &Arc<Self>, stage: StageId) {
        let _hot = HotScope::enter();
        if self.stages[usize::from(stage.0)].arrive() {
            if self.halted() {
                if self.kill_set() {
                    self.note(QuantumOutcome::Killed);
                }
                return;
            }
            self.live.fetch_add(1, Ordering::AcqRel);
            self.gate_and_offer(stage);
        }
    }

    /// Java `publishGated`: asks the stage whether it may be published now; a stage that holds itself at its gate is
    /// published by whoever releases the gate, and a gate that fails fails the quantum and drops the count.
    ///
    /// The caller's own live count covers the quantum for the whole call (this runs from an arrival, inside the frame or
    /// admission that still holds its count), so the run may be read and the quantum failed after the stage returns, even
    /// though a released gate has published the stage and the stage's own count may be gone.
    fn gate_and_offer(self: &Arc<Self>, stage: StageId) {
        let index = usize::from(stage.0);
        let taken = std::cell::Cell::new(false);
        let gate = GateRef {
            source: self,
            stage,
            taken: &taken,
        };
        // SAFETY: the quantum is bound until it retires, which waits for the caller's live count (see above).
        let passed = unsafe {
            self.quantum.read(|quantum| {
                let quantum = quantum.as_ref().expect("a bound run has its quantum");
                catch_unwind(AssertUnwindSafe(|| {
                    self.core.stages[index].pass_gate(quantum, gate)
                }))
            })
        };
        if taken.get() {
            // Whoever holds the gate publishes the stage and its count is theirs. A stage that took a gate and then
            // panicked fails the quantum all the same (a gate dropped by the unwind already did, and dropped the count).
            if passed.is_err() {
                self.fail(
                    stage,
                    StageError::Fatal("the stage's pass_gate panicked".into()),
                );
            }
            return;
        }
        match passed {
            Ok(true) => self.offer(stage),
            Ok(false) => {
                self.fail(
                    stage,
                    StageError::Fatal("a stage held itself at its gate without taking one".into()),
                );
                self.finish_one();
            }
            Err(_) => {
                self.fail(
                    stage,
                    StageError::Fatal("the stage's pass_gate panicked".into()),
                );
                self.finish_one();
            }
        }
    }

    /// Drops one live count; the last one quiesces the quantum.
    fn finish_one(self: &Arc<Self>) {
        if self.live.fetch_sub(1, Ordering::AcqRel) == 1 {
            self.quiesce();
        }
    }

    /// Runs `stage` (in its frame's `execute`), unless the quantum stopped. `inputs` and `waits` are the frame's
    /// scratch: the markers the stage's predecessors and external edges brought, and those of them on other lanes.
    fn run_stage(
        self: &Arc<Self>,
        stage: StageId,
        inputs: &mut Vec<Marker>,
        waits: &mut Vec<Marker>,
        held: &AtomicBool,
    ) -> Option<Result<Submitted, StageError>> {
        let _hot = HotScope::enter();
        if self.stopped.load(Ordering::Acquire) {
            return None;
        }
        let index = usize::from(stage.0);
        let state = &self.stages[index];
        let topology = &self.core.topology;
        // SAFETY: this thread owns the stage's frame: `lane` is its to write, and every stage it reads
        // (its predecessors, and the path predecessor among them) submitted before its final arrival, which ordered
        // their writes before this call.
        let lane = unsafe {
            state.attempted.set(true);
            let spec = &self.core.spec.stages[index];
            let lane = (!spec.lanes.is_empty()).then(|| {
                // Java `continuedLane`: a root continues its graph's home lane; any other stage its path predecessor's
                // (none when that predecessor ran on no lane).
                let continued = if topology.submission_predecessors[index].is_empty() {
                    Some(self.home_lane)
                } else {
                    topology.path_predecessor[index]
                        .map(|p| self.stages[usize::from(p.0)].lane.get())
                        .filter(|&l| l != NO_LANE)
                        .map(|l| LaneId(l as u16))
                };
                self.core
                    .runtime
                    .lanes
                    .choose(stage.0, &spec.lanes, continued)
            });
            if let Some(lane) = lane {
                state.lane.set(u32::from(lane.0));
            }
            inputs.clear();
            for &predecessor in &topology.submission_predecessors[index] {
                self.stages[predecessor].out.read(|out| {
                    for &marker in out {
                        merge(inputs, marker);
                    }
                });
            }
            self.ext.read(|ext| {
                for slot in &ext.active {
                    if slot.to_stage == stage && slot.kind == EdgeKind::Submission {
                        slot.markers.read(|markers| {
                            for &marker in markers {
                                merge(inputs, marker);
                            }
                        });
                    }
                }
            });
            lane
        };
        waits.clear();
        waits.extend(inputs.iter().copied().filter(|m| Some(m.lane) != lane));
        let cx = StageCx {
            stage,
            lane,
            waits,
            lanes: &self.core.runtime.lanes,
            gates: self,
            held,
        };
        // SAFETY: the quantum is taken by the retirement, after every stage is done.
        Some(unsafe {
            self.quantum.read(|quantum| {
                self.core.stages[index]
                    .run(quantum.as_ref().expect("a running quantum is bound"), &cx)
            })
        })
    }

    /// A stage frame ended with `result`. Returns whether its live count is to be dropped (a retried stage keeps it, and
    /// a stage that took a gate in its run passes it to the gate). `held`: the run took a gate.
    fn stage_ended(
        self: &Arc<Self>,
        stage: StageId,
        result: Option<Result<Submitted, StageError>>,
        inputs: &[Marker],
        alive: bool,
        held: bool,
    ) -> bool {
        if held {
            return self.held_ended(stage, result, inputs);
        }
        match result {
            Some(Ok(submitted)) => match self.foreign(&submitted) {
                None => self.submitted(stage, submitted, inputs),
                Some(error) => {
                    self.fail(stage, error);
                    self.release_foreign(stage, submitted);
                }
            },
            Some(Err(StageError::Retry)) if alive => {
                // The live count passes to the retry, which offers the stage again.
                self.retry(stage);
                return false;
            }
            Some(Err(StageError::Retry)) => {
                self.note(QuantumOutcome::Killed);
            }
            Some(Err(StageError::Held)) => {
                self.fail(
                    stage,
                    StageError::Fatal("a stage answered Held without taking a gate".into()),
                );
                self.release_pins(stage, None);
            }
            Some(Err(error)) => {
                self.fail(stage, error);
                self.release_pins(stage, None);
            }
            None if !alive => self.note(QuantumOutcome::Killed),
            None => {}
        }
        true
    }

    /// The failed quantum's stage `stage` answered with a marker that is no lane of the pool: its pins go, and its host
    /// completion still holds the quantum live.
    fn release_foreign(self: &Arc<Self>, stage: StageId, submitted: Submitted) {
        self.release_pins(stage, None);
        if let Some(host) = submitted.completion {
            self.live.fetch_add(1, Ordering::AcqRel);
            self.completes(stage, None, Some(host));
        }
    }

    /// `stage`'s run took a gate ([`StageCx::hold`]) and ended with `result`. Returns whether its live count is to be
    /// dropped: not when the count is the gate's (the stage is parked, or the gate was released while the run went on and
    /// the stage is published again here).
    fn held_ended(
        self: &Arc<Self>,
        stage: StageId,
        result: Option<Result<Submitted, StageError>>,
        inputs: &[Marker],
    ) -> bool {
        let broke =
            || StageError::Fatal("a stage that took a gate in run did not answer Held".into());
        match result {
            Some(Err(StageError::Held)) | None => {}
            Some(Err(StageError::Retry)) => {
                self.fail(stage, broke());
                self.release_pins(stage, None);
            }
            // The stage's own error stands as the cause.
            Some(Err(error)) => {
                self.fail(stage, error);
                self.release_pins(stage, None);
            }
            // The quantum fails first, so that the submission below is handled as a stopped quantum's: the stage did
            // submit, so its marker joins the boundary and its host completion keeps the quantum live until it is ready
            // (device work and the work a completion polls may still read what the quantum holds).
            Some(Ok(submitted)) => {
                self.fail(stage, broke());
                match self.foreign(&submitted) {
                    None => self.submitted(stage, submitted, inputs),
                    Some(_) => self.release_foreign(stage, submitted),
                }
            }
        }
        self.gate_after_run(stage)
    }

    /// The run of `stage`, which took a gate, ended: parks the stage, or does what the spender of the gate left to this
    /// end. Returns whether the stage's live count is to be dropped.
    fn gate_after_run(self: &Arc<Self>, stage: StageId) -> bool {
        match self.stages[usize::from(stage.0)].run_ended_with_gate() {
            RunEnd::Parked => false,
            RunEnd::Released => {
                self.publish_or_run(stage);
                false
            }
            RunEnd::Failed => true,
            RunEnd::NoGate => {
                // `held` says the run took a gate, and only this end moves the gate out of `GATE_RUNNING` but a spender.
                debug_assert!(false, "a run that took a gate ended with no gate out");
                true
            }
        }
    }

    /// The failure of a stage whose marker names a queue that is no lane of the pool: the boundary could never
    /// reach it.
    fn foreign(&self, submitted: &Submitted) -> Option<StageError> {
        let marker = submitted.marker?;
        (!self.is_lane_marker(marker)).then(|| {
            StageError::Fatal(format!(
                "a marker of {} is no lane {} of the lane pool",
                marker.queue, marker.lane.0
            ))
        })
    }

    /// Whether `marker` is of a lane of this graph's pool, and of that lane's queue.
    fn is_lane_marker(&self, marker: Marker) -> bool {
        let lanes = &self.core.runtime.lanes;
        usize::from(marker.lane.0) < lanes.len() && lanes.lane(marker.lane).id() == marker.queue
    }

    /// A stage frame failed outright (`execute` panicked or returned an error). Returns whether its live count is to be
    /// dropped (not when a gate taken in the run holds it). `held`: the run took a gate.
    fn stage_errored(self: &Arc<Self>, stage: StageId, error: FrameError, held: bool) -> bool {
        let name = self.core.spec.stages[usize::from(stage.0)].name;
        self.fail(
            stage,
            StageError::Fatal(format!("stage {name} failed: {error}")),
        );
        self.release_pins(stage, None);
        !held || self.gate_after_run(stage)
    }

    /// `stage` answered [`StageError::Retry`]: it runs again on a lane's progress or at the
    /// backstop (its live count is kept).
    fn retry(self: &Arc<Self>, stage: StageId) {
        // It has not submitted: a kill while it waits abandons it.
        let runtime = &self.core.runtime;
        runtime.retries.retries.fetch_add(1, Ordering::Relaxed);
        let wake = &self.retry_wakes[usize::from(stage.0)];
        // SAFETY: the stage runs once at a time, and its previous wake-up fired before the frame that runs it again was
        // published.
        unsafe { wake.prepare() };
        let frame = self
            .core
            .frame(self, Kind::Retry(stage), self.header(None, true));
        runtime.completions.after_wake(wake.clone(), frame);
    }

    /// A retried stage's wake-up arrived. Returns whether its live count is to be dropped.
    fn retry_woke(self: &Arc<Self>, stage: StageId, alive: bool) -> bool {
        if !alive {
            self.note(QuantumOutcome::Killed);
            return true;
        }
        if self.stopped.load(Ordering::Acquire) {
            return true;
        }
        self.offer(stage);
        false
    }

    /// The latest marker of `marker`'s lane joins the quantum's boundary.
    fn record(&self, marker: Marker) {
        self.lanes.record(usize::from(marker.lane.0), marker.value);
    }

    /// The latest marker of every lane the quantum used, appended to `into`.
    fn frontier_into(&self, into: &mut Vec<Marker>) {
        let mut used = self.lanes.used();
        while used != 0 {
            let lane = used.trailing_zeros() as usize;
            used &= used - 1;
            into.push(self.lane_marker(lane));
        }
    }

    /// The latest marker of `lane`, a lane the quantum used.
    fn lane_marker(&self, lane: usize) -> Marker {
        let id = LaneId(lane as u16);
        Marker::new(
            self.core.runtime.lanes.lane(id).id(),
            id,
            self.lanes.value(lane),
        )
    }

    /// `stage` submitted: its edges are satisfied (submission edges now, completion edges once its
    /// work completed). `inputs` are the markers it ran after, passed on when it queued none.
    fn submitted(self: &Arc<Self>, stage: StageId, submitted: Submitted, inputs: &[Marker]) {
        let _hot = HotScope::enter();
        let index = usize::from(stage.0);
        let Submitted { marker, completion } = submitted;
        // Device order passes through a stage that queued nothing.
        // SAFETY: this thread owns the stage's frame; the readers of `out` come after the seal below, or after the
        // successors' final arrival.
        unsafe {
            self.stages[index].submitted.set(true);
            self.stages[index].out.write(|out| {
                out.clear();
                match marker {
                    Some(marker) => out.push(marker),
                    None => out.extend_from_slice(inputs),
                }
            });
        }
        if let Some(marker) = marker {
            self.record(marker);
        }
        let completion_edge = self.core.topology.has_completion_successor[index];
        self.release_pins(stage, marker);
        self.release_waiters(stage, Sealed::Submitted);
        if self.halted() {
            if self.kill_set() {
                self.note(QuantumOutcome::Killed);
            }
            // A host completion still holds the quantum live, whether or not an edge would consume it (Java's
            // deferral): its work may still use what the quantum holds.
            if let Some(host) = completion {
                self.live.fetch_add(1, Ordering::AcqRel);
                self.completes(stage, None, Some(host));
            }
            return;
        }
        if completion_edge {
            self.live.fetch_add(1, Ordering::AcqRel);
            self.completes(stage, marker, completion);
        } else if let Some(host) = completion {
            // The boundary has the marker; the host completion holds a live count until it is ready.
            self.live.fetch_add(1, Ordering::AcqRel);
            self.completes(stage, None, Some(host));
        }
        for &(successor, kind) in &self.core.topology.successors[index] {
            if kind == EdgeKind::Submission {
                self.arrive(successor);
            }
        }
    }

    /// Registers the arrival frame of `stage`'s completion: when its `marker` is reached and its `host` completion is
    /// ready.
    ///
    /// On the predecessor's ordered lane, if it has one: its completions (emitted in registration order) then publish
    /// their successors in that order, so an ordered successor keeps the predecessor's order.
    fn completes(
        self: &Arc<Self>,
        stage: StageId,
        marker: Option<Marker>,
        host: Option<Box<dyn Ready>>,
    ) {
        let ordered = self.core.spec.stages[usize::from(stage.0)].ordered;
        let frame = self
            .core
            .frame(self, Kind::Completions(stage), self.header(ordered, true));
        let completions = &self.core.runtime.completions;
        // Java `RetiredEdge`: the retirement is armed on the producer's lane, and its status settled when it arrives.
        if let Some(marker) = marker
            && self.arm_edge(stage, marker)
        {
            let state = &self.stages[usize::from(stage.0)];
            match host {
                None => completions.after_retired(state.edge_cell.clone(), frame),
                Some(host) => completions.after(
                    Box::new(RetiredAndHost {
                        cell: state.edge_cell.clone(),
                        host,
                        host_ready: false,
                    }),
                    frame,
                ),
            }
            return;
        }
        match (marker, host) {
            (Some(marker), None) => completions.after_marker(marker, frame),
            (None, Some(host)) => completions.after(host, frame),
            (Some(marker), Some(host)) => completions.after(
                Box::new(MarkerAndHost {
                    lanes: self.core.runtime.lanes.clone(),
                    marker,
                    marker_reached: false,
                    host,
                    host_ready: false,
                }),
                frame,
            ),
            // Nothing to wait for: the next poll emits it.
            (None, None) => completions.at(Instant::now(), frame),
        }
    }

    /// Arms the retirement of a completion edge on the lane `marker` is of, when that lane has control. False when it
    /// has none, or the arming was refused.
    fn arm_edge(&self, stage: StageId, marker: Marker) -> bool {
        let state = &self.stages[usize::from(stage.0)];
        let lanes = &self.core.runtime.lanes;
        if usize::from(marker.lane.0) >= lanes.len() {
            return false;
        }
        let Some(control) = lanes.lane(marker.lane).control() else {
            return false;
        };
        state.edge_cell.rearm();
        if control.notify_retired(&state.edge_cell).is_err() {
            return false;
        }
        // SAFETY: this thread owns the stage's frame until its arrival is registered.
        unsafe { state.edge_armed.set(true) };
        true
    }

    /// A completion arrival of `stage`'s ended: its completion-edge successors arrive.
    fn completed(self: &Arc<Self>, stage: StageId, alive: bool) {
        if alive && self.edge_failed(stage) {
            return;
        }
        if !alive {
            self.note(QuantumOutcome::Killed);
        } else if !self.stopped.load(Ordering::Acquire) {
            for &(successor, kind) in &self.core.topology.successors[usize::from(stage.0)] {
                if kind == EdgeKind::Completion {
                    self.arrive(successor);
                }
            }
        }
    }

    /// Settles the retirement of `stage`'s completion edge, if one was armed: a device failure fails the quantum and
    /// releases no successor (Java `RetiredEdge.execute`).
    fn edge_failed(&self, stage: StageId) -> bool {
        let state = &self.stages[usize::from(stage.0)];
        // SAFETY: the arrival that settles the edge runs after the stage's frame ended.
        let (armed, lane) = unsafe { (state.edge_armed.get(), state.lane.get()) };
        if !armed || lane == NO_LANE {
            return false;
        }
        let Some(status) = state.edge_cell.status() else {
            return false;
        };
        let Some(control) = self.core.runtime.lanes.lane(LaneId(lane as u16)).control() else {
            return false;
        };
        match control.confirm_retired(status) {
            Some(error) => {
                self.fail(stage, StageError::Fatal(error.to_string()));
                true
            }
            None => false,
        }
    }

    /// The marker of an external completion edge was reached. Returns whether this was the last, and the edge's live
    /// count is to be dropped.
    fn external_reached(self: &Arc<Self>, index: usize, alive: bool) -> bool {
        // SAFETY: the slots do not change while the quantum runs.
        let (to, last) = unsafe {
            self.ext.read(|ext| {
                let slot = &ext.active[index];
                (slot.to_stage, slot.left.fetch_sub(1, Ordering::AcqRel) == 1)
            })
        };
        if !last {
            return false;
        }
        if !alive {
            self.note(QuantumOutcome::Killed);
        } else if !self.stopped.load(Ordering::Acquire) {
            self.arrive(to);
        }
        true
    }

    /// Ends the pins this quantum's external edges into `stage` hold, the used stage's `marker`
    /// joining the earlier quanta's boundaries.
    fn release_pins(&self, stage: StageId, marker: Option<Marker>) {
        // SAFETY: the slots do not change while the quantum runs, and a pin is taken by the stage it serves (once:
        // it submits or fails once) or by the quiesce, after every stage is done.
        let count = unsafe { self.ext.read(|ext| ext.active.len()) };
        for index in 0..count {
            // SAFETY: as above.
            let pin = unsafe {
                self.ext.read(|ext| {
                    let slot = &ext.active[index];
                    if slot.to_stage == stage {
                        slot.pin.write(Option::take)
                    } else {
                        None
                    }
                })
            };
            if let Some(pin) = pin {
                if let Some(marker) = marker {
                    pin.upstream().cover(marker);
                }
                drop(pin);
            }
        }
    }

    /// Seals `stage`'s stack for `why`, and satisfies the external edges that waited on it.
    fn release_waiters(&self, stage: StageId, why: Sealed) {
        for node in self.stages[usize::from(stage.0)].waiters.seal(why) {
            self.deliver(stage, why, node);
        }
    }

    /// Satisfies the external edge in `node` with what `stage` has to give: its output markers, or (swept) the
    /// markers of the whole quantum.
    fn deliver(&self, stage: StageId, why: Sealed, node: NonNull<ExtSlot>) {
        // SAFETY: the stack handed the node over (or refused it): this thread owns its markers and target until the
        // target arrives, and the node is valid until then. `out` is final once the stack is sealed `Submitted`.
        let (target, index) = unsafe {
            let slot = node.as_ref();
            slot.markers.write(|markers| {
                markers.clear();
                match why {
                    Sealed::Submitted => self.stages[usize::from(stage.0)]
                        .out
                        .read(|out| markers.extend_from_slice(out)),
                    Sealed::Swept => self.frontier_into(markers),
                }
            });
            let target = slot
                .target
                .write(Option::take)
                .expect("an edge arrives once");
            (target, slot.index)
        };
        target.external_arrived(index);
    }

    /// Java `quiesce`, up to the sweep: the quantum's own lanes join the home lane, the stages that never submitted are
    /// abandoned (with the home lane, behind that join), and only then are later quanta waiting on its stages released,
    /// so what a hook releases is there for them. Its own token is dropped (the retirement is armed once no pin is left).
    fn quiesce(self: &Arc<Self>) {
        self.take_over_external_waits();
        let used = self.lanes.used();
        if used != 0 {
            // A join this early covers the quantum's own markers; the boundary joins again for the markers later quanta
            // add while they pin it (an await of a marker that is already awaited orders nothing more).
            let _ = self.join_lanes(used);
        }
        let any = self.abandon_unsubmitted();
        // SAFETY: every stage is done and every pin is gone or kept by a later quantum: this thread is the only writer.
        unsafe { self.abandoned_on_lanes.set(any) };
        for index in 0..self.stages.len() {
            self.release_waiters(StageId(index as u16), Sealed::Swept);
        }
        // SAFETY: every stage is done: no one releases a pin any more.
        let count = unsafe { self.ext.read(|ext| ext.active.len()) };
        for index in 0..count {
            // SAFETY: as above.
            let pin = unsafe {
                self.ext
                    .read(|ext| ext.active[index].pin.write(Option::take))
            };
            drop(pin);
        }
        self.clone().release_hold();
    }

    /// The quantum's boundary takes over the markers its external edges brought (Java `awaitExternals`, which joins
    /// every slot's marker to the home lane), so a later quantum released onto this one still follows the quanta before
    /// it. A stage that never submitted (killed, failed, a CPU stage) did not wait for them itself.
    fn take_over_external_waits(&self) {
        // SAFETY: every stage is done: the slots and their markers are no longer written.
        let count = unsafe { self.ext.read(|ext| ext.active.len()) };
        for index in 0..count {
            // SAFETY: as above.
            unsafe {
                self.ext.read(|ext| {
                    let slot = &ext.active[index];
                    slot.markers.read(|markers| {
                        for &marker in markers {
                            if self.is_lane_marker(marker) {
                                self.record(marker);
                            }
                        }
                    });
                });
            }
        }
    }

    /// Every pin is gone: Java `quiesce`. The lanes the quantum used join the home lane (when they have control), the
    /// stages that never submitted are abandoned (with the home lane, behind the join), and the retirement is armed:
    /// one on the home lane, or one completion per lane the quantum used.
    fn arm_boundary(self: &Arc<Self>) {
        let used = self.lanes.used();
        let joined = used != 0 && self.join_lanes(used);
        // SAFETY: written by `quiesce`, which happens before this call.
        let hooks_ran = unsafe { self.abandoned_on_lanes.get() };
        // Java always arms `home.notifyRetired` after `abandonUnsubmitted(home)`: here when a lane was used, or a hook of a
        // stage that names a lane ran (what it submitted to home is covered before `retired` and `retire` drop anything).
        // A quantum with neither touched no lane: it has nothing on a lane for a retirement to cover, arms none, and
        // concludes here, in the call that dropped its last count or hold, with no frame of its own.
        if used == 0 && !hooks_ran {
            self.conclude();
            return;
        }
        if (used == 0 && hooks_ran || joined) && self.retire_on_home() {
            return;
        }
        if used == 0 {
            // A hook of a stage that names a lane ran and the retirement could not be armed. A home lane with control that refused it leaves the
            // hooks' work uncovered, which the quantum reports; a lane type with no control (a query-only `DeviceQueue`)
            // gives a hook nothing to enqueue, so there is nothing to cover.
            if self
                .core
                .runtime
                .lanes
                .lane(self.home_lane)
                .control()
                .is_some()
            {
                self.note(QuantumOutcome::DeviceFailed {
                    cause: "the retirement covering the abandoned hooks' work could not be armed on the home lane".into(),
                });
            }
            self.offer_retirement();
            return;
        }
        // The lanes are only a query side (or the join could not be made): one completion per lane.
        // One per lane, and one for this arming, so that the boundary cannot end before every lane is in.
        self.boundary_left
            .store(used.count_ones() as usize + 1, Ordering::Release);
        let mut lanes = used;
        while lanes != 0 {
            let lane = lanes.trailing_zeros() as usize;
            lanes &= lanes - 1;
            let frame = self
                .core
                .frame(self, Kind::Boundary, self.header(None, false));
            self.core
                .runtime
                .completions
                .after_marker(self.lane_marker(lane), frame);
        }
        if self.boundary_left.fetch_sub(1, Ordering::AcqRel) == 1 {
            self.offer_retirement();
        }
    }

    /// Every other lane the quantum used joins the home lane on the device (it awaits that lane's latest marker of the
    /// quantum). False when the lanes offer no control or a join was refused: the caller waits for each lane's marker
    /// instead.
    fn join_lanes(&self, used: u64) -> bool {
        let lanes = &self.core.runtime.lanes;
        let Some(home) = lanes.lane(self.home_lane).control() else {
            return false;
        };
        let mut others = used & !(1u64 << self.home_lane.0);
        while others != 0 {
            let lane = others.trailing_zeros() as usize;
            others &= others - 1;
            let marker = self.lane_marker(lane);
            // SAFETY: `quiesce` and then the boundary call this, one after the other, never at once.
            let seen = unsafe { self.joined_values.read(|v| v[lane]) };
            if seen == marker.value {
                continue;
            }
            if home.await_marker(marker).is_err() {
                return false;
            }
            // SAFETY: as above.
            unsafe { self.joined_values.write(|v| v[lane] = marker.value) };
        }
        true
    }

    /// Arms one retirement on the home lane, whose callback leaves its status in this run's cell. A frame emitted when
    /// the cell is filled settles it and concludes the quantum. False when the arming was refused (nothing is then
    /// waiting on the cell).
    fn retire_on_home(self: &Arc<Self>) -> bool {
        let lanes = &self.core.runtime.lanes;
        let Some(home) = lanes.lane(self.home_lane).control() else {
            return false;
        };
        self.retire_cell.rearm();
        if home.notify_retired(&self.retire_cell).is_err() {
            return false;
        }
        let frame = self
            .core
            .frame(self, Kind::Retired, self.header(None, false));
        self.core
            .runtime
            .completions
            .after_retired(self.retire_cell.clone(), frame);
        true
    }

    /// Java `abandonUnsubmitted(home)`: every stage that never submitted (it never ran, or ran and failed) gets its hook,
    /// in stage order, once, with the home lane. A hook that panics fails the quantum and the others still run. Whether
    /// a hook of a stage that names a lane ran: only such a hook may have enqueued on the home lane (Java `host()`: a
    /// stage that names no lane does host work only).
    fn abandon_unsubmitted(&self) -> bool {
        let home = self.core.runtime.lanes.lane(self.home_lane);
        let mut any = false;
        // SAFETY: every stage is done; the quantum is taken by the retirement, after the boundary.
        unsafe {
            self.quantum.read(|quantum| {
                let quantum = quantum.as_ref().expect("a quiescing quantum is bound");
                for (index, state) in self.stages.iter().enumerate() {
                    if state.submitted.get() {
                        continue;
                    }
                    any |= !self.core.spec.stages[index].lanes.is_empty();
                    let hook = catch_unwind(AssertUnwindSafe(|| {
                        self.core.stages[index].abandoned(quantum, &**home);
                    }));
                    if hook.is_err() {
                        self.note(QuantumOutcome::Failed {
                            stage: StageId(index as u16),
                            error: StageError::Fatal("the stage's abandoned hook panicked".into()),
                        });
                    }
                }
            });
        }
        any
    }

    /// The home lane's retirement arrived: settles its status. A failure of the device work fails the quantum.
    fn settle_boundary(&self) {
        let lanes = &self.core.runtime.lanes;
        let Some(status) = self.retire_cell.status() else {
            return;
        };
        if let Some(control) = lanes.lane(self.home_lane).control()
            && let Some(error) = control.confirm_retired(status)
        {
            self.note(QuantumOutcome::DeviceFailed {
                cause: error.to_string(),
            });
        }
    }

    fn offer_retirement(self: &Arc<Self>) {
        let frame = self
            .core
            .frame(self, Kind::Retire, self.header(None, false));
        self.offer_frame(frame);
    }

    fn conclude(self: &Arc<Self>) {
        // SAFETY: the boundary is reached, every pin is gone and every frame is done: the retirement is the only
        // thread that touches the quantum and the cause.
        let Some(quantum) = (unsafe { self.quantum.write(Option::take) }) else {
            return;
        };
        self.run_retired(&quantum);
        // SAFETY: as above.
        let cause = unsafe { self.cause.take() };
        let outcome = cause.unwrap_or(QuantumOutcome::Completed);
        let retired = catch_unwind(AssertUnwindSafe(|| (self.core.retire)(quantum, outcome)));
        self.core.live.fetch_sub(1, Ordering::AcqRel);
        self.unbind();
        if let Err(panic) = retired {
            resume_unwind(panic);
        }
    }
}

impl<Q: Send + Sync + 'static, L: DeviceQueue + ?Sized + 'static> Run<Q, L> {
    /// Each stage that ran gets its `retired` hook, in stage order (Java `conclude`): `committed` is true while nothing
    /// failed or stopped the quantum and the stage submitted. A hook that panics fails the quantum and ends the commits.
    fn run_retired(&self, quantum: &Q) {
        let mut committed = !self.halted() && !self.cause.is_claimed();
        for (index, state) in self.stages.iter().enumerate() {
            // SAFETY: every stage is done, and `attempted` and `submitted` are final.
            let (attempted, submitted) = unsafe { (state.attempted.get(), state.submitted.get()) };
            if !attempted {
                continue;
            }
            let stage = &self.core.stages[index];
            let hook = catch_unwind(AssertUnwindSafe(|| {
                stage.retired(quantum, committed && submitted);
            }));
            if hook.is_err() {
                committed = false;
                self.note(QuantumOutcome::Failed {
                    stage: StageId(index as u16),
                    error: StageError::Fatal("the stage's retired hook panicked".into()),
                });
            }
        }
    }
}

impl<Q: Send + Sync + 'static, L: DeviceQueue + ?Sized + 'static> GateSource for Arc<Run<Q, L>> {
    fn hold_gate(&self, stage: StageId) -> Gate {
        self.stages[usize::from(stage.0)]
            .gate
            .store(GATE_HELD, Ordering::Release);
        Gate {
            held: Some((self.clone() as Arc<dyn GateTarget>, stage)),
        }
    }

    fn hold_gate_in_run(&self, stage: StageId) -> Gate {
        // Relaxed: the spender gets the gate from this thread by some hand-over of its own (a channel, a lock), which
        // orders this store before its exchange.
        self.stages[usize::from(stage.0)]
            .gate
            .store(GATE_RUNNING, Ordering::Relaxed);
        Gate {
            held: Some((self.clone() as Arc<dyn GateTarget>, stage)),
        }
    }
}

impl<Q: Send + Sync + 'static, L: DeviceQueue + ?Sized + 'static> GateTarget for Run<Q, L> {
    fn release_gate(self: Arc<Self>, stage: StageId) {
        // A stage whose run is still going is published by that run's end, never beside it.
        if self.stages[usize::from(stage.0)].spend_gate(GATE_RELEASED) == Spend::Now {
            self.publish_or_run(stage);
        }
    }

    fn fail_gate(self: Arc<Self>, stage: StageId, error: StageError) {
        let spend = fail_gate_of(
            &self.cause,
            &self.stopped,
            &self.stages[usize::from(stage.0)],
            stage,
            error,
            self.kill_set(),
        );
        if spend == Spend::Now {
            self.finish_one();
        }
    }
}

/// Fails the quantum for a gate of `stage_run` that is spent as failed, and spends it.
///
/// The cause is noted before the gate is spent: once it is, the end of the run may drop the stage's count, and with the
/// last count the quantum retires and reads the cause. A killed quantum ends as killed, whichever way its gate is spent
/// (unless it had failed already: the first cause stands). A function of its own so that the loom model runs this code.
fn fail_gate_of(
    cause: &FirstCause<QuantumOutcome>,
    stopped: &AtomicBool,
    stage_run: &StageRun,
    stage: StageId,
    error: StageError,
    killed: bool,
) -> Spend {
    if killed {
        cause.claim(QuantumOutcome::Killed);
    }
    cause.claim(QuantumOutcome::Failed { stage, error });
    stopped.store(true, Ordering::Release);
    stage_run.spend_gate(GATE_FAILED)
}

impl<Q: Send + Sync + 'static, L: DeviceQueue + ?Sized + 'static> Upstream for Run<Q, L> {
    fn subscribe(&self, stage: StageId, node: NonNull<ExtSlot>) {
        // SAFETY: the node is a later quantum's slot, valid until its edge arrives, and in no stack.
        if let Err(why) = unsafe { self.stages[usize::from(stage.0)].waiters.push(node) } {
            self.deliver(stage, why, node);
        }
    }

    fn cover(&self, marker: Marker) {
        // A later quantum of another graph may run on another pool of lanes; its markers are no lane of this one.
        if self.is_lane_marker(marker) {
            self.record(marker);
        }
    }

    fn add_hold(&self) {
        self.holds.fetch_add(1, Ordering::Relaxed);
    }

    fn release_hold(self: Arc<Self>) {
        if self.holds.fetch_sub(1, Ordering::AcqRel) == 1 {
            self.arm_boundary();
        }
    }
}

impl<Q: Send + Sync + 'static, L: DeviceQueue + ?Sized + 'static> Downstream for Run<Q, L> {
    fn external_arrived(self: Arc<Self>, index: usize) {
        // SAFETY: the slots do not change while the quantum runs; the thread that satisfied the edge wrote the markers
        // before this call.
        let (to, kind, count) = unsafe {
            self.ext.read(|ext| {
                let slot = &ext.active[index];
                (slot.to_stage, slot.kind, slot.markers.read(Vec::len))
            })
        };
        if kind == EdgeKind::Completion && count > 0 && !self.halted() {
            // The edge's live count passes to the frames that arrive once the markers are reached.
            // SAFETY: as above.
            unsafe {
                self.ext
                    .read(|ext| ext.active[index].left.store(count, Ordering::Release))
            };
            for at in 0..count {
                // SAFETY: as above.
                let marker = unsafe {
                    self.ext
                        .read(|ext| ext.active[index].markers.read(|m| m[at]))
                };
                let frame = self
                    .core
                    .frame(&self, Kind::External(index), self.header(None, true));
                self.core.runtime.completions.after_marker(marker, frame);
            }
            return;
        }
        self.arrive(to);
        self.finish_one();
    }
}

/// A stage's completion that waits for a marker and for a host completion.
struct MarkerAndHost<L: DeviceQueue + ?Sized> {
    lanes: Arc<LanePool<L>>,
    marker: Marker,
    marker_reached: bool,
    host: Box<dyn Ready>,
    host_ready: bool,
}

impl<L: DeviceQueue + ?Sized + 'static> Ready for MarkerAndHost<L> {
    fn is_ready(&mut self) -> bool {
        if !self.marker_reached {
            self.marker_reached = self.lanes.reached(self.marker);
        }
        if !self.host_ready {
            self.host_ready = self.host.is_ready();
        }
        self.marker_reached && self.host_ready
    }
}

/// A stage's completion that waits for a lane's retirement and a host completion.
struct RetiredAndHost {
    cell: Arc<RetireCell>,
    host: Box<dyn Ready>,
    host_ready: bool,
}

impl Ready for RetiredAndHost {
    fn is_ready(&mut self) -> bool {
        if !self.host_ready {
            self.host_ready = self.host.is_ready();
        }
        self.host_ready && self.cell.arrived()
    }
}

/// A retried stage's wake-up: any lane of the pool past where it was at the retry, or the backstop's deadline.
///
/// One per stage of a run, made with the run and prepared again by each retry: the completions hold a clone of its
/// `Arc`, so a retry allocates nothing.
struct RetryWake<L: DeviceQueue + ?Sized> {
    lanes: Arc<LanePool<L>>,
    counts: Arc<RetryCounts>,
    backstop: Duration,
    /// Each lane's completed count at the retry, and the deadline. Written by the thread that retries the stage before
    /// it registers the wake; read by the completions' polls until the wake fires. The registration queue hands them
    /// over, and the next retry comes only after the frame the wake was registered with has been emitted.
    seen: Owned<Vec<u64>>,
    deadline: Owned<Instant>,
}

impl<L: DeviceQueue + ?Sized> RetryWake<L> {
    fn new(lanes: Arc<LanePool<L>>, counts: Arc<RetryCounts>, backstop: Duration) -> Self {
        let seen = Vec::with_capacity(lanes.len());
        Self {
            lanes,
            counts,
            backstop,
            seen: Owned::new(seen),
            deadline: Owned::new(Instant::now()),
        }
    }

    /// Notes where the lanes are and when the backstop falls due.
    ///
    /// # Safety
    ///
    /// The previous registration of this wake fired and its frame was emitted (or the wake was never registered), and
    /// this wake is registered after this call.
    unsafe fn prepare(&self) {
        // SAFETY: the caller's contract: no poll reads the wake, and this thread owns it until it registers it.
        unsafe {
            self.seen.write(|seen| {
                seen.clear();
                seen.extend(
                    (0..self.lanes.len()).map(|i| self.lanes.lane(LaneId(i as u16)).completed()),
                );
            });
            self.deadline.set(Instant::now() + self.backstop);
        }
    }
}

impl<L: DeviceQueue + ?Sized + 'static> Wake for RetryWake<L> {
    fn is_ready(&self) -> bool {
        // SAFETY: `prepare` wrote these before the registration, which handed them to the polling thread; only that
        // thread reads them until the wake fires, and nothing prepares it again before then.
        let progressed = unsafe {
            self.seen.read(|seen| {
                seen.iter()
                    .enumerate()
                    .any(|(i, &v)| self.lanes.lane(LaneId(i as u16)).completed() > v)
            })
        };
        if progressed {
            self.counts.progress.fetch_add(1, Ordering::Relaxed);
            return true;
        }
        // SAFETY: as above.
        if Instant::now() >= unsafe { self.deadline.get() } {
            self.counts.backstops.fetch_add(1, Ordering::Relaxed);
            return true;
        }
        false
    }
}

/// What a frame of a graph does.
#[derive(Clone, Copy, Debug)]
enum Kind {
    /// In the pool.
    Idle,
    /// Runs the stage in `execute`, hands on to its successors in `do_finally`.
    Stage(StageId),
    /// The completion of this stage's work: its completion-edge successors arrive (a stage with none only holds the
    /// quantum live until its host completion is ready).
    Completions(StageId),
    /// One marker of an external completion edge (the slot's index) was reached; the last arrives at the stage.
    External(usize),
    /// A retried stage runs again (its live count is the arrival's).
    Retry(StageId),
    /// One lane's marker of the retirement boundary was reached; the last retires the quantum.
    Boundary,
    /// The quantum's end when its home lane refused the retirement of the hooks' work, or cannot arm one: hands it and its
    /// outcome to `retire`.
    Retire,
    /// The home lane's retirement arrived: settles its status, then retires the quantum.
    Retired,
}

/// The one frame type of a graph, pooled: a stage, an arrival, a boundary or a retirement, by its [`Kind`].
struct GraphFrame<Q, L: DeviceQueue + ?Sized> {
    header: FrameHeader,
    /// The run it serves, while it is in flight.
    run: Option<Arc<Run<Q, L>>>,
    kind: Kind,
    result: Option<Result<Submitted, StageError>>,
    /// Whether the stage's run took a gate (`StageCx::hold`): its count is then the gate's.
    held: AtomicBool,
    /// A stage's scratch: the markers it ran after, and those of them on other lanes. Their capacity stays with the
    /// frame.
    inputs: Vec<Marker>,
    waits: Vec<Marker>,
    pool: Weak<Producer<Box<GraphFrame<Q, L>>, Multi>>,
}

impl<Q: Send + Sync + 'static, L: DeviceQueue + ?Sized + 'static> GraphFrame<Q, L> {
    fn new(pool: Weak<Producer<Box<GraphFrame<Q, L>>, Multi>>) -> Self {
        Self {
            header: FrameHeader::new(0),
            run: None,
            kind: Kind::Idle,
            result: None,
            held: AtomicBool::new(false),
            inputs: Vec::new(),
            waits: Vec::new(),
            pool,
        }
    }
}

impl<Q: Send + Sync + 'static, L: DeviceQueue + ?Sized + 'static> Frame for GraphFrame<Q, L> {
    fn header(&self) -> &FrameHeader {
        &self.header
    }

    fn header_mut(&mut self) -> &mut FrameHeader {
        &mut self.header
    }

    fn execute(&mut self) -> Result<(), FrameError> {
        if let (Kind::Stage(stage), Some(run)) = (self.kind, &self.run) {
            self.result = run.run_stage(stage, &mut self.inputs, &mut self.waits, &self.held);
        }
        Ok(())
    }

    fn do_finally(mut self: Box<Self>) {
        let alive = self.header.is_alive();
        let Some(run) = self.run.take() else {
            self.recycle();
            return;
        };
        let finish = match self.kind {
            Kind::Stage(stage) => {
                let held = self.held.load(Ordering::Relaxed);
                run.stage_ended(stage, self.result.take(), &self.inputs, alive, held)
            }
            Kind::Completions(stage) => {
                run.completed(stage, alive);
                true
            }
            Kind::External(index) => run.external_reached(index, alive),
            Kind::Retry(stage) => run.retry_woke(stage, alive),
            Kind::Boundary => {
                let last = run.boundary_left.fetch_sub(1, Ordering::AcqRel) == 1;
                self.recycle();
                if last {
                    run.conclude();
                }
                return;
            }
            Kind::Retire => {
                self.recycle();
                run.conclude();
                return;
            }
            Kind::Retired => {
                self.recycle();
                run.settle_boundary();
                run.conclude();
                return;
            }
            Kind::Idle => false,
        };
        self.recycle();
        if finish {
            run.finish_one();
        }
    }

    fn do_finally_with_error(mut self: Box<Self>, error: FrameError) {
        let Kind::Stage(stage) = self.kind else {
            return self.do_finally();
        };
        let Some(run) = self.run.take() else {
            return self.recycle();
        };
        let held = self.held.load(Ordering::Relaxed);
        let finish = run.stage_errored(stage, error, held);
        self.recycle();
        if finish {
            run.finish_one();
        }
    }

    /// Back to the graph's pool, serving no run.
    fn recycle(mut self: Box<Self>) {
        self.run = None;
        self.kind = Kind::Idle;
        self.result = None;
        self.held.store(false, Ordering::Relaxed);
        self.header = FrameHeader::new(0);
        if let Some(pool) = self.pool.upgrade() {
            let _ = pool.offer(self);
        }
    }
}

/// The pieces a run's phases are handed over by, under every interleaving the memory model allows: the first cause,
/// the final arrival that reads what the other predecessors wrote, and the lane table. (The run itself is not modelled:
/// its queues, `Arc`s and the completions use `std`'s atomics.)
#[cfg(all(loom, test))]
mod loom_tests {
    use super::*;
    use crate::QueueId;
    use loom::sync::Arc;
    use loom::thread;

    #[test]
    fn loom_the_first_cause_stands_whoever_races() {
        loom::model(|| {
            let cause = Arc::new(FirstCause::<u32>::new());
            let other = {
                let cause = cause.clone();
                thread::spawn(move || cause.claim(1))
            };
            let mine = cause.claim(2);
            let theirs = other.join().unwrap();
            assert_ne!(mine, theirs, "exactly one claim wins");
            // SAFETY: both claimants joined.
            let taken = unsafe { cause.take() };
            assert_eq!(taken, Some(if mine { 2 } else { 1 }));
        });
    }

    // A stage held at its gate: the gating thread writes what the stage will read and takes the gate (the gate state); a
    // releaser that spends the gate (however the gate reached it) sees the write, and a second party racing for the same
    // gate (a fail against a release) loses. The stage is published at most once, and exactly once when the gate was
    // taken before both looked.
    #[test]
    fn loom_a_held_stage_is_published_once_and_the_releaser_sees_what_the_gating_thread_wrote() {
        loom::model(|| {
            let stage = Arc::new(StageRun::new());
            let data = Arc::new(Owned::new(0u32));
            let published = Arc::new(crate::sync::AtomicUsize::new(0));
            let spend = |stage: Arc<StageRun>,
                         data: Arc<Owned<u32>>,
                         published: Arc<crate::sync::AtomicUsize>| {
                move || {
                    if stage.spend_gate(GATE_RELEASED) == Spend::Now {
                        // SAFETY: the swap made this the only party past the gate, after the write.
                        assert_eq!(unsafe { data.read(|d| *d) }, 7);
                        published.fetch_add(1, Ordering::AcqRel);
                    }
                }
            };
            let release = thread::spawn(spend(stage.clone(), data.clone(), published.clone()));
            let fail = thread::spawn(spend(stage.clone(), data.clone(), published.clone()));
            // SAFETY: no one reads the cell before the flag is set.
            unsafe { data.write(|d| *d = 7) };
            stage.gate.store(GATE_HELD, Ordering::Release);
            release.join().unwrap();
            fail.join().unwrap();
            assert!(
                published.load(Ordering::Acquire) <= 1,
                "published at most once"
            );
        });
    }

    // A gate taken in `run`: the end of the run and the spender of the gate each arrive once, in either order, and the
    // second to arrive publishes the stage (or drops its count), seeing what the first wrote before it. Exactly one of
    // them acts, so the stage is published exactly once, never beside its running `run`.
    #[test]
    fn loom_the_second_of_a_runs_end_and_its_gates_release_publishes_the_stage_and_sees_both_writes()
     {
        loom::model(|| {
            let stage = Arc::new(StageRun::new());
            let by_run = Arc::new(Owned::new(0u32));
            let by_releaser = Arc::new(Owned::new(0u32));
            stage.gate.store(GATE_RUNNING, Ordering::Relaxed);
            let releaser = {
                let (stage, by_run, by_releaser) =
                    (stage.clone(), by_run.clone(), by_releaser.clone());
                thread::spawn(move || {
                    // SAFETY: only this thread writes the cell, and the other reads it after the exchange orders it.
                    unsafe { by_releaser.write(|d| *d = 8) };
                    match stage.spend_gate(GATE_RELEASED) {
                        Spend::Now => {
                            // SAFETY: the exchange that found the stage parked follows the run's writes.
                            assert_eq!(unsafe { by_run.read(|d| *d) }, 7);
                            true
                        }
                        Spend::AtRunEnd => false,
                        Spend::Nothing => panic!("the gate was out"),
                    }
                })
            };
            // SAFETY: only this thread writes the cell, and the other reads it after the exchange orders it.
            unsafe { by_run.write(|d| *d = 7) };
            let ended = match stage.run_ended_with_gate() {
                RunEnd::Released => {
                    // SAFETY: the exchange that found the gate released follows the releaser's write.
                    assert_eq!(unsafe { by_releaser.read(|d| *d) }, 8);
                    true
                }
                RunEnd::Parked => false,
                other => panic!("a release left {other:?}"),
            };
            let released = releaser.join().unwrap();
            assert_ne!(ended, released, "exactly one of them publishes");
            assert_eq!(stage.gate.load(Ordering::Acquire), GATE_IDLE);
        });
    }

    // The same for a gate that fails, through the code `fail_gate` runs (`fail_gate_of`): the failer notes the cause
    // before it spends the gate, and the party that acts drops the stage's count, once. The end of the run that finds the
    // gate failed sees the cause claimed, whenever it sees `GATE_FAILED`.
    #[test]
    fn loom_the_second_of_a_runs_end_and_its_gates_failure_drops_the_count_once_and_sees_the_cause()
    {
        loom::model(|| {
            let stage = Arc::new(StageRun::new());
            let cause = Arc::new(FirstCause::<QuantumOutcome>::new());
            let stopped = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let drops = Arc::new(crate::sync::AtomicUsize::new(0));
            stage.gate.store(GATE_RUNNING, Ordering::Relaxed);
            let failer = {
                let (stage, cause, stopped, drops) =
                    (stage.clone(), cause.clone(), stopped.clone(), drops.clone());
                thread::spawn(move || {
                    let error = StageError::Corrupt("the artifact would not read".into());
                    if fail_gate_of(&cause, &stopped, &stage, StageId(0), error, false)
                        == Spend::Now
                    {
                        drops.fetch_add(1, Ordering::AcqRel);
                    }
                })
            };
            if stage.run_ended_with_gate() == RunEnd::Failed {
                assert!(cause.is_claimed(), "the run's end sees the cause");
                assert!(
                    stopped.load(std::sync::atomic::Ordering::Acquire),
                    "and that the quantum stopped"
                );
                drops.fetch_add(1, Ordering::AcqRel);
            }
            failer.join().unwrap();
            assert_eq!(
                drops.load(Ordering::Acquire),
                1,
                "the count is dropped once"
            );
        });
    }

    // The in-run gate is stored with Relaxed, and the gate then reaches the spender by a hand-over of the stage's own
    // (here a flag stored with Release and loaded with Acquire, as a lock or a channel would): the spender sees the gate
    // out, so exactly one of the run's end and the spender acts.
    #[test]
    fn loom_a_gate_stored_in_run_and_handed_to_another_thread_is_spent_by_exactly_one_party() {
        loom::model(|| {
            let stage = Arc::new(StageRun::new());
            let handed = Arc::new(crate::sync::AtomicBool::new(false));
            let spender = {
                let (stage, handed) = (stage.clone(), handed.clone());
                thread::spawn(move || {
                    while !handed.load(Ordering::Acquire) {
                        thread::yield_now();
                    }
                    stage.spend_gate(GATE_RELEASED)
                })
            };
            // What `hold_gate_in_run` stores, then the hand-over.
            stage.gate.store(GATE_RUNNING, Ordering::Relaxed);
            handed.store(true, Ordering::Release);
            let ended = stage.run_ended_with_gate();
            let spent = spender.join().unwrap();
            match (ended, spent) {
                (RunEnd::Parked, Spend::Now) | (RunEnd::Released, Spend::AtRunEnd) => {}
                other => panic!("a gate out in run was spent twice or not at all: {other:?}"),
            }
        });
    }

    // A gate that is spent twice (a release and a fail both reach the stage) acts once, whichever of the run's end and
    // the two spenders comes first.
    #[test]
    fn loom_a_gate_spent_twice_while_its_run_ends_acts_once() {
        loom::model(|| {
            let stage = Arc::new(StageRun::new());
            let acts = Arc::new(crate::sync::AtomicUsize::new(0));
            stage.gate.store(GATE_RUNNING, Ordering::Relaxed);
            let spenders: Vec<_> = [GATE_RELEASED, GATE_FAILED]
                .into_iter()
                .map(|to| {
                    let (stage, acts) = (stage.clone(), acts.clone());
                    thread::spawn(move || {
                        if stage.spend_gate(to) == Spend::Now {
                            acts.fetch_add(1, Ordering::AcqRel);
                        }
                    })
                })
                .collect();
            if matches!(
                stage.run_ended_with_gate(),
                RunEnd::Released | RunEnd::Failed
            ) {
                acts.fetch_add(1, Ordering::AcqRel);
            }
            for spender in spenders {
                spender.join().unwrap();
            }
            assert_eq!(acts.load(Ordering::Acquire), 1, "a gate acts once");
        });
    }

    // A join with two predecessors: each writes its output, then arrives. The arrival that finds it was the last
    // reads both outputs, whichever thread it is on.
    #[test]
    fn loom_the_last_arrival_sees_every_predecessors_markers() {
        loom::model(|| {
            let stages: Arc<[StageRun; 3]> =
                Arc::new([StageRun::new(), StageRun::new(), StageRun::new()]);
            stages[2].pending.store(2, Ordering::Relaxed);
            let queue = QueueId::next();
            let arrive = move |stages: &[StageRun; 3], which: usize| -> bool {
                // SAFETY: this thread owns predecessor `which`'s output until it arrives.
                unsafe {
                    stages[which]
                        .out
                        .write(|out| out.push(Marker::new(queue, LaneId(which as u16), 1)))
                };
                if !stages[2].arrive() {
                    return false;
                }
                for predecessor in &stages[..2] {
                    // SAFETY: the last arrival follows both predecessors' writes.
                    let seen = unsafe { predecessor.out.read(Vec::len) };
                    assert_eq!(seen, 1, "the last arrival sees every output");
                }
                true
            };
            let other = {
                let stages = stages.clone();
                thread::spawn(move || arrive(&stages, 1))
            };
            let mine = arrive(&stages, 0);
            let theirs = other.join().unwrap();
            assert_ne!(mine, theirs, "exactly one arrival is the last");
        });
    }

    // A reader that sees a lane in the table sees at least the value recorded before the bit; the latest of two
    // recorders stands.
    #[test]
    fn loom_a_reader_that_sees_a_lane_sees_its_value() {
        loom::model(|| {
            let table = Arc::new(LaneTable::new(2));
            let recorders: Vec<_> = [5u64, 7]
                .into_iter()
                .map(|value| {
                    let table = table.clone();
                    thread::spawn(move || table.record(1, value))
                })
                .collect();
            let used = table.used();
            if used & 0b10 != 0 {
                assert!(table.value(1) >= 5, "a lane in the table has its value");
            }
            for recorder in recorders {
                recorder.join().unwrap();
            }
            assert_eq!(table.used(), 0b10);
            assert_eq!(table.value(1), 7);
            table.clear();
            assert_eq!((table.used(), table.value(1)), (0, 0));
        });
    }
}

/// The port of the upstream `StageGraphTest`, `StageGraphHostStageTest` and
/// `CrossGraphEdgesTest` (the parts that apply: CUDA capture, transfer lanes, workspaces and graph
/// recycling have no counterpart here). Frames are run by hand, the way an executor runs them, from
/// the stage sink and the completions.
#[cfg(all(test, not(loom)))]
mod tests {
    use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU32, AtomicU64};
    use std::sync::{Barrier, Mutex, MutexGuard, OnceLock, PoisonError};

    use euhedral_core::generics::LatticeSource;
    use euhedral_core::ingest::{IngestSink, QueueIngestSink};

    use super::*;
    use crate::{ControlError, DeviceQueue, LaneControl, QueueId};

    fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
        mutex.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// A lane whose work completes when the test says so. With `controlled`, it also takes a retirement and a wait on
    /// another lane's marker, as a driver's stream does; the retirement arrives at [`finish`](Self::finish).
    #[derive(Debug)]
    struct HandLane {
        id: QueueId,
        submitted: AtomicU64,
        completed: AtomicU64,
        controlled: bool,
        /// The markers other lanes awaited on this one, in order.
        awaits: Mutex<Vec<Marker>>,
        /// The retirements armed and not yet arrived.
        cells: Mutex<Vec<Arc<RetireCell>>>,
        armed: AtomicU32,
        /// The status the next arrivals carry.
        status: AtomicI32,
        /// Refuses the next arming (`Busy`).
        refuse: AtomicBool,
    }

    impl Default for HandLane {
        fn default() -> Self {
            Self::with(false)
        }
    }

    impl HandLane {
        fn with(controlled: bool) -> Self {
            Self {
                id: QueueId::next(),
                submitted: AtomicU64::new(0),
                completed: AtomicU64::new(0),
                controlled,
                awaits: Mutex::default(),
                cells: Mutex::default(),
                armed: AtomicU32::new(0),
                status: AtomicI32::new(0),
                refuse: AtomicBool::new(false),
            }
        }

        fn submit(&self) -> u64 {
            self.submitted.fetch_add(1, Ordering::AcqRel) + 1
        }

        /// Completes what was submitted, and tells every retirement armed so far.
        fn finish(&self) {
            self.completed
                .store(self.submitted.load(Ordering::Acquire), Ordering::Release);
            let status = self.status.load(Ordering::Acquire);
            for cell in lock(&self.cells).drain(..) {
                cell.arrive(status);
            }
        }

        fn armed(&self) -> u32 {
            self.armed.load(Ordering::Acquire)
        }
    }

    impl DeviceQueue for HandLane {
        fn id(&self) -> QueueId {
            self.id
        }

        fn completed(&self) -> u64 {
            self.completed.load(Ordering::Acquire)
        }

        fn name(&self) -> &str {
            "hand"
        }

        fn control(&self) -> Option<&dyn LaneControl> {
            self.controlled.then_some(self as &dyn LaneControl)
        }
    }

    impl LaneControl for HandLane {
        fn await_marker(&self, marker: Marker) -> Result<(), ControlError> {
            lock(&self.awaits).push(marker);
            Ok(())
        }

        fn notify_retired(&self, cell: &Arc<RetireCell>) -> Result<(), ControlError> {
            if self.refuse.swap(false, Ordering::AcqRel) {
                return Err(ControlError::Busy);
            }
            self.armed.fetch_add(1, Ordering::AcqRel);
            lock(&self.cells).push(cell.clone());
            Ok(())
        }

        fn confirm_retired(&self, status: i32) -> Option<ControlError> {
            (status != 0).then(|| ControlError::Failed(format!("device status {status}")))
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
        /// A stage's gate was asked.
        Gated {
            q: u64,
            stage: u16,
        },
        /// A stage's `abandoned` hook ran, with this many lane joins awaited on the home lane and retirements armed there.
        AbandonedAt {
            q: u64,
            stage: u16,
            awaited: usize,
            armed: u32,
        },
        /// A stage's `retired` hook.
        Hook {
            q: u64,
            stage: u16,
            committed: bool,
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

    /// What a probe does at its gate.
    #[derive(Clone)]
    enum GateAct {
        Pass,
        /// Holds the stage: the gate goes to the list, for the test to release.
        Hold(Arc<Mutex<Vec<Gate>>>),
        /// Panics.
        Panic,
        /// Answers `false` without taking a gate.
        HoldNothing,
        /// Takes a gate and drops it unreleased.
        DropIt,
    }

    struct Probe {
        act: Act,
        gate: GateAct,
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
            let marker = cx.lane.map(|lane| {
                let hand = &self.hands[usize::from(lane.0)];
                Marker::new(hand.id, lane, hand.submit())
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

        fn pass_gate(&self, q: &u64, gate: GateRef<'_>) -> bool {
            lock(&self.1.log).push(Event::Gated {
                q: *q,
                stage: self.0,
            });
            match &self.1.gate {
                GateAct::Pass => true,
                GateAct::Hold(gates) => {
                    lock(gates).push(gate.hold());
                    false
                }
                GateAct::Panic => panic!("the gate broke"),
                GateAct::HoldNothing => false,
                GateAct::DropIt => {
                    drop(gate.hold());
                    false
                }
            }
        }

        fn abandoned(&self, q: &u64, _home: &dyn DeviceQueue) {
            lock(&self.1.log).push(Event::Abandoned {
                q: *q,
                stage: self.0,
            });
            // What the home lane had seen when the hook ran: the joins awaited, the retirements armed.
            if let Some(home) = self.1.hands.first()
                && home.controlled
            {
                lock(&self.1.log).push(Event::AbandonedAt {
                    q: *q,
                    stage: self.0,
                    awaited: lock(&home.awaits).len(),
                    armed: home.armed(),
                });
            }
        }

        fn retired(&self, q: &u64, committed: bool) {
            lock(&self.1.log).push(Event::Hook {
                q: *q,
                stage: self.0,
                committed,
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

        /// A harness with `Placement::Fixed`: every stage takes `allowed[lane_of(stage) % allowed.len()]`, whatever lane it
        /// continues.
        fn fixed(lanes: usize, lane_of: fn(u16) -> usize) -> Self {
            Self::build_with(
                lanes,
                DEFAULT_RETRY_BACKSTOP,
                crate::Placement::Fixed(lane_of),
                false,
            )
        }

        fn with_backstop(lanes: usize, retry_backstop: Duration) -> Self {
            Self::build_with(lanes, retry_backstop, crate::Placement::Random, false)
        }

        /// A harness whose lanes take retirements and waits (a lane of a driver's stream), with `lane_of` placement.
        fn controlled(lanes: usize, lane_of: fn(u16) -> usize) -> Self {
            Self::build_with(
                lanes,
                DEFAULT_RETRY_BACKSTOP,
                crate::Placement::Fixed(lane_of),
                true,
            )
        }

        fn build_with(
            lanes: usize,
            retry_backstop: Duration,
            placement: crate::Placement,
            controlled: bool,
        ) -> Self {
            let hands: Vec<Arc<HandLane>> = (0..lanes)
                .map(|_| Arc::new(HandLane::with(controlled)))
                .collect();
            // Fixed placement: the tests below name the lanes stages take (Java's `fixed` placement).
            let pool = Arc::new(
                LanePool::new(
                    hands
                        .iter()
                        .map(|h| h.clone() as Arc<dyn DeviceQueue>)
                        .collect(),
                )
                .with_placement(placement),
            );
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
            self.gated_graph(stages, edges, &[])
        }

        /// As `graph`, stage `i` passing its gate as `gates[i]` (`Pass` past the end).
        fn gated_graph(
            &self,
            stages: &[(Option<u64>, &[u16], Act)],
            edges: &[(u16, u16, EdgeKind)],
            gates: &[GateAct],
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
                    gate: gates.get(i).cloned().unwrap_or(GateAct::Pass),
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
        // 0 -> 1 -> 3 -> 4 is the longest path; 0 -> 2 -> 4 is the side branch, which only the other lane allows.
        let graph = h.graph(
            &[
                (None, both, Act::Submit),
                (None, both, Act::Submit),
                (None, &[1], Act::Submit),
                (None, both, Act::Submit),
                (None, both, Act::Submit),
            ],
            &[
                (0, 1, Submission),
                (1, 3, Submission),
                (3, 4, Submission),
                (0, 2, Submission),
                (2, 4, Submission),
            ],
        );
        let _ = graph.start(1, alive(), Vec::new());
        h.pump();
        let mut ran = h.ran(1);
        ran.sort_unstable();
        assert_eq!(ran, [0, 1, 2, 3, 4], "each stage ran once");
        let lane = |s| h.ran_with(1, s).unwrap().0.unwrap();
        assert_eq!(lane(1), lane(0), "the long chain keeps its root's lane");
        assert_eq!(lane(3), lane(0));
        assert_ne!(lane(2), lane(0), "the side branch takes the other lane");
        assert_eq!(lane(4), lane(3), "the join continues the chain's lane");
        let (_, waits) = h.ran_with(1, 4).unwrap();
        assert_eq!(waits.len(), 1, "the join waits for the side branch only");
        assert_eq!(waits[0].0, lane(2));
    }

    /// Java `fixed` placement overrides everything: a stage takes its fixed lane although it is the critical successor of
    /// a predecessor on another one.
    #[test]
    fn a_fixed_placement_overrides_the_lane_a_stage_would_continue() {
        let h = Harness::fixed(2, usize::from);
        // 0 -> 1: stage 1 would continue stage 0's lane (0), but its fixed lane is 1.
        let graph = open_graph(&h, 2, 2, &[(0, 1, Submission)]);
        let _ = graph.start(1, alive(), Vec::new());
        h.pump();
        assert_eq!(lanes_of(&h, 2), [0, 1]);
        let (_, waits) = h.ran_with(1, 1).unwrap();
        assert_eq!(waits.len(), 1, "it waits on the predecessor's marker");
    }

    /// The lane each stage of a graph ran on, after one quantum ran to its end.
    fn lanes_of(h: &Harness, stages: usize) -> Vec<u16> {
        (0..stages)
            .map(|s| h.ran_with(1, s as u16).unwrap().0.unwrap())
            .collect()
    }

    /// A graph of `stages` stages that may run on any of the harness's `lanes`.
    fn open_graph(
        h: &Harness,
        lanes: u16,
        stages: usize,
        edges: &[(u16, u16, EdgeKind)],
    ) -> Arc<StageGraph<u64>> {
        let allowed: Vec<u16> = (0..lanes).collect();
        let stages: Vec<(Option<u64>, &[u16], Act)> = (0..stages)
            .map(|_| (None, allowed.as_slice(), Act::Submit))
            .collect();
        h.graph(&stages, edges)
    }

    /// A graph whose stage `i` may run on the lanes `allowed[i]`.
    fn graph_allowed(
        h: &Harness,
        allowed: &[&[u16]],
        edges: &[(u16, u16, EdgeKind)],
    ) -> Arc<StageGraph<u64>> {
        let stages: Vec<(Option<u64>, &[u16], Act)> = allowed
            .iter()
            .map(|lanes| (None, *lanes, Act::Submit))
            .collect();
        h.graph(&stages, edges)
    }

    #[test]
    fn a_fan_out_spreads_its_branches_over_the_lanes_and_the_join_waits_for_the_other_branch() {
        let h = Harness::new(2);
        // 0 -> {1, 2} -> 3. The branches tie; the first listed, 1, continues the root's lane; 2 is allowed on the
        // other lane only.
        let graph = graph_allowed(
            &h,
            &[&[0, 1], &[0, 1], &[1], &[0, 1]],
            &[
                (0, 1, Submission),
                (0, 2, Submission),
                (1, 3, Submission),
                (2, 3, Submission),
            ],
        );
        let _ = graph.start(1, alive(), Vec::new());
        h.pump();
        let lane = lanes_of(&h, 4);
        assert_eq!(
            lane[1], lane[0],
            "the critical branch continues the root's lane"
        );
        assert_ne!(lane[2], lane[1], "the other branch takes the other lane");
        // Both branches are critical successors of the join; the one visited last (2) is continued.
        assert_eq!(lane[3], lane[2], "the join continues a branch's lane");
        let (_, waits) = h.ran_with(1, 3).unwrap();
        assert_eq!(
            waits,
            [(lane[1], 2)],
            "the join waits on the marker of the branch it did not continue"
        );
    }

    #[test]
    fn a_chain_keeps_one_lane() {
        let h = Harness::new(3);
        let graph = open_graph(
            &h,
            3,
            4,
            &[(0, 1, Submission), (1, 2, Submission), (2, 3, Submission)],
        );
        let _ = graph.start(1, alive(), Vec::new());
        h.pump();
        let lane = lanes_of(&h, 4);
        assert!(lane.iter().all(|&l| l == lane[0]), "{lane:?}");
        let (_, waits) = h.ran_with(1, 3).unwrap();
        assert!(waits.is_empty(), "one lane: no waits");
    }

    #[test]
    fn a_wider_fan_out_uses_every_lane() {
        let h = Harness::new(3);
        // The first listed branch continues the root; the others are allowed on one other lane each.
        let graph = graph_allowed(
            &h,
            &[&[0, 1, 2], &[0, 1, 2], &[1], &[2]],
            &[(0, 1, Submission), (0, 2, Submission), (0, 3, Submission)],
        );
        let _ = graph.start(1, alive(), Vec::new());
        h.pump();
        let lane = lanes_of(&h, 4);
        assert_eq!(
            lane[1], lane[0],
            "the first listed branch continues the root"
        );
        let mut branches = lane[1..].to_vec();
        branches.sort_unstable();
        assert_eq!(
            branches,
            [0, 1, 2],
            "three branches on three lanes: {lane:?}"
        );
    }

    #[test]
    fn a_longer_branch_continues_the_lane_whatever_its_place_in_the_edge_order() {
        let h = Harness::new(2);
        // 0 -> 1 (a sink, allowed on the other lane only) and 0 -> 2 -> 3: 2 has the greater height although 1 is listed
        // first.
        let graph = graph_allowed(
            &h,
            &[&[0, 1], &[1], &[0, 1], &[0, 1]],
            &[(0, 1, Submission), (0, 2, Submission), (2, 3, Submission)],
        );
        let _ = graph.start(1, alive(), Vec::new());
        h.pump();
        let lane = lanes_of(&h, 4);
        assert_eq!(lane[2], lane[0]);
        assert_eq!(lane[3], lane[0]);
        assert_ne!(lane[1], lane[0]);
    }

    /// The stage each stage continues, by `markPaths`' rule, for `edges` over `stages` stages.
    fn continues(stages: u16, edges: &[(u16, u16, EdgeKind)]) -> Vec<Option<u16>> {
        let mut spec = StageGraphSpec::new();
        for _ in 0..stages {
            spec.stage(StageSpec {
                name: "s",
                ordered: None,
                lanes: vec![LaneId(0)],
            });
        }
        for &(from, to, kind) in edges {
            spec.edge(StageId(from), StageId(to), kind);
        }
        Topology::of(&spec)
            .unwrap()
            .path_predecessor
            .iter()
            .map(|p| p.map(|p| p.0))
            .collect()
    }

    #[test]
    fn of_equal_heights_the_first_listed_successor_continues() {
        assert_eq!(
            continues(3, &[(0, 1, Submission), (0, 2, Submission)]),
            [None, Some(0), None]
        );
        assert_eq!(
            continues(3, &[(0, 2, Submission), (0, 1, Submission)]),
            [None, None, Some(0)]
        );
    }

    #[test]
    fn a_strictly_greater_height_beats_an_earlier_successor() {
        // 1 is a sink listed first; 2 leads on to 3.
        assert_eq!(
            continues(
                4,
                &[(0, 1, Submission), (0, 2, Submission), (2, 3, Submission)]
            ),
            [None, None, Some(0), Some(2)]
        );
    }

    #[test]
    fn a_join_continues_the_last_visited_of_the_predecessors_it_is_critical_for() {
        // 0 -> 1 -> 3 -> 4 and 0 -> 2 -> 4: sinks first, 4's predecessors are visited in stage
        // order (2, then 3), so 3 continues into 4 and the side branch 2 continues nothing.
        let edges = [
            (0, 1, Submission),
            (1, 3, Submission),
            (3, 4, Submission),
            (0, 2, Submission),
            (2, 4, Submission),
        ];
        assert_eq!(
            continues(5, &edges),
            [None, Some(0), None, Some(1), Some(3)]
        );
        // The same graph with the branches' stage indexes swapped visits the chain first.
        let swapped = [
            (0, 1, Submission),
            (1, 2, Submission),
            (2, 4, Submission),
            (0, 3, Submission),
            (3, 4, Submission),
        ];
        assert_eq!(
            continues(5, &swapped),
            [None, Some(0), Some(1), None, Some(3)]
        );
    }

    #[test]
    fn a_join_continues_only_the_predecessor_it_is_the_critical_successor_of() {
        // 0 -> {2, 3}, 1 -> 2; 3 -> 4 -> 5. 0's critical successor is 3 (taller), and 1's is 2,
        // so 2 continues 1; it is not the critical successor of 0.
        assert_eq!(
            continues(
                6,
                &[
                    (0, 2, Submission),
                    (0, 3, Submission),
                    (1, 2, Submission),
                    (3, 4, Submission),
                    (4, 5, Submission),
                ]
            ),
            [None, None, Some(1), Some(0), Some(3), Some(4)]
        );
    }

    #[test]
    fn a_completion_edge_never_carries_a_lane() {
        assert_eq!(
            continues(3, &[(0, 1, Completion), (1, 2, Submission)]),
            [None, None, Some(1)]
        );
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
            [1, 2, 3],
            "the stages that never submitted are abandoned, in order: the one that failed, and those behind it"
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
            [0, 1],
            "a stage that panicked never submitted: it is abandoned with its successor"
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
        // The failure stops the quantum: the failed stage, and the other branch, published but not yet run, are
        // abandoned (Java abandons every stage that never submitted).
        assert_eq!(h.abandoned(1), [1, 2, 3]);
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

    /// Each retry prepares its stage's wake-up again and registers it; it fires once, by a lane's progress or by the
    /// backstop, and `RetryCounts` say which, exactly. Nothing here depends on the clock (an hour away, or already past),
    /// so it also runs under Miri.
    #[test]
    fn a_retry_wake_is_prepared_again_by_each_retry_and_fires_once_by_progress_or_by_the_backstop()
    {
        // Progress only: the backstop is an hour away.
        let h = Harness::with_backstop(1, Duration::from_secs(3600));
        let graph = h.graph(
            &[
                (None, &[], Act::Cpu),
                (None, &[], Act::Retry(Arc::new(AtomicU32::new(3)))),
            ],
            &[],
        );
        let _ = graph.start(1, alive(), Vec::new());
        h.pump();
        assert_eq!(h.ran(1), [0, 1], "the retried stage waits");
        for round in 1..=3u64 {
            let counts = &h.runtime.retries;
            assert_eq!((counts.retries(), counts.progress()), (round, round - 1));
            h.hands[0].submit();
            h.finish_lanes();
            h.pump();
            assert_eq!(counts.progress(), round, "woken by the lane's progress");
        }
        let counts = &h.runtime.retries;
        assert_eq!(
            (counts.retries(), counts.progress(), counts.backstops()),
            (3, 3, 0)
        );
        assert_eq!(h.ran(1), [0, 1, 1, 1, 1]);
        assert_eq!(h.retired(1), [QuantumOutcome::Completed]);
        assert_eq!(h.runtime.completions.pending(), 0);

        // Backstop only: it is already due, and no lane moves.
        let h = Harness::with_backstop(1, Duration::ZERO);
        let graph = h.graph(&[(None, &[], Act::Retry(Arc::new(AtomicU32::new(3))))], &[]);
        let _ = graph.start(1, alive(), Vec::new());
        h.pump();
        let counts = &h.runtime.retries;
        assert_eq!(
            (counts.retries(), counts.progress(), counts.backstops()),
            (3, 0, 3)
        );
        assert_eq!(h.ran(1), [0, 0, 0, 0]);
        assert_eq!(h.retired(1), [QuantumOutcome::Completed]);
        assert_eq!(h.runtime.completions.pending(), 0);
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
        for round in 0..if cfg!(miri) { 2 } else { 20 } {
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

    /// Runs frames from the harness's sink on `workers` threads, and its completions, until `stop` says so.
    #[allow(clippy::disallowed_methods)] // The test's own threads run frames, as lattice workers do.
    fn drive(h: &Arc<Harness>, workers: usize, stop: impl Fn() -> bool + Send + Sync + 'static) {
        let pull = Arc::new(Mutex::new(()));
        let stop = Arc::new(stop);
        let deadline = Instant::now() + Duration::from_secs(60);
        let threads: Vec<_> = (0..workers)
            .map(|_| {
                let (h, pull, stop) = (h.clone(), pull.clone(), stop.clone());
                std::thread::spawn(move || {
                    while !stop() {
                        assert!(Instant::now() < deadline, "the workers did not finish");
                        let mut frame = None;
                        {
                            let _one = lock(&pull);
                            h.sink
                                .delegate()
                                .pull(&mut |f| frame = Some(f), &mut |_| false, 1);
                            if frame.is_none() {
                                h.runtime.completions.pull(
                                    &mut |f| frame = Some(f),
                                    &mut |_| false,
                                    1,
                                );
                            }
                        }
                        match frame {
                            Some(frame) => run_frame(frame),
                            None => {
                                h.finish_lanes();
                                std::thread::yield_now();
                            }
                        }
                    }
                })
            })
            .collect();
        for thread in threads {
            thread.join().unwrap();
        }
    }

    /// A graph over a typed pool: a stage reaches the lane it was given through its context, and a CPU stage has none.
    #[test]
    fn a_stage_reaches_its_lane_through_a_typed_pool() {
        type Seen = Arc<Mutex<Vec<(u16, Option<usize>)>>>;
        struct Submits(Seen);
        impl Stage<u64, HandLane> for Submits {
            fn run(&self, _: &u64, cx: &StageCx<'_, HandLane>) -> Result<Submitted, StageError> {
                let handle = cx.lane_handle();
                lock(&self.0).push((cx.stage.0, handle.map(|h| Arc::as_ptr(h) as usize)));
                Ok(match (cx.lane, handle) {
                    (Some(lane), Some(hand)) => {
                        Submitted::marker(Marker::new(hand.id, lane, hand.submit()))
                    }
                    _ => Submitted::done(),
                })
            }
        }
        let hands: Vec<Arc<HandLane>> = (0..2).map(|_| Arc::new(HandLane::default())).collect();
        let pool = Arc::new(LanePool::typed(hands.clone()));
        let completions = Completions::new(pool.as_queues());
        let sink = Arc::new(QueueIngestSink::new());
        sink.delegate().add_downstream(Arc::new(Unattended));
        let mut spec = StageGraphSpec::new();
        let on_lane_one = spec.stage(StageSpec {
            name: "device",
            ordered: None,
            lanes: vec![LaneId(1)],
        });
        let cpu = spec.stage(StageSpec {
            name: "cpu",
            ordered: None,
            lanes: vec![],
        });
        spec.edge(on_lane_one, cpu, Submission);
        let seen = Arc::new(Mutex::new(Vec::new()));
        let retired = Arc::new(Mutex::new(Vec::new()));
        let log = retired.clone();
        let graph = StageGraph::new(
            spec,
            vec![
                Arc::new(Submits(seen.clone())) as Arc<dyn Stage<u64, HandLane>>,
                Arc::new(Submits(seen.clone())),
            ],
            GraphRuntime {
                stages: Arc::new(QueueStages(sink.clone())),
                completions: completions.clone(),
                lanes: pool,
                retries: Arc::default(),
                retry_backstop: DEFAULT_RETRY_BACKSTOP,
            },
            move |q, outcome| lock(&log).push((q, outcome)),
        )
        .unwrap();
        let _ = graph.start(1, alive(), Vec::new());
        let pump = || loop {
            let mut frame = None;
            sink.delegate()
                .pull(&mut |f| frame = Some(f), &mut |_| false, 1);
            if frame.is_none() {
                completions.pull(&mut |f| frame = Some(f), &mut |_| false, 1);
            }
            match frame {
                Some(frame) => run_frame(frame),
                None => return,
            }
        };
        pump();
        assert_eq!(
            *lock(&seen),
            [(0, Some(Arc::as_ptr(&hands[1]) as usize)), (1, None)],
            "the stage got lane 1's own handle; the CPU stage none"
        );
        assert!(
            lock(&retired).is_empty(),
            "lane 1 has not completed its work"
        );
        hands[1].finish();
        pump();
        assert_eq!(*lock(&retired), [(1, QuantumOutcome::Completed)]);
    }

    /// Java `continuedLane`: a root continues its graph's home lane, taken in turn over the pool's lanes when the run is
    /// made. Quanta that overlap use different runs, so different homes; a run keeps its home for the quanta it serves
    /// after, as a Java graph does.
    #[test]
    fn a_root_continues_its_runs_home_lane() {
        let h = Harness::new(2);
        let graph = h.graph(
            &[(None, &[0, 1], Act::Submit), (None, &[0, 1], Act::Submit)],
            &[(0, 1, Submission)],
        );
        // Two quanta at once: two runs, homes 0 and 1.
        let _ = graph.start(1, alive(), Vec::new());
        let _ = graph.start(2, alive(), Vec::new());
        h.pump();
        h.finish_lanes();
        h.pump();
        let lane = |q, s| h.ran_with(q, s).unwrap().0.unwrap();
        assert_eq!(lane(1, 0), 0, "the first run's home");
        assert_eq!(lane(1, 1), 0, "its chain continues the root's lane");
        assert_eq!(lane(2, 0), 1, "the second run's home");
        assert_eq!(lane(2, 1), 1);
        // The runs are pooled: quanta after them reuse a run, and so its home.
        let _ = graph.start(3, alive(), Vec::new());
        h.pump();
        h.finish_lanes();
        h.pump();
        assert!(
            [0, 1].contains(&lane(3, 0)) && lane(3, 1) == lane(3, 0),
            "a pooled run keeps its home"
        );
    }

    /// Java `AbstractQuantum.failure.compareAndSet(null, cause)`: the first cause stands.
    #[test]
    fn the_first_failure_wins_among_8_failing_stages() {
        let h = Arc::new(Harness::new(1));
        let stages: Vec<(Option<u64>, &[u16], Act)> =
            (0..8).map(|_| (None, &[][..], Act::Fail)).collect();
        let graph = h.graph(&stages, &[]);
        for q in 0..if cfg!(miri) { 3 } else { 50 } {
            let _ = graph.start(q, alive(), Vec::new());
            let g = graph.clone();
            drive(&h, 4, move || g.live() == 0);
            let retired = h.retired(q);
            assert!(
                matches!(
                    &retired[..],
                    [QuantumOutcome::Failed {
                        error: StageError::Corrupt(_),
                        ..
                    }]
                ),
                "one outcome, the first failure: {retired:?}"
            );
        }
    }

    /// A failure then a kill (and a kill then a failure) keep the first cause.
    #[test]
    fn a_kill_after_a_failure_keeps_the_failure_and_a_failure_after_a_kill_keeps_the_kill() {
        let h = Harness::new(1);
        let graph = h.graph(
            &[(None, &[], Act::Fail), (None, &[], Act::Cpu)],
            &[(0, 1, Submission)],
        );
        let kill = alive();
        let _ = graph.start(1, kill.clone(), Vec::new());
        h.pump_once();
        kill.store(true, Ordering::Release);
        h.pump();
        assert!(matches!(
            &h.retired(1)[..],
            [QuantumOutcome::Failed {
                stage: StageId(0),
                ..
            }]
        ));
        let run = h.graph(
            &[(None, &[], Act::Cpu), (None, &[], Act::Fail)],
            &[(0, 1, Submission)],
        );
        let kill = alive();
        let _ = run.start(2, kill.clone(), Vec::new());
        kill.store(true, Ordering::Release);
        h.pump();
        assert_eq!(h.retired(2), [QuantumOutcome::Killed]);
    }

    /// A diamond whose branches run on other lanes, on 8 workers: the join always sees both branches' markers.
    #[test]
    fn a_diamond_on_8_workers_runs_the_join_with_every_input() {
        let h = Arc::new(Harness::new(4));
        let graph = h.graph(
            &[
                (None, &[0], Act::Submit),
                (None, &[1], Act::Submit),
                (None, &[2], Act::Submit),
                (None, &[3], Act::Submit),
            ],
            &[
                (0, 1, Submission),
                (0, 2, Submission),
                (1, 3, Submission),
                (2, 3, Submission),
            ],
        );
        let total = if cfg!(miri) { 6 } else { 300 };
        for q in 0..total {
            let _ = graph.start(q, alive(), Vec::new());
        }
        let g = graph.clone();
        drive(&h, 8, move || g.live() == 0);
        let events = h.events();
        for q in 0..total {
            let mut joins = events.iter().filter_map(|e| match e {
                Event::Ran {
                    q: eq,
                    stage: 3,
                    waits,
                    ..
                } if *eq == q => Some(waits.clone()),
                _ => None,
            });
            let mut waits = joins.next().expect("the join ran");
            assert!(joins.next().is_none(), "the join ran once");
            waits.sort_unstable();
            assert_eq!(
                waits.iter().map(|w| w.0).collect::<Vec<_>>(),
                [1, 2],
                "the join waits for both branches: {waits:?}"
            );
            assert_eq!(h.retired(q), [QuantumOutcome::Completed]);
        }
    }

    /// Java `StageFrame.externals`: an edge registered before the stage submitted, one registered while another waits,
    /// and one registered after, each arrive once with the stage's marker.
    #[test]
    fn subscriptions_before_during_and_after_publication_each_arrive_once() {
        let h = Harness::new(2);
        let graph = crossed(&h, Act::Submit);
        let edge = |from: &QuantumRef| ExternalEdge {
            from: from.clone(),
            from_stage: StageId(1),
            to_stage: StageId(0),
            kind: Submission,
        };
        let a = graph.start(1, alive(), Vec::new());
        let _ = graph.start(2, alive(), vec![edge(&a)]);
        assert!(h.pump_once(), "the earlier quantum's root");
        let _ = graph.start(3, alive(), vec![edge(&a)]);
        assert!(
            h.pump_once(),
            "the earlier quantum's stage 1 seals its stack"
        );
        let _ = graph.start(4, alive(), vec![edge(&a)]);
        drop(a);
        h.pump();
        for q in 2..=4 {
            assert_eq!(h.ran(q), [0, 1], "quantum {q}");
            assert_eq!(h.ran_with(q, 0).unwrap().1, [(0, 1)], "quantum {q}");
        }
        h.finish_lanes();
        h.pump();
        for q in 1..=4 {
            assert_eq!(h.retired(q), [QuantumOutcome::Completed], "quantum {q}");
        }
        assert_eq!(graph.live(), 0);
    }

    /// Java `CrossGraphEdgesTest`: pins are released once, whether the stage that used the quantum ran or was swept.
    #[test]
    fn external_pins_are_released_once_whether_the_stage_runs_or_is_swept() {
        let h = Harness::new(2);
        let graph = crossed(&h, Act::Submit);
        let a = graph.start(1, alive(), Vec::new());
        let edge = |from: &QuantumRef| ExternalEdge {
            from: from.clone(),
            from_stage: StageId(1),
            to_stage: StageId(0),
            kind: Submission,
        };
        let _ = graph.start(2, alive(), vec![edge(&a)]);
        let _ = graph.start(3, Arc::new(AtomicBool::new(true)), vec![edge(&a)]);
        drop(a);
        h.pump();
        h.finish_lanes();
        h.pump();
        assert_eq!(h.retired(1), [QuantumOutcome::Completed]);
        assert_eq!(h.retired(2), [QuantumOutcome::Completed]);
        assert_eq!(h.retired(3), [QuantumOutcome::Killed]);
        assert_eq!(h.abandoned(3), [0, 1]);
        assert_eq!(graph.live(), 0);
        assert_eq!(h.runtime.completions.pending(), 0);
    }

    /// Java `CrossGraphEdgesTest.aCancelledGraphBetweenTwoOthersPassesTheEarlierOnesOrderOn`: a quantum killed before
    /// its stage ran passes on the order of what its external edge named.
    #[test]
    fn a_cancelled_quantum_between_two_others_passes_the_earlier_ones_order_on() {
        let h = Harness::new(2);
        let first = h.graph(&[(None, &[0], Act::Submit)], &[]);
        let middle = h.graph(&[(None, &[1], Act::Submit)], &[]);
        let last = h.graph(&[(None, &[1], Act::Submit)], &[]);
        let a = first.start(1, alive(), Vec::new());
        let edge = |from: QuantumRef| ExternalEdge {
            from,
            from_stage: StageId(0),
            to_stage: StageId(0),
            kind: Submission,
        };
        let b = middle.start(2, Arc::new(AtomicBool::new(true)), vec![edge(a)]);
        let _ = last.start(3, alive(), vec![edge(b)]);
        h.pump();
        assert!(h.ran(2).is_empty(), "the middle quantum was killed");
        assert_eq!(
            h.ran_with(3, 0).unwrap().1,
            [(0, 1)],
            "the last follows the first through the middle"
        );
        h.finish_lanes();
        h.pump();
        assert_eq!(h.retired(2), [QuantumOutcome::Killed]);
        assert_eq!(h.retired(1), [QuantumOutcome::Completed]);
        assert_eq!(h.retired(3), [QuantumOutcome::Completed]);
    }

    /// Java `StageGraphHostStageTest.aFailedQuantumStillWaitsForADeferredStageAndNeverPublishesPastIt`.
    #[test]
    fn a_failed_quantum_still_waits_for_a_host_completion_and_publishes_nothing_past_it() {
        let h = Harness::new(1);
        let token = Token::default();
        let graph = h.graph(
            &[
                (None, &[], Act::Host(token.clone())),
                (None, &[], Act::Fail),
                (None, &[], Act::Cpu),
            ],
            &[(0, 1, Submission), (1, 2, Submission)],
        );
        let _ = graph.start(1, alive(), Vec::new());
        h.pump();
        assert_eq!(h.ran(1), [0, 1], "nothing runs past the failure");
        assert!(
            h.retired(1).is_empty(),
            "the host completion holds the retirement back"
        );
        assert_eq!(h.abandoned(1), [], "the quantum has not quiesced");
        token.0.store(true, Ordering::Release);
        h.pump();
        assert_eq!(h.abandoned(1), [1, 2]);
        assert!(matches!(
            &h.retired(1)[..],
            [QuantumOutcome::Failed {
                stage: StageId(1),
                ..
            }]
        ));
        assert_eq!(graph.live(), 0);
    }

    /// The hooks a quantum ran, as (stage, committed), in order.
    fn hooks(h: &Harness, q: u64) -> Vec<(u16, bool)> {
        h.events()
            .into_iter()
            .filter_map(|e| match e {
                Event::Hook {
                    q: eq,
                    stage,
                    committed,
                } if eq == q => Some((stage, committed)),
                _ => None,
            })
            .collect()
    }

    /// Java `quiesce`: every other lane the quantum used joins the home lane on the device, and one retirement is armed
    /// on the home lane, which ends the quantum.
    #[test]
    fn a_quantum_joins_its_lanes_to_the_home_lane_and_retires_on_one_retirement() {
        let h = Harness::controlled(2, |s| usize::from(s).saturating_sub(1));
        // Stage 0 continues the home lane (0); stage 1 is the other branch, on lane 1.
        let graph = h.graph(
            &[
                (None, &[0, 1], Act::Submit),
                (None, &[0, 1], Act::Submit),
                (None, &[0, 1], Act::Submit),
            ],
            &[(0, 1, Submission), (0, 2, Submission)],
        );
        let _ = graph.start(1, alive(), Vec::new());
        h.pump();
        let lanes = [h.ran_with(1, 1).unwrap().0, h.ran_with(1, 2).unwrap().0];
        assert_ne!(lanes[0], lanes[1], "the branches used both lanes");
        let [home, other] = [&h.hands[0], &h.hands[1]];
        assert_eq!(home.armed(), 1, "one retirement, armed on the home lane");
        assert_eq!(other.armed(), 0, "none on the other lane");
        let awaited = lock(&home.awaits).clone();
        assert_eq!(
            awaited.len(),
            1,
            "the home lane awaited the other lane once"
        );
        assert_eq!((awaited[0].queue, awaited[0].lane), (other.id, LaneId(1)));
        assert_eq!(awaited[0].value, other.submitted.load(Ordering::Acquire));
        assert!(h.retired(1).is_empty(), "the retirement has not arrived");
        assert_eq!(
            hooks(&h, 1),
            [],
            "no hook runs before the device work ended"
        );
        h.finish_lanes();
        h.pump();
        assert_eq!(h.retired(1), [QuantumOutcome::Completed]);
        assert_eq!(
            hooks(&h, 1),
            [(0, true), (1, true), (2, true)],
            "each stage that ran, in stage order, committed"
        );
        let events = h.events();
        let hook = events.iter().position(|e| matches!(e, Event::Hook { .. }));
        let retired = events
            .iter()
            .position(|e| matches!(e, Event::Retired { .. }));
        assert!(
            hook < retired,
            "the hooks run before the quantum is handed on"
        );
        assert_eq!(graph.live(), 0);
    }

    #[test]
    fn a_device_failure_on_the_retirement_fails_the_quantum_and_its_hooks_are_not_committed() {
        let h = Harness::controlled(1, usize::from);
        let graph = h.graph(
            &[(None, &[0], Act::Submit), (None, &[0], Act::Submit)],
            &[(0, 1, Submission)],
        );
        h.hands[0].status.store(7, Ordering::Release);
        let _ = graph.start(1, alive(), Vec::new());
        h.pump();
        assert!(h.retired(1).is_empty());
        h.finish_lanes();
        h.pump();
        // The failure is the device's: it names no stage (Java `retire(ticket)` fails the quantum without one).
        assert_eq!(
            h.retired(1),
            [QuantumOutcome::DeviceFailed {
                cause: "device status 7".into()
            }]
        );
        assert_eq!(hooks(&h, 1), [(0, false), (1, false)]);
        assert_eq!(graph.live(), 0);
    }

    #[test]
    fn a_hook_runs_for_each_stage_that_was_attempted_and_for_no_other() {
        let h = Harness::controlled(1, usize::from);
        let graph = h.graph(
            &[
                (None, &[0], Act::Submit),
                (None, &[0], Act::Fail),
                (None, &[0], Act::Submit),
            ],
            &[(0, 1, Submission), (1, 2, Submission)],
        );
        let _ = graph.start(1, alive(), Vec::new());
        h.pump();
        h.finish_lanes();
        h.pump();
        assert_eq!(
            hooks(&h, 1),
            [(0, false), (1, false)],
            "the failed stage was attempted; the one behind it never ran; nothing is committed"
        );
        assert_eq!(h.abandoned(1), [1, 2]);
    }

    #[test]
    fn a_quantum_that_ran_nothing_on_a_lane_arms_no_retirement() {
        let h = Harness::controlled(1, usize::from);
        let graph = h.graph(&[(None, &[], Act::Cpu)], &[]);
        let _ = graph.start(1, alive(), Vec::new());
        h.pump();
        assert_eq!(h.hands[0].armed(), 0);
        assert_eq!(h.retired(1), [QuantumOutcome::Completed]);
        assert_eq!(hooks(&h, 1), [(0, true)]);
    }

    #[test]
    fn a_refused_retirement_falls_back_to_waiting_for_each_lanes_marker() {
        let h = Harness::controlled(1, usize::from);
        let graph = h.graph(&[(None, &[0], Act::Submit)], &[]);
        h.hands[0].refuse.store(true, Ordering::Release);
        let _ = graph.start(1, alive(), Vec::new());
        h.pump();
        assert_eq!(h.hands[0].armed(), 0, "the arming was refused");
        assert!(
            h.retired(1).is_empty(),
            "the lane's marker has not completed"
        );
        h.finish_lanes();
        h.pump();
        assert_eq!(h.retired(1), [QuantumOutcome::Completed]);
        assert_eq!(hooks(&h, 1), [(0, true)]);
    }

    /// Java `RetiredEdge`: a completion edge arms a retirement on the lane of the stage's work, and its successor runs
    /// only once that retirement arrived.
    #[test]
    fn a_completion_edge_waits_for_a_retirement_armed_on_the_producers_lane() {
        let h = Harness::controlled(1, usize::from);
        let graph = h.graph(
            &[(None, &[0], Act::Submit), (None, &[0], Act::Submit)],
            &[(0, 1, Completion)],
        );
        let _ = graph.start(1, alive(), Vec::new());
        h.pump();
        assert_eq!(h.ran(1), [0], "the successor waits for the retirement");
        assert_eq!(h.hands[0].armed(), 1, "armed on the producer's lane");
        h.finish_lanes();
        h.pump();
        assert_eq!(h.ran(1), [0, 1]);
        h.finish_lanes();
        h.pump();
        assert_eq!(h.retired(1), [QuantumOutcome::Completed]);
    }

    #[test]
    fn a_device_failure_on_a_completion_edge_fails_the_quantum_and_releases_no_successor() {
        let h = Harness::controlled(1, usize::from);
        let graph = h.graph(
            &[(None, &[0], Act::Submit), (None, &[0], Act::Submit)],
            &[(0, 1, Completion)],
        );
        h.hands[0].status.store(3, Ordering::Release);
        let _ = graph.start(1, alive(), Vec::new());
        h.pump();
        h.finish_lanes();
        h.pump();
        h.finish_lanes();
        h.pump();
        assert_eq!(h.ran(1), [0], "the successor never ran");
        assert!(matches!(
            &h.retired(1)[..],
            [QuantumOutcome::Failed {
                stage: StageId(0),
                ..
            }]
        ));
    }

    /// The gates a test took, in the order the stages asked.
    fn gates() -> Arc<Mutex<Vec<Gate>>> {
        Arc::default()
    }

    fn gated(h: &Harness, q: u64) -> Vec<u16> {
        h.events()
            .into_iter()
            .filter_map(|e| match e {
                Event::Gated { q: eq, stage } if eq == q => Some(stage),
                _ => None,
            })
            .collect()
    }

    /// Java `passGate`/`releaseGate`: a stage whose gate holds it is published when the gate is released, from any thread,
    /// and not before; the held stage keeps the quantum live.
    #[test]
    fn a_stage_held_at_its_gate_runs_when_the_gate_is_released_from_another_thread() {
        let h = Harness::new(1);
        let held = gates();
        let graph = h.gated_graph(
            &[
                (None, &[], Act::Cpu),
                (None, &[], Act::Cpu),
                (None, &[], Act::Cpu),
            ],
            &[(0, 1, Submission), (1, 2, Submission)],
            &[GateAct::Pass, GateAct::Hold(held.clone()), GateAct::Pass],
        );
        let _ = graph.start(1, alive(), Vec::new());
        h.pump();
        assert_eq!(h.ran(1), [0], "the held stage does not run");
        assert_eq!(
            gated(&h, 1),
            [0, 1],
            "every stage is asked once, in its turn"
        );
        assert_eq!(graph.live(), 1, "the held stage keeps the quantum live");
        assert!(h.retired(1).is_empty());
        let gate = lock(&held).pop().expect("a gate was taken");
        std::thread::spawn(move || gate.release()).join().unwrap();
        h.pump();
        assert_eq!(
            h.ran(1),
            [0, 1, 2],
            "released: the stage and its successor ran"
        );
        assert_eq!(h.retired(1), [QuantumOutcome::Completed]);
        assert_eq!(graph.live(), 0);
    }

    /// A quantum with several held stages: they are released from several threads at once; the join behind them runs
    /// exactly once, after all.
    #[test]
    fn gates_released_from_four_threads_at_once_publish_each_stage_and_the_join_once() {
        let h = Harness::new(1);
        let held = gates();
        let hold = || GateAct::Hold(held.clone());
        let graph = h.gated_graph(
            &[
                (None, &[], Act::Cpu),
                (None, &[], Act::Cpu),
                (None, &[], Act::Cpu),
                (None, &[], Act::Cpu),
                (None, &[], Act::Cpu),
                (None, &[], Act::Cpu),
            ],
            &[
                (0, 1, Submission),
                (0, 2, Submission),
                (0, 3, Submission),
                (0, 4, Submission),
                (1, 5, Submission),
                (2, 5, Submission),
                (3, 5, Submission),
                (4, 5, Submission),
            ],
            &[GateAct::Pass, hold(), hold(), hold(), hold(), GateAct::Pass],
        );
        let _ = graph.start(1, alive(), Vec::new());
        h.pump();
        let taken: Vec<Gate> = lock(&held).drain(..).collect();
        assert_eq!(taken.len(), 4);
        let released = Arc::new(Barrier::new(taken.len()));
        let threads: Vec<_> = taken
            .into_iter()
            .map(|gate| {
                let released = released.clone();
                std::thread::spawn(move || {
                    released.wait();
                    gate.release();
                })
            })
            .collect();
        for thread in threads {
            thread.join().unwrap();
        }
        h.pump();
        let mut ran = h.ran(1);
        ran.sort_unstable();
        assert_eq!(
            ran,
            [0, 1, 2, 3, 4, 5],
            "each stage ran once, the join once"
        );
        assert_eq!(h.retired(1), [QuantumOutcome::Completed]);
    }

    /// A gate that cannot be released because what the stage waits for failed: the quantum fails, the count is dropped,
    /// and the stage and what is behind it are abandoned.
    #[test]
    fn a_gate_that_fails_fails_the_quantum_and_drops_the_count() {
        let h = Harness::new(1);
        let held = gates();
        let graph = h.gated_graph(
            &[
                (None, &[], Act::Cpu),
                (None, &[], Act::Cpu),
                (None, &[], Act::Cpu),
            ],
            &[(0, 1, Submission), (1, 2, Submission)],
            &[GateAct::Pass, GateAct::Hold(held.clone())],
        );
        let _ = graph.start(1, alive(), Vec::new());
        h.pump();
        let gate = lock(&held).pop().unwrap();
        gate.fail(StageError::Corrupt("the artifact would not read".into()));
        h.pump();
        assert_eq!(h.ran(1), [0], "the held stage never ran");
        assert!(matches!(
            &h.retired(1)[..],
            [QuantumOutcome::Failed { stage: StageId(1), error }] if *error == StageError::Corrupt("the artifact would not read".into())
        ));
        assert_eq!(h.abandoned(1), [1, 2]);
        assert_eq!(graph.live(), 0);
    }

    #[test]
    fn a_gate_that_panics_fails_the_quantum_and_the_stage_never_runs() {
        let h = Harness::new(1);
        let graph = h.gated_graph(
            &[(None, &[], Act::Cpu), (None, &[], Act::Cpu)],
            &[(0, 1, Submission)],
            &[GateAct::Pass, GateAct::Panic],
        );
        let _ = graph.start(1, alive(), Vec::new());
        h.pump();
        assert_eq!(h.ran(1), [0]);
        assert!(matches!(
            &h.retired(1)[..],
            [QuantumOutcome::Failed {
                stage: StageId(1),
                ..
            }]
        ));
        assert_eq!(graph.live(), 0);
    }

    /// A stage that answers `false` without taking a gate, or drops the gate it took, would keep the quantum live for ever:
    /// the quantum fails instead.
    #[test]
    fn a_stage_held_with_no_gate_to_release_fails_the_quantum_instead_of_hanging_it() {
        for act in [GateAct::HoldNothing, GateAct::DropIt] {
            let h = Harness::new(1);
            let graph = h.gated_graph(
                &[(None, &[], Act::Cpu), (None, &[], Act::Cpu)],
                &[(0, 1, Submission)],
                &[GateAct::Pass, act],
            );
            let _ = graph.start(1, alive(), Vec::new());
            h.pump();
            assert_eq!(h.ran(1), [0]);
            assert!(
                matches!(
                    &h.retired(1)[..],
                    [QuantumOutcome::Failed {
                        stage: StageId(1),
                        ..
                    }]
                ),
                "{:?}",
                h.retired(1)
            );
            assert_eq!(graph.live(), 0);
        }
    }

    /// A gate released after the quantum was killed publishes the stage all the same: it ends without running, and the
    /// quantum retires.
    #[test]
    fn a_gate_released_after_a_kill_ends_the_stage_without_running_it() {
        let h = Harness::new(1);
        let held = gates();
        let graph = h.gated_graph(
            &[(None, &[], Act::Cpu), (None, &[], Act::Cpu)],
            &[(0, 1, Submission)],
            &[GateAct::Pass, GateAct::Hold(held.clone())],
        );
        let kill = alive();
        let _ = graph.start(1, kill.clone(), Vec::new());
        h.pump();
        kill.store(true, Ordering::Release);
        lock(&held).pop().unwrap().release();
        h.pump();
        assert_eq!(h.ran(1), [0]);
        assert_eq!(h.retired(1), [QuantumOutcome::Killed]);
        assert_eq!(graph.live(), 0);
    }

    /// Java `abandoned(home)`: the stages that never submitted are abandoned after the quantum's other lanes joined the
    /// home lane, and before the retirement is armed on it, so what the hook submits to home is ordered after the quantum\'s
    /// device work and covered by its retirement.
    #[test]
    fn a_stage_is_abandoned_after_the_lanes_joined_home_and_before_the_retirement_is_armed() {
        let h = Harness::controlled(2, usize::from);
        // 0 (home lane) -> 1 (the other lane, submits) and 0 -> 2 (home lane, fails: it never submits).
        let graph = h.graph(
            &[
                (None, &[0, 1], Act::Submit),
                (None, &[0, 1], Act::Submit),
                (None, &[0], Act::Fail),
            ],
            &[(0, 1, Submission), (0, 2, Submission)],
        );
        let _ = graph.start(1, alive(), Vec::new());
        h.pump();
        let at: Vec<(u16, usize, u32)> = h
            .events()
            .into_iter()
            .filter_map(|e| match e {
                Event::AbandonedAt {
                    stage,
                    awaited,
                    armed,
                    ..
                } => Some((stage, awaited, armed)),
                _ => None,
            })
            .collect();
        assert_eq!(
            at,
            [(2, 1, 0)],
            "the failed stage was abandoned once, after the join (one await on home) and before the retirement"
        );
        assert_eq!(h.hands[0].armed(), 1, "and the retirement was armed after");
        h.finish_lanes();
        h.pump();
        assert!(matches!(
            &h.retired(1)[..],
            [QuantumOutcome::Failed {
                stage: StageId(2),
                ..
            }]
        ));
    }

    /// Each stage that never submitted is abandoned exactly once, in stage order, however it ended: killed before it ran,
    /// failed, behind a failure.
    #[test]
    fn every_stage_that_never_submitted_is_abandoned_once_in_stage_order() {
        let h = Harness::controlled(1, usize::from);
        let graph = h.graph(
            &[
                (None, &[0], Act::Submit),
                (None, &[0], Act::Fail),
                (None, &[0], Act::Submit),
                (None, &[], Act::Cpu),
            ],
            &[(0, 1, Submission), (1, 2, Submission), (1, 3, Submission)],
        );
        let _ = graph.start(1, alive(), Vec::new());
        h.pump();
        h.finish_lanes();
        h.pump();
        assert_eq!(h.abandoned(1), [1, 2, 3]);
    }

    /// A stage whose `pass_gate` is a closure; it runs as a CPU stage and logs `Ran`.
    struct GateFn<F: Fn(GateRef<'_>) -> bool + Send + Sync>(Log, F);

    impl<F: Fn(GateRef<'_>) -> bool + Send + Sync> Stage<u64> for GateFn<F> {
        fn run(&self, q: &u64, cx: &StageCx<'_>) -> Result<Submitted, StageError> {
            lock(&self.0).push(Event::Ran {
                q: *q,
                stage: cx.stage.0,
                lane: None,
                waits: Vec::new(),
            });
            Ok(Submitted::done())
        }

        fn pass_gate(&self, _q: &u64, gate: GateRef<'_>) -> bool {
            (self.1)(gate)
        }
    }

    fn one_stage_graph(
        h: &Harness,
        stage: Arc<dyn Stage<u64>>,
        ordered: Option<u64>,
    ) -> Arc<StageGraph<u64>> {
        let mut spec = StageGraphSpec::new();
        spec.stage(StageSpec {
            name: "g",
            ordered,
            lanes: vec![],
        });
        let log = h.log.clone();
        StageGraph::new(spec, vec![stage], h.runtime.clone(), move |q, outcome| {
            lock(&log).push(Event::Retired { q, outcome });
        })
        .unwrap()
    }

    /// A `pass_gate` that panics after it took (and kept) a gate fails the quantum all the same.
    #[test]
    fn a_pass_gate_that_panics_after_taking_a_gate_fails_the_quantum() {
        let h = Harness::new(1);
        let kept: Arc<Mutex<Vec<Gate>>> = Arc::default();
        let k = kept.clone();
        let stage = Arc::new(GateFn(h.log.clone(), move |gate: GateRef<'_>| {
            lock(&k).push(gate.hold());
            panic!("the gate broke after it was taken")
        }));
        let graph = one_stage_graph(&h, stage, None);
        let _ = graph.start(1, alive(), Vec::new());
        h.pump();
        lock(&kept).pop().unwrap().release();
        h.pump();
        assert!(
            matches!(&h.retired(1)[..], [QuantumOutcome::Failed { .. }]),
            "retired {:?}, ran {:?}",
            h.retired(1),
            h.ran(1)
        );
        assert_eq!(graph.live(), 0);
    }

    /// Java `quiesce` arms `home.notifyRetired` after `abandonUnsubmitted(home)`: a quantum that used no lane (killed
    /// before any stage ran) still has what its hooks submitted to home covered by a retirement.
    #[test]
    fn a_killed_quantum_that_used_no_lane_arms_a_retirement_after_its_abandoned_hooks() {
        let h = Harness::controlled(1, usize::from);
        let graph = h.graph(
            &[(None, &[0], Act::Submit), (None, &[0], Act::Submit)],
            &[(0, 1, Submission)],
        );
        let _ = graph.start(1, Arc::new(AtomicBool::new(true)), Vec::new());
        h.pump();
        assert_eq!(h.abandoned(1), [0, 1]);
        assert_eq!(
            h.hands[0].armed(),
            1,
            "a retirement on home covers the abandoned hooks' work (retired: {:?})",
            h.retired(1)
        );
        assert!(h.retired(1).is_empty(), "retired only when it arrives");
        h.finish_lanes();
        h.pump();
        assert_eq!(h.retired(1), [QuantumOutcome::Killed]);
    }

    /// A quantum that used no lane but ran hooks, whose retirement the home lane refuses: it still retires (it is not left
    /// waiting on a retirement that was never armed), and nothing was armed.
    #[test]
    fn a_refused_retirement_for_the_hooks_of_a_quantum_with_no_lane_still_retires_it() {
        let h = Harness::controlled(1, usize::from);
        let graph = h.graph(&[(None, &[0], Act::Submit)], &[]);
        h.hands[0].refuse.store(true, Ordering::Release);
        let _ = graph.start(1, Arc::new(AtomicBool::new(true)), Vec::new());
        h.pump();
        assert_eq!(h.abandoned(1), [0]);
        assert_eq!(h.hands[0].armed(), 0, "the arming was refused");
        assert_eq!(
            h.retired(1),
            [QuantumOutcome::Killed],
            "the first cause stands"
        );
        assert_eq!(graph.live(), 0);
    }

    /// A quantum with nothing on a lane and no hook has nothing for a retirement to cover.
    #[test]
    fn a_quantum_that_used_no_lane_and_abandoned_nothing_arms_no_retirement() {
        let h = Harness::controlled(1, usize::from);
        let graph = h.graph(&[(None, &[], Act::Cpu)], &[]);
        let _ = graph.start(1, alive(), Vec::new());
        h.pump();
        assert_eq!(h.hands[0].armed(), 0);
        assert_eq!(h.retired(1), [QuantumOutcome::Completed]);
    }

    /// Both kinds of home lane: one that arms retirements, and a query-only one. A quantum with no lane in it behaves the
    /// same on either.
    fn both_kinds_of_lane() -> [Harness; 2] {
        [Harness::controlled(1, usize::from), Harness::new(1)]
    }

    /// The conclude rule, for a quantum that touched no lane: it arms no retirement, and it concludes in the frame that
    /// ends its last stage, so its stages are all the frames it ever runs.
    #[test]
    fn a_quantum_that_used_no_lane_concludes_in_the_frame_of_its_last_stage() {
        for h in both_kinds_of_lane() {
            let graph = h.graph(
                &[
                    (None, &[], Act::Cpu),
                    (None, &[], Act::Cpu),
                    (None, &[], Act::Cpu),
                ],
                &[(0, 1, Submission), (1, 2, Submission)],
            );
            let _ = graph.start(1, alive(), Vec::new());
            for stage in 0..3 {
                assert!(h.retired(1).is_empty(), "stage {stage} has not run");
                assert!(h.pump_once(), "the frame of stage {stage}");
            }
            assert_eq!(
                h.retired(1),
                [QuantumOutcome::Completed],
                "concluded in the frame of the last stage"
            );
            assert_eq!(h.ran(1), [0, 1, 2]);
            assert_eq!(hooks(&h, 1), [(0, true), (1, true), (2, true)]);
            assert!(!h.pump_once(), "no frame is left: the quantum ran 3");
            assert_eq!(h.hands[0].armed(), 0, "no retirement was armed");
            assert_eq!(graph.live(), 0);
        }
    }

    /// A quantum whose last stage hands back a host completion concludes in the frame of that completion.
    #[test]
    fn a_quantum_that_used_no_lane_concludes_in_the_frame_of_its_host_completion() {
        for h in both_kinds_of_lane() {
            let token = Token::default();
            let graph = h.graph(&[(None, &[], Act::Host(token.clone()))], &[]);
            let _ = graph.start(1, alive(), Vec::new());
            assert_eq!(h.pump(), 1, "the stage's frame");
            assert!(h.retired(1).is_empty(), "the completion holds it live");
            token.0.store(true, Ordering::Release);
            assert_eq!(h.pump(), 1, "the completion's frame is the last");
            assert_eq!(h.retired(1), [QuantumOutcome::Completed]);
            assert_eq!(h.hands[0].armed(), 0);
        }
    }

    /// A killed quantum whose stages name no lane: its abandon hooks run in the frame that ends it, which concludes it.
    #[test]
    fn a_killed_quantum_that_used_no_lane_and_names_none_concludes_in_its_abandon_frame() {
        for h in both_kinds_of_lane() {
            let graph = h.graph(
                &[(None, &[], Act::Cpu), (None, &[], Act::Cpu)],
                &[(0, 1, Submission)],
            );
            // A refusal waiting on the home lane is never used: nothing is armed.
            h.hands[0].refuse.store(true, Ordering::Release);
            let _ = graph.start(1, Arc::new(AtomicBool::new(true)), Vec::new());
            assert_eq!(h.pump(), 1, "the root's frame ends the quantum");
            assert_eq!(h.abandoned(1), [0, 1]);
            assert_eq!(h.retired(1), [QuantumOutcome::Killed]);
            assert_eq!(h.hands[0].armed(), 0);
            assert!(
                h.hands[0].refuse.load(Ordering::Acquire),
                "no arming was attempted"
            );
            assert_eq!(hooks(&h, 1), Vec::<(u16, bool)>::new(), "nothing ran");
            assert_eq!(graph.live(), 0);
        }
    }

    /// A quantum whose lane-less stage failed: the hooks run in the failing stage's frame, which concludes it.
    #[test]
    fn a_quantum_that_used_no_lane_and_failed_concludes_in_the_frame_of_the_failing_stage() {
        for h in both_kinds_of_lane() {
            let graph = h.graph(
                &[(None, &[], Act::Fail), (None, &[], Act::Cpu)],
                &[(0, 1, Submission)],
            );
            let _ = graph.start(1, alive(), Vec::new());
            assert_eq!(h.pump(), 1);
            assert_eq!(h.abandoned(1), [0, 1]);
            assert!(
                matches!(
                    h.retired(1).as_slice(),
                    [QuantumOutcome::Failed {
                        stage: StageId(0),
                        ..
                    }]
                ),
                "retired once, failed at stage 0: {:?}",
                h.retired(1)
            );
            assert_eq!(hooks(&h, 1), [(0, false)]);
            assert_eq!(h.hands[0].armed(), 0);
        }
    }

    /// Quantum after quantum, run to its end, killed and failed alike: each costs the frames of its stages, and arms
    /// nothing.
    #[test]
    fn a_quantum_that_used_no_lane_costs_the_frames_of_its_stages_every_time() {
        for h in both_kinds_of_lane() {
            let graph = h.graph(&[(None, &[], Act::Cpu)], &[]);
            let failing = h.graph(&[(None, &[], Act::Fail)], &[]);
            for q in 0..if cfg!(miri) { 3 } else { 30 } {
                let _ = graph.start(3 * q, alive(), Vec::new());
                assert_eq!(h.pump(), 1, "run, quantum {q}");
                let _ = graph.start(3 * q + 1, Arc::new(AtomicBool::new(true)), Vec::new());
                assert_eq!(h.pump(), 1, "kill, quantum {q}");
                let _ = failing.start(3 * q + 2, alive(), Vec::new());
                assert_eq!(h.pump(), 1, "abandon, quantum {q}");
                for n in 0..3 {
                    assert_eq!(h.retired(3 * q + n).len(), 1, "quantum {}", 3 * q + n);
                }
            }
            assert_eq!(h.hands[0].armed(), 0);
            assert_eq!((graph.live(), failing.live()), (0, 0));
        }
    }

    /// A reference to a quantum that ended keeps it open: it concludes when the last reference is dropped, on the
    /// dropping thread and with no frame of its own.
    #[test]
    fn a_quantum_that_used_no_lane_concludes_when_its_last_reference_is_dropped() {
        for h in both_kinds_of_lane() {
            let graph = h.graph(&[(None, &[], Act::Cpu)], &[]);
            let held = graph.start(1, alive(), Vec::new());
            assert_eq!(h.pump(), 1);
            assert!(h.retired(1).is_empty(), "the reference holds it");
            drop(held);
            assert_eq!(h.retired(1), [QuantumOutcome::Completed], "at the drop");
            assert!(!h.pump_once(), "no frame was offered for it");
            assert_eq!(h.hands[0].armed(), 0);
        }
    }

    /// A later quantum's stage that submitted on a lane after this one ran covers its marker with this quantum's
    /// boundary: that quantum touched a lane, and keeps its retirement (and one that touched none still concludes at its
    /// last frame, side by side).
    #[test]
    fn a_quantum_that_used_no_lane_until_a_later_stage_covered_it_keeps_its_retirement() {
        let h = Harness::controlled(1, usize::from);
        let host = h.graph(&[(None, &[], Act::Cpu)], &[]);
        let device = h.graph(&[(None, &[0], Act::Submit)], &[]);
        let _ = host.start(9, alive(), Vec::new());
        assert_eq!(h.pump(), 1);
        assert_eq!(h.retired(9), [QuantumOutcome::Completed], "touched no lane");
        let first = host.start(1, alive(), Vec::new());
        assert_eq!(h.pump(), 1);
        let edge = ExternalEdge {
            from: first.clone(),
            from_stage: StageId(0),
            to_stage: StageId(0),
            kind: Submission,
        };
        drop(first);
        let _ = device.start(2, alive(), vec![edge]);
        h.pump();
        assert_eq!(h.hands[0].armed(), 2, "one retirement for each quantum");
        assert!(h.retired(1).is_empty(), "the retirement has not arrived");
        h.finish_lanes();
        h.pump();
        assert_eq!(h.retired(1), [QuantumOutcome::Completed]);
        assert_eq!(h.retired(2), [QuantumOutcome::Completed]);
    }

    /// One stage that names no lane and one that names a lane, both unsubmitted: the second's hook may have enqueued on
    /// the home lane, so the quantum keeps its retirement.
    #[test]
    fn a_killed_quantum_that_used_no_lane_but_names_one_keeps_its_retirement() {
        let h = Harness::controlled(1, usize::from);
        let host = h.graph(&[(None, &[], Act::Cpu)], &[]);
        let _ = host.start(9, alive(), Vec::new());
        assert_eq!(h.pump(), 1);
        assert_eq!(h.retired(9), [QuantumOutcome::Completed], "touched no lane");
        let graph = h.graph(
            &[(None, &[], Act::Cpu), (None, &[0], Act::Submit)],
            &[(0, 1, Submission)],
        );
        let kill = Arc::new(AtomicBool::new(false));
        let _ = graph.start(1, kill.clone(), Vec::new());
        assert!(h.pump_once(), "the first stage runs");
        kill.store(true, Ordering::Release);
        h.pump();
        assert_eq!(h.abandoned(1), [1]);
        assert_eq!(h.hands[0].armed(), 1, "a retirement covers the hook");
        assert!(h.retired(1).is_empty(), "retired only when it arrives");
        h.finish_lanes();
        h.pump();
        assert_eq!(h.retired(1), [QuantumOutcome::Killed]);
    }

    /// Java runs `abandonUnsubmitted(home)` before it sweeps the externals, so a later quantum waiting on an abandoned stage
    /// runs after the hook (and what the hook released).
    #[test]
    fn a_later_quantum_on_an_abandoned_stage_runs_after_the_abandoned_hook() {
        let h = Harness::controlled(1, usize::from);
        let graph = h.graph(&[(None, &[0], Act::Submit)], &[]);
        let a = graph.start(1, Arc::new(AtomicBool::new(true)), Vec::new());
        let edge = ExternalEdge {
            from: a.clone(),
            from_stage: StageId(0),
            to_stage: StageId(0),
            kind: Submission,
        };
        drop(a);
        let _ = graph.start(2, alive(), vec![edge]);
        h.pump();
        let events = h.events();
        let abandoned = events
            .iter()
            .position(|e| matches!(e, Event::Abandoned { q: 1, .. }));
        let ran = events
            .iter()
            .position(|e| matches!(e, Event::Ran { q: 2, .. }));
        assert!(abandoned.is_some() && ran.is_some(), "{events:?}");
        assert!(
            abandoned < ran,
            "q1's abandoned hook ran before q2's stage: {events:?}"
        );
    }

    /// A sink that takes frames until told to refuse them.
    struct Refuses {
        inner: Arc<QueueIngestSink>,
        refuse: AtomicBool,
    }

    impl FrameSink for Refuses {
        fn offer(&self, frame: Box<dyn Frame>) -> Result<(), Box<dyn Frame>> {
            if self.refuse.load(Ordering::Acquire) {
                Err(frame)
            } else {
                self.inner.offer(frame)
            }
        }
    }

    /// Java `publishOrRun`: a lake that refuses the stage a gate held has it run right there, so the held count is dropped
    /// and the stage still does its work.
    #[test]
    fn a_stage_released_into_a_sink_that_refuses_it_runs_where_it_was_released() {
        let h = Harness::new(1);
        let sink = Arc::new(Refuses {
            inner: h.sink.clone(),
            refuse: AtomicBool::new(false),
        });
        let mut runtime = h.runtime.clone();
        runtime.stages = sink.clone();
        let held = gates();
        let mut spec = StageGraphSpec::new();
        for _ in 0..2 {
            spec.stage(StageSpec {
                name: "g",
                ordered: None,
                lanes: vec![],
            });
        }
        spec.edge(StageId(0), StageId(1), Submission);
        let log = h.log.clone();
        let stages: Vec<Arc<dyn Stage<u64>>> = [GateAct::Pass, GateAct::Hold(held.clone())]
            .into_iter()
            .enumerate()
            .map(|(i, gate)| {
                Arc::new(Named(
                    i as u16,
                    Probe {
                        act: Act::Cpu,
                        gate,
                        log: h.log.clone(),
                        hands: h.hands.clone(),
                    },
                )) as Arc<dyn Stage<u64>>
            })
            .collect();
        let graph = StageGraph::new(spec, stages, runtime, move |q, outcome| {
            lock(&log).push(Event::Retired { q, outcome });
        })
        .unwrap();
        let _ = graph.start(1, alive(), Vec::new());
        h.pump();
        assert_eq!(h.ran(1), [0]);
        sink.refuse.store(true, Ordering::Release);
        lock(&held).pop().unwrap().release();
        assert_eq!(h.ran(1), [0, 1], "the refused stage ran in place");
        assert_eq!(h.retired(1), [QuantumOutcome::Completed]);
        assert_eq!(graph.live(), 0);
    }

    /// An ordered stage cannot be run beside its sink (its order is the sink\'s to keep): a refused one fails the quantum
    /// and drops its count.
    #[test]
    fn an_ordered_stage_released_into_a_sink_that_refuses_it_fails_the_quantum() {
        let h = Harness::new(1);
        let sink = Arc::new(Refuses {
            inner: h.sink.clone(),
            refuse: AtomicBool::new(false),
        });
        let mut runtime = h.runtime.clone();
        runtime.stages = sink.clone();
        let held: Arc<Mutex<Vec<Gate>>> = Arc::default();
        let k = held.clone();
        let mut spec = StageGraphSpec::new();
        spec.stage(StageSpec {
            name: "g",
            ordered: Some(9),
            lanes: vec![],
        });
        let log = h.log.clone();
        let stage = Arc::new(GateFn(h.log.clone(), move |gate: GateRef<'_>| {
            lock(&k).push(gate.hold());
            false
        }));
        let graph = StageGraph::new(spec, vec![stage], runtime, move |q, outcome| {
            lock(&log).push(Event::Retired { q, outcome });
        })
        .unwrap();
        let _ = graph.start(1, alive(), Vec::new());
        sink.refuse.store(true, Ordering::Release);
        lock(&held).pop().unwrap().release();
        assert!(h.ran(1).is_empty());
        assert!(matches!(
            &h.retired(1)[..],
            [QuantumOutcome::Failed {
                stage: StageId(0),
                ..
            }]
        ));
        assert_eq!(graph.live(), 0);
    }

    /// Java `StageGraphTest.repeatedQuantaReuseTheSameGraphFramesAndResetReadiness`: quanta one after another reuse
    /// one run and the same frames, and each starts with its readiness reset.
    #[test]
    fn repeated_quanta_reuse_the_pooled_run_and_frames_and_reset_readiness() {
        let h = Harness::new(2);
        let graph = h.graph(
            &[
                (None, &[0], Act::Submit),
                (None, &[1], Act::Submit),
                (None, &[], Act::Cpu),
                (None, &[0], Act::Submit),
            ],
            &[
                (0, 1, Submission),
                (0, 2, Completion),
                (1, 3, Submission),
                (2, 3, Submission),
            ],
        );
        let mut pooled = None;
        for q in 0..if cfg!(miri) { 4 } else { 20 } {
            let _ = graph.start(q, alive(), Vec::new());
            for _ in 0..4 {
                h.pump();
                h.finish_lanes();
            }
            h.pump();
            assert_eq!(h.retired(q), [QuantumOutcome::Completed], "quantum {q}");
            let now = (
                graph.runs.consumer.size_long(),
                graph.core.frames.consumer.size_long(),
            );
            assert_eq!(now.0, 1, "one run serves every quantum");
            match pooled {
                None => pooled = Some(now),
                Some(first) => assert_eq!(now, first, "no new frame after the first quantum"),
            }
            assert_eq!(h.ran(q).len(), 4, "every stage ran in quantum {q}");
        }
    }

    /// A run recycled with a stack sealed `Submitted` and a stage's old markers opens clean for its next quantum.
    #[test]
    fn a_rebound_run_starts_with_clean_stacks_and_markers() {
        let h = Harness::new(2);
        let graph = crossed(&h, Act::Submit);
        let _ = graph.start(1, alive(), Vec::new());
        h.pump();
        h.finish_lanes();
        h.pump();
        assert_eq!(h.retired(1), [QuantumOutcome::Completed]);
        // The same run again: a later quantum names its stage 1 before it submitted.
        let second = graph.start(2, alive(), Vec::new());
        let edge = ExternalEdge {
            from: second.clone(),
            from_stage: StageId(1),
            to_stage: StageId(0),
            kind: Submission,
        };
        let _ = graph.start(3, alive(), vec![edge]);
        drop(second);
        assert!(
            h.ran(3).is_empty(),
            "it waits for quantum 2's stage 1, not the old binding's"
        );
        h.pump();
        assert_eq!(
            h.ran_with(3, 0).unwrap().1,
            [(0, 2)],
            "quantum 2's marker, value 2 on lane 0"
        );
        h.finish_lanes();
        h.pump();
        assert_eq!(graph.live(), 0);
    }

    /// Java `concurrentFinishersRecycleGraphsForLaterQuanta`: quanta started from several threads while workers run
    /// and retire others each retire once, and the pool serves them.
    #[test]
    #[allow(clippy::disallowed_methods)] // The test's own threads start quanta.
    fn quanta_started_on_four_threads_retire_once_each_while_the_pool_serves_them() {
        let h = Arc::new(Harness::new(2));
        let graph = h.graph(
            &[
                (None, &[0], Act::Submit),
                (None, &[1], Act::Submit),
                (None, &[], Act::Cpu),
            ],
            &[(0, 1, Submission), (1, 2, Completion)],
        );
        let per_thread = if cfg!(miri) { 5 } else { 100u64 };
        let started = Arc::new(AtomicU64::new(0));
        let starters: Vec<_> = (0..4u64)
            .map(|t| {
                let (graph, started) = (graph.clone(), started.clone());
                std::thread::spawn(move || {
                    for i in 0..per_thread {
                        let _ = graph.start(t * per_thread + i, alive(), Vec::new());
                        started.fetch_add(1, Ordering::AcqRel);
                    }
                })
            })
            .collect();
        let (g, s) = (graph.clone(), started.clone());
        drive(&h, 4, move || {
            s.load(Ordering::Acquire) == 4 * per_thread && g.live() == 0
        });
        for starter in starters {
            starter.join().unwrap();
        }
        for q in 0..4 * per_thread {
            assert_eq!(h.retired(q), [QuantumOutcome::Completed], "quantum {q}");
            assert_eq!(h.ran(q).len(), 3, "quantum {q}");
        }
        assert_eq!(h.runtime.completions.pending(), 0);
    }

    /// The retire callback may start the next quantum: the retiring run is not yet in the pool, and the next quantum
    /// runs on another.
    #[test]
    fn a_quantum_may_start_the_next_from_its_retire_callback() {
        let h = Harness::new(1);
        let graph: Arc<OnceLock<Weak<StageGraph<u64>>>> = Arc::default();
        let log = h.log.clone();
        let again = graph.clone();
        let spec = {
            let mut spec = StageGraphSpec::new();
            spec.stage(StageSpec {
                name: "probe",
                ordered: None,
                lanes: vec![],
            });
            spec
        };
        let built = StageGraph::new(
            spec,
            vec![Arc::new(Named(
                0,
                Probe {
                    act: Act::Cpu,
                    gate: GateAct::Pass,
                    log: h.log.clone(),
                    hands: h.hands.clone(),
                },
            ))],
            h.runtime.clone(),
            move |q: u64, outcome| {
                lock(&log).push(Event::Retired { q, outcome });
                if q < 3 {
                    let next = again.get().unwrap().upgrade().unwrap();
                    let _ = next.start(q + 1, alive(), Vec::new());
                }
            },
        )
        .unwrap();
        graph.set(Arc::downgrade(&built)).ok().unwrap();
        let _ = built.start(0, alive(), Vec::new());
        h.pump();
        for q in 0..=3 {
            assert_eq!(h.retired(q), [QuantumOutcome::Completed], "quantum {q}");
        }
        assert_eq!(built.live(), 0);
    }

    /// Java `awaitExternals` joins every external slot's marker at quiesce: a middle quantum whose stage ran and
    /// failed (so it never waited on the device for its external markers) still passes the earlier order on.
    #[test]
    fn a_failed_middle_stage_passes_the_earlier_order_on() {
        let h = Harness::new(2);
        let first = h.graph(&[(None, &[0], Act::Submit)], &[]);
        let middle = h.graph(&[(None, &[1], Act::Fail)], &[]);
        let last = h.graph(&[(None, &[1], Act::Submit)], &[]);
        let a = first.start(1, alive(), Vec::new());
        let edge = |from: QuantumRef| ExternalEdge {
            from,
            from_stage: StageId(0),
            to_stage: StageId(0),
            kind: Submission,
        };
        let b = middle.start(2, alive(), vec![edge(a)]);
        let _ = last.start(3, alive(), vec![edge(b)]);
        h.pump();
        assert_eq!(h.ran(2), [0], "the middle stage ran (and failed)");
        assert_eq!(
            h.ran_with(3, 0).unwrap().1,
            [(0, 1)],
            "the last follows the first through the failed middle"
        );
        h.finish_lanes();
        h.pump();
        assert_eq!(h.retired(1), [QuantumOutcome::Completed]);
        assert_eq!(h.retired(3), [QuantumOutcome::Completed]);
    }

    /// The same for a CPU-only middle stage: it submits no marker and waits for none.
    #[test]
    fn a_cpu_middle_stage_passes_the_earlier_order_on() {
        let h = Harness::new(2);
        let first = h.graph(&[(None, &[0], Act::Submit)], &[]);
        let middle = h.graph(&[(None, &[], Act::Cpu)], &[]);
        let last = h.graph(&[(None, &[1], Act::Submit)], &[]);
        let a = first.start(1, alive(), Vec::new());
        let edge = |from: QuantumRef| ExternalEdge {
            from,
            from_stage: StageId(0),
            to_stage: StageId(0),
            kind: Submission,
        };
        let b = middle.start(2, alive(), vec![edge(a)]);
        let _ = last.start(3, alive(), vec![edge(b)]);
        h.pump();
        assert_eq!(h.ran_with(3, 0).unwrap().1, [(0, 1)]);
    }

    /// A stage that sets its own quantum's kill switch while it runs, then returns a host completion and has a
    /// completion-edge successor: Java's deferral keeps the quantum live until the host work ends.
    struct KillsThenDefers {
        kill: Arc<AtomicBool>,
        token: Token,
        log: Log,
    }

    impl Stage<u64> for KillsThenDefers {
        fn run(&self, q: &u64, cx: &StageCx<'_>) -> Result<Submitted, StageError> {
            lock(&self.log).push(Event::Ran {
                q: *q,
                stage: cx.stage.0,
                lane: None,
                waits: Vec::new(),
            });
            self.kill.store(true, Ordering::Release);
            Ok(Submitted {
                marker: None,
                completion: Some(Box::new(self.token.clone())),
            })
        }
    }

    #[test]
    fn a_halted_quantum_keeps_a_host_completion_that_feeds_a_completion_edge() {
        let h = Harness::new(1);
        let kill = alive();
        let token = Token::default();
        let mut spec = StageGraphSpec::new();
        for _ in 0..2 {
            spec.stage(StageSpec {
                name: "probe",
                ordered: None,
                lanes: Vec::new(),
            });
        }
        spec.edge(StageId(0), StageId(1), EdgeKind::Completion);
        let log = h.log.clone();
        let stages: Vec<Arc<dyn Stage<u64>>> = vec![
            Arc::new(KillsThenDefers {
                kill: kill.clone(),
                token: token.clone(),
                log: h.log.clone(),
            }),
            Arc::new(Named(
                1,
                Probe {
                    act: Act::Cpu,
                    gate: GateAct::Pass,
                    log: h.log.clone(),
                    hands: h.hands.clone(),
                },
            )),
        ];
        let graph = StageGraph::new(spec, stages, h.runtime.clone(), move |q, outcome| {
            lock(&log).push(Event::Retired { q, outcome });
        })
        .unwrap();
        let _ = graph.start(1, kill, Vec::new());
        h.pump();
        assert_eq!(h.ran(1), [0]);
        assert!(
            h.retired(1).is_empty(),
            "the host completion is not ready: the quantum must not retire yet"
        );
        assert_eq!(h.abandoned(1), [], "nor quiesce");
        token.0.store(true, Ordering::Release);
        h.pump();
        assert_eq!(h.ran(1), [0], "nothing runs past a killed quantum");
        assert_eq!(h.retired(1), [QuantumOutcome::Killed]);
        assert_eq!(h.abandoned(1), [1]);
    }

    /// A stage that returns a marker of a queue that is no lane of the pool, with a host completion.
    struct ForeignWithHost(Token);

    impl Stage<u64> for ForeignWithHost {
        fn run(&self, _: &u64, _: &StageCx<'_>) -> Result<Submitted, StageError> {
            Ok(Submitted {
                marker: Some(Marker::new(QueueId::next(), LaneId(0), 1)),
                completion: Some(Box::new(self.0.clone())),
            })
        }
    }

    #[test]
    fn a_stage_that_fails_on_a_foreign_marker_keeps_its_host_completion() {
        let h = Harness::new(1);
        let token = Token::default();
        let mut spec = StageGraphSpec::new();
        spec.stage(StageSpec {
            name: "foreign",
            ordered: None,
            lanes: Vec::new(),
        });
        let log = h.log.clone();
        let graph = StageGraph::new(
            spec,
            vec![Arc::new(ForeignWithHost(token.clone())) as Arc<dyn Stage<u64>>],
            h.runtime.clone(),
            move |q, outcome| lock(&log).push(Event::Retired { q, outcome }),
        )
        .unwrap();
        let _ = graph.start(1, alive(), Vec::new());
        h.pump();
        assert!(
            h.retired(1).is_empty(),
            "the host completion still holds the quantum"
        );
        token.0.store(true, Ordering::Release);
        h.pump();
        assert!(matches!(
            &h.retired(1)[..],
            [QuantumOutcome::Failed {
                error: StageError::Fatal(_),
                ..
            }]
        ));
    }

    /// A later quantum of a graph on a wider lane pool pins an earlier one: its marker is no lane of the earlier pool,
    /// and the earlier quantum's boundary leaves it out instead of indexing past its lanes.
    #[test]
    fn a_later_quantum_on_another_lane_pool_is_left_out_of_the_boundary() {
        let small = Harness::new(1);
        let big = Harness::new(2);
        let earlier = small.graph(&[(None, &[0], Act::Submit)], &[]);
        let later = big.graph(&[(None, &[1], Act::Submit)], &[]);
        let a = earlier.start(1, alive(), Vec::new());
        let edge = ExternalEdge {
            from: a,
            from_stage: StageId(0),
            to_stage: StageId(0),
            kind: Submission,
        };
        let _ = later.start(2, alive(), vec![edge]);
        small.pump();
        big.pump();
        small.finish_lanes();
        big.finish_lanes();
        small.pump();
        big.pump();
        assert_eq!(small.retired(1), [QuantumOutcome::Completed]);
        assert_eq!(big.retired(2), [QuantumOutcome::Completed]);
    }

    /// Java `abandoned` is for every stage that never submitted: also one that ran and failed, or panicked.
    #[test]
    fn a_stage_that_ran_and_failed_is_abandoned_with_the_stages_behind_it() {
        let h = Harness::new(1);
        let graph = h.graph(
            &[
                (None, &[], Act::Cpu),
                (None, &[], Act::Fail),
                (None, &[], Act::Cpu),
            ],
            &[(0, 1, Submission), (1, 2, Submission)],
        );
        let _ = graph.start(1, alive(), Vec::new());
        h.pump();
        assert_eq!(h.ran(1), [0, 1]);
        assert_eq!(h.abandoned(1), [1, 2], "the failed stage never submitted");
    }

    // ---- A stage holds itself at its gate from inside `run` (`StageCx::hold`, `StageError::Held`) ----

    /// What a [`Holds`] stage does in `run`.
    #[derive(Clone)]
    enum HoldAct {
        /// The first `n` runs take the gate, hand it to the list and answer `Held`; later runs finish.
        Park(u32, Arc<Mutex<Vec<Gate>>>),
        /// The first `n` runs take the gate and release it on this thread before they answer `Held`; the sink must hold
        /// nothing meanwhile.
        ReleaseAtOnce(u32),
        /// As `ReleaseAtOnce`, the release made by a thread that is joined before the run answers.
        ReleaseOnAThread,
        /// The first run takes the gate and fails it before it answers `Held`.
        FailAtOnce(StageError),
        /// The first run takes the gate and drops it before it answers `Held`.
        DropAtOnce,
        /// The first run takes the gate (kept in the list), then breaks the contract as told.
        Break(Broken, Arc<Mutex<Vec<Gate>>>),
        /// Every run takes the gate and hands it to the list; the test sets the kill switch in the first.
        KillThenPark(Arc<AtomicBool>, Arc<Mutex<Vec<Gate>>>),
        /// The first run answers `Held` with no gate taken.
        HeldWithoutGate,
        /// The first run sets the kill switch, takes the gate and fails it before it answers `Held`.
        KillThenFail(Arc<AtomicBool>),
        /// The first run sets the kill switch, takes the gate and drops it before it answers `Held`.
        KillThenDrop(Arc<AtomicBool>),
    }

    #[derive(Clone, Copy)]
    enum Broken {
        /// Answers `Ok`.
        Ok,
        /// Answers `Retry`.
        Retry,
        /// Answers an error.
        Corrupt,
        /// Panics.
        Panic,
        /// Takes a second gate.
        Twice,
    }

    /// A CPU stage that holds itself in `run` as `act` says, and logs like [`Named`].
    struct Holds {
        id: u16,
        act: HoldAct,
        log: Log,
        runs: AtomicU32,
        /// The harness's stage sink: nothing is published there while a run goes on.
        sink: Arc<QueueIngestSink>,
    }

    impl Holds {
        fn new(h: &Harness, id: u16, act: HoldAct) -> Arc<Self> {
            Arc::new(Self {
                id,
                act,
                log: h.log.clone(),
                runs: AtomicU32::new(0),
                sink: h.sink.clone(),
            })
        }

        fn runs(&self) -> u32 {
            self.runs.load(Ordering::Acquire)
        }
    }

    impl Stage<u64> for Holds {
        fn run(&self, q: &u64, cx: &StageCx<'_>) -> Result<Submitted, StageError> {
            let n = self.runs.fetch_add(1, Ordering::AcqRel);
            lock(&self.log).push(Event::Ran {
                q: *q,
                stage: cx.stage.0,
                lane: cx.lane.map(|l| l.0),
                waits: Vec::new(),
            });
            match &self.act {
                HoldAct::HeldWithoutGate if n == 0 => return Err(StageError::Held),
                HoldAct::Park(times, gates) if n < *times => {
                    lock(gates).push(cx.hold());
                    return Err(StageError::Held);
                }
                HoldAct::ReleaseAtOnce(times) if n < *times => {
                    cx.hold().release();
                    assert_eq!(self.sink.size(), 0, "published while its run goes on");
                    return Err(StageError::Held);
                }
                HoldAct::ReleaseOnAThread if n == 0 => {
                    let gate = cx.hold();
                    let sink = self.sink.clone();
                    std::thread::spawn(move || {
                        gate.release();
                        assert_eq!(sink.size(), 0, "published while its run goes on");
                    })
                    .join()
                    .unwrap();
                    return Err(StageError::Held);
                }
                HoldAct::KillThenFail(kill) if n == 0 => {
                    kill.store(true, Ordering::Release);
                    cx.hold().fail(StageError::Corrupt("a bad packet".into()));
                    return Err(StageError::Held);
                }
                HoldAct::KillThenDrop(kill) if n == 0 => {
                    kill.store(true, Ordering::Release);
                    drop(cx.hold());
                    return Err(StageError::Held);
                }
                HoldAct::FailAtOnce(error) if n == 0 => {
                    cx.hold().fail(error.clone());
                    return Err(StageError::Held);
                }
                HoldAct::DropAtOnce if n == 0 => {
                    drop(cx.hold());
                    return Err(StageError::Held);
                }
                HoldAct::Break(how, gates) if n == 0 => {
                    lock(gates).push(cx.hold());
                    return match how {
                        Broken::Ok => Ok(Submitted::done()),
                        Broken::Retry => Err(StageError::Retry),
                        Broken::Corrupt => Err(StageError::Corrupt("a bad packet".into())),
                        Broken::Panic => panic!("the stage broke after it took its gate"),
                        Broken::Twice => {
                            let _second = cx.hold();
                            unreachable!("a second gate in one run panics")
                        }
                    };
                }
                HoldAct::KillThenPark(kill, gates) => {
                    kill.store(true, Ordering::Release);
                    lock(gates).push(cx.hold());
                    return Err(StageError::Held);
                }
                _ => {}
            }
            Ok(Submitted::done())
        }

        fn pass_gate(&self, q: &u64, _gate: GateRef<'_>) -> bool {
            lock(&self.log).push(Event::Gated {
                q: *q,
                stage: self.id,
            });
            true
        }

        fn abandoned(&self, q: &u64, _home: &dyn DeviceQueue) {
            lock(&self.log).push(Event::Abandoned {
                q: *q,
                stage: self.id,
            });
        }

        fn retired(&self, q: &u64, committed: bool) {
            lock(&self.log).push(Event::Hook {
                q: *q,
                stage: self.id,
                committed,
            });
        }
    }

    /// A CPU stage that logs like the harness's probes.
    fn cpu_stage(h: &Harness, id: u16) -> Arc<dyn Stage<u64>> {
        Arc::new(Named(
            id,
            Probe {
                act: Act::Cpu,
                gate: GateAct::Pass,
                log: h.log.clone(),
                hands: h.hands.clone(),
            },
        ))
    }

    /// A graph of CPU stages (all unordered) joined by submission edges.
    fn cpu_graph(
        h: &Harness,
        stages: Vec<Arc<dyn Stage<u64>>>,
        edges: &[(u16, u16)],
    ) -> Arc<StageGraph<u64>> {
        let mut spec = StageGraphSpec::new();
        for _ in &stages {
            spec.stage(StageSpec {
                name: "g",
                ordered: None,
                lanes: vec![],
            });
        }
        for &(from, to) in edges {
            spec.edge(StageId(from), StageId(to), Submission);
        }
        let log = h.log.clone();
        StageGraph::new(spec, stages, h.runtime.clone(), move |q, outcome| {
            lock(&log).push(Event::Retired { q, outcome });
        })
        .unwrap()
    }

    /// 0 -> 1 -> 2, the middle stage `mid`.
    fn holding_chain(h: &Harness, mid: Arc<Holds>) -> Arc<StageGraph<u64>> {
        cpu_graph(
            h,
            vec![cpu_stage(h, 0), mid, cpu_stage(h, 2)],
            &[(0, 1), (1, 2)],
        )
    }

    /// The context a stage is given crosses threads as before (a stage may hand `&StageCx` to scoped threads).
    #[test]
    fn the_context_of_a_run_is_send_and_sync() {
        fn both<T: Send + Sync>() {}
        both::<StageCx<'static>>();
    }

    /// Nothing the quantum waited for is left: no registration, no retry, no count.
    fn assert_no_retry(h: &Harness) {
        assert_eq!(h.runtime.completions.pending(), 0, "nothing was registered");
        let counts = &h.runtime.retries;
        assert_eq!(
            (counts.retries(), counts.progress(), counts.backstops()),
            (0, 0, 0),
            "no retry, no wake-up"
        );
    }

    /// A stage that finds its input not ready inside `run` holds, is released from another thread, runs exactly once more,
    /// and answers no `Retry`.
    #[test]
    fn a_gate_taken_in_run_holds_the_stage_until_a_release_from_another_thread_runs_it_once_more() {
        let h = Harness::new(1);
        let held = gates();
        let mid = Holds::new(&h, 1, HoldAct::Park(1, held.clone()));
        let graph = holding_chain(&h, mid.clone());
        let _ = graph.start(1, alive(), Vec::new());
        h.pump();
        assert_eq!(
            h.ran(1),
            [0, 1],
            "the stage ran, held itself, and its successor waits"
        );
        assert_eq!(mid.runs(), 1);
        assert_eq!(graph.live(), 1, "the held stage keeps the quantum live");
        assert!(h.retired(1).is_empty());
        assert_eq!(lock(&held).len(), 1, "the gate is out");
        assert_no_retry(&h);
        h.pump();
        assert_eq!(h.ran(1), [0, 1], "nothing runs it again by itself");
        let gate = lock(&held).pop().unwrap();
        std::thread::spawn(move || gate.release()).join().unwrap();
        assert_eq!(h.sink.size(), 1, "the release published the stage once");
        h.pump();
        assert_eq!(
            h.ran(1),
            [0, 1, 1, 2],
            "exactly one more run, then its successor"
        );
        assert_eq!(mid.runs(), 2);
        assert_eq!(
            gated(&h, 1),
            [0, 1, 2],
            "its gate is asked once per quantum"
        );
        assert_eq!(h.retired(1), [QuantumOutcome::Completed]);
        assert_eq!(hooks(&h, 1), [(0, true), (1, true), (2, true)]);
        assert_eq!(graph.live(), 0);
        assert_no_retry(&h);
    }

    /// A release that comes before `run` returns (the releaser is the running thread itself, or one it joins) publishes
    /// the stage when `run` returns, never beside it: the sink holds nothing until then.
    #[test]
    fn a_gate_taken_in_run_and_released_before_run_returns_publishes_the_stage_when_it_returns() {
        for act in [HoldAct::ReleaseAtOnce(1), HoldAct::ReleaseOnAThread] {
            let h = Harness::new(1);
            let mid = Holds::new(&h, 1, act);
            let graph = holding_chain(&h, mid.clone());
            let _ = graph.start(1, alive(), Vec::new());
            assert!(h.pump_once(), "stage 0");
            assert!(
                h.pump_once(),
                "stage 1: takes its gate, releases it, returns Held"
            );
            assert_eq!(mid.runs(), 1);
            assert_eq!(h.sink.size(), 1, "published when run returned");
            assert_eq!(h.ran(1), [0, 1]);
            h.pump();
            assert_eq!(h.ran(1), [0, 1, 1, 2]);
            assert_eq!(mid.runs(), 2, "exactly once more");
            assert_eq!(h.retired(1), [QuantumOutcome::Completed]);
            assert_eq!(graph.live(), 0);
            assert_no_retry(&h);
        }
    }

    /// Each release is one more run: a stage may hold again from the run a release gave it.
    #[test]
    fn a_gate_taken_in_run_may_be_taken_again_by_the_run_it_released() {
        let h = Harness::new(1);
        let held = gates();
        let mid = Holds::new(&h, 1, HoldAct::Park(3, held.clone()));
        let graph = holding_chain(&h, mid.clone());
        let _ = graph.start(1, alive(), Vec::new());
        for round in 1..=3u32 {
            h.pump();
            assert_eq!(mid.runs(), round, "round {round}");
            assert_eq!(lock(&held).len(), 1);
            assert_eq!(graph.live(), 1);
            lock(&held).pop().unwrap().release();
        }
        h.pump();
        assert_eq!(mid.runs(), 4, "three holds, three releases, one more run");
        assert_eq!(h.ran(1), [0, 1, 1, 1, 1, 2]);
        assert_eq!(h.retired(1), [QuantumOutcome::Completed]);
        assert_no_retry(&h);
    }

    /// A gate that fails, before `run` returns or after, fails the quantum with its error and drops the stage's count
    /// once: the stage does not run again, and it is abandoned with what is behind it.
    #[test]
    fn a_gate_taken_in_run_that_fails_fails_the_quantum_and_drops_the_count_once() {
        let error = StageError::Corrupt("the artifact would not read".into());
        for before_return in [true, false] {
            let h = Harness::new(1);
            let held = gates();
            let act = if before_return {
                HoldAct::FailAtOnce(error.clone())
            } else {
                HoldAct::Park(1, held.clone())
            };
            let mid = Holds::new(&h, 1, act);
            let graph = holding_chain(&h, mid.clone());
            let _ = graph.start(1, alive(), Vec::new());
            h.pump();
            if !before_return {
                assert_eq!(graph.live(), 1, "held");
                lock(&held).pop().unwrap().fail(error.clone());
                h.pump();
            }
            assert_eq!(mid.runs(), 1, "never run again");
            assert_eq!(h.ran(1), [0, 1]);
            assert_eq!(
                h.retired(1),
                [QuantumOutcome::Failed {
                    stage: StageId(1),
                    error: error.clone()
                }]
            );
            assert_eq!(h.abandoned(1), [1, 2], "abandoned once each");
            assert_eq!(hooks(&h, 1), [(0, false), (1, false)]);
            assert_eq!(graph.live(), 0);
            assert_no_retry(&h);
        }
    }

    /// A gate dropped unspent fails the quantum like any gate dropped.
    #[test]
    fn a_gate_taken_in_run_that_is_dropped_fails_the_quantum_and_drops_the_count_once() {
        for before_return in [true, false] {
            let h = Harness::new(1);
            let held = gates();
            let act = if before_return {
                HoldAct::DropAtOnce
            } else {
                HoldAct::Park(1, held.clone())
            };
            let mid = Holds::new(&h, 1, act);
            let graph = holding_chain(&h, mid.clone());
            let _ = graph.start(1, alive(), Vec::new());
            h.pump();
            if !before_return {
                assert_eq!(graph.live(), 1, "held");
                drop(lock(&held).pop().unwrap());
                h.pump();
            }
            assert_eq!(mid.runs(), 1);
            assert!(
                matches!(
                    &h.retired(1)[..],
                    [QuantumOutcome::Failed {
                        stage: StageId(1),
                        error: StageError::Fatal(_)
                    }]
                ),
                "{:?}",
                h.retired(1)
            );
            assert_eq!(h.abandoned(1), [1, 2]);
            assert_eq!(graph.live(), 0);
            assert_no_retry(&h);
        }
    }

    /// A quantum killed while its stage is held stays live until the gate is spent; the release then ends the stage without
    /// running it, the quantum retires as killed, and the stage is abandoned exactly once.
    #[test]
    fn a_gate_taken_in_run_and_released_after_a_kill_ends_the_stage_without_running_it() {
        let h = Harness::new(1);
        let held = gates();
        let mid = Holds::new(&h, 1, HoldAct::Park(1, held.clone()));
        let graph = holding_chain(&h, mid.clone());
        let kill = alive();
        let _ = graph.start(1, kill.clone(), Vec::new());
        h.pump();
        kill.store(true, Ordering::Release);
        h.pump();
        assert!(h.retired(1).is_empty(), "the gate keeps the quantum live");
        assert_eq!(graph.live(), 1);
        lock(&held).pop().unwrap().release();
        h.pump();
        assert_eq!(mid.runs(), 1, "not run again");
        assert_eq!(h.ran(1), [0, 1]);
        assert_eq!(h.retired(1), [QuantumOutcome::Killed]);
        assert_eq!(h.abandoned(1), [1, 2], "abandoned once each");
        assert_eq!(hooks(&h, 1), [(0, false), (1, false)]);
        assert_eq!(graph.live(), 0);
        assert_no_retry(&h);
    }

    /// The kill is set by the stage's own run, which then holds: the quantum is killed already, and the release ends it.
    #[test]
    fn a_gate_taken_in_run_by_a_stage_that_killed_its_quantum_ends_it_at_the_release() {
        let h = Harness::new(1);
        let held = gates();
        let kill = alive();
        let mid = Holds::new(&h, 1, HoldAct::KillThenPark(kill.clone(), held.clone()));
        let graph = holding_chain(&h, mid.clone());
        let _ = graph.start(1, kill, Vec::new());
        h.pump();
        assert_eq!(graph.live(), 1);
        lock(&held).pop().unwrap().release();
        h.pump();
        assert_eq!(mid.runs(), 1);
        assert_eq!(h.retired(1), [QuantumOutcome::Killed]);
        assert_eq!(h.abandoned(1), [1, 2]);
        assert_eq!(graph.live(), 0);
    }

    /// Once the quantum's kill switch is set, spending its gate any way (release, `fail`, drop) ends the quantum as killed,
    /// and the stage is abandoned once: the kill is noted whichever spend comes first.
    #[test]
    fn a_gate_taken_in_run_and_spent_after_a_kill_ends_the_quantum_as_killed() {
        for spend in ["release", "fail", "drop"] {
            let h = Harness::new(1);
            let held = gates();
            let mid = Holds::new(&h, 1, HoldAct::Park(1, held.clone()));
            let graph = holding_chain(&h, mid.clone());
            let kill = alive();
            let _ = graph.start(1, kill.clone(), Vec::new());
            h.pump();
            kill.store(true, Ordering::Release);
            let gate = lock(&held).pop().unwrap();
            match spend {
                "release" => gate.release(),
                "fail" => gate.fail(StageError::Corrupt("the artifact would not read".into())),
                _ => drop(gate),
            }
            h.pump();
            assert_eq!(mid.runs(), 1, "{spend}: not run again");
            assert_eq!(h.retired(1), [QuantumOutcome::Killed], "{spend}");
            assert_eq!(h.abandoned(1), [1, 2], "{spend}: abandoned once each");
            assert_eq!(graph.live(), 0, "{spend}");
            assert_no_retry(&h);
        }
    }

    /// The same when the kill switch is set by the stage's own run, which then spends the gate before it returns.
    #[test]
    fn a_gate_taken_in_run_and_spent_in_a_run_that_killed_its_quantum_ends_it_as_killed() {
        for fail in [true, false] {
            let h = Harness::new(1);
            let kill = alive();
            let act = if fail {
                HoldAct::KillThenFail(kill.clone())
            } else {
                HoldAct::KillThenDrop(kill.clone())
            };
            let mid = Holds::new(&h, 1, act);
            let graph = holding_chain(&h, mid.clone());
            let _ = graph.start(1, kill, Vec::new());
            h.pump();
            assert_eq!(mid.runs(), 1);
            assert_eq!(h.retired(1), [QuantumOutcome::Killed], "fail: {fail}");
            assert_eq!(h.abandoned(1), [1, 2]);
            assert_eq!(graph.live(), 0);
        }
    }

    /// A gate taken in `pass_gate` ends the same way: `fail` and drop after a kill end the quantum as killed.
    #[test]
    fn a_gate_taken_in_pass_gate_and_failed_or_dropped_after_a_kill_ends_the_quantum_as_killed() {
        for fail in [true, false] {
            let h = Harness::new(1);
            let held = gates();
            let graph = h.gated_graph(
                &[(None, &[], Act::Cpu), (None, &[], Act::Cpu)],
                &[(0, 1, Submission)],
                &[GateAct::Pass, GateAct::Hold(held.clone())],
            );
            let kill = alive();
            let _ = graph.start(1, kill.clone(), Vec::new());
            h.pump();
            kill.store(true, Ordering::Release);
            let gate = lock(&held).pop().unwrap();
            if fail {
                gate.fail(StageError::Corrupt("the artifact would not read".into()));
            } else {
                drop(gate);
            }
            h.pump();
            assert_eq!(h.retired(1), [QuantumOutcome::Killed], "fail: {fail}");
            assert_eq!(h.abandoned(1), [1], "fail: {fail}");
            assert_eq!(graph.live(), 0);
        }
    }

    /// `Retry` and `Held` are no failures: a gate failed with either fails the quantum with `Fatal` (the cause names which),
    /// whether the gate was taken in `run` or in `pass_gate`.
    #[test]
    fn a_gate_failed_with_retry_or_held_fails_the_quantum_with_fatal() {
        for error in [StageError::Retry, StageError::Held] {
            for in_run in [true, false] {
                let h = Harness::new(1);
                let held = gates();
                let graph = if in_run {
                    holding_chain(&h, Holds::new(&h, 1, HoldAct::Park(1, held.clone())))
                } else {
                    h.gated_graph(
                        &[
                            (None, &[], Act::Cpu),
                            (None, &[], Act::Cpu),
                            (None, &[], Act::Cpu),
                        ],
                        &[(0, 1, Submission), (1, 2, Submission)],
                        &[GateAct::Pass, GateAct::Hold(held.clone())],
                    )
                };
                let _ = graph.start(1, alive(), Vec::new());
                h.pump();
                lock(&held).pop().unwrap().fail(error.clone());
                h.pump();
                assert!(
                    matches!(
                        &h.retired(1)[..],
                        [QuantumOutcome::Failed {
                            stage: StageId(1),
                            error: StageError::Fatal(_)
                        }]
                    ),
                    "{error:?} in_run {in_run}: {:?}",
                    h.retired(1)
                );
                assert_eq!(graph.live(), 0);
                assert_no_retry(&h);
            }
        }
    }

    /// A stage that took a gate and answered `Ok` has submitted device work and a host completion all the same: the
    /// boundary joins its lane and the quantum waits for the completion before it ends as failed.
    #[test]
    fn a_gate_taken_in_run_with_ok_answered_still_joins_its_marker_and_waits_for_its_completion() {
        struct TakesAndSubmits {
            hand: Arc<HandLane>,
            token: Token,
            gates: Arc<Mutex<Vec<Gate>>>,
        }
        impl Stage<u64> for TakesAndSubmits {
            fn run(&self, _: &u64, cx: &StageCx<'_>) -> Result<Submitted, StageError> {
                lock(&self.gates).push(cx.hold());
                let lane = cx.lane.expect("a lane stage");
                Ok(Submitted {
                    marker: Some(Marker::new(self.hand.id, lane, self.hand.submit())),
                    completion: Some(Box::new(self.token.clone())),
                })
            }
        }
        let h = Harness::new(1);
        let held = gates();
        let token = Token::default();
        let mut spec = StageGraphSpec::new();
        spec.stage(StageSpec {
            name: "g",
            ordered: None,
            lanes: vec![LaneId(0)],
        });
        let log = h.log.clone();
        let stage = TakesAndSubmits {
            hand: h.hands[0].clone(),
            token: token.clone(),
            gates: held.clone(),
        };
        let graph = StageGraph::new(
            spec,
            vec![Arc::new(stage) as Arc<dyn Stage<u64>>],
            h.runtime.clone(),
            move |q, outcome| lock(&log).push(Event::Retired { q, outcome }),
        )
        .unwrap();
        let _ = graph.start(1, alive(), Vec::new());
        h.pump();
        lock(&held).pop().unwrap().release();
        h.pump();
        assert!(
            h.retired(1).is_empty(),
            "its host completion is not ready: {:?}",
            h.retired(1)
        );
        token.0.store(true, Ordering::Release);
        h.pump();
        assert!(
            h.retired(1).is_empty(),
            "its marker is not reached: the boundary joined its lane"
        );
        h.finish_lanes();
        h.pump();
        assert!(
            matches!(
                &h.retired(1)[..],
                [QuantumOutcome::Failed {
                    stage: StageId(0),
                    error: StageError::Fatal(_)
                }]
            ),
            "{:?}",
            h.retired(1)
        );
        assert_eq!(graph.live(), 0);
    }

    /// A stage that breaks the contract fails the quantum and is not run again; the gate it took stays valid, holds its
    /// count, and is harmless to spend.
    #[test]
    fn a_gate_taken_in_run_by_a_stage_that_answers_anything_but_held_fails_the_quantum() {
        for (how, expect) in [
            (
                Broken::Ok,
                StageError::Fatal("a stage that took a gate in run did not answer Held".into()),
            ),
            (
                Broken::Retry,
                StageError::Fatal("a stage that took a gate in run did not answer Held".into()),
            ),
            (Broken::Corrupt, StageError::Corrupt("a bad packet".into())),
        ] {
            let h = Harness::new(1);
            let held = gates();
            let mid = Holds::new(&h, 1, HoldAct::Break(how, held.clone()));
            let graph = holding_chain(&h, mid.clone());
            let _ = graph.start(1, alive(), Vec::new());
            h.pump();
            assert_eq!(graph.live(), 1, "the gate holds the count");
            assert!(h.retired(1).is_empty());
            lock(&held).pop().unwrap().release();
            h.pump();
            assert_eq!(mid.runs(), 1, "never run again");
            assert_eq!(
                h.retired(1),
                [QuantumOutcome::Failed {
                    stage: StageId(1),
                    error: expect
                }]
            );
            // An `Ok` answer is a submission all the same, so only the stage behind it is abandoned.
            let abandoned: &[u16] = if matches!(how, Broken::Ok) {
                &[2]
            } else {
                &[1, 2]
            };
            assert_eq!(h.abandoned(1), abandoned);
            assert_eq!(graph.live(), 0);
            assert_no_retry(&h);
        }
    }

    /// A panic in `run` after the gate was taken, or a second gate in one run (a panic), fails the quantum with `Fatal`;
    /// the first gate stays valid and harmless.
    #[test]
    fn a_gate_taken_in_run_by_a_stage_that_then_panics_fails_the_quantum_and_stays_harmless() {
        for how in [Broken::Panic, Broken::Twice] {
            let h = Harness::new(1);
            let held = gates();
            let mid = Holds::new(&h, 1, HoldAct::Break(how, held.clone()));
            let graph = holding_chain(&h, mid.clone());
            let _ = graph.start(1, alive(), Vec::new());
            h.pump();
            assert_eq!(graph.live(), 1, "the gate holds the count");
            lock(&held).pop().unwrap().release();
            h.pump();
            assert_eq!(mid.runs(), 1);
            assert!(
                matches!(
                    &h.retired(1)[..],
                    [QuantumOutcome::Failed {
                        stage: StageId(1),
                        error: StageError::Fatal(_)
                    }]
                ),
                "{:?}",
                h.retired(1)
            );
            assert_eq!(graph.live(), 0);
            assert_no_retry(&h);
        }
    }

    /// A panic after the gate was taken and released from another thread before the panic: the end of the run drops the
    /// count once and nothing publishes the stage.
    #[test]
    fn a_gate_released_by_a_stage_that_then_panics_fails_the_quantum_without_a_second_run() {
        struct ReleasesThenPanics;
        impl Stage<u64> for ReleasesThenPanics {
            fn run(&self, _: &u64, cx: &StageCx<'_>) -> Result<Submitted, StageError> {
                cx.hold().release();
                panic!("the stage broke after it released its gate")
            }
        }
        let h = Harness::new(1);
        let graph = cpu_graph(
            &h,
            vec![Arc::new(ReleasesThenPanics), cpu_stage(&h, 1)],
            &[(0, 1)],
        );
        let _ = graph.start(1, alive(), Vec::new());
        h.pump();
        assert!(
            matches!(
                &h.retired(1)[..],
                [QuantumOutcome::Failed {
                    stage: StageId(0),
                    error: StageError::Fatal(_)
                }]
            ),
            "{:?}",
            h.retired(1)
        );
        assert_eq!(h.abandoned(1), [1], "its successor, which never ran");
        assert_eq!(graph.live(), 0);
    }

    /// `Held` with no gate taken would hang the quantum: it fails instead.
    #[test]
    fn a_gate_never_taken_by_a_stage_that_answers_held_fails_the_quantum() {
        let h = Harness::new(1);
        let mid = Holds::new(&h, 1, HoldAct::HeldWithoutGate);
        let graph = holding_chain(&h, mid.clone());
        let _ = graph.start(1, alive(), Vec::new());
        h.pump();
        assert_eq!(mid.runs(), 1);
        assert!(
            matches!(
                &h.retired(1)[..],
                [QuantumOutcome::Failed {
                    stage: StageId(1),
                    error: StageError::Fatal(_)
                }]
            ),
            "{:?}",
            h.retired(1)
        );
        assert_eq!(graph.live(), 0);
        assert_no_retry(&h);
    }

    /// A sink that refuses the stage when its gate is released: an unordered stage runs where it was released, the count is
    /// dropped. Released before `run` returns, that is where `run` returns.
    #[test]
    fn a_gate_taken_in_run_and_released_into_a_sink_that_refuses_the_stage_runs_it_in_place() {
        for before_return in [false, true] {
            let h = Harness::new(1);
            let sink = Arc::new(Refuses {
                inner: h.sink.clone(),
                refuse: AtomicBool::new(false),
            });
            let mut runtime = h.runtime.clone();
            runtime.stages = sink.clone();
            let held = gates();
            let act = if before_return {
                // Refuses from the moment the run starts, so the release at its end is refused.
                HoldAct::ReleaseAtOnce(1)
            } else {
                HoldAct::Park(1, held.clone())
            };
            let mid = Holds::new(&h, 1, act);
            let mut spec = StageGraphSpec::new();
            spec.stage(StageSpec {
                name: "g",
                ordered: None,
                lanes: vec![],
            });
            let log = h.log.clone();
            let graph = StageGraph::new(
                spec,
                vec![mid.clone() as Arc<dyn Stage<u64>>],
                runtime,
                move |q, outcome| {
                    lock(&log).push(Event::Retired { q, outcome });
                },
            )
            .unwrap();
            let _ = graph.start(1, alive(), Vec::new());
            if before_return {
                // The run asserts the sink is empty while it goes on; the sink refuses what is published at its end.
                sink.refuse.store(true, Ordering::Release);
            }
            h.pump();
            if !before_return {
                assert_eq!(mid.runs(), 1);
                sink.refuse.store(true, Ordering::Release);
                lock(&held).pop().unwrap().release();
            }
            assert_eq!(mid.runs(), 2, "ran again where the refusal was found");
            assert_eq!(h.retired(1), [QuantumOutcome::Completed]);
            assert_eq!(graph.live(), 0);
        }
    }

    /// An ordered stage cannot be run beside its sink: released into a sink that refuses it, the quantum fails and the
    /// count is dropped, as for a gate taken in `pass_gate`.
    #[test]
    fn a_gate_taken_in_run_by_an_ordered_stage_and_released_into_a_refusing_sink_fails_the_quantum()
    {
        let h = Harness::new(1);
        let sink = Arc::new(Refuses {
            inner: h.sink.clone(),
            refuse: AtomicBool::new(false),
        });
        let mut runtime = h.runtime.clone();
        runtime.stages = sink.clone();
        let held = gates();
        let mid = Holds::new(&h, 0, HoldAct::Park(1, held.clone()));
        let mut spec = StageGraphSpec::new();
        spec.stage(StageSpec {
            name: "g",
            ordered: Some(9),
            lanes: vec![],
        });
        let log = h.log.clone();
        let graph = StageGraph::new(
            spec,
            vec![mid.clone() as Arc<dyn Stage<u64>>],
            runtime,
            move |q, outcome| {
                lock(&log).push(Event::Retired { q, outcome });
            },
        )
        .unwrap();
        let _ = graph.start(1, alive(), Vec::new());
        h.pump();
        assert_eq!(mid.runs(), 1);
        sink.refuse.store(true, Ordering::Release);
        lock(&held).pop().unwrap().release();
        assert_eq!(mid.runs(), 1, "an ordered stage is not run beside its sink");
        assert!(matches!(
            &h.retired(1)[..],
            [QuantumOutcome::Failed {
                stage: StageId(0),
                ..
            }]
        ));
        assert_eq!(graph.live(), 0);
    }

    /// Four stages hold in one quantum and are released from four threads at once; the join behind them runs once, after all.
    #[test]
    fn a_gate_taken_in_run_by_four_stages_and_released_from_four_threads_at_once_runs_each_once_more_and_the_join_once()
     {
        let h = Harness::new(1);
        let held = gates();
        let holders: Vec<Arc<Holds>> = (1..=4u16)
            .map(|id| Holds::new(&h, id, HoldAct::Park(1, held.clone())))
            .collect();
        let mut stages = vec![cpu_stage(&h, 0)];
        stages.extend(holders.iter().map(|s| s.clone() as Arc<dyn Stage<u64>>));
        stages.push(cpu_stage(&h, 5));
        let graph = cpu_graph(
            &h,
            stages,
            &[
                (0, 1),
                (0, 2),
                (0, 3),
                (0, 4),
                (1, 5),
                (2, 5),
                (3, 5),
                (4, 5),
            ],
        );
        let _ = graph.start(1, alive(), Vec::new());
        h.pump();
        let taken: Vec<Gate> = lock(&held).drain(..).collect();
        assert_eq!(taken.len(), 4);
        assert!(h.retired(1).is_empty(), "four gates keep the quantum live");
        let released = Arc::new(Barrier::new(taken.len()));
        let threads: Vec<_> = taken
            .into_iter()
            .map(|gate| {
                let released = released.clone();
                std::thread::spawn(move || {
                    released.wait();
                    gate.release();
                })
            })
            .collect();
        for thread in threads {
            thread.join().unwrap();
        }
        h.pump();
        let mut ran = h.ran(1);
        ran.sort_unstable();
        assert_eq!(
            ran,
            [0, 1, 1, 2, 2, 3, 3, 4, 4, 5],
            "each holder twice, the join once"
        );
        assert!(holders.iter().all(|s| s.runs() == 2));
        assert_eq!(h.retired(1), [QuantumOutcome::Completed]);
        assert_no_retry(&h);
    }

    /// The release races the end of the run, both on their own threads with nothing between them: the stage is run once
    /// more whichever comes first (the deterministic cases above pin each order; this one lets the scheduler pick).
    #[test]
    fn a_gate_taken_in_run_released_by_a_racing_thread_runs_the_stage_once_more_whoever_is_first() {
        struct Races {
            runs: AtomicU32,
            start: Arc<Barrier>,
            releaser: Mutex<Option<std::thread::JoinHandle<()>>>,
        }
        impl Stage<u64> for Races {
            fn run(&self, _: &u64, cx: &StageCx<'_>) -> Result<Submitted, StageError> {
                if self.runs.fetch_add(1, Ordering::AcqRel) > 0 {
                    return Ok(Submitted::done());
                }
                let gate = cx.hold();
                let start = self.start.clone();
                *lock(&self.releaser) = Some(std::thread::spawn(move || {
                    start.wait();
                    gate.release();
                }));
                self.start.wait();
                Err(StageError::Held)
            }
        }
        let rounds = if cfg!(miri) { 3 } else { 300 };
        for round in 0..rounds {
            let h = Harness::new(1);
            let stage = Arc::new(Races {
                runs: AtomicU32::new(0),
                start: Arc::new(Barrier::new(2)),
                releaser: Mutex::default(),
            });
            let graph = cpu_graph(&h, vec![stage.clone(), cpu_stage(&h, 1)], &[(0, 1)]);
            let _ = graph.start(round, alive(), Vec::new());
            h.pump();
            lock(&stage.releaser).take().unwrap().join().unwrap();
            h.pump();
            assert_eq!(stage.runs.load(Ordering::Acquire), 2, "round {round}");
            assert_eq!(h.ran(round), [1], "the successor ran once");
            assert_eq!(
                h.retired(round),
                [QuantumOutcome::Completed],
                "round {round}"
            );
            assert_eq!(graph.live(), 0);
        }
    }
}
