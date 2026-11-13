//! [`StageHarness`]: calls a [`Stage`]'s hooks directly, with no graph, lattice or workers.

use std::cell::Cell;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use crate::lanes::{DeviceQueue, LaneId, LanePool, Marker};
use crate::stage_graph::{
    Gate, GateRef, GateSource, GateTarget, Stage, StageCx, StageError, StageId, Submitted,
};

/// How a [`Gate`] a stage took was spent.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GateEvent {
    /// [`Gate::release`]: in a graph, the stage would be published and run again.
    Released,
    /// [`Gate::fail`] with this error (`Held` and `Retry` arrive as the `Fatal` a graph makes of them), or a drop of the
    /// gate unspent, which arrives as `Fatal`: in a graph, the quantum would fail with it.
    Failed(StageError),
}

/// What one call of [`StageHarness::run`] did.
#[derive(Debug)]
pub struct StageRun {
    /// What `run` answered: the marker and host completion it submitted, or the error. In a graph `Err(Held)` and
    /// `Err(Retry)` are no failures; here they are only answers.
    pub result: Result<Submitted, StageError>,
    /// Whether the run took its gate ([`StageCx::hold`]).
    pub held: bool,
}

/// What [`StageHarness::pass_gate`] did.
#[derive(Debug)]
pub struct PassGate {
    /// What `pass_gate` answered: `true` publishes the stage.
    pub open: bool,
    /// Whether it took a gate ([`GateRef::hold`]); in a graph that holds the stage whatever `open` says.
    pub held: bool,
}

/// Where the gates of a harness are spent.
#[derive(Debug, Default)]
struct GateLog {
    events: Mutex<Vec<GateEvent>>,
}

impl GateLog {
    fn push(&self, event: GateEvent) {
        self.events
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(event);
    }
}

impl GateTarget for GateLog {
    fn release_gate(self: Arc<Self>, _stage: StageId) {
        self.push(GateEvent::Released);
    }

    fn fail_gate(self: Arc<Self>, _stage: StageId, error: StageError) {
        self.push(GateEvent::Failed(error));
    }
}

impl GateSource for Arc<GateLog> {
    fn hold_gate(&self, stage: StageId) -> Gate {
        Gate::into_target(self.clone(), stage)
    }

    fn hold_gate_in_run(&self, stage: StageId) -> Gate {
        Gate::into_target(self.clone(), stage)
    }
}

/// Calls the hooks of one [`Stage`] directly, so that a test of a custom stage needs no [`StageGraph`](crate::StageGraph):
/// it states what the stage sees (its id, its lane, the markers it must wait for, the pool), calls a hook, and reads what
/// the stage did as plain data: what `run` answered (the marker it submitted, its host completion, the error), whether
/// it took its gate and how that gate was spent, and what it enqueued on the lanes (the pool is the test's own, such as
/// [`FakeDevice`](crate::testing::FakeDevice)'s).
///
/// Nothing is scheduled: a call runs the hook on the calling thread and returns. The effects that need a graph do not
/// happen. A gate released here does not run the stage again (the test calls [`run`](Self::run) again, as the graph
/// would), a failed gate does not fail a quantum (it is recorded in [`gate_events`](Self::gate_events)), and the hooks of
/// other stages are not called. A panic of the stage propagates to the test. Feature `testing`.
///
/// ```
/// use euhedral_graph::testing::{FakeDevice, GateEvent, StageHarness};
/// use euhedral_graph::{Stage, StageCx, StageError, Submitted};
/// use std::time::Duration;
///
/// // Enqueues 5 ms of work on its lane and submits the marker; fails on a quantum that is not positive.
/// struct Scale(Vec<std::sync::Arc<FakeDevice>>);
/// impl Stage<i64> for Scale {
///     fn run(&self, quantum: &i64, cx: &StageCx<'_>) -> Result<Submitted, StageError> {
///         if *quantum <= 0 {
///             return Err(StageError::Corrupt(format!("bad input {quantum}")));
///         }
///         let lane = cx.lane.expect("a stage that submits has a lane");
///         let marker = self.0[usize::from(lane.0)].submit(Duration::from_millis(5), cx.waits);
///         Ok(Submitted::marker(marker))
///     }
/// }
///
/// let (lanes, devices) = FakeDevice::pool(&["compute"]);
/// let stage = Scale(devices);
/// let harness = StageHarness::new(lanes).on_lane(euhedral_graph::LaneId(0));
///
/// let ran = harness.run(&stage, &3);
/// let marker = ran.result.unwrap().marker.expect("it submitted");
/// assert_eq!(marker.value, 1);
///
/// let failed = harness.run(&stage, &0);
/// assert_eq!(failed.result.unwrap_err(), StageError::Corrupt("bad input 0".into()));
/// assert!(!failed.held);
/// assert!(harness.gate_events().is_empty());
/// ```
pub struct StageHarness<L: DeviceQueue + ?Sized = dyn DeviceQueue> {
    lanes: Arc<LanePool<L>>,
    stage: StageId,
    lane: Option<LaneId>,
    waits: Vec<Marker>,
    log: Arc<GateLog>,
}

impl<L: DeviceQueue + ?Sized> fmt::Debug for StageHarness<L> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StageHarness")
            .field("stage", &self.stage)
            .field("lane", &self.lane)
            .field("waits", &self.waits)
            .finish_non_exhaustive()
    }
}

impl<L: DeviceQueue + ?Sized> StageHarness<L> {
    /// A harness over `lanes` for stage 0, with no lane (a CPU-only stage) and no markers to wait for.
    pub fn new(lanes: Arc<LanePool<L>>) -> Self {
        Self {
            lanes,
            stage: StageId(0),
            lane: None,
            waits: Vec::new(),
            log: Arc::new(GateLog::default()),
        }
    }

    /// The id the stage sees in [`StageCx::stage`].
    #[must_use]
    pub fn with_stage_id(mut self, stage: StageId) -> Self {
        self.stage = stage;
        self
    }

    /// The lane the stage is given ([`StageCx::lane`], [`StageCx::lane_handle`]).
    #[must_use]
    pub fn on_lane(mut self, lane: LaneId) -> Self {
        self.lane = Some(lane);
        self
    }

    /// The markers the stage must wait for on the device ([`StageCx::waits`]).
    #[must_use]
    pub fn waiting_for(mut self, waits: &[Marker]) -> Self {
        self.waits = waits.to_vec();
        self
    }

    /// Calls `stage.run(quantum, ..)` with a context built from the harness's inputs.
    pub fn run<Q>(&self, stage: &dyn Stage<Q, L>, quantum: &Q) -> StageRun {
        let held = AtomicBool::new(false);
        let gates = self.log.clone();
        let cx = StageCx::outside_graph(
            self.stage,
            self.lane,
            &self.waits,
            &self.lanes,
            &gates,
            &held,
        );
        let result = stage.run(quantum, &cx);
        StageRun {
            result,
            held: held.load(Ordering::Relaxed),
        }
    }

    /// Calls `stage.pass_gate(quantum, ..)`. A gate it takes is spent into [`gate_events`](Self::gate_events).
    pub fn pass_gate<Q>(&self, stage: &dyn Stage<Q, L>, quantum: &Q) -> PassGate {
        let taken = Cell::new(false);
        let gates = self.log.clone();
        let open = stage.pass_gate(quantum, GateRef::outside_graph(&gates, self.stage, &taken));
        PassGate {
            open,
            held: taken.get(),
        }
    }

    /// Calls `stage.abandoned(quantum, home)`. What it enqueues on `home` is read from the test's lane.
    pub fn abandoned<Q>(&self, stage: &dyn Stage<Q, L>, quantum: &Q, home: &L) {
        stage.abandoned(quantum, home);
    }

    /// Calls `stage.retired(quantum, committed)`.
    pub fn retired<Q>(&self, stage: &dyn Stage<Q, L>, quantum: &Q, committed: bool) {
        stage.retired(quantum, committed);
    }

    /// How the gates the stage took so far were spent, in order. A gate still held is not in it.
    pub fn gate_events(&self) -> Vec<GateEvent> {
        self.log
            .events
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicU32;
    use std::time::Duration;

    use super::*;
    use crate::testing::FakeDevice;

    /// Submits a 1 ms op on its lane after its waits, counts its hooks, fails on quantum 13, holds on quantum 7.
    struct Probe {
        devices: Vec<Arc<FakeDevice>>,
        gate: Mutex<Option<Gate>>,
        abandoned: AtomicU32,
        retired: Mutex<Vec<bool>>,
    }

    impl Probe {
        fn new(devices: Vec<Arc<FakeDevice>>) -> Self {
            Self {
                devices,
                gate: Mutex::new(None),
                abandoned: AtomicU32::new(0),
                retired: Mutex::new(Vec::new()),
            }
        }
    }

    impl Stage<u64> for Probe {
        fn run(&self, q: &u64, cx: &StageCx<'_>) -> Result<Submitted, StageError> {
            match *q {
                13 => Err(StageError::Fatal("unlucky".into())),
                7 => {
                    *self.gate.lock().unwrap() = Some(cx.hold());
                    Err(StageError::Held)
                }
                _ => {
                    let lane = cx.lane.unwrap();
                    Ok(Submitted::marker(
                        self.devices[usize::from(lane.0)]
                            .submit(Duration::from_millis(1), cx.waits),
                    ))
                }
            }
        }

        fn pass_gate(&self, q: &u64, gate: GateRef<'_>) -> bool {
            if *q == 5 {
                *self.gate.lock().unwrap() = Some(gate.hold());
                false
            } else {
                true
            }
        }

        fn abandoned(&self, _: &u64, home: &dyn DeviceQueue) {
            assert_eq!(home.name(), "b");
            self.abandoned.fetch_add(1, Ordering::Relaxed);
        }

        fn retired(&self, _: &u64, committed: bool) {
            self.retired.lock().unwrap().push(committed);
        }
    }

    fn harness() -> (StageHarness, Probe, Vec<Arc<FakeDevice>>) {
        let (lanes, devices) = FakeDevice::pool(&["a", "b"]);
        let probe = Probe::new(devices.clone());
        (StageHarness::new(lanes).on_lane(LaneId(1)), probe, devices)
    }

    #[test]
    fn run_shows_the_context_it_was_given_and_what_it_submitted() {
        let (harness, probe, devices) = harness();
        let earlier = devices[0].submit(Duration::from_millis(40), &[]);
        let harness = harness.waiting_for(&[earlier]);
        let ran = harness.run(&probe, &1);
        let marker = ran.result.unwrap().marker.unwrap();
        assert_eq!((marker.lane, marker.value), (LaneId(1), 1));
        // The stage waited behind `earlier`, so its work completes after it.
        assert!(devices[1].done_at_ns(marker) > devices[0].done_at_ns(earlier));
        assert!(!ran.held);
    }

    #[test]
    fn a_failing_stage_is_seen_as_failed() {
        let (harness, probe, _) = harness();
        let ran = harness.run(&probe, &13);
        assert_eq!(ran.result.unwrap_err(), StageError::Fatal("unlucky".into()));
    }

    #[test]
    fn a_held_run_shows_its_gate_and_how_it_is_spent() {
        let (harness, probe, _) = harness();
        let ran = harness.run(&probe, &7);
        assert!(ran.held);
        assert_eq!(ran.result.unwrap_err(), StageError::Held);
        assert!(harness.gate_events().is_empty(), "the gate is still out");
        probe.gate.lock().unwrap().take().unwrap().release();
        let again = harness.run(&probe, &7);
        assert!(again.held);
        probe
            .gate
            .lock()
            .unwrap()
            .take()
            .unwrap()
            .fail(StageError::Fatal("gate 2".into()));
        assert_eq!(
            harness.gate_events(),
            [
                GateEvent::Released,
                GateEvent::Failed(StageError::Fatal("gate 2".into()))
            ]
        );
    }

    #[test]
    fn a_gate_dropped_unspent_arrives_as_fatal_and_held_given_to_fail_too() {
        let (harness, probe, _) = harness();
        harness.run(&probe, &7);
        drop(probe.gate.lock().unwrap().take());
        harness.run(&probe, &7);
        probe
            .gate
            .lock()
            .unwrap()
            .take()
            .unwrap()
            .fail(StageError::Held);
        let events = harness.gate_events();
        assert!(matches!(
            &events[0],
            GateEvent::Failed(StageError::Fatal(_))
        ));
        assert!(matches!(
            &events[1],
            GateEvent::Failed(StageError::Fatal(_))
        ));
    }

    #[test]
    fn pass_gate_shows_whether_it_held() {
        let (harness, probe, _) = harness();
        let open = harness.pass_gate(&probe, &1);
        assert!(open.open && !open.held);
        let shut = harness.pass_gate(&probe, &5);
        assert!(!shut.open && shut.held);
        probe.gate.lock().unwrap().take().unwrap().release();
        assert_eq!(harness.gate_events(), [GateEvent::Released]);
    }

    #[test]
    #[should_panic(expected = "one gate per run")]
    fn a_second_hold_in_a_run_panics_as_in_a_graph() {
        struct Twice;
        impl Stage<u64> for Twice {
            fn run(&self, _: &u64, cx: &StageCx<'_>) -> Result<Submitted, StageError> {
                let first = cx.hold();
                let second = cx.hold();
                first.release();
                second.release();
                Err(StageError::Held)
            }
        }
        let (lanes, _) = FakeDevice::pool(&["a"]);
        StageHarness::new(lanes).run(&Twice, &0);
    }

    #[test]
    fn the_abandoned_and_retired_hooks_are_called() {
        let (harness, probe, devices) = harness();
        harness.abandoned(&probe, &1, &*devices[1] as &dyn DeviceQueue);
        harness.retired(&probe, &1, true);
        harness.retired(&probe, &1, false);
        assert_eq!(probe.abandoned.load(Ordering::Relaxed), 1);
        assert_eq!(*probe.retired.lock().unwrap(), [true, false]);
    }
}
