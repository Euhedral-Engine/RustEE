//! [`StageHarness`]: calls a [`Stage`]'s hooks directly, with no graph, lattice or workers.

use std::cell::Cell;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use crate::lanes::{DeviceQueue, LaneId, LanePool, Marker};
use euhedral_data_structures::sync::HotScope;

use crate::stage_graph::{
    Gate, GateRef, GateSource, GateTarget, Stage, StageCx, StageError, StageId, Submitted,
    foreign_marker, gate_taken_but_not_held, held_at_gate_without_one, held_without_gate, merge,
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
///
/// `result` is only what the stage answered; a graph also judges the answer, and fails the quantum with
/// [`StageError::Fatal`] when the stage broke the contract of [`Stage::run`]: `Err(Held)` answered with no gate taken,
/// a gate taken and `Ok` or `Retry` answered (any other error stands as the cause), or a marker that is not of the
/// pool. [`contract_breach`](Self::contract_breach) says which, as the graph would.
#[derive(Debug)]
pub struct HarnessRun {
    /// What `run` answered: the marker and host completion it submitted, or the error. No answer is a failure here:
    /// `Err(Held)` and `Err(Retry)` are only answers (see the type's docs for what a graph makes of them).
    pub result: Result<Submitted, StageError>,
    /// Whether the run took its gate ([`StageCx::hold`]).
    pub held: bool,
    breach: Option<StageError>,
}

impl HarnessRun {
    /// The [`StageError::Fatal`] a graph would fail the quantum with for this answer because the stage broke the
    /// contract of [`Stage::run`], or `None` when the answer is no breach (an error the stage returned itself is not
    /// one; the graph fails the quantum with it as the cause, and `Retry` retries). A breach is: `Held` answered
    /// without a gate taken; a gate taken and `Ok` or `Retry` answered; a marker that is no lane of the pool.
    ///
    /// It is derived with the graph's own conditions and messages. A panic in `run` is not judged here: it propagates
    /// to the test, and a graph fails the quantum with it.
    pub fn contract_breach(&self) -> Option<&StageError> {
        self.breach.as_ref()
    }
}

/// What [`StageHarness::pass_gate`] did.
///
/// A graph fails the quantum with [`StageError::Fatal`] when `pass_gate` returns `false` without taking a gate
/// ([`contract_breach`](Self::contract_breach)), and when it panics (here the panic propagates to the test).
#[derive(Debug)]
pub struct PassGate {
    /// What `pass_gate` answered: `true` publishes the stage.
    pub open: bool,
    /// Whether it took a gate ([`GateRef::hold`]); in a graph that holds the stage whatever `open` says.
    pub held: bool,
}

impl PassGate {
    /// The [`StageError::Fatal`] a graph would fail the quantum with because `pass_gate` returned `false` without
    /// taking a gate (it holds the stage, and nothing would ever publish it), or `None`.
    pub fn contract_breach(&self) -> Option<StageError> {
        (!self.open && !self.held).then(held_at_gate_without_one)
    }
}

/// Where the gates of a harness are spent.
#[derive(Debug, Default)]
struct GateLog {
    events: Mutex<Vec<GateEvent>>,
    /// The stage each event is of, in the same order.
    stages: Mutex<Vec<StageId>>,
}

impl GateLog {
    fn push(&self, stage: StageId, event: GateEvent) {
        self.stages
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(stage);
        self.events
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(event);
    }
}

impl GateTarget for GateLog {
    fn release_gate(self: Arc<Self>, stage: StageId) {
        self.push(stage, GateEvent::Released);
    }

    fn fail_gate(self: Arc<Self>, stage: StageId, error: StageError) {
        self.push(stage, GateEvent::Failed(error));
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
/// other stages are not called. A panic of the stage propagates to the test (a graph fails the quantum with it). Feature
/// `testing`.
///
/// What does run is what a graph does around the hook: `run` and `pass_gate` are called inside a
/// [`HotScope`](euhedral_data_structures::sync::HotScope), so a lifecycle lock taken in them is a violation here as in a
/// graph, and the markers to wait for are normalised as the graph does (see [`waiting_for`](Self::waiting_for)).
///
/// A graph also judges the answers, and fails the quantum with [`StageError::Fatal`] where a stage broke the contract:
/// `Held` answered without a gate taken, a gate taken and `Ok` or `Retry` answered, a marker that is no lane of the
/// pool, `pass_gate` returning `false` without taking a gate, and a panic in either hook. The harness does not fail
/// anything; [`HarnessRun::contract_breach`] and [`PassGate::contract_breach`] say what a graph would do (a panic
/// propagates to the test).
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
    inputs: Vec<Marker>,
    log: Arc<GateLog>,
}

impl<L: DeviceQueue + ?Sized> fmt::Debug for StageHarness<L> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StageHarness")
            .field("stage", &self.stage)
            .field("lane", &self.lane)
            .field("waits", &self.inputs)
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
            inputs: Vec::new(),
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

    /// The markers the stage must wait for on the device, as the graph would find them: [`StageCx::waits`] holds the
    /// latest marker per lane (merged as the graph merges), without those of the stage's own lane (a lane runs in
    /// order, so it needs no wait for itself). The stage's lane is the one in [`on_lane`](Self::on_lane), whatever the
    /// order of the calls.
    #[must_use]
    pub fn waiting_for(mut self, waits: &[Marker]) -> Self {
        self.inputs.clear();
        for &marker in waits {
            merge(&mut self.inputs, marker);
        }
        self
    }

    /// Calls `stage.run(quantum, ..)` with a context built from the harness's inputs.
    pub fn run<Q>(&self, stage: &dyn Stage<Q, L>, quantum: &Q) -> HarnessRun {
        let held = AtomicBool::new(false);
        let gates = self.log.clone();
        // As the graph: the latest marker per lane, the stage's own lane left out.
        let waits: Vec<Marker> = self
            .inputs
            .iter()
            .copied()
            .filter(|m| Some(m.lane) != self.lane)
            .collect();
        let cx = StageCx::outside_graph(self.stage, self.lane, &waits, &self.lanes, &gates, &held);
        let result = {
            // As the graph's `run_stage`: a lifecycle lock in `run` is a violation.
            let _hot = HotScope::enter();
            stage.run(quantum, &cx)
        };
        let held = held.load(Ordering::Relaxed);
        // The graph's judgement (`stage_ended`, `held_ended`), with its conditions and messages.
        let breach = match (&result, held) {
            (Ok(_) | Err(StageError::Retry), true) => Some(gate_taken_but_not_held()),
            (Err(StageError::Held), false) => Some(held_without_gate()),
            (Ok(submitted), false) => foreign_marker(&self.lanes, submitted),
            _ => None,
        };
        HarnessRun {
            result,
            held,
            breach,
        }
    }

    /// Calls `stage.pass_gate(quantum, ..)`. A gate it takes is spent into [`gate_events`](Self::gate_events).
    pub fn pass_gate<Q>(&self, stage: &dyn Stage<Q, L>, quantum: &Q) -> PassGate {
        let taken = Cell::new(false);
        let gates = self.log.clone();
        let open = {
            // As the graph's `arrive`, which offers a stage from inside a hot scope.
            let _hot = HotScope::enter();
            stage.pass_gate(quantum, GateRef::outside_graph(&gates, self.stage, &taken))
        };
        PassGate {
            open,
            held: taken.get(),
        }
    }

    /// Calls `stage.abandoned(quantum, home)` with the queue of lane `home` of the harness's pool, as the graph passes
    /// its home lane. What the hook enqueued is read from the test's own device. (`Stage::retired` takes nothing the
    /// harness holds: call it on the stage.)
    pub fn abandoned<Q>(&self, stage: &dyn Stage<Q, L>, quantum: &Q, home: LaneId) {
        stage.abandoned(quantum, &**self.lanes.lane(home));
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

    use euhedral_data_structures::sync::LifecycleMutex;

    use super::*;
    use crate::testing::FakeDevice;

    /// Submits a 1 ms op on its lane after its waits, counts its hooks, fails on quantum 13, holds on quantum 7.
    struct Probe {
        devices: Vec<Arc<FakeDevice>>,
        gate: Mutex<Option<Gate>>,
        abandoned: AtomicU32,
        /// The `cx.stage` of every run, and the `cx.waits` of the last.
        seen: Mutex<Vec<StageId>>,
        waits: Mutex<Vec<Marker>>,
    }

    impl Probe {
        fn new(devices: Vec<Arc<FakeDevice>>) -> Self {
            Self {
                devices,
                gate: Mutex::new(None),
                abandoned: AtomicU32::new(0),
                seen: Mutex::new(Vec::new()),
                waits: Mutex::new(Vec::new()),
            }
        }
    }

    impl Stage<u64> for Probe {
        fn run(&self, q: &u64, cx: &StageCx<'_>) -> Result<Submitted, StageError> {
            self.seen.lock().unwrap().push(cx.stage);
            *self.waits.lock().unwrap() = cx.waits.to_vec();
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
    fn the_abandoned_hook_is_called_with_the_queue_of_the_lane_given() {
        let (harness, probe, _) = harness();
        harness.abandoned(&probe, &1, LaneId(1));
        assert_eq!(probe.abandoned.load(Ordering::Relaxed), 1);
    }

    #[test]
    #[should_panic(expected = "assertion")]
    fn the_abandoned_hook_gets_the_lane_it_is_given_not_another() {
        let (harness, probe, _) = harness();
        harness.abandoned(&probe, &1, LaneId(0));
    }

    #[test]
    fn the_stage_sees_the_id_it_was_given() {
        let (lanes, devices) = FakeDevice::pool(&["a", "b"]);
        let probe = Probe::new(devices);
        StageHarness::new(lanes.clone())
            .on_lane(LaneId(1))
            .with_stage_id(StageId(9))
            .run(&probe, &1);
        StageHarness::new(lanes).on_lane(LaneId(1)).run(&probe, &1);
        assert_eq!(*probe.seen.lock().unwrap(), [StageId(9), StageId(0)]);
    }

    #[test]
    fn a_gate_is_spent_for_the_stage_id_of_the_harness() {
        let (lanes, devices) = FakeDevice::pool(&["a", "b"]);
        let probe = Probe::new(devices);
        let harness = StageHarness::new(lanes)
            .on_lane(LaneId(1))
            .with_stage_id(StageId(4));
        // A gate taken in `run`, and one taken in `pass_gate`.
        assert!(harness.run(&probe, &7).held);
        probe.gate.lock().unwrap().take().unwrap().release();
        assert!(harness.pass_gate(&probe, &5).held);
        probe.gate.lock().unwrap().take().unwrap().release();
        let stages = harness.log.stages.lock().unwrap().clone();
        assert_eq!(stages, [StageId(4), StageId(4)]);
    }

    #[test]
    fn waits_are_normalised_as_the_graph_does() {
        let (harness, probe, devices) = harness();
        let a1 = devices[0].submit(Duration::from_millis(1), &[]);
        let a2 = devices[0].submit(Duration::from_millis(1), &[]);
        let b1 = devices[1].submit(Duration::from_millis(1), &[]);
        // Two markers of lane a (the later covers the earlier), one of the stage's own lane b (dropped), whichever
        // order the builder is called in.
        let harness = harness.waiting_for(&[a2, b1, a1]);
        harness.run(&probe, &1);
        assert_eq!(*probe.waits.lock().unwrap(), [a2]);
    }

    /// Takes a lifecycle lock in `run` and in `pass_gate`.
    struct Locker;

    impl Stage<u64> for Locker {
        fn run(&self, _: &u64, _: &StageCx<'_>) -> Result<Submitted, StageError> {
            let lock = LifecycleMutex::new(0);
            drop(lock.lock());
            Ok(Submitted::default())
        }

        fn pass_gate(&self, _: &u64, _: GateRef<'_>) -> bool {
            let lock = LifecycleMutex::new(0);
            drop(lock.lock());
            true
        }
    }

    #[cfg(debug_assertions)]
    mod hot {
        use std::panic::Location;

        use euhedral_data_structures::sync::HotScope;

        use super::*;

        static PLACES: Mutex<Vec<String>> = Mutex::new(Vec::new());
        static TURN: Mutex<()> = Mutex::new(());

        fn record(at: &'static Location<'static>) {
            if at.file() == file!() {
                PLACES
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .push(at.to_string());
            } else {
                HotScope::report_default(at);
            }
        }

        fn violations(provoke: impl FnOnce()) -> usize {
            let _turn = TURN.lock().unwrap_or_else(PoisonError::into_inner);
            PLACES
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .clear();
            {
                let _hook = HotScope::install_violation_hook(record);
                provoke();
            }
            let places = PLACES.lock().unwrap_or_else(PoisonError::into_inner);
            places.len()
        }

        #[test]
        fn a_lifecycle_lock_in_run_is_a_violation_as_in_a_graph() {
            let (lanes, _) = FakeDevice::pool(&["a"]);
            let harness = StageHarness::new(lanes);
            assert_eq!(
                violations(|| {
                    harness.run(&Locker, &0);
                }),
                1
            );
        }

        #[test]
        fn a_lifecycle_lock_in_pass_gate_is_a_violation_as_in_a_graph() {
            let (lanes, _) = FakeDevice::pool(&["a"]);
            let harness = StageHarness::new(lanes);
            assert_eq!(
                violations(|| {
                    harness.pass_gate(&Locker, &0);
                }),
                1
            );
        }
    }

    /// Answers as its quantum says: 0 `Held` with no gate, 1 `Held` with a gate, 2 `Ok` with a gate, 3 `Retry` with a
    /// gate, 4 `Retry` with none, 5 `Ok` with a marker of another pool, 6 `Ok` with a marker of the pool, 7 an error
    /// with a gate taken.
    struct Answers {
        own: Arc<FakeDevice>,
        other: Arc<FakeDevice>,
        gate: Mutex<Option<Gate>>,
    }

    impl Stage<u64> for Answers {
        fn run(&self, q: &u64, cx: &StageCx<'_>) -> Result<Submitted, StageError> {
            if matches!(*q, 1..=3 | 7) {
                *self.gate.lock().unwrap() = Some(cx.hold());
            }
            match *q {
                0 | 1 => Err(StageError::Held),
                2 | 6 => Ok(Submitted::marker(self.own.submit(Duration::ZERO, &[]))),
                3 | 4 => Err(StageError::Retry),
                5 => Ok(Submitted::marker(self.other.submit(Duration::ZERO, &[]))),
                _ => Err(StageError::Corrupt("mine".into())),
            }
        }

        fn pass_gate(&self, q: &u64, gate: GateRef<'_>) -> bool {
            if *q == 1 {
                *self.gate.lock().unwrap() = Some(gate.hold());
            }
            *q == 2
        }
    }

    fn answers() -> (StageHarness, Answers) {
        let (lanes, devices) = FakeDevice::pool(&["a"]);
        let (_, other) = FakeDevice::pool(&["z"]);
        let stage = Answers {
            own: devices[0].clone(),
            other: other[0].clone(),
            gate: Mutex::new(None),
        };
        (StageHarness::new(lanes).on_lane(LaneId(0)), stage)
    }

    fn breach(q: u64) -> Option<StageError> {
        let (harness, stage) = answers();
        let ran = harness.run(&stage, &q);
        let breach = ran.contract_breach().cloned();
        drop(stage.gate.lock().unwrap().take());
        breach
    }

    #[test]
    fn the_run_says_what_a_graph_makes_of_the_answer() {
        let fatal = |text: &str| Some(StageError::Fatal(text.into()));
        assert_eq!(
            breach(0),
            fatal("a stage answered Held without taking a gate")
        );
        assert_eq!(breach(1), None);
        assert_eq!(
            breach(2),
            fatal("a stage that took a gate in run did not answer Held")
        );
        assert_eq!(
            breach(3),
            fatal("a stage that took a gate in run did not answer Held")
        );
        assert_eq!(breach(4), None);
        assert!(
            matches!(breach(5), Some(StageError::Fatal(m)) if m.contains("is no lane 0 of the lane pool"))
        );
        assert_eq!(breach(6), None);
        assert_eq!(breach(7), None, "the stage's own error stands");
    }

    #[test]
    fn pass_gate_says_what_a_graph_makes_of_the_answer() {
        let (harness, stage) = answers();
        // 0: false without a gate; 1: false with a gate; 2: true.
        assert_eq!(
            harness.pass_gate(&stage, &0).contract_breach(),
            Some(StageError::Fatal(
                "a stage held itself at its gate without taking one".into()
            ))
        );
        assert_eq!(harness.pass_gate(&stage, &1).contract_breach(), None);
        assert_eq!(harness.pass_gate(&stage, &2).contract_breach(), None);
        drop(stage.gate.lock().unwrap().take());
    }
}
