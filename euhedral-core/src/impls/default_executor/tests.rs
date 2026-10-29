use super::*;
use crate::config::CloneConfig;
use crate::frames::{Failure, FrameHeader, PipelineFrame, Status};
use crate::generics::{CloneableObject, FrameSink, LatticeReceiver, LatticeSource};
use euhedral_hardware_utils::affinity::{self, PinnedThreadExecutor};
use euhedral_hardware_utils::util::CpuSet;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

type Log = Arc<Mutex<Vec<String>>>;

#[derive(Clone, Copy)]
enum Behaviour {
    Succeed,
    Cancel,
    Fail,
    Panic,
}

/// A frame that logs its lifecycle and behaves as scripted when executed.
struct Scripted {
    header: FrameHeader,
    behaviour: Behaviour,
    panic_in_finalizer: bool,
    log: Log,
}

impl Scripted {
    fn new(behaviour: Behaviour, log: &Log) -> Self {
        Self {
            header: FrameHeader::new(1),
            behaviour,
            panic_in_finalizer: false,
            log: log.clone(),
        }
    }

    fn dead(self) -> Self {
        Self {
            header: FrameHeader::with_kill_switch(1, Some(Arc::new(AtomicBool::new(true)))),
            ..self
        }
    }

    fn record(&self, event: impl Into<String>) {
        self.log.lock().unwrap().push(event.into());
    }
}

impl Frame for Scripted {
    fn header(&self) -> &FrameHeader {
        &self.header
    }

    fn header_mut(&mut self) -> &mut FrameHeader {
        &mut self.header
    }

    fn execute(&mut self) -> Result<(), FrameError> {
        self.record("execute");
        match self.behaviour {
            Behaviour::Succeed => Ok(()),
            Behaviour::Cancel => Err(FrameError::Cancelled),
            Behaviour::Fail => Err(FrameError::failed("scripted failure")),
            Behaviour::Panic => panic!("scripted panic"),
        }
    }

    fn do_finally(self: Box<Self>) {
        self.record("finally");
        if self.panic_in_finalizer {
            panic!("finalizer panic");
        }
    }

    fn do_finally_with_error(self: Box<Self>, error: FrameError) {
        self.record(format!("finally with error: {error}"));
    }
}

fn log() -> Log {
    Arc::new(Mutex::new(Vec::new()))
}

fn events(log: &Log) -> Vec<String> {
    log.lock().unwrap().clone()
}

/// Pushes its frames as demand arrives, and records the demand it was asked for.
#[derive(Default)]
struct TestSource {
    frames: Mutex<VecDeque<Box<dyn Frame>>>,
    downstream: Mutex<Option<Arc<dyn LatticeReceiver>>>,
    requested: AtomicU64,
}

impl TestSource {
    fn with(frames: Vec<Box<dyn Frame>>) -> Arc<Self> {
        Arc::new(Self {
            frames: Mutex::new(frames.into()),
            ..Self::default()
        })
    }

    fn deliver(&self) {
        let Some(downstream) = self.downstream.lock().unwrap().clone() else {
            return;
        };
        while self.requested.load(Ordering::Acquire) > 0 {
            let Some(frame) = self.frames.lock().unwrap().pop_front() else {
                return;
            };
            if self.requested.load(Ordering::Acquire) != u64::MAX {
                self.requested.fetch_sub(1, Ordering::AcqRel);
            }
            downstream.push(frame);
        }
    }
}

impl LatticeSource for TestSource {
    fn add_downstream(self: Arc<Self>, downstream: Arc<dyn LatticeReceiver>) {
        *self.downstream.lock().unwrap() = Some(downstream.clone());
        downstream.add_upstream(self);
    }

    fn pull(
        &self,
        _consumer: &mut dyn FnMut(Box<dyn Frame>),
        _stop: &mut dyn FnMut(&dyn Frame) -> bool,
        _demand: u64,
    ) -> u64 {
        0
    }

    fn request(&self, demand: u64) {
        self.requested.store(demand, Ordering::Release);
        self.deliver();
    }

    fn complete(&self) {}

    fn is_complete(&self) -> bool {
        false
    }
}

impl FrameSink for TestSource {
    fn offer(&self, frame: Box<dyn Frame>) -> Result<(), Box<dyn Frame>> {
        self.frames.lock().unwrap().push_back(frame);
        self.deliver();
        Ok(())
    }
}

#[test]
fn constructors_set_the_cpu() {
    assert_eq!(DefaultExecutor::new().cpu(), None);
    assert_eq!(DefaultExecutor::default().cpu(), None);
    assert_eq!(DefaultExecutor::on_cpu(3).cpu(), Some(3));
}

#[test]
fn a_clone_runs_on_the_first_effective_cpu() {
    let executor = DefaultExecutor::on_cpu(0);
    let config = CloneConfig::new("shard", 1, CpuSet::from_iter([5, 4]));
    assert_eq!(executor.clone_for(&config).cpu(), Some(4));
    let empty = CloneConfig::new("shard", 1, CpuSet::new());
    assert_eq!(executor.clone_for(&empty).cpu(), None);
}

#[test]
fn clone_object_makes_a_new_executor() {
    let executor: Arc<dyn CloneableObject> = Arc::new(DefaultExecutor::on_cpu(0));
    let clone = executor.clone_object(&CloneConfig::new("", 0, CpuSet::new()));
    assert!(!Arc::ptr_eq(&executor, &clone));
}

#[test]
fn clone_object_on_a_pinned_executor_makes_a_new_executor() {
    let Some(cpu) = affinity::base_mask().first() else {
        return;
    };
    let _turn = crate::control_plane::testing::pinned_turn();
    let pinned = PinnedThreadExecutor::get_or_create(cpu, "executor-test").unwrap();
    let executor: Arc<dyn CloneableObject> = Arc::new(DefaultExecutor::on_cpu(cpu));
    let config = CloneConfig::new("", 0, CpuSet::from_iter([cpu]));
    let clone = executor.clone_object_on(&config, &pinned);
    assert!(!Arc::ptr_eq(&executor, &clone));
}

#[test]
fn input_requests_unbounded_demand_and_runs_every_pushed_frame() {
    let log = log();
    let source = TestSource::with(vec![
        Box::new(Scripted::new(Behaviour::Succeed, &log)),
        Box::new(Scripted::new(Behaviour::Succeed, &log)),
    ]);
    Arc::new(DefaultExecutor::new()).input(source.clone());

    assert_eq!(source.requested.load(Ordering::Acquire), u64::MAX);
    assert_eq!(events(&log), ["execute", "finally", "execute", "finally"]);
}

#[test]
fn a_dead_frame_is_finalized_without_running() {
    let log = log();
    DefaultExecutor::new().run(Box::new(Scripted::new(Behaviour::Succeed, &log).dead()));
    assert_eq!(events(&log), ["finally"]);
}

#[test]
fn a_cancelled_frame_finishes_normally() {
    let log = log();
    DefaultExecutor::new().run(Box::new(Scripted::new(Behaviour::Cancel, &log)));
    assert_eq!(events(&log), ["execute", "finally"]);
}

#[test]
fn a_failed_frame_finishes_with_its_error() {
    let log = log();
    DefaultExecutor::new().run(Box::new(Scripted::new(Behaviour::Fail, &log)));
    assert_eq!(
        events(&log),
        [
            "execute",
            "finally with error: frame failed: scripted failure"
        ]
    );
}

#[test]
fn a_panicking_frame_finishes_with_the_panic_as_its_error() {
    let log = log();
    let executor = DefaultExecutor::new();
    executor.run(Box::new(Scripted::new(Behaviour::Panic, &log)));
    executor.run(Box::new(Scripted::new(Behaviour::Succeed, &log)));
    assert_eq!(
        events(&log),
        [
            "execute",
            "finally with error: frame failed: scripted panic",
            "execute",
            "finally"
        ]
    );
}

#[test]
fn a_panicking_finalizer_does_not_escape_the_executor() {
    let log = log();
    let executor = DefaultExecutor::new();
    let frame = Scripted {
        panic_in_finalizer: true,
        ..Scripted::new(Behaviour::Succeed, &log)
    };
    executor.run(Box::new(frame));
    executor.run(Box::new(Scripted::new(Behaviour::Succeed, &log)));
    assert_eq!(events(&log), ["execute", "finally", "execute", "finally"]);
}

#[test]
fn a_custom_executor_supplies_execute_and_clone() {
    struct Counting {
        cpu: Option<usize>,
        executed: Arc<AtomicUsize>,
    }

    impl AbstractExecutor for Counting {
        fn cpu(&self) -> Option<usize> {
            self.cpu
        }

        fn execute(&self, frame: &mut dyn Frame) -> Result<(), FrameError> {
            self.executed.fetch_add(1, Ordering::Relaxed);
            frame.execute()
        }

        fn hook_on_clone(&self, cpu: Option<usize>) -> Self {
            Self {
                cpu,
                executed: self.executed.clone(),
            }
        }
    }

    let executed = Arc::new(AtomicUsize::new(0));
    let original = Counting {
        cpu: None,
        executed: executed.clone(),
    };
    let clone = Arc::new(original.clone_for(&CloneConfig::new("", 0, CpuSet::from_iter([2]))));
    assert_eq!(clone.cpu(), Some(2));

    let log = log();
    let source = TestSource::with(vec![
        Box::new(Scripted::new(Behaviour::Succeed, &log)),
        Box::new(Scripted::new(Behaviour::Succeed, &log).dead()),
    ]);
    clone.input(source);
    assert_eq!(
        executed.load(Ordering::Relaxed),
        1,
        "dead frames never reach execute"
    );
    assert_eq!(events(&log), ["execute", "finally", "finally"]);
}

#[test]
fn an_attached_executor_drives_a_pipeline_to_completion() {
    let source = TestSource::with(Vec::new());
    Arc::new(DefaultExecutor::new()).input(source.clone());

    let output = Arc::new(Mutex::new(Vec::new()));
    let consumer_output = output.clone();
    let mut chains = PipelineFrame::<u32>::builder()
        .fan_out(|value| value + 1)
        .fan_in(|value| value * 10)
        .compose_fanned_out(
            source.clone(),
            move |value| consumer_output.lock().unwrap().push(value),
            Arc::new(AtomicBool::new(false)),
        );
    let statuses = Arc::new(Mutex::new(Vec::new()));
    for input in 0..3 {
        let mut chain = chains.get_or_create(input);
        let statuses = statuses.clone();
        chain.observe(move |outcome| statuses.lock().unwrap().push(outcome.status()));
        assert!(source.offer(chain).is_ok());
    }

    assert_eq!(*output.lock().unwrap(), [10, 20, 30]);
    assert_eq!(*statuses.lock().unwrap(), [Status::Success; 3]);
}

#[test]
fn a_panicking_pipeline_stage_still_reports_its_outcome() {
    let source = TestSource::with(Vec::new());
    Arc::new(DefaultExecutor::new()).input(source.clone());
    let mut chains = PipelineFrame::<u32>::builder()
        .fan_out(|value: u32| -> u32 { panic!("stage {value} broke") })
        .compose_fanned_out(source.clone(), |_| {}, Arc::new(AtomicBool::new(false)));
    let outcome: Arc<Mutex<Option<Failure>>> = Arc::new(Mutex::new(None));
    let mut chain = chains.get_or_create(7);
    let slot = outcome.clone();
    chain.observe(move |outcome| {
        if let crate::frames::Outcome::Failed(failure) = outcome {
            *slot.lock().unwrap() = Some(failure);
        }
    });

    assert!(source.offer(chain).is_ok());
    assert_eq!(
        outcome.lock().unwrap().as_ref().unwrap().to_string(),
        "stage 7 broke"
    );
    assert_eq!(chains.recycle_queue_size(), 1);
}
