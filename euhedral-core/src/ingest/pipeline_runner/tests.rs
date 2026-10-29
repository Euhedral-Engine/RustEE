use super::*;
use crate::control_plane::testing::{PATIENCE, wait_until};
use crate::flow_control::testing::TestReceiver;
use crate::frames::{Failure, Status};
use crate::generics::{AbstractExecutor, CloneableObject, LatticeReceiver, LatticeTerminal};
use crate::impls::DefaultExecutor;
use euhedral_data_structures::sync::RecvError;
use std::sync::{Barrier, Mutex};
use std::thread;
use std::time::{Duration, Instant};

fn values<T>() -> Arc<Mutex<Vec<T>>> {
    Arc::new(Mutex::new(Vec::new()))
}

fn collect<T: Send + 'static>(values: &Arc<Mutex<Vec<T>>>) -> impl Fn(T) + Send + Sync + 'static {
    let values = values.clone();
    move |value| values.lock().unwrap().push(value)
}

fn sorted<T: Ord + Clone>(values: &Arc<Mutex<Vec<T>>>) -> Vec<T> {
    let mut values = values.lock().unwrap().clone();
    values.sort_unstable();
    values
}

/// The runner's delegate, with an executor attached that asks for unbounded demand.
fn executed<I: Send + 'static>(runner: &PipelineRunner<I>) -> Arc<dyn LatticeSource> {
    let source = runner.delegate();
    Arc::new(DefaultExecutor::new()).input(source.clone());
    source
}

/// The runner's delegate, with a receiver attached that only records.
fn recorded<I: Send + 'static>(runner: &PipelineRunner<I>) -> Arc<dyn LatticeSource> {
    let source = runner.delegate();
    source.clone().add_downstream(TestReceiver::new());
    source
}

/// Requests until nothing is queued.
fn drive<I>(runner: &PipelineRunner<I>, source: &Arc<dyn LatticeSource>) {
    for _ in 0..1_000 {
        if runner.size() == 0 {
            return;
        }
        source.request(1);
    }
    panic!("the runner never drained");
}

/// Pulls the next queued chain and keeps it.
fn hold(source: &Arc<dyn LatticeSource>) -> Box<dyn Frame> {
    let mut held = None;
    assert_eq!(
        source.pull(&mut |frame| held = Some(frame), &mut |_| false, 1),
        1
    );
    held.unwrap()
}

/// Checks `outcome` until the chain has ended or been dropped, or fails after [`PATIENCE`]: nothing waits on a
/// one-shot.
fn resolve(mut outcome: OneShotReceiver<Outcome>) -> Result<Outcome, RecvError> {
    let deadline = Instant::now() + PATIENCE;
    loop {
        match outcome.try_take() {
            Ok(Some(outcome)) => return Ok(outcome),
            Ok(None) => {
                assert!(
                    Instant::now() < deadline,
                    "timed out waiting for an outcome"
                );
                thread::sleep(Duration::from_millis(1));
            }
            Err(error) => return Err(error),
        }
    }
}

fn outcome_of(outcome: OneShotReceiver<Outcome>) -> Outcome {
    resolve(outcome).expect("an outcome")
}

fn status(outcome: OneShotReceiver<Outcome>) -> Status {
    outcome_of(outcome).status()
}

fn pending(outcome: &mut OneShotReceiver<Outcome>) -> bool {
    matches!(outcome.try_take(), Ok(None))
}

fn refused<I, T>(result: Result<T, AdmissionClosed<I>>) -> I {
    match result {
        Ok(_) => panic!("admission should be closed"),
        Err(AdmissionClosed(input)) => input,
    }
}

type Complete = dyn Fn(&PipelineRunner<i32>, &Arc<dyn LatticeSource>);

/// Holds whoever enters until released, or until [`PATIENCE`] runs out so a failing test cannot
/// hang.
#[derive(Default)]
struct Gate {
    entered: AtomicBool,
    released: AtomicBool,
}

impl Gate {
    fn new() -> Arc<Self> {
        Arc::default()
    }

    fn pass(&self) {
        self.entered.store(true, Ordering::Release);
        let deadline = Instant::now() + PATIENCE;
        while !self.released.load(Ordering::Acquire) && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(1));
        }
    }

    fn await_entered(&self) {
        wait_until("a stage to enter the gate", || {
            self.entered.load(Ordering::Acquire)
        });
    }

    fn release(&self) {
        self.released.store(true, Ordering::Release);
    }
}

/// Sets its flag when dropped, so threads waiting on it stop even if the test panics.
struct SetOnDrop<'a>(&'a AtomicBool);

impl Drop for SetOnDrop<'_> {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Release);
    }
}

#[test]
fn submissions_report_success_once_the_consumer_takes_the_output() {
    let output = values();
    let mut runner = PipelineRunner::new(
        &PipelineFrame::<i32>::builder()
            .fan_out(|value| value * 2)
            .fan_in(|value| value + 1),
        collect(&output),
        true,
    );
    let source = executed(&runner);
    let mut first = runner.submit(3).unwrap();
    runner.run(4).unwrap();
    assert!(pending(&mut first));
    drive(&runner, &source);
    assert_eq!(status(first), Status::Success);
    assert_eq!(sorted(&output), [7, 9]);
    assert_eq!(runner.manager.recycle_queue_size(), 2);

    let again = runner.submit(5).unwrap();
    drive(&runner, &source);
    assert_eq!(status(again), Status::Success);
    assert_eq!(sorted(&output), [7, 9, 11]);
    assert!(!runner.is_complete());
}

#[test]
fn a_filtered_chain_never_reaches_the_consumer() {
    let output = values();
    let mut runner = PipelineRunner::new(
        &PipelineFrame::<i32>::builder()
            .fan_out(|value| value)
            .filter_output(|value| value % 2 == 0),
        collect(&output),
        true,
    );
    let source = executed(&runner);
    let odd = runner.submit(1).unwrap();
    let even = runner.submit(2).unwrap();
    drive(&runner, &source);
    assert_eq!(status(odd), Status::Filtered);
    assert_eq!(status(even), Status::Success);
    assert_eq!(*output.lock().unwrap(), [2]);
}

#[test]
fn a_cancelling_stage_cancels_its_chain() {
    let output = values();
    let mut runner = PipelineRunner::new(
        &PipelineFrame::<i32>::builder().try_fan_out(|value: i32| {
            if value == 0 {
                Err(FrameError::Cancelled)
            } else {
                Ok(value)
            }
        }),
        collect(&output),
        true,
    );
    let source = executed(&runner);
    let cancelled = runner.submit(0).unwrap();
    let kept = runner.submit(1).unwrap();
    drive(&runner, &source);
    assert_eq!(status(cancelled), Status::Cancelled);
    assert_eq!(status(kept), Status::Success);
    assert_eq!(*output.lock().unwrap(), [1]);
}

#[test]
fn a_failing_stage_or_consumer_reports_its_failure() {
    let mut runner = PipelineRunner::new(
        &PipelineFrame::<i32>::builder().try_fan_out(|value: i32| {
            if value == 0 {
                Err(FrameError::failed("stage failed"))
            } else {
                Ok(value)
            }
        }),
        |_| panic!("consumer failed"),
        true,
    );
    let source = executed(&runner);
    let stage = runner.submit(0).unwrap();
    let consumer = runner.submit(1).unwrap();
    drive(&runner, &source);
    let outcome = outcome_of(stage);
    assert_eq!(outcome.status(), Status::Failed);
    assert_eq!(outcome.failure().unwrap().to_string(), "stage failed");
    let outcome = outcome_of(consumer);
    assert_eq!(outcome.status(), Status::Failed);
    assert!(
        outcome
            .failure()
            .unwrap()
            .to_string()
            .contains("consumer failed")
    );
    runner.complete_gracefully();
    assert!(runner.is_complete());
}

#[test]
fn the_consumer_runs_in_order_unless_it_may_run_in_parallel() {
    for parallel in [false, true] {
        let mut runner = PipelineRunner::new(&PipelineFrame::<i32>::builder(), drop, parallel);
        let source = recorded(&runner);
        runner.run(1).unwrap();
        let root = hold(&source);
        assert_eq!(root.header().is_ordered(), !parallel);
        root.do_finally_with_error(FrameError::Cancelled);
    }
}

#[test]
fn graceful_completion_waits_for_accepted_chains_and_refuses_new_ones() {
    let output = values();
    let mut runner = PipelineRunner::with_partitions(
        &PipelineFrame::<i32>::builder().fan_out(|value| value + 1),
        collect(&output),
        true,
        NonZeroUsize::new(2).unwrap(),
    );
    let source = executed(&runner);
    let accepted = runner.submit(1).unwrap();
    runner.complete_gracefully();
    assert!(!runner.is_complete());

    assert_eq!(refused(runner.run(2)), 2);
    assert_eq!(refused(runner.run_seeded(1, 3)), 3);
    assert_eq!(refused(runner.run_on(1, 4)), 4);
    assert_eq!(refused(runner.submit(5)), 5);
    assert_eq!(refused(runner.submit_seeded(1, 6)), 6);
    assert_eq!(refused(runner.submit_on(1, 7)), 7);
    assert_eq!(
        runner.run(8).unwrap_err().to_string(),
        "pipeline admission is closed"
    );

    drive(&runner, &source);
    assert_eq!(status(accepted), Status::Success);
    assert_eq!(*output.lock().unwrap(), [2]);
    assert!(runner.is_complete());
}

#[test]
fn graceful_completion_of_an_idle_runner_completes_at_once() {
    let runner = PipelineRunner::new(&PipelineFrame::<i32>::builder(), drop, true);
    let source = runner.delegate();
    let receiver = TestReceiver::new();
    source.clone().add_downstream(receiver.clone());
    source.request(5);
    assert_eq!(runner.demand(), 5);
    runner.control().complete_gracefully();
    assert!(runner.is_complete());
    assert!(runner.control().is_complete());
    assert!(receiver.is_completed());
    assert_eq!(runner.demand(), 0);
}

#[test]
fn completing_cancels_queued_chains_and_held_ones_when_they_return() {
    let complete_runner =
        |runner: &PipelineRunner<i32>, _: &Arc<dyn LatticeSource>| runner.complete();
    let complete_control =
        |runner: &PipelineRunner<i32>, _: &Arc<dyn LatticeSource>| runner.control().complete();
    let complete_delegate =
        |_: &PipelineRunner<i32>, source: &Arc<dyn LatticeSource>| source.complete();
    let completions: [&Complete; 3] = [&complete_runner, &complete_control, &complete_delegate];
    for complete in completions {
        let output = values();
        let mut runner = PipelineRunner::new(
            &PipelineFrame::<i32>::builder().fan_out(|value| value + 1),
            collect(&output),
            true,
        );
        let source = recorded(&runner);
        let mut active = runner.submit(1).unwrap();
        let stage = hold(&source);
        let queued = runner.submit(2).unwrap();

        complete(&runner, &source);
        assert!(runner.is_complete());
        assert!(!stage.is_alive());
        assert_eq!(status(queued), Status::Cancelled);
        assert!(pending(&mut active));
        assert_eq!(runner.size(), 0);

        DefaultExecutor::new().run(stage);
        assert_eq!(status(active), Status::Cancelled);
        assert!(output.lock().unwrap().is_empty());
        assert_eq!(refused(runner.submit(3)), 3);
        assert_eq!(runner.manager.recycle_queue_size(), 2);
    }
}

#[test]
fn completing_inside_a_pull_cancels_once_the_pull_lets_go() {
    let mut runner = PipelineRunner::new(&PipelineFrame::<i32>::builder(), drop, true);
    let source = recorded(&runner);
    let mut outcome = runner.submit(1).unwrap();
    let control = runner.control();
    let mut cancelled_inside = None;
    let pulled = source.pull(
        &mut |_| panic!("the pull should stop"),
        &mut |_| {
            control.complete();
            cancelled_inside = Some(!pending(&mut outcome));
            true
        },
        1,
    );
    assert_eq!(pulled, 0);
    assert_eq!(cancelled_inside, Some(false));
    assert_eq!(status(outcome), Status::Cancelled);
    assert_eq!(runner.size(), 0);
}

#[test]
fn every_submission_mode_reaches_the_partitions() {
    let output = values();
    let mut runner = PipelineRunner::with_partitions(
        &PipelineFrame::<i32>::builder(),
        collect(&output),
        true,
        NonZeroUsize::new(3).unwrap(),
    );
    assert_eq!(runner.partitions(), 3);
    let source = executed(&runner);
    runner.run(0).unwrap();
    runner.run_seeded(9, 1).unwrap();
    runner.run_on(2, 2).unwrap();
    let outcomes = [
        runner.submit(3).unwrap(),
        runner.submit_seeded(9, 4).unwrap(),
        runner.submit_on(1, 5).unwrap(),
    ];
    assert_eq!(runner.size(), 6);
    drive(&runner, &source);
    for outcome in outcomes {
        assert_eq!(status(outcome), Status::Success);
    }
    assert_eq!(sorted(&output), [0, 1, 2, 3, 4, 5]);
    runner.complete_gracefully();
    assert!(runner.is_complete());
}

#[test]
fn an_out_of_range_partition_panics_before_checking_out_a_chain() {
    let mut runner = PipelineRunner::with_partitions(
        &PipelineFrame::<i32>::builder(),
        drop,
        true,
        NonZeroUsize::new(3).unwrap(),
    );
    let run = panic::catch_unwind(AssertUnwindSafe(|| runner.run_on(3, 0)));
    let message = run.unwrap_err();
    assert_eq!(
        message.downcast_ref::<String>().unwrap(),
        "partition 3 is out of range for 3 partitions"
    );
    assert!(panic::catch_unwind(AssertUnwindSafe(|| runner.submit_on(3, 0))).is_err());
    runner.complete_gracefully();
    assert!(runner.is_complete());
}

#[test]
fn completion_during_a_request_leaves_no_demand() {
    let mut runner = PipelineRunner::new(&PipelineFrame::<i32>::builder(), drop, true);
    let source = executed(&runner);
    let outcome = runner.submit(1).unwrap();
    runner.complete_gracefully();
    source.request(1);
    assert_eq!(status(outcome), Status::Success);
    assert!(runner.is_complete());
    assert_eq!(runner.demand(), 0);
}

/// Fails when told of completion.
struct FailingCompletion;

impl LatticeTerminal for FailingCompletion {
    fn add_upstream(&self, _upstream: Arc<dyn LatticeSource>) {}
}

impl LatticeReceiver for FailingCompletion {
    fn push(&self, _frame: Box<dyn Frame>) {}

    fn on_complete(&self) {
        panic!("receiver failed");
    }

    fn on_error(&self, _error: Failure) {}
}

#[test]
fn a_failing_downstream_completion_still_cancels_queued_chains() {
    let mut runner = PipelineRunner::new(&PipelineFrame::<i32>::builder(), drop, true);
    runner
        .delegate()
        .add_downstream(Arc::new(FailingCompletion));
    let outcome = runner.submit(1).unwrap();
    assert!(panic::catch_unwind(AssertUnwindSafe(|| runner.complete())).is_err());
    assert_eq!(status(outcome), Status::Cancelled);
    assert_eq!(runner.size(), 0);
    assert!(runner.is_complete());
}

#[test]
fn admission_racing_completion_never_strands_a_submission() {
    for iteration in 0..200 {
        let mut runner = PipelineRunner::new(&PipelineFrame::<i32>::builder(), drop, true);
        let control = runner.control();
        let start = Barrier::new(2);
        let outcome = thread::scope(|scope| {
            scope.spawn(|| {
                start.wait();
                control.complete();
            });
            start.wait();
            runner.submit(iteration)
        });
        if let Ok(outcome) = outcome {
            assert_eq!(status(outcome), Status::Cancelled);
        }
        assert_eq!(runner.size(), 0);
        assert!(runner.is_complete());
    }
}

#[test]
fn completing_while_a_stage_runs_cancels_its_chain_when_it_returns() {
    let gate = Gate::new();
    let entering = gate.clone();
    let mut runner = PipelineRunner::new(
        &PipelineFrame::<i32>::builder().fan_out(move |value| {
            entering.pass();
            value
        }),
        |_| panic!("a cancelled chain reached the consumer"),
        true,
    );
    let source = executed(&runner);
    let mut active = runner.submit(1).unwrap();
    let mut queued = runner.submit(2).unwrap();
    thread::scope(|scope| {
        let source = source.clone();
        let requesting = scope.spawn(move || source.request(1));
        gate.await_entered();
        runner.complete();
        assert!(pending(&mut active));
        assert!(pending(&mut queued));
        gate.release();
        requesting.join().unwrap();
    });
    assert_eq!(status(active), Status::Cancelled);
    assert_eq!(status(queued), Status::Cancelled);
    assert_eq!(runner.size(), 0);
}

#[test]
fn graceful_completion_while_a_stage_runs_still_publishes_its_successor() {
    let gate = Gate::new();
    let entering = gate.clone();
    let output = values();
    let mut runner = PipelineRunner::new(
        &PipelineFrame::<i32>::builder()
            .fan_out(move |value| {
                entering.pass();
                value
            })
            .fan_out(|value| value * 10),
        collect(&output),
        true,
    );
    let source = executed(&runner);
    let accepted = runner.submit(4).unwrap();
    thread::scope(|scope| {
        let source = source.clone();
        let requesting = scope.spawn(move || source.request(1));
        gate.await_entered();
        runner.complete_gracefully();
        assert!(!runner.is_complete());
        gate.release();
        requesting.join().unwrap();
    });
    drive(&runner, &source);
    assert_eq!(status(accepted), Status::Success);
    assert_eq!(*output.lock().unwrap(), [40]);
    wait_until("the runner to complete", || runner.is_complete());
}

#[test]
fn chains_finish_on_many_threads_and_graceful_completion_waits_for_all() {
    const CHAINS: i32 = 2_000;
    let output = values();
    let mut runner = PipelineRunner::with_partitions(
        &PipelineFrame::<i32>::builder()
            .fan_out(|value| value + 1)
            .fan_in(|value| value * 2),
        collect(&output),
        true,
        NonZeroUsize::new(4).unwrap(),
    );
    let source = executed(&runner);
    let done = AtomicBool::new(false);
    thread::scope(|scope| {
        let _stop = SetOnDrop(&done);
        for _ in 0..3 {
            scope.spawn(|| {
                while !done.load(Ordering::Acquire) {
                    source.request(1);
                    thread::yield_now();
                }
            });
        }
        let outcomes: Vec<_> = (0..CHAINS)
            .map(|value| runner.submit(value).unwrap())
            .collect();
        runner.complete_gracefully();
        for outcome in outcomes {
            assert_eq!(status(outcome), Status::Success);
        }
        wait_until("the runner to complete", || runner.is_complete());
    });
    assert_eq!(
        sorted(&output),
        (0..CHAINS).map(|value| (value + 1) * 2).collect::<Vec<_>>()
    );
    assert_eq!(runner.size(), 0);
}

#[test]
fn outcomes_of_1000_chains_from_4_submitters_each_arrive_once() {
    const PER_SUBMITTER: i32 = 250;
    let output = values();
    let runner = Mutex::new(PipelineRunner::with_partitions(
        &PipelineFrame::<i32>::builder()
            .fan_out(|value| value + 1)
            .fan_in(|value| value * 2),
        collect(&output),
        true,
        NonZeroUsize::new(4).unwrap(),
    ));
    let source = executed(&runner.lock().unwrap());
    let done = AtomicBool::new(false);
    let arrived = thread::scope(|scope| {
        let _stop = SetOnDrop(&done);
        for _ in 0..2 {
            scope.spawn(|| {
                while !done.load(Ordering::Acquire) {
                    source.request(1);
                    thread::yield_now();
                }
            });
        }
        let submitters: Vec<_> = (0..4)
            .map(|submitter| {
                let runner = &runner;
                scope.spawn(move || {
                    let mut arrived = 0;
                    for value in submitter * PER_SUBMITTER..(submitter + 1) * PER_SUBMITTER {
                        let outcome = runner.lock().unwrap().submit(value).unwrap();
                        assert_eq!(status(outcome), Status::Success);
                        arrived += 1;
                    }
                    arrived
                })
            })
            .collect();
        submitters
            .into_iter()
            .map(|submitter| submitter.join().unwrap())
            .sum::<i32>()
    });
    assert_eq!(arrived, 4 * PER_SUBMITTER);
    assert_eq!(
        sorted(&output),
        (0..4 * PER_SUBMITTER)
            .map(|value| (value + 1) * 2)
            .collect::<Vec<_>>()
    );
}

/// Kills its runner, then fails, while a drain pushes to it.
struct KillingReceiver(PipelineControl);

impl LatticeTerminal for KillingReceiver {
    fn add_upstream(&self, _upstream: Arc<dyn LatticeSource>) {}
}

impl LatticeReceiver for KillingReceiver {
    fn push(&self, _frame: Box<dyn Frame>) {
        self.0.complete();
        panic!("receiver failed");
    }

    fn on_complete(&self) {}

    fn on_error(&self, _error: Failure) {}
}

#[test]
fn a_drain_that_fails_after_a_kill_still_cancels_the_chains_behind_it() {
    let mut runner = PipelineRunner::new(&PipelineFrame::<i32>::builder(), drop, true);
    let source = runner.delegate();
    source
        .clone()
        .add_downstream(Arc::new(KillingReceiver(runner.control())));
    let pushed = runner.submit(1).unwrap();
    let behind = runner.submit(2).unwrap();
    assert!(panic::catch_unwind(AssertUnwindSafe(|| source.request(2))).is_err());
    assert!(resolve(pushed).is_err());
    assert_eq!(status(behind), Status::Cancelled);
    assert_eq!(runner.size(), 0);
    assert!(runner.is_complete());
}
