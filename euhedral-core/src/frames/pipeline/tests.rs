use super::*;
use crate::frames::testing::run;
use std::collections::{HashSet, VecDeque};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Mutex;
use std::sync::atomic::AtomicUsize;
use std::thread;
use std::time::{Duration, Instant};

type Hook = Box<dyn Fn() + Send + Sync>;

/// A FIFO sink that can be closed, and can run a hook before rejecting a frame.
#[derive(Default)]
struct TestSink {
    queue: Mutex<VecDeque<Box<dyn Frame>>>,
    closed: AtomicBool,
    on_reject: Option<Hook>,
}

impl TestSink {
    fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    fn rejecting(on_reject: impl Fn() + Send + Sync + 'static) -> Arc<Self> {
        Arc::new(Self {
            closed: AtomicBool::new(true),
            on_reject: Some(Box::new(on_reject)),
            ..Self::default()
        })
    }

    fn close(&self) {
        self.closed.store(true, Ordering::Release);
    }

    fn len(&self) -> usize {
        self.queue.lock().unwrap().len()
    }

    fn poll(&self) -> Option<Box<dyn Frame>> {
        self.queue.lock().unwrap().pop_front()
    }
}

impl FrameSink for TestSink {
    fn offer(&self, frame: Box<dyn Frame>) -> Result<(), Box<dyn Frame>> {
        if self.closed.load(Ordering::Acquire) {
            if let Some(hook) = &self.on_reject {
                hook();
            }
            return Err(frame);
        }
        self.queue.lock().unwrap().push_back(frame);
        Ok(())
    }
}

/// What one scheduled stage looked like when it ran.
#[derive(Debug)]
struct Step {
    id_hash: u64,
    routing_hash: u64,
    ordered: bool,
}

/// Runs a chain to its end, executing every stage the sink receives.
fn execute_pipeline(root: Box<dyn Frame>, sink: &TestSink) -> Vec<Step> {
    let mut steps = Vec::new();
    let mut current = Some(root);
    while let Some(frame) = current {
        steps.push(Step {
            id_hash: frame.header().id_hash(),
            routing_hash: frame.header().routing_hash(),
            ordered: frame.header().is_ordered(),
        });
        run(frame);
        current = sink.poll();
    }
    steps
}

fn execute(mut frame: Box<dyn Frame>) {
    let _ = frame.execute();
    frame.do_finally();
}

fn results<T>() -> Arc<Mutex<Vec<T>>> {
    Arc::new(Mutex::new(Vec::new()))
}

fn collect<T: Send + 'static>(results: &Arc<Mutex<Vec<T>>>) -> impl Fn(T) + Send + Sync + 'static {
    let results = results.clone();
    move |value| results.lock().unwrap().push(value)
}

fn switch() -> Arc<AtomicBool> {
    Arc::new(AtomicBool::new(false))
}

fn address<I>(chain: &PipelineFrame<I>) -> usize {
    chain as *const PipelineFrame<I> as usize
}

/// An observer slot and the closure that fills it.
fn observer() -> (
    Arc<Mutex<Option<Outcome>>>,
    impl FnOnce(Outcome) + Send + 'static,
) {
    let slot = Arc::new(Mutex::new(None));
    let fill = slot.clone();
    (slot, move |outcome| *fill.lock().unwrap() = Some(outcome))
}

fn status(slot: &Arc<Mutex<Option<Outcome>>>) -> Option<Status> {
    slot.lock().unwrap().as_ref().map(Outcome::status)
}

#[test]
fn composes_ordinary_closures_across_types() {
    let sink = TestSink::new();
    let output = results();
    let mut manager = PipelineFrame::<i32>::builder()
        .fan_out(|value| value * 2)
        .fan_out(|value| value.to_string())
        .fan_out(|value| format!("result={value}"))
        .compose_fanned_out(sink.clone(), collect(&output), switch());
    let root = manager.get_or_create(21);
    let id_hash = root.header().id_hash();
    assert_eq!(root.stage_count(), 4);

    let steps = execute_pipeline(root, &sink);

    assert_eq!(*output.lock().unwrap(), ["result=42"]);
    assert_eq!(steps.len(), 4);
    assert!(steps.iter().all(|step| step.id_hash == id_hash));
    assert!(steps.iter().all(|step| !step.ordered));
    let distinct: HashSet<u64> = steps.iter().map(|step| step.routing_hash).collect();
    assert_eq!(distinct.len(), 4);
}

#[test]
fn every_stage_is_ordered_or_parallel_as_declared() {
    let sink = TestSink::new();
    let output = results();
    let mut manager = PipelineFrame::<i32>::builder()
        .fan_in(|value| value * 2)
        .fan_out(|value| value.to_string())
        .fan_in(|value| format!("result={value}"))
        .compose_fanned_out(sink.clone(), collect(&output), switch());
    let root = manager.get_or_create(21);
    let id_hash = root.header().id_hash();

    let steps = execute_pipeline(root, &sink);

    assert_eq!(*output.lock().unwrap(), ["result=42"]);
    let ordered: Vec<bool> = steps.iter().map(|step| step.ordered).collect();
    assert_eq!(ordered, [true, false, true, false]);
    assert!(steps.iter().all(|step| step.id_hash == id_hash));
}

#[test]
fn a_consumer_only_pipeline_can_be_ordered() {
    let sink = TestSink::new();
    let output = results();
    let mut manager =
        PipelineFrame::<i32>::builder().compose_fanned_in(sink.clone(), collect(&output), switch());
    let terminal = manager.get_or_create(7);
    assert!(terminal.header().is_ordered());
    assert_eq!(terminal.stage_count(), 1);

    execute(terminal);
    assert_eq!(*output.lock().unwrap(), [7]);
}

#[test]
fn a_consumer_only_pipeline_returns_its_chain_to_the_pool() {
    let sink = TestSink::new();
    let output = results();
    let mut manager = PipelineFrame::<String>::builder().compose_fanned_out(
        sink.clone(),
        collect(&output),
        switch(),
    );
    let root = manager.get_or_create("value".to_owned());
    assert!(!root.header().is_ordered());

    execute(root);
    assert_eq!(*output.lock().unwrap(), ["value"]);
    assert_eq!(manager.recycle_queue_size(), 1);
    assert_eq!(sink.len(), 0);
}

#[test]
fn the_pool_reuses_a_finished_chain_with_new_input() {
    let sink = TestSink::new();
    let output = results();
    let pipeline = PipelineFrame::<i32>::builder().fan_in(|value| value * 2);
    let mut manager = pipeline.compose_fanned_in(sink.clone(), collect(&output), switch());

    let first = manager.get_or_create(3);
    let first_address = address(&first);
    execute_pipeline(first, &sink);
    let reused = manager.get_or_create(5);
    assert_eq!(address(&reused), first_address);
    assert!(reused.header().is_ordered());
    assert_eq!(reused.stage(), 0);
    execute_pipeline(reused, &sink);

    assert_eq!(*output.lock().unwrap(), [6, 10]);
}

#[test]
fn one_builder_composes_independent_pools() {
    let sink = TestSink::new();
    let pipeline = PipelineFrame::<i32>::builder().fan_in(|value| value * 2);
    let (first_switch, second_switch) = (switch(), switch());
    let (first_output, second_output) = (results(), results());
    let mut first_manager =
        pipeline.compose_fanned_out(sink.clone(), collect(&first_output), first_switch.clone());
    let mut second_manager =
        pipeline.compose_fanned_out(sink.clone(), collect(&second_output), second_switch.clone());

    let first = first_manager.get_or_create(3);
    let second = second_manager.get_or_create(5);
    assert!(first.recycler.as_ref().unwrap().belongs_to(&first_manager));
    assert!(
        second
            .recycler
            .as_ref()
            .unwrap()
            .belongs_to(&second_manager)
    );
    assert!(Arc::ptr_eq(
        first.header().kill_switch().unwrap(),
        &first_switch
    ));
    assert!(Arc::ptr_eq(
        second.header().kill_switch().unwrap(),
        &second_switch
    ));

    first.kill();
    assert!(!first.is_alive());
    assert!(second.is_alive());
    assert!(!second_switch.load(Ordering::Acquire));

    first.do_finally();
    assert_eq!(first_manager.recycle_queue_size(), 1);
    assert_eq!(second_manager.recycle_queue_size(), 0);

    execute_pipeline(second, &sink);
    let reused = second_manager.get_or_create(7);
    execute_pipeline(reused, &sink);
    assert_eq!(*second_output.lock().unwrap(), [10, 14]);
    assert!(first_output.lock().unwrap().is_empty());
}

#[test]
fn branches_of_a_shared_prefix_stay_independent() {
    let sink = TestSink::new();
    let prefix = PipelineFrame::<i32>::builder().fan_in(|value| value + 1);
    let doubled = prefix.clone().fan_out(|value| value * 2);
    let labeled = prefix.fan_in(|value| format!("value={value}"));
    let (doubled_output, labeled_output) = (results(), results());
    let mut doubled_manager =
        doubled.compose_fanned_out(sink.clone(), collect(&doubled_output), switch());
    let mut labeled_manager =
        labeled.compose_fanned_in(sink.clone(), collect(&labeled_output), switch());

    let doubled_root = doubled_manager.get_or_create(2);
    let doubled_address = address(&doubled_root);
    execute_pipeline(doubled_root, &sink);
    execute_pipeline(labeled_manager.get_or_create(4), &sink);
    let recycled = doubled_manager.get_or_create(5);
    assert_eq!(address(&recycled), doubled_address);
    execute_pipeline(recycled, &sink);

    assert_eq!(*doubled_output.lock().unwrap(), [6, 12]);
    assert_eq!(*labeled_output.lock().unwrap(), ["value=5"]);
}

#[test]
fn killing_a_later_stage_ends_the_chain_without_advancing() {
    let sink = TestSink::new();
    let kill_switch = switch();
    let mut manager = PipelineFrame::<i32>::builder()
        .fan_out(|value| value + 1)
        .compose_fanned_out(sink.clone(), |_| {}, kill_switch.clone());
    let mut root = manager.get_or_create(1);
    let (outcome, fill) = observer();
    root.observe(fill);

    execute(root);
    let intermediate = sink.poll().unwrap();
    intermediate.kill();
    intermediate.do_finally();

    assert!(kill_switch.load(Ordering::Acquire));
    assert_eq!(sink.len(), 0);
    assert_eq!(manager.recycle_queue_size(), 1);
    assert_eq!(status(&outcome), Some(Status::Cancelled));
}

#[test]
fn a_failing_stage_ends_the_chain_as_failed() {
    let sink = TestSink::new();
    let mut manager = PipelineFrame::<i32>::builder()
        .fan_out(|value| value + 1)
        .try_fan_out(|_: i32| -> Result<i32, FrameError> {
            Err(FrameError::failed("failed stage"))
        })
        .compose_fanned_out(sink.clone(), |_| {}, switch());
    let mut root = manager.get_or_create(1);
    let (outcome, fill) = observer();
    root.observe(fill);

    execute(root);
    let mut failing = sink.poll().unwrap();
    let error = failing.execute().unwrap_err();
    assert_eq!(error.to_string(), "frame failed: failed stage");
    failing.do_finally_with_error(error);

    assert_eq!(manager.recycle_queue_size(), 1);
    assert_eq!(sink.len(), 0);
    let outcome = outcome.lock().unwrap().take().unwrap();
    assert_eq!(outcome.status(), Status::Failed);
    assert_eq!(outcome.failure().unwrap().to_string(), "failed stage");
}

#[test]
fn a_stage_cancelling_through_an_executor_never_publishes_its_successor() {
    let sink = TestSink::new();
    let output = results();
    let mut manager = PipelineFrame::<i32>::builder()
        .try_fan_out(|_: i32| -> Result<i32, FrameError> { Err(FrameError::Cancelled) })
        .compose_fanned_out(sink.clone(), collect(&output), switch());
    let mut root = manager.get_or_create(1);
    let (outcome, fill) = observer();
    root.observe(fill);

    run(root);
    assert_eq!(sink.len(), 0);
    assert!(output.lock().unwrap().is_empty());
    assert_eq!(manager.recycle_queue_size(), 1);
    assert_eq!(status(&outcome), Some(Status::Cancelled));
}

#[test]
fn a_chain_killed_before_it_runs_is_skipped_and_cancelled() {
    let sink = TestSink::new();
    let output = results();
    let kill_switch = switch();
    let mut manager = PipelineFrame::<i32>::builder()
        .fan_out(|value| value + 1)
        .compose_fanned_out(sink.clone(), collect(&output), kill_switch.clone());
    let mut root = manager.get_or_create(1);
    let (outcome, fill) = observer();
    root.observe(fill);
    kill_switch.store(true, Ordering::Release);

    run(root);
    assert_eq!(status(&outcome), Some(Status::Cancelled));
    assert_eq!(sink.len(), 0);
    let returned = manager.get().unwrap();
    assert!(returned.data.is_none());
}

#[test]
fn a_reused_chain_shares_its_kill_switch() {
    let sink = TestSink::new();
    let kill_switch = switch();
    let mut manager = PipelineFrame::<i32>::builder()
        .fan_out(|value| value * 2)
        .compose_fanned_out(sink.clone(), |_| {}, kill_switch.clone());
    execute_pipeline(manager.get_or_create(10), &sink);

    let second = manager.get_or_create(20);
    second.kill();
    assert!(kill_switch.load(Ordering::Acquire));
    assert!(!second.is_alive());
}

#[test]
fn an_ordered_first_stage_stays_ordered_across_reuse() {
    let sink = TestSink::new();
    let mut manager = PipelineFrame::<i32>::builder()
        .fan_in(|value| value + 1)
        .fan_out(|value| value * 2)
        .compose_fanned_out(sink.clone(), |_| {}, switch());
    let first = manager.get_or_create(10);
    assert!(first.header().is_ordered());
    let first_address = address(&first);
    execute_pipeline(first, &sink);
    assert_eq!(manager.recycle_queue_size(), 1);

    let second = manager.get_or_create(20);
    assert_eq!(address(&second), first_address);
    assert!(second.header().is_ordered());
    assert_eq!(second.header().routing_hash(), second.header().id_hash());
}

#[test]
fn a_parallel_first_stage_stays_parallel_across_reuse() {
    let sink = TestSink::new();
    let mut manager = PipelineFrame::<i32>::builder()
        .fan_out(|value| value + 1)
        .fan_in(|value| value * 2)
        .compose_fanned_out(sink.clone(), |_| {}, switch());
    let first = manager.get_or_create(10);
    assert!(!first.header().is_ordered());
    let first_hash = first.header().routing_hash();
    execute_pipeline(first, &sink);

    let second = manager.get_or_create(20);
    assert!(!second.header().is_ordered());
    assert_ne!(
        second.header().routing_hash(),
        first_hash,
        "reuse draws a fresh seed"
    );
}

#[test]
fn mixed_stages_keep_their_ordering_across_reuse() {
    let sink = TestSink::new();
    let output = results();
    let mut manager = PipelineFrame::<i32>::builder()
        .fan_out(|value| value + 1)
        .fan_in(|value| value * 2)
        .fan_out(|value| value - 3)
        .compose_fanned_in(sink.clone(), collect(&output), switch());

    for input in [10, 20] {
        let steps = execute_pipeline(manager.get_or_create(input), &sink);
        let ordered: Vec<bool> = steps.iter().map(|step| step.ordered).collect();
        assert_eq!(ordered, [false, true, false, true]);
    }
    assert_eq!(*output.lock().unwrap(), [19, 39]);
}

#[test]
fn chains_of_one_pool_never_share_a_stage_seed() {
    let sink = TestSink::new();
    let mut manager = PipelineFrame::<i32>::builder()
        .fan_out(|value| value)
        .fan_out(|value| value)
        .fan_out(|value| value)
        .compose_fanned_out(sink, |_| {}, switch());
    let chains: Vec<_> = (0..4).map(|input| manager.get_or_create(input)).collect();
    let mut seen = HashSet::new();
    for chain in &chains {
        for hash in &chain.routing[1..] {
            assert!(
                seen.insert(*hash),
                "two stages share routing hash {hash:#x}"
            );
        }
    }
}

#[test]
fn an_intermediate_filter_drops_failing_values() {
    let sink = TestSink::new();
    let output = results();
    let mut manager = PipelineFrame::<i32>::builder()
        .fan_out(|value| value * 2)
        .filter_output(|value| *value > 10)
        .compose_fanned_out(sink.clone(), collect(&output), switch());

    let mut failing = manager.get_or_create(3);
    let (outcome, fill) = observer();
    failing.observe(fill);
    execute_pipeline(failing, &sink);
    assert!(output.lock().unwrap().is_empty());
    assert_eq!(sink.len(), 0);
    assert_eq!(status(&outcome), Some(Status::Filtered));

    execute_pipeline(manager.get_or_create(7), &sink);
    assert_eq!(*output.lock().unwrap(), [14]);
}

#[test]
fn a_filter_on_the_input_applies_at_the_consumer() {
    let sink = TestSink::new();
    let output = results();
    let mut manager = PipelineFrame::<i32>::builder()
        .filter_output(|value| value % 2 == 0)
        .compose_fanned_out(sink.clone(), collect(&output), switch());
    execute(manager.get_or_create(4));
    execute(manager.get_or_create(5));
    assert_eq!(*output.lock().unwrap(), [4]);
}

#[test]
fn filters_apply_stage_by_stage() {
    let sink = TestSink::new();
    let output = results();
    let mut manager = PipelineFrame::<i32>::builder()
        .fan_in(|value| value + 1)
        .filter_output(|value| *value > 5)
        .fan_out(|value| value * 2)
        .filter_output(|value| *value < 20)
        .compose_fanned_out(sink.clone(), collect(&output), switch());
    for input in [3, 15, 6] {
        execute_pipeline(manager.get_or_create(input), &sink);
    }
    assert_eq!(*output.lock().unwrap(), [14]);
}

#[test]
fn a_filtered_chain_is_reused_with_its_flag_cleared() {
    let sink = TestSink::new();
    let output = results();
    let mut manager = PipelineFrame::<i32>::builder()
        .fan_in(|value| value * 2)
        .filter_output(|value| *value > 10)
        .compose_fanned_in(sink.clone(), collect(&output), switch());

    let first = manager.get_or_create(3);
    let first_address = address(&first);
    execute_pipeline(first, &sink);
    assert!(output.lock().unwrap().is_empty());
    assert_eq!(manager.recycle_queue_size(), 1);

    let reused = manager.get_or_create(8);
    assert_eq!(address(&reused), first_address);
    assert!(!reused.filtered);
    execute_pipeline(reused, &sink);
    assert_eq!(*output.lock().unwrap(), [16]);
}

#[test]
fn filtering_one_branch_leaves_the_other_alone() {
    let sink = TestSink::new();
    let base = PipelineFrame::<i32>::builder().fan_out(|value| value + 1);
    let filtered = base.clone().filter_output(|value| *value > 5);
    let unfiltered = base.fan_out(|value| value * 2);
    let (filtered_output, unfiltered_output) = (results(), results());
    let mut filtered_manager =
        filtered.compose_fanned_out(sink.clone(), collect(&filtered_output), switch());
    let mut unfiltered_manager =
        unfiltered.compose_fanned_out(sink.clone(), collect(&unfiltered_output), switch());

    execute_pipeline(filtered_manager.get_or_create(2), &sink);
    execute_pipeline(unfiltered_manager.get_or_create(2), &sink);
    assert!(filtered_output.lock().unwrap().is_empty());
    assert_eq!(*unfiltered_output.lock().unwrap(), [6]);
}

#[test]
fn filters_apply_across_changing_types() {
    struct Payload {
        value: String,
        length: usize,
    }

    let sink = TestSink::new();
    let output = results();
    let mut manager = PipelineFrame::<i32>::builder()
        .fan_out(|value| value.to_string())
        .filter_output(|text| text.len() >= 2)
        .fan_in(|text| Payload {
            length: text.len(),
            value: text,
        })
        .filter_output(|payload| payload.length % 2 == 0)
        .fan_out(|payload| payload.value.starts_with('1'))
        .filter_output(|starts_with_one| *starts_with_one)
        .compose_fanned_out(sink.clone(), collect(&output), switch());

    for input in [5, 35, 1234] {
        execute_pipeline(manager.get_or_create(input), &sink);
    }
    assert_eq!(*output.lock().unwrap(), [true]);
}

#[test]
fn an_input_filter_runs_before_a_type_changing_stage() {
    let sink = TestSink::new();
    let output = results();
    let mut manager = PipelineFrame::<i32>::builder()
        .filter_output(|value| *value > 0)
        .fan_out(|value| value.to_string())
        .filter_output(|text| !text.is_empty())
        .compose_fanned_out(sink.clone(), collect(&output), switch());
    for input in [-1, 2] {
        execute_pipeline(manager.get_or_create(input), &sink);
    }
    assert_eq!(*output.lock().unwrap(), ["2"]);
}

#[test]
fn repeated_filters_must_all_pass() {
    for with_stage in [false, true] {
        let sink = TestSink::new();
        let output = results();
        let mut builder = PipelineFrame::<i32>::builder();
        if with_stage {
            builder = builder.fan_out(|value| value);
        }
        let mut manager = builder
            .filter_output(|value| *value > 0)
            .filter_output(|value| *value < 10)
            .compose_fanned_out(sink.clone(), collect(&output), switch());
        for input in [-1, 5, 11] {
            execute_pipeline(manager.get_or_create(input), &sink);
        }
        assert_eq!(*output.lock().unwrap(), [5], "with_stage = {with_stage}");
    }
}

#[test]
fn a_closed_sink_fails_the_chain_and_returns_it() {
    let sink = TestSink::new();
    let mut manager = PipelineFrame::<i32>::builder()
        .fan_out(|value| value + 1)
        .compose_fanned_out(sink.clone(), |_| {}, switch());
    let mut root = manager.get_or_create(1);
    let (outcome, fill) = observer();
    root.observe(fill);
    root.execute().unwrap();
    sink.close();

    root.do_finally();
    assert_eq!(manager.recycle_queue_size(), 1);
    let outcome = outcome.lock().unwrap().take().unwrap();
    assert_eq!(outcome.status(), Status::Failed);
    assert_eq!(
        outcome.failure().unwrap().to_string(),
        "could not enqueue the next pipeline stage"
    );
    let returned = manager.get().unwrap();
    assert_eq!(returned.stage(), 0, "a returned chain is rewound");
    assert!(!returned.header().is_ordered());
}

#[test]
fn a_kill_that_wins_against_scheduling_reports_cancelled() {
    let kill_switch = switch();
    let killer = kill_switch.clone();
    let sink = TestSink::rejecting(move || killer.store(true, Ordering::Release));
    let mut manager = PipelineFrame::<i32>::builder()
        .fan_out(|value| value + 1)
        .compose_fanned_out(sink, |_| {}, kill_switch);
    let mut root = manager.get_or_create(1);
    let (outcome, fill) = observer();
    root.observe(fill);
    root.execute().unwrap();

    root.do_finally();
    assert_eq!(status(&outcome), Some(Status::Cancelled));
    assert_eq!(manager.recycle_queue_size(), 1);
}

#[test]
fn every_ending_clears_every_payload() {
    struct Tracked(Arc<AtomicUsize>);

    impl Drop for Tracked {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }

    for ending in ["success", "failure", "cancellation", "filter"] {
        let drops = Arc::new(AtomicUsize::new(0));
        let stage_drops = drops.clone();
        let sink = TestSink::new();
        let mut manager = PipelineFrame::<Tracked>::builder()
            .fan_out(move |input: Tracked| {
                drop(input);
                Tracked(stage_drops.clone())
            })
            .filter_output(move |_| ending != "filter")
            .compose_fanned_out(sink.clone(), drop, switch());
        let mut root = manager.get_or_create(Tracked(drops.clone()));
        root.execute().unwrap();
        match ending {
            "success" => {
                root.do_finally();
                execute(sink.poll().unwrap());
            }
            "failure" => root.do_finally_with_error(FrameError::failed("boom")),
            "cancellation" => {
                root.kill();
                root.do_finally();
            }
            _ => root.do_finally(),
        }

        assert_eq!(drops.load(Ordering::Relaxed), 2, "ending by {ending}");
        let returned = manager.get().unwrap();
        assert!(returned.data.is_none(), "ending by {ending}");
        assert_eq!(returned.stage(), 0);
    }
}

#[test]
fn a_stage_without_input_fails() {
    let sink = TestSink::new();
    let mut manager = PipelineFrame::<i32>::builder()
        .fan_out(|value| value)
        .compose_fanned_out(sink.clone(), |_| {}, switch());
    execute_pipeline(manager.get_or_create(1), &sink);

    let mut raw = manager.get().unwrap();
    assert!(
        raw.execute()
            .unwrap_err()
            .to_string()
            .contains("without its input")
    );
}

#[test]
fn the_observer_runs_after_the_chain_is_back_and_before_the_completion() {
    let sink = TestSink::new();
    let mut manager =
        PipelineFrame::<i32>::builder().compose_fanned_out(sink.clone(), |_| {}, switch());
    let events = Arc::new(Mutex::new(Vec::new()));
    let mut root = manager.get_or_create(1);
    let observed = events.clone();
    root.observe(move |outcome| {
        observed
            .lock()
            .unwrap()
            .push(format!("{:?}", outcome.status()))
    });
    let completed = events.clone();
    root.on_completion(Arc::new(move || {
        completed.lock().unwrap().push("done".to_owned())
    }));

    execute(root);
    assert_eq!(*events.lock().unwrap(), ["Success", "done"]);
    assert_eq!(manager.recycle_queue_size(), 1);

    execute(manager.get_or_create(2));
    assert_eq!(
        events.lock().unwrap().len(),
        2,
        "observers belong to one run"
    );
}

#[test]
fn the_completion_runs_even_when_the_observer_panics() {
    let sink = TestSink::new();
    let mut manager =
        PipelineFrame::<i32>::builder().compose_fanned_out(sink.clone(), |_| {}, switch());
    let completions = Arc::new(AtomicUsize::new(0));
    let mut root = manager.get_or_create(1);
    root.observe(|_| panic!("observer failed"));
    let counter = completions.clone();
    root.on_completion(Arc::new(move || {
        counter.fetch_add(1, Ordering::Relaxed);
    }));

    let result = catch_unwind(AssertUnwindSafe(|| execute(root)));
    assert!(result.is_err());
    assert_eq!(completions.load(Ordering::Relaxed), 1);
    assert_eq!(manager.recycle_queue_size(), 1);
}

#[test]
fn a_recycled_chain_forgets_its_observer() {
    let sink = TestSink::new();
    let mut manager = PipelineFrame::<i32>::builder()
        .fan_out(|value| value)
        .compose_fanned_out(sink.clone(), |_| {}, switch());
    let mut root = manager.get_or_create(1);
    let (outcome, fill) = observer();
    root.observe(fill);
    root.recycle();

    let reused = manager.get_or_create(2);
    execute_pipeline(reused, &sink);
    assert_eq!(status(&outcome), None);
}

#[test]
fn outcomes_report_their_status_and_failure() {
    assert_eq!(Outcome::Success.status(), Status::Success);
    assert_eq!(Outcome::Filtered.status(), Status::Filtered);
    assert_eq!(Outcome::Cancelled.status(), Status::Cancelled);
    assert!(Outcome::Cancelled.failure().is_none());
    let failed = Outcome::Failed("boom".into());
    assert_eq!(failed.status(), Status::Failed);
    assert_eq!(failed.failure().unwrap().to_string(), "boom");
}

#[test]
fn chains_run_to_completion_across_worker_threads() {
    const WORKERS: usize = 4;
    const CHAINS: i64 = 500;

    let sink = TestSink::new();
    let output = results();
    let pipeline = PipelineFrame::<i64>::builder()
        .fan_out(|value| value * 3)
        .fan_in(|value| value + 1)
        .fan_out(|value| value.to_string());
    let mut manager = pipeline.compose_fanned_out(sink.clone(), collect(&output), switch());
    let finished = Arc::new(AtomicUsize::new(0));
    let completion: Completion = {
        let finished = finished.clone();
        Arc::new(move || {
            finished.fetch_add(1, Ordering::AcqRel);
        })
    };

    let deadline = Instant::now() + Duration::from_secs(10);
    let workers: Vec<_> = (0..WORKERS)
        .map(|_| {
            let sink = sink.clone();
            let finished = finished.clone();
            thread::spawn(move || {
                while finished.load(Ordering::Acquire) < CHAINS as usize
                    && Instant::now() < deadline
                {
                    match sink.poll() {
                        Some(frame) => run(frame),
                        None => thread::yield_now(),
                    }
                }
            })
        })
        .collect();

    let statuses = results();
    for input in 0..CHAINS {
        let mut chain = manager.get_or_create(input);
        let statuses = statuses.clone();
        chain.observe(move |outcome| statuses.lock().unwrap().push(outcome.status()));
        chain.on_completion(completion.clone());
        assert!(sink.offer(chain).is_ok());
    }
    for worker in workers {
        worker.join().unwrap();
    }
    assert_eq!(
        finished.load(Ordering::Acquire),
        CHAINS as usize,
        "every chain ran its completion before the deadline"
    );

    let mut produced = std::mem::take(&mut *output.lock().unwrap());
    produced.sort_unstable();
    let mut expected: Vec<String> = (0..CHAINS)
        .map(|input| (input * 3 + 1).to_string())
        .collect();
    expected.sort_unstable();
    assert_eq!(produced, expected);
    assert!(
        statuses
            .lock()
            .unwrap()
            .iter()
            .all(|status| *status == Status::Success)
    );
    assert_eq!(statuses.lock().unwrap().len(), CHAINS as usize);
}
