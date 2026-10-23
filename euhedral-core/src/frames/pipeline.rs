use super::{Failure, Frame, FrameError, FrameHeader, randomized_hash};
use crate::generics::FrameSink;
use crate::impls::{FrameFactory, FrameManager, Recycler};
use crate::internal::random;
use std::any::Any;
use std::fmt;
use std::marker::PhantomData;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

type Value = Box<dyn Any + Send>;
type StageFn = Arc<dyn Fn(Value) -> Result<Value, FrameError> + Send + Sync>;
type Filter = Arc<dyn Fn(&(dyn Any + Send)) -> bool + Send + Sync>;
type Terminal = Box<dyn Fn(Value) + Send + Sync>;
type Observer = Box<dyn FnOnce(Outcome) + Send>;

/// The owner notification a [`PipelineFrame`] runs when its chain finishes.
pub type Completion = Arc<dyn Fn() + Send + Sync>;

/// The seed an unordered first stage is randomized with until its factory gives it its own.
const ROOT_SEED: u64 = 17;

/// How a chain ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Status {
    Success,
    Filtered,
    Cancelled,
    Failed,
}

/// How a chain ended, with the failure if it failed.
#[derive(Debug)]
pub enum Outcome {
    /// The terminal consumer took the output.
    Success,
    /// A filter rejected the value; later stages never ran.
    Filtered,
    /// The chain was killed or a stage cancelled it.
    Cancelled,
    /// A stage failed, or the next stage could not be scheduled.
    Failed(Failure),
}

impl Outcome {
    pub fn status(&self) -> Status {
        match self {
            Self::Success => Status::Success,
            Self::Filtered => Status::Filtered,
            Self::Cancelled => Status::Cancelled,
            Self::Failed(_) => Status::Failed,
        }
    }

    pub fn failure(&self) -> Option<&(dyn std::error::Error + Send + Sync + 'static)> {
        match self {
            Self::Failed(failure) => Some(&**failure),
            _ => None,
        }
    }
}

#[derive(Clone)]
struct Stage {
    /// `None` passes the value through unchanged.
    function: Option<StageFn>,
    ordered: bool,
    filter: Option<Filter>,
}

/// What every chain of one composed pipeline shares.
struct Definition {
    /// The transforms, then the terminal stage that hands the output to `consumer`.
    stages: Box<[Stage]>,
    consumer: Terminal,
    sink: Arc<dyn FrameSink>,
}

/// One run of a typed, multi-stage pipeline whose stages are scheduled one after another.
///
/// Build a pipeline with [`builder`](Self::builder), add stages with
/// [`fan_out`](PipelineBuilder::fan_out) or [`fan_in`](PipelineBuilder::fan_in), and bind it to a
/// sink and a final consumer with [`compose_fanned_out`](PipelineBuilder::compose_fanned_out) or
/// [`compose_fanned_in`](PipelineBuilder::compose_fanned_in). That returns a [`FrameManager`] that
/// hands out chains: [`get_or_create`](FrameManager::get_or_create) takes the input, and the chain is
/// then dispatched like any other frame.
///
/// A chain is one frame that steps through its stages. Each execution runs the current stage. When
/// it finishes, the chain moves to the next stage, takes that stage's routing hash, and offers itself
/// to the sink, so each stage, and the final consumer, is scheduled separately. A fan-in stage keeps
/// the chain's id hash and stays ordered with the other chains of its pipeline; a fan-out stage gets
/// its own routing hash and may run in parallel with them.
///
/// When the chain ends (consumed, filtered, cancelled or failed) it clears its payload and returns
/// to its pool, then reports its [`Outcome`] to the observer and runs the completion, in that order.
///
/// ```
/// use euhedral_core::frames::{Frame, FrameError, PipelineFrame};
/// use euhedral_core::generics::FrameSink;
/// use std::collections::VecDeque;
/// use std::sync::atomic::AtomicBool;
/// use std::sync::{Arc, Mutex};
///
/// #[derive(Default)]
/// struct Queue(Mutex<VecDeque<Box<dyn Frame>>>);
///
/// impl FrameSink for Queue {
///     fn offer(&self, frame: Box<dyn Frame>) -> Result<(), Box<dyn Frame>> {
///         self.0.lock().unwrap().push_back(frame);
///         Ok(())
///     }
/// }
///
/// let sink = Arc::new(Queue::default());
/// let output = Arc::new(Mutex::new(Vec::new()));
/// let consumer_output = output.clone();
/// let mut chains = PipelineFrame::<u32>::builder()
///     .fan_out(|value| value * 2)
///     .fan_in(|value| format!("result={value}"))
///     .compose_fanned_out(sink.clone(), move |text| consumer_output.lock().unwrap().push(text),
///         Arc::new(AtomicBool::new(false)));
///
/// let mut frame: Box<dyn Frame> = chains.get_or_create(21);
/// loop {
///     match frame.execute() {
///         Ok(()) | Err(FrameError::Cancelled) => frame.do_finally(),
///         Err(error) => frame.do_finally_with_error(error),
///     }
///     let Some(next) = sink.0.lock().unwrap().pop_front() else { break };
///     frame = next;
/// }
/// assert_eq!(*output.lock().unwrap(), ["result=42"]);
/// ```
pub struct PipelineFrame<I> {
    header: FrameHeader,
    definition: Arc<Definition>,
    /// The routing hash of each stage; the first is current only while the chain is on another stage.
    routing: Box<[u64]>,
    stage: usize,
    data: Option<Value>,
    filtered: bool,
    cancelled: bool,
    observer: Option<Observer>,
    completion: Option<Completion>,
    recycler: Option<Recycler<PipelineFrame<I>>>,
    input: PhantomData<fn(I)>,
}

impl<I: Send + 'static> PipelineFrame<I> {
    /// Starts a pipeline definition taking `I`.
    pub fn builder() -> PipelineBuilder<I, I> {
        PipelineBuilder {
            stages: Vec::new(),
            filter: None,
            types: PhantomData,
        }
    }

    fn chain(
        id_hash: u64,
        input: I,
        definition: Arc<Definition>,
        recycler: Recycler<PipelineFrame<I>>,
        kill_switch: Arc<AtomicBool>,
        routing_seed: u64,
    ) -> Self {
        let mut header = FrameHeader::with_kill_switch(id_hash, Some(kill_switch));
        if !definition.stages[0].ordered {
            header.randomize_hash(ROOT_SEED);
        }
        let mut chain = Self {
            header,
            routing: vec![id_hash; definition.stages.len()].into(),
            definition,
            stage: 0,
            data: Some(Box::new(input)),
            filtered: false,
            cancelled: false,
            observer: None,
            completion: None,
            recycler: Some(recycler),
            input: PhantomData,
        };
        chain.route(routing_seed);
        chain
    }

    /// Gives the stages after the first their routing hashes. The first stage's is the header's,
    /// which the factory owns.
    fn route(&mut self, routing_seed: u64) {
        let id_hash = self.header.id_hash();
        for (index, stage) in self.definition.stages.iter().enumerate().skip(1) {
            let seed = routing_seed.wrapping_add(index as u64 - 1);
            self.routing[index] = if stage.ordered {
                id_hash
            } else {
                randomized_hash(id_hash, seed)
            };
        }
    }

    fn refill(&mut self, input: I, routing_seed: u64) {
        self.data = Some(Box::new(input));
        self.route(routing_seed);
    }
}

impl<I> PipelineFrame<I> {
    /// Installs the observer that receives this chain's [`Outcome`]. Install it before the chain is
    /// dispatched.
    ///
    /// The observer runs on the thread that finishes the chain, after the chain is back in its pool,
    /// so it may check the next chain out at once. It must not block waiting on the pipeline.
    pub fn observe(&mut self, observer: impl FnOnce(Outcome) + Send + 'static) {
        self.observer = Some(Box::new(observer));
    }

    /// Installs the owner notification, run after the observer when this chain finishes, even if the
    /// observer panics. It must not block.
    pub fn on_completion(&mut self, completion: Completion) {
        self.completion = Some(completion);
    }

    /// The index of the stage the chain runs next; the last stage is the consumer.
    pub fn stage(&self) -> usize {
        self.stage
    }

    /// The number of scheduled stages, the consumer included.
    pub fn stage_count(&self) -> usize {
        self.definition.stages.len()
    }

    fn is_terminal(&self) -> bool {
        self.stage + 1 == self.definition.stages.len()
    }
}

impl<I: Send + 'static> PipelineFrame<I> {
    /// Moves to the next stage and offers the chain to the sink. A rejected chain ends as cancelled
    /// if it has been killed meanwhile, and as failed otherwise.
    fn advance(mut self: Box<Self>) {
        self.routing[self.stage] = self.header.routing_hash();
        self.stage += 1;
        self.header.set_routing_hash(self.routing[self.stage]);

        let definition = Arc::clone(&self.definition);
        let Err(rejected) = definition.sink.offer(self) else {
            return;
        };
        let rejected: Box<dyn Any> = rejected;
        let chain = rejected.downcast::<Self>().unwrap_or_else(|_| {
            panic!("a FrameSink handed back a different frame than it was offered")
        });
        if !chain.is_alive() {
            chain.finish(Outcome::Cancelled);
        } else {
            if chain.observer.is_none() {
                log::error!("Could not enqueue the next pipeline stage");
            }
            chain.finish(Outcome::Failed(
                "could not enqueue the next pipeline stage".into(),
            ));
        }
    }

    /// Returns the chain to its pool, then reports `outcome` and notifies the owner. Nothing touches
    /// the chain after it is back in the pool: its owner may reuse it at once.
    fn finish(mut self: Box<Self>, outcome: Outcome) {
        let observer = self.observer.take();
        let notify = Notify(self.completion.take());
        self.recycle();
        if let Some(observer) = observer {
            observer(outcome);
        }
        drop(notify);
    }
}

/// Runs the completion when dropped, so it runs even while an observer panic unwinds.
struct Notify(Option<Completion>);

impl Drop for Notify {
    fn drop(&mut self) {
        if let Some(completion) = self.0.take() {
            completion();
        }
    }
}

impl<I: Send + 'static> Frame for PipelineFrame<I> {
    fn header(&self) -> &FrameHeader {
        &self.header
    }

    fn header_mut(&mut self) -> &mut FrameHeader {
        &mut self.header
    }

    /// Runs the current stage: transforms the value, filters it, and passes it to the next stage or,
    /// on the last stage, to the consumer.
    fn execute(&mut self) -> Result<(), FrameError> {
        let stage = &self.definition.stages[self.stage];
        let Some(value) = self.data.take() else {
            return Err(FrameError::failed("pipeline stage ran without its input"));
        };
        let result = match &stage.function {
            None => value,
            Some(function) => function(value).inspect_err(|error| {
                self.cancelled = error.is_cancelled();
            })?,
        };
        if let Some(filter) = &stage.filter
            && !filter(&*result)
        {
            self.filtered = true;
            return Ok(());
        }
        if self.is_terminal() {
            (self.definition.consumer)(result);
        } else {
            self.data = Some(result);
        }
        Ok(())
    }

    /// Ends the chain if it was killed, cancelled, filtered or consumed; otherwise schedules the next
    /// stage.
    fn do_finally(self: Box<Self>) {
        if !self.is_alive() || self.cancelled {
            self.finish(Outcome::Cancelled);
        } else if self.filtered {
            self.finish(Outcome::Filtered);
        } else if self.is_terminal() {
            self.finish(Outcome::Success);
        } else {
            self.advance();
        }
    }

    fn do_finally_with_error(self: Box<Self>, error: FrameError) {
        self.finish(match error {
            FrameError::Cancelled => Outcome::Cancelled,
            FrameError::Failed(failure) => Outcome::Failed(failure),
        });
    }

    /// Rewinds the chain to its first stage, drops its payload and any pending observer, and returns
    /// it to its pool.
    fn recycle(mut self: Box<Self>) {
        if self.stage != 0 {
            self.stage = 0;
            self.header.set_routing_hash(self.routing[0]);
        }
        self.data = None;
        self.filtered = false;
        self.cancelled = false;
        self.observer = None;
        self.completion = None;
        Recycler::recycle_owned(self, |chain| chain.recycler.as_ref());
    }
}

impl<I> fmt::Debug for PipelineFrame<I> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PipelineFrame")
            .field("header", &self.header)
            .field("stage", &self.stage)
            .field("stage_count", &self.definition.stages.len())
            .finish_non_exhaustive()
    }
}

/// A pipeline definition from input `I` to the current output `O`.
///
/// Each method consumes the builder and returns the extended one; clone a builder to branch a shared
/// prefix into several pipelines.
pub struct PipelineBuilder<I, O> {
    stages: Vec<Stage>,
    /// A filter on the input, waiting for the first stage.
    filter: Option<Filter>,
    types: PhantomData<fn(I) -> O>,
}

impl<I, O> Clone for PipelineBuilder<I, O> {
    fn clone(&self) -> Self {
        Self {
            stages: self.stages.clone(),
            filter: self.filter.clone(),
            types: PhantomData,
        }
    }
}

impl<I: Send + 'static, O: Send + 'static> PipelineBuilder<I, O> {
    /// Drops values for which `predicate` is false, ending their chains as
    /// [filtered](Outcome::Filtered). Repeated filters must all pass.
    pub fn filter_output(mut self, predicate: impl Fn(&O) -> bool + Send + Sync + 'static) -> Self {
        let predicate: Filter = Arc::new(move |value| predicate(downcast_ref(value)));
        let slot = match self.stages.last_mut() {
            Some(last) => &mut last.filter,
            None => &mut self.filter,
        };
        *slot = Some(match slot.take() {
            Some(first) => Arc::new(move |value| first(value) && predicate(value)),
            None => predicate,
        });
        self
    }

    /// Appends a stage that may run in parallel across chains.
    pub fn fan_out<N: Send + 'static>(
        self,
        function: impl Fn(O) -> N + Send + Sync + 'static,
    ) -> PipelineBuilder<I, N> {
        self.append(false, move |value| Ok(function(value)))
    }

    /// Appends a stage that runs in order across the chains of a pipeline.
    pub fn fan_in<N: Send + 'static>(
        self,
        function: impl Fn(O) -> N + Send + Sync + 'static,
    ) -> PipelineBuilder<I, N> {
        self.append(true, move |value| Ok(function(value)))
    }

    /// Like [`fan_out`](Self::fan_out), for a stage that can fail or cancel its chain.
    pub fn try_fan_out<N: Send + 'static>(
        self,
        function: impl Fn(O) -> Result<N, FrameError> + Send + Sync + 'static,
    ) -> PipelineBuilder<I, N> {
        self.append(false, function)
    }

    /// Like [`fan_in`](Self::fan_in), for a stage that can fail or cancel its chain.
    pub fn try_fan_in<N: Send + 'static>(
        self,
        function: impl Fn(O) -> Result<N, FrameError> + Send + Sync + 'static,
    ) -> PipelineBuilder<I, N> {
        self.append(true, function)
    }

    fn append<N: Send + 'static>(
        mut self,
        ordered: bool,
        function: impl Fn(O) -> Result<N, FrameError> + Send + Sync + 'static,
    ) -> PipelineBuilder<I, N> {
        if let Some(filter) = self.filter.take() {
            self.stages.push(Stage {
                function: None,
                ordered,
                filter: Some(filter),
            });
        }
        let function: StageFn =
            Arc::new(move |value| function(downcast(value)).map(|next| Box::new(next) as Value));
        self.stages.push(Stage {
            function: Some(function),
            ordered,
            filter: None,
        });
        PipelineBuilder {
            stages: self.stages,
            filter: None,
            types: PhantomData,
        }
    }

    /// Binds the pipeline to `sink` with a consumer that may run in parallel across chains.
    ///
    /// Returns the pool that hands out chains. Every chain dies with `kill_switch`.
    pub fn compose_fanned_out(
        &self,
        sink: Arc<dyn FrameSink>,
        consumer: impl Fn(O) + Send + Sync + 'static,
        kill_switch: Arc<AtomicBool>,
    ) -> FrameManager<I, PipelineFrame<I>> {
        self.compose(sink, consumer, false, kill_switch)
    }

    /// Binds the pipeline to `sink` with a consumer that runs in order across chains.
    ///
    /// Returns the pool that hands out chains. Every chain dies with `kill_switch`.
    pub fn compose_fanned_in(
        &self,
        sink: Arc<dyn FrameSink>,
        consumer: impl Fn(O) + Send + Sync + 'static,
        kill_switch: Arc<AtomicBool>,
    ) -> FrameManager<I, PipelineFrame<I>> {
        self.compose(sink, consumer, true, kill_switch)
    }

    fn compose(
        &self,
        sink: Arc<dyn FrameSink>,
        consumer: impl Fn(O) + Send + Sync + 'static,
        terminal_ordered: bool,
        kill_switch: Arc<AtomicBool>,
    ) -> FrameManager<I, PipelineFrame<I>> {
        let mut stages = self.stages.clone();
        stages.push(Stage {
            function: None,
            ordered: terminal_ordered,
            filter: self.filter.clone(),
        });
        let definition = Arc::new(Definition {
            stages: stages.into(),
            consumer: Box::new(move |value| consumer(downcast(value))),
            sink,
        });

        // Each chain draws one seed per stage after the first, so no two chains share a stage seed.
        let span = (definition.stages.len() as u64 - 1).max(1);
        let create_seeds = Arc::new(AtomicU64::new(random::next_u64()));
        let replace_seeds = Arc::clone(&create_seeds);

        let mut manager = FrameManager::new();
        let recycler = manager.recycler();
        let factory = FrameFactory::new(
            move |id_hash, input| {
                PipelineFrame::chain(
                    id_hash,
                    input,
                    Arc::clone(&definition),
                    recycler.clone(),
                    Arc::clone(&kill_switch),
                    create_seeds.fetch_add(span, Ordering::Relaxed),
                )
            },
            move |input, chain: &mut PipelineFrame<I>| {
                chain.refill(input, replace_seeds.fetch_add(span, Ordering::Relaxed));
            },
        );
        manager
            .set_factory(factory)
            .expect("a new FrameManager has no factory");
        manager
    }
}

fn downcast<T: 'static>(value: Value) -> T {
    *value
        .downcast()
        .unwrap_or_else(|_| unreachable!("the builder types every stage's input"))
}

fn downcast_ref<T: 'static>(value: &(dyn Any + Send)) -> &T {
    value
        .downcast_ref()
        .unwrap_or_else(|| unreachable!("the builder types every stage's input"))
}

#[cfg(test)]
mod tests;
