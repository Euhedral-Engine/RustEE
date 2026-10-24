use super::{CloneableObject, LatticeReceiver, LatticeSource, LatticeTerminal};
use crate::config::CloneConfig;
use crate::frames::{Failure, Frame, FrameError};
use crate::metrics::time_execute;
use std::any::Any;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;

/// The terminal sink of the lattice: receives frames and owns their execution and finalization.
///
/// An executor supplies [`execute`](Self::execute) and [`hook_on_clone`](Self::hook_on_clone); every
/// executor is a [`CloneableObject`] whose [`input`](CloneableObject::input) attaches a terminal that
/// requests unbounded demand and [runs](Self::run) each frame pushed to it.
pub trait AbstractExecutor: Sized + Send + Sync + 'static {
    /// The CPU this executor runs on, if it is bound to one.
    fn cpu(&self) -> Option<usize>;

    /// Does the work of one live frame.
    fn execute(&self, frame: &mut dyn Frame) -> Result<(), FrameError>;

    /// A new executor like this one, for `cpu`.
    fn hook_on_clone(&self, cpu: Option<usize>) -> Self;

    /// A new executor on the first of `config`'s effective CPUs.
    fn clone_for(&self, config: &CloneConfig) -> Self {
        self.hook_on_clone(config.effective_cpus().first())
    }

    /// Runs `frame` through its lifecycle and hands it to its finalizer:
    ///
    /// - a frame that is not alive is not executed, and finishes with
    ///   [`do_finally`](Frame::do_finally);
    /// - a frame whose execution succeeds or is cancelled finishes with `do_finally`;
    /// - a frame whose execution fails, or panics, finishes with
    ///   [`do_finally_with_error`](Frame::do_finally_with_error).
    ///
    /// A finalizer that panics is logged; the frame it held is gone.
    ///
    /// On a lattice worker with metrics, `execute` is timed into the worker's
    /// `.execution.serviceTime` summary ([`time_execute`]); the finalizer is not. An executor that
    /// overrides `run` opts in by wrapping its call to [`execute`](Self::execute) in
    /// [`time_execute`].
    fn run(&self, mut frame: Box<dyn Frame>) {
        if !frame.is_alive() {
            finalize(|| frame.do_finally());
            return;
        }
        match catch_unwind(AssertUnwindSafe(|| {
            time_execute(|| self.execute(&mut *frame))
        })) {
            Ok(Ok(())) | Ok(Err(FrameError::Cancelled)) => finalize(|| frame.do_finally()),
            Ok(Err(error)) => {
                log::error!("Uncaught error while executing frame: {error}");
                finalize(|| frame.do_finally_with_error(error));
            }
            Err(panic) => {
                let failure = panic_failure(panic);
                log::error!("Frame panicked while executing: {failure}");
                finalize(|| frame.do_finally_with_error(FrameError::Failed(failure)));
            }
        }
    }
}

impl<E: AbstractExecutor> CloneableObject for E {
    fn clone_object(&self, config: &CloneConfig) -> Arc<dyn CloneableObject> {
        Arc::new(self.clone_for(config))
    }

    fn input(self: Arc<Self>, stream: Arc<dyn LatticeSource>) {
        stream.add_downstream(Arc::new(ExecutionTerminal { executor: self }));
    }
}

/// The receiver an executor attaches to each input stream.
struct ExecutionTerminal<E> {
    executor: Arc<E>,
}

impl<E: AbstractExecutor> LatticeTerminal for ExecutionTerminal<E> {
    fn add_upstream(&self, upstream: Arc<dyn LatticeSource>) {
        upstream.request(u64::MAX);
    }
}

impl<E: AbstractExecutor> LatticeReceiver for ExecutionTerminal<E> {
    fn push(&self, frame: Box<dyn Frame>) {
        self.executor.run(frame);
    }

    fn on_complete(&self) {}

    fn on_error(&self, error: Failure) {
        log::error!("Executor input failed: {error}");
    }
}

fn finalize(finalizer: impl FnOnce()) {
    if let Err(panic) = catch_unwind(AssertUnwindSafe(finalizer)) {
        log::error!("Frame finalizer panicked: {}", panic_failure(panic));
    }
}

fn panic_failure(panic: Box<dyn Any + Send>) -> Failure {
    let message = match panic.downcast::<String>() {
        Ok(message) => *message,
        Err(panic) => match panic.downcast::<&'static str>() {
            Ok(message) => (*message).to_owned(),
            Err(_) => "frame panicked".to_owned(),
        },
    };
    message.into()
}
