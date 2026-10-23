//! The interfaces frames flow through: lattice sources, receivers and interceptors, and the
//! simpler hand-off points frames use themselves.

mod cloneable;
mod executor;
mod lattice;

pub use cloneable::CloneableObject;
pub use executor::AbstractExecutor;
pub use lattice::{LatticeInterceptor, LatticeReceiver, LatticeSource, LatticeTerminal};

use crate::frames::Frame;

/// Accepts frames for scheduling, such as the next stage of a
/// [`PipelineFrame`](crate::frames::PipelineFrame).
pub trait FrameSink: Send + Sync {
    /// Takes `frame`, or hands that same frame back when it cannot.
    fn offer(&self, frame: Box<dyn Frame>) -> Result<(), Box<dyn Frame>>;
}

/// Takes ownership of frames handed back by the frames themselves, such as finished
/// [`CallbackFrame`](crate::frames::CallbackFrame)s.
///
/// Any `Fn(Box<F>) + Send + Sync` closure is a pusher.
pub trait FramePusher<F: ?Sized>: Send + Sync {
    fn push(&self, frame: Box<F>);
}

impl<F: ?Sized, P> FramePusher<F> for P
where
    P: Fn(Box<F>) + Send + Sync,
{
    fn push(&self, frame: Box<F>) {
        self(frame)
    }
}
