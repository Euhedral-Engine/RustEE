//! Sources that feed frames into a [`ControlPlaneLattice`](crate::control_plane::ControlPlaneLattice):
//! hand an [`IngestSink`]'s [`delegate`](IngestSink::delegate) to
//! [`add_upstream`](crate::control_plane::ControlPlaneLattice::add_upstream).
//!
//! - [`QueueIngestSink`]: a partitioned queue that any thread publishes to.
//! - [`ArrayIngestSink`]: a fixed batch of frames, complete once every frame is taken.
//! - [`PipelineRunner`]: runs typed [`PipelineFrame`](crate::frames::PipelineFrame) chains, with
//!   admission that completes gracefully once every accepted chain has finished.
//! - [`SingleUseSource`]: one frame.

mod array;
mod pipeline_runner;
mod queue;
mod single_use;

pub use array::ArrayIngestSink;
pub use pipeline_runner::{AdmissionClosed, OutcomeSlot, PipelineControl, PipelineRunner};
pub use queue::QueueIngestSink;
pub use single_use::SingleUseSource;

use crate::generics::LatticeSource;
use std::sync::Arc;

/// A source of frames for a lattice, and the handle to finish it.
pub trait IngestSink {
    /// The source the lattice ingests. A completed source completes any downstream that attaches.
    fn delegate(&self) -> Arc<dyn LatticeSource>;

    /// Finishes the source at once and detaches it from the lattice.
    fn complete(&self);

    fn is_complete(&self) -> bool;
}
