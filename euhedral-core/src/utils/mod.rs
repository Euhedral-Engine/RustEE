//! Small helpers shared by the control plane: integer and EWMA math, spin waiting, sampled timing,
//! flow statistics, per-thread flow bookkeeping, and CPU-work calibration.

mod average_flow;
pub mod calibration;
mod flow_context;
mod flow_recorder;
pub mod math;
pub mod spin_wait;
mod stop_watch;

pub use average_flow::{ACQUIRE_CONTENTION_SCALE, AverageFlow, SampleOutOfRange};
pub use flow_context::FlowContext;
pub use flow_recorder::FlowRecorder;
pub use stop_watch::StopWatch;
