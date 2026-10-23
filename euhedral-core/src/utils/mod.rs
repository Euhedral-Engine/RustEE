//! Small helpers shared by the control plane: integer and EWMA math, spin waiting, and sampled
//! timing.

pub mod math;
pub mod spin_wait;
mod stop_watch;

pub use stop_watch::{DEFAULT_INTERVAL, StopWatch};
