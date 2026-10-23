//! Frame pooling: [`FrameManager`] keeps finished frames for reuse and [`FrameFactory`] makes or
//! refills them (upstream package `impl`).

mod frame_factory;
mod frame_manager;

pub use frame_factory::FrameFactory;
pub use frame_manager::{DEFAULT_CAPACITY, FrameManager, Recycler};
