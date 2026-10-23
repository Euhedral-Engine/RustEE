//! Frame pooling, where [`FrameManager`] keeps finished frames for reuse and [`FrameFactory`]
//! makes or refills them, and the stock [`DefaultExecutor`] (upstream package `impl`).

mod default_executor;
mod frame_factory;
mod frame_manager;

pub use default_executor::DefaultExecutor;
pub use frame_factory::FrameFactory;
pub use frame_manager::{DEFAULT_CAPACITY, FrameManager, Recycler};
