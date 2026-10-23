//! Frame pooling, where [`FrameManager`] keeps finished frames for reuse and [`FrameFactory`]
//! makes or refills them, the stock [`DefaultExecutor`], and [`BaseCloneableObject`], which joins a
//! fragment to an executor on every core (upstream package `impl`).

mod base_cloneable;
mod default_executor;
mod frame_factory;
mod frame_manager;

pub use base_cloneable::BaseCloneableObject;
pub use default_executor::DefaultExecutor;
pub use frame_factory::FrameFactory;
pub use frame_manager::{DEFAULT_CAPACITY, FrameManager, Recycler};
