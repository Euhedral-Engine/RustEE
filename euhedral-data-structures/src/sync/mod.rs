//! Lock-free cells shared across threads.
//!
//! - [`SetOnce`]: a value installed at most once by a compare-and-set, read with an acquire load.
//!
//! Locks for lifecycle work, and the debug check that keeps them off the hot path:
//!
//! - [`LifecycleMutex`], [`LifecycleRwLock`]: the only lock types the workspace lints allow.
//! - [`HotScope`]: marks the code a worker or producer runs per frame; a lifecycle lock taken inside one panics in
//!   debug builds.

mod lifecycle;
mod loom;
mod set_once;

pub use set_once::SetOnce;
pub use lifecycle::{ColdGuard, HotGuard, HotScope, LifecycleMutex, LifecycleRwLock};
