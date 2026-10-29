//! Lock-free cells shared across threads.
//!
//! - [`SetOnce`]: a value installed at most once by a compare-and-set, read with an acquire load.
//! - [`ClaimCell`]: plain state that one thread at a time claims with a compare-and-set that fails instead of
//!   waiting.
//!
//! Locks for lifecycle work, and the debug check that keeps them off the hot path:
//!
//! - [`LifecycleMutex`], [`LifecycleRwLock`]: the lock types meant for start, stop and rebalance state.
//! - [`HotScope`]: marks the code a thread runs once per unit of work; a lifecycle lock taken inside one aborts in
//!   debug builds.

mod claim_cell;
mod lifecycle;
mod loom;
mod set_once;

pub use claim_cell::{ClaimCell, ClaimGuard};
pub use lifecycle::{ColdGuard, HotGuard, HotScope, LifecycleMutex, LifecycleRwLock};
pub use set_once::SetOnce;
