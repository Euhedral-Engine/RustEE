//! Lock-free cells shared across threads.
//!
//! - [`SetOnce`]: a value installed at most once by a compare-and-set, read with an acquire load.
//! - [`Published`]: an atomically replaceable shared value with count-free reads, reclaimed once every reader's epoch
//!   has passed (the stand-in for a collector).
//! - [`Admission`]: a reader count with a retired bit, Java's `admission` of a routing state: readers take a count while
//!   the state is current, and a writer that retires it learns when the last reader leaves.
//! - [`ClaimCell`]: plain state that one thread at a time claims with a compare-and-set that fails instead of
//!   waiting.
//!
//! Locks for lifecycle work, and the debug check that keeps them off the hot path:
//!
//! - [`LifecycleMutex`], [`LifecycleRwLock`]: the lock types meant for start, stop and rebalance state.
//! - [`HotScope`]: marks the code a thread runs once per unit of work; a lifecycle lock taken inside one aborts in
//!   debug builds.

mod admission;
mod barrier;
mod claim_cell;
mod lifecycle;
mod loom;
mod published;
mod set_once;

pub use admission::Admission;
pub use claim_cell::{ClaimCell, ClaimGuard};
pub use lifecycle::{ColdGuard, HotGuard, HotScope, LifecycleMutex, LifecycleRwLock};
pub use published::{EpochDomain, EpochGuard, EpochSlot, Published, Retired};
pub use set_once::SetOnce;
