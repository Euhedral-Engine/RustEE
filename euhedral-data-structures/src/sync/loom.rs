//! The atomics the cells in this module use. Under `--cfg loom` they are the model checker's, so a
//! loom test explores every interleaving; otherwise they are the standard library's.

#[cfg(loom)]
pub(crate) use loom::sync::atomic::{AtomicBool, AtomicI64, AtomicPtr, AtomicU64, Ordering, fence};
#[cfg(not(loom))]
pub(crate) use std::sync::atomic::{AtomicBool, AtomicI64, AtomicPtr, AtomicU64, Ordering, fence};
