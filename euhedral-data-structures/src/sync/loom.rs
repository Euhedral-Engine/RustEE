//! The atomics the cells in this module use. Under `--cfg loom` they are the model checker's, so a
//! loom test explores every interleaving; otherwise they are the standard library's.

#[cfg(loom)]
pub(crate) use loom::sync::atomic::{AtomicPtr, Ordering};
#[cfg(not(loom))]
pub(crate) use std::sync::atomic::{AtomicPtr, Ordering};
