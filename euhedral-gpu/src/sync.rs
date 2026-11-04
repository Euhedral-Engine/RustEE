//! The atomics the lock-free marker ring is built from. A build with `--cfg loom` swaps them for the `loom` model
//! checker's, which explores every interleaving the memory model allows.

#[cfg(loom)]
pub(crate) use loom::sync::atomic::{AtomicU32, AtomicU64, Ordering};
#[cfg(not(loom))]
pub(crate) use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
