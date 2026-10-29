//! Cache-line padded atomics and lock-free queues used by Euhedral.
//!
//! Upstream pads fields with dead `long` members; here a `#[repr(align(128))]` wrapper does the same
//! job, applied to the same groups of fields in the same order. Java's `getPlain`/`setPlain` have no
//! atomic-free counterpart on a shared `&self`, so they map to relaxed operations.
//!
//! The queues replace "any thread may call any method, by convention" with role handles: a queue hands
//! out producer and consumer handles, a *single* role has at most one live handle and takes `&mut self`,
//! and a *multi* role is `Clone`. The public API is entirely safe.

pub mod atomics;
pub mod queues;
pub mod sync;

/// Compiles and runs the examples in the crate README as doctests.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;
