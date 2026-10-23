//! Cache-line padded atomics and lock-free queues used by Euhedral.
//!
//! Upstream pads fields with dead `long` members; here a `#[repr(align(128))]` wrapper does the same
//! job. Java's `getPlain`/`setPlain` have no atomic-free counterpart on a shared `&self`, so they map to
//! relaxed operations.

pub mod atomics;
pub mod queues;
