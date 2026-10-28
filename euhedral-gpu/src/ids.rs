//! Identities of streams.

use std::sync::atomic::{AtomicU64, Ordering};

static NEXT: AtomicU64 = AtomicU64::new(1);

/// A stream identity, unique in the process and never 0.
#[cfg_attr(
    not(any(feature = "fake", feature = "cuda")),
    allow(dead_code, reason = "used by the backends")
)]
pub(crate) fn next_stream_id() -> u64 {
    NEXT.fetch_add(1, Ordering::Relaxed)
}
