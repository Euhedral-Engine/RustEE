//! Values held until a retirement is confirmed, and the values a recording keeps.
//!
//! Nothing here is shared mutably. A submitter owns its [`Kept`] list and changes it through `&mut self`. A recording
//! collects its values in a `Vec` while it runs, then freezes them into an immutable [`Frozen`] that the graph and the
//! streams the recording ran on share by `Arc`.

use std::any::Any;
use std::sync::Arc;

use euhedral_data_structures::queues::PlainQueue;

/// A kept value, shared only to be dropped. It is never read, so sharing it between threads cannot race: whichever
/// thread holds the last handle drops it, which a `Send` value permits.
pub(crate) struct Held(#[allow(dead_code, reason = "held to be dropped")] Box<dyn Any + Send>);

// SAFETY: a `Held` is never accessed, only dropped, and the value inside is `Send`.
unsafe impl Sync for Held {}

impl std::fmt::Debug for Held {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Held")
    }
}

/// The values a recording keeps, frozen: shared by the recording's own run, its graph and every replay. Each holds a
/// handle through the streams it ran on, so the buffers live until the last of them retired.
pub(crate) type Frozen = Arc<[Held]>;

/// Freezes the values a recording collected.
pub(crate) fn freeze(values: Vec<Box<dyn Any + Send>>) -> Frozen {
    values.into_iter().map(Held).collect()
}

/// Values held until a retirement is confirmed: `(retirement seq, value)`, released in order. Owned by one submitter:
/// a `PlainQueue`, the single-owner FIFO of `euhedral-data-structures`, which needs no synchronisation.
pub(crate) struct Kept {
    values: PlainQueue<(u64, Box<dyn Any + Send>)>,
}

impl Default for Kept {
    fn default() -> Self {
        Self {
            values: PlainQueue::new(16),
        }
    }
}

impl std::fmt::Debug for Kept {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Kept")
            .field("values", &self.values.size_long())
            .finish()
    }
}

impl Kept {
    pub(crate) fn push(&mut self, seq: u64, value: Box<dyn Any + Send>) {
        // An unbounded queue never refuses; if it did, the value would be dropped here rather than lost.
        let _ = self.values.offer((seq, value));
    }

    /// Drops every value kept for a retirement numbered `seq` or lower.
    pub(crate) fn release_through(&mut self, seq: u64) {
        self.values
            .drain_while(drop, |(kept_for, _)| *kept_for > seq, u64::MAX);
    }

    /// Drops everything kept.
    #[cfg_attr(
        not(feature = "cuda"),
        allow(dead_code, reason = "used by the CUDA stream's drop")
    )]
    pub(crate) fn clear(&mut self) {
        self.values.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kept_values_release_in_order_through_a_seq() {
        let mut kept = Kept::default();
        let (a, b, c) = (Arc::new(()), Arc::new(()), Arc::new(()));
        kept.push(1, Box::new(a.clone()));
        kept.push(1, Box::new(b.clone()));
        kept.push(2, Box::new(c.clone()));
        kept.release_through(1);
        assert_eq!(
            (
                Arc::strong_count(&a),
                Arc::strong_count(&b),
                Arc::strong_count(&c)
            ),
            (1, 1, 2)
        );
        kept.clear();
        assert_eq!(Arc::strong_count(&c), 1);
    }

    #[test]
    fn frozen_values_live_until_the_last_handle_drops() {
        let token = Arc::new(());
        let frozen = freeze(vec![Box::new(token.clone())]);
        let second = frozen.clone();
        drop(frozen);
        assert_eq!(Arc::strong_count(&token), 2);
        drop(second);
        assert_eq!(Arc::strong_count(&token), 1);
    }
}
