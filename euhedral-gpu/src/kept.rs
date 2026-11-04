//! Values an owned submitter holds until a retirement is confirmed.
//!
//! Nothing here is shared mutably. A submitter owns its [`Kept`] list and changes it through `&mut self`. (The values a
//! recording keeps are in [`hold`](crate::hold).)

use euhedral_data_structures::queues::PlainQueue;
use std::any::Any;

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
    use std::sync::Arc;

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
}
