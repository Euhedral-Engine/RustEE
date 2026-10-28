//! Values held until a retirement is confirmed.

use std::any::Any;
use std::collections::VecDeque;
use std::sync::{Mutex, PoisonError};

/// Values held until a retirement is confirmed: `(retirement seq, value)`, released in order.
#[derive(Default)]
pub(crate) struct Kept {
    values: Mutex<VecDeque<(u64, Box<dyn Any + Send>)>>,
}

impl std::fmt::Debug for Kept {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Kept").finish_non_exhaustive()
    }
}

impl Kept {
    pub(crate) fn push(&self, seq: u64, value: Box<dyn Any + Send>) {
        self.values
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push_back((seq, value));
    }

    /// Drops every value kept for a retirement numbered `seq` or lower.
    pub(crate) fn release_through(&self, seq: u64) {
        let mut released = Vec::new();
        {
            let mut values = self.values.lock().unwrap_or_else(PoisonError::into_inner);
            while values.front().is_some_and(|(s, _)| *s <= seq) {
                released.extend(values.pop_front());
            }
        }
        drop(released);
    }

    /// Drops everything kept.
    pub(crate) fn clear(&self) {
        let all = std::mem::take(&mut *self.values.lock().unwrap_or_else(PoisonError::into_inner));
        drop(all);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn kept_values_release_in_order_through_a_seq() {
        let kept = Kept::default();
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
