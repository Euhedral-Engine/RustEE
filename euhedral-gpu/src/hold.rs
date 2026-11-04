//! What a recording or a replay on a shared lane hands to its caller to hold until the device work ended.
//!
//! An owned stream keeps the values a graph's work reads (the host buffers its copies read) in its own list, released when
//! a retirement armed after the work is confirmed. A shared lane keeps none, since any worker submits to it and a
//! retirement armed by one party must not release another's. So the values go to the caller, in a [`GraphHold`]: the stage
//! that recorded, replayed or ran a sequence holds it in its quantum and drops it in [`Stage::retired`], which runs after
//! the quantum's device work ended (Java `StageFrame.retired`).
//!
//! Nothing here is shared mutably. A recording collects its values in a `Vec` while it runs, then freezes them into an
//! immutable [`Frozen`] that the graph and the holds of the runs that used it share by `Arc`: cloning one allocates
//! nothing.
//!
//! [`Stage::retired`]: euhedral_graph::Stage::retired

use std::any::Any;
use std::sync::Arc;

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
/// handle (through its stream's kept list, or its caller's [`GraphHold`]), so the buffers live until the last of them
/// ended.
pub(crate) type Frozen = Option<Arc<[Held]>>;

/// Freezes the values a recording collected: `None` for none, which allocates nothing.
pub(crate) fn freeze(values: Vec<Box<dyn Any + Send>>) -> Frozen {
    if values.is_empty() {
        return None;
    }
    Some(values.into_iter().map(Held).collect())
}

/// The values the work a caller submitted through a shared lane reads: held by the caller until that work ended.
///
/// A recording, a replay and a direct run of a sequence add to the hold they are given: a recording or a replay a handle
/// to the graph's frozen values (an `Arc` clone, no allocation), a direct run the values its body kept. The caller keeps
/// the hold in its quantum and drops it in [`Stage::retired`](euhedral_graph::Stage::retired): not before, since the
/// device reads those buffers until the work ended, whatever became of the graph, the cache or the lane meanwhile.
///
/// An empty hold costs nothing. Holding the values of one run allocates nothing; a second run into the same hold, or a
/// direct run that keeps values, allocates a list.
#[derive(Debug, Default)]
pub struct GraphHold {
    first: Frozen,
    more: Vec<Arc<[Held]>>,
    direct: Vec<Held>,
}

impl GraphHold {
    /// An empty hold.
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether the hold keeps nothing alive.
    pub fn is_empty(&self) -> bool {
        self.first.is_none() && self.more.is_empty() && self.direct.is_empty()
    }

    /// Adds a handle to a recording's frozen values.
    pub(crate) fn add_frozen(&mut self, frozen: &Frozen) {
        let Some(frozen) = frozen else { return };
        if self.first.is_none() {
            self.first = Some(frozen.clone());
        } else {
            self.more.push(frozen.clone());
        }
    }

    /// Adds a value a direct run kept.
    pub(crate) fn add_value(&mut self, value: Box<dyn Any + Send>) {
        self.direct.push(Held(value));
    }

    /// Drops what the hold keeps alive. The device work it covers must have ended.
    pub fn release(&mut self) {
        self.first = None;
        self.more.clear();
        self.direct.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frozen_values_live_until_the_last_handle_drops() {
        let token = Arc::new(());
        let frozen = freeze(vec![Box::new(token.clone())]);
        assert!(
            freeze(Vec::new()).is_none(),
            "nothing kept, nothing allocated"
        );
        let second = frozen.clone();
        drop(frozen);
        assert_eq!(Arc::strong_count(&token), 2);
        drop(second);
        assert_eq!(Arc::strong_count(&token), 1);
    }

    #[test]
    fn a_hold_keeps_what_it_was_given_until_it_is_released_or_dropped() {
        let (a, b, c) = (Arc::new(()), Arc::new(()), Arc::new(()));
        let fa = freeze(vec![Box::new(a.clone())]);
        let fb = freeze(vec![Box::new(b.clone())]);
        let mut hold = GraphHold::new();
        assert!(hold.is_empty());
        hold.add_frozen(&fa);
        hold.add_frozen(&fb);
        hold.add_value(Box::new(c.clone()));
        drop((fa, fb));
        assert!(!hold.is_empty());
        assert_eq!(
            (
                Arc::strong_count(&a),
                Arc::strong_count(&b),
                Arc::strong_count(&c)
            ),
            (2, 2, 2)
        );
        hold.release();
        assert!(hold.is_empty());
        assert_eq!(
            (
                Arc::strong_count(&a),
                Arc::strong_count(&b),
                Arc::strong_count(&c)
            ),
            (1, 1, 1)
        );
        let mut again = GraphHold::new();
        again.add_value(Box::new(c.clone()));
        drop(again);
        assert_eq!(Arc::strong_count(&c), 1);
    }
}
