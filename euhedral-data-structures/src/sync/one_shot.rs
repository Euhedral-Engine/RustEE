//! A one-shot hand-off of a value from one thread to another: Java's `CompletableFuture` with one completer and one
//! reader.
//!
//! [`one_shot`] makes the two ends. The sender writes the value and then stores `SET` with release ordering: one plain
//! store, with no read-modify-write and no barrier. A sender dropped without sending stores `ABANDONED`, which the
//! receiver sees as [`RecvError::Abandoned`]. The receiver never waits: it checks with [`try_take`](OneShotReceiver::try_take)
//! and [`is_set`](OneShotReceiver::is_set) when it chooses to. Nothing here parks a thread or wakes one.

use super::loom::{AtomicU8, Ordering, UnsafeCell};
use std::fmt;
use std::sync::Arc;

const EMPTY: u8 = 0;
const SET: u8 = 1;
const ABANDONED: u8 = 2;

/// Why a take can never produce a value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecvError {
    /// The sender was dropped without sending, or the value was already taken.
    Abandoned,
}

impl fmt::Display for RecvError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("the one-shot sender was dropped without sending, or the value was taken")
    }
}

impl std::error::Error for RecvError {}

struct Shared<T> {
    state: AtomicU8,
    /// Written by the sender before it stores `SET`, taken by the receiver after it saw `SET`; dropped with the shared
    /// state otherwise.
    value: UnsafeCell<Option<T>>,
}

// SAFETY: the value moves from the sender to the receiver through the state's release/acquire pair.
unsafe impl<T: Send> Send for Shared<T> {}
// SAFETY: as above.
unsafe impl<T: Send> Sync for Shared<T> {}

/// Makes a one-shot hand-off: the end that sends the value and the end that receives it.
pub fn one_shot<T: Send>() -> (OneShotSender<T>, OneShotReceiver<T>) {
    let shared = Arc::new(Shared {
        state: AtomicU8::new(EMPTY),
        value: UnsafeCell::new(None),
    });
    (
        OneShotSender {
            shared: Some(shared.clone()),
        },
        OneShotReceiver { shared },
    )
}

/// The end that sends. Dropping it without sending abandons the hand-off.
pub struct OneShotSender<T: Send> {
    /// `None` once sent.
    shared: Option<Arc<Shared<T>>>,
}

impl<T: Send> fmt::Debug for OneShotSender<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OneShotSender").finish_non_exhaustive()
    }
}

impl<T: Send> OneShotSender<T> {
    /// Hands `value` over. Never blocks and takes no read-modify-write. If the receiver is gone the value is dropped
    /// with the shared state.
    pub fn send(mut self, value: T) {
        let shared = self.shared.take().expect("a sender sends once");
        // SAFETY: nothing reads the value before the store below publishes it.
        shared.value.with_mut(|slot| unsafe { *slot = Some(value) });
        shared.state.store(SET, Ordering::Release);
    }
}

impl<T: Send> Drop for OneShotSender<T> {
    fn drop(&mut self) {
        if let Some(shared) = self.shared.take() {
            shared.state.store(ABANDONED, Ordering::Release);
        }
    }
}

/// The end that receives.
pub struct OneShotReceiver<T: Send> {
    shared: Arc<Shared<T>>,
}

impl<T: Send> fmt::Debug for OneShotReceiver<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OneShotReceiver").finish_non_exhaustive()
    }
}

impl<T: Send> OneShotReceiver<T> {
    /// Whether a value was sent: true from the sender's `send` on, also after the value was taken. An acquire load.
    pub fn is_set(&self) -> bool {
        self.shared.state.load(Ordering::Acquire) == SET
    }

    /// The value, if it was sent: `Ok(None)` while it was not, [`RecvError::Abandoned`] if the sender was dropped
    /// without sending, or if the value was taken already. Never waits.
    pub fn try_take(&mut self) -> Result<Option<T>, RecvError> {
        match self.shared.state.load(Ordering::Acquire) {
            EMPTY => Ok(None),
            SET => {
                // SAFETY: the state was `SET` (acquire), so the sender's write is visible and it writes no more;
                // this receiver is the only reader.
                let value = self.shared.value.with_mut(|slot| unsafe { (*slot).take() });
                value.map(Some).ok_or(RecvError::Abandoned)
            }
            _ => Err(RecvError::Abandoned),
        }
    }
}

#[cfg(test)]
mod tests;
