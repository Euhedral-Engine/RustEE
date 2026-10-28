//! The mutual exclusion of a stream's submissions.

use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard, PoisonError};

use crate::GpuError;

thread_local! {
    static TOKEN: u8 = const { 0 };
}

/// A token that differs between live threads: the address of a thread-local.
fn thread_token() -> u64 {
    TOKEN.with(|t| std::ptr::from_ref(t) as u64)
}

/// A mutex that reports a second lock from the thread that already holds it, instead of deadlocking.
///
/// It gives a submission contiguous operations on its stream. It is held only while enqueueing, which is a few
/// non-blocking driver calls.
#[derive(Debug, Default)]
pub struct StreamLock<T> {
    inner: Mutex<T>,
    /// The token of the thread holding `inner`, else 0.
    holder: AtomicU64,
}

/// A held [`StreamLock`].
#[derive(Debug)]
pub struct StreamGuard<'a, T> {
    guard: MutexGuard<'a, T>,
    holder: &'a AtomicU64,
}

impl<T> StreamLock<T> {
    pub fn new(value: T) -> Self {
        Self {
            inner: Mutex::new(value),
            holder: AtomicU64::new(0),
        }
    }

    /// Locks. `Err(Invalid("re-entered"))` when the calling thread already holds the lock. A lock poisoned by a
    /// panic elsewhere is recovered.
    pub fn lock(&self) -> Result<StreamGuard<'_, T>, GpuError> {
        let token = thread_token();
        if self.holder.load(Ordering::Acquire) == token {
            return Err(GpuError::Invalid("re-entered".into()));
        }
        let guard = self.inner.lock().unwrap_or_else(PoisonError::into_inner);
        self.holder.store(token, Ordering::Release);
        Ok(StreamGuard {
            guard,
            holder: &self.holder,
        })
    }
}

impl<T> Deref for StreamGuard<'_, T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.guard
    }
}

impl<T> DerefMut for StreamGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.guard
    }
}

impl<T> Drop for StreamGuard<'_, T> {
    fn drop(&mut self) {
        // Cleared while the mutex is still held; the field drops after this.
        self.holder.store(0, Ordering::Release);
    }
}

#[cfg(test)]
#[allow(
    clippy::disallowed_methods,
    reason = "a peer thread stands in for another lattice worker"
)]
mod tests {
    use super::*;
    use crate::GpuError;

    #[test]
    fn a_stream_lock_reports_reentry_and_serves_other_threads() {
        let lock = StreamLock::new(5u32);
        {
            let mut guard = lock.lock().unwrap();
            *guard += 1;
            assert!(matches!(lock.lock(), Err(GpuError::Invalid(_))));
        }
        std::thread::scope(|s| {
            s.spawn(|| {
                assert_eq!(*lock.lock().unwrap(), 6);
            });
        });
        assert_eq!(
            *lock.lock().unwrap(),
            6,
            "released after the nested refusal"
        );
    }
}
