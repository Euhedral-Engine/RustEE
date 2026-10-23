use crate::generics::{LatticeReceiver, LatticeSource};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, PoisonError, RwLock};

/// The one receiver an ingest source pushes to, its completion, and the demand it asked for.
///
/// The receiver hears of completion exactly once, whether it attaches before or after.
#[derive(Default)]
pub(super) struct Downstream {
    receiver: RwLock<Option<Arc<dyn LatticeReceiver>>>,
    attached: AtomicBool,
    complete: AtomicBool,
    demand: AtomicU64,
}

impl Downstream {
    /// Attaches `receiver` below `source`, telling it of `source`, or of completion if `source` has
    /// completed. A second receiver is told of the error instead.
    pub(super) fn attach(
        &self,
        source: Arc<dyn LatticeSource>,
        receiver: Arc<dyn LatticeReceiver>,
    ) {
        if self.attached.swap(true, Ordering::AcqRel) {
            receiver.on_error("the source already has a downstream".into());
            return;
        }
        {
            let mut slot = self
                .receiver
                .write()
                .unwrap_or_else(PoisonError::into_inner);
            if !self.complete.load(Ordering::Acquire) {
                *slot = Some(receiver.clone());
                drop(slot);
                receiver.add_upstream(source);
                return;
            }
        }
        receiver.on_complete();
    }

    /// The attached receiver, unless the source has completed.
    pub(super) fn receiver(&self) -> Option<Arc<dyn LatticeReceiver>> {
        self.receiver
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    /// Completes the source: forgets the demand and the receiver, then tells the receiver, which
    /// only the first call finds.
    pub(super) fn complete(&self) {
        self.complete.store(true, Ordering::Release);
        self.demand.store(0, Ordering::Release);
        let receiver = self
            .receiver
            .write()
            .unwrap_or_else(PoisonError::into_inner)
            .take();
        if let Some(receiver) = receiver {
            receiver.on_complete();
        }
    }

    pub(super) fn is_complete(&self) -> bool {
        self.complete.load(Ordering::Acquire)
    }

    /// Adds `demand`, saturating at unbounded, and returns the total.
    pub(super) fn add_demand(&self, demand: u64) -> u64 {
        let previous = self
            .demand
            .try_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                Some(current.saturating_add(demand))
            })
            .unwrap_or_else(|current| current);
        previous.saturating_add(demand)
    }

    /// Takes `delivered` off the demand, stopping at zero.
    pub(super) fn deliver(&self, delivered: u64) {
        let _ = self
            .demand
            .try_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                Some(current.saturating_sub(delivered))
            });
    }

    pub(super) fn demand(&self) -> u64 {
        self.demand.load(Ordering::Acquire)
    }
}
