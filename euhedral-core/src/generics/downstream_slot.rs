//! The one receiver a source pushes to.

use super::{LatticeReceiver, LatticeSource};
use crate::frames::Failure;
use std::cell::UnsafeCell;
use std::error::Error;
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// Returned to a receiver that attaches to a source that already has one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AlreadyAttached;

impl fmt::Display for AlreadyAttached {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("the source already has a downstream")
    }
}

impl Error for AlreadyAttached {}

/// A source's receiver, installed once, completed once, and read without a lock.
///
/// - `attached` admits one receiver.
/// - `complete` is the source finishing.
/// - `live` is "a receiver is installed and not yet told the source completed". Whichever side
///   clears it, [`attach`](Self::attach) or [`complete`](Self::complete), owns the receiver's
///   `on_complete`, so the receiver hears it exactly once however the two race.
///
/// The receiver's `Arc` stays in the slot until the slot drops. A worker may be reading it through
/// [`receiver`](Self::receiver) when the source completes, and clearing it under that reader would
/// need reclamation; `live` is what ends its use.
///
/// The `Arc` sits inline in the slot, with no atomic of its own: `attached` admits one writer, which writes it before
/// it stores `live` (sequentially consistent, so a release), and every reader gets to it through a `live` it saw set
/// (an acquire load, or the swap in `complete`). A read costs the `live` load and no pointer hop.
///
/// Every method touches atomics only. A source's lifecycle calls (`complete`, and any clear or kill
/// it offers) may arrive from any thread while a worker is inside its `request` or `pull`, so they
/// set atomic state, and the work that needs the source's plain state ([`SourceCell`](super::SourceCell))
/// happens in the source's next `request` or `pull`, where it is the only user.
pub struct DownstreamSlot {
    attached: AtomicBool,
    complete: AtomicBool,
    live: AtomicBool,
    /// Written once, by the attach that won `attached`, before it sets `live`; read only after a `live` that was set.
    receiver: UnsafeCell<Option<Arc<dyn LatticeReceiver>>>,
}

// SAFETY: the receiver is written once, before the release store of `live` that every reader acquires, and is
// shared as `&Arc<dyn LatticeReceiver>` after that, which `LatticeReceiver: Send + Sync` allows.
unsafe impl Sync for DownstreamSlot {}

impl DownstreamSlot {
    pub const fn new() -> Self {
        Self {
            attached: AtomicBool::new(false),
            complete: AtomicBool::new(false),
            live: AtomicBool::new(false),
            receiver: UnsafeCell::new(None),
        }
    }

    /// Takes `receiver` as the downstream of `source`.
    ///
    /// - A second receiver is told [`on_error`](LatticeReceiver::on_error) and the first is left
    ///   alone.
    /// - Once the slot is complete, a receiver is told [`on_complete`](LatticeReceiver::on_complete)
    ///   and nothing is installed.
    /// - Otherwise the receiver is installed and `live` is set. If the slot completed meanwhile,
    ///   the attach that clears `live` delivers `on_complete`. The receiver is then handed
    ///   `source` through [`add_upstream`](super::LatticeTerminal::add_upstream).
    pub fn attach(&self, source: Arc<dyn LatticeSource>, receiver: Arc<dyn LatticeReceiver>) {
        if self
            .attached
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            receiver.on_error(Failure::from(AlreadyAttached));
            return;
        }
        if self.is_complete() {
            receiver.on_complete();
            return;
        }
        // SAFETY: `attached` admitted this call alone, and no reader looks before `live` is set below.
        unsafe { *self.receiver.get() = Some(receiver.clone()) };
        self.live.store(true, Ordering::SeqCst);
        // `complete` may have run between the check above and the store. Whichever side clears
        // `live` owns the notification.
        if self.complete.load(Ordering::SeqCst)
            && self
                .live
                .compare_exchange(true, false, Ordering::SeqCst, Ordering::SeqCst)
                .is_ok()
        {
            receiver.on_complete();
        }
        receiver.add_upstream(source);
    }

    /// The receiver while the slot is live: installed and not completed.
    pub fn receiver(&self) -> Option<&Arc<dyn LatticeReceiver>> {
        if self.live.load(Ordering::Acquire) {
            // SAFETY: `live` was set after the receiver was written, and nothing writes it again.
            unsafe { (*self.receiver.get()).as_ref() }
        } else {
            None
        }
    }

    /// Completes the slot; the call that does so tells a live receiver `on_complete`. Returns
    /// whether this call completed it.
    pub fn complete(&self) -> bool {
        if self
            .complete
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return false;
        }
        if self.live.swap(false, Ordering::SeqCst) {
            // SAFETY: the swap saw `live` set, which was after the receiver was written; nothing writes it again.
            if let Some(receiver) = unsafe { (*self.receiver.get()).as_ref() } {
                receiver.on_complete();
            }
        }
        true
    }

    pub fn is_complete(&self) -> bool {
        self.complete.load(Ordering::Acquire)
    }
}

impl Default for DownstreamSlot {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for DownstreamSlot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DownstreamSlot")
            .field("attached", &self.attached.load(Ordering::Relaxed))
            .field("complete", &self.complete.load(Ordering::Relaxed))
            .field("live", &self.live.load(Ordering::Relaxed))
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frames::Frame;
    use crate::generics::LatticeTerminal;
    use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};
    use std::sync::{Barrier, Mutex};
    use std::thread;

    /// A source that is only the identity `attach` hands on.
    struct Nothing;

    impl LatticeSource for Nothing {
        fn add_downstream(self: Arc<Self>, _downstream: Arc<dyn LatticeReceiver>) {}
        fn pull(
            &self,
            _consumer: &mut dyn FnMut(Box<dyn Frame>),
            _stop: &mut dyn FnMut(&dyn Frame) -> bool,
            _demand: u64,
        ) -> u64 {
            0
        }
        fn request(&self, _demand: u64) {}
        fn complete(&self) {}
        fn is_complete(&self) -> bool {
            false
        }
    }

    fn source() -> Arc<dyn LatticeSource> {
        Arc::new(Nothing)
    }

    /// Counts what it is told.
    #[derive(Default)]
    struct Counting {
        completes: AtomicUsize,
        errors: Mutex<Vec<String>>,
        upstreams: AtomicUsize,
    }

    impl LatticeTerminal for Counting {
        fn add_upstream(&self, _upstream: Arc<dyn LatticeSource>) {
            self.upstreams.fetch_add(1, SeqCst);
        }
    }

    impl LatticeReceiver for Counting {
        fn push(&self, _frame: Box<dyn Frame>) {}

        fn on_complete(&self) {
            self.completes.fetch_add(1, SeqCst);
        }

        fn on_error(&self, error: Failure) {
            self.errors.lock().unwrap().push(error.to_string());
        }
    }

    #[test]
    fn attach_then_complete_tells_the_receiver_once() {
        let slot = DownstreamSlot::new();
        let receiver = Arc::new(Counting::default());
        slot.attach(source(), receiver.clone());
        assert_eq!(receiver.upstreams.load(SeqCst), 1);
        assert!(slot.receiver().is_some());
        assert!(slot.complete());
        assert!(!slot.complete());
        assert_eq!(receiver.completes.load(SeqCst), 1);
        assert!(slot.is_complete());
    }

    #[test]
    fn complete_before_attach_tells_the_late_receiver_once() {
        let slot = DownstreamSlot::new();
        assert!(slot.complete());
        let receiver = Arc::new(Counting::default());
        slot.attach(source(), receiver.clone());
        assert_eq!(receiver.completes.load(SeqCst), 1);
        assert!(slot.receiver().is_none());
        assert!(!slot.complete());
        assert_eq!(receiver.completes.load(SeqCst), 1);
    }

    #[test]
    fn a_second_attach_is_refused_with_an_error() {
        let slot = DownstreamSlot::new();
        let first = Arc::new(Counting::default());
        let second = Arc::new(Counting::default());
        slot.attach(source(), first.clone());
        slot.attach(source(), second.clone());
        assert_eq!(
            *second.errors.lock().unwrap(),
            ["the source already has a downstream"]
        );
        assert_eq!(second.upstreams.load(SeqCst), 0);
        assert!(first.errors.lock().unwrap().is_empty());
        assert!(slot.receiver().is_some());
        slot.complete();
        assert_eq!(first.completes.load(SeqCst), 1);
        assert_eq!(second.completes.load(SeqCst), 0);
    }

    #[test]
    fn attach_racing_complete_tells_the_receiver_exactly_once() {
        const ROUNDS: usize = 10_000;
        let barrier = Arc::new(Barrier::new(2));
        let slots: Arc<Vec<DownstreamSlot>> =
            Arc::new((0..ROUNDS).map(|_| DownstreamSlot::new()).collect());
        let receivers: Arc<Vec<Arc<Counting>>> =
            Arc::new((0..ROUNDS).map(|_| Arc::new(Counting::default())).collect());
        let completer = {
            let (slots, barrier) = (slots.clone(), barrier.clone());
            thread::spawn(move || {
                barrier.wait();
                for slot in slots.iter() {
                    slot.complete();
                }
            })
        };
        barrier.wait();
        for (slot, receiver) in slots.iter().zip(receivers.iter()) {
            slot.attach(source(), receiver.clone());
        }
        completer.join().unwrap();
        for (round, receiver) in receivers.iter().enumerate() {
            assert_eq!(receiver.completes.load(SeqCst), 1, "round {round}");
        }
    }

    #[test]
    fn receiver_is_none_after_complete_and_the_arc_lives_until_the_slot_drops() {
        let slot = DownstreamSlot::new();
        let receiver = Arc::new(Counting::default());
        slot.attach(source(), receiver.clone());
        assert_eq!(Arc::strong_count(&receiver), 2);
        slot.complete();
        assert!(slot.receiver().is_none());
        assert_eq!(
            Arc::strong_count(&receiver),
            2,
            "a reader may still hold the slot's Arc"
        );
        drop(slot);
        assert_eq!(Arc::strong_count(&receiver), 1);
    }
}
