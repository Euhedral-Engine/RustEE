//! The one receiver a source pushes to.

use super::SourceCell;
use super::{LatticeReceiver, LatticeSource};
use crate::frames::Failure;
use euhedral_data_structures::queues::{Consumer, MpscQueue, Multi, Producer, Single};
use euhedral_data_structures::sync::SetOnce;
use std::error::Error;
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering, fence};

type Action = Box<dyn FnOnce() + Send>;

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
pub struct DownstreamSlot {
    attached: AtomicBool,
    complete: AtomicBool,
    live: AtomicBool,
    receiver: SetOnce<Arc<dyn LatticeReceiver>>,
    /// Actions handed to [`serialize`](Self::serialize) while a receiver is attaching, before it is
    /// installed. Whoever finds the receiver installed forwards them to it.
    deferred: Producer<Action, Multi>,
    forward: SourceCell<Consumer<Action, Single>>,
    forwarding: AtomicBool,
}

impl DownstreamSlot {
    pub fn new() -> Self {
        let deferred = MpscQueue::new(8);
        Self {
            attached: AtomicBool::new(false),
            complete: AtomicBool::new(false),
            live: AtomicBool::new(false),
            receiver: SetOnce::new(),
            deferred: deferred.producer(),
            forward: SourceCell::new(deferred.consumer()),
            forwarding: AtomicBool::new(false),
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
        let installed = self.receiver.set(receiver.clone());
        debug_assert!(installed.is_ok(), "the attach flag admits one receiver");
        if self.is_complete() {
            // Nothing will push, but lifecycle actions still go through the receiver.
            self.forward_deferred();
            receiver.on_complete();
            return;
        }
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
        self.forward_deferred();
    }

    /// Hands the actions deferred during an attach to the receiver, if it is installed.
    ///
    /// Exactly one caller forwards at a time (the `forwarding` flag), and a caller that finds the
    /// flag taken relies on the forwarder, which looks at the queue again after clearing it. A
    /// fence on each side orders an offer before the look at the flag, and the clear before the
    /// look at the queue, so one of the two always sees the other.
    fn forward_deferred(&self) {
        fence(Ordering::SeqCst);
        let Some(receiver) = self.receiver.get() else {
            return;
        };
        loop {
            if self
                .forwarding
                .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
                .is_err()
            {
                return;
            }
            {
                // SAFETY: the `forwarding` flag admits one thread at a time.
                let mut forward = unsafe { self.forward.enter() };
                while let Some(action) = forward.poll() {
                    receiver.serialize(action);
                }
            }
            self.forwarding.store(false, Ordering::SeqCst);
            fence(Ordering::SeqCst);
            if self.deferred.is_empty() {
                return;
            }
        }
    }

    /// The receiver while the slot is live: installed and not completed.
    pub fn receiver(&self) -> Option<&Arc<dyn LatticeReceiver>> {
        if self.live.load(Ordering::Acquire) {
            self.receiver.get()
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
        if self.live.swap(false, Ordering::SeqCst)
            && let Some(receiver) = self.receiver.get()
        {
            receiver.on_complete();
        }
        true
    }

    pub fn is_complete(&self) -> bool {
        self.complete.load(Ordering::Acquire)
    }

    /// Hands `action` to the attached receiver's [`serialize`](LatticeReceiver::serialize), which
    /// runs it when no `pull` or `request` of the source is in progress.
    ///
    /// - With a receiver installed, the action goes straight to it.
    /// - While a receiver is attaching and not yet installed, the action waits in the slot and the
    ///   attach forwards it, so it never runs beside the attach.
    /// - With no attach begun there is no lattice to wait for, and it runs at once.
    pub fn serialize(&self, action: Box<dyn FnOnce() + Send>) {
        if let Some(receiver) = self.receiver.get() {
            receiver.serialize(action);
        } else if self.attached.load(Ordering::SeqCst) {
            if self.deferred.offer(action).is_err() {
                unreachable!("the deferred queue is unbounded");
            }
            self.forward_deferred();
        } else {
            action();
        }
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

    /// Counts what it is told and records `serialize` calls.
    #[derive(Default)]
    struct Counting {
        completes: AtomicUsize,
        errors: Mutex<Vec<String>>,
        upstreams: AtomicUsize,
        serialized: AtomicUsize,
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

        fn serialize(&self, action: Box<dyn FnOnce() + Send>) {
            self.serialized.fetch_add(1, SeqCst);
            action();
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

    #[test]
    fn serialize_without_a_receiver_runs_at_once() {
        let slot = DownstreamSlot::new();
        let ran = Arc::new(AtomicUsize::new(0));
        let counted = ran.clone();
        slot.serialize(Box::new(move || {
            counted.fetch_add(1, SeqCst);
        }));
        assert_eq!(ran.load(SeqCst), 1);
    }

    #[test]
    fn serialize_goes_through_the_attached_receiver() {
        let slot = DownstreamSlot::new();
        let receiver = Arc::new(Counting::default());
        slot.attach(source(), receiver.clone());
        let ran = Arc::new(AtomicUsize::new(0));
        for _ in 0..3 {
            let counted = ran.clone();
            slot.serialize(Box::new(move || {
                counted.fetch_add(1, SeqCst);
            }));
        }
        assert_eq!(receiver.serialized.load(SeqCst), 3);
        assert_eq!(ran.load(SeqCst), 3);
        slot.complete();
        let counted = ran.clone();
        slot.serialize(Box::new(move || {
            counted.fetch_add(1, SeqCst);
        }));
        assert_eq!(
            receiver.serialized.load(SeqCst),
            4,
            "lifecycle actions still reach a completed slot's receiver"
        );
    }

    #[test]
    fn serialize_racing_attach_runs_every_action_exactly_once() {
        const ROUNDS: usize = 5_000;
        let slots: Arc<Vec<DownstreamSlot>> =
            Arc::new((0..ROUNDS).map(|_| DownstreamSlot::new()).collect());
        let receivers: Arc<Vec<Arc<Counting>>> =
            Arc::new((0..ROUNDS).map(|_| Arc::new(Counting::default())).collect());
        let ran: Arc<Vec<AtomicUsize>> =
            Arc::new((0..ROUNDS).map(|_| AtomicUsize::new(0)).collect());
        let start = Arc::new(AtomicUsize::new(0));
        let finished = Arc::new(AtomicUsize::new(0));
        let spin_until = |flag: &AtomicUsize, value: usize| {
            while flag.load(SeqCst) < value {
                std::hint::spin_loop();
            }
        };
        let attacher = {
            let (slots, receivers, start, finished) = (
                slots.clone(),
                receivers.clone(),
                start.clone(),
                finished.clone(),
            );
            thread::spawn(move || {
                for round in 0..ROUNDS {
                    spin_until(&start, round + 1);
                    slots[round].attach(source(), receivers[round].clone());
                    finished.fetch_add(1, SeqCst);
                }
            })
        };
        let serializers: Vec<_> = (0..2)
            .map(|_| {
                let (slots, ran, start, finished) =
                    (slots.clone(), ran.clone(), start.clone(), finished.clone());
                thread::spawn(move || {
                    for round in 0..ROUNDS {
                        spin_until(&start, round + 1);
                        let counted = ran.clone();
                        slots[round].serialize(Box::new(move || {
                            counted[round].fetch_add(1, SeqCst);
                        }));
                        finished.fetch_add(1, SeqCst);
                    }
                })
            })
            .collect();
        for round in 0..ROUNDS {
            start.store(round + 1, SeqCst);
            spin_until(&finished, (round + 1) * 3);
            assert_eq!(ran[round].load(SeqCst), 2, "round {round}");
        }
        attacher.join().unwrap();
        for serializer in serializers {
            serializer.join().unwrap();
        }
    }

    #[test]
    fn an_action_waits_for_an_attach_in_flight_and_goes_through_its_receiver() {
        /// Holds the attach inside `add_upstream` until the test lets go.
        struct Gate {
            entered: std::sync::mpsc::SyncSender<()>,
            go: Mutex<std::sync::mpsc::Receiver<()>>,
            serialized: AtomicUsize,
        }
        impl LatticeTerminal for Gate {
            fn add_upstream(&self, _upstream: Arc<dyn LatticeSource>) {
                self.entered.send(()).unwrap();
                self.go.lock().unwrap().recv().unwrap();
            }
        }
        impl LatticeReceiver for Gate {
            fn push(&self, _frame: Box<dyn Frame>) {}
            fn on_complete(&self) {}
            fn on_error(&self, _error: Failure) {}
            fn serialize(&self, action: Box<dyn FnOnce() + Send>) {
                self.serialized.fetch_add(1, SeqCst);
                action();
            }
        }

        let (entered_tx, entered_rx) = std::sync::mpsc::sync_channel(0);
        let (go_tx, go_rx) = std::sync::mpsc::channel();
        let gate = Arc::new(Gate {
            entered: entered_tx,
            go: Mutex::new(go_rx),
            serialized: AtomicUsize::new(0),
        });
        let slot = Arc::new(DownstreamSlot::new());
        let attach = {
            let (slot, gate) = (slot.clone(), gate.clone());
            thread::spawn(move || slot.attach(source(), gate))
        };
        entered_rx.recv().unwrap();

        let ran = Arc::new(AtomicUsize::new(0));
        let counted = ran.clone();
        slot.serialize(Box::new(move || {
            counted.fetch_add(1, SeqCst);
        }));
        assert_eq!(
            gate.serialized.load(SeqCst),
            1,
            "it went through the receiver"
        );
        assert_eq!(ran.load(SeqCst), 1);

        go_tx.send(()).unwrap();
        attach.join().unwrap();
    }
}
