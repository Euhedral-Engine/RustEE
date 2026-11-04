//! Retirement into [`Completions`], and the channel that carries a retirement from the backend's callback to the
//! stream's owner.
//!
//! **No allocation, free or lock in the callback.** Each stream owns one unbounded, lock-free MPSC queue (`Queue<_,
//! Multi, Single>` of `euhedral-data-structures`, chunk-growing, with a small pool of chunks so that steady state
//! recycles them). The backend's callback pushes the retirement's status and nothing else ([`RetireSink::push`]). The
//! owner ([`Retirements`]) keeps what to do with each retirement, by number, in a `PlainQueue` of its own, drains the
//! queue when it polls, and dispatches. The listeners are made and dropped inside the owning source's calls only.
//!
//! **The retirement number is not sent.** A stream's callbacks run in the order they were enqueued, and every armed
//! retirement enqueues exactly one, so the n-th status that arrives belongs to the n-th retirement armed. Carrying the
//! number would need a second word of user data per callback (the driver gives one pointer), that is, an allocation
//! per retirement. Where the order could break (a retirement the poisoned device never reports), the owner reports the
//! rest as [`POISONED_STATUS`] and drops what arrives later (see [`Retirements::poll`]).

#![cfg_attr(
    not(any(feature = "fake", feature = "cuda")),
    allow(dead_code, reason = "the retirement channel is used by the backends")
)]

use std::any::Any;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use euhedral_core::frames::{Frame, FrameError, FrameHeader};
use euhedral_data_structures::queues::{Consumer, Multi, PlainQueue, Producer, Queue, Single};
use euhedral_data_structures::sync::ClaimCell;
use euhedral_graph::{Completions, LaneId, Pending, QueueId};

#[cfg(any(feature = "fake", feature = "cuda"))]
use crate::kept::Kept;
use crate::{GpuError, POISONED_STATUS, RetireListener, RetireTicket, Stream};

/// A frame that runs a closure once and owns whatever the closure moved in: the frame of a chain that carries a
/// submitter (and a graph) from one continuation to the next.
///
/// [`RunnableFrame`](euhedral_core::frames::RunnableFrame) runs a shared `Fn`, which cannot give away what it holds;
/// this runs an `FnOnce` through `&mut self`, so the closure may own a submitter and hand it on, with no lock or
/// atomic to take it out of. A frame dropped without running drops the closure, and what it owns, where it is
/// dropped.
pub struct OnceFrame {
    header: FrameHeader,
    run: Option<Box<dyn FnOnce() + Send>>,
}

impl OnceFrame {
    /// An ordered frame of lane `id_hash` that runs `run` once.
    pub fn new(id_hash: u64, run: impl FnOnce() + Send + 'static) -> Self {
        Self {
            header: FrameHeader::new(id_hash),
            run: Some(Box::new(run)),
        }
    }
}

impl std::fmt::Debug for OnceFrame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OnceFrame")
            .field("pending", &self.run.is_some())
            .finish()
    }
}

impl Frame for OnceFrame {
    fn header(&self) -> &FrameHeader {
        &self.header
    }

    fn header_mut(&mut self) -> &mut FrameHeader {
        &mut self.header
    }

    fn execute(&mut self) -> Result<(), FrameError> {
        if let Some(run) = self.run.take() {
            run();
        }
        Ok(())
    }
}

/// A stream (and the state that travelled with it) handed back with the error that stopped [`retire_into`] or
/// [`retire_into_with`] from arming the retirement.
#[derive(Debug)]
pub struct RetireError<S, T = ()> {
    pub stream: S,
    pub state: T,
    pub error: GpuError,
}

/// Arms a retirement on `stream` whose continuation is a frame emitted by `completions`. The frame is built by the
/// lattice's own poll, never by the driver's callback.
///
/// The stream **travels with its continuation**: this takes the submitter by value, and `then(stream, ticket)` gets it
/// back together with the ticket. A submitter has one owner, so the frame that continues the work is the next owner,
/// and nothing is shared or locked between the frames of a chain. `then` settles the ticket with
/// [`Stream::confirm_retired`], submits more, and arms the next retirement with the stream it was given. If the arming
/// fails the stream comes back in the error.
///
/// `completions` polls the stream's retirement queue ([`Stream::poll_retired`]); there is no per-retirement cell. The
/// stream must have no other listener-free retirement outstanding ([`Stream::outstanding_tickets`]): its ticket would
/// reach this continuation in place of the one armed here, possibly before the work this retirement follows. In that
/// case nothing is armed and the stream comes back in the error.
///
/// If the device is poisoned before the callback runs, the continuation is emitted at once with
/// [`POISONED_STATUS`], so nothing is stranded. **Dropping a submitter is a shutdown call** (it waits for the device):
/// a continuation that is done with a stream hands it back to the source that owns it, which drops it in its lifecycle
/// stop path, instead of dropping it in a frame; a [`Completions`] dropped or completed with a continuation still
/// pending drops the stream there.
pub fn retire_into<S: Stream>(
    stream: S,
    completions: &Completions,
    then: impl FnOnce(S, RetireTicket) -> Box<dyn Frame> + Send + 'static,
) -> Result<(), RetireError<S>> {
    retire_into_with(stream, (), completions, move |stream, (), ticket| {
        then(stream, ticket)
    })
}

/// [`retire_into`] with state that travels with the stream: `state` is handed to `then` with the stream and the
/// ticket, and comes back with the stream if arming fails, so nothing the chain owns is lost or dropped on the
/// lattice by an error.
pub fn retire_into_with<S: Stream, T: Send + 'static>(
    mut stream: S,
    state: T,
    completions: &Completions,
    then: impl FnOnce(S, T, RetireTicket) -> Box<dyn Frame> + Send + 'static,
) -> Result<(), RetireError<S, T>> {
    if stream.outstanding_tickets() > 0 {
        return Err(RetireError {
            stream,
            state,
            error: GpuError::Invalid(
                "a listener-free retirement is already outstanding on this stream".into(),
            ),
        });
    }
    if let Err(error) = stream.arm_retirement() {
        return Err(RetireError {
            stream,
            state,
            error,
        });
    }
    completions.register(Box::new(StreamPending {
        stream: Some(stream),
        state: Some(state),
        then: Some(Box::new(then)),
    }));
    Ok(())
}

type Then<S, T> = Box<dyn FnOnce(S, T, RetireTicket) -> Box<dyn Frame> + Send>;

/// A pending retirement: it owns the stream, and each poll drains the stream's retirement queue.
struct StreamPending<S: Stream, T> {
    stream: Option<S>,
    state: Option<T>,
    then: Option<Then<S, T>>,
}

impl<S: Stream, T: Send> Pending for StreamPending<S, T> {
    fn poll(&mut self) -> Option<Box<dyn Frame>> {
        let ticket = self.stream.as_mut()?.poll_retired()?;
        let (stream, state, then) = (self.stream.take()?, self.state.take()?, self.then.take()?);
        Some(then(stream, state, ticket))
    }
}

/// The callback's end of a stream's retirement queue. It lives as long as callbacks can fire (the stream's shared
/// queue side owns it), and the backend passes a pointer to it as the callback's user data: no per-retirement box.
pub(crate) struct RetireSink {
    producer: Producer<i32, Multi>,
}

impl std::fmt::Debug for RetireSink {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RetireSink").finish_non_exhaustive()
    }
}

impl RetireSink {
    /// Hands one retirement's status to the owner. Lock-free; makes no driver call and frees nothing. It allocates
    /// nothing while fewer than [`UNPOLLED_RETIREMENTS`] statuses wait unpolled: the queue starts with [`POOLED_CHUNKS`]
    /// spare chunks, one is linked in per [`CHUNK`]` - 1` statuses, and the owner's polling returns them. (A chunk pool
    /// that is contended at that instant is skipped for a fresh allocation: `euhedral-data-structures`' `ChunkPool`
    /// guards its stack with a try-lock, which the separate lock audit tracks.)
    pub(crate) fn push(&self, status: i32) {
        // The queue is unbounded: an offer is never refused.
        let _ = self.producer.offer(status);
    }
}

/// What the owner does when a retirement arrives.
enum Armed {
    /// Tell this listener, inside the owner's poll.
    Listener(Box<dyn RetireListener>),
    /// Hand the ticket to the caller of [`Retirements::poll`].
    Pull,
}

/// The owner's end of a stream's retirement queue, and the retirements it has armed, in order.
pub(crate) struct Retirements {
    consumer: Consumer<i32, Single>,
    armed: PlainQueue<(u64, Armed)>,
    next: u64,
    id: QueueId,
    lane: LaneId,
    /// The device was poisoned and the outstanding retirements were reported as such: anything that arrives later no
    /// longer says which retirement it belongs to, and is dropped.
    abandoned: bool,
    /// Retirements armed without a listener whose tickets `poll` has not returned.
    tickets: usize,
    /// A status arrived with no armed retirement to belong to: the order the design relies on broke.
    desync: bool,
    /// Statuses taken off the queue, whatever became of them (dispatched, dropped as late, or unmatched). Compared
    /// with the retirements armed, it says whether a callback is still to come.
    arrived: u64,
}

impl std::fmt::Debug for Retirements {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Retirements")
            .field("outstanding", &self.armed.size_long())
            .field("next", &self.next)
            .finish()
    }
}

/// Slots of a queue chunk (one of them links to the next chunk, so a chunk holds `CHUNK - 1` statuses), and how many
/// retired chunks the queue keeps for reuse.
const CHUNK: usize = 64;
const POOLED_CHUNKS: usize = 4;
/// A stream's driver callback allocates nothing while fewer retirement statuses than this wait unpolled: the room left in the active chunk is
/// at least one slot, and each pooled chunk adds `CHUNK - 1`.
pub const UNPOLLED_RETIREMENTS: usize = POOLED_CHUNKS * (CHUNK - 1);

/// The queue, with its chunk pool seeded: filling and draining `POOLED_CHUNKS + 1` chunks retires `POOLED_CHUNKS` of
/// them into the pool, so the callback finds spare chunks from the first retirement on.
fn seeded_queue() -> Queue<i32, Multi, Single> {
    let queue = Queue::<i32, Multi, Single>::with_pool(CHUNK, POOLED_CHUNKS);
    let producer = queue.producer();
    for _ in 0..(POOLED_CHUNKS + 1) * (CHUNK - 1) {
        // An unbounded queue never refuses.
        let _ = producer.offer(0);
    }
    queue.consumer().clear();
    queue
}

impl Retirements {
    /// A retirement channel for the stream `id` on `lane`: the owner's end, and the callback's.
    pub(crate) fn new(id: QueueId, lane: LaneId) -> (Self, RetireSink) {
        let queue = seeded_queue();
        let sink = RetireSink {
            producer: queue.producer(),
        };
        let retirements = Self {
            consumer: queue.consumer(),
            armed: PlainQueue::new(16),
            next: 0,
            id,
            lane,
            abandoned: false,
            tickets: 0,
            desync: false,
            arrived: 0,
        };
        (retirements, sink)
    }

    /// Retirements armed and not yet reported, listeners and tickets together.
    pub(crate) fn outstanding(&self) -> usize {
        usize::try_from(self.armed.size_long()).unwrap_or(usize::MAX)
    }

    /// Retirements armed without a listener whose tickets have not been returned.
    pub(crate) fn outstanding_tickets(&self) -> usize {
        self.tickets
    }

    #[cfg_attr(
        not(feature = "cuda"),
        allow(dead_code, reason = "used by the CUDA stream's drop")
    )]
    /// Whether a driver callback armed here may still be to come: drains whatever has arrived (dropping it, counted) and
    /// compares the retirements armed with the statuses seen. A stream that cannot wait for its callbacks keeps their
    /// destination alive while this is true.
    pub(crate) fn callbacks_pending(&mut self) -> bool {
        self.arrived += self.consumer.drain_all(drop);
        self.next > self.arrived
    }

    /// Whether a status arrived with no retirement to belong to (once; the flag is cleared).
    pub(crate) fn take_desync(&mut self) -> bool {
        std::mem::take(&mut self.desync)
    }

    /// The number the next retirement armed gets.
    pub(crate) fn next_seq(&self) -> u64 {
        self.next
    }

    /// Records that a retirement was armed (its callback is enqueued), with the listener to tell, if any. Called after
    /// the backend accepted the callback: whatever it pushes is read only by a later poll.
    pub(crate) fn register(&mut self, listener: Option<Box<dyn RetireListener>>) {
        self.tickets += usize::from(listener.is_none());
        let armed = listener.map_or(Armed::Pull, Armed::Listener);
        // An unbounded queue never refuses.
        let _ = self.armed.offer((self.next, armed));
        self.next += 1;
    }

    /// Drains the retirements that arrived: each listener is told, on this thread, in order; the first retirement armed
    /// without a listener returns its ticket (the rest stay queued for later polls).
    ///
    /// With `poisoned` set, the retirements still outstanding after what arrived are reported with
    /// [`POISONED_STATUS`] (a poisoned device may never report them), and what arrives afterwards is dropped.
    pub(crate) fn poll(&mut self, poisoned: bool) -> Option<RetireTicket> {
        if !self.abandoned {
            while let Some(status) = self.consumer.poll() {
                self.arrived += 1;
                // Callbacks run in the order they were enqueued, so this status belongs to the oldest armed one.
                let Some((seq, armed)) = self.armed.poll() else {
                    // A status nobody armed: never drop it quietly. The stream poisons its device with this.
                    self.desync = true;
                    continue;
                };
                if let Some(ticket) = self.dispatch(seq, armed, status) {
                    return Some(ticket);
                }
            }
        }
        if poisoned {
            self.abandoned = true;
            self.arrived += self.consumer.drain_all(drop);
            while let Some((seq, armed)) = self.armed.poll() {
                if let Some(ticket) = self.dispatch(seq, armed, POISONED_STATUS) {
                    return Some(ticket);
                }
            }
        }
        None
    }

    fn dispatch(&mut self, seq: u64, armed: Armed, status: i32) -> Option<RetireTicket> {
        let ticket = RetireTicket::new(self.id, self.lane, seq, status);
        match armed {
            Armed::Listener(listener) => {
                listener.retired(ticket);
                None
            }
            Armed::Pull => {
                self.tickets -= 1;
                Some(ticket)
            }
        }
    }
}

#[cfg(any(feature = "fake", feature = "cuda"))]
mod shared {
    use super::*;

    /// A lane's retirement and kept values, shared by every thread that submits to it.
    ///
    /// The callback side is the lock-free sink every retirement channel has; what changes is the owner side, which several
    /// threads reach:
    /// - **Arming and polling** go through a try-claim ([`ClaimCell`]) over the lane's [`Retirements`]: one thread at a time
    ///   arms or drains, and a thread that finds it held does not wait ([`arm`](Self::arm) answers `Busy`,
    ///   [`poll`](Self::poll) finds nothing yet and its caller polls again). The claim makes the order of the callbacks the
    ///   order of the arms, which the channel relies on (the n-th status that arrives belongs to the n-th retirement armed).
    /// - **Kept values** are pushed by any thread to a lock-free queue, tagged with the number of retirements armed so far
    ///   when the value was kept. Retirement `s` releases the values tagged `s` or less: the retirement count moves *before*
    ///   the callback is enqueued, so a value kept while its work was enqueued before the callback is tagged at most `s`, and
    ///   a value tagged `s` or less never outlives work that the callback does not cover. (A value is released by a
    ///   retirement at or after its tag, sometimes later than the first that covers it: never earlier.)
    /// - **Release** happens in [`release_through`](Self::release_through), called from the continuation that confirms a
    ///   retirement, on an ordinary thread in a frame. Values are dropped outside the claim, one at a time.
    pub(crate) struct SharedRetire {
        sink: Arc<RetireSink>,
        state: ClaimCell<LaneRetire>,
        /// Retirements armed so far, as the keepers see them.
        armed: AtomicU64,
        kept_in: Producer<(u64, Box<dyn Any + Send>), Multi>,
    }

    /// What the claim holder reaches.
    struct LaneRetire {
        retire: Retirements,
        kept_out: Consumer<(u64, Box<dyn Any + Send>), Single>,
        kept: Kept,
        /// When the lane last looked at the device for a fault with retirements outstanding.
        last_probe: Option<Instant>,
    }

    impl std::fmt::Debug for SharedRetire {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("SharedRetire")
                .field("armed", &self.armed.load(Ordering::Relaxed))
                .finish_non_exhaustive()
        }
    }

    /// How often a lane with retirements outstanding looks at the device while no callback arrives.
    const PROBE_EVERY: Duration = Duration::from_millis(1);

    impl SharedRetire {
        /// The retirement channel of the lane `lane` of the stream `id`.
        pub(crate) fn new(id: QueueId, lane: LaneId) -> Self {
            let (retire, sink) = Retirements::new(id, lane);
            let kept =
                Queue::<(u64, Box<dyn Any + Send>), Multi, Single>::with_pool(CHUNK, POOLED_CHUNKS);
            Self {
                sink: Arc::new(sink),
                state: ClaimCell::new(LaneRetire {
                    retire,
                    kept_out: kept.consumer(),
                    kept: Kept::default(),
                    last_probe: None,
                }),
                armed: AtomicU64::new(0),
                kept_in: kept.producer(),
            }
        }

        /// The callback's end: the backend passes a pointer to it as the callback's user data.
        pub(crate) fn sink(&self) -> &Arc<RetireSink> {
            &self.sink
        }

        /// Arms a retirement after the work enqueued so far. `enqueue(seq)` enqueues the backend's callback for retirement
        /// number `seq`. `Err(Busy)` (retry; this never waits) when another thread is arming or polling.
        pub(crate) fn arm(
            &self,
            enqueue: impl FnOnce(u64) -> Result<(), GpuError>,
        ) -> Result<(), GpuError> {
            let Some(mut state) = self.state.try_claim() else {
                return Err(GpuError::Busy);
            };
            // The count moves before the callback is enqueued (see the type's documentation).
            self.armed.fetch_add(1, Ordering::SeqCst);
            if let Err(error) = enqueue(state.retire.next_seq()) {
                // Nothing was armed. A value kept meanwhile is tagged one too high, which only delays its release.
                self.armed.fetch_sub(1, Ordering::SeqCst);
                return Err(error);
            }
            state.retire.register(None);
            Ok(())
        }

        /// Keeps `value` until a retirement armed after this call is confirmed.
        pub(crate) fn keep(&self, value: Box<dyn Any + Send>) {
            let tag = self.armed.load(Ordering::SeqCst);
            // An unbounded queue never refuses.
            let _ = self.kept_in.offer((tag, value));
        }

        /// The ticket of the next retirement that arrived, if the claim is free and one did. `look` is one look at the
        /// device, which a lane makes (at most every millisecond) when retirements are outstanding and none arrived: a
        /// retirement behind a lane that faulted never gets its callback. Returns whether a status arrived that nobody had
        /// armed (the order the channel relies on broke).
        pub(crate) fn poll(
            &self,
            poisoned: impl Fn() -> bool,
            look: impl FnOnce(),
        ) -> (Option<RetireTicket>, bool) {
            let Some(mut state) = self.state.try_claim() else {
                return (None, false);
            };
            let mut ticket = state.retire.poll(poisoned());
            if ticket.is_none() && state.retire.outstanding() > 0 {
                let now = Instant::now();
                if state
                    .last_probe
                    .is_none_or(|last| now.duration_since(last) >= PROBE_EVERY)
                {
                    state.last_probe = Some(now);
                    look();
                    ticket = state.retire.poll(poisoned());
                }
            }
            let desync = state.retire.take_desync();
            (ticket, desync)
        }

        /// Releases the values kept for retirement `seq` and the ones before it, in the calling thread. Each value is
        /// dropped outside the claim.
        pub(crate) fn release_through(&self, seq: u64) {
            loop {
                let value = loop {
                    if let Some(mut state) = self.state.try_claim() {
                        let LaneRetire { kept_out, kept, .. } = &mut *state;
                        kept_out.drain_all(|(tag, value)| kept.push(tag, value));
                        break kept.pop_through(seq);
                    }
                };
                match value {
                    Some(value) => drop(value),
                    None => return,
                }
            }
        }

        /// Whether a callback armed here may still be to come (see [`Retirements::callbacks_pending`]).
        #[cfg(feature = "cuda")]
        pub(crate) fn callbacks_pending(&mut self) -> bool {
            self.state.get_mut().retire.callbacks_pending()
        }
    }
}
#[cfg(any(feature = "fake", feature = "cuda"))]
pub(crate) use shared::SharedRetire;

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;

    fn channel() -> (Retirements, RetireSink, QueueId) {
        let id = QueueId::next();
        let (retirements, sink) = Retirements::new(id, LaneId(3));
        (retirements, sink, id)
    }

    #[test]
    fn retirements_arrive_in_order_with_their_numbers() {
        let (mut r, sink, id) = channel();
        for _ in 0..3 {
            r.register(None);
        }
        assert!(r.poll(false).is_none(), "nothing arrived yet");
        sink.push(0);
        sink.push(0);
        sink.push(7);
        let tickets: Vec<RetireTicket> = std::iter::from_fn(|| r.poll(false)).collect();
        assert_eq!(
            tickets
                .iter()
                .map(|t| (t.seq(), t.status()))
                .collect::<Vec<_>>(),
            [(0, 0), (1, 0), (2, 7)]
        );
        assert!(
            tickets
                .iter()
                .all(|t| t.stream() == id && t.lane() == LaneId(3))
        );
        assert!(tickets[0].ok() && !tickets[2].ok());
        assert!(r.poll(false).is_none(), "each exactly once");
    }

    #[test]
    fn listeners_are_told_on_the_polling_thread_in_order() {
        struct Told(Arc<Mutex<Vec<(u64, i32, std::thread::ThreadId)>>>);
        impl RetireListener for Told {
            fn retired(self: Box<Self>, ticket: RetireTicket) {
                self.0.lock().unwrap().push((
                    ticket.seq(),
                    ticket.status(),
                    std::thread::current().id(),
                ));
            }
        }
        let (mut r, sink, _) = channel();
        let told = Arc::new(Mutex::new(Vec::new()));
        for _ in 0..3 {
            r.register(Some(Box::new(Told(told.clone()))));
        }
        // The callback side is another thread; it only pushes.
        #[allow(
            clippy::disallowed_methods,
            reason = "a thread stands in for the driver's callback thread"
        )]
        std::thread::scope(|scope| {
            scope.spawn(|| {
                sink.push(0);
                sink.push(700);
                sink.push(0);
            });
        });
        assert!(
            told.lock().unwrap().is_empty(),
            "nothing is told until the owner polls"
        );
        assert!(
            r.poll(false).is_none(),
            "listener-style retirements return no ticket"
        );
        let told = told.lock().unwrap();
        assert_eq!(
            told.iter().map(|(s, st, _)| (*s, *st)).collect::<Vec<_>>(),
            [(0, 0), (1, 700), (2, 0)]
        );
        assert!(
            told.iter()
                .all(|(_, _, t)| *t == std::thread::current().id())
        );
    }

    #[test]
    fn a_poisoned_device_reports_the_rest_and_drops_late_arrivals() {
        let (mut r, sink, _) = channel();
        for _ in 0..3 {
            r.register(None);
        }
        sink.push(700);
        // What arrived keeps its status; the two never reported are poisoned.
        let first = r.poll(true).unwrap();
        assert_eq!((first.seq(), first.status()), (0, 700));
        let second = r.poll(true).unwrap();
        assert_eq!((second.seq(), second.status()), (1, POISONED_STATUS));
        let third = r.poll(true).unwrap();
        assert_eq!((third.seq(), third.status()), (2, POISONED_STATUS));
        // Late callbacks no longer say whose they are: dropped, never matched to anything.
        sink.push(0);
        assert!(r.poll(true).is_none());
        assert!(r.poll(false).is_none());
    }

    #[test]
    fn arrivals_are_counted_even_when_they_are_dropped() {
        let (mut r, sink, _) = channel();
        for _ in 0..3 {
            r.register(None);
        }
        assert!(r.callbacks_pending(), "none arrived yet");
        sink.push(0);
        // The retirements were reported as poisoned before their callbacks arrived.
        let reported: Vec<_> = std::iter::from_fn(|| r.poll(true)).collect();
        assert_eq!(reported.len(), 3);
        assert_eq!(r.outstanding(), 0, "nothing is armed any more");
        assert!(
            r.callbacks_pending(),
            "yet two callbacks are still to come, and the sink must outlive them"
        );
        sink.push(0);
        sink.push(0);
        assert!(!r.callbacks_pending(), "all three arrived");
    }

    #[test]
    fn the_chunk_pool_is_seeded_at_creation() {
        assert_eq!(seeded_queue().pooled_chunks(), POOLED_CHUNKS);
    }

    #[test]
    fn a_status_nobody_armed_is_flagged() {
        let (mut r, sink, _) = channel();
        sink.push(0);
        assert!(r.poll(false).is_none());
        assert!(r.take_desync(), "a desync must never be silent");
        assert!(!r.take_desync(), "reported once");
    }

    #[test]
    fn listener_free_retirements_are_counted_until_their_tickets_return() {
        let (mut r, sink, _) = channel();
        r.register(None);
        r.register(None);
        assert_eq!((r.outstanding(), r.outstanding_tickets()), (2, 2));
        sink.push(0);
        assert!(r.poll(false).is_some());
        assert_eq!((r.outstanding(), r.outstanding_tickets()), (1, 1));
    }

    /// The lane's shared channel: arms from many threads, the order of the callbacks the order of the arms, kept values
    /// released once, by the retirement that covers them.
    #[cfg(any(feature = "fake", feature = "cuda"))]
    mod shared {
        use std::sync::atomic::{AtomicUsize, Ordering};

        use super::*;

        /// A value that counts its own drop.
        struct Counted(Arc<AtomicUsize>);

        impl Drop for Counted {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }

        fn lane() -> SharedRetire {
            SharedRetire::new(QueueId::next(), LaneId(1))
        }

        #[test]
        fn a_value_kept_is_released_once_by_the_retirement_that_covers_it() {
            let shared = lane();
            let dropped = Arc::new(AtomicUsize::new(0));
            // Kept before the first retirement is armed: tagged 0, covered by retirement 0.
            shared.keep(Box::new(Counted(dropped.clone())));
            shared
                .arm(|seq| {
                    assert_eq!(seq, 0);
                    shared.sink().push(0);
                    Ok(())
                })
                .unwrap();
            // Kept after it: tagged 1, covered by retirement 1.
            shared.keep(Box::new(Counted(dropped.clone())));
            shared
                .arm(|_| {
                    shared.sink().push(0);
                    Ok(())
                })
                .unwrap();
            let (first, _) = shared.poll(|| false, || {});
            let first = first.expect("the first retirement arrived");
            assert_eq!(first.seq(), 0);
            assert_eq!(
                dropped.load(Ordering::SeqCst),
                0,
                "nothing is released by a poll"
            );
            shared.release_through(first.seq());
            assert_eq!(dropped.load(Ordering::SeqCst), 1);
            shared.release_through(first.seq());
            assert_eq!(dropped.load(Ordering::SeqCst), 1, "released once");
            let (second, _) = shared.poll(|| false, || {});
            shared.release_through(second.expect("the second arrived").seq());
            assert_eq!(dropped.load(Ordering::SeqCst), 2);
        }

        /// Eight threads arm retirements while another polls: the callbacks (pushed inside the claim, as the driver's are
        /// enqueued in call order) come back as tickets numbered in the order of the arms, each exactly once.
        #[test]
        #[allow(
            clippy::disallowed_methods,
            reason = "threads stand in for the workers that arm a lane's retirements"
        )]
        fn arms_from_many_threads_come_back_in_order_each_once() {
            let rounds = if cfg!(miri) { 4 } else { 500 };
            let shared = lane();
            let armed = AtomicUsize::new(0);
            let mut tickets = Vec::new();
            std::thread::scope(|scope| {
                let workers: Vec<_> = (0..8)
                    .map(|_| {
                        scope.spawn(|| {
                            for _ in 0..rounds {
                                while shared
                                    .arm(|_| {
                                        shared.sink().push(0);
                                        Ok(())
                                    })
                                    .is_err()
                                {}
                                armed.fetch_add(1, Ordering::SeqCst);
                            }
                        })
                    })
                    .collect();
                while tickets.len() < 8 * rounds {
                    if let (Some(ticket), _) = shared.poll(|| false, || {}) {
                        tickets.push(ticket.seq());
                    }
                }
                for worker in workers {
                    worker.join().unwrap();
                }
            });
            assert_eq!(armed.load(Ordering::SeqCst), 8 * rounds);
            assert_eq!(
                tickets,
                (0..(8 * rounds) as u64).collect::<Vec<_>>(),
                "in the order armed, none twice"
            );
        }

        #[test]
        fn a_failed_enqueue_arms_nothing() {
            let shared = lane();
            let refused = shared.arm(|_| Err(GpuError::Invalid("refused".into())));
            assert!(matches!(refused, Err(GpuError::Invalid(_))));
            assert!(shared.poll(|| false, || {}).0.is_none());
            shared
                .arm(|seq| {
                    assert_eq!(seq, 0, "the failed arm took no number");
                    shared.sink().push(0);
                    Ok(())
                })
                .unwrap();
            assert_eq!(shared.poll(|| false, || {}).0.unwrap().seq(), 0);
        }
    }
}
