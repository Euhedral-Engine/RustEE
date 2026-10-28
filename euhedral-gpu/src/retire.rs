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

use euhedral_core::frames::{Frame, FrameError, FrameHeader};
use euhedral_data_structures::queues::{Consumer, Multi, PlainQueue, Producer, Queue, Single};
use euhedral_graph::{Completions, LaneId, Pending, QueueId};

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
    #[cfg_attr(
        not(feature = "cuda"),
        allow(dead_code, reason = "used by the CUDA stream")
    )]
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
    /// `verify` is asked before a status of success is believed, and says whether the device failed meanwhile (then
    /// the status becomes [`POISONED_STATUS`]). A backend whose callback sees the work's real outcome passes
    /// `|| false`; one whose callback always reports success (a host function) passes a check of the stream.
    ///
    /// With `poisoned` set, the retirements still outstanding after what arrived are reported with
    /// [`POISONED_STATUS`] (a poisoned device may never report them), and what arrives afterwards is dropped.
    pub(crate) fn poll(
        &mut self,
        poisoned: bool,
        verify: &mut dyn FnMut() -> bool,
    ) -> Option<RetireTicket> {
        if !self.abandoned {
            while let Some(status) = self.consumer.poll() {
                self.arrived += 1;
                // A status of success from a backend that cannot see the work's outcome is checked against the device
                // before it is believed (see `verify`).
                let status = if status == 0 && verify() {
                    POISONED_STATUS
                } else {
                    status
                };
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
        assert!(
            r.poll(false, &mut || false).is_none(),
            "nothing arrived yet"
        );
        sink.push(0);
        sink.push(0);
        sink.push(7);
        let tickets: Vec<RetireTicket> =
            std::iter::from_fn(|| r.poll(false, &mut || false)).collect();
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
        assert!(r.poll(false, &mut || false).is_none(), "each exactly once");
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
            r.poll(false, &mut || false).is_none(),
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
        let first = r.poll(true, &mut || false).unwrap();
        assert_eq!((first.seq(), first.status()), (0, 700));
        let second = r.poll(true, &mut || false).unwrap();
        assert_eq!((second.seq(), second.status()), (1, POISONED_STATUS));
        let third = r.poll(true, &mut || false).unwrap();
        assert_eq!((third.seq(), third.status()), (2, POISONED_STATUS));
        // Late callbacks no longer say whose they are: dropped, never matched to anything.
        sink.push(0);
        assert!(r.poll(true, &mut || false).is_none());
        assert!(r.poll(false, &mut || false).is_none());
    }

    #[test]
    fn a_success_the_device_contradicts_is_reported_poisoned() {
        let (mut r, sink, _) = channel();
        for _ in 0..3 {
            r.register(None);
        }
        sink.push(0);
        sink.push(0);
        sink.push(5);
        let mut asked = 0;
        let mut verify = || {
            asked += 1;
            asked >= 2
        };
        let statuses: Vec<i32> = std::iter::from_fn(|| r.poll(false, &mut verify))
            .map(|t| t.status())
            .collect();
        assert_eq!(
            statuses,
            [0, POISONED_STATUS, 5],
            "asked before each success, never about a failure"
        );
        assert_eq!(asked, 2);
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
        let reported: Vec<_> = std::iter::from_fn(|| r.poll(true, &mut || false)).collect();
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
        assert!(r.poll(false, &mut || false).is_none());
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
        assert!(r.poll(false, &mut || false).is_some());
        assert_eq!((r.outstanding(), r.outstanding_tickets()), (1, 1));
    }
}
