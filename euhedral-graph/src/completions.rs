//! Completions found by polls (Euhedral-Inference `GpuCompletions`).

use std::collections::VecDeque;
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

use euhedral_core::frames::Frame;
use euhedral_core::generics::{DownstreamSlot, LatticeReceiver, LatticeSource};
use euhedral_data_structures::queues::{Consumer, MpscQueue, Multi, Producer, Single};
use euhedral_data_structures::sync::{ClaimCell, ClaimGuard, HotScope};

use euhedral_core::metrics::Histogram;

use crate::lanes::{LanePool, Marker};

/// Whether an operation has completed. One cheap query; never blocks.
pub trait Ready: Send {
    fn is_ready(&mut self) -> bool;
}

/// An operation in flight and what continues it: polled until it yields the continuation frame. Use
/// it when the completion carries a result (an encoded packet fetched from a driver without
/// waiting, a model's output); use [`Completions::after`] when it doesn't.
pub trait Pending: Send {
    /// `Some(frame)` once the operation completed (the frame carries its result or its error);
    /// `None` while it is in flight. Never blocks.
    fn poll(&mut self) -> Option<Box<dyn Frame>>;
}

/// A [`LatticeSource`] of continuations behind completions.
///
/// [`after`](Self::after), [`after_marker`](Self::after_marker), [`at`](Self::at) and
/// [`register`](Self::register) may be called from any thread. The lattice's polls
/// (`pull`/`request`, one thread at a time) query each pending operation once and emit the
/// continuations of those that completed, in the order they were registered. Nothing waits.
///
/// A `pull` stops before the first completed continuation its `stop` matches, leaving it (and
/// everything registered after it) for a later poll; the lattice's direct pulls stop before ordered
/// frames, which then go out through `request`'s pushes and the lattice's routing, so continuations
/// on one ordered lane keep their registration order. A completed operation is never queried again.
///
/// A continuation is emitted even if its frame has died (its kill switch is set): the executor then
/// skips it and runs its `do_finally`, which ends what it holds. So at shutdown
/// [`pending`](Self::pending) drains to zero on its own.
///
/// Complete it ([`LatticeSource::complete`]) only once [`pending`](Self::pending) is zero. A
/// registration after that is a bug that fails visibly: its frame finishes at once through
/// `do_finally` without running (as a frame refused by a completed RustEE ingest queue is
/// cancelled), so whatever it holds still ends, and debug builds then panic. A [`Pending`]
/// registered after completion has no frame to finish: it is dropped.
///
/// A registration that races `complete` (the flag was not yet visible to it) lands in the queue
/// and ends exactly one way: ahead of `complete`'s marker in the queue it is a registration like
/// any other and is emitted; behind it, the next poll (`pull` or `request`) cancels it through
/// `do_finally` without running it, and so does the last drop of the source if no poll comes. That
/// cancellation does not panic: only the polling thread sees it, and it cannot tell a misuse from a
/// poll that was late.
pub struct Completions {
    /// The lanes polled for markers: fixed for the source's life and dropped with it (a marker is
    /// treated as reached once the source has completed, whatever its lane says).
    lanes: Arc<LanePool>,
    /// Registered since the last poll; any thread appends, lock-free. [`LatticeSource::complete`]
    /// appends one [`Slot::Closed`] marker: what is behind it was registered too late.
    incoming: Producer<Slot, Multi>,
    /// The polling side: the queue's consumer and what it drained and has not emitted. A claim,
    /// as a queue sink's `draining` flag is: `pull` and `request` (one thread at a time) take it
    /// without contention, a late registration's sweep takes it only if free, and whoever finds it
    /// held leaves its work to the holder, which looks for it after letting go.
    poll: ClaimCell<Poll>,
    /// A sweep waiting for the holder of `poll`: set before the claim is tried, read by the holder
    /// after it releases.
    sweep_pending: AtomicBool,
    /// A request waiting for the holder of `poll`; its demand is already banked in `demand`.
    request_pending: AtomicBool,
    pending: AtomicUsize,
    demand: AtomicU64,
    downstream: DownstreamSlot,
    complete: AtomicBool,
    /// How late each emission was, in µs, since the start (lock-free: polls record, readers copy).
    lateness: Histogram,
}

/// What travels through the registration queue.
enum Slot {
    Waiting(Waiting),
    /// Appended once, by `complete`, after it set the flag: everything behind it was registered
    /// after the source completed.
    Closed,
}

/// What only the polling thread touches.
struct Poll {
    consumer: Consumer<Slot, Single>,
    /// Drained from the queue, oldest first; entries are marked done in place.
    waiting: VecDeque<Waiting>,
    /// The `Closed` marker was drained: whatever is drained from now on is late.
    closed: bool,
}

/// A registered operation and when it was last seen in flight (an upper bound on when it completed
/// is the poll that found it done; its lateness is at most the time since the previous look).
struct Waiting {
    what: Waits,
    last_seen: Instant,
}

enum Waits {
    Ready(Box<dyn Ready>, Box<dyn Frame>),
    Marker(Marker, Box<dyn Frame>),
    At(Instant, Box<dyn Frame>),
    Pending(Box<dyn Pending>),
    /// Completed, its frame held back by a pull's `stop` (with when it was last seen in flight, or
    /// its deadline): handed on by a later poll, before anything registered after it.
    Done(Box<dyn Frame>, Instant),
    /// Only while an entry changes into `Done`.
    Moving,
}

impl Drop for Completions {
    /// The last owner lets go: no thread can register any more, and none will poll. What is still
    /// in the queue behind the marker is cancelled, so no registration is left without an end.
    fn drop(&mut self) {
        let late = Self::absorb(self.poll.get_mut());
        for what in late {
            Self::cancel(what);
        }
    }
}

impl fmt::Debug for Completions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Completions")
            .field("pending", &self.pending())
            .finish_non_exhaustive()
    }
}

/// Slots per chunk of the registration queue (rounded to a power of two by the queue).
const INCOMING_CHUNK: usize = 256;

impl Completions {
    /// Polls of `marker`s use `lanes`.
    pub fn new(lanes: Arc<LanePool>) -> Arc<Self> {
        let queue: MpscQueue<Slot> = MpscQueue::with_pool(INCOMING_CHUNK, 2);
        Arc::new(Self {
            lanes,
            incoming: queue.producer(),
            poll: ClaimCell::new(Poll {
                consumer: queue.consumer(),
                waiting: VecDeque::new(),
                closed: false,
            }),
            sweep_pending: AtomicBool::new(false),
            request_pending: AtomicBool::new(false),
            pending: AtomicUsize::new(0),
            demand: AtomicU64::new(0),
            downstream: DownstreamSlot::new(),
            complete: AtomicBool::new(false),
            lateness: Histogram::new(),
        })
    }

    /// Registers `what`. Exactly one end for its frame, whatever races `complete` (the argument is
    /// on [`LatticeSource::complete`]'s implementation below).
    fn add(&self, what: Waits) {
        let _hot = HotScope::enter();
        // The fast path of the sequential case: the flag is set and visible (`complete` returned
        // before this call, by any synchronisation), so nothing is queued and the frame ends here.
        // Only a speed-up and the visible failure: a registration that misses the flag takes the
        // queue and is told apart by its position behind `Slot::Closed`.
        if self.complete.load(Ordering::SeqCst) {
            Self::cancel(what);
            debug_assert!(
                false,
                "a completion registered after Completions::complete (its frame was cancelled)"
            );
            return;
        }
        // Counted before it can be seen, so `pending` never reads zero while a registration is
        // between here and the queue.
        self.pending.fetch_add(1, Ordering::AcqRel);
        let slot = Slot::Waiting(Waiting {
            what,
            last_seen: Instant::now(),
        });
        if let Err(Slot::Waiting(Waiting { what, .. })) = self.incoming.offer(slot) {
            // Unreachable for the unbounded queue; ends the frame rather than losing it.
            self.pending.fetch_sub(1, Ordering::AcqRel);
            Self::cancel(what);
            return;
        }
        // Racing `complete`: the flag may be set by now, and this one may be behind the marker.
        // Do not wait for the next poll to find it (best effort; the poll, or the last drop, is
        // what guarantees it).
        if self.complete.load(Ordering::SeqCst) {
            self.sweep();
        }
    }

    /// Ends what a refused registration holds: its frame is finished without running; a
    /// [`Pending`] has no frame, so it is dropped.
    fn cancel(what: Waits) {
        match what {
            Waits::Ready(_, frame)
            | Waits::Marker(_, frame)
            | Waits::At(_, frame)
            | Waits::Done(frame, _) => frame.do_finally(),
            Waits::Pending(pending) => drop(pending),
            Waits::Moving => {}
        }
    }

    /// Moves what landed in the queue into `poll.waiting`, and takes out what landed behind the
    /// `Closed` marker. Returns the late ones to cancel (the caller does it after letting go of the
    /// lock: `do_finally` is the application's).
    fn absorb(poll: &mut Poll) -> Vec<Waits> {
        let Poll {
            consumer,
            waiting,
            closed,
        } = poll;
        let mut late = Vec::new();
        consumer.drain_all(|slot| match slot {
            Slot::Closed => *closed = true,
            Slot::Waiting(entry) if *closed => late.push(entry.what),
            Slot::Waiting(entry) => waiting.push_back(entry),
        });
        late
    }

    /// Cancels the late ones `absorb` found.
    fn cancel_late(&self, late: Vec<Waits>) {
        if late.is_empty() {
            return;
        }
        let count = late.len();
        for what in late {
            Self::cancel(what);
        }
        self.pending.fetch_sub(count, Ordering::AcqRel);
    }

    /// Cancels what landed too late now, unless a poll holds the claim: that holder absorbs the
    /// queue itself, and looks for this sweep after it lets go. For a late registration, and for a
    /// `request` of a completed source (which emits nothing, but is a poll).
    fn sweep(&self) {
        // The wakeup is published before the claim is tried: the releasing holder either sees it,
        // or this call takes the claim itself.
        self.sweep_pending.store(true, Ordering::SeqCst);
        self.run_sweeps();
    }

    /// Absorbs the queue and cancels the late ones while a sweep is waiting and the claim is free.
    fn run_sweeps(&self) {
        while self.sweep_pending.load(Ordering::SeqCst) {
            let Some(mut poll) = self.poll.try_claim() else {
                return;
            };
            self.sweep_pending.store(false, Ordering::SeqCst);
            let late = Self::absorb(&mut poll);
            self.cancel_late(late);
            poll.release_seq_cst();
        }
    }

    /// What a holder of the claim does after it lets go: the sweeps and requests that arrived while
    /// it held the claim (and so were refused it), as a queue sink's holder serves its requests.
    fn after_release(&self) {
        self.run_sweeps();
        self.serve_requests();
    }

    /// Pushes completions to the downstream, up to the banked demand, while requests are waiting and
    /// the claim is free.
    fn serve_requests(&self) {
        while self.request_pending.load(Ordering::SeqCst) {
            let Some(mut poll) = self.poll.try_claim() else {
                return;
            };
            self.request_pending.store(false, Ordering::SeqCst);
            if let Some(downstream) = self.downstream.receiver()
                && !self.complete.load(Ordering::SeqCst)
            {
                let emitted = self.emit(
                    &mut poll,
                    &mut |frame| downstream.push(frame),
                    &mut |_| false,
                    self.demand.load(Ordering::Acquire),
                );
                if emitted > 0 {
                    // Accounted before the claim is released, so the next holder cannot
                    // over-deliver.
                    let _ = self
                        .demand
                        .try_update(Ordering::SeqCst, Ordering::SeqCst, |d| {
                            Some(d.saturating_sub(emitted))
                        });
                }
            }
            poll.release_seq_cst();
            self.run_sweeps();
        }
    }

    /// Emits `frame` once `ready` reports completion.
    pub fn after(&self, ready: Box<dyn Ready>, frame: Box<dyn Frame>) {
        self.add(Waits::Ready(ready, frame));
    }

    /// Emits `frame` once `marker`'s work completed (a completion edge on a device lane). A marker of a queue that
    /// is not in this source's pool is never reached, so its frame is never emitted.
    pub fn after_marker(&self, marker: Marker, frame: Box<dyn Frame>) {
        self.add(Waits::Marker(marker, frame));
    }

    /// Emits `frame` at or after `deadline` (timers: a periodic tick, a watchdog, a backoff).
    pub fn at(&self, deadline: Instant, frame: Box<dyn Frame>) {
        self.add(Waits::At(deadline, frame));
    }

    /// Polls `pending` until it yields its continuation.
    pub fn register(&self, pending: Box<dyn Pending>) {
        self.add(Waits::Pending(pending));
    }

    /// Operations still in flight.
    pub fn pending(&self) -> usize {
        self.pending.load(Ordering::Acquire)
    }

    /// How late emissions were after completion, in µs (p50, p99, max since the start, each rounded
    /// up): the cost of finding completions by polls. The maximum is exact; the percentiles are
    /// interpolated within a bucket of [`Histogram`], so they are within one bucket's width (a
    /// quarter of the value) of the exact ones.
    ///
    /// A timer's lateness is exact (emission − deadline). Another operation's is an upper bound:
    /// the time from the last look that found it in flight (or its registration) to the poll that
    /// found it done.
    pub fn lateness_us(&self) -> (u64, u64, u64) {
        let h = self.lateness.snapshot();
        let up = |us: f64| us.ceil() as u64;
        (
            up(h.percentile(0.5).unwrap_or(0.0)),
            up(h.percentile(0.99).unwrap_or(0.0)),
            up(h.max),
        )
    }

    /// One poll, by the holder of the claim: every operation is queried once, and up to `demand`
    /// continuations of the completed ones go to `consumer`, in registration order. Returns how
    /// many.
    fn emit(
        &self,
        poll: &mut ClaimGuard<'_, Poll>,
        consumer: &mut dyn FnMut(Box<dyn Frame>),
        stop: &mut dyn FnMut(&dyn Frame) -> bool,
        demand: u64,
    ) -> u64 {
        if demand == 0 {
            return 0;
        }
        let late = Self::absorb(poll);
        self.cancel_late(late);
        if poll.waiting.is_empty() {
            return 0;
        }
        let waiting = &mut poll.waiting;
        let now = Instant::now();
        // A completed source has let its lanes go: nothing it still holds can be waiting for a
        // device.
        let released = self.complete.load(Ordering::Acquire);
        let mut emitted = 0;
        let mut late = Vec::new();
        let mut index = 0;
        while index < waiting.len() && emitted < demand {
            let entry = &mut waiting[index];
            // Each operation is queried once; a completed one becomes `Done`, so it is never
            // queried again (a query may have effects: a driver call that fetches the result).
            let done = match &mut entry.what {
                Waits::Ready(ready, _) => ready.is_ready(),
                Waits::Marker(marker, _) => released || self.lanes.reached(*marker),
                Waits::At(deadline, _) => now >= *deadline,
                Waits::Pending(pending) => match pending.poll() {
                    Some(frame) => {
                        entry.what = Waits::Done(frame, entry.last_seen);
                        true
                    }
                    None => false,
                },
                Waits::Done(..) => true,
                Waits::Moving => unreachable!("an entry is moving only inside this match"),
            };
            if !done {
                entry.last_seen = now;
                index += 1;
                continue;
            }
            entry.what = match std::mem::replace(&mut entry.what, Waits::Moving) {
                Waits::Ready(_, frame) | Waits::Marker(_, frame) => {
                    Waits::Done(frame, entry.last_seen)
                }
                Waits::At(deadline, frame) => Waits::Done(frame, deadline),
                done @ Waits::Done(..) => done,
                Waits::Pending(_) | Waits::Moving => unreachable!("resolved above"),
            };
            let Waits::Done(frame, _) = &entry.what else {
                unreachable!("just made done")
            };
            // The lattice's direct pulls stop before ordered frames (they go through its routing,
            // which keeps their order): leave it, and everything after it, for a later poll.
            if stop(&**frame) {
                break;
            }
            let Some(Waiting {
                what: Waits::Done(frame, since),
                ..
            }) = waiting.remove(index)
            else {
                unreachable!("a done entry at index")
            };
            late.push(now.saturating_duration_since(since));
            consumer(frame);
            emitted += 1;
        }
        if emitted > 0 {
            self.pending.fetch_sub(emitted as usize, Ordering::AcqRel);
            for sample in late {
                self.lateness.record(sample.as_micros() as f64);
            }
        }
        emitted
    }
}

impl LatticeSource for Completions {
    fn add_downstream(self: Arc<Self>, downstream: Arc<dyn LatticeReceiver>) {
        self.downstream.attach(self.clone(), downstream);
    }

    /// Hands up to `demand` completed continuations to `consumer`, unless another thread holds the
    /// poll: then it returns 0 at once, and the holder serves the requests that arrived meanwhile
    /// when it lets go.
    fn pull(
        &self,
        consumer: &mut dyn FnMut(Box<dyn Frame>),
        stop: &mut dyn FnMut(&dyn Frame) -> bool,
        demand: u64,
    ) -> u64 {
        if demand == 0 {
            return 0;
        }
        let Some(mut poll) = self.poll.try_claim() else {
            return 0;
        };
        let emitted = self.emit(&mut poll, consumer, stop, demand);
        poll.release_seq_cst();
        self.after_release();
        emitted
    }

    /// Banks `demand` and pushes completed continuations to the downstream up to the total. If
    /// another thread holds the poll the request returns at once, and the holder pushes when it lets
    /// go.
    fn request(&self, demand: u64) {
        if self.complete.load(Ordering::SeqCst) {
            // Nothing is requested of a completed source, but this is a poll: it cancels what
            // was registered too late.
            self.sweep();
            return;
        }
        if demand == 0 || self.downstream.receiver().is_none() {
            return;
        }
        // Java's `accumulate`: saturating.
        let _ = self
            .demand
            .try_update(Ordering::SeqCst, Ordering::SeqCst, |current| {
                Some(current.saturating_add(demand))
            });
        self.request_pending.store(true, Ordering::SeqCst);
        self.serve_requests();
    }

    /// # Why every registration ends exactly one way, whatever races this
    ///
    /// A registration is a frame (or a [`Pending`], which has none) and has three possible lives:
    /// refused by `add` on the spot (`cancel`), emitted by a poll, or cancelled behind the marker.
    ///
    /// 1. `add` reads the flag once, before it queues anything. If it reads `true`, the frame is
    ///    cancelled there and never reaches the queue. If `false`, it is queued, exactly once
    ///    (`offer` on the unbounded queue cannot fail; if it did, the frame is cancelled there and
    ///    was not queued). So no frame is both refused and queued.
    /// 2. The first `complete` (the `swap` that reads `false`) sets the flag and *then* appends one
    ///    `Slot::Closed`. Registrations queued by threads that read `false` are ordered against that
    ///    marker by the queue itself: its multi-producer `offer` claims its slot with an `AcqRel`
    ///    compare-exchange on one tail counter, so slots have one total order, and the single consumer
    ///    drains in that order. No memory-ordering argument is needed for the split: a queued frame is
    ///    either ahead of the marker or behind it, by position.
    /// 3. The one consumer (holding the `poll` claim) takes each slot exactly once. A frame drained
    ///    ahead of the marker goes to `waiting` and leaves it only by emission, once. A frame drained
    ///    behind the marker (`closed` is set) goes to `late`, and its `cancel` runs once, outside
    ///    the lock. A frame is in exactly one of those lists, so it ends exactly once: emitted
    ///    (the executor then runs or skips it and calls its `do_finally`) or cancelled.
    ///
    /// The flag's orderings matter only for promptness. `Acquire` on the load and `AcqRel` on the swap
    /// make a `complete` that happened before an `add` (by any synchronisation) visible to it, so the
    /// sequential misuse is refused on the spot and fails visibly in debug builds. A load that misses
    /// the flag costs nothing in correctness: the frame is queued and handled by position, never
    /// stranded in a state with no owner.
    ///
    /// What the queue cannot give is a poll. A frame behind the marker is cancelled by the next `pull`
    /// or `request` (a late `add` also tries to, through `sweep`), and if no poll ever comes, when
    /// the last `Arc` drops (`Drop`): no frame is left without an end. Slots a producer has
    /// claimed but not yet published are invisible to a drain; they are drained by a later poll,
    /// and (`Drop` has `&mut self`) a producer cannot still be inside `offer` then.
    fn complete(&self) {
        if !self.complete.swap(true, Ordering::SeqCst) {
            // Exactly one marker. A registration that did not see the flag is queued behind it
            // or ahead of it, and the queue's order says which (see `add`). No poll state is
            // touched: the next poll, a sweep or the last drop cancels what is behind the marker.
            let _ = self.incoming.offer(Slot::Closed);
            self.downstream.complete();
        }
    }

    fn is_complete(&self) -> bool {
        self.complete.load(Ordering::Acquire)
    }
}

#[cfg(test)]
// The tests' own threads register, complete and poll, as lattice workers would.
#[allow(clippy::disallowed_methods, clippy::disallowed_types)]
mod tests {
    use std::sync::Mutex;
    use std::sync::atomic::AtomicU64;
    use std::time::Duration;

    use euhedral_core::frames::FrameHeader;

    use super::*;
    use crate::{DeviceQueue, LaneId, QueueId};

    /// A lane whose completed value a test sets by hand.
    #[derive(Debug)]
    struct HandLane {
        id: QueueId,
        completed: AtomicU64,
    }

    impl Default for HandLane {
        fn default() -> Self {
            Self {
                id: QueueId::next(),
                completed: AtomicU64::new(0),
            }
        }
    }

    impl DeviceQueue for HandLane {
        fn id(&self) -> QueueId {
            self.id
        }

        fn completed(&self) -> u64 {
            self.completed.load(Ordering::Acquire)
        }

        fn name(&self) -> &str {
            "hand"
        }
    }

    /// A token a test completes by hand, counting how often it was queried.
    #[derive(Clone, Default)]
    struct Token {
        done: Arc<AtomicBool>,
        queries: Arc<AtomicUsize>,
    }

    impl Ready for Token {
        fn is_ready(&mut self) -> bool {
            self.queries.fetch_add(1, Ordering::Relaxed);
            self.done.load(Ordering::Acquire)
        }
    }

    struct Tagged {
        header: FrameHeader,
    }

    impl Frame for Tagged {
        fn header(&self) -> &FrameHeader {
            &self.header
        }

        fn header_mut(&mut self) -> &mut FrameHeader {
            &mut self.header
        }
    }

    fn tagged(id: u64) -> Box<dyn Frame> {
        Box::new(Tagged {
            header: FrameHeader::new(id),
        })
    }

    fn setup() -> (Arc<Completions>, Arc<HandLane>) {
        let lane = Arc::new(HandLane::default());
        let pool = Arc::new(LanePool::new(vec![lane.clone() as Arc<dyn DeviceQueue>]));
        (Completions::new(pool), lane)
    }

    fn poll(completions: &Completions) -> Vec<u64> {
        let mut out = Vec::new();
        completions.pull(
            &mut |frame| out.push(frame.header().id_hash()),
            &mut |_| false,
            u64::MAX,
        );
        out
    }

    #[test]
    fn a_continuation_is_emitted_only_once_its_token_completed() {
        let (completions, _) = setup();
        let token = Token::default();
        completions.after(Box::new(token.clone()), tagged(1));
        assert_eq!(completions.pending(), 1);
        assert!(poll(&completions).is_empty(), "nothing is done at once");
        token.done.store(true, Ordering::Release);
        assert_eq!(poll(&completions), [1]);
        assert_eq!(completions.pending(), 0);
        assert!(poll(&completions).is_empty(), "emitted once");
    }

    #[test]
    fn each_poll_queries_each_pending_token_once() {
        let (completions, _) = setup();
        let tokens: Vec<Token> = (0..5).map(|_| Token::default()).collect();
        for (i, token) in tokens.iter().enumerate() {
            completions.after(Box::new(token.clone()), tagged(i as u64));
        }
        for _ in 0..3 {
            poll(&completions);
        }
        assert!(
            tokens
                .iter()
                .all(|t| t.queries.load(Ordering::Relaxed) == 3)
        );
    }

    #[test]
    fn completed_operations_are_emitted_in_registration_order() {
        let (completions, lane) = setup();
        let late = Token::default();
        completions.after_marker(Marker::new(lane.id, LaneId(0), 2), tagged(10));
        completions.after(Box::new(late.clone()), tagged(11));
        completions.after_marker(Marker::new(lane.id, LaneId(0), 1), tagged(12));
        completions.at(Instant::now(), tagged(13));
        assert_eq!(poll(&completions), [13]);
        lane.completed.store(2, Ordering::Release);
        late.done.store(true, Ordering::Release);
        assert_eq!(poll(&completions), [10, 11, 12]);
    }

    #[test]
    fn a_timer_fires_at_or_after_its_deadline_and_its_lateness_is_measured() {
        let (completions, _) = setup();
        completions.at(Instant::now() + Duration::from_secs(3600), tagged(1));
        completions.at(Instant::now() - Duration::from_millis(5), tagged(2));
        assert_eq!(poll(&completions), [2]);
        assert_eq!(completions.pending(), 1);
        let (p50, _, max) = completions.lateness_us();
        // The maximum is exact. A percentile is interpolated within its bucket (a quarter of the
        // value wide), and never above the maximum.
        assert!(max >= 5_000, "a timer 5 ms overdue is at least 5 ms late");
        assert!(
            p50 >= 5_000 * 3 / 4 && p50 <= max,
            "the median is within a bucket of the one sample: {p50} of {max}"
        );
    }

    #[test]
    fn a_pending_operation_yields_its_own_continuation() {
        struct CountDown(u32);
        impl Pending for CountDown {
            fn poll(&mut self) -> Option<Box<dyn Frame>> {
                self.0 = self.0.saturating_sub(1);
                (self.0 == 0).then(|| tagged(7))
            }
        }
        let (completions, _) = setup();
        completions.register(Box::new(CountDown(3)));
        assert!(poll(&completions).is_empty());
        assert!(poll(&completions).is_empty());
        assert_eq!(poll(&completions), [7]);
        assert_eq!(completions.pending(), 0);
    }

    #[test]
    fn a_poll_emits_no_more_than_its_demand() {
        let (completions, _) = setup();
        for i in 0..5 {
            completions.at(Instant::now(), tagged(i));
        }
        let mut out = Vec::new();
        assert_eq!(
            completions.pull(&mut |f| out.push(f.header().id_hash()), &mut |_| false, 2),
            2
        );
        assert_eq!(out, [0, 1]);
        assert_eq!(poll(&completions), [2, 3, 4]);
    }

    /// The lattice's direct pulls stop before ordered frames: a held-back continuation keeps its
    /// place, and what was registered after it waits behind it.
    #[test]
    fn a_pull_stops_before_the_continuation_its_stop_matches_and_keeps_the_order() {
        let (completions, _) = setup();
        let token = Token::default();
        token.done.store(true, Ordering::Release);
        completions.after(Box::new(token.clone()), tagged(1));
        completions.at(Instant::now(), tagged(2));
        let mut out = Vec::new();
        let pulled = completions.pull(
            &mut |f| out.push(f.header().id_hash()),
            &mut |f| f.header().id_hash() == 1,
            9,
        );
        assert_eq!(
            (pulled, out.len()),
            (0, 0),
            "nothing passes the held-back continuation"
        );
        assert_eq!(completions.pending(), 2);
        assert_eq!(
            poll(&completions),
            [1, 2],
            "a later poll hands both on, in registration order"
        );
        assert_eq!(
            token.queries.load(Ordering::Relaxed),
            1,
            "a completed operation is not queried again"
        );
    }

    #[test]
    fn a_dead_continuation_is_still_emitted_so_its_holdings_end() {
        let (completions, _) = setup();
        let kill = Arc::new(AtomicBool::new(true));
        completions.at(
            Instant::now(),
            Box::new(Tagged {
                header: FrameHeader::with_kill_switch(5, Some(kill)),
            }),
        );
        assert_eq!(poll(&completions), [5]);
    }

    /// A frame that records that its `do_finally` ran.
    struct Finished {
        header: FrameHeader,
        executed: Arc<AtomicBool>,
        finished: Arc<AtomicBool>,
    }

    impl Frame for Finished {
        fn header(&self) -> &FrameHeader {
            &self.header
        }

        fn header_mut(&mut self) -> &mut FrameHeader {
            &mut self.header
        }

        fn execute(&mut self) -> Result<(), euhedral_core::frames::FrameError> {
            self.executed.store(true, Ordering::Release);
            Ok(())
        }

        fn do_finally(self: Box<Self>) {
            self.finished.store(true, Ordering::Release);
        }
    }

    #[test]
    fn a_registration_after_complete_fails_visibly_and_still_finishes_its_frame() {
        let (completions, _) = setup();
        completions.complete();
        let (executed, finished) = (
            Arc::new(AtomicBool::new(false)),
            Arc::new(AtomicBool::new(false)),
        );
        let frame = Finished {
            header: FrameHeader::new(1),
            executed: executed.clone(),
            finished: finished.clone(),
        };
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            completions.at(Instant::now(), Box::new(frame));
        }));
        assert!(
            finished.load(Ordering::Acquire),
            "its do_finally ran, so what it holds ends"
        );
        assert!(
            !executed.load(Ordering::Acquire),
            "it was cancelled, not run"
        );
        assert_eq!(
            result.is_err(),
            cfg!(debug_assertions),
            "debug builds panic"
        );
        assert_eq!(completions.pending(), 0);
        assert!(poll(&completions).is_empty());
    }

    /// A receiver that runs what it is pushed, recording the ids.
    #[derive(Default)]
    struct Recorder {
        ids: Mutex<Vec<u64>>,
        completed: AtomicBool,
    }

    impl euhedral_core::generics::LatticeTerminal for Recorder {
        fn add_upstream(&self, _: Arc<dyn LatticeSource>) {}
    }

    impl LatticeReceiver for Recorder {
        fn push(&self, frame: Box<dyn Frame>) {
            self.ids.lock().unwrap().push(frame.header().id_hash());
        }

        fn on_complete(&self) {
            self.completed.store(true, Ordering::Release);
        }

        fn on_error(&self, _: euhedral_core::frames::Failure) {}
    }

    #[test]
    fn requests_push_up_to_the_accumulated_demand_and_complete_tells_the_downstream() {
        let (completions, _) = setup();
        let recorder = Arc::new(Recorder::default());
        completions.clone().add_downstream(recorder.clone());
        for i in 0..3 {
            completions.at(Instant::now(), tagged(i));
        }
        completions.request(1);
        assert_eq!(*recorder.ids.lock().unwrap(), [0]);
        completions.request(5);
        assert_eq!(*recorder.ids.lock().unwrap(), [0, 1, 2]);
        completions.at(Instant::now(), tagged(3));
        completions.request(0);
        assert_eq!(
            recorder.ids.lock().unwrap().len(),
            3,
            "demand left over from earlier requests is not a request"
        );
        completions.complete();
        assert!(completions.is_complete() && recorder.completed.load(Ordering::Acquire));
    }

    #[test]
    fn a_source_holds_its_lanes_for_its_life_and_drops_them_with_it() {
        // The lane pool is a fixed field, as a final field is: the source lets go of it when the
        // last owner of the source does.
        let lane = Arc::new(HandLane::default());
        let pool = Arc::new(LanePool::new(vec![lane.clone()]));
        let completions = Completions::new(pool.clone());
        drop(pool);
        assert_eq!(
            Arc::strong_count(&lane),
            2,
            "the source holds the lane while it runs"
        );
        LatticeSource::complete(&*completions);
        assert_eq!(
            Arc::strong_count(&lane),
            2,
            "completing touches no lane state"
        );
        drop(completions);
        assert_eq!(
            Arc::strong_count(&lane),
            1,
            "the last owner's drop lets the lane go"
        );
    }

    /// A frame whose lifecycle ends are counted (`do_finally`), and who got it: a test's `ends`
    /// must come to exactly one per frame, whichever of emitted or cancelled it was.
    struct Counted {
        header: FrameHeader,
        ends: Arc<AtomicUsize>,
    }

    impl Frame for Counted {
        fn header(&self) -> &FrameHeader {
            &self.header
        }

        fn header_mut(&mut self) -> &mut FrameHeader {
            &mut self.header
        }

        fn do_finally(self: Box<Self>) {
            self.ends.fetch_add(1, Ordering::AcqRel);
        }
    }

    fn counted(id: u64) -> (Box<dyn Frame>, Arc<AtomicUsize>) {
        let ends = Arc::new(AtomicUsize::new(0));
        let frame = Box::new(Counted {
            header: FrameHeader::new(id),
            ends: ends.clone(),
        });
        (frame, ends)
    }

    /// Runs `body` on its own thread and fails if it does not finish in `limit`: a hung test must
    /// not hang the suite.
    fn within<R: Send + 'static>(limit: Duration, body: impl FnOnce() -> R + Send + 'static) -> R {
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _ = tx.send(body());
        });
        rx.recv_timeout(limit)
            .expect("the test did not finish in time")
    }

    /// A pull as the executor would see it: the continuations are run to their end.
    fn pull_and_end(completions: &Completions) -> usize {
        let mut emitted = 0;
        completions.pull(
            &mut |frame| {
                emitted += 1;
                frame.do_finally();
            },
            &mut |_| false,
            u64::MAX,
        );
        emitted
    }

    /// Behind the marker, by position: queued as a registration that missed the flag would be.
    fn queue_behind_the_marker(completions: &Completions, frame: Box<dyn Frame>) {
        completions.pending.fetch_add(1, Ordering::AcqRel);
        let queued = completions.incoming.offer(Slot::Waiting(Waiting {
            what: Waits::At(Instant::now(), frame),
            last_seen: Instant::now(),
        }));
        assert!(queued.is_ok());
    }

    #[test]
    fn a_registration_ahead_of_the_marker_is_emitted_and_one_behind_it_is_cancelled() {
        let (completions, _) = setup();
        let (ahead, ahead_ends) = counted(1);
        completions.at(Instant::now(), ahead);
        completions.complete();
        let (behind, behind_ends) = counted(2);
        queue_behind_the_marker(&completions, behind);
        assert_eq!(completions.pending(), 2);
        let mut seen = Vec::new();
        completions.pull(
            &mut |frame| {
                seen.push(frame.header().id_hash());
                frame.do_finally();
            },
            &mut |_| false,
            u64::MAX,
        );
        assert_eq!(
            seen,
            [1],
            "only the one registered before complete is emitted"
        );
        assert_eq!(
            (
                ahead_ends.load(Ordering::Acquire),
                behind_ends.load(Ordering::Acquire)
            ),
            (1, 1),
            "each ended once: the first by its emission, the second by its cancellation"
        );
        assert_eq!(completions.pending(), 0);
    }

    #[test]
    fn a_request_of_a_completed_source_cancels_what_landed_behind_the_marker() {
        let (completions, _) = setup();
        completions.complete();
        let (behind, ends) = counted(3);
        queue_behind_the_marker(&completions, behind);
        completions.request(1);
        assert_eq!(ends.load(Ordering::Acquire), 1);
        assert_eq!(completions.pending(), 0);
    }

    #[test]
    fn with_no_poll_the_last_drop_cancels_what_landed_behind_the_marker() {
        let (completions, _) = setup();
        completions.complete();
        let (behind, ends) = counted(4);
        queue_behind_the_marker(&completions, behind);
        assert_eq!(ends.load(Ordering::Acquire), 0, "nobody polled yet");
        drop(completions);
        assert_eq!(
            ends.load(Ordering::Acquire),
            1,
            "no registration is left without an end"
        );
    }

    /// A registration racing `complete` ends exactly once: emitted or cancelled, never both and
    /// never neither. One thread registers while another completes at an arbitrary point and the
    /// test thread polls; the frames that end up behind the marker are cancelled by a poll, and any
    /// left at the end by the drop.
    #[test]
    fn a_registration_racing_complete_runs_exactly_one_lifecycle_end() {
        const ROUNDS: usize = 1500;
        const PER_ROUND: usize = 64;
        within(Duration::from_secs(120), || {
            let mut refused_or_late = 0usize;
            for round in 0..ROUNDS {
                let (completions, _) = setup();
                let ends: Vec<Arc<AtomicUsize>> = (0..PER_ROUND).map(|_| Arc::default()).collect();
                let go = Arc::new(std::sync::Barrier::new(3));
                let registrar = {
                    let (completions, go) = (completions.clone(), go.clone());
                    let frames: Vec<Box<dyn Frame>> = ends
                        .iter()
                        .enumerate()
                        .map(|(i, ends)| {
                            Box::new(Counted {
                                header: FrameHeader::new(i as u64),
                                ends: ends.clone(),
                            }) as Box<dyn Frame>
                        })
                        .collect();
                    std::thread::spawn(move || {
                        go.wait();
                        for frame in frames {
                            // Debug builds panic on a registration that sees the flag: that is
                            // the visible failure, and the frame is cancelled all the same.
                            let completions = &completions;
                            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                completions.at(Instant::now(), frame)
                            }));
                        }
                    })
                };
                let completer = {
                    let (completions, go) = (completions.clone(), go.clone());
                    // Different offsets across rounds put the marker at different places.
                    let spin = (round * 37) % 400;
                    std::thread::spawn(move || {
                        go.wait();
                        for _ in 0..spin {
                            std::hint::spin_loop();
                        }
                        completions.complete();
                    })
                };
                go.wait();
                let mut emitted = 0;
                while !registrar.is_finished() || !completer.is_finished() {
                    emitted += pull_and_end(&completions);
                }
                registrar.join().unwrap();
                completer.join().unwrap();
                emitted += pull_and_end(&completions);
                // A last poll cancels what the registrar queued behind the marker.
                drop(completions);
                for (i, ends) in ends.iter().enumerate() {
                    assert_eq!(
                        ends.load(Ordering::Acquire),
                        1,
                        "round {round}: frame {i} must end exactly once"
                    );
                }
                refused_or_late += PER_ROUND - emitted;
            }
            // The marker falls inside the registrations in some rounds (a cancelled frame exists).
            assert!(
                refused_or_late > 0,
                "no round had a registration after complete"
            );
        });
    }

    /// 8 threads register 10 000 continuations each while the test thread polls by hand: every
    /// continuation is emitted exactly once, and each thread's keep the order it registered them.
    #[test]
    fn eight_threads_register_ten_thousand_each_and_each_is_emitted_once_in_thread_order() {
        const THREADS: u64 = 8;
        const EACH: u64 = 10_000;
        within(Duration::from_secs(120), || {
            let (completions, _) = setup();
            let registrars: Vec<_> = (0..THREADS)
                .map(|t| {
                    let completions = completions.clone();
                    std::thread::spawn(move || {
                        for i in 0..EACH {
                            completions.at(Instant::now(), tagged(t << 32 | i));
                        }
                    })
                })
                .collect();
            let mut next = [0u64; THREADS as usize];
            let mut total = 0u64;
            let deadline = Instant::now() + Duration::from_secs(100);
            while total < THREADS * EACH {
                assert!(
                    Instant::now() < deadline,
                    "the continuations did not all arrive"
                );
                completions.pull(
                    &mut |frame| {
                        let id = frame.header().id_hash();
                        let (t, i) = ((id >> 32) as usize, id & u64::from(u32::MAX));
                        assert_eq!(i, next[t], "thread {t}'s continuations keep their order");
                        next[t] += 1;
                        total += 1;
                    },
                    &mut |_| false,
                    u64::MAX,
                );
            }
            for registrar in registrars {
                registrar.join().unwrap();
            }
            assert_eq!(next, [EACH; THREADS as usize]);
            assert!(poll(&completions).is_empty(), "nothing is emitted twice");
            assert_eq!(completions.pending(), 0);
        });
    }

    /// A receiver that ends each frame it is pushed, as an executor does.
    struct Finisher;

    impl euhedral_core::generics::LatticeTerminal for Finisher {
        fn add_upstream(&self, _: Arc<dyn LatticeSource>) {}
    }

    impl LatticeReceiver for Finisher {
        fn push(&self, frame: Box<dyn Frame>) {
            frame.do_finally();
        }

        fn on_complete(&self) {}

        fn on_error(&self, _: euhedral_core::frames::Failure) {}
    }

    /// 4 threads register while a fifth completes the source and the test thread drives only
    /// `request`: each continuation ends exactly once, emitted ahead of the marker or cancelled
    /// behind it.
    #[test]
    fn registrations_racing_complete_are_each_cancelled_or_emitted_once() {
        const THREADS: usize = 4;
        const PER_THREAD: usize = if cfg!(miri) { 4 } else { 50 };
        const ROUNDS: usize = if cfg!(miri) { 2 } else { 300 };
        within(Duration::from_secs(300), || {
            for round in 0..ROUNDS {
                let (completions, lane) = setup();
                completions.clone().add_downstream(Arc::new(Finisher));
                let ends: Vec<Arc<AtomicUsize>> =
                    (0..THREADS * PER_THREAD).map(|_| Arc::default()).collect();
                let go = Arc::new(std::sync::Barrier::new(THREADS + 2));
                let mut handles = Vec::new();
                for thread in 0..THREADS {
                    let (completions, go) = (completions.clone(), go.clone());
                    let frames: Vec<Box<dyn Frame>> = (0..PER_THREAD)
                        .map(|i| {
                            let id = thread * PER_THREAD + i;
                            Box::new(Counted {
                                header: FrameHeader::new(id as u64),
                                ends: ends[id].clone(),
                            }) as Box<dyn Frame>
                        })
                        .collect();
                    let marker = Marker::new(lane.id, LaneId(0), 0);
                    handles.push(std::thread::spawn(move || {
                        go.wait();
                        for frame in frames {
                            // A registration that sees the flag fails visibly in debug builds.
                            let completions = &completions;
                            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                completions.after_marker(marker, frame)
                            }));
                        }
                    }));
                }
                let completer = {
                    let (completions, go) = (completions.clone(), go.clone());
                    let spin = (round * 29) % 300;
                    std::thread::spawn(move || {
                        go.wait();
                        for _ in 0..spin {
                            std::hint::spin_loop();
                        }
                        completions.complete();
                    })
                };
                go.wait();
                while !handles.iter().all(|h| h.is_finished()) || !completer.is_finished() {
                    completions.request(8);
                }
                for handle in handles {
                    handle.join().unwrap();
                }
                completer.join().unwrap();
                // After `complete` only a pull emits (what was registered ahead of the marker), and
                // it cancels what is behind the marker.
                pull_and_end(&completions);
                drop(completions);
                for (i, ends) in ends.iter().enumerate() {
                    assert_eq!(
                        ends.load(Ordering::Acquire),
                        1,
                        "round {round}: frame {i} must end exactly once"
                    );
                }
            }
        });
    }

    #[test]
    fn a_request_while_another_thread_polls_returns_at_once() {
        let (completions, _) = setup();
        let recorder = Arc::new(Recorder::default());
        completions.clone().add_downstream(recorder.clone());
        completions.at(Instant::now(), tagged(1));
        completions.at(Instant::now(), tagged(2));
        let inside = std::sync::Barrier::new(2);
        let release = std::sync::Barrier::new(2);
        std::thread::scope(|scope| {
            let puller = scope.spawn(|| {
                completions.pull(
                    &mut |_frame| {
                        inside.wait();
                        release.wait();
                    },
                    &mut |_| false,
                    1,
                )
            });
            inside.wait();
            completions.request(1);
            assert!(
                recorder.ids.lock().unwrap().is_empty(),
                "the request did not wait for the poll, and pushed nothing yet"
            );
            assert_eq!(
                completions.demand.load(Ordering::Acquire),
                1,
                "it banked its demand"
            );
            release.wait();
            assert_eq!(puller.join().unwrap(), 1);
        });
        assert_eq!(
            *recorder.ids.lock().unwrap(),
            [2],
            "the poll served the request as it let go"
        );
        assert_eq!(completions.demand.load(Ordering::Acquire), 0);
        assert_eq!(completions.pending(), 0);
    }

    #[test]
    fn a_sweep_that_finds_the_poll_held_is_run_by_the_holder_when_it_lets_go() {
        let (completions, _) = setup();
        completions.at(Instant::now(), tagged(1));
        let (behind, ends) = counted(9);
        let inside = std::sync::Barrier::new(2);
        let release = std::sync::Barrier::new(2);
        std::thread::scope(|scope| {
            let puller = scope.spawn(|| {
                completions.pull(
                    &mut |_frame| {
                        inside.wait();
                        release.wait();
                    },
                    &mut |_| false,
                    1,
                )
            });
            inside.wait();
            completions.complete();
            queue_behind_the_marker(&completions, behind);
            // A poll of a completed source is a sweep; the holder has the claim.
            completions.request(1);
            assert_eq!(
                ends.load(Ordering::Acquire),
                0,
                "the sweep left it to the holder"
            );
            release.wait();
            puller.join().unwrap();
        });
        assert_eq!(
            ends.load(Ordering::Acquire),
            1,
            "the holder swept as it let go"
        );
        assert_eq!(completions.pending(), 0);
    }
}
