//! Completions found by polls (the upstream Java `GpuCompletions`).

use std::cell::Cell;
use std::collections::VecDeque;
use std::fmt;
use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering, fence};
use std::sync::{Arc, Weak};
use std::time::Instant;

use euhedral_core::frames::{Frame, FrameError, FrameHeader};
use euhedral_core::generics::{
    DownstreamSlot, KeyPacing, KeySlot, LatticeReceiver, LatticeSource, SourceCell, SourceGuard,
};
use euhedral_data_structures::queues::{Consumer, MpscQueue, Multi, Producer, Single, SpscQueue};
use euhedral_data_structures::sync::HotScope;

use euhedral_core::metrics::Histogram;

use crate::lanes::{Bell, LanePool, Marker, RetireCell};

/// Whether an operation has completed. One cheap query; never blocks.
pub trait Ready: Send {
    fn is_ready(&mut self) -> bool;
}

/// A condition that its owner keeps and registers again and again ([`Completions::after_wake`]): polled through a
/// shared reference, so the registration holds a clone of an `Arc` and allocates nothing. The owner prepares the
/// condition before each registration and does not touch it again until the frame it was registered with has been
/// emitted.
pub(crate) trait Wake: Send + Sync {
    /// Whether the condition holds now. Called by the polling thread, once per poll, until it does.
    fn is_ready(&self) -> bool;
}

/// An operation in flight and what continues it: polled until it yields the continuation frame. Use
/// it when the completion carries a result (a result fetched from a driver without
/// waiting, a model's output); use [`Completions::after`] when it doesn't.
pub trait Pending: Send {
    /// `Some(frame)` once the operation completed (the frame carries its result or its error);
    /// `None` while it is in flight. Never blocks.
    fn poll(&mut self) -> Option<Box<dyn Frame>>;
}

/// A [`LatticeSource`] of continuations behind completions.
///
/// [`after`](Self::after), [`after_marker`](Self::after_marker), [`at`](Self::at) and
/// [`register`](Self::register) may be called from any thread. **Looking at the operations is a frame's
/// work.** `request` and `pull` run at every visit of a worker, and querying every operation that waits (with the
/// application's code) is high-frequency work, so they hand out the continuations that are already ready, within the
/// demand, and query no operation. While operations wait and no poll frame is out,
/// `request` pushes the source's one key frame, the *poll frame*, and a worker runs it like any other frame. The poll is
/// long work (every registration, and the application's `Pending::poll` code, whatever it costs), so it leaves only
/// through a `request`'s push. A `pull` neither pushes it nor hands it out: a worker that was handed it by a pull would
/// run it through its executor while it holds the handle's lock, and nothing else could take ready work. Its `execute` ([`reap`](Self::reap)) queries each pending operation once (a completion's `is_ready`, a
/// [`Pending`]'s `poll`, a marker, a timer) and queues the continuations of those that completed, in the
/// order they were registered, for the hand-out. When the frame has run it goes home, and the next `request`
/// may push it again, but only when there is something to look at, so that a source that only waits reads
/// unproductive and the worker parks: a registration since the last poll, a timer's deadline (`at`; a timer is not
/// polled before it), a bell that rang (`after_retired`: the lane's callback rings the source's bell when it fills the
/// cell, so the wait costs nothing until then), or, for a wait that can only be polled blind (a marker, a `Ready`, a
/// `Wake`, a `Pending`), the end of a window of time after an empty poll (16 us, doubling to 100 us:
/// [`KeyPacing`](euhedral_core::generics::KeyPacing)). A find ends the window.
/// If a query panics (a `Ready`, a `Wake`, a `Pending`), its entry is finished without running (or dropped), counted
/// ([`panicked`](Self::panicked)) and logged, and the entries behind it are queried all the same. So an operation's query is measured as the frame's cost, and nothing is pushed while
/// nothing waits.
///
/// **Driving it by hand** (tests, or any caller outside a lattice): call [`reap`](Self::reap), which does
/// what the poll frame does, then `pull`, which hands the ready continuations to its `consumer`. A reap
/// finds only what completed by then: reap and pull again until the continuations you expect came out.
///
/// A `pull` stops before the first ready continuation its `stop` matches, leaving it (and everything
/// registered after it) for a later hand-out; the lattice's direct pulls stop before ordered frames, which
/// then go out through `request`'s pushes and the lattice's routing, so continuations on one ordered lane
/// keep their registration order. A completed operation is never queried again.
///
/// A continuation is handed out even if its frame has died (its kill switch is set): the executor then
/// skips it and runs its `do_finally`, which ends what it holds. So at shutdown
/// [`pending`](Self::pending) drains to zero on its own.
///
/// Complete it ([`LatticeSource::complete`]) only once [`pending`](Self::pending) is zero. A
/// registration after that is a bug that fails visibly: its frame finishes at once through
/// `do_finally` without running, so whatever it holds still ends, and debug builds then panic. (An
/// ingest queue that has completed behaves differently: its `offer` hands the refused frame back to the
/// caller as `Err(frame)`, and a frame is cancelled only when the queue is cleared or dropped. These
/// registration methods return nothing, so there is no caller to hand the frame back to.) A [`Pending`]
/// registered after completion has no frame to finish: it is dropped.
///
/// A registration that races `complete` (the flag was not yet visible to it) lands in the queue
/// and ends exactly one way: ahead of `complete`'s marker in the queue it is a registration like
/// any other and is handed out; behind it, the next poll ([`reap`](Self::reap): the poll frame, or a
/// registration's own sweep) cancels it through `do_finally` without running it, and so does the last drop
/// of the source if no poll comes. That cancellation does not panic: only the polling thread sees it, and it
/// cannot tell a misuse from a poll that was late. A `request` or `pull` of a completed source cancels
/// nothing (it runs no application code of its own), and a completed source pushes no poll frame.
pub struct Completions {
    /// The lanes polled for markers: fixed for the source's life and dropped with it. A marker
    /// counts as reached when its lane says so, also after the source has completed.
    lanes: Arc<LanePool>,
    /// Registered since the last poll; any thread appends, lock-free. [`LatticeSource::complete`]
    /// appends one [`Slot::Closed`] marker: what is behind it was registered too late.
    incoming: Producer<Slot, Multi>,
    /// The polling side (the poll frame's, or a caller driving by hand): the queue's consumer, what
    /// it drained and has not found completed, and the end of the queue the completed continuations go to.
    /// State the holder mutates, so a `SourceCell`: [`reap`](Self::reap) enters it, a late registration's
    /// sweep enters it only if free, and whoever finds it entered leaves its work to the holder, which
    /// looks for it after letting go.
    poll: SourceCell<Poll>,
    /// The hand-out side (`request` and `pull`): the continuations found completed, waiting to go out.
    out: SourceCell<Out>,
    /// A sweep waiting for the holder of `poll`: set before the claim is tried, read by the holder
    /// after it releases.
    sweep_pending: AtomicBool,
    /// A request waiting for the holder of `out`; its demand is already banked in `demand`.
    request_pending: AtomicBool,
    /// Registered and not yet handed out.
    pending: AtomicUsize,
    /// Registered and not yet found completed: what a poll has to look at. The poll frame is pushed only
    /// while it is not zero.
    waiting: AtomicUsize,
    demand: AtomicU64,
    downstream: DownstreamSlot,
    complete: AtomicBool,
    /// Registrations whose marker was still unreached when they were cancelled, leaked (see
    /// `cancel`); shared, so it can be read after the drop.
    leaked: Arc<AtomicUsize>,
    /// How late each emission was, in µs, since the start (lock-free: polls record, readers copy).
    lateness: Histogram,
    /// The poll frame, while it is home.
    key: KeySlot<PollFrame>,
    /// The back-off of the poll frame while it polls blind (see [`KeyPacing`]).
    pacing: KeyPacing,
    /// The things that give the poll frame something new to look at: a registration (counted after it is queued) and a
    /// bell that rang (a retirement callback that filled its cell). Shared with the cells that ring it.
    events: Arc<Bell>,
    /// What the clock is measured from: the deadlines below are nanoseconds since it.
    epoch: Instant,
    /// The earliest deadline of the timers waiting (`u64::MAX`: none), as the last poll saw them or a registration
    /// lowered it. A timer needs no polling before its deadline.
    next_at: AtomicU64,
    /// Some waiting entry can only be polled blind (a query that has no deadline and rings no bell).
    blind: AtomicBool,
    /// Entries whose query panicked: finished without running (their frame) or dropped, and counted.
    panicked: AtomicUsize,
}

/// What travels through the registration queue.
enum Slot {
    Waiting(Waiting),
    /// Appended once, by `complete`, after it set the flag: everything behind it was registered
    /// after the source completed.
    Closed,
}

/// What only the polling side touches.
struct Poll {
    consumer: Consumer<Slot, Single>,
    /// Drained from the queue, oldest first, and not yet found completed.
    waiting: VecDeque<Waiting>,
    /// The `Closed` marker was drained: whatever is drained from now on is late.
    closed: bool,
    /// Where the continuations found completed go, in the order they were found.
    ready: Producer<Box<dyn Frame>, Single>,
}

/// What only the hand-out side touches.
struct Out {
    /// The continuations found completed, in the order they were found.
    ready: Consumer<Box<dyn Frame>, Single>,
    /// Continuations a pull's `stop` matched, the oldest first: they go out before anything in `ready`.
    held: VecDeque<Box<dyn Frame>>,
}

/// A registered operation and when it was last seen in flight (an upper bound on when it completed
/// is the poll that found it done; its lateness is at most the time since the previous look).
struct Waiting {
    what: Waits,
    last_seen: Instant,
}

enum Waits {
    Ready(Box<dyn Ready>, Box<dyn Frame>),
    /// A condition its owner keeps (see [`Wake`]).
    Wake(Arc<dyn Wake>, Box<dyn Frame>),
    Marker(Marker, Box<dyn Frame>),
    /// A lane's retirement callback, named by the cell it fills.
    Retired(Arc<RetireCell>, Box<dyn Frame>),
    At(Instant, Box<dyn Frame>),
    Pending(Box<dyn Pending>),
}

/// The id hash of the poll frame. The routing hash of each push is this mixed with the push's own seed, which
/// spreads the pushes over the workers.
const KEY_ID: u64 = 0x9e37_79b9_7f4a_7c15;

/// The key frame of a [`Completions`]: its `execute` polls ([`Completions::reap`]).
struct PollFrame {
    header: FrameHeader,
    source: Weak<Completions>,
    /// Counts the pushes, so that each is routed by a hash of its own.
    seed: u64,
}

impl PollFrame {
    fn new(source: Weak<Completions>) -> Box<Self> {
        Box::new(Self {
            header: FrameHeader::new(KEY_ID),
            source,
            seed: 0,
        })
    }

    /// Unordered, and a new routing hash for each push: the poll may run on any worker.
    fn arm(&mut self) {
        self.seed = self.seed.wrapping_add(1);
        self.header.randomize_hash(self.seed);
    }
}

impl Frame for PollFrame {
    fn header(&self) -> &FrameHeader {
        &self.header
    }

    fn header_mut(&mut self) -> &mut FrameHeader {
        &mut self.header
    }

    fn execute(&mut self) -> Result<(), FrameError> {
        if let Some(source) = self.source.upgrade() {
            // A blind poll that found nothing is followed by a window of time in which a visit pushes this frame only for
            // something new, so that the handle reads unproductive between polls and the worker parks.
            source.pacing.polling(source.events.load(Ordering::Acquire));
            if source.reap() > 0 {
                source.pacing.found();
            } else {
                source.pacing.empty(source.events.load(Ordering::Acquire));
            }
        }
        Ok(())
    }

    /// The key goes home. (A source that is gone has no home: the frame is dropped.)
    fn recycle(self: Box<Self>) {
        if let Some(source) = self.source.upgrade() {
            source.key.home(self);
        }
    }
}

impl Drop for PollFrame {
    /// A key that is dropped without having run (a lattice that discarded it) must not strand the polling: a new
    /// one goes home. A key that is homed does not get here, and there is no other: the slot is empty.
    fn drop(&mut self) {
        if let Some(source) = self.source.upgrade() {
            source
                .key
                .home_replacement(PollFrame::new(self.source.clone()), |frame| {
                    frame.source = Weak::new();
                });
        }
    }
}

impl Drop for Completions {
    /// The last owner lets go: no thread can register any more, and none will poll. What is still
    /// in the queue, and what was absorbed and never found completed, ends here through `Completions::cancel`
    /// (finished, or leaked when its marker is unreached), and so do the continuations that were found
    /// completed and never handed out, so no registration is left without an end.
    fn drop(&mut self) {
        let poll = self.poll.get_mut();
        let mut ends = Self::absorb(poll);
        ends.extend(
            std::mem::take(&mut poll.waiting)
                .into_iter()
                .map(|entry| entry.what),
        );
        for what in ends {
            self.cancel(what);
        }
        let out = self.out.get_mut();
        while let Some(frame) = out.held.pop_front() {
            frame.do_finally();
        }
        while let Some(frame) = out.ready.poll() {
            frame.do_finally();
        }
    }
}

/// Serves what was refused the poll while it was held, when a poll lets go. `serves` says whether the holder may also
/// serve the requests that arrived meanwhile, which pushes: a `request` and a poll frame may, a `pull` may not (it
/// only runs the sweeps; the requests wait in `request_pending` and the ready continuations in `out`, for the next
/// `pull` to hand out or the next `request` to serve).
struct AfterRelease<'a> {
    source: &'a Completions,
    armed: bool,
    serves: bool,
}

impl Drop for AfterRelease<'_> {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        // The entry is a plain drop on the unwinding path: order the reads below after it.
        fence(Ordering::SeqCst);
        if std::thread::panicking() {
            let _ =
                panic::catch_unwind(AssertUnwindSafe(|| self.source.after_release(self.serves)));
        } else {
            self.source.after_release(self.serves);
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
        let ready: SpscQueue<Box<dyn Frame>> = SpscQueue::with_pool(INCOMING_CHUNK, 2);
        Arc::new_cyclic(|me| Self {
            lanes,
            incoming: queue.producer(),
            poll: SourceCell::new(Poll {
                consumer: queue.consumer(),
                waiting: VecDeque::new(),
                closed: false,
                ready: ready.producer(),
            }),
            out: SourceCell::new(Out {
                ready: ready.consumer(),
                held: VecDeque::new(),
            }),
            sweep_pending: AtomicBool::new(false),
            request_pending: AtomicBool::new(false),
            pending: AtomicUsize::new(0),
            waiting: AtomicUsize::new(0),
            demand: AtomicU64::new(0),
            downstream: DownstreamSlot::new(),
            complete: AtomicBool::new(false),
            leaked: Arc::new(AtomicUsize::new(0)),
            lateness: Histogram::new(),
            key: KeySlot::with_key(PollFrame::new(me.clone())),
            pacing: KeyPacing::new(),
            events: Arc::new(Bell::new(0)),
            epoch: Instant::now(),
            next_at: AtomicU64::new(u64::MAX),
            blind: AtomicBool::new(false),
            panicked: AtomicUsize::new(0),
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
            self.cancel(what);
            debug_assert!(
                false,
                "a completion registered after Completions::complete (its frame was cancelled)"
            );
            return;
        }
        // Counted before it can be seen, so `pending` never reads zero while a registration is
        // between here and the queue.
        self.pending.fetch_add(1, Ordering::AcqRel);
        self.waiting.fetch_add(1, Ordering::AcqRel);
        let deadline = match &what {
            Waits::At(deadline, _) => Some(self.nanos(*deadline)),
            _ => None,
        };
        let slot = Slot::Waiting(Waiting {
            what,
            last_seen: Instant::now(),
        });
        if let Err(Slot::Waiting(Waiting { what, .. })) = self.incoming.offer(slot) {
            // Unreachable for the unbounded queue; ends the frame rather than losing it.
            self.pending.fetch_sub(1, Ordering::AcqRel);
            self.waiting.fetch_sub(1, Ordering::AcqRel);
            self.cancel(what);
            return;
        }
        // Published: tell the poll frame. The deadline first, the count last: a visit that sees the count move sees
        // the deadline.
        if let Some(deadline) = deadline {
            self.next_at.fetch_min(deadline, Ordering::Release);
        }
        self.events.fetch_add(1, Ordering::Release);
        // Racing `complete`: the flag may be set by now, and this one may be behind the marker.
        // Do not wait for the next poll to find it (best effort; the poll, or the last drop, is
        // what guarantees it).
        if self.complete.load(Ordering::SeqCst) {
            self.sweep();
        }
    }

    /// Ends what a refused, late or abandoned registration holds: its frame is finished without
    /// running; a [`Pending`] has no frame, so it is dropped.
    ///
    /// A registration that waits for a lane's marker the lane has still not reached is **leaked**
    /// instead (`mem::forget`) and counted in [`leaked_unreached`](Completions::leaked_unreached):
    /// the device may still use what its frame holds, and finishing the frame would release it. It
    /// is the choice an I/O ring makes for its in-flight buffers. Wherever a frame's resources
    /// could be released while the device may still use them, they are not.
    fn cancel(&self, what: Waits) {
        let unreached = match &what {
            Waits::Marker(marker, _) => !self.lanes.reached(*marker),
            Waits::Retired(cell, _) => !cell.arrived(),
            _ => false,
        };
        if unreached {
            self.leaked.fetch_add(1, Ordering::Release);
            std::mem::forget(what);
            return;
        }
        match what {
            Waits::Ready(_, frame)
            | Waits::Wake(_, frame)
            | Waits::Marker(_, frame)
            | Waits::Retired(_, frame)
            | Waits::At(_, frame) => frame.do_finally(),
            Waits::Pending(pending) => drop(pending),
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
            ..
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
            self.cancel(what);
        }
        self.pending.fetch_sub(count, Ordering::AcqRel);
        self.waiting.fetch_sub(count, Ordering::AcqRel);
    }

    /// Cancels what landed too late now, unless a poll holds the claim: that holder absorbs the
    /// queue itself, and looks for this sweep after it lets go. For a late registration (which is the
    /// registering thread's own work; `request` and `pull` cancel nothing).
    fn sweep(&self) {
        // The wakeup is published before the claim is tried: the releasing holder either sees it,
        // or this call takes the claim itself.
        self.sweep_pending.store(true, Ordering::SeqCst);
        self.run_sweeps();
    }

    /// Absorbs the queue and cancels the late ones while a sweep is waiting and the claim is free.
    fn run_sweeps(&self) {
        while self.sweep_pending.load(Ordering::SeqCst) {
            let Some(mut poll) = self.poll.try_enter() else {
                return;
            };
            // One read-modify-write takes the wakeup and synchronizes with the store that set it, so
            // the registration made before that store is seen by the absorb below.
            if !self.sweep_pending.swap(false, Ordering::SeqCst) {
                poll.release_seq_cst();
                continue;
            }
            let late = Self::absorb(&mut poll);
            self.cancel_late(late);
            poll.release_seq_cst();
        }
    }

    /// What a holder of a claim does after it lets go: the sweeps and requests that arrived while
    /// it held the claim (and so were refused it), as a queue sink's holder serves its requests. A pull's `serves` is
    /// false: the requests refused meanwhile stay banked, since a pull must not push.
    fn after_release(&self, serves: bool) {
        self.run_sweeps();
        if serves {
            self.serve_requests();
        }
    }

    /// Pushes the ready continuations to the downstream, up to the banked demand, while requests are
    /// waiting and the hand-out side is free.
    fn serve_requests(&self) {
        while self.request_pending.load(Ordering::SeqCst) {
            let Some(mut out) = self.out.try_enter() else {
                return;
            };
            // As in `run_sweeps`: the read-modify-write sees the demand banked before the request's
            // store, which a plain store of `false` over a racing request's would not guarantee.
            if !self.request_pending.swap(false, Ordering::SeqCst) {
                out.release_seq_cst();
                continue;
            }
            if let Some(downstream) = self.downstream.receiver()
                && !self.complete.load(Ordering::SeqCst)
            {
                // Counted before each push, so a push that panics still counts: the frame left
                // the source.
                let pushed = Cell::new(0_u64);
                let handed = panic::catch_unwind(AssertUnwindSafe(|| {
                    self.hand_out(
                        &mut out,
                        &mut |frame| {
                            pushed.set(pushed.get() + 1);
                            downstream.push(frame);
                        },
                        &mut |_| false,
                        self.demand.load(Ordering::SeqCst),
                    )
                }));
                if pushed.get() > 0 {
                    // Accounted before the side is released, so the next holder cannot
                    // over-deliver.
                    let _ = self
                        .demand
                        .try_update(Ordering::SeqCst, Ordering::SeqCst, |d| {
                            Some(d.saturating_sub(pushed.get()))
                        });
                }
                if let Err(panic) = handed {
                    // The wakeups refused meanwhile are still served.
                    out.release_seq_cst();
                    let _ = panic::catch_unwind(AssertUnwindSafe(|| self.after_release(true)));
                    panic::resume_unwind(panic);
                }
            }
            out.release_seq_cst();
        }
    }

    /// Nanoseconds from the epoch to `at` (0 for an instant before it).
    fn nanos(&self, at: Instant) -> u64 {
        u64::try_from(at.saturating_duration_since(self.epoch).as_nanos()).unwrap_or(u64::MAX - 1)
    }

    /// Whether there is something to look at now: an event since the last poll started (a registration, a bell), a
    /// timer's deadline, or, for entries that can only be polled blind, the end of the back-off window. Nothing else
    /// is: an entry that waits for a bell, or for a deadline not yet reached, is not polled. One clock read, and only
    /// while a timer or a blind entry waits.
    fn poll_due(&self) -> bool {
        if self.pacing.fresh(self.events.load(Ordering::Acquire)) {
            return true;
        }
        let blind = self.blind.load(Ordering::Relaxed);
        let next_at = self.next_at.load(Ordering::Relaxed);
        if !blind && next_at == u64::MAX {
            return false;
        }
        let now = Instant::now();
        (blind && self.pacing.ready_at(now)) || (next_at != u64::MAX && self.nanos(now) >= next_at)
    }

    /// Pushes the poll frame to the downstream: operations wait, there is something to look at, the source is not
    /// complete and the poll frame is home. The key is a control frame, not part of the demand (like a hot source's
    /// pushes), and there is at most one. Does nothing without a downstream (a caller driving by hand has none). Only
    /// `request` calls it: the key is long, so no `pull` hands it out or pushes it.
    fn push_poll(&self) {
        if self.waiting.load(Ordering::Acquire) == 0 || self.complete.load(Ordering::Acquire) {
            return;
        }
        let Some(downstream) = self.downstream.receiver() else {
            return;
        };
        if !self.poll_due() {
            return;
        }
        let Some(mut key) = self.key.take() else {
            return;
        };
        key.arm();
        downstream.push(key);
    }

    /// Emits `frame` once `ready` reports completion.
    pub fn after(&self, ready: Box<dyn Ready>, frame: Box<dyn Frame>) {
        self.add(Waits::Ready(ready, frame));
    }

    /// Emits `frame` once `wake` holds. Nothing is allocated: the registration holds a clone of the `Arc`.
    pub(crate) fn after_wake(&self, wake: Arc<dyn Wake>, frame: Box<dyn Frame>) {
        self.add(Waits::Wake(wake, frame));
    }

    /// Emits `frame` once `marker`'s work completed (a completion edge on a device lane). A marker of a queue that
    /// is not in this source's pool is never reached, so its frame is never emitted.
    pub fn after_marker(&self, marker: Marker, frame: Box<dyn Frame>) {
        self.add(Waits::Marker(marker, frame));
    }

    /// Emits `frame` once a lane's retirement callback filled `cell` (see
    /// [`LaneControl::notify_retired`](crate::LaneControl::notify_retired)): the work submitted before the retirement was
    /// armed has ended, and `cell` holds its status. Nothing is allocated: the registration holds a clone of the `Arc`.
    /// A source dropped before the callback arrived leaks the registration, as it does for a marker not reached.
    pub fn after_retired(&self, cell: Arc<RetireCell>, frame: Box<dyn Frame>) {
        // The callback rings this source's bell when it fills the cell: the wait costs nothing until then. (A cell
        // another source claimed is polled blind.) One look follows from the registration itself.
        cell.ring(&self.events);
        self.add(Waits::Retired(cell, frame));
    }

    /// Emits `frame` at or after `deadline` (timers: a periodic tick, a watchdog, a backoff).
    pub fn at(&self, deadline: Instant, frame: Box<dyn Frame>) {
        self.add(Waits::At(deadline, frame));
    }

    /// Polls `pending` until it yields its continuation.
    pub fn register(&self, pending: Box<dyn Pending>) {
        self.add(Waits::Pending(pending));
    }

    /// How many registrations the source leaked, instead of finishing them, because they were
    /// refused, late or abandoned while their marker was still unreached (see `cancel`). The
    /// counter outlives the source.
    pub fn leaked_unreached(&self) -> Arc<AtomicUsize> {
        self.leaked.clone()
    }

    /// Entries lost because their query panicked (see [`reap`](Self::reap)).
    pub fn panicked(&self) -> usize {
        self.panicked.load(Ordering::Relaxed)
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

    /// Polls once: every operation is queried once, and the continuations of those that completed are
    /// queued, in the order they were registered, for the hand-out ([`pull`](LatticeSource::pull) and
    /// `request`). Returns how many it queued. Returns 0 at once if another poll is inside.
    ///
    /// This is the work of the poll frame that `request` pushes into a lattice, which runs on a worker like
    /// any other frame; `request` and `pull` never do it. A caller that drives a `Completions` by hand calls
    /// it before it pulls. A completed source is polled too (it cancels what landed behind its marker).
    pub fn reap(&self) -> usize {
        // Declared before the claim, so it runs after the claim is let go, also when a query unwinds: the
        // requests and sweeps refused meanwhile are still served.
        let mut after = AfterRelease {
            source: self,
            armed: false,
            serves: true,
        };
        let Some(mut poll) = self.poll.try_enter() else {
            return 0;
        };
        after.armed = true;
        let queued = self.reap_entered(&mut poll);
        poll.release_seq_cst();
        if queued > 0 && self.demand.load(Ordering::SeqCst) > 0 {
            // What was found goes out within the demand already banked, without waiting for the next
            // request: `after` serves it as it drops.
            self.request_pending.store(true, Ordering::SeqCst);
        }
        queued
    }

    /// [`reap`](Self::reap) by the holder of the claim.
    fn reap_entered(&self, poll: &mut SourceGuard<'_, Poll>) -> usize {
        let late = Self::absorb(poll);
        self.cancel_late(late);
        let Poll { waiting, ready, .. } = &mut **poll;
        if waiting.is_empty() {
            self.next_at.store(u64::MAX, Ordering::Relaxed);
            self.blind.store(false, Ordering::Relaxed);
            return 0;
        }
        let now = Instant::now();
        let mut queued = 0;
        let mut index = 0;
        // What the entries left behind need next: the earliest deadline, and whether any can only be polled blind.
        let mut earliest = u64::MAX;
        let mut blind = false;
        while index < waiting.len() {
            let entry = &mut waiting[index];
            // Each operation is queried once, and a completed one leaves the list, so it is never
            // queried again (a query may have effects: a driver call that fetches the result).
            // A query is the application's code: one that panics loses its entry (finished without running, or
            // dropped), is counted, and does not stop the entries behind it.
            let queried = panic::catch_unwind(AssertUnwindSafe(|| match &mut entry.what {
                Waits::Ready(ready, _) => (ready.is_ready(), None),
                Waits::Wake(wake, _) => (wake.is_ready(), None),
                Waits::Marker(marker, _) => (self.lanes.reached(*marker), None),
                Waits::Retired(cell, _) => (cell.arrived(), None),
                Waits::At(deadline, _) => (now >= *deadline, None),
                Waits::Pending(pending) => {
                    let yielded = pending.poll();
                    (yielded.is_some(), yielded)
                }
            }));
            let Ok((done, mut yielded)) = queried else {
                self.panicked.fetch_add(1, Ordering::Relaxed);
                log::error!("an operation's query panicked; its entry is dropped");
                if let Some(Waiting { what, .. }) = waiting.remove(index) {
                    // Ending it runs the application's `do_finally`, which may panic too.
                    let _ = panic::catch_unwind(AssertUnwindSafe(|| self.cancel(what)));
                }
                self.pending.fetch_sub(1, Ordering::AcqRel);
                self.waiting.fetch_sub(1, Ordering::AcqRel);
                continue;
            };
            if !done {
                entry.last_seen = now;
                match &entry.what {
                    Waits::At(deadline, _) => earliest = earliest.min(self.nanos(*deadline)),
                    Waits::Retired(cell, _) if cell.rings(&self.events) => {}
                    _ => blind = true,
                }
                index += 1;
                continue;
            }
            let Some(Waiting { what, last_seen }) = waiting.remove(index) else {
                unreachable!("a done entry at index")
            };
            let (frame, since) = match what {
                Waits::Ready(_, frame)
                | Waits::Wake(_, frame)
                | Waits::Marker(_, frame)
                | Waits::Retired(_, frame) => (frame, last_seen),
                Waits::At(deadline, frame) => (frame, deadline),
                Waits::Pending(_) => (
                    yielded.take().expect("a pending that is done yielded"),
                    last_seen,
                ),
            };
            self.waiting.fetch_sub(1, Ordering::AcqRel);
            // The poll claim is held, so one thread records at a time.
            self.lateness
                .record_owned(now.saturating_duration_since(since).as_micros() as f64);
            // The queue is unbounded: an offer is never refused.
            let _ = ready.offer(frame);
            queued += 1;
        }
        // A registration that raced this scan moved the event count, which the key's `execute` sees: the values it
        // overwrites here are looked at again at the next visit.
        self.next_at.store(earliest, Ordering::Relaxed);
        self.blind.store(blind, Ordering::Relaxed);
        queued
    }

    /// Hands up to `demand` ready continuations to `consumer`, in the order they were found, holding back
    /// the first one `stop` matches. Returns how many.
    fn hand_out(
        &self,
        out: &mut SourceGuard<'_, Out>,
        consumer: &mut dyn FnMut(Box<dyn Frame>),
        stop: &mut dyn FnMut(&dyn Frame) -> bool,
        demand: u64,
    ) -> u64 {
        let mut handed = 0;
        while handed < demand {
            // What an earlier pull's `stop` held back goes first.
            let frame = match out.held.pop_front() {
                Some(frame) => frame,
                None => match out.ready.poll() {
                    Some(frame) => frame,
                    None => break,
                },
            };
            // The lattice's direct pulls stop before ordered frames (they go through its routing,
            // which keeps their order): leave it, and everything after it, for a later hand-out.
            if stop(&*frame) {
                out.held.push_front(frame);
                break;
            }
            // The frame leaves the source here: it is off `pending` before the consumer sees it, so a
            // consumer that unwinds leaves it exact.
            self.pending.fetch_sub(1, Ordering::AcqRel);
            handed += 1;
            consumer(frame);
        }
        handed
    }
}

impl LatticeSource for Completions {
    fn add_downstream(self: Arc<Self>, downstream: Arc<dyn LatticeReceiver>) {
        self.downstream.attach(self.clone(), downstream);
    }

    /// Hands up to `demand` ready continuations to `consumer`, unless another thread holds the
    /// hand-out: then it returns 0 at once. It pushes nothing, not even as it lets go (a request that arrived meanwhile
    /// stays banked for the next `request`, or for the next `pull` to hand out what is ready), and it hands out no
    /// poll frame. It queries no operation: that is the poll frame's work (see [`reap`](Self::reap)), which a
    /// `request` pushes to the downstream, if there is one, while operations wait.
    fn pull(
        &self,
        consumer: &mut dyn FnMut(Box<dyn Frame>),
        stop: &mut dyn FnMut(&dyn Frame) -> bool,
        demand: u64,
    ) -> u64 {
        if demand == 0 {
            return 0;
        }
        // Declared before the entry, so it runs after the entry is let go, also when `consumer`
        // unwinds: the requests and sweeps refused meanwhile are still served.
        let mut after = AfterRelease {
            source: self,
            armed: false,
            serves: false,
        };
        let Some(mut out) = self.out.try_enter() else {
            return 0;
        };
        after.armed = true;
        let handed = self.hand_out(&mut out, consumer, stop, demand);
        out.release_seq_cst();
        handed
    }

    /// Banks `demand`, pushes the ready continuations to the downstream up to the total, and pushes the
    /// poll frame while operations wait (the frame is not part of the demand). If another thread holds the hand-out the request returns at
    /// once, and the holder pushes when it lets go if it is a `request` or a poll frame; a holder that is a `pull` pushes
    /// nothing, and the next `request` serves it.
    fn request(&self, demand: u64) {
        if demand == 0
            || self.complete.load(Ordering::SeqCst)
            || self.downstream.receiver().is_none()
        {
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
        self.push_poll();
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
    /// What the queue cannot give is a poll. A frame behind the marker is cancelled by the next poll
    /// ([`reap`](Completions::reap): the poll frame's work; `request` and `pull` cancel nothing), by a late `add`'s own
    /// `sweep`, and if no poll ever comes, when the last `Arc` drops (`Drop`): no frame is left without an end. Slots a producer has
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

    /// One poll by hand: what the poll frame does, then a pull of what it found.
    fn poll(completions: &Completions) -> Vec<u64> {
        completions.reap();
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

    /// A condition its owner keeps and registers again after each emission.
    #[derive(Default)]
    struct Flag {
        set: AtomicBool,
        queries: AtomicUsize,
    }

    impl Wake for Flag {
        fn is_ready(&self) -> bool {
            self.queries.fetch_add(1, Ordering::Relaxed);
            self.set.load(Ordering::Acquire)
        }
    }

    #[test]
    fn a_kept_wake_is_emitted_when_it_holds_and_can_be_registered_again() {
        let (completions, _) = setup();
        let flag = Arc::new(Flag::default());
        for round in 1..=3u64 {
            completions.after_wake(flag.clone(), tagged(round));
            assert_eq!(completions.pending(), 1);
            assert!(poll(&completions).is_empty(), "round {round}: not yet");
            assert_eq!(flag.queries.load(Ordering::Relaxed), 1, "queried once");
            flag.queries.store(0, Ordering::Relaxed);
            flag.set.store(true, Ordering::Release);
            assert_eq!(poll(&completions), [round]);
            assert_eq!(completions.pending(), 0);
            assert_eq!(
                Arc::strong_count(&flag),
                1,
                "the source let go of it when it emitted"
            );
            flag.set.store(false, Ordering::Release);
            flag.queries.store(0, Ordering::Relaxed);
        }
        assert!(poll(&completions).is_empty(), "emitted once each");
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
        completions.reap();
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
        completions.reap();
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
        // Nothing is found yet (a request polls nothing), so each continuation waits: the poll frame is what the
        // request pushes, and it is not part of the demand.
        completions.request(1);
        assert_eq!(*recorder.ids.lock().unwrap(), [KEY_ID]);
        // What the poll finds goes out within the demand already banked (1), and the rest with the next requests.
        completions.reap();
        assert_eq!(*recorder.ids.lock().unwrap(), [KEY_ID, 0]);
        completions.request(1);
        assert_eq!(
            *recorder.ids.lock().unwrap(),
            [KEY_ID, 0, 1],
            "the demand banked by the requests, up to the total"
        );
        completions.request(5);
        assert_eq!(*recorder.ids.lock().unwrap(), [KEY_ID, 0, 1, 2]);
        completions.at(Instant::now(), tagged(3));
        completions.request(0);
        assert_eq!(
            recorder.ids.lock().unwrap().len(),
            4,
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
        completions.reap();
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
        completions.waiting.fetch_add(1, Ordering::AcqRel);
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
        completions.reap();
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
    fn a_reap_of_a_completed_source_cancels_what_landed_behind_the_marker_and_a_request_does_not() {
        let (completions, _) = setup();
        let hands = Arc::new(Hands::default());
        completions.clone().add_downstream(hands.clone());
        completions.complete();
        let (behind, ends) = counted(3);
        queue_behind_the_marker(&completions, behind);
        // A request runs no application code, and a completed source pushes no poll frame.
        completions.request(1);
        assert_eq!(hands.kept(), 0);
        assert_eq!(ends.load(Ordering::Acquire), 0, "a request cancels nothing");
        completions.reap();
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
                completions.reap();
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

    /// A receiver that runs each frame it is pushed and ends it, as an executor does.
    struct Finisher;

    impl euhedral_core::generics::LatticeTerminal for Finisher {
        fn add_upstream(&self, _: Arc<dyn LatticeSource>) {}
    }

    impl LatticeReceiver for Finisher {
        fn push(&self, mut frame: Box<dyn Frame>) {
            frame.execute().unwrap();
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
    fn a_request_while_another_thread_hands_out_returns_at_once() {
        let (completions, _) = setup();
        let recorder = Arc::new(Recorder::default());
        completions.clone().add_downstream(recorder.clone());
        completions.at(Instant::now(), tagged(1));
        completions.at(Instant::now(), tagged(2));
        completions.reap();
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
        assert!(
            recorder.ids.lock().unwrap().is_empty(),
            "the pull pushed nothing as it let go: a pull never pushes"
        );
        assert_eq!(
            completions.pending(),
            1,
            "the continuation waits in the hand-out"
        );
        // The next request serves it, within the demand banked (1, and this request's 1).
        completions.request(1);
        assert_eq!(*recorder.ids.lock().unwrap(), [2]);
        assert_eq!(completions.demand.load(Ordering::Acquire), 1);
        assert_eq!(completions.pending(), 0);
    }

    /// A pending operation whose poll waits for the test, inside the poll.
    struct HeldPoll {
        inside: Arc<std::sync::Barrier>,
        release: Arc<std::sync::Barrier>,
    }

    impl Pending for HeldPoll {
        fn poll(&mut self) -> Option<Box<dyn Frame>> {
            self.inside.wait();
            self.release.wait();
            Some(tagged(1))
        }
    }

    #[test]
    fn a_sweep_that_finds_the_poll_held_is_run_by_the_holder_when_it_lets_go() {
        let (completions, _) = setup();
        let (inside, release) = (
            Arc::new(std::sync::Barrier::new(2)),
            Arc::new(std::sync::Barrier::new(2)),
        );
        completions.register(Box::new(HeldPoll {
            inside: inside.clone(),
            release: release.clone(),
        }));
        let (behind, ends) = counted(9);
        std::thread::scope(|scope| {
            let poller = scope.spawn(|| completions.reap());
            inside.wait();
            completions.complete();
            queue_behind_the_marker(&completions, behind);
            // A late registration's sweep; the holder has the claim.
            completions.sweep();
            assert_eq!(
                ends.load(Ordering::Acquire),
                0,
                "the sweep left it to the holder"
            );
            release.wait();
            assert_eq!(poller.join().unwrap(), 1);
        });
        assert_eq!(
            ends.load(Ordering::Acquire),
            1,
            "the holder swept as it let go"
        );
        assert_eq!(
            completions.pending(),
            1,
            "the continuation waits for a hand-out"
        );
        assert_eq!(poll(&completions), [1]);
        assert_eq!(completions.pending(), 0);
    }

    #[test]
    fn a_marker_registration_racing_complete_on_an_unreached_lane_is_not_emitted() {
        for _ in 0..if cfg!(miri) { 4 } else { 200 } {
            let (completions, lane) = setup();
            let (frame, ends) = counted(1);
            let marker = Marker::new(lane.id, LaneId(0), 5);
            std::thread::scope(|scope| {
                scope.spawn(|| {
                    // A registration that sees the flag fails visibly in debug builds.
                    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        completions.after_marker(marker, frame)
                    }));
                });
                scope.spawn(|| completions.complete());
            });
            // Registered ahead of the marker or cancelled behind it; never emitted while the lane
            // has not reached it.
            let emitted = pull_and_end(&completions);
            assert_eq!(
                emitted, 0,
                "an unreached lane emits nothing, also after complete"
            );
            let leaks = completions.leaked_unreached();
            drop(completions);
            // Cancelled behind the marker (ended), or registered ahead of it and still waiting on an
            // unreached lane at drop (leaked and counted): one or the other, exactly once.
            assert_eq!(
                ends.load(Ordering::Acquire) + leaks.load(Ordering::Acquire),
                1,
                "the frame ends once or is leaked once"
            );
        }
    }

    #[test]
    fn dropping_the_source_cancels_what_it_absorbed_and_never_emitted() {
        let (completions, _) = setup();
        let token = Token::default();
        let (frame, ends) = counted(1);
        completions.after(Box::new(token), frame);
        assert!(poll(&completions).is_empty(), "absorbed, in flight");
        assert_eq!(ends.load(Ordering::Acquire), 0);
        drop(completions);
        assert_eq!(
            ends.load(Ordering::Acquire),
            1,
            "ended by the drop, not lost"
        );
    }

    #[test]
    fn a_consumer_that_panics_leaves_a_request_refused_meanwhile_to_the_next_request() {
        let (completions, _) = setup();
        let recorder = Arc::new(Recorder::default());
        completions.clone().add_downstream(recorder.clone());
        completions.at(Instant::now(), tagged(1));
        completions.at(Instant::now(), tagged(2));
        completions.reap();
        let inside = std::sync::Barrier::new(2);
        let release = std::sync::Barrier::new(2);
        std::thread::scope(|scope| {
            let puller = scope.spawn(|| {
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    completions.pull(
                        &mut |_frame| {
                            inside.wait();
                            release.wait();
                            panic!("consumer failed");
                        },
                        &mut |_| false,
                        1,
                    )
                }))
            });
            inside.wait();
            completions.request(1);
            release.wait();
            assert!(puller.join().unwrap().is_err());
        });
        assert!(
            recorder.ids.lock().unwrap().is_empty(),
            "the unwinding pull pushed nothing"
        );
        completions.request(1);
        assert_eq!(
            *recorder.ids.lock().unwrap(),
            [2],
            "the next request served it: the banked demand was not lost"
        );
    }

    #[test]
    fn a_panicking_consumer_leaves_pending_exact() {
        let (completions, _) = setup();
        completions.at(Instant::now(), tagged(1));
        assert_eq!(completions.pending(), 1);
        completions.reap();
        let pulled = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            completions.pull(&mut |_| panic!("consumer failed"), &mut |_| false, 1)
        }));
        assert!(pulled.is_err());
        assert_eq!(
            completions.pending(),
            0,
            "the frame left the source when it was handed over"
        );
    }

    /// Takes frames until told otherwise; the first push waits for the test and then panics.
    struct PanicOncePush {
        inside: std::sync::Barrier,
        release: std::sync::Barrier,
        first: AtomicBool,
        ids: Mutex<Vec<u64>>,
    }

    impl euhedral_core::generics::LatticeTerminal for PanicOncePush {
        fn add_upstream(&self, _: Arc<dyn LatticeSource>) {}
    }

    impl LatticeReceiver for PanicOncePush {
        fn push(&self, frame: Box<dyn Frame>) {
            if self.first.swap(false, Ordering::SeqCst) {
                self.inside.wait();
                self.release.wait();
                panic!("push failed");
            }
            self.ids.lock().unwrap().push(frame.header().id_hash());
        }

        fn on_complete(&self) {}

        fn on_error(&self, _: euhedral_core::frames::Failure) {}
    }

    #[test]
    fn a_panicking_push_keeps_demand_exact_and_serves_the_request_refused_meanwhile() {
        let (completions, _) = setup();
        let receiver = Arc::new(PanicOncePush {
            inside: std::sync::Barrier::new(2),
            release: std::sync::Barrier::new(2),
            first: AtomicBool::new(true),
            ids: Mutex::default(),
        });
        completions.clone().add_downstream(receiver.clone());
        completions.at(Instant::now(), tagged(1));
        completions.at(Instant::now(), tagged(2));
        completions.reap();
        std::thread::scope(|scope| {
            let requester = scope.spawn(|| {
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| completions.request(1)))
            });
            receiver.inside.wait();
            completions.request(1);
            receiver.release.wait();
            assert!(requester.join().unwrap().is_err());
        });
        assert_eq!(
            *receiver.ids.lock().unwrap(),
            [2],
            "the unwinding request served the one refused meanwhile"
        );
        assert_eq!(
            completions.demand.load(Ordering::SeqCst),
            0,
            "each frame handed over was taken off the demand, also the one whose push panicked"
        );
        assert_eq!(completions.pending(), 0);
    }

    #[test]
    fn dropping_with_an_unreached_marker_leaks_the_frame_and_counts_it() {
        let (completions, lane) = setup();
        let leaks = completions.leaked_unreached();
        let (unreached, unreached_ends) = counted(1);
        let (reached, reached_ends) = counted(2);
        completions.after_marker(Marker::new(lane.id, LaneId(0), 9), unreached);
        completions.after_marker(Marker::new(lane.id, LaneId(0), 1), reached);
        lane.completed.store(1, Ordering::Release);
        drop(completions);
        assert_eq!(
            unreached_ends.load(Ordering::Acquire),
            0,
            "a frame whose device work may still run is not finished"
        );
        assert_eq!(leaks.load(Ordering::Acquire), 1, "and it is counted");
        assert_eq!(
            reached_ends.load(Ordering::Acquire),
            1,
            "one whose marker was reached is finished"
        );
    }

    #[test]
    fn a_late_registration_on_an_unreached_marker_is_leaked_and_counted() {
        let (completions, lane) = setup();
        let leaks = completions.leaked_unreached();
        completions.complete();
        let late_marker = |seq: u64, id: u64| {
            let (frame, ends) = counted(id);
            completions.pending.fetch_add(1, Ordering::AcqRel);
            completions.waiting.fetch_add(1, Ordering::AcqRel);
            let queued = completions.incoming.offer(Slot::Waiting(Waiting {
                what: Waits::Marker(Marker::new(lane.id, LaneId(0), seq), frame),
                last_seen: Instant::now(),
            }));
            assert!(queued.is_ok());
            ends
        };
        let unreached = late_marker(9, 1);
        let reached = late_marker(0, 2);
        // A poll of a completed source sweeps what landed behind the marker.
        completions.reap();
        assert_eq!(
            unreached.load(Ordering::Acquire),
            0,
            "the device may still use its frame"
        );
        assert_eq!(
            reached.load(Ordering::Acquire),
            1,
            "a reached marker is cancelled"
        );
        assert_eq!(leaks.load(Ordering::Acquire), 1);
        assert_eq!(completions.pending(), 0);
        // A registration refused on the spot is handled the same way (debug builds also panic).
        let (frame, refused) = counted(3);
        let marker = Marker::new(lane.id, LaneId(0), 9);
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            completions.after_marker(marker, frame)
        }));
        assert_eq!(refused.load(Ordering::Acquire), 0);
        assert_eq!(leaks.load(Ordering::Acquire), 2);
    }

    thread_local! {
        /// This thread is executing a frame: what a lattice worker is doing when it runs one.
        static IN_FRAME: Cell<bool> = const { Cell::new(false) };
    }

    /// A receiver that keeps what the source pushes, to be run by the test as a worker would.
    #[derive(Default)]
    struct Hands {
        frames: Mutex<Vec<Box<dyn Frame>>>,
    }

    impl Hands {
        fn kept(&self) -> usize {
            self.frames.lock().unwrap().len()
        }

        /// The ids of what is kept, the poll frame's apart.
        fn ids(&self) -> Vec<u64> {
            self.frames
                .lock()
                .unwrap()
                .iter()
                .map(|f| f.header().id_hash())
                .collect()
        }

        /// Runs every frame kept, as a worker's executor does, and returns how many.
        fn run(&self) -> usize {
            let frames = std::mem::take(&mut *self.frames.lock().unwrap());
            let n = frames.len();
            for mut frame in frames {
                IN_FRAME.set(true);
                let ran = frame.execute();
                IN_FRAME.set(false);
                ran.unwrap();
                frame.do_finally();
            }
            n
        }
    }

    impl euhedral_core::generics::LatticeTerminal for Hands {
        fn add_upstream(&self, _: Arc<dyn LatticeSource>) {}
    }

    impl LatticeReceiver for Hands {
        fn push(&self, frame: Box<dyn Frame>) {
            self.frames.lock().unwrap().push(frame);
        }

        fn on_complete(&self) {}

        fn on_error(&self, _: euhedral_core::frames::Failure) {}
    }

    /// A pending operation whose poll must run inside a frame's execution.
    struct PollsInAFrame(Arc<AtomicUsize>);

    impl Pending for PollsInAFrame {
        fn poll(&mut self) -> Option<Box<dyn Frame>> {
            assert!(
                IN_FRAME.get(),
                "an operation was polled outside a frame's execution (in a request or a pull)"
            );
            self.0.fetch_add(1, Ordering::AcqRel);
            Some(tagged(2))
        }
    }

    #[test]
    fn request_and_pull_query_nothing_and_the_poll_frame_does() {
        let (completions, _) = setup();
        let hands = Arc::new(Hands::default());
        completions.clone().add_downstream(hands.clone());
        let token = Token::default();
        completions.after(Box::new(token.clone()), tagged(1));
        let polls = Arc::new(AtomicUsize::new(0));
        completions.register(Box::new(PollsInAFrame(polls.clone())));

        // Neither call looks at an operation, and nothing is ready to hand out. A pull hands out and pushes no poll
        // frame (it is long); a request pushes it.
        assert_eq!(
            completions.pull(&mut |_| panic!("nothing is ready"), &mut |_| false, 9),
            0
        );
        assert_eq!(
            token.queries.load(Ordering::Relaxed),
            0,
            "no query in a pull"
        );
        assert_eq!(hands.kept(), 0, "a pull pushes nothing");
        completions.request(4);
        assert_eq!(hands.ids(), [KEY_ID], "a request pushes the poll frame");
        completions.request(4);
        assert_eq!(
            token.queries.load(Ordering::Relaxed),
            0,
            "no query in a request"
        );
        assert_eq!(polls.load(Ordering::Relaxed), 0, "no poll in a request");
        assert_eq!(hands.kept(), 1, "one poll frame at a time");

        // The worker that runs it queries each operation once; what it found goes out within the demand banked
        // (8 requested; the poll frame is not part of it).
        assert_eq!(hands.run(), 1);
        assert_eq!(token.queries.load(Ordering::Relaxed), 1);
        assert_eq!(polls.load(Ordering::Relaxed), 1);
        assert_eq!(hands.ids(), [2], "the pending operation's continuation");
        hands.run();

        // The token completes: the next request pushes the poll frame again (it came home), and its
        // continuation follows.
        token.done.store(true, Ordering::Release);
        completions.request(1);
        assert_eq!(hands.ids(), [KEY_ID]);
        hands.run();
        assert_eq!(hands.ids(), [1]);
        hands.run();
        assert_eq!(completions.pending(), 0);
        assert_eq!(token.queries.load(Ordering::Relaxed), 2);
        completions.request(1);
        assert_eq!(hands.kept(), 0, "nothing waits: nothing is pushed");
    }

    #[test]
    fn a_pull_hands_out_what_a_reap_found_without_querying_again() {
        let (completions, _) = setup();
        let token = Token::default();
        token.done.store(true, Ordering::Release);
        completions.after(Box::new(token.clone()), tagged(1));
        completions.at(Instant::now(), tagged(2));
        assert_eq!(completions.reap(), 2);
        assert_eq!(completions.pending(), 2, "found, not yet handed out");
        let mut out = Vec::new();
        let queries = token.queries.load(Ordering::Relaxed);
        for _ in 0..3 {
            completions.pull(&mut |f| out.push(f.header().id_hash()), &mut |_| false, 1);
        }
        assert_eq!(out, [1, 2]);
        assert_eq!(token.queries.load(Ordering::Relaxed), queries);
        assert_eq!(completions.pending(), 0);
    }

    /// Drops what it is pushed, unrun: a lattice that discarded the frame.
    #[derive(Default)]
    struct Discard {
        pushed: AtomicUsize,
    }

    impl euhedral_core::generics::LatticeTerminal for Discard {
        fn add_upstream(&self, _: Arc<dyn LatticeSource>) {}
    }

    impl LatticeReceiver for Discard {
        fn push(&self, _frame: Box<dyn Frame>) {
            self.pushed.fetch_add(1, Ordering::AcqRel);
        }

        fn on_complete(&self) {}

        fn on_error(&self, _: euhedral_core::frames::Failure) {}
    }

    #[test]
    fn a_poll_frame_that_was_dropped_unrun_is_replaced_so_the_polling_is_not_stranded() {
        let (completions, _) = setup();
        let discard = Arc::new(Discard::default());
        completions.clone().add_downstream(discard.clone());
        completions.at(Instant::now() + Duration::from_secs(3600), tagged(1));
        completions.request(1);
        assert_eq!(discard.pushed.load(Ordering::Relaxed), 1);
        // The frame was dropped without running: a new one is at home, and the next request pushes it.
        completions.request(1);
        assert_eq!(discard.pushed.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn a_poll_frame_whose_source_is_gone_does_nothing() {
        let (completions, _) = setup();
        let hands = Arc::new(Hands::default());
        completions.clone().add_downstream(hands.clone());
        completions.at(Instant::now() + Duration::from_secs(3600), tagged(1));
        completions.request(1);
        assert_eq!(hands.kept(), 1);
        drop(completions);
        // The frame holds the source weakly: it runs and ends without one.
        assert_eq!(hands.run(), 1);
    }

    #[test]
    fn the_poll_frame_is_not_part_of_the_demand_and_a_pull_neither_pushes_nor_hands_it_out() {
        let (completions, _) = setup();
        let hands = Arc::new(Hands::default());
        completions.clone().add_downstream(hands.clone());
        completions.at(Instant::now() + Duration::from_secs(3600), tagged(1));
        completions.request(1);
        assert_eq!(hands.kept(), 1);
        assert_eq!(
            completions.demand.load(Ordering::SeqCst),
            1,
            "the demand is untouched"
        );
        hands.run();
        // A timer an hour away needs no polling.
        completions.pull(&mut |_| panic!("nothing is ready"), &mut |_| false, 1);
        completions.request(1);
        assert_eq!(hands.kept(), 0);
        // A new registration is something new to look at, and the poll is wanted. A pull neither pushes the frame nor
        // hands it to its consumer: it is long (it queries every operation, with the application's code), and a frame a
        // worker runs from its pull keeps the worker in the handle meanwhile. A request pushes it.
        completions.at(Instant::now() + Duration::from_secs(3600), tagged(2));
        assert_eq!(
            completions.pull(
                &mut |_| panic!("a key is not handed out by a pull"),
                &mut |_| false,
                8
            ),
            0
        );
        assert_eq!(hands.kept(), 0, "a pull pushes nothing");
        completions.request(1);
        assert_eq!(hands.kept(), 1);
        // One at a time, whichever way it was asked for.
        completions.pull(
            &mut |_| panic!("a key is not handed out by a pull"),
            &mut |_| false,
            8,
        );
        completions.request(1);
        assert_eq!(hands.kept(), 1);
    }

    /// A pending operation that never ends, counting its polls.
    struct NeverReady(Arc<AtomicUsize>);

    impl Pending for NeverReady {
        fn poll(&mut self) -> Option<Box<dyn Frame>> {
            self.0.fetch_add(1, Ordering::AcqRel);
            None
        }
    }

    #[test]
    fn a_registration_that_never_completes_is_polled_ever_less_often_and_the_source_reads_unproductive_between()
     {
        // A worker that finds nothing to do parks only if the handle it visits reads unproductive: a visit that
        // pushes a frame does not. So between polls, a visit must push nothing. The polls of a blind wait are
        // spaced by a window of time that doubles from 16 us to 100 us, whatever the visit rate.
        let (completions, _) = setup();
        let hands = Arc::new(Hands::default());
        completions.clone().add_downstream(hands.clone());
        let polls = Arc::new(AtomicUsize::new(0));
        completions.register(Box::new(NeverReady(polls.clone())));
        let started = Instant::now();
        let mut visits = 0_u32;
        let mut pushes = 0;
        while started.elapsed() < Duration::from_millis(30) {
            if visits.is_multiple_of(2) {
                completions.pull(&mut |_| panic!("nothing is ready"), &mut |_| false, 8);
            } else {
                completions.request(8);
            }
            visits += 1;
            pushes += hands.run();
        }
        let elapsed = started.elapsed().as_micros() as usize;
        let polled = polls.load(Ordering::Relaxed);
        assert_eq!(
            polled, pushes,
            "one poll per poll frame, and nothing else polls"
        );
        assert!(polled >= 4, "it is still polled ({polled})");
        // After the first polls (the window starts at 16 us and reaches 100 us at the fourth) no two polls are less
        // than 100 us apart, so there are at most one per 100 us of the run, and a few more at the start.
        assert!(
            polled <= 6 + elapsed / 100,
            "{polled} polls in {elapsed} us and {visits} visits: the back-off did not grow"
        );
        // A new registration is polled at the next visit, however far the window had grown, and finding something
        // ends the back-off.
        completions.at(Instant::now(), tagged(9));
        completions.request(8);
        assert_eq!(hands.run(), 1, "a registration asks for a poll at once");
        assert_eq!(polls.load(Ordering::Relaxed), polled + 1);
        assert_eq!(hands.ids(), [9], "and what it found goes out");
        hands.run();
        completions.request(8);
        assert_eq!(hands.kept(), 1, "it found something: no window");
    }

    #[test]
    fn a_timer_needs_no_polling_before_its_deadline_and_is_polled_at_the_first_visit_after_it() {
        let (completions, _) = setup();
        let hands = Arc::new(Hands::default());
        completions.clone().add_downstream(hands.clone());
        let deadline = Instant::now() + Duration::from_millis(3);
        completions.at(deadline, tagged(5));
        // The registration is something new: one poll, which finds nothing and learns the deadline.
        completions.request(8);
        assert_eq!(hands.run(), 1);
        // No visit pushes anything while the deadline is ahead (a visit that ended before it read the clock before it).
        let mut visits = 0_u32;
        loop {
            if visits.is_multiple_of(2) {
                completions.pull(&mut |_| panic!("nothing is ready"), &mut |_| false, 8);
            } else {
                completions.request(8);
            }
            visits += 1;
            if Instant::now() >= deadline {
                break;
            }
            assert_eq!(hands.kept(), 0, "polled before the deadline");
        }
        // The first request after the deadline pushes the poll frame, which finds the timer. (A pull pushes nothing.)
        while Instant::now() < deadline {}
        if hands.kept() == 0 {
            completions.pull(&mut |_| panic!("nothing is ready"), &mut |_| false, 8);
            assert_eq!(hands.kept(), 0, "a pull pushes nothing");
            completions.request(8);
        }
        assert_eq!(
            hands.run(),
            1,
            "polled at the first visit after the deadline"
        );
        // What it found went out within the demand the requests banked.
        assert_eq!(hands.ids(), [5]);
        assert_eq!(completions.pending(), 0);
    }

    #[test]
    fn a_timer_registered_earlier_than_the_one_waiting_is_polled_at_its_own_deadline() {
        let (completions, _) = setup();
        let hands = Arc::new(Hands::default());
        completions.clone().add_downstream(hands.clone());
        completions.at(Instant::now() + Duration::from_secs(3600), tagged(1));
        completions.request(8);
        hands.run();
        completions.request(8);
        assert_eq!(hands.kept(), 0, "an hour away: nothing to look at");
        let soon = Instant::now() + Duration::from_millis(1);
        completions.at(soon, tagged(2));
        completions.request(8);
        assert_eq!(hands.run(), 1, "the new timer is something new to look at");
        while Instant::now() < soon {}
        completions.request(8);
        assert_eq!(
            hands.run(),
            1,
            "and it is polled at its deadline, not the hour's"
        );
    }

    #[test]
    fn a_wait_that_rings_a_bell_is_not_polled_until_it_rings() {
        let (completions, _) = setup();
        let hands = Arc::new(Hands::default());
        completions.clone().add_downstream(hands.clone());
        let cell = Arc::new(RetireCell::new());
        completions.after_retired(cell.clone(), tagged(8));
        completions.request(8);
        assert_eq!(hands.run(), 1, "one look, from the registration");
        for visit in 0..20_000 {
            if visit % 2 == 0 {
                completions.pull(&mut |_| panic!("nothing is ready"), &mut |_| false, 8);
            } else {
                completions.request(8);
            }
            assert_eq!(
                hands.kept(),
                0,
                "a wait with a bell costs nothing while it waits"
            );
        }
        // The callback fills the cell and rings the bell: the next visit looks.
        cell.arrive(0);
        completions.request(8);
        assert_eq!(hands.run(), 1);
        assert_eq!(hands.ids(), [8], "and what it found goes out");
    }

    #[test]
    fn a_callback_that_arrived_before_the_registration_is_found_by_the_look_it_asks_for() {
        let (completions, _) = setup();
        let hands = Arc::new(Hands::default());
        completions.clone().add_downstream(hands.clone());
        let cell = Arc::new(RetireCell::new());
        cell.arrive(0);
        completions.after_retired(cell.clone(), tagged(4));
        // The callback rang nothing (nobody had claimed the cell), and the wait is not blind: the registration is the
        // event that makes the key look, and the look finds the cell filled.
        completions.request(8);
        assert_eq!(hands.run(), 1);
        completions.request(8);
        assert_eq!(hands.ids(), [4]);
    }

    #[test]
    fn a_cell_another_source_claimed_is_polled_blind() {
        let (first, _) = setup();
        let (second, _) = setup();
        let hands = Arc::new(Hands::default());
        second.clone().add_downstream(hands.clone());
        let cell = Arc::new(RetireCell::new());
        first.after_retired(cell.clone(), tagged(1));
        // The cell rings the first source's bell, not this one's: this one cannot wait for a ring.
        second.after_retired(cell.clone(), tagged(2));
        second.request(8);
        hands.run();
        cell.arrive(0);
        let deadline = Instant::now() + Duration::from_secs(5);
        while !hands.ids().contains(&2) {
            assert!(Instant::now() < deadline, "the cell was never seen");
            second.request(8);
            hands.run();
        }
    }

    #[test]
    fn a_registration_that_races_a_poll_is_not_lost_to_its_back_off() {
        let (completions, _) = setup();
        let hands = Arc::new(Hands::default());
        completions.clone().add_downstream(hands.clone());
        let polls = Arc::new(AtomicUsize::new(0));
        completions.register(Box::new(NeverReady(polls.clone())));
        completions.request(8);
        hands.run();
        // The key starts a poll; a registration arrives while it runs; the poll ends empty.
        completions
            .pacing
            .polling(completions.events.load(Ordering::Acquire));
        completions.register(Box::new(NeverReady(polls.clone())));
        completions
            .pacing
            .empty(completions.events.load(Ordering::Acquire));
        completions.request(8);
        assert_eq!(
            hands.kept(),
            1,
            "the new registration is polled at the next visit"
        );
    }

    /// A pending operation whose poll panics.
    struct PanicsOnPoll;

    impl Pending for PanicsOnPoll {
        fn poll(&mut self) -> Option<Box<dyn Frame>> {
            panic!("a poll failed");
        }
    }

    /// A completion whose query panics.
    struct PanicsOnQuery;

    impl Ready for PanicsOnQuery {
        fn is_ready(&mut self) -> bool {
            panic!("a query failed");
        }
    }

    #[test]
    fn a_query_that_panics_loses_its_entry_and_not_the_others() {
        let (completions, _) = setup();
        let (before, before_ends) = counted(1);
        let (panicking, panicking_ends) = counted(2);
        completions.at(Instant::now(), tagged(0));
        completions.after(Box::new(PanicsOnQuery), panicking);
        completions.register(Box::new(PanicsOnPoll));
        completions.at(Instant::now(), tagged(3));
        completions.at(Instant::now(), before);
        // The reap does not unwind, and goes on to the entries behind the one that panicked.
        assert_eq!(completions.reap(), 3);
        assert_eq!(poll(&completions), [0, 3, 1]);
        assert_eq!(completions.pending(), 0, "the lost entries left `pending`");
        assert_eq!(
            panicking_ends.load(Ordering::Acquire),
            1,
            "the frame of the entry that panicked was finished, not run"
        );
        assert_eq!(before_ends.load(Ordering::Acquire), 0);
        assert_eq!(completions.waiting.load(Ordering::Acquire), 0);
        // And nothing loops: the entries are gone.
        assert_eq!(completions.reap(), 0);
    }

    #[test]
    fn the_poll_frame_is_pushed_by_a_request_never_by_a_pull_and_a_key_out_is_never_out_twice() {
        let (completions, _) = setup();
        let hands = Arc::new(Hands::default());
        completions.clone().add_downstream(hands.clone());
        let polls = Arc::new(AtomicUsize::new(0));
        completions.register(Box::new(NeverReady(polls.clone())));
        // The poll is wanted (a registration) and the frame is home, and every pull, with whatever demand, hands out and
        // pushes nothing: only a request sends the poll frame.
        for demand in [1, 8, u64::MAX] {
            assert_eq!(
                completions.pull(
                    &mut |_| panic!("a key is not handed out by a pull"),
                    &mut |_| true,
                    demand
                ),
                0
            );
            assert_eq!(
                completions.pull(
                    &mut |_| panic!("a key is not handed out by a pull"),
                    &mut |_| false,
                    demand
                ),
                0
            );
        }
        assert_eq!(hands.kept(), 0, "no pull pushed it");
        assert_eq!(polls.load(Ordering::Relaxed), 0, "and none polled");
        completions.request(1);
        assert_eq!(hands.kept(), 1, "a request pushes it");
        // It is out: neither another request nor a pull sends a second.
        completions.request(1);
        completions.pull(
            &mut |_| panic!("a key is not handed out by a pull"),
            &mut |_| false,
            8,
        );
        assert_eq!(hands.kept(), 1);
        assert_eq!(hands.run(), 1);
        assert_eq!(polls.load(Ordering::Relaxed), 1, "it polled once");
    }

    #[test]
    fn a_poll_frame_that_runs_while_a_pull_holds_the_hand_out_leaves_the_pushing_to_a_request() {
        let (completions, _) = setup();
        let hands = Arc::new(Hands::default());
        completions.clone().add_downstream(hands.clone());
        let done = |tag: u64| {
            let token = Token::default();
            token.done.store(true, Ordering::Release);
            completions.after(Box::new(token), tagged(tag));
        };
        for tag in [1, 2, 3] {
            done(tag);
        }
        assert_eq!(completions.reap(), 3);
        let mut handed = Vec::new();
        completions.pull(
            &mut |frame| {
                handed.push(frame.header().id_hash());
                if handed.len() > 1 {
                    return;
                }
                // The worker that ran the poll frame while this pull holds the hand-out: a request banks demand, and a
                // reap finds one more. Neither can enter the hand-out, so what they leave is the next call's.
                completions.request(8);
                done(4);
                assert_eq!(completions.reap(), 1);
            },
            &mut |_| false,
            1,
        );
        assert_eq!(handed, [1]);
        assert_eq!(
            hands.kept(),
            0,
            "the pull's release pushed nothing: a pull never pushes"
        );
        // The next request serves what waited, within the demand it banked.
        completions.request(1);
        assert_eq!(hands.ids(), [2, 3, 4]);
        assert_eq!(completions.pending(), 0);
    }
}
