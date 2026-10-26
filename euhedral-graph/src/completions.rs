//! Completions found by polls (Euhedral-Inference `GpuCompletions`).

use std::collections::VecDeque;
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::Instant;

use euhedral_core::frames::Frame;
use euhedral_core::generics::{LatticeReceiver, LatticeSource};

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
pub struct Completions {
    /// The lanes polled for markers; released at [`LatticeSource::complete`] (the lattice that
    /// holds this source may outlive the run, and the lanes must not outlive it: they keep the
    /// device alive).
    lanes: Mutex<Option<Arc<LanePool>>>,
    /// Registered since the last poll; any thread appends.
    incoming: Mutex<Vec<Waiting>>,
    /// Only the polling thread touches it (`pull` and `request` are called by one thread at a
    /// time); the lock is never contended.
    waiting: Mutex<VecDeque<Waiting>>,
    pending: AtomicUsize,
    demand: AtomicU64,
    downstream: Mutex<Option<Arc<dyn LatticeReceiver>>>,
    complete: AtomicBool,
    /// How late each emission was, in µs, since the start (lock-free: polls record, readers copy).
    lateness: Histogram,
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

impl fmt::Debug for Completions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Completions")
            .field("pending", &self.pending())
            .finish_non_exhaustive()
    }
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

impl Completions {
    /// Polls of `marker`s use `lanes`.
    pub fn new(lanes: Arc<LanePool>) -> Arc<Self> {
        Arc::new(Self {
            lanes: Mutex::new(Some(lanes)),
            incoming: Mutex::new(Vec::new()),
            waiting: Mutex::new(VecDeque::new()),
            pending: AtomicUsize::new(0),
            demand: AtomicU64::new(0),
            downstream: Mutex::new(None),
            complete: AtomicBool::new(false),
            lateness: Histogram::new(),
        })
    }

    fn add(&self, what: Waits) {
        {
            // `complete` sets the flag under this lock, so a registration either lands before it or
            // is refused.
            let mut incoming = lock(&self.incoming);
            if !self.complete.load(Ordering::Acquire) {
                self.pending.fetch_add(1, Ordering::AcqRel);
                incoming.push(Waiting {
                    what,
                    last_seen: Instant::now(),
                });
                return;
            }
        }
        match what {
            Waits::Ready(_, frame)
            | Waits::Marker(_, frame)
            | Waits::At(_, frame)
            | Waits::Done(frame, _) => frame.do_finally(),
            Waits::Pending(pending) => drop(pending),
            Waits::Moving => {}
        }
        debug_assert!(
            false,
            "a completion registered after Completions::complete (its frame was cancelled)"
        );
    }

    /// Emits `frame` once `ready` reports completion.
    pub fn after(&self, ready: Box<dyn Ready>, frame: Box<dyn Frame>) {
        self.add(Waits::Ready(ready, frame));
    }

    /// Emits `frame` once `marker`'s work completed (a completion edge on a device lane).
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

    /// One poll: every operation is queried once, and up to `demand` continuations of the completed
    /// ones go to `consumer`, in registration order. Returns how many.
    fn emit(
        &self,
        consumer: &mut dyn FnMut(Box<dyn Frame>),
        stop: &mut dyn FnMut(&dyn Frame) -> bool,
        demand: u64,
    ) -> u64 {
        if demand == 0 {
            return 0;
        }
        let mut waiting = lock(&self.waiting);
        {
            let mut incoming = lock(&self.incoming);
            if incoming.is_empty() && waiting.is_empty() {
                return 0;
            }
            waiting.extend(incoming.drain(..));
        }
        let now = Instant::now();
        let lanes = lock(&self.lanes).clone();
        let mut emitted = 0;
        let mut late = Vec::new();
        let mut index = 0;
        while index < waiting.len() && emitted < demand {
            let entry = &mut waiting[index];
            // Each operation is queried once; a completed one becomes `Done`, so it is never
            // queried again (a query may have effects: a driver call that fetches the result).
            let done = match &mut entry.what {
                Waits::Ready(ready, _) => ready.is_ready(),
                // A completed source has let its lanes go: nothing it still holds can be waiting
                // for a device.
                Waits::Marker(marker, _) => lanes.as_ref().is_none_or(|l| l.reached(*marker)),
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
        drop(waiting);
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
        if self.complete.load(Ordering::Acquire) {
            downstream.on_complete();
            return;
        }
        *lock(&self.downstream) = Some(downstream.clone());
        downstream.add_upstream(self);
    }

    fn pull(
        &self,
        consumer: &mut dyn FnMut(Box<dyn Frame>),
        stop: &mut dyn FnMut(&dyn Frame) -> bool,
        demand: u64,
    ) -> u64 {
        self.emit(consumer, stop, demand)
    }

    fn request(&self, demand: u64) {
        if demand == 0 || self.complete.load(Ordering::Acquire) {
            return;
        }
        let total = self
            .demand
            .fetch_add(demand, Ordering::AcqRel)
            .saturating_add(demand);
        let Some(downstream) = lock(&self.downstream).clone() else {
            return;
        };
        let emitted = self.emit(&mut |frame| downstream.push(frame), &mut |_| false, total);
        if emitted > 0 {
            let _ = self
                .demand
                .try_update(Ordering::AcqRel, Ordering::Acquire, |d| {
                    Some(d.saturating_sub(emitted))
                });
        }
    }

    fn complete(&self) {
        let first = {
            let _incoming = lock(&self.incoming);
            !self.complete.swap(true, Ordering::AcqRel)
        };
        if first {
            lock(&self.lanes).take();
            if let Some(downstream) = lock(&self.downstream).take() {
                downstream.on_complete();
            }
        }
    }

    fn is_complete(&self) -> bool {
        self.complete.load(Ordering::Acquire)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicU64;
    use std::time::Duration;

    use euhedral_core::frames::FrameHeader;

    use super::*;
    use crate::{DeviceQueue, LaneId};

    /// A lane whose completed value a test sets by hand.
    #[derive(Debug, Default)]
    struct HandLane {
        completed: AtomicU64,
    }

    impl DeviceQueue for HandLane {
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
        completions.after_marker(
            Marker {
                lane: LaneId(0),
                value: 2,
            },
            tagged(10),
        );
        completions.after(Box::new(late.clone()), tagged(11));
        completions.after_marker(
            Marker {
                lane: LaneId(0),
                value: 1,
            },
            tagged(12),
        );
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
    fn a_completed_source_lets_its_lanes_go() {
        // The lattice that holds a source may outlive the run; the lanes (and the device behind
        // them) must not.
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
            1,
            "a completed source holds no lane"
        );
    }
}
