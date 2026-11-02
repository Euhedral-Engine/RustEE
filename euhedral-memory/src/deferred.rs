//! Blocking calls that wait for a moment when nothing is in flight.
//!
//! On the CUDA driver, a handful of calls make every other thread's enqueue on the same context wait for the kernels that
//! are running when the call is made: `cuModuleLoadDataEx`, `cuMemFree`, `cuMemFreeHost` and `cuMemHostUnregister`. A
//! `cuEventRecord` on an unrelated stream, a lane's marker, then takes as long as the longest running kernel (measured at
//! 249 ms and 1.25 s on an RTX 5070 Ti, driver 615.71). Running the call on another thread does not help: the stall is the
//! context's, not the caller's. `cuMemHostAlloc` and `cuMemHostRegister` hold the driver for their own length instead. A
//! call that stalls has to be made when no lane has work in flight.
//!
//! [`DeferredCalls`] is a [`LatticeSource`] that holds such calls until that moment. It belongs to a device
//! ([`Device::attach_deferral`](crate::cuda::Device::attach_deferral) in the `cuda` feature), and what the device
//! allocates and loads (`DeviceMemory::free`, `PinnedHost::trim`, dropping an `Imported`, euhedral-gpu's module unload)
//! queues its call through a [`Deferral`], the producer handle, instead of making it where the memory is dropped.
//!
//! - **One consumer.** The calls are made by one thread at a time, in the order they were queued, each exactly once: the
//!   consumer's end of the queue lives in the source's [`SourceCell`]. A call is made inside the cell, and a drain also
//!   makes the calls queued while it runs. A thread that finds the cell entered does not see an empty queue, it sees a
//!   busy one: [`DeferredCalls::drain`] returns 0 at once and [`DeferredCalls::run_through`] waits.
//! - **Key frames.** The calls leave as blocking key frames, by request and push, never by pull: [`DeferredCalls::pull`]
//!   hands out nothing. The owner attaches the source to the lattice that may block (`add_downstream`) and requests
//!   frames with `request`. A frame drains the whole queue.
//! - **When the frame leaves.** [`DeferredCalls::poll`] (also called by every `defer`) pushes one when calls are queued
//!   and either the attached lane pool has nothing in flight ([`LanePool::idle`], every lane's last marker reached) or a
//!   bound is passed: the queued bytes reach [`Bounds::max_bytes`], or the oldest call has waited [`Bounds::max_age`].
//!   **A forced drain stalls every stream's markers for as long as the kernels running at that moment.** Nothing calls
//!   `poll` on its own: the owner calls it from a tick it already has.
//! - **Tickets.** `defer` returns a [`Ticket`]; [`DeferredCalls::run_through`] makes the calls queued up to it on the calling
//!   thread (waiting for a drain that is running), for the owner that cannot go on without the memory.
//! - **Panics.** Each call runs under `catch_unwind`; a panic is counted ([`DeferredCalls::panicked`]) and the rest still run.
//! - **Drop.** Dropping the source makes the calls still queued, on the dropping thread: a deferred free is a promise to free,
//!   and the values a call owns (a mapping that must be unmapped after it is unregistered) are never dropped unrun. A call
//!   queued after a stop, with no drain left, is not made; the owner drains at its stop.
//!
//! The queue is lock-free. A `defer` makes one allocation (the boxed call) and a chunk of the queue per 64 calls (pooled).

use std::fmt;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use euhedral_core::flow_control::RoutingPolicy;
use euhedral_core::frames::{Frame, RunnableFrame};
use euhedral_core::generics::{DownstreamSlot, LatticeReceiver, LatticeSource, SourceCell};
use euhedral_data_structures::queues::{Consumer, MpscQueue, Multi, Producer, Single};
use euhedral_data_structures::sync::HotScope;
use euhedral_graph::LanePool;

/// A position in the queue: the calls queued up to and including one `defer`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Ticket(u64);

impl Ticket {
    /// The ticket of nothing queued: already reached.
    pub const NONE: Ticket = Ticket(0);
}

/// When a drain is forced although lanes have work in flight.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bounds {
    /// Queued bytes (the sizes the callers gave `defer`) that force a drain.
    pub max_bytes: u64,
    /// How long the oldest queued call may wait before a drain is forced.
    pub max_age: Duration,
}

impl Default for Bounds {
    fn default() -> Self {
        Bounds {
            max_bytes: 512 << 20,
            max_age: Duration::from_secs(1),
        }
    }
}

struct Call {
    bytes: u64,
    run: Box<dyn FnOnce() + Send>,
}

/// The consumer's plain state, entered by one thread at a time.
struct Drain {
    consumer: Consumer<Call, Single>,
    /// Calls made so far (the value `completed` publishes).
    made: u64,
}

thread_local! {
    /// This thread is inside a drain (a call made by it is running): a nested wait would never end.
    static DRAINING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// The deferred calls of one device: a `LatticeSource` of key frames with one consumer. See the [module](self).
pub struct DeferredCalls {
    // Held so the queue outlives its two ends.
    _queue: MpscQueue<Call>,
    producer: Producer<Call, Multi>,
    drain: SourceCell<Drain>,
    /// Calls queued (each `defer` adds one after its call is in the queue).
    queued: AtomicU64,
    /// Calls made (the consumer publishes after each).
    completed: AtomicU64,
    /// Bytes queued and not yet made.
    bytes: AtomicU64,
    /// Milliseconds since `epoch` at which the oldest queued call was queued, plus one; 0 when the queue is empty.
    oldest: AtomicU64,
    epoch: Instant,
    panicked: AtomicU64,
    /// After `stop`, `defer` makes its call at once.
    stopped: AtomicBool,
    bounds: Bounds,
    pool: OnceLock<Arc<LanePool>>,
    downstream: DownstreamSlot,
    /// Frames the owner asked for and has not been given (`u64::MAX`: unbounded).
    demand: AtomicU64,
    /// A drain frame is out (pushed and not yet finished or dropped).
    frame_out: AtomicBool,
    me: std::sync::Weak<DeferredCalls>,
}

impl fmt::Debug for DeferredCalls {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DeferredCalls")
            .field("pending", &self.pending())
            .field("pending_bytes", &self.pending_bytes())
            .field("panicked", &self.panicked())
            .finish_non_exhaustive()
    }
}

/// The producer end: any thread queues a call through it. Cloning shares the same queue.
#[derive(Clone)]
pub struct Deferral(Arc<DeferredCalls>);

impl fmt::Debug for Deferral {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl Deferral {
    /// Queues `call` (`bytes` is the memory it gives back, 0 if none) and returns its ticket. Never waits. May push the
    /// drain frame (see [`DeferredCalls::poll`]).
    pub fn defer(&self, bytes: u64, call: impl FnOnce() + Send + 'static) -> Ticket {
        self.0.push(bytes, Box::new(call))
    }

    /// The ticket of the last call queued.
    pub fn last_ticket(&self) -> Ticket {
        self.0.last_ticket()
    }

    /// Makes the calls queued up to `ticket`; see [`DeferredCalls::run_through`].
    pub fn run_through(&self, ticket: Ticket) -> bool {
        self.0.run_through(ticket)
    }

    /// Pushes the drain frame if its conditions hold; see [`DeferredCalls::poll`].
    pub fn poll(&self) {
        self.0.poll();
    }

    /// How many calls wait.
    pub fn pending(&self) -> usize {
        self.0.pending()
    }

    /// The memory the waiting calls give back.
    pub fn pending_bytes(&self) -> u64 {
        self.0.pending_bytes()
    }

    /// For an allocation that failed for lack of memory: makes the calls queued so far, on this thread (waiting for a drain
    /// that is running), and says whether any were waiting, so the allocation is worth trying once more.
    pub fn reclaim(&self) -> bool {
        let ticket = self.last_ticket();
        ticket > Ticket::NONE
            && self.0.completed.load(Ordering::Acquire) < ticket.0
            && self.run_through(ticket)
    }
}

impl DeferredCalls {
    /// An empty source with `bounds`. Attach it to a lattice with `add_downstream`, give it frames with `request`, and the
    /// device's lane pool with [`attach_pool`](Self::attach_pool).
    pub fn new(bounds: Bounds) -> Arc<Self> {
        let queue = MpscQueue::with_pool(64, 4);
        let producer = queue.producer();
        let consumer = queue.consumer();
        Arc::new_cyclic(|me| DeferredCalls {
            _queue: queue,
            producer,
            drain: SourceCell::new(Drain { consumer, made: 0 }),
            queued: AtomicU64::new(0),
            completed: AtomicU64::new(0),
            bytes: AtomicU64::new(0),
            oldest: AtomicU64::new(0),
            epoch: Instant::now(),
            panicked: AtomicU64::new(0),
            stopped: AtomicBool::new(false),
            bounds,
            pool: OnceLock::new(),
            downstream: DownstreamSlot::new(),
            demand: AtomicU64::new(0),
            frame_out: AtomicBool::new(false),
            me: me.clone(),
        })
    }

    /// The producer handle.
    pub fn handle(self: &Arc<Self>) -> Deferral {
        Deferral(self.clone())
    }

    /// The lane pool whose idle moments release the calls (once; a second pool is refused and returned).
    pub fn attach_pool(&self, pool: Arc<LanePool>) -> Result<(), Arc<LanePool>> {
        self.pool.set(pool)
    }

    /// Ends deferral: from now on `defer` makes its call at once, on the calling thread. The owner's stop is `stop()`, then
    /// `drain()` once the last user of the device has let go; a call that raced `stop` and was queued after that drain is made
    /// when the source drops.
    pub fn stop(&self) {
        self.stopped.store(true, Ordering::Release);
    }

    /// How many calls wait.
    pub fn pending(&self) -> usize {
        let queued = self.queued.load(Ordering::Acquire);
        let made = self.completed.load(Ordering::Acquire);
        usize::try_from(queued.saturating_sub(made)).unwrap_or(usize::MAX)
    }

    /// The memory the waiting calls give back (what the callers said): memory the owner's books may count as certain
    /// reclaim.
    pub fn pending_bytes(&self) -> u64 {
        self.bytes.load(Ordering::Relaxed)
    }

    /// Calls that panicked (the rest of the queue still ran).
    pub fn panicked(&self) -> u64 {
        self.panicked.load(Ordering::Relaxed)
    }

    /// The ticket of the last call queued.
    pub fn last_ticket(&self) -> Ticket {
        Ticket(self.queued.load(Ordering::Acquire))
    }

    fn push(&self, bytes: u64, run: Box<dyn FnOnce() + Send>) -> Ticket {
        if self.stopped.load(Ordering::Acquire) {
            run_call(&self.panicked, run);
            return self.last_ticket();
        }
        self.bytes.fetch_add(bytes, Ordering::Relaxed);
        let now = self.epoch.elapsed().as_millis() as u64 + 1;
        let _ = self
            .oldest
            .compare_exchange(0, now, Ordering::Relaxed, Ordering::Relaxed);
        if let Err(call) = self.producer.offer(Call { bytes, run }) {
            // The queue never refuses; if it did, the call runs here rather than being lost.
            self.bytes.fetch_sub(bytes, Ordering::Relaxed);
            (call.run)();
            return self.last_ticket();
        }
        let ticket = Ticket(self.queued.fetch_add(1, Ordering::AcqRel) + 1);
        self.poll();
        ticket
    }

    /// Whether the oldest call has waited too long or the queued bytes passed their bound.
    fn forced(&self) -> bool {
        if self.pending_bytes() >= self.bounds.max_bytes {
            return true;
        }
        let oldest = self.oldest.load(Ordering::Relaxed);
        oldest != 0
            && (self.epoch.elapsed().as_millis() as u64 + 1).saturating_sub(oldest)
                >= self.bounds.max_age.as_millis() as u64
    }

    fn quiet(&self) -> bool {
        self.pool.get().is_some_and(|pool| pool.idle())
    }

    /// Pushes the drain frame when calls wait, no drain frame is out, a downstream asked for one, and the pool is idle or a
    /// bound is passed. Cheap when nothing waits (two atomic loads).
    pub fn poll(&self) {
        if self.pending() == 0 || self.frame_out.load(Ordering::Acquire) {
            return;
        }
        if self.forced() || self.quiet() {
            self.push_frame();
        }
    }

    fn push_frame(&self) {
        let Some(me) = self.me.upgrade() else { return };
        let Some(receiver) = self.downstream.receiver() else {
            return;
        };
        if self.downstream.is_complete() || self.frame_out.swap(true, Ordering::AcqRel) {
            return;
        }
        // One frame of the demand; `u64::MAX` is unbounded.
        let banked = self
            .demand
            .try_update(Ordering::AcqRel, Ordering::Acquire, |d| {
                if d == u64::MAX {
                    Some(d)
                } else {
                    d.checked_sub(1)
                }
            });
        if banked.is_err() {
            self.frame_out.store(false, Ordering::Release);
            return;
        }
        // The guard clears `frame_out` when the frame is dropped, run or not (a killed frame is dropped unrun).
        struct Out(Arc<DeferredCalls>);
        impl Drop for Out {
            fn drop(&mut self) {
                self.0.frame_out.store(false, Ordering::Release);
            }
        }
        let out = Out(me);
        let id =
            euhedral_hashing::hasher_api::mix_with_zero_check(Arc::as_ptr(&out.0) as usize as u64);
        let frame = RunnableFrame::new(
            id,
            Arc::new(move || {
                out.0.drain();
            }),
        );
        let mut frame = Box::new(frame) as Box<dyn Frame>;
        // A drain blocks: the mark a placement policy reads to keep such a frame off the workers that run short work.
        frame
            .header_mut()
            .set_routing_policy(RoutingPolicy::CacheLocal);
        receiver.push(frame);
    }

    /// Makes the calls that are queued, in order, on this thread, and returns how many it made. Returns 0 at once, having
    /// made none, when another thread (or this one, inside a call) is draining: the queue is then busy, not empty
    /// (`pending` says). Never waits.
    pub fn drain(&self) -> usize {
        let _cold = HotScope::leave();
        let Some(mut drain) = self.drain.try_enter() else {
            return 0;
        };
        let was = DRAINING.replace(true);
        let made = self.make_all(&mut drain);
        DRAINING.set(was);
        made
    }

    fn make_all(&self, drain: &mut Drain) -> usize {
        let mut ran = 0;
        while let Some(call) = drain.consumer.poll() {
            run_call(&self.panicked, call.run);
            drain.made += 1;
            ran += 1;
            self.bytes.fetch_sub(call.bytes, Ordering::Relaxed);
            self.completed.store(drain.made, Ordering::Release);
        }
        if self.pending() == 0 {
            self.oldest.store(0, Ordering::Relaxed);
        }
        ran
    }

    /// Makes the calls queued up to `ticket` on this thread, waiting for a drain another thread is running, and says whether
    /// they were made. `false` only when this thread is inside a call (it cannot wait for itself). For an allocation that
    /// ran out of memory: a build-time call, which may block.
    #[allow(
        clippy::disallowed_methods,
        reason = "LIFECYCLE(deferred-wait): a build-time call waits for the drain that holds the calls it needs"
    )]
    pub fn run_through(&self, ticket: Ticket) -> bool {
        loop {
            if self.completed.load(Ordering::Acquire) >= ticket.0 {
                return true;
            }
            if DRAINING.get() {
                return false;
            }
            if self.drain() == 0 {
                std::thread::yield_now();
            }
        }
    }
}

fn run_call(panicked: &AtomicU64, call: Box<dyn FnOnce() + Send>) {
    if catch_unwind(AssertUnwindSafe(call)).is_err() {
        panicked.fetch_add(1, Ordering::Relaxed);
    }
}

impl Drop for DeferredCalls {
    fn drop(&mut self) {
        let _cold = HotScope::leave();
        let drain = self.drain.get_mut();
        while let Some(call) = drain.consumer.poll() {
            run_call(&self.panicked, call.run);
        }
    }
}

impl LatticeSource for DeferredCalls {
    fn add_downstream(self: Arc<Self>, downstream: Arc<dyn LatticeReceiver>) {
        self.downstream.attach(self.clone(), downstream);
    }

    /// Hands out nothing: the calls leave as key frames, by [`request`](Self::request) and push.
    fn pull(
        &self,
        _consumer: &mut dyn FnMut(Box<dyn Frame>),
        _stop: &mut dyn FnMut(&dyn Frame) -> bool,
        _demand: u64,
    ) -> u64 {
        0
    }

    /// Banks `demand` frames (`u64::MAX`: unbounded) for the drains that [`poll`](Self::poll) pushes, and pushes one now if
    /// the conditions hold.
    fn request(&self, demand: u64) {
        if demand == 0 || self.downstream.is_complete() {
            return;
        }
        let _ = self
            .demand
            .try_update(Ordering::AcqRel, Ordering::Acquire, |d| {
                Some(d.saturating_add(demand))
            });
        self.poll();
    }

    /// Stops pushing frames and tells the downstream. The calls still queued stay for [`drain`](DeferredCalls::drain) and the
    /// drop.
    fn complete(&self) {
        self.downstream.complete();
    }

    fn is_complete(&self) -> bool {
        self.downstream.is_complete()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;
    use std::sync::atomic::AtomicUsize;

    use euhedral_core::generics::LatticeTerminal;
    use euhedral_graph::{DeviceQueue, QueueId};

    use super::*;

    fn calls() -> Arc<DeferredCalls> {
        DeferredCalls::new(Bounds::default())
    }

    type Log = Arc<Mutex<Vec<u32>>>;
    type Boxed = Box<dyn FnOnce() + Send>;

    fn recorder() -> (Log, impl Fn(u32) -> Boxed) {
        let log = Arc::new(Mutex::new(Vec::new()));
        let l = log.clone();
        (log, move |n| {
            let l = l.clone();
            Box::new(move || l.lock().unwrap().push(n))
        })
    }

    fn wait_for(mut done: impl FnMut() -> bool) {
        let end = Instant::now() + Duration::from_secs(10);
        while !done() {
            assert!(Instant::now() < end, "timed out");
            std::thread::yield_now();
        }
    }

    #[test]
    fn calls_are_made_in_the_order_they_were_queued_each_once() {
        let calls = calls();
        let d = calls.handle();
        let (log, call) = recorder();
        let tickets: Vec<Ticket> = (0..6).map(|i| d.defer(10, call(i))).collect();
        assert_eq!(tickets.windows(2).filter(|w| w[0] < w[1]).count(), 5);
        assert_eq!((calls.pending(), calls.pending_bytes()), (6, 60));
        assert_eq!(calls.drain(), 6);
        assert_eq!(*log.lock().unwrap(), vec![0, 1, 2, 3, 4, 5]);
        assert_eq!((calls.pending(), calls.pending_bytes()), (0, 0));
        assert_eq!(calls.drain(), 0, "nothing is made twice");
    }

    #[test]
    fn a_drain_in_progress_is_busy_not_empty_and_the_wait_ends_in_order() {
        let calls = calls();
        let d = calls.handle();
        let (log, call) = recorder();
        let started = Arc::new(AtomicBool::new(false));
        let release = Arc::new(AtomicBool::new(false));
        let (s, r, l) = (started.clone(), release.clone(), log.clone());
        d.defer(0, move || {
            s.store(true, Ordering::Release);
            wait_for(|| r.load(Ordering::Acquire));
            l.lock().unwrap().push(0);
        });
        let b = d.defer(0, call(1));
        std::thread::scope(|scope| {
            let first = scope.spawn({
                let calls = calls.clone();
                move || calls.drain()
            });
            wait_for(|| started.load(Ordering::Acquire));
            // The first call is running, the second waits behind it, and a third arrives meanwhile.
            let c = d.defer(0, call(2));
            assert_eq!(calls.pending(), 3);
            assert_eq!(calls.drain(), 0, "busy: this thread made none");
            assert_eq!(calls.pending(), 3, "and says so");
            let waiter = scope.spawn({
                let calls = calls.clone();
                move || calls.run_through(c)
            });
            std::thread::yield_now();
            assert!(
                log.lock().unwrap().is_empty(),
                "nothing ran out of order meanwhile"
            );
            release.store(true, Ordering::Release);
            assert!(waiter.join().unwrap());
            assert_eq!(
                first.join().unwrap(),
                3,
                "the drain that held the calls made them all"
            );
            assert!(calls.run_through(b), "an earlier ticket is already reached");
        });
        assert_eq!(*log.lock().unwrap(), vec![0, 1, 2]);
        assert_eq!(calls.pending(), 0);
    }

    #[test]
    fn a_panicking_call_loses_nothing_behind_it() {
        let calls = calls();
        let d = calls.handle();
        let (log, call) = recorder();
        d.defer(0, call(0));
        d.defer(0, || panic!("a call that fails"));
        d.defer(0, call(1));
        assert_eq!(calls.drain(), 3);
        assert_eq!(*log.lock().unwrap(), vec![0, 1]);
        assert_eq!((calls.panicked(), calls.pending()), (1, 0));
    }

    #[test]
    fn dropping_the_source_makes_the_calls_still_queued() {
        let calls = calls();
        let d = calls.handle();
        let (log, call) = recorder();
        d.defer(0, call(0));
        d.defer(0, call(1));
        drop(d);
        drop(calls);
        assert_eq!(*log.lock().unwrap(), vec![0, 1]);
    }

    #[test]
    fn after_stop_a_deferral_makes_its_call_at_once() {
        let calls = calls();
        let d = calls.handle();
        let (log, call) = recorder();
        d.defer(0, call(0));
        calls.stop();
        d.defer(0, call(1));
        assert_eq!(
            *log.lock().unwrap(),
            vec![1],
            "made at once; the earlier one still waits"
        );
        assert_eq!(calls.drain(), 1);
        assert_eq!(*log.lock().unwrap(), vec![1, 0]);
    }

    #[test]
    fn a_call_cannot_wait_for_itself() {
        let calls = calls();
        let d = calls.handle();
        let inner = calls.clone();
        let seen = Arc::new(Mutex::new(None));
        let s = seen.clone();
        let t = d.defer(0, move || {
            let later = inner.handle().defer(0, || {});
            *s.lock().unwrap() = Some((inner.run_through(later), inner.drain()));
        });
        assert!(calls.run_through(t));
        assert_eq!(*seen.lock().unwrap(), Some((false, 0)));
        assert_eq!(
            calls.pending(),
            0,
            "the call it queued was made by the same drain"
        );
    }

    #[test]
    fn reclaim_makes_what_waits_and_says_whether_anything_did() {
        let calls = calls();
        let d = calls.handle();
        assert!(!d.reclaim(), "nothing queued");
        let (log, call) = recorder();
        d.defer(0, call(7));
        assert!(d.reclaim());
        assert_eq!(*log.lock().unwrap(), vec![7]);
        assert!(!d.reclaim(), "and nothing waits now");
    }

    #[test]
    fn many_threads_queue_and_every_call_is_made_once() {
        let calls = calls();
        let made = Arc::new(AtomicUsize::new(0));
        std::thread::scope(|scope| {
            for _ in 0..8 {
                let d = calls.handle();
                let made = made.clone();
                scope.spawn(move || {
                    for _ in 0..200 {
                        let made = made.clone();
                        d.defer(1, move || {
                            made.fetch_add(1, Ordering::Relaxed);
                        });
                    }
                });
            }
        });
        assert_eq!(calls.pending(), 1600);
        calls.drain();
        assert_eq!(made.load(Ordering::Relaxed), 1600);
    }

    /// A receiver that keeps the frames pushed to it.
    #[derive(Default)]
    struct Frames(Mutex<Vec<Box<dyn Frame>>>);

    impl LatticeTerminal for Frames {
        fn add_upstream(&self, _upstream: Arc<dyn LatticeSource>) {}
    }

    impl LatticeReceiver for Frames {
        fn push(&self, frame: Box<dyn Frame>) {
            self.0.lock().unwrap().push(frame);
        }

        fn on_complete(&self) {}

        fn on_error(&self, _error: euhedral_core::frames::Failure) {}
    }

    /// A queue whose issued and completed values the test sets.
    #[derive(Debug)]
    struct Lane {
        id: QueueId,
        issued: AtomicU64,
        completed: AtomicU64,
    }

    impl DeviceQueue for Lane {
        fn id(&self) -> QueueId {
            self.id
        }

        fn completed(&self) -> u64 {
            self.completed.load(Ordering::Acquire)
        }

        fn issued(&self) -> u64 {
            self.issued.load(Ordering::Acquire)
        }

        fn name(&self) -> &str {
            "lane"
        }
    }

    fn run(frames: &Frames) -> usize {
        let taken: Vec<_> = frames.0.lock().unwrap().drain(..).collect();
        let n = taken.len();
        for mut frame in taken {
            frame.execute().unwrap();
            frame.do_finally();
        }
        n
    }

    #[test]
    fn a_frame_leaves_when_the_pool_is_idle_and_not_before_and_it_makes_the_calls() {
        let calls = DeferredCalls::new(Bounds::default());
        let frames = Arc::new(Frames::default());
        calls.clone().add_downstream(frames.clone());
        calls.request(u64::MAX);
        let lane = Arc::new(Lane {
            id: QueueId::next(),
            issued: 5.into(),
            completed: 4.into(),
        });
        calls
            .attach_pool(Arc::new(LanePool::new(vec![lane.clone()])))
            .unwrap();
        let (log, call) = recorder();
        let d = calls.handle();
        d.defer(0, call(0));
        calls.poll();
        assert_eq!(run(&frames), 0, "a lane has marked work in flight");
        lane.completed.store(5, Ordering::Release);
        calls.poll();
        calls.poll();
        assert_eq!(frames.0.lock().unwrap().len(), 1, "one frame at a time");
        d.defer(0, call(1));
        assert_eq!(run(&frames), 1);
        assert_eq!(
            *log.lock().unwrap(),
            vec![0, 1],
            "the frame made everything queued"
        );
        d.defer(0, call(2));
        assert_eq!(
            run(&frames),
            1,
            "the frame that finished let the next one leave"
        );
        assert_eq!(*log.lock().unwrap(), vec![0, 1, 2]);
    }

    #[test]
    fn a_bound_forces_a_frame_while_lanes_are_busy() {
        let calls = DeferredCalls::new(Bounds {
            max_bytes: 100,
            max_age: Duration::from_millis(50),
        });
        let frames = Arc::new(Frames::default());
        calls.clone().add_downstream(frames.clone());
        calls.request(u64::MAX);
        let busy = Arc::new(Lane {
            id: QueueId::next(),
            issued: 2.into(),
            completed: 0.into(),
        });
        calls
            .attach_pool(Arc::new(LanePool::new(vec![busy])))
            .unwrap();
        let d = calls.handle();
        d.defer(60, || {});
        calls.poll();
        assert_eq!(run(&frames), 0, "under both bounds");
        d.defer(60, || {});
        assert_eq!(run(&frames), 1, "the bytes bound");
        assert_eq!(calls.pending(), 0);
        d.defer(1, || {});
        wait_for(|| {
            calls.poll();
            !frames.0.lock().unwrap().is_empty()
        });
        assert_eq!(run(&frames), 1, "the age bound");
    }

    #[test]
    fn no_frame_leaves_without_a_downstream_or_demand_and_a_dropped_frame_lets_the_next_leave() {
        let calls = DeferredCalls::new(Bounds {
            max_bytes: 1,
            ..Bounds::default()
        });
        calls.handle().defer(5, || {});
        calls.poll();
        let frames = Arc::new(Frames::default());
        calls.clone().add_downstream(frames.clone());
        calls.poll();
        assert_eq!(frames.0.lock().unwrap().len(), 0, "no demand yet");
        calls.request(1);
        assert_eq!(frames.0.lock().unwrap().len(), 1);
        // The frame is dropped unrun (a killed frame): the queue still has its call and the demand is spent.
        frames.0.lock().unwrap().clear();
        calls.request(1);
        assert_eq!(
            frames.0.lock().unwrap().len(),
            1,
            "a dropped frame does not hold the slot"
        );
        assert_eq!(run(&frames), 1);
        assert_eq!(calls.pending(), 0);
        assert_eq!(
            calls.pull(&mut |_| {}, &mut |_| false, 10),
            0,
            "nothing leaves by pull"
        );
    }
}
