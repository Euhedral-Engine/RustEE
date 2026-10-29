//! [`LaneSource`]: a device lane as a lattice source.

// `SourceCell::enter` is `unsafe` because the compiler cannot see the lattice's single-entry
// guarantee, and the item queue's drop order needs a `ManuallyDrop`; each use says why it holds.
#![allow(
    unsafe_code,
    reason = "the lane's state is a SourceCell, entered under the lattice's single-entry guarantee"
)]

use std::collections::VecDeque;
use std::mem::ManuallyDrop;
use std::panic::{self, AssertUnwindSafe};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

use euhedral_core::frames::{Frame, FrameError};
use euhedral_core::generics::{DownstreamSlot, LatticeReceiver, LatticeSource, SourceCell};
use euhedral_data_structures::atomics::CachePadded;
use euhedral_data_structures::queues::{Consumer, Multi, Single};

use super::intake::Envelope;
use super::{
    LaneCx, LaneInputs, LaneIntakes, LaneMetrics, LaneState, LaneSubmit, LaneWork, Outcome,
};
use crate::lanes::{LaneId, Marker, QueueId};

/// An item the source holds back: it waits for the lane's state, or behind an earlier item of its
/// order key.
struct Waiting<W, A> {
    work: W,
    /// `None` for an item parked behind its key only.
    wait: Option<A>,
    /// Its input markers were awaited (it ran before).
    awaited: bool,
    delivered: Instant,
}

/// What running one item left to its caller.
enum Flow<W, A> {
    /// The item is finished with, or armed for its retirement.
    Done,
    /// The item could not be marked: the lane stalls.
    Stalled,
    /// The item waits.
    Waits(Waiting<W, A>),
}

/// Everything a worker touches only while it is inside `request` or `pull`.
struct Inner<S: LaneSubmit, X: LaneState, W> {
    /// Dropped first (see the `Drop`): its shutdown waits for the device work the items still hold.
    submitter: ManuallyDrop<S>,
    queue_id: QueueId,
    state: X,
    own: Consumer<Envelope<W>, Single>,
    shared: Option<Consumer<Envelope<W>, Multi>>,
    /// Frames made by items, in the order they were made.
    ready: VecDeque<Box<dyn Frame>>,
    waiting: VecDeque<Waiting<W, X::Wait>>,
    /// The waiting pass's scratch, swapped with `waiting`; it keeps its capacity.
    keep: VecDeque<Waiting<W, X::Wait>>,
    /// Items with a retirement armed, in arm order, with the sequence number the submitter gave it.
    /// A retirement is paired with its item by that number, never by position: a retirement armed
    /// from inside an item is not the lane's.
    armed: VecDeque<(u64, W, Marker)>,
    /// An item whose work was submitted but could not be marked (the lane's ring was full), and
    /// whether it asked for a retirement.
    unmarked: Option<(W, bool)>,
    /// Order keys held back in this request (reused: no allocation after warm-up).
    blocked_keys: Vec<u64>,
}

impl<S: LaneSubmit, X: LaneState, W> Drop for Inner<S, X, W> {
    fn drop(&mut self) {
        // The submitter goes first: its shutdown waits for the device work the items still hold.
        // If that panics, everything else is still released before the panic goes on.
        // SAFETY: `submitter` is dropped here only, and no field is used again.
        let shutdown = panic::catch_unwind(AssertUnwindSafe(|| unsafe {
            ManuallyDrop::drop(&mut self.submitter);
        }));
        self.armed.clear();
        self.unmarked = None;
        self.waiting.clear();
        self.keep.clear();
        // Work still in the lane's own intake: the lane is closed, so nothing more arrives.
        while self.own.poll().is_some() {}
        while let Some(frame) = self.ready.pop_front() {
            frame.do_finally_with_error(FrameError::failed("the lane source was dropped"));
        }
        if let Err(payload) = shutdown
            && !std::thread::panicking()
        {
            panic::resume_unwind(payload);
        }
    }
}

/// A device lane as a lattice source.
///
/// Each [`request`](LatticeSource::request) banks the demand, drains the lane's retirements, reads
/// the lane's hardware state once, runs every item that is ready (the released waiting items, then
/// the lane's own intake, then its group's shared intake) and pushes at most the banked demand of
/// ready frames. [`pull`](LatticeSource::pull) hands over ready frames, within its own demand, and
/// does nothing else.
///
/// Everything the lane keeps is plain state in a `SourceCell`, entered inside `request` and `pull`,
/// which the lattice serializes. The cell's claim is checked in every build: a `request` that finds
/// it taken (a direct call racing the lattice, or a call from inside an item) has banked its demand
/// and left a wakeup, and the holder serves it when it lets go; a `pull` that finds it taken hands
/// over nothing. The lifecycle (`complete`) touches atomics only.
///
/// Work an item answers `Wait` for stays in this source, even when it came from the group's shared
/// intake: it runs when this lane's state meets the wait.
pub struct LaneSource<S: LaneSubmit, X: LaneState, W> {
    lane: LaneId,
    complete: AtomicBool,
    /// The demand banked by `request` and not yet spent on a push.
    demand: CachePadded<AtomicU64>,
    downstream: DownstreamSlot,
    /// The depth of `ready`, published at the end of each call.
    ready_len: AtomicUsize,
    /// A `request` happened that no pass has served yet: a wakeup, as the ingest sinks'
    /// `requestPending`. A caller that cannot enter the cell leaves it set for whoever is inside.
    pending: AtomicBool,
    intakes: Arc<LaneIntakes<W>>,
    metrics: LaneMetrics,
    cell: SourceCell<Inner<S, X, W>>,
}

impl<S: LaneSubmit, X: LaneState, W: LaneWork<S, X>> LaneSource<S, X, W> {
    /// The source of `submitter`'s lane, fed by `inputs`.
    ///
    /// # Panics
    ///
    /// If `inputs` belongs to another lane than the submitter's. In debug builds also if `intakes`
    /// is not the set `inputs` came from, or the pool's queue at the lane is not the submitter's.
    pub fn new(
        submitter: S,
        state: X,
        inputs: LaneInputs<W>,
        intakes: Arc<LaneIntakes<W>>,
    ) -> Arc<Self> {
        let lane = submitter.lane();
        assert_eq!(
            inputs.lane(),
            lane,
            "a lane source takes the inputs of its own lane"
        );
        let queue_id = submitter.id();
        debug_assert_eq!(
            inputs.intakes_id(),
            intakes.id(),
            "a lane source takes the inputs of the intakes it delivers through"
        );
        debug_assert_eq!(
            intakes.pool().lane(lane).id(),
            queue_id,
            "the pool's queue at the lane is the submitter's"
        );
        Arc::new(Self {
            lane,
            complete: AtomicBool::new(false),
            demand: CachePadded::new(AtomicU64::new(0)),
            downstream: DownstreamSlot::new(),
            ready_len: AtomicUsize::new(0),
            pending: AtomicBool::new(false),
            intakes,
            metrics: LaneMetrics::new(W::KINDS.len()),
            cell: SourceCell::new(Inner {
                submitter: ManuallyDrop::new(submitter),
                queue_id,
                state,
                own: inputs.own,
                shared: inputs.shared,
                ready: VecDeque::new(),
                waiting: VecDeque::new(),
                keep: VecDeque::new(),
                armed: VecDeque::new(),
                unmarked: None,
                blocked_keys: Vec::new(),
            }),
        })
    }

    pub fn lane(&self) -> LaneId {
        self.lane
    }

    /// The demand banked and not yet spent.
    pub fn banked(&self) -> u64 {
        self.demand.load(Ordering::Acquire)
    }

    /// The frames ready to push or pull, as of the end of the last call.
    pub fn ready_len(&self) -> usize {
        self.ready_len.load(Ordering::Relaxed)
    }

    pub fn metrics(&self) -> &LaneMetrics {
        &self.metrics
    }

    /// Runs `work` on the lane, once its input markers are ordered.
    fn run_one(
        &self,
        inner: &mut Inner<S, X, W>,
        now: Instant,
        clock: &mut Instant,
        mut item: Waiting<W, X::Wait>,
    ) -> Flow<W, X::Wait> {
        if !item.awaited {
            item.awaited = true;
            let mut refused = None;
            for &marker in item.work.waits() {
                if marker.queue == inner.queue_id || !inner.submitter.accepts(marker) {
                    continue;
                }
                if let Err(error) = inner.submitter.await_marker(marker) {
                    refused = Some(error);
                    break;
                }
            }
            if let Some(error) = refused {
                self.fail(inner, now, item.work, error);
                return Flow::Done;
            }
            self.metrics
                .intake_to_run_ns
                .record(ns(clock.saturating_duration_since(item.delivered)));
        }
        item.wait = None;
        // One clock read per item: the end of the previous item is the start of this one.
        let started = *clock;
        let outcome = {
            let mut cx = self.cx(inner, now);
            item.work.run(&mut cx)
        };
        *clock = Instant::now();
        if let Some(histogram) = self.metrics.item_ns.get(item.work.kind()) {
            histogram.record(ns(clock.saturating_duration_since(started)));
        }
        match outcome {
            Outcome::Submitted { retire } => match inner.submitter.mark() {
                Ok(Some(marker)) => {
                    self.marked(inner, now, item.work, marker, retire);
                    Flow::Done
                }
                Ok(None) => {
                    inner.unmarked = Some((item.work, retire));
                    Flow::Stalled
                }
                Err(error) => {
                    self.fail(inner, now, item.work, error);
                    Flow::Done
                }
            },
            Outcome::Finished => {
                let mut cx = self.cx(inner, now);
                item.work.then(None, &mut cx);
                Flow::Done
            }
            Outcome::Wait(wait) => {
                item.wait = Some(wait);
                Flow::Waits(item)
            }
            Outcome::Failed(error) => {
                self.fail(inner, now, item.work, error);
                Flow::Done
            }
        }
    }

    fn cx<'a>(&'a self, inner: &'a mut Inner<S, X, W>, now: Instant) -> LaneCx<'a, S, X, W> {
        LaneCx {
            submitter: &mut *inner.submitter,
            state: &mut inner.state,
            ready: &mut inner.ready,
            intakes: &self.intakes,
            now,
            lane: self.lane,
        }
    }

    /// The work was marked: the item delivers what follows, and a retirement is armed if asked.
    fn marked(
        &self,
        inner: &mut Inner<S, X, W>,
        now: Instant,
        mut work: W,
        marker: Marker,
        retire: bool,
    ) {
        {
            let mut cx = self.cx(inner, now);
            work.then(Some(marker), &mut cx);
        }
        if !retire {
            return;
        }
        match inner.submitter.arm_retirement() {
            Ok(seq) => inner.armed.push_back((seq, work, marker)),
            Err(error) => self.fail(inner, now, work, error),
        }
    }

    fn fail(&self, inner: &mut Inner<S, X, W>, now: Instant, work: W, error: S::Error) {
        let mut cx = self.cx(inner, now);
        work.failed(error, &mut cx);
    }

    /// Step 1: the retirements that arrived. Each is paired with the item that armed it by its
    /// sequence number. One the lane did not arm (an item armed it itself) pairs with nothing.
    fn drain_retirements(&self, inner: &mut Inner<S, X, W>, now: Instant) {
        while let Some(retired) = inner.submitter.poll_retired() {
            // The match is at the front unless an earlier retirement was taken from the submitter
            // by something else. Retirements arrive in arm order, so those entries can never
            // arrive: drop them, and the front of `armed` is the next to arrive again. An arrival
            // with no entry is not the lane's (an item armed it itself) and pairs with nothing.
            let Some(position) = inner.armed.iter().position(|armed| armed.0 == retired.seq) else {
                continue;
            };
            if position > 0 {
                // Entries that can never arrive: an item took their retirements from the submitter.
                self.metrics
                    .stranded
                    .fetch_add(position as u64, Ordering::Relaxed);
                inner.armed.drain(..position);
            }
            let (_, work, marker) = inner.armed.pop_front().expect("the match is at the front");
            let mut cx = self.cx(inner, now);
            work.retired(marker, retired.failure, &mut cx);
        }
    }

    /// Step 3: marks the item that could not be marked before. Returns whether the lane is still
    /// stalled.
    fn retry_mark(&self, inner: &mut Inner<S, X, W>, now: Instant) -> bool {
        let Some((work, retire)) = inner.unmarked.take() else {
            return false;
        };
        match inner.submitter.mark() {
            Ok(Some(marker)) => {
                self.marked(inner, now, work, marker, retire);
                false
            }
            Ok(None) => {
                inner.unmarked = Some((work, retire));
                true
            }
            Err(error) => {
                self.fail(inner, now, work, error);
                false
            }
        }
    }

    /// Step 4: the waiting items, in order. An item behind a held-back key stays; an item whose
    /// wait is unmet stays and holds back its key.
    fn run_waiting(&self, inner: &mut Inner<S, X, W>, now: Instant, clock: &mut Instant) -> bool {
        let mut stalled = false;
        debug_assert!(inner.keep.is_empty());
        while let Some(item) = inner.waiting.pop_front() {
            let key = item.work.order();
            if stalled || key.is_some_and(|k| inner.blocked_keys.contains(&k)) {
                inner.keep.push_back(item);
                continue;
            }
            if let Some(wait) = &item.wait
                && !inner.state.met(wait, now)
            {
                if let Some(k) = key {
                    inner.blocked_keys.push(k);
                }
                inner.keep.push_back(item);
                continue;
            }
            match self.run_one(inner, now, clock, item) {
                Flow::Done => {}
                Flow::Stalled => stalled = true,
                Flow::Waits(item) => {
                    if let Some(k) = item.work.order() {
                        inner.blocked_keys.push(k);
                    }
                    inner.keep.push_back(item);
                }
            }
        }
        std::mem::swap(&mut inner.waiting, &mut inner.keep);
        stalled
    }

    /// Runs, or parks behind its key, an envelope just taken from an intake. Returns whether the
    /// lane stalled.
    fn admit(
        &self,
        inner: &mut Inner<S, X, W>,
        now: Instant,
        clock: &mut Instant,
        envelope: Envelope<W>,
    ) -> bool {
        let item = Waiting {
            work: envelope.work,
            wait: None,
            awaited: false,
            delivered: envelope.delivered,
        };
        if item
            .work
            .order()
            .is_some_and(|k| inner.blocked_keys.contains(&k))
        {
            inner.waiting.push_back(item);
            return false;
        }
        match self.run_one(inner, now, clock, item) {
            Flow::Done => false,
            Flow::Stalled => true,
            Flow::Waits(item) => {
                if let Some(k) = item.work.order() {
                    inner.blocked_keys.push(k);
                }
                inner.waiting.push_back(item);
                false
            }
        }
    }

    /// Serves the requests that arrived, one pass per claim. A caller that cannot take the claim
    /// leaves `pending` set for the holder, which looks again after it lets go.
    fn drain_requests(&self) {
        while self.pending.load(Ordering::SeqCst) {
            let Some(mut guard) = self.cell.try_enter() else {
                return;
            };
            // One swap reads and clears the flag, so a request after it keeps the flag set.
            if self.pending.swap(false, Ordering::SeqCst) {
                self.serve(&mut guard);
            }
            drop(guard);
        }
    }

    /// One pass of `request`, with the claim held.
    fn serve(&self, inner: &mut Inner<S, X, W>) {
        if self.is_complete() || self.downstream.receiver().is_none() {
            return;
        }
        let started = Instant::now();
        let now = started;
        let mut clock = started;
        self.drain_retirements(inner, now);
        inner.state.check(now);
        inner.blocked_keys.clear();
        let mut stalled = self.retry_mark(inner, now);
        if !stalled {
            stalled = self.run_waiting(inner, now, &mut clock);
        }
        while !stalled {
            let Some(envelope) = inner.own.poll() else {
                break;
            };
            stalled = self.admit(inner, now, &mut clock, envelope);
        }
        while !stalled {
            let Some(envelope) = inner.shared.as_ref().and_then(|shared| shared.poll()) else {
                break;
            };
            debug_assert!(
                envelope.work.order().is_none(),
                "work in a shared intake is not bound to a lane"
            );
            stalled = self.admit(inner, now, &mut clock, envelope);
        }
        // Push at most the banked demand of ready frames.
        if let Some(receiver) = self.downstream.receiver() {
            let banked = self.demand.load(Ordering::Acquire);
            let n = banked.min(inner.ready.len() as u64);
            for _ in 0..n {
                let frame = inner
                    .ready
                    .pop_front()
                    .expect("n is at most the queue's length");
                receiver.push(frame);
            }
            if n > 0 {
                let _ = self
                    .demand
                    .try_update(Ordering::SeqCst, Ordering::SeqCst, |current| {
                        Some(spend(current, n))
                    });
            }
        }
        self.metrics.ready_depth.record(inner.ready.len() as f64);
        self.publish(inner);
        self.metrics.service_ns.record(ns(started.elapsed()));
    }

    /// Publishes the ready depth for [`ready_len`](Self::ready_len).
    fn publish(&self, inner: &Inner<S, X, W>) {
        self.ready_len.store(inner.ready.len(), Ordering::Relaxed);
    }
}

fn ns(d: std::time::Duration) -> f64 {
    d.as_nanos() as f64
}

/// Adds banked demand: saturating, with `u64::MAX` meaning unbounded.
fn bank(current: u64, add: u64) -> u64 {
    current.saturating_add(add)
}

/// Spends banked demand: floors at zero.
fn spend(current: u64, spent: u64) -> u64 {
    current.saturating_sub(spent)
}

impl<S: LaneSubmit, X: LaneState, W> Drop for LaneSource<S, X, W> {
    fn drop(&mut self) {
        // Refuse deliveries first; the cell's drop then ends what the lane still holds.
        self.intakes.close(self.lane);
    }
}

impl<S: LaneSubmit, X: LaneState, W> std::fmt::Debug for LaneSource<S, X, W> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LaneSource")
            .field("lane", &self.lane)
            .field("banked", &self.demand.load(Ordering::Relaxed))
            .field("ready", &self.ready_len.load(Ordering::Relaxed))
            .field("complete", &self.complete.load(Ordering::Relaxed))
            .finish()
    }
}

impl<S: LaneSubmit, X: LaneState, W: LaneWork<S, X>> LatticeSource for LaneSource<S, X, W> {
    fn add_downstream(self: Arc<Self>, downstream: Arc<dyn LatticeReceiver>) {
        let source: Arc<dyn LatticeSource> = self.clone();
        self.downstream.attach(source, downstream);
    }

    fn pull(
        &self,
        consumer: &mut dyn FnMut(Box<dyn Frame>),
        stop: &mut dyn FnMut(&dyn Frame) -> bool,
        demand: u64,
    ) -> u64 {
        if self.is_complete() || demand == 0 || self.downstream.receiver().is_none() {
            return 0;
        }
        let started = Instant::now();
        // The lattice serializes `pull` and `request`, so the claim is free. A caller that finds it
        // taken (a direct call racing the lattice, or a call from inside an item) hands over
        // nothing, as the ingest sinks' `pull` does when their drain is taken.
        let Some(mut inner) = self.cell.try_enter() else {
            return 0;
        };
        let mut pulled = 0;
        while pulled < demand {
            let Some(front) = inner.ready.front() else {
                break;
            };
            if stop(&**front) {
                break;
            }
            let frame = inner.ready.pop_front().expect("the front was just seen");
            consumer(frame);
            pulled += 1;
        }
        self.publish(&inner);
        drop(inner);
        self.metrics.service_ns.record(ns(started.elapsed()));
        // A request made while this pull held the claim (from the consumer) left a wakeup: serve it
        // now, as the ingest sinks' pull does in its `finally`.
        self.drain_requests();
        pulled
    }

    fn request(&self, demand: u64) {
        if self.is_complete() || demand == 0 || self.downstream.receiver().is_none() {
            return;
        }
        let _ = self
            .demand
            .try_update(Ordering::SeqCst, Ordering::SeqCst, |current| {
                Some(bank(current, demand))
            });
        // The wakeup is published before the claim is tried: the holder either sees it when it
        // lets go, or this caller gets the claim itself. Nothing waits, and the demand is banked
        // either way.
        self.pending.store(true, Ordering::SeqCst);
        self.drain_requests();
    }

    fn complete(&self) {
        if self
            .complete
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return;
        }
        self.demand.store(0, Ordering::Release);
        self.intakes.close(self.lane);
        self.downstream.complete();
    }

    fn is_complete(&self) -> bool {
        self.complete.load(Ordering::Acquire)
    }
}
