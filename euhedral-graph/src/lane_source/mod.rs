//! A device lane as one [`LatticeSource`](euhedral_core::generics::LatticeSource).
//!
//! A lane's submitter has one owner, so the lane runs as a source: the interceptor's claim lets one
//! worker at a time into the source's `request` and `pull`, and everything the lane keeps (its
//! submitter, its waiting items, its ready frames, its armed continuations) is plain state in a
//! `SourceCell`. Work reaches a lane through [`LaneIntakes`]: each lane has its own intake, and the
//! lanes of a group share one more. Delivering work only enqueues it. The lane runs it in its next
//! `request`, on whichever worker enters that source, and pushes the frames it made within the
//! demand it has banked.
//!
//! | Piece | What it is |
//! |---|---|
//! | [`LaneSubmit`] | the submitter a lane drives: markers, awaits and retirements, backend-neutral |
//! | [`LaneState`] | the lane's own hardware state, read once per request, and what an item can wait for |
//! | [`LaneWork`] | one item of work: it runs on the lane, delivers its successors, and ends on a retirement |
//! | [`LaneCx`] | what an item reaches while it runs |
//! | [`LaneIntakes`] | delivery into the lanes' own intakes and the groups' shared intakes |
//! | [`LaneSource`] | the lane as a source |
//! | [`LaneMetrics`] | service time, per-kind item time, intake-to-run latency and ready depth |

mod intake;
mod metrics;
mod source;

use std::collections::VecDeque;
use std::convert::Infallible;
use std::sync::Arc;
use std::time::Instant;

use euhedral_core::frames::Frame;

use crate::lanes::{DeviceQueue, LaneId, Marker, QueueId};

pub use intake::{Deliver, GroupId, LaneClosed, LaneInputs, LaneIntakes};
pub use metrics::LaneMetrics;
pub use source::LaneSource;

/// A retirement that arrived: the device finished the work submitted before it was armed.
#[derive(Debug)]
pub struct Retired<E> {
    /// The retirement's place in the order it was armed.
    pub seq: u64,
    /// The device failure, when the work did not complete.
    pub failure: Option<E>,
}

/// The submitter of one lane. One owner drives it, so its methods take `&mut self`.
#[allow(
    unsafe_code,
    reason = "`raw` lends the queue's handle; its contract is the safety section"
)]
pub trait LaneSubmit: Send + 'static {
    type Error: std::error::Error + Send + Sync + 'static;
    /// The queue's raw handle, lent to an item that enqueues work of its own.
    type Raw: Copy;

    fn lane(&self) -> LaneId;

    /// The identity stamped on this lane's markers.
    fn id(&self) -> QueueId;

    fn queue(&self) -> Arc<dyn DeviceQueue>;

    /// Whether [`await_marker`](Self::await_marker) can order this lane behind `marker` on the
    /// device: a lane of this backend in this pool.
    fn accepts(&self, marker: Marker) -> bool;

    /// Orders this lane's later work behind `marker` on the device.
    fn await_marker(&mut self, marker: Marker) -> Result<(), Self::Error>;

    /// A marker after the work submitted so far; `Ok(None)` when none can be made now (its ring is
    /// full). Never waits.
    fn mark(&mut self) -> Result<Option<Marker>, Self::Error>;

    /// Arms a retirement after the work submitted so far and returns its sequence number, which
    /// [`poll_retired`](Self::poll_retired) reports in [`Retired::seq`]. Retirements come back in
    /// the order they were armed; the lane source pairs each with its item by this number.
    fn arm_retirement(&mut self) -> Result<u64, Self::Error>;

    /// The next retirement that arrived, settled: the backend has released what it kept for it.
    fn poll_retired(&mut self) -> Option<Retired<Self::Error>>;

    /// The queue's raw handle.
    ///
    /// # Safety
    ///
    /// The caller enqueues on the handle only from inside an item of this lane, and a
    /// [`mark`](Self::mark) afterwards covers it.
    unsafe fn raw(&mut self) -> Self::Raw;
}

/// A lane's own hardware state, and the conditions an item can wait for.
pub trait LaneState: Send + 'static {
    /// What an item can wait for.
    type Wait: Send + 'static;

    /// Step 3 of each request: one read of each hardware state. Never a loop or a wait.
    fn check(&mut self, now: Instant) {
        let _ = now;
    }

    /// Whether `wait` is met. Asked once per request for each waiting head.
    fn met(&mut self, wait: &Self::Wait, now: Instant) -> bool;
}

/// A lane with no state of its own: nothing to wait for.
impl LaneState for () {
    type Wait = Infallible;

    fn met(&mut self, wait: &Self::Wait, _now: Instant) -> bool {
        match *wait {}
    }
}

/// What running an item decided.
#[derive(Debug)]
pub enum Outcome<A> {
    /// The item submitted work. The lane marks it; `retire` asks for a retirement to follow.
    Submitted { retire: bool },
    /// The item needed no device work.
    Finished,
    /// The item cannot go on until the lane's state meets `A`. It stays in the source, and items
    /// with its order key wait behind it.
    Wait(A),
}

/// One item of work on a lane.
pub trait LaneWork<S: LaneSubmit, X: LaneState>: Send + Sized + 'static {
    /// The names of the item kinds, for the metrics. [`kind`](Self::kind) indexes this list.
    const KINDS: &'static [&'static str];

    fn kind(&self) -> usize;

    /// The markers this item's work must follow. The lane orders itself behind the ones of its
    /// backend; the item bridges the others itself.
    fn waits(&self) -> &[Marker];

    /// Items of one order key run in delivery order, even across a wait.
    fn order(&self) -> Option<u64> {
        None
    }

    /// Runs the item on its lane.
    fn run(&mut self, cx: &mut LaneCx<'_, S, X, Self>) -> Outcome<X::Wait>;

    /// After the lane marked the work (`Some`) or the item finished with no device work (`None`):
    /// deliver the successors and queue the frames.
    fn then(&mut self, marker: Option<Marker>, cx: &mut LaneCx<'_, S, X, Self>) {
        let _ = (marker, cx);
    }

    /// The retirement armed for the item arrived.
    fn retired(self, marker: Marker, failure: Option<S::Error>, cx: &mut LaneCx<'_, S, X, Self>) {
        let _ = (marker, failure, cx);
    }

    /// The lane could not order, mark or arm the item, or its retirement failed to arm.
    fn failed(self, error: S::Error, cx: &mut LaneCx<'_, S, X, Self>);
}

/// What an item reaches while it runs on its lane.
pub struct LaneCx<'a, S, X, W> {
    pub(crate) submitter: &'a mut S,
    pub(crate) state: &'a mut X,
    pub(crate) ready: &'a mut VecDeque<Box<dyn Frame>>,
    pub(crate) intakes: &'a Arc<LaneIntakes<W>>,
    pub(crate) now: Instant,
    pub(crate) lane: LaneId,
}

impl<S, X, W> std::fmt::Debug for LaneCx<'_, S, X, W> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LaneCx").field("lane", &self.lane).finish()
    }
}

impl<S: LaneSubmit, X, W: Send + 'static> LaneCx<'_, S, X, W> {
    pub fn lane(&self) -> LaneId {
        self.lane
    }

    pub fn submitter(&mut self) -> &mut S {
        self.submitter
    }

    pub fn state(&mut self) -> &mut X {
        self.state
    }

    /// The submitter and the state together.
    pub fn parts(&mut self) -> (&mut S, &mut X) {
        (&mut *self.submitter, &mut *self.state)
    }

    /// The entry's one clock read.
    pub fn now(&self) -> Instant {
        self.now
    }

    /// True for a marker the lane did not order itself behind: one of another backend, for the item
    /// to bridge.
    pub fn bridges(&self, marker: Marker) -> bool {
        marker.queue != self.submitter.id() && !self.submitter.accepts(marker)
    }

    /// Delivers `work` to a lane or a group. Delivery only enqueues: it spends no demand and runs
    /// nothing. A lane that completed or was dropped refuses the work and hands it back.
    pub fn deliver(&self, work: W, to: Deliver) -> Result<(), LaneClosed<W>> {
        self.intakes.deliver(work, to)
    }

    /// Queues `frame`; the lane pushes it within the demand it has banked.
    pub fn ready(&mut self, frame: Box<dyn Frame>) {
        self.ready.push_back(frame);
    }

    pub fn intakes(&self) -> &Arc<LaneIntakes<W>> {
        self.intakes
    }
}
