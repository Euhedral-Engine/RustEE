//! A stepped fake [`LaneSubmit`](crate::lane_source::LaneSubmit): no wall clock, no device. A test
//! decides when the queue's work completes and when a retirement arrives.

#![allow(unsafe_code, reason = "it implements the unsafe `raw` of `LaneSubmit`")]
#![allow(
    clippy::disallowed_types,
    reason = "a test double: its records are locked by the test thread and the code under test"
)]

use std::collections::VecDeque;
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use crate::lane_source::{LaneSubmit, Retired};
use crate::lanes::{DeviceQueue, LaneId, Marker, QueueId};

/// What a fake lane was asked to do, in order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FakeOp {
    /// The lane was ordered behind a marker.
    Await(Marker),
    /// A marker was made.
    Mark(Marker),
    /// A retirement with this sequence number was armed.
    Arm(u64),
    /// The raw handle was lent.
    Raw,
}

/// The error a fake lane reports.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FakeLaneError {
    /// An `await_marker` the test made fail.
    AwaitFailed,
    /// A retirement the test made fail.
    RetirementFailed,
    /// A `mark` the test made fail.
    MarkFailed,
    /// An `arm_retirement` the test made fail.
    ArmFailed,
    /// A marker of a queue this backend does not own.
    Refused,
}

impl fmt::Display for FakeLaneError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::AwaitFailed => "the fake lane's await failed",
            Self::RetirementFailed => "the fake lane's retirement failed",
            Self::MarkFailed => "the fake lane's mark failed",
            Self::ArmFailed => "the fake lane's arm failed",
            Self::Refused => "the fake lane refuses a marker of another backend",
        })
    }
}

impl std::error::Error for FakeLaneError {}

#[derive(Default)]
struct LaneInner {
    marks: u64,
    busy: usize,
    fail_awaits: usize,
    fail_marks: usize,
    fail_arms: usize,
    fail_arrivals: bool,
    next_seq: u64,
    armed: VecDeque<u64>,
    arrived: VecDeque<Retired<FakeLaneError>>,
    raws: u64,
    log: Vec<FakeOp>,
}

struct Shared {
    first: u16,
    ids: Vec<QueueId>,
    completed: Vec<AtomicU64>,
    /// One lock per lane: lanes do not contend with each other.
    inner: Vec<Mutex<LaneInner>>,
    /// Set when the lane's submitter is dropped.
    dropped: Vec<AtomicBool>,
    /// The submitter's drop panics.
    panic_on_drop: Vec<AtomicBool>,
    quiet: AtomicBool,
}

impl Shared {
    fn lock(&self, index: usize) -> MutexGuard<'_, LaneInner> {
        self.inner[index]
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }

    fn index(&self, lane: LaneId) -> usize {
        usize::from(lane.0)
            .checked_sub(usize::from(self.first))
            .filter(|&i| i < self.ids.len())
            .unwrap_or_else(|| panic!("lane {} is not one of this fake's", lane.0))
    }

    fn record(&self, lane: &mut LaneInner, op: FakeOp) {
        if !self.quiet.load(Ordering::Relaxed) {
            lane.log.push(op);
        }
    }
}

/// The test's side of a set of fake lanes.
#[derive(Clone)]
pub struct FakeLanes {
    shared: Arc<Shared>,
}

impl fmt::Debug for FakeLanes {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FakeLanes")
            .field("first", &self.shared.first)
            .field("lanes", &self.shared.ids.len())
            .finish()
    }
}

/// One fake lane's submitter.
pub struct FakeLane {
    shared: Arc<Shared>,
    index: usize,
    lane: LaneId,
}

impl fmt::Debug for FakeLane {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FakeLane")
            .field("lane", &self.lane)
            .finish()
    }
}

/// A fake lane's queue: it reports the value the test completed.
struct FakeQueue {
    shared: Arc<Shared>,
    index: usize,
}

impl fmt::Debug for FakeQueue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FakeQueue")
            .field("index", &self.index)
            .finish()
    }
}

impl DeviceQueue for FakeQueue {
    fn id(&self) -> QueueId {
        self.shared.ids[self.index]
    }

    fn completed(&self) -> u64 {
        self.shared.completed[self.index].load(Ordering::Acquire)
    }

    fn name(&self) -> &str {
        "fake-lane"
    }
}

impl FakeLanes {
    /// `count` lanes at ids `first..first + count` (reserved from a `LanePoolBuilder`), with their
    /// queues to `set` into the builder.
    pub fn open(
        first: LaneId,
        count: usize,
    ) -> (FakeLanes, Vec<FakeLane>, Vec<Arc<dyn DeviceQueue>>) {
        let shared = Arc::new(Shared {
            first: first.0,
            ids: (0..count).map(|_| QueueId::next()).collect(),
            completed: (0..count).map(|_| AtomicU64::new(0)).collect(),
            inner: (0..count).map(|_| Mutex::default()).collect(),
            dropped: (0..count).map(|_| AtomicBool::new(false)).collect(),
            panic_on_drop: (0..count).map(|_| AtomicBool::new(false)).collect(),
            quiet: AtomicBool::new(false),
        });
        let lanes = (0..count)
            .map(|index| FakeLane {
                shared: shared.clone(),
                index,
                lane: LaneId(first.0 + u16::try_from(index).expect("at most 64 lanes")),
            })
            .collect();
        let queues = (0..count)
            .map(|index| {
                Arc::new(FakeQueue {
                    shared: shared.clone(),
                    index,
                }) as Arc<dyn DeviceQueue>
            })
            .collect();
        (FakeLanes { shared }, lanes, queues)
    }

    /// The queue's completed value becomes `value`.
    pub fn complete_through(&self, lane: LaneId, value: u64) {
        let i = self.shared.index(lane);
        self.shared.completed[i].fetch_max(value, Ordering::AcqRel);
    }

    /// Every armed retirement arrives, ok (or failed, after [`fail_retirements`](Self::fail_retirements)).
    pub fn retire_all(&self) {
        for index in 0..self.shared.ids.len() {
            let mut lane = self.shared.lock(index);
            let failing = std::mem::take(&mut lane.fail_arrivals);
            while let Some(seq) = lane.armed.pop_front() {
                lane.arrived.push_back(Retired {
                    seq,
                    failure: failing.then_some(FakeLaneError::RetirementFailed),
                });
            }
        }
    }

    /// The next retirements to arrive on `lane` carry a failure.
    pub fn fail_retirements(&self, lane: LaneId) {
        let i = self.shared.index(lane);
        self.shared.lock(i).fail_arrivals = true;
    }

    /// The next `n` marks of `lane` return `Ok(None)`: its marker ring is full.
    pub fn busy_marks(&self, lane: LaneId, n: usize) {
        let i = self.shared.index(lane);
        self.shared.lock(i).busy = n;
    }

    /// The next `n` marks of `lane` fail.
    pub fn fail_marks(&self, lane: LaneId, n: usize) {
        let i = self.shared.index(lane);
        self.shared.lock(i).fail_marks = n;
    }

    /// The next `n` retirement arms of `lane` fail.
    pub fn fail_arms(&self, lane: LaneId, n: usize) {
        let i = self.shared.index(lane);
        self.shared.lock(i).fail_arms = n;
    }

    /// Only the first `n` armed retirements of `lane` arrive (ok, or failed after
    /// [`fail_retirements`](Self::fail_retirements)).
    pub fn retire_next(&self, lane: LaneId, n: usize) {
        let i = self.shared.index(lane);
        let mut lane = self.shared.lock(i);
        let failing = std::mem::take(&mut lane.fail_arrivals);
        for _ in 0..n {
            let Some(seq) = lane.armed.pop_front() else {
                break;
            };
            lane.arrived.push_back(Retired {
                seq,
                failure: failing.then_some(FakeLaneError::RetirementFailed),
            });
        }
    }

    /// Whether `lane`'s submitter has been dropped.
    pub fn dropped(&self, lane: LaneId) -> bool {
        self.shared.dropped[self.shared.index(lane)].load(Ordering::SeqCst)
    }

    /// The drop of `lane`'s submitter panics (after it is recorded as dropped).
    pub fn panic_on_drop(&self, lane: LaneId) {
        self.shared.panic_on_drop[self.shared.index(lane)].store(true, Ordering::SeqCst);
    }

    /// The next `n` awaits of `lane` fail.
    pub fn fail_awaits(&self, lane: LaneId, n: usize) {
        let i = self.shared.index(lane);
        self.shared.lock(i).fail_awaits = n;
    }

    /// What `lane` was asked to do. Empty after [`quiet`](Self::quiet).
    pub fn log(&self, lane: LaneId) -> Vec<FakeOp> {
        let i = self.shared.index(lane);
        self.shared.lock(i).log.clone()
    }

    /// Armed retirements that have not arrived yet.
    pub fn armed(&self, lane: LaneId) -> usize {
        let i = self.shared.index(lane);
        self.shared.lock(i).armed.len()
    }

    /// Stops logging: no allocation per operation once the lanes are warm.
    pub fn quiet(self) -> Self {
        self.shared.quiet.store(true, Ordering::Relaxed);
        self
    }
}

impl LaneSubmit for FakeLane {
    type Error = FakeLaneError;
    type Raw = u64;

    fn lane(&self) -> LaneId {
        self.lane
    }

    fn id(&self) -> QueueId {
        self.shared.ids[self.index]
    }

    fn queue(&self) -> Arc<dyn DeviceQueue> {
        Arc::new(FakeQueue {
            shared: self.shared.clone(),
            index: self.index,
        })
    }

    fn accepts(&self, marker: Marker) -> bool {
        self.shared.ids.contains(&marker.queue)
    }

    fn await_marker(&mut self, marker: Marker) -> Result<(), FakeLaneError> {
        if !self.accepts(marker) {
            return Err(FakeLaneError::Refused);
        }
        let mut lane = self.shared.lock(self.index);
        let lane = &mut *lane;
        if lane.fail_awaits > 0 {
            lane.fail_awaits -= 1;
            return Err(FakeLaneError::AwaitFailed);
        }
        self.shared.record(lane, FakeOp::Await(marker));
        Ok(())
    }

    fn mark(&mut self) -> Result<Option<Marker>, FakeLaneError> {
        let mut lane = self.shared.lock(self.index);
        let lane = &mut *lane;
        if lane.fail_marks > 0 {
            lane.fail_marks -= 1;
            return Err(FakeLaneError::MarkFailed);
        }
        if lane.busy > 0 {
            lane.busy -= 1;
            return Ok(None);
        }
        lane.marks += 1;
        let marker = Marker::new(self.shared.ids[self.index], self.lane, lane.marks);
        self.shared.record(lane, FakeOp::Mark(marker));
        Ok(Some(marker))
    }

    fn arm_retirement(&mut self) -> Result<u64, FakeLaneError> {
        let mut lane = self.shared.lock(self.index);
        let lane = &mut *lane;
        if lane.fail_arms > 0 {
            lane.fail_arms -= 1;
            return Err(FakeLaneError::ArmFailed);
        }
        let seq = lane.next_seq;
        lane.next_seq += 1;
        lane.armed.push_back(seq);
        self.shared.record(lane, FakeOp::Arm(seq));
        Ok(seq)
    }

    fn poll_retired(&mut self) -> Option<Retired<FakeLaneError>> {
        self.shared.lock(self.index).arrived.pop_front()
    }

    unsafe fn raw(&mut self) -> u64 {
        let mut lane = self.shared.lock(self.index);
        let lane = &mut *lane;
        lane.raws += 1;
        self.shared.record(lane, FakeOp::Raw);
        lane.raws
    }
}

impl Drop for FakeLane {
    fn drop(&mut self) {
        self.shared.dropped[self.index].store(true, Ordering::SeqCst);
        if self.shared.panic_on_drop[self.index].load(Ordering::SeqCst) {
            panic!("the fake lane's shutdown panicked");
        }
    }
}
