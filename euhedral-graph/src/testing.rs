//! Fakes for tests of code that submits to device lanes (feature `testing`): [`FakeDevice`], an in-order fake
//! device queue on a timeline (the wall clock, or a [`StepClock`] the test sets), [`FakeLanes`], a stepped fake lane submitter, and [`StageHarness`], which calls a
//! [`Stage`](crate::Stage)'s hooks without a graph.

#![allow(
    clippy::disallowed_types,
    reason = "a test double: its records are locked by the test thread and the code under test"
)]

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

use crate::lanes::{DeviceQueue, LaneId, LanePool, Marker, QueueId};

mod lanes;
mod stage;

pub use lanes::{FakeLane, FakeLaneError, FakeLanes, FakeOp};
pub use stage::{GateEvent, HarnessRun, PassGate, StageHarness};

/// A clock a test owns: it stands still until the test sets it. A [`FakeDevice`] pool built with
/// [`FakeDevice::pool_stepped`] reads its time from one, so what has completed depends on what the
/// test did, not on how fast the machine ran.
#[derive(Debug, Default)]
pub struct StepClock {
    ns: AtomicU64,
}

impl StepClock {
    /// A clock at zero.
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// The time now, in ns.
    pub fn now_ns(&self) -> u64 {
        self.ns.load(Ordering::Acquire)
    }

    /// Moves the clock forward to `ns` (never back).
    pub fn advance_to(&self, ns: u64) {
        self.ns.fetch_max(ns, Ordering::AcqRel);
    }
}

/// Where a pool's time comes from.
#[derive(Debug)]
enum TimeSource {
    Wall(Instant),
    Stepped(Arc<StepClock>),
}

/// A fake device queue: submissions run in order, each after the queue's earlier work and after the
/// markers it waits for (on any lane of its pool), taking its given cost. A query is one comparison
/// with the pool's clock (the wall clock, or a [`StepClock`]); nothing waits.
#[derive(Debug)]
pub struct FakeDevice {
    id: QueueId,
    name: String,
    lane: LaneId,
    timelines: Arc<Timelines>,
}

#[derive(Debug)]
struct Timelines {
    time: TimeSource,
    /// Each lane's queue identity: a marker of another queue is no lane of this pool.
    ids: Vec<QueueId>,
    /// Per lane: when each submitted value completes (value k at index k − 1), in ns from `start`.
    done_at: Mutex<Vec<Vec<u64>>>,
}

impl FakeDevice {
    /// A pool of fake lanes named `names` (lane i is `names[i]`), sharing one timeline so that
    /// cross-lane waits work.
    pub fn pool(names: &[&str]) -> (Arc<LanePool>, Vec<Arc<FakeDevice>>) {
        Self::pool_on(names, TimeSource::Wall(Instant::now()))
    }

    /// As [`pool`](Self::pool), on a clock the test steps: a submission starts at the clock's time
    /// (or later, behind earlier work) and completes when the clock reaches its end.
    pub fn pool_stepped(
        names: &[&str],
        clock: &Arc<StepClock>,
    ) -> (Arc<LanePool>, Vec<Arc<FakeDevice>>) {
        Self::pool_on(names, TimeSource::Stepped(clock.clone()))
    }

    fn pool_on(names: &[&str], time: TimeSource) -> (Arc<LanePool>, Vec<Arc<FakeDevice>>) {
        let ids: Vec<QueueId> = names.iter().map(|_| QueueId::next()).collect();
        let timelines = Arc::new(Timelines {
            time,
            ids: ids.clone(),
            done_at: Mutex::new(vec![Vec::new(); names.len()]),
        });
        let devices: Vec<Arc<FakeDevice>> = names
            .iter()
            .enumerate()
            .map(|(i, name)| {
                Arc::new(FakeDevice {
                    id: ids[i],
                    name: (*name).to_owned(),
                    lane: LaneId(u16::try_from(i).expect("at most 64 lanes")),
                    timelines: timelines.clone(),
                })
            })
            .collect();
        let queues = devices
            .iter()
            .map(|d| d.clone() as Arc<dyn DeviceQueue>)
            .collect();
        (Arc::new(LanePool::new(queues)), devices)
    }

    pub fn lane(&self) -> LaneId {
        self.lane
    }

    fn now_ns(&self) -> u64 {
        match &self.timelines.time {
            TimeSource::Wall(start) => {
                u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX)
            }
            TimeSource::Stepped(clock) => clock.now_ns(),
        }
    }

    /// The earliest completion time after `after_ns` of any submission on any lane of the pool, in
    /// ns (for a test that steps its clock to the next event).
    pub fn next_done_after_ns(&self, after_ns: u64) -> Option<u64> {
        let done_at = self
            .timelines
            .done_at
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        done_at
            .iter()
            .filter_map(|lane| {
                lane.get(lane.partition_point(|&at| at <= after_ns))
                    .copied()
            })
            .min()
    }

    /// Queues `cost` of work behind this lane's earlier work and behind `waits`; returns its marker
    /// at once.
    pub fn submit(&self, cost: Duration, waits: &[Marker]) -> Marker {
        let now = self.now_ns();
        let mut done_at = self
            .timelines
            .done_at
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let lane = usize::from(self.lane.0);
        let mut start = now.max(done_at[lane].last().copied().unwrap_or(0));
        for wait in waits {
            // A marker of another pool names no lane here: it orders nothing.
            if self.timelines.ids.get(usize::from(wait.lane.0)) != Some(&wait.queue) {
                continue;
            }
            let index = usize::try_from(wait.value).unwrap_or(usize::MAX);
            if let Some(&at) = index
                .checked_sub(1)
                .and_then(|i| done_at.get(usize::from(wait.lane.0))?.get(i))
            {
                start = start.max(at);
            }
        }
        let done = start.saturating_add(u64::try_from(cost.as_nanos()).unwrap_or(u64::MAX));
        done_at[lane].push(done);
        Marker::new(self.id, self.lane, done_at[lane].len() as u64)
    }

    /// How long until `marker` completes (zero once it has; for a fake of a blocking wait).
    pub fn until_done(&self, marker: Marker) -> Duration {
        let now = self.now_ns();
        Duration::from_nanos(
            self.done_at_ns(marker)
                .map_or(0, |at| at.saturating_sub(now)),
        )
    }

    /// When `marker` completes, in ns from the pool's start (for tests).
    pub fn done_at_ns(&self, marker: Marker) -> Option<u64> {
        if self.timelines.ids.get(usize::from(marker.lane.0)) != Some(&marker.queue) {
            return None;
        }
        let done_at = self
            .timelines
            .done_at
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let index = usize::try_from(marker.value).ok()?.checked_sub(1)?;
        done_at.get(usize::from(marker.lane.0))?.get(index).copied()
    }
}

impl DeviceQueue for FakeDevice {
    fn id(&self) -> QueueId {
        self.id
    }

    fn completed(&self) -> u64 {
        let now = self.now_ns();
        let done_at = self
            .timelines
            .done_at
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        done_at[usize::from(self.lane.0)].partition_point(|&at| at <= now) as u64
    }

    fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn work_runs_in_order_each_after_its_cost() {
        let (pool, devices) = FakeDevice::pool(&["compute"]);
        let first = devices[0].submit(Duration::from_millis(50), &[]);
        let second = devices[0].submit(Duration::from_millis(50), &[]);
        assert_eq!((first.value, second.value), (1, 2));
        assert!(!pool.reached(first), "nothing is done at once");
        let (a, b) = (
            devices[0].done_at_ns(first).unwrap(),
            devices[0].done_at_ns(second).unwrap(),
        );
        assert!(b >= a + 50_000_000, "in order, each after its cost");
    }

    #[test]
    fn a_wait_on_another_lane_orders_behind_its_marker() {
        let (_pool, devices) = FakeDevice::pool(&["transfer", "compute"]);
        let transferred = devices[0].submit(Duration::from_millis(40), &[]);
        let computed = devices[1].submit(Duration::from_millis(1), &[transferred]);
        assert!(
            devices[1].done_at_ns(computed).unwrap()
                >= devices[0].done_at_ns(transferred).unwrap() + 1_000_000
        );
    }

    #[test]
    fn zero_cost_work_with_no_waits_is_reached_at_once() {
        let (pool, devices) = FakeDevice::pool(&["copy"]);
        let marker = devices[0].submit(Duration::ZERO, &[]);
        assert!(pool.reached(marker));
    }
}
