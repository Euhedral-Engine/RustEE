//! A fake device queue, for tests of code that submits to device lanes (feature `testing`).

use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

use crate::lanes::{DeviceQueue, LaneId, LanePool, Marker};

/// A fake device queue: submissions run in order, each after the queue's earlier work and after the
/// markers it waits for (on any lane of its pool), taking its given cost. A query is one comparison
/// with the wall clock; nothing waits.
#[derive(Debug)]
pub struct FakeDevice {
    name: String,
    lane: LaneId,
    timelines: Arc<Timelines>,
}

#[derive(Debug)]
struct Timelines {
    start: Instant,
    /// Per lane: when each submitted value completes (value k at index k − 1), in ns from `start`.
    done_at: Mutex<Vec<Vec<u64>>>,
}

impl FakeDevice {
    /// A pool of fake lanes named `names` (lane i is `names[i]`), sharing one timeline so that
    /// cross-lane waits work.
    pub fn pool(names: &[&str]) -> (Arc<LanePool>, Vec<Arc<FakeDevice>>) {
        let timelines = Arc::new(Timelines {
            start: Instant::now(),
            done_at: Mutex::new(vec![Vec::new(); names.len()]),
        });
        let devices: Vec<Arc<FakeDevice>> = names
            .iter()
            .enumerate()
            .map(|(i, name)| {
                Arc::new(FakeDevice {
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
        u64::try_from(self.timelines.start.elapsed().as_nanos()).unwrap_or(u64::MAX)
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
        Marker {
            lane: self.lane,
            value: done_at[lane].len() as u64,
        }
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
