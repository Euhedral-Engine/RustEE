//! Device lanes and markers (Euhedral-Inference `LanePool`).

use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Which lane of a [`LanePool`] a marker belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LaneId(pub u16);

/// A point on a lane's timeline: the lane's work up to `value` (a Vulkan timeline-semaphore value;
/// for CUDA, the index of an event the lane recorded). Copyable; querying it never blocks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Marker {
    pub lane: LaneId,
    pub value: u64,
}

/// A device queue that runs its submissions in order: a Vulkan queue with a timeline semaphore, a
/// CUDA stream with an event ring, a hardware codec engine behind a session, or a fake.
///
/// Submitting is technology-specific and lives in the adapter that owns the queue; what is generic
/// is the query. An adapter's submit takes the [`Marker`]s its work must wait for on the device and
/// returns the marker of the new work.
pub trait DeviceQueue: Send + Sync + fmt::Debug {
    /// The highest value whose work has completed. One cheap query (`vkGetSemaphoreCounterValue`,
    /// `cuEventQuery`); never blocks.
    fn completed(&self) -> u64;

    /// A short name for logs and metrics.
    fn name(&self) -> &str;
}

/// The device lanes shared by every stage graph of a process.
///
/// A stage chooses its lane each time it runs: it continues the lane of its critical-path
/// predecessor, or takes a lane in turn (a root, or another branch of a fan-out). A dependency
/// between stages on different lanes is a device-side wait on the predecessor's marker; stages on
/// one lane rely on the lane's order.
pub struct LanePool {
    lanes: Vec<Arc<dyn DeviceQueue>>,
    /// The turn for stages that continue no lane.
    next: AtomicUsize,
}

impl fmt::Debug for LanePool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LanePool")
            .field("lanes", &self.lanes.len())
            .finish()
    }
}

impl LanePool {
    /// A pool over `lanes`; [`LaneId`] `i` is `lanes[i]`. Panics with no lanes or more than 64.
    pub fn new(lanes: Vec<Arc<dyn DeviceQueue>>) -> Self {
        assert!(
            !lanes.is_empty() && lanes.len() <= 64,
            "a lane pool needs 1 to 64 lanes"
        );
        Self {
            lanes,
            next: AtomicUsize::new(0),
        }
    }

    pub fn len(&self) -> usize {
        self.lanes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.lanes.is_empty()
    }

    pub fn lane(&self, id: LaneId) -> &Arc<dyn DeviceQueue> {
        &self.lanes[id.0 as usize]
    }

    /// Whether `marker`'s work has completed. Never blocks.
    pub fn reached(&self, marker: Marker) -> bool {
        self.lane(marker.lane).completed() >= marker.value
    }

    /// The lane for a stage: `continued` (its critical-path predecessor's lane) when given and
    /// among `allowed`, otherwise one of `allowed` in turn. `allowed` lists the lanes able to run
    /// the stage's kind of work (compute, copy, a codec engine). Panics if `allowed` is empty (a
    /// CPU-only stage chooses no lane).
    pub fn choose(&self, allowed: &[LaneId], continued: Option<LaneId>) -> LaneId {
        assert!(
            !allowed.is_empty(),
            "a stage with device work needs at least one allowed lane"
        );
        if let Some(lane) = continued
            && allowed.contains(&lane)
        {
            return lane;
        }
        if let [only] = allowed {
            return *only;
        }
        allowed[self.next.fetch_add(1, Ordering::Relaxed) % allowed.len()]
    }
}

/// The port of Euhedral-Inference's `LanePool` placement rules.
#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct Idle;

    impl DeviceQueue for Idle {
        fn completed(&self) -> u64 {
            0
        }

        fn name(&self) -> &str {
            "idle"
        }
    }

    fn pool(lanes: usize) -> LanePool {
        LanePool::new(
            (0..lanes)
                .map(|_| Arc::new(Idle) as Arc<dyn DeviceQueue>)
                .collect(),
        )
    }

    #[test]
    fn a_stage_continues_its_predecessors_lane_when_it_may_run_there() {
        let pool = pool(4);
        let compute = [LaneId(1), LaneId(2)];
        for _ in 0..10 {
            assert_eq!(pool.choose(&compute, Some(LaneId(2))), LaneId(2));
        }
    }

    #[test]
    fn otherwise_it_takes_its_allowed_lanes_in_turn() {
        let pool = pool(4);
        let compute = [LaneId(1), LaneId(2), LaneId(3)];
        let chosen: Vec<LaneId> = (0..6)
            .map(|_| pool.choose(&compute, Some(LaneId(0))))
            .collect();
        assert!(
            chosen.iter().all(|lane| compute.contains(lane)),
            "never a lane it may not run on"
        );
        for lane in compute {
            assert_eq!(
                chosen.iter().filter(|&&c| c == lane).count(),
                2,
                "each lane in turn"
            );
        }
        assert_eq!(
            pool.choose(&[LaneId(3)], None),
            LaneId(3),
            "a single allowed lane"
        );
    }

    #[test]
    #[should_panic(expected = "at least one allowed lane")]
    fn a_stage_with_no_allowed_lane_cannot_choose() {
        let _ = pool(1).choose(&[], None);
    }

    #[test]
    fn a_marker_is_reached_once_its_lane_completed_it() {
        let pool = pool(1);
        assert!(pool.reached(Marker {
            lane: LaneId(0),
            value: 0
        }));
        assert!(!pool.reached(Marker {
            lane: LaneId(0),
            value: 1
        }));
    }
}
