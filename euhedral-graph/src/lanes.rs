//! Device lanes and markers (Euhedral-Inference `LanePool`).

use std::fmt;
use std::num::NonZeroU64;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// Which lane of a [`LanePool`] a marker belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LaneId(pub u16);

/// The identity of a [`DeviceQueue`], unique in the process. It is opaque: a queue draws one with
/// [`QueueId::next`] when it is made and stamps it on the [`Marker`]s it makes, so a marker can be
/// told from one of another queue even when their lanes and values coincide.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct QueueId(NonZeroU64);

static NEXT_QUEUE: AtomicU64 = AtomicU64::new(1);

impl QueueId {
    /// A new identity, never drawn before in this process.
    pub fn next() -> Self {
        let id = NEXT_QUEUE.fetch_add(1, Ordering::Relaxed);
        Self(
            NonZeroU64::new(id)
                .expect("the counter starts at 1 and does not wrap in a process's life"),
        )
    }

    /// The identity as a number, for logs.
    pub fn get(self) -> u64 {
        self.0.get()
    }
}

impl fmt::Display for QueueId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "queue#{}", self.0)
    }
}

/// A point on a lane's timeline: the lane's work up to `value` (a Vulkan timeline-semaphore value;
/// for CUDA, the index of an event the lane recorded). Copyable; querying it never blocks.
///
/// A marker names the queue that made it (`queue`, from [`DeviceQueue::id`]) as well as the lane
/// and value, so a marker of another queue is never mistaken for one of this pool: a pool does not
/// reach it, and a backend refuses to wait for it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Marker {
    pub queue: QueueId,
    pub lane: LaneId,
    pub value: u64,
}

impl Marker {
    pub fn new(queue: QueueId, lane: LaneId, value: u64) -> Self {
        Self { queue, lane, value }
    }
}

/// A device queue that runs its submissions in order: a Vulkan queue with a timeline semaphore, a
/// CUDA stream with an event ring, a hardware codec engine behind a session, or a fake.
///
/// Submitting is technology-specific and lives in the adapter that owns the queue; what is generic
/// is the query. An adapter's submit takes the [`Marker`]s its work must wait for on the device and
/// returns the marker of the new work.
pub trait DeviceQueue: Send + Sync + fmt::Debug {
    /// This queue's identity, stamped on its markers. Draw it once, with [`QueueId::next`].
    fn id(&self) -> QueueId;

    /// The highest value whose work has completed. One cheap query (`vkGetSemaphoreCounterValue`,
    /// `cuEventQuery`); never blocks.
    fn completed(&self) -> u64;

    /// A short name for logs and metrics.
    fn name(&self) -> &str;
}

/// The device lanes shared by every stage graph of a process.
///
/// A stage chooses its lane each time it runs: it continues the lane of the predecessor whose
/// critical successor it is (each stage's lane is continued by one successor only), or takes a lane
/// in turn (a root, or any other branch of a fan-out). A dependency between stages on different
/// lanes is a device-side wait on the predecessor's marker; stages on one lane rely on the lane's
/// order.
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

    /// Whether `marker`'s work has completed. Never blocks. A marker of another queue, or of a lane
    /// this pool does not have, is never reached.
    pub fn reached(&self, marker: Marker) -> bool {
        self.lanes
            .get(usize::from(marker.lane.0))
            .is_some_and(|queue| queue.id() == marker.queue && queue.completed() >= marker.value)
    }

    /// The lane for a stage: `continued` (the lane of the predecessor it is the critical successor
    /// of) when given and among `allowed`, otherwise one of `allowed` in turn. `allowed` lists the lanes able to run
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
    struct Idle(QueueId);

    impl DeviceQueue for Idle {
        fn id(&self) -> QueueId {
            self.0
        }

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
                .map(|_| Arc::new(Idle(QueueId::next())) as Arc<dyn DeviceQueue>)
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
        let queue = pool.lane(LaneId(0)).id();
        assert!(pool.reached(Marker::new(queue, LaneId(0), 0)));
        assert!(!pool.reached(Marker::new(queue, LaneId(0), 1)));
    }

    #[test]
    fn a_marker_of_another_queue_or_lane_is_never_reached() {
        let (ours, theirs) = (pool(2), pool(2));
        let foreign = theirs.lane(LaneId(0)).id();
        assert_ne!(foreign, ours.lane(LaneId(0)).id());
        // The lane exists here and the value is long done, but the queue is another's.
        assert!(!ours.reached(Marker::new(foreign, LaneId(0), 0)));
        // A lane this pool does not have.
        let own = ours.lane(LaneId(0)).id();
        assert!(!ours.reached(Marker::new(own, LaneId(9), 0)));
        // A queue's own lane in the wrong place.
        assert!(!ours.reached(Marker::new(own, LaneId(1), 0)));
    }

    #[test]
    fn queue_ids_are_unique_and_non_zero() {
        let ids: Vec<QueueId> = (0..100).map(|_| QueueId::next()).collect();
        for (i, a) in ids.iter().enumerate() {
            assert!(a.get() > 0);
            assert!(ids[i + 1..].iter().all(|b| a != b));
        }
    }
}
