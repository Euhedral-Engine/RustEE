//! Device lanes and markers (the upstream Java `LanePool`).

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
/// CUDA stream with an event ring, a fixed-function hardware engine behind a session, or a fake.
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

    /// The highest value of a marker issued on this queue: the lane's work up to it has been submitted. When
    /// [`completed`](Self::completed) has reached it, the queue has no marked work in flight. One cheap atomic read. The
    /// default is [`completed`](Self::completed): a queue that cannot say what it issued reports itself idle.
    fn issued(&self) -> u64 {
        self.completed()
    }

    /// A short name for logs and metrics.
    fn name(&self) -> &str;
}

/// How a stage that continues no lane picks one of its allowed lanes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Placement {
    /// A random allowed lane from a per-thread generator (Java's `ThreadLocalRandom`): independent frames spread over
    /// many device orderings and the workers that submit them rarely share one, with no shared counter to contend on.
    #[default]
    Random,
    /// The allowed lanes in turn, from one shared counter. Deterministic, which tests need (Java's `fixed`
    /// placement); a counter shared by every worker.
    Turns,
}

/// The device lanes shared by every stage graph of a process.
///
/// A stage chooses its lane each time it runs: it continues the lane of the predecessor whose
/// critical successor it is (each stage's lane is continued by one successor only), a root continues its graph's home
/// lane ([`next_home`](Self::next_home)), and any other branch of a fan-out takes a random lane. A dependency between
/// stages on different lanes is a device-side wait on the predecessor's marker; stages on one lane rely on the lane's
/// order.
///
/// The pool is typed over its lane, as Java's `LanePool` holds `GpuStream`s: with the default `dyn DeviceQueue` a lane
/// is only the query side; a pool of a backend's shared lane type gives a stage the lane it was given to submit to
/// (see [`StageCx::lane_handle`](crate::StageCx::lane_handle)).
pub struct LanePool<L: DeviceQueue + ?Sized = dyn DeviceQueue> {
    lanes: Vec<Arc<L>>,
    placement: Placement,
    /// The turn for stages that continue no lane ([`Placement::Turns`]).
    next: AtomicUsize,
    /// The home lane of the next graph run ([`next_home`](Self::next_home)).
    next_home: AtomicUsize,
}

impl<L: DeviceQueue + ?Sized> fmt::Debug for LanePool<L> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LanePool")
            .field("lanes", &self.lanes.len())
            .finish()
    }
}

impl LanePool {
    /// A pool over `lanes`; [`LaneId`] `i` is `lanes[i]`. Panics with no lanes or more than 64.
    pub fn new(lanes: Vec<Arc<dyn DeviceQueue>>) -> Self {
        Self::typed(lanes)
    }
}

impl<L: DeviceQueue + Sized + 'static> LanePool<L> {
    /// The query side of this pool: the same lanes as `dyn DeviceQueue`, for what polls them without submitting
    /// ([`Completions`](crate::Completions), the lane intakes). Made once, at set-up.
    pub fn as_queues(&self) -> Arc<LanePool> {
        let queues = self
            .lanes
            .iter()
            .map(|lane| lane.clone() as Arc<dyn DeviceQueue>)
            .collect();
        Arc::new(LanePool::new(queues).with_placement(self.placement))
    }
}

impl<L: DeviceQueue + ?Sized> LanePool<L> {
    /// A pool over lanes of one type; [`LaneId`] `i` is `lanes[i]`. Panics with no lanes or more than 64.
    pub fn typed(lanes: Vec<Arc<L>>) -> Self {
        assert!(
            !lanes.is_empty() && lanes.len() <= 64,
            "a lane pool needs 1 to 64 lanes"
        );
        Self {
            lanes,
            placement: Placement::default(),
            next: AtomicUsize::new(0),
            next_home: AtomicUsize::new(0),
        }
    }

    /// The same pool with `placement` for stages that continue no lane.
    #[must_use]
    pub fn with_placement(mut self, placement: Placement) -> Self {
        self.placement = placement;
        self
    }

    pub fn len(&self) -> usize {
        self.lanes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.lanes.is_empty()
    }

    pub fn lane(&self, id: LaneId) -> &Arc<L> {
        &self.lanes[id.0 as usize]
    }

    /// Whether `marker`'s work has completed. Never blocks. A marker of another queue, or of a lane
    /// this pool does not have, is never reached.
    pub fn reached(&self, marker: Marker) -> bool {
        self.lanes
            .get(usize::from(marker.lane.0))
            .is_some_and(|queue| queue.id() == marker.queue && queue.completed() >= marker.value)
    }

    /// Whether no lane has marked work in flight: every lane's [`completed`](DeviceQueue::completed) has reached its
    /// [`issued`](DeviceQueue::issued) marker. A moment of this reading is not a hold: a lane may submit the next instant.
    /// Work a lane has submitted but not yet marked is not seen; a lane that marks after each unit of work (as the lane
    /// sources do) has none outside a unit.
    pub fn idle(&self) -> bool {
        self.lanes.iter().all(|q| q.completed() >= q.issued())
    }

    /// The home lane of a new graph run, taken in turn over the pool's lanes (Java `LanePool.nextHome`): the lane its
    /// roots, and so its critical chain, continue in every quantum.
    pub fn next_home(&self) -> LaneId {
        let turn = self.next_home.fetch_add(1, Ordering::Relaxed) % self.lanes.len();
        LaneId(u16::try_from(turn).expect("at most 64 lanes"))
    }

    /// The lane for a stage: `continued` (the lane of the predecessor it is the critical successor
    /// of, or the home lane of a root) when given and among `allowed`, otherwise one of `allowed` by the pool's
    /// [`Placement`]. `allowed` lists the lanes able to run the stage's kind of work (compute, copy, a hardware
    /// engine). Panics if `allowed` is empty (a CPU-only stage chooses no lane).
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
        let pick = match self.placement {
            Placement::Random => {
                euhedral_data_structures::atomics::thread_random::next_u64() as usize
            }
            Placement::Turns => self.next.fetch_add(1, Ordering::Relaxed),
        };
        allowed[pick % allowed.len()]
    }
}

/// Composes a [`LanePool`] from the queues of one or several backends.
///
/// A backend takes a range of lane ids with [`reserve`](Self::reserve), opens its queues at those
/// ids, and [`set`](Self::set)s each one. The builder is plain and single-threaded; the pool it
/// builds is immutable.
#[derive(Debug, Default)]
pub struct LanePoolBuilder {
    lanes: Vec<Option<Arc<dyn DeviceQueue>>>,
}

impl LanePoolBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// The first of `count` consecutive lane ids, reserved for one backend's queues. Panics past
    /// 64 lanes.
    pub fn reserve(&mut self, count: usize) -> LaneId {
        let first = self.lanes.len();
        assert!(
            first + count <= 64,
            "a lane pool holds at most 64 lanes ({first} reserved, {count} more asked)"
        );
        self.lanes.resize_with(first + count, || None);
        LaneId(u16::try_from(first).expect("at most 64 lanes"))
    }

    /// Puts `queue` at a reserved `lane`. Panics if the lane was not reserved or is already set.
    pub fn set(&mut self, lane: LaneId, queue: Arc<dyn DeviceQueue>) {
        let slot = self
            .lanes
            .get_mut(usize::from(lane.0))
            .unwrap_or_else(|| panic!("lane {} was not reserved", lane.0));
        assert!(slot.is_none(), "lane {} is already set", lane.0);
        *slot = Some(queue);
    }

    /// The pool: lane `i` is the queue set at `i`. Panics if a reserved lane is unset.
    pub fn build(self) -> Arc<LanePool> {
        let queues = self
            .lanes
            .into_iter()
            .enumerate()
            .map(|(i, queue)| queue.unwrap_or_else(|| panic!("reserved lane {i} was never set")))
            .collect();
        Arc::new(LanePool::new(queues))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering;

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

    /// A queue with a settable issued and completed value.
    #[derive(Debug)]
    struct Counting {
        id: QueueId,
        issued: std::sync::atomic::AtomicU64,
        completed: std::sync::atomic::AtomicU64,
    }

    impl DeviceQueue for Counting {
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
            "counting"
        }
    }

    #[test]
    fn a_pool_is_idle_when_every_lane_reached_what_it_issued() {
        let make = || {
            Arc::new(Counting {
                id: QueueId::next(),
                issued: 0.into(),
                completed: 0.into(),
            })
        };
        let (a, b) = (make(), make());
        let pool = LanePool::new(vec![a.clone(), b.clone()]);
        assert!(pool.idle(), "nothing issued");
        a.issued.store(3, Ordering::Release);
        assert!(!pool.idle(), "lane a has marked work in flight");
        a.completed.store(2, Ordering::Release);
        assert!(!pool.idle());
        a.completed.store(3, Ordering::Release);
        assert!(pool.idle());
        b.issued.store(1, Ordering::Release);
        assert!(!pool.idle(), "lane b has");
        assert!(
            LanePool::new(vec![Arc::new(Idle(QueueId::next()))]).idle(),
            "the default issued is completed"
        );
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
    fn a_random_placement_picks_only_allowed_lanes_and_reaches_every_one() {
        let pool = pool(4);
        let allowed = [LaneId(1), LaneId(2), LaneId(3)];
        let mut seen = [false; 4];
        for _ in 0..2_000 {
            let lane = pool.choose(&allowed, Some(LaneId(0)));
            assert!(
                allowed.contains(&lane),
                "a lane it may not run on: {lane:?}"
            );
            seen[usize::from(lane.0)] = true;
        }
        assert_eq!(
            seen,
            [false, true, true, true],
            "every allowed lane, and no other"
        );
    }

    #[test]
    fn a_random_placement_shares_no_counter_between_threads() {
        // The turn counter of `Turns` is untouched by `Random`: the pool places in turns only when asked to.
        let pool = pool(2);
        for _ in 0..10 {
            pool.choose(&[LaneId(0), LaneId(1)], None);
        }
        assert_eq!(pool.next.load(Ordering::Relaxed), 0);
        let turns = pool.with_placement(Placement::Turns);
        turns.choose(&[LaneId(0), LaneId(1)], None);
        assert_eq!(turns.next.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn homes_are_taken_in_turn_over_the_lanes() {
        let pool = pool(3);
        let homes: Vec<LaneId> = (0..7).map(|_| pool.next_home()).collect();
        assert_eq!(
            homes,
            [0, 1, 2, 0, 1, 2, 0].map(LaneId),
            "Java `nextHome`: a graph's home lane, round robin"
        );
    }

    #[test]
    fn otherwise_it_takes_its_allowed_lanes_in_turn() {
        let pool = pool(4).with_placement(Placement::Turns);
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

    #[test]
    fn a_built_pool_holds_each_reserved_lane_at_its_id() {
        let mut builder = LanePoolBuilder::new();
        let a = builder.reserve(2);
        let b = builder.reserve(3);
        assert_eq!((a, b), (LaneId(0), LaneId(2)));
        let queues: Vec<Arc<dyn DeviceQueue>> = (0..5)
            .map(|_| Arc::new(Idle(QueueId::next())) as Arc<dyn DeviceQueue>)
            .collect();
        // Set out of order: ids, not call order, place a queue.
        for i in [3usize, 0, 4, 1, 2] {
            builder.set(LaneId(i as u16), queues[i].clone());
        }
        let pool = builder.build();
        assert_eq!(pool.len(), 5);
        for (i, queue) in queues.iter().enumerate() {
            assert_eq!(pool.lane(LaneId(i as u16)).id(), queue.id());
        }
    }

    #[test]
    fn a_pool_of_two_backends_reaches_markers_of_both() {
        let mut builder = LanePoolBuilder::new();
        let first = builder.reserve(1);
        let second = builder.reserve(1);
        let (x, y) = (
            Arc::new(Idle(QueueId::next())),
            Arc::new(Idle(QueueId::next())),
        );
        builder.set(first, x.clone());
        builder.set(second, y.clone());
        let pool = builder.build();
        assert!(pool.reached(Marker::new(x.id(), first, 0)));
        assert!(pool.reached(Marker::new(y.id(), second, 0)));
        assert!(
            !pool.reached(Marker::new(x.id(), second, 0)),
            "a queue at another lane"
        );
    }

    #[test]
    #[should_panic(expected = "never set")]
    fn building_with_an_unset_lane_panics() {
        let mut builder = LanePoolBuilder::new();
        builder.reserve(2);
        builder.set(LaneId(0), Arc::new(Idle(QueueId::next())));
        let _ = builder.build();
    }

    #[test]
    #[should_panic(expected = "already set")]
    fn setting_a_lane_twice_panics() {
        let mut builder = LanePoolBuilder::new();
        let lane = builder.reserve(1);
        builder.set(lane, Arc::new(Idle(QueueId::next())));
        builder.set(lane, Arc::new(Idle(QueueId::next())));
    }

    #[test]
    #[should_panic(expected = "at most 64")]
    fn reserving_past_64_lanes_panics() {
        let mut builder = LanePoolBuilder::new();
        builder.reserve(60);
        builder.reserve(5);
    }
}
