//! Delivery into the lanes' intakes.
//!
//! A lane's own intake is an MPSC queue: any thread delivers, the lane's source takes, and delivery
//! order is kept per lane. A lane group's shared intake is an MPMC queue every source of the group
//! takes from, one item at a time. Delivery is an `offer` and nothing more.

use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

use euhedral_data_structures::queues::{Consumer, MpmcQueue, MpscQueue, Multi, Producer, Single};

use crate::lanes::{LaneId, LanePool};

/// Chunk size and pooled chunks of every intake.
const CHUNK: usize = 256;
const POOLED_CHUNKS: usize = 4;

/// A lane group: lanes whose sources share one intake.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GroupId(pub u16);

/// Where delivered work goes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Deliver {
    /// One lane's own intake: work bound to that lane.
    Lane(LaneId),
    /// A group's shared intake: work that runs on whichever source of the group a worker enters
    /// first.
    Group(GroupId),
}

static NEXT_INTAKES: AtomicU64 = AtomicU64::new(1);

/// Work refused because its lane completed or was dropped, handed back to the deliverer.
pub struct LaneClosed<W>(pub W);

impl<W> fmt::Debug for LaneClosed<W> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("LaneClosed(..)")
    }
}

impl<W> fmt::Display for LaneClosed<W> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("the lane completed or was dropped, so it takes no more work")
    }
}

impl<W> std::error::Error for LaneClosed<W> {}

/// Delivered work and when it was delivered.
pub(crate) struct Envelope<W> {
    pub(crate) work: W,
    pub(crate) delivered: Instant,
}

/// The delivery side of every lane's intakes.
pub struct LaneIntakes<W> {
    /// Tells this set of intakes from another, for `LaneInputs`.
    id: u64,
    pool: Arc<LanePool>,
    /// Per lane: set when the lane's source completes or is dropped.
    closed: Box<[AtomicBool]>,
    own: Box<[Producer<Envelope<W>, Multi>]>,
    shared: Box<[Producer<Envelope<W>, Multi>]>,
    group_of: Box<[Option<GroupId>]>,
    /// Each group's lanes.
    members: Box<[Box<[LaneId]>]>,
    /// A consumer of each shared intake, for tests that watch it from outside a source.
    #[cfg(feature = "testing")]
    probes: Box<[Consumer<Envelope<W>, Multi>]>,
}

impl<W> std::fmt::Debug for LaneIntakes<W> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LaneIntakes")
            .field("lanes", &self.own.len())
            .field("groups", &self.shared.len())
            .finish()
    }
}

/// The taking side of one lane's intakes, for its [`LaneSource`](super::LaneSource).
pub struct LaneInputs<W> {
    lane: LaneId,
    /// The `LaneIntakes` these inputs belong to.
    intakes: u64,
    pub(crate) own: Consumer<Envelope<W>, Single>,
    pub(crate) shared: Option<Consumer<Envelope<W>, Multi>>,
}

impl<W> std::fmt::Debug for LaneInputs<W> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LaneInputs")
            .field("lane", &self.lane)
            .field("grouped", &self.shared.is_some())
            .finish()
    }
}

impl<W> LaneInputs<W> {
    pub fn lane(&self) -> LaneId {
        self.lane
    }

    /// The identity of the `LaneIntakes` these inputs were made with.
    pub(crate) fn intakes_id(&self) -> u64 {
        self.intakes
    }
}

impl<W> LaneIntakes<W> {
    /// Whether `lane`'s source completed or was dropped.
    pub fn is_closed(&self, lane: LaneId) -> bool {
        self.closed[usize::from(lane.0)].load(Ordering::SeqCst)
    }

    /// Closes `lane`: later deliveries to it are refused.
    pub(crate) fn close(&self, lane: LaneId) {
        self.closed[usize::from(lane.0)].store(true, Ordering::SeqCst);
    }

    pub(crate) fn id(&self) -> u64 {
        self.id
    }
}

impl<W: Send + 'static> LaneIntakes<W> {
    /// One MPSC intake per lane of `pool` and one MPMC intake per group; `groups[g]` lists group
    /// `g`'s lanes. The inputs come back in lane order: `inputs[i]` is lane `i`'s.
    ///
    /// # Panics
    ///
    /// If a lane is in two groups or is not a lane of `pool`.
    pub fn new(pool: Arc<LanePool>, groups: &[&[LaneId]]) -> (Arc<Self>, Vec<LaneInputs<W>>) {
        let lanes = pool.len();
        let mut group_of: Vec<Option<GroupId>> = vec![None; lanes];
        let mut shared = Vec::with_capacity(groups.len());
        let mut shared_queues = Vec::with_capacity(groups.len());
        for (g, members) in groups.iter().enumerate() {
            let id = GroupId(u16::try_from(g).expect("fewer than 65536 groups"));
            for lane in *members {
                let slot = group_of
                    .get_mut(usize::from(lane.0))
                    .unwrap_or_else(|| panic!("lane {} is not a lane of the pool", lane.0));
                assert!(slot.is_none(), "lane {} is in two groups", lane.0);
                *slot = Some(id);
            }
            let queue: MpmcQueue<Envelope<W>> = MpmcQueue::with_pool(CHUNK, POOLED_CHUNKS);
            shared.push(queue.producer());
            shared_queues.push(queue);
        }
        let id = NEXT_INTAKES.fetch_add(1, Ordering::Relaxed);
        let mut own = Vec::with_capacity(lanes);
        let mut inputs = Vec::with_capacity(lanes);
        for (i, group) in group_of.iter().enumerate() {
            let queue: MpscQueue<Envelope<W>> = MpscQueue::with_pool(CHUNK, POOLED_CHUNKS);
            own.push(queue.producer());
            inputs.push(LaneInputs {
                lane: LaneId(u16::try_from(i).expect("at most 64 lanes")),
                intakes: id,
                own: queue.consumer(),
                shared: group.map(|g| shared_queues[usize::from(g.0)].consumer()),
            });
        }
        let intakes = Arc::new(Self {
            id,
            closed: (0..lanes).map(|_| AtomicBool::new(false)).collect(),
            pool,
            own: own.into_boxed_slice(),
            shared: shared.into_boxed_slice(),
            group_of: group_of.into_boxed_slice(),
            members: groups
                .iter()
                .map(|g| g.to_vec().into_boxed_slice())
                .collect(),
            #[cfg(feature = "testing")]
            probes: shared_queues.iter().map(MpmcQueue::consumer).collect(),
        });
        (intakes, inputs)
    }

    /// Queues `work` in a lane's own intake or a group's shared intake. It never runs the work and
    /// never waits.
    ///
    /// A lane that completed or was dropped refuses the work, and so does a group whose lanes all
    /// did: it comes back as `Err`, so the deliverer ends it with an error. A delivery that races the lane's closing can still land in
    /// the intake just after the lane drained it; that item is dropped with the intakes.
    ///
    /// # Panics
    ///
    /// On a lane or group the pool lacks.
    pub fn deliver(&self, work: W, to: Deliver) -> Result<(), LaneClosed<W>> {
        let producer = match to {
            Deliver::Lane(lane) => {
                let producer = self
                    .own
                    .get(usize::from(lane.0))
                    .unwrap_or_else(|| panic!("lane {} is not a lane of the pool", lane.0));
                if self.is_closed(lane) {
                    return Err(LaneClosed(work));
                }
                producer
            }
            Deliver::Group(group) => {
                let producer = self
                    .shared
                    .get(usize::from(group.0))
                    .unwrap_or_else(|| panic!("group {} does not exist", group.0));
                if self.members[usize::from(group.0)]
                    .iter()
                    .all(|&lane| self.is_closed(lane))
                {
                    return Err(LaneClosed(work));
                }
                producer
            }
        };
        let envelope = Envelope {
            work,
            delivered: Instant::now(),
        };
        if producer.offer(envelope).is_err() {
            unreachable!("an unbounded intake takes every offer");
        }
        Ok(())
    }

    pub fn pool(&self) -> &Arc<LanePool> {
        &self.pool
    }

    pub fn group_of(&self, lane: LaneId) -> Option<GroupId> {
        self.group_of.get(usize::from(lane.0)).copied().flatten()
    }

    /// Takes the next item of `group`'s shared intake as a bystander would, for tests that check
    /// the intake is free while a source is inside an item. Needs feature `testing`.
    #[cfg(feature = "testing")]
    pub fn probe_shared(&self, group: GroupId) -> Option<W> {
        self.probes[usize::from(group.0)].poll().map(|e| e.work)
    }

    /// Items waiting in `lane`'s own intake. Needs feature `testing`.
    #[cfg(feature = "testing")]
    pub fn own_len(&self, lane: LaneId) -> u64 {
        self.own[usize::from(lane.0)].size_long()
    }

    /// Items waiting in `group`'s shared intake. Needs feature `testing`.
    #[cfg(feature = "testing")]
    pub fn shared_len(&self, group: GroupId) -> u64 {
        self.shared[usize::from(group.0)].size_long()
    }
}
