//! A lane's cache of recorded graphs: the same sequence of operations run for each unit of work, recorded once and updated and
//! replayed after. [`GraphCache`] is an owned lane's (one owner, `&mut`); [`SharedGraphCache`] is the cache of shared lanes,
//! for any number of workers through `&self`.
//!
//! A lane that runs the same short sequence of launches for each unit of work (a stage of several kernels) records it as a graph the
//! first time and replays it after, giving each replay the unit's own arguments. The cache is one lane's: a graph is replayed with
//! `&mut`, so the lane that owns a graph owns its executable and nothing is locked. A graph that several lanes need is one graph
//! per lane. Keep the cache in the lane's state next to its submitter and call [`GraphCache::run`] from the lane's items (a
//! lane context's `parts()` lends both).
//!
//! - **A hit**: the body runs as an *update pass* (it touches no device: launch *n* of the pass gives launch *n* of the graph its
//!   arguments, per node) and the graph is replayed on the lane's stream. An update does not touch a replay in flight.
//! - **A miss** (a new key): the body runs for real, once, and is recorded; the driver instantiates and uploads the executable.
//! - **A changed shape** (the body under a known key no longer matches the recording's fingerprint): the body runs for real and
//!   the graph is recorded again, in place when the driver can update the executable, else as a new one.
//! - **A warm-up** ([`GraphCache::warm`]) only records: it runs the body for real on units nobody reads, ahead of the first
//!   unit, so no unit pays the instantiation. A key already recorded is left as it is.
//! - **Drops**: [`GraphCache::retain`] moves the graphs whose key went stale to a list, and [`GraphCache::free_stale`] drops it. A
//!   lane calls it from the continuation of the retirement armed after its last replay, when nothing of the stale graphs is in
//!   flight (a driver frees an executable once its launches end, so the drop finds nothing to wait for either way).
//!
//! The body is one piece of code for all three passes (direct, record, update): it enqueues through [`Enqueue`], so its launches
//! line up one to one with the recording.

use std::hash::Hash;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use euhedral_graph::Marker;

use crate::instances::{InstancePool, tag_of};
use crate::{Enqueue, Gpu, GpuError, Graph, GraphHold, Lane, Sequence, Stream};

/// The enqueue body of a recurring sequence: the same code serves a direct run, a recording and an update.
pub trait GraphBody<G: Gpu> {
    /// Enqueues the sequence's operations on `enq`, in order, with this unit's arguments.
    fn enqueue<E: Enqueue<Kernel = G::Kernel>>(&mut self, enq: &mut E) -> Result<(), GpuError>;
}

/// How a unit ran.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ran {
    /// A recorded graph took the arguments and replayed.
    Replayed,
    /// No graph yet: the body ran for real and was recorded.
    Recorded,
    /// The body's shape changed: it ran for real and the graph was recorded again.
    Rerecorded,
    /// Caching is off: the body ran as direct launches.
    Direct,
    /// A warm-up found the key recorded already and did nothing.
    Present,
}

/// What the cache counted.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GraphStats {
    pub replays: u64,
    pub records: u64,
    pub rerecords: u64,
    pub directs: u64,
}

struct Entry<G: Gpu, K> {
    key: K,
    graph: G::Graph,
}

/// One lane's graphs, by key.
pub struct GraphCache<G: Gpu, K> {
    entries: Vec<Entry<G, K>>,
    stale: Vec<G::Graph>,
    enabled: bool,
    capacity: usize,
    stats: GraphStats,
}

impl<G: Gpu, K> std::fmt::Debug for GraphCache<G, K> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GraphCache")
            .field("graphs", &self.entries.len())
            .field("stale", &self.stale.len())
            .field("stats", &self.stats)
            .finish_non_exhaustive()
    }
}

/// The most graphs a cache keeps by default.
pub const DEFAULT_GRAPHS: usize = 128;

impl<G: Gpu, K: Copy + Eq> Default for GraphCache<G, K> {
    fn default() -> Self {
        Self::new()
    }
}

impl<G: Gpu, K: Copy + Eq> GraphCache<G, K> {
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_GRAPHS)
    }

    /// A cache of at most `graphs` graphs: at the bound, the oldest graph goes to the stale list for the next one. The lists
    /// are sized for it, so a lane that stays within the bound allocates nothing after its graphs are recorded.
    pub fn with_capacity(graphs: usize) -> Self {
        GraphCache {
            entries: Vec::with_capacity(graphs),
            stale: Vec::with_capacity(graphs.min(16)),
            enabled: true,
            capacity: graphs.max(1),
            stats: GraphStats::default(),
        }
    }

    /// Turns caching off or on (to compare a sequence's direct launches with its graph).
    pub fn set_enabled(&mut self, on: bool) {
        self.enabled = on;
    }

    pub fn stats(&self) -> GraphStats {
        self.stats
    }

    /// The graphs held.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn contains(&self, key: &K) -> bool {
        self.entries.iter().any(|e| e.key == *key)
    }

    /// Runs the sequence `body` on `stream` under `key`.
    pub fn run<B: GraphBody<G>>(
        &mut self,
        gpu: &G,
        stream: &mut G::Stream,
        key: K,
        body: &mut B,
    ) -> Result<Ran, GpuError> {
        if !self.enabled {
            stream.submit(|enq| body.enqueue(enq))?;
            self.stats.directs += 1;
            return Ok(Ran::Direct);
        }
        if let Some(at) = self.entries.iter().position(|e| e.key == key) {
            let entry = &mut self.entries[at];
            return match gpu.update(&mut entry.graph, |seq| {
                seq.submit(0, |enq| body.enqueue(enq))
            }) {
                Ok(()) => {
                    entry.graph.replay(stream)?;
                    self.stats.replays += 1;
                    Ok(Ran::Replayed)
                }
                Err(GpuError::ShapeChanged { .. }) => {
                    // The body is not the recorded sequence any more: record it again (in place when the driver can).
                    gpu.rerecord(&mut entry.graph, &mut [&mut *stream], |seq| {
                        seq.submit(0, |enq| body.enqueue(enq))
                    })?;
                    self.stats.rerecords += 1;
                    Ok(Ran::Rerecorded)
                }
                Err(other) => Err(other),
            };
        }
        if self.entries.len() >= self.capacity {
            // A bound, not a policy: the oldest graph goes (it is the least likely to be the next unit's).
            let old = self.entries.remove(0);
            self.stale.push(old.graph);
        }
        let graph = gpu.record(&mut [&mut *stream], |seq| {
            seq.submit(0, |enq| body.enqueue(enq))
        })?;
        self.entries.push(Entry { key, graph });
        self.stats.records += 1;
        Ok(Ran::Recorded)
    }

    /// A warm-up: records `key` (running the body for real, on units nobody reads) unless it is recorded already, and replays
    /// nothing.
    pub fn warm<B: GraphBody<G>>(
        &mut self,
        gpu: &G,
        stream: &mut G::Stream,
        key: K,
        body: &mut B,
    ) -> Result<Ran, GpuError> {
        if self.enabled && self.contains(&key) {
            return Ok(Ran::Present);
        }
        self.run(gpu, stream, key, body)
    }

    /// Moves the graphs whose key `keep` refuses to the stale list (an old shape, a released prepare).
    pub fn retain(&mut self, keep: impl Fn(&K) -> bool) {
        let mut i = 0;
        while i < self.entries.len() {
            if keep(&self.entries[i].key) {
                i += 1;
            } else {
                self.stale.push(self.entries.remove(i).graph);
            }
        }
    }

    /// Drops the stale graphs; returns how many. Call it from the continuation of the retirement armed after the lane's last
    /// replay.
    pub fn free_stale(&mut self) -> usize {
        let n = self.stale.len();
        self.stale.clear();
        n
    }

    /// Graphs waiting to be freed.
    pub fn stale(&self) -> usize {
        self.stale.len()
    }
}

/// The shared lane type of a backend's streams.
type LaneOf<G> = <<G as Gpu>::Stream as Stream>::Lane;

/// What a run remembers of a key besides its instances: whether it was seen, and whether recording it failed.
const SEEN: u64 = 1;
const UNRECORDABLE: u64 = 2;

/// Java's per-key sightings (`Capture.sightings`, `Capture.unrecordable`) in fixed tables of atomic words holding a key's
/// tag: a key is recorded on its second sighting, and a key whose capture failed is never recorded again. The two live in
/// separate tables, so a crowd of ordinary sightings never forgets an unrecordable mark (which would make the key record,
/// fail and run its work again). A slot is found by probing a few from the key's tag; a sightings table that is full
/// forgets: the key it overwrites is seen for the first time again, which only delays its recording by a unit.
struct Sightings {
    seen: Box<[AtomicU64]>,
    unrecordable: Box<[AtomicU64]>,
}

impl Sightings {
    const PROBES: usize = 8;

    fn new(slots: usize) -> Self {
        let table = |n: usize| {
            (0..n.max(Self::PROBES))
                .map(|_| AtomicU64::new(0))
                .collect()
        };
        Self {
            seen: table(slots),
            unrecordable: table(slots),
        }
    }

    /// Whether `tag` is in `table`; with `insert`, it is put in an empty slot of its probe if it was not, and at its first
    /// probe (forgetting what was there) when the probe is full.
    fn find(table: &[AtomicU64], tag: u64, insert: bool) -> bool {
        let n = table.len();
        let start = (tag % n as u64) as usize;
        for step in 0..Self::PROBES {
            let slot = &table[(start + step) % n];
            let mut seen = slot.load(Ordering::Relaxed);
            if seen == 0 && insert {
                match slot.compare_exchange(0, tag + 1, Ordering::Relaxed, Ordering::Relaxed) {
                    Ok(_) => return false,
                    Err(now) => seen = now,
                }
            }
            if seen == tag + 1 {
                return true;
            }
        }
        if insert {
            table[start].store(tag + 1, Ordering::Relaxed);
        }
        false
    }

    /// The flags the key had before this sighting, which counts.
    fn see(&self, tag: u64) -> u64 {
        let bad = if Self::find(&self.unrecordable, tag, false) {
            UNRECORDABLE
        } else {
            0
        };
        let seen = if Self::find(&self.seen, tag, true) {
            SEEN
        } else {
            0
        };
        bad | seen
    }

    fn mark_unrecordable(&self, tag: u64) {
        Self::find(&self.unrecordable, tag, true);
    }
}

/// Graphs by key for any number of workers running on shared lanes, through `&self`, without a lock.
///
/// The owned [`GraphCache`] is one lane's: its graphs are replayed with `&mut`, by the lane's owner. A stage on a shared
/// [`Lane`] runs on whichever worker the lattice gives it, and a key can be running on several at once, so the graphs
/// are *instances*, pooled by key ([`InstancePool`](crate::instances)):
///
/// - **First sighting of a key**: the body runs directly (as Java's `StageGraph` runs a quantum stage by stage the first
///   time it sees its capture key). **Second sighting**: it is recorded (the recording runs the work for real). A
///   warm-up ([`warm`](Self::warm)) records at once.
/// - **A hit**: a free instance of the key is taken by one compare-and-set (the entry is claimed, not locked), updated with
///   the unit's arguments and replayed on the lane. The instance goes back to the table once the replay is enqueued.
/// - **A second concurrent user of a key** finds the instance taken and does not wait or spin: it takes another free
///   instance of the key, or records one of its own and pools it for the next. The instances of a key settle at the peak
///   number of workers that ran it at once, and a run after warm-up allocates nothing.
/// - **A changed shape** (the body under a known key no longer matches the recording's fingerprint) runs for real and
///   records again, in place when the driver can update the executable.
/// - **Capacity counts instances**, not keys. A full table evicts the least recently used free instance (Java's
///   `MAX_CAPTURES` is least recently used too). If every instance is in use, the unit runs directly: nothing is recorded
///   to be thrown away. Dropping an instance is safe whatever its launches are doing: the driver frees an executable once
///   they end.
/// - **A capture that fails** (the work still ran, for real) marks the key unrecordable: it runs directly from then on.
///
/// **A replay never launches on the lane.** With several threads submitting to one stream, a `cuGraphLaunch` can overtake
/// the `cuStreamWaitEvent` its own thread issued just before it (seen with driver 615.71.09 on CUDA 13.4: a graph's work
/// ran before what it waited for, in tens of overlapping runs out of a hundred; a plain launch, one submitter per stream, or
/// a wait and a launch locked together do not do it, and neither does reusing an instance whose launch is pending). So
/// each worker forks a stream of its own from the lane for its replay (an event recorded on the lane and awaited by the
/// private stream), makes its cross-lane waits and the launch there, and joins the lane back (an event recorded on the
/// private stream and awaited by the lane); the lane's marker is recorded after that join. The private streams are
/// stocked per lane and reused, so nothing is created after warm-up. The waits are therefore an argument of the run
/// ([`run_after`](Self::run_after)), not something the stage enqueues on the lane first.
///
/// **Why the instance may go back once the replay is enqueued.** A launch's parameters are fixed when it is issued (an
/// update applies to the next launch), and the driver orders every launch of one executable behind the previous one,
/// whichever stream it is on; a stress test of overlapping quanta that returns an instance at once passes. What the replay
/// reads that the instance does not own are the recording's host buffers: the run adds a handle to them to the caller's
/// [`GraphHold`], which the caller holds in its quantum and drops in `Stage::retired`, after the device work ended, so the
/// instance, the cache and the lane may all be gone by then without the work losing them.
///
/// **Markers.** Run it inside [`Lane::submit_marked`]: the marker is claimed before the run, any `Busy` comes before any
/// recording or replay, and the marker is recorded behind it (the closure's enqueue argument is not needed: the run reaches
/// the lane through `lane`).
pub struct SharedGraphCache<G: Gpu, K> {
    pool: InstancePool<K, G::Graph>,
    sightings: Sightings,
    enabled: AtomicBool,
    replays: AtomicU64,
    records: AtomicU64,
    rerecords: AtomicU64,
    directs: AtomicU64,
}

impl<G: Gpu, K> std::fmt::Debug for SharedGraphCache<G, K> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SharedGraphCache")
            .field("pool", &self.pool)
            .finish_non_exhaustive()
    }
}

impl<G: Gpu, K: Copy + Eq + Hash + Send> Default for SharedGraphCache<G, K> {
    fn default() -> Self {
        Self::new()
    }
}

impl<G: Gpu, K: Copy + Eq + Hash + Send> SharedGraphCache<G, K> {
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_GRAPHS)
    }

    /// A cache of at most `instances` graph instances (all keys together).
    pub fn with_capacity(instances: usize) -> Self {
        Self {
            pool: InstancePool::new(instances),
            sightings: Sightings::new(instances.saturating_mul(4)),
            enabled: AtomicBool::new(true),
            replays: AtomicU64::new(0),
            records: AtomicU64::new(0),
            rerecords: AtomicU64::new(0),
            directs: AtomicU64::new(0),
        }
    }

    /// Turns caching off or on.
    pub fn set_enabled(&self, on: bool) {
        self.enabled.store(on, Ordering::Relaxed);
    }

    pub fn stats(&self) -> GraphStats {
        GraphStats {
            replays: self.replays.load(Ordering::Relaxed),
            records: self.records.load(Ordering::Relaxed),
            rerecords: self.rerecords.load(Ordering::Relaxed),
            directs: self.directs.load(Ordering::Relaxed),
        }
    }

    /// The instances held, in use or not, and those being recorded.
    pub fn len(&self) -> usize {
        self.pool.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pool.len() == 0
    }

    /// The most instances the cache keeps.
    pub fn capacity(&self) -> usize {
        self.pool.capacity()
    }

    /// Whether `key` has an instance (in use or not; one still being recorded does not count).
    pub fn contains(&self, key: &K) -> bool {
        self.pool.contains(tag_of(key))
    }

    /// Runs the sequence `body` on `lane` under `key`, adding what the work reads to `hold` (see the type's docs).
    pub fn run<B: GraphBody<G>>(
        &self,
        gpu: &G,
        lane: &LaneOf<G>,
        key: K,
        body: &mut B,
        hold: &mut GraphHold,
    ) -> Result<Ran, GpuError> {
        self.run_after(gpu, lane, &[], key, body, hold)
    }

    /// As [`run`](Self::run), ordered behind `waits` (the markers of the stage's predecessors on other lanes: a stage's
    /// `cx.waits`). A replay makes them on its private stream; any other way to run makes them on the lane first.
    pub fn run_after<B: GraphBody<G>>(
        &self,
        gpu: &G,
        lane: &LaneOf<G>,
        waits: &[Marker],
        key: K,
        body: &mut B,
        hold: &mut GraphHold,
    ) -> Result<Ran, GpuError> {
        self.run_inner(gpu, lane, waits, key, body, hold, false)
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "one body behind run, run_after and warm"
    )]
    fn run_inner<B: GraphBody<G>>(
        &self,
        gpu: &G,
        lane: &LaneOf<G>,
        waits: &[Marker],
        key: K,
        body: &mut B,
        hold: &mut GraphHold,
        record_now: bool,
    ) -> Result<Ran, GpuError> {
        if !self.enabled.load(Ordering::Relaxed) {
            return self.direct(lane, waits, body, hold);
        }
        let tag = tag_of(&key);
        if let Some(mut lease) = self.pool.take(&key, tag) {
            let graph = lease.get();
            return match gpu.update(graph, |seq| seq.submit(0, |enq| body.enqueue(enq))) {
                Ok(()) => {
                    graph.replay_on(lane, waits, hold)?;
                    self.replays.fetch_add(1, Ordering::Relaxed);
                    Ok(Ran::Replayed)
                }
                Err(GpuError::ShapeChanged { .. }) => {
                    // The body is not the recorded sequence any more: record it again (in place when the driver can).
                    wait_all(lane, waits)?;
                    gpu.rerecord_on(graph, &[lane], hold, |seq| {
                        seq.submit(0, |enq| body.enqueue(enq))
                    })?;
                    self.rerecords.fetch_add(1, Ordering::Relaxed);
                    Ok(Ran::Rerecorded)
                }
                Err(other) => Err(other),
            };
        }
        // No free instance of the key: the first sighting runs directly, the second records.
        let flags = self.sightings.see(tag);
        if flags & UNRECORDABLE != 0 || (flags & SEEN == 0 && !record_now) {
            return self.direct(lane, waits, body, hold);
        }
        // A slot is reserved first; with none free, the unit runs directly (nothing is recorded to be thrown away).
        let Some(slot) = self.pool.reserve(tag) else {
            return self.direct(lane, waits, body, hold);
        };
        wait_all(lane, waits)?;
        let ran = std::cell::Cell::new(false);
        let recorded = gpu.record_on(&[lane], hold, |seq| {
            ran.set(true);
            seq.submit(0, |enq| body.enqueue(enq))
        });
        match recorded {
            Ok(graph) => {
                self.records.fetch_add(1, Ordering::Relaxed);
                slot.fill(key, graph);
                Ok(Ran::Recorded)
            }
            Err(GpuError::Capture(_)) => {
                // Only the recording failed: the key runs directly from now on.
                self.sightings.mark_unrecordable(tag);
                drop(slot);
                if ran.get() {
                    // The body ran for real, unaffected by the failed capture: the unit is done.
                    self.directs.fetch_add(1, Ordering::Relaxed);
                    Ok(Ran::Direct)
                } else {
                    // The capture failed before the body ran (the shadow could not be started): the unit has not run.
                    self.direct(lane, &[], body, hold)
                }
            }
            Err(other) => Err(other),
        }
    }

    /// The body as direct launches on the lane, after `waits`.
    fn direct<B: GraphBody<G>>(
        &self,
        lane: &LaneOf<G>,
        waits: &[Marker],
        body: &mut B,
        hold: &mut GraphHold,
    ) -> Result<Ran, GpuError> {
        wait_all(lane, waits)?;
        lane.submit_held(hold, |enq| body.enqueue(enq))?;
        self.directs.fetch_add(1, Ordering::Relaxed);
        Ok(Ran::Direct)
    }

    /// A warm-up: records an instance of `key` (running the body for real, on units nobody reads) unless it has one, and
    /// replays nothing.
    pub fn warm<B: GraphBody<G>>(
        &self,
        gpu: &G,
        lane: &LaneOf<G>,
        key: K,
        body: &mut B,
        hold: &mut GraphHold,
    ) -> Result<Ran, GpuError> {
        if self.enabled.load(Ordering::Relaxed) && self.contains(&key) {
            return Ok(Ran::Present);
        }
        self.run_inner(gpu, lane, &[], key, body, hold, true)
    }

    /// Drops the free instances whose key `keep` refuses (an old shape, a released prepare); instances in use stay for
    /// the next call. Returns how many went.
    pub fn retain(&self, keep: impl Fn(&K) -> bool) -> usize {
        self.pool.retain(keep)
    }
}

/// Orders the lane behind `waits` (the markers of other lanes), on the lane itself.
fn wait_all<L: Lane>(lane: &L, waits: &[Marker]) -> Result<(), GpuError> {
    for marker in waits {
        lane.await_marker(*marker)?;
    }
    Ok(())
}

#[cfg(all(test, feature = "fake"))]
mod tests {
    use std::sync::Arc;

    use crate::fake::{FakeGpu, FakeKernel, FakeStream};
    use crate::{Args, Dim3, Launch, Module, StreamOptions};

    use super::*;

    /// A sequence of `launches` launches of one kernel, each with its own argument, over the fake device.
    struct Steps<'a> {
        kernel: &'a FakeKernel,
        launches: u32,
        value: u32,
    }

    impl GraphBody<FakeGpu> for Steps<'_> {
        fn enqueue<E: Enqueue<Kernel = FakeKernel>>(
            &mut self,
            enq: &mut E,
        ) -> Result<(), GpuError> {
            for i in 0..self.launches {
                let launch = Launch {
                    grid: Dim3::x(1),
                    block: Dim3::x(32),
                    shared_bytes: 0,
                };
                // SAFETY: the fake device reads no addresses.
                unsafe { enq.launch(self.kernel, launch, &Args::new().arg(self.value + i))? };
            }
            Ok(())
        }
    }

    fn rig() -> (Arc<FakeGpu>, FakeStream, FakeKernel) {
        let gpu = FakeGpu::new();
        gpu.define("steps", |_, _, _| Ok(()));
        let (_pool, mut streams) = gpu
            .open_lanes(&["lane"], StreamOptions::default())
            .expect("a lane");
        let module = crate::fake::FakeModule::from_source(&gpu, "steps", &Default::default())
            .expect("a module");
        let kernel = module.kernel("steps").expect("a kernel");
        (gpu, streams.remove(0), kernel)
    }

    const KEY: (u16, u32, u64) = (1, 0, 7);

    #[test]
    fn a_new_key_records_and_the_next_unit_updates_and_replays() {
        let (gpu, mut stream, kernel) = rig();
        let mut cache = GraphCache::<FakeGpu, _>::new();
        let mut body = Steps {
            kernel: &kernel,
            launches: 4,
            value: 10,
        };
        assert_eq!(
            cache.run(&gpu, &mut stream, KEY, &mut body).unwrap(),
            Ran::Recorded
        );
        body.value = 20;
        assert_eq!(
            cache.run(&gpu, &mut stream, KEY, &mut body).unwrap(),
            Ran::Replayed
        );
        assert_eq!(
            cache.run(&gpu, &mut stream, KEY, &mut body).unwrap(),
            Ran::Replayed
        );
        assert_eq!(
            cache.stats(),
            GraphStats {
                replays: 2,
                records: 1,
                rerecords: 0,
                directs: 0
            }
        );
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn variants_and_shapes_are_separate_graphs() {
        let (gpu, mut stream, kernel) = rig();
        let mut cache = GraphCache::<FakeGpu, _>::new();
        let mut body = Steps {
            kernel: &kernel,
            launches: 3,
            value: 0,
        };
        for key in [KEY, (1, 1, 7), (1, 0, 8)] {
            assert_eq!(
                cache.run(&gpu, &mut stream, key, &mut body).unwrap(),
                Ran::Recorded
            );
        }
        assert_eq!(cache.len(), 3);
        assert_eq!(
            cache.run(&gpu, &mut stream, KEY, &mut body).unwrap(),
            Ran::Replayed
        );
    }

    #[test]
    fn a_body_of_another_shape_under_a_known_key_is_recorded_again() {
        let (gpu, mut stream, kernel) = rig();
        let mut cache = GraphCache::<FakeGpu, _>::new();
        let mut body = Steps {
            kernel: &kernel,
            launches: 3,
            value: 0,
        };
        cache.run(&gpu, &mut stream, KEY, &mut body).unwrap();
        body.launches = 5;
        assert_eq!(
            cache.run(&gpu, &mut stream, KEY, &mut body).unwrap(),
            Ran::Rerecorded
        );
        assert_eq!(
            cache.run(&gpu, &mut stream, KEY, &mut body).unwrap(),
            Ran::Replayed
        );
        assert_eq!(cache.len(), 1, "one graph per key");
    }

    #[test]
    fn a_warm_up_records_once_and_replays_nothing() {
        let (gpu, mut stream, kernel) = rig();
        let mut cache = GraphCache::<FakeGpu, _>::new();
        let mut body = Steps {
            kernel: &kernel,
            launches: 2,
            value: 0,
        };
        assert_eq!(
            cache.warm(&gpu, &mut stream, KEY, &mut body).unwrap(),
            Ran::Recorded
        );
        assert_eq!(
            cache.warm(&gpu, &mut stream, KEY, &mut body).unwrap(),
            Ran::Present
        );
        assert_eq!(
            cache.stats(),
            GraphStats {
                replays: 0,
                records: 1,
                rerecords: 0,
                directs: 0
            }
        );
        assert_eq!(
            cache.run(&gpu, &mut stream, KEY, &mut body).unwrap(),
            Ran::Replayed,
            "the first unit pays no recording"
        );
    }

    #[test]
    fn a_stale_key_is_freed_by_the_retirement_not_at_once() {
        let (gpu, mut stream, kernel) = rig();
        let mut cache = GraphCache::<FakeGpu, _>::new();
        let mut body = Steps {
            kernel: &kernel,
            launches: 2,
            value: 0,
        };
        for shape in 0..3u64 {
            cache
                .run(&gpu, &mut stream, (1, 0, shape), &mut body)
                .unwrap();
        }
        cache.retain(|k| k.2 == 2);
        assert_eq!((cache.len(), cache.stale()), (1, 2));
        assert_eq!(cache.free_stale(), 2);
        assert_eq!(cache.stale(), 0);
        assert!(cache.contains(&(1, 0, 2)));
    }

    #[test]
    fn with_caching_off_the_body_runs_as_direct_launches() {
        let (gpu, mut stream, kernel) = rig();
        let mut cache = GraphCache::<FakeGpu, _>::new();
        cache.set_enabled(false);
        let mut body = Steps {
            kernel: &kernel,
            launches: 2,
            value: 0,
        };
        assert_eq!(
            cache.run(&gpu, &mut stream, KEY, &mut body).unwrap(),
            Ran::Direct
        );
        assert!(cache.is_empty());
    }

    #[test]
    fn the_cache_is_bounded() {
        let (gpu, mut stream, kernel) = rig();
        let mut cache = GraphCache::<FakeGpu, _>::with_capacity(4);
        let mut body = Steps {
            kernel: &kernel,
            launches: 1,
            value: 0,
        };
        for shape in 0..9u64 {
            cache
                .run(&gpu, &mut stream, (1, 0, shape), &mut body)
                .unwrap();
        }
        assert_eq!(cache.len(), 4);
        assert_eq!(cache.stale(), 5);
    }
}

/// The shared cache over the fake device: instances by key, taken by claim, the hold of what the work reads.
#[cfg(all(test, feature = "fake"))]
mod shared_tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Barrier};

    use crate::fake::{FakeGpu, FakeKernel, FakeModule, FakeQueue};
    use crate::{Args, Dim3, Lane, Launch, Module, StreamOptions};

    use super::*;

    const KEY: (u16, u32, u64) = (1, 0, 7);

    /// A kernel that adds its second argument to the word its first names.
    fn rig() -> (Arc<FakeGpu>, Arc<FakeQueue>, FakeKernel, u64) {
        let gpu = FakeGpu::new();
        gpu.define("addv", |mem, _launch, args| {
            let at = args.read::<u64>(0).unwrap();
            let by = args.read::<u32>(1).unwrap();
            let word = mem.bytes_mut(at, 4)?;
            let v = u32::from_le_bytes(word.try_into().unwrap()) + by;
            word.copy_from_slice(&v.to_le_bytes());
            Ok(())
        });
        let lanes = crate::open_shared_lanes(
            &gpu,
            &[(
                "lane",
                StreamOptions {
                    markers: 1024,
                    ..StreamOptions::default()
                },
            )],
        )
        .expect("a lane");
        let lane = lanes.pool.lane(euhedral_graph::LaneId(0)).clone();
        let module = FakeModule::from_source(&gpu, "addv", &Default::default()).unwrap();
        let kernel = module.kernel("addv").unwrap();
        let word = gpu.alloc(4);
        (gpu, lane, kernel, word)
    }

    fn word(gpu: &FakeGpu, at: u64) -> u32 {
        u32::from_le_bytes(gpu.read(at, 4).try_into().unwrap())
    }

    /// `launches` launches that add `by` to the word, and a value kept for them.
    struct Adds<'a> {
        kernel: &'a FakeKernel,
        word: u64,
        launches: u32,
        by: u32,
        keeps: Option<Arc<AtomicUsize>>,
        /// Called inside every pass, to hold threads together.
        inside: Option<&'a Barrier>,
    }

    struct Counted(Arc<AtomicUsize>);

    impl Drop for Counted {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    impl GraphBody<FakeGpu> for Adds<'_> {
        fn enqueue<E: Enqueue<Kernel = FakeKernel>>(
            &mut self,
            enq: &mut E,
        ) -> Result<(), GpuError> {
            if let Some(barrier) = self.inside {
                barrier.wait();
            }
            for _ in 0..self.launches {
                let launch = Launch {
                    grid: Dim3::x(1),
                    block: Dim3::x(32),
                    shared_bytes: 0,
                };
                // SAFETY: a live fake allocation.
                unsafe {
                    enq.launch(
                        self.kernel,
                        launch,
                        &Args::new().device(self.word).arg(self.by),
                    )?
                };
            }
            if let Some(count) = &self.keeps {
                enq.keep(Box::new(Counted(count.clone())));
            }
            Ok(())
        }
    }

    fn body<'a>(kernel: &'a FakeKernel, word: u64, launches: u32, by: u32) -> Adds<'a> {
        Adds {
            kernel,
            word,
            launches,
            by,
            keeps: None,
            inside: None,
        }
    }

    #[test]
    fn the_first_sighting_runs_directly_the_second_records_and_the_next_replays_on_the_lane() {
        let (gpu, lane, kernel, at) = rig();
        let cache = SharedGraphCache::<FakeGpu, _>::new();
        let mut hold = GraphHold::new();
        let mut b = body(&kernel, at, 2, 10);
        let mut ran = |cache: &SharedGraphCache<FakeGpu, _>, b: &mut Adds<'_>| {
            cache.run(&gpu, &lane, KEY, b, &mut hold).unwrap()
        };
        assert_eq!(ran(&cache, &mut b), Ran::Direct, "first sighting");
        assert!(cache.is_empty());
        assert_eq!(ran(&cache, &mut b), Ran::Recorded, "second sighting");
        b.by = 1;
        assert_eq!(ran(&cache, &mut b), Ran::Replayed);
        gpu.run_until_idle();
        assert_eq!(word(&gpu, at), 20 + 20 + 2, "every run did its work, once");
        assert_eq!(
            cache.stats(),
            GraphStats {
                replays: 1,
                records: 1,
                rerecords: 0,
                directs: 1
            }
        );
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn variants_shapes_warm_ups_and_retain_work_as_the_owned_caches() {
        let (gpu, lane, kernel, at) = rig();
        let cache = SharedGraphCache::<FakeGpu, _>::new();
        let mut hold = GraphHold::new();
        let mut b = body(&kernel, at, 3, 0);
        for key in [KEY, (1, 1, 7), (1, 0, 8)] {
            assert_eq!(
                cache.warm(&gpu, &lane, key, &mut b, &mut hold).unwrap(),
                Ran::Recorded,
                "a warm-up records at once"
            );
        }
        assert_eq!(cache.len(), 3);
        b.launches = 5;
        assert_eq!(
            cache.run(&gpu, &lane, KEY, &mut b, &mut hold).unwrap(),
            Ran::Rerecorded
        );
        assert_eq!(
            cache.run(&gpu, &lane, KEY, &mut b, &mut hold).unwrap(),
            Ran::Replayed
        );
        assert_eq!(cache.len(), 3, "one instance per key");
        assert_eq!(
            cache.warm(&gpu, &lane, KEY, &mut b, &mut hold).unwrap(),
            Ran::Present
        );
        assert_eq!(
            cache
                .warm(&gpu, &lane, (9, 9, 9), &mut b, &mut hold)
                .unwrap(),
            Ran::Recorded
        );
        assert_eq!(cache.retain(|k| k.0 == 1), 1);
        assert!(!cache.contains(&(9, 9, 9)) && cache.contains(&KEY));
    }

    /// Capacity counts instances, and a full cache evicts the least recently used free one.
    #[test]
    fn the_capacity_counts_instances_and_a_full_cache_evicts_the_least_recently_used() {
        let (gpu, lane, kernel, at) = rig();
        let cache = SharedGraphCache::<FakeGpu, _>::with_capacity(3);
        let mut hold = GraphHold::new();
        let mut b = body(&kernel, at, 1, 0);
        for shape in 0..3u64 {
            cache
                .warm(&gpu, &lane, (1, 0, shape), &mut b, &mut hold)
                .unwrap();
        }
        // Key 0 is used again: key 1 is the oldest.
        assert_eq!(
            cache
                .run(&gpu, &lane, (1, 0, 0), &mut b, &mut hold)
                .unwrap(),
            Ran::Replayed
        );
        cache
            .warm(&gpu, &lane, (1, 0, 3), &mut b, &mut hold)
            .unwrap();
        assert_eq!(cache.len(), 3);
        assert_eq!(cache.capacity(), 3);
        assert!(
            cache.contains(&(1, 0, 0)) && cache.contains(&(1, 0, 2)) && cache.contains(&(1, 0, 3))
        );
        assert!(!cache.contains(&(1, 0, 1)), "the least recently used went");
    }

    #[test]
    fn a_capture_that_fails_runs_the_work_and_marks_the_key_unrecordable() {
        let (gpu, lane, kernel, at) = rig();
        let cache = SharedGraphCache::<FakeGpu, _>::new();
        let mut hold = GraphHold::new();
        let mut b = body(&kernel, at, 1, 1);
        assert_eq!(
            cache.run(&gpu, &lane, KEY, &mut b, &mut hold).unwrap(),
            Ran::Direct
        );
        gpu.fail_next_recording();
        assert_eq!(
            cache.run(&gpu, &lane, KEY, &mut b, &mut hold).unwrap(),
            Ran::Direct,
            "the capture failed: the unit ran directly"
        );
        for _ in 0..3 {
            assert_eq!(
                cache.run(&gpu, &lane, KEY, &mut b, &mut hold).unwrap(),
                Ran::Direct,
                "never recorded again"
            );
        }
        assert_eq!(cache.stats().records, 0);
        assert!(cache.is_empty(), "a failed recording leaves no slot behind");
        gpu.run_until_idle();
        assert_eq!(word(&gpu, at), 5, "every unit's work ran once");
        // A key that never failed still records.
        assert_eq!(
            cache
                .warm(&gpu, &lane, (2, 2, 2), &mut b, &mut hold)
                .unwrap(),
            Ran::Recorded
        );
    }

    /// A capture that cannot start fails before the body ran: the unit has not run, so it runs directly, once.
    #[test]
    fn a_capture_that_fails_before_the_body_ran_still_runs_the_unit_once() {
        let (gpu, lane, kernel, at) = rig();
        let cache = SharedGraphCache::<FakeGpu, _>::new();
        let mut hold = GraphHold::new();
        let mut b = body(&kernel, at, 1, 1);
        cache.run(&gpu, &lane, KEY, &mut b, &mut hold).unwrap(); // first sighting
        gpu.fail_next_recording_before_the_body();
        assert_eq!(
            cache.run(&gpu, &lane, KEY, &mut b, &mut hold).unwrap(),
            Ran::Direct
        );
        gpu.run_until_idle();
        assert_eq!(word(&gpu, at), 2, "two units, each run exactly once");
        assert_eq!(cache.stats().directs, 2);
        assert!(cache.is_empty());
        assert_eq!(
            cache.run(&gpu, &lane, KEY, &mut b, &mut hold).unwrap(),
            Ran::Direct,
            "the key is unrecordable"
        );
    }

    /// An unrecordable mark is not forgotten by a crowd of other keys\' sightings.
    #[test]
    fn a_crowd_of_sightings_does_not_forget_an_unrecordable_key() {
        let sightings = Sightings::new(16);
        let bad = tag_of(&(9u16, 9u32, 9u64));
        sightings.mark_unrecordable(bad);
        for key in 0..(if cfg!(miri) { 200 } else { 50_000u64 }) {
            sightings.see(tag_of(&(1u16, 1u32, key)));
        }
        assert_eq!(sightings.see(bad) & UNRECORDABLE, UNRECORDABLE);
    }

    #[test]
    fn caching_off_runs_the_body_directly_and_holds_what_it_keeps() {
        let (gpu, lane, kernel, at) = rig();
        let cache = SharedGraphCache::<FakeGpu, _>::new();
        cache.set_enabled(false);
        let drops = Arc::new(AtomicUsize::new(0));
        let mut b = body(&kernel, at, 1, 3);
        b.keeps = Some(drops.clone());
        let mut hold = GraphHold::new();
        assert_eq!(
            cache.run(&gpu, &lane, KEY, &mut b, &mut hold).unwrap(),
            Ran::Direct
        );
        gpu.run_until_idle();
        assert_eq!(word(&gpu, at), 3);
        assert_eq!(drops.load(Ordering::SeqCst), 0, "the hold keeps it");
        drop(hold);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        assert!(cache.is_empty());
    }

    /// What the recording keeps lives until the caller's hold is dropped: not until the cache, the graph or the lane are
    /// gone, and not before the hold of the last run that used it.
    #[test]
    fn what_a_recording_keeps_is_held_by_the_callers_until_they_drop_their_holds() {
        let (gpu, lane, kernel, at) = rig();
        let drops = Arc::new(AtomicUsize::new(0));
        let cache = SharedGraphCache::<FakeGpu, _>::new();
        let mut recorded = GraphHold::new();
        let mut b = body(&kernel, at, 1, 1);
        b.keeps = Some(drops.clone());
        cache.warm(&gpu, &lane, KEY, &mut b, &mut recorded).unwrap();
        let mut replayed = GraphHold::new();
        assert_eq!(
            cache.run(&gpu, &lane, KEY, &mut b, &mut replayed).unwrap(),
            Ran::Replayed
        );
        // The update pass's keeps are not the graph's: they were dropped at once, and the recording's are the one value.
        assert_eq!(drops.load(Ordering::SeqCst), 1, "the update pass's value");
        // The cache goes (the instance with it): the work in flight still has its buffers.
        drop(cache);
        gpu.run_until_idle();
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        drop(recorded);
        assert_eq!(
            drops.load(Ordering::SeqCst),
            1,
            "the replay's hold has it too"
        );
        drop(replayed);
        assert_eq!(
            drops.load(Ordering::SeqCst),
            2,
            "released with the last hold"
        );
    }

    /// A replayed instance goes back once its replay is enqueued: the next unit of the key takes it at once, and no instance
    /// is added.
    #[test]
    fn a_replayed_instance_goes_back_once_the_replay_is_enqueued() {
        let (gpu, lane, kernel, at) = rig();
        let cache = SharedGraphCache::<FakeGpu, _>::new();
        let mut b = body(&kernel, at, 1, 1);
        let mut holds: Vec<GraphHold> = (0..4).map(|_| GraphHold::new()).collect();
        cache.warm(&gpu, &lane, KEY, &mut b, &mut holds[0]).unwrap();
        for hold in &mut holds[1..] {
            assert_eq!(
                cache.run(&gpu, &lane, KEY, &mut b, hold).unwrap(),
                Ran::Replayed
            );
        }
        assert_eq!(
            cache.len(),
            1,
            "the holds are alive, and no instance was added"
        );
    }

    /// Workers that run one key at once each get an instance of their own: none waits for another (they are held together
    /// inside the passes, so a wait would never end), and every replay runs.
    #[test]
    fn workers_running_one_key_at_once_each_get_their_own_instance_and_no_one_waits() {
        const N: usize = 4;
        let (gpu, lane, kernel, at) = rig();
        let cache = SharedGraphCache::<FakeGpu, _>::with_capacity(16);
        let barrier = Barrier::new(N);
        let round = |expect: Ran, warm: bool| {
            let ran = std::thread::scope(|scope| {
                let threads: Vec<_> = (0..N)
                    .map(|_| {
                        scope.spawn(|| {
                            let mut hold = GraphHold::new();
                            let mut b = body(&kernel, at, 1, 1);
                            b.inside = Some(&barrier);
                            if warm {
                                cache.warm(&gpu, &lane, KEY, &mut b, &mut hold).unwrap()
                            } else {
                                cache.run(&gpu, &lane, KEY, &mut b, &mut hold).unwrap()
                            }
                        })
                    })
                    .collect();
                threads
                    .into_iter()
                    .map(|t| t.join().unwrap())
                    .collect::<Vec<_>>()
            });
            assert!(ran.iter().all(|r| *r == expect), "{ran:?}");
        };
        // None has an instance: all record at once (a warm-up records without waiting for a second sighting).
        round(Ran::Recorded, true);
        assert_eq!(cache.len(), N);
        // Each takes one of the N instances at the same moment (inside the update pass, all together).
        round(Ran::Replayed, false);
        assert_eq!(cache.len(), N, "no instance was added: each had its own");
        gpu.run_until_idle();
        assert_eq!(word(&gpu, at) as usize, 2 * N, "every launch ran once");
        assert_eq!(
            cache.stats(),
            GraphStats {
                replays: N as u64,
                records: N as u64,
                rerecords: 0,
                directs: 0
            }
        );
    }

    /// A run inside `submit_marked`: a refused marker comes before any recording or replay.
    #[test]
    fn a_busy_marker_ring_refuses_the_run_before_it_records_or_replays() {
        let gpu = FakeGpu::new();
        gpu.define("addv", |_, _, _| Ok(()));
        let lanes = crate::open_shared_lanes(
            &gpu,
            &[(
                "lane",
                StreamOptions {
                    markers: 1,
                    ..StreamOptions::default()
                },
            )],
        )
        .unwrap();
        let lane = lanes.pool.lane(euhedral_graph::LaneId(0)).clone();
        let module = FakeModule::from_source(&gpu, "addv", &Default::default()).unwrap();
        let kernel = module.kernel("addv").unwrap();
        let at = gpu.alloc(4);
        let cache = SharedGraphCache::<FakeGpu, _>::new();
        let mut hold = GraphHold::new();
        let mut b = body(&kernel, at, 1, 1);
        cache.warm(&gpu, &lane, KEY, &mut b, &mut hold).unwrap();
        let ((ran, ()), _marker) = lane
            .submit_marked(|_| {
                cache
                    .run(&gpu, &lane, KEY, &mut b, &mut hold)
                    .map(|r| (r, ()))
            })
            .unwrap();
        assert_eq!(ran, Ran::Replayed);
        let refused = lane.submit_marked(|_| {
            cache
                .run(&gpu, &lane, KEY, &mut b, &mut hold)
                .map(|r| (r, ()))
        });
        assert!(matches!(refused, Err(GpuError::Busy)));
        assert_eq!(cache.stats().replays, 1, "nothing more was replayed");
        gpu.run_until_idle();
        let (r, _) = lane
            .submit_marked(|_| {
                cache
                    .run(&gpu, &lane, KEY, &mut b, &mut hold)
                    .map(|r| (r, ()))
            })
            .unwrap();
        assert_eq!(r.0, Ran::Replayed);
    }

    /// A replay orders itself behind the markers it is given, on the device, before its work.
    #[test]
    fn a_replay_orders_itself_behind_its_waits_before_its_work() {
        let gpu = FakeGpu::new();
        gpu.define("addv", |_, _, _| Ok(()));
        let lanes = crate::open_shared_lanes(
            &gpu,
            &[
                ("a", StreamOptions::default()),
                ("b", StreamOptions::default()),
            ],
        )
        .unwrap();
        let (a, b_lane) = (
            lanes.pool.lane(euhedral_graph::LaneId(0)).clone(),
            lanes.pool.lane(euhedral_graph::LaneId(1)).clone(),
        );
        let module = FakeModule::from_source(&gpu, "addv", &Default::default()).unwrap();
        let kernel = module.kernel("addv").unwrap();
        let at = gpu.alloc(4);
        let cache = SharedGraphCache::<FakeGpu, _>::new();
        let mut hold = GraphHold::new();
        let mut body = body(&kernel, at, 1, 1);
        cache
            .warm(&gpu, &b_lane, KEY, &mut body, &mut hold)
            .unwrap();
        let (_, marker) = a.submit_marked(|_| Ok(())).unwrap();
        cache
            .run_after(&gpu, &b_lane, &[marker], KEY, &mut body, &mut hold)
            .unwrap();
        gpu.run_until_idle();
        let log = gpu.log();
        let wait = log
            .iter()
            .rposition(|op| matches!(op, crate::fake::FakeOp::Await { lane: 1, .. }))
            .expect("lane b waited");
        let launch = log
            .iter()
            .rposition(|op| matches!(op, crate::fake::FakeOp::Launch { lane: 1, .. }))
            .expect("the replay's launch");
        assert!(wait < launch, "{log:?}");
    }
}
