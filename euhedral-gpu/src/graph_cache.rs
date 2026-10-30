//! A lane's cache of recorded graphs: the same sequence of operations run for each unit of work, recorded once and updated and
//! replayed after.
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

use crate::{Enqueue, Gpu, GpuError, Graph, Sequence, Stream};

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
