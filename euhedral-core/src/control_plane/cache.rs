use crate::config::{CacheConfig, CloneConfig};
use crate::flow_control::{LatticeEdge, LatticeNode, UpstreamRegistry, route_by_hash};
use crate::frames::{DummyFrame, Failure, Frame, FrameError};
use crate::utils::FlowContext;
use crate::utils::math::ewma;
use euhedral_data_structures::queues::partitioned::{PartitionedMpscQueue, PartitionedProducer};
use euhedral_data_structures::queues::queue::{Consumer, Multi, Single};
use euhedral_hardware_utils::monitor::CoreSnapshot;
use euhedral_hardware_utils::topology::{CpuCacheLayout, Topology};
use euhedral_hardware_utils::util::CpuSet;
use std::mem;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, RwLock, TryLockError, Weak};

type Partition = Consumer<Box<dyn Frame>, Single>;

/// The smallest chunk, in frames, a partition is sized to.
const MIN_CHUNK: usize = 512;
/// The lowest cap factor heavy pressure can push a cache to.
const MIN_CAP_FACTOR: f64 = 0.15;
/// How many times over [`ControlPlaneCache::first_touch`] fills each partition's chunks.
const FIRST_TOUCH_ROUNDS: usize = 3;

/// The [`ControlPlaneCache`]s of one lattice by the CPU they run on, which work stealing visits,
/// with the upstream registry and topology they share.
pub struct CacheDirectory {
    registry: Arc<UpstreamRegistry>,
    topology: Topology,
    slots: Vec<RwLock<Weak<ControlPlaneCache>>>,
}

impl CacheDirectory {
    /// An empty directory with a slot for every CPU of `topology`.
    pub fn new(registry: Arc<UpstreamRegistry>, topology: Topology) -> Arc<Self> {
        let slots = (0..topology.cpu_span())
            .map(|_| RwLock::new(Weak::new()))
            .collect();
        Arc::new(Self {
            registry,
            topology,
            slots,
        })
    }

    pub fn registry(&self) -> &Arc<UpstreamRegistry> {
        &self.registry
    }

    pub fn topology(&self) -> &Topology {
        &self.topology
    }

    /// The live cache at `cursor`, wrapped over the CPU slots.
    fn cache_at(&self, cursor: usize) -> Option<Arc<ControlPlaneCache>> {
        if self.slots.is_empty() {
            return None;
        }
        let slot = &self.slots[cursor % self.slots.len()];
        slot.read()
            .unwrap_or_else(PoisonError::into_inner)
            .upgrade()
    }

    fn claim(&self, cpu: usize, cache: Weak<ControlPlaneCache>) {
        if let Some(slot) = self.slots.get(cpu) {
            *slot.write().unwrap_or_else(PoisonError::into_inner) = cache;
        }
    }

    /// Empties `cpu`'s slot if `cache` still holds it.
    fn withdraw(&self, cpu: usize, cache: &Weak<ControlPlaneCache>) {
        if let Some(slot) = self.slots.get(cpu) {
            let mut slot = slot.write().unwrap_or_else(PoisonError::into_inner);
            if Weak::ptr_eq(&slot, cache) {
                *slot = Weak::new();
            }
        }
    }
}

/// A core-local cache of frames, and the demand that keeps it filled.
///
/// The cache is a lattice node: placed below its core's edge with [`attach`](Self::attach), it
/// queues the frames pushed to it, and [`pull`](Self::pull) fills it through the edges above.
/// [`request`](Self::request) asks the calling worker's upstreams for a batch sized to the cache's
/// capacity, which shrinks under CPU pressure. The owner takes frames out with
/// [`drain`](Self::drain).
///
/// The queue is split into partitions by routing hash, so frames of one ordered stream stay in one
/// partition, in order. With work stealing on, other caches of the lattice may take a partition's
/// unordered frames from the front with [`work_steal`](Self::work_steal); a partition is held by one
/// consumer at a time, and the others pass it over.
///
/// Build the cache on its owner's CPU and call [`first_touch`](Self::first_touch) there before it
/// receives work, so its queue's memory lands on that CPU's NUMA node.
///
/// A closed cache fails new frames and leaves the lattice's work stealing, but keeps what it holds
/// for its owner to drain.
pub struct ControlPlaneCache {
    name: String,
    core: usize,
    cpu: usize,
    cpus: CpuSet,
    config: CacheConfig,
    directory: Arc<CacheDirectory>,
    edge: LatticeEdge,
    queue: PartitionedMpscQueue<Box<dyn Frame>>,
    producer: PartitionedProducer<Box<dyn Frame>, Multi>,
    partitions: Vec<Mutex<Partition>>,
    chunk_size: usize,
    frame_quota: u64,
    pull_multiplier: u64,
    socket_count: u64,
    /// Frames queued, give or take those in flight; may dip below zero for a moment.
    count: AtomicI64,
    cap_factor: AtomicU64,
    retired: AtomicBool,
    primed: AtomicBool,
    this: Weak<ControlPlaneCache>,
}

impl ControlPlaneCache {
    /// A cache for the clone `clone` places, run from `cpu`, in `directory`'s lattice.
    /// `smt_enabled` says whether the core's SMT siblings each run a cache, which then share the
    /// core's L1 and L2.
    pub fn new(
        config: &CacheConfig,
        clone: &CloneConfig,
        cpu: usize,
        smt_enabled: bool,
        directory: &Arc<CacheDirectory>,
    ) -> Arc<Self> {
        let topology = directory.topology();
        let partitions = config.partitions().get();
        let chunk_size = chunk_size(config, topology.cache_layout(cpu), smt_enabled);
        let queue = PartitionedMpscQueue::<Box<dyn Frame>>::with_pool(
            partitions,
            chunk_size,
            config.max_pooled_chunks(),
        );
        let socket_cores = topology
            .core_info(clone.core_id())
            .and_then(|core| topology.socket_info(core.socket))
            .map_or(0, |socket| socket.cores.len() as u64);
        // 37.5% of the socket's cores.
        let pull_multiplier = ((socket_cores * 3) >> 3).max(2);
        let name = format!(
            "{}-ControlPlaneCache-{}",
            clone.shard_name(),
            clone.core_id()
        );
        log::debug!(
            target: &name,
            "partitions: {partitions}, partition chunk size: {chunk_size}, capacity: {}",
            partitions * chunk_size
        );

        let cache = Arc::new_cyclic(|this: &Weak<Self>| {
            let node: Weak<dyn LatticeNode> = this.clone();
            Self {
                core: clone.core_id(),
                cpu,
                cpus: clone.effective_cpus().clone(),
                config: config.clone(),
                directory: directory.clone(),
                edge: LatticeEdge::for_node(directory.registry(), Default::default(), node),
                producer: queue.producer(),
                partitions: (0..partitions)
                    .map(|partition| Mutex::new(queue.partition(partition).consumer()))
                    .collect(),
                chunk_size,
                frame_quota: (chunk_size * partitions) as u64,
                pull_multiplier,
                socket_count: topology.socket_count().max(1) as u64,
                count: AtomicI64::new(0),
                cap_factor: AtomicU64::new(1.0f64.to_bits()),
                retired: AtomicBool::new(false),
                primed: AtomicBool::new(false),
                queue,
                this: this.clone(),
                name,
            }
        });
        directory.claim(cpu, Arc::downgrade(&cache));
        cache
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn core(&self) -> usize {
        self.core
    }

    /// The CPU the cache runs from, and its slot in the lattice's work stealing.
    pub fn cpu(&self) -> usize {
        self.cpu
    }

    /// Frames per chunk of each partition.
    pub fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    /// What the cache's first chunks hold: the most it asks for at full capacity. The queue still
    /// grows past it.
    pub fn frame_quota(&self) -> u64 {
        self.frame_quota
    }

    /// Retired chunks kept for reuse, over every partition: with pooling configured, the chunks
    /// [`first_touch`](Self::first_touch) wrote.
    pub fn pooled_chunks(&self) -> usize {
        (0..self.queue.partitions())
            .map(|partition| self.queue.partition(partition).pooled_chunks())
            .sum()
    }

    /// The share of the frame quota the cache currently asks for, in `[0.15, 1]`.
    pub fn cap_factor(&self) -> f64 {
        f64::from_bits(self.cap_factor.load(Ordering::Acquire))
    }

    /// The most frames the cache currently asks for.
    pub fn max_local_cache_count(&self) -> u64 {
        (self.frame_quota as f64 * self.cap_factor()) as u64
    }

    /// The frames queued.
    pub fn local_cache_count(&self) -> u64 {
        self.count.load(Ordering::Acquire).max(0) as u64
    }

    pub fn is_drained(&self) -> bool {
        self.count.load(Ordering::Acquire) <= 0
    }

    /// While set, demand from [`pull`](Self::pull) is ignored and new upstream handles wait.
    pub fn set_drain_mode(&self, drain: bool) {
        self.edge.drain_flag().store(drain, Ordering::Release);
    }

    /// Fills every partition three chunks over with dummy frames, then clears them and resets the
    /// cache, so the queue's chunks are written first from the calling thread: call it once, on the
    /// owner's CPU, before the cache starts receiving work. Chunks beyond each partition's first stay
    /// only if the config pools them. Does nothing after the first call.
    pub fn first_touch(&self) {
        if self.primed.swap(true, Ordering::AcqRel) {
            return;
        }
        for _ in 0..FIRST_TOUCH_ROUNDS {
            for partition in 0..self.partitions.len() {
                for _ in 0..self.chunk_size {
                    if self
                        .producer
                        .offer_to(partition, Box::new(DummyFrame::INSTANCE))
                        .is_err()
                    {
                        unreachable!("an unbounded partition takes every frame");
                    }
                }
            }
        }
        for partition in &self.partitions {
            hold(partition).clear();
        }
        self.cap_factor.store(1.0f64.to_bits(), Ordering::Release);
    }

    /// Places the cache below `edge`, so the frames pushed into `edge` arrive here and pulls climb
    /// through it.
    pub fn attach(&self, edge: &LatticeEdge) {
        let this = self.this.upgrade().expect("a live cache is shared");
        edge.add_downstream_node(this);
    }

    /// Takes up to `limit` frames from the upstreams above into the cache and returns how many
    /// were taken.
    pub fn pull(&self, limit: u64) -> u64 {
        if limit == 0 {
            return 0;
        }
        let mut added = 0;
        let taken = self.edge.pull(
            &mut |frame| {
                if self.offer(frame) {
                    added += 1;
                }
            },
            &mut |_| false,
            limit,
        );
        self.count.fetch_add(added, Ordering::AcqRel);
        taken
    }

    /// Takes up to `limit` frames from the calling worker's upstreams straight into `consume`,
    /// bypassing the cache, and stops before the first ordered frame.
    pub fn upstream_pull(&self, mut consume: impl FnMut(Box<dyn Frame>), limit: u64) -> u64 {
        if limit == 0 {
            return 0;
        }
        self.directory
            .registry()
            .with_thread_queue(|queue| {
                queue.pull(
                    &mut consume,
                    &mut |frame| frame.header().is_ordered(),
                    limit,
                )
            })
            .unwrap_or(0)
    }

    /// Asks the calling worker's upstreams for a batch: `batch_size` scaled by the socket's size,
    /// capped at what the cache currently asks for, from every socket. Resets the worker's
    /// [`FlowContext`] counters and records the demand as its original request.
    pub fn request(&self, batch_size: u64) {
        let demand = batch_size
            .saturating_mul(self.pull_multiplier)
            .min(self.max_local_cache_count());
        FlowContext::with(|context| {
            if let Some(context) = context {
                context.clear_counters();
                context.set_original_request(demand);
            }
        });
        let demand = demand.saturating_mul(self.socket_count);
        self.directory
            .registry()
            .with_thread_queue(|queue| queue.request(demand));
    }

    /// Moves up to `limit` frames into `consume`, partition by partition, and returns how many.
    pub fn drain(&self, consume: impl FnMut(Box<dyn Frame>), limit: u64) -> u64 {
        self.drain_while(consume, |_| false, limit)
    }

    /// Like [`drain`](Self::drain), but leaves each partition at its first frame `stop` matches.
    ///
    /// The partitions are walked in turn, skipping any another consumer holds, and walked again
    /// while a walk collects more than the ring-walk reset threshold.
    pub fn drain_while(
        &self,
        mut consume: impl FnMut(Box<dyn Frame>),
        mut stop: impl FnMut(&dyn Frame) -> bool,
        limit: u64,
    ) -> u64 {
        if limit == 0 || self.count.load(Ordering::Acquire) <= 0 {
            return 0;
        }
        let mut taken = Taken {
            count: &self.count,
            frames: 0,
        };
        loop {
            let walk_start = taken.frames;
            for partition in &self.partitions {
                let remaining = limit - taken.frames;
                if remaining == 0 {
                    break;
                }
                let Some(mut partition) = try_hold(partition) else {
                    continue;
                };
                partition.drain_while(
                    |frame| {
                        taken.frames += 1;
                        consume(frame);
                    },
                    |frame| stop(&**frame),
                    remaining,
                );
            }
            let walked = taken.frames - walk_start;
            if taken.frames >= limit || walked <= self.config.ring_walk_reset_threshold() {
                return taken.frames;
            }
        }
    }

    /// Takes up to `limit` unordered frames from the front of the partitions of the cache at
    /// `cursor` in the lattice's work stealing, which wraps over the CPUs. Nothing without work
    /// stealing.
    pub fn work_steal(
        &self,
        consume: impl FnMut(Box<dyn Frame>),
        limit: u64,
        cursor: usize,
    ) -> u64 {
        if !self.config.work_steal() {
            return 0;
        }
        match self.directory.cache_at(cursor) {
            Some(target) => target.drain_while(consume, |frame| frame.header().is_ordered(), limit),
            None => 0,
        }
    }

    /// Moves the cap factor toward what the core's CPU pressure allows: quickly down, slowly up.
    pub fn update(&self, snapshot: &CoreSnapshot) {
        let pressure = self
            .cpus
            .iter()
            .filter_map(|cpu| snapshot.cpu_snapshots.get(cpu)?.as_ref())
            .map(|cpu| cpu.pressure)
            .fold(0.0, f64::max)
            .clamp(0.0, 1.0);
        let pressure = (pressure * 10_000.0).round() / 10_000.0;
        let target = 1.0 - 0.85 * pressure;
        let current = self.cap_factor();
        let alpha = if target < current { 0.2 } else { 0.02 };
        let next = ewma(current, target, alpha).clamp(MIN_CAP_FACTOR, 1.0);
        self.cap_factor.store(next.to_bits(), Ordering::Release);
    }

    /// Drops every queued frame and restores full capacity, for a new trial once ingest is frozen.
    /// Returns how many frames were dropped.
    pub fn reset_for_next_trial(&self) -> u64 {
        let cleared = self.local_cache_count();
        for partition in &self.partitions {
            hold(partition).clear();
        }
        self.count.store(0, Ordering::Release);
        self.cap_factor.store(1.0f64.to_bits(), Ordering::Release);
        cleared
    }

    /// Retires the cache: new frames fail, other caches stop stealing from it, and its edge
    /// closes. What it holds stays for its owner to drain.
    pub fn close(&self) {
        if self.retired.swap(true, Ordering::AcqRel) {
            return;
        }
        self.directory.withdraw(self.cpu, &self.this);
        self.edge.close_edge();
    }

    /// Queues `frame` in its partition, or fails it once the cache is retired.
    fn offer(&self, frame: Box<dyn Frame>) -> bool {
        if self.retired.load(Ordering::Acquire) {
            frame.do_finally_with_error(FrameError::failed("the cache retired"));
            return false;
        }
        let partition = route_by_hash(&*frame, self.partitions.len());
        if self.producer.offer_to(partition, frame).is_err() {
            unreachable!("an unbounded partition takes every frame");
        }
        true
    }
}

impl LatticeNode for ControlPlaneCache {
    fn edge(&self) -> &LatticeEdge {
        &self.edge
    }

    /// Queues `frame` and counts it toward the calling worker's request.
    fn push(&self, frame: Box<dyn Frame>) {
        if self.offer(frame) {
            self.count.fetch_add(1, Ordering::AcqRel);
            FlowContext::with(|context| {
                if let Some(context) = context {
                    context.add_satisfied_request(1);
                }
            });
        }
    }

    /// The cache is the floor of its branch: errors stop here.
    fn on_error(&self, error: Failure) {
        log::debug!(target: &self.name, "ignoring an upstream error: {error}");
    }

    fn close(&self) {
        ControlPlaneCache::close(self);
    }

    fn is_closed(&self) -> bool {
        self.retired.load(Ordering::Acquire)
    }
}

/// Subtracts the frames a drain took from the count when dropped, so a consumer that panics
/// still leaves the count right.
struct Taken<'a> {
    count: &'a AtomicI64,
    frames: u64,
}

impl Drop for Taken<'_> {
    fn drop(&mut self) {
        self.count.fetch_sub(self.frames as i64, Ordering::AcqRel);
    }
}

/// The partition, unless another consumer holds it. A consumer that panicked mid-drain leaves the
/// partition intact, so its lock is taken over.
fn try_hold(partition: &Mutex<Partition>) -> Option<MutexGuard<'_, Partition>> {
    match partition.try_lock() {
        Ok(partition) => Some(partition),
        Err(TryLockError::Poisoned(poisoned)) => Some(poisoned.into_inner()),
        Err(TryLockError::WouldBlock) => None,
    }
}

fn hold(partition: &Mutex<Partition>) -> MutexGuard<'_, Partition> {
    partition.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Frames per chunk: the configured share of the L1 and L2 the cache gets, split over the
/// partitions and rounded down to a power of two, at least [`MIN_CHUNK`].
fn chunk_size(config: &CacheConfig, layout: Option<&CpuCacheLayout>, smt_enabled: bool) -> usize {
    let Some(layout) = layout else {
        return MIN_CHUNK;
    };
    let share = |bytes: u64, sharers: usize| {
        if sharers > 2 || smt_enabled {
            bytes / sharers.max(1) as u64
        } else {
            bytes
        }
    };
    let bytes = share(layout.bytes_l2, layout.shares_l2())
        .saturating_add(share(layout.bytes_l1, layout.shares_l1()));
    let per_partition = bytes / config.partitions().get() as u64;
    let budgeted = (per_partition as f64 * config.memory_budget()) as u64;
    let power = if budgeted == 0 {
        0
    } else {
        1 << budgeted.ilog2()
    };
    let frames = power / mem::size_of::<Box<dyn Frame>>() as u64;
    usize::try_from(frames).unwrap_or(usize::MAX).max(MIN_CHUNK)
}

#[cfg(test)]
mod tests;
