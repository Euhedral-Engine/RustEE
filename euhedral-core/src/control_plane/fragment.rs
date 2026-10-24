use super::cache::{CacheDirectory, ControlPlaneCache};
use super::decision_tree::{BodyCost, DecisionTree, ExecutionPath};
use crate::config::{CloneConfig, CloneLivenessRegistry, FragmentConfig};
use crate::flow_control::{LatticeEdge, LatticeHotSource, UpstreamQueue, UpstreamRegistry};
use crate::frames::Frame;
use crate::generics::{CloneableObject, LatticeSource};
use crate::utils::{FlowContext, StopWatch};
use euhedral_hardware_utils::affinity::{self, ExecutorError, PinnedThreadExecutor};
use euhedral_hardware_utils::monitor::CoreSnapshot;
use std::error::Error;
use std::fmt;
use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, Weak};
use std::thread::{self, Thread};
use std::time::{Duration, Instant};

/// How long a worker with neither upstreams nor cached frames parks.
const EMPTY_PARK: Duration = Duration::from_micros(15);
/// The last snapshot timestamp before any snapshot is followed: older than any real one.
const NO_SNAPSHOT: i64 = i64::MIN;
/// How often [`ControlPlaneFragment::reset`] checks on the worker.
const RESET_POLL: Duration = Duration::from_micros(5);

/// Why a fragment could not be built.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FragmentError {
    /// The clone config was not built by a lattice's shard, so it names no cache directory.
    NoCacheDirectory,
    /// The clone config has no effective CPU.
    NoCpu,
    /// The lattice's topology does not know the CPU.
    UnknownCpu(usize),
    /// The CPU's pinned executor could not be had.
    Executor(ExecutorError),
}

impl fmt::Display for FragmentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoCacheDirectory => f.write_str("the clone config names no cache directory"),
            Self::NoCpu => f.write_str("the clone config has no effective CPU"),
            Self::UnknownCpu(cpu) => write!(f, "CPU {cpu} is not in the lattice's topology"),
            Self::Executor(error) => write!(f, "no pinned executor: {error}"),
        }
    }
}

impl Error for FragmentError {}

/// The worker that runs one core's frames: it owns a thread pinned to one CPU of the core, which
/// takes frames from the core's [`ControlPlaneCache`] and the worker's upstreams and pushes them
/// out through [`output`](CloneableObject::output).
///
/// Each cycle the worker runs cached frames first, then picks a path with its policy: run frames
/// straight from upstream, stage them in the cache by requesting a batch, or sit the cycle out.
/// When that finds nothing it steals from the next cache of the lattice, and failing that it parks.
/// With a max batch of [`FragmentConfig::POLL_MODE_MAX_BATCH`] or less the worker polls its cache
/// and the caches it steals from instead of draining them, so a frame that blocks holds up nothing
/// another worker could run ([`ControlPlaneCache::poll`]).
/// Batches grow and shrink with the frames' service time, within a cap that falls as the CPU's
/// pressure rises.
///
/// With SMT enabled and two effective CPUs, the fragment runs a second worker, its SMT buddy, on
/// the core's second CPU. The core's frames reach this fragment's cache, which hands those with an
/// odd routing hash to the buddy's. The buddy follows this fragment through its lifecycle and has
/// its own [`output`](CloneableObject::output).
///
/// Build fragments by cloning a [`FragmentConfig`] onto a core.
pub struct ControlPlaneFragment {
    name: String,
    core: usize,
    cpu: usize,
    socket: usize,
    p_core: bool,
    config: FragmentConfig,
    liveness: Arc<CloneLivenessRegistry>,
    registry: Arc<UpstreamRegistry>,
    cache: Arc<ControlPlaneCache>,
    output: Arc<LatticeHotSource>,
    body_cost: Arc<BodyCost>,
    executor: PinnedThreadExecutor,
    buddy: Option<Arc<ControlPlaneFragment>>,
    /// Serializes start, reset and close.
    lifecycle: Mutex<()>,
    /// The timestamp of the last CPU snapshot the batch cap followed.
    /// [`NO_SNAPSHOT`] until one is followed.
    last_snapshot: AtomicI64,
    running: AtomicBool,
    close_requested: AtomicBool,
    initialized: AtomicBool,
    worker: Mutex<Option<Thread>>,
    reset_requested: AtomicU64,
    reset_completed: AtomicU64,
    reset_cleared: AtomicU64,
    in_progress: AtomicU64,
    adaptive_batch_cap: AtomicU64,
    this: Weak<ControlPlaneFragment>,
}

impl ControlPlaneFragment {
    /// A fragment placed by `clone`, which a lattice's shard built: on the first effective CPU, with
    /// an SMT buddy on the second if `config` enables SMT.
    pub fn new(config: &FragmentConfig, clone: &CloneConfig) -> Result<Arc<Self>, FragmentError> {
        let directory = clone
            .cache_directory()
            .ok_or(FragmentError::NoCacheDirectory)?;
        let cpus = clone.cpus();
        let &cpu = cpus.first().ok_or(FragmentError::NoCpu)?;
        let buddy = match cpus.get(1) {
            Some(&second) if config.smt_enabled() => {
                Some(Self::on_cpu(config, clone, directory, second, None)?)
            }
            _ => None,
        };
        Self::on_cpu(config, clone, directory, cpu, buddy)
    }

    fn on_cpu(
        config: &FragmentConfig,
        clone: &CloneConfig,
        directory: &Arc<CacheDirectory>,
        cpu: usize,
        buddy: Option<Arc<Self>>,
    ) -> Result<Arc<Self>, FragmentError> {
        let topology = directory.topology();
        let info = topology
            .cpu_info(cpu)
            .ok_or(FragmentError::UnknownCpu(cpu))?;
        let p_core = topology
            .core_info(info.core)
            .is_some_and(|core| core.p_core);
        let name = format!("{}-Worker-{}", clone.shard_name(), clone.core_id());
        let executor =
            PinnedThreadExecutor::get_or_create(cpu, &name).map_err(FragmentError::Executor)?;
        let cache = ControlPlaneCache::new(
            config.cache_config(),
            clone,
            cpu,
            config.smt_enabled(),
            directory,
        );
        if let Some(buddy) = &buddy {
            cache.pair_with(buddy.cache.clone());
        }
        let body_cost = Arc::new(BodyCost::new());
        let output = Arc::new(timed_output(body_cost.clone()));
        let initial_cap = initial_batch_cap(config, &cache);
        Ok(Arc::new_cyclic(|this| Self {
            name,
            core: clone.core_id(),
            cpu,
            socket: info.socket,
            p_core,
            config: config.clone(),
            liveness: clone.liveness_registry().clone(),
            registry: directory.registry().clone(),
            cache,
            output,
            body_cost,
            executor,
            buddy,
            lifecycle: Mutex::new(()),
            last_snapshot: AtomicI64::new(NO_SNAPSHOT),
            running: AtomicBool::new(false),
            close_requested: AtomicBool::new(false),
            initialized: AtomicBool::new(false),
            worker: Mutex::new(None),
            reset_requested: AtomicU64::new(0),
            reset_completed: AtomicU64::new(0),
            reset_cleared: AtomicU64::new(0),
            in_progress: AtomicU64::new(0),
            adaptive_batch_cap: AtomicU64::new(initial_cap),
            this: this.clone(),
        }))
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// The CPU the worker is pinned to.
    pub fn cpu(&self) -> usize {
        self.cpu
    }

    pub fn socket(&self) -> usize {
        self.socket
    }

    /// Whether the worker's core is a performance core.
    pub fn is_p_core(&self) -> bool {
        self.p_core
    }

    pub fn config(&self) -> &FragmentConfig {
        &self.config
    }

    pub fn cache(&self) -> &Arc<ControlPlaneCache> {
        &self.cache
    }

    /// The worker on the core's second CPU, if SMT gave the fragment one.
    pub fn smt_buddy(&self) -> Option<&Arc<ControlPlaneFragment>> {
        self.buddy.as_ref()
    }

    /// The largest batch the worker currently runs: lower the more pressure its CPU is under.
    pub fn batch_cap(&self) -> u64 {
        self.adaptive_batch_cap.load(Ordering::Acquire)
    }

    /// Waits up to `timeout` for the executors of both workers to finish their tasks.
    #[cfg(test)]
    pub(crate) fn await_executors(&self, timeout: Duration) -> bool {
        self.executor.await_termination(timeout)
            && self
                .buddy
                .as_ref()
                .is_none_or(|buddy| buddy.await_executors(timeout))
    }

    fn lock_lifecycle(&self) -> MutexGuard<'_, ()> {
        lock(&self.lifecycle)
    }

    fn initial_batch_cap(&self) -> u64 {
        initial_batch_cap(&self.config, &self.cache)
    }

    /// Stops the worker without waiting for it: it finishes its cycle and exits.
    fn request_stop(&self) {
        self.close_requested.store(true, Ordering::Release);
        self.running.store(false, Ordering::Release);
        self.unpark_worker();
    }

    fn unpark_worker(&self) {
        let worker = lock(&self.worker).clone();
        if let Some(worker) = worker
            && worker.id() != thread::current().id()
        {
            worker.unpark();
        }
    }

    /// Clears the cache and restores the batch cap; the worker clears its own state.
    fn reset_shared_state(&self) -> u64 {
        let cleared = self.cache.reset_for_next_trial();
        self.adaptive_batch_cap
            .store(self.initial_batch_cap(), Ordering::Release);
        self.last_snapshot.store(NO_SNAPSHOT, Ordering::Release);
        cleared
    }

    /// The pinned worker: registers with the lattice's registry, then runs cycles until stopped.
    fn run_worker(self: Arc<Self>) {
        let slot = self.liveness.slot(self.cpu);
        if let Some(slot) = slot {
            slot.enter(thread::current());
        }
        *lock(&self.worker) = Some(thread::current());
        let mut exit = WorkerExit {
            fragment: &self,
            registered: false,
        };
        if self.close_requested.load(Ordering::Acquire) {
            return;
        }
        match affinity::current_cpu_info() {
            Some(origin) if origin.core != self.core => log::warn!(
                target: &self.name,
                "pinned to core {} CPU {} but running on {origin:?}",
                self.core,
                self.cpu
            ),
            _ => log::debug!(
                target: &self.name,
                "pinned to core {} CPU {}, P-core: {}",
                self.core,
                self.cpu,
                self.p_core
            ),
        }
        affinity::set_timer_resolution(1);
        self.registry.register();
        exit.registered = true;
        let looped = panic::catch_unwind(AssertUnwindSafe(|| self.run_owner_loop()));
        if looped.is_err() {
            log::error!(target: &self.name, "the worker loop panicked; the worker stops");
        }
    }

    fn run_owner_loop(&self) {
        let mut policy =
            DecisionTree::calibrated(self.config.idle_policy().clone(), self.body_cost.clone());
        let mut cycle = Cycle::new(self.cpu.wrapping_add(1));
        FlowContext::initialize();
        self.initialized.store(true, Ordering::Release);
        while self.running.load(Ordering::Acquire) && !affinity::is_interrupted() {
            self.run_cycle(&mut policy, &mut cycle);
        }
    }

    fn run_cycle(&self, policy: &mut DecisionTree, cycle: &mut Cycle) {
        self.handle_reset_request(policy, cycle);

        let half_life = policy.idle_policy().contention_half_life();
        let (contention, upstreams) = self
            .with_queue(|queue| {
                (
                    queue.adaptive_contention(cycle.now, half_life),
                    queue.cached_upstream_count(),
                )
            })
            .unwrap_or_default();
        if upstreams == 0 && self.cache.is_drained() {
            thread::park_timeout(EMPTY_PARK);
            return;
        }

        let mut limit = cycle.batch_size.saturating_sub(cycle.completed);
        let mut run = Run::default();
        let polls = self.config.polls();
        if limit > 0 && !self.cache.is_drained() {
            let count = self.timed(&mut run, cycle, |output| {
                if polls {
                    self.cache.poll(output, limit)
                } else {
                    self.cache.drain(output, limit)
                }
            });
            limit -= count;
        }

        let (productive, upstreams) = self
            .with_queue(|queue| {
                (
                    queue.productive_handle_count(),
                    queue.cached_upstream_count(),
                )
            })
            .unwrap_or_default();
        let registered = cycle.registered_workers;
        let rank = self.registry.thread_rank(self.cpu);
        let path = if policy.should_idle(contention, productive, registered, rank) {
            ExecutionPath::Idle
        } else {
            policy.select_path(productive, upstreams, registered, contention)
        };

        match path {
            ExecutionPath::Direct => {
                if limit > 0 {
                    self.timed(&mut run, cycle, |output| {
                        self.cache.upstream_pull(output, limit)
                    });
                }
                if run.processed == 0 {
                    self.cache.request(cycle.batch_size);
                }
            }
            ExecutionPath::Staged
                if limit > 0 && self.cache.local_cache_count() <= cycle.batch_size =>
            {
                self.cache.request(cycle.batch_size);
                cycle.now = Instant::now();
            }
            _ => {}
        }

        if run.processed == 0 {
            let cursor = cycle.neighbor_cursor;
            cycle.neighbor_cursor = cursor.wrapping_add(1);
            self.timed(&mut run, cycle, |output| {
                if polls {
                    self.cache.work_steal_polled(output, limit, cursor)
                } else {
                    self.cache.work_steal(output, limit, cursor)
                }
            });
        }

        if run.processed == 0 {
            if path == ExecutionPath::Idle {
                let park = self
                    .with_queue(|queue| policy.idle_park(queue, cycle.now, registered, productive))
                    .unwrap_or(policy.idle_policy().idle_park());
                thread::park_timeout(park);
                cycle.now = Instant::now();
                thread::yield_now();
                return;
            }
            let park = policy.idle_policy().unproductive_park();
            if !park.is_zero()
                && self.with_queue(|queue| queue.productive_handle_count()) == Some(0)
            {
                thread::park_timeout(park);
                cycle.now = Instant::now();
            } else {
                thread::yield_now();
            }
            return;
        }

        cycle.complete(&run, policy, self.batch_cap(), || {
            self.registry.thread_count()
        });
        std::hint::spin_loop();
    }

    /// Runs `execute` with a consumer that pushes frames out, counted as in progress, and adds what
    /// it ran to `run`.
    fn timed(
        &self,
        run: &mut Run,
        cycle: &mut Cycle,
        execute: impl FnOnce(&mut dyn FnMut(Box<dyn Frame>)) -> u64,
    ) -> u64 {
        let _in_progress = InProgress::enter(&self.in_progress);
        let start = Instant::now();
        let count = execute(&mut |frame| self.output.accept(frame));
        let end = Instant::now();
        if count > 0 {
            run.processed += count;
            run.elapsed += end - start;
        }
        cycle.now = end;
        count
    }

    fn with_queue<R>(&self, f: impl FnOnce(&mut UpstreamQueue) -> R) -> Option<R> {
        self.registry.with_thread_queue(f)
    }

    fn handle_reset_request(&self, policy: &mut DecisionTree, cycle: &mut Cycle) {
        let requested = self.reset_requested.load(Ordering::Acquire);
        if requested <= self.reset_completed.load(Ordering::Relaxed) {
            return;
        }
        let cleared = self.reset_shared_state();
        *cycle = Cycle::new(cycle.neighbor_cursor);
        self.with_queue(|queue| queue.reset_for_next_trial());
        policy.reset();
        FlowContext::with(|context| {
            if let Some(context) = context {
                context.clear_counters();
            }
        });
        self.reset_cleared.store(cleared, Ordering::Release);
        self.reset_completed.store(requested, Ordering::Release);
    }
}

impl CloneableObject for ControlPlaneFragment {
    /// A fragment placed by `config` with this fragment's settings.
    ///
    /// # Panics
    ///
    /// If the fragment cannot be built; see [`FragmentError`].
    fn clone_object(&self, config: &CloneConfig) -> Arc<dyn CloneableObject> {
        self.config.clone_object(config)
    }

    /// Starts the worker, and the buddy's, on their pinned executors. Does nothing while running.
    ///
    /// # Panics
    ///
    /// If the fragment is closed, or its executor is closed or rejects the worker.
    fn start(&self) {
        let _lifecycle = self.lock_lifecycle();
        assert!(
            !self.close_requested.load(Ordering::Acquire),
            "{}: cannot start a closed fragment",
            self.name
        );
        if self
            .running
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return;
        }
        if let Err(error) = self.executor.start(&self.name) {
            self.running.store(false, Ordering::Release);
            panic!("{}: cannot restart the executor: {error}", self.name);
        }
        if let Some(buddy) = &self.buddy {
            buddy.start();
        }
        let this = self.this.upgrade().expect("a live fragment is shared");
        if let Err(rejected) = self.executor.execute(move || this.run_worker()) {
            self.running.store(false, Ordering::Release);
            panic!(
                "{}: the executor rejected the worker: {rejected}",
                self.name
            );
        }
    }

    /// Fills the buddy's cache and then this one's from the calling thread; see
    /// [`ControlPlaneCache::first_touch`].
    fn first_touch(&self) {
        if let Some(buddy) = &self.buddy {
            buddy.first_touch();
        }
        self.cache.first_touch();
    }

    fn is_started(&self) -> bool {
        self.running.load(Ordering::Acquire)
            && self.buddy.as_ref().is_none_or(|buddy| buddy.is_started())
    }

    /// Whether the worker, and the buddy's, has registered and runs cycles.
    fn ready(&self) -> bool {
        self.running.load(Ordering::Acquire)
            && self.initialized.load(Ordering::Acquire)
            && self.buddy.as_ref().is_none_or(|buddy| buddy.ready())
    }

    /// Sets the batch cap from the pressure on the worker's CPU, and passes the snapshot to the
    /// buddy and the cache. A snapshot without this CPU, or no newer than the last one followed,
    /// changes nothing here.
    ///
    /// Only the lattice's hardware monitor thread calls it.
    fn update(&self, snapshot: &CoreSnapshot) {
        if let Some(buddy) = &self.buddy {
            buddy.update(snapshot);
        }
        let Some(cpu) = snapshot
            .cpu_snapshots
            .get(self.cpu)
            .and_then(Option::as_ref)
        else {
            return;
        };
        if cpu.last_usage_ns <= self.last_snapshot.load(Ordering::Acquire) {
            return;
        }
        self.last_snapshot
            .store(cpu.last_usage_ns, Ordering::Release);
        let max = self.initial_batch_cap();
        let cap = (max as f64 - cpu.pressure * (max - 2) as f64).round() as u64;
        self.adaptive_batch_cap
            .store(cap.clamp(2, max), Ordering::Release);
        self.cache.update(snapshot);
    }

    /// Attaches the cache below `edge`.
    fn input_edge(self: Arc<Self>, edge: Arc<LatticeEdge>) {
        self.cache.attach(&edge);
    }

    /// Where the worker pushes the frames it runs. The buddy has its own.
    fn output(&self) -> Option<Arc<dyn LatticeSource>> {
        Some(self.output.clone())
    }

    /// Whether the caches are empty and neither worker holds frames.
    fn is_drained(&self) -> bool {
        self.cache.is_drained()
            && self.in_progress.load(Ordering::Acquire) == 0
            && self.buddy.as_ref().is_none_or(|buddy| buddy.is_drained())
    }

    fn set_drain_mode(&self, drain: bool) {
        self.cache.set_drain_mode(drain);
        if let Some(buddy) = &self.buddy {
            buddy.set_drain_mode(drain);
        }
    }

    /// Drops the cached frames and restores the batch cap. A running worker does it itself and
    /// also starts its policy, cycle and upstream queue over; this waits for it until `deadline`.
    /// Returns the frames dropped, the buddy's included.
    ///
    /// # Panics
    ///
    /// If a running worker has not reset by `deadline`.
    fn reset(&self, deadline: Instant) -> u64 {
        let _lifecycle = self.lock_lifecycle();
        let cleared = if self.running.load(Ordering::Acquire) {
            let request = self.reset_requested.fetch_add(1, Ordering::AcqRel) + 1;
            self.unpark_worker();
            while self.reset_completed.load(Ordering::Acquire) < request
                && self.running.load(Ordering::Acquire)
                && Instant::now() < deadline
            {
                thread::park_timeout(RESET_POLL);
            }
            assert!(
                self.reset_completed.load(Ordering::Acquire) >= request,
                "{}: timed out resetting the fragment on core {}",
                self.name,
                self.core
            );
            self.reset_cleared.load(Ordering::Acquire)
        } else {
            self.reset_shared_state()
        };
        cleared + self.buddy.as_ref().map_or(0, |buddy| buddy.reset(deadline))
    }

    /// Closes the executor, and the buddy's.
    fn dump_locks(&self) {
        self.executor.close();
        if let Some(buddy) = &self.buddy {
            buddy.dump_locks();
        }
    }

    fn core(&self) -> Option<usize> {
        Some(self.core)
    }

    fn liveness_registry(&self) -> Option<Arc<CloneLivenessRegistry>> {
        Some(self.liveness.clone())
    }

    /// Stops both workers, then closes the buddy, the executor and the cache. The workers exit
    /// after their current cycle; what the cache holds stays in it.
    fn close(&self) {
        let _lifecycle = self.lock_lifecycle();
        self.request_stop();
        if let Some(buddy) = &self.buddy {
            buddy.request_stop();
            buddy.close();
        }
        self.executor.close();
        self.cache.close();
        log::debug!(target: &self.name, "closed");
    }
}

impl CloneableObject for FragmentConfig {
    /// A [`ControlPlaneFragment`] placed by `config`.
    ///
    /// # Panics
    ///
    /// If the fragment cannot be built; see [`FragmentError`].
    fn clone_object(&self, config: &CloneConfig) -> Arc<dyn CloneableObject> {
        match ControlPlaneFragment::new(self, config) {
            Ok(fragment) => fragment,
            Err(error) => panic!(
                "cannot build the fragment of core {}: {error}",
                config.core_id()
            ),
        }
    }
}

/// The largest batch: the configured maximum, capped by the cache's capacity, and at least two.
fn initial_batch_cap(config: &FragmentConfig, cache: &ControlPlaneCache) -> u64 {
    config
        .max_batch_size()
        .get()
        .min(cache.frame_quota())
        .max(2)
}

/// An output that times a sample of the frame bodies it pushes into `body_cost`.
fn timed_output(body_cost: Arc<BodyCost>) -> LatticeHotSource {
    let watch = Arc::new(Mutex::new(StopWatch::new()));
    let started = watch.clone();
    LatticeHotSource::new()
        .before_each(move |_| lock(&started).start())
        .after_each(move || {
            if let Some(elapsed) = lock(&watch).stop() {
                body_cost.record(elapsed);
            }
        })
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The worker's own state between cycles.
struct Cycle {
    batch_size: u64,
    completed: u64,
    registered_workers: usize,
    now: Instant,
    neighbor_cursor: usize,
}

impl Cycle {
    fn new(neighbor_cursor: usize) -> Self {
        Self {
            batch_size: 2,
            completed: 0,
            registered_workers: 0,
            now: Instant::now(),
            neighbor_cursor,
        }
    }

    /// Counts `run` toward the batch and its time toward `policy`. At the end of a batch, starts
    /// the next, sized by `policy` within `cap`, and takes the current count of `workers`.
    fn complete(
        &mut self,
        run: &Run,
        policy: &mut DecisionTree,
        cap: u64,
        workers: impl FnOnce() -> usize,
    ) {
        self.completed += run.processed;
        policy.record_execution(run.elapsed, run.processed);
        if self.completed >= self.batch_size {
            self.completed = 0;
            self.batch_size = policy.complete_batch(cap);
            self.registered_workers = workers();
        }
    }
}

/// What a cycle has run so far.
#[derive(Default)]
struct Run {
    processed: u64,
    elapsed: Duration,
}

/// Counts an execution in progress until dropped.
struct InProgress<'a>(&'a AtomicU64);

impl<'a> InProgress<'a> {
    fn enter(count: &'a AtomicU64) -> Self {
        count.fetch_add(1, Ordering::AcqRel);
        Self(count)
    }
}

impl Drop for InProgress<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

/// Undoes the worker's registration and publishes that it stopped, however the worker ends.
struct WorkerExit<'a> {
    fragment: &'a ControlPlaneFragment,
    registered: bool,
}

impl Drop for WorkerExit<'_> {
    fn drop(&mut self) {
        let fragment = self.fragment;
        fragment.initialized.store(false, Ordering::Release);
        if self.registered {
            fragment.registry.remove_thread();
        }
        FlowContext::clear();
        *lock(&fragment.worker) = None;
        fragment.running.store(false, Ordering::Release);
        if let Some(slot) = fragment.liveness.slot(fragment.cpu) {
            slot.exit();
        }
    }
}

#[cfg(test)]
mod tests;
