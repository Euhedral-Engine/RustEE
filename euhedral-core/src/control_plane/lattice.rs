use super::cache::CacheDirectory;
use super::countdown::Countdown;
use super::shard::{ControlPlaneShard, RebalanceTimedOut, nodes};
use crate::config::LatticeConfig;
use crate::flow_control::{
    LatticeEdge, LatticeNode, LatticeVertex, Placement, RoutingPolicy, RoutingState,
    UpstreamInterceptor, UpstreamRegistry, route_by_hash,
};
use crate::frames::Frame;
use crate::generics::LatticeSource;
use crate::internal::task_pool::TaskPool;
use crate::metrics::{MetricsRegistry, MetricsSnapshot};
use crate::utils::spin_wait;
use euhedral_data_structures::sync::{HotScope, LifecycleMutex};
use euhedral_hardware_utils::monitor::{
    HardwareUtilization, MonitorError, ResourceMonitor, SocketSnapshot,
};
use euhedral_hardware_utils::topology::{
    EffectiveSocketTopology, EffectiveSystemTopology, TopologyMapper,
};
use euhedral_hardware_utils::util::CpuSet;
use std::error::Error;
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc;
#[allow(
    clippy::disallowed_types,
    reason = "HOT-FIX(C18), HOT-FIX(C17), HOT-FIX(C16)"
)]
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, RwLock, Weak};
use std::thread;
use std::time::{Duration, Instant};

/// A listener for each new [`HardwareUtilization`].
pub type UtilizationListener = Box<dyn Fn(&Arc<HardwareUtilization>) + Send + Sync>;

/// Where a lattice gets the host's utilization: a [`ResourceMonitor`] unless one is supplied.
pub trait UtilizationSource: Send + Sync {
    /// The latest utilization, if there is one.
    fn utilization(&self) -> Option<Arc<HardwareUtilization>>;

    /// Calls `listener` with each new utilization once started.
    fn subscribe(&self, listener: UtilizationListener);

    /// Starts publishing to listeners.
    fn start(&self) -> Result<(), MonitorError>;

    /// Stops publishing for good.
    fn close(&self);
}

impl UtilizationSource for ResourceMonitor {
    fn utilization(&self) -> Option<Arc<HardwareUtilization>> {
        ResourceMonitor::utilization(self)
    }

    fn subscribe(&self, listener: UtilizationListener) {
        self.add_listener(listener);
    }

    fn start(&self) -> Result<(), MonitorError> {
        ResourceMonitor::start(self)
    }

    fn close(&self) {
        ResourceMonitor::close(self);
    }
}

/// Why a lattice could not do what it was asked.
#[derive(Debug)]
pub enum LatticeError {
    /// The lattice is closed.
    Closed,
    /// The utilization source had no utilization to start from.
    NoUtilization,
    /// The shards were not ready within the startup timeout.
    StartTimedOut,
    /// The utilization source failed.
    Monitor(MonitorError),
    /// Another reset is under way.
    ResetInProgress,
    /// A reset was given no time.
    ZeroTimeout,
    /// The lattice or one of its shards was still rebalancing at a reset's deadline.
    RebalanceTimedOut(RebalanceTimedOut),
    /// The lattice is running, and the operation needs it shut down.
    Running,
}

impl fmt::Display for LatticeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Closed => f.write_str("the lattice is closed"),
            Self::NoUtilization => f.write_str("no utilization to start the lattice from"),
            Self::StartTimedOut => f.write_str("timed out waiting for the shards to start"),
            Self::Monitor(error) => write!(f, "the utilization source failed: {error}"),
            Self::ResetInProgress => f.write_str("a lattice reset is already in progress"),
            Self::ZeroTimeout => f.write_str("a reset needs a positive timeout"),
            Self::RebalanceTimedOut(error) => error.fmt(f),
            Self::Running => f.write_str("the lattice is running; it must be shut down first"),
        }
    }
}

impl Error for LatticeError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Monitor(error) => Some(error),
            Self::RebalanceTimedOut(error) => Some(error),
            _ => None,
        }
    }
}

/// What a [`clear`](ControlPlaneLattice::clear) dropped, and how many workers remain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CacheReset {
    pub cleared_frames: u64,
    pub active_workers: usize,
}

struct State {
    primed: bool,
    rebalancing: bool,
    version: Option<u64>,
    handles: Vec<Option<Arc<LatticeEdge>>>,
    active: Vec<usize>,
}

/// Where frames enter the control plane: routes them to a [`ControlPlaneShard`] per socket and
/// keeps the shards in line with the host's effective topology.
///
/// A global distributor (a [`LatticeVertex`]) routes each frame to the shard of one active socket:
/// a frame with a [`SocketLocal`](RoutingPolicy::SocketLocal) or tighter policy goes to its
/// origin's socket while that socket is active, the rest by routing hash. A socket is active while
/// it has an effective CPU; frames never route to one that has none.
///
/// The lattice starts on its first [`add_upstream`](Self::add_upstream) (or
/// [`start`](Self::start)), then follows each utilization its source publishes: when the effective
/// topology changes it stops ingest, starts or rebalances the shards of active sockets, shuts down
/// those of sockets that went inactive, and reopens ingest once they are gone.
pub struct ControlPlaneLattice {
    name: String,
    config: LatticeConfig,
    mapper: Arc<TopologyMapper>,
    source: Arc<dyn UtilizationSource>,
    controller: Arc<LatticeVertex>,
    /// The placement policy that runs first on every frame the global distributor routes, and the CPUs it is given.
    placement: Arc<Placement>,
    /// The topology version whose effective CPUs the placement was last given. Written under `updating`.
    placed_version: AtomicU64,
    shards: Vec<Arc<ControlPlaneShard>>,
    pool: TaskPool,
    #[allow(clippy::disallowed_types, reason = "HOT-FIX(C18)")]
    state: Mutex<State>,
    /// Serializes updates.
    #[allow(clippy::disallowed_types, reason = "HOT-FIX(C17)")]
    updating: Mutex<()>,
    /// Held to start: whether the lattice started.
    started: LifecycleMutex<bool>,
    resetting: AtomicBool,
    closed: AtomicBool,
    /// Read-held by every update; closing write-locks it once, so no update outlives the close.
    #[allow(clippy::disallowed_types, reason = "HOT-FIX(C16)")]
    gate: RwLock<()>,
    this: Weak<ControlPlaneLattice>,
}

impl ControlPlaneLattice {
    /// A lattice over the host, following a [`ResourceMonitor`] of it.
    pub fn new(config: LatticeConfig) -> Result<Arc<Self>, LatticeError> {
        let mapper = Arc::new(TopologyMapper::with_allowed(&config.allowed_cpus));
        let monitor = ResourceMonitor::new(mapper.clone()).map_err(LatticeError::Monitor)?;
        Ok(Self::with_source(config, mapper, Arc::new(monitor)))
    }

    /// A lattice over `mapper`'s topology, following `source`. The config's allowed CPUs are
    /// ignored: `mapper` decides which CPUs are allowed.
    #[allow(
        clippy::disallowed_types,
        reason = "HOT-FIX(C18), HOT-FIX(C17), HOT-FIX(C16)"
    )]
    pub fn with_source(
        config: LatticeConfig,
        mapper: Arc<TopologyMapper>,
        source: Arc<dyn UtilizationSource>,
    ) -> Arc<Self> {
        let name = if config.name.trim().is_empty() {
            "ControlPlaneLattice".to_string()
        } else {
            config.name.clone()
        };
        let registry = UpstreamRegistry::new();
        let topology = mapper.topology();
        let directory = CacheDirectory::new(registry.clone(), topology.clone());
        let sockets = topology.socket_count();
        let shards = (0..sockets)
            .map(|socket| {
                log::info!(target: &name, "creating the shard of socket {socket}");
                ControlPlaneShard::new(
                    &directory,
                    socket,
                    format!("{name}-{}-{socket}", config.shard_name),
                    config.cloneable.clone(),
                    config.shutdown_timeout,
                    topology.core_count(),
                )
            })
            .collect();
        let placement = Placement::new(&registry);
        let controller = LatticeVertex::with_placement(
            &registry,
            format!("{name}-GlobalDistributor"),
            sockets,
            placement.clone(),
            route_to_socket,
        );
        let lattice = Arc::new_cyclic(|this| Self {
            pool: TaskPool::new(&name, sockets),
            name,
            config,
            source: source.clone(),
            controller,
            placement,
            placed_version: AtomicU64::new(u64::MAX),
            shards,
            state: Mutex::new(State {
                primed: false,
                rebalancing: false,
                version: None,
                handles: vec![None; sockets],
                active: Vec::new(),
            }),
            mapper,
            updating: Mutex::new(()),
            started: LifecycleMutex::new(false),
            resetting: AtomicBool::new(false),
            closed: AtomicBool::new(false),
            gate: RwLock::new(()),
            this: this.clone(),
        });
        let listener = Arc::downgrade(&lattice);
        source.subscribe(Box::new(move |utilization| {
            if let Some(lattice) = listener.upgrade() {
                lattice.update(utilization);
            }
        }));
        lattice
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn config(&self) -> &LatticeConfig {
        &self.config
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn arc(&self) -> Arc<Self> {
        self.this.upgrade().expect("a live lattice is shared")
    }

    /// Starts the lattice from its source's latest utilization, waits until the shard of every
    /// active socket is ready, then starts following the source. Does nothing once started; after a
    /// failure it may be called again.
    pub fn start(&self) -> Result<(), LatticeError> {
        let mut started = self.started.lock();
        if self.closed.load(Ordering::Acquire) {
            return Err(LatticeError::Closed);
        }
        if *started {
            return Ok(());
        }
        log::info!(target: &self.name, "starting");
        let utilization = self
            .source
            .utilization()
            .ok_or(LatticeError::NoUtilization)?;
        self.update(&utilization);

        log::trace!(target: &self.name, "awaiting readiness");
        let deadline = Instant::now() + self.config.startup_timeout;
        spin_wait::await_while(|| !self.ready() && Instant::now() < deadline);
        if !self.ready() {
            return Err(LatticeError::StartTimedOut);
        }
        self.source.start().map_err(LatticeError::Monitor)?;
        *started = true;
        log::info!(target: &self.name, "startup complete");
        Ok(())
    }

    /// Whether the lattice is primed, not rebalancing, and the shard of every active socket is
    /// ready. Sockets without an effective CPU have no shard running and are not waited on.
    fn ready(&self) -> bool {
        let active = {
            let state = self.lock();
            if !state.primed || state.rebalancing {
                return false;
            }
            state.active.clone()
        };
        active
            .iter()
            .all(|&socket| self.shards[socket].is_started())
    }

    /// The lattice's upstream registry: its workers' queues and the cells its workers read from their views (see
    /// [`UpstreamRegistry::with_view`]).
    pub fn registry(&self) -> &Arc<UpstreamRegistry> {
        self.controller.edge().registry()
    }

    /// Sets the placement policy: one callback that runs first on every frame the lattice routes, before the shard and
    /// the core are chosen. It sees the frame and the CPUs the lattice routes with now (a snapshot that is not cloned
    /// per frame), and says where the frame goes by setting its origin and a cache-local routing policy, which the
    /// routing that follows honours. [`PlacementPicker`](super::PlacementPicker) builds the common policy.
    ///
    /// The policy runs on the frame path, from every thread that routes frames, at the same time: it is `Fn`, not
    /// `FnMut`, and must be re-entrant, take no lock, not allocate and not block. Anything it reads is the caller's to
    /// make safe to read from many threads.
    ///
    /// The policy returns `true` when it placed the frame (set its origin and routing policy). Until one is set, the
    /// lattice has a policy that places nothing and returns `false`.
    ///
    /// It can be set only before the lattice starts, once or more (the last one stands): it is a plain field the frame
    /// path reads with no atomic. A lattice that is running, or has been closed (close returns while frames it admitted
    /// may still be winding down), refuses it.
    ///
    /// The policy covers pushed and pulled frames alike: a frame a source pushes is placed when the lattice routes it,
    /// and a frame a worker pulls from a source is placed as it comes out of the source's pull, before the worker's
    /// own consumer sees it. A frame the policy gives an origin goes through the routing, which honours the origin;
    /// the others reach the worker's consumer as they would without a policy. Producers do not have to place their
    /// frames.
    ///
    /// The routing honours the origin only while the origin's core is active. During a rebalance, or after a clone failed
    /// to start, a frame placed on an inactive core falls back to a hash over the active cores (and sockets), which can
    /// move it out of the set the policy chose; the policy chooses among the CPUs the lattice reports as effective, not
    /// among those of cores that are running a clone.
    ///
    /// # Errors
    ///
    /// [`LatticeError::Running`] once the lattice has started, [`LatticeError::Closed`] once it is closed.
    pub fn set_placement_policy(
        &self,
        policy: impl Fn(&mut dyn Frame, &CpuSet) -> bool + Send + Sync + 'static,
    ) -> Result<(), LatticeError> {
        // `start` holds this lock from before it ever starts the source until it has set the flag, and nothing sets the
        // flag back.
        let started = self.started.lock();
        if self.closed.load(Ordering::Acquire) {
            return Err(LatticeError::Closed);
        }
        if *started {
            return Err(LatticeError::Running);
        }
        // SAFETY: the lattice has not started, so no source has been ingested (`add_upstream` starts first and ingests
        // after, and the controller is reachable by no other path): nothing routes through or pulls from the controller.
        // The unlock below releases this write, and the lock that `start` takes in any `add_upstream` that follows
        // acquires it before the ingest, from which every thread that learns of the source follows.
        unsafe { self.controller.set_placement_policy(policy) };
        Ok(())
    }

    /// Starts the lattice if needed, then feeds it from `source`.
    pub fn add_upstream(
        &self,
        source: Arc<dyn LatticeSource>,
    ) -> Result<Arc<UpstreamInterceptor>, LatticeError> {
        // Attaching a source is rebalance work, also when a frame does it.
        let _cold = HotScope::leave();
        log::trace!(target: &self.name, "adding an upstream");
        self.start()?;
        self.controller
            .ingest(source)
            .map_err(|_| LatticeError::Closed)
    }

    /// Follows one utilization: rebalances if the effective topology changed, otherwise passes each
    /// active shard its share.
    fn update(&self, utilization: &HardwareUtilization) {
        let _gate = self.gate.read().unwrap_or_else(PoisonError::into_inner);
        if self.closed.load(Ordering::Acquire) || self.resetting.load(Ordering::Acquire) {
            return;
        }
        let _updating = self.updating.lock().unwrap_or_else(PoisonError::into_inner);
        self.mapper.update_from(utilization);
        let topology = self.mapper.effective_topology();
        // The placement policy is given the CPUs the lattice routes with now. `updating` makes this the only writer.
        if self.placed_version.load(Ordering::Relaxed) != topology.global_version {
            self.placement
                .publish_cpus(self.registry(), topology.effective_cpus.clone());
            self.placed_version
                .store(topology.global_version, Ordering::Relaxed);
        }
        let (version, active) = {
            let state = self.lock();
            (state.version, state.active.clone())
        };

        if version != Some(topology.global_version) {
            if version.is_none() {
                log::info!(
                    target: &self.name,
                    "initializing for global topology v{}",
                    topology.global_version
                );
            } else {
                log::warn!(
                    target: &self.name,
                    "the global topology changed; rebalancing for v{}",
                    topology.global_version
                );
            }
            self.handle_system_topology_change(utilization, topology.clone());
            if topology.effective_cpus.is_empty() {
                log::error!(target: &self.name, "there are no usable CPUs for this lattice");
            }
            return;
        }

        for socket in active {
            let Some(socket_topology) = socket_topology(&topology, socket) else {
                continue;
            };
            let snapshot = self.snapshot(utilization, &topology, &socket_topology);
            let this = self.arc();
            self.pool.execute_or_run(move || {
                let _gate = this.gate.read().unwrap_or_else(PoisonError::into_inner);
                if this.closed.load(Ordering::Acquire) {
                    return;
                }
                this.shards[socket].update(snapshot.as_ref(), &socket_topology);
            });
        }
    }

    /// Starts or rebalances the shards of active sockets, remaps ingest to them, and shuts down the
    /// shards of sockets that went inactive before reopening ingest.
    fn handle_system_topology_change(
        &self,
        utilization: &HardwareUtilization,
        topology: Arc<EffectiveSystemTopology>,
    ) {
        let (handles, retired, first) = {
            let mut state = self.lock();
            if state.rebalancing {
                return;
            }
            state.rebalancing = true;
            state.version = Some(topology.global_version);
            self.controller.set_drain(true);
            for socket in self.active_sockets(&topology) {
                if state.handles[socket].is_none() {
                    let drain = self.controller.edge().drain_flag().clone();
                    state.handles[socket] =
                        Some(LatticeEdge::new(self.controller.edge().registry(), drain));
                }
            }
            let mut retired = Vec::new();
            for socket in 0..self.shards.len() {
                if !topology.effective_sockets.contains(socket)
                    && state.handles[socket].take().is_some()
                {
                    retired.push(socket);
                }
            }
            (state.handles.clone(), retired, !state.primed)
        };

        let active: Vec<usize> = self.active_sockets(&topology).collect();
        for &socket in &active {
            let Some(socket_topology) = socket_topology(&topology, socket) else {
                continue;
            };
            let snapshot = self.snapshot(utilization, &topology, &socket_topology);
            let shard = &self.shards[socket];
            if shard.is_running() {
                shard.update(snapshot.as_ref(), &socket_topology);
            } else {
                let handle = handles[socket]
                    .clone()
                    .expect("an active socket has a handle");
                shard.start(snapshot.as_ref(), &socket_topology, handle);
            }
        }

        let mut state = self.lock();
        log::trace!(target: &self.name, "remapping ingest");
        if !self
            .controller
            .set_downstream_mapping(&active, &nodes(&handles))
        {
            log::error!(target: &self.name, "the global distributor refused the remap");
        }
        state.active = active;
        if first || retired.is_empty() {
            self.controller.set_drain(false);
            state.primed = true;
            state.rebalancing = false;
            return;
        }
        drop(state);

        log::info!(target: &self.name, "shutting down {} retired shards", retired.len());
        let this = self.this.clone();
        let countdown = Countdown::new(
            retired.len(),
            Box::new(move || {
                if let Some(this) = this.upgrade() {
                    this.finish_rebalance();
                }
            }),
        );
        for socket in retired {
            let countdown = countdown.clone();
            self.shards[socket].shut_down(move || countdown.tick());
        }
    }

    /// Reopens ingest once the retired shards are shut down.
    fn finish_rebalance(&self) {
        let mut state = self.lock();
        if !self.closed.load(Ordering::Acquire) {
            self.controller.set_drain(false);
        }
        state.rebalancing = false;
        log::info!(target: &self.name, "rebalance complete");
    }

    /// The sockets of `topology` this lattice has a shard for.
    fn active_sockets(&self, topology: &EffectiveSystemTopology) -> impl Iterator<Item = usize> {
        let sockets = self.shards.len();
        topology
            .effective_sockets
            .iter()
            .filter(move |&socket| socket < sockets)
    }

    /// The socket's share of `utilization`, its quota in proportion to its effective CPUs.
    fn snapshot(
        &self,
        utilization: &HardwareUtilization,
        topology: &EffectiveSystemTopology,
        socket: &EffectiveSocketTopology,
    ) -> Option<SocketSnapshot> {
        let total = topology.effective_cpus.len().max(1) as f64;
        let quota = socket.effective_cpus.len() as f64 / total * utilization.quota_cpus;
        log::trace!(
            target: &self.name,
            "socket {} gets {quota} of {} quota CPUs",
            socket.socket_id,
            utilization.quota_cpus
        );
        match utilization.socket_snapshot(socket.socket_id, &socket.effective_core_to_cpu, quota) {
            Ok(snapshot) => snapshot,
            Err(error) => {
                log::warn!(
                    target: &self.name,
                    "no snapshot for socket {}: {error}",
                    socket.socket_id
                );
                None
            }
        }
    }

    /// Stops ingest and clears what every shard holds for the current trial, runs
    /// `reset_while_drained`, then reopens ingest. Pause ingest sources first.
    pub fn clear(
        &self,
        timeout: Duration,
        reset_while_drained: impl FnOnce(),
    ) -> Result<CacheReset, LatticeError> {
        if timeout.is_zero() {
            return Err(LatticeError::ZeroTimeout);
        }
        if self.closed.load(Ordering::Acquire) {
            return Err(LatticeError::Closed);
        }
        if !*self.started.lock() {
            return Ok(CacheReset {
                cleared_frames: 0,
                active_workers: 0,
            });
        }
        if self.resetting.swap(true, Ordering::AcqRel) {
            return Err(LatticeError::ResetInProgress);
        }
        let _resetting = Resetting(self);

        let deadline = Instant::now() + timeout;
        spin_wait::await_while(|| self.is_rebalancing() && Instant::now() < deadline);
        if self.is_rebalancing() {
            return Err(LatticeError::RebalanceTimedOut(RebalanceTimedOut {
                name: self.name.clone(),
            }));
        }

        self.controller.set_drain(true);
        let _reopen = Reopen(self);
        let mut cleared_frames = 0;
        for shard in &self.shards {
            cleared_frames += shard
                .reset_for_next_trial(deadline)
                .map_err(LatticeError::RebalanceTimedOut)?;
        }
        reset_while_drained();
        Ok(CacheReset {
            cleared_frames,
            active_workers: self.active_workers(),
        })
    }

    fn is_rebalancing(&self) -> bool {
        self.lock().rebalancing
    }

    /// The number of cores with a clone, over every shard.
    pub fn active_workers(&self) -> usize {
        self.shards.iter().map(|shard| shard.active_cores()).sum()
    }

    /// Every worker's and cache's meters now, read without stopping the workers; `None` unless
    /// the cloned object has a [`MetricsRegistry`]
    /// ([`CloneableObject::metrics_registry`](crate::generics::CloneableObject::metrics_registry)).
    pub fn metrics(&self) -> Option<MetricsSnapshot> {
        self.metrics_registry().map(|registry| registry.snapshot())
    }

    /// Where the cloned object's workers and caches register their meters, if anywhere.
    pub fn metrics_registry(&self) -> Option<Arc<MetricsRegistry>> {
        self.config.cloneable.metrics_registry()
    }

    /// Whether every shard has finished its work.
    pub fn is_drained(&self) -> bool {
        self.shards.iter().all(|shard| shard.is_drained())
    }

    /// Closes the lattice for good: stops following the source, and closes the shards once no
    /// frame can still route to them, waiting up to the shutdown timeout for that.
    pub fn close(&self) {
        if self.closed.swap(true, Ordering::AcqRel) {
            return;
        }
        drop(self.gate.write().unwrap_or_else(PoisonError::into_inner));
        log::info!(target: &self.name, "closing");
        self.source.close();

        let (closed, done) = mpsc::channel();
        let shards = self.shards.clone();
        let close_shards = move || {
            for shard in &shards {
                shard.close();
            }
            let _ = closed.send(());
        };
        let name = self.name.clone();
        self.controller.close_after_routes_retire(move || {
            // The last route can retire on a worker mid-push, and closing a shard waits for its
            // workers, so the shards close on a thread of their own.
            let task = Arc::new(LifecycleMutex::new(Some(close_shards)));
            let spawned = {
                let task = task.clone();
                thread::Builder::new()
                    .name(format!("{name}-close"))
                    .spawn(move || run_once(&task))
            };
            if spawned.is_err() {
                run_once(&task);
            }
        });

        let timeout = self.config.shutdown_timeout;
        if !timeout.is_zero() && done.recv_timeout(timeout).is_err() {
            log::warn!(target: &self.name, "close is waiting for admitted routes to retire");
        }
        self.pool.shutdown();
        log::info!(target: &self.name, "closed");
    }
}

impl Drop for ControlPlaneLattice {
    fn drop(&mut self) {
        self.close();
    }
}

/// Ends a reset when dropped.
struct Resetting<'a>(&'a ControlPlaneLattice);

impl Drop for Resetting<'_> {
    fn drop(&mut self) {
        self.0.resetting.store(false, Ordering::Release);
    }
}

/// Reopens ingest when dropped, so a failed reset still reopens it.
struct Reopen<'a>(&'a ControlPlaneLattice);

impl Drop for Reopen<'_> {
    fn drop(&mut self) {
        self.0.controller.set_drain(false);
    }
}

fn run_once(task: &LifecycleMutex<Option<impl FnOnce()>>) {
    if let Some(task) = task.lock().take() {
        task();
    }
}

/// Routes to the origin's socket for a socket-local or tighter frame whose socket is active,
/// otherwise by hash.
fn route_to_socket(frame: &dyn Frame, state: &RoutingState) -> usize {
    let header = frame.header();
    if header.routing_policy() > RoutingPolicy::Anywhere
        && let Some(origin) = header.origin()
        && let Some(index) = state.active_index(origin.socket)
    {
        return index;
    }
    route_by_hash(frame, state.len())
}

fn socket_topology(
    topology: &EffectiveSystemTopology,
    socket: usize,
) -> Option<EffectiveSocketTopology> {
    topology.socket_topologies.get(socket)?.clone()
}

#[cfg(test)]
mod tests;
