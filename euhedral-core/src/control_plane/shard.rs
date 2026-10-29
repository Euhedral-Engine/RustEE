use super::cache::CacheDirectory;
use super::countdown::{Countdown, Done};
use crate::config::CloneConfig;
use crate::flow_control::{LatticeEdge, LatticeNode, LatticeVertex, RoutingPolicy, RoutingState};
use crate::frames::Frame;
use crate::generics::CloneableObject;
use crate::internal::task_pool::TaskPool;
use crate::utils::math::unsigned_multiply_high;
use crate::utils::spin_wait;
use euhedral_hardware_utils::monitor::{CoreSnapshot, SocketSnapshot};
use euhedral_hardware_utils::topology::EffectiveSocketTopology;
use euhedral_hardware_utils::util::CpuSet;
use std::error::Error;
use std::fmt;
use std::panic::{self, AssertUnwindSafe};
#[allow(clippy::disallowed_types, reason = "HOT-FIX(C19)")]
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, Weak};
use std::time::{Duration, Instant};

type ObjectClone = Arc<dyn CloneableObject>;

/// Returned when a shard is still rebalancing at a reset's deadline.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RebalanceTimedOut {
    pub name: String,
}

impl fmt::Display for RebalanceTimedOut {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "timed out waiting for {} to finish rebalancing",
            self.name
        )
    }
}

impl Error for RebalanceTimedOut {}

#[derive(Default)]
struct State {
    started: bool,
    primed: bool,
    rebalancing: bool,
    shutdown_scheduled: bool,
    /// Bumped on every start, shutdown and close, so work begun for one run of the shard can tell
    /// that the shard has moved on.
    generation: u64,
    version: Option<u64>,
    cores_to_drain: usize,
    distributor: Option<Arc<LatticeVertex>>,
    upstream: Option<Arc<LatticeEdge>>,
    pool: Option<Arc<TaskPool>>,
    core_handles: Vec<Option<Arc<LatticeEdge>>>,
    clones: Vec<Option<ObjectClone>>,
    active_cores: Vec<usize>,
}

/// The control plane of one socket: keeps a clone of a [`CloneableObject`] on each of the socket's
/// effective cores and routes the socket's frames among them.
///
/// Frames arrive through the edge the shard is [started](Self::start) with, and a core distributor
/// (a [`LatticeVertex`]) routes them to the edge of one core, below which that core's clone sits.
/// Frames route by hash, except that a frame with a [`CacheLocal`](RoutingPolicy::CacheLocal)
/// policy goes to its origin's core while that core is active.
///
/// When the socket's topology changes, the shard stops ingest, spawns clones for new cores, drains
/// every existing clone, closes those whose cores are gone or that fail to drain in time (replacing
/// the latter), and reopens ingest. A clone is closed only once no routing generation can still
/// send it frames.
pub struct ControlPlaneShard {
    socket: usize,
    name: String,
    cloneable: ObjectClone,
    shutdown_timeout: Duration,
    directory: Arc<CacheDirectory>,
    core_span: usize,
    #[allow(clippy::disallowed_types, reason = "HOT-FIX(C19)")]
    state: Mutex<State>,
    this: Weak<ControlPlaneShard>,
}

impl ControlPlaneShard {
    /// A shard for `socket` of `directory`'s lattice that clones `cloneable` onto cores with ids
    /// below `core_span`. Clones get `shutdown_timeout` to drain before they are closed.
    #[allow(clippy::disallowed_types, reason = "HOT-FIX(C19)")]
    pub fn new(
        directory: &Arc<CacheDirectory>,
        socket: usize,
        name: impl Into<String>,
        cloneable: ObjectClone,
        shutdown_timeout: Duration,
        core_span: usize,
    ) -> Arc<Self> {
        Arc::new_cyclic(|this| Self {
            socket,
            name: name.into(),
            cloneable,
            shutdown_timeout,
            directory: directory.clone(),
            core_span,
            state: Mutex::new(State::default()),
            this: this.clone(),
        })
    }

    pub fn socket(&self) -> usize {
        self.socket
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn arc(&self) -> Arc<Self> {
        self.this.upgrade().expect("a live shard is shared")
    }

    /// The number of cores with a clone.
    pub fn active_cores(&self) -> usize {
        self.lock().active_cores.len()
    }

    /// Whether the shard is running and every clone is ready.
    pub fn is_started(&self) -> bool {
        let clones = {
            let state = self.lock();
            if !state.started {
                return false;
            }
            state.clones.clone()
        };
        clones.iter().flatten().all(|clone| clone.ready())
    }

    /// Whether the shard is running, ready or not.
    pub(crate) fn is_running(&self) -> bool {
        self.lock().started
    }

    /// Whether the shard is rebalancing; it takes no frames meanwhile.
    pub fn is_rebalancing(&self) -> bool {
        self.lock().rebalancing
    }

    /// Whether every clone has finished its work. A stopped shard is drained; a rebalancing one is
    /// not.
    pub fn is_drained(&self) -> bool {
        let clones = {
            let state = self.lock();
            if !state.started {
                return true;
            }
            if state.rebalancing {
                return false;
            }
            state.clones.clone()
        };
        clones.iter().flatten().all(|clone| clone.is_drained())
    }

    /// Starts the shard below `upstream` and places clones by `topology`, passing each its core's
    /// share of `snapshot` if there is one. Does nothing while running.
    pub fn start(
        &self,
        snapshot: Option<&SocketSnapshot>,
        topology: &EffectiveSocketTopology,
        upstream: Arc<LatticeEdge>,
    ) {
        let distributor = {
            let mut state = self.lock();
            if state.started {
                return;
            }
            log::info!(target: &self.name, "starting");
            let distributor = LatticeVertex::with_routing(
                self.directory.registry(),
                format!("{}-CoreDistributor", self.name),
                self.core_span,
                route_to_core,
            );
            *state = State {
                started: true,
                generation: state.generation + 1,
                distributor: Some(distributor.clone()),
                upstream: Some(upstream.clone()),
                pool: Some(Arc::new(TaskPool::new(
                    &format!("{}-ExecutorService", self.name),
                    self.core_span,
                ))),
                core_handles: vec![None; self.core_span],
                clones: vec![None; self.core_span],
                ..State::default()
            };
            distributor
        };
        if distributor.add_upstream_node(&*upstream).is_err() {
            unreachable!("a new distributor is open");
        }
        self.update(snapshot, topology);
    }

    /// Passes each clone its core's share of `snapshot`, or rebalances if the topology changed.
    pub fn update(&self, snapshot: Option<&SocketSnapshot>, topology: &EffectiveSocketTopology) {
        let (primed, version, clones) = {
            let state = self.lock();
            if !state.started {
                log::error!(target: &self.name, "cannot update a shard that is not started");
                return;
            }
            if state.rebalancing {
                log::warn!(
                    target: &self.name,
                    "cannot update while rebalancing; {} cores left to drain",
                    state.cores_to_drain
                );
                return;
            }
            let clones: Vec<(usize, ObjectClone)> = state
                .active_cores
                .iter()
                .filter_map(|&core| Some((core, state.clones[core].clone()?)))
                .collect();
            (state.primed, state.version, clones)
        };
        if !primed {
            log::info!(target: &self.name, "priming clones for socket topology v{}", topology.version);
            self.handle_topology_change(snapshot, topology);
        } else if version != Some(topology.version) {
            log::warn!(
                target: &self.name,
                "the topology changed; rebalancing for socket topology v{}",
                topology.version
            );
            self.handle_topology_change(snapshot, topology);
        } else {
            for (core, clone) in clones {
                update_clone(&self.name, &clone, core_snapshot(snapshot, core));
            }
        }
    }

    fn handle_topology_change(
        &self,
        snapshot: Option<&SocketSnapshot>,
        topology: &EffectiveSocketTopology,
    ) {
        let (distributor, old_clones, handles, first, generation, pool) = {
            let mut state = self.lock();
            if state.rebalancing || !state.started {
                return;
            }
            let Some(distributor) = state.distributor.clone() else {
                return;
            };
            state.rebalancing = true;
            state.version = Some(topology.version);
            distributor.set_drain(true);
            for core in topology
                .effective_cores
                .iter()
                .filter(|&core| core < self.core_span)
            {
                if state.core_handles[core].is_none() {
                    state.core_handles[core] = Some(self.core_edge(&distributor));
                }
            }
            (
                distributor,
                state.clones.clone(),
                state.core_handles.clone(),
                !state.primed,
                state.generation,
                state.pool.clone(),
            )
        };

        log::info!(target: &self.name, "creating new clones");
        let cores: Vec<usize> = topology
            .effective_cores
            .iter()
            .filter(|&core| core < self.core_span)
            .collect();
        let mut next = vec![None; self.core_span];
        let mut spawned = Vec::new();
        for &core in &cores {
            match &old_clones[core] {
                Some(clone) => {
                    next[core] = Some(clone.clone());
                    let (this, clone) = (self.arc(), clone.clone());
                    let core_snapshot = core_snapshot(snapshot, core).cloned();
                    run_on(pool.as_deref(), move || {
                        let state = this.lock();
                        if state.started && state.generation == generation {
                            update_clone(&this.name, &clone, core_snapshot.as_ref());
                        }
                    });
                }
                None => {
                    let clone = self.spawn_clone(
                        core,
                        handles[core].clone(),
                        core_cpus(topology, core),
                        core_snapshot(snapshot, core),
                        true,
                    );
                    spawned.push(clone.clone());
                    next[core] = Some(clone);
                }
            }
        }

        {
            let mut state = self.lock();
            if !state.started || state.generation != generation {
                drop(state);
                let deadline = self.deadline();
                for clone in spawned {
                    close_clone(&self.name, &clone, deadline);
                }
                return;
            }
            state.clones = next;
            state.active_cores = cores;
            if !distributor.set_downstream_mapping(&state.active_cores, &nodes(&state.core_handles))
            {
                log::error!(target: &self.name, "the core distributor refused a prepared mapping");
            }
        }

        if first {
            let (clones, idle) = {
                let state = self.lock();
                (state.clones.clone(), state.active_cores.is_empty())
            };
            for clone in clones.iter().flatten() {
                clone.set_drain_mode(false);
            }
            distributor.set_drain(idle);
            let mut state = self.lock();
            state.primed = true;
            state.rebalancing = false;
            log::info!(target: &self.name, "priming complete");
        } else {
            self.drain_and_prune(old_clones, snapshot, topology, generation);
        }
    }

    fn core_edge(&self, distributor: &LatticeVertex) -> Arc<LatticeEdge> {
        LatticeEdge::new(
            self.directory.registry(),
            distributor.edge().drain_flag().clone(),
        )
    }

    /// Clones the object onto `core`, attaches it below `handle`, starts it and passes it the
    /// core's utilization.
    fn spawn_clone(
        &self,
        core: usize,
        handle: Option<Arc<LatticeEdge>>,
        cpus: CpuSet,
        snapshot: Option<&CoreSnapshot>,
        drain: bool,
    ) -> ObjectClone {
        log::trace!(target: &self.name, "spawning a clone on core {core}");
        let clone = self.cloneable.clone_object(
            &CloneConfig::new(self.name.clone(), core, cpus)
                .with_cache_directory(self.directory.clone()),
        );
        if let Some(handle) = handle {
            clone.clone().input_edge(handle);
        }
        clone.set_drain_mode(drain);
        clone.start();
        if let Some(snapshot) = snapshot {
            clone.update(snapshot);
        }
        clone
    }

    /// Drains every clone of the previous run and closes those not in the new one, then reopens
    /// ingest once all have drained.
    fn drain_and_prune(
        &self,
        old_clones: Vec<Option<ObjectClone>>,
        snapshot: Option<&SocketSnapshot>,
        topology: &EffectiveSocketTopology,
        generation: u64,
    ) {
        log::info!(target: &self.name, "draining and pruning clones");
        let mut tasks = Vec::new();
        let pool = {
            let mut state = self.lock();
            if !state.started || state.generation != generation {
                return;
            }
            let retiring = state.core_handles.clone();
            for (core, clone) in old_clones.into_iter().enumerate() {
                let Some(clone) = clone else {
                    continue;
                };
                state.cores_to_drain += 1;
                let dead = state.clones[core].is_none();
                if dead {
                    log::info!(target: &self.name, "removing the clone on core {core}");
                    state.core_handles[core] = None;
                }
                tasks.push(DrainTask {
                    core,
                    clone,
                    dead,
                    retiring: retiring[core].clone(),
                    cpus: core_cpus(topology, core),
                    snapshot: core_snapshot(snapshot, core).cloned(),
                    generation,
                });
            }
            if tasks.is_empty() {
                self.restart_ingest(&mut state);
            }
            state.pool.clone()
        };
        for task in &tasks {
            task.clone.set_drain_mode(true);
        }
        for task in tasks {
            let this = self.arc();
            run_on(pool.as_deref(), move || this.finish_drain(task));
        }
    }

    fn finish_drain(&self, task: DrainTask) {
        let DrainTask {
            core,
            clone,
            dead,
            retiring,
            cpus,
            snapshot,
            generation,
        } = task;
        let deadline = self.deadline();
        spin_wait::await_while(|| !clone.is_drained() && Instant::now() < deadline);
        let close = dead || !clone.is_drained();

        let mut state = self.lock();
        if !state.started || state.generation != generation {
            drop(state);
            if close {
                close_clone(&self.name, &clone, deadline);
            }
            return;
        }
        if close && !dead {
            log::info!(target: &self.name, "restarting the clone on core {core}");
            let distributor = state
                .distributor
                .clone()
                .expect("a started shard has a distributor");
            let handle = self.core_edge(&distributor);
            state.core_handles[core] = Some(handle.clone());
            let replacement = self.spawn_clone(core, Some(handle), cpus, snapshot.as_ref(), true);
            state.clones[core] = Some(replacement);
        }
        state.cores_to_drain -= 1;
        if state.cores_to_drain == 0 {
            log::info!(target: &self.name, "drain complete");
            self.restart_ingest(&mut state);
        }
        drop(state);
        if close {
            let name = self.name.clone();
            let retire = move || close_clone(&name, &clone, deadline);
            match retiring {
                Some(handle) => handle.defer_retirement(Box::new(retire)),
                None => retire(),
            }
        }
    }

    /// Maps the active cores and reopens ingest, once no clone is left to drain.
    fn restart_ingest(&self, state: &mut State) {
        if !state.started || state.cores_to_drain != 0 {
            return;
        }
        let Some(distributor) = state.distributor.clone() else {
            return;
        };
        log::info!(target: &self.name, "restarting ingest");
        if !distributor.set_downstream_mapping(&state.active_cores, &nodes(&state.core_handles)) {
            log::error!(target: &self.name, "the core distributor refused the restart mapping");
        }
        for clone in state.clones.iter().flatten() {
            clone.set_drain_mode(false);
        }
        distributor.set_drain(state.active_cores.is_empty());
        state.rebalancing = false;
    }

    /// Shuts the shard down once no routing generation can send it frames any more, draining and
    /// closing every clone, then calls `done`. Calls `done` at once if the shard is not running or
    /// its shutdown is already under way.
    pub fn shut_down(&self, done: impl FnOnce() + Send + 'static) {
        let done: Done = Box::new(done);
        let (incoming, generation) = {
            let mut state = self.lock();
            if !state.started || state.shutdown_scheduled {
                drop(state);
                done();
                return;
            }
            state.shutdown_scheduled = true;
            (state.upstream.clone(), state.generation)
        };
        let this = self.arc();
        let shutdown = move || this.shut_down_now(generation, done);
        match incoming {
            Some(edge) => edge.defer_retirement(Box::new(shutdown)),
            None => shutdown(),
        }
    }

    fn shut_down_now(&self, generation: u64, done: Done) {
        let Some(stopped) = self.stop(Some(generation)) else {
            done();
            return;
        };
        log::info!(target: &self.name, "shutting down");
        let countdown = Countdown::new(stopped.clones.len(), done);
        let deadline = self.deadline();
        for (core, clone) in stopped.clones {
            log::trace!(target: &self.name, "shutting down the clone on core {core}");
            clone.set_drain_mode(true);
            let (name, tick) = (self.name.clone(), Tick(countdown.clone()));
            run_on(stopped.pool.as_deref(), move || {
                let _tick = tick;
                spin_wait::await_while(|| !clone.is_drained() && Instant::now() < deadline);
                if !clone.is_drained() {
                    log::error!(target: &name, "the clone on core {core} timed out; forcing shutdown");
                }
                close_clone(&name, &clone, deadline);
            });
        }
        if let Some(distributor) = stopped.distributor {
            distributor.close();
        }
        if let Some(pool) = stopped.pool {
            pool.shutdown();
        }
        log::info!(target: &self.name, "shutdown complete");
    }

    /// Closes the shard and every clone right away.
    pub fn close(&self) {
        let Some(stopped) = self.stop(None) else {
            return;
        };
        log::info!(target: &self.name, "closing");
        let deadline = self.deadline();
        if let Some(distributor) = stopped.distributor {
            distributor.close();
        }
        for (_, clone) in stopped.clones {
            close_clone(&self.name, &clone, deadline);
        }
        // Queued drain tasks still run: each closes its clone once it sees the shard has stopped.
        if let Some(pool) = stopped.pool {
            pool.shutdown();
        }
        log::info!(target: &self.name, "closed");
    }

    /// Stops the run, if it is `generation` (or any run with `None`), and hands back what it held.
    fn stop(&self, generation: Option<u64>) -> Option<Stopped> {
        let mut state = self.lock();
        if !state.started || generation.is_some_and(|generation| generation != state.generation) {
            return None;
        }
        let generation = state.generation + 1;
        let previous = std::mem::replace(
            &mut *state,
            State {
                generation,
                ..State::default()
            },
        );
        let clones = previous
            .clones
            .into_iter()
            .enumerate()
            .filter_map(|(core, clone)| Some((core, clone?)))
            .collect();
        Some(Stopped {
            clones,
            distributor: previous.distributor,
            pool: previous.pool,
        })
    }

    /// Drains every clone, clears what each holds for the current trial, and reopens ingest.
    /// Returns how many buffered frames the clones dropped.
    pub fn reset_for_next_trial(&self, deadline: Instant) -> Result<u64, RebalanceTimedOut> {
        spin_wait::await_while(|| self.is_rebalancing() && Instant::now() < deadline);
        let (distributor, clones) = {
            let state = self.lock();
            if !state.started {
                return Ok(0);
            }
            if state.rebalancing {
                return Err(RebalanceTimedOut {
                    name: self.name.clone(),
                });
            }
            let Some(distributor) = state.distributor.clone() else {
                return Ok(0);
            };
            (distributor, state.clones.clone())
        };
        distributor.set_drain(true);
        let clones: Vec<ObjectClone> = clones.into_iter().flatten().collect();
        for clone in &clones {
            clone.set_drain_mode(true);
        }
        let reopen = Reopen {
            distributor: &distributor,
            clones: &clones,
        };
        Ok(reopen
            .clones
            .iter()
            .map(|clone| clone.reset(deadline))
            .sum())
    }

    fn deadline(&self) -> Instant {
        Instant::now() + self.shutdown_timeout
    }
}

/// What a stopped run held.
struct Stopped {
    clones: Vec<(usize, ObjectClone)>,
    distributor: Option<Arc<LatticeVertex>>,
    pool: Option<Arc<TaskPool>>,
}

/// A clone of the previous run waiting to drain.
struct DrainTask {
    core: usize,
    clone: ObjectClone,
    /// Whether the clone's core is gone.
    dead: bool,
    /// The edge the clone was fed through.
    retiring: Option<Arc<LatticeEdge>>,
    cpus: CpuSet,
    snapshot: Option<CoreSnapshot>,
    generation: u64,
}

/// Reopens ingest when dropped, so a panicking reset still reopens it.
struct Reopen<'a> {
    distributor: &'a LatticeVertex,
    clones: &'a [ObjectClone],
}

impl Drop for Reopen<'_> {
    fn drop(&mut self) {
        for clone in self.clones {
            clone.set_drain_mode(false);
        }
        self.distributor.set_drain(false);
    }
}

/// Ticks its countdown when dropped, so a clone that panics while shutting down still counts.
struct Tick(Arc<Countdown>);

impl Drop for Tick {
    fn drop(&mut self) {
        self.0.tick();
    }
}

/// Routes to the origin's core for a cache-local frame whose core is active, otherwise by hash.
/// The hash is rotated so the shard's choice does not repeat the lattice's.
fn route_to_core(frame: &dyn Frame, state: &RoutingState) -> usize {
    let header = frame.header();
    if header.routing_policy() > RoutingPolicy::SocketLocal
        && let Some(origin) = header.origin()
        && let Some(index) = state.active_index(origin.core)
    {
        return index;
    }
    let rotated = header.routing_hash().rotate_left(31);
    unsigned_multiply_high(rotated, state.len() as u64) as usize
}

/// The handles as lattice nodes, for a vertex mapping.
pub(super) fn nodes(handles: &[Option<Arc<LatticeEdge>>]) -> Vec<Option<Arc<dyn LatticeNode>>> {
    handles
        .iter()
        .map(|handle| handle.clone().map(|handle| handle as Arc<dyn LatticeNode>))
        .collect()
}

fn core_snapshot(snapshot: Option<&SocketSnapshot>, core: usize) -> Option<&CoreSnapshot> {
    snapshot?.core_snapshots.get(core)?.as_ref()
}

fn core_cpus(topology: &EffectiveSocketTopology, core: usize) -> CpuSet {
    topology
        .effective_core_to_cpu
        .get(core)
        .cloned()
        .flatten()
        .unwrap_or_default()
}

fn run_on(pool: Option<&TaskPool>, task: impl FnOnce() + Send + 'static) {
    match pool {
        Some(pool) => pool.execute_or_run(task),
        None => task(),
    }
}

fn update_clone(name: &str, clone: &ObjectClone, snapshot: Option<&CoreSnapshot>) {
    let Some(snapshot) = snapshot else {
        return;
    };
    if !clone.is_started() {
        log::trace!(target: name, "starting the clone on core {}", snapshot.core_id);
        clone.start();
    }
    clone.update(snapshot);
}

/// Closes `clone`, waits for its workers until `deadline`, and dumps its locks.
fn close_clone(name: &str, clone: &ObjectClone, deadline: Instant) {
    if panic::catch_unwind(AssertUnwindSafe(|| clone.close())).is_err() {
        log::error!(target: name, "failed to close the clone on core {:?}", clone.core());
    }
    if let Some(registry) = clone.liveness_registry()
        && !registry.await_termination(deadline)
    {
        log::warn!(
            target: name,
            "timed out waiting for the workers of the clone on core {:?}; continuing shutdown",
            clone.core()
        );
    }
    clone.dump_locks();
}

#[cfg(test)]
mod tests;
