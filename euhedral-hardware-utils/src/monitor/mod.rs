//! Resource-pressure monitoring (upstream `monitor` package and `monitor.internal`).
//!
//! A [`ResourceMonitor`] polls a platform provider on a fixed cadence, turns the samples into a
//! smoothed [`HardwareUtilization`], keeps a [`TopologyMapper`] in step with the effective CPUs and
//! hands each result to its listeners on a dedicated thread.

mod compat;
mod dispatcher;
pub(crate) mod engine;
pub(crate) mod pressure;
pub mod sampling;
pub mod utilization;

pub use dispatcher::ListenerId;
pub use utilization::{
    CoreSnapshot, CpuSnapshot, HardwareUtilization, HardwareUtilizationParts, SocketSnapshot,
    SystemSnapshot, UtilizationError,
};

use crate::topology::TopologyMapper;
use compat::LegacyAdapter;
use dispatcher::LatestValueDispatcher;
use engine::SampleStateEngine;
use pressure::PressureState;
use sampling::{DetailedSnapshotProvider, SystemSnapshotProvider};
use std::cell::RefCell;
use std::fmt;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock, PoisonError};
use std::thread::{self, JoinHandle, ThreadId};
use std::time::{Duration, Instant};

/// The default polling cadence.
pub const DEFAULT_PERIOD: Duration = Duration::from_millis(200);

/// A monotonic nanosecond clock (upstream `MonotonicClock`). Only differences between readings are
/// meaningful.
pub trait MonotonicClock: Send + Sync {
    fn now_ns(&self) -> i64;
}

/// The process's monotonic clock: nanoseconds since the first reading.
#[derive(Clone, Copy, Debug, Default)]
pub struct SystemClock;

impl MonotonicClock for SystemClock {
    fn now_ns(&self) -> i64 {
        monotonic_now_ns()
    }
}

/// [`SystemClock`]'s reading; providers use it to timestamp their signals.
pub fn monotonic_now_ns() -> i64 {
    static EPOCH: OnceLock<Instant> = OnceLock::new();
    let elapsed = EPOCH.get_or_init(Instant::now).elapsed();
    i64::try_from(elapsed.as_nanos()).unwrap_or(i64::MAX)
}

/// One reading from this platform's legacy snapshot provider (upstream
/// `SystemInfo.getSystemSnapshot()`). `None` on an unsupported platform or when the read failed.
///
/// The provider is shared by every caller, so its cumulative counters are the caller's to difference;
/// monitors never share it.
pub fn system_snapshot() -> Option<SystemSnapshot> {
    static PROVIDER: OnceLock<Option<Mutex<Box<dyn SystemSnapshotProvider>>>> = OnceLock::new();
    let provider = PROVIDER
        .get_or_init(|| crate::platform::snapshot_provider().map(Mutex::new))
        .as_ref()?;
    provider
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .snapshot()
}

/// Why a monitor could not be built or started.
#[derive(Debug)]
pub enum MonitorError {
    /// The period is zero or does not fit in `i64` nanoseconds.
    InvalidPeriod(Duration),
    /// This platform has no resource provider.
    Unsupported,
    /// The monitor was closed.
    Closed,
    /// A monitor thread could not be spawned.
    Spawn(std::io::Error),
}

impl fmt::Display for MonitorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MonitorError::InvalidPeriod(period) => write!(f, "invalid sample period {period:?}"),
            MonitorError::Unsupported => {
                write!(
                    f,
                    "resource sampling is unavailable on {}",
                    crate::util::OsName::current()
                )
            }
            MonitorError::Closed => f.write_str("the monitor is closed"),
            MonitorError::Spawn(error) => write!(f, "failed to spawn a monitor thread: {error}"),
        }
    }
}

impl std::error::Error for MonitorError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            MonitorError::Spawn(error) => Some(error),
            _ => None,
        }
    }
}

/// A monitor's lifecycle.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MonitorState {
    /// Built, never started.
    New,
    /// The poller is starting.
    Starting,
    Running,
    /// Stopped; [`ResourceMonitor::start`] resumes.
    Stopped,
    /// Shutting down; no further results are published.
    Closing,
    Closed,
}

type TopologyUpdate = Box<dyn Fn(&HardwareUtilization) + Send + Sync>;

struct Lifecycle {
    state: MonitorState,
    /// Set by `stop`: the poller takes one more reading so a stopped monitor's last value is fresh.
    coalesced_read: bool,
    /// Set by a restart: the next evaluation starts every counter from a new baseline, so the first
    /// interval does not span the stopped period.
    reset_counters: bool,
    evaluating: usize,
    poller_alive: bool,
    poller: Option<ThreadId>,
}

struct Evaluation {
    provider: Box<dyn DetailedSnapshotProvider>,
    engine: SampleStateEngine,
    pressure: PressureState,
    failures: u64,
}

struct Shared {
    id: u64,
    period_ns: i64,
    clock: Arc<dyn MonotonicClock>,
    /// Whether the poller sleeps until each tick's deadline; tests with a synthetic clock tick
    /// without waiting.
    paced: bool,
    update_topology: TopologyUpdate,
    lifecycle: Mutex<Lifecycle>,
    changed: Condvar,
    evaluation: Mutex<Evaluation>,
    last: Mutex<Option<Arc<HardwareUtilization>>>,
    dispatcher: LatestValueDispatcher<Arc<HardwareUtilization>>,
}

thread_local! {
    /// The monitors whose evaluation this thread is running, so a call back into one of them from
    /// its provider or topology update does not wait on itself.
    static EVALUATING: RefCell<Vec<u64>> = const { RefCell::new(Vec::new()) };
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    // Every critical section here leaves its data consistent, and user code runs under a lock only
    // inside `catch_unwind`, so a poisoned lock carries nothing to distrust.
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

impl Shared {
    fn life(&self) -> MutexGuard<'_, Lifecycle> {
        lock(&self.lifecycle)
    }

    fn in_evaluation(&self) -> bool {
        EVALUATING.with(|stack| stack.borrow().contains(&self.id))
    }

    /// Whether the caller would wait on itself: the poller, a listener, or code inside an evaluation.
    fn is_own_thread(&self, life: &Lifecycle) -> bool {
        life.poller == Some(thread::current().id())
            || self.dispatcher.is_dispatch_thread()
            || self.in_evaluation()
    }

    /// Moves `Closing` to `Closed` once no evaluation is in flight.
    fn try_finish_close(&self, life: &mut Lifecycle) {
        if life.state == MonitorState::Closing && life.evaluating == 0 {
            life.state = MonitorState::Closed;
            self.changed.notify_all();
        }
    }

    fn fully_closed(&self, life: &Lifecycle) -> bool {
        life.state == MonitorState::Closed && !life.poller_alive && self.dispatcher.is_closed()
    }

    fn begin_close(self: &Arc<Self>, life: &mut Lifecycle) {
        if matches!(life.state, MonitorState::Closing | MonitorState::Closed) {
            return;
        }
        life.state = MonitorState::Closing;
        let weak = Arc::downgrade(self);
        self.dispatcher.begin_close(Some(Box::new(move || {
            if let Some(shared) = weak.upgrade() {
                let _life = shared.life();
                shared.changed.notify_all();
            }
        })));
        self.try_finish_close(life);
        self.changed.notify_all();
    }

    /// Takes one sample and publishes it unless the monitor is closing (upstream
    /// `evaluateAndPublish`). A panicking provider is logged and its state discarded; the monitor
    /// keeps running.
    ///
    /// The reading after a stop (`tick == false`) only refreshes [`ResourceMonitor::utilization`]:
    /// it never reaches listeners and leaves a pending counter reset to the next tick, even when a
    /// restart overtakes it.
    fn evaluate_and_publish(&self, tick: bool) {
        let begun = self.begin_evaluation(&mut self.life(), tick);
        if let Some(reset) = begun {
            self.run_evaluation(reset, tick);
        }
    }

    /// Counts an evaluation as in flight, or returns `None` when the monitor is closing. The result
    /// says whether the evaluation must reset the counters first.
    fn begin_evaluation(&self, life: &mut Lifecycle, tick: bool) -> Option<bool> {
        if matches!(life.state, MonitorState::Closing | MonitorState::Closed) {
            return None;
        }
        life.evaluating += 1;
        Some(tick && std::mem::take(&mut life.reset_counters))
    }

    /// Runs an evaluation counted by [`begin_evaluation`](Self::begin_evaluation).
    fn run_evaluation(&self, reset: bool, tick: bool) {
        EVALUATING.with(|stack| stack.borrow_mut().push(self.id));
        {
            let mut evaluation = lock(&self.evaluation);
            if reset {
                evaluation.engine.reset_counter_state();
            }
            let outcome = catch_unwind(AssertUnwindSafe(|| self.evaluate(&mut evaluation, tick)));
            if outcome.is_err() {
                evaluation.engine.clear_all();
                evaluation.pressure.clear();
                evaluation.failures += 1;
                if evaluation.failures == 1 {
                    log::warn!(
                        "resource monitor evaluation panicked; continuing with the next tick"
                    );
                } else {
                    log::debug!(
                        "resource monitor evaluation panicked again (failure #{})",
                        evaluation.failures
                    );
                }
            }
        }
        EVALUATING.with(|stack| {
            let mut stack = stack.borrow_mut();
            if let Some(index) = stack.iter().rposition(|&id| id == self.id) {
                stack.remove(index);
            }
        });
        let mut life = self.life();
        life.evaluating -= 1;
        self.try_finish_close(&mut life);
    }

    fn evaluate(&self, evaluation: &mut Evaluation, tick: bool) {
        let poll_start_ns = self.clock.now_ns();
        if evaluation.engine.is_slow_due(poll_start_ns) {
            let slow = evaluation.provider.sample_slow(poll_start_ns);
            evaluation.engine.process_slow(poll_start_ns, slow);
        }
        let fast = evaluation.provider.sample_fast(poll_start_ns);
        let evaluation_ns = self.clock.now_ns();
        let Some(interval) = evaluation.engine.process_fast(evaluation_ns, fast.as_ref()) else {
            return;
        };
        let utilization = Arc::new(pressure::evaluate(
            &interval,
            &mut evaluation.pressure,
            evaluation_ns,
        ));
        *lock(&self.last) = Some(Arc::clone(&utilization));
        let state = self.life().state;
        if !matches!(state, MonitorState::Closing | MonitorState::Closed) {
            (self.update_topology)(&utilization);
            if tick && state == MonitorState::Running {
                self.dispatcher.offer(utilization);
            }
        }
    }
}

/// Polls resource usage and publishes smoothed pressure (upstream `ResourceMonitor`).
///
/// Lifecycle: [`New`](MonitorState::New) → [`start`](Self::start) → `Running` ⇄ `Stopped` →
/// [`close`](Self::close) → `Closed`. Dropping the monitor closes it.
///
/// Results go to listeners on a dedicated thread, latest value first: a slow listener skips
/// intermediate values rather than queueing them. Each evaluation also updates the monitor's
/// [`TopologyMapper`]. Each monitor owns its provider, so monitors never share counter intervals.
pub struct ResourceMonitor {
    shared: Arc<Shared>,
    poller: Mutex<Option<JoinHandle<()>>>,
}

impl ResourceMonitor {
    /// A monitor of this platform's resources at the default 200 ms cadence.
    pub fn new(mapper: Arc<TopologyMapper>) -> Result<Self, MonitorError> {
        Self::with_period(mapper, DEFAULT_PERIOD)
    }

    /// A monitor of this platform's resources, or [`MonitorError::Unsupported`] on a platform without
    /// a provider.
    pub fn with_period(
        mapper: Arc<TopologyMapper>,
        period: Duration,
    ) -> Result<Self, MonitorError> {
        let provider = crate::platform::detailed_provider(mapper.topology())
            .ok_or(MonitorError::Unsupported)?;
        Self::with_provider(mapper, period, provider)
    }

    /// A monitor over a caller-supplied detailed provider.
    pub fn with_provider(
        mapper: Arc<TopologyMapper>,
        period: Duration,
        provider: Box<dyn DetailedSnapshotProvider>,
    ) -> Result<Self, MonitorError> {
        let span = mapper.topology().cpu_span();
        let update: TopologyUpdate = Box::new(move |utilization| mapper.update_from(utilization));
        Self::build(span, period, provider, Arc::new(SystemClock), true, update)
    }

    /// A monitor over a legacy snapshot provider (upstream wrapped these in
    /// `SystemSnapshotCompatibilityAdapter`).
    pub fn with_legacy_provider(
        mapper: Arc<TopologyMapper>,
        period: Duration,
        provider: impl SystemSnapshotProvider + 'static,
    ) -> Result<Self, MonitorError> {
        Self::with_provider(mapper, period, Box::new(LegacyAdapter::new(provider)))
    }

    fn build(
        span: usize,
        period: Duration,
        provider: Box<dyn DetailedSnapshotProvider>,
        clock: Arc<dyn MonotonicClock>,
        paced: bool,
        update_topology: TopologyUpdate,
    ) -> Result<Self, MonitorError> {
        static NEXT_ID: AtomicU64 = AtomicU64::new(0);
        let period_ns = i64::try_from(period.as_nanos())
            .ok()
            .filter(|&ns| ns > 0)
            .ok_or(MonitorError::InvalidPeriod(period))?;
        let span = span.max(1);
        let dispatcher = LatestValueDispatcher::new("euhedral-resource-monitor-dispatcher")
            .map_err(MonitorError::Spawn)?;
        Ok(Self {
            shared: Arc::new(Shared {
                id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
                period_ns,
                clock,
                paced,
                update_topology,
                lifecycle: Mutex::new(Lifecycle {
                    state: MonitorState::New,
                    coalesced_read: false,
                    reset_counters: false,
                    evaluating: 0,
                    poller_alive: false,
                    poller: None,
                }),
                changed: Condvar::new(),
                evaluation: Mutex::new(Evaluation {
                    provider,
                    engine: SampleStateEngine::new(span, period_ns),
                    pressure: PressureState::new(span),
                    failures: 0,
                }),
                last: Mutex::new(None),
                dispatcher,
            }),
            poller: Mutex::new(None),
        })
    }

    pub fn state(&self) -> MonitorState {
        self.shared.life().state
    }

    /// Starts polling, or resumes after [`stop`](Self::stop); a no-op while running. Fails on a closed
    /// monitor, or when the polling thread cannot be spawned (the monitor stays `New` and a later call
    /// may retry).
    pub fn start(&self) -> Result<(), MonitorError> {
        let mut life = self.shared.life();
        match life.state {
            MonitorState::Closing | MonitorState::Closed => Err(MonitorError::Closed),
            MonitorState::Starting | MonitorState::Running => Ok(()),
            MonitorState::Stopped => {
                life.state = MonitorState::Running;
                life.coalesced_read = false;
                life.reset_counters = true;
                self.shared.changed.notify_all();
                Ok(())
            }
            MonitorState::New => {
                life.state = MonitorState::Starting;
                life.poller_alive = true;
                let shared = Arc::clone(&self.shared);
                let spawned = thread::Builder::new()
                    .name("euhedral-resource-monitor".into())
                    .spawn(move || poll(&shared));
                match spawned {
                    Ok(handle) => {
                        life.poller = Some(handle.thread().id());
                        *lock(&self.poller) = Some(handle);
                        Ok(())
                    }
                    Err(error) => {
                        life.state = MonitorState::New;
                        life.poller_alive = false;
                        Err(MonitorError::Spawn(error))
                    }
                }
            }
        }
    }

    /// Stops publishing to listeners. The poller takes one more reading so
    /// [`utilization`](Self::utilization) is fresh, then idles until [`start`](Self::start).
    pub fn stop(&self) {
        let mut life = self.shared.life();
        if matches!(life.state, MonitorState::Starting | MonitorState::Running) {
            life.state = MonitorState::Stopped;
            life.coalesced_read = true;
            self.shared.changed.notify_all();
        }
    }

    /// Closes the monitor and waits for shutdown, unless called from a listener, from the poller or
    /// from inside an evaluation: those cannot wait for themselves, so they start the shutdown and
    /// return.
    pub fn close(&self) {
        self.close_within(None);
    }

    /// Closes the monitor, waiting at most `timeout`. Returns whether it is fully closed (its threads
    /// done). From a listener, the poller or inside an evaluation it never waits.
    pub fn close_timeout(&self, timeout: Duration) -> bool {
        self.close_within(Some(timeout))
    }

    fn close_within(&self, timeout: Option<Duration>) -> bool {
        let shared = &self.shared;
        let deadline = timeout.and_then(|timeout| Instant::now().checked_add(timeout));
        let mut life = shared.life();
        shared.begin_close(&mut life);
        if shared.is_own_thread(&life) {
            return shared.fully_closed(&life);
        }
        while !shared.fully_closed(&life) {
            life = match (timeout, deadline) {
                (None, _) | (Some(_), None) => shared
                    .changed
                    .wait(life)
                    .unwrap_or_else(PoisonError::into_inner),
                (Some(_), Some(deadline)) => {
                    let remaining = deadline.saturating_duration_since(Instant::now());
                    if remaining.is_zero() {
                        return false;
                    }
                    // The dispatcher's close hook signals through this condvar; the bounded wait
                    // also covers a hook that could not run because the dispatcher was dropped.
                    let wait = remaining.min(Duration::from_millis(50));
                    shared
                        .changed
                        .wait_timeout(life, wait)
                        .unwrap_or_else(PoisonError::into_inner)
                        .0
                }
            };
        }
        drop(life);
        if let Some(handle) = lock(&self.poller).take() {
            let _ = handle.join();
        }
        true
    }

    /// Registers a listener for each published result while running.
    pub fn add_listener(
        &self,
        listener: impl Fn(&Arc<HardwareUtilization>) + Send + Sync + 'static,
    ) -> ListenerId {
        self.shared.dispatcher.add_listener(listener)
    }

    pub fn remove_listener(&self, id: ListenerId) -> bool {
        self.shared.dispatcher.remove_listener(id)
    }

    /// The latest result. Before the first published result, a `New` or `Stopped` monitor evaluates
    /// once on the calling thread. `None` when no result exists yet (for instance, the provider failed).
    pub fn utilization(&self) -> Option<Arc<HardwareUtilization>> {
        if let Some(utilization) = lock(&self.shared.last).clone() {
            return Some(utilization);
        }
        let state = self.state();
        if matches!(state, MonitorState::New | MonitorState::Stopped)
            && !self.shared.in_evaluation()
        {
            self.shared.evaluate_and_publish(false);
        }
        lock(&self.shared.last).clone()
    }
}

impl Drop for ResourceMonitor {
    fn drop(&mut self) {
        self.close();
    }
}

impl fmt::Debug for ResourceMonitor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ResourceMonitor")
            .field("state", &self.state())
            .field("period_ns", &self.shared.period_ns)
            .finish_non_exhaustive()
    }
}

/// Marks the poller gone on every exit path, including a panic, and closes the monitor so no caller
/// waits for a poller that no longer exists.
struct PollerExit<'a>(&'a Arc<Shared>);

impl Drop for PollerExit<'_> {
    fn drop(&mut self) {
        let shared = self.0;
        let mut life = shared.life();
        if thread::panicking() {
            log::error!("resource monitor poller terminated by a panic; closing the monitor");
        }
        shared.begin_close(&mut life);
        life.poller_alive = false;
        shared.try_finish_close(&mut life);
        shared.changed.notify_all();
    }
}

enum Wake {
    Tick,
    Reanchor,
    Recheck,
}

/// The polling loop: ticks on a grid anchored at start, skipping missed points rather than catching
/// up, and re-anchoring when the clock goes backwards.
fn poll(shared: &Arc<Shared>) {
    let _exit = PollerExit(shared);
    let period = shared.period_ns;
    let mut anchor = shared.clock.now_ns();
    let mut next = anchor.saturating_add(period);
    let mut last_now = anchor;
    loop {
        {
            let mut life = shared.life();
            match life.state {
                MonitorState::Closing | MonitorState::Closed => return,
                MonitorState::Stopped => {
                    if std::mem::take(&mut life.coalesced_read) {
                        // Counted under the same lock that took the flag, so the reading is never
                        // invisible to an observer of the lifecycle.
                        let begun = shared.begin_evaluation(&mut life, false);
                        drop(life);
                        if let Some(reset) = begun {
                            shared.run_evaluation(reset, false);
                        }
                    } else {
                        drop(
                            shared
                                .changed
                                .wait(life)
                                .unwrap_or_else(PoisonError::into_inner),
                        );
                    }
                    continue;
                }
                MonitorState::Starting | MonitorState::New => life.state = MonitorState::Running,
                MonitorState::Running => {}
            }
        }
        let now = shared.clock.now_ns();
        if now < last_now {
            anchor = now;
            next = now.saturating_add(period);
        } else if now >= next {
            let skipped = (now - anchor) / period;
            next = anchor.saturating_add((skipped + 1).saturating_mul(period));
        }
        last_now = now;
        match wait_for_tick(shared, next, last_now) {
            Wake::Tick => {
                shared.evaluate_and_publish(true);
                last_now = last_now.max(shared.clock.now_ns());
            }
            Wake::Reanchor => {
                anchor = shared.clock.now_ns();
                next = anchor.saturating_add(period);
                last_now = anchor;
            }
            Wake::Recheck => {}
        }
    }
}

/// Waits until the clock reaches `deadline` while running.
fn wait_for_tick(shared: &Shared, deadline: i64, last_now: i64) -> Wake {
    let mut life = shared.life();
    loop {
        if life.state != MonitorState::Running {
            return Wake::Recheck;
        }
        let now = shared.clock.now_ns();
        if now < last_now {
            return Wake::Reanchor;
        }
        if now >= deadline || !shared.paced {
            return Wake::Tick;
        }
        let remaining = Duration::from_nanos(u64::try_from(deadline - now).unwrap_or(0));
        life = shared
            .changed
            .wait_timeout(life, remaining)
            .unwrap_or_else(PoisonError::into_inner)
            .0;
    }
}

#[cfg(test)]
mod tests;
