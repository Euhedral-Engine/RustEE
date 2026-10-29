//! An executor that runs each command on a fresh thread pinned to one logical CPU (upstream
//! `PinnedThreadExecutor`).
//!
//! Executors live in a registry with at most one live executor per CPU. Every handle is one *use* of
//! the executor: cloning a handle adds a use, and releasing or dropping a handle ends its own. The
//! executor closes when its last use ends, as upstream's `Cleaner` did when an executor became
//! unreachable, so a user that is done with a CPU's executor releases it without disturbing the
//! others that share it. [`close`](PinnedThreadExecutor::close) is different: it closes the executor
//! for every user.
//!
//! Divergences from upstream:
//! - No thread priority or daemon flag: std threads have neither, and a Rust process exits when
//!   `main` returns regardless of other threads.
//! - No JVM shutdown hook closing every executor at exit; process exit ends every thread.
//! - Interruption is cooperative: [`shutdown_now`](PinnedThreadExecutor::shutdown_now) and closing
//!   set a flag the task reads with [`is_interrupted`] and unpark the task's thread.
//! - A CPU that is a hole in a sparse topology is rejected up front.

use euhedral_data_structures::sync::LifecycleMutex;
use std::any::Any;
use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;
use std::io;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
#[allow(clippy::disallowed_types, reason = "LIFECYCLE(H8)")]
use std::sync::{Arc, Condvar, MutexGuard, OnceLock, PoisonError, Weak};
use std::thread::{self, JoinHandle, Thread};
use std::time::{Duration, Instant};

thread_local! {
    static INTERRUPTED: RefCell<Option<Arc<AtomicBool>>> = const { RefCell::new(None) };
}

/// Whether the calling pinned task has been asked to stop, by
/// [`shutdown_now`](PinnedThreadExecutor::shutdown_now) or by closing its executor. Always `false`
/// outside a pinned task.
pub fn is_interrupted() -> bool {
    INTERRUPTED.with(|flag| {
        flag.borrow()
            .as_ref()
            .is_some_and(|flag| flag.load(Ordering::Acquire))
    })
}

/// Why a registry call failed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExecutorError {
    /// The CPU is not an active logical CPU.
    InvalidCpu(usize),
    /// The thread name contains a NUL byte.
    InvalidName,
    /// A closed executor for this CPU still has running tasks, so it cannot be replaced yet.
    ClosedWithActiveTasks(usize),
    /// The executor is closed and cannot restart.
    Closed,
}

impl fmt::Display for ExecutorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ExecutorError::InvalidCpu(cpu) => write!(f, "cpu {cpu} is not an active logical CPU"),
            ExecutorError::InvalidName => f.write_str("thread names may not contain NUL bytes"),
            ExecutorError::ClosedWithActiveTasks(cpu) => {
                write!(f, "a closed executor still owns active tasks for CPU {cpu}")
            }
            ExecutorError::Closed => f.write_str("a closed executor cannot restart"),
        }
    }
}

impl std::error::Error for ExecutorError {}

/// Why a command was not accepted.
#[derive(Debug)]
pub enum Rejected {
    /// The executor is shut down or closed.
    NotRunning,
    /// The task thread could not be spawned.
    Spawn(io::Error),
}

impl fmt::Display for Rejected {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Rejected::NotRunning => f.write_str("the executor is not running"),
            Rejected::Spawn(error) => write!(f, "failed to spawn a pinned thread: {error}"),
        }
    }
}

impl std::error::Error for Rejected {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Rejected::Spawn(error) => Some(error),
            Rejected::NotRunning => None,
        }
    }
}

/// The affinity steps around each task, injectable for tests.
pub(crate) trait TaskBinding: Send + Sync {
    /// Opens the managed-CPU scope on the task thread; the guard ends it.
    fn bind(&self, cpu: usize) -> Option<Box<dyn Any>>;
    fn set_affinity(&self, cpu: usize) -> bool;
    fn release_affinity(&self);
}

struct ControllerBinding;

impl TaskBinding for ControllerBinding {
    fn bind(&self, cpu: usize) -> Option<Box<dyn Any>> {
        super::controller()
            .bind_managed_cpu(cpu)
            .map(|binding| Box::new(binding) as Box<dyn Any>)
    }

    fn set_affinity(&self, cpu: usize) -> bool {
        super::controller().set_affinity(cpu)
    }

    fn release_affinity(&self) {
        super::controller().release_affinity();
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State {
    Running,
    Shutdown,
    Closed,
}

/// Interrupts one task of an executor and no other: the flag [`is_interrupted`] reads on that
/// task's thread. Cheap to clone; valid after the task ends, when interrupting is a no-op.
#[derive(Clone, Debug)]
pub struct TaskInterrupt {
    flag: Arc<AtomicBool>,
    finished: Arc<AtomicBool>,
    thread: Thread,
}

impl TaskInterrupt {
    /// Asks the task to stop and wakes its thread if it is parked.
    pub fn interrupt(&self) {
        self.flag.store(true, Ordering::Release);
        self.thread.unpark();
    }

    /// Whether the task has ended and left its executor.
    pub fn is_finished(&self) -> bool {
        self.finished.load(Ordering::Acquire)
    }
}

struct ControlState {
    state: State,
    name: String,
    /// The live handles: the uses of this executor.
    users: usize,
    tasks: HashMap<u64, TaskInterrupt>,
    next_task: u64,
}

/// One executor's lifecycle and its running tasks (upstream `LifecycleControl`).
struct Control {
    state: LifecycleMutex<ControlState>,
    #[allow(clippy::disallowed_types, reason = "LIFECYCLE(H8)")]
    changed: Condvar,
}

impl Control {
    fn lock(&self) -> MutexGuard<'_, ControlState> {
        self.state.lock()
    }

    fn state(&self) -> State {
        self.lock().state
    }

    fn is_empty(&self) -> bool {
        self.lock().tasks.is_empty()
    }

    /// Waits until no task is left, or `deadline`; whether none is.
    fn wait_empty(&self, deadline: Instant) -> bool {
        let mut control = self.lock();
        while !control.tasks.is_empty() {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return false;
            }
            control = self
                .changed
                .wait_timeout(control, remaining)
                .unwrap_or_else(PoisonError::into_inner)
                .0;
        }
        true
    }

    /// Adds a use if the executor can still be had, restarting a shut-down one under `restart` when
    /// that is given. `false` for a closed executor, and for a shut-down one when not restarting.
    /// The check and the new use are one step, so a use never joins an executor that has just closed.
    fn join(&self, restart: Option<&str>) -> bool {
        let mut control = self.lock();
        match (control.state, restart) {
            (State::Running, _) => {}
            (State::Shutdown, Some(name)) => {
                control.state = State::Running;
                control.name = name.to_owned();
                self.changed.notify_all();
            }
            _ => return false,
        }
        control.users += 1;
        true
    }

    /// Ends one use; the last one closes the executor. Returns whether the executor is now closed
    /// with no task running.
    fn leave(&self) -> bool {
        let mut control = self.lock();
        debug_assert!(control.users > 0, "a use ended twice");
        control.users = control.users.saturating_sub(1);
        if control.users > 0 {
            return false;
        }
        Self::close_locked(&self.changed, &mut control)
    }

    /// Closes permanently and interrupts every task; returns whether no task is running.
    fn close(&self) -> bool {
        Self::close_locked(&self.changed, &mut self.lock())
    }

    #[allow(clippy::disallowed_types, reason = "LIFECYCLE(H8)")]
    fn close_locked(changed: &Condvar, control: &mut ControlState) -> bool {
        control.state = State::Closed;
        control.tasks.values().for_each(TaskInterrupt::interrupt);
        changed.notify_all();
        control.tasks.is_empty()
    }

    /// Removes a finished task; returns whether the executor is closed with no task left.
    fn remove_task(&self, id: u64) -> bool {
        let mut control = self.lock();
        control.tasks.remove(&id);
        self.changed.notify_all();
        control.state == State::Closed && control.tasks.is_empty()
    }
}

struct Entry {
    id: u64,
    executor: Weak<Inner>,
    control: Arc<Control>,
}

struct RegistryShared {
    entries: LifecycleMutex<HashMap<usize, Entry>>,
    binding: Arc<dyn TaskBinding>,
    validator: Box<dyn Fn(usize) -> bool + Send + Sync>,
}

impl RegistryShared {
    /// Removes the entry for `cpu` if it is still entry `id` and has no running task.
    fn remove_exact(&self, cpu: usize, id: u64) {
        let mut entries = self.entries.lock();
        if entries
            .get(&cpu)
            .is_some_and(|entry| entry.id == id && entry.control.is_empty())
        {
            entries.remove(&cpu);
        }
    }
}

/// A set of executors, at most one live per CPU (upstream `Registry`). The process-wide one backs
/// [`PinnedThreadExecutor::get_or_create`]; tests build isolated ones.
pub(crate) struct Registry {
    shared: Arc<RegistryShared>,
}

static NEXT_ENTRY: AtomicU64 = AtomicU64::new(0);

impl Registry {
    pub(crate) fn new(
        binding: Arc<dyn TaskBinding>,
        validator: impl Fn(usize) -> bool + Send + Sync + 'static,
    ) -> Self {
        Self {
            shared: Arc::new(RegistryShared {
                entries: LifecycleMutex::new(HashMap::new()),
                binding,
                validator: Box::new(validator),
            }),
        }
    }

    fn global() -> &'static Registry {
        static GLOBAL: OnceLock<Registry> = OnceLock::new();
        GLOBAL.get_or_init(|| {
            Registry::new(Arc::new(ControllerBinding), |cpu| {
                crate::topology::Topology::system().is_active_cpu(cpu)
            })
        })
    }

    fn validate(&self, cpu: usize) -> Result<(), ExecutorError> {
        if (self.shared.validator)(cpu) {
            Ok(())
        } else {
            Err(ExecutorError::InvalidCpu(cpu))
        }
    }

    /// A new use of the live executor for `cpu`: a running one as is, a shut-down one restarted under
    /// `name`, or a new one.
    pub(crate) fn acquire(
        &self,
        cpu: usize,
        name: &str,
    ) -> Result<PinnedThreadExecutor, ExecutorError> {
        self.validate(cpu)?;
        check_name(name)?;
        // An upgraded executor may be the last reference; drop it only after releasing the registry
        // lock.
        let mut stale = None;
        let result = {
            let mut entries = self.shared.entries.lock();
            let reuse = match entries.get(&cpu) {
                Some(entry) => match entry.executor.upgrade() {
                    Some(inner) if inner.control.join(Some(name)) => {
                        Some(Ok(PinnedThreadExecutor::of(inner)))
                    }
                    live => {
                        stale = live;
                        if entry.control.is_empty() {
                            entries.remove(&cpu);
                            None
                        } else {
                            Some(Err(ExecutorError::ClosedWithActiveTasks(cpu)))
                        }
                    }
                },
                None => None,
            };
            reuse.unwrap_or_else(|| {
                let id = NEXT_ENTRY.fetch_add(1, Ordering::Relaxed);
                #[allow(clippy::disallowed_types, reason = "LIFECYCLE(H8)")]
                let control = Arc::new(Control {
                    state: LifecycleMutex::new(ControlState {
                        state: State::Running,
                        name: name.to_owned(),
                        users: 1,
                        tasks: HashMap::new(),
                        next_task: 0,
                    }),
                    changed: Condvar::new(),
                });
                let inner = Arc::new(Inner {
                    cpu,
                    entry: id,
                    control: Arc::clone(&control),
                    registry: Arc::downgrade(&self.shared),
                    binding: Arc::clone(&self.shared.binding),
                });
                entries.insert(
                    cpu,
                    Entry {
                        id,
                        executor: Arc::downgrade(&inner),
                        control,
                    },
                );
                Ok(PinnedThreadExecutor::of(inner))
            })
        };
        drop(stale);
        result
    }

    /// Like [`acquire`](Self::acquire), but a closed executor whose tasks are still winding down is waited for, up to
    /// `timeout`, instead of refused at once.
    pub(crate) fn acquire_within(
        &self,
        cpu: usize,
        name: &str,
        timeout: Duration,
    ) -> Result<PinnedThreadExecutor, ExecutorError> {
        let deadline = Instant::now() + timeout;
        loop {
            match self.acquire(cpu, name) {
                Err(ExecutorError::ClosedWithActiveTasks(busy)) => {
                    let control = self
                        .shared
                        .entries
                        .lock()
                        .get(&cpu)
                        .map(|entry| entry.control.clone());
                    match control {
                        // Wait only for the closed executor `acquire` refused. If another executor has replaced it
                        // since, or it ended, the next `acquire` decides; do not wait on a running one's tasks.
                        Some(control) if control.state() == State::Closed => {
                            if !control.wait_empty(deadline) {
                                return Err(ExecutorError::ClosedWithActiveTasks(busy));
                            }
                        }
                        _ if Instant::now() >= deadline => {
                            return Err(ExecutorError::ClosedWithActiveTasks(busy));
                        }
                        _ => {}
                    }
                }
                other => return other,
            }
        }
    }

    /// A new use of the running executor for `cpu`, if any; never restarts one.
    pub(crate) fn get(&self, cpu: usize) -> Result<Option<PinnedThreadExecutor>, ExecutorError> {
        self.validate(cpu)?;
        let live;
        let found = {
            let mut entries = self.shared.entries.lock();
            let Some(entry) = entries.get(&cpu) else {
                return Ok(None);
            };
            live = entry.executor.upgrade();
            match &live {
                Some(inner) if inner.control.join(None) => {
                    Some(PinnedThreadExecutor::of(Arc::clone(inner)))
                }
                _ => {
                    if (live.is_none() || entry.control.state() == State::Closed)
                        && entry.control.is_empty()
                    {
                        entries.remove(&cpu);
                    }
                    None
                }
            }
        };
        drop(live);
        Ok(found)
    }

    /// Closes every executor; later acquisitions create new ones.
    pub(crate) fn close_all(&self) {
        // An executor with running tasks keeps its entry until the last task exits.
        self.shared
            .entries
            .lock()
            .retain(|_, entry| !entry.control.close());
    }

    #[cfg(test)]
    pub(crate) fn entry_count(&self) -> usize {
        self.shared.entries.lock().len()
    }
}

fn check_name(name: &str) -> Result<(), ExecutorError> {
    if name.contains('\0') {
        Err(ExecutorError::InvalidName)
    } else {
        Ok(())
    }
}

struct Inner {
    cpu: usize,
    entry: u64,
    control: Arc<Control>,
    registry: Weak<RegistryShared>,
    binding: Arc<dyn TaskBinding>,
}

impl Drop for Inner {
    fn drop(&mut self) {
        // The registry entry is removed by the last task or the next registry access: this may run
        // under the registry lock.
        self.control.close();
    }
}

/// Runs each command on a fresh thread pinned to one logical CPU.
///
/// A handle is one use of its CPU's executor, which is shared by everyone who asks for that CPU. The
/// executor closes when its last use ends: when the last handle is dropped or
/// [released](Self::release). A user that only wants to stop its own work releases its handle and
/// interrupts its own tasks ([`submit_interruptible`](Self::submit_interruptible)); it does not call
/// [`close`](Self::close), which closes the executor for everyone.
pub struct PinnedThreadExecutor {
    inner: Arc<Inner>,
    released: AtomicBool,
}

impl Clone for PinnedThreadExecutor {
    /// Another use of the same executor.
    fn clone(&self) -> Self {
        self.inner.control.lock().users += 1;
        Self::of(Arc::clone(&self.inner))
    }
}

impl Drop for PinnedThreadExecutor {
    fn drop(&mut self) {
        self.release();
    }
}

impl fmt::Debug for PinnedThreadExecutor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let control = self.inner.control.lock();
        f.debug_struct("PinnedThreadExecutor")
            .field("cpu", &self.inner.cpu)
            .field("state", &control.state)
            .field("name", &control.name)
            .field("active_tasks", &control.tasks.len())
            .finish()
    }
}

impl PinnedThreadExecutor {
    /// A handle for a use the caller has already counted.
    fn of(inner: Arc<Inner>) -> Self {
        Self {
            inner,
            released: AtomicBool::new(false),
        }
    }

    /// A new use of the one live executor for `cpu`, creating it, or restarting a shut-down one, with
    /// thread name `name` (upstream `getOrSetIfAbsent`).
    pub fn get_or_create(cpu: usize, name: &str) -> Result<Self, ExecutorError> {
        Registry::global().acquire(cpu, name)
    }

    /// Like [`get_or_create`](Self::get_or_create), but a closed executor whose tasks are still winding down is waited
    /// for, up to `timeout`, instead of refused at once: closing an executor returns before its tasks have ended, and
    /// a build on the same CPU right after would otherwise fail with
    /// [`ClosedWithActiveTasks`](ExecutorError::ClosedWithActiveTasks).
    pub fn get_or_create_within(
        cpu: usize,
        name: &str,
        timeout: Duration,
    ) -> Result<Self, ExecutorError> {
        Registry::global().acquire_within(cpu, name, timeout)
    }

    /// A new use of the running executor for `cpu`, if one exists.
    pub fn get(cpu: usize) -> Result<Option<Self>, ExecutorError> {
        Registry::global().get(cpu)
    }

    /// Permanently closes every executor in the process-wide registry.
    pub fn close_all() {
        Registry::global().close_all();
    }

    pub fn cpu(&self) -> usize {
        self.inner.cpu
    }

    /// Restarts a shut-down executor under a new thread name; a no-op while running.
    pub fn start(&self, name: &str) -> Result<(), ExecutorError> {
        check_name(name)?;
        let mut control = self.inner.control.lock();
        match control.state {
            State::Running => Ok(()),
            State::Closed => Err(ExecutorError::Closed),
            State::Shutdown => {
                control.state = State::Running;
                control.name = name.to_owned();
                self.inner.control.changed.notify_all();
                Ok(())
            }
        }
    }

    /// Runs `command` on a new pinned thread.
    pub fn execute(&self, command: impl FnOnce() + Send + 'static) -> Result<(), Rejected> {
        self.submit(command).map(drop)
    }

    /// Runs `task` on a new pinned thread and returns its handle.
    pub fn submit<T: Send + 'static>(
        &self,
        task: impl FnOnce() -> T + Send + 'static,
    ) -> Result<JoinHandle<T>, Rejected> {
        self.submit_interruptible(task).map(|(join, _)| join)
    }

    /// Like [`submit`](Self::submit), and also returns the means to interrupt this task alone. The
    /// executor's other tasks, and their users, are untouched.
    pub fn submit_interruptible<T: Send + 'static>(
        &self,
        task: impl FnOnce() -> T + Send + 'static,
    ) -> Result<(JoinHandle<T>, TaskInterrupt), Rejected> {
        let control = &self.inner.control;
        let mut state = control.lock();
        if state.state != State::Running {
            return Err(Rejected::NotRunning);
        }
        let id = state.next_task;
        state.next_task += 1;
        let interrupted = Arc::new(AtomicBool::new(false));
        let finished = Arc::new(AtomicBool::new(false));
        let finish = TaskFinish {
            finished: Arc::clone(&finished),
            id,
            cpu: self.inner.cpu,
            entry: self.inner.entry,
            control: Arc::clone(control),
            registry: Weak::clone(&self.inner.registry),
        };
        let binding = Arc::clone(&self.inner.binding);
        let (cpu, flag) = (self.inner.cpu, Arc::clone(&interrupted));
        // The task removes itself under this lock, so it cannot finish before it is registered.
        let spawned = thread::Builder::new()
            .name(state.name.clone())
            .spawn(move || {
                let _finish = finish;
                run_pinned(&*binding, cpu, Some(flag), task)
            })
            .map_err(Rejected::Spawn)?;
        let interrupt = TaskInterrupt {
            flag: interrupted,
            finished,
            thread: spawned.thread().clone(),
        };
        state.tasks.insert(id, interrupt.clone());
        Ok((spawned, interrupt))
    }

    /// Spawns a pinned thread outside the lifecycle: it is not counted, not interrupted, and allowed in
    /// any state (upstream's `pinnedFactory`).
    pub fn spawn_untracked<T: Send + 'static>(
        &self,
        task: impl FnOnce() -> T + Send + 'static,
    ) -> io::Result<JoinHandle<T>> {
        let binding = Arc::clone(&self.inner.binding);
        let cpu = self.inner.cpu;
        let name = self.inner.control.lock().name.clone();
        thread::Builder::new()
            .name(name)
            .spawn(move || run_pinned(&*binding, cpu, None, task))
    }

    /// Stops accepting commands; running tasks continue.
    pub fn shutdown(&self) {
        let mut control = self.inner.control.lock();
        if control.state == State::Running {
            control.state = State::Shutdown;
            self.inner.control.changed.notify_all();
        }
    }

    /// Stops accepting commands and interrupts every running task.
    pub fn shutdown_now(&self) {
        let mut control = self.inner.control.lock();
        if control.state == State::Running {
            control.state = State::Shutdown;
        }
        control.tasks.values().for_each(TaskInterrupt::interrupt);
        self.inner.control.changed.notify_all();
    }

    /// Closes permanently for every user and interrupts every running task, whoever submitted it.
    /// Other handles see it closed. A user that only wants to leave calls [`release`](Self::release).
    pub fn close(&self) {
        if self.inner.control.close() {
            self.unregister();
        }
    }

    /// Ends this handle's use of the executor. Other users keep it, running, with their tasks
    /// untouched; the last use to end closes it, as [`close`](Self::close) does. Idempotent, and done
    /// when the handle is dropped. The handle still answers queries afterwards, but no longer keeps
    /// the executor alive.
    pub fn release(&self) {
        if !self.released.swap(true, Ordering::AcqRel) && self.inner.control.leave() {
            self.unregister();
        }
    }

    /// Frees the CPU's registry slot once the executor is closed with no task left.
    fn unregister(&self) {
        if let Some(registry) = self.inner.registry.upgrade() {
            registry.remove_exact(self.inner.cpu, self.inner.entry);
        }
    }

    pub fn is_shutdown(&self) -> bool {
        self.inner.control.state() != State::Running
    }

    pub fn is_closed(&self) -> bool {
        self.inner.control.state() == State::Closed
    }

    /// Not running and no task left.
    pub fn is_terminated(&self) -> bool {
        let control = self.inner.control.lock();
        control.state != State::Running && control.tasks.is_empty()
    }

    /// Waits until terminated; `false` on timeout, or at once while running.
    pub fn await_termination(&self, timeout: Duration) -> bool {
        let deadline = Instant::now().checked_add(timeout);
        let mut control = self.inner.control.lock();
        loop {
            if control.state == State::Running {
                return false;
            }
            if control.tasks.is_empty() {
                return true;
            }
            let remaining = match deadline {
                Some(deadline) => deadline.saturating_duration_since(Instant::now()),
                None => Duration::MAX,
            };
            if remaining.is_zero() {
                return false;
            }
            control = self
                .inner
                .control
                .changed
                .wait_timeout(control, remaining)
                .unwrap_or_else(PoisonError::into_inner)
                .0;
        }
    }

    #[cfg(test)]
    fn active_tasks(&self) -> usize {
        self.inner.control.lock().tasks.len()
    }
}

/// Removes a task from its executor when the task thread ends, panicking or not.
struct TaskFinish {
    finished: Arc<AtomicBool>,
    id: u64,
    cpu: usize,
    entry: u64,
    control: Arc<Control>,
    registry: Weak<RegistryShared>,
}

impl Drop for TaskFinish {
    fn drop(&mut self) {
        let closed = self.control.remove_task(self.id);
        self.finished.store(true, Ordering::Release);
        if closed && let Some(registry) = self.registry.upgrade() {
            registry.remove_exact(self.cpu, self.entry);
        }
    }
}

/// Runs `task` inside the managed binding and affinity, undoing both even if it panics.
fn run_pinned<T>(
    binding: &dyn TaskBinding,
    cpu: usize,
    interrupted: Option<Arc<AtomicBool>>,
    task: impl FnOnce() -> T,
) -> T {
    struct Undo<'a> {
        binding: &'a dyn TaskBinding,
        managed: Option<Box<dyn Any>>,
    }
    impl Drop for Undo<'_> {
        fn drop(&mut self) {
            self.binding.release_affinity();
            drop(self.managed.take());
            INTERRUPTED.with(|flag| flag.borrow_mut().take());
        }
    }
    INTERRUPTED.with(|flag| *flag.borrow_mut() = interrupted);
    let managed = binding.bind(cpu);
    if managed.is_none() {
        log::debug!("no managed binding for logical CPU {cpu}");
    }
    let _undo = Undo { binding, managed };
    if !binding.set_affinity(cpu) {
        log::debug!("could not pin a task to logical CPU {cpu}");
    }
    task()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, mpsc};

    fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
        mutex.lock().unwrap_or_else(PoisonError::into_inner)
    }

    #[derive(Default)]
    struct Recording {
        events: Mutex<Vec<String>>,
    }

    impl TaskBinding for Recording {
        fn bind(&self, cpu: usize) -> Option<Box<dyn Any>> {
            lock(&self.events).push(format!("bind {cpu}"));
            Some(Box::new(()))
        }

        fn set_affinity(&self, cpu: usize) -> bool {
            lock(&self.events).push(format!("pin {cpu}"));
            true
        }

        fn release_affinity(&self) {
            lock(&self.events).push("release".into());
        }
    }

    fn registry() -> (Registry, Arc<Recording>) {
        let binding = Arc::new(Recording::default());
        // CPU 2 is a hole.
        (
            Registry::new(binding.clone(), |cpu| cpu < 4 && cpu != 2),
            binding,
        )
    }

    #[test]
    fn runs_each_command_pinned_and_releases_after() {
        let (registry, binding) = registry();
        let executor = registry.acquire(1, "pinned-1").unwrap();
        let handle = executor
            .submit(|| thread::current().name().map(str::to_owned))
            .unwrap();
        assert_eq!(handle.join().unwrap().as_deref(), Some("pinned-1"));
        executor.shutdown();
        assert!(executor.await_termination(Duration::from_secs(5)));
        assert_eq!(*lock(&binding.events), ["bind 1", "pin 1", "release"]);
    }

    #[test]
    fn holes_and_bad_names_are_rejected() {
        let (registry, _) = registry();
        assert_eq!(
            registry.acquire(2, "x").unwrap_err(),
            ExecutorError::InvalidCpu(2)
        );
        assert_eq!(
            registry.acquire(9, "x").unwrap_err(),
            ExecutorError::InvalidCpu(9)
        );
        assert_eq!(registry.get(2).unwrap_err(), ExecutorError::InvalidCpu(2));
        assert_eq!(
            registry.acquire(0, "a\0b").unwrap_err(),
            ExecutorError::InvalidName
        );
    }

    #[test]
    fn one_live_executor_per_cpu() {
        let (registry, _) = registry();
        let first = registry.acquire(0, "a").unwrap();
        let second = registry.acquire(0, "b").unwrap();
        assert!(Arc::ptr_eq(&first.inner, &second.inner));
        assert!(registry.get(0).unwrap().is_some());
        assert!(registry.get(1).unwrap().is_none());
    }

    #[test]
    fn a_shut_down_executor_restarts_on_acquire() {
        let (registry, _) = registry();
        let executor = registry.acquire(0, "before").unwrap();
        executor.shutdown();
        assert!(matches!(executor.execute(|| {}), Err(Rejected::NotRunning)));
        assert!(registry.get(0).unwrap().is_none(), "get never restarts");
        let restarted = registry.acquire(0, "after").unwrap();
        assert!(Arc::ptr_eq(&executor.inner, &restarted.inner));
        let name = restarted
            .submit(|| thread::current().name().map(str::to_owned))
            .unwrap();
        assert_eq!(name.join().unwrap().as_deref(), Some("after"));
    }

    #[test]
    fn dropping_the_last_handle_closes_and_frees_the_slot() {
        let (registry, _) = registry();
        let executor = registry.acquire(0, "a").unwrap();
        let control = Arc::clone(&executor.inner.control);
        drop(executor);
        assert_eq!(control.state(), State::Closed);
        assert!(registry.get(0).unwrap().is_none());
        assert_eq!(registry.entry_count(), 0);
        let replacement = registry.acquire(0, "b").unwrap();
        assert!(!replacement.is_shutdown());
    }

    #[test]
    fn executor_closes_after_last_user() {
        let (registry, _) = registry();
        let first = registry.acquire(0, "a").unwrap();
        let second = registry.acquire(0, "b").unwrap();
        let third = registry.get(0).unwrap().unwrap();
        let cloned = second.clone();
        let control = Arc::clone(&first.inner.control);
        assert!(Arc::ptr_eq(&first.inner, &third.inner));
        assert_eq!(control.lock().users, 4);

        first.release();
        // Releasing twice ends the use once.
        first.release();
        drop(second);
        third.release();
        assert_eq!(control.state(), State::Running, "a user is left");
        assert!(!cloned.is_shutdown());
        cloned.submit(|| {}).unwrap().join().unwrap();
        assert_eq!(registry.entry_count(), 1);

        drop(cloned);
        assert_eq!(control.state(), State::Closed);
        assert_eq!(registry.entry_count(), 0);
        assert!(registry.get(0).unwrap().is_none());
        // Dropping the released handles ends nothing a second time.
        drop(first);
        let replacement = registry.acquire(0, "c").unwrap();
        assert_eq!(replacement.inner.control.lock().users, 1);
        assert!(!replacement.is_shutdown());
    }

    #[test]
    fn releasing_a_user_leaves_the_other_users_tasks_running() {
        let (registry, _) = registry();
        let leaving = registry.acquire(1, "a").unwrap();
        let staying = registry.acquire(1, "b").unwrap();
        let (started, started_rx) = mpsc::channel();
        let (stop, stop_rx) = mpsc::channel::<()>();
        let kept = staying
            .submit(move || {
                started.send(()).unwrap();
                stop_rx.recv().unwrap();
                is_interrupted()
            })
            .unwrap();
        started_rx.recv().unwrap();
        leaving.release();
        stop.send(()).unwrap();
        assert!(
            !kept.join().unwrap(),
            "the other user's task was not interrupted"
        );
        staying.submit(|| {}).unwrap().join().unwrap();
    }

    #[test]
    fn a_task_interrupt_reaches_its_own_task_only() {
        let (registry, _) = registry();
        let executor = registry.acquire(1, "a").unwrap();
        let (started, started_rx) = mpsc::channel();
        let spin = |started: mpsc::Sender<()>| {
            move || {
                started.send(()).unwrap();
                while !is_interrupted() {
                    thread::park_timeout(Duration::from_millis(10));
                }
            }
        };
        let (mine, mine_interrupt) = executor
            .submit_interruptible(spin(started.clone()))
            .unwrap();
        let (other, other_interrupt) = executor.submit_interruptible(spin(started)).unwrap();
        started_rx.recv().unwrap();
        started_rx.recv().unwrap();
        mine_interrupt.interrupt();
        mine.join().unwrap();
        assert!(mine_interrupt.is_finished());
        assert!(!other_interrupt.is_finished());
        assert!(!other.is_finished(), "the other task is still running");
        assert_eq!(executor.active_tasks(), 1);
        // Interrupting a finished task is harmless.
        mine_interrupt.interrupt();
        other_interrupt.interrupt();
        other.join().unwrap();
    }

    #[test]
    fn a_closed_executor_with_running_tasks_blocks_replacement() {
        let (registry, _) = registry();
        let executor = registry.acquire(3, "a").unwrap();
        let (started, started_rx) = mpsc::channel();
        let (release, release_rx) = mpsc::channel::<()>();
        let handle = executor
            .submit(move || {
                started.send(()).unwrap();
                release_rx.recv().unwrap();
                is_interrupted()
            })
            .unwrap();
        started_rx.recv().unwrap();
        executor.close();
        assert_eq!(
            registry.acquire(3, "b").unwrap_err(),
            ExecutorError::ClosedWithActiveTasks(3)
        );
        release.send(()).unwrap();
        assert!(handle.join().unwrap(), "closing interrupts running tasks");
        assert!(executor.await_termination(Duration::from_secs(5)));
        assert_eq!(registry.entry_count(), 0);
        assert!(registry.acquire(3, "b").is_ok());
    }

    #[test]
    fn acquiring_within_a_timeout_waits_for_a_closed_executors_tasks_to_end() {
        let (registry, _) = registry();
        let executor = registry.acquire(3, "a").unwrap();
        let (started, started_rx) = mpsc::channel();
        let handle = executor
            .submit(move || {
                started.send(()).unwrap();
                while !is_interrupted() {
                    thread::park_timeout(Duration::from_millis(5));
                }
                // Winding down takes a while after the interrupt.
                thread::sleep(Duration::from_millis(80));
            })
            .unwrap();
        started_rx.recv().unwrap();
        executor.close();
        assert_eq!(
            registry.acquire(3, "b").unwrap_err(),
            ExecutorError::ClosedWithActiveTasks(3),
            "the plain acquire refuses at once"
        );
        let started = Instant::now();
        let again = registry
            .acquire_within(3, "b", Duration::from_secs(10))
            .expect("the closed executor's task ended, so the CPU is free");
        assert!(started.elapsed() >= Duration::from_millis(50));
        handle.join().unwrap();
        drop(again);
        // A task that does not end within the timeout is still refused.
        let held = registry.acquire(1, "a").unwrap();
        let (running, running_rx) = mpsc::channel();
        let (release, release_rx) = mpsc::channel::<()>();
        let blocked = held
            .submit(move || {
                running.send(()).unwrap();
                release_rx.recv().unwrap();
            })
            .unwrap();
        running_rx.recv().unwrap();
        held.close();
        assert_eq!(
            registry
                .acquire_within(1, "b", Duration::from_millis(30))
                .unwrap_err(),
            ExecutorError::ClosedWithActiveTasks(1)
        );
        release.send(()).unwrap();
        blocked.join().unwrap();
    }

    #[test]
    fn acquiring_within_a_timeout_returns_as_soon_as_the_closed_executors_tasks_end() {
        let (registry, _) = registry();
        let executor = registry.acquire(1, "a").unwrap();
        let (started, started_rx) = mpsc::channel();
        let (release, release_rx) = mpsc::channel::<()>();
        let old = executor
            .submit(move || {
                started.send(()).unwrap();
                release_rx.recv().unwrap();
            })
            .unwrap();
        started_rx.recv().unwrap();
        executor.close();
        // The closed executor winds down in a moment; a replacement is running by the time the waiter looks again.
        let waiter = {
            let registry = &registry;
            thread::scope(|scope| {
                let waiting = scope.spawn(|| {
                    let started = Instant::now();
                    let got = registry.acquire_within(1, "b", Duration::from_secs(10));
                    (got.is_ok(), started.elapsed())
                });
                thread::sleep(Duration::from_millis(30));
                release.send(()).unwrap();
                old.join().unwrap();
                waiting.join().unwrap()
            })
        };
        assert!(waiter.0, "the CPU was free once the old tasks ended");
        assert!(waiter.1 < Duration::from_secs(5));
    }

    #[test]
    fn shutdown_now_interrupts_but_keeps_the_executor_restartable() {
        let (registry, _) = registry();
        let executor = registry.acquire(1, "a").unwrap();
        let (started, started_rx) = mpsc::channel();
        let handle = executor
            .submit(move || {
                started.send(()).unwrap();
                while !is_interrupted() {
                    thread::park_timeout(Duration::from_millis(10));
                }
            })
            .unwrap();
        started_rx.recv().unwrap();
        assert!(!executor.await_termination(Duration::ZERO));
        executor.shutdown_now();
        handle.join().unwrap();
        assert!(executor.await_termination(Duration::from_secs(5)));
        assert!(executor.is_terminated());
        executor.start("b").unwrap();
        assert!(!executor.is_shutdown());
        assert!(!is_interrupted());
    }

    #[test]
    fn a_panicking_task_still_releases_and_unregisters() {
        let (registry, binding) = registry();
        let executor = registry.acquire(0, "a").unwrap();
        let handle = executor.submit(|| panic!("task failure")).unwrap();
        assert!(handle.join().is_err());
        assert_eq!(executor.active_tasks(), 0);
        assert_eq!(
            lock(&binding.events).last().map(String::as_str),
            Some("release")
        );
    }

    #[test]
    fn close_all_closes_every_executor_and_restart_is_refused() {
        let (registry, _) = registry();
        let zero = registry.acquire(0, "a").unwrap();
        let one = registry.acquire(1, "b").unwrap();
        registry.close_all();
        assert!(zero.is_closed() && one.is_closed());
        assert_eq!(zero.start("c").unwrap_err(), ExecutorError::Closed);
        assert_eq!(registry.entry_count(), 0);
        let fresh = registry.acquire(0, "d").unwrap();
        assert!(!Arc::ptr_eq(&fresh.inner, &zero.inner));
    }

    #[test]
    fn untracked_threads_are_pinned_but_not_counted() {
        let (registry, binding) = registry();
        let executor = registry.acquire(3, "a").unwrap();
        executor.shutdown();
        let handle = executor.spawn_untracked(|| 7).unwrap();
        assert_eq!(handle.join().unwrap(), 7);
        assert!(executor.is_terminated());
        assert_eq!(*lock(&binding.events), ["bind 3", "pin 3", "release"]);
    }
}
