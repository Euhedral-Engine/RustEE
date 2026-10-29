//! The lake: the pool of ready work of one kind, as several ingest sinks that are each an upstream
//! source of the lattice.
//!
//! Producers offer frames from anywhere: a stage that made its successors ready, a driver
//! callback, a request. A frame goes into the sink its routing hash selects, so the frames of one
//! lane (equal hashes) keep to one sink and the frames that may run in parallel spread over all of
//! them. Every sink is a [`QueueIngestSink`] of one partition, attached to the lattice on its own:
//! the workers pull from many sources at once instead of all queuing at one, which keeps producers
//! and consumers off a common cache line.
//!
//! A sink is like a GPU stream. It has one producer partition, so it keeps the order frames were
//! offered in, whatever mixture of ordered and unordered frames it carries, and an ordered frame
//! runs in its own chain's order (frames of one id hash), not in queue order: two chains in one
//! sink may run at the same time. More sinks, not producer partitions, reduce contention.
//!
//! **Units.** The lake counts the units it carries (a quantum, a host task). [`Lake::admit`]
//! counts one in and returns its [`UnitGuard`]; dropping the guard terminates the unit. A frame is
//! accepted only while a unit is admitted, so none is in flight when the sinks go.
//! [`Lake::complete_gracefully`] closes admission; once every admitted unit terminated, the sinks
//! drain what they hold, complete, and the lattice detaches them.
//!
//! **Busy periods.** A worker polls every source attached to the lattice for as long as it is
//! attached, and an idle worker looks again within microseconds, so sinks that stayed attached
//! would keep a core busy after the last request. The sinks are therefore attached for a busy
//! period: beside them the lake attaches an idle watch, a source the workers poll like the sinks,
//! and when no unit has been admitted for the idle time the watch detaches the sinks and itself,
//! leaving the lattice with nothing to poll. The next unit attaches a fresh set.
//! [`Duration::ZERO`] as the idle time keeps the sinks attached.
//!
//! **The hot side is lock-free** (Java's volatile `generation`): the attached set is a [`Published`] reference in the
//! lattice's epoch domain, so [`Lake::offer`] and [`Lake::admit`] read it with one pinned load, with no lock and no
//! reference count. Attaching a set (the first unit of a busy period) and detaching it are rebalance work under a
//! lifecycle lock, taken off the hot scope; a set that is detached stays allocated until the offers that read it
//! have passed.

use std::error::Error;
use std::fmt;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Weak};
use std::time::{Duration, Instant};

use euhedral_core::control_plane::{ControlPlaneLattice, LatticeError};
use euhedral_core::flow_control::UpstreamRegistry;
use euhedral_core::frames::Frame;
use euhedral_core::generics::{DownstreamSlot, FrameSink, LatticeReceiver, LatticeSource};
use euhedral_core::ingest::{IngestSink, QueueIngestSink};
use euhedral_data_structures::sync::{HotScope, LifecycleMutex, Published};

/// Set in the admission word once admission closes; the rest counts the units admitted and not
/// yet terminated.
const CLOSED: u64 = 1 << 63;

/// Why a lake takes no new unit.
#[derive(Debug)]
pub enum LakeClosed {
    /// [`Lake::complete_gracefully`] closed admission, or the lake completed.
    Completed,
    /// The lattice would not take the lake's sinks.
    Lattice(LatticeError),
}

impl fmt::Display for LakeClosed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Completed => f.write_str("the lake is closed"),
            Self::Lattice(error) => {
                write!(f, "the lattice would not take the lake's sinks: {error:?}")
            }
        }
    }
}

impl Error for LakeClosed {}

/// The units a lake carries, and when it last had none. Shared with the idle watch, which the
/// workers poll often, so reading it takes no lock.
struct Admission {
    /// The admission word: [`CLOSED`] and the count of admitted units.
    word: AtomicU64,
    epoch: Instant,
    /// When the last unit terminated, in nanoseconds since `epoch`.
    idle_since: AtomicU64,
}

impl Admission {
    fn new() -> Self {
        Self {
            word: AtomicU64::new(0),
            epoch: Instant::now(),
            idle_since: AtomicU64::new(0),
        }
    }

    fn nanos(&self) -> u64 {
        u64::try_from(self.epoch.elapsed().as_nanos()).unwrap_or(u64::MAX)
    }

    fn units(&self) -> u64 {
        self.word.load(Ordering::Acquire) & !CLOSED
    }

    /// Counts a unit in, unless admission is closed.
    fn admit(&self) -> bool {
        self.word
            .try_update(Ordering::AcqRel, Ordering::Acquire, |word| {
                (word & CLOSED == 0).then_some(word + 1)
            })
            .is_ok()
    }

    /// Counts a unit out; returns the word after.
    fn terminate(&self) -> u64 {
        // Before the count can reach zero, so a watch that sees the lake idle sees when it became so.
        self.idle_since.store(self.nanos(), Ordering::Release);
        let before = self.word.fetch_sub(1, Ordering::AcqRel);
        debug_assert!(
            before & !CLOSED > 0,
            "a unit terminated that was not admitted"
        );
        before - 1
    }

    /// Has the lake been without a unit for `idle`?
    fn idle_for(&self, idle: Duration) -> bool {
        // The count first: a terminate stores the time before it lowers the count, so a watch that sees the count
        // at zero sees the time it reached zero, not an earlier one.
        if self.units() != 0 {
            return false;
        }
        let since = self.idle_since.load(Ordering::Acquire);
        u128::from(self.nanos().saturating_sub(since)) >= idle.as_nanos()
    }
}

/// The sinks attached for one busy period, and the watch that detaches them.
struct Generation {
    id: u64,
    sinks: Box<[QueueIngestSink]>,
    watch: Option<Arc<IdleWatch>>,
}

impl Generation {
    fn sink_of(&self, frame: &dyn Frame) -> &QueueIngestSink {
        let index = frame.header().routing_hash() % self.sinks.len() as u64;
        &self.sinks[index as usize]
    }

    /// The sinks drain what they hold and complete; the watch ends.
    fn finish(&self) {
        for sink in &self.sinks {
            sink.complete_gracefully();
        }
        if let Some(watch) = &self.watch {
            watch.complete();
        }
    }

    /// The sinks never reached the lattice: they refuse frames and cancel what they hold.
    fn abort(&self) {
        for sink in &self.sinks {
            sink.complete();
            sink.clear();
        }
        if let Some(watch) = &self.watch {
            watch.complete();
        }
    }
}

/// The pool of ready work of one kind. See the [module](self) documentation.
pub struct Lake {
    lattice: Arc<ControlPlaneLattice>,
    /// The lattice's registry, whose epoch domain `generation` is read under.
    registry: Arc<UpstreamRegistry>,
    sinks: NonZeroUsize,
    idle_release: Duration,
    admission: Arc<Admission>,
    /// The sinks attached now with their watch, or empty while none are (Java's volatile `generation`): read with
    /// a pinned load, changed under `current`.
    generation: Published<Generation>,
    /// The attach lock (Java's `synchronized (attachedSources)`) and the set it guards: the one attached now. Taken
    /// by `attach`, `detach`, `discard` and `complete`, off the hot scope.
    current: LifecycleMutex<Option<Arc<Generation>>>,
    /// Generations made so far.
    attachments: AtomicU64,
    /// Generations taken out of `generation` whose sinks have not all completed yet: draining.
    retiring: LifecycleMutex<Vec<Arc<Generation>>>,
    finished: AtomicBool,
    this: Weak<Lake>,
}

impl fmt::Debug for Lake {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Lake")
            .field("sinks", &self.sinks)
            .field("idle_release", &self.idle_release)
            .field("active", &self.active())
            .finish_non_exhaustive()
    }
}

impl Lake {
    /// A lake of `sinks` ingest sinks. They attach to `lattice` when the first unit is admitted,
    /// not before, and detach once no unit has been admitted for `idle_release`
    /// ([`Duration::ZERO`]: they stay attached).
    pub fn new(
        lattice: Arc<ControlPlaneLattice>,
        sinks: NonZeroUsize,
        idle_release: Duration,
    ) -> Arc<Self> {
        let registry = lattice.registry().clone();
        Arc::new_cyclic(|this| Self {
            generation: Published::new(registry.epochs().clone(), None),
            registry,
            lattice,
            sinks,
            idle_release,
            admission: Arc::new(Admission::new()),
            current: LifecycleMutex::new(None),
            attachments: AtomicU64::new(0),
            retiring: LifecycleMutex::new(Vec::new()),
            finished: AtomicBool::new(false),
            this: this.clone(),
        })
    }

    /// Counts a unit in and attaches the sinks if they are detached. Dropping the guard
    /// terminates the unit.
    ///
    /// Fails once admission is closed ([`complete_gracefully`](Self::complete_gracefully)), or if
    /// the lattice would not take the sinks.
    pub fn admit(&self) -> Result<UnitGuard, LakeClosed> {
        if !self.admission.admit() {
            return Err(LakeClosed::Completed);
        }
        // Counted first: a watch that finds the lake idle after this cannot detach the set that
        // `attach` returns. The guard ends the unit again if the attach fails.
        let guard = UnitGuard {
            lake: self
                .this
                .upgrade()
                .expect("a lake that is being used is alive"),
        };
        self.ensure_attached()?;
        Ok(guard)
    }

    /// Queues `frame` in the sink its routing hash selects, or hands it back when no unit is
    /// admitted or the lake has completed.
    ///
    /// Java's `publish` loop: read the set, offer to the sink the routing hash picks; a set detached while the
    /// frame was on its way refuses it, and the frame goes to the set attached since. A lake that finished, or an
    /// attach that returns the set that refused, hands the frame back.
    pub fn offer(&self, frame: Box<dyn Frame>) -> Result<(), Box<dyn Frame>> {
        let _hot = HotScope::enter();
        let mut frame = frame;
        let mut refused_by = None;
        loop {
            if self.admission.units() == 0 || self.is_complete() {
                return Err(frame);
            }
            let attempt =
                self.registry
                    .with_pinned(move |guard| match self.generation.load(guard) {
                        None => Attempt::Detached(frame),
                        Some(generation) => match generation.sink_of(&*frame).offer(frame) {
                            Ok(()) => Attempt::Done,
                            Err(refused) => Attempt::Refused(refused, generation.id),
                        },
                    });
            match attempt {
                Attempt::Done => return Ok(()),
                Attempt::Detached(returned) => frame = returned,
                Attempt::Refused(returned, id) => {
                    frame = returned;
                    refused_by = Some(id);
                }
            }
            // The set detached while the frame was on its way, or none is attached: the frame goes to the set
            // attached since, if its unit still holds the lake open.
            match self.attach(true) {
                Ok(next) if refused_by != Some(next.id) => {}
                _ => return Err(frame),
            }
        }
    }

    /// Closes admission. The lake completes once every admitted unit terminated: the sinks drain
    /// what they hold, complete, and the lattice detaches them. Units already admitted may still
    /// offer frames and [`fork`](UnitGuard::fork) more units until then.
    pub fn complete_gracefully(&self) {
        let before = self.admission.word.fetch_or(CLOSED, Ordering::AcqRel);
        if before == 0 {
            self.complete();
        }
    }

    /// Units admitted and not yet terminated.
    pub fn active(&self) -> u64 {
        self.admission.units()
    }

    /// Whether the lake has completed: admission closed and every unit terminated.
    pub fn is_complete(&self) -> bool {
        self.finished.load(Ordering::Acquire)
    }

    /// Whether any sink of the lake is still attached to the lattice: the sinks of a busy period
    /// that is under way, and those of a set that was detached or completed and is still draining
    /// what it holds. False once every sink has delivered its frames and completed.
    pub fn is_attached(&self) -> bool {
        let live =
            |generation: &Generation| generation.sinks.iter().any(|sink| !sink.is_complete());
        if self
            .current
            .lock()
            .as_ref()
            .is_some_and(|generation| live(generation))
        {
            return true;
        }
        let mut retiring = self.retiring.lock();
        retiring.retain(|generation| live(generation));
        !retiring.is_empty()
    }

    /// Finishes a generation taken out of the lake, and keeps track of it until its sinks complete.
    fn retire(&self, generation: Arc<Generation>) {
        // Retiring is rebalance work, also when the idle watch's poll or a frame ends the last unit.
        let _cold = HotScope::leave();
        generation.finish();
        let mut retiring = self.retiring.lock();
        retiring.retain(|generation| generation.sinks.iter().any(|sink| !sink.is_complete()));
        retiring.push(generation);
    }

    /// How many times a set of sinks has been attached: one per busy period.
    pub fn attachments(&self) -> u64 {
        self.attachments.load(Ordering::Acquire)
    }

    /// The number of ingest sinks (upstream sources of the lattice) while attached.
    pub fn sinks(&self) -> usize {
        self.sinks.get()
    }

    /// Makes sure a set is attached: one pinned load while one is.
    fn ensure_attached(&self) -> Result<(), LakeClosed> {
        let attached = self
            .registry
            .with_pinned(|guard| self.generation.load(guard).is_some());
        if attached {
            return Ok(());
        }
        self.attach(false).map(drop)
    }

    /// The sinks attached now; attaches a fresh set (and the watch) when there are none (Java's `attach`, under
    /// the attach lock, which is rebalance work and so taken off the hot scope).
    ///
    /// `for_offer`: the attach is for a frame, so no generation is made unless a unit is admitted
    /// when the lock is held (a unit that ended since the caller looked must not leave sinks
    /// attached for nothing).
    fn attach(&self, for_offer: bool) -> Result<Arc<Generation>, LakeClosed> {
        let _cold = HotScope::leave();
        let (generation, fresh) = {
            let mut current = self.current.lock();
            if let Some(attached) = current.as_ref() {
                return Ok(attached.clone());
            }
            if self.finished.load(Ordering::Acquire) {
                return Err(LakeClosed::Completed);
            }
            if for_offer && self.admission.units() == 0 {
                return Err(LakeClosed::Completed);
            }
            let generation = Arc::new(self.generation_of_sinks());
            self.generation.store(Some(generation.clone())).defer();
            *current = Some(generation.clone());
            (generation, true)
        };
        // Outside the lock: attaching asks the new sources for demand, and a watch may poll at once.
        if fresh && let Err(error) = self.register(&generation) {
            self.discard(&generation);
            return Err(error);
        }
        Ok(generation)
    }

    fn generation_of_sinks(&self) -> Generation {
        let id = self.attachments.fetch_add(1, Ordering::AcqRel);
        Generation {
            id,
            sinks: (0..self.sinks.get())
                .map(|_| QueueIngestSink::new())
                .collect(),
            watch: (!self.idle_release.is_zero()).then(|| {
                Arc::new(IdleWatch {
                    lake: self.this.clone(),
                    admission: self.admission.clone(),
                    idle: self.idle_release,
                    generation: id,
                    downstream: DownstreamSlot::new(),
                })
            }),
        }
    }

    /// Hands the new generation's sources to the lattice.
    fn register(&self, generation: &Generation) -> Result<(), LakeClosed> {
        for sink in &generation.sinks {
            self.lattice
                .add_upstream(sink.delegate())
                .map_err(LakeClosed::Lattice)?;
        }
        if let Some(watch) = &generation.watch {
            self.lattice
                .add_upstream(watch.clone())
                .map_err(LakeClosed::Lattice)?;
        }
        Ok(())
    }

    /// Drops a generation the lattice would not take.
    fn discard(&self, generation: &Arc<Generation>) {
        {
            let mut current = self.current.lock();
            if current
                .as_ref()
                .is_some_and(|attached| Arc::ptr_eq(attached, generation))
            {
                *current = None;
                self.generation.store(None).defer();
            }
        }
        generation.abort();
    }

    /// Detaches generation `id` when no unit is admitted: its watch found the lake idle (Java's `detach`, under
    /// the attach lock).
    fn detach(&self, id: u64) {
        // The idle watch detaches from a poll: rebalance work on a worker's thread.
        let _cold = HotScope::leave();
        let detached = {
            let mut current = self.current.lock();
            // A unit admitted after this check attaches a fresh set: it counts itself before it
            // asks for one.
            if self.admission.units() == 0
                && !self.is_complete()
                && current.as_ref().is_some_and(|attached| attached.id == id)
            {
                self.generation.store(None).defer();
                current.take()
            } else {
                None
            }
        };
        if let Some(generation) = detached {
            self.retire(generation);
        }
    }

    /// Admission is closed and no unit is left: the sinks drain and complete.
    fn complete(&self) {
        // The last unit ending completes the lake, from whichever thread ends it.
        let _cold = HotScope::leave();
        if self.finished.swap(true, Ordering::AcqRel) {
            return;
        }
        let attached = {
            let mut current = self.current.lock();
            self.generation.store(None).defer();
            current.take()
        };
        if let Some(generation) = attached {
            self.retire(generation);
        }
    }
}

impl Drop for Lake {
    /// A lake dropped before it completed lets its sinks drain and complete.
    fn drop(&mut self) {
        let attached = self.current.lock().take();
        if let Some(generation) = attached {
            generation.finish();
        }
    }
}

impl FrameSink for Lake {
    fn offer(&self, frame: Box<dyn Frame>) -> Result<(), Box<dyn Frame>> {
        Lake::offer(self, frame)
    }
}

/// One admitted unit of a lake. Dropping it terminates the unit.
pub struct UnitGuard {
    lake: Arc<Lake>,
}

impl fmt::Debug for UnitGuard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("UnitGuard").finish_non_exhaustive()
    }
}

impl UnitGuard {
    /// Another unit of the same lake, admitted even while the lake drains: a unit that is still
    /// running may start more work before it ends.
    pub fn fork(&self) -> UnitGuard {
        // This guard holds the count above zero, so the sinks are attached and stay so.
        self.lake.admission.word.fetch_add(1, Ordering::AcqRel);
        UnitGuard {
            lake: self.lake.clone(),
        }
    }
}

impl Drop for UnitGuard {
    fn drop(&mut self) {
        if self.lake.admission.terminate() == CLOSED {
            self.lake.complete();
        }
    }
}

/// What one pinned look at the attached set came to.
enum Attempt {
    /// A sink took the frame.
    Done,
    /// No set is attached: the frame comes back.
    Detached(Box<dyn Frame>),
    /// The set `id` refused the frame, having detached: the frame comes back.
    Refused(Box<dyn Frame>, u64),
}

/// A source the workers poll beside the sinks: it notices that the lake has been idle long enough.
struct IdleWatch {
    lake: Weak<Lake>,
    admission: Arc<Admission>,
    idle: Duration,
    generation: u64,
    /// Java's idle watch is an ingest delegate: the downstream is installed once and completed once.
    downstream: DownstreamSlot,
}

impl IdleWatch {
    /// One look at the lake: two atomic loads while it is busy.
    fn poll(&self) {
        if self.downstream.is_complete() || !self.admission.idle_for(self.idle) {
            return;
        }
        match self.lake.upgrade() {
            Some(lake) => lake.detach(self.generation),
            None => self.complete(),
        }
    }
}

impl LatticeSource for IdleWatch {
    fn add_downstream(self: Arc<Self>, downstream: Arc<dyn LatticeReceiver>) {
        let source: Arc<dyn LatticeSource> = self.clone();
        self.downstream.attach(source, downstream);
    }

    /// It holds no frames; a pull is the worker's poll.
    fn pull(
        &self,
        _consumer: &mut dyn FnMut(Box<dyn Frame>),
        _stop: &mut dyn FnMut(&dyn Frame) -> bool,
        _demand: u64,
    ) -> u64 {
        self.poll();
        0
    }

    fn request(&self, _demand: u64) {
        self.poll();
    }

    fn complete(&self) {
        // Completing is lifecycle work, also when a frame does it.
        let _cold = HotScope::leave();
        self.downstream.complete();
    }

    fn is_complete(&self) -> bool {
        self.downstream.is_complete()
    }
}
