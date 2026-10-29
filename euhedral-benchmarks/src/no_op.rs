//! The benchmark's frame and source.
//!
//! [`NoOpFrame`] does no work: finishing it bumps its CPU's cell of a striped completion counter and hands the frame back to its source,
//! so a fixed set of frames circulates, as the same frame objects are pushed again and again. [`RepeatingSource`]
//! pushes those frames to whatever receiver the lattice attaches, as many as the demand allows.

use crate::latency::Recorders;
use euhedral_core::frames::{Frame, FrameHeader};
use euhedral_core::generics::{DownstreamSlot, LatticeReceiver, LatticeSource, SourceCell};
use euhedral_data_structures::atomics::PaddedLongAdder;
use euhedral_data_structures::queues::{Consumer, MpscQueue, Multi, Producer, Single};
use euhedral_hardware_utils::affinity;
use euhedral_hardware_utils::util::CpuSet;
use std::cell::Cell;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

/// One frame in this many is stamped and timed.
pub const SAMPLE_EVERY: u64 = 64;

/// How many frames a source circulates.
pub const FRAMES_PER_SOURCE: usize = 2_048;

/// Where a thread that finishes frames counts and records: a cell of the [`Tally`] and whether the thread is the only
/// writer of it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Slot {
    pub index: usize,
    /// Whether no other thread writes the cell, so that writing it is a load and a store.
    pub owned: bool,
}

/// The completion counter: a [`PaddedLongAdder`] (Java's `PaddedLongAdder(availableProcessors, true, true)`: checked
/// indexes, and each cell padded to two cache lines so the adjacent-line prefetcher never couples two cells) with one
/// cell per CPU id.
///
/// A worker is pinned to its own CPU, so the cell of the CPU it runs on has exactly one writer and finishing a frame
/// is a load and a store, not a read-modify-write. The CPU is read once, when the thread first finishes a frame, from
/// the worker's affinity, and the cell is cached for the thread; nothing asks the OS per increment. The cell is
/// claimed on that first use, so two threads that resolve to the same CPU (workers that share one, an unpinned thread)
/// cannot both own it: the second one, and any thread whose CPU is not in the worker set or is unknown (a platform that
/// does not pin), takes a shared cell by its arrival order, which is a `fetch_add`, and says so once on stderr. The
/// harness thread never counts.
///
/// The cells after the first `cpus` are those shared cells. The harness sums all of them with loads.
///
/// A claimed cell stays claimed: a tally lives for one fork (one lattice), and its worker threads end with it, so
/// nothing is released on a worker's exit.
///
/// **Where it differs from Java.** Java's `NoOpFrame` counts with `PaddedLongAdder.getAndAdd`, a read-modify-write on a
/// slot chosen by CPU id. Here a cell with one writer is a load and a store (the user's choice, for a path that takes
/// no locked instruction); only the shared fallback keeps the add.
pub struct Tally {
    id: u64,
    adder: PaddedLongAdder,
    cpus: usize,
    workers: CpuSet,
    claimed: Vec<AtomicBool>,
    next_shared: AtomicUsize,
}

/// Distinguishes tallies, so that a thread-local cell of an earlier tally is never taken for this one's.
static NEXT_TALLY: AtomicU64 = AtomicU64::new(0);
/// Whether the fallback was already reported.
static FALLBACK_REPORTED: AtomicBool = AtomicBool::new(false);

thread_local! {
    /// The calling thread's cell, as `(tally id, slot)`.
    static RESOLVED: Cell<Option<(u64, Slot)>> = const { Cell::new(None) };
}

impl Tally {
    /// A tally for workers pinned to `workers`, with a cell for each of `cpus` CPU ids.
    ///
    /// # Panics
    ///
    /// If a worker CPU is not below `cpus`.
    pub fn new(cpus: usize, workers: CpuSet) -> Self {
        let cpus = cpus.max(1);
        assert!(
            workers.last().is_none_or(|last| last < cpus),
            "a worker CPU is beyond the {cpus} cells"
        );
        Self {
            id: NEXT_TALLY.fetch_add(1, Ordering::Relaxed),
            // One cell per CPU id, then as many shared cells as there are CPU ids.
            adder: PaddedLongAdder::with_options(cpus * 2, true, true),
            cpus,
            workers,
            claimed: (0..cpus).map(|_| AtomicBool::new(false)).collect(),
            next_shared: AtomicUsize::new(0),
        }
    }

    /// The cells, for the harness to sum.
    pub fn adder(&self) -> &PaddedLongAdder {
        &self.adder
    }

    /// The number of cells that record latency: one per cell of the adder.
    pub fn cells(&self) -> usize {
        self.cpus * 2
    }

    /// The calling thread's cell, resolved on its first call.
    pub fn slot(&self) -> Slot {
        RESOLVED.with(|resolved| {
            if let Some((id, slot)) = resolved.get()
                && id == self.id
            {
                return slot;
            }
            let slot = self.resolve();
            resolved.set(Some((self.id, slot)));
            slot
        })
    }

    fn resolve(&self) -> Slot {
        let cpu = affinity::current_cpu();
        if let Some(cpu) = cpu
            && self.workers.contains(cpu)
            && !self.claimed[cpu].swap(true, Ordering::AcqRel)
        {
            return Slot {
                index: cpu,
                owned: true,
            };
        }
        if !FALLBACK_REPORTED.swap(true, Ordering::Relaxed) {
            eprintln!(
                "a thread that finishes frames is not the only one on its CPU ({cpu:?}) or is not pinned to a worker \
                 CPU; it counts into a shared cell with a read-modify-write"
            );
        }
        let shared = self.next_shared.fetch_add(1, Ordering::Relaxed) % self.cpus;
        Slot {
            index: self.cpus + shared,
            owned: false,
        }
    }

    /// Counts one completion in the calling thread's cell.
    #[inline]
    pub fn increment(&self) {
        let slot = self.slot();
        if slot.owned {
            let count = self.adder.get_opaque(slot.index);
            self.adder.set_release(slot.index, count + 1);
        } else {
            self.adder.increment_at(slot.index);
        }
    }
}

/// Where a finished frame goes back to: its source's queue. Leaked once per source, so that finishing a frame touches
/// no reference count that every worker shares; a benchmark process builds a handful.
type Home = &'static Producer<Box<NoOpFrame>, Multi>;

/// A frame that does nothing, and counts itself when it finishes (`NoOpFrame` upstream).
pub struct NoOpFrame {
    header: FrameHeader,
    tally: Arc<Tally>,
    /// The source's queue, or `None` for a frame of a pooled source, which goes back to the pool of the thread that
    /// finished it.
    home: Option<Home>,
    /// Where the latency of a sampled frame is recorded; `None` for a frame that is not sampled.
    latency: Option<Arc<Recorders>>,
    /// When the source handed the frame out, if it is being sampled.
    stamp: Option<Instant>,
}

impl NoOpFrame {
    /// Notes the time a sampled frame leaves its source.
    fn stamp(&mut self) {
        if self.latency.is_some() {
            self.stamp = Some(Instant::now());
        }
    }
}

impl Frame for NoOpFrame {
    fn header(&self) -> &FrameHeader {
        &self.header
    }

    fn header_mut(&mut self) -> &mut FrameHeader {
        &mut self.header
    }

    /// Records the time since a sampled frame left its source, on the worker that runs it.
    fn execute(&mut self) -> Result<(), euhedral_core::frames::FrameError> {
        if let Some(stamp) = self.stamp.take()
            && let Some(recorders) = &self.latency
        {
            let nanos = u64::try_from(stamp.elapsed().as_nanos()).unwrap_or(u64::MAX);
            let slot = self.tally.slot();
            recorders.slot(slot.index).record(nanos, slot.owned);
        }
        Ok(())
    }

    /// Counts the completion, then returns the frame to its source.
    fn do_finally(self: Box<Self>) {
        self.tally.increment();
        self.recycle();
    }

    fn recycle(self: Box<Self>) {
        match self.home {
            // The queue is unbounded: a frame always fits.
            Some(home) => {
                let _ = home.offer(self);
            }
            None => pool_put(self),
        }
    }
}

/// The finished frames of a pooled source, per thread: a frame goes back to the pool of the thread that finished it
/// and the next request on that thread takes it, so a frame crosses no core between two uses unless the work moved.
const POOL_CAP: usize = 4096;

thread_local! {
    #[allow(clippy::vec_box, reason = "the frames are pushed as boxes and come back as boxes")]
    static POOL: std::cell::RefCell<Vec<Box<NoOpFrame>>> = const { std::cell::RefCell::new(Vec::new()) };
}

fn pool_put(frame: Box<NoOpFrame>) {
    let _ = POOL.try_with(|pool| {
        let mut pool = pool.borrow_mut();
        if pool.len() < POOL_CAP {
            pool.push(frame);
        }
    });
}

fn pool_take() -> Option<Box<NoOpFrame>> {
    POOL.try_with(|pool| pool.borrow_mut().pop()).ok().flatten()
}

/// Makes frames for a pooled source when its thread's pool is empty.
struct Maker {
    id_hash: u64,
    seed: u64,
    made: std::sync::atomic::AtomicU64,
    tally: Arc<Tally>,
    latency: Option<Arc<Recorders>>,
}

impl Maker {
    fn fresh(&self) -> Box<NoOpFrame> {
        let index = self.made.fetch_add(1, Ordering::Relaxed);
        let mut header = FrameHeader::new(self.id_hash);
        header.randomize_hash(self.seed.wrapping_add(index));
        Box::new(NoOpFrame {
            header,
            tally: self.tally.clone(),
            home: None,
            latency: self
                .latency
                .as_ref()
                .filter(|_| index.is_multiple_of(SAMPLE_EVERY))
                .cloned(),
            stamp: None,
        })
    }
}

/// `length` unordered frames of `id_hash`, each with a routing hash of its own from `seed`
/// (`NoOpFrame.generate` upstream).
pub fn generate(
    id_hash: u64,
    length: usize,
    seed: u64,
    tally: &Arc<Tally>,
    latency: Option<&Arc<Recorders>>,
    home: Home,
) -> Vec<Box<NoOpFrame>> {
    (0..length as u64)
        .map(|index| {
            let mut header = FrameHeader::new(id_hash);
            header.randomize_hash(seed.wrapping_add(index));
            Box::new(NoOpFrame {
                header,
                tally: tally.clone(),
                home: Some(home),
                // One frame in `SAMPLE_EVERY` is sampled, every time it circulates.
                latency: latency.filter(|_| index % SAMPLE_EVERY == 0).cloned(),
                stamp: None,
            })
        })
        .collect()
}

/// What only a source's `request` and `pull` touch: the queue the finished frames come back to, read by its one
/// consumer, and the banked demand.
struct Inner {
    /// The queue the finished frames come back to; `None` for a pooled source.
    frames: Option<Consumer<Box<NoOpFrame>, Single>>,
    /// Banked demand; `u64::MAX` is unbounded. The demand a request leaves unmet stays banked for the next one.
    demand: u64,
}

/// A source that pushes a fixed set of [`NoOpFrame`]s over and over (`RepeatingSink` upstream): a request or pull takes
/// the next frames that are back from the lattice, as many as the demand allows.
///
/// **Where it differs from Java, and why.** Java's sink re-pushes a fixed array of frames and takes nothing back. A
/// Rust frame is a `Box<dyn Frame>` that the lattice owns while it runs and that cannot be aliased, so the source
/// gets each frame back through a queue (`recycle`) and pushes it again. That return path, one queue slot claimed per
/// finished frame, is a harness cost with no Java counterpart (the audit lists it as row 26). Its consumer side is a
/// single consumer: the lattice serializes a source's `request` and `pull`, and a [`SourceCell`] holds the consumer and
/// the demand, so the per-frame consumer lock a multi-consumer queue would take is gone and only one claim per call
/// is left.
pub struct RepeatingSource {
    downstream: DownstreamSlot,
    inner: SourceCell<Inner>,
    home: Option<Home>,
    maker: Option<Maker>,
}

impl RepeatingSource {
    /// A source of `length` frames counting into `tally`; `seed` randomizes their routing hashes.
    pub fn new(
        id_hash: u64,
        length: usize,
        seed: u64,
        tally: &Arc<Tally>,
        latency: Option<&Arc<Recorders>>,
    ) -> Arc<Self> {
        // Many workers return frames; the one request or pull in progress takes them.
        let queue = MpscQueue::new(length.next_power_of_two());
        let home: Home = Box::leak(Box::new(queue.producer()));
        for frame in generate(id_hash, length, seed, tally, latency, home) {
            let _ = home.offer(frame);
        }
        Arc::new(Self {
            downstream: DownstreamSlot::new(),
            inner: SourceCell::new(Inner {
                frames: Some(queue.consumer()),
                demand: 0,
            }),
            home: Some(home),
            maker: None,
        })
    }

    /// A source that keeps no frames of its own: each frame it pushes comes from the pool of the thread that runs the
    /// request (or is made when that pool is empty), and goes back to the pool of the thread that finishes it. This is
    /// the nearest Rust has to Java's sink re-pushing one array: no frame travels from the worker that finished it to
    /// the worker that pushes it again. The number of frames made is bounded by the demand the lattice grants.
    pub fn pooled(
        id_hash: u64,
        seed: u64,
        tally: &Arc<Tally>,
        latency: Option<&Arc<Recorders>>,
    ) -> Arc<Self> {
        Arc::new(Self {
            downstream: DownstreamSlot::new(),
            inner: SourceCell::new(Inner {
                frames: None,
                demand: 0,
            }),
            home: None,
            maker: Some(Maker {
                id_hash,
                seed,
                made: std::sync::atomic::AtomicU64::new(0),
                tally: tally.clone(),
                latency: latency.cloned(),
            }),
        })
    }

    /// The next frame to push, if there is one.
    fn next_frame(&self, inner: &mut Inner) -> Option<Box<NoOpFrame>> {
        match (&mut inner.frames, &self.maker) {
            (Some(frames), _) => frames.poll(),
            (None, Some(maker)) => Some(pool_take().unwrap_or_else(|| maker.fresh())),
            (None, None) => None,
        }
    }

    /// Gives back a frame the source took and did not push.
    fn give_back(&self, frame: Box<NoOpFrame>) {
        match self.home {
            Some(home) => {
                let _ = home.offer(frame);
            }
            None => pool_put(frame),
        }
    }

    /// Pushes frames to the receiver while demand is banked and frames are back.
    fn push_banked(&self, inner: &mut Inner) {
        let Some(receiver) = self.downstream.receiver() else {
            return;
        };
        while !self.downstream.is_complete() {
            if inner.demand == 0 {
                return;
            }
            let Some(frame) = self.next_frame(inner) else {
                return;
            };
            if inner.demand != u64::MAX {
                inner.demand -= 1;
            }
            let mut frame = frame;
            frame.stamp();
            receiver.push(frame);
        }
    }
}

impl LatticeSource for RepeatingSource {
    fn add_downstream(self: Arc<Self>, downstream: Arc<dyn LatticeReceiver>) {
        let source: Arc<dyn LatticeSource> = self.clone();
        self.downstream.attach(source, downstream);
    }

    fn pull(
        &self,
        consumer: &mut dyn FnMut(Box<dyn Frame>),
        stop: &mut dyn FnMut(&dyn Frame) -> bool,
        demand: u64,
    ) -> u64 {
        let mut inner = self.inner.enter();
        let mut taken = 0;
        while taken < demand && !self.downstream.is_complete() {
            let Some(frame) = self.next_frame(&mut inner) else {
                break;
            };
            if stop(&*frame) {
                // Left in the source, behind the others: the frames of this source are interchangeable.
                self.give_back(frame);
                break;
            }
            let mut frame = frame;
            frame.stamp();
            consumer(frame);
            taken += 1;
        }
        taken
    }

    fn request(&self, demand: u64) {
        let mut inner = self.inner.enter();
        inner.demand = inner.demand.saturating_add(demand);
        self.push_banked(&mut inner);
    }

    fn complete(&self) {
        self.downstream.complete();
    }

    fn is_complete(&self) -> bool {
        self.downstream.is_complete()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    /// Runs `body` on a thread pinned to `cpu`, or returns `None` when this platform cannot pin.
    fn pinned<R: Send>(cpu: usize, body: impl FnOnce() -> R + Send) -> Option<R> {
        thread::scope(|scope| {
            scope
                .spawn(|| affinity::set_affinity(cpu).then(body))
                .join()
                .unwrap()
        })
    }

    fn two_cpus() -> Option<(usize, usize)> {
        let base = affinity::base_mask();
        let mut cpus = base.iter();
        Some((cpus.next()?, cpus.next()?))
    }

    #[test]
    fn a_pinned_worker_counts_into_the_cell_of_its_cpu_with_a_plain_store() {
        let Some((first, _)) = two_cpus() else { return };
        let tally = Tally::new(64, CpuSet::range(first, first + 1));
        let Some(slot) = pinned(first, || {
            for _ in 0..1000 {
                tally.increment();
            }
            tally.slot()
        }) else {
            return;
        };
        assert_eq!(
            slot,
            Slot {
                index: first,
                owned: true
            }
        );
        assert_eq!(tally.adder().get(first), 1000);
        assert_eq!(tally.adder().sum(), 1000);
    }

    #[test]
    fn two_threads_on_one_cpu_cannot_both_own_its_cell() {
        let Some((first, _)) = two_cpus() else { return };
        let tally = Tally::new(64, CpuSet::range(first, first + 1));
        let slots: Vec<Slot> = (0..2)
            .map(|_| {
                pinned(first, || {
                    tally.increment();
                    tally.slot()
                })
            })
            .collect::<Option<_>>()
            .unwrap_or_default();
        if slots.is_empty() {
            return;
        }
        assert_eq!(slots.iter().filter(|slot| slot.owned).count(), 1);
        let shared = slots.iter().find(|slot| !slot.owned).unwrap();
        assert!(
            shared.index >= 64,
            "a shared cell comes after the CPU cells"
        );
        assert_eq!(tally.adder().sum(), 2);
    }

    #[test]
    fn a_thread_outside_the_worker_set_counts_into_a_shared_cell_exactly() {
        let tally = Arc::new(Tally::new(128, CpuSet::range(100, 101)));
        thread::scope(|scope| {
            for _ in 0..4 {
                let tally = tally.clone();
                scope.spawn(move || {
                    for _ in 0..10_000 {
                        tally.increment();
                    }
                    assert!(!tally.slot().owned);
                });
            }
        });
        assert_eq!(tally.adder().sum(), 40_000);
    }

    #[test]
    fn a_thread_resolves_again_for_a_new_tally() {
        let first = Tally::new(128, CpuSet::range(100, 101));
        let second = Tally::new(128, CpuSet::range(100, 101));
        first.increment();
        second.increment();
        assert_eq!(first.adder().sum(), 1);
        assert_eq!(second.adder().sum(), 1);
    }
}
