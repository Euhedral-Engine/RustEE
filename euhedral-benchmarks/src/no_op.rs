//! The benchmark's frame and source.
//!
//! [`NoOpFrame`] does no work: finishing it bumps a striped completion counter and hands the frame back to its source,
//! so a fixed set of frames circulates, as the same frame objects are pushed again and again. [`RepeatingSource`]
//! pushes those frames to whatever receiver the lattice attaches, as many as the demand allows.

use crate::latency::Recorders;
use euhedral_core::frames::{Frame, FrameHeader};
use euhedral_core::generics::{DownstreamSlot, LatticeReceiver, LatticeSource};
use euhedral_data_structures::atomics::PaddedLongAdder;
use euhedral_data_structures::queues::{Consumer, MpmcQueue, Multi, Producer};
use std::cell::Cell;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

/// One frame in this many is stamped and timed.
pub const SAMPLE_EVERY: u64 = 64;

/// How many frames a source circulates.
pub const FRAMES_PER_SOURCE: usize = 2_048;

/// Spreads the threads that finish frames over the counter's slots, one slot per thread until they run out.
static NEXT_STRIPE: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    static STRIPE: Cell<Option<usize>> = const { Cell::new(None) };
}

/// The completion counter slot of the calling thread.
pub fn stripe(slots: usize) -> usize {
    STRIPE.with(|stripe| {
        let chosen = stripe.get().unwrap_or_else(|| {
            let next = NEXT_STRIPE.fetch_add(1, Ordering::Relaxed);
            stripe.set(Some(next));
            next
        });
        chosen % slots
    })
}

/// Where a finished frame goes back to: its source's queue. Leaked once per source, so that finishing a frame touches
/// no reference count that every worker shares; a benchmark process builds a handful.
type Home = &'static Producer<Box<NoOpFrame>, Multi>;

/// A frame that does nothing, and counts itself when it finishes (`NoOpFrame` upstream).
pub struct NoOpFrame {
    header: FrameHeader,
    counters: Arc<PaddedLongAdder>,
    /// The counter's slot count.
    slots: usize,
    home: Home,
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
            recorders.slot(stripe(self.slots)).record(nanos);
        }
        Ok(())
    }

    /// Counts the completion, then returns the frame to its source.
    fn do_finally(self: Box<Self>) {
        self.counters.increment_at(stripe(self.slots));
        self.recycle();
    }

    fn recycle(self: Box<Self>) {
        let home = self.home;
        // The queue is unbounded: a frame always fits.
        let _ = home.offer(self);
    }
}

/// `length` unordered frames of `id_hash`, each with a routing hash of its own from `seed`
/// (`NoOpFrame.generate` upstream).
pub fn generate(
    id_hash: u64,
    length: usize,
    seed: u64,
    counters: &Arc<PaddedLongAdder>,
    slots: usize,
    latency: Option<&Arc<Recorders>>,
    home: Home,
) -> Vec<Box<NoOpFrame>> {
    (0..length as u64)
        .map(|index| {
            let mut header = FrameHeader::new(id_hash);
            header.randomize_hash(seed.wrapping_add(index));
            Box::new(NoOpFrame {
                header,
                counters: counters.clone(),
                slots,
                home,
                // One frame in `SAMPLE_EVERY` is sampled, every time it circulates.
                latency: latency.filter(|_| index % SAMPLE_EVERY == 0).cloned(),
                stamp: None,
            })
        })
        .collect()
}

/// A source that pushes a fixed set of [`NoOpFrame`]s over and over (`RepeatingSink` upstream): a request or pull takes
/// the next frames that are back from the lattice, as many as the demand allows. The demand a request leaves unmet
/// stays banked for the next one.
pub struct RepeatingSource {
    downstream: DownstreamSlot,
    demand: AtomicU64,
    frames: Consumer<Box<NoOpFrame>, Multi>,
    home: Home,
}

impl RepeatingSource {
    /// A source of `length` frames counting into `counters`, which has `slots` slots; `seed` randomizes their routing
    /// hashes.
    pub fn new(
        id_hash: u64,
        length: usize,
        seed: u64,
        counters: &Arc<PaddedLongAdder>,
        slots: usize,
        latency: Option<&Arc<Recorders>>,
    ) -> Arc<Self> {
        let queue = MpmcQueue::new(length.next_power_of_two());
        let home: Home = Box::leak(Box::new(queue.producer()));
        for frame in generate(id_hash, length, seed, counters, slots, latency, home) {
            let _ = home.offer(frame);
        }
        Arc::new(Self {
            downstream: DownstreamSlot::new(),
            demand: AtomicU64::new(0),
            frames: queue.consumer(),
            home,
        })
    }

    /// Banks `demand`; `u64::MAX` is unbounded.
    ///
    /// The lattice serializes a source's `request` and `pull`: one worker is inside either at a time, and the
    /// interceptor's claim orders one worker's exit before the next one's entry. The banked demand is therefore plain
    /// state of whoever is inside, kept in an atomic only so that the source is `Sync`: it changes by a load and a
    /// store, never a read-modify-write.
    fn bank(&self, demand: u64) {
        if demand == 0 {
            return;
        }
        let banked = self.demand.load(Ordering::Relaxed);
        self.demand
            .store(banked.saturating_add(demand), Ordering::Relaxed);
    }

    /// Pushes frames to the receiver while demand is banked and frames are back.
    fn push_banked(&self) {
        let Some(receiver) = self.downstream.receiver() else {
            return;
        };
        while !self.downstream.is_complete() {
            let banked = self.demand.load(Ordering::Relaxed);
            if banked == 0 {
                return;
            }
            let Some(frame) = self.frames.poll() else {
                return;
            };
            if banked != u64::MAX {
                self.demand.store(banked - 1, Ordering::Relaxed);
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
        let mut taken = 0;
        while taken < demand && !self.downstream.is_complete() {
            let Some(frame) = self.frames.poll() else {
                break;
            };
            if stop(&*frame) {
                // Left in the source, behind the others: the frames of this source are interchangeable.
                let _ = self.home.offer(frame);
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
        self.bank(demand);
        self.push_banked();
    }

    fn complete(&self) {
        self.downstream.complete();
    }

    fn is_complete(&self) -> bool {
        self.downstream.is_complete()
    }
}
