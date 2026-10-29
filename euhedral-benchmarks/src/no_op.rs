//! The benchmark's frame and source.
//!
//! [`NoOpFrame`] does no work: finishing it bumps a striped completion counter and hands the frame back to its source,
//! so a fixed set of frames circulates, as the same frame objects are pushed again and again. [`RepeatingSource`]
//! pushes those frames to whatever receiver the lattice attaches, as many as the demand allows.

use euhedral_core::frames::{Frame, FrameHeader};
use euhedral_core::generics::{DownstreamSlot, LatticeReceiver, LatticeSource};
use euhedral_data_structures::atomics::PaddedLongAdder;
use euhedral_data_structures::queues::{Consumer, MpmcQueue, Multi, Producer};
use std::cell::Cell;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// How many frames a source circulates.
pub const FRAMES_PER_SOURCE: usize = 2_048;

/// Spreads the threads that finish frames over the counter's slots, one slot per thread until they run out.
static NEXT_STRIPE: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    static STRIPE: Cell<Option<usize>> = const { Cell::new(None) };
}

/// The completion counter slot of the calling thread.
fn stripe(slots: usize) -> usize {
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
}

impl Frame for NoOpFrame {
    fn header(&self) -> &FrameHeader {
        &self.header
    }

    fn header_mut(&mut self) -> &mut FrameHeader {
        &mut self.header
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
    ) -> Arc<Self> {
        let queue = MpmcQueue::new(length.next_power_of_two());
        let home: Home = Box::leak(Box::new(queue.producer()));
        for frame in generate(id_hash, length, seed, counters, slots, home) {
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
    fn bank(&self, demand: u64) {
        if demand == 0 {
            return;
        }
        let _ = self
            .demand
            .try_update(Ordering::AcqRel, Ordering::Acquire, |banked| {
                Some(banked.saturating_add(demand))
            });
    }

    /// Pushes frames to the receiver while demand is banked and frames are back.
    fn push_banked(&self) {
        let Some(receiver) = self.downstream.receiver() else {
            return;
        };
        while !self.downstream.is_complete() {
            let banked = self.demand.load(Ordering::Acquire);
            if banked == 0 {
                return;
            }
            let Some(frame) = self.frames.poll() else {
                return;
            };
            if banked != u64::MAX {
                self.demand.fetch_sub(1, Ordering::AcqRel);
            }
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
