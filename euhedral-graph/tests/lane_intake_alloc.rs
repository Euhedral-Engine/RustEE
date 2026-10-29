//! Delivery and `request` allocate nothing in steady state. This binary counts every allocation of
//! the process, so it holds one test and runs alone.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use euhedral_core::frames::Frame;
use euhedral_core::generics::{LatticeReceiver, LatticeSource, LatticeTerminal};
use euhedral_graph::testing::{FakeLane, FakeLaneError, FakeLanes};
use euhedral_graph::{
    Deliver, GroupId, LaneCx, LaneId, LaneIntakes, LanePoolBuilder, LaneSource, LaneWork, Marker,
    Outcome,
};

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

struct Counting;

// SAFETY: every call is forwarded unchanged to the system allocator; the counter is an atomic.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: the caller's contract, forwarded.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: the caller's contract, forwarded.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: the caller's contract, forwarded.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static COUNTING: Counting = Counting;

/// An item that makes no frame.
struct Item {
    retire: bool,
    retired: Arc<AtomicUsize>,
}

type Cx<'a> = LaneCx<'a, FakeLane, (), Item>;

impl LaneWork<FakeLane, ()> for Item {
    const KINDS: &'static [&'static str] = &["item"];

    fn kind(&self) -> usize {
        0
    }

    fn waits(&self) -> &[Marker] {
        &[]
    }

    fn run(&mut self, _cx: &mut Cx<'_>) -> Outcome<std::convert::Infallible> {
        Outcome::Submitted {
            retire: self.retire,
        }
    }

    fn retired(self, _marker: Marker, _failure: Option<FakeLaneError>, _cx: &mut Cx<'_>) {
        self.retired.fetch_add(1, Ordering::Relaxed);
    }

    fn failed(self, _error: FakeLaneError, _cx: &mut Cx<'_>) {
        unreachable!("the fake lane does not fail");
    }
}

struct Discard;

impl LatticeTerminal for Discard {
    fn add_upstream(&self, _upstream: Arc<dyn LatticeSource>) {}
}

impl LatticeReceiver for Discard {
    fn push(&self, _frame: Box<dyn Frame>) {}
    fn on_complete(&self) {}
    fn on_error(&self, _error: Box<dyn std::error::Error + Send + Sync>) {}
}

const BATCH: usize = 64;

#[test]
fn steady_state_delivery_and_requests_allocate_nothing() {
    // The counter sees allocations at all.
    let probe = ALLOCATIONS.load(Ordering::Relaxed);
    drop(std::hint::black_box(Box::new(7u64)));
    assert!(
        ALLOCATIONS.load(Ordering::Relaxed) > probe,
        "the counting allocator is live"
    );

    let mut builder = LanePoolBuilder::new();
    let first = builder.reserve(2);
    let (fake, submitters, queues) = FakeLanes::open(first, 2);
    let fake = fake.quiet();
    for (i, queue) in queues.into_iter().enumerate() {
        builder.set(LaneId(i as u16), queue);
    }
    let (intakes, inputs) = LaneIntakes::<Item>::new(builder.build(), &[&[LaneId(0), LaneId(1)]]);
    let sources: Vec<_> = submitters
        .into_iter()
        .zip(inputs)
        .map(|(submitter, input)| {
            let source = LaneSource::new(submitter, (), input, intakes.clone());
            source.clone().add_downstream(Arc::new(Discard));
            source
        })
        .collect();
    let retired = Arc::new(AtomicUsize::new(0));

    let round = |count: usize| {
        for i in 0..count {
            let item = Item {
                retire: i % 2 == 0,
                retired: retired.clone(),
            };
            let to = if i % 4 < 2 {
                Deliver::Group(GroupId(0))
            } else {
                Deliver::Lane(LaneId(0))
            };
            intakes.deliver(item, to);
        }
        sources[0].request(1);
        sources[1].request(1);
        fake.retire_all();
        sources[0].request(1);
        sources[1].request(1);
    };

    for _ in 0..2000 / BATCH {
        round(BATCH);
    }
    let before = ALLOCATIONS.load(Ordering::Relaxed);
    let retired_before = retired.load(Ordering::Relaxed);
    for _ in 0..20_000 / BATCH {
        round(BATCH);
    }
    let allocated = ALLOCATIONS.load(Ordering::Relaxed) - before;
    assert!(
        retired.load(Ordering::Relaxed) - retired_before >= 9_000,
        "the retirements ran"
    );
    assert_eq!(allocated, 0, "no allocation in steady state");
}
