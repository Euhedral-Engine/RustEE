//! Item liveness on a lane source: an item that is not alive is never run, and the source ends it
//! with `skipped`.

#![allow(
    clippy::disallowed_methods,
    clippy::disallowed_types,
    reason = "test threads stand in for workers"
)]

mod common;

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use common::{Collect, Rig, lock};
use euhedral_core::generics::LatticeSource;
use euhedral_graph::testing::{FakeLane, FakeLaneError, FakeLanes, FakeOp};
use euhedral_graph::{
    Deliver, GroupId, LaneCx, LaneId, LaneIntakes, LanePoolBuilder, LaneSource, LaneWork, Marker,
    Outcome,
};

fn marks(rig: &Rig, lane: u16) -> usize {
    rig.fake
        .log(LaneId(lane))
        .iter()
        .filter(|op| matches!(op, FakeOp::Mark(_)))
        .count()
}

fn skipped(rig: &Rig, source: usize) -> u64 {
    rig.sources[source].metrics().skipped[0].load(Ordering::SeqCst)
}

fn switch(alive: bool) -> Arc<AtomicBool> {
    Arc::new(AtomicBool::new(alive))
}

#[test]
fn a_dead_item_is_skipped_once_and_never_run() {
    let rig = Rig::new(2, &[&[0, 1]]);
    // Dead on arrival: in the lane's own intake, with a key, and in the group's shared intake.
    rig.deliver(rig.job(1).dead().frame().submit(true), rig.lane(0));
    rig.deliver(rig.job(2).dead().key(7).frame(), rig.lane(0));
    rig.deliver(rig.job(3).key(7).frame(), rig.lane(0));
    rig.deliver(rig.job(4).dead().frame(), Deliver::Group(GroupId(0)));
    rig.deliver(rig.job(5).frame(), Deliver::Group(GroupId(0)));
    rig.sources[0].request(10);
    assert_eq!(rig.probe.ran(), vec![3, 5], "the dead items never ran");
    assert_eq!(
        rig.probe.runs.load(Ordering::SeqCst),
        2,
        "not even `run` was called for them"
    );
    assert_eq!(
        *lock(&rig.probe.skipped),
        vec![1, 2, 4],
        "each was ended once, in the order the source met them"
    );
    assert_eq!(skipped(&rig, 0), 3);
    assert_eq!(
        rig.received(0),
        vec![3, 5],
        "a skipped item makes no frame, and the key's next item went on"
    );
    assert_eq!(
        marks(&rig, 0),
        0,
        "nothing was submitted for a skipped item"
    );
    assert_eq!(
        rig.probe.dropped.load(Ordering::SeqCst),
        5,
        "each item ended exactly once: skipped, then dropped by its default or by `skipped`"
    );
    // Later requests never meet them again.
    rig.sources[0].request(10);
    rig.sources[1].request(10);
    assert_eq!(*lock(&rig.probe.skipped), vec![1, 2, 4]);
    assert_eq!(skipped(&rig, 0), 3);
    assert_eq!(skipped(&rig, 1), 0, "counted on the lane that skipped");
}

#[test]
fn a_waiting_item_that_dies_leaves_without_its_wait_and_frees_its_order_key() {
    let rig = Rig::new(1, &[]);
    let alive = switch(true);
    // Item 1 waits for the gate; 2 and 4 are behind its key; 3 has another key.
    rig.deliver(rig.job(1).key(7).need(5).switch(&alive), rig.lane(0));
    rig.deliver(rig.job(2).key(7).frame(), rig.lane(0));
    rig.deliver(rig.job(3).key(8).frame(), rig.lane(0));
    rig.deliver(rig.job(4).key(7).frame(), rig.lane(0));
    rig.sources[0].request(10);
    assert_eq!(rig.probe.ran(), vec![3], "key 7 is held back by item 1");
    assert_eq!(
        rig.probe.runs.load(Ordering::SeqCst),
        2,
        "item 1 ran once and answered Wait; 3 ran"
    );
    assert!(lock(&rig.probe.skipped).is_empty());
    let asked = rig.gates[0].asked.load(Ordering::SeqCst);

    alive.store(false, Ordering::SeqCst);
    rig.sources[0].request(10);
    assert_eq!(
        *lock(&rig.probe.skipped),
        vec![1],
        "ended by skipped, the gate is still at 0"
    );
    assert_eq!(skipped(&rig, 0), 1);
    assert_eq!(
        rig.gates[0].asked.load(Ordering::SeqCst),
        asked,
        "its wait was not asked about"
    );
    assert_eq!(
        rig.probe.runs.load(Ordering::SeqCst),
        4,
        "item 1 was not run again; 2 and 4 went on in the same request"
    );
    assert_eq!(
        rig.probe.ran(),
        vec![3, 2, 4],
        "the key's items in delivery order"
    );
    assert_eq!(rig.received(0), vec![3, 2, 4]);
    assert_eq!(rig.gates[0].value.load(Ordering::SeqCst), 0);

    // Later requests never meet it again.
    rig.sources[0].request(10);
    assert_eq!(*lock(&rig.probe.skipped), vec![1]);
    assert_eq!(skipped(&rig, 0), 1);
}

#[test]
fn a_parked_item_that_dies_is_skipped_in_its_keys_order() {
    let rig = Rig::new(1, &[]);
    let alive = switch(true);
    // Item 2 is parked behind item 1's key and dies there.
    rig.deliver(rig.job(1).key(7).need(1).frame(), rig.lane(0));
    rig.deliver(rig.job(2).key(7).frame().switch(&alive), rig.lane(0));
    rig.deliver(rig.job(3).key(7).frame(), rig.lane(0));
    rig.sources[0].request(1);
    assert!(rig.probe.ran().is_empty());
    alive.store(false, Ordering::SeqCst);
    rig.sources[0].request(1);
    assert!(
        lock(&rig.probe.skipped).is_empty(),
        "behind a held-back key it stays, so the key's order holds for its end too"
    );
    rig.gates[0].value.store(1, Ordering::SeqCst);
    rig.sources[0].request(1);
    assert_eq!(rig.probe.ran(), vec![1, 3]);
    assert_eq!(*lock(&rig.probe.skipped), vec![2]);
    assert_eq!(skipped(&rig, 0), 1);
}

#[test]
fn an_armed_or_unmarked_item_that_dies_is_still_retired() {
    let rig = Rig::new(1, &[]);
    let armed = switch(true);
    let unmarked = switch(true);
    // Item 1 is marked and armed for its retirement.
    rig.deliver(rig.job(1).submit(true).switch(&armed), rig.lane(0));
    rig.sources[0].request(1);
    assert_eq!(rig.fake.armed(LaneId(0)), 1);
    // Item 2 runs and cannot be marked: the lane stalls, and item 3 stays in the intake.
    rig.fake.busy_marks(LaneId(0), 1);
    rig.deliver(rig.job(2).submit(true).switch(&unmarked), rig.lane(0));
    rig.deliver(rig.job(3).submit(true).dead(), rig.lane(0));
    rig.sources[0].request(1);
    assert_eq!(rig.probe.ran(), vec![1, 2]);
    assert!(lock(&rig.probe.skipped).is_empty(), "the lane is stalled");
    // Both die. Their device work exists, so neither is skipped.
    armed.store(false, Ordering::SeqCst);
    unmarked.store(false, Ordering::SeqCst);
    rig.sources[0].request(1);
    assert_eq!(rig.probe.ran(), vec![1, 2], "item 3 is dead: it never runs");
    assert_eq!(*lock(&rig.probe.skipped), vec![3]);
    rig.fake.retire_all();
    rig.sources[0].request(1);
    assert_eq!(
        *lock(&rig.probe.retired),
        vec![(1, None), (2, None)],
        "dead, armed items are still retired"
    );
    assert_eq!(skipped(&rig, 0), 1);
}

#[test]
fn items_without_liveness_run_as_before() {
    // An item type that never overrides `is_alive` or `skipped`.
    let mut builder = LanePoolBuilder::new();
    let first = builder.reserve(1);
    let (fake, mut lanes, mut queues) = FakeLanes::open(first, 1);
    builder.set(first, queues.remove(0));
    let pool = builder.build();
    let (intakes, mut inputs) = LaneIntakes::new(pool, &[]);
    let source = LaneSource::new(lanes.remove(0), (), inputs.remove(0), intakes.clone());
    let receiver = Collect::new();
    source.clone().add_downstream(receiver.clone());
    let ran = Arc::new(Mutex::new(Vec::new()));
    let dropped = Arc::new(AtomicUsize::new(0));
    for tag in 0..4 {
        let item = Plain {
            tag,
            key: (tag % 2 == 0).then_some(7),
            ran: ran.clone(),
            dropped: dropped.clone(),
        };
        intakes.deliver(item, Deliver::Lane(first)).unwrap();
    }
    source.request(1);
    assert_eq!(*lock(&ran), vec![0, 1, 2, 3], "every item ran, in order");
    assert_eq!(dropped.load(Ordering::SeqCst), 4);
    assert_eq!(source.metrics().skipped[0].load(Ordering::SeqCst), 0);
    assert_eq!(
        fake.log(first)
            .iter()
            .filter(|op| matches!(op, FakeOp::Mark(_)))
            .count(),
        4
    );

    // The same through the scripted items, with switches that stay alive.
    let rig = Rig::new(1, &[]);
    let alive = switch(true);
    rig.deliver(
        rig.job(1).key(7).need(1).frame().switch(&alive),
        rig.lane(0),
    );
    rig.deliver(rig.job(2).key(7).frame().switch(&alive), rig.lane(0));
    rig.deliver(rig.job(3).frame(), rig.lane(0));
    rig.sources[0].request(1);
    rig.gates[0].value.store(1, Ordering::SeqCst);
    rig.sources[0].request(1);
    assert_eq!(rig.probe.ran(), vec![3, 1, 2]);
    assert!(lock(&rig.probe.skipped).is_empty());
    assert_eq!(skipped(&rig, 0), 0);
}

/// An item that only does what `LaneWork` requires.
struct Plain {
    tag: u64,
    key: Option<u64>,
    ran: Arc<Mutex<Vec<u64>>>,
    dropped: Arc<AtomicUsize>,
}

impl Drop for Plain {
    fn drop(&mut self) {
        self.dropped.fetch_add(1, Ordering::SeqCst);
    }
}

impl LaneWork<FakeLane, ()> for Plain {
    const KINDS: &'static [&'static str] = &["plain"];

    fn kind(&self) -> usize {
        0
    }

    fn waits(&self) -> &[Marker] {
        &[]
    }

    fn order(&self) -> Option<u64> {
        self.key
    }

    fn run(
        &mut self,
        _cx: &mut LaneCx<'_, FakeLane, (), Self>,
    ) -> Outcome<std::convert::Infallible, FakeLaneError> {
        lock(&self.ran).push(self.tag);
        Outcome::Submitted { retire: false }
    }

    fn failed(self, _error: FakeLaneError, _cx: &mut LaneCx<'_, FakeLane, (), Self>) {}
}
