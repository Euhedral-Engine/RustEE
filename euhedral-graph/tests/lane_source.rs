//! Lane sources on stepped fake lanes: delivery, `pull`, completion, and every rule of `request`.

#![allow(
    clippy::disallowed_methods,
    clippy::disallowed_types,
    reason = "test threads stand in for workers; a long item holds one on a barrier"
)]

mod common;

use std::sync::atomic::Ordering;
use std::sync::{Arc, Barrier};

use common::{Job, Rig, lock};
use euhedral_core::frames::Frame;
use euhedral_core::generics::LatticeSource;
use euhedral_graph::testing::{FakeLanes, FakeOp};
use euhedral_graph::{Deliver, GroupId, LaneId, LaneIntakes, LanePoolBuilder};

fn marks(rig: &Rig, lane: u16) -> usize {
    rig.fake
        .log(LaneId(lane))
        .iter()
        .filter(|op| matches!(op, FakeOp::Mark(_)))
        .count()
}

fn awaits(rig: &Rig, lane: u16) -> usize {
    rig.fake
        .log(LaneId(lane))
        .iter()
        .filter(|op| matches!(op, FakeOp::Await(_)))
        .count()
}

// ---- intakes and the fake ----------------------------------------------------------------

#[test]
fn deliver_to_a_lane_lands_in_that_lanes_own_intake_only() {
    let rig = Rig::new(2, &[]);
    rig.deliver(rig.job(1).frame(), rig.lane(1));
    rig.sources[0].request(10);
    assert!(
        rig.probe.ran().is_empty(),
        "lane 0 does not see lane 1's work"
    );
    rig.sources[1].request(10);
    assert_eq!(rig.probe.ran(), vec![1]);
    assert_eq!(rig.received(1), vec![1]);
}

#[test]
fn deliver_to_a_group_lands_in_its_shared_intake() {
    let rig = Rig::new(3, &[&[0, 1]]);
    rig.deliver(rig.job(1), Deliver::Group(GroupId(0)));
    rig.deliver(rig.job(2), Deliver::Group(GroupId(0)));
    assert_eq!(rig.intakes.shared_len(GroupId(0)), 2);
    assert_eq!(rig.intakes.group_of(LaneId(1)), Some(GroupId(0)));
    assert_eq!(rig.intakes.group_of(LaneId(2)), None);
    rig.sources[2].request(10);
    assert!(
        rig.probe.ran().is_empty(),
        "a lane outside the group does not take it"
    );
    rig.sources[1].request(10);
    assert_eq!(rig.probe.ran(), vec![1, 2], "in delivery order");
}

#[test]
#[should_panic(expected = "two groups")]
fn a_lane_in_two_groups_is_refused() {
    let _ = Rig::new(2, &[&[0, 1], &[1]]);
}

#[test]
fn fake_lane_marks_awaits_and_retires_in_order() {
    use euhedral_graph::LaneSubmit;
    let mut builder = LanePoolBuilder::new();
    let first = builder.reserve(2);
    let (fake, mut lanes, queues) = FakeLanes::open(first, 2);
    for (i, q) in queues.into_iter().enumerate() {
        builder.set(LaneId(i as u16), q);
    }
    let pool = builder.build();
    let a = lanes[0].mark().unwrap().unwrap();
    let b = lanes[1].mark().unwrap().unwrap();
    assert_eq!((a.value, b.value), (1, 1));
    lanes[0].await_marker(b).unwrap();
    lanes[0].arm_retirement().unwrap();
    lanes[0].arm_retirement().unwrap();
    assert!(lanes[0].poll_retired().is_none(), "nothing arrived yet");
    fake.retire_all();
    let seqs: Vec<u64> = std::iter::from_fn(|| lanes[0].poll_retired())
        .map(|r| r.seq)
        .collect();
    assert_eq!(seqs, vec![0, 1]);
    assert!(!pool.reached(a));
    fake.complete_through(LaneId(0), 1);
    assert!(pool.reached(a));
    assert_eq!(
        fake.log(LaneId(0)),
        vec![
            FakeOp::Mark(a),
            FakeOp::Await(b),
            FakeOp::Arm(0),
            FakeOp::Arm(1)
        ]
    );
}

// ---- pull, completion, drop --------------------------------------------------------------

#[test]
fn pull_hands_over_ready_frames_up_to_its_demand_and_leaves_the_bank_alone() {
    let rig = Rig::new(1, &[]);
    for tag in 0..5 {
        rig.deliver(rig.job(tag).frame(), rig.lane(0));
    }
    // One frame is pushed; the other four are ready and unclaimed.
    rig.sources[0].request(1);
    assert_eq!(rig.received(0), vec![0]);
    assert_eq!(rig.sources[0].banked(), 0);
    assert_eq!(rig.sources[0].ready_len(), 4);
    let mut got = Vec::new();
    let mut stop_at_2 = |frame: &dyn Frame| frame.header().id_hash() == 2;
    let pulled = rig.sources[0].pull(
        &mut |frame| got.push(frame.header().id_hash()),
        &mut stop_at_2,
        3,
    );
    assert_eq!(
        (pulled, got.clone()),
        (1, vec![1]),
        "stops before the frame it refuses"
    );
    let pulled = rig.sources[0].pull(
        &mut |frame| got.push(frame.header().id_hash()),
        &mut |_| false,
        3,
    );
    assert_eq!((pulled, got), (3, vec![1, 2, 3, 4]));
    assert_eq!(rig.sources[0].banked(), 0, "pull never banks demand");
    assert_eq!(rig.sources[0].ready_len(), 0);
}

#[test]
fn pull_never_runs_items_or_drains_retirements() {
    let rig = Rig::new(1, &[]);
    rig.deliver(rig.job(1).submit(true), rig.lane(0));
    rig.sources[0].request(1);
    rig.fake.retire_all();
    for tag in 2..5 {
        rig.deliver(rig.job(tag).frame(), rig.lane(0));
    }
    let log = rig.fake.log(LaneId(0));
    for _ in 0..5 {
        let pulled = rig.sources[0].pull(&mut |_| {}, &mut |_| false, 10);
        assert_eq!(pulled, 0);
    }
    assert_eq!(rig.fake.log(LaneId(0)), log, "pull submitted nothing");
    assert_eq!(rig.probe.ran(), vec![1], "no queued item ran");
    assert!(
        lock(&rig.probe.retired).is_empty(),
        "no retirement was drained"
    );
}

#[test]
fn nothing_is_pulled_before_a_receiver_is_attached() {
    let mut builder = LanePoolBuilder::new();
    let first = builder.reserve(1);
    let (_fake, mut lanes, queues) = FakeLanes::open(first, 1);
    builder.set(first, queues.into_iter().next().unwrap());
    let (intakes, mut inputs) = LaneIntakes::<Job>::new(builder.build(), &[]);
    let source = euhedral_graph::LaneSource::new(
        lanes.remove(0),
        common::Gate::default(),
        inputs.remove(0),
        intakes.clone(),
    );
    let probe = common::Probe::new();
    intakes.deliver(Job::new(&probe, 1).frame(), Deliver::Lane(first));
    source.request(10);
    assert!(probe.ran().is_empty(), "no receiver: request does nothing");
    assert_eq!(source.banked(), 0);
    assert_eq!(source.pull(&mut |_| {}, &mut |_| false, 10), 0);
}

#[test]
fn complete_tells_the_receiver_once_and_zeroes_the_bank() {
    let rig = Rig::new(1, &[]);
    rig.sources[0].request(5);
    assert_eq!(rig.sources[0].banked(), 5);
    assert!(!rig.sources[0].is_complete());
    rig.sources[0].complete();
    rig.sources[0].complete();
    assert!(rig.sources[0].is_complete());
    assert_eq!(rig.sources[0].banked(), 0);
    assert_eq!(rig.receivers[0].completes.load(Ordering::SeqCst), 1);
    rig.sources[0].request(5);
    assert_eq!(
        rig.sources[0].banked(),
        0,
        "a completed source banks nothing"
    );
}

#[test]
fn dropping_the_source_ends_every_item_it_still_holds() {
    let rig = Rig::new(1, &[]);
    let probe = rig.probe.clone();
    // Item 1 is armed (its retirement is pending) and item 2 waits on the gate.
    rig.deliver(rig.job(1).submit(true), rig.lane(0));
    rig.deliver(rig.job(2).need(9), rig.lane(0));
    rig.sources[0].request(1);
    // Items 10-12 make frames; the banked demand is 2 by now, so one frame stays ready. Item 3
    // cannot be marked (it is left unmarked), so item 4 stays in the intake.
    rig.fake.busy_marks(LaneId(0), 1);
    rig.deliver(rig.job(10).frame(), rig.lane(0));
    rig.deliver(rig.job(11).frame(), rig.lane(0));
    rig.deliver(rig.job(12).frame(), rig.lane(0));
    rig.deliver(rig.job(3).submit(false), rig.lane(0));
    rig.deliver(rig.job(4), rig.lane(0));
    rig.sources[0].request(1);
    assert_eq!(rig.received(0), vec![10, 11]);
    assert_eq!(rig.sources[0].ready_len(), 1);
    assert_eq!(probe.ran(), vec![1, 10, 11, 12, 3], "item 4 never ran");
    drop(rig);
    assert_eq!(
        probe.dropped.load(Ordering::SeqCst),
        7,
        "every item was dropped"
    );
    assert_eq!(
        probe.errored.load(Ordering::SeqCst),
        1,
        "the unpushed frame was ended"
    );
}

#[test]
fn metrics_record_service_items_and_intake_latency() {
    let rig = Rig::new(1, &[]);
    for tag in 0..4 {
        rig.deliver(rig.job(tag).frame(), rig.lane(0));
    }
    rig.sources[0].request(2);
    rig.sources[0].pull(&mut |_| {}, &mut |_| false, 1);
    let m = rig.sources[0].metrics();
    assert_eq!(m.service_ns.count(), 2, "one per request and per pull");
    assert_eq!(m.item_ns[0].count(), 4);
    assert_eq!(m.intake_to_run_ns.count(), 4);
    assert_eq!(m.ready_depth.count(), 1);
    assert_eq!(m.ready_depth.snapshot().max, 2.0);
}

// ---- request -----------------------------------------------------------------------------

#[test]
fn a_source_driven_only_by_request_delivers_every_frame_within_demand() {
    let rig = Rig::new(1, &[]);
    let source = &rig.sources[0];
    let cycle = [1u64, 3, 0, 7, 2];
    let mut next_tag = 0u64;
    let mut asked = 0u64;
    let mut turn = 0;
    while rig.received(0).len() < 1000 {
        for _ in 0..37 {
            if next_tag < 1000 {
                rig.deliver(rig.job(next_tag).frame(), rig.lane(0));
                next_tag += 1;
            }
        }
        let d = cycle[turn % cycle.len()];
        turn += 1;
        if d == 0 {
            continue;
        }
        source.request(d);
        asked += d;
        let received = rig.received(0);
        assert!(
            received.len() as u64 <= asked,
            "never more than the cumulative demand"
        );
        assert!(
            received.iter().copied().eq(0..received.len() as u64),
            "in ready order"
        );
        assert!(turn < 100_000, "it finishes");
    }
    assert_eq!(rig.received(0).len(), 1000);
    assert_eq!(source.banked(), asked - 1000);
}

#[test]
fn demand_banked_before_the_work_is_ready_is_spent_when_it_becomes_ready() {
    let rig = Rig::new(1, &[]);
    rig.sources[0].request(5);
    assert!(rig.received(0).is_empty());
    for tag in 0..3 {
        rig.deliver(rig.job(tag).frame(), rig.lane(0));
    }
    rig.sources[0].request(1);
    assert_eq!(rig.received(0), vec![0, 1, 2]);
    assert_eq!(rig.sources[0].banked(), 3);
}

#[test]
fn nothing_ready_pushes_nothing() {
    let rig = Rig::new(1, &[]);
    rig.deliver(rig.job(1).need(1).frame(), rig.lane(0));
    rig.sources[0].request(10);
    assert!(rig.received(0).is_empty());
    assert_eq!(rig.sources[0].ready_len(), 0);
    assert_eq!(rig.sources[0].banked(), 10, "the demand waits");
}

#[test]
fn request_runs_every_ready_item_with_no_budget() {
    let rig = Rig::new(1, &[]);
    for tag in 0..500 {
        rig.deliver(rig.job(tag).submit(false), rig.lane(0));
    }
    rig.sources[0].request(1);
    assert_eq!(marks(&rig, 0), 500);
    assert_eq!(rig.probe.ran().len(), 500);
}

#[test]
fn delivery_is_not_a_push_and_spends_no_demand() {
    let rig = Rig::new(1, &[]);
    rig.sources[0].request(3);
    for tag in 0..10 {
        rig.deliver(rig.job(tag), rig.lane(0));
    }
    assert_eq!(rig.sources[0].banked(), 3, "delivery spends nothing");
    rig.sources[0].request(1);
    assert_eq!(
        rig.sources[0].banked(),
        4,
        "items that make no frames spend nothing"
    );
}

#[test]
fn input_markers_of_the_own_lane_are_not_awaited() {
    let rig = Rig::new(2, &[]);
    rig.deliver(
        rig.job(1).waits(&[rig.marker(0, 7)]).submit(false),
        rig.lane(0),
    );
    rig.sources[0].request(1);
    assert_eq!(rig.probe.ran(), vec![1]);
    assert_eq!(awaits(&rig, 0), 0);
}

#[test]
fn input_markers_of_another_lane_are_awaited_on_the_device() {
    let rig = Rig::new(2, &[]);
    let m = rig.marker(1, 3);
    rig.deliver(rig.job(1).waits(&[m]).submit(false), rig.lane(0));
    rig.sources[0].request(1);
    assert_eq!(
        rig.fake.log(LaneId(0))[0],
        FakeOp::Await(m),
        "ordered before the work"
    );
    assert_eq!(rig.probe.ran(), vec![1]);
}

#[test]
fn a_foreign_marker_is_handed_to_the_item_not_awaited() {
    let mut builder = LanePoolBuilder::new();
    let (a, b) = (builder.reserve(1), builder.reserve(1));
    let (fake_a, mut lanes_a, queues_a) = FakeLanes::open(a, 1);
    let (_fake_b, _lanes_b, queues_b) = FakeLanes::open(b, 1);
    builder.set(a, queues_a.into_iter().next().unwrap());
    builder.set(b, queues_b.into_iter().next().unwrap());
    let pool = builder.build();
    let foreign = euhedral_graph::Marker::new(pool.lane(b).id(), b, 4);
    let (intakes, mut inputs) = LaneIntakes::<Job>::new(pool, &[]);
    let source = euhedral_graph::LaneSource::new(
        lanes_a.remove(0),
        common::Gate::default(),
        inputs.remove(0),
        intakes.clone(),
    );
    source.clone().add_downstream(common::Collect::new());
    let probe = common::Probe::new();
    intakes.deliver(
        Job::new(&probe, 1).waits(&[foreign]).submit(false),
        Deliver::Lane(a),
    );
    source.request(1);
    assert_eq!(probe.ran(), vec![1]);
    assert_eq!(
        *lock(&probe.bridged),
        vec![(1, foreign)],
        "the item sees the marker to bridge"
    );
    assert!(
        !fake_a
            .log(a)
            .iter()
            .any(|op| matches!(op, FakeOp::Await(_))),
        "the lane did not await it"
    );
}

#[test]
fn a_waiting_item_holds_back_its_order_key_and_lets_others_through() {
    let rig = Rig::new(1, &[]);
    rig.deliver(rig.job(1).key(7).need(1), rig.lane(0));
    rig.deliver(rig.job(2).key(7), rig.lane(0));
    rig.deliver(rig.job(3).key(8), rig.lane(0));
    rig.deliver(rig.job(4), rig.lane(0));
    rig.sources[0].request(1);
    assert_eq!(
        rig.probe.ran(),
        vec![3, 4],
        "key 7 holds back; key 8 and unkeyed go"
    );
    rig.gates[0].value.store(1, Ordering::SeqCst);
    rig.sources[0].request(1);
    assert_eq!(
        rig.probe.ran(),
        vec![3, 4, 1, 2],
        "key 7 runs in delivery order"
    );
}

#[test]
fn a_waiting_item_is_released_when_its_state_is_met() {
    let rig = Rig::new(1, &[]);
    rig.deliver(rig.job(1).need(5).frame(), rig.lane(0));
    rig.sources[0].request(1);
    assert!(rig.received(0).is_empty());
    rig.gates[0].value.store(4, Ordering::SeqCst);
    rig.sources[0].request(1);
    assert!(rig.received(0).is_empty(), "not yet");
    rig.gates[0].value.store(5, Ordering::SeqCst);
    rig.sources[0].request(1);
    assert_eq!(rig.received(0), vec![1], "released at the next request");
    assert_eq!(
        rig.gates[0].checks.load(Ordering::SeqCst),
        3,
        "check once per request"
    );
}

#[test]
fn a_retirement_runs_the_items_continuation_in_arm_order() {
    let rig = Rig::new(1, &[]);
    for tag in 0..3 {
        rig.deliver(rig.job(tag).submit(true), rig.lane(0));
    }
    rig.sources[0].request(1);
    assert!(lock(&rig.probe.retired).is_empty());
    rig.fake.retire_all();
    rig.sources[0].request(1);
    let tags: Vec<u64> = lock(&rig.probe.retired).iter().map(|(t, _)| *t).collect();
    assert_eq!(tags, vec![0, 1, 2]);
}

#[test]
fn a_successor_delivered_by_a_continuation_runs_in_the_same_request() {
    let rig = Rig::new(1, &[]);
    rig.deliver(
        rig.job(1)
            .submit(true)
            .retired_deliver(rig.job(2).frame(), rig.lane(0)),
        rig.lane(0),
    );
    rig.sources[0].request(1);
    rig.fake.retire_all();
    rig.sources[0].request(1);
    assert_eq!(
        rig.received(0),
        vec![2],
        "retired, delivered, run and pushed in one request"
    );
}

#[test]
fn a_busy_mark_stalls_the_lane_until_the_ring_frees() {
    let rig = Rig::new(1, &[]);
    rig.fake.busy_marks(LaneId(0), 2);
    for tag in 1..=3 {
        rig.deliver(rig.job(tag).submit(false), rig.lane(0));
    }
    rig.sources[0].request(1);
    assert_eq!(rig.probe.ran(), vec![1]);
    assert_eq!(marks(&rig, 0), 0);
    rig.sources[0].request(1);
    assert_eq!(rig.probe.ran(), vec![1], "still busy: nothing new runs");
    assert_eq!(marks(&rig, 0), 0);
    rig.sources[0].request(1);
    assert_eq!(rig.probe.ran(), vec![1, 2, 3]);
    let values: Vec<u64> = rig
        .fake
        .log(LaneId(0))
        .iter()
        .filter_map(|op| match op {
            FakeOp::Mark(m) => Some(m.value),
            _ => None,
        })
        .collect();
    assert_eq!(
        values,
        vec![1, 2, 3],
        "item 1 is marked first, then 2 and 3 in order"
    );
}

#[test]
fn a_failed_retirement_reaches_the_item_as_a_failure() {
    let rig = Rig::new(1, &[]);
    rig.deliver(rig.job(1).submit(true), rig.lane(0));
    rig.deliver(rig.job(2).submit(true), rig.lane(0));
    rig.sources[0].request(1);
    rig.fake.fail_retirements(LaneId(0));
    rig.fake.retire_all();
    rig.sources[0].request(1);
    let retired = lock(&rig.probe.retired).clone();
    assert_eq!(retired.len(), 2);
    assert!(retired.iter().all(|(_, failure)| failure.is_some()));
}

#[test]
fn a_failed_await_ends_the_item_with_failed() {
    let rig = Rig::new(2, &[]);
    rig.fake.fail_awaits(LaneId(0), 1);
    rig.deliver(
        rig.job(1).waits(&[rig.marker(1, 1)]).submit(false),
        rig.lane(0),
    );
    rig.deliver(rig.job(2).submit(false), rig.lane(0));
    rig.sources[0].request(1);
    assert_eq!(*lock(&rig.probe.failed), vec![1]);
    assert_eq!(
        rig.probe.ran(),
        vec![2],
        "the failed item never ran; the next did"
    );
}

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "nested entry")]
fn an_item_that_requests_its_own_source_panics_in_debug() {
    let rig = Rig::new(1, &[]);
    let mut job = rig.job(1);
    job.reenter = Some(rig.sources[0].clone());
    rig.deliver(job, rig.lane(0));
    rig.sources[0].request(1);
}

// ---- groups --------------------------------------------------------------------------------

#[test]
fn group_work_runs_on_the_first_source_entered() {
    let rig = Rig::new(3, &[&[0, 1, 2]]);
    for tag in 0..10 {
        rig.deliver(rig.job(tag).submit(false), Deliver::Group(GroupId(0)));
    }
    rig.sources[1].request(1);
    assert_eq!(marks(&rig, 1), 10);
    assert_eq!(marks(&rig, 0), 0);
    assert_eq!(marks(&rig, 2), 0);
}

#[test]
#[allow(
    clippy::disallowed_methods,
    reason = "test threads stand in for workers"
)]
fn a_source_inside_a_long_item_leaves_group_work_to_the_others() {
    let rig = Rig::new(2, &[&[0, 1]]);
    let (a, b) = (Arc::new(Barrier::new(2)), Arc::new(Barrier::new(2)));
    rig.deliver(
        rig.job(0).submit(false).hold(&a, &b),
        Deliver::Group(GroupId(0)),
    );
    for tag in 1..10 {
        rig.deliver(rig.job(tag).submit(false), Deliver::Group(GroupId(0)));
    }
    let (first, second) = (rig.sources[0].clone(), rig.sources[1].clone());
    std::thread::scope(|scope| {
        let long = scope.spawn(move || first.request(1));
        a.wait();
        // Source 0 is inside the long item: source 1 takes the rest of the group's work.
        second.request(1);
        assert_eq!(
            marks(&rig, 1),
            9,
            "the others ran while one source was inside an item"
        );
        assert_eq!(marks(&rig, 0), 0, "the long item is not marked yet");
        b.wait();
        long.join().unwrap();
    });
    assert_eq!(marks(&rig, 0), 1);
}

#[test]
#[allow(
    clippy::disallowed_methods,
    reason = "test threads stand in for workers"
)]
fn the_shared_intake_is_free_while_an_item_runs() {
    let rig = Rig::new(1, &[&[0]]);
    let (a, b) = (Arc::new(Barrier::new(2)), Arc::new(Barrier::new(2)));
    rig.deliver(rig.job(0).hold(&a, &b), Deliver::Group(GroupId(0)));
    rig.deliver(rig.job(1), Deliver::Group(GroupId(0)));
    let source = rig.sources[0].clone();
    std::thread::scope(|scope| {
        let long = scope.spawn(move || source.request(1));
        a.wait();
        // The source holds the item it took; the claim on the intake is not held across it.
        let taken = rig.intakes.probe_shared(GroupId(0));
        assert_eq!(taken.as_ref().map(|j| j.tag), Some(1));
        b.wait();
        long.join().unwrap();
    });
}
