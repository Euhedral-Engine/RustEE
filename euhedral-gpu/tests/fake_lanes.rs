//! Lane sources on streams of the fake backend: chains across lanes, retirement, a foreign lane in the pool, a full
//! marker ring and the priority. No GPU, no wall clock.
#![cfg(feature = "fake")]

mod common;

use std::marker::PhantomData;
use std::sync::Arc;

use common::{HostWord, Item, Plan, Probe, collect_all, lock, pump, pump_from};
use euhedral_core::generics::LatticeSource as _;
use euhedral_gpu::fake::{FAKE_GREATEST_PRIORITY, FakeGpu, FakeModule, FakeOp, FakeStream};
use euhedral_gpu::{
    Args, CompileOptions, Enqueue, Gpu, GpuError, LaneSpec, Module, Stream, StreamOptions,
    StreamPriority, open_lane_sources,
};
use euhedral_graph::testing::FakeLanes;
use euhedral_graph::{
    Deliver, GroupId, LaneId, LaneIntakes, LanePoolBuilder, LaneSource, LaneSubmit,
};

type Item1 = Item<FakeStream>;

fn gpu() -> Arc<FakeGpu> {
    let gpu = FakeGpu::new();
    // word = word * 31 + c: an order-sensitive fold.
    gpu.define("fold", |mem, _launch, args| {
        let at = args.read::<u64>(0).unwrap();
        let c = args.read::<u32>(1).unwrap();
        let word = mem.bytes_mut(at, 4)?;
        let v = u32::from_le_bytes(word.try_into().unwrap())
            .wrapping_mul(31)
            .wrapping_add(c);
        word.copy_from_slice(&v.to_le_bytes());
        Ok(())
    });
    gpu
}

fn fold_plan(
    gpu: &Arc<FakeGpu>,
    steps: u32,
    route: impl Fn(u32) -> Deliver + Send + Sync + 'static,
) -> (Arc<Plan<FakeStream>>, u32) {
    let module =
        FakeModule::from_source(gpu, "// fake source", &CompileOptions::default()).unwrap();
    let kernel = module.kernel("fold").unwrap();
    let word = gpu.alloc(4);
    let expected = (0..steps).fold(0u32, |v, i| v.wrapping_mul(31).wrapping_add(i + 1));
    let plan = Arc::new(Plan {
        steps,
        route: Box::new(route),
        launches: Box::new(move |step| {
            vec![(kernel.clone(), Args::new().device(word).arg(step + 1))]
        }),
        device_word: word,
        host_word: HostWord::leaked(),
        probe: Probe::new(),
        retire_steps: false,
        _stream: PhantomData,
    });
    (plan, expected)
}

fn specs(names: &[&str]) -> Vec<LaneSpec<()>> {
    names
        .iter()
        .map(|name| LaneSpec {
            name: (*name).to_owned(),
            options: StreamOptions::default(),
            state: (),
        })
        .collect()
}

#[test]
fn a_chain_of_kernel_items_across_lanes_runs_in_order() {
    let gpu = gpu();
    // Lanes a, b, c form a group; x is on its own.
    let opened = open_lane_sources::<_, (), Item1>(
        &gpu,
        specs(&["a", "b", "c", "x"]),
        &[&[LaneId(0), LaneId(1), LaneId(2)]],
    )
    .unwrap();
    let steps = 40;
    let (plan, expected) = fold_plan(&gpu, steps, |step| {
        if step % 2 == 0 {
            Deliver::Group(GroupId(0))
        } else {
            Deliver::Lane(LaneId(3))
        }
    });
    let collect = collect_all(&opened.sources);
    opened
        .intakes
        .deliver(Item1::first(&plan), Deliver::Group(GroupId(0)))
        .unwrap();
    // The device does not run until the whole chain is submitted, so no marker has completed when its successor orders
    // itself behind it (an await on a completed marker is elided, as the driver's would be).
    for round in 0..1000 {
        // The start rotates, so each of the group's lanes takes group work in turn.
        pump_from(&opened.sources, round % 4);
        if plan.probe.ran().len() == steps as usize + 1 {
            break;
        }
        assert!(round < 999, "the chain did not finish");
    }
    gpu.run_until_idle();
    pump(&opened.sources);
    assert_eq!(collect.received(), vec![u64::from(expected)]);
    assert_eq!(plan.host_word.load(), expected);
    assert!(lock(&plan.probe.failed).is_empty());
    assert!(
        lock(&plan.probe.retired)
            .iter()
            .all(|(_, _, f)| f.is_none())
    );
    // Odd steps ran on lane x (3); even steps on the group's lanes.
    let log = gpu.log();
    let launches: Vec<(usize, u16)> = log
        .iter()
        .enumerate()
        .filter_map(|(at, op)| match op {
            FakeOp::Launch { lane, .. } => Some((at, *lane)),
            _ => None,
        })
        .collect();
    assert_eq!(launches.len(), steps as usize);
    for (i, (_, lane)) in launches.iter().enumerate() {
        assert_eq!(*lane == 3, i % 2 == 1, "step {i} ran on lane {lane}");
    }
    // Every step but the first ran behind an await of the previous step's marker on its own lane (each is on another
    // lane than its predecessor): a lane that skipped the await would show a launch with no `Await` before it.
    for (i, &(at, lane)) in launches.iter().enumerate().skip(1) {
        let before = launches[i - 1].1;
        let predecessor_mark = log[..at]
            .iter()
            .rev()
            .find_map(|op| match op {
                FakeOp::Mark { lane: l, value } if *l == before => Some(*value),
                _ => None,
            })
            .expect("the predecessor was marked");
        assert_eq!(
            log[at - 1],
            FakeOp::Await {
                lane,
                value: predecessor_mark
            },
            "step {i} is not ordered behind its predecessor's marker"
        );
    }
    // The group's lanes all took work.
    let group_lanes: std::collections::BTreeSet<u16> = launches
        .iter()
        .filter(|(_, lane)| *lane != 3)
        .map(|(_, lane)| *lane)
        .collect();
    assert!(group_lanes.len() > 1, "{group_lanes:?}");
}

#[test]
fn a_retirement_continuation_reads_back_the_result() {
    let gpu = gpu();
    let opened = open_lane_sources::<_, (), Item1>(&gpu, specs(&["a"]), &[]).unwrap();
    let (plan, _) = fold_plan(&gpu, 3, |_| Deliver::Lane(LaneId(0)));
    gpu.write(plan.device_word, &7u32.to_le_bytes());
    let expected = (0..3u32).fold(7u32, |v, i| v.wrapping_mul(31).wrapping_add(i + 1));
    let collect = collect_all(&opened.sources);
    opened
        .intakes
        .deliver(Item1::first(&plan), Deliver::Lane(LaneId(0)))
        .unwrap();
    // The chain runs on the one lane; the copy is enqueued last and asks for a retirement.
    opened.sources[0].request(1);
    assert!(collect.received().is_empty(), "nothing retired yet");
    gpu.run_until_idle();
    opened.sources[0].request(1);
    assert_eq!(collect.received(), vec![u64::from(expected)]);
    assert_eq!(lock(&plan.probe.retired).len(), 1);
}

#[test]
fn a_marker_of_a_foreign_lane_in_the_pool_is_bridged() {
    let gpu = gpu();
    // One pool, two backends: the device's lane 0 and a foreign lane 1.
    let mut builder = LanePoolBuilder::new();
    let first = builder.reserve(1);
    let foreign_first = builder.reserve(1);
    let mut streams = gpu
        .open_lanes_at(first, &["gpu"], &[StreamOptions::default()])
        .unwrap();
    let stream = streams.remove(0);
    builder.set(first, Stream::queue(&stream));
    let (_fake, mut foreign, queues) = FakeLanes::open(foreign_first, 1);
    builder.set(foreign_first, queues.into_iter().next().unwrap());
    let pool = builder.build();
    let marker = LaneSubmit::mark(&mut foreign[0]).unwrap().unwrap();
    assert!(
        !LaneSubmit::accepts(&stream, marker),
        "a lane of another backend is not this call's"
    );
    let (intakes, mut inputs) = LaneIntakes::<Item1>::new(pool, &[]);
    let source = LaneSource::new(stream, (), inputs.remove(0), intakes.clone());
    let sources = [source];
    let collect = collect_all(&sources);
    let (plan, expected) = fold_plan(&gpu, 0, move |_| Deliver::Lane(first));
    let mut item = Item1::first(&plan);
    item.waits = vec![marker];
    intakes.deliver(item, Deliver::Lane(first)).unwrap();
    sources[0].request(1);
    gpu.run_until_idle();
    sources[0].request(1);
    // The item saw the marker as one to bridge, the lane did not wait for it, and the item ran.
    assert_eq!(*lock(&plan.probe.bridged), vec![(0, marker)]);
    assert!(lock(&plan.probe.failed).is_empty(), "no await was refused");
    assert_eq!(collect.received(), vec![u64::from(expected)]);
}

#[test]
fn a_busy_marker_ring_stalls_then_resumes() {
    let gpu = gpu();
    let lanes = vec![LaneSpec {
        name: "a".to_owned(),
        options: StreamOptions {
            markers: 2,
            ..StreamOptions::default()
        },
        state: (),
    }];
    let opened = open_lane_sources::<_, (), Item1>(&gpu, lanes, &[]).unwrap();
    let _collect = collect_all(&opened.sources);
    let (plan, _) = fold_plan(&gpu, 0, |_| Deliver::Lane(LaneId(0)));
    // Four independent items, each a launch and a marker, before the device completes anything.
    let module =
        FakeModule::from_source(&gpu, "// fake source", &CompileOptions::default()).unwrap();
    let kernel = module.kernel("fold").unwrap();
    let word = plan.device_word;
    let chain = Arc::new(Plan::<FakeStream> {
        steps: 100,
        route: Box::new(|_| Deliver::Lane(LaneId(0))),
        launches: Box::new(move |step| {
            vec![(kernel.clone(), Args::new().device(word).arg(step + 1))]
        }),
        device_word: word,
        host_word: HostWord::leaked(),
        probe: Probe::new(),
        retire_steps: false,
        _stream: PhantomData,
    });
    for tag in 1..=4 {
        let mut item = Item1::first(&chain);
        item.tag = tag;
        opened
            .intakes
            .deliver(item, Deliver::Lane(LaneId(0)))
            .unwrap();
    }
    // `then` of each item delivers its successor; only the first four steps matter, so look at the first pass.
    opened.sources[0].request(1);
    let tags: Vec<u64> = chain.probe.ran().iter().map(|(tag, _)| *tag).collect();
    assert_eq!(
        tags,
        vec![1, 2, 3],
        "the third could not be marked: the lane stalled"
    );
    let launches = |gpu: &Arc<FakeGpu>| {
        gpu.log()
            .iter()
            .filter(|op| matches!(op, FakeOp::Launch { .. }))
            .count()
    };
    gpu.run_until_idle();
    assert_eq!(launches(&gpu), 3, "the fourth item did not run");
    // The device completed the markers; the lane marks the third item first, then goes on in order.
    for _ in 0..50 {
        gpu.run_until_idle();
        opened.sources[0].request(1);
    }
    let tags: Vec<u64> = chain.probe.ran().iter().map(|(tag, _)| *tag).collect();
    assert!(
        tags.len() > 4,
        "the lane went on after the ring freed: {tags:?}"
    );
    assert_eq!(&tags[..4], &[1, 2, 3, 4], "in delivery order, none lost");
}

#[test]
fn the_high_priority_lane_is_opened_with_high_priority() {
    let gpu = gpu();
    let lanes = vec![
        LaneSpec {
            name: "kv".to_owned(),
            options: StreamOptions {
                priority: StreamPriority::High,
                ..StreamOptions::default()
            },
            state: (),
        },
        LaneSpec {
            name: "k0".to_owned(),
            options: StreamOptions::default(),
            state: (),
        },
    ];
    let opened = open_lane_sources::<_, (), Item1>(&gpu, lanes, &[]).unwrap();
    let priority = |lane: u16| gpu.lane_priority(opened.pool.lane(LaneId(lane)).id());
    assert_eq!(priority(0), Some(FAKE_GREATEST_PRIORITY));
    assert_eq!(priority(1), Some(0));
}

#[test]
fn bad_group_lists_are_refused_not_panicked() {
    let gpu = gpu();
    let two = |g: &[&[LaneId]]| open_lane_sources::<_, (), Item1>(&gpu, specs(&["a", "b"]), g);
    assert!(two(&[&[LaneId(0), LaneId(1)]]).is_ok());
    assert!(two(&[&[LaneId(0)], &[LaneId(0)]]).is_err(), "two groups");
    assert!(two(&[&[LaneId(2)]]).is_err(), "not a lane");
    assert!(open_lane_sources::<_, (), Item1>(&gpu, specs(&[]), &[]).is_err());
}

/// Lane a faults on the device; lane b's item waits for lane a's marker and asks for a retirement. With `retire_steps`
/// the faulting item asks for one as well.
fn fault_run(retire_steps: bool) {
    let gpu = gpu();
    let opened = open_lane_sources::<_, (), Item1>(&gpu, specs(&["a", "b"]), &[]).unwrap();
    let _collect = collect_all(&opened.sources);
    let module =
        FakeModule::from_source(&gpu, "// fake source", &CompileOptions::default()).unwrap();
    gpu.define("illegal", |_, _, _| {
        Err(euhedral_gpu::fake::FakeFault::IllegalAddress)
    });
    let kernel = module.kernel("illegal").unwrap();
    let plan = Arc::new(Plan::<FakeStream> {
        steps: 1,
        route: Box::new(|_| Deliver::Lane(LaneId(1))),
        launches: Box::new(move |_| vec![(kernel.clone(), Args::new())]),
        device_word: gpu.alloc(256),
        host_word: HostWord::leaked(),
        probe: Probe::new(),
        retire_steps,
        _stream: PhantomData,
    });
    opened
        .intakes
        .deliver(Item1::first(&plan), Deliver::Lane(LaneId(0)))
        .unwrap();
    // Step 0 faults on lane a; step 1 on lane b waits for lane a's marker, which never completes, so its callback
    // never fires. The lane's look at the device is what finds the fault and reports the retirement.
    for _ in 0..10 {
        pump(&opened.sources);
        gpu.run_until_idle();
    }
    let retired = lock(&plan.probe.retired).clone();
    let step = |n: u32| retired.iter().find(|(_, step, _)| *step == n);
    let Some((_, _, Some(read))) = step(1) else {
        panic!("step 1's retirement was stranded or carried no failure: {retired:?}");
    };
    assert!(
        matches!(read, GpuError::Poisoned(_) | GpuError::Fault { .. }),
        "{read:?}"
    );
    if retire_steps {
        let Some((_, _, Some(first))) = step(0) else {
            panic!("step 0's retirement is missing: {retired:?}");
        };
        assert!(
            matches!(first, GpuError::Fault { code: 700, .. }),
            "{first:?}"
        );
    } else {
        assert!(step(0).is_none());
    }
    assert!(
        lock(&plan.probe.failed).is_empty(),
        "nothing was refused yet"
    );
    // Later work fails fast: an item's submission on the poisoned device ends it with `failed`.
    let mut late = Item1::first(&plan);
    late.tag = 9;
    opened
        .intakes
        .deliver(late, Deliver::Lane(LaneId(0)))
        .unwrap();
    pump(&opened.sources);
    let failed = lock(&plan.probe.failed).clone();
    assert_eq!(failed.len(), 1, "{failed:?}");
    assert_eq!((failed[0].0, failed[0].1), (9, 0));
    assert!(
        matches!(failed[0].2, GpuError::Poisoned(_)),
        "{:?}",
        failed[0].2
    );
    assert!(
        gpu.log()
            .iter()
            .filter(|op| matches!(op, FakeOp::Launch { .. }))
            .count()
            <= 1,
        "nothing ran behind the fault"
    );
}

#[test]
fn a_retirement_behind_a_faulted_lane_is_reported_not_stranded() {
    fault_run(false);
}

#[test]
fn every_retirement_of_a_fault_run_reaches_its_item() {
    fault_run(true);
}

#[test]
fn the_stream_as_a_lane_submitter_numbers_arms_and_settles_arrivals() {
    let gpu = gpu();
    let mut streams = gpu
        .open_lanes_at(LaneId(2), &["a", "b"], &[StreamOptions::default(); 2])
        .unwrap();
    let mut b = streams.pop().unwrap();
    let mut a = streams.pop().unwrap();
    let (seq0, seq1) = (
        LaneSubmit::arm_retirement(&mut a).unwrap(),
        LaneSubmit::arm_retirement(&mut a).unwrap(),
    );
    assert_eq!((seq0, seq1), (0, 1));
    assert!(
        LaneSubmit::poll_retired(&mut a).is_none(),
        "nothing arrived"
    );
    gpu.run_until_idle();
    let seqs: Vec<u64> = std::iter::from_fn(|| LaneSubmit::poll_retired(&mut a))
        .map(|r| {
            assert!(r.failure.is_none());
            r.seq
        })
        .collect();
    assert_eq!(seqs, vec![0, 1]);
    // A lane of the same call is accepted, a lane outside it is not, and a full ring is `Ok(None)`.
    let mb = LaneSubmit::mark(&mut b).unwrap().unwrap();
    assert!(LaneSubmit::accepts(&a, mb));
    assert!(!LaneSubmit::accepts(
        &a,
        euhedral_graph::Marker::new(Stream::id(&a), LaneId(9), 1)
    ));
    assert!(!LaneSubmit::accepts(
        &a,
        euhedral_graph::Marker::new(euhedral_graph::QueueId::next(), LaneId(3), 1)
    ));
    // SAFETY: the fake's handle is the queue id; nothing is enqueued through it.
    assert_eq!(unsafe { LaneSubmit::raw(&mut a) }, Stream::id(&a).get());
}

#[test]
fn a_retirement_that_arrives_ok_is_confirmed_with_a_look_even_when_it_was_the_last_in_flight() {
    let gpu = gpu();
    gpu.define("illegal", |_, _, _| {
        Err(euhedral_gpu::fake::FakeFault::IllegalAddress)
    });
    let module =
        FakeModule::from_source(&gpu, "// fake source", &CompileOptions::default()).unwrap();
    let bad = module.kernel("illegal").unwrap();
    let mut streams = gpu
        .open_lanes_at(LaneId(0), &["a", "b"], &[StreamOptions::default(); 2])
        .unwrap();
    let mut b = streams.pop().unwrap();
    let mut a = streams.pop().unwrap();
    // Lane b's retirement is armed first and completes cleanly. Then lane a faults, and lane b's later marker waits
    // behind lane a's, so it never completes.
    LaneSubmit::arm_retirement(&mut b).unwrap();
    a.submit(|q| {
        // SAFETY: the fake kernel touches no memory.
        unsafe { q.launch(&bad, euhedral_gpu::Launch::linear(1, 1), &Args::new()) }
    })
    .unwrap();
    let after_fault = LaneSubmit::mark(&mut a).unwrap().unwrap();
    LaneSubmit::await_marker(&mut b, after_fault).unwrap();
    LaneSubmit::mark(&mut b).unwrap().unwrap();
    gpu.run_until_idle();
    assert!(
        gpu.poisoned().is_none(),
        "the host has not learned of the fault"
    );
    // The one retirement in flight arrives with status 0, and is the last: the look that confirms it still happens, and
    // finds the fault.
    let retired = LaneSubmit::poll_retired(&mut b).expect("the retirement arrived");
    assert_eq!(retired.seq, 0);
    assert!(
        matches!(retired.failure, Some(GpuError::Poisoned(_))),
        "{:?}",
        retired.failure
    );
}
