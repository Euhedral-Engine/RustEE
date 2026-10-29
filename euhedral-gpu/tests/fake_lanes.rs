//! Lane sources on streams of the fake backend: chains across lanes, retirement, a foreign lane in the pool, a full
//! marker ring and the priority. No GPU, no wall clock.
#![cfg(feature = "fake")]

mod common;

use std::marker::PhantomData;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use common::{Item, Plan, Probe, collect_all, lock, pump};
use euhedral_core::generics::LatticeSource as _;
use euhedral_gpu::fake::{FAKE_GREATEST_PRIORITY, FakeGpu, FakeModule, FakeOp, FakeStream};
use euhedral_gpu::{
    Args, CompileOptions, Gpu, LaneSpec, Module, Stream, StreamOptions, StreamPriority,
    open_lane_sources,
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
        host_word: Arc::new(AtomicU32::new(0)),
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
    for round in 0..1000 {
        pump(&opened.sources);
        gpu.run_until_idle();
        if !collect.received().is_empty() {
            break;
        }
        assert!(round < 999, "the chain did not finish");
    }
    pump(&opened.sources);
    assert_eq!(collect.received(), vec![u64::from(expected)]);
    assert_eq!(plan.host_word.load(Ordering::SeqCst), expected);
    assert!(lock(&plan.probe.failed).is_empty());
    assert!(lock(&plan.probe.retired).iter().all(|(_, f)| f.is_none()));
    // Odd steps ran on lane x (3); even steps on the group's lanes.
    let lanes: Vec<(u16, String)> = gpu
        .log()
        .into_iter()
        .filter_map(|op| match op {
            FakeOp::Launch { lane, kernel } => Some((lane, kernel)),
            _ => None,
        })
        .collect();
    assert_eq!(lanes.len(), steps as usize);
    for (i, (lane, _)) in lanes.iter().enumerate() {
        assert_eq!(*lane == 3, i % 2 == 1, "step {i} ran on lane {lane}");
    }
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
        host_word: Arc::new(AtomicU32::new(0)),
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
    let tags: Vec<u64> = chain.probe.ran().iter().map(|t| t / 1_000_000).collect();
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
    let tags: Vec<u64> = chain.probe.ran().iter().map(|t| t / 1_000_000).collect();
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

#[test]
fn a_retirement_failure_from_a_faulted_device_reaches_the_item() {
    let gpu = gpu();
    gpu.define("trap", |_, _, _| Err(euhedral_gpu::fake::FakeFault::Trap));
    let opened = open_lane_sources::<_, (), Item1>(&gpu, specs(&["a"]), &[]).unwrap();
    let _collect = collect_all(&opened.sources);
    let module =
        FakeModule::from_source(&gpu, "// fake source", &CompileOptions::default()).unwrap();
    let kernel = module.kernel("trap").unwrap();
    let word = gpu.alloc(4);
    let plan = Arc::new(Plan::<FakeStream> {
        steps: 1,
        route: Box::new(|_| Deliver::Lane(LaneId(0))),
        launches: Box::new(move |_| vec![(kernel.clone(), Args::new())]),
        device_word: word,
        host_word: Arc::new(AtomicU32::new(0)),
        probe: Probe::new(),
        retire_steps: false,
        _stream: PhantomData,
    });
    opened
        .intakes
        .deliver(Item1::first(&plan), Deliver::Lane(LaneId(0)))
        .unwrap();
    // Step 0 traps on the device; the read item (step 1) is delivered behind its marker, and its retirement reports
    // the poisoned device.
    for _ in 0..10 {
        opened.sources[0].request(1);
        gpu.run_until_idle();
    }
    opened.sources[0].request(1);
    let retired = lock(&plan.probe.retired).clone();
    assert_eq!(retired.len(), 1);
    assert!(retired[0].1.is_some(), "{retired:?}");
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
