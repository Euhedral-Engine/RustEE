//! Lane sources on a real lattice: three sources of one group and one source with its own intake,
//! 4 workers. Every delivered item makes one frame, whether it ends at once or after a retirement,
//! and every frame executes exactly once.
//!
//! `EUHEDRAL_TEST_CPUS` (default 0,1,2,3) picks the lattice's CPUs; as in the other lattice tests, the
//! test passes without checking anything when this process may not run on all of them.

#![cfg(target_os = "linux")]
#![allow(
    clippy::disallowed_methods,
    reason = "the test thread is not a lattice worker: it delivers and retires on a timer"
)]

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use euhedral_core::config::{FragmentConfig, IdlePolicy, LatticeConfig};
use euhedral_core::control_plane::ControlPlaneLattice;
use euhedral_core::frames::{Frame, FrameError, FrameHeader};
use euhedral_core::generics::LatticeSource;
use euhedral_core::impls::{BaseCloneableObject, DefaultExecutor};
use euhedral_graph::testing::{FakeLane, FakeLaneError, FakeLanes};
use euhedral_graph::{
    Deliver, GroupId, LaneCx, LaneId, LaneIntakes, LanePoolBuilder, LaneSource, LaneWork, Marker,
    Outcome,
};
use euhedral_hardware_utils::affinity;

const ITEMS: usize = 20_000;
const GROUP_SOURCES: usize = 3;

type Counters = Arc<Vec<AtomicUsize>>;

struct CountFrame {
    header: FrameHeader,
    index: usize,
    counters: Counters,
}

impl Frame for CountFrame {
    fn header(&self) -> &FrameHeader {
        &self.header
    }

    fn header_mut(&mut self) -> &mut FrameHeader {
        &mut self.header
    }

    fn execute(&mut self) -> Result<(), FrameError> {
        self.counters[self.index].fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

struct Item {
    index: usize,
    retire: bool,
    counters: Counters,
    failures: Arc<AtomicUsize>,
}

impl Item {
    fn frame(&self) -> Box<dyn Frame> {
        Box::new(CountFrame {
            header: FrameHeader::new(self.index as u64),
            index: self.index,
            counters: self.counters.clone(),
        })
    }
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
        if self.retire {
            Outcome::Submitted { retire: true }
        } else {
            Outcome::Finished
        }
    }

    fn then(&mut self, _marker: Option<Marker>, cx: &mut Cx<'_>) {
        if !self.retire {
            cx.ready(self.frame());
        }
    }

    fn retired(self, _marker: Marker, failure: Option<FakeLaneError>, cx: &mut Cx<'_>) {
        if failure.is_some() {
            self.failures.fetch_add(1, Ordering::SeqCst);
        }
        cx.ready(self.frame());
    }

    fn failed(self, _error: FakeLaneError, _cx: &mut Cx<'_>) {
        self.failures.fetch_add(1, Ordering::SeqCst);
    }
}

fn cpus() -> Vec<usize> {
    std::env::var("EUHEDRAL_TEST_CPUS")
        .ok()
        .map(|v| {
            v.split(',')
                .filter_map(|c| c.trim().parse().ok())
                .collect::<Vec<usize>>()
        })
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| (0..4).collect())
}

fn lattice(cpus: &[usize]) -> Arc<ControlPlaneLattice> {
    let park = Duration::from_micros(100);
    let idle = IdlePolicy::fixed(park, IdlePolicy::DEFAULT_CONTENTION_HALF_LIFE)
        .unwrap()
        .with_unproductive_park(park);
    let fragment = FragmentConfig::default().with_idle_policy(idle);
    let mut config = LatticeConfig::new(Arc::new(BaseCloneableObject::new(
        fragment,
        DefaultExecutor::new(),
    )));
    config.name = "lane-source-test".into();
    config.shard_name = "lane-source-test".into();
    config.allowed_cpus = cpus.iter().copied().collect();
    config.shutdown_timeout = Duration::from_secs(5);
    config.startup_timeout = Duration::from_secs(10);
    let lattice = ControlPlaneLattice::new(config).unwrap();
    lattice.start().unwrap();
    lattice
}

#[test]
fn lane_sources_on_a_real_lattice_deliver_every_frame_once() {
    let cpus = cpus();
    if !cpus.iter().all(|&cpu| affinity::base_mask().contains(cpu)) {
        eprintln!("skipped: this process may not run on CPUs {cpus:?}");
        return;
    }
    let lattice = lattice(&cpus);

    let lanes = GROUP_SOURCES + 1;
    let mut builder = LanePoolBuilder::new();
    let first = builder.reserve(lanes);
    let (fake, submitters, queues) = FakeLanes::open(first, lanes);
    let fake = fake.quiet();
    for (i, queue) in queues.into_iter().enumerate() {
        builder.set(LaneId(i as u16), queue);
    }
    let pool = builder.build();
    let group: Vec<LaneId> = (0..GROUP_SOURCES as u16).map(LaneId).collect();
    let (intakes, inputs) = LaneIntakes::<Item>::new(pool, &[&group]);
    let own_lane = LaneId(GROUP_SOURCES as u16);

    let sources: Vec<_> = submitters
        .into_iter()
        .zip(inputs)
        .map(|(submitter, input)| LaneSource::new(submitter, (), input, intakes.clone()))
        .collect();
    for source in &sources {
        lattice
            .add_upstream(source.clone() as Arc<dyn LatticeSource>)
            .unwrap();
    }

    let counters: Counters = Arc::new((0..ITEMS).map(|_| AtomicUsize::new(0)).collect());
    let failures = Arc::new(AtomicUsize::new(0));
    let done = Arc::new(AtomicBool::new(false));

    // Retirements arrive on a timer, as a driver's callbacks would.
    let retirer = {
        let (fake, done) = (fake.clone(), done.clone());
        std::thread::spawn(move || {
            while !done.load(Ordering::Acquire) {
                fake.retire_all();
                std::thread::sleep(Duration::from_micros(200));
            }
        })
    };

    for index in 0..ITEMS {
        let item = Item {
            index,
            retire: index % 3 == 0,
            counters: counters.clone(),
            failures: failures.clone(),
        };
        let to = if index % 5 == 0 {
            Deliver::Lane(own_lane)
        } else {
            Deliver::Group(GroupId(0))
        };
        intakes.deliver(item, to);
        if index % 512 == 0 {
            std::thread::sleep(Duration::from_micros(100));
        }
    }

    let deadline = Instant::now() + Duration::from_secs(30);
    while counters.iter().any(|c| c.load(Ordering::SeqCst) == 0) {
        assert!(Instant::now() < deadline, "every frame arrives within 30 s");
        std::thread::sleep(Duration::from_millis(1));
    }
    done.store(true, Ordering::Release);
    retirer.join().unwrap();
    // Let any wrongly duplicated frame show itself.
    std::thread::sleep(Duration::from_millis(50));
    for (index, counter) in counters.iter().enumerate() {
        assert_eq!(counter.load(Ordering::SeqCst), 1, "frame {index} executed once");
    }
    assert_eq!(failures.load(Ordering::SeqCst), 0);
    lattice.close();
    let _: Vec<Arc<LaneSource<FakeLane, (), Item>>> = sources;
}
