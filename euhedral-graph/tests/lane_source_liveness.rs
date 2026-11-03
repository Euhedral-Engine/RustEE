//! A lane source with items in its intake is entered by some worker, even while the first-ranked
//! worker is held inside a frame.
//!
//! Lane sources are served only when a worker enters their handles. A worker whose handles all looked
//! empty to it used to sit out for good (it entered no handle, so nothing changed its view), which left
//! the first-ranked worker, the only one that never sits out, alone in entering the handles of the whole
//! lattice. Held in a frame, it left items in the lanes' intakes unserved for as long as the frame
//! lasted, with every other worker idle.
//!
//! The lattice is warmed with a stream of items until every worker has a body-cost history (a worker
//! sits out only with one) and left alone until its workers have found the lanes empty. The first-ranked
//! worker is then held in a frame that waits for the test's flag, and the test delivers items.
//!
//! `EUHEDRAL_TEST_CPUS` (default 0,1,2,3) picks the lattice's CPUs; see `lane_source_lattice`.

#![cfg(target_os = "linux")]
#![allow(
    clippy::disallowed_methods,
    clippy::disallowed_types,
    reason = "the test thread is not a lattice worker: it delivers, and the held frame waits on a flag"
)]

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

use euhedral_core::config::{FragmentConfig, IdlePolicy, LatticeConfig};
use euhedral_core::control_plane::ControlPlaneLattice;
use euhedral_core::flow_control::UpstreamRegistry;
use euhedral_core::frames::{Frame, FrameError, FrameHeader};
use euhedral_core::generics::LatticeSource;
use euhedral_core::impls::{BaseCloneableObject, DefaultExecutor};
use euhedral_graph::testing::{FakeLane, FakeLaneError, FakeLanes};
use euhedral_graph::{
    Deliver, GroupId, LaneCx, LaneId, LaneIntakes, LanePoolBuilder, LaneSource, LaneWork, Marker,
    Outcome,
};
use euhedral_hardware_utils::affinity;

const LANES: usize = 4;
/// Enough frames for every worker to time the 32 bodies of a body-cost window (one in 256 is timed).
const WARM_UP: usize = 200_000;
const HELD_ITEMS: usize = 10;
const PATIENCE: Duration = Duration::from_secs(10);

struct Shared {
    /// Frames of the warm-up that ran.
    ran: AtomicUsize,
    /// Items of the held phase that were served (run in a lane source).
    served: AtomicUsize,
    held: AtomicBool,
    release: AtomicBool,
    registry: Mutex<Option<Arc<UpstreamRegistry>>>,
}

/// A frame that waits for the test's flag when the first-ranked worker runs it.
struct Hold {
    header: FrameHeader,
    shared: Arc<Shared>,
}

impl Frame for Hold {
    fn header(&self) -> &FrameHeader {
        &self.header
    }

    fn header_mut(&mut self) -> &mut FrameHeader {
        &mut self.header
    }

    fn execute(&mut self) -> Result<(), FrameError> {
        let registry = self
            .shared
            .registry
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
            .expect("the registry is set before items are delivered");
        let first_ranked =
            affinity::current_cpu().and_then(|cpu| registry.thread_rank(cpu)) == Some(1);
        if first_ranked && !self.shared.held.swap(true, Ordering::SeqCst) {
            while !self.shared.release.load(Ordering::Acquire) {
                std::thread::yield_now();
            }
        }
        Ok(())
    }
}

struct Warm {
    header: FrameHeader,
    shared: Arc<Shared>,
}

impl Frame for Warm {
    fn header(&self) -> &FrameHeader {
        &self.header
    }

    fn header_mut(&mut self) -> &mut FrameHeader {
        &mut self.header
    }

    fn execute(&mut self) -> Result<(), FrameError> {
        self.shared.ran.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Warm,
    Hold,
    Held,
}

struct Item {
    kind: Kind,
    index: u64,
    shared: Arc<Shared>,
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

    fn run(&mut self, _cx: &mut Cx<'_>) -> Outcome<std::convert::Infallible, FakeLaneError> {
        if self.kind == Kind::Held {
            self.shared.served.fetch_add(1, Ordering::SeqCst);
        }
        Outcome::Finished
    }

    fn then(&mut self, _marker: Option<Marker>, cx: &mut Cx<'_>) {
        // Unordered, so that whichever worker is free takes the frame.
        let mut header = FrameHeader::new(self.index);
        header.randomize_hash(self.index.wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let shared = self.shared.clone();
        match self.kind {
            Kind::Hold => cx.ready(Box::new(Hold { header, shared })),
            Kind::Warm => cx.ready(Box::new(Warm { header, shared })),
            Kind::Held => {}
        }
    }

    fn retired(self, _marker: Marker, _failure: Option<FakeLaneError>, _cx: &mut Cx<'_>) {}

    fn failed(self, _error: FakeLaneError, _cx: &mut Cx<'_>) {}
}

fn cpus() -> Vec<usize> {
    let allowed = affinity::base_mask();
    let wanted: Vec<usize> = std::env::var("EUHEDRAL_TEST_CPUS")
        .ok()
        .map(|v| v.split(',').filter_map(|c| c.trim().parse().ok()).collect())
        .filter(|v: &Vec<usize>| !v.is_empty())
        .unwrap_or_else(|| (0..4).collect());
    let mut cpus: Vec<usize> = wanted
        .into_iter()
        .filter(|&c| allowed.contains(c))
        .collect();
    if cpus.len() < 2 {
        cpus = allowed.iter().take(4).collect();
    }
    assert!(
        cpus.len() >= 2,
        "the test needs two CPUs: the held worker and one to serve"
    );
    cpus
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
    config.name = "lane-source-liveness".into();
    config.shard_name = "lane-source-liveness".into();
    config.allowed_cpus = cpus.iter().copied().collect();
    config.shutdown_timeout = Duration::from_secs(5);
    config.startup_timeout = Duration::from_secs(10);
    let lattice = ControlPlaneLattice::new(config).unwrap();
    lattice.start().unwrap();
    lattice
}

#[test]
fn lane_sources_are_entered_while_the_first_ranked_worker_is_held_in_a_frame() {
    let lattice = lattice(&cpus());
    let mut builder = LanePoolBuilder::new();
    let first = builder.reserve(LANES);
    let (fake, submitters, queues) = FakeLanes::open(first, LANES);
    let _fake = fake.quiet();
    for (i, queue) in queues.into_iter().enumerate() {
        builder.set(LaneId(i as u16), queue);
    }
    let pool = builder.build();
    let group: Vec<LaneId> = (0..LANES as u16).map(LaneId).collect();
    let (intakes, inputs) = LaneIntakes::<Item>::new(pool, &[&group]);
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
    let shared = Arc::new(Shared {
        ran: AtomicUsize::new(0),
        served: AtomicUsize::new(0),
        held: AtomicBool::new(false),
        release: AtomicBool::new(false),
        registry: Mutex::new(Some(lattice.registry().clone())),
    });
    let item = |kind, index| Item {
        kind,
        index,
        shared: shared.clone(),
    };
    let deliver = |item: Item| intakes.deliver(item, Deliver::Group(GroupId(0))).unwrap();
    let until = |what: &str, condition: &mut dyn FnMut() -> bool| {
        let deadline = Instant::now() + PATIENCE;
        while !condition() {
            assert!(Instant::now() < deadline, "timed out waiting for {what}");
            std::thread::sleep(Duration::from_millis(1));
        }
    };

    for index in 0..WARM_UP as u64 {
        deliver(item(Kind::Warm, index));
        if index % 256 == 0 {
            std::thread::sleep(Duration::from_micros(300));
        }
    }
    until("the warm-up frames", &mut || {
        shared.ran.load(Ordering::SeqCst) == WARM_UP
    });
    // Every worker finds the lanes empty and, with a body-cost history, sits out.
    std::thread::sleep(Duration::from_millis(1500));

    // Hold the first-ranked worker: a frame that waits is delivered until that worker takes one.
    let mut index = WARM_UP as u64;
    until("the first-ranked worker to be held", &mut || {
        if !shared.held.load(Ordering::SeqCst) {
            deliver(item(Kind::Hold, index));
            index += 1;
        }
        shared.held.load(Ordering::SeqCst)
    });

    for k in 0..HELD_ITEMS as u64 {
        deliver(item(Kind::Held, 1_000_000 + k));
    }
    let served = || shared.served.load(Ordering::SeqCst) == HELD_ITEMS;
    let deadline = Instant::now() + PATIENCE;
    while !served() && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(1));
    }
    let served_in_time = served();
    shared.release.store(true, Ordering::Release);
    assert!(
        served_in_time,
        "no worker entered a lane source for {PATIENCE:?} with {} items in the intake while the first-ranked worker was held \
         (served {} of {HELD_ITEMS})",
        HELD_ITEMS - shared.served.load(Ordering::SeqCst),
        shared.served.load(Ordering::SeqCst),
    );
    lattice.close();
    let _: Vec<Arc<LaneSource<FakeLane, (), Item>>> = sources;
}
