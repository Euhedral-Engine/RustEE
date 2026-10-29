use super::*;
use crate::flow_control::testing::{Fate, Fates, Tracked, frame};
use crate::flow_control::{Claim, UpstreamHandle};
use euhedral_hardware_utils::monitor::{HardwareUtilization, HardwareUtilizationParts};
use euhedral_hardware_utils::topology::CpuInfo;
use euhedral_hardware_utils::topology::source::{
    CacheDomain, CoreKind, LogicalCpu, TopologyInput, normalize,
};
use std::collections::VecDeque;
use std::num::NonZeroUsize;
use std::panic::{self, AssertUnwindSafe};
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

const L1: u64 = 32 * 1024;
const L2: u64 = 1024 * 1024;

/// Socket 0 has eight cores, socket 1 two; core `c` has CPUs `2c` and `2c + 1`, which share a
/// 32 KiB L1 and a 1 MiB L2.
fn topology() -> Topology {
    let socket_of = |core: usize| if core < 8 { 0 } else { 1 };
    let cpus = (0..20)
        .map(|cpu| LogicalCpu {
            id: cpu,
            socket_key: format!("linux:package:{:010}", socket_of(cpu / 2)),
            die_key: "linux:die:0000000000".to_string(),
            core_key: format!("linux:core:{:010}", cpu / 2),
            kind: CoreKind::Unknown,
        })
        .collect();
    let caches = (0..10)
        .flat_map(|core| {
            let sharers = CpuSet::range(2 * core, 2 * core + 2);
            [(1, L1), (2, L2)].map(|(level, size_bytes)| CacheDomain {
                level,
                size_bytes,
                line_size_bytes: 64,
                sharers: sharers.clone(),
            })
        })
        .collect();
    normalize(TopologyInput {
        provider: "linux".to_string(),
        cpus,
        caches,
    })
    .unwrap()
}

fn directory() -> Arc<CacheDirectory> {
    CacheDirectory::new(UpstreamRegistry::new(), topology())
}

fn config(partitions: usize) -> CacheConfig {
    CacheConfig::default().with_partitions(NonZeroUsize::new(partitions).unwrap())
}

fn clone_config(core: usize) -> CloneConfig {
    CloneConfig::new("shard", core, CpuSet::range(2 * core, 2 * core + 2))
}

/// A cache on `core`, run from the core's first CPU.
fn cache_on(
    directory: &Arc<CacheDirectory>,
    config: &CacheConfig,
    core: usize,
) -> Arc<ControlPlaneCache> {
    ControlPlaneCache::new(config, &clone_config(core), 2 * core, false, directory)
}

fn cache(partitions: usize) -> Arc<ControlPlaneCache> {
    cache_on(&directory(), &config(partitions), 0)
}

/// An ordered frame: it routes by its id hash.
fn ordered(id: u64) -> Box<dyn Frame> {
    frame(id)
}

/// An unordered frame with routing hash `hash`.
fn unordered(id: u64, hash: u64) -> Box<dyn Frame> {
    let mut frame = frame(id);
    frame.header_mut().set_routing_hash(hash);
    assert!(!frame.header().is_ordered());
    frame
}

fn ids(frames: &[Box<dyn Frame>]) -> Vec<u64> {
    frames
        .iter()
        .map(|frame| frame.header().id_hash())
        .collect()
}

/// Hands out queued frames and records the demand requested of it.
#[derive(Default)]
struct Source {
    frames: Mutex<VecDeque<Box<dyn Frame>>>,
    requested: AtomicU64,
}

impl UpstreamHandle for Source {
    fn sequence(&self) -> u64 {
        0
    }

    fn pull(
        &self,
        _claim: &mut Claim,
        consumer: &mut dyn FnMut(Box<dyn Frame>),
        stop: &mut dyn FnMut(&dyn Frame) -> bool,
        demand: u64,
    ) -> Option<u64> {
        let mut frames = self.frames.lock().unwrap();
        let mut pulled = 0;
        while pulled < demand {
            match frames.front() {
                Some(frame) if !stop(&**frame) => {
                    consumer(frames.pop_front().unwrap());
                    pulled += 1;
                }
                _ => break,
            }
        }
        Some(pulled)
    }

    fn request(&self, _claim: &mut Claim, demand: u64) {
        self.requested.fetch_add(demand, Ordering::SeqCst);
    }

    fn is_complete(&self) -> bool {
        false
    }
}

/// Makes the calling thread a worker of `directory`'s registry, fed by a new source.
fn source(directory: &CacheDirectory, frames: Vec<Box<dyn Frame>>) -> Arc<Source> {
    let source = Arc::new(Source {
        frames: Mutex::new(frames.into()),
        ..Default::default()
    });
    directory.registry().register();
    directory.registry().add_handle(source.clone());
    source
}

#[test]
fn a_cache_is_named_and_placed_by_its_clone() {
    let directory = directory();
    let cache = ControlPlaneCache::new(&config(2), &clone_config(3), 7, false, &directory);
    assert_eq!(cache.name(), "shard-ControlPlaneCache-3");
    assert_eq!(cache.core(), 3);
    assert_eq!(cache.cpu(), 7);
    assert!(Arc::ptr_eq(&directory.cache_at(7).unwrap(), &cache));
    assert!(cache.is_drained());
    assert_eq!(cache.local_cache_count(), 0);
    assert_eq!(cache.cap_factor(), 1.0);
}

#[test]
fn chunks_fill_the_budgeted_share_of_l1_and_l2() {
    let directory = directory();
    // (1 MiB + 32 KiB) / 8 partitions * 0.7 = 94,617 bytes: 64 KiB, 4,096 boxed frames.
    let cache = cache_on(&directory, &config(8), 0);
    assert_eq!(cache.chunk_size(), 4096);
    assert_eq!(cache.frame_quota(), 8 * 4096);
    assert_eq!(cache.max_local_cache_count(), 8 * 4096);

    // SMT siblings split the core's caches: half of each, 32 KiB, 2,048 frames.
    let sibling = ControlPlaneCache::new(&config(8), &clone_config(1), 2, true, &directory);
    assert_eq!(sibling.chunk_size(), 2048);
}

#[test]
fn chunks_never_shrink_below_the_minimum() {
    let tiny = CacheConfig::default()
        .with_memory_budget(0.001)
        .unwrap()
        .with_partitions(NonZeroUsize::new(64).unwrap());
    let layout = topology().cache_layout(0).unwrap().clone();
    assert_eq!(chunk_size(&tiny, Some(&layout), false), MIN_CHUNK);
    assert_eq!(chunk_size(&config(8), None, false), MIN_CHUNK);
    let unshared = CpuCacheLayout {
        mask_l1: CpuSet::range(0, 1),
        mask_l2: CpuSet::range(0, 1),
        ..layout
    };
    assert_eq!(chunk_size(&config(8), Some(&unshared), true), 4096);
}

#[test]
fn caches_on_crowded_l2s_split_them() {
    let layout = topology().cache_layout(0).unwrap().clone();
    let crowded = CpuCacheLayout {
        mask_l2: CpuSet::range(0, 4),
        ..layout
    };
    // (256 KiB + 32 KiB) / 8 * 0.7 = 25,804 bytes: 16 KiB, 1,024 frames.
    assert_eq!(chunk_size(&config(8), Some(&crowded), false), 1024);
}

#[test]
fn pushed_frames_are_counted_and_taken_in_partition_order() {
    let cache = cache(2);
    FlowContext::initialize();
    // Hashes in the lower half go to the first partition, the upper half to the second.
    for (id, hash) in [(1, 0), (2, 1), (3, 1 << 63), (4, u64::MAX)] {
        cache.push(unordered(id, hash));
    }
    let satisfied = FlowContext::with(|context| context.unwrap().satisfied_request());
    FlowContext::clear();
    assert_eq!(satisfied, 4);
    assert_eq!(cache.local_cache_count(), 4);
    assert!(!cache.is_drained());

    let mut taken_frames = Vec::new();
    assert_eq!(cache.greedy(|frame| taken_frames.push(frame), 3), 3);
    assert_eq!(ids(&taken_frames), [1, 2, 3]);
    assert_eq!(cache.local_cache_count(), 1);

    taken_frames.clear();
    assert_eq!(cache.greedy(|frame| taken_frames.push(frame), 3), 1);
    assert_eq!(ids(&taken_frames), [4]);
    assert!(cache.is_drained());
}

#[test]
fn a_zero_limit_takes_nothing() {
    let cache = cache(2);
    cache.push(ordered(1));
    assert_eq!(cache.greedy(|_| panic!("not consumed"), 0), 0);
    assert_eq!(
        cache.greedy_while(|_| panic!("not consumed"), |_| panic!("not asked"), 0),
        0
    );
    assert_eq!(cache.pull(0), 0);
    assert_eq!(cache.upstream_pull(|_| panic!("not consumed"), 0), 0);
    assert_eq!(cache.local_cache_count(), 1);
}

#[test]
fn a_stopped_partition_keeps_its_head_while_others_progress() {
    let cache = cache(2);
    cache.push(unordered(1, 0));
    cache.push(unordered(2, 1));
    cache.push(unordered(3, u64::MAX));

    let mut taken_frames = Vec::new();
    let taken = cache.greedy_while(
        |frame| taken_frames.push(frame),
        |frame| frame.header().id_hash() == 1,
        3,
    );
    assert_eq!(taken, 1);
    assert_eq!(ids(&taken_frames), [3]);
    assert_eq!(cache.local_cache_count(), 2);

    taken_frames.clear();
    assert_eq!(cache.greedy(|frame| taken_frames.push(frame), 3), 2);
    assert_eq!(ids(&taken_frames), [1, 2]);
}

#[test]
fn the_queue_grows_past_the_frame_quota() {
    let cache = cache(1);
    let frames = cache.frame_quota() + 17;
    for id in 0..frames {
        cache.push(ordered(id));
    }
    assert_eq!(cache.local_cache_count(), frames);
    assert!(cache.local_cache_count() > cache.max_local_cache_count());
    let mut taken_frames = Vec::new();
    assert_eq!(
        cache.greedy(|frame| taken_frames.push(frame), u64::MAX),
        frames
    );
    assert_eq!(ids(&taken_frames), (0..frames).collect::<Vec<_>>());
}

/// Takes greedily from two partitions where taking the second partition's frame queues one in the first,
/// which only another walk over the partitions finds.
fn greedy_with_refill(threshold: u64) -> u64 {
    let directory = directory();
    let config = config(2).with_ring_walk_reset_threshold(threshold);
    let cache = cache_on(&directory, &config, 0);
    cache.push(unordered(1, u64::MAX));
    let refill = cache.clone();
    cache.greedy(
        |frame| {
            if frame.header().id_hash() == 1 {
                refill.push(unordered(2, 1));
            }
        },
        10,
    )
}

#[test]
fn a_productive_walk_walks_the_partitions_again() {
    assert_eq!(greedy_with_refill(0), 2);
}

#[test]
fn a_walk_at_the_threshold_ends_the_greedy_take() {
    assert_eq!(greedy_with_refill(1), 1);
}

#[test]
fn a_consumer_that_panics_leaves_the_count_and_partition_usable() {
    let cache = cache(1);
    for id in 0..3 {
        cache.push(ordered(id));
    }
    let failed = panic::catch_unwind(AssertUnwindSafe(|| {
        cache.greedy(
            |frame| {
                if frame.header().id_hash() == 1 {
                    panic!("consumer failed");
                }
            },
            3,
        )
    }));
    assert!(failed.is_err());
    assert_eq!(cache.local_cache_count(), 1);

    let mut taken_frames = Vec::new();
    assert_eq!(cache.greedy(|frame| taken_frames.push(frame), 3), 1);
    assert_eq!(ids(&taken_frames), [2]);
}

#[test]
fn work_stealing_takes_unordered_frames_and_leaves_ordered_heads() {
    let directory = directory();
    let thief = cache_on(&directory, &config(1), 0);
    let victim = cache_on(&directory, &config(1), 1);
    victim.push(unordered(1, 5));
    victim.push(ordered(2));
    victim.push(unordered(3, 7));

    let mut stolen = Vec::new();
    assert_eq!(
        thief.work_steal(|frame| stolen.push(frame), 10, victim.cpu()),
        1
    );
    assert_eq!(ids(&stolen), [1]);
    assert_eq!(victim.local_cache_count(), 2);

    // The cursor wraps over the CPUs.
    victim.push(unordered(4, 9));
    let span = directory.topology().cpu_span();
    let mut owner = Vec::new();
    victim.greedy_while(
        |frame| owner.push(frame),
        |frame| frame.header().id_hash() == 3,
        10,
    );
    assert_eq!(ids(&owner), [2]);
    stolen.clear();
    assert_eq!(
        thief.work_steal(|frame| stolen.push(frame), 10, victim.cpu() + span),
        2
    );
    assert_eq!(ids(&stolen), [3, 4]);
    assert!(victim.is_drained());
}

/// The owner runs the front frame of a one-partition victim on another thread and blocks in it;
/// returns how many of the frames behind it a polling thief steals meanwhile.
fn stolen_while_the_owner_runs_a_frame(owner_polls: bool) -> u64 {
    let directory = directory();
    let thief = cache_on(&directory, &config(1), 0);
    let victim = cache_on(&directory, &config(1), 1);
    victim.push(unordered(0, 5));
    for id in 1..=3 {
        victim.push(unordered(id, 5));
    }
    let (running_tx, running) = mpsc::channel();
    let (release, released) = mpsc::channel::<()>();
    let owner = {
        let victim = victim.clone();
        thread::spawn(move || {
            let blocking = |_frame: Box<dyn Frame>| {
                running_tx.send(()).unwrap();
                released.recv().unwrap();
            };
            if owner_polls {
                victim.poll(blocking, 1)
            } else {
                victim.greedy(blocking, 1)
            }
        })
    };
    running.recv().unwrap();
    let mut stolen = Vec::new();
    let count = thief.work_steal_polled(|frame| stolen.push(frame), 10, victim.cpu());
    release.send(()).unwrap();
    assert_eq!(owner.join().unwrap(), 1);
    count
}

#[test]
fn a_poll_frees_the_partition_while_its_frame_runs() {
    assert_eq!(stolen_while_the_owner_runs_a_frame(true), 3);
}

#[test]
fn a_greedy_take_holds_the_partition_while_its_frames_run() {
    assert_eq!(stolen_while_the_owner_runs_a_frame(false), 0);
}

#[test]
fn polls_take_one_frame_at_a_time_in_order_and_leave_ordered_heads_to_the_owner() {
    let directory = directory();
    let thief = cache_on(&directory, &config(1), 0);
    let victim = cache_on(&directory, &config(1), 1);
    victim.push(unordered(1, 5));
    victim.push(ordered(2));
    victim.push(unordered(3, 7));

    let mut stolen = Vec::new();
    assert_eq!(
        thief.work_steal_polled(|frame| stolen.push(frame), 10, victim.cpu()),
        1
    );
    assert_eq!(ids(&stolen), [1]);

    let mut owner = Vec::new();
    assert_eq!(victim.poll(|frame| owner.push(frame), 1), 1);
    assert_eq!(ids(&owner), [2]);
    assert_eq!(victim.local_cache_count(), 1);
    assert_eq!(victim.poll(|frame| owner.push(frame), 10), 1);
    assert_eq!(ids(&owner), [2, 3]);
    assert!(victim.is_drained());
    assert_eq!(victim.poll(|_| panic!("nothing left"), 10), 0);
}

#[test]
fn polls_walk_every_partition() {
    let cache = cache(4);
    for id in 0..8 {
        cache.push(unordered(id, id + 100));
    }
    let mut taken = Vec::new();
    assert_eq!(cache.poll(|frame| taken.push(frame), 2), 2);
    assert_eq!(cache.poll(|frame| taken.push(frame), 10), 6);
    let mut ids = ids(&taken);
    ids.sort_unstable();
    assert_eq!(ids, (0..8).collect::<Vec<_>>());
    assert!(cache.is_drained());
}

/// An unordered frame that must stay as close to `origin_core` (of `origin_socket`) as `policy` says.
fn local(
    id: u64,
    policy: RoutingPolicy,
    origin_core: usize,
    origin_socket: usize,
) -> Box<dyn Frame> {
    let mut frame = unordered(id, id + 100);
    let header = frame.header_mut();
    header.set_routing_policy(policy);
    header.set_origin(Some(CpuInfo {
        cpu: 2 * origin_core,
        core: origin_core,
        socket: origin_socket,
    }));
    frame
}

/// The cache of core `core`'s second CPU: the SMT buddy of `cache_on(.., core)`.
fn buddy_on(directory: &Arc<CacheDirectory>, core: usize) -> Arc<ControlPlaneCache> {
    ControlPlaneCache::new(
        &config(1),
        &clone_config(core),
        2 * core + 1,
        false,
        directory,
    )
}

#[test]
fn locality_allows_what_each_routing_policy_lets_leave() {
    let at = |policy, core, socket| local(1, policy, core, socket);
    // Core 1 of socket 0; core 2 shares the socket, core 8 is on socket 1.
    let cache_local = at(RoutingPolicy::CacheLocal, 1, 0);
    assert!(locality_allows(&*cache_local, (1, 0), Some((1, 0))));
    assert!(!locality_allows(&*cache_local, (2, 0), Some((1, 0))));
    assert!(locality_allows(&*cache_local, (1, 0), None));
    assert!(!locality_allows(&*cache_local, (2, 0), None));
    // Fell back to core 3 (its core inactive): it keeps to its socket.
    assert!(locality_allows(&*cache_local, (2, 0), Some((3, 0))));
    assert!(!locality_allows(&*cache_local, (8, 1), Some((3, 0))));

    let socket_local = at(RoutingPolicy::SocketLocal, 1, 0);
    assert!(locality_allows(&*socket_local, (2, 0), Some((1, 0))));
    assert!(!locality_allows(&*socket_local, (8, 1), Some((1, 0))));
    assert!(!locality_allows(&*socket_local, (8, 1), None));
    // Fell back to socket 1: anywhere.
    assert!(locality_allows(&*socket_local, (2, 0), Some((8, 1))));

    let anywhere = at(RoutingPolicy::Anywhere, 1, 0);
    assert!(locality_allows(&*anywhere, (8, 1), Some((1, 0))));
    assert!(locality_allows(&*anywhere, (8, 1), None));
    assert!(locality_allows(&*unordered(2, 9), (8, 1), Some((1, 0))));
}

#[test]
fn stealing_leaves_cache_local_frames_to_their_core_in_greedy_and_poll_mode() {
    let directory = directory();
    let victim = cache_on(&directory, &config(1), 1);
    let other_core = cache_on(&directory, &config(1), 2);
    let buddy = buddy_on(&directory, 1);
    for id in 1..=3 {
        victim.push(local(id, RoutingPolicy::CacheLocal, 1, 0));
    }

    assert_eq!(
        other_core.work_steal(|_| panic!("not its core"), 10, victim.cpu()),
        0
    );
    assert_eq!(
        other_core.work_steal_polled(|_| panic!("not its core"), 10, victim.cpu()),
        0
    );
    let mut stolen = Vec::new();
    assert_eq!(
        buddy.work_steal_polled(|frame| stolen.push(frame), 2, victim.cpu()),
        2
    );
    assert_eq!(
        buddy.work_steal(|frame| stolen.push(frame), 10, victim.cpu()),
        1
    );
    assert_eq!(ids(&stolen), [1, 2, 3]);
    assert!(victim.is_drained());
}

#[test]
fn a_cache_local_frame_behind_a_blocking_one_waits_for_its_own_core() {
    let directory = directory();
    let victim = cache_on(&directory, &config(1), 1);
    let other_core = cache_on(&directory, &config(1), 2);
    let buddy = buddy_on(&directory, 1);
    victim.push(local(0, RoutingPolicy::CacheLocal, 1, 0));
    for id in 1..=3 {
        victim.push(local(id, RoutingPolicy::CacheLocal, 1, 0));
    }
    let (running_tx, running) = mpsc::channel();
    let (release, released) = mpsc::channel::<()>();
    let owner = {
        let victim = victim.clone();
        thread::spawn(move || {
            victim.poll(
                |_frame| {
                    running_tx.send(()).unwrap();
                    released.recv().unwrap();
                },
                1,
            )
        })
    };
    running.recv().unwrap();
    assert_eq!(
        other_core.work_steal_polled(|_| panic!("not its core"), 10, victim.cpu()),
        0
    );
    let mut stolen = Vec::new();
    assert_eq!(
        buddy.work_steal_polled(|frame| stolen.push(frame), 10, victim.cpu()),
        3
    );
    release.send(()).unwrap();
    assert_eq!(owner.join().unwrap(), 1);
    assert_eq!(ids(&stolen), [1, 2, 3]);
}

#[test]
fn socket_local_frames_are_stolen_only_on_their_socket_and_anywhere_frames_by_all() {
    let directory = directory();
    let victim = cache_on(&directory, &config(1), 1);
    let same_socket = cache_on(&directory, &config(1), 2);
    let other_socket = cache_on(&directory, &config(1), 8);
    victim.push(local(1, RoutingPolicy::SocketLocal, 1, 0));
    assert_eq!(
        other_socket.work_steal(|_| panic!("not its socket"), 10, victim.cpu()),
        0
    );
    assert_eq!(
        other_socket.work_steal_polled(|_| panic!("not its socket"), 10, victim.cpu()),
        0
    );
    let mut stolen = Vec::new();
    assert_eq!(
        same_socket.work_steal(|frame| stolen.push(frame), 10, victim.cpu()),
        1
    );

    victim.push(local(2, RoutingPolicy::Anywhere, 1, 0));
    victim.push(unordered(3, 7));
    assert_eq!(
        other_socket.work_steal_polled(|frame| stolen.push(frame), 10, victim.cpu()),
        2
    );
    assert_eq!(ids(&stolen), [1, 2, 3]);
}

#[test]
fn a_cache_local_frame_that_fell_back_to_another_core_stays_stealable_on_its_socket() {
    let directory = directory();
    // Its core, 5, was inactive: routing put it on core 1.
    let victim = cache_on(&directory, &config(1), 1);
    let same_socket = cache_on(&directory, &config(1), 2);
    let other_socket = cache_on(&directory, &config(1), 8);
    victim.push(local(1, RoutingPolicy::CacheLocal, 5, 0));
    assert_eq!(
        other_socket.work_steal(|_| panic!("not its socket"), 10, victim.cpu()),
        0
    );
    let mut stolen = Vec::new();
    assert_eq!(
        same_socket.work_steal(|frame| stolen.push(frame), 10, victim.cpu()),
        1
    );
    assert_eq!(ids(&stolen), [1]);
}

/// What a worker on `core` pulls straight from an upstream holding an unordered frame, a frame bound
/// to core 3 and another unordered frame.
fn pulled_on(core: usize) -> Vec<u64> {
    let directory = directory();
    let cache = cache_on(&directory, &config(1), core);
    let _source = source(
        &directory,
        vec![
            unordered(1, 5),
            local(2, RoutingPolicy::CacheLocal, 3, 0),
            unordered(3, 6),
        ],
    );
    let mut pulled = Vec::new();
    cache.upstream_pull(|frame| pulled.push(frame), 10);
    directory.registry().remove_thread();
    ids(&pulled)
}

#[test]
fn upstream_pull_leaves_a_frame_bound_to_another_core_to_routing() {
    assert_eq!(pulled_on(0), [1]);
}

#[test]
fn upstream_pull_runs_a_frame_bound_to_its_own_core() {
    assert_eq!(pulled_on(3), [1, 2, 3]);
}

#[test]
fn work_stealing_finds_nothing_at_an_empty_slot_or_when_off() {
    let directory = directory();
    let thief = cache_on(&directory, &config(1), 0);
    let victim = cache_on(&directory, &config(1), 1);
    victim.push(unordered(1, 5));
    assert_eq!(thief.work_steal(|_| panic!("nothing to steal"), 10, 5), 0);

    let honest = cache_on(&directory, &config(1).with_work_steal(false), 2);
    assert_eq!(
        honest.work_steal(|_| panic!("stealing is off"), 10, victim.cpu()),
        0
    );
    assert_eq!(victim.local_cache_count(), 1);
}

#[test]
fn a_held_partition_is_passed_over() {
    let directory = directory();
    let thief = cache_on(&directory, &config(1), 0);
    let victim = cache_on(&directory, &config(1), 1);
    victim.push(unordered(1, 5));

    let held = victim.partitions[0].lock().unwrap();
    assert_eq!(
        thief.work_steal(|_| panic!("the partition is held"), 10, victim.cpu()),
        0
    );
    assert_eq!(victim.greedy(|_| panic!("the partition is held"), 10), 0);
    drop(held);

    assert_eq!(thief.work_steal(drop, 10, victim.cpu()), 1);
}

#[test]
fn a_closed_cache_fails_new_frames_and_keeps_its_queue() {
    let directory = directory();
    let thief = cache_on(&directory, &config(1), 0);
    let victim = cache_on(&directory, &config(1), 1);
    victim.push(unordered(1, 5));

    victim.close();
    victim.close();

    assert!(victim.is_closed());
    assert!(victim.edge().is_edge_closed());
    assert!(directory.cache_at(victim.cpu()).is_none());
    assert_eq!(
        thief.work_steal(|_| panic!("withdrawn"), 10, victim.cpu()),
        0
    );

    let fates = Fates::default();
    victim.push(Tracked::new(2, &fates));
    let fates = fates.lock().unwrap().clone();
    assert_eq!(fates.len(), 1);
    assert!(matches!(fates[0].1, Fate::Failed(_)));
    assert_eq!(victim.local_cache_count(), 1);
    assert_eq!(victim.greedy(drop, 10), 1);
}

/// A cache on core 0's first CPU paired with one on its second.
fn smt_pair(directory: &Arc<CacheDirectory>) -> (Arc<ControlPlaneCache>, Arc<ControlPlaneCache>) {
    let config = config(2);
    let first = ControlPlaneCache::new(&config, &clone_config(0), 0, true, directory);
    let second = ControlPlaneCache::new(&config, &clone_config(0), 1, true, directory);
    first.pair_with(second.clone());
    (first, second)
}

fn taken_ids(cache: &ControlPlaneCache) -> Vec<u64> {
    let mut taken_frames = Vec::new();
    cache.greedy(|frame| taken_frames.push(frame), 100);
    let mut ids = ids(&taken_frames);
    ids.sort_unstable();
    ids
}

#[test]
fn a_paired_cache_sends_its_odd_lane_to_the_sibling() {
    let directory = directory();
    let (first, second) = smt_pair(&directory);
    assert!(Arc::ptr_eq(first.sibling().unwrap(), &second));
    assert!(second.sibling().is_none());
    FlowContext::initialize();

    for (id, hash) in [(1, 4), (2, 7), (3, u64::MAX), (4, 1 << 63)] {
        first.push(unordered(id, hash));
    }

    assert_eq!(first.local_cache_count(), 2);
    assert_eq!(second.local_cache_count(), 2);
    assert_eq!(FlowContext::with(|c| c.unwrap().satisfied_request()), 4);
    assert_eq!(taken_ids(&first), [1, 4]);
    assert_eq!(taken_ids(&second), [2, 3]);
    assert!(first.is_drained() && second.is_drained());
    FlowContext::clear();
}

#[test]
fn a_paired_cache_pulls_for_both_lanes() {
    let directory = directory();
    let (first, second) = smt_pair(&directory);
    let edge = LatticeEdge::new(directory.registry(), Default::default());
    first.attach(&edge);
    let _source = source(
        &directory,
        (1..=6).map(|id| unordered(id, id + 100)).collect(),
    );

    assert_eq!(first.pull(10), 6);

    assert_eq!(first.local_cache_count(), 3);
    assert_eq!(second.local_cache_count(), 3);
    assert_eq!(taken_ids(&first), [2, 4, 6]);
    assert_eq!(taken_ids(&second), [1, 3, 5]);
    directory.registry().remove_thread();
}

fn tracked(id: u64, hash: u64, fates: &Fates) -> Box<dyn Frame> {
    let mut frame = Tracked::new(id, fates);
    frame.header_mut().set_routing_hash(hash);
    frame
}

#[test]
fn a_retired_lane_fails_its_frames() {
    let directory = directory();
    let (first, second) = smt_pair(&directory);
    second.close();
    let fates = Fates::default();
    first.push(tracked(1, 3, &fates));
    first.push(tracked(2, 2, &fates));
    assert!(matches!(fates.lock().unwrap()[..], [(1, Fate::Failed(_))]));
    assert_eq!(first.local_cache_count(), 1);
    assert_eq!(second.local_cache_count(), 0);

    first.close();
    first.push(tracked(3, 2, &fates));
    assert!(matches!(fates.lock().unwrap()[1], (3, Fate::Failed(_))));
    assert_eq!(first.local_cache_count(), 1);
}

#[test]
fn a_retired_cache_fails_its_siblings_lane_too() {
    let directory = directory();
    let (first, second) = smt_pair(&directory);
    first.close();
    let fates = Fates::default();
    first.push(tracked(1, 3, &fates));
    assert!(matches!(fates.lock().unwrap()[..], [(1, Fate::Failed(_))]));
    assert_eq!(second.local_cache_count(), 0);
}

#[test]
#[should_panic(expected = "already has a sibling")]
fn a_cache_pairs_once() {
    let directory = directory();
    let (first, second) = smt_pair(&directory);
    first.pair_with(second);
}

#[test]
#[should_panic(expected = "its own sibling")]
fn a_cache_cannot_pair_with_itself() {
    let cache = cache(1);
    cache.pair_with(cache.clone());
}

#[test]
fn closing_an_old_cache_leaves_its_replacement_in_place() {
    let directory = directory();
    let old = cache_on(&directory, &config(1), 1);
    let replacement = cache_on(&directory, &config(1), 1);
    old.close();
    assert!(Arc::ptr_eq(&directory.cache_at(2).unwrap(), &replacement));
}

#[test]
fn closing_does_not_wait_for_a_thief_mid_take() {
    let directory = directory();
    let thief = cache_on(&directory, &config(1), 0);
    let victim = cache_on(&directory, &config(1), 1);
    victim.push(unordered(1, 5));
    victim.push(unordered(2, 6));
    let (entered, inside) = mpsc::channel();
    let (release, released) = mpsc::channel::<()>();
    let cpu = victim.cpu();
    let stealing = thread::spawn(move || {
        thief.work_steal(
            |_| {
                entered.send(()).unwrap();
                released.recv().unwrap();
            },
            1,
            cpu,
        )
    });
    inside.recv().unwrap();

    let closer = victim.clone();
    let (closed, done) = mpsc::channel();
    thread::spawn(move || {
        closer.close();
        closed.send(()).unwrap();
    });
    done.recv_timeout(Duration::from_secs(5)).unwrap();

    release.send(()).unwrap();
    assert_eq!(stealing.join().unwrap(), 1);
    assert_eq!(victim.local_cache_count(), 1);
}

#[test]
fn an_attached_cache_takes_the_frames_pushed_into_its_edge() {
    let directory = directory();
    let cache = cache_on(&directory, &config(1), 0);
    let edge = LatticeEdge::new(directory.registry(), Default::default());
    cache.attach(&edge);
    edge.push_downstream(ordered(1));
    assert_eq!(cache.local_cache_count(), 1);
    assert!(cache.edge().parent().is_some());
}

#[test]
fn pull_fills_the_cache_through_the_edges_above() {
    let directory = directory();
    let cache = cache_on(&directory, &config(1), 0);
    let edge = LatticeEdge::new(directory.registry(), Default::default());
    cache.attach(&edge);
    let source = source(&directory, (0..5).map(ordered).collect());

    cache.set_drain_mode(true);
    assert_eq!(cache.pull(3), 0);
    cache.set_drain_mode(false);

    assert_eq!(cache.pull(3), 3);
    assert_eq!(cache.local_cache_count(), 3);
    let mut taken_frames = Vec::new();
    cache.greedy(|frame| taken_frames.push(frame), 10);
    assert_eq!(ids(&taken_frames), [0, 1, 2]);
    assert_eq!(source.frames.lock().unwrap().len(), 2);
    directory.registry().remove_thread();
}

#[test]
fn upstream_pull_bypasses_the_cache_and_stops_at_ordered_frames() {
    let directory = directory();
    let cache = cache_on(&directory, &config(1), 0);
    let _source = source(
        &directory,
        vec![unordered(1, 5), unordered(2, 6), ordered(3)],
    );
    let mut pulled = Vec::new();
    assert_eq!(cache.upstream_pull(|frame| pulled.push(frame), 10), 2);
    assert_eq!(ids(&pulled), [1, 2]);
    assert!(cache.is_drained());
    directory.registry().remove_thread();
}

#[test]
fn request_scales_the_batch_by_the_socket_and_fans_it_out() {
    let directory = directory();
    // Core 0 sits on socket 0, with eight cores: the batch scales by 3.
    let cache = cache_on(&directory, &config(1), 0);
    let source = source(&directory, Vec::new());
    FlowContext::initialize();
    FlowContext::with(|context| {
        let context = context.unwrap();
        context.add_satisfied_request(11);
        context.add_satisfied_pull(13);
        context.set_original_pull(14);
    });

    cache.request(4);

    let (satisfied, original, satisfied_pull, original_pull) = FlowContext::with(|context| {
        let context = context.unwrap();
        (
            context.satisfied_request(),
            context.original_request(),
            context.satisfied_pull(),
            context.original_pull(),
        )
    });
    FlowContext::clear();
    assert_eq!((satisfied, original), (0, 12));
    assert_eq!((satisfied_pull, original_pull), (0, 0));
    // Two sockets.
    assert_eq!(source.requested.load(Ordering::SeqCst), 24);
    directory.registry().remove_thread();
}

#[test]
fn request_is_capped_by_the_cache_capacity() {
    let directory = directory();
    let cache = cache_on(&directory, &config(1), 9);
    let source = source(&directory, Vec::new());
    cache.request(u64::MAX);
    assert_eq!(
        source.requested.load(Ordering::SeqCst),
        2 * cache.max_local_cache_count()
    );
    directory.registry().remove_thread();
}

#[test]
fn a_small_socket_still_doubles_the_batch() {
    let directory = directory();
    // Core 9 sits on socket 1, with two cores: 37.5% rounds down to zero, raised to two.
    let cache = cache_on(&directory, &config(1), 9);
    let source = source(&directory, Vec::new());
    cache.request(5);
    assert_eq!(source.requested.load(Ordering::SeqCst), 20);
    directory.registry().remove_thread();
}

/// A snapshot of `core` where each CPU has the pressure given for it.
fn snapshot(core: usize, pressures: &[(usize, f64)]) -> CoreSnapshot {
    let mut per_cpu = vec![0.0; 20];
    for &(cpu, pressure) in pressures {
        per_cpu[cpu] = pressure;
    }
    let utilization = HardwareUtilization::new(HardwareUtilizationParts {
        quota_cpus: 20.0,
        global_effective_cpus: CpuSet::range(0, 20),
        per_quota_cpu_pressure: per_cpu,
        ..Default::default()
    });
    utilization.core_snapshot(core, &CpuSet::range(2 * core, 2 * core + 2), 2.0)
}

#[test]
fn pressure_lowers_the_cap_quickly_and_relief_raises_it_slowly() {
    let cache = cache(1);
    // The core's busiest CPU counts: the target is 1 - 0.85 * 0.8 = 0.32.
    cache.update(&snapshot(0, &[(0, 0.2), (1, 0.8)]));
    let lowered = 1.0 * 0.8 + 0.32 * 0.2;
    assert!((cache.cap_factor() - lowered).abs() < 1e-12);
    assert_eq!(
        cache.max_local_cache_count(),
        (cache.frame_quota() as f64 * lowered) as u64
    );

    cache.update(&snapshot(0, &[]));
    let raised = lowered * 0.98 + 0.02;
    assert!((cache.cap_factor() - raised).abs() < 1e-12);
}

#[test]
fn pressure_on_other_cores_is_ignored() {
    let cache = cache(1);
    cache.update(&snapshot(0, &[(2, 1.0), (3, 1.0)]));
    assert_eq!(cache.cap_factor(), 1.0);
}

#[test]
fn the_cap_never_falls_below_its_floor() {
    let cache = cache(1);
    for _ in 0..200 {
        cache.update(&snapshot(0, &[(0, 1.0)]));
    }
    assert!(cache.cap_factor() >= MIN_CAP_FACTOR);
    assert!(cache.cap_factor() < MIN_CAP_FACTOR + 1e-9);
}

#[test]
fn reset_drops_every_frame_and_restores_capacity() {
    let cache = cache(2);
    cache.update(&snapshot(0, &[(0, 1.0)]));
    for id in 0..5 {
        cache.push(ordered(id));
    }
    assert_eq!(cache.reset_for_next_trial(), 5);
    assert!(cache.is_drained());
    assert_eq!(cache.cap_factor(), 1.0);
    assert_eq!(cache.greedy(|_| panic!("cleared"), 10), 0);
    assert_eq!(cache.reset_for_next_trial(), 0);
}

#[test]
fn errors_from_upstream_stop_at_the_cache() {
    let cache = cache(1);
    LatticeNode::on_error(&*cache, "upstream failed".into());
    LatticeNode::close(&*cache);
    assert!(LatticeNode::is_closed(&*cache));
}

#[test]
fn a_count_that_dips_below_zero_reads_as_empty() {
    let cache = cache(1);
    cache.count.store(-1, Ordering::SeqCst);
    assert_eq!(cache.local_cache_count(), 0);
    assert!(cache.is_drained());
}

#[test]
fn pressure_is_rounded_to_four_places() {
    let cache = cache(1);
    cache.update(&snapshot(0, &[(0, 0.123_449)]));
    let target = 1.0 - 0.85 * 0.1234;
    assert!((cache.cap_factor() - (0.8 + 0.2 * target)).abs() < 1e-12);
}

#[test]
fn frames_queued_after_a_reset_are_the_only_ones_left() {
    let cache = cache(1);
    for id in 0..3 {
        cache.push(ordered(id));
    }
    cache.reset_for_next_trial();
    cache.push(ordered(9));
    let mut taken_frames = Vec::new();
    assert_eq!(cache.greedy(|frame| taken_frames.push(frame), 10), 1);
    assert_eq!(ids(&taken_frames), [9]);
}

#[test]
fn chunks_count_the_l1_with_the_l2() {
    let layout = CpuCacheLayout {
        bytes_l1: 64 * 1024,
        bytes_l2: 64 * 1024,
        mask_l1: CpuSet::range(0, 1),
        mask_l2: CpuSet::range(0, 1),
        ..topology().cache_layout(0).unwrap().clone()
    };
    let whole = config(1).with_memory_budget(1.0).unwrap();
    // 128 KiB in one partition: 8,192 boxed frames.
    assert_eq!(chunk_size(&whole, Some(&layout), false), 8192);
}

#[test]
fn first_touch_writes_the_chunks_then_resets_the_cache() {
    let directory = directory();
    let config = config(2).with_max_pooled_chunks(2);
    let cache = cache_on(&directory, &config, 0);
    cache.update(&snapshot(0, &[(0, 1.0)]));

    cache.first_touch();

    // Three chunks' worth of frames retires at least two chunks per partition; the pool keeps two.
    assert_eq!(cache.pooled_chunks(), 4);
    assert!(cache.is_drained());
    assert_eq!(cache.cap_factor(), 1.0);
    assert_eq!(
        cache.greedy(|_| panic!("the dummies are cleared"), u64::MAX),
        0
    );
    cache.push(ordered(1));
    let mut taken_frames = Vec::new();
    cache.greedy(|frame| taken_frames.push(frame), 10);
    assert_eq!(ids(&taken_frames), [1]);
}

#[test]
fn first_touch_runs_once() {
    let directory = directory();
    let cache = cache_on(&directory, &config(1).with_max_pooled_chunks(4), 0);
    cache.first_touch();
    cache.update(&snapshot(0, &[(0, 1.0)]));
    cache.first_touch();
    // A chunk spends a slot on its link, so three chunks' worth of frames retired three chunks.
    assert_eq!(cache.pooled_chunks(), 3);
    assert!(cache.cap_factor() < 1.0);
}

fn metered(registry: &Arc<crate::metrics::MetricsRegistry>) -> CacheConfig {
    config(2)
        .with_metric_prefix("test")
        .with_registry(registry.clone())
}

#[test]
fn a_cache_without_a_registry_registers_and_records_nothing() {
    let cache = cache(1);
    assert!(cache.metrics.is_none());
    cache.update(&snapshot(0, &[(0, 1.0)]));
    let mut meters = Vec::new();
    MeterSource::collect(&*cache, &mut meters);
    assert!(meters.is_empty());
}

#[test]
fn a_metered_cache_reports_its_backlog_and_cap_factor_tagged_with_its_core() {
    use crate::metrics::{CAP_FACTOR_SUFFIX, MetricsRegistry, metric_name};

    let registry = MetricsRegistry::new();
    let directory = directory();
    let cache = cache_on(&directory, &metered(&registry), 3);
    assert_eq!(registry.len(), 1);

    for id in 0..5 {
        cache.push(unordered(id, id * 2 + 1));
    }
    cache.update(&snapshot(3, &[(6, 1.0)]));
    cache.update(&snapshot(3, &[(6, 0.0)]));

    let snapshot = registry.snapshot();
    let backlog = snapshot.named("test.cache.backlog").collect::<Vec<_>>();
    assert_eq!(backlog.len(), 1);
    assert_eq!((backlog[0].core, backlog[0].cpu), (3, 6));
    assert_eq!(backlog[0].tag(crate::metrics::CORE_TAG), Some(3));
    assert_eq!(backlog[0].gauge(), Some(5.0));
    let cap = snapshot
        .get(&metric_name("test", CAP_FACTOR_SUFFIX), 6)
        .unwrap()
        .summary()
        .unwrap();
    assert_eq!(cap.count, 2);
    // Each update records the cap factor it set.
    let lowered = 0.8 + 0.15 * 0.2;
    let raised = lowered * 0.98 + 0.02;
    assert!((cap.sum - (lowered + raised)).abs() < 1e-12);
    assert_eq!(cap.max, raised);
    assert_eq!(snapshot.cap_factor_summaries().count(), 1);
    assert_eq!(snapshot.cache_backlog_gauges().count(), 1);

    let mut taken = 0;
    cache.greedy(|_| taken += 1, u64::MAX);
    assert_eq!(taken, 5);
    let backlog = registry.snapshot();
    assert_eq!(
        backlog.named("test.cache.backlog").next().unwrap().gauge(),
        Some(0.0)
    );
}

#[test]
fn closing_or_dropping_a_cache_takes_its_meters_out_of_the_registry() {
    let registry = crate::metrics::MetricsRegistry::new();
    let directory = directory();
    let closed = cache_on(&directory, &metered(&registry), 0);
    let dropped = cache_on(&directory, &metered(&registry), 1);
    assert_eq!(registry.len(), 2);
    closed.close();
    assert_eq!(registry.len(), 1);
    assert!(
        registry
            .snapshot()
            .meters()
            .iter()
            .all(|meter| meter.core == 1)
    );
    drop(dropped);
    assert!(registry.is_empty());
    assert!(registry.snapshot().meters().is_empty());
}
