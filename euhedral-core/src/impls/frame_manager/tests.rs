use super::*;
use crate::flow_control::RoutingPolicy;
use crate::frames::{FrameError, FrameHeader};
use euhedral_hardware_utils::topology::CpuInfo;
use std::collections::HashSet;
use std::sync::Barrier;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread;

struct Pooled {
    header: FrameHeader,
    payload: u64,
    attachment: Option<u64>,
    recycler: Option<Recycler<Pooled>>,
    _drops: Option<DropCounter>,
}

impl Pooled {
    fn new(id_hash: u64, payload: u64) -> Self {
        Self {
            header: FrameHeader::new(id_hash),
            payload,
            attachment: None,
            recycler: None,
            _drops: None,
        }
    }

    fn managed(id_hash: u64, payload: u64, manager: &FrameManager<u64, Pooled>) -> Self {
        Self {
            recycler: Some(manager.recycler()),
            ..Self::new(id_hash, payload)
        }
    }
}

impl Frame for Pooled {
    fn header(&self) -> &FrameHeader {
        &self.header
    }

    fn header_mut(&mut self) -> &mut FrameHeader {
        &mut self.header
    }

    fn recycle(self: Box<Self>) {
        Recycler::recycle_owned(self, |frame| frame.recycler.as_ref());
    }
}

/// Counts the drops of the frame that owns it.
struct DropCounter(Arc<AtomicUsize>);

impl Drop for DropCounter {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }
}

fn factory() -> FrameFactory<u64, Pooled> {
    FrameFactory::new(Pooled::new, |payload, frame: &mut Pooled| {
        frame.payload = payload;
    })
}

fn address(frame: &Pooled) -> usize {
    frame as *const Pooled as usize
}

#[test]
fn a_default_pool_holds_8192_frames() {
    let manager = FrameManager::<u64, Pooled>::new();
    assert_eq!(manager.capacity(), DEFAULT_CAPACITY);
    assert_eq!(manager.total_recycled(), 0);
    assert_eq!(manager.recycle_queue_size(), 0);
    assert!(manager.factory().is_none());
    assert_eq!(FrameManager::<u64, Pooled>::default().capacity(), 8_192);
}

#[test]
fn capacity_rounds_up_to_a_power_of_two() {
    for (requested, actual) in [(0, 1), (1, 1), (2, 2), (3, 4), (8, 8), (1000, 1024)] {
        assert_eq!(
            FrameManager::<u64, Pooled>::with_capacity(requested).capacity(),
            actual,
            "capacity {requested}"
        );
    }
}

#[test]
#[should_panic(expected = "overflows")]
fn a_capacity_past_the_largest_power_of_two_panics() {
    FrameManager::<u64, Pooled>::with_capacity(usize::MAX / 2 + 2);
}

#[test]
fn a_pool_takes_one_factory_for_life() {
    let mut manager = FrameManager::<u64, Pooled>::new();
    let first = factory();
    let first_id = first.id_hash();
    assert!(manager.set_factory(first).is_ok());
    let rejected = manager.set_factory(factory()).unwrap_err();
    assert_ne!(rejected.id_hash(), first_id);
    assert_eq!(manager.factory().unwrap().id_hash(), first_id);
}

#[test]
fn recycle_accepts_frames_until_the_queue_is_full() {
    let manager = FrameManager::<u64, Pooled>::with_capacity(4);
    for i in 0..4 {
        assert!(manager.recycle(Box::new(Pooled::new(i, i))));
    }
    assert!(!manager.recycle(Box::new(Pooled::new(4, 4))));
    assert_eq!(manager.recycle_queue_size(), 4);
}

#[test]
fn close_discards_queued_and_buffered_frames() {
    let mut manager = FrameManager::<u64, Pooled>::with_capacity(4);
    for i in 0..4 {
        manager.recycle(Box::new(Pooled::new(i, i)));
    }
    assert!(manager.get().is_some());
    assert_eq!(manager.buffered(), 3);
    for i in 0..2 {
        manager.recycle(Box::new(Pooled::new(i, i)));
    }

    manager.close();
    assert_eq!(manager.buffered(), 0);
    assert_eq!(manager.recycle_queue_size(), 0);
    assert!(manager.get().is_none());
    assert!(
        manager.recycle(Box::new(Pooled::new(9, 9))),
        "a closed pool is still usable"
    );
}

#[test]
fn get_on_an_empty_pool_is_none() {
    assert!(FrameManager::<u64, Pooled>::new().get().is_none());
}

#[test]
fn get_moves_the_queue_into_the_buffer_and_counts_it() {
    let mut manager = FrameManager::<u64, Pooled>::with_capacity(8);
    for i in 0..3 {
        manager.recycle(Box::new(Pooled::new(i, i)));
    }

    let mut payloads = vec![manager.get().unwrap().payload];
    assert_eq!(manager.total_recycled(), 3);
    assert_eq!(manager.recycle_queue_size(), 0);
    assert_eq!(manager.buffered(), 2);
    payloads.extend(std::iter::from_fn(|| {
        manager.get().map(|frame| frame.payload)
    }));
    payloads.sort_unstable();
    assert_eq!(payloads, [0, 1, 2]);
    assert_eq!(manager.total_recycled(), 3);
}

#[test]
#[should_panic(expected = "needs a FrameFactory")]
fn get_or_create_without_a_factory_panics() {
    FrameManager::<u64, Pooled>::new().get_or_create(1);
}

#[test]
fn get_or_create_makes_a_frame_when_the_pool_is_empty() {
    let mut manager = FrameManager::new();
    manager.set_factory(factory()).unwrap();
    let frame = manager.get_or_create(7);
    assert_eq!(frame.payload, 7);
    assert_eq!(
        frame.header().id_hash(),
        manager.factory().unwrap().id_hash()
    );
    assert_eq!(manager.total_recycled(), 0);
}

#[test]
fn get_or_create_refills_a_returned_frame() {
    let mut manager = FrameManager::new();
    manager.set_factory(factory()).unwrap();
    let frame = manager.get_or_create(1);
    let first = address(&frame);
    manager.recycle(frame);

    let reused = manager.get_or_create(2);
    assert_eq!(address(&reused), first);
    assert_eq!(reused.payload, 2);
    assert_eq!(manager.total_recycled(), 1);
}

#[test]
fn a_raw_checkout_keeps_the_last_run_and_a_refill_resets_it() {
    let kill_switch = Arc::new(AtomicBool::new(false));
    let mut manager = FrameManager::<u64, Pooled>::with_capacity(8);
    let recycler = manager.recycler();
    let switch = kill_switch.clone();
    manager
        .set_factory(FrameFactory::new(
            move |id, payload| Pooled {
                header: FrameHeader::with_kill_switch(id, Some(switch.clone())),
                recycler: Some(recycler.clone()),
                ..Pooled::new(id, payload)
            },
            |payload, frame: &mut Pooled| {
                assert_eq!(frame.payload, 1);
                assert_eq!(frame.attachment, Some(11));
                assert_eq!(frame.header().routing_policy(), RoutingPolicy::CacheLocal);
                assert!(
                    frame.header().is_ordered(),
                    "the factory resets the hash first"
                );
                frame.payload = payload;
                frame.attachment = None;
                frame
                    .header_mut()
                    .set_routing_policy(RoutingPolicy::Anywhere);
            },
        ))
        .unwrap();

    let mut frame = manager.get_or_create(1);
    let factory_origin = frame.header().origin();
    let elsewhere = CpuInfo {
        cpu: 99,
        core: 98,
        socket: 97,
    };
    frame.header_mut().set_origin(Some(elsewhere));
    frame.attachment = Some(11);
    frame
        .header_mut()
        .set_routing_policy(RoutingPolicy::CacheLocal);
    frame.header_mut().randomize_hash(17);
    let last_hash = frame.header().routing_hash();
    frame.kill();
    let first = address(&frame);
    frame.do_finally();

    let raw = manager.get().unwrap();
    assert_eq!(address(&raw), first);
    assert_eq!(raw.payload, 1);
    assert_eq!(raw.attachment, Some(11));
    assert_eq!(raw.header().origin(), Some(elsewhere));
    assert_eq!(raw.header().routing_hash(), last_hash);
    assert!(!raw.is_alive());
    assert!(manager.get().is_none());
    raw.do_finally();

    let refilled = manager.get_or_create(2);
    assert_eq!(address(&refilled), first);
    assert_eq!(refilled.payload, 2);
    assert_eq!(refilled.attachment, None);
    assert_eq!(refilled.header().origin(), factory_origin);
    assert_eq!(refilled.header().routing_policy(), RoutingPolicy::Anywhere);
    assert!(
        !refilled.is_alive(),
        "a refill does not revive a killed generation"
    );
}

#[test]
fn dump_drops_buffered_frames_then_queued_ones() {
    let mut manager = FrameManager::<u64, Pooled>::with_capacity(8);
    for i in 0..4 {
        manager.recycle(Box::new(Pooled::new(i, i)));
    }
    manager.get();
    assert_eq!(manager.buffered(), 3);
    for i in 0..4 {
        manager.recycle(Box::new(Pooled::new(i, i)));
    }

    assert_eq!(manager.dump(0), 0);
    assert_eq!(manager.dump(2), 2);
    assert_eq!(manager.buffered(), 1);
    assert_eq!(manager.recycle_queue_size(), 4);
    assert_eq!(manager.dump(3), 3);
    assert_eq!(manager.buffered(), 0);
    assert_eq!(manager.recycle_queue_size(), 2);
    assert_eq!(manager.dump(u64::MAX), 2);
    assert_eq!(manager.dump(u64::MAX), 0);
}

#[test]
fn copy_empty_keeps_the_capacity_only() {
    let mut manager = FrameManager::<u64, Pooled>::with_capacity(16);
    manager.set_factory(factory()).unwrap();
    manager.recycle(Box::new(Pooled::new(1, 1)));

    let copy = manager.copy_empty();
    assert_eq!(copy.capacity(), 16);
    assert!(copy.factory().is_none());
    assert_eq!(copy.recycle_queue_size(), 0);
    assert!(!manager.recycler().same_pool(&copy.recycler()));
}

#[test]
fn recyclers_name_their_pool() {
    let manager = FrameManager::<u64, Pooled>::new();
    let other = FrameManager::<u64, Pooled>::new();
    let recycler = manager.recycler();
    assert!(recycler.belongs_to(&manager));
    assert!(!recycler.belongs_to(&other));
    assert!(recycler.same_pool(&recycler.clone()));
    assert!(!recycler.same_pool(&other.recycler()));
}

#[test]
fn a_frame_returns_itself_through_its_recycler() {
    let mut manager = FrameManager::new();
    Box::new(Pooled::managed(1, 1, &manager)).do_finally();
    Box::new(Pooled::managed(2, 2, &manager)).do_finally_with_error(FrameError::Cancelled);
    Box::new(Pooled::managed(3, 3, &manager)).recycle();
    assert_eq!(manager.recycle_queue_size(), 3);
    assert_eq!(manager.dump(u64::MAX), 3);
}

#[test]
fn frames_returned_after_the_pool_is_gone_are_dropped() {
    let drops = Arc::new(AtomicUsize::new(0));
    let manager = FrameManager::new();
    let recycler = manager.recycler();
    let frame = Box::new(Pooled {
        _drops: Some(DropCounter(drops.clone())),
        ..Pooled::managed(1, 1, &manager)
    });
    drop(manager);

    assert!(!recycler.recycle(Box::new(Pooled::new(2, 2))));
    frame.do_finally();
    assert_eq!(drops.load(Ordering::Relaxed), 1);
}

#[test]
fn dropping_the_pool_drops_the_frames_it_holds() {
    let drops = Arc::new(AtomicUsize::new(0));
    let mut manager = FrameManager::with_capacity(8);
    for i in 0..5 {
        Box::new(Pooled {
            _drops: Some(DropCounter(drops.clone())),
            ..Pooled::managed(i, i, &manager)
        })
        .do_finally();
    }
    manager.get();
    assert_eq!(manager.buffered(), 4);

    drop(manager);
    assert_eq!(
        drops.load(Ordering::Relaxed),
        5,
        "queued frames hold only a weak reference to their pool"
    );
}

#[test]
fn concurrent_returns_reach_the_single_owner_exactly_once() {
    const THREADS: u64 = 4;
    const PER_THREAD: u64 = 1_000;
    let mut manager = FrameManager::<u64, Pooled>::new();
    manager.set_factory(factory()).unwrap();
    let start = Arc::new(Barrier::new(THREADS as usize + 1));

    let producers: Vec<_> = (0..THREADS)
        .map(|t| {
            let recycler = manager.recycler();
            let start = start.clone();
            thread::spawn(move || {
                start.wait();
                for i in 0..PER_THREAD {
                    let id = t * PER_THREAD + i;
                    assert!(recycler.recycle(Box::new(Pooled::new(id, id))));
                }
            })
        })
        .collect();

    start.wait();
    let mut seen = HashSet::new();
    let total = THREADS * PER_THREAD;
    while (seen.len() as u64) < total {
        if let Some(frame) = manager.get() {
            assert!(
                seen.insert(frame.payload),
                "frame {} came back twice",
                frame.payload
            );
        } else if producers.iter().all(|p| p.is_finished()) && manager.recycle_queue_size() == 0 {
            break;
        }
    }
    for producer in producers {
        producer.join().unwrap();
    }
    seen.extend(std::iter::from_fn(|| {
        manager.get().map(|frame| frame.payload)
    }));

    assert_eq!(seen.len() as u64, total);
    assert_eq!(manager.total_recycled(), total);
    let fresh = manager.get_or_create(u64::MAX);
    assert_eq!(
        fresh.payload,
        u64::MAX,
        "an empty pool falls back to the factory"
    );
}
