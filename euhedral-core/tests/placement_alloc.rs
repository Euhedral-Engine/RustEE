//! Placing a frame allocates nothing. This binary counts every allocation of the process, so it holds one test and runs
//! alone.
#![cfg(not(loom))]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use euhedral_core::control_plane::PlacementPicker;
use euhedral_core::flow_control::{Placement, UpstreamRegistry};
use euhedral_core::frames::{BenchmarkFrame, Frame};
use euhedral_hardware_utils::topology::Topology;
use euhedral_hardware_utils::util::CpuSet;

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

const ROUTES: u64 = 100_000;

#[test]
fn placing_a_frame_allocates_nothing() {
    // The counter sees allocations at all.
    let probe = ALLOCATIONS.load(Ordering::Relaxed);
    drop(std::hint::black_box(Box::new(7u64)));
    assert!(
        ALLOCATIONS.load(Ordering::Relaxed) > probe,
        "the counting allocator is live"
    );

    let topology = Topology::system().clone();
    let all = topology.cpu_set().clone();
    let first = all.first().expect("the host has a CPU");
    let a: CpuSet = [first].into_iter().collect();
    let picker = PlacementPicker::new(
        topology,
        |frame: &dyn Frame| frame.header().id_hash().is_multiple_of(2),
        a,
        all.clone(),
    );

    // The picker on its own.
    let mut frames: Vec<Box<dyn Frame>> = (0..1024)
        .map(|id| {
            let mut frame = BenchmarkFrame::new(id, None);
            frame.header_mut().randomize_hash(id);
            Box::new(frame) as Box<dyn Frame>
        })
        .collect();
    let before = ALLOCATIONS.load(Ordering::Relaxed);
    for route in 0..ROUTES {
        picker.pick(&mut *frames[(route % 1024) as usize], &all);
    }
    assert_eq!(
        ALLOCATIONS.load(Ordering::Relaxed) - before,
        0,
        "the picker allocates nothing"
    );

    // And through a vertex's placement, as a frame meets it: a policy read from the thread's view.
    let registry = UpstreamRegistry::new();
    let placement = Placement::new(&registry);
    placement.publish_cpus(&registry, all.clone());
    placement.set_policy(&registry, picker.into_policy());
    registry.register_reader();
    placement.apply(&registry, &mut *frames[0]);
    let before = ALLOCATIONS.load(Ordering::Relaxed);
    for route in 0..ROUTES {
        placement.apply(&registry, &mut *frames[(route % 1024) as usize]);
    }
    assert_eq!(
        ALLOCATIONS.load(Ordering::Relaxed) - before,
        0,
        "applying the policy from a registered reader allocates nothing"
    );
    registry.remove_reader();
    // A thread that is not registered reads through a light view: no allocation either once it is one.
    placement.apply(&registry, &mut *frames[0]);
    let before = ALLOCATIONS.load(Ordering::Relaxed);
    for route in 0..ROUTES {
        placement.apply(&registry, &mut *frames[(route % 1024) as usize]);
    }
    assert_eq!(
        ALLOCATIONS.load(Ordering::Relaxed) - before,
        0,
        "applying the policy from a light reader allocates nothing"
    );
}
