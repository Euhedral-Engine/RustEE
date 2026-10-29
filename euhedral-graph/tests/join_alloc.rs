//! A re-armed `Join` allocates nothing. This binary counts every allocation of the process, so it holds one
//! test and runs alone.
#![cfg(not(loom))]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use euhedral_graph::Join;

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

const PARTS: usize = 4;
const ROUNDS: usize = 10_000;

#[test]
fn a_join_rearmed_10_000_times_allocates_nothing_after_construction() {
    // The counter sees allocations at all.
    let probe = ALLOCATIONS.load(Ordering::Relaxed);
    drop(std::hint::black_box(Box::new(7u64)));
    assert!(
        ALLOCATIONS.load(Ordering::Relaxed) > probe,
        "the counting allocator is live"
    );

    let rounds = Arc::new(AtomicUsize::new(0));
    let sum = Arc::new(AtomicUsize::new(0));
    let (counted, summed) = (rounds.clone(), sum.clone());
    let join = Join::<usize>::with_capacity(PARTS, move |mut round| {
        let round_no = counted.fetch_add(1, Ordering::AcqRel);
        let total: usize = (0..round.len()).filter_map(|part| round.take(part)).sum();
        summed.fetch_add(total, Ordering::AcqRel);
        if round_no + 1 < ROUNDS {
            round.rearm(PARTS);
        }
    });
    join.reset(PARTS);

    let before = ALLOCATIONS.load(Ordering::Relaxed);
    for round in 0..ROUNDS {
        for part in 0..PARTS {
            // The last arrival of the round runs the continuation, which arms the next.
            join.arrive(part, round + part);
        }
    }
    let after = ALLOCATIONS.load(Ordering::Relaxed);

    assert_eq!(rounds.load(Ordering::Acquire), ROUNDS);
    let expected: usize = (0..ROUNDS)
        .map(|r| (0..PARTS).map(|p| r + p).sum::<usize>())
        .sum();
    assert_eq!(sum.load(Ordering::Acquire), expected);
    assert_eq!(
        after - before,
        0,
        "{} allocations in {ROUNDS} rounds",
        after - before
    );

    // The explicit form, for a unit taken from a pool and re-armed from outside, allocates nothing either.
    let before = ALLOCATIONS.load(Ordering::Relaxed);
    let join = Join::<usize>::with_capacity(PARTS, |_| {});
    let built = ALLOCATIONS.load(Ordering::Relaxed);
    assert!(built > before, "construction allocates the slots");
    for round in 0..ROUNDS {
        join.reset(1 + round % PARTS);
        for part in 0..1 + round % PARTS {
            join.arrive(part, part);
        }
    }
    assert_eq!(ALLOCATIONS.load(Ordering::Relaxed), built);
}
