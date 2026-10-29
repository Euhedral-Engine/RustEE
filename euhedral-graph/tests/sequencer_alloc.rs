//! A sequencer of pooled entries allocates nothing after warm-up. This binary counts every allocation of the process,
//! so it holds one test and runs alone.
#![cfg(not(loom))]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use euhedral_graph::{Entry, Progress, Sequencer};

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

/// The caller's own entry: made once, offered again and again.
struct Pooled {
    progress: Progress,
}

impl Entry for Pooled {
    fn ready(&self) -> bool {
        self.progress.ready()
    }
}

const DEPTH: usize = 8;
const ROUNDS: usize = 20_000;

#[test]
fn a_sequencer_of_pooled_entries_allocates_nothing_after_warm_up() {
    // The counter sees allocations at all.
    let probe = ALLOCATIONS.load(Ordering::Relaxed);
    drop(std::hint::black_box(Box::new(7u64)));
    assert!(
        ALLOCATIONS.load(Ordering::Relaxed) > probe,
        "the counting allocator is live"
    );

    let concluded = Arc::new(AtomicUsize::new(0));
    let counted = concluded.clone();
    let sequencer = Sequencer::new(move |_: &Arc<Pooled>| {
        counted.fetch_add(1, Ordering::Relaxed);
    });
    // The pool: entries the caller made once. A round offers them all, completes them newest first (the head is the
    // last to finish), and takes them back.
    let pool: Vec<Arc<Pooled>> = (0..DEPTH)
        .map(|_| {
            Arc::new(Pooled {
                progress: Progress::new(),
            })
        })
        .collect();
    let round = |pool: &[Arc<Pooled>]| {
        for entry in pool {
            sequencer.offer(entry.clone());
        }
        for entry in pool.iter().rev() {
            entry.progress.finish();
            sequencer.drain();
        }
        assert!(sequencer.is_empty());
        for entry in pool {
            entry.progress.reset();
        }
    };

    // Warm-up: the order's chunks are made and pooled.
    for _ in 0..1_000 {
        round(&pool);
    }
    let before = ALLOCATIONS.load(Ordering::Relaxed);
    for _ in 0..ROUNDS {
        round(&pool);
    }
    let allocations = ALLOCATIONS.load(Ordering::Relaxed) - before;
    assert_eq!(
        concluded.load(Ordering::Relaxed),
        (1_000 + ROUNDS) * DEPTH,
        "every entry concluded once per round"
    );
    assert_eq!(
        allocations, 0,
        "offers, completions and drains allocate nothing after warm-up"
    );
}
