//! The reaper frees the retired prefix of its queue in order, keeps an unretired head at the front without
//! allocating, and never waits.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use euhedral_memory::Reaper;

thread_local! {
    static ALLOCS: Cell<usize> = const { Cell::new(0) };
}

struct Counting;

// SAFETY: forwards to the system allocator; the counter is a plain thread-local cell.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let _ = ALLOCS.try_with(|a| a.set(a.get() + 1));
        // SAFETY: the caller's contract.
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: the caller's contract.
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let _ = ALLOCS.try_with(|a| a.set(a.get() + 1));
        // SAFETY: the caller's contract.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static COUNTING: Counting = Counting;

fn allocations() -> usize {
    ALLOCS.with(Cell::get)
}

/// Records the order its items were freed in: the item's number goes into the next slot.
struct Order {
    next: AtomicUsize,
    slots: Vec<AtomicUsize>,
}

impl Order {
    fn new(n: usize) -> Arc<Self> {
        Arc::new(Order {
            next: AtomicUsize::new(0),
            slots: (0..n).map(|_| AtomicUsize::new(usize::MAX)).collect(),
        })
    }

    fn freed(&self) -> Vec<usize> {
        let n = self.next.load(Ordering::SeqCst);
        self.slots[..n]
            .iter()
            .map(|s| s.load(Ordering::SeqCst))
            .collect()
    }
}

struct Item(usize, Arc<Order>);

impl Drop for Item {
    fn drop(&mut self) {
        let at = self.1.next.fetch_add(1, Ordering::SeqCst);
        self.1.slots[at].store(self.0, Ordering::SeqCst);
    }
}

fn gate() -> Arc<AtomicBool> {
    Arc::new(AtomicBool::new(false))
}

fn open(g: &Arc<AtomicBool>) {
    g.store(true, Ordering::SeqCst);
}

fn push_gated(r: &Reaper<Item>, order: &Arc<Order>, n: usize, g: &Arc<AtomicBool>) {
    let g = g.clone();
    r.push(move || g.load(Ordering::SeqCst), Item(n, order.clone()));
}

#[test]
fn reap_frees_the_retired_prefix_and_allocates_nothing() {
    let order = Order::new(8);
    let r = Reaper::new();
    let held = gate();
    for n in 0..5 {
        r.push(|| true, Item(n, order.clone()));
    }
    push_gated(&r, &order, 5, &held);
    r.push(|| true, Item(6, order.clone()));
    let before = allocations();
    assert_eq!(r.reap(), 5, "the retired prefix, up to the unretired item");
    assert_eq!(
        allocations(),
        before,
        "a reap that stops at an unretired item allocates nothing"
    );
    assert_eq!(order.freed(), vec![0, 1, 2, 3, 4], "freed in push order");
    assert_eq!(r.pending(), 2);
}

#[test]
fn an_unretired_head_is_kept_at_the_front() {
    let order = Order::new(8);
    let r = Reaper::new();
    let head = gate();
    push_gated(&r, &order, 0, &head);
    r.push(|| true, Item(1, order.clone()));
    r.push(|| true, Item(2, order.clone()));
    assert_eq!(r.reap(), 0, "nothing behind an unretired head is freed");
    assert_eq!(r.pending(), 3);
    assert!(order.freed().is_empty());
    open(&head);
    assert_eq!(r.reap(), 3);
    assert_eq!(order.freed(), vec![0, 1, 2], "the head stayed at the front");
    assert_eq!(r.pending(), 0);
}

#[test]
fn a_reap_from_inside_a_reap_returns_at_once() {
    // The inner reap runs from inside the outer one (a free, or a retirement check, that reaps): the claim is
    // held, so it returns at once with nothing freed.
    let order = Order::new(4);
    let inner_result = Arc::new(AtomicUsize::new(usize::MAX));
    let r = Arc::new(Reaper::new());
    let weak = Arc::downgrade(&r);
    let seen = inner_result.clone();
    r.push(
        move || {
            let again = weak.upgrade().expect("alive").reap();
            seen.store(again, Ordering::SeqCst);
            true
        },
        Item(0, order.clone()),
    );
    r.push(|| true, Item(1, order.clone()));
    assert_eq!(r.reap(), 2);
    assert_eq!(inner_result.load(Ordering::SeqCst), 0);
    assert_eq!(order.freed(), vec![0, 1]);
    assert_eq!(r.reap_all(), 0);
}

#[test]
fn a_panicking_free_releases_the_claim() {
    struct Bomb;
    impl Drop for Bomb {
        fn drop(&mut self) {
            panic!("a free that panics");
        }
    }
    let order = Order::new(4);
    let r: Reaper<Box<dyn Send>> = Reaper::new();
    r.push(|| true, Box::new(Bomb));
    r.push(|| true, Box::new(Item(1, order.clone())));
    let first = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| r.reap()));
    assert!(first.is_err());
    assert_eq!(r.reap(), 1, "the claim was released; the rest is freed");
    assert_eq!(order.freed(), vec![1]);
}
