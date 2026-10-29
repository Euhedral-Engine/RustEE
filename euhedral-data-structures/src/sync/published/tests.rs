use super::*;
use std::sync::atomic::{AtomicU64 as StdAtomicU64, AtomicUsize, Ordering as StdOrdering};
use std::thread;
use std::time::Duration;

#[cfg(not(miri))]
const ROUNDS: usize = 20_000;
#[cfg(miri)]
const ROUNDS: usize = 8;

/// A value that counts its drops and poisons itself when dropped.
struct Counted {
    drops: Arc<AtomicUsize>,
    magic: StdAtomicU64,
}

const LIVE: u64 = 0x11FE;
const DEAD: u64 = 0xDEAD;

impl Counted {
    fn new(drops: &Arc<AtomicUsize>) -> Arc<Self> {
        Arc::new(Self {
            drops: drops.clone(),
            magic: StdAtomicU64::new(LIVE),
        })
    }
}

impl Drop for Counted {
    fn drop(&mut self) {
        self.magic.store(DEAD, StdOrdering::SeqCst);
        self.drops.fetch_add(1, StdOrdering::SeqCst);
    }
}

fn drops() -> Arc<AtomicUsize> {
    Arc::new(AtomicUsize::new(0))
}

fn soon() -> Instant {
    Instant::now() + Duration::from_millis(20)
}

#[test]
fn a_value_is_never_dropped_while_a_reader_holds_it() {
    let domain = EpochDomain::new(2);
    let count = drops();
    let cell = Published::new(domain.clone(), Some(Counted::new(&count)));
    let reader = domain.register();
    let guard = reader.pin();
    let held = cell.load(&guard).expect("a value is published");

    let retired = cell.store(Some(Counted::new(&count)));
    let retired = retired
        .wait(soon())
        .expect_err("the reader has not passed the epoch");
    assert_eq!(count.load(StdOrdering::SeqCst), 0);
    assert_eq!(held.magic.load(StdOrdering::SeqCst), LIVE);

    drop(guard);
    retired
        .wait(soon())
        .unwrap_or_else(|_| panic!("the reader went idle"));
    assert_eq!(count.load(StdOrdering::SeqCst), 1);
}

#[test]
fn a_reader_pinned_before_the_store_keeps_the_old_value_until_it_is_idle() {
    let domain = EpochDomain::new(2);
    let count = drops();
    let cell = Published::new(domain.clone(), Some(Counted::new(&count)));
    let reader = domain.register();
    reader.quiescent();
    let old = cell.load(&reader.pin()).unwrap() as *const Counted;

    cell.store(Some(Counted::new(&count))).defer();
    cell.try_reclaim();
    assert_eq!(
        count.load(StdOrdering::SeqCst),
        0,
        "the reader is mid-cycle"
    );
    // SAFETY: the reader's announcement predates the store, so the old value is still allocated.
    assert_eq!(unsafe { &*old }.magic.load(StdOrdering::SeqCst), LIVE);

    reader.idle();
    cell.try_reclaim();
    assert_eq!(count.load(StdOrdering::SeqCst), 1);
}

#[test]
fn wait_never_returns_before_every_slot_passed_the_epoch() {
    let domain = EpochDomain::new(2);
    let count = drops();
    let cell = Arc::new(Published::new(domain.clone(), Some(Counted::new(&count))));
    let reader = domain.register();
    reader.quiescent();

    let stored = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let waiter = {
        let (cell, count, stored) = (cell.clone(), count.clone(), stored.clone());
        thread::spawn(move || {
            let retired = cell.store(Some(Counted::new(&count)));
            stored.store(true, StdOrdering::SeqCst);
            retired
                .wait(Instant::now() + Duration::from_secs(3600))
                .unwrap_or_else(|_| panic!("the reader announced a later epoch"));
            count.load(StdOrdering::SeqCst)
        })
    };
    // The waiter cannot finish: the reader's announcement is older than its store.
    while !stored.load(StdOrdering::SeqCst) {
        thread::yield_now();
    }
    for _ in 0..if cfg!(miri) { 30 } else { 500 } {
        if cfg!(miri) {
            // Miri's clock moves with the steps it takes, which a spinning waiter slows: yield instead.
            thread::yield_now();
        } else {
            thread::sleep(Duration::from_micros(100));
        }
    }
    assert!(!waiter.is_finished());
    assert_eq!(count.load(StdOrdering::SeqCst), 0);

    reader.quiescent();
    assert_eq!(waiter.join().unwrap(), 1, "freed once the reader passed");
}

#[test]
fn a_store_never_waits_and_the_retirements_chain_until_the_readers_pass() {
    let domain = EpochDomain::new(2);
    let count = drops();
    let cell = Published::new(domain.clone(), Some(Counted::new(&count)));
    let reader = domain.register();
    let guard = reader.pin();

    for _ in 0..50 {
        cell.store(Some(Counted::new(&count))).defer();
    }
    assert_eq!(count.load(StdOrdering::SeqCst), 0);
    assert_eq!(cell.pending(), 50);

    drop(guard);
    cell.try_reclaim();
    assert_eq!(count.load(StdOrdering::SeqCst), 50);
    assert_eq!(cell.pending(), 0);
}

#[test]
fn an_idle_slot_holds_nothing_back() {
    let domain = EpochDomain::new(4);
    let count = drops();
    let cell = Published::new(domain.clone(), Some(Counted::new(&count)));
    let slots: Vec<_> = (0..4).map(|_| domain.register()).collect();
    for slot in &slots {
        slot.quiescent();
        slot.idle();
    }
    let started = Instant::now();
    cell.store(None)
        .wait(Instant::now() + Duration::from_secs(1))
        .unwrap_or_else(|_| panic!("every slot is idle"));
    assert!(started.elapsed() < Duration::from_millis(500));
    assert_eq!(count.load(StdOrdering::SeqCst), 1);
}

#[test]
fn a_pin_inside_an_active_slot_keeps_the_slots_older_announcement() {
    let domain = EpochDomain::new(2);
    let count = drops();
    let cell = Published::new(domain.clone(), Some(Counted::new(&count)));
    let worker = domain.register();
    worker.quiescent();
    let outer = worker.pin();
    let held = cell.load(&outer).unwrap();

    cell.store(Some(Counted::new(&count))).defer();
    // A nested pin must not raise the announcement past the store.
    drop(worker.pin());
    cell.try_reclaim();
    assert_eq!(count.load(StdOrdering::SeqCst), 0);
    assert_eq!(held.magic.load(StdOrdering::SeqCst), LIVE);
    assert!(worker.is_active());

    drop(outer);
    worker.quiescent();
    cell.try_reclaim();
    assert_eq!(count.load(StdOrdering::SeqCst), 1);
}

#[test]
fn quiescent_and_idle_while_a_guard_is_alive_leave_what_it_protects_protected() {
    let domain = EpochDomain::new(2);
    let count = drops();
    let cell = Published::new(domain.clone(), Some(Counted::new(&count)));
    let worker = domain.register();
    worker.quiescent();
    let guard = worker.pin();
    let held = cell.load(&guard).unwrap();

    cell.store(Some(Counted::new(&count))).defer();
    // The owner asks for a new point and for idle while the guard's reference is live: both wait for the guard.
    worker.quiescent();
    worker.idle();
    cell.try_reclaim();
    assert_eq!(count.load(StdOrdering::SeqCst), 0);
    assert_eq!(held.magic.load(StdOrdering::SeqCst), LIVE);

    drop(guard);
    assert!(!worker.is_active(), "the deferred idle took effect");
    cell.try_reclaim();
    assert_eq!(count.load(StdOrdering::SeqCst), 1);
}

#[test]
fn a_quiescent_point_during_a_pin_from_idle_is_ignored_and_the_pin_still_withdraws() {
    let domain = EpochDomain::new(1);
    let count = drops();
    let cell = Published::new(domain.clone(), Some(Counted::new(&count)));
    let thread = domain.register();
    let guard = thread.pin();
    let held = cell.load(&guard).unwrap();
    cell.store(None).defer();
    thread.quiescent();
    cell.try_reclaim();
    assert_eq!(count.load(StdOrdering::SeqCst), 0);
    assert_eq!(held.magic.load(StdOrdering::SeqCst), LIVE);
    drop(guard);
    assert!(!thread.is_active());
    cell.try_reclaim();
    assert_eq!(count.load(StdOrdering::SeqCst), 1);
}

#[test]
fn a_guard_that_announced_the_slot_withdraws_it() {
    let domain = EpochDomain::new(1);
    let slot = domain.register();
    {
        let _guard = slot.pin();
        assert!(slot.is_active());
        assert!(!domain.passed(domain.epoch.load(StdOrdering::SeqCst) + 1));
    }
    assert!(!slot.is_active());
    assert!(domain.passed(u64::MAX));
}

#[test]
fn a_slot_given_back_is_reused() {
    let domain = EpochDomain::new(0);
    let first = domain.register();
    let second = domain.register();
    assert_eq!(domain.slot_count(), 2);
    drop(first);
    let third = domain.register();
    assert_eq!(domain.slot_count(), 2, "the freed slot is taken again");
    drop((second, third));
    drop(domain.register());
    assert_eq!(domain.slot_count(), 2);
}

#[test]
fn a_dropped_active_slot_stops_holding_writers_back() {
    let domain = EpochDomain::new(1);
    let count = drops();
    let cell = Published::new(domain.clone(), Some(Counted::new(&count)));
    let slot = domain.register();
    slot.quiescent();
    cell.store(None).defer();
    assert_eq!(count.load(StdOrdering::SeqCst), 0);
    drop(slot);
    cell.try_reclaim();
    assert_eq!(count.load(StdOrdering::SeqCst), 1);
}

#[test]
fn store_if_current_refuses_a_value_that_was_replaced() {
    let domain = EpochDomain::new(1);
    let count = drops();
    let first = Counted::new(&count);
    let cell = Published::new(domain.clone(), Some(first.clone()));
    cell.store(Some(Counted::new(&count))).wait(soon()).ok();

    let late = Counted::new(&count);
    let refused = cell
        .store_if_current(Some(&first), Some(late.clone()))
        .err()
        .expect("the cell holds another value now");
    assert!(Arc::ptr_eq(&refused.unwrap(), &late));
    assert!(
        cell.store_if_current(None, None).is_err(),
        "the cell is not empty"
    );

    let reader = domain.register();
    let guard = reader.pin();
    let current = cell.load(&guard).unwrap();
    cell.store_if_current(Some(current), None)
        .unwrap_or_else(|_| panic!("the cell still holds that value"))
        .defer();
    drop(guard);
    assert!(cell.load(&reader.pin()).is_none());
    assert!(
        cell.store_if_current(None, Some(late)).is_ok(),
        "the cell is empty now"
    );
}

#[test]
fn dropping_the_cell_frees_the_current_and_the_pending_values() {
    let domain = EpochDomain::new(1);
    let count = drops();
    let cell = Published::new(domain.clone(), Some(Counted::new(&count)));
    let reader = domain.register();
    let guard = reader.pin();
    cell.store(Some(Counted::new(&count))).defer();
    drop(guard);
    // The retirement is still on the list: nothing reclaimed it yet.
    assert_eq!(cell.pending(), 1);
    drop(cell);
    assert_eq!(count.load(StdOrdering::SeqCst), 2);
}

#[test]
fn dropping_a_retirement_defers_it() {
    let domain = EpochDomain::new(1);
    let count = drops();
    let cell = Published::new(domain.clone(), Some(Counted::new(&count)));
    let reader = domain.register();
    let guard = reader.pin();
    drop(cell.store(None));
    assert_eq!(count.load(StdOrdering::SeqCst), 0);
    assert_eq!(cell.pending(), 1);
    drop(guard);
    cell.try_reclaim();
    assert_eq!(count.load(StdOrdering::SeqCst), 1);
}

#[test]
fn an_empty_cell_loads_nothing_and_retires_nothing() {
    let domain = EpochDomain::new(1);
    let cell: Published<Counted> = Published::new(domain.clone(), None);
    let reader = domain.register();
    assert!(cell.load(&reader.pin()).is_none());
    cell.store(None)
        .wait(soon())
        .unwrap_or_else(|_| panic!("nothing to wait for"));
    assert_eq!(cell.pending(), 0);
}

#[test]
#[should_panic(expected = "a guard protects the values of its own domain only")]
fn a_guard_of_another_domain_is_refused() {
    let cell: Published<Counted> = Published::new(EpochDomain::new(1), None);
    let other = EpochDomain::new(1);
    let slot = other.register();
    let guard = slot.pin();
    let _ = cell.load(&guard);
}

/// Readers that pin from idle, readers that announce quiescent points like a worker, and a writer that replaces the
/// value and reclaims: no reader ever sees a dropped value.
#[test]
fn readers_never_see_a_freed_value() {
    let domain = EpochDomain::new(4);
    let count = drops();
    let cell = Arc::new(Published::new(domain.clone(), Some(Counted::new(&count))));
    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let started = Arc::new(AtomicUsize::new(0));

    let mut readers = Vec::new();
    let styles: &[bool] = if cfg!(miri) {
        &[false, true]
    } else {
        &[false, true, false, true]
    };
    for &worker_style in styles {
        let (cell, domain, stop) = (cell.clone(), domain.clone(), stop.clone());
        let started = started.clone();
        readers.push(thread::spawn(move || {
            let slot = domain.register();
            let mut reads = 0_u64;
            while !stop.load(StdOrdering::Relaxed) {
                if worker_style {
                    slot.quiescent();
                }
                let guard = slot.pin();
                if let Some(value) = cell.load(&guard) {
                    for _ in 0..4 {
                        assert_eq!(value.magic.load(StdOrdering::SeqCst), LIVE, "a freed value");
                    }
                }
                reads += 1;
                if reads == 1 {
                    started.fetch_add(1, StdOrdering::SeqCst);
                }
                if reads.is_multiple_of(64) {
                    slot.idle();
                }
            }
            reads
        }));
    }

    // Replace the value only while every reader is reading.
    while started.load(StdOrdering::SeqCst) < readers.len() {
        thread::yield_now();
    }
    for _ in 0..ROUNDS {
        cell.store(Some(Counted::new(&count))).defer();
    }
    stop.store(true, StdOrdering::Relaxed);
    for reader in readers {
        assert!(reader.join().unwrap() > 0);
    }
    cell.try_reclaim();
    drop(cell);
    assert_eq!(count.load(StdOrdering::SeqCst), ROUNDS + 1);
}

#[cfg(loom)]
mod loom {
    use super::super::*;
    use ::loom::cell::UnsafeCell;
    use ::loom::thread;

    /// A value whose reads and whose drop loom tracks: a drop that is not ordered after a read is a data race.
    struct Payload {
        value: UnsafeCell<u32>,
    }

    impl Payload {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                value: UnsafeCell::new(7),
            })
        }

        fn read(&self) -> u32 {
            // SAFETY: nothing writes the value until the drop.
            self.value.with(|value| unsafe { *value })
        }
    }

    impl Drop for Payload {
        fn drop(&mut self) {
            // SAFETY: the drop has exclusive access to the value.
            self.value.with_mut(|value| unsafe { *value = 0 });
        }
    }

    // SAFETY: the value is read-only until the drop, which has exclusive access.
    unsafe impl Send for Payload {}
    // SAFETY: as above.
    unsafe impl Sync for Payload {}

    // The reader announces before the writer starts, so the model checks the part loom can: a value is freed only
    // after the reader's later announcement, and the free is ordered after the reader's last read. (The fence that
    // orders a thread's first announcement against a writer's scan is a sequentially consistent fence, which loom
    // treats as acquire-release; `readers_never_see_a_freed_value` runs that part on real threads.)
    #[test]
    fn loom_reader_and_writer_never_see_a_freed_value() {
        ::loom::model(|| {
            let domain = EpochDomain::new(1);
            let cell = Arc::new(Published::new(domain.clone(), Some(Payload::new())));
            let reader = domain.register();
            reader.quiescent();

            let read = {
                let cell = cell.clone();
                thread::spawn(move || {
                    let held = {
                        let guard = reader.pin();
                        cell.load(&guard).map(|value| value.read())
                    };
                    assert!(matches!(held, Some(7) | None));
                    // The reference is gone: the next quiescent point lets the writer free the old value.
                    reader.quiescent();
                    reader
                })
            };
            let write = {
                let cell = cell.clone();
                thread::spawn(move || {
                    cell.store(Some(Payload::new())).defer();
                    cell.try_reclaim();
                })
            };
            let reader = read.join().unwrap();
            write.join().unwrap();
            reader.idle();
            cell.try_reclaim();
            assert_eq!(cell.pending(), 0);
        });
    }

    // The reader's slot is idle: its first pin announces the epoch, fences and then loads, while the writer swaps the
    // pointer, advances the epoch and scans. Either the pin's load sees the new value, or the scan sees the
    // announcement and the old value stays; loom treats the sequentially consistent fence as acquire-release, so
    // this checks the reasoning the model can (the free is ordered after the read) and
    // `readers_never_see_a_freed_value` runs the store-buffer part on real threads.
    #[test]
    fn loom_a_first_pin_from_idle_races_a_store() {
        ::loom::model(|| {
            let domain = EpochDomain::new(1);
            let cell = Arc::new(Published::new(domain.clone(), Some(Payload::new())));
            let reader = domain.register();
            let read = {
                let cell = cell.clone();
                thread::spawn(move || {
                    let held = {
                        let guard = reader.pin();
                        cell.load(&guard).map(|value| value.read())
                    };
                    assert!(matches!(held, Some(7)), "read {held:?}");
                    assert!(!reader.is_active(), "the guard withdrew the announcement");
                })
            };
            let write = {
                let cell = cell.clone();
                thread::spawn(move || {
                    cell.store(Some(Payload::new())).defer();
                    cell.try_reclaim();
                })
            };
            read.join().unwrap();
            write.join().unwrap();
            cell.try_reclaim();
            assert_eq!(
                cell.pending(),
                0,
                "every retirement was freed once the reader left"
            );
        });
    }

    /// Counts its drops.
    struct Counting(Arc<::loom::sync::atomic::AtomicUsize>);

    impl Drop for Counting {
        fn drop(&mut self) {
            self.0.fetch_add(1, ::loom::sync::atomic::Ordering::SeqCst);
        }
    }

    // Two threads defer retirements (`push_chain`, one by `store`, one by `store_if_current` under a pin), a third
    // reclaims (take the list whole, free what passed, push the rest back) while the main thread counts with
    // `pending()` (which takes the list too). Every value is dropped exactly once and none is lost.
    #[test]
    fn loom_two_deferrers_a_reclaimer_and_pending() {
        let mut builder = ::loom::model::Builder::new();
        builder.preemption_bound = Some(3);
        builder.check(|| {
            let drops = Arc::new(::loom::sync::atomic::AtomicUsize::new(0));
            let domain = EpochDomain::new(2);
            let cell = Arc::new(Published::new(
                domain.clone(),
                Some(Arc::new(Counting(drops.clone()))),
            ));
            let slot = domain.register();
            let storer = {
                let (cell, drops) = (cell.clone(), drops.clone());
                thread::spawn(move || {
                    cell.store(Some(Arc::new(Counting(drops)))).defer();
                })
            };
            let swapper = {
                let (cell, drops) = (cell.clone(), drops.clone());
                thread::spawn(move || {
                    let guard = slot.pin();
                    let current = cell.load(&guard);
                    if let Ok(retired) =
                        cell.store_if_current(current, Some(Arc::new(Counting(drops))))
                    {
                        retired.defer();
                    }
                })
            };
            let reclaimer = {
                let cell = cell.clone();
                thread::spawn(move || cell.try_reclaim())
            };
            let seen = cell.pending();
            assert!(seen <= 2, "{seen} pending");
            storer.join().unwrap();
            swapper.join().unwrap();
            reclaimer.join().unwrap();
            // All threads are idle now: a sweep frees everything retired.
            cell.try_reclaim();
            assert_eq!(cell.pending(), 0);
            let cell = Arc::into_inner(cell).expect("the threads are done");
            drop(cell);
            // The initial value plus up to two replacements; a refused replacement is dropped by its caller.
            let dropped = drops.load(::loom::sync::atomic::Ordering::SeqCst);
            assert_eq!(dropped, 3, "every value is dropped exactly once");
        });
    }
}
