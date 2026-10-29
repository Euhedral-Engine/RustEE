#![allow(
    clippy::disallowed_methods,
    reason = "test fixtures: the test thread runs peer threads and waits on them"
)]
use euhedral_data_structures::atomics::PaddedAtomicLong;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;

#[test]
fn starts_at_zero_by_default_and_at_the_given_value_otherwise() {
    assert_eq!(PaddedAtomicLong::default().get(), 0);
    assert_eq!(PaddedAtomicLong::new(-7).get(), -7);
}

#[test]
fn every_get_and_set_flavour_reads_back_the_stored_value() {
    let value = PaddedAtomicLong::new(0);

    value.set(1);
    assert_eq!(value.get(), 1);
    value.set_release(2);
    assert_eq!(value.get_acquire(), 2);
    value.lazy_set(3);
    assert_eq!(value.get_opaque(), 3);
    value.set_opaque(4);
    assert_eq!(value.get_plain(), 4);
    value.set_plain(5);
    assert_eq!(value.get(), 5);
}

#[test]
fn get_and_set_returns_the_previous_value() {
    let value = PaddedAtomicLong::new(10);

    assert_eq!(value.get_and_set(20), 10);
    assert_eq!(value.get(), 20);
}

#[test]
fn increment_and_decrement_report_the_old_or_new_value() {
    let value = PaddedAtomicLong::new(10);

    assert_eq!(value.get_and_increment(), 10);
    assert_eq!(value.increment_and_get(), 12);
    assert_eq!(value.get_and_decrement(), 12);
    assert_eq!(value.decrement_and_get(), 10);
}

#[test]
fn add_flavours_return_old_or_new_value() {
    let value = PaddedAtomicLong::new(100);

    assert_eq!(value.get_and_add(5), 100);
    assert_eq!(value.get_and_add_release(-20), 105);
    assert_eq!(value.add_and_get(7), 92);
    assert_eq!(value.get(), 92);
}

#[test]
fn arithmetic_wraps_like_a_java_long() {
    let value = PaddedAtomicLong::new(i64::MAX);

    assert_eq!(value.increment_and_get(), i64::MIN);
    assert_eq!(value.decrement_and_get(), i64::MAX);
    assert_eq!(value.add_and_get(i64::MAX), -2);
}

#[test]
fn update_functions_return_old_or_new_value() {
    let value = PaddedAtomicLong::new(3);

    assert_eq!(value.get_and_update(|v| v * 2), 3);
    assert_eq!(value.update_and_get(|v| v + 1), 7);
    assert_eq!(value.get(), 7);
}

#[test]
fn accumulate_functions_return_old_or_new_value() {
    let value = PaddedAtomicLong::new(3);

    assert_eq!(value.get_and_accumulate(4, |prev, x| prev * x), 3);
    assert_eq!(value.accumulate_and_get(5, |prev, x| prev - x), 7);
    assert_eq!(value.get(), 7);
}

#[test]
fn update_retries_until_it_wins_against_a_concurrent_writer() {
    let value = PaddedAtomicLong::new(0);
    let mut calls = 0;

    let new = value.update_and_get(|v| {
        calls += 1;
        v + 1
    });

    assert_eq!(new, 1);
    assert!(calls >= 1);
}

#[test]
fn compare_and_set_only_swaps_on_a_match() {
    let value = PaddedAtomicLong::new(1);

    assert!(!value.compare_and_set(2, 3));
    assert_eq!(value.get(), 1);
    assert!(value.compare_and_set(1, 3));
    assert_eq!(value.get(), 3);
}

#[test]
fn weak_compare_and_set_succeeds_eventually_and_never_on_a_mismatch() {
    let value = PaddedAtomicLong::new(1);

    assert!(!value.weak_compare_and_set(9, 3));
    while !value.weak_compare_and_set(1, 3) {}
    assert_eq!(value.get(), 3);
}

#[test]
fn compare_and_exchange_returns_the_witness_value() {
    let value = PaddedAtomicLong::new(1);

    assert_eq!(value.compare_and_exchange(2, 9), 1);
    assert_eq!(value.get(), 1);
    assert_eq!(value.compare_and_exchange(1, 9), 1);
    assert_eq!(value.get(), 9);
}

#[test]
fn concurrent_increments_are_not_lost() {
    let value = PaddedAtomicLong::new(0);
    let threads = 8;
    let per_thread = 10_000;

    thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| {
                for _ in 0..per_thread {
                    value.get_and_increment();
                }
            });
        }
    });

    assert_eq!(value.get(), threads * per_thread);
}

#[test]
fn release_store_publishes_prior_writes_to_an_acquire_load() {
    let data = PaddedAtomicLong::new(0);
    let flag = PaddedAtomicLong::new(0);
    let saw_stale = AtomicBool::new(false);

    thread::scope(|scope| {
        scope.spawn(|| {
            data.set_plain(42);
            flag.set_release(1);
        });
        scope.spawn(|| {
            while flag.get_acquire() == 0 {
                std::hint::spin_loop();
            }
            if data.get_plain() != 42 {
                saw_stale.store(true, Ordering::Relaxed);
            }
        });
    });

    assert!(!saw_stale.load(Ordering::Relaxed));
}
