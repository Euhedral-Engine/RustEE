use euhedral_data_structures::atomics::AtomicDouble;
use std::thread;

#[test]
fn starts_at_zero_by_default_and_at_the_given_value_otherwise() {
    assert_eq!(AtomicDouble::default().get(), 0.0);
    assert_eq!(AtomicDouble::new(-7.5).get(), -7.5);
}

#[test]
fn every_get_and_set_flavour_reads_back_the_stored_value() {
    let value = AtomicDouble::new(0.0);

    value.set(1.5);
    assert_eq!(value.get(), 1.5);
    value.set_release(2.5);
    assert_eq!(value.get_acquire(), 2.5);
    value.lazy_set(3.5);
    assert_eq!(value.get_opaque(), 3.5);
    value.set_opaque(4.5);
    assert_eq!(value.get_plain(), 4.5);
    value.set_plain(5.5);
    assert_eq!(value.get(), 5.5);
}

#[test]
fn stores_preserve_the_exact_bit_pattern() {
    let value = AtomicDouble::new(0.0);

    value.set(-0.0);
    assert_eq!(value.get().to_bits(), (-0.0_f64).to_bits());
    value.set(f64::NAN);
    assert!(value.get().is_nan());
    value.set(f64::NEG_INFINITY);
    assert_eq!(value.get(), f64::NEG_INFINITY);
}

#[test]
fn get_and_set_returns_the_previous_value() {
    let value = AtomicDouble::new(10.0);

    assert_eq!(value.get_and_set(20.0), 10.0);
    assert_eq!(value.get(), 20.0);
}

#[test]
fn increment_and_decrement_report_the_old_or_new_value() {
    let value = AtomicDouble::new(10.0);

    assert_eq!(value.get_and_increment(), 10.0);
    assert_eq!(value.increment_and_get(), 12.0);
    assert_eq!(value.get_and_decrement(), 12.0);
    assert_eq!(value.decrement_and_get(), 10.0);
}

#[test]
fn add_flavours_return_old_or_new_value() {
    let value = AtomicDouble::new(100.0);

    assert_eq!(value.get_and_add(0.5), 100.0);
    assert_eq!(value.add_and_get(-20.25), 80.25);
    assert_eq!(value.get(), 80.25);
}

#[test]
fn add_follows_ieee_semantics() {
    let value = AtomicDouble::new(f64::MAX);
    assert_eq!(value.add_and_get(f64::MAX), f64::INFINITY);

    let value = AtomicDouble::new(f64::INFINITY);
    assert!(value.add_and_get(f64::NEG_INFINITY).is_nan());
}

#[test]
fn update_functions_return_old_or_new_value() {
    let value = AtomicDouble::new(3.0);

    assert_eq!(value.get_and_update(|v| v * 2.0), 3.0);
    assert_eq!(value.update_and_get(|v| v + 1.0), 7.0);
    assert_eq!(value.get(), 7.0);
}

#[test]
fn accumulate_functions_return_old_or_new_value() {
    let value = AtomicDouble::new(3.0);

    assert_eq!(value.get_and_accumulate(4.0, |prev, x| prev * x), 3.0);
    assert_eq!(value.accumulate_and_get(5.0, |prev, x| prev - x), 7.0);
    assert_eq!(value.get(), 7.0);
}

#[test]
fn compare_and_set_only_swaps_on_a_match() {
    let value = AtomicDouble::new(1.0);

    assert!(!value.compare_and_set(2.0, 3.0));
    assert_eq!(value.get(), 1.0);
    assert!(value.compare_and_set(1.0, 3.0));
    assert_eq!(value.get(), 3.0);
}

#[test]
fn compare_and_set_compares_raw_bits() {
    let value = AtomicDouble::new(0.0);
    assert!(
        !value.compare_and_set(-0.0, 1.0),
        "-0.0 is not bitwise equal to 0.0"
    );
    assert!(value.compare_and_set(0.0, 1.0));

    let value = AtomicDouble::new(f64::NAN);
    assert!(
        value.compare_and_set(f64::NAN, 2.0),
        "NaN matches itself bit for bit"
    );
    assert_eq!(value.get(), 2.0);

    let value = AtomicDouble::new(f64::from_bits(0x7ff8_0000_0000_0001));
    assert!(
        !value.compare_and_set(f64::NAN, 2.0),
        "a different NaN payload does not match"
    );
}

#[test]
fn weak_compare_and_set_succeeds_eventually_and_never_on_a_mismatch() {
    let value = AtomicDouble::new(1.0);

    assert!(!value.weak_compare_and_set(9.0, 3.0));
    while !value.weak_compare_and_set(1.0, 3.0) {}
    assert_eq!(value.get(), 3.0);
}

#[test]
fn compare_and_exchange_returns_the_witness_value() {
    let value = AtomicDouble::new(1.0);

    assert_eq!(value.compare_and_exchange(2.0, 9.0), 1.0);
    assert_eq!(value.get(), 1.0);
    assert_eq!(value.compare_and_exchange(1.0, 9.0), 1.0);
    assert_eq!(value.get(), 9.0);
}

#[test]
fn concurrent_adds_are_not_lost() {
    let value = AtomicDouble::new(0.0);
    let threads = 8;
    let per_thread = 10_000;

    thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| {
                for _ in 0..per_thread {
                    value.get_and_add(0.5);
                }
            });
        }
    });

    assert_eq!(value.get(), threads as f64 * per_thread as f64 * 0.5);
}
