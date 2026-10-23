use euhedral_data_structures::atomics::PaddedAtomicDoubleArray;
use std::thread;

#[test]
fn reports_its_logical_length_and_starts_zeroed() {
    let array = PaddedAtomicDoubleArray::new(5);

    assert_eq!(array.length(), 5);
    for idx in 0..5 {
        assert_eq!(array.get(idx), 0.0);
    }
}

#[test]
fn shares_the_long_array_layout() {
    // (length, pad128, physical index for raw index 0, 1, 5, -1) from the upstream Java fromRawIdx.
    let cases: &[(usize, bool, [usize; 4])] = &[
        (6, false, [7, 15, 47, 47]),
        (6, true, [15, 31, 95, 95]),
        (8, false, [7, 15, 47, 63]),
        (8, true, [15, 31, 95, 127]),
    ];
    for &(length, pad128, expected) in cases {
        let array = PaddedAtomicDoubleArray::with_options(length, true, pad128);
        let actual = [0, 1, 5, -1].map(|raw| array.physical_idx_for_raw(raw));
        assert_eq!(actual, expected, "length {length} pad128 {pad128}");
    }
}

#[test]
fn elements_are_independent() {
    let array = PaddedAtomicDoubleArray::new(4);

    array.set(1, 1.5);
    array.set(2, 2.5);

    assert_eq!(
        [array.get(0), array.get(1), array.get(2), array.get(3)],
        [0.0, 1.5, 2.5, 0.0]
    );
}

#[test]
fn every_get_and_set_flavour_reads_back_the_stored_value() {
    let array = PaddedAtomicDoubleArray::new(2);

    array.set(1, 1.5);
    assert_eq!(array.get(1), 1.5);
    array.set_release(1, 2.5);
    assert_eq!(array.get_acquire(1), 2.5);
    array.lazy_set(1, 3.5);
    assert_eq!(array.get_opaque(1), 3.5);
    array.set_opaque(1, 4.5);
    assert_eq!(array.get_plain(1), 4.5);
    array.set_plain(1, 5.5);
    assert_eq!(array.get(1), 5.5);
    assert_eq!(array.get(0), 0.0);
}

#[test]
fn stores_preserve_the_exact_bit_pattern() {
    let array = PaddedAtomicDoubleArray::new(1);

    array.set(0, -0.0);
    assert_eq!(array.get(0).to_bits(), (-0.0_f64).to_bits());
    array.set(0, f64::NAN);
    assert!(array.get(0).is_nan());
}

#[test]
fn get_and_set_returns_the_previous_value() {
    let array = PaddedAtomicDoubleArray::new(2);
    array.set(1, 10.0);

    assert_eq!(array.get_and_set(1, 20.0), 10.0);
    assert_eq!(array.get(1), 20.0);
}

#[test]
fn fill_variants_write_every_logical_element() {
    let array = PaddedAtomicDoubleArray::new(7);

    array.fill(1.5);
    assert!((0..7).all(|i| array.get(i) == 1.5));
    array.lazy_fill(2.5);
    assert!((0..7).all(|i| array.get(i) == 2.5));
    array.fill_release(3.5);
    assert!((0..7).all(|i| array.get(i) == 3.5));
    array.fill_opaque(4.5);
    assert!((0..7).all(|i| array.get(i) == 4.5));
    array.fill_plain(5.5);
    assert!((0..7).all(|i| array.get(i) == 5.5));
}

#[test]
fn fill_leaves_the_padding_slots_untouched() {
    let array = PaddedAtomicDoubleArray::with_options(3, false, false);

    array.fill(7.0);

    for logical in 0..3 {
        assert_eq!(array.get(array.physical_idx_for_raw(logical)), 7.0);
    }
    assert!(
        (0..7)
            .chain(8..15)
            .all(|physical| array.get(physical) == 0.0)
    );
}

#[test]
fn increment_decrement_and_add_report_the_old_or_new_value() {
    let array = PaddedAtomicDoubleArray::new(3);
    array.set(2, 10.0);

    assert_eq!(array.get_and_increment(2), 10.0);
    assert_eq!(array.increment_and_get(2), 12.0);
    assert_eq!(array.get_and_decrement(2), 12.0);
    assert_eq!(array.decrement_and_get(2), 10.0);
    assert_eq!(array.get_and_add(2, 0.5), 10.0);
    assert_eq!(array.add_and_get(2, 0.5), 11.0);
    assert_eq!(array.get_and_add_release(2, -4.0), 11.0);
    assert_eq!(array.add_release_and_get(2, -4.0), 3.0);
    assert_eq!(array.get(2), 3.0);
    assert_eq!(array.get(1), 0.0);
}

#[test]
fn add_follows_ieee_semantics() {
    let array = PaddedAtomicDoubleArray::new(1);
    array.set(0, f64::MAX);

    assert_eq!(array.add_and_get(0, f64::MAX), f64::INFINITY);
    assert!(array.add_release_and_get(0, f64::NEG_INFINITY).is_nan());
}

#[test]
fn update_and_accumulate_return_old_or_new_value() {
    let array = PaddedAtomicDoubleArray::new(2);
    array.set(1, 3.0);

    assert_eq!(array.get_and_update(1, |v| v * 2.0), 3.0);
    assert_eq!(array.update_and_get(1, |v| v + 1.0), 7.0);
    assert_eq!(array.get_and_accumulate(1, 4.0, |prev, x| prev * x), 7.0);
    assert_eq!(array.accumulate_and_get(1, 5.0, |prev, x| prev - x), 23.0);
    assert_eq!(array.get(1), 23.0);
    assert_eq!(array.get(0), 0.0);
}

#[test]
fn compare_and_set_family_compares_raw_bits() {
    let array = PaddedAtomicDoubleArray::new(2);
    array.set(1, 1.0);

    assert!(!array.compare_and_set(1, 2.0, 3.0));
    assert!(!array.weak_compare_and_set(1, 2.0, 3.0));
    assert_eq!(array.compare_and_exchange(1, 2.0, 3.0), 1.0);
    assert_eq!(array.get(1), 1.0);

    assert!(array.compare_and_set(1, 1.0, 3.0));
    assert_eq!(array.compare_and_exchange(1, 3.0, 4.0), 3.0);
    while !array.weak_compare_and_set(1, 4.0, 5.0) {}
    assert_eq!(array.get(1), 5.0);

    array.set(0, 0.0);
    assert!(
        !array.compare_and_set(0, -0.0, 1.0),
        "-0.0 is not bitwise equal to 0.0"
    );
    array.set(0, f64::NAN);
    assert!(
        array.compare_and_set(0, f64::NAN, 1.0),
        "NaN matches itself bit for bit"
    );
}

#[test]
#[should_panic(expected = "Index 4 out of bounds for length 4")]
fn bounds_checked_access_panics_past_the_end() {
    PaddedAtomicDoubleArray::new(4).get(4);
}

#[test]
fn without_bounds_checks_indexes_are_physical_slots() {
    let array = PaddedAtomicDoubleArray::with_options(4, false, false);
    let slot = array.physical_idx_for_raw(2);

    array.set(slot, 9.5);

    assert_eq!(array.get(slot), 9.5);
    assert_eq!(
        array.get(0),
        0.0,
        "raw index 0 is a padding slot, reachable without bounds checks"
    );
}

#[test]
fn concurrent_adds_to_one_element_are_not_lost() {
    let array = PaddedAtomicDoubleArray::new(4);
    let threads = 8;
    let per_thread = 10_000;

    thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| {
                for _ in 0..per_thread {
                    array.get_and_add_release(3, 0.5);
                }
            });
        }
    });

    assert_eq!(array.get(3), threads as f64 * per_thread as f64 * 0.5);
    assert_eq!(array.get(2), 0.0);
}

#[test]
fn debug_lists_the_logical_elements() {
    let array = PaddedAtomicDoubleArray::new(3);
    array.set(0, 1.5);
    array.set(2, 3.0);

    assert_eq!(format!("{array:?}"), "[1.5, 0.0, 3.0]");
}
