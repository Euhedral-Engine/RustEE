use euhedral_data_structures::atomics::PaddedAtomicLongArray;
use std::thread;

const RAW_INDEXES: [i64; 17] = [
    0,
    1,
    5,
    6,
    7,
    8,
    9,
    100,
    -1,
    -8,
    -9,
    (1 << 32) | 3,
    (1 << 32) | 6,
    0x8000_0000,
    0xFFFF_FFFF,
    i64::MIN,
    i64::MAX,
];

/// `(length, pad128, physical index of each RAW_INDEXES entry)` taken from the upstream Java
/// `fromRawIdx`. Power-of-two lengths mask; the others take `floorMod((int) raw, length)`.
const RAW_IDX_TABLE: &[(usize, bool, [usize; 17])] = &[
    (1, false, [7; 17]),
    (1, true, [15; 17]),
    (
        2,
        false,
        [7, 15, 15, 7, 15, 7, 15, 7, 15, 7, 15, 15, 7, 7, 15, 7, 15],
    ),
    (
        2,
        true,
        [
            15, 31, 31, 15, 31, 15, 31, 15, 31, 15, 31, 31, 15, 15, 31, 15, 31,
        ],
    ),
    (
        6,
        false,
        [
            7, 15, 47, 7, 15, 23, 31, 39, 47, 39, 31, 31, 7, 39, 47, 7, 47,
        ],
    ),
    (
        6,
        true,
        [
            15, 31, 95, 15, 31, 47, 63, 79, 95, 79, 63, 63, 15, 79, 95, 15, 95,
        ],
    ),
    (
        8,
        false,
        [
            7, 15, 47, 55, 63, 7, 15, 39, 63, 7, 63, 31, 55, 7, 63, 7, 63,
        ],
    ),
    (
        8,
        true,
        [
            15, 31, 95, 111, 127, 15, 31, 79, 127, 15, 127, 63, 111, 15, 127, 15, 127,
        ],
    ),
    (
        10,
        false,
        [
            7, 15, 47, 55, 63, 71, 79, 7, 79, 23, 15, 31, 55, 23, 79, 7, 79,
        ],
    ),
    (
        10,
        true,
        [
            15, 31, 95, 111, 127, 143, 159, 15, 159, 47, 31, 63, 111, 47, 159, 15, 159,
        ],
    ),
    (
        64,
        false,
        [
            7, 15, 47, 55, 63, 71, 79, 295, 511, 455, 447, 31, 55, 7, 511, 7, 511,
        ],
    ),
    (
        64,
        true,
        [
            15, 31, 95, 111, 127, 143, 159, 591, 1023, 911, 895, 63, 111, 15, 1023, 15, 1023,
        ],
    ),
];

#[test]
fn reports_its_logical_length_and_starts_zeroed() {
    let array = PaddedAtomicLongArray::new(5);

    assert_eq!(array.length(), 5);
    for idx in 0..5 {
        assert_eq!(array.get(idx), 0);
    }
}

#[test]
fn physical_idx_for_raw_matches_upstream_for_pow2_and_non_pow2_lengths() {
    for &(length, pad128, expected) in RAW_IDX_TABLE {
        let array = PaddedAtomicLongArray::with_options(length, true, pad128);
        for (raw, expected) in RAW_INDEXES.iter().zip(expected) {
            assert_eq!(
                array.physical_idx_for_raw(*raw),
                expected,
                "length {length} pad128 {pad128} raw {raw}"
            );
        }
    }
}

#[test]
fn consecutive_elements_are_a_cache_line_apart_or_two_with_pad128() {
    let narrow = PaddedAtomicLongArray::with_options(4, true, false);
    let wide = PaddedAtomicLongArray::with_options(4, true, true);

    assert_eq!(
        narrow.physical_idx_for_raw(1) - narrow.physical_idx_for_raw(0),
        8
    );
    assert_eq!(
        wide.physical_idx_for_raw(1) - wide.physical_idx_for_raw(0),
        16
    );
}

#[test]
fn elements_are_independent() {
    let array = PaddedAtomicLongArray::new(4);

    array.set(1, 11);
    array.set(2, 22);

    assert_eq!(
        [array.get(0), array.get(1), array.get(2), array.get(3)],
        [0, 11, 22, 0]
    );
}

#[test]
fn every_get_and_set_flavour_reads_back_the_stored_value() {
    let array = PaddedAtomicLongArray::new(2);

    array.set(1, 1);
    assert_eq!(array.get(1), 1);
    array.set_release(1, 2);
    assert_eq!(array.get_acquire(1), 2);
    array.lazy_set(1, 3);
    assert_eq!(array.get_opaque(1), 3);
    array.set_opaque(1, 4);
    assert_eq!(array.get_plain(1), 4);
    array.set_plain(1, 5);
    assert_eq!(array.get(1), 5);
    assert_eq!(array.get(0), 0);
}

#[test]
fn get_and_set_returns_the_previous_value() {
    let array = PaddedAtomicLongArray::new(2);
    array.set(1, 10);

    assert_eq!(array.get_and_set(1, 20), 10);
    assert_eq!(array.get(1), 20);
}

#[test]
fn fill_variants_write_every_logical_element() {
    let array = PaddedAtomicLongArray::new(7);

    array.fill(1);
    assert!((0..7).all(|i| array.get(i) == 1));
    array.lazy_fill(2);
    assert!((0..7).all(|i| array.get(i) == 2));
    array.fill_release(3);
    assert!((0..7).all(|i| array.get(i) == 3));
    array.fill_opaque(4);
    assert!((0..7).all(|i| array.get(i) == 4));
    array.fill_plain(5);
    assert!((0..7).all(|i| array.get(i) == 5));
}

#[test]
fn increment_decrement_and_add_report_the_old_or_new_value() {
    let array = PaddedAtomicLongArray::new(3);
    array.set(2, 10);

    assert_eq!(array.get_and_increment(2), 10);
    assert_eq!(array.increment_and_get(2), 12);
    assert_eq!(array.get_and_decrement(2), 12);
    assert_eq!(array.decrement_and_get(2), 10);
    assert_eq!(array.get_and_add(2, 5), 10);
    assert_eq!(array.add_and_get(2, 5), 20);
    assert_eq!(array.get_and_add_release(2, -4), 20);
    assert_eq!(array.add_release_and_get(2, -4), 12);
    assert_eq!(array.get(2), 12);
    assert_eq!(array.get(1), 0);
}

#[test]
fn arithmetic_wraps_like_a_java_long() {
    let array = PaddedAtomicLongArray::new(1);
    array.set(0, i64::MAX);

    assert_eq!(array.increment_and_get(0), i64::MIN);
    assert_eq!(array.add_release_and_get(0, -1), i64::MAX);
}

#[test]
fn update_and_accumulate_return_old_or_new_value() {
    let array = PaddedAtomicLongArray::new(2);
    array.set(1, 3);

    assert_eq!(array.get_and_update(1, |v| v * 2), 3);
    assert_eq!(array.update_and_get(1, |v| v + 1), 7);
    assert_eq!(array.get_and_accumulate(1, 4, |prev, x| prev * x), 7);
    assert_eq!(array.accumulate_and_get(1, 5, |prev, x| prev - x), 23);
    assert_eq!(array.get(1), 23);
    assert_eq!(array.get(0), 0);
}

#[test]
fn compare_and_set_family_only_swaps_on_a_match() {
    let array = PaddedAtomicLongArray::new(2);
    array.set(1, 1);

    assert!(!array.compare_and_set(1, 2, 3));
    assert!(!array.weak_compare_and_set(1, 2, 3));
    assert_eq!(array.compare_and_exchange(1, 2, 3), 1);
    assert_eq!(array.get(1), 1);

    assert!(array.compare_and_set(1, 1, 3));
    assert_eq!(array.compare_and_exchange(1, 3, 4), 3);
    while !array.weak_compare_and_set(1, 4, 5) {}
    assert_eq!(array.get(1), 5);
}

#[test]
#[should_panic(expected = "Index 4 out of bounds for length 4")]
fn bounds_checked_access_panics_past_the_end() {
    PaddedAtomicLongArray::new(4).get(4);
}

#[test]
fn without_bounds_checks_indexes_are_physical_slots() {
    let array = PaddedAtomicLongArray::with_options(4, false, false);
    let slot = array.physical_idx_for_raw(2);

    array.set(slot, 9);

    assert_eq!(array.get(slot), 9);
    assert_eq!(
        array.get(0),
        0,
        "raw index 0 is a padding slot, reachable without bounds checks"
    );
}

#[test]
fn concurrent_updates_to_one_element_are_not_lost() {
    let array = PaddedAtomicLongArray::new(4);
    let threads = 8;
    let per_thread = 10_000;

    thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| {
                for _ in 0..per_thread {
                    array.get_and_increment(3);
                }
            });
        }
    });

    assert_eq!(array.get(3), threads * per_thread);
    assert_eq!(array.get(2), 0);
}

#[test]
fn debug_lists_the_logical_elements() {
    let array = PaddedAtomicLongArray::new(3);
    array.set(0, 1);
    array.set(2, 3);

    assert_eq!(format!("{array:?}"), "[1, 0, 3]");
}

#[test]
fn fill_leaves_the_padding_slots_untouched() {
    let array = PaddedAtomicLongArray::with_options(3, false, false);

    array.fill(7);

    for logical in 0..3 {
        assert_eq!(array.get(array.physical_idx_for_raw(logical)), 7);
    }
    // Physical slots 0..7 are padding in front of element 0, and 8..15 sit between elements 0 and 1.
    assert!((0..7).chain(8..15).all(|physical| array.get(physical) == 0));
}
