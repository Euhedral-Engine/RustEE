use euhedral_data_structures::atomics::PaddedLongAdder;
use std::thread;

fn signed(bits: u64) -> i64 {
    bits as i64
}

/// An adder holding `values`, one per logical slot, in either indexing mode.
fn adder(values: &[i64], bounds_check: bool) -> PaddedLongAdder {
    let adder = PaddedLongAdder::with_options(values.len(), bounds_check, false);
    for (logical, value) in values.iter().enumerate() {
        let idx = if bounds_check {
            logical
        } else {
            adder.physical_idx_for_raw(logical as i64)
        };
        adder.add_at(idx, *value);
    }
    adder
}

/// `(slot values, sum, min, max, mean bits)` from the upstream Java adder with bounds checks on. The
/// mean is the exact 128-bit sum divided by the slot count.
fn reductions() -> Vec<(Vec<i64>, i64, i64, i64, u64)> {
    vec![
        (vec![1, 2, 3, 4], 10, 1, 4, 0x4004000000000000),
        (vec![-1, -2, -3], -6, -3, -1, 0xc000000000000000),
        (
            vec![i64::MAX; 3],
            signed(0x7ffffffffffffffd),
            i64::MAX,
            i64::MAX,
            0x43e0000000000000,
        ),
        (vec![i64::MIN; 4], 0, i64::MIN, i64::MIN, 0xc3e0000000000000),
        (vec![i64::MAX, 1], i64::MIN, 1, i64::MAX, 0x43d0000000000000),
        (
            vec![i64::MIN, -1],
            i64::MAX,
            i64::MIN,
            -1,
            0xc3d0000000000000,
        ),
        (
            vec![i64::MAX, i64::MIN],
            -1,
            i64::MIN,
            i64::MAX,
            0xbfe0000000000000,
        ),
        (
            vec![123456789012345, -98765432109876, 5],
            0x00001674e78ea44a,
            signed(0xffffa62c6180c4cc),
            0x00007048860ddf79,
            0x429df134be3062ab,
        ),
        (vec![0, 0, 0], 0, 0, 0, 0),
        (
            vec![1 << 62; 5],
            1 << 62,
            1 << 62,
            1 << 62,
            0x43d0000000000000,
        ),
        (vec![7], 7, 7, 7, 0x401c000000000000),
    ]
}

#[test]
fn default_adder_has_one_slot_per_cpu() {
    let cpus = thread::available_parallelism().unwrap().get();

    assert_eq!(PaddedLongAdder::new().length(), cpus);
    assert_eq!(PaddedLongAdder::default().length(), cpus);
    assert_eq!(PaddedLongAdder::with_length(3).length(), 3);
}

#[test]
fn indexed_updates_hit_only_the_named_slot() {
    let adder = PaddedLongAdder::with_length(3);

    adder.increment_at(1);
    adder.increment_at(1);
    adder.decrement_at(2);
    adder.add_at(0, 40);

    assert_eq!([adder.get(0), adder.get(1), adder.get(2)], [40, 2, -1]);
}

#[test]
fn reductions_match_upstream_with_bounds_checks_on() {
    for (values, sum, min, max, mean) in reductions() {
        let adder = adder(&values, true);
        assert_eq!(adder.sum(), sum, "sum of {values:?}");
        assert_eq!(adder.min(), min, "min of {values:?}");
        assert_eq!(adder.max(), max, "max of {values:?}");
        assert_eq!(adder.mean().to_bits(), mean, "mean of {values:?}");
    }
}

#[test]
fn reductions_ignore_padding_when_bounds_checks_are_off() {
    // Upstream folded the zeroed padding slots into min, max and mean in this mode; the answers must
    // be the same as with bounds checks on.
    for (values, sum, min, max, mean) in reductions() {
        let adder = adder(&values, false);
        assert_eq!(adder.sum(), sum, "sum of {values:?}");
        assert_eq!(adder.min(), min, "min of {values:?}");
        assert_eq!(adder.max(), max, "max of {values:?}");
        assert_eq!(adder.mean().to_bits(), mean, "mean of {values:?}");
    }
}

#[test]
fn reductions_of_an_adder_with_no_slots_use_the_identity_values() {
    let adder = PaddedLongAdder::with_length(0);

    assert_eq!(adder.sum(), 0);
    assert_eq!(adder.min(), i64::MAX);
    assert_eq!(adder.max(), i64::MIN);
    assert_eq!(adder.mean(), 0.0);
}

#[test]
fn random_updates_work_with_and_without_bounds_checks() {
    // Upstream computed a physical slot and then bounds-checked it as a logical one, so increment()
    // threw on any adder built with bounds checks, which includes every default adder.
    for bounds_check in [true, false] {
        let adder = PaddedLongAdder::with_options(8, bounds_check, false);

        for _ in 0..1_000 {
            adder.increment();
        }
        for _ in 0..200 {
            adder.decrement();
        }
        adder.add(5);

        assert_eq!(adder.sum(), 805, "bounds_check {bounds_check}");
    }
}

#[test]
fn random_updates_spread_over_the_slots() {
    let adder = PaddedLongAdder::with_length(8);

    for _ in 0..10_000 {
        adder.increment();
    }

    let used = (0..8).filter(|idx| adder.get(*idx) != 0).count();
    assert!(used > 1, "all updates landed in one slot");
}

#[test]
fn sum_and_reset_returns_the_sum_and_zeroes_every_slot() {
    for bounds_check in [true, false] {
        let adder = adder(&[5, 6, 7], bounds_check);

        assert_eq!(adder.sum_and_reset(), 18);
        assert_eq!(adder.sum(), 0);
        assert_eq!((adder.min(), adder.max()), (0, 0));
    }
}

#[test]
fn reset_zeroes_every_slot() {
    let adder = adder(&[5, 6, 7], true);

    adder.reset();

    assert_eq!(adder.sum(), 0);
    assert_eq!(adder.max(), 0);
}

#[test]
fn concurrent_random_increments_are_not_lost() {
    let adder = PaddedLongAdder::new();
    let threads = 8;
    let per_thread = 10_000;

    thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| {
                for _ in 0..per_thread {
                    adder.increment();
                }
            });
        }
    });

    assert_eq!(adder.sum(), threads * per_thread);
}
