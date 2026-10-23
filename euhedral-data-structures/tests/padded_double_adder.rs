use euhedral_data_structures::atomics::PaddedDoubleAdder;
use std::thread;

/// An adder holding `values`, one per logical slot, in either indexing mode.
fn adder(values: &[f64], bounds_check: bool) -> PaddedDoubleAdder {
    let adder = PaddedDoubleAdder::with_options(values.len(), bounds_check, false);
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

struct Case {
    values: Vec<f64>,
    sum: u64,
    min: u64,
    max: u64,
    mean: u64,
}

fn case(values: &[f64], sum: u64, min: u64, max: u64, mean: u64) -> Case {
    Case {
        values: values.to_vec(),
        sum,
        min,
        max,
        mean,
    }
}

/// Bit patterns from the upstream Java adder with bounds checks on, except `max`: upstream started it
/// at `Double.MIN_VALUE` (the smallest positive number), so it reported `5e-324` whenever every finite
/// slot was negative or zero. Those expectations are the true maximum.
fn cases() -> Vec<Case> {
    let f64_min = f64::MIN.to_bits();
    vec![
        case(
            &[1.5, 2.5, 3.0],
            0x401c000000000000,
            0x3ff8000000000000,
            0x4008000000000000,
            0x4002aaaaaaaaaaab,
        ),
        case(
            &[-1.5, -2.5, -3.0],
            0xc01c000000000000,
            0xc008000000000000,
            (-1.5_f64).to_bits(),
            0xc002aaaaaaaaaaab,
        ),
        case(
            &[f64::NAN, 1.0, 2.0],
            0x4008000000000000,
            0x3ff0000000000000,
            0x4000000000000000,
            0x3ff8000000000000,
        ),
        case(
            &[f64::INFINITY, f64::NEG_INFINITY, 4.0],
            0x4010000000000000,
            0x4010000000000000,
            0x4010000000000000,
            0x4010000000000000,
        ),
        case(
            &[f64::NAN, f64::INFINITY],
            0,
            f64::MAX.to_bits(),
            f64_min,
            0,
        ),
        case(&[0.0, -0.0], 0, 0, 0, 0),
        case(
            &[f64::MAX, f64::MAX],
            0x7ff0000000000000,
            f64::MAX.to_bits(),
            f64::MAX.to_bits(),
            0x7ff0000000000000,
        ),
        case(
            &[1e-300, 2e-300],
            0x01c01297d23ab683,
            0x01a56e1fc2f8f359,
            0x01b56e1fc2f8f359,
            0x01b01297d23ab683,
        ),
        case(
            &[-5.0],
            0xc014000000000000,
            0xc014000000000000,
            0xc014000000000000,
            0xc014000000000000,
        ),
        case(
            &[0.1, 0.2, 0.3],
            0x3fe3333333333334,
            0x3fb999999999999a,
            0x3fd3333333333333,
            0x3fc999999999999b,
        ),
    ]
}

fn assert_reductions(bounds_check: bool) {
    for case in cases() {
        let adder = adder(&case.values, bounds_check);
        let name = format!("{:?} (bounds_check {bounds_check})", case.values);
        assert_eq!(adder.sum().to_bits(), case.sum, "sum of {name}");
        assert_eq!(adder.min().to_bits(), case.min, "min of {name}");
        assert_eq!(adder.max().to_bits(), case.max, "max of {name}");
        assert_eq!(adder.mean().to_bits(), case.mean, "mean of {name}");
    }
}

#[test]
fn default_adder_has_one_slot_per_cpu() {
    let cpus = thread::available_parallelism().unwrap().get();

    assert_eq!(PaddedDoubleAdder::new().length(), cpus);
    assert_eq!(PaddedDoubleAdder::default().length(), cpus);
    assert_eq!(PaddedDoubleAdder::with_length(3).length(), 3);
}

#[test]
fn indexed_updates_hit_only_the_named_slot() {
    let adder = PaddedDoubleAdder::with_length(3);

    adder.increment_at(1);
    adder.increment_at(1);
    adder.decrement_at(2);
    adder.add_at(0, 40.5);

    assert_eq!(
        [adder.get(0), adder.get(1), adder.get(2)],
        [40.5, 2.0, -1.0]
    );
}

#[test]
fn reductions_match_upstream_with_bounds_checks_on() {
    assert_reductions(true);
}

#[test]
fn reductions_ignore_padding_when_bounds_checks_are_off() {
    // Upstream folded the zeroed padding slots into min, max and mean in this mode.
    assert_reductions(false);
}

#[test]
fn max_of_all_negative_slots_is_the_largest_negative_not_a_positive_sentinel() {
    let adder = adder(&[-4.0, -2.0, -8.0], true);

    assert_eq!(adder.max(), -2.0);
}

#[test]
fn reductions_of_an_adder_with_no_slots_use_the_identity_values() {
    let adder = PaddedDoubleAdder::with_length(0);

    assert_eq!(adder.sum(), 0.0);
    assert_eq!(adder.min(), f64::MAX);
    assert_eq!(adder.max(), f64::MIN);
    assert_eq!(adder.mean(), 0.0);
}

#[test]
fn random_updates_work_with_and_without_bounds_checks() {
    for bounds_check in [true, false] {
        let adder = PaddedDoubleAdder::with_options(8, bounds_check, false);

        for _ in 0..1_000 {
            adder.increment();
        }
        for _ in 0..200 {
            adder.decrement();
        }
        adder.add(5.5);

        assert_eq!(adder.sum(), 805.5, "bounds_check {bounds_check}");
    }
}

#[test]
fn random_updates_spread_over_the_slots() {
    let adder = PaddedDoubleAdder::with_length(8);

    for _ in 0..10_000 {
        adder.increment();
    }

    let used = (0..8).filter(|idx| adder.get(*idx) != 0.0).count();
    assert!(used > 1, "all updates landed in one slot");
}

#[test]
fn sum_and_reset_sums_finite_slots_and_zeroes_all_of_them() {
    for bounds_check in [true, false] {
        let adder = adder(&[5.5, f64::NAN, 7.25], bounds_check);

        assert_eq!(adder.sum_and_reset().to_bits(), 0x4029800000000000);
        assert_eq!(adder.sum(), 0.0);
        assert!((0..3).all(|logical| {
            let idx = if bounds_check {
                logical
            } else {
                adder.physical_idx_for_raw(logical as i64)
            };
            adder.get(idx) == 0.0
        }));
    }
}

#[test]
fn reset_zeroes_every_slot_including_non_finite_ones() {
    let adder = adder(&[1.0, f64::INFINITY], true);

    adder.reset();

    assert_eq!((adder.get(0), adder.get(1)), (0.0, 0.0));
}

#[test]
fn concurrent_random_increments_are_not_lost() {
    let adder = PaddedDoubleAdder::new();
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

    assert_eq!(adder.sum(), (threads * per_thread) as f64);
}
