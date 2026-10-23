use euhedral_data_structures::atomics::PaddedAtomicDouble;
use std::mem::{align_of, size_of};

#[test]
fn starts_at_zero_by_default_and_at_the_given_value_otherwise() {
    assert_eq!(PaddedAtomicDouble::default().get(), 0.0);
    assert_eq!(PaddedAtomicDouble::new(2.5).get(), 2.5);
}

#[test]
fn exposes_the_full_atomic_double_api() {
    let value = PaddedAtomicDouble::new(1.0);

    assert_eq!(value.get_and_add(2.0), 1.0);
    assert!(value.compare_and_set(3.0, 4.0));
    assert_eq!(value.update_and_get(|v| v * 2.0), 8.0);
}

#[test]
fn occupies_a_full_cache_line_pair() {
    assert_eq!(align_of::<PaddedAtomicDouble>(), 128);
    assert_eq!(size_of::<PaddedAtomicDouble>(), 128);

    let pair = [PaddedAtomicDouble::new(1.0), PaddedAtomicDouble::new(2.0)];
    let a = &pair[0] as *const _ as usize;
    let b = &pair[1] as *const _ as usize;
    assert_eq!(b - a, 128);
}
