use euhedral_data_structures::atomics::{CachePadded, PaddedAtomicLong};
use std::mem::{align_of, size_of};

#[test]
fn cache_padded_occupies_a_full_128_byte_line() {
    assert_eq!(align_of::<CachePadded<u8>>(), 128);
    assert_eq!(size_of::<CachePadded<u8>>(), 128);
    assert_eq!(size_of::<CachePadded<[u8; 129]>>(), 256);
}

#[test]
fn cache_padded_derefs_to_the_wrapped_value() {
    let mut padded = CachePadded::new(5_u32);
    assert_eq!(*padded, 5);
    *padded += 1;
    assert_eq!(padded.0, 6);
}

#[test]
fn padded_atomics_never_share_a_128_byte_line_with_a_neighbour() {
    let pair = [PaddedAtomicLong::new(1), PaddedAtomicLong::new(2)];
    let a = &pair[0] as *const _ as usize;
    let b = &pair[1] as *const _ as usize;

    assert_eq!(a % 128, 0);
    assert_eq!(b - a, 128);
}
