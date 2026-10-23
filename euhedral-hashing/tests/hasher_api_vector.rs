use euhedral_hashing::hasher_api::{hash_bytes, hash_f64s, hash_f64s_with_seed};

const SEED: u64 = 0x1234_5678_9abc_def0;

fn le_bytes(values: &[f64]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|v| v.to_bits().to_le_bytes())
        .collect()
}

/// `i * 1.25 - 3.0` for `i in 0..len`, the vector shape the upstream reference values were taken on.
fn ramp(len: usize) -> Vec<f64> {
    (0..len).map(|i| i as f64 * 1.25 - 3.0).collect()
}

#[test]
fn hashes_double_vectors_without_changing_xxhash_layout() {
    let vector = [0.0, -0.0, 1.25, -9.5, f64::NAN, f64::INFINITY];

    assert_eq!(hash_f64s(&vector), hash_bytes(&le_bytes(&vector)));
}

#[test]
fn seeded_vector_hash_is_deterministic_and_order_sensitive() {
    let first = [1.0, 2.0, 3.0, 4.0];
    let second = [4.0, 3.0, 2.0, 1.0];
    let seed = 42;

    assert_eq!(
        hash_f64s_with_seed(&first, seed),
        hash_f64s_with_seed(&first, seed)
    );
    assert_ne!(
        hash_f64s_with_seed(&first, seed),
        hash_f64s_with_seed(&second, seed)
    );
}

#[test]
fn matches_upstream_vectors_for_every_stripe_and_tail_length() {
    // (length, default seed, explicit seed) from the upstream Java HasherApi.getHash(double[]).
    let expected: &[(usize, u64, u64)] = &[
        (0, 0xc4349fc93c010000, 0x7fef5b0c316777ed),
        (1, 0x2c7d29600a034aa6, 0xfd0c5d54f1919704),
        (2, 0xdbc6b5f42a32ef66, 0x1cb812ccb913f081),
        (3, 0xb4fc96e2ad1aeb2e, 0x84d541cbaa5caf23),
        (4, 0x1908a07c037f5474, 0x3e0a54d0c1249efe),
        (5, 0xfa3bb26cb8363e22, 0xbd5cdad4b6d70fb6),
        (6, 0x39fc976960dbfaa9, 0xbf577e35907119a9),
        (7, 0x5a378a0bb835a519, 0x7fb8387d9aead49c),
        (8, 0xa6292347210bc89f, 0xb0b838bc9bd73d0c),
        (9, 0x58cc482e7bd01c7c, 0xd5f44ccf08c0fafb),
    ];
    for &(len, default_seed, explicit_seed) in expected {
        let data = ramp(len);
        assert_eq!(hash_f64s(&data), default_seed, "length {len}");
        assert_eq!(
            hash_f64s_with_seed(&data, SEED),
            explicit_seed,
            "length {len}"
        );
    }
}

#[test]
fn matches_upstream_vector_for_special_values() {
    let vector = [0.0, -0.0, 1.25, -9.5, f64::NAN, f64::INFINITY];

    assert_eq!(hash_f64s(&vector), 0xff7db9b8d05ae256);
}

#[test]
fn matches_byte_hash_at_every_length_with_and_without_seed() {
    for len in 0..=17 {
        let data = ramp(len);
        let bytes = le_bytes(&data);
        assert_eq!(hash_f64s(&data), hash_bytes(&bytes), "length {len}");
        assert_eq!(
            hash_f64s_with_seed(&data, SEED),
            euhedral_hashing::hasher_api::hash_bytes_with_seed(&bytes, SEED),
            "length {len}"
        );
    }
}

#[test]
fn distinguishes_signed_zero_and_nan_payloads_by_raw_bits() {
    assert_ne!(hash_f64s(&[0.0]), hash_f64s(&[-0.0]));
    assert_ne!(
        hash_f64s(&[f64::from_bits(0x7ff8_0000_0000_0000)]),
        hash_f64s(&[f64::from_bits(0x7ff8_0000_0000_0001)])
    );
}
