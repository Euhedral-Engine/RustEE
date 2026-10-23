use euhedral_hashing::hasher_api::{
    BASE_SEED, combine, hash_bytes, hash_bytes_pair, hash_bytes_with_seed, hash_str, hash_str_pair,
    hash_str_with_seed, mix, mix_with_zero_check,
};

fn utf16le_bytes(s: &str) -> Vec<u8> {
    s.encode_utf16().flat_map(u16::to_le_bytes).collect()
}

fn pattern(len: usize) -> Vec<u8> {
    (0..len).map(|i| (i * 31 + 7) as u8).collect()
}

#[test]
fn same_input_produces_same_hash() {
    let input = "shard-routing-key-123";
    assert_eq!(hash_str(input), hash_str(input));
}

#[test]
fn nearby_inputs_produce_different_hashes() {
    assert_ne!(hash_str("user_1000"), hash_str("user_1001"));
}

#[test]
fn string_and_utf16le_byte_hashes_are_equal() {
    let input = "test-vector";
    let bytes = utf16le_bytes(input);

    assert_ne!(hash_str(input), 0);
    assert_ne!(hash_bytes(&bytes), 0);
    assert_eq!(hash_str(input), hash_bytes(&bytes));
}

#[test]
fn different_seeds_produce_different_hashes() {
    let input = "same-data";
    assert_ne!(
        hash_str_with_seed(input, 0x12345),
        hash_str_with_seed(input, 0x67890),
        "Different seeds must produce different hashes"
    );
}

#[test]
fn multi_part_hashing_is_chainable() {
    let (part1, part2) = ("namespace", "entityId");

    let manual = hash_str_with_seed(part2, hash_str(part1));
    let helper = hash_str_pair(part1, part2);

    assert_eq!(manual, helper);
    assert_eq!(helper, 0x41d2a58fd17ec22a);
}

#[test]
fn byte_pair_hashing_is_chainable_and_ordered() {
    let (x, y) = (pattern(13), pattern(40));

    assert_eq!(
        hash_bytes_pair(&x, &y),
        hash_bytes_with_seed(&y, hash_bytes(&x))
    );
    assert_ne!(hash_bytes_pair(&x, &y), hash_bytes_pair(&y, &x));
}

#[test]
fn byte_hashing_matches_upstream_at_every_tail_boundary() {
    // Upstream asserts ByteHasher == BufferHasher at these lengths; both produced these values.
    let expected: &[(usize, u64)] = &[
        (0, 0xc4349fc93c010000),
        (1, 0x585882422a6165e7),
        (3, 0x5acb303e78133c22),
        (4, 0x7d51d5e2461732b3),
        (7, 0x2ce9adec2b2c8104),
        (8, 0x758848f033fa76a2),
        (15, 0xa18d5c90d722cee3),
        (16, 0xe3594f9058b426e7),
        (31, 0x8137041f5af88413),
        (32, 0x184ebcf3745cd46c),
        (33, 0x52fac3c981f3cc2e),
        (63, 0x64ef99a2e94cc7bd),
        (64, 0xf7f22435fe1ab128),
        (65, 0x7a0b76a812617c73),
    ];
    for &(len, hash) in expected {
        assert_eq!(hash_bytes(&pattern(len)), hash, "length {len}");
    }
}

#[test]
fn default_seed_is_the_golden_ratio_constant() {
    assert_eq!(BASE_SEED, 0x9e3779b97f4a7c15);
    assert_eq!(hash_bytes(b"abc"), hash_bytes_with_seed(b"abc", BASE_SEED));
}

#[test]
fn mix_with_zero_check_never_returns_zero() {
    assert_ne!(mix_with_zero_check(0), 0);
    assert_ne!(mix_with_zero_check(i64::MAX as u64), 0);
}

#[test]
fn mix_with_zero_check_replaces_a_zero_mix_with_the_base_seed() {
    // mix is a bijection with mix(0) == 0, so zero is the only input that needs the fallback.
    assert_eq!(mix(0), 0);
    assert_eq!(mix_with_zero_check(0), BASE_SEED);
}

#[test]
fn mix_with_zero_check_equals_mix_for_nonzero_results() {
    for input in [
        1,
        u64::MAX,
        i64::MAX as u64,
        i64::MIN as u64,
        0x123456789abcdef0,
    ] {
        assert_eq!(mix_with_zero_check(input), mix(input));
    }
}

#[test]
fn mix_and_combine_match_upstream_through_the_public_api() {
    assert_eq!(mix(1), 0x283a72a5b9ab93d3);
    assert_eq!(mix(0x123456789abcdef0), 0xf3f28b660e840ce1);
    assert_eq!(combine(1, 2), 0x653da2852392b6ee);
    assert_eq!(combine(2, 1), 0x3644b2eb76fefa1c);
}
