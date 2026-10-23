//! xxHash64 over a byte slice (`ByteHasher` upstream).
//!
//! Words are read little-endian. This is plain XXH64 when the seed is `0`.

use crate::primitives::{BASE_SEED, P1, P2, P5, merge, mix, round, tail1, tail2, tail3};

pub fn get_hash(data: &[u8]) -> u64 {
    get_hash_with_seed(data, BASE_SEED)
}

pub fn get_hash_with_seed(data: &[u8], seed: u64) -> u64 {
    let len = data.len();
    let mut h64;
    let mut i = 0;

    if len >= 32 {
        let mut v1 = seed.wrapping_add(P1).wrapping_add(P2);
        let mut v2 = seed.wrapping_add(P2);
        let mut v3 = seed;
        let mut v4 = seed.wrapping_sub(P1);

        let limit = len - 32;
        while i <= limit {
            v1 = round(v1, read_u64(data, i));
            v2 = round(v2, read_u64(data, i + 8));
            v3 = round(v3, read_u64(data, i + 16));
            v4 = round(v4, read_u64(data, i + 24));
            i += 32;
        }

        h64 = merge(v1, v2, v3, v4);
    } else {
        h64 = seed.wrapping_add(P5);
    }

    h64 = h64.wrapping_add(len as u64);

    while i + 8 <= len {
        h64 = tail1(h64, read_u64(data, i));
        i += 8;
    }
    if i + 4 <= len {
        h64 = tail2(h64, u64::from(read_u32(data, i)));
        i += 4;
    }
    while i < len {
        h64 = tail3(h64, u64::from(data[i]));
        i += 1;
    }

    mix(h64)
}

/// Hashes `data2` seeded with the hash of `data1`, so multi-part keys chain.
pub fn get_hash_pair(data1: &[u8], data2: &[u8]) -> u64 {
    get_hash_with_seed(data2, get_hash(data1))
}

fn read_u64(data: &[u8], i: usize) -> u64 {
    u64::from_le_bytes(data[i..i + 8].try_into().expect("slice is 8 bytes"))
}

fn read_u32(data: &[u8], i: usize) -> u32 {
    u32::from_le_bytes(data[i..i + 4].try_into().expect("slice is 4 bytes"))
}

#[cfg(test)]
mod tests {
    use super::*;

    const SEED: u64 = 0x1234_5678_9abc_def0;

    fn pattern(len: usize) -> Vec<u8> {
        (0..len).map(|i| (i * 31 + 7) as u8).collect()
    }

    /// `(length, hash with BASE_SEED, hash with SEED)` produced by the upstream Java `ByteHasher`
    /// on `pattern(length)`. The lengths cross every stripe, word, half-word and byte tail boundary.
    const VECTORS: &[(usize, u64, u64)] = &[
        (0, 0xc4349fc93c010000, 0x7fef5b0c316777ed),
        (1, 0x585882422a6165e7, 0x424f926440206cd7),
        (2, 0x641d1a59667ae544, 0xd4ff8e6e2b3431ba),
        (3, 0x5acb303e78133c22, 0x257677d35b1faf7f),
        (4, 0x7d51d5e2461732b3, 0x7d85e8506b3cb82a),
        (5, 0x7f8b72856a42bb63, 0x3f397c0af47ea2c7),
        (7, 0x2ce9adec2b2c8104, 0x3a3ba93bfa68b7b0),
        (8, 0x758848f033fa76a2, 0xa59181f19d10d83d),
        (9, 0xd4576cf554b7d929, 0x564ba7d73b30e64d),
        (12, 0xeaa6d4d803d21fd7, 0x05cdd8e4c3ac328b),
        (15, 0xa18d5c90d722cee3, 0xb94f7c0697b78139),
        (16, 0xe3594f9058b426e7, 0x9d09694cc958eb5e),
        (17, 0xa0c8a40ef8f3a9f7, 0x60ecce12a1d726c9),
        (31, 0x8137041f5af88413, 0xaa02010f95889a51),
        (32, 0x184ebcf3745cd46c, 0x5ce1d85d3d4d2ef2),
        (33, 0x52fac3c981f3cc2e, 0xc36086e70edaf8c4),
        (39, 0x5661126b7efaf137, 0x1aae5a47c1dcd837),
        (40, 0x36758d31812c0e57, 0xc7f8d3d7c77b7cf5),
        (63, 0x64ef99a2e94cc7bd, 0x69e4eaa16332b74a),
        (64, 0xf7f22435fe1ab128, 0x227c621649348fd3),
        (65, 0x7a0b76a812617c73, 0x80b1a6a914433694),
        (100, 0xbc7ab33be7528c18, 0xb54ec29e0056c27e),
    ];

    #[test]
    fn matches_upstream_vectors_with_default_seed() {
        for &(len, expected, _) in VECTORS {
            assert_eq!(get_hash(&pattern(len)), expected, "length {len}");
        }
    }

    #[test]
    fn matches_upstream_vectors_with_explicit_seed() {
        for &(len, _, expected) in VECTORS {
            assert_eq!(
                get_hash_with_seed(&pattern(len), SEED),
                expected,
                "length {len}"
            );
        }
    }

    #[test]
    fn default_seed_is_base_seed() {
        let data = pattern(45);
        assert_eq!(get_hash(&data), get_hash_with_seed(&data, BASE_SEED));
    }

    #[test]
    fn seed_zero_is_standard_xxh64() {
        // Published XXH64 reference values (seed 0).
        assert_eq!(get_hash_with_seed(b"", 0), 0xef46db3751d8e999);
        assert_eq!(get_hash_with_seed(b"a", 0), 0xd24ec4f1a98c6e5b);
        assert_eq!(get_hash_with_seed(b"abc", 0), 0x44bc2cf5ad770999);
        assert_eq!(
            get_hash_with_seed(b"Nobody inspects the spammish repetition", 0),
            0xfbcea83c8a378bf1
        );
    }

    #[test]
    fn pair_matches_upstream_vector_and_chains_the_seed() {
        let (x, y) = (pattern(13), pattern(40));
        assert_eq!(get_hash_pair(&x, &y), 0x189462b01774a6d1);
        assert_eq!(get_hash_pair(&y, &x), 0xcafc8cb1e7d2b0bc);
        assert_eq!(get_hash_pair(&x, &y), get_hash_with_seed(&y, get_hash(&x)));
    }

    #[test]
    fn seed_changes_the_hash() {
        let data = pattern(20);
        assert_ne!(get_hash_with_seed(&data, 1), get_hash_with_seed(&data, 2));
    }

    #[test]
    fn one_flipped_bit_changes_the_hash_at_every_tail_length() {
        for &(len, expected, _) in VECTORS.iter().filter(|v| v.0 > 0) {
            let mut data = pattern(len);
            data[len - 1] ^= 1;
            assert_ne!(get_hash(&data), expected, "length {len}");
        }
    }
}
