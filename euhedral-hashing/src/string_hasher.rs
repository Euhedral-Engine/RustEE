//! xxHash64 over the UTF-16 code units of a string (`StringHasher` upstream).
//!
//! Java strings are sequences of UTF-16 code units, so hashing a `&str` encodes it to UTF-16 first.
//! That keeps hashes identical to the upstream hasher and to the byte hasher run over UTF-16LE bytes.
//! Lone surrogates, which a `&str` cannot hold, are reachable through the `utf16` functions.

use crate::primitives::{BASE_SEED, P1, P2, P5, merge, mix, round, tail1, tail2, tail3};

pub fn get_hash(s: &str) -> u64 {
    get_hash_with_seed(s, BASE_SEED)
}

pub fn get_hash_with_seed(s: &str, seed: u64) -> u64 {
    let units: Vec<u16> = s.encode_utf16().collect();
    get_hash_utf16_with_seed(&units, seed)
}

/// Hashes `s2` seeded with the hash of `s1`, so multi-part keys chain.
pub fn get_hash_pair(s1: &str, s2: &str) -> u64 {
    get_hash_with_seed(s2, get_hash(s1))
}

pub fn get_hash_utf16(units: &[u16]) -> u64 {
    get_hash_utf16_with_seed(units, BASE_SEED)
}

pub fn get_hash_utf16_with_seed(units: &[u16], seed: u64) -> u64 {
    let char_len = units.len();
    let byte_len = char_len * 2;
    let mut h64;
    let mut i = 0;

    if byte_len >= 32 {
        let mut v1 = seed.wrapping_add(P1).wrapping_add(P2);
        let mut v2 = seed.wrapping_add(P2);
        let mut v3 = seed;
        let mut v4 = seed.wrapping_sub(P1);

        let limit = char_len - 16;
        while i <= limit {
            v1 = round(v1, read_u64(units, i));
            v2 = round(v2, read_u64(units, i + 4));
            v3 = round(v3, read_u64(units, i + 8));
            v4 = round(v4, read_u64(units, i + 12));
            i += 16;
        }

        h64 = merge(v1, v2, v3, v4);
    } else {
        h64 = seed.wrapping_add(P5);
    }

    h64 = h64.wrapping_add(byte_len as u64);

    while i + 4 <= char_len {
        h64 = tail1(h64, read_u64(units, i));
        i += 4;
    }
    if i + 2 <= char_len {
        h64 = tail2(h64, read_u32(units, i));
        i += 2;
    }
    if i < char_len {
        let c = u64::from(units[i]);
        h64 = tail3(h64, c);
        h64 = tail3(h64, c >> 8);
    }

    mix(h64)
}

/// Four code units packed little-endian, the same lane layout as eight UTF-16LE bytes.
fn read_u64(units: &[u16], i: usize) -> u64 {
    u64::from(units[i])
        | (u64::from(units[i + 1]) << 16)
        | (u64::from(units[i + 2]) << 32)
        | (u64::from(units[i + 3]) << 48)
}

fn read_u32(units: &[u16], i: usize) -> u64 {
    u64::from(units[i]) | (u64::from(units[i + 1]) << 16)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::byte_hasher;

    const SEED: u64 = 0x1234_5678_9abc_def0;

    fn utf16le_bytes(s: &str) -> Vec<u8> {
        s.encode_utf16().flat_map(u16::to_le_bytes).collect()
    }

    /// `(text, hash with BASE_SEED, hash with SEED)` produced by the upstream Java `StringHasher`.
    const VECTORS: &[(&str, u64, u64)] = &[
        ("", 0xc4349fc93c010000, 0x7fef5b0c316777ed),
        ("a", 0xcac2d42e522e9abf, 0xa884869f94c37c49),
        ("ab", 0x2042878a0de58721, 0xdb2e4f4021d2cdcc),
        ("abc", 0x9439cdbcd0f6561c, 0x32bad8af53cd72c8),
        ("abcd", 0x5ec45db87c158d09, 0x246eae73c42485f3),
        ("abcdefgh", 0xd49cd97ef8631196, 0xbfa606b8fe434259),
        ("abcdefghijklmnop", 0x77cfa778dee9890c, 0x975ba279b2ba0c6c),
        ("abcdefghijklmnopq", 0x2b6c8bb14a33e63a, 0xa24fcdde1a5619ce),
        ("test-vector", 0x86f8f80d8b3e7058, 0x1dd3ee9f30b0cd11),
        (
            "shard-routing-key-123",
            0xade147feca8ae336,
            0x53263eca5ecb55dd,
        ),
        ("user_1000", 0x259bba69286fda34, 0xffa3a6c812d74a0e),
        ("user_1001", 0xe9d66b5c7bef4b1f, 0x22a39632e40fb283),
        (
            "The quick brown fox jumps over the lazy dog",
            0xe2b6e30d0a3d7d78,
            0xb35d0323bf7636c7,
        ),
        (
            "h\u{e9}llo w\u{f6}rld \u{fc}n\u{ef}code \u{2713}",
            0x609b4cd6aeb77249,
            0x8ed816b34c5439b7,
        ),
        (
            "\u{1F600}\u{1F600}\u{1F600}\u{1F600}\u{1F600}\u{1F600}\u{1F600}\u{1F600}\u{1F600}",
            0xfcabf4a9dd054685,
            0x2e6982023bdcfb5f,
        ),
        ("namespace", 0xcd9bf8aefd5678f6, 0x43415dce1faa18c0),
        ("entityId", 0xb5b68a944cb2c2f7, 0x8d497248ae819986),
    ];

    #[test]
    fn matches_upstream_vectors_with_default_seed() {
        for &(text, expected, _) in VECTORS {
            assert_eq!(get_hash(text), expected, "{text:?}");
        }
    }

    #[test]
    fn matches_upstream_vectors_with_explicit_seed() {
        for &(text, _, expected) in VECTORS {
            assert_eq!(get_hash_with_seed(text, SEED), expected, "{text:?}");
        }
    }

    #[test]
    fn equals_byte_hasher_over_utf16le_bytes() {
        for &(text, _, _) in VECTORS {
            let bytes = utf16le_bytes(text);
            assert_eq!(get_hash(text), byte_hasher::get_hash(&bytes), "{text:?}");
            assert_eq!(
                get_hash_with_seed(text, SEED),
                byte_hasher::get_hash_with_seed(&bytes, SEED),
                "{text:?}"
            );
        }
    }

    #[test]
    fn equals_byte_hasher_at_every_length_across_stripe_and_tail_boundaries() {
        for len in 0..=70 {
            let text: String = (0..len)
                .map(|i| char::from(b'a' + (i % 26) as u8))
                .collect();
            assert_eq!(
                get_hash(&text),
                byte_hasher::get_hash(&utf16le_bytes(&text)),
                "{len} chars"
            );
        }
    }

    #[test]
    fn pair_matches_upstream_vector_and_chains_the_seed() {
        assert_eq!(get_hash_pair("namespace", "entityId"), 0x41d2a58fd17ec22a);
        assert_eq!(get_hash_pair("entityId", "namespace"), 0x3101e4240ddcd867);
        assert_eq!(
            get_hash_pair("namespace", "entityId"),
            get_hash_with_seed("entityId", get_hash("namespace"))
        );
    }

    #[test]
    fn utf16_matches_upstream_for_lone_surrogates() {
        let cases: &[(&[u16], u64, u64)] = &[
            (
                &[0xd800, 0x0061, 0x0062],
                0x146af97a99007dbb,
                0x7c34b9b68e9709ae,
            ),
            (&[0xdc00], 0x14be179390249605, 0xadab44e00ec29d68),
            (
                &[0x0061, 0x0062, 0xd800],
                0x4afa44f7be88bcf2,
                0x1e169d651c5deeb8,
            ),
            (
                &[0xdbff, 0xdc00, 0x0078, 0xd800],
                0xab526632066a0307,
                0xa44fb3a70f870872,
            ),
        ];
        for &(units, default_seed, explicit_seed) in cases {
            assert_eq!(get_hash_utf16(units), default_seed, "{units:x?}");
            assert_eq!(
                get_hash_utf16_with_seed(units, SEED),
                explicit_seed,
                "{units:x?}"
            );
        }
    }

    #[test]
    fn utf16_matches_upstream_for_odd_tails_past_the_stripe_boundary() {
        let expected: &[(usize, u64)] = &[
            (15, 0x42196fef4153b4f0),
            (16, 0xfb9d2e0f201000de),
            (17, 0x5e6e779267ffacb4),
            (18, 0xbeb275adcd7adbae),
            (19, 0x0ef32cab3f0e80ac),
            (20, 0xb95d63be3e01121d),
            (21, 0x7cbfa0a7587d4c43),
            (31, 0x6c2bf7c1e329c767),
            (32, 0x5f3048a58e11347d),
            (33, 0x51684e3d2f019943),
        ];
        for &(count, hash) in expected {
            let units: Vec<u16> = (0..count).map(|i| (0x4e00 + i * 37) as u16).collect();
            assert_eq!(get_hash_utf16(&units), hash, "{count} units");
        }
    }

    #[test]
    fn str_hash_equals_utf16_hash_of_its_encoding() {
        let text = "mixed \u{e9} \u{1F600} text";
        let units: Vec<u16> = text.encode_utf16().collect();
        assert_eq!(get_hash(text), get_hash_utf16(&units));
    }
}
