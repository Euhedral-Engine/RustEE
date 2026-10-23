//! xxHash64 building blocks shared by every hasher (`AbstractHasher` upstream).
//!
//! All arithmetic wraps, matching Java `long` overflow semantics, and hashes are unsigned so `>>>`
//! becomes a plain `>>`.

pub const BASE_SEED: u64 = 0x9e37_79b9_7f4a_7c15;

// xxHash64 primes
pub(crate) const P1: u64 = 0x9E37_79B1_85EB_CA87;
pub(crate) const P2: u64 = 0xC2B2_AE3D_27D4_EB4F;
pub(crate) const P3: u64 = 0x1656_67B1_9E37_79F9;
pub(crate) const P4: u64 = 0x85EB_CA77_C2B2_AE63;
pub(crate) const P5: u64 = 0x27D4_EB2F_1656_67C5;

pub(crate) fn merge(v1: u64, v2: u64, v3: u64, v4: u64) -> u64 {
    let mut h64 = v1
        .rotate_left(1)
        .wrapping_add(v2.rotate_left(7))
        .wrapping_add(v3.rotate_left(12))
        .wrapping_add(v4.rotate_left(18));
    h64 = merge_round(h64, v1);
    h64 = merge_round(h64, v2);
    h64 = merge_round(h64, v3);
    merge_round(h64, v4)
}

fn merge_round(acc: u64, val: u64) -> u64 {
    (acc ^ round(0, val)).wrapping_mul(P1).wrapping_add(P4)
}

pub(crate) fn round(acc: u64, input: u64) -> u64 {
    acc.wrapping_add(input.wrapping_mul(P2))
        .rotate_left(31)
        .wrapping_mul(P1)
}

pub(crate) fn tail1(h64: u64, val: u64) -> u64 {
    (h64 ^ round(0, val))
        .rotate_left(27)
        .wrapping_mul(P1)
        .wrapping_add(P4)
}

pub(crate) fn tail2(h64: u64, val: u64) -> u64 {
    (h64 ^ (val & 0xFFFF_FFFF).wrapping_mul(P1))
        .rotate_left(23)
        .wrapping_mul(P2)
        .wrapping_add(P3)
}

pub(crate) fn tail3(h64: u64, val: u64) -> u64 {
    (h64 ^ (val & 0xFF).wrapping_mul(P5))
        .rotate_left(11)
        .wrapping_mul(P1)
}

pub(crate) fn mix(hash: u64) -> u64 {
    let mut h = hash;
    h ^= h >> 33;
    h = h.wrapping_mul(P2);
    h ^= h >> 29;
    h = h.wrapping_mul(P3);
    h ^ (h >> 32)
}

pub fn combine(h1: u64, h2: u64) -> u64 {
    mix(h1 ^ h2.wrapping_mul(P1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constants_match_xxhash64_primes_and_base_seed() {
        assert_eq!(P1, 11400714785074694791);
        assert_eq!(P2, 14029467366897019727);
        assert_eq!(P3, 1609587929392839161);
        assert_eq!(P4, 9650029242287828579);
        assert_eq!(P5, 2870177450012600261);
        assert_eq!(BASE_SEED, 11400714819323198485);
    }

    #[test]
    fn round_matches_upstream_vectors() {
        assert_eq!(round(0, 0), 0);
        assert_eq!(round(0, 1), 0x7f0e345be3649cd2);
        assert_eq!(
            round(0x0123456789abcdef, 0xfedcba9876543210),
            0x9e9fa3bea5c7bcd2
        );
        assert_eq!(round(u64::MAX, 0x8000000000000000), 0x004d93acba143579);
    }

    #[test]
    fn round_accumulates_into_acc() {
        assert_ne!(round(1, 5), round(0, 5));
    }

    #[test]
    fn merge_matches_upstream_vectors() {
        assert_eq!(merge(1, 2, 3, 4), 0xcbd45f8da9d14190);
        assert_eq!(merge(0, 0, 0, 0), 0x39a1c7c279a349b0);
        assert_eq!(
            merge(
                u64::MAX,
                0x8000000000000000,
                0x7fffffffffffffff,
                0x0123456789abcdef
            ),
            0xa5141b67d5933951
        );
    }

    #[test]
    fn tail1_matches_upstream_vectors() {
        assert_eq!(tail1(0, 0), 0x85ebca77c2b2ae63);
        assert_eq!(tail1(0, 1), 0x652e3bed4e166ed1);
        assert_eq!(
            tail1(0x0123456789abcdef, 0xfedcba9876543210),
            0x1c861d5b0f222540
        );
        assert_eq!(tail1(u64::MAX, 0x8000000000000000), 0x8903e078faa5a249);
    }

    #[test]
    fn tail2_matches_upstream_vectors_and_uses_only_low_32_bits() {
        let h = 0x0123456789abcdef;
        assert_eq!(tail2(h, 0x00000000deadbeef), 0x1937a47a5efb15cb);
        assert_eq!(tail2(h, 0xffffffffdeadbeef), 0x1937a47a5efb15cb);
        assert_eq!(tail2(h, 0xab), 0x3d52d62b2abc6f53);
        assert_eq!(tail2(h, 0x1234567812345fab), 0x2a040262f3a1cd8e);
        assert_eq!(tail2(0, u64::MAX), 0xc412dd1a87da8451);
    }

    #[test]
    fn tail3_matches_upstream_vectors_and_uses_only_low_8_bits() {
        let h = 0x0123456789abcdef;
        assert_eq!(tail3(h, 0x00000000deadbeef), 0x7a8263d9ab8e8bc1);
        assert_eq!(tail3(h, 0xffffffffdeadbeef), 0x7a8263d9ab8e8bc1);
        assert_eq!(tail3(h, 0xab), 0xa08358087c90e9b0);
        assert_eq!(tail3(h, 0x1234567812345fab), 0xa08358087c90e9b0);
        assert_eq!(tail3(0, u64::MAX), 0x6f974583107fd1d8);
    }

    #[test]
    fn mix_matches_upstream_vectors() {
        assert_eq!(mix(0), 0);
        assert_eq!(mix(1), 0x283a72a5b9ab93d3);
        assert_eq!(mix(u64::MAX), 0x100ab62daadd9289);
        assert_eq!(mix(i64::MAX as u64), 0x88055b16556ec944);
        assert_eq!(mix(i64::MIN as u64), 0x21bccfd08328a27e);
        assert_eq!(mix(0x123456789abcdef0), 0xf3f28b660e840ce1);
    }

    #[test]
    fn combine_matches_upstream_vectors_and_is_order_sensitive() {
        assert_eq!(combine(1, 2), 0x653da2852392b6ee);
        assert_eq!(combine(2, 1), 0x3644b2eb76fefa1c);
        assert_eq!(combine(u64::MAX, i64::MIN as u64), 0x88055b16556ec944);
    }
}
