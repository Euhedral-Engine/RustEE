//! Public entry point for hashing (`HasherApi` upstream): byte slices, strings and `f64` vectors.
//!
//! Upstream overloads `getHash`; here each overload gets a distinct name.

use crate::primitives::{self, P1, P2, P5, merge, round, tail1};
use crate::{byte_hasher, string_hasher};

pub use crate::primitives::{BASE_SEED, combine};

pub fn hash_bytes(data: &[u8]) -> u64 {
    byte_hasher::get_hash(data)
}

pub fn hash_bytes_with_seed(data: &[u8], seed: u64) -> u64 {
    byte_hasher::get_hash_with_seed(data, seed)
}

pub fn hash_bytes_pair(data1: &[u8], data2: &[u8]) -> u64 {
    byte_hasher::get_hash_pair(data1, data2)
}

pub fn hash_str(s: &str) -> u64 {
    string_hasher::get_hash(s)
}

pub fn hash_str_with_seed(s: &str, seed: u64) -> u64 {
    string_hasher::get_hash_with_seed(s, seed)
}

pub fn hash_str_pair(s1: &str, s2: &str) -> u64 {
    string_hasher::get_hash_pair(s1, s2)
}

/// Hashes the raw IEEE-754 representation of an `f64` slice without allocating an intermediate byte
/// buffer. The result equals [`hash_bytes`] over the little-endian bytes of the same values.
pub fn hash_f64s(data: &[f64]) -> u64 {
    hash_f64s_with_seed(data, BASE_SEED)
}

/// Hashes the raw IEEE-754 representation of an `f64` slice with the supplied seed.
pub fn hash_f64s_with_seed(data: &[f64], seed: u64) -> u64 {
    let mut index = 0;
    let mut hash;

    if data.len() >= 4 {
        let mut v1 = seed.wrapping_add(P1).wrapping_add(P2);
        let mut v2 = seed.wrapping_add(P2);
        let mut v3 = seed;
        let mut v4 = seed.wrapping_sub(P1);

        let limit = data.len() - 4;
        while index <= limit {
            v1 = round(v1, data[index].to_bits());
            v2 = round(v2, data[index + 1].to_bits());
            v3 = round(v3, data[index + 2].to_bits());
            v4 = round(v4, data[index + 3].to_bits());
            index += 4;
        }
        hash = merge(v1, v2, v3, v4);
    } else {
        hash = seed.wrapping_add(P5);
    }

    hash = hash.wrapping_add(size_of_val(data) as u64);
    while index < data.len() {
        hash = tail1(hash, data[index].to_bits());
        index += 1;
    }
    primitives::mix(hash)
}

/// Like [`mix`], but never returns `0` so the result can be used where zero means "unset".
pub fn mix_with_zero_check(hash: u64) -> u64 {
    match mix(hash) {
        0 => BASE_SEED,
        mixed => mixed,
    }
}

pub fn mix(hash: u64) -> u64 {
    primitives::mix(hash)
}
