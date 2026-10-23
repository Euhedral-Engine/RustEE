//! xxHash64-based hashing used for Euhedral routing keys, ordered-chain ids and vector fingerprints.
//!
//! Every hasher shares one lane layout: with seed `0` the byte hasher is plain XXH64, and the default
//! seed is [`hasher_api::BASE_SEED`]. Strings hash their UTF-16 code units, so a `&str` hashes the same
//! as its UTF-16LE bytes.

pub mod byte_hasher;
pub mod hasher_api;
pub mod string_hasher;

mod primitives;
