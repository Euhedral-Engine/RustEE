# euhedral-hashing glossary

The implementation-level terms of [`euhedral-hashing`](README.md). What a user does with the hashes (give
related frames one id hash, spread unordered frames with a seed) is in the
[top-level glossary](../GLOSSARY.md). Paths are relative to this crate. Entries run from the base
function up to the forms built on it.

**XXH64, lane layout** ([`byte_hasher`](src/byte_hasher.rs)): every hasher in the crate shares the XXH64
lane layout and gives the same hashes as the Java hashers for the same input. With seed `0` the byte hasher
is plain XXH64. A hash is a function of its input and seed alone, on every platform. All arithmetic wraps,
and hashes are unsigned `u64`.

**BASE_SEED** ([`hasher_api::BASE_SEED`](src/hasher_api.rs)): the default seed, `0x9e37_79b9_7f4a_7c15`.
`mix_with_zero_check` returns it in place of zero.

**mix** ([`hasher_api::mix`](src/hasher_api.rs)): the XXH64 avalanche step on its own.

**mix_with_zero_check** ([`hasher_api::mix_with_zero_check`](src/hasher_api.rs)): `mix`, except that it
never returns `0` (it returns `BASE_SEED` instead), for places where zero means "unset".
`FrameHeader::randomize_hash` XORs it with the id hash, so a randomized routing hash never equals the id
hash.

**combine** ([`hasher_api::combine`](src/hasher_api.rs)): mixes two hashes into one. It is order-sensitive:
`combine(1, 2)` differs from `combine(2, 1)`.

**pair functions** ([`hasher_api`](src/hasher_api.rs)): `hash_bytes_pair` and `hash_str_pair` hash a
multi-part key by seeding the second part with the hash of the first, so
`hash_str_pair(a, b) == hash_str_with_seed(b, hash_str(a))`. Order matters.

**string hashing** ([`string_hasher`](src/string_hasher.rs)): a string hashes as UTF-16, so `hash_str(s)`
equals `hash_bytes` over the UTF-16LE bytes of `s`; that keeps the hashes identical to the Java port. The
`utf16` functions take raw code units, lone surrogates included. `hash_f64s` hashes the raw IEEE-754 bits
of a slice, equal to `hash_bytes` over the little-endian bytes of the same values.
