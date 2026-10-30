# euhedral-hashing

xxHash64-based hashing for RustEE's routing keys, ordered-chain ids and vector fingerprints. It is a
port of the upstream Java `AbstractHasher`, `ByteHasher`, `StringHasher` and `HasherApi`, and gives
the same hashes as the Java code for the same input.

**Where it sits.** No dependencies. It hashes bytes, strings and float slices, and mixes and combines hashes
(`mix`, `mix_with_zero_check`, `combine`), which is what a caller needs to build a routing key, an ordered-chain
id or a vector fingerprint from parts. A hash is a function of its input and seed alone, on every platform.

## Modules

| Module | What it holds |
|---|---|
| [`hasher_api`](src/hasher_api.rs) | The entry point: `hash_bytes`, `hash_str`, `hash_f64s`, their `_with_seed` and `_pair` forms, `mix`, `mix_with_zero_check`, `combine` and `BASE_SEED`. Upstream overloads `getHash`; here each overload has its own name. |
| [`byte_hasher`](src/byte_hasher.rs) | XXH64 over a byte slice, words read little-endian. Plain XXH64 when the seed is `0`. |
| [`string_hasher`](src/string_hasher.rs) | XXH64 over the UTF-16 code units of a string, so a `&str` hashes like its UTF-16LE bytes. The `utf16` functions take raw code units, lone surrogates included. |

## Key concepts

- **One lane layout.** Every hasher shares the XXH64 lane layout. The default seed is
  `BASE_SEED` (`0x9e37_79b9_7f4a_7c15`); with seed `0` the byte hasher is plain XXH64.
- **Strings hash as UTF-16.** Java strings are UTF-16, so `hash_str(s)` equals `hash_bytes` over the
  UTF-16LE bytes of `s`. That keeps hashes identical to the Java port.
- **Multi-part keys chain.** The `_pair` functions hash the second part seeded with the hash of the
  first, so `hash_str_pair(a, b) == hash_str_with_seed(b, hash_str(a))`. Order matters.
- **`f64` vectors** hash their raw IEEE-754 bits without an intermediate buffer; the result equals
  `hash_bytes` over the little-endian bytes of the same values.
- **Mixing.** `mix` is the XXH64 avalanche step on its own. `mix_with_zero_check` never returns `0`
  (it returns `BASE_SEED` instead), for places where zero means "unset". `combine(h1, h2)` mixes
  two hashes into one.
- All arithmetic wraps, like Java `long`, and hashes are unsigned `u64`.

```rust
use euhedral_hashing::hasher_api::{
    BASE_SEED, hash_bytes, hash_str, hash_str_pair, hash_str_with_seed, mix_with_zero_check,
};

// A string hashes like its UTF-16LE bytes.
let utf16le: Vec<u8> = "key".encode_utf16().flat_map(u16::to_le_bytes).collect();
assert_eq!(hash_str("key"), hash_bytes(&utf16le));

// Multi-part keys chain: the second part is seeded with the hash of the first.
assert_eq!(
    hash_str_pair("namespace", "entityId"),
    hash_str_with_seed("entityId", hash_str("namespace")),
);

// Never zero, so it can seed a routing hash.
assert_ne!(mix_with_zero_check(0), 0);
assert_ne!(BASE_SEED, 0);
```

## Build and test

```sh
mise exec -- cargo test -p euhedral-hashing
```

The tests compare against reference values produced by the upstream Java hashers, at lengths that
cross every stripe, word, half-word and byte tail boundary. No features or special hardware are
needed. The example above is compiled and run as a doctest.
