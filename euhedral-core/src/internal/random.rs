use std::cell::Cell;
use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hasher};

thread_local! {
    static STATE: Cell<u64> = Cell::new(seed());
}

/// A distinct, unpredictable starting point per thread; `RandomState` is the only randomness in std.
fn seed() -> u64 {
    RandomState::new().build_hasher().finish()
}

/// The next value of a per-thread SplitMix64 stream, standing in for `ThreadLocalRandom.nextLong()`.
pub(crate) fn next_u64() -> u64 {
    STATE.with(|state| {
        let next = state.get().wrapping_add(0x9e37_79b9_7f4a_7c15);
        state.set(next);
        let mut z = next;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    })
}
