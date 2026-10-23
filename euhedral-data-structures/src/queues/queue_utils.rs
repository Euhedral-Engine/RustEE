//! Sizing and indexing helpers shared by the queues (`QueueUtils` upstream).

/// Largest ring a queue will allocate.
pub const MAX_SIZE: u64 = 1 << 30;

/// Counters advance by two per element, so the low bit is free for the multi-producer resize flag.
pub(crate) const SHIFT: u32 = 1;
pub(crate) const INCREMENT: u64 = 1 << SHIFT;
/// Set in the tail while one producer is installing a new chunk.
pub(crate) const RESIZING: u64 = INCREMENT >> 1;

/// The high 64 bits of the unsigned 128-bit product, used to map a random seed onto `0..n`.
pub fn unsigned_multiply_high(a: u64, b: u64) -> u64 {
    ((u128::from(a) * u128::from(b)) >> 64) as u64
}

/// Rounds a chunk size up to a power of two, capped at [`MAX_SIZE`].
///
/// # Panics
///
/// Panics if `chunk_size` is zero.
pub fn round_chunk_size(chunk_size: u64) -> u64 {
    assert!(chunk_size > 0, "chunk size must be positive");
    if chunk_size >= MAX_SIZE {
        return MAX_SIZE;
    }
    chunk_size.next_power_of_two()
}

/// The mask that wraps a counter onto a ring of `round_chunk_size(chunk_size)` slots, in counter units.
pub fn chunk_mask(chunk_size: u64) -> u64 {
    (round_chunk_size(chunk_size) - 1) << SHIFT
}
