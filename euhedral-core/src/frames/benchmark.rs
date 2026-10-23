use super::{Frame, FrameHeader};
use crate::internal::random;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

/// A frame that does nothing, for measuring the cost of scheduling itself.
#[derive(Debug)]
pub struct BenchmarkFrame {
    header: FrameHeader,
}

impl BenchmarkFrame {
    pub fn new(id_hash: u64, kill_switch: Option<Arc<AtomicBool>>) -> Self {
        Self {
            header: FrameHeader::with_kill_switch(id_hash, kill_switch),
        }
    }

    /// `count` frames sharing `id_hash` and `kill_switch`. Unless `ordered`, each frame gets its own
    /// routing hash from a random seed.
    pub fn generate(
        count: usize,
        ordered: bool,
        id_hash: u64,
        kill_switch: Option<Arc<AtomicBool>>,
    ) -> Vec<Self> {
        Self::generate_seeded(count, ordered, id_hash, random::next_u64(), kill_switch)
    }

    /// Like [`generate`](Self::generate), but frame `i` is randomized with `routing_seed + i`.
    pub fn generate_seeded(
        count: usize,
        ordered: bool,
        id_hash: u64,
        routing_seed: u64,
        kill_switch: Option<Arc<AtomicBool>>,
    ) -> Vec<Self> {
        (0..count as u64)
            .map(|i| {
                let mut frame = Self::new(id_hash, kill_switch.clone());
                if !ordered {
                    frame.header.randomize_hash(routing_seed.wrapping_add(i));
                }
                frame
            })
            .collect()
    }
}

impl Frame for BenchmarkFrame {
    fn header(&self) -> &FrameHeader {
        &self.header
    }

    fn header_mut(&mut self) -> &mut FrameHeader {
        &mut self.header
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use euhedral_hashing::hasher_api::{self, mix};
    use std::sync::atomic::Ordering;

    #[test]
    fn unordered_frames_use_the_supplied_seed() {
        let frames = BenchmarkFrame::generate_seeded(4, false, 0x1234, 0xabc, None);
        assert_eq!(frames.len(), 4);
        for (i, frame) in frames.iter().enumerate() {
            assert_eq!(frame.header().id_hash(), 0x1234);
            assert_eq!(
                frame.header().routing_hash(),
                0x1234 ^ mix(0xabc + i as u64)
            );
        }
    }

    #[test]
    fn ordered_frames_keep_their_id_hash() {
        for frame in BenchmarkFrame::generate_seeded(3, true, 0x5555, 0xabc, None) {
            assert_eq!(frame.header().id_hash(), 0x5555);
            assert_eq!(frame.header().routing_hash(), 0x5555);
        }
    }

    #[test]
    fn every_unordered_frame_is_unordered_even_from_seed_zero() {
        let frames = BenchmarkFrame::generate_seeded(3, false, 7, 0, None);
        assert!(frames.iter().all(|frame| !frame.header().is_ordered()));
    }

    #[test]
    fn the_seed_wraps_instead_of_overflowing() {
        let frames = BenchmarkFrame::generate_seeded(2, false, 7, u64::MAX, None);
        assert_eq!(frames[0].header().routing_hash(), 7 ^ mix(u64::MAX));
        assert_eq!(
            frames[1].header().routing_hash(),
            7 ^ hasher_api::mix_with_zero_check(0)
        );
    }

    #[test]
    fn random_seeds_spread_unordered_frames() {
        let frames = BenchmarkFrame::generate(8, false, 7, None);
        let mut hashes: Vec<u64> = frames.iter().map(|f| f.header().routing_hash()).collect();
        hashes.sort_unstable();
        hashes.dedup();
        assert_eq!(hashes.len(), 8);
        assert!(frames.iter().all(|frame| !frame.header().is_ordered()));
    }

    #[test]
    fn frames_share_the_kill_switch() {
        let kill_switch = Arc::new(AtomicBool::new(false));
        let frames = BenchmarkFrame::generate_seeded(2, false, 7, 9, Some(kill_switch.clone()));
        assert_eq!(frames[0].header().routing_hash(), 7 ^ mix(9));
        assert!(frames.iter().all(|frame| frame.is_alive()));
        frames[0].kill();
        assert!(kill_switch.load(Ordering::Acquire));
        assert!(frames.iter().all(|frame| !frame.is_alive()));
    }

    #[test]
    fn zero_frames_is_empty() {
        assert!(BenchmarkFrame::generate(0, false, 7, None).is_empty());
    }
}
