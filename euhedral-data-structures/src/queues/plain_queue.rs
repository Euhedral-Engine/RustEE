use super::queue::Sink;
use super::queue_utils::round_chunk_size;
use crate::atomics::CachePadded;
use std::collections::VecDeque;

/// A FIFO queue for one thread (`PlainQueue` upstream): the same batch API as the concurrent queues,
/// without any synchronisation.
///
/// Upstream backs it with the chunked ring, which never grows and ignores a drain's stop condition.
/// Here a `VecDeque` holds the elements. The queue's state still sits alone in a 128-byte block, with
/// the configuration in the next one, as upstream pads its head/tail/chunk holder.
#[repr(C)]
pub struct PlainQueue<T> {
    items: CachePadded<VecDeque<T>>,
    capacity: u64,
}

impl<T> PlainQueue<T> {
    /// An unbounded queue; `chunk_size` is a hint for how much room to reserve up front.
    pub fn new(chunk_size: usize) -> Self {
        let room = round_chunk_size(chunk_size.max(2) as u64) as usize;
        Self {
            items: CachePadded::new(VecDeque::with_capacity(room)),
            capacity: u64::MAX,
        }
    }

    /// A queue that holds exactly `capacity` elements.
    ///
    /// # Panics
    ///
    /// Panics if `capacity` is zero.
    pub fn bounded(capacity: usize) -> Self {
        assert!(capacity > 0, "capacity must be positive");
        Self {
            items: CachePadded::new(VecDeque::with_capacity(capacity)),
            capacity: capacity as u64,
        }
    }

    /// Appends `item`, or returns it if a bounded queue is full.
    pub fn offer(&mut self, item: T) -> Result<(), T> {
        if self.size_long() >= self.capacity {
            return Err(item);
        }
        self.items.push_back(item);
        Ok(())
    }

    /// Moves items from the front of `items` until the queue is full, returning how many.
    pub fn fill(&mut self, items: &mut Vec<T>) -> usize {
        let room = self.capacity.saturating_sub(self.size_long());
        let count = (items.len() as u64).min(room) as usize;
        self.items.extend(items.drain(..count));
        count
    }

    pub fn poll(&mut self) -> Option<T> {
        self.items.pop_front()
    }

    pub fn peek(&self) -> Option<&T> {
        self.items.front()
    }

    /// Hands up to `limit` elements to `consume`, oldest first, and returns how many.
    pub fn drain(&mut self, consume: impl FnMut(T), limit: u64) -> u64 {
        self.drain_while(consume, |_| false, limit)
    }

    pub fn drain_all(&mut self, consume: impl FnMut(T)) -> u64 {
        self.drain(consume, u64::MAX)
    }

    /// Like [`drain`](Self::drain), but stops before the first element for which `stop` returns true.
    pub fn drain_while(
        &mut self,
        mut consume: impl FnMut(T),
        mut stop: impl FnMut(&T) -> bool,
        limit: u64,
    ) -> u64 {
        let mut total = 0;
        while total < limit {
            match self.items.front() {
                Some(item) if !stop(item) => {}
                _ => break,
            }
            let item = self.items.pop_front().expect("the front was just seen");
            total += 1;
            consume(item);
        }
        total
    }

    pub fn clear(&mut self) {
        self.items.clear();
    }

    pub fn size_long(&self) -> u64 {
        self.items.len() as u64
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// `u64::MAX` for an unbounded queue.
    pub fn capacity(&self) -> u64 {
        self.capacity
    }
}

impl<T> Sink<T> for PlainQueue<T> {
    fn push(&mut self, item: T) -> Result<(), T> {
        self.offer(item)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{align_of, offset_of, size_of};

    #[test]
    fn its_state_is_padded_like_upstreams_queue_holder() {
        // Upstream surrounds head, tail and the chunk with 128 bytes of dead longs on each side.
        assert_eq!(align_of::<PlainQueue<u64>>(), 128);
        assert_eq!(size_of::<PlainQueue<u64>>() % 128, 0);
        assert_eq!(offset_of!(PlainQueue<u64>, items), 0);
        assert!(size_of::<VecDeque<u64>>() <= 128);
        assert!(
            offset_of!(PlainQueue<u64>, capacity) >= 128,
            "the configuration sits in a block of its own, away from the hot state"
        );
    }
}
