//! Queues split into independent partitions behind one handle pair (`Partitioned*Queue` upstream).
//!
//! Producers pick a partition; consumers visit them in order. There is no ordering between partitions.

use super::queue::{Consumer, Multi, Producer, Queue, Role, Single};
use super::queue_utils::unsigned_multiply_high;
use crate::atomics::thread_random;

/// Single producer, single consumer.
pub type PartitionedSpscQueue<T> = PartitionedQueue<T, Single, Single>;
/// Multiple producers, single consumer.
pub type PartitionedMpscQueue<T> = PartitionedQueue<T, Multi, Single>;
/// Single producer, multiple consumers.
pub type PartitionedSpmcQueue<T> = PartitionedQueue<T, Single, Multi>;
/// Multiple producers, multiple consumers.
pub type PartitionedMpmcQueue<T> = PartitionedQueue<T, Multi, Multi>;

/// An unbounded queue made of independent partitions.
///
/// Every partition is a separate [`Queue`] with its own allocation and its own padded state, so
/// partitions never share a cache line with each other.
pub struct PartitionedQueue<T, P: Role, C: Role> {
    queues: Vec<Queue<T, P, C>>,
}

impl<T, P: Role, C: Role> PartitionedQueue<T, P, C> {
    /// One partition whose chunks hold `chunk_size` slots, with no chunk pooling.
    pub fn new(chunk_size: usize) -> Self {
        Self::with_pool(1, chunk_size, 0)
    }

    /// `partitions` partitions, without chunk pooling.
    pub fn with_partitions(partitions: usize, chunk_size: usize) -> Self {
        Self::with_pool(partitions, chunk_size, 0)
    }

    /// `partitions` partitions, each keeping up to `max_pooled_chunks` retired chunks for reuse.
    ///
    /// # Panics
    ///
    /// Panics if `partitions` is zero.
    pub fn with_pool(partitions: usize, chunk_size: usize, max_pooled_chunks: usize) -> Self {
        assert!(partitions > 0, "partitions must be positive");
        Self {
            queues: (0..partitions)
                .map(|_| Queue::with_pool(chunk_size, max_pooled_chunks))
                .collect(),
        }
    }

    /// The queue behind one partition, for a producer or consumer that owns just that partition.
    pub fn partition(&self, partition: usize) -> &Queue<T, P, C> {
        &self.queues[partition]
    }

    /// A producer for every partition. For a [`Single`] producer this takes every partition's role.
    pub fn producer(&self) -> PartitionedProducer<T, P> {
        PartitionedProducer {
            producers: self.queues.iter().map(Queue::producer).collect(),
        }
    }

    /// A consumer for every partition. For a [`Single`] consumer this takes every partition's role.
    pub fn consumer(&self) -> PartitionedConsumer<T, C> {
        PartitionedConsumer {
            consumers: self.queues.iter().map(Queue::consumer).collect(),
        }
    }

    pub fn partitions(&self) -> usize {
        self.queues.len()
    }

    /// Always `u64::MAX`: partitions are unbounded.
    pub fn capacity(&self) -> u64 {
        u64::MAX
    }

    /// Retired chunks each partition may keep.
    pub fn max_pooled_chunks(&self) -> usize {
        self.queues[0].max_pooled_chunks()
    }

    /// Reserved slots over every partition.
    pub fn size_long(&self) -> u64 {
        self.queues
            .iter()
            .fold(0, |total, queue| total.saturating_add(queue.size_long()))
    }

    /// Reserved slots in one partition.
    pub fn size(&self, partition: usize) -> u64 {
        self.queues[partition].size_long()
    }

    pub fn is_empty(&self) -> bool {
        self.queues.iter().all(Queue::is_empty)
    }

    pub fn is_empty_partition(&self, partition: usize) -> bool {
        self.queues[partition].is_empty()
    }
}

/// The producing side of a [`PartitionedQueue`].
pub struct PartitionedProducer<T, P: Role> {
    producers: Vec<Producer<T, P>>,
}

macro_rules! partitioned_producer {
    ($role:ty, $recv:ty) => {
        impl<T> PartitionedProducer<T, $role> {
            /// Appends `item` to the named partition.
            pub fn offer_to(self: $recv, partition: usize, item: T) -> Result<(), T> {
                self.producers[partition].offer(item)
            }

            /// Appends `item` to the partition `seed` maps to: the same seed always picks the same one.
            pub fn offer_seeded(self: $recv, seed: u64, item: T) -> Result<(), T> {
                let partition = unsigned_multiply_high(seed, self.producers.len() as u64) as usize;
                self.producers[partition].offer(item)
            }

            /// Appends `item` to a randomly chosen partition.
            pub fn offer(self: $recv, item: T) -> Result<(), T> {
                let seed = thread_random::next_u64();
                let partition = unsigned_multiply_high(seed, self.producers.len() as u64) as usize;
                self.producers[partition].offer(item)
            }
        }
    };
}

partitioned_producer!(Single, &mut Self);
partitioned_producer!(Multi, &Self);

impl<T> Clone for PartitionedProducer<T, Multi> {
    fn clone(&self) -> Self {
        Self {
            producers: self.producers.clone(),
        }
    }
}

/// The consuming side of a [`PartitionedQueue`].
pub struct PartitionedConsumer<T, C: Role> {
    consumers: Vec<Consumer<T, C>>,
}

macro_rules! partitioned_consumer {
    ($role:ty, $recv:ty) => {
        impl<T> PartitionedConsumer<T, $role> {
            /// Removes the oldest published element of one partition.
            pub fn poll_partition(self: $recv, partition: usize) -> Option<T> {
                self.consumers[partition].poll()
            }

            /// Removes the oldest published element of the first partition that has one.
            pub fn poll(self: $recv) -> Option<T> {
                for index in 0..self.consumers.len() {
                    if let Some(item) = self.consumers[index].poll() {
                        return Some(item);
                    }
                }
                None
            }

            /// Hands up to `limit` elements to `consume`, partition by partition, and returns how many.
            pub fn drain(self: $recv, consume: impl FnMut(T), limit: u64) -> u64 {
                self.drain_while(consume, |_| false, limit)
            }

            /// [`drain`](Self::drain) without a limit.
            pub fn drain_all(self: $recv, consume: impl FnMut(T)) -> u64 {
                self.drain(consume, u64::MAX)
            }

            /// Like [`drain`](Self::drain). Each partition stops at its first element for which `stop`
            /// returns true, leaving it queued, and the budget carries on into the next partition.
            pub fn drain_while(
                self: $recv,
                mut consume: impl FnMut(T),
                mut stop: impl FnMut(&T) -> bool,
                limit: u64,
            ) -> u64 {
                let mut total = 0;
                for index in 0..self.consumers.len() {
                    total +=
                        self.consumers[index].drain_while(&mut consume, &mut stop, limit - total);
                    if total == limit {
                        break;
                    }
                }
                total
            }

            /// Like [`drain`](Self::drain), but starts at `start_partition` (wrapped to the partition
            /// count) and visits each partition once.
            pub fn drain_from(
                self: $recv,
                mut consume: impl FnMut(T),
                limit: u64,
                start_partition: usize,
            ) -> u64 {
                let partitions = self.consumers.len();
                let mut index = start_partition % partitions;
                let mut total = 0;
                for _ in 0..partitions {
                    if total >= limit {
                        break;
                    }
                    total += self.consumers[index].drain(&mut consume, limit - total);
                    index = (index + 1) % partitions;
                }
                total
            }

            /// Hands up to `limit` elements of one partition to `consume`.
            pub fn drain_partition(
                self: $recv,
                partition: usize,
                consume: impl FnMut(T),
                limit: u64,
            ) -> u64 {
                self.consumers[partition].drain(consume, limit)
            }

            /// [`drain_partition`](Self::drain_partition) with a stop condition.
            pub fn drain_partition_while(
                self: $recv,
                partition: usize,
                consume: impl FnMut(T),
                stop: impl FnMut(&T) -> bool,
                limit: u64,
            ) -> u64 {
                self.consumers[partition].drain_while(consume, stop, limit)
            }

            /// Removes every published element of every partition.
            pub fn clear(self: $recv) {
                for index in 0..self.consumers.len() {
                    self.consumers[index].clear();
                }
            }

            /// Removes every published element of one partition.
            pub fn clear_partition(self: $recv, partition: usize) {
                self.consumers[partition].clear()
            }
        }
    };
}

partitioned_consumer!(Single, &mut Self);
partitioned_consumer!(Multi, &Self);

impl<T> PartitionedConsumer<T, Single> {
    /// The oldest published element of one partition, left in place.
    pub fn peek_partition(&mut self, partition: usize) -> Option<&T> {
        self.consumers[partition].peek()
    }

    /// The oldest published element of the first partition that has one, left in place.
    pub fn peek(&mut self) -> Option<&T> {
        let index =
            (0..self.consumers.len()).find(|&index| self.consumers[index].peek().is_some())?;
        self.consumers[index].peek()
    }
}

impl<T> PartitionedConsumer<T, Multi> {
    /// Runs `look` on the oldest published element of one partition.
    pub fn peek_partition_with<R>(
        &self,
        partition: usize,
        look: impl FnOnce(&T) -> R,
    ) -> Option<R> {
        self.consumers[partition].peek_with(look)
    }

    /// Runs `look` on the oldest published element of the first partition that has one.
    pub fn peek_with<R>(&self, look: impl FnOnce(&T) -> R) -> Option<R> {
        // `look` runs at most once: only a partition that has an element calls it.
        let mut look = Some(look);
        for consumer in &self.consumers {
            let found = consumer.peek_with(|item| look.take().expect("look is called once")(item));
            if found.is_some() {
                return found;
            }
        }
        None
    }
}

impl<T> Clone for PartitionedConsumer<T, Multi> {
    fn clone(&self) -> Self {
        Self {
            consumers: self.consumers.clone(),
        }
    }
}
