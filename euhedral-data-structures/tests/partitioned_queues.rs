//! Partitioned queues: a fixed set of independent queues behind one handle pair. The shared behaviour
//! is run against all four flavours through the adapter traits below.
#![allow(
    clippy::disallowed_methods,
    clippy::disallowed_types,
    reason = "test fixtures: the test thread runs peer threads and waits on them and locks its records"
)]

use euhedral_data_structures::queues::{
    Multi, PartitionedConsumer, PartitionedMpmcQueue, PartitionedMpscQueue, PartitionedProducer,
    PartitionedQueue, PartitionedSpmcQueue, PartitionedSpscQueue, Single,
};
use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

fn deadline() -> Instant {
    Instant::now() + Duration::from_secs(20)
}

/// The partitioned producer operations, whatever the role's receiver type.
trait Put {
    fn put_to(&mut self, partition: usize, item: u64) -> Result<(), u64>;
    fn put_seeded(&mut self, seed: u64, item: u64) -> Result<(), u64>;
    fn put_anywhere(&mut self, item: u64) -> Result<(), u64>;
}

impl Put for PartitionedProducer<u64, Single> {
    fn put_to(&mut self, partition: usize, item: u64) -> Result<(), u64> {
        self.offer_to(partition, item)
    }

    fn put_seeded(&mut self, seed: u64, item: u64) -> Result<(), u64> {
        self.offer_seeded(seed, item)
    }

    fn put_anywhere(&mut self, item: u64) -> Result<(), u64> {
        self.offer(item)
    }
}

impl Put for PartitionedProducer<u64, Multi> {
    fn put_to(&mut self, partition: usize, item: u64) -> Result<(), u64> {
        self.offer_to(partition, item)
    }

    fn put_seeded(&mut self, seed: u64, item: u64) -> Result<(), u64> {
        self.offer_seeded(seed, item)
    }

    fn put_anywhere(&mut self, item: u64) -> Result<(), u64> {
        self.offer(item)
    }
}

/// The partitioned consumer operations, whatever the role's receiver type.
trait Take {
    fn take_from(&mut self, partition: usize) -> Option<u64>;
    fn take_any(&mut self) -> Option<u64>;
    fn look_in(&mut self, partition: usize) -> Option<u64>;
    fn look_any(&mut self) -> Option<u64>;
    fn drain_up_to(&mut self, limit: u64) -> Vec<u64>;
    fn drain_until(&mut self, stop: impl FnMut(&u64) -> bool, limit: u64) -> Vec<u64>;
    fn drain_starting_at(&mut self, start: usize, limit: u64) -> Vec<u64>;
    fn drain_one(&mut self, partition: usize, limit: u64) -> Vec<u64>;
    fn drain_one_until(
        &mut self,
        partition: usize,
        stop: impl FnMut(&u64) -> bool,
        limit: u64,
    ) -> Vec<u64>;
    fn wipe(&mut self);
    fn wipe_partition(&mut self, partition: usize);
}

macro_rules! impl_take {
    ($role:ty, $peek_in:expr, $peek_any:expr) => {
        impl Take for PartitionedConsumer<u64, $role> {
            fn take_from(&mut self, partition: usize) -> Option<u64> {
                self.poll_partition(partition)
            }

            fn take_any(&mut self) -> Option<u64> {
                self.poll()
            }

            fn look_in(&mut self, partition: usize) -> Option<u64> {
                $peek_in(self, partition)
            }

            fn look_any(&mut self) -> Option<u64> {
                $peek_any(self)
            }

            fn drain_up_to(&mut self, limit: u64) -> Vec<u64> {
                let mut out = Vec::new();
                self.drain(|v| out.push(v), limit);
                out
            }

            fn drain_until(&mut self, stop: impl FnMut(&u64) -> bool, limit: u64) -> Vec<u64> {
                let mut out = Vec::new();
                self.drain_while(|v| out.push(v), stop, limit);
                out
            }

            fn drain_starting_at(&mut self, start: usize, limit: u64) -> Vec<u64> {
                let mut out = Vec::new();
                self.drain_from(|v| out.push(v), limit, start);
                out
            }

            fn drain_one(&mut self, partition: usize, limit: u64) -> Vec<u64> {
                let mut out = Vec::new();
                self.drain_partition(partition, |v| out.push(v), limit);
                out
            }

            fn drain_one_until(
                &mut self,
                partition: usize,
                stop: impl FnMut(&u64) -> bool,
                limit: u64,
            ) -> Vec<u64> {
                let mut out = Vec::new();
                self.drain_partition_while(partition, |v| out.push(v), stop, limit);
                out
            }

            fn wipe(&mut self) {
                self.clear();
            }

            fn wipe_partition(&mut self, partition: usize) {
                self.clear_partition(partition);
            }
        }
    };
}

impl_take!(
    Single,
    |c: &mut PartitionedConsumer<u64, Single>, p| c.peek_partition(p).copied(),
    |c: &mut PartitionedConsumer<u64, Single>| c.peek().copied()
);
impl_take!(
    Multi,
    |c: &mut PartitionedConsumer<u64, Multi>, p| c.peek_partition_with(p, |v| *v),
    |c: &mut PartitionedConsumer<u64, Multi>| c.peek_with(|v| *v)
);

macro_rules! shared_tests {
    ($module:ident, $queue:ident) => {
        mod $module {
            use super::*;

            fn queue(partitions: usize) -> $queue<u64> {
                $queue::with_pool(partitions, 4, 2)
            }

            #[test]
            fn a_new_queue_reports_its_shape_and_is_empty() {
                let queue = queue(3);

                assert_eq!(queue.partitions(), 3);
                assert_eq!(queue.max_pooled_chunks(), 2);
                assert_eq!(queue.capacity(), u64::MAX);
                assert!(queue.is_empty());
                assert_eq!(queue.size_long(), 0);
                assert!((0..3).all(|p| queue.is_empty_partition(p) && queue.size(p) == 0));
            }

            #[test]
            fn a_queue_built_from_a_chunk_size_alone_has_one_unpooled_partition() {
                let queue = $queue::<u64>::new(8);

                assert_eq!(queue.partitions(), 1);
                assert_eq!(queue.max_pooled_chunks(), 0);
            }

            #[test]
            #[should_panic(expected = "partitions must be positive")]
            fn zero_partitions_are_rejected() {
                $queue::<u64>::with_pool(0, 4, 0);
            }

            #[test]
            fn queue_wide_operations_cover_every_partition() {
                // Port of the upstream queueWideOperationsCoverEveryPartition test.
                let queue = queue(3);
                let (mut producer, mut consumer) = (queue.producer(), queue.consumer());
                producer.put_to(2, 3).unwrap();
                producer.put_to(0, 1).unwrap();
                producer.put_to(1, 2).unwrap();

                assert_eq!(queue.size_long(), 3);
                assert_eq!(consumer.look_any(), Some(1));
                assert_eq!(consumer.take_any(), Some(1));
                assert_eq!(consumer.take_any(), Some(2));
                consumer.wipe();
                assert!(queue.is_empty());
                assert_eq!(consumer.take_any(), None);
            }

            #[test]
            fn offers_land_in_the_named_partition_only() {
                let queue = queue(3);
                let (mut producer, mut consumer) = (queue.producer(), queue.consumer());

                producer.put_to(1, 10).unwrap();
                producer.put_to(1, 11).unwrap();
                producer.put_to(2, 20).unwrap();

                assert_eq!([queue.size(0), queue.size(1), queue.size(2)], [0, 2, 1]);
                assert_eq!(consumer.look_in(1), Some(10));
                assert_eq!(consumer.take_from(0), None);
                assert_eq!(consumer.take_from(1), Some(10));
                assert_eq!(consumer.take_from(2), Some(20));
                assert!(queue.is_empty_partition(0) && !queue.is_empty_partition(1));
            }

            #[test]
            fn a_seed_picks_its_partition_by_multiply_high() {
                let queue = queue(4);
                let (mut producer, mut consumer) = (queue.producer(), queue.consumer());

                producer.put_seeded(0, 100).unwrap();
                producer.put_seeded(u64::MAX, 101).unwrap();
                producer.put_seeded(u64::MAX / 2 + 1, 102).unwrap();
                producer.put_seeded(u64::MAX / 4 + 1, 103).unwrap();
                producer.put_seeded(5, 104).unwrap();

                assert_eq!(consumer.drain_one(0, u64::MAX), [100, 104]);
                assert_eq!(consumer.drain_one(1, u64::MAX), [103]);
                assert_eq!(consumer.drain_one(2, u64::MAX), [102]);
                assert_eq!(consumer.drain_one(3, u64::MAX), [101]);
            }

            #[test]
            fn an_unseeded_offer_goes_to_exactly_one_partition() {
                let queue = queue(5);
                let (mut producer, mut consumer) = (queue.producer(), queue.consumer());

                for value in 0..200 {
                    producer.put_anywhere(value).unwrap();
                }

                assert_eq!(queue.size_long(), 200);
                let spread = (0..5).filter(|p| queue.size(*p) > 0).count();
                assert!(spread > 1, "200 random offers all landed in one partition");
                let mut all = consumer.drain_up_to(u64::MAX);
                all.sort_unstable();
                assert_eq!(all, (0..200).collect::<Vec<_>>());
            }

            #[test]
            fn clearing_one_partition_leaves_the_others_alone() {
                // Port of the upstream clearOnlyRemovesTheSelectedPartition test.
                let queue = queue(2);
                let (mut producer, mut consumer) = (queue.producer(), queue.consumer());
                producer.put_to(0, 1).unwrap();
                producer.put_to(0, 2).unwrap();
                producer.put_to(1, 3).unwrap();

                consumer.wipe_partition(0);

                assert!(queue.is_empty_partition(0));
                assert_eq!(queue.size(1), 1);
                assert_eq!(consumer.take_from(1), Some(3));
                assert!(queue.is_empty());
            }

            #[test]
            fn a_quiescent_partition_clear_does_not_replay_old_payloads_or_touch_others() {
                // Port of the upstream quiescentPartitionClearDoesNotReplayOldPayloads... test.
                let queue = queue(2);
                let (mut producer, mut consumer) = (queue.producer(), queue.consumer());
                producer.put_to(1, 999).unwrap();

                for generation in 0..4 {
                    for i in 0..32 {
                        producer.put_to(0, generation * 100 + i).unwrap();
                    }
                    consumer.wipe_partition(0);
                    assert!(queue.is_empty_partition(0));
                    assert_eq!(consumer.look_in(1), Some(999));

                    let expected: Vec<u64> = (0..32).map(|i| 5000 + generation * 100 + i).collect();
                    for value in &expected {
                        producer.put_to(0, *value).unwrap();
                    }
                    assert_eq!(consumer.drain_one(0, u64::MAX), expected);
                    assert_eq!(consumer.take_from(0), None);
                    assert_eq!(consumer.look_in(1), Some(999));
                }
                assert_eq!(consumer.take_from(1), Some(999));
                assert!(queue.is_empty());
                assert_eq!(queue.capacity(), u64::MAX);
            }

            #[test]
            fn drain_visits_partitions_in_order_up_to_the_limit() {
                // Port of the upstream drainsAcrossPartitionsUpToTheRequestedLimit test.
                let queue = queue(3);
                let (mut producer, mut consumer) = (queue.producer(), queue.consumer());
                producer.put_to(0, 1).unwrap();
                producer.put_to(1, 2).unwrap();
                producer.put_to(2, 3).unwrap();
                producer.put_to(2, 4).unwrap();

                assert_eq!(consumer.drain_up_to(3), [1, 2, 3]);
                assert_eq!(queue.size_long(), 1);
                assert_eq!(consumer.drain_up_to(0), Vec::<u64>::new());
            }

            #[test]
            fn the_whole_queue_budget_continues_past_a_partition_that_stopped() {
                // Port of the upstream wholeQueueBudgetContinuesPastAStoppedPartition... test.
                let queue = queue(3);
                let (mut producer, mut consumer) = (queue.producer(), queue.consumer());
                const STOPPED: u64 = 1000;
                producer.put_to(0, STOPPED).unwrap();
                producer.put_to(0, 1001).unwrap();
                producer.put_to(1, 2000).unwrap();
                producer.put_to(2, 3000).unwrap();
                producer.put_to(2, 3001).unwrap();

                let received = consumer.drain_until(|v| *v == STOPPED, 2);

                assert_eq!(received, [2000, 3000]);
                assert_eq!(
                    consumer.look_in(0),
                    Some(STOPPED),
                    "the stopped head was not consumed"
                );
                assert_eq!(queue.size(0), 2);
                assert_eq!(consumer.look_in(2), Some(3001));
                assert_eq!(queue.size_long(), 3);
            }

            #[test]
            fn a_stop_condition_applies_to_each_partition_separately() {
                let queue = queue(2);
                let (mut producer, mut consumer) = (queue.producer(), queue.consumer());
                for value in [1, 2, 3] {
                    producer.put_to(0, value).unwrap();
                }
                for value in [11, 12, 13] {
                    producer.put_to(1, value).unwrap();
                }

                let received = consumer.drain_until(|v| *v % 10 == 2, u64::MAX);

                assert_eq!(received, [1, 11]);
                assert_eq!(queue.size_long(), 4);
            }

            #[test]
            fn drain_from_starts_at_the_given_partition_and_wraps_once() {
                let queue = queue(3);
                let (mut producer, mut consumer) = (queue.producer(), queue.consumer());
                for partition in 0..3 {
                    for n in 0..2 {
                        producer
                            .put_to(partition, partition as u64 * 10 + n)
                            .unwrap();
                    }
                }

                assert_eq!(consumer.drain_starting_at(2, 3), [20, 21, 0]);
                assert_eq!(consumer.drain_starting_at(1, u64::MAX), [10, 11, 1]);
                assert_eq!(queue.size_long(), 0);
            }

            #[test]
            fn a_drain_start_past_the_last_partition_wraps_instead_of_failing() {
                // Upstream did not normalize the starting partition.
                let queue = queue(3);
                let (mut producer, mut consumer) = (queue.producer(), queue.consumer());
                for partition in 0..3 {
                    producer.put_to(partition, partition as u64).unwrap();
                }

                assert_eq!(consumer.drain_starting_at(7, 1), [1], "7 % 3 == 1");
                assert_eq!(consumer.drain_starting_at(usize::MAX, 1).len(), 1);
            }

            #[test]
            fn a_single_partition_drain_respects_its_limit_and_stop() {
                // Port of the upstream stopConditionLeavesTheRemainingValuesQueued test.
                let queue = queue(1);
                let (mut producer, mut consumer) = (queue.producer(), queue.consumer());
                for value in 0..6 {
                    producer.put_to(0, value).unwrap();
                }

                assert_eq!(consumer.drain_one_until(0, |v| *v == 2, 6), [0, 1]);
                assert_eq!(queue.size_long(), 4);
                assert_eq!(consumer.look_any(), Some(2));
                assert_eq!(consumer.drain_one(0, 2), [2, 3]);
            }

            #[test]
            fn queue_wide_poll_and_peek_search_from_partition_zero() {
                let queue = queue(3);
                let (mut producer, mut consumer) = (queue.producer(), queue.consumer());
                producer.put_to(2, 30).unwrap();
                producer.put_to(1, 20).unwrap();

                assert_eq!(consumer.look_any(), Some(20));
                assert_eq!(consumer.take_any(), Some(20));
                assert_eq!(consumer.take_any(), Some(30));
                assert_eq!(consumer.take_any(), None);
                assert_eq!(consumer.look_any(), None);
            }

            #[test]
            fn partitions_can_be_reached_directly_for_their_own_producer_and_consumer() {
                let queue = queue(2);

                #[allow(unused_mut)] // only the single-producer flavours need `mut`
                let mut producer = queue.partition(1).producer();
                producer.offer(7).unwrap();
                drop(producer);

                assert_eq!(queue.size(1), 1);
                #[allow(unused_mut)]
                let mut consumer = queue.partition(1).consumer();
                let mut drained = Vec::new();
                consumer.drain_all(|v| drained.push(v));
                assert_eq!(drained, [7]);
            }

            #[test]
            fn dropping_the_queue_drops_each_remaining_element_once() {
                struct Tracked(u64, Arc<Mutex<Vec<u64>>>);
                impl Drop for Tracked {
                    fn drop(&mut self) {
                        self.1.lock().unwrap().push(self.0);
                    }
                }

                let dropped = Arc::new(Mutex::new(Vec::new()));
                {
                    let queue: PartitionedQueue<Tracked, _, _> = $queue::with_pool(3, 4, 2);
                    for partition in 0..3 {
                        #[allow(unused_mut)]
                        let mut p = queue.partition(partition).producer();
                        for n in 0..20 {
                            let id = partition as u64 * 100 + n;
                            assert!(p.offer(Tracked(id, Arc::clone(&dropped))).is_ok());
                        }
                    }
                }

                let mut ids = dropped.lock().unwrap().clone();
                ids.sort_unstable();
                let mut expected: Vec<u64> = (0..3)
                    .flat_map(|p| (0..20).map(move |n| p * 100 + n))
                    .collect();
                expected.sort_unstable();
                assert_eq!(ids, expected);
            }
        }
    };
}

shared_tests!(spsc, PartitionedSpscQueue);
shared_tests!(mpsc, PartitionedMpscQueue);
shared_tests!(spmc, PartitionedSpmcQueue);
shared_tests!(mpmc, PartitionedMpmcQueue);

#[test]
#[should_panic(expected = "single producer handle is already in use")]
fn a_single_producer_queue_refuses_a_second_whole_queue_producer() {
    let queue = PartitionedSpscQueue::<u64>::with_partitions(2, 4);
    let _first = queue.producer();
    let _second = queue.producer();
}

#[test]
#[should_panic(expected = "single producer handle is already in use")]
fn a_partition_producer_conflicts_with_the_whole_queue_producer() {
    let queue = PartitionedSpscQueue::<u64>::with_partitions(2, 4);
    let _whole = queue.producer();
    let _one = queue.partition(1).producer();
}

#[test]
fn single_role_partitions_hand_their_roles_back_when_a_handle_is_dropped() {
    let queue = PartitionedSpscQueue::<u64>::with_partitions(2, 4);

    drop(queue.producer());
    drop(queue.consumer());
    let mut producer = queue.partition(0).producer();
    producer.offer(1).unwrap();
    drop(producer);

    assert_eq!(queue.consumer().poll(), Some(1));
}

#[test]
fn handles_for_multi_roles_are_cloneable_and_shareable() {
    fn assert_clone_send_sync<T: Clone + Send + Sync>() {}
    assert_clone_send_sync::<PartitionedProducer<String, Multi>>();
    assert_clone_send_sync::<PartitionedConsumer<String, Multi>>();

    fn assert_send<T: Send>() {}
    assert_send::<PartitionedProducer<String, Single>>();
    assert_send::<PartitionedConsumer<String, Single>>();
}

fn value(producer: u64, item: u64) -> u64 {
    producer << 32 | item
}

#[test]
fn spsc_one_producer_and_one_consumer_keep_order_across_chunk_rollover() {
    // Port of the upstream singleProducerAndConsumerPreserveOrderAcrossChunkRollover test.
    let count = 100_000_u64;
    let queue = PartitionedSpscQueue::with_pool(1, 8, 2);
    let (mut producer, mut consumer) = (queue.producer(), queue.consumer());

    thread::scope(|scope| {
        scope.spawn(move || {
            for item in 0..count {
                producer.offer_to(0, item).unwrap();
            }
        });
        scope.spawn(move || {
            let limit = deadline();
            let mut expected = 0;
            while expected < count {
                match consumer.poll_partition(0) {
                    Some(item) => {
                        assert_eq!(item, expected);
                        expected += 1;
                    }
                    None => {
                        assert!(Instant::now() < limit, "consumer timed out at {expected}");
                        std::hint::spin_loop();
                    }
                }
            }
        });
    });
}

#[test]
fn spsc_each_partition_has_its_own_producer_thread() {
    let (partitions, per_partition) = (4, 25_000_u64);
    let queue = PartitionedSpscQueue::with_pool(partitions, 8, 2);
    let mut consumer = queue.consumer();
    let mut received: Vec<Vec<u64>> = vec![Vec::new(); partitions];

    thread::scope(|scope| {
        for partition in 0..partitions {
            let mut producer = queue.partition(partition).producer();
            scope.spawn(move || {
                for item in 0..per_partition {
                    producer.offer(item).unwrap();
                }
            });
        }
        let limit = deadline();
        let total = partitions as u64 * per_partition;
        let mut count = 0;
        while count < total {
            let mut progressed = 0;
            for (partition, bucket) in received.iter_mut().enumerate() {
                progressed += consumer.drain_partition(partition, |v| bucket.push(v), 64);
            }
            count += progressed;
            if progressed == 0 {
                assert!(Instant::now() < limit, "consumer timed out at {count}");
                std::hint::spin_loop();
            }
        }
    });

    for bucket in &received {
        assert_eq!(*bucket, (0..per_partition).collect::<Vec<_>>());
    }
}

#[test]
fn mpsc_multiple_producers_on_one_partition_lose_and_duplicate_nothing() {
    // Port of the upstream multipleProducersDoNotLoseOrDuplicateValues test.
    let (producers, per_producer) = (4_u64, 10_000_u64);
    let queue = PartitionedMpscQueue::with_pool(1, 8, 2);
    let mut consumer = queue.consumer();
    let mut seen = HashSet::new();

    thread::scope(|scope| {
        for id in 0..producers {
            let producer = queue.producer();
            scope.spawn(move || {
                for item in 0..per_producer {
                    producer.offer_to(0, value(id, item)).unwrap();
                }
            });
        }
        let limit = deadline();
        while (seen.len() as u64) < producers * per_producer {
            let drained =
                consumer.drain_partition(0, |v| assert!(seen.insert(v), "duplicate {v:#x}"), 256);
            if drained == 0 {
                assert!(Instant::now() < limit, "consumer timed out");
                std::hint::spin_loop();
            }
        }
    });

    assert_eq!(seen.len() as u64, producers * per_producer);
}

#[test]
fn mpmc_random_partitions_with_many_producers_and_consumers_deliver_everything_once() {
    let (producers, consumers, partitions, per_producer) = (4_u64, 3, 4, 10_000_u64);
    let total = (producers * per_producer) as usize;
    let queue = PartitionedMpmcQueue::with_pool(partitions, 8, 2);
    let seen = Mutex::new(HashSet::new());
    let consumed = AtomicUsize::new(0);

    thread::scope(|scope| {
        for id in 0..producers {
            let producer = queue.producer();
            scope.spawn(move || {
                for item in 0..per_producer {
                    producer.offer(value(id, item)).unwrap();
                }
            });
        }
        for _ in 0..consumers {
            let consumer = queue.consumer();
            let (seen, consumed) = (&seen, &consumed);
            scope.spawn(move || {
                let limit = deadline();
                while consumed.load(Ordering::Relaxed) < total {
                    let mut batch = Vec::new();
                    if consumer.drain(|v| batch.push(v), 32) == 0 {
                        assert!(Instant::now() < limit, "consumer timed out");
                        std::hint::spin_loop();
                        continue;
                    }
                    consumed.fetch_add(batch.len(), Ordering::Relaxed);
                    let mut seen = seen.lock().unwrap();
                    for item in batch {
                        assert!(seen.insert(item), "duplicate {item:#x}");
                    }
                }
            });
        }
    });

    assert_eq!(seen.lock().unwrap().len(), total);
    assert!(queue.is_empty());
}

#[test]
fn spmc_many_consumers_share_one_partition_without_loss_or_duplication() {
    // Port of the upstream multipleConsumersDoNotLoseOrDuplicateValues test.
    let (consumers, count) = (4, 20_000_u64);
    let queue = PartitionedSpmcQueue::with_pool(1, 8, 2);
    let mut producer = queue.producer();
    for item in 0..count {
        producer.offer_to(0, item).unwrap();
    }
    let seen = Mutex::new(HashSet::new());
    let consumed = AtomicUsize::new(0);

    thread::scope(|scope| {
        for _ in 0..consumers {
            let consumer = queue.consumer();
            let (seen, consumed) = (&seen, &consumed);
            scope.spawn(move || {
                let limit = deadline();
                while consumed.load(Ordering::Relaxed) < count as usize {
                    match consumer.poll_partition(0) {
                        Some(item) => {
                            assert!(seen.lock().unwrap().insert(item), "duplicate {item}");
                            consumed.fetch_add(1, Ordering::Relaxed);
                        }
                        None => {
                            assert!(Instant::now() < limit, "consumer timed out");
                            std::hint::spin_loop();
                        }
                    }
                }
            });
        }
    });

    assert_eq!(seen.lock().unwrap().len() as u64, count);
    assert!(queue.is_empty());
}

/// Random-operation model: a `Vec<VecDeque>` that applies the documented semantics of every
/// partitioned operation, compared with the real queue after each step.
mod model {
    use super::*;
    use euhedral_data_structures::queues::Role;
    use euhedral_data_structures::queues::queue_utils::unsigned_multiply_high;
    use std::collections::VecDeque;

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
    }

    fn model_drain(
        model: &mut [VecDeque<u64>],
        partition: usize,
        mut stop: impl FnMut(&u64) -> bool,
        limit: u64,
    ) -> Vec<u64> {
        let mut out = Vec::new();
        while (out.len() as u64) < limit && model[partition].front().is_some_and(|v| !stop(v)) {
            out.push(model[partition].pop_front().unwrap());
        }
        out
    }

    fn run<P: Role, C: Role>(queue: PartitionedQueue<u64, P, C>, seed: u64, steps: usize)
    where
        PartitionedProducer<u64, P>: Put,
        PartitionedConsumer<u64, C>: Take,
    {
        let partitions = queue.partitions();
        let mut rng = Rng(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1);
        let (mut producer, mut consumer) = (queue.producer(), queue.consumer());
        let mut model: Vec<VecDeque<u64>> = vec![VecDeque::new(); partitions];
        let mut next = 0_u64;
        let ctx = |step: usize, op: &str| format!("seed {seed} step {step} op {op}");

        for step in 0..steps {
            match rng.below(10) {
                0..=2 => {
                    let partition = rng.below(partitions as u64) as usize;
                    producer.put_to(partition, next).unwrap();
                    model[partition].push_back(next);
                    next += 1;
                }
                3 => {
                    let seed = rng.next();
                    producer.put_seeded(seed, next).unwrap();
                    let partition = unsigned_multiply_high(seed, partitions as u64) as usize;
                    model[partition].push_back(next);
                    next += 1;
                }
                4 => {
                    let partition = rng.below(partitions as u64) as usize;
                    assert_eq!(
                        consumer.take_from(partition),
                        model[partition].pop_front(),
                        "{}",
                        ctx(step, "poll_partition")
                    );
                }
                5 => {
                    let expected = model.iter_mut().find_map(|q| q.pop_front());
                    assert_eq!(consumer.take_any(), expected, "{}", ctx(step, "poll"));
                }
                6 => {
                    let partition = rng.below(partitions as u64) as usize;
                    assert_eq!(
                        consumer.look_in(partition),
                        model[partition].front().copied(),
                        "{}",
                        ctx(step, "peek_partition")
                    );
                    let first = model.iter().find_map(|q| q.front().copied());
                    assert_eq!(consumer.look_any(), first, "{}", ctx(step, "peek"));
                }
                7 => {
                    let limit = rng.below(7);
                    let mut expected = Vec::new();
                    for partition in 0..partitions {
                        let remaining = limit - expected.len() as u64;
                        expected.extend(model_drain(&mut model, partition, |_| false, remaining));
                    }
                    assert_eq!(
                        consumer.drain_up_to(limit),
                        expected,
                        "{}",
                        ctx(step, "drain")
                    );
                }
                8 => {
                    let divisor = 2 + rng.below(4);
                    let limit = 1 + rng.below(9);
                    let mut expected = Vec::new();
                    for partition in 0..partitions {
                        let remaining = limit - expected.len() as u64;
                        expected.extend(model_drain(
                            &mut model,
                            partition,
                            |v| v % divisor == 0,
                            remaining,
                        ));
                    }
                    assert_eq!(
                        consumer.drain_until(|v| v % divisor == 0, limit),
                        expected,
                        "{}",
                        ctx(step, "drain_while")
                    );
                }
                _ => match rng.below(5) {
                    0 => {
                        let start = rng.below(partitions as u64 * 3) as usize;
                        let limit = rng.below(9);
                        let mut expected = Vec::new();
                        for offset in 0..partitions {
                            let partition = (start % partitions + offset) % partitions;
                            let remaining = limit.saturating_sub(expected.len() as u64);
                            expected.extend(model_drain(
                                &mut model,
                                partition,
                                |_| false,
                                remaining,
                            ));
                        }
                        assert_eq!(
                            consumer.drain_starting_at(start, limit),
                            expected,
                            "{}",
                            ctx(step, "drain_from")
                        );
                    }
                    1 => {
                        let partition = rng.below(partitions as u64) as usize;
                        let limit = rng.below(6);
                        let expected = model_drain(&mut model, partition, |_| false, limit);
                        assert_eq!(
                            consumer.drain_one(partition, limit),
                            expected,
                            "{}",
                            ctx(step, "drain_partition")
                        );
                    }
                    2 => {
                        let partition = rng.below(partitions as u64) as usize;
                        let expected = model_drain(&mut model, partition, |v| v % 3 == 0, 99);
                        assert_eq!(
                            consumer.drain_one_until(partition, |v| v % 3 == 0, 99),
                            expected,
                            "{}",
                            ctx(step, "drain_partition_while")
                        );
                    }
                    3 => {
                        let partition = rng.below(partitions as u64) as usize;
                        consumer.wipe_partition(partition);
                        model[partition].clear();
                    }
                    _ => {
                        if rng.below(3) == 0 {
                            consumer.wipe();
                            model.iter_mut().for_each(VecDeque::clear);
                        }
                    }
                },
            }
            let total: usize = model.iter().map(VecDeque::len).sum();
            assert_eq!(
                queue.size_long(),
                total as u64,
                "{}",
                ctx(step, "size_long")
            );
            for (partition, expected) in model.iter().enumerate() {
                assert_eq!(
                    queue.size(partition),
                    expected.len() as u64,
                    "{}",
                    ctx(step, "size")
                );
                assert_eq!(
                    queue.is_empty_partition(partition),
                    expected.is_empty(),
                    "{}",
                    ctx(step, "is_empty_partition")
                );
            }
        }
    }

    macro_rules! partitioned_model_tests {
        ($module:ident, $queue:ident) => {
            #[test]
            fn $module() {
                for partitions in [1, 2, 3, 5] {
                    for chunk_size in [2, 4] {
                        for max_pooled in [0, 2] {
                            for seed in 1..=25 {
                                run(
                                    $queue::with_pool(partitions, chunk_size, max_pooled),
                                    seed,
                                    500,
                                );
                            }
                        }
                    }
                }
            }
        };
    }

    partitioned_model_tests!(
        spsc_matches_a_model_over_random_operations,
        PartitionedSpscQueue
    );
    partitioned_model_tests!(
        mpsc_matches_a_model_over_random_operations,
        PartitionedMpscQueue
    );
    partitioned_model_tests!(
        spmc_matches_a_model_over_random_operations,
        PartitionedSpmcQueue
    );
    partitioned_model_tests!(
        mpmc_matches_a_model_over_random_operations,
        PartitionedMpmcQueue
    );
}
