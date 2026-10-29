//! Behaviour shared by the two multi-consumer flavours, run against both.
#![allow(
    clippy::disallowed_methods,
    clippy::disallowed_types,
    reason = "test fixtures: the test thread runs peer threads and waits on them and locks its records"
)]

use euhedral_data_structures::queues::{Consumer, MpmcQueue, Multi, Queue, Role, SpmcQueue};
use std::collections::HashSet;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

fn deadline() -> Instant {
    Instant::now() + Duration::from_secs(20)
}

/// Offers through whichever producer role the flavour has.
trait Feed {
    fn feed(&self, value: u64);
}

impl<C: Role> Feed for Queue<u64, euhedral_data_structures::queues::Single, C> {
    fn feed(&self, value: u64) {
        self.producer().offer(value).unwrap();
    }
}

impl<C: Role> Feed for Queue<u64, Multi, C> {
    fn feed(&self, value: u64) {
        self.producer().offer(value).unwrap();
    }
}

macro_rules! multi_consumer_tests {
    ($module:ident, $queue:ident) => {
        mod $module {
            use super::*;

            fn filled(chunk_size: usize, count: u64) -> $queue<u64> {
                let queue = $queue::new(chunk_size);
                for value in 0..count {
                    queue.feed(value);
                }
                queue
            }

            #[test]
            fn poll_returns_elements_in_order_and_none_when_empty() {
                let queue = filled(4, 20);
                let consumer = queue.consumer();

                for value in 0..20 {
                    assert_eq!(consumer.poll(), Some(value));
                }
                assert_eq!(consumer.poll(), None);
            }

            #[test]
            fn peek_with_looks_without_consuming() {
                let queue = filled(2, 6);
                let consumer = queue.consumer();

                assert_eq!(consumer.peek_with(|v| *v), Some(0));
                assert_eq!(consumer.peek_with(|v| *v + 100), Some(100));
                assert_eq!(consumer.poll(), Some(0));
                assert_eq!(consumer.peek_with(|v| *v), Some(1));
                consumer.clear();
                assert_eq!(consumer.peek_with(|v| *v), None);
            }

            #[test]
            fn drain_hands_over_up_to_the_limit_in_order() {
                let queue = filled(4, 10);
                let consumer = queue.consumer();
                let mut seen = Vec::new();

                assert_eq!(consumer.drain(|v| seen.push(v), 4), 4);
                assert_eq!(consumer.drain(|v| seen.push(v), 0), 0);
                assert_eq!(consumer.drain_all(|v| seen.push(v)), 6);

                assert_eq!(seen, (0..10).collect::<Vec<_>>());
            }

            #[test]
            fn the_stop_condition_leaves_the_remaining_values_queued() {
                // Port of the upstream PartitionedSpmcQueueTest.stopConditionLeavesTheRemainingValuesQueued.
                let queue = filled(4, 6);
                let consumer = queue.consumer();
                let mut drained = Vec::new();

                let count = consumer.drain_while(|v| drained.push(v), |v| *v == 2, 6);

                assert_eq!(count, 2);
                assert_eq!(drained, [0, 1]);
                assert_eq!(queue.size_long(), 4);
                assert_eq!(consumer.peek_with(|v| *v), Some(2));
            }

            #[test]
            fn drain_while_with_no_match_drains_across_rollovers() {
                let queue = filled(2, 100);
                let consumer = queue.consumer();
                let mut drained = Vec::new();

                assert_eq!(
                    consumer.drain_while(|v| drained.push(v), |v| *v > 1000, u64::MAX),
                    100
                );

                assert_eq!(drained, (0..100).collect::<Vec<_>>());
            }

            #[test]
            fn clear_removes_everything() {
                let queue = filled(4, 50);

                queue.consumer().clear();

                assert!(queue.is_empty());
                assert_eq!(queue.consumer().poll(), None);
            }

            #[test]
            fn a_batch_limit_is_applied_per_lock_hold_but_not_to_the_total() {
                let queue = $queue::new(4).with_max_consume_batch(3);
                assert_eq!(queue.max_consume_batch(), 3);
                for value in 0..20 {
                    queue.feed(value);
                }
                let consumer = queue.consumer();
                let mut seen = Vec::new();

                assert_eq!(consumer.drain(|v| seen.push(v), 10), 10);
                assert_eq!(consumer.drain_while(|v| seen.push(v), |_| false, 100), 10);

                assert_eq!(seen, (0..20).collect::<Vec<_>>());
            }

            #[test]
            fn progress_is_published_after_every_batch() {
                // The head only advances when a batch ends, which is also when the lock is released, so
                // the size seen from inside the callbacks drops once per batch of three.
                let queue = $queue::new(4).with_max_consume_batch(3);
                for value in 0..10 {
                    queue.feed(value);
                }
                let consumer = queue.consumer();
                let mut sizes = Vec::new();

                consumer.drain_all(|_| sizes.push(queue.size_long()));

                assert_eq!(sizes, [10, 10, 10, 7, 7, 7, 4, 4, 4, 1]);
            }

            #[test]
            fn the_default_batch_is_unlimited() {
                assert_eq!($queue::<u64>::new(4).max_consume_batch(), u64::MAX);
            }

            #[test]
            #[should_panic(expected = "max consume batch must be positive")]
            fn a_zero_batch_is_rejected() {
                $queue::<u64>::new(4).with_max_consume_batch(0);
            }

            #[test]
            fn consumers_are_cloneable_and_share_the_queue() {
                fn assert_clone_send_sync<T: Clone + Send + Sync>() {}
                assert_clone_send_sync::<Consumer<String, Multi>>();

                let queue = filled(4, 4);
                let first = queue.consumer();
                let second = first.clone();

                assert_eq!(first.poll(), Some(0));
                assert_eq!(second.poll(), Some(1));
                assert_eq!(queue.consumer().poll(), Some(2));
            }

            #[test]
            fn a_consumer_that_finds_another_one_active_gets_nothing_instead_of_waiting() {
                let queue = filled(4, 10);
                let busy = queue.consumer();
                let other = busy.clone();
                let mut observed = Vec::new();

                busy.drain(
                    |value| {
                        // This callback runs while `busy` holds the consumer lock.
                        observed.push((
                            value,
                            other.poll(),
                            other.peek_with(|v| *v),
                            other.drain_all(drop),
                        ));
                    },
                    3,
                );

                assert_eq!(observed.len(), 3);
                assert!(observed.iter().all(|(_, polled, peeked, drained)| {
                    polled.is_none() && peeked.is_none() && *drained == 0
                }));
                assert_eq!(other.poll(), Some(3), "the lock was released afterwards");
            }

            #[test]
            fn a_failed_lock_attempt_does_not_release_the_active_consumers_lock() {
                let queue = filled(4, 10);
                let busy = queue.consumer();
                let other = busy.clone();
                let mut stolen = Vec::new();

                busy.drain(
                    |_| {
                        assert_eq!(other.poll(), None, "fails: `busy` holds the lock");
                        // Had that failed attempt unlocked the queue, a second thread could now take it.
                        thread::scope(|scope| {
                            scope.spawn(|| stolen.push(other.poll())).join().unwrap();
                        });
                    },
                    3,
                );

                assert_eq!(stolen, [None, None, None]);
            }

            #[test]
            #[should_panic(expected = "clear called from inside a consumer callback")]
            fn clearing_from_inside_a_callback_panics_instead_of_deadlocking() {
                let queue = filled(4, 10);
                let consumer = queue.consumer();
                let same_handle = consumer.clone();

                consumer.drain_all(|_| same_handle.clear());
            }

            #[test]
            fn a_panicking_callback_releases_the_lock_and_keeps_the_queue_consistent() {
                let queue = filled(4, 10);
                let consumer = queue.consumer();

                let result = catch_unwind(AssertUnwindSafe(|| {
                    consumer.drain_all(|v| {
                        if v == 3 {
                            panic!("callback failed");
                        }
                    })
                }));

                assert!(result.is_err());
                assert_eq!(
                    consumer.poll(),
                    Some(4),
                    "the lock is free and element 3 is gone, not duplicated"
                );
                assert_eq!(queue.size_long(), 5);
            }

            #[test]
            fn many_consumers_neither_lose_nor_duplicate_elements() {
                // Port of the upstream multipleConsumersDoNotLoseOrDuplicateValues test.
                let (consumers, count) = (4, 20_000_u64);
                let queue = filled(8, count);
                let seen = Mutex::new(HashSet::new());
                let consumed = AtomicUsize::new(0);

                thread::scope(|scope| {
                    for _ in 0..consumers {
                        let consumer = queue.consumer();
                        let (seen, consumed) = (&seen, &consumed);
                        scope.spawn(move || {
                            let limit = deadline();
                            while consumed.load(Ordering::Relaxed) < count as usize {
                                match consumer.poll() {
                                    Some(value) => {
                                        assert!(
                                            seen.lock().unwrap().insert(value),
                                            "duplicate {value}"
                                        );
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

            #[test]
            fn consumers_that_drain_in_batches_neither_lose_nor_duplicate_elements() {
                let (consumers, count) = (4, 20_000_u64);
                let queue = $queue::with_pool(8, 2).with_max_consume_batch(16);
                for value in 0..count {
                    queue.feed(value);
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
                                let mut batch = Vec::new();
                                if consumer.drain(|v| batch.push(v), 64) == 0 {
                                    assert!(Instant::now() < limit, "consumer timed out");
                                    std::hint::spin_loop();
                                    continue;
                                }
                                consumed.fetch_add(batch.len(), Ordering::Relaxed);
                                let mut seen = seen.lock().unwrap();
                                for value in batch {
                                    assert!(seen.insert(value), "duplicate {value}");
                                }
                            }
                        });
                    }
                });

                assert_eq!(seen.lock().unwrap().len() as u64, count);
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
                    let queue: Queue<Tracked, _, Multi> = $queue::with_pool(4, 2);
                    #[allow(unused_mut)] // only the single-producer flavour needs `mut`
                    let mut producer = queue.producer();
                    for id in 0..40 {
                        assert!(producer.offer(Tracked(id, Arc::clone(&dropped))).is_ok());
                    }
                    let consumer = queue.consumer();
                    for _ in 0..15 {
                        consumer.poll();
                    }
                }

                let mut ids = dropped.lock().unwrap().clone();
                ids.sort_unstable();
                assert_eq!(ids, (0..40).collect::<Vec<_>>());
            }
        }
    };
}

multi_consumer_tests!(spmc, SpmcQueue);
multi_consumer_tests!(mpmc, MpmcQueue);

#[test]
fn spmc_one_producer_feeds_many_consumers_in_order_per_consumer() {
    let count = 100_000_u64;
    let queue = SpmcQueue::new(32);
    let mut producer = queue.producer();
    let consumed = AtomicUsize::new(0);

    thread::scope(|scope| {
        for _ in 0..3 {
            let consumer = queue.consumer();
            let consumed = &consumed;
            scope.spawn(move || {
                let limit = deadline();
                let mut last = None;
                while consumed.load(Ordering::Relaxed) < count as usize {
                    match consumer.poll() {
                        Some(value) => {
                            // Elements one consumer sees are always increasing: the queue is FIFO.
                            assert!(last.is_none_or(|previous| value > previous));
                            last = Some(value);
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
        scope.spawn(move || {
            for value in 0..count {
                producer.offer(value).unwrap();
            }
        });
    });
}

#[test]
fn mpmc_many_producers_and_many_consumers_deliver_everything_once() {
    // Port of the upstream PartitionedMpmcQueueTest.multipleProducersAndConsumersDoNotLoseOrDuplicateValues.
    let (producers, consumers, per_producer) = (4_u64, 4, 10_000_u64);
    let total = (producers * per_producer) as usize;
    let queue = MpmcQueue::with_pool(8, 2);
    let seen = Mutex::new(HashSet::new());
    let consumed = AtomicUsize::new(0);

    thread::scope(|scope| {
        for id in 0..producers {
            let producer = queue.producer();
            scope.spawn(move || {
                for n in 0..per_producer {
                    producer.offer(id << 32 | n).unwrap();
                }
            });
        }
        for _ in 0..consumers {
            let consumer = queue.consumer();
            let (seen, consumed) = (&seen, &consumed);
            scope.spawn(move || {
                let limit = deadline();
                while consumed.load(Ordering::Relaxed) < total {
                    match consumer.poll() {
                        Some(value) => {
                            assert!(seen.lock().unwrap().insert(value), "duplicate {value}");
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

    assert_eq!(seen.lock().unwrap().len(), total);
    assert!(queue.is_empty());
}

#[test]
fn a_bounded_multi_consumer_queue_keeps_its_capacity() {
    let queue = SpmcQueue::bounded(3);
    let mut producer = queue.producer();
    let consumer = queue.consumer();

    for value in 0..3 {
        producer.offer(value).unwrap();
    }
    assert_eq!(producer.offer(9), Err(9));
    assert_eq!(consumer.poll(), Some(0));
    assert_eq!(producer.offer(9), Ok(()));
    assert_eq!(queue.capacity(), 3);
}
