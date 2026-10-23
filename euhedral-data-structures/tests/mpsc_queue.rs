use euhedral_data_structures::queues::{Consumer, MpscQueue, Multi, Producer, Single};
use std::collections::HashMap;
use std::sync::mpsc;
use std::sync::{Arc, Barrier, Mutex};
use std::thread;
use std::time::{Duration, Instant};

/// Runs `body` on its own thread and fails the test if it does not finish, so a livelock fails fast
/// instead of hanging the suite.
fn within<R: Send + 'static>(seconds: u64, body: impl FnOnce() -> R + Send + 'static) -> R {
    let (sender, receiver) = mpsc::channel();
    thread::spawn(move || {
        let _ = sender.send(body());
    });
    receiver
        .recv_timeout(Duration::from_secs(seconds))
        .expect("the operation did not finish: it is livelocked")
}

fn deadline() -> Instant {
    Instant::now() + Duration::from_secs(20)
}

/// Encodes which producer sent an item and its position in that producer's sequence.
fn tag(producer: u64, sequence: u64) -> u64 {
    producer << 32 | sequence
}

/// Checks that every item from `producers` producers arrived exactly once and in per-producer order.
fn assert_exactly_once_and_ordered(received: &[u64], producers: u64, per_producer: u64) {
    assert_eq!(received.len() as u64, producers * per_producer);
    let mut next: HashMap<u64, u64> = HashMap::new();
    for item in received {
        let (producer, sequence) = (item >> 32, item & 0xFFFF_FFFF);
        let expected = next.entry(producer).or_insert(0);
        assert_eq!(
            sequence, *expected,
            "producer {producer} out of order or duplicated"
        );
        *expected += 1;
    }
    assert_eq!(next.len() as u64, producers);
    assert!(next.values().all(|count| *count == per_producer));
}

#[test]
fn a_single_thread_gets_fifo_order_across_chunk_rollovers() {
    let queue = MpscQueue::new(4);
    let (producer, mut consumer) = (queue.producer(), queue.consumer());

    for value in 0..100 {
        assert_eq!(producer.offer(value), Ok(()));
    }
    assert_eq!(queue.size_long(), 100);
    for value in 0..100 {
        assert_eq!(consumer.poll(), Some(value));
    }
    assert_eq!(consumer.poll(), None);
}

#[test]
fn a_bounded_queue_holds_exactly_its_capacity() {
    for capacity in [1, 2, 3, 4, 5, 8, 9] {
        let queue = MpscQueue::bounded(capacity);
        let producer = queue.producer();

        for value in 0..capacity {
            assert_eq!(producer.offer(value), Ok(()), "capacity {capacity}");
        }
        assert_eq!(
            producer.offer(usize::MAX),
            Err(usize::MAX),
            "capacity {capacity}"
        );
        assert_eq!(queue.size_long(), capacity as u64);
    }
}

#[test]
fn a_bounded_queue_wraps_its_ring_many_times() {
    let queue = MpscQueue::bounded(3);
    let (producer, mut consumer) = (queue.producer(), queue.consumer());
    let (mut next_in, mut next_out) = (0, 0);

    for _ in 0..300 {
        while producer.offer(next_in).is_ok() {
            next_in += 1;
        }
        assert_eq!(queue.size_long(), 3);
        for _ in 0..2 {
            assert_eq!(consumer.poll(), Some(next_out));
            next_out += 1;
        }
    }
}

#[test]
fn fill_moves_everything_into_an_unbounded_queue_in_order() {
    let queue = MpscQueue::new(4);
    let (producer, mut consumer) = (queue.producer(), queue.consumer());
    let mut items: Vec<u32> = (0..50).collect();

    assert_eq!(producer.fill(&mut items), 50);

    assert!(items.is_empty());
    let mut seen = Vec::new();
    consumer.drain_all(|v| seen.push(v));
    assert_eq!(seen, (0..50).collect::<Vec<_>>());
}

#[test]
fn fill_into_a_full_bounded_queue_returns_zero_instead_of_spinning() {
    // Upstream mpFill claimed zero slots, counted that as success, and looped forever.
    let queue = MpscQueue::bounded(3);
    let producer = queue.producer();
    for value in 0..3 {
        producer.offer(value).unwrap();
    }

    let inserted = within(5, move || {
        let mut items = vec![99];
        let inserted = producer.fill(&mut items);
        (inserted, items)
    });

    assert_eq!(inserted, (0, vec![99]));
}

#[test]
fn fill_into_an_unbounded_queue_that_has_just_run_out_of_ring_succeeds() {
    // Upstream hit the same zero-claim spin when three scalar offers had filled a chunk of four.
    let queue = MpscQueue::new(4);
    let producer = queue.producer();
    for value in 0..3 {
        producer.offer(value).unwrap();
    }

    let (inserted, rest) = within(5, move || {
        let mut items = vec![3];
        (producer.fill(&mut items), items)
    });

    assert_eq!((inserted, rest), (1, vec![]));
    let mut consumer = queue.consumer();
    let mut seen = Vec::new();
    consumer.drain_all(|v| seen.push(v));
    assert_eq!(seen, [0, 1, 2, 3]);
}

#[test]
fn an_oversized_first_fill_does_not_overwrite_its_own_items() {
    // Upstream claimed the whole input on a fresh queue, wrapping the ring over items it had just written.
    let queue = MpscQueue::new(4);
    let (producer, mut consumer) = (queue.producer(), queue.consumer());
    let mut items: Vec<u32> = (0..8).collect();

    let inserted = within(5, move || {
        let inserted = producer.fill(&mut items);
        (inserted, items)
    });

    assert_eq!(inserted, (8, vec![]));
    let mut seen = Vec::new();
    consumer.drain_all(|v| seen.push(v));
    assert_eq!(seen, (0..8).collect::<Vec<_>>());
}

#[test]
fn an_oversized_first_fill_into_a_bounded_queue_takes_only_what_fits() {
    let queue = MpscQueue::bounded(4);
    let (producer, mut consumer) = (queue.producer(), queue.consumer());
    let mut items: Vec<u32> = (0..8).collect();

    let inserted = within(5, move || {
        let inserted = producer.fill(&mut items);
        (inserted, items)
    });

    assert_eq!(inserted, (4, vec![4, 5, 6, 7]));
    let mut seen = Vec::new();
    consumer.drain_all(|v| seen.push(v));
    assert_eq!(seen, [0, 1, 2, 3]);
}

#[test]
fn fill_and_offer_interleave_in_order() {
    let queue = MpscQueue::new(2);
    let (producer, mut consumer) = (queue.producer(), queue.consumer());

    producer.offer(0).unwrap();
    producer.fill(&mut vec![1, 2, 3, 4, 5]);
    producer.offer(6).unwrap();
    producer.fill(&mut vec![7, 8]);

    let mut seen = Vec::new();
    consumer.drain_all(|v| seen.push(v));
    assert_eq!(seen, (0..9).collect::<Vec<_>>());
}

#[test]
fn producers_are_cloneable_and_many_can_exist() {
    fn assert_clone_send_sync<T: Clone + Send + Sync>() {}
    assert_clone_send_sync::<Producer<String, Multi>>();

    let queue = MpscQueue::new(4);
    let (first, second) = (queue.producer(), queue.producer());
    let third = first.clone();
    first.offer(1).unwrap();
    second.offer(2).unwrap();
    third.offer(3).unwrap();

    let mut consumer = queue.consumer();
    assert_eq!(
        (consumer.poll(), consumer.poll(), consumer.poll()),
        (Some(1), Some(2), Some(3))
    );
}

#[test]
#[should_panic(expected = "single consumer handle is already in use")]
fn only_one_consumer_may_exist() {
    let queue = MpscQueue::<i32>::new(4);
    let _first = queue.consumer();
    let _second = queue.consumer();
}

#[test]
fn the_consumer_handle_can_move_to_another_thread() {
    fn assert_send<T: Send>() {}
    assert_send::<Consumer<String, Single>>();
}

#[test]
fn many_producers_deliver_every_item_once_and_in_per_producer_order() {
    let (producers, per_producer) = (8_u64, 20_000_u64);
    let queue = MpscQueue::new(64);
    let mut consumer = queue.consumer();
    let start = Arc::new(Barrier::new(producers as usize + 1));

    let mut received = Vec::new();
    thread::scope(|scope| {
        for id in 0..producers {
            let (producer, start) = (queue.producer(), Arc::clone(&start));
            scope.spawn(move || {
                start.wait();
                for sequence in 0..per_producer {
                    producer.offer(tag(id, sequence)).unwrap();
                }
            });
        }
        start.wait();
        let limit = deadline();
        while (received.len() as u64) < producers * per_producer {
            if consumer.drain(|v| received.push(v), 256) == 0 {
                assert!(
                    Instant::now() < limit,
                    "consumer timed out after {} items",
                    received.len()
                );
                std::hint::spin_loop();
            }
        }
    });

    assert_exactly_once_and_ordered(&received, producers, per_producer);
    assert!(queue.is_empty());
}

#[test]
fn heavy_resizing_with_the_smallest_chunks_loses_nothing() {
    let (producers, per_producer) = (8_u64, 10_000_u64);
    let queue = MpscQueue::with_pool(2, 4);
    let mut consumer = queue.consumer();
    let mut received = Vec::new();

    thread::scope(|scope| {
        for id in 0..producers {
            let producer = queue.producer();
            scope.spawn(move || {
                for sequence in 0..per_producer {
                    producer.offer(tag(id, sequence)).unwrap();
                }
            });
        }
        let limit = deadline();
        while (received.len() as u64) < producers * per_producer {
            if consumer.poll().map(|v| received.push(v)).is_none() {
                assert!(
                    Instant::now() < limit,
                    "consumer timed out after {} items",
                    received.len()
                );
                std::hint::spin_loop();
            }
        }
    });

    assert_exactly_once_and_ordered(&received, producers, per_producer);
}

#[test]
fn concurrent_bulk_fills_keep_every_item_and_each_producers_order() {
    // Upstream wrote fill items through the *current* tail chunk, not the one it had reserved from, so a
    // concurrent resize could send them to the wrong chunk.
    let (producers, batches, batch_len) = (6_u64, 400_u64, 7_u64);
    let per_producer = batches * batch_len;
    let queue = MpscQueue::with_pool(4, 4);
    let mut consumer = queue.consumer();
    let mut received = Vec::new();

    thread::scope(|scope| {
        for id in 0..producers {
            let producer = queue.producer();
            scope.spawn(move || {
                for batch in 0..batches {
                    let mut items: Vec<u64> = (0..batch_len)
                        .map(|i| tag(id, batch * batch_len + i))
                        .collect();
                    while !items.is_empty() {
                        producer.fill(&mut items);
                    }
                }
            });
        }
        let limit = deadline();
        while (received.len() as u64) < producers * per_producer {
            if consumer.drain(|v| received.push(v), 64) == 0 {
                assert!(
                    Instant::now() < limit,
                    "consumer timed out after {} items",
                    received.len()
                );
                std::hint::spin_loop();
            }
        }
    });

    assert_exactly_once_and_ordered(&received, producers, per_producer);
}

#[test]
fn offers_and_fills_from_different_producers_mix_safely() {
    let (per_producer, queue) = (30_000_u64, MpscQueue::new(8));
    let mut consumer = queue.consumer();
    let mut received = Vec::new();

    thread::scope(|scope| {
        let offering = queue.producer();
        scope.spawn(move || {
            for sequence in 0..per_producer {
                offering.offer(tag(0, sequence)).unwrap();
            }
        });
        let filling = queue.producer();
        scope.spawn(move || {
            let mut sequence = 0;
            while sequence < per_producer {
                let len = 5.min(per_producer - sequence);
                let mut items: Vec<u64> = (sequence..sequence + len).map(|s| tag(1, s)).collect();
                sequence += len;
                while !items.is_empty() {
                    filling.fill(&mut items);
                }
            }
        });
        let limit = deadline();
        while (received.len() as u64) < 2 * per_producer {
            if consumer.drain(|v| received.push(v), 128) == 0 {
                assert!(Instant::now() < limit, "consumer timed out");
                std::hint::spin_loop();
            }
        }
    });

    assert_exactly_once_and_ordered(&received, 2, per_producer);
}

#[test]
fn a_bounded_queue_carries_every_item_from_many_producers() {
    let (producers, per_producer) = (4_u64, 15_000_u64);
    let queue = MpscQueue::bounded(8);
    let mut consumer = queue.consumer();
    let mut received = Vec::new();

    thread::scope(|scope| {
        for id in 0..producers {
            let producer = queue.producer();
            scope.spawn(move || {
                let limit = deadline();
                for sequence in 0..per_producer {
                    let mut pending = tag(id, sequence);
                    while let Err(rejected) = producer.offer(pending) {
                        assert!(
                            Instant::now() < limit,
                            "producer {id} timed out at {sequence}"
                        );
                        pending = rejected;
                        std::hint::spin_loop();
                    }
                }
            });
        }
        let limit = deadline();
        while (received.len() as u64) < producers * per_producer {
            if consumer.poll().map(|v| received.push(v)).is_none() {
                assert!(
                    Instant::now() < limit,
                    "consumer timed out after {} items",
                    received.len()
                );
                std::hint::spin_loop();
            }
        }
    });

    assert_exactly_once_and_ordered(&received, producers, per_producer);
}

#[test]
fn heap_data_written_before_offer_is_visible_to_drain_and_poll() {
    // Upstream's drain did not acquire the slot it read; here every consuming path does.
    struct Payload {
        id: u64,
        data: Vec<u64>,
        text: String,
    }

    let (producers, per_producer) = (4_u64, 10_000_u64);
    let queue = MpscQueue::new(8);
    let mut consumer = queue.consumer();
    let mut checked = 0_u64;

    thread::scope(|scope| {
        for id in 0..producers {
            let producer = queue.producer();
            scope.spawn(move || {
                for n in 0..per_producer {
                    let value = tag(id, n);
                    let payload = Payload {
                        id: value,
                        data: vec![value; 12],
                        text: value.to_string(),
                    };
                    assert!(producer.offer(payload).is_ok());
                }
            });
        }
        let limit = deadline();
        let verify = |payload: Payload| {
            assert!(
                payload.data.iter().all(|v| *v == payload.id),
                "torn vector for {}",
                payload.id
            );
            assert_eq!(payload.text, payload.id.to_string());
        };
        while checked < producers * per_producer {
            let before = checked;
            if checked.is_multiple_of(2) {
                checked += consumer.drain(verify, 32);
            } else if let Some(payload) = consumer.poll() {
                verify(payload);
                checked += 1;
            }
            if checked == before {
                assert!(
                    Instant::now() < limit,
                    "consumer timed out after {checked} items"
                );
                std::hint::spin_loop();
            }
        }
    });
}

#[test]
fn dropping_a_queue_with_in_flight_producers_drops_each_remaining_item_once() {
    struct Tracked(u64, Arc<Mutex<Vec<u64>>>);
    impl Drop for Tracked {
        fn drop(&mut self) {
            self.1.lock().unwrap().push(self.0);
        }
    }

    let dropped = Arc::new(Mutex::new(Vec::new()));
    {
        let queue = MpscQueue::with_pool(4, 2);
        let mut consumer = queue.consumer();
        thread::scope(|scope| {
            for id in 0..4_u64 {
                let (producer, dropped) = (queue.producer(), Arc::clone(&dropped));
                scope.spawn(move || {
                    for n in 0..250 {
                        assert!(
                            producer
                                .offer(Tracked(id * 1000 + n, Arc::clone(&dropped)))
                                .is_ok()
                        );
                    }
                });
            }
        });
        for _ in 0..300 {
            consumer.poll();
        }
    }

    let mut ids = dropped.lock().unwrap().clone();
    ids.sort_unstable();
    let mut expected: Vec<u64> = (0..4)
        .flat_map(|id| (0..250).map(move |n| id * 1000 + n))
        .collect();
    expected.sort_unstable();
    assert_eq!(ids, expected);
}
