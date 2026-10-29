#![allow(
    clippy::disallowed_methods,
    clippy::disallowed_types,
    reason = "test fixtures: the test thread runs peer threads and waits on them and locks its records"
)]
use euhedral_data_structures::queues::{Consumer, Producer, Single, SpscQueue};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

/// Records its id when dropped, so tests can prove every element is dropped exactly once.
struct Tracked {
    id: usize,
    dropped: Arc<Mutex<Vec<usize>>>,
}

impl Tracked {
    fn new(id: usize, dropped: &Arc<Mutex<Vec<usize>>>) -> Self {
        Self {
            id,
            dropped: Arc::clone(dropped),
        }
    }
}

impl Drop for Tracked {
    fn drop(&mut self) {
        self.dropped.lock().unwrap().push(self.id);
    }
}

fn deadline() -> Instant {
    Instant::now() + Duration::from_secs(10)
}

#[test]
fn a_new_queue_is_empty_unbounded_and_unpooled() {
    let queue = SpscQueue::<i32>::new(4);

    assert!(queue.is_empty());
    assert_eq!(queue.size_long(), 0);
    assert_eq!(queue.capacity(), u64::MAX);
    assert_eq!(queue.max_pooled_chunks(), 0);
}

#[test]
fn poll_and_peek_on_an_empty_queue_return_none() {
    let queue = SpscQueue::<i32>::new(4);
    let mut consumer = queue.consumer();

    assert_eq!(consumer.poll(), None);
    assert_eq!(consumer.peek(), None);
}

#[test]
fn elements_come_out_in_order_across_many_chunk_rollovers() {
    let queue = SpscQueue::new(4);
    let (mut producer, mut consumer) = (queue.producer(), queue.consumer());

    for value in 0..100 {
        assert_eq!(producer.offer(value), Ok(()));
    }
    assert_eq!(queue.size_long(), 100);
    for value in 0..100 {
        assert_eq!(consumer.poll(), Some(value));
    }
    assert_eq!(consumer.poll(), None);
    assert!(queue.is_empty());
}

#[test]
fn order_holds_when_offers_and_polls_interleave() {
    let queue = SpscQueue::new(2);
    let (mut producer, mut consumer) = (queue.producer(), queue.consumer());
    let mut next_out = 0;

    for value in 0..500 {
        producer.offer(value).unwrap();
        if value % 3 == 2 {
            assert_eq!(consumer.poll(), Some(next_out));
            next_out += 1;
        }
    }
    while let Some(value) = consumer.poll() {
        assert_eq!(value, next_out);
        next_out += 1;
    }
    assert_eq!(next_out, 500);
}

#[test]
fn peek_leaves_the_element_queued_even_across_a_chunk_boundary() {
    let queue = SpscQueue::new(2);
    let (mut producer, mut consumer) = (queue.producer(), queue.consumer());
    for value in 0..10 {
        producer.offer(value).unwrap();
    }

    for expected in 0..10 {
        assert_eq!(consumer.peek(), Some(&expected));
        assert_eq!(consumer.peek(), Some(&expected));
        assert_eq!(consumer.poll(), Some(expected));
    }
    assert_eq!(consumer.peek(), None);
}

#[test]
fn size_tracks_offers_and_polls() {
    let queue = SpscQueue::new(4);
    let (mut producer, mut consumer) = (queue.producer(), queue.consumer());

    for value in 0..7 {
        producer.offer(value).unwrap();
        assert_eq!(queue.size_long(), value as u64 + 1);
    }
    consumer.poll();
    consumer.poll();
    assert_eq!(queue.size_long(), 5);
    assert_eq!(producer.size_long(), 5);
    assert_eq!(consumer.size_long(), 5);
    assert!(!consumer.is_empty());
}

#[test]
fn a_bounded_queue_holds_exactly_its_capacity() {
    for capacity in [1, 2, 3, 4, 5, 7, 8, 9, 100] {
        let queue = SpscQueue::bounded(capacity);
        let mut producer = queue.producer();

        assert_eq!(queue.capacity(), capacity as u64);
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
fn a_bounded_queue_recovers_room_after_polls_and_wraps_its_ring() {
    let queue = SpscQueue::bounded(3);
    let (mut producer, mut consumer) = (queue.producer(), queue.consumer());
    let mut next_in = 0;
    let mut next_out = 0;

    for _ in 0..200 {
        while producer.offer(next_in).is_ok() {
            next_in += 1;
        }
        assert_eq!(queue.size_long(), 3);
        assert_eq!(consumer.poll(), Some(next_out));
        next_out += 1;
        assert_eq!(consumer.poll(), Some(next_out));
        next_out += 1;
    }
    while let Some(value) = consumer.poll() {
        assert_eq!(value, next_out);
        next_out += 1;
    }
    assert_eq!(next_out, next_in);
}

#[test]
fn a_rejected_offer_hands_the_element_back() {
    let queue = SpscQueue::bounded(1);
    let mut producer = queue.producer();
    producer.offer(String::from("kept")).unwrap();

    assert_eq!(
        producer.offer(String::from("rejected")),
        Err(String::from("rejected"))
    );
}

#[test]
#[should_panic(expected = "capacity must be positive")]
fn a_bounded_queue_needs_room_for_at_least_one_element() {
    SpscQueue::<i32>::bounded(0);
}

#[test]
fn drain_hands_over_up_to_the_limit_in_order() {
    let queue = SpscQueue::new(4);
    let (mut producer, mut consumer) = (queue.producer(), queue.consumer());
    for value in 0..10 {
        producer.offer(value).unwrap();
    }

    let mut seen = Vec::new();
    assert_eq!(consumer.drain(|v| seen.push(v), 4), 4);
    assert_eq!(seen, [0, 1, 2, 3]);
    assert_eq!(consumer.drain(|v| seen.push(v), 0), 0);
    assert_eq!(seen.len(), 4);
    assert_eq!(consumer.drain_all(|v| seen.push(v)), 6);
    assert_eq!(seen, (0..10).collect::<Vec<_>>());
    assert_eq!(consumer.drain_all(|v| seen.push(v)), 0);
}

#[test]
fn drain_while_stops_before_the_first_match_and_leaves_it_queued() {
    // Port of the upstream SingleConsumerDrainTest: stop predicates across chunk rollover and reuse.
    let queue = SpscQueue::new(4);
    let (mut producer, mut consumer) = (queue.producer(), queue.consumer());

    for _generation in 0..3 {
        for value in 0..100 {
            producer.offer(value).unwrap();
        }
        let mut drained = Vec::new();
        assert_eq!(
            consumer.drain_while(|v| drained.push(v), |v| *v == 25, 100),
            25
        );
        assert_eq!(consumer.peek(), Some(&25));
        assert_eq!(
            consumer.drain_while(|v| drained.push(v), |v| *v < 0, 10),
            10
        );
        assert_eq!(
            consumer.drain_while(|v| drained.push(v), |v| *v < 0, 100),
            65
        );
        assert_eq!(drained, (0..100).collect::<Vec<_>>());
        assert_eq!(consumer.poll(), None);
    }
}

#[test]
fn the_stop_predicate_only_ever_sees_queued_elements() {
    let queue = SpscQueue::new(2);
    let (mut producer, mut consumer) = (queue.producer(), queue.consumer());
    for value in 0..60 {
        producer.offer(value).unwrap();
    }

    let mut inspected = Vec::new();
    let drained = consumer.drain_while(
        |_| {},
        |v| {
            inspected.push(*v);
            *v == 25
        },
        u64::MAX,
    );

    assert_eq!(drained, 25);
    assert_eq!(inspected, (0..=25).collect::<Vec<_>>());
}

#[test]
fn a_panicking_consumer_still_counts_the_element_it_took() {
    let queue = SpscQueue::new(4);
    let (mut producer, mut consumer) = (queue.producer(), queue.consumer());
    for value in 0..10 {
        producer.offer(value).unwrap();
    }

    let mut seen = Vec::new();
    let result = catch_unwind(AssertUnwindSafe(|| {
        consumer.drain_all(|v| {
            if v == 3 {
                panic!("consumer failed");
            }
            seen.push(v);
        })
    }));

    assert!(result.is_err());
    assert_eq!(seen, [0, 1, 2]);
    // Upstream cleared the slot but never advanced the head, leaving a hole at element 3.
    assert_eq!(consumer.poll(), Some(4));
    assert_eq!(queue.size_long(), 5);
}

#[test]
fn a_panicking_stop_predicate_leaves_the_element_queued() {
    let queue = SpscQueue::new(4);
    let (mut producer, mut consumer) = (queue.producer(), queue.consumer());
    for value in 0..5 {
        producer.offer(value).unwrap();
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        consumer.drain_while(
            |_| {},
            |v| {
                if *v == 2 {
                    panic!("predicate failed");
                }
                false
            },
            u64::MAX,
        )
    }));

    assert!(result.is_err());
    assert_eq!(consumer.poll(), Some(2));
}

#[test]
fn clear_removes_everything_and_the_queue_stays_usable() {
    let queue = SpscQueue::new(4);
    let (mut producer, mut consumer) = (queue.producer(), queue.consumer());
    for value in 0..50 {
        producer.offer(value).unwrap();
    }

    consumer.clear();

    assert!(queue.is_empty());
    assert_eq!(consumer.poll(), None);
    producer.offer(7).unwrap();
    assert_eq!(consumer.poll(), Some(7));
}

#[test]
fn fill_moves_every_element_of_an_unbounded_queue_in_order() {
    let queue = SpscQueue::new(4);
    let (mut producer, mut consumer) = (queue.producer(), queue.consumer());
    let mut items: Vec<i32> = (0..50).collect();

    assert_eq!(producer.fill(&mut items), 50);

    assert!(items.is_empty());
    let mut seen = Vec::new();
    consumer.drain_all(|v| seen.push(v));
    assert_eq!(seen, (0..50).collect::<Vec<_>>());
}

#[test]
fn fill_into_a_bounded_queue_leaves_the_rest_with_the_caller() {
    let queue = SpscQueue::bounded(3);
    let (mut producer, mut consumer) = (queue.producer(), queue.consumer());
    let mut items = vec![1, 2, 3, 4, 5];

    assert_eq!(producer.fill(&mut items), 3);
    assert_eq!(items, [4, 5]);
    assert_eq!(producer.fill(&mut items), 0, "the queue is full");
    assert_eq!(items, [4, 5]);

    assert_eq!(consumer.poll(), Some(1));
    assert_eq!(producer.fill(&mut items), 1);
    assert_eq!(items, [5]);
    let mut seen = Vec::new();
    consumer.drain_all(|v| seen.push(v));
    assert_eq!(seen, [2, 3, 4]);
}

#[test]
fn fill_of_nothing_inserts_nothing() {
    let queue = SpscQueue::<i32>::new(4);
    let mut producer = queue.producer();

    assert_eq!(producer.fill(&mut Vec::new()), 0);
    assert!(queue.is_empty());
}

#[test]
fn fill_and_offer_interleave_in_order() {
    let queue = SpscQueue::new(2);
    let (mut producer, mut consumer) = (queue.producer(), queue.consumer());

    producer.offer(0).unwrap();
    producer.fill(&mut vec![1, 2, 3, 4, 5]);
    producer.offer(6).unwrap();
    producer.fill(&mut vec![7, 8]);

    let mut seen = Vec::new();
    consumer.drain_all(|v| seen.push(v));
    assert_eq!(seen, (0..9).collect::<Vec<_>>());
}

#[test]
fn retired_chunks_are_reused_instead_of_reallocated() {
    let queue = SpscQueue::with_pool(4, 8);
    let (mut producer, mut consumer) = (queue.producer(), queue.consumer());

    for value in 0..24 {
        producer.offer(value).unwrap();
    }
    let allocated_in_the_first_round = queue.chunks_allocated();
    assert!(allocated_in_the_first_round > 0);
    consumer.clear();
    assert!(
        queue.pooled_chunks() > 0,
        "drained chunks go back to the pool"
    );

    for value in 0..24 {
        producer.offer(value).unwrap();
    }

    assert_eq!(queue.chunks_allocated(), allocated_in_the_first_round);
    let mut seen = Vec::new();
    consumer.drain_all(|v| seen.push(v));
    assert_eq!(seen, (0..24).collect::<Vec<_>>());
}

#[test]
fn without_a_pool_every_growth_allocates() {
    let queue = SpscQueue::new(4);
    let (mut producer, mut consumer) = (queue.producer(), queue.consumer());

    for round in 1..=3 {
        for value in 0..24 {
            producer.offer(value).unwrap();
        }
        consumer.clear();
        assert_eq!(queue.pooled_chunks(), 0);
        assert!(queue.chunks_allocated() >= round * 7, "round {round}");
    }
}

#[test]
fn the_pool_never_holds_more_than_its_maximum() {
    let queue = SpscQueue::with_pool(4, 2);
    let (mut producer, mut consumer) = (queue.producer(), queue.consumer());
    assert_eq!(queue.max_pooled_chunks(), 2);

    for value in 0..100 {
        producer.offer(value).unwrap();
    }
    consumer.clear();

    assert_eq!(queue.pooled_chunks(), 2);
}

#[test]
fn reused_chunks_do_not_replay_old_payloads() {
    // Port of the upstream retiredChunkIsActuallyReused test: after reuse the new payloads, and only
    // those, come out in order.
    let queue = SpscQueue::with_pool(4, 8);
    let (mut producer, mut consumer) = (queue.producer(), queue.consumer());

    for generation in 0..5 {
        let payloads: Vec<String> = (0..20).map(|i| format!("g{generation}-{i}")).collect();
        for payload in &payloads {
            producer.offer(payload.clone()).unwrap();
        }
        let mut received = Vec::new();
        assert_eq!(consumer.drain_all(|p| received.push(p)), 20);
        assert_eq!(received, payloads);
        assert_eq!(consumer.poll(), None);
        assert!(queue.is_empty());
    }
}

#[test]
fn dropping_the_queue_drops_each_remaining_element_exactly_once() {
    let dropped = Arc::new(Mutex::new(Vec::new()));
    {
        let queue = SpscQueue::with_pool(4, 3);
        let (mut producer, mut consumer) = (queue.producer(), queue.consumer());
        for id in 0..30 {
            producer.offer(Tracked::new(id, &dropped)).ok().unwrap();
        }
        for _ in 0..12 {
            consumer.poll();
        }
        assert_eq!(dropped.lock().unwrap().len(), 12);
    }

    let mut ids = dropped.lock().unwrap().clone();
    ids.sort_unstable();
    assert_eq!(ids, (0..30).collect::<Vec<_>>());
}

#[test]
fn elements_outlive_the_queue_value_while_a_handle_remains() {
    let dropped = Arc::new(Mutex::new(Vec::new()));
    let queue = SpscQueue::new(4);
    let mut producer = queue.producer();
    producer.offer(Tracked::new(1, &dropped)).ok().unwrap();

    drop(queue);
    assert!(dropped.lock().unwrap().is_empty());
    drop(producer);

    assert_eq!(*dropped.lock().unwrap(), [1]);
}

#[test]
fn a_rejected_element_is_dropped_by_the_caller_not_the_queue() {
    let dropped = Arc::new(Mutex::new(Vec::new()));
    let queue = SpscQueue::bounded(1);
    let mut producer = queue.producer();
    producer.offer(Tracked::new(0, &dropped)).ok().unwrap();

    let rejected = producer.offer(Tracked::new(1, &dropped)).err().unwrap();
    assert!(dropped.lock().unwrap().is_empty());
    assert_eq!(rejected.id, 1);
    drop(rejected);

    assert_eq!(*dropped.lock().unwrap(), [1]);
}

#[test]
#[should_panic(expected = "single producer handle is already in use")]
fn a_second_producer_handle_is_refused_while_the_first_lives() {
    let queue = SpscQueue::<i32>::new(4);
    let _first = queue.producer();
    let _second = queue.producer();
}

#[test]
#[should_panic(expected = "single consumer handle is already in use")]
fn a_second_consumer_handle_is_refused_while_the_first_lives() {
    let queue = SpscQueue::<i32>::new(4);
    let _first = queue.consumer();
    let _second = queue.consumer();
}

#[test]
fn a_role_can_be_taken_again_after_its_handle_is_dropped() {
    let queue = SpscQueue::new(4);

    let mut producer = queue.producer();
    producer.offer(1).unwrap();
    drop(producer);
    let mut producer = queue.producer();
    producer.offer(2).unwrap();

    let mut consumer = queue.consumer();
    assert_eq!(consumer.poll(), Some(1));
    drop(consumer);
    assert_eq!(queue.consumer().poll(), Some(2));
}

#[test]
fn the_producer_and_consumer_roles_are_independent() {
    let queue = SpscQueue::<i32>::new(4);

    let _producer = queue.producer();
    let _consumer = queue.consumer();
}

#[test]
fn handles_can_be_sent_to_other_threads() {
    fn assert_send<T: Send>() {}
    fn assert_send_sync<T: Send + Sync>() {}

    assert_send::<Producer<String, Single>>();
    assert_send::<Consumer<String, Single>>();
    assert_send_sync::<SpscQueue<String>>();
}

#[test]
fn a_producer_thread_and_a_consumer_thread_keep_order() {
    let count = 200_000_u64;
    let queue = SpscQueue::new(64);
    let (mut producer, mut consumer) = (queue.producer(), queue.consumer());

    thread::scope(|scope| {
        scope.spawn(move || {
            for value in 0..count {
                producer.offer(value).unwrap();
            }
        });
        scope.spawn(move || {
            let limit = deadline();
            let mut expected = 0;
            while expected < count {
                match consumer.poll() {
                    Some(value) => {
                        assert_eq!(value, expected);
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

    assert!(queue.is_empty());
}

#[test]
fn a_pooled_queue_keeps_order_under_concurrent_reuse() {
    let count = 200_000_u64;
    let queue = SpscQueue::with_pool(4, 4);
    let (mut producer, mut consumer) = (queue.producer(), queue.consumer());

    thread::scope(|scope| {
        scope.spawn(move || {
            for value in 0..count {
                producer.offer(value).unwrap();
            }
        });
        scope.spawn(move || {
            let limit = deadline();
            let mut expected = 0;
            while expected < count {
                let drained = consumer.drain(
                    |value| {
                        assert_eq!(value, expected);
                        expected += 1;
                    },
                    16,
                );
                if drained == 0 {
                    assert!(Instant::now() < limit, "consumer timed out at {expected}");
                    std::hint::spin_loop();
                }
            }
        });
    });
}

#[test]
fn heap_data_written_before_offer_is_visible_after_poll() {
    struct Payload {
        id: u64,
        data: Vec<u64>,
        boxed: Box<u64>,
    }

    let count = 50_000_u64;
    let queue = SpscQueue::new(8);
    let (mut producer, mut consumer) = (queue.producer(), queue.consumer());
    let torn = AtomicUsize::new(0);

    thread::scope(|scope| {
        scope.spawn(move || {
            for id in 0..count {
                let payload = Payload {
                    id,
                    data: vec![id; 16],
                    boxed: Box::new(id),
                };
                assert!(producer.offer(payload).is_ok());
            }
        });
        scope.spawn(|| {
            let limit = deadline();
            let mut seen = 0;
            while seen < count {
                match consumer.poll() {
                    Some(payload) => {
                        if payload.data.iter().any(|v| *v != payload.id)
                            || *payload.boxed != payload.id
                        {
                            torn.fetch_add(1, Ordering::Relaxed);
                        }
                        seen += 1;
                    }
                    None => {
                        assert!(Instant::now() < limit, "consumer timed out");
                        std::hint::spin_loop();
                    }
                }
            }
        });
    });

    assert_eq!(torn.load(Ordering::Relaxed), 0);
}

#[test]
fn a_bounded_queue_carries_every_element_between_threads() {
    let count = 100_000_u64;
    let queue = SpscQueue::bounded(8);
    let (mut producer, mut consumer) = (queue.producer(), queue.consumer());

    thread::scope(|scope| {
        scope.spawn(move || {
            let limit = deadline();
            for value in 0..count {
                let mut pending = value;
                while let Err(rejected) = producer.offer(pending) {
                    assert!(Instant::now() < limit, "producer timed out at {value}");
                    pending = rejected;
                    std::hint::spin_loop();
                }
            }
        });
        scope.spawn(move || {
            let limit = deadline();
            let mut expected = 0;
            while expected < count {
                match consumer.poll() {
                    Some(value) => {
                        assert_eq!(value, expected);
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
