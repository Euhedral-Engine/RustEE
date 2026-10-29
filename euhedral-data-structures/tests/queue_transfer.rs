//! Moving elements from one queue straight into another (`drain(receiver, ...)` upstream).
#![allow(
    clippy::disallowed_methods,
    clippy::disallowed_types,
    reason = "test fixtures: the test thread runs peer threads and waits on them and locks its records"
)]

use euhedral_data_structures::queues::{MpmcQueue, MpscQueue, Sink, SpscQueue};
use std::collections::VecDeque;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, Mutex};

#[test]
fn a_transfer_moves_elements_in_order_up_to_the_limit() {
    let source = SpscQueue::new(4);
    let (mut producer, mut consumer) = (source.producer(), source.consumer());
    for value in 0..10 {
        producer.offer(value).unwrap();
    }
    let mut sink = Vec::new();

    assert_eq!(consumer.drain_to(&mut sink, 4), 4);

    assert_eq!(sink, [0, 1, 2, 3]);
    assert_eq!(source.size_long(), 6);
    assert_eq!(consumer.poll(), Some(4));
}

#[test]
fn transferred_elements_leave_the_source() {
    // Upstream's transfer into a multi-producer receiver advanced a local head only, so the source kept
    // returning the same element and a second transfer duplicated it.
    let source = MpscQueue::new(4);
    let (producer, mut consumer) = (source.producer(), source.consumer());
    let target = MpscQueue::new(4);
    let (target_producer, mut target_consumer) = (target.producer(), target.consumer());
    producer.offer("x").unwrap();
    let mut sink = target_producer.clone();

    assert_eq!(consumer.drain_to(&mut sink, 1), 1);
    assert_eq!(consumer.drain_to(&mut sink, 1), 0);

    assert_eq!(consumer.poll(), None);
    assert!(source.is_empty());
    assert_eq!(target_consumer.poll(), Some("x"));
    assert_eq!(target_consumer.poll(), None);
}

#[test]
fn an_unlimited_transfer_moves_everything() {
    // Upstream shifted the limit left, so Long.MAX_VALUE overflowed and nothing moved.
    let source = SpscQueue::new(4);
    let (mut producer, mut consumer) = (source.producer(), source.consumer());
    producer.offer(1).unwrap();
    let mut sink = Vec::new();

    assert_eq!(consumer.drain_to(&mut sink, u64::MAX), 1);

    assert_eq!(sink, [1]);
}

#[test]
fn a_zero_limit_moves_nothing() {
    let source = SpscQueue::new(4);
    let (mut producer, mut consumer) = (source.producer(), source.consumer());
    producer.offer(1).unwrap();
    let mut sink = Vec::new();

    assert_eq!(consumer.drain_to(&mut sink, 0), 0);

    assert!(sink.is_empty());
    assert_eq!(source.size_long(), 1);
}

#[test]
fn every_receiver_kind_accepts_a_transfer() {
    let source = SpscQueue::new(2);
    let (mut producer, mut consumer) = (source.producer(), source.consumer());
    for value in 0..9 {
        producer.offer(value).unwrap();
    }

    let single = SpscQueue::new(2);
    let mut single_producer = single.producer();
    assert_eq!(consumer.drain_to(&mut single_producer, 3), 3);

    let multi = MpmcQueue::new(2);
    let mut multi_producer = multi.producer();
    assert_eq!(consumer.drain_to(&mut multi_producer, 3), 3);

    let mut deque = VecDeque::new();
    assert_eq!(consumer.drain_to(&mut deque, 3), 3);

    let mut first = Vec::new();
    single.consumer().drain_all(|v| first.push(v));
    let mut second = Vec::new();
    multi.consumer().drain_all(|v| second.push(v));
    assert_eq!(
        (first, second, deque),
        (vec![0, 1, 2], vec![3, 4, 5], VecDeque::from([6, 7, 8]))
    );
}

#[test]
fn a_full_receiver_stops_the_transfer_and_the_rejected_element_stays_first_in_line() {
    let source = SpscQueue::new(4);
    let (mut producer, mut consumer) = (source.producer(), source.consumer());
    for value in 0..10 {
        producer.offer(value).unwrap();
    }
    let target = SpscQueue::bounded(3);
    let mut target_producer = target.producer();
    let mut target_consumer = target.consumer();

    assert_eq!(consumer.drain_to(&mut target_producer, u64::MAX), 3);
    assert_eq!(source.size_long(), 7);
    assert_eq!(
        consumer.peek(),
        Some(&3),
        "the rejected element is still at the head"
    );
    assert_eq!(consumer.drain_to(&mut target_producer, u64::MAX), 0);

    assert_eq!(target_consumer.poll(), Some(0));
    assert_eq!(consumer.drain_to(&mut target_producer, u64::MAX), 1);
    let mut seen = Vec::new();
    target_consumer.drain_all(|v| seen.push(v));
    assert_eq!(seen, [1, 2, 3]);
    let mut rest = Vec::new();
    consumer.drain_all(|v| rest.push(v));
    assert_eq!(rest, (4..10).collect::<Vec<_>>());
}

#[test]
fn repeated_transfers_across_chunk_rollover_and_reuse_never_replay_old_elements() {
    // Upstream left the first payload of a successor chunk in its slot after a transfer.
    let source = SpscQueue::with_pool(2, 4);
    let (mut producer, mut consumer) = (source.producer(), source.consumer());
    let mut next = 0;
    let mut received = Vec::new();

    for round in 0..40 {
        for _ in 0..(round % 7 + 1) {
            producer.offer(next).unwrap();
            next += 1;
        }
        consumer.drain_to(&mut received, 3);
    }
    consumer.drain_to(&mut received, u64::MAX);

    assert_eq!(received, (0..next).collect::<Vec<_>>());
    assert_eq!(consumer.poll(), None);
}

#[test]
fn the_inspector_sees_each_element_before_it_moves() {
    let source = SpscQueue::new(4);
    let (mut producer, mut consumer) = (source.producer(), source.consumer());
    for value in 0..5 {
        producer.offer(value).unwrap();
    }
    let mut inspected = Vec::new();
    let mut sink = Vec::new();

    let moved = consumer.drain_to_inspect(&mut sink, |v| inspected.push(*v), 3);

    assert_eq!(moved, 3);
    assert_eq!(inspected, [0, 1, 2]);
    assert_eq!(sink, [0, 1, 2]);
}

#[test]
fn a_sink_that_panics_leaves_the_source_consistent() {
    struct Exploding(u64);
    impl Sink<u64> for Exploding {
        fn push(&mut self, item: u64) -> Result<(), u64> {
            if item == self.0 {
                panic!("sink failed");
            }
            Ok(())
        }
    }

    let source = SpscQueue::new(4);
    let (mut producer, mut consumer) = (source.producer(), source.consumer());
    for value in 0..6 {
        producer.offer(value).unwrap();
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        consumer.drain_to(&mut Exploding(2), u64::MAX)
    }));

    assert!(result.is_err());
    // Elements 0 and 1 were delivered and element 2 was handed to the failing sink; 3.. are still queued.
    assert_eq!(consumer.poll(), Some(3));
    assert_eq!(source.size_long(), 2);
}

#[test]
fn rejected_elements_are_not_dropped_twice() {
    struct Tracked(u64, Arc<Mutex<Vec<u64>>>);
    impl Drop for Tracked {
        fn drop(&mut self) {
            self.1.lock().unwrap().push(self.0);
        }
    }

    let dropped = Arc::new(Mutex::new(Vec::new()));
    {
        let source = SpscQueue::new(4);
        let (mut producer, mut consumer) = (source.producer(), source.consumer());
        for id in 0..6 {
            producer
                .offer(Tracked(id, Arc::clone(&dropped)))
                .ok()
                .unwrap();
        }
        let target = SpscQueue::bounded(2);
        let mut target_producer = target.producer();

        assert_eq!(consumer.drain_to(&mut target_producer, u64::MAX), 2);
        assert!(
            dropped.lock().unwrap().is_empty(),
            "nothing is dropped while it is still queued"
        );
    }

    let mut ids = dropped.lock().unwrap().clone();
    ids.sort_unstable();
    assert_eq!(ids, (0..6).collect::<Vec<_>>());
}

#[test]
fn a_relay_between_two_queues_loses_and_duplicates_nothing_under_load() {
    use std::collections::HashSet;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::{Duration, Instant};

    let (producers, per_producer) = (3_u64, 20_000_u64);
    let total = (producers * per_producer) as usize;
    let upstream = MpscQueue::with_pool(8, 2);
    // Small and bounded, so the relay is rejected constantly and keeps putting the element back.
    let downstream = MpmcQueue::bounded(16);
    let seen = Mutex::new(HashSet::new());
    let consumed = AtomicUsize::new(0);
    let deadline = Instant::now() + Duration::from_secs(30);

    std::thread::scope(|scope| {
        for id in 0..producers {
            let producer = upstream.producer();
            scope.spawn(move || {
                for n in 0..per_producer {
                    producer.offer(id << 32 | n).unwrap();
                }
            });
        }
        let mut relay_in = upstream.consumer();
        let mut relay_out = downstream.producer();
        let consumed_ref = &consumed;
        scope.spawn(move || {
            while consumed_ref.load(Ordering::Relaxed) < total {
                if relay_in.drain_to(&mut relay_out, 32) == 0 {
                    assert!(Instant::now() < deadline, "relay timed out");
                    std::hint::spin_loop();
                }
            }
        });
        for _ in 0..2 {
            let consumer = downstream.consumer();
            let (seen, consumed) = (&seen, &consumed);
            scope.spawn(move || {
                while consumed.load(Ordering::Relaxed) < total {
                    match consumer.poll() {
                        Some(value) => {
                            assert!(seen.lock().unwrap().insert(value), "duplicate {value:#x}");
                            consumed.fetch_add(1, Ordering::Relaxed);
                        }
                        None => {
                            assert!(Instant::now() < deadline, "consumer timed out");
                            std::hint::spin_loop();
                        }
                    }
                }
            });
        }
    });

    assert_eq!(seen.lock().unwrap().len(), total);
}
