#![allow(
    clippy::disallowed_types,
    reason = "test fixtures: the test thread locks its records"
)]
use euhedral_data_structures::queues::{PlainQueue, Sink};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, Mutex};

#[test]
fn a_new_queue_is_empty_and_unbounded() {
    let queue = PlainQueue::<i32>::new(4);

    assert!(queue.is_empty());
    assert_eq!(queue.size_long(), 0);
    assert_eq!(queue.capacity(), u64::MAX);
    assert_eq!(queue.peek(), None);
}

#[test]
fn elements_come_out_in_order_however_many_go_in() {
    // Upstream's ring never grew, so the 5th offer into a chunk of 4 overwrote the 1st.
    let mut queue = PlainQueue::new(4);

    for value in 0..1000 {
        assert_eq!(queue.offer(value), Ok(()));
    }
    assert_eq!(queue.size_long(), 1000);
    for value in 0..1000 {
        assert_eq!(queue.poll(), Some(value));
    }
    assert_eq!(queue.poll(), None);
}

#[test]
fn offers_and_polls_interleave_in_order() {
    let mut queue = PlainQueue::new(2);
    let mut next_out = 0;

    for value in 0..500 {
        queue.offer(value).unwrap();
        if value % 3 == 2 {
            assert_eq!(queue.poll(), Some(next_out));
            next_out += 1;
        }
    }
    while let Some(value) = queue.poll() {
        assert_eq!(value, next_out);
        next_out += 1;
    }
    assert_eq!(next_out, 500);
}

#[test]
fn peek_leaves_the_element_queued() {
    let mut queue = PlainQueue::new(4);
    queue.offer("a").unwrap();
    queue.offer("b").unwrap();

    assert_eq!(queue.peek(), Some(&"a"));
    assert_eq!(queue.peek(), Some(&"a"));
    assert_eq!(queue.poll(), Some("a"));
    assert_eq!(queue.peek(), Some(&"b"));
}

#[test]
fn a_bounded_queue_holds_exactly_its_capacity_and_wraps_many_times() {
    for capacity in [1, 2, 3, 4, 5, 8, 9] {
        let mut queue = PlainQueue::bounded(capacity);
        assert_eq!(queue.capacity(), capacity as u64);

        for value in 0..capacity {
            assert_eq!(queue.offer(value), Ok(()), "capacity {capacity}");
        }
        assert_eq!(
            queue.offer(usize::MAX),
            Err(usize::MAX),
            "capacity {capacity}"
        );

        for round in 0..50 {
            assert_eq!(queue.poll(), Some(round));
            assert_eq!(queue.offer(capacity + round), Ok(()));
            assert_eq!(queue.size_long(), capacity as u64);
        }
    }
}

#[test]
#[should_panic(expected = "capacity must be positive")]
fn a_bounded_queue_needs_room_for_at_least_one_element() {
    PlainQueue::<i32>::bounded(0);
}

#[test]
fn fill_takes_what_fits_and_leaves_the_rest() {
    let mut queue = PlainQueue::bounded(3);
    let mut items = vec![1, 2, 3, 4, 5];

    assert_eq!(queue.fill(&mut items), 3);
    assert_eq!(items, [4, 5]);
    assert_eq!(queue.fill(&mut items), 0);

    let mut unbounded = PlainQueue::new(2);
    let mut all: Vec<i32> = (0..40).collect();
    assert_eq!(unbounded.fill(&mut all), 40);
    assert!(all.is_empty());
    assert_eq!(unbounded.fill(&mut Vec::new()), 0);
}

#[test]
fn drain_hands_over_up_to_the_limit_in_order() {
    let mut queue = PlainQueue::new(4);
    queue.fill(&mut (0..10).collect());
    let mut seen = Vec::new();

    assert_eq!(queue.drain(|v| seen.push(v), 4), 4);
    assert_eq!(queue.drain(|v| seen.push(v), 0), 0);
    assert_eq!(queue.drain_all(|v| seen.push(v)), 6);

    assert_eq!(seen, (0..10).collect::<Vec<_>>());
}

#[test]
fn drain_while_honours_its_stop_condition() {
    // Upstream's PlainQueue.drain accepted a stop condition and ignored it.
    let mut queue = PlainQueue::new(4);
    queue.fill(&mut (0..10).collect());
    let mut seen = Vec::new();

    let count = queue.drain_while(|v| seen.push(v), |v| *v == 6, u64::MAX);

    assert_eq!(count, 6);
    assert_eq!(seen, [0, 1, 2, 3, 4, 5]);
    assert_eq!(queue.peek(), Some(&6));
    assert_eq!(queue.size_long(), 4);
}

#[test]
fn a_panicking_consumer_still_counts_the_element_it_took() {
    let mut queue = PlainQueue::new(4);
    queue.fill(&mut (0..6).collect());

    let result = catch_unwind(AssertUnwindSafe(|| {
        queue.drain_all(|v| {
            if v == 2 {
                panic!("consumer failed");
            }
        })
    }));

    assert!(result.is_err());
    assert_eq!(queue.poll(), Some(3));
}

#[test]
fn clear_empties_the_queue_and_it_stays_usable() {
    let mut queue = PlainQueue::new(4);
    queue.fill(&mut (0..30).collect());

    queue.clear();

    assert!(queue.is_empty());
    queue.offer(5).unwrap();
    assert_eq!(queue.poll(), Some(5));
}

#[test]
fn it_is_a_sink_so_other_queues_can_drain_into_it() {
    let mut queue = PlainQueue::bounded(2);

    assert_eq!(Sink::push(&mut queue, 1), Ok(()));
    assert_eq!(Sink::push(&mut queue, 2), Ok(()));
    assert_eq!(Sink::push(&mut queue, 3), Err(3));
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
        let mut queue = PlainQueue::new(4);
        for id in 0..20 {
            assert!(queue.offer(Tracked(id, Arc::clone(&dropped))).is_ok());
        }
        for _ in 0..7 {
            queue.poll();
        }
    }

    let mut ids = dropped.lock().unwrap().clone();
    ids.sort_unstable();
    assert_eq!(ids, (0..20).collect::<Vec<_>>());
}
