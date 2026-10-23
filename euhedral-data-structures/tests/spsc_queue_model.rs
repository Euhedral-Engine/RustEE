//! Drives an SPSC queue with random operation sequences and checks every result against a `VecDeque`.
//! Small chunks and pools force constant chunk rollover, reuse and ring wrap-around.

use euhedral_data_structures::queues::SpscQueue;
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

fn run(queue: SpscQueue<u64>, capacity: Option<usize>, seed: u64, steps: usize) {
    let mut rng = Rng(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1);
    let (mut producer, mut consumer) = (queue.producer(), queue.consumer());
    let mut model: VecDeque<u64> = VecDeque::new();
    let mut next_value = 0_u64;
    let room = |len: usize| capacity.map_or(usize::MAX, |c| c - len);
    let context = |step: usize, op: &str| format!("seed {seed} step {step} op {op}");

    for step in 0..steps {
        match rng.below(8) {
            0 | 1 => {
                let value = next_value;
                next_value += 1;
                let accepted = room(model.len()) > 0;
                assert_eq!(
                    producer.offer(value).is_ok(),
                    accepted,
                    "{}",
                    context(step, "offer")
                );
                if accepted {
                    model.push_back(value);
                }
            }
            2 => {
                let count = 1 + rng.below(9) as usize;
                let mut items: Vec<u64> = (0..count as u64).map(|i| next_value + i).collect();
                next_value += count as u64;
                let expected = count.min(room(model.len()));
                assert_eq!(
                    producer.fill(&mut items),
                    expected,
                    "{}",
                    context(step, "fill")
                );
                assert_eq!(
                    items.len(),
                    count - expected,
                    "{}",
                    context(step, "fill leftover")
                );
                model.extend(
                    (next_value - count as u64)..(next_value - count as u64 + expected as u64),
                );
            }
            3 | 4 => {
                assert_eq!(
                    consumer.poll(),
                    model.pop_front(),
                    "{}",
                    context(step, "poll")
                );
            }
            5 => {
                assert_eq!(consumer.peek(), model.front(), "{}", context(step, "peek"));
            }
            6 => {
                let limit = rng.below(6);
                let mut drained = Vec::new();
                let count = consumer.drain(|v| drained.push(v), limit);
                let expected: Vec<u64> = (0..limit).map_while(|_| model.pop_front()).collect();
                assert_eq!(
                    count as usize,
                    expected.len(),
                    "{}",
                    context(step, "drain count")
                );
                assert_eq!(drained, expected, "{}", context(step, "drain"));
            }
            _ => {
                if rng.below(4) == 0 {
                    consumer.clear();
                    model.clear();
                } else {
                    let divisor = 2 + rng.below(5);
                    let limit = 1 + rng.below(8);
                    let mut drained = Vec::new();
                    consumer.drain_while(|v| drained.push(v), |v| v % divisor == 0, limit);
                    let mut expected = Vec::new();
                    while (expected.len() as u64) < limit
                        && model.front().is_some_and(|v| v % divisor != 0)
                    {
                        expected.push(model.pop_front().unwrap());
                    }
                    assert_eq!(drained, expected, "{}", context(step, "drain_while"));
                }
            }
        }
        assert_eq!(
            queue.size_long(),
            model.len() as u64,
            "{}",
            context(step, "size")
        );
    }

    let mut rest = Vec::new();
    consumer.drain_all(|v| rest.push(v));
    assert_eq!(
        rest,
        model.into_iter().collect::<Vec<_>>(),
        "seed {seed} final drain"
    );
    assert!(queue.is_empty());
}

#[test]
fn unbounded_queues_match_a_vecdeque_over_random_operations() {
    for chunk_size in [2, 3, 4, 8, 16] {
        for max_pooled in [0, 1, 3] {
            for seed in 1..=60 {
                run(
                    SpscQueue::with_pool(chunk_size, max_pooled),
                    None,
                    seed,
                    600,
                );
            }
        }
    }
}

#[test]
fn bounded_queues_match_a_vecdeque_over_random_operations() {
    for capacity in [1, 2, 3, 4, 5, 7, 8, 9, 15] {
        for seed in 1..=60 {
            run(SpscQueue::bounded(capacity), Some(capacity), seed, 600);
        }
    }
}
