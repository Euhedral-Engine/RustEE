//! Drives every queue flavour with random single-threaded operation sequences and checks each result
//! against a `VecDeque`. Small chunks and pools force constant chunk rollover, reuse and ring wrap-around.

use euhedral_data_structures::queues::{
    Consumer, MpscQueue, Multi, Producer, Queue, Role, Single, SpscQueue,
};
use std::collections::VecDeque;

/// The producer operations the model uses, whatever the role's receiver type.
trait ModelProducer {
    fn put(&mut self, value: u64) -> Result<(), u64>;
    fn put_all(&mut self, items: &mut Vec<u64>) -> usize;
}

impl ModelProducer for Producer<u64, Single> {
    fn put(&mut self, value: u64) -> Result<(), u64> {
        self.offer(value)
    }

    fn put_all(&mut self, items: &mut Vec<u64>) -> usize {
        self.fill(items)
    }
}

impl ModelProducer for Producer<u64, Multi> {
    fn put(&mut self, value: u64) -> Result<(), u64> {
        self.offer(value)
    }

    fn put_all(&mut self, items: &mut Vec<u64>) -> usize {
        self.fill(items)
    }
}

/// The consumer operations the model uses.
trait ModelConsumer {
    fn take(&mut self) -> Option<u64>;
    fn look(&mut self) -> Option<u64>;
    fn take_up_to(&mut self, limit: u64) -> (u64, Vec<u64>);
    fn take_until_multiple_of(&mut self, divisor: u64, limit: u64) -> Vec<u64>;
    fn wipe(&mut self);
}

impl ModelConsumer for Consumer<u64, Single> {
    fn take(&mut self) -> Option<u64> {
        self.poll()
    }

    fn look(&mut self) -> Option<u64> {
        self.peek().copied()
    }

    fn take_up_to(&mut self, limit: u64) -> (u64, Vec<u64>) {
        let mut drained = Vec::new();
        let count = self.drain(|v| drained.push(v), limit);
        (count, drained)
    }

    fn take_until_multiple_of(&mut self, divisor: u64, limit: u64) -> Vec<u64> {
        let mut drained = Vec::new();
        self.drain_while(|v| drained.push(v), |v| v % divisor == 0, limit);
        drained
    }

    fn wipe(&mut self) {
        self.clear();
    }
}

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

fn run<P: Role, C: Role>(queue: Queue<u64, P, C>, capacity: Option<usize>, seed: u64, steps: usize)
where
    Producer<u64, P>: ModelProducer,
    Consumer<u64, C>: ModelConsumer,
{
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
                    producer.put(value).is_ok(),
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
                    producer.put_all(&mut items),
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
                    consumer.take(),
                    model.pop_front(),
                    "{}",
                    context(step, "poll")
                );
            }
            5 => {
                assert_eq!(
                    consumer.look(),
                    model.front().copied(),
                    "{}",
                    context(step, "peek")
                );
            }
            6 => {
                let limit = rng.below(6);
                let (count, drained) = consumer.take_up_to(limit);
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
                    consumer.wipe();
                    model.clear();
                } else {
                    let divisor = 2 + rng.below(5);
                    let limit = 1 + rng.below(8);
                    let drained = consumer.take_until_multiple_of(divisor, limit);
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

    let (_, rest) = consumer.take_up_to(u64::MAX);
    assert_eq!(
        rest,
        model.into_iter().collect::<Vec<_>>(),
        "seed {seed} final drain"
    );
    assert!(queue.is_empty());
}

macro_rules! model_tests {
    ($module:ident, $queue:ident) => {
        mod $module {
            use super::*;

            #[test]
            fn unbounded_queues_match_a_vecdeque_over_random_operations() {
                for chunk_size in [2, 3, 4, 8, 16] {
                    for max_pooled in [0, 1, 3] {
                        for seed in 1..=60 {
                            run($queue::with_pool(chunk_size, max_pooled), None, seed, 600);
                        }
                    }
                }
            }

            #[test]
            fn bounded_queues_match_a_vecdeque_over_random_operations() {
                for capacity in [1, 2, 3, 4, 5, 7, 8, 9, 15] {
                    for seed in 1..=60 {
                        run($queue::bounded(capacity), Some(capacity), seed, 600);
                    }
                }
            }
        }
    };
}

model_tests!(spsc, SpscQueue);
model_tests!(mpsc, MpscQueue);
