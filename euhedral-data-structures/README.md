# euhedral-data-structures

Cache-line padded atomics and lock-free chunked queues for RustEE. It is a port of the upstream
Java `euhedral-data-structures`, padded the same way: where Java pads with dead `long` fields, a
`#[repr(align(128))]` wrapper ([`CachePadded`](src/atomics/padding.rs)) pads the same groups of
fields in the same order.

**Where it sits.** No dependencies. `euhedral-core` builds on it: each partition of a
`ControlPlaneCache`, the upstream registry's per-worker handle queues, the `QueueIngestSink` and
the `FrameManager`'s recycle queue are queues from this crate.

## Modules

| Module | Main types |
|---|---|
| [`queues::queue`](src/queues/queue.rs) | `Queue<T, P, C>` and its aliases `SpscQueue`, `MpscQueue`, `SpmcQueue`, `MpmcQueue`; the role handles `Producer` and `Consumer`; the roles `Single` and `Multi`; `Sink`, which a consumer can drain into. |
| [`queues::partitioned`](src/queues/partitioned.rs) | `PartitionedQueue` (and its four aliases): independent queues behind one handle pair, with `PartitionedProducer` and `PartitionedConsumer`. |
| `queues::PlainQueue` ([source](src/queues/plain_queue.rs)) | A single-thread FIFO with the same batch API and no synchronisation. |
| [`queues::queue_utils`](src/queues/queue_utils.rs) | Sizing helpers: `MAX_SIZE`, `round_chunk_size`, `chunk_mask`, `unsigned_multiply_high`. |
| `queues` internals: [`core`](src/queues/core.rs), [`chunk`](src/queues/chunk.rs) | The chunked ring buffer behind every queue, its slots, and the pool that recycles retired chunks. Private. |
| [`sync`](src/sync/mod.rs) | `SetOnce<T>`: a lock-free cell installed at most once by a compare-and-set and read with an acquire load (its loom model runs with `RUSTFLAGS="--cfg loom"`). `ClaimCell<T>`: plain state one thread at a time claims with a compare-and-set that fails instead of waiting (Java's `draining` flag), released by a drop or `release_seq_cst`; it has a loom model too. `LifecycleMutex<T>` and `LifecycleRwLock<T>`: the locks for start, stop and rebalance state. `HotScope`: the debug check that aborts when one is taken inside code that runs once per unit of work. |
| [`atomics`](src/atomics/mod.rs) | `CachePadded<T>`; `AtomicDouble`; `PaddedAtomicLong`, `PaddedAtomicDouble`; `PaddedAtomicLongArray`, `PaddedAtomicDoubleArray` (each element on its own cache line); `PaddedLongAdder`, `PaddedDoubleAdder` (striped counters). |

## Key concepts

### Role handles

Upstream lets any thread call any queue method and documents who may call what. Here a queue hands
out **role handles** instead, so the whole public API is safe:

- A **`Single`** role has at most one live handle at a time, and its methods take `&mut self`.
  Asking for a second handle while the first is alive panics; dropping the handle frees the role.
- A **`Multi`** role's handle is `Clone` and may be used from many threads at once.

So `MpscQueue<T>` gives a cloneable `Producer<T, Multi>` and one `Consumer<T, Single>`.

### Chunked queues

Elements live in power-of-two ring **chunks**. An unbounded queue (`Queue::new(chunk_size)`) links
a fresh chunk when its ring is full, and the consumer follows the link; `with_pool` keeps retired
chunks for reuse. A bounded queue (`Queue::bounded(capacity)`) holds exactly `capacity` elements,
and `offer` hands the element back when it is full. Every slot is published with a release store
and read with an acquire load, so a consumer that sees an element sees everything its producer
wrote before publishing it.

A consumer can `poll` one element, `peek` at it, `drain` up to a limit into a closure, `drain_while`
up to the first element a `stop` predicate matches (leaving it queued), or `drain_to` another
queue's producer or a `Vec`. A `Multi` consumer takes a consumer lock for each call: `poll` returns
`None` and `drain` stops early if another consumer holds it. `with_max_consume_batch` caps how many
elements one consumer takes per lock hold. Here `drain*` means taking items out of a container;
in `euhedral-core`, "drain" means stopping work and finishing what is in flight (see
[`GLOSSARY.md`](../GLOSSARY.md)).

### Partitioned queues

A **partitioned queue** is several independent queues behind one handle pair. Each partition has
its own allocation and padded state, so partitions never share a cache line. Producers pick a
partition (`offer_to(partition)`, `offer_seeded(seed)`: the same seed always picks the same
partition, or `offer` for a random one); consumers visit the partitions in order. There is no
ordering between partitions. A partitioned queue with one partition is FIFO.

```rust
use euhedral_data_structures::queues::{MpscQueue, PartitionedMpscQueue};

// Many producers, one consumer.
let queue = MpscQueue::<u32>::new(64);
let producer = queue.producer();
let mut consumer = queue.consumer();
producer.clone().offer(1).unwrap();
producer.offer(2).unwrap();
let mut taken = Vec::new();
assert_eq!(consumer.drain_all(|item| taken.push(item)), 2);
assert_eq!(taken, [1, 2]);

// Four partitions; the same seed always lands in the same partition, in order.
let partitioned = PartitionedMpscQueue::<u32>::with_partitions(4, 64);
let producer = partitioned.producer();
let mut consumer = partitioned.consumer();
for item in 0..3 {
    producer.offer_seeded(7, item).unwrap();
}
let mut taken = Vec::new();
consumer.drain_all(|item| taken.push(item));
assert_eq!(taken, [0, 1, 2]);
```

### Padded atomics

The padded atomics follow Java's access modes: `get`/`set` are sequentially consistent,
`*_acquire`/`*_release` map to the same orderings, and `*_opaque`/`*_plain` map to relaxed (Java's
plain access has no atomic-free counterpart on a shared `&self`). Integer arithmetic wraps like a
Java `long`. `AtomicDouble` stores an `f64` as its bits, so compare-and-set compares raw bits: `NaN`
matches itself and `0.0` does not match `-0.0`. The adders spread writers over padded slots and sum
them on read; a read is not a snapshot.

## Build and test

```sh
mise exec -- cargo test -p euhedral-data-structures
```

The integration tests in `tests/` cover each queue flavour (SPSC, MPSC, multi-consumer,
partitioned, transfers between queues), a model check of the queue against `VecDeque`, and the
padded types' layouts. The threaded tests run under deadlines, so a livelock fails instead of
hanging. No features or special hardware are needed. The example above is compiled and run as a
doctest.
