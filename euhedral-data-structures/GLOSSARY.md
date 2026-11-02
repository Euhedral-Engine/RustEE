# euhedral-data-structures glossary

The implementation-level terms of [`euhedral-data-structures`](README.md): how its types are laid out and
how the workspace polices locks. The types a user builds with (queues, `ClaimCell`, the lifecycle locks and
`HotScope`) are in the [top-level glossary](../GLOSSARY.md). Paths are relative to this crate. Entries run
from the lowest layer up: padding, queue storage, queues, the lock policy.

## Padding

**padded atomics** ([`atomics`](src/atomics/mod.rs)): atomics, arrays and striped adders with each value
alone on a 128-byte block (`CachePadded`), so neighbours never share a cache line. The access modes follow
Java's: `get` and `set` are sequentially consistent, `*_acquire` and `*_release` map to the same orderings,
`*_opaque` and `*_plain` map to relaxed. Integer arithmetic wraps. `AtomicDouble` stores an `f64` as its
bits, so compare-and-set compares raw bits. The adders spread writers over padded slots and sum them on
read; a read is not a snapshot.

## Queues

**chunk** ([`queues::chunk`](src/queues/chunk.rs)): a power-of-two ring of slots that a queue keeps its
elements in. Each slot is published with a release store and read with an acquire load, so a consumer that
sees an element sees everything its producer wrote before publishing it. An unbounded queue links a fresh
chunk when its ring is full, and can pool retired chunks for reuse.

**bounded queue, unbounded queue** ([`Queue::bounded`, `Queue::new`](src/queues/queue.rs)): a bounded queue
holds exactly its capacity, and `offer` hands the element back when it is full. An unbounded queue links a
fresh chunk when its ring is full, so `offer` never refuses; `with_pool` keeps retired chunks for reuse.

**drain methods** ([`Consumer::drain`, `drain_while`, `drain_all`](src/queues/queue.rs)): take items out of
a container, up to a limit, up to the first item a `stop` predicate matches (leaving it queued), or all of
them. They are not the lattice's *drain* (stop accepting work and finish what is in flight).

**plain queue** ([`PlainQueue`](src/queues/plain_queue.rs)): a FIFO for one thread with the same batch API
as the concurrent queues and no synchronisation.

## Lock policy

**HOT-FIX tag**: `reason = "HOT-FIX(<id>)"` on the `#[allow]` of a hot-path lock the audit lists, `<id>`
being the audit's finding (`C4`, `G10`, `I7`, ...). `euhedral-core/tests/hot_fix_ratchet.rs` counts the
distinct ids, and the count only goes down as the locks are replaced. The same test accepts, on an allowed
lock or blocking call, only `LIFECYCLE(<id>)` (start, stop or rebalance work that still uses a standard
blocking primitive), `JAVA-BACKOFF` (a call only: a back-off the upstream design has too), a test-only
attribute, or a file on its exempt list (the lifecycle locks themselves and the simulated device).
