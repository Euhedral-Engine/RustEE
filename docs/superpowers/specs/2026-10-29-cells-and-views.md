# Cells and views: how the lattice's changing state reaches its workers (design note)

Rule: wherever threads meet to hand something off, a queue variant carries it. The lattice's changing state (a vertex's
routing generation, an edge's parent, a CPU's cache, the lake's attached set) is read by every worker on every frame and
replaced only by lifecycle work. Java publishes each with a volatile or release store and lets the collector keep a
replaced value alive for a reader that still holds it. Rust has no collector.

## What was considered

- **An epoch domain with a `Published<T>` cell** (a pointer read under a per-thread announcement, replaced values
  reclaimed when every announcement has passed, a process-wide barrier to make the announcement free of a fence). Built
  and measured (412-414 M ops/s in the high-contention benchmark, spread 3-5 M). It was dropped: about 1 500 lines with
  three `unsafe` surfaces and platform code, and the correctness argument lives in a slot list, a retirement chain and a
  barrier.
- **A global MPSC or MPMC queue of changes.** Every reader would have to consume every change; one queue cannot be
  drained by many readers each seeing all of them. A queue per reader does that.

## What is built

- A **cell** (`UpstreamRegistry::cell`) holds the current value under a lifecycle lock. A write (`set_cell`,
  `set_cell_if`) stores it and, in the same critical section, offers it to the mailbox of every reader: one
  `MpscQueue` per reader (many writers, one consumer).
- A **reader** owns a mailbox and a thread-local **view**. A worker drains its mailbox in one batch at the start of each
  cycle and at the start of an outermost read; a read is a load from the view: no lock, no atomic read-modify-write, no
  shared line, no barrier.
- `Arc` is the reclamation: the writer and each view hold a clone, the last holder to let go drops the value, and a
  routing generation releases its downstreams from `Drop`. A thread that exits drops its view and withdraws its mailbox.
- **No reader locks.** A worker holds the values (a *full* reader). Any other thread that reads gets a *light* mailbox on
  its first read, through the same queue, and holds weak references: a read upgrades one for the length of the call
  (one reference-count update), so a thread that reads once and goes idle never keeps a replaced generation alive and
  a close never waits for it. A long-lived producer thread can `register_reader` to become a full reader.

## Costs and limits

- A change costs one queue offer and one `Arc` clone per reader (31 here), under the lock; a lattice's construction
  sends about (edges + caches) x workers messages once.
- A full reader that stops reading keeps what it last saw until it drains; a worker drains every cycle and its parks are
  timed.
- `ParentRef` and `CacheRef` hold weak references (a strong one would form a cycle): the climb and the steal upgrade one
  per hop, a reference-count update that a borrowed reference would avoid but only with a proof that the target outlives
  the call, which a view cannot give.
- A light reader that finds its weak reference dead drains again: the update that replaced the value was queued before
  it died, so the newer value is there.

## Other hand-rolled hand-offs, for a later chunk

Each is a place a queue variant could carry the hand-off:
- `OneShot` (a result from a completer to a reader that polls): a bounded single-producer single-consumer queue of one.
- The sequencer's ready flags and work-in-progress count: the order is already a ring; the "ready" hand-off is the part
  a queue could carry (an MPSC of completions drained by the releaser).
- `Join`'s slot writes and arrival count: a fan-in that an MPSC of parts could carry.
- The lake's attach path (a lock and a cell): a message to the registry's mailboxes (done for the read side).
- `Completions`' incoming list and `Sequencer`'s `wip`: already queues plus a claim.
- The interceptor's `serving` pointer and the registry's per-worker handle queues: queues already.
