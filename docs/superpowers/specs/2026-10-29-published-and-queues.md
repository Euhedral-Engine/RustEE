# `Published` and the epoch domain, against queue-based hand-off (design note)

Review question on `published.rs`: why not a global MPSC or MPMC? The queues drain in batches, are asymmetric, and
would remove a place where correctness lives.

## What `published.rs` does today

| part (lines) | job | who needs it |
|---|---|---|
| `Published<T>` (`current` pointer, `store`, `load` under a guard) | a shared `Option<Arc<T>>` that readers load with one acquire load and no count | routing generation (vertex), edge `parent`, cache directory (one cell per CPU), lake generation |
| `Retired` + the retirement chain (`push_chain`, swap-all `try_reclaim`) | frees a replaced value once no reader can hold it | the four users above |
| `EpochDomain` (push-only slot list, global epoch, `floor`, `passed`) | tells a writer that every reader that began before a swap has left | the same |
| `EpochSlot` / `EpochGuard` (per-thread announcement, `quiescent`, `idle`, `pin`) | the reader's side: its announcement is the proof of "I hold nothing from before" | workers (in-cycle: one load), any other thread (two stores) |
| `barrier.rs` (`membarrier`, `FlushProcessWriteBuffers`) | makes the reader's announcement free of a fence | the asymmetry the review refers to |
| `registry.defer_after_readers` / `run_due` (4b) | runs the release of a retired routing generation when the epoch has passed | vertex |

About 1 500 lines with the tests, three `unsafe` surfaces (the slot list, the retirement chain, the syscalls), a loom
model, and platform code.

## The queue-based alternative: message passing, no shared pointer

A change (a remap, a new parent, a new cache) is not published to a shared cell. The writer sends it to every worker's
mailbox (an MPSC queue the registry already creates per worker for upstream handles, or one per worker beside it).
Each worker, at the start of its cycle, drains its mailbox in one batch and applies the messages to its own view: a
plain per-thread table `vertex id -> Arc<RoutingState>`, `edge id -> parent`, `cpu -> Arc<Cache>`. A frame reads only
its worker's view: no atomic, no shared line. The replaced `Arc` is dropped by the worker; when the last clone drops,
the generation's `Drop` releases its downstreams (closing them, running their retirements). `Arc` is the reclamation
protocol, and the "grace period" is "every worker has drained its mailbox", which a worker reaches at its next cycle
start, or at once when it parks (it drops its view before it parks and refetches under the registry lock when it wakes).

- **Retirement chain and swap-all reclaim:** disappear (the `Arc` count does it). The mailbox is the batch-drained queue.
- **Slot registry:** the registry's list of workers, which already exists (`State::workers`), under its lock; a worker
  that registers gets the current state with its registration.
- **Announcement and barrier:** both disappear. There is no reader-side proof to make cheap, so no `membarrier`, no
  platform code, no asymmetric fence to model.
- **Frame-path cost, in loads and RMWs per frame:**

| | loads | RMW |
|---|---|---|
| today (epoch pin, 4b) | thread-local slot lookup (borrow, linear search), pin (1), closed (1), routing pointer (1) | 0 |
| mailbox view | thread-local view (1-2), entry for the vertex (1), closed (1) | 0 |

  The same order of cost. The gain is not speed on the frame path; it is that the reclamation and the proof of
  safety are `Arc` and a queue.
- **Guarantees kept:** a remap is seen by a worker at its next cycle start; a worker mid-cycle finishes on the old
  generation, as with the epoch. The old generation's downstreams close only after every worker dropped its copy.
- **Guarantees lost or changed:**
  - release latency is bounded by a worker's cycle (or its park), not by the next reader leaving; a stalled worker holds
    its generation (as a stalled pinned reader does today);
  - a remap costs `N` pushes (31 here) and `N` `Arc` clones, per change; construction of a lattice sends about
    (edges + caches) x workers messages once;
  - per-vertex state is duplicated per worker (small: a table of at most a handful of handles).

## Where the queues cannot cover a user, and why

1. **Readers that are not workers.** A mailbox is drained by its owner at a point it chooses. A thread that is not a
   registered worker has no such point. Three kinds exist:
   - **`Lake::offer` and `admit` are called by the application's own producer threads**, by design the hot path of the
     ingest front. They read the lake's attached generation on every call. They cannot have a mailbox the lake drains
     for them. Options: (a) a producer *handle* that registers a mailbox and drains it inside `offer` (API change; one
     load when empty); (b) keep one small `Published`-like cell for the lake only; (c) a lock on the producer path
     (against the rule).
   - **Public `push`/`routing_state` on a vertex from any thread** (tests, a hot source pushed by an application
     thread). These can take a slow path under the registry lock; they are not frame paths of the lattice.
   - **Lifecycle code** (remappers, closers) reads under the lock anyway.
2. **The cache directory is read by every worker per steal attempt** (a table of `cpu -> cache`). Per-worker copies work
   (a message per cache created or closed), but a *closed* cache then stays referenced by every worker's table until
   they drain, which delays its drop by a cycle. Acceptable, but a different life cycle from today's `Weak` upgrade.
3. **Edge parents** are read per hop by `climb` for every request and pull. Per-worker copies of the resolved chain per
   edge work, with one message per `set_parent`. This is the largest plumbing change (every edge, every worker).

So: the queues cover routing (C6), the cache directory (C5), edge parents (C8/C9) and delete `EpochDomain`, `Retired`,
the barrier and `run_due` *for workers*. They do **not** cover the lake's external producers without the API change in
1(a), and they leave the non-worker slow path to the registry lock.

## Options

- **A. Mailboxes for routing, directory and parents; a producer handle for the lake.** Deletes `published.rs`, `barrier.rs`,
  the epoch slots and `run_due` (about 1 500 lines and two `unsafe` surfaces); adds a mailbox type (a queue of
  closures or an enum), a per-worker view, and an API change on `Lake` (producers drain their own mailbox). Largest
  change; moves the correctness argument into `Arc` and the queue.
- **B. Mailboxes for routing and the directory only; `Published` stays for edge parents and the lake.** Smaller, but
  keeps all of `published.rs` and its barrier, so it removes no place where correctness lives.
- **C. Keep the epoch scheme (today).** The cost on the frame path is zero RMW and a narrow spread (4b: 412-414 M, sd
  3-5 M); the reclamation proof stays in `published.rs` (loom, Miri, stress, `membarrier`).

Recommendation: A, if the lake's producer handle is acceptable. It is the only option that deletes the machinery
rather than moving it. Before building it: confirm with the user that `Lake::offer` may take a per-producer handle
(or tell me which of 1(a)-(c) they prefer), because it is the one case the queues cannot cover on their own.
