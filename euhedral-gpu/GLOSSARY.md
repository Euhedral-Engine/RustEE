# euhedral-gpu glossary

The implementation-level terms of [`euhedral-gpu`](README.md): how a recording is identified and captured,
and the rings and caches under a stream. The terms a user submits work with (streams, markers, retirement,
modules, sequences, graphs) and the lifecycle are in the [top-level glossary](../GLOSSARY.md). Paths are
relative to this crate. Entries run from the lowest layer up: streams, recording, caches.

**marker ring** ([`StreamOptions::markers`](src/api.rs), [`ring`](src/ring.rs)): each CUDA stream has a ring of events for its
markers (256 by default), made by any number of threads. A number is claimed by compare-and-set once the slot's previous
marker completed (`mark()` returns `Err(Busy)` rather than reuse a slot whose marker has not completed), the claimant records
the event and stores the number's entry, and a scan finds markers complete in order, stopping at a number whose entry is not
stored yet; a claimant that cannot record stores a skipped number. An await on a marker that already completed adds no device
wait. An await that races a claimant re-recording a slot is made safe without a lock: the awaiter raises a per-slot pin before
it enqueues its wait, and a claimant that finds the slot pinned waits out the pin (one driver call) or gives its number up.

**shared lane** ([`Lane`](src/api.rs), [`cuda::CudaQueue`](src/cuda/stream.rs)): the shared side of a stream that any worker
submits to through `&self`: `submit`, `submit_marked` (the marker is claimed before the work, so a refusal comes before it) and
`mark`, and the graph-facing `LaneControl` (`await_marker`, `notify_retired`, `confirm_retired`). It has no retirement channel
and keeps no values: each retirement names the `RetireCell` its callback fills, and a stage's held values are released by its
`retired` hook. `Stream::shared()` is the handle; `open_shared_lanes` opens a device's lanes as a `LanePool` typed over it.
The driver stream is destroyed, after its work, with the last handle.

**retirement queue** ([`retire`](src/retire.rs), [`cuda::retire`](src/cuda/retire.rs)): each stream owns
one unbounded lock-free MPSC queue. The driver's callback (`cuStreamAddCallback`, which reports the status
of the work before it exactly once, even after a device fault) only pushes that status onto it: no driver
call, no lock, no free, and no allocation while fewer than `UNPOLLED_RETIREMENTS` (252) statuses wait
unpolled. A retirement's number is not sent: a stream's callbacks run in the order they were enqueued, so
the n-th status that arrives belongs to the n-th retirement armed. `poll_retired` drains the queue and
dispatches. If the device is poisoned, it reports the retirements still outstanding with
`POISONED_STATUS`; a status that arrives with no retirement armed poisons the device with `Invalid`.

**kept values** ([`Enqueue::keep`](src/api.rs), [`kept`](src/kept.rs)): values a submitter holds until a
retirement is confirmed, so the host buffers a copy reads stay live. The submitter owns them in a
`PlainQueue`, so releasing them needs no lock. The values a recording keeps are frozen into an immutable
`Arc<[..]>` that the recording's own run, its graph and every replay share.

**shape and fingerprint** ([`ShapeHasher`, `Fingerprint`](src/shape.rs)): the shape of a sequence is its
operations in order with their streams, order edges, kernels, geometry, argument layouts and copy
parameters; the fingerprint is 128 bits of it. A changed shape is `GpuError::ShapeChanged`.

**shadow stream** ([`CudaSequence`](src/cuda/graph.rs)): the stream a recording captures a stream's work
on, so the real stream is never in capture mode and a failed capture never disturbs the work.

**shared graph cache, instance, graph hold** ([`SharedGraphCache`](src/graph_cache.rs), [`instances`](src/instances.rs), [`GraphHold`](src/hold.rs)): the graph cache of shared lanes. A recorded graph is used by one thread at a time, so a key that several workers run at once has several instances, in a fixed table of slots whose one atomic word says empty, free or busy: a worker takes a free instance of its key by one compare-and-set, and one that finds none records an instance of its own (it never waits). The instance goes back once the replay is enqueued; the recording\'s host buffers go to the caller\'s `GraphHold`, which the stage drops in `Stage::retired`, after the device work ended. Capacity counts instances.

**graph cache** ([`GraphCache`](src/graph_cache.rs)): a lane's recorded graphs by key. A hit runs the body
as an update pass and replays the graph; a miss runs the body for real once and records it; a body whose
shape no longer matches is run for real and recorded again, in place when the driver can. `warm` records a
missing key and leaves a recorded one alone. The cache holds at most `capacity` graphs (128 by default): at the bound the
oldest goes to the stale list, which `free_stale` frees. After the first record of a key, a hit allocates
nothing.

**span** ([`Span`](src/spans.rs)): a timed span of one stream's work. A stream keeps
`StreamOptions::spans` of them (16 by default, two timing events each); `span_begin` is `Err(Busy)` when
the ring is full of open or unread spans, and a stream opened with 0 refuses them.

**texture object** ([`cuda::texture`](src/cuda/texture.rs)): `TexObject` over pitched device memory,
described by a `View` (filter, address and coordinate modes). The object does not own the memory. A view
whose base or pitch misses the device's alignment, or whose pitch is less than a row, is an `Err` before
the driver is asked.
