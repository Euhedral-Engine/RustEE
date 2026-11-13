# Changelog

Changes to the lattice's execution model, newest first. Each entry says what changed, why, how to use it, its exact
semantics and the tests that pin it.

## A launch can be a programmatic dependent launch (2026-11-13)

**What.** `Launch` gains `programmatic: bool` (false from `Launch::linear`) and the builder-style `Launch::programmatic(bool)`. On CUDA a programmatic launch goes through `cuLaunchKernelEx` with `CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION` set to 1; any other launch still goes through `cuLaunchKernel`, unchanged. The flag is part of the shape fingerprint. `FakeOp::Launch` gains `programmatic`.

**Why.** A kernel that reads what the previous kernel on its stream wrote can overlap its own start-up with the previous kernel's tail, and wait for the data itself, but only if it is launched with the attribute; a lane had no way to ask for it, eagerly or in a graph.

**How to use it.** `Launch::linear(n, block).programmatic(true)`, and the kernel calls `asm volatile("griddepcontrol.wait;");` before it reads the previous kernel's output (without the wait the kernel races). Needs compute capability 9.0 or newer. A struct literal of `Launch` needs the new field; a match on `FakeOp::Launch` needs `..`.

**Semantics.**
- A recording captures the launch with the attribute, so the captured graph holds a programmatic edge (the device test reads the edge type through the `test-hooks` method `CudaGraph::edge_types_into`, and sees a plain edge without the flag). An update pass only changes node parameters, never the edges; the executable's edges are not queryable, so the test pins the results after an update, not the edge.
- The flag is in the fingerprint: a body with the flag never matches a graph recorded without it (`ShapeChanged` on an update, a re-record in a graph cache), nor the reverse.
- If the loaded driver has no `cuLaunchKernelEx`, a programmatic launch is `GpuError::Invalid` at the launch call; launches without the flag work as before.
- The fake backend carries the flag and reports it in `FakeOp::Launch`.

**Tests.** `tests/fake_graph.rs`: `the_programmatic_flag_changes_the_fingerprint_and_a_body_with_it_does_not_hit_a_recording_without`, `the_fake_reports_whether_a_launch_was_programmatic`; the shape tests in `shape.rs`; device: `tests/cuda_programmatic.rs` `a_programmatic_chain_gives_the_same_result_eagerly_and_from_a_recorded_updated_graph`.

## A failed recording is not a failed unit of work (2026-11-10)

**What.** Three changes to `SharedGraphCache` and the CUDA capture.
- A re-record whose capture fails (`GpuError::Capture`) after the body ran returns `Ok(Ran::Direct)` instead of the error; with the body not run, the unit runs directly. Any other error is still an error.
- A capture that fails to end returns an error that carries the first refused shadow call, then the end-of-capture error.
- A key whose first recording failed is recorded again at its next sighting, and is marked unrecordable after `MAX_FAILED_RECORDINGS` (3) consecutive failed recordings or re-records. `GraphStats` gains `failed_rerecords`, `failed_records` and `unrecordable_keys`.

**Why.** A recording runs the real launches on the lane first and captures on a shadow stream, so a capture that the driver invalidates (another thread's work in the same context can do it) comes after the work ran. The first-recording path already returned `Ran::Direct` for it; the re-record path sent the error to the caller with `?`, who then saw a failure for a unit of work that was done. The end-of-capture error ("the capture was invalidated") hid the call the driver had refused, which is the one that names the cause. And one such failure marked the key unrecordable for the life of the cache, with no count to read.

**How to use it.** Nothing to change. A caller that reads `GraphStats` sees the new fields (a struct literal of `GraphStats` needs `..Default::default()`). `failed_records`, `failed_rerecords` and `unrecordable_keys` are a shared cache's; the owned `GraphCache` leaves them at 0.

**Semantics.**
- A failed re-record replaces nothing: the instance keeps its old recording and goes back to the pool, so the key's next run finds the shape changed again and tries again.
- `Ran::Direct` after a failed recording or re-record means the body ran exactly once, as direct launches. A failure to instantiate the recorded graph (out of memory included) after the work ran is a failed recording too. An error of the body itself is returned as it is, never absorbed; so is any error that is not a `Capture`.
- A key has one count of *consecutive* failed recordings and re-records, saturated at 3 and cleared by a successful recording or re-record (never on the replay path). At `MAX_FAILED_RECORDINGS` the key is marked unrecordable; a failed re-record at the bound also drops the instance. After that no unit of the key pays a capture: a free instance whose shape changed is dropped and the unit runs directly, and a unit with no free instance runs directly. A free instance whose shape still matches keeps replaying.
- A failed first recording leaves no instance; the next sighting records (the key was seen).
- The counts live in a fixed table (a third sightings table, in the two spare bits of the tag's word): no lock, no allocation and no wider entry on the run path. A key whose probe of eight slots is taken by other keys' words is counted at the bound at once, so an overflow marks the key rather than letting keys erase each other's counts.
- A body that asks for the library stream is unrecordable at once (a property, not a failure); it counts in `unrecordable_keys`, not in `failed_records`.
- `failed_rerecords`, `failed_records` and `unrecordable_keys` are the shared cache's only.
- A capture that fails to end returns the first refused shadow call, then the end-of-capture error; an earlier error of another kind (a failed join of the real streams) is returned as it is.

**Tests.** `graph_cache::shared_tests`: `a_rerecord_that_fails_after_the_body_ran_is_a_direct_unit_and_the_next_run_records_again`, `a_rerecord_that_fails_before_the_body_ran_runs_the_unit_directly`, `a_capture_that_fails_once_runs_the_work_and_the_key_records_at_a_later_sighting`, `a_key_whose_recordings_keep_failing_becomes_unrecordable`, `a_capture_that_fails_before_the_body_ran_still_runs_the_unit_once`, `failed_recordings_are_counted_per_key`, `rerecords_that_keep_failing_mark_the_key_and_drop_its_instance`, `a_marked_keys_instance_replays_while_its_shape_matches_and_goes_when_it_changes`, `a_full_probe_counts_a_key_at_the_bound`, `a_recording_that_succeeds_clears_the_keys_count`, `failures_apart_do_not_add_up`, `threads_failing_one_key_together_lose_no_update`, `a_bodys_own_capture_error_is_returned_not_absorbed`; `cuda::graph::tests`: `a_failed_end_of_capture_names_the_first_refused_call`, `a_failed_end_of_capture_alone_is_reported_as_it_is`, `an_earlier_error_of_another_kind_is_returned_as_it_is`. The device stress tests in `tests/cuda_lane_graphs.rs` assert that no recording failed.

## A request that issues no frame shows its handle empty (2026-11-06)

**What.** A worker's view of an upstream handle after a request is productive only if the request issued frames: a push through the handle was routed, or the thread's satisfied-request counter moved across the request. A request that issued none marks the view unproductive, as a pull that finds nothing does. Before, a request kept the view the worker already had.

**Why.** Whether frames were issued is the evidence of productivity, for a request as for a pull (Java's worker checks whether its satisfied-request counter changed after the request). Keeping the prior view meant that a view could turn productive and never turn back for a worker that only requests, as one on the Staged path does: a handle whose polling key had once been pushed read productive for good, `should_idle` kept the worker in, and it yielded in a loop it never left. A lattice that had run work (a body-cost history, a body estimate of a few hundred nanoseconds from the polls) lost its workers to that loop one at a time, until every core was at 100 % running a few thousand cheap frames a second.

**How to use it.** Nothing to change. A source that issues frames only on some requests (a polled key at its pacing) makes its handle productive for the requests that issue them, and a worker that finds only such sources parks between them.

**Semantics.**
- A request that issued a frame (a routed push through the handle, or a moved satisfied-request counter) shows the handle productive; one that issued none shows it unproductive, whatever the view was.
- A pull is as before: frames found show productive, none shows unproductive, a pull that stopped at a frame or could not be made changes nothing.
- A handle never serviced by a worker is still productive in that worker's view.

**Tests.** `a_request_that_issues_no_frame_shows_an_empty_source`, `a_worker_on_the_staged_path_stops_entering_a_source_that_issues_nothing` (a worker with a body-cost history on the Staged path over a source that issues one key and then nothing: it entered the handle at every cycle before); the acquisition, last-push and push-only-source tests state the rule.

## A pull serves no request: `LaneSource` and the ingest queue (2026-11-05)

**What.** Two more sources pushed from the tail of a `pull`. Neither does now.
- `LaneSource::pull` ended by serving the requests that arrived while it held the source's claim (`drain_requests`). Serving drains the lane's retirements, runs its waiting items and admits new ones (the lane's work), and pushes the ready frames within the banked demand. A frame the pulling worker runs inline can request the lane, and so can another thread, while the pull holds the claim.
- The ingest queue (`QueueSource`, behind `QueueIngestSink`) ended every pull with `after_release`: the waiting `clear` and the owner's hook, then `drain_requests`, which pushes the queued frames to the receiver within the banked demand. It was reached when a request raced a pull outside the lattice's serialization, or came from the pull's own consumer.

**Why.** A worker that has entered from a pull must not push: only `request` and a push keep the order of frames, and a worker stops its pull before an ordered frame. The lane's serving is also long work (items run there), and nothing long may go out through a `pull`.

**How to use it.** Nothing to change for a lattice. A caller that drives these sources by hand and relied on a pull to serve a request it made from inside a consumer (or from another thread while the pull held the source) finds it served by its next `request` of the source. `request` returns nothing, so a caller cannot tell that one was refused; `LatticeSource::request` documents this.

**Semantics.**
- The tail of a `LaneSource` pull does nothing more than record its metric. The tail of a queue pull is the lifecycle work only: the waiting `clear` and the owner's drain hook (`drain_finished`). Both leave the wakeup set and the demand banked.
- **No request is lost, and it waits for the lattice's next request of the source.** A refused `request` banks its demand first and then sets the wakeup, and the wakeup is cleared only by the swap in `drain_requests`, which a `request` runs. So it stays set until the next `request` of that source, which then serves everything banked, whatever its own size. That next request comes from the lattice when a worker's whole cycle processed nothing, its cache included (the Direct path requests only then); the upstream queue holds back only the last empty handle of a pull for the request that follows. A visit after a pull that found frames is usually a pull of the next handle, not a request, so under steady load a refused request can wait for as long as the workers keep finding frames to pull; the frames the source already holds are handed out by those pulls meanwhile. No push was added to a pull to shorten that wait.
- `clear` (a producer-side call, not a worker's pull) still serves the requests refused while it held the queue. The discard of the queue (`discard_queued`, run by a kill and, through the owner's `drain_finished`, from the tail of a pull while a kill is under way) serves none: it pushed from the pull's tail in that window, and Java's `discardQueued` never served requests. The kill completes the downstream right after, so a request left pending by it is not lost.

**Work from outside.** Only the lattice's workers call `request` and `pull`. Work that comes from outside (a submitter, a completion poller, another runtime) is offered into a queue the source owns and the outside thread returns; it never calls `request` or `pull` to get its work admitted, since a call made while a worker's `pull` holds the source banks its demand and waits for the lattice's next `request` of that source. The queues already exist: a lane's intake (`LaneIntakes::deliver`, an MPSC per lane and an MPMC per group), the ingest queue (`QueueIngestSink::offer`), the registrations of `Completions`, the submissions of `AsyncIo`, the moves of `Moves`. Nothing was added to them and no wakeup, retry or pending-request machinery was added. `LatticeSource` states the rule, and the lane source and its intake say how the lane takes outside work (a worker admits it at its next `request`; a pull hands out what is ready and serves nothing). The calls from outside that the audit found in this repository are all tests, benchmarks and by-hand drivers standing in for a worker: none in a production path. Tests: `an_outside_delivery_while_a_pull_holds_the_lane_is_run_by_a_workers_next_request` (an outside thread delivers while a pull holds the lane; a thousand pulls never admit it; the next worker `request` runs it; nobody outside banked demand) and, on a real lattice, `work_offered_from_outside_runs_while_two_or_more_workers_are_busy_with_nothing_requesting_it`.

**Java correspondence.** Java's `QueueIngestSink.Delegate.hookOnPull` called `drainRequests()` in its `finally`, the same place. It no longer does, in a small Java change stacked on the Java counterpart of the upstream queue change. Java has no `LaneSource`.

**Tests (each failed on the code before, and passes now).** `euhedral-graph/tests/lane_source.rs`: `a_request_made_by_the_consumer_of_a_pull_is_served_by_the_next_request_and_the_pull_pushes_nothing` (a frame the pulling worker runs requests the lane: before, the receiver got item 3's frame before the pull returned and the item ran in the pull; now neither, and the next `request` runs the item and pushes the frame). `euhedral-core/src/ingest/queue/tests.rs`: `a_request_inside_a_pull_consumer_is_delivered_by_the_next_request_and_the_pull_pushes_nothing` and `a_request_while_a_pull_holds_the_queue_is_delivered_by_the_next_request_not_by_the_pull` (the same with a request from another thread) and `a_discard_from_the_tail_of_a_pull_during_a_kill_serves_no_request` (the kill window). Each asserts the receiver saw nothing when the pull returned, then that the next request delivers.

## A pull that finds nothing holds the handle for the request that follows (2026-11-05)

**What.** `UpstreamQueue::pull` holds back a handle whose pull found no frames (and did not stop at one), so the next `request` services that same handle first. Until now only a handle whose pull stopped at a frame was held back.

**Why.** A demand under `pull_bucket_target` (2048) is one bucket, so a pull and a request each service one handle. A Direct worker that finds nothing runs one pull and then one request, so the rotation moved two handles per cycle. With an even number of handles the pulls always landed on one half of them and the requests on the other: the half that was only ever requested never read as empty, so its productivity stayed at its initial value for ever and `productive_handle_count()` never reached 0; and the half that was only ever pulled was never requested, so a source that does its work in `request` was never reached. An idle lattice with an even number of idle sources kept its first-ranked worker out of the unproductive park: one CPU with 6 handles and nothing to do, against 0.32 with 5.

**How to use it.** Nothing to change. It is how the queue now reads its own rotation: every handle gets both a pull and a request, one handle per cycle, whatever the number of handles.

**Semantics.**
- A pull that ends having found 0 frames and without a stop holds its handle in `pending_request`; the request that follows takes it first, and returns it to the queue behind the others. The rotation therefore advances by one handle per pull-and-request cycle. The same holds when the pull could not be made at all (a closed or draining handle): its request does nothing either.
- A handle that found nothing never displaces one that stopped at a frame: the frame still gets its request first.
- A handle held for finding nothing goes back to the queue when the next pull comes before any request, so pulls reach it again; a complete one is dropped then. A handle held for a stop stays held until a request, as before.
- A pull that found frames holds nothing; a lattice with work behaves as before. Productivity is unchanged: a pull that found nothing marks the handle unproductive, and the request after it is no evidence to the contrary.
- Java's `UpstreamQueue` has the same rotation and gets the same change. Java leaves the held handle in the queue as a hint and clears the hint at the start of a pull that is not holding a stop; here the entry itself is released at that point, so the request services the same handles in both.

**Sources whose key frame leaves only by `request`.** The sources of the change below this one (`Completions`, `AsyncIo`, `Moves`, the lake's idle watch) push their key frame from `request` alone: a `pull` pushes nothing. Before this change an idle or loaded lattice reached only half of an even number of handles with a `request`, so such a key was not pushed for as long as the load lasted: `offers_racing_an_idle_detach_land_in_exactly_one_set` (the lake's idle watch) timed out waiting for the set to detach, and a timer was not seen (`timer_latency`, run by hand). With this change every handle gets a request, and that test is no longer ignored.

**Tests.** `flow_control::upstream_queue::tests`: five fail on the code before: `every_idle_handle_is_pulled_and_requested_whatever_their_number`, `idle_handles_are_serviced_evenly_by_pulls_and_requests`, `a_lattice_of_idle_handles_reads_as_unproductive_whatever_their_number` (1000 cycles, or 600 for the even split, of `pull(64)` then `request(64)` over 1, 2, 3, 4, 5, 6 and 8 idle handles: each is pulled and requested, and the productive count reaches 0), `a_pull_that_found_nothing_holds_the_handle_for_the_request_that_follows` and `control_plane::fragment::tests::a_worker_over_idle_handles_reaches_the_unproductive_park_whatever_their_number` (a worker's real cycles over 1 to 8 idle handles reach a productive count of 0 within twice as many cycles as handles). The others are guards that pass before and after: `a_pull_that_found_frames_does_not_hold_the_handle`, `a_handle_that_found_nothing_is_pulled_again_when_no_request_comes`, `a_handle_that_found_nothing_does_not_displace_one_that_stopped`, `a_hint_to_an_empty_handle_is_gone_when_a_later_pull_finds_frames` (a pull of A finds nothing, a pull of B finds frames, the request goes to the queue's front as in Java).

## `request` and `pull` do no heavy or high-frequency work (2026-11-05)

**What.** Five places did heavy or high-frequency work inside a source's `request` or `pull`. Four of them no longer do:
- `Completions` queried every registered operation on each call (a `Ready`'s `is_ready`, a `Pending`'s `poll`, markers, retirement cells, timers) and called the continuations that build frames.
- `AsyncIo` asked the engine for finished operations (reading the rings' completion queues, flushing the kernel's backlog with a system call, resubmitting short pieces) and called each operation's continuation.
- `Moves` (`euhedral-memory`) called `Mover::pump` on every `pull`: it asked every in-flight copy whether it retired, started second hops, freed slots and scanned every resident for one to bring home.
- `IdleWatch` (the lake's idle watch) detached the set of sinks (the attach lock, completing each sink) from the poll that found the lake idle.

The fifth, `GiveBacks`, stays as it is: it pops one order and builds its frame, and the handler already runs as a frame.

Now the work is a frame's. Each of these sources owns one reusable **key frame** and pushes it from `request` when work is due: `Completions` while operations wait, `AsyncIo` while operations are in flight, `Moves` when `Mover::pump_due`, `IdleWatch` once the lake has been idle for the idle time. A worker executes the frame (so its cost is measured as any frame's is), and the frame's `recycle` sends it home. `request` and `pull` hand out what is ready: they pass on the continuations the key frame found, within the demand, and ask nothing of the operations. (`pull` no longer finds anything on its own for the same reason: a poll at every pull is the high-frequency work the rule keeps out of it.)

**The rule.** `request` and `pull` run at every visit of a worker, so they do no heavy or high-frequency work. A small continuation may run there: O(1) bookkeeping, handing out a result that is ready. What moved, and why, in those terms:
- `Completions`: a query of every registered operation, with the application's code, at every visit: high-frequency.
- `AsyncIo`: a system call (`io_uring_enter`), a lock and the application's continuations at every visit: heavy and high-frequency.
- `Moves`: a device query per copy in flight, a scan of every resident and copy submissions at every pull: heavy.
- `IdleWatch`: the look (two atomic loads) stays; the detach is a lock plus completing every sink: heavy.
- `GiveBacks` stays: a pop and a frame, O(1).

**Why.** The workers can only measure how long work takes if the frame comes back through their executor; that is how they learn the shape of the workload, size their batches, and decide to idle or to take part. Heavy work, or work that recurs at the rate of the visits, done in a source's `request` or `pull` is invisible to them.

**Java correspondence.** Java's `GpuCompletions`, `AsyncReads` and `InferenceLake`'s idle watch do their work in the delegate's `hookOnPull` and `hookOnRequest`. This is a deliberate departure from them: the rule above supersedes. `Moves` and `GiveBacks` have no Java original.

**How to use it.**
- New in `euhedral-core`: `generics::KeySlot<K>`, the key's home. One atomic reference: `take` (a swap with `Acquire` alone: null means the key is away; there is no flag to read first), `home` (a swap with `AcqRel`; a second key homed by a bug displaces the one found there, which is dropped, and a debug build fails) and `home_replacement` (one compare-exchange from null, for the replacement of a key dropped unrun: a refused replacement is made inert and dropped, so a chain of replacements ends at once). A source that needs a key frame holds one `KeySlot<K>`.
- A caller that drives a source by hand (tests, or any code outside a lattice) calls `Completions::reap` or `AsyncIo::reap` before it pulls. They do what the key frame does. **Breaking:** `Completions::pull` and `AsyncIo::pull` find nothing on their own; they hand out what a reap queued. `Mover::pump` is unchanged (public, any thread).
- `Mover::pump_due()`: whether a move is in flight, or a slot freed in any pool since the last pump.

**Semantics.**
- **At most one key frame per source is out.** The source pushes it only while it is home. A `request` that finds it away does nothing more than the swap that finds the slot empty. The frame is unordered and routed by a new hash for each push, so it may run on any worker.
- **No `push` is reachable from `pull`, and a key never leaves through a pull.** A worker that has entered from a pull must not push: a `request` pushes a key, and a `pull` neither pushes anything nor hands the key to its `consumer`. A frame handed to a pull's `consumer` is run by the pulling worker through its executor while the source's handle is held, so nothing else could take ready work from that source until the frame is done, and the work behind each of the four keys is long, taken at its worst case: the `Completions` poll (every registration, and the application's `Pending::poll` code), the `AsyncIo` reap (a system call, the operation table's lock, the application's continuations), the `Moves` pump (a device query per copy, second-hop submissions, a scan of every resident) and the `IdleWatch` detach (the attach lock, completing every sink). So the four go out through `request`'s push alone, are unordered (the work can run on any core, so no order is needed) and are routed by a new hash for each push. That includes what a pull leaves behind: when a reap runs on one worker while another is in a pull, the requests banked meanwhile are served by the next `request`, or handed out by the next `pull`, and not by the pull that held the hand-out when it lets go (`Completions` and `AsyncIo` used to push from there). A key out is never out twice: the slot's take is a swap.
- **A key that leaves only through `request` needs a `request` to reach every handle.** An idle lattice can phase-lock so that a `request` never reaches half of an even number of handles (each `UpstreamQueue` pull and request services one handle, and an idle Direct cycle is one empty pull and then one request); the key of such a handle is then not pushed for as long as the load lasts (the lake's idle set never detaches, a timer is not seen). The fix is in the upstream queue, in the last change of this stack; until it lands the lake test `offers_racing_an_idle_detach_land_in_exactly_one_set` is ignored, and the `timer_latency` probe (ignored, run by hand) shows the stall under load.
- **A key that polls is paced, by what it waits for.** A key pushed at every visit while nothing ends reads as a productive handle and counts as work the worker ran, so the worker never parks (a `Completions` with one registration that never completes used a whole core). So a visit pushes the key only when there is something to look at:
  - **Something new:** a registration, a submission, a move started, or a bell that rang. The source keeps a counter of these; the key records it before it polls, and a poll that found nothing backs off only if the counter did not move meanwhile, so a registration that races a poll is never lost to the back-off.
  - **A deadline:** a `Completions::at` timer is not polled before its deadline. The poll records the earliest deadline of the timers waiting (a registration lowers it); a visit is due when the clock has reached it. One clock read per visit while a timer waits.
  - **A bell:** `Completions::after_retired` claims the `RetireCell` for the source, and the lane's callback adds one to the source's event counter after it fills the cell (`RetireCell::arrive`: one atomic exchange of the status, then one atomic add if the cell was claimed). The claim installs the bell and then does one read-modify-write on the status, and `arrive` is a read-modify-write too: the two are totally ordered on the status, so either the callback sees the bell and rings it or the claim sees the status, and the registration that follows the claim makes the key look. (A store followed by a load of the bell lets both sides read the old value, and the wake-up is lost; a loom model, `lanes::loom_tests`, pins it.). A wait with a bell is not polled at all until it rings. A cell another source claimed first is polled blind.
  - **A blind poll** (a marker, a `Ready`, a `Wake`, a `Pending`, `AsyncIo`'s ring, a copy in flight): after an empty poll the next is not due for a window of **time**, 16 us doubling to 100 us (`KeyPacing`), so the added latency is bounded by 100 us whatever the load. (A count of visits, which an earlier version used, made a wait late by up to its own duration: a visit's interval varies by two orders of magnitude.) A find ends the window.
  - `Moves`: a freed slot is a signal and is never backed off; a move's copy is polled blind.
- **A query or continuation that panics loses its entry, not the source.** The key's `execute` catches it: a `Completions` entry is finished without running (or dropped), an `AsyncIo` operation is lost; both are counted (`Completions::panicked`, `AsyncIo::panicked`) and the entries behind go on. For `AsyncIo` the panic is caught where the operation has already left the engine (`conclude`), and its frame is one that does nothing, so nothing unwinds through the engine.
- **A resident that could not be brought home is retried.** `Mover::pump` leaves a pump due (paced) when a spilled resident's home has room but the resident was busy (inside `Mover::with`) or its move could not start.
- **Pushed only when there is work.** The check is an O(1) counter (`Completions`: registrations not yet found completed; `AsyncIo`: operations submitted and not yet reaped; `Mover::pump_due`; the lake's idle test, two atomic loads). A source with nothing to look at pushes nothing, so an idle source costs a worker no frame.
- **Order and demand are unchanged.** The frames a reap finds go into one single-producer, single-consumer queue in the order found and out through `request`/`pull` with `stop` and the banked demand as before (a `Completions` holds back the first frame a pull's `stop` matches, as before). The worker that ran the key frame hands out within the demand already banked when it has found something.
- **A completed `Completions` pushes no key frame, and its `request` and `pull` cancel nothing.** What was registered behind `complete`'s marker is cancelled by a registration's own sweep, by `reap`, or by the last drop, as before; none of them is `request` or `pull`.
- **A key frame dropped without having run** (a frame the lattice discarded) is replaced by a new one, at home, so the work cannot be stranded. A frame that finds its source gone does nothing.
- **A detach frame that finds the lake busy again** leaves the set alone and goes home; the watch pushes it again at the next idle time.
- **Allocation:** none added per frame. The key and the queues' chunks are made once (the chunks are pooled).

**Tests.** A test per moved site detects work in `request`/`pull` without timing, with a receiver that keeps what the source pushes (the test runs it as a worker would) and a counter of the queries made, or a mark that the work itself reads to see whether it runs inside a frame: `euhedral-memory/tests/tiers.rs` (`a_request_or_pull_of_the_moves_source_drives_no_move_and_a_frame_does`, `a_source_with_nothing_in_flight_pushes_no_pump_frame`, `a_freed_slot_asks_for_one_pump_to_bring_a_spilled_resident_home`), `euhedral-io/tests/key_frame.rs`, `euhedral-graph/tests/lake.rs` (`an_idle_watch_detaches_in_a_frame_and_not_in_a_request`, `a_detach_frame_that_finds_the_lake_busy_again_keeps_the_set_and_goes_home`), `completions::tests` (names start `request_and_pull_query_nothing`, `the_poll_frame`), and `euhedral-memory/tests/planner.rs` (`a_give_back_order_runs_its_handler_only_when_its_frame_executes`) for the site that stayed. `KeySlot` has its own tests (`generics::key_slot::tests`), including eight threads racing for the key. A pull pushes nothing and hands no key out, a request pushes it, and a key out is never out twice, in each source: `the_poll_frame_is_pushed_by_a_request_never_by_a_pull_and_a_key_out_is_never_out_twice` (`Completions`), `a_pull_pushes_and_hands_out_no_key_a_request_pushes_it_and_a_key_out_is_never_out_twice` (`AsyncIo` in `key_frame.rs`, `Moves` in `tiers.rs`) and the first idle-watch test of `euhedral-graph/tests/lake.rs`. A reap that runs while a pull holds the hand-out leaves the pushing to a request: `a_poll_frame_that_runs_while_a_pull_holds_the_hand_out_leaves_the_pushing_to_a_request` (`Completions`) and `a_reap_that_runs_while_a_pull_holds_the_hand_out_leaves_the_pushing_to_a_request` (`AsyncIo`). A replacement of a dropped key never starts a chain: `a_stray_key_that_is_dropped_does_not_start_a_chain_of_replacements`.

## The benchmark times one push in 64, not one frame in 64 (2026-11-05)

**What.** The benchmark's latency sampling is decided by the source, per push: each source counts its pushes and stamps the 1st, 65th, 129th and so on. It was decided by the frame: one frame in 64 was made as a sampled frame and stamped every time it was pushed.

**Why.** A pooled source pushes the frames that came back last, not the ones it made first, so which frames run is up to the pools and the scheduling, and a frame's identity says nothing of how often it runs. With a few workers a pool can hold the sampled frame at the bottom of its stack while the others circulate, or a source can make fewer than 65 frames in a run, and an iteration then times nothing: its quantiles are `None`, and `a_fork_with_pooled_frames_completes_and_records_latency` failed on that (once in CI; and once in five runs with the workers limited to 2 CPUs, which timed none in 39 invocations). Counting pushes makes the rate the same whichever frames run, and makes an iteration that completes `tasks` frames time about `tasks / 64` of them.

**Semantics.** Unchanged for the reports: one frame in 64 handed out is stamped and timed from the hand-out to a worker starting it. A push that is not timed clears its frame's stamp, so a frame that never ran (a killed one) leaves nothing for its next push. Frames of both kinds of source now hold the recorders, as the sampled ones alone did.

**Tests.** `no_op::tests` (`a_pooled_source_times_one_push_in_64_whichever_frames_circulate`, `a_returning_source_times_one_push_in_64`, `a_source_without_recorders_times_nothing`): a source whose two frames both finish and then run one frame at a time times exactly pushes 0, 64, 128, ... (the pooled source timed 1 of 11 before). The smoke test of pooled frames asserts that the measured iteration timed some.

## A quantum that touched no lane concludes only in its own last frame (2026-11-05)

**What.** The conclude rule for a quantum that touched no lane (the entry below) names where it concludes: in its own last frame, or in one pooled frame.
- **In its own last frame**: the call that drops the quantum's last live count is the end of one of its frames (a stage, a host completion, an external edge's marker, a retry, or the frame that abandons its stages) and nothing else holds the quantum. The `retired` hooks, `retire` and the return of the run to the pool happen there.
- **In one pooled `Retire` frame**: the last hold goes anywhere else. That is a dropped `QuantumRef`, a later quantum's stage releasing its pin, a gate failed from outside the quantum, the arrival of an external edge, or a drop inside another quantum's `retire`. The frame is offered at that drop and `retire` runs on the thread that takes it.
- **A home lane that cannot arm a retirement** (`DeviceQueue::control()` is `None`): a quantum whose only unsubmitted hooks are of stages that name a lane touched no lane either, since a hook given such a lane has nothing to enqueue. It follows the same rule and costs the frames of its stages.
- A quantum that touched a lane keeps its boundary, and so does one whose lane-naming hooks ran on a home lane that arms retirements (a refused arming still falls back to the pooled frame).

**Java correspondence.** Java concludes every quantum through the retirement frame, which is a frame whoever dropped the last hold. The pooled frame for a foreign last drop is that structure; only the quantum's own last frame skips it, because there the frame already exists.

**Why.** The entry below concluded on whichever thread dropped the last hold. A `retire` that panicked there unwound through the dropping thread: when that was a later quantum's stage releasing its pin, the later quantum never finished (its live count was never dropped), and it was stranded. A quantum whose state holds the `QuantumRef` of an earlier quantum, in a chain, concluded one nested call per link, so the stack grew with the length of the chain. Both are gone: `retire` never runs inside the call that drops a hold.

**Contract** (module documentation, `StageGraph`, `StageGraph::new`, `Stage::retired`, `QuantumRef`, README and GLOSSARY; pinned by tests with no tolerance):
- `retire` and the `retired` hooks of a quantum that touched no lane run in the quantum's own last frame, or in the one pooled frame offered by the drop of its last hold. They do not run inside a `QuantumRef` drop, a pin release, a gate failure or an edge's arrival.
- `retire` never runs inside the call that drops a hold, except where the sink refuses the pooled frame (it is closing, or full): the frame then ends in place on the offering thread, and a panic of `retire` there is contained (the quantum has concluded; the payload is dropped). The same holds for a stage's own frame that a refusing sink hands back (an offer, a gate released into a refusing sink). Java's `conclude` catches a throwing `retire` and records it on the quantum; here the outcome is `retire`'s argument, so a panic is rethrown to the executor from a frame of the quantum, and contained where a refused frame ends in place. A frame that a `Completions` source cancels ends in place only after the source completed (shutdown).
- Retirements armed by such a quantum: 0. Frames: one per stage frame run, plus one per host completion, plus one pooled frame when the last hold goes elsewhere.
- Frames are pooled; the pooled frame allocates nothing after warm-up.

**Tests.** `stage_graph::tests` (names start `a_quantum_that_used_no_lane` and `a_killed_quantum_that_used_no_lane`, which the Miri shard selects): a quantum whose `retire` panics after a later quantum's stage released its pin concludes in a pooled frame and the later quantum retires (before: the later quantum was stranded); a chain of 100000 quanta, each holding the reference of the one before in its state, retires one pooled frame per quantum on a thread with a 256 KiB stack (before: stack overflow); a dropped last reference and a gate failed from outside each conclude in exactly one pooled frame; a killed quantum whose unsubmitted stages name a lane, on a query-only home lane, costs exactly the frame of its root (before: two). The panic scenario again with a sink that refuses the pooled frame: `retire` runs in place inside the later quantum's frame, the panic is contained, and the later quantum retires (before: stranded, `live() == 1`); a gate released into a refusing sink, for an ordered and an unordered stage, returns to the releaser with `retire` having panicked (before: the panic unwound into the releaser). The chain test asserts the order: the last quantum started retires first.

## A quantum that touched no lane concludes in its last frame (2026-11-05)

**What.** A quantum whose stages name no device lane runs to its end in the frames of its stages. The conclude rule, per quantum:
- A quantum **touches a lane** when one of its stages submitted a marker, a marker was recorded for it (an external edge brought one, or a later quantum's stage that used it submitted one), or the `Stage::abandoned` hook of a stage that names a lane (`StageSpec::lanes` not empty) ran.
- A quantum that touches a lane keeps its boundary exactly as before: the lanes it used join its home lane and one retirement is armed there (or one completion per lane on a query-only pool), and it concludes in the frame that retirement or completion emits.
- A quantum that touches none arms no retirement. It concludes (runs the `retired` hooks, calls `retire`, returns its run to the pool) in the call that drops its last live count: the frame that ends its last stage, the frame of its last host completion, or the frame that abandons its stages (a kill, a failed stage). When a `QuantumRef` or a later quantum's stage still holds it, it concludes in the call that drops the last hold, on that thread. The frames of such a quantum are its stages' and its completions' frames.

**Java correspondence.** Java arms `home.notifyRetired` for every quantum (its `usedLanes` starts with the home lane) and concludes through the retirement frame. Java's `StageFrame.host()` names a stage that submits nothing to a lane and records no marker; `StageSpec::lanes` empty is that declaration here, and a quantum of such stages alone has no device work to retire. A quantum that touches a lane follows Java as before.

**Why.** `abandon_unsubmitted` counted every stage that did not submit as a hook that ran, even the default no-op `abandoned`, so `arm_boundary` armed a retirement on the home lane (a host callback on the lane, a registration and two frames) for each killed or failed quantum of host-only stages, and every quantum of such a graph ended in a separate retirement frame. A graph of host stages needed a query-only home pool to avoid it.

**Contract** (stated on `Stage::abandoned`, `StageSpec::lanes`, `StageGraph` and in the module documentation; pinned by tests with no tolerance):
- The hook of a stage that names no lane runs at the same place with the same arguments as before (once, in stage order, with the home lane). A retirement covers the hooks of stages that name a lane.
- Retirements armed by a quantum that touched no lane: 0, for a run, a kill and an abandon, on a lane that arms retirements and on a query-only lane. Frames: one per stage frame run, plus one per host completion.
- `retire` and the `retired` hooks of such a quantum run on the thread that drops its last count or hold. (The entry above refines this: a last hold that goes outside the quantum's own frames takes one pooled frame.)

**Migration.** A stage that names no lane and enqueues work on the home lane from `abandoned` names that lane in `StageSpec::lanes`.

**Tests.** `stage_graph::tests` (names start `a_quantum_that_used_no_lane`, `a_killed_quantum_that_used_no_lane`, which the Miri shard selects): a chain of host stages concludes in its last stage's frame (3 stages, 3 frames); a host completion's frame concludes it; a kill and a failed stage conclude in the abandoning frame (1 frame, hooks in stage order, no arming attempted even with the lane set to refuse one); 30 quanta of run, kill and abandon each cost the frames of their stages and arm 0 retirements; the last dropped reference concludes it at the drop; a quantum covered by a later quantum's marker keeps its retirement; a quantum with a stage that names a lane keeps its retirement. Each on a lane that arms retirements and on a query-only lane. `tests/stage_graph_lane_less_alloc.rs`: 3000 quanta of run, kill and abandon allocate 0 after warm-up and cost exactly their stages' frames.

## A stage holds itself at its gate from inside `run` (2026-11-04)

**What.** A stage that finds, while it runs, that the state it needs is not ready takes its gate with `StageCx::hold()` and answers `Err(StageError::Held)`. Whoever makes the state ready releases the `Gate`; the stage then runs again, once, with nothing retried and nothing allocated. Until now only `Stage::pass_gate` (Java `StageFrame.passGate`), consulted before the first run, could hold a stage, so a wait found only by running could be answered by `StageError::Retry` alone, which re-places the stage and registers a wake-up for every attempt.
- `StageCx::hold(&self) -> Gate`: takes the stage's gate in `run`. Once per run (a second call panics, which fails the quantum).
- `StageError::Held` (**breaking** for an exhaustive `match` on `StageError`): the answer that goes with a gate taken in `run`. It is not a failure.

**Java correspondence.** Java has the gate only before the stage runs (`passGate` / `releaseGate`, `publishGated` / `releaseGated`) and no retry at all: a stage there cannot hold from inside `submit`. The release, the live count the gate holds, `publishOrRun` for a sink that no longer takes frames and the failure of a gate that throws or is dropped are the existing gate's, and a gate taken in `run` is the same object: `Gate::release`, `Gate::fail` and its drop. The one addition is what the end of a run does when the gate was taken inside it, which Java's gate has no counterpart for.

**Contract** (stated on `Stage::run`; each point is pinned by a test with no tolerance):
- **Who may release.** The holder of the `Gate`, once, from any thread that may publish frames, the thread running the stage included.
- **Exactly once after a release.** Each release publishes the stage once and `run` is called once more, on a new frame, with a new `StageCx`; `pass_gate` is not asked again; the second run may hold again. The answer `Held` submits nothing, satisfies no edge and keeps the stage's live count, so the quantum cannot retire while a gate is out.
- **A release before `run` returns** (the releaser races the end of the run, or is the running thread itself) does not publish the stage beside its running `run`: the stage is published when `run` returns, and the next run starts after it ended and sees everything the releaser wrote before releasing. Of the end of the run and the release, the second to arrive acts: one exchange on the stage's gate state (`GATE_RUNNING` to `GATE_HELD` by the run's end, to `GATE_RELEASED` or `GATE_FAILED` by the spender), release on the way in and acquire on the way out, with no loop and no lock.
- **Kill and stop.** A killed or failed quantum stays live until the gate is spent. A release then publishes the stage, which ends without running (the frame carries the kill switch); the stage is abandoned exactly once, at the quantum's boundary, and its `retired` hook runs with `committed` false. A holder that learns of the kill may release, fail or drop the gate: all three are harmless, and with the kill switch set each ends the quantum as `Killed`, unless the quantum had failed already (the first cause stands; `fail_gate` notes the kill before the failure, so a `fail` or a drop after a kill no longer reports `Failed`; this holds for a gate taken in `pass_gate` too).
- **`Gate::fail` and a dropped gate** fail the quantum with the given error (`Fatal` for a drop) and drop the stage's count once, whether before `run` returns (then at its return) or after; the stage does not run again. `Retry` and `Held` are no failures, so `Gate::fail` given either fails the quantum with `Fatal`.
- **A stage that breaks the contract** (`Held` with no gate taken; a gate taken and `Ok` or `Retry` answered, or a panic in `run` after it; a second `hold` in a run) fails the quantum with `Fatal` (a stage's own error stands as the cause), is not run again, and the gate it took stays valid: it keeps the count until its holder spends it, which is harmless. A stage that answered `Ok` after taking a gate did submit: its marker joins the quantum's boundary and its host completion keeps the quantum live until it is ready, as for any stage whose quantum stopped meanwhile.
- **Retry untouched.** `Held` registers nothing with the completions, and `RetryCounts` do not move.

**Why.** A dependent finds, in `run`, that a result it needs is not stored yet; that wait is host-side readiness, not device progress, so it should be a gate (no polling, nothing registered) and not a retry, which the marker ring's full-ring case alone should answer.

**Migration.** A `match` on `StageError` without a wildcard adds `Held`. `StageCx` gained private fields (nothing outside the crate builds one).

**Tests.** `stage_graph::tests` (the names start `a_gate_taken_in_run`, which the Miri shard of the gate tests selects): a held stage is released from another thread, runs exactly once more and answers no retry (`RetryCounts` zero, nothing registered with the completions); a release before `run` returns, on the running thread and on a thread it joins (the sink holds nothing until `run` returns); a hold taken again by the run a release gave; fail and drop, before the return and after; a kill while held; the stage that kills its own quantum and holds; a drop after a kill; every way to break the contract; a panic after the gate was released; a sink that refuses the released stage (an unordered stage runs where it was found, an ordered one fails the quantum); four holders released from four threads at once; a release racing the end of the run on a thread of its own, 300 rounds. Loom models of the exchange (`loom_the_second_of_a_runs_end_and_its_gates_release_publishes_the_stage_and_sees_both_writes`, `..._failure_drops_the_count_once_and_sees_the_cause` (the failer writes the cause before it spends the gate), `loom_a_gate_stored_in_run_and_handed_to_another_thread_is_spent_by_exactly_one_party` (the Relaxed store, then a hand-over), `loom_a_gate_spent_twice_while_its_run_ends_acts_once`). A gate is spent with one `swap`. `tests/stage_graph_hold_alloc.rs` counts every allocation of the process over 500 quanta that each hold a stage once, released by the driver, by the stage before it returns, and by a thread of its own: 0 allocations, two runs per quantum, `RetryCounts` zero.

## A retried stage allocates nothing when it runs again (2026-11-04)

**What.** A stage that answers `StageError::Retry` no longer allocates when it runs again. Its wake-up (the lanes' completed counts at the retry and the backstop's deadline) used to be made anew by every retry: a `Vec` and a `Box`. It is now one record per stage of a run, made with the run, prepared again by each retry, and registered with `Completions` as a clone of its `Arc` (the crate-internal `Completions::after_wake`, as `after_retired` registers a clone of a `RetireCell`). The pooled run grows to its peak in warm-up, as before, and nothing after it.

**Semantics unchanged.** A retried stage runs again when any lane of the pool completed more work than at the retry, or after `GraphRuntime::retry_backstop`. `RetryCounts` count the same: `retries` once per `Retry` answered, and each wake-up once, in `progress` or in `backstops` (progress first, when both hold). A kill or a stop while the stage waits ends it as before.

**Why.** Retry is what a stage answers when the marker ring is full; a dependent saw about a thousand retries a minute on one ordered lane, each allocating twice.

**Tests.** `tests/stage_graph_retry_alloc.rs` counts every allocation of the process over 500 quanta of 3 retries each, once with every wake-up by a lane's progress and once with every wake-up by the backstop, and asserts 0 allocations and exact `RetryCounts` (it failed first with 2 allocations per retry). `completions::tests::a_kept_wake_is_emitted_when_it_holds_and_can_be_registered_again`.

## Stage gates, abandoned behind the join, and the lane's raw stream (2026-11-04)

**What.** Three more of Java's `StageFrame`/`GpuStream` behaviours, each drift from Java fixed.
- **Stage gates** (`StageFrame.passGate` / `releaseGate`, `StageGraph.publishGated` / `releaseGated`). `Stage::pass_gate(&self, &Q, GateRef) -> bool` (default `true`) is called when a stage's last incoming edge arrives (a root's too), before it is published. `false` holds the stage, with its live count kept; the stage takes a `Gate` with `GateRef::hold(self)` (which consumes the `GateRef`, so a stage cannot take two gates: the type enforces "released at most once"), and something later calls `Gate::release()` once, from any thread that may publish frames (a frame's `do_finally`, a `Completions` continuation). That publishes the stage; a lake that no longer takes frames has an unordered stage run right there (Java `publishOrRun`), so the held count is always dropped. An *ordered* stage cannot run beside its sink (its frames must run in published order, one at a time on their lane, which only the sink enforces), so a refused ordered stage ends in place and fails the quantum. A held stage keeps the quantum live. `Gate::fail(error)` fails the quantum and drops the count; a panicking `pass_gate` fails it (and drops the count unless a gate was taken, whose holder owns it); a stage that answers `false` without taking a gate, or drops the gate it took, fails the quantum instead of hanging it. This is how a stage waits for host-side readiness without polling: the work that makes it ready releases the gate, so no stream ever waits for the host and nothing retries.
- **`abandoned` runs where Java runs it** (`abandonUnsubmitted(home)` at `quiesce`, before `home.mark(joined)` and `sweepExternals()`). `Stage::abandoned(&self, &Q, home: &L)` runs at the quantum's quiesce, after its lanes joined the home lane (with lanes that have control) and *before* later quanta waiting on its stages are released, so a later stage that gates on what a hook releases finds it released. The home lane is passed in, and the retirement is armed on it after the hooks (also for a quantum that used no lane but ran a hook), so what a hook submits to home is covered before `retired` and `retire` drop anything. The boundary joins again only the markers later quanta added while they pinned the quantum. It used to run before the device boundary, with no lane, and once the later quanta's pins were gone. Each stage that never submitted gets exactly one call, in stage order; a panicking hook fails the quantum and the others still run. A quantum that used no lane and ran no hook arms no retirement (nothing on a lane for it to cover; Java always arms one).
- **Library calls on a shared lane's stream** (Java `GpuStream.submit(Runnable)`: the work runs with the stream selected). `LaneStream::raw_stream()` on the enqueue handle (`Enqueue` extends `LaneStream`, so it is on the handle of `Lane::submit`, `submit_marked`, `submit_held`, an owned stream's `submit`, and in a recurring sequence's body) lends the driver stream as a `RawStream<'_, R>` (CUDA `CuStream`; the fake device's `FakeRawStream`): it borrows the handle and is `!Send`, so the borrow checker keeps it inside the closure and on the calling thread (compile-fail tests), for library calls that enqueue on a stream (any library that enqueues on a stream). **On a shared CUDA lane it is a private stream forked from the lane, never the lane's own**: a library may call `cuGraphLaunch` internally, and a graph launch on a stream several threads submit to can overtake the wait its own thread issued before it (the #52 driver race); the fork is made on the first request (an event recorded on the lane and awaited by the private stream, so it is ordered after everything enqueued so far), the closure's later operations go to it too, and when the closure ends the lane waits for it, and the marker is recorded after that. In a pass of a recurring sequence (record, update, probe) there is no stream to lend: `raw_stream` is `None` and the pass notes the request (`Sequence::library_stream_requested`), so a body that does library work is never replayed with the call silently skipped: `SharedGraphCache` probes the body (no device) before it records and checks the update pass of a hit, marks the key unrecordable, and runs the unit directly, every time, with the call made (a recording pass that still sees a request fails with an error rather than skipping the call). What the library does with the stream is the caller's contract (documented on `LaneStream`): enqueue only work that completes on its own, do not synchronize, destroy or capture the stream, keep what the work reads valid until it retires. The marker claim still precedes the closure, so a refused marker means no library call, and the marker is recorded behind the library work.

**Migration.** `Stage::abandoned` gains the `home: &L` parameter (`&dyn DeviceQueue` for the default lane type); implementors add it. `Stage::pass_gate` is new with a default. `Enqueue` extends `LaneStream` and `Lane::Enqueue` / `Lane::Held` implement it (an implementor outside this repository adds it).

**`StageError::Retry` after gates.** Java has no retry. Here it is still used by one thing: a stage whose `Lane::submit_marked` answers `Busy` because the marker ring is full or an awaiter holds a slot. A gate cannot replace that: what the stage waits for is device progress, not host readiness. Recommendation: keep `Retry` for now; remove it when a lane's markers cannot be refused (pooled or unbounded events, as Java's `openMarker` allows), and move the host-readiness waits that polled before to gates.

**Tests.** `stage_graph::tests`: a held stage runs when its gate is released from another thread and not before (the quantum stays live), four gates released from four threads at once publish each stage and the join once, a failing gate fails the quantum and abandons the stage and what is behind it, a panicking gate, a `pass_gate` that panics after taking a gate fails the quantum, a stage held with no gate to release (or a dropped gate) fails the quantum, a gate released after a kill ends the stage without running it, a stage released into a refusing sink runs in place (an ordered one fails), the `GateRef` compile-fail doctest (a second `hold` does not compile); a stage is abandoned after the lanes joined home and before the retirement is armed, a refused retirement for the hooks of a quantum with no lane still retires it, a killed quantum that used no lane arms a retirement after its hooks (and one with no hook arms none), a later quantum on an abandoned stage runs after the hook, every stage that never submitted is abandoned once in stage order; loom model `loom_a_held_stage_is_published_once_and_the_releaser_sees_what_the_gating_thread_wrote` (weakening the held flag's store to relaxed fails it); `tests/fake_shared_lane.rs` and `tests/fake_lane_graphs.rs` (the raw stream is lent inside `submit_marked` and `submit_held`, and not behind a refused marker; library work runs for every unit through the cache; a recorded key whose body starts to do library work runs directly from then on), `RawStream` compile-fail doctests (it cannot be kept past the closure or sent to a thread); `tests/cuda_shared_lane.rs` (the lent stream is not the lane's own, the closure's work before and after the request and the library work are ordered, and a later submission follows it; a driver memset on the raw stream inside `submit_marked` is ordered before the closure's following copy and covered by the marker).

## Recorded graphs run through a shared lane (2026-11-04)

**What.** A stage on a shared lane can record, update and replay a CUDA graph (or the fake device's recording) through `&self` of the lane, and a cache of recorded graphs serves any number of workers at once without a lock.
- `euhedral-gpu`: `Gpu::record_on` and `Gpu::rerecord_on` (record over shared lanes), `Graph::replay_on` (replay on a lane), `Lane::submit_held` (a body written against `Enqueue` on a lane); `GraphHold`, the values a run adds for its caller to hold; `SharedGraphCache` (`run`, `warm`, `retain`, `stats`, `contains`), the cache of instances by key; `Enqueue::keep_with`.
- `Enqueue` now extends `LaneEnqueue`: the operations (`launch`, the copies, `memset`) are `LaneEnqueue`'s, and `Enqueue` adds `keep` and `keep_with`. One kernel body written against `LaneEnqueue` serves a shared lane's `submit` and `submit_marked`, an owned stream's `submit`, and every pass of a recurring sequence (record, update, direct, replay), so no body is written twice.

**Why.** `Graph::replay(&mut self, &mut Stream)`, `Gpu::record` and `GraphCache::run` took the owned stream, so a stage on a shared lane could launch and copy but not record or replay. Replay also handed the graph's kept values to the stream, and a shared lane keeps none.

**Design.**
- **Instances by key.** A recorded graph is used by one thread at a time (its executable takes new arguments and launches), so a key that several workers run at once has several *instances* of its graph, in a fixed table of slots (`InstancePool`). A worker takes a free instance of its key by one compare-and-set on the slot's word; a worker that finds none does not wait or spin: it reserves a slot, records an instance of its own (the recording runs the work for real), and pools it for the next. The instances of a key settle at the peak number of workers that ran it at once; after warm-up nothing allocates. The slot's word is `(tag << 2) | state` (empty, free, busy; the tag is a hash of the key, confirmed against the key once taken), and a slot's contents move in and out of it under a release store and an acquire compare-and-set.
- **Capacity counts instances**, not keys. A full table evicts the free instance used least recently, approximately (each slot keeps the tick of its last take, read from a coarse clock that advances only when an instance is filled, so a take is a read and a store and nothing shared is read-modify-written per run; Java's `MAX_CAPTURES` map is least recently used too, and a wrong pick only records an instance again). With every instance in use or being recorded, the unit runs directly (`submit_held`): nothing is recorded to be thrown away. Dropping an instance is safe whatever its launches are doing (the driver frees an executable once they end). A key still being recorded does not count as present.
- **A replay never launches on the lane (a driver race).** With several threads submitting to one stream, a `cuGraphLaunch` can overtake the `cuStreamWaitEvent` its own thread issued just before it: seen with driver 615.71.09 on CUDA 13.4, where a graph's work ran before what it waited for. A stress test of 256 quanta of two replaying stages on two lanes with at most 16 or 32 in flight lost the ordering in a large share of runs, and a plain C driver-API reproduction without RustEE did too (the second stage reading the first's word before it was written); plain launches, one submitter per stream, and a wait and launch locked together did not, and neither did reusing an executable whose launch is pending (a single thread updating and relaunching it). So each worker replays on a private stream forked from the lane: an event recorded on the lane and awaited by the private stream, the cross-lane marker waits and the `cuGraphLaunch` on the private stream, then an event recorded on the private stream and awaited by the lane; the stage's marker is recorded on the lane after that join. The private streams are stocked per lane in a lock-free table of handles (taken by a swap to empty, given back by a store into an empty slot), made once and reused. The table counts the handles that exist and those stocked or landing, so a take misses only when none is stocked (it retries across a peer's few-instruction step instead of missing) and a give never fails while the handles fit the slots (it loops until it lands); a stream beyond the slots goes to an overflow list and is destroyed with the lane, never synchronized on the thread that gave it back. So nothing is created or destroyed after warm-up, however the workers contend; the same streams are the shadow streams a recording captures on (one per lane, not one per recording). The marker waits are therefore an argument of the run (`SharedGraphCache::run_after(.., waits, ..)`, a stage's `cx.waits`), not something the stage enqueues on the lane before it.
- **The instance goes back once its replay is enqueued.** A launch's parameters are fixed when it is issued (an update applies to the next launch) and the driver orders the launches of one executable behind each other, whichever stream they are on; the overlapping-quanta stress test, which returns instances at once, passes (150 runs each at 16 and 32 in flight). What the replay reads that the instance does not own are the recording's host buffers, which stay in the caller's `GraphHold` until `retired`. So the table does not grow with the replays in flight: the instances of a key settle at the peak number of workers running it at once.
- **Java parity.** The first sighting of a key runs directly and the second records (`Capture.sightings`; `warm` records at once). A capture that fails marks the key unrecordable, and it runs directly from then on (`Capture.unrecordable`): if the body had run (for real, unaffected by the failed capture) the unit is done, and if the capture failed before the body ran the unit runs directly at once, so a unit is never counted as run when it did not run. The sightings and the unrecordable marks are separate fixed tables of atomic words, so a crowd of ordinary sightings never forgets a mark.
- **Kept values go to the caller.** A shared lane keeps no values. A recording, a replay and a direct run add what their work reads to the caller's `GraphHold`: a recording or a replay a handle to the graph's frozen values (an `Arc` clone: no allocation), a direct run the values its body kept. The stage keeps the hold in its quantum and drops it in `Stage::retired`, after the quantum's device work ended; the instance, the cache and the lane may be gone by then without the work losing its buffers. Holding the values of one run allocates nothing; a second run into the same hold, or a direct run that keeps values, allocates a list (give each run its own hold to avoid it).
- **Markers.** Run the cache inside `Lane::submit_marked`: the marker is claimed before the run, so any `Busy` comes before any recording or replay (the closure's enqueue argument is not needed; the run reaches the lane through the `lane` argument).
- **Recording on a lane.** The recording runs the sequence for real on the lane and captures the same work on a shadow stream, one per lane taken from the lane's stock of private streams (the owned stream keeps one for its life). Other lanes of the sequence fork from the first at the start and join it at the end, as before.
- **The owned API is unchanged.** `GraphCache` (`&mut self`, one owner, no atomics) and `Graph::replay(&mut Stream)` keep working for `LaneSource` users; the shared cache is a separate type because its state really is shared.
- **`keep_with`.** A body runs in every pass but only a recording (and a direct run) keeps its values: `Enqueue::keep_with(|| value)` makes the value only when it is kept, so a replay allocates nothing.

**Java correspondence.** Java captures at the stage graph level (`StageGraph` Capture/Replay: each stage submits as usual and again to the lane's shadow, `endCapture`, `launchGraph` on the home stream); RustEE's cache is the stage-level form, keyed by the caller. The shadow-and-real submission, the end of the capture on the pass's own shadow, and the launch on the lane are the same.

**Migration.** `Enqueue` extends `LaneEnqueue`: code that called `launch`, `copy_*` or `memset` on a concrete enqueue type (a `CudaEnqueue`, a `FakeEnqueue`) with only `Enqueue` imported also imports `LaneEnqueue` (generic code bounded by `Enqueue` is unchanged). `Gpu` gains `record_on` and `rerecord_on`, `Graph` gains `replay_on(lane, waits, hold)` and `Lane` gains `Held` and `submit_held` (an implementor outside this repository adds them).

**Tests.** `instances.rs` (unit tests: least recently used eviction, a key still being recorded is not present, a panicking `retain` leaves its slot free; an 8-thread test that no instance is ever used by two threads and no thread waits; loom models `loom_one_instance_is_taken_by_one_thread`, `loom_a_given_back_instance_hands_over_its_contents` and `loom_a_thread_that_finds_the_instance_in_use_records_another`); `handles.rs` (the stock of private streams: unit tests, an overflow list, an 8-thread test, a 7-thread test of 3 million take/give rounds over 8 handles in 8 slots that a take never misses and a give is never refused, loom model `loom_one_handle_is_used_by_one_thread_and_its_contents_are_ordered`); `hold.rs`; `graph_cache.rs` (`shared_tests`: first sighting direct, second records, then replays; shapes, warm-up, retain; capacity counts instances and evicts the least recently used; a failed capture runs the work and marks the key unrecordable, one that failed before the body ran still runs the unit once, a crowd of sightings does not forget an unrecordable key; caching off holds what the body keeps; what a recording keeps is held until the callers' holds drop; a replayed instance goes back once its replay is enqueued; four workers inside the passes at once each get an instance; a busy marker ring refuses the run before it records; a replay orders itself behind its waits before its work); `tests/fake_lane_graphs.rs` (one body for every way to submit and every pass; a stage replays a cached graph in a quantum and the successor awaits its marker; nothing released while the work is queued, the hooks release after); `tests/cuda_lane_graphs.rs` (256 quanta of two replaying stages on two lanes with 16 and with 32 in flight, the second ordered behind the first, most stage runs real replays: each fails in a large share of runs when a replay launches on the lane, and passes 150 of 150 with the private streams; six workers inside the passes at once each get an instance and every replay runs, the holds kept until the work ended; six workers replaying under contention make no private stream after warm-up); `tests/cuda_lane_graph_alloc.rs` (2200 quanta of two stages that replay, mark, hold and release, one at a time and eight in flight: 0 allocations after warm-up).

## A stage submits to its lane directly, as in Java (2026-11-03)

**What.** A stage of a `StageGraph` can submit device work to the lane it runs on, from whichever worker runs it, and return the marker of that work synchronously; the graph retires a quantum the way Java does (`quiesce`), and a stage releases what it held in a retirement hook.
- `euhedral-graph`: `LanePool`, `GraphRuntime`, `StageCx`, `Stage` and `StageGraph` take a lane type `L` (default `dyn DeviceQueue`, the query side, so existing code is unchanged). `StageCx::lane_handle()` is the lane the stage was given. `LanePool::typed`, `LanePool::as_queues` (the query view for `Completions`), `LanePool::next_home`, `LanePool::with_reserved_lanes`, `Placement`. `LaneControl` (`await_marker`, `notify_retired`, `confirm_retired`), reached through `DeviceQueue::control()` (default `None`); `RetireCell`; `ControlError`; `Completions::after_retired`; `Stage::retired(quantum, committed)`.
- `euhedral-gpu`: the `Lane` trait, the shared submit side of a lane (`CudaQueue`, `FakeQueue`): `submit`, `submit_marked` and `mark` through `&self`; it is a `LaneControl`. `LaneEnqueue` is what a shared submission enqueues. `Stream::shared()` is the handle; `open_shared_lanes` opens a device's lanes as a typed pool (`SharedLanes`).
- `euhedral-data-structures`: `atomics::thread_random` is public (`next_u64`, the per-thread generator).

**Why.** Java's `GpuStream.submit` runs on the calling thread, from any worker, and the driver serializes the launches on one stream; RustEE's streams took `&mut self`, owned by a lane source, so a stage could not launch on its lane and return `Submitted::marker`. The user chose Java's model: a shared stream that any worker submits to, not routing by lane hash.

**Java correspondence.**
- `GpuStream.submit(Runnable, overlap)` on the calling thread: `Lane::submit(&self, f)`.
- A stage's marker, opened before the stage submits (`openMarker`): `Lane::submit_marked`, which claims the marker number first, runs the work, then records the marker. Any `Busy` is the claim's and comes before any work, so a retried stage has submitted nothing. RustEE keeps a ring of events and numbers instead of Java's reusable event per marker, so that `DeviceQueue::completed` stays a monotonic count that `Completions` polls. `Lane::mark` (a marker with no work of its own) is the same claim and record.
- `GpuStream.await(marker)`: `LaneControl::await_marker`, with the pin handshake that keeps a slot from being re-recorded under an awaiter.
- `GpuStream.notifyRetired(listener)` / `confirmRetired`: `LaneControl::notify_retired(cell)` / `confirm_retired`. Each arm names its sink, a `RetireCell`; the driver callback stores the status there and does nothing else; a frame settles it, never a source's `request` or `pull`.
- `StageGraph.quiesce`: when the lanes have control, every other lane a quantum used joins the home lane on the device (the home lane awaits that lane's latest marker of the quantum), one retirement is armed on the home lane with the quantum's own cell, and the retirement frame settles it, runs each attempted stage's `retired` hook in stage order (`StageFrame.retired(committed)`) and the quantum's terminal work, then recycles. A device failure of that retirement ends the quantum with the new `QuantumOutcome::DeviceFailed { cause }`, which names no stage (Java's `retire(ticket)` calls `retiring.fail(deviceFailure)` without one).
- `RetiredEdge`: a completion edge arms a retirement on the lane of the stage's marker with the stage's own cell, and its successors arrive when it does; a device failure fails the quantum and releases no successor.
- `LanePool.choose` (`fixed` placement) and `nextHome` (compute lanes only; `with_reserved_lanes` is the reserved transfer lane): see Drift.

**Semantics.**
- The marker ring is lock-free for any number of submitters (`MarkerRing`): a claim checks the slot's previous marker completed and its pin, then takes the number by compare-and-set (acquire-release; only the pin handshake with an awaiter is sequentially consistent); `recorded[slot]` is stored after the record call; `done` is advanced by scans that stop at a number whose record is not stored. A claim dropped unrecorded (the work failed, or the driver refused the record) is a skipped number, a hole that completes without an event. `issued()` is the highest number claimed. The owned `CudaStream` uses the same ring; an owner and a shared handle can both mark.
- A shared lane has no retirement channel, no claim and no kept values. A retirement cell is refcounted (the callback holds a clone), so it outlives a lane that is dropped first. What a stage's work reads is held by the stage (in its quantum) and released in `Stage::retired`, so no other party's retirement can release it early, and nothing piles up: the hooks run once per attempted stage per quantum.
- The driver stream belongs to `CudaQueue` and is destroyed, after its work, with the last handle. `CudaStream`'s drop still waits for its work and callbacks.
- Over a query-only lane (`DeviceQueue` without `control`) the boundary stays one `Completions` marker edge per lane and a completion edge waits on its stage's marker; the `retired` hooks run there too.
- A stage returns the marker of the work it submitted; the join awaits the quantum's latest marker of each lane.

**Drift from Java fixed.**
- `LanePool::choose` took the allowed lanes in turn from one counter shared by every worker. Java picks a random compute lane from `ThreadLocalRandom`, so that independent frames spread over many device orderings and the workers that submit them rarely share one, and nothing contends. The default is now random (`Placement::Random`, a per-thread generator); `Placement::Fixed(f)` is Java's `fixed` placement (a stage maps to its lane by a function, `allowed[f(stage) % allowed.len()]`, before the continued-lane rule, as Java does: deterministic under any number of workers, which tests use).
- A root took a lane in turn. In Java a root continues its graph's home lane (`continuedLane`: no submitted predecessors), taken in turn per graph (`nextHome`, over the compute lanes), so each graph's critical chain keeps one lane in every quantum. A run now has a home lane taken in turn over the pool's non-reserved lanes when the run is made, and a root continues it when it is allowed there.
- The boundary joined nothing and armed one completion per lane; it is now `quiesce` when the lanes allow it.
- Kept as RustEE extensions: a stage's allowed lanes (Java: compute lanes plus an optional reserved transfer lane, and no continuing of a transfer chain; RustEE: any set of lanes per stage, and a continued lane must be among them).

**Migration.** `QuantumOutcome` gains the variant `DeviceFailed { cause: String }`: a `match` on it must handle it (typically as `Failed`, with no stage to name; the stage-less failure of a quantum's device work used to be unreportable). Otherwise nothing breaks for existing users: the new type parameters default to `dyn DeviceQueue`, `LanePool::new` is unchanged, `Stream` gains the associated type `Lane` and the method `shared` (an implementor outside this repository must add them), and `LaneSource` consumes the owned stream as before. `Placement::Random` is the new default for stages that continue no lane: code that relied on the old turns passes `.with_placement(Placement::Fixed(..))`, which now overrides the continued lane too (a test that wants the continuation rule keeps the default placement and restricts the allowed lanes of the stages it wants apart). A `Stage` may implement `retired` (default: nothing). A stage that wants the allocation-free path keeps an `Args` it rewrites in place (a thread-local or its quantum's).

**Tests.** `euhedral-gpu/src/ring.rs` (unit tests, among them a pinned slot refused before a number is claimed, a claim recorded after the claims behind it and nothing vouched for early, a dropped claim as a hole, `Busy` only from the claim; a stress test with 8 threads; loom models `loom_two_markers_get_distinct_numbers_and_both_complete` and `loom_a_scan_never_vouches_for_a_marker_still_being_recorded`); `euhedral-graph/src/lanes.rs` (loom models of the retirement cell: `loom_a_reader_that_sees_the_status_sees_what_was_written_before_it`, `loom_a_rearmed_cell_shows_only_the_arrival_that_follows_the_arming`; `Fixed` placement determinism under 4 threads; reserved lanes are never a home); `stage_graph::tests` (the join and one retirement on the home lane, a device failure on the retirement (`DeviceFailed`), `Fixed` overriding a continued lane, hooks for each attempted stage and no other, a refused arming falling back to per-lane markers, a completion edge waiting for a retirement and failing the quantum on a device failure); `tests/fake_shared_lane.rs` (a full ring refuses a submission before its work, a failed submission gives its slot back, parties arming on one lane each read their own cell, held values released by the hook after the work and never by another party's retirement, 50 quanta release 50 holds, a retried stage does not run its work twice, a device fault fails the quantum and the hooks still run, the quantum does not retire before every lane's work ran); `tests/cuda_shared_lane.rs` (16 workers on one lane: 3200 launches and markers, distinct, contiguous, rising per worker; 16 parties arming 800 retirements on one lane; a retirement cell settled in a frame; a two-lane graph run by 4 workers over 64 quanta with holds released by the hooks); `tests/cuda_stage_alloc.rs` (2200 quanta of two stages that launch, wait, mark, hold a value and release it in their hook: 0 allocations after warm-up, every hold released once).

## A stage graph's quantum takes no lock of its own and allocates nothing after warm-up (2026-11-03)

**What.** `StageGraph`'s per-quantum path no longer takes a lock of its own (the `HOT-FIX` ids G10 to G15 are gone) and no longer allocates once
warmed up. The pools of runs and frames are bounded multi-consumer queues, whose consumer side is a try-claim; a take retries it instead of waiting. A quantum is a pooled run, taken by `start` and given back by `retire`; the graph's frames are one pooled type. `start` takes
its external edges as any `IntoIterator<Item = ExternalEdge>` (a `Vec` still works; an array or an `Option` needs none). `QuantumRef` is
unchanged to use.

**Why.** A quantum cost about 12 lock acquisitions and about 14 allocations, and quanta are about to be small and frequent. Euhedral-Inference's
`StageGraph` keeps a quantum's state without locks (volatile and atomic counts, single-writer fields, reused graph and frame objects); this
brings the port to it. The Java graph is rebound to each quantum; here the rebound object is a pooled run, as frames are recycled through
a `FrameManager`-style pool (a bounded multi-consumer queue: `FrameManager::get` takes `&mut self`).

**What replaced each lock** (Java counterpart in brackets):
- G10 `quantum: RwLock<Option<Q>>` becomes a plain cell set by `start` before any frame is published, read while `holds > 0`, taken by the
  retirement (`StageGraph.quantum`, a plain field).
- G11 `cause: Mutex` becomes one compare-and-swap that claims the first cause, whose winner writes it (`AbstractQuantum.failure.compareAndSet`).
- G12 per-stage `inputs: Mutex<Vec<Marker>>` goes: a stage keeps its output markers (`StageFrame.marker`, `lane`), and a stage merges its inputs
  from its predecessors' outputs when it runs, after its final arrival (the arrival count orders the reads).
- G13 `externals: Mutex` becomes a closable stack per stage, sealed `Submitted` when the stage submitted and `Swept` when its quantum quiesced
  without it (`StageFrame.externals`, `SUBMITTED`, `SWEPT`). New: `take_all::TakeAllStack`, intrusive over the later quantum's pooled edge
  records, so a push allocates nothing.
- G14 `boundary: Mutex<BoundarySet>` becomes the used-lane mask and the latest marker value per lane (`StageGraph.usedLanes`); the boundary is
  armed as one `Completions` edge per lane, and the last one retires the quantum. A host completion holds a live count until
  it is ready (`deferral`), so the boundary needs no host list.
- G15 `pins: Mutex<Vec<(StageId, QuantumRef)>>` becomes the pin in each external-edge record, taken by the stage that used it or by the quiesce
  (`externalIn`, `firstSlot`: fixed slots, each holding a live count).

**Semantics.**
- Quanta, frames, markers and edge records are pooled. A full pool drops what comes back. A take polls the queue again for as long as it holds
  anything (up to 65536 polls), so a lost race for the consumer claim, or a head that is claimed and not yet published, costs a retry and not a new
  object; a new one is made when the pool is empty. The pool therefore grows to the number of objects in flight at once and no further (with 8
  workers a single poll made a new frame for about one take in eight). A stage that returns a host completion allocates it; a retried stage,
  and a completion that waits for both a marker and a host completion, allocate their wake-ups.
- The boundary is armed as one edge per used lane, not one edge over all of them: a retirement frame no longer travels through the stage
  sink when the quantum used a lane (the last lane's arrival retires it in place); a quantum that used none still offers its retirement frame to
  the sink.
- A host completion holds the quantum live (as in Java's deferral), whether or not a completion edge consumes it, and also when the quantum
  halted (killed or failed) or the stage failed on a foreign marker. The quantum quiesces, and abandons its stages that did not submit, after it.
  Before, a host completion only delayed the retirement, and one that fed a completion edge was dropped when the quantum had halted.
- **`Stage::abandoned` is for every stage that did not submit** (Java `!submitted`): also one that ran and failed or panicked, and not only
  those that never ran. Before, a stage that ran and failed was never abandoned.
- **Completions that wait on a marker are `Completions::after_marker` registrations**: the quantum's boundary (one per used lane), a stage's
  completion edge whose work has a marker and no host completion, and an external completion edge (one per marker). A `Completions` that is
  dropped while the lane has not reached such a marker leaks the registration, as it does for any marker registration, and the quantum it
  belongs to never retires: its `retire` does not run and `live()` does not return to zero. That is reachable at shutdown, or when the
  completions source is dropped while quanta are in flight, and is safer than concluding the quantum while the device may still use what it
  holds. (A completion that waits for a host completion, or for both, is a `Completions::after` registration as before, and is finished
  when the source is dropped.)
- A stage whose marker names a queue that is no lane of the pool fails the quantum (`StageError::Fatal`); the boundary could never reach it.
  A later quantum of another graph on a pool with other lanes, which pins an earlier quantum, adds its markers to the earlier quantum's
  boundary only if they are lanes of the earlier pool.
- **Drift from Java fixed.** The quantum's boundary takes over the markers of all its external edges, whatever became of their stages (Java
  `awaitExternals` joins every slot to the home lane): a quantum released onto one whose stage was killed, failed or ran on the CPU still
  follows the quanta before it. Before, a later quantum named through a killed quantum lost the order of the one before.

**Tests.** `stage_graph::tests`: `the_first_failure_wins_among_8_failing_stages`, `a_kill_after_a_failure_keeps_the_failure_and_a_failure_after_a_kill_keeps_the_kill`,
`a_diamond_on_8_workers_runs_the_join_with_every_input`, `subscriptions_before_during_and_after_publication_each_arrive_once`,
`external_pins_are_released_once_whether_the_stage_runs_or_is_swept`, `a_cancelled_quantum_between_two_others_passes_the_earlier_ones_order_on`
(port of `CrossGraphEdgesTest.aCancelledGraphBetweenTwoOthersPassesTheEarlierOnesOrderOn`), `a_failed_quantum_still_waits_for_a_host_completion_and_publishes_nothing_past_it`
(`StageGraphHostStageTest.aFailedQuantumStillWaitsForADeferredStageAndNeverPublishesPastIt`),
`repeated_quanta_reuse_the_pooled_run_and_frames_and_reset_readiness` (`StageGraphTest.repeatedQuantaReuseTheSameGraphFramesAndResetReadiness`),
`a_rebound_run_starts_with_clean_stacks_and_markers`, `quanta_started_on_four_threads_retire_once_each_while_the_pool_serves_them`
(`StageGraphReuseTest.concurrentFinishersRecycleGraphsForLaterQuanta`), `a_quantum_may_start_the_next_from_its_retire_callback`;
`a_failed_middle_stage_passes_the_earlier_order_on`, `a_cpu_middle_stage_passes_the_earlier_order_on`, `a_halted_quantum_keeps_a_host_completion_that_feeds_a_completion_edge`,
`a_stage_that_fails_on_a_foreign_marker_keeps_its_host_completion`, `a_later_quantum_on_another_lane_pool_is_left_out_of_the_boundary`,
`a_stage_that_ran_and_failed_is_abandoned_with_the_stages_behind_it`;
`take_all` (pushes racing a seal, and the loom model `loom_two_pushes_and_a_seal_account_for_every_node_once`);
`euhedral-graph/tests/stage_graph_alloc.rs` (3000 quanta of 4 stages over two lanes, each with an external edge, allocate 0 times after warm-up);
`euhedral-graph/tests/stage_graph_alloc_threads.rs` (12800 quanta run by 8 workers: at most the frames in flight at once are made after warm-up; 0 to 15 allocations observed, against about 78000 when a take polled the queue once);
`euhedral-core/tests/hot_fix_ratchet.rs` (the six ids are removed from `HOT_FIX_TAGS`, and `the_stage_graph_path_names_no_lock_and_carries_no_hot_fix_tag`).

## A worker that sits out still enters a handle every probe interval (2026-11-03)

**What.** A worker on the Idle path now takes a Direct cycle once per `IdlePolicy::probe_interval` (1 ms by default): it pulls one handle,
and requests the next if the pull ran nothing, as any Direct cycle does. `IdlePolicy::with_probe_interval` changes it; zero never sits out and
`Duration::MAX` sits out for good, as before.

**Why.** Handles are served only when a worker enters them. The first-ranked worker never sits out; every other worker sits out when none of
its handles looks productive to it (or as the participation model says), and a worker on the Idle path enters no handle, so its view of the
handles, which is what keeps it out, changes only if it enters one. After a quiet spell every worker but the first-ranked one was out for good.
While that one ran a frame that waited (or ran long), no worker entered any handle: a lane source with items in its intake went unserved for
as long as the frame lasted, with every other worker idle. Euhedral-Execution's worker loop has the same rule and the same hole.

**Semantics.** The clock is the worker's own: `Cycle` keeps the time of the last cycle that chose a path other than Idle. Nothing is shared,
so the probe adds no atomic. A probe is an ordinary Direct cycle, so what it finds (frames pulled, frames pushed by the request) updates the worker's view
and can end the sitting out. Between probes the worker sits out exactly as before.

**Tests.** `fragment::tests::a_worker_that_sits_out_still_enters_a_handle` (a rank-2 worker whose policy says Idle, with the rank-1 worker held:
never entered the handle before), `a_worker_sits_out_between_probes`, `a_worker_that_sits_out_probes_once_per_interval`;
`euhedral-graph/tests/lane_source_liveness.rs` (lane sources on a real lattice with the first-ranked worker held in a frame: 10 items in the intake
unserved for 10 s before, served at once after).

## `DeferredCalls::run_through` waits for a place claimed and not yet published (2026-11-02)

**What.** `run_through(ticket)` returned `true` before the ticket's call had been made when another producer had claimed an earlier
place in the queue and not yet published its call. It now makes as many calls as the queue had claimed when the caller got in, and
leaves the cell between tries.

**Why.** The queue's consumer sees up to the first place that is claimed and not yet published. A drain entered by `run_through`
therefore stopped at that place and found the queue "empty" although the caller's own, later, call was already published behind it.
`reclaim` goes through `run_through` and had the same hole. `run_through_under_contention_never_returns_early` failed in most runs.

**Semantics.** `defer` counts its call (`queued`) before it claims a place in the queue. `run_through` reads `queued` once, when it is
called, and returns once that many calls have been made (`completed`, published with release), by itself or by another drain. Every place
claimed before that read was counted before it, so the ticket's call is among them; nothing walks or counts the queue, and a call counted
and not yet published is made as soon as its thread, which is running, publishes it. `pending` (`queued - completed`) now includes calls
on their way into the queue, so it is never below the calls that wait; before, it could read 0 while one waited behind a stalled
producer. Tickets are unchanged: they are the count at the call's own increment, so ticket order is not queue order and the target is the
count at the wait, not the ticket.

**Tests.** `deferred::tests::run_through_under_contention_never_returns_early` (12 of 20 runs failed before; 0 of 80 after) and
`run_through_never_returns_before_the_tickets_call_was_made` (a producer parked between its count and its offer: `run_through` waits for it, `pending` counts it).

## A worker's view of a handle changes by one service of it, and a push counts only through the handle (2026-11-02)

**What.** `UpstreamQueue` marks a handle productive or unproductive, per worker, by what one service of it showed. Pushes are
attributed to the handle they are pushed through: `note_routed_push(owner, routed)` takes the handle's claim owner and whether the
frame was routed, and counts only for the service of that handle in progress on the thread; the last push of a service decides
(a frame that finds no downstream leaves the handle unproductive for it). A pull that finds nothing but routed a frame through its
handle shows it productive. Before, a request counted any frame routed on the thread during it, and a pull that found nothing was
unproductive whatever was routed.

**Why.** This is the productivity observation of the original (`LatticeVertex.UpstreamInterceptor`, per worker and handle: begin
at acquisition, a push during the service sets it by whether it was routed, a pull that finds frames sets it, an early exit restores it), with the one
rule `UpstreamQueue` adds on top: a request that routes nothing keeps the worker's last view. The marking was already that rule; the
thread-wide push counter was not, since it also counted frames routed through other handles. The productive count steers idling and
the park of a cycle that ran nothing (`should_idle`, the Idle path's park, the unproductive park, `plentiful`).

**Same as the original.** A push counts only inside the service of its own handle, and the last push of a service decides (a frame
that finds no downstream demotes the handle for it). The original used to set the pushing thread's observation for any push,
including one outside a service of that handle on that thread, so the queue's `nonproductiveCount` could end one too high; it was
changed to Rust's scoping in Euhedral-Execution commit `1effc607` (`ProductivityObservation` records a push only between
`begin()` and `releaseLock()`), and the two now agree.

**How to use.** Nothing changes for a source that pushes through its own receiver inside `request` or `pull`.

**Semantics.**
- A new handle is productive. `reset_for_next_trial` makes every live handle productive again.
- Pull: frames found, productive; none found, productive only if the last push through the handle during the call was routed; a pull
  that stopped at a frame with none found, or that could not be made (complete, no demand, the lattice closed or draining, a panic),
  keeps the view.
- Request: frames delivered against the thread's request, or a last push that was routed, productive; otherwise the view is kept.
- The view is plain state in the worker's queue: another worker's pull does not change it, and a push by another thread does not
  mark it. The span of one service is a thread-local owner and flag that the push of that owner sets.
- A handle that a worker only requests, never pulls (a worker on the Staged path), stays productive for that worker, as a request that
  routes nothing shows nothing. This is the original's behaviour and is not changed here.

**Tests.** `an_acquisition_marks_the_handle_by_what_it_showed`, `the_last_push_through_a_handle_wins`,
`a_push_through_another_handle_is_no_evidence_about_this_one`, `each_worker_has_its_own_view_of_a_handle`
(`flow_control::upstream_queue`); `a_source_that_only_pushes_is_productive_when_it_pushes` (`control_plane::decision_tree`: a push-only
source that does not push is counted as a lane source is; one that pushes on every request is the one productive handle where workers
pull, and counts as any handle where they only request). Benchmark: `idle_polls`.

## Blocking frees and unloads wait in a device's `DeferredCalls` until its lanes are idle (2026-11-02)

**What.** `euhedral-memory` gains `DeferredCalls`, a `LatticeSource` of blocking key frames that holds a device's blocking driver
calls, `Deferral` (its producer handle), `Ticket` and `Bounds`. `Device::attach_deferral` gives a device one (once).
`DeviceMemory::free`, `PinnedHost::trim`, dropping an `Imported` mapping and euhedral-gpu's module unload then queue their call, with
the bytes it gives back, instead of making it where the memory is dropped. Without a deferral attached every call runs where it
is made, as before. `DeviceQueue` gains `issued()` (default `completed()`) and `LanePool` gains `idle()`; `CudaQueue::issued` is
the last marker. `Device` carries a weak handle to its own `Arc` (`Device::arc`).

**Why.** On the CUDA driver (615.71, RTX 5070 Ti) `cuModuleLoadDataEx`, `cuMemFree`, `cuMemFreeHost` and `cuMemHostUnregister`
make every other thread's `cuEventRecord` wait for the kernels running at that moment (370 ms for a 400 ms spin, each);
`cuMemHostAlloc` and `cuMemHostRegister` hold the driver for their own length (14.5 and 5.7 ms for 64 MiB); `cuModuleUnload` and
`cuCtxSynchronize` make only their caller wait; the allocation, stream, event, graph, texture and VMM calls do not stall another
stream. `Reaper` queues a free behind the device work that last used the memory, which does not help: the call stalls the other
streams' markers on whatever thread it runs. The call has to be made when no lane has work in flight.

**How to use.** Make a `DeferredCalls`, give it a receiver that feeds a lattice that may block (`add_downstream`) and frames to push (`request`),
`attach_pool(&pool)` the device's `LanePool` (held weakly), `Device::attach_deferral(calls.handle())` (the device holds it weakly: the owner
keeps the `Arc<DeferredCalls>`, so a queued call that owns the device does not keep the source alive), and call `calls.poll()` from a tick the
owner has: a frame leaves when calls are queued and the pool is idle, a bound is passed, or a thread asked with `drain_soon()`. `Bounds` counts device bytes only. A drain forced by a bound stalls every stream's
markers for as long as the kernels running at that moment. `Deferral::pending_bytes(Place)` is memory a planner may count as certain
reclaim; a thread that must not block (a lane's worker) counts it and calls `drain_soon()`, which pushes the drain frame without waiting. At a stop, `stop()` (from then on `defer` makes its call at once) and then `drain()` once the last user of the device has let go.
The drain frame is marked `RoutingPolicy::CacheLocal`, which a placement policy may read to keep blocking work off the workers that
run short work.

**Semantics.** One consumer: the calls are made by one thread at a time, in queue order, each exactly once, inside the source's
`SourceCell`. `drain()` returns 0 at once when another drain is running (the queue is busy, not empty); `run_through(ticket)` makes
every call queued when it is called, which includes the ticket's, on the calling thread: it becomes the consumer as soon as the cell is
free (a thread that offered a call and has not yet taken its ticket cannot make a counter overtake a call that has not run), and
returns `false` only when called from inside a call. An allocation that fails with `CUDA_ERROR_OUT_OF_MEMORY` (`DeviceMemory`, `PinnedHost`, `Vmm::alloc_vmm`,
`StreamPool`) makes the calls queued so far and tries once more; a second failure is returned. A call that panics is counted
(`panicked`) and the rest still run. Dropping the source (when the owner lets go) makes the calls still queued. `can_drain()` says whether a frame can still leave (not stopped, downstream not completed): a thread that waits for memory the queue holds checks it first. `pull` hands out nothing. `free_blocks`
of `DeviceMemory` is `false` once a deferral is attached. `LanePool::idle` sees marked work only: a lane that submitted and has
not marked yet is not seen.

`unsafe DeviceMemory::free_block` and `unsafe PinnedHost::free_host` free at once whatever the deferral, for an owner that queues the free itself to
keep something alive (a planner's claim) until the memory is really free.

**Tests.** `deferred::tests` (order, a drain in progress is busy not empty and the wait ends in order, a panic, drop, a call
cannot wait for itself, reclaim, eight producers, the frame leaves when the pool is idle and not before, the byte and age bounds,
no frame without demand, a dropped frame frees the slot); `lanes::tests::a_pool_is_idle_when_every_lane_reached_what_it_issued`;
`euhedral-gpu/tests/cuda_deferred.rs` (a drop while a kernel spins returns at once, the marker is made in microseconds, the calls
are made when the pool is idle; an out-of-memory allocation makes the deferred free and succeeds).

## A failed retirement reports one value: `Poisoned(cause)`, and the cause does not name the call that saw it (2026-11-02)

**What.** `Stream::confirm_retired` on the CUDA backend returns `Some(Poisoned(cause))` for every retirement that ends because
of a device fault, the faulting work's own included. It used to return `Fault { call: "device work", .. }` for a retirement whose
callback carried the fault status, and `Poisoned(Fault { call: "cuEventQuery", .. })` for one the idle look at the device reported
first. `cause` is exactly `Gpu::poisoned()`. The device's recorded fault is now the same on every route: `Fault { call: "device
work", code, name }` (the CUDA backend), with the driver's code and its name; the call that noticed it is not recorded. The fake
backend does the same with its own label (`"fake kernel"`). Direct calls (a launch, a copy) still return the `Fault` of their own call.

**Why.** The driver's callback thread and the owner's idle look race to see a fault first, so the value depended on timing:
`cuda_lanes_fault::a_poisoned_device_fails_armed_items` saw `Fault` or `Poisoned` from run to run. A retirement that ends in a fault
now has one value however the fault was learned.

**Semantics.** A retirement whose status is a device fault code poisons the device first, if the look has not, and reports
`Poisoned(poison)`. A status that is not a device fault (the work failed, the device is intact) is still `Call`. `Poisoned`
retirements (status `POISONED_STATUS`) and retirements that arrive with status 0 on a poisoned device are unchanged.

**Migrating.** Code that matched `Some(Fault { .. })` on a confirmed retirement matches `Some(Poisoned(cause))` and reads the
fault from `cause`.

**Tests.** `cuda_lanes_fault` and `cuda_lanes_stranded` (the device), `fake_lanes::*fault*`, `fake_streams::a_fault_surfaces_through_
confirm_retired_and_poisons_the_device` and `cuda_fault` assert `Poisoned(cause)` with `cause == gpu.poisoned()` exactly; `cuda::tests`
pins that a fault from two different calls records the same cause. `a_full_marker_ring_is_busy_and_old_markers_need_no_wait` moved to
`tests/cuda_marker_ring.rs`, a process of its own (the file says why).

## A polling stealer takes only the frame its stop looked at: placement is exact in poll mode (2026-10-30)

**What.** `ControlPlaneCache`'s one-frame take (the owner's `poll` and the stealer's `work_steal_polled`) returns nothing when the
partition it looks at is empty. It used to take the front anyway, right after the look.

**Why.** The take looked at the front, let `stop` judge it, and then polled. When the look found the partition empty and a producer
queued a frame before the poll, the poll took a frame `stop` never saw. For a stealer `stop` is what keeps ordered frames and frames
bound to the victim's core where they are, so a stealer on another core could run a frame the placement policy put on the victim's
core (1 in about 3,000 frames in the sink-pull placement test under CI load, always through `work_steal_polled`), or run an ordered
frame off its owner. Greedy takes were never affected: they check and take the same slot in one drain.

**Semantics.** The take holds the partition, so it is the partition's only consumer: the front it looked at is the front it takes,
and a frame published meanwhile queues behind it. A frame that arrives after a look at an empty partition stays for the next look.
Nothing is added to the frame path; an empty partition now costs one look instead of a look and a poll.

**Tests.** `a_polling_thief_never_takes_a_frame_that_arrived_after_its_look` forces the window (a cache-local and an ordered frame
queued between the look and the take; it fails on the old take); `a_placement_policy_also_places_the_frames_a_worker_pulls_from_a_sink`
asserts that no frame runs off its set.

## `one_shot` and `SetOnce` removed; users hold what they hand over (2026-10-30)

**What.** `euhedral-data-structures::sync` loses `one_shot`, `OneShotSender`, `OneShotReceiver`, `RecvError` and `SetOnce<T>`
(**breaking**), with their loom models and the shims only they used. `PipelineRunner::submit`, `submit_seeded` and `submit_on` return
`Arc<OutcomeSlot>` (new, `euhedral_core::ingest`) instead of `OneShotReceiver<Outcome>`.

**Why.** Each was a type around one atomic pointer. A user that hands one value over holds an `AtomicPtr<T>` itself: a boxed value,
published with a release swap, taken with an acquire swap, a leftover freed in `Drop`. `SetOnce` boxed its value behind an atomic
pointer and carried a loom duplicate of its constructor, a `swap_exclusive` helper and a `get_mut` that swapped twice to read.
`one_shot` kept a state byte and a cell, and an abandoned state that a rebuild on a pointer could only keep by counting handles.

**Semantics.**
- `OutcomeSlot::take()` is `None` until the chain ended, then the `Outcome` once; `is_set()` says whether one is waiting. A chain
  that is dropped without ending (a lattice closed under it, a drain that failed) publishes `Outcome::Cancelled`, so a reader never
  infers anything from a missing value; "abandoned" is not a concept any more.
- The frame-hand-off tests (`tests/lake.rs`) and the `graph_replay` example hold a `Handoff` of their own (an `AtomicPtr`).
- `LatticeEdge::downstream` and `UpstreamInterceptor::identity` are an `AtomicPtr` to a leaked `Box`, installed by a
  compare-and-set (sequentially consistent, as Java's `compareAndSet`; the edge's close pairs with it), read with an acquire
  load and freed in `Drop`. `DownstreamSlot` keeps the receiver inline with no atomic of its own: `attached` admits one writer
  and `live` publishes the write, so a read is the `live` load and no pointer hop. The process-wide driver and compiler handles
  use `std::sync::OnceLock`.

**Tests.** The pipeline runner's tests read `OutcomeSlot` (the chain a failing drain dropped now reports `Cancelled`); `tests/lake.rs`
hands a value from a frame to the main thread through a `Handoff`; the edge, interceptor and slot tests are unchanged.

## `high_contention_throughput` defaults to one source per logical CPU (2026-10-30)

**What.** The default source count is the logical CPU count (was the physical cores minus one). `--sources` and
`EUHEDRAL_HCT_SOURCES` still override it.

**Why.** With fewer sources than CPUs some workers have no source of their own to serve: on a 32-CPU machine 23 sources
measured 441.6 +- 3.2 M operations per second (p50 23 ns, p99 70 ns) and 32 measured 626.9 +- 39 M (p50 24 ns, p99 64 ns).

**Tests.** `the_sources_default_to_the_logical_cpus_and_can_be_overridden`.

## `euhedral-gpu`: texture objects over pitched device memory (2026-10-30)

**What.** `cuda::texture`: `View`, `Format`, `Filter`, `Address`, `TexObject`, `alignments` and `check_alignment`. Additive.

**Why.** A kernel that reads 2D memory through the texture units gets the hardware's filtering, clamping and normalised
coordinates for free, and the driver call that makes the object needs the descriptor structures and the device's alignment
rules, which every user would otherwise rewrite.

**How to use it.** Describe the memory and the sampling in a `View`, check it against `alignments(&device)` when the memory is
planned, and `TexObject::create` it in a prepare; pass `handle()` as a 64-bit kernel argument and read it with `tex2D<float>`
(normalised formats) or `tex2D<unsigned>` with `read_as_integer`. Drop the object in a prepare, after its last reader retired.

**Semantics.** The object is a 2D pitched resource (`CU_RESOURCE_TYPE_PITCH2D`) with one address mode on all axes, point or
linear filtering, and normalised or texel coordinates. It does not own the memory. A view whose base or pitch misses the device's
alignment, or whose pitch is less than a row, is an `Err` before the driver is asked.

**Tests.** `tests/cuda_texture.rs` on a real device: misaligned views are refused; point sampling in texels reads exact values
and clamps outside the image; linear normalised reads match the exact bilinear value to within the fixed-point weights' step.
Descriptor sizes and the alignment check are unit tests.

## `euhedral-gpu`: a lane's graph cache (2026-10-30)

**What.** `GraphCache<G, K>`, `GraphBody`, `Ran` and `GraphStats`: a lane's recorded graphs by key `K`, with `run(gpu, stream, key,
body)`, `warm`, `retain` and `free_stale`. Additive.

**Why.** A lane that runs the same short sequence of launches for each unit of work (a stage of several kernels) pays for each
launch separately. Recording the sequence once and replaying it with the unit's arguments costs one update and one launch, and
`euhedral-gpu` already records, fingerprints, updates and re-records; the cache is the bookkeeping that every such lane would
otherwise write: which key has a graph, what a changed shape does, when a stale graph may go.

**How to use it.** Keep a `GraphCache` in the lane's state beside its submitter (a lane context's `parts()` lends both), and put the
sequence in one `GraphBody::enqueue` that serves the direct run, the recording and the update. Warm each key ahead of the first
unit with `warm`. When a key goes stale (a new shape, a released prepare) call `retain`, and `free_stale` from the continuation
of the retirement armed after the lane's last replay.

**Semantics.** A hit runs the body as an update pass (it touches no device) and replays the graph: launch *n* of the pass gives
launch *n* of the graph its arguments. A miss runs the body for real once and records it. A body whose shape no longer matches
the recording is run for real and the graph is recorded again, in place when the driver can. A warm-up records a missing key and
leaves a recorded one alone. The cache holds at most `capacity` graphs: at the bound the oldest goes to the stale list. After the
first record of a key, a hit allocates nothing.

**Tests.** `graph_cache::tests` on the fake backend (hit, miss, shape change, warm-up, stale and bound), and
`tests/cuda_launch_alloc.rs`: 200 warm hits on a real device allocate 0.

## `euhedral-gpu`: timed spans around a lane item, and arguments reused in place (2026-10-30)

**What.** Two additions, no breaking change (`StreamOptions` gains a field: build it with `..Default::default()`).
- `Stream::span_begin` / `span_end` / `span_poll` / `span_cancel` and `Span`: a timed span of one stream's work. A stream keeps
  `StreamOptions::spans` of them (default 16, two timing events each); `span_begin` is `Err(Busy)` when the ring is full of open or
  unread spans, and a stream opened with 0 refuses them.
- `Args::clear`, `push`, `push_device`, `set`, `set_device` and `assign`: an argument list rewritten in place. `Args::layout` and a
  launch list their arguments on the stack (`MAX_ARGS`, 64).

**Why.** An item that launches the same kernels on every run would otherwise allocate a list of arguments per launch, a
vector of pointers per launch and, on a graph update, a clone of every launch's arguments. And a lane source that times its items
needs a duration it can read at its next entry without ever waiting on the device.

**How to use it.** Keep an `Args` for each launch the lane repeats, and refill it (`clear`, then `push`) or change one argument
(`set`). Bracket an item's launches with `span_begin`/`span_end`; at the next entry call `span_poll`: `None` means the end is not
reached yet (nothing was waited for), `Some(duration)` frees the slot. A killed item calls `span_cancel`.

**Semantics.** A span is two events recorded on the stream, one behind the work submitted before `span_begin` and one behind the
work before `span_end`; `span_poll` queries the end event once (`cuEventQuery`) and reads the elapsed time between the two. The
fake backend times a span in its own clock: one microsecond per operation the device ran. A graph's update pass keeps its staged
arguments, hasher and recording in the graph and reuses them, so after the first update it allocates nothing; a recording that
kept no values holds no allocation for them (`Frozen` is an `Option`).

**Tests.** `tests/cuda_launch_alloc.rs` counts every allocation of the process: 200 launches of a reused `Args` and 200 updates
with replays allocate 0, and a span around a 2e9-cycle kernel reads `None` at once, then its duration, and its slot is freed.
`tests/fake_streams.rs` pins the span ring on the fake backend, `src/args.rs` the in-place API.

## `Join` removed: fan-in is a count on the shared object (2026-10-29)

**What.** `euhedral-graph`'s `Join`, `Part`, `Parts` and `Round` are gone (**breaking**), with `tests/join_alloc.rs` and
the `join::loom` CI step.

**Why.** A fan-in needs nothing more than a counter the branches share. Java's `Join` is 49 lines (`pending`, the first
failure, a continuation); the Rust one had grown to 578 lines of value slots, move-only part tokens, rounds and re-arming.
Each of those is already given elsewhere: the frame lifecycle runs `do_finally` exactly once whether the frame finished
or was killed, so it is the arrival and needs no token; results belong in fields of the object the branches share;
re-arming is one store of the count before the branches are published.

**How to use it.** Put `pending: AtomicU32` and the result fields on the shared object (a record, a pair, a lease).
Store the branch count before publishing the branches. Each branch writes its field, then
`if pending.fetch_sub(1, Ordering::AcqRel) == 1 { /* run the next step */ }` where its frame ends. The read-modify-write
chain makes every branch's write visible to the last one, as an `Arc`'s last drop sees every earlier write.

**Tests.** `tests/credit_graph.rs`' pool lease is now such a count (`Lease`: the item's drop is its retirement end, the
staged item's drop its sent-or-dropped end), and the test no longer takes a `Mutex` around the lease's parts.

## Sequencer on Java's terms: any thread offers, the order grows, the entry is the caller's (2026-10-29)

**What.** `Sequencer<E: Entry>` is Java's `Sequencer` again. **Breaking:** `Registrar`, the fixed ring with its capacity
and `register` returning `None` are gone.
- The order is a growing multi-producer queue: `offer(entry)` from any thread, never refused (`MpscQueue`, whose chunks
  are recycled).
- The entry is the caller's own object and carries `Entry::ready`; the sequencer allocates nothing for it. `Progress` is
  the readiness word an entry embeds (pending, ready or abandoned; an abandoned entry is ready and concludes with
  nothing; its owner abandons it when it dies, or the order waits for it, and only a `Slot` does that by itself when it
  is dropped). A caller that pools its entries allocates nothing
  on the whole path (`tests/sequencer_alloc.rs`: 0 allocations over 20 000 rounds after warm-up).
- `drain()` is unchanged: the work-in-progress count, the `missed` loop, and a panic in the conclusion is the entry's own
  (the entries behind it still conclude; the first panic resumes on the draining thread once the drain ended). The
  conclusion takes `&E` and the entry is polled after it, as Java's `finally`.
- For callers without an entry type: `Ticket<T>` (an entry that carries a value), `Slot<T>` (its holder's handle: complete
  it, abandon it, or drop it), `Sequencer::ordered(emit)` and `register()`, which allocates the ticket.

**Why.** The ring needed one registering thread (a `Registrar` that is not `Clone`), so a consumer that admits from more
than one thread put a lock around it (`tests/credit_graph.rs`), and a full ring refused. Java's order is a queue any
thread offers to, and its entry is the caller's.

**Tests.** The reference ports, growth to 100 000 entries, offers from four threads, the caller's own entry type, loom
models of an offer racing a drain, two completions racing, and a dropped slot racing a completion (each fails if the
`missed` loop is removed; the order queue is not modelled).

## Placement policy (2026-10-29)

**What.**
- `ControlPlaneLattice::set_placement_policy(policy)`: one callback (not a registry) that runs first on every frame the
  lattice routes, before the shard and the core are chosen. It sees the frame and `&CpuSet`, the CPUs the lattice routes
  with now, and says where the frame goes by setting its origin and a cache-local routing policy; the routing that
  follows honours them. **The policy returns a `bool`: `true` when it placed the frame.** A lattice always has a policy:
  until one is set, it is a function that places nothing and returns `false`. It can be set only before the lattice
  starts (`LatticeError::Running` once started, `LatticeError::Closed` once closed, since a close returns while frames it
  admitted may still be winding down); a later call before the start replaces the earlier one. The policy is a plain
  field of the controller vertex, written before any source is ingested, so a frame reads it with no atomic and no view;
  the start, which every ingest follows, is what makes the write visible to the threads that read it.
- **The pull path.** A worker's pull from a source takes frames without the lattice's routing, so the interceptor's pull
  wraps the consumer the fragment passes in: each frame the source hands over is run through the policy. A frame the
  policy placed (it returned `true`) on a core other than the puller's, and kept there (a cache-local or socket-local frame whose origin the
  puller does not match), goes through the interceptor's push, the routing, which honours the origin; every other frame
  reaches the fragment's consumer on the core it is already on. The pull runs inside one read of the routing view
  (`Placement::within`), so a worker reads the CPUs once per pull, with no read-modify-write on anything
  shared. Where the puller runs is the (core, socket) its fragment recorded when it registered
  (`UpstreamRegistry::set_worker_location`); a thread that did not say routes every frame the policy placed. The wrapper
  is a stack closure. Ordered frames stop the fragment's pull before they reach its consumer, as before.
- **Concurrency.** The policy is `Fn + Send + Sync + 'static`, never `FnMut`: it runs on the frame path, from every
  thread that routes frames, at the same time. It must be re-entrant, take no lock, not allocate and not block; what it
  reads is the caller's to make safe to read from many threads.
- **The CPU set** is passed by reference, from the lattice's current routing snapshot, not by value: `CpuSet` is
  vector-backed, so a clone per frame would allocate on the frame path. The snapshot is a cell of the registry, republished
  when the effective topology changes, and the policy reads it from the thread's view with no lock.
- `PlacementPicker` implements the common policy (it returns `true` when it set an origin): a predicate on the frame chooses CPU set A or B; the chosen set is
  intersected with the lattice's; the nth CPU of the intersection is picked by `unsigned_multiply_high(routing_hash,
  count)`, and the frame's origin becomes that CPU, cache-local. No lock, no allocation (`CpuSet::intersection_len` and
  `nth_of_intersection` work word by word). A frame whose chosen set shares no CPU with the lattice's keeps its origin.
- `TopologyMapper::effective_cpus()`: the effective CPU set, public.
- `LatticeVertex::with_placement` / `flow_control::Placement`: the hook a vertex runs before its routing function.
  `set_policy` and `LatticeVertex::set_placement_policy` are `unsafe`: the caller guarantees nothing reads while they run.

**Limit.** The routing honours the origin only while the origin's core is active; during a rebalance, or after a clone failed to
start, a frame placed on an inactive core falls back to a hash, which can leave the chosen set. The policy chooses among the
CPUs the lattice reports as effective, not among those of cores that run a clone (publishing the latter would need the
shards to tell the lattice when a core starts or stops, which is not small and is left out).

**Cost.** Every lattice has a policy (the default places nothing), but the vertex keeps a plain `custom` flag, written with
the policy before the start: while it is false a push and a pull skip everything placement-related (no policy call, no read of
the view, no wrapper around the pull's consumer), and cost that one load. With a policy of its own set, a light reader (an
unregistered producer thread) pays one reference-count pair more per push, for the CPU set, and one per pull; a worker pays no
read-modify-write on anything shared. Measured with the high-contention benchmark (23 sources, 31 workers, `system-run
--system`, interleaved): before the flag, with the default policy always run, the head below this change gave 437, 434, 436
million ops/s and the default policy 394, 412, 410 (about 6% lower). After the flag, one interleaved round gave 420 for the head
below, 443 with no policy set and 418 with a policy that returns `false` (`--trivial-placement`); the remaining rounds were not
run, so the no-policy difference from the head below is still to be measured with three runs each.

**Java.** None of this exists in Java (frames are routed by hash and routing policy only); there is nothing to mirror.

**Tests.** The policy is settable only before the start (and refused after a close); it runs first and sees the unrouted frame and the effective CPUs;
frames land on the picked core; long calls land in set A and ordered lanes never do while every core is active; eight threads route at once;
placing allocates nothing (`tests/placement_alloc.rs`: the picker, and the policy read from a registered and from a light
reader).

## Lane source gaps from the first consumer (2026-10-29)

**What.**
- Teardown order. `LaneSubmit::quiesce` (default empty; the CUDA stream synchronizes) and `LaneState::close(&mut self,
  raw: &dyn Any)` (default empty) run when a lane source drops: quiesce the submitter, close the state against the
  submitter's raw handle (`LaneSubmit::Raw` is now `Copy + 'static`), drop the state, then drop the submitter. A state
  that holds device resources releases them while the stream still exists.
- `LaneSource::request` no longer allocates: `LaneSource::with_bound` sizes the waiting, armed, ready and scratch queues
  for a bound (`new` uses `DEFAULT_BOUND`, 1024).
- `euhedral_memory::cuda::StreamCopier::on_stream` (unsafe) queues its copies on a stream the caller made, so they order
  with the caller's other work on it; the copier never synchronizes or destroys that stream.
- Pitched copies: `Copier::start_pitched(from, to, PitchedCopy { src, src_pitch, dst, dst_pitch, row_bytes, rows })`
  copies rows between layouts with their own row pitches (padded 2D buffers). The default refuses; `HostCopier` copies
  row by row; `StreamCopier` issues one `cuMemcpy2DAsync`.

**Why.** The first consumer of lane sources dropped a state that owned device memory after its stream was gone, grew
the source's queues on the request path, and had to copy padded 2D buffers row by row.

**Semantics.** A pitch shorter than the row is refused before any copy starts.

## Queue fidelity, step 1: the audit and the first weakening (2026-10-29)

**What.** `mp_reserve` reads the producer's tail chunk with a relaxed load (Java reads it plain); the acquire load of the
tail that precedes it already orders it after the resizing producer's store. The field-by-field audit of the queues
against Java is in the pull request; the remaining rows are not changed here.
- `ChunkPool` is a wait-free single-producer single-consumer ring of chunk pointers (Java's `ChunkAllocator`), where it was
  a try-lock stack: the queue's consumer retires chunks into it, the growing producer takes them out, and neither waits.
- `StopWatch` counts its skipped pairs down instead of dividing a running count by the interval: the common start is a
  compare and a decrement.
- Benchmark: `--pooled` frames come from, and return to, the pool of the thread that runs them, the nearest Rust has to
  Java's sink re-pushing one array (a frame crosses no core between two uses).
- Other repos' names are gone from doc comments and READMEs; the behaviour is described instead.
- Benchmark: `--poll-spins N` sets how many spins the harness thread makes between two sums of the completion counter
  (default 32, upstream's). Each sum reads every worker's counter line (`perf c2c`: the harness thread is the loader of
  the hottest shared lines, 95 to 97% of their loads); at 1024 spins or more throughput is 5% higher (447, 443, 449 M
  against 424, 422, 424 M at 32, interleaved, same session).
- A loom model of the `ChunkPool` ring: one freeing side and one allocating side lose and duplicate no chunk, and the freeing
  side's writes are visible to the taker (weakening the tail store to `Relaxed` makes it fail).

## High-contention throughput benchmark (2026-10-29)

**What.** A new workspace member, `euhedral-benchmarks`, with the binary `high_contention_throughput`: the average time
and throughput of a lattice that many sources feed with no-op frames.

**Why.** An end-to-end figure for the control plane, to check each change of the hot path against.

**How.** `cargo run --release -p euhedral-benchmarks --bin high_contention_throughput` (see the crate's README for the
flags and `EUHEDRAL_HCT_SOURCES`).

**Semantics.** One frame in 64 is timed from its source handing it out to a worker executing it, and each iteration and the summary report the latency quantiles P0, P25, P50, P90, P99 and P100 (`--no-latency` turns it off). 3 forks (processes) of 3 warm-up iterations of 3 s and 5 measured iterations of 5 s; one invocation
waits for 32,000,000 completions. The harness thread has a core of its own. The output is one line per iteration and a
summary of the mean and standard deviation of the measured iterations.

The repeating source holds its frame queue's consumer and its banked demand in a `SourceCell`: the lattice serializes a
source's `request` and `pull`, so they are plain state of whoever is inside, and the queue the finished frames return
to is multi-producer, single-consumer (no per-frame consumer lock; one claim per call). Java's sink re-pushes a fixed
array and takes nothing back; a Rust frame cannot be aliased, so the return path is a harness cost with no Java
counterpart.

The completion counter has one cell per CPU id (a `PaddedLongAdder` with Java's `(cpus, true, true)` options: checked
indexes and each cell padded to two cache lines). A worker is pinned to its own CPU, so its cell has one writer and
finishing a frame is a load and a store; the CPU is read once per worker thread, from its affinity, and cached. A thread
that shares its CPU with another, runs on no worker CPU, or cannot learn its CPU (a platform that does not pin) takes a
shared cell by arrival order with a `fetch_add` and says so once on stderr. The latency recorders follow the same
cells. The harness thread never counts. Java counts with `getAndAdd`; the plain store on a cell with one writer is a
deviation the user asked for. A claimed cell is never released: a tally lives for one fork. The latency measured is from
the source handing a frame out to a worker starting it, and excludes the recycle path.

**Tests.** The harness-core selection, the worker CPU set, the completion target and its bounded wait, the sources
default, the output lines, and a smoke run of a fork with a tiny target on a real lattice.

## Fan-in without locks: one-shot, sequencer, lake; `Latch` removed (2026-10-29)

**What.**
- `one_shot` / `OneShotSender` / `OneShotReceiver` / `RecvError` (`euhedral-data-structures::sync`): Java's
  `CompletableFuture` for one result. The sender writes the value and publishes it with one release store; the receiver
  checks with `try_take` and `is_set`. A sender dropped without sending abandons the hand-off. **Nothing waits:** there
  is no blocking receive, no park and no wake.
- `PipelineRunner::submit`, `submit_seeded` and `submit_on` (audit C14) return `OneShotReceiver<Outcome>` instead of
  an `mpsc::Receiver` (**breaking**): poll with `try_take()` (`Ok(None)` while the chain runs).
- **`Latch`, `LatchSetter` and `LatchWaiter` are removed** (**breaking**; G5). A latch let the main thread block until
  the lattice's work was done; the main thread can check on it periodically instead, with a one-shot. It was the last
  blocking call in `euhedral-graph`.
- `Sequencer` (G6) is Java's: a ring of cells indexed by sequence number, each with a ready state word, and the thread
  that raises the work-in-progress count from zero concludes the ready prefix. **Breaking:** `Sequencer::new(capacity,
  emit)` returns the sequencer with its one `Registrar` (not `Clone`; `register(&mut self)` returns `None` while
  `capacity` slots are unreleased). A registration allocates nothing and takes no read-modify-write; the registrar
  keeps the order, so registration is by one thread at a time. A slot still holds an `Arc` of the sequencer.
- `Lake` (G7-G9): the attached set is a cell of the lattice's registry; `offer` and `admit` read it from the calling
  thread's view (a producer thread that offers every frame calls `register_reader` once, to hold the set and read it
  with no reference-count update). Attach and detach run under a lifecycle lock off the hot scope. The idle watch's downstream
  is a `DownstreamSlot`. `ControlPlaneLattice::registry()` is new.
- Lane sources and `Join` (G16, G17; barriers, not locks): `LaneSource::request` banks demand with one release
  `fetch_add` (and spends unbounded demand with none) and sets its wake flag with a release store, where it did
  three sequentially consistent read-modify-writes per call. `euhedral-data-structures` now builds the loom crate as a
  dependency under `--cfg loom` (the graph's join models run again).
- **`Join` API change (breaking).** A part is a move-only token: `Join::reset`, `Round::rearm`, `Join::once` (was
  `Join::new`) and the unsafe `Join::arm_owned` return the round's `Parts`, an iterator of `Part`s;
  `Part::arrive(value)` and `Part::abandon()` consume a part, and dropping one abandons. `Join::arrive(index, value)`
  and `Join::abandon(index)` are gone. A part ends once because it is consumed, so a double arrival does not compile,
  and the debug-only `closed` flag and the range check are gone; an arrival stays one `fetch_sub(AcqRel)`. A killed
  unit drops its parts and the round completes with `None`s. `Join::with_capacity` returns an `Arc<Join>` (a part holds
  its join, because there is no collector to keep it alive). `Join::discard_round(&mut self)` (for an `Arc::get_mut`
  owner), `Join::arm_owned` (unsafe: only the caller knows that no part of the old round will end) and `Join::is_idle`
  remain; `Join::reset` keeps its check. Reason: the earlier contract ("each part ends exactly once") was undefined
  behaviour reachable from safe code.

- `HOT_FIX_TAGS` loses C14, G5, G6 and G7 and lists 35.

**Why.** An outcome crossed a channel whose waker took a mutex on the worker that finished the chain; a latch took a
mutex per set; the sequencer took a mutex per registration, completion and released slot; and every offer to a lake
took the read lock of its generation and cloned its `Arc`. Locking is a failure of the dependency chain, and a frame's
path should hit no memory barrier.

**How.**
- `let mut outcome = runner.submit(x)?; ... outcome.try_take()` from the thread that wants the result, on its own
  schedule (a short sleep between checks is fine outside the lattice).
- A frame that hands a value to the main thread holds an `OneShotSender` and `send`s; the main thread checks `is_set`.
- A lake is used as before; its `offer` takes no lock while a set is attached.

**Semantics.**
- A one-shot has one receiver and one value. `try_take` returns the value once; a later call reports
  `RecvError::Abandoned`; `is_set` stays true after the take.
- The sequencer's drain is one thread at a time; a completion that arrives during a drain makes it look again. A
  panic in `emit` does not stop the slots behind it and resumes on the releasing thread once the drain ended. The
  `ready` flag is stored with release and loaded with acquire; the work-in-progress count is the only read-modify-write
  (the claim).
- A lake that finished, or an attach that returns the set that just refused a frame, hands the frame back. A set
  detached while an offer was on its way stays allocated until the offer has passed.
- The lake's watch stores the time a unit ended before it lowers the count, and reads the count before the time.

**Tests.** `send_before_take_and_take_before_send_both_deliver`, `a_dropped_sender_abandons`,
`ten_thousand_pairs_across_two_threads_each_deliver_once`, `loom_try_take_never_sees_a_value_before_it_is_complete`,
`outcomes_of_1000_chains_from_4_submitters_each_arrive_once`,
`registrations_from_4_threads_complete_in_reverse_and_emit_in_order`, `an_emit_that_panics_does_not_stop_the_ones_behind_it`,
`offers_racing_an_idle_detach_land_in_exactly_one_set`, `offers_on_an_attached_set_take_no_lock`,
`an_attached_for_life_lake_never_detaches`, `offers_from_8_threads_racing_complete_finish_every_admitted_unit`.

## Routing, cache and edges without locks (2026-10-29)

**What.**
- A lattice's changing state lives in **cells** of its `UpstreamRegistry` (`registry.cell::<T>()`, `set_cell`,
  `set_cell_if`, `cell_value`): the writer keeps the current value under a lifecycle lock and sends it, through a queue
  (`MpscQueue`), to the mailbox of every registered reader (`register_reader`, called by `register` for a worker). A
  worker drains its mailbox in one batch at the start of each cycle (`refresh_view`) and at a read, and applies the
  values to a view only it reads (`with_view(|view| view.get(&cell))`). A frame reads worker-owned state: no lock, no
  atomic read-modify-write, no barrier instruction. No reader locks: a thread that reads without registering gets a
  light mailbox on its first read (through the same queue) and withdraws it when it exits; its view holds weak
  references, upgraded for the length of one call (one reference-count update), so an idle thread never keeps a replaced
  value alive.
- Reclamation is `Arc`: a worker's view holds a replaced value until it drains (at its next cycle or read); a thread
  that exits drops its view, and the last holder to let go of a value drops it.
- `ControlPlaneCache` partitions (audit C4) are `ClaimCell`s, Java's `pLocks`: a consumer that finds a partition held
  passes it over, as before, with no mutex.
- `CacheDirectory` slots (C5) are cells of weak references: a steal reads one from its view with no lock.
- `LatticeVertex` routing (C6, C7): the routing generation is a cell of `Arc<RoutingState>`; a push reads it from the
  worker's view. The generation's downstreams are released from the `Drop` of the state, by whichever holder lets go of
  it last. Java's reader count (`admission`) is not needed: a push takes no count, and neither does it wait for a
  retirement. `routing_state()` and `RouteGuard` give way to `with_routing_state(|state| ..)`.
- `UpstreamInterceptor::request` names its vertex for the call, so the source's pushes inside it take no reference of
  their own. The drain flag and the edge's closed flag on the request and pull path are `Relaxed` (Java `getOpaque`).
- `LatticeEdge` (C8, C9): the downstream is a `SetOnce` and the parent a cell; neither read takes a lock. A receiver
  added to a closed edge is told `on_complete`, once, whether the edge took it or refused it.
- `HOT_FIX_TAGS` loses C4 to C9 and lists 39.

**Why.** Every frame a source pushes through the lattice crossed two vertex routing locks and two edge downstream
locks, each with a reference-count update on a line every worker shares; every steal attempt took a directory lock and
every cache take a partition mutex. Java has none of them: it reads the routing state, the downstream and the parent
with volatile loads and claims a partition with a compare-and-set. Rust has no collector to keep a replaced value
alive for a late reader, so a queue carries each replacement to the threads that read it, and they drop their copy when
they drain.

**How.**
- Read a routing generation with `vertex.with_routing_state(|state| ..)`.
- A long-lived thread that reads cells on every frame outside a worker's cycle calls `registry.register_reader()` once,
  to hold the values and read with no reference-count update.

**Semantics.**
- A remap reaches a worker at its next read or cycle; a frame in flight finishes on the generation it began with
  (nested reads do not drain). A dropped vertex clears its cell.
- **Retirement timing (deviation from Java).** A retired generation's downstreams are released when the last view
  holding it lets go: for each worker, at its next cycle or read. Java releases them when the last push that counted
  against the state ends. Category: missing Rust equivalent (there is no per-push count, which is what the views remove
  from the frame path); the delay is bounded by a worker's cycle and its timed parks.
- **Retirements never wait on a frame path (deviation from Java).** When the last holder is a reader letting go (a
  worker draining, or a light reader ending a read: a producer, another lattice's worker), `release_generation` hands the
  deferred retirements to a thread of its own (a worker that is a generation's last holder takes the book lock, allocates
  and hands off or spawns: per generation, not per frame); the writer's own drop on a lifecycle thread runs them inline; Java runs them on the last
  reader, where `closeClone` waits for the clone's workers, and for the clone's own worker that is a wait on itself.
  Category: Java bug against the frame-path rule.
- **Cost per push, exactly.** Inside the interceptor's `request` the routing state is read from the view: a worker pays no
  read-modify-write, a light reader one reference-count pair (the state's upgrade). Outside a request every push also
  upgrades the interceptor's `Weak` of its vertex, one more pair for any thread (a worker one, a light reader two).
- A dropped vertex runs the retirements still waiting for a generation to be released, on a thread of their own: the release
  of a generation needs the vertex, so nothing else would run them (a clone removed by a rebalance and closed by such a
  retirement would otherwise live, with its pinned worker, for the life of the process). No frame can route through a
  dropped vertex, so nothing is left to wait for.
- Cells are freed: a dropped `Cell` returns its id to the registry and tells every reader to forget its value, so
  rebalances no longer grow the store and every view. Each id carries a reuse epoch (in the `Cell`, the updates and the
  views): a read in progress cannot drain, so its view may still hold the dead cell's value for a recycled id, and a read
  whose epoch does not match the cell's reads the value under the lock instead. The same fallback serves a light read
  whose reader the writer dropped for a backed-up mailbox. A light reader whose mailbox holds more than 1024 updates (a
  thread that read once and never again) is dropped by the writer and registers again at its next read. Lifecycle code
  reads `LatticeEdge::parent()` under the lock and does not become a reader.
- `euhedral_hardware_utils::affinity::PinnedThreadExecutor::get_or_create_within` waits (up to a timeout) for a closed
  executor's tasks to end where `get_or_create` refuses at once with `ClosedWithActiveTasks`; building a fragment or
  placing a clone uses it, because closing an executor returns before its workers have wound down.
- The downstream of an edge stays allocated until the edge drops. A cache stays visible until it closes, and a
  stealer that found one holds it for the steal, so a closed cache lives until its last stealer lets go.

**Tests.** The cell and view tests in `flow_control::views`; `stealers_racing_a_greedy_owner_never_take_an_ordered_frame`,
`a_stealer_skips_a_claimed_partition_without_waiting`, `a_cache_closed_while_stolen_from_lives_for_the_steal_and_its_slot_reference_is_reclaimed`,
`pushes_racing_remaps_reach_only_validated_downstreams`,
`the_last_reader_of_a_retired_generation_closes_its_dropped_downstreams_once`,
`a_panicking_routing_function_gives_its_count_back`, `close_while_pushing_finishes_or_delivers_every_frame`,
`a_closed_edge_refuses_a_new_downstream`, `attaches_racing_a_close_tell_the_receiver_exactly_once`,
`set_parent_with_the_same_parent_stores_nothing`. The concurrency tests run under Miri with many seeds (the `miri` CI
job).

## The hot source and the fragment's output without locks (2026-10-29)

**What.**
- `LatticeHotSource` (audit C1, C10):
  - `before_each` and `after_each` take `Fn + Send + Sync` hooks, set when the source is built and called without a lock
    (**breaking**: they took `FnMut + Send`);
  - the downstream is one atomic word (Java keeps two, `complete` and `downstream`): null while open with no receiver,
    a link while one is attached, and a `COMPLETE` tag once completed. `accept` reads it (`Acquire`), runs the
    before-hook, reads it again, pushes and runs the after-hook;
  - `add_downstream` is a `SeqCst` compare-and-set loop that installs over null or `COMPLETE` and refuses over a
    link, `complete` is `swap(COMPLETE)`, and the receiver it removed hears `on_complete` once. Racing `complete`
    and `add_downstream` therefore never leave a receiver untold or tell a live one;
  - the after-hook follows every before-hook that returned, also when the push panics; a before-hook that panics
    finishes the frame with an error.
- `StopWatch` (C2): plain fields, `start` and `stop` take `&mut self`, as Java's watch is plain state of one thread.
  `StopWatch` is still `Clone`.
- `BodyCost` (C3): plain fields owned by the worker's decision tree; `record`, `calibrate`, `reset` and the new
  `timed` take `&mut self`. The worker times its own pushes with `BodyCost::timed` (a guard that records also when the
  push panics) instead of a pair of hooks sharing an `Arc<StopWatch>`, so the fragment's output carries no hooks.
  Another thread reads a snapshot, `BodyCostView`, that the owner publishes once per completed window of 32 samples
  and on reset; `ControlPlaneFragment::body_cost_ns()` returns it (`None` before the first full window).
- `HOT_FIX_TAGS` loses C1, C2, C3 and C10 and lists 45.

**Why.** The fragment's output runs for every frame its worker runs. Its two hooks shared a `StopWatch` behind a
mutex, took a second mutex per sampled frame for the body cost, and the source read its downstream under a read lock.
Java has none of that: the hooks are final fields, the stop watch and cost window are plain fields of the one worker,
and the downstream is read with an opaque load. The first Rust design kept the watch and window as relaxed atomics
because two hooks shared them; the worker times its own pushes instead, so both are plain.

**How.**
- Build a hot source with `LatticeHotSource::new().before_each(|frame| ..).after_each(|| ..)`; hooks that count or time
  use atomics (a hot source may have several pushers).
- State one thread owns needs no hook: the fragment's worker times its pushes with `BodyCost::timed`.

**Semantics.**
- Every receiver a source installed stays alive until the source drops (Java's collector keeps a detached receiver for
  a pusher still holding it). The source holds each in a link; a removed link is pushed onto a stack that only grows
  and is freed with the source. A source attached `k` times holds `k` retired receivers, and what they own, until it
  drops. No lifecycle call takes a lock.
- `accept` finishes the frame with an error when the source is complete or has no downstream, also when the
  before-hook completes it; the after-hook runs whenever the before-hook returned.
- A reopened source (`complete` then `add_downstream`) takes the new receiver; a second `add_downstream` on a live
  source is told `AlreadyAttached` and the first is kept. Racing attaches install exactly one.
- A `StopWatch` serves one thread, as Java's does; `&mut self` says so.
- `BodyCostView::publish` runs once per window (every 8192 frames at the default interval) and on reset, never per
  sample.

**Tests.** `hooks_run_around_each_push_from_one_pusher`, `complete_while_accepting_never_frees_the_receiver_in_use`,
`a_complete_source_reopens_with_a_new_receiver`, `complete_tells_the_old_receiver_on_complete_once`,
`racing_attaches_install_one_receiver_and_refuse_the_rest`,
`attaching_and_completing_from_many_threads_leaves_every_receiver_told_at_most_once`,
`dropping_the_source_drops_every_receiver_it_kept`,
`a_source_completed_by_its_before_hook_fails_the_frame_and_runs_the_after_hook`,
`attach_racing_complete_tells_every_receiver_exactly_once`, `the_after_hook_runs_when_the_push_panics`,
`a_panicking_before_hook_fails_the_frame_and_the_panic_goes_on`, `a_stop_watch_times_one_call_in_256`,
`the_estimate_follows_expensive_and_cheap_windows`, `the_policy_times_a_sample_of_bodies`. They run under Miri with many seeds (the `miri` CI job).

## Lock-free sources and claims (2026-10-29)

**What.**
- `euhedral_data_structures::sync`: `ClaimCell<T>` and `ClaimGuard`. `try_claim` is a `SeqCst` compare-and-set that
  returns `None` instead of waiting; dropping the guard releases with a release store, and
  `ClaimGuard::release_seq_cst` releases with a `SeqCst` store and a `SeqCst` fence, for a holder that reads other state
  next (its "request pending" flag, the queue's emptiness).
- Ingest (`QueueIngestSink`, `ArrayIngestSink`, `PipelineRunner`, `SingleUseSource`; audit C11-C13):
  - the receiver is a `DownstreamSlot` and the demand a padded atomic with Java's saturating `accumulate`; a source that
    is complete, has no receiver or has no demand refuses `request` and `pull` at once, and a `request` with no receiver
    banks nothing;
  - the queue's consumer is a `ClaimCell`, Java's `draining` flag. `pull`, the request drain and `discard_queued` each try
    it and never wait. A request that finds it held leaves `request_pending` (a wakeup, not demand), and the holder
    serves it after releasing;
  - `QueueIngestSink::clear` no longer waits for a running drain: it sets `clear_requested` and takes the claim if it is
    free; the holder applies the request as it lets go. A frame is cancelled exactly once either way;
  - the kill switch of a `PipelineRunner` is written and read `SeqCst`;
  - dropping a queue source cancels every frame still queued;
  - `SingleUseSource` takes its frame with a compare-and-set (`pull`, `request` and `complete` each try it). `complete`
    cancels an untaken frame, a completed source delivers nothing to a later `pull`, a `pull` whose stop condition
    matches the frame gives it back, and a frame nobody took is cancelled when the source drops.
- `euhedral-graph`: `Completions` (G1-G3) holds its poll state in a `SourceCell` (state the holder mutates) that `pull`,
  `request` and a late registration's sweep each try to enter. A `request` that finds it held banks its demand and leaves a wakeup, and a sweep that
  finds it held sets `sweep_pending`; the holder serves both after it releases. The lane pool is a fixed `Arc` that lives
  as long as the source (a marker is reached when its lane says so, also after `complete`), and the downstream is a
  `DownstreamSlot`. `complete` touches no poll state. Dropping the source, a sweep and a refused registration finish (`do_finally`) what
  they cancel, except a registration whose marker its lane has still not reached: that frame is leaked (`mem::forget`,
  as an I/O ring leaks in-flight buffers) and counted in `Completions::leaked_unreached()`, because the device may
  still use what the frame holds. `pending()` is decremented as each frame is handed to the consumer, and
  `demand` as each is pushed, so a consumer or downstream that panics leaves both exact.
- `euhedral-io`: the `AsyncIo` front (I1, I2) keeps the continuations a pull's `stop` held back in a `SourceCell` and its
  downstream in a `DownstreamSlot`; a `request` banks its demand with the same wakeup. The engines lock their tables
  through a `lock!` macro (the function it replaces named the lock type).
- State one thread owns is plain, or a load and a store, instead of a read-modify-write:
  - `Histogram::record_owned` records with plain loads and stores for a histogram with one recording thread at a time
    (`record` stays for any number of threads). A worker's service-time and throughput summaries, `Completions`'
    lateness (recorded under the poll claim) use it;
  - a worker's frame count and its in-progress count change by a load and a release store (the worker is the only
    writer; other threads read);
  - a pipeline's seed counter, drawn by the one owner of its frame manager, is a load and a store.
- `HOT_FIX_TAGS` loses 8 ids (C11, C12, C13, G1, G2, G3, I1, I2) and lists 49. The ratchet also counts the engines' `lock!(` call sites per file (5 in `uring.rs`, 31 in `win/mod.rs`), so the macro cannot hide a new lock.

**Why.** A source's `request` and `pull` are serialized by the lattice, and its lifecycle calls (`complete`, `clear`,
kill) arrive from any thread. None of that needs a lock: the lifecycle calls set atomics, the thread that holds the
source applies them, and a thread that finds the source held leaves a wakeup the holder serves.

**How.** A source keeps its consumer or poll state in a `ClaimCell` (or its worker-only state in a `SourceCell`), calls
`try_claim` and returns when it gets `None`, and after releasing re-reads the flags a refused caller sets before it tried
(`request_pending`, `sweep_pending`, `clear_requested`).

**Semantics.**
- `QueueIngestSink::clear` returns at once; the frames are cancelled by the caller if the queue was free, otherwise by
  the drain that held it, before that drain returns.
- A refused `request` is never lost. It banks its demand and stores `request_pending` (`SeqCst`) before it tries the
  claim; every holder, whatever it was doing (a pull, a request drain, a `clear`, a `discard_queued`, a sweep), ends
  with one tail that re-reads the flag after a `SeqCst` release and serves the request. The holder takes the flag with
  one `swap(false)`, which synchronizes with the store that set it, so it reads the banked demand with the flag (a
  plain store of `false` could overwrite a racing request's and read older demand on weakly ordered hardware). The
  same holds for `sweep_pending` and `clear_requested`.
- `Completions::pull` and `AsyncIo::pull` return 0 when another poll is inside; a panicking consumer still lets the
  poll serve what was refused. A panic while cancelling frames (`clear`, `discard_queued`) is raised after the rest
  were cancelled and the tail ran. A panicking stop condition or consumer ends a `SingleUseSource`'s frame and
  completes the source.
- Rule for the two cells: a `SourceCell` wherever the claim guards state the holder mutates (`Completions`' poll, the
  `AsyncIo` front's held frames); a `ClaimCell` where the claim guards no state of its own (the ingest queue's
  consumer role handle, Java's `draining` flag).
- `Completions` holds its lanes until its last owner drops it, not from `complete`.

**Tests.** `ClaimCell`: `a_second_claim_fails_while_the_first_is_held`, `a_released_claim_can_be_taken_again`,
`claims_from_8_threads_never_overlap`, `writes_under_one_claim_are_seen_by_the_next`, the wakeup test, and the `loom`
model `loom_two_claimers_exclude_each_other_and_see_prior_writes`
(`RUSTFLAGS="--cfg loom" cargo test -p euhedral-data-structures --release --lib sync::claim_cell::loom`). Ingest:
`a_request_while_a_pull_holds_the_queue_is_served_when_the_pull_releases`, `request_pushes_no_more_than_the_banked_demand`,
`a_source_driven_only_by_request_delivers_every_frame`, `a_publish_after_close_is_discarded_by_someone`,
`frames_queued_when_the_source_drops_are_finished`, `clear_racing_a_pull_discards_every_frame_once`,
`a_frame_is_delivered_or_cancelled_exactly_once`, `a_completed_source_delivers_nothing_to_a_later_pull`. Completions:
`registrations_racing_complete_are_each_cancelled_or_emitted_once`, `a_request_while_another_thread_polls_returns_at_once`,
`a_sweep_that_finds_the_poll_held_is_run_by_the_holder_when_it_lets_go`,
`a_request_refused_by_a_clear_holder_is_served`, `a_panicking_cancel_does_not_stop_the_clear_or_wedge_the_queue`,
`a_marker_registration_racing_complete_on_an_unreached_lane_is_not_emitted`. `AsyncIo`:
`a_request_only_lattice_delivers_every_completion`,
`a_request_while_a_pull_holds_the_front_returns_at_once_and_is_served_after`.

## Hot-path guard rails (2026-10-29)

**What.**
- `euhedral_data_structures::sync`: `LifecycleMutex<T>` and `LifecycleRwLock<T>` (locks for start, stop and rebalance
  state; they ignore poison) and `HotScope` (`enter`, `leave`, `active`), the debug check that keeps them off the hot
  path.
- A workspace `clippy.toml` (and the same entries in the `euhedral-graph`, `euhedral-io` and `euhedral-gpu`
  configurations) lists the standard lock types (`Mutex`, `RwLock`, `Condvar`, `Barrier`, the `mpsc` endpoints) and the
  calls that park, sleep, yield or spin.
- Every lifecycle lock of the audit (worker slots, fragment and lattice start/stop/close state, the registry, the
  pinned executors, the monitor's poller, the lake's retiring generations and idle watch, the driver and compiler
  handles) is a `LifecycleMutex` or `LifecycleRwLock`.
- Every hot-path lock the audit lists is tagged `HOT-FIX(<id>)` at its field and at each use the lint reports: 57
  ids (the audit's findings C1-C14, C16-C20, G1-G3, G5-G7, G10-G15, I1-I15, H1-H7, M1, and the source-contract markers
  S1, S2 and S6a). `euhedral-core/tests/hot_fix_ratchet.rs` lists the ids with the number of places each is used and
  checks each against a row of the audit. It also checks that every `allow` of a disallowed type or call in the library
  sources begins its reason with a listed `HOT-FIX(<id>)`, a `LIFECYCLE(<id>)` that is a row of the audit, or (for a
  call) `JAVA-BACKOFF`, and it refuses a type alias of a lock type. A change that replaces a lock lowers or deletes
  its entry; a reused or new id fails.
- The lint list also names `parking_lot`, `spin` and `lock_api` locks (`allow-invalid`, so an absent crate is fine).
  `Once`, `OnceLock` and `LazyLock` are lifecycle state (C34, H11) and are not listed. CI lints `euhedral-memory` with
  its `cuda` feature.
- `euhedral-hardware-utils` depends on `euhedral-data-structures`, so the layers are: data-structures and hashing,
  hardware-utils, core, graph and io, memory, gpu (`euhedral-gpu/tests/layering.rs`).

**Why.** The hot path (per frame, pull, request, offer, completion and tick) takes no lock. The lint makes a new lock
on it a build failure, the debug check makes a lifecycle lock reached from a frame end the test run, and the ratchet
makes the remaining hot locks a visible, shrinking list.

**How.**
- A new lock for start, stop or rebalance state is a `LifecycleMutex` or `LifecycleRwLock`; `lock()`, `read()` and
  `write()` return the standard guards directly.
- A wait, park, sleep or spin that is part of lifecycle work, or of an upstream back-off, carries
  `#[allow(clippy::disallowed_methods, reason = "LIFECYCLE(<id>)")]` or `reason = "JAVA-BACKOFF"`.
- A lifecycle step that a frame legitimately reaches (the first attach of a lake, the retirement of a routing
  generation, a sink's completion, a lattice's `add_upstream`) starts with `let _cold = HotScope::leave();`.
- Code that runs per frame, pull, request, offer or completion starts with `let _hot = HotScope::enter();`. The worker
  cycle, the upstream queue's `request` and `pull`, `LatticeVertex::try_push`, the cache's `push`, `greedy_while`,
  `poll_while` and steals, the ingest queue's `offer*`, `Lake::offer`, `Completions::add`, `Join` arrivals,
  `Sequencer::register` and `finish`, the stage graph's `arrive`, `run_stage` and `submitted`, `AsyncIo`'s submissions,
  the memory planner's and tier mover's `wake`, `Reaper::reap` and the CUDA retirement callback are covered.
- The process-wide compiler and driver handles (`Nvrtc::shared`, `Driver::shared`) read a `SetOnce` after the first
  load, so a stage that compiles a kernel takes no lock; the first load runs under `HotScope::leave()`.

**Semantics.**
- Release builds carry none of it: `HotScope` does nothing, `active()` is `false`, and the lifecycle locks are the
  standard locks.
- A debug build prints "lifecycle lock taken inside a hot scope, at <file:line>" (the lock call, by `#[track_caller]`)
  and aborts the process when a lifecycle lock is taken while the thread's hot depth is above zero. It aborts because
  the frame runners catch panics: a panic would be reported as the frame's failure and could pass unseen. The depth is
  per thread, scopes nest, and `leave()` suspends all of them until its guard drops. A test that provokes a violation
  installs `HotScope::install_violation_hook` (debug builds; the returned guard restores the previous hook, also when
  the test panics). The hook receives the location and the lock then proceeds. The hook is process-wide, so it keeps
  the locations it expects and forwards the rest to `HotScope::report_default`; the lake test has a test file, and so a
  process, of its own. The default report is written straight to stderr, because the test harness's output capture
  would otherwise swallow it when the process aborts.
  `HotGuard::drop` asserts the depth is above zero.
- The same locks guard the same state in the same places. The lifecycle locks ignore poison; the upstream
  registry's state, which took `lock().unwrap()` and so panicked after a panic in a holder, continues instead, as the
  rest of the lifecycle state already did and as the audit specifies.

**Tests.**
- `sync::lifecycle::tests` (data-structures): the lock outside a scope, the hook receiving each lock call inside one
  (debug), the default report aborting even inside `catch_unwind` (a child process, debug on Unix), `leave` and the
  restored depth, a poisoned lock, `try_lock`.
- `a_worker_cycle_runs_inside_a_hot_scope` (core fragment tests), `a_frame_that_completes_its_sink_does_not_panic` and
  `a_frame_taking_a_lifecycle_lock_through_a_lake_reaches_the_violation_hook` (graph `tests/lake.rs`).
- `hot_fix_ratchet` (core): the tags and their uses equal `HOT_FIX_TAGS` (57 ids), each id is a row of the audit, every
  disallowed `allow` is explained by a tag at the start of its reason, no type alias hides a lock, and the scan finds
  library code after an inline test module.

## GPU lane sources (2026-10-29)

**What.** `euhedral-gpu`:
- `Gpu::open_lanes_at(first, names, options)`: streams for lanes `first..first + n` of a pool built elsewhere
  (`LanePoolBuilder::reserve`). `open_lanes` is the same call at lane 0, with the pool made of the streams' queues (a
  provided method).
- `StreamOptions::priority` (`StreamPriority::{Normal, High}`): `High` is the context's greatest stream priority
  (`cuCtxGetStreamPriorityRange`, `cuStreamCreateWithPriority`). `CudaStream::priority()` reports it
  (`cuStreamGetPriority`); the fake reports `FAKE_NORMAL_PRIORITY` or `FAKE_GREATEST_PRIORITY`.
- `LaneSubmit` for `CudaStream` and `FakeStream` (`Error = GpuError`; `Raw` is the `CUstream`, or the queue id for the fake).
- `euhedral-graph`: `Outcome::Failed(S::Error)` (the item could not do its work: the source calls `failed` and not
  `then`, so no successor is delivered; `Outcome` takes the error type as its second parameter) and
  `LaneMetrics::stranded` (items dropped without `retired` because an item took their retirement).
- `open_lane_sources(gpu, lanes, groups)`: one pool of the device's lanes, its `LaneIntakes` and a `LaneSource` per lane
  (`LaneSpec`, `OpenedLanes`).

**Why.** A lane is a source that owns its stream, so the lattice's claim gives one worker at a time the lane's
submitter. The streams needed to join a pool of several backends, to carry a priority, and to speak the backend-neutral
`LaneSubmit`.

**How.** `open_lane_sources::<_, X, W>(&gpu, vec![LaneSpec { name, options, state }, ..], &[&[LaneId(0), LaneId(1)]])`,
then attach a receiver to each source and deliver work through `opened.intakes`. A pool of several backends: reserve
a range per backend in a `LanePoolBuilder`, `open_lanes_at` at the range's first lane, set each stream's `queue()`, then
`LaneIntakes::new` and `LaneSource::new`.

**Semantics.**
- The CUDA lane array is `Arc<[Option<Arc<LaneRing>>]>` of length `first + n`: the ids below the call's first lane are
  holes. `await_marker` refuses (`Err(Invalid)`) a marker of a lane the call did not open, or under another queue;
  `LaneSubmit::accepts` is the same test, so the lane source hands such a marker to the item (`LaneCx::bridges`).
- `LaneSubmit::mark` maps `Err(Busy)` (a full ring, or a slot an awaiter pins) to `Ok(None)`. `arm_retirement` returns
  the number the stream's retirement queue gives the retirement, which `poll_retired` reports as `Retired::seq`;
  `poll_retired` settles the ticket (`confirm_retired`) and reports the device failure.
- A retirement armed behind a lane that faulted may never get its callback: the work it follows never completes, and a
  drain of the stream's queue sees nothing. So when no callback has arrived and retirements are outstanding,
  `poll_retired` looks at the device once (one `cuEventQuery` of the stream's marker events, at most every millisecond;
  nothing when every marker completed). A query that fails for any reason but "not ready" poisons the device, and the
  next drain reports the outstanding retirements with `POISONED_STATUS`, so the items get `retired` with
  `Poisoned`. A retirement that arrives with status 0 is confirmed with the same query, as Java's `confirmRetired`
  queries the completion event. A retirement armed behind a fault is stranded only if no marker is outstanding on its
  own lane (nothing to query).
- The fake models the device's fault apart from the host's knowledge of it: work behind a fault never runs, its markers
  never complete, and the device is poisoned when a call that reaches it learns of the fault (a submission, a marker
  query, a retirement's status; `FakeGpu::observe_fault` stands for such a call in a test).
- `raw` lends the `CUstream`. The caller makes the context current (`Device::enter`) and a mark afterwards covers the
  work.
- `open_lanes_at` is `Err(Invalid)` for no names, a different number of options, a ring of no markers or lanes past 63;
  `open_lane_sources` is `Err(Invalid)` for a group lane that is not a lane or a lane in two groups.
- Lanes of two `open_lanes_at` calls on one device cannot order on the device: each call's streams refuse the other's
  markers. Open the lanes that must order each other in one call.
- Both `Stream` and `LaneSubmit` have `mark`, `id` and `await_marker`: code that imports both writes
  `Stream::mark(&mut s)`.

**Tests.**
- `tests/fake_streams.rs`: lanes at reserved ids, a marker of a lane the call did not open, the priority option.
- `tests/fake_lanes.rs`: a chain across a lane group, retirement read-back, a foreign lane's marker bridged, a full
  marker ring, the priority, bad group lists, a retirement behind a faulted lane reported with and without the
  faulting item arming one, a failed submission ending the item, and the submitter's numbering of arms.
- `tests/cuda_lanes.rs` (feature `cuda`, on the device): a 300-item chain across a group and a high-priority lane, the
  greatest priority, raw-handle work covered by the next mark, a foreign lane bridged with the CUDA lanes opened at
  ids above a hole. `tests/cuda_lanes_fault.rs` and `tests/cuda_lanes_stranded.rs` (a process each): a poisoned device
  fails armed items, with and without the faulting item arming a retirement.

## Lane sources (2026-10-29)

**What.** `euhedral-graph`:
- `LanePoolBuilder`: composes a `LanePool` from the queues of several backends. `reserve` takes consecutive lane ids,
  `set` places a queue at one, `build` makes the immutable pool.
- `LaneSubmit`: the backend-neutral submitter of one lane (`lane`, `id`, `queue`, `accepts`, `await_marker`, `mark`,
  `arm_retirement`, `poll_retired`, and the `unsafe` `raw`). `LaneState`: a lane's own hardware state (`check`,
  `met`) and what an item can wait for. `LaneWork`: an item (`run`, `then`, `retired`, `failed`, with `Outcome` and the
  `LaneCx` it runs in).
- `LaneIntakes`, `LaneInputs`, `Deliver` and `GroupId`: delivery into each lane's own intake (MPSC) and each lane
  group's shared intake (MPMC).
- `LaneSource<S, X, W>`: a `LatticeSource` per lane; `LaneMetrics`.
- `testing`: `FakeLanes`, `FakeLane`, `FakeOp`, `FakeLaneError`: a stepped fake `LaneSubmit`.

**Why.** A device lane's submitter has one owner. Running the lane as a source lets the interceptor's claim give one
worker at a time the lane's state, with no lock, no owner thread and no hand-off. A long call runs on the worker that
needs it, and other sources of the group take the shared intake meanwhile.

**How.** Build the pool with `LanePoolBuilder`, make the intakes with `LaneIntakes::new(pool, groups)`, make one
`LaneSource::new(submitter, state, inputs, intakes)` per lane and attach each with `add_upstream`. Deliver work with
`LaneIntakes::deliver` (or `LaneCx::deliver` from an item).

**Semantics.**
- `request(demand)` is always called. It banks the demand (a saturating add; a spend floors at zero), then in this order: drains the retirements that arrived and runs their continuations; calls
  `LaneState::check` once; retries the mark of an item the marker ring could not take; runs the waiting items whose wait
  is met; runs the lane's own intake; runs the group's shared intake; pushes at most the banked demand of ready frames,
  in the order they were made, and subtracts what it pushed. It runs every ready item, with no budget.
- Readiness is selective admission: nothing ready pushes nothing. There is no "has work" flag. Unmet demand
  accumulates.
- `pull(demand)` hands over at most `demand` ready frames, stopping before the first the `stop` callback refuses. It
  never banks demand, never runs an item and never drains a retirement.
- An item awaits input markers on its first run: markers of its own lane are skipped, markers the submitter `accepts`
  are awaited on the device, and any other (another backend's) is left for the item to bridge (`LaneCx::bridges`).
- `Outcome::Submitted { retire }`: the lane marks, calls `then(Some(marker))`, and arms a retirement if asked; the item
  waits in arm order for `retired`. `Finished` calls `then(None)`. `Wait(a)` keeps the item in the source; the lane asks
  `met` once per request, and items of the same order key wait behind it.
- A full marker ring (`mark` returns `Ok(None)`) stalls the lane: the item is marked first at the next entry, and
  no other item runs before it (retirement continuations still run first).
- The source's `SourceCell` claim is taken in every build (`SourceCell::try_enter`, one `SeqCst` compare-and-set). A
  `request` that finds it taken (a direct call racing the lattice, or a call from inside an item) has banked its demand
  and set a wakeup (`pending`, the ingest sinks' `requestPending`); the holder serves the wakeup when it lets go, so no
  demand is lost. A `pull` that finds the claim taken hands over nothing, as the sinks' `pull` does when their drain is
  taken.
- A retirement is paired with its item by the sequence number `arm_retirement` returned, in every build. A retirement an
  item armed itself (through `LaneCx::submitter`) pairs with no continuation. Items must not call `poll_retired` or `arm_retirement`; a retirement an item polls away strands that item's
  continuation, and the item is dropped when a later retirement arrives (arrivals are in arm order).
- A `pull` serves the wakeups left by requests made while it held the claim (from its consumer), as the ingest sinks'
  `pull` does in its `finally`.
- `LaneIntakes::deliver` to a lane that completed or was dropped, or to a group whose lanes all did, returns the work in
  `Err(LaneClosed)`, to be ended with an error. A delivery that races the closing can land after the lane drained its intake; that item is dropped with the
  intakes.
- Work an item answers `Wait` for stays in the source that took it, also when it came from a group's shared intake.
- Delivery is an enqueue. It spends no demand and runs nothing. The shared intake is taken one item at a time and its
  claim is not held while an item runs.
- `complete` is the ingest sources' `complete`: compare-and-set, zero the bank, tell the receiver; it drains nothing.
  Dropping the source drops the submitter first, then every item it holds, then ends the unpushed frames with an error.

**Tests.**
- `euhedral-graph/tests/lane_source.rs`: delivery, `pull`, completion, drop, and every rule of `request` on the stepped
  fake, with two-thread tests for the shared intake.
- `tests/lane_source_lattice.rs`: three group sources and one lane source on a real lattice (4 workers), 20 000 items
  with retirements, every frame executed once.
- `tests/lane_intake_alloc.rs`: delivery, `request` and retirements allocate nothing in steady state.
- `lanes.rs`: `LanePoolBuilder`.
- `euhedral-core` `SourceCell`: entry refusal in every build, `try_enter`, and a racing `try_enter` never giving two threads the state.

## Re-armable `Join` (2026-10-29)

**What.** `euhedral-graph`: `Join` is a lock-free fan-in cell that is armed again each round without allocating.
- `Join::with_capacity(capacity, then)` makes a cell for a pooled unit: `capacity` part slots and a continuation stored
  once. `Join::reset(parts)` arms a round (1 to the capacity); `arrive(part, value)` and `abandon(part)` end a part.
- The continuation receives a `Round`: `len`, `get`, `take` and `values` read the round's values (`None` for an
  abandoned part). The round ends with `Round::rearm(parts)` (starts the next round), `Round::end()` or a drop (the
  join is idle, ready for `reset`). It drops the values the continuation left. `Round` is `Sync` only when `T` is.
- `Join::new(parts, then)` keeps its behaviour: it arms at construction and runs the `FnOnce` with a `Vec` of the
  values (its one allocation per round). A join made by `new` has one round; `reset` on it panics.
- `new`, `arrive` and `abandon` are unchanged for callers. The `Mutex` is gone.

**Why.** Every completion took a `Mutex`, and a unit that recycles (a pooled object that joins its retirement with its
sender, round after round) had to build a new `Arc<Join>` and boxed continuation each time.

**How.**
```rust
let join = Join::<u32>::with_capacity(2, |mut round| {
    let (a, b) = (round.take(0), round.take(1));
    // Another round follows: arm it, then deliver its work.
    // round.rearm(2);
    // Otherwise, end the round BEFORE the unit that holds the join goes back to a pool:
    // until then `reset` refuses, and another worker that takes the unit would panic.
    round.end();
    deliver(a, b);
});
join.reset(2);
join.arrive(0, 10); // from any thread
join.abandon(1);    // the last arrival runs the continuation, once
```
Semantics:
- The count is one atomic. An arrival writes its slot, then `fetch_sub(1, SeqCst)`; the arrival that read 1 runs the
  continuation and sees every earlier arrival's value.
- A part is taken by one swap on its own flag, so two arrivals of one part cannot both write: the second panics. An
  arrival at a part beyond the round, at a join that is not armed, or after its round completed panics the same way.
  These checks run in every build, because a slot is shared memory.
- `reset` is one compare-and-swap from the idle state. It panics while a round is armed or its round is still held (not yet ended).
  `Round::rearm` is the form for inside the continuation: it consumes the round, so the values cannot be read once the
  next round's parts can write them. `rearm` stores the count and then reopens the parts, so a late arrival of the
  previous round is refused before it touches the count.
- Values a round leaves in a slot are dropped when the round ends (a panicking drop does not strand the join: every
  value is dropped, the join is released, then the panic continues), so a slot is empty whenever a part can write it.
- The continuation must not wait; it delivers work.

**Tests.** `a_join_of_64_parts_from_8_threads_runs_once_per_round`, `reset_from_inside_the_continuation_arms_the_next_round`,
`a_re_armed_round_never_sees_the_previous_rounds_values`, `fewer_parts_than_capacity_use_only_the_first_slots`,
`an_abandoned_part_reaches_the_continuation_as_none`, `a_unit_handed_back_inside_the_continuation_is_re_armed_at_once_by_another_thread`,
`a_duplicate_arrival_around_a_re_arm_is_refused_and_the_count_survives`, `a_value_whose_drop_panics_does_not_strand_the_join`, a `compile_fail` doctest that `Round<Cell<_>>` is not `Sync`, `reset_before_the_round_completes_panics`,
`reset_through_the_join_while_the_continuation_runs_is_refused` and the arrival-refusal tests (`join`);
`a_join_rearmed_10_000_times_allocates_nothing_after_construction` (`tests/join_alloc.rs`, a counting allocator); the
`loom` models `loom_two_parts_run_the_continuation_once_and_rearm`, `loom_a_unit_ended_and_handed_over_is_re_armed_by_another_thread`,
`loom_an_abandoned_part_and_an_outside_reset_start_a_clean_round` and `loom_one_part_racing_itself_is_taken_once`
(`RUSTFLAGS="--cfg loom" cargo test -p euhedral-graph --release --lib join::loom`).

## The source invariant's primitives (2026-10-29)

**What.**
- `euhedral-data-structures`: `sync::SetOnce<T>`, a cell installed at most once by a compare-and-set (`SeqCst`) and read
  with an acquire load. Its value lives until the cell drops.
- `euhedral-core`:
  - `SourceCell<T>` and `SourceGuard`: a source's plain state under the lattice's single-entry guarantee. (Amended by
    "Lane sources" below: `enter` and `try_enter` are safe and take a claim in every build.)
  - `DownstreamSlot`: the one receiver a source pushes to. `attach` refuses a second receiver with `on_error`, tells a
    late receiver of a completed slot `on_complete`, and otherwise installs the receiver, sets `live` and hands the
    receiver the source through `add_upstream`. `complete` tells a live receiver `on_complete` exactly once, however
    attach and completion race. `receiver()` reads without a lock.
  - `LatticeReceiver::serialize(action)`: runs `action` when no `pull` or `request` of the source the receiver serves is
    in progress, without waiting. The default runs it at once. It is the interceptor's own hand-off; a source's
    lifecycle calls do not use it.
  - `UpstreamInterceptor` implements `serialize`: actions go on an MPSC queue, and whoever holds the claim runs them
    before it releases. The interceptor installs its source before it is registered and drops it, through
    `serialize`, when it finishes.
- **Breaking** for code that implements or drives `UpstreamHandle`:
  - `try_acquire` returns `Option<Claim>` where it returned `bool`. A `Claim` is the proof that the caller holds the
    handle's claim. Code outside `euhedral-core` cannot make one; the default `try_acquire` still never contends.
  - `pull` and `request` take `&mut Claim` first, and `release` takes the `Claim` by value. A handle refuses a claim
    taken from another handle. This makes the interceptor's `unsafe` entry into its source sound for safe callers: a
    spurious `release`, or a `pull` while a serialized action runs, cannot be written.
  - `UpstreamInterceptor::upstream()` is replaced by `has_upstream(&Arc<dyn LatticeSource>)`. The source is identified
    by a weak reference whose allocation outlives the source, so an address reused after the source is dropped never
    matches.
  - The interceptor's claim is a counter (`wip`): `try_acquire` is `fetch_add(1) == 0`, and release is one swap to zero.
    A source completed through its interceptor (a source panic, `complete()`) is now completed when the worker that
    holds it releases it, not at the call.

**Why.** A source's state is touched by exactly one worker at a time, so it should need no atomics. Its lifecycle
calls (completion, release) arrive from other threads while a worker may be inside `pull` or `request`, and used to
need a lock or a flag per field. A source's lifecycle calls now set atomics only and the source applies them in its next
`request` or `pull`, where its state is single-consumer. The interceptor's claim as a counter lets a releasing worker
see an action that arrived while it held the claim, with no fence on the uncontended path.

**How.** A source keeps its worker-only state in a `SourceCell` and its receiver in a `DownstreamSlot`. Its lifecycle
calls (`complete`, and any clear or kill it offers) set atomics; its `request` and `pull` read them and apply the change
to the cell. The interceptor releases the source it holds through its own `serialize`. An implementer of
`LatticeReceiver` that is no lattice needs nothing: the default runs the action at once.

**Semantics.**
- Every serialized action runs exactly once, in the order it was handed over, never while a `pull` or `request` of the
  source is in progress, on the worker that holds the claim or on the caller if no worker does.
- The claim: `try_acquire` is `wip.fetch_add(1, SeqCst) == 0`. `serialize` offers the action, sets `pending`
  (`SeqCst`), then tries the claim; if it gets it, it runs the actions (clearing `pending` with one swap) and releases.
  `release` is one swap of `wip` to zero (`AcqRel`). If the swap returns 1, no caller tried the claim meanwhile; a later
  `serialize` finds zero and runs its own action. If it returns more, the swap has acquired from a caller's
  `fetch_add`, which follows that caller's `pending` store, so the holder sees `pending`, takes the claim again and runs
  the actions. An uncontended service is the `fetch_add` and one swap.
- The source is installed (compare-and-set, once) before the interceptor is registered, so a worker that finds the
  handle finds the source. The interceptor's `complete` flag is checked again after the install, and the source is
  released through `serialize`.
- `DownstreamSlot` touches atomics only. A lifecycle call from another thread never enters the source's `SourceCell`.
- `SourceGuard` is `Send` when `T` is and `Sync` only when `T` is.
- A panicking action is caught and logged; the actions behind it still run.
- `has_upstream` answers by identity (a weak reference, whose allocation is not reused while the interceptor lives) and stays true after the interceptor has dropped its source.

**Tests.**
- `SetOnce`: `set_hands_the_value_back_when_one_is_set`, `a_set_value_is_dropped_exactly_once`,
  `racing_setters_install_exactly_one`, and the loom model `loom_one_setter_wins_and_every_reader_sees_it`
  (`RUSTFLAGS="--cfg loom" cargo test -p euhedral-data-structures --release --lib sync::set_once::loom`).
- `SourceCell`: `a_second_thread_entering_while_one_is_inside_panics_in_debug`, `a_nested_enter_panics_in_debug`,
  `a_direct_request_racing_the_lattice_is_caught_in_debug`, `entries_in_turn_from_many_threads_are_accepted`.
- `DownstreamSlot`: `attach_racing_complete_tells_the_receiver_exactly_once` (10 000 rounds), the attach and complete
  order tests.
- The interceptor (`flow_control::vertex`): `serialize_runs_at_release_when_another_worker_holds_the_source`,
  `serialize_from_inside_the_sources_own_pull_runs_before_release`,
  `an_action_never_runs_while_a_pull_or_request_is_in_progress`, `release_races_serialize_and_no_action_is_stranded`,
  `a_release_and_a_serialize_in_the_same_instant_never_strand_the_action` and
  `two_serializers_and_a_release_in_the_same_instant_never_strand_an_action` (ignoring the release swap's result makes
  both fail), `the_source_is_in_place_before_a_worker_can_find_the_handle`,
  `a_new_source_is_never_taken_for_a_dropped_one_at_the_same_address`, `a_claim_taken_from_another_handle_is_refused`,
  `the_interceptor_drops_its_source_when_it_finishes`, `complete_completes_the_source_through_serialize`.
  `lifecycle_calls_from_another_thread_touch_only_atomics_and_take_effect_on_the_next_request`, and the compile-fail
  doctest on `SourceGuard`. CI runs the stress tests in release and the loom model. The `SetOnce` loom model carries a loom
  `UnsafeCell` payload, so a `get` weaker than `Acquire` fails it.

## `Marker` names its queue (2026-10-28)

**What.** `euhedral-graph`: `Marker` gained `queue: QueueId`, and `Marker::new(queue, lane, value)`. `QueueId` is an opaque,
non-zero identity drawn with `QueueId::next()` from one process-wide counter. `DeviceQueue` gained `fn id(&self) -> QueueId`.
**Breaking** for code that builds a marker with a struct literal (`Marker { lane, value }`) or implements `DeviceQueue`:
draw an id once per queue, return it from `id()`, and stamp it on the markers the queue makes (`Marker::new`).
`LanePool::reached` is false for a marker whose queue is not the lane's (or whose lane the pool does not have);
`FakeDevice` and `MarkerRetired` check the same. `euhedral-gpu`'s `await_marker` refuses a marker of another pool with
`Err(Invalid)`, and its `RetireTicket` carries the stream's `QueueId`.

**Why.** A marker named only a lane and a value, so a marker of another pool, with a lane in range and a value already
done, looked like one of this pool and could be treated as done.

**Tests.** `a_marker_of_another_queue_or_lane_is_never_reached`, `queue_ids_are_unique_and_non_zero` (`euhedral-graph`);
`a_marker_of_another_pool_is_refused_not_treated_as_done` (`euhedral-gpu`, fake and CUDA).

## GPU access: `euhedral-gpu` (2026-10-28)

**What.**
- A new workspace member, `euhedral-gpu`: backend-neutral, asynchronous GPU access on the lattice, with a stepped fake
  backend (feature `fake`) and a CUDA backend (feature `cuda`).
  - A stream is a shared, `Sync` queue (`DeviceQueue`, held by the `LanePool`) and a single-owner submitter (`Stream`:
    `Send`, not `Sync`). `submit`, `mark`, `await_marker`, `notify_retired`, `confirm_retired` and `Graph::replay`
    take `&mut`, so ownership replaces locking: `euhedral-gpu`'s own submission, marker, retirement and replay paths take no lock (the lock-free chunk pool's try-lock and `Completions`' internal locks are tracked by the separate lock audit). `submit` enqueues launches, copies and
    memsets; `mark` and `await_marker` order work across streams on the device.
  - `notify_retired` arms a completion point with the stream callback (`cuStreamAddCallback`, required at load; it reports
    the device status exactly once, even after a fault). The driver's callback only pushes the outcome onto the stream's
    unbounded, lock-free MPSC queue (pooled chunks, no allocation or free in steady state); the owner drains it with
    `poll_retired` and tells the listeners inside the owning source's `request` or `pull`. `arm_retirement` is the listener-free form. `retire_into` takes the
    submitter by value, turns the retirement into a frame of a `Completions`, and hands the submitter back to the
    continuation with the ticket; `OnceFrame` runs an `FnOnce` that owns it. `confirm_retired` is a submitter method.
  - A `Module` loads from source (NVRTC, loaded at run time, cached under a content hash) or from a prebuilt image
    (PTX, cubin, fatbin). `Args` packs typed arguments; the driver's parameter sizes are checked before a launch.
  - `Gpu::record`, `fingerprint`, `update` and `rerecord` record a recurring sequence on shadow streams, probe its
    shape, set new argument values in place and re-record it; `Graph::replay` launches it on any stream.
  - A device fault poisons the device; later calls fail fast with the cause.
- `euhedral-memory`: `cuda::Driver::shared()` (one library and one primary context per process) and
  `Driver::symbol(name)` (an entry point beyond the loaded table).
- The workspace layering is a test (`euhedral-gpu/tests/layering.rs`): a crate depends only on lower layers, through
  any dependency table. CI builds, lints and tests the `cuda` feature on Linux.

**Why.** Programs on the lattice drive a GPU without a frame ever blocking on it. Each application would otherwise
write its own submission, completion and interop layer.

**Use.** Open a device (`CudaGpu::open(0)`), load kernels with `CudaModule::from_source` or `from_image` at start-up,
open streams with `open_lanes`, and submit from any worker. See the [crate README](euhedral-gpu/README.md) and the
program `examples/graph_replay.rs`.

**Tests.**
- `cargo test -p euhedral-gpu` (no device): unit tests; `tests/layering.rs`; `tests/fake_streams.rs` (ordering,
  markers across streams, retirement once and in order, kept values, contiguous concurrent submissions, faults and
  poisoning, `retire_into` on a poisoned device); `tests/fake_graph.rs` (record, probe, update, re-record, replay).
- `--features cuda`, with a device: `tests/cuda.rs` (streams, retirement, NVRTC, modules and the cache, launches
  against the CPU bit for bit, cross-stream markers, the marker ring), `tests/cuda_graph.rs` (record, replay, updates
  in flight, re-record, a failed capture), and `tests/cuda_fault.rs` (a deliberate fault, in its own process). Without
  a device they skip, passing.
- The example `graph_replay` replays a recorded graph 100 times with changing arguments and checks each result against
  a CPU reference.

## `AsyncIo` moves to its own crate, `euhedral-io` (2026-10-28)

**What.** `euhedral-graph`'s `io` module (the `AsyncIo` API, the io_uring engine, the Windows IoRing and completion
port engines and their simulated tests), its integration test `tests/io_owned.rs`, its `async-io` feature and its
`io-uring`, `windows-sys` and (for the engines) `libc` dependencies move into the new workspace member `euhedral-io`.
`Continuation<T>` (what turns a completed operation's result into the next frame) moves with it: only the I/O used it.
`euhedral-graph` loses the `async-io` feature and the `io-uring` and `windows-sys` dependencies; `Completions`,
`ThreadUsage` and the rest stay.

**Why.** Asynchronous I/O is a piece of the control plane of its own, apart from completions, lanes and stage graphs.
`euhedral-io` builds on `euhedral-core` and `euhedral-data-structures`.

**Use.** `euhedral_graph::{AsyncIo, AsyncIoOptions, Continuation, Create, IoBuf, IoDone, IoDoneOwned, IoMemory, IoThen,
OpenSpec, OwnedFile, OwnedSock, RawFile, RawSock}` become `euhedral_io::{...}`; add `euhedral-io` to the dependencies.
Behaviour is unchanged: the code, its docs and its tests moved as they were, with their paths adjusted.

**Tests.** The same tests, now `cargo test -p euhedral-io`; the Windows workflow runs that package.

## Tiered admission and the copy rate: `euhedral-memory` (2026-10-28)

**What.**
- `PlanItem::tiered(consumer, priority, required, unit, units, min_first, order)`: a consumer of equal units that may
  live in any of `order`'s locations, fastest first. `Planner::admit` places its `min_first` units in `order[0]` with
  the plan's other items (spill, cut, refuse as before); once those fit, the other units fill the first location of
  `order` with free room, then the next, highest priority first. `Admission::tiers` gives the units per location;
  units that fit nowhere are left out (the consumer runs with fewer). Extra units use free room only, never ordering a
  give-back.
- Example `copy_rate` (feature `cuda`): pinned host memory to and from the device, by how the host memory is pinned
  (the driver's `cuMemHostAlloc`, a registered mapping on huge pages, on small pages), copy size and stream count.

**Why.** Residency in tiers: as much as possible in the fastest location, the rest streamed from the next, refused only
when nothing can hold it. And a measured answer to how to saturate the link.

**Measured** (RTX 5070 Ti, PCIe 5, 1 GiB per direction): registered huge pages 50-52 GB/s host to device, 57 device to
host, 58 both ways; registered small pages and `cuMemHostAlloc` 24-27 one way, 31 both. Copy size (2-128 MiB) and
stream count (1-4) change nothing: pin on huge pages.

**Tests.** `tests/planner.rs`: `a_tiered_consumer_fills_the_device_then_the_slower_tiers_in_order`,
`tiered_consumers_take_the_fast_tier_by_priority_and_what_fits_nowhere_is_left_out`,
`a_tiered_consumer_whose_required_units_do_not_fit_is_refused_or_dropped`.

## Tier moves: `euhedral-memory::tiers` (2026-10-28)

**What.** Planned movement of claimed memory between locations (device, pinned host, pageable host, any modeled one):
- `TierPool`: one planned allocation per tier, cut into slots, leased and returned without allocating;
- `Mover`: residents (one slot of an owner's data each) placed by priority, moved by asynchronous copies, spilled
  under budget pressure and brought back as room returns;
- `Copier`: the copy engine per pair of locations: `HostCopier` (`memcpy`), `cuda::StreamCopier` (`cuMemcpy*Async` on
  one stream, a ring of events made once), or any copier whose completion is a lane's `Marker` (`MarkerRetired`);
- `Moves`: a `LatticeSource` whose frames deliver completions to the residents' owners and drive the moves;
- planner: `Claim::transfer` (a claim changes owner without leaving its location's total), `Claim::is_elastic`,
  `Elastic::into_claim`.

**Why.** Planning where memory lives is half the job; the other half is moving it when the plan changes: a
higher-priority claim arrives, a consumer turns off, a budget shrinks. Each consumer copying its own buffers to host
memory on its own repeats the race the planner removed. One mover does every move under the planner's accounting.

**How to use it.** Allocate each tier once and give it to a `TierPool` under one owner. Make a `Mover` with the pools
(fastest first), the copiers and an optional staging tier. `place` residents with a priority and a completion handler;
`move_to` moves one; `with` reads or writes it while it is still; `release` gives it up (and cancels a move). Call
`pump` from any thread, or add `mover.moves()` upstream of a lattice. Install `give_back_handler()` as the pools'
owner's give-back handler and the handler of residents' owners, and call `restore` when budget frees up.

**Semantics.**
- No locks: residents are cells of atomics (free, resident, moving, busy); moves, free slots and completions go
  through lock-free queues. Nothing waits: a copy's completion is a test (`Retired`) checked by `pump`.
- Exact accounting at every point: a move leases the destination slot (its bytes become the resident owner's) before
  the copy starts and returns the source slot after it retires. A location's total never changes during moves.
- A resident placed in a full tier evicts that tier's lowest-priority resident below it to the next tier with room.
  Spilled residents come back home, highest priority first, whenever their home tier has a free slot.
- Slots of residents that can spill (not in the slowest tier) are elastic holdings of their owner. A give-back order
  at a tier retires free slots first, then spills the lowest-priority residents; a spill for an order retires its
  source slot rather than freeing it, so bring-back cannot refill it. `restore` takes retired slots back when the
  location has room.
- A pair no copier joins goes through the staging tier in two hops; a move that cannot continue leaves the resident
  in the staging slot.
- Releasing a resident mid-move marks it cancelled: when the copy retires, both slots go back and the owner hears a
  cancelled `MoveDone`.

**Tests.**
- 5 tests without a device (a copier whose copies retire a few checks late): moves between every pair with the bytes
  intact and the accounts exact while in flight and after; a move released midway; eviction by priority and
  bring-back; a give-back order that retires a free slot, spills a resident and gets its budget back; eight owners
  placing, filling and moving at once with no lock.
- 1 test on a device (feature `cuda`): one resident from pageable host to device to pinned host and back through stream
  copies and two-hop staging, bytes intact, accounts exact at each step.

## Memory planner: `euhedral-memory` (2026-10-28)

**What.** A new crate, [`euhedral-memory`](euhedral-memory/README.md):
- a location-agnostic `Planner`: budgets per location, owners with priorities and scopes, firm claims, elastic
  holdings, give-back orders, and a leak check;
- admission of a plan's whole working set before it allocates (`admit`: spill, then cut, then refuse);
- arenas (`SlotArena`, `ScratchArena`) that hand a session's planned memory out and back without the driver, with
  `BackendCalls` to prove the steady state allocates nothing;
- CUDA allocation backends (feature `cuda`): `DeviceMemory`, `StreamPool`, `Vmm` with POSIX-fd export, `PinnedHost` with
  a capacity cache, and `Device::import_fd`.

The rest of the workspace is unchanged.

**Why.** Consumers that size themselves from the free memory, each on its own, race: the last to allocate fails, and a
leak shows only as an out-of-memory error somewhere else. One planner per process accounts every location and owner,
admits a plan before it allocates and orders give-back by priority: "a pool where you can claim or free memory on a
device or the host".

**How to use it.** Make one `Planner` with the process's locations. Open an owner per component and lifetime.
- `admit` a plan before building it, and allocate from its `Reservation` once.
- Lease slots and scratch from arenas per unit of work.
- Drop the arenas (and so the claims) at the session's end, then `end_scope`.
- Add `planner.give_backs()` upstream of a lattice so give-back orders run as frames.

**Semantics.**
- No locks.
  - Claims are CAS loops on a location's `used`, which never exceeds capacity less reserve.
  - Owner slots are recycled through a lock-free queue once an ended owner's last claim drops. The slot's generation
    changes, so a stale `OwnerId` is refused.
  - A claim pins its slot; an order reads its owner's handler only under a pin.
- Give-back takes only from strictly lower priorities: lowest first, then the biggest holder. An elastic holding does not
  grow while an order to its owner is open. Freed elastic bytes count against the open order first.
- `end_owner` ends the owner's live descendants too. What they still hold is reported, and stays accounted until those
  claims drop (the memory is still allocated).
- Admission places an item's bytes per location. It keeps everything preferred if that fits in free plus reclaimable.
  Otherwise it spills items lowest priority first, each only if the spill location has room, then drops optional items
  lowest priority first, else returns `Shortfall`.
  - With give-back needed, it returns the orders and no reservation.
  - Otherwise a `Reservation` holds firm claims per location, and `take` moves bytes from it to an allocation's owner.

**Tests.**
- 14 tests without a device: claims, reserve, elastic give-back and its frames through the `GiveBacks` source, priority,
  leaks, baseline, admission fit / spill / cut / refuse / reclaim, the reaper, slot recycling under 1000 owners, eight
  threads claiming at once, and arenas over 10,000 units of work with no backend call.
- 3 tests on a device (feature `cuda`): every backend at 256 B to 100 MiB, VMM fd export, and pinned cache reuse.
- The `alloc_sweep` example measures CUDA against Vulkan allocation, import and copy on the local GPU.
## `Completions` and the io_uring ring take registrations through `MpscQueue` (2026-10-26)

**What.** In `euhedral-graph`, `Completions`' registrations (`after`, `after_marker`, `at`, `register`) go into a
`euhedral_data_structures::queues::MpscQueue` instead of a `Mutex<Vec>`, and an `AsyncIo` io_uring ring's entries that were
queued while another thread held its `sq` lock go into one instead of a `Mutex<VecDeque>`. The crate now depends on
`euhedral-data-structures`. `held`, `Completions`' waiting list, `Sequencer`, `Join`, the retry counts and the Windows
engine's backlog are unchanged.

**Why.** A thread registering a completion or a ring entry no longer takes a lock the polling thread also takes; the
poll no longer waits behind a registrant, and the registrant never waits behind a poll.

**Semantics that changed.** A registration that races `complete`. Before, `complete` set its flag under the registration
lock, so a registration either landed or was refused on the spot. Now `complete` sets the flag and appends one marker to the
queue. A registration that sees the flag is refused on the spot, as before (`do_finally`, a panic in debug builds). One that
does not is queued, and its position against the marker decides: ahead of it, it is a registration like any other and is
emitted; behind it, the next poll (`pull` or `request`, also of a completed source) cancels it through `do_finally` without
running it, and so does the last drop of the source if no poll comes. It ends exactly one way, never both and never neither;
that cancellation does not panic. The argument is on `Completions::complete`.

**The ring.** The `sq` holder drains the queue before it lets go, and looks again after it does; `flush_waiting` has the
argument that an entry pushed just after the holder's last drain is submitted by its pusher taking `sq` or by that re-check.
The flag that says "something was queued" stays, set after the entry is published: a queue's size counts reserved slots,
so it cannot tell a producer between reservation and publish from a hidden entry, and the crate does not spin.

**Tests.** `Completions`: a registration ahead of the marker is emitted and one behind it is cancelled; a `request` of a
completed source and the last drop cancel what is behind the marker; a registration racing `complete` (1500 rounds) ends
exactly once; 8 threads x 10 000 registrations against a hand-driven poller are emitted once each, in per-thread order.
Ring: entries queued while another thread holds `sq` are never stranded (1000 rounds of 4 threads against a slow enter,
the last pushes together so no later push can rescue one). The existing backlog and refused-enter tests pass unchanged.

## The branches of a fan-out run on different device lanes (2026-10-26)

**What.** A stage graph now places stages on lanes by Euhedral-Inference's PATH placement
(`StageGraph.markPaths`): each stage's lane is continued by exactly one successor, its critical
successor, and every other successor of a fan-out takes a lane in turn. `Topology` computes this once,
when the graph is built, as `path_predecessor`.

**Why.** `path_predecessor` was the predecessor on a stage's longest path from a root, so in a fan-out
A -> {B, C} both B and C continued A's lane. The branches never spread over lanes and ran one after the
other on the device, although the docs say a fan-out's other branches take a lane in turn.

**How to use.** Nothing changes in the graph's API. Give the branches of a fan-out several allowed
lanes in `StageSpec::lanes`; with a pool of two or more lanes they now run on different ones, and a join
waits (device side, in `StageCx::waits`) for the markers of the predecessors whose lane it does not
continue.

**Semantics.**
- Only submission edges place lanes; a completion edge never carries one.
- A stage's critical successor is its successor of greatest height (the stages on its longest
  submission path to a sink). Of equal heights, the first listed (in edge order) wins.
- A stage continues the lane of the predecessor it is the critical successor of. Stages are visited
  sinks first in a first-in first-out order (sinks by stage index; a stage is queued when its last
  successor was visited, and its predecessors are taken in stage index order), so a join that is the
  critical successor of several predecessors continues the one visited last. A join that is the
  critical successor of none continues no lane.
- The Java `transfers()` rule (a transfer stage never continues a compute chain) has no counterpart: a
  stage's allowed lanes already say which lanes it may use, and a continued lane outside them is
  ignored.
- Frame routing, id hashes and sinks are unchanged; this concerns the device lane a stage submits to.

**Tests.** In `stage_graph`: `fan_out_runs_both_branches_and_fan_in_runs_the_join_once_behind_the_other_lane`,
`a_fan_out_spreads_its_branches_over_the_lanes_and_the_join_waits_for_the_other_branch`,
`a_chain_keeps_one_lane`, `a_wider_fan_out_uses_every_lane`,
`a_longer_branch_continues_the_lane_whatever_its_place_in_the_edge_order`,
`of_equal_heights_the_first_listed_successor_continues`,
`a_strictly_greater_height_beats_an_earlier_successor`,
`a_join_continues_the_last_visited_of_the_predecessors_it_is_critical_for`,
`a_join_continues_only_the_predecessor_it_is_the_critical_successor_of` and
`a_completion_edge_never_carries_a_lane`.

## A fragment closes only its own use of its CPU's executor (2026-10-26)

**What.** `PinnedThreadExecutor` handles are now counted uses of the CPU's one shared executor, and
`ControlPlaneFragment::close` and `dump_locks` end only their own fragment's use: they interrupt the
fragment's worker task and `release` the handle. The executor closes when its last use ends. Fragments
of different lattices on one CPU, and a shard's replacement clone on a core whose old clone is
closing, no longer interrupt each other's workers.

**Why.** `close` and `dump_locks` called `PinnedThreadExecutor::close`, which closes the executor for
every user for good and interrupts every task on it. A second lattice on the same CPU lost its worker
to the first one's close, and its next `start` panicked ("a closed executor cannot restart"). In a
shard, a clone that failed to drain in time was replaced on the same core; the replacement shared the
running executor, and the deferred retirement of the old clone then closed it. The replacement's worker
left on `is_interrupted()` while its fragment still counted as running, so the core stopped without a
sound and its frames hung.

**API.**
- `PinnedThreadExecutor::release(&self)`: ends this handle's use. Idempotent; dropping the handle
  does it too.
- `PinnedThreadExecutor::submit_interruptible(task)`: `submit`, plus a `TaskInterrupt` whose
  `interrupt()` sets the `is_interrupted()` flag of that task alone and whose `is_finished()` says
  the task has ended and left the executor.
- `PinnedThreadExecutor::close` keeps its meaning: closed for every user. A user that only wants to
  stop its own work does not call it.

**Semantics.**
- A use is a handle. `get_or_create` and `get` each return a new use of the live executor; `Clone`
  adds one. A shut-down executor is restarted by a `get_or_create` as before. A handle joins an
  executor in one step with the check that it is not closed, so a use never lands on an executor that
  has just closed.
- When the last use ends the executor closes as `close` does (interrupting any task still on it) and
  its registry slot is freed once no task is left.
- A fragment's worker task has its own `TaskInterrupt`, set on `close` and on `dump_locks`. The
  fragment's other users of the CPU keep running, and `start` on a fragment sharing the executor finds
  it running.
- When the last use ends while its worker is still finishing, the executor is closed with a task left, and
  its slot stays taken until that task ends. A re-placement on that CPU in the meantime gets
  `ClosedWithActiveTasks`, and `BaseCloneableObject::place` panics on it. This limit of the registry is
  unchanged; closing a lattice waits for its workers (which leave the registry just before the task
  ends), so only a quick re-placement from another thread can meet it.
- `BaseCloneableObject::place` no longer shuts the executor down after the build: that interrupted
  every task on a shared executor, and a build racing another lattice's start could stop its worker.
  The executor is created under the fragment's worker name and left as found.

**Tests.** `executor_closes_after_last_user`, `releasing_a_user_leaves_the_other_users_tasks_running`
and `a_task_interrupt_reaches_its_own_task_only` (`euhedral-hardware-utils`);
`replacement_clone_on_same_core_keeps_running` and `dumping_locks_interrupts_only_the_fragments_own_worker`
(`fragment`); `two_lattices_on_one_cpu_close_independently` (`base_cloneable`, three lattices on one CPU
closing in turn).

## Lakes, latches and stage graphs over any frame sink (2026-10-26)

**What.** `euhedral-graph` gets `Lake` (with `UnitGuard` and `LakeClosed`) and `Latch` (with `LatchSetter` and `LatchWaiter`).
`GraphRuntime::stages` is now an `Arc<dyn FrameSink>` instead of an `Arc<QueueIngestSink>`, so a stage graph publishes into
a lake (or any other sink) through `offer`. The workspace gets one CI workflow, `.github/workflows/ci.yml`.

**Why.** A program has several kinds of ready work (stage frames, host work, control), each of which wants its own pool, spread
over several sources so producers and workers stay off one cache line, and attached to the lattice only while there is work,
because a worker polls every attached source for as long as it is attached. And the main thread needs one place to wait for
the lattice to hand it a result.

**Usage.**
- `Lake::new(lattice, sinks, idle_release)` makes a lake of `sinks` queue ingest sinks of one partition each.
  `Duration::ZERO` as `idle_release` never detaches them.
- `lake.admit()` counts a unit in (a quantum, a host task), attaches the sinks if they are detached, and returns a `UnitGuard`;
  dropping the guard terminates the unit. `UnitGuard::fork()` admits another unit even while the lake drains.
- `lake.offer(frame)` (also `FrameSink::offer`) queues the frame in the sink `routing_hash % sinks` picks, or hands it back.
- `lake.complete_gracefully()` closes admission.
- `Latch::new()` returns a `LatchSetter` to move into a frame and a `LatchWaiter` for the main thread; `wait` parks until
  `set`.
- `GraphRuntime { stages: lake.clone(), .. }`. The lake's units are the graph's quanta: admit before `StageGraph::start` and
  drop the guard in `retire`, or hold one for as long as the graph runs.

**Semantics.**
- A frame goes into one sink, by `routing_hash` taken as unsigned modulo the sink count. Frames of one hash therefore keep their
  offer order, and a sink has one partition, so a sink delivers in offer order; an ordered frame still runs in its own chain's
  order, and two chains in one sink may run at the same time.
- A frame is accepted only while a unit is admitted: with none, `offer` hands the frame back and attaches nothing. A refused
  frame is the caller's to end (the stage graph ends it in place).
- Admission counts units in one word with a closed bit. `complete_gracefully` sets the bit; admission then fails with
  `LakeClosed`, units already admitted may still offer frames and fork units, and when the last one terminates every sink
  completes gracefully (it delivers what it holds, then completes and detaches) and the idle watch ends. A lake with no unit
  when it is closed completes at once. After that `offer` hands every frame back.
- The sinks attach to the lattice at the first `admit`, not at `Lake::new`. The attachment is a *generation*: the sinks and
  their idle watch. The watch is a `LatticeSource` that holds no frames and checks the lake whenever a worker pulls from it or
  requests from it (two atomic loads while a unit is admitted). When no unit has been admitted for `idle_release`, it takes
  the generation out of the lake, completes its sinks gracefully and completes itself, which detaches all of them. The next
  `admit` makes a new generation (`Lake::attachments` counts them). `admit` counts the unit before it asks for a generation,
  and the watch checks the count again under the lock that takes the generation, so a unit never loses its sinks.
- A frame that races a detach (its sink completed first) goes to the generation attached since, as long as a unit is admitted;
  an offer re-checks the unit count under the lock before it makes a generation, so a unit that ended meanwhile leaves no sinks
  attached. `Lake::is_attached` is true until every sink, including those of a detached set that is still draining, has completed.
- Attaching registers the new sources with the lattice outside the lake's lock (registration asks the source for demand, which
  polls the watch). A lattice that refuses (closed) fails `admit` with `LakeClosed::Lattice` and the half-attached generation
  cancels what it holds.
- A lake dropped before it completed lets its sinks drain and complete.
- `Latch::wait` is the one blocking call: it parks the thread until the value is set, returning at once if it was set before.
  A `LatchSetter` dropped without setting makes `wait` panic instead of hanging.

**Tests.** `tests/lake.rs`, on a lattice of one CPU (the first the process may run on, so the tests assume no second core):
`frames_of_one_hash_keep_order`, `sinks_detach_after_idle_and_reattach_on_admit`, `a_lake_with_no_idle_time_never_detaches`,
`offer_without_admitted_unit_is_refused`, `complete_gracefully_drains_then_completes` (with a fork), `latch_set_before_wait_returns`,
`latch_set_from_frame_wakes_waiter`, and a latch whose setter was dropped. They take turns (two lattices of one CPU running
together made a rebalance panic with "cannot restart the executor"). `tests/credit_graph.rs` publishes its stage frames through
a lake of two sinks. The stage graph's unit tests publish through a queue sink behind a small `FrameSink` adapter.

## `AsyncIo::read_into`: reads into caller-owned memory (2026-10-26)

**What.** `AsyncIo::read_into(file, offset, buf, then)` reads into memory the caller owns, such as a pinned host
arena, instead of an `IoBuf`'s `Vec`. `buf` is any `IoMemory`; `then` receives an `IoDoneOwned { buf, result }`. The
engines (io_uring, IoRing and the completion port) work on a view of the region, so nothing is copied.

**Why.** The inference engine reads weights from disk straight into host memory it allocates and pins itself.

**API.** `pub unsafe trait IoMemory: Send + 'static { as_mut_ptr, len, is_empty }`, implemented for `Vec<u8>`;
`IoDoneOwned<B> { buf: Option<B>, result, filled }`. `read` is unchanged.

**Semantics.**
- The result is a `read`'s: `Ok(n)` with `n` the region's length, or less if the file ended after some bytes (the first
  `n` bytes are the file's, the rest untouched); `UnexpectedEof` when no byte was read. An empty region is refused at
  submission, with nothing started.
- The region is cut into pieces of at most 256 KiB, as a `read`'s buffer is; each piece lands at its own offset and
  the engines' short-transfer and slot-identity rules apply unchanged.
- `then` always runs. The value comes back in `IoDoneOwned::buf` after a success and after an error alike. `buf` is
  `None` only for an operation the engine cannot prove finished with the kernel, which already leaked its buffer on
  Windows (a ring that broke, a ring entry that may have reached the kernel): the value is leaked (never dropped) and
  `result` is the ring's failure. A submission that returns `Err` started nothing and drops the value.
- `filled` is the count of bytes at the start of the region that arrived: `n` for `Ok(n)`; after an error, the bytes of a
  Windows read that failed with a warning status (`ERROR_MORE_DATA`) and moved bytes anyway, as `IoBuf::filled` reports.
- `IoMemory` is the only new `unsafe` surface; its contract (initialized, stable when moved
  without claiming unique ownership, exclusive until returned) is on the trait.

**Tests.** `tests/io_owned.rs` (`read_into_fills_owned_region`, `read_into_split_pieces_preserve_offsets`,
`read_into_short_read_reports_count`, `owned_buffer_returned_on_error`); in `io/uring.rs`,
`reads_into_parts_of_one_arena_land_in_place`; in `io/win/tests.rs`, the owned reads through the simulated ring and port
and the leak of a broken ring's region (the continuation runs with `buf` `None`); in `io/win/live.rs`, a read into owned memory on a real Windows.

## `AsyncIo` on Windows (2026-10-26)

**What.** `euhedral-graph`'s `AsyncIo` gets a Windows engine: IoRing for file reads and writes where the OS has a kernel
ring, and I/O completion ports for everything else (sockets, accept, connect, and files without a ring). Linux and
Android keep io_uring. The portable half of `io` (the public API, the `LatticeSource` behaviour, the frames a pull's
`stop` holds back) moved to `io/mod.rs`; the Linux engine and its tests are otherwise unchanged. The default feature is
now `async-io` (it was `io-uring`).

**Why.** Applications need `AsyncIo` on Windows too, and it was Linux-only.

**API additions, on every platform.**
- `RawFile`, `RawSock`, `OwnedFile` and `OwnedSock`: the platform's handles (descriptors on Linux and Android, `HANDLE` and
  `SOCKET` on Windows). Linux signatures are unchanged.
- `AsyncIo::open_path(path, OpenSpec, then)` with `OpenSpec { read, write, create: Create, unix_mode }`. `AsyncIo::open()`
  stays the constructor, so the plan's "`AsyncIo::open(path, ..)`" is `open_path`; `open_file(path, flags, mode, then)`
  stays on Linux and Android.
- `AsyncIoOptions { windows_ioring }` with `AsyncIo::open_with_options`; the option is ignored off Windows.

**Semantics on Windows.**
- Each operation ends through a poll: the "skip the port on success" mode is not set. An overlapped call that fails at
  once (no packet will come) ends with that error at the next poll.
- File reads and writes use rings only if the probe says the OS has a kernel ring (not `IORING_FEATURE_UM_EMULATION`) with
  both operations supported (`IORING_VERSION_3`, Windows 11 22H2 and later); otherwise both go through the port. A file keeps
  its route: IoRing refuses a handle that belongs to a completion port (`ERROR_INVALID_PARAMETER`, found by a live test), so
  a handle that joined the port stays on it and a handle a ring serves is never joined. The IoRing functions are resolved with `LoadLibraryExW` and `GetProcAddress`, never linked, so the
  library loads on Windows without them.
- Open and rename run on the OS thread pool (`TrySubmitThreadpoolCallback`) and post their result to the port, so they
  come out of the same poll. This is the one place the crate uses threads it did not create.
- A handle joins the engine's port when its first operation is submitted and must not belong to another port. A file handle
  opened without `FILE_FLAG_OVERLAPPED` is refused with `InvalidInput`.
- A full ring submission queue goes to the ring's backlog and is handed over by a later submission or poll, as on Linux.
  A flush builds at most 16 entries before it submits, and a submission never waits behind a submit.
- **Operation identity.** An operation's record lives in a slab and is never freed while the engine lives, so a late
  packet naming its `OVERLAPPED` can be read safely. A slot is idle, issuing, in flight or busy; its generation changes at
  every issue, and a ring entry carries slot and generation. A port packet carries only an address, so it is accepted for a
  slot that is in flight on the port with an `Internal` status that is not `STATUS_PENDING`. A poll leaves the packet of a
  slot whose call is still returning for the next poll.
- **Warning statuses.** A call that reports failure with a warning status (`ERROR_MORE_DATA`, `WSAEMSGSIZE`) still queues a
  packet; the real layer reads the `OVERLAPPED`'s status after such a call and treats a warning as "a packet follows".
- Results are decoded from the packet's `NTSTATUS` and byte count, not through the handle (which the caller may close). Socket
  statuses map to Winsock errors for the common cases (refused, reset, aborted, unreachable, timed out, message too long).
- A handle that skips the port on success, or a socket opened without overlapped I/O, is refused. A handle already on
  another port is refused too (`ERROR_INVALID_PARAMETER` for a handle this engine did not join): a handle serves one
  `AsyncIo`, ever.
- A slot stays `Issuing` until its call is made (or its ring entry or pool job is handed over): nothing that ends or fails
  operations touches a record being filled. A ring that cannot be read ends its operations and the entries behind it, and an
  operation sent to a broken ring ends with an error instead of waiting. A poll whose inbox holds only packets of returning
  calls also looks at the port. A packet is taken only if its count agrees with the `OVERLAPPED`'s `InternalHigh`; status and
  bytes are read from the `OVERLAPPED` as atomics. A read or receive that fails with a warning status keeps the bytes that
  arrived: the result is the error (`ERROR_MORE_DATA`, `WSAEMSGSIZE`) and `IoBuf::filled` is their count.
- Open and rename leave their result in a queue the poll drains (a job that panics or is dropped still ends the operation
  with an error). Writes go in pieces of at most 256 KiB, since a write that extends a file is synchronous on NTFS. Paths
  near the Win32 limit go as verbatim paths. A ring whose completions cannot be read ends its operations with the error.

**Tests.** 45 tests run on every platform against a simulated `Sys` (every operation's success, error and short transfers,
the ring and its fallbacks, a full submission queue, the order across ring and port, a pull's `stop`, late and warning packets,
slot reuse, routes and generations, handle modes, pool failures). 16 run on Windows
only (`io/win/live.rs`: files, loopback TCP and UDP, accept, connect, open and rename, threads each with a ring), each twice:
with IoRing and forced onto the port. `.github/workflows/windows.yml` runs the crate's tests on a `windows-2025` runner on
pushes to `graph`.

## Dataflow sources: `euhedral-graph` (2026-10-26)

**What.** A new crate, [`euhedral-graph`](euhedral-graph/README.md), adds two kinds of `LatticeSource` that the
lattice's own polls drive, and the structures built on them: `Completions` (continuation frames behind markers,
timers and completion tokens), `AsyncIo` (io_uring file and socket operations whose completions are frames; Linux and
Android), `Sequencer`, `Join`, `LanePool` with `DeviceQueue` and `Marker`, and `StageGraph`. They are ports of
Euhedral-Inference's `runtime.graph` package (`StageGraph`, `LanePool`, `Join`, `AsyncReads`) and
`quantization.lattice` (`GpuCompletions`), and of Euhedral-Execution's `FrameSequencer`. `euhedral-core` is unchanged.

**Why.** A frame must not wait: a worker is a core. Work that finishes elsewhere (a device queue, a file, a socket, a
deadline) was left to the application to poll by hand. These sources make the completion itself a frame: the lattice
polls the source like any other, and the continuation runs on whichever worker it is routed to.

**How to use it.**
```rust
let (lanes, devices) = FakeDevice::pool(&["compute"]); // or your own DeviceQueue lanes
let completions = Completions::new(lanes.clone());
lattice.add_upstream(completions.clone())?;
let marker = devices[0].submit(cost, &[]);            // submit; this returns at once
completions.after_marker(marker, Box::new(next_frame)); // next_frame runs once the marker is reached
```
- A `StageGraph<Q>` runs a static DAG of stages per quantum `Q`, with submission, completion and external edges, a
  kill switch per quantum and a `retire` callback that receives the `QuantumOutcome`. Its stage frames go to a
  `QueueIngestSink`; ordered stages use the id hash (`StageSpec::ordered`), unordered ones `randomize_hash`. Frames keep
  the default routing policy.
- `GraphRuntime::retry_backstop` (default `DEFAULT_RETRY_BACKSTOP`, 20 ms) bounds how long a retried stage waits for
  a lane's progress.
- Features: `io-uring` (default; `AsyncIo`, Linux and Android only) and `testing` (`testing::FakeDevice`). Windows and
  macOS get everything but `AsyncIo` for now.

**Semantics.**
- `Completions` queries each pending token once per poll, emits continuations in registration order and honours the
  pull's `stop` (ordered frames go out through `request`'s pushes and the lattice's routing, so ordered lanes keep
  registration order). A frame whose kill switch is set is emitted anyway: the executor skips it and runs its
  `do_finally`, so `pending()` drains to zero. A registration after `complete` finishes its frame through `do_finally`
  and fails a debug assertion.
- `Completions::lateness_us` reports `(p50, p99, max)` in µs, each rounded up, from a `metrics::Histogram`: the maximum
  is exact, the percentiles are interpolated within a bucket (a quarter of the value wide).
- A `StageGraph` never scans for ready work: the last arrival at a stage publishes it. A quantum retires once none of
  its frames can run again and every lane it used reached its last marker, and not before every `QuantumRef` clone was
  dropped.
- `QuantumOutcome` is how a quantum ended; it is not `frames::Outcome` (how a pipeline chain ended).
- `AsyncIo` has one submission ring per submitting thread; an operation belongs to the thread that submitted it.

**Tests.** 75 unit tests at that point (`Completions`, `AsyncIo`, `Sequencer`, `Join`, `LanePool`, `StageGraph`, `FakeDevice`), the
README doctests, and `tests/credit_graph.rs`: a 4-worker lattice running a per-item stage graph with a `Sequencer`, a
`Join`, credit-bounded admission, a jump to a new generation and a pause, which fails if a frame callback runs over 200 µs without the OS
preempting it.

## Metrics restored (2026-10-24)

**What.** `euhedral-core` gets a [`metrics`](euhedral-core/src/metrics/mod.rs) module, the port of upstream's
`metrics` package (`ExecutionMetrics`, `CacheMetrics`, `MetricsAggregator`), which the port had dropped. Each fragment
worker and each cache registers meters in a `MetricsRegistry`; `MetricsRegistry::snapshot` and
`ControlPlaneLattice::metrics` read them all without stopping the workers. The fragment's cycle state has upstream's
two `FlowRecorder`s back, which feed the throughput summary and the two flow meters.

**Why.** Upstream reports latency, throughput, in-progress, backlog and cap-factor meters through Micrometer; the
port had the raw state (`in_progress`, `cap_factor`, the cache's count) but no way to read it as metrics.

**Departure from upstream: service time, not latency.** At the user's direction, RustEE has no `.execution.latency`.
Upstream's was a batch-end sample of a smoothed per-cycle service time (cycle time over frames). RustEE's
`.execution.serviceTime` summary records every frame's own body time: how long its `execute` takes to return. It is
`execute` only, not `do_finally` or the rest of the frame's lifecycle, and not the time a frame takes to get from
ingest to a CPU. Its percentiles are the frames'.

**How to use it.**
```rust
let registry = MetricsRegistry::new();
let config = LatticeConfig::with_metrics("app", registry.clone()); // or:
let fragment = FragmentConfig::default().with_metrics("app", registry.clone()); // fragments and their caches
// ... run the lattice ...
let snapshot = registry.snapshot(); // or lattice.metrics()
let p99 = snapshot.get("app.execution.serviceTime", cpu).and_then(|m| m.summary()).and_then(|s| s.percentile(0.99));
```
- `FragmentConfig` and `CacheConfig` have upstream's `metric_prefix()` and `registry()`, with `with_metric_prefix`
  and `with_registry` (`None` turns metrics off). `FragmentConfig::with_metrics` sets both its own and its cache
  config's, like `FragmentConfig.ofDefaults(metricPrefix, registry)`; `LatticeConfig::with_metrics` is
  `LatticeConfig.ofDefaults(.., metricPrefix, registry)`.
- `CloneableObject::metrics_registry` (default `None`) lets a lattice find its object's registry.

**Semantics.**

| Meter | Kind | Recorded | Upstream |
|---|---|---|---|
| `<prefix>.execution.serviceTime` | summary, ns | every frame, around its `execute` call in `AbstractExecutor::run` (`time_execute`) | replaces `.execution.latency` (`reportLatency` in `reportMetrics`) |
| `<prefix>.execution.serviceTime.flow` | flow, ns | after each productive cycle: the `FlowStats` of a recorder of the cycle's mean `execute` time | none |
| `<prefix>.execution.throughput` | summary, frames/s | at each batch end: the frame recorder's rate × 10^9 | `reportThroughput` in `reportMetrics` |
| `<prefix>.execution.throughput.flow` | flow, frames/s | after each productive cycle: the frame recorder's rate `FlowStats` | none |
| `<prefix>.execution.inProgress` | gauge | 1 while the worker is inside a cache take, a direct pull or a steal (`timed`) | `executionStarted`/`executionFinished` around `drainLocalCache`, `pullAvailableUpstream`, `stealWork` |
| `<prefix>.execution.frames` | counter | after each productive cycle | none |
| `<prefix>.cache.backlog` | gauge, frames | read at snapshot: the cache's count | `CacheMetrics` gauge on `TOTAL_COUNT` |
| `<prefix>.cache.capFactor` | summary | in `ControlPlaneCache::update`, the cap factor it set | `recordCapFactor` in `update` |

- **Execute timing.** `AbstractExecutor::run` wraps its `execute` call in `metrics::time_execute`. A metered worker
  installs its meters in a thread-local for its loop, so on that thread `time_execute` reads the clock before and
  after `execute` and records the difference; the finalizer is outside it. A frame that is not alive is not executed,
  so not timed; an `execute` that panics is not recorded. An executor that overrides `run` opts in by wrapping its own
  `execute` call in `time_execute`, and nothing else. An executor that runs frames on another thread is not timed.
  The decision tree's body cost keeps its own sampled timing of the push (one in 256), unchanged.
- **The recorders** are recorded where upstream's `recordProgress` records them, after each productive cycle and at
  the cycle's end time: the mean `execute` time of the cycle's frames (from the service-time summary's count and sum,
  so no further clock read) and the frames run. A worker reset starts them over, as `CycleState.reset()` does; the summaries
  stay cumulative.
- **Prefix:** `euhedral` unless set; blank means the default, and a trailing `.` is dropped (`metricName`).
- **Tags, per CPU:** `core` (the clone's core id) and `cpu`. Upstream has only `core`, so a core's two SMT workers
  share one Micrometer meter per name; here each worker and each cache has its own, which is what the user wants.
  `HistogramSnapshot::merge` combines them if a reader needs a core's total.
- **Summaries** are cumulative histograms of atomics: count, sum, max, and log2 buckets (each octave from `2^-16` to
  `2^48` split in four), so percentiles are within a quarter-octave. Negative and `NaN` values are ignored, as
  Micrometer ignores negative amounts.
- **Flows** are published by the worker through a sequence lock: the worker never waits, readers retry a read a write
  overlapped.
- **Lifecycle:** a fragment or cache registers when built and leaves its registry when closed or dropped (upstream's
  caches never removed their meters).
- **Cost.** No lock and no allocation on a worker's path; each worker owns its meters on their own cache lines.
  Without a registry a cycle pays one untaken branch, and each frame's `run` one relaxed load of a process-wide flag
  (and, once any metered worker has started in the process, a thread-local lookup) before calling `execute`. With
  one, each frame pays two clock reads and a histogram record, and each productive cycle up to about 30 ns, mostly the
  two `FlowRecorder` updates (`metrics_cost_per_cycle`, which records no `execute`: 21 to 22 ns over 4 to 8 ns off).
  Measured on an i9-14900K, in release, one SMT core's two workers running empty frames from an array sink
  (`metrics_overhead_per_frame`, medians of two runs of seven rounds, off and on alternating in one process): greedy
  mode 156 ns/frame off, 175 to 176 on (+19 to +20 ns, +12 to +13%); poll mode, with a cycle per one or two frames,
  244 to 246 off, 302 to 303 on (+55 to +59 ns, +22 to +24%). The user's frames run for milliseconds, so this is noise
  for them.
- `MetricsAggregator.getInProgressGauges(registry)` matched the latency suffix upstream; `in_progress_gauges` matches
  the in-progress suffix.
- No exporter: the `metrics` crate facade is push-based, so a bridge would either record through a global recorder on
  the hot path or poll snapshots on a thread of its own; an application can write the latter in a few lines.

**Tests:**
- `metrics/histogram.rs`: buckets and their bounds, recording, percentiles, `since` and `merge`, concurrent writers.
- `metrics/flow.rs`: publishing, no torn reads under a concurrent writer, `FlowStats` from a recorder.
- `metrics/tests.rs`: the names (upstream's, and service time) and prefix rules, nothing built without a registry,
  meters and tags, the aggregator lookups, sources leaving the registry, snapshots while four writers record.
- `config/*`: the prefix and registry settings and defaults.
- `control_plane/cache/tests.rs`: no meters without a registry; backlog and cap factor tagged with the core; closing or
  dropping a cache removes its meters.
- `metrics/tests.rs` also: `run` times `execute` alone on a thread with a sink, not a 3 ms `do_finally` and not a dead
  frame, and nothing without a sink; `time_execute` returns what `execute` returns.
- `control_plane/fragment/tests.rs`: no meters without a registry; both SMT workers and caches register, tagged;
  running workers record frames, throughput and flows while a reader snapshots; a cycle counts frames, feeds the
  service-time flow from the timed `execute` calls and reports throughput at batch ends.
- `impls/base_cloneable/tests.rs`: a metered lattice reports every worker's meters, every frame's `execute` timed,
  until closed; `the_service_time_is_execute_alone_not_the_finalizer`: 50 µs `execute`s with 2 ms finalizers record
  at least 50 µs each and under 1 ms.
  Ignored measurements: `metrics_cost_per_cycle` (fragment tests) and `metrics_overhead_per_frame`.

## Greedy, not drain, for holding a partition while its frames run (2026-10-23)

**What.** `ControlPlaneCache::drain` and `drain_while` are renamed `greedy` and `greedy_while`; the mode they make is
**greedy mode**, against **poll mode**. The tests and docs that named it "drain" follow.

**Why.** In RustEE, **drain** means stop accepting work and finish what is in flight (`set_drain_mode`, a draining
edge or vertex, `is_drained`, a shard draining its clones). **Greedy** means consume as fast as possible by holding
locks longer, which is what this take does: it holds a partition while it runs the frames it takes from it. One word
for both was confusing.

**How to use it.** Replace `cache.drain(..)` with `cache.greedy(..)` and `cache.drain_while(..)` with
`cache.greedy_while(..)`. Behaviour is unchanged. The queues in `euhedral-data-structures` keep their `drain*`
methods: there the word means taking items out of a container.

## Routing policy honoured by stealing and direct pulls (939ccfb, 2026-10-23)

**What.** Work stealing and a worker's direct pull from its upstreams now follow a frame's
[`RoutingPolicy`](euhedral-core/src/flow_control/mod.rs). Until this change, stealing stopped only at ordered frames.
So:
- an idle worker on any core could steal an unordered `CacheLocal` frame from the core it was routed to;
- a `SocketLocal` frame could be stolen across sockets;
- the direct path could run a `CacheLocal` frame bound to another core.

**Why.** The routing policy and the ordered flag are both binding. A frame's policy says how close to its origin it
must run. Routing already placed frames by it:
- the lattice's `route_to_socket` sends `SocketLocal` and tighter frames to the origin's socket;
- a shard's `route_to_core` sends `CacheLocal` frames to the origin's core.

Stealing and direct pulls must not undo that placement.

**How to use it.** Set the policy and the origin on the frame's header before it is published:
```rust
let mut header = FrameHeader::new(id_hash);
header.set_origin(euhedral_hardware_utils::affinity::current_cpu_info().copied()); // or the CpuInfo to stay near
header.set_routing_policy(RoutingPolicy::CacheLocal);
```
A frame with no origin, or with `Anywhere` (the default), goes anywhere, as before.

**Semantics.** `locality_allows(frame, here, queued)` in `control_plane/cache.rs` decides whether a worker on core
`here.0` of socket `here.1` may take a frame that another cache holds (`queued`), or that is still in an upstream
(`queued = None`). The cache that holds a frame may always run it.

| Policy | Who may take it from elsewhere |
|---|---|
| `Anywhere`, or no origin | any worker |
| `CacheLocal` | a worker on the origin's core only: its SMT threads, where `route_to_core` places it |
| `SocketLocal` | a worker on the origin's socket only |

**Fallbacks keep a frame stealable:**
- **`CacheLocal`, origin core inactive:** routing falls back to a hashed core. The frame then keeps to the origin's socket, so it is never left to one worker.
- **`SocketLocal`, origin socket inactive:** routing falls back to another socket. The frame then goes anywhere.

**Stealing:**
- `work_steal` and `work_steal_polled` stop, in each partition, at the first frame that is ordered or that its policy keeps where it is.
- `upstream_pull` (the direct path) stops before the first ordered frame and the first frame bound to another core, leaving both to the lattice's routing.
- Ordered frames are unchanged: only the owning cache's CPU runs them (see poll mode below).

**Tests** (`control_plane/cache/tests.rs`, `impls/base_cloneable/tests.rs`):
- `locality_allows_what_each_routing_policy_lets_leave`: the predicate for every policy and fallback.
- `stealing_leaves_cache_local_frames_to_their_core_in_greedy_and_poll_mode`.
- `a_cache_local_frame_behind_a_blocking_one_waits_for_its_own_core`.
- `socket_local_frames_are_stolen_only_on_their_socket_and_anywhere_frames_by_all`.
- `a_cache_local_frame_that_fell_back_to_another_core_stays_stealable_on_its_socket`.
- `upstream_pull_leaves_a_frame_bound_to_another_core_to_routing`, `upstream_pull_runs_a_frame_bound_to_its_own_core`.
- `cache_local_frames_behind_a_blocking_one_stay_on_their_core_in_both_modes`: on a real lattice of two cores, 64 `CacheLocal` frames queued behind a blocking one run only on their core, in greedy and in poll mode. Before the change, the other core ran them.

## Poll mode (8cb789d, 845d5e9, 2026-10-23)

**What.** A worker whose max batch is at most `FragmentConfig::POLL_MODE_MAX_BATCH` (2) **polls** its cache instead
of taking from it greedily.
- **A poll takes one frame at a time,** holding the frame's partition only while it takes the frame out.
- **It steals the same way.**
- **So a frame that blocks no longer holds up the unordered frames queued behind it:** idle workers steal them.

**Why.** The two modes exist for two kinds of work:
- **Greedy mode is for short frames.** A greedy take holds a partition while it runs the frames it takes, so the consumer consumes them as fast as it can without touching the producer side. If one of those frames blocks, though, it holds up every frame queued behind it in its partition, and work stealing passes the held partition over.
- **Poll mode is for lattices that run slow or blocking work.** Each frame costs the partition hold once, and in exchange no partition is held while a frame runs.

**How to use it.** Give the lattice's fragments a max batch of 2 or less:
```rust
let fragment = FragmentConfig::default()
    .with_max_batch_size(NonZeroU64::new(FragmentConfig::POLL_MODE_MAX_BATCH).unwrap());
assert!(fragment.polls());
let config = LatticeConfig::new(Arc::new(BaseCloneableObject::new(fragment, DefaultExecutor::new())));
```
- **`FragmentConfig::polls()`** says whether a config's workers poll. The fragment's cycle uses it to choose `poll` or `greedy` for its own cache, and `work_steal_polled` or `work_steal` for stealing.
- **A process can run lattices in both modes,** for example one greedy lattice for short frames and one polling lattice for blocking frames, each on its own CPUs (`LatticeConfig::allowed_cpus`).

**Semantics:**
- **`ControlPlaneCache::poll(consume, limit)`** moves up to `limit` frames into `consume` one at a time. While `consume` runs a frame, that frame's partition is free.
- **`poll_while(consume, stop, limit)`** does the same, but leaves each partition at its first frame that `stop` matches. Its walk starts one partition further on each call, so a busy partition does not starve the others.
- **`work_steal_polled`** is `work_steal` over `poll_while`: a stolen frame that blocks does not hold up the victim's partition.
- **Ordered frames keep their order with the hold released:**
  - only the cache's owner takes them;
  - stealers stop at an ordered head, and an SMT buddy takes from its own cache;
  - the owner runs them one after another.
  - So an ordered stream runs on its owning CPU, one frame at a time and in sequence, while unordered frames are stolen around it.
- **The routing policy limits polls and stolen polls** just as it limits greedy takes (see the entry above).

**Tests:**
- **`config/fragment_config.rs`:**
  - `workers_poll_at_a_max_batch_of_two_or_less`.
- **`control_plane/cache/tests.rs`:**
  - `a_poll_frees_the_partition_while_its_frame_runs`, `a_greedy_take_holds_the_partition_while_its_frames_run`;
  - `polls_take_one_frame_at_a_time_in_order_and_leave_ordered_heads_to_the_owner`;
  - `polls_walk_every_partition`.
- **`impls/base_cloneable/tests.rs`**, on a real SMT core:
  - `a_polling_lattice_runs_the_frames_queued_behind_one_that_blocks`: in poll mode the 512 unordered frames queued behind a 2 s blocker all run within a second. In greedy mode, 21 of them wait for it.
  - `a_polling_lattice_runs_ordered_frames_on_their_own_cpu_in_sequence_while_others_steal` (845d5e9): two ordered streams, one per SMT thread. Each runs only on its own CPU, one frame at a time and in sequence, while both workers steal the 20 µs unordered frames queued between them. Stealers take only unordered work.
