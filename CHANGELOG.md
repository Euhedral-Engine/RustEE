# Changelog

Changes to the lattice's execution model, newest first. Each entry says what changed, why, how to use it, its exact
semantics and the tests that pin it.

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
- `Lake` (G7-G9): the attached set is a `Published` reference in the lattice's epoch domain; `offer` and `admit` read
  it with one pinned load. Attach and detach run under a lifecycle lock off the hot scope. The idle watch's downstream
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
- `Published<T>`, `EpochDomain`, `EpochSlot`, `EpochGuard` and `Retired<T>` (`euhedral-data-structures::sync`): an
  atomically replaceable `Option<Arc<T>>` read with an acquire load under an epoch guard, written by lifecycle
  code, and reclaimed once every reader has passed the epoch that replaced the value. This stands in for the
  collector wherever Java publishes a reference and later replaces it while readers may still hold the old one.
- `Admission` (`euhedral-data-structures::sync`): Java's routing-state reader count, with the retired bit as the sign
  bit.
- Workers announce a quiescent point once per cycle and go idle before they park. `UpstreamRegistry::epochs`,
  `thread_epoch` and `with_pinned` expose the lattice's domain.
- `ControlPlaneCache` partitions (audit C4) are `ClaimCell`s, Java's `pLocks`: a consumer that finds a partition held
  passes it over, as before, with no mutex.
- `CacheDirectory` slots (C5) are `Published` weak references: a steal reads one with no lock under the thread's pin.
- `LatticeVertex` routing (C6, C7): the published `RoutingState` is an `AtomicPtr`; `try_push` takes a count on it
  with a compare-and-set, re-checks the pointer and releases, in place of an `RwLock` read and two reference-count
  updates. `routing_state()` returns a `RouteGuard` (it was an `Arc<RoutingState>`): holding the guard keeps the
  generation's downstreams open. The book is a lifecycle lock, taken by the thread whose release retires a
  generation, off the hot scope, and releases a generation once: the check is made under the lock, since a recycled
  state could otherwise be released again in its next life by a thread that decided in its last.
- `LatticeEdge` (C8, C9): the downstream is a `SetOnce` and the parent a `Published` weak reference; neither read
  takes a lock. A receiver added to a closed edge is told `on_complete`, once, whether the edge took it or refused it. `request` and `pull` pin once for the whole climb through the parents.
- `HOT_FIX_TAGS` loses C4 to C9 and lists 39.

**Why.** Every frame a source pushes through the lattice crossed two vertex routing locks and two edge downstream
locks, each with a reference-count update on a line every worker shares; every steal attempt took a directory lock and
every cache take a partition mutex. Java has none of them: it takes a counter on the routing state, reads the
downstream with an opaque load and the parent with an acquire load, and claims a partition with a compare-and-set.

**How.**
- Hold a routing generation with `let state = vertex.routing_state();` and drop the guard to let go.
- A thread that reads published values outside a worker's cycle goes through
  `registry.with_pinned(|guard| cell.load(guard))`.

**Semantics.**
- Routing states are never freed before their vertex drops. A remap rewrites a state for the next generation once it
  is retired, has no readers and has released its handles; a reader that counted against a state just before a
  rewrite either finds it published (and routes with it) or gives the count back. The vertex keeps as many states
  as there are generations still being read, plus one.
- The new generation is published before the old one is retired. Java retires first, so a reader that loads the
  retired state waits for the writer; here no reader waits.
- The downstream of an edge stays allocated until the edge drops. A cache stays published until it closes, and a
  stealer that found one holds it for the steal, so a closed cache lives until its last stealer lets go.
- Readers of a `Published` cell pay no barrier instruction where the process can issue heavy barriers (Linux `membarrier` registered once at domain creation, Windows `FlushProcessWriteBuffers`): a thread that leaves idle stores its announcement with release ordering and a compiler fence, and a writer forces a barrier on every running thread before it scans the announcements (userspace RCU's "memb" flavour). Elsewhere, and where registration fails, readers keep the sequentially consistent store and fence: the OS offers no equivalent. Under loom and Miri the symmetric form is the one modelled, so the asymmetric path is exercised only by the real-thread stress tests on x86_64 (CI has no aarch64 job).
- The writer issues a heavy barrier only when no earlier one covers the retirement it checks (`EpochDomain` remembers the newest epoch a barrier covered): a `defer`, `try_reclaim`, `Retired::is_free` or `wait` iteration, and the cells of one domain, share it; a `wait` issues at most one. `EpochDomain::heavy_barriers()` counts them. A barrier that fails after registration succeeded does not abort: nothing is reclaimed (retired values wait for a barrier that works), and the failure is printed once.
- A `Published` store never waits and never fails; the retirements a cell cannot free yet chain on a list, which
  each store, deferral and `try_reclaim` sweeps. A worker that parks, or whose loop ends by a panic, holds nothing
  back. While an `EpochGuard` is alive its slot keeps its announcement: a quiescent point or an idle requested
  meanwhile takes effect when the last guard drops, so a reference loaded under a guard stays valid as long as the
  guard.

**Tests.** `a_value_is_never_dropped_while_a_reader_holds_it`, `wait_never_returns_before_every_slot_passed_the_epoch`,
`a_store_never_waits_and_the_retirements_chain_until_the_readers_pass`, `readers_never_see_a_freed_value`,
`loom_reader_and_writer_never_see_a_freed_value`, `loom_stale_reader_and_recycler`,
`a_parked_worker_does_not_hold_back_a_retirement`, `stealers_racing_a_greedy_owner_never_take_an_ordered_frame`,
`a_stealer_skips_a_claimed_partition_without_waiting`, `a_cache_closed_while_stolen_from_lives_for_the_steal_and_its_slot_reference_is_reclaimed`,
`pushes_racing_remaps_reach_only_validated_downstreams`,
`the_last_reader_of_a_retired_generation_closes_its_dropped_downstreams_once`, `a_remap_reuses_a_retired_state_once_its_readers_left`,
`a_panicking_routing_function_gives_its_count_back`, `close_while_pushing_finishes_or_delivers_every_frame`,
`a_closed_edge_refuses_a_new_downstream`, `attaches_racing_a_close_tell_the_receiver_exactly_once`,
`set_parent_with_the_same_parent_stores_nothing`. The concurrency tests run under Miri with many seeds (the `miri` CI
job) and the loom models under `--cfg loom`.

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

**Why.** MySphere's players run on Windows too, and `AsyncIo` was Linux-only.

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
`Join`, credit-bounded admission, a seek and a pause, which fails if a frame callback runs over 200 µs without the OS
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
