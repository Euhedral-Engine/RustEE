# Lock-free hot path Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** No lock on any per-frame, per-pull, per-request, per-offer, per-completion or per-tick path of
`euhedral-core`, `euhedral-data-structures`, `euhedral-hardware-utils`, `euhedral-graph`, `euhedral-io` and
`euhedral-memory`; locks remain only for start, stop and rebalance.

**Architecture:**
- Every place mirrors its Java counterpart's fields, atomics and orderings (audit principle 7, §3.12).
- Source state uses the lane-source plan's `SourceCell` and `DownstreamSlot`. A source's lifecycle calls (`complete`,
  `clear`, kill) set atomics, and the source applies them in its next `request` or `pull`, or in `Drop`, as Java's
  (lane-source Part 1 as merged, PR #11 at `29c97af`). No source uses `serialize`; only the interceptor keeps it, for
  its own finish.
- This plan adds four primitives to `euhedral-data-structures::sync`: `ClaimCell` (Java's try-claims), `Published` (the
  stand-in for Java's garbage collector), `OneShot` (`CompletableFuture`) and `TakeAllStack` (`StageFrame.externals`).
- A clippy configuration, a debug-only hot-scope check and a ratchet test keep new locks out.

**Tech Stack:**
- Rust 1.99.1, edition 2024 (`mise.toml`).
- The workspace crates. Nothing new at run time.
- `loom` 0.7 as a `cfg(loom)` dev-dependency (approved).
- Miri on a nightly toolchain for the new primitives' unit tests (approved).

**Spec:** `docs/superpowers/specs/2026-10-28-lock-free-hot-path-audit.md` (this branch). The finding ids (C1, G6, I4,
S1 ...) and section numbers below are the spec's. Read §2.0, §3 and §3.12 before any task. Read the Java counterpart
the task names before writing its code.

**Java references** (read-only):
- core places (`euhedral-core`, `euhedral-data-structures`, `euhedral-hardware-utils`): the upstream Java runtime's
  `euhedral-core/src/main/java/.../core/`, `euhedral-data-structures/...` and `euhedral-hardware-utils/...` modules;
- graph places (`euhedral-graph`, `euhedral-io`, `euhedral-memory`): the Java runtime `euhedral-graph` derives from,
  `core/src/main/java/.../core/runtime/graph/` and `.../core/gpu/`.

The coordinator gives the checkout paths.

## Global Constraints

- **Hot path lock-free.** "The hot path must be lock-free. There should never be locking for anything other than life
  cycle things like start, stop, and rebalance."
- **Mirror Java.** Deviate only for a genuine Java bug (cite the Java file and method) or a missing Rust equivalent;
  every deviation is the one the spec lists in §3.12, or the PR adds it there.
- **Single consumer.** The lattice serializes a source's `request` and `pull`. `request` is always called and pushes
  all ready work within the banked demand. `pull` may never be called and never banks demand.
- **Selective admission:** no "has work" flags and no wake paths beyond Java's own.
- **Long calls** run on the worker that needs them. There is no blocking-work facility and no thread of ours for them.
- **Layering:** `euhedral-data-structures`, `euhedral-hashing` → `euhedral-hardware-utils` (which uses the lifecycle
  locks) → `euhedral-core` → `euhedral-graph`, `euhedral-io` → `euhedral-memory` → `euhedral-gpu`. No upward
  dependency, dev-dependency or feature; `euhedral-gpu/tests/layering.rs` checks it.
- **Unsafe:**
  - every `unsafe` block has a `// SAFETY:` comment;
  - `euhedral-graph` stays `#![deny(unsafe_code)]`, and a module that needs `unsafe` carries a reasoned module-level
    `allow`;
  - Miri runs on every new primitive's unit tests.
- **Wording:** code, docs, comments, READMEs, CHANGELOG and GLOSSARY name no other repository and no application
  domain, and make no negative self-descriptions. Commit messages may name repositories.
- **Every commit** passes `mise exec -- cargo fmt --all --check` and `mise exec -- cargo clippy --workspace
  --all-targets -- -D warnings`.
- **Every PR** passes:
  - `mise exec -- cargo test --workspace`;
  - its loom models: `RUSTFLAGS="--cfg loom" mise exec -- cargo test -p <crate> --release --lib <module>::loom`;
  - Miri: `mise exec -- cargo +nightly miri test -p euhedral-data-structures sync::`;
  - for `euhedral-io`, the Windows workflow.
- **Build directory.** `CARGO_TARGET_DIR` is set to the machine's build directory. No scratch in `/tmp`.
- **Docs.** Each chunk ends with a CHANGELOG entry (What, Why, How, Tests), GLOSSARY terms for new public concepts, and
  crate README rows.
- **PRs.** One draft PR per chunk, stacked on the newest open PR: `gh pr create --draft --base <that PR's branch>`
  (`main` once the stack below it has merged). No agent marks a PR ready. Commits end with the session line the
  coordinator gives.
- **Ratchet.** Each PR deletes from `HOT_FIX_IDS` in `euhedral-core/tests/hot_fix_ratchet.rs` exactly the ids whose
  tags it removes.

## Ordering with the lane-source plan

| This plan | Waits for | Why |
|---|---|---|
| Chunk 1 (and so every chunk) | lane-source Parts 1-4 in the stack it is based on (PRs #11-#15; chunk 1 is #16 on `lane-source-gpu`) | Chunk 1 edits `euhedral-data-structures/src/sync/mod.rs`, `src/lib.rs`, `Cargo.toml` and `euhedral-core/src/flow_control/vertex.rs` (Part 1), and tags sites in `euhedral-graph` after `join.rs` lost its `Mutex` (Part 2). |
| Chunks 2-11 | chunk 1, then the dependencies below | — |
| Every chunk's docs task | no open lane-source PR editing `CHANGELOG.md`, `GLOSSARY.md` or the same crate README | Parts 3 and 4 edit those files. |

- Chunks stack on the newest open PR, so the lane-source files (`euhedral-graph/src/lanes.rs`, `src/lane_source/`,
  `src/testing.rs`, `src/lib.rs`, `tests/lane_*.rs`, `euhedral-gpu/`) are already in a chunk's base and a chunk may
  edit them. A chunk that rebases onto a merge re-checks them.
- Chunk dependencies:
  - chunk 4 needs chunk 2 (`ClaimCell`);
  - chunk 7 needs chunk 2 (the `AsyncIo` front);
  - chunk 8 needs chunk 7 (shared slots and the refusal API);
  - chunk 10 needs chunk 4 (`Published`);
  - chunk 5 needs chunk 4 (`Published`, for the lake's generation);
  - chunks 3, 6 and 9 need only chunk 1;
  - chunk 11 needs all.
- Chunks that may run in parallel rebase the ratchet constant when the other lands.

## Review Focus

1. **A lifecycle call must never be stranded by a worker's release.** A `clear` or kill racing a pulling worker
   discards every queued frame exactly once. Test `clear_racing_a_pull_discards_every_frame_once` (Task 2.4).
2. **A stale routing reader must never use a recycled state's old contents.** Under 8 pushers and a remapping thread,
   every pushed frame goes to a downstream of the generation its reader validated. Test
   `pushes_racing_remaps_reach_only_validated_downstreams` and the loom model `loom_stale_reader_and_recycler`
   (Task 4.5).
3. **A refused I/O submission must leave nothing behind.** No completion ever follows a refused `submit`, and its
   buffer comes back to the caller. Test `a_refused_submission_hands_the_request_back_and_completes_nothing`
   (Task 7.3).
4. **A sequencer must release in order under any completion order.** Slots completed in reverse from 4 threads are
   emitted in registration order, each once. Test `registrations_from_4_threads_complete_in_reverse_and_emit_in_order`
   (Task 5.4).
5. **A dispatcher closed while values are offered must neither lose the last value nor deliver after close.** Test
   `offers_racing_close_deliver_the_latest_or_nothing_after_close` (Task 10.7).

## File map

| File | Chunk | Responsibility |
|---|---|---|
| `clippy.toml` (new, workspace root), `euhedral-graph/clippy.toml`, `euhedral-io/clippy.toml` | 1 | disallowed lock types and blocking calls |
| `euhedral-data-structures/src/sync/lifecycle.rs` | 1 | `LifecycleMutex`, `LifecycleRwLock`, `HotScope` |
| `euhedral-core/tests/hot_fix_ratchet.rs` | 1, every PR | counts `HOT-FIX(<id>)` tags |
| `euhedral-data-structures/src/sync/claim_cell.rs` | 2 | `ClaimCell` |
| `euhedral-core/src/ingest/{downstream.rs,queue.rs,single_use.rs}` | 2 | C11-C13 |
| `euhedral-graph/src/completions.rs` | 2 | G1-G3 |
| `euhedral-io/src/lib.rs` (`Front`) | 2, 7 | I1, I2, the `pending` gate |
| `euhedral-core/src/flow_control/hot_source.rs`, `utils/stop_watch.rs`, `control_plane/{fragment.rs,decision_tree.rs}` | 3 | C1, C2, C3, C10 |
| `euhedral-data-structures/src/sync/published.rs`, `euhedral-core/src/flow_control/registry.rs` | 4 | `Published`, epoch slots |
| `euhedral-core/src/control_plane/cache.rs`, `flow_control/{vertex.rs,edge.rs}` | 4 | C4-C9 |
| `euhedral-data-structures/src/sync/one_shot.rs` | 5 | `OneShot` |
| `euhedral-core/src/ingest/pipeline_runner.rs`, `euhedral-graph/src/{latch.rs,sequencer.rs,lake.rs}` | 5 | C14, G5, G6, G7 (G8, G9 lifecycle) |
| `euhedral-data-structures/src/sync/take_all.rs`, `euhedral-graph/src/stage_graph.rs` | 6 | `TakeAllStack`, G10-G15 |
| `euhedral-io/src/uring.rs` | 7 | I3-I6, S6a |
| `euhedral-io/src/slots.rs` (new) | 7, 8 | tag-indexed operation slots shared by both engines |
| `euhedral-io/src/win/{mod.rs,real.rs,sim.rs,sys.rs,tests.rs}` | 8 | I7-I15 |
| `euhedral-memory/src/{backend.rs,cuda/backends.rs,planner.rs,tiers.rs}`, `src/cuda/pinned_pool.rs` (new) | 9 | M1, S1, S2 |
| `euhedral-core/src/internal/task_pool.rs`, `control_plane/{lattice.rs,shard.rs}` | 10 | C16-C20 |
| `euhedral-hardware-utils/src/monitor/{mod.rs,dispatcher.rs}`, `topology/mapper.rs` | 10 | H1-H7 |

---

## Chunk 1 (draft PR, branch `hot-path-1-guards`): guard rails, no behaviour change

### Task 1.1: `LifecycleMutex`, `LifecycleRwLock` and `HotScope`

**Files:**
- Create: `euhedral-data-structures/src/sync/lifecycle.rs`
- Modify: `euhedral-data-structures/src/sync/mod.rs` (`pub use lifecycle::{...}`)

**Interfaces:**
- Produces:

```rust
pub struct LifecycleMutex<T>(std::sync::Mutex<T>);
impl<T> LifecycleMutex<T> {
    pub const fn new(value: T) -> Self;
    /// Debug: prints "lifecycle lock taken inside a hot scope, at <location>" and aborts when `HotScope::active()`
    /// (a panic would be caught by the frame runners); `HotScope::set_violation_hook` replaces the abort in tests.
    /// Poison is ignored.
    pub fn lock(&self) -> std::sync::MutexGuard<'_, T>;
    pub fn get_mut(&mut self) -> &mut T;
    pub fn into_inner(self) -> T;
}
pub struct LifecycleRwLock<T>(std::sync::RwLock<T>); // read() / write(): the same check
pub struct HotScope; // thread-local depth counter; every method is a no-op in release builds
impl HotScope {
    pub fn enter() -> HotGuard;   // depth += 1; the guard's drop: depth -= 1
    pub fn leave() -> ColdGuard;  // saves the depth and sets 0; the guard's drop restores it
    pub fn active() -> bool;      // depth > 0 (always false in release)
}
```

- [ ] **Step 1: Write the failing tests** in `lifecycle.rs`.
  - `a_lifecycle_lock_outside_a_hot_scope_locks`.
  - `#[cfg(debug_assertions)] a_lifecycle_lock_inside_a_hot_scope_reaches_the_hook_and_then_locks` (a recorder installed
    with `HotScope::set_violation_hook`), and `the_default_report_aborts_even_inside_catch_unwind` (a child process).
  - `leaving_the_hot_scope_allows_the_lock_and_restores_the_depth`: enter, enter, `leave` → `!active()`, lock;
    drop → `active()`.
  - `a_poisoned_lifecycle_lock_still_locks`.
- [ ] **Step 2: Run them and see them fail.** `mise exec -- cargo test -p euhedral-data-structures sync::lifecycle`.
  Expected: FAIL (no module).
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run them.** PASS in debug; `--release` PASS with the hook tests compiled out.
- [ ] **Step 5: Commit.** `git commit -m "euhedral-data-structures: LifecycleMutex, LifecycleRwLock and HotScope"`

### Task 1.2: Hot scopes on the worker and producer paths

**Files:**
- Modify:
  - `euhedral-core/src/control_plane/fragment.rs` (`run_cycle`);
  - `flow_control/upstream_queue.rs` (`pull`, `request`);
  - `flow_control/vertex.rs` (`try_push`);
  - `control_plane/cache.rs` (`push`, `greedy_while`, `poll_while`, `work_steal`, `work_steal_polled`);
  - `ingest/queue.rs` (`offer*`);
  - `euhedral-graph/src/{completions.rs,join.rs,sequencer.rs,stage_graph.rs}` (`add`, `arrive`/`abandon`,
    `register`/`finish`, `Run::{arrive, run_stage, submitted}`);
  - `euhedral-io/src/lib.rs` (`submit*`).
- Test: `euhedral-core/src/control_plane/fragment/tests.rs`, `euhedral-graph/tests/lake.rs`

**Interfaces:**
- Consumes: `HotScope::enter`, `HotScope::leave` (1.1).
- Lifecycle steps reached from inside a frame take `let _cold = HotScope::leave();` first:
  - `QueueIngestSink::complete`;
  - `Lake`'s first attach;
  - `ControlPlaneLattice::add_upstream`;
  - the C7 retirement in `RoutingState`'s drop.

- [ ] **Step 1: Write the failing test** `a_worker_cycle_runs_inside_a_hot_scope`: a frame records
  `HotScope::active()`; run one cycle; assert `true`. And `a_frame_that_completes_its_sink_does_not_panic` (graph
  `tests/lake.rs`): a frame calls `complete_gracefully` on the lake that fed it; debug run passes.
- [ ] **Step 2: Run them and see the first fail.** `mise exec -- cargo test -p euhedral-core
  a_worker_cycle_runs_inside_a_hot_scope`.
- [ ] **Step 3: Add the guards** at the listed entry points.
- [ ] **Step 4: Run** `mise exec -- cargo test --workspace` in debug. PASS. Lifecycle locks still use `std` types,
  so nothing checks yet; this step proves the scopes break nothing.
- [ ] **Step 5: Commit.** `git commit -m "Hot scopes on worker and producer paths"`

### Task 1.3: Lifecycle locks, the clippy configuration and the `HOT-FIX` tags

**Files:**
- Create: `clippy.toml`.
- Modify: `euhedral-graph/clippy.toml`, `euhedral-io/clippy.toml` (add the same entries), and every lock site of
  spec §2.2-§2.4.

**Interfaces:**
- `clippy.toml`:
  - `disallowed-types`: `std::sync::Mutex`, `std::sync::RwLock`, `std::sync::Condvar`, `std::sync::Barrier`,
    `std::sync::mpsc::Sender`, `std::sync::mpsc::SyncSender`, `std::sync::mpsc::Receiver`;
  - `disallowed-methods`: `std::thread::park`, `std::thread::park_timeout`, `std::thread::sleep`,
    `std::thread::yield_now`, `std::hint::spin_loop`.
- Each `allow` carries one reason:
  - **LIFECYCLE** rows (C21-C34, D2, H8-H11, G8, G9, I16, M2, M3): switch `Mutex`/`RwLock` to
    `LifecycleMutex`/`LifecycleRwLock`. A remaining `Condvar`, park or sleep gets `#[allow(clippy::disallowed_*,
    reason = "LIFECYCLE(<id>)")]`.
  - **HOT and MIXED** rows: `#[allow(clippy::disallowed_types, reason = "HOT-FIX(<id>)")]` at the field, and at each
    use the lint reports. There is one tag id per finding: C1-C14, C16-C20, G1-G3, G5-G7, G10-G15, I1-I15, H1-H7. M1
    has no lock type, so it gets a `// HOT-FIX(M1)` comment at `euhedral-memory/src/backend.rs:95`.
  - **Source-contract markers**, as `// HOT-FIX(<id>)` comments:
    - `S1` at `planner.rs:1266`;
    - `S2` at `tiers.rs:1119`;
    - `S6a` at `uring.rs:66`.
  - **Java's own back-offs:** `utils/spin_wait.rs` (C15's back-off), the queues' CAS loops and the idle policy's
    parks get `reason = "JAVA-BACKOFF"`.
- Distinct `HOT-FIX` ids: 54 lock findings (M1 among them) + S1 + S2 + S6a = 57.

- [ ] **Step 1: Add `clippy.toml`.** Run `mise exec -- cargo clippy --workspace --all-targets -- -D warnings`. It fails
  at every lock site: that list is the work.
- [ ] **Step 2: Convert the LIFECYCLE rows** to `LifecycleMutex`/`LifecycleRwLock`. Run `mise exec -- cargo test
  --workspace` in debug. PASS. A failure here is a lifecycle lock reached inside a hot scope: add a `HotScope::leave`
  at the lifecycle step (Task 1.2's list), never at the lock.
- [ ] **Step 3: Tag the HOT and MIXED sites and the markers** as above. Clippy: clean.
- [ ] **Step 4: Commit.** `git commit -m "Lifecycle locks and the disallowed-lock lint; tag every hot lock HOT-FIX(<id>)"`

### Task 1.4: The ratchet test, docs and the draft PR

**Files:**
- Create: `euhedral-core/tests/hot_fix_ratchet.rs`
- Modify: `CHANGELOG.md`, `GLOSSARY.md` (**hot scope**, **lifecycle lock**, **HOT-FIX tag**),
  `euhedral-data-structures/README.md`

**Interfaces:**
- `const HOT_FIX_IDS: &[&str]`: the 57 ids. Each PR deletes the ids whose tags it removes.
- `fn tags() -> BTreeSet<String>`:
  - walks `<workspace>/*/src/**/*.rs`, the workspace being `CARGO_MANIFEST_DIR/..`;
  - skips `tests.rs`, `testing.rs` and any `tests/` directory;
  - collects the distinct ids of `HOT-FIX(<id>)`.

- [ ] **Step 1: Write the tests.**
  - `the_hot_fix_tags_are_exactly_the_listed_ids`: `tags()` equals `HOT_FIX_IDS`.
  - `every_listed_id_is_a_finding_of_the_audit`: each id is a row of the audit's tables.
  - `every_disallowed_allow_is_explained_and_no_alias_hides_a_lock`: every `allow` of a disallowed type or call in the
    library sources carries a `HOT-FIX(`, `LIFECYCLE(` or `JAVA-BACKOFF` reason (or is a test-only `cfg_attr`, or a
    file-level allow of an exempt file), and no `type` alias names a lock type.
- [ ] **Step 2: Run.** `mise exec -- cargo test -p euhedral-core --test hot_fix_ratchet`. PASS at 57.
- [ ] **Step 3: Write the docs and run the gate** (Global Constraints).
- [ ] **Step 4: Commit, push, open the draft PR.** `git commit -m "Ratchet test for hot-path lock tags"`; `gh pr create
  --draft --base <the newest open PR's branch> --title "Hot-path guard rails: lifecycle locks, hot scopes, lint, ratchet"`.

---

## Chunk 2 (draft PR, `hot-path-2-sources`): ingest, `Completions`, the `AsyncIo` front (C11-C13, G1-G3, I1, I2)

### Task 2.1: `ClaimCell<T>`

**Files:**
- Create: `euhedral-data-structures/src/sync/claim_cell.rs` (unit tests, a `#[cfg(loom)] mod loom`)
- Modify: `euhedral-data-structures/src/sync/mod.rs`

**Interfaces:**
- Produces (spec §3.0 item 1):

```rust
pub struct ClaimCell<T> { claimed: AtomicBool, value: UnsafeCell<T> }
impl<T> ClaimCell<T> {
    pub const fn new(value: T) -> Self;
    /// compare_exchange(false, true, SeqCst, SeqCst); None when another thread holds it. Never waits.
    pub fn try_claim(&self) -> Option<ClaimGuard<'_, T>>;
    pub fn is_claimed(&self) -> bool; // Relaxed
    pub fn get_mut(&mut self) -> &mut T;
    pub fn into_inner(self) -> T;
}
pub struct ClaimGuard<'a, T>; // Deref/DerefMut; Drop: store(false, Release)
impl<T> ClaimGuard<'_, T> { pub fn release_seq_cst(self); } // store(false, SeqCst)
unsafe impl<T: Send> Sync for ClaimCell<T> {}
```

- [ ] **Step 1: Write the failing tests.**
  - `a_second_claim_fails_while_the_first_is_held`.
  - `a_released_claim_can_be_taken_again`, for both release paths.
  - `claims_from_8_threads_never_overlap`: each holder sets and clears an `inside` flag and asserts it was clear;
    8 × 10 000 attempts.
  - `writes_under_one_claim_are_seen_by_the_next`: 4 threads increment through claims until each has done 1000;
    the total is 4000.
  - `#[cfg(loom)] loom_two_claimers_exclude_each_other_and_see_prior_writes`.
- [ ] **Step 2: Run them and see them fail.** `mise exec -- cargo test -p euhedral-data-structures sync::claim_cell`.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run them, the loom model and Miri.** PASS.
- [ ] **Step 5: Commit.** `git commit -m "euhedral-data-structures: ClaimCell, Java's try-claim over plain state"`

### Task 2.2: Ingest receiver and demand (C11)

**Files:**
- Modify: `euhedral-core/src/ingest/downstream.rs`, `ingest/queue.rs`, `ingest/single_use.rs`
- Test: `euhedral-core/src/ingest/queue/tests.rs`

**Interfaces:**
- Consumes: `DownstreamSlot` (lane-source Part 1: `attach`, `receiver`, `complete`, `is_complete`; it has no
  `serialize`, and every method touches atomics only).
- `Downstream` is replaced by `DownstreamSlot` plus `demand: CachePadded<AtomicU64>`.
  - `fn add_demand(&self, d: i64) -> u64` is Java's `accumulate`, a `fetch_update(SeqCst, SeqCst)`: saturating, and
    a negative step floors at 0.
- The guards of Java `AbstractIngestSink.Delegate.pull`/`request`: return at once when the source is complete, has no
  receiver, or the demand is 0.

- [ ] **Step 1: Write the failing tests.**
  - `request_without_a_receiver_banks_nothing`.
  - `demand_saturates_and_floors_at_zero`: `add_demand(u64::MAX as i64)` twice gives `u64::MAX`; `-(huge)` gives 0.
  - `complete_before_attach_tells_the_late_receiver_once`.
- [ ] **Step 2: Run them and see them fail.** `mise exec -- cargo test -p euhedral-core ingest::queue`.
- [ ] **Step 3: Implement.** Delete the `HOT-FIX(C11)` tags. Lower `HOT_FIX_TAGS` by 1.
- [ ] **Step 4: Run** `mise exec -- cargo test -p euhedral-core`. PASS.
- [ ] **Step 5: Commit.** `git commit -m "Ingest: DownstreamSlot and Java's demand accumulate (C11)"`

### Task 2.3: The queue consumer under Java's `draining` claim (C12)

**Files:**
- Modify: `euhedral-core/src/ingest/queue.rs` (`QueueSource`)
- Test: `euhedral-core/src/ingest/queue/tests.rs`

**Interfaces:**
- Consumes: `ClaimCell` (2.1).
- `consumer: ClaimCell<PartitionedConsumer<Box<dyn Frame>, Single>>` replaces `Mutex<Consumer>` and `try_hold`.
- `request_pending: AtomicBool` stays. All its accesses become `SeqCst`, as Java's volatile `requestPending`.
- Paths, each as the Java method in parentheses (spec §3.1):
  - `pull` (`hookOnPull`);
  - `request` (`hookOnRequest`);
  - `drain_requests` (`drainRequests`): subtract before releasing; release with `release_seq_cst`;
  - `discard_queued` (`discardQueued`);
  - `publish` (`afterPublish`: `fence(SeqCst)` after the offer, then `on_published_after_close` when closed).
- Lifecycle calls set atomics, then try the claim once: `complete` sets `publication_closed` and completes the slot;
  kill sets the kill switch (`PipelineRunner`). The holder applies them in `on_drain_finished`, and the next
  `request`/`pull` otherwise.
- `impl Drop for QueueSource` finishes every frame still queued (`&mut self`, through `ClaimCell::get_mut`).

- [ ] **Step 1: Write the failing tests.**
  - `a_request_while_a_pull_holds_the_queue_is_served_when_the_pull_releases`: thread A pulls with a consumer that
    blocks on a barrier; thread B `request(4)` returns at once; A releases; 4 queued frames reach the receiver.
  - `request_pushes_no_more_than_the_banked_demand`: 10 frames, `request(3)` delivers 3 and leaves demand 0;
    `request(10)` delivers 7 and leaves demand 3.
  - `a_source_driven_only_by_request_delivers_every_frame`: 1000 frames offered from 4 threads while a driver calls
    only `request(1)` in a loop; all 1000 arrive, each once.
  - `a_publish_after_close_is_discarded_by_someone`: close the sink, offer from 4 threads racing a pulling thread;
    every frame is finished exactly once (drop counter).
  - `frames_queued_when_the_source_drops_are_finished`: queue 10, complete, drop the last `Arc`; all 10 finished.
- [ ] **Step 2: Run them and see them fail.** The first fails today: the `Mutex` blocks.
- [ ] **Step 3: Implement.** Delete `HOT-FIX(C12)`; lower the ratchet by 1.
- [ ] **Step 4: Run** `mise exec -- cargo test -p euhedral-core ingest` and `mise exec -- cargo test -p
  euhedral-core pipeline_runner`. PASS.
- [ ] **Step 5: Commit.** `git commit -m "Ingest: the queue consumer under Java's draining claim (C12)"`

### Task 2.4: `clear` claims (C12's Java-bug deviation)

**Files:** `euhedral-core/src/ingest/queue.rs`; test in `ingest/queue/tests.rs`.

**Interfaces:**
- New field `clear_requested: AtomicBool`.
- `clear()` sets it (`SeqCst`), then runs `discard_queued(recycle)`.
- `on_drain_finished` runs `discard_queued(recycle)` while `clear_requested.swap(false, SeqCst)`.

- [ ] **Step 1: Write the failing test** `clear_racing_a_pull_discards_every_frame_once`: 1000 frames; one thread pulls
  slowly (1 per call); another calls `clear()`. Every frame is either delivered or recycled, exactly once, and none
  remains queued after both return.
- [ ] **Step 2: Run it and see it fail.** Today `clear` blocks on the lock.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run it.** PASS. Add the C12 `clear` row to the spec's §3.12 only if the wording changed.
- [ ] **Step 5: Commit.** `git commit -m "Ingest: clear takes the draining claim (Java's clear drains unclaimed)"`

### Task 2.5: `SingleUseSource` (C13)

**Files:** `euhedral-core/src/ingest/single_use.rs` (module-level `#![allow(unsafe_code, reason = "the CAS winner owns the frame")]` if the crate denies it).

**Interfaces:**
- `taken: AtomicBool`, `frame: UnsafeCell<Option<Box<dyn Frame>>>`.
- `pull`, `request` and `complete` each `compare_exchange(false, true, SeqCst, SeqCst)`. The winner takes the frame:
  `pull` and `request` deliver it; `complete` cancels it (`do_finally_with_error(FrameError::cancelled())`).

- [ ] **Step 1: Write the failing tests.**
  - `a_frame_is_delivered_or_cancelled_exactly_once`: 10 000 rounds of `request(1)` racing `complete()`.
  - `a_completed_source_delivers_nothing_to_a_later_pull`.
- [ ] **Step 2: Run them and see them fail.**
- [ ] **Step 3: Implement.** Delete `HOT-FIX(C13)`; ratchet -1.
- [ ] **Step 4: Run** `mise exec -- cargo test -p euhedral-core ingest`. PASS.
- [ ] **Step 5: Commit.** `git commit -m "SingleUseSource: Java's COMPLETE CAS takes the frame (C13)"`

### Task 2.6: `Completions` (G1, G2, G3)

**Files:** `euhedral-graph/src/completions.rs`; tests in the module.

**Interfaces:**
- Consumes: `ClaimCell`, `DownstreamSlot`.
- `poll: ClaimCell<Poll>`, claimed by `emit` and by `add` when it finds the source complete (the existing sweep,
  Java's `onPublishedAfterClose`).
- A losing `emit` returns 0. A losing sweep sets `sweep_pending` (`SeqCst`). The holder re-checks it after
  `release_seq_cst` and claims again, as `drain_requests` does.
- `lanes: Arc<LanePool>` is a plain field.
- `downstream: DownstreamSlot`.
- `complete` sets `closed` (`SeqCst`) and completes the slot; it touches no poll state. The next poll, the sweep or
  `Drop` cancels what remains.

- [ ] **Step 1: Write the failing tests.**
  - `registrations_racing_complete_are_each_cancelled_or_emitted_once`: 4 threads `after_marker` while a fifth calls
    `complete()` during a `request` loop; every continuation runs exactly once.
  - `a_request_while_another_thread_polls_returns_at_once`.
- [ ] **Step 2: Run them and see them fail.**
- [ ] **Step 3: Implement.** Delete `HOT-FIX(G1)`, `(G2)`, `(G3)`; ratchet -3.
- [ ] **Step 4: Run** `mise exec -- cargo test -p euhedral-graph`. PASS.
- [ ] **Step 5: Commit.** `git commit -m "Completions: poll state under a claim, immutable lane pool, DownstreamSlot (G1-G3)"`

### Task 2.7: The `AsyncIo` front (I1, I2)

**Files:** `euhedral-io/src/lib.rs` (`Front`); tests in `euhedral-io/tests/io_owned.rs`.

**Interfaces:**
- `held: SourceCell<VecDeque<..>>`, entered only from `reap`.
- `downstream: DownstreamSlot`.
- `demand` as in Task 2.2.

- [ ] **Step 1: Write the failing test** `a_request_only_lattice_delivers_every_completion`: 500 reads from 4 threads; the
  driver calls only `request(8)`; every completion arrives once.
- [ ] **Step 2: Run it.** It may pass today; it pins behaviour across the change.
- [ ] **Step 3: Implement.** Delete `HOT-FIX(I1)`, `(I2)`; ratchet -2.
- [ ] **Step 4: Run** `mise exec -- cargo test -p euhedral-io`. PASS.
- [ ] **Step 5: Commit.** `git commit -m "AsyncIo front: SourceCell and DownstreamSlot (I1, I2)"`

### Task 2.8: Chunk 2 docs and the draft PR

- [ ] **Step 1: Write the docs.**
  - CHANGELOG "Lock-free sources and claims".
  - GLOSSARY **ClaimCell**.
  - READMEs of `euhedral-data-structures`, `euhedral-core`, `euhedral-graph`, `euhedral-io`.
- [ ] **Step 2: Run the gate.** Ratchet test PASS at 49.
- [ ] **Step 3: Push and open the draft PR** "Sources and claims: ingest, Completions, AsyncIo front".

---

## Chunk 3 (draft PR, `hot-path-3-hot-source`): the hot source and the fragment's output (C1, C2, C3, C10)

### Task 3.1: `StopWatch` and `BodyCost` as relaxed atomics (C2, C3)

**Files:** `euhedral-core/src/utils/stop_watch.rs`, `control_plane/decision_tree.rs`; tests in
`control_plane/decision_tree/tests.rs` and `stop_watch.rs`.

**Interfaces:**
- `StopWatch::{start, stop}` take `&self`. The fields `started: AtomicBool`, `ticks: AtomicU64` and `last_ns:
  AtomicU64` use `Relaxed` loads and stores only.
- `BodyCost::Window` fields become `samples: [AtomicU64; WINDOW]` (`f64::to_bits`), `expensive_threshold_ns:
  AtomicU64` and `confirmations: AtomicU32`, all `Relaxed`. `record`, `calibrate` and `reset` keep their logic.

- [ ] **Step 1: Write the failing tests.**
  - `a_shared_stop_watch_times_one_call_in_256`: through `&self`, 1024 start/stop pairs give 4 samples.
  - `body_cost_estimate_is_unchanged_by_the_atomics`: feed the existing test's sample series; the estimate equals the
    pre-change value (copy the expected number from the current test).
- [ ] **Step 2: Run them and see the first fail** (`start` takes `&mut self` today).
- [ ] **Step 3: Implement.** Delete `HOT-FIX(C2)`, `(C3)`; ratchet -2.
- [ ] **Step 4: Run** `mise exec -- cargo test -p euhedral-core control_plane`. PASS.
- [ ] **Step 5: Commit.** `git commit -m "StopWatch and body cost: Java's plain fields as relaxed atomics (C2, C3)"`

### Task 3.2: Final hooks (C1)

**Files:** `euhedral-core/src/flow_control/hot_source.rs`, `control_plane/fragment.rs` (`timed_output`).

**Interfaces:**
- `before_each(self, hook: impl Fn(&dyn Frame) + Send + Sync + 'static) -> Self`.
- `after_each(self, hook: impl Fn() + Send + Sync + 'static) -> Self`.
- Fields `Option<Box<dyn Fn ...>>`, called without a lock.
- `timed_output(body_cost: Arc<BodyCost>)` shares one `Arc<StopWatch>` between the hooks.

- [ ] **Step 1: Write the failing test** `hooks_run_around_each_push_from_one_pusher`: hooks count into atomics;
  1000 accepts; both counts are 1000 and the before-count leads at each push.
- [ ] **Step 2: Run it.** It fails to compile against `FnMut`.
- [ ] **Step 3: Implement.** Delete `HOT-FIX(C1)`; ratchet -1.
- [ ] **Step 4: Run** `mise exec -- cargo test -p euhedral-core`. PASS.
- [ ] **Step 5: Commit.** `git commit -m "LatticeHotSource: final Fn hooks, no lock (C1)"`

### Task 3.3: The downstream as Java's atomic pointer (C10)

**Files:** `euhedral-core/src/flow_control/hot_source.rs` (module-level `allow(unsafe_code)` with a reason).

**Interfaces:** spec §3.2, C10:
- `downstream: AtomicPtr<Arc<dyn LatticeReceiver>>`, `complete: AtomicBool`.
- `retired: LifecycleMutex<Vec<Box<Arc<dyn LatticeReceiver>>>>`, freed on drop.
- `add_downstream` follows Java's CAS sequence.
- `complete` swaps the pointer to null, then calls `on_complete` on the old receiver.
- `accept` with no downstream finishes the frame with an error.

- [ ] **Step 1: Write the failing tests.**
  - `complete_while_accepting_never_frees_the_receiver_in_use`: one pusher thread accepts 100 000 frames; another
    completes and re-attaches 100 times. No use-after-free under Miri (`cargo +nightly miri test -p euhedral-core
    hot_source`, a reduced count under `cfg(miri)`); every frame is delivered or finished with an error.
  - `a_complete_source_reopens_with_a_new_receiver`.
  - `complete_tells_the_old_receiver_on_complete_once`.
- [ ] **Step 2: Run them and see them fail.**
- [ ] **Step 3: Implement.** Delete `HOT-FIX(C10)`; ratchet -1.
- [ ] **Step 4: Run them, the workspace tests and Miri.** PASS.
- [ ] **Step 5: Commit.** `git commit -m "LatticeHotSource: Java's atomic downstream, retired receivers kept to drop (C10)"`

### Task 3.4: Paired benchmark, docs and the draft PR

- [ ] **Step 1: Benchmark.** Run the paired benchmark (profile, hypothesis, paired gate) of frames per second and p99
  service time on short frames, `main` against this branch, on a quiet machine, in the machine's whole-system benchmark slot. Keep the change only if it
  is no slower at p99 within the gate's noise. Record the numbers in the PR body.
- [ ] **Step 2: Write the docs** (CHANGELOG: the hooks' `Fn` change is breaking) and run the gate. Ratchet at 45
  (chunk 2 merged), or 53 rebased alone.
- [ ] **Step 3: Push and open the draft PR** "Hot source: Java's final hooks and atomic downstream".

---

## Chunk 4 (draft PR, `hot-path-4-routing`): cache, vertex, edges (C4-C9)

### Task 4.1: `Published<T>`

**Files:** `euhedral-data-structures/src/sync/published.rs` (+ loom).

**Interfaces:**
- Produces (spec §3.0 item 2):

```rust
pub struct EpochDomain;                    // global epoch + padded per-thread slots
impl EpochDomain {
    pub fn new(slots: usize) -> Arc<Self>;
    pub fn register(&self) -> EpochSlot;
}
pub struct EpochSlot;                      // one thread's slot
impl EpochSlot {
    pub fn quiescent(&self);               // load the epoch, store it to the slot (Release)
    pub fn idle(&self);
    pub fn pin(&self) -> EpochGuard<'_>;   // active at the current epoch; a fence(SeqCst)
}
pub struct Published<T> { /* AtomicPtr<T>, pending: Option<Retired<T>> */ }
impl<T: Send + Sync> Published<T> {
    pub fn new(domain: Arc<EpochDomain>, value: Option<Arc<T>>) -> Self;
    /// Caller announced quiescence this cycle.
    pub fn load_in_cycle(&self) -> Option<&T>;
    pub fn load<'g>(&self, guard: &'g EpochGuard<'_>) -> Option<&'g T>;
    pub fn store(&self, value: Option<Arc<T>>) -> Retired<T>;
    pub fn try_store(&self, value: Option<Arc<T>>) -> Result<Retired<T>, Option<Arc<T>>>;
}
impl<T> Retired<T> {
    pub fn wait(self, deadline: Instant) -> Result<(), Self>;
    pub fn defer(self);
}
```

- [ ] **Step 1: Write the failing tests.**
  - `a_value_is_never_dropped_while_a_reader_holds_it`: drop counter; a reader pinned before `store` still reads the
    old value; drop happens only after `idle`.
  - `wait_never_returns_before_every_slot_passed_the_epoch`.
  - `try_store_refuses_while_a_retirement_is_pending`.
  - `loom_reader_and_writer_never_see_a_freed_value`.
- [ ] **Step 2: Run them and see them fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run them, loom and Miri.** PASS.
- [ ] **Step 5: Commit.** `git commit -m "euhedral-data-structures: Published, the stand-in for Java's collector"`

### Task 4.2: Epoch slots for workers

**Files:** `euhedral-core/src/flow_control/registry.rs`, `control_plane/fragment.rs` (`run_cycle`, the idle park).

**Interfaces:**
- `UpstreamRegistry::epochs() -> &Arc<EpochDomain>`.
- Each registered worker holds an `EpochSlot`. It calls `quiescent()` at `run_cycle` start and `idle()` before it
  parks.

- [ ] **Step 1: Write the failing test** `a_parked_worker_does_not_hold_back_a_retirement`: a lattice with 4 idle
  workers; a `Published` store's `wait(1 s)` returns `Ok` within 10 ms.
- [ ] **Step 2: Run it and see it fail.** Then **Step 3: implement**, and **Step 4: run** `mise exec -- cargo test -p
  euhedral-core`.
- [ ] **Step 5: Commit.** `git commit -m "Workers announce quiescence once per cycle"`

### Task 4.3: Cache partitions under Java's `pLocks` (C4)

**Files:** `euhedral-core/src/control_plane/cache.rs`; tests in `control_plane/cache/tests.rs`.

**Interfaces:** `partitions: Box<[CachePadded<ClaimCell<Partition>>]>`; `try_hold` → `try_claim`; `hold` (first touch,
reset) loops on `try_claim` with `spin_wait` (lifecycle; allowed by its reason tag).

- [ ] **Step 1: Write the failing tests.**
  - `stealers_racing_a_greedy_owner_never_take_an_ordered_frame`: 1 owner and 7 stealers, 100 000 frames, 10% ordered;
    every ordered frame runs on the owner, and every frame runs once.
  - `a_stealer_skips_a_claimed_partition_without_waiting`: the owner holds partition 0; a steal returns within 1 ms
    having taken from partition 1.
- [ ] **Step 2: Run them; the second fails today** (`try_lock` works, so check the return time against a held
  partition with work only there: expect 0 taken and an immediate return).
- [ ] **Step 3: Implement.** Delete `HOT-FIX(C4)`; ratchet -1.
- [ ] **Step 4: Run** `mise exec -- cargo test -p euhedral-core cache`. PASS.
- [ ] **Step 5: Commit.** `git commit -m "Cache partitions under per-partition claims (C4)"`

### Task 4.4: The cache directory (C5)

**Files:** `euhedral-core/src/control_plane/cache.rs` (`CacheDirectory`).

**Interfaces:**
- `slots: Box<[Published<ControlPlaneCache>]>`.
- `claim` stores. `cache_at` uses `load_in_cycle` and returns `Option<&ControlPlaneCache>` valid for the cycle.
- `withdraw` stores `None` when the slot holds this cache, then `defer`s the retirement.

- [ ] **Step 1: Write the failing test** `a_cache_closed_while_stolen_from_is_freed_after_the_stealers_cycle`: a drop
  counter on the cache; a stealer mid-steal; close; the counter stays 0 until the stealer's next `quiescent`.
- [ ] **Step 2: Run it and see it fail; Step 3: implement** (delete `HOT-FIX(C5)`, ratchet -1); **Step 4: run** the
  cache tests.
- [ ] **Step 5: Commit.** `git commit -m "Cache directory slots published, retired after the stealers' cycle (C5)"`

### Task 4.5: Routing states with Java's admission count, recycled (C6)

**Files:** `euhedral-core/src/flow_control/vertex.rs` (module-level `allow(unsafe_code)` with a reason);
`flow_control/vertex/tests.rs`.

**Interfaces:**
- `RoutingState` gains `admission: AtomicI64` with `RETIRED = i64::MIN`, and
  `fn try_acquire(&self) -> bool`, `fn retire(&self)`, `fn release(&self) -> bool` (true when the count left is
  exactly `RETIRED`), as Java's.
- `routing: AtomicPtr<RoutingState>` and `free: LifecycleMutex<Vec<Box<RoutingState>>>`. A state is reused when its
  `admission.load(SeqCst) == RETIRED`.
- `try_push` follows the spec §3.3 C6 protocol: load, `try_acquire`, re-check, push, `release`.
- `fn acquire_routing(&self) -> Option<RouteGuard<'_>>`; the guard's drop is `release`, plus
  `release_handles_when_quiescent` when it returns true.

- [ ] **Step 1: Write the failing tests.**
  - `pushes_racing_remaps_reach_only_validated_downstreams`: 8 pushers, 1 remapper doing 1000 remaps between two
    downstream sets; each downstream records the generation it belonged to when it received a frame; no frame
    reaches a downstream outside the generation its push validated; no frame is lost.
  - `a_remap_reuses_a_retired_state_once_its_readers_left`: the free list's length never exceeds 2 in a remap loop
    with no concurrent pushers.
  - `#[cfg(loom)] loom_stale_reader_and_recycler`: one reader loads the pointer and pauses; the writer retires,
    recycles and republishes; the reader's `try_acquire` succeeds only on the published state, else it releases.
- [ ] **Step 2: Run them and see them fail.**
- [ ] **Step 3: Implement.** Delete `HOT-FIX(C6)`; ratchet -1.
- [ ] **Step 4: Run** `mise exec -- cargo test -p euhedral-core flow_control` and the loom model. PASS.
- [ ] **Step 5: Commit.** `git commit -m "Vertex routing: Java's admission count, recycled states (C6)"`

### Task 4.6: Routing retirement as rebalance work (C7)

**Files:** `euhedral-core/src/flow_control/vertex.rs`.

**Interfaces:** `book: LifecycleMutex<Book>`. `release_handles_when_quiescent` (Java's name) takes it after
`HotScope::leave()`; the `handles_released` CAS (Java `handlesReleased`) runs it once per state.

- [ ] **Step 1: Write the failing test** `the_last_reader_of_a_retired_generation_closes_its_dropped_downstreams_once`:
  a downstream dropped by a remap is closed exactly once, after its last push, with 8 pushers.
- [ ] **Step 2: Run it; Step 3: implement** (delete `HOT-FIX(C7)`, ratchet -1); **Step 4: run** the vertex tests in
  debug (no hot-scope violation).
- [ ] **Step 5: Commit.** `git commit -m "Vertex routing retirement under the rebalance lock, as Java (C7)"`

### Task 4.7: Edge downstream and parent (C8, C9)

**Files:** `euhedral-core/src/flow_control/edge.rs`.

**Interfaces:**
- `downstream: SetOnce<Downstream>`. `add_downstream` and `add_downstream_node` call `set`; on `Err`, they attach
  below the existing node or report the error, as today.
- `push_downstream` reads `closed` (`Relaxed`), then `downstream.get()`.
- `parent: Published<Weak<dyn LatticeNode>>`. `set_parent` stores when the parent changed and defers the old one.

- [ ] **Step 1: Write the failing tests.**
  - `close_while_pushing_finishes_or_delivers_every_frame`.
  - `a_closed_edge_refuses_a_new_downstream`.
  - `set_parent_with_the_same_parent_stores_nothing`.
- [ ] **Step 2: Run them; Step 3: implement** (delete `HOT-FIX(C8)`, `(C9)`, ratchet -2); **Step 4: run**
  `mise exec -- cargo test -p euhedral-core`.
- [ ] **Step 5: Commit.** `git commit -m "Edges: set-once downstream, published parent (C8, C9)"`

### Task 4.8: Paired benchmark, docs and the draft PR

- [ ] **Step 1: Benchmark as Task 3.4**, adding a remap-under-load run (p99 push latency during 100 remaps). Keep or
  discard by the paired gate.
- [ ] **Step 2: Write the docs** (GLOSSARY **Published**, **routing admission**) and run the gate.
- [ ] **Step 3: Push and open the draft PR** "Cache, vertex, edges: Java's claims and admission, Published".

---

## Chunk 5 (draft PR, `hot-path-5-fan-in`): one-shot, latch, sequencer, lake (C14, G5, G6, G7)

### Task 5.1: `OneShot<T>`

**Files:** `euhedral-data-structures/src/sync/one_shot.rs` (+ loom).

**Interfaces:**

```rust
pub fn one_shot<T: Send>() -> (OneShotSender<T>, OneShotReceiver<T>);
impl<T: Send> OneShotSender<T> { pub fn send(self, value: T); } // Drop without send: abandoned
pub enum RecvError { Abandoned }
impl<T: Send> OneShotReceiver<T> {
    pub fn try_recv(&mut self) -> Result<Option<T>, RecvError>;
    pub fn recv(self) -> Result<T, RecvError>; // parks; for threads outside the lattice
    pub fn recv_timeout(self, d: Duration) -> Result<T, Result<Self, RecvError>>;
}
```

- [ ] **Step 1: Write the failing tests.**
  - `send_before_recv_and_recv_before_send_both_deliver`.
  - `a_dropped_sender_abandons`.
  - `10_000_pairs_across_two_threads_each_deliver_once`.
  - `loom_send_racing_recv_parks_and_wakes_once`.
- [ ] **Step 2: Run them and see them fail; Step 3: implement; Step 4: run them, loom and Miri.**
- [ ] **Step 5: Commit.** `git commit -m "euhedral-data-structures: OneShot, Java's CompletableFuture for one result"`

### Task 5.2: `PipelineRunner::submit*` returns a one-shot (C14)

**Files:** `euhedral-core/src/ingest/pipeline_runner.rs`, `pipeline_runner/tests.rs`.

**Interfaces:**
- `submit`, `submit_seeded` and `submit_on` return `Result<OneShotReceiver<Outcome>, AdmissionClosed<I>>`.
- The chain's last finalizer calls `send`.

- [ ] **Step 1: Update the existing tests** to `recv()`, and add
  `outcomes_of_1000_chains_from_4_submitters_each_arrive_once`.
- [ ] **Step 2: Run them and see them fail to compile; Step 3: implement** (delete `HOT-FIX(C14)`, ratchet -1);
  **Step 4: run** `mise exec -- cargo test -p euhedral-core pipeline_runner`.
- [ ] **Step 5: Commit.** `git commit -m "PipelineRunner::submit returns a lock-free one-shot (C14)"`

### Task 5.3: `Latch` on `OneShot` (G5)

**Files:** `euhedral-graph/src/latch.rs`.

**Interfaces:** `Latch::new() -> (LatchSetter<T>, LatchWaiter<T>)` unchanged. The setter wraps `OneShotSender`; its drop
without `set` abandons. `LatchWaiter::wait` is `recv`.

- [ ] **Step 1: Keep the existing tests**; add `set_from_8_workers_wakes_the_waiter_once`.
- [ ] **Step 2-4: Implement and run** `mise exec -- cargo test -p euhedral-graph latch` (delete `HOT-FIX(G5)`,
  ratchet -1).
- [ ] **Step 5: Commit.** `git commit -m "Latch on OneShot (G5)"`

### Task 5.4: `Sequencer` as Java's (G6)

**Files:** `euhedral-graph/src/sequencer.rs` (module-level `allow(unsafe_code, reason = "the wip winner owns the order
consumer")`).

**Interfaces:**
- API unchanged (`new`, `register -> Slot<T>`, `Slot::{complete, abandon}`, `len`, `is_empty`).
- Inside, Java's design (spec §3.4 G6):
  - `order: Producer<Arc<Entry<T>>, Multi>`;
  - `consumer: UnsafeCell<Consumer<Arc<Entry<T>>, Single>>`;
  - `wip: CachePadded<AtomicU32>`;
  - `Entry { ready: AtomicBool, value: UnsafeCell<Option<T>> }`.
- `register` also keeps a `len: AtomicUsize`.

- [ ] **Step 1: Write the failing tests.**
  - `registrations_from_4_threads_complete_in_reverse_and_emit_in_order`: register 4 × 250 from 4 threads, each in
    order; complete all from 4 threads in reverse order; the emitted sequence equals each thread's registration order
    (merged by registration), each once.
  - `an_emit_that_panics_does_not_stop_the_ones_behind_it`.
- [ ] **Step 2: Run them and see them fail; Step 3: implement** (delete `HOT-FIX(G6)`, ratchet -1); **Step 4: run**
  `mise exec -- cargo test -p euhedral-graph sequencer`.
- [ ] **Step 5: Commit.** `git commit -m "Sequencer: Java's MPSC order and wip drain (G6)"`

### Task 5.5: The lake's two modes on Java's hot side (G7, G8, G9)

**Files:** `euhedral-graph/src/lake.rs`, `euhedral-graph/tests/lake.rs`.

**Interfaces** (spec §3.4, G7-G9; Java's `FrameLake` implementation):
- The public API is unchanged: `Lake::new(.., idle_release: Duration)` (`Duration::ZERO` = attached for life),
  `admit`, `offer`, `complete_gracefully`, `active`, `is_attached`, `attachments`, `sinks`.
- `generation: Published<Generation>` (Consumes: `Published`, Task 4.1). `offer` and `admit` read it with one pinned
  `load()`.
- `attach_lock: LifecycleMutex<()>` around `attach` and `detach`, taken after `HotScope::leave()` and only when no set
  is attached (first use, or after a detach) or the watch detaches.
- `retiring: LifecycleMutex<Vec<Retired<Generation>>>` (G8) holds detached generations until their sinks complete.
- `idle_since_ns: AtomicU64`, stored (`SeqCst`) by the `terminated` that brings the count to 0.
- `accepted` (the `CLOSED` sign bit) and `finished` as Java's.
- The watch's `downstream: DownstreamSlot` (G9). The watch is attached in both modes; its `request`/`pull` return at
  once when `idle_release` is zero, a unit is active, or the lake has been idle for less than `idle_release`.
- `offer` follows Java's `publish` loop: a detached set's sinks refuse, `attach` gives the set attached since, and a
  finished lake or an unchanged set refuses.

- [ ] **Step 1: Write the failing tests** in `tests/lake.rs`. Keep the existing detach and re-attach tests. Add:
  - `offers_racing_an_idle_detach_land_in_exactly_one_set`: `idle_release` 1 ms; 8 threads `admit`, `offer` and
    terminate in bursts with idle gaps; every offered frame runs once; `attachments() > 1`.
  - `offers_on_an_attached_set_take_no_lock`: inside `HotScope::enter()`, 10 000 offers to an attached lake; no
    debug violation.
  - `an_attached_for_life_lake_never_detaches`: `Duration::ZERO`; idle 50 ms; `is_attached()` stays true and
    `attachments() == 1`.
  - `offers_from_8_threads_racing_complete_finish_every_admitted_unit`.
- [ ] **Step 2: Run them and see the second fail** (the generation `RwLock` read is a std lock inside a hot scope
  once Task 1.3 tagged it, and the `Arc` clone remains).
- [ ] **Step 3: Implement.** Delete `HOT-FIX(G7)`; ratchet -1.
- [ ] **Step 4: Run** `mise exec -- cargo test -p euhedral-graph`. PASS.
- [ ] **Step 5: Commit.** `git commit -m "Lake: Java's published generation; attach and detach as rebalance work (G7-G9)"`

### Task 5.6: Chunk 5 docs and the draft PR

- [ ] **Step 1: Write the docs.** CHANGELOG: `submit*` is breaking. GLOSSARY: **OneShot**, **idle detach**. Run the
  gate.
- [ ] **Step 2: Push and open the draft PR** "Fan-in: one-shot, latch, Java's sequencer, lake on Java's hot side".

---

## Chunk 6 (draft PR, `hot-path-6-stage-graph`): stage graph (G10-G15)

### Task 6.1: `TakeAllStack<T>`

**Files:** `euhedral-data-structures/src/sync/take_all.rs` (+ loom).

**Interfaces:**

```rust
pub struct TakeAllStack<T, S: Copy + Eq> { /* AtomicPtr head; sentinel nodes per S */ }
impl<T, S: Copy + Eq> TakeAllStack<T, S> {
    pub const fn new() -> Self;
    /// Err((value, sentinel)) once closed.
    pub fn push(&self, value: T) -> Result<(), (T, S)>;
    /// swap; returns the items, oldest first.
    pub fn close(&self, sentinel: S) -> Vec<T>;
    /// For the next binding.
    pub fn reopen(&mut self);
}
```

- [ ] **Step 1: Write the failing tests.**
  - `pushes_racing_close_are_either_taken_or_handed_back`: 8 pushers × 10 000 against one `close`; every value is
    accounted for once.
  - `loom_push_and_close`.
- [ ] **Step 2-4: Implement; run the tests, loom and Miri.**
- [ ] **Step 5: Commit.** `git commit -m "euhedral-data-structures: TakeAllStack, Java's closable CAS stack"`

### Task 6.2: Quantum and cause (G10, G11)

**Files:** `euhedral-graph/src/stage_graph.rs` (module-level `allow(unsafe_code, reason = "the run's counts order the
quantum's phases")`).

**Interfaces:** `quantum: UnsafeCell<Option<Q>>`; `fn quantum(&self) -> &Q` (`// SAFETY:` `holds > 0`);
`fn take_quantum(&self) -> Q` (after `holds` reached zero); `cause: SetOnce<QuantumOutcome>`.

- [ ] **Step 1: Write the failing test** `the_first_failure_wins_among_8_failing_stages` and keep the existing
  `stage_graph` tests.
- [ ] **Step 2-4: Implement; run** `mise exec -- cargo test -p euhedral-graph stage_graph` and
  `--test credit_graph` (delete `HOT-FIX(G10)`, `(G11)`; ratchet -2).
- [ ] **Step 5: Commit.** `git commit -m "Stage graph: plain quantum ordered by the counts, first-wins cause (G10, G11)"`

### Task 6.3: Arrival slots (G12)

**Interfaces:** `StageRun.inputs: Box<[UnsafeCell<Option<Arrival>>]>`, one per incoming edge (edge index from the shape).
The arrival writes its slot, then `pending.fetch_sub(1, SeqCst)`; the arrival that reaches zero reads the slots.

- [ ] **Step 1: Write the failing test** `a_diamond_on_8_workers_runs_the_join_with_every_input`: a diamond graph,
  1000 quanta; the join stage sees exactly its two predecessors' markers each time.
- [ ] **Step 2-4: Implement and run** (delete `HOT-FIX(G12)`; ratchet -1).
- [ ] **Step 5: Commit.** `git commit -m "Stage graph: per-edge arrival slots, read after the last arrival (G12)"`

### Task 6.4: Externals (G13)

**Interfaces:**
- `StageRun.waiters: TakeAllStack<ArriveWith, Sealed>`, where `enum Sealed { Submitted, Swept }`.
- `StageRun.out: UnsafeCell<Option<Vec<Marker>>>`, written before `close(Submitted)`.
- `subscribe` on a closed stack arrives at once with `out` (`Submitted`) or the graph's joined marker (`Swept`).

- [ ] **Step 1: Write the failing test** `subscriptions_before_during_and_after_publication_each_arrive_once`.
- [ ] **Step 2-4: Implement and run** (delete `HOT-FIX(G13)`; ratchet -1).
- [ ] **Step 5: Commit.** `git commit -m "Stage graph: externals on a closable stack (G13)"`

### Task 6.5: Boundary from the used-lane mask (G14)

**Interfaces:**
- `used_lanes: Box<[AtomicU64]>`, one word per 64 lanes. The bit is set with `load`, then `fetch_or(SeqCst)` when it
  is missing.
- `arm_boundary` marks each used lane other than home and awaits it on home.
- A deferred host completion increments `live` and decrements it when done. `BoundarySet` goes.

- [ ] **Step 1: Write the failing test** `the_boundary_covers_work_on_every_used_lane`: a fake pool of 3 lanes; stages
  on lanes 1 and 2; the retirement fires only after both lanes' last work completed (fake markers).
- [ ] **Step 2: Add** `a_deferred_host_stage_delays_quiesce_until_it_completes`.
- [ ] **Step 3-4: Implement and run** (delete `HOT-FIX(G14)`; ratchet -1).
- [ ] **Step 5: Commit.** `git commit -m "Stage graph: Java's used-lane mask and deferred live count (G14)"`

### Task 6.6: Pin slots (G15)

**Interfaces:** `pins: Box<[UnsafeCell<Option<QuantumRef>>]>` indexed by await slot (registration assigns
`first_slot` per stage), written before `start` publishes; taken by the stage after its final arrival, or by `quiesce`
once `live == 0`.

- [ ] **Step 1: Write the failing test** `external_pins_are_released_once_whether_the_stage_runs_or_is_swept`.
- [ ] **Step 2-4: Implement and run** the stage-graph tests (delete `HOT-FIX(G15)`; ratchet -1).
- [ ] **Step 5: Commit.** `git commit -m "Stage graph: pins in fixed await slots (G15)"`

### Task 6.7: Chunk 6 docs and the draft PR

- [ ] **Step 1: Docs** (GLOSSARY **TakeAllStack**) and the gate. **Step 2: Push and open the draft PR** "Stage graph:
  Java's counts, slots and closable stacks".

---

## Chunk 7 (draft PR, `hot-path-7-uring`): io_uring (I3, I4, I5, I6, S6a)

### Task 7.1: Tag-indexed operation slots (I6)

**Files:** create `euhedral-io/src/slots.rs` (module-level `allow(unsafe_code)` if the crate denies it); modify
`uring.rs`.

**Interfaces:**

```rust
pub(crate) struct Slots<Op> { tags: AtomicU64, slots: Box<[Slot<Op>]> } // len: a power of two
impl<Op> Slots<Op> {
    pub(crate) fn new(len: usize) -> Self;
    /// tag = tags.fetch_add(1) & (len - 1); CAS EMPTY → BUSY (SeqCst); write op; store FULL (Release).
    /// Err(op) when the slot is not empty.
    pub(crate) fn insert(&self, op: Op) -> Result<u64, Op>;
    /// Acquire; FULL → take it → EMPTY (Release). Reaper only.
    pub(crate) fn take(&self, tag: u64) -> Option<Op>;
    /// Reaper only, for a short piece.
    pub(crate) fn get_mut(&self, tag: u64) -> Option<&mut Op>;
    pub(crate) fn in_flight(&self) -> usize;
}
```

- [ ] **Step 1: Write the failing tests.**
  - `inserts_from_8_threads_and_one_reaper_lose_nothing`: 100 000 operations; every tag taken once.
  - `a_busy_slot_hands_the_operation_back`.
  - Miri on both.
- [ ] **Step 2-4: Implement, then switch `Ring.ops` to it.** Delete `HOT-FIX(I6)`; ratchet -1. Run
  `mise exec -- cargo test -p euhedral-io`.
- [ ] **Step 5: Commit.** `git commit -m "euhedral-io: Java's tag-indexed operation slots (I6)"`

### Task 7.2: The ring list (I3)

**Interfaces:**
- `rings: Box<[SetOnce<Arc<Ring>>]>` of `MAX_RINGS = 64`, and `ring_count: AtomicUsize`.
- `ring()` takes an index with `fetch_add` and refuses past `MAX_RINGS`. `poll` walks `0..min(count, MAX_RINGS)`.

- [ ] **Step 1: Write the failing test** `a_poll_allocates_nothing`: count allocations around 1000 polls with 8 rings
  (counting global allocator, as `euhedral-gpu/tests/cuda_alloc.rs`); 0.
- [ ] **Step 2-4: Implement and run** (delete `HOT-FIX(I3)`; ratchet -1).
- [ ] **Step 5: Commit.** `git commit -m "io_uring: ring list as Java's AtomicReferenceArray (I3)"`

### Task 7.3: The thread is its ring's only submitter (I4, S6a)

**Interfaces:**
- `Ring.sq: UnsafeCell<SubmissionSide>`, reached only through the owning thread's `MINE` entry (`// SAFETY:` thread
  confinement). A debug check compares `owner: ThreadId`.
- `submit*` builds the entry and calls `enter` at once.
- A refusal returns `Err(Refused { request, buf, then })`. Its causes are a full submission queue,
  `in_kernel == capacity`, a busy tag, no ring, or an enter error before the kernel took the entry.
- `incoming`, `waiting`, `flush_waiting`, `Queue::backlog` and `unsubmitted` go.
- The public `AsyncIo::submit*` return `Result<(), Refused>`.

- [ ] **Step 1: Write the failing tests.**
  - `a_refused_submission_hands_the_request_back_and_completes_nothing`: fill the ring (a small capacity), submit
    once more; `Err` carries the buffer back; after the ring drains, exactly `capacity` completions arrived.
  - `submissions_from_8_threads_enter_on_their_own_rings`: every completion arrives once; no ring saw two submitting
    threads (debug owner check).
  - Update `operations_beyond_the_queues_room_wait_and_all_complete` to resubmit refused requests in the test.
- [ ] **Step 2: Run them and see them fail.**
- [ ] **Step 3: Implement.** Delete `HOT-FIX(I4)` and the `S6a` marker; ratchet -2.
- [ ] **Step 4: Run** `mise exec -- cargo test -p euhedral-io`. PASS.
- [ ] **Step 5: Commit.** `git commit -m "io_uring: each ring's thread submits and enters the kernel itself (I4, S6a)"`

### Task 7.4: Short pieces on the reaper's ring; the completion queue under the source (I5)

**Interfaces:**
- `reap_one` resubmits a short rest through `ring()` of the reaping thread. A refusal ends the operation with
  `io::Error::other("the rest of a short operation could not be submitted")`.
- The `cq` lock goes; the completion queue is reaped inside `Front::reap` (`SourceCell`).
- The `pending() == 0` gate in `Front::reap` goes.

- [ ] **Step 1: Keep** `short_reads_and_writes_are_resubmitted_until_every_byte_moved` and
  `a_short_send_is_resubmitted_until_every_byte_is_in_the_kernel`. Add
  `a_short_rest_that_cannot_be_submitted_fails_the_operation` (the reaper's ring forced full).
- [ ] **Step 2-4: Implement and run** (delete `HOT-FIX(I5)`; ratchet -1).
- [ ] **Step 5: Commit.** `git commit -m "io_uring: rests on the reaper's own ring; completion queue under the source (I5)"`

### Task 7.5: Chunk 7 docs and the draft PR

- [ ] **Step 1: Write the docs.** CHANGELOG: `submit*` may refuse (breaking). GLOSSARY: **refused submission**. Run
  the gate.
- [ ] **Step 2: Push and open the draft PR** "io_uring: Java's per-thread rings and tag slots".

---

## Chunk 8 (draft PR, `hot-path-8-windows`): Windows engine (I7-I15)

All tasks: files `euhedral-io/src/win/{mod.rs,real.rs,sim.rs,sys.rs,tests.rs}`. Tests run on every platform through
`sim.rs`; live tests run in the Windows workflow (`win/live.rs`).

### Task 8.1: Ring list (I13)
- [ ] **Step 1-5:** as Task 7.2 for `win::Engine.rings`. Test `a_windows_poll_allocates_nothing` (sim). Delete
  `HOT-FIX(I13)`; ratchet -1. Commit "Windows: ring list as Java's (I13)".

### Task 8.2: Tag slots and the `OVERLAPPED` array (I7)

**Interfaces:**
- `Slots<Op>` (7.1) with an `overlapped: Box<[OVERLAPPED]>` array indexed by tag.
- `tag_of(ptr) = (ptr - base) / size_of::<OVERLAPPED>()`, checked against the engine's array bounds; a stray pointer
  is rejected.
- `by_addr` goes.

- [ ] **Step 1: Write the failing tests.**
  - `a_packet_from_another_engine_is_rejected`.
  - `operations_from_8_threads_complete_once` (sim).
- [ ] **Step 2-4: Implement and run.** Delete `HOT-FIX(I7)`; ratchet -1.
- [ ] **Step 5: Commit.** `git commit -m "Windows: tag slots and one OVERLAPPED per tag (I7)"`

### Task 8.3: A failed start is posted as its completion (I8)

**Interfaces:**
- `refused: Box<[AtomicU32]>` by tag.
- On a failed start: store the error, then `PostQueuedCompletionStatus` the tag's `OVERLAPPED`.
- If the post fails, clear the slot and refuse.
- `reap` reports `refused[tag]` first.

- [ ] **Step 1: Write the failing test** `a_call_that_fails_at_once_completes_with_its_error` (sim injects the failure).
- [ ] **Step 2-4: Implement and run.** Delete `HOT-FIX(I8)`; ratchet -1.
- [ ] **Step 5: Commit.** "Windows: failed starts posted to the port, as Java (I8)".

### Task 8.4: Pool-job results (I9)

**Interfaces:** `results: MpscQueue<PoolResult>`; pool threads `offer`; the source drains with the `Single` consumer in
a `SourceCell`.

- [ ] **Step 1-5:** test `opens_and_renames_from_4_threads_complete_once` (sim). Delete `HOT-FIX(I9)`; ratchet -1.
  Commit "Windows: pool results on an MPSC queue (I9)".

### Task 8.5: Taken entries under the source (I10)

- [ ] **Step 1-5:** `inbox: SourceCell<Inbox>` (Java `entries`/`taken`/`next`). Test
  `a_pull_stopped_midway_hands_the_rest_on_later_in_order` (sim). Delete `HOT-FIX(I10)`; ratchet -1. Commit "Windows:
  taken entries are reap state (I10)".

### Task 8.6: Association at open and `register` (I11)

**Interfaces:**
- `AsyncIo::register(&self, handle: RawHandle) -> io::Result<()>` (Windows) associates a handle once (lifecycle). Files
  opened by the engine are associated at open.
- The per-operation `join_port` and the `HashSet` go.
- A debug assertion checks that the handle was registered (a `LifecycleMutex<HashSet>` read only in debug builds).

- [ ] **Step 1: Write the failing tests.**
  - `a_registered_socket_sends_and_receives`.
  - `#[cfg(debug_assertions)] #[should_panic(expected = "not registered")] an_unregistered_socket_panics_in_debug`.
- [ ] **Step 2-4: Implement and run.** Update the live tests to `register`. Delete `HOT-FIX(I11)`; ratchet -1.
- [ ] **Step 5: Commit.** "Windows: associate at open or register, never per operation (I11)".

### Task 8.7: Raw handle references (I12)

- [ ] **Step 1-5:** I/O ring entries use raw handle references (Java `IORING_REF_RAW`); the ring handle set goes. Test
  `file_reads_on_rings_need_no_handle_registration` (sim). Delete `HOT-FIX(I12)`; ratchet -1. Commit "Windows: raw
  handle refs on I/O rings (I12)".

### Task 8.8: The ring is its thread's; retired, not broken (I14, I15)

**Interfaces:**
- The build-and-submit side is thread-confined (as 7.3).
- `in_flight` is bounded by the ring's entries: refuse beyond them.
- A failed submit sets `retired: AtomicBool` (`Release`), and the thread's next `ring()` makes a new one.
- The reaper reaps a retired ring until it is empty.
- The backlog and `break_ring`'s lock go. Refusals as 7.3.

- [ ] **Step 1: Write the failing tests.**
  - `a_ring_that_fails_retires_and_its_thread_moves_to_a_new_one` (sim).
  - `a_retired_ring_is_reaped_until_empty`.
- [ ] **Step 2-4: Implement and run.** Delete `HOT-FIX(I14)`, `(I15)`; ratchet -2.
- [ ] **Step 5: Commit.** "Windows: per-thread rings retire instead of breaking (I14, I15)".

### Task 8.9: Chunk 8 docs and the draft PR

- [ ] **Step 1: Write the docs.** CHANGELOG: `register` is new; Windows refusals. Run the gate and the Windows workflow.
- [ ] **Step 2: Push and open the draft PR** "Windows engine: Java's tag slots, port posts and per-thread rings".

---

## Chunk 9 (draft PR, `hot-path-9-memory`): memory (M1, S1, S2)

### Task 9.1: Pinned host pool (M1)

**Files:** create `euhedral-memory/src/cuda/pinned_pool.rs`; modify `cuda/backends.rs` (the `cuMemHostAlloc` backend,
`:444`, `:478`), `cuda/mod.rs`.

**Interfaces:**

```rust
pub struct PinnedPool { classes: Box<[MpmcQueue<Block>]>, pool_bytes: AtomicU64, limit: u64,
                        excess: MpmcQueue<Block>, overflows: AtomicU64, closing: AtomicBool }
impl PinnedPool {
    /// next power of two, at least 64 KiB (Java pinnedCapacity)
    pub fn capacity(bytes: u64) -> u64;
    /// default: max(64 MiB, total memory / 256)
    pub fn new(limit: u64) -> Self;
    /// a pooled block of its class, or None
    pub fn acquire(&self, bytes: u64) -> Option<Block>;
    /// pool it, or queue it as excess
    pub fn release(&self, block: Block);
    /// frees the excess on the calling worker; returns how many
    pub fn trim(&self, free: impl FnMut(Block)) -> usize;
    /// frees everything; close
    pub fn drain(&self, free: impl FnMut(Block));
    pub fn pool_bytes(&self) -> u64;
    pub fn overflows(&self) -> u64;
}
```

- [ ] **Step 1: Write the failing tests** (no device; `free` is a counting closure).
  - `a_released_block_serves_the_next_request_of_its_class`.
  - `releases_beyond_the_limit_go_to_excess_and_trim_frees_them`.
  - `capacity_rounds_to_a_power_of_two_of_at_least_64_kib`: `capacity(1) == 65536`, `capacity(65537) == 131072`.
  - `drain_frees_every_block_once`.
- [ ] **Step 2-4: Implement, then wire the pinned backend.** Its `allocate` tries `acquire`; its `free` calls `release`.
  Run `mise exec -- cargo test -p euhedral-memory`, and `--features cuda` on the GPU through the machine's GPU
  scheduler.
- [ ] **Step 5: Commit.** `git commit -m "euhedral-memory: pinned host pool by size class, as Java's (M1)"`

### Task 9.2: `Reaper` on the worker, and the docs text (M1)

**Files:** `euhedral-memory/src/backend.rs`.

**Interfaces:**
- `reap` stops at the first unretired item, with no keep-`Vec`.
- The module doc (`:5-7`) and the `Reaper` doc (`:95-97`) get the spec's §3.8 replacement texts, verbatim.

- [ ] **Step 1: Write the failing test** `reap_frees_the_retired_prefix_and_allocates_nothing` (counting allocator).
- [ ] **Step 2-4: Implement and run.** Delete the `HOT-FIX(M1)` marker; ratchet -1.
- [ ] **Step 5: Commit.** `git commit -m "Reaper reaps on the worker; docs say where long frees run (M1)"`

### Task 9.3: `GiveBacks` without a wake (S1)

**Files:** `euhedral-memory/src/planner.rs`, `tests/planner.rs`.

**Interfaces:**
- `GiveBacks::wake` and its call at `:733` go.
- `demand` is Java's accumulate (as 2.2).
- `request` pushes ready orders within it; `pull` hands over its own demand.
- The order consumer lives in a `SourceCell`.

- [ ] **Step 1: Write the failing test** `give_backs_complete_in_a_lattice_that_never_pulls`: a driver calling only
  `request(1)`; every ordered give-back is delivered once.
- [ ] **Step 2-4: Implement and run.** Delete the `S1` marker; ratchet -1.
- [ ] **Step 5: Commit.** `git commit -m "Planner give-backs: request drains within demand, no producer wake (S1)"`

### Task 9.4: `Moves` pumped by `request` (S2)

**Files:** `euhedral-memory/src/tiers.rs`, `tests/tiers.rs`.

**Interfaces:** `Moves::wake` goes; `request` runs `Mover::pump`, then pushes finished moves within demand; `pull`
pumps and hands over at most its own demand.

- [ ] **Step 1: Write the failing test** `moves_advance_in_a_lattice_that_never_pulls`.
- [ ] **Step 2-4: Implement and run.** Delete the `S2` marker; ratchet -1.
- [ ] **Step 5: Commit.** `git commit -m "Tier moves pumped by request (S2)"`

### Task 9.5: Chunk 9 docs and the draft PR

- [ ] Docs (GLOSSARY **pinned pool**), the gate, push, draft PR "Memory: pinned pool, worker reaping, request-driven
  sources".

---

## Chunk 10 (draft PR, `hot-path-10-ticks`): tick path (C16-C20, H1-H7)

### Task 10.1: Lock-free `TaskPool` (C20)

**Files:** `euhedral-core/src/internal/task_pool.rs`.

**Interfaces:**
- `jobs: MpmcQueue<Task>`; `idle: MpmcQueue<Thread>` (parked threads announce themselves).
- `execute` offers the job, then pops one idle thread and unparks it. It never waits.
- Workers: pop a job, or announce themselves idle, re-check, and park (`#[allow(.., reason = "LIFECYCLE(C20)")]` on
  the park: pool threads are lifecycle-created and wait on nobody).
- `shutdown` sets `closing` and unparks all.

- [ ] **Step 1: Write the failing tests.**
  - `jobs_from_4_submitters_each_run_once`.
  - `an_idle_pool_runs_a_new_job_within_1_ms`.
  - `shutdown_runs_queued_jobs_then_joins`.
- [ ] **Step 2-4: Implement and run.** Delete `HOT-FIX(C20)`; ratchet -1.
- [ ] **Step 5: Commit.** `git commit -m "TaskPool: lock-free job queue, park-on-idle workers (C20)"`

### Task 10.2: Lattice update state (C16, C17, C18)

**Files:** `euhedral-core/src/control_plane/lattice.rs`, `lattice/tests.rs`.

**Interfaces:**
- `closed`, `resetting`, `primed`, `rebalancing` and `ready` are `AtomicBool`; `version: AtomicI64`.
- `active_shards: Published<Box<[usize]>>`. `update` reads them without a lock.
- `rebalancing.compare_exchange` guards `handle_system_topology_change`.
- The `close` gate and `updating` locks go. `State` keeps lifecycle-only fields under a `LifecycleMutex`.

- [ ] **Step 1: Write the failing test** `10_000_ticks_take_no_lifecycle_lock`: the dispatcher thread inside
  `HotScope::enter()` calls `update` 10 000 times; no violation in debug.
- [ ] **Step 2-4: Implement and run.** Delete `HOT-FIX(C16)`, `(C17)`, `(C18)`; ratchet -3.
- [ ] **Step 5: Commit.** `git commit -m "Lattice update on Java's atomics (C16-C18)"`

### Task 10.3: Shard update state (C19)

**Files:** `euhedral-core/src/control_plane/shard.rs`.

**Interfaces:** `clones: Published<Box<[Option<Arc<Clone>>]>>`, `active_cores: Published<Box<[usize]>>`, and
`current_version`, `started`, `primed`, `rebalancing` as atomics. `update` iterates the published slices, with no
`Vec`.

- [ ] **Step 1: Write the failing test** `shard_ticks_take_no_lock_and_allocate_nothing`.
- [ ] **Step 2-4: Implement and run.** Delete `HOT-FIX(C19)`; ratchet -1.
- [ ] **Step 5: Commit.** `git commit -m "Shard update on Java's published arrays (C19)"`

### Task 10.4: Monitor lifecycle state (H1)

**Files:** `euhedral-hardware-utils/src/monitor/mod.rs`, `monitor/tests.rs`.

**Interfaces:**
- `state: AtomicU8`, with Java's states NEW, RUNNING, STOPPED, CLOSING, CLOSED.
- `polling_thread: SetOnce<Thread>`.
- The poller waits for its tick with `park_timeout` until the deadline
  (`#[allow(.., reason = "JAVA-BACKOFF")]`, Java `DeadlineWaiter`).
- `start`, `stop` and `close` CAS the state, then unpark.

- [ ] **Step 1: Write the failing tests.** `stop_and_start_while_polling_never_lose_a_transition`; keep the existing
  close tests.
- [ ] **Step 2-4: Implement and run.** Delete `HOT-FIX(H1)`; ratchet -1.
- [ ] **Step 5: Commit.** `git commit -m "Monitor lifecycle as Java's CAS state and park (H1)"`

### Task 10.5: Evaluation under a claim; the provider per monitor (H2, H4)

**Interfaces:**
- `evaluation: ClaimCell<Evaluation>` holds the state engine, the pressure state and the provider.
- `evaluate_and_publish` claims it or returns the last published utilization.
- The `EVAL_ACTIVE`/`PUB_CLAIMED` flags stay, `SeqCst`. The process-wide `system_snapshot` stays for outside callers.

- [ ] **Step 1: Write the failing test** `an_on_demand_read_during_a_tick_returns_the_last_value_without_waiting`.
- [ ] **Step 2-4: Implement and run.** Delete `HOT-FIX(H2)`, `(H4)`; ratchet -2.
- [ ] **Step 5: Commit.** `git commit -m "Monitor evaluation under a claim, provider per monitor (H2, H4)"`

### Task 10.6: Last utilization published (H3)

- [ ] **Step 1-5:** `last: Published<HardwareUtilization>`. Test `utilization_reads_during_ticks_see_a_whole_value`.
  Delete `HOT-FIX(H3)`; ratchet -1. Commit "Monitor: last utilization published (H3)".

### Task 10.7: Dispatcher mailbox (H5)

**Files:** `euhedral-hardware-utils/src/monitor/dispatcher.rs`.

**Interfaces:**
- `mailbox: AtomicPtr<T>`. `offer` boxes the value, `swap`s it in, drops the replaced value and unparks the dispatch
  thread.
- `listeners: Published<Box<[(ListenerId, Listener)]>>`, changed under a `LifecycleMutex` in `add_listener` and
  `remove_listener`.
- `closing: AtomicBool` (`SeqCst`).
- The dispatch loop swaps the mailbox to null, delivers, and parks when it is empty and not closing.

- [ ] **Step 1: Write the failing tests.**
  - `offers_racing_close_deliver_the_latest_or_nothing_after_close`.
  - `a_burst_of_offers_delivers_the_last`.
  - `listeners_added_during_dispatch_see_the_next_value`.
- [ ] **Step 2-4: Implement and run.** Delete `HOT-FIX(H5)`; ratchet -1.
- [ ] **Step 5: Commit.** `git commit -m "Dispatcher: latest-value mailbox and copy-on-write listeners (H5)"`

### Task 10.8: Topology mapper (H6, H7)

**Files:** `euhedral-hardware-utils/src/topology/mapper.rs`.

**Interfaces:**
- `pending: AtomicPtr<PendingRequest>`, installed when greatest by CAS (Java `installGreatest`).
- `versions: ClaimCell<Box<[u32]>>`; the drain loop releases with `release_seq_cst`, then re-checks `pending`.
- `effective: Published<EffectiveTopology>`.

- [ ] **Step 1: Write the failing test** `updates_from_4_threads_publish_the_greatest_sequence_last`.
- [ ] **Step 2-4: Implement and run.** Delete `HOT-FIX(H6)`, `(H7)`; ratchet -2.
- [ ] **Step 5: Commit.** `git commit -m "Topology mapper as Java's drain owner (H6, H7)"`

### Task 10.9: Chunk 10 docs and the draft PR

- [ ] The gate (ratchet at 0 once chunks 2-9 have merged), docs, push, draft PR "Tick path: Java's atomics, lock-free
  task pool and dispatcher".

---

## Chunk 11 (draft PR, `hot-path-11-door`): lock the door

### Task 11.1: No tag may come back

**Files:** `euhedral-core/tests/hot_fix_ratchet.rs`, `CHANGELOG.md`.

- [ ] **Step 1: Change the test** to `there_are_no_hot_fix_tags`: `assert!(tags().is_empty(), "{:?}", tags())`.
  Remove `HOT_FIX_TAGS`.
- [ ] **Step 2: Run the full gate.** A new `Mutex` in hot code fails clippy; a lifecycle lock in a hot scope fails
  the debug suite; a second thread in a source fails `SourceCell`'s debug check.
- [ ] **Step 3: Commit, push, open the draft PR** "Lock the door: no hot-path lock tags remain".
