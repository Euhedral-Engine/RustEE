# Lanes on `PipelineRunner`

Status: design, no code. Base: `origin/main` at 47b957f.

## The change

Today `LaneSource::serve` runs each item inside the source's own `request` (`euhedral-graph/src/lane_source/source.rs:478-533`).
Workers learn the shape of the workload only from the frames they execute (`fragment.rs:498-515`), so that work is invisible
to their batch sizing, idling and participation.

A lane becomes a `PipelineRunner` whose items are chains. The lattice runs every step of an item as a stage, and
`PipelineRunner`, `PipelineFrame` and `Completions` already provide everything else a lane needs, apart from the three small
additions in §2. `LaneSource` and `LaneIntakes` go away.

## 1. A lane as a pipeline

One `PipelineRunner<W>` per lane, built with the lane's hash as its id hash and one partition:

```text
PipelineFrame::<W>::builder()
    .hold(|w| w.gate(), |gate, chain| completions.after(gate, chain))   // admission: not released until ready
    .try_fan_in(submit)                                                 // the lane's hash: in order, on one core
    .hold(|s| s.retirement, |marker, chain| completions.after_marker(marker, chain))
    .try_fan_in(retire)                                                 // the lane's hash: retirements in arm order
// composed fanned out: the consumer runs in parallel
```

| Lane concern | On `PipelineRunner` |
|---|---|
| The item's ordered submission | `try_fan_in(submit)`. Every chain of the pipeline takes the lane's hash (`id_hash == routing_hash`), so submissions run one at a time, in arrival order, on the lane's core, and workers never steal them (`cache.rs:395, 545-549`, `vertex.rs:505-547`; Java `ControlPlaneCache.java:197, 252`). The stage enters the lane's state, checks `is_alive`, awaits the item's markers, runs it, marks it, calls `then`, and arms a retirement if one was asked for. |
| Follow-ups that need no order | `fan_out` stages and the consumer (`compose_fanned_out`) run in parallel across chains. Frames an item makes for others are offered to a source (a `QueueIngestSink`) from the stage, so they are spawned, not run in place. |
| "Not released until ready" | Admission. `run`/`submit` check a chain out and publish it. When the first stage has a hold and the item has an unmet gate, the chain is published to `Completions::after(gate, chain)` instead of the runner's queue. `Completions` only queries, and releases the chain into the lattice once the gate opens. Nothing waits after release. A mark that still finds no room after the gate opened is a failure (`failed`). |
| Retirements | The submit stage marks again after `arm_retirement`, so the second marker is reached only after the retirement callback has run. The hold hands the chain to `Completions::after_marker`. The retire stage, on the lane's core, takes its own ticket with `poll_retired` and calls `retired`. Markers of one lane are reached in order and the core keeps the order, so `retired` calls come in arm order with no `armed` queue. On a faulted device, `completed()` returns the issued value once the device is poisoned, so the chain is released anyway and `poll_retired` reports the fault (`euhedral-gpu/src/cuda/stream.rs:259-270`). An item that armed no retirement has no hold, so its retire stage is a no-op. |
| Kill switch | `PipelineControl::complete` (or the lattice completing the delegate) sets the chains' switch. Queued chains end `Cancelled` at once, and a running stage's chain ends `Cancelled` when it returns. A held chain is skipped when `Completions` emits it: the lane wraps every gate so that it also opens once the lane is killed, and markers are reached as the device drains. `PipelineFrame::do_finally` ends a dead chain and its recycle drops the value, so a dead item's cleanup is its `Drop`. An item that is dead while its lane lives is found by the submit stage, which runs `skipped` in the lane's state and returns `FrameError::Cancelled`. `complete_gracefully` stops admission and completes once every accepted chain has ended. |
| Outcome | `submit` returns an `OutcomeSlot`: `Success`, `Cancelled` (killed or skipped), or `Failed` with the lane's error (`failed` ran). `run` reports nothing and allocates nothing. A closed lane hands the item back in `AdmissionClosed`. |
| Recycling | The runner's `FrameManager` is the `GrpcFrame` pattern: chains are made once, refilled on checkout, and recycled through a bounded MPSC queue drained 256 at a time. No allocation after warm-up, except the `Box<dyn Ready>` of an item that is held. |

**The lane's state.** The submitter and the `LaneState` live in a `SourceCell` that only the lane's `fan_in` stages enter.
`SourceCell` already exists: its checked claim (one uncontended CAS and one store per stage) is what lets an `Fn` stage take
`&mut` safely, and its docs gain one line saying that the ordered stages of one pipeline may hold it too. The rebalance drain
keeps two of those stages apart, as it keeps a source's calls apart (`vertex.rs:974-981`; `LatticeVertex.java:589-593,
621-625`).

**Hashes.** The caller picks each lane's hash and spreads lanes over cores by its choice. A rebalance may still put two lanes
on one core, which is accepted. Nothing in the core places lanes.

## 2. What `PipelineRunner` lacks

| Addition | Why a lane needs it | Java today |
|---|---|---|
| **A chosen id hash.** `PipelineRunner::with_hash(builder, consumer, parallel, partitions, id_hash)`, which passes the hash to the pipeline's `FrameFactory` (`FrameFactory::with_id_hash`). | Lane placement is by hash. Today the factory draws a random id hash (`frame_factory.rs:33`). | Same: `FrameFactory.idHash` is random. Java gets the same constructor. |
| **More than one submitter.** `PipelineRunner::submitter() -> PipelineSubmitter<I>`, a handle with its own `FrameManager` over the same composed pipeline that shares the runner's queue, admission word and kill switch. It has `run`/`submit` through `&mut self`. | Work reaches a lane from any thread and from other lanes' stages. The runner's pool has one consumer. | Same single owner (the `password`-guarded `FrameManager`). Java's gRPC transport solves it the same way: each handler owns its `FrameManager` (`GrpcFrame`). Java gets `submitter()`. |
| **A hold before a stage.** `PipelineBuilder::hold(gate: Fn(&O) -> Option<G>, hold: Fn(G, Box<dyn Frame>))`. When the next stage is scheduled (at admission for the first stage, otherwise in `advance`) and `gate` returns `Some`, the chain goes to `hold` instead of the runner's queue. `hold` must release it into the lattice or finish it. | Waits and retirements. Both are "release this chain when a condition holds", and `Completions` already does the holding. | `PipelineFrame.advance` always offers to the runner's `QueueIngestSink`. Java gets the same builder step. |

It needs nothing else:
- in-order work is `fan_in`;
- unordered work is `fan_out`;
- the kill switch, graceful completion, `OutcomeSlot`, `AdmissionClosed` and recycling exist already;
- retirement polling is `hold` plus `Completions::after_marker`, with no poll frame and no idle backstop.

## 3. What happens to the lane types

| Today | After |
|---|---|
| `LaneSource` (a `LatticeSource`) | Replaced. `open_lanes` builds one `PipelineRunner` per lane (the pipeline in §1, the lane's hash, the `SourceCell`), and the runner's `delegate()` is what the lattice polls. The item pool, ready queue, held queue, `armed` queue, poll frame and `unmarked` retry all go. |
| `LaneIntakes`, `LaneInputs`, `Deliver` | Replaced by submitters. A producer holds one `PipelineSubmitter` per lane. `Deliver::Group` binds at delivery to the group's next lane in turn, as `LanePool` already does for roots. |
| `LaneWork` | Kept as the item trait. It gains `is_alive`/`skipped` (draft PR #47's API) and `gate(&self) -> Option<Box<dyn Ready>>`, built from what the lane publishes. It loses `Outcome::Wait` and `order`: a lane runs its items in arrival order, a held item lets later ones pass, and work that must follow a held item is delivered as its successor. |
| `LaneCx` | Kept as what the submit and retire stages hand an item: the submitter, the state, `now`, the lane, and `deliver`, which runs through the submitters the lane's state holds. `ready` goes: frames are offered to a source. |
| `LaneSubmit`, `LaneState` | Unchanged, except `LaneState::met`, which goes because gates replace it. |
| `LaneMetrics` | `item_ns[kind]`, `intake_to_run_ns` (the delivery stamp travels in the value), and `skipped[kind]`. `service_ns` and `ready_depth` go, and `PipelineRunner::size` gives the queue depth. |

Submitters held in lane states make reference cycles between lanes. Closing a lane (after its runner has completed, so no stage
runs) takes them out of its state, which breaks the cycles, then quiesces the submitter and closes the state as
`Inner::drop` does today (`source.rs:77-113`). Lanes complete before the `Completions` that holds their chains.

## 4. Sources that execute work in `request`/`pull`

| # | Site | What runs in the pass | Replacement |
|---|---|---|---|
| 1 | `LaneSource::serve` (`euhedral-graph/src/lane_source/source.rs:478-533`) | every ready item's `run`: device submissions and whatever else the item does | `PipelineRunner` (this design) |
| 2 | `Moves::emit` → `pump` (`euhedral-memory/src/tiers.rs:1155-1164, 866-906`) | finishes moves, starts second-hop copies (driver calls), frees slots, and allocates a `Vec` per pass | `PipelineRunner`: a move is a chain of `fan_out` stages (first hop, second hop, finish), each held on its copy's retirement with `Completions::after`. The finish stage offers the owner's handler frame, ordered on the owner's hash, to a `QueueIngestSink`. |
| 3 | `AsyncIo` (`euhedral-io/src/lib.rs:917-957`, `uring.rs:623-651`) | reaping (a query, fine), plus an SQE flush (`io_uring_enter`) and `EAGAIN` requeues | `QueueIngestSink`: the pass offers one recycled flush frame per ring while entries wait, and the frame flushes and requeues |
| 4 | `retire_into` → `StreamPending::poll` (`euhedral-gpu/src/retire.rs:102-160`) | `poll_retired`: settles the ticket and frees what the stream kept, inside `Completions`' pass | `PipelineRunner` with a hold on the retirement's marker, as the lane's retire stage |
| 5 | `IdleWatch::poll` (`euhedral-graph/src/lake.rs:562-570`) | a detach under a lock, once per idle period | Neither. The watch keeps its two-load check and hands over one recycled, low-priority detach frame. |
| 6 | `GiveBacks` (`euhedral-memory/src/planner.rs:1205`) | wraps orders (fine), but boxes a frame and an `Arc` per order | `QueueIngestSink`: the order is offered as a frame where it is placed, from the placer's `FrameManager` |
| — | `QueueSource`, `SingleUseSource`, `LatticeHotSource`; `Completions` (queries only); `DeferredCalls` (pushes one key frame) | — | fine |

## 5. Migration

Each step can be merged on its own.

1. **euhedral-core** (and Java): `with_hash`, `submitter()`, `hold`, and the `SourceCell` doc line. Tests:
   `chains_of_a_chosen_hash_run_their_fan_in_stages_one_at_a_time_in_order`,
   `submitters_share_admission_and_the_kill_switch`, `a_held_chain_runs_its_next_stage_only_once_released`,
   `a_killed_held_chain_ends_cancelled_when_released`.
2. **euhedral-graph**: the lane on `PipelineRunner` (§1, §3). Re-pin `tests/lane_source*.rs` and `lane_intake_alloc.rs`.
   Draft PR #47 is superseded.
3. **euhedral-gpu**: `open_lane_sources` becomes `open_lanes(.., hashes)`, and the retirement marker follows
   `arm_retirement`. Re-pin the fake and CUDA lane tests and `bin/idle_polls.rs`. Then row 4.
4. **euhedral-memory**: rows 2 and 6.
5. **euhedral-io**: row 3.
6. **euhedral-graph**: row 5.
