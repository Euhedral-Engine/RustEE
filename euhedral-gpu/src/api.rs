//! The backend-neutral traits.
//!
//! - [`Gpu`]: a device. It opens streams as the lanes of a [`LanePool`] and records recurring sequences.
//! - [`Stream`]: the submitter of one ordered device queue. One owner enqueues through `&mut self`; the queue's query
//!   side is a separate, shared [`DeviceQueue`] in the pool.
//! - [`Enqueue`]: what a submission can do: launches, copies, memsets.
//! - [`Module`] and [`Kernel`]: code loaded at run time, from source or from a prebuilt image.
//! - [`Sequence`] and [`Graph`]: a recurring sequence, recorded once and replayed.
//! - [`RetireListener`]: told, once, that the work submitted so far has retired.
//!
//! The traits use associated types, not trait objects, so a backend costs no indirection and no allocation per call.
//!
//! **Ownership replaces locking.** A stream is split in two:
//! - the **queue** (`DeviceQueue`: `id`, `completed`, `name`) is shared and `Sync`, held by the `LanePool` and
//!   `Completions`; it reads atomics and never waits;
//! - the **submitter** (`Stream`) is `Send` and not `Sync`: it has exactly one owner, and everything that enqueues
//!   (`submit`, `mark`, `await_marker`, `notify_retired`, `confirm_retired`, and a graph's `replay`) takes `&mut`.
//!   Exclusivity is the borrow checker's, so there is no lock to take or to wait on, and a call on a stream from
//!   inside its own `submit` closure does not compile.
//!
//! A lane can also be **shared**: [`Stream::shared`] is the handle (a [`Lane`]) that any worker submits to through `&self`,
//! as a stage of a stage graph does. The owner's `&mut` calls and the lane's `&self` calls make markers by the same
//! lock-free protocol, so the two may coexist. A shared lane keeps no values and has no retirement channel: each
//! retirement armed on it names the cell its callback fills ([`LaneControl::notify_retired`]), and what a stage's work
//! reads is held by the stage, released in its `retired` hook.
//!
//! In RustEE's model the owner is a lane's `LatticeSource`: the lattice serializes its `request` and `pull`, so
//! whichever worker runs them is the submitter's only user, and nothing is locked. [`open_lane_sources`](crate::open_lane_sources)
//! opens a device's streams that way; work for a lane from other frames reaches the source through the lock-free
//! intakes of `euhedral-graph`. A chain of frames can instead carry the submitter along, as
//! [`retire_into`](crate::retire_into) does.

use std::any::Any;
use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use euhedral_graph::{DeviceQueue, LaneControl, LaneId, LanePool, Marker, QueueId};

use crate::{Args, Fingerprint, GpuError, GraphHold, Launch, Span};

/// The pool and the streams that [`Gpu::open_lanes`] makes: stream `i` is lane `i` of the pool.
pub type Lanes<G> = (Arc<LanePool>, Vec<<G as Gpu>::Stream>);

/// A device and its backend.
pub trait Gpu: Send + Sync + fmt::Debug + Sized + 'static {
    type Stream: Stream<Gpu = Self>;
    type Module: Module<Gpu = Self>;
    type Kernel: Kernel;
    type Graph: Graph<Gpu = Self>;
    type Sequence<'a>: Sequence<Kernel = Self::Kernel>;

    /// New streams, `names[i]` as lane `i` of a new pool (1 to 64 names), all set up with `options`. The pool holds
    /// each stream's query side; the submitters are returned to be owned, one per lane.
    ///
    /// This is [`open_lanes_at`](Self::open_lanes_at) at lane 0, with the pool made of the streams' queues.
    fn open_lanes(
        self: &Arc<Self>,
        names: &[&str],
        options: StreamOptions,
    ) -> Result<Lanes<Self>, GpuError> {
        let streams = self.open_lanes_at(LaneId(0), names, &vec![options; names.len()])?;
        let queues = streams.iter().map(Stream::queue).collect();
        Ok((Arc::new(LanePool::new(queues)), streams))
    }

    /// New streams for lanes `first..first + names.len()` of a pool built elsewhere
    /// ([`LanePoolBuilder::reserve`](euhedral_graph::LanePoolBuilder::reserve)), `options[i]` for `names[i]`. Their
    /// markers name those lanes; the caller sets each stream's [`queue`](Stream::queue) into the builder at its lane.
    ///
    /// A marker of a lane that this call did not open is refused by [`Stream::await_marker`] (`Err(Invalid)`), as a
    /// marker of any other queue is: a pool of several backends orders across them by handing the marker to the item
    /// that bridges it. Lanes of two calls on one device therefore cannot order on the device; open the lanes that must
    /// order each other in one call.
    ///
    /// `Err(Invalid)` for no names, a different number of options, or lanes past 63.
    fn open_lanes_at(
        self: &Arc<Self>,
        first: LaneId,
        names: &[&str],
        options: &[StreamOptions],
    ) -> Result<Vec<Self::Stream>, GpuError>;

    /// The device's first fault, once it has one.
    fn poisoned(&self) -> Option<GpuError>;

    /// Runs `f` once for real, as a replay on `streams[0]` would, and records it. Every pointer the closure names
    /// (host buffers above all) must stay valid for every replay until the graph drops: the recording replays raw
    /// work. The other streams fork from
    /// `streams[0]` at the start and join it at the end. A failed recording is `Err(Capture)`; the work itself was
    /// submitted and is unaffected.
    fn record<F>(&self, streams: &mut [&mut Self::Stream], f: F) -> Result<Self::Graph, GpuError>
    where
        F: FnOnce(&mut Self::Sequence<'_>) -> Result<(), GpuError>;

    /// Records `f` again into `graph`: in place when the shape allows, else as a new executable.
    fn rerecord<F>(
        &self,
        graph: &mut Self::Graph,
        streams: &mut [&mut Self::Stream],
        f: F,
    ) -> Result<Rerecorded, GpuError>
    where
        F: FnOnce(&mut Self::Sequence<'_>) -> Result<(), GpuError>;

    /// As [`record`](Self::record), over shared lanes, through `&self`: `f` runs once for real, as a replay on `lanes[0]`
    /// would, and is recorded. The other lanes fork from `lanes[0]` at the start and join it at the end.
    ///
    /// A shared lane keeps no values, so what the real run and the recording read (the buffers the closure keeps) go to
    /// the caller: a handle to them is added to `hold`, whatever becomes of the recording, and the caller holds it until
    /// the device work ended (see [`GraphHold`]). A shadow stream that captures the work is made for the call and
    /// destroyed after it.
    fn record_on<F>(
        &self,
        lanes: &[&<Self::Stream as Stream>::Lane],
        hold: &mut GraphHold,
        f: F,
    ) -> Result<Self::Graph, GpuError>
    where
        F: FnOnce(&mut Self::Sequence<'_>) -> Result<(), GpuError>;

    /// As [`rerecord`](Self::rerecord), over shared lanes (see [`record_on`](Self::record_on)).
    fn rerecord_on<F>(
        &self,
        graph: &mut Self::Graph,
        lanes: &[&<Self::Stream as Stream>::Lane],
        hold: &mut GraphHold,
        f: F,
    ) -> Result<Rerecorded, GpuError>
    where
        F: FnOnce(&mut Self::Sequence<'_>) -> Result<(), GpuError>;

    /// The fingerprint `f` would record over `streams` streams. Nothing reaches the device.
    fn fingerprint<F>(&self, streams: usize, f: F) -> Result<Fingerprint, GpuError>
    where
        F: FnOnce(&mut Self::Sequence<'_>) -> Result<(), GpuError>;

    /// New argument values: launch `n` of `f` is launch `n` of the recording. A sequence of another shape is
    /// `Err(ShapeChanged)` and applies nothing.
    fn update<F>(&self, graph: &mut Self::Graph, f: F) -> Result<(), GpuError>
    where
        F: FnOnce(&mut Self::Sequence<'_>) -> Result<(), GpuError>;
}

/// The submitter of one ordered queue of device work, and a lane of a [`LanePool`].
///
/// A stream has one owner: every call that enqueues takes `&mut self`, and the type is `Send` (it can move to another
/// thread, or travel inside a frame) but not `Sync`. In RustEE's model the owner is a `LatticeSource`: the lattice
/// serializes its `request` and `pull`, so the source is the submitter's only user, whichever worker runs it. The pool
/// and `Completions` hold the stream's [`queue`](Self::queue) instead, which is shared and only reads.
///
/// **Dropping a submitter is a shutdown call**: it waits for the stream's work, and every callback it armed, to end. A
/// submitter is dropped in the owning source's lifecycle stop path; a continuation that is done with a stream hands it
/// back to that source instead of dropping it in a frame (see [`retire_into`](crate::retire_into)).
pub trait Stream: Send + Sized + 'static {
    type Gpu: Gpu<Stream = Self>;
    type Enqueue<'a>: Enqueue<Kernel = <Self::Gpu as Gpu>::Kernel>;
    /// The shared handle over this stream's queue: any thread may submit to it through `&self` (see [`Lane`]).
    type Lane: Lane<Gpu = Self::Gpu>;

    fn gpu(&self) -> &Arc<Self::Gpu>;

    fn lane(&self) -> LaneId;

    /// This stream's identity, stamped on its markers and tickets: the same as its queue's.
    fn id(&self) -> QueueId;

    /// The shared query side of this stream, as the pool holds it.
    fn queue(&self) -> Arc<dyn DeviceQueue>;

    /// The shared handle over this stream's queue: the query side of [`queue`](Self::queue) and the submit side of a
    /// [`Lane`], for a [`LanePool`] of lanes that stages submit to from any worker. It keeps the device queue alive: the
    /// stream may be dropped, and the queue is closed (after its work) when the last handle goes.
    fn shared(&self) -> Arc<Self::Lane>;

    /// Runs `f` with the stream selected for every operation; the operations are contiguous on the stream. Keep `f`
    /// to enqueueing. The stream is borrowed for the call, so `f` cannot use it again: a call on the stream from
    /// inside its own `submit` does not compile.
    ///
    /// ```compile_fail,E0499
    /// use euhedral_gpu::fake::FakeGpu;
    /// use euhedral_gpu::{Gpu, Stream, StreamOptions};
    ///
    /// let gpu = FakeGpu::new();
    /// let (_pool, mut streams) = gpu.open_lanes(&["a"], StreamOptions::default()).unwrap();
    /// let stream = &mut streams[0];
    /// stream.submit(|_| stream.mark().map(|_| ())).unwrap();
    /// ```
    fn submit<R>(
        &mut self,
        f: impl FnOnce(&mut Self::Enqueue<'_>) -> Result<R, GpuError>,
    ) -> Result<R, GpuError>;

    /// Waits for the work submitted so far, and the callbacks armed for it, to end. A shutdown aid: the default does
    /// nothing, for a stream whose drop already waits and that has no state made against it to release first.
    fn quiesce(&mut self) {}

    /// A marker after the work submitted so far. `Err(Busy)` (retry; this never waits) when the marker ring is full of
    /// markers that have not completed, or when the slot the marker would reuse is pinned by an awaiter on another
    /// lane.
    fn mark(&mut self) -> Result<Marker, GpuError>;

    /// Later work on this stream waits, on the device, for `marker`. The host does not wait.
    ///
    /// A marker names the queue that made it ([`DeviceQueue::id`]) as well as a lane and a value. A marker of a queue
    /// that is not one of this stream's pool, a lane beyond the pool, or a value that was never made, is
    /// `Err(Invalid)`; it is never treated as already done.
    fn await_marker(&mut self, marker: Marker) -> Result<(), GpuError>;

    /// Arms a retirement after the work submitted so far: `listener` is told once, inside the owning source's
    /// [`poll_retired`](Self::poll_retired) call (on whichever worker runs its `request` or `pull`), in the order the
    /// retirements were armed. The backend's callback only queues the outcome (it makes no driver call, frees and
    /// allocates nothing); the listener is made and dropped inside the owner's calls.
    fn notify_retired(&mut self, listener: Box<dyn RetireListener>) -> Result<(), GpuError>;

    /// Arms a retirement after the work submitted so far, without a listener: its ticket comes back from
    /// [`poll_retired`](Self::poll_retired). This is what [`retire_into`](crate::retire_into) uses.
    fn arm_retirement(&mut self) -> Result<(), GpuError>;

    /// Drains the stream's retirement queue, which the backend's callbacks fill. Each listener of an arrived
    /// retirement is told here, in order; the first retirement armed with [`arm_retirement`](Self::arm_retirement)
    /// returns its ticket. `None` when nothing more arrived. Once the device is poisoned, the retirements still
    /// outstanding are reported with [`POISONED_STATUS`], so nothing is stranded.
    fn poll_retired(&mut self) -> Option<RetireTicket>;

    /// How many retirements armed with [`arm_retirement`](Self::arm_retirement) have not returned their ticket from
    /// [`poll_retired`](Self::poll_retired) yet. [`retire_into`](crate::retire_into) refuses a stream with any
    /// outstanding: their tickets would reach its continuation in place of its own.
    fn outstanding_tickets(&self) -> usize;

    /// Opens a timed span: the device time of the work submitted from here to the matching [`span_end`](Self::span_end).
    /// The span's start is recorded behind the work submitted so far. `Err(Busy)` (retry; this never waits) when every
    /// span of the stream's ring is open or unread. A stream opened with no spans refuses.
    fn span_begin(&mut self) -> Result<Span, GpuError> {
        Err(GpuError::Invalid("this stream keeps no timed spans".into()))
    }

    /// Closes `span`: its end is recorded behind the work submitted since [`span_begin`](Self::span_begin).
    fn span_end(&mut self, span: Span) -> Result<(), GpuError> {
        let _ = span;
        Err(GpuError::Invalid("this stream keeps no timed spans".into()))
    }

    /// The duration of a closed span, once its end was reached; `None` while the work is still to run. It reads the end
    /// once and never waits. A read span is freed: its slot serves another, and the handle is stale.
    fn span_poll(&mut self, span: Span) -> Result<Option<Duration>, GpuError> {
        let _ = span;
        Err(GpuError::Invalid("this stream keeps no timed spans".into()))
    }

    /// Gives up a span (an item that was killed between its begin and end, or whose time is not wanted): its slot is
    /// freed at once. A span the device may still be recording into is safe to free: its events are recorded again
    /// before the slot's next use.
    fn span_cancel(&mut self, span: Span) -> Result<(), GpuError> {
        let _ = span;
        Err(GpuError::Invalid("this stream keeps no timed spans".into()))
    }

    /// Settles a retirement: the device failure behind it, or `None`. Never panics. Values kept for the retirement are
    /// released. A ticket armed on another stream is refused with `Some(Invalid)` and releases nothing.
    ///
    /// This is a method of the submitter, which owns the kept values, so the continuation that follows a retirement
    /// owns the stream too (see [`retire_into`](crate::retire_into)): nothing is shared, and nothing is locked or
    /// queued to release the values.
    fn confirm_retired(&mut self, ticket: RetireTicket) -> Option<GpuError>;
}

/// The shared submit side of one ordered queue of device work: a lane that any worker submits to through `&self`, as in
/// Java's `GpuStream`, where a stage submits to its lane directly from whichever worker runs it.
///
/// A lane is a [`DeviceQueue`] too, so a [`LanePool`] of lanes (typed over the lane, as Java's `LanePool` holds
/// streams) is both what [`Completions`](euhedral_graph::Completions) polls and what a stage reaches through
/// [`StageCx::lane_handle`](euhedral_graph::StageCx::lane_handle). It is a [`LaneControl`] as well: it orders its later
/// work behind another lane's marker and arms retirements that name their sink, which is what a stage graph uses to
/// join its lanes and retire a quantum.
///
/// - **Submit and mark are lock-free.** Submissions are the driver's: it serializes the launches it gets on one stream.
///   A marker is claimed *before* the work it follows ([`submit_marked`](Self::submit_marked), Java opens a stage's
///   marker before the stage submits) and recorded after it; the marker records after all the work submitted so far,
///   which may include another thread's work (that only over-waits). The numbers are handed out by compare-and-set and
///   found complete in order by [`DeviceQueue::completed`].
/// - **Nothing waits, and a refusal comes before the work.** A marker ring that is full, or a slot an awaiter holds,
///   answers `Err(Busy)` when the marker is claimed, before anything was submitted: a stage that is told to retry has
///   submitted nothing. Once the work is in, the marker is recorded and cannot be refused.
/// - **Retirement** names its sink: [`LaneControl::notify_retired`] takes the [`RetireCell`](euhedral_graph::RetireCell)
///   the driver callback stores the status in (Java `notifyRetired(listener)`). The callback only fills the cell; the
///   continuation that finds it filled settles the status with [`LaneControl::confirm_retired`] in a frame, on an
///   ordinary thread, never in the callback and never in a source's `request` or `pull`. Nothing is claimed or queued
///   in between, so two parties retiring on one lane never see each other's status.
/// - **A shared lane keeps no values.** What a submission's work reads (a host buffer a copy reads) lives in the
///   work's own quantum and is released by the stage's retirement hook ([`Stage::retired`](euhedral_graph::Stage::retired),
///   Java `StageFrame.retired`), once the quantum's boundary was reached.
pub trait Lane: DeviceQueue + LaneControl + Sized + 'static {
    type Gpu: Gpu;
    type Enqueue<'a>: LaneEnqueue<Kernel = <Self::Gpu as Gpu>::Kernel> + LaneStream
    where
        Self: 'a;
    /// What [`submit_held`](Self::submit_held) enqueues: an [`Enqueue`] whose kept values go to a [`GraphHold`].
    type Held<'a>: Enqueue<Kernel = <Self::Gpu as Gpu>::Kernel> + LaneStream
    where
        Self: 'a;

    /// This lane's id in its pool.
    fn lane(&self) -> LaneId;

    /// Runs `f` with the lane selected for every operation, on the calling thread: the operations are contiguous in this
    /// call's submissions to the driver, though another thread's may interleave between two calls.
    fn submit<R>(
        &self,
        f: impl FnOnce(&mut Self::Enqueue<'_>) -> Result<R, GpuError>,
    ) -> Result<R, GpuError>;

    /// Claims a marker, runs `f` as [`submit`](Self::submit) does, and records the marker after `f`'s work: Java's
    /// stage, which has its marker before it submits. `Err(Busy)` (retry) only when the marker could not be claimed
    /// (the ring is full of markers that have not completed, or an awaiter holds the slot), and then `f` has not run. An
    /// `Err` from `f` gives the marker up.
    fn submit_marked<R>(
        &self,
        f: impl FnOnce(&mut Self::Enqueue<'_>) -> Result<R, GpuError>,
    ) -> Result<(R, Marker), GpuError>;

    /// As [`submit`](Self::submit), for a body written against [`Enqueue`] (a recurring sequence's body): the values the
    /// body keeps (`Enqueue::keep`) are added to `hold`, since a shared lane keeps none. The caller holds them until the
    /// work ended.
    fn submit_held<R>(
        &self,
        hold: &mut GraphHold,
        f: impl FnOnce(&mut Self::Held<'_>) -> Result<R, GpuError>,
    ) -> Result<R, GpuError>;

    /// A marker after the work submitted so far. `Err(Busy)` (retry) as for [`submit_marked`](Self::submit_marked). Use
    /// that for a marker after work you cannot repeat.
    fn mark(&self) -> Result<Marker, GpuError>;
}

/// What a submission to a shared lane enqueues: the operations of [`Enqueue`], without `keep`. A shared lane keeps no
/// values; see [`Lane`].
pub trait LaneEnqueue {
    type Kernel: Kernel;

    /// Launches `kernel`.
    ///
    /// # Safety
    /// `args` match the kernel's parameters, and every address the kernel touches is valid for that access until the
    /// work retires.
    unsafe fn launch(
        &mut self,
        kernel: &Self::Kernel,
        launch: Launch,
        args: &Args,
    ) -> Result<(), GpuError>;

    /// Copies `bytes` from host memory at `src` to the device address `dst`.
    ///
    /// # Safety
    /// The ranges are valid, not used by other work in the meantime, and stay valid until the work retires.
    unsafe fn copy_to_device(
        &mut self,
        dst: u64,
        src: *const u8,
        bytes: usize,
    ) -> Result<(), GpuError>;

    /// Copies `bytes` from the device address `src` to host memory at `dst`.
    ///
    /// # Safety
    /// As [`copy_to_device`](Self::copy_to_device).
    unsafe fn copy_to_host(&mut self, dst: *mut u8, src: u64, bytes: usize)
    -> Result<(), GpuError>;

    /// Copies `bytes` between device addresses.
    ///
    /// # Safety
    /// As [`copy_to_device`](Self::copy_to_device).
    unsafe fn copy_on_device(&mut self, dst: u64, src: u64, bytes: usize) -> Result<(), GpuError>;

    /// Sets `bytes` at the device address `dst` to `value`.
    ///
    /// # Safety
    /// As [`copy_to_device`](Self::copy_to_device).
    unsafe fn memset(&mut self, dst: u64, value: u8, bytes: usize) -> Result<(), GpuError>;
}

/// A driver stream lent for the closure of a submission, for library calls that enqueue on a stream of their own accord. It
/// borrows the enqueue handle it came from and is `!Send`, so it
/// cannot outlive the closure or leave the calling thread:
///
/// ```compile_fail,E0521
/// use euhedral_gpu::fake::FakeGpu;
/// use euhedral_gpu::{Lane, LaneStream, LaneEnqueue, open_shared_lanes, StreamOptions};
///
/// let gpu = FakeGpu::new();
/// let lanes = open_shared_lanes(&gpu, &[("a", StreamOptions::default())]).unwrap();
/// let lane = lanes.pool.lane(euhedral_graph::LaneId(0));
/// let mut kept = None;
/// // the loan cannot be kept past the closure
/// lane.submit(|q| { kept = q.raw_stream(); Ok(()) }).unwrap();
/// ```
///
/// ```compile_fail,E0277
/// use euhedral_gpu::fake::FakeGpu;
/// use euhedral_gpu::{Lane, LaneStream, open_shared_lanes, StreamOptions};
///
/// let gpu = FakeGpu::new();
/// let lanes = open_shared_lanes(&gpu, &[("a", StreamOptions::default())]).unwrap();
/// let lane = lanes.pool.lane(euhedral_graph::LaneId(0));
/// // nor leave the calling thread
/// lane.submit(|q| {
///     let raw = q.raw_stream().unwrap();
///     std::thread::scope(|s| { s.spawn(move || drop(raw)); });
///     Ok(())
/// }).unwrap();
/// ```
#[derive(Debug)]
pub struct RawStream<'a, R> {
    raw: R,
    /// Borrowed from the handle, and not `Send` or `Sync`.
    _lent: std::marker::PhantomData<(&'a (), *const ())>,
}

impl<R: Copy> RawStream<'_, R> {
    pub(crate) fn new(raw: R) -> Self {
        Self {
            raw,
            _lent: std::marker::PhantomData,
        }
    }

    /// The backend\'s stream handle (CUDA: `CuStream`), to pass to the library call.
    pub fn as_raw(&self) -> R {
        self.raw
    }
}

/// The driver stream behind a submission, for library calls that enqueue on a stream (Java `GpuStream.submit(Runnable)`,
/// which runs the library calls with the stream selected).
///
/// **The lent stream is private to the closure.** On a shared lane it is not the lane's own stream but one forked from it
/// for this submission: ordered after everything enqueued so far, and the closure's later operations go to it too, so what
/// follows the library call follows its work; when the closure ends the lane waits for it, and the marker is recorded after
/// that. The lane's stream is never lent because a library may call `cuGraphLaunch` (or another call that behaves like it)
/// internally, and with several threads submitting to one stream a `cuGraphLaunch` can overtake the `cuStreamWaitEvent` its
/// own thread issued just before it (driver 615.71.09, CUDA 13.4), so the library's work would run before what it waited for.
///
/// The loan is a [`RawStream`]: the borrow checker keeps it inside the closure and on the calling thread. What the
/// library does with the stream is outside Rust's view, so the contract of the call is the caller's:
/// - Enqueue only work that completes on its own on the stream: do not synchronize the stream, destroy it, or capture it.
/// - The device context is the one the backend selected for the closure; a call that switches it restores it.
/// - Every pointer the enqueued work reads or writes stays valid until the work retires (hold it in the quantum, as for a
///   copy).
///
/// Library calls are not part of a recording: in a pass of a recurring sequence (probe, update, record) there is no stream
/// to lend, `raw_stream` is `None`, and the pass notes the request ([`Sequence::library_stream_requested`]). A caller never
/// lets that skip the call: [`SharedGraphCache`](crate::SharedGraphCache) finds the request out before it records or
/// replays, makes the key unrecordable, and runs the unit directly, every time, with the call made. An owned stream has a
/// single submitter, so it lends its own stream.
pub trait LaneStream: LaneEnqueue {
    /// The backend\'s stream handle.
    type Raw: Copy + fmt::Debug;

    /// The stream this submission enqueues on, lent for as long as `self` is borrowed.
    fn raw_stream(&self) -> Option<RawStream<'_, Self::Raw>>;
}

/// What a submission on an owned stream, or into a recurring sequence's passes, enqueues: the operations of
/// [`LaneEnqueue`] (a kernel body written once against that trait serves a shared lane's submission, an owned stream's
/// and every pass of a recurring sequence), [`LaneStream`] (the driver stream for library calls, where there is one), and
/// the values it keeps alive for its work.
pub trait Enqueue: LaneStream {
    /// Keeps `value` alive until the next retirement armed on this stream is confirmed. Host buffers that a copy
    /// reads belong here. On a shared lane, which keeps none, the value goes to the caller's [`GraphHold`]
    /// ([`Lane::submit_held`]).
    ///
    /// In a recording the value belongs to the graph. A recorded copy reads its host buffer again at every replay, so
    /// every pointer a recording names must stay valid for every replay until the graph drops, and a host buffer must
    /// be kept. Each replay hands the graph's values to its stream, which holds them until the next retirement armed
    /// after the replay is confirmed, so a replay in flight survives the graph dropping.
    fn keep(&mut self, value: Box<dyn Any + Send>);

    /// As [`keep`](Self::keep), making the value only when it is kept. A recurring sequence's body runs in every pass,
    /// but only a recording (and a direct run) keeps its values: an update pass drops them. A body that makes its value
    /// here allocates it once, when it is recorded, and not at every replay.
    fn keep_with<T: Any + Send>(&mut self, make: impl FnOnce() -> T)
    where
        Self: Sized,
    {
        self.keep(Box::new(make()));
    }
}

/// A function of a loaded module.
pub trait Kernel: Clone + Send + Sync + fmt::Debug {
    fn name(&self) -> &str;

    /// Identifies the kernel in a shape fingerprint.
    fn id(&self) -> u64;
}

/// Code loaded at run time. Loading is a long call: it blocks (compile, file reads, the driver's load), so it runs in
/// the frame or item that needs it, on the worker that reaches it (a worker is a CPU), and not per item.
pub trait Module: Send + Sync + fmt::Debug + Sized {
    type Gpu: Gpu<Module = Self>;

    /// Compiles `source` for the device (cached when `options.cache_dir` is set) and loads it.
    fn from_source(
        gpu: &Arc<Self::Gpu>,
        source: &str,
        options: &CompileOptions,
    ) -> Result<Self, GpuError>;

    /// Loads a prebuilt image (PTX, cubin or fatbin for CUDA).
    fn from_image(gpu: &Arc<Self::Gpu>, image: &[u8]) -> Result<Self, GpuError>;

    /// The kernel named `name` (its unmangled symbol).
    fn kernel(&self, name: &str) -> Result<<Self::Gpu as Gpu>::Kernel, GpuError>;
}

/// A recurring sequence of submissions over a fixed set of streams, run as a recording, a probe or an update.
pub trait Sequence {
    type Kernel: Kernel;
    type Enqueue<'a>: Enqueue<Kernel = Self::Kernel>;

    /// Submits on stream `stream` of the sequence (an index into the streams it runs over).
    fn submit(
        &mut self,
        stream: usize,
        f: impl FnOnce(&mut Self::Enqueue<'_>) -> Result<(), GpuError>,
    ) -> Result<(), GpuError>;

    /// Later work on `before` waits for the work submitted so far on `after`.
    fn order(&mut self, after: usize, before: usize) -> Result<(), GpuError>;

    /// Whether a body asked for the driver stream ([`LaneStream::raw_stream`]) in this pass. A pass has no stream to lend
    /// (what ran on it would not be recorded, and an update or probe touches no device), so the request was answered
    /// `None`: a body that does library work cannot be replayed, and a caller that sees this runs it directly (see
    /// [`SharedGraphCache`](crate::SharedGraphCache)).
    fn library_stream_requested(&self) -> bool;
}

/// A recorded sequence.
pub trait Graph: Send + fmt::Debug + 'static {
    type Gpu: Gpu<Graph = Self>;

    fn fingerprint(&self) -> Fingerprint;

    /// How many launches the recording holds.
    fn launches(&self) -> usize;

    /// Replays the recording as one submission on `stream`, with the arguments set by the last update.
    ///
    /// Takes `&mut self`: a replay has exclusive use of the graph's executable, and the owner of the graph decides
    /// which stream replays it. A graph shared by several lanes is one graph per lane (record it again), or one owner
    /// that hands it from lane to lane.
    ///
    /// The values the recording keeps (`Enqueue::keep`) are handed to `stream`, which holds them until the next
    /// retirement armed on it after this call is confirmed: arm one per replay (or per batch of replays), or they
    /// are held until the stream drops. The graph may drop while its replays are in flight.
    fn replay(&mut self, stream: &mut <Self::Gpu as Gpu>::Stream) -> Result<(), GpuError>;

    /// Replays the recording as one submission on a shared lane, through `&self` of the lane, with the arguments set by
    /// the last update.
    ///
    /// The replay orders itself behind `waits` (markers of other lanes) on the device, then launches; the lane's later
    /// work and its next marker follow it. On CUDA it launches on a private stream forked from the lane and joined back,
    /// never on the lane itself: with several threads submitting to one stream, a `cuGraphLaunch` can overtake the
    /// `cuStreamWaitEvent` its own thread issued just before it (see `SharedGraphCache`).
    ///
    /// The graph is still used by one thread at a time (`&mut self`): a key that several workers run at once has an
    /// instance of its graph for each (see [`SharedGraphCache`](crate::SharedGraphCache)). The instance may be updated
    /// and replayed again as soon as this returns: a launch's parameters are fixed when it is issued, and the launches of
    /// one executable are ordered behind each other. An instance dropped while its launches are in flight is freed by the
    /// driver once they end.
    ///
    /// The values the recording keeps are not given to the lane, which keeps none: a handle to them is added to `hold`,
    /// and the caller holds it until the work ended (in its quantum, released in `Stage::retired`).
    fn replay_on(
        &mut self,
        lane: &<<Self::Gpu as Gpu>::Stream as Stream>::Lane,
        waits: &[Marker],
        hold: &mut GraphHold,
    ) -> Result<(), GpuError>;
}

/// Told once that a retirement happened.
///
/// The listener is made, told and dropped inside the owning source's calls, in [`Stream::poll_retired`]; the backend's
/// callback thread never touches it. So it may do what the source may do in its `request` or `pull`, and it runs
/// there: keep it short.
pub trait RetireListener: Send + 'static {
    /// Runs once, inside the owning source's [`Stream::poll_retired`].
    fn retired(self: Box<Self>, ticket: RetireTicket);
}

/// The status of a retirement that never reached the device: the device was poisoned first.
pub const POISONED_STATUS: i32 = -1;

/// A retirement: which stream armed it, the retirement's number on it, and how the work before it ended.
///
/// Only a backend makes tickets. A ticket settles only on the stream that armed it
/// ([`Stream::confirm_retired`]); a ticket of another stream is refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RetireTicket {
    stream: QueueId,
    lane: LaneId,
    seq: u64,
    status: i32,
}

impl RetireTicket {
    pub(crate) fn new(stream: QueueId, lane: LaneId, seq: u64, status: i32) -> Self {
        Self {
            stream,
            lane,
            seq,
            status,
        }
    }

    /// The identity ([`DeviceQueue::id`]) of the stream that armed the retirement.
    pub fn stream(&self) -> QueueId {
        self.stream
    }

    /// The lane of the stream that armed the retirement.
    pub fn lane(&self) -> LaneId {
        self.lane
    }

    /// The retirement's number on its stream, counting the retirements armed on it.
    pub fn seq(&self) -> u64 {
        self.seq
    }

    /// 0 when the work before it succeeded; a backend error code, or [`POISONED_STATUS`].
    pub fn status(&self) -> i32 {
        self.status
    }

    pub fn ok(&self) -> bool {
        self.status == 0
    }
}

/// How the device schedules a stream's work against other streams' (a device property: the driver decides what it
/// does with the hint).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum StreamPriority {
    /// The driver's default priority.
    #[default]
    Normal,
    /// The greatest priority the device offers: its waiting blocks are scheduled ahead of a normal stream's.
    High,
}

/// How a stream is set up.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StreamOptions {
    /// Markers a stream can have outstanding: the size of its event ring.
    pub markers: u32,
    /// The stream's priority on the device.
    pub priority: StreamPriority,
    /// Timed spans a stream can have open or unread at once: the size of its span ring (two timing events each).
    /// 0: none.
    pub spans: u32,
}

impl Default for StreamOptions {
    fn default() -> Self {
        Self {
            markers: 256,
            priority: StreamPriority::Normal,
            spans: 16,
        }
    }
}

/// How a kernel source is compiled.
#[derive(Clone, Debug, Default)]
pub struct CompileOptions {
    /// The program's name in logs ("module.cu" when empty).
    pub name: String,
    /// Compiler options, passed as given; the backend adds the architecture option.
    pub options: Vec<String>,
    /// Directories searched for includes. Their files are not part of the cache key.
    pub include_paths: Vec<PathBuf>,
    /// In-memory headers `(include name, contents)`; part of the cache key.
    pub headers: Vec<(String, String)>,
    /// Where compiled modules are cached. `None`: no cache.
    pub cache_dir: Option<PathBuf>,
    /// The run-time compiler library to load, else the backend's search order.
    pub compiler: Option<PathBuf>,
    /// Compile for the architecture-specific target of the device (`sm_XYa`, for example `sm_120a`), so kernels
    /// that use instructions of one architecture variant compile. The module runs on that architecture only. When
    /// the compiler does not list the device's exact architecture, the compile is an error; there is no fallback to
    /// the generic target or to PTX. The target is part of the cache key.
    pub arch_specific: bool,
}

/// How [`Gpu::rerecord`] made the new recording executable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rerecorded {
    /// The existing executable took the new recording in place.
    Updated,
    /// A new executable was instantiated.
    Instantiated,
}
