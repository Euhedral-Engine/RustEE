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

use euhedral_graph::{DeviceQueue, LaneId, LanePool, Marker, QueueId};

use crate::{Args, Fingerprint, GpuError, Launch, Span};

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

    fn gpu(&self) -> &Arc<Self::Gpu>;

    fn lane(&self) -> LaneId;

    /// This stream's identity, stamped on its markers and tickets: the same as its queue's.
    fn id(&self) -> QueueId;

    /// The shared query side of this stream, as the pool holds it.
    fn queue(&self) -> Arc<dyn DeviceQueue>;

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

/// What a submission enqueues.
pub trait Enqueue {
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

    /// Keeps `value` alive until the next retirement armed on this stream is confirmed. Host buffers that a copy
    /// reads belong here.
    ///
    /// In a recording the value belongs to the graph. A recorded copy reads its host buffer again at every replay, so
    /// every pointer a recording names must stay valid for every replay until the graph drops, and a host buffer must
    /// be kept. Each replay hands the graph's values to its stream, which holds them until the next retirement armed
    /// after the replay is confirmed, so a replay in flight survives the graph dropping.
    fn keep(&mut self, value: Box<dyn Any + Send>);
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
}

/// A recorded sequence.
pub trait Graph: Send + fmt::Debug {
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
}

/// How [`Gpu::rerecord`] made the new recording executable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rerecorded {
    /// The existing executable took the new recording in place.
    Updated,
    /// A new executable was instantiated.
    Instantiated,
}
