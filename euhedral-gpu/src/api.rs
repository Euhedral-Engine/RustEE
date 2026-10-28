//! The backend-neutral traits.
//!
//! - [`Gpu`]: a device. It opens streams as the lanes of a [`LanePool`] and records recurring sequences.
//! - [`Stream`]: an ordered device queue ([`DeviceQueue`]). Submissions run on the calling thread and only enqueue.
//! - [`Enqueue`]: what a submission can do: launches, copies, memsets.
//! - [`Module`] and [`Kernel`]: code loaded at run time, from source or from a prebuilt image.
//! - [`Sequence`] and [`Graph`]: a recurring sequence, recorded once and replayed.
//! - [`RetireListener`]: told, once, that the work submitted so far has retired.
//!
//! The traits use associated types, not trait objects, so a backend costs no indirection and no allocation per call.

use std::any::Any;
use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;

use euhedral_graph::{DeviceQueue, LaneId, LanePool, Marker, QueueId};

use crate::{Args, Fingerprint, GpuError, Launch};

/// The pool and the streams that [`Gpu::open_lanes`] makes: stream `i` is lane `i` of the pool.
pub type Lanes<G> = (Arc<LanePool>, Vec<Arc<<G as Gpu>::Stream>>);

/// A device and its backend.
pub trait Gpu: Send + Sync + fmt::Debug + Sized + 'static {
    type Stream: Stream<Gpu = Self>;
    type Module: Module<Gpu = Self>;
    type Kernel: Kernel;
    type Graph: Graph<Gpu = Self>;
    type Sequence<'a>: Sequence<Kernel = Self::Kernel>;

    /// New streams, `names[i]` as lane `i` of a new pool (1 to 64 names).
    fn open_lanes(
        self: &Arc<Self>,
        names: &[&str],
        options: StreamOptions,
    ) -> Result<Lanes<Self>, GpuError>;

    /// The device's first fault, once it has one.
    fn poisoned(&self) -> Option<GpuError>;

    /// Runs `f` once for real, as a replay on `streams[0]` would, and records it. Every pointer the closure names
    /// (host buffers above all) must stay valid for every replay until the graph drops: the recording replays raw
    /// work. The other streams fork from
    /// `streams[0]` at the start and join it at the end. A failed recording is `Err(Capture)`; the work itself was
    /// submitted and is unaffected.
    fn record<F>(&self, streams: &[&Arc<Self::Stream>], f: F) -> Result<Self::Graph, GpuError>
    where
        F: FnOnce(&mut Self::Sequence<'_>) -> Result<(), GpuError>;

    /// Records `f` again into `graph`: in place when the shape allows, else as a new executable.
    fn rerecord<F>(
        &self,
        graph: &mut Self::Graph,
        streams: &[&Arc<Self::Stream>],
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

/// An ordered queue of device work and a lane of a [`LanePool`].
pub trait Stream: DeviceQueue + Sized + 'static {
    type Gpu: Gpu<Stream = Self>;
    type Enqueue<'a>: Enqueue<Kernel = <Self::Gpu as Gpu>::Kernel>;

    fn gpu(&self) -> &Arc<Self::Gpu>;

    fn lane(&self) -> LaneId;

    /// Runs `f` on the calling thread with the stream selected for every operation; the operations are contiguous
    /// on the stream. Keep `f` to enqueueing. A call on this stream from inside `f` is `Err(Invalid)`.
    fn submit<R>(
        &self,
        f: impl FnOnce(&mut Self::Enqueue<'_>) -> Result<R, GpuError>,
    ) -> Result<R, GpuError>;

    /// A marker after the work submitted so far. `Err(Busy)` when the marker ring is full of markers that have not
    /// completed.
    fn mark(&self) -> Result<Marker, GpuError>;

    /// Later work on this stream waits, on the device, for `marker`. The host does not wait.
    ///
    /// A marker names the queue that made it ([`DeviceQueue::id`]) as well as a lane and a value. A marker of a queue
    /// that is not one of this stream's pool, a lane beyond the pool, or a value that was never made, is
    /// `Err(Invalid)`; it is never treated as already done.
    fn await_marker(&self, marker: Marker) -> Result<(), GpuError>;

    /// Arms a retirement after the work submitted so far: `listener` runs once.
    fn notify_retired(&self, listener: Box<dyn RetireListener>) -> Result<(), GpuError>;

    /// Settles a retirement on an ordinary thread: the device failure behind it, or `None`. Never panics. Values
    /// kept for the retirement are released. A ticket armed on another stream is refused with `Some(Invalid)` and
    /// releases nothing.
    fn confirm_retired(&self, ticket: RetireTicket) -> Option<GpuError>;
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

/// Code loaded at run time. Loading is a build-time call: it blocks (compile, file reads, the driver's load), so it
/// runs at start-up or on a lattice set aside for blocking work.
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
pub trait Graph: Send + Sync + fmt::Debug {
    type Gpu: Gpu<Graph = Self>;

    fn fingerprint(&self) -> Fingerprint;

    /// How many launches the recording holds.
    fn launches(&self) -> usize;

    /// Replays the recording as one submission on `stream`, with the arguments set by the last update.
    ///
    /// The values the recording keeps (`Enqueue::keep`) are handed to `stream`, which holds them until the next
    /// retirement armed on it after this call is confirmed: arm one per replay (or per batch of replays), or they
    /// are held until the stream drops. The graph may drop while its replays are in flight.
    fn replay(&self, stream: &<Self::Gpu as Gpu>::Stream) -> Result<(), GpuError>;
}

/// Told once that a retirement happened.
///
/// The listener is consumed and dropped on the backend's callback thread. It must not own the last handle to a
/// stream, a graph or a device (dropping those synchronizes with the device, which the callback must never do): hold
/// only plain data, atomics and queue handles, as [`retire_into`](crate::retire_into)'s listener does.
pub trait RetireListener: Send + 'static {
    /// Runs once, on the backend's callback thread: enqueue only. No driver calls, no blocking, no frames built or
    /// dropped.
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

/// How a stream is set up.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StreamOptions {
    /// Markers a stream can have outstanding: the size of its event ring.
    pub markers: u32,
}

impl Default for StreamOptions {
    fn default() -> Self {
        Self { markers: 256 }
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
