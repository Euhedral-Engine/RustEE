//! A stepped, wall-clock-free fake device (feature `fake`), for tests of code built on this crate.
//!
//! Nothing runs until a test steps the device: [`FakeGpu::step`] runs the oldest runnable operation of any stream, so
//! a test decides the interleaving and every run is the same. Fake kernels are host closures registered by name that
//! run on a bounds-checked fake memory; a kernel fault poisons the device once the host observes it (a marker query, a retirement's status, a submission),
//! as a CUDA fault does; work behind it never runs and its markers never complete.
//!
//! Two simplifications, both the worse case for a caller. A wait on a marker of a faulted lane never ends, where a real
//! driver may instead finish the waiting stream's callbacks with an error status. And a submission, a marker or a wait on
//! the poisoned device returns `Poisoned` on first contact, where the CUDA backend returns the driver's `Fault` for the
//! call that met it and `Poisoned` afterwards.
//!
//! The streams of the fake follow the same ownership rules as any backend's: a shared queue and a single-owner
//! submitter, with no lock in the submitter. The simulated *device* (its memory, its per-lane operation logs and its
//! kernel registry) is one state behind a mutex: it stands for hardware in tests, not for a submission path, and a real
//! backend has no such state in the host.

#![allow(
    clippy::disallowed_types,
    reason = "the simulated device is one state behind a mutex: it stands for hardware, not for a submission path"
)]

mod enqueue;
mod graph;
mod lane;
mod module;
mod stream;

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use euhedral_graph::{LaneId, QueueId, RetireCell};

use crate::retire::RetireSink;

use crate::{
    Args, DeviceAttributes, Fingerprint, Gpu, GpuError, GraphHold, Kernel, Launch, POISONED_STATUS,
    Poison, Rerecorded, StreamOptions, StreamPriority,
};

pub use enqueue::FakeEnqueue;
use enqueue::Real;
pub use graph::{FakeGraph, FakeSequence};
pub use lane::{FakeLaneEnqueue, FakeRawStream};
pub use module::{DEFAULT_DYNAMIC_SHARED_BYTES, FakeKernel, FakeModule, MAX_DYNAMIC_SHARED_BYTES};
use stream::Placement;
pub use stream::{FakeQueue, FakeStream};

/// The priority a fake stream of [`StreamPriority::Normal`](crate::StreamPriority::Normal) reports.
pub const FAKE_NORMAL_PRIORITY: i32 = 0;
/// The priority a fake stream of [`StreamPriority::High`](crate::StreamPriority::High) reports: the greatest the fake
/// device offers, numerically lower than the normal one, as a driver's.
pub const FAKE_GREATEST_PRIORITY: i32 = -1;

/// A fake kernel: runs on the fake memory with the launch geometry and arguments.
pub type FakeKernelFn =
    Arc<dyn Fn(&mut FakeMemory, &Launch, &Args) -> Result<(), FakeFault> + Send + Sync>;

/// How a fake kernel fails.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FakeFault {
    /// Code 700.
    IllegalAddress,
    /// Code 710.
    Trap,
}

impl FakeFault {
    fn error(self) -> GpuError {
        let (code, name) = match self {
            FakeFault::IllegalAddress => (700, "ILLEGAL_ADDRESS"),
            FakeFault::Trap => (710, "ASSERT"),
        };
        GpuError::Fault {
            call: "fake kernel",
            code,
            name: name.into(),
        }
    }
}

/// The fake device's memory: allocations by base address.
#[derive(Debug, Default)]
pub struct FakeMemory {
    allocations: BTreeMap<u64, Vec<u8>>,
}

impl FakeMemory {
    fn range(&self, addr: u64, len: usize) -> Result<(u64, usize), FakeFault> {
        let (&base, bytes) = self
            .allocations
            .range(..=addr)
            .next_back()
            .ok_or(FakeFault::IllegalAddress)?;
        let start = usize::try_from(addr - base).map_err(|_| FakeFault::IllegalAddress)?;
        if start.checked_add(len).is_some_and(|end| end <= bytes.len()) {
            Ok((base, start))
        } else {
            Err(FakeFault::IllegalAddress)
        }
    }

    /// `len` bytes at `addr`, inside one allocation, else `IllegalAddress`.
    pub fn bytes(&self, addr: u64, len: usize) -> Result<&[u8], FakeFault> {
        let (base, start) = self.range(addr, len)?;
        Ok(&self.allocations[&base][start..start + len])
    }

    /// As [`bytes`](Self::bytes), writable.
    pub fn bytes_mut(&mut self, addr: u64, len: usize) -> Result<&mut [u8], FakeFault> {
        let (base, start) = self.range(addr, len)?;
        let bytes = self
            .allocations
            .get_mut(&base)
            .ok_or(FakeFault::IllegalAddress)?;
        Ok(&mut bytes[start..start + len])
    }
}

/// What ran, in the order it ran.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FakeOp {
    Launch {
        lane: u16,
        kernel: String,
        /// The launch was a programmatic dependent launch.
        programmatic: bool,
    },
    Copy {
        lane: u16,
    },
    Memset {
        lane: u16,
    },
    Mark {
        lane: u16,
        value: u64,
    },
    /// The lane's later work waited for marker `value` of another lane, and it was reached.
    Await {
        lane: u16,
        value: u64,
    },
    Retire {
        lane: u16,
        seq: u64,
    },
}

/// A queued operation.
pub(crate) enum QOp {
    Launch {
        kernel: FakeKernel,
        launch: Launch,
        args: Args,
    },
    ToDevice {
        dst: u64,
        src: usize,
        bytes: usize,
    },
    ToHost {
        dst: usize,
        src: u64,
        bytes: usize,
    },
    OnDevice {
        dst: u64,
        src: u64,
        bytes: usize,
    },
    OnDevicePitched {
        dst: u64,
        dst_pitch: usize,
        src: u64,
        src_pitch: usize,
        row_bytes: usize,
        rows: usize,
    },
    Memset {
        dst: u64,
        value: u8,
        bytes: usize,
    },
    Mark {
        value: u64,
    },
    /// Waits for `value` on the lane with this global index.
    Await {
        lane: usize,
        value: u64,
    },
    Retire {
        /// The retirement's number on its stream, and the end of the stream's retirement queue the device pushes to.
        seq: u64,
        sink: Arc<RetireSink>,
    },
    /// A retirement armed on a shared lane: the device stores its status in the cell the arm named.
    Notify {
        cell: Arc<RetireCell>,
    },
    /// A time stamp of a timed span's edge: the device's tick when the lane reaches it.
    Stamp {
        id: u64,
    },
}

pub(crate) struct Queued {
    seq: u64,
    /// The lane's id in its pool, for the log and for tickets.
    lane: u16,
    op: QOp,
}

/// Whom a retirement the device reached tells.
pub(crate) enum Told {
    Sink(Arc<RetireSink>),
    Cell(Arc<RetireCell>),
}

pub(crate) struct Lane {
    /// The identity of the stream that owns the lane, stamped on its markers.
    pub(crate) id: QueueId,
    queue: VecDeque<Queued>,
    /// Marks submitted, and marks run.
    pub(crate) marks_enqueued: u64,
    pub(crate) marks_run: u64,
    /// Marks reserved ahead of the work they follow, and not yet enqueued.
    pub(crate) marks_reserved: u64,
    pub(crate) markers: u32,
    priority: StreamPriority,
}

#[derive(Default)]
pub(crate) struct State {
    memory: FakeMemory,
    next_addr: u64,
    seq: u64,
    pub(crate) lanes: Vec<Lane>,
    log: Vec<FakeOp>,
    /// The device's fault, as the hardware has it: work behind it never runs and its markers never complete. The
    /// host does not know until something observes it (see [`FakeGpu::observe_fault`]), as with a driver.
    pub(crate) faulted: Option<GpuError>,
    /// The fake device's clock: one tick (a microsecond) for each operation run. The time stamps of spans are read from it.
    ticks: u64,
    stamps: HashMap<u64, u64>,
    /// The next span edge's id.
    next_stamp: u64,
}

/// The fake device's context accounting: how often the device's context was retained and released, kept by the
/// [`FakeContext`] and so beyond one open. The CUDA backend's counterpart is the driver's primary context: retained by
/// `Device::open`, released when the last holder of the device drops, and reported by
/// `Driver::primary_context_active`.
#[derive(Debug, Default)]
struct ContextRecord {
    retained: AtomicU64,
    released: AtomicU64,
}

/// A fake device's context that can be opened, fully released and opened again, as a real device's primary context can: the
/// record of retains and releases belongs to the context, not to one open, and each [`open`](Self::open) is a [`FakeGpu`] with its own memory,
/// kernels and streams (a new context has none of the last one's). Two opens alive at once get separate memory, where a
/// second retain of a live primary context shares the first's: the fake does not model that overlap.
#[derive(Clone, Debug, Default)]
pub struct FakeContext {
    record: Arc<ContextRecord>,
}

impl FakeContext {
    pub fn new() -> Self {
        Self::default()
    }

    /// Retains the context and opens it: the device is active until the returned gpu and everything that holds it
    /// (streams, modules, graphs) drop.
    pub fn open(&self) -> Arc<FakeGpu> {
        self.record.retained.fetch_add(1, Ordering::AcqRel);
        Arc::new(FakeGpu::over(self.clone()))
    }

    /// A holder of the context that is not a gpu: it retains the context now and releases it when dropped, as a backend
    /// that holds the device itself (the CUDA backend's `PinnedHost`, `DeviceMemory` and `StreamCopier` hold an
    /// `Arc<Device>`, not the gpu) does. The context stays active until it and every opened gpu have gone.
    pub fn hold(&self) -> FakeContextHold {
        self.record.retained.fetch_add(1, Ordering::AcqRel);
        FakeContextHold {
            context: self.clone(),
        }
    }

    /// Contexts retained so far.
    pub fn retained(&self) -> u64 {
        self.record.retained.load(Ordering::Acquire)
    }

    /// Contexts released so far.
    pub fn released(&self) -> u64 {
        self.record.released.load(Ordering::Acquire)
    }

    /// Whether a context is held: retained more often than released.
    pub fn active(&self) -> bool {
        self.retained() > self.released()
    }
}

/// A retain of a [`FakeContext`] made by [`FakeContext::hold`]; the drop releases it.
#[derive(Debug)]
pub struct FakeContextHold {
    context: FakeContext,
}

impl Drop for FakeContextHold {
    fn drop(&mut self) {
        self.context.record.released.fetch_add(1, Ordering::AcqRel);
    }
}

/// What a new [`FakeGpu`] reports as its limits: values for tests, not any card's.
pub const DEFAULT_ATTRIBUTES: DeviceAttributes = DeviceAttributes {
    multiprocessors: 8,
    max_shared_memory_per_block_optin: 48 * 1024,
};

/// The fake device.
pub struct FakeGpu {
    /// The device this was opened on; released when this drops.
    context: FakeContext,
    pub(crate) state: Mutex<State>,
    pub(crate) poison: Poison,
    kernels: Mutex<HashMap<String, FakeKernelFn>>,
    attributes: Mutex<DeviceAttributes>,
    /// A test hook: how the next recording fails (0: it does not; 1: after its work ran; 2: before the body ran).
    fail_recording: std::sync::atomic::AtomicU8,
}

impl Drop for FakeGpu {
    fn drop(&mut self) {
        self.context.record.released.fetch_add(1, Ordering::AcqRel);
    }
}

impl fmt::Debug for FakeGpu {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FakeGpu")
            .field("poisoned", &self.poison.get().is_some())
            .finish_non_exhaustive()
    }
}

impl FakeGpu {
    /// A gpu on a device of its own (see [`FakeContext::open`] to open one device again).
    pub fn new() -> Arc<Self> {
        FakeContext::new().open()
    }

    fn over(context: FakeContext) -> Self {
        Self {
            context,
            state: Mutex::new(State {
                next_addr: 0x1000_0000,
                ..State::default()
            }),
            poison: Poison::new(),
            kernels: Mutex::new(HashMap::new()),
            attributes: Mutex::new(DEFAULT_ATTRIBUTES),
            fail_recording: std::sync::atomic::AtomicU8::new(0),
        }
    }

    /// Sets the limits [`Gpu::attributes`] reports (a new fake reports [`DEFAULT_ATTRIBUTES`]).
    pub fn set_attributes(&self, attributes: DeviceAttributes) {
        *self
            .attributes
            .lock()
            .unwrap_or_else(PoisonError::into_inner) = attributes;
    }

    /// The context this gpu is an open of: it is held until this gpu and everything holding it drop.
    pub fn context(&self) -> &FakeContext {
        &self.context
    }

    /// A test hook: the next recording ends with `Err(Capture)`, as a capture the driver refused does; the work it ran
    /// for real is unaffected.
    pub fn fail_next_recording(&self) {
        self.fail_recording
            .store(1, std::sync::atomic::Ordering::Release);
    }

    /// A test hook: the next recording fails before its body ran, as a capture that cannot start does (the unit has not
    /// run).
    pub fn fail_next_recording_before_the_body(&self) {
        self.fail_recording
            .store(2, std::sync::atomic::Ordering::Release);
    }

    pub(crate) fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Registers the kernel `name`.
    pub fn define(
        &self,
        name: &str,
        f: impl Fn(&mut FakeMemory, &Launch, &Args) -> Result<(), FakeFault> + Send + Sync + 'static,
    ) {
        self.kernels
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(name.to_owned(), Arc::new(f));
    }

    /// A new id for a span edge.
    pub(crate) fn new_stamp(&self) -> u64 {
        let mut st = self.lock();
        st.next_stamp += 1;
        st.next_stamp
    }

    /// The tick at which stamp `id` ran, once it has.
    pub(crate) fn stamp(&self, id: u64) -> Option<u64> {
        self.lock().stamps.get(&id).copied()
    }

    pub(crate) fn kernel_fn(&self, name: &str) -> Option<FakeKernelFn> {
        self.kernels
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get(name)
            .cloned()
    }

    /// The priority the stream with this identity was opened with, as [`FakeStream::priority`] reports it.
    pub fn lane_priority(&self, id: QueueId) -> Option<i32> {
        self.lock()
            .lanes
            .iter()
            .find(|lane| lane.id == id)
            .map(|lane| match lane.priority {
                StreamPriority::Normal => FAKE_NORMAL_PRIORITY,
                StreamPriority::High => FAKE_GREATEST_PRIORITY,
            })
    }

    /// Zeroed device memory of `bytes`, 256-aligned, never at 0.
    pub fn alloc(&self, bytes: usize) -> u64 {
        let mut st = self.lock();
        let base = st.next_addr;
        st.next_addr += (bytes as u64).next_multiple_of(256).max(256) + 256;
        st.memory.allocations.insert(base, vec![0; bytes]);
        base
    }

    /// Frees the allocation at `addr`.
    ///
    /// # Panics
    /// When `addr` is not the base of a live allocation.
    pub fn free(&self, addr: u64) {
        self.lock()
            .memory
            .allocations
            .remove(&addr)
            .expect("the base of a live allocation");
    }

    /// Allocations not yet freed.
    pub fn live_allocations(&self) -> usize {
        self.lock().memory.allocations.len()
    }

    /// Writes host bytes into device memory (for tests).
    ///
    /// # Panics
    /// Outside an allocation.
    pub fn write(&self, addr: u64, bytes: &[u8]) {
        self.lock()
            .memory
            .bytes_mut(addr, bytes.len())
            .expect("inside an allocation")
            .copy_from_slice(bytes);
    }

    /// Reads device memory (for tests).
    ///
    /// # Panics
    /// Outside an allocation.
    pub fn read(&self, addr: u64, len: usize) -> Vec<u8> {
        self.lock()
            .memory
            .bytes(addr, len)
            .expect("inside an allocation")
            .to_vec()
    }

    /// The operations that ran, in order.
    pub fn log(&self) -> Vec<FakeOp> {
        self.lock().log.clone()
    }

    /// What a host call that reaches the driver learns: if the device faulted, the device is poisoned with the fault.
    /// Every call that asks the device (a submission, a marker query, a retirement's status) does this; a test calls it
    /// to stand for such a call.
    pub fn observe_fault(&self) {
        let faulted = self.lock().faulted.clone();
        if let Some(fault) = faulted {
            self.poison.set(fault);
        }
    }

    /// `Err` once the device is poisoned, after observing a fault the device has.
    pub(crate) fn alive(&self) -> Result<(), GpuError> {
        self.observe_fault();
        self.poison.check()
    }

    /// Runs the oldest runnable operation of any stream; false when none is runnable.
    pub fn step(&self) -> bool {
        let retire = {
            let mut st = self.lock();
            let fault = st.faulted.clone();
            let mut best: Option<(u64, usize)> = None;
            for (index, lane) in st.lanes.iter().enumerate() {
                let Some(head) = lane.queue.front() else {
                    continue;
                };
                let runnable = match &head.op {
                    // A mark behind a fault never completes, so a wait for it never ends.
                    QOp::Await { lane, value } => st.lanes[*lane].marks_run >= *value,
                    _ => true,
                };
                if runnable && best.is_none_or(|(seq, _)| head.seq < seq) {
                    best = Some((head.seq, index));
                }
            }
            let Some((_, index)) = best else {
                return false;
            };
            let Queued { lane, op, .. } = st.lanes[index].queue.pop_front().expect("a head");
            self.run(&mut st, index, lane, op, fault.as_ref())
        };
        // The device only queues the outcome, as a driver's callback would; the stream's owner dispatches it.
        match retire {
            Some((Told::Sink(sink), status)) => sink.push(status),
            Some((Told::Cell(cell), status)) => cell.arrive(status),
            None => {}
        }
        true
    }

    /// Runs one operation under the state lock. A retirement's outcome is returned to be queued after the lock is
    /// released.
    fn run(
        &self,
        st: &mut State,
        index: usize,
        lane: u16,
        op: QOp,
        fault: Option<&GpuError>,
    ) -> Option<(Told, i32)> {
        let status = match fault {
            Some(GpuError::Fault { code, .. }) => *code,
            Some(_) => POISONED_STATUS,
            None => 0,
        };
        if fault.is_some() {
            // A faulted device runs no work and its markers never complete (a query of one reports the fault); the
            // callbacks of the stream that faulted are still told, with the fault's code.
            return match op {
                QOp::Retire { seq, sink } => {
                    st.log.push(FakeOp::Retire { lane, seq });
                    Some((Told::Sink(sink), status))
                }
                QOp::Notify { cell } => Some((Told::Cell(cell), status)),
                _ => None,
            };
        }
        st.ticks += 1;
        let outcome: Result<(), FakeFault> = match op {
            QOp::Stamp { id } => {
                let now = st.ticks;
                st.stamps.insert(id, now);
                Ok(())
            }
            QOp::Launch {
                kernel,
                launch,
                args,
            } => {
                st.log.push(FakeOp::Launch {
                    lane,
                    kernel: kernel.name().to_owned(),
                    programmatic: launch.programmatic,
                });
                (kernel.f)(&mut st.memory, &launch, &args)
            }
            QOp::ToDevice { dst, src, bytes } => {
                st.log.push(FakeOp::Copy { lane });
                st.memory.bytes_mut(dst, bytes).map(|to| {
                    // SAFETY: the `Enqueue` contract keeps the `bytes` at `src` valid and unaliased until retired.
                    unsafe {
                        std::ptr::copy_nonoverlapping(src as *const u8, to.as_mut_ptr(), bytes)
                    };
                })
            }
            QOp::ToHost { dst, src, bytes } => {
                st.log.push(FakeOp::Copy { lane });
                st.memory.bytes(src, bytes).map(|from| {
                    // SAFETY: as above, for the host range at `dst`.
                    unsafe { std::ptr::copy_nonoverlapping(from.as_ptr(), dst as *mut u8, bytes) };
                })
            }
            QOp::OnDevice { dst, src, bytes } => {
                st.log.push(FakeOp::Copy { lane });
                st.memory
                    .bytes(src, bytes)
                    .map(<[u8]>::to_vec)
                    .and_then(|data| {
                        st.memory
                            .bytes_mut(dst, bytes)
                            .map(|to| to.copy_from_slice(&data))
                    })
            }
            QOp::OnDevicePitched {
                dst,
                dst_pitch,
                src,
                src_pitch,
                row_bytes,
                rows,
            } => {
                // An empty copy touches no memory and logs nothing, as on a card, where nothing is enqueued.
                if rows == 0 || row_bytes == 0 {
                    return None;
                }
                st.log.push(FakeOp::Copy { lane });
                // The spans were checked at enqueue.
                let span = |pitch: usize| (rows - 1) * pitch + row_bytes;
                st.memory
                    .bytes(src, span(src_pitch))
                    .map(<[u8]>::to_vec)
                    .and_then(|data| {
                        st.memory.bytes_mut(dst, span(dst_pitch)).map(|to| {
                            for row in 0..rows {
                                to[row * dst_pitch..][..row_bytes]
                                    .copy_from_slice(&data[row * src_pitch..][..row_bytes]);
                            }
                        })
                    })
            }
            QOp::Memset { dst, value, bytes } => {
                st.log.push(FakeOp::Memset { lane });
                st.memory.bytes_mut(dst, bytes).map(|to| to.fill(value))
            }
            QOp::Mark { value } => {
                st.log.push(FakeOp::Mark { lane, value });
                st.lanes[index].marks_run = value;
                Ok(())
            }
            QOp::Await { value, .. } => {
                st.log.push(FakeOp::Await { lane, value });
                Ok(())
            }
            QOp::Retire { seq, sink } => {
                st.log.push(FakeOp::Retire { lane, seq });
                return Some((Told::Sink(sink), 0));
            }
            QOp::Notify { cell } => return Some((Told::Cell(cell), 0)),
        };
        if let Err(fault) = outcome {
            st.faulted.get_or_insert(fault.error());
        }
        None
    }

    /// Steps until nothing is runnable; returns how many operations ran.
    pub fn run_until_idle(&self) -> usize {
        let mut n = 0;
        while self.step() {
            n += 1;
        }
        n
    }

    /// Appends an operation to the lane with global index `index`.
    /// Reserves a marker of lane `index` ahead of the work it will follow, so that a full ring is learned before the work
    /// is submitted. `Err(Busy)` when every slot holds a marker that has not run, or is reserved.
    pub(crate) fn reserve_mark(&self, index: usize) -> Result<(), GpuError> {
        let mut st = self.lock();
        let lane = &mut st.lanes[index];
        if lane.marks_enqueued + lane.marks_reserved - lane.marks_run >= u64::from(lane.markers) {
            return Err(GpuError::Busy);
        }
        lane.marks_reserved += 1;
        Ok(())
    }

    /// Enqueues the marker reserved with [`reserve_mark`](Self::reserve_mark), behind the work enqueued so far.
    pub(crate) fn push_reserved_mark(&self, index: usize, lane: u16) -> u64 {
        let mut st = self.lock();
        st.seq += 1;
        let seq = st.seq;
        let entry = &mut st.lanes[index];
        entry.marks_reserved -= 1;
        let value = entry.marks_enqueued + 1;
        entry.marks_enqueued = value;
        entry.queue.push_back(Queued {
            seq,
            lane,
            op: QOp::Mark { value },
        });
        value
    }

    /// Gives a reservation up without a marker.
    pub(crate) fn cancel_reserved_mark(&self, index: usize) {
        self.lock().lanes[index].marks_reserved -= 1;
    }

    /// Makes the next marker of lane `index` in one step: reserved and enqueued under one hold of the device's state, so
    /// threads that mark one lane get distinct numbers. `Err(Busy)` when every slot holds a marker that has not run.
    pub(crate) fn push_mark(&self, index: usize, lane: u16) -> Result<u64, GpuError> {
        self.reserve_mark(index)?;
        Ok(self.push_reserved_mark(index, lane))
    }

    pub(crate) fn push(&self, index: usize, lane: u16, op: QOp) {
        let mut st = self.lock();
        st.seq += 1;
        let seq = st.seq;
        if matches!(op, QOp::Mark { .. }) {
            st.lanes[index].marks_enqueued += 1;
        }
        st.lanes[index].queue.push_back(Queued { seq, lane, op });
    }
}

impl Gpu for FakeGpu {
    type Stream = FakeStream;
    type Module = FakeModule;
    type Kernel = FakeKernel;
    type Graph = FakeGraph;
    type Sequence<'a> = FakeSequence<'a>;

    fn open_lanes_at(
        self: &Arc<Self>,
        first: LaneId,
        names: &[&str],
        options: &[StreamOptions],
    ) -> Result<Vec<FakeStream>, GpuError> {
        if names.is_empty() || usize::from(first.0) + names.len() > 64 {
            return Err(GpuError::Invalid(
                "a pool has 1 to 64 lanes, and the streams of a call fit in it".into(),
            ));
        }
        if options.len() != names.len() {
            return Err(GpuError::Invalid("one options entry per lane name".into()));
        }
        if options.iter().any(|o| o.markers == 0) {
            return Err(GpuError::Invalid("a lane needs at least one marker".into()));
        }
        self.alive()?;
        let ids: Vec<QueueId> = names.iter().map(|_| QueueId::next()).collect();
        let base = {
            let mut st = self.lock();
            let base = st.lanes.len();
            for (id, o) in ids.iter().zip(options) {
                st.lanes.push(Lane {
                    id: *id,
                    queue: VecDeque::new(),
                    marks_enqueued: 0,
                    marks_reserved: 0,
                    marks_run: 0,
                    markers: o.markers,
                    priority: o.priority,
                });
            }
            base
        };
        let count = names.len();
        Ok(names
            .iter()
            .enumerate()
            .map(|(i, name)| {
                let place = Placement {
                    base,
                    first: first.0,
                    index: i,
                    count,
                };
                FakeStream::new(
                    self.clone(),
                    ids[i],
                    name,
                    place,
                    options[i].priority,
                    options[i].spans,
                )
            })
            .collect())
    }

    fn attributes(&self) -> DeviceAttributes {
        *self
            .attributes
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }

    fn poisoned(&self) -> Option<GpuError> {
        self.poison.get()
    }

    fn record<F>(&self, streams: &mut [&mut FakeStream], f: F) -> Result<FakeGraph, GpuError>
    where
        F: FnOnce(&mut FakeSequence<'_>) -> Result<(), GpuError>,
    {
        let real = streams.iter_mut().map(|s| Real::Stream(s)).collect();
        self.record_graph(real, None, f)
    }

    fn record_on<F>(
        &self,
        lanes: &[&FakeQueue],
        hold: &mut GraphHold,
        f: F,
    ) -> Result<FakeGraph, GpuError>
    where
        F: FnOnce(&mut FakeSequence<'_>) -> Result<(), GpuError>,
    {
        let real = lanes.iter().map(|l| Real::Lane(l)).collect();
        self.record_graph(real, Some(hold), f)
    }

    fn rerecord_on<F>(
        &self,
        graph: &mut FakeGraph,
        lanes: &[&FakeQueue],
        hold: &mut GraphHold,
        f: F,
    ) -> Result<Rerecorded, GpuError>
    where
        F: FnOnce(&mut FakeSequence<'_>) -> Result<(), GpuError>,
    {
        let real = lanes.iter().map(|l| Real::Lane(l)).collect();
        self.rerecord_graph(graph, real, Some(hold), f)
    }

    fn rerecord<F>(
        &self,
        graph: &mut FakeGraph,
        streams: &mut [&mut FakeStream],
        f: F,
    ) -> Result<Rerecorded, GpuError>
    where
        F: FnOnce(&mut FakeSequence<'_>) -> Result<(), GpuError>,
    {
        let real = streams.iter_mut().map(|s| Real::Stream(s)).collect();
        self.rerecord_graph(graph, real, None, f)
    }

    fn fingerprint<F>(&self, streams: usize, f: F) -> Result<Fingerprint, GpuError>
    where
        F: FnOnce(&mut FakeSequence<'_>) -> Result<(), GpuError>,
    {
        Ok(self
            .probe_pass(enqueue::Mode::Probe, streams, f)?
            .hasher
            .finish())
    }

    fn update<F>(&self, graph: &mut FakeGraph, f: F) -> Result<(), GpuError>
    where
        F: FnOnce(&mut FakeSequence<'_>) -> Result<(), GpuError>,
    {
        self.update_graph(graph, f)
    }

    fn unchanged<F>(&self, graph: &mut FakeGraph, f: F) -> Result<Option<bool>, GpuError>
    where
        F: FnOnce(&mut FakeSequence<'_>) -> Result<(), GpuError>,
    {
        self.unchanged_graph(graph, f)
    }
}
