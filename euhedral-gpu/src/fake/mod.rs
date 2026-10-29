//! A stepped, wall-clock-free fake device (feature `fake`), for tests of code built on this crate.
//!
//! Nothing runs until a test steps the device: [`FakeGpu::step`] runs the oldest runnable operation of any stream, so
//! a test decides the interleaving and every run is the same. Fake kernels are host closures registered by name that
//! run on a bounds-checked fake memory; a kernel fault poisons the device as a CUDA fault does.
//!
//! The streams of the fake follow the same ownership rules as any backend's: a shared queue and a single-owner
//! submitter, with no lock in the submitter. The simulated *device* (its memory, its per-lane operation logs and its
//! kernel registry) is one state behind a mutex: it stands for hardware in tests, not for a submission path, and a real
//! backend has no such state in the host.

mod enqueue;
mod graph;
mod module;
mod stream;

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use euhedral_graph::{LanePool, QueueId};

use crate::retire::RetireSink;

use crate::{
    Args, Fingerprint, Gpu, GpuError, Kernel, Launch, POISONED_STATUS, Poison, Rerecorded,
    StreamOptions,
};

pub use enqueue::FakeEnqueue;
pub use graph::{FakeGraph, FakeSequence};
pub use module::{FakeKernel, FakeModule};
pub use stream::{FakeQueue, FakeStream};

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
    Launch { lane: u16, kernel: String },
    Copy { lane: u16 },
    Memset { lane: u16 },
    Mark { lane: u16, value: u64 },
    Retire { lane: u16, seq: u64 },
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
}

pub(crate) struct Queued {
    seq: u64,
    /// The lane's id in its pool, for the log and for tickets.
    lane: u16,
    op: QOp,
}

pub(crate) struct Lane {
    /// The identity of the stream that owns the lane, stamped on its markers.
    pub(crate) id: QueueId,
    queue: VecDeque<Queued>,
    /// Marks submitted, and marks run.
    pub(crate) marks_enqueued: u64,
    pub(crate) marks_run: u64,
    pub(crate) markers: u32,
}

#[derive(Default)]
pub(crate) struct State {
    memory: FakeMemory,
    next_addr: u64,
    seq: u64,
    pub(crate) lanes: Vec<Lane>,
    log: Vec<FakeOp>,
}

/// The fake device.
pub struct FakeGpu {
    pub(crate) state: Mutex<State>,
    pub(crate) poison: Poison,
    kernels: Mutex<HashMap<String, FakeKernelFn>>,
}

impl fmt::Debug for FakeGpu {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FakeGpu")
            .field("poisoned", &self.poison.get().is_some())
            .finish_non_exhaustive()
    }
}

impl FakeGpu {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(State {
                next_addr: 0x1000_0000,
                ..State::default()
            }),
            poison: Poison::new(),
            kernels: Mutex::new(HashMap::new()),
        })
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

    pub(crate) fn kernel_fn(&self, name: &str) -> Option<FakeKernelFn> {
        self.kernels
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get(name)
            .cloned()
    }

    /// Zeroed device memory of `bytes`, 256-aligned, never at 0.
    pub fn alloc(&self, bytes: usize) -> u64 {
        let mut st = self.lock();
        let base = st.next_addr;
        st.next_addr += (bytes as u64).next_multiple_of(256).max(256) + 256;
        st.memory.allocations.insert(base, vec![0; bytes]);
        base
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

    /// Runs the oldest runnable operation of any stream; false when none is runnable.
    pub fn step(&self) -> bool {
        let fault = self.poison.get();
        let retire = {
            let mut st = self.lock();
            let mut best: Option<(u64, usize)> = None;
            for (index, lane) in st.lanes.iter().enumerate() {
                let Some(head) = lane.queue.front() else {
                    continue;
                };
                let runnable = match &head.op {
                    QOp::Await { lane, value } => {
                        fault.is_some() || st.lanes[*lane].marks_run >= *value
                    }
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
        if let Some((sink, status)) = retire {
            sink.push(status);
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
    ) -> Option<(Arc<RetireSink>, i32)> {
        let status = match fault {
            Some(GpuError::Fault { code, .. }) => *code,
            Some(_) => POISONED_STATUS,
            None => 0,
        };
        if fault.is_some() {
            // A faulted device runs no work: its marks are still reached and its retirements still told.
            return match op {
                QOp::Mark { value } => {
                    st.lanes[index].marks_run = value;
                    None
                }
                QOp::Retire { seq, sink } => {
                    st.log.push(FakeOp::Retire { lane, seq });
                    Some((sink, status))
                }
                _ => None,
            };
        }
        let outcome: Result<(), FakeFault> = match op {
            QOp::Launch {
                kernel,
                launch,
                args,
            } => {
                st.log.push(FakeOp::Launch {
                    lane,
                    kernel: kernel.name().to_owned(),
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
            QOp::Memset { dst, value, bytes } => {
                st.log.push(FakeOp::Memset { lane });
                st.memory.bytes_mut(dst, bytes).map(|to| to.fill(value))
            }
            QOp::Mark { value } => {
                st.log.push(FakeOp::Mark { lane, value });
                st.lanes[index].marks_run = value;
                Ok(())
            }
            QOp::Await { .. } => Ok(()),
            QOp::Retire { seq, sink } => {
                st.log.push(FakeOp::Retire { lane, seq });
                return Some((sink, 0));
            }
        };
        if let Err(fault) = outcome {
            self.poison.set(fault.error());
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

    fn open_lanes(
        self: &Arc<Self>,
        names: &[&str],
        options: StreamOptions,
    ) -> Result<crate::Lanes<Self>, GpuError> {
        if names.is_empty() || names.len() > 64 {
            return Err(GpuError::Invalid("a pool has 1 to 64 lanes".into()));
        }
        self.poison.check()?;
        let ids: Vec<QueueId> = names.iter().map(|_| QueueId::next()).collect();
        let base = {
            let mut st = self.lock();
            let base = st.lanes.len();
            for id in &ids {
                st.lanes.push(Lane {
                    id: *id,
                    queue: VecDeque::new(),
                    marks_enqueued: 0,
                    marks_run: 0,
                    markers: options.markers,
                });
            }
            base
        };
        let count = names.len();
        let streams: Vec<FakeStream> = names
            .iter()
            .enumerate()
            .map(|(i, name)| FakeStream::new(self.clone(), ids[i], name, base, i, count))
            .collect();
        // The pool holds the shared query sides; the submitters go to their owners.
        let queues = streams
            .iter()
            .map(|s| s.queue_arc() as Arc<dyn euhedral_graph::DeviceQueue>)
            .collect();
        Ok((Arc::new(LanePool::new(queues)), streams))
    }

    fn poisoned(&self) -> Option<GpuError> {
        self.poison.get()
    }

    fn record<F>(&self, streams: &mut [&mut FakeStream], f: F) -> Result<FakeGraph, GpuError>
    where
        F: FnOnce(&mut FakeSequence<'_>) -> Result<(), GpuError>,
    {
        self.record_graph(streams, f)
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
        self.rerecord_graph(graph, streams, f)
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
}
