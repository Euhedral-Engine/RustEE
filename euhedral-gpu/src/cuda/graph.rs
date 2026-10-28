//! CUDA graphs: a recurring sequence recorded on shadow streams, instantiated, and replayed.
//!
//! Recording runs the sequence for real on the streams it is given, and at the same moment captures the same work
//! on one shadow stream per stream, so a real stream is never in capture mode and a failed capture never disturbs
//! the work itself.

use std::any::Any;
use std::sync::{Arc, Mutex, PoisonError};

use euhedral_memory::cuda::{CuEvent, CuStream, STREAM_NON_BLOCKING};

use crate::stream_lock::StreamGuard;
use crate::{Args, Fingerprint, GpuError, Graph, Launch, Rerecorded, Sequence};

use super::enqueue::{Mode, Node, Recording, Target};
use super::fns::{
    CAPTURE_MODE_RELAXED, CuGraph, CuGraphExec, ExecUpdateResultInfo, KernelNodeParams,
};
use super::stream::{Handle, StreamState};
use super::{CudaEnqueue, CudaGpu, CudaKernel, CudaStream, from_cuda};

/// `CU_EVENT_DISABLE_TIMING`.
const EVENT_DISABLE_TIMING: u32 = 2;

/// A recording. Drop it after its last replay has retired: it is a shutdown call.
#[derive(Debug)]
pub struct CudaGraph {
    gpu: Arc<CudaGpu>,
    graph: CuGraph,
    exec: CuGraphExec,
    nodes: Vec<Node>,
    launches: Vec<(CudaKernel, Launch, Args)>,
    fingerprint: Fingerprint,
    /// The number of streams the recording ran over.
    streams: usize,
    /// Held for the graph's life: the buffers its copies read.
    #[allow(dead_code, reason = "held so the buffers outlive the replays")]
    kept: Mutex<Vec<Box<dyn Any + Send>>>,
}

// SAFETY: the graph and its executable belong to the primary context; every use is a driver call, and the
// executable is replayed through `&self` only by calls the driver orders.
unsafe impl Send for CudaGraph {}
// SAFETY: as above.
unsafe impl Sync for CudaGraph {}

impl Drop for CudaGraph {
    fn drop(&mut self) {
        let Ok(_c) = self.gpu.device().enter() else {
            return;
        };
        // SAFETY: the executable and graph made in `instantiate` and `capture`, destroyed once; the driver frees the
        // executable once its launches in flight end.
        unsafe {
            (self.gpu.fns.graph_exec_destroy)(self.exec);
            (self.gpu.fns.graph_destroy)(self.graph);
        }
    }
}

impl CudaGraph {
    pub(crate) fn gpu(&self) -> &Arc<CudaGpu> {
        &self.gpu
    }

    /// Takes a new recording: into the existing executable when the driver can update it, else as a new executable.
    /// An update keeps the original graph and its node handles, which name the executable's nodes.
    pub(crate) fn replace(
        &mut self,
        rec: Recording,
        new: CuGraph,
        streams: usize,
    ) -> Result<Rerecorded, GpuError> {
        let gpu = self.gpu.clone();
        let _c = gpu.device().enter().map_err(from_cuda)?;
        let mut info = ExecUpdateResultInfo {
            result: 0,
            error_node: std::ptr::null_mut(),
            error_from_node: std::ptr::null_mut(),
        };
        // SAFETY: the executable and a graph of the same device; an out pointer; the context is current.
        let code = unsafe { (gpu.fns.graph_exec_update)(self.exec, new, &mut info) };
        if code == 0 && info.result == 0 {
            // SAFETY: the new graph is no longer needed: the executable took its parameters.
            unsafe { (gpu.fns.graph_destroy)(new) };
            self.launches = rec.launches;
            self.fingerprint = rec.hasher.finish();
            self.streams = streams;
            // The old recording's buffers stay: replays of it may be in flight.
            self.kept
                .get_mut()
                .unwrap_or_else(PoisonError::into_inner)
                .extend(rec.kept);
            return Ok(Rerecorded::Updated);
        }
        // Any other outcome leaves the executable as it was (an update that fails does not half apply).
        let mut fresh = gpu.instantiate(rec, new, streams)?;
        std::mem::swap(self, &mut fresh);
        let old_kept = fresh.kept.get_mut().unwrap_or_else(PoisonError::into_inner);
        let kept = std::mem::take(old_kept);
        self.kept
            .get_mut()
            .unwrap_or_else(PoisonError::into_inner)
            .extend(kept);
        // `fresh` is now the old graph: dropping it frees its executable once its launches in flight end.
        Ok(Rerecorded::Instantiated)
    }

    /// Runs `f` as an update pass: new argument values, launch `n` of `f` to launch `n` of the recording. A sequence
    /// of another shape is `Err(ShapeChanged)` and applies nothing.
    pub(crate) fn update<F>(&mut self, f: F) -> Result<(), GpuError>
    where
        F: FnOnce(&mut CudaSequence<'_>) -> Result<(), GpuError>,
    {
        let gpu = self.gpu.clone();
        let rec = gpu.probe(Mode::Update, self.streams, f)?;
        let now = rec.hasher.finish();
        if now != self.fingerprint || rec.staged.len() != self.launches.len() {
            return Err(GpuError::ShapeChanged {
                recorded: self.fingerprint,
                now,
            });
        }
        let _c = gpu.device().enter().map_err(from_cuda)?;
        for (n, staged) in rec.staged.into_iter().enumerate() {
            let (kernel, launch, args) = &mut self.launches[n];
            if *args == staged {
                continue;
            }
            let mut pointers = staged.pointers();
            let params = KernelNodeParams {
                func: kernel.raw(),
                grid_x: launch.grid.x,
                grid_y: launch.grid.y,
                grid_z: launch.grid.z,
                block_x: launch.block.x,
                block_y: launch.block.y,
                block_z: launch.block.z,
                shared_bytes: launch.shared_bytes,
                kernel_params: pointers.as_mut_ptr(),
                extra: std::ptr::null_mut(),
                kern: std::ptr::null_mut(),
                ctx: std::ptr::null_mut(),
            };
            // SAFETY: the executable and the node it was instantiated with; the parameters (copied by the driver
            // during the call) point into `staged`; the context is current.
            gpu.check("cuGraphExecKernelNodeSetParams", unsafe {
                (gpu.fns.graph_exec_kernel_node_set_params)(self.exec, self.nodes[n].0, &params)
            })?;
            *args = staged;
        }
        Ok(())
    }
}

impl Graph for CudaGraph {
    type Gpu = CudaGpu;

    fn fingerprint(&self) -> Fingerprint {
        self.fingerprint
    }

    fn launches(&self) -> usize {
        self.launches.len()
    }

    fn replay(&self, stream: &CudaStream) -> Result<(), GpuError> {
        let _guard = stream.state.lock()?;
        self.gpu.poison.check()?;
        let _c = stream.enter()?;
        // SAFETY: the executable and a stream of the same device; the context is current.
        self.gpu.check("cuGraphLaunch", unsafe {
            (self.gpu.fns.graph_launch)(self.exec, stream.raw)
        })
    }
}

/// One stream of a record pass: held for the whole closure.
#[derive(Debug)]
pub(crate) struct Real<'a> {
    stream: &'a CudaStream,
    guard: StreamGuard<'a, StreamState>,
    shadow: CuStream,
}

/// A pass over a recurring sequence.
#[derive(Debug)]
pub struct CudaSequence<'a> {
    gpu: &'a CudaGpu,
    pub(crate) rec: Recording,
    streams: usize,
    pub(crate) real: Vec<Real<'a>>,
    /// Events made for the capture, destroyed when it ends.
    events: Vec<CuEvent>,
    capturing: bool,
}

impl CudaSequence<'_> {
    /// A test hook: marks the capture failed, as a refused operation would. The sequence's work still runs for
    /// real, and recording ends with `Err(Capture)`.
    #[doc(hidden)]
    pub fn fail_capture(&mut self) {
        if self.rec.failed.is_none() {
            self.rec.failed = Some(GpuError::Capture("failed by the test hook".into()));
        }
    }

    fn event(&mut self) -> Result<CuEvent, GpuError> {
        let mut event = std::ptr::null_mut();
        // SAFETY: an out pointer; the context is current.
        let code = unsafe {
            (self.gpu.device().driver.fns.event_create)(&mut event, EVENT_DISABLE_TIMING)
        };
        self.gpu.check("cuEventCreate", code)?;
        self.events.push(event);
        Ok(event)
    }

    /// `after`'s shadow records an event that `before`'s shadow waits on.
    fn shadow_order(&mut self, after: usize, before: usize) -> Result<(), GpuError> {
        let event = self.event()?;
        let fns = &self.gpu.device().driver.fns;
        // SAFETY: an event of this sequence and the shadow of a locked stream; the context is current.
        self.gpu.check("cuEventRecord", unsafe {
            (fns.event_record)(event, self.real[after].shadow)
        })?;
        // SAFETY: as above.
        self.gpu.check("cuStreamWaitEvent", unsafe {
            (self.gpu.fns.stream_wait_event)(self.real[before].shadow, event, 0)
        })
    }

    /// Starts the capture on the first shadow and forks the others (and the real streams) from it.
    fn begin(&mut self) -> Result<(), GpuError> {
        let fns = &self.gpu.fns;
        // SAFETY: a shadow stream of this device, not in capture; the context is current.
        self.gpu.check("cuStreamBeginCapture", unsafe {
            (fns.stream_begin_capture)(self.real[0].shadow, CAPTURE_MODE_RELAXED)
        })?;
        self.capturing = true;
        for other in 1..self.real.len() {
            self.shadow_order(0, other)
                .map_err(|e| GpuError::Capture(e.to_string()))?;
        }
        // The real streams fork the same way, so the real run equals a replay on the first stream.
        if self.real.len() > 1 {
            let marker = self.real[0].stream.mark_held()?;
            for r in &self.real[1..] {
                r.stream.await_held(r.stream.raw, marker)?;
            }
        }
        Ok(())
    }

    /// Joins the streams back into the first, ends the capture, and returns the graph.
    fn finish(&mut self) -> Result<CuGraph, GpuError> {
        let mut real = Ok(());
        for other in 1..self.real.len() {
            real = real.and_then(|()| {
                let marker = self.real[other].stream.mark_held()?;
                let first = &self.real[0].stream;
                first.await_held(first.raw, marker)
            });
        }
        // The shadows join even after a failure, so the end of the capture has the best chance to succeed.
        let mut shadows = Ok(());
        for other in 1..self.real.len() {
            let joined = self.shadow_order(other, 0);
            if shadows.is_ok() {
                shadows = joined;
            }
        }
        let mut graph: CuGraph = std::ptr::null_mut();
        // SAFETY: the first shadow, which is capturing; an out pointer; the context is current.
        let code = unsafe { (self.gpu.fns.stream_end_capture)(self.real[0].shadow, &mut graph) };
        self.capturing = false;
        let ended = self.gpu.check("cuStreamEndCapture", code);
        real?;
        if let Err(e) = shadows.and(ended) {
            if !graph.is_null() {
                // SAFETY: a graph the driver returned and nothing else owns.
                unsafe { (self.gpu.fns.graph_destroy)(graph) };
            }
            return Err(GpuError::Capture(e.to_string()));
        }
        Ok(graph)
    }
}

impl Drop for CudaSequence<'_> {
    fn drop(&mut self) {
        let f = &self.gpu.device().driver.fns;
        for event in self.events.drain(..) {
            // SAFETY: events made by `event`, unused after the capture ended.
            unsafe { (f.event_destroy)(event) };
        }
    }
}

impl Sequence for CudaSequence<'_> {
    type Kernel = CudaKernel;
    type Enqueue<'a> = CudaEnqueue<'a>;

    fn submit(
        &mut self,
        stream: usize,
        f: impl FnOnce(&mut CudaEnqueue<'_>) -> Result<(), GpuError>,
    ) -> Result<(), GpuError> {
        if stream >= self.streams {
            return Err(GpuError::Invalid(format!(
                "stream {stream} is not one of the sequence's {}",
                self.streams
            )));
        }
        let record = self.rec.mode == Mode::Record;
        let target = self.real.get_mut(stream).map(|r| Target {
            stream: r.stream,
            state: &mut r.guard,
            shadow: record.then_some(r.shadow),
        });
        let mut q = CudaEnqueue {
            gpu: self.gpu,
            target,
            rec: Some((&mut self.rec, stream)),
        };
        f(&mut q)
    }

    fn order(&mut self, after: usize, before: usize) -> Result<(), GpuError> {
        if after >= self.streams || before >= self.streams {
            return Err(GpuError::Invalid(format!(
                "order({after}, {before}) names a stream outside the sequence's {}",
                self.streams
            )));
        }
        if after == before {
            return Ok(());
        }
        self.rec.hasher.order(after, before);
        if self.rec.mode == Mode::Record {
            let marker = self.real[after].stream.mark_held()?;
            let waiter = &self.real[before].stream;
            waiter.await_held(waiter.raw, marker)?;
            if self.rec.failed.is_none()
                && let Err(e) = self.shadow_order(after, before)
            {
                self.rec.failed = Some(GpuError::Capture(e.to_string()));
            }
        }
        Ok(())
    }
}

impl CudaGpu {
    /// Runs `f` as a pass that touches no device: a fingerprint, or the staged arguments of an update.
    pub(crate) fn probe<F>(&self, mode: Mode, streams: usize, f: F) -> Result<Recording, GpuError>
    where
        F: FnOnce(&mut CudaSequence<'_>) -> Result<(), GpuError>,
    {
        self.poison.check()?;
        let mut seq = CudaSequence {
            gpu: self,
            rec: Recording::new(mode),
            streams,
            real: Vec::new(),
            events: Vec::new(),
            capturing: false,
        };
        f(&mut seq)?;
        Ok(std::mem::replace(&mut seq.rec, Recording::new(Mode::Probe)))
    }

    /// Runs `f` for real over `streams` and captures it: the recording, and the graph the driver made of it.
    pub(crate) fn capture<F>(
        self: &Arc<Self>,
        streams: &[&Arc<CudaStream>],
        f: F,
    ) -> Result<(Recording, CuGraph), GpuError>
    where
        F: FnOnce(&mut CudaSequence<'_>) -> Result<(), GpuError>,
    {
        self.poison.check()?;
        if streams.is_empty() {
            return Err(GpuError::Invalid("a recording needs a stream".into()));
        }
        for (i, a) in streams.iter().enumerate() {
            if streams[..i].iter().any(|b| Arc::ptr_eq(a, b)) {
                return Err(GpuError::Invalid("a stream listed twice".into()));
            }
        }
        // Lock in ascending lane order, so two recordings never wait on each other crosswise.
        let mut order: Vec<usize> = (0..streams.len()).collect();
        order.sort_by_key(|&i| (streams[i].lane_id(), Arc::as_ptr(streams[i]) as usize));
        let mut guards: Vec<Option<StreamGuard<'_, StreamState>>> =
            (0..streams.len()).map(|_| None).collect();
        for i in order {
            guards[i] = Some(streams[i].state.lock()?);
        }
        let _c = self.device().enter().map_err(from_cuda)?;
        let mut real = Vec::with_capacity(streams.len());
        for (stream, guard) in streams.iter().zip(guards) {
            let Some(mut guard) = guard else {
                return Err(GpuError::Invalid("a stream was not locked".into()));
            };
            let shadow = match guard.shadow {
                Some(shadow) => shadow.0,
                None => {
                    let mut raw = std::ptr::null_mut();
                    // SAFETY: an out pointer; the context is current.
                    let code = unsafe {
                        (self.device().driver.fns.stream_create)(&mut raw, STREAM_NON_BLOCKING)
                    };
                    self.check("cuStreamCreate", code)?;
                    guard.shadow = Some(Handle(raw));
                    raw
                }
            };
            real.push(Real {
                stream,
                guard,
                shadow,
            });
        }
        let mut seq = CudaSequence {
            gpu: self,
            rec: Recording::new(Mode::Record),
            streams: real.len(),
            real,
            events: Vec::new(),
            capturing: false,
        };
        let begun = seq.begin();
        let ran = begun.and_then(|()| f(&mut seq));
        let ended = if seq.capturing {
            seq.finish().map(Some)
        } else {
            Ok(None)
        };
        let mut rec = std::mem::replace(&mut seq.rec, Recording::new(Mode::Probe));
        drop(seq);
        // The real work was submitted whatever came of the capture.
        match (ran, ended) {
            (Err(e), Ok(Some(graph))) => {
                // SAFETY: a graph the driver returned and nothing else owns.
                unsafe { (self.fns.graph_destroy)(graph) };
                Err(e)
            }
            (Err(e), _) => Err(e),
            (Ok(()), Err(e)) => Err(e),
            (Ok(()), Ok(None)) => Err(GpuError::Capture("the capture never began".into())),
            (Ok(()), Ok(Some(graph))) => {
                if let Some(e) = rec.failed.take() {
                    // SAFETY: as above.
                    unsafe { (self.fns.graph_destroy)(graph) };
                    return Err(e);
                }
                if rec.nodes.len() != rec.launches.len() {
                    // SAFETY: as above.
                    unsafe { (self.fns.graph_destroy)(graph) };
                    return Err(GpuError::Capture(format!(
                        "{} launches but {} captured nodes",
                        rec.launches.len(),
                        rec.nodes.len()
                    )));
                }
                Ok((rec, graph))
            }
        }
    }

    /// Instantiates `graph` and wraps both with the recording's launches.
    pub(crate) fn instantiate(
        self: &Arc<Self>,
        rec: Recording,
        graph: CuGraph,
        streams: usize,
    ) -> Result<CudaGraph, GpuError> {
        let _c = self.device().enter().map_err(from_cuda)?;
        let mut exec: CuGraphExec = std::ptr::null_mut();
        // SAFETY: an out pointer and a graph the driver made; the context is current.
        let code = unsafe { (self.fns.graph_instantiate)(&mut exec, graph, 0) };
        if let Err(e) = self.check("cuGraphInstantiateWithFlags", code) {
            // SAFETY: a graph nothing else owns.
            unsafe { (self.fns.graph_destroy)(graph) };
            return Err(GpuError::Capture(e.to_string()));
        }
        Ok(CudaGraph {
            gpu: self.clone(),
            graph,
            exec,
            nodes: rec.nodes,
            launches: rec.launches,
            fingerprint: rec.hasher.finish(),
            streams,
            kept: Mutex::new(rec.kept),
        })
    }
}
