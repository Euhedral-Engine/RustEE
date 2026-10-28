//! CUDA graphs: a recurring sequence recorded on shadow streams, instantiated, and replayed.
//!
//! Recording runs the sequence for real on the streams it is given, and at the same moment captures the same work
//! on one shadow stream per stream, so a real stream is never in capture mode and a failed capture never disturbs
//! the work itself.

use std::sync::{Arc, Mutex, PoisonError};

use euhedral_memory::cuda::{CuEvent, CuStream, STREAM_NON_BLOCKING};

use crate::kept::SharedKept;
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

/// A recording. Dropping it is a shutdown call; replays still in flight when it drops keep the values it holds (see
/// [`Graph::replay`]) and finish.
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
    /// The buffers its copies read, shared with every replay: a replay hands a handle to its stream, which holds it
    /// until the next retirement armed after the replay is confirmed.
    kept: SharedKept,
    /// Launches of one executable are serialized here, whatever streams they come from.
    launch_lock: Mutex<()>,
}

// SAFETY: the graph and its executable belong to the primary context. Every use is a driver call: launches take
// `launch_lock`, and argument updates and re-recording take `&mut self`, so no two calls on one executable overlap.
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
            // The new recording's buffers replace the old ones. A replay of the old recording that is still in flight
            // holds its own handle to them.
            self.kept = rec.kept;
            return Ok(Rerecorded::Updated);
        }
        // Any other outcome leaves the executable as it was (an update that fails does not half apply).
        let mut fresh = gpu.instantiate(rec, new, streams)?;
        std::mem::swap(self, &mut fresh);
        // `fresh` is now the old graph: dropping it frees its executable once its launches in flight end. Replays in
        // flight hold their own handle to its buffers.
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
        let state = stream.state.lock()?;
        self.gpu.poison.check()?;
        let _c = stream.enter()?;
        let _serial = self
            .launch_lock
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        // SAFETY: the executable and a stream of the same device; launches of this executable are serialized by
        // `launch_lock`, and the context is current.
        self.gpu.check("cuGraphLaunch", unsafe {
            (self.gpu.fns.graph_launch)(self.exec, stream.raw)
        })?;
        // The replay reads the recording's buffers when it runs, which may be after this graph dropped: the stream
        // keeps them until the next retirement armed on it after this replay is confirmed.
        if !self
            .kept
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .is_empty()
        {
            stream
                .kept
                .push(state.next_retire, Box::new(self.kept.clone()));
        }
        Ok(())
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
    /// Whether the recording's kept values were handed to the real streams.
    handed_over: bool,
}

impl CudaSequence<'_> {
    /// Hands the values the recording kept to every real stream, to be held until that stream's next retirement is
    /// confirmed. The record pass runs the work for real, and its copies may still be reading those buffers when
    /// the recording ends, fails or is dropped; the graph shares the same values, so they outlive whichever of the
    /// run and the graph ends last. Done while the streams are still locked.
    fn hand_over_kept(&mut self) {
        if self.rec.mode != Mode::Record || self.handed_over {
            return;
        }
        self.handed_over = true;
        if crate::kept::is_empty(&self.rec.kept) {
            return;
        }
        for r in &mut self.real {
            r.stream
                .kept
                .push(r.guard.next_retire, Box::new(self.rec.kept.clone()));
        }
    }

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

    /// `later` waits, on the device, for the work enqueued so far on `earlier`: an event made for the purpose, so the
    /// streams' marker rings are never involved.
    fn link(&mut self, earlier: CuStream, later: CuStream) -> Result<(), GpuError> {
        self.gpu.poison.check()?;
        let event = self.event()?;
        let fns = &self.gpu.device().driver.fns;
        // SAFETY: an event of this sequence and a stream of a locked stream (or its shadow); the context is current.
        self.gpu.check("cuEventRecord", unsafe {
            (fns.event_record)(event, earlier)
        })?;
        // SAFETY: as above.
        self.gpu.check("cuStreamWaitEvent", unsafe {
            (self.gpu.fns.stream_wait_event)(later, event, 0)
        })
    }

    fn raw(&self, index: usize) -> CuStream {
        self.real[index].stream.raw
    }

    fn shadow(&self, index: usize) -> CuStream {
        self.real[index].shadow
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
            self.link(self.shadow(0), self.shadow(other))
                .map_err(|e| GpuError::Capture(e.to_string()))?;
        }
        // The real streams fork the same way, so the real run equals a replay on the first stream.
        for other in 1..self.real.len() {
            self.link(self.raw(0), self.raw(other))?;
        }
        Ok(())
    }

    /// Joins the streams back into the first, ends the capture, and returns the graph. Every step is attempted
    /// whatever failed before it, and the graph is destroyed on any error.
    fn finish(&mut self) -> Result<CuGraph, GpuError> {
        let mut failure: Option<GpuError> = None;
        // The real streams join first: the first stream ends behind all the others, whatever happens to the capture.
        for other in 1..self.real.len() {
            if let Err(e) = self.link(self.raw(other), self.raw(0)) {
                failure.get_or_insert(e);
            }
        }
        // The shadows join even after a failure, so the end of the capture has the best chance to succeed.
        for other in 1..self.real.len() {
            if let Err(e) = self.link(self.shadow(other), self.shadow(0)) {
                failure.get_or_insert(GpuError::Capture(e.to_string()));
            }
        }
        let mut graph: CuGraph = std::ptr::null_mut();
        // SAFETY: the first shadow, which is capturing; an out pointer; the context is current.
        let code = unsafe { (self.gpu.fns.stream_end_capture)(self.real[0].shadow, &mut graph) };
        self.capturing = false;
        if let Err(e) = self.gpu.check("cuStreamEndCapture", code) {
            failure.get_or_insert(GpuError::Capture(e.to_string()));
        }
        match failure {
            None => Ok(graph),
            Some(e) => {
                if !graph.is_null() {
                    // SAFETY: a graph the driver returned and nothing else owns.
                    unsafe { (self.gpu.fns.graph_destroy)(graph) };
                }
                Err(e)
            }
        }
    }
}

impl Drop for CudaSequence<'_> {
    fn drop(&mut self) {
        self.hand_over_kept();
        // A recording the closure never finished (it panicked): end the capture, so the shadows leave capture mode
        // and later recordings work.
        if self.capturing
            && let Ok(graph) = self.finish()
        {
            // SAFETY: a graph the driver returned and nothing else owns.
            unsafe { (self.gpu.fns.graph_destroy)(graph) };
        }
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
            self.link(self.raw(after), self.raw(before))?;
            if self.rec.failed.is_none()
                && let Err(e) = self.link(self.shadow(after), self.shadow(before))
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
            handed_over: false,
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
            handed_over: false,
        };
        let begun = seq.begin();
        let ran = begun.and_then(|()| f(&mut seq));
        let ended = if seq.capturing {
            seq.finish().map(Some)
        } else {
            Ok(None)
        };
        seq.hand_over_kept();
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
            kept: rec.kept,
            launch_lock: Mutex::new(()),
        })
    }
}
