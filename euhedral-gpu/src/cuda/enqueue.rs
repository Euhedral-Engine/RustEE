//! What a CUDA submission enqueues: onto a stream, and while recording also onto the stream's shadow.

use std::any::Any;
use std::ffi::c_void;

use euhedral_memory::cuda::CuStream;

use crate::kept::{Frozen, freeze};
use crate::shape::{CopyKind, ShapeHasher};
use crate::{Args, Enqueue, GpuError, Kernel, Launch};

use super::fns::{CAPTURE_STATUS_ACTIVE, CaptureInfo, CuGraph, CuGraphNode};
use super::{CudaGpu, CudaKernel, CudaStream};

/// Which pass a sequence is running.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Mode {
    /// Run for real, capture the same work on shadow streams, and keep what is needed to update it.
    Record,
    /// Fingerprint only.
    Probe,
    /// Fingerprint, and stage the arguments of each launch.
    Update,
}

/// A graph node handle that moves between threads.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Node(
    #[allow(dead_code, reason = "read by the update pass, which comes next")] pub CuGraphNode,
);

// SAFETY: a node handle is an identifier; the graph it belongs to is only changed through the driver's calls.
unsafe impl Send for Node {}
// SAFETY: as above.
unsafe impl Sync for Node {}

/// What a pass collects.
#[derive(Debug)]
pub(crate) struct Recording {
    pub(crate) mode: Mode,
    /// Each launch: the kernel, the geometry and the arguments, in launch order.
    pub(crate) launches: Vec<(CudaKernel, Launch, Args)>,
    /// The captured node of each launch (a record pass).
    pub(crate) nodes: Vec<Node>,
    pub(crate) hasher: ShapeHasher,
    /// The values the closure kept while recording.
    pub(crate) kept: Vec<Box<dyn Any + Send>>,
    /// The same, frozen once the run ended, to be shared by the streams, the graph and every replay.
    pub(crate) frozen: Frozen,
    /// The arguments of each launch (an update pass).
    pub(crate) staged: Vec<Args>,
    /// The first reason the capture cannot be used. The real work is unaffected.
    pub(crate) failed: Option<GpuError>,
}

impl Recording {
    pub(crate) fn new(mode: Mode) -> Self {
        Self {
            mode,
            launches: Vec::new(),
            nodes: Vec::new(),
            hasher: ShapeHasher::default(),
            kept: Vec::new(),
            frozen: freeze(Vec::new()),
            staged: Vec::new(),
            failed: None,
        }
    }
}

/// The stream an operation goes to, and the shadow that captures it while recording.
#[derive(Debug)]
pub(crate) struct Target<'a> {
    pub(crate) stream: &'a mut CudaStream,
    pub(crate) shadow: Option<CuStream>,
}

/// What a CUDA submission enqueues.
#[derive(Debug)]
pub struct CudaEnqueue<'a> {
    pub(crate) gpu: &'a CudaGpu,
    pub(crate) target: Option<Target<'a>>,
    /// The pass collecting the operations, and the sequence's index of the stream.
    pub(crate) rec: Option<(&'a mut Recording, usize)>,
}

impl<'a> CudaEnqueue<'a> {
    pub(crate) fn on_stream(gpu: &'a CudaGpu, stream: &'a mut CudaStream) -> Self {
        Self {
            gpu,
            target: Some(Target {
                stream,
                shadow: None,
            }),
            rec: None,
        }
    }

    /// Runs `op` on the real stream and, while recording, on its shadow. A failure on the shadow only marks the
    /// capture failed.
    fn run(&mut self, call: &'static str, op: impl Fn(CuStream) -> i32) -> Result<(), GpuError> {
        let Some(target) = &self.target else {
            return Ok(());
        };
        self.gpu.check(call, op(target.stream.raw))?;
        if let (Some(shadow), Some((rec, _))) = (target.shadow, &mut self.rec)
            && rec.failed.is_none()
            && let Err(e) = self.gpu.check(call, op(shadow))
        {
            rec.failed = Some(GpuError::Capture(format!(
                "{call} on the recording stream: {e}"
            )));
        }
        Ok(())
    }

    /// The node the shadow's last launch created: the capture's only current dependency.
    fn capture_node(gpu: &CudaGpu, shadow: CuStream) -> Result<CuGraphNode, GpuError> {
        let (mut status, mut id, mut graph) = (0i32, 0u64, std::ptr::null_mut() as CuGraph);
        let mut deps: *const CuGraphNode = std::ptr::null();
        let mut count = 0usize;
        // SAFETY: out pointers; the shadow is a stream of this device; the context is current.
        let code = unsafe {
            match gpu.fns.capture_info {
                CaptureInfo::V3(info) => {
                    let mut edges: *const c_void = std::ptr::null();
                    info(
                        shadow,
                        &mut status,
                        &mut id,
                        &mut graph,
                        &mut deps,
                        &mut edges,
                        &mut count,
                    )
                }
                CaptureInfo::V2(info) => info(
                    shadow,
                    &mut status,
                    &mut id,
                    &mut graph,
                    &mut deps,
                    &mut count,
                ),
            }
        };
        gpu.check("cuStreamGetCaptureInfo", code)?;
        if status != CAPTURE_STATUS_ACTIVE || count != 1 || deps.is_null() {
            return Err(GpuError::Capture(format!(
                "a launch left the capture in status {status} with {count} dependencies (expected one node)"
            )));
        }
        // SAFETY: the driver reported `count` (one) dependency at `deps`.
        Ok(unsafe { *deps })
    }
}

impl Enqueue for CudaEnqueue<'_> {
    type Kernel = CudaKernel;

    unsafe fn launch(
        &mut self,
        kernel: &CudaKernel,
        launch: Launch,
        args: &Args,
    ) -> Result<(), GpuError> {
        if let Some((rec, stream)) = &mut self.rec {
            rec.hasher
                .launch(*stream, kernel.id(), &launch, args.layout());
            match rec.mode {
                Mode::Record => rec.launches.push((kernel.clone(), launch, args.clone())),
                Mode::Update => rec.staged.push(args.clone()),
                Mode::Probe => {}
            }
        }
        if self.target.is_none() {
            return Ok(());
        }
        if let Some(expected) = kernel.parameter_sizes() {
            let got: Vec<usize> = args.sizes().collect();
            if expected.len() != got.len()
                || expected.iter().zip(&got).any(|(e, g)| *e as usize != *g)
            {
                return Err(GpuError::Args(format!(
                    "{}: expected sizes {expected:?}, got {got:?}",
                    kernel.name()
                )));
            }
        }
        let fns = &self.gpu.fns;
        let mut pointers = args.pointers();
        let (func, params) = (kernel.raw(), pointers.as_mut_ptr());
        // SAFETY: the caller's contract (arguments match the kernel; addresses valid until retired); the pointers
        // address `args`, which outlives the call; the context is current.
        self.run("cuLaunchKernel", |stream| unsafe {
            (fns.launch_kernel)(
                func,
                launch.grid.x,
                launch.grid.y,
                launch.grid.z,
                launch.block.x,
                launch.block.y,
                launch.block.z,
                launch.shared_bytes,
                stream,
                params,
                std::ptr::null_mut(),
            )
        })?;
        // While recording: the node this launch made.
        if let (
            Some(Target {
                shadow: Some(shadow),
                ..
            }),
            Some((rec, _)),
        ) = (&self.target, &mut self.rec)
            && rec.failed.is_none()
        {
            match Self::capture_node(self.gpu, *shadow) {
                Ok(node) => rec.nodes.push(Node(node)),
                Err(e) => rec.failed = Some(e),
            }
        }
        Ok(())
    }

    unsafe fn copy_to_device(
        &mut self,
        dst: u64,
        src: *const u8,
        bytes: usize,
    ) -> Result<(), GpuError> {
        if let Some((rec, stream)) = &mut self.rec {
            rec.hasher
                .copy(*stream, CopyKind::ToDevice, dst, src as u64, bytes as u64);
        }
        let f = self.gpu.device().driver.fns.memcpy_htod_async;
        // SAFETY: the caller's contract (valid ranges untouched until retired); the context is current.
        self.run("cuMemcpyHtoDAsync", |s| unsafe {
            f(dst, src.cast::<c_void>(), bytes, s)
        })
    }

    unsafe fn copy_to_host(
        &mut self,
        dst: *mut u8,
        src: u64,
        bytes: usize,
    ) -> Result<(), GpuError> {
        if let Some((rec, stream)) = &mut self.rec {
            rec.hasher
                .copy(*stream, CopyKind::ToHost, dst as u64, src, bytes as u64);
        }
        let f = self.gpu.device().driver.fns.memcpy_dtoh_async;
        // SAFETY: as above.
        self.run("cuMemcpyDtoHAsync", |s| unsafe {
            f(dst.cast::<c_void>(), src, bytes, s)
        })
    }

    unsafe fn copy_on_device(&mut self, dst: u64, src: u64, bytes: usize) -> Result<(), GpuError> {
        if let Some((rec, stream)) = &mut self.rec {
            rec.hasher
                .copy(*stream, CopyKind::OnDevice, dst, src, bytes as u64);
        }
        let f = self.gpu.device().driver.fns.memcpy_dtod_async;
        // SAFETY: as above.
        self.run("cuMemcpyDtoDAsync", |s| unsafe { f(dst, src, bytes, s) })
    }

    unsafe fn memset(&mut self, dst: u64, value: u8, bytes: usize) -> Result<(), GpuError> {
        if let Some((rec, stream)) = &mut self.rec {
            rec.hasher.memset(*stream, dst, value, bytes as u64);
        }
        let f = self.gpu.device().driver.fns.memset_d8_async;
        // SAFETY: as above.
        self.run("cuMemsetD8Async", |s| unsafe { f(dst, value, bytes, s) })
    }

    fn keep(&mut self, value: Box<dyn Any + Send>) {
        match (&mut self.rec, &mut self.target) {
            // In a recording the value belongs to the graph.
            (Some((rec, _)), _) if rec.mode == Mode::Record => rec.kept.push(value),
            (Some(_), _) => drop(value),
            (None, Some(target)) => {
                let seq = target.stream.next_retire;
                target.stream.kept.push(seq, value);
            }
            (None, None) => drop(value),
        }
    }
}
