//! A CUDA stream as a shared lane: any worker submits to it through `&self`, as a stage of a graph does.
//!
//! The state workers share is atomic (the marker ring, see `ring`); the driver serializes the launches it gets on the
//! stream. A retirement names the cell its callback fills (see `retire`), so nothing is claimed or queued.

use std::ffi::c_void;
use std::sync::Arc;

use euhedral_graph::{ControlError, LaneControl, LaneId, Marker, RetireCell};

use crate::{
    Args, GpuError, GraphHold, Kernel, Lane, LaneEnqueue, LaneStream, Launch, MAX_ARGS,
    POISONED_STATUS, RawStream,
};

use euhedral_memory::cuda::CuStream;

use super::launch::{issue, supported};
use super::stream::{CudaQueue, LaneFork};
use super::{CudaEnqueue, CudaGpu, CudaKernel, from_cuda};

/// What a submission to a shared CUDA lane enqueues.
#[derive(Debug)]
pub struct CudaLaneEnqueue<'a> {
    queue: &'a CudaQueue,
    /// The private stream lent to library calls, if one was (see `LaneFork`).
    fork: &'a LaneFork<'a>,
}

impl CudaLaneEnqueue<'_> {
    fn gpu(&self) -> &CudaGpu {
        &self.queue.gpu
    }
}

impl LaneEnqueue for CudaLaneEnqueue<'_> {
    type Kernel = CudaKernel;

    unsafe fn launch(
        &mut self,
        kernel: &CudaKernel,
        launch: Launch,
        args: &Args,
    ) -> Result<(), GpuError> {
        if let Some(expected) = kernel.parameter_sizes()
            && (expected.len() != args.len()
                || expected
                    .iter()
                    .zip(args.sizes())
                    .any(|(e, g)| *e as usize != g))
        {
            let got: Vec<usize> = args.sizes().collect();
            return Err(GpuError::Args(format!(
                "{}: expected sizes {expected:?}, got {got:?}",
                kernel.name()
            )));
        }
        let fns = &self.gpu().fns;
        supported(fns, &launch)?;
        kernel.check_shared(&launch)?;
        // The argument addresses go on the stack: a launch allocates nothing.
        let mut pointers = [std::ptr::null_mut::<c_void>(); MAX_ARGS];
        if args.pointers_into(&mut pointers).is_none() {
            return Err(GpuError::Args(format!(
                "{}: {} arguments, at most {MAX_ARGS}",
                kernel.name(),
                args.len()
            )));
        }
        let (func, params, stream) = (kernel.raw(), pointers.as_mut_ptr(), self.fork.current());
        // SAFETY: the caller's contract (arguments match the kernel; addresses valid until retired); the pointers
        // address `args`, which outlives the call; the context is current (`Lane::submit`).
        self.gpu().check("cuLaunchKernel", unsafe {
            issue(fns, func, &launch, stream, params)
        })
    }

    unsafe fn copy_to_device(
        &mut self,
        dst: u64,
        src: *const u8,
        bytes: usize,
    ) -> Result<(), GpuError> {
        let f = self.gpu().device().driver.fns.memcpy_htod_async;
        // SAFETY: the caller's contract (valid ranges untouched until retired); the context is current.
        self.gpu().check("cuMemcpyHtoDAsync", unsafe {
            f(dst, src.cast::<c_void>(), bytes, self.fork.current())
        })
    }

    unsafe fn copy_to_host(
        &mut self,
        dst: *mut u8,
        src: u64,
        bytes: usize,
    ) -> Result<(), GpuError> {
        let f = self.gpu().device().driver.fns.memcpy_dtoh_async;
        // SAFETY: as above.
        self.gpu().check("cuMemcpyDtoHAsync", unsafe {
            f(dst.cast::<c_void>(), src, bytes, self.fork.current())
        })
    }

    unsafe fn copy_on_device(&mut self, dst: u64, src: u64, bytes: usize) -> Result<(), GpuError> {
        let f = self.gpu().device().driver.fns.memcpy_dtod_async;
        // SAFETY: as above.
        self.gpu().check("cuMemcpyDtoDAsync", unsafe {
            f(dst, src, bytes, self.fork.current())
        })
    }

    unsafe fn memset(&mut self, dst: u64, value: u8, bytes: usize) -> Result<(), GpuError> {
        let f = self.gpu().device().driver.fns.memset_d8_async;
        // SAFETY: as above.
        self.gpu().check("cuMemsetD8Async", unsafe {
            f(dst, value, bytes, self.fork.current())
        })
    }
}

impl LaneStream for CudaLaneEnqueue<'_> {
    type Raw = CuStream;

    fn raw_stream(&self) -> Option<RawStream<'_, CuStream>> {
        self.fork.private().map(RawStream::new)
    }
}

fn control_error(error: GpuError) -> ControlError {
    match error {
        GpuError::Busy => ControlError::Busy,
        GpuError::Invalid(what) => ControlError::Invalid(what),
        other => ControlError::Failed(other.to_string()),
    }
}

impl Lane for CudaQueue {
    type Gpu = CudaGpu;
    type Enqueue<'a> = CudaLaneEnqueue<'a>;
    type Held<'a> = CudaEnqueue<'a>;

    fn lane(&self) -> LaneId {
        self.lane
    }

    fn submit<R>(
        &self,
        f: impl FnOnce(&mut CudaLaneEnqueue<'_>) -> Result<R, GpuError>,
    ) -> Result<R, GpuError> {
        self.gpu.poison.check()?;
        let _c = self.gpu.device().enter().map_err(from_cuda)?;
        let fork = LaneFork::new(self);
        let result = f(&mut CudaLaneEnqueue {
            queue: self,
            fork: &fork,
        });
        fork.finish()?;
        result
    }

    fn submit_held<R>(
        &self,
        hold: &mut GraphHold,
        f: impl FnOnce(&mut CudaEnqueue<'_>) -> Result<R, GpuError>,
    ) -> Result<R, GpuError> {
        self.gpu.poison.check()?;
        let _c = self.gpu.device().enter().map_err(from_cuda)?;
        let fork = LaneFork::new(self);
        let result = f(&mut CudaEnqueue::on_lane(&self.gpu, self, &fork, hold));
        fork.finish()?;
        result
    }

    fn submit_marked<R>(
        &self,
        f: impl FnOnce(&mut CudaLaneEnqueue<'_>) -> Result<R, GpuError>,
    ) -> Result<(R, Marker), GpuError> {
        // The refusal comes here, before the work: the claim is made first.
        let claim = self.reserve()?;
        let _c = self.gpu.device().enter().map_err(from_cuda)?;
        let fork = LaneFork::new(self);
        let result = f(&mut CudaLaneEnqueue {
            queue: self,
            fork: &fork,
        });
        // The lane joins the private stream before the marker is recorded: the marker covers the library work too.
        fork.finish()?;
        let result = result?;
        let marker = self.record_claim(claim)?;
        Ok((result, marker))
    }

    fn mark(&self) -> Result<Marker, GpuError> {
        let _c = self.gpu.device().enter().map_err(from_cuda)?;
        self.mark_now()
    }
}

impl LaneControl for CudaQueue {
    fn await_marker(&self, marker: Marker) -> Result<(), ControlError> {
        let _c = self
            .gpu
            .device()
            .enter()
            .map_err(|e| control_error(from_cuda(e)))?;
        self.await_now(self.raw, marker).map_err(control_error)
    }

    fn notify_retired(&self, cell: &Arc<RetireCell>) -> Result<(), ControlError> {
        self.gpu.poison.check().map_err(control_error)?;
        let _c = self
            .gpu
            .device()
            .enter()
            .map_err(|e| control_error(from_cuda(e)))?;
        self.arm_cell(cell).map_err(control_error)
    }

    fn confirm_retired(&self, status: i32) -> Option<ControlError> {
        let poisoned = || {
            self.gpu
                .poison
                .get()
                .map(|cause| ControlError::Failed(GpuError::Poisoned(Box::new(cause)).to_string()))
        };
        if status == POISONED_STATUS || status == 0 {
            return poisoned();
        }
        // Asking the driver for the error's name is a plain call. A device fault poisons the device there, so the
        // retirement that carries the fault reports what a retirement behind it reports.
        let err = self
            .gpu
            .check("device work", status)
            .err()
            .unwrap_or(GpuError::Call {
                call: "device work",
                code: status,
                name: "unknown".into(),
            });
        if matches!(err, GpuError::Fault { .. }) {
            poisoned()
        } else {
            Some(control_error(err))
        }
    }
}
