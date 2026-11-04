//! A CUDA stream as a shared lane: any worker submits to it through `&self`, as a stage of a graph does.
//!
//! The state workers share is atomic (the marker ring, see `ring`) or reached through a claim (the retirement channel,
//! see `retire`); the driver serializes the launches it gets on the stream.

use std::any::Any;
use std::ffi::c_void;

use euhedral_graph::{DeviceQueue, LaneId, Marker};

use crate::POISONED_STATUS;
use crate::{Args, Enqueue, GpuError, Kernel, Lane, Launch, MAX_ARGS, RetireTicket};

use super::stream::CudaQueue;
use super::{CudaGpu, CudaKernel, from_cuda};

/// What a submission to a shared CUDA lane enqueues.
#[derive(Debug)]
pub struct CudaLaneEnqueue<'a> {
    queue: &'a CudaQueue,
}

impl CudaLaneEnqueue<'_> {
    fn gpu(&self) -> &CudaGpu {
        &self.queue.gpu
    }
}

impl Enqueue for CudaLaneEnqueue<'_> {
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
        // The argument addresses go on the stack: a launch allocates nothing.
        let mut pointers = [std::ptr::null_mut::<c_void>(); MAX_ARGS];
        if args.pointers_into(&mut pointers).is_none() {
            return Err(GpuError::Args(format!(
                "{}: {} arguments, at most {MAX_ARGS}",
                kernel.name(),
                args.len()
            )));
        }
        let (func, params, stream) = (kernel.raw(), pointers.as_mut_ptr(), self.queue.raw);
        // SAFETY: the caller's contract (arguments match the kernel; addresses valid until retired); the pointers
        // address `args`, which outlives the call; the context is current (`Lane::submit`).
        self.gpu().check("cuLaunchKernel", unsafe {
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
            f(dst, src.cast::<c_void>(), bytes, self.queue.raw)
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
            f(dst.cast::<c_void>(), src, bytes, self.queue.raw)
        })
    }

    unsafe fn copy_on_device(&mut self, dst: u64, src: u64, bytes: usize) -> Result<(), GpuError> {
        let f = self.gpu().device().driver.fns.memcpy_dtod_async;
        // SAFETY: as above.
        self.gpu().check("cuMemcpyDtoDAsync", unsafe {
            f(dst, src, bytes, self.queue.raw)
        })
    }

    unsafe fn memset(&mut self, dst: u64, value: u8, bytes: usize) -> Result<(), GpuError> {
        let f = self.gpu().device().driver.fns.memset_d8_async;
        // SAFETY: as above.
        self.gpu().check("cuMemsetD8Async", unsafe {
            f(dst, value, bytes, self.queue.raw)
        })
    }

    fn keep(&mut self, value: Box<dyn Any + Send>) {
        self.queue.shared.keep(value);
    }
}

impl Lane for CudaQueue {
    type Gpu = CudaGpu;
    type Enqueue<'a> = CudaLaneEnqueue<'a>;

    fn lane(&self) -> LaneId {
        self.lane
    }

    fn submit<R>(
        &self,
        f: impl FnOnce(&mut CudaLaneEnqueue<'_>) -> Result<R, GpuError>,
    ) -> Result<R, GpuError> {
        self.gpu.poison.check()?;
        let _c = self.gpu.device().enter().map_err(from_cuda)?;
        f(&mut CudaLaneEnqueue { queue: self })
    }

    fn mark(&self) -> Result<Marker, GpuError> {
        let _c = self.gpu.device().enter().map_err(from_cuda)?;
        self.mark_now()
    }

    fn await_marker(&self, marker: Marker) -> Result<(), GpuError> {
        let _c = self.gpu.device().enter().map_err(from_cuda)?;
        self.await_now(self.raw, marker)
    }

    fn arm_retirement(&self) -> Result<(), GpuError> {
        self.gpu.poison.check()?;
        let _c = self.gpu.device().enter().map_err(from_cuda)?;
        self.shared
            .arm(|_seq| self.arm_callback_to(self.shared.sink()))
    }

    fn poll_retired(&self) -> Option<RetireTicket> {
        let (ticket, desync) = self
            .shared
            .poll(|| self.gpu.poison.get().is_some(), || self.look());
        if desync {
            self.gpu
                .fail_invalid("a retirement outcome arrived that no one had armed");
        }
        ticket
    }

    fn confirm_retired(&self, ticket: RetireTicket) -> Option<GpuError> {
        if ticket.stream() != DeviceQueue::id(self) {
            return Some(GpuError::Invalid(
                "a retirement ticket of another stream".into(),
            ));
        }
        let poisoned = || {
            self.gpu
                .poison
                .get()
                .map(|cause| GpuError::Poisoned(Box::new(cause)))
        };
        if ticket.status() == POISONED_STATUS {
            return poisoned();
        }
        // Release only what this retirement covered; a poisoned ticket does not say which that was.
        self.shared.release_through(ticket.seq());
        if ticket.status() == 0 {
            return poisoned();
        }
        let err = self
            .gpu
            .check("device work", ticket.status())
            .err()
            .unwrap_or(GpuError::Call {
                call: "device work",
                code: ticket.status(),
                name: "unknown".into(),
            });
        if matches!(err, GpuError::Fault { .. }) {
            poisoned()
        } else {
            Some(err)
        }
    }
}
