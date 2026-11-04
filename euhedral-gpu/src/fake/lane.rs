//! A fake stream as a shared lane: any thread submits to it through `&self`. The fake device keeps its queues in one
//! state behind a lock, which stands for the hardware (see `FakeGpu`); the lane adds no lock of its own.

use std::sync::Arc;

use euhedral_graph::{ControlError, DeviceQueue, LaneControl, LaneId, Marker, RetireCell};

use crate::{Args, Gpu, GpuError, GraphHold, Lane, LaneEnqueue, Launch, POISONED_STATUS};

use super::enqueue::Real;
use super::stream::FakeQueue;
use super::{FakeEnqueue, FakeGpu, FakeKernel, QOp};

/// What a submission to a shared fake lane enqueues.
#[derive(Debug)]
pub struct FakeLaneEnqueue<'a> {
    queue: &'a FakeQueue,
}

impl FakeLaneEnqueue<'_> {
    fn push(&self, op: QOp) {
        self.queue
            .gpu
            .push(self.queue.global(), self.queue.lane.0, op);
    }
}

impl LaneEnqueue for FakeLaneEnqueue<'_> {
    type Kernel = FakeKernel;

    unsafe fn launch(
        &mut self,
        kernel: &FakeKernel,
        launch: Launch,
        args: &Args,
    ) -> Result<(), GpuError> {
        self.push(QOp::Launch {
            kernel: kernel.clone(),
            launch,
            args: args.clone(),
        });
        Ok(())
    }

    unsafe fn copy_to_device(
        &mut self,
        dst: u64,
        src: *const u8,
        bytes: usize,
    ) -> Result<(), GpuError> {
        self.push(QOp::ToDevice {
            dst,
            src: src as usize,
            bytes,
        });
        Ok(())
    }

    unsafe fn copy_to_host(
        &mut self,
        dst: *mut u8,
        src: u64,
        bytes: usize,
    ) -> Result<(), GpuError> {
        self.push(QOp::ToHost {
            dst: dst as usize,
            src,
            bytes,
        });
        Ok(())
    }

    unsafe fn copy_on_device(&mut self, dst: u64, src: u64, bytes: usize) -> Result<(), GpuError> {
        self.push(QOp::OnDevice { dst, src, bytes });
        Ok(())
    }

    unsafe fn memset(&mut self, dst: u64, value: u8, bytes: usize) -> Result<(), GpuError> {
        self.push(QOp::Memset { dst, value, bytes });
        Ok(())
    }
}

fn control_error(error: GpuError) -> ControlError {
    match error {
        GpuError::Busy => ControlError::Busy,
        GpuError::Invalid(what) => ControlError::Invalid(what),
        other => ControlError::Failed(other.to_string()),
    }
}

fn gpu_error(error: ControlError) -> GpuError {
    match error {
        ControlError::Busy => GpuError::Busy,
        ControlError::Invalid(what) | ControlError::Failed(what) => GpuError::Invalid(what),
    }
}

impl FakeQueue {
    /// Appends an operation to this lane.
    pub(crate) fn push_op(&self, op: QOp) {
        self.gpu.push(self.global(), self.lane.0, op);
    }

    /// Later work on this lane waits for `marker`.
    pub(crate) fn await_now(&self, marker: Marker) -> Result<(), GpuError> {
        self.await_marker(marker).map_err(gpu_error)
    }
}

impl Lane for FakeQueue {
    type Gpu = FakeGpu;
    type Enqueue<'a> = FakeLaneEnqueue<'a>;
    type Held<'a> = FakeEnqueue<'a>;

    fn lane(&self) -> LaneId {
        self.lane
    }

    fn submit<R>(
        &self,
        f: impl FnOnce(&mut FakeLaneEnqueue<'_>) -> Result<R, GpuError>,
    ) -> Result<R, GpuError> {
        self.gpu.alive()?;
        f(&mut FakeLaneEnqueue { queue: self })
    }

    fn submit_held<R>(
        &self,
        hold: &mut GraphHold,
        f: impl FnOnce(&mut FakeEnqueue<'_>) -> Result<R, GpuError>,
    ) -> Result<R, GpuError> {
        self.gpu.alive()?;
        f(&mut FakeEnqueue {
            target: Some(Real::Lane(self)),
            rec: None,
            hold: Some(hold),
        })
    }

    fn submit_marked<R>(
        &self,
        f: impl FnOnce(&mut FakeLaneEnqueue<'_>) -> Result<R, GpuError>,
    ) -> Result<(R, Marker), GpuError> {
        self.gpu.alive()?;
        // The refusal comes here, before the work.
        self.gpu.reserve_mark(self.global())?;
        match f(&mut FakeLaneEnqueue { queue: self }) {
            Ok(result) => {
                let value = self.gpu.push_reserved_mark(self.global(), self.lane.0);
                Ok((result, Marker::new(DeviceQueue::id(self), self.lane, value)))
            }
            Err(error) => {
                self.gpu.cancel_reserved_mark(self.global());
                Err(error)
            }
        }
    }

    fn mark(&self) -> Result<Marker, GpuError> {
        self.gpu.alive()?;
        let value = self.gpu.push_mark(self.global(), self.lane.0)?;
        Ok(Marker::new(DeviceQueue::id(self), self.lane, value))
    }
}

impl LaneControl for FakeQueue {
    fn await_marker(&self, marker: Marker) -> Result<(), ControlError> {
        self.gpu.alive().map_err(control_error)?;
        let offset = match marker.lane.0.checked_sub(self.first) {
            Some(offset) if usize::from(offset) < self.count => usize::from(offset),
            _ => {
                return Err(ControlError::Invalid(format!(
                    "{} lane {} is not a lane of this stream's pool",
                    marker.queue, marker.lane.0
                )));
            }
        };
        let target = self.base + offset;
        {
            let st = self.gpu.lock();
            let lane = &st.lanes[target];
            // A marker of another queue is refused, whatever its lane and value say.
            if lane.id != marker.queue {
                return Err(ControlError::Invalid(format!(
                    "{} lane {} is not a lane of this stream's pool",
                    marker.queue, marker.lane.0
                )));
            }
            if target == self.global() {
                return Ok(());
            }
            if marker.value > lane.marks_enqueued {
                return Err(ControlError::Invalid("a marker that was never made".into()));
            }
            if marker.value <= lane.marks_run {
                return Ok(());
            }
        }
        self.gpu.push(
            self.global(),
            self.lane.0,
            QOp::Await {
                lane: target,
                value: marker.value,
            },
        );
        Ok(())
    }

    fn notify_retired(&self, cell: &Arc<RetireCell>) -> Result<(), ControlError> {
        self.gpu.alive().map_err(control_error)?;
        self.gpu.push(
            self.global(),
            self.lane.0,
            QOp::Notify { cell: cell.clone() },
        );
        Ok(())
    }

    fn confirm_retired(&self, status: i32) -> Option<ControlError> {
        let poisoned = || {
            self.gpu
                .poisoned()
                .map(|cause| ControlError::Failed(GpuError::Poisoned(Box::new(cause)).to_string()))
        };
        if status == POISONED_STATUS {
            return poisoned();
        }
        if status != 0 {
            // The status is the device's own report of the fault.
            self.gpu.observe_fault();
        }
        poisoned()
    }
}
