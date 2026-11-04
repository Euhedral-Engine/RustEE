//! A fake stream as a shared lane: any thread submits to it through `&self`. The fake device keeps its queues in one
//! state behind a lock, which stands for the hardware (see `FakeGpu`); the lane adds no lock of its own.

use std::any::Any;

use euhedral_graph::{DeviceQueue, LaneId, Marker};

use crate::{Args, Enqueue, Gpu, GpuError, Lane, Launch, POISONED_STATUS, RetireTicket};

use super::stream::FakeQueue;
use super::{FakeGpu, FakeKernel, QOp};

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

impl Enqueue for FakeLaneEnqueue<'_> {
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

    fn keep(&mut self, value: Box<dyn Any + Send>) {
        self.queue.shared.keep(value);
    }
}

impl Lane for FakeQueue {
    type Gpu = FakeGpu;
    type Enqueue<'a> = FakeLaneEnqueue<'a>;

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

    fn mark(&self) -> Result<Marker, GpuError> {
        self.gpu.alive()?;
        let value = self.gpu.push_mark(self.global(), self.lane.0)?;
        Ok(Marker::new(DeviceQueue::id(self), self.lane, value))
    }

    fn await_marker(&self, marker: Marker) -> Result<(), GpuError> {
        self.gpu.alive()?;
        let offset = match marker.lane.0.checked_sub(self.first) {
            Some(offset) if usize::from(offset) < self.count => usize::from(offset),
            _ => {
                return Err(GpuError::Invalid(format!(
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
                return Err(GpuError::Invalid(format!(
                    "{} lane {} is not a lane of this stream's pool",
                    marker.queue, marker.lane.0
                )));
            }
            if target == self.global() {
                return Ok(());
            }
            if marker.value > lane.marks_enqueued {
                return Err(GpuError::Invalid("a marker that was never made".into()));
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

    fn arm_retirement(&self) -> Result<(), GpuError> {
        self.gpu.alive()?;
        self.shared.arm(|seq| {
            self.gpu.push(
                self.global(),
                self.lane.0,
                QOp::Retire {
                    seq,
                    sink: self.shared.sink().clone(),
                },
            );
            Ok(())
        })
    }

    fn poll_retired(&self) -> Option<RetireTicket> {
        let (ticket, desync) = self.shared.poll(
            || self.gpu.poison.get().is_some(),
            || {
                let _ = DeviceQueue::completed(self);
            },
        );
        if desync {
            self.gpu.poison.set(GpuError::Invalid(
                "a retirement outcome arrived that no one had armed".into(),
            ));
        }
        ticket
    }

    fn confirm_retired(&self, ticket: RetireTicket) -> Option<GpuError> {
        if ticket.stream() != DeviceQueue::id(self) {
            return Some(GpuError::Invalid(
                "a retirement ticket of another stream".into(),
            ));
        }
        if ticket.status() == POISONED_STATUS {
            return self
                .gpu
                .poisoned()
                .map(|cause| GpuError::Poisoned(Box::new(cause)));
        }
        self.shared.release_through(ticket.seq());
        if ticket.status() != 0 {
            // The status is the device's own report of the fault.
            self.gpu.observe_fault();
        }
        self.gpu
            .poisoned()
            .map(|cause| GpuError::Poisoned(Box::new(cause)))
    }
}
