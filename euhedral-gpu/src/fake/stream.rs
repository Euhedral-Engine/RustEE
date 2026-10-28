//! Fake streams: a shared query side and a single-owner submitter.

use std::sync::Arc;

use euhedral_graph::{DeviceQueue, LaneId, Marker, QueueId};

use crate::kept::Kept;
use crate::retire::{RetireSink, Retirements};
use crate::{Gpu, GpuError, POISONED_STATUS, RetireListener, RetireTicket, Stream};

use super::{FakeEnqueue, FakeGpu, QOp};

/// The query side of a fake stream: what the pool and `Completions` hold. Shared and `Sync`.
#[derive(Debug)]
pub struct FakeQueue {
    gpu: Arc<FakeGpu>,
    id: QueueId,
    name: String,
    /// Global index of this stream's lane.
    global: usize,
    /// The end of the retirement queue the device pushes to. It lives as long as the shared side does, as a driver's
    /// callback context would.
    sink: Arc<RetireSink>,
}

impl DeviceQueue for FakeQueue {
    fn id(&self) -> QueueId {
        self.id
    }

    fn completed(&self) -> u64 {
        let poisoned = self.gpu.poison.get().is_some();
        let st = self.gpu.lock();
        let lane = &st.lanes[self.global];
        if poisoned {
            lane.marks_enqueued
        } else {
            lane.marks_run
        }
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// A fake stream's submitter: lane `lane` of its pool. It has one owner, `Send` and not `Sync`; everything that
/// enqueues takes `&mut self`.
#[derive(Debug)]
pub struct FakeStream {
    gpu: Arc<FakeGpu>,
    queue: Arc<FakeQueue>,
    id: QueueId,
    /// Global index of lane 0 of this stream's pool, this stream's index in the pool, and the pool's size.
    base: usize,
    index: usize,
    count: usize,
    /// The owner's end of the retirement queue, and the retirements armed.
    pub(crate) retire: Retirements,
    pub(crate) kept: Kept,
}

impl FakeStream {
    pub(crate) fn new(
        gpu: Arc<FakeGpu>,
        id: QueueId,
        name: &str,
        base: usize,
        index: usize,
        count: usize,
    ) -> Self {
        let lane = LaneId(u16::try_from(index).expect("at most 64 lanes"));
        let (retire, sink) = Retirements::new(id, lane);
        let queue = Arc::new(FakeQueue {
            gpu: gpu.clone(),
            id,
            name: name.to_owned(),
            global: base + index,
            sink: Arc::new(sink),
        });
        Self {
            gpu,
            queue,
            id,
            base,
            index,
            count,
            retire,
            kept: Kept::default(),
        }
    }

    pub(crate) fn queue_arc(&self) -> Arc<FakeQueue> {
        self.queue.clone()
    }

    fn global(&self) -> usize {
        self.base + self.index
    }

    fn lane_id(&self) -> u16 {
        u16::try_from(self.index).expect("at most 64 lanes")
    }

    pub(crate) fn push(&self, op: QOp) {
        self.gpu.push(self.global(), self.lane_id(), op);
    }

    fn arm(&mut self, listener: Option<Box<dyn RetireListener>>) -> Result<(), GpuError> {
        self.gpu.poison.check()?;
        self.push(QOp::Retire {
            seq: self.retire.next_seq(),
            sink: self.queue.sink.clone(),
        });
        self.retire.register(listener);
        Ok(())
    }

    /// A marker after the work enqueued so far.
    pub(crate) fn mark_now(&self) -> Result<Marker, GpuError> {
        self.gpu.poison.check()?;
        let value = {
            let st = self.gpu.lock();
            let lane = &st.lanes[self.global()];
            if lane.marks_enqueued - lane.marks_run >= u64::from(lane.markers) {
                return Err(GpuError::Busy);
            }
            lane.marks_enqueued + 1
        };
        self.push(QOp::Mark { value });
        Ok(Marker::new(self.id, LaneId(self.lane_id()), value))
    }

    /// Later work on this stream waits for `marker`.
    pub(crate) fn await_now(&self, marker: Marker) -> Result<(), GpuError> {
        self.gpu.poison.check()?;
        let lane = usize::from(marker.lane.0);
        if lane >= self.count {
            return Err(GpuError::Invalid(format!(
                "{} lane {} is not a lane of this stream's pool",
                marker.queue, marker.lane.0
            )));
        }
        let target = self.base + lane;
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
        self.push(QOp::Await {
            lane: target,
            value: marker.value,
        });
        Ok(())
    }
}

impl Stream for FakeStream {
    type Gpu = FakeGpu;
    type Enqueue<'a> = FakeEnqueue<'a>;

    fn gpu(&self) -> &Arc<FakeGpu> {
        &self.gpu
    }

    fn lane(&self) -> LaneId {
        LaneId(self.lane_id())
    }

    fn id(&self) -> QueueId {
        self.id
    }

    fn queue(&self) -> Arc<dyn DeviceQueue> {
        self.queue.clone()
    }

    fn submit<R>(
        &mut self,
        f: impl FnOnce(&mut FakeEnqueue<'_>) -> Result<R, GpuError>,
    ) -> Result<R, GpuError> {
        self.gpu.poison.check()?;
        f(&mut FakeEnqueue {
            target: Some(self),
            rec: None,
        })
    }

    fn mark(&mut self) -> Result<Marker, GpuError> {
        self.mark_now()
    }

    fn await_marker(&mut self, marker: Marker) -> Result<(), GpuError> {
        self.await_now(marker)
    }

    fn notify_retired(&mut self, listener: Box<dyn RetireListener>) -> Result<(), GpuError> {
        self.arm(Some(listener))
    }

    fn arm_retirement(&mut self) -> Result<(), GpuError> {
        self.arm(None)
    }

    fn poll_retired(&mut self) -> Option<RetireTicket> {
        let poisoned = self.gpu.poison.get().is_some();
        let ticket = self.retire.poll(poisoned);
        if self.retire.take_desync() {
            self.gpu.poison.set(GpuError::Invalid(
                "a retirement outcome arrived that no one had armed".into(),
            ));
        }
        ticket
    }

    fn outstanding_tickets(&self) -> usize {
        self.retire.outstanding_tickets()
    }

    fn confirm_retired(&mut self, ticket: RetireTicket) -> Option<GpuError> {
        if ticket.stream() != self.id {
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
        self.kept.release_through(ticket.seq());
        if ticket.status() != 0 {
            return self.gpu.poisoned();
        }
        self.gpu
            .poisoned()
            .map(|cause| GpuError::Poisoned(Box::new(cause)))
    }
}
