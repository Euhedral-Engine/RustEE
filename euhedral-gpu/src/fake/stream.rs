//! Fake streams: a shared query side and a single-owner submitter.

use std::sync::Arc;

use euhedral_graph::{DeviceQueue, LaneId, Marker, QueueId};

use crate::kept::Kept;
use crate::retire::{RetireSink, Retirements};
use crate::{Gpu, GpuError, POISONED_STATUS, RetireListener, RetireTicket, Stream, StreamPriority};

use crate::lanes::Probe;

use super::{FAKE_GREATEST_PRIORITY, FAKE_NORMAL_PRIORITY, FakeEnqueue, FakeGpu, QOp};

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
        let (run, enqueued) = {
            let st = self.gpu.lock();
            let lane = &st.lanes[self.global];
            (lane.marks_run, lane.marks_enqueued)
        };
        if run < enqueued {
            // A query of a marker that has not completed reaches the device: a fault it has is learned here.
            self.gpu.observe_fault();
        }
        if self.gpu.poison.get().is_some() {
            enqueued
        } else {
            run
        }
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Where a stream sits among the lanes of the call that opened it.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Placement {
    /// Global index of the call's first lane.
    pub(crate) base: usize,
    /// The id of that lane in its pool.
    pub(crate) first: u16,
    /// This stream's index among the call's lanes, and how many lanes the call made.
    pub(crate) index: usize,
    pub(crate) count: usize,
}

/// A fake stream's submitter: lane `lane` of its pool. It has one owner, `Send` and not `Sync`; everything that
/// enqueues takes `&mut self`.
#[derive(Debug)]
pub struct FakeStream {
    gpu: Arc<FakeGpu>,
    queue: Arc<FakeQueue>,
    id: QueueId,
    /// Global index of the first lane the opening call made, that lane's id, this stream's index among the call's
    /// lanes, and how many lanes the call made.
    base: usize,
    first: u16,
    index: usize,
    count: usize,
    priority: StreamPriority,
    /// The owner's end of the retirement queue, and the retirements armed.
    pub(crate) retire: Retirements,
    pub(crate) kept: Kept,
}

impl FakeStream {
    pub(crate) fn new(
        gpu: Arc<FakeGpu>,
        id: QueueId,
        name: &str,
        place: Placement,
        priority: StreamPriority,
    ) -> Self {
        let Placement {
            base,
            first,
            index,
            count,
        } = place;
        let lane = LaneId(first + u16::try_from(index).expect("at most 64 lanes"));
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
            first,
            index,
            count,
            priority,
            retire,
            kept: Kept::default(),
        }
    }

    /// The priority the stream was opened with, as a driver reports it: [`FAKE_NORMAL_PRIORITY`] or
    /// [`FAKE_GREATEST_PRIORITY`]. The signature is `CudaStream::priority`'s.
    pub fn priority(&self) -> Result<i32, GpuError> {
        Ok(match self.priority {
            StreamPriority::Normal => FAKE_NORMAL_PRIORITY,
            StreamPriority::High => FAKE_GREATEST_PRIORITY,
        })
    }

    /// One look at the device: the fake's queue query, which learns a fault the device has when a marker of the lane is
    /// still to complete. `CudaStream` queries its marker events the same way; [`Probe::Idle`] looks only while
    /// retirements are outstanding, [`Probe::Confirm`] whenever a retirement just arrived. The fake does not throttle.
    pub(crate) fn probe_fault(&mut self, mode: Probe) {
        if mode == Probe::Confirm || self.retire.outstanding() > 0 {
            let _ = self.queue.completed();
        }
    }

    /// Whether `marker` names a lane this stream's opening call made, under that lane's queue.
    pub(crate) fn owns_marker_lane(&self, marker: Marker) -> bool {
        let Some(offset) = marker.lane.0.checked_sub(self.first) else {
            return false;
        };
        let offset = usize::from(offset);
        offset < self.count && self.gpu.lock().lanes[self.base + offset].id == marker.queue
    }

    fn global(&self) -> usize {
        self.base + self.index
    }

    fn lane_id(&self) -> u16 {
        self.first + u16::try_from(self.index).expect("at most 64 lanes")
    }

    pub(crate) fn push(&self, op: QOp) {
        self.gpu.push(self.global(), self.lane_id(), op);
    }

    fn arm(&mut self, listener: Option<Box<dyn RetireListener>>) -> Result<(), GpuError> {
        self.gpu.alive()?;
        self.push(QOp::Retire {
            seq: self.retire.next_seq(),
            sink: self.queue.sink.clone(),
        });
        self.retire.register(listener);
        Ok(())
    }

    /// A marker after the work enqueued so far.
    pub(crate) fn mark_now(&self) -> Result<Marker, GpuError> {
        self.gpu.alive()?;
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
        self.gpu.alive()?;
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
            // The status is the device's own report of the fault.
            self.gpu.observe_fault();
            return self.gpu.poisoned();
        }
        self.gpu
            .poisoned()
            .map(|cause| GpuError::Poisoned(Box::new(cause)))
    }
}
