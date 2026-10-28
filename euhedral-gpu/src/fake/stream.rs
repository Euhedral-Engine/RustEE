//! Fake streams.

use std::sync::Arc;

use euhedral_graph::{DeviceQueue, LaneId, Marker};

use crate::kept::Kept;
use crate::stream_lock::StreamLock;
use crate::{Gpu, GpuError, POISONED_STATUS, RetireListener, RetireTicket, Stream};

use super::{FakeEnqueue, FakeGpu, QOp};

#[derive(Debug)]
pub(crate) struct StreamState {
    /// The number the next armed retirement gets.
    pub(crate) next_retire: u64,
}

/// A fake stream: lane `lane` of its pool.
#[derive(Debug)]
pub struct FakeStream {
    gpu: Arc<FakeGpu>,
    name: String,
    /// Global index of lane 0 of this stream's pool, this stream's index in the pool, and the pool's size.
    base: usize,
    index: usize,
    count: usize,
    state: StreamLock<StreamState>,
    pub(crate) kept: Kept,
}

impl FakeStream {
    pub(crate) fn new(
        gpu: Arc<FakeGpu>,
        name: &str,
        base: usize,
        index: usize,
        count: usize,
    ) -> Self {
        Self {
            gpu,
            name: name.to_owned(),
            base,
            index,
            count,
            state: StreamLock::new(StreamState { next_retire: 1 }),
            kept: Kept::default(),
        }
    }

    pub(crate) fn global_index(&self) -> usize {
        self.global()
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

    fn mark_locked(&self) -> Result<Marker, GpuError> {
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
        Ok(Marker {
            lane: LaneId(self.lane_id()),
            value,
        })
    }

    fn await_locked(&self, marker: Marker) -> Result<(), GpuError> {
        self.gpu.poison.check()?;
        if marker.lane.0 == self.lane_id() {
            return Ok(());
        }
        let target = usize::from(marker.lane.0);
        if target >= self.count {
            return Err(GpuError::Invalid("a marker of another pool".into()));
        }
        let target = self.base + target;
        {
            let st = self.gpu.lock();
            let lane = &st.lanes[target];
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

    /// `mark` for a sequence that already holds this stream's lock.
    pub(crate) fn mark_held(&self) -> Result<Marker, GpuError> {
        self.mark_locked()
    }

    /// `await_marker` for a sequence that already holds this stream's lock.
    pub(crate) fn await_held(&self, marker: Marker) -> Result<(), GpuError> {
        self.await_locked(marker)
    }

    pub(crate) fn lock(
        &self,
    ) -> Result<crate::stream_lock::StreamGuard<'_, StreamState>, GpuError> {
        self.state.lock()
    }

    pub(crate) fn enqueue<'a>(&'a self, state: &'a mut StreamState) -> FakeEnqueue<'a> {
        FakeEnqueue {
            target: Some((self, state)),
            rec: None,
        }
    }
}

impl Drop for FakeStream {
    fn drop(&mut self) {
        self.kept.clear();
    }
}

impl DeviceQueue for FakeStream {
    fn completed(&self) -> u64 {
        let poisoned = self.gpu.poison.get().is_some();
        let st = self.gpu.lock();
        let lane = &st.lanes[self.global()];
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

impl Stream for FakeStream {
    type Gpu = FakeGpu;
    type Enqueue<'a> = FakeEnqueue<'a>;

    fn gpu(&self) -> &Arc<FakeGpu> {
        &self.gpu
    }

    fn lane(&self) -> LaneId {
        LaneId(self.lane_id())
    }

    fn submit<R>(
        &self,
        f: impl FnOnce(&mut FakeEnqueue<'_>) -> Result<R, GpuError>,
    ) -> Result<R, GpuError> {
        let mut guard = self.state.lock()?;
        self.gpu.poison.check()?;
        f(&mut self.enqueue(&mut guard))
    }

    fn mark(&self) -> Result<Marker, GpuError> {
        let _guard = self.state.lock()?;
        self.mark_locked()
    }

    fn await_marker(&self, marker: Marker) -> Result<(), GpuError> {
        let _guard = self.state.lock()?;
        self.await_locked(marker)
    }

    fn notify_retired(&self, listener: Box<dyn RetireListener>) -> Result<(), GpuError> {
        let mut guard = self.state.lock()?;
        self.gpu.poison.check()?;
        let seq = guard.next_retire;
        guard.next_retire += 1;
        self.push(QOp::Retire { seq, listener });
        Ok(())
    }

    fn confirm_retired(&self, ticket: RetireTicket) -> Option<GpuError> {
        if ticket.status == POISONED_STATUS {
            return self
                .gpu
                .poisoned()
                .map(|cause| GpuError::Poisoned(Box::new(cause)));
        }
        self.kept.release_through(ticket.seq);
        if ticket.status != 0 {
            return self.gpu.poisoned();
        }
        self.gpu
            .poisoned()
            .map(|cause| GpuError::Poisoned(Box::new(cause)))
    }
}
