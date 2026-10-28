//! CUDA streams as device lanes: an event ring behind `DeviceQueue::completed`, device-side awaits, and enqueueing.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use euhedral_graph::{DeviceQueue, LaneId, Marker, QueueId};
use euhedral_memory::cuda::{CuEvent, CuStream, Device, STREAM_NON_BLOCKING};

use crate::kept::Kept;
use crate::{GpuError, POISONED_STATUS, RetireListener, RetireTicket, Stream, StreamOptions};

use super::{CudaEnqueue, CudaGpu, from_cuda};

/// `CU_EVENT_DISABLE_TIMING`: the events only mark completion.
const EVENT_DISABLE_TIMING: u32 = 2;
/// `CUDA_ERROR_NOT_READY`, from `cuEventQuery` while the work before the event runs.
const NOT_READY: i32 = 600;

/// A lane's marker events: value `v` is recorded in slot `v % events.len()`.
#[derive(Debug)]
pub(crate) struct LaneRing {
    /// The identity of the stream this ring belongs to, stamped on its markers.
    id: QueueId,
    device: Arc<Device>,
    events: Vec<CuEvent>,
    /// Markers made so far (the highest value), and the highest value known complete.
    marked: AtomicU64,
    done: AtomicU64,
}

// SAFETY: events of the primary context; `marked` has one writer at a time (the stream's lock) and `done` only grows.
unsafe impl Send for LaneRing {}
// SAFETY: as above.
unsafe impl Sync for LaneRing {}

impl LaneRing {
    fn new(id: QueueId, device: &Arc<Device>, markers: u32) -> Result<Self, GpuError> {
        let _c = device.enter().map_err(from_cuda)?;
        let f = &device.driver.fns;
        // The ring owns the events from the first one made, so a failure part way destroys them.
        let mut ring = Self {
            id,
            device: device.clone(),
            events: Vec::with_capacity(markers as usize),
            marked: AtomicU64::new(0),
            done: AtomicU64::new(0),
        };
        for _ in 0..markers {
            let mut event = std::ptr::null_mut();
            // SAFETY: an out pointer; the context is current.
            let code = unsafe { (f.event_create)(&mut event, EVENT_DISABLE_TIMING) };
            device.check("cuEventCreate", code).map_err(from_cuda)?;
            ring.events.push(event);
        }
        Ok(ring)
    }

    fn slot(&self, value: u64) -> CuEvent {
        self.events[(value % self.events.len() as u64) as usize]
    }

    /// Advances `done` over the markers that completed. The context is current. A query that fails for any reason
    /// but "not ready" poisons the device, so `completed` reports every marker and nothing is stranded behind it.
    fn scan(&self, gpu: &CudaGpu) {
        let f = &gpu.device().driver.fns;
        let marked = self.marked.load(Ordering::Acquire);
        let from = self.done.load(Ordering::Acquire);
        let (done, failed) = advance(from, marked, |value| {
            // SAFETY: an event of this ring; the context is current.
            unsafe { (f.event_query)(self.slot(value)) }
        });
        self.done.fetch_max(done, Ordering::AcqRel);
        if let Some(code) = failed {
            gpu.fail("cuEventQuery", code);
        }
    }
}

/// Walks the markers after `done` up to `marked`, asking `query` (a driver result code) about each. Returns the new
/// `done` and the error code that stopped the walk, if it was neither success nor "not ready".
fn advance(mut done: u64, marked: u64, mut query: impl FnMut(u64) -> i32) -> (u64, Option<i32>) {
    while done < marked {
        match query(done + 1) {
            0 => done += 1,
            NOT_READY => return (done, None),
            other => return (done, Some(other)),
        }
    }
    (done, None)
}

impl Drop for LaneRing {
    fn drop(&mut self) {
        let Ok(_c) = self.device.enter() else { return };
        let f = &self.device.driver.fns;
        for event in &self.events {
            // SAFETY: events made in `new`; destroying one with a wait on it pending is deferred by the driver.
            unsafe { (f.event_destroy)(*event) };
        }
    }
}

/// The query side of a CUDA stream: what the pool and `Completions` hold. It reads the marker ring's atomics and asks
/// the driver whether an event completed; it never waits and takes no lock.
#[derive(Debug)]
pub struct CudaQueue {
    gpu: Arc<CudaGpu>,
    ring: Arc<LaneRing>,
    name: String,
}

impl DeviceQueue for CudaQueue {
    fn id(&self) -> QueueId {
        self.ring.id
    }

    fn completed(&self) -> u64 {
        if self.gpu.poison.get().is_some() {
            return self.ring.marked.load(Ordering::Acquire);
        }
        if let Ok(_c) = self.gpu.device().enter().map_err(from_cuda) {
            self.ring.scan(&self.gpu);
        }
        if self.gpu.poison.get().is_some() {
            return self.ring.marked.load(Ordering::Acquire);
        }
        self.ring.done.load(Ordering::Acquire)
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// A CUDA stream's submitter: lane `lane` of its pool. It has one owner: it is `Send` and not `Sync`, and everything
/// that enqueues takes `&mut self`, so nothing is locked. The shared query side is [`CudaQueue`].
#[derive(Debug)]
pub struct CudaStream {
    pub(crate) gpu: Arc<CudaGpu>,
    pub(crate) id: QueueId,
    lane: LaneId,
    pub(crate) raw: CuStream,
    queue: Arc<CudaQueue>,
    ring: Arc<LaneRing>,
    lanes: Arc<[Arc<LaneRing>]>,
    /// The stream a recording captures this stream's work on, made on first use.
    pub(crate) shadow: Option<CuStream>,
    /// The number the next armed retirement gets.
    pub(crate) next_retire: u64,
    pub(crate) kept: Kept,
}

// SAFETY: the stream belongs to the primary context and is usable from any thread with it current. A submitter has
// one owner at a time (every enqueue takes `&mut self`), so moving it to another thread moves the whole of its state.
// It is deliberately not `Sync`: the `Kept` list holds values that are `Send` only.
unsafe impl Send for CudaStream {}

impl CudaStream {
    /// New streams `names[i]` as lane `i`, each with a ring of `options.markers` events.
    pub(crate) fn open(
        gpu: &Arc<CudaGpu>,
        names: &[&str],
        options: StreamOptions,
    ) -> Result<Vec<CudaStream>, GpuError> {
        if names.is_empty() || names.len() > 64 {
            return Err(GpuError::Invalid("a pool has 1 to 64 lanes".into()));
        }
        if options.markers == 0 {
            return Err(GpuError::Invalid("a lane needs at least one marker".into()));
        }
        gpu.poison.check()?;
        let device = gpu.device();
        let rings = names
            .iter()
            .map(|_| LaneRing::new(QueueId::next(), device, options.markers).map(Arc::new))
            .collect::<Result<Vec<_>, _>>()?;
        let lanes: Arc<[Arc<LaneRing>]> = rings.into();
        let _c = device.enter().map_err(from_cuda)?;
        let f = &device.driver.fns;
        let mut streams = Vec::with_capacity(names.len());
        for (i, name) in names.iter().enumerate() {
            let mut raw = std::ptr::null_mut();
            // SAFETY: an out pointer; the context is current.
            let code = unsafe { (f.stream_create)(&mut raw, STREAM_NON_BLOCKING) };
            device.check("cuStreamCreate", code).map_err(from_cuda)?;
            streams.push(CudaStream {
                gpu: gpu.clone(),
                id: lanes[i].id,
                lane: LaneId(u16::try_from(i).expect("at most 64 lanes")),
                raw,
                queue: Arc::new(CudaQueue {
                    gpu: gpu.clone(),
                    ring: lanes[i].clone(),
                    name: (*name).to_owned(),
                }),
                ring: lanes[i].clone(),
                lanes: lanes.clone(),
                shadow: None,
                next_retire: 0,
                kept: Kept::default(),
            });
        }
        Ok(streams)
    }

    pub(crate) fn queue_arc(&self) -> Arc<CudaQueue> {
        self.queue.clone()
    }

    /// A marker after the work submitted so far. The context is current.
    pub(crate) fn mark_now(&self) -> Result<Marker, GpuError> {
        self.gpu.poison.check()?;
        let ring = &self.ring;
        let n = ring.events.len() as u64;
        let value = ring.marked.load(Ordering::Acquire) + 1;
        if value > n && value - n > ring.done.load(Ordering::Acquire) {
            ring.scan(&self.gpu);
            if value - n > ring.done.load(Ordering::Acquire) {
                return Err(GpuError::Busy);
            }
        }
        let f = &self.gpu.device().driver.fns;
        // SAFETY: the slot's event and this stream; the slot's previous marker completed (checked above).
        self.gpu.check("cuEventRecord", unsafe {
            (f.event_record)(ring.slot(value), self.raw)
        })?;
        ring.marked.store(value, Ordering::Release);
        Ok(Marker::new(self.id, self.lane, value))
    }

    /// Later work on `target` (this stream, or its shadow) waits for `marker`. The context is current.
    pub(crate) fn await_now(&self, target: CuStream, marker: Marker) -> Result<(), GpuError> {
        self.gpu.poison.check()?;
        let other = self
            .lanes
            .get(usize::from(marker.lane.0))
            .filter(|ring| ring.id == marker.queue)
            .ok_or_else(|| {
                GpuError::Invalid(format!(
                    "{} lane {} is not a lane of this stream's pool",
                    marker.queue, marker.lane.0
                ))
            })?;
        if marker.value == 0 {
            return Ok(());
        }
        if marker.value > other.marked.load(Ordering::Acquire) {
            return Err(GpuError::Invalid("a marker that was never made".into()));
        }
        if marker.value <= other.done.load(Ordering::Acquire) {
            return Ok(());
        }
        other.scan(&self.gpu);
        if marker.value <= other.done.load(Ordering::Acquire) {
            return Ok(());
        }
        // SAFETY: an event of the pool's ring and a stream of this device; the context is current.
        self.gpu.check("cuStreamWaitEvent", unsafe {
            (self.gpu.fns.stream_wait_event)(target, other.slot(marker.value), 0)
        })
    }

    pub(crate) fn lane_id(&self) -> LaneId {
        self.lane
    }

    /// The context current on the calling thread for as long as the guard lives.
    pub(crate) fn enter(&self) -> Result<euhedral_memory::cuda::Current<'_>, GpuError> {
        self.gpu.device().enter().map_err(from_cuda)
    }
}

impl Drop for CudaStream {
    /// A shutdown call: it waits for the stream's work to end.
    fn drop(&mut self) {
        let Ok(_c) = self.gpu.device().enter() else {
            return;
        };
        let f = &self.gpu.device().driver.fns;
        // SAFETY: streams made by this value, no longer in use; the context is current.
        unsafe {
            (f.stream_synchronize)(self.raw);
            if let Some(shadow) = self.shadow {
                (f.stream_synchronize)(shadow);
            }
        }
        self.kept.clear();
        // SAFETY: as above.
        unsafe {
            (f.stream_destroy)(self.raw);
            if let Some(shadow) = self.shadow {
                (f.stream_destroy)(shadow);
            }
        }
    }
}

impl Stream for CudaStream {
    type Gpu = CudaGpu;
    type Enqueue<'a> = CudaEnqueue<'a>;

    fn gpu(&self) -> &Arc<CudaGpu> {
        &self.gpu
    }

    fn lane(&self) -> LaneId {
        self.lane
    }

    fn id(&self) -> QueueId {
        self.id
    }

    fn queue(&self) -> Arc<dyn DeviceQueue> {
        self.queue.clone()
    }

    fn submit<R>(
        &mut self,
        f: impl FnOnce(&mut CudaEnqueue<'_>) -> Result<R, GpuError>,
    ) -> Result<R, GpuError> {
        self.gpu.poison.check()?;
        // One reference-count increment, so the context guard does not borrow the stream that `f` enqueues on.
        let gpu = self.gpu.clone();
        let _c = gpu.device().enter().map_err(from_cuda)?;
        f(&mut CudaEnqueue::on_stream(&gpu, self))
    }

    fn mark(&mut self) -> Result<Marker, GpuError> {
        let _c = self.enter()?;
        self.mark_now()
    }

    fn await_marker(&mut self, marker: Marker) -> Result<(), GpuError> {
        let _c = self.enter()?;
        self.await_now(self.raw, marker)
    }

    fn notify_retired(&mut self, listener: Box<dyn RetireListener>) -> Result<(), GpuError> {
        self.gpu.poison.check()?;
        let gpu = self.gpu.clone();
        let _c = gpu.device().enter().map_err(from_cuda)?;
        self.arm(self.next_retire, listener)?;
        self.next_retire += 1;
        Ok(())
    }

    fn confirm_retired(&mut self, ticket: RetireTicket) -> Option<GpuError> {
        if ticket.stream() != self.id {
            return Some(GpuError::Invalid(
                "a retirement ticket of another stream".into(),
            ));
        }
        if ticket.status() == POISONED_STATUS {
            self.gpu
                .poison
                .get()
                .map(|cause| GpuError::Poisoned(Box::new(cause)))
        } else {
            // Release only what this retirement covered; a poisoned ticket does not say which that was.
            self.kept.release_through(ticket.seq());
            if ticket.status() != 0 {
                // An ordinary thread may ask the driver for the error's name.
                let err = self
                    .gpu
                    .check("device work", ticket.status())
                    .err()
                    .unwrap_or(GpuError::Call {
                        call: "device work",
                        code: ticket.status(),
                        name: "unknown".into(),
                    });
                Some(err)
            } else {
                self.gpu
                    .poison
                    .get()
                    .map(|cause| GpuError::Poisoned(Box::new(cause)))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_walk_stops_at_the_first_marker_not_ready() {
        let results = [0, 0, NOT_READY, 0];
        let (done, failed) = advance(0, 4, |v| results[(v - 1) as usize]);
        assert_eq!((done, failed), (2, None));
        assert_eq!(advance(2, 2, |_| unreachable!()), (2, None));
    }

    #[test]
    fn a_walk_reports_an_error_other_than_not_ready() {
        let results = [0, 101, 0];
        let (done, failed) = advance(0, 3, |v| results[(v - 1) as usize]);
        assert_eq!((done, failed), (1, Some(101)));
    }
}
