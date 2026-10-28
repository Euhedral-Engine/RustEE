//! CUDA streams as device lanes: an event ring behind `DeviceQueue::completed`, device-side awaits, and enqueueing.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use euhedral_graph::{DeviceQueue, LaneId, Marker};
use euhedral_memory::cuda::{CuEvent, CuStream, Device, STREAM_NON_BLOCKING};

use crate::kept::Kept;
use crate::stream_lock::StreamLock;
use crate::{GpuError, POISONED_STATUS, RetireListener, RetireTicket, Stream, StreamOptions};

use super::{CudaEnqueue, CudaGpu, from_cuda};

/// `CU_EVENT_DISABLE_TIMING`: the events only mark completion.
const EVENT_DISABLE_TIMING: u32 = 2;
/// `CUDA_ERROR_NOT_READY`, from `cuEventQuery` while the work before the event runs.
const NOT_READY: i32 = 600;

/// A driver handle that moves between threads.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Handle(pub CuStream);

// SAFETY: a stream of the primary context, usable from any thread with that context current.
unsafe impl Send for Handle {}
// SAFETY: as above; the handle is only read.
unsafe impl Sync for Handle {}

/// A lane's marker events: value `v` is recorded in slot `v % events.len()`.
#[derive(Debug)]
pub(crate) struct LaneRing {
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
    fn new(device: &Arc<Device>, markers: u32) -> Result<Self, GpuError> {
        let _c = device.enter().map_err(from_cuda)?;
        let f = &device.driver.fns;
        // The ring owns the events from the first one made, so a failure part way destroys them.
        let mut ring = Self {
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

    /// Advances `done` over the markers that completed. The context is current.
    fn scan(&self, gpu: &CudaGpu) {
        let f = &gpu.device().driver.fns;
        let marked = self.marked.load(Ordering::Acquire);
        let mut done = self.done.load(Ordering::Acquire);
        while done < marked {
            // SAFETY: an event of this ring; the context is current.
            let code = unsafe { (f.event_query)(self.slot(done + 1)) };
            match code {
                0 => done += 1,
                NOT_READY => break,
                // A fault poisons the device (`completed` then reports every marker); another error leaves the
                // marker unreached.
                other => {
                    let _ = gpu.check("cuEventQuery", other);
                    break;
                }
            }
        }
        self.done.fetch_max(done, Ordering::AcqRel);
    }
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

#[derive(Debug)]
pub(crate) struct StreamState {
    /// The stream a recording captures this stream's work on, made on first use.
    pub(crate) shadow: Option<Handle>,
    /// The number the next armed retirement gets.
    pub(crate) next_retire: u64,
}

/// A CUDA stream: lane `lane` of its pool.
#[derive(Debug)]
pub struct CudaStream {
    pub(crate) gpu: Arc<CudaGpu>,
    lane: LaneId,
    name: String,
    pub(crate) raw: CuStream,
    ring: Arc<LaneRing>,
    lanes: Arc<[Arc<LaneRing>]>,
    pub(crate) state: StreamLock<StreamState>,
    pub(crate) kept: Kept,
}

// SAFETY: the stream belongs to the primary context and is usable from any thread with it current; every enqueue
// holds the stream's lock.
unsafe impl Send for CudaStream {}
// SAFETY: as above.
unsafe impl Sync for CudaStream {}

impl CudaStream {
    /// New streams `names[i]` as lane `i`, each with a ring of `options.markers` events.
    pub(crate) fn open(
        gpu: &Arc<CudaGpu>,
        names: &[&str],
        options: StreamOptions,
    ) -> Result<Vec<Arc<CudaStream>>, GpuError> {
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
            .map(|_| LaneRing::new(device, options.markers).map(Arc::new))
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
            streams.push(Arc::new(CudaStream {
                gpu: gpu.clone(),
                lane: LaneId(u16::try_from(i).expect("at most 64 lanes")),
                name: (*name).to_owned(),
                raw,
                ring: lanes[i].clone(),
                lanes: lanes.clone(),
                state: StreamLock::new(StreamState {
                    shadow: None,
                    next_retire: 0,
                }),
                kept: Kept::default(),
            }));
        }
        Ok(streams)
    }

    /// A marker after the work submitted so far. The caller holds the stream's lock and the context is current.
    pub(crate) fn mark_held(&self) -> Result<Marker, GpuError> {
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
        Ok(Marker {
            lane: self.lane,
            value,
        })
    }

    /// Later work on `target` (this stream, or a stream of a recording) waits for `marker`. The caller holds the
    /// lock of `target` and the context is current.
    pub(crate) fn await_held(&self, target: CuStream, marker: Marker) -> Result<(), GpuError> {
        self.gpu.poison.check()?;
        let other = self
            .lanes
            .get(usize::from(marker.lane.0))
            .ok_or_else(|| GpuError::Invalid("a marker of another pool".into()))?;
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
        let shadow = self
            .state
            .lock()
            .ok()
            .and_then(|state| state.shadow)
            .map(|h| h.0);
        // SAFETY: streams made by this value, no longer in use; the context is current.
        unsafe {
            (f.stream_synchronize)(self.raw);
            if let Some(shadow) = shadow {
                (f.stream_synchronize)(shadow);
            }
        }
        self.kept.clear();
        // SAFETY: as above.
        unsafe {
            (f.stream_destroy)(self.raw);
            if let Some(shadow) = shadow {
                (f.stream_destroy)(shadow);
            }
        }
    }
}

impl DeviceQueue for CudaStream {
    fn completed(&self) -> u64 {
        if self.gpu.poison.get().is_some() {
            return self.ring.marked.load(Ordering::Acquire);
        }
        if let Ok(_c) = self.enter() {
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

impl Stream for CudaStream {
    type Gpu = CudaGpu;
    type Enqueue<'a> = CudaEnqueue<'a>;

    fn gpu(&self) -> &Arc<CudaGpu> {
        &self.gpu
    }

    fn lane(&self) -> LaneId {
        self.lane
    }

    fn submit<R>(
        &self,
        f: impl FnOnce(&mut CudaEnqueue<'_>) -> Result<R, GpuError>,
    ) -> Result<R, GpuError> {
        let mut guard = self.state.lock()?;
        self.gpu.poison.check()?;
        let _c = self.enter()?;
        f(&mut CudaEnqueue::on_stream(self, &mut guard))
    }

    fn mark(&self) -> Result<Marker, GpuError> {
        let _guard = self.state.lock()?;
        let _c = self.enter()?;
        self.mark_held()
    }

    fn await_marker(&self, marker: Marker) -> Result<(), GpuError> {
        let _guard = self.state.lock()?;
        let _c = self.enter()?;
        self.await_held(self.raw, marker)
    }

    fn notify_retired(&self, listener: Box<dyn RetireListener>) -> Result<(), GpuError> {
        let mut guard = self.state.lock()?;
        self.gpu.poison.check()?;
        let _c = self.enter()?;
        self.arm(guard.next_retire, listener)?;
        guard.next_retire += 1;
        Ok(())
    }

    fn confirm_retired(&self, ticket: RetireTicket) -> Option<GpuError> {
        if ticket.status == POISONED_STATUS {
            self.gpu
                .poison
                .get()
                .map(|cause| GpuError::Poisoned(Box::new(cause)))
        } else {
            // Release only what this retirement covered; a poisoned ticket does not say which that was.
            self.kept.release_through(ticket.seq);
            if ticket.status != 0 {
                // An ordinary thread may ask the driver for the error's name.
                let err = self
                    .gpu
                    .check("device work", ticket.status)
                    .err()
                    .unwrap_or(GpuError::Call {
                        call: "device work",
                        code: ticket.status,
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
