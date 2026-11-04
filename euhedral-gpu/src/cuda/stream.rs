//! CUDA streams as device lanes: an event ring behind `DeviceQueue::completed`, device-side awaits, and enqueueing.

use std::sync::Arc;
use std::time::{Duration, Instant};

use euhedral_graph::{DeviceQueue, LaneControl, LaneId, Marker, QueueId};
use euhedral_memory::cuda::{CuEvent, CuStream, Device, STREAM_NON_BLOCKING};

use crate::handles::HandlePool;
use crate::kept::Kept;
use crate::retire::{RetireSink, Retirements};
use crate::spans::{Span, SpanBook};
use crate::{
    GpuError, POISONED_STATUS, RetireListener, RetireTicket, Stream, StreamOptions, StreamPriority,
};

use super::{CudaEnqueue, CudaGpu, from_cuda};
use crate::lanes::Probe;
use crate::ring::{MarkerRing, Query};

/// How often a stream with retirements outstanding looks at the device while no callback arrives: one event query.
const PROBE_EVERY: Duration = Duration::from_millis(1);

/// `CU_EVENT_DISABLE_TIMING`: the events only mark completion.
const EVENT_DISABLE_TIMING: u32 = 2;
/// `CU_EVENT_DEFAULT`: events that keep their time stamp, for spans.
const EVENT_DEFAULT: u32 = 0;
/// `CUDA_ERROR_NOT_READY`, from `cuEventQuery` while the work before the event runs.
const NOT_READY: i32 = 600;

/// A lane's marker events: value `v` is recorded in slot `v % events.len()`, by the protocol of [`MarkerRing`].
#[derive(Debug)]
pub(crate) struct LaneRing {
    /// The identity of the stream this ring belongs to, stamped on its markers.
    id: QueueId,
    device: Arc<Device>,
    events: Vec<CuEvent>,
    /// The numbers: claimed, published and complete, and the handshake with awaiters.
    ring: MarkerRing,
}

// SAFETY: events of the primary context, which any thread may record and query with the context current; the ring is
// atomics.
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
            ring: MarkerRing::new(markers as usize),
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
        self.events[self.ring.slot(value)]
    }

    /// Advances `done` over the markers that completed. The context is current. A query that fails for any reason
    /// but "not ready" poisons the device, so `completed` reports every marker and nothing is stranded behind it.
    fn scan(&self, gpu: &CudaGpu) {
        let f = &gpu.device().driver.fns;
        let failed = self.ring.scan(|slot| {
            // SAFETY: an event of this ring; the context is current.
            match unsafe { (f.event_query)(self.events[slot]) } {
                0 => Query::Done,
                NOT_READY => Query::Pending,
                other => Query::Failed(other),
            }
        });
        if let Some(code) = failed {
            gpu.fail("cuEventQuery", code);
        }
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

/// The timing events of a stream's spans: a start and an end for each slot of the ring.
#[derive(Debug)]
struct SpanEvents {
    device: Arc<Device>,
    events: Vec<(CuEvent, CuEvent)>,
}

impl SpanEvents {
    fn new(device: &Arc<Device>, slots: u32) -> Result<Self, GpuError> {
        let mut made = SpanEvents {
            device: device.clone(),
            events: Vec::with_capacity(slots as usize),
        };
        if slots == 0 {
            return Ok(made);
        }
        let _c = device.enter().map_err(from_cuda)?;
        let f = &device.driver.fns;
        for _ in 0..slots {
            let (mut start, mut end) = (std::ptr::null_mut(), std::ptr::null_mut());
            // SAFETY: an out pointer; the context is current.
            let code = unsafe { (f.event_create)(&mut start, EVENT_DEFAULT) };
            device.check("cuEventCreate", code).map_err(from_cuda)?;
            // The slot owns the start event from here, so a failure making the end destroys it.
            made.events.push((start, end));
            // SAFETY: as above.
            let code = unsafe { (f.event_create)(&mut end, EVENT_DEFAULT) };
            device.check("cuEventCreate", code).map_err(from_cuda)?;
            made.events.last_mut().expect("just pushed").1 = end;
        }
        Ok(made)
    }
}

// SAFETY: events of the primary context, usable from any thread with it current; the owner (one stream) is their only user.
unsafe impl Send for SpanEvents {}

impl Drop for SpanEvents {
    fn drop(&mut self) {
        let Ok(_c) = self.device.enter() else { return };
        let f = &self.device.driver.fns;
        for (start, end) in &self.events {
            for event in [start, end] {
                if !event.is_null() {
                    // SAFETY: events made in `new`; destroying one with work recorded on it is deferred by the driver.
                    unsafe { (f.event_destroy)(*event) };
                }
            }
        }
    }
}

/// The shared side of a CUDA stream: what the pool and `Completions` hold, and what any worker submits to (it is the
/// stream's [`Lane`](crate::Lane)). Query methods read the marker ring's atomics and ask the driver whether an event
/// completed; they never wait and take no lock. It owns the driver stream, which is destroyed (after its work) when the
/// last handle goes.
#[derive(Debug)]
pub struct CudaQueue {
    pub(crate) gpu: Arc<CudaGpu>,
    pub(crate) ring: Arc<LaneRing>,
    name: String,
    pub(crate) lane: LaneId,
    /// The driver stream this queue stands for: usable from any thread with the context current.
    pub(crate) raw: CuStream,
    /// The rings of the lanes the opening call made, at their lane ids; the ids below the call's first lane are
    /// `None`. An await accepts only these lanes' markers.
    lanes: Arc<[Option<Arc<LaneRing>>]>,
    /// The callback's end of the stream's retirement queue. Its address is the callbacks' user data, so it must outlive
    /// every callback the stream armed. The stream's drop waits for them (`cuStreamSynchronize`); if that wait cannot
    /// complete (the context is lost, or the wait fails) and retirements are still outstanding, the drop leaks one
    /// reference to this shared side, so the sink is never freed under a callback that is still to run.
    sink: RetireSink,
    /// The private streams of this lane, stocked: a worker takes one to replay a graph or to capture on, uses it alone,
    /// and gives it back (see [`Fork`]).
    forks: HandlePool,
}

/// The most private streams a lane keeps stocked; a stream given back to a full stock is destroyed.
const FORKS_STOCKED: usize = 64;

/// A private stream of a lane, with the two events that fork it from the lane and join it back.
///
/// Several threads submitting to one stream let a `cuGraphLaunch` overtake the `cuStreamWaitEvent` its own thread issued
/// just before it (seen with driver 615.71.09 on CUDA 13.4: a graph's work then runs before what it waited for; plain
/// launches, a single submitter per stream, and a wait and launch locked together do not do it). So a replay never launches
/// on the lane: its worker forks a stream of its own from the lane (an event recorded on the lane, awaited by the private
/// stream), does its cross-lane waits and the launch there, and joins the lane back (an event recorded on the private
/// stream, awaited by the lane); the marker is recorded on the lane after the join. One stream is used by one worker at a
/// time, so each launches alone on its stream. The same streams are the shadow streams a recording captures on.
#[derive(Debug)]
pub(crate) struct Fork {
    pub(crate) stream: CuStream,
    pub(crate) fork: CuEvent,
    pub(crate) join: CuEvent,
}

// SAFETY: driver handles of the primary context, used by one thread at a time (taken from the stock, given back).
unsafe impl Send for Fork {}

// SAFETY: the stream handle belongs to the primary context and is usable from any thread with it current; everything
// else is atomics, immutable, or reached through a claim.
unsafe impl Send for CudaQueue {}
// SAFETY: as above.
unsafe impl Sync for CudaQueue {}

impl CudaQueue {
    pub(crate) fn sink(&self) -> &RetireSink {
        &self.sink
    }

    /// A private stream of this lane: one of the stock, else a new one. The context is current. Give it back with
    /// [`give_fork`](Self::give_fork).
    pub(crate) fn take_fork(&self) -> Result<Box<Fork>, GpuError> {
        if let Some(handle) = self.forks.take() {
            // SAFETY: the handle was made by `Box::into_raw` below and is held by no one else once taken.
            return Ok(unsafe { Box::from_raw(handle as *mut Fork) });
        }
        let f = &self.gpu.device().driver.fns;
        let mut stream: CuStream = std::ptr::null_mut();
        // SAFETY: an out pointer; the context is current.
        self.gpu.check("cuStreamCreate", unsafe {
            (f.stream_create)(&mut stream, STREAM_NON_BLOCKING)
        })?;
        let mut events: [CuEvent; 2] = [std::ptr::null_mut(); 2];
        for at in 0..2 {
            // SAFETY: an out pointer; the context is current. `2` is `CU_EVENT_DISABLE_TIMING`.
            let made = self.gpu.check("cuEventCreate", unsafe {
                (f.event_create)(&mut events[at], 2)
            });
            if let Err(e) = made {
                // SAFETY: handles made above, unused.
                unsafe {
                    for event in &events[..at] {
                        (f.event_destroy)(*event);
                    }
                    (f.stream_destroy)(stream);
                }
                return Err(e);
            }
        }
        Ok(Box::new(Fork {
            stream,
            fork: events[0],
            join: events[1],
        }))
    }

    /// Gives a private stream back to the stock (destroyed if the stock is full). The context is current. Whatever was
    /// enqueued on the stream stays ordered on it; a stream that was capturing must have ended its capture.
    pub(crate) fn give_fork(&self, fork: Box<Fork>) {
        let handle = Box::into_raw(fork);
        if let Err(handle) = self.forks.give(handle as usize) {
            // SAFETY: the pointer made above, given to no one.
            destroy_fork(&self.gpu, &unsafe { *Box::from_raw(handle as *mut Fork) });
        }
    }

    /// Claims the next marker, ahead of the work it will follow. `Err(Busy)` before anything is submitted.
    pub(crate) fn reserve(&self) -> Result<crate::ring::Claim<'_>, GpuError> {
        self.gpu.poison.check()?;
        self.ring.ring.reserve(|| self.ring.scan(&self.gpu))
    }

    /// A marker after the work submitted so far (which may include other threads' work: that only over-waits). The
    /// context is current. Any thread may call it.
    pub(crate) fn mark_now(&self) -> Result<Marker, GpuError> {
        let claim = self.reserve()?;
        self.record_claim(claim)
    }

    /// Later work on `target` (this stream, or its shadow) waits for `marker`. The context is current.
    pub(crate) fn await_now(&self, target: CuStream, marker: Marker) -> Result<(), GpuError> {
        self.gpu.poison.check()?;
        let other = self
            .lanes
            .get(usize::from(marker.lane.0))
            .and_then(Option::as_ref)
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
        if marker.value > other.ring.claimed() {
            return Err(GpuError::Invalid("a marker that was never made".into()));
        }
        if marker.value <= other.ring.done() {
            return Ok(());
        }
        other.scan(&self.gpu);
        if marker.value <= other.ring.done() {
            return Ok(());
        }
        // The slot must still hold this value's record when the wait is enqueued (see `MarkerRing`).
        let Some(_pin) = other.ring.pin(marker.value) else {
            return Ok(());
        };
        // SAFETY: an event of the pool's ring and a stream of this device; the pin keeps a claimant from re-recording
        // the slot; the context is current.
        self.gpu.check("cuStreamWaitEvent", unsafe {
            (self.gpu.fns.stream_wait_event)(target, other.slot(marker.value), 0)
        })
    }

    /// Whether `marker` names a lane this stream's opening call made, under that lane's queue.
    pub(crate) fn owns_marker_lane(&self, marker: Marker) -> bool {
        self.lanes
            .get(usize::from(marker.lane.0))
            .and_then(Option::as_ref)
            .is_some_and(|ring| ring.id == marker.queue)
    }

    /// One look at the device: a query of the stream's marker events, which poisons the device when the query fails for
    /// any reason but "not ready".
    pub(crate) fn look(&self) {
        let _ = DeviceQueue::completed(self);
    }

    /// A marker for the claim `claim`, recorded after the work submitted so far. The context is current.
    pub(crate) fn record_claim(&self, claim: crate::ring::Claim<'_>) -> Result<Marker, GpuError> {
        let ring = &self.ring;
        let f = &self.gpu.device().driver.fns;
        let value = claim.record(|slot| {
            // SAFETY: the slot's event and this stream; the slot's previous marker completed and no awaiter holds the
            // slot (`MarkerRing::reserve`); the context is current.
            self.gpu.check("cuEventRecord", unsafe {
                (f.event_record)(ring.events[slot], self.raw)
            })
        })?;
        Ok(Marker::new(self.ring.id, self.lane, value))
    }
}

/// Destroys a private stream and its events (after the work on it ended: its lane waited for the join).
fn destroy_fork(gpu: &CudaGpu, fork: &Fork) {
    let f = &gpu.device().driver.fns;
    // SAFETY: handles made by `take_fork`, no longer in use; the context is current.
    unsafe {
        (f.stream_synchronize)(fork.stream);
        (f.event_destroy)(fork.fork);
        (f.event_destroy)(fork.join);
        (f.stream_destroy)(fork.stream);
    }
}

impl Drop for CudaQueue {
    /// The last handle is gone: the stream's work ends before the stream is destroyed. A retirement a lane armed holds
    /// its own clone of the cell the callback fills, so no callback can outlive what it writes.
    fn drop(&mut self) {
        if let Ok(_c) = self.gpu.device().enter() {
            let f = &self.gpu.device().driver.fns;
            // SAFETY: the stream this value stands for, no longer in use; the context is current.
            unsafe { (f.stream_synchronize)(self.raw) };
            for handle in self.forks.drain() {
                // SAFETY: the stock holds pointers made by `give_fork`, which no one else holds.
                destroy_fork(&self.gpu, &unsafe { *Box::from_raw(handle as *mut Fork) });
            }
            // SAFETY: as above.
            unsafe { (f.stream_destroy)(self.raw) };
        }
    }
}

impl DeviceQueue for CudaQueue {
    fn id(&self) -> QueueId {
        self.ring.id
    }

    fn completed(&self) -> u64 {
        if self.gpu.poison.get().is_some() {
            return self.ring.ring.claimed();
        }
        if let Ok(_c) = self.gpu.device().enter().map_err(from_cuda) {
            self.ring.scan(&self.gpu);
        }
        if self.gpu.poison.get().is_some() {
            return self.ring.ring.claimed();
        }
        self.ring.ring.done()
    }

    fn issued(&self) -> u64 {
        self.ring.ring.claimed()
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn control(&self) -> Option<&dyn LaneControl> {
        Some(self)
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
    pub(crate) queue: Arc<CudaQueue>,
    /// The stream a recording captures this stream's work on, made on first use.
    pub(crate) shadow: Option<CuStream>,
    /// The owner's end of the retirement queue, and the retirements armed.
    pub(crate) retire: Retirements,
    pub(crate) kept: Kept,
    /// When the stream last queried its marker events for a fault with retirements outstanding.
    last_probe: Option<Instant>,
    /// The ring of timed spans: its slots, and the timing events behind them.
    spans: SpanBook,
    span_events: SpanEvents,
}

// SAFETY: the stream belongs to the primary context and is usable from any thread with it current. A submitter has
// one owner at a time (every enqueue takes `&mut self`), so moving it to another thread moves the whole of its state.
// It is deliberately not `Sync`: the `Kept` list holds values that are `Send` only.
unsafe impl Send for CudaStream {}

impl CudaStream {
    /// New streams `names[i]` as lane `first + i`, each with a ring of `options[i].markers` events and the priority
    /// `options[i].priority` names.
    pub(crate) fn open(
        gpu: &Arc<CudaGpu>,
        first: LaneId,
        names: &[&str],
        options: &[StreamOptions],
    ) -> Result<Vec<CudaStream>, GpuError> {
        let first = usize::from(first.0);
        if names.is_empty() || first + names.len() > 64 {
            return Err(GpuError::Invalid(
                "a pool has 1 to 64 lanes, and the streams of a call fit in it".into(),
            ));
        }
        if options.len() != names.len() {
            return Err(GpuError::Invalid("one options entry per lane name".into()));
        }
        if options.iter().any(|o| o.markers == 0) {
            return Err(GpuError::Invalid("a lane needs at least one marker".into()));
        }
        gpu.poison.check()?;
        let device = gpu.device();
        let rings = options
            .iter()
            .map(|o| LaneRing::new(QueueId::next(), device, o.markers).map(Arc::new))
            .collect::<Result<Vec<_>, _>>()?;
        let mut slots: Vec<Option<Arc<LaneRing>>> = vec![None; first];
        slots.extend(rings.iter().cloned().map(Some));
        let lanes: Arc<[Option<Arc<LaneRing>>]> = slots.into();
        let _c = device.enter().map_err(from_cuda)?;
        let greatest = if options.iter().any(|o| o.priority == StreamPriority::High) {
            let (mut least, mut greatest) = (0, 0);
            // SAFETY: out pointers; the context is current.
            let code =
                unsafe { (gpu.fns.ctx_get_stream_priority_range)(&mut least, &mut greatest) };
            gpu.check("cuCtxGetStreamPriorityRange", code)?;
            greatest
        } else {
            0
        };
        let mut streams = Vec::with_capacity(names.len());
        for (i, name) in names.iter().enumerate() {
            let priority = match options[i].priority {
                StreamPriority::Normal => 0,
                StreamPriority::High => greatest,
            };
            let mut raw = std::ptr::null_mut();
            // SAFETY: an out pointer; the context is current.
            let code = unsafe {
                (gpu.fns.stream_create_with_priority)(&mut raw, STREAM_NON_BLOCKING, priority)
            };
            gpu.check("cuStreamCreateWithPriority", code)?;
            let lane = LaneId(u16::try_from(first + i).expect("at most 64 lanes"));
            let ring = rings[i].clone();
            let ring_id = ring.id;
            let (retire, sink) = Retirements::new(ring.id, lane);
            streams.push(CudaStream {
                gpu: gpu.clone(),
                id: ring.id,
                lane,
                raw,
                queue: Arc::new(CudaQueue {
                    gpu: gpu.clone(),
                    ring,
                    name: (*name).to_owned(),
                    lane,
                    raw,
                    lanes: lanes.clone(),
                    sink,
                    forks: HandlePool::new(FORKS_STOCKED),
                }),
                shadow: None,
                retire,
                kept: Kept::default(),
                last_probe: None,
                spans: SpanBook::new(ring_id, options[i].spans),
                span_events: SpanEvents::new(device, options[i].spans)?,
            });
        }
        Ok(streams)
    }

    /// The stream's priority as the driver reports it (`cuStreamGetPriority`): 0 for [`StreamPriority::Normal`], the
    /// device's greatest (numerically lowest) for [`StreamPriority::High`].
    pub fn priority(&self) -> Result<i32, GpuError> {
        let _c = self.enter()?;
        let mut priority = 0;
        // SAFETY: this stream and an out pointer; the context is current.
        let code = unsafe { (self.gpu.fns.stream_get_priority)(self.raw, &mut priority) };
        self.gpu.check("cuStreamGetPriority", code)?;
        Ok(priority)
    }

    /// A marker after the work submitted so far. The context is current.
    pub(crate) fn mark_now(&self) -> Result<Marker, GpuError> {
        self.queue.mark_now()
    }

    /// Later work on `target` (this stream, or its shadow) waits for `marker`. The context is current.
    pub(crate) fn await_now(&self, target: CuStream, marker: Marker) -> Result<(), GpuError> {
        self.queue.await_now(target, marker)
    }

    /// One look at the device: a query of the stream's marker events, which poisons the device when the query fails for
    /// any reason but "not ready". A retirement armed behind a lane that faulted may never get its callback (the work
    /// it follows never completes), and nothing else would notice; this is how Java's `confirmRetired` learns of it,
    /// from the completion event. It costs one `cuEventQuery`, and nothing when every marker completed.
    ///
    /// [`Probe::Idle`] looks only while retirements are outstanding, and no more often than every [`PROBE_EVERY`].
    /// [`Probe::Confirm`] looks whenever a marker is still to complete: the ticket it confirms is already taken, so it
    /// may have been the last retirement in flight.
    pub(crate) fn probe_fault(&mut self, mode: Probe) {
        if mode == Probe::Idle && self.retire.outstanding() == 0 {
            return;
        }
        if self.queue.ring.ring.done() >= self.queue.ring.ring.claimed() {
            return;
        }
        if mode == Probe::Idle {
            let now = Instant::now();
            if self
                .last_probe
                .is_some_and(|last| now.duration_since(last) < PROBE_EVERY)
            {
                return;
            }
            self.last_probe = Some(now);
        }
        self.queue.look();
    }

    /// Whether `marker` names a lane this stream's opening call made, under that lane's queue.
    pub(crate) fn owns_marker_lane(&self, marker: Marker) -> bool {
        self.queue.owns_marker_lane(marker)
    }

    /// Arms a retirement: the callback is enqueued first, and registered once the driver accepted it.
    fn arm(&mut self, listener: Option<Box<dyn RetireListener>>) -> Result<(), GpuError> {
        self.gpu.poison.check()?;
        let gpu = self.gpu.clone();
        let _c = gpu.device().enter().map_err(from_cuda)?;
        self.arm_callback()?;
        self.retire.register(listener);
        Ok(())
    }

    /// The context current on the calling thread for as long as the guard lives.
    pub(crate) fn enter(&self) -> Result<euhedral_memory::cuda::Current<'_>, GpuError> {
        self.gpu.device().enter().map_err(from_cuda)
    }
}

/// How many shared sides were leaked by streams that could not wait for their callbacks (test builds only).
#[cfg(feature = "test-hooks")]
static LEAKED_SHARED_SIDES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// A test hook (feature `test-hooks`): how many stream drops leaked their shared side.
#[cfg(feature = "test-hooks")]
pub fn leaked_shared_sides() -> u64 {
    LEAKED_SHARED_SIDES.load(std::sync::atomic::Ordering::Relaxed)
}

impl Drop for CudaStream {
    /// A shutdown call: it waits for the stream's work, and every callback it armed, to end.
    fn drop(&mut self) {
        let mut waited = false;
        if let Ok(_c) = self.gpu.device().enter() {
            let f = &self.gpu.device().driver.fns;
            // SAFETY: streams made by this value, no longer in use; the context is current.
            let (a, b) = unsafe {
                (
                    (f.stream_synchronize)(self.raw),
                    self.shadow
                        .map_or(0, |shadow| (f.stream_synchronize)(shadow)),
                )
            };
            waited = a == 0 && b == 0;
            self.kept.clear();
            // The driver stream itself belongs to the shared queue, which destroys it with its last handle.
            // SAFETY: as above.
            unsafe {
                if let Some(shadow) = self.shadow {
                    (f.stream_destroy)(shadow);
                }
            }
        }
        if !waited && self.retire.callbacks_pending() {
            // The wait did not complete (the context is lost, or a sticky fault made it return at once) while a callback
            // armed here is still to arrive (more retirements armed than statuses seen, counting those reported as
            // poisoned before their callbacks came), and each pushes into the shared side's sink. Keep that side alive
            // for good: a leaked reference is the price of never freeing the sink under a callback.
            std::mem::forget(self.queue.clone());
            #[cfg(feature = "test-hooks")]
            LEAKED_SHARED_SIDES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }
}

impl Stream for CudaStream {
    type Gpu = CudaGpu;
    type Enqueue<'a> = CudaEnqueue<'a>;
    type Lane = CudaQueue;

    /// Synchronizes the stream (and its shadow stream): the work submitted so far and its callbacks have ended on return,
    /// unless the context is lost.
    fn quiesce(&mut self) {
        if let Ok(_c) = self.gpu.device().enter() {
            let f = &self.gpu.device().driver.fns;
            // SAFETY: streams made by this value; the context is current.
            unsafe {
                (f.stream_synchronize)(self.raw);
                if let Some(shadow) = self.shadow {
                    (f.stream_synchronize)(shadow);
                }
            }
        }
    }

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

    fn shared(&self) -> Arc<CudaQueue> {
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

    fn span_begin(&mut self) -> Result<Span, GpuError> {
        self.gpu.poison.check()?;
        let span = self.spans.begin()?;
        let (start, _) = self.span_events.events[span.slot()];
        let gpu = self.gpu.clone();
        let _c = gpu.device().enter().map_err(from_cuda)?;
        // SAFETY: an event of this stream's ring and this stream; the context is current.
        let recorded = self.gpu.check("cuEventRecord", unsafe {
            (self.gpu.device().driver.fns.event_record)(start, self.raw)
        });
        if let Err(e) = recorded {
            let _ = self.spans.release(span);
            return Err(e);
        }
        Ok(span)
    }

    fn span_end(&mut self, span: Span) -> Result<(), GpuError> {
        self.gpu.poison.check()?;
        let at = self.spans.end(span)?;
        let (_, end) = self.span_events.events[at];
        let gpu = self.gpu.clone();
        let _c = gpu.device().enter().map_err(from_cuda)?;
        // SAFETY: as in `span_begin`.
        self.gpu.check("cuEventRecord", unsafe {
            (self.gpu.device().driver.fns.event_record)(end, self.raw)
        })
    }

    fn span_poll(&mut self, span: Span) -> Result<Option<Duration>, GpuError> {
        let at = self.spans.closed(span)?;
        if self.gpu.poison.get().is_some() {
            self.spans.release(span)?;
            return Err(GpuError::Poisoned(Box::new(
                self.gpu.poison.get().expect("checked"),
            )));
        }
        let (start, end) = self.span_events.events[at];
        let gpu = self.gpu.clone();
        let _c = gpu.device().enter().map_err(from_cuda)?;
        let f = &gpu.device().driver.fns;
        // SAFETY: an event of this stream's ring; the context is current.
        match unsafe { (f.event_query)(end) } {
            0 => {}
            NOT_READY => return Ok(None),
            other => return Err(self.gpu.fail("cuEventQuery", other)),
        }
        let mut ms = 0f32;
        // SAFETY: two completed events of this stream's ring, recorded on one stream; an out pointer.
        self.gpu.check("cuEventElapsedTime", unsafe {
            (f.event_elapsed_time)(&mut ms, start, end)
        })?;
        self.spans.release(span)?;
        Ok(Some(Duration::from_secs_f64(
            f64::from(ms.max(0.0)) / 1000.0,
        )))
    }

    fn span_cancel(&mut self, span: Span) -> Result<(), GpuError> {
        self.spans.release(span)
    }

    fn await_marker(&mut self, marker: Marker) -> Result<(), GpuError> {
        let _c = self.enter()?;
        self.await_now(self.raw, marker)
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
            self.gpu
                .fail_invalid("a retirement outcome arrived that no one had armed");
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
        self.kept.release_through(ticket.seq());
        if ticket.status() == 0 {
            return poisoned();
        }
        // Asking the driver for the error's name is a plain call. A device fault poisons the device there, so the
        // retirement that carries the fault reports what a retirement behind it reports, whichever of the callback
        // and the idle look at the device saw the fault first.
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
