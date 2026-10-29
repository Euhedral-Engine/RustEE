//! CUDA streams as device lanes: an event ring behind `DeviceQueue::completed`, device-side awaits, and enqueueing.

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use euhedral_graph::{DeviceQueue, LaneId, Marker, QueueId};
use euhedral_memory::cuda::{CuEvent, CuStream, Device, STREAM_NON_BLOCKING};

use crate::kept::Kept;
use crate::retire::{RetireSink, Retirements};
use crate::{
    GpuError, POISONED_STATUS, RetireListener, RetireTicket, Stream, StreamOptions, StreamPriority,
};

use super::{CudaEnqueue, CudaGpu, from_cuda};

/// How often a stream with retirements outstanding looks at the device while no callback arrives: one event query.
const PROBE_EVERY: Duration = Duration::from_millis(1);

/// `CU_EVENT_DISABLE_TIMING`: the events only mark completion.
const EVENT_DISABLE_TIMING: u32 = 2;
/// `CUDA_ERROR_NOT_READY`, from `cuEventQuery` while the work before the event runs.
const NOT_READY: i32 = 600;

/// Keeps a marker slot from being re-recorded under an awaiter, without a lock.
///
/// Value `v` is recorded in slot `v % n`; the owner may record `v + n` there once `v` is complete. An awaiter of `v`
/// that has found `v` incomplete must enqueue its wait on the slot's event *while the slot still holds `v`'s record*,
/// or it would wait for the later value `v + n`, which can deadlock. The two sides meet in a Dekker handshake of
/// sequentially consistent operations:
/// - the **owner** announces the value it is about to record (`reserved = w`) and then reads the slot's pin count:
///   a pinned slot is not recorded, and the owner returns `Busy` (it never waits);
/// - an **awaiter** raises the slot's pin count and then reads `reserved`: if the owner is already past `v`
///   (`reserved >= v + n`) the slot is being (or has been) re-recorded, which the owner does only when `v` is
///   complete, so there is nothing to wait for and the pin is dropped.
///
/// Whichever of the two reads comes second sees the other's write, so an awaiter that keeps its pin always has the
/// slot as `v` left it, until it lowers the pin after `cuStreamWaitEvent`. A pin is held for one driver call.
#[derive(Debug)]
struct SlotPins {
    pins: Vec<AtomicU32>,
    reserved: AtomicU64,
}

/// A raised pin on a marker slot; lowers it on drop.
struct Pin<'a>(&'a AtomicU32);

impl Drop for Pin<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

impl SlotPins {
    fn new(slots: usize) -> Self {
        Self {
            pins: (0..slots).map(|_| AtomicU32::new(0)).collect(),
            reserved: AtomicU64::new(0),
        }
    }

    fn slots(&self) -> u64 {
        self.pins.len() as u64
    }

    /// The owner is about to record value `w`. `Err(Busy)` if an awaiter holds the slot (nothing is reserved then).
    fn begin_record(&self, w: u64) -> Result<(), GpuError> {
        self.reserved.store(w, Ordering::SeqCst);
        if self.pins[(w % self.slots()) as usize].load(Ordering::SeqCst) != 0 {
            self.reserved.store(w - 1, Ordering::SeqCst);
            return Err(GpuError::Busy);
        }
        Ok(())
    }

    /// An awaiter of value `v`. `Some(pin)`: the slot holds `v`'s record until the pin drops. `None`: the owner has
    /// started re-recording the slot, so `v` is complete and needs no wait.
    fn pin(&self, v: u64) -> Option<Pin<'_>> {
        let slot = &self.pins[(v % self.slots()) as usize];
        slot.fetch_add(1, Ordering::SeqCst);
        if self.reserved.load(Ordering::SeqCst) >= v + self.slots() {
            slot.fetch_sub(1, Ordering::SeqCst);
            return None;
        }
        Some(Pin(slot))
    }
}

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
    /// Keeps a slot from being re-recorded under an awaiter.
    pins: SlotPins,
}

// SAFETY: events of the primary context; `marked` has one writer (the stream's owner) and `done` only grows; the pins
// are atomics.
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
            pins: SlotPins::new(markers as usize),
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
    /// The callback's end of the stream's retirement queue. Its address is the callbacks' user data, so it must outlive
    /// every callback the stream armed. The stream's drop waits for them (`cuStreamSynchronize`); if that wait cannot
    /// complete (the context is lost, or the wait fails) and retirements are still outstanding, the drop leaks one
    /// reference to this shared side, so the sink is never freed under a callback that is still to run.
    sink: RetireSink,
}

impl CudaQueue {
    pub(crate) fn sink(&self) -> &RetireSink {
        &self.sink
    }
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
    pub(crate) queue: Arc<CudaQueue>,
    ring: Arc<LaneRing>,
    /// The rings of the lanes the opening call made, at their lane ids; the ids below the call's first lane are
    /// `None`. An await accepts only these lanes' markers.
    lanes: Arc<[Option<Arc<LaneRing>>]>,
    /// The stream a recording captures this stream's work on, made on first use.
    pub(crate) shadow: Option<CuStream>,
    /// The owner's end of the retirement queue, and the retirements armed.
    pub(crate) retire: Retirements,
    pub(crate) kept: Kept,
    /// When the stream last queried its marker events for a fault with retirements outstanding.
    last_probe: Option<Instant>,
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
            let (retire, sink) = Retirements::new(ring.id, lane);
            streams.push(CudaStream {
                gpu: gpu.clone(),
                id: ring.id,
                lane,
                raw,
                queue: Arc::new(CudaQueue {
                    gpu: gpu.clone(),
                    ring: ring.clone(),
                    name: (*name).to_owned(),
                    sink,
                }),
                ring,
                lanes: lanes.clone(),
                shadow: None,
                retire,
                kept: Kept::default(),
                last_probe: None,
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
        // An awaiter may be about to wait on the slot's previous record: then this is `Busy`, not a wait.
        ring.pins.begin_record(value)?;
        let f = &self.gpu.device().driver.fns;
        // SAFETY: the slot's event and this stream; the slot's previous marker completed (checked above), and no
        // awaiter holds the slot (`begin_record`).
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
        // The slot must still hold this value's record when the wait is enqueued (see `SlotPins`).
        let Some(_pin) = other.pins.pin(marker.value) else {
            return Ok(());
        };
        // SAFETY: an event of the pool's ring and a stream of this device; the pin keeps the owner from re-recording
        // the slot; the context is current.
        self.gpu.check("cuStreamWaitEvent", unsafe {
            (self.gpu.fns.stream_wait_event)(target, other.slot(marker.value), 0)
        })
    }

    /// One look at the device on behalf of a stream with retirements outstanding: a query of its marker events, which
    /// poisons the device when the query fails for any reason but "not ready". A retirement armed behind a lane that
    /// faulted may never get its callback (the work it follows never completes), and nothing else would notice; this
    /// is how Java's `confirmRetired` learns of it, from the completion event. It costs one `cuEventQuery`, and nothing
    /// when every marker completed. With `throttled`, no more often than every [`PROBE_EVERY`].
    pub(crate) fn probe_fault(&mut self, throttled: bool) {
        if self.retire.outstanding() == 0 {
            return;
        }
        if self.ring.done.load(Ordering::Acquire) >= self.ring.marked.load(Ordering::Acquire) {
            return;
        }
        if throttled {
            let now = Instant::now();
            if self
                .last_probe
                .is_some_and(|last| now.duration_since(last) < PROBE_EVERY)
            {
                return;
            }
            self.last_probe = Some(now);
        }
        let _ = DeviceQueue::completed(&*self.queue);
    }

    /// Whether `marker` names a lane this stream's opening call made, under that lane's queue.
    pub(crate) fn owns_marker_lane(&self, marker: Marker) -> bool {
        self.lanes
            .get(usize::from(marker.lane.0))
            .and_then(Option::as_ref)
            .is_some_and(|ring| ring.id == marker.queue)
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
static LEAKED_SHARED_SIDES: AtomicU64 = AtomicU64::new(0);

/// A test hook (feature `test-hooks`): how many stream drops leaked their shared side.
#[cfg(feature = "test-hooks")]
pub fn leaked_shared_sides() -> u64 {
    LEAKED_SHARED_SIDES.load(Ordering::Relaxed)
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
            // SAFETY: as above.
            unsafe {
                (f.stream_destroy)(self.raw);
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
            LEAKED_SHARED_SIDES.fetch_add(1, Ordering::Relaxed);
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
        if ticket.status() == POISONED_STATUS {
            self.gpu
                .poison
                .get()
                .map(|cause| GpuError::Poisoned(Box::new(cause)))
        } else {
            // Release only what this retirement covered; a poisoned ticket does not say which that was.
            self.kept.release_through(ticket.seq());
            if ticket.status() != 0 {
                // Asking the driver for the error's name is a plain call.
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

    #[test]
    fn a_pinned_slot_is_not_recorded_over() {
        let pins = SlotPins::new(4);
        // Value 1 is made; an awaiter of 1 pins its slot.
        pins.begin_record(1).unwrap();
        let pin = pins.pin(1).expect("nothing past value 1 yet");
        // Value 5 would reuse slot 1: refused while the pin is held, and nothing stays reserved.
        assert_eq!(pins.begin_record(5), Err(GpuError::Busy));
        assert!(pins.pin(1).is_some(), "the refused record reserved nothing");
        drop(pin);
        assert!(pins.begin_record(5).is_ok(), "released with the pin");
    }

    #[test]
    fn an_awaiter_that_arrives_after_the_owner_moved_on_skips_the_wait() {
        let pins = SlotPins::new(4);
        pins.begin_record(1).unwrap();
        // Value 1 completed, and the owner started recording value 5 over its slot.
        pins.begin_record(5).unwrap();
        assert!(
            pins.pin(1).is_none(),
            "the slot no longer holds value 1, which is complete: no wait"
        );
        // The refused pin left nothing raised, so the owner is not held up by it.
        assert!(pins.begin_record(5).is_ok());
    }

    /// An owner re-records two slots as fast as it can while an awaiter pins the latest value: a pin that is granted
    /// always finds the slot as its value left it, for as long as the pin is up.
    #[test]
    #[allow(
        clippy::disallowed_methods,
        reason = "two threads stand in for the owner of a lane and an awaiter on another lane"
    )]
    fn a_granted_pin_always_sees_its_own_value_in_the_slot() {
        use std::sync::atomic::AtomicBool;
        const SLOTS: u64 = 2;
        let pins = SlotPins::new(SLOTS as usize);
        let recorded: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];
        let marked = AtomicU64::new(0);
        let stop = AtomicBool::new(false);
        let granted = AtomicU64::new(0);
        std::thread::scope(|scope| {
            scope.spawn(|| {
                let mut w = 1;
                while w <= 200_000 {
                    // Busy is retried: the owner never waits on an awaiter, it tries again.
                    if pins.begin_record(w).is_ok() {
                        recorded[(w % SLOTS) as usize].store(w, Ordering::SeqCst);
                        marked.store(w, Ordering::SeqCst);
                        w += 1;
                    }
                }
                stop.store(true, Ordering::SeqCst);
            });
            scope.spawn(|| {
                while !stop.load(Ordering::SeqCst) {
                    let v = marked.load(Ordering::SeqCst);
                    if v == 0 {
                        continue;
                    }
                    if let Some(_pin) = pins.pin(v) {
                        for _ in 0..4 {
                            assert_eq!(
                                recorded[(v % SLOTS) as usize].load(Ordering::SeqCst),
                                v,
                                "the slot was re-recorded under a pin"
                            );
                        }
                        granted.fetch_add(1, Ordering::Relaxed);
                    }
                }
            });
        });
        assert!(
            granted.load(Ordering::Relaxed) > 0,
            "some pins were granted"
        );
    }
}
