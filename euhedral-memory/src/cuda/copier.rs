//! Asynchronous copies between a device and host memory for the [`Mover`](crate::Mover): one stream, and a ring of
//! completion events created once.

use std::ffi::c_void;
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use euhedral_data_structures::queues::{Consumer, MpmcQueue, Multi, Producer};

use super::{CuEvent, CuStream, CudaError, Device, Memcpy2D, STREAM_NON_BLOCKING};
use crate::backend::Retired;
use crate::planner::LocationId;
use crate::tiers::{Copier, CopyError, PitchedCopy};

/// `CU_EVENT_DISABLE_TIMING`: the events only mark completion.
const EVENT_DISABLE_TIMING: u32 = 2;
/// `CUDA_SUCCESS`, and `CUDA_ERROR_NOT_READY` from `cuEventQuery` while the work before the event runs.
const SUCCESS: i32 = 0;

struct Inner {
    device: Arc<Device>,
    stream: CuStream,
    /// Whether this copier made the stream (and so synchronizes and destroys it), or borrows the caller's.
    owns_stream: bool,
    events: Vec<CuEvent>,
    free: (Producer<u32, Multi>, Consumer<u32, Multi>),
    copies: AtomicU64,
}

// SAFETY: the stream and events belong to the device's primary context, usable from any thread with it current;
// each event is used by one copy at a time (the free ring hands it out once).
unsafe impl Send for Inner {}
// SAFETY: as above; the handles are only read.
unsafe impl Sync for Inner {}

impl Drop for Inner {
    fn drop(&mut self) {
        let Ok(_c) = self.device.enter() else { return };
        let f = &self.device.driver.fns;
        // SAFETY: handles created in `StreamCopier::new` or `on_stream`; every copy's test held an `Arc` of this, so
        // none is in use. A borrowed stream is left to its owner; a pending event is released by the driver once the
        // stream passes it.
        unsafe {
            if self.owns_stream {
                (f.stream_synchronize)(self.stream);
            }
            for e in &self.events {
                (f.event_destroy)(*e);
            }
            if self.owns_stream {
                (f.stream_destroy)(self.stream);
            }
        }
    }
}

/// Copies between a device's locations and host locations (pinned for full speed; pageable works, staged by the
/// driver) with `cuMemcpy{HtoD,DtoH,DtoD}Async` on one non-blocking stream. Each copy records one event from a ring
/// created up front; the copy's completion test queries it and hands it back. A copy that finds every event in use
/// is refused (`CopyError`): size the ring for the moves in flight.
pub struct StreamCopier {
    inner: Arc<Inner>,
    device_locations: Vec<LocationId>,
    host_locations: Vec<LocationId>,
}

impl fmt::Debug for StreamCopier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StreamCopier")
            .field("device_locations", &self.device_locations)
            .field("host_locations", &self.host_locations)
            .field("events", &self.inner.events.len())
            .field("copies", &self.copies())
            .finish()
    }
}

impl StreamCopier {
    /// A copier for `device`'s memory at `device_locations` and host memory at `host_locations`, with `events`
    /// completion events (at least one).
    pub fn new(
        device: Arc<Device>,
        device_locations: impl IntoIterator<Item = LocationId>,
        host_locations: impl IntoIterator<Item = LocationId>,
        events: u32,
    ) -> Result<Self, CudaError> {
        let _c = device.enter()?;
        let f = &device.driver.fns;
        let mut stream = std::ptr::null_mut();
        // SAFETY: out pointer; the context is current; destroyed in `Inner::drop`.
        device.check("cuStreamCreate", unsafe {
            (f.stream_create)(&mut stream, STREAM_NON_BLOCKING)
        })?;
        drop(_c);
        Self::build(
            device,
            stream,
            true,
            device_locations.into_iter().collect(),
            host_locations.into_iter().collect(),
            events,
        )
    }

    /// A copier that queues its copies on the caller's `stream` instead of one of its own, so they order with the
    /// caller's other work on it. The copier never synchronizes or destroys the stream.
    ///
    /// # Safety
    /// `stream` is a live stream of `device`'s primary context, and stays alive until every copy this copier
    /// started has retired and the copier has dropped.
    pub unsafe fn on_stream(
        device: Arc<Device>,
        stream: CuStream,
        device_locations: impl IntoIterator<Item = LocationId>,
        host_locations: impl IntoIterator<Item = LocationId>,
        events: u32,
    ) -> Result<Self, CudaError> {
        Self::build(
            device,
            stream,
            false,
            device_locations.into_iter().collect(),
            host_locations.into_iter().collect(),
            events,
        )
    }

    fn build(
        device: Arc<Device>,
        stream: CuStream,
        owns_stream: bool,
        device_locations: Vec<LocationId>,
        host_locations: Vec<LocationId>,
        events: u32,
    ) -> Result<Self, CudaError> {
        let n = events.max(1);
        let _c = device.enter()?;
        let f = &device.driver.fns;
        let q = MpmcQueue::<u32>::new((n as usize).next_power_of_two());
        let (p, c) = (q.producer(), q.consumer());
        let mut list = Vec::with_capacity(n as usize);
        for i in 0..n {
            let mut e = std::ptr::null_mut();
            // SAFETY: out pointer; the context is current.
            let made = device.check("cuEventCreate", unsafe {
                (f.event_create)(&mut e, EVENT_DISABLE_TIMING)
            });
            if let Err(err) = made {
                // SAFETY: the handles made so far, unused.
                unsafe {
                    for e in &list {
                        (f.event_destroy)(*e);
                    }
                    if owns_stream {
                        (f.stream_destroy)(stream);
                    }
                }
                return Err(err);
            }
            list.push(e);
            let _ = p.offer(i);
        }
        drop(_c);
        Ok(StreamCopier {
            inner: Arc::new(Inner {
                device,
                stream,
                owns_stream,
                events: list,
                free: (p, c),
                copies: AtomicU64::new(0),
            }),
            device_locations,
            host_locations,
        })
    }

    /// Copies started so far.
    pub fn copies(&self) -> u64 {
        self.inner.copies.load(Ordering::Acquire)
    }

    fn on_device(&self, l: LocationId) -> bool {
        self.device_locations.contains(&l)
    }
}

/// A copy's completion: its event, handed back to the ring once it fired (or the test is dropped).
struct EventRetired {
    inner: Arc<Inner>,
    index: u32,
    returned: AtomicBool,
}

impl EventRetired {
    fn give_back(&self) {
        if !self.returned.swap(true, Ordering::AcqRel) {
            let _ = self.inner.free.0.offer(self.index);
        }
    }
}

impl Retired for EventRetired {
    fn retired(&self) -> bool {
        if self.returned.load(Ordering::Acquire) {
            return true;
        }
        let Ok(_c) = self.inner.device.enter() else {
            return false;
        };
        // SAFETY: an event of the ring, recorded by this copy and not handed out again until given back.
        let done = unsafe {
            (self.inner.device.driver.fns.event_query)(self.inner.events[self.index as usize])
        };
        if done == SUCCESS {
            self.give_back();
            true
        } else {
            false
        }
    }
}

impl Drop for EventRetired {
    fn drop(&mut self) {
        // Re-recording an event later is safe: it then marks the later copy.
        self.give_back();
    }
}

impl StreamCopier {
    /// Takes an event from the ring for a copy, or refuses when every one is in use.
    fn take_event(&self) -> Result<EventRetired, CopyError> {
        let index = self.inner.free.1.poll().ok_or_else(|| CopyError {
            what: "every completion event is in use".into(),
        })?;
        Ok(EventRetired {
            inner: self.inner.clone(),
            index,
            returned: AtomicBool::new(false),
        })
    }

    /// Queues a copy on the stream through `copy` (which makes the driver call), then records the copy's event.
    ///
    /// # Safety
    /// `copy` calls the driver with ranges that satisfy the contract of [`Copier::start`] or
    /// [`Copier::start_pitched`].
    unsafe fn queue(
        &self,
        what: &'static str,
        copy: impl FnOnce(&super::Fns, CuStream) -> i32,
    ) -> Result<Box<dyn Retired>, CopyError> {
        let fail = |e: CudaError| CopyError {
            what: e.to_string(),
        };
        let test = self.take_event()?;
        let d = &self.inner.device;
        let f = &d.driver.fns;
        let s = self.inner.stream;
        let _c = d.enter().map_err(fail)?;
        d.check(what, copy(f, s)).map_err(fail)?;
        // SAFETY: the ring's event, this copy's alone until given back; the stream is this copier's (or the one the
        // caller lent it).
        d.check("cuEventRecord", unsafe {
            (f.event_record)(self.inner.events[test.index as usize], s)
        })
        .map_err(fail)?;
        self.inner.copies.fetch_add(1, Ordering::AcqRel);
        Ok(Box::new(test))
    }
}

impl Copier for StreamCopier {
    fn joins(&self, from: LocationId, to: LocationId) -> bool {
        let known = |l| self.on_device(l) || self.host_locations.contains(&l);
        known(from) && known(to) && (self.on_device(from) || self.on_device(to))
    }

    unsafe fn start(
        &self,
        from: LocationId,
        src: u64,
        to: LocationId,
        dst: u64,
        bytes: u64,
    ) -> Result<Box<dyn Retired>, CopyError> {
        let n = bytes as usize;
        let (from_device, to_device) = (self.on_device(from), self.on_device(to));
        if !from_device && !to_device {
            return Err(CopyError {
                what: "host to host: not this copier's".into(),
            });
        }
        // SAFETY: the caller's contract (valid, non-overlapping ranges untouched until retired); the context is
        // current inside `queue`; the stream is this copier's.
        unsafe {
            self.queue("cuMemcpyAsync", |f, s| match (from_device, to_device) {
                (true, true) => (f.memcpy_dtod_async)(dst, src, n, s),
                (true, false) => (f.memcpy_dtoh_async)(dst as *mut c_void, src, n, s),
                _ => (f.memcpy_htod_async)(dst, src as *const c_void, n, s),
            })
        }
    }

    unsafe fn start_pitched(
        &self,
        from: LocationId,
        to: LocationId,
        copy: PitchedCopy,
    ) -> Result<Box<dyn Retired>, CopyError> {
        let (from_device, to_device) = (self.on_device(from), self.on_device(to));
        if !from_device && !to_device {
            return Err(CopyError {
                what: "host to host: not this copier's".into(),
            });
        }
        if copy.src_pitch < copy.row_bytes || copy.dst_pitch < copy.row_bytes {
            return Err(CopyError {
                what: "a row pitch is shorter than the row".into(),
            });
        }
        let description = Memcpy2D::pitched(from_device, to_device, &copy);
        // SAFETY: the caller's contract for every row; the description is read by the call, which returns before
        // `description` drops.
        unsafe {
            self.queue("cuMemcpy2DAsync", |f, s| {
                (f.memcpy_2d_async)(&description, s)
            })
        }
    }
}
