//! Retirement through the driver's stream callbacks.
//!
//! The callback runs on a thread the driver owns. It makes no driver call, builds no frame and blocks on nothing: it
//! hands the ticket to the listener, which enqueues. A panic in a listener is caught here and never crosses the
//! boundary into the driver.

use std::ffi::c_void;
use std::panic::{AssertUnwindSafe, catch_unwind};

use euhedral_graph::LaneId;
use euhedral_memory::cuda::{CuResult, CuStream};

use crate::{GpuError, RetireListener, RetireTicket};

use super::stream::CudaStream;

/// What a callback carries: the listener, and the ticket's lane and number.
struct CallbackRecord {
    listener: Box<dyn RetireListener>,
    stream: u64,
    lane: LaneId,
    seq: u64,
}

fn tell(user: *mut c_void, status: CuResult) {
    // SAFETY: `user` is the pointer `arm` made with `Box::into_raw`; the driver calls a callback exactly once, so the
    // record is reclaimed exactly once.
    let record = unsafe { Box::from_raw(user.cast::<CallbackRecord>()) };
    let CallbackRecord {
        listener,
        stream,
        lane,
        seq,
    } = *record;
    let ticket = RetireTicket::new(stream, lane, seq, status);
    let _ = catch_unwind(AssertUnwindSafe(|| listener.retired(ticket)));
}

/// `CUstreamCallback`: the driver passes the status of the work before it.
unsafe extern "C" fn trampoline(_stream: CuStream, status: CuResult, user: *mut c_void) {
    tell(user, status);
}

/// `CUhostFn`: no status is passed; a launch that follows a fault never runs this.
unsafe extern "C" fn host_trampoline(user: *mut c_void) {
    tell(user, 0);
}

impl CudaStream {
    /// Arms `listener` after the work enqueued so far. The caller holds the lock and the context is current.
    pub(crate) fn arm(&self, seq: u64, listener: Box<dyn RetireListener>) -> Result<(), GpuError> {
        let user = Box::into_raw(Box::new(CallbackRecord {
            listener,
            stream: self.id,
            lane: self.lane_id(),
            seq,
        }))
        .cast::<c_void>();
        let fns = &self.gpu.fns;
        // SAFETY: a stream of this device and a record the callback reclaims; the context is current.
        let result = unsafe {
            match fns.stream_add_callback {
                Some(add) => self
                    .gpu
                    .check("cuStreamAddCallback", add(self.raw, trampoline, user, 0)),
                None => self.gpu.check(
                    "cuLaunchHostFunc",
                    (fns.launch_host_func)(self.raw, host_trampoline, user),
                ),
            }
        };
        if result.is_err() {
            // SAFETY: the driver did not take the record, so it will not call back: reclaim it here.
            drop(unsafe { Box::from_raw(user.cast::<CallbackRecord>()) });
        }
        result
    }
}
