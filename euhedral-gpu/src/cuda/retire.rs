//! Retirement through the driver's stream callback (`cuStreamAddCallback`), which reports the status of the work
//! before it exactly once, even after a device fault. The entry point is required when the backend loads.
//!
//! The callback runs on a thread the driver owns. It reads a pointer to the stream's [`RetireSink`] (its user data,
//! stable for the stream's life), pushes the status of the work before it onto the stream's lock-free retirement
//! queue, and returns. It makes no driver call, allocates nothing in steady state, frees nothing, takes no lock, and
//! touches no listener: the stream's owner (a lattice source, in its `request` or `pull`) drains the queue
//! (`Stream::poll_retired`) and dispatches.

use std::ffi::c_void;
use std::panic::{AssertUnwindSafe, catch_unwind};

use std::sync::Arc;

use euhedral_data_structures::sync::HotScope;
use euhedral_graph::RetireCell;
use euhedral_memory::cuda::{CuResult, CuStream};

use crate::GpuError;
use crate::retire::RetireSink;

use super::stream::{CudaQueue, CudaStream};

fn tell(user: *mut c_void, status: CuResult) {
    let _hot = HotScope::enter();
    // SAFETY: `user` is the pointer `arm` made from the stream's `RetireSink`, which the stream's shared queue side
    // owns; the stream's drop waits for every callback before that side can go.
    let sink = unsafe { &*user.cast::<RetireSink>() };
    // A panic must not cross into the driver; a push cannot normally panic.
    let _ = catch_unwind(AssertUnwindSafe(|| sink.push(status)));
}

/// The callback of a retirement that names its cell: it stores the status there and does nothing else.
fn tell_cell(user: *mut c_void, status: CuResult) {
    let _hot = HotScope::enter();
    // SAFETY: `user` is the pointer `arm_cell` made with `Arc::into_raw`, which this callback takes back: the reference
    // count it held keeps the cell alive until here, however the lane has gone since.
    let cell = unsafe { Arc::from_raw(user.cast::<RetireCell>()) };
    // A panic must not cross into the driver; a store cannot normally panic.
    let _ = catch_unwind(AssertUnwindSafe(|| cell.arrive(status)));
    // The lane that armed the retirement keeps its own reference for as long as it uses the cell; the last drop of a
    // cell nobody holds any more is the only free this callback can make.
    drop(cell);
}

/// `CUstreamCallback` of [`CudaQueue::arm_cell`].
unsafe extern "system" fn trampoline_cell(_stream: CuStream, status: CuResult, user: *mut c_void) {
    tell_cell(user, status);
}

/// `CUstreamCallback`: the driver passes the status of the work before it.
unsafe extern "system" fn trampoline(_stream: CuStream, status: CuResult, user: *mut c_void) {
    tell(user, status);
}

impl CudaStream {
    /// Enqueues a callback after the work enqueued so far. The context is current. Nothing is allocated: the user data
    /// is the stream's own sink.
    pub(crate) fn arm_callback(&self) -> Result<(), GpuError> {
        let user = std::ptr::from_ref(self.queue.sink())
            .cast_mut()
            .cast::<c_void>();
        let fns = &self.gpu.fns;
        // SAFETY: a stream of this device and the sink the callback reads; the context is current.
        self.gpu.check("cuStreamAddCallback", unsafe {
            (fns.stream_add_callback)(self.raw, trampoline, user, 0)
        })
    }
}

impl CudaQueue {
    /// Enqueues a callback after the work enqueued so far that stores the work's status in `cell`. The context is
    /// current. Nothing is allocated: the user data is a clone of the `Arc`.
    pub(crate) fn arm_cell(&self, cell: &Arc<RetireCell>) -> Result<(), GpuError> {
        let user = Arc::into_raw(cell.clone()).cast_mut().cast::<c_void>();
        let fns = &self.gpu.fns;
        // SAFETY: a stream of this device; the user data is the reference the callback takes back; the context is
        // current.
        let armed = self.gpu.check("cuStreamAddCallback", unsafe {
            (fns.stream_add_callback)(self.raw, trampoline_cell, user, 0)
        });
        if armed.is_err() {
            // SAFETY: the driver refused the callback, so nothing else holds the reference made above.
            drop(unsafe { Arc::from_raw(user.cast::<RetireCell>()) });
        }
        armed
    }
}
