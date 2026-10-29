//! Shared fixtures of the lane-source tests: a chain item generic over the backend, a recording receiver, and a pump
//! that requests every source.

#![allow(dead_code, reason = "each test binary uses part of the fixtures")]
#![allow(
    clippy::disallowed_methods,
    clippy::disallowed_types,
    reason = "test threads stand in for workers"
)]

use std::convert::Infallible;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use euhedral_core::frames::{Failure, Frame, FrameHeader};
use euhedral_core::generics::{LatticeReceiver, LatticeSource, LatticeTerminal};
use euhedral_gpu::{Args, Enqueue, Gpu, GpuError, Launch, Stream};
use euhedral_graph::{Deliver, LaneCx, LaneSource, LaneSubmit, LaneWork, Marker, Outcome};

pub fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

pub type Kern<S> = <<S as Stream>::Gpu as Gpu>::Kernel;
/// The launches of one kernel item: kernels with their arguments.
pub type Launches<S> = Vec<(Kern<S>, Args)>;

/// What the items of a test record. Items are named by `(tag, step)`: the tag of the chain, the step in it.
#[derive(Default)]
pub struct Probe {
    /// `(tag, step)` of the items that ran, in order.
    pub ran: Mutex<Vec<(u64, u32)>>,
    pub retired: Mutex<Vec<(u64, u32, Option<GpuError>)>>,
    /// Items the lane ended with `failed`: a submission the device refused, or an await or mark it refused.
    pub failed: Mutex<Vec<(u64, u32, GpuError)>>,
    pub bridged: Mutex<Vec<(u64, Marker)>>,
    pub dropped: AtomicUsize,
}

impl Probe {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    pub fn ran(&self) -> Vec<(u64, u32)> {
        lock(&self.ran).clone()
    }
}

/// A host word the device copies into: a pointer to 4 bytes, `Copy`, valid as its constructor says.
#[derive(Clone, Copy, Debug)]
pub struct HostWord(*mut u32);

// SAFETY: the word is plain memory that the device writes and the host reads after the retirement; the constructor's
// contract keeps it alive.
unsafe impl Send for HostWord {}
// SAFETY: as above.
unsafe impl Sync for HostWord {}

impl HostWord {
    /// A word that is never freed: 4 bytes per call.
    pub fn leaked() -> Self {
        Self(Box::leak(Box::new(AtomicU32::new(0))).as_ptr())
    }

    /// # Safety
    /// `ptr` is aligned and valid for reads and writes of 4 bytes (pinned host memory for a CUDA copy) until every item
    /// that holds the word is gone.
    pub unsafe fn from_raw(ptr: *mut u32) -> Self {
        Self(ptr)
    }

    pub fn bytes(&self) -> *mut u8 {
        self.0.cast()
    }

    /// The word, after the retirement that covers the copy.
    pub fn load(&self) -> u32 {
        // SAFETY: valid for reads by the constructor's contract.
        unsafe { self.0.read_volatile() }
    }
}

/// A chain: `steps` kernel items, each delivering the next behind its marker, then one item that copies the device
/// word to the host and asks for a retirement.
pub struct Plan<S: Stream> {
    pub steps: u32,
    /// Where item `step` is delivered.
    pub route: Box<dyn Fn(u32) -> Deliver + Send + Sync>,
    /// The launches of kernel item `step`.
    pub launches: Box<dyn Fn(u32) -> Launches<S> + Send + Sync>,
    /// The device word the last item copies, and the host word it copies to.
    pub device_word: u64,
    pub host_word: HostWord,
    pub probe: Arc<Probe>,
    /// Every kernel item asks for a retirement too, not only the last item.
    pub retire_steps: bool,
    pub _stream: PhantomData<fn() -> S>,
}

pub struct Item<S: Stream> {
    pub tag: u64,
    pub step: u32,
    pub plan: Arc<Plan<S>>,
    pub waits: Vec<Marker>,
}

impl<S: Stream> Item<S> {
    /// Step 0 of a chain.
    pub fn first(plan: &Arc<Plan<S>>) -> Self {
        Self {
            tag: 0,
            step: 0,
            plan: plan.clone(),
            waits: Vec::new(),
        }
    }

    fn last(&self) -> bool {
        self.step == self.plan.steps
    }
}

impl<S: Stream> Drop for Item<S> {
    fn drop(&mut self) {
        self.plan.probe.dropped.fetch_add(1, Ordering::SeqCst);
    }
}

impl<S> LaneWork<S, ()> for Item<S>
where
    S: Stream + LaneSubmit<Error = GpuError>,
{
    const KINDS: &'static [&'static str] = &["step", "read"];

    fn kind(&self) -> usize {
        usize::from(self.last())
    }

    fn waits(&self) -> &[Marker] {
        &self.waits
    }

    fn run(&mut self, cx: &mut LaneCx<'_, S, (), Self>) -> Outcome<Infallible, GpuError> {
        let probe = self.plan.probe.clone();
        lock(&probe.ran).push((self.tag, self.step));
        for &marker in &self.waits {
            if cx.bridges(marker) {
                lock(&probe.bridged).push((self.tag, marker));
            }
        }
        let plan = self.plan.clone();
        let step = self.step;
        let last = self.last();
        let result = cx.submitter().submit(|q| {
            if last {
                // SAFETY: the host word lives in the plan, which this item keeps until the retirement arrives; the
                // device word is a live allocation that only the chain touches.
                unsafe { q.copy_to_host(plan.host_word.bytes(), plan.device_word, 4) }
            } else {
                for (kernel, args) in (plan.launches)(step) {
                    // SAFETY: the arguments match the kernel; the device word is a live allocation that only the
                    // chain touches.
                    unsafe { q.launch(&kernel, Launch::linear(1, 1), &args)? };
                }
                Ok(())
            }
        });
        match result {
            Ok(()) => Outcome::Submitted {
                retire: last || plan.retire_steps,
            },
            Err(error) => Outcome::Failed(error),
        }
    }

    fn then(&mut self, marker: Option<Marker>, cx: &mut LaneCx<'_, S, (), Self>) {
        if self.last() {
            return;
        }
        let next = Item {
            tag: self.tag,
            step: self.step + 1,
            plan: self.plan.clone(),
            waits: marker.into_iter().collect(),
        };
        let to = (self.plan.route)(next.step);
        cx.deliver(next, to)
            .unwrap_or_else(|_| panic!("a lane is closed"));
    }

    fn retired(self, _marker: Marker, failure: Option<GpuError>, cx: &mut LaneCx<'_, S, (), Self>) {
        lock(&self.plan.probe.retired).push((self.tag, self.step, failure));
        cx.ready(Value::frame(u64::from(self.plan.host_word.load())));
    }

    fn failed(self, error: GpuError, _cx: &mut LaneCx<'_, S, (), Self>) {
        lock(&self.plan.probe.failed).push((self.tag, self.step, error));
    }
}

/// A frame that carries a value in its id hash.
pub struct Value {
    header: FrameHeader,
}

impl Value {
    pub fn frame(value: u64) -> Box<dyn Frame> {
        Box::new(Self {
            header: FrameHeader::new(value),
        })
    }
}

impl Frame for Value {
    fn header(&self) -> &FrameHeader {
        &self.header
    }

    fn header_mut(&mut self) -> &mut FrameHeader {
        &mut self.header
    }
}

/// Records the values of the frames it is pushed.
#[derive(Default)]
pub struct Collect {
    pub received: Mutex<Vec<u64>>,
}

impl Collect {
    pub fn received(&self) -> Vec<u64> {
        lock(&self.received).clone()
    }
}

impl LatticeTerminal for Collect {
    fn add_upstream(&self, _upstream: Arc<dyn LatticeSource>) {}
}

impl LatticeReceiver for Collect {
    fn push(&self, frame: Box<dyn Frame>) {
        lock(&self.received).push(frame.header().id_hash());
    }

    fn on_complete(&self) {}

    fn on_error(&self, _error: Failure) {}
}

/// One recording receiver behind every source.
pub fn collect_all<S, W>(sources: &[Arc<LaneSource<S, (), W>>]) -> Arc<Collect>
where
    S: LaneSubmit,
    W: LaneWork<S, ()>,
{
    let collect = Arc::new(Collect::default());
    for source in sources {
        source.clone().add_downstream(collect.clone());
    }
    collect
}

/// One round of requests: what a worker pool does when it visits every source.
pub fn pump<S, W>(sources: &[Arc<LaneSource<S, (), W>>])
where
    S: LaneSubmit,
    W: LaneWork<S, ()>,
{
    for source in sources {
        source.request(u64::MAX);
    }
}

/// One round that starts at source `start` and goes round: the sources visit the shared intake in turn, so each of a
/// group's lanes takes group work.
pub fn pump_from<S, W>(sources: &[Arc<LaneSource<S, (), W>>], start: usize)
where
    S: LaneSubmit,
    W: LaneWork<S, ()>,
{
    for i in 0..sources.len() {
        sources[(start + i) % sources.len()].request(u64::MAX);
    }
}

#[cfg(feature = "cuda")]
pub mod cuda_support;
