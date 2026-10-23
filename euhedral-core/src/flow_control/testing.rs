//! Frames and receivers shared by the flow-control tests.

use crate::frames::{Failure, Frame, FrameError, FrameHeader};
use crate::generics::{LatticeReceiver, LatticeSource, LatticeTerminal};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

/// How a [`Tracked`] frame was finished.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Fate {
    Finished,
    Failed(String),
}

/// Where [`Tracked`] frames report how they were finished.
pub(crate) type Fates = Arc<Mutex<Vec<(u64, Fate)>>>;

/// A frame that reports to `fates` how it was finished.
pub(crate) struct Tracked {
    header: FrameHeader,
    fates: Fates,
}

impl Tracked {
    pub(crate) fn new(id_hash: u64, fates: &Fates) -> Box<Self> {
        Box::new(Self {
            header: FrameHeader::new(id_hash),
            fates: fates.clone(),
        })
    }
}

impl Frame for Tracked {
    fn header(&self) -> &FrameHeader {
        &self.header
    }

    fn header_mut(&mut self) -> &mut FrameHeader {
        &mut self.header
    }

    fn do_finally(self: Box<Self>) {
        let id = self.header.id_hash();
        self.fates.lock().unwrap().push((id, Fate::Finished));
    }

    fn do_finally_with_error(self: Box<Self>, error: FrameError) {
        let id = self.header.id_hash();
        let fate = Fate::Failed(error.to_string());
        self.fates.lock().unwrap().push((id, fate));
    }
}

/// A frame that is only an id.
pub(crate) fn frame(id_hash: u64) -> Box<dyn Frame> {
    Box::new(crate::frames::BenchmarkFrame::new(id_hash, None))
}

/// Records what reaches it.
#[derive(Default)]
pub(crate) struct TestReceiver {
    pub(crate) received: Mutex<Vec<u64>>,
    pub(crate) errors: Mutex<Vec<String>>,
    pub(crate) completed: AtomicBool,
    pub(crate) upstream: Mutex<Option<Arc<dyn LatticeSource>>>,
}

impl TestReceiver {
    pub(crate) fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    pub(crate) fn received(&self) -> Vec<u64> {
        self.received.lock().unwrap().clone()
    }

    pub(crate) fn errors(&self) -> Vec<String> {
        self.errors.lock().unwrap().clone()
    }

    pub(crate) fn is_completed(&self) -> bool {
        self.completed.load(Ordering::Acquire)
    }
}

impl LatticeTerminal for TestReceiver {
    fn add_upstream(&self, upstream: Arc<dyn LatticeSource>) {
        *self.upstream.lock().unwrap() = Some(upstream);
    }
}

impl LatticeReceiver for TestReceiver {
    fn push(&self, frame: Box<dyn Frame>) {
        self.received.lock().unwrap().push(frame.header().id_hash());
    }

    fn on_complete(&self) {
        self.completed.store(true, Ordering::Release);
    }

    fn on_error(&self, error: Failure) {
        self.errors.lock().unwrap().push(error.to_string());
    }
}
