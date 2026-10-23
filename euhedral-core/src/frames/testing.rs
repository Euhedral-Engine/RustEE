//! Frames and an executor shared by the frame tests.

use super::{Frame, FrameError, FrameHeader};
use crate::generics::AbstractExecutor;
use crate::impls::DefaultExecutor;
use std::sync::{Arc, Mutex};

/// Drives `frame` through its lifecycle on a [`DefaultExecutor`].
pub(crate) fn run(frame: Box<dyn Frame>) {
    DefaultExecutor::new().run(frame);
}

/// Appends its id hash to a shared log when executed, or fails with `error` if set.
pub(crate) struct Recording {
    header: FrameHeader,
    log: Arc<Mutex<Vec<u64>>>,
    error: Option<fn() -> FrameError>,
}

impl Recording {
    pub(crate) fn new(id_hash: u64, log: &Arc<Mutex<Vec<u64>>>) -> Self {
        Self {
            header: FrameHeader::new(id_hash),
            log: Arc::clone(log),
            error: None,
        }
    }

    pub(crate) fn failing(
        id_hash: u64,
        log: &Arc<Mutex<Vec<u64>>>,
        error: fn() -> FrameError,
    ) -> Self {
        Self {
            error: Some(error),
            ..Self::new(id_hash, log)
        }
    }

    pub(crate) fn boxed(self) -> Box<dyn Frame> {
        Box::new(self)
    }
}

impl Frame for Recording {
    fn header(&self) -> &FrameHeader {
        &self.header
    }

    fn header_mut(&mut self) -> &mut FrameHeader {
        &mut self.header
    }

    fn execute(&mut self) -> Result<(), FrameError> {
        self.log.lock().unwrap().push(self.header.id_hash());
        match self.error {
            Some(error) => Err(error()),
            None => Ok(()),
        }
    }
}
