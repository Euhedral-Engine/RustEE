//! Retirement into [`Completions`].

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU64, Ordering};

use euhedral_core::frames::{Frame, FrameError, FrameHeader};
use euhedral_graph::{Completions, LaneId, Pending, QueueId};

use crate::{Gpu, GpuError, POISONED_STATUS, RetireListener, RetireTicket, Stream};

/// A frame that runs a closure once and owns whatever the closure moved in: the frame of a chain that carries a
/// submitter (and a graph) from one continuation to the next.
///
/// [`RunnableFrame`](euhedral_core::frames::RunnableFrame) runs a shared `Fn`, which cannot give away what it holds;
/// this runs an `FnOnce` through `&mut self`, so the closure may own a submitter and hand it on, with no lock or
/// atomic to take it out of. A frame dropped without running drops the closure, and what it owns, where it is
/// dropped.
pub struct OnceFrame {
    header: FrameHeader,
    run: Option<Box<dyn FnOnce() + Send>>,
}

impl OnceFrame {
    /// An ordered frame of lane `id_hash` that runs `run` once.
    pub fn new(id_hash: u64, run: impl FnOnce() + Send + 'static) -> Self {
        Self {
            header: FrameHeader::new(id_hash),
            run: Some(Box::new(run)),
        }
    }
}

impl std::fmt::Debug for OnceFrame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OnceFrame")
            .field("pending", &self.run.is_some())
            .finish()
    }
}

impl Frame for OnceFrame {
    fn header(&self) -> &FrameHeader {
        &self.header
    }

    fn header_mut(&mut self) -> &mut FrameHeader {
        &mut self.header
    }

    fn execute(&mut self) -> Result<(), FrameError> {
        if let Some(run) = self.run.take() {
            run();
        }
        Ok(())
    }
}

/// A stream handed back with the error that stopped [`retire_into`] from arming its retirement.
#[derive(Debug)]
pub struct RetireError<S> {
    pub stream: S,
    pub error: GpuError,
}

/// Arms a retirement on `stream` whose continuation is a frame emitted by `completions`. The frame is built by the
/// lattice's own poll, never by the driver's callback.
///
/// The stream **travels with its continuation**: this takes the submitter by value, and `then(stream, ticket)` gets it
/// back together with the ticket. A submitter has one owner, so the frame that continues the work is the next owner,
/// and nothing is shared or locked between the frames of a chain. `then` settles the ticket with
/// [`Stream::confirm_retired`], submits more, and arms the next retirement with the stream it was given. If the arming
/// fails the stream comes back in the error.
///
/// If the device is poisoned before the callback runs, the continuation is emitted at once with
/// [`POISONED_STATUS`], so nothing is stranded. If `completions` is dropped or completed with the continuation still
/// pending, the stream is dropped there (a shutdown call: it waits for the device).
pub fn retire_into<S: Stream>(
    mut stream: S,
    completions: &Completions,
    then: impl FnOnce(S, RetireTicket) -> Box<dyn Frame> + Send + 'static,
) -> Result<(), RetireError<S>> {
    let (listener, cell) = new_cell();
    if let Err(error) = stream.notify_retired(listener) {
        return Err(RetireError { stream, error });
    }
    let gpu = stream.gpu().clone();
    completions.register(Box::new(CellPending {
        cell,
        id: stream.id(),
        lane: stream.lane(),
        stream: Some(stream),
        poisoned: Box::new(move || gpu.poisoned().is_some()),
        then: Some(Box::new(then)),
    }));
    Ok(())
}

/// What the listener stores and the pending polls.
#[derive(Debug, Default)]
struct RetireCell {
    seq: AtomicU64,
    status: AtomicI32,
    done: AtomicBool,
}

struct CellListener(Arc<RetireCell>);

impl RetireListener for CellListener {
    fn retired(self: Box<Self>, ticket: RetireTicket) {
        self.0.seq.store(ticket.seq(), Ordering::Relaxed);
        self.0.status.store(ticket.status(), Ordering::Relaxed);
        self.0.done.store(true, Ordering::Release);
    }
}

type Then<S> = Box<dyn FnOnce(S, RetireTicket) -> Box<dyn Frame> + Send>;

struct CellPending<S> {
    cell: Arc<RetireCell>,
    id: QueueId,
    lane: LaneId,
    stream: Option<S>,
    poisoned: Box<dyn Fn() -> bool + Send>,
    then: Option<Then<S>>,
}

impl<S: Send> Pending for CellPending<S> {
    fn poll(&mut self) -> Option<Box<dyn Frame>> {
        let ticket = if self.cell.done.load(Ordering::Acquire) {
            RetireTicket::new(
                self.id,
                self.lane,
                self.cell.seq.load(Ordering::Relaxed),
                self.cell.status.load(Ordering::Relaxed),
            )
        } else if (self.poisoned)() {
            RetireTicket::new(self.id, self.lane, u64::MAX, POISONED_STATUS)
        } else {
            return None;
        };
        let (stream, then) = (self.stream.take()?, self.then.take()?);
        Some(then(stream, ticket))
    }
}

/// A listener and the cell its retirement lands in.
fn new_cell() -> (Box<dyn RetireListener>, Arc<RetireCell>) {
    let cell = Arc::new(RetireCell::default());
    (Box::new(CellListener(cell.clone())), cell)
}

#[cfg(test)]
mod tests {
    use super::*;
    use euhedral_core::frames::RunnableFrame;
    use std::sync::Mutex;

    /// A listener and a pending over a stand-in stream (the unit `()`), as `retire_into` builds them.
    fn armed(
        id: QueueId,
        lane: LaneId,
        poisoned: Box<dyn Fn() -> bool + Send>,
        then: Then<()>,
    ) -> (Box<dyn RetireListener>, Box<dyn Pending>) {
        let (listener, cell) = new_cell();
        let pending = CellPending {
            cell,
            id,
            lane,
            stream: Some(()),
            poisoned,
            then: Some(then),
        };
        (listener, Box::new(pending))
    }

    fn frame_recording(
        seen: &Arc<Mutex<Vec<RetireTicket>>>,
        ticket: RetireTicket,
    ) -> Box<dyn Frame> {
        let seen = seen.clone();
        seen.lock().unwrap().push(ticket);
        Box::new(RunnableFrame::new(1, Arc::new(|| {})))
    }

    #[test]
    fn retire_cell_ticket_round_trips() {
        let id = QueueId::next();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let (listener, mut pending) = armed(id, LaneId(2), Box::new(|| false), {
            let seen = seen.clone();
            Box::new(move |(), t| frame_recording(&seen, t))
        });
        assert!(pending.poll().is_none(), "not retired yet");
        listener.retired(RetireTicket::new(id, LaneId(2), 7, 0));
        assert!(pending.poll().is_some());
        let seen = seen.lock().unwrap();
        assert_eq!(seen.len(), 1);
        assert_eq!(
            (seen[0].seq(), seen[0].lane(), seen[0].stream()),
            (7, LaneId(2), id)
        );
        assert!(seen[0].ok());
    }

    #[test]
    fn a_poisoned_device_emits_the_continuation_without_the_listener() {
        let id = QueueId::next();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let (_listener, mut pending) = armed(id, LaneId(0), Box::new(|| true), {
            let seen = seen.clone();
            Box::new(move |(), t| frame_recording(&seen, t))
        });
        assert!(pending.poll().is_some());
        assert!(pending.poll().is_none(), "emitted once");
        let seen = seen.lock().unwrap();
        assert_eq!(seen[0].status(), POISONED_STATUS);
        assert!(!seen[0].ok());
    }
}
