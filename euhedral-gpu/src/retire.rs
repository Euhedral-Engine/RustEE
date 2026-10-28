//! Retirement into [`Completions`].

use std::any::Any;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use euhedral_core::frames::Frame;
use euhedral_graph::{Completions, LaneId, Pending};

use crate::{Gpu, GpuError, POISONED_STATUS, RetireListener, RetireTicket, Stream};

/// Arms a retirement on `stream` whose continuation `then(ticket)` is emitted by `completions`. The frame is built
/// by the lattice's own poll, never by the driver's callback. If the device is poisoned before the callback runs, the
/// continuation is emitted at once with [`POISONED_STATUS`], so nothing is stranded.
///
/// The continuation settles the ticket with [`Stream::confirm_retired`].
pub fn retire_into<S: Stream>(
    stream: &Arc<S>,
    completions: &Completions,
    then: impl FnOnce(RetireTicket) -> Box<dyn Frame> + Send + 'static,
) -> Result<(), GpuError> {
    let gpu = stream.gpu().clone();
    let (listener, pending) = armed(
        stream.lane(),
        Box::new(move || gpu.poisoned().is_some()),
        Box::new(then),
    );
    stream.notify_retired(listener)?;
    completions.register(pending);
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
        self.0.seq.store(ticket.seq, Ordering::Relaxed);
        self.0.status.store(ticket.status, Ordering::Relaxed);
        self.0.done.store(true, Ordering::Release);
    }
}

type Then = Box<dyn FnOnce(RetireTicket) -> Box<dyn Frame> + Send>;

struct CellPending {
    cell: Arc<RetireCell>,
    lane: LaneId,
    poisoned: Box<dyn Fn() -> bool + Send>,
    then: Option<Then>,
}

impl Pending for CellPending {
    fn poll(&mut self) -> Option<Box<dyn Frame>> {
        let ticket = if self.cell.done.load(Ordering::Acquire) {
            RetireTicket {
                lane: self.lane,
                seq: self.cell.seq.load(Ordering::Relaxed),
                status: self.cell.status.load(Ordering::Relaxed),
            }
        } else if (self.poisoned)() {
            RetireTicket {
                lane: self.lane,
                seq: u64::MAX,
                status: POISONED_STATUS,
            }
        } else {
            return None;
        };
        self.then.take().map(|then| then(ticket))
    }
}

/// A listener and the pending that turns its retirement into `then`'s frame. `poisoned` says whether the device
/// faulted.
fn armed(
    lane: LaneId,
    poisoned: Box<dyn Fn() -> bool + Send>,
    then: Then,
) -> (Box<dyn RetireListener>, Box<dyn Pending>) {
    let cell = Arc::new(RetireCell::default());
    (
        Box::new(CellListener(cell.clone())),
        Box::new(CellPending {
            cell,
            lane,
            poisoned,
            then: Some(then),
        }),
    )
}

#[cfg_attr(
    all(),
    allow(dead_code, reason = "used by the backends, which come next")
)]
/// Values held until a retirement is confirmed: `(retirement seq, value)`, released in order.
#[derive(Default)]
pub(crate) struct Kept {
    values: Mutex<VecDeque<(u64, Box<dyn Any + Send>)>>,
}

impl std::fmt::Debug for Kept {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Kept").finish_non_exhaustive()
    }
}

#[allow(dead_code, reason = "used by the backends, which come next")]
impl Kept {
    pub(crate) fn push(&self, seq: u64, value: Box<dyn Any + Send>) {
        self.values
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push_back((seq, value));
    }

    /// Drops every value kept for a retirement numbered `seq` or lower.
    pub(crate) fn release_through(&self, seq: u64) {
        let mut released = Vec::new();
        {
            let mut values = self.values.lock().unwrap_or_else(PoisonError::into_inner);
            while values.front().is_some_and(|(s, _)| *s <= seq) {
                released.extend(values.pop_front());
            }
        }
        drop(released);
    }

    /// Drops everything kept.
    pub(crate) fn clear(&self) {
        let all = std::mem::take(&mut *self.values.lock().unwrap_or_else(PoisonError::into_inner));
        drop(all);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use euhedral_core::frames::RunnableFrame;

    #[test]
    fn kept_values_release_in_order_through_a_seq() {
        let kept = Kept::default();
        let (a, b, c) = (Arc::new(()), Arc::new(()), Arc::new(()));
        kept.push(1, Box::new(a.clone()));
        kept.push(1, Box::new(b.clone()));
        kept.push(2, Box::new(c.clone()));
        kept.release_through(1);
        assert_eq!(
            (
                Arc::strong_count(&a),
                Arc::strong_count(&b),
                Arc::strong_count(&c)
            ),
            (1, 1, 2)
        );
        kept.clear();
        assert_eq!(Arc::strong_count(&c), 1);
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
        let seen = Arc::new(Mutex::new(Vec::new()));
        let (listener, mut pending) = armed(LaneId(2), Box::new(|| false), {
            let seen = seen.clone();
            Box::new(move |t| frame_recording(&seen, t))
        });
        assert!(pending.poll().is_none(), "not retired yet");
        listener.retired(RetireTicket {
            lane: LaneId(2),
            seq: 7,
            status: 0,
        });
        assert!(pending.poll().is_some());
        let seen = seen.lock().unwrap();
        assert_eq!(seen.len(), 1);
        assert_eq!((seen[0].seq, seen[0].lane), (7, LaneId(2)));
        assert!(seen[0].ok());
    }

    #[test]
    fn a_poisoned_device_emits_the_continuation_without_the_listener() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let (_listener, mut pending) = armed(LaneId(0), Box::new(|| true), {
            let seen = seen.clone();
            Box::new(move |t| frame_recording(&seen, t))
        });
        assert!(pending.poll().is_some());
        assert!(pending.poll().is_none(), "emitted once");
        let seen = seen.lock().unwrap();
        assert_eq!(seen[0].status, POISONED_STATUS);
        assert!(!seen[0].ok());
    }
}
