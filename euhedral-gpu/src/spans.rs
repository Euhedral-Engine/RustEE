//! Timed spans: the bookkeeping behind [`Stream::span_begin`](crate::Stream::span_begin).
//!
//! A span times a stretch of one stream's work. The owner opens it before the work and closes it after, then polls for the
//! duration from the same stream's source, whenever it enters: the poll reads the end once (a query, never a wait), so a
//! span is read only after its end was reached. A stream keeps a fixed number of spans in flight
//! ([`StreamOptions::spans`](crate::StreamOptions::spans)); an open or unread span holds its slot until it is read or cancelled.
//! Nothing here allocates after the stream is opened: the slots are plain fields of the single owner.

use euhedral_graph::QueueId;

use crate::GpuError;

/// A timed span of a stream's work: opened by `span_begin`, closed by `span_end`, read by `span_poll`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Span {
    stream: QueueId,
    slot: u32,
    generation: u32,
}

impl Span {
    /// The slot of the stream's ring this span holds.
    pub fn slot(&self) -> usize {
        self.slot as usize
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State {
    Free,
    Open,
    Closed,
}

/// The slots of one stream's spans.
#[derive(Debug)]
pub(crate) struct SpanBook {
    stream: QueueId,
    slots: Vec<(State, u32)>,
    /// Where the next search for a free slot starts: spans are mostly read in the order they were opened.
    cursor: usize,
}

impl SpanBook {
    pub(crate) fn new(stream: QueueId, slots: u32) -> Self {
        SpanBook {
            stream,
            slots: vec![(State::Free, 0); slots as usize],
            cursor: 0,
        }
    }

    /// Takes a free slot: `Err(Busy)` when every slot is open or unread, `Err(Invalid)` for a stream with none.
    pub(crate) fn begin(&mut self) -> Result<Span, GpuError> {
        let n = self.slots.len();
        if n == 0 {
            return Err(GpuError::Invalid(
                "this stream keeps no timed spans (StreamOptions::spans is 0)".into(),
            ));
        }
        for step in 0..n {
            let at = (self.cursor + step) % n;
            if self.slots[at].0 == State::Free {
                self.slots[at].0 = State::Open;
                self.cursor = (at + 1) % n;
                return Ok(Span {
                    stream: self.stream,
                    slot: at as u32,
                    generation: self.slots[at].1,
                });
            }
        }
        Err(GpuError::Busy)
    }

    fn check(&self, span: Span, state: State) -> Result<usize, GpuError> {
        let at = span.slot as usize;
        match self.slots.get(at) {
            Some(&(s, generation))
                if span.stream == self.stream && generation == span.generation && s == state =>
            {
                Ok(at)
            }
            _ => Err(GpuError::Invalid(format!(
                "a span that is not {state:?} on this stream"
            ))),
        }
    }

    /// Closes an open span.
    pub(crate) fn end(&mut self, span: Span) -> Result<usize, GpuError> {
        let at = self.check(span, State::Open)?;
        self.slots[at].0 = State::Closed;
        Ok(at)
    }

    /// The slot of a closed span, to be read.
    pub(crate) fn closed(&self, span: Span) -> Result<usize, GpuError> {
        self.check(span, State::Closed)
    }

    /// Frees a span's slot (read, or cancelled); the span's handle is stale from here.
    pub(crate) fn release(&mut self, span: Span) -> Result<(), GpuError> {
        let at = self
            .slots
            .get(span.slot as usize)
            .and(Some(span.slot as usize))
            .ok_or_else(|| GpuError::Invalid("a span that is not on this stream".into()))?;
        if span.stream != self.stream
            || self.slots[at].1 != span.generation
            || self.slots[at].0 == State::Free
        {
            return Err(GpuError::Invalid(
                "a span that is not held on this stream".into(),
            ));
        }
        self.slots[at] = (State::Free, span.generation.wrapping_add(1));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn book(n: u32) -> SpanBook {
        SpanBook::new(QueueId::next(), n)
    }

    #[test]
    fn a_full_ring_is_busy_until_a_span_is_read() {
        let mut b = book(2);
        let (x, y) = (b.begin().unwrap(), b.begin().unwrap());
        assert_eq!(b.begin(), Err(GpuError::Busy));
        b.end(x).unwrap();
        assert_eq!(
            b.begin(),
            Err(GpuError::Busy),
            "closed and unread still holds its slot"
        );
        b.release(x).unwrap();
        assert!(b.begin().is_ok());
        assert_ne!(x.slot, y.slot);
    }

    #[test]
    fn a_stale_or_foreign_span_is_refused() {
        let mut b = book(1);
        let a = b.begin().unwrap();
        b.end(a).unwrap();
        b.release(a).unwrap();
        assert!(b.closed(a).is_err(), "released: stale");
        assert!(b.release(a).is_err());
        let again = b.begin().unwrap();
        assert!(
            b.end(a).is_err(),
            "the old handle names the slot's earlier use"
        );
        assert!(b.end(again).is_ok());
        let mut other = book(1);
        assert!(other.end(again).is_err(), "another stream's");
    }

    #[test]
    fn a_stream_without_spans_says_so() {
        let mut b = book(0);
        assert!(matches!(b.begin(), Err(GpuError::Invalid(_))));
    }
}
