use super::IngestSink;
use super::queue::{DEFAULT_CHUNK, Ending, QueueSource};
use crate::frames::Frame;
use crate::generics::LatticeSource;
use std::num::NonZeroUsize;
use std::sync::Arc;

/// A fixed batch of frames for the lattice, delivered in order. It completes as soon as its last
/// frame is taken.
pub struct ArrayIngestSink {
    source: Arc<QueueSource>,
}

impl ArrayIngestSink {
    pub fn new(frames: impl IntoIterator<Item = Box<dyn Frame>>) -> Self {
        let frames: Vec<_> = frames.into_iter().collect();
        let chunk = frames.len().clamp(1, DEFAULT_CHUNK);
        let source = QueueSource::new(NonZeroUsize::MIN, chunk, Ending::Exhausted);
        for frame in frames {
            if source.offer_to(0, frame).is_err() {
                unreachable!("a new queue takes every frame");
            }
        }
        source.finish();
        Self { source }
    }

    /// Frames not yet taken.
    pub fn remaining(&self) -> u64 {
        self.source.size()
    }
}

impl IngestSink for ArrayIngestSink {
    fn delegate(&self) -> Arc<dyn LatticeSource> {
        self.source.clone()
    }

    /// Detaches from the lattice at once; frames not taken stay in the sink until it is dropped.
    fn complete(&self) {
        self.source.close();
    }

    fn is_complete(&self) -> bool {
        self.source.is_complete()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flow_control::testing::{TestReceiver, frame};

    fn attached(ids: std::ops::Range<u64>) -> (ArrayIngestSink, Arc<TestReceiver>) {
        let sink = ArrayIngestSink::new(ids.map(frame));
        let receiver = TestReceiver::new();
        sink.delegate().add_downstream(receiver.clone());
        (sink, receiver)
    }

    #[test]
    fn requests_deliver_frames_in_order() {
        let (sink, receiver) = attached(0..5);
        let source = sink.delegate();
        assert!(receiver.upstream.lock().unwrap().is_some());
        source.request(2);
        assert_eq!(receiver.received(), [0, 1]);
        assert_eq!(sink.remaining(), 3);
        source.request(2);
        assert_eq!(receiver.received(), [0, 1, 2, 3]);
        assert!(!sink.is_complete());
    }

    #[test]
    fn taking_the_last_frame_completes_the_sink() {
        let (sink, receiver) = attached(0..3);
        sink.delegate().request(10);
        assert_eq!(receiver.received(), [0, 1, 2]);
        assert!(sink.is_complete());
        assert!(receiver.is_completed());

        let (sink, receiver) = attached(0..3);
        let mut pulled = Vec::new();
        assert_eq!(
            sink.delegate()
                .pull(&mut |frame| pulled.push(frame), &mut |_| false, 3),
            3
        );
        assert!(sink.is_complete());
        assert!(receiver.is_completed());
    }

    #[test]
    fn an_empty_sink_completes_on_first_demand() {
        let (sink, receiver) = attached(0..0);
        assert!(!sink.is_complete());
        sink.delegate().request(1);
        assert!(sink.is_complete());
        assert!(receiver.is_completed());
    }

    #[test]
    fn zero_demand_delivers_nothing() {
        let (sink, receiver) = attached(0..3);
        sink.delegate().request(0);
        let pulled = sink.delegate().pull(&mut drop, &mut |_| false, 0);
        assert_eq!(pulled, 0);
        assert!(receiver.received().is_empty());
        assert_eq!(sink.remaining(), 3);
    }

    #[test]
    fn a_pull_stops_before_a_matching_frame() {
        let (sink, _receiver) = attached(0..4);
        let mut pulled = Vec::new();
        let count = sink.delegate().pull(
            &mut |frame| pulled.push(frame.header().id_hash()),
            &mut |frame| frame.header().id_hash() == 2,
            10,
        );
        assert_eq!(count, 2);
        assert_eq!(pulled, [0, 1]);
        assert!(!sink.is_complete());
    }

    #[test]
    fn only_one_downstream_attaches() {
        let (sink, _receiver) = attached(0..1);
        let second = TestReceiver::new();
        sink.delegate().add_downstream(second.clone());
        assert_eq!(second.errors(), ["the source already has a downstream"]);
    }

    #[test]
    fn completing_detaches_at_once() {
        let (sink, receiver) = attached(0..3);
        sink.complete();
        assert!(sink.is_complete());
        assert!(receiver.is_completed());
        sink.delegate().request(10);
        assert!(receiver.received().is_empty());
        assert_eq!(sink.remaining(), 3);
    }

    #[test]
    fn demand_saturates_instead_of_overflowing() {
        let (sink, receiver) = attached(0..3);
        let source = sink.delegate();
        source.request(u64::MAX);
        assert_eq!(receiver.received(), [0, 1, 2]);
        let (sink, receiver) = attached(0..3);
        let source = sink.delegate();
        source.request(1);
        source.request(u64::MAX);
        assert_eq!(receiver.received(), [0, 1, 2]);
    }
}
