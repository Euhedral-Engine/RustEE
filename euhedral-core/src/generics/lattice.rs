use crate::frames::{Failure, Frame};
use std::sync::Arc;

/// The bottom of a chain of lattices: something that attaches to a source.
pub trait LatticeTerminal: Send + Sync {
    /// Called when `upstream` attaches this terminal.
    fn add_upstream(&self, upstream: Arc<dyn LatticeSource>);
}

/// Where frames flow to.
pub trait LatticeReceiver: LatticeTerminal {
    /// Takes ownership of a frame the source pushes.
    fn push(&self, frame: Box<dyn Frame>);

    /// The source has finished and will push nothing more.
    fn on_complete(&self);

    /// The source failed.
    fn on_error(&self, error: Failure);

    /// Runs `action` when no `pull` or `request` of the source this receiver serves is in progress,
    /// without waiting: at once if none is, otherwise on the worker that holds the source, after
    /// that worker has let it go. Actions run one at a time, in the order they were handed over,
    /// each exactly once.
    ///
    /// This is the lattice's own hand-off, for state that belongs to the source's attachment (the
    /// interceptor releasing the source it holds). A source's lifecycle calls do not use it: they set
    /// atomic state, and the source applies that state in its next `request` or `pull`. The default
    /// runs `action` at once: a receiver that is no lattice has the caller as the source's only
    /// driver.
    fn serialize(&self, action: Box<dyn FnOnce() + Send>) {
        action();
    }
}

/// Where frames come from.
///
/// **`request` and `pull` do no heavy or high-frequency work.** They run at every visit of a worker. They assemble
/// frames, hand them out and push them, and a small continuation may run there: pop a ready frame, read a counter that
/// says whether work is due. Anything heavy (looking at a device or the operating system, finishing copies, a lock and
/// a system call, detaching sinks), or that would recur at the rate of the visits (querying every operation that waits),
/// is a frame's work, because a worker can only measure how long work takes when the frame comes back through its
/// executor, and that is how it sizes its batches and decides to idle or take part. A source with such work owns a *key frame* ([`KeySlot`](super::KeySlot)):
/// `request` pushes it to the downstream when the work is due and it is home, a worker runs its `execute`, and
/// its `recycle` sends it home. **No `push` is reachable from `pull`**: a `pull` runs on a worker that has entered
/// from a pull, and the downstream's push is for a `request`. A `pull` does not hand the key to its `consumer` either:
/// the pulling worker would run it through its executor while the source's handle is held, and the work behind a key is
/// long, so nothing else could take ready work from the source meanwhile. So a key leaves only through `request`'s push.
/// The key is a control frame, not part of the demand. A key that polls blind is
/// paced ([`KeyPacing`](super::KeyPacing)): it is not pushed at every visit while nothing ends. A worker takes a
/// handle as productive only if its last service issued frames: a request that issued none (the key was not due) shows
/// the handle empty, as a pull that finds none does, and a request that pushed the key shows it productive for that one
/// service, so a worker that sees only keys at their pacing parks between them.
///
/// **Only the lattice's workers call `request` and `pull`.** Work that comes from outside (a submitter, a completion
/// poller, another runtime) is offered into a queue the source owns, an MPSC the source's `pull` and `request` take
/// from on the workers' own visits, and the outside thread returns at once: it never calls `request` or `pull` to get
/// its work admitted. A call from outside is not served by anything that is waiting for it: it can find the source
/// held by a worker's `pull`, which pushes nothing and serves nothing, and then waits for the lattice's next `request`
/// of that source. (A test or a caller with no lattice may stand in for the worker, and drive a source by hand.)
pub trait LatticeSource: Send + Sync {
    /// Attaches the receiver this source pushes to. A source that tells the receiver about itself
    /// does so through [`LatticeTerminal::add_upstream`], passing this `Arc`.
    fn add_downstream(self: Arc<Self>, downstream: Arc<dyn LatticeReceiver>);

    /// Hands up to `demand` frames straight to `consumer`, without pushing, and reports how many.
    ///
    /// The demand is not accumulated like [`request`](Self::request)'s, and no frame is made to meet
    /// it. The pull stops before the first frame for which `stop` is true, leaving that frame in the
    /// source.
    fn pull(
        &self,
        consumer: &mut dyn FnMut(Box<dyn Frame>),
        stop: &mut dyn FnMut(&dyn Frame) -> bool,
        demand: u64,
    ) -> u64;

    /// Signals demand for `demand` more frames; `u64::MAX` means unbounded and `0` asks for nothing.
    /// The source may push during this call or later from another thread. Each push is a separate
    /// admission that uses whatever routing is live when that push begins: returning from `request`
    /// reserves nothing for frames pushed later. A source that pushes only later, from another thread, shows no issued
    /// frame at any request, so a worker reads its handle as empty each time (its frames still run from the caches;
    /// only the worker's participation changes).
    ///
    /// Only a worker calls it (see the trait's documentation). A `request` made while a `pull` of the same source holds
    /// it (from the pull's consumer, or from a caller that is not a worker) banks its demand and is not served by that pull: a pull pushes nothing, not even as it lets go. The source
    /// serves it at the lattice's next `request` of that source, which comes when a worker's whole cycle processed
    /// nothing (its cache included), and `request` returns nothing, so a caller cannot tell it was refused. Frames
    /// the source already holds are handed out by the pulls meanwhile.
    fn request(&self, demand: u64);

    /// Finishes the source.
    fn complete(&self);

    fn is_complete(&self) -> bool;
}

/// A stage that frames flow through, between a source and a receiver.
pub trait LatticeInterceptor: LatticeSource + LatticeReceiver {
    /// Called when `upstream` is chained in front of this interceptor. The default ignores it.
    fn add_upstream_interceptor(&self, upstream: Arc<dyn LatticeInterceptor>) {
        let _ = upstream;
    }

    /// Chains `downstream` behind this interceptor. The default ignores it.
    fn add_downstream_interceptor(self: Arc<Self>, downstream: Arc<dyn LatticeInterceptor>) {
        let _ = downstream;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frames::{BenchmarkFrame, FrameError};
    use std::collections::VecDeque;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

    /// Pushes queued frames as demand arrives, and serves pulls from the same queue.
    #[derive(Default)]
    struct QueueSource {
        frames: Mutex<VecDeque<Box<dyn Frame>>>,
        downstream: Mutex<Option<Arc<dyn LatticeReceiver>>>,
        demand: AtomicU64,
        complete: AtomicBool,
    }

    impl QueueSource {
        fn with(ids: &[u64]) -> Arc<Self> {
            let frames = ids
                .iter()
                .map(|&id| Box::new(BenchmarkFrame::new(id, None)) as Box<dyn Frame>)
                .collect();
            Arc::new(Self {
                frames: Mutex::new(frames),
                ..Self::default()
            })
        }

        fn deliver(&self) {
            let Some(downstream) = self.downstream.lock().unwrap().clone() else {
                return;
            };
            while self.demand.load(Ordering::Acquire) > 0 {
                let Some(frame) = self.frames.lock().unwrap().pop_front() else {
                    return;
                };
                self.demand.fetch_sub(1, Ordering::AcqRel);
                downstream.push(frame);
            }
        }
    }

    impl LatticeSource for QueueSource {
        fn add_downstream(self: Arc<Self>, downstream: Arc<dyn LatticeReceiver>) {
            *self.downstream.lock().unwrap() = Some(downstream.clone());
            downstream.add_upstream(self);
        }

        fn pull(
            &self,
            consumer: &mut dyn FnMut(Box<dyn Frame>),
            stop: &mut dyn FnMut(&dyn Frame) -> bool,
            demand: u64,
        ) -> u64 {
            let mut frames = self.frames.lock().unwrap();
            let mut pulled = 0;
            while pulled < demand {
                match frames.front() {
                    Some(frame) if !stop(&**frame) => {
                        consumer(frames.pop_front().unwrap());
                        pulled += 1;
                    }
                    _ => break,
                }
            }
            pulled
        }

        fn request(&self, demand: u64) {
            if self.is_complete() || demand == 0 {
                return;
            }
            self.demand
                .try_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                    Some(current.saturating_add(demand))
                })
                .unwrap();
            self.deliver();
        }

        fn complete(&self) {
            if !self.complete.swap(true, Ordering::AcqRel)
                && let Some(downstream) = self.downstream.lock().unwrap().take()
            {
                downstream.on_complete();
            }
        }

        fn is_complete(&self) -> bool {
            self.complete.load(Ordering::Acquire)
        }
    }

    /// Records what it receives, and requests `initial` frames when attached.
    #[derive(Default)]
    struct Recorder {
        initial: u64,
        received: Mutex<Vec<u64>>,
        errors: Mutex<Vec<String>>,
        completed: AtomicBool,
        upstream: Mutex<Option<Arc<dyn LatticeSource>>>,
    }

    impl LatticeTerminal for Recorder {
        fn add_upstream(&self, upstream: Arc<dyn LatticeSource>) {
            *self.upstream.lock().unwrap() = Some(upstream.clone());
            upstream.request(self.initial);
        }
    }

    impl LatticeReceiver for Recorder {
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

    /// Forwards frames from its upstream to its downstream, doubling their id hash.
    #[derive(Default)]
    struct Doubler {
        downstream: Mutex<Option<Arc<dyn LatticeReceiver>>>,
        upstream: Mutex<Option<Arc<dyn LatticeSource>>>,
        chained_upstreams: AtomicU64,
    }

    impl LatticeTerminal for Doubler {
        fn add_upstream(&self, upstream: Arc<dyn LatticeSource>) {
            *self.upstream.lock().unwrap() = Some(upstream);
        }
    }

    impl LatticeReceiver for Doubler {
        fn push(&self, frame: Box<dyn Frame>) {
            let doubled = Box::new(BenchmarkFrame::new(frame.header().id_hash() * 2, None));
            if let Some(downstream) = self.downstream.lock().unwrap().as_ref() {
                downstream.push(doubled);
            }
        }

        fn on_complete(&self) {
            if let Some(downstream) = self.downstream.lock().unwrap().as_ref() {
                downstream.on_complete();
            }
        }

        fn on_error(&self, error: Failure) {
            if let Some(downstream) = self.downstream.lock().unwrap().as_ref() {
                downstream.on_error(error);
            }
        }
    }

    impl LatticeSource for Doubler {
        fn add_downstream(self: Arc<Self>, downstream: Arc<dyn LatticeReceiver>) {
            *self.downstream.lock().unwrap() = Some(downstream.clone());
            downstream.add_upstream(self);
        }

        fn pull(
            &self,
            consumer: &mut dyn FnMut(Box<dyn Frame>),
            stop: &mut dyn FnMut(&dyn Frame) -> bool,
            demand: u64,
        ) -> u64 {
            let upstream = self.upstream.lock().unwrap().clone();
            upstream.map_or(0, |upstream| upstream.pull(consumer, stop, demand))
        }

        fn request(&self, demand: u64) {
            let upstream = self.upstream.lock().unwrap().clone();
            if let Some(upstream) = upstream {
                upstream.request(demand);
            }
        }

        fn complete(&self) {}

        fn is_complete(&self) -> bool {
            false
        }
    }

    impl LatticeInterceptor for Doubler {
        fn add_upstream_interceptor(&self, _upstream: Arc<dyn LatticeInterceptor>) {
            self.chained_upstreams.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// An interceptor relying on the default chaining hooks.
    struct Passive;

    impl LatticeTerminal for Passive {
        fn add_upstream(&self, _upstream: Arc<dyn LatticeSource>) {}
    }

    impl LatticeReceiver for Passive {
        fn push(&self, _frame: Box<dyn Frame>) {}
        fn on_complete(&self) {}
        fn on_error(&self, _error: Failure) {}
    }

    impl LatticeSource for Passive {
        fn add_downstream(self: Arc<Self>, _downstream: Arc<dyn LatticeReceiver>) {}
        fn pull(
            &self,
            _consumer: &mut dyn FnMut(Box<dyn Frame>),
            _stop: &mut dyn FnMut(&dyn Frame) -> bool,
            _demand: u64,
        ) -> u64 {
            0
        }
        fn request(&self, _demand: u64) {}
        fn complete(&self) {}
        fn is_complete(&self) -> bool {
            true
        }
    }

    impl LatticeInterceptor for Passive {}

    #[test]
    fn a_source_attaches_itself_and_pushes_on_demand() {
        let source = QueueSource::with(&[1, 2, 3]);
        let recorder = Arc::new(Recorder {
            initial: 2,
            ..Recorder::default()
        });
        source.clone().add_downstream(recorder.clone());

        assert!(recorder.upstream.lock().unwrap().is_some());
        assert_eq!(*recorder.received.lock().unwrap(), [1, 2]);
        source.request(u64::MAX);
        assert_eq!(*recorder.received.lock().unwrap(), [1, 2, 3]);
    }

    #[test]
    fn pull_stops_before_the_first_frame_matching_the_stop_condition() {
        let source = QueueSource::with(&[1, 2, 3, 4]);
        let mut pulled = Vec::new();
        let count = source.pull(
            &mut |frame| pulled.push(frame.header().id_hash()),
            &mut |frame| frame.header().id_hash() == 3,
            10,
        );
        assert_eq!(count, 2);
        assert_eq!(pulled, [1, 2]);

        let count = source.pull(
            &mut |frame| pulled.push(frame.header().id_hash()),
            &mut |_| false,
            1,
        );
        assert_eq!(count, 1, "the stopping frame stayed in the source");
        assert_eq!(pulled, [1, 2, 3]);
    }

    #[test]
    fn completion_reaches_the_receiver_once() {
        let source = QueueSource::with(&[]);
        let recorder = Arc::new(Recorder::default());
        source.clone().add_downstream(recorder.clone());
        source.complete();
        source.complete();
        assert!(source.is_complete());
        assert!(recorder.completed.load(Ordering::Acquire));
    }

    #[test]
    fn an_interceptor_sits_between_a_source_and_a_receiver() {
        let source = QueueSource::with(&[1, 2, 3]);
        let doubler = Arc::new(Doubler::default());
        let recorder = Arc::new(Recorder {
            initial: u64::MAX,
            ..Recorder::default()
        });

        let as_source: Arc<dyn LatticeSource> = doubler.clone();
        source.clone().add_downstream(doubler.clone());
        as_source.add_downstream(recorder.clone());
        assert_eq!(*recorder.received.lock().unwrap(), [2, 4, 6]);

        doubler.on_error(FrameError::failed("upstream broke").into());
        assert_eq!(
            *recorder.errors.lock().unwrap(),
            ["frame failed: upstream broke"]
        );
    }

    #[test]
    fn interceptors_chain_through_the_interceptor_hooks() {
        let first = Arc::new(Doubler::default());
        let second = Arc::new(Doubler::default());
        second.add_upstream_interceptor(first.clone());
        assert_eq!(second.chained_upstreams.load(Ordering::Relaxed), 1);

        let passive = Arc::new(Passive);
        passive.add_upstream_interceptor(first.clone());
        passive.clone().add_downstream_interceptor(second);
        let as_receiver: Arc<dyn LatticeReceiver> = passive;
        as_receiver.push(Box::new(BenchmarkFrame::new(1, None)));
    }
}
