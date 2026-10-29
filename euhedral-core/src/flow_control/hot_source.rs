use crate::frames::{Frame, FrameError};
use crate::generics::{LatticeReceiver, LatticeSource};
#[allow(clippy::disallowed_types, reason = "HOT-FIX(C10)")]
use std::sync::{Arc, RwLock};

type BeforeEach = Box<dyn Fn(&dyn Frame) + Send + Sync>;
type AfterEach = Box<dyn Fn() + Send + Sync>;

#[derive(Default)]
struct State {
    downstream: Option<Arc<dyn LatticeReceiver>>,
    complete: bool,
}

/// A source its owner pushes frames into by hand, one stage handing frames to the next. It has at
/// most one downstream and pushes without waiting for demand; one thread is expected to do the
/// pushing.
///
/// Optional hooks run around each push, for example to time it.
#[derive(Default)]
pub struct LatticeHotSource {
    before_each: Option<BeforeEach>,
    after_each: Option<AfterEach>,
    #[allow(clippy::disallowed_types, reason = "HOT-FIX(C10)")]
    state: RwLock<State>,
}

impl LatticeHotSource {
    pub fn new() -> Self {
        Self::default()
    }

    /// Runs `hook` with each frame just before it is pushed.
    pub fn before_each(mut self, hook: impl Fn(&dyn Frame) + Send + Sync + 'static) -> Self {
        self.before_each = Some(Box::new(hook));
        self
    }

    /// Runs `hook` just after each push returns.
    pub fn after_each(mut self, hook: impl Fn() + Send + Sync + 'static) -> Self {
        self.after_each = Some(Box::new(hook));
        self
    }

    /// Pushes `frame` to the downstream. Without a downstream, or once the source is complete, the
    /// frame is finished with an error instead.
    ///
    /// A frame accepted while another thread completes the source may still reach the downstream
    /// being detached.
    pub fn accept(&self, frame: Box<dyn Frame>) {
        let downstream = self.state.read().unwrap().downstream.clone();
        let Some(downstream) = downstream else {
            frame.do_finally_with_error(FrameError::failed("the source has no downstream"));
            return;
        };

        if let Some(before_each) = &self.before_each {
            before_each(&*frame);
        }
        downstream.push(frame);
        if let Some(after_each) = &self.after_each {
            after_each();
        }
    }
}

impl LatticeSource for LatticeHotSource {
    /// Attaches `downstream` if the source has none, or reopens a complete source with it.
    /// Otherwise `downstream` is told of the error and the current one is kept.
    fn add_downstream(self: Arc<Self>, downstream: Arc<dyn LatticeReceiver>) {
        {
            let mut state = self.state.write().unwrap();
            if state.complete {
                state.complete = false;
            } else if state.downstream.is_some() {
                drop(state);
                downstream.on_error("the source already has a downstream".into());
                return;
            }
            state.downstream = Some(downstream.clone());
        }
        downstream.add_upstream(self);
    }

    /// Frames are only pushed, so there is never anything to pull.
    fn pull(
        &self,
        _consumer: &mut dyn FnMut(Box<dyn Frame>),
        _stop: &mut dyn FnMut(&dyn Frame) -> bool,
        _demand: u64,
    ) -> u64 {
        0
    }

    /// The source pushes regardless of demand.
    fn request(&self, _demand: u64) {}

    /// Detaches the downstream and tells it the source is complete.
    fn complete(&self) {
        let downstream = {
            let mut state = self.state.write().unwrap();
            state.complete = true;
            state.downstream.take()
        };
        if let Some(downstream) = downstream {
            downstream.on_complete();
        }
    }

    fn is_complete(&self) -> bool {
        self.state.read().unwrap().complete
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flow_control::testing::{Fate, Fates, TestReceiver, Tracked, frame};
    use crate::frames::Failure;
    use crate::generics::LatticeTerminal;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn attached() -> (Arc<LatticeHotSource>, Arc<TestReceiver>) {
        let source = Arc::new(LatticeHotSource::new());
        let receiver = TestReceiver::new();
        source.clone().add_downstream(receiver.clone());
        (source, receiver)
    }

    #[test]
    fn pushes_accepted_frames_to_the_downstream() {
        let (source, receiver) = attached();
        source.accept(frame(1));
        source.accept(frame(2));
        assert_eq!(receiver.received(), [1, 2]);
    }

    #[test]
    fn attaching_tells_the_downstream_about_the_source() {
        let (source, receiver) = attached();
        let upstream = receiver.upstream.lock().unwrap().clone().unwrap();
        let source: Arc<dyn LatticeSource> = source;
        assert!(Arc::ptr_eq(&upstream, &source));
    }

    #[test]
    fn hooks_run_around_each_push() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let (before, after) = (log.clone(), log.clone());
        let source = Arc::new(
            LatticeHotSource::new()
                .before_each(move |frame| {
                    before
                        .lock()
                        .unwrap()
                        .push(format!("before {}", frame.header().id_hash()))
                })
                .after_each(move || after.lock().unwrap().push("after".to_string())),
        );
        let receiver = TestReceiver::new();
        source.clone().add_downstream(receiver.clone());

        source.accept(frame(5));
        assert_eq!(*log.lock().unwrap(), ["before 5", "after"]);
        assert_eq!(receiver.received(), [5]);
    }

    /// Counts its pushes, and the pushes at which the before-hook led the after-hook by one.
    struct Leading {
        before: Arc<AtomicUsize>,
        after: Arc<AtomicUsize>,
        pushes: AtomicUsize,
        led: AtomicUsize,
    }

    impl LatticeTerminal for Leading {
        fn add_upstream(&self, _upstream: Arc<dyn LatticeSource>) {}
    }

    impl LatticeReceiver for Leading {
        fn push(&self, _frame: Box<dyn Frame>) {
            self.pushes.fetch_add(1, Ordering::SeqCst);
            if self.before.load(Ordering::SeqCst) == self.after.load(Ordering::SeqCst) + 1 {
                self.led.fetch_add(1, Ordering::SeqCst);
            }
        }

        fn on_complete(&self) {}

        fn on_error(&self, _error: Failure) {}
    }

    #[test]
    fn hooks_run_around_each_push_from_one_pusher() {
        const ROUNDS: usize = if cfg!(miri) { 20 } else { 1000 };
        let before = Arc::new(AtomicUsize::new(0));
        let after = Arc::new(AtomicUsize::new(0));
        let source = Arc::new(
            LatticeHotSource::new()
                .before_each({
                    let before = before.clone();
                    move |_| {
                        before.fetch_add(1, Ordering::SeqCst);
                    }
                })
                .after_each({
                    let after = after.clone();
                    move || {
                        after.fetch_add(1, Ordering::SeqCst);
                    }
                }),
        );
        let receiver = Arc::new(Leading {
            before: before.clone(),
            after: after.clone(),
            pushes: AtomicUsize::new(0),
            led: AtomicUsize::new(0),
        });
        source.clone().add_downstream(receiver.clone());

        for round in 1..=ROUNDS {
            source.accept(frame(round as u64));
            assert_eq!(before.load(Ordering::SeqCst), round);
            assert_eq!(after.load(Ordering::SeqCst), round);
        }
        assert_eq!(receiver.pushes.load(Ordering::SeqCst), ROUNDS);
        assert_eq!(receiver.led.load(Ordering::SeqCst), ROUNDS);
    }

    #[test]
    fn a_frame_without_a_downstream_fails_instead_of_vanishing() {
        let source = LatticeHotSource::new().before_each(|_| panic!("nothing is pushed"));
        let fates = Fates::default();
        source.accept(Tracked::new(3, &fates));
        assert_eq!(
            *fates.lock().unwrap(),
            [(
                3,
                Fate::Failed("frame failed: the source has no downstream".into())
            )]
        );
    }

    #[test]
    fn completing_detaches_and_notifies_the_downstream_once() {
        let (source, receiver) = attached();
        source.complete();
        assert!(source.is_complete());
        assert!(receiver.is_completed());

        receiver
            .completed
            .store(false, std::sync::atomic::Ordering::Release);
        source.complete();
        assert!(!receiver.is_completed());

        let fates = Fates::default();
        source.accept(Tracked::new(4, &fates));
        assert!(receiver.received().is_empty());
        assert!(matches!(fates.lock().unwrap()[0], (4, Fate::Failed(_))));
    }

    #[test]
    fn completing_without_a_downstream_is_quiet() {
        let source = LatticeHotSource::new();
        source.complete();
        assert!(source.is_complete());
    }

    #[test]
    fn rejects_a_second_downstream() {
        let (source, first) = attached();
        let second = TestReceiver::new();
        source.clone().add_downstream(second.clone());

        assert_eq!(second.errors(), ["the source already has a downstream"]);
        assert!(second.upstream.lock().unwrap().is_none());
        source.accept(frame(1));
        assert_eq!(first.received(), [1]);
        assert!(second.received().is_empty());
    }

    #[test]
    fn a_complete_source_reopens_with_a_new_downstream() {
        let (source, first) = attached();
        source.complete();

        let second = TestReceiver::new();
        source.clone().add_downstream(second.clone());
        assert!(!source.is_complete());
        assert!(second.errors().is_empty());
        source.accept(frame(9));
        assert_eq!(second.received(), [9]);
        assert!(first.received().is_empty());
    }

    #[test]
    fn has_nothing_to_pull_and_ignores_demand() {
        let (source, receiver) = attached();
        source.request(u64::MAX);
        let pulled = source.pull(&mut |_| panic!("nothing to pull"), &mut |_| false, 10);
        assert_eq!(pulled, 0);
        assert!(receiver.received().is_empty());
    }
}
