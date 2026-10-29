//! A source its owner pushes frames into by hand.
//!
//! The source keeps its state in one word, `downstream`: null while it is open with no receiver, a [`Link`] while a
//! receiver is attached, and [`COMPLETE`] once it has completed. Java keeps two words, `complete` and `downstream`,
//! and sets them in two steps; with one word `complete` and `add_downstream` are each a single atomic step, so
//! however they race, a receiver is either refused, or attached and told `on_complete` once if the source ends
//! complete.
//!
//! A receiver that is detached may still be in the hands of a pusher that read it just before, and Java's collector
//! keeps it alive; here every receiver the source installs stays in a [`Link`] until the source drops. A link is
//! installed once, removed once (by `complete`), and then pushed onto the `retired` stack, which only grows and is read
//! again only when the source drops. A source attached `k` times holds `k` retired receivers, and what they own,
//! until it drops.

// The links the source installed are freed only when it drops, so a pusher may read one through a shared reference.
#![allow(
    unsafe_code,
    reason = "links are kept until the source drops, so the pointer a pusher read stays valid"
)]

use crate::frames::{Failure, Frame, FrameError};
use crate::generics::{AlreadyAttached, LatticeReceiver, LatticeSource};
use std::panic::{self, AssertUnwindSafe};
use std::ptr;
use std::sync::Arc;
use std::sync::atomic::{AtomicPtr, Ordering};
use std::thread;

type BeforeEach = Box<dyn Fn(&dyn Frame) + Send + Sync>;
type AfterEach = Box<dyn Fn() + Send + Sync>;

/// The `downstream` word of a completed source. Never the address of a link: no allocation sits at address 8.
const COMPLETE: *mut Link = ptr::without_provenance_mut(8);

/// A receiver the source installed. Its allocation lives until the source drops.
struct Link {
    receiver: Arc<dyn LatticeReceiver>,
    /// The link retired before this one. Stored once, before this link is published on `retired`.
    next: AtomicPtr<Link>,
}

// A pusher on any thread reads `receiver` through a shared reference to the link.
const _: () = {
    const fn shared<T: Send + Sync>() {}
    shared::<Link>();
};

/// The link a `downstream` word holds, if it holds one.
///
/// # Safety
/// `word` was read from a source's `downstream`, which the caller still borrows.
unsafe fn link_in<'a>(word: *mut Link) -> Option<&'a Link> {
    if word.is_null() || word == COMPLETE {
        return None;
    }
    // SAFETY: every other value of the word is a link that was published by a compare-and-set, and a published link
    // is freed only when the source drops.
    Some(unsafe { &*word })
}

/// Runs the after-hook when dropped, so it follows the before-hook whether or not the push returns.
struct AfterGuard<'a>(Option<&'a AfterEach>);

impl Drop for AfterGuard<'_> {
    fn drop(&mut self) {
        let Some(after_each) = self.0 else {
            return;
        };
        if thread::panicking() {
            // A second panic during unwinding would abort; the first one is the one that goes on.
            let _ = panic::catch_unwind(AssertUnwindSafe(after_each));
        } else {
            after_each();
        }
    }
}

/// A source its owner pushes frames into by hand, one stage handing frames to the next. It has at
/// most one downstream and pushes without waiting for demand; one thread is expected to do the
/// pushing.
///
/// Optional hooks run around each push, for example to time it. They are set when the source is
/// built and called without a lock, so they are `Fn` and `Sync`. The after-hook follows every
/// before-hook that returned, also when the push panics.
///
/// Every state change of the source is an atomic operation: attaching, completing and pushing take no lock,
/// and a call that detaches a receiver does not free it.
#[derive(Default)]
pub struct LatticeHotSource {
    before_each: Option<BeforeEach>,
    after_each: Option<AfterEach>,
    /// Null, a published link, or [`COMPLETE`].
    downstream: AtomicPtr<Link>,
    /// Links that were installed and then removed, newest first.
    retired: AtomicPtr<Link>,
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
    /// frame is finished with an error instead. A before-hook that panics finishes the frame with an
    /// error too, and the panic goes on.
    ///
    /// A frame accepted while another thread completes the source may still reach the downstream
    /// being detached.
    pub fn accept(&self, frame: Box<dyn Frame>) {
        // SAFETY: `&self` borrows the source.
        if unsafe { link_in(self.downstream.load(Ordering::Acquire)) }.is_none() {
            frame.do_finally_with_error(FrameError::failed("the source has no downstream"));
            return;
        }

        if let Some(before_each) = &self.before_each
            && let Err(panic) = panic::catch_unwind(AssertUnwindSafe(|| before_each(&*frame)))
        {
            frame.do_finally_with_error(FrameError::failed("the source's before-hook panicked"));
            panic::resume_unwind(panic);
        }
        let _after = AfterGuard(self.after_each.as_ref());
        // Read again after the hook, as Java does: the source may have completed meanwhile.
        // SAFETY: `&self` borrows the source.
        match unsafe { link_in(self.downstream.load(Ordering::Acquire)) } {
            Some(link) => link.receiver.push(frame),
            None => frame.do_finally_with_error(FrameError::failed("the source has no downstream")),
        }
    }

    /// Keeps `link`, which this call has just removed from `downstream`, until the source drops.
    fn retire(&self, link: *mut Link) {
        // SAFETY: the caller removed the link from `downstream`, so it is no one's but the stack's from here on, and
        // it is kept until the source drops.
        let link_ref = unsafe { &*link };
        let mut head = self.retired.load(Ordering::Relaxed);
        loop {
            link_ref.next.store(head, Ordering::Relaxed);
            match self.retired.compare_exchange_weak(
                head,
                link,
                Ordering::Release,
                Ordering::Relaxed,
            ) {
                Ok(_) => return,
                Err(observed) => head = observed,
            }
        }
    }
}

impl Drop for LatticeHotSource {
    fn drop(&mut self) {
        let installed = *self.downstream.get_mut();
        let installed = if installed == COMPLETE {
            ptr::null_mut()
        } else {
            installed
        };
        for head in [
            installed,
            std::mem::replace(self.retired.get_mut(), ptr::null_mut()),
        ] {
            let mut next = head;
            while !next.is_null() {
                // SAFETY: `&mut self` means no pusher or lifecycle call is running. Each link is either installed
                // or on the stack, never both, and was made by `Box::into_raw`.
                let link = unsafe { Box::from_raw(next) };
                next = link.next.load(Ordering::Relaxed);
            }
        }
    }
}

impl LatticeSource for LatticeHotSource {
    /// Attaches `downstream` if the source has none, or reopens a complete source with it. Otherwise
    /// `downstream` is told of the error and the current one is kept.
    fn add_downstream(self: Arc<Self>, downstream: Arc<dyn LatticeReceiver>) {
        let link = Box::into_raw(Box::new(Link {
            receiver: downstream.clone(),
            next: AtomicPtr::new(ptr::null_mut()),
        }));
        let mut observed = self.downstream.load(Ordering::Acquire);
        loop {
            // SAFETY: `self` borrows the source.
            if unsafe { link_in(observed) }.is_some() {
                // SAFETY: the link was made above and no compare-and-set installed it.
                drop(unsafe { Box::from_raw(link) });
                downstream.on_error(Failure::from(AlreadyAttached));
                return;
            }
            match self.downstream.compare_exchange(
                observed,
                link,
                Ordering::SeqCst,
                Ordering::SeqCst,
            ) {
                Ok(_) => break,
                Err(current) => observed = current,
            }
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
        let word = self.downstream.swap(COMPLETE, Ordering::SeqCst);
        // SAFETY: `&self` borrows the source.
        let Some(removed) = (unsafe { link_in(word) }) else {
            return;
        };
        // The swap made this call the link's remover.
        let receiver = removed.receiver.clone();
        self.retire(word);
        receiver.on_complete();
    }

    fn is_complete(&self) -> bool {
        self.downstream.load(Ordering::Acquire) == COMPLETE
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flow_control::testing::{Fate, Fates, TestReceiver, Tracked, frame};
    use crate::frames::FrameHeader;
    use crate::generics::LatticeTerminal;
    use std::panic::AssertUnwindSafe;
    use std::sync::atomic::AtomicUsize;
    use std::sync::{Barrier, Mutex, Weak};
    use std::thread;

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

    /// How many frames were finished, and how many were failed.
    #[derive(Default)]
    struct Outcomes {
        finished: AtomicUsize,
        failed: AtomicUsize,
    }

    impl Outcomes {
        fn total(&self) -> usize {
            self.finished.load(Ordering::SeqCst) + self.failed.load(Ordering::SeqCst)
        }
    }

    /// A frame that reports how it was finished.
    struct Counted {
        header: FrameHeader,
        outcomes: Arc<Outcomes>,
    }

    impl Counted {
        fn new(id: u64, outcomes: &Arc<Outcomes>) -> Box<Self> {
            Box::new(Self {
                header: FrameHeader::new(id),
                outcomes: outcomes.clone(),
            })
        }
    }

    impl Frame for Counted {
        fn header(&self) -> &FrameHeader {
            &self.header
        }

        fn header_mut(&mut self) -> &mut FrameHeader {
            &mut self.header
        }

        fn do_finally(self: Box<Self>) {
            self.outcomes.finished.fetch_add(1, Ordering::SeqCst);
        }

        fn do_finally_with_error(self: Box<Self>, _error: FrameError) {
            self.outcomes.failed.fetch_add(1, Ordering::SeqCst);
        }
    }

    /// Counts what it is told, and finishes the frames it is pushed. It holds no reference to its source.
    #[derive(Default)]
    struct Tally {
        pushes: AtomicUsize,
        completes: AtomicUsize,
        errors: AtomicUsize,
        upstreams: AtomicUsize,
    }

    impl LatticeTerminal for Tally {
        fn add_upstream(&self, _upstream: Arc<dyn LatticeSource>) {
            self.upstreams.fetch_add(1, Ordering::SeqCst);
        }
    }

    impl LatticeReceiver for Tally {
        fn push(&self, frame: Box<dyn Frame>) {
            self.pushes.fetch_add(1, Ordering::SeqCst);
            frame.do_finally();
        }

        fn on_complete(&self) {
            self.completes.fetch_add(1, Ordering::SeqCst);
        }

        fn on_error(&self, _error: Failure) {
            self.errors.fetch_add(1, Ordering::SeqCst);
        }
    }

    const FRAMES: u64 = if cfg!(miri) { 200 } else { 100_000 };
    const CYCLES: usize = if cfg!(miri) { 6 } else { 100 };

    #[test]
    fn complete_while_accepting_never_frees_the_receiver_in_use() {
        let source = Arc::new(LatticeHotSource::new());
        let outcomes = Arc::new(Outcomes::default());
        let first = Arc::new(Tally::default());
        source.clone().add_downstream(first.clone());
        let mut kept: Vec<Weak<Tally>> = vec![Arc::downgrade(&first)];
        drop(first);

        thread::scope(|scope| {
            let pusher = scope.spawn(|| {
                for id in 0..FRAMES {
                    source.accept(Counted::new(id, &outcomes));
                }
            });
            // Each receiver is owned by the source alone once attached.
            for _ in 0..CYCLES {
                source.complete();
                let receiver = Arc::new(Tally::default());
                kept.push(Arc::downgrade(&receiver));
                source.clone().add_downstream(receiver);
                thread::yield_now();
            }
            source.complete();
            pusher.join().unwrap();
        });

        // Every frame was delivered or failed, none lost.
        assert_eq!(outcomes.total(), FRAMES as usize);
        let tallies: Vec<Arc<Tally>> = kept.iter().map(|tally| tally.upgrade().unwrap()).collect();
        let delivered: usize = tallies
            .iter()
            .map(|tally| tally.pushes.load(Ordering::SeqCst))
            .sum();
        assert_eq!(delivered, outcomes.finished.load(Ordering::SeqCst));
        for tally in &tallies {
            assert_eq!(tally.completes.load(Ordering::SeqCst), 1);
            assert_eq!(tally.upstreams.load(Ordering::SeqCst), 1);
            assert_eq!(tally.errors.load(Ordering::SeqCst), 0);
        }
    }

    #[test]
    fn a_complete_source_reopens_with_a_new_receiver() {
        let source = Arc::new(LatticeHotSource::new());
        let outcomes = Arc::new(Outcomes::default());
        let mut receivers = Vec::new();
        for round in 0..3 {
            let receiver = Arc::new(Tally::default());
            source.clone().add_downstream(receiver.clone());
            assert!(!source.is_complete());
            source.accept(Counted::new(round, &outcomes));
            source.complete();
            assert!(source.is_complete());
            source.accept(Counted::new(round, &outcomes));
            receivers.push(receiver);
        }
        for receiver in &receivers {
            assert_eq!(receiver.pushes.load(Ordering::SeqCst), 1);
            assert_eq!(receiver.completes.load(Ordering::SeqCst), 1);
            assert_eq!(receiver.errors.load(Ordering::SeqCst), 0);
        }
        assert_eq!(outcomes.finished.load(Ordering::SeqCst), 3);
        assert_eq!(outcomes.failed.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn complete_tells_the_old_receiver_on_complete_once() {
        const COMPLETERS: usize = if cfg!(miri) { 2 } else { 4 };
        let rounds = if cfg!(miri) { 4 } else { 500 };
        for _ in 0..rounds {
            let source = Arc::new(LatticeHotSource::new());
            let receiver = Arc::new(Tally::default());
            source.clone().add_downstream(receiver.clone());
            let barrier = Barrier::new(COMPLETERS);
            thread::scope(|scope| {
                for _ in 0..COMPLETERS {
                    scope.spawn(|| {
                        barrier.wait();
                        source.complete();
                    });
                }
            });
            assert_eq!(receiver.completes.load(Ordering::SeqCst), 1);
            assert!(source.is_complete());
        }
    }

    #[test]
    fn racing_attaches_install_one_receiver_and_refuse_the_rest() {
        const ATTACHERS: usize = if cfg!(miri) { 3 } else { 8 };
        let rounds = if cfg!(miri) { 3 } else { 200 };
        for _ in 0..rounds {
            let source = Arc::new(LatticeHotSource::new());
            let receivers: Vec<Arc<Tally>> =
                (0..ATTACHERS).map(|_| Arc::new(Tally::default())).collect();
            let barrier = Barrier::new(ATTACHERS);
            thread::scope(|scope| {
                for receiver in &receivers {
                    let (source, barrier) = (&source, &barrier);
                    scope.spawn(move || {
                        barrier.wait();
                        source.clone().add_downstream(receiver.clone());
                    });
                }
            });
            let attached: Vec<_> = receivers
                .iter()
                .filter(|receiver| receiver.upstreams.load(Ordering::SeqCst) == 1)
                .collect();
            assert_eq!(attached.len(), 1);
            let refused: usize = receivers
                .iter()
                .map(|receiver| receiver.errors.load(Ordering::SeqCst))
                .sum();
            assert_eq!(refused, ATTACHERS - 1);

            let outcomes = Arc::new(Outcomes::default());
            source.accept(Counted::new(1, &outcomes));
            assert_eq!(attached[0].pushes.load(Ordering::SeqCst), 1);
        }
    }

    #[test]
    fn attaching_and_completing_from_many_threads_leaves_every_receiver_told_at_most_once() {
        const LIFECYCLE_THREADS: usize = 3;
        let source = Arc::new(LatticeHotSource::new());
        let outcomes = Arc::new(Outcomes::default());
        let attempts = Mutex::new(Vec::new());
        thread::scope(|scope| {
            let pusher = scope.spawn(|| {
                for id in 0..FRAMES {
                    source.accept(Counted::new(id, &outcomes));
                }
            });
            for _ in 0..LIFECYCLE_THREADS {
                scope.spawn(|| {
                    let mut mine = Vec::new();
                    for step in 0..CYCLES * 2 {
                        if step % 2 == 0 {
                            source.complete();
                        } else {
                            let receiver = Arc::new(Tally::default());
                            mine.push(receiver.clone());
                            source.clone().add_downstream(receiver);
                        }
                    }
                    attempts.lock().unwrap().extend(mine);
                });
            }
            pusher.join().unwrap();
        });
        assert_eq!(outcomes.total(), FRAMES as usize);
        let attempts = attempts.into_inner().unwrap();
        let delivered: usize = attempts
            .iter()
            .map(|tally| tally.pushes.load(Ordering::SeqCst))
            .sum();
        assert_eq!(delivered, outcomes.finished.load(Ordering::SeqCst));
        for tally in &attempts {
            // Told of its attach or of its refusal, never both; told of the end at most once.
            let attached = tally.upstreams.load(Ordering::SeqCst);
            let refused = tally.errors.load(Ordering::SeqCst);
            assert_eq!(attached + refused, 1);
            assert!(tally.completes.load(Ordering::SeqCst) <= 1);
            if refused == 1 {
                assert_eq!(tally.pushes.load(Ordering::SeqCst), 0);
                assert_eq!(tally.completes.load(Ordering::SeqCst), 0);
            }
        }
    }

    #[test]
    fn dropping_the_source_drops_every_receiver_it_kept() {
        let source = Arc::new(LatticeHotSource::new());
        let mut kept = Vec::new();
        for _ in 0..3 {
            let receiver = Arc::new(Tally::default());
            kept.push(Arc::downgrade(&receiver));
            source.clone().add_downstream(receiver);
            source.complete();
        }
        // One attached and not completed, one refused.
        let attached = Arc::new(Tally::default());
        kept.push(Arc::downgrade(&attached));
        source.clone().add_downstream(attached);
        let refused = Arc::new(Tally::default());
        source.clone().add_downstream(refused.clone());
        assert_eq!(refused.errors.load(Ordering::SeqCst), 1);
        assert!(kept.iter().all(|receiver| receiver.upgrade().is_some()));

        drop(source);

        assert!(kept.iter().all(|receiver| receiver.upgrade().is_none()));
        assert_eq!(Arc::strong_count(&refused), 1);
    }

    #[test]
    fn a_source_completed_by_its_before_hook_fails_the_frame_and_runs_the_after_hook() {
        let after = Arc::new(AtomicUsize::new(0));
        let source = Arc::new_cyclic(|this: &Weak<LatticeHotSource>| {
            let this = this.clone();
            LatticeHotSource::new()
                .before_each(move |_| this.upgrade().unwrap().complete())
                .after_each({
                    let after = after.clone();
                    move || {
                        after.fetch_add(1, Ordering::SeqCst);
                    }
                })
        });
        let receiver = Arc::new(Tally::default());
        source.clone().add_downstream(receiver.clone());
        let outcomes = Arc::new(Outcomes::default());
        source.accept(Counted::new(1, &outcomes));
        assert_eq!(outcomes.failed.load(Ordering::SeqCst), 1);
        assert_eq!(receiver.pushes.load(Ordering::SeqCst), 0);
        assert_eq!(receiver.completes.load(Ordering::SeqCst), 1);
        assert_eq!(after.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn attach_racing_complete_tells_every_receiver_exactly_once() {
        const ATTACHERS: usize = 2;
        let rounds = if cfg!(miri) { 4 } else { 20_000 };
        for round in 0..rounds {
            let source = Arc::new(LatticeHotSource::new());
            // Half of the rounds start with a receiver already attached.
            let first = (round % 2 == 0).then(|| {
                let first = Arc::new(Tally::default());
                source.clone().add_downstream(first.clone());
                first
            });
            let attached: Vec<Arc<Tally>> =
                (0..ATTACHERS).map(|_| Arc::new(Tally::default())).collect();
            let barrier = Barrier::new(ATTACHERS + 1);
            thread::scope(|scope| {
                scope.spawn(|| {
                    barrier.wait();
                    source.complete();
                });
                for receiver in &attached {
                    let (source, barrier) = (&source, &barrier);
                    scope.spawn(move || {
                        barrier.wait();
                        source.clone().add_downstream(receiver.clone());
                    });
                }
            });

            let all: Vec<&Arc<Tally>> = first.iter().chain(attached.iter()).collect();
            let live = |tally: &&Arc<Tally>| tally.errors.load(Ordering::SeqCst) == 0;
            let told = |tally: &Arc<Tally>| tally.completes.load(Ordering::SeqCst);
            // A refused receiver hears of nothing else.
            for refused in all.iter().filter(|tally| !live(tally)) {
                assert_eq!(refused.upstreams.load(Ordering::SeqCst), 0);
                assert_eq!(told(refused), 0);
            }
            let outcomes = Arc::new(Outcomes::default());
            source.accept(Counted::new(1, &outcomes));
            if source.is_complete() {
                // Every receiver the source held was told.
                assert_eq!(outcomes.failed.load(Ordering::SeqCst), 1);
                for tally in all.iter().filter(|tally| live(tally)) {
                    assert_eq!(told(tally), 1, "round {round}: a receiver was never told");
                }
            } else {
                // An open source has one live receiver, and it was not told; the others were.
                assert_eq!(outcomes.finished.load(Ordering::SeqCst), 1);
                let open: Vec<_> = all
                    .iter()
                    .filter(|tally| live(tally) && told(tally) == 0)
                    .collect();
                assert_eq!(open.len(), 1, "round {round}");
                assert_eq!(open[0].pushes.load(Ordering::SeqCst), 1, "round {round}");
                for tally in all.iter().filter(|tally| live(tally)) {
                    assert!(told(tally) <= 1);
                }
            }
            source.complete();
            for tally in all.iter().filter(|tally| live(tally)) {
                assert_eq!(told(tally), 1, "round {round}");
            }
        }
    }

    /// Panics in `push`.
    struct PanicsOnPush;

    impl LatticeTerminal for PanicsOnPush {
        fn add_upstream(&self, _upstream: Arc<dyn LatticeSource>) {}
    }

    impl LatticeReceiver for PanicsOnPush {
        fn push(&self, _frame: Box<dyn Frame>) {
            panic!("the receiver failed");
        }

        fn on_complete(&self) {}

        fn on_error(&self, _error: Failure) {}
    }

    #[test]
    fn the_after_hook_runs_when_the_push_panics() {
        let (before, after) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
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
        source.clone().add_downstream(Arc::new(PanicsOnPush));
        let outcomes = Arc::new(Outcomes::default());
        let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
            source.accept(Counted::new(1, &outcomes));
        }));
        assert!(result.is_err());
        assert_eq!(before.load(Ordering::SeqCst), 1);
        assert_eq!(after.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn a_panicking_before_hook_fails_the_frame_and_the_panic_goes_on() {
        let after = Arc::new(AtomicUsize::new(0));
        let source = Arc::new(
            LatticeHotSource::new()
                .before_each(|_| panic!("the hook failed"))
                .after_each({
                    let after = after.clone();
                    move || {
                        after.fetch_add(1, Ordering::SeqCst);
                    }
                }),
        );
        let receiver = Arc::new(Tally::default());
        source.clone().add_downstream(receiver.clone());
        let outcomes = Arc::new(Outcomes::default());
        let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
            source.accept(Counted::new(1, &outcomes));
        }));
        assert!(result.is_err());
        assert_eq!(outcomes.failed.load(Ordering::SeqCst), 1);
        assert_eq!(receiver.pushes.load(Ordering::SeqCst), 0);
        // The hooks pair up after a before-hook that returned.
        assert_eq!(after.load(Ordering::SeqCst), 0);
    }
}
