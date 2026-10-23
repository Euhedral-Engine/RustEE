use crate::frames::Frame;
use crate::generics::FramePusher;
use crate::utils::spin_wait;

/// What a [`PushHook`] did with a frame.
pub enum Response<F: ?Sized> {
    /// The frame was taken.
    Ok,
    /// The frame could not be taken yet; it is handed back to be offered again.
    Retry(Box<F>),
    /// The frame is refused; it is recycled.
    Cancel(Box<F>),
    /// The frame is refused and its work stopped: it is killed, then recycled.
    Terminate(Box<F>),
}

/// The decision behind a [`LatticeHotReceiver`]: takes, defers or refuses each frame.
pub trait PushHook<F: ?Sized>: Send + Sync {
    fn hook_on_push(&self, frame: Box<F>) -> Response<F>;
}

/// A [`FramePusher`] that offers each frame to its [`PushHook`] until the hook stops asking for a
/// retry, backing off between attempts the way [`spin_wait::await_while`] does.
pub struct LatticeHotReceiver<H> {
    hook: H,
}

impl<H> LatticeHotReceiver<H> {
    pub fn new(hook: H) -> Self {
        Self { hook }
    }

    pub fn hook(&self) -> &H {
        &self.hook
    }
}

impl<F, H> FramePusher<F> for LatticeHotReceiver<H>
where
    F: Frame + ?Sized,
    H: PushHook<F>,
{
    fn push(&self, frame: Box<F>) {
        let mut pending = Some(frame);
        let mut refused = None;
        spin_wait::await_while(|| {
            let frame = pending
                .take()
                .expect("a retried frame is always handed back");
            match self.hook.hook_on_push(frame) {
                Response::Retry(frame) => {
                    pending = Some(frame);
                    true
                }
                response => {
                    refused = Some(response);
                    false
                }
            }
        });
        match refused {
            Some(Response::Cancel(frame)) => frame.recycle(),
            Some(Response::Terminate(frame)) => {
                frame.kill();
                frame.recycle();
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frames::{FrameError, FrameHeader};
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    /// Counts how it was finished.
    struct Probe {
        header: FrameHeader,
        recycled: Arc<AtomicUsize>,
    }

    impl Probe {
        fn new(kill_switch: Option<Arc<AtomicBool>>) -> (Box<Self>, Arc<AtomicUsize>) {
            let recycled = Arc::new(AtomicUsize::new(0));
            let probe = Box::new(Self {
                header: FrameHeader::with_kill_switch(7, kill_switch),
                recycled: recycled.clone(),
            });
            (probe, recycled)
        }
    }

    impl Frame for Probe {
        fn header(&self) -> &FrameHeader {
            &self.header
        }

        fn header_mut(&mut self) -> &mut FrameHeader {
            &mut self.header
        }

        fn do_finally_with_error(self: Box<Self>, error: FrameError) {
            panic!("a refused frame is recycled, not failed: {error}");
        }

        fn recycle(self: Box<Self>) {
            self.recycled.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Replays a script of responses, recording each frame it is offered.
    struct Script {
        responses: Mutex<Vec<&'static str>>,
        kept: Mutex<Vec<Probe>>,
        offers: AtomicUsize,
    }

    impl Script {
        fn new(responses: &[&'static str]) -> Self {
            let mut responses = responses.to_vec();
            responses.reverse();
            Self {
                responses: Mutex::new(responses),
                kept: Mutex::new(Vec::new()),
                offers: AtomicUsize::new(0),
            }
        }
    }

    impl PushHook<Probe> for Script {
        fn hook_on_push(&self, frame: Box<Probe>) -> Response<Probe> {
            self.offers.fetch_add(1, Ordering::Relaxed);
            match self.responses.lock().unwrap().pop().unwrap() {
                "ok" => {
                    self.kept.lock().unwrap().push(*frame);
                    Response::Ok
                }
                "retry" => Response::Retry(frame),
                "cancel" => Response::Cancel(frame),
                "terminate" => Response::Terminate(frame),
                other => unreachable!("{other}"),
            }
        }
    }

    #[test]
    fn a_taken_frame_is_offered_once() {
        let receiver = LatticeHotReceiver::new(Script::new(&["ok"]));
        let (frame, recycled) = Probe::new(None);
        receiver.push(frame);
        assert_eq!(receiver.hook().offers.load(Ordering::Relaxed), 1);
        assert_eq!(receiver.hook().kept.lock().unwrap().len(), 1);
        assert_eq!(recycled.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn retries_until_the_hook_takes_the_frame() {
        let mut script = vec!["retry"; 600];
        script.push("ok");
        let receiver = LatticeHotReceiver::new(Script::new(&script));
        let (frame, _) = Probe::new(None);
        receiver.push(frame);
        assert_eq!(receiver.hook().offers.load(Ordering::Relaxed), 601);
        assert_eq!(receiver.hook().kept.lock().unwrap().len(), 1);
    }

    #[test]
    fn a_cancelled_frame_is_recycled_and_left_alive() {
        let receiver = LatticeHotReceiver::new(Script::new(&["retry", "cancel"]));
        let kill_switch = Arc::new(AtomicBool::new(false));
        let (frame, recycled) = Probe::new(Some(kill_switch.clone()));
        receiver.push(frame);
        assert_eq!(recycled.load(Ordering::Relaxed), 1);
        assert!(!kill_switch.load(Ordering::Acquire));
    }

    #[test]
    fn a_terminated_frame_is_killed_and_recycled() {
        let receiver = LatticeHotReceiver::new(Script::new(&["terminate"]));
        let kill_switch = Arc::new(AtomicBool::new(false));
        let (frame, recycled) = Probe::new(Some(kill_switch.clone()));
        receiver.push(frame);
        assert_eq!(recycled.load(Ordering::Relaxed), 1);
        assert!(kill_switch.load(Ordering::Acquire));
    }

    #[test]
    fn works_with_trait_object_frames() {
        struct TakeAll(AtomicUsize);
        impl PushHook<dyn Frame> for TakeAll {
            fn hook_on_push(&self, _frame: Box<dyn Frame>) -> Response<dyn Frame> {
                self.0.fetch_add(1, Ordering::Relaxed);
                Response::Ok
            }
        }

        let receiver = LatticeHotReceiver::new(TakeAll(AtomicUsize::new(0)));
        let pusher: &dyn FramePusher<dyn Frame> = &receiver;
        pusher.push(Box::new(crate::frames::BenchmarkFrame::new(1, None)));
        assert_eq!(receiver.hook().0.load(Ordering::Relaxed), 1);
    }
}
