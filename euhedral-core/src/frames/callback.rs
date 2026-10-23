use super::{Frame, FrameError, FrameHeader};
use crate::generics::FramePusher;
use crate::impls::Recycler;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

/// The function a [`CallbackFrame`] applies, shareable between the frames of one pool.
pub type Callback<T, R> = Arc<dyn Fn(&T) -> R + Send + Sync>;

/// Where a [`CallbackFrame`] goes once it has a result.
pub type CallbackReceiver<T, R> = Arc<dyn FramePusher<CallbackFrame<T, R>>>;

/// A frame that applies a function to its payload and hands itself, result and all, to a receiver.
///
/// A frame that ran is pushed to the receiver when it finishes, and the receiver owns it from then
/// on: it reads the result and calls [`recycle`](Frame::recycle) when done. A frame that never ran,
/// because it was dead, and a frame that failed go straight back to their pool instead.
pub struct CallbackFrame<T, R> {
    header: FrameHeader,
    payload: T,
    ret_val: Option<R>,
    function: Callback<T, R>,
    receiver: CallbackReceiver<T, R>,
    recycler: Option<Recycler<CallbackFrame<T, R>>>,
}

impl<T, R> CallbackFrame<T, R> {
    /// An ordered frame without a pool or kill switch.
    pub fn new(
        id_hash: u64,
        payload: T,
        function: Callback<T, R>,
        receiver: CallbackReceiver<T, R>,
    ) -> Self {
        Self::with_recycler(id_hash, payload, function, receiver, None, None)
    }

    /// A frame that the receiver can return to `recycler`, and that dies with `kill_switch`.
    pub fn with_recycler(
        id_hash: u64,
        payload: T,
        function: Callback<T, R>,
        receiver: CallbackReceiver<T, R>,
        recycler: Option<Recycler<CallbackFrame<T, R>>>,
        kill_switch: Option<Arc<AtomicBool>>,
    ) -> Self {
        Self {
            header: FrameHeader::with_kill_switch(id_hash, kill_switch),
            payload,
            ret_val: None,
            function,
            receiver,
            recycler,
        }
    }

    pub fn payload(&self) -> &T {
        &self.payload
    }

    /// The function's result, once the frame has run.
    pub fn ret_val(&self) -> Option<&R> {
        self.ret_val.as_ref()
    }

    pub fn take_ret_val(&mut self) -> Option<R> {
        self.ret_val.take()
    }

    /// Puts in a new payload and clears the last result, handing back the old payload.
    pub fn replace(&mut self, payload: T) -> T {
        self.ret_val = None;
        std::mem::replace(&mut self.payload, payload)
    }
}

impl<T, R> Frame for CallbackFrame<T, R>
where
    T: Send + 'static,
    R: Send + 'static,
{
    fn header(&self) -> &FrameHeader {
        &self.header
    }

    fn header_mut(&mut self) -> &mut FrameHeader {
        &mut self.header
    }

    fn execute(&mut self) -> Result<(), FrameError> {
        self.ret_val = Some((self.function)(&self.payload));
        Ok(())
    }

    /// Hands a frame that ran to the receiver; returns one that did not to its pool.
    fn do_finally(self: Box<Self>) {
        if self.ret_val.is_some() {
            let receiver = Arc::clone(&self.receiver);
            receiver.push(self);
        } else {
            self.recycle();
        }
    }

    /// Clears the result and returns the frame to its pool, if it has one.
    fn recycle(mut self: Box<Self>) {
        self.ret_val = None;
        Recycler::recycle_owned(self, |frame| frame.recycler.as_ref());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frames::testing::run;
    use crate::impls::{FrameFactory, FrameManager};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::mpsc;

    type Frame64 = CallbackFrame<u64, String>;

    fn label() -> Callback<u64, String> {
        Arc::new(|value| format!("value-{value}"))
    }

    fn channel() -> (CallbackReceiver<u64, String>, mpsc::Receiver<Box<Frame64>>) {
        let (sender, received) = mpsc::channel();
        let receiver: CallbackReceiver<u64, String> = Arc::new(move |frame: Box<Frame64>| {
            sender.send(frame).unwrap();
        });
        (receiver, received)
    }

    fn pool(
        receiver: CallbackReceiver<u64, String>,
        calls: Arc<AtomicUsize>,
    ) -> FrameManager<u64, Frame64> {
        let mut manager = FrameManager::with_capacity(8);
        let recycler = manager.recycler();
        let function: Callback<u64, String> = Arc::new(move |value| {
            calls.fetch_add(1, Ordering::Relaxed);
            format!("value-{value}")
        });
        manager
            .set_factory(FrameFactory::new(
                move |id, payload| {
                    CallbackFrame::with_recycler(
                        id,
                        payload,
                        function.clone(),
                        receiver.clone(),
                        Some(recycler.clone()),
                        None,
                    )
                },
                |payload, frame: &mut Frame64| {
                    frame.replace(payload);
                },
            ))
            .unwrap();
        manager
    }

    #[test]
    fn execute_applies_the_function_to_the_payload() {
        let (receiver, _received) = channel();
        let mut frame = CallbackFrame::new(1, 7, label(), receiver);
        assert_eq!(frame.payload(), &7);
        assert_eq!(frame.ret_val(), None);
        frame.execute().unwrap();
        assert_eq!(frame.ret_val().map(String::as_str), Some("value-7"));
        assert_eq!(frame.take_ret_val().as_deref(), Some("value-7"));
        assert_eq!(frame.ret_val(), None);
    }

    #[test]
    fn a_frame_that_ran_goes_to_the_receiver_until_it_recycles_it() {
        let (receiver, received) = channel();
        let calls = Arc::new(AtomicUsize::new(0));
        let mut manager = pool(receiver, calls.clone());

        let frame = manager.get_or_create(7);
        let first = &*frame as *const Frame64;
        run(frame);
        assert_eq!(manager.recycle_queue_size(), 0, "the receiver owns it now");

        let delivered = received.try_recv().unwrap();
        assert_eq!(&*delivered as *const Frame64, first);
        assert_eq!(delivered.payload(), &7);
        assert_eq!(delivered.ret_val().map(String::as_str), Some("value-7"));
        delivered.recycle();
        assert_eq!(manager.recycle_queue_size(), 1);

        let reused = manager.get_or_create(8);
        assert_eq!(&*reused as *const Frame64, first);
        assert_eq!(reused.payload(), &8);
        assert_eq!(reused.ret_val(), None);
        run(reused);
        let delivered = received.try_recv().unwrap();
        assert_eq!(delivered.ret_val().map(String::as_str), Some("value-8"));
        assert_eq!(calls.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn a_dead_frame_goes_back_to_its_pool_without_running() {
        let (receiver, received) = channel();
        let calls = Arc::new(AtomicUsize::new(0));
        let manager = pool(receiver.clone(), calls.clone());
        let kill_switch = Arc::new(AtomicBool::new(true));
        let frame = CallbackFrame::with_recycler(
            1,
            7,
            label(),
            receiver,
            Some(manager.recycler()),
            Some(kill_switch),
        );

        run(Box::new(frame));
        assert!(received.try_recv().is_err());
        assert_eq!(manager.recycle_queue_size(), 1);
        assert_eq!(calls.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn a_failed_frame_goes_back_to_its_pool_without_a_result() {
        let (receiver, received) = channel();
        let mut manager = FrameManager::<u64, Frame64>::with_capacity(8);
        let mut frame = Box::new(CallbackFrame::with_recycler(
            1,
            7,
            label(),
            receiver,
            Some(manager.recycler()),
            None,
        ));
        frame.execute().unwrap();
        frame.do_finally_with_error(FrameError::failed("downstream failed"));

        assert!(received.try_recv().is_err());
        let returned = manager.get().unwrap();
        assert_eq!(returned.ret_val(), None, "recycling clears the result");
    }

    #[test]
    fn replace_hands_back_the_old_payload_and_clears_the_result() {
        let (receiver, _received) = channel();
        let mut frame = CallbackFrame::new(1, 7, label(), receiver);
        frame.execute().unwrap();
        assert_eq!(frame.replace(9), 7);
        assert_eq!(frame.payload(), &9);
        assert_eq!(frame.ret_val(), None);
    }

    #[test]
    fn a_frame_without_a_pool_is_dropped_by_recycle() {
        let (receiver, received) = channel();
        run(Box::new(CallbackFrame::new(1, 7, label(), receiver)));
        received.try_recv().unwrap().recycle();
    }
}
