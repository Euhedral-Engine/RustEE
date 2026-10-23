//! Interfaces between frames and whatever receives them.

/// Takes ownership of frames handed back by the frames themselves, such as finished
/// [`CallbackFrame`](crate::frames::CallbackFrame)s.
///
/// Any `Fn(Box<F>) + Send + Sync` closure is a pusher.
pub trait FramePusher<F: ?Sized>: Send + Sync {
    fn push(&self, frame: Box<F>);
}

impl<F: ?Sized, P> FramePusher<F> for P
where
    P: Fn(Box<F>) + Send + Sync,
{
    fn push(&self, frame: Box<F>) {
        self(frame)
    }
}
