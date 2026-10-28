//! The split between a stream's shared query side and its single-owner submitter, checked by the compiler.
#![cfg(feature = "fake")]

use euhedral_gpu::fake::{FakeGraph, FakeQueue, FakeStream};

/// `check` resolves only when `T` is *not* `Sync`: for a `Sync` type both impls apply and the call is ambiguous, so
/// the file does not compile.
trait AmbiguousIfSync<A> {
    fn check() {}
}

impl<T: ?Sized> AmbiguousIfSync<()> for T {}

impl<T: ?Sized + Sync> AmbiguousIfSync<u8> for T {}

fn is_send<T: Send>() {}

fn is_sync<T: Sync>() {}

#[test]
fn a_submitter_has_one_owner_it_moves_but_is_not_shared() {
    // Send: the submitter moves to another thread, or inside a frame.
    is_send::<FakeStream>();
    is_send::<FakeGraph>();
    // Not Sync: two threads can never hold `&FakeStream`, so nothing needs a lock.
    let _ = <FakeStream as AmbiguousIfSync<_>>::check;
}

#[test]
fn the_query_side_is_shared() {
    is_send::<FakeQueue>();
    is_sync::<FakeQueue>();
}

#[cfg(feature = "cuda")]
#[test]
fn the_cuda_submitter_is_single_owner_and_its_queue_is_shared() {
    use euhedral_gpu::cuda::{CudaGraph, CudaQueue, CudaStream};
    is_send::<CudaStream>();
    is_send::<CudaGraph>();
    let _ = <CudaStream as AmbiguousIfSync<_>>::check;
    let _ = <CudaGraph as AmbiguousIfSync<_>>::check;
    is_send::<CudaQueue>();
    is_sync::<CudaQueue>();
}
