//! Asynchronous I/O whose completions are frames (Euhedral-Inference `AsyncReads`, extended).
//!
//! [`AsyncIo`] is the one entry point. Its backend is chosen by `cfg`: io_uring on Linux and
//! Android (the default feature `io-uring`). A backend module defines `AsyncIo` with the same
//! methods over the shared buffer and continuation types below, so another platform's backend slots
//! in beside the `uring` module under its own `cfg`.

use std::io;

use crate::Continuation;

/// A buffer whose ownership passes to the kernel for the operation and comes back in its
/// [`IoDone`].
#[derive(Debug, Default)]
pub struct IoBuf {
    pub bytes: Vec<u8>,
    /// For reads and receives: how many bytes arrived (`bytes[..filled]`). For writes and sends:
    /// how many to send from the start.
    pub filled: usize,
}

/// A finished operation: its buffer back, and the bytes moved or the error.
#[derive(Debug)]
pub struct IoDone {
    pub buf: IoBuf,
    pub result: io::Result<usize>,
}

/// What continues an operation once it ended.
pub type IoThen = Continuation<IoDone>;

mod uring;

pub use uring::AsyncIo;
