//! Asynchronous I/O whose completions are frames (Euhedral-Inference `AsyncReads`, extended).
//!
//! [`AsyncIo`] is the one entry point, and this file is its portable half: the public API, the
//! buffer and continuation types, and the `LatticeSource` behaviour (`pending`, demand, the
//! downstream, `complete`, and the frames a pull's `stop` holds back). The platform's engine only
//! submits an operation and reports its end:
//!
//! | Platform | Engine | Module |
//! |---|---|---|
//! | Linux, Android | io_uring | `uring` |
//! | Windows | IoRing for file reads and writes where the OS has a kernel ring, I/O completion ports for everything else | `win` |
//!
//! The feature `async-io` (default) builds this module. macOS has no engine yet.

use std::collections::VecDeque;
use std::fmt;
use std::io;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use euhedral_core::frames::Frame;
use euhedral_core::generics::{LatticeReceiver, LatticeSource};

use crate::Continuation;

#[cfg(any(target_os = "linux", target_os = "android"))]
mod uring;
// Windows runs the engine; every other platform compiles it for its simulated tests.
#[cfg(any(windows, test))]
mod win;

/// The platform's file handle, borrowed: what `AsyncIo` reads, writes and renames through.
#[cfg(unix)]
pub type RawFile = std::os::fd::RawFd;
/// The platform's socket handle, borrowed.
#[cfg(unix)]
pub type RawSock = std::os::fd::RawFd;
/// The platform's file handle, owned: what [`AsyncIo::open_path`] returns.
#[cfg(unix)]
pub type OwnedFile = std::os::fd::OwnedFd;
/// The platform's socket handle, owned: what [`AsyncIo::accept`] returns.
#[cfg(unix)]
pub type OwnedSock = std::os::fd::OwnedFd;
/// The platform's file handle, borrowed: what `AsyncIo` reads, writes and renames through.
#[cfg(windows)]
pub type RawFile = std::os::windows::io::RawHandle;
/// The platform's socket handle, borrowed.
#[cfg(windows)]
pub type RawSock = std::os::windows::io::RawSocket;
/// The platform's file handle, owned: what [`AsyncIo::open_path`] returns.
#[cfg(windows)]
pub type OwnedFile = std::os::windows::io::OwnedHandle;
/// The platform's socket handle, owned: what [`AsyncIo::accept`] returns.
#[cfg(windows)]
pub type OwnedSock = std::os::windows::io::OwnedSocket;

/// A buffer whose ownership passes to the kernel for the operation and comes back in its
/// [`IoDone`].
#[derive(Debug, Default)]
pub struct IoBuf {
    pub bytes: Vec<u8>,
    /// For reads and receives: how many bytes arrived (`bytes[..filled]`). For writes and sends:
    /// how many to send from the start. On Windows a read or receive that fails with a warning
    /// status (`ERROR_MORE_DATA` of a message-mode pipe, `WSAEMSGSIZE` of a datagram that did not
    /// fit) still moved bytes: the result is that error, and `filled` says how many arrived.
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

/// What [`AsyncIo::open_path`] does about a file that exists or is missing.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Create {
    /// The file must exist.
    #[default]
    Never,
    /// Create it if it is missing; open it if it exists.
    IfMissing,
    /// Create it; fail if it exists.
    New,
    /// Create it if it is missing; empty it if it exists.
    Truncate,
}

/// How [`AsyncIo::open_path`] opens a file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OpenSpec {
    /// Open for reading.
    pub read: bool,
    /// Open for writing.
    pub write: bool,
    /// What to do about a missing or existing file.
    pub create: Create,
    /// The permissions of a created file (Unix only; ignored on Windows).
    pub unix_mode: u32,
}

impl Default for OpenSpec {
    /// An existing file, for reading.
    fn default() -> Self {
        Self {
            read: true,
            write: false,
            create: Create::Never,
            unix_mode: 0o644,
        }
    }
}

/// How [`AsyncIo::open_with_options`] chooses its engine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AsyncIoOptions {
    /// On Windows, use IoRing for file reads and writes where the OS has a kernel ring (Windows 11 22H2 and
    /// later, which has file reads and writes); `false` forces completion ports everywhere. Ignored on other platforms.
    pub windows_ioring: bool,
}

impl Default for AsyncIoOptions {
    fn default() -> Self {
        Self {
            windows_ioring: true,
        }
    }
}

/// The most bytes one read, write or send piece moves (see the engines' read pieces).
pub(crate) const READ_PIECE: usize = 256 * 1024;
/// The longest piece of a read or write handed to the kernel at once; the rest follows as a short
/// one's would.
pub(crate) const MAX_PIECE: usize = (i32::MAX as usize) & !4095;

/// What an operation asks of an engine. Handles are the platform's raw values (a descriptor, a
/// handle or a socket) as a `usize`.
pub(crate) enum Request {
    Read {
        file: usize,
        offset: u64,
    },
    Write {
        file: usize,
        offset: u64,
    },
    Send {
        sock: usize,
    },
    Recv {
        sock: usize,
    },
    Accept {
        listener: usize,
    },
    SendTo {
        sock: usize,
        to: std::net::SocketAddr,
    },
    Connect {
        sock: usize,
        to: std::net::SocketAddr,
    },
    Open {
        path: PathBuf,
        spec: OpenSpec,
    },
    Rename {
        from: PathBuf,
        to: PathBuf,
    },
    /// `open(2)` with raw flags (Linux and Android).
    #[cfg(any(target_os = "linux", target_os = "android"))]
    OpenAt {
        path: PathBuf,
        flags: i32,
        mode: u32,
    },
}

/// What continues an operation, as an engine sees it.
pub(crate) enum Then {
    /// Reads, writes, sends and receives: the buffer and the bytes moved.
    Io(IoThen),
    /// Accept and open (the result is a new handle's raw value), rename and connect (the value is
    /// unused).
    Raw(Box<dyn FnOnce(io::Result<usize>) -> Box<dyn Frame> + Send>),
}

/// The platform's half of [`AsyncIo`]: it starts operations and finds their ends.
pub(crate) trait Engine: Send + Sync {
    /// Starts the operation and returns at once. On `Err`, nothing was started: `then` and the
    /// buffer are dropped without running.
    fn submit(&self, request: Request, buf: IoBuf, then: Then) -> io::Result<()>;

    /// Finds ended operations. Each is given to `offer` as its continuation's frame while `room`
    /// says the caller can take more; once `offer` returns `false` (it kept the frame), the engine
    /// stops without reaping anything else.
    fn poll(&self, room: &mut dyn FnMut() -> bool, offer: &mut dyn FnMut(Box<dyn Frame>) -> bool);
}

#[cfg(any(target_os = "linux", target_os = "android"))]
type Platform = uring::Uring;
#[cfg(windows)]
type Platform = win::Engine;

/// File and socket I/O that holds no worker: a submission returns at once, and the operation's
/// completion is the frame its continuation builds, emitted when the lattice's workers poll this
/// source.
///
/// The engine is io_uring on Linux and Android. On Windows it is IoRing for file reads and writes
/// where the OS has a kernel ring, and I/O completion ports for everything else (sockets, accept,
/// connect, and files without a ring); open and rename run on the OS thread pool, the one place
/// this crate uses threads it did not create, and post their result to the completion port. Every
/// submitting worker has its own submission queue where the engine has queues (it is the queue's
/// only submitter); completions are reaped by the lattice's polls, one poll at a time. A read or
/// write is complete when all its bytes moved (a short one is resubmitted for the rest by the poll
/// that reaped it) or it failed. One send per connection is in flight at a time (the caller's
/// ordered lane ensures it), so bytes keep their order.
///
/// - **A read** completes with `Ok(n)`: `n` is the buffer's length, or less if the file ended
///   after some bytes (`filled` = `n`); a read that finds no bytes at all (at or past the end)
///   fails with `UnexpectedEof`.
/// - **A write** completes with `Ok(filled)`, or fails (`WriteZero` if the kernel wrote nothing).
/// - **A send** behaves like a write: a short send is resubmitted for the rest, and it completes
///   with `Ok(filled)` once every byte is in the kernel, or with the error that stopped it (bytes
///   sent before the error are not reported). On Linux it never raises `SIGPIPE`: a closed peer is
///   an `EPIPE` or `ECONNRESET` error.
/// - **A receive** completes with what one operation moved: `Ok(n)`, `filled` = `n`; `Ok(0)` is
///   the peer's end of stream.
/// - **A datagram** (`send_to`) is one operation.
/// - **A submission** returns `Err` only when nothing was submitted (an empty read or write, a
///   completed source, too many submitting threads, a handle that would block the worker);
///   `then` and the buffer are then dropped without running. When the kernel's queue is full or
///   the kernel refuses an enter for a while, the operation stays queued and is handed over by a
///   later submission or poll: it still completes, once.
/// - **An operation belongs to the thread that submitted it:** io_uring cancels it (`ECANCELED`, or
///   `EINTR`) if that thread exits first. Lattice workers live as long as the lattice. On
///   Windows an operation is not cancelled when its thread exits (a ring is not tied to a thread,
///   and a handle joined to a completion port keeps its I/O), but rely only on the contract above.
/// - **Handles on Windows** must be opened for overlapped I/O (`FILE_FLAG_OVERLAPPED`: [`open_path`]
///   does, and Rust's sockets are) and must not skip the port on success
///   (`FILE_SKIP_COMPLETION_PORT_ON_SUCCESS`): a handle of either kind is refused with
///   `InvalidInput`, since its operations would block the worker or never complete. A handle joins
///   this source's completion port when its first operation on the port is submitted, and a handle
///   can belong to one port only: **it can serve one `AsyncIo`, ever**, and a handle that already
///   belongs to another port is refused. File reads and writes through IoRing need no port.
/// - **Sockets on Windows** must belong to a provider that fills `OVERLAPPED`'s `Internal` and
///   `InternalHigh` on completion, as the system's own TCP and UDP do; a socket of a legacy,
///   non-IFS layered provider may not, and is not supported.
/// - **Writes that extend a file** are synchronous on NTFS (the call returns when the data is
///   written). On Windows every write goes in pieces of at most 256 KiB, so that one piece is a
///   bounded wait for the worker that submits it.
/// - **Dropping** an `AsyncIo` with operations in flight leaks their buffers (the kernel may still
///   write them).
///
/// **Driving it by hand** (tests, or any caller outside a lattice): `pull(consumer, &mut |_|
/// false, u64::MAX)` reaps on its own; no `request` is needed before it (a `request` only pushes
/// to an attached downstream). During the pull, each ended operation's continuation is called and
/// its frame handed to `consumer`; `AsyncIo` never runs a frame. The consumer decides when the
/// frame runs: a lattice queues it for a worker (so it runs after the pull returns), and a test
/// may run it inside the consumer or after `pull` returns. A pull finds only what completed by
/// then: poll again until the completions you expect came out.
///
/// A pull stops before the first continuation its `stop` matches; that one (and what is reaped
/// after it) goes out on a later poll, in order. The lattice's direct pulls stop before ordered
/// frames, which so reach their ordered lane through `request`'s pushes and the lattice's routing,
/// in the order they were reaped.
///
/// [`open_path`]: AsyncIo::open_path
pub struct AsyncIo {
    front: Front,
    engine: Platform,
}

/// The portable state of the `LatticeSource`.
pub(crate) struct Front {
    /// Continuations reaped while a pull's `stop` matched them, the oldest first; only the polling
    /// thread touches it.
    held: Mutex<VecDeque<Box<dyn Frame>>>,
    pending: AtomicUsize,
    demand: AtomicU64,
    downstream: Mutex<Option<Arc<dyn LatticeReceiver>>>,
    complete: AtomicBool,
}

impl Front {
    pub(crate) fn new() -> Self {
        Self {
            held: Mutex::new(VecDeque::new()),
            pending: AtomicUsize::new(0),
            demand: AtomicU64::new(0),
            downstream: Mutex::new(None),
            complete: AtomicBool::new(false),
        }
    }

    pub(crate) fn pending(&self) -> usize {
        self.pending.load(Ordering::Acquire)
    }

    /// One poll: holds back or hands up to `demand` continuations' frames to `consumer`.
    pub(crate) fn reap(
        &self,
        engine: &dyn Engine,
        consumer: &mut dyn FnMut(Box<dyn Frame>),
        stop: &mut dyn FnMut(&dyn Frame) -> bool,
        demand: u64,
    ) -> u64 {
        if demand == 0 || self.pending() == 0 {
            return 0;
        }
        let mut emitted = 0u64;
        // Continuations an earlier pull's `stop` held back go first, in the order they were reaped.
        let mut held = lock(&self.held);
        while emitted < demand {
            let Some(frame) = held.front() else { break };
            if stop(&**frame) {
                return emitted;
            }
            let frame = held.pop_front().expect("the front");
            self.pending.fetch_sub(1, Ordering::AcqRel);
            consumer(frame);
            emitted += 1;
        }
        let emitted = std::cell::Cell::new(emitted);
        engine.poll(&mut || emitted.get() < demand, &mut |frame| {
            if stop(&*frame) {
                // The lattice's direct pulls stop before ordered frames: this one goes out
                // through a later poll (the lattice's routing), and nothing reaped after it
                // passes it.
                held.push_back(frame);
                return false;
            }
            self.pending.fetch_sub(1, Ordering::AcqRel);
            consumer(frame);
            emitted.set(emitted.get() + 1);
            true
        });
        emitted.get()
    }
}

impl fmt::Debug for AsyncIo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AsyncIo")
            .field("pending", &self.pending())
            .finish_non_exhaustive()
    }
}

pub(crate) fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

fn invalid(what: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, what)
}

/// The platform's descriptor or handle as the engines' `usize`.
fn raw_file(file: RawFile) -> usize {
    file as usize
}

fn raw_sock(sock: RawSock) -> usize {
    sock as usize
}

/// A new file the engine opened for an operation, from its raw value.
#[cfg(unix)]
#[allow(unsafe_code)]
fn owned_file(raw: usize) -> OwnedFile {
    use std::os::fd::FromRawFd;
    // SAFETY: an open's or accept's result is a new descriptor the kernel opened for this
    // operation: no one else owns it.
    unsafe { OwnedFile::from_raw_fd(raw as std::os::fd::RawFd) }
}
#[cfg(windows)]
#[allow(unsafe_code)]
fn owned_file(raw: usize) -> OwnedFile {
    use std::os::windows::io::FromRawHandle;
    // SAFETY: the handle was just opened for this operation: no one else owns it.
    unsafe { OwnedFile::from_raw_handle(raw as std::os::windows::io::RawHandle) }
}
#[cfg(unix)]
fn owned_sock(raw: usize) -> OwnedSock {
    owned_file(raw)
}
#[cfg(windows)]
#[allow(unsafe_code)]
fn owned_sock(raw: usize) -> OwnedSock {
    use std::os::windows::io::FromRawSocket;
    // SAFETY: the socket was just made for this accept: no one else owns it.
    unsafe { OwnedSock::from_raw_socket(raw as std::os::windows::io::RawSocket) }
}

impl AsyncIo {
    fn from_engine(engine: Platform) -> Arc<Self> {
        Arc::new(Self {
            front: Front::new(),
            engine,
        })
    }

    fn submit(&self, request: Request, buf: IoBuf, then: Then) -> io::Result<()> {
        if self.front.complete.load(Ordering::Acquire) {
            return Err(io::Error::other("this AsyncIo is complete"));
        }
        // Counted before the engine may see the operation: a poll then never finds an end that is
        // not pending.
        self.front.pending.fetch_add(1, Ordering::AcqRel);
        let started = self.engine.submit(request, buf, then);
        if started.is_err() {
            self.front.pending.fetch_sub(1, Ordering::AcqRel);
        }
        started
    }

    /// Reads up to `buf.bytes.len()` bytes at `offset` of `file`.
    pub fn read(&self, file: RawFile, offset: u64, buf: IoBuf, then: IoThen) -> io::Result<()> {
        if buf.bytes.is_empty() {
            return Err(invalid("a read into an empty buffer"));
        }
        let file = raw_file(file);
        self.submit(Request::Read { file, offset }, buf, Then::Io(then))
    }

    /// Writes `buf.bytes[..buf.filled]` at `offset` of `file`.
    pub fn write(&self, file: RawFile, offset: u64, buf: IoBuf, then: IoThen) -> io::Result<()> {
        if buf.filled == 0 || buf.filled > buf.bytes.len() {
            return Err(invalid("a write needs 1..=bytes.len() bytes to write"));
        }
        let file = raw_file(file);
        self.submit(Request::Write { file, offset }, buf, Then::Io(then))
    }

    /// Sends `buf.bytes[..buf.filled]` on a connected socket.
    pub fn send(&self, socket: RawSock, buf: IoBuf, then: IoThen) -> io::Result<()> {
        if buf.filled > buf.bytes.len() {
            return Err(invalid("filled is beyond the buffer"));
        }
        let sock = raw_sock(socket);
        self.submit(Request::Send { sock }, buf, Then::Io(then))
    }

    /// Receives into `buf.bytes` from a connected socket; `Ok(0)` is the peer's end of stream.
    pub fn recv(&self, socket: RawSock, buf: IoBuf, then: IoThen) -> io::Result<()> {
        let sock = raw_sock(socket);
        self.submit(Request::Recv { sock }, buf, Then::Io(then))
    }

    /// Accepts one connection on a listening socket.
    pub fn accept(
        &self,
        listener: RawSock,
        then: Continuation<io::Result<OwnedSock>>,
    ) -> io::Result<()> {
        let listener = raw_sock(listener);
        let then = Then::Raw(Box::new(move |result| then(result.map(owned_sock))));
        self.submit(Request::Accept { listener }, IoBuf::default(), then)
    }

    /// Opens `path` as `spec` says. The file is opened for overlapped I/O on Windows. The open
    /// itself runs on the kernel's workers (io_uring) or the OS thread pool (Windows), not on the
    /// submitting worker.
    pub fn open_path(
        &self,
        path: &std::path::Path,
        spec: OpenSpec,
        then: Continuation<io::Result<OwnedFile>>,
    ) -> io::Result<()> {
        if !spec.read && !spec.write {
            return Err(invalid("a file is opened for reading, writing or both"));
        }
        let request = Request::Open {
            path: path.to_owned(),
            spec,
        };
        let then = Then::Raw(Box::new(move |result| then(result.map(owned_file))));
        self.submit(request, IoBuf::default(), then)
    }

    /// Opens `path` with `open(2)` `flags` (`O_CLOEXEC` is always added) and `mode` for a created
    /// file: for example `O_WRONLY | O_CREAT | O_TRUNC` and `0o644` for the temporary file of an
    /// atomic save.
    #[cfg(any(target_os = "linux", target_os = "android"))]
    pub fn open_file(
        &self,
        path: &std::path::Path,
        flags: i32,
        mode: u32,
        then: Continuation<io::Result<OwnedFile>>,
    ) -> io::Result<()> {
        let request = Request::OpenAt {
            path: path.to_owned(),
            flags,
            mode,
        };
        let then = Then::Raw(Box::new(move |result| then(result.map(owned_file))));
        self.submit(request, IoBuf::default(), then)
    }

    /// Renames `from` to `to`, replacing `to` atomically if it exists (`renameat(2)`; on Windows
    /// `MoveFileExW` with `MOVEFILE_REPLACE_EXISTING`): the last step of an atomic save.
    pub fn rename(
        &self,
        from: &std::path::Path,
        to: &std::path::Path,
        then: Continuation<io::Result<()>>,
    ) -> io::Result<()> {
        let request = Request::Rename {
            from: from.to_owned(),
            to: to.to_owned(),
        };
        let then = Then::Raw(Box::new(move |result| then(result.map(|_| ()))));
        self.submit(request, IoBuf::default(), then)
    }

    /// Sends one datagram to `to`.
    pub fn send_to(
        &self,
        socket: RawSock,
        to: std::net::SocketAddr,
        buf: IoBuf,
        then: IoThen,
    ) -> io::Result<()> {
        if buf.filled > buf.bytes.len() {
            return Err(invalid("filled is beyond the buffer"));
        }
        let sock = raw_sock(socket);
        self.submit(Request::SendTo { sock, to }, buf, Then::Io(then))
    }

    /// Connects a stream socket (made by the caller, blocking or not: the engine waits for the
    /// handshake either way) to `to`; `Ok(())` once connected. A client's connection, for instance
    /// to a service on the same host.
    pub fn connect(
        &self,
        socket: RawSock,
        to: std::net::SocketAddr,
        then: Continuation<io::Result<()>>,
    ) -> io::Result<()> {
        let sock = raw_sock(socket);
        let then = Then::Raw(Box::new(move |result| then(result.map(|_| ()))));
        self.submit(Request::Connect { sock, to }, IoBuf::default(), then)
    }

    /// Operations in flight.
    pub fn pending(&self) -> usize {
        self.front.pending()
    }
}

impl LatticeSource for AsyncIo {
    fn add_downstream(self: Arc<Self>, downstream: Arc<dyn LatticeReceiver>) {
        if self.front.complete.load(Ordering::Acquire) {
            downstream.on_complete();
            return;
        }
        *lock(&self.front.downstream) = Some(downstream.clone());
        downstream.add_upstream(self);
    }

    fn pull(
        &self,
        consumer: &mut dyn FnMut(Box<dyn Frame>),
        stop: &mut dyn FnMut(&dyn Frame) -> bool,
        demand: u64,
    ) -> u64 {
        self.front.reap(&self.engine, consumer, stop, demand)
    }

    fn request(&self, demand: u64) {
        if demand == 0 || self.front.complete.load(Ordering::Acquire) {
            return;
        }
        let total = self
            .front
            .demand
            .fetch_add(demand, Ordering::AcqRel)
            .saturating_add(demand);
        let Some(downstream) = lock(&self.front.downstream).clone() else {
            return;
        };
        let emitted = self.front.reap(
            &self.engine,
            &mut |frame| downstream.push(frame),
            &mut |_| false,
            total,
        );
        if emitted > 0 {
            let _ = self
                .front
                .demand
                .try_update(Ordering::AcqRel, Ordering::Acquire, |d| {
                    Some(d.saturating_sub(emitted))
                });
        }
    }

    /// Stops new submissions (they fail) and tells the downstream. Complete it once
    /// [`AsyncIo::pending`] is zero.
    fn complete(&self) {
        if !self.front.complete.swap(true, Ordering::AcqRel)
            && let Some(downstream) = lock(&self.front.downstream).take()
        {
            downstream.on_complete();
        }
    }

    fn is_complete(&self) -> bool {
        self.front.complete.load(Ordering::Acquire)
    }
}
