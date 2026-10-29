//! Asynchronous I/O whose completions are frames.
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
//! The feature `async-io` (default) builds the crate's contents; without it, or on a platform with no
//! engine (macOS yet), the crate is empty.
//!
//! Unsafe code lives here and in the engines, nowhere else in the crate: the handle conversions, and
//! the [`IoMemory`] contract that lets an operation read into memory the caller owns (`// SAFETY:` on
//! every block).
#![allow(unsafe_code)]
#![cfg(all(
    feature = "async-io",
    any(target_os = "linux", target_os = "android", windows)
))]

use std::collections::VecDeque;
use std::fmt;
use std::io;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use euhedral_core::frames::Frame;
use euhedral_core::generics::{LatticeReceiver, LatticeSource};
use euhedral_data_structures::sync::HotScope;

/// What turns a completed operation's result into the frame that continues the work.
pub type Continuation<T> = Box<dyn FnOnce(T) -> Box<dyn Frame> + Send>;

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

/// Memory the caller owns, that [`AsyncIo::read_into`] reads into: a pinned host arena, a mapped
/// region, or any block the kernel may fill. The operation owns the value until it ends, and its
/// continuation gets it back in an [`IoDoneOwned`].
///
/// # Safety
///
/// The engine hands the region to the kernel, which writes it while the operation runs, at times
/// the program cannot observe. So the implementor promises, for the whole time from the call of
/// [`as_mut_ptr`](Self::as_mut_ptr) until the value is dropped or handed back in
/// [`IoDoneOwned::buf`]:
///
/// - the pointer addresses [`len`](Self::len) bytes that are allocated, **initialized**, valid for
///   reads and writes, and not freed, remapped or reused;
/// - **moving the value does not move the region, and does not assert unique ownership of it**
///   (the pointer stays valid after the value moved into the operation): a value that owns its
///   memory through a heap block or a mapping behind a raw pointer, such as `Vec<u8>`, qualifies;
///   a value that holds the bytes inline does not, and neither does one whose move is a unique
///   retag of the region, such as `Box<[u8]>` (under Stacked Borrows a move of a `Box` invalidates
///   the raw pointers derived from it), unless the region is reached through a raw-pointer wrapper;
/// - the region is exclusively the operation's: nothing else, in this process or another, reads or
///   writes it (so a region of a shared arena must be one no other value refers to), and the value
///   itself touches it only through the pointer;
/// - `len` returns the same value on every call.
///
/// The engine calls `as_mut_ptr` and `len` once, when the read is submitted, and promises that the
/// value is dropped or returned only after the kernel can no longer touch the region. When it
/// cannot tell (a ring that broke), it **leaks** the value instead: the value is never dropped, the
/// continuation still runs, and its [`IoDoneOwned::buf`] is `None`.
pub unsafe trait IoMemory: Send + 'static {
    /// The start of the region.
    fn as_mut_ptr(&mut self) -> *mut u8;

    /// The region's length in bytes.
    fn len(&self) -> usize;

    /// Whether the region has no bytes (a read into one is refused).
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// SAFETY: a `Vec`'s elements live in one heap block that stays where it is when the `Vec` moves;
// `len` initialized bytes are valid for reads and writes; the `Vec` owns the block exclusively.
unsafe impl IoMemory for Vec<u8> {
    fn as_mut_ptr(&mut self) -> *mut u8 {
        Vec::as_mut_ptr(self)
    }

    fn len(&self) -> usize {
        Vec::len(self)
    }
}

/// A finished read into caller-owned memory: the memory back, and the bytes moved or the error.
#[derive(Debug)]
pub struct IoDoneOwned<B> {
    /// The memory, whether the read succeeded or failed. `None` only when the engine could not tell
    /// that the kernel was done with it (a ring that broke) and leaked it instead: it is never
    /// dropped, and `result` is then the ring's failure.
    pub buf: Option<B>,
    pub result: io::Result<usize>,
    /// How many bytes at the start of the region arrived: `n` for `Ok(n)`. After an error it is
    /// the bytes that arrived before it, which is non-zero only for a Windows read that failed with
    /// a warning status (`ERROR_MORE_DATA` of a message-mode pipe) and moved bytes anyway.
    pub filled: usize,
}

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
    /// A read into caller-owned memory: the closure holds the [`IoMemory`] value that the
    /// operation's [`Buf::Region`] points into. It is called with the result, how many bytes
    /// arrived, and whether the region was leaked (the closure then forgets the value).
    Owned(OwnedThen),
}

/// What ends a read into caller-owned memory: the result, how many bytes arrived, and whether the
/// region was leaked.
pub(crate) type OwnedThen =
    Box<dyn FnOnce(io::Result<usize>, usize, bool) -> Box<dyn Frame> + Send>;

/// The memory of an operation, as an engine sees it.
pub(crate) enum Buf {
    /// The buffer of [`IoBuf`], owned here.
    Vec(IoBuf),
    /// Caller-owned memory, owned by the operation's [`Then::Owned`].
    Region(Region),
}

impl From<IoBuf> for Buf {
    fn from(buf: IoBuf) -> Self {
        Self::Vec(buf)
    }
}

/// A view of an [`IoMemory`] value's region. The value is held by the operation's continuation, so
/// the region outlives every use the engine makes of it unless the operation is leaked (the
/// continuation is then told, and forgets the value, see [`Buf::leak`]).
pub(crate) struct Region {
    ptr: *mut u8,
    len: usize,
    filled: usize,
    leaked: bool,
}

// SAFETY: the region is exclusively the operation's (the `IoMemory` contract), and an operation is
// used by one thread at a time (or by the kernel).
unsafe impl Send for Region {}

impl Region {
    /// The region of `memory`, which must stay alive (and in place) for as long as the view is
    /// used.
    pub(crate) fn of<B: IoMemory>(memory: &mut B) -> Self {
        Self {
            ptr: memory.as_mut_ptr(),
            len: memory.len(),
            filled: 0,
            leaked: false,
        }
    }
}

impl Buf {
    pub(crate) fn len(&self) -> usize {
        match self {
            Self::Vec(buf) => buf.bytes.len(),
            Self::Region(region) => region.len,
        }
    }

    pub(crate) fn filled(&self) -> usize {
        match self {
            Self::Vec(buf) => buf.filled,
            Self::Region(region) => region.filled,
        }
    }

    pub(crate) fn set_filled(&mut self, filled: usize) {
        match self {
            Self::Vec(buf) => buf.filled = filled,
            Self::Region(region) => region.filled = filled,
        }
    }

    pub(crate) fn bytes(&self) -> &[u8] {
        match self {
            Self::Vec(buf) => &buf.bytes,
            // SAFETY: the `IoMemory` contract: `len` initialized bytes, exclusively the operation's,
            // alive while the operation's continuation holds the value.
            Self::Region(region) => unsafe { std::slice::from_raw_parts(region.ptr, region.len) },
        }
    }

    pub(crate) fn bytes_mut(&mut self) -> &mut [u8] {
        match self {
            Self::Vec(buf) => &mut buf.bytes,
            // SAFETY: as in `bytes`; `&mut self` makes the view unique.
            Self::Region(region) => unsafe {
                std::slice::from_raw_parts_mut(region.ptr, region.len)
            },
        }
    }

    /// The kernel may still write the memory, so it must never be freed or returned: a buffer is
    /// leaked and the operation ends with an empty one; caller-owned memory is marked, and its
    /// continuation is told to leak the value it holds.
    #[cfg_attr(
        not(windows),
        allow(
            dead_code,
            reason = "only the Windows engine ends operations that may still be in the kernel"
        )
    )]
    pub(crate) fn leak(&mut self) {
        match self {
            Self::Vec(buf) => std::mem::forget(std::mem::take(&mut buf.bytes)),
            Self::Region(region) => region.leaked = true,
        }
    }

    /// Whether the continuation of this operation must be leaked rather than run.
    fn is_leaked(&self) -> bool {
        matches!(self, Self::Region(Region { leaked: true, .. }))
    }
}

/// The continuation's frame for an ended operation, the part the engines share. `fills` says the
/// operation was a read or receive, whose success `Ok(n)` is how many bytes arrived.
pub(crate) fn conclude(
    mut buf: Buf,
    then: Then,
    fills: bool,
    result: io::Result<usize>,
) -> Box<dyn Frame> {
    match then {
        Then::Io(then) => {
            if fills && let Ok(n) = result {
                buf.set_filled(n);
            }
            let Buf::Vec(buf) = buf else {
                unreachable!("an IoThen is paired with an IoBuf")
            };
            then(IoDone { buf, result })
        }
        Then::Raw(then) => then(result),
        Then::Owned(then) => {
            if fills && let Ok(n) = result {
                buf.set_filled(n);
            }
            let leaked = buf.is_leaked();
            then(result, buf.filled(), leaked)
        }
    }
}

/// The platform's half of [`AsyncIo`]: it starts operations and finds their ends.
pub(crate) trait Engine: Send + Sync {
    /// Starts the operation and returns at once. On `Err`, nothing was started: `then` and the
    /// buffer are dropped without running.
    fn submit(&self, request: Request, buf: Buf, then: Then) -> io::Result<()>;

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

    fn submit(&self, request: Request, buf: Buf, then: Then) -> io::Result<()> {
        let _hot = HotScope::enter();
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
        self.submit(Request::Read { file, offset }, buf.into(), Then::Io(then))
    }

    /// Reads into memory the caller owns: up to `buf.len()` bytes at `offset` of `file`, straight
    /// into the region `buf` addresses (a pinned arena, say), with no copy. `buf` belongs to the
    /// operation until it ends and comes back in the [`IoDoneOwned`] that `then` receives, whether
    /// the read succeeded or failed.
    ///
    /// The result is as [`read`](Self::read)'s: `Ok(n)` with `n` the region's length, or less if
    /// the file ended after some bytes (the first `n` bytes of the region are the file's); a read
    /// that finds no bytes at all fails with `UnexpectedEof`. The region is cut into pieces as a
    /// `read`'s buffer is, and the pieces land at their own offsets. A submission that returns `Err`
    /// started nothing: `buf` and `then` are dropped without running, and an empty region is
    /// refused. `then` always runs once the read was submitted. If the engine can no longer tell
    /// whether the kernel still writes the region (a ring that broke), it leaks `buf`: `then` runs
    /// with `buf` `None` and the ring's failure as the result (see [`IoMemory`]).
    pub fn read_into<B: IoMemory>(
        &self,
        file: RawFile,
        offset: u64,
        mut buf: B,
        then: Continuation<IoDoneOwned<B>>,
    ) -> io::Result<()> {
        if buf.is_empty() {
            return Err(invalid("a read into an empty region"));
        }
        let region = Region::of(&mut buf);
        let file = raw_file(file);
        // The closure keeps `buf` alive and in place until the operation ends.
        let then = Then::Owned(Box::new(move |result, filled, leaked| {
            let buf = if leaked {
                // The kernel may still write the region: the value that owns it is never dropped.
                std::mem::forget(buf);
                None
            } else {
                Some(buf)
            };
            then(IoDoneOwned {
                buf,
                result,
                filled,
            })
        }));
        self.submit(Request::Read { file, offset }, Buf::Region(region), then)
    }

    /// Writes `buf.bytes[..buf.filled]` at `offset` of `file`.
    pub fn write(&self, file: RawFile, offset: u64, buf: IoBuf, then: IoThen) -> io::Result<()> {
        if buf.filled == 0 || buf.filled > buf.bytes.len() {
            return Err(invalid("a write needs 1..=bytes.len() bytes to write"));
        }
        let file = raw_file(file);
        self.submit(Request::Write { file, offset }, buf.into(), Then::Io(then))
    }

    /// Sends `buf.bytes[..buf.filled]` on a connected socket.
    pub fn send(&self, socket: RawSock, buf: IoBuf, then: IoThen) -> io::Result<()> {
        if buf.filled > buf.bytes.len() {
            return Err(invalid("filled is beyond the buffer"));
        }
        let sock = raw_sock(socket);
        self.submit(Request::Send { sock }, buf.into(), Then::Io(then))
    }

    /// Receives into `buf.bytes` from a connected socket; `Ok(0)` is the peer's end of stream.
    pub fn recv(&self, socket: RawSock, buf: IoBuf, then: IoThen) -> io::Result<()> {
        let sock = raw_sock(socket);
        self.submit(Request::Recv { sock }, buf.into(), Then::Io(then))
    }

    /// Accepts one connection on a listening socket.
    pub fn accept(
        &self,
        listener: RawSock,
        then: Continuation<io::Result<OwnedSock>>,
    ) -> io::Result<()> {
        let listener = raw_sock(listener);
        let then = Then::Raw(Box::new(move |result| then(result.map(owned_sock))));
        self.submit(Request::Accept { listener }, IoBuf::default().into(), then)
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
        self.submit(request, IoBuf::default().into(), then)
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
        self.submit(request, IoBuf::default().into(), then)
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
        self.submit(request, IoBuf::default().into(), then)
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
        self.submit(Request::SendTo { sock, to }, buf.into(), Then::Io(then))
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
        self.submit(Request::Connect { sock, to }, IoBuf::default().into(), then)
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
