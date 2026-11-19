//! The positional backend of [`AsyncIo`], for Linux and Android where io_uring is refused.
//!
//! It owns no thread and waits for nothing. A submission only queues the operation; the work is done by the
//! reap frame that the `AsyncIo` pushes into the lattice, inside [`Engine::poll`]: each operation takes one
//! step per visit (one piece of a read or write, one non-blocking attempt of a socket call, or the whole of an
//! open, a rename or a length), and the operations that did not end go to the back of the line. The operations
//! that wait for a descriptor (sockets, accept, pipes and the like) are asked about together: one `poll(2)` call
//! with no timeout per visit over their descriptors, and only the ones it reports ready are attempted. A visit
//! walks the line once and makes at most [`ATTEMPTS_PER_POLL`] attempts, so nothing waits and what is idle costs
//! one system call between all of it.
//!
//! Everything here is plain state of the one poll that is inside (`SourceCell`); only the queue of messages
//! from the submitting and cancelling threads is shared.
#![allow(unsafe_code)]

use std::cell::Cell;
use std::collections::VecDeque;
use std::ffi::CString;
use std::io;
use std::os::fd::RawFd;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use euhedral_core::frames::Frame;
use euhedral_core::generics::SourceCell;
use euhedral_data_structures::queues::{Consumer, MpscQueue, Multi, Producer, Single};

use super::uring::{c_path, open_flags, socket_address};
use super::{
    Buf, Cancelled, Engine, LimitPassed, MAX_PIECE, OpId, READ_PIECE, Request, Then,
    conclude as conclude_io,
};

/// Slots per chunk of the queue of messages.
const INCOMING_CHUNK: usize = 256;
/// The most attempts one poll makes, so that a reap is bounded work however much is queued. Every attempt counts,
/// one that finds its descriptor not ready after all included; an operation that readiness says is not ready is
/// not attempted and costs nothing.
const ATTEMPTS_PER_POLL: usize = 64;
/// The most pieces of one read or write a poll moves in a row before the next operation's turn, while other
/// operations are also due. An operation that is the only one due in a visit goes on until the visit's attempt
/// budget is used.
const PIECES_PER_TURN: usize = 4;
/// The most visits an open that cannot proceed yet (a FIFO with no reader, a leased file) lets pass between two
/// of its attempts. The wait doubles from one visit up to this.
const OPEN_RETRY_VISITS_MAX: u32 = 64;
/// Marks this engine's ids, so that an id of the io_uring engine is never taken for one of this.
const ID_MARK: u64 = 1 << 62;

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

/// The positional engine.
pub(super) struct Positional {
    id: u64,
    /// The next operation's token.
    next: AtomicU64,
    incoming: Producer<Message, Multi>,
    state: SourceCell<State>,
    /// The most bytes one step of a read, write or send moves; a test lowers it to force short ones.
    max_piece: usize,
    /// How many `poll(2)` readiness calls the visits made (tests).
    #[cfg(test)]
    pub(super) readiness_calls: std::sync::atomic::AtomicUsize,
    /// The descriptors the last readiness call was given.
    #[cfg(test)]
    pub(super) entries: std::sync::atomic::AtomicUsize,
    /// How many attempts (steps) the visits made (tests).
    #[cfg(test)]
    pub(super) attempts: std::sync::atomic::AtomicUsize,
}

/// What only the poll that is inside touches.
struct State {
    incoming: Consumer<Message, Single>,
    /// The operations that are not done, the oldest first.
    active: VecDeque<Op>,
    /// The descriptors the waiting operations ask about, reused by every visit.
    fds: Vec<libc::pollfd>,
    /// Each waiting operation as (descriptor, events, index in `active`), sorted by descriptor so that `fds` holds
    /// one entry per distinct descriptor.
    slots: Vec<(RawFd, i16, usize)>,
}

enum Message {
    Submit(Box<Op>),
    Cancel(u64),
}

enum Kind {
    Read {
        fd: RawFd,
        offset: u64,
    },
    Write {
        fd: RawFd,
        offset: u64,
    },
    Send {
        fd: RawFd,
    },
    Recv {
        fd: RawFd,
    },
    Accept {
        fd: RawFd,
    },
    Open {
        path: CString,
        flags: i32,
        mode: libc::mode_t,
    },
    Rename {
        from: CString,
        to: CString,
    },
    FileLen {
        fd: RawFd,
    },
    SendTo {
        fd: RawFd,
        addr: Box<libc::sockaddr_storage>,
        len: libc::socklen_t,
    },
    Connect {
        fd: RawFd,
        addr: Box<libc::sockaddr_storage>,
        len: libc::socklen_t,
        state: Connecting,
    },
}

/// Where a connect is.
enum Connecting {
    /// Not started.
    New,
    /// The handshake is under way.
    Under(Restore),
}

/// The flags of a socket the engine made non-blocking, put back when the operation ends (explicitly, or when it is
/// dropped with the connect still under way). Nothing to put back for a socket that was non-blocking already.
pub(super) struct Restore {
    fd: RawFd,
    flags: Option<i32>,
}

impl Restore {
    #[cfg(test)]
    pub(super) fn new(fd: RawFd, flags: Option<i32>) -> Self {
        Self { fd, flags }
    }

    fn restore(&mut self) {
        if let Some(flags) = self.flags.take() {
            // SAFETY: a plain `fcntl` of the caller's socket, which stays open until the continuation runs.
            unsafe { libc::fcntl(self.fd, libc::F_SETFL, flags) };
        }
    }
}

impl Drop for Restore {
    fn drop(&mut self) {
        self.restore();
    }
}

struct Op {
    token: u64,
    kind: Kind,
    buf: Buf,
    /// Bytes moved so far (reads, writes and sends go in pieces).
    moved: usize,
    then: Then,
    /// The time limit, as an instant.
    limit: Option<Instant>,
    cancelled: bool,
    /// A read or write that found its descriptor is not seekable (a pipe, a terminal): it goes on as a stream,
    /// waiting for readiness and calling `read` and `write`.
    stream: bool,
    /// This visit: the operation asked `poll(2)` about its descriptor, and what it answered.
    waiting: bool,
    ready: bool,
    /// An open that could not proceed: how many visits pass before it is attempted again, and the wait after
    /// that one (doubling to [`OPEN_RETRY_VISITS_MAX`]).
    skip: u32,
    next_skip: u32,
}

/// What one step of an operation came to.
enum Step {
    Done(io::Result<usize>),
    /// Moved forward, and has more to do.
    Again,
    /// Could not move: the descriptor is not ready, or the open has no peer yet.
    Idle,
}

fn last_error() -> io::Error {
    io::Error::last_os_error()
}

fn would_block(error: &io::Error) -> bool {
    let code = error.raw_os_error();
    code == Some(libc::EAGAIN) || code == Some(libc::EWOULDBLOCK)
}

fn interrupted(error: &io::Error) -> bool {
    error.raw_os_error() == Some(libc::EINTR)
}

/// What a call that returned `n` came to.
fn counted(n: isize) -> Result<usize, io::Error> {
    if n < 0 {
        Err(last_error())
    } else {
        Ok(n as usize)
    }
}

/// The most a write to a blocking stream moves in one call: `poll(2)` promises that much room on a pipe.
const PIPE_BUF: usize = libc::PIPE_BUF;

/// The descriptor's flags, read fresh: they belong to the open file description, so another holder may change them
/// at any time and nothing is cached.
fn flags_of(fd: RawFd) -> io::Result<i32> {
    // SAFETY: a plain `fcntl` of the caller's descriptor.
    let flags = unsafe { libc::fcntl(fd, libc::F_GETFL) };
    if flags < 0 {
        Err(last_error())
    } else {
        Ok(flags)
    }
}

/// Runs `call` (a system call that returns a count or a negative number) with `fd` non-blocking, and puts the
/// flags back right after, on every path. The flags are read just before the call. The error number is taken
/// before the flags are put back. The flags belong to the open file description, so another holder of it (a
/// listener shared with a blocking accepter) may see `EAGAIN` during the call.
fn nonblocking(fd: RawFd, call: impl FnOnce() -> isize) -> io::Result<usize> {
    let flags = flags_of(fd)?;
    if flags & libc::O_NONBLOCK != 0 {
        return counted(call());
    }
    // SAFETY: as in `flags_of`.
    if unsafe { libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK) } < 0 {
        return Err(last_error());
    }
    let result = counted(call());
    // SAFETY: as above.
    unsafe { libc::fcntl(fd, libc::F_SETFL, flags) };
    result
}

impl Op {
    /// The descriptor and the events this operation waits for before it is attempted, if it waits for any. One
    /// that was cancelled or whose limit passed does not wait: its next step ends it.
    fn waits_for(&self, now: Instant) -> Option<(RawFd, i16)> {
        if self.cancelled || self.limit.is_some_and(|at| now >= at) {
            return None;
        }
        match &self.kind {
            Kind::Read { fd, .. } if self.stream => Some((*fd, libc::POLLIN)),
            Kind::Write { fd, .. } if self.stream => Some((*fd, libc::POLLOUT)),
            Kind::Recv { fd } | Kind::Accept { fd } => Some((*fd, libc::POLLIN)),
            Kind::Send { fd } | Kind::SendTo { fd, .. } => Some((*fd, libc::POLLOUT)),
            Kind::Connect {
                fd,
                state: Connecting::Under(_),
                ..
            } => Some((*fd, libc::POLLOUT)),
            _ => None,
        }
    }

    /// One step: ends early for a cancel or a passed limit, else does the operation's next piece or attempt.
    fn step(&mut self, max_piece: usize) -> Step {
        if self.cancelled {
            return Step::Done(Err(Cancelled::error(self.moved)));
        }
        if self.limit.is_some_and(|at| Instant::now() >= at) {
            return Step::Done(Err(LimitPassed::error(self.moved)));
        }
        let piece = max_piece.min(READ_PIECE);
        match &mut self.kind {
            Kind::Read { fd, offset } => {
                let rest = &mut self.buf.bytes_mut()[self.moved..];
                let len = rest.len().min(piece);
                let ptr = rest.as_mut_ptr().cast();
                let n = if self.stream {
                    // SAFETY: `rest` is a live, writable region of at least `len` bytes, and nothing else
                    // touches it while the operation is here.
                    nonblocking(*fd, || unsafe { libc::read(*fd, ptr, len) })
                } else {
                    let at = offset.wrapping_add(self.moved as u64) as libc::off_t;
                    // SAFETY: as above.
                    counted(unsafe { libc::pread(*fd, ptr, len, at) })
                };
                match n {
                    Err(error) if error.raw_os_error() == Some(libc::ESPIPE) => {
                        self.stream = true;
                        Step::Idle
                    }
                    Err(error) if interrupted(&error) || (self.stream && would_block(&error)) => {
                        Step::Idle
                    }
                    Err(error) => Step::Done(Err(error)),
                    Ok(0) if self.moved == 0 => Step::Done(Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "the read is at or past the end of the file",
                    ))),
                    Ok(0) => Step::Done(Ok(self.moved)),
                    Ok(n) => {
                        self.moved += n;
                        if self.moved < self.buf.len() {
                            Step::Again
                        } else {
                            Step::Done(Ok(self.moved))
                        }
                    }
                }
            }
            Kind::Write { fd, offset } => {
                let rest = &self.buf.bytes()[self.moved..self.buf.filled()];
                let mut len = rest.len().min(piece);
                let ptr = rest.as_ptr().cast();
                let n = if self.stream {
                    // The flag is never toggled for a write (the descriptor may be a shared stdout): a blocking
                    // descriptor gets at most what `poll(2)` promised room for, a non-blocking one the whole piece.
                    match flags_of(*fd) {
                        Ok(flags) => {
                            if flags & libc::O_NONBLOCK == 0 {
                                len = len.min(PIPE_BUF);
                            }
                            // SAFETY: `rest` is a live, readable region of at least `len` bytes.
                            counted(unsafe { libc::write(*fd, ptr, len) })
                        }
                        Err(error) => Err(error),
                    }
                } else {
                    let at = offset.wrapping_add(self.moved as u64) as libc::off_t;
                    // SAFETY: `rest` is a live, readable region of at least `len` bytes.
                    counted(unsafe { libc::pwrite(*fd, ptr, len, at) })
                };
                match n {
                    Err(error) if error.raw_os_error() == Some(libc::ESPIPE) => {
                        self.stream = true;
                        Step::Idle
                    }
                    Err(error) if interrupted(&error) || (self.stream && would_block(&error)) => {
                        Step::Idle
                    }
                    Err(error) => Step::Done(Err(error)),
                    Ok(0) => Step::Done(Err(io::Error::from(io::ErrorKind::WriteZero))),
                    Ok(n) => {
                        self.moved += n;
                        if self.moved < self.buf.filled() {
                            Step::Again
                        } else {
                            Step::Done(Ok(self.moved))
                        }
                    }
                }
            }
            Kind::Send { fd } => {
                let rest = &self.buf.bytes()[self.moved..self.buf.filled()];
                if rest.is_empty() {
                    return Step::Done(Ok(self.moved));
                }
                let len = rest.len().min(piece);
                // SAFETY: `rest` is a live, readable region of at least `len` bytes.
                let n = unsafe {
                    libc::send(
                        *fd,
                        rest.as_ptr().cast(),
                        len,
                        libc::MSG_DONTWAIT | libc::MSG_NOSIGNAL,
                    )
                };
                match counted(n) {
                    Err(error) if would_block(&error) || interrupted(&error) => Step::Idle,
                    Err(error) => Step::Done(Err(error)),
                    Ok(0) => Step::Done(Err(io::Error::from(io::ErrorKind::WriteZero))),
                    Ok(n) => {
                        self.moved += n;
                        if self.moved < self.buf.filled() {
                            Step::Again
                        } else {
                            Step::Done(Ok(self.moved))
                        }
                    }
                }
            }
            Kind::Recv { fd } => {
                let bytes = self.buf.bytes_mut();
                let len = bytes.len().min(max_piece);
                // SAFETY: `bytes` is a live, writable region of at least `len` bytes.
                let n =
                    unsafe { libc::recv(*fd, bytes.as_mut_ptr().cast(), len, libc::MSG_DONTWAIT) };
                match counted(n) {
                    Err(error) if would_block(&error) || interrupted(&error) => Step::Idle,
                    Err(error) => Step::Done(Err(error)),
                    Ok(n) => Step::Done(Ok(n)),
                }
            }
            Kind::Accept { fd } => {
                // SAFETY: null address buffers are allowed; the call reads no memory of ours.
                let accepted = nonblocking(*fd, || unsafe {
                    libc::accept4(
                        *fd,
                        std::ptr::null_mut(),
                        std::ptr::null_mut(),
                        libc::SOCK_CLOEXEC,
                    ) as isize
                });
                match accepted {
                    Err(error) if would_block(&error) || interrupted(&error) => Step::Idle,
                    Err(error) => Step::Done(Err(error)),
                    Ok(fd) => Step::Done(Ok(fd)),
                }
            }
            Kind::Open { path, flags, mode } => open_step(path, *flags, *mode),
            Kind::Rename { from, to } => {
                // SAFETY: both paths are NUL-terminated strings that live through the call.
                let done = unsafe { libc::rename(from.as_ptr(), to.as_ptr()) };
                Step::Done(counted(done as isize))
            }
            Kind::FileLen { fd } => {
                // SAFETY: `stat` is plain integers: all zeros is a valid value.
                let mut stat: libc::stat = unsafe { std::mem::zeroed() };
                // SAFETY: `stat` is a valid `stat` for the call to write.
                if unsafe { libc::fstat(*fd, &raw mut stat) } < 0 {
                    return Step::Done(Err(last_error()));
                }
                Step::Done(
                    usize::try_from(stat.st_size)
                        .map_err(|_| io::Error::from_raw_os_error(libc::EOVERFLOW)),
                )
            }
            Kind::SendTo { fd, addr, len } => {
                let bytes = &self.buf.bytes()[..self.buf.filled()];
                // SAFETY: `bytes` is a live, readable region; `addr` holds `len` bytes of address.
                let n = unsafe {
                    libc::sendto(
                        *fd,
                        bytes.as_ptr().cast(),
                        bytes.len(),
                        libc::MSG_DONTWAIT | libc::MSG_NOSIGNAL,
                        std::ptr::from_ref(&**addr).cast(),
                        *len,
                    )
                };
                match counted(n) {
                    Err(error) if would_block(&error) || interrupted(&error) => Step::Idle,
                    Err(error) => Step::Done(Err(error)),
                    Ok(n) => Step::Done(Ok(n)),
                }
            }
            Kind::Connect {
                fd,
                addr,
                len,
                state,
            } => match state {
                Connecting::New => connect_start(*fd, addr, *len, state),
                Connecting::Under(_) => connect_check(*fd),
            },
        }
    }

    /// Called when the operation ends, whatever ended it: puts back what a connect changed.
    fn release(&mut self) {
        if let Kind::Connect {
            state: Connecting::Under(restore),
            ..
        } = &mut self.kind
        {
            restore.restore();
        }
    }
}

/// The step of an open. The file is opened non-blocking so that a FIFO or a device cannot hold the worker, and
/// made blocking again as the caller would get it from io_uring, unless the caller asked for `O_NONBLOCK`.
fn open_step(path: &CString, flags: i32, mode: libc::mode_t) -> Step {
    // SAFETY: `path` is a NUL-terminated string that lives through the call.
    let fd = unsafe {
        libc::open(
            path.as_ptr(),
            flags | libc::O_CLOEXEC | libc::O_NONBLOCK,
            libc::c_uint::from(mode),
        )
    };
    if fd < 0 {
        let error = last_error();
        if error.raw_os_error() == Some(libc::ENXIO) && is_fifo(path) {
            // A FIFO opened for writing has no reader yet, and nothing tells when one appears: try again at a
            // later visit, with growing gaps.
            return Step::Idle;
        }
        // A file leased to another process: the lease holder is told, and a blocking open waits for the lease to
        // break (or the holder to let go) where the non-blocking one this makes is refused at once. The caller
        // who asked for `O_NONBLOCK` gets the refusal.
        if flags & libc::O_NONBLOCK == 0 && would_block(&error) && is_regular(path) {
            return Step::Idle;
        }
        if interrupted(&error) {
            return Step::Idle;
        }
        return Step::Done(Err(error));
    }
    if flags & libc::O_NONBLOCK == 0 {
        // SAFETY: plain `fcntl` calls of the descriptor just opened.
        let set = unsafe {
            let current = libc::fcntl(fd, libc::F_GETFL);
            current < 0 || libc::fcntl(fd, libc::F_SETFL, current & !libc::O_NONBLOCK) < 0
        };
        if set {
            let error = last_error();
            // SAFETY: `fd` was just opened and nothing else owns it.
            unsafe { libc::close(fd) };
            return Step::Done(Err(error));
        }
    }
    Step::Done(Ok(fd as usize))
}

fn is_regular(path: &CString) -> bool {
    // SAFETY: `stat` is plain integers: all zeros is a valid value.
    let mut stat: libc::stat = unsafe { std::mem::zeroed() };
    // SAFETY: `path` is a NUL-terminated string; `stat` is valid for the call to write.
    unsafe {
        libc::stat(path.as_ptr(), &raw mut stat) == 0
            && stat.st_mode & libc::S_IFMT == libc::S_IFREG
    }
}

fn is_fifo(path: &CString) -> bool {
    // SAFETY: `stat` is plain integers: all zeros is a valid value.
    let mut stat: libc::stat = unsafe { std::mem::zeroed() };
    // SAFETY: `path` is a NUL-terminated string; `stat` is valid for the call to write.
    unsafe {
        libc::stat(path.as_ptr(), &raw mut stat) == 0
            && stat.st_mode & libc::S_IFMT == libc::S_IFIFO
    }
}

/// The first step of a connect: makes the socket non-blocking for the handshake and starts it.
fn connect_start(
    fd: RawFd,
    addr: &libc::sockaddr_storage,
    len: libc::socklen_t,
    state: &mut Connecting,
) -> Step {
    // SAFETY: a plain `fcntl` of the caller's socket.
    let flags = unsafe { libc::fcntl(fd, libc::F_GETFL) };
    if flags < 0 {
        return Step::Done(Err(last_error()));
    }
    let blocking = flags & libc::O_NONBLOCK == 0;
    let mut restore = Restore {
        fd,
        flags: blocking.then_some(flags),
    };
    if blocking {
        // SAFETY: as above.
        if unsafe { libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK) } < 0 {
            restore.flags = None;
            return Step::Done(Err(last_error()));
        }
    }
    // SAFETY: `addr` holds `len` bytes of address.
    let done = unsafe { libc::connect(fd, std::ptr::from_ref(addr).cast(), len) };
    if done == 0 {
        return Step::Done(Ok(0));
    }
    let error = last_error();
    if error.raw_os_error() == Some(libc::EINPROGRESS) || interrupted(&error) {
        *state = Connecting::Under(restore);
        Step::Idle
    } else {
        Step::Done(Err(error))
    }
}

/// A later step of a connect, once `poll(2)` said the socket has something: how the handshake ended.
fn connect_check(fd: RawFd) -> Step {
    let mut code: libc::c_int = 0;
    let mut size = size_of::<libc::c_int>() as libc::socklen_t;
    // SAFETY: `code` and `size` are valid for the call to write.
    let got = unsafe {
        libc::getsockopt(
            fd,
            libc::SOL_SOCKET,
            libc::SO_ERROR,
            (&raw mut code).cast(),
            &raw mut size,
        )
    };
    if got < 0 {
        Step::Done(Err(last_error()))
    } else if code != 0 {
        Step::Done(Err(io::Error::from_raw_os_error(code)))
    } else {
        Step::Done(Ok(0))
    }
}

fn kind(request: Request) -> io::Result<Kind> {
    Ok(match request {
        Request::Read { file, offset } => Kind::Read {
            fd: file as RawFd,
            offset,
        },
        Request::Write { file, offset } => Kind::Write {
            fd: file as RawFd,
            offset,
        },
        Request::Send { sock } => Kind::Send { fd: sock as RawFd },
        Request::Recv { sock } => Kind::Recv { fd: sock as RawFd },
        Request::Accept { listener } => Kind::Accept {
            fd: listener as RawFd,
        },
        Request::OpenAt { path, flags, mode } => Kind::Open {
            path: c_path(&path)?,
            flags,
            mode: mode as libc::mode_t,
        },
        Request::Open { path, spec } => Kind::Open {
            path: c_path(&path)?,
            flags: open_flags(spec),
            mode: spec.unix_mode as libc::mode_t,
        },
        Request::Rename { from, to } => Kind::Rename {
            from: c_path(&from)?,
            to: c_path(&to)?,
        },
        Request::FileLen { file } => Kind::FileLen { fd: file as RawFd },
        Request::SendTo { sock, to } => {
            let (addr, len) = socket_address(to);
            Kind::SendTo {
                fd: sock as RawFd,
                addr,
                len,
            }
        }
        Request::Connect { sock, to } => {
            let (addr, len) = socket_address(to);
            Kind::Connect {
                fd: sock as RawFd,
                addr,
                len,
                state: Connecting::New,
            }
        }
    })
}

impl Positional {
    pub(super) fn new() -> Self {
        Self::with_piece(MAX_PIECE)
    }

    /// An engine whose read, write and send steps move at most `max_piece` bytes (tests force short ones).
    pub(super) fn with_piece(max_piece: usize) -> Self {
        let queue: MpscQueue<Message> = MpscQueue::with_pool(INCOMING_CHUNK, 2);
        Self {
            id: ID_MARK | NEXT_ID.fetch_add(1, Ordering::Relaxed),
            next: AtomicU64::new(1),
            incoming: queue.producer(),
            state: SourceCell::new(State {
                incoming: queue.consumer(),
                active: VecDeque::new(),
                fds: Vec::new(),
                slots: Vec::new(),
            }),
            max_piece: max_piece.clamp(1, MAX_PIECE),
            #[cfg(test)]
            readiness_calls: std::sync::atomic::AtomicUsize::new(0),
            #[cfg(test)]
            entries: std::sync::atomic::AtomicUsize::new(0),
            #[cfg(test)]
            attempts: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    fn send(&self, message: Message) {
        // The queue is unbounded: an offer is never refused.
        let queued = self.incoming.offer(message);
        assert!(queued.is_ok(), "the unbounded queue refused a message");
    }
}

impl Engine for Positional {
    fn submit(
        &self,
        request: Request,
        buf: Buf,
        then: Then,
        limit: Option<Duration>,
        taken: &Cell<bool>,
    ) -> io::Result<OpId> {
        let kind = kind(request)?;
        let token = self.next.fetch_add(1, Ordering::Relaxed);
        self.send(Message::Submit(Box::new(Op {
            token,
            kind,
            buf,
            moved: 0,
            then,
            limit: limit.and_then(|limit| Instant::now().checked_add(limit)),
            cancelled: false,
            stream: false,
            waiting: false,
            ready: false,
            skip: 0,
            next_skip: 1,
        })));
        taken.set(true);
        Ok(OpId::new(self.id, 0, token))
    }

    fn cancel(&self, id: OpId) {
        if id.engine() == self.id {
            self.send(Message::Cancel(id.token()));
        }
    }

    fn poll(&self, room: &mut dyn FnMut() -> bool, offer: &mut dyn FnMut(Box<dyn Frame>) -> bool) {
        let Some(mut guard) = self.state.try_enter() else {
            return;
        };
        let State {
            incoming,
            active,
            fds,
            slots,
        } = &mut *guard;
        incoming.drain_all(|message| match message {
            Message::Submit(op) => active.push_back(*op),
            Message::Cancel(token) => {
                if let Some(op) = active.iter_mut().find(|op| op.token == token) {
                    op.cancelled = true;
                }
            }
        });
        // One readiness call for everything that waits for a descriptor.
        fds.clear();
        slots.clear();
        let now = Instant::now();
        for (at, op) in active.iter_mut().enumerate() {
            op.ready = false;
            op.waiting = false;
            if let Some((fd, events)) = op.waits_for(now) {
                op.waiting = true;
                slots.push((fd, events, at));
            }
        }
        // One entry per distinct descriptor, with the events of every operation on it merged: the count can never
        // pass the descriptor limit, which `poll(2)` refuses.
        slots.sort_unstable_by_key(|&(fd, _, _)| fd);
        for &(fd, events, _) in slots.iter() {
            match fds.last_mut() {
                Some(last) if last.fd == fd => last.events |= events,
                _ => fds.push(libc::pollfd {
                    fd,
                    events,
                    revents: 0,
                }),
            }
        }
        if !fds.is_empty() {
            #[cfg(test)]
            {
                self.readiness_calls.fetch_add(1, Ordering::Relaxed);
                self.entries.store(fds.len(), Ordering::Relaxed);
            }
            // SAFETY: `fds` is a valid array of `fds.len()` `pollfd`s; a zero timeout never waits.
            let found = unsafe { libc::poll(fds.as_mut_ptr(), fds.len() as libc::nfds_t, 0) };
            if found > 0 {
                let mut entry = 0;
                for &(fd, events, at) in slots.iter() {
                    while fds[entry].fd != fd {
                        entry += 1;
                    }
                    let wanted = events | libc::POLLERR | libc::POLLHUP | libc::POLLNVAL;
                    active[at].ready = fds[entry].revents & wanted != 0;
                }
            } else if found < 0 && !interrupted(&last_error()) {
                // The call itself failed: let each operation's own call say what is wrong.
                for &(_, _, at) in slots.iter() {
                    active[at].ready = true;
                }
            }
        }
        // Operations that will be attempted this visit: one that is not due (its descriptor is not ready, or it is
        // backing off) is no rival for the budget. A lone one takes all of it.
        let mut due = 0;
        for op in active.iter_mut() {
            let ending = op.cancelled || op.limit.is_some_and(|at| now >= at);
            if op.skip > 0 && !ending {
                // Backing off: this visit passes it by, as for an operation whose descriptor is not ready.
                op.skip -= 1;
                op.waiting = true;
                op.ready = false;
            } else if (!op.waiting || op.ready) && !op.cancelled {
                op.skip = 0;
                due += 1;
            }
        }
        let turns = if due == 1 {
            ATTEMPTS_PER_POLL
        } else {
            PIECES_PER_TURN
        };
        let mut budget = ATTEMPTS_PER_POLL;
        for _ in 0..active.len() {
            let Some(mut op) = active.pop_front() else {
                return;
            };
            if op.waiting && !op.ready {
                active.push_back(op);
                continue;
            }
            let mut turn = 0;
            loop {
                if budget == 0 || !room() {
                    active.push_front(op);
                    return;
                }
                budget -= 1;
                turn += 1;
                #[cfg(test)]
                self.attempts.fetch_add(1, Ordering::Relaxed);
                match op.step(self.max_piece) {
                    Step::Done(result) => {
                        op.release();
                        let fills = matches!(op.kind, Kind::Read { .. } | Kind::Recv { .. });
                        let Op { buf, then, .. } = op;
                        if !offer(conclude_io(buf, then, fills, result)) {
                            return;
                        }
                        break;
                    }
                    Step::Again if turn < turns && !op.stream => continue,
                    Step::Again | Step::Idle => {
                        if matches!(op.kind, Kind::Open { .. }) {
                            op.skip = op.next_skip;
                            op.next_skip = (op.next_skip * 2).min(OPEN_RETRY_VISITS_MAX);
                        }
                        active.push_back(op);
                        break;
                    }
                }
            }
        }
    }
}
