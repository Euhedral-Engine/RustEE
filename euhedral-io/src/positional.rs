//! The positional backend of [`AsyncIo`], for Linux and Android where io_uring is refused.
//!
//! It owns no thread and waits for nothing. A submission only queues the operation; the work is done by the
//! reap frame that the `AsyncIo` pushes into the lattice, inside [`Engine::poll`]: each operation takes one
//! step per visit (one piece of a read or write, one non-blocking attempt of a socket call, or the whole of an
//! open, a rename or a length), and the operations that did not end go to the back of the line. Sockets are
//! asked with `MSG_DONTWAIT` (or `poll(2)` with no timeout), so a socket that is not ready costs a visit and
//! never a wait.
//!
//! Everything here is plain state of the one poll that is inside (`SourceCell`); only the queue of messages
//! from the submitting and cancelling threads is shared.
#![allow(unsafe_code)]

use std::collections::VecDeque;
use std::ffi::CString;
use std::io;
use std::os::fd::RawFd;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use euhedral_core::frames::Frame;
use euhedral_core::generics::SourceCell;
use euhedral_data_structures::queues::{Consumer, MpscQueue, Multi, Producer, Single};

use super::{
    Buf, Cancelled, Create, Engine, LimitPassed, MAX_PIECE, OpId, OpenSpec, READ_PIECE, Request,
    Then, conclude as conclude_io,
};

/// Slots per chunk of the queue of messages.
const INCOMING_CHUNK: usize = 256;
/// The most steps one poll takes, so that a reap is bounded work however much is queued.
const STEPS_PER_POLL: usize = 64;
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
}

/// What only the poll that is inside touches.
struct State {
    incoming: Consumer<Message, Single>,
    /// The operations that are not done, the oldest first.
    active: VecDeque<Op>,
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
    /// The handshake is under way; `restore` holds the descriptor's flags to put back when it ends, if the
    /// engine made a blocking socket non-blocking for it.
    Under { restore: Option<i32> },
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
}

/// What one step of an operation came to.
enum Step {
    Done(io::Result<usize>),
    /// Moved forward, and has more to do.
    Again,
    /// Could not move: the socket is not ready.
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

/// What a socket call that returned `n` came to, for a call that moves bytes of a buffer's rest.
fn counted(n: isize) -> Result<usize, io::Error> {
    if n < 0 {
        Err(last_error())
    } else {
        Ok(n as usize)
    }
}

impl Op {
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
                let at = offset.wrapping_add(self.moved as u64) as libc::off_t;
                // SAFETY: `rest` is a live, writable region of at least `len` bytes, and nothing else
                // touches it while the operation is here.
                let n = unsafe { libc::pread(*fd, rest.as_mut_ptr().cast(), len, at) };
                match counted(n) {
                    Err(error) if interrupted(&error) => Step::Again,
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
                let len = rest.len().min(piece);
                let at = offset.wrapping_add(self.moved as u64) as libc::off_t;
                // SAFETY: `rest` is a live, readable region of at least `len` bytes.
                let n = unsafe { libc::pwrite(*fd, rest.as_ptr().cast(), len, at) };
                match counted(n) {
                    Err(error) if interrupted(&error) => Step::Again,
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
                let mut ask = libc::pollfd {
                    fd: *fd,
                    events: libc::POLLIN,
                    revents: 0,
                };
                // SAFETY: one valid `pollfd`; a zero timeout never waits.
                let ready = unsafe { libc::poll(&raw mut ask, 1, 0) };
                if ready == 0 || (ready < 0 && interrupted(&last_error())) {
                    return Step::Idle;
                }
                // SAFETY: null address buffers are allowed; the call reads no memory of ours.
                let accepted = unsafe {
                    libc::accept4(
                        *fd,
                        std::ptr::null_mut(),
                        std::ptr::null_mut(),
                        libc::SOCK_CLOEXEC,
                    )
                };
                match counted(accepted as isize) {
                    Err(error) if would_block(&error) || interrupted(&error) => Step::Idle,
                    Err(error) => Step::Done(Err(error)),
                    Ok(fd) => Step::Done(Ok(fd)),
                }
            }
            Kind::Open { path, flags, mode } => {
                // SAFETY: `path` is a NUL-terminated string that lives through the call.
                let fd = unsafe {
                    libc::open(
                        path.as_ptr(),
                        *flags | libc::O_CLOEXEC,
                        libc::c_uint::from(*mode),
                    )
                };
                Step::Done(counted(fd as isize))
            }
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
                Connecting::Under { .. } => connect_check(*fd),
            },
        }
    }

    /// Called when the operation ends, whatever ended it: puts back what a connect changed.
    fn release(&mut self) {
        if let Kind::Connect {
            fd,
            state: Connecting::Under {
                restore: Some(flags),
            },
            ..
        } = &self.kind
        {
            // SAFETY: a plain `fcntl` of the caller's socket, which stays open until the continuation runs.
            unsafe { libc::fcntl(*fd, libc::F_SETFL, *flags) };
        }
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
    let restore = || {
        if blocking {
            // SAFETY: as above.
            unsafe { libc::fcntl(fd, libc::F_SETFL, flags) };
        }
    };
    if blocking {
        // SAFETY: as above.
        if unsafe { libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK) } < 0 {
            return Step::Done(Err(last_error()));
        }
    }
    // SAFETY: `addr` holds `len` bytes of address.
    let done = unsafe { libc::connect(fd, std::ptr::from_ref(addr).cast(), len) };
    if done == 0 {
        restore();
        return Step::Done(Ok(0));
    }
    let error = last_error();
    if error.raw_os_error() == Some(libc::EINPROGRESS) || interrupted(&error) {
        *state = Connecting::Under {
            restore: blocking.then_some(flags),
        };
        Step::Idle
    } else {
        restore();
        Step::Done(Err(error))
    }
}

/// A later step of a connect: has the handshake ended, and how.
fn connect_check(fd: RawFd) -> Step {
    let mut ask = libc::pollfd {
        fd,
        events: libc::POLLOUT,
        revents: 0,
    };
    // SAFETY: one valid `pollfd`; a zero timeout never waits.
    let ready = unsafe { libc::poll(&raw mut ask, 1, 0) };
    if ready == 0 || (ready < 0 && interrupted(&last_error())) {
        return Step::Idle;
    }
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

fn c_path(path: &std::path::Path) -> io::Result<CString> {
    use std::os::unix::ffi::OsStrExt;
    CString::new(path.as_os_str().as_bytes())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "a path with a NUL byte"))
}

/// `open(2)` flags for an [`OpenSpec`].
fn open_flags(spec: OpenSpec) -> i32 {
    let access = match (spec.read, spec.write) {
        (_, false) => libc::O_RDONLY,
        (false, true) => libc::O_WRONLY,
        (true, true) => libc::O_RDWR,
    };
    let create = match spec.create {
        Create::Never => 0,
        Create::IfMissing => libc::O_CREAT,
        Create::New => libc::O_CREAT | libc::O_EXCL,
        Create::Truncate => libc::O_CREAT | libc::O_TRUNC,
    };
    access | create
}

fn socket_address(to: std::net::SocketAddr) -> (Box<libc::sockaddr_storage>, libc::socklen_t) {
    // SAFETY: `sockaddr_storage` is plain old data; all zeros is a valid (unspecified) address.
    let mut storage: Box<libc::sockaddr_storage> = Box::new(unsafe { std::mem::zeroed() });
    let len = match to {
        std::net::SocketAddr::V4(v4) => {
            let addr = libc::sockaddr_in {
                sin_family: libc::AF_INET as libc::sa_family_t,
                sin_port: v4.port().to_be(),
                sin_addr: libc::in_addr {
                    s_addr: u32::from_ne_bytes(v4.ip().octets()),
                },
                sin_zero: [0; 8],
            };
            // SAFETY: `sockaddr_storage` is larger than and aligned for every socket address type.
            unsafe {
                std::ptr::from_mut(&mut *storage)
                    .cast::<libc::sockaddr_in>()
                    .write(addr)
            };
            size_of::<libc::sockaddr_in>()
        }
        std::net::SocketAddr::V6(v6) => {
            let addr = libc::sockaddr_in6 {
                sin6_family: libc::AF_INET6 as libc::sa_family_t,
                sin6_port: v6.port().to_be(),
                sin6_flowinfo: v6.flowinfo(),
                sin6_addr: libc::in6_addr {
                    s6_addr: v6.ip().octets(),
                },
                sin6_scope_id: v6.scope_id(),
            };
            // SAFETY: as above.
            unsafe {
                std::ptr::from_mut(&mut *storage)
                    .cast::<libc::sockaddr_in6>()
                    .write(addr)
            };
            size_of::<libc::sockaddr_in6>()
        }
    };
    (
        storage,
        libc::socklen_t::try_from(len).expect("a socket address length"),
    )
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
            }),
            max_piece: max_piece.clamp(1, MAX_PIECE),
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
        })));
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
        let State { incoming, active } = &mut *guard;
        incoming.drain_all(|message| match message {
            Message::Submit(op) => active.push_back(*op),
            Message::Cancel(token) => {
                if let Some(op) = active.iter_mut().find(|op| op.token == token) {
                    op.cancelled = true;
                }
            }
        });
        let mut budget = STEPS_PER_POLL;
        loop {
            let mut moved_on = false;
            for _ in 0..active.len() {
                if budget == 0 || !room() {
                    return;
                }
                let Some(mut op) = active.pop_front() else {
                    return;
                };
                match op.step(self.max_piece) {
                    Step::Done(result) => {
                        budget -= 1;
                        moved_on = true;
                        op.release();
                        let fills = matches!(op.kind, Kind::Read { .. } | Kind::Recv { .. });
                        let Op { buf, then, .. } = op;
                        if !offer(conclude_io(buf, then, fills, result)) {
                            return;
                        }
                    }
                    Step::Again => {
                        budget -= 1;
                        moved_on = true;
                        active.push_back(op);
                    }
                    Step::Idle => active.push_back(op),
                }
            }
            if !moved_on {
                return;
            }
        }
    }
}
