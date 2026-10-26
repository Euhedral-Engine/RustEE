//! The Windows engine of `AsyncIo`: I/O completion ports for every operation, and IoRing (a kernel
//! ring, as io_uring is) for file reads and writes where the OS has one.
//!
//! - **One completion port per `AsyncIo`.** Any worker issues overlapped calls; the lattice's
//!   polls dequeue (`GetQueuedCompletionStatusEx` with a zero timeout, so nothing waits). Every
//!   operation ends through a poll: the "skip the port on success" mode is not set.
//! - **One IoRing per submitting thread**, for file reads and writes, when the OS offers a kernel
//!   ring ([`Sys::ring_probe`]: not emulated in user mode, with the operation supported). A full
//!   submission queue goes to the ring's backlog and is handed over by a later submission or poll.
//!   Each operation falls back to the port on its own.
//! - **Open and rename** have no asynchronous call. They run on the OS thread pool and post their
//!   result to the port, so it comes out of the same poll. It is the one place the crate uses
//!   threads it did not create.
//! - **Records.** An operation's state lives in a boxed [`Rec`], whose `OVERLAPPED` is its first
//!   field. The record's address is the operation's identity: a dequeued packet carries it, and a
//!   ring completion has it as its user data. The record is found in the table by that address and
//!   is never dereferenced from it.
//!
//! Every Windows call goes through [`Sys`], so the state machines run on any platform against a
//! simulated `Sys` (`sim`, in the tests). Only [`real`] (Windows) makes the calls.

#![allow(unsafe_code)]
#![cfg_attr(not(windows), allow(dead_code))]

use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::io;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, Weak};

use euhedral_core::frames::Frame;

use self::sys::{
    ACCEPT_BUF, BuildError, Cqe, Handle, IORING_VERSION_3, Overlapped, Packet, SockAddr, Sys,
    hresult_error, is_end_of_file,
};
use super::{
    AsyncIo, AsyncIoOptions, Engine as EngineTrait, IoBuf, IoDone, MAX_PIECE, OpenSpec, READ_PIECE,
    Request, Then, invalid, lock,
};

#[cfg(all(windows, test))]
mod live;
#[cfg(windows)]
mod real;
#[cfg(test)]
mod sim;
pub(super) mod sys;
#[cfg(test)]
mod tests;

/// Submission entries of each ring (its completion queue has twice as many).
const RING_ENTRIES: u32 = 256;
/// Submitting threads (rings) one `AsyncIo` serves.
const MAX_RINGS: usize = 256;
/// Packets taken from the port by one system call.
const BATCH: usize = 64;

static NEXT_ID: AtomicUsize = AtomicUsize::new(1);

thread_local! {
    /// This thread's ring of each engine it submitted to (by id).
    static MINE: RefCell<Vec<(usize, Weak<Ring>)>> = const { RefCell::new(Vec::new()) };
}

/// The completion port, closed when the engine and every pool job that may post to it are gone.
struct Port {
    sys: Arc<dyn Sys>,
    handle: Handle,
}

impl Drop for Port {
    fn drop(&mut self) {
        self.sys.close_port(self.handle);
    }
}

/// What an operation does, with the handles it uses.
enum Kind {
    Read { file: Handle, offset: u64 },
    Write { file: Handle, offset: u64 },
    Send { sock: Handle },
    Recv { sock: Handle },
    SendTo { sock: Handle },
    Accept { listener: Handle },
    Connect { sock: Handle },
    Open { path: PathBuf, spec: OpenSpec },
    Rename { from: PathBuf, to: PathBuf },
}

/// How the operation reaches the kernel.
enum Route {
    Port,
    Ring(Arc<Ring>),
}

/// The socket an accept fills, and the buffer `AcceptEx` writes the two addresses to.
struct Accepting {
    socket: Handle,
    #[allow(dead_code, reason = "written by the kernel")]
    addresses: Box<[u8; ACCEPT_BUF]>,
}

/// The result an open or rename leaves for the poll that dequeues its packet.
type PoolResult = Arc<Mutex<Option<io::Result<usize>>>>;

/// An operation in flight. `ov` is first, so the record's address is the `OVERLAPPED`'s; the
/// record's buffer and address blocks stay put until the operation ends.
#[repr(C)]
struct Rec {
    ov: Overlapped,
    kind: Kind,
    route: Route,
    buf: IoBuf,
    /// Bytes moved so far (reads, writes and sends resubmit the rest).
    moved: usize,
    then: Option<Then>,
    /// The destination of a datagram or a connect.
    addr: Option<Box<SockAddr>>,
    accepting: Option<Accepting>,
    pool: Option<PoolResult>,
    /// The call failed at once: no packet will come, so the error is the operation's result.
    failed: Option<io::Error>,
}

/// The call that starts or continues an operation, with its pointers taken before the record is
/// shared (nothing of the record is touched after the call is made).
enum Call {
    Read {
        file: Handle,
        buf: *mut u8,
        len: u32,
        ov: *mut Overlapped,
    },
    Write {
        file: Handle,
        buf: *const u8,
        len: u32,
        ov: *mut Overlapped,
    },
    Send {
        sock: Handle,
        buf: *const u8,
        len: u32,
        ov: *mut Overlapped,
    },
    Recv {
        sock: Handle,
        buf: *mut u8,
        len: u32,
        ov: *mut Overlapped,
    },
    SendTo {
        sock: Handle,
        buf: *const u8,
        len: u32,
        to: *const SockAddr,
        ov: *mut Overlapped,
    },
    Accept {
        listener: Handle,
        accepted: Handle,
        addresses: *mut u8,
        ov: *mut Overlapped,
    },
    Connect {
        sock: Handle,
        to: *const SockAddr,
        ov: *mut Overlapped,
    },
    Pool(Box<dyn FnOnce() + Send>),
    Ring(Arc<Ring>, Entry),
}

/// One submitting thread's ring.
struct Ring {
    sys: Arc<dyn Sys>,
    handle: Handle,
    inner: Mutex<RingInner>,
    /// Entries built and not yet popped; at most the completion queue's size, so it never
    /// overflows.
    in_kernel: AtomicUsize,
    capacity: usize,
}

struct RingInner {
    /// Entries waiting for room in the kernel's queues, the oldest first.
    backlog: VecDeque<Entry>,
    /// Entries are built that the kernel has not taken (a submit failed).
    unsubmitted: bool,
}

impl Drop for Ring {
    fn drop(&mut self) {
        // A ring with operations in flight leaks: the kernel may still write their buffers.
        if self.in_kernel.load(Ordering::Acquire) == 0 {
            self.sys.ring_close(self.handle);
        }
    }
}

/// A ring entry waiting to be built.
#[derive(Clone, Copy)]
struct Entry {
    write: bool,
    file: Handle,
    buf: usize,
    len: u32,
    offset: u64,
    user_data: usize,
}

pub(super) struct Engine {
    id: usize,
    sys: Arc<dyn Sys>,
    port: Arc<Port>,
    /// The operations in flight, by their record's address.
    ops: Mutex<HashMap<usize, Box<Rec>>>,
    /// Operations whose call failed at once, to end with that error at the next poll.
    ready: Mutex<VecDeque<Box<Rec>>>,
    /// Packets taken from the port and not yet handled (a poll stops when the caller has room for
    /// no more); only the polling thread touches it.
    inbox: Mutex<VecDeque<Packet>>,
    ring_reads: bool,
    ring_writes: bool,
    rings: Mutex<Vec<Arc<Ring>>>,
    max_piece: usize,
}

impl Drop for Engine {
    fn drop(&mut self) {
        // The kernel may still write into the buffers of operations in flight: leak them.
        for (_, rec) in std::mem::take(&mut *lock(&self.ops)) {
            std::mem::forget(rec);
        }
    }
}

impl Engine {
    /// An engine over `sys`. Fails if there is no completion port.
    pub(super) fn new(
        sys: Arc<dyn Sys>,
        options: AsyncIoOptions,
        max_piece: usize,
    ) -> io::Result<Self> {
        let handle = sys.create_port()?;
        let port = Arc::new(Port {
            sys: sys.clone(),
            handle,
        });
        // A ring is used only where the OS has a kernel ring: not one emulated in user mode.
        let probe = options
            .windows_ioring
            .then(|| sys.ring_probe())
            .flatten()
            .filter(|probe| !probe.emulated);
        Ok(Self {
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            sys,
            port,
            ops: Mutex::new(HashMap::new()),
            ready: Mutex::new(VecDeque::new()),
            inbox: Mutex::new(VecDeque::new()),
            ring_reads: probe.is_some_and(|probe| probe.read),
            ring_writes: probe
                .is_some_and(|probe| probe.write && probe.version >= IORING_VERSION_3),
            rings: Mutex::new(Vec::new()),
            max_piece: max_piece.clamp(1, MAX_PIECE),
        })
    }

    /// Whether file reads and writes go through a ring (for tests).
    #[cfg(test)]
    pub(super) fn ring_use(&self) -> (bool, bool) {
        (self.ring_reads, self.ring_writes)
    }

    /// The number of bytes of the next piece of `n` left: at most `max_piece`, and a `u32`.
    fn piece(&self, n: usize) -> u32 {
        u32::try_from(n.min(self.max_piece)).unwrap_or(u32::MAX)
    }

    /// The calling thread's ring, made on first use; `None` if there is none to be had.
    fn ring(&self) -> Option<Arc<Ring>> {
        MINE.with_borrow_mut(|mine| {
            mine.retain(|(_, ring)| ring.strong_count() > 0);
            if let Some(ring) = mine
                .iter()
                .find(|(id, _)| *id == self.id)
                .and_then(|(_, ring)| ring.upgrade())
            {
                return Some(ring);
            }
            let mut rings = lock(&self.rings);
            if rings.len() >= MAX_RINGS {
                return None;
            }
            let handle = self.sys.ring_create(RING_ENTRIES).ok()?;
            let ring = Arc::new(Ring {
                sys: self.sys.clone(),
                handle,
                inner: Mutex::new(RingInner {
                    backlog: VecDeque::new(),
                    unsubmitted: false,
                }),
                in_kernel: AtomicUsize::new(0),
                capacity: 2 * RING_ENTRIES as usize,
            });
            rings.push(ring.clone());
            mine.push((self.id, Arc::downgrade(&ring)));
            Some(ring)
        })
    }

    /// A file handle that would block the worker is refused.
    fn check_file(&self, file: Handle) -> io::Result<()> {
        if self.sys.is_synchronous(file)? {
            return Err(invalid(
                "a file handle opened without overlapped I/O would block the worker",
            ));
        }
        Ok(())
    }

    fn rec(kind: Kind, buf: IoBuf, then: Then) -> Box<Rec> {
        Box::new(Rec {
            ov: Overlapped::default(),
            kind,
            route: Route::Port,
            buf,
            moved: 0,
            then: Some(then),
            addr: None,
            accepting: None,
            pool: None,
            failed: None,
        })
    }

    /// Makes `rec` ring-routed if a ring serves this kind of operation and one can be had;
    /// otherwise it goes through the port, and its handle joins the port.
    fn route(&self, rec: &mut Rec, handle: Handle, ring_ok: bool) -> io::Result<()> {
        if ring_ok && let Some(ring) = self.ring() {
            rec.route = Route::Ring(ring);
            return Ok(());
        }
        self.sys.associate(self.port.handle, handle)
    }

    /// Starts or continues the operation in `rec`. Nothing of the record is touched once its call
    /// is made: the completion may already be on another thread.
    fn start(&self, mut rec: Box<Rec>) {
        let addr = std::ptr::from_ref::<Rec>(&*rec) as usize;
        let call = self.call(&mut rec, addr);
        lock(&self.ops).insert(addr, rec);
        if let Err(error) = self.issue(call) {
            // The call failed at once, so no completion will come: end it with the error.
            let rec = lock(&self.ops).remove(&addr);
            if let Some(rec) = rec {
                self.fail(rec, error);
            }
        }
    }

    /// Ends the record with `error` at the next poll.
    fn fail(&self, mut rec: Box<Rec>, error: io::Error) {
        rec.failed = Some(error);
        lock(&self.ready).push_back(rec);
    }

    /// Ends the operation at `addr`, still in the table, with `error`.
    fn fail_at(&self, addr: usize, error: io::Error) {
        let rec = lock(&self.ops).remove(&addr);
        if let Some(rec) = rec {
            self.fail(rec, error);
        }
    }

    /// The call for the record's next piece.
    fn call(&self, rec: &mut Rec, addr: usize) -> Call {
        let ov: *mut Overlapped = &raw mut rec.ov;
        let moved = rec.moved;
        match &rec.kind {
            Kind::Read { file, offset } => {
                let file = *file;
                let at = offset + moved as u64;
                // A read the page cache can serve is copied inside the call, on the submitting
                // worker: pieces of at most READ_PIECE, each piece's completion submitting the
                // next.
                let rest = &mut rec.buf.bytes[moved..];
                let len = self.piece(rest.len().min(READ_PIECE));
                let buf = rest.as_mut_ptr();
                if let Route::Ring(ring) = &rec.route {
                    let entry = Entry {
                        write: false,
                        file,
                        buf: buf as usize,
                        len,
                        offset: at,
                        user_data: addr,
                    };
                    return Call::Ring(ring.clone(), entry);
                }
                rec.ov.reset(at);
                Call::Read { file, buf, len, ov }
            }
            Kind::Write { file, offset } => {
                let file = *file;
                let at = offset + moved as u64;
                let rest = &rec.buf.bytes[moved..rec.buf.filled];
                let len = self.piece(rest.len());
                let buf = rest.as_ptr();
                if let Route::Ring(ring) = &rec.route {
                    let entry = Entry {
                        write: true,
                        file,
                        buf: buf as usize,
                        len,
                        offset: at,
                        user_data: addr,
                    };
                    return Call::Ring(ring.clone(), entry);
                }
                rec.ov.reset(at);
                Call::Write { file, buf, len, ov }
            }
            Kind::Send { sock } => {
                let sock = *sock;
                let bytes = &rec.buf.bytes[moved..rec.buf.filled];
                let len = self.piece(bytes.len().min(READ_PIECE));
                let buf = bytes.as_ptr();
                rec.ov.reset(0);
                Call::Send { sock, buf, len, ov }
            }
            Kind::Recv { sock } => {
                let sock = *sock;
                let bytes = &mut rec.buf.bytes[..];
                let len = self.piece(bytes.len());
                let buf = bytes.as_mut_ptr();
                rec.ov.reset(0);
                Call::Recv { sock, buf, len, ov }
            }
            Kind::SendTo { sock } => {
                let sock = *sock;
                let bytes = &rec.buf.bytes[..rec.buf.filled];
                let len = self.piece(bytes.len());
                let buf = bytes.as_ptr();
                let to: *const SockAddr = rec.addr.as_deref().expect("a datagram's address");
                rec.ov.reset(0);
                Call::SendTo {
                    sock,
                    buf,
                    len,
                    to,
                    ov,
                }
            }
            Kind::Accept { listener } => {
                let listener = *listener;
                let accepting = rec.accepting.as_mut().expect("an accept's socket");
                let accepted = accepting.socket;
                let addresses = accepting.addresses.as_mut_ptr();
                rec.ov.reset(0);
                Call::Accept {
                    listener,
                    accepted,
                    addresses,
                    ov,
                }
            }
            Kind::Connect { sock } => {
                let sock = *sock;
                let to: *const SockAddr = rec.addr.as_deref().expect("a connect's address");
                rec.ov.reset(0);
                Call::Connect { sock, to, ov }
            }
            Kind::Open { path, spec } => {
                let (path, spec) = (path.clone(), *spec);
                let slot = rec.pool.clone().expect("an open's result slot");
                let (sys, port) = (self.sys.clone(), self.port.clone());
                Call::Pool(Box::new(move || {
                    let result = sys.open_file(&path, spec);
                    *lock(&slot) = Some(result);
                    // The port stays open while this job holds it.
                    let _ = sys.post(port.handle, addr);
                }))
            }
            Kind::Rename { from, to } => {
                let (from, to) = (from.clone(), to.clone());
                let slot = rec.pool.clone().expect("a rename's result slot");
                let (sys, port) = (self.sys.clone(), self.port.clone());
                Call::Pool(Box::new(move || {
                    let result = sys.rename(&from, &to).map(|()| 0);
                    *lock(&slot) = Some(result);
                    let _ = sys.post(port.handle, addr);
                }))
            }
        }
    }

    /// Makes the call. `Ok` means an end will follow (a packet, a ring completion, or, for a ring
    /// entry, the entry's own failure at the next poll).
    fn issue(&self, call: Call) -> io::Result<()> {
        let sys = &*self.sys;
        match call {
            Call::Read { file, buf, len, ov } => {
                // SAFETY: the file is overlapped and associated with the port (`submit`); `buf`
                // points into the record's buffer and `ov` into the record, which the table keeps
                // untouched until the packet is dequeued.
                unsafe { sys.read_file(file, buf, len, ov) }
            }
            Call::Write { file, buf, len, ov } => {
                // SAFETY: as for a read.
                unsafe { sys.write_file(file, buf, len, ov) }
            }
            Call::Send { sock, buf, len, ov } => {
                // SAFETY: as for a read, on a connected socket.
                unsafe { sys.send(sock, buf, len, ov) }
            }
            Call::Recv { sock, buf, len, ov } => {
                // SAFETY: as for a read, on a connected socket.
                unsafe { sys.recv(sock, buf, len, ov) }
            }
            Call::SendTo {
                sock,
                buf,
                len,
                to,
                ov,
            } => {
                // SAFETY: as for a read; `to` points at the record's boxed address.
                unsafe { sys.send_to(sock, buf, len, to, ov) }
            }
            Call::Accept {
                listener,
                accepted,
                addresses,
                ov,
            } => {
                // SAFETY: as for a read; `accepted` is a fresh socket the record owns, and
                // `addresses` is the record's `ACCEPT_BUF`-byte block.
                unsafe { sys.accept(listener, accepted, addresses, ov) }
            }
            Call::Connect { sock, to, ov } => {
                // SAFETY: as for a read; the socket is bound and `to` is the record's boxed
                // address.
                unsafe { sys.connect(sock, to, ov) }
            }
            Call::Pool(job) => sys.run_on_pool(job),
            Call::Ring(ring, entry) => {
                lock(&ring.inner).backlog.push_back(entry);
                self.flush_ring(&ring);
                Ok(())
            }
        }
    }

    /// Hands what waits in the ring's backlog (or was built and not submitted) to the kernel,
    /// unless another thread is doing so right now.
    fn flush_ring(&self, ring: &Ring) {
        let Ok(mut inner) = ring.inner.try_lock() else {
            return;
        };
        self.flush_locked(ring, &mut inner);
    }

    fn flush_locked(&self, ring: &Ring, inner: &mut RingInner) {
        let sys = &*self.sys;
        let mut built = false;
        while ring.in_kernel.load(Ordering::Acquire) < ring.capacity {
            let Some(entry) = inner.backlog.front().copied() else {
                break;
            };
            // SAFETY: the entry's pointer is into its record's buffer, which the table keeps
            // untouched until its completion is popped; the file is overlapped.
            let built_now = unsafe {
                if entry.write {
                    sys.ring_write(
                        ring.handle,
                        entry.file,
                        entry.buf as *const u8,
                        entry.len,
                        entry.offset,
                        entry.user_data,
                    )
                } else {
                    sys.ring_read(
                        ring.handle,
                        entry.file,
                        entry.buf as *mut u8,
                        entry.len,
                        entry.offset,
                        entry.user_data,
                    )
                }
            };
            match built_now {
                Ok(()) => {
                    inner.backlog.pop_front();
                    ring.in_kernel.fetch_add(1, Ordering::AcqRel);
                    built = true;
                }
                Err(BuildError::Full) => break,
                Err(BuildError::Other(error)) => {
                    inner.backlog.pop_front();
                    self.fail_at(entry.user_data, error);
                }
            }
        }
        if built || inner.unsubmitted {
            inner.unsubmitted = sys.ring_submit(ring.handle).is_err();
        }
    }

    /// The result of the operation a dequeued packet ended.
    fn packet_result(&self, rec: &mut Rec) -> io::Result<usize> {
        if let Some(slot) = &rec.pool {
            return lock(slot)
                .take()
                .unwrap_or_else(|| Err(io::Error::other("the pool job left no result")));
        }
        let (handle, socket) = match rec.kind {
            Kind::Read { file, .. } | Kind::Write { file, .. } => (file, false),
            Kind::Send { sock } | Kind::Recv { sock } | Kind::SendTo { sock } => (sock, true),
            Kind::Connect { sock } => (sock, true),
            Kind::Accept { listener } => (listener, true),
            Kind::Open { .. } | Kind::Rename { .. } => unreachable!("they have a pool result"),
        };
        // SAFETY: the operation's packet was dequeued, so the call finished; `ov` is the record's.
        unsafe { self.sys.result(handle, &raw const rec.ov, socket) }
    }

    /// What a finished piece means for its operation: the rest goes in again, or it ended and its
    /// continuation's frame comes out.
    fn advance(&self, mut rec: Box<Rec>, result: io::Result<usize>) -> Option<Box<dyn Frame>> {
        let piecewise = matches!(
            rec.kind,
            Kind::Read { .. } | Kind::Write { .. } | Kind::Send { .. }
        );
        let total = match rec.kind {
            Kind::Read { .. } => rec.buf.bytes.len(),
            _ => rec.buf.filled,
        };
        // A read at the end of the data is a read of no bytes.
        let result = match (&rec.kind, result) {
            (Kind::Read { .. }, Err(error)) if is_end_of_file(&error) => Ok(0),
            (_, result) => result,
        };
        let ended = match (&rec.kind, result) {
            (_, Ok(n)) if piecewise && n > 0 => {
                rec.moved += n;
                if rec.moved < total {
                    self.start(rec);
                    return None;
                }
                Ok(rec.moved)
            }
            (Kind::Read { .. }, Ok(0)) => {
                if rec.moved == 0 {
                    Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "the read is at or past the end of the file",
                    ))
                } else {
                    Ok(rec.moved)
                }
            }
            (Kind::Write { .. }, Ok(0)) => Err(io::Error::from(io::ErrorKind::WriteZero)),
            (Kind::Send { .. }, Ok(0)) if rec.buf.filled > 0 => {
                Err(io::Error::from(io::ErrorKind::WriteZero))
            }
            (_, other) => other,
        };
        let ended = match rec.kind {
            Kind::Accept { listener } => {
                let accepting = rec.accepting.take().expect("an accept's socket");
                let accepted = ended.and_then(|_| {
                    self.sys.update_accept(accepting.socket, listener)?;
                    Ok(accepting.socket)
                });
                match accepted {
                    Ok(socket) => Ok(socket),
                    Err(error) => {
                        self.sys.close_socket(accepting.socket);
                        Err(error)
                    }
                }
            }
            Kind::Connect { sock } => ended.and_then(|_| {
                self.sys.update_connect(sock)?;
                Ok(0)
            }),
            _ => ended,
        };
        Some(conclude(*rec, ended))
    }

    /// A record leaves the table by its address; `None` for a packet of no operation.
    fn take(&self, addr: usize) -> Option<Box<Rec>> {
        lock(&self.ops).remove(&addr)
    }

    fn next_packet(&self) -> Option<Packet> {
        let mut inbox = lock(&self.inbox);
        if inbox.is_empty() {
            let mut taken = Vec::new();
            if self
                .sys
                .dequeue(self.port.handle, &mut taken, BATCH)
                .is_err()
            {
                return None;
            }
            inbox.extend(taken);
        }
        inbox.pop_front()
    }

    /// The next completion of `ring`, if the ring is not busy.
    fn pop(&self, ring: &Ring) -> Option<Cqe> {
        let Ok(_inner) = ring.inner.try_lock() else {
            return None;
        };
        match self.sys.ring_pop(ring.handle) {
            Ok(Some(cqe)) => {
                ring.in_kernel.fetch_sub(1, Ordering::AcqRel);
                Some(cqe)
            }
            Ok(None) | Err(_) => None,
        }
    }
}

/// The continuation's frame for an ended operation.
fn conclude(rec: Rec, result: io::Result<usize>) -> Box<dyn Frame> {
    let Rec {
        kind,
        mut buf,
        then,
        ..
    } = rec;
    match then.expect("a continuation, once") {
        Then::Io(then) => {
            if matches!(kind, Kind::Read { .. } | Kind::Recv { .. })
                && let Ok(n) = result
            {
                buf.filled = n;
            }
            then(IoDone { buf, result })
        }
        Then::Raw(then) => then(result),
    }
}

impl EngineTrait for Engine {
    fn submit(&self, request: Request, buf: IoBuf, then: Then) -> io::Result<()> {
        let sys = &*self.sys;
        let port = self.port.handle;
        let rec = match request {
            Request::Read { file, offset } => {
                self.check_file(file)?;
                let mut rec = Self::rec(Kind::Read { file, offset }, buf, then);
                self.route(&mut rec, file, self.ring_reads)?;
                rec
            }
            Request::Write { file, offset } => {
                self.check_file(file)?;
                let mut rec = Self::rec(Kind::Write { file, offset }, buf, then);
                self.route(&mut rec, file, self.ring_writes)?;
                rec
            }
            Request::Send { sock } => {
                sys.associate(port, sock)?;
                Self::rec(Kind::Send { sock }, buf, then)
            }
            Request::Recv { sock } => {
                sys.associate(port, sock)?;
                Self::rec(Kind::Recv { sock }, buf, then)
            }
            Request::SendTo { sock, to } => {
                sys.associate(port, sock)?;
                let mut rec = Self::rec(Kind::SendTo { sock }, buf, then);
                rec.addr = Some(Box::new(SockAddr::new(to)));
                rec
            }
            Request::Accept { listener } => {
                sys.associate(port, listener)?;
                let socket = sys.accept_socket(listener)?;
                let mut rec = Self::rec(Kind::Accept { listener }, buf, then);
                rec.accepting = Some(Accepting {
                    socket,
                    addresses: Box::new([0; ACCEPT_BUF]),
                });
                rec
            }
            Request::Connect { sock, to } => {
                sys.associate(port, sock)?;
                // `ConnectEx` needs a bound socket.
                sys.bind_any(sock, &SockAddr::unspecified(to))?;
                let mut rec = Self::rec(Kind::Connect { sock }, buf, then);
                rec.addr = Some(Box::new(SockAddr::new(to)));
                rec
            }
            Request::Open { path, spec } => {
                let mut rec = Self::rec(Kind::Open { path, spec }, buf, then);
                rec.pool = Some(Arc::default());
                rec
            }
            Request::Rename { from, to } => {
                let mut rec = Self::rec(Kind::Rename { from, to }, buf, then);
                rec.pool = Some(Arc::default());
                rec
            }
            #[cfg(any(target_os = "linux", target_os = "android"))]
            Request::OpenAt { .. } => return Err(invalid("open_file is not available here")),
        };
        self.start(rec);
        Ok(())
    }

    fn poll(&self, room: &mut dyn FnMut() -> bool, offer: &mut dyn FnMut(Box<dyn Frame>) -> bool) {
        // Operations whose call failed at once.
        while room() {
            let Some(mut rec) = lock(&self.ready).pop_front() else {
                break;
            };
            let error = rec.failed.take().expect("a failed operation's error");
            if let Some(frame) = self.advance_failed(rec, error)
                && !offer(frame)
            {
                return;
            }
        }
        // The port.
        while room() {
            let Some(packet) = self.next_packet() else {
                break;
            };
            // A packet of no operation is ignored (it cannot be told from a record's end).
            let Some(mut rec) = self.take(packet.overlapped) else {
                continue;
            };
            let result = self.packet_result(&mut rec);
            if let Some(frame) = self.advance(rec, result)
                && !offer(frame)
            {
                return;
            }
        }
        // The rings.
        let rings: Vec<Arc<Ring>> = lock(&self.rings).clone();
        for ring in &rings {
            self.flush_ring(ring);
            while room() {
                let Some(cqe) = self.pop(ring) else {
                    break;
                };
                let Some(rec) = self.take(cqe.user_data) else {
                    continue;
                };
                let result = if cqe.result >= 0 {
                    Ok(cqe.info)
                } else {
                    Err(hresult_error(cqe.result))
                };
                if let Some(frame) = self.advance(rec, result)
                    && !offer(frame)
                {
                    self.flush_ring(ring);
                    return;
                }
            }
            // Room freed in the completion queue may let the backlog in.
            self.flush_ring(ring);
        }
    }
}

impl Engine {
    /// An operation that failed at its call ends as any other that failed.
    fn advance_failed(&self, rec: Box<Rec>, error: io::Error) -> Option<Box<dyn Frame>> {
        self.advance(rec, Err(error))
    }
}

impl AsyncIo {
    /// Asynchronous I/O on this machine: IoRing for file reads and writes where the OS has a
    /// kernel ring, and completion ports for the rest.
    #[cfg(windows)]
    pub fn open() -> io::Result<Arc<Self>> {
        Self::open_with_options(AsyncIoOptions::default())
    }

    /// As [`open`](Self::open), with options.
    #[cfg(windows)]
    pub fn open_with_options(options: AsyncIoOptions) -> io::Result<Arc<Self>> {
        let sys: Arc<dyn Sys> = Arc::new(real::RealSys::new());
        Ok(Self::from_engine(Engine::new(sys, options, MAX_PIECE)?))
    }
}
