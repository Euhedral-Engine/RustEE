//! The io_uring backend of [`AsyncIo`], for Linux and Android.
//!
//! The only code of the crate besides [`ThreadUsage`](crate::ThreadUsage) with unsafe code: the
//! io_uring rings (`// SAFETY:` on every block).
#![allow(unsafe_code)]

use std::cell::RefCell;
use std::collections::VecDeque;
use std::ffi::CString;
use std::io;
use std::os::fd::RawFd;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, Weak};

use euhedral_core::frames::Frame;
use euhedral_data_structures::queues::{Consumer, MpscQueue, Multi, Producer, Single};
use io_uring::{IoUring, Submitter, opcode, squeue, types};

use super::{
    AsyncIo, AsyncIoOptions, Buf, Create, Engine, MAX_PIECE, OpenSpec, READ_PIECE, Request, Then,
    conclude as conclude_io, lock,
};

/// Entries of each ring's submission queue (its completion queue has twice as many).
const RING_ENTRIES: u32 = 256;
/// Slots per chunk of a ring's queue of entries that wait for `sq`.
const INCOMING_CHUNK: usize = 256;
/// Submitting threads (rings) one `AsyncIo` serves.
const MAX_RINGS: usize = 256;

/// Replaces `io_uring_enter` (the submission of a ring's queued entries): the native call, or a
/// test's stand-in. It returns how many entries the kernel took, or why it refused; refused entries
/// stay queued and the next enter on the ring (a later submission or poll) hands them over.
type Enter = dyn Fn(&Submitter<'_>) -> io::Result<usize> + Send + Sync;

/// The io_uring engine: a ring per submitting thread.
pub(super) struct Uring {
    id: usize,
    rings: Mutex<Vec<Arc<Ring>>>,
    enter: Box<Enter>,
    /// The most bytes one read, write or send hands the kernel; a test's stand-in lowers it to
    /// force short ones.
    max_piece: usize,
}

static NEXT_ID: AtomicUsize = AtomicUsize::new(1);

thread_local! {
    /// This thread's ring of each `AsyncIo` it submitted to (by id).
    static MINE: RefCell<Vec<(usize, Weak<Ring>)>> = const { RefCell::new(Vec::new()) };
}

/// One submitting thread's ring and the operations in flight on it.
struct Ring {
    uring: IoUring,
    /// Pushing entries and entering the kernel: the owning thread, and a poll that resubmits the
    /// rest of a short read or hands over entries the kernel refused or had no room for.
    sq: Mutex<Queue>,
    /// Entries queued while another thread held `sq` (a poll chaining a read's pieces, each copied
    /// inside its enter: a lane's small read waited 1.4 ms for the lock). Lock-free: any thread
    /// appends; the `sq` holder drains it (the consumer lives in [`Queue`]) before it lets go, and
    /// looks again after it does (see [`Ring::flush_waiting`]).
    incoming: Producer<squeue::Entry, Multi>,
    /// Set after a push to `incoming`, once the entry is published; the `sq` holder clears it
    /// before it drains, and looks again after it lets go.
    waiting: AtomicBool,
    /// Reaping: one poll at a time (the lattice's guarantee; the lock makes it a fact).
    cq: Mutex<()>,
    ops: Mutex<Ops>,
    /// Entries handed to the submission queue and not yet reaped; at most the completion queue's
    /// size, so the completion queue never overflows.
    in_kernel: AtomicUsize,
    capacity: usize,
}

struct Queue {
    /// The drain side of [`Ring::incoming`]: only the `sq` holder has it.
    incoming: Consumer<squeue::Entry, Single>,
    /// Entries waiting for room in the kernel's queues, the oldest first.
    backlog: VecDeque<squeue::Entry>,
    /// Entries are in the submission queue that the kernel has not taken (an enter was refused).
    unsubmitted: bool,
}

/// The operations in flight on a ring, by slot; `user_data` is the slot and its generation.
#[derive(Default)]
struct Ops {
    slots: Vec<Option<Op>>,
    generations: Vec<u32>,
    free: Vec<usize>,
}

impl Ops {
    fn insert(&mut self, op: Op) -> u64 {
        let slot = self.free.pop().unwrap_or_else(|| {
            self.slots.push(None);
            self.generations.push(0);
            self.slots.len() - 1
        });
        self.generations[slot] = self.generations[slot].wrapping_add(1);
        self.slots[slot] = Some(op);
        (u64::from(self.generations[slot]) << 32) | slot as u64
    }

    fn get(&mut self, user_data: u64) -> Option<&mut Op> {
        let slot = (user_data & u64::from(u32::MAX)) as usize;
        if self.generations.get(slot).copied() != Some((user_data >> 32) as u32) {
            return None;
        }
        self.slots.get_mut(slot)?.as_mut()
    }

    fn remove(&mut self, user_data: u64) -> Option<Op> {
        let slot = (user_data & u64::from(u32::MAX)) as usize;
        let op = self.slots.get_mut(slot)?.take()?;
        self.free.push(slot);
        Some(op)
    }
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
    /// The path stays in its own heap block (a `CString`) where the kernel reads it.
    OpenAt {
        path: CString,
        flags: i32,
        mode: libc::mode_t,
    },
    RenameAt {
        from: CString,
        to: CString,
    },
    /// The address is boxed so that it stays where the kernel reads it.
    SendTo {
        fd: RawFd,
        addr: Box<libc::sockaddr_storage>,
        len: libc::socklen_t,
    },
    /// As `SendTo`, the address is boxed.
    Connect {
        fd: RawFd,
        addr: Box<libc::sockaddr_storage>,
        len: libc::socklen_t,
    },
}

/// An operation in flight. Its buffer stays put (the `Vec`'s heap block) until the operation is
/// reaped.
struct Op {
    kind: Kind,
    buf: Buf,
    /// Bytes moved so far (reads and writes resubmit the rest).
    moved: usize,
    then: Then,
}

impl Op {
    /// The submission entry for the operation's next piece.
    fn entry(&mut self, user_data: u64, max_piece: usize) -> squeue::Entry {
        let piece = |n: usize| u32::try_from(n.min(max_piece)).unwrap_or(u32::MAX);
        let entry = match &self.kind {
            Kind::Read { fd, offset } => {
                let rest = &mut self.buf.bytes_mut()[self.moved..];
                // A read the page cache can serve is copied inside `io_uring_enter`, on the
                // submitting worker: about 0.2 ms per MB (a read of several MB copied in one
                // completion held a worker for a millisecond). So a read goes in pieces of at most
                // READ_PIECE, each piece's completion submitting the next (`IOSQE_ASYNC` instead
                // made the submitting worker create an io worker thread now and then: worse).
                let len = piece(rest.len().min(READ_PIECE));
                opcode::Read::new(types::Fd(*fd), rest.as_mut_ptr(), len)
                    .offset(offset + self.moved as u64)
                    .build()
            }
            Kind::Write { fd, offset } => {
                let rest = &self.buf.bytes()[self.moved..self.buf.filled()];
                opcode::Write::new(types::Fd(*fd), rest.as_ptr(), piece(rest.len()))
                    .offset(offset + self.moved as u64)
                    .build()
            }
            Kind::Send { fd } => {
                // Copied into the socket inside `io_uring_enter` too: pieces of at most READ_PIECE,
                // as reads.
                let bytes = &self.buf.bytes()[self.moved..self.buf.filled()];
                opcode::Send::new(
                    types::Fd(*fd),
                    bytes.as_ptr(),
                    piece(bytes.len().min(READ_PIECE)),
                )
                .flags(libc::MSG_NOSIGNAL)
                .build()
            }
            Kind::Recv { fd } => {
                let bytes = self.buf.bytes_mut();
                opcode::Recv::new(types::Fd(*fd), bytes.as_mut_ptr(), piece(bytes.len())).build()
            }
            Kind::Accept { fd } => {
                opcode::Accept::new(types::Fd(*fd), std::ptr::null_mut(), std::ptr::null_mut())
                    .flags(libc::SOCK_CLOEXEC)
                    .build()
            }
            Kind::OpenAt { path, flags, mode } => {
                opcode::OpenAt::new(types::Fd(libc::AT_FDCWD), path.as_ptr())
                    .flags(*flags | libc::O_CLOEXEC)
                    .mode(*mode)
                    .build()
            }
            Kind::RenameAt { from, to } => opcode::RenameAt::new(
                types::Fd(libc::AT_FDCWD),
                from.as_ptr(),
                types::Fd(libc::AT_FDCWD),
                to.as_ptr(),
            )
            .build(),
            Kind::SendTo { fd, addr, len } => {
                let bytes = &self.buf.bytes()[..self.buf.filled()];
                opcode::Send::new(types::Fd(*fd), bytes.as_ptr(), piece(bytes.len()))
                    .flags(libc::MSG_NOSIGNAL)
                    .dest_addr(std::ptr::from_ref(&**addr).cast())
                    .dest_addr_len(*len)
                    .build()
            }
            Kind::Connect { fd, addr, len } => {
                opcode::Connect::new(types::Fd(*fd), std::ptr::from_ref(&**addr).cast(), *len)
                    .build()
            }
        };
        entry.user_data(user_data)
    }
}

/// What a reaped completion means for its operation.
enum Reaped {
    /// It ended: its continuation's frame goes out.
    Ended(Op, io::Result<usize>),
    /// A short read or write: the rest goes in again.
    Again(squeue::Entry),
    /// A completion of no operation (never expected).
    Stray,
}

impl Ring {
    fn new() -> io::Result<Self> {
        let uring = IoUring::builder().setup_clamp().build(RING_ENTRIES)?;
        let capacity = uring.params().cq_entries() as usize;
        let incoming: MpscQueue<squeue::Entry> = MpscQueue::with_pool(INCOMING_CHUNK, 2);
        Ok(Self {
            uring,
            sq: Mutex::new(Queue {
                incoming: incoming.consumer(),
                backlog: VecDeque::new(),
                unsubmitted: false,
            }),
            incoming: incoming.producer(),
            waiting: AtomicBool::new(false),
            cq: Mutex::new(()),
            ops: Mutex::new(Ops::default()),
            in_kernel: AtomicUsize::new(0),
            capacity,
        })
    }

    /// Queues `entry` and hands what fits to the kernel: here, or, when another thread is
    /// submitting on the ring right now, by that thread before it lets go (never waiting for it).
    fn queue(&self, entry: squeue::Entry, enter: &Enter) {
        // Cannot fail: the queue is unbounded. Were it ever to, the entry would be lost with its
        // operation still in `ops`: say so loudly.
        let queued = self.incoming.offer(entry);
        assert!(
            queued.is_ok(),
            "the unbounded incoming queue refused an entry"
        );
        // After the publish: a holder that reads this `true` finds the entry.
        self.waiting.store(true, Ordering::SeqCst);
        self.flush_waiting(enter);
    }

    /// Moves backlog entries into the submission queue while the completion queue has room for
    /// them, and enters the kernel when there is anything for it.
    fn flush(&self, queue: &mut Queue, enter: &Enter) {
        let mut pushed = false;
        if !queue.backlog.is_empty() {
            // SAFETY: the `sq` lock is held, so this is the only `SubmissionQueue` of the ring.
            let mut sq = unsafe { self.uring.submission_shared() };
            while self.in_kernel.load(Ordering::Acquire) < self.capacity {
                let Some(entry) = queue.backlog.front() else {
                    break;
                };
                // SAFETY: every pointer in the entry is into its operation's buffer (or its boxed
                // address), which the ring's `ops` own and leave untouched until the operation is
                // reaped; a ring dropped with operations in flight leaks them (`Drop for Ring`).
                // Accept passes null address pointers.
                if unsafe { sq.push(entry) }.is_err() {
                    break;
                }
                queue.backlog.pop_front();
                self.in_kernel.fetch_add(1, Ordering::AcqRel);
                pushed = true;
            }
            sq.sync();
        }
        if pushed || queue.unsubmitted {
            queue.unsubmitted = enter(&self.uring.submitter()).is_err();
        }
    }

    /// Hands over what waits (queued entries, a backlog, or entries an enter refused), unless
    /// another thread is submitting right now: it then hands them over itself.
    ///
    /// # Why no queued entry is stranded
    ///
    /// A pusher P appends to `incoming` (the queue publishes the entry with a `Release` store when
    /// `offer` returns), then sets `waiting` (`SeqCst`), then runs a `SeqCst` fence and
    /// `try_lock`s `sq`. If it gets `sq`, it drains `incoming`, which holds its own entry, and is
    /// done. If it fails, the holder H it read as locked unlocks later in the lock word's order,
    /// then runs a `SeqCst` fence and reads `waiting`. The two fences are ordered one way or the
    /// other in the single `SeqCst` order:
    /// - H's fence first: H's unlock precedes it, so P's `try_lock` after its own fence would
    ///   have seen the lock free (or taken by a later holder). P read it locked at H's time, so
    ///   this case cannot happen for H, nor for any later holder (each unlocks after P's read).
    /// - P's fence first: P's store to `waiting` precedes it, so H's read after its fence sees
    ///   `true` (or the `false` of a later holder's swap, which took the entry: below). H then
    ///   loops: it takes `sq` and drains, or fails to, and then another holder is in the same
    ///   position, with the same two cases.
    ///
    /// A holder clears `waiting` (a `SeqCst` swap) before it drains. The swap that reads P's
    /// `true` synchronises with P's store, which follows P's publish, so the drain after the swap
    /// sees P's entry. A `true` stored after the swap is left for the holder's re-check.
    ///
    /// `waiting` is set after the entry is published, not when its slot is reserved. A queue's
    /// size counts reserved slots, so a re-check by size alone could not tell a producer between
    /// its reservation and its publish from an entry the holder's drain missed, and would have to
    /// wait for the producer (nothing here spins). That producer sets `waiting` itself once it has
    /// published.
    fn flush_waiting(&self, enter: &Enter) {
        loop {
            // Pairs with the fence after a holder lets go (see above): either this thread gets
            // `sq`, or the holder sees what this thread queued.
            std::sync::atomic::fence(Ordering::SeqCst);
            let Ok(mut queue) = self.sq.try_lock() else {
                return;
            };
            // The swap reads the pusher's `true`, so its entry is published for the drain below.
            self.waiting.swap(false, Ordering::SeqCst);
            let Queue {
                incoming, backlog, ..
            } = &mut *queue;
            incoming.drain_all(|entry| backlog.push_back(entry));
            if queue.unsubmitted || !queue.backlog.is_empty() {
                self.flush(&mut queue, enter);
            }
            drop(queue);
            std::sync::atomic::fence(Ordering::SeqCst);
            // Queued while this thread held `sq` (its pusher's try_lock failed): take it now.
            if !self.waiting.load(Ordering::SeqCst) {
                return;
            }
        }
    }

    fn reap_one(&self, user_data: u64, result: i32, max_piece: usize) -> Reaped {
        let mut ops = lock(&self.ops);
        let Some(op) = ops.get(user_data) else {
            return Reaped::Stray;
        };
        let ended = match op.kind {
            Kind::Read { .. } | Kind::Write { .. } | Kind::Send { .. } if result > 0 => {
                op.moved += result as usize;
                let total = if matches!(op.kind, Kind::Read { .. }) {
                    op.buf.len()
                } else {
                    op.buf.filled()
                };
                if op.moved < total {
                    return Reaped::Again(op.entry(user_data, max_piece));
                }
                Ok(op.moved)
            }
            Kind::Read { .. } if result == 0 => {
                if op.moved == 0 {
                    Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "the read is at or past the end of the file",
                    ))
                } else {
                    Ok(op.moved)
                }
            }
            Kind::Write { .. } if result == 0 => Err(io::Error::from(io::ErrorKind::WriteZero)),
            Kind::Send { .. } if result == 0 && op.buf.filled() > 0 => {
                Err(io::Error::from(io::ErrorKind::WriteZero))
            }
            _ if result < 0 => Err(io::Error::from_raw_os_error(-result)),
            _ => Ok(result as usize),
        };
        let op = ops.remove(user_data).expect("the operation just found");
        Reaped::Ended(op, ended)
    }
}

impl Drop for Ring {
    fn drop(&mut self) {
        // The kernel may still write into the buffers of operations in flight while the ring tears
        // down: leak them.
        let ops = std::mem::take(&mut *lock(&self.ops));
        for op in ops.slots.into_iter().flatten() {
            std::mem::forget(op);
        }
    }
}

/// The continuation's frame for an ended operation.
fn conclude(op: Op, result: io::Result<usize>) -> Box<dyn Frame> {
    let Op {
        kind, buf, then, ..
    } = op;
    let fills = matches!(kind, Kind::Read { .. } | Kind::Recv { .. });
    conclude_io(buf, then, fills, result)
}

fn c_path(path: &std::path::Path) -> io::Result<CString> {
    use std::os::unix::ffi::OsStrExt;
    CString::new(path.as_os_str().as_bytes())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "a path with a NUL byte"))
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

impl Uring {
    /// An engine with `enter` in place of `io_uring_enter` and pieces of at most `max_piece` bytes
    /// (tests: the stand-in backend).
    fn with(enter: Box<Enter>, max_piece: usize) -> io::Result<Self> {
        // A ring proves the kernel allows io_uring before anything relies on it.
        drop(Ring::new()?);
        Ok(Self {
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            rings: Mutex::new(Vec::new()),
            enter,
            max_piece: max_piece.clamp(1, MAX_PIECE),
        })
    }

    /// The calling thread's ring, made on first use.
    fn ring(&self) -> io::Result<Arc<Ring>> {
        MINE.with_borrow_mut(|mine| {
            mine.retain(|(_, ring)| ring.strong_count() > 0);
            if let Some(ring) = mine
                .iter()
                .find(|(id, _)| *id == self.id)
                .and_then(|(_, ring)| ring.upgrade())
            {
                return Ok(ring);
            }
            let mut rings = lock(&self.rings);
            if rings.len() >= MAX_RINGS {
                return Err(io::Error::other("too many threads submit to this AsyncIo"));
            }
            let ring = Arc::new(Ring::new()?);
            rings.push(ring.clone());
            mine.push((self.id, Arc::downgrade(&ring)));
            Ok(ring)
        })
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
            Request::OpenAt { path, flags, mode } => Kind::OpenAt {
                path: c_path(&path)?,
                flags,
                mode: mode as libc::mode_t,
            },
            Request::Open { path, spec } => Kind::OpenAt {
                path: c_path(&path)?,
                flags: open_flags(spec),
                mode: spec.unix_mode as libc::mode_t,
            },
            Request::Rename { from, to } => Kind::RenameAt {
                from: c_path(&from)?,
                to: c_path(&to)?,
            },
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
                }
            }
        })
    }
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

impl Engine for Uring {
    fn submit(&self, request: Request, buf: Buf, then: Then) -> io::Result<()> {
        let kind = Self::kind(request)?;
        let ring = self.ring()?;
        let entry = {
            let mut ops = lock(&ring.ops);
            let user_data = ops.insert(Op {
                kind,
                buf,
                moved: 0,
                then,
            });
            ops.get(user_data)
                .expect("just inserted")
                .entry(user_data, self.max_piece)
        };
        ring.queue(entry, &*self.enter);
        Ok(())
    }

    fn poll(&self, room: &mut dyn FnMut() -> bool, offer: &mut dyn FnMut(Box<dyn Frame>) -> bool) {
        let rings: Vec<Arc<Ring>> = lock(&self.rings).clone();
        for ring in &rings {
            ring.flush_waiting(&*self.enter);
            let Ok(_reaping) = ring.cq.try_lock() else {
                continue;
            };
            loop {
                if !room() {
                    break;
                }
                let next = {
                    // SAFETY: the `cq` lock is held, so this is the only `CompletionQueue` of the
                    // ring. It is dropped (publishing the new head) before anything else runs.
                    let mut cq = unsafe { ring.uring.completion_shared() };
                    cq.next().map(|cqe| (cqe.user_data(), cqe.result()))
                };
                let Some((user_data, result)) = next else {
                    break;
                };
                ring.in_kernel.fetch_sub(1, Ordering::AcqRel);
                match ring.reap_one(user_data, result, self.max_piece) {
                    Reaped::Ended(op, result) => {
                        if !offer(conclude(op, result)) {
                            ring.flush_waiting(&*self.enter);
                            return;
                        }
                    }
                    Reaped::Again(entry) => ring.queue(entry, &*self.enter),
                    Reaped::Stray => {
                        debug_assert!(false, "a completion ({user_data:#x}) of no operation")
                    }
                }
            }
            // Room freed in the completion queue may let the backlog in.
            ring.flush_waiting(&*self.enter);
        }
    }
}

impl AsyncIo {
    /// Asynchronous I/O on this machine, or why there is none (no io_uring).
    pub fn open() -> io::Result<Arc<Self>> {
        Self::open_with_options(AsyncIoOptions::default())
    }

    /// As [`open`](Self::open), with options (none of them applies to io_uring).
    pub fn open_with_options(_options: AsyncIoOptions) -> io::Result<Arc<Self>> {
        Self::open_with(
            Box::new(|submitter: &Submitter<'_>| submitter.submit()),
            MAX_PIECE,
        )
    }

    /// As [`open`](Self::open), with `enter` in place of `io_uring_enter` and pieces of at most
    /// `max_piece` bytes (tests: the stand-in backend).
    fn open_with(enter: Box<Enter>, max_piece: usize) -> io::Result<Arc<Self>> {
        Ok(Self::from_engine(Uring::with(enter, max_piece)?))
    }
}

/// The port of Euhedral-Inference's `AsyncReadsTest`, extended to writes, sockets and datagrams.
/// The stand-in for `io_uring_enter` replaces Java's `Submitter` stand-in.
#[cfg(test)]
#[allow(clippy::disallowed_methods)] // Test fixtures write files and open sockets synchronously, outside any lattice.
mod tests {
    use std::io::{Read, Write};
    use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
    use std::path::PathBuf;
    use std::time::{Duration, Instant};

    use euhedral_core::frames::{FrameHeader, RunnableFrame};
    use euhedral_core::generics::{LatticeReceiver, LatticeSource};

    use super::super::{Continuation, IoBuf, IoDone, IoDoneOwned, IoMemory, IoThen};
    use super::*;

    /// A completion frame: carries what the operation ended with.
    struct Done {
        header: FrameHeader,
        done: Option<IoDone>,
        accepted: Option<io::Result<OwnedFd>>,
    }

    impl Frame for Done {
        fn header(&self) -> &FrameHeader {
            &self.header
        }

        fn header_mut(&mut self) -> &mut FrameHeader {
            &mut self.header
        }
    }

    fn then(tag: u64) -> IoThen {
        Box::new(move |done| {
            Box::new(Done {
                header: FrameHeader::new(tag),
                done: Some(done),
                accepted: None,
            })
        })
    }

    /// Polls as a worker does until `count` completions came out (10 s at most).
    fn poll(io: &AsyncIo, count: usize) -> Vec<(u64, Box<Done>)> {
        let mut out = Vec::new();
        let deadline = Instant::now() + Duration::from_secs(10);
        while out.len() < count {
            assert!(Instant::now() < deadline, "the operations did not complete");
            io.pull(
                &mut |frame| {
                    let tag = frame.header().id_hash();
                    let done = (frame as Box<dyn std::any::Any>)
                        .downcast::<Done>()
                        .expect("a Done frame");
                    out.push((tag, done));
                },
                &mut |_| false,
                u64::MAX,
            );
        }
        out
    }

    struct TempFile(PathBuf);

    impl Drop for TempFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    /// A file of `content` in the crate's scratch directory under the target directory.
    fn temp_file(name: &str, content: &[u8]) -> (TempFile, std::fs::File) {
        let exe = std::env::current_exe().expect("the test binary's path");
        // The test binary is `target/<profile>/deps/<name>`: scratch goes beside it, never /tmp or
        // $HOME.
        let dir = exe
            .parent()
            .and_then(std::path::Path::parent)
            .expect("a target directory")
            .join("euhedral-graph-scratch");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(format!(
            "{name}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::write(&path, content).unwrap();
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .unwrap();
        (TempFile(path), file)
    }

    fn content(len: usize, seed: u64) -> Vec<u8> {
        let mut state = seed | 1;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state as u8
            })
            .collect()
    }

    fn buf(len: usize) -> IoBuf {
        IoBuf {
            bytes: vec![0; len],
            filled: 0,
        }
    }

    /// AsyncReadsTest.aReadsCompletionIsAFrameEmittedWhenTheSinkIsPolledAndItsBytesArrived
    #[test]
    fn a_reads_completion_is_a_frame_emitted_when_polled_after_its_bytes_arrived() {
        let data = content(3 * 4096 + 123, 3);
        let (_guard, file) = temp_file("data", &data);
        let io = AsyncIo::open().unwrap();
        io.read(file.as_raw_fd(), 0, buf(4096), then(1)).unwrap();
        io.read(file.as_raw_fd(), 4096, buf(data.len() - 4096), then(2))
            .unwrap();
        let mut out = poll(&io, 2);
        out.sort_by_key(|(tag, _)| *tag);
        let a = out[0].1.done.as_ref().unwrap();
        let b = out[1].1.done.as_ref().unwrap();
        assert_eq!(a.result.as_ref().unwrap(), &4096);
        assert_eq!(b.result.as_ref().unwrap(), &(data.len() - 4096));
        assert_eq!(a.buf.bytes, data[..4096]);
        assert_eq!(&b.buf.bytes[..b.buf.filled], &data[4096..]);
        assert_eq!(io.pending(), 0);
    }

    /// One shared allocation, as a pinned host arena is: parts of it go to readers, each reaching its
    /// bytes through the raw base pointer (never through a reference to the whole).
    struct Arena {
        base: *mut u8,
        len: usize,
        capacity: usize,
    }

    // SAFETY: the base pointer is only read; the bytes are reached through parts that are disjoint
    // while their reads run, and through `bytes` after all parts came back.
    unsafe impl Send for Arena {}
    // SAFETY: as for `Send`.
    unsafe impl Sync for Arena {}

    impl Arena {
        fn new(bytes: Vec<u8>) -> Self {
            let mut bytes = std::mem::ManuallyDrop::new(bytes);
            Self {
                base: bytes.as_mut_ptr(),
                len: bytes.len(),
                capacity: bytes.capacity(),
            }
        }

        /// All of it; only when no part is out.
        fn bytes(&self) -> &[u8] {
            // SAFETY: `len` initialized bytes of the block; the caller has every part back.
            unsafe { std::slice::from_raw_parts(self.base, self.len) }
        }
    }

    impl Drop for Arena {
        fn drop(&mut self) {
            // SAFETY: the parts of `new`'s vector, not used since.
            drop(unsafe { Vec::from_raw_parts(self.base, self.len, self.capacity) });
        }
    }

    struct ArenaPart {
        arena: Arc<Arena>,
        offset: usize,
        len: usize,
    }

    // SAFETY: the part addresses `len` initialized bytes of the arena's block, which does not move
    // or go away while an `Arc` of it exists; it holds no reference to them, only a pointer derived
    // from the arena's base; the parts of this test are disjoint, and nothing else touches the arena
    // until they all came back.
    unsafe impl IoMemory for ArenaPart {
        fn as_mut_ptr(&mut self) -> *mut u8 {
            // SAFETY: `offset` is inside the block.
            unsafe { self.arena.base.add(self.offset) }
        }

        fn len(&self) -> usize {
            self.len
        }
    }

    /// Two reads into disjoint parts of one arena, each cut into short pieces, land at their own
    /// offsets of the arena, without a copy.
    #[test]
    fn reads_into_parts_of_one_arena_land_in_place() {
        let data = content(3 * 4096, 11);
        let (_guard, file) = temp_file("arena", &data);
        let io = AsyncIo::open_with(Box::new(|s: &Submitter<'_>| s.submit()), 1000).unwrap();
        let arena = Arc::new(Arena::new(vec![0xEE; 2 * 4096 + 100]));
        let returned = Arc::new(Mutex::new(Vec::new()));
        // The first part reads the file's last page and the second its first, in the arena's
        // reverse order; the part in between stays untouched.
        for (offset, len, from) in [(0, 4096, 2 * 4096), (4096 + 100, 4096, 0)] {
            let part = ArenaPart {
                arena: arena.clone(),
                offset,
                len,
            };
            let returned = returned.clone();
            io.read_into(
                file.as_raw_fd(),
                from,
                part,
                Box::new(move |done: IoDoneOwned<ArenaPart>| {
                    lock(&returned).push((done.buf.expect("memory back"), done.result.unwrap()));
                    Box::new(RunnableFrame::new(1, Arc::new(|| {})))
                }),
            )
            .unwrap();
        }
        let deadline = Instant::now() + Duration::from_secs(10);
        while io.pending() > 0 {
            assert!(Instant::now() < deadline, "the reads did not complete");
            io.pull(&mut |frame| drop(frame), &mut |_| false, u64::MAX);
        }
        let returned = std::mem::take(&mut *lock(&returned));
        assert_eq!(returned.len(), 2);
        assert!(returned.iter().all(|(_, n)| *n == 4096));
        drop(returned);
        let arena = arena.bytes();
        assert_eq!(arena[..4096], data[2 * 4096..]);
        assert!(arena[4096..4096 + 100].iter().all(|&b| b == 0xEE));
        assert_eq!(arena[4096 + 100..], data[..4096]);
    }

    /// AsyncReadsTest.readsFromManyThreadsEachCompleteOnce: 8 threads submit on rings of their own;
    /// each stays alive (polling, one poller at a time) until every read completed, since io_uring
    /// cancels an exited thread's operations.
    #[test]
    fn reads_from_many_threads_each_complete_once() {
        let data = Arc::new(content(64 * 4096, 5));
        let (_guard, file) = temp_file("many", &data);
        let io = AsyncIo::open().unwrap();
        let fd = file.as_raw_fd();
        let results = Arc::new(Mutex::new(Vec::new()));
        let poller = Arc::new(Mutex::new(()));
        let threads: Vec<_> = (0..8u64)
            .map(|t| {
                let (io, results, poller) = (io.clone(), results.clone(), poller.clone());
                std::thread::spawn(move || {
                    for i in t * 8..t * 8 + 8 {
                        io.read(fd, 4096 * i, buf(4096), then(i)).unwrap();
                    }
                    let deadline = Instant::now() + Duration::from_secs(10);
                    while lock(&results).len() < 64 {
                        assert!(Instant::now() < deadline, "the reads did not complete");
                        let _one_poller = lock(&poller);
                        io.pull(
                            &mut |frame| lock(&results).push(frame),
                            &mut |_| false,
                            u64::MAX,
                        );
                    }
                })
            })
            .collect();
        for thread in threads {
            thread.join().unwrap();
        }
        let mut results: Vec<(u64, Box<Done>)> = std::mem::take(&mut *lock(&results))
            .into_iter()
            .map(|f| {
                (
                    f.header().id_hash(),
                    (f as Box<dyn std::any::Any>).downcast::<Done>().unwrap(),
                )
            })
            .collect();
        results.sort_by_key(|(tag, _)| *tag);
        assert_eq!(
            results.iter().map(|(tag, _)| *tag).collect::<Vec<_>>(),
            (0..64).collect::<Vec<_>>()
        );
        for (i, done) in results {
            let done = done.done.unwrap();
            assert_eq!(done.result.unwrap(), 4096);
            assert_eq!(
                done.buf.bytes,
                data[4096 * i as usize..4096 * (i as usize + 1)]
            );
        }
        assert_eq!(
            lock(&io.engine.rings).len(),
            8,
            "a ring per submitting thread"
        );
        assert_eq!(io.pending(), 0);
    }

    /// AsyncReadsTest.aReadPastTheEndOfTheFileFails
    #[test]
    fn a_read_past_the_end_of_the_file_fails() {
        let (_guard, file) = temp_file("short", &[0; 100]);
        let io = AsyncIo::open().unwrap();
        io.read(file.as_raw_fd(), 8192, buf(4096), then(9)).unwrap();
        let done = poll(&io, 1).pop().unwrap().1.done.unwrap();
        assert_eq!(
            done.result.unwrap_err().kind(),
            io::ErrorKind::UnexpectedEof
        );
    }

    /// The file ends inside the read: it completes with the bytes there were (a read of "up to" its
    /// buffer).
    #[test]
    fn a_read_the_file_ends_inside_completes_with_the_bytes_there_were() {
        let data = content(4096 + 100, 13);
        let (_guard, file) = temp_file("tail", &data);
        let io = AsyncIo::open().unwrap();
        io.read(file.as_raw_fd(), 0, buf(2 * 4096), then(1))
            .unwrap();
        let done = poll(&io, 1).pop().unwrap().1.done.unwrap();
        assert_eq!(done.result.unwrap(), data.len());
        assert_eq!(&done.buf.bytes[..done.buf.filled], &data[..]);
    }

    /// AsyncReadsTest.aRefusedSubmissionLeavesNoEntryThatReadsOrCompletesLater, as this port does
    /// it: a refused enter leaves the operation queued (the caller has handed its buffer over), and
    /// a later poll hands it to the kernel; it completes once, with its bytes, and nothing else
    /// completes.
    #[test]
    fn a_refused_enter_is_handed_over_by_a_later_poll_and_completes_once() {
        let data = content(2 * 4096, 7);
        let (_guard, file) = temp_file("refused", &data);
        let calls = Arc::new(AtomicUsize::new(0));
        let counted = calls.clone();
        let io = AsyncIo::open_with(
            Box::new(move |submitter: &Submitter<'_>| {
                if counted.fetch_add(1, Ordering::AcqRel) == 0 {
                    Err(io::Error::from_raw_os_error(libc::EBUSY))
                } else {
                    submitter.submit()
                }
            }),
            MAX_PIECE,
        )
        .unwrap();
        io.read(file.as_raw_fd(), 4096, buf(4096), then(1)).unwrap();
        assert_eq!(io.pending(), 1);
        let mut out = poll(&io, 1);
        for _ in 0..100 {
            io.pull(
                &mut |_| panic!("nothing else completes"),
                &mut |_| false,
                u64::MAX,
            );
        }
        let done = out.pop().unwrap().1.done.unwrap();
        assert_eq!(done.result.unwrap(), 4096);
        assert_eq!(done.buf.bytes, data[4096..]);
        assert!(calls.load(Ordering::Acquire) >= 2, "the poll entered again");
        assert_eq!(io.pending(), 0);
    }

    /// The stand-in backend with pieces of at most `max_piece` bytes: every read, write and send
    /// longer than that completes short and is resubmitted for the rest.
    fn short_pieces(max_piece: usize) -> Arc<AsyncIo> {
        AsyncIo::open_with(
            Box::new(|submitter: &Submitter<'_>| submitter.submit()),
            max_piece,
        )
        .unwrap()
    }

    /// AsyncReadsTest's short read: the rest is resubmitted by the poll that reaped the short part,
    /// and the read completes once, with all its bytes.
    #[test]
    fn short_reads_and_writes_are_resubmitted_until_every_byte_moved() {
        let data = content(10 * 4096 + 7, 31);
        let (guard, file) = temp_file("pieces", &data);
        let io = short_pieces(4096);
        io.read(file.as_raw_fd(), 0, buf(data.len()), then(1))
            .unwrap();
        let read = poll(&io, 1).pop().unwrap().1.done.unwrap();
        assert_eq!(read.result.unwrap(), data.len());
        assert_eq!(read.buf.bytes, data);
        let more = content(5 * 4096 + 1, 37);
        io.write(
            file.as_raw_fd(),
            0,
            IoBuf {
                filled: more.len(),
                bytes: more.clone(),
            },
            then(2),
        )
        .unwrap();
        assert_eq!(
            poll(&io, 1).pop().unwrap().1.done.unwrap().result.unwrap(),
            more.len()
        );
        assert_eq!(&std::fs::read(&guard.0).unwrap()[..more.len()], &more[..]);
        assert_eq!(io.pending(), 0);
    }

    fn socket_pair(sndbuf: i32) -> (OwnedFd, std::os::unix::net::UnixStream) {
        let (ours, theirs) = std::os::unix::net::UnixStream::pair().unwrap();
        // SAFETY: setsockopt on a socket we own, with a pointer to an int that outlives the call.
        let set = unsafe {
            libc::setsockopt(
                ours.as_raw_fd(),
                libc::SOL_SOCKET,
                libc::SO_SNDBUF,
                std::ptr::from_ref(&sndbuf).cast(),
                size_of::<i32>() as libc::socklen_t,
            )
        };
        assert_eq!(set, 0);
        (OwnedFd::from(ours), theirs)
    }

    /// A send larger than the socket's buffer and the stand-in's pieces completes only when every
    /// byte is in the kernel, in order, after many short sends (the peer reads as it goes).
    #[test]
    fn a_short_send_is_resubmitted_until_every_byte_is_in_the_kernel() {
        let io = short_pieces(3000);
        let (ours, mut theirs) = socket_pair(4096);
        let data = content(1 << 20, 41);
        let expected = data.clone();
        let reader = std::thread::spawn(move || {
            let mut got = vec![0; expected.len()];
            theirs.read_exact(&mut got).unwrap();
            assert!(got == expected, "the bytes arrive whole and in order");
        });
        io.send(
            ours.as_raw_fd(),
            IoBuf {
                filled: data.len(),
                bytes: data,
            },
            then(1),
        )
        .unwrap();
        let sent = poll(&io, 1).pop().unwrap().1.done.unwrap();
        assert_eq!(
            sent.result.unwrap(),
            1 << 20,
            "one completion, for all of it"
        );
        reader.join().unwrap();
        assert_eq!(io.pending(), 0);
    }

    /// The peer goes away part-way through a send: the error that stopped it reaches the caller's
    /// continuation.
    #[test]
    fn a_send_stopped_part_way_carries_its_error() {
        let io = short_pieces(3000);
        let (ours, mut theirs) = socket_pair(4096);
        let reader = std::thread::spawn(move || {
            let mut some = vec![0; 20_000];
            theirs.read_exact(&mut some).unwrap();
        });
        io.send(
            ours.as_raw_fd(),
            IoBuf {
                filled: 1 << 20,
                bytes: vec![9; 1 << 20],
            },
            then(1),
        )
        .unwrap();
        let sent = poll(&io, 1).pop().unwrap().1.done.unwrap();
        reader.join().unwrap();
        let error = sent.result.unwrap_err();
        assert!(
            matches!(error.raw_os_error(), Some(libc::EPIPE | libc::ECONNRESET)),
            "the closed peer's error: {error}"
        );
        assert_eq!(io.pending(), 0);
    }

    /// More operations than the completion queue holds wait in the backlog and all complete, each
    /// once.
    #[test]
    fn operations_beyond_the_queues_room_wait_and_all_complete() {
        let data = content(4096, 17);
        let (_guard, file) = temp_file("backlog", &data);
        let io = AsyncIo::open().unwrap();
        let count = 3 * RING_ENTRIES as u64;
        for i in 0..count {
            io.read(file.as_raw_fd(), i % 4096, buf(1), then(i))
                .unwrap();
        }
        let mut out = poll(&io, count as usize);
        out.sort_by_key(|(tag, _)| *tag);
        for (i, done) in out {
            let done = done.done.unwrap();
            assert_eq!(done.result.unwrap(), 1);
            assert_eq!(done.buf.bytes[0], data[(i % 4096) as usize]);
        }
        assert_eq!(io.pending(), 0);
    }

    #[test]
    fn a_poll_hands_on_no_more_than_its_demand() {
        let (_guard, file) = temp_file("demand", &content(4096, 19));
        let io = AsyncIo::open().unwrap();
        for i in 0..4 {
            io.read(file.as_raw_fd(), 0, buf(16), then(i)).unwrap();
        }
        let deadline = Instant::now() + Duration::from_secs(10);
        let mut seen = 0;
        while seen < 4 {
            assert!(Instant::now() < deadline);
            let mut batch = 0;
            io.pull(&mut |_| batch += 1, &mut |_| false, 1);
            assert!(batch <= 1);
            seen += batch;
        }
        assert_eq!(io.pending(), 0);
    }

    /// The lattice's direct pulls stop before ordered frames: a held-back completion goes out first
    /// on a later poll.
    #[test]
    fn a_pull_stops_before_the_completion_its_stop_matches_and_hands_it_on_later_in_order() {
        let (_guard, file) = temp_file("stop", &content(64, 43));
        let io = AsyncIo::open().unwrap();
        io.read(file.as_raw_fd(), 0, buf(8), then(1)).unwrap();
        let deadline = Instant::now() + Duration::from_secs(10);
        while lock(&io.front.held).is_empty() {
            assert!(Instant::now() < deadline, "the read did not complete");
            let pulled = io.pull(&mut |_| panic!("held back"), &mut |_| true, u64::MAX);
            assert_eq!(pulled, 0);
        }
        assert_eq!(io.pending(), 1, "a held-back completion is still pending");
        io.read(file.as_raw_fd(), 8, buf(8), then(2)).unwrap();
        let tags: Vec<u64> = poll(&io, 2).into_iter().map(|(tag, _)| tag).collect();
        assert_eq!(tags, [1, 2], "the held-back completion goes first");
        assert_eq!(io.pending(), 0);
    }

    #[test]
    fn a_write_completes_when_all_its_bytes_are_in_the_file() {
        let (guard, file) = temp_file("write", &[]);
        let io = AsyncIo::open().unwrap();
        let data = content(3 * 4096 + 5, 23);
        io.write(
            file.as_raw_fd(),
            10,
            IoBuf {
                filled: data.len(),
                bytes: data.clone(),
            },
            then(1),
        )
        .unwrap();
        let done = poll(&io, 1).pop().unwrap().1.done.unwrap();
        assert_eq!(done.result.unwrap(), data.len());
        let written = std::fs::read(&guard.0).unwrap();
        assert_eq!(&written[10..], &data[..]);
        assert_eq!(&written[..10], &[0; 10]);
    }

    #[test]
    fn empty_reads_and_writes_and_submissions_after_complete_are_refused_at_once() {
        let (_guard, file) = temp_file("refuse", &[1; 16]);
        let io = AsyncIo::open().unwrap();
        assert!(
            io.read(file.as_raw_fd(), 0, IoBuf::default(), then(1))
                .is_err()
        );
        assert!(
            io.write(
                file.as_raw_fd(),
                0,
                IoBuf {
                    bytes: vec![1],
                    filled: 2
                },
                then(2)
            )
            .is_err()
        );
        io.complete();
        assert!(io.read(file.as_raw_fd(), 0, buf(4), then(3)).is_err());
        assert_eq!(io.pending(), 0);
    }

    #[test]
    fn accept_send_and_receive_on_a_loopback_connection() {
        let io = AsyncIo::open().unwrap();
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        io.accept(
            listener.as_raw_fd(),
            Box::new(|accepted| {
                Box::new(Done {
                    header: FrameHeader::new(1),
                    done: None,
                    accepted: Some(accepted),
                })
            }),
        )
        .unwrap();
        #[allow(clippy::disallowed_types)] // the test's peer
        let mut peer = std::net::TcpStream::connect(addr).unwrap();
        let accepted = poll(&io, 1).pop().unwrap().1.accepted.unwrap().unwrap();

        let message = b"hello, world".to_vec();
        io.send(
            accepted.as_raw_fd(),
            IoBuf {
                filled: message.len(),
                bytes: message.clone(),
            },
            then(2),
        )
        .unwrap();
        let sent = poll(&io, 1).pop().unwrap().1.done.unwrap();
        assert_eq!(sent.result.unwrap(), message.len());
        let mut got = vec![0; message.len()];
        peer.read_exact(&mut got).unwrap();
        assert_eq!(got, message);

        io.recv(accepted.as_raw_fd(), buf(64), then(3)).unwrap();
        peer.write_all(b"credit").unwrap();
        let received = poll(&io, 1).pop().unwrap().1.done.unwrap();
        assert_eq!(received.result.unwrap(), 6);
        assert_eq!(&received.buf.bytes[..received.buf.filled], b"credit");

        io.recv(accepted.as_raw_fd(), buf(64), then(4)).unwrap();
        drop(peer);
        let end = poll(&io, 1).pop().unwrap().1.done.unwrap();
        assert_eq!(end.result.unwrap(), 0, "the peer's end of stream");
    }

    /// A new TCP socket for a connect (the caller makes it; `connect` only connects it).
    fn tcp_socket() -> OwnedFd {
        // SAFETY: plain socket(2); the result is checked and owned at once.
        let fd = unsafe { libc::socket(libc::AF_INET, libc::SOCK_STREAM | libc::SOCK_CLOEXEC, 0) };
        assert!(fd >= 0, "socket: {}", io::Error::last_os_error());
        // SAFETY: a new descriptor no one else owns.
        unsafe { OwnedFd::from_raw_fd(fd) }
    }

    fn connected(tag: u64) -> Continuation<io::Result<()>> {
        Box::new(move |result| {
            let done = IoDone {
                buf: IoBuf::default(),
                result: result.map(|()| 0),
            };
            Box::new(Done {
                header: FrameHeader::new(tag),
                done: Some(done),
                accepted: None,
            })
        })
    }

    #[test]
    fn a_connect_completes_once_connected_or_refused() {
        let io = AsyncIo::open().unwrap();
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let socket = tcp_socket();
        io.connect(
            socket.as_raw_fd(),
            listener.local_addr().unwrap(),
            connected(1),
        )
        .unwrap();
        assert!(poll(&io, 1).pop().unwrap().1.done.unwrap().result.is_ok());
        let (mut theirs, _) = listener.accept().unwrap();
        io.send(
            socket.as_raw_fd(),
            IoBuf {
                filled: 4,
                bytes: b"OKAY".to_vec(),
            },
            then(2),
        )
        .unwrap();
        assert_eq!(
            poll(&io, 1).pop().unwrap().1.done.unwrap().result.unwrap(),
            4
        );
        let mut got = [0; 4];
        theirs.read_exact(&mut got).unwrap();
        assert_eq!(&got, b"OKAY");

        // Nothing listens there any more: refused (no server running).
        let closed = listener.local_addr().unwrap();
        drop((listener, theirs));
        let socket = tcp_socket();
        io.connect(socket.as_raw_fd(), closed, connected(3))
            .unwrap();
        let refused = poll(&io, 1)
            .pop()
            .unwrap()
            .1
            .done
            .unwrap()
            .result
            .unwrap_err();
        assert_eq!(refused.kind(), io::ErrorKind::ConnectionRefused);
        assert_eq!(io.pending(), 0);
    }

    #[test]
    fn a_send_to_a_closed_peer_fails_without_a_signal() {
        let io = AsyncIo::open().unwrap();
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        #[allow(clippy::disallowed_types)] // the test's peer
        let ours = std::net::TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let (theirs, _) = listener.accept().unwrap();
        drop(theirs);
        let mut failed = false;
        for i in 0..20 {
            io.send(
                ours.as_raw_fd(),
                IoBuf {
                    bytes: vec![7; 65536],
                    filled: 65536,
                },
                then(i),
            )
            .unwrap();
            if poll(&io, 1).pop().unwrap().1.done.unwrap().result.is_err() {
                failed = true;
                break;
            }
        }
        assert!(
            failed,
            "a closed peer fails the send (EPIPE or ECONNRESET), and the process is still here"
        );
    }

    #[test]
    fn a_datagram_goes_to_its_address() {
        let io = AsyncIo::open().unwrap();
        let receiver = std::net::UdpSocket::bind("127.0.0.1:0").unwrap();
        receiver
            .set_read_timeout(Some(Duration::from_secs(10)))
            .unwrap();
        let sender = std::net::UdpSocket::bind("127.0.0.1:0").unwrap();
        let datagram = b"datagram payload".to_vec();
        io.send_to(
            sender.as_raw_fd(),
            receiver.local_addr().unwrap(),
            IoBuf {
                filled: datagram.len(),
                bytes: datagram.clone(),
            },
            then(1),
        )
        .unwrap();
        assert_eq!(
            poll(&io, 1).pop().unwrap().1.done.unwrap().result.unwrap(),
            datagram.len()
        );
        let mut got = [0; 64];
        let (n, from) = receiver.recv_from(&mut got).unwrap();
        assert_eq!(&got[..n], &datagram[..]);
        assert_eq!(from, sender.local_addr().unwrap());
    }

    #[test]
    fn an_atomic_save_opens_a_temporary_file_writes_it_and_renames_it_over_the_old_one() {
        let (target, _old) = temp_file("config", b"old settings");
        let temp = TempFile(target.0.with_extension("tmp"));
        let io = AsyncIo::open().unwrap();
        type Opened = Arc<Mutex<Option<io::Result<OwnedFd>>>>;
        let opened: Opened = Arc::new(Mutex::new(None));
        let slot = opened.clone();
        io.open_file(
            &temp.0,
            libc::O_WRONLY | libc::O_CREAT | libc::O_TRUNC,
            0o644,
            Box::new(move |fd| {
                *lock(&slot) = Some(fd);
                Box::new(Done {
                    header: FrameHeader::new(1),
                    done: None,
                    accepted: None,
                })
            }),
        )
        .unwrap();
        poll(&io, 1);
        let fd = lock(&opened).take().unwrap().unwrap();
        let settings = b"new settings, longer than the old".to_vec();
        io.write(
            fd.as_raw_fd(),
            0,
            IoBuf {
                filled: settings.len(),
                bytes: settings.clone(),
            },
            then(2),
        )
        .unwrap();
        assert_eq!(
            poll(&io, 1).pop().unwrap().1.done.unwrap().result.unwrap(),
            settings.len()
        );
        drop(fd);
        let renamed = Arc::new(Mutex::new(None));
        let slot = renamed.clone();
        io.rename(
            &temp.0,
            &target.0,
            Box::new(move |result| {
                *lock(&slot) = Some(result);
                Box::new(Done {
                    header: FrameHeader::new(3),
                    done: None,
                    accepted: None,
                })
            }),
        )
        .unwrap();
        poll(&io, 1);
        lock(&renamed).take().unwrap().unwrap();
        assert_eq!(std::fs::read(&target.0).unwrap(), settings);
        assert!(!temp.0.exists(), "the temporary file was renamed");
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(
            std::fs::metadata(&target.0).unwrap().permissions().mode() & 0o777,
            0o644 & !umask()
        );
    }

    fn umask() -> u32 {
        // SAFETY: umask only swaps the process's mask; it is set straight back.
        let mask = unsafe { libc::umask(0o022) };
        // SAFETY: as above.
        unsafe { libc::umask(mask) };
        mask as u32
    }

    #[test]
    fn opening_or_renaming_what_is_not_there_fails_with_the_kernels_error() {
        let io = AsyncIo::open().unwrap();
        let missing = std::path::Path::new("/nonexistent-euhedral-dir/file");
        let errors = Arc::new(Mutex::new(Vec::new()));
        let (a, b) = (errors.clone(), errors.clone());
        io.open_file(
            missing,
            libc::O_RDONLY,
            0,
            Box::new(move |r| {
                lock(&a).push(r.unwrap_err().raw_os_error());
                Box::new(Done {
                    header: FrameHeader::new(1),
                    done: None,
                    accepted: None,
                })
            }),
        )
        .unwrap();
        io.rename(
            missing,
            &missing.with_extension("x"),
            Box::new(move |r| {
                lock(&b).push(r.unwrap_err().raw_os_error());
                Box::new(Done {
                    header: FrameHeader::new(2),
                    done: None,
                    accepted: None,
                })
            }),
        )
        .unwrap();
        poll(&io, 2);
        assert_eq!(*lock(&errors), [Some(libc::ENOENT), Some(libc::ENOENT)]);
        assert!(
            io.open_file(
                std::path::Path::new("nul\0byte"),
                0,
                0,
                Box::new(|_| unreachable!())
            )
            .is_err(),
            "a path with a NUL byte is refused at once"
        );
    }

    /// The lattice's other path: a request pushes completions to the downstream, up to the
    /// accumulated demand.
    #[test]
    fn requests_push_completions_downstream() {
        #[derive(Default)]
        struct Recorder(Mutex<Vec<u64>>, AtomicBool);
        impl euhedral_core::generics::LatticeTerminal for Recorder {
            fn add_upstream(&self, _: Arc<dyn LatticeSource>) {}
        }
        impl LatticeReceiver for Recorder {
            fn push(&self, frame: Box<dyn Frame>) {
                lock(&self.0).push(frame.header().id_hash());
            }
            fn on_complete(&self) {
                self.1.store(true, Ordering::Release);
            }
            fn on_error(&self, _: euhedral_core::frames::Failure) {}
        }
        let (_guard, file) = temp_file("request", &content(64, 29));
        let io = AsyncIo::open().unwrap();
        let recorder = Arc::new(Recorder::default());
        io.clone().add_downstream(recorder.clone());
        io.read(file.as_raw_fd(), 0, buf(8), then(5)).unwrap();
        let deadline = Instant::now() + Duration::from_secs(10);
        while lock(&recorder.0).is_empty() {
            assert!(Instant::now() < deadline);
            io.request(1);
        }
        assert_eq!(*lock(&recorder.0), [5]);
        io.complete();
        assert!(recorder.1.load(Ordering::Acquire));
    }

    /// Entries queued while another thread holds `sq` are submitted by that thread or by a re-check,
    /// never stranded: no entry is left in `incoming` or the backlog once every pusher returned,
    /// with no later flush to rescue it. Each round, 4 threads queue 2 entries each on one ring
    /// whose enter is slow, so the holder is inside it when the others push, and the last pushes
    /// come together (a rescue by a later push would hide a stranded entry).
    #[test]
    #[allow(clippy::disallowed_types)] // The pushers start together.
    fn entries_queued_while_another_thread_holds_sq_are_never_stranded() {
        const ROUNDS: usize = 1000;
        const THREADS: usize = 4;
        const EACH: usize = 2;
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            for round in 0..ROUNDS {
                let ring = Arc::new(Ring::new().unwrap());
                let slow: Arc<Enter> = Arc::new(|submitter: &Submitter<'_>| {
                    // Long enough for the other threads to push meanwhile.
                    let until = Instant::now() + Duration::from_micros(30);
                    while Instant::now() < until {
                        std::hint::spin_loop();
                    }
                    submitter.submit()
                });
                let go = Arc::new(std::sync::Barrier::new(THREADS));
                let threads: Vec<_> = (0..THREADS)
                    .map(|t| {
                        let (ring, slow, go) = (ring.clone(), slow.clone(), go.clone());
                        std::thread::spawn(move || {
                            go.wait();
                            for i in 0..EACH {
                                let entry =
                                    opcode::Nop::new().build().user_data((t * EACH + i) as u64);
                                ring.queue(entry, &*slow);
                            }
                        })
                    })
                    .collect();
                for thread in threads {
                    thread.join().unwrap();
                }
                let queue = ring.sq.try_lock().expect("every pusher has returned");
                assert!(
                    ring.incoming.is_empty()
                        && queue.backlog.is_empty()
                        && !queue.unsubmitted
                        && ring.in_kernel.load(Ordering::Acquire) == THREADS * EACH,
                    "round {round}: {} in the kernel of {}, {} backlogged, {} still queued",
                    ring.in_kernel.load(Ordering::Acquire),
                    THREADS * EACH,
                    queue.backlog.len(),
                    ring.incoming.size_long()
                );
            }
            let _ = tx.send(());
        });
        rx.recv_timeout(Duration::from_secs(120))
            .expect("the rounds did not finish in time");
    }
}
