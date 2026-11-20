//! The Windows engine of `AsyncIo`: I/O completion ports for every operation, and IoRing (a kernel
//! ring, as io_uring is) for file reads and writes where the OS has one.
//!
//! - **One completion port per `AsyncIo`.** Any worker issues overlapped calls; the lattice's
//!   polls dequeue (`GetQueuedCompletionStatusEx` with a zero timeout, so nothing waits). Every
//!   operation ends through a poll: the "skip the port on success" mode is not set, and a handle
//!   that has it set is refused.
//! - **One IoRing per submitting thread**, for file reads and writes, when the OS offers a kernel
//!   ring for both ([`Sys::ring_probe`]: not emulated in user mode, with reads and writes
//!   supported: Windows 11 22H2 and later). A full
//!   submission queue goes to the ring's backlog and is handed over by a later submission or poll.
//!   A file keeps its route: IoRing refuses a handle that belongs to a completion port.
//! - **Open, rename and file length** have no asynchronous call. They run on the OS thread pool and
//!   leave their result in a queue the poll drains, so a job that panics or is dropped still ends
//!   its operation, with an error. It is the one place the crate uses threads it did not create.
//!
//! # Records and identity
//!
//! An operation's state lives in a [`Rec`] in a slab: a record is made once per slot and **never
//! freed while the engine lives**, so a late packet that carries its address can always be read
//! safely. A slot is `Idle`, `Issuing` (its call is being made), `InFlight` (the kernel or the
//! pool has it), or `Busy` (a poll took it and is ending or continuing it). The slot's
//! generation changes at every issue; a ring entry's user data is the slot and the generation,
//! and a pool result carries both. A port packet carries only the `OVERLAPPED`'s address, so it
//! is accepted only for a slot that is `InFlight` on the port route and whose `Internal` is not
//! `STATUS_PENDING` (a packet of an earlier use of the slot meets the new call's pending status).
//!
//! A call returns before its operation can be completed: a poll that meets the packet of an
//! `Issuing` slot leaves it for the next poll. That is what lets the call's caller read the
//! `OVERLAPPED` after the call to tell "failed, no packet" from "failed with a warning status, a
//! packet follows".
//!
//! Every Windows call goes through [`Sys`], so the state machines run on any platform against a
//! simulated `Sys` (`sim`, in the tests). Only [`real`] (Windows) makes the calls.

#![allow(unsafe_code)]
#![cfg_attr(not(windows), allow(dead_code))]

use std::cell::{Cell, RefCell, UnsafeCell};
use std::collections::{BTreeSet, VecDeque};
use std::io;
use std::ops::{Deref, DerefMut};
use std::path::PathBuf;
use std::ptr::NonNull;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Weak};
use std::time::Duration;

use euhedral_core::frames::Frame;
use euhedral_data_structures::queues::{Consumer, MpscQueue, Multi, Producer, Single};
use euhedral_data_structures::sync::ClaimCell;

use self::handles::HandleSet;
use self::slab::{Asked, End, RouteTag, Slab, SlotCell, State};
use self::sys::{
    ACCEPT_BUF, BuildError, Cqe, Handle, IORING_VERSION_3, Overlapped, Packet, STATUS_CANCELLED,
    STATUS_END_OF_FILE, STATUS_PENDING, SockAddr, Sys, hresult_error, is_end_of_file,
    is_nt_warning,
};
use super::{
    AsyncIo, AsyncIoOptions, Buf, Cancelled, Engine as EngineTrait, LimitPassed, MAX_PIECE, OpId,
    OpenSpec, READ_PIECE, Request, Then, conclude as conclude_io, invalid,
};

mod handles;
#[cfg(all(windows, test))]
mod live;
#[cfg(windows)]
mod real;
#[cfg(test)]
mod sim;
mod slab;
pub(super) mod sys;
#[cfg(test)]
mod tests;

/// Submission entries of each ring (its completion queue has twice as many).
const RING_ENTRIES: u32 = 256;
/// Submitting threads (rings) one `AsyncIo` serves.
const MAX_RINGS: usize = 256;
/// Packets taken from the port by one system call.
const BATCH: usize = 64;
/// Entries one flush builds before it submits: a submit that copies is not unbounded work for the
/// worker that made it.
const FLUSH_ENTRIES: usize = 16;
/// Bits of a ring entry's user data that name the slot.
const SLOT_BITS: u32 = 20;
/// Operations in flight at once.
const MAX_OPS: usize = (1 << SLOT_BITS) - 1;
/// What the generation of a token keeps (a token is a `usize`): 44 bits on 64-bit targets, but
/// only 12 on 32-bit ones, where a slot's generation wraps after 4096 issues. That is enough: a
/// stale ring completion would need its slot reissued exactly a multiple of 4096 times meanwhile.
const GENERATION_MASK: usize = usize::MAX >> SLOT_BITS;

/// Entries of a chunk of the queues that hand failures and pool results to the poller.
const HAND_OFF_CHUNK: usize = 256;

static NEXT_ID: AtomicUsize = AtomicUsize::new(1);
/// `Engine::next_deadline` when no operation has a limit.
const NO_DEADLINE: u64 = u64::MAX;

thread_local! {
    /// This thread's ring of each engine it submitted to (by id).
    static MINE: RefCell<Vec<(usize, Weak<Ring>)>> = const { RefCell::new(Vec::new()) };
}

/// The completion port, closed with the engine.
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
    FileLen { file: Handle },
}

/// How the operation reaches the kernel.
#[derive(Clone)]
enum Route {
    /// An overlapped call; its end is a packet of the port.
    Port,
    /// A ring entry; its end is a completion of the ring.
    Ring(Arc<Ring>),
    /// The OS thread pool; its end is a result in the engine's queue.
    Pool,
}

/// The socket an accept fills, and the buffer `AcceptEx` writes the two addresses to.
struct Accepting {
    socket: Handle,
    #[allow(dead_code, reason = "written by the kernel")]
    addresses: Box<[u8; ACCEPT_BUF]>,
}

/// An operation in flight.
struct Op {
    kind: Kind,
    route: Route,
    buf: Buf,
    /// Bytes moved so far (reads, writes and sends resubmit the rest).
    moved: usize,
    then: Then,
    /// The destination of a datagram or a connect.
    addr: Option<Box<SockAddr>>,
    accepting: Option<Accepting>,
    /// When the operation's time limit ends, on the engine's clock ([`Sys::now_ns`]).
    deadline: Option<u64>,
}

impl Op {
    /// The handle a cancel of this operation's call is made on: a socket operation's.
    fn cancel_target(&self) -> Option<Handle> {
        match (&self.route, &self.kind) {
            (
                Route::Port,
                Kind::Send { sock: handle }
                | Kind::Recv { sock: handle }
                | Kind::SendTo { sock: handle }
                | Kind::Accept { listener: handle }
                | Kind::Connect { sock: handle },
            ) => Some(*handle),
            _ => None,
        }
    }
}

impl End {
    /// The error of an operation that ended for this reason after moving `moved` bytes. Neither
    /// has an OS code, so both are told from an operation the OS aborted (`ERROR_OPERATION_ABORTED`,
    /// which `std` reads as `TimedOut`, with its code).
    fn error(self, moved: usize) -> io::Error {
        match self {
            Self::Cancelled => Cancelled::error(moved),
            Self::TimedOut => LimitPassed::error(moved),
        }
    }
}

/// A slot's record. `ov` is first (it is what the kernel is given); the record is never freed
/// while the engine lives.
#[repr(C)]
#[derive(Default)]
struct Rec {
    ov: Overlapped,
    op: Option<Op>,
}

/// An operation a thread holds: its slot is `Issuing` or `Busy` (or `InFlight` on a ring or the
/// pool, until its entry is pushed).
#[derive(Clone, Copy)]
struct Held {
    slot: u32,
    generation: u32,
    serial: u64,
    rec: NonNull<Rec>,
}

/// A ring entry's user data: the slot and the generation of this issue.
fn token(slot: u32, generation: u32) -> usize {
    ((generation as usize & GENERATION_MASK) << SLOT_BITS) | slot as usize
}

fn untoken(token: usize) -> (u32, usize) {
    ((token & ((1 << SLOT_BITS) - 1)) as u32, token >> SLOT_BITS)
}

/// An operation that failed at its call: ends with the error at the next poll.
struct Failed {
    slot: u32,
    generation: u32,
    error: io::Error,
}

/// What the pool left for an open, a rename or a file length.
struct PoolDone {
    slot: u32,
    generation: u32,
    result: io::Result<usize>,
}

/// Delivers a pool job's result, or, if the job is dropped or panics first, an error.
struct Delivery {
    queue: Producer<PoolDone, Multi>,
    slot: u32,
    generation: u32,
    result: Option<io::Result<usize>>,
}

impl Delivery {
    /// The job's result; the delivery then goes out as it is dropped.
    fn deliver(mut self, result: io::Result<usize>) {
        self.result = Some(result);
    }
}

impl Drop for Delivery {
    fn drop(&mut self) {
        let result = self.result.take().unwrap_or_else(|| {
            Err(io::Error::other(
                "the OS thread pool did not finish the job (it panicked or refused it)",
            ))
        });
        let queued = self.queue.offer(PoolDone {
            slot: self.slot,
            generation: self.generation,
            result,
        });
        debug_assert!(queued.is_ok(), "the unbounded queue refused a pool result");
    }
}

/// The call that starts or continues an operation, with its pointers taken after its slot was
/// marked (nothing of the record is touched after the call is made).
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
    /// Its place in the engine's rings.
    index: usize,
    handle: Handle,
    /// Building entries, submitting and popping: one thread at a time. A thread that finds it
    /// claimed has left its entry in `incoming`, for the holder (who looks again after it lets go)
    /// or the next poll.
    inner: ClaimCell<RingInner>,
    /// Entries waiting for room in the kernel's queues, the oldest first; submissions only add.
    incoming: Producer<Entry, Multi>,
    /// Entries built and not yet popped; at most the completion queue's size, so it never
    /// overflows.
    in_kernel: AtomicUsize,
    capacity: usize,
    /// A pop failed for good: the ring's operations were ended with that error.
    broken: AtomicBool,
}

struct RingInner {
    /// Entries are built that the kernel has not taken (a submit failed).
    unsubmitted: bool,
    /// The other end of `Ring::incoming`.
    backlog: Consumer<Entry, Single>,
}

impl Drop for Ring {
    fn drop(&mut self) {
        // A ring with operations in flight (or one that broke) leaks: the kernel may still write
        // their buffers.
        if self.in_kernel.load(Ordering::Acquire) == 0 && !self.broken.load(Ordering::Acquire) {
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

/// The state of the polling thread: the drain sides of the engine's hand-off queues.
struct Poller {
    failed: Consumer<Failed, Single>,
    pool_done: Consumer<PoolDone, Single>,
    /// Packets taken from the port and not yet handled (a poll stops when the caller has room for
    /// no more, and leaves the packets of slots whose call is still returning).
    inbox: VecDeque<Packet>,
    /// The limits of the operations that have one and have not ended: deadline and slot, the
    /// soonest first, so that a poll looks at the head only.
    limits: BTreeSet<(u64, u32)>,
    /// The limits `begin` offered since the last poll.
    new_limits: Consumer<Limit, Single>,
}

/// A limit an operation began with.
struct Limit {
    deadline: u64,
    slot: u32,
}

/// State that only the polling thread touches. The lattice runs one poll at a time (the reaper
/// cell of `Front` admits one), so no claim is needed; a debug build checks it.
struct PollOwned<T> {
    value: UnsafeCell<T>,
    #[cfg(debug_assertions)]
    inside: AtomicBool,
}

// SAFETY: the value is reached only through `enter`, whose caller promises it is the only one
// inside.
unsafe impl<T: Send> Sync for PollOwned<T> {}

struct PollGuard<'a, T>(&'a PollOwned<T>);

impl<T> PollOwned<T> {
    fn new(value: T) -> Self {
        Self {
            value: UnsafeCell::new(value),
            #[cfg(debug_assertions)]
            inside: AtomicBool::new(false),
        }
    }

    /// Enters the state.
    ///
    /// # Safety
    /// No other thread is inside, and this thread is not inside already: the caller is `poll`, which
    /// the lattice does not run twice at once.
    unsafe fn enter(&self) -> PollGuard<'_, T> {
        #[cfg(debug_assertions)]
        assert!(
            !self.inside.swap(true, Ordering::Acquire),
            "two polls are inside one engine"
        );
        PollGuard(self)
    }
}

impl<T> Deref for PollGuard<'_, T> {
    type Target = T;

    fn deref(&self) -> &T {
        // SAFETY: by `enter`'s contract.
        unsafe { &*self.0.value.get() }
    }
}

impl<T> DerefMut for PollGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        // SAFETY: by `enter`'s contract.
        unsafe { &mut *self.0.value.get() }
    }
}

impl<T> Drop for PollGuard<'_, T> {
    fn drop(&mut self) {
        #[cfg(debug_assertions)]
        self.0.inside.store(false, Ordering::Release);
    }
}

pub(super) struct Engine {
    id: usize,
    sys: Arc<dyn Sys>,
    port: Port,
    /// The records of the operations, with their states.
    slab: Slab<Rec>,
    /// The serial of the last operation begun.
    last_serial: AtomicU64,
    /// The limits of new operations, for the poller.
    limits: Producer<Limit, Multi>,
    /// Operations whose call failed at once, to end with that error at the next poll: any thread
    /// offers, the poller drains ([`Poller::failed`]).
    failed: Producer<Failed, Multi>,
    /// What open, rename and file length jobs left: the pool's threads offer, the poller drains
    /// ([`Poller::pool_done`]).
    pool_done: Producer<PoolDone, Multi>,
    /// What only the polling thread touches.
    poller: PollOwned<Poller>,
    /// The handles this engine joined to its port.
    associated: HandleSet,
    /// File reads and writes go through rings (only where the OS has both: a handle that a ring
    /// serves must not belong to a port, and a write has to follow its reads there).
    ring_files: bool,
    /// The file handles a ring serves.
    ringed: HandleSet,
    /// The rings, by index: a ring is set once, never removed, and made after its index was
    /// reserved in `ring_count` (a poll that finds the index not set yet skips it: the ring has no
    /// operation).
    rings: Box<[OnceLock<Arc<Ring>>]>,
    ring_count: AtomicUsize,
    max_piece: usize,
    /// The soonest deadline of the limits (`NO_DEADLINE` when there is none): `begin` lowers it, the
    /// poller sets it to the head of its set. A poll before it does nothing for limits.
    next_deadline: AtomicU64,
    /// Packets and completions of no operation (ignored), for tests.
    strays: AtomicUsize,
}

/// What a dequeued packet leads to.
enum Taken {
    /// The operation, now `Busy`, with the status and byte count of its call and why it was asked
    /// to end, if it was (read in the lock that made it `Busy`).
    Op(Held, i32, u32, Option<End>),
    /// Its call is still returning: look again at the next poll.
    Later,
    /// A packet of no operation in flight on the port.
    Stray,
}

impl Engine {
    /// A new unconnected stream socket of `to`'s family, ready for a connect.
    pub(super) fn new_socket(&self, to: std::net::SocketAddr) -> io::Result<usize> {
        let socket = self.sys.new_socket(to)?;
        // The system made the value new: whatever closed with it before is not this socket.
        self.forget(socket);
        Ok(socket)
    }

    /// An engine over `sys`. Fails if there is no completion port.
    pub(super) fn new(
        sys: Arc<dyn Sys>,
        options: AsyncIoOptions,
        max_piece: usize,
    ) -> io::Result<Self> {
        let handle = sys.create_port()?;
        let port = Port {
            sys: sys.clone(),
            handle,
        };
        // A ring is used only where the OS has a kernel ring: not one emulated in user mode.
        let probe = options
            .windows_ioring
            .then(|| sys.ring_probe())
            .flatten()
            .filter(|probe| !probe.emulated);
        let failed: MpscQueue<Failed> = MpscQueue::with_pool(HAND_OFF_CHUNK, 2);
        let pool_done: MpscQueue<PoolDone> = MpscQueue::with_pool(HAND_OFF_CHUNK, 2);
        let limits: MpscQueue<Limit> = MpscQueue::with_pool(HAND_OFF_CHUNK, 2);
        Ok(Self {
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            sys,
            port,
            failed: failed.producer(),
            pool_done: pool_done.producer(),
            poller: PollOwned::new(Poller {
                failed: failed.consumer(),
                pool_done: pool_done.consumer(),
                inbox: VecDeque::new(),
                limits: BTreeSet::new(),
                new_limits: limits.consumer(),
            }),
            slab: Slab::new(MAX_OPS as u32),
            last_serial: AtomicU64::new(0),
            limits: limits.producer(),
            associated: HandleSet::default(),
            ring_files: probe.is_some_and(|probe| {
                probe.read && probe.write && probe.version >= IORING_VERSION_3
            }),
            ringed: HandleSet::default(),
            rings: (0..MAX_RINGS).map(|_| OnceLock::new()).collect(),
            ring_count: AtomicUsize::new(0),
            max_piece: max_piece.clamp(1, MAX_PIECE),
            next_deadline: AtomicU64::new(NO_DEADLINE),
            strays: AtomicUsize::new(0),
        })
    }

    /// Whether file reads and writes go through a ring (for tests).
    #[cfg(test)]
    pub(super) fn ring_use(&self) -> (bool, bool) {
        (self.ring_files, self.ring_files)
    }

    /// The number of bytes of the next piece of `n` left: at most `max_piece`, and a `u32`.
    fn piece(&self, n: usize) -> u32 {
        u32::try_from(n.min(self.max_piece)).unwrap_or(u32::MAX)
    }

    fn stray(&self) {
        self.strays.fetch_add(1, Ordering::Relaxed);
    }

    /// Packets the poller holds back, for tests (no poll runs meanwhile).
    #[cfg(test)]
    fn inbox_len(&self) -> usize {
        // SAFETY: the tests that call this run no poll at the same time.
        unsafe { self.poller.enter() }.inbox.len()
    }

    // ---- the slab

    fn cell(&self, slot: u32) -> &SlotCell<Rec> {
        self.slab.cell(slot).expect("a slot that was made")
    }

    fn held(&self, slot: u32, generation: u32, cell: &SlotCell<Rec>) -> Held {
        Held {
            slot,
            generation,
            serial: cell.serial(),
            rec: NonNull::new(cell.rec()).expect("a record"),
        }
    }

    /// A slot for `op`, in `state`: its record is filled, and nothing else refers to it.
    fn begin(&self, op: Op, state: State) -> io::Result<Held> {
        let slot = self
            .slab
            .alloc()
            .ok_or_else(|| io::Error::other("too many operations in flight"))?;
        let cell = self.cell(slot);
        let serial = self.last_serial.fetch_add(1, Ordering::Relaxed) + 1;
        let (route, ring) = match &op.route {
            Route::Port => (RouteTag::Port, 0),
            Route::Ring(ring) => (RouteTag::Ring, ring.index),
            Route::Pool => (RouteTag::Pool, 0),
        };
        let cancellable = op.cancel_target().is_some();
        let deadline = op.deadline;
        // SAFETY: the slot is idle and ours: no other thread and no kernel call refers to the
        // record.
        unsafe { (*cell.rec()).op = Some(op) };
        let generation = cell.open(state, route, ring, cancellable, serial);
        if let Some(deadline) = deadline {
            // The poller finds the limit before it finds the lowered deadline.
            let queued = self.limits.offer(Limit { deadline, slot });
            debug_assert!(queued.is_ok(), "the unbounded queue refused a limit");
            self.next_deadline.fetch_min(deadline, Ordering::AcqRel);
        }
        Ok(self.held(slot, generation, cell))
    }

    /// A taken operation goes in again: a new issue of the same slot. Not if it was asked to end
    /// meanwhile: that is the reason, and the slot stays as it was.
    fn rebegin(&self, held: Held, state: State) -> Result<Held, End> {
        let generation = self.cell(held.slot).reissue(state)?;
        Ok(Held { generation, ..held })
    }

    /// Moves the slot to `state`. A slot that goes in flight after being asked to end meanwhile
    /// (a cancel or a limit while its call was being made) has its call cancelled now.
    fn set_state(&self, held: Held, state: State) {
        let cell = self.cell(held.slot);
        if cell.arm(state) {
            self.cancel_call(cell);
            cell.unpin();
        }
    }

    /// The slot is free again, and so is its limit.
    fn release(&self, poller: &mut Poller, held: Held, deadline: Option<u64>) {
        self.cell(held.slot).release();
        if let Some(deadline) = deadline {
            self.take_new_limits(poller);
            poller.limits.remove(&(deadline, held.slot));
            self.publish_deadline(poller);
        }
        self.slab.free(held.slot);
    }

    /// The call of an operation in flight is cancelled, on its handle: the kernel then ends it with
    /// `STATUS_CANCELLED`, or with its own result if that came first. Only socket operations are
    /// cancelled. Made while the slot is pinned, which is what keeps the record from being reused
    /// meanwhile: a cancel of a reused `OVERLAPPED` would end another operation's call.
    fn cancel_call(&self, cell: &SlotCell<Rec>) {
        let rec = cell.rec();
        // SAFETY: a slab record, valid while the engine lives; `op` is read while the slot is
        // `InFlight` and pinned, when nothing writes it.
        let target = unsafe { (*rec).op.as_ref().and_then(Op::cancel_target) };
        if let Some(handle) = target {
            // SAFETY: `ov` is the record's `OVERLAPPED` that the call of `handle` was given; the
            // call is cancelled by its address and the handle, and one that has ended is not found.
            unsafe { self.sys.cancel_io(handle, &raw mut (*rec).ov) };
        }
    }

    /// Asks the operation `serial` of `slot` to end early, for `why`; nothing if it has ended.
    fn request_end(&self, slot: u32, serial: u64, why: End) {
        let Some(cell) = self.slab.cell(slot) else {
            return;
        };
        if cell.ask_to_end(serial, why) == Asked::Cancel {
            self.cancel_call(cell);
            cell.unpin();
        }
    }

    /// Makes `next_deadline` the soonest of the limits. The poller's.
    fn publish_deadline(&self, poller: &Poller) {
        let soonest = poller
            .limits
            .first()
            .map_or(NO_DEADLINE, |&(deadline, _)| deadline);
        self.next_deadline.store(soonest, Ordering::Release);
    }

    /// The limits operations began with since the last poll go into the set.
    fn take_new_limits(&self, poller: &mut Poller) {
        let Poller {
            limits, new_limits, ..
        } = poller;
        new_limits.drain_all(|limit| {
            limits.insert((limit.deadline, limit.slot));
        });
    }

    /// Asks every operation whose limit has passed to end, as `request_end` does. A poll's work, so
    /// that no timer and no thread is needed: a limit ends at the first poll after it passed.
    fn expire_limits(&self, poller: &mut Poller) {
        if !poller.new_limits.is_empty() {
            self.take_new_limits(poller);
            self.publish_deadline(poller);
        }
        if self.next_deadline.load(Ordering::Acquire) == NO_DEADLINE {
            return;
        }
        let now = self.sys.now_ns();
        if now < self.next_deadline.load(Ordering::Acquire) {
            return;
        }
        while let Some(&(deadline, slot)) = poller.limits.first() {
            if deadline > now {
                break;
            }
            poller.limits.pop_first();
            // The limit is in the set until its operation is released, by this thread: the slot is
            // still that operation's.
            let cell = self.cell(slot);
            self.request_end(slot, cell.serial(), End::TimedOut);
        }
        self.publish_deadline(poller);
    }

    /// The operation whose `OVERLAPPED` is at the packet's address, if it is in flight on the port
    /// and its call finished: its slot becomes `Busy`. The call's status and byte count are read
    /// from the `OVERLAPPED`, and the packet's count must agree with it.
    fn take_packet(&self, packet: &Packet) -> Taken {
        let Some(slot) = self.slab.slot_at(packet.overlapped) else {
            return Taken::Stray;
        };
        let cell = self.cell(slot);
        let mut status = 0;
        let taken = cell.take(|word| {
            match word.state() {
                State::Idle | State::Busy => return Err(Taken::Stray),
                // The call is still returning: its caller has not looked at the record yet.
                State::Issuing => return Err(Taken::Later),
                State::InFlight => {}
            }
            // A cancel is in the call: the slot is left for the next poll, so that it is not
            // reused while the cancel names its record.
            if word.pinned() {
                return Err(Taken::Later);
            }
            if word.route() != RouteTag::Port {
                return Err(Taken::Stray);
            }
            let rec = cell.rec();
            // SAFETY: a slab record, valid while the engine lives. The kernel writes `Internal` and
            // `InternalHigh` at completion, so they are loaded as atomics. Only this thread takes an
            // `InFlight` slot of the port.
            let (internal, high) = unsafe {
                (
                    Overlapped::load_internal(&raw const (*rec).ov),
                    Overlapped::load_internal_high(&raw const (*rec).ov),
                )
            };
            status = internal as u32 as i32;
            // A packet of an earlier use of the slot, while the new call is pending (or whose
            // count is not this call's).
            if status == STATUS_PENDING || high != packet.bytes as usize {
                return Err(Taken::Stray);
            }
            Ok(())
        });
        match taken {
            Err(other) => other,
            Ok(word) => Taken::Op(
                self.held(slot, word.generation(), cell),
                status,
                packet.bytes,
                word.end(),
            ),
        }
    }

    /// The operation a ring completion (`ring`) or a pool result (`None`) names, if its slot is
    /// `InFlight` on that route with that generation: the slot becomes `Busy`.
    fn take_token(&self, slot: u32, generation: usize, ring: Option<&Arc<Ring>>) -> Option<Held> {
        let cell = self.slab.cell(slot)?;
        let word = cell
            .take(|word| {
                let route_ok = match (word.route(), ring) {
                    (RouteTag::Pool, None) => true,
                    (RouteTag::Ring, Some(ring)) => word.ring() == ring.index,
                    _ => false,
                };
                if word.state() == State::InFlight
                    && (word.generation() as usize) & GENERATION_MASK
                        == generation & GENERATION_MASK
                    && route_ok
                {
                    Ok(())
                } else {
                    Err(())
                }
            })
            .ok()?;
        Some(self.held(slot, word.generation(), cell))
    }

    /// The operation that failed at its call (a `Busy` slot of that generation).
    fn take_failed(&self, failed: &Failed) -> Option<Held> {
        let cell = self.slab.cell(failed.slot)?;
        let word = cell.load();
        (word.state() == State::Busy && word.generation() == failed.generation)
            .then(|| self.held(failed.slot, failed.generation, cell))
    }

    /// The operation fails at its call: it ends with `error` at the next poll.
    fn fail(&self, held: Held, error: io::Error) {
        self.set_state(held, State::Busy);
        self.offer_failed(Failed {
            slot: held.slot,
            generation: held.generation,
            error,
        });
    }

    /// Hands a failure to the poller.
    fn offer_failed(&self, failed: Failed) {
        let queued = self.failed.offer(failed);
        debug_assert!(queued.is_ok(), "the unbounded queue refused a failure");
    }

    /// A ring entry ends its operation with `error`, if the operation is still in flight. With
    /// `leak`, the entry may have been handed to the kernel (it cannot be known), so the buffer is
    /// leaked rather than returned, as `break_ring` leaks the buffers of what it ends.
    fn fail_token(&self, user_data: usize, error: io::Error, leak: bool) {
        let (slot, generation) = untoken(user_data);
        let Some(cell) = self.slab.cell(slot) else {
            return;
        };
        let taken = cell.take(|word| {
            if word.state() == State::InFlight
                && (word.generation() as usize) & GENERATION_MASK == generation & GENERATION_MASK
            {
                Ok(())
            } else {
                Err(())
            }
        });
        let Ok(word) = taken else {
            return;
        };
        if leak {
            // SAFETY: the slot is `Busy` now: ours.
            if let Some(op) = unsafe { (*cell.rec()).op.as_mut() } {
                op.buf.leak();
            }
        }
        self.offer_failed(Failed {
            slot,
            generation: word.generation(),
            error,
        });
    }

    /// A ring that broke meanwhile has no one to hand its backlog over (`break_ring` may have
    /// cleared it before the entry was pushed): the operation ends here, unless `break_ring`
    /// already ended it. A concurrent flush may have built the entry before the ring broke, so
    /// the kernel may own the buffer: it is leaked.
    fn end_if_broken(&self, ring: &Ring, user_data: usize) {
        if ring.broken.load(Ordering::Acquire) {
            self.fail_token(user_data, ring_failed(), true);
        }
    }

    // ---- rings

    /// The calling thread's ring, made on first use; `None` if there is none to be had.
    fn ring(&self) -> Option<Arc<Ring>> {
        MINE.with_borrow_mut(|mine| {
            mine.retain(|(_, ring)| ring.strong_count() > 0);
            if let Some(ring) = mine
                .iter()
                .find(|(id, _)| *id == self.id)
                .and_then(|(_, ring)| ring.upgrade())
            {
                if !ring.broken.load(Ordering::Acquire) {
                    return Some(ring);
                }
                mine.retain(|(id, _)| *id != self.id);
            }
            if self.ring_count.load(Ordering::Acquire) >= MAX_RINGS {
                return None;
            }
            let handle = self.sys.ring_create(RING_ENTRIES).ok()?;
            // Another thread may have taken the last index meanwhile.
            let Ok(index) = self
                .ring_count
                .try_update(Ordering::AcqRel, Ordering::Acquire, |n| {
                    (n < MAX_RINGS).then_some(n + 1)
                })
            else {
                self.sys.ring_close(handle);
                return None;
            };
            let backlog: MpscQueue<Entry> = MpscQueue::with_pool(HAND_OFF_CHUNK, 2);
            let ring = Arc::new(Ring {
                sys: self.sys.clone(),
                index,
                handle,
                inner: ClaimCell::new(RingInner {
                    unsubmitted: false,
                    backlog: backlog.consumer(),
                }),
                incoming: backlog.producer(),
                in_kernel: AtomicUsize::new(0),
                capacity: 2 * RING_ENTRIES as usize,
                broken: AtomicBool::new(false),
            });
            // Each index is reserved by one thread, which sets it once.
            let _ = self.rings[index].set(ring.clone());
            mine.push((self.id, Arc::downgrade(&ring)));
            Some(ring)
        })
    }

    /// Hands what waits in the ring's backlog (or was built and not submitted) to the kernel,
    /// unless another thread is doing so right now.
    fn flush_ring(&self, ring: &Ring) {
        loop {
            let Some(mut claim) = ring.inner.try_claim() else {
                // The holder looks for what was offered before it lets go.
                return;
            };
            let drained = self.flush_claimed(ring, &mut claim);
            // An entry offered after the holder's last look finds the claim held; letting go with
            // a full fence makes the holder see it, or the offerer win the claim.
            claim.release_seq_cst();
            if !drained || ring.incoming.is_empty() {
                return;
            }
        }
    }

    /// The flush, by the holder of the claim. `true` if it left nothing waiting that it could have
    /// built (it did not stop for room).
    fn flush_claimed(&self, ring: &Ring, inner: &mut RingInner) -> bool {
        if ring.broken.load(Ordering::Acquire) {
            // Nothing is handed to a ring that cannot be read; its entries were ended.
            inner.backlog.drain_all(drop);
            return true;
        }
        let sys = &*self.sys;
        let mut built = 0;
        let mut drained = true;
        while built < FLUSH_ENTRIES && ring.in_kernel.load(Ordering::Acquire) < ring.capacity {
            let Some(entry) = inner.backlog.peek().copied() else {
                break;
            };
            // SAFETY: the entry's pointer is into its record's buffer, which stays untouched
            // until its completion is popped; the file is overlapped.
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
                    inner.backlog.poll();
                    ring.in_kernel.fetch_add(1, Ordering::AcqRel);
                    built += 1;
                }
                Err(BuildError::Full) => {
                    drained = false;
                    break;
                }
                Err(BuildError::Other(error)) => {
                    inner.backlog.poll();
                    // Not built: the kernel never had the buffer.
                    self.fail_token(entry.user_data, error, false);
                }
            }
        }
        if built >= FLUSH_ENTRIES || ring.in_kernel.load(Ordering::Acquire) >= ring.capacity {
            drained = false;
        }
        if built > 0 || inner.unsubmitted {
            inner.unsubmitted = sys.ring_submit(ring.handle).is_err();
        }
        drained
    }

    /// The next completion of `ring`, if the ring is not busy. A ring that cannot be read ends
    /// its operations with the error.
    fn pop(&self, ring: &Arc<Ring>) -> Option<Cqe> {
        let claim = ring.inner.try_claim()?;
        let popped = self.sys.ring_pop(ring.handle);
        drop(claim);
        match popped {
            Ok(Some(cqe)) => {
                ring.in_kernel.fetch_sub(1, Ordering::AcqRel);
                Some(cqe)
            }
            Ok(None) => None,
            Err(error) => {
                self.break_ring(ring, &error);
                None
            }
        }
    }

    /// The ring cannot be read: every operation routed through it ends with `error`. Their
    /// buffers are leaked, since the kernel may still write them.
    fn break_ring(&self, ring: &Arc<Ring>, error: &io::Error) {
        if ring.broken.swap(true, Ordering::AcqRel) {
            return;
        }
        let mut failed = Vec::new();
        for slot in 0..self.slab.made() {
            let cell = self.cell(slot);
            let word = cell.load();
            if word.state() != State::InFlight
                || word.route() != RouteTag::Ring
                || word.ring() != ring.index
            {
                continue;
            }
            if let Ok(word) = cell.take(|word| {
                if word.state() == State::InFlight
                    && word.route() == RouteTag::Ring
                    && word.ring() == ring.index
                {
                    Ok(())
                } else {
                    Err(())
                }
            }) {
                failed.push((slot, word.generation(), cell.rec()));
            }
        }
        for (slot, generation, rec) in failed {
            // SAFETY: the slot is `Busy` now: ours.
            let op = unsafe { (*rec).op.as_mut() };
            if let Some(op) = op {
                op.buf.leak();
            }
            self.offer_failed(Failed {
                slot,
                generation,
                error: ring_failed_with(error),
            });
        }
        // The backlog is cleared by the holder of the claim. A flush that holds it clears it as it
        // finds the ring broken.
        if let Some(mut inner) = ring.inner.try_claim() {
            inner.backlog.drain_all(drop);
        }
    }

    // ---- starting operations

    /// A handle that would block the worker, or that completes without a packet, is refused.
    fn check_handle(&self, handle: Handle, socket: bool) -> io::Result<()> {
        let mode = self.sys.handle_mode(handle, socket)?;
        if mode.synchronous {
            return Err(invalid(
                "a handle opened without overlapped I/O would block the worker",
            ));
        }
        if mode.skips_port_on_success {
            return Err(invalid(
                "a handle that skips the completion port on success would never complete",
            ));
        }
        Ok(())
    }

    /// Forgets a handle value: the OS gives a closed handle's value to a new handle, which is not the one this
    /// engine joined. The engine calls this itself for every handle it makes (a socket, an accepted socket, an
    /// opened file); for a handle the caller made, the caller says so before it closes it, with nothing in flight
    /// on it. Without it, a new handle of the same value that sits on another port would pass for ours (the OS's
    /// refusal would be taken for "already joined"), and a new file would be kept off a ring.
    pub(super) fn forget(&self, handle: Handle) {
        self.associated.remove(handle);
        self.ringed.remove(handle);
    }

    /// Joins the handle to the engine's port. A handle already on a port is refused by the OS
    /// with `ERROR_INVALID_PARAMETER`: that is fine only for a handle this engine joined.
    fn join_port(&self, handle: Handle) -> io::Result<()> {
        match self.sys.associate(self.port.handle, handle) {
            Ok(()) => {
                self.associated.insert(handle);
                Ok(())
            }
            Err(error) if error.raw_os_error() == Some(87) => {
                if self.associated.contains(handle) {
                    Ok(())
                } else {
                    Err(invalid(
                        "the handle already belongs to another completion port (a handle serves one AsyncIo, ever)",
                    ))
                }
            }
            Err(error) => Err(error),
        }
    }

    /// The route of a file operation: a ring if the OS has them for files and one can be had,
    /// otherwise the port, which the handle then joins. A file keeps its route: IoRing refuses a
    /// handle that belongs to a port (`ERROR_INVALID_PARAMETER`), so a handle that joined the
    /// port stays on it, and one a ring serves is never joined.
    fn file_route(&self, handle: Handle) -> io::Result<Route> {
        if self.ring_files && !self.associated.contains(handle) {
            if let Some(ring) = self.ring() {
                self.ringed.insert(handle);
                return Ok(Route::Ring(ring));
            }
            if self.ringed.contains(handle) {
                return Err(io::Error::other(
                    "this file handle is served by IoRing, and no ring is available to this thread",
                ));
            }
        }
        self.join_port(handle)?;
        Ok(Route::Port)
    }

    fn op(kind: Kind, route: Route, buf: Buf, then: Then) -> Op {
        Op {
            kind,
            route,
            buf,
            moved: 0,
            then,
            addr: None,
            accepting: None,
            deadline: None,
        }
    }

    /// Starts the operation: its slot, then its call. The slot stays `Issuing` until the call is
    /// made (or, on a ring or the pool, until the entry or the job is handed over), so nothing
    /// that completes or fails operations touches the record while it is being filled.
    fn start(&self, op: Op, taken: &Cell<bool>) -> io::Result<OpId> {
        let held = self.begin(op, State::Issuing)?;
        // The table has the operation: from here on a panic is not a refusal.
        taken.set(true);
        let id = OpId::new(self.id as u64, held.slot, held.serial);
        self.launch(held);
        Ok(id)
    }

    /// Makes the call for the record's next piece. Nothing of the record is touched once the call
    /// is made.
    fn launch(&self, held: Held) {
        let call = self.call(held);
        let sys = &*self.sys;
        let made = match call {
            Call::Read { file, buf, len, ov } => {
                // SAFETY: the file is overlapped, joined to the port, and its mode checked; `buf`
                // points into the record's buffer and `ov` into the record, which stay untouched
                // until the packet is dequeued.
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
            Call::Pool(job) => {
                // The record is not touched again: the job's result may come at once.
                self.set_state(held, State::InFlight);
                // A job the pool refuses is dropped, and its delivery ends the operation with an
                // error.
                let _ = sys.run_on_pool(job);
                return;
            }
            Call::Ring(ring, entry) => {
                // The record is not touched again: the entry may complete as soon as it is built.
                self.set_state(held, State::InFlight);
                let queued = ring.incoming.offer(entry);
                debug_assert!(queued.is_ok(), "the unbounded queue refused an entry");
                self.end_if_broken(&ring, entry.user_data);
                if !ring.broken.load(Ordering::Acquire) {
                    self.flush_ring(&ring);
                }
                return;
            }
        };
        match made {
            // A packet follows. Its slot may be completed from now on.
            Ok(()) => self.set_state(held, State::InFlight),
            // The call failed at once and no packet follows: end it with the error.
            Err(error) => self.fail(held, error),
        }
    }

    /// The call for the record's next piece.
    fn call(&self, held: Held) -> Call {
        // SAFETY: the slot is `Issuing` or ours (`InFlight` with nothing issued): no other thread
        // and no kernel call refers to the record. The reference ends before the call is made;
        // the pointers taken from it are what the kernel gets.
        let rec = unsafe { &mut *held.rec.as_ptr() };
        let ov: *mut Overlapped = &raw mut rec.ov;
        let op = rec.op.as_mut().expect("a started operation");
        let moved = op.moved;
        let user_data = token(held.slot, held.generation);
        match &op.kind {
            Kind::Read { file, offset } => {
                let file = *file;
                let at = offset + moved as u64;
                // A read the page cache can serve is copied inside the call, on the submitting
                // worker: pieces of at most READ_PIECE, each piece's completion submitting the
                // next.
                let rest = &mut op.buf.bytes_mut()[moved..];
                let len = self.piece(rest.len().min(READ_PIECE));
                let buf = rest.as_mut_ptr();
                if let Route::Ring(ring) = &op.route {
                    let entry = Entry {
                        write: false,
                        file,
                        buf: buf as usize,
                        len,
                        offset: at,
                        user_data,
                    };
                    return Call::Ring(ring.clone(), entry);
                }
                reset(ov, at);
                Call::Read { file, buf, len, ov }
            }
            Kind::Write { file, offset } => {
                let file = *file;
                let at = offset + moved as u64;
                // Pieces of at most READ_PIECE too: a write that extends a file is synchronous on
                // NTFS (the call returns when the data is written), so its size is the worker's
                // wait.
                let rest = &op.buf.bytes()[moved..op.buf.filled()];
                let len = self.piece(rest.len().min(READ_PIECE));
                let buf = rest.as_ptr();
                if let Route::Ring(ring) = &op.route {
                    let entry = Entry {
                        write: true,
                        file,
                        buf: buf as usize,
                        len,
                        offset: at,
                        user_data,
                    };
                    return Call::Ring(ring.clone(), entry);
                }
                reset(ov, at);
                Call::Write { file, buf, len, ov }
            }
            Kind::Send { sock } => {
                let sock = *sock;
                let bytes = &op.buf.bytes()[moved..op.buf.filled()];
                let len = self.piece(bytes.len().min(READ_PIECE));
                let buf = bytes.as_ptr();
                reset(ov, 0);
                Call::Send { sock, buf, len, ov }
            }
            Kind::Recv { sock } => {
                let sock = *sock;
                let bytes = op.buf.bytes_mut();
                let len = self.piece(bytes.len());
                let buf = bytes.as_mut_ptr();
                reset(ov, 0);
                Call::Recv { sock, buf, len, ov }
            }
            Kind::SendTo { sock } => {
                let sock = *sock;
                let bytes = &op.buf.bytes()[..op.buf.filled()];
                let len = self.piece(bytes.len());
                let buf = bytes.as_ptr();
                let to: *const SockAddr = op.addr.as_deref().expect("a datagram's address");
                reset(ov, 0);
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
                let accepting = op.accepting.as_mut().expect("an accept's socket");
                let accepted = accepting.socket;
                let addresses = accepting.addresses.as_mut_ptr();
                reset(ov, 0);
                Call::Accept {
                    listener,
                    accepted,
                    addresses,
                    ov,
                }
            }
            Kind::Connect { sock } => {
                let sock = *sock;
                let to: *const SockAddr = op.addr.as_deref().expect("a connect's address");
                reset(ov, 0);
                Call::Connect { sock, to, ov }
            }
            Kind::Open { path, spec } => {
                let (path, spec) = (path.clone(), *spec);
                let sys = self.sys.clone();
                let delivery = self.delivery(held);
                Call::Pool(Box::new(move || {
                    delivery.deliver(sys.open_file(&path, spec));
                }))
            }
            Kind::FileLen { file } => {
                let file = *file;
                let sys = self.sys.clone();
                let delivery = self.delivery(held);
                Call::Pool(Box::new(move || {
                    delivery.deliver(sys.file_len(file).and_then(|len| {
                        usize::try_from(len).map_err(|_| io::Error::from_raw_os_error(534))
                    }));
                }))
            }
            Kind::Rename { from, to } => {
                let (from, to) = (from.clone(), to.clone());
                let sys = self.sys.clone();
                let delivery = self.delivery(held);
                Call::Pool(Box::new(move || {
                    delivery.deliver(sys.rename(&from, &to).map(|()| 0));
                }))
            }
        }
    }

    fn delivery(&self, held: Held) -> Delivery {
        Delivery {
            queue: self.pool_done.clone(),
            slot: held.slot,
            generation: held.generation,
            result: None,
        }
    }

    // ---- ending operations

    /// What a finished piece means for its operation: the rest goes in again, or it ended and its
    /// continuation's frame comes out. The slot is `Busy`.
    fn advance(
        &self,
        poller: &mut Poller,
        held: Held,
        result: io::Result<usize>,
        partial: usize,
    ) -> Option<Box<dyn Frame>> {
        // SAFETY: the slot is `Busy`: this thread has the record.
        let rec = unsafe { &mut *held.rec.as_ptr() };
        let op = rec.op.as_mut().expect("an operation in flight");
        let piecewise = matches!(
            op.kind,
            Kind::Read { .. } | Kind::Write { .. } | Kind::Send { .. }
        );
        let total = match op.kind {
            Kind::Read { .. } => op.buf.len(),
            _ => op.buf.filled(),
        };
        // A read at the end of the data is a read of no bytes.
        let result = match (&op.kind, result) {
            (Kind::Read { .. }, Err(error)) if is_end_of_file(&error) => Ok(0),
            (_, result) => result,
        };
        // A file the engine just opened has a value the system gave anew.
        if let (Kind::Open { .. }, Ok(file)) = (&op.kind, &result) {
            self.forget(*file);
        }
        let ended = match (&op.kind, result) {
            (_, Ok(n)) if piecewise && n > 0 => {
                op.moved += n;
                if op.moved < total {
                    match self.rebegin(held, State::Issuing) {
                        Ok(held) => {
                            self.launch(held);
                            return None;
                        }
                        // Asked to end between two pieces: the rest does not go in.
                        Err(why) => Err(why.error(op.moved)),
                    }
                } else {
                    Ok(op.moved)
                }
            }
            (Kind::Read { .. }, Ok(0)) => {
                if op.moved == 0 {
                    Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "the read is at or past the end of the file",
                    ))
                } else {
                    Ok(op.moved)
                }
            }
            (Kind::Write { .. }, Ok(0)) => Err(io::Error::from(io::ErrorKind::WriteZero)),
            (Kind::Send { .. }, Ok(0)) if op.buf.filled() > 0 => {
                Err(io::Error::from(io::ErrorKind::WriteZero))
            }
            (_, other) => other,
        };
        let ended = match op.kind {
            Kind::Accept { listener } => {
                let accepting = op.accepting.take().expect("an accept's socket");
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
        let mut op = rec.op.take().expect("an operation in flight");
        // A receive or read that failed with a warning (`ERROR_MORE_DATA`, `WSAEMSGSIZE`) still
        // moved bytes: they are in the buffer, and `filled` says how many.
        if ended.is_err() && partial > 0 && matches!(op.kind, Kind::Read { .. } | Kind::Recv { .. })
        {
            let filled = (op.moved + partial).min(op.buf.len());
            op.buf.set_filled(filled);
        }
        self.release(poller, held, op.deadline);
        Some(conclude(op, ended))
    }

    /// The result of a finished call, from its `NTSTATUS` and byte count, and the bytes a warning
    /// status delivered anyway.
    fn packet_result(&self, kind: &Kind, status: i32, bytes: u32) -> (io::Result<usize>, usize) {
        if status >= 0 {
            return (Ok(bytes as usize), 0);
        }
        if status == STATUS_END_OF_FILE && matches!(kind, Kind::Read { .. }) {
            return (Ok(0), 0);
        }
        let socket = !matches!(kind, Kind::Read { .. } | Kind::Write { .. });
        // A warning (`ERROR_MORE_DATA`, `WSAEMSGSIZE`) is a failure that moved bytes.
        let partial = if is_nt_warning(status) {
            bytes as usize
        } else {
            0
        };
        (Err(self.sys.status_error(status, socket)), partial)
    }

    /// Ends the operations of `batch`, handing frames to `offer`. Packets of slots whose call is
    /// still returning go to `later`. `Some(progress)`: whether any packet was not `Later`;
    /// `None` once `offer` kept a frame back.
    fn process_packets(
        &self,
        poller: &mut Poller,
        batch: &mut VecDeque<Packet>,
        later: &mut VecDeque<Packet>,
        room: &mut dyn FnMut() -> bool,
        offer: &mut dyn FnMut(Box<dyn Frame>) -> bool,
    ) -> Option<bool> {
        let mut progress = false;
        while let Some(packet) = batch.pop_front() {
            if !room() {
                batch.push_front(packet);
                break;
            }
            match self.take_packet(&packet) {
                Taken::Stray => {
                    progress = true;
                    self.stray();
                }
                Taken::Later => later.push_back(packet),
                Taken::Op(held, status, bytes, end) => {
                    progress = true;
                    let (result, partial) = {
                        // SAFETY: the slot is `Busy`: this thread has the record.
                        let rec = unsafe { &*held.rec.as_ptr() };
                        let op = rec.op.as_ref().expect("an operation in flight");
                        let (result, partial) = self.packet_result(&op.kind, status, bytes);
                        // A call that was cancelled for a reason ends with it.
                        match end {
                            Some(why) if status == STATUS_CANCELLED => {
                                (Err(why.error(op.moved)), partial)
                            }
                            _ => (result, partial),
                        }
                    };
                    if let Some(frame) = self.advance(poller, held, result, partial)
                        && !offer(frame)
                    {
                        return None;
                    }
                }
            }
        }
        Some(progress)
    }

    fn dequeue(&self) -> VecDeque<Packet> {
        let mut taken = Vec::new();
        // An error leaves nothing taken: the next poll asks again.
        let _ = self.sys.dequeue(self.port.handle, &mut taken, BATCH);
        taken.into()
    }

    /// Packets of the port: ends operations, hands frames to `offer`. `false` once `offer` kept a
    /// frame back.
    fn poll_port(
        &self,
        poller: &mut Poller,
        room: &mut dyn FnMut() -> bool,
        offer: &mut dyn FnMut(Box<dyn Frame>) -> bool,
    ) -> bool {
        let mut batch: VecDeque<Packet> = std::mem::take(&mut poller.inbox);
        let from_inbox = !batch.is_empty();
        if !from_inbox {
            batch = self.dequeue();
        }
        // Packets of slots whose call is still returning wait for the next poll.
        let mut later = VecDeque::new();
        let mut go_on = true;
        match self.process_packets(poller, &mut batch, &mut later, room, offer) {
            None => go_on = false,
            // The inbox held only packets of calls still returning: a preempted issuer must not
            // stall the completions behind them, so look at the port as well.
            Some(false) if from_inbox && batch.is_empty() && room() => {
                let mut fresh = self.dequeue();
                if self
                    .process_packets(poller, &mut fresh, &mut later, room, offer)
                    .is_none()
                {
                    go_on = false;
                }
                batch.extend(fresh);
            }
            Some(_) => {}
        }
        later.extend(batch);
        poller.inbox = later;
        go_on
    }

    /// Operations whose call failed at once, or whose ring broke.
    fn poll_failed(
        &self,
        poller: &mut Poller,
        room: &mut dyn FnMut() -> bool,
        offer: &mut dyn FnMut(Box<dyn Frame>) -> bool,
    ) -> bool {
        while room() {
            let Some(failed) = poller.failed.poll() else {
                break;
            };
            let Some(held) = self.take_failed(&failed) else {
                self.stray();
                continue;
            };
            if let Some(frame) = self.advance(poller, held, Err(failed.error), 0)
                && !offer(frame)
            {
                return false;
            }
        }
        true
    }
}

/// The error of an operation on a ring that cannot be read.
fn ring_failed() -> io::Error {
    io::Error::other("the IoRing failed")
}

fn ring_failed_with(error: &io::Error) -> io::Error {
    io::Error::new(error.kind(), format!("the IoRing failed: {error}"))
}

/// Clears the `OVERLAPPED` for a new call at `offset`.
fn reset(ov: *mut Overlapped, offset: u64) {
    // SAFETY: `ov` points at the record's `OVERLAPPED`, which no kernel call refers to between
    // calls, and which this thread has (its slot is `Issuing`, or ours).
    unsafe { (*ov).reset(offset) };
}

/// The continuation's frame for an ended operation.
fn conclude(op: Op, result: io::Result<usize>) -> Box<dyn Frame> {
    let Op {
        kind, buf, then, ..
    } = op;
    let fills = matches!(kind, Kind::Read { .. } | Kind::Recv { .. });
    conclude_io(buf, then, fills, result)
}

impl EngineTrait for Engine {
    fn submit(
        &self,
        request: Request,
        buf: Buf,
        then: Then,
        limit: Option<Duration>,
        taken: &Cell<bool>,
    ) -> io::Result<OpId> {
        let mut op = match request {
            Request::Read { file, offset } => {
                self.check_handle(file, false)?;
                let route = self.file_route(file)?;
                Self::op(Kind::Read { file, offset }, route, buf, then)
            }
            Request::Write { file, offset } => {
                self.check_handle(file, false)?;
                let route = self.file_route(file)?;
                Self::op(Kind::Write { file, offset }, route, buf, then)
            }
            Request::Send { sock } => {
                self.check_handle(sock, true)?;
                self.join_port(sock)?;
                Self::op(Kind::Send { sock }, Route::Port, buf, then)
            }
            Request::Recv { sock } => {
                self.check_handle(sock, true)?;
                self.join_port(sock)?;
                Self::op(Kind::Recv { sock }, Route::Port, buf, then)
            }
            Request::SendTo { sock, to } => {
                self.check_handle(sock, true)?;
                self.join_port(sock)?;
                let mut op = Self::op(Kind::SendTo { sock }, Route::Port, buf, then);
                op.addr = Some(Box::new(SockAddr::new(to)));
                op
            }
            Request::Accept { listener } => {
                self.check_handle(listener, true)?;
                self.join_port(listener)?;
                let socket = self.sys.accept_socket(listener)?;
                self.forget(socket);
                let mut op = Self::op(Kind::Accept { listener }, Route::Port, buf, then);
                op.accepting = Some(Accepting {
                    socket,
                    addresses: Box::new([0; ACCEPT_BUF]),
                });
                op
            }
            Request::Connect { sock, to } => {
                self.check_handle(sock, true)?;
                self.join_port(sock)?;
                // `ConnectEx` needs a bound socket.
                self.sys.bind_any(sock, &SockAddr::unspecified(to))?;
                let mut op = Self::op(Kind::Connect { sock }, Route::Port, buf, then);
                op.addr = Some(Box::new(SockAddr::new(to)));
                op
            }
            Request::Open { path, spec } => {
                Self::op(Kind::Open { path, spec }, Route::Pool, buf, then)
            }
            Request::Rename { from, to } => {
                Self::op(Kind::Rename { from, to }, Route::Pool, buf, then)
            }
            Request::FileLen { file } => Self::op(Kind::FileLen { file }, Route::Pool, buf, then),
            #[cfg(any(target_os = "linux", target_os = "android"))]
            Request::OpenAt { .. } => return Err(invalid("open_file is not available here")),
        };
        op.deadline = limit.map(|limit| {
            self.sys
                .now_ns()
                .saturating_add(u64::try_from(limit.as_nanos()).unwrap_or(u64::MAX))
        });
        // An accept's socket made above is closed if no slot can be had.
        let accepted = op.accepting.as_ref().map(|accepting| accepting.socket);
        let started = self.start(op, taken);
        if started.is_err()
            && let Some(socket) = accepted
        {
            self.sys.close_socket(socket);
        }
        started
    }

    fn cancel(&self, id: OpId) {
        if id.engine() == self.id as u64 {
            self.request_end(id.lane(), id.token(), End::Cancelled);
        }
    }

    fn poll(&self, room: &mut dyn FnMut() -> bool, offer: &mut dyn FnMut(Box<dyn Frame>) -> bool) {
        // SAFETY: the lattice runs one poll at a time (the reaper cell admits one).
        let mut poller = unsafe { self.poller.enter() };
        let poller = &mut *poller;
        self.expire_limits(poller);
        if !self.poll_failed(poller, room, offer) {
            return;
        }
        // What open, rename and file length jobs left.
        while room() {
            let Some(done) = poller.pool_done.poll() else {
                break;
            };
            let Some(held) = self.take_token(done.slot, done.generation as usize, None) else {
                self.stray();
                continue;
            };
            if let Some(frame) = self.advance(poller, held, done.result, 0)
                && !offer(frame)
            {
                return;
            }
        }
        if !self.poll_port(poller, room, offer) {
            return;
        }
        let made = self.ring_count.load(Ordering::Acquire).min(MAX_RINGS);
        for ring in self.rings[..made].iter().filter_map(OnceLock::get) {
            self.flush_ring(ring);
            while room() {
                let Some(cqe) = self.pop(ring) else {
                    break;
                };
                let (slot, generation) = untoken(cqe.user_data);
                let Some(held) = self.take_token(slot, generation, Some(ring)) else {
                    self.stray();
                    continue;
                };
                let result = if cqe.result >= 0 {
                    Ok(cqe.info)
                } else {
                    Err(hresult_error(cqe.result))
                };
                if let Some(frame) = self.advance(poller, held, result, 0)
                    && !offer(frame)
                {
                    self.flush_ring(ring);
                    return;
                }
            }
            // Room freed in the completion queue may let the backlog in.
            self.flush_ring(ring);
        }
        // Operations of a ring that broke during this poll, and those that failed to build.
        self.poll_failed(poller, room, offer);
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
