//! The Windows engine of `AsyncIo`: I/O completion ports for every operation, and IoRing (a kernel
//! ring, as io_uring is) for file reads and writes where the OS has one.
//!
//! - **One completion port per `AsyncIo`.** Any worker issues overlapped calls; the lattice's
//!   polls dequeue (`GetQueuedCompletionStatusEx` with a zero timeout, so nothing waits). Every
//!   operation ends through a poll: the "skip the port on success" mode is not set, and a handle
//!   that has it set is refused.
//! - **One IoRing per submitting thread**, for file reads and writes, when the OS offers a kernel
//!   ring ([`Sys::ring_probe`]: not emulated in user mode, with the operation supported). A full
//!   submission queue goes to the ring's backlog and is handed over by a later submission or poll.
//!   Each operation falls back to the port on its own.
//! - **Open and rename** have no asynchronous call. They run on the OS thread pool and leave their
//!   result in a queue the poll drains, so a job that panics or is dropped still ends its
//!   operation, with an error. It is the one place the crate uses threads it did not create.
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

use std::cell::RefCell;
use std::collections::{HashMap, HashSet, VecDeque};
use std::io;
use std::path::PathBuf;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, Weak};

use euhedral_core::frames::Frame;

use self::sys::{
    ACCEPT_BUF, BuildError, Cqe, Handle, IORING_VERSION_3, Overlapped, Packet, STATUS_END_OF_FILE,
    STATUS_PENDING, SockAddr, Sys, hresult_error, is_end_of_file,
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
/// Entries one flush builds before it submits: a submit that copies is not unbounded work for the
/// worker that made it.
const FLUSH_ENTRIES: usize = 16;
/// Bits of a ring entry's user data that name the slot.
const SLOT_BITS: u32 = 20;
/// Operations in flight at once.
const MAX_OPS: usize = (1 << SLOT_BITS) - 1;
/// What the generation of a token keeps (a token is a `usize`).
const GENERATION_MASK: usize = usize::MAX >> SLOT_BITS;

static NEXT_ID: AtomicUsize = AtomicUsize::new(1);

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
    buf: IoBuf,
    /// Bytes moved so far (reads, writes and sends resubmit the rest).
    moved: usize,
    then: Then,
    /// The destination of a datagram or a connect.
    addr: Option<Box<SockAddr>>,
    accepting: Option<Accepting>,
}

/// A slot's record. `ov` is first (it is what the kernel is given); the record is never freed
/// while the engine lives.
#[repr(C)]
struct Rec {
    ov: Overlapped,
    op: Option<Op>,
}

/// A slot's state; see the module docs.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum State {
    Idle,
    Issuing,
    InFlight,
    Busy,
}

/// A pointer to a slab record, which the slot's state makes exclusive to one thread at a time
/// (or, for `ov` and the buffers while `InFlight`, to the kernel).
struct RecPtr(NonNull<Rec>);

// SAFETY: a record is touched by one thread at a time, as the slot's state says: the issuer while
// `Issuing`, the thread that took it while `Busy`, and the kernel (through raw pointers) while
// `InFlight`. The state changes under the table's lock.
unsafe impl Send for RecPtr {}

struct Slot {
    rec: RecPtr,
    generation: u32,
    state: State,
}

/// The slab of records.
#[derive(Default)]
struct Table {
    slots: Vec<Slot>,
    free: Vec<u32>,
    /// The slot of each record, by the address of its `OVERLAPPED`.
    by_addr: HashMap<usize, u32>,
}

impl Drop for Table {
    fn drop(&mut self) {
        for slot in &self.slots {
            // An operation in flight leaks: the kernel may still write into its buffers.
            if slot.state == State::Idle {
                // SAFETY: made by `Box::leak`, idle, so nothing else refers to it.
                drop(unsafe { Box::from_raw(slot.rec.0.as_ptr()) });
            }
        }
    }
}

/// An operation a thread holds: its slot is `Issuing` or `Busy` (or `InFlight` on a ring or the
/// pool, until its entry is pushed).
#[derive(Clone, Copy)]
struct Held {
    slot: u32,
    generation: u32,
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

/// What the pool left for an open or rename.
struct PoolDone {
    slot: u32,
    generation: u32,
    result: io::Result<usize>,
}

type PoolQueue = Arc<Mutex<VecDeque<PoolDone>>>;

/// Delivers a pool job's result, or, if the job is dropped or panics first, an error.
struct Delivery {
    queue: PoolQueue,
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
        lock(&self.queue).push_back(PoolDone {
            slot: self.slot,
            generation: self.generation,
            result,
        });
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
    handle: Handle,
    /// Building entries and submitting: one thread at a time, and a thread that finds it taken
    /// leaves its entry in the backlog for the holder or the next poll.
    inner: Mutex<RingInner>,
    /// Entries waiting for room in the kernel's queues, the oldest first. Its own lock, held only
    /// to push or take, so a submission never waits behind a submit.
    backlog: Mutex<VecDeque<Entry>>,
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

pub(super) struct Engine {
    id: usize,
    sys: Arc<dyn Sys>,
    port: Port,
    table: Mutex<Table>,
    /// Operations whose call failed at once, to end with that error at the next poll.
    ready: Mutex<VecDeque<Failed>>,
    /// What open and rename jobs left.
    pool_done: PoolQueue,
    /// Packets taken from the port and not yet handled (a poll stops when the caller has room for
    /// no more, and leaves the packets of slots whose call is still returning); only the polling
    /// thread touches it.
    inbox: Mutex<VecDeque<Packet>>,
    /// The handles this engine joined to its port.
    associated: Mutex<HashSet<Handle>>,
    ring_reads: bool,
    ring_writes: bool,
    rings: Mutex<Vec<Arc<Ring>>>,
    max_piece: usize,
    /// Packets and completions of no operation (ignored), for tests.
    strays: AtomicUsize,
}

/// What a dequeued packet leads to.
enum Taken {
    /// The operation, now `Busy`.
    Op(Held),
    /// Its call is still returning: look again at the next poll.
    Later,
    /// A packet of no operation in flight on the port.
    Stray,
}

impl Engine {
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
        Ok(Self {
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            sys,
            port,
            table: Mutex::new(Table::default()),
            ready: Mutex::new(VecDeque::new()),
            pool_done: Arc::default(),
            inbox: Mutex::new(VecDeque::new()),
            associated: Mutex::new(HashSet::new()),
            ring_reads: probe.is_some_and(|probe| probe.read),
            ring_writes: probe
                .is_some_and(|probe| probe.write && probe.version >= IORING_VERSION_3),
            rings: Mutex::new(Vec::new()),
            max_piece: max_piece.clamp(1, MAX_PIECE),
            strays: AtomicUsize::new(0),
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

    fn stray(&self) {
        self.strays.fetch_add(1, Ordering::Relaxed);
    }

    // ---- the slab

    /// A slot for `op`, in `state`: its record is filled, and nothing else refers to it.
    fn begin(&self, op: Op, state: State) -> io::Result<Held> {
        let mut table = lock(&self.table);
        let slot = match table.free.pop() {
            Some(slot) => slot,
            None => {
                if table.slots.len() >= MAX_OPS {
                    return Err(io::Error::other("too many operations in flight"));
                }
                let rec = Box::new(Rec {
                    ov: Overlapped::default(),
                    op: None,
                });
                let rec = NonNull::from(Box::leak(rec));
                let slot = table.slots.len() as u32;
                table.by_addr.insert(rec.as_ptr() as usize, slot);
                table.slots.push(Slot {
                    rec: RecPtr(rec),
                    generation: 0,
                    state: State::Idle,
                });
                slot
            }
        };
        let entry = &mut table.slots[slot as usize];
        debug_assert_eq!(entry.state, State::Idle);
        entry.state = state;
        entry.generation = entry.generation.wrapping_add(1);
        let held = Held {
            slot,
            generation: entry.generation,
            rec: entry.rec.0,
        };
        drop(table);
        // SAFETY: the slot was `Idle` and is now ours (`Issuing`, or `InFlight` with nothing
        // issued yet): no other thread and no kernel call refers to the record.
        unsafe { (*held.rec.as_ptr()).op = Some(op) };
        Ok(held)
    }

    /// A taken operation goes in again: a new issue of the same slot.
    fn rebegin(&self, held: Held, state: State) -> Held {
        let mut table = lock(&self.table);
        let entry = &mut table.slots[held.slot as usize];
        debug_assert_eq!(entry.state, State::Busy);
        entry.state = state;
        entry.generation = entry.generation.wrapping_add(1);
        Held {
            generation: entry.generation,
            ..held
        }
    }

    fn set_state(&self, held: Held, state: State) {
        lock(&self.table).slots[held.slot as usize].state = state;
    }

    /// The slot is free again.
    fn release(&self, held: Held) {
        let mut table = lock(&self.table);
        table.slots[held.slot as usize].state = State::Idle;
        table.free.push(held.slot);
    }

    /// The operation whose `OVERLAPPED` is at `addr`, if it is in flight on the port and its call
    /// finished: its slot becomes `Busy`.
    fn take_packet(&self, addr: usize) -> Taken {
        let mut table = lock(&self.table);
        let Some(&slot) = table.by_addr.get(&addr) else {
            return Taken::Stray;
        };
        let entry = &mut table.slots[slot as usize];
        match entry.state {
            State::Idle | State::Busy => return Taken::Stray,
            // The call is still returning: its caller has not looked at the record yet.
            State::Issuing => return Taken::Later,
            State::InFlight => {}
        }
        let rec = entry.rec.0.as_ptr();
        // SAFETY: a slab record, valid while the engine lives. `ov.Internal` is written by the
        // kernel at completion, so it is read volatile; `op` is read while the slot is
        // `InFlight`, when nothing writes it.
        let (internal, on_port) = unsafe {
            (
                std::ptr::read_volatile(&raw const (*rec).ov.internal),
                matches!((*rec).op.as_ref().map(|op| &op.route), Some(Route::Port)),
            )
        };
        // A packet of an earlier use of the slot, while the new call is pending, or a packet of
        // an operation that is not on the port.
        if !on_port || internal as u32 as i32 == STATUS_PENDING {
            return Taken::Stray;
        }
        entry.state = State::Busy;
        Taken::Op(Held {
            slot,
            generation: entry.generation,
            rec: entry.rec.0,
        })
    }

    /// The operation a ring completion (`ring`) or a pool result (`None`) names, if its slot is
    /// `InFlight` on that route with that generation: the slot becomes `Busy`.
    fn take_token(&self, slot: u32, generation: usize, ring: Option<&Arc<Ring>>) -> Option<Held> {
        let mut table = lock(&self.table);
        let entry = table.slots.get_mut(slot as usize)?;
        if entry.state != State::InFlight
            || (entry.generation as usize) & GENERATION_MASK != generation & GENERATION_MASK
        {
            return None;
        }
        // SAFETY: a slab record; `op` is read while the slot is `InFlight`, when nothing writes
        // it.
        let route_ok = unsafe {
            match (
                (*entry.rec.0.as_ptr()).op.as_ref().map(|op| &op.route),
                ring,
            ) {
                (Some(Route::Pool), None) => true,
                (Some(Route::Ring(own)), Some(ring)) => Arc::ptr_eq(own, ring),
                _ => false,
            }
        };
        if !route_ok {
            return None;
        }
        entry.state = State::Busy;
        Some(Held {
            slot,
            generation: entry.generation,
            rec: entry.rec.0,
        })
    }

    /// The operation that failed at its call (a `Busy` slot of that generation).
    fn take_failed(&self, failed: &Failed) -> Option<Held> {
        let table = lock(&self.table);
        let entry = table.slots.get(failed.slot as usize)?;
        (entry.state == State::Busy && entry.generation == failed.generation).then_some(Held {
            slot: failed.slot,
            generation: failed.generation,
            rec: entry.rec.0,
        })
    }

    /// The operation fails at its call: it ends with `error` at the next poll.
    fn fail(&self, held: Held, error: io::Error) {
        self.set_state(held, State::Busy);
        lock(&self.ready).push_back(Failed {
            slot: held.slot,
            generation: held.generation,
            error,
        });
    }

    /// A ring entry that cannot be built ends its operation with `error`.
    fn fail_token(&self, user_data: usize, error: io::Error) {
        let (slot, generation) = untoken(user_data);
        let held = {
            let mut table = lock(&self.table);
            let Some(entry) = table.slots.get_mut(slot as usize) else {
                return;
            };
            if entry.state != State::InFlight
                || (entry.generation as usize) & GENERATION_MASK != generation & GENERATION_MASK
            {
                return;
            }
            entry.state = State::Busy;
            (slot, entry.generation)
        };
        lock(&self.ready).push_back(Failed {
            slot: held.0,
            generation: held.1,
            error,
        });
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
            let mut rings = lock(&self.rings);
            if rings.len() >= MAX_RINGS {
                return None;
            }
            let handle = self.sys.ring_create(RING_ENTRIES).ok()?;
            let ring = Arc::new(Ring {
                sys: self.sys.clone(),
                handle,
                inner: Mutex::new(RingInner { unsubmitted: false }),
                backlog: Mutex::new(VecDeque::new()),
                in_kernel: AtomicUsize::new(0),
                capacity: 2 * RING_ENTRIES as usize,
                broken: AtomicBool::new(false),
            });
            rings.push(ring.clone());
            mine.push((self.id, Arc::downgrade(&ring)));
            Some(ring)
        })
    }

    /// Hands what waits in the ring's backlog (or was built and not submitted) to the kernel,
    /// unless another thread is doing so right now.
    fn flush_ring(&self, ring: &Ring) {
        let Ok(mut inner) = ring.inner.try_lock() else {
            return;
        };
        let sys = &*self.sys;
        let mut built = 0;
        while built < FLUSH_ENTRIES && ring.in_kernel.load(Ordering::Acquire) < ring.capacity {
            let Some(entry) = lock(&ring.backlog).front().copied() else {
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
                    lock(&ring.backlog).pop_front();
                    ring.in_kernel.fetch_add(1, Ordering::AcqRel);
                    built += 1;
                }
                Err(BuildError::Full) => break,
                Err(BuildError::Other(error)) => {
                    lock(&ring.backlog).pop_front();
                    self.fail_token(entry.user_data, error);
                }
            }
        }
        if built > 0 || inner.unsubmitted {
            inner.unsubmitted = sys.ring_submit(ring.handle).is_err();
        }
    }

    /// The next completion of `ring`, if the ring is not busy. A ring that cannot be read ends
    /// its operations with the error.
    fn pop(&self, ring: &Arc<Ring>) -> Option<Cqe> {
        let Ok(inner) = ring.inner.try_lock() else {
            return None;
        };
        let popped = self.sys.ring_pop(ring.handle);
        drop(inner);
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
        {
            let mut table = lock(&self.table);
            for (slot, entry) in table.slots.iter_mut().enumerate() {
                if entry.state != State::InFlight {
                    continue;
                }
                // SAFETY: a slab record whose slot is `InFlight`: `op` is not being written.
                let ours = unsafe {
                    matches!(
                        (*entry.rec.0.as_ptr()).op.as_ref().map(|op| &op.route),
                        Some(Route::Ring(own)) if Arc::ptr_eq(own, ring)
                    )
                };
                if ours {
                    entry.state = State::Busy;
                    failed.push((slot as u32, entry.generation, entry.rec.0));
                }
            }
        }
        for (slot, generation, rec) in failed {
            // SAFETY: the slot is `Busy` now: ours.
            let op = unsafe { (*rec.as_ptr()).op.as_mut() };
            if let Some(op) = op {
                std::mem::forget(std::mem::take(&mut op.buf.bytes));
            }
            lock(&self.ready).push_back(Failed {
                slot,
                generation,
                error: io::Error::new(error.kind(), format!("the IoRing failed: {error}")),
            });
        }
        lock(&ring.backlog).clear();
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

    /// Joins the handle to the engine's port. A handle already on a port is refused by the OS
    /// with `ERROR_INVALID_PARAMETER`: that is fine only for a handle this engine joined.
    fn join_port(&self, handle: Handle) -> io::Result<()> {
        match self.sys.associate(self.port.handle, handle) {
            Ok(()) => {
                lock(&self.associated).insert(handle);
                Ok(())
            }
            Err(error) if error.raw_os_error() == Some(87) => {
                if lock(&self.associated).contains(&handle) {
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

    /// The route of a file operation: a ring if one serves this kind of operation and can be had,
    /// otherwise the port, which the handle then joins.
    fn file_route(&self, handle: Handle, ring_ok: bool) -> io::Result<Route> {
        if ring_ok && let Some(ring) = self.ring() {
            return Ok(Route::Ring(ring));
        }
        self.join_port(handle)?;
        Ok(Route::Port)
    }

    fn op(kind: Kind, route: Route, buf: IoBuf, then: Then) -> Op {
        Op {
            kind,
            route,
            buf,
            moved: 0,
            then,
            addr: None,
            accepting: None,
        }
    }

    /// Starts the operation: its slot, then its call.
    fn start(&self, op: Op) -> io::Result<()> {
        let state = match op.route {
            Route::Port => State::Issuing,
            Route::Ring(_) | Route::Pool => State::InFlight,
        };
        let held = self.begin(op, state)?;
        self.launch(held);
        Ok(())
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
                // A job the pool refuses is dropped, and its delivery ends the operation with an
                // error.
                let _ = sys.run_on_pool(job);
                return;
            }
            Call::Ring(ring, entry) => {
                lock(&ring.backlog).push_back(entry);
                self.flush_ring(&ring);
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
                let rest = &mut op.buf.bytes[moved..];
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
                let rest = &op.buf.bytes[moved..op.buf.filled];
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
                let bytes = &op.buf.bytes[moved..op.buf.filled];
                let len = self.piece(bytes.len().min(READ_PIECE));
                let buf = bytes.as_ptr();
                reset(ov, 0);
                Call::Send { sock, buf, len, ov }
            }
            Kind::Recv { sock } => {
                let sock = *sock;
                let bytes = &mut op.buf.bytes[..];
                let len = self.piece(bytes.len());
                let buf = bytes.as_mut_ptr();
                reset(ov, 0);
                Call::Recv { sock, buf, len, ov }
            }
            Kind::SendTo { sock } => {
                let sock = *sock;
                let bytes = &op.buf.bytes[..op.buf.filled];
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
    fn advance(&self, held: Held, result: io::Result<usize>) -> Option<Box<dyn Frame>> {
        // SAFETY: the slot is `Busy`: this thread has the record.
        let rec = unsafe { &mut *held.rec.as_ptr() };
        let op = rec.op.as_mut().expect("an operation in flight");
        let piecewise = matches!(
            op.kind,
            Kind::Read { .. } | Kind::Write { .. } | Kind::Send { .. }
        );
        let total = match op.kind {
            Kind::Read { .. } => op.buf.bytes.len(),
            _ => op.buf.filled,
        };
        // A read at the end of the data is a read of no bytes.
        let result = match (&op.kind, result) {
            (Kind::Read { .. }, Err(error)) if is_end_of_file(&error) => Ok(0),
            (_, result) => result,
        };
        let ended = match (&op.kind, result) {
            (_, Ok(n)) if piecewise && n > 0 => {
                op.moved += n;
                if op.moved < total {
                    let state = match op.route {
                        Route::Port => State::Issuing,
                        _ => State::InFlight,
                    };
                    let held = self.rebegin(held, state);
                    self.launch(held);
                    return None;
                }
                Ok(op.moved)
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
            (Kind::Send { .. }, Ok(0)) if op.buf.filled > 0 => {
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
        let op = rec.op.take().expect("an operation in flight");
        self.release(held);
        Some(conclude(op, ended))
    }

    /// The result of a dequeued packet.
    fn packet_result(&self, kind: &Kind, packet: &Packet) -> io::Result<usize> {
        if packet.status >= 0 {
            return Ok(packet.bytes as usize);
        }
        if packet.status == STATUS_END_OF_FILE && matches!(kind, Kind::Read { .. }) {
            return Ok(0);
        }
        let socket = !matches!(kind, Kind::Read { .. } | Kind::Write { .. });
        Err(self.sys.status_error(packet.status, socket))
    }

    /// Packets of the port: ends operations, hands frames to `offer`. `false` once `offer` kept a
    /// frame back.
    fn poll_port(
        &self,
        room: &mut dyn FnMut() -> bool,
        offer: &mut dyn FnMut(Box<dyn Frame>) -> bool,
    ) -> bool {
        let mut batch: VecDeque<Packet> = std::mem::take(&mut *lock(&self.inbox));
        if batch.is_empty() {
            let mut taken = Vec::new();
            if self
                .sys
                .dequeue(self.port.handle, &mut taken, BATCH)
                .is_ok()
            {
                batch.extend(taken);
            }
        }
        // Packets of slots whose call is still returning wait for the next poll.
        let mut later = VecDeque::new();
        let mut go_on = true;
        while let Some(packet) = batch.pop_front() {
            if !room() {
                batch.push_front(packet);
                break;
            }
            match self.take_packet(packet.overlapped) {
                Taken::Stray => self.stray(),
                Taken::Later => later.push_back(packet),
                Taken::Op(held) => {
                    let result = {
                        // SAFETY: the slot is `Busy`: this thread has the record.
                        let rec = unsafe { &*held.rec.as_ptr() };
                        let kind = &rec.op.as_ref().expect("an operation in flight").kind;
                        self.packet_result(kind, &packet)
                    };
                    if let Some(frame) = self.advance(held, result)
                        && !offer(frame)
                    {
                        go_on = false;
                        break;
                    }
                }
            }
        }
        later.extend(batch);
        *lock(&self.inbox) = later;
        go_on
    }

    /// Operations whose call failed at once, or whose ring broke.
    fn poll_failed(
        &self,
        room: &mut dyn FnMut() -> bool,
        offer: &mut dyn FnMut(Box<dyn Frame>) -> bool,
    ) -> bool {
        while room() {
            let Some(failed) = lock(&self.ready).pop_front() else {
                break;
            };
            let Some(held) = self.take_failed(&failed) else {
                self.stray();
                continue;
            };
            if let Some(frame) = self.advance(held, Err(failed.error))
                && !offer(frame)
            {
                return false;
            }
        }
        true
    }
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
        kind,
        mut buf,
        then,
        ..
    } = op;
    match then {
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
        let op = match request {
            Request::Read { file, offset } => {
                self.check_handle(file, false)?;
                let route = self.file_route(file, self.ring_reads)?;
                Self::op(Kind::Read { file, offset }, route, buf, then)
            }
            Request::Write { file, offset } => {
                self.check_handle(file, false)?;
                let route = self.file_route(file, self.ring_writes)?;
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
            #[cfg(any(target_os = "linux", target_os = "android"))]
            Request::OpenAt { .. } => return Err(invalid("open_file is not available here")),
        };
        // An accept's socket made above is closed if no slot can be had.
        let accepted = op.accepting.as_ref().map(|accepting| accepting.socket);
        let started = self.start(op);
        if started.is_err()
            && let Some(socket) = accepted
        {
            self.sys.close_socket(socket);
        }
        started
    }

    fn poll(&self, room: &mut dyn FnMut() -> bool, offer: &mut dyn FnMut(Box<dyn Frame>) -> bool) {
        if !self.poll_failed(room, offer) {
            return;
        }
        // What open and rename jobs left.
        while room() {
            let Some(done) = lock(&self.pool_done).pop_front() else {
                break;
            };
            let Some(held) = self.take_token(done.slot, done.generation as usize, None) else {
                self.stray();
                continue;
            };
            if let Some(frame) = self.advance(held, done.result)
                && !offer(frame)
            {
                return;
            }
        }
        if !self.poll_port(room, offer) {
            return;
        }
        let rings: Vec<Arc<Ring>> = lock(&self.rings).clone();
        for ring in &rings {
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
                if let Some(frame) = self.advance(held, result)
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
        self.poll_failed(room, offer);
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
