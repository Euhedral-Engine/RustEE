//! The io_uring backend of [`AsyncIo`], for Linux and Android.
//!
//! Besides the portable half's handle conversions and `IoMemory` contract, the only code of the crate
//! with unsafe code: the io_uring rings (`// SAFETY:` on every block).
#![allow(unsafe_code)]

use std::cell::{Cell, RefCell};
use std::collections::VecDeque;
use std::ffi::CString;
use std::io;
use std::os::fd::RawFd;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
#[allow(
    clippy::disallowed_types,
    reason = "HOT-FIX(I3), HOT-FIX(I4), HOT-FIX(I5), HOT-FIX(I6)"
)]
use std::sync::{Arc, Mutex, TryLockError, Weak};
use std::time::Duration;

use euhedral_core::frames::Frame;
use euhedral_data_structures::queues::{Consumer, MpscQueue, Multi, Producer, Single};
use io_uring::{IoUring, Submitter, opcode, squeue, types};

use super::{
    AsyncIo, Buf, Cancelled, Create, Engine, LimitPassed, MAX_PIECE, OpId, OpenSpec, READ_PIECE,
    Request, Then, conclude as conclude_io, note_stray,
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
    #[allow(clippy::disallowed_types, reason = "HOT-FIX(I3)")]
    rings: Mutex<Vec<Arc<Ring>>>,
    enter: Box<Enter>,
    /// Entries of each ring's submission queue.
    ring_entries: u32,
    /// The most bytes one read, write or send hands the kernel; a test's stand-in lowers it to
    /// force short ones.
    max_piece: usize,
}

/// The user data of entries whose completion means nothing to an operation: an async cancel, and a
/// linked timeout (which ends with `ETIME` when it fired, or `ECANCELED` when its operation ended
/// first; either way the operation's own completion says what happened).
const IGNORED: u64 = u64::MAX;

static NEXT_ID: AtomicUsize = AtomicUsize::new(1);

thread_local! {
    /// This thread's ring of each `AsyncIo` it submitted to (by id).
    static MINE: RefCell<Vec<(usize, Weak<Ring>)>> = const { RefCell::new(Vec::new()) };
}

/// One submitting thread's ring and the operations in flight on it.
struct Ring {
    uring: IoUring,
    /// Where this ring is in the engine's list: the lane of its operations' [`OpId`]s.
    index: u32,
    /// Pushing entries and entering the kernel: the owning thread, and a poll that resubmits the
    /// rest of a short read or hands over entries the kernel refused or had no room for.
    #[allow(clippy::disallowed_types, reason = "HOT-FIX(I4)")]
    sq: Mutex<Queue>,
    /// Entries queued while another thread held `sq` (a poll chaining a read's pieces, each copied
    /// inside its enter: a lane's small read waited 1.4 ms for the lock). Lock-free: any thread
    /// appends; the `sq` holder drains it (the consumer lives in [`Queue`]) before it lets go, and
    /// looks again after it does (see [`Ring::flush_waiting`]).
    incoming: Producer<Item, Multi>,
    /// Set after a push to `incoming`, once the entry is published; the `sq` holder clears it
    /// before it drains, and looks again after it lets go.
    // HOT-FIX(S6a): the flag hands entries from a second submitter to the `sq` holder.
    waiting: AtomicBool,
    /// Reaping: one poll at a time (the lattice's guarantee; the lock makes it a fact).
    #[allow(clippy::disallowed_types, reason = "HOT-FIX(I5)")]
    cq: Mutex<Consumer<Ended, Single>>,
    #[allow(clippy::disallowed_types, reason = "HOT-FIX(I6)")]
    ops: Mutex<Ops>,
    /// Entries handed to the submission queue and not yet reaped; at most the completion queue's
    /// size, so the completion queue never overflows.
    in_kernel: AtomicUsize,
    capacity: usize,
    /// Completion-queue entries only async cancels may use: operations stop at `capacity -
    /// reserve`, so a cancel always finds room once the cancels before it completed (they need no
    /// peer to speak), and never waits for an operation. A cancel whose operation's piece still
    /// waits in the backlog ends that operation there, and takes no room.
    reserve: usize,
}

/// What goes to the submission queue as one: an entry, or an entry and the timeout linked to it (the
/// kernel must get the two together, so they are never split between two flushes).
enum Item {
    One(squeue::Entry),
    /// The entry, its linked timeout, and the limit's point on the monotonic clock (a piece that
    /// still waits when that passes never goes in).
    Linked(Box<[squeue::Entry; 2]>, u64),
    /// An async cancel of the operation with this user data. It is built when it goes in, may use
    /// the completion queue's reserve (see [`Ring::reserve`]), and never goes in before an item of
    /// its operation that still waits (the kernel must already hold what it cancels).
    Cancel(u64),
}

impl Item {
    /// Submission-queue entries (and so completions) the item takes.
    fn entries(&self) -> usize {
        match self {
            Self::One(_) | Self::Cancel(_) => 1,
            Self::Linked(..) => 2,
        }
    }

    /// The user data of the operation a piece belongs to.
    fn operation(&self) -> Option<u64> {
        match self {
            Self::One(entry) => Some(entry.get_user_data()),
            Self::Linked(pair, _) => Some(pair[0].get_user_data()),
            Self::Cancel(_) => None,
        }
    }
}

struct Queue {
    /// The drain side of [`Ring::incoming`]: only the `sq` holder has it.
    incoming: Consumer<Item, Single>,
    /// Entries waiting for room in the kernel's queues, the oldest first.
    backlog: VecDeque<Item>,
    /// Entries are in the submission queue that the kernel has not taken (an enter was refused).
    unsubmitted: bool,
    /// Operations that ended while their next piece waited in the backlog (a cancel, or their
    /// limit): nothing of them is in the kernel. The poll concludes them.
    ended: Producer<Ended, Multi>,
    /// `Cancel` items in the backlog: the flush stops walking the backlog when none is left.
    cancels: usize,
    /// `Linked` items in the backlog, and the earliest limit among them or before it (a lower
    /// bound: it is not raised when the earliest leaves).
    limited: usize,
    earliest_ns: u64,
}

/// An operation that ended before its next piece reached the kernel, and why.
struct Ended {
    user_data: u64,
    why: End,
}

#[derive(Clone, Copy)]
enum End {
    Cancelled,
    LimitPassed,
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

/// The empty path of a `statx` of a descriptor.
static EMPTY_PATH: &std::ffi::CStr = c"";

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
    /// `statx(2)` of the descriptor itself (an empty path with `AT_EMPTY_PATH`). The result block
    /// is boxed so that it stays where the kernel writes it.
    Statx {
        fd: RawFd,
        out: Box<libc::statx>,
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
    /// The time limit, if the operation has one.
    limit: Option<Box<Limit>>,
    /// [`Engine::cancel`] asked for the operation to end early.
    cancelled: bool,
}

/// An operation's time limit as an absolute point of the monotonic clock: the kernel's linked
/// timeout reads `spec` (so it stays in its own block) at every submission of a piece, and the
/// whole operation, not each piece, is bound by it.
struct Limit {
    spec: types::Timespec,
    at_ns: u64,
}

/// The monotonic clock in nanoseconds, the clock a linked timeout counts on.
fn monotonic_ns() -> u64 {
    let mut now = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `now` is a valid `timespec` for the call to write.
    unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &raw mut now) };
    (now.tv_sec as u64)
        .saturating_mul(1_000_000_000)
        .saturating_add(now.tv_nsec as u64)
}

impl Limit {
    /// A limit that ends `limit` from now.
    fn after(limit: Duration) -> Box<Self> {
        let at_ns =
            monotonic_ns().saturating_add(u64::try_from(limit.as_nanos()).unwrap_or(u64::MAX));
        // Seconds beyond what a `Timespec` holds are far past any process's life.
        let spec = types::Timespec::new()
            .sec((at_ns / 1_000_000_000).min(i64::MAX as u64))
            .nsec((at_ns % 1_000_000_000) as u32);
        Box::new(Self { spec, at_ns })
    }

    fn passed(&self) -> bool {
        monotonic_ns() >= self.at_ns
    }
}

impl Op {
    /// What goes to the submission queue for the operation's next piece: its entry, with the
    /// linked timeout of its limit.
    fn item(&mut self, user_data: u64, max_piece: usize) -> Item {
        let entry = self.entry(user_data, max_piece);
        match &self.limit {
            None => Item::One(entry),
            Some(limit) => {
                let timeout = opcode::LinkTimeout::new(&raw const limit.spec)
                    .flags(types::TimeoutFlags::ABS)
                    .build()
                    .user_data(IGNORED);
                Item::Linked(
                    Box::new([entry.flags(squeue::Flags::IO_LINK), timeout]),
                    limit.at_ns,
                )
            }
        }
    }

    /// The submission entry for the operation's next piece.
    fn entry(&mut self, user_data: u64, max_piece: usize) -> squeue::Entry {
        let piece = |n: usize| u32::try_from(n.min(max_piece)).unwrap_or(u32::MAX);
        let entry = match &mut self.kind {
            Kind::Read { fd, offset } => {
                let rest = &mut self.buf.bytes_mut()[self.moved..];
                // A read the page cache can serve is copied inside `io_uring_enter`, on the
                // submitting worker: about 0.2 ms per MB (a read of several MB copied in one
                // completion held a worker for a millisecond). So a read goes in pieces of at most
                // READ_PIECE, each piece's completion submitting the next (`IOSQE_ASYNC` instead
                // made the submitting worker create an io worker thread now and then: worse).
                let len = piece(rest.len().min(READ_PIECE));
                opcode::Read::new(types::Fd(*fd), rest.as_mut_ptr(), len)
                    .offset(*offset + self.moved as u64)
                    .build()
            }
            Kind::Write { fd, offset } => {
                let rest = &self.buf.bytes()[self.moved..self.buf.filled()];
                opcode::Write::new(types::Fd(*fd), rest.as_ptr(), piece(rest.len()))
                    .offset(*offset + self.moved as u64)
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
            Kind::Statx { fd, out } => opcode::Statx::new(
                types::Fd(*fd),
                EMPTY_PATH.as_ptr(),
                // The kernel writes through this pointer: it comes from a mutable borrow.
                (&raw mut **out).cast(),
            )
            .flags(libc::AT_EMPTY_PATH)
            .mask(libc::STATX_SIZE)
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

/// What reached an operation: a completion, or the end of an operation whose next piece never went
/// in.
enum Arrival {
    Completion(i32),
    Unsubmitted(End),
}

/// What a reaped completion means for its operation.
enum Reaped {
    /// It ended: its continuation's frame goes out.
    Ended(Op, io::Result<usize>),
    /// A short read or write: the rest was offered to the ring's queue (under the `ops` lock, so
    /// that a cancel made after it queues behind it); the caller flushes.
    Again,
    /// A completion of no operation (never expected).
    Stray,
}

impl Ring {
    #[allow(
        clippy::disallowed_types,
        reason = "HOT-FIX(I4), HOT-FIX(I5), HOT-FIX(I6)"
    )]
    fn new(index: u32, entries: u32) -> io::Result<Self> {
        let uring = IoUring::builder().setup_clamp().build(entries)?;
        let capacity = uring.params().cq_entries() as usize;
        // A linked pair must always fit between the operations' room and the reserve.
        let reserve = (capacity / 8).max(1);
        debug_assert!(
            capacity - reserve >= 2,
            "a ring too small for a linked pair"
        );
        let incoming: MpscQueue<Item> = MpscQueue::with_pool(INCOMING_CHUNK, 2);
        let ended: MpscQueue<Ended> = MpscQueue::with_pool(INCOMING_CHUNK, 2);
        Ok(Self {
            uring,
            index,
            sq: Mutex::new(Queue {
                incoming: incoming.consumer(),
                backlog: VecDeque::new(),
                unsubmitted: false,
                ended: ended.producer(),
                cancels: 0,
                limited: 0,
                earliest_ns: u64::MAX,
            }),
            incoming: incoming.producer(),
            waiting: AtomicBool::new(false),
            cq: Mutex::new(ended.consumer()),
            ops: Mutex::new(Ops::default()),
            in_kernel: AtomicUsize::new(0),
            capacity,
            reserve,
        })
    }

    /// Appends `item` to the entries that wait for `sq`, without handing anything to the kernel:
    /// [`flush_waiting`](Self::flush_waiting) does that. Callers that must keep two items in order
    /// (a piece and the cancel of its operation) offer both under the `ops` lock.
    fn offer(&self, item: Item) {
        // Cannot fail: the queue is unbounded. Were it ever to, the entry would be lost with its
        // operation still in `ops`: say so loudly.
        let queued = self.incoming.offer(item);
        assert!(
            queued.is_ok(),
            "the unbounded incoming queue refused an entry"
        );
        // After the publish: a holder that reads this `true` finds the entry.
        self.waiting.store(true, Ordering::SeqCst);
    }

    /// Records that the operation ended with its next piece never reaching the kernel.
    fn end_unsubmitted(queue: &Queue, user_data: u64, why: End) {
        // Cannot fail: the queue is unbounded.
        let queued = queue.ended.offer(Ended { user_data, why });
        assert!(
            queued.is_ok(),
            "the unbounded queue of ended operations refused an entry"
        );
    }

    /// Ends the operations whose next piece waits in the backlog past their limit: no piece starts
    /// after the limit. Reads the clock on each flush while a limited piece waits, and walks the
    /// backlog only once the earliest of their limits may have passed.
    fn end_passed_limits(queue: &mut Queue) {
        if queue.limited == 0 {
            return;
        }
        let now = monotonic_ns();
        if now < queue.earliest_ns {
            return;
        }
        let mut earliest = u64::MAX;
        let mut left = 0;
        let mut passed = Vec::new();
        queue.backlog.retain(|item| match item {
            Item::Linked(pair, at_ns) if *at_ns <= now => {
                passed.push(pair[0].get_user_data());
                false
            }
            Item::Linked(_, at_ns) => {
                earliest = earliest.min(*at_ns);
                left += 1;
                true
            }
            _ => true,
        });
        queue.limited = left;
        queue.earliest_ns = earliest;
        for user_data in passed {
            Self::end_unsubmitted(queue, user_data, End::LimitPassed);
        }
    }

    /// Moves backlog entries into the submission queue while the completion queue has room for
    /// them, and enters the kernel when there is anything for it.
    fn flush(&self, queue: &mut Queue, enter: &Enter) {
        let mut pushed = false;
        if !queue.backlog.is_empty() {
            Self::end_passed_limits(queue);
            // SAFETY: the `sq` lock is held, so this is the only `SubmissionQueue` of the ring.
            let mut sq = unsafe { self.uring.submission_shared() };
            // Operations go in order; once one waits for room, the ones after it wait too. A cancel
            // may pass them (it takes the reserve); one whose operation waits ahead of it ends
            // that operation right here instead, as nothing of it is in the kernel.
            let mut operations_wait = false;
            let mut at = 0;
            while let Some(item) = queue.backlog.get(at) {
                if operations_wait && queue.cancels == 0 {
                    // Nothing past here can go in.
                    break;
                }
                let in_kernel = self.in_kernel.load(Ordering::Acquire);
                let entries = item.entries();
                if let Item::Cancel(target) = item {
                    let target = *target;
                    let waiting = queue
                        .backlog
                        .iter()
                        .take(at)
                        .position(|ahead| ahead.operation() == Some(target));
                    if let Some(waiting) = waiting {
                        queue.backlog.remove(at);
                        if let Some(Item::Linked(..)) = queue.backlog.remove(waiting) {
                            queue.limited -= 1;
                        }
                        queue.cancels -= 1;
                        Self::end_unsubmitted(queue, target, End::Cancelled);
                        at -= 1;
                        continue;
                    }
                    if in_kernel + entries > self.capacity {
                        break;
                    }
                } else if operations_wait || in_kernel + entries > self.capacity - self.reserve {
                    // An item's entries go in together, and each has a completion: room for all.
                    operations_wait = true;
                    at += 1;
                    continue;
                }
                // SAFETY: every pointer in the entry is into its operation's buffer (or its boxed
                // address), which the ring's `ops` own and leave untouched until the operation is
                // reaped; a ring dropped with operations in flight leaks them (`Drop for Ring`).
                // Accept passes null address pointers; a linked timeout points at its operation's
                // `Limit`, and an async cancel at nothing.
                let pushed_item = match item {
                    // SAFETY: as above.
                    Item::One(entry) => unsafe { sq.push(entry) },
                    // SAFETY: as above.
                    Item::Linked(pair, _) => unsafe { sq.push_multiple(&pair[..]) },
                    Item::Cancel(target) => {
                        let cancel = opcode::AsyncCancel::new(*target).build().user_data(IGNORED);
                        // SAFETY: as above.
                        unsafe { sq.push(&cancel) }
                    }
                };
                if pushed_item.is_err() {
                    break;
                }
                match queue.backlog.remove(at) {
                    Some(Item::Cancel(_)) => queue.cancels -= 1,
                    Some(Item::Linked(..)) => queue.limited -= 1,
                    _ => {}
                }
                self.in_kernel.fetch_add(entries, Ordering::AcqRel);
                pushed = true;
            }
            sq.sync();
        }
        if pushed || queue.unsubmitted {
            // Set before the call: if the enter unwinds, the entries are queued and a later poll hands them over.
            queue.unsubmitted = true;
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
            // A poisoned lock is taken, not skipped: a panic under it (a refused enter's hook, a failed
            // check) must not leave the ring's entries unsubmitted for ever.
            let mut queue = match self.sq.try_lock() {
                Ok(queue) => queue,
                Err(TryLockError::Poisoned(poisoned)) => poisoned.into_inner(),
                Err(TryLockError::WouldBlock) => return,
            };
            // The swap reads the pusher's `true`, so its entry is published for the drain below.
            self.waiting.swap(false, Ordering::SeqCst);
            let Queue {
                incoming,
                backlog,
                cancels,
                limited,
                earliest_ns,
                ..
            } = &mut *queue;
            incoming.drain_all(|item| {
                match &item {
                    Item::Cancel(_) => *cancels += 1,
                    Item::Linked(_, at_ns) => {
                        *limited += 1;
                        *earliest_ns = (*earliest_ns).min(*at_ns);
                    }
                    Item::One(_) => {}
                }
                backlog.push_back(item);
            });
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

    fn reap_one(&self, user_data: u64, arrival: Arrival, max_piece: usize) -> Reaped {
        let mut ops = lock!(&self.ops);
        let Some(op) = ops.get(user_data) else {
            return Reaped::Stray;
        };
        let result = match arrival {
            Arrival::Completion(result) => result,
            Arrival::Unsubmitted(why) => {
                // As in the kernel's path: a cancelled operation ends as cancelled, whatever
                // else ended it too.
                let error = match why {
                    End::LimitPassed if !op.cancelled => LimitPassed::error(op.moved),
                    _ => Cancelled::error(op.moved),
                };
                let op = ops.remove(user_data).expect("the operation just found");
                return Reaped::Ended(op, Err(error));
            }
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
                    if op.cancelled {
                        // The cancel found the operation between two pieces: nothing in the
                        // kernel was left to cancel, so this is where it ends.
                        Err(Cancelled::error(op.moved))
                    } else if op.limit.as_ref().is_some_and(|l| l.passed()) {
                        // The whole operation is bound by the limit: no piece starts after it.
                        Err(LimitPassed::error(op.moved))
                    } else {
                        let item = op.item(user_data, max_piece);
                        self.offer(item);
                        return Reaped::Again;
                    }
                } else {
                    Ok(op.moved)
                }
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
            // Ended by the cancel (`ECANCELED`), or interrupted while it was being cancelled.
            _ if (result == -libc::ECANCELED || result == -libc::EINTR) && op.cancelled => {
                Err(Cancelled::error(op.moved))
            }
            // The linked timeout cancelled it: the clock it counted on has reached the limit.
            _ if (result == -libc::ECANCELED || result == -libc::EINTR)
                && op.limit.as_ref().is_some_and(|l| l.passed()) =>
            {
                Err(LimitPassed::error(op.moved))
            }
            _ if result < 0 => Err(io::Error::from_raw_os_error(-result)),
            Kind::Statx { ref out, .. } => {
                if out.stx_mask & libc::STATX_SIZE == 0 {
                    Err(io::Error::other("the file system gave no size"))
                } else {
                    usize::try_from(out.stx_size)
                        .map_err(|_| io::Error::from_raw_os_error(libc::EOVERFLOW))
                }
            }
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
        let ops = std::mem::take(&mut *lock!(&self.ops));
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

pub(super) fn c_path(path: &std::path::Path) -> io::Result<CString> {
    use std::os::unix::ffi::OsStrExt;
    CString::new(path.as_os_str().as_bytes())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "a path with a NUL byte"))
}

pub(super) fn socket_address(
    to: std::net::SocketAddr,
) -> (Box<libc::sockaddr_storage>, libc::socklen_t) {
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
    /// The engine on the native `io_uring_enter`; the error is the first ring's setup refusal.
    pub(super) fn native() -> io::Result<Self> {
        Self::probed(
            Box::new(|submitter: &Submitter<'_>| submitter.submit()),
            MAX_PIECE,
        )
    }

    /// As [`with`](Self::with), after proving that the kernel accepts a submission: some seccomp profiles allow
    /// `io_uring_setup` and refuse `io_uring_enter`, and every operation would then wait for ever. A no-op is
    /// submitted through `enter` on a ring of its own, which returns at once (it does not wait for the
    /// completion); the error is whatever `enter` refused with.
    pub(super) fn probed(enter: Box<Enter>, max_piece: usize) -> io::Result<Self> {
        let ring = Ring::new(0, 2)?;
        {
            // SAFETY: the ring is this function's own: this is its only `SubmissionQueue`.
            let mut sq = unsafe { ring.uring.submission_shared() };
            let nop = opcode::Nop::new().build().user_data(IGNORED);
            // SAFETY: a no-op entry reads no memory.
            if unsafe { sq.push(&nop) }.is_err() {
                return Err(io::Error::other("the probe ring has no room"));
            }
        }
        enter(&ring.uring.submitter())?;
        drop(ring);
        Self::with(enter, max_piece)
    }

    /// An engine with `enter` in place of `io_uring_enter` and pieces of at most `max_piece` bytes
    /// (tests: the stand-in backend).
    fn with(enter: Box<Enter>, max_piece: usize) -> io::Result<Self> {
        Self::sized(enter, max_piece, RING_ENTRIES)
    }

    /// As [`with`](Self::with), with rings of `ring_entries` submission-queue entries.
    #[allow(clippy::disallowed_types, reason = "HOT-FIX(I3)")]
    fn sized(enter: Box<Enter>, max_piece: usize, ring_entries: u32) -> io::Result<Self> {
        // A ring proves the kernel allows io_uring before anything relies on it.
        drop(Ring::new(0, ring_entries)?);
        Ok(Self {
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            rings: Mutex::new(Vec::new()),
            enter,
            ring_entries,
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
            let mut rings = lock!(&self.rings);
            if rings.len() >= MAX_RINGS {
                return Err(io::Error::other("too many threads submit to this AsyncIo"));
            }
            let ring = Arc::new(Ring::new(rings.len() as u32, self.ring_entries)?);
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
            Request::FileLen { file } => Kind::Statx {
                fd: file as RawFd,
                // SAFETY: `statx` is plain integers: all zeros is a valid value.
                out: Box::new(unsafe { std::mem::zeroed() }),
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
pub(super) fn open_flags(spec: OpenSpec) -> i32 {
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
    fn submit(
        &self,
        request: Request,
        buf: Buf,
        then: Then,
        limit: Option<Duration>,
        taken: &Cell<bool>,
    ) -> io::Result<OpId> {
        let kind = Self::kind(request)?;
        let ring = self.ring()?;
        let user_data = {
            let mut ops = lock!(&ring.ops);
            let user_data = ops.insert(Op {
                kind,
                buf,
                moved: 0,
                then,
                limit: limit.map(Limit::after),
                cancelled: false,
            });
            let item = ops
                .get(user_data)
                .expect("just inserted")
                .item(user_data, self.max_piece);
            // Offered before the lock lets go: a cancel finds the operation only after its first
            // piece is queued.
            ring.offer(item);
            user_data
        };
        // The ring has the operation: from here on a panic is not a refusal.
        taken.set(true);
        ring.flush_waiting(&*self.enter);
        Ok(OpId::new(self.id as u64, ring.index, user_data))
    }

    fn cancel(&self, id: OpId) {
        if id.engine() != self.id as u64 {
            return;
        }
        let Some(ring) = lock!(&self.rings).get(id.lane() as usize).cloned() else {
            return;
        };
        {
            let mut ops = lock!(&ring.ops);
            // An operation that ended (or whose slot a later one took: the generation differs) is
            // not there, and nothing is submitted. A second cancel adds nothing to the first.
            let Some(op) = ops.get(id.token()) else {
                return;
            };
            if op.cancelled {
                return;
            }
            op.cancelled = true;
            // Offered under the lock that `reap_one` holds when it queues the next piece: the
            // cancel is behind that piece, whichever of the two came first.
            ring.offer(Item::Cancel(id.token()));
        }
        ring.flush_waiting(&*self.enter);
    }

    fn poll(&self, room: &mut dyn FnMut() -> bool, offer: &mut dyn FnMut(Box<dyn Frame>) -> bool) {
        let rings: Vec<Arc<Ring>> = lock!(&self.rings).clone();
        for ring in &rings {
            ring.flush_waiting(&*self.enter);
            // A poisoned lock is taken, not skipped: it guards a queue of ended operations that a
            // panic cannot leave half-changed, and skipping it would never reap this ring again.
            let mut reaping = match ring.cq.try_lock() {
                Ok(reaping) => reaping,
                Err(TryLockError::Poisoned(poisoned)) => poisoned.into_inner(),
                Err(TryLockError::WouldBlock) => continue,
            };
            // Operations that ended in the backlog, a cancel or a limit: no completion will say so.
            while room() {
                let Some(Ended { user_data, why }) = reaping.poll() else {
                    break;
                };
                match ring.reap_one(user_data, Arrival::Unsubmitted(why), self.max_piece) {
                    Reaped::Ended(op, result) => {
                        if !offer(conclude(op, result)) {
                            ring.flush_waiting(&*self.enter);
                            return;
                        }
                    }
                    Reaped::Again | Reaped::Stray => {
                        note_stray("an ended operation", user_data);
                    }
                }
            }
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
                if user_data == IGNORED {
                    continue;
                }
                match ring.reap_one(user_data, Arrival::Completion(result), self.max_piece) {
                    Reaped::Ended(op, result) => {
                        if !offer(conclude(op, result)) {
                            ring.flush_waiting(&*self.enter);
                            return;
                        }
                    }
                    Reaped::Again => ring.flush_waiting(&*self.enter),
                    Reaped::Stray => {
                        note_stray("a completion", user_data);
                    }
                }
            }
            // Room freed in the completion queue may let the backlog in.
            ring.flush_waiting(&*self.enter);
        }
    }
}

#[cfg(test)]
use super::Platform;

impl AsyncIo {
    /// An `AsyncIo` on io_uring with `enter` in place of `io_uring_enter` and pieces of at most
    /// `max_piece` bytes (tests: the stand-in backend).
    #[cfg(test)]
    fn open_with(enter: Box<Enter>, max_piece: usize) -> io::Result<Arc<Self>> {
        Ok(Self::from_engine(Platform::Uring(Uring::with(
            enter, max_piece,
        )?)))
    }

    /// As [`open_with`](Self::open_with), with rings of `ring_entries` submission-queue entries
    /// (tests: a ring small enough to fill).
    #[cfg(test)]
    fn open_sized(enter: Box<Enter>, max_piece: usize, ring_entries: u32) -> io::Result<Arc<Self>> {
        Ok(Self::from_engine(Platform::Uring(Uring::sized(
            enter,
            max_piece,
            ring_entries,
        )?)))
    }
}

/// Tests of the io_uring engine: reads, writes, sockets and datagrams.
/// A stand-in for `io_uring_enter` plays the kernel's submission.
#[cfg(test)]
#[allow(clippy::disallowed_methods)] // Test fixtures write files and open sockets synchronously, outside any lattice.
mod tests {
    use std::io::{Read, Write};
    use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
    use std::path::PathBuf;
    use std::time::{Duration, Instant};

    use euhedral_core::frames::{FrameHeader, RunnableFrame};
    use euhedral_core::generics::{LatticeReceiver, LatticeSource};

    use super::super::{
        AsyncIoOptions, Continuation, IoBuf, IoDone, IoDoneOwned, IoMemory, IoThen, OpOptions,
    };
    use super::*;

    /// The io_uring engine of an `AsyncIo` opened with `open_with`.
    fn uring_of(io: &AsyncIo) -> &Uring {
        match &io.engine {
            Platform::Uring(uring) => uring,
            Platform::Positional(_) => unreachable!("opened on io_uring"),
        }
    }

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
            io.reap();
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

    /// A read's completion is a frame emitted when the sink is polled and its bytes have arrived.
    fn a_reads_completion_is_a_frame_emitted_when_polled_after_its_bytes_arrived(io: Arc<AsyncIo>) {
        let data = content(3 * 4096 + 123, 3);
        let (_guard, file) = temp_file("data", &data);
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
                    lock!(&returned).push((done.buf.expect("memory back"), done.result.unwrap()));
                    Box::new(RunnableFrame::new(1, Arc::new(|| {})))
                }),
            )
            .unwrap();
        }
        let deadline = Instant::now() + Duration::from_secs(10);
        while io.pending() > 0 {
            assert!(Instant::now() < deadline, "the reads did not complete");
            io.reap();
            io.pull(&mut |frame| drop(frame), &mut |_| false, u64::MAX);
        }
        let returned = std::mem::take(&mut *lock!(&returned));
        assert_eq!(returned.len(), 2);
        assert!(returned.iter().all(|(_, n)| *n == 4096));
        drop(returned);
        let arena = arena.bytes();
        assert_eq!(arena[..4096], data[2 * 4096..]);
        assert!(arena[4096..4096 + 100].iter().all(|&b| b == 0xEE));
        assert_eq!(arena[4096 + 100..], data[..4096]);
    }

    /// Reads from many threads each complete once: 8 threads submit on rings of their own;
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
                    while lock!(&results).len() < 64 {
                        assert!(Instant::now() < deadline, "the reads did not complete");
                        let _one_poller = lock!(&poller);
                        io.reap();
                        io.pull(
                            &mut |frame| lock!(&results).push(frame),
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
        let mut results: Vec<(u64, Box<Done>)> = std::mem::take(&mut *lock!(&results))
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
            lock!(&uring_of(&io).rings).len(),
            8,
            "a ring per submitting thread"
        );
        assert_eq!(io.pending(), 0);
    }

    /// A read past the end of the file fails.
    fn a_read_past_the_end_of_the_file_fails(io: Arc<AsyncIo>) {
        let (_guard, file) = temp_file("short", &[0; 100]);
        io.read(file.as_raw_fd(), 8192, buf(4096), then(9)).unwrap();
        let done = poll(&io, 1).pop().unwrap().1.done.unwrap();
        assert_eq!(
            done.result.unwrap_err().kind(),
            io::ErrorKind::UnexpectedEof
        );
    }

    /// The file ends inside the read: it completes with the bytes there were (a read of "up to" its
    /// buffer).
    fn a_read_the_file_ends_inside_completes_with_the_bytes_there_were(io: Arc<AsyncIo>) {
        let data = content(4096 + 100, 13);
        let (_guard, file) = temp_file("tail", &data);
        io.read(file.as_raw_fd(), 0, buf(2 * 4096), then(1))
            .unwrap();
        let done = poll(&io, 1).pop().unwrap().1.done.unwrap();
        assert_eq!(done.result.unwrap(), data.len());
        assert_eq!(&done.buf.bytes[..done.buf.filled], &data[..]);
    }

    /// A refused submission leaves no entry that reads or completes later: a refused enter leaves the operation queued (the caller has handed its buffer over), and
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
            io.reap();
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

    /// A short read: the rest is resubmitted by the poll that reaped the short part,
    /// and the read completes once, with all its bytes.
    fn short_reads_and_writes_are_resubmitted_until_every_byte_moved(
        short: fn(usize) -> Arc<AsyncIo>,
    ) {
        let data = content(10 * 4096 + 7, 31);
        let (guard, file) = temp_file("pieces", &data);
        let io = short(4096);
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

    /// An `AsyncIo` on the positional engine whose pieces are at most `max_piece` bytes.
    fn short_positional(max_piece: usize) -> Arc<AsyncIo> {
        AsyncIo::from_engine(Platform::Positional(
            crate::positional::Positional::with_piece(max_piece),
        ))
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
    fn a_short_send_is_resubmitted_until_every_byte_is_in_the_kernel(
        short: fn(usize) -> Arc<AsyncIo>,
    ) {
        let io = short(3000);
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
    fn a_send_stopped_part_way_carries_its_error(short: fn(usize) -> Arc<AsyncIo>) {
        let io = short(3000);
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
    fn operations_beyond_the_queues_room_wait_and_all_complete(io: Arc<AsyncIo>) {
        let data = content(4096, 17);
        let (_guard, file) = temp_file("backlog", &data);
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

    fn a_poll_hands_on_no_more_than_its_demand(io: Arc<AsyncIo>) {
        let (_guard, file) = temp_file("demand", &content(4096, 19));
        for i in 0..4 {
            io.read(file.as_raw_fd(), 0, buf(16), then(i)).unwrap();
        }
        let deadline = Instant::now() + Duration::from_secs(10);
        let mut seen = 0;
        while seen < 4 {
            assert!(Instant::now() < deadline);
            let mut batch = 0;
            io.reap();
            io.pull(&mut |_| batch += 1, &mut |_| false, 1);
            assert!(batch <= 1);
            seen += batch;
        }
        assert_eq!(io.pending(), 0);
    }

    /// The lattice's direct pulls stop before ordered frames: a held-back completion goes out first
    /// on a later poll.
    fn a_pull_stops_before_the_completion_its_stop_matches_and_hands_it_on_later_in_order(
        io: Arc<AsyncIo>,
    ) {
        let (_guard, file) = temp_file("stop", &content(64, 43));
        io.read(file.as_raw_fd(), 0, buf(8), then(1)).unwrap();
        let deadline = Instant::now() + Duration::from_secs(10);
        while io.front.out.enter().held.is_empty() {
            assert!(Instant::now() < deadline, "the read did not complete");
            io.reap();
            let pulled = io.pull(&mut |_| panic!("held back"), &mut |_| true, u64::MAX);
            assert_eq!(pulled, 0);
        }
        assert_eq!(io.pending(), 1, "a held-back completion is still pending");
        io.read(file.as_raw_fd(), 8, buf(8), then(2)).unwrap();
        let tags: Vec<u64> = poll(&io, 2).into_iter().map(|(tag, _)| tag).collect();
        assert_eq!(tags, [1, 2], "the held-back completion goes first");
        assert_eq!(io.pending(), 0);
    }

    fn a_write_completes_when_all_its_bytes_are_in_the_file(io: Arc<AsyncIo>) {
        let (guard, file) = temp_file("write", &[]);
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

    fn empty_reads_and_writes_and_submissions_after_complete_are_refused_at_once(io: Arc<AsyncIo>) {
        let (_guard, file) = temp_file("refuse", &[1; 16]);
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

    fn accept_send_and_receive_on_a_loopback_connection(io: Arc<AsyncIo>) {
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

    fn a_connect_completes_once_connected_or_refused(io: Arc<AsyncIo>) {
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

    fn a_send_to_a_closed_peer_fails_without_a_signal(io: Arc<AsyncIo>) {
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

    fn a_datagram_goes_to_its_address(io: Arc<AsyncIo>) {
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

    fn an_atomic_save_opens_a_temporary_file_writes_it_and_renames_it_over_the_old_one(
        io: Arc<AsyncIo>,
    ) {
        let (target, _old) = temp_file("config", b"old settings");
        let temp = TempFile(target.0.with_extension("tmp"));
        type Opened = Arc<Mutex<Option<io::Result<OwnedFd>>>>;
        let opened: Opened = Arc::new(Mutex::new(None));
        let slot = opened.clone();
        io.open_file(
            &temp.0,
            libc::O_WRONLY | libc::O_CREAT | libc::O_TRUNC,
            0o644,
            Box::new(move |fd| {
                *lock!(&slot) = Some(fd);
                Box::new(Done {
                    header: FrameHeader::new(1),
                    done: None,
                    accepted: None,
                })
            }),
        )
        .unwrap();
        poll(&io, 1);
        let fd = lock!(&opened).take().unwrap().unwrap();
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
                *lock!(&slot) = Some(result);
                Box::new(Done {
                    header: FrameHeader::new(3),
                    done: None,
                    accepted: None,
                })
            }),
        )
        .unwrap();
        poll(&io, 1);
        lock!(&renamed).take().unwrap().unwrap();
        assert_eq!(std::fs::read(&target.0).unwrap(), settings);
        assert!(!temp.0.exists(), "the temporary file was renamed");
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(
            std::fs::metadata(&target.0).unwrap().permissions().mode() & 0o777,
            0o644 & !umask()
        );
    }

    /// What `file_len` of `fd` ended with, polled to the end.
    fn file_len_of(io: &AsyncIo, fd: RawFd) -> io::Result<u64> {
        let slot = Arc::new(Mutex::new(None));
        let put = slot.clone();
        io.file_len(
            fd,
            Box::new(move |result| {
                *lock!(&put) = Some(result);
                Box::new(Done {
                    header: FrameHeader::new(1),
                    done: None,
                    accepted: None,
                })
            }),
        )
        .unwrap();
        poll(io, 1);
        lock!(&slot).take().unwrap()
    }

    fn file_len_gives_the_size_of_an_open_file_and_the_kernels_error_for_a_bad_descriptor(
        io: Arc<AsyncIo>,
    ) {
        let (_path, file) = temp_file("sized", &[7u8; 12345]);
        assert_eq!(file_len_of(&io, file.as_raw_fd()).unwrap(), 12345);
        // It follows the file as it grows.
        file.set_len(70_000).unwrap();
        assert_eq!(file_len_of(&io, file.as_raw_fd()).unwrap(), 70_000);
        let (_empty_path, empty) = temp_file("empty", b"");
        assert_eq!(file_len_of(&io, empty.as_raw_fd()).unwrap(), 0);
        let error = file_len_of(&io, -1).unwrap_err();
        assert_eq!(error.raw_os_error(), Some(libc::EBADF));
    }

    fn umask() -> u32 {
        // SAFETY: umask only swaps the process's mask; it is set straight back.
        let mask = unsafe { libc::umask(0o022) };
        // SAFETY: as above.
        unsafe { libc::umask(mask) };
        mask as u32
    }

    fn opening_or_renaming_what_is_not_there_fails_with_the_kernels_error(io: Arc<AsyncIo>) {
        let missing = std::path::Path::new("/nonexistent-euhedral-dir/file");
        let errors = Arc::new(Mutex::new(Vec::new()));
        let (a, b) = (errors.clone(), errors.clone());
        io.open_file(
            missing,
            libc::O_RDONLY,
            0,
            Box::new(move |r| {
                lock!(&a).push(r.unwrap_err().raw_os_error());
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
                lock!(&b).push(r.unwrap_err().raw_os_error());
                Box::new(Done {
                    header: FrameHeader::new(2),
                    done: None,
                    accepted: None,
                })
            }),
        )
        .unwrap();
        poll(&io, 2);
        assert_eq!(*lock!(&errors), [Some(libc::ENOENT), Some(libc::ENOENT)]);
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
    fn requests_push_completions_downstream(io: Arc<AsyncIo>) {
        #[derive(Default)]
        struct Recorder(Mutex<Vec<u64>>, AtomicBool);
        impl euhedral_core::generics::LatticeTerminal for Recorder {
            fn add_upstream(&self, _: Arc<dyn LatticeSource>) {}
        }
        impl LatticeReceiver for Recorder {
            fn push(&self, mut frame: Box<dyn Frame>) {
                lock!(&self.0).push(frame.header().id_hash());
                // As a worker does: the reap frame reaps, and what it reaped is pushed within the demand.
                frame.execute().unwrap();
                frame.do_finally();
            }
            fn on_complete(&self) {
                self.1.store(true, Ordering::Release);
            }
            fn on_error(&self, _: euhedral_core::frames::Failure) {}
        }
        let (_guard, file) = temp_file("request", &content(64, 29));
        let recorder = Arc::new(Recorder::default());
        io.clone().add_downstream(recorder.clone());
        io.read(file.as_raw_fd(), 0, buf(8), then(5)).unwrap();
        let deadline = Instant::now() + Duration::from_secs(10);
        while !lock!(&recorder.0).contains(&5) {
            assert!(Instant::now() < deadline);
            io.request(1);
        }
        let pushed = lock!(&recorder.0).clone();
        assert_eq!(
            pushed.iter().filter(|id| **id == crate::KEY_ID).count(),
            pushed.len() - 1,
            "the reap frame, as many times as it was pushed, and the one completion"
        );
        assert_eq!(pushed.iter().filter(|id| **id == 5).count(), 1);
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
                let ring = Arc::new(Ring::new(0, RING_ENTRIES).unwrap());
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
                                ring.offer(Item::One(entry));
                                ring.flush_waiting(&*slow);
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

    /// What an engine's polls found, until `count` operations ended (10 s at most).
    fn engine_polls(engine: &Uring, count: usize) -> Vec<Done> {
        let mut out = Vec::new();
        let deadline = Instant::now() + Duration::from_secs(10);
        while out.len() < count {
            assert!(Instant::now() < deadline, "the operations did not end");
            engine.poll(&mut || true, &mut |frame| {
                out.push(
                    *(frame as Box<dyn std::any::Any>)
                        .downcast::<Done>()
                        .expect("a Done frame"),
                );
                true
            });
        }
        out
    }

    /// A cancel made after a poll reaped a piece and before the next piece went in must not be
    /// lost: both are offered in one order, under the lock the reaping holds, so the kernel gets
    /// the piece first and the cancel finds it. Driven by hand, the interleaving being the point.
    #[test]
    fn a_cancel_between_two_pieces_queues_behind_the_next_piece_and_ends_the_send() {
        let engine = Uring::with(
            Box::new(|submitter: &Submitter<'_>| submitter.submit()),
            4096,
        )
        .unwrap();
        let (ours, _theirs) = socket_pair(4096);
        let total = 8 * 4096;
        let id = engine
            .submit(
                Request::Send {
                    sock: crate::raw_sock(ours.as_raw_fd()),
                },
                IoBuf {
                    bytes: vec![7; total],
                    filled: total,
                }
                .into(),
                Then::Io(then(1)),
                None,
                &Cell::new(false),
            )
            .unwrap();
        let ring = engine.ring().unwrap();
        // The first piece's completion, taken by hand: this is the poll that is about to reap it.
        let deadline = Instant::now() + Duration::from_secs(10);
        let (user_data, result) = loop {
            let next = {
                // SAFETY: the test is the only reader of the ring's completion queue.
                let mut cq = unsafe { ring.uring.completion_shared() };
                cq.next().map(|cqe| (cqe.user_data(), cqe.result()))
            };
            if let Some(found) = next {
                break found;
            }
            assert!(
                Instant::now() < deadline,
                "the first piece did not complete"
            );
        };
        ring.in_kernel.fetch_sub(1, Ordering::AcqRel);
        let first = usize::try_from(result).expect("the first piece moved bytes");
        assert!(first < total);
        // No room is left in the socket: the next piece waits for the peer, which never reads.
        let fill = [0_u8; 1];
        loop {
            // SAFETY: a non-blocking send of one byte from a live buffer on a socket we own.
            let sent = unsafe {
                libc::send(
                    ours.as_raw_fd(),
                    fill.as_ptr().cast(),
                    1,
                    libc::MSG_DONTWAIT,
                )
            };
            if sent < 0 {
                break;
            }
        }
        assert!(matches!(
            ring.reap_one(user_data, Arrival::Completion(result), engine.max_piece),
            Reaped::Again
        ));
        // The cancel arrives here: after the piece was queued, before the poll handed it over.
        engine.cancel(id);
        let done = engine_polls(&engine, 1).pop().unwrap().done.unwrap();
        let error = done.result.unwrap_err();
        assert_eq!(Cancelled::moved_of(&error), Some(first));
        assert_eq!(done.buf.bytes.len(), total);
        assert_eq!(ring.in_kernel.load(Ordering::Acquire), 0);
    }

    /// Cancels have a reserve in the completion queue: with every completion entry an operation may
    /// use taken by operations that wait on silent peers, a cancel still goes in, and ends its
    /// operation without any peer speaking.
    #[test]
    fn a_cancel_does_not_wait_for_room_in_a_full_completion_queue() {
        let io = AsyncIo::open_sized(
            Box::new(|submitter: &Submitter<'_>| submitter.submit()),
            MAX_PIECE,
            4,
        )
        .unwrap();
        // The ring's completion queue has 8 entries: more silent receives than it holds.
        let pairs: Vec<_> = (0..10)
            .map(|_| std::os::unix::net::UnixStream::pair().unwrap())
            .collect();
        let ids: Vec<_> = pairs
            .iter()
            .enumerate()
            .map(|(i, (ours, _))| {
                io.recv_with(
                    ours.as_raw_fd(),
                    buf(16),
                    crate::OpOptions::default(),
                    then(i as u64),
                )
                .unwrap()
            })
            .collect();
        for _ in 0..200 {
            io.reap();
        }
        assert_eq!(io.pending(), 10);
        // Seven operations are in the kernel (the eighth entry is the reserve): this one is among
        // them, and its cancel goes in at once.
        io.cancel(ids[0]);
        let done = poll(&io, 1);
        for (_, frame) in done {
            let error = frame.done.unwrap().result.unwrap_err();
            assert!(Cancelled::is(&error));
        }
        // The rest still waits for its peers, and the cancelled operation's room went to an
        // operation that had waited for it: cancel them all and every one ends once.
        for id in &ids[1..] {
            io.cancel(*id);
        }
        let rest = poll(&io, 9);
        assert!(rest.iter().all(|(_, frame)| {
            frame
                .done
                .as_ref()
                .unwrap()
                .result
                .as_ref()
                .is_err_and(Cancelled::is)
        }));
        assert_eq!(io.pending(), 0);
    }

    /// A ring with every completion entry an operation may use taken by silent receives, and
    /// `waiting` more behind them in the backlog (the small ring takes seven in the kernel).
    fn full_ring(
        waiting: usize,
        last_limit: Option<Duration>,
    ) -> (
        std::sync::Arc<AsyncIo>,
        Vec<OpId>,
        Vec<(
            std::os::unix::net::UnixStream,
            std::os::unix::net::UnixStream,
        )>,
    ) {
        let io = AsyncIo::open_sized(
            Box::new(|submitter: &Submitter<'_>| submitter.submit()),
            MAX_PIECE,
            4,
        )
        .unwrap();
        let total = 7 + waiting;
        let pairs: Vec<_> = (0..total)
            .map(|_| std::os::unix::net::UnixStream::pair().unwrap())
            .collect();
        let ids: Vec<_> = pairs
            .iter()
            .enumerate()
            .map(|(i, (ours, _))| {
                let options = crate::OpOptions {
                    limit: if i + 1 == total { last_limit } else { None },
                };
                io.recv_with(ours.as_raw_fd(), buf(16), options, then(i as u64))
                    .unwrap()
            })
            .collect();
        for _ in 0..50 {
            io.reap();
        }
        assert_eq!(io.pending(), total);
        (io, ids, pairs)
    }

    /// An operation still waiting for room in the kernel's queues ends on its cancel, with no other
    /// operation completing and no peer speaking.
    #[test]
    fn a_cancel_ends_an_operation_that_waits_in_the_backlog() {
        let (io, ids, _pairs) = full_ring(3, None);
        io.cancel(ids[9]);
        let done = poll(&io, 1);
        assert_eq!(done[0].0, 9);
        let error = done[0]
            .1
            .done
            .as_ref()
            .unwrap()
            .result
            .as_ref()
            .unwrap_err();
        assert!(Cancelled::is(error));
        assert_eq!(Cancelled::moved_of(error), Some(0));
        assert_eq!(io.pending(), 9);
        for id in &ids[..9] {
            io.cancel(*id);
        }
        assert_eq!(poll(&io, 9).len(), 9);
        assert_eq!(io.pending(), 0);
    }

    /// The limit of an operation that waits in the backlog runs from its submission: at the limit
    /// it ends with [`LimitPassed`], whatever else is (not) completing.
    #[test]
    fn a_limit_ends_an_operation_that_waits_in_the_backlog() {
        let started = Instant::now();
        let (io, ids, _pairs) = full_ring(1, Some(Duration::from_millis(50)));
        let done = poll(&io, 1);
        assert!(started.elapsed() >= Duration::from_millis(50));
        assert_eq!(done[0].0, 7);
        let error = done[0]
            .1
            .done
            .as_ref()
            .unwrap()
            .result
            .as_ref()
            .unwrap_err();
        assert!(crate::LimitPassed::is(error));
        assert_eq!(io.pending(), 7);
        for id in &ids[..7] {
            io.cancel(*id);
        }
        assert_eq!(poll(&io, 7).len(), 7);
        assert_eq!(io.pending(), 0);
    }

    /// An operation cancelled and past its limit while it waits ends as cancelled, as one in the
    /// kernel does.
    #[test]
    fn a_cancelled_operation_past_its_limit_in_the_backlog_ends_as_cancelled() {
        let started = Instant::now();
        let (io, ids, _pairs) = full_ring(1, Some(Duration::from_millis(50)));
        // The limit passes before the cancel is flushed (the clock, not a poll, is waited for).
        while started.elapsed() < Duration::from_millis(60) {
            std::hint::spin_loop();
        }
        io.cancel(ids[7]);
        let done = poll(&io, 1);
        assert_eq!(done[0].0, 7);
        let error = done[0]
            .1
            .done
            .as_ref()
            .unwrap()
            .result
            .as_ref()
            .unwrap_err();
        assert!(Cancelled::is(error));
        for id in &ids[..7] {
            io.cancel(*id);
        }
        assert_eq!(poll(&io, 7).len(), 7);
    }

    /// A limited send is bound as a whole: a piece that finishes after the limit is the last.
    #[test]
    fn no_piece_of_a_send_starts_after_its_limit() {
        let io = short_pieces(1024);
        let (ours, mut theirs) = socket_pair(1 << 20);
        let total = 64 * 1024;
        io.send_with(
            ours.as_raw_fd(),
            IoBuf {
                bytes: vec![1; total],
                filled: total,
            },
            crate::OpOptions {
                limit: Some(Duration::ZERO),
            },
            then(1),
        )
        .unwrap();
        let done = poll(&io, 1).pop().unwrap().1.done.unwrap();
        let error = done.result.unwrap_err();
        let moved = crate::LimitPassed::moved_of(&error).expect("the limit's error");
        assert!(
            moved < total,
            "the send stopped at its limit, not at its end"
        );
        // Every byte that went out is in the peer's buffer already.
        theirs.set_nonblocking(true).unwrap();
        let mut sink = vec![0; total];
        let mut got = 0;
        while let Ok(n) = theirs.read(&mut sink[got..]) {
            if n == 0 {
                break;
            }
            got += n;
        }
        assert_eq!(
            got, moved,
            "the bytes that arrived are those the error counts"
        );
    }

    /// Tests of what only the positional engine does: how it ends operations early, and what it owns.
    fn positional() -> Arc<AsyncIo> {
        open_on(crate::IoBackend::Positional)
    }

    fn only_result(io: &AsyncIo) -> io::Error {
        poll(io, 1)
            .pop()
            .unwrap()
            .1
            .done
            .unwrap()
            .result
            .unwrap_err()
    }

    /// A read of a descriptor that is not open ends with the system's error, once, and the engine goes on.
    #[test]
    fn a_refused_positional_read_carries_the_systems_error_and_the_next_operation_runs() {
        let io = positional();
        let data = content(100, 5);
        let (_guard, file) = temp_file("refused", &data);
        io.read(-1, 0, buf(10), then(1)).unwrap();
        io.read(file.as_raw_fd(), 0, buf(100), then(2)).unwrap();
        let mut done = poll(&io, 2);
        done.sort_by_key(|(tag, _)| *tag);
        let first = done.remove(0).1.done.unwrap();
        assert_eq!(first.result.unwrap_err().raw_os_error(), Some(libc::EBADF));
        let second = done.remove(0).1.done.unwrap();
        assert_eq!(second.result.unwrap(), 100);
        assert_eq!(second.buf.bytes, data);
        assert_eq!(io.pending(), 0);
    }

    /// A cancel that arrives before the operation's first step ends it as cancelled, with nothing moved.
    #[test]
    fn a_cancel_ends_a_positional_receive_that_has_nothing_to_read() {
        let io = positional();
        let (ours, theirs) = socket_pair(4096);
        let id = io
            .recv_with(ours.as_raw_fd(), buf(64), OpOptions::default(), then(1))
            .unwrap();
        // A visit that finds the socket empty leaves the operation in flight.
        assert_eq!(io.reap(), 0);
        assert_eq!(io.pending(), 1);
        io.cancel(id);
        let error = only_result(&io);
        assert!(Cancelled::is(&error), "{error}");
        assert_eq!(Cancelled::moved_of(&error), Some(0));
        // A cancel of an operation that ended does nothing.
        io.cancel(id);
        assert_eq!(io.reap(), 0);
        drop(theirs);
    }

    /// A limit that passes ends an idle receive; the socket stays usable for the next one.
    #[test]
    fn a_limit_ends_a_positional_receive_and_the_socket_stays_usable() {
        let io = positional();
        let (ours, mut theirs) = socket_pair(4096);
        let limit = OpOptions {
            limit: Some(Duration::ZERO),
        };
        io.recv_with(ours.as_raw_fd(), buf(64), limit, then(1))
            .unwrap();
        let error = only_result(&io);
        assert!(LimitPassed::is(&error), "{error}");
        theirs.write_all(b"hello").unwrap();
        io.recv(ours.as_raw_fd(), buf(64), then(2)).unwrap();
        let got = poll(&io, 1).pop().unwrap().1.done.unwrap();
        assert_eq!(got.result.unwrap(), 5);
        assert_eq!(&got.buf.bytes[..5], b"hello");
        assert_eq!(io.pending(), 0);
    }

    /// Dropping an `AsyncIo` with operations in flight frees what they hold and runs no continuation.
    #[test]
    fn dropping_a_positional_engine_with_operations_in_flight_frees_them_without_running_them() {
        let token = Arc::new(());
        let io = positional();
        let (ours, _theirs) = socket_pair(4096);
        // One the engine has seen (a visit left it waiting) and one it has not.
        let seen = token.clone();
        io.recv(
            ours.as_raw_fd(),
            buf(64),
            Box::new(move |_| {
                let _ = &seen;
                panic!("a continuation ran after the drop");
            }),
        )
        .unwrap();
        assert_eq!(io.reap(), 0);
        let unseen = token.clone();
        io.recv(
            ours.as_raw_fd(),
            buf(64),
            Box::new(move |_| {
                let _ = &unseen;
                panic!("a continuation ran after the drop");
            }),
        )
        .unwrap();
        assert_eq!(Arc::strong_count(&token), 3);
        drop(io);
        assert_eq!(Arc::strong_count(&token), 1);
    }

    /// A connect makes a blocking socket non-blocking for its handshake and gives it back as it was.
    #[test]
    fn a_positional_connect_leaves_a_blocking_socket_blocking() {
        let io = positional();
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let socket = io.socket_for(listener.local_addr().unwrap()).unwrap();
        let before = fd_flags(socket.as_raw_fd());
        assert_eq!(before & libc::O_NONBLOCK, 0);
        io.connect(
            socket.as_raw_fd(),
            listener.local_addr().unwrap(),
            connected(1),
        )
        .unwrap();
        let done = poll(&io, 1).pop().unwrap().1.done.unwrap();
        assert!(done.result.is_ok());
        assert_eq!(fd_flags(socket.as_raw_fd()), before);
        // The handshake ran non-blocking: the engine asked `poll(2)` about the socket.
        assert!(
            counts(&io).0 >= 1,
            "the connect went through the readiness call"
        );
    }

    fn fd_flags(fd: RawFd) -> i32 {
        // SAFETY: a plain `fcntl` of a descriptor the test owns.
        unsafe { libc::fcntl(fd, libc::F_GETFL) }
    }

    /// The positional engine of an `AsyncIo` opened on it.
    fn positional_of(io: &AsyncIo) -> &crate::positional::Positional {
        match &io.engine {
            Platform::Positional(engine) => engine,
            Platform::Uring(_) => unreachable!("opened on the positional engine"),
        }
    }

    fn counts(io: &AsyncIo) -> (usize, usize) {
        let engine = positional_of(io);
        (
            engine.readiness_calls.load(Ordering::Relaxed),
            engine.attempts.load(Ordering::Relaxed),
        )
    }

    /// Operations that wait for a descriptor cost one readiness call between them, and no attempt of their own.
    #[test]
    fn idle_receives_cost_one_readiness_call_per_reap_and_no_attempts() {
        let io = positional();
        let pairs: Vec<_> = (0..50).map(|_| socket_pair(4096)).collect();
        for (n, (ours, _theirs)) in pairs.iter().enumerate() {
            io.recv(ours.as_raw_fd(), buf(64), then(n as u64)).unwrap();
        }
        assert_eq!(io.reap(), 0);
        assert_eq!(counts(&io), (1, 0), "one call, no receive attempted");
        assert_eq!(io.reap(), 0);
        assert_eq!(counts(&io), (2, 0));
        // A file read beside them is attempted (it needs no readiness) and the idle ones still are not.
        let data = content(100, 9);
        let (_guard, file) = temp_file("beside", &data);
        io.read(file.as_raw_fd(), 0, buf(100), then(1000)).unwrap();
        assert_eq!(io.reap(), 1);
        assert_eq!(counts(&io), (3, 1));
        // One of them becomes ready: only it is attempted.
        let (_, theirs) = &pairs[7];
        (&*theirs).write_all(b"x").unwrap();
        assert_eq!(io.reap(), 1);
        assert_eq!(counts(&io), (4, 2));
    }

    /// The attempts of a reap are bounded however many operations are ready.
    #[test]
    fn a_reap_makes_at_most_the_attempt_budget() {
        let io = positional();
        let (ours, mut theirs) = socket_pair(4096);
        theirs.write_all(&[1; 200]).unwrap();
        for n in 0..100 {
            io.recv(ours.as_raw_fd(), buf(1), then(n)).unwrap();
        }
        io.reap();
        assert_eq!(counts(&io).1, 64);
    }

    /// Two operations on one descriptor are one entry of the readiness call, and both are told.
    #[test]
    fn a_receive_and_a_send_on_one_descriptor_share_one_readiness_entry() {
        let io = positional();
        let (ours, mut theirs) = socket_pair(4096);
        io.recv(ours.as_raw_fd(), buf(8), then(1)).unwrap();
        io.send(
            ours.as_raw_fd(),
            IoBuf {
                filled: 3,
                bytes: vec![7; 3],
            },
            then(2),
        )
        .unwrap();
        theirs.write_all(b"hi").unwrap();
        let engine = positional_of(&io);
        let done = poll(&io, 2);
        assert_eq!(done.len(), 2);
        assert!(
            done.iter()
                .all(|(_, d)| d.done.as_ref().unwrap().result.is_ok())
        );
        assert_eq!(
            engine.entries.load(Ordering::Relaxed),
            1,
            "one pollfd for the two operations"
        );
    }

    /// A write to a blocking pipe moves at most `PIPE_BUF` per attempt, never toggles the flag, and the reap
    /// returns however full the pipe is.
    #[test]
    fn a_stream_write_to_a_blocking_pipe_moves_at_most_pipe_buf_an_attempt() {
        let io = positional();
        let mut ends = [0; 2];
        // SAFETY: `pipe2` fills the two descriptors of the array.
        let made = unsafe { libc::pipe2(ends.as_mut_ptr(), libc::O_CLOEXEC) };
        assert_eq!(made, 0);
        // SAFETY: both descriptors were just made and nothing else owns them.
        let (read_end, write_end) =
            unsafe { (OwnedFd::from_raw_fd(ends[0]), OwnedFd::from_raw_fd(ends[1])) };
        let flags = fd_flags(write_end.as_raw_fd());
        assert_eq!(flags & libc::O_NONBLOCK, 0);
        let total = 1 << 20;
        io.write(
            write_end.as_raw_fd(),
            0,
            IoBuf {
                filled: total,
                bytes: vec![5; total],
            },
            then(1),
        )
        .unwrap();
        for _ in 0..200 {
            io.reap();
        }
        let mut queued: libc::c_int = 0;
        // SAFETY: FIONREAD writes one int.
        let asked = unsafe { libc::ioctl(read_end.as_raw_fd(), libc::FIONREAD, &raw mut queued) };
        assert_eq!(asked, 0);
        let (_, attempts) = counts(&io);
        assert!(queued > 0 && (queued as usize) < total);
        // The first attempt is the positional write that finds the pipe is a stream; each later one wrote one piece.
        assert_eq!(attempts, 1 + queued as usize / libc::PIPE_BUF);
        assert_eq!(fd_flags(write_end.as_raw_fd()), flags);
        assert_eq!(io.pending(), 1);
    }

    /// A pipe is not seekable: a read and a write on it go on as a stream, on either engine.
    fn a_pipe_carries_a_write_and_a_read(io: Arc<AsyncIo>) {
        let mut ends = [0; 2];
        // SAFETY: `pipe2` fills the two descriptors of the array.
        let made = unsafe { libc::pipe2(ends.as_mut_ptr(), libc::O_CLOEXEC) };
        assert_eq!(made, 0);
        // SAFETY: both descriptors were just made and nothing else owns them.
        let (read_end, write_end) =
            unsafe { (OwnedFd::from_raw_fd(ends[0]), OwnedFd::from_raw_fd(ends[1])) };
        let data = content(100, 21);
        // The read waits for the write.
        io.read(read_end.as_raw_fd(), 0, buf(100), then(1)).unwrap();
        io.write(
            write_end.as_raw_fd(),
            0,
            IoBuf {
                filled: data.len(),
                bytes: data.clone(),
            },
            then(2),
        )
        .unwrap();
        let mut done = poll(&io, 2);
        done.sort_by_key(|(tag, _)| *tag);
        let read = done.remove(0).1.done.unwrap();
        assert_eq!(read.result.unwrap(), 100);
        assert_eq!(read.buf.bytes, data);
        assert_eq!(done.remove(0).1.done.unwrap().result.unwrap(), 100);
        assert_eq!(io.pending(), 0);
    }

    fn fifo_path(name: &str) -> (TempFile, std::ffi::CString) {
        let (guard, file) = temp_file(name, b"");
        drop(file);
        std::fs::remove_file(&guard.0).unwrap();
        let path = {
            use std::os::unix::ffi::OsStrExt;
            std::ffi::CString::new(guard.0.as_os_str().as_bytes()).unwrap()
        };
        // SAFETY: `path` is a NUL-terminated string.
        assert_eq!(unsafe { libc::mkfifo(path.as_ptr(), 0o600) }, 0);
        (guard, path)
    }

    fn open_request(path: &std::path::Path, write: bool) -> Request {
        Request::Open {
            path: path.to_owned(),
            spec: crate::OpenSpec {
                read: !write,
                write,
                create: Create::Never,
                unix_mode: 0o644,
            },
        }
    }

    fn open_then(tag: u64) -> Then {
        Then::Raw(Box::new(move |result| {
            Box::new(Done {
                header: FrameHeader::new(tag),
                done: Some(IoDone {
                    buf: IoBuf::default(),
                    result,
                }),
                accepted: None,
            })
        }))
    }

    /// The frames an engine's poll found.
    #[allow(clippy::vec_box)] // the frames come out of the engine boxed
    fn engine_found(engine: &dyn Engine) -> Vec<Box<Done>> {
        let mut found = Vec::new();
        engine.poll(&mut || true, &mut |frame| {
            found.push(
                (frame as Box<dyn std::any::Any>)
                    .downcast::<Done>()
                    .expect("a Done frame"),
            );
            true
        });
        found
    }

    /// Opening a FIFO never holds the reaping worker; a write open with no reader waits in the line and a cancel
    /// reaches it. The descriptor comes back blocking, as from io_uring.
    #[test]
    fn a_fifo_open_never_holds_the_reap_and_a_cancel_ends_the_open_that_has_no_reader() {
        let (guard, _path) = fifo_path("fifo");
        let engine = crate::positional::Positional::new();
        engine
            .submit(
                open_request(&guard.0, false),
                IoBuf::default().into(),
                open_then(1),
                None,
                &Cell::new(false),
            )
            .unwrap();
        let mut found = engine_found(&engine);
        let read_end = found
            .pop()
            .expect("the read open ended at once")
            .done
            .unwrap()
            .result
            .unwrap();
        assert_eq!(
            fd_flags(read_end as RawFd) & libc::O_NONBLOCK,
            0,
            "blocking, as io_uring's"
        );
        // SAFETY: the engine opened this descriptor for us.
        unsafe { libc::close(read_end as RawFd) };
        let id = engine
            .submit(
                open_request(&guard.0, true),
                IoBuf::default().into(),
                open_then(2),
                None,
                &Cell::new(false),
            )
            .unwrap();
        assert!(engine_found(&engine).is_empty());
        assert!(engine_found(&engine).is_empty());
        engine.cancel(id);
        let error = engine_found(&engine)
            .pop()
            .unwrap()
            .done
            .unwrap()
            .result
            .unwrap_err();
        assert!(Cancelled::is(&error), "{error}");
    }

    fn attempts_of(engine: &crate::positional::Positional) -> usize {
        engine.attempts.load(Ordering::Relaxed)
    }

    /// A write open of a FIFO with no reader is not attempted at every visit, and still opens once a reader is
    /// there.
    #[test]
    fn a_fifo_open_with_no_reader_backs_off() {
        let (guard, _path) = fifo_path("fifo-backoff");
        let engine = crate::positional::Positional::new();
        engine
            .submit(
                open_request(&guard.0, true),
                IoBuf::default().into(),
                open_then(1),
                None,
                &Cell::new(false),
            )
            .unwrap();
        for _ in 0..100 {
            assert!(engine_found(&engine).is_empty());
        }
        let tried = attempts_of(&engine);
        assert!(
            (1..=10).contains(&tried),
            "{tried} attempts in 100 visits of an open that has no reader"
        );
        let reader = {
            use std::os::unix::fs::OpenOptionsExt;
            std::fs::OpenOptions::new()
                .read(true)
                .custom_flags(libc::O_NONBLOCK)
                .open(&guard.0)
                .unwrap()
        };
        // The wait is bounded: the open is attempted again within the longest gap.
        let mut found = Vec::new();
        for _ in 0..=64 {
            found = engine_found(&engine);
            if !found.is_empty() {
                break;
            }
        }
        let write_end = found.pop().expect("the open ended").done.unwrap().result;
        // SAFETY: the engine opened this descriptor for us.
        unsafe { libc::close(write_end.unwrap() as RawFd) };
        drop(reader);
    }

    /// A file leased to another process: the open the engine makes non-blocking is refused with EWOULDBLOCK, and
    /// the caller, who asked for a blocking open, gets the file once the lease is gone.
    #[test]
    fn an_open_of_a_leased_file_waits_for_the_lease_like_a_blocking_open() {
        let (guard, file) = temp_file("leased", b"x");
        drop(file);
        // The lease holder is told of a break with SIGIO, which ends a process that does not handle it.
        // SAFETY: setting a signal's disposition to ignore; nothing else here uses SIGIO.
        unsafe { libc::signal(libc::SIGIO, libc::SIG_IGN) };
        let holder = std::fs::File::open(&guard.0).unwrap();
        // SAFETY: a plain `fcntl` of a descriptor the test owns.
        let leased = unsafe { libc::fcntl(holder.as_raw_fd(), libc::F_SETLEASE, libc::F_RDLCK) };
        if leased != 0 {
            let error = io::Error::last_os_error();
            // Only a file system without leases is a reason to skip; anything else (a stray writer, say) is a
            // test that is broken.
            assert_eq!(error.raw_os_error(), Some(libc::EINVAL), "{error}");
            eprintln!("no lease on this file system: the test is skipped");
            return;
        }
        let engine = crate::positional::Positional::new();
        engine
            .submit(
                open_request(&guard.0, true),
                IoBuf::default().into(),
                open_then(1),
                None,
                &Cell::new(false),
            )
            .unwrap();
        for _ in 0..20 {
            assert!(
                engine_found(&engine).is_empty(),
                "the open waits while the lease is held"
            );
        }
        assert!(attempts_of(&engine) >= 1);
        // SAFETY: as above.
        let released = unsafe { libc::fcntl(holder.as_raw_fd(), libc::F_SETLEASE, libc::F_UNLCK) };
        assert_eq!(released, 0);
        let mut found = Vec::new();
        for _ in 0..=64 {
            found = engine_found(&engine);
            if !found.is_empty() {
                break;
            }
        }
        let opened = found.pop().expect("the open ended").done.unwrap().result;
        // SAFETY: the engine opened this descriptor for us.
        unsafe { libc::close(opened.unwrap() as RawFd) };
    }

    /// A read that is the only operation due takes the visit's whole budget; two share it a few pieces at a time.
    #[test]
    fn a_lone_long_read_moves_more_than_a_turn_in_one_reap_and_two_share() {
        let pieces = 12;
        let data = content(pieces * crate::READ_PIECE, 5);
        let (_guard, file) = temp_file("long-read", &data);
        let io = positional();
        io.read(file.as_raw_fd(), 0, buf(data.len()), then(1))
            .unwrap();
        assert_eq!(io.reap(), 1, "one visit moved all the pieces");
        assert_eq!(counts(&io).1, pieces);

        let io = positional();
        io.read(file.as_raw_fd(), 0, buf(data.len()), then(1))
            .unwrap();
        io.read(file.as_raw_fd(), 0, buf(data.len()), then(2))
            .unwrap();
        assert_eq!(io.reap(), 0, "neither ran away with the visit");
        assert_eq!(counts(&io).1, 8, "four pieces each");
        assert_eq!(io.reap(), 0);
        assert_eq!(io.reap(), 2);
    }

    /// A cancelled operation due in the same visit is a rival: the live read moves four pieces, not all the budget.
    #[test]
    fn a_cancelled_operation_in_the_visit_is_a_rival_for_the_budget() {
        let data = content(12 * crate::READ_PIECE, 6);
        let (_guard, file) = temp_file("long-read-rival", &data);
        let io = positional();
        io.read(file.as_raw_fd(), 0, buf(data.len()), then(1))
            .unwrap();
        let (ours, _theirs) = socket_pair(4096);
        let doomed = io
            .recv_with(ours.as_raw_fd(), buf(64), OpOptions::default(), then(2))
            .unwrap();
        io.cancel(doomed);
        assert_eq!(
            io.reap(),
            1,
            "the read did not take the whole visit: only the cancelled operation ended"
        );
        assert_eq!(counts(&io).1, 5, "four pieces and the cancelled attempt");
        let ended = poll(&io, 2);
        let cancelled = ended.iter().find(|(tag, _)| *tag == 2).unwrap();
        assert!(Cancelled::is(
            cancelled
                .1
                .done
                .as_ref()
                .unwrap()
                .result
                .as_ref()
                .unwrap_err()
        ));
    }

    /// An accept on a blocking listener cannot block the reap, and leaves the listener as it was.
    #[test]
    fn a_positional_accept_leaves_a_blocking_listener_blocking() {
        let io = positional();
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let before = fd_flags(listener.as_raw_fd());
        assert_eq!(before & libc::O_NONBLOCK, 0);
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
        assert_eq!(io.reap(), 0, "nothing to accept: the reap returns");
        assert_eq!(fd_flags(listener.as_raw_fd()), before);
        #[allow(clippy::disallowed_types)] // the test's peer
        let _peer = std::net::TcpStream::connect(addr).unwrap();
        assert!(poll(&io, 1).pop().unwrap().1.accepted.unwrap().is_ok());
        assert_eq!(fd_flags(listener.as_raw_fd()), before);
    }

    /// A connect (or its guard) dropped with the handshake under way puts the socket's flags back.
    #[test]
    fn dropping_a_connect_under_way_restores_the_sockets_flags() {
        let (ours, _theirs) = socket_pair(4096);
        let before = fd_flags(ours.as_raw_fd());
        // SAFETY: a plain `fcntl` of a socket the test owns.
        unsafe { libc::fcntl(ours.as_raw_fd(), libc::F_SETFL, before | libc::O_NONBLOCK) };
        let guard = crate::positional::Restore::new(ours.as_raw_fd(), Some(before));
        drop(guard);
        assert_eq!(fd_flags(ours.as_raw_fd()), before);
    }

    /// An `AsyncIo` on io_uring, or on the positional engine.
    fn open_on(backend: crate::IoBackend) -> Arc<AsyncIo> {
        let io = AsyncIo::open_with_options(AsyncIoOptions {
            positional_io: backend == crate::IoBackend::Positional,
            ..AsyncIoOptions::default()
        })
        .unwrap();
        if backend == crate::IoBackend::IoUring && io.backend() != backend {
            // The system refuses io_uring: the io_uring engine's own tests cannot run here.
            panic!("this machine does not allow io_uring");
        }
        io
    }

    /// Runs each of the listed tests once on io_uring and once on the positional engine, so that both meet
    /// the same contract.
    macro_rules! on_both_engines {
        ($($name:ident),* $(,)?) => {
            mod on_io_uring {
                $(#[test] fn $name() { super::$name(super::open_on(crate::IoBackend::IoUring)); })*
            }
            mod on_positional {
                $(#[test] fn $name() { super::$name(super::open_on(crate::IoBackend::Positional)); })*
            }
        };
    }

    #[test]
    fn short_reads_and_writes_are_resubmitted_until_every_byte_moved_on_either_engine() {
        short_reads_and_writes_are_resubmitted_until_every_byte_moved(short_pieces);
        short_reads_and_writes_are_resubmitted_until_every_byte_moved(short_positional);
    }

    #[test]
    fn a_short_send_is_resubmitted_until_every_byte_is_in_the_kernel_on_either_engine() {
        a_short_send_is_resubmitted_until_every_byte_is_in_the_kernel(short_pieces);
        a_short_send_is_resubmitted_until_every_byte_is_in_the_kernel(short_positional);
    }

    #[test]
    fn a_send_stopped_part_way_carries_its_error_on_either_engine() {
        a_send_stopped_part_way_carries_its_error(short_pieces);
        a_send_stopped_part_way_carries_its_error(short_positional);
    }

    on_both_engines!(
        a_pipe_carries_a_write_and_a_read,
        a_reads_completion_is_a_frame_emitted_when_polled_after_its_bytes_arrived,
        a_read_past_the_end_of_the_file_fails,
        a_read_the_file_ends_inside_completes_with_the_bytes_there_were,
        operations_beyond_the_queues_room_wait_and_all_complete,
        a_poll_hands_on_no_more_than_its_demand,
        a_pull_stops_before_the_completion_its_stop_matches_and_hands_it_on_later_in_order,
        a_write_completes_when_all_its_bytes_are_in_the_file,
        empty_reads_and_writes_and_submissions_after_complete_are_refused_at_once,
        accept_send_and_receive_on_a_loopback_connection,
        a_connect_completes_once_connected_or_refused,
        a_send_to_a_closed_peer_fails_without_a_signal,
        a_datagram_goes_to_its_address,
        an_atomic_save_opens_a_temporary_file_writes_it_and_renames_it_over_the_old_one,
        file_len_gives_the_size_of_an_open_file_and_the_kernels_error_for_a_bad_descriptor,
        opening_or_renaming_what_is_not_there_fails_with_the_kernels_error,
        requests_push_completions_downstream,
    );

    /// A completion that belongs to no operation (the `debug_assert!` of the poll fails in a debug build) does not
    /// stop the ring's reaping for good: the next poll reaps it, and an operation submitted after it completes.
    #[test]
    fn a_stray_completion_does_not_stop_the_ring_from_being_reaped() {
        let data = content(4096, 3);
        let (_guard, file) = temp_file("stray", &data);
        let io = AsyncIo::open_with(
            Box::new(|submitter: &Submitter<'_>| submitter.submit()),
            MAX_PIECE,
        )
        .unwrap();
        // The first operation makes the thread's ring; the stray completion is queued behind it.
        io.read(file.as_raw_fd(), 0, buf(16), then(1)).unwrap();
        let uring = uring_of(&io);
        let ring = lock!(&uring.rings)[0].clone();
        ring.offer(Item::One(opcode::Nop::new().build().user_data(0x5757)));
        ring.flush_waiting(&*uring.enter);
        let deadline = Instant::now() + Duration::from_secs(10);
        let mut seen: Vec<u64> = Vec::new();
        let collect = |io: &AsyncIo, seen: &mut Vec<u64>| {
            io.pull(
                &mut |frame| seen.push(frame.header().id_hash()),
                &mut |_| false,
                u64::MAX,
            );
        };
        // Reaps until the stray completion is met: it fails the check of the poll (in a debug build).
        loop {
            assert!(
                Instant::now() < deadline,
                "the stray completion was not reaped"
            );
            let reaped = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| io.reap()));
            collect(&io, &mut seen);
            if reaped.is_err()
                || (!cfg!(debug_assertions) && ring.in_kernel.load(Ordering::Acquire) == 0)
            {
                break;
            }
        }
        // An operation submitted after it completes, and so does the first if the stray was ahead of it.
        io.read(file.as_raw_fd(), 0, buf(16), then(2)).unwrap();
        while !(seen.contains(&1) && seen.contains(&2)) {
            assert!(
                Instant::now() < deadline,
                "the ring was never reaped again after the stray completion"
            );
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| io.reap()));
            collect(&io, &mut seen);
        }
        // The panic of a debug build skipped no count: nothing is left for a key to look at.
        assert_eq!(io.pending(), 0);
        assert_eq!(io.unreaped.load(Ordering::Acquire), 0);
    }

    /// An engine that refuses an operation drops its continuation; if that drop panics, `pending` and `unreaped`
    /// are back where they were.
    #[test]
    fn a_refused_operation_whose_continuation_panics_on_drop_leaves_the_counts() {
        struct Bomb;
        impl Drop for Bomb {
            fn drop(&mut self) {
                if !std::thread::panicking() {
                    panic!("the continuation's state failed to drop");
                }
            }
        }
        let io = AsyncIo::open_with(
            Box::new(|submitter: &Submitter<'_>| submitter.submit()),
            MAX_PIECE,
        )
        .unwrap();
        let bomb = Bomb;
        // A path with a NUL byte is refused by the engine, which drops the continuation.
        let refused = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            io.open_path(
                std::path::Path::new("a\0b"),
                OpenSpec::default(),
                Box::new(move |_| {
                    let _bomb = &bomb;
                    Box::new(Done {
                        header: FrameHeader::new(1),
                        done: None,
                        accepted: None,
                    })
                }),
            )
        }));
        assert!(refused.is_err(), "the drop of the continuation panicked");
        assert_eq!(io.pending(), 0);
        assert_eq!(io.unreaped.load(Ordering::Acquire), 0);
    }

    /// A panic under the submission-queue lock (here the enter hook's, once) does not leave the ring's entries
    /// unsubmitted: the operation was accepted, so its counts stay, and a later poll hands it over and reaps it.
    #[test]
    fn a_panic_under_the_submission_lock_does_not_stop_the_ring_from_submitting() {
        let data = content(4096, 5);
        let (_guard, file) = temp_file("sqpanic", &data);
        let calls = Arc::new(AtomicUsize::new(0));
        let counted = calls.clone();
        let io = AsyncIo::open_with(
            Box::new(move |submitter: &Submitter<'_>| {
                if counted.fetch_add(1, Ordering::AcqRel) == 0 {
                    panic!("the enter hook failed");
                }
                submitter.submit()
            }),
            MAX_PIECE,
        )
        .unwrap();
        let first = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            io.read(file.as_raw_fd(), 0, buf(16), then(1))
        }));
        assert!(first.is_err(), "the hook's panic reaches the submitter");
        // The operation was accepted before the panic: counted, and handed over by a later poll, with no
        // other submission to do it.
        assert_eq!(io.pending(), 1);
        let out = poll(&io, 1);
        assert_eq!(out[0].0, 1);
        assert_eq!(io.pending(), 0);
        assert_eq!(io.unreaped.load(Ordering::Acquire), 0);
    }

    /// A panic under the completion-queue lock (a consumer's, in the test-only direct reap) does not stop the ring
    /// from being reaped: the next reap takes the poisoned lock.
    #[test]
    fn a_panic_under_the_completion_lock_does_not_stop_the_ring_from_being_reaped() {
        let data = content(4096, 9);
        let (_guard, file) = temp_file("cqpanic", &data);
        let io = AsyncIo::open_with(
            Box::new(|submitter: &Submitter<'_>| submitter.submit()),
            MAX_PIECE,
        )
        .unwrap();
        io.read(file.as_raw_fd(), 0, buf(16), then(1)).unwrap();
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            assert!(Instant::now() < deadline, "the read did not complete");
            let reaped = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                io.front.reap(
                    &io.engine,
                    &mut |_| panic!("the consumer failed"),
                    &mut |_| false,
                    8,
                )
            }));
            if reaped.is_err() {
                break;
            }
        }
        io.read(file.as_raw_fd(), 0, buf(16), then(2)).unwrap();
        let out = poll(&io, 1);
        assert_eq!(out[0].0, 2);
    }

    /// A refused operation's continuation that submits another operation as it drops: the refusal still takes its
    /// own counts back, and the inner operation keeps its own.
    #[test]
    fn a_refused_operation_whose_continuation_submits_as_it_drops_leaves_the_right_counts() {
        struct Submits {
            io: Arc<AsyncIo>,
            fd: i32,
        }
        impl Drop for Submits {
            fn drop(&mut self) {
                self.io.read(self.fd, 0, buf(16), then(1)).unwrap();
            }
        }
        let data = content(4096, 11);
        let (_guard, file) = temp_file("nested", &data);
        let io = AsyncIo::open_with(
            Box::new(|submitter: &Submitter<'_>| submitter.submit()),
            MAX_PIECE,
        )
        .unwrap();
        let held = Submits {
            io: io.clone(),
            fd: file.as_raw_fd(),
        };
        let refused = io.open_path(
            std::path::Path::new("a\0b"),
            OpenSpec::default(),
            Box::new(move |_| {
                let _held = &held;
                Box::new(Done {
                    header: FrameHeader::new(9),
                    done: None,
                    accepted: None,
                })
            }),
        );
        assert!(refused.is_err());
        assert_eq!(io.pending(), 1);
        assert_eq!(io.unreaped.load(Ordering::Acquire), 1);
        let out = poll(&io, 1);
        assert_eq!(out[0].0, 1);
        assert_eq!(io.pending(), 0);
        assert_eq!(io.unreaped.load(Ordering::Acquire), 0);
    }

    /// A reap that a continuation runs (here on another `AsyncIo`) does not clear the stray completion the
    /// enclosing reap met: the enclosing reap still fails its check (a debug build).
    #[cfg(debug_assertions)]
    #[test]
    fn a_reap_inside_a_continuation_leaves_the_enclosing_reaps_stray_check() {
        let data = content(4096, 13);
        let (_guard, file) = temp_file("enclosing", &data);
        let open = || {
            AsyncIo::open_with(
                Box::new(|submitter: &Submitter<'_>| submitter.submit()),
                MAX_PIECE,
            )
            .unwrap()
        };
        let (io, other) = (open(), open());
        io.read(file.as_raw_fd(), 0, buf(16), then(1)).unwrap();
        poll(&io, 1);
        let uring = uring_of(&io);
        let ring = lock!(&uring.rings)[0].clone();
        // The stray completion first, then an operation whose continuation reaps `other`.
        ring.offer(Item::One(opcode::Nop::new().build().user_data(0x5757)));
        ring.flush_waiting(&*uring.enter);
        let ran = Arc::new(AtomicUsize::new(0));
        let (seen, inner) = (ran.clone(), then(2));
        io.read(
            file.as_raw_fd(),
            0,
            buf(16),
            Box::new(move |done| {
                other.reap();
                seen.fetch_add(1, Ordering::AcqRel);
                inner(done)
            }),
        )
        .unwrap();
        // Both completions are in the ring's queue before the reap: the stray one first (it was posted before the
        // read was submitted), so the reap meets it, then the continuation that reaps.
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            let queued = {
                let _reaping = ring.cq.try_lock().expect("no reap is inside");
                // SAFETY: the `cq` lock is held, so this is the only reader of the ring's completion queue; it
                // takes nothing off.
                unsafe { ring.uring.completion_shared() }.len()
            };
            if queued >= 2 {
                break;
            }
            assert!(Instant::now() < deadline, "the completions were not posted");
        }
        let reaped = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| io.reap()));
        assert_eq!(ran.load(Ordering::Acquire), 1, "the nested reap ran");
        assert!(reaped.is_err(), "the stray completion failed the check");
    }
}
