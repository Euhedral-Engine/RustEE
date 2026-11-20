//! The slab of operation records, and the state word of each slot.
//!
//! A slot's word says in one atomic what a lock over a table said before: its state, whether the
//! operation was asked to end and why, whether a cancel of its call is under way (the pin), the
//! route it took, and the generation of its issue. Every change of the word is a compare-and-set, so
//! a thread that loses to another re-reads and decides again; nothing waits.
//!
//! The records themselves sit in chunks that are never moved or freed while the slab lives, so a
//! packet that carries a record's address can always be mapped to its slot ([`Slab::slot_at`]) and
//! read safely. This module holds the pure state machine: it makes no system call, so the tests run
//! under Miri.

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicPtr, AtomicU32, AtomicU64, Ordering};

use euhedral_data_structures::queues::{Consumer, MpmcQueue, Multi, Producer};

/// A slot's state; see the module docs of the engine.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub(super) enum State {
    Idle = 0,
    Issuing = 1,
    InFlight = 2,
    Busy = 3,
}

impl State {
    fn from_bits(bits: u64) -> Self {
        match bits & 3 {
            0 => Self::Idle,
            1 => Self::Issuing,
            2 => Self::InFlight,
            _ => Self::Busy,
        }
    }
}

/// Why an operation ends before its call did.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub(super) enum End {
    Cancelled = 1,
    TimedOut = 2,
}

/// How the operation reaches the kernel.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub(super) enum RouteTag {
    Port = 0,
    Ring = 1,
    Pool = 2,
}

const STATE_MASK: u64 = 0b11;
const END_SHIFT: u32 = 2;
const END_MASK: u64 = 0b11 << END_SHIFT;
const PIN: u64 = 1 << 4;
const CANCELLABLE: u64 = 1 << 5;
const ROUTE_SHIFT: u32 = 6;
const ROUTE_MASK: u64 = 0b11 << ROUTE_SHIFT;
const RING_SHIFT: u32 = 8;
const RING_MASK: u64 = 0xff << RING_SHIFT;
const GENERATION_SHIFT: u32 = 32;

/// A slot's state word: generation (32 bits), ring index (8), route (2), cancellable (1), pin (1),
/// end (2), state (2).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) struct Word(u64);

impl Word {
    pub(super) fn state(self) -> State {
        State::from_bits(self.0)
    }

    /// Why the operation was asked to end, if it was.
    pub(super) fn end(self) -> Option<End> {
        match (self.0 & END_MASK) >> END_SHIFT {
            1 => Some(End::Cancelled),
            2 => Some(End::TimedOut),
            _ => None,
        }
    }

    /// A cancel of the call is being made: the slot is not taken, and so not reused, until it ends.
    pub(super) fn pinned(self) -> bool {
        self.0 & PIN != 0
    }

    /// The call can be cancelled (a socket operation on the port).
    pub(super) fn cancellable(self) -> bool {
        self.0 & CANCELLABLE != 0
    }

    pub(super) fn route(self) -> RouteTag {
        match (self.0 & ROUTE_MASK) >> ROUTE_SHIFT {
            0 => RouteTag::Port,
            1 => RouteTag::Ring,
            _ => RouteTag::Pool,
        }
    }

    /// The index of the ring a ring operation was routed through.
    pub(super) fn ring(self) -> usize {
        ((self.0 & RING_MASK) >> RING_SHIFT) as usize
    }

    pub(super) fn generation(self) -> u32 {
        (self.0 >> GENERATION_SHIFT) as u32
    }

    fn with_state(self, state: State) -> Self {
        Self((self.0 & !STATE_MASK) | state as u64)
    }

    fn next_generation(self) -> Self {
        let generation = self.generation().wrapping_add(1);
        Self((self.0 & u64::from(u32::MAX)) | (u64::from(generation) << GENERATION_SHIFT))
    }
}

/// What a thread that asks an operation to end finds.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Asked {
    /// The operation ended, or was asked already: nothing was done.
    Nothing,
    /// The reason was recorded; the call is made to end when it is armed or taken again.
    Recorded,
    /// The reason was recorded and the call is in flight: the caller cancels it, then calls
    /// [`SlotCell::unpin`].
    Cancel,
}

/// A slot: its record, its state word and the serial of its operation. `rec` is first, so a
/// record's address is its slot's.
#[repr(C)]
pub(super) struct SlotCell<T> {
    rec: UnsafeCell<T>,
    word: AtomicU64,
    /// Names the operation in the slot: it stays the same through the operation's pieces (the
    /// generation changes at each), and is new for each operation.
    serial: AtomicU64,
}

// SAFETY: the record is touched by one thread at a time, as the word says: the issuer while
// `Issuing`, the thread that took it while `Busy`, and the kernel (through raw pointers) while
// `InFlight`; the canceller of a pinned slot reads it, which no one writes while the pin is up.
unsafe impl<T: Send> Sync for SlotCell<T> {}

impl<T: Default> SlotCell<T> {
    fn new() -> Self {
        Self {
            rec: UnsafeCell::new(T::default()),
            word: AtomicU64::new(Word(0).with_state(State::Idle).0),
            serial: AtomicU64::new(0),
        }
    }
}

impl<T> SlotCell<T> {
    /// The record. What may be done with it depends on the word.
    pub(super) fn rec(&self) -> *mut T {
        self.rec.get()
    }

    pub(super) fn load(&self) -> Word {
        Word(self.word.load(Ordering::Acquire))
    }

    pub(super) fn serial(&self) -> u64 {
        self.serial.load(Ordering::Acquire)
    }

    fn swap(&self, seen: Word, new: Word) -> bool {
        self.word
            .compare_exchange(seen.0, new.0, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
    }

    /// The idle slot, which the caller owns (it came from [`Slab::alloc`]), begins an operation:
    /// the generation is new. Returns it.
    pub(super) fn open(
        &self,
        state: State,
        route: RouteTag,
        ring: usize,
        cancellable: bool,
        serial: u64,
    ) -> u32 {
        let old = self.load();
        debug_assert_eq!(old.state(), State::Idle);
        let mut word = old.next_generation().with_state(state).0 & !(END_MASK | PIN);
        word &= !(CANCELLABLE | ROUTE_MASK | RING_MASK);
        word |= (route as u64) << ROUTE_SHIFT;
        word |= (ring as u64 & 0xff) << RING_SHIFT;
        if cancellable {
            word |= CANCELLABLE;
        }
        // The serial is published before the word: a reader that sees the word sees it.
        self.serial.store(serial, Ordering::Release);
        self.word.store(word, Ordering::Release);
        Word(word).generation()
    }

    /// The holder of a `Busy` slot issues its next piece: a new generation in `state`. Not if it
    /// was asked to end meanwhile: that is the reason, and the slot stays as it was.
    pub(super) fn reissue(&self, state: State) -> Result<u32, End> {
        loop {
            let seen = self.load();
            debug_assert_eq!(seen.state(), State::Busy);
            if let Some(why) = seen.end() {
                return Err(why);
            }
            let new = seen.next_generation().with_state(state);
            if self.swap(seen, new) {
                return Ok(new.generation());
            }
        }
    }

    /// The holder moves the slot to `state`. Going in flight after being asked to end meanwhile (a
    /// cancel or a limit while its call was being made) pins the slot if its call can be cancelled:
    /// returns `true`, and the caller cancels the call, then calls [`unpin`](Self::unpin).
    pub(super) fn arm(&self, state: State) -> bool {
        loop {
            let seen = self.load();
            let mut new = seen.with_state(state);
            let pin = state == State::InFlight && seen.end().is_some() && seen.cancellable();
            if pin {
                new.0 |= PIN;
            }
            if self.swap(seen, new) {
                return pin;
            }
        }
    }

    /// The cancel is made: the slot can be taken.
    pub(super) fn unpin(&self) {
        self.word.fetch_and(!PIN, Ordering::AcqRel);
    }

    /// The holder is done with the slot. The word goes idle before the slot is offered again.
    pub(super) fn release(&self) {
        let seen = self.load();
        debug_assert_eq!(seen.state(), State::Busy);
        let idle = Word(seen.0 & !(END_MASK | PIN | STATE_MASK));
        self.word.store(idle.0, Ordering::Release);
    }

    /// Asks the operation `serial` to end, for `why`, from any thread.
    pub(super) fn ask_to_end(&self, serial: u64, why: End) -> Asked {
        loop {
            let seen = self.load();
            if seen.state() == State::Idle || seen.end().is_some() {
                return Asked::Nothing;
            }
            // The serial belongs to the operation of `seen` if the word did not change meanwhile:
            // the slot goes idle before it is begun again, and a begin writes the serial first.
            let found = self.serial();
            if self.load() != seen {
                continue;
            }
            if found != serial {
                return Asked::Nothing;
            }
            let pin = seen.state() == State::InFlight && seen.cancellable();
            let mut new = Word(seen.0 | ((why as u64) << END_SHIFT));
            if pin {
                new.0 |= PIN;
            }
            if self.swap(seen, new) {
                return if pin { Asked::Cancel } else { Asked::Recorded };
            }
        }
    }

    /// Takes the slot (`Busy`) if `decide` says so of its word; returns the word it was taken
    /// from. `decide` runs again when the word changed under it, and may read what the word says
    /// is stable (the record of an `InFlight` slot of the port, which only the caller takes).
    pub(super) fn take<R>(&self, mut decide: impl FnMut(Word) -> Result<(), R>) -> Result<Word, R> {
        loop {
            let seen = self.load();
            decide(seen)?;
            if self.swap(seen, seen.with_state(State::Busy)) {
                return Ok(seen);
            }
        }
    }
}

/// Slots in the first chunk (the chunks double up to [`LARGEST_CHUNK`]).
const FIRST_CHUNK: u32 = 64;
const LARGEST_CHUNK: u32 = 4096;
/// Chunks that double: 64, 128, ..., 4096.
const GROWING_CHUNKS: u32 = 7;
/// The slots of the growing chunks.
const GROWN: u32 = FIRST_CHUNK * ((1 << GROWING_CHUNKS) - 1);

/// The first slot of chunk `chunk` and its size.
fn chunk_span(chunk: usize) -> (u32, u32) {
    let chunk = chunk as u32;
    if chunk < GROWING_CHUNKS {
        (FIRST_CHUNK * ((1 << chunk) - 1), FIRST_CHUNK << chunk)
    } else {
        (
            GROWN + (chunk - GROWING_CHUNKS) * LARGEST_CHUNK,
            LARGEST_CHUNK,
        )
    }
}

/// The chunk of `slot`, and its place in the chunk.
fn locate(slot: u32) -> (usize, usize) {
    let chunk = if slot < GROWN {
        (slot / FIRST_CHUNK + 1).ilog2() as usize
    } else {
        (GROWING_CHUNKS + (slot - GROWN) / LARGEST_CHUNK) as usize
    };
    let (start, _) = chunk_span(chunk);
    (chunk, (slot - start) as usize)
}

/// The records of operations, by slot.
pub(super) struct Slab<T> {
    chunks: Box<[AtomicPtr<SlotCell<T>>]>,
    /// Slots made so far.
    made: AtomicU32,
    max: u32,
    freed: Producer<u32, Multi>,
    free: Consumer<u32, Multi>,
}

// SAFETY: the chunks are only reached through `SlotCell`s, which are `Sync` for `T: Send`.
unsafe impl<T: Send> Sync for Slab<T> {}
// SAFETY: as above; the raw chunk pointers own their cells.
unsafe impl<T: Send> Send for Slab<T> {}

impl<T: Default> Slab<T> {
    /// A slab of at most `max` slots.
    pub(super) fn new(max: u32) -> Self {
        let (last, _) = locate(max.saturating_sub(1));
        let queue: MpmcQueue<u32> = MpmcQueue::with_pool(256, 2);
        Self {
            chunks: (0..=last)
                .map(|_| AtomicPtr::new(std::ptr::null_mut()))
                .collect(),
            made: AtomicU32::new(0),
            max,
            freed: queue.producer(),
            free: queue.consumer(),
        }
    }

    /// A slot no one holds, or `None` if all `max` are in use. Its word is idle.
    pub(super) fn alloc(&self) -> Option<u32> {
        if let Some(slot) = self.free.poll() {
            return Some(slot);
        }
        let slot = self
            .made
            .try_update(Ordering::AcqRel, Ordering::Acquire, |n| {
                (n < self.max).then_some(n + 1)
            })
            .ok()?;
        self.install(locate(slot).0);
        Some(slot)
    }

    /// Makes chunk `chunk` if it does not exist: whoever loses the race drops its own.
    fn install(&self, chunk: usize) {
        let slot = &self.chunks[chunk];
        if !slot.load(Ordering::Acquire).is_null() {
            return;
        }
        let (_, len) = chunk_span(chunk);
        let cells: Box<[SlotCell<T>]> = (0..len).map(|_| SlotCell::new()).collect();
        let ptr = Box::into_raw(cells).cast::<SlotCell<T>>();
        if slot
            .compare_exchange(
                std::ptr::null_mut(),
                ptr,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_err()
        {
            // SAFETY: made just above with `len` cells and shared with no one.
            drop(unsafe { Box::from_raw(std::ptr::slice_from_raw_parts_mut(ptr, len as usize)) });
        }
    }
}

impl<T> Slab<T> {
    /// The slot is free again; its word is idle.
    pub(super) fn free(&self, slot: u32) {
        let queued = self.freed.offer(slot);
        debug_assert!(queued.is_ok(), "the unbounded queue refused a slot");
    }

    /// Counts a slot as made without making its chunk, as a thread that was paused between the
    /// two does.
    #[cfg(test)]
    pub(super) fn reserve_without_chunk(&self) {
        self.made.fetch_add(1, Ordering::AcqRel);
    }

    /// Slots made so far (some may not be readable yet).
    pub(super) fn made(&self) -> u32 {
        self.made.load(Ordering::Acquire).min(self.max)
    }

    /// The cell of `slot`, if it was made.
    pub(super) fn cell(&self, slot: u32) -> Option<&SlotCell<T>> {
        if slot >= self.max {
            return None;
        }
        let (chunk, at) = locate(slot);
        let base = self.chunks[chunk].load(Ordering::Acquire);
        if base.is_null() {
            return None;
        }
        // SAFETY: the chunk has `chunk_span(chunk).1` cells and `at` is inside it; the chunk is
        // never freed while the slab lives.
        Some(unsafe { &*base.add(at) })
    }

    /// The slot whose record is at `address`, if one is.
    pub(super) fn slot_at(&self, address: usize) -> Option<u32> {
        let size = std::mem::size_of::<SlotCell<T>>();
        let last = locate(self.made().saturating_sub(1)).0;
        for (chunk, base) in self.chunks.iter().enumerate().take(last + 1) {
            let base = base.load(Ordering::Acquire) as usize;
            if base == 0 || address < base {
                continue;
            }
            let (start, len) = chunk_span(chunk);
            let offset = address - base;
            if offset < size * len as usize && offset.is_multiple_of(size) {
                return Some(start + (offset / size) as u32);
            }
        }
        None
    }
}

impl<T> Drop for Slab<T> {
    fn drop(&mut self) {
        for (chunk, base) in self.chunks.iter().enumerate() {
            let base = base.load(Ordering::Acquire);
            if base.is_null() {
                continue;
            }
            let (_, len) = chunk_span(chunk);
            // SAFETY: the chunk has `len` cells and nothing else refers to them (`&mut self`).
            let cells = unsafe { std::slice::from_raw_parts(base, len as usize) };
            // An operation in flight leaks: the kernel may still write into its buffers.
            if cells.iter().all(|cell| cell.load().state() == State::Idle) {
                // SAFETY: made by `Box::into_raw` of a slice of `len` cells, all idle.
                drop(unsafe {
                    Box::from_raw(std::ptr::slice_from_raw_parts_mut(base, len as usize))
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    type Rec = Option<u32>;

    #[test]
    fn slabs_chunks_are_laid_end_to_end() {
        let mut next = 0;
        for chunk in 0..12 {
            let (start, len) = chunk_span(chunk);
            assert_eq!(start, next);
            assert_eq!(locate(start), (chunk, 0));
            assert_eq!(locate(start + len - 1), (chunk, len as usize - 1));
            next = start + len;
        }
    }

    #[test]
    fn the_word_keeps_each_field_apart() {
        let slab: Slab<Rec> = Slab::new(100);
        let slot = slab.alloc().unwrap();
        let cell = slab.cell(slot).unwrap();
        let generation = cell.open(State::Issuing, RouteTag::Ring, 200, true, 9);
        let word = cell.load();
        assert_eq!(word.generation(), generation);
        assert_eq!(
            (word.state(), word.route(), word.ring(), word.cancellable()),
            (State::Issuing, RouteTag::Ring, 200, true)
        );
        assert_eq!((word.end(), word.pinned()), (None, false));
        assert_eq!(cell.serial(), 9);
    }

    #[test]
    fn a_slot_is_reused_with_a_new_generation_and_an_id_that_ended_asks_nothing() {
        let slab: Slab<Rec> = Slab::new(100);
        let slot = slab.alloc().unwrap();
        let cell = slab.cell(slot).unwrap();
        let first = cell.open(State::Issuing, RouteTag::Port, 0, true, 1);
        cell.arm(State::Busy);
        cell.release();
        slab.free(slot);
        assert_eq!(cell.ask_to_end(1, End::Cancelled), Asked::Nothing);
        let again = slab.alloc().unwrap();
        assert_eq!(again, slot);
        let second = cell.open(State::Issuing, RouteTag::Port, 0, true, 2);
        assert_eq!(second, first.wrapping_add(1));
        assert_eq!(
            cell.ask_to_end(1, End::Cancelled),
            Asked::Nothing,
            "an old serial"
        );
        assert_eq!(cell.ask_to_end(2, End::Cancelled), Asked::Recorded);
        assert_eq!(
            cell.ask_to_end(2, End::TimedOut),
            Asked::Nothing,
            "the first reason stays"
        );
        assert_eq!(cell.load().end(), Some(End::Cancelled));
    }

    #[test]
    fn a_cancel_of_a_call_in_flight_pins_the_slot_until_it_is_made() {
        let slab: Slab<Rec> = Slab::new(100);
        let slot = slab.alloc().unwrap();
        let cell = slab.cell(slot).unwrap();
        cell.open(State::Issuing, RouteTag::Port, 0, true, 5);
        assert!(!cell.arm(State::InFlight));
        assert_eq!(cell.ask_to_end(5, End::Cancelled), Asked::Cancel);
        let taken = cell.take(|word| if word.pinned() { Err(()) } else { Ok(()) });
        assert_eq!(taken, Err(()), "a pinned slot is left for the next poll");
        cell.unpin();
        let taken = cell.take(|word| if word.pinned() { Err(()) } else { Ok(()) });
        assert_eq!(taken.unwrap().end(), Some(End::Cancelled));
        assert_eq!(cell.load().state(), State::Busy);
        assert_eq!(cell.reissue(State::Issuing), Err(End::Cancelled));
    }

    #[test]
    fn an_operation_asked_to_end_while_issuing_has_its_call_cancelled_when_armed() {
        let slab: Slab<Rec> = Slab::new(100);
        let slot = slab.alloc().unwrap();
        let cell = slab.cell(slot).unwrap();
        cell.open(State::Issuing, RouteTag::Port, 0, true, 5);
        assert_eq!(cell.ask_to_end(5, End::TimedOut), Asked::Recorded);
        assert!(cell.arm(State::InFlight));
        assert!(cell.load().pinned());
        cell.unpin();
        assert!(!cell.load().pinned());
    }

    #[test]
    fn an_address_maps_to_its_slot_and_a_stranger_to_none() {
        let slab: Slab<Rec> = Slab::new(10_000);
        let slots: Vec<u32> = (0..200).map(|_| slab.alloc().unwrap()).collect();
        for &slot in &slots {
            let cell = slab.cell(slot).unwrap();
            assert_eq!(slab.slot_at(cell.rec() as usize), Some(slot));
            assert_eq!(slab.slot_at(cell.rec() as usize + 1), None);
        }
        assert_eq!(slab.slot_at(8), None);
        let stranger = 5u32;
        assert_eq!(slab.slot_at(&raw const stranger as usize), None);
    }

    #[test]
    fn slots_given_out_from_many_threads_are_each_held_once() {
        let slab: Slab<Rec> = Slab::new(10_000);
        let rounds = if cfg!(miri) { 20 } else { 300 };
        let got: Vec<Vec<u32>> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..4)
                .map(|_| {
                    scope.spawn(|| {
                        let mut mine = Vec::new();
                        for _ in 0..rounds {
                            let slot = slab.alloc().unwrap();
                            let cell = slab.cell(slot).unwrap();
                            cell.open(State::Issuing, RouteTag::Pool, 0, false, 1);
                            mine.push(slot);
                        }
                        mine
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let all: HashSet<u32> = got.iter().flatten().copied().collect();
        assert_eq!(all.len(), 4 * rounds, "no slot went to two threads");
        for slot in all {
            let cell = slab.cell(slot).unwrap();
            cell.arm(State::Busy);
            cell.release();
        }
    }

    #[test]
    fn a_cancel_racing_the_end_of_the_operation_never_pins_the_next_one() {
        let slab: Slab<Rec> = Slab::new(100);
        let slot = slab.alloc().unwrap();
        let cell = slab.cell(slot).unwrap();
        let rounds = if cfg!(miri) { 10 } else { 200 };
        // Set when the canceller ends, even by a failed assertion: the poller's side then stops
        // waiting for an unpin that will not come.
        let gone = std::sync::atomic::AtomicBool::new(false);
        struct Gone<'a>(&'a std::sync::atomic::AtomicBool);
        impl Drop for Gone<'_> {
            fn drop(&mut self) {
                self.0.store(true, Ordering::Release);
            }
        }
        std::thread::scope(|scope| {
            let canceller = scope.spawn(|| {
                let _gone = Gone(&gone);
                let mut pinned = Vec::new();
                for serial in 1..=rounds {
                    if cell.ask_to_end(serial, End::Cancelled) == Asked::Cancel {
                        // The slot is pinned, so it is still this operation's.
                        assert_eq!(cell.serial(), serial);
                        assert!(cell.load().pinned());
                        cell.unpin();
                        pinned.push(serial);
                    }
                }
                pinned
            });
            for serial in 1..=rounds {
                cell.open(State::Issuing, RouteTag::Port, 0, true, serial);
                cell.arm(State::InFlight);
                // The poller's side: a pinned slot is left, and taken once it is not.
                while cell
                    .take(|word| if word.pinned() { Err(()) } else { Ok(()) })
                    .is_err()
                {
                    if gone.load(Ordering::Acquire) {
                        // The canceller failed or ended with the slot pinned.
                        assert!(!cell.load().pinned(), "a pin was left behind");
                    }
                }
                cell.release();
            }
            canceller.join().unwrap();
        });
    }
}
