//! The marker ring of a lane: a lock-free protocol for any number of threads to make markers on one stream, order
//! behind them from other lanes, and learn which completed. It touches no driver; the backend does the recording and
//! the querying through closures.
//!
//! A lane's markers are numbered 1, 2, 3, ...; marker `v` is recorded in slot `v % n` of a ring of `n` events. The
//! state any thread may touch is atomic, and only that state:
//!
//! - **`claimed`** is the highest number handed out. A marker's number is claimed by compare-and-set, after checking
//!   that the slot's previous record (marker `v - n`) completed, so no two threads ever share a slot.
//! - **`recorded[slot]`** is set, with release ordering, once the claimant's record call returned: `(v << 1) | skipped`.
//!   A claimant that gives its number up (the driver refused the record, or the claim was dropped before the work it
//!   was for) stores it as **skipped**, a hole nobody was given.
//! - **`done`** is the highest number known complete, advanced by whoever scans. A scan walks the numbers after `done`
//!   in order and stops at the first it cannot vouch for: a number whose entry is not stored yet (its claimant is still
//!   inside the record call, and the slot holds an older number's record, which must not be mistaken for it), or one
//!   whose event is not complete. So a number is never counted complete ahead of one before it, and no thread waits for
//!   another: a scan that finds a claimant mid-record stops, and the next scan goes on.
//!
//! **A refusal comes before the work.** A number is claimed ([`MarkerRing::reserve`]) *ahead* of the work it follows, as
//! Java opens a stage's marker before the stage submits, and the only refusal (`Busy`: the ring is full, or an awaiter
//! holds the slot) is answered then. Recording the marker afterwards ([`Claim::record`]) is never refused, so a stage
//! that is told to retry has submitted nothing. A number claimed and not yet recorded holds up the completion count of the
//! numbers after it, and no more.
//!
//! A marker records *after all the work submitted so far on the stream*, which may include another thread's work: that
//! only over-waits. The driver serializes the calls it gets on one stream; the ring keeps their numbers consistent.
//!
//! Awaiters meet the claimants in a Dekker handshake (see [`SlotPins`]): an awaiter that found marker `v` incomplete
//! must enqueue its wait on the slot while it still holds `v`'s record.

#![cfg_attr(
    not(any(feature = "cuda", test)),
    allow(dead_code, reason = "the CUDA backend uses the ring")
)]

use crate::GpuError;
use crate::sync::{AtomicU32, AtomicU64, Ordering};

/// What a query of one marker's event found.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Query {
    /// The marker's work completed.
    Done,
    /// It is still to run.
    Pending,
    /// The query failed with this backend code: the walk stops.
    Failed(i32),
}

/// Keeps a marker slot from being re-recorded under an awaiter, without a lock.
///
/// Value `v` is recorded in slot `v % n`; a claimant may record `v + n` there once `v` is complete. An awaiter of `v`
/// that has found `v` incomplete must enqueue its wait on the slot's event *while the slot still holds `v`'s record*,
/// or it would wait for the later value `v + n`, which can deadlock. The two sides meet in a Dekker handshake of
/// sequentially consistent operations:
/// - a **claimant** announces the value it is about to record (`reserved = max(reserved, w)`) and then reads the slot's
///   pin count: a pinned slot is not recorded over until the pin drops;
/// - an **awaiter** raises the slot's pin count and then reads `reserved`: if a claimant is already past `v`
///   (`reserved >= v + n`) the slot is being (or has been) re-recorded, which happens only when `v` is complete, so
///   there is nothing to wait for and the pin is dropped.
///
/// Whichever of the two reads comes second sees the other's write, so an awaiter that keeps its pin always has the slot
/// as `v` left it, until it lowers the pin after its wait is enqueued. A pin is held for one driver call.
#[derive(Debug)]
struct SlotPins {
    pins: Box<[AtomicU32]>,
    reserved: AtomicU64,
}

/// A raised pin on a marker slot; lowers it on drop.
pub(crate) struct Pin<'a>(&'a AtomicU32);

impl Drop for Pin<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

/// See the [module documentation](self).
#[derive(Debug)]
pub(crate) struct MarkerRing {
    claimed: AtomicU64,
    done: AtomicU64,
    /// Per slot: `(v << 1) | skipped` of the number whose record call last finished there.
    recorded: Box<[AtomicU64]>,
    pins: SlotPins,
}

impl MarkerRing {
    /// A ring of `slots` markers (at least one).
    pub(crate) fn new(slots: usize) -> Self {
        assert!(slots > 0, "a ring has a slot");
        Self {
            claimed: AtomicU64::new(0),
            done: AtomicU64::new(0),
            recorded: (0..slots).map(|_| AtomicU64::new(0)).collect(),
            pins: SlotPins {
                pins: (0..slots).map(|_| AtomicU32::new(0)).collect(),
                reserved: AtomicU64::new(0),
            },
        }
    }

    pub(crate) fn slots(&self) -> u64 {
        self.recorded.len() as u64
    }

    /// The slot marker `value` is recorded in.
    pub(crate) fn slot(&self, value: u64) -> usize {
        (value % self.slots()) as usize
    }

    /// The highest number claimed: the markers issued, some perhaps still being recorded.
    pub(crate) fn claimed(&self) -> u64 {
        self.claimed.load(Ordering::Acquire)
    }

    /// The highest number known complete.
    pub(crate) fn done(&self) -> u64 {
        self.done.load(Ordering::Acquire)
    }

    /// Claims the number of the next marker, ahead of the work it will follow: `Err(Busy)` (retry; this never waits for
    /// the device) when every slot holds a marker that has not completed, or an awaiter holds the next slot. `scan` is
    /// called once if the ring looks full, to learn of completions before refusing.
    ///
    /// A refusal is the only failure, and it comes before the caller submits anything: a [`Claim`] is recorded
    /// afterwards ([`Claim::record`]) and cannot be refused. Dropping a claim unrecorded gives its number up as a hole.
    pub(crate) fn reserve(&self, scan: impl FnOnce()) -> Result<Claim<'_>, GpuError> {
        let n = self.slots();
        let mut scan = Some(scan);
        let value = loop {
            let claimed = self.claimed.load(Ordering::Acquire);
            let value = claimed + 1;
            if value > n && value - n > self.done.load(Ordering::Acquire) {
                if let Some(scan) = scan.take() {
                    scan();
                }
                if value - n > self.done.load(Ordering::Acquire) {
                    return Err(GpuError::Busy);
                }
            }
            // An awaiter that holds the slot: the claim is refused now, before the caller's work, not after it.
            if self.pins.pins[self.slot(value)].load(Ordering::SeqCst) != 0 {
                return Err(GpuError::Busy);
            }
            if self
                .claimed
                .compare_exchange(claimed, value, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                break value;
            }
        };
        // The slot's previous marker completed (checked above): an awaiter that comes now sees this and waits for nothing.
        self.pins.reserved.fetch_max(value, Ordering::SeqCst);
        Ok(Claim {
            ring: self,
            value,
            recorded: false,
        })
    }

    /// Makes the next marker: [`reserve`](Self::reserve) and [`record`](Claim::record) in one step.
    #[cfg(test)]
    pub(crate) fn mark(
        &self,
        scan: impl FnOnce(),
        record: impl FnOnce(usize) -> Result<(), GpuError>,
    ) -> Result<u64, GpuError> {
        self.reserve(scan)?.record(record)
    }

    /// Stores number `value`'s entry: from here a scan can vouch for it.
    fn store_entry(&self, value: u64, skipped: bool) {
        self.recorded[self.slot(value)].store((value << 1) | u64::from(skipped), Ordering::Release);
    }

    /// Advances `done` over the markers that completed, asking `query(slot)` about each in order. Returns the code of a
    /// failed query, which stopped the walk.
    pub(crate) fn scan(&self, mut query: impl FnMut(usize) -> Query) -> Option<i32> {
        let claimed = self.claimed.load(Ordering::Acquire);
        let mut done = self.done.load(Ordering::Acquire);
        let mut failed = None;
        while done < claimed {
            let value = done + 1;
            let raw = self.recorded[self.slot(value)].load(Ordering::Acquire);
            let (entry, skipped) = (raw >> 1, raw & 1 == 1);
            // An entry of an older number: the claimant of `value` is still recording, and the event holds the older
            // number's record. Stop; the next scan goes on.
            if entry < value {
                break;
            }
            // An entry of a later number: the slot was claimed again, which happens only once `value` completed. A
            // skipped number has no event.
            if entry > value || skipped {
                done = value;
                continue;
            }
            match query(self.slot(value)) {
                Query::Done => done = value,
                Query::Pending => break,
                Query::Failed(code) => {
                    failed = Some(code);
                    break;
                }
            }
        }
        self.done.fetch_max(done, Ordering::AcqRel);
        failed
    }

    /// An awaiter of marker `value`, which it found incomplete. `Some(pin)`: the slot holds `value`'s record until the
    /// pin drops. `None`: a claimant has started re-recording the slot, so `value` is complete and needs no wait.
    pub(crate) fn pin(&self, value: u64) -> Option<Pin<'_>> {
        let slot = &self.pins.pins[self.slot(value)];
        slot.fetch_add(1, Ordering::SeqCst);
        if self.pins.reserved.load(Ordering::SeqCst) >= value + self.slots() {
            slot.fetch_sub(1, Ordering::SeqCst);
            return None;
        }
        Some(Pin(slot))
    }
}

/// A marker number claimed ahead of the work it will follow (Java's marker opened before the stage submits). Record it
/// once the work is in; dropped without recording, it is given up as a hole nobody was handed.
#[derive(Debug)]
pub(crate) struct Claim<'a> {
    ring: &'a MarkerRing,
    value: u64,
    recorded: bool,
}

impl Claim<'_> {
    #[cfg(test)]
    pub(crate) fn value(&self) -> u64 {
        self.value
    }

    /// Has `record(slot)` record the marker's event after the work submitted so far, and stores its entry. Never
    /// refused: a pin an awaiter placed before the claim is waited out (it is held for one driver call). An `Err` from
    /// `record` gives the number up and is returned.
    pub(crate) fn record(
        mut self,
        record: impl FnOnce(usize) -> Result<(), GpuError>,
    ) -> Result<u64, GpuError> {
        let ring = self.ring;
        let slot = ring.slot(self.value);
        while ring.pins.pins[slot].load(Ordering::SeqCst) != 0 {
            #[allow(
                clippy::disallowed_methods,
                reason = "JAVA-BACKOFF: the wait is for an awaiter inside one driver call, as Java's `SpinWait` waits for a peer's short step"
            )]
            std::hint::spin_loop();
        }
        self.recorded = true;
        match record(slot) {
            Ok(()) => {
                ring.store_entry(self.value, false);
                Ok(self.value)
            }
            Err(error) => {
                ring.store_entry(self.value, true);
                Err(error)
            }
        }
    }
}

impl Drop for Claim<'_> {
    fn drop(&mut self) {
        if !self.recorded {
            self.ring.store_entry(self.value, true);
        }
    }
}

#[cfg(all(test, not(loom)))]
mod tests {
    use super::*;

    fn ok(_: usize) -> Result<(), GpuError> {
        Ok(())
    }

    #[test]
    fn markers_are_numbered_in_order_and_complete_in_order() {
        let ring = MarkerRing::new(4);
        for expected in 1..=3 {
            assert_eq!(ring.mark(|| {}, ok).unwrap(), expected);
        }
        assert_eq!((ring.claimed(), ring.done()), (3, 0));
        assert_eq!(ring.scan(|_| Query::Done), None);
        assert_eq!(ring.done(), 3);
    }

    #[test]
    fn a_full_ring_is_busy_until_a_scan_finds_a_completion() {
        let ring = MarkerRing::new(2);
        ring.mark(|| {}, ok).unwrap();
        ring.mark(|| {}, ok).unwrap();
        let mut scanned = 0;
        let refused = ring.mark(|| scanned += 1, ok);
        assert_eq!(refused, Err(GpuError::Busy));
        assert_eq!(scanned, 1, "it looked once before it refused");
        ring.scan(|slot| {
            if slot == 1 {
                Query::Done
            } else {
                Query::Pending
            }
        });
        assert_eq!(
            ring.done(),
            1,
            "the walk stops at the first marker still to run"
        );
        assert_eq!(
            ring.mark(|| {}, ok).unwrap(),
            3,
            "marker 1's slot is free again"
        );
    }

    #[test]
    fn a_walk_reports_a_failed_query_and_stops() {
        let ring = MarkerRing::new(4);
        for _ in 0..3 {
            ring.mark(|| {}, ok).unwrap();
        }
        let mut seen = 0;
        let failed = ring.scan(|_| {
            seen += 1;
            if seen == 2 {
                Query::Failed(101)
            } else {
                Query::Done
            }
        });
        assert_eq!((failed, ring.done()), (Some(101), 1));
    }

    #[test]
    fn a_refused_record_skips_its_number_and_the_ring_goes_on() {
        let ring = MarkerRing::new(4);
        ring.mark(|| {}, ok).unwrap();
        let refused = ring.mark(|| {}, |_| Err(GpuError::Invalid("refused".into())));
        assert!(matches!(refused, Err(GpuError::Invalid(_))));
        assert_eq!(ring.mark(|| {}, ok).unwrap(), 3);
        // The skipped number has no event: it is complete once the one before it is.
        let mut queried = Vec::new();
        ring.scan(|slot| {
            queried.push(slot);
            Query::Done
        });
        assert_eq!(queried, [1, 3], "no query for the skipped number 2");
        assert_eq!(ring.done(), 3);
    }

    #[test]
    fn a_pinned_slot_is_refused_before_a_number_is_claimed() {
        let ring = MarkerRing::new(1);
        ring.mark(|| {}, ok).unwrap();
        let pin = ring.pin(1).expect("nothing past marker 1 yet");
        ring.scan(|_| Query::Done);
        // Marker 2 reuses the only slot, which an awaiter holds: refused at the claim, nothing claimed.
        assert!(matches!(ring.reserve(|| {}), Err(GpuError::Busy)));
        assert_eq!(ring.claimed(), 1);
        drop(pin);
        assert_eq!(
            ring.mark(|| {}, ok).unwrap(),
            2,
            "the same number, once the pin dropped"
        );
    }

    #[test]
    fn a_claim_is_recorded_after_the_claims_that_followed_it_and_nothing_is_vouched_for_early() {
        let ring = MarkerRing::new(4);
        let first = ring.reserve(|| {}).unwrap();
        let second = ring.reserve(|| {}).unwrap();
        assert_eq!((first.value(), second.value()), (1, 2));
        second.record(ok).unwrap();
        let mut asked = Vec::new();
        ring.scan(|slot| {
            asked.push(slot);
            Query::Done
        });
        assert!(
            asked.is_empty(),
            "number 1 is not recorded: the scan stops before it"
        );
        assert_eq!(ring.done(), 0);
        first.record(ok).unwrap();
        ring.scan(|_| Query::Done);
        assert_eq!(ring.done(), 2);
    }

    #[test]
    fn a_claim_dropped_before_its_work_is_a_hole_that_completes_without_an_event() {
        let ring = MarkerRing::new(4);
        drop(ring.reserve(|| {}).unwrap());
        assert_eq!(ring.mark(|| {}, ok).unwrap(), 2);
        let mut asked = Vec::new();
        ring.scan(|slot| {
            asked.push(slot);
            Query::Done
        });
        assert_eq!(asked, [2], "no event for the hole");
        assert_eq!(ring.done(), 2);
    }

    #[test]
    fn busy_is_the_claims_alone() {
        let ring = MarkerRing::new(1);
        let claim = ring.reserve(|| {}).unwrap();
        assert_eq!(claim.record(ok), Ok(1), "a claim records");
        assert!(
            matches!(ring.reserve(|| {}), Err(GpuError::Busy)),
            "the next is refused, before any work"
        );
    }

    #[test]
    fn an_awaiter_that_arrives_after_the_claimant_moved_on_skips_the_wait() {
        let ring = MarkerRing::new(2);
        ring.mark(|| {}, ok).unwrap();
        ring.mark(|| {}, ok).unwrap();
        ring.scan(|_| Query::Done);
        // Marker 3 claims marker 1's slot.
        ring.mark(|| {}, ok).unwrap();
        assert!(
            ring.pin(1).is_none(),
            "the slot no longer holds marker 1, which is complete: no wait"
        );
        assert_eq!(
            ring.mark(|| {}, ok).unwrap(),
            4,
            "the refused pin left nothing raised"
        );
    }

    /// A claimant re-records two slots as fast as it can while an awaiter pins the latest marker it was given: a pin that
    /// is granted always finds the slot as its marker left it, for as long as the pin is up.
    #[test]
    #[allow(
        clippy::disallowed_methods,
        reason = "two threads stand in for a lane's submitter and an awaiter on another lane"
    )]
    fn a_granted_pin_always_sees_its_own_marker_in_the_slot() {
        use std::sync::atomic::AtomicBool;
        const SLOTS: u64 = 2;
        let ring = MarkerRing::new(SLOTS as usize);
        let in_slot: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];
        let given = AtomicU64::new(0);
        let stop = AtomicBool::new(false);
        let granted = AtomicU64::new(0);
        let rounds = if cfg!(miri) { 200 } else { 200_000 };
        std::thread::scope(|scope| {
            scope.spawn(|| {
                let mut made = 0;
                while made < rounds {
                    ring.scan(|_| Query::Done);
                    if let Ok(value) = ring.mark(
                        || {},
                        |slot| {
                            // The number is in the slot only once it is recorded: set it as the record call does.
                            let value = ring.claimed();
                            in_slot[slot].store(value, Ordering::SeqCst);
                            Ok(())
                        },
                    ) {
                        given.store(value, Ordering::SeqCst);
                        made += 1;
                    }
                }
                stop.store(true, Ordering::SeqCst);
            });
            scope.spawn(|| {
                while !stop.load(Ordering::SeqCst) {
                    let value = given.load(Ordering::SeqCst);
                    if value == 0 {
                        continue;
                    }
                    if let Some(_pin) = ring.pin(value) {
                        for _ in 0..4 {
                            assert_eq!(
                                in_slot[ring.slot(value)].load(Ordering::SeqCst),
                                value,
                                "the slot was re-recorded under a pin"
                            );
                        }
                        granted.fetch_add(1, Ordering::Relaxed);
                    }
                }
            });
        });
        assert!(
            granted.load(Ordering::Relaxed) > 0,
            "some pins were granted"
        );
    }

    /// Eight threads make markers on a ring of 16 while another scans: every number is handed out once, each slot is
    /// recorded by one thread at a time, and every number completes.
    #[test]
    #[allow(
        clippy::disallowed_methods,
        reason = "threads stand in for the workers that submit to one lane"
    )]
    fn many_threads_make_distinct_markers_and_all_complete() {
        use std::sync::atomic::AtomicBool;
        const SLOTS: usize = 16;
        const PER_THREAD: u64 = if cfg!(miri) { 20 } else { 5_000 };
        let ring = MarkerRing::new(SLOTS);
        let in_slot: Vec<AtomicU32> = (0..SLOTS).map(|_| AtomicU32::new(0)).collect();
        let seen: Vec<AtomicU32> = (0..8 * PER_THREAD as usize + 1)
            .map(|_| AtomicU32::new(0))
            .collect();
        let stop = AtomicBool::new(false);
        std::thread::scope(|scope| {
            scope.spawn(|| {
                while !stop.load(Ordering::Acquire) {
                    ring.scan(|_| Query::Done);
                }
            });
            let workers: Vec<_> = (0..8)
                .map(|_| {
                    scope.spawn(|| {
                        for _ in 0..PER_THREAD {
                            let value = loop {
                                let made = ring.mark(
                                    || {
                                        ring.scan(|_| Query::Done);
                                    },
                                    |slot| {
                                        assert_eq!(
                                            in_slot[slot].fetch_add(1, Ordering::SeqCst),
                                            0,
                                            "two threads record one slot at once"
                                        );
                                        in_slot[slot].fetch_sub(1, Ordering::SeqCst);
                                        Ok(())
                                    },
                                );
                                match made {
                                    Ok(value) => break value,
                                    Err(GpuError::Busy) => {}
                                    Err(other) => panic!("{other}"),
                                }
                            };
                            seen[value as usize].fetch_add(1, Ordering::Relaxed);
                        }
                    })
                })
                .collect();
            for worker in workers {
                worker.join().unwrap();
            }
            stop.store(true, Ordering::Release);
        });
        ring.scan(|_| Query::Done);
        let made = (1..seen.len())
            .filter(|&v| seen[v].load(Ordering::Relaxed) == 1)
            .count() as u64;
        assert!(
            (1..seen.len()).all(|v| seen[v].load(Ordering::Relaxed) <= 1),
            "a number handed out twice"
        );
        assert_eq!(made, 8 * PER_THREAD, "every call ended in a marker");
        assert_eq!(ring.done(), ring.claimed());
    }
}

/// The ring's handshakes under every interleaving the memory model allows.
#[cfg(all(loom, test))]
mod loom_tests {
    use super::*;
    use loom::sync::Arc;
    use loom::sync::atomic::AtomicU64 as LoomCell;
    use loom::thread;

    fn ok(_: usize) -> Result<(), GpuError> {
        Ok(())
    }

    // Two threads make a marker each on a ring of two: they get 1 and 2, whichever records first, and a scan that
    // follows both counts both complete.
    #[test]
    fn loom_two_markers_get_distinct_numbers_and_both_complete() {
        loom::model(|| {
            let ring = Arc::new(MarkerRing::new(2));
            let other = {
                let ring = ring.clone();
                thread::spawn(move || ring.mark(|| {}, ok).unwrap())
            };
            let mine = ring.mark(|| {}, ok).unwrap();
            let theirs = other.join().unwrap();
            assert_eq!(mine.min(theirs), 1);
            assert_eq!(mine.max(theirs), 2);
            ring.scan(|_| Query::Done);
            assert_eq!(ring.done(), 2);
        });
    }

    /// What the record call writes: plain memory, so that the checker reports a read that is not ordered after the write.
    struct Slots([loom::cell::UnsafeCell<u64>; 2]);

    // SAFETY: the ring orders every access; the model exists to check that it does.
    unsafe impl Sync for Slots {}

    // A scan that runs while a marker is being recorded never vouches for a slot whose record call has not returned:
    // what it queries it finds recorded, with the write of the record call ordered before the query. (Both threads
    // make a marker, so that the checker has a conflicting access to interleave the scan with.)
    #[test]
    fn loom_a_scan_never_vouches_for_a_marker_still_being_recorded() {
        loom::model(|| {
            let ring = Arc::new(MarkerRing::new(2));
            let slots = Arc::new(Slots([
                loom::cell::UnsafeCell::new(0),
                loom::cell::UnsafeCell::new(0),
            ]));
            let record = |slots: Arc<Slots>| {
                move |slot: usize| {
                    slots.0[slot].with_mut(|cell| {
                        // SAFETY: the claimant owns the slot until it stores its entry.
                        unsafe { *cell += 1 }
                    });
                    Ok(())
                }
            };
            let maker = {
                let (ring, slots) = (ring.clone(), slots.clone());
                thread::spawn(move || {
                    let _ = ring.mark(|| {}, record(slots));
                })
            };
            let _ = ring.mark(|| {}, record(slots.clone()));
            ring.scan(|slot| {
                // A query reads the slot the record call wrote: ordered after it, or the checker says so.
                let written = slots.0[slot].with(|cell| {
                    // SAFETY: ordered after the write by the entry the scan saw.
                    unsafe { *cell }
                });
                assert!(written > 0, "a query of a slot whose record is not stored");
                Query::Done
            });
            maker.join().unwrap();
        });
    }
}
