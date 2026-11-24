//! What the engine knows of each handle's route, kept without a lock.
//!
//! A handle is `Joining` the port (one thread is making the system call), `Joined` to it, or
//! served by a `Ring`; a handle the engine has not seen has no state. The state changes only by a
//! compare-and-set, so of any threads that race to give a handle its first state exactly one wins,
//! and the others read the winner's.
//!
//! Open addressing over levels of atomic words: a level is a power-of-two array, probed linearly
//! for a few slots from the handle's hash, and a handle that finds its probe window full in a level
//! goes to the next, twice as large. Levels are added by a compare-and-set and never moved or
//! freed until the set drops, so a reader never meets a freed array. A word is the handle and its
//! state; the state `NONE` is a handle that was seen and has none now. A slot goes empty, then
//! belongs to one handle value for good: it is never given to another and never goes back to empty.
//! That makes a handle's slot unique (the first slot in probe order that is empty or its own), so
//! two threads never hold two states for one handle, and a lookup stops at the first empty slot.
//! The set holds one slot per handle value ever seen: the OS reuses the values of closed handles.

use std::ptr;
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};

use super::sys::Handle;

const EMPTY: usize = 0;
const STATE_BITS: u32 = 2;
const STATE_MASK: usize = (1 << STATE_BITS) - 1;
/// The slots a handle looks at in one level.
const PROBE: usize = 8;
const FIRST_LEVEL: usize = 64;
const LEVELS: usize = 40;

/// What a handle's route is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub(super) enum RouteState {
    /// One thread is joining the handle to the port: it is the port's, or refused, when it ends.
    Joining = 1,
    /// The handle is on the engine's port.
    Joined = 2,
    /// A ring serves the handle: it is never joined to the port.
    Ring = 3,
}

impl RouteState {
    fn from_bits(bits: usize) -> Option<Self> {
        match bits {
            1 => Some(Self::Joining),
            2 => Some(Self::Joined),
            3 => Some(Self::Ring),
            _ => None,
        }
    }
}

/// What a claim of a handle's route found.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Claim {
    /// The handle had no state: it has the one asked for now, and the caller owns that claim.
    Won,
    /// The handle already had this state.
    Held(RouteState),
    /// The set is out of room (it cannot be, short of 2^45 handle values).
    Full,
}

pub(super) struct HandleStates {
    levels: [AtomicPtr<Level>; LEVELS],
}

struct Level {
    slots: Box<[AtomicUsize]>,
}

impl Default for HandleStates {
    fn default() -> Self {
        Self {
            levels: std::array::from_fn(|_| AtomicPtr::new(ptr::null_mut())),
        }
    }
}

/// The handle as a word with no state; `None` for a value that does not fit.
fn owner(handle: Handle) -> Option<usize> {
    handle
        .checked_add(1)
        .filter(|value| *value <= usize::MAX >> STATE_BITS)
        .map(|value| value << STATE_BITS)
}

fn start(handle: Handle) -> usize {
    (((handle as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)) >> 32) as usize
}

impl HandleStates {
    fn level(&self, index: usize) -> Option<&Level> {
        // SAFETY: a level is set once from a live box and freed only when the set drops.
        unsafe { self.levels[index].load(Ordering::Acquire).as_ref() }
    }

    fn level_or_new(&self, index: usize) -> &Level {
        if let Some(level) = self.level(index) {
            return level;
        }
        let size = FIRST_LEVEL << index;
        let fresh = Box::into_raw(Box::new(Level {
            slots: (0..size).map(|_| AtomicUsize::new(EMPTY)).collect(),
        }));
        match self.levels[index].compare_exchange(
            ptr::null_mut(),
            fresh,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            // SAFETY: just set, and never freed before the set drops.
            Ok(_) => unsafe { &*fresh },
            Err(set) => {
                // SAFETY: `fresh` was never shared; `set` is a live level.
                unsafe {
                    drop(Box::from_raw(fresh));
                    &*set
                }
            }
        }
    }

    /// The handle's slot: the first in probe order that is its own. With `create`, an empty slot
    /// met first becomes its own; without, it ends the search.
    fn slot(&self, handle: Handle, create: bool) -> Option<&AtomicUsize> {
        let owner = owner(handle)?;
        for index in 0..LEVELS {
            let level = if create {
                self.level_or_new(index)
            } else {
                self.level(index)?
            };
            let mask = level.slots.len() - 1;
            for step in 0..PROBE {
                let slot = &level.slots[(start(handle) + step) & mask];
                let mut word = slot.load(Ordering::Acquire);
                loop {
                    if word == EMPTY {
                        if !create {
                            return None;
                        }
                        match slot.compare_exchange(
                            EMPTY,
                            owner,
                            Ordering::AcqRel,
                            Ordering::Acquire,
                        ) {
                            Ok(_) => return Some(slot),
                            Err(now) => {
                                word = now;
                                continue;
                            }
                        }
                    }
                    if word & !STATE_MASK == owner {
                        return Some(slot);
                    }
                    break;
                }
            }
        }
        None
    }

    /// The handle's state, if it has one.
    pub(super) fn get(&self, handle: Handle) -> Option<RouteState> {
        let slot = self.slot(handle, false)?;
        RouteState::from_bits(slot.load(Ordering::Acquire) & STATE_MASK)
    }

    /// Gives the handle `state` if it has none. Of racing claims exactly one is `Won`; the others
    /// are `Held` with the winner's state, whatever the state asked for.
    pub(super) fn claim(&self, handle: Handle, state: RouteState) -> Claim {
        let Some(slot) = self.slot(handle, true) else {
            return Claim::Full;
        };
        let mut word = slot.load(Ordering::Acquire);
        loop {
            if let Some(held) = RouteState::from_bits(word & STATE_MASK) {
                return Claim::Held(held);
            }
            match slot.compare_exchange(
                word,
                word | state as usize,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return Claim::Won,
                Err(now) => word = now,
            }
        }
    }

    /// Moves the handle from `from` to `to`; false if it was not in `from`.
    pub(super) fn advance(&self, handle: Handle, from: RouteState, to: RouteState) -> bool {
        let Some(slot) = self.slot(handle, false) else {
            return false;
        };
        let Some(owner) = owner(handle) else {
            return false;
        };
        slot.compare_exchange(
            owner | from as usize,
            owner | to as usize,
            Ordering::AcqRel,
            Ordering::Acquire,
        )
        .is_ok()
    }

    /// Takes the state off the handle if it is in `from`; false if it was not.
    pub(super) fn clear_if(&self, handle: Handle, from: RouteState) -> bool {
        let Some(slot) = self.slot(handle, false) else {
            return false;
        };
        let Some(owner) = owner(handle) else {
            return false;
        };
        slot.compare_exchange(
            owner | from as usize,
            owner,
            Ordering::AcqRel,
            Ordering::Acquire,
        )
        .is_ok()
    }

    /// Takes any state off the handle.
    pub(super) fn clear(&self, handle: Handle) {
        let Some(slot) = self.slot(handle, false) else {
            return;
        };
        slot.fetch_and(!STATE_MASK, Ordering::AcqRel);
    }
}

impl Drop for HandleStates {
    fn drop(&mut self) {
        for level in &mut self.levels {
            let level = *level.get_mut();
            if !level.is_null() {
                // SAFETY: set from a box, once, and not shared any more.
                drop(unsafe { Box::from_raw(level) });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_handle_has_the_state_that_claimed_it_until_it_is_cleared() {
        let set = HandleStates::default();
        assert_eq!(set.get(40), None);
        assert_eq!(set.claim(40, RouteState::Joining), Claim::Won);
        assert_eq!(
            set.claim(40, RouteState::Ring),
            Claim::Held(RouteState::Joining)
        );
        assert_eq!(set.get(40), Some(RouteState::Joining));
        assert_eq!(set.get(44), None);
        assert!(!set.advance(40, RouteState::Ring, RouteState::Joined));
        assert!(set.advance(40, RouteState::Joining, RouteState::Joined));
        assert!(!set.clear_if(40, RouteState::Joining));
        assert_eq!(set.get(40), Some(RouteState::Joined));
        set.clear(40);
        assert_eq!(set.get(40), None);
        assert_eq!(
            set.claim(40, RouteState::Ring),
            Claim::Won,
            "a reused handle value is claimed again"
        );
        assert!(set.clear_if(40, RouteState::Ring));
        assert_eq!(set.get(40), None);
    }

    #[test]
    fn many_handles_spill_into_larger_levels_and_are_all_found() {
        let set = HandleStates::default();
        let count = if cfg!(miri) { 300 } else { 5000 };
        for handle in 0..count {
            assert_eq!(set.claim(handle * 4, RouteState::Joined), Claim::Won);
        }
        for handle in 0..count {
            assert_eq!(set.get(handle * 4), Some(RouteState::Joined), "{handle}");
            assert_eq!(set.get(handle * 4 + 1), None);
        }
        for handle in (0..count).step_by(2) {
            set.clear(handle * 4);
        }
        for handle in 0..count {
            assert_eq!(set.get(handle * 4).is_some(), handle % 2 == 1, "{handle}");
        }
    }

    #[test]
    fn a_value_that_does_not_fit_never_has_a_state() {
        let set = HandleStates::default();
        assert_eq!(set.claim(usize::MAX, RouteState::Joined), Claim::Full);
        assert_eq!(set.get(usize::MAX), None);
        set.clear(usize::MAX);
    }

    #[test]
    fn threads_claiming_disjoint_handles_lose_none() {
        let set = HandleStates::default();
        let per = if cfg!(miri) { 40 } else { 2000 };
        std::thread::scope(|scope| {
            for thread in 0..4usize {
                let set = &set;
                scope.spawn(move || {
                    for i in 0..per {
                        let handle = (i * 4 + thread) * 4;
                        assert_eq!(set.claim(handle, RouteState::Joined), Claim::Won);
                        assert_eq!(set.get(handle), Some(RouteState::Joined));
                        if i % 3 == 0 {
                            set.clear(handle);
                            assert_eq!(set.get(handle), None);
                        }
                    }
                });
            }
        });
        for thread in 0..4usize {
            for i in 0..per {
                assert_eq!(
                    set.get((i * 4 + thread) * 4).is_some(),
                    i % 3 != 0,
                    "{thread} {i}"
                );
            }
        }
    }

    #[test]
    fn of_threads_claiming_one_handle_exactly_one_wins_each_round() {
        let rounds = if cfg!(miri) { 20 } else { 500 };
        for round in 0..rounds {
            let set = HandleStates::default();
            // Other handles around it, some cleared, so that its probe window is not clean.
            for other in 0..20 {
                set.claim(other * 4, RouteState::Joined);
                if other % 2 == 0 {
                    set.clear(other * 4);
                }
            }
            let handle = 4 * (round % 20);
            let won: usize = std::thread::scope(|scope| {
                let workers: Vec<_> = (0..4)
                    .map(|thread| {
                        let set = &set;
                        scope.spawn(move || {
                            let state = if thread % 2 == 0 {
                                RouteState::Joining
                            } else {
                                RouteState::Ring
                            };
                            usize::from(set.claim(handle, state) == Claim::Won)
                        })
                    })
                    .collect();
                workers.into_iter().map(|w| w.join().unwrap()).sum()
            });
            let was_held = round % 20 % 2 == 1;
            assert_eq!(won, usize::from(!was_held), "round {round}");
        }
    }
}
