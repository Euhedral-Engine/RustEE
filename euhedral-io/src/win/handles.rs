//! A set of handle values that threads insert into, remove from and look up in without a lock.
//!
//! Open addressing over levels of atomic words: a level is a power-of-two array, probed linearly
//! for a few slots from the handle's hash, and a handle that finds its probe window full in a level
//! goes to the next, twice as large. Levels are added by a compare-and-set and never moved or
//! freed until the set drops, so a reader never meets a freed array. A slot goes empty, then
//! holds a key, then a tombstone, then a key again; a lookup stops at the first empty slot, which
//! is sound because no slot ever returns to empty. A handle inserted by two threads at once may sit
//! in two slots: removal clears every copy.

use std::ptr;
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};

use super::sys::Handle;

const EMPTY: usize = 0;
const TOMBSTONE: usize = 1;
/// The slots a handle looks at in one level.
const PROBE: usize = 8;
const FIRST_LEVEL: usize = 64;
const LEVELS: usize = 40;

pub(super) struct HandleSet {
    levels: [AtomicPtr<Level>; LEVELS],
}

struct Level {
    slots: Box<[AtomicUsize]>,
}

impl Default for HandleSet {
    fn default() -> Self {
        Self {
            levels: std::array::from_fn(|_| AtomicPtr::new(ptr::null_mut())),
        }
    }
}

/// The word a handle is stored as; `None` for the two values that would read as empty or a
/// tombstone (no valid handle has them).
fn key(handle: Handle) -> Option<usize> {
    handle.checked_add(2)
}

fn start(handle: Handle) -> usize {
    (((handle as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)) >> 32) as usize
}

impl HandleSet {
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

    pub(super) fn contains(&self, handle: Handle) -> bool {
        let Some(key) = key(handle) else {
            return false;
        };
        for index in 0..LEVELS {
            let Some(level) = self.level(index) else {
                return false;
            };
            let mask = level.slots.len() - 1;
            for step in 0..PROBE {
                match level.slots[(start(handle) + step) & mask].load(Ordering::Acquire) {
                    EMPTY => break,
                    found if found == key => return true,
                    _ => {}
                }
            }
        }
        false
    }

    pub(super) fn insert(&self, handle: Handle) {
        let Some(key) = key(handle) else {
            return;
        };
        if self.contains(handle) {
            return;
        }
        for index in 0..LEVELS {
            let level = self.level_or_new(index);
            let mask = level.slots.len() - 1;
            for step in 0..PROBE {
                let slot = &level.slots[(start(handle) + step) & mask];
                let mut seen = slot.load(Ordering::Acquire);
                while seen == EMPTY || seen == TOMBSTONE {
                    match slot.compare_exchange(seen, key, Ordering::AcqRel, Ordering::Acquire) {
                        Ok(_) => return,
                        Err(now) => seen = now,
                    }
                }
                if seen == key {
                    return;
                }
            }
        }
    }

    pub(super) fn remove(&self, handle: Handle) {
        let Some(key) = key(handle) else {
            return;
        };
        for index in 0..LEVELS {
            let Some(level) = self.level(index) else {
                return;
            };
            let mask = level.slots.len() - 1;
            for step in 0..PROBE {
                let slot = &level.slots[(start(handle) + step) & mask];
                match slot.load(Ordering::Acquire) {
                    EMPTY => break,
                    found if found == key => {
                        let _ = slot.compare_exchange(
                            key,
                            TOMBSTONE,
                            Ordering::AcqRel,
                            Ordering::Acquire,
                        );
                    }
                    _ => {}
                }
            }
        }
    }
}

impl Drop for HandleSet {
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
    fn a_handle_is_found_after_it_is_inserted_and_not_after_it_is_removed() {
        let set = HandleSet::default();
        assert!(!set.contains(40));
        set.insert(40);
        set.insert(40);
        assert!(set.contains(40) && !set.contains(44));
        set.remove(40);
        assert!(!set.contains(40));
        set.insert(40);
        assert!(set.contains(40), "a reused handle value joins again");
    }

    #[test]
    fn many_handles_spill_into_larger_levels_and_are_all_found() {
        let set = HandleSet::default();
        let count = if cfg!(miri) { 300 } else { 5000 };
        for handle in 0..count {
            set.insert(handle * 4);
        }
        for handle in 0..count {
            assert!(set.contains(handle * 4), "{handle}");
            assert!(!set.contains(handle * 4 + 1));
        }
        for handle in (0..count).step_by(2) {
            set.remove(handle * 4);
        }
        for handle in 0..count {
            assert_eq!(set.contains(handle * 4), handle % 2 == 1, "{handle}");
        }
    }

    #[test]
    fn the_two_values_that_read_as_empty_are_never_members() {
        let set = HandleSet::default();
        set.insert(usize::MAX);
        set.insert(usize::MAX - 1);
        assert!(!set.contains(usize::MAX) && !set.contains(usize::MAX - 1));
        set.remove(usize::MAX);
    }

    #[test]
    fn threads_inserting_and_removing_disjoint_handles_lose_none() {
        let set = HandleSet::default();
        let per = if cfg!(miri) { 40 } else { 2000 };
        std::thread::scope(|scope| {
            for thread in 0..4usize {
                let set = &set;
                scope.spawn(move || {
                    for i in 0..per {
                        let handle = (i * 4 + thread) * 4;
                        set.insert(handle);
                        assert!(set.contains(handle));
                        if i % 3 == 0 {
                            set.remove(handle);
                            assert!(!set.contains(handle));
                        }
                    }
                });
            }
        });
        for thread in 0..4usize {
            for i in 0..per {
                assert_eq!(set.contains((i * 4 + thread) * 4), i % 3 != 0);
            }
        }
    }

    #[test]
    fn threads_inserting_one_handle_leave_one_that_a_removal_clears() {
        let set = HandleSet::default();
        std::thread::scope(|scope| {
            for _ in 0..4 {
                scope.spawn(|| set.insert(77));
            }
        });
        assert!(set.contains(77));
        set.remove(77);
        assert!(!set.contains(77), "every copy is cleared");
    }
}
