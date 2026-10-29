//! The asymmetric fence: readers pay no barrier, the writer pays a heavy one.
//!
//! Dekker's pattern needs a full fence on both sides: a reader stores its announcement and then loads the pointer; a
//! writer stores the pointer and then loads the announcements; each must see the other's store or be seen by it. A
//! reader that pays the fence pays it on the frame's path. The asymmetric form (userspace RCU's "memb" flavour) moves
//! the cost: the reader stores, then only stops the compiler from reordering (`compiler_fence`); the writer, between
//! its store and its loads, forces a full barrier on every running thread of the process:
//!
//! - Linux: `membarrier(MEMBARRIER_CMD_PRIVATE_EXPEDITED)`, after registering the process once with
//!   `MEMBARRIER_CMD_REGISTER_PRIVATE_EXPEDITED`.
//! - Windows: `FlushProcessWriteBuffers`.
//!
//! Elsewhere, and where registration fails, the readers keep the symmetric fence (a sequentially consistent store and
//! fence): there is no equivalent to ask for, and the fallback is correct, only slower for the readers. Under loom
//! and Miri the symmetric form is the one modelled.

#[cfg(not(any(loom, miri)))]
use std::sync::atomic::{AtomicU8, Ordering};

/// Not yet decided.
#[cfg(not(any(loom, miri)))]
const UNKNOWN: u8 = 0;
#[cfg(not(any(loom, miri)))]
const ASYMMETRIC: u8 = 1;
#[cfg(not(any(loom, miri)))]
const SYMMETRIC: u8 = 2;

#[cfg(not(any(loom, miri)))]
static MODE: AtomicU8 = AtomicU8::new(UNKNOWN);

#[cfg(all(
    target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64"),
    not(any(loom, miri))
))]
mod os {
    use std::ffi::c_long;

    unsafe extern "C" {
        fn syscall(number: c_long, ...) -> c_long;
    }

    #[cfg(target_arch = "x86_64")]
    const SYS_MEMBARRIER: c_long = 324;
    #[cfg(target_arch = "aarch64")]
    const SYS_MEMBARRIER: c_long = 283;

    const CMD_PRIVATE_EXPEDITED: c_long = 1 << 3;
    const CMD_REGISTER_PRIVATE_EXPEDITED: c_long = 1 << 4;

    /// Registers the process for expedited private barriers; whether the kernel accepted.
    pub fn register() -> bool {
        // SAFETY: `membarrier` takes integers only and touches no memory of ours.
        unsafe {
            syscall(
                SYS_MEMBARRIER,
                CMD_REGISTER_PRIVATE_EXPEDITED,
                0 as c_long,
                0 as c_long,
            ) == 0
        }
    }

    /// A full barrier on every running thread of the process.
    pub fn heavy() {
        // SAFETY: as in `register`.
        let issued = unsafe {
            syscall(
                SYS_MEMBARRIER,
                CMD_PRIVATE_EXPEDITED,
                0 as c_long,
                0 as c_long,
            )
        } == 0;
        if !issued {
            // The kernel stopped honouring it (a sandbox filter, say): the symmetric fence is the safe answer for
            // this call, though readers that skipped theirs cannot be helped. Registration succeeded, so this is
            // not expected; abort rather than run a protocol with a missing side.
            std::process::abort();
        }
    }
}

#[cfg(all(windows, not(any(loom, miri))))]
mod os {
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn FlushProcessWriteBuffers();
    }

    pub fn register() -> bool {
        true
    }

    pub fn heavy() {
        // SAFETY: the call takes no arguments and only flushes write buffers.
        unsafe { FlushProcessWriteBuffers() };
    }
}

#[cfg(not(any(
    all(
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64"),
        not(any(loom, miri))
    ),
    all(windows, not(any(loom, miri)))
)))]
mod os {
    pub fn register() -> bool {
        false
    }

    pub fn heavy() {}
}

/// Whether readers may skip their fence: the process is registered for heavy barriers. Decided once.
#[cfg(not(any(loom, miri)))]
pub(crate) fn asymmetric() -> bool {
    match MODE.load(Ordering::Relaxed) {
        ASYMMETRIC => true,
        SYMMETRIC => false,
        _ => {
            let supported = os::register();
            MODE.store(
                if supported { ASYMMETRIC } else { SYMMETRIC },
                Ordering::Relaxed,
            );
            supported
        }
    }
}

#[cfg(any(loom, miri))]
pub(crate) fn asymmetric() -> bool {
    false
}

/// The writer's half: a full barrier on every thread that reads asymmetrically. Call it between the store and the
/// loads of the announcements, and only when [`asymmetric`] said yes.
pub(crate) fn heavy() {
    os::heavy();
}

#[cfg(all(test, not(any(loom, miri))))]
mod tests {
    use super::*;

    #[test]
    fn the_mode_is_decided_once_and_the_heavy_barrier_runs_when_asymmetric() {
        let first = asymmetric();
        eprintln!("heavy barriers available: {first}");
        assert_eq!(asymmetric(), first);
        if first {
            for _ in 0..100 {
                heavy();
            }
        }
    }
}
