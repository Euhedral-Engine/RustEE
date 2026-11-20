use std::hint;
use std::thread;
use std::time::Duration;

/// Busy-spins this many checks before yielding.
const SPINS: u32 = 128;
/// Yields until this many checks before parking.
const YIELDS: u32 = 512;
/// How long each park lasts once spinning and yielding have not been enough.
const PARK: Duration = Duration::from_micros(10);

/// Waits while `condition` is true: spins briefly, then yields, then parks in short naps.
pub fn await_while(mut condition: impl FnMut() -> bool) {
    let mut checks: u32 = 0;
    while condition() {
        if checks < SPINS {
            #[allow(clippy::disallowed_methods, reason = "JAVA-BACKOFF")]
            hint::spin_loop();
        } else if checks < YIELDS {
            #[allow(clippy::disallowed_methods, reason = "JAVA-BACKOFF")]
            thread::yield_now();
        } else {
            #[allow(clippy::disallowed_methods, reason = "JAVA-BACKOFF")]
            thread::park_timeout(PARK);
        }
        checks = checks.saturating_add(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
    use std::time::Instant;

    #[test]
    fn returns_at_once_when_the_condition_is_false() {
        let mut checks = 0;
        await_while(|| {
            checks += 1;
            false
        });
        assert_eq!(checks, 1);
    }

    #[test]
    fn checks_until_the_condition_turns_false() {
        let mut remaining = 1_000;
        await_while(|| {
            remaining -= 1;
            remaining > 0
        });
        assert_eq!(remaining, 0);
    }

    #[test]
    fn waits_for_another_thread() {
        // The releaser sets the flag only after the waiter has checked 1,000 times, which is past
        // its spinning and yielding and into its parks; the waiter may not return before that.
        // Both waits end at the deadline, so a wait that never returns fails the test.
        const CHECKS_BEFORE_RELEASE: u32 = 1_000;
        struct Release<'a>(&'a AtomicBool);
        impl Drop for Release<'_> {
            fn drop(&mut self) {
                self.0.store(true, Ordering::Release);
            }
        }
        let deadline = Instant::now() + Duration::from_secs(30);
        let released = AtomicBool::new(false);
        let checks = AtomicU32::new(0);
        thread::scope(|scope| {
            let _release = Release(&released);
            scope.spawn(|| {
                await_while(|| {
                    checks.load(Ordering::Acquire) < CHECKS_BEFORE_RELEASE
                        && Instant::now() < deadline
                });
                released.store(true, Ordering::Release);
            });
            let mut flag_was_set = false;
            await_while(|| {
                checks.fetch_add(1, Ordering::AcqRel);
                flag_was_set = released.load(Ordering::Acquire);
                !flag_was_set && Instant::now() < deadline
            });
            assert!(
                flag_was_set,
                "the wait ended at the deadline, not at the flag"
            );
            assert!(checks.load(Ordering::Acquire) >= CHECKS_BEFORE_RELEASE);
        });
    }
}
