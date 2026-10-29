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
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
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
        let released = Arc::new(AtomicBool::new(false));
        let releaser = {
            let released = released.clone();
            thread::spawn(move || {
                thread::sleep(Duration::from_millis(20));
                released.store(true, Ordering::Release);
            })
        };
        let start = Instant::now();
        await_while(|| !released.load(Ordering::Acquire));
        assert!(start.elapsed() >= Duration::from_millis(20));
        releaser.join().unwrap();
    }
}
