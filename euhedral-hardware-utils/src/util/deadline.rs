//! A deadline that a cheap, coarse clock filters.

use std::sync::OnceLock;
use std::time::{Duration, Instant};

/// A point in time on the monotonic clock, with a cheap coarse clock in front of it.
///
/// The answer is exactly [`Instant::now`]'s: [`expired`](Self::expired) is true from the first call
/// that [`Instant::now`] would see at or past the deadline, and never before. The coarse clock only
/// decides, without reading [`Instant::now`], that the deadline is still far off.
///
/// The coarse clock moves in ticks and reads off the monotonic clock by less than one tick in either
/// direction (ahead or behind). The deadline keeps a threshold [`coarse_bound`] (two ticks and more)
/// before itself, so a coarse reading below the threshold proves the deadline has not come. From the
/// threshold on, every call reads [`Instant::now`]: the last few milliseconds before the deadline cost one
/// clock read per call, as they would without the filter. A deadline nearer than the bound is read from
/// [`Instant::now`] throughout.
///
/// Use it for timeouts, patience and polling limits that a loop checks, where most of the wait is
/// far from the deadline.
///
/// The deadline owns its clock, so reading takes `&mut self` and one thread reads it at a time. Make
/// one per wait, on the waiting thread; making one costs about as much as two clock reads.
#[derive(Clone, Debug)]
pub struct Deadline {
    clock: fast_time::Clock,
    /// On the coarse clock's timeline: a coarse reading below it proves the deadline is ahead.
    /// `None` when the deadline is too near the clock's origin to have a threshold.
    filter: Option<fast_time::Instant>,
    exact: Instant,
}

impl Deadline {
    /// A deadline `timeout` from now.
    ///
    /// # Panics
    ///
    /// If `timeout` is so large that the monotonic clock cannot represent the deadline.
    pub fn after(timeout: Duration) -> Self {
        Self::at(Instant::now() + timeout)
    }

    /// A deadline at `exact`, a reading of the monotonic clock.
    pub fn at(exact: Instant) -> Self {
        Self {
            clock: fast_time::Clock::new(),
            filter: Self::filter_for(exact),
            exact,
        }
    }

    fn filter_for(exact: Instant) -> Option<fast_time::Instant> {
        exact.checked_sub(coarse_bound()).map(Into::into)
    }

    /// Moves the deadline to `timeout` from now, keeping the clock: a throttle that arms again
    /// each time it lets one through does not make a clock each time.
    ///
    /// # Panics
    ///
    /// If `timeout` is so large that the monotonic clock cannot represent the deadline.
    pub fn restart(&mut self, timeout: Duration) {
        self.exact = Instant::now() + timeout;
        self.filter = Self::filter_for(self.exact);
    }

    /// The deadline on the monotonic clock, for passing it on.
    pub fn exact(&self) -> Instant {
        self.exact
    }

    /// Whether the deadline has passed, as [`Instant::now`] reads it.
    pub fn expired(&mut self) -> bool {
        if self.ahead() {
            return false;
        }
        Instant::now() >= self.exact
    }

    /// How long until the deadline, exactly zero once [`expired`](Self::expired) would say so. Far
    /// from the deadline the value is a lower bound of the true remainder (never zero, never
    /// past the deadline): a wait of that length ends before the deadline, and the next call
    /// is closer. Within the coarse clock's bound of the deadline it is the true remainder.
    pub fn remaining(&mut self) -> Duration {
        if let Some(filter) = self.filter {
            let ahead = filter.saturating_duration_since(self.clock.now());
            if !ahead.is_zero() {
                return ahead;
            }
        }
        self.exact.saturating_duration_since(Instant::now())
    }

    /// Whether the coarse clock proves the deadline has not come: it is below the threshold.
    fn ahead(&mut self) -> bool {
        self.filter.is_some_and(|filter| self.clock.now() < filter)
    }
}

/// A bound on how far the coarse clock can read from the monotonic clock, in either direction: with
/// margin, twice its resolution.
///
/// - Linux: twice `clock_getres(CLOCK_MONOTONIC_COARSE)` (one tick: 1 to 4 ms), and a millisecond.
/// - Windows: 32 ms. The clock is `GetTickCount64`, whose resolution is documented as 10 to 16 ms.
/// - Elsewhere, and under Miri, the clock is the monotonic clock itself: zero.
///
/// An overestimate costs only reads of the monotonic clock, so each figure is generous.
pub fn coarse_bound() -> Duration {
    static BOUND: OnceLock<Duration> = OnceLock::new();
    *BOUND.get_or_init(measure_bound)
}

#[cfg(all(target_os = "linux", not(miri)))]
fn measure_bound() -> Duration {
    // The kernel's coarse clock is `CLOCK_MONOTONIC_COARSE`, a tick of `1/CONFIG_HZ`.
    let mut res = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `res` is a valid timespec to write to.
    let ok = unsafe { libc::clock_getres(libc::CLOCK_MONOTONIC_COARSE, &raw mut res) } == 0;
    let tick = match (ok, u64::try_from(res.tv_sec), u32::try_from(res.tv_nsec)) {
        (true, Ok(secs), Ok(nanos)) => Duration::new(secs, nanos),
        _ => Duration::from_millis(16),
    };
    2 * tick + Duration::from_millis(1)
}

#[cfg(all(windows, not(miri)))]
fn measure_bound() -> Duration {
    Duration::from_millis(32)
}

#[cfg(any(miri, not(any(target_os = "linux", windows))))]
fn measure_bound() -> Duration {
    Duration::ZERO
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spin_until(at: Instant) {
        while Instant::now() < at {
            std::hint::spin_loop();
        }
    }

    #[test]
    fn a_deadline_in_the_past_is_expired_and_has_nothing_remaining() {
        let mut deadline = Deadline::at(Instant::now());
        assert!(deadline.expired());
        assert_eq!(deadline.remaining(), Duration::ZERO);
        assert_eq!(Deadline::after(Duration::ZERO).remaining(), Duration::ZERO);
    }

    #[test]
    fn a_deadline_ahead_is_not_expired_and_remains() {
        let mut deadline = Deadline::after(Duration::from_secs(3600));
        assert!(!deadline.expired());
        let remaining = deadline.remaining();
        assert!(remaining > Duration::from_secs(3590), "{remaining:?}");
        assert!(remaining <= Duration::from_secs(3600), "{remaining:?}");
    }

    #[test]
    fn it_is_never_early() {
        // Many short deadlines, from every phase of the coarse clock's tick.
        for micros in (0..60).map(|i| 200 + i * 97) {
            let timeout = Duration::from_micros(micros);
            let mut deadline = Deadline::after(timeout);
            let exact = deadline.exact();
            while !deadline.expired() {
                assert!(Instant::now() < exact + Duration::from_secs(5), "never");
            }
            assert!(Instant::now() >= exact, "expired before the deadline");
        }
    }

    #[test]
    fn expiry_is_seen_by_the_first_check_after_the_monotonic_clock_passes_it() {
        // The deadline is made at every phase of the coarse tick, and checked the moment
        // `Instant` passes it: the filter must never hold it back.
        for i in 0..400_u64 {
            let mut deadline = Deadline::after(Duration::from_micros(50 + i * 37));
            spin_until(deadline.exact());
            assert!(deadline.expired(), "late at {i}");
            assert_eq!(deadline.remaining(), Duration::ZERO, "late at {i}");
        }
        // And from a clock made long before: its anchor is at another phase of the tick.
        let mut deadline = Deadline::after(Duration::from_millis(40));
        for i in 0..100_u64 {
            let at = deadline.exact();
            spin_until(at);
            assert!(deadline.expired(), "late at {i}");
            deadline.restart(Duration::from_micros(300 + i * 71));
        }
    }

    #[test]
    fn the_remainder_never_runs_past_the_deadline_and_ends_at_it() {
        for i in 0..50_u64 {
            let mut deadline = Deadline::after(Duration::from_micros(500 + i * 211));
            let exact = deadline.exact();
            loop {
                let remaining = deadline.remaining();
                if remaining.is_zero() {
                    break;
                }
                // Not past the deadline: a wait of `remaining` ends at or before it.
                assert!(Instant::now() + remaining <= exact + Duration::from_micros(50));
                spin_until(Instant::now() + remaining.min(Duration::from_micros(100)));
            }
            assert!(Instant::now() >= exact, "zero before the deadline");
        }
    }

    #[test]
    fn the_bound_covers_the_coarse_clock_skew_measured_here() {
        let bound = coarse_bound();
        let mut worst = Duration::ZERO;
        for _ in 0..60 {
            let mut clock = fast_time::Clock::new();
            let start = Instant::now() + Duration::from_millis(5);
            let skew = |coarse: Instant, exact: Instant| {
                coarse
                    .saturating_duration_since(exact)
                    .max(exact.saturating_duration_since(coarse))
            };
            while Instant::now() < start {
                let coarse: Instant = clock.now().into();
                worst = worst.max(skew(coarse, Instant::now()));
            }
        }
        assert!(worst < bound, "skew {worst:?} against bound {bound:?}");
    }

    #[test]
    fn restarting_moves_the_deadline_from_now() {
        let mut deadline = Deadline::at(Instant::now());
        assert!(deadline.expired());
        deadline.restart(Duration::from_secs(3600));
        assert!(!deadline.expired());
        assert!(deadline.remaining() > Duration::from_secs(3590));
        deadline.restart(Duration::ZERO);
        assert!(deadline.expired());
    }

    #[test]
    fn the_exact_deadline_is_the_one_given() {
        let at = Instant::now() + Duration::from_secs(5);
        assert_eq!(Deadline::at(at).exact(), at);
    }

    #[test]
    fn a_timed_wait_by_the_remainder_ends_at_the_deadline() {
        let timeout = Duration::from_millis(30);
        let mut deadline = Deadline::after(timeout);
        let exact = deadline.exact();
        loop {
            let remaining = deadline.remaining();
            if remaining.is_zero() {
                break;
            }
            std::thread::sleep(remaining);
        }
        assert!(Instant::now() >= exact, "early");
    }
}
