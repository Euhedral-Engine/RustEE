use std::num::NonZeroU64;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// A sampling timer: of every `interval` start/stop pairs, only the first is timed, which keeps the
/// clock reads off the hot path.
///
/// The fields are plain state used by one thread at a time, held in relaxed atomics so that the two
/// hooks of an output can share the watch through an `Arc`. The count and the start time are only
/// loaded and stored: a start and a stop racing from two threads can shift which pair is timed, and
/// nothing else. The running flag is the one handshake: [`stop`](Self::stop) takes it with a swap,
/// so one stop takes each measurement however many race.
#[derive(Debug)]
pub struct StopWatch {
    interval: NonZeroU64,
    /// The instant `last_ns` counts from.
    origin: Instant,
    /// Whether a measurement is running.
    started: AtomicBool,
    ticks: AtomicU64,
    /// Nanoseconds from `origin` to the start of the running measurement.
    last_ns: AtomicU64,
}

impl StopWatch {
    /// The default number of start/stop pairs per measurement.
    pub const DEFAULT_INTERVAL: NonZeroU64 = NonZeroU64::new(256).unwrap();

    /// Times one in every [`DEFAULT_INTERVAL`](Self::DEFAULT_INTERVAL) pairs.
    pub fn new() -> Self {
        Self::with_interval(Self::DEFAULT_INTERVAL)
    }

    /// Times one in every `interval` pairs.
    pub fn with_interval(interval: NonZeroU64) -> Self {
        Self {
            interval,
            origin: Instant::now(),
            started: AtomicBool::new(false),
            ticks: AtomicU64::new(0),
            last_ns: AtomicU64::new(0),
        }
    }

    pub fn interval(&self) -> NonZeroU64 {
        self.interval
    }

    /// Nanoseconds since `origin`.
    fn now_ns(&self) -> u64 {
        self.origin.elapsed().as_nanos() as u64
    }

    /// Starts the clock if this pair is the one to time; otherwise only counts the pair. Does
    /// nothing while the clock is running.
    pub fn start(&self) {
        if self.started.load(Ordering::Relaxed) {
            return;
        }
        let ticks = self.ticks.load(Ordering::Relaxed);
        if ticks % self.interval == 0 {
            self.last_ns.store(self.now_ns(), Ordering::Relaxed);
            // The flag publishes the start time to the stop that takes it.
            self.started.store(true, Ordering::Release);
        } else {
            self.ticks.store(ticks.wrapping_add(1), Ordering::Relaxed);
        }
    }

    /// The time since the matching [`start`](Self::start), if that start began a measurement.
    pub fn stop(&self) -> Option<Duration> {
        // Most stops find no measurement and only read the flag.
        if !self.started.load(Ordering::Relaxed) || !self.started.swap(false, Ordering::Acquire) {
            return None;
        }
        let elapsed = self
            .now_ns()
            .saturating_sub(self.last_ns.load(Ordering::Relaxed));
        self.ticks.fetch_add(1, Ordering::Relaxed);
        Some(Duration::from_nanos(elapsed))
    }
}

impl Clone for StopWatch {
    fn clone(&self) -> Self {
        Self {
            interval: self.interval,
            origin: self.origin,
            started: AtomicBool::new(self.started.load(Ordering::Acquire)),
            ticks: AtomicU64::new(self.ticks.load(Ordering::Relaxed)),
            last_ns: AtomicU64::new(self.last_ns.load(Ordering::Relaxed)),
        }
    }
}

impl Default for StopWatch {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Barrier;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::thread;

    fn every(interval: u64) -> StopWatch {
        StopWatch::with_interval(NonZeroU64::new(interval).unwrap())
    }

    #[test]
    fn stop_without_start_measures_nothing() {
        assert_eq!(every(1).stop(), None);
    }

    #[test]
    fn an_interval_of_one_times_every_pair() {
        let watch = every(1);
        for _ in 0..3 {
            watch.start();
            assert!(watch.stop().is_some());
        }
    }

    #[test]
    fn only_the_first_of_each_interval_is_timed() {
        let watch = every(4);
        let timed: Vec<bool> = (0..9)
            .map(|_| {
                watch.start();
                watch.stop().is_some()
            })
            .collect();
        assert_eq!(
            timed,
            [true, false, false, false, true, false, false, false, true]
        );
    }

    #[test]
    fn intervals_need_not_be_powers_of_two() {
        let watch = every(3);
        let timed: Vec<bool> = (0..7)
            .map(|_| {
                watch.start();
                watch.stop().is_some()
            })
            .collect();
        assert_eq!(timed, [true, false, false, true, false, false, true]);
    }

    #[test]
    fn a_second_start_does_not_restart_the_clock() {
        let watch = every(1);
        watch.start();
        thread::sleep(Duration::from_millis(5));
        watch.start();
        assert!(watch.stop().unwrap() >= Duration::from_millis(5));
        assert_eq!(watch.stop(), None);
    }

    #[test]
    fn a_shared_stop_watch_times_one_call_in_256() {
        let watch = StopWatch::new();
        let shared = &watch;
        let timed = (0..1024)
            .filter(|_| {
                shared.start();
                shared.stop().is_some()
            })
            .count();
        assert_eq!(timed, 4);
    }

    #[test]
    fn one_of_several_racing_stops_takes_the_measurement() {
        const STOPPERS: usize = if cfg!(miri) { 2 } else { 4 };
        const ROUNDS: usize = if cfg!(miri) { 8 } else { 2000 };
        let watch = StopWatch::with_interval(NonZeroU64::new(1).unwrap());
        let barrier = Barrier::new(STOPPERS + 1);
        let taken = AtomicUsize::new(0);
        thread::scope(|scope| {
            for _ in 0..STOPPERS {
                scope.spawn(|| {
                    for _ in 0..ROUNDS {
                        barrier.wait();
                        if watch.stop().is_some() {
                            taken.fetch_add(1, Ordering::Relaxed);
                        }
                        barrier.wait();
                    }
                });
            }
            for _ in 0..ROUNDS {
                watch.start();
                barrier.wait();
                barrier.wait();
            }
        });
        assert_eq!(taken.load(Ordering::Relaxed), ROUNDS);
    }

    #[test]
    fn a_clone_keeps_the_count_and_the_running_clock() {
        let watch = every(2);
        watch.start();
        let copy = watch.clone();
        assert!(copy.stop().is_some());
        assert!(watch.stop().is_some());
        watch.start();
        assert_eq!(watch.stop(), None);
    }

    #[test]
    fn the_default_interval_is_256() {
        assert_eq!(StopWatch::new().interval().get(), 256);
        assert_eq!(StopWatch::default().interval(), StopWatch::DEFAULT_INTERVAL);
    }
}
