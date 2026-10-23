use std::num::NonZeroU64;
use std::time::{Duration, Instant};

/// A sampling timer: of every `interval` start/stop pairs, only the first is timed, which keeps the
/// clock reads off the hot path.
#[derive(Clone, Debug)]
pub struct StopWatch {
    interval: NonZeroU64,
    ticks: u64,
    started: Option<Instant>,
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
            ticks: 0,
            started: None,
        }
    }

    pub fn interval(&self) -> NonZeroU64 {
        self.interval
    }

    /// Starts the clock if this pair is the one to time; otherwise only counts the pair. Does
    /// nothing while the clock is running.
    pub fn start(&mut self) {
        if self.started.is_some() {
            return;
        }
        if self.ticks % self.interval == 0 {
            self.started = Some(Instant::now());
        } else {
            self.ticks = self.ticks.wrapping_add(1);
        }
    }

    /// The time since the matching [`start`](Self::start), if that start began a measurement.
    pub fn stop(&mut self) -> Option<Duration> {
        let started = self.started.take()?;
        self.ticks = self.ticks.wrapping_add(1);
        Some(started.elapsed())
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
        let mut watch = every(1);
        for _ in 0..3 {
            watch.start();
            assert!(watch.stop().is_some());
        }
    }

    #[test]
    fn only_the_first_of_each_interval_is_timed() {
        let mut watch = every(4);
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
        let mut watch = every(3);
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
        let mut watch = every(1);
        watch.start();
        thread::sleep(Duration::from_millis(5));
        watch.start();
        assert!(watch.stop().unwrap() >= Duration::from_millis(5));
        assert_eq!(watch.stop(), None);
    }

    #[test]
    fn the_default_interval_is_256() {
        assert_eq!(StopWatch::new().interval().get(), 256);
        assert_eq!(StopWatch::default().interval(), StopWatch::DEFAULT_INTERVAL);
    }
}
