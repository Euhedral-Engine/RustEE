use super::math::ewma;
use std::time::{Duration, Instant};

/// Running statistics of a flow of measurements: how many units each recording carried, how long
/// apart the recordings were, and the rate between them.
///
/// Each statistic is an EWMA with weight `alpha` on the newest sample, with an exponentially
/// weighted variance and trend, and min and max values that decay to within three standard
/// deviations of the average. Recordings are also counted per measurement window, so the
/// [rolling average](Self::rolling_average_at) can blend the previous window into the current one.
///
/// The first recording only starts the clock; the second initializes the statistics; later
/// recordings update them. An interval that differs from the average interval by more than a whole
/// window restarts the recorder. A clone is a snapshot.
#[derive(Clone, Debug)]
pub struct FlowRecorder {
    alpha: f64,
    window: Option<Duration>,
    window_start: Instant,
    prev_window_count: u64,
    curr_window_count: u64,

    last_recording: Option<Instant>,
    last_interval: Duration,
    last_recorded_units: i64,
    rolling_sum: i64,

    /// The statistics below have been initialized from a first interval.
    measured: bool,
    /// The rate average has taken its first sample.
    rate_seeded: bool,

    average_units: f64,
    average_interval: f64,
    average_rate: f64,
    min_units: i64,
    max_units: i64,
    min_interval: i64,
    max_interval: i64,
    min_rate: f64,
    max_rate: f64,
    unit_variance: f64,
    interval_variance: f64,
    rate_variance: f64,
    unit_trend: f64,
    interval_trend: f64,
    rate_trend: f64,
}

impl FlowRecorder {
    /// The default measurement window.
    pub const DEFAULT_WINDOW: Duration = Duration::from_millis(10);

    /// The default EWMA weight of the newest sample.
    pub const DEFAULT_ALPHA: f64 = 0.05;

    /// A recorder with a [`DEFAULT_WINDOW`](Self::DEFAULT_WINDOW) window and weight
    /// [`DEFAULT_ALPHA`](Self::DEFAULT_ALPHA).
    pub fn new() -> Self {
        Self::with_window(Some(Self::DEFAULT_WINDOW), Self::DEFAULT_ALPHA)
    }

    /// A recorder counting recordings per `window` (`None` for one unbounded window), weighting
    /// each new sample by `alpha`.
    ///
    /// # Panics
    ///
    /// Panics unless `0 < alpha <= 1`.
    pub fn with_window(window: Option<Duration>, alpha: f64) -> Self {
        assert!(
            alpha > 0.0 && alpha <= 1.0,
            "alpha must be in (0, 1], got {alpha}"
        );
        let mut recorder = Self {
            alpha,
            window: window.filter(|window| !window.is_zero()),
            window_start: Instant::now(),
            prev_window_count: 0,
            curr_window_count: 0,
            last_recording: None,
            last_interval: Duration::ZERO,
            last_recorded_units: 0,
            rolling_sum: 0,
            measured: false,
            rate_seeded: false,
            average_units: 0.0,
            average_interval: 0.0,
            average_rate: 0.0,
            min_units: 0,
            max_units: 0,
            min_interval: 0,
            max_interval: 0,
            min_rate: 0.0,
            max_rate: 0.0,
            unit_variance: 0.0,
            interval_variance: 0.0,
            rate_variance: 0.0,
            unit_trend: 0.0,
            interval_trend: 0.0,
            rate_trend: 0.0,
        };
        recorder.reset();
        recorder
    }

    /// Records `units` now.
    pub fn record_units(&mut self, units: i64) {
        self.record_units_at(Instant::now(), units);
    }

    /// Records `units` at `now`. A recording no later than the previous one is ignored.
    pub fn record_units_at(&mut self, now: Instant, units: i64) {
        let Some(last) = self.last_recording else {
            self.last_recording = Some(now);
            self.last_recorded_units = units;
            self.rolling_sum = units;
            self.prev_window_count = 0;
            self.curr_window_count = 1;
            return;
        };
        if now <= last {
            return;
        }
        let interval = now - last;
        self.last_recording = Some(now);
        self.last_recorded_units = units;
        self.curr_window_count += 1;
        self.last_interval = interval;

        self.update(now, units, nanos(interval));

        if let Some(window) = self.window
            && now.saturating_duration_since(self.window_start) > window
        {
            self.prev_window_count = self.curr_window_count;
            self.curr_window_count = 0;
            self.window_start = now;
        }
    }

    fn update(&mut self, now: Instant, units: i64, interval: i64) {
        let alpha = self.alpha;
        let rate = units as f64 / (interval as f64).max(1e-9);
        self.rolling_sum = ((1.0 - alpha) * self.rolling_sum as f64 + units as f64).round() as i64;

        if !self.measured {
            self.measured = true;
            self.average_units = units as f64;
            self.average_interval = interval as f64;
            self.min_units = units;
            self.max_units = units;
            self.min_interval = interval;
            self.max_interval = interval;
            self.min_rate = rate;
            self.max_rate = rate;
            return;
        }
        if !self.rate_seeded {
            self.rate_seeded = true;
            self.average_rate = rate;
        }

        if let Some(window) = self.window
            && (interval as f64 - self.average_interval).abs() > window.as_nanos() as f64
        {
            self.reset_at(now);
            return;
        }

        self.average_units = ewma(self.average_units, units as f64, alpha);
        self.average_interval = ewma(self.average_interval, interval as f64, alpha);
        self.average_rate = ewma(self.average_rate, rate, alpha);

        self.decay_extremes();
        self.min_units = self.min_units.min(units);
        self.max_units = self.max_units.max(units);
        self.min_interval = self.min_interval.min(interval);
        self.max_interval = self.max_interval.max(interval);
        self.min_rate = self.min_rate.min(rate);
        self.max_rate = self.max_rate.max(rate);

        let decay = 1.0 - alpha;
        let delta = units as f64 - self.average_units;
        self.unit_variance = decay * (self.unit_variance + alpha * delta * delta);
        self.unit_trend = trend(self.unit_variance, self.unit_trend, delta, alpha);

        let delta = interval as f64 - self.average_interval;
        self.interval_variance = decay * (self.interval_variance + alpha * delta * delta);
        self.interval_trend = trend(self.interval_variance, self.interval_trend, delta, alpha);

        let delta = rate - self.average_rate;
        self.rate_variance = decay * (self.rate_variance + alpha * delta * delta);
        self.rate_trend = trend(self.rate_variance, self.rate_trend, delta, alpha);
    }

    /// Pulls each min and max to within three standard deviations of its average.
    fn decay_extremes(&mut self) {
        let spread = 3.0 * self.unit_standard_deviation();
        let (low, high) = (self.average_units - spread, self.average_units + spread);
        if (self.min_units as f64) < low {
            self.min_units = low.floor() as i64;
        }
        if (self.max_units as f64) > high {
            self.max_units = high.ceil() as i64;
        }

        let spread = 3.0 * self.interval_standard_deviation();
        let (low, high) = (
            self.average_interval - spread,
            self.average_interval + spread,
        );
        if (self.min_interval as f64) < low {
            self.min_interval = low.floor() as i64;
        }
        if (self.max_interval as f64) > high {
            self.max_interval = high.ceil() as i64;
        }

        let spread = 3.0 * self.units_over_time_standard_deviation();
        self.min_rate = self.min_rate.max(self.average_rate - spread);
        self.max_rate = self.max_rate.min(self.average_rate + spread);
    }

    /// Forgets every recording and starts a new window now.
    pub fn reset(&mut self) {
        self.reset_at(Instant::now());
    }

    /// Forgets every recording and starts a new window at `now`.
    pub fn reset_at(&mut self, now: Instant) {
        self.prev_window_count = 0;
        self.curr_window_count = 0;
        self.last_recording = None;
        self.window_start = now;
        self.rolling_sum = 0;
        self.measured = false;
        self.rate_seeded = false;
        self.average_units = 0.0;
        self.average_interval = 0.0;
        self.average_rate = 0.0;
        self.unit_variance = 0.0;
        self.interval_variance = 0.0;
        self.rate_variance = 0.0;
        self.unit_trend = 0.0;
        self.interval_trend = 0.0;
        self.rate_trend = 0.0;
    }

    /// The decayed sum of recorded units over the [effective window
    /// count](Self::effective_window_count_at), or zero with no recent recordings.
    pub fn rolling_average_at(&self, now: Instant) -> f64 {
        let count = self.effective_window_count_at(now);
        if count == 0.0 {
            0.0
        } else {
            self.rolling_sum as f64 / count
        }
    }

    pub fn rolling_average(&self) -> f64 {
        self.rolling_average_at(Instant::now())
    }

    /// Recordings in the current window, plus the previous window's count weighted by how much of
    /// the current window is still to come: `prev * (1 - progress) + current`, at least one.
    ///
    /// Zero once a whole window has passed with no roll-over. Without a window, the count of every
    /// recording since the last reset.
    pub fn effective_window_count_at(&self, now: Instant) -> f64 {
        let Some(window) = self.window else {
            return self.curr_window_count as f64;
        };
        let elapsed = now.saturating_duration_since(self.window_start);
        if elapsed > window {
            return 0.0;
        }
        if elapsed.is_zero() {
            return self.prev_window_count as f64;
        }
        if self.prev_window_count == 0 && self.last_recording.is_none_or(|last| now >= last) {
            return self.curr_window_count as f64;
        }
        let progress = elapsed.as_secs_f64() / window.as_secs_f64();
        let total =
            self.prev_window_count as f64 * (1.0 - progress) + self.curr_window_count as f64;
        total.max(1.0)
    }

    pub fn effective_window_count(&self) -> f64 {
        self.effective_window_count_at(Instant::now())
    }

    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// The measurement window; `None` for one unbounded window.
    pub fn window(&self) -> Option<Duration> {
        self.window
    }

    pub fn current_window_count(&self) -> u64 {
        self.curr_window_count
    }

    pub fn last_recording(&self) -> Option<Instant> {
        self.last_recording
    }

    pub fn last_interval(&self) -> Duration {
        self.last_interval
    }

    pub fn last_recorded_units(&self) -> i64 {
        self.last_recorded_units
    }

    pub fn rolling_sum(&self) -> i64 {
        self.rolling_sum
    }

    /// False until two recordings have produced a first interval, and again after a reset.
    pub fn is_measured(&self) -> bool {
        self.measured
    }

    pub fn average_units(&self) -> f64 {
        self.average_units
    }

    pub fn unit_variance(&self) -> f64 {
        self.unit_variance
    }

    pub fn unit_trend(&self) -> f64 {
        self.unit_trend
    }

    pub fn unit_standard_deviation(&self) -> f64 {
        self.unit_variance.sqrt()
    }

    /// The coefficient of variation of the units, zero while the average is zero.
    pub fn unit_cv(&self) -> f64 {
        cv(self.unit_standard_deviation(), self.average_units)
    }

    /// The average interval, in nanoseconds.
    pub fn average_interval(&self) -> f64 {
        self.average_interval
    }

    pub fn interval_variance(&self) -> f64 {
        self.interval_variance
    }

    pub fn interval_trend(&self) -> f64 {
        self.interval_trend
    }

    pub fn interval_standard_deviation(&self) -> f64 {
        self.interval_variance.sqrt()
    }

    pub fn interval_cv(&self) -> f64 {
        cv(self.interval_standard_deviation(), self.average_interval)
    }

    /// The average rate, in units per nanosecond.
    pub fn average_units_over_time(&self) -> f64 {
        self.average_rate
    }

    pub fn units_over_time_variance(&self) -> f64 {
        self.rate_variance
    }

    pub fn units_over_time_trend(&self) -> f64 {
        self.rate_trend
    }

    pub fn units_over_time_standard_deviation(&self) -> f64 {
        self.rate_variance.sqrt()
    }

    pub fn units_over_time_cv(&self) -> f64 {
        cv(self.units_over_time_standard_deviation(), self.average_rate)
    }

    pub fn min_units(&self) -> Option<i64> {
        self.measured.then_some(self.min_units)
    }

    pub fn max_units(&self) -> Option<i64> {
        self.measured.then_some(self.max_units)
    }

    /// The smallest recent interval, in nanoseconds.
    pub fn min_interval(&self) -> Option<i64> {
        self.measured.then_some(self.min_interval)
    }

    /// The largest recent interval, in nanoseconds.
    pub fn max_interval(&self) -> Option<i64> {
        self.measured.then_some(self.max_interval)
    }

    pub fn min_units_over_time(&self) -> Option<f64> {
        self.measured.then_some(self.min_rate)
    }

    pub fn max_units_over_time(&self) -> Option<f64> {
        self.measured.then_some(self.max_rate)
    }
}

impl Default for FlowRecorder {
    fn default() -> Self {
        Self::new()
    }
}

fn nanos(duration: Duration) -> i64 {
    i64::try_from(duration.as_nanos()).unwrap_or(i64::MAX)
}

fn trend(variance: f64, trend: f64, delta: f64, alpha: f64) -> f64 {
    if variance == 0.0 {
        0.0
    } else {
        ewma(trend, delta, alpha)
    }
}

fn cv(standard_deviation: f64, average: f64) -> f64 {
    if average == 0.0 {
        0.0
    } else {
        standard_deviation / average
    }
}

#[cfg(test)]
mod tests;
