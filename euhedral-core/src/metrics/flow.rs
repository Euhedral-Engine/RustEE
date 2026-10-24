use crate::utils::FlowRecorder;
use euhedral_data_structures::atomics::CachePadded;
use std::sync::atomic::{AtomicU64, Ordering, fence};

/// How many times a reader retries a read a write overlapped before it gives up.
const READ_ATTEMPTS: usize = 64;

/// The current statistics of a [`FlowRecorder`], in one meter's unit: its decaying average,
/// standard deviation and trend, and its recent minimum and maximum (which decay to within three
/// standard deviations of the average).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FlowStats {
    pub average: f64,
    pub standard_deviation: f64,
    pub trend: f64,
    pub min: f64,
    pub max: f64,
}

impl FlowStats {
    /// The statistics of `recorder`'s units, `None` until it has been measured.
    pub fn of_units(recorder: &FlowRecorder) -> Option<Self> {
        Some(Self {
            average: recorder.average_units(),
            standard_deviation: recorder.unit_standard_deviation(),
            trend: recorder.unit_trend(),
            min: recorder.min_units()? as f64,
            max: recorder.max_units()? as f64,
        })
    }

    /// The statistics of `recorder`'s rate, in units per nanosecond times `scale`, `None` until it
    /// has been measured.
    pub fn of_rate(recorder: &FlowRecorder, scale: f64) -> Option<Self> {
        Some(Self {
            average: recorder.average_units_over_time() * scale,
            standard_deviation: recorder.units_over_time_standard_deviation() * scale,
            trend: recorder.units_over_time_trend() * scale,
            min: recorder.min_units_over_time()? * scale,
            max: recorder.max_units_over_time()? * scale,
        })
    }
}

/// [`FlowStats`] one writer publishes and any reader reads whole, through a sequence lock: the
/// writer never waits, and a reader retries a read that a write overlapped.
pub(crate) struct PublishedFlow {
    inner: CachePadded<Inner>,
}

struct Inner {
    /// Odd while a write is in progress; zero before the first.
    sequence: AtomicU64,
    fields: [AtomicU64; 5],
}

impl PublishedFlow {
    pub(crate) fn new() -> Self {
        Self {
            inner: CachePadded::new(Inner {
                sequence: AtomicU64::new(0),
                fields: [const { AtomicU64::new(0) }; 5],
            }),
        }
    }

    /// Publishes `stats`. Only one thread may publish.
    #[inline]
    pub(crate) fn publish(&self, stats: &FlowStats) {
        let inner = &*self.inner;
        let sequence = inner.sequence.load(Ordering::Relaxed);
        inner.sequence.store(sequence + 1, Ordering::Relaxed);
        fence(Ordering::Release);
        let values = [
            stats.average,
            stats.standard_deviation,
            stats.trend,
            stats.min,
            stats.max,
        ];
        for (field, value) in inner.fields.iter().zip(values) {
            field.store(value.to_bits(), Ordering::Relaxed);
        }
        inner.sequence.store(sequence + 2, Ordering::Release);
    }

    /// The last statistics published, `None` before any (or if writes kept overlapping the read).
    pub(crate) fn read(&self) -> Option<FlowStats> {
        let inner = &*self.inner;
        for _ in 0..READ_ATTEMPTS {
            let before = inner.sequence.load(Ordering::Acquire);
            if before == 0 {
                return None;
            }
            if before % 2 == 1 {
                std::hint::spin_loop();
                continue;
            }
            let [average, standard_deviation, trend, min, max] = inner
                .fields
                .each_ref()
                .map(|field| f64::from_bits(field.load(Ordering::Relaxed)));
            fence(Ordering::Acquire);
            if inner.sequence.load(Ordering::Relaxed) == before {
                return Some(FlowStats {
                    average,
                    standard_deviation,
                    trend,
                    min,
                    max,
                });
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    fn stats(value: f64) -> FlowStats {
        FlowStats {
            average: value,
            standard_deviation: value,
            trend: value,
            min: value,
            max: value,
        }
    }

    #[test]
    fn nothing_is_read_before_the_first_publish() {
        let flow = PublishedFlow::new();
        assert_eq!(flow.read(), None);
        flow.publish(&stats(2.0));
        assert_eq!(flow.read(), Some(stats(2.0)));
        flow.publish(&stats(3.0));
        assert_eq!(flow.read(), Some(stats(3.0)));
    }

    #[test]
    fn readers_never_see_a_torn_publish() {
        let flow = PublishedFlow::new();
        let stop = std::sync::atomic::AtomicBool::new(false);
        std::thread::scope(|scope| {
            scope.spawn(|| {
                let mut value = 1.0;
                while !stop.load(Ordering::Relaxed) {
                    flow.publish(&stats(value));
                    value += 1.0;
                }
            });
            let mut seen = 0;
            for _ in 0..20_000 {
                if let Some(read) = flow.read() {
                    assert_eq!(read, stats(read.average));
                    seen += 1;
                }
            }
            stop.store(true, Ordering::Relaxed);
            assert!(seen > 0);
        });
    }

    #[test]
    fn stats_come_from_the_recorder_once_measured() {
        let mut recorder = FlowRecorder::new();
        let start = Instant::now();
        recorder.record_units_at(start, 10);
        assert_eq!(FlowStats::of_units(&recorder), None);
        assert_eq!(FlowStats::of_rate(&recorder, 1.0), None);
        for step in 1..=3u32 {
            recorder.record_units_at(start + Duration::from_micros(u64::from(step)), 10);
        }
        let units = FlowStats::of_units(&recorder).unwrap();
        assert_eq!((units.average, units.min, units.max), (10.0, 10.0, 10.0));
        assert_eq!(units.standard_deviation, 0.0);
        // Ten units a microsecond: ten million a second.
        let rate = FlowStats::of_rate(&recorder, 1e9).unwrap();
        assert!((rate.average - 1e7).abs() < 1e-3, "{rate:?}");
    }
}
