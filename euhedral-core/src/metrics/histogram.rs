use euhedral_data_structures::atomics::AtomicDouble;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

/// The exponent of the lowest octave: values below `2^MIN_EXPONENT` share the first bucket.
const MIN_EXPONENT: i32 = -16;
/// Octaves above the first bucket: `2^-16` up to `2^48`. Values from `2^48` up share the last.
const OCTAVES: usize = 64;
/// Linear sub-buckets per octave, as a power of two.
const SUB_BUCKET_BITS: u32 = 2;
const SUB_BUCKETS: usize = 1 << SUB_BUCKET_BITS;
/// The bucket count: one below the octaves, then `SUB_BUCKETS` per octave.
pub const BUCKETS: usize = 1 + OCTAVES * SUB_BUCKETS;

/// A cumulative distribution summary: the count, sum and maximum of the values recorded, and how
/// many fell in each of [`BUCKETS`] log2 buckets, all atomics.
///
/// Each octave `[2^e, 2^(e+1))` from `2^-16` to `2^48` is split into four equal buckets, so a
/// bucket is at most 25% wide relative to its lower bound. Values below `2^-16` (zero included)
/// share the first bucket and values of `2^48` or more the last. Negative and `NaN` values are
/// ignored, as Micrometer's distribution summaries ignore negative amounts.
///
/// Recording takes no lock and allocates nothing. Readers take a [`snapshot`](Self::snapshot) at
/// any time without stopping writers.
pub struct Histogram {
    count: AtomicU64,
    sum: AtomicDouble,
    /// The bits of the largest value: non-negative `f64`s order like their bits.
    max: AtomicU64,
    buckets: [AtomicU64; BUCKETS],
}

impl Histogram {
    pub fn new() -> Self {
        Self {
            count: AtomicU64::new(0),
            sum: AtomicDouble::new(0.0),
            max: AtomicU64::new(0.0f64.to_bits()),
            buckets: [const { AtomicU64::new(0) }; BUCKETS],
        }
    }

    /// Adds `value` to the distribution. Negative and `NaN` values are ignored.
    #[inline]
    pub fn record(&self, value: f64) {
        if value.is_nan() || value < 0.0 {
            return;
        }
        // -0.0 becomes 0.0, whose bits order below every positive value's.
        let value = value + 0.0;
        self.buckets[bucket_of(value)].fetch_add(1, Ordering::Relaxed);
        self.sum.get_and_add(value);
        self.max.fetch_max(value.to_bits(), Ordering::Relaxed);
        self.count.fetch_add(1, Ordering::Release);
    }

    /// The values recorded so far.
    pub fn count(&self) -> u64 {
        self.count.load(Ordering::Acquire)
    }

    /// The values recorded so far and their sum, read together only if no other thread records.
    #[inline]
    pub(crate) fn totals(&self) -> (u64, f64) {
        (self.count.load(Ordering::Relaxed), self.sum.get_opaque())
    }

    /// The distribution so far. Each field is read once while writers may keep recording, so a
    /// record in flight may show in the buckets before it shows in the count; percentiles use the
    /// buckets.
    pub fn snapshot(&self) -> HistogramSnapshot {
        let count = self.count.load(Ordering::Acquire);
        HistogramSnapshot {
            count,
            sum: self.sum.get_acquire(),
            max: f64::from_bits(self.max.load(Ordering::Relaxed)),
            buckets: self
                .buckets
                .iter()
                .map(|bucket| bucket.load(Ordering::Relaxed))
                .collect(),
        }
    }
}

impl Default for Histogram {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for Histogram {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let snapshot = self.snapshot();
        f.debug_struct("Histogram")
            .field("count", &snapshot.count)
            .field("sum", &snapshot.sum)
            .field("max", &snapshot.max)
            .finish_non_exhaustive()
    }
}

/// The bucket `value` (non-negative, not `NaN`) falls in.
#[inline]
fn bucket_of(value: f64) -> usize {
    let bits = value.to_bits();
    let exponent = ((bits >> 52) & 0x7ff) as i32 - 1023;
    if exponent < MIN_EXPONENT {
        return 0;
    }
    let sub = ((bits >> (52 - SUB_BUCKET_BITS)) as usize) & (SUB_BUCKETS - 1);
    let index = 1 + (exponent - MIN_EXPONENT) as usize * SUB_BUCKETS + sub;
    index.min(BUCKETS - 1)
}

/// The smallest value bucket `index` holds: `0` for the first bucket.
pub fn bucket_lower_bound(index: usize) -> f64 {
    if index == 0 {
        return 0.0;
    }
    let octave = (index - 1) / SUB_BUCKETS;
    let sub = (index - 1) % SUB_BUCKETS;
    let base = 2f64.powi(MIN_EXPONENT + octave as i32);
    base * (1.0 + sub as f64 / SUB_BUCKETS as f64)
}

/// The value bucket `index` holds values below: the next bucket's lower bound, infinite for the
/// last bucket.
pub fn bucket_upper_bound(index: usize) -> f64 {
    if index + 1 >= BUCKETS {
        f64::INFINITY
    } else {
        bucket_lower_bound(index + 1)
    }
}

/// A copy of a [`Histogram`] at one moment.
#[derive(Clone, Debug, PartialEq)]
pub struct HistogramSnapshot {
    /// The values recorded.
    pub count: u64,
    /// Their sum.
    pub sum: f64,
    /// The largest, or `0` before any.
    pub max: f64,
    /// How many values fell in each bucket; see [`bucket_lower_bound`] and [`bucket_upper_bound`].
    pub buckets: Box<[u64]>,
}

impl HistogramSnapshot {
    /// An empty distribution.
    pub fn empty() -> Self {
        Self {
            count: 0,
            sum: 0.0,
            max: 0.0,
            buckets: vec![0; BUCKETS].into_boxed_slice(),
        }
    }

    /// The mean value, or `0` before any.
    pub fn mean(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.sum / self.count as f64
        }
    }

    /// The value below which a share `quantile` (in `[0, 1]`) of the values fall, interpolated
    /// linearly within its bucket and capped at the maximum: within one bucket's width of the
    /// exact value. `None` before any value.
    pub fn percentile(&self, quantile: f64) -> Option<f64> {
        let total: u64 = self.buckets.iter().sum();
        if total == 0 {
            return None;
        }
        let rank = quantile.clamp(0.0, 1.0) * total as f64;
        let mut below = 0u64;
        for (index, &count) in self.buckets.iter().enumerate() {
            if count == 0 {
                continue;
            }
            if (below + count) as f64 >= rank {
                let lower = bucket_lower_bound(index);
                let upper = bucket_upper_bound(index).min(self.max.max(lower));
                let share = ((rank - below as f64) / count as f64).clamp(0.0, 1.0);
                return Some(lower + (upper - lower) * share);
            }
            below += count;
        }
        Some(self.max)
    }

    /// The values recorded since `earlier`, a snapshot of the same histogram. The maximum is this
    /// snapshot's, which may predate `earlier`.
    pub fn since(&self, earlier: &HistogramSnapshot) -> HistogramSnapshot {
        HistogramSnapshot {
            count: self.count.saturating_sub(earlier.count),
            sum: (self.sum - earlier.sum).max(0.0),
            max: self.max,
            buckets: self
                .buckets
                .iter()
                .zip(earlier.buckets.iter())
                .map(|(now, then)| now.saturating_sub(*then))
                .collect(),
        }
    }

    /// Adds `other`'s values to this snapshot, for example to combine a core's two SMT workers.
    pub fn merge(&mut self, other: &HistogramSnapshot) {
        self.count += other.count;
        self.sum += other.sum;
        self.max = self.max.max(other.max);
        for (mine, theirs) in self.buckets.iter_mut().zip(other.buckets.iter()) {
            *mine += theirs;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buckets_split_each_octave_in_four() {
        assert_eq!(bucket_of(0.0), 0);
        assert_eq!(bucket_of(2f64.powi(-17)), 0);
        assert_eq!(bucket_of(2f64.powi(-16)), 1);
        assert_eq!(bucket_of(1.0), 1 + 16 * 4);
        assert_eq!(bucket_of(1.24), 1 + 16 * 4);
        assert_eq!(bucket_of(1.25), 1 + 16 * 4 + 1);
        assert_eq!(bucket_of(1.99), 1 + 16 * 4 + 3);
        assert_eq!(bucket_of(2.0), 1 + 17 * 4);
        assert_eq!(bucket_of(2f64.powi(48)), BUCKETS - 1);
        assert_eq!(bucket_of(f64::MAX), BUCKETS - 1);
        assert_eq!(bucket_of(f64::INFINITY), BUCKETS - 1);
        for index in 1..BUCKETS {
            let lower = bucket_lower_bound(index);
            assert_eq!(bucket_of(lower), index, "lower bound of {index}");
            if index + 1 < BUCKETS {
                assert_eq!(bucket_of(bucket_upper_bound(index).next_down()), index);
            }
        }
        assert_eq!(bucket_upper_bound(BUCKETS - 1), f64::INFINITY);
    }

    #[test]
    fn recording_counts_sums_and_keeps_the_max() {
        let histogram = Histogram::new();
        for value in [3.0, 0.5, 100.0, 0.0] {
            histogram.record(value);
        }
        histogram.record(-1.0);
        histogram.record(f64::NAN);
        let zeros = Histogram::new();
        zeros.record(-0.0);
        assert_eq!(zeros.snapshot().max.to_bits(), 0.0f64.to_bits());
        assert_eq!(zeros.snapshot().buckets[0], 1);
        let snapshot = histogram.snapshot();
        assert_eq!(snapshot.count, 4);
        assert_eq!(histogram.count(), 4);
        assert_eq!(snapshot.sum, 103.5);
        assert_eq!(snapshot.max, 100.0);
        assert_eq!(snapshot.mean(), 103.5 / 4.0);
        assert_eq!(snapshot.buckets.iter().sum::<u64>(), 4);
        assert_eq!(snapshot.buckets[bucket_of(3.0)], 1);
    }

    #[test]
    fn percentiles_are_within_a_bucket_of_the_exact_value() {
        let histogram = Histogram::new();
        for value in 1..=1000 {
            histogram.record(value as f64);
        }
        let snapshot = histogram.snapshot();
        for (quantile, exact) in [(0.5, 500.0), (0.9, 900.0), (0.99, 990.0)] {
            let estimate = snapshot.percentile(quantile).unwrap();
            assert!(
                (estimate - exact).abs() <= exact * 0.25,
                "p{quantile}: {estimate} vs {exact}"
            );
        }
        assert_eq!(snapshot.percentile(1.0), Some(1000.0));
        assert!(snapshot.percentile(0.0).unwrap() <= 1.25);
        assert_eq!(HistogramSnapshot::empty().percentile(0.5), None);
    }

    #[test]
    fn percentiles_of_one_value_are_that_value() {
        let histogram = Histogram::new();
        for _ in 0..10 {
            histogram.record(0.6);
        }
        let snapshot = histogram.snapshot();
        let p50 = snapshot.percentile(0.5).unwrap();
        assert!((0.5..=0.6).contains(&p50), "{p50}");
        assert_eq!(snapshot.percentile(1.0), Some(0.6));
    }

    #[test]
    fn since_and_merge_combine_snapshots() {
        let histogram = Histogram::new();
        histogram.record(1.0);
        let earlier = histogram.snapshot();
        histogram.record(8.0);
        histogram.record(8.0);
        let delta = histogram.snapshot().since(&earlier);
        assert_eq!(delta.count, 2);
        assert_eq!(delta.sum, 16.0);
        assert_eq!(delta.buckets.iter().sum::<u64>(), 2);
        assert_eq!(delta.buckets[bucket_of(1.0)], 0);

        let mut merged = earlier.clone();
        merged.merge(&delta);
        assert_eq!(merged, histogram.snapshot());
    }

    #[test]
    fn concurrent_writers_lose_no_value() {
        let histogram = Histogram::new();
        std::thread::scope(|scope| {
            for writer in 0..4 {
                let histogram = &histogram;
                scope.spawn(move || {
                    for value in 0..10_000u64 {
                        histogram.record((value % 64 + writer) as f64);
                    }
                });
            }
            // A reader alongside the writers sees a consistent prefix.
            let snapshot = histogram.snapshot();
            assert!(snapshot.buckets.iter().sum::<u64>() >= snapshot.count);
        });
        let snapshot = histogram.snapshot();
        assert_eq!(snapshot.count, 40_000);
        assert_eq!(snapshot.buckets.iter().sum::<u64>(), 40_000);
        assert_eq!(snapshot.max, 66.0);
        let exact: u64 = (0..4u64)
            .map(|writer| (0..10_000u64).map(|v| v % 64 + writer).sum::<u64>())
            .sum();
        assert_eq!(snapshot.sum, exact as f64);
    }
}
