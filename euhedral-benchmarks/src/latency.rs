//! Per-frame latency: a log-linear histogram, and the recorders the executing workers write to.
//!
//! A sampled frame is stamped when its source hands it out and, when a worker executes it, the worker records the time
//! since the stamp (the use `SingleWriterRecorder` has in the upstream benchmarks). Each worker records into a
//! histogram of its own, with relaxed atomic increments and no lock; the harness reads the counts at the end of each
//! iteration and takes the difference from the start's.
//!
//! The buckets are fixed and log-linear: values below 128 ns are exact, and every power of two above has 128 equal
//! buckets, so a value is known to within 0.8%. The range is 1 ns to about 2^37 ns (137 s); larger values land in the
//! last bucket.
//!
//! The latency measured is the time from a source handing a frame out to a worker starting to execute it. It excludes
//! the way back: the finished frame's return to its source (`recycle`) is not in it.

use std::sync::atomic::{AtomicU64, Ordering};

/// Buckets per power of two.
const SUB_BUCKETS: usize = 128;
const SUB_BITS: u32 = 7;
/// The highest power of two with its own buckets.
const MAX_SHIFT: usize = 29;
/// All buckets.
pub const BUCKETS: usize = SUB_BUCKETS + (MAX_SHIFT + 1) * SUB_BUCKETS;

/// The bucket of `value` nanoseconds.
pub fn bucket_of(value: u64) -> usize {
    if value < SUB_BUCKETS as u64 {
        return value as usize;
    }
    let shift = (63 - value.leading_zeros() - SUB_BITS) as usize;
    if shift > MAX_SHIFT {
        return BUCKETS - 1;
    }
    SUB_BUCKETS + shift * SUB_BUCKETS + ((value >> shift) as usize - SUB_BUCKETS)
}

/// The smallest value of `bucket`.
pub fn bucket_low(bucket: usize) -> u64 {
    if bucket < SUB_BUCKETS {
        return bucket as u64;
    }
    let shift = (bucket - SUB_BUCKETS) / SUB_BUCKETS;
    let mantissa = SUB_BUCKETS + (bucket - SUB_BUCKETS) % SUB_BUCKETS;
    (mantissa as u64) << shift
}

/// The largest value of `bucket`.
pub fn bucket_high(bucket: usize) -> u64 {
    if bucket < SUB_BUCKETS {
        return bucket as u64;
    }
    let shift = (bucket - SUB_BUCKETS) / SUB_BUCKETS;
    bucket_low(bucket) + (1_u64 << shift) - 1
}

/// The quantiles the benchmark reports, in nanoseconds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Quantiles {
    pub count: u64,
    pub p0: u64,
    pub p25: u64,
    pub p50: u64,
    pub p90: u64,
    pub p99: u64,
    pub p100: u64,
}

/// Counts of recorded values by bucket.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Histogram {
    counts: Vec<u64>,
}

impl Default for Histogram {
    fn default() -> Self {
        Self::new()
    }
}

impl Histogram {
    pub fn new() -> Self {
        Self {
            counts: vec![0; BUCKETS],
        }
    }

    pub fn record(&mut self, value: u64) {
        self.counts[bucket_of(value)] += 1;
    }

    pub fn add_count(&mut self, bucket: usize, count: u64) {
        self.counts[bucket.min(BUCKETS - 1)] += count;
    }

    pub fn count(&self) -> u64 {
        self.counts.iter().sum()
    }

    /// Adds every count of `other`.
    pub fn merge(&mut self, other: &Histogram) {
        for (mine, theirs) in self.counts.iter_mut().zip(&other.counts) {
            *mine += theirs;
        }
    }

    /// The counts recorded since `earlier`, a snapshot of the same recorders.
    pub fn since(&self, earlier: &Histogram) -> Histogram {
        Histogram {
            counts: self
                .counts
                .iter()
                .zip(&earlier.counts)
                .map(|(now, before)| now.saturating_sub(*before))
                .collect(),
        }
    }

    /// The value at quantile `q` in `[0, 1]`: the middle of the bucket that holds the `q`-th value; `0` is the low end of
    /// the first occupied bucket and `1` the high end of the last. `None` when nothing was recorded.
    pub fn quantile(&self, q: f64) -> Option<u64> {
        let total = self.count();
        if total == 0 {
            return None;
        }
        let first = self.counts.iter().position(|&c| c > 0)?;
        let last = self.counts.iter().rposition(|&c| c > 0)?;
        if q <= 0.0 {
            return Some(bucket_low(first));
        }
        if q >= 1.0 {
            return Some(bucket_high(last));
        }
        let rank = ((q * total as f64).ceil() as u64).clamp(1, total);
        let mut seen = 0;
        for (bucket, &count) in self.counts.iter().enumerate() {
            seen += count;
            if seen >= rank {
                return Some((bucket_low(bucket) + bucket_high(bucket)) / 2);
            }
        }
        Some(bucket_high(last))
    }

    /// P0, P25, P50, P90, P99 and P100, or `None` when nothing was recorded.
    pub fn quantiles(&self) -> Option<Quantiles> {
        Some(Quantiles {
            count: self.count(),
            p0: self.quantile(0.0)?,
            p25: self.quantile(0.25)?,
            p50: self.quantile(0.50)?,
            p90: self.quantile(0.90)?,
            p99: self.quantile(0.99)?,
            p100: self.quantile(1.0)?,
        })
    }

    /// `bucket:count` pairs of the occupied buckets, comma separated.
    pub fn to_sparse(&self) -> String {
        self.counts
            .iter()
            .enumerate()
            .filter(|(_, count)| **count > 0)
            .map(|(bucket, count)| format!("{bucket}:{count}"))
            .collect::<Vec<_>>()
            .join(",")
    }

    /// The histogram [`to_sparse`](Self::to_sparse) wrote.
    pub fn from_sparse(text: &str) -> Option<Histogram> {
        let mut histogram = Histogram::new();
        for pair in text.split(',').filter(|pair| !pair.is_empty()) {
            let (bucket, count) = pair.split_once(':')?;
            histogram.add_count(bucket.parse().ok()?, count.parse().ok()?);
        }
        Some(histogram)
    }
}

/// A worker's histogram: written with relaxed atomics, read by the harness at any time.
pub struct Recorder {
    counts: Vec<AtomicU64>,
}

impl Recorder {
    fn new() -> Self {
        Self {
            counts: (0..BUCKETS).map(|_| AtomicU64::new(0)).collect(),
        }
    }

    /// Counts `nanos`. An `owned` recorder has one writing thread, so the count is a load and a store.
    pub fn record(&self, nanos: u64, owned: bool) {
        let count = &self.counts[bucket_of(nanos)];
        if owned {
            count.store(count.load(Ordering::Relaxed) + 1, Ordering::Relaxed);
        } else {
            count.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn add_to(&self, histogram: &mut Histogram) {
        for (bucket, count) in self.counts.iter().enumerate() {
            let count = count.load(Ordering::Relaxed);
            if count > 0 {
                histogram.add_count(bucket, count);
            }
        }
    }
}

/// The recorders of a run, one per counter slot: a worker records into the slot of its thread.
pub struct Recorders {
    slots: Vec<Recorder>,
}

impl Recorders {
    pub fn new(slots: usize) -> Self {
        Self {
            slots: (0..slots.max(1)).map(|_| Recorder::new()).collect(),
        }
    }

    pub fn slot(&self, index: usize) -> &Recorder {
        &self.slots[index % self.slots.len()]
    }

    /// Everything recorded so far, merged.
    pub fn snapshot(&self) -> Histogram {
        let mut merged = Histogram::new();
        for slot in &self.slots {
            slot.add_to(&mut merged);
        }
        merged
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_values_are_exact_and_buckets_tile_the_range() {
        for value in 0..128 {
            assert_eq!(bucket_low(bucket_of(value)), value);
            assert_eq!(bucket_high(bucket_of(value)), value);
        }
        let mut next = 0;
        for bucket in 0..BUCKETS {
            assert_eq!(bucket_low(bucket), next, "bucket {bucket}");
            next = bucket_high(bucket) + 1;
        }
    }

    #[test]
    fn a_value_is_within_a_percent_of_its_bucket() {
        for value in [
            200_u64,
            1_000,
            12_345,
            1_000_000,
            987_654_321,
            40_000_000_000,
        ] {
            let bucket = bucket_of(value);
            assert!(bucket_low(bucket) <= value && value <= bucket_high(bucket));
            let width = (bucket_high(bucket) - bucket_low(bucket) + 1) as f64;
            assert!(width / value as f64 <= 0.0079, "{value}");
        }
        assert_eq!(bucket_of(u64::MAX), BUCKETS - 1);
    }

    #[test]
    fn quantiles_of_a_known_distribution() {
        let mut histogram = Histogram::new();
        for value in 1..=100 {
            histogram.record(value);
        }
        let q = histogram.quantiles().expect("recorded");
        assert_eq!((q.count, q.p0, q.p100), (100, 1, 100));
        assert_eq!((q.p25, q.p50, q.p90, q.p99), (25, 50, 90, 99));
        assert!(Histogram::new().quantiles().is_none());
    }

    #[test]
    fn quantiles_of_a_wide_distribution_are_close() {
        let mut histogram = Histogram::new();
        for value in 1..=10_000_u64 {
            histogram.record(value * 1_000);
        }
        let q = histogram.quantiles().expect("recorded");
        for (got, want) in [
            (q.p25, 2_500_000.0),
            (q.p50, 5_000_000.0),
            (q.p99, 9_900_000.0),
        ] {
            assert!((got as f64 - want).abs() / want < 0.01, "{got} vs {want}");
        }
    }

    #[test]
    fn merge_and_difference_and_the_sparse_text_round_trip() {
        let mut a = Histogram::new();
        let mut b = Histogram::new();
        a.record(10);
        a.record(5_000);
        b.record(10);
        let mut both = a.clone();
        both.merge(&b);
        assert_eq!(both.count(), 3);
        assert_eq!(both.since(&a), b);
        assert_eq!(Histogram::from_sparse(&both.to_sparse()), Some(both));
        assert!(Histogram::from_sparse("x:1").is_none());
    }

    #[test]
    fn recorders_merge_their_slots_and_snapshots_subtract() {
        let recorders = Recorders::new(3);
        recorders.slot(0).record(100, true);
        let before = recorders.snapshot();
        recorders.slot(1).record(300, true);
        recorders.slot(4).record(300, false);
        let during = recorders.snapshot().since(&before);
        assert_eq!(during.count(), 2);
        assert_eq!(during.quantile(0.5), Some(300));
    }
}
