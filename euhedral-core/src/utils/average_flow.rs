use std::error::Error;
use std::fmt;

/// The fixed-point scale of an acquisition-contention sample: `SCALE` is full contention.
pub const ACQUIRE_CONTENTION_SCALE: u64 = 1_000_000;

/// Each sample moves the average by `1 / DIVISOR` of its distance from the sample.
pub const DIVISOR: u64 = 16;

/// A fixed-point EWMA of samples in `0..=`[`ACQUIRE_CONTENTION_SCALE`], with no clock reads or
/// floating-point work.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AverageFlow {
    value: Option<u64>,
}

impl AverageFlow {
    pub const fn new() -> Self {
        Self { value: None }
    }

    /// Folds in one sample; the first sample since a reset is taken as is.
    pub fn record(&mut self, sample: u64) -> Result<(), SampleOutOfRange> {
        if sample > ACQUIRE_CONTENTION_SCALE {
            return Err(SampleOutOfRange(sample));
        }
        self.value = Some(match self.value {
            None => sample,
            Some(value) if sample >= value => value + (sample - value) / DIVISOR,
            Some(value) => value - (value - sample) / DIVISOR,
        });
        Ok(())
    }

    /// The average, once at least one sample has been recorded since the last reset.
    pub fn value(&self) -> Option<u64> {
        self.value
    }

    pub fn reset(&mut self) {
        self.value = None;
    }
}

/// A sample above [`ACQUIRE_CONTENTION_SCALE`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SampleOutOfRange(pub u64);

impl fmt::Display for SampleOutOfRange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "acquisition contention sample {} is outside 0..={ACQUIRE_CONTENTION_SCALE}",
            self.0
        )
    }
}

impl Error for SampleOutOfRange {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_first_sample_is_taken_exactly() {
        let mut average = AverageFlow::new();
        assert_eq!(average.value(), None);
        average.record(375_000).unwrap();
        assert_eq!(average.value(), Some(375_000));
    }

    #[test]
    fn repeated_equal_samples_stay_put() {
        let mut average = AverageFlow::new();
        for _ in 0..1_000 {
            average.record(625_000).unwrap();
        }
        assert_eq!(average.value(), Some(625_000));
    }

    #[test]
    fn a_step_moves_one_sixteenth_of_the_distance() {
        let mut average = AverageFlow::new();
        average.record(0).unwrap();
        average.record(160).unwrap();
        assert_eq!(average.value(), Some(10));
        average.record(1_000_000).unwrap();
        assert_eq!(average.value(), Some(10 + 999_990 / 16));
        average.record(0).unwrap();
        assert_eq!(average.value(), Some(62_509 - 62_509 / 16));
    }

    #[test]
    fn steps_shorter_than_the_divisor_do_not_move_the_average() {
        let mut average = AverageFlow::new();
        average.record(100).unwrap();
        average.record(115).unwrap();
        average.record(85).unwrap();
        assert_eq!(average.value(), Some(100));
    }

    #[test]
    fn low_high_low_steps_move_monotonically_and_stay_bounded() {
        let mut average = AverageFlow::new();
        average.record(0).unwrap();
        let mut previous = average.value().unwrap();
        for _ in 0..256 {
            average.record(ACQUIRE_CONTENTION_SCALE).unwrap();
            let value = average.value().unwrap();
            assert!(value >= previous && value <= ACQUIRE_CONTENTION_SCALE);
            previous = value;
        }
        assert!(previous >= ACQUIRE_CONTENTION_SCALE - DIVISOR);

        for _ in 0..256 {
            average.record(0).unwrap();
            let value = average.value().unwrap();
            assert!(value <= previous);
            previous = value;
        }
        assert!(previous < DIVISOR);
    }

    #[test]
    fn reset_forgets_the_average() {
        let mut average = AverageFlow::new();
        average.record(500_000).unwrap();
        average.reset();
        assert_eq!(average.value(), None);
        average.record(7).unwrap();
        assert_eq!(average.value(), Some(7));
    }

    #[test]
    fn samples_above_the_scale_are_rejected() {
        let mut average = AverageFlow::new();
        assert_eq!(
            average.record(ACQUIRE_CONTENTION_SCALE + 1),
            Err(SampleOutOfRange(ACQUIRE_CONTENTION_SCALE + 1))
        );
        assert_eq!(average.value(), None);
        assert!(average.record(ACQUIRE_CONTENTION_SCALE).is_ok());
        assert!(
            SampleOutOfRange(2_000_000)
                .to_string()
                .contains("outside 0..=1000000")
        );
    }
}
