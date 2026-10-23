use super::{IdleTimingFunction, TimingProvider};
use std::error::Error;
use std::fmt;
use std::sync::Arc;
use std::time::Duration;

/// Timing for workers that have no work.
#[derive(Clone, Debug)]
pub struct IdlePolicy {
    idle_park: Duration,
    contention_half_life: Duration,
    function: Arc<dyn TimingProvider>,
    unproductive_park: Duration,
}

/// A contention half-life of zero.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ZeroHalfLife;

impl fmt::Display for ZeroHalfLife {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("the contention half-life must be positive")
    }
}

impl Error for ZeroHalfLife {}

impl IdlePolicy {
    pub const DEFAULT_IDLE_PARK: Duration = Duration::from_micros(15);
    pub const DEFAULT_UNPRODUCTIVE_PARK: Duration = Duration::from_micros(15);
    pub const DEFAULT_CONTENTION_HALF_LIFE: Duration = Duration::from_millis(1);

    /// A policy whose park and half-life `function` picks, falling back to `idle_park` and
    /// `contention_half_life`, with the default unproductive park.
    pub fn new(
        idle_park: Duration,
        contention_half_life: Duration,
        function: Arc<dyn TimingProvider>,
    ) -> Result<Self, ZeroHalfLife> {
        if contention_half_life.is_zero() {
            return Err(ZeroHalfLife);
        }
        Ok(Self {
            idle_park,
            contention_half_life,
            function,
            unproductive_park: Self::DEFAULT_UNPRODUCTIVE_PARK,
        })
    }

    /// A policy that always parks for `idle_park` and keeps `contention_half_life`.
    pub fn fixed(
        idle_park: Duration,
        contention_half_life: Duration,
    ) -> Result<Self, ZeroHalfLife> {
        Self::new(
            idle_park,
            contention_half_life,
            Arc::new(super::FixedTiming),
        )
    }

    /// This policy with a different [`unproductive_park`](Self::unproductive_park).
    pub fn with_unproductive_park(self, unproductive_park: Duration) -> Self {
        Self {
            unproductive_park,
            ..self
        }
    }

    /// The park of an idle worker where the function has no answer.
    pub fn idle_park(&self) -> Duration {
        self.idle_park
    }

    /// The half-life of acquisition contention history where the function has no answer, and
    /// until it first installs one.
    pub fn contention_half_life(&self) -> Duration {
        self.contention_half_life
    }

    /// Picks the idle park and contention half-life.
    pub fn function(&self) -> &Arc<dyn TimingProvider> {
        &self.function
    }

    /// The park of a worker that processed nothing while none of its upstream handles emitted work
    /// on their last attempt. Longer parks save power at the cost of the time a parked worker takes
    /// to notice new work. Zero disables the park.
    pub fn unproductive_park(&self) -> Duration {
        self.unproductive_park
    }
}

impl Default for IdlePolicy {
    /// The default parks and half-life, picked by
    /// [`IdleTimingFunction::cache_local_bounded`].
    fn default() -> Self {
        Self::new(
            Self::DEFAULT_IDLE_PARK,
            Self::DEFAULT_CONTENTION_HALF_LIFE,
            Arc::new(IdleTimingFunction::cache_local_bounded()),
        )
        .expect("the default half-life is positive")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults() {
        let policy = IdlePolicy::default();
        assert_eq!(policy.idle_park(), Duration::from_nanos(15_000));
        assert_eq!(policy.unproductive_park(), Duration::from_nanos(15_000));
        assert_eq!(
            policy.contention_half_life(),
            Duration::from_nanos(1_000_000)
        );
        let park = policy
            .function()
            .park(0.5, 2.0, 8f64.exp_m1(), Duration::ZERO);
        let built_in = IdleTimingFunction::cache_local_bounded();
        assert_eq!(park, built_in.park(0.5, 2.0, 8f64.exp_m1(), Duration::ZERO));
    }

    #[test]
    fn a_fixed_policy_answers_with_its_own_timing() {
        let policy =
            IdlePolicy::fixed(Duration::from_micros(3), Duration::from_micros(40)).unwrap();
        assert_eq!(policy.idle_park(), Duration::from_micros(3));
        assert_eq!(policy.contention_half_life(), Duration::from_micros(40));
        assert_eq!(
            policy.unproductive_park(),
            IdlePolicy::DEFAULT_UNPRODUCTIVE_PARK
        );
        let fallback = Duration::from_nanos(9);
        assert_eq!(policy.function().park(0.5, 1.0, 10.0, fallback), fallback);
        assert_eq!(
            policy.function().half_life(0.5, 1.0, 10.0, fallback),
            fallback
        );
    }

    #[test]
    fn the_half_life_must_be_positive() {
        assert_eq!(
            IdlePolicy::fixed(Duration::ZERO, Duration::ZERO).unwrap_err(),
            ZeroHalfLife
        );
        assert_eq!(
            IdlePolicy::new(
                Duration::ZERO,
                Duration::ZERO,
                Arc::new(IdleTimingFunction::default())
            )
            .unwrap_err()
            .to_string(),
            "the contention half-life must be positive"
        );
        assert!(IdlePolicy::fixed(Duration::ZERO, Duration::from_nanos(1)).is_ok());
    }

    #[test]
    fn the_unproductive_park_can_be_changed_or_disabled() {
        let policy = IdlePolicy::default().with_unproductive_park(Duration::ZERO);
        assert_eq!(policy.unproductive_park(), Duration::ZERO);
        assert_eq!(policy.idle_park(), IdlePolicy::DEFAULT_IDLE_PARK);
    }
}
