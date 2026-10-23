use std::error::Error;
use std::fmt;
use std::time::Duration;

/// Picks how long an idle worker parks and how fast its contention history fades, from what the
/// worker sees: the acquisition contention of its upstreams in `[0, 1]`, its productive upstream
/// handles per registered worker, and the smoothed cost of a frame body in nanoseconds.
pub trait TimingProvider: Send + Sync + fmt::Debug {
    /// How long to park, or `fallback` where the provider has no answer.
    fn park(
        &self,
        contention: f64,
        productive_ratio: f64,
        body_nanos: f64,
        fallback: Duration,
    ) -> Duration;

    /// The contention half-life to install, or `fallback` where the provider has no answer. Never
    /// zero.
    fn half_life(
        &self,
        contention: f64,
        productive_ratio: f64,
        body_nanos: f64,
        fallback: Duration,
    ) -> Duration;
}

/// Always answers with the fallback: the policy's own idle park and contention half-life.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FixedTiming;

impl TimingProvider for FixedTiming {
    fn park(&self, _: f64, _: f64, _: f64, fallback: Duration) -> Duration {
        fallback
    }

    fn half_life(&self, _: f64, _: f64, _: f64, fallback: Duration) -> Duration {
        fallback
    }
}

/// The coefficients of one output of a [`TimingModel`], over the normalized features: contention,
/// productive ratio and `ln(1 + body nanoseconds)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TimingCoefficients {
    /// The intercept, then one weight per feature.
    Affine([f64; 4]),
    /// The affine terms, then the pairwise products: contention × ratio, contention × body, and
    /// ratio × body.
    Pairwise([f64; 7]),
}

impl TimingCoefficients {
    fn as_slice(&self) -> &[f64] {
        match self {
            Self::Affine(coefficients) => coefficients,
            Self::Pairwise(coefficients) => coefficients,
        }
    }
}

/// A fitted model of idle timing: each output is `reference · exp(linear model of the normalized
/// features)`, clamped to its bounds. Features are normalized as `(x - mean) / scale` and must lie
/// within `[support_min, support_max]`; outside it the model answers with the fallback.
///
/// Feature order in every array: contention, productive ratio, `ln(1 + body nanoseconds)`.
#[derive(Clone, Debug, PartialEq)]
pub struct TimingModel {
    /// Names the coordinate system the coefficients were fitted in.
    pub normalization_version: String,
    pub means: [f64; 3],
    pub scales: [f64; 3],
    pub support_min: [f64; 3],
    pub support_max: [f64; 3],
    pub park: TimingCoefficients,
    pub half_life: TimingCoefficients,
    pub park_reference_ns: u64,
    pub half_life_reference_ns: u64,
    pub park_min_ns: u64,
    pub park_max_ns: u64,
    pub half_life_min_ns: u64,
    pub half_life_max_ns: u64,
}

/// Why a [`TimingModel`] was rejected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InvalidTimingModel {
    /// The normalization version is blank.
    UnnamedVersion,
    /// A mean, scale, support bound or coefficient is not finite.
    NonFinite,
    /// One output is affine and the other pairwise.
    MixedBases,
    /// A scale is not positive, a support is negative or inverted, a normalized support bound is
    /// not finite, or contention may exceed one.
    InvalidSupport,
    /// The minimum half-life or the park reference is zero, or a reference lies outside its
    /// bounds.
    InvalidBounds,
    /// Some input within the support could overflow the exponent.
    UnboundedEffect,
}

impl fmt::Display for InvalidTimingModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::UnnamedVersion => "the normalization version must be named",
            Self::NonFinite => "means, scales, supports and coefficients must be finite",
            Self::MixedBases => "both outputs must use the same basis",
            Self::InvalidSupport => "invalid normalization or support",
            Self::InvalidBounds => "invalid timing references or bounds",
            Self::UnboundedEffect => "the log-time effect over the support is not finite",
        })
    }
}

impl Error for InvalidTimingModel {}

/// A [`TimingProvider`] backed by a validated [`TimingModel`].
#[derive(Clone, Debug, PartialEq)]
pub struct IdleTimingFunction {
    model: TimingModel,
}

impl IdleTimingFunction {
    pub fn new(model: TimingModel) -> Result<Self, InvalidTimingModel> {
        if model.normalization_version.trim().is_empty() {
            return Err(InvalidTimingModel::UnnamedVersion);
        }
        let features = [
            model.means,
            model.scales,
            model.support_min,
            model.support_max,
        ];
        let coefficients = [model.park.as_slice(), model.half_life.as_slice()];
        if !features.iter().flatten().all(|value| value.is_finite())
            || !coefficients
                .iter()
                .copied()
                .flatten()
                .all(|c| c.is_finite())
        {
            return Err(InvalidTimingModel::NonFinite);
        }
        if coefficients[0].len() != coefficients[1].len() {
            return Err(InvalidTimingModel::MixedBases);
        }
        for i in 0..3 {
            let (min, max) = (model.support_min[i], model.support_max[i]);
            let normalized = |x: f64| (x - model.means[i]) / model.scales[i];
            if model.scales[i] <= 0.0
                || min < 0.0
                || max < min
                || !normalized(min).is_finite()
                || !normalized(max).is_finite()
            {
                return Err(InvalidTimingModel::InvalidSupport);
            }
        }
        if model.support_max[0] > 1.0 {
            return Err(InvalidTimingModel::InvalidSupport);
        }
        if model.half_life_min_ns == 0
            || model.park_reference_ns == 0
            || !(model.park_min_ns..=model.park_max_ns).contains(&model.park_reference_ns)
            || !(model.half_life_min_ns..=model.half_life_max_ns)
                .contains(&model.half_life_reference_ns)
        {
            return Err(InvalidTimingModel::InvalidBounds);
        }
        let function = Self { model };
        if !function.max_effect(&function.model.park).is_finite()
            || !function.max_effect(&function.model.half_life).is_finite()
        {
            return Err(InvalidTimingModel::UnboundedEffect);
        }
        Ok(function)
    }

    /// The `cache-local-bounded-v1` model, calibrated on scarce sources.
    pub fn cache_local_bounded() -> Self {
        Self::new(TimingModel {
            normalization_version: "cache-local-bounded-v1".to_string(),
            means: [0.5, 2.0, 8.0],
            scales: [0.5, 2.0, 8.0],
            support_min: [0.0, 0.0, 0.0],
            support_max: [1.0, 4.0, 16.0],
            park: TimingCoefficients::Pairwise([
                1.8063635377779264,
                -5.551115123125783e-16,
                -1.1782901489929007,
                -0.0885253008972973,
                -3.2959746043559335e-17,
                -1.0787811616230769e-16,
                -1.3530843112619095e-16,
            ]),
            half_life: TimingCoefficients::Pairwise([
                -0.7202320591072836,
                2.220446049250313e-16,
                0.4005695409857165,
                3.642919299551295e-17,
                -8.239936510889834e-18,
                -2.888721163316066e-17,
                -4.0766001685454967e-17,
            ]),
            park_reference_ns: 15_000,
            half_life_reference_ns: 1_000_000,
            park_min_ns: 15_000,
            park_max_ns: 814_375,
            half_life_min_ns: 250_000,
            half_life_max_ns: 2_000_000,
        })
        .expect("the built-in model is valid")
    }

    pub fn model(&self) -> &TimingModel {
        &self.model
    }

    /// The largest the exponent's terms can add up to over the support.
    fn max_effect(&self, coefficients: &TimingCoefficients) -> f64 {
        let reach: [f64; 3] = std::array::from_fn(|i| {
            let normalized = |x: f64| ((x - self.model.means[i]) / self.model.scales[i]).abs();
            normalized(self.model.support_min[i]).max(normalized(self.model.support_max[i]))
        });
        let weights = coefficients.as_slice();
        let mut effect =
            weights[0].abs() + (0..3).map(|i| weights[i + 1].abs() * reach[i]).sum::<f64>();
        if let TimingCoefficients::Pairwise(_) = coefficients {
            effect += weights[4].abs() * reach[0] * reach[1]
                + weights[5].abs() * reach[0] * reach[2]
                + weights[6].abs() * reach[1] * reach[2];
        }
        effect
    }

    #[allow(clippy::too_many_arguments)]
    fn output(
        &self,
        coefficients: &TimingCoefficients,
        reference_ns: u64,
        min_ns: u64,
        max_ns: u64,
        contention: f64,
        productive_ratio: f64,
        body_nanos: f64,
        fallback: Duration,
    ) -> Duration {
        let features = [contention, productive_ratio, body_nanos.ln_1p()];
        let model = &self.model;
        let supported = (0..3)
            .all(|i| features[i] >= model.support_min[i] && features[i] <= model.support_max[i]);
        if !supported {
            return fallback;
        }
        let z: [f64; 3] = std::array::from_fn(|i| (features[i] - model.means[i]) / model.scales[i]);
        let weights = coefficients.as_slice();
        let mut exponent = weights[0] + weights[1] * z[0] + weights[2] * z[1] + weights[3] * z[2];
        if let TimingCoefficients::Pairwise(_) = coefficients {
            exponent += weights[4] * (z[0] * z[1])
                + weights[5] * (z[0] * z[2])
                + weights[6] * (z[1] * z[2]);
        }
        let bounded = (reference_ns as f64 * exponent.exp()).clamp(min_ns as f64, max_ns as f64);
        Duration::from_nanos((bounded.round() as u64).clamp(min_ns, max_ns))
    }
}

impl Default for IdleTimingFunction {
    /// [`cache_local_bounded`](Self::cache_local_bounded).
    fn default() -> Self {
        Self::cache_local_bounded()
    }
}

impl TimingProvider for IdleTimingFunction {
    fn park(
        &self,
        contention: f64,
        productive_ratio: f64,
        body_nanos: f64,
        fallback: Duration,
    ) -> Duration {
        let model = &self.model;
        self.output(
            &model.park,
            model.park_reference_ns,
            model.park_min_ns,
            model.park_max_ns,
            contention,
            productive_ratio,
            body_nanos,
            fallback,
        )
    }

    fn half_life(
        &self,
        contention: f64,
        productive_ratio: f64,
        body_nanos: f64,
        fallback: Duration,
    ) -> Duration {
        let model = &self.model;
        self.output(
            &model.half_life,
            model.half_life_reference_ns,
            model.half_life_min_ns,
            model.half_life_max_ns,
            contention,
            productive_ratio,
            body_nanos,
            fallback,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FALLBACK: Duration = Duration::from_nanos(7);

    /// An affine model over unit features with a reference of 1000 ns in `[10, 100_000]`.
    fn affine(park: [f64; 4]) -> TimingModel {
        TimingModel {
            normalization_version: "test".to_string(),
            means: [0.0; 3],
            scales: [1.0; 3],
            support_min: [0.0; 3],
            support_max: [1.0, 10.0, 20.0],
            park: TimingCoefficients::Affine(park),
            half_life: TimingCoefficients::Affine([0.0; 4]),
            park_reference_ns: 1000,
            half_life_reference_ns: 1000,
            park_min_ns: 10,
            park_max_ns: 100_000,
            half_life_min_ns: 10,
            half_life_max_ns: 100_000,
        }
    }

    fn park(model: TimingModel, contention: f64, ratio: f64, body: f64) -> Duration {
        IdleTimingFunction::new(model)
            .unwrap()
            .park(contention, ratio, body, FALLBACK)
    }

    #[test]
    fn the_built_in_model_answers_at_its_means() {
        let function = IdleTimingFunction::default();
        assert_eq!(function, IdleTimingFunction::cache_local_bounded());
        assert_eq!(
            function.model().normalization_version,
            "cache-local-bounded-v1"
        );
        // At the means every normalized feature is zero, leaving the intercepts.
        let body = 8f64.exp_m1();
        let park = (15_000.0 * 1.8063635377779264f64.exp()).round() as u64;
        let half_life = (1_000_000.0 * (-0.7202320591072836f64).exp()).round() as u64;
        assert_eq!(
            function.park(0.5, 2.0, body, FALLBACK).as_nanos() as u64,
            park
        );
        assert_eq!(
            function.half_life(0.5, 2.0, body, FALLBACK).as_nanos() as u64,
            half_life
        );
    }

    #[test]
    fn the_built_in_model_parks_less_as_productive_handles_grow() {
        let function = IdleTimingFunction::default();
        let scarce = function.park(0.5, 0.0, 1000.0, FALLBACK);
        let plenty = function.park(0.5, 4.0, 1000.0, FALLBACK);
        assert!(scarce > plenty, "{scarce:?} should exceed {plenty:?}");
        assert!(plenty >= Duration::from_nanos(15_000));
        assert!(scarce <= Duration::from_nanos(814_375));
    }

    #[test]
    fn affine_terms_weigh_each_feature() {
        assert_eq!(
            park(affine([0.0; 4]), 0.0, 0.0, 0.0),
            Duration::from_nanos(1000)
        );
        let weighted = |weights, c, p, b| park(affine(weights), c, p, b).as_nanos() as f64;
        let ln2 = std::f64::consts::LN_2;
        assert_eq!(weighted([ln2, 0.0, 0.0, 0.0], 0.0, 0.0, 0.0), 2000.0);
        assert_eq!(weighted([0.0, ln2, 0.0, 0.0], 1.0, 0.0, 0.0), 2000.0);
        assert_eq!(weighted([0.0, 0.0, ln2, 0.0], 0.0, 2.0, 0.0), 4000.0);
        let body = 3f64.exp_m1();
        assert_eq!(weighted([0.0, 0.0, 0.0, ln2], 0.0, 0.0, body), 8000.0);
    }

    #[test]
    fn features_are_normalized_by_mean_and_scale() {
        let mut model = affine([0.0, std::f64::consts::LN_2, 0.0, 0.0]);
        model.means = [0.5, 0.0, 0.0];
        model.scales = [0.25, 1.0, 1.0];
        // (1 - 0.5) / 0.25 = 2 halvings' worth of doubling.
        assert_eq!(park(model, 1.0, 0.0, 0.0), Duration::from_nanos(4000));
    }

    #[test]
    fn pairwise_terms_weigh_each_product() {
        let ln2 = std::f64::consts::LN_2;
        let pairwise = |weight_index: usize, c, p, b| {
            let mut weights = [0.0; 7];
            weights[weight_index] = ln2;
            let mut model = affine([0.0; 4]);
            model.park = TimingCoefficients::Pairwise(weights);
            model.half_life = TimingCoefficients::Pairwise([0.0; 7]);
            park(model, c, p, b).as_nanos()
        };
        let body = 3f64.exp_m1();
        assert_eq!(pairwise(4, 1.0, 2.0, 0.0), 4000);
        assert_eq!(pairwise(4, 1.0, 0.0, body), 1000);
        assert_eq!(pairwise(5, 1.0, 0.0, body), 8000);
        assert_eq!(pairwise(5, 0.0, 2.0, body), 1000);
        assert_eq!(pairwise(6, 0.0, 2.0, body), 64_000);
        assert_eq!(pairwise(6, 1.0, 2.0, 0.0), 1000);
    }

    #[test]
    fn outputs_are_clamped_to_their_bounds() {
        assert_eq!(
            park(affine([20.0, 0.0, 0.0, 0.0]), 0.0, 0.0, 0.0),
            Duration::from_nanos(100_000)
        );
        assert_eq!(
            park(affine([-20.0, 0.0, 0.0, 0.0]), 0.0, 0.0, 0.0),
            Duration::from_nanos(10)
        );
        let mut model = affine([0.0; 4]);
        model.half_life = TimingCoefficients::Affine([-20.0, 0.0, 0.0, 0.0]);
        let function = IdleTimingFunction::new(model).unwrap();
        assert_eq!(
            function.half_life(0.0, 0.0, 0.0, FALLBACK),
            Duration::from_nanos(10)
        );
    }

    #[test]
    fn inputs_outside_the_support_fall_back() {
        let function = IdleTimingFunction::new(affine([0.0; 4])).unwrap();
        let answer = |c, p, b| function.park(c, p, b, FALLBACK);
        assert_eq!(
            answer(1.0, 10.0, 19.5f64.exp_m1()),
            Duration::from_nanos(1000)
        );
        assert_eq!(answer(0.0, 0.0, -0.0), Duration::from_nanos(1000));
        for (c, p, b) in [
            (1.01, 0.0, 0.0),
            (-0.01, 0.0, 0.0),
            (0.0, 10.01, 0.0),
            (0.0, -0.01, 0.0),
            (0.0, 0.0, 21f64.exp_m1()),
            (0.0, 0.0, -0.5),
            (0.0, 0.0, -2.0),
            (f64::NAN, 0.0, 0.0),
            (0.0, f64::NAN, 0.0),
            (0.0, f64::INFINITY, 0.0),
            (0.0, 0.0, f64::NAN),
            (0.0, 0.0, f64::INFINITY),
        ] {
            assert_eq!(answer(c, p, b), FALLBACK, "{c} {p} {b}");
            assert_eq!(function.half_life(c, p, b, FALLBACK), FALLBACK);
        }
    }

    #[test]
    fn invalid_models_are_rejected() {
        let reject = |change: &dyn Fn(&mut TimingModel)| {
            let mut model = affine([0.0; 4]);
            change(&mut model);
            IdleTimingFunction::new(model).unwrap_err()
        };
        use InvalidTimingModel::*;
        assert_eq!(
            reject(&|m| m.normalization_version = " ".into()),
            UnnamedVersion
        );
        assert_eq!(reject(&|m| m.means[1] = f64::NAN), NonFinite);
        assert_eq!(reject(&|m| m.scales[2] = f64::INFINITY), NonFinite);
        assert_eq!(reject(&|m| m.support_min[0] = f64::NAN), NonFinite);
        assert_eq!(reject(&|m| m.support_max[1] = f64::INFINITY), NonFinite);
        assert_eq!(
            reject(&|m| m.park = TimingCoefficients::Affine([0.0, f64::NAN, 0.0, 0.0])),
            NonFinite
        );
        assert_eq!(
            reject(&|m| m.half_life = TimingCoefficients::Affine([0.0, 0.0, 0.0, f64::INFINITY])),
            NonFinite
        );
        assert_eq!(
            reject(&|m| m.half_life = TimingCoefficients::Pairwise([0.0; 7])),
            MixedBases
        );
        assert_eq!(reject(&|m| m.scales[0] = 0.0), InvalidSupport);
        assert_eq!(reject(&|m| m.scales[1] = -1.0), InvalidSupport);
        assert_eq!(reject(&|m| m.support_min[2] = -0.5), InvalidSupport);
        assert_eq!(
            reject(&|m| {
                m.support_min[1] = 5.0;
                m.support_max[1] = 4.0;
            }),
            InvalidSupport
        );
        assert_eq!(
            reject(&|m| {
                m.scales[2] = 1e-308;
                m.support_max[2] = 1e10;
            }),
            InvalidSupport
        );
        assert_eq!(reject(&|m| m.support_max[0] = 1.5), InvalidSupport);
        assert_eq!(reject(&|m| m.half_life_min_ns = 0), InvalidBounds);
        assert_eq!(
            reject(&|m| {
                m.park_min_ns = 0;
                m.park_reference_ns = 0;
            }),
            InvalidBounds
        );
        assert_eq!(reject(&|m| m.half_life_reference_ns = 0), InvalidBounds);
        assert_eq!(reject(&|m| m.park_reference_ns = 5), InvalidBounds);
        assert_eq!(reject(&|m| m.park_reference_ns = 100_001), InvalidBounds);
        assert_eq!(reject(&|m| m.half_life_reference_ns = 9), InvalidBounds);
        assert_eq!(
            reject(&|m| m.half_life_reference_ns = 100_001),
            InvalidBounds
        );
        assert_eq!(
            reject(&|m| m.park = TimingCoefficients::Affine([0.0, f64::MAX, f64::MAX, 0.0])),
            UnboundedEffect
        );
        assert_eq!(
            reject(&|m| {
                m.park = TimingCoefficients::Pairwise([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, f64::MAX]);
                m.half_life = TimingCoefficients::Pairwise([0.0; 7]);
            }),
            UnboundedEffect
        );
        assert_eq!(
            reject(&|m| m.half_life = TimingCoefficients::Affine([f64::MAX, f64::MAX, 0.0, 0.0])),
            UnboundedEffect
        );
        // The support's far end from the mean bounds the effect, on either side.
        assert_eq!(
            reject(&|m| {
                m.means[1] = 10.0;
                m.park = TimingCoefficients::Affine([0.0, 0.0, f64::MAX / 5.0, 0.0]);
            }),
            UnboundedEffect
        );
        for product in 4..7 {
            assert_eq!(
                reject(&|m| {
                    let mut weights = [0.0; 7];
                    weights[product] = f64::MAX / 2.0;
                    m.park = TimingCoefficients::Pairwise(weights);
                    m.half_life = TimingCoefficients::Pairwise([0.0; 7]);
                }),
                UnboundedEffect,
                "product {product}"
            );
        }
        assert_eq!(
            reject(&|m| {
                m.means[1] = 1e300;
                m.scales[1] = 1e-10;
                m.support_max[1] = 1e300;
            }),
            InvalidSupport
        );
        assert_eq!(
            UnboundedEffect.to_string(),
            "the log-time effect over the support is not finite"
        );
    }

    #[test]
    fn a_point_support_and_a_reference_at_its_bounds_are_valid() {
        let mut model = affine([0.0; 4]);
        model.support_min = [0.5, 1.0, 0.0];
        model.support_max = [0.5, 1.0, 0.0];
        model.park_min_ns = 1000;
        model.half_life_max_ns = 1000;
        let function = IdleTimingFunction::new(model).unwrap();
        assert_eq!(
            function.park(0.5, 1.0, 0.0, FALLBACK),
            Duration::from_nanos(1000)
        );
    }

    #[test]
    fn fixed_timing_always_falls_back() {
        assert_eq!(FixedTiming.park(0.5, 1.0, 100.0, FALLBACK), FALLBACK);
        assert_eq!(FixedTiming.half_life(0.5, 1.0, 100.0, FALLBACK), FALLBACK);
    }
}
