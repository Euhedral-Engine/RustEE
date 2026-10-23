/// The high 64 bits of the 128-bit product of `a` and `b`.
///
/// Maps a hash onto `0..n` without division or bias: `unsigned_multiply_high(hash, n)`.
pub fn unsigned_multiply_high(a: u64, b: u64) -> u64 {
    ((u128::from(a) * u128::from(b)) >> 64) as u64
}

/// An exponentially weighted moving average step: moves `current` toward `next` by `alpha`.
pub fn ewma(current: f64, next: f64, alpha: f64) -> f64 {
    current * (1.0 - alpha) + next * alpha
}

/// [`ewma`] on integers, truncating the result toward zero (saturating at the `i64` range).
pub fn ewma_i64(current: i64, next: i64, alpha: f64) -> i64 {
    ewma(current as f64, next as f64, alpha) as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multiply_high_matches_the_wide_product() {
        assert_eq!(unsigned_multiply_high(0, u64::MAX), 0);
        assert_eq!(unsigned_multiply_high(1, u64::MAX), 0);
        assert_eq!(unsigned_multiply_high(1 << 63, 2), 1);
        assert_eq!(unsigned_multiply_high(1 << 32, 1 << 32), 1);
        assert_eq!(unsigned_multiply_high(u64::MAX, u64::MAX), u64::MAX - 1);
        assert_eq!(
            unsigned_multiply_high(0x9e37_79b9_7f4a_7c15, 0xbf58_476d_1ce4_e5b9),
            0x7641_f308_0ff9_2329
        );
    }

    #[test]
    fn multiply_high_maps_hashes_onto_a_range() {
        for hash in [0, 1, u64::MAX / 3, u64::MAX / 2, u64::MAX] {
            assert!(unsigned_multiply_high(hash, 7) < 7);
        }
        assert_eq!(unsigned_multiply_high(u64::MAX, 7), 6);
    }

    #[test]
    fn ewma_moves_toward_the_sample_by_alpha() {
        assert_eq!(ewma(10.0, 20.0, 0.0), 10.0);
        assert_eq!(ewma(10.0, 20.0, 1.0), 20.0);
        assert_eq!(ewma(10.0, 20.0, 0.25), 12.5);
    }

    #[test]
    fn integer_ewma_truncates_and_saturates() {
        assert_eq!(ewma_i64(10, 13, 0.5), 11);
        assert_eq!(ewma_i64(-10, -13, 0.5), -11);
        assert_eq!(ewma_i64(i64::MAX, i64::MAX, 0.5), i64::MAX);
    }
}
