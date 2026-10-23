//! Measuring what a fixed amount of CPU work costs on this machine.

use std::hint::black_box;
use std::time::{Duration, Instant};

/// How many timed runs [`benchmark`] takes the median of.
pub const BENCHMARK_RUNS: usize = 1001;

/// A fixed amount of integer work: `cycles` rounds of mixing, starting from `seed`.
pub fn cpu_work(seed: u64, cycles: u32) -> u64 {
    let mut value = seed;
    for i in 0..u64::from(cycles) {
        value ^= 0x9e37_79b9_7f4a_7c15_u64.wrapping_add(i);
        value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9).rotate_left(17);
    }
    value
}

/// The median time [`cpu_work`] takes for `cycles` rounds, over [`BENCHMARK_RUNS`] runs. Zero for
/// zero cycles.
pub fn benchmark(cycles: u32) -> Duration {
    if cycles == 0 {
        return Duration::ZERO;
    }
    let mut value = black_box(euhedral_hashing::hasher_api::BASE_SEED);
    let mut times: Vec<Duration> = (0..BENCHMARK_RUNS)
        .map(|_| {
            let start = Instant::now();
            value = black_box(cpu_work(black_box(value), black_box(cycles)));
            start.elapsed()
        })
        .collect();
    times.sort_unstable();
    times[times.len() / 2]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_work_is_a_fixed_function() {
        assert_eq!(cpu_work(0x9e37_79b9_7f4a_7c15, 3), 0x1980_fcfa_62af_c03d);
        assert_eq!(cpu_work(0, 1), 0x2388_d90c_e85b_ace8);
        assert_eq!(cpu_work(42, 1000), 0x4240_3635_b1dc_bc7a);
    }

    #[test]
    fn no_cycles_is_no_work() {
        assert_eq!(cpu_work(1234, 0), 1234);
        assert_eq!(benchmark(0), Duration::ZERO);
    }

    #[test]
    fn more_work_takes_longer() {
        let small = benchmark(100);
        let large = benchmark(100_000);
        assert!(large > small, "{large:?} should exceed {small:?}");
    }
}
