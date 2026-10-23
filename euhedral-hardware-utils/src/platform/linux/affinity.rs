//! Linux thread affinity through `sched_getaffinity`/`sched_setaffinity`, `sched_getcpu` and
//! per-thread timer slack (upstream `LinuxAffinity` and its native library).

use crate::affinity::{AffinityCapability, AffinityProvider, masks::MAX_BITS};
use crate::util::CpuSet;
use std::sync::atomic::{AtomicUsize, Ordering};

const LONG_BITS: usize = libc::c_ulong::BITS as usize;

/// Exact affinity for the calling thread.
#[derive(Debug)]
pub(crate) struct LinuxAffinity {
    /// The kernel's CPU-mask size in bits, learned from the first successful capture.
    mask_bits: AtomicUsize,
}

impl LinuxAffinity {
    /// `span` seeds the first capture buffer; a larger kernel mask is found by doubling.
    pub(crate) fn new(span: usize) -> Self {
        Self {
            mask_bits: AtomicUsize::new(span.max(1024).next_multiple_of(LONG_BITS)),
        }
    }

    fn to_longs(mask: &[u64]) -> Vec<libc::c_ulong> {
        let cpus = CpuSet::from_words(mask);
        let mut longs = vec![0 as libc::c_ulong; cpus.span().div_ceil(LONG_BITS).max(1)];
        for cpu in cpus.iter() {
            longs[cpu / LONG_BITS] |= 1 << (cpu % LONG_BITS);
        }
        longs
    }

    fn set(mask: &[u64]) -> bool {
        let longs = Self::to_longs(mask);
        // SAFETY: the pointer and byte length describe `longs`, which outlives the call.
        let result = unsafe {
            libc::syscall(
                libc::SYS_sched_setaffinity,
                0,
                longs.len() * std::mem::size_of::<libc::c_ulong>(),
                longs.as_ptr(),
            )
        };
        result == 0
    }
}

impl AffinityProvider for LinuxAffinity {
    fn capability(&self) -> AffinityCapability {
        AffinityCapability::Exact
    }

    fn current_cpu(&self) -> Option<usize> {
        // SAFETY: sched_getcpu has no preconditions.
        let cpu = unsafe { libc::sched_getcpu() };
        usize::try_from(cpu).ok()
    }

    /// The kernel rejects a buffer smaller than its own mask with `EINVAL`, so the buffer doubles
    /// until it fits.
    fn capture_affinity(&self) -> Option<Vec<u64>> {
        let mut bits = self.mask_bits.load(Ordering::Relaxed);
        loop {
            let mut longs = vec![0 as libc::c_ulong; bits / LONG_BITS];
            // SAFETY: the pointer and byte length describe `longs`, which outlives the call.
            let copied = unsafe {
                libc::syscall(
                    libc::SYS_sched_getaffinity,
                    0,
                    longs.len() * std::mem::size_of::<libc::c_ulong>(),
                    longs.as_mut_ptr(),
                )
            };
            if copied > 0 {
                self.mask_bits.store(bits, Ordering::Relaxed);
                let mut words = Vec::new();
                for (index, &long) in longs.iter().enumerate() {
                    for bit in (0..LONG_BITS).filter(|bit| long & (1 << bit) != 0) {
                        crate::util::cpu_set::set_bit(&mut words, index * LONG_BITS + bit);
                    }
                }
                return Some(CpuSet::from_words(&words).words().to_vec())
                    .filter(|words| !words.is_empty());
            }
            let error = std::io::Error::last_os_error();
            if error.raw_os_error() != Some(libc::EINVAL) || bits >= MAX_BITS {
                log::debug!("sched_getaffinity failed: {error}");
                return None;
            }
            bits *= 2;
        }
    }

    fn apply_exact(&self, mask: &[u64]) -> bool {
        Self::set(mask)
    }

    fn restore_exact(&self, mask: &[u64]) -> bool {
        Self::set(mask)
    }

    /// Sets the calling thread's timer slack; zero would restore the default, so it is raised to 1 ns.
    fn set_timer_resolution(&self, nanos: u64) -> bool {
        let slack = libc::c_ulong::try_from(nanos.max(1)).unwrap_or(libc::c_ulong::MAX);
        // SAFETY: PR_SET_TIMERSLACK takes one integer argument and touches no memory.
        unsafe { libc::prctl(libc::PR_SET_TIMERSLACK, slack, 0, 0, 0) == 0 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn captures_applies_and_restores_on_a_scratch_thread() {
        std::thread::spawn(|| {
            let affinity = LinuxAffinity::new(1);
            let original = affinity
                .capture_affinity()
                .expect("a thread always has an affinity mask");
            let cpus = CpuSet::from_words(&original);
            let first = cpus.first().unwrap();
            assert!(affinity.apply_exact(CpuSet::range(first, first + 1).words()));
            assert_eq!(
                affinity.capture_affinity(),
                Some(CpuSet::range(first, first + 1).words().to_vec())
            );
            assert_eq!(affinity.current_cpu(), Some(first));
            assert!(affinity.restore_exact(&original));
            assert_eq!(affinity.capture_affinity(), Some(original));
            assert!(affinity.set_timer_resolution(50_000));
        })
        .join()
        .unwrap();
    }

    #[test]
    fn long_conversion_places_every_bit() {
        let longs = LinuxAffinity::to_longs(&[1, 1 << 63, 1]);
        let bits: Vec<usize> = longs
            .iter()
            .enumerate()
            .flat_map(|(i, &long)| {
                (0..LONG_BITS)
                    .filter(move |b| long & (1 << b) != 0)
                    .map(move |b| i * LONG_BITS + b)
            })
            .collect();
        assert_eq!(bits, [0, 127, 128]);
    }
}
