//! macOS thread affinity tags (upstream `MacosAffinity`).
//!
//! `THREAD_AFFINITY_POLICY` asks the scheduler to keep threads with equal tags near each other; it
//! never binds a thread to a CPU, so this is a locality hint. CPU `n` maps to tag `n + 1`, and tag `0`
//! clears the preference. The current CPU is not observable, and there is no per-thread timer
//! granularity to set.

use crate::affinity::{AffinityCapability, AffinityProvider};

#[derive(Debug, Default)]
pub(crate) struct MacosAffinity;

fn set_tag(tag: libc::integer_t) -> bool {
    let mut policy = libc::thread_affinity_policy { affinity_tag: tag };
    // SAFETY: the policy pointer and count describe `policy`, and the port is the calling thread's.
    let result = unsafe {
        libc::thread_policy_set(
            libc::pthread_mach_thread_np(libc::pthread_self()),
            libc::THREAD_AFFINITY_POLICY as libc::thread_policy_flavor_t,
            (&raw mut policy).cast(),
            libc::THREAD_AFFINITY_POLICY_COUNT,
        )
    };
    result == libc::KERN_SUCCESS
}

impl AffinityProvider for MacosAffinity {
    fn capability(&self) -> AffinityCapability {
        AffinityCapability::LocalityHint
    }

    fn locality_for_cpu(&self, cpu: usize) -> Option<u32> {
        u32::try_from(cpu.checked_add(1)?)
            .ok()
            .filter(|&tag| libc::integer_t::try_from(tag).is_ok())
    }

    fn apply_locality(&self, locality: u32) -> bool {
        locality != 0 && libc::integer_t::try_from(locality).is_ok_and(set_tag)
    }

    fn release_locality(&self) -> bool {
        set_tag(0)
    }

    fn set_timer_resolution(&self, _nanos: u64) -> bool {
        false
    }
}
