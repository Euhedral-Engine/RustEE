//! The common affinity policy over a platform provider (upstream `AffinityController`).
//!
//! A thread's first successful request takes a *lease*: an exact provider's original mask, or a note
//! that a locality hint is set. [`AffinityController::release_affinity`] undoes that first state, so
//! nested requests all release to the thread's original binding.

use super::masks::{self, MAX_BITS};
use super::{AffinityCapability, AffinityProvider};
use crate::util::CpuSet;
use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Debug)]
enum Lease {
    Exact(Vec<u64>),
    Locality,
}

thread_local! {
    /// Per-thread leases, keyed by controller id.
    static LEASES: RefCell<HashMap<u64, Lease>> = RefCell::new(HashMap::new());
    /// Per-thread managed-owner stacks of `(token, cpu)`, keyed by controller id.
    static OWNERS: RefCell<HashMap<u64, Vec<(u64, usize)>>> = RefCell::new(HashMap::new());
}

static NEXT_ID: AtomicU64 = AtomicU64::new(0);

/// Why a controller could not be built.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ControllerError {
    /// The span is zero or above [`MAX_BITS`].
    InvalidSpan(usize),
    /// The supported set is empty or reaches the span.
    InvalidSupportedSet,
}

impl fmt::Display for ControllerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ControllerError::InvalidSpan(span) => write!(f, "invalid logical CPU span {span}"),
            ControllerError::InvalidSupportedSet => {
                f.write_str("the supported CPU set is empty or outside its span")
            }
        }
    }
}

impl std::error::Error for ControllerError {}

/// Affinity policy for one topology: validates every request against the supported CPUs and keeps
/// per-thread leases so a release restores the thread's original binding.
pub struct AffinityController {
    id: u64,
    provider: Option<Box<dyn AffinityProvider>>,
    supported: CpuSet,
    span: usize,
    capability: AffinityCapability,
    base_mask: CpuSet,
}

impl fmt::Debug for AffinityController {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AffinityController")
            .field("capability", &self.capability)
            .field("span", &self.span)
            .field("supported", &self.supported)
            .field("base_mask", &self.base_mask)
            .finish_non_exhaustive()
    }
}

impl AffinityController {
    /// `supported` holds the logical CPU ids requests may name, `span` bounds them (holes included).
    /// An exact provider whose mask cannot be captured, or shares no CPU with `supported`, is treated
    /// as unsupported.
    pub fn new(
        provider: Option<Box<dyn AffinityProvider>>,
        supported: &CpuSet,
        span: usize,
    ) -> Result<Self, ControllerError> {
        if span == 0 || span > MAX_BITS {
            return Err(ControllerError::InvalidSpan(span));
        }
        if supported.is_empty() || supported.span() > span {
            return Err(ControllerError::InvalidSupportedSet);
        }
        let mut controller = Self {
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            provider: None,
            supported: supported.clone(),
            span,
            capability: AffinityCapability::Unsupported,
            base_mask: supported.clone(),
        };
        if let Some(provider) = &provider {
            match provider.capability() {
                AffinityCapability::Exact => {
                    // The base is the captured mask's overlap with the supported CPUs.
                    let base = provider
                        .capture_affinity()
                        .map(|mask| CpuSet::from_words(&mask).intersection(supported))
                        .filter(|base| !base.is_empty());
                    if let Some(base) = base {
                        controller.base_mask = base;
                        controller.capability = AffinityCapability::Exact;
                    }
                }
                AffinityCapability::LocalityHint => {
                    controller.capability = AffinityCapability::LocalityHint
                }
                AffinityCapability::Unsupported => {}
            }
        }
        controller.provider = provider;
        Ok(controller)
    }

    pub fn capability(&self) -> AffinityCapability {
        self.capability
    }

    /// The thread's mask captured at construction under [`Exact`](AffinityCapability::Exact),
    /// otherwise the supported CPUs.
    pub fn base_mask(&self) -> &CpuSet {
        &self.base_mask
    }

    pub fn supported(&self) -> &CpuSet {
        &self.supported
    }

    pub fn span(&self) -> usize {
        self.span
    }

    fn canonical(&self, mask: &[u64]) -> Option<Vec<u64>> {
        masks::canonical(mask, self.span, &self.supported)
    }

    fn is_supported(&self, cpu: usize) -> bool {
        cpu < self.span && self.supported.contains(cpu)
    }

    /// Applies a complete little-endian mask. `false` for an invalid mask or an unsupported or
    /// failing provider.
    pub fn set_affinity_mask(&self, mask: &[u64]) -> bool {
        let Some(request) = self.canonical(mask) else {
            return false;
        };
        match self.capability {
            AffinityCapability::Exact => self.apply_exact(&request),
            AffinityCapability::LocalityHint => self.apply_locality(&request),
            AffinityCapability::Unsupported => false,
        }
    }

    pub fn set_affinity_set(&self, cpus: &CpuSet) -> bool {
        !cpus.is_empty() && cpus.span() <= MAX_BITS && self.set_affinity_mask(cpus.words())
    }

    /// `false` when the list is empty or names an unsupported CPU.
    pub fn set_affinity_cpus(&self, cpus: &[usize]) -> bool {
        if cpus.is_empty()
            || cpus.len() > MAX_BITS
            || !cpus.iter().all(|&cpu| self.is_supported(cpu))
        {
            return false;
        }
        self.set_affinity_set(&cpus.iter().copied().collect())
    }

    pub fn set_affinity(&self, cpu: usize) -> bool {
        self.is_supported(cpu) && self.set_affinity_set(&CpuSet::range(cpu, cpu + 1))
    }

    /// Pins to the CPU the thread runs on (or its managed CPU); `false` when that is unknown.
    pub fn set_affinity_current(&self) -> bool {
        self.current_cpu().is_some_and(|cpu| self.set_affinity(cpu))
    }

    /// The CPU the thread runs on when the provider reports it truthfully, else the CPU of the
    /// innermost managed binding on this thread.
    pub fn current_cpu(&self) -> Option<usize> {
        let reported = self
            .provider
            .as_ref()
            .and_then(|provider| provider.current_cpu());
        reported.filter(|&cpu| self.is_supported(cpu)).or_else(|| {
            OWNERS.with(|owners| {
                owners
                    .borrow()
                    .get(&self.id)
                    .and_then(|stack| stack.last())
                    .map(|&(_, cpu)| cpu)
            })
        })
    }

    /// Restores the mask from before this thread's first request, or clears its locality hint.
    pub fn release_affinity(&self) {
        let Some(lease) = LEASES.with(|leases| leases.borrow_mut().remove(&self.id)) else {
            return;
        };
        let Some(provider) = &self.provider else {
            return;
        };
        let restored = match &lease {
            Lease::Exact(mask) => provider.restore_exact(mask),
            Lease::Locality => provider.release_locality(),
        };
        if !restored {
            log::debug!("affinity release failed");
        }
    }

    /// Whether this thread holds a lease to release.
    pub fn has_affinity_lease(&self) -> bool {
        LEASES.with(|leases| leases.borrow().contains_key(&self.id))
    }

    /// Requests a timer granularity; `false` without a provider or when the platform refuses.
    pub fn set_timer_resolution(&self, nanos: u64) -> bool {
        self.provider
            .as_ref()
            .is_some_and(|provider| provider.set_timer_resolution(nanos))
    }

    /// Associates the calling thread with a logical CPU until the returned guard drops, without
    /// claiming it runs there (upstream `bindManagedCpu`). `None` for an unsupported CPU.
    pub fn bind_managed_cpu(&self, cpu: usize) -> Option<ManagedCpuBinding> {
        if !self.is_supported(cpu) {
            return None;
        }
        static NEXT_TOKEN: AtomicU64 = AtomicU64::new(0);
        let token = NEXT_TOKEN.fetch_add(1, Ordering::Relaxed);
        OWNERS.with(|owners| {
            owners
                .borrow_mut()
                .entry(self.id)
                .or_default()
                .push((token, cpu))
        });
        Some(ManagedCpuBinding {
            controller: self.id,
            token,
            _not_send: PhantomData,
        })
    }

    /// Whether the calling thread has a managed binding.
    pub fn has_managed_owner(&self) -> bool {
        OWNERS.with(|owners| {
            owners
                .borrow()
                .get(&self.id)
                .is_some_and(|stack| !stack.is_empty())
        })
    }

    /// Applies an exact mask, first capturing the thread's original mask if it holds no lease.
    fn apply_exact(&self, request: &[u64]) -> bool {
        let Some(provider) = &self.provider else {
            return false;
        };
        let leased = self.has_affinity_lease();
        if !leased {
            let Some(original) = provider.capture_affinity() else {
                return false;
            };
            // The original binding is restored as captured, without topology validation.
            if masks::nonzero_words(&original) == 0 {
                return false;
            }
            LEASES.with(|leases| leases.borrow_mut().insert(self.id, Lease::Exact(original)));
        }
        let applied = provider.apply_exact(request);
        if !applied && !leased {
            LEASES.with(|leases| leases.borrow_mut().remove(&self.id));
        }
        applied
    }

    /// Applies one locality hint shared by every requested CPU; a request spanning two localities is
    /// refused.
    fn apply_locality(&self, request: &[u64]) -> bool {
        let Some(provider) = &self.provider else {
            return false;
        };
        let mut locality = None;
        for cpu in CpuSet::from_words(request).iter() {
            match (provider.locality_for_cpu(cpu), locality) {
                (None, _) => return false,
                (Some(mapped), Some(previous)) if mapped != previous => return false,
                (Some(mapped), _) => locality = Some(mapped),
            }
        }
        let Some(locality) = locality else {
            return false;
        };
        let leased = self.has_affinity_lease();
        if !leased {
            LEASES.with(|leases| leases.borrow_mut().insert(self.id, Lease::Locality));
        }
        let applied = provider.apply_locality(locality);
        if !applied && !leased {
            LEASES.with(|leases| leases.borrow_mut().remove(&self.id));
        }
        applied
    }
}

impl Drop for AffinityController {
    fn drop(&mut self) {
        // Only this thread's entries are reachable; other threads' entries are inert once the id is
        // retired.
        let _ = LEASES.try_with(|leases| leases.borrow_mut().remove(&self.id));
        let _ = OWNERS.try_with(|owners| owners.borrow_mut().remove(&self.id));
    }
}

/// A scoped managed-CPU association on the creating thread (upstream `ManagedCpuBinding`). It cannot
/// leave its thread. Dropping it removes the association; bindings normally drop innermost first,
/// and one dropped out of order removes only itself.
#[must_use = "the binding ends when dropped"]
pub struct ManagedCpuBinding {
    controller: u64,
    token: u64,
    _not_send: PhantomData<*const ()>,
}

impl fmt::Debug for ManagedCpuBinding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ManagedCpuBinding")
            .field("token", &self.token)
            .finish()
    }
}

impl Drop for ManagedCpuBinding {
    fn drop(&mut self) {
        let _ = OWNERS.try_with(|owners| {
            let mut owners = owners.borrow_mut();
            if let Some(stack) = owners.get_mut(&self.controller) {
                if let Some(index) = stack.iter().rposition(|&(token, _)| token == self.token) {
                    if index + 1 != stack.len() {
                        log::debug!("managed CPU binding dropped out of order");
                    }
                    stack.remove(index);
                }
                if stack.is_empty() {
                    owners.remove(&self.controller);
                }
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[derive(Default)]
    struct Calls {
        applied: Vec<Vec<u64>>,
        restored: Vec<Vec<u64>>,
        localities: Vec<u32>,
        released: usize,
    }

    struct Fake {
        capability: AffinityCapability,
        mask: Mutex<Vec<u64>>,
        calls: Arc<Mutex<Calls>>,
        fail_apply: bool,
        cpu: Option<usize>,
    }

    impl Fake {
        fn new(capability: AffinityCapability, calls: &Arc<Mutex<Calls>>) -> Self {
            Self {
                capability,
                mask: Mutex::new(vec![0b1111]),
                calls: Arc::clone(calls),
                fail_apply: false,
                cpu: None,
            }
        }
    }

    impl AffinityProvider for Fake {
        fn capability(&self) -> AffinityCapability {
            self.capability
        }

        fn current_cpu(&self) -> Option<usize> {
            self.cpu
        }

        fn capture_affinity(&self) -> Option<Vec<u64>> {
            Some(self.mask.lock().unwrap().clone())
        }

        fn apply_exact(&self, mask: &[u64]) -> bool {
            self.calls.lock().unwrap().applied.push(mask.to_vec());
            if !self.fail_apply {
                *self.mask.lock().unwrap() = mask.to_vec();
            }
            !self.fail_apply
        }

        fn restore_exact(&self, mask: &[u64]) -> bool {
            self.calls.lock().unwrap().restored.push(mask.to_vec());
            *self.mask.lock().unwrap() = mask.to_vec();
            true
        }

        fn locality_for_cpu(&self, cpu: usize) -> Option<u32> {
            Some(cpu as u32 + 1)
        }

        fn apply_locality(&self, locality: u32) -> bool {
            self.calls.lock().unwrap().localities.push(locality);
            true
        }

        fn release_locality(&self) -> bool {
            self.calls.lock().unwrap().released += 1;
            true
        }

        fn set_timer_resolution(&self, _nanos: u64) -> bool {
            true
        }
    }

    fn controller(provider: Fake) -> AffinityController {
        let supported: CpuSet = [0, 1, 3].into_iter().collect();
        AffinityController::new(Some(Box::new(provider)), &supported, 4).unwrap()
    }

    #[test]
    fn exact_requests_lease_the_first_mask_and_release_restores_it() {
        let calls = Arc::default();
        let controller = controller(Fake::new(AffinityCapability::Exact, &calls));
        assert_eq!(controller.capability(), AffinityCapability::Exact);
        assert!(controller.set_affinity(1));
        assert!(controller.set_affinity_cpus(&[0, 3]));
        assert!(controller.has_affinity_lease());
        controller.release_affinity();
        assert!(!controller.has_affinity_lease());
        let calls = calls.lock().unwrap();
        assert_eq!(calls.applied, [vec![0b10], vec![0b1001]]);
        assert_eq!(calls.restored, [vec![0b1111]]);
    }

    #[test]
    fn invalid_requests_are_refused_before_the_provider() {
        let calls = Arc::default();
        let controller = controller(Fake::new(AffinityCapability::Exact, &calls));
        assert!(!controller.set_affinity(2), "a hole");
        assert!(!controller.set_affinity(4), "outside the span");
        assert!(!controller.set_affinity_cpus(&[]));
        assert!(!controller.set_affinity_cpus(&[0, 2]));
        assert!(!controller.set_affinity_mask(&[0]));
        assert!(!controller.set_affinity_set(&CpuSet::new()));
        assert!(calls.lock().unwrap().applied.is_empty());
        assert!(!controller.has_affinity_lease());
    }

    #[test]
    fn a_failed_first_apply_drops_the_lease() {
        let calls = Arc::default();
        let controller = controller(Fake {
            fail_apply: true,
            ..Fake::new(AffinityCapability::Exact, &calls)
        });
        assert!(!controller.set_affinity(0));
        assert!(!controller.has_affinity_lease());
    }

    #[test]
    fn locality_hints_need_one_shared_locality() {
        let calls = Arc::default();
        let controller = controller(Fake::new(AffinityCapability::LocalityHint, &calls));
        assert!(controller.set_affinity(3));
        assert!(!controller.set_affinity_cpus(&[0, 1]), "two localities");
        controller.release_affinity();
        controller.release_affinity();
        let calls = calls.lock().unwrap();
        assert_eq!(calls.localities, [4]);
        assert_eq!(calls.released, 1);
    }

    #[test]
    fn unsupported_providers_refuse_everything() {
        let calls = Arc::default();
        let controller = controller(Fake::new(AffinityCapability::Unsupported, &calls));
        assert_eq!(controller.capability(), AffinityCapability::Unsupported);
        assert!(!controller.set_affinity(0));
        assert_eq!(controller.base_mask(), &[0, 1, 3].into_iter().collect());
        let none = AffinityController::new(None, &CpuSet::range(0, 1), 1).unwrap();
        assert!(!none.set_timer_resolution(1));
        assert_eq!(none.current_cpu(), None);
    }

    #[test]
    fn managed_bindings_nest_and_report_the_innermost_cpu() {
        let calls = Arc::default();
        let controller = controller(Fake::new(AffinityCapability::Exact, &calls));
        assert!(controller.bind_managed_cpu(2).is_none());
        let outer = controller.bind_managed_cpu(1).unwrap();
        let inner = controller.bind_managed_cpu(3).unwrap();
        assert_eq!(controller.current_cpu(), Some(3));
        assert!(controller.set_affinity_current());
        drop(inner);
        assert_eq!(controller.current_cpu(), Some(1));
        drop(outer);
        assert_eq!(controller.current_cpu(), None);
        assert!(!controller.has_managed_owner());
        // Out-of-order drops remove only the dropped binding.
        let first = controller.bind_managed_cpu(0).unwrap();
        let second = controller.bind_managed_cpu(1).unwrap();
        drop(first);
        assert_eq!(controller.current_cpu(), Some(1));
        drop(second);
        assert!(!controller.has_managed_owner());
    }

    #[test]
    fn a_truthful_provider_cpu_wins_over_the_managed_cpu() {
        let calls = Arc::default();
        let controller = controller(Fake {
            cpu: Some(3),
            ..Fake::new(AffinityCapability::Exact, &calls)
        });
        let _binding = controller.bind_managed_cpu(0).unwrap();
        assert_eq!(controller.current_cpu(), Some(3));
    }

    #[test]
    fn construction_validates_the_span() {
        assert_eq!(
            AffinityController::new(None, &CpuSet::range(0, 1), 0).unwrap_err(),
            ControllerError::InvalidSpan(0)
        );
        assert_eq!(
            AffinityController::new(None, &CpuSet::range(0, 3), 2).unwrap_err(),
            ControllerError::InvalidSupportedSet
        );
    }
}
