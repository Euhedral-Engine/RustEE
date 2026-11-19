//! [`PlacementPicker`]: a placement policy that sends each frame to a core of one of two CPU sets.
//!
//! A [`ControlPlaneLattice`](super::ControlPlaneLattice) calls its placement policy first on every frame it routes
//! (see [`set_placement_policy`](super::ControlPlaneLattice::set_placement_policy)). The picker is a policy for the
//! common case: a predicate on the frame chooses set A or set B, the chosen set is intersected with the CPUs the
//! lattice routes with, the frame's routing hash picks the nth CPU of that intersection
//! (`unsigned_multiply_high(routing_hash, count)`, as the vertices pick a downstream), and the frame's origin becomes
//! that CPU with a cache-local routing policy, which the routing that follows honours.
//!
//! The picker runs on the frame path from many threads at once: the predicate is `Fn + Send + Sync`, and must be
//! re-entrant, take no lock, not allocate and not block. The picker itself reads only its own immutable sets. It
//! allocates nothing and locks nothing.
//!
//! A frame that cannot be placed (the chosen set has no CPU in common with the lattice's) keeps the origin it had, and
//! the picker returns `false` for it; it returns `true` for a frame whose origin it set.
//!
//! The policy covers pushed and pulled frames alike (see
//! [`set_placement_policy`](super::ControlPlaneLattice::set_placement_policy)); producers need not call the picker.

use crate::flow_control::RoutingPolicy;
use crate::frames::{Frame, FrameHeader, PlacementMark};
use crate::utils::math::unsigned_multiply_high;
use euhedral_hardware_utils::topology::Topology;
use euhedral_hardware_utils::util::CpuSet;

/// Chooses a core of set A or set B for each frame.
pub struct PlacementPicker<P> {
    predicate: P,
    a: CpuSet,
    b: CpuSet,
    topology: Topology,
}

impl<P: Fn(&dyn Frame) -> bool + Send + Sync + 'static> PlacementPicker<P> {
    /// A picker over `topology`'s CPUs: frames for which `predicate` holds go to a CPU of `a`, the others to one of
    /// `b`.
    ///
    /// The predicate sees the frame as its owner left it. On a first placement that is the frame as made; on a later
    /// one (a key frame pushed again, header kept) the picker has first put back the routing policy and origin the
    /// owner had set, so its own earlier mark (`CacheLocal` and the chosen CPU) is never seen. If the owner set a
    /// policy or origin after the earlier placement, the predicate sees those. A frame the picker then cannot place
    /// keeps the owner's fields.
    pub fn new(topology: Topology, predicate: P, a: CpuSet, b: CpuSet) -> Self {
        Self {
            predicate,
            a,
            b,
            topology,
        }
    }

    /// Picks a CPU for `frame` among `lattice_cpus`, the CPUs the lattice routes with, and sets the frame's origin to
    /// it. Returns whether it did: `false` when the frame cannot be placed and keeps the origin it had.
    pub fn pick(&self, frame: &mut dyn Frame, lattice_cpus: &CpuSet) -> bool {
        Self::restore_owner_fields(frame.header_mut());
        let chosen = if (self.predicate)(&*frame) {
            &self.a
        } else {
            &self.b
        };
        let count = chosen.intersection_len(lattice_cpus);
        if count == 0 {
            return false;
        }
        let nth = unsigned_multiply_high(frame.header().routing_hash(), count as u64) as usize;
        let Some(cpu) = chosen.nth_of_intersection(lattice_cpus, nth) else {
            return false;
        };
        let Some(info) = self.topology.cpu_info(cpu) else {
            return false;
        };
        let header = frame.header_mut();
        header.set_placement(Some(PlacementMark {
            owner_policy: header.routing_policy(),
            owner_origin: header.origin(),
            placed_origin: *info,
        }));
        header.set_origin(Some(*info));
        header.set_routing_policy(RoutingPolicy::CacheLocal);
        true
    }

    /// Gives a frame this picker placed before the policy and origin its owner had set, if the owner has not set
    /// others since: a frame pushed again is judged on what its owner left, not on the earlier mark.
    fn restore_owner_fields(header: &mut FrameHeader) {
        let Some(mark) = header.placement() else {
            return;
        };
        header.set_placement(None);
        if header.routing_policy() == RoutingPolicy::CacheLocal
            && header.origin() == Some(mark.placed_origin)
        {
            header.set_routing_policy(mark.owner_policy);
            header.set_origin(mark.owner_origin);
        }
    }

    /// The picker as a placement policy for
    /// [`set_placement_policy`](super::ControlPlaneLattice::set_placement_policy).
    pub fn into_policy(self) -> impl Fn(&mut dyn Frame, &CpuSet) -> bool + Send + Sync + 'static {
        move |frame, cpus| self.pick(frame, cpus)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flow_control::RoutingPolicy;
    use crate::frames::BenchmarkFrame;
    use euhedral_hardware_utils::topology::source::{
        CoreKind, LogicalCpu, TopologyInput, normalize,
    };

    /// Eight CPUs, two to a core, one socket.
    fn topology() -> Topology {
        let cpus = (0..8)
            .map(|cpu| LogicalCpu {
                id: cpu,
                socket_key: "linux:package:0000000000".to_string(),
                die_key: "linux:die:0000000000".to_string(),
                core_key: format!("linux:core:{:010}", cpu / 2),
                kind: CoreKind::Unknown,
            })
            .collect();
        normalize(TopologyInput {
            provider: "linux".to_string(),
            cpus,
            caches: Vec::new(),
        })
        .unwrap()
    }

    fn set(cpus: &[usize]) -> CpuSet {
        cpus.iter().copied().collect()
    }

    fn frame(id: u64, hash: u64) -> Box<dyn Frame> {
        let mut frame = BenchmarkFrame::new(id, None);
        frame.header_mut().set_routing_hash(hash);
        Box::new(frame)
    }

    #[test]
    fn the_hash_picks_the_nth_cpu_of_the_chosen_set_within_the_lattices() {
        let picker = PlacementPicker::new(
            topology(),
            |frame: &dyn Frame| frame.header().id_hash() == 1,
            set(&[0, 2, 4, 6]),
            set(&[1, 3, 5, 7]),
        );
        // The lattice routes with 2, 4, 5, 6 only: set A within it is {2, 4, 6}, set B is {5}.
        let lattice = set(&[2, 4, 5, 6]);
        let origin = |id, hash| {
            let mut frame = frame(id, hash);
            assert!(picker.pick(&mut *frame, &lattice));
            (
                frame.header().origin().map(|info| info.cpu),
                frame.header().routing_policy(),
            )
        };
        assert_eq!(origin(1, 0), (Some(2), RoutingPolicy::CacheLocal));
        assert_eq!(
            origin(1, u64::MAX / 2),
            (Some(4), RoutingPolicy::CacheLocal)
        );
        assert_eq!(origin(1, u64::MAX), (Some(6), RoutingPolicy::CacheLocal));
        assert_eq!(origin(0, 0), (Some(5), RoutingPolicy::CacheLocal));
        assert_eq!(origin(0, u64::MAX), (Some(5), RoutingPolicy::CacheLocal));
    }

    #[test]
    fn a_frame_placed_again_is_judged_on_its_owners_policy_both_times() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};
        let calls = Arc::new(AtomicUsize::new(0));
        let marked = Arc::new(AtomicUsize::new(0));
        let (seen_calls, seen_marked) = (calls.clone(), marked.clone());
        let picker = PlacementPicker::new(
            topology(),
            move |frame: &dyn Frame| {
                let header = frame.header();
                seen_calls.fetch_add(1, Ordering::Relaxed);
                if header.origin().is_some() {
                    seen_marked.fetch_add(1, Ordering::Relaxed);
                }
                header.routing_policy() == RoutingPolicy::CacheLocal
            },
            set(&[0, 2]),
            set(&[1, 3]),
        );
        let lattice = set(&[0, 1, 2, 3]);
        let mut frame = frame(5, 0);
        frame
            .header_mut()
            .set_routing_policy(RoutingPolicy::CacheLocal);
        for _ in 0..2 {
            frame.header_mut().randomize_hash(9);
            assert!(picker.pick(&mut *frame, &lattice));
            let cpu = frame.header().origin().map(|info| info.cpu).unwrap();
            assert!(
                cpu.is_multiple_of(2),
                "the owner's policy sends it to set A"
            );
        }
        assert_eq!(calls.load(Ordering::Relaxed), 2);
        assert_eq!(
            marked.load(Ordering::Relaxed),
            0,
            "the predicate never sees the picker's origin"
        );
    }

    #[test]
    fn a_frame_the_owner_re_marked_is_judged_on_the_new_marks() {
        let picker = PlacementPicker::new(
            topology(),
            |frame: &dyn Frame| frame.header().routing_policy() == RoutingPolicy::Anywhere,
            set(&[0]),
            set(&[1]),
        );
        let lattice = set(&[0, 1]);
        let mut frame = frame(5, 0);
        assert!(picker.pick(&mut *frame, &lattice));
        assert_eq!(frame.header().origin().map(|i| i.cpu), Some(0));
        frame
            .header_mut()
            .set_routing_policy(RoutingPolicy::Anywhere);
        assert!(picker.pick(&mut *frame, &lattice));
        assert_eq!(frame.header().origin().map(|i| i.cpu), Some(0));
        frame
            .header_mut()
            .set_routing_policy(RoutingPolicy::Anywhere);
        frame.header_mut().set_origin(None);
        frame
            .header_mut()
            .set_routing_policy(RoutingPolicy::CacheLocal);
        assert!(picker.pick(&mut *frame, &lattice));
        assert_eq!(frame.header().origin().map(|i| i.cpu), Some(1));
    }

    #[test]
    fn a_frame_that_cannot_be_placed_keeps_its_origin() {
        let picker = PlacementPicker::new(topology(), |_: &dyn Frame| true, set(&[0]), set(&[1]));
        let mut frame = frame(7, 3);
        assert!(!picker.pick(&mut *frame, &set(&[4, 5])));
        assert!(frame.header().origin().is_none());
        assert_eq!(frame.header().routing_policy(), RoutingPolicy::Anywhere);
        assert!(!picker.pick(&mut *frame, &CpuSet::new()));
        assert!(frame.header().origin().is_none());
    }
}
