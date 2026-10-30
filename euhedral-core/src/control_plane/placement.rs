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
//! A frame that cannot be placed (the chosen set has no CPU in common with the lattice's) keeps the origin it had.
//!
//! **The routing that follows can still move a frame out of its set.** The picker chooses among the CPUs the lattice
//! reports as effective. A cache-local frame goes to the core of its origin only while that core is active; when it is
//! not (during a rebalance, or after a clone failed to start), the routing falls back to a hash over the active cores,
//! which can land in the other set. The guarantee that a set's frames stay in the set therefore holds while every core of
//! the set is active.

use crate::flow_control::RoutingPolicy;
use crate::frames::Frame;
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
    pub fn new(topology: Topology, predicate: P, a: CpuSet, b: CpuSet) -> Self {
        Self {
            predicate,
            a,
            b,
            topology,
        }
    }

    /// Places `frame` among `lattice_cpus`, the CPUs the lattice routes with.
    pub fn place(&self, frame: &mut dyn Frame, lattice_cpus: &CpuSet) {
        let chosen = if (self.predicate)(&*frame) {
            &self.a
        } else {
            &self.b
        };
        let count = chosen.intersection_len(lattice_cpus);
        if count == 0 {
            return;
        }
        let nth = unsigned_multiply_high(frame.header().routing_hash(), count as u64) as usize;
        let Some(cpu) = chosen.nth_of_intersection(lattice_cpus, nth) else {
            return;
        };
        let Some(info) = self.topology.cpu_info(cpu) else {
            return;
        };
        let header = frame.header_mut();
        header.set_origin(Some(*info));
        header.set_routing_policy(RoutingPolicy::CacheLocal);
    }

    /// The picker as a placement policy for
    /// [`set_placement_policy`](super::ControlPlaneLattice::set_placement_policy).
    pub fn into_policy(self) -> impl Fn(&mut dyn Frame, &CpuSet) + Send + Sync + 'static {
        move |frame, cpus| self.place(frame, cpus)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
            picker.place(&mut *frame, &lattice);
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
    fn a_frame_that_cannot_be_placed_keeps_its_origin() {
        let picker = PlacementPicker::new(topology(), |_: &dyn Frame| true, set(&[0]), set(&[1]));
        let mut frame = frame(7, 3);
        picker.place(&mut *frame, &set(&[4, 5]));
        assert!(frame.header().origin().is_none());
        assert_eq!(frame.header().routing_policy(), RoutingPolicy::Anywhere);
        picker.place(&mut *frame, &CpuSet::new());
        assert!(frame.header().origin().is_none());
    }
}
