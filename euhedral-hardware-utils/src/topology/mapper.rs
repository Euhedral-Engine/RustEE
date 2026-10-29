use super::Topology;
use crate::monitor::HardwareUtilization;
use crate::util::CpuSet;
#[allow(clippy::disallowed_types, reason = "HOT-FIX(H6), HOT-FIX(H7)")]
use std::sync::{Arc, Mutex, PoisonError, RwLock};

/// The effective CPUs, cores and core-to-CPU map of one socket.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EffectiveSocketTopology {
    /// Bumped each time this socket's membership changes; 1 on its first publication.
    pub version: u64,
    pub socket_id: usize,
    pub effective_cores: CpuSet,
    pub effective_cpus: CpuSet,
    /// Indexed by global core id (`max_core_id + 1` entries); `None` for a core with no effective
    /// CPU on this socket.
    pub effective_core_to_cpu: Vec<Option<CpuSet>>,
}

/// One published effective topology.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EffectiveSystemTopology {
    pub effective_sockets: CpuSet,
    pub effective_cores: CpuSet,
    pub effective_cpus: CpuSet,
    /// Indexed by socket id (`max_socket_id + 1` entries); `None` for an inactive socket.
    pub socket_topologies: Vec<Option<EffectiveSocketTopology>>,
    /// Bumped on every membership change; 0 until the first publication.
    pub global_version: u64,
}

impl EffectiveSystemTopology {
    /// The active sockets in ascending id order, without the `None` holes.
    pub fn active_sockets(&self) -> impl Iterator<Item = &EffectiveSocketTopology> {
        self.socket_topologies.iter().flatten()
    }
}

/// Tracks which part of a topology is currently usable and publishes it as an immutable snapshot
/// (upstream `TopologyMapper`).
///
/// Effective membership is the intersection of what the resource monitor reports as effective, the
/// topology's CPU set and the CPUs allowed at construction. A snapshot is published, and versions
/// move, only when membership changes. Readers never block on an update: they clone an `Arc` of the
/// current snapshot.
///
/// Updates are serialized by a mutex: the last applied wins and nothing is lost.
#[derive(Debug)]
pub struct TopologyMapper {
    topology: Topology,
    allowed_cpus: CpuSet,
    #[allow(clippy::disallowed_types, reason = "HOT-FIX(H6)")]
    publish: Mutex<Vec<u64>>,
    #[allow(clippy::disallowed_types, reason = "HOT-FIX(H7)")]
    current: RwLock<Arc<EffectiveSystemTopology>>,
}

impl TopologyMapper {
    /// A mapper over the host topology that allows every CPU.
    pub fn new() -> Self {
        let topology = Topology::system().clone();
        let allowed = topology.cpu_set().clone();
        Self::with_topology(topology, &allowed)
    }

    /// A mapper over the host topology that allows only `allowed_cpus`.
    pub fn with_allowed(allowed_cpus: &CpuSet) -> Self {
        Self::with_topology(Topology::system().clone(), allowed_cpus)
    }

    #[allow(clippy::disallowed_types, reason = "HOT-FIX(H6), HOT-FIX(H7)")]
    pub fn with_topology(topology: Topology, allowed_cpus: &CpuSet) -> Self {
        let allowed_cpus = allowed_cpus.intersection(topology.cpu_set());
        let sockets = topology.socket_count();
        let initial = EffectiveSystemTopology {
            effective_sockets: CpuSet::new(),
            effective_cores: CpuSet::new(),
            effective_cpus: CpuSet::new(),
            socket_topologies: vec![None; sockets],
            global_version: 0,
        };
        Self {
            topology,
            allowed_cpus,
            publish: Mutex::new(vec![0; sockets]),
            current: RwLock::new(Arc::new(initial)),
        }
    }

    pub fn topology(&self) -> &Topology {
        &self.topology
    }

    pub fn allowed_cpus(&self) -> &CpuSet {
        &self.allowed_cpus
    }

    pub fn effective_topology(&self) -> Arc<EffectiveSystemTopology> {
        self.current
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    /// The CPUs that are effective now: the monitor's, within the topology and the allowed set.
    pub fn effective_cpus(&self) -> CpuSet {
        self.effective_topology().effective_cpus.clone()
    }

    pub fn global_version(&self) -> u64 {
        self.effective_topology().global_version
    }

    /// The socket's current effective topology, or `None` when the socket is inactive or out of
    /// range.
    pub fn effective_socket_topology(&self, socket_id: usize) -> Option<EffectiveSocketTopology> {
        self.effective_topology()
            .socket_topologies
            .get(socket_id)
            .cloned()
            .flatten()
    }

    /// Feeds one monitor sample's effective CPUs.
    pub fn update_from(&self, utilization: &HardwareUtilization) {
        self.update(&utilization.global_effective_cpus);
    }

    /// Feeds a set of effective CPUs; publishes only if membership changed.
    pub fn update(&self, effective_cpus: &CpuSet) {
        let effective_cpus = effective_cpus
            .intersection(self.topology.cpu_set())
            .intersection(&self.allowed_cpus);
        let mut socket_versions = self.publish.lock().unwrap_or_else(PoisonError::into_inner);

        let mut cores = Vec::new();
        let mut sockets = Vec::new();
        for cpu in &effective_cpus {
            let info = self
                .topology
                .cpu_info(cpu)
                .expect("effective CPUs are topology CPUs");
            cores.push(info.core);
            sockets.push(info.socket);
        }
        let effective_cores: CpuSet = cores.into_iter().collect();
        let effective_sockets: CpuSet = sockets.into_iter().collect();

        let previous = self.effective_topology();
        if effective_cpus == previous.effective_cpus
            && effective_cores == previous.effective_cores
            && effective_sockets == previous.effective_sockets
        {
            return;
        }

        let mut socket_topologies = Vec::with_capacity(socket_versions.len());
        for (socket_id, version) in socket_versions.iter_mut().enumerate() {
            let candidate = effective_sockets
                .contains(socket_id)
                .then(|| self.socket_topology(socket_id, &effective_cpus, &effective_cores));
            let prior = previous.socket_topologies[socket_id].as_ref();
            let changed = match (prior, &candidate) {
                (Some(prior), Some((cores, cpus, _))) => {
                    prior.effective_cores != *cores || prior.effective_cpus != *cpus
                }
                (None, None) => false,
                _ => true,
            };
            if changed {
                *version += 1;
            }
            socket_topologies.push(candidate.map(
                |(effective_cores, effective_cpus, effective_core_to_cpu)| {
                    EffectiveSocketTopology {
                        version: *version,
                        socket_id,
                        effective_cores,
                        effective_cpus,
                        effective_core_to_cpu,
                    }
                },
            ));
        }

        let next = EffectiveSystemTopology {
            effective_sockets,
            effective_cores,
            effective_cpus,
            socket_topologies,
            global_version: previous.global_version + 1,
        };
        *self.current.write().unwrap_or_else(PoisonError::into_inner) = Arc::new(next);
    }

    fn socket_topology(
        &self,
        socket_id: usize,
        global_cpus: &CpuSet,
        global_cores: &CpuSet,
    ) -> (CpuSet, CpuSet, Vec<Option<CpuSet>>) {
        let info = self
            .topology
            .socket_info(socket_id)
            .expect("effective sockets are topology sockets");
        let cpus = info.cpus.intersection(global_cpus);
        let cores = info.cores.intersection(global_cores);
        let mut core_to_cpu: Vec<Vec<usize>> = vec![Vec::new(); self.topology.core_count()];
        for cpu in &cpus {
            let core = self
                .topology
                .cpu_info(cpu)
                .expect("socket CPUs are topology CPUs")
                .core;
            core_to_cpu[core].push(cpu);
        }
        let core_to_cpu = core_to_cpu
            .into_iter()
            .map(|members| (!members.is_empty()).then(|| members.into_iter().collect()))
            .collect();
        (cores, cpus, core_to_cpu)
    }
}

impl Default for TopologyMapper {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::source::{CoreKind, LogicalCpu, TopologyInput, normalize};
    use std::thread;

    fn linux_cpu(id: usize, socket: usize, core: usize) -> LogicalCpu {
        LogicalCpu {
            id,
            socket_key: format!("linux:package:{socket:010}"),
            die_key: "linux:die:0000000000".to_string(),
            core_key: format!("linux:core:{core:010}"),
            kind: CoreKind::Unknown,
        }
    }

    fn two_socket() -> Topology {
        normalize(TopologyInput {
            provider: "linux".to_string(),
            cpus: vec![linux_cpu(0, 0, 0), linux_cpu(3, 0, 1), linux_cpu(7, 1, 0)],
            caches: Vec::new(),
        })
        .unwrap()
    }

    fn bits(ids: &[usize]) -> CpuSet {
        ids.iter().copied().collect()
    }

    #[test]
    fn publishes_owned_coalesced_topology() {
        let mapper = Arc::new(TopologyMapper::with_topology(
            two_socket(),
            &bits(&[0, 3, 7, 100]),
        ));
        let first = {
            let mapper = mapper.clone();
            thread::spawn(move || mapper.update(&bits(&[0, 3])))
        };
        let second = {
            let mapper = mapper.clone();
            thread::spawn(move || mapper.update(&bits(&[0, 3, 7, 100])))
        };
        first.join().unwrap();
        second.join().unwrap();
        mapper.update(&bits(&[0, 3, 7, 100]));

        let topology = mapper.effective_topology();
        assert_eq!(topology.effective_cpus, bits(&[0, 3, 7]));
        assert_eq!(topology.socket_topologies.len(), 2);
        for socket in topology.active_sockets() {
            assert_eq!(socket.effective_core_to_cpu.len(), 3);
        }
        let socket0 = topology.socket_topologies[0].as_ref().unwrap();
        assert_eq!(socket0.effective_core_to_cpu[0], Some(bits(&[0])));
        assert_eq!(socket0.effective_core_to_cpu[2], None);
    }

    #[test]
    fn versions_only_published_membership_changes() {
        let mapper = TopologyMapper::with_topology(two_socket(), &bits(&[0, 3, 7]));
        assert_eq!(mapper.global_version(), 0);
        assert_eq!(mapper.effective_socket_topology(0), None);

        mapper.update(&bits(&[0, 3, 7]));
        let both = mapper.effective_topology();
        assert_eq!(both.global_version, 1);
        assert_eq!(both.socket_topologies[0].as_ref().unwrap().version, 1);
        assert_eq!(both.socket_topologies[1].as_ref().unwrap().version, 1);
        mapper.update(&bits(&[0, 3, 7]));
        assert!(Arc::ptr_eq(&both, &mapper.effective_topology()));

        mapper.update(&bits(&[0, 3]));
        let one = mapper.effective_topology();
        assert_eq!(one.global_version, 2);
        assert_eq!(one.socket_topologies[1], None);
        assert_eq!(one.socket_topologies[0].as_ref().unwrap().version, 1);

        mapper.update(&bits(&[0, 3, 7]));
        let reactivated = mapper.effective_topology();
        assert_eq!(reactivated.global_version, 3);
        assert_eq!(
            reactivated.socket_topologies[1].as_ref().unwrap().version,
            3
        );
        assert_eq!(
            reactivated.socket_topologies[0].as_ref().unwrap().version,
            1
        );
        assert_eq!(mapper.effective_socket_topology(5), None);
    }

    #[test]
    fn membership_outside_the_topology_or_allowed_set_is_ignored() {
        let topology = normalize(TopologyInput {
            provider: "fallback".to_string(),
            cpus: vec![LogicalCpu {
                id: 5,
                socket_key: "fallback:package:0".to_string(),
                die_key: "fallback:die:0".to_string(),
                core_key: "fallback:core:00000000".to_string(),
                kind: CoreKind::Unknown,
            }],
            caches: Vec::new(),
        })
        .unwrap();
        let mapper = TopologyMapper::with_topology(topology, &bits(&[5, 99]));
        assert_eq!(mapper.allowed_cpus(), &bits(&[5]));
        mapper.update(&bits(&[5, 99]));
        assert_eq!(mapper.effective_topology().effective_cpus, bits(&[5]));
        mapper.update(&bits(&[99]));
        assert!(mapper.effective_topology().effective_cpus.is_empty());
        assert_eq!(mapper.global_version(), 2);
    }

    #[test]
    fn publishes_the_system_topology_once() {
        let mapper = TopologyMapper::new();
        let all = Topology::system().cpu_set().clone();
        mapper.update(&all);
        let topology = mapper.effective_topology();
        assert_eq!(topology.effective_cpus, all);
        assert_eq!(topology.global_version, 1);
        for socket in topology.effective_sockets.iter() {
            assert!(
                mapper
                    .effective_socket_topology(socket)
                    .unwrap()
                    .effective_cpus
                    .intersects(&all)
            );
        }
        mapper.update(&all);
        assert_eq!(mapper.global_version(), 1);
    }
}
