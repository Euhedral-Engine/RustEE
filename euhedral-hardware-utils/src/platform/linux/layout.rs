//! Linux topology from sysfs (upstream `LinuxSystemLayout`).
//!
//! Reads `<sys>/devices/system/cpu/cpuN/{topology,cache,cpufreq}` for every online CPU. A CPU without
//! a package id joins one shared unknown package; a CPU without a core id is its own core.

use super::fs::read_trimmed;
use super::parse::{cpu_list, decimal, size_bytes};
use crate::topology::Topology;
use crate::topology::mask::parse_hex_mask;
use crate::topology::source::{self, CacheDomain, CoreKind, LogicalCpu, TopologyInput};
use crate::util::CpuSet;
use std::collections::BTreeMap;
use std::path::Path;

/// Cores whose scores differ by less than this fraction are not split into performance and
/// efficiency classes.
const MIN_RELATIVE_GAP: f64 = 0.15;

/// The topology under `sys` (normally `/sys`), or the fallback when it cannot be read.
pub(crate) fn discover(sys: &Path) -> Topology {
    let input = collect(sys);
    if input.cpus.is_empty() {
        return source::fallback(std::thread::available_parallelism().map_or(1, |n| n.get()));
    }
    source::normalize(input).unwrap_or_else(|error| {
        log::error!("failed to read the Linux topology ({error}); using the fallback topology");
        source::fallback(std::thread::available_parallelism().map_or(1, |n| n.get()))
    })
}

#[derive(Debug)]
struct RawCpu {
    id: usize,
    package: Option<u32>,
    die: u32,
    core: Option<u32>,
    capacity: Option<u64>,
    max_frequency: Option<u64>,
    cache_score: Option<u64>,
}

impl RawCpu {
    fn package_key(&self) -> String {
        match self.package {
            Some(package) => format!("linux:package:{package:010}"),
            None => "linux:package:unknown".to_owned(),
        }
    }

    fn core_key(&self) -> String {
        match self.core {
            Some(core) => format!("linux:core:{core:010}"),
            None => format!("linux:core:cpu{:07}", self.id),
        }
    }

    /// The core this CPU belongs to: package, die and core.
    fn core_tuple(&self) -> (String, u32, String) {
        (self.package_key(), self.die, self.core_key())
    }
}

pub(crate) fn collect(sys: &Path) -> TopologyInput {
    let cpu_root = sys.join("devices/system/cpu");
    let mut ids: Vec<usize> = std::fs::read_dir(&cpu_root)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|entry| {
            let name = entry.file_name();
            let id = decimal(name.to_str()?.strip_prefix("cpu")?)?;
            usize::try_from(id).ok()
        })
        .collect();
    ids.sort_unstable();
    if let Some(online) = read_trimmed(&cpu_root.join("online")).and_then(|text| cpu_list(&text)) {
        ids.retain(|&id| online.contains(id));
    }

    let mut caches = Vec::new();
    let raw: Vec<RawCpu> = ids
        .iter()
        .map(|&id| {
            let dir = cpu_root.join(format!("cpu{id}"));
            let attribute = |name: &str| {
                read_trimmed(&dir.join(name))
                    .and_then(|text| decimal(&text))
                    .and_then(|value| u32::try_from(value).ok())
            };
            RawCpu {
                id,
                package: attribute("topology/physical_package_id"),
                die: attribute("topology/die_id").unwrap_or(0),
                core: attribute("topology/core_id"),
                capacity: read_trimmed(&dir.join("cpu_capacity"))
                    .and_then(|text| decimal(&text))
                    .filter(|&v| v > 0),
                max_frequency: read_trimmed(&dir.join("cpufreq/cpuinfo_max_freq"))
                    .and_then(|text| decimal(&text))
                    .filter(|&v| v > 0),
                cache_score: collect_caches(&dir.join("cache"), &mut caches),
            }
        })
        .collect();

    let kinds = classify(sys, &raw);
    let cpus = raw
        .iter()
        .map(|cpu| LogicalCpu {
            id: cpu.id,
            socket_key: cpu.package_key(),
            die_key: format!("linux:die:{:010}", cpu.die),
            core_key: cpu.core_key(),
            kind: kinds
                .get(&cpu.core_tuple())
                .copied()
                .unwrap_or(CoreKind::Unknown),
        })
        .collect();
    TopologyInput {
        provider: "linux".into(),
        cpus,
        caches,
    }
}

/// Adds this CPU's data and unified caches to `target` and returns its cache-capacity score.
fn collect_caches(dir: &Path, target: &mut Vec<CacheDomain>) -> Option<u64> {
    let mut l1 = None;
    let mut l2 = None;
    for entry in std::fs::read_dir(dir).into_iter().flatten().flatten() {
        let path = entry.path();
        let read = |name: &str| read_trimmed(&path.join(name));
        let Some(kind) = read("type") else { continue };
        if kind.to_ascii_lowercase().starts_with("instruction") {
            continue;
        }
        let level = read("level")
            .and_then(|text| decimal(&text))
            .and_then(|v| u8::try_from(v).ok());
        let size = read("size").and_then(|text| size_bytes(&text));
        let line = read("coherency_line_size")
            .and_then(|text| decimal(&text))
            .and_then(|v| u32::try_from(v).ok());
        let sharers = read("shared_cpu_list")
            .and_then(|text| cpu_list(&text))
            .or_else(|| read("shared_cpu_map").and_then(|text| parse_hex_mask(&text).ok()));
        let (Some(level), Some(size), Some(line), Some(sharers)) = (level, size, line, sharers)
        else {
            continue;
        };
        let count = sharers.len() as u64;
        match level {
            1 => l1 = size.checked_mul(count).filter(|&v| v > 0),
            2 => l2 = Some(size / if count > 2 { count } else { 1 }).filter(|&v| v > 0),
            _ => {}
        }
        target.push(CacheDomain {
            level,
            size_bytes: size,
            line_size_bytes: line,
            sharers,
        });
    }
    Some(l1?.saturating_add(l2?))
}

type CoreTuple = (String, u32, String);

/// Performance and efficiency classes per core: from the kernel's hybrid PMU CPU lists when present,
/// else from the first complete score of CPU capacity, maximum frequency or cache capacity.
fn classify(sys: &Path, cpus: &[RawCpu]) -> BTreeMap<CoreTuple, CoreKind> {
    let list = |pmu: &str| {
        read_trimmed(&sys.join(format!("devices/{pmu}/cpus"))).and_then(|text| cpu_list(&text))
    };
    if let (Some(performance), Some(efficiency)) = (list("cpu_core"), list("cpu_atom")) {
        return classify_by_lists(cpus, &performance, &efficiency);
    }
    let scorers: [fn(&RawCpu) -> Option<u64>; 3] = [
        |cpu| cpu.capacity,
        |cpu| cpu.max_frequency,
        |cpu| cpu.cache_score,
    ];
    for score in scorers {
        if let Some(kinds) = classify_by_score(cpus, score) {
            return kinds;
        }
    }
    BTreeMap::new()
}

fn classify_by_lists(
    cpus: &[RawCpu],
    performance: &CpuSet,
    efficiency: &CpuSet,
) -> BTreeMap<CoreTuple, CoreKind> {
    let mut kinds = BTreeMap::new();
    for cpu in cpus {
        let kind = match (performance.contains(cpu.id), efficiency.contains(cpu.id)) {
            (true, false) => CoreKind::Performance,
            (false, true) => CoreKind::Efficiency,
            _ => CoreKind::Unknown,
        };
        kinds
            .entry(cpu.core_tuple())
            .and_modify(|existing| {
                if *existing != kind {
                    *existing = CoreKind::Unknown;
                }
            })
            .or_insert(kind);
    }
    kinds
}

/// `None` when some CPU has no score, siblings disagree, or the scores do not separate into two
/// classes with a relative gap of at least [`MIN_RELATIVE_GAP`]. Cores above the widest gap are
/// performance cores.
fn classify_by_score(
    cpus: &[RawCpu],
    score: fn(&RawCpu) -> Option<u64>,
) -> Option<BTreeMap<CoreTuple, CoreKind>> {
    let mut scores = BTreeMap::new();
    for cpu in cpus {
        let value = score(cpu)?;
        if *scores.entry(cpu.core_tuple()).or_insert(value) != value {
            return None;
        }
    }
    let mut distinct: Vec<u64> = scores.values().copied().collect();
    distinct.sort_unstable_by(|a, b| b.cmp(a));
    distinct.dedup();
    let (threshold, gap) = distinct
        .windows(2)
        .map(|pair| (pair[0], (pair[0] - pair[1]) as f64 / pair[0] as f64))
        .max_by(|a, b| a.1.total_cmp(&b.1))?;
    if gap < MIN_RELATIVE_GAP {
        return None;
    }
    Some(
        scores
            .into_iter()
            .map(|(tuple, value)| {
                let kind = if value >= threshold {
                    CoreKind::Performance
                } else {
                    CoreKind::Efficiency
                };
                (tuple, kind)
            })
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::platform::testing::{TempDir, write};

    fn cpu(root: &Path, id: usize, package: Option<&str>, core: Option<&str>) {
        let base = format!("devices/system/cpu/cpu{id}");
        write(root, &format!("{base}/uevent"), "");
        if let Some(package) = package {
            write(
                root,
                &format!("{base}/topology/physical_package_id"),
                package,
            );
        }
        if let Some(core) = core {
            write(root, &format!("{base}/topology/core_id"), core);
        }
    }

    fn cache(root: &Path, id: usize, index: usize, kind: &str, level: u8, size: &str, map: &str) {
        let base = format!("devices/system/cpu/cpu{id}/cache/index{index}");
        write(root, &format!("{base}/type"), kind);
        write(root, &format!("{base}/level"), &level.to_string());
        write(root, &format!("{base}/size"), size);
        write(root, &format!("{base}/coherency_line_size"), "64");
        write(root, &format!("{base}/shared_cpu_map"), map);
    }

    fn frequency(root: &Path, id: usize, khz: u64) {
        write(
            root,
            &format!("devices/system/cpu/cpu{id}/cpufreq/cpuinfo_max_freq"),
            &khz.to_string(),
        );
    }

    fn p_core(topology: &Topology, cpu: usize) -> bool {
        topology
            .core_info(topology.cpu_info(cpu).unwrap().core)
            .unwrap()
            .p_core
    }

    #[test]
    fn scans_a_sparse_multi_socket_tree() {
        let dir = TempDir::new("layout-sparse");
        cpu(dir.path(), 0, Some("0"), Some("0"));
        cache(dir.path(), 0, 0, "Data", 1, "32K", "1");
        cpu(dir.path(), 2, Some("0"), Some("1"));
        cpu(dir.path(), 8, Some("1"), Some("0"));
        let topology = discover(dir.path());
        assert_eq!(topology.system_cpus(), [0, 2, 8]);
        assert_eq!((topology.core_count(), topology.socket_count()), (3, 2));
        assert!(topology.cpu_info(1).is_none());
        assert_ne!(
            topology.cpu_info(0).unwrap().core,
            topology.cpu_info(8).unwrap().core
        );
        assert_eq!(topology.cache_layout(0).unwrap().bytes_l1, 32 << 10);
    }

    #[test]
    fn classifies_hybrid_cores_by_frequency() {
        let dir = TempDir::new("layout-freq");
        cpu(dir.path(), 0, Some("0"), Some("0"));
        frequency(dir.path(), 0, 5_000_000);
        cpu(dir.path(), 1, Some("0"), Some("1"));
        frequency(dir.path(), 1, 3_000_000);
        let topology = discover(dir.path());
        assert!(p_core(&topology, 0));
        assert!(!p_core(&topology, 1));
    }

    #[test]
    fn classifies_hybrid_cores_by_cache_capacity_without_frequencies() {
        let dir = TempDir::new("layout-cache");
        cpu(dir.path(), 0, Some("0"), Some("0"));
        cache(dir.path(), 0, 0, "Data", 1, "64K", "1");
        cache(dir.path(), 0, 1, "Unified", 2, "2M", "1");
        cpu(dir.path(), 1, Some("0"), Some("1"));
        cache(dir.path(), 1, 0, "Data", 1, "32K", "2");
        cache(dir.path(), 1, 1, "Unified", 2, "512K", "2");
        let topology = discover(dir.path());
        assert!(p_core(&topology, 0));
        assert!(!p_core(&topology, 1));
    }

    #[test]
    fn small_or_absent_differences_leave_cores_unclassified() {
        let dir = TempDir::new("layout-homogeneous");
        cpu(dir.path(), 0, Some("0"), Some("0"));
        frequency(dir.path(), 0, 3_000_000);
        cpu(dir.path(), 1, Some("0"), Some("1"));
        frequency(dir.path(), 1, 3_000_000);
        cpu(dir.path(), 2, Some("0"), Some("2"));
        // A preferred core boosting 10% higher is not an efficiency split.
        frequency(dir.path(), 2, 3_300_000);
        let topology = discover(dir.path());
        assert!((0..3).all(|cpu| p_core(&topology, cpu)));
        assert!(topology.e_core_set().is_empty());
    }

    #[test]
    fn hybrid_pmu_lists_take_precedence() {
        let dir = TempDir::new("layout-pmu");
        for id in 0..4 {
            cpu(dir.path(), id, Some("0"), Some(&id.to_string()));
            frequency(dir.path(), id, 4_000_000);
        }
        write(dir.path(), "devices/cpu_core/cpus", "0-1\n");
        write(dir.path(), "devices/cpu_atom/cpus", "2-3\n");
        let topology = discover(dir.path());
        assert_eq!(topology.p_cpu_set(), &CpuSet::range(0, 2));
        assert_eq!(topology.e_cpu_set(), &CpuSet::range(2, 4));
    }

    #[test]
    fn cpu_capacity_classifies_arm_big_little() {
        let dir = TempDir::new("layout-capacity");
        for (id, capacity) in [(0, "1024"), (1, "1024"), (2, "446"), (3, "446")] {
            cpu(dir.path(), id, Some("0"), Some(&id.to_string()));
            write(
                dir.path(),
                &format!("devices/system/cpu/cpu{id}/cpu_capacity"),
                capacity,
            );
        }
        let topology = discover(dir.path());
        assert_eq!(topology.e_cpu_set(), &CpuSet::range(2, 4));
    }

    #[test]
    fn an_unreadable_root_falls_back() {
        let dir = TempDir::new("layout-missing");
        let topology = discover(&dir.path().join("nonexistent"));
        let expected = std::thread::available_parallelism().map_or(1, |n| n.get());
        assert_eq!(topology.cpu_count(), expected);
        assert!(topology.cpu_info(0).is_some());
    }

    #[test]
    fn offline_cpus_are_excluded() {
        let dir = TempDir::new("layout-offline");
        cpu(dir.path(), 0, Some("0"), Some("0"));
        cpu(dir.path(), 1, Some("0"), Some("1"));
        cache(dir.path(), 2, 0, "Data", 1, "32K", "4");
        cpu(dir.path(), 3, Some("0"), Some("3"));
        write(dir.path(), "devices/system/cpu/online", "0-1,3\n");
        let topology = discover(dir.path());
        assert_eq!(topology.system_cpus(), [0, 1, 3]);
        assert_eq!(topology.core_count(), 3);
        assert!(topology.cache_layout(2).is_none());
        assert!(
            topology
                .cache_layouts()
                .all(|layout| !layout.mask_l1.contains(2))
        );
    }

    #[test]
    fn missing_identities_neither_merge_nor_inflate() {
        let dir = TempDir::new("layout-identity");
        for id in 0..4 {
            write(
                dir.path(),
                &format!("devices/system/cpu/cpu{id}/uevent"),
                "",
            );
        }
        let topology = discover(dir.path());
        assert_eq!(
            (
                topology.cpu_count(),
                topology.socket_count(),
                topology.core_count()
            ),
            (4, 1, 4)
        );

        let dir = TempDir::new("layout-core-id");
        cpu(dir.path(), 0, Some("0"), Some("0"));
        cpu(dir.path(), 5, Some("0"), None);
        cpu(dir.path(), 6, Some("0"), None);
        let topology = discover(dir.path());
        assert_eq!(topology.core_count(), 3);

        let dir = TempDir::new("layout-package-id");
        cpu(dir.path(), 0, Some("0"), Some("0"));
        cpu(dir.path(), 1, None, Some("0"));
        cpu(dir.path(), 2, None, Some("1"));
        let topology = discover(dir.path());
        assert_eq!(topology.socket_count(), 2);
        let socket = |cpu| topology.cpu_info(cpu).unwrap().socket;
        assert_eq!(socket(0), 0, "real packages keep the low socket ids");
        assert_eq!(socket(1), socket(2));
        assert_ne!(
            topology.cpu_info(0).unwrap().core,
            topology.cpu_info(1).unwrap().core
        );
    }

    #[test]
    fn a_malformed_online_list_keeps_every_cpu() {
        let dir = TempDir::new("layout-online");
        cpu(dir.path(), 0, Some("0"), Some("0"));
        cpu(dir.path(), 1, Some("0"), Some("1"));
        write(dir.path(), "devices/system/cpu/online", "3-1");
        assert_eq!(discover(dir.path()).cpu_count(), 2);
    }
}
