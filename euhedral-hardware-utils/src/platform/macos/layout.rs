//! macOS topology from sysctl (upstream `MacosSystemLayout`).
//!
//! On a machine with two performance levels, `hw.perflevel1` (efficiency) CPUs come first and
//! `hw.perflevel0` (performance) CPUs follow, each its own core. Otherwise the CPUs are homogeneous
//! and `hw.logicalcpu / hw.physicalcpu` consecutive CPUs share a core.

use super::Sysctl;
use crate::topology::Topology;
use crate::topology::source::{self, CacheDomain, CoreKind, LogicalCpu, TopologyInput};
use crate::util::CpuSet;

const DEFAULT_LINE_BYTES: u64 = 64;

/// The topology `sysctl` describes, or the fallback when it is invalid.
pub(crate) fn discover(sysctl: &dyn Sysctl, processors: usize) -> Topology {
    source::normalize(collect(sysctl, processors)).unwrap_or_else(|error| {
        log::error!("failed to read the macOS topology ({error}); using the fallback topology");
        source::fallback(processors)
    })
}

/// One performance level: its CPUs are `first..first + count`.
struct Level {
    /// The `hw.perflevelN` index.
    index: u32,
    first: usize,
    count: usize,
    kind: CoreKind,
}

fn positive(sysctl: &dyn Sysctl, name: &str) -> Option<u64> {
    sysctl.integer(name).filter(|&value| value > 0)
}

fn count(sysctl: &dyn Sysctl, name: &str) -> Option<usize> {
    positive(sysctl, name).and_then(|value| usize::try_from(value).ok())
}

/// The efficiency and performance levels, when they exist and account for every CPU.
fn levels(sysctl: &dyn Sysctl, logical: usize) -> Option<[Level; 2]> {
    if positive(sysctl, "hw.nperflevels")? < 2 {
        return None;
    }
    let performance = count(sysctl, "hw.perflevel0.logicalcpu")?;
    let efficiency = count(sysctl, "hw.perflevel1.logicalcpu")?;
    (performance.checked_add(efficiency)? == logical).then_some([
        Level {
            index: 1,
            first: 0,
            count: efficiency,
            kind: CoreKind::Efficiency,
        },
        Level {
            index: 0,
            first: efficiency,
            count: performance,
            kind: CoreKind::Performance,
        },
    ])
}

/// `first..first + count` split into consecutive groups of `per_group`.
fn groups(first: usize, count: usize, per_group: usize) -> impl Iterator<Item = CpuSet> {
    let per_group = per_group.clamp(1, count.max(1));
    (first..first + count)
        .step_by(per_group)
        .map(move |start| CpuSet::range(start, (start + per_group).min(first + count)))
}

fn cpu(id: usize, core: usize, kind: CoreKind) -> LogicalCpu {
    LogicalCpu {
        id,
        socket_key: "macos:package:0".to_owned(),
        die_key: "macos:die:0".to_owned(),
        core_key: format!("macos:core:{core:08x}"),
        kind,
    }
}

pub(crate) fn collect(sysctl: &dyn Sysctl, processors: usize) -> TopologyInput {
    let logical = count(sysctl, "hw.logicalcpu").unwrap_or(processors.max(1));
    let physical = count(sysctl, "hw.physicalcpu").unwrap_or(logical);
    let line = u32::try_from(positive(sysctl, "hw.cachelinesize").unwrap_or(DEFAULT_LINE_BYTES))
        .unwrap_or(u32::MAX);
    let cache = |level: u8, size: Option<u64>, sharers: CpuSet| {
        size.map(|size_bytes| CacheDomain {
            level,
            size_bytes,
            line_size_bytes: line,
            sharers,
        })
    };
    let mut cpus = Vec::with_capacity(logical);
    let mut caches = Vec::new();

    if let Some(levels) = levels(sysctl, logical) {
        for level in &levels {
            let key = |name: &str| format!("hw.perflevel{}.{name}", level.index);
            cpus.extend((level.first..level.first + level.count).map(|id| cpu(id, id, level.kind)));
            let l1 = positive(sysctl, &key("l1dcachesize"))
                .or_else(|| positive(sysctl, "hw.l1dcachesize"));
            caches
                .extend(groups(level.first, level.count, 1).filter_map(|core| cache(1, l1, core)));
            let l2 = positive(sysctl, &key("l2cachesize"))
                .or_else(|| positive(sysctl, "hw.l2cachesize"));
            let per_l2 = count(sysctl, &key("cpusperl2")).unwrap_or(level.count);
            caches.extend(
                groups(level.first, level.count, per_l2)
                    .filter_map(|cluster| cache(2, l2, cluster)),
            );
        }
    } else {
        let threads_per_core = if logical > physical {
            logical / physical
        } else {
            1
        };
        cpus.extend((0..logical).map(|id| cpu(id, id / threads_per_core, CoreKind::Unknown)));
        let l1 = positive(sysctl, "hw.perflevel0.l1dcachesize")
            .or_else(|| positive(sysctl, "hw.l1dcachesize"));
        caches.extend(groups(0, logical, threads_per_core).filter_map(|core| cache(1, l1, core)));
        let l2 = positive(sysctl, "hw.perflevel0.l2cachesize")
            .or_else(|| positive(sysctl, "hw.l2cachesize"));
        let per_l2 = count(sysctl, "hw.perflevel0.cpusperl2").unwrap_or(logical);
        caches.extend(groups(0, logical, per_l2).filter_map(|cluster| cache(2, l2, cluster)));
    }

    let l3 = positive(sysctl, "hw.l3cachesize");
    caches.extend(cache(3, l3, CpuSet::range(0, logical)));
    TopologyInput {
        provider: "macos".to_owned(),
        cpus,
        caches,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// An M1 Pro: two efficiency cores sharing a 4 MiB L2, eight performance cores in two 12 MiB
    /// L2 clusters.
    fn m1_pro() -> HashMap<&'static str, u64> {
        HashMap::from([
            ("hw.logicalcpu", 10),
            ("hw.physicalcpu", 10),
            ("hw.nperflevels", 2),
            ("hw.perflevel0.logicalcpu", 8),
            ("hw.perflevel0.l1dcachesize", 128 * 1024),
            ("hw.perflevel0.l2cachesize", 12 * 1024 * 1024),
            ("hw.perflevel0.cpusperl2", 4),
            ("hw.perflevel1.logicalcpu", 2),
            ("hw.perflevel1.l1dcachesize", 64 * 1024),
            ("hw.perflevel1.l2cachesize", 4 * 1024 * 1024),
            ("hw.perflevel1.cpusperl2", 2),
            ("hw.l1dcachesize", 64 * 1024),
            ("hw.l2cachesize", 4 * 1024 * 1024),
            ("hw.cachelinesize", 128),
        ])
    }

    #[test]
    fn efficiency_cores_come_first_with_their_own_caches() {
        let input = collect(&m1_pro(), 1);
        let kinds: Vec<CoreKind> = input.cpus.iter().map(|cpu| cpu.kind).collect();
        assert_eq!(&kinds[..2], [CoreKind::Efficiency; 2]);
        assert_eq!(&kinds[2..], [CoreKind::Performance; 8]);
        assert_eq!(input.cpus[9].core_key, "macos:core:00000009");

        let l2: Vec<(u64, Vec<usize>)> = input
            .caches
            .iter()
            .filter(|cache| cache.level == 2)
            .map(|cache| (cache.size_bytes, cache.sharers.iter().collect()))
            .collect();
        assert_eq!(
            l2,
            [
                (4 * 1024 * 1024, vec![0, 1]),
                (12 * 1024 * 1024, vec![2, 3, 4, 5]),
                (12 * 1024 * 1024, vec![6, 7, 8, 9]),
            ]
        );
        let l1: Vec<u64> = input
            .caches
            .iter()
            .filter(|c| c.level == 1)
            .map(|c| c.size_bytes)
            .collect();
        assert_eq!(l1, [[64 * 1024; 2].as_slice(), &[128 * 1024; 8]].concat());
        assert!(
            input
                .caches
                .iter()
                .all(|cache| cache.line_size_bytes == 128 && cache.level != 3)
        );

        let topology = source::normalize(input).unwrap();
        assert_eq!(topology.cpu_span(), 10);
    }

    #[test]
    fn mismatched_levels_are_homogeneous_with_smt_siblings() {
        let sysctl = HashMap::from([
            ("hw.logicalcpu", 8),
            ("hw.physicalcpu", 4),
            ("hw.nperflevels", 2),
            ("hw.perflevel0.logicalcpu", 3),
            ("hw.perflevel1.logicalcpu", 3),
            ("hw.l1dcachesize", 32 * 1024),
            ("hw.l2cachesize", 256 * 1024),
            ("hw.perflevel0.cpusperl2", 2),
            ("hw.l3cachesize", 8 * 1024 * 1024),
        ]);
        let input = collect(&sysctl, 1);
        assert!(input.cpus.iter().all(|cpu| cpu.kind == CoreKind::Unknown));
        assert_eq!(input.cpus[1].core_key, input.cpus[0].core_key);
        assert_ne!(input.cpus[2].core_key, input.cpus[1].core_key);
        let sharers = |level| -> Vec<Vec<usize>> {
            input
                .caches
                .iter()
                .filter(|cache| cache.level == level)
                .map(|cache| cache.sharers.iter().collect())
                .collect()
        };
        assert_eq!(sharers(1), [vec![0, 1], vec![2, 3], vec![4, 5], vec![6, 7]]);
        assert_eq!(sharers(2), sharers(1));
        assert_eq!(sharers(3), [(0..8).collect::<Vec<_>>()]);
        assert_eq!(input.caches[0].line_size_bytes, 64);
        assert!(source::normalize(input).is_ok());
    }

    #[test]
    fn missing_sysctls_use_the_processor_count() {
        let input = collect(&HashMap::new(), 3);
        assert_eq!(input.cpus.len(), 3);
        assert!(input.caches.is_empty());
        assert_eq!(discover(&HashMap::new(), 3).cpu_span(), 3);
    }

    #[test]
    fn groups_cover_the_range_exactly() {
        let all: Vec<Vec<usize>> = groups(2, 5, 2).map(|set| set.iter().collect()).collect();
        assert_eq!(all, [vec![2, 3], vec![4, 5], vec![6]]);
        assert_eq!(groups(0, 3, 0).count(), 3);
        assert_eq!(groups(0, 0, 4).count(), 0);
    }
}
