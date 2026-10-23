//! Windows topology from `GetLogicalProcessorInformationEx(RelationAll)` (upstream
//! `WindowsSystemLayout` and its `win32` records).
//!
//! The buffer is parsed here rather than through the Win32 structs so the layout builds and is tested
//! on every platform. Logical CPU `n` of processor group `g` has id `g * 64 + n`.

use crate::topology::Topology;
use crate::topology::source::{self, CacheDomain, CoreKind, LogicalCpu, TopologyInput};
use crate::util::CpuSet;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

const RELATION_PROCESSOR_CORE: u32 = 0;
const RELATION_CACHE: u32 = 2;
const RELATION_PROCESSOR_PACKAGE: u32 = 3;

const CACHE_UNIFIED: u32 = 0;
const CACHE_DATA: u32 = 2;

/// `KAFFINITY` is pointer-sized, so a `GROUP_AFFINITY` is 16 bytes on 64-bit Windows and 12 on
/// 32-bit.
const KAFFINITY_BYTES: usize = std::mem::size_of::<usize>();
const GROUP_AFFINITY_BYTES: usize = KAFFINITY_BYTES + 8;

/// The topology a `GetLogicalProcessorInformationEx` buffer describes, or the fallback when there is
/// none or it is invalid.
pub(crate) fn discover(buffer: Option<&[u8]>, processors: usize) -> Topology {
    let input = buffer
        .ok_or_else(|| "no processor information".to_owned())
        .and_then(|buffer| {
            parse(buffer).ok_or_else(|| "malformed processor information".to_owned())
        })
        .and_then(|records| translate(&records))
        .and_then(|input| source::normalize(input).map_err(|error| error.to_string()));
    input.unwrap_or_else(|error| {
        log::error!("failed to read the Windows topology ({error}); using the fallback topology");
        source::fallback(processors)
    })
}

/// One `GROUP_AFFINITY`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GroupAffinity {
    pub(crate) group: u16,
    pub(crate) mask: u64,
}

impl GroupAffinity {
    fn cpus(self) -> impl Iterator<Item = usize> {
        let base = usize::from(self.group) * 64;
        (0..64)
            .filter(move |bit| self.mask & (1 << bit) != 0)
            .map(move |bit| base + bit)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Record {
    Core {
        efficiency_class: u8,
        groups: Vec<GroupAffinity>,
    },
    Package {
        groups: Vec<GroupAffinity>,
    },
    /// A data or unified cache.
    Cache {
        level: u8,
        line_size: u16,
        size: u32,
        groups: Vec<GroupAffinity>,
    },
}

fn u16_at(buffer: &[u8], at: usize) -> Option<u16> {
    Some(u16::from_le_bytes(buffer.get(at..at + 2)?.try_into().ok()?))
}

fn u32_at(buffer: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_le_bytes(buffer.get(at..at + 4)?.try_into().ok()?))
}

/// `count` group affinities starting at `at`. Before Windows 11 a cache record's count field was
/// reserved and zero, with exactly one affinity present, so a zero count reads one.
fn affinities(payload: &[u8], count_at: usize, at: usize) -> Option<Vec<GroupAffinity>> {
    let count = usize::from(u16_at(payload, count_at)?).max(1);
    (0..count)
        .map(|index| {
            let start = at + index * GROUP_AFFINITY_BYTES;
            let mask = payload.get(start..start + KAFFINITY_BYTES)?;
            let mut bytes = [0u8; 8];
            bytes[..KAFFINITY_BYTES].copy_from_slice(mask);
            Some(GroupAffinity {
                group: u16_at(payload, start + KAFFINITY_BYTES)?,
                mask: u64::from_le_bytes(bytes),
            })
        })
        .collect()
}

/// Parses a `SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX` buffer; `None` when any record is malformed.
pub(crate) fn parse(buffer: &[u8]) -> Option<Vec<Record>> {
    let mut records = Vec::new();
    let mut at = 0;
    while at < buffer.len() {
        let relationship = u32_at(buffer, at)?;
        let size = usize::try_from(u32_at(buffer, at + 4)?).ok()?;
        if size < 8 {
            return None;
        }
        let payload = buffer.get(at + 8..at.checked_add(size)?)?;
        match relationship {
            RELATION_PROCESSOR_CORE | RELATION_PROCESSOR_PACKAGE => {
                let groups = affinities(payload, 22, 24)?;
                records.push(if relationship == RELATION_PROCESSOR_CORE {
                    Record::Core {
                        efficiency_class: *payload.get(1)?,
                        groups,
                    }
                } else {
                    Record::Package { groups }
                });
            }
            RELATION_CACHE => {
                let kind = u32_at(payload, 8)?;
                let groups = affinities(payload, 30, 32)?;
                if matches!(kind, CACHE_UNIFIED | CACHE_DATA) {
                    records.push(Record::Cache {
                        level: *payload.first()?,
                        line_size: u16_at(payload, 2)?,
                        size: u32_at(payload, 4)?,
                        groups,
                    });
                }
            }
            _ => {}
        }
        at += size;
    }
    Some(records)
}

/// `windows:<kind>:g<group>=<mask>;…` over the nonzero affinities in group order.
fn signature(kind: &str, groups: &[GroupAffinity]) -> Result<String, String> {
    let nonzero: BTreeMap<u16, u64> = groups
        .iter()
        .filter(|affinity| affinity.mask != 0)
        .map(|affinity| (affinity.group, affinity.mask))
        .collect();
    if nonzero.is_empty() {
        return Err(format!("empty windows {kind} affinity"));
    }
    let mut key = format!("windows:{kind}:");
    for (index, (group, mask)) in nonzero.iter().enumerate() {
        if index > 0 {
            key.push(';');
        }
        let _ = write!(key, "g{group}={mask:016x}");
    }
    Ok(key)
}

/// Builds the topology input. A CPU outside exactly one package, or claimed by two cores, makes the
/// whole description invalid.
pub(crate) fn translate(records: &[Record]) -> Result<TopologyInput, String> {
    let mut package_of: BTreeMap<usize, BTreeSet<String>> = BTreeMap::new();
    for record in records {
        if let Record::Package { groups } = record {
            let key = signature("package", groups)?;
            for cpu in groups.iter().flat_map(|affinity| affinity.cpus()) {
                package_of.entry(cpu).or_default().insert(key.clone());
            }
        }
    }
    let classes: BTreeSet<u8> = records
        .iter()
        .filter_map(|record| match record {
            Record::Core {
                efficiency_class, ..
            } => Some(*efficiency_class),
            _ => None,
        })
        .collect();
    // A higher efficiency class is a faster core; only the fastest class counts as performance.
    let fastest = classes.last().copied().filter(|_| classes.len() > 1);

    let mut cpus: BTreeMap<usize, LogicalCpu> = BTreeMap::new();
    let mut caches = Vec::new();
    for record in records {
        match record {
            Record::Core {
                efficiency_class,
                groups,
            } => {
                let core_key = signature("core", groups)?;
                let kind = match fastest {
                    Some(fastest) if *efficiency_class == fastest => CoreKind::Performance,
                    Some(_) => CoreKind::Efficiency,
                    None => CoreKind::Unknown,
                };
                for id in groups.iter().flat_map(|affinity| affinity.cpus()) {
                    let socket_key = match package_of.get(&id) {
                        Some(owners) if owners.len() == 1 => {
                            owners.first().cloned().unwrap_or_default()
                        }
                        _ => return Err(format!("windows CPU {id} must have exactly one package")),
                    };
                    let cpu = LogicalCpu {
                        id,
                        socket_key,
                        die_key: "windows:die:0".to_owned(),
                        core_key: core_key.clone(),
                        kind,
                    };
                    if cpus
                        .insert(id, cpu.clone())
                        .is_some_and(|prior| prior != cpu)
                    {
                        return Err(format!("windows CPU {id} has conflicting core ownership"));
                    }
                }
            }
            Record::Cache {
                level,
                line_size,
                size,
                groups,
            } => {
                let mut words = Vec::new();
                for cpu in groups.iter().flat_map(|affinity| affinity.cpus()) {
                    crate::util::cpu_set::set_bit(&mut words, cpu);
                }
                caches.push(CacheDomain {
                    level: *level,
                    size_bytes: u64::from(*size),
                    line_size_bytes: u32::from(*line_size),
                    sharers: CpuSet::from_owned_words(words),
                });
            }
            Record::Package { .. } => {}
        }
    }
    Ok(TopologyInput {
        provider: "windows".to_owned(),
        cpus: cpus.into_values().collect(),
        caches,
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::topology::source;

    fn affinity_bytes(groups: &[(u16, u64)]) -> Vec<u8> {
        let mut bytes = Vec::new();
        for &(group, mask) in groups {
            bytes.extend_from_slice(&mask.to_le_bytes()[..KAFFINITY_BYTES]);
            bytes.extend_from_slice(&group.to_le_bytes());
            bytes.extend_from_slice(&[0; 6]);
        }
        bytes
    }

    fn record(relationship: u32, payload: Vec<u8>) -> Vec<u8> {
        let mut bytes = relationship.to_le_bytes().to_vec();
        bytes.extend_from_slice(&(payload.len() as u32 + 8).to_le_bytes());
        bytes.extend(payload);
        bytes
    }

    pub(crate) fn processor(relationship: u32, class: u8, groups: &[(u16, u64)]) -> Vec<u8> {
        let mut payload = vec![0u8; 24];
        payload[1] = class;
        payload[22..24].copy_from_slice(&(groups.len() as u16).to_le_bytes());
        payload.extend(affinity_bytes(groups));
        record(relationship, payload)
    }

    /// `count` overrides the group count field, as older Windows leaves it zero.
    pub(crate) fn cache(
        level: u8,
        kind: u32,
        size: u32,
        groups: &[(u16, u64)],
        count: u16,
    ) -> Vec<u8> {
        let mut payload = vec![0u8; 32];
        payload[0] = level;
        payload[2..4].copy_from_slice(&64u16.to_le_bytes());
        payload[4..8].copy_from_slice(&size.to_le_bytes());
        payload[8..12].copy_from_slice(&kind.to_le_bytes());
        payload[30..32].copy_from_slice(&count.to_le_bytes());
        payload.extend(affinity_bytes(groups));
        record(RELATION_CACHE, payload)
    }

    /// A hybrid package: two SMT performance cores (class 2), two efficiency cores (class 1) and one
    /// low-power core (class 0).
    pub(crate) fn hybrid() -> Vec<u8> {
        [
            processor(RELATION_PROCESSOR_PACKAGE, 0, &[(0, 0x7f)]),
            processor(RELATION_PROCESSOR_CORE, 2, &[(0, 0b11)]),
            processor(RELATION_PROCESSOR_CORE, 2, &[(0, 0b1100)]),
            processor(RELATION_PROCESSOR_CORE, 1, &[(0, 0b1_0000)]),
            processor(RELATION_PROCESSOR_CORE, 1, &[(0, 0b10_0000)]),
            processor(RELATION_PROCESSOR_CORE, 0, &[(0, 0b100_0000)]),
            cache(1, CACHE_DATA, 48 * 1024, &[(0, 0b11)], 1),
            cache(1, 1, 32 * 1024, &[(0, 0b11)], 1),
            cache(2, CACHE_UNIFIED, 2 << 20, &[(0, 0b1111)], 0),
            cache(3, 3, 1 << 20, &[(0, 0x7f)], 1),
            record(4, vec![0; 16]),
        ]
        .concat()
    }

    #[test]
    fn hybrid_cores_and_caches() {
        let records = parse(&hybrid()).unwrap();
        let input = translate(&records).unwrap();
        let kinds: Vec<CoreKind> = input.cpus.iter().map(|cpu| cpu.kind).collect();
        assert_eq!(
            kinds,
            [
                CoreKind::Performance,
                CoreKind::Performance,
                CoreKind::Performance,
                CoreKind::Performance,
                CoreKind::Efficiency,
                CoreKind::Efficiency,
                CoreKind::Efficiency,
            ]
        );
        assert_eq!(input.cpus[1].core_key, "windows:core:g0=0000000000000003");
        assert_eq!(
            input.cpus[0].socket_key,
            "windows:package:g0=000000000000007f"
        );
        // The instruction and trace caches are dropped; the pre-Windows 11 L2 record still reads.
        let caches: Vec<(u8, Vec<usize>)> = input
            .caches
            .iter()
            .map(|cache| (cache.level, cache.sharers.iter().collect()))
            .collect();
        assert_eq!(caches, [(1, vec![0, 1]), (2, vec![0, 1, 2, 3])]);
        assert_eq!(source::normalize(input).unwrap().cpu_span(), 7);
    }

    #[test]
    fn second_group_cpus_start_at_64() {
        let buffer = [
            processor(RELATION_PROCESSOR_PACKAGE, 0, &[(0, 0b1), (1, 0b1)]),
            processor(RELATION_PROCESSOR_CORE, 0, &[(0, 0b1)]),
            processor(RELATION_PROCESSOR_CORE, 0, &[(1, 0b1)]),
        ]
        .concat();
        let input = translate(&parse(&buffer).unwrap()).unwrap();
        let ids: Vec<usize> = input.cpus.iter().map(|cpu| cpu.id).collect();
        assert_eq!(ids, [0, 64]);
        assert_eq!(
            input.cpus[1].socket_key,
            "windows:package:g0=0000000000000001;g1=0000000000000001"
        );
        assert!(input.cpus.iter().all(|cpu| cpu.kind == CoreKind::Unknown));
        assert!(source::normalize(input).is_ok());
    }

    #[test]
    fn malformed_buffers_and_inconsistent_ownership_are_rejected() {
        let good = processor(RELATION_PROCESSOR_CORE, 0, &[(0, 1)]);
        assert!(parse(&good[..good.len() - 1]).is_none());
        let mut tiny = good.clone();
        tiny[4..8].copy_from_slice(&4u32.to_le_bytes());
        assert!(parse(&tiny).is_none());
        assert_eq!(parse(&[]), Some(vec![]));

        let orphan = parse(&good).unwrap();
        assert!(
            translate(&orphan)
                .unwrap_err()
                .contains("exactly one package")
        );
        let doubled = parse(
            &[
                processor(RELATION_PROCESSOR_PACKAGE, 0, &[(0, 0b11)]),
                processor(RELATION_PROCESSOR_CORE, 0, &[(0, 0b11)]),
                processor(RELATION_PROCESSOR_CORE, 0, &[(0, 0b10)]),
            ]
            .concat(),
        )
        .unwrap();
        assert!(translate(&doubled).unwrap_err().contains("conflicting"));
        let empty = parse(&processor(RELATION_PROCESSOR_PACKAGE, 0, &[(0, 0)])).unwrap();
        assert!(translate(&empty).is_err());
    }

    #[test]
    fn discovery_falls_back_without_a_valid_buffer() {
        assert_eq!(discover(Some(&hybrid()), 1).cpu_span(), 7);
        assert_eq!(discover(None, 3).cpu_span(), 3);
        assert_eq!(discover(Some(&[1, 2, 3]), 2).cpu_span(), 2);
    }
}
