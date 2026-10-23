//! Raw topology observations and their normalization into a [`Topology`] (upstream
//! `topology.internal`: `TopologyInput`, `LogicalCpu`, `CacheDomain`, `TopologyNormalizer`,
//! `TopologyBootstrap.fallback`).
//!
//! These are public so a caller can build a fixture topology, or describe a machine this crate does
//! not probe, and hand it to everything that takes a `&Topology`.

use super::{CoreInfo, CpuCacheLayout, CpuInfo, SocketInfo, Topology};
use crate::util::CpuSet;
use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::error::Error;
use std::fmt;

/// Largest accepted logical CPU id.
pub const MAX_LOGICAL_CPU_ID: usize = 1_048_575;
/// Largest accepted number of logical CPUs.
pub const MAX_ACTIVE_CPUS: usize = 65_536;
/// Bound on the sum over cores of (highest CPU id + 1), which bounds the memory of the per-core
/// masks.
pub const MAX_CORE_INDEX_SUM: u64 = 16_777_216;

pub const DEFAULT_L1_BYTES: u64 = 32 * 1024;
pub const DEFAULT_L2_BYTES: u64 = 256 * 1024;
pub const DEFAULT_L3_BYTES: u64 = 4 * 1024 * 1024;
const DEFAULT_LINE_BYTES: u32 = 64;

/// Performance class of a core.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CoreKind {
    Performance,
    Efficiency,
    /// The platform cannot tell; treated as a performance core.
    Unknown,
}

/// One logical CPU as a provider observed it.
///
/// The keys identify the package, die and core the CPU belongs to. Each provider has one canonical
/// key encoding (see [`normalize`]), so equal keys mean the same hardware and sorting the keys gives
/// a stable numbering.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LogicalCpu {
    pub id: usize,
    pub socket_key: String,
    pub die_key: String,
    pub core_key: String,
    pub kind: CoreKind,
}

/// One data or unified cache and the logical CPUs sharing it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CacheDomain {
    /// 1, 2 or 3; other levels are ignored.
    pub level: u8,
    pub size_bytes: u64,
    pub line_size_bytes: u32,
    pub sharers: CpuSet,
}

/// Everything a provider observed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TopologyInput {
    /// `linux`, `windows`, `macos` or `fallback`; selects the key encoding that is enforced.
    pub provider: String,
    pub cpus: Vec<LogicalCpu>,
    pub caches: Vec<CacheDomain>,
}

/// Why a [`TopologyInput`] was rejected (upstream `TopologyValidationException`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TopologyValidationError {
    pub provider: String,
    pub category: &'static str,
    pub value: String,
    pub reason: String,
}

impl fmt::Display for TopologyValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "topology provider={} category={} value={}: {}",
            self.provider, self.category, self.value, self.reason
        )
    }
}

impl Error for TopologyValidationError {}

fn failure(
    provider: &str,
    category: &'static str,
    value: impl Into<String>,
    reason: impl Into<String>,
) -> TopologyValidationError {
    TopologyValidationError {
        provider: provider.to_string(),
        category,
        value: value.into(),
        reason: reason.into(),
    }
}

/// The topology used when the platform cannot be probed: one package, one core per CPU, private L1
/// and L2, a shared L3, CPUs `0..processor_count` (at least one).
pub fn fallback(processor_count: usize) -> Topology {
    let cpus = (0..processor_count.max(1))
        .map(|cpu| LogicalCpu {
            id: cpu,
            socket_key: "fallback:package:0".to_string(),
            die_key: "fallback:die:0".to_string(),
            core_key: format!("fallback:core:{cpu:08x}"),
            kind: CoreKind::Unknown,
        })
        .collect();
    normalize(TopologyInput {
        provider: "fallback".to_string(),
        cpus,
        caches: Vec::new(),
    })
    .expect("the fallback topology is always valid")
}

/// Validates provider observations and numbers sockets and cores densely.
///
/// Sockets are numbered in key order, and cores in (socket, die key, core key) order, so the result
/// does not depend on the order the provider listed CPUs in. Logical CPU ids are kept as the OS
/// reports them and may be sparse. A cache level a CPU has no unique observation for falls back to
/// its core (L1, L2) or socket (L3) with a default size; a cache shared across sockets is dropped.
///
/// Key encodings, all lowercase ASCII:
/// - `linux`: `linux:package:` then ten decimal digits or `unknown`; `linux:die:` then ten digits;
///   `linux:core:` then ten digits or `cpu` and seven digits (a CPU without a core id).
/// - `windows`: `windows:package:` / `windows:core:` then `g<group>=<16 hex digits>` items joined
///   by `;`; the die is `windows:die:0`.
/// - `macos` and `fallback`: `<p>:package:0`, `<p>:die:0`, `<p>:core:` then 8 hex digits.
pub fn normalize(source: TopologyInput) -> Result<Topology, TopologyValidationError> {
    let provider = source.provider.as_str();
    validate_key(provider, provider, "provider")?;
    if source.cpus.is_empty() {
        return Err(failure(
            provider,
            "logical-cpus",
            "empty",
            "at least one CPU is required",
        ));
    }
    if source.cpus.len() > MAX_ACTIVE_CPUS {
        return Err(failure(
            provider,
            "active-count",
            source.cpus.len().to_string(),
            format!("active CPU count exceeds {MAX_ACTIVE_CPUS}"),
        ));
    }
    let mut ids = HashSet::with_capacity(source.cpus.len());
    for cpu in &source.cpus {
        if cpu.id > MAX_LOGICAL_CPU_ID {
            return Err(failure(
                provider,
                "logical-id",
                cpu.id.to_string(),
                format!("logical ID is outside [0,{MAX_LOGICAL_CPU_ID}]"),
            ));
        }
        if !ids.insert(cpu.id) {
            return Err(failure(
                provider,
                "logical-id",
                cpu.id.to_string(),
                "logical ID is duplicated",
            ));
        }
        validate_key(provider, &cpu.socket_key, "socket-key")?;
        validate_key(provider, &cpu.die_key, "die-key")?;
        validate_key(provider, &cpu.core_key, "core-key")?;
        if !valid_encoding(provider, cpu) {
            return Err(failure(
                provider,
                "source-key",
                cpu.id.to_string(),
                "keys do not match the provider's canonical encoding",
            ));
        }
    }
    let mut cpus = source.cpus;
    cpus.sort_by_key(|cpu| cpu.id);

    let mut socket_keys: Vec<&str> = cpus.iter().map(|cpu| cpu.socket_key.as_str()).collect();
    socket_keys.sort_unstable_by(|a, b| compare_keys(a, b));
    socket_keys.dedup();
    let sockets: HashMap<&str, usize> = socket_keys
        .iter()
        .enumerate()
        .map(|(i, key)| (*key, i))
        .collect();

    let mut core_keys: Vec<(usize, &str, &str)> = cpus
        .iter()
        .map(|cpu| {
            (
                sockets[cpu.socket_key.as_str()],
                cpu.die_key.as_str(),
                cpu.core_key.as_str(),
            )
        })
        .collect();
    core_keys.sort_unstable_by(|a, b| {
        a.0.cmp(&b.0)
            .then_with(|| compare_keys(a.1, b.1))
            .then_with(|| compare_keys(a.2, b.2))
    });
    core_keys.dedup();
    let cores: HashMap<(usize, &str, &str), usize> = core_keys
        .iter()
        .enumerate()
        .map(|(i, key)| (*key, i))
        .collect();
    let core_of = |cpu: &LogicalCpu| {
        let socket = sockets[cpu.socket_key.as_str()];
        (
            socket,
            cores[&(socket, cpu.die_key.as_str(), cpu.core_key.as_str())],
        )
    };

    let mut highest_cpu_by_core = vec![0usize; core_keys.len()];
    for cpu in &cpus {
        let (_, core) = core_of(cpu);
        highest_cpu_by_core[core] = highest_cpu_by_core[core].max(cpu.id);
    }
    let index_sum: u64 = highest_cpu_by_core
        .iter()
        .map(|&highest| highest as u64 + 1)
        .sum();
    if index_sum > MAX_CORE_INDEX_SUM {
        return Err(failure(
            provider,
            "core-index-sum",
            index_sum.to_string(),
            format!("core index sum exceeds {MAX_CORE_INDEX_SUM}"),
        ));
    }

    let mut core_cpus: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    let mut socket_cpus: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    let mut socket_cores: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    let mut core_kinds: HashMap<usize, CoreKind> = HashMap::new();
    let mut cpu_info = BTreeMap::new();
    for cpu in &cpus {
        let (socket, core) = core_of(cpu);
        let previous = *core_kinds.entry(core).or_insert(cpu.kind);
        if previous != cpu.kind {
            return Err(failure(
                provider,
                "core-kind",
                cpu.core_key.clone(),
                "logical siblings disagree on core kind",
            ));
        }
        core_cpus.entry(core).or_default().push(cpu.id);
        socket_cpus.entry(socket).or_default().push(cpu.id);
        socket_cores.entry(socket).or_default().push(core);
        cpu_info.insert(
            cpu.id,
            CpuInfo {
                cpu: cpu.id,
                core,
                socket,
            },
        );
    }
    let core_cpus: BTreeMap<usize, CpuSet> = core_cpus
        .into_iter()
        .map(|(core, ids)| (core, ids.into_iter().collect()))
        .collect();
    let socket_cpus: BTreeMap<usize, CpuSet> = socket_cpus
        .into_iter()
        .map(|(socket, ids)| (socket, ids.into_iter().collect()))
        .collect();

    let mut core_info = Vec::with_capacity(core_cpus.len());
    let (mut p_cores, mut e_cores, mut p_cpus, mut e_cpus) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for (&core, members) in &core_cpus {
        let first = members.first().expect("every core has a CPU");
        let socket = cpu_info[&first].socket;
        let performance = core_kinds[&core] != CoreKind::Efficiency;
        if performance {
            p_cores.push(core);
            p_cpus.extend(members.iter());
        } else {
            e_cores.push(core);
            e_cpus.extend(members.iter());
        }
        core_info.push(CoreInfo {
            core,
            socket,
            p_core: performance,
            cpus: members.clone(),
        });
    }
    let socket_info: Vec<SocketInfo> = (0..socket_keys.len())
        .map(|socket| SocketInfo {
            socket,
            cpus: socket_cpus[&socket].clone(),
            cores: socket_cores[&socket].iter().copied().collect(),
        })
        .collect();

    let active: CpuSet = cpus.iter().map(|cpu| cpu.id).collect();
    let domains = canonical_domains(&source.caches, &active, &cpu_info);
    let mut cache_layout = BTreeMap::new();
    let mut max_line = DEFAULT_LINE_BYTES;
    for cpu in &cpus {
        let identity = cpu_info[&cpu.id];
        let core_mask = &core_cpus[&identity.core];
        let socket_mask = &socket_cpus[&identity.socket];
        let l1 = select(&domains, 1, cpu.id, core_mask, DEFAULT_L1_BYTES);
        let l2 = select(&domains, 2, cpu.id, core_mask, DEFAULT_L2_BYTES);
        let l3 = select(&domains, 3, cpu.id, socket_mask, DEFAULT_L3_BYTES);
        let line = l1.line.max(l2.line).max(l3.line);
        max_line = max_line.max(line);
        cache_layout.insert(
            cpu.id,
            CpuCacheLayout {
                cpu: cpu.id,
                bytes_l1: l1.bytes,
                bytes_l2: l2.bytes,
                bytes_l3: l3.bytes,
                mask_l1: l1.mask,
                mask_l2: l2.mask,
                mask_l3: l3.mask,
                cache_line_bytes: line,
            },
        );
    }

    let core_set: CpuSet = (0..core_info.len()).collect();
    Ok(Topology {
        active_ids: cpus.iter().map(|cpu| cpu.id).collect(),
        cpu_span: cpus.last().expect("at least one CPU").id + 1,
        cpu_set: active,
        core_set,
        p_core_set: p_cores.into_iter().collect(),
        e_core_set: e_cores.into_iter().collect(),
        p_cpu_set: p_cpus.into_iter().collect(),
        e_cpu_set: e_cpus.into_iter().collect(),
        cache_layout,
        cpu_info,
        core_info,
        socket_info,
        cache_line_bytes: max_line,
    })
}

struct Domain {
    level: u8,
    bytes: u64,
    line: u32,
    mask: CpuSet,
}

struct Selection {
    bytes: u64,
    line: u32,
    mask: CpuSet,
}

fn canonical_domains(
    input: &[CacheDomain],
    active: &CpuSet,
    identities: &BTreeMap<usize, CpuInfo>,
) -> Vec<Domain> {
    let mut unique: BTreeMap<(u8, u64, u32, Vec<u64>), Domain> = BTreeMap::new();
    for value in input {
        if !(1..=3).contains(&value.level) || value.size_bytes == 0 {
            continue;
        }
        let mask = value.sharers.intersection(active);
        if mask.is_empty() {
            continue;
        }
        let mut sockets = mask
            .iter()
            .filter_map(|cpu| identities.get(&cpu))
            .map(|info| info.socket);
        let first = sockets.next();
        if sockets.any(|socket| Some(socket) != first) {
            continue;
        }
        let line = if valid_line(value.line_size_bytes) {
            value.line_size_bytes
        } else {
            DEFAULT_LINE_BYTES
        };
        let key = (value.level, value.size_bytes, line, mask.words().to_vec());
        unique.insert(
            key,
            Domain {
                level: value.level,
                bytes: value.size_bytes,
                line,
                mask,
            },
        );
    }
    unique.into_values().collect()
}

fn select(
    domains: &[Domain],
    level: u8,
    cpu: usize,
    fallback_mask: &CpuSet,
    fallback_bytes: u64,
) -> Selection {
    let mut matches = domains
        .iter()
        .filter(|domain| domain.level == level && domain.mask.contains(cpu));
    match (matches.next(), matches.next()) {
        (Some(domain), None) => Selection {
            bytes: domain.bytes,
            line: domain.line,
            mask: domain.mask.clone(),
        },
        _ => Selection {
            bytes: fallback_bytes,
            line: DEFAULT_LINE_BYTES,
            mask: fallback_mask.clone(),
        },
    }
}

fn valid_line(line: u32) -> bool {
    (16..=1024).contains(&line) && line.is_power_of_two()
}

fn validate_key(
    provider: &str,
    key: &str,
    category: &'static str,
) -> Result<(), TopologyValidationError> {
    if key.trim().is_empty() || key != key.trim() {
        return Err(failure(
            provider,
            category,
            key,
            "key must be nonblank and trimmed",
        ));
    }
    if !key.bytes().all(|b| b.is_ascii() && !b.is_ascii_uppercase()) {
        return Err(failure(
            provider,
            category,
            key,
            "key must be lowercase ASCII",
        ));
    }
    Ok(())
}

fn valid_encoding(provider: &str, cpu: &LogicalCpu) -> bool {
    let digits = |s: &str, n: usize| s.len() == n && s.bytes().all(|b| b.is_ascii_digit());
    let hex = |s: &str, n: usize| {
        s.len() == n && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
    };
    let groups = |s: &str| {
        s.split(';').all(|item| {
            item.strip_prefix('g')
                .and_then(|rest| rest.split_once('='))
                .is_some_and(|(group, mask)| {
                    !group.is_empty() && digits(group, group.len()) && hex(mask, 16)
                })
        })
    };
    match provider {
        "linux" => {
            cpu.socket_key
                .strip_prefix("linux:package:")
                .is_some_and(|rest| rest == "unknown" || digits(rest, 10))
                && cpu
                    .die_key
                    .strip_prefix("linux:die:")
                    .is_some_and(|rest| digits(rest, 10))
                && cpu
                    .core_key
                    .strip_prefix("linux:core:")
                    .is_some_and(|rest| {
                        digits(rest, 10) || rest.strip_prefix("cpu").is_some_and(|id| digits(id, 7))
                    })
        }
        "windows" => {
            cpu.socket_key
                .strip_prefix("windows:package:")
                .is_some_and(groups)
                && cpu.die_key == "windows:die:0"
                && cpu
                    .core_key
                    .strip_prefix("windows:core:")
                    .is_some_and(groups)
        }
        "macos" | "fallback" => {
            cpu.socket_key
                .strip_prefix(provider)
                .is_some_and(|rest| rest == ":package:0")
                && cpu
                    .die_key
                    .strip_prefix(provider)
                    .is_some_and(|rest| rest == ":die:0")
                && cpu
                    .core_key
                    .strip_prefix(provider)
                    .and_then(|rest| rest.strip_prefix(":core:"))
                    .is_some_and(|rest| hex(rest, 8))
        }
        _ => false,
    }
}

/// Unsigned byte-wise order; every canonical encoding is fixed-width where it is numeric, so this is
/// also numeric order.
fn compare_keys(left: &str, right: &str) -> Ordering {
    left.as_bytes().cmp(right.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fallback_cpu(id: usize, core: usize, kind: CoreKind) -> LogicalCpu {
        LogicalCpu {
            id,
            socket_key: "fallback:package:0".to_string(),
            die_key: "fallback:die:0".to_string(),
            core_key: format!("fallback:core:{core:08x}"),
            kind,
        }
    }

    fn input(cpus: Vec<LogicalCpu>) -> TopologyInput {
        TopologyInput {
            provider: "fallback".to_string(),
            cpus,
            caches: Vec::new(),
        }
    }

    #[test]
    fn accepts_the_maximum_logical_id_and_rejects_the_next() {
        let id = MAX_LOGICAL_CPU_ID;
        let topology = normalize(input(vec![fallback_cpu(id, id, CoreKind::Unknown)])).unwrap();
        assert_eq!(topology.cpu_span(), 1_048_576);
        let error = normalize(input(vec![fallback_cpu(id + 1, 0, CoreKind::Unknown)])).unwrap_err();
        assert_eq!(error.category, "logical-id");
    }

    #[test]
    fn rejects_duplicate_ids_and_conflicting_sibling_kinds() {
        let cpu = fallback_cpu(0, 0, CoreKind::Performance);
        assert_eq!(
            normalize(input(vec![cpu.clone(), cpu]))
                .unwrap_err()
                .category,
            "logical-id"
        );
        let siblings = vec![
            fallback_cpu(0, 0, CoreKind::Performance),
            fallback_cpu(1, 0, CoreKind::Efficiency),
        ];
        assert_eq!(
            normalize(input(siblings)).unwrap_err().category,
            "core-kind"
        );
    }

    #[test]
    fn rejects_the_index_budget_before_building_per_core_masks() {
        let cpus = (0..18)
            .map(|core| fallback_cpu(1_000_000 + core, core, CoreKind::Unknown))
            .collect();
        assert_eq!(
            normalize(input(cpus)).unwrap_err().category,
            "core-index-sum"
        );
    }

    #[test]
    fn rejects_more_active_cpus_than_the_bound_before_anything_else() {
        let cpus = vec![fallback_cpu(0, 0, CoreKind::Unknown); MAX_ACTIVE_CPUS + 1];
        assert_eq!(normalize(input(cpus)).unwrap_err().category, "active-count");
        assert_eq!(
            normalize(input(Vec::new())).unwrap_err().category,
            "logical-cpus"
        );
    }

    #[test]
    fn shuffled_input_produces_an_identical_topology() {
        let mut cpus = vec![
            fallback_cpu(8, 8, CoreKind::Unknown),
            fallback_cpu(0, 0, CoreKind::Unknown),
            fallback_cpu(2, 2, CoreKind::Unknown),
        ];
        let first = normalize(input(cpus.clone())).unwrap();
        cpus.reverse();
        assert_eq!(normalize(input(cpus)).unwrap(), first);
    }

    #[test]
    fn rejects_keys_outside_the_provider_encoding() {
        let mut cpu = fallback_cpu(0, 0, CoreKind::Unknown);
        cpu.core_key = "fallback:core:0".to_string();
        assert_eq!(
            normalize(input(vec![cpu])).unwrap_err().category,
            "source-key"
        );
        let mut upper = fallback_cpu(0, 0, CoreKind::Unknown);
        upper.die_key = "fallback:Die:0".to_string();
        assert_eq!(
            normalize(input(vec![upper])).unwrap_err().category,
            "die-key"
        );
        let mut linux = fallback_cpu(0, 0, CoreKind::Unknown);
        linux.socket_key = "linux:package:0000000000".to_string();
        let error = normalize(TopologyInput {
            provider: "linux".to_string(),
            cpus: vec![linux],
            caches: Vec::new(),
        })
        .unwrap_err();
        assert_eq!(error.category, "source-key");
    }

    #[test]
    fn caches_resolve_to_unique_observations_or_documented_defaults() {
        let cpus = vec![
            fallback_cpu(0, 0, CoreKind::Unknown),
            fallback_cpu(1, 0, CoreKind::Unknown),
            fallback_cpu(2, 1, CoreKind::Unknown),
        ];
        let caches = vec![
            CacheDomain {
                level: 1,
                size_bytes: 48 * 1024,
                line_size_bytes: 64,
                sharers: [0, 1].into_iter().collect(),
            },
            // Two different L2 observations for CPU 2 are ambiguous, so it uses the default.
            CacheDomain {
                level: 2,
                size_bytes: 1 << 20,
                line_size_bytes: 128,
                sharers: [2].into_iter().collect(),
            },
            CacheDomain {
                level: 2,
                size_bytes: 2 << 20,
                line_size_bytes: 128,
                sharers: [2].into_iter().collect(),
            },
            // Inactive sharers are dropped; an invalid line size becomes 64.
            CacheDomain {
                level: 3,
                size_bytes: 8 << 20,
                line_size_bytes: 3,
                sharers: [0, 1, 2, 9].into_iter().collect(),
            },
        ];
        let topology = normalize(TopologyInput {
            provider: "fallback".to_string(),
            cpus,
            caches,
        })
        .unwrap();
        let cpu0 = topology.cache_layout(0).unwrap();
        assert_eq!((cpu0.bytes_l1, cpu0.shares_l1()), (48 * 1024, 2));
        assert_eq!(
            (cpu0.bytes_l2, cpu0.mask_l2.clone()),
            (DEFAULT_L2_BYTES, [0, 1].into_iter().collect())
        );
        assert_eq!((cpu0.bytes_l3, cpu0.shares_l3()), (8 << 20, 3));
        let cpu2 = topology.cache_layout(2).unwrap();
        assert_eq!(cpu2.bytes_l1, DEFAULT_L1_BYTES);
        assert_eq!(
            (cpu2.bytes_l2, cpu2.cache_line_bytes),
            (DEFAULT_L2_BYTES, 64)
        );
        assert_eq!(topology.cache_line_bytes(), 64);
        assert_eq!(topology.socket_l3_cache_bytes(0), 8 << 20);
    }

    #[test]
    fn fallback_has_one_socket_and_one_core_per_cpu() {
        let topology = fallback(0);
        assert_eq!(
            (
                topology.cpu_span(),
                topology.core_count(),
                topology.socket_count()
            ),
            (1, 1, 1)
        );
        let topology = fallback(4);
        assert_eq!(topology.core_count(), 4);
        assert_eq!(topology.cpu_set(), &CpuSet::range(0, 4));
        assert_eq!(topology.p_core_set(), &CpuSet::range(0, 4));
        assert!(topology.e_cpu_set().is_empty());
    }
}
