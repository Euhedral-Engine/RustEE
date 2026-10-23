//! CPU topology: logical CPUs, cores, sockets and caches (upstream `topology`: `SystemInfo`,
//! `TopologyMapper`).
//!
//! A [`Topology`] is an ordinary immutable value: [`Topology::system`] probes the host once, lazily,
//! and everything else takes a `&Topology`.
//!
//! Logical CPU ids are the OS ids and may be sparse. [`Topology::cpu_span`] is the highest id plus
//! one (upstream `getCpuCount()`); size id-indexed arrays by it and iterate [`Topology::cpu_set`]
//! instead of assuming `0..span` is dense. Lookups for an id that is not active return `None`.

mod mapper;
pub mod mask;
pub mod source;

pub use mapper::{EffectiveSocketTopology, EffectiveSystemTopology, TopologyMapper};
pub use mask::{MaskError, parse_hex_mask, to_hex_mask};

use crate::platform;
use crate::util::CpuSet;
use std::collections::{BTreeMap, HashSet};
use std::error::Error;
use std::fmt;
use std::sync::OnceLock;

/// Where one logical CPU sits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CpuInfo {
    pub cpu: usize,
    pub core: usize,
    pub socket: usize,
}

/// One core and its logical CPUs.
///
/// [`CoreInfo::cpu_hex_mask`] formats the members as a hex mask.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CoreInfo {
    pub core: usize,
    pub socket: usize,
    /// False only for a core the platform classified as an efficiency core.
    pub p_core: bool,
    pub cpus: CpuSet,
}

impl CoreInfo {
    pub fn cpu_hex_mask(&self) -> String {
        to_hex_mask(&self.cpus)
    }
}

/// One socket (package) and its logical CPUs and cores.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SocketInfo {
    pub socket: usize,
    pub cpus: CpuSet,
    pub cores: CpuSet,
}

impl SocketInfo {
    pub fn cpu_hex_mask(&self) -> String {
        to_hex_mask(&self.cpus)
    }

    pub fn core_hex_mask(&self) -> String {
        to_hex_mask(&self.cores)
    }
}

/// The caches one logical CPU uses. A mask with one member means the CPU has that level to itself.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CpuCacheLayout {
    pub cpu: usize,
    pub bytes_l1: u64,
    pub bytes_l2: u64,
    pub bytes_l3: u64,
    pub mask_l1: CpuSet,
    pub mask_l2: CpuSet,
    pub mask_l3: CpuSet,
    pub cache_line_bytes: u32,
}

impl CpuCacheLayout {
    pub fn shares_l1(&self) -> usize {
        self.mask_l1.len()
    }

    pub fn shares_l2(&self) -> usize {
        self.mask_l2.len()
    }

    pub fn shares_l3(&self) -> usize {
        self.mask_l3.len()
    }
}

fn format_bytes(bytes: u64) -> String {
    let kib = bytes / 1024;
    if kib < 1024 {
        format!("{kib}KB")
    } else {
        format!("{}MB", kib / 1024)
    }
}

impl fmt::Display for CpuInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CpuInfo: [cpu = {}, core = {}, socket = {}]",
            self.cpu, self.core, self.socket
        )
    }
}

impl fmt::Display for CoreInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CoreInfo: [cpuHexMask = \"{}\", pCore = {}, core = {}, socket = {}]",
            self.cpu_hex_mask(),
            self.p_core,
            self.core,
            self.socket
        )
    }
}

impl fmt::Display for SocketInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SocketInfo: [cpuHexMask = \"{}\", coreHexMask = \"{}\", socket = {}]",
            self.cpu_hex_mask(),
            self.core_hex_mask(),
            self.socket
        )
    }
}

impl fmt::Display for CpuCacheLayout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CpuCacheLayout: [cpu = {}, bytesL1 = {}, bytesL2 = {}, bytesL3 = {}, sharesL1 = {}, sharesL2 = {}, \
             sharesL3 = {}, maskL1 = \"{}\", maskL2 = \"{}\", maskL3 = \"{}\", cacheLineBytes = {}]",
            self.cpu,
            format_bytes(self.bytes_l1),
            format_bytes(self.bytes_l2),
            format_bytes(self.bytes_l3),
            self.shares_l1(),
            self.shares_l2(),
            self.shares_l3(),
            to_hex_mask(&self.mask_l1),
            to_hex_mask(&self.mask_l2),
            to_hex_mask(&self.mask_l3),
            self.cache_line_bytes
        )
    }
}

/// A lookup named an id that is not active (the error of the upstream `get*OrThrow` accessors). The
/// message lists the active ids as ranges, like `0-3,6-7`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InactiveIdError {
    pub kind: &'static str,
    pub id: usize,
    pub active: String,
}

impl fmt::Display for InactiveIdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let noun = if self.kind == "cpu" {
            "logical CPU"
        } else {
            self.kind
        };
        write!(
            f,
            "{} {} is not an active {noun} (active: {})",
            self.kind, self.id, self.active
        )
    }
}

impl Error for InactiveIdError {}

/// Linux-style range list such as `0-3,6-7` for ascending ids.
pub(crate) fn ranges(ids: impl IntoIterator<Item = usize>) -> String {
    let mut text = String::new();
    let mut open: Option<(usize, usize)> = None;
    let flush = |text: &mut String, (start, end): (usize, usize)| {
        if !text.is_empty() {
            text.push(',');
        }
        if start == end {
            text.push_str(&start.to_string());
        } else {
            text.push_str(&format!("{start}-{end}"));
        }
    };
    for id in ids {
        open = match open {
            Some((start, prev)) if id == prev + 1 => Some((start, id)),
            Some(range) => {
                flush(&mut text, range);
                Some((id, id))
            }
            None => Some((id, id)),
        };
    }
    if let Some(range) = open {
        flush(&mut text, range);
    }
    text
}

/// An immutable, normalized CPU topology (upstream `TopologyModel` behind `SystemInfo`).
///
/// Built by [`source::normalize`] (or [`source::fallback`]); [`Topology::system`] is the host's.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Topology {
    pub(crate) active_ids: Vec<usize>,
    pub(crate) cpu_span: usize,
    pub(crate) cpu_set: CpuSet,
    pub(crate) core_set: CpuSet,
    pub(crate) p_core_set: CpuSet,
    pub(crate) e_core_set: CpuSet,
    pub(crate) p_cpu_set: CpuSet,
    pub(crate) e_cpu_set: CpuSet,
    pub(crate) cache_layout: BTreeMap<usize, CpuCacheLayout>,
    pub(crate) cpu_info: BTreeMap<usize, CpuInfo>,
    pub(crate) core_info: Vec<CoreInfo>,
    pub(crate) socket_info: Vec<SocketInfo>,
    pub(crate) cache_line_bytes: u32,
}

impl Topology {
    /// The host topology, probed on first use. A platform probe that fails or is unsupported falls
    /// back to [`source::fallback`] over `available_parallelism()`, so this never fails; callers must
    /// tolerate that shape.
    pub fn system() -> &'static Topology {
        static SYSTEM: OnceLock<Topology> = OnceLock::new();
        SYSTEM.get_or_init(|| {
            let topology = platform::discover_topology();
            if log::log_enabled!(log::Level::Debug) {
                log::debug!("\n{topology}");
            }
            topology
        })
    }

    /// The highest logical CPU id plus one (upstream `getCpuCount()`). Not the number of CPUs: use
    /// [`Topology::cpu_count`] for that.
    pub fn cpu_span(&self) -> usize {
        self.cpu_span
    }

    /// The number of active logical CPUs.
    pub fn cpu_count(&self) -> usize {
        self.active_ids.len()
    }

    pub fn core_count(&self) -> usize {
        self.core_info.len()
    }

    pub fn socket_count(&self) -> usize {
        self.socket_info.len()
    }

    /// Core ids are dense, so this is `core_count() - 1`.
    pub fn max_core_id(&self) -> usize {
        self.core_info.len() - 1
    }

    /// Socket ids are dense, so this is `socket_count() - 1`.
    pub fn max_socket_id(&self) -> usize {
        self.socket_info.len() - 1
    }

    /// The largest cache line any CPU reported, at least 64.
    pub fn cache_line_bytes(&self) -> u32 {
        self.cache_line_bytes
    }

    /// Active logical CPU ids in ascending order (upstream `getSystemCpus()`).
    pub fn system_cpus(&self) -> &[usize] {
        &self.active_ids
    }

    pub fn cpu_set(&self) -> &CpuSet {
        &self.cpu_set
    }

    /// Every core id; the union of the P-core and E-core sets.
    pub fn core_set(&self) -> &CpuSet {
        &self.core_set
    }

    pub fn p_core_set(&self) -> &CpuSet {
        &self.p_core_set
    }

    pub fn e_core_set(&self) -> &CpuSet {
        &self.e_core_set
    }

    pub fn p_cpu_set(&self) -> &CpuSet {
        &self.p_cpu_set
    }

    pub fn e_cpu_set(&self) -> &CpuSet {
        &self.e_cpu_set
    }

    pub fn cpu_info(&self, cpu: usize) -> Option<&CpuInfo> {
        self.cpu_info.get(&cpu)
    }

    pub fn core_info(&self, core: usize) -> Option<&CoreInfo> {
        self.core_info.get(core)
    }

    pub fn socket_info(&self, socket: usize) -> Option<&SocketInfo> {
        self.socket_info.get(socket)
    }

    pub fn cache_layout(&self, cpu: usize) -> Option<&CpuCacheLayout> {
        self.cache_layout.get(&cpu)
    }

    pub fn cpu_infos(&self) -> impl Iterator<Item = &CpuInfo> {
        self.cpu_info.values()
    }

    pub fn core_infos(&self) -> &[CoreInfo] {
        &self.core_info
    }

    pub fn socket_infos(&self) -> &[SocketInfo] {
        &self.socket_info
    }

    pub fn cache_layouts(&self) -> impl Iterator<Item = &CpuCacheLayout> {
        self.cache_layout.values()
    }

    pub fn is_active_cpu(&self, cpu: usize) -> bool {
        self.cpu_info.contains_key(&cpu)
    }

    pub fn is_active_core(&self, core: usize) -> bool {
        core < self.core_info.len()
    }

    pub fn is_active_socket(&self, socket: usize) -> bool {
        socket < self.socket_info.len()
    }

    pub fn cpu_info_or_err(&self, cpu: usize) -> Result<&CpuInfo, InactiveIdError> {
        self.cpu_info(cpu)
            .ok_or_else(|| self.inactive("cpu", cpu, self.active_ids.iter().copied()))
    }

    pub fn core_info_or_err(&self, core: usize) -> Result<&CoreInfo, InactiveIdError> {
        self.core_info(core)
            .ok_or_else(|| self.inactive("core", core, 0..self.core_info.len()))
    }

    pub fn socket_info_or_err(&self, socket: usize) -> Result<&SocketInfo, InactiveIdError> {
        self.socket_info(socket)
            .ok_or_else(|| self.inactive("socket", socket, 0..self.socket_info.len()))
    }

    pub fn cache_layout_or_err(&self, cpu: usize) -> Result<&CpuCacheLayout, InactiveIdError> {
        self.cache_layout(cpu)
            .ok_or_else(|| self.inactive("cpu", cpu, self.active_ids.iter().copied()))
    }

    fn inactive(
        &self,
        kind: &'static str,
        id: usize,
        active: impl IntoIterator<Item = usize>,
    ) -> InactiveIdError {
        InactiveIdError {
            kind,
            id,
            active: ranges(active),
        }
    }

    /// The L3 bytes of a socket, counting each distinct L3 once; 0 for an inactive socket.
    pub fn socket_l3_cache_bytes(&self, socket: usize) -> u64 {
        let Some(info) = self.socket_info(socket) else {
            return 0;
        };
        let mut visited = HashSet::new();
        info.cpus
            .iter()
            .filter_map(|cpu| self.cache_layout(cpu))
            .filter(|layout| visited.insert(layout.mask_l3.clone()))
            .map(|layout| layout.bytes_l3)
            .sum()
    }
}

/// A multi-line dump of every socket, core, CPU and cache layout (upstream `SystemInfo.asString()`,
/// without the terminal colour codes).
impl fmt::Display for Topology {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let line = "-".repeat(80);
        writeln!(f, "{line}\nTopology\n{line}")?;
        for info in &self.socket_info {
            writeln!(f, "{info}")?;
        }
        writeln!(f, "{line}")?;
        for info in &self.core_info {
            writeln!(f, "{info}")?;
        }
        writeln!(f, "{line}")?;
        for info in self.cpu_info.values() {
            writeln!(f, "{info}")?;
        }
        writeln!(f, "{line}")?;
        for layout in self.cache_layout.values() {
            writeln!(f, "{layout}")?;
        }
        write!(f, "{line}")
    }
}

#[cfg(test)]
mod tests {
    use super::source::{CoreKind, LogicalCpu, TopologyInput, normalize};
    use super::*;

    fn sparse() -> Topology {
        let cpu = |id: usize, socket: usize, core: usize| LogicalCpu {
            id,
            socket_key: format!("linux:package:{socket:010}"),
            die_key: "linux:die:0000000000".to_string(),
            core_key: format!("linux:core:{core:010}"),
            kind: CoreKind::Unknown,
        };
        normalize(TopologyInput {
            provider: "linux".to_string(),
            cpus: vec![
                cpu(0, 0, 0),
                cpu(1, 0, 1),
                cpu(2, 0, 2),
                cpu(3, 0, 3),
                cpu(6, 1, 0),
                cpu(7, 1, 1),
            ],
            caches: Vec::new(),
        })
        .unwrap()
    }

    #[test]
    fn presence_matches_the_nullable_lookups_on_sparse_ids() {
        let topology = sparse();
        assert_eq!(topology.cpu_span(), 8);
        assert_eq!(topology.cpu_count(), 6);
        for cpu in 0..10 {
            assert_eq!(
                topology.is_active_cpu(cpu),
                topology.cpu_info(cpu).is_some(),
                "{cpu}"
            );
            assert_eq!(
                topology.cpu_info_or_err(cpu).is_ok(),
                topology.is_active_cpu(cpu)
            );
        }
        assert!(!topology.is_active_core(6));
        assert!(topology.is_active_socket(1) && !topology.is_active_socket(2));
    }

    #[test]
    fn strict_lookups_name_the_id_and_the_active_ranges() {
        let topology = sparse();
        let error = topology.cpu_info_or_err(4).unwrap_err();
        assert_eq!(
            error.to_string(),
            "cpu 4 is not an active logical CPU (active: 0-3,6-7)"
        );
        assert_eq!(
            topology.core_info_or_err(9).unwrap_err().to_string(),
            "core 9 is not an active core (active: 0-5)"
        );
        assert_eq!(
            topology.socket_info_or_err(3).unwrap_err().to_string(),
            "socket 3 is not an active socket (active: 0-1)"
        );
        assert!(topology.cache_layout_or_err(5).is_err());
    }

    #[test]
    fn ranges_formats_runs_and_singletons() {
        assert_eq!(ranges([0, 1, 2, 3, 6, 7]), "0-3,6-7");
        assert_eq!(ranges([5]), "5");
        assert_eq!(ranges([1, 3, 4]), "1,3-4");
        assert_eq!(ranges(std::iter::empty()), "");
    }

    #[test]
    fn records_display_like_upstream() {
        let topology = sparse();
        assert_eq!(
            topology.cpu_info(6).unwrap().to_string(),
            "CpuInfo: [cpu = 6, core = 4, socket = 1]"
        );
        assert_eq!(
            topology.socket_info(1).unwrap().to_string(),
            "SocketInfo: [cpuHexMask = \"c0\", coreHexMask = \"30\", socket = 1]"
        );
        assert!(
            topology
                .cache_layout(0)
                .unwrap()
                .to_string()
                .contains("bytesL3 = 4MB")
        );
        assert!(topology.to_string().lines().count() > 20);
    }

    #[test]
    fn the_system_topology_is_consistent() {
        let topology = Topology::system();
        assert!(topology.cpu_count() >= 1);
        assert_eq!(topology.cpu_set().len(), topology.cpu_count());
        assert_eq!(topology.cpu_set().span(), topology.cpu_span());
        assert_eq!(
            topology.p_core_set().union(topology.e_core_set()),
            *topology.core_set()
        );
        for cpu in topology.cpu_set() {
            let info = topology.cpu_info(cpu).unwrap();
            assert!(topology.core_info(info.core).unwrap().cpus.contains(cpu));
            assert!(
                topology
                    .socket_info(info.socket)
                    .unwrap()
                    .cpus
                    .contains(cpu)
            );
        }
    }
}
