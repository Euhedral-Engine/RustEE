//! Small shared types (upstream `util`).
//!
//! `UnmodifiableDoubleArray` has no counterpart: a `Box<[f64]>` handed out as `&[f64]` is already
//! immutable, so the records that carried one expose slices instead.

pub(crate) mod cpu_set;

pub use cpu_set::{CpuSet, Iter as CpuSetIter};

use std::fmt;

/// The operating system this crate was compiled for (upstream `OSName`), fixed at compile time.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OsName {
    Linux,
    MacOs,
    Windows,
    Unsupported,
}

impl OsName {
    pub const CURRENT: OsName = if cfg!(target_os = "linux") {
        OsName::Linux
    } else if cfg!(target_os = "macos") {
        OsName::MacOs
    } else if cfg!(target_os = "windows") {
        OsName::Windows
    } else {
        OsName::Unsupported
    };

    pub const fn current() -> OsName {
        Self::CURRENT
    }

    pub const fn is_linux() -> bool {
        matches!(Self::CURRENT, OsName::Linux)
    }

    pub const fn is_macos() -> bool {
        matches!(Self::CURRENT, OsName::MacOs)
    }

    pub const fn is_windows() -> bool {
        matches!(Self::CURRENT, OsName::Windows)
    }

    /// The display name upstream used ("Linux", "MacOS", "Windows", "UNSUPPORTED").
    pub const fn name(self) -> &'static str {
        match self {
            OsName::Linux => "Linux",
            OsName::MacOs => "MacOS",
            OsName::Windows => "Windows",
            OsName::Unsupported => "UNSUPPORTED",
        }
    }
}

impl fmt::Display for OsName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Whether the target is an x86 CPU (upstream `SystemInfo.isX86()`).
pub const fn is_x86() -> bool {
    cfg!(any(target_arch = "x86", target_arch = "x86_64"))
}
