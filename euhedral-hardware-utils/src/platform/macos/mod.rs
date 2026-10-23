//! The macOS back end: sysctl topology, process and host resources, and thread affinity tags.
//!
//! The layout and the resource arithmetic read the system through small traits, so they build and
//! are tested on every platform; only `system` and `affinity` call into macOS.

#[cfg(target_os = "macos")]
pub(crate) mod affinity;
pub(crate) mod layout;
pub(crate) mod resources;
#[cfg(target_os = "macos")]
pub(crate) mod system;

/// Integer sysctl values by name (upstream `SysctlProvider`).
pub(crate) trait Sysctl {
    /// The value of a 4- or 8-byte integer sysctl, or `None` when it does not exist or has another
    /// size.
    fn integer(&self, name: &str) -> Option<u64>;
}

#[cfg(test)]
impl Sysctl for std::collections::HashMap<&str, u64> {
    fn integer(&self, name: &str) -> Option<u64> {
        self.get(name).copied()
    }
}
