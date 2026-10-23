//! The Linux back end: sysfs topology, cgroup and `/proc` resources, and thread affinity.
//!
//! Everything but the affinity calls is plain file parsing, so it builds and is tested on every
//! platform.

#[cfg(target_os = "linux")]
pub(crate) mod affinity;
pub(crate) mod fs;
pub(crate) mod layout;
pub(crate) mod parse;
pub(crate) mod paths;
pub(crate) mod resources;
