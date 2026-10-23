//! The Windows back end: processor-group topology, job and process resources, and group affinity.
//!
//! The topology parser and the resource arithmetic build and are tested on every platform; only
//! `system` and `affinity` call into Windows.

#[cfg(windows)]
pub(crate) mod affinity;
pub(crate) mod layout;
pub(crate) mod resources;
#[cfg(windows)]
pub(crate) mod system;
