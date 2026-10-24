//! CPU topology, thread affinity, pinned executors and resource-pressure monitoring for Euhedral.
//!
//! - [`topology`]: the machine's logical CPUs, cores, sockets and caches, and the effective topology
//!   left by a CPU allowance (upstream `SystemInfo` and `TopologyMapper`).
//! - [`affinity`]: binding the calling thread to CPUs, and executors whose threads stay on one CPU.
//! - [`monitor`]: a background sampler turning CPU, memory, I/O and power signals into per-CPU
//!   pressure.
//! - [`util`]: the sparse [`CpuSet`](util::CpuSet) the other modules share.
//!
//! The platform back end is chosen at compile time: sysfs, cgroups and `/proc` on Linux, sysctl,
//! Mach and `NSProcessInfo` on macOS, and Win32 processor groups and job objects on Windows. The
//! upstream native library is replaced by direct system calls, so there is nothing to extract or
//! load at run time. On any other platform the topology is a flat fallback, affinity is unsupported,
//! and a monitor needs a caller-supplied provider
//! ([`ResourceMonitor::with_provider`](monitor::ResourceMonitor::with_provider)).
//!
//! Values upstream marks "not available" with `null` are `Option`s here, and sparse logical CPU ids
//! are kept as the platform reports them: a span is the highest id plus one.

pub mod affinity;
pub mod monitor;
mod platform;
pub mod topology;
pub mod util;

/// Compiles and runs the examples in the crate README as doctests.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;
