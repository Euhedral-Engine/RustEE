//! No-wait dataflow on the Euhedral control plane: completions, sequencing, joins, device lanes
//! and stage graphs.
//!
//! Nothing here waits. Work is *submitted* and returns at once, and what follows it starts on one
//! of two events:
//! - **the submission** (a device successor orders itself behind the predecessor's [`Marker`] on
//!   the device);
//! - **the completion**, found when a lattice worker polls [`Completions`] or `euhedral-io`'s
//!   `AsyncIo`.
//!
//! No piece owns a thread. The sources ([`Completions`], and `euhedral-io`'s `AsyncIo`) are
//! `LatticeSource`s that the lattice's own polls drive, and the rest are structures run from frame
//! callbacks.
//!
//! | Piece | Module | What it is |
//! |---|---|---|
//! | [`Completions`] | [`completions`] | completions found by polls |
//! | [`Sequencer`] | [`sequencer`] | execute in any order, conclude in offer order |
//! | [`Join`], [`Part`], [`Parts`], [`Round`] | [`join`] | re-armable run-time fan-in: the last arrival runs the continuation |
//! | [`LanePool`], [`Marker`], [`DeviceQueue`] | [`lanes`] | device lanes and markers |
//! | [`StageGraph`] | [`stage_graph`] | static stage DAGs run per quantum |
//! | [`Lake`], [`UnitGuard`] | [`lake`] | the pool of ready work of one kind |
//!
//! Every public type is also re-exported at the crate root. Types of `euhedral-core` (`Frame`,
//! `LatticeSource`, ...) are not: import them from `euhedral_core`.
//!
//! Feature: `testing` adds `testing::FakeDevice`. Asynchronous file and socket I/O (`AsyncIo`) is in
//! `euhedral-io`.

// `ThreadUsage` makes system calls and `Join` shares its part slots across threads, each with a
// `// SAFETY:` comment; nowhere else.
#![deny(unsafe_code)]
#![cfg_attr(
    test,
    allow(
        clippy::disallowed_types,
        clippy::disallowed_methods,
        reason = "test fixtures lock their records and run peer threads"
    )
)]
pub mod completions;
pub mod join;
pub mod lake;
pub mod lane_source;
pub mod lanes;
pub mod sequencer;
pub mod stage_graph;
pub(crate) mod sync;
#[cfg(feature = "testing")]
pub mod testing;

pub use completions::{Completions, Pending, Ready};
pub use join::{Join, Part, Parts, Round};
pub use lake::{Lake, LakeClosed, UnitGuard};
pub use lane_source::{
    Deliver, GroupId, LaneClosed, LaneCx, LaneInputs, LaneIntakes, LaneMetrics, LaneSource,
    LaneState, LaneSubmit, LaneWork, Outcome, Retired,
};
pub use lanes::{DeviceQueue, LaneId, LanePool, LanePoolBuilder, Marker, QueueId};
pub use sequencer::{Entry, Progress, Sequencer, Slot, Ticket};
pub use stage_graph::{
    DEFAULT_RETRY_BACKSTOP, EdgeKind, ExternalEdge, GraphError, GraphRuntime, QuantumOutcome,
    QuantumRef, RetryCounts, Stage, StageCx, StageError, StageGraph, StageGraphSpec, StageId,
    StageSpec, Submitted,
};

/// Compiles and runs the examples in the crate README as doctests.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;

/// The calling thread's CPU time and involuntary context switches so far: a callback that ran over
/// its budget but used less CPU than that, with the scheduler having taken the CPU away, was
/// preempted (the OS), not slow (its code).
///
/// Measured on Linux and Android; elsewhere [`ThreadUsage::now`] returns the default (zero, with
/// `on_cpu` unknown).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ThreadUsage {
    pub cpu_ns: u64,
    pub involuntary_switches: u64,
    /// The CPU the thread is on now (`u32::MAX` if unknown).
    pub on_cpu: u32,
}

/// `getrusage`'s calling-thread target (Linux's 1; the libc crate does not name it for Android).
#[cfg(target_os = "linux")]
const RUSAGE_THREAD: libc::c_int = libc::RUSAGE_THREAD;
#[cfg(target_os = "android")]
const RUSAGE_THREAD: libc::c_int = 1;

impl ThreadUsage {
    /// Now, for the calling thread (two cheap system calls; zero where they fail).
    #[cfg(any(target_os = "linux", target_os = "android"))]
    #[allow(unsafe_code, reason = "clock_gettime and getrusage with out pointers")]
    pub fn now() -> Self {
        let mut ts = libc::timespec {
            tv_sec: 0,
            tv_nsec: 0,
        };
        // SAFETY: an out pointer to a timespec, valid for the call.
        let cpu = unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut ts) };
        // SAFETY: an all-zero rusage is a valid value to be overwritten.
        let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
        // SAFETY: an out pointer to an rusage; RUSAGE_THREAD is the calling thread.
        let ru = unsafe { libc::getrusage(RUSAGE_THREAD, &mut usage) };
        // SAFETY: no arguments; it only reads the calling thread's CPU.
        let on_cpu = unsafe { libc::sched_getcpu() };
        Self {
            cpu_ns: if cpu == 0 {
                (ts.tv_sec as u64).saturating_mul(1_000_000_000) + ts.tv_nsec as u64
            } else {
                0
            },
            involuntary_switches: if ru == 0 { usage.ru_nivcsw as u64 } else { 0 },
            on_cpu: u32::try_from(on_cpu).unwrap_or(u32::MAX),
        }
    }

    /// Not measured on this platform: the default, with the CPU unknown.
    #[cfg(not(any(target_os = "linux", target_os = "android")))]
    pub fn now() -> Self {
        Self {
            on_cpu: u32::MAX,
            ..Self::default()
        }
    }
}
