//! No-wait dataflow on the Euhedral control plane: completions, asynchronous I/O, sequencing,
//! joins, device lanes and stage graphs.
//!
//! Nothing here waits. Work is *submitted* and returns at once, and what follows it starts on one
//! of two events:
//! - **the submission** (a device successor orders itself behind the predecessor's [`Marker`] on
//!   the device);
//! - **the completion**, found when a lattice worker polls [`Completions`] or `AsyncIo`.
//!
//! No piece owns a thread. The sources ([`Completions`], `AsyncIo`) are `LatticeSource`s that the
//! lattice's own polls drive, and the rest are structures run from frame callbacks.
//!
//! | Piece | Module | Pattern it follows |
//! |---|---|---|
//! | [`Completions`] | [`completions`] | Euhedral-Inference `GpuCompletions` |
//! | `AsyncIo` | `io` (Linux and Android) | Euhedral-Inference `AsyncReads`, extended to writes, sends, receives, accepts |
//! | [`Sequencer`] | [`sequencer`] | Euhedral-Execution `FrameSequencer` / Kafka `OffsetCollector` |
//! | [`Join`] | [`join`] | Euhedral-Inference `Join` |
//! | [`LanePool`], [`Marker`], [`DeviceQueue`] | [`lanes`] | Euhedral-Inference `LanePool` |
//! | [`StageGraph`] | [`stage_graph`] | Euhedral-Inference `StageGraph` |
//!
//! Every public type is also re-exported at the crate root. Types of `euhedral-core` (`Frame`,
//! `LatticeSource`, ...) are not: import them from `euhedral_core`.
//!
//! Features: `async-io` (default) builds `AsyncIo` on Linux, Android and Windows; `testing` adds
//! `testing::FakeDevice`.

// io_uring needs unsafe in `io`, and `ThreadUsage` makes two system calls, each with a `// SAFETY:`
// comment; nowhere else.
#![deny(unsafe_code)]

pub mod completions;
#[cfg(all(
    feature = "async-io",
    any(target_os = "linux", target_os = "android", windows)
))]
pub mod io;
pub mod join;
pub mod lanes;
pub mod sequencer;
pub mod stage_graph;
#[cfg(feature = "testing")]
pub mod testing;

pub use completions::{Completions, Pending, Ready};
#[cfg(all(
    feature = "async-io",
    any(target_os = "linux", target_os = "android", windows)
))]
pub use io::{
    AsyncIo, AsyncIoOptions, Create, IoBuf, IoDone, IoThen, OpenSpec, OwnedFile, OwnedSock,
    RawFile, RawSock,
};
pub use join::Join;
pub use lanes::{DeviceQueue, LaneId, LanePool, Marker};
pub use sequencer::{Sequencer, Slot};
pub use stage_graph::{
    DEFAULT_RETRY_BACKSTOP, EdgeKind, ExternalEdge, GraphError, GraphRuntime, QuantumOutcome,
    QuantumRef, RetryCounts, Stage, StageCx, StageError, StageGraph, StageGraphSpec, StageId,
    StageSpec, Submitted,
};

use euhedral_core::frames::Frame;

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

/// What turns a completed operation's result into the frame that continues the work.
pub type Continuation<T> = Box<dyn FnOnce(T) -> Box<dyn Frame> + Send>;
