//! The crate's one error type.

use std::fmt;

use crate::shape::Fingerprint;

/// Why a call failed. Failures are values: nothing in this crate panics on the lattice.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GpuError {
    /// A backend call failed; the device is intact.
    Call {
        call: &'static str,
        code: i32,
        name: String,
    },
    /// A device fault (an illegal address, a launch failure): the device is poisoned from now on.
    Fault {
        call: &'static str,
        code: i32,
        name: String,
    },
    /// The device faulted earlier; the cause. Later submissions fail fast with it.
    Poisoned(Box<GpuError>),
    /// A resource is full (the marker ring): retry later.
    Busy,
    /// Misuse: re-entry, a bad stream index, a marker of another pool.
    Invalid(String),
    /// A kernel source that does not compile; the compiler's log.
    Compile { log: String },
    /// A library or an image that will not load.
    Load(String),
    /// A kernel symbol the module does not have.
    NotFound(String),
    /// Arguments that do not fit the kernel.
    Args(String),
    /// A recording failed; the work submitted while recording was unaffected.
    Capture(String),
    /// A sequence no longer has the shape that was recorded: record it again.
    ShapeChanged {
        recorded: Fingerprint,
        now: Fingerprint,
    },
}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GpuError::Call { call, code, name } => write!(f, "{call}: {name} ({code})"),
            GpuError::Fault { call, code, name } => {
                write!(f, "device fault in {call}: {name} ({code})")
            }
            GpuError::Poisoned(cause) => write!(f, "the device is poisoned by: {cause}"),
            GpuError::Busy => f.write_str("a resource is full; retry"),
            GpuError::Invalid(what) => write!(f, "invalid use: {what}"),
            GpuError::Compile { log } => write!(f, "compile error: {log}"),
            GpuError::Load(what) => write!(f, "load error: {what}"),
            GpuError::NotFound(what) => write!(f, "not found: {what}"),
            GpuError::Args(what) => write!(f, "arguments: {what}"),
            GpuError::Capture(what) => write!(f, "recording failed: {what}"),
            GpuError::ShapeChanged { recorded, now } => write!(
                f,
                "the sequence changed shape (recorded {:016x}{:016x}, now {:016x}{:016x})",
                recorded.0, recorded.1, now.0, now.1
            ),
        }
    }
}

impl std::error::Error for GpuError {}
