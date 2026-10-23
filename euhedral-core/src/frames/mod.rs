//! The unit of work: [`Frame`], its [`FrameHeader`], and the stock frames.
//!
//! A frame carries routing hashes, a liveness flag and lifecycle hooks. Frames are reusable: once a
//! frame finishes it goes back to the pool it came from (see [`recycle`](Frame::recycle)), is given
//! new input, and is dispatched again, so steady-state execution allocates nothing.
//!
//! # Hashing and routing
//!
//! A frame is routed by its [routing hash](FrameHeader::routing_hash). Frames sharing a routing hash
//! keep their relative order; frames with different routing hashes may spread across parallel paths.
//! A new frame's routing hash is its id hash, which makes it *ordered*. For stable ordering, give
//! related frames one id hash:
//!
//! ```
//! use euhedral_core::frames::FrameHeader;
//! use euhedral_hashing::hasher_api;
//!
//! let id_hash = hasher_api::mix(1234);
//! let first = FrameHeader::new(id_hash);
//! let second = FrameHeader::new(id_hash);
//! assert_eq!(first.routing_hash(), second.routing_hash());
//! ```
//!
//! For parallel distribution, vary the seed so the frames spread out:
//!
//! ```
//! # use euhedral_core::frames::FrameHeader;
//! let mut first = FrameHeader::new(7);
//! let mut second = FrameHeader::new(7);
//! first.randomize_hash(123);
//! second.randomize_hash(124);
//! assert!(!first.is_ordered());
//! assert_ne!(first.routing_hash(), second.routing_hash());
//! ```
//!
//! # Lifecycle
//!
//! An executor that receives a frame:
//!
//! 1. if the frame is not [alive](Frame::is_alive), skips it and calls [`do_finally`](Frame::do_finally);
//! 2. otherwise calls [`execute`](Frame::execute), then
//!    - [`do_finally`](Frame::do_finally) when it returns `Ok` or [`FrameError::Cancelled`], or
//!    - [`do_finally_with_error`](Frame::do_finally_with_error) with any other error.
//!
//! The finalizer takes the frame by value: it is the frame's last owner on that executor, and
//! decides where the frame goes next (a pool, a receiver, or another queue).

mod benchmark;
mod dummy;

pub use benchmark::BenchmarkFrame;
pub use dummy::DummyFrame;

use crate::flow_control::RoutingPolicy;
use euhedral_hardware_utils::topology::CpuInfo;
use euhedral_hashing::hasher_api;
use std::any::Any;
use std::error::Error;
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// An error raised while running a frame.
pub type Failure = Box<dyn Error + Send + Sync + 'static>;

/// Why [`Frame::execute`] stopped early.
#[derive(Debug)]
pub enum FrameError {
    /// The frame, or the work it belongs to, was cancelled. Executors finish a cancelled frame with
    /// [`Frame::do_finally`].
    Cancelled,
    /// The frame failed. Executors finish a failed frame with [`Frame::do_finally_with_error`].
    Failed(Failure),
}

impl FrameError {
    /// A failure wrapping `error`.
    pub fn failed(error: impl Into<Failure>) -> Self {
        Self::Failed(error.into())
    }

    pub fn is_cancelled(&self) -> bool {
        matches!(self, Self::Cancelled)
    }
}

impl fmt::Display for FrameError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cancelled => f.write_str("frame cancelled"),
            Self::Failed(error) => write!(f, "frame failed: {error}"),
        }
    }
}

impl Error for FrameError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Cancelled => None,
            Self::Failed(error) => Some(&**error),
        }
    }
}

/// The state every frame carries: identity and routing hashes, origin, routing policy and kill
/// switch.
#[derive(Debug)]
pub struct FrameHeader {
    id_hash: u64,
    routing_hash: u64,
    origin: Option<CpuInfo>,
    routing_policy: RoutingPolicy,
    kill_switch: Option<Arc<AtomicBool>>,
}

impl FrameHeader {
    /// An ordered header without a kill switch: the frame is always alive.
    pub const fn new(id_hash: u64) -> Self {
        Self {
            id_hash,
            routing_hash: id_hash,
            origin: None,
            routing_policy: RoutingPolicy::Anywhere,
            kill_switch: None,
        }
    }

    /// An ordered header whose frame dies once `kill_switch` is set.
    pub fn with_kill_switch(id_hash: u64, kill_switch: Option<Arc<AtomicBool>>) -> Self {
        Self {
            kill_switch,
            ..Self::new(id_hash)
        }
    }

    pub fn id_hash(&self) -> u64 {
        self.id_hash
    }

    pub fn routing_hash(&self) -> u64 {
        self.routing_hash
    }

    /// Whether the frame routes by its id hash, keeping order with frames of the same id.
    pub fn is_ordered(&self) -> bool {
        self.id_hash == self.routing_hash
    }

    /// Sets the routing hash to the id hash mixed with `seed`, making the frame unordered for every
    /// seed.
    pub fn randomize_hash(&mut self, seed: u64) {
        self.routing_hash = self.id_hash ^ hasher_api::mix_with_zero_check(seed);
    }

    /// Restores the routing hash to the id hash, making the frame ordered.
    pub fn reset_hash(&mut self) {
        self.routing_hash = self.id_hash;
    }

    /// The CPU the frame was made on, if known.
    pub fn origin(&self) -> Option<CpuInfo> {
        self.origin
    }

    pub fn set_origin(&mut self, origin: Option<CpuInfo>) {
        self.origin = origin;
    }

    pub fn routing_policy(&self) -> RoutingPolicy {
        self.routing_policy
    }

    pub fn set_routing_policy(&mut self, routing_policy: RoutingPolicy) {
        self.routing_policy = routing_policy;
    }

    pub fn kill_switch(&self) -> Option<&Arc<AtomicBool>> {
        self.kill_switch.as_ref()
    }

    /// False once the kill switch is set. A frame without one is always alive.
    pub fn is_alive(&self) -> bool {
        self.kill_switch
            .as_ref()
            .is_none_or(|kill_switch| !kill_switch.load(Ordering::Acquire))
    }

    /// Sets the kill switch, stopping this frame and every frame sharing the switch. Does nothing
    /// without one.
    pub fn kill(&self) {
        if let Some(kill_switch) = &self.kill_switch {
            kill_switch.store(true, Ordering::Release);
        }
    }
}

/// A unit of work. See the [module documentation](self) for the lifecycle an executor drives.
///
/// Implementors provide the [header](Self::header) and usually [`execute`](Self::execute); a frame
/// that belongs to a pool also overrides [`recycle`](Self::recycle).
pub trait Frame: Any + Send {
    fn header(&self) -> &FrameHeader;

    fn header_mut(&mut self) -> &mut FrameHeader;

    /// Does the work. The default does nothing.
    fn execute(&mut self) -> Result<(), FrameError> {
        Ok(())
    }

    /// Liveness check: an executor may skip a frame that is not alive.
    fn is_alive(&self) -> bool {
        self.header().is_alive()
    }

    /// Hard stop for this frame and the work sharing its kill switch.
    fn kill(&self) {
        self.header().kill();
    }

    /// Runs after [`execute`](Self::execute) returned `Ok` or [`FrameError::Cancelled`], or after the
    /// executor skipped a dead frame. The default recycles the frame.
    fn do_finally(self: Box<Self>) {
        self.recycle();
    }

    /// Runs after [`execute`](Self::execute) failed. The default recycles the frame.
    fn do_finally_with_error(self: Box<Self>, error: FrameError) {
        let _ = error;
        self.recycle();
    }

    /// Returns the frame to its pool. The default, for frames without one, drops it.
    fn recycle(self: Box<Self>) {}
}

#[cfg(test)]
mod tests;
