//! How frames and demand move through a lattice.
//!
//! - [`RoutingPolicy`]: how far a frame may travel from where it was made.
//! - [`LatticeHotSource`]: a source its owner pushes frames into by hand.
//! - [`LatticeHotReceiver`]: a pusher that retries a frame until its [`PushHook`] takes it.
//! - [`LatticeEdge`]: the link every [`LatticeNode`] is built on: frames flow down through it and
//!   demand flows up.
//! - [`LatticeVertex`]: a node routing each frame to one of its downstreams, fed by sources it
//!   [ingests](LatticeVertex::ingest).
//! - [`UpstreamRegistry`]: the [`UpstreamHandle`]s feeding a lattice and the workers servicing
//!   them, each through its own [`UpstreamQueue`].

mod edge;
mod hot_receiver;
mod hot_source;
mod registry;
#[cfg(test)]
pub(crate) mod testing;
mod upstream_queue;
mod vertex;

pub use edge::{LatticeEdge, LatticeNode, Retirement};
pub use hot_receiver::{LatticeHotReceiver, PushHook, Response};
pub use hot_source::LatticeHotSource;
pub use registry::UpstreamRegistry;
pub use upstream_queue::{PullBucketDivisionMode, UpstreamHandle, UpstreamQueue, decay_contention};
pub use vertex::{
    LatticeClosed, LatticeVertex, RoutingFunction, RoutingState, UpstreamInterceptor, route_by_hash,
};

/// How close to its [origin](crate::frames::FrameHeader::origin) a frame must stay.
///
/// The policy is binding, like the ordered flag: routing places a frame by it, and work stealing
/// and a worker's direct pull from its upstreams take a frame only where its policy lets it run.
/// A frame without an origin goes anywhere.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum RoutingPolicy {
    /// Any CPU.
    #[default]
    Anywhere,
    /// A CPU on the origin's socket; if that socket is inactive, routing falls back to another
    /// socket and the frame goes anywhere.
    SocketLocal,
    /// A CPU of the origin's core (its SMT threads); if that core is inactive, routing falls back
    /// to a hashed core and the frame keeps to the origin's socket.
    CacheLocal,
}

impl RoutingPolicy {
    /// The constraint's strength: `0` for [`Anywhere`](Self::Anywhere), rising as it tightens.
    pub const fn level(self) -> u8 {
        match self {
            Self::Anywhere => 0,
            Self::SocketLocal => 1,
            Self::CacheLocal => 2,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn levels_rise_with_locality() {
        assert_eq!(RoutingPolicy::Anywhere.level(), 0);
        assert_eq!(RoutingPolicy::SocketLocal.level(), 1);
        assert_eq!(RoutingPolicy::CacheLocal.level(), 2);
        assert!(RoutingPolicy::Anywhere < RoutingPolicy::SocketLocal);
        assert!(RoutingPolicy::SocketLocal < RoutingPolicy::CacheLocal);
    }

    #[test]
    fn defaults_to_anywhere() {
        assert_eq!(RoutingPolicy::default(), RoutingPolicy::Anywhere);
    }
}
