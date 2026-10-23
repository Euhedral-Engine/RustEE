//! Routing constraints on where a frame may run.

/// How close to its [origin](crate::frames::FrameHeader::origin) a frame must stay.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum RoutingPolicy {
    /// Any CPU.
    #[default]
    Anywhere,
    /// A CPU on the origin's socket.
    SocketLocal,
    /// A CPU sharing a cache with the origin.
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
