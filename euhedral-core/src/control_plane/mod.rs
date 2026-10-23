//! The control plane: a lattice of shards, one per socket, each keeping a clone of a
//! [`CloneableObject`](crate::generics::CloneableObject) on every effective core of its socket.

mod shard;

pub use shard::{ControlPlaneShard, RebalanceTimedOut};
