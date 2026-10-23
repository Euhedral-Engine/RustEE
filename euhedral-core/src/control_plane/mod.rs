//! The control plane: a [`ControlPlaneLattice`] where frames enter, routing them to a
//! [`ControlPlaneShard`] per socket, each keeping a clone of a
//! [`CloneableObject`](crate::generics::CloneableObject) on every effective core of its socket.
//! A clone keeps the frames routed to its core in a [`ControlPlaneCache`], which also asks the
//! worker's upstreams for more.

mod cache;
mod countdown;
#[allow(dead_code)]
mod decision_tree;
mod lattice;
#[allow(dead_code)]
mod participation;
mod shard;
#[cfg(test)]
pub(crate) mod testing;

pub use cache::{CacheDirectory, ControlPlaneCache};
pub use lattice::{
    CacheReset, ControlPlaneLattice, LatticeError, UtilizationListener, UtilizationSource,
};
pub use shard::{ControlPlaneShard, RebalanceTimedOut};
