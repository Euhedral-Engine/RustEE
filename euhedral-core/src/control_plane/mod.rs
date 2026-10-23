//! The control plane: a [`ControlPlaneLattice`] where frames enter, routing them to a
//! [`ControlPlaneShard`] per socket, each keeping a clone of a
//! [`CloneableObject`](crate::generics::CloneableObject) on every effective core of its socket.

mod countdown;
mod lattice;
mod shard;
#[cfg(test)]
pub(crate) mod testing;

pub use lattice::{
    CacheReset, ControlPlaneLattice, LatticeError, UtilizationListener, UtilizationSource,
};
pub use shard::{ControlPlaneShard, RebalanceTimedOut};
