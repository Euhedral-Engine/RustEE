//! The control plane: a [`ControlPlaneLattice`] where frames enter, routing them to a
//! [`ControlPlaneShard`] per socket, each keeping a clone of a
//! [`CloneableObject`](crate::generics::CloneableObject) on every effective core of its socket.
//! A clone keeps the frames routed to its core in a [`ControlPlaneCache`], which also asks the
//! worker's upstreams for more.

mod cache;
mod countdown;
mod decision_tree;
mod fragment;
mod lattice;
mod participation;
mod placement;
mod shard;
#[cfg(test)]
pub(crate) mod testing;

pub(crate) use cache::locality_allows;
pub use cache::{CacheDirectory, ControlPlaneCache};
pub use fragment::{ControlPlaneFragment, FragmentError};
pub use lattice::{
    CacheReset, ControlPlaneLattice, LatticeError, UtilizationListener, UtilizationSource,
};
pub use placement::PlacementPicker;
pub use shard::{ControlPlaneShard, RebalanceTimedOut};
