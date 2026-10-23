use super::{AtomicDouble, CachePadded};
use std::ops::Deref;

/// An [`AtomicDouble`] alone on its cache line (`PaddedAtomicDouble` upstream). All operations are
/// reached through `Deref`.
#[derive(Debug, Default)]
pub struct PaddedAtomicDouble(CachePadded<AtomicDouble>);

impl PaddedAtomicDouble {
    pub fn new(value: f64) -> Self {
        Self(CachePadded::new(AtomicDouble::new(value)))
    }
}

impl Deref for PaddedAtomicDouble {
    type Target = AtomicDouble;

    fn deref(&self) -> &AtomicDouble {
        &self.0
    }
}
