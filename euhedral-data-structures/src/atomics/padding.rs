use std::ops::{Deref, DerefMut};

/// Aligns and pads `T` to 128 bytes so neighbouring values never share a cache line (or the adjacent
/// line the hardware prefetcher pairs with it).
#[derive(Debug, Default)]
#[repr(align(128))]
pub struct CachePadded<T>(pub T);

impl<T> CachePadded<T> {
    pub const fn new(value: T) -> Self {
        Self(value)
    }
}

impl<T> Deref for CachePadded<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.0
    }
}

impl<T> DerefMut for CachePadded<T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.0
    }
}
