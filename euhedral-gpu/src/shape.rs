//! The shape of a recurring sequence, and its fingerprint.

use euhedral_hashing::hasher_api::hash_bytes_with_seed;

use crate::Launch;

/// 128 bits of a sequence's shape: its operations in order, with their streams, geometry, argument layouts and
/// copy parameters, but not the argument values.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Fingerprint(pub u64, pub u64);

/// Which way a copy goes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CopyKind {
    ToDevice,
    ToHost,
    OnDevice,
}

/// Builds a [`Fingerprint`] from a sequence's operations, one call per operation in order.
#[derive(Debug, Default)]
pub struct ShapeHasher {
    bytes: Vec<u8>,
}

const TAG_LAUNCH: u8 = 1;
const TAG_COPY: u8 = 2;
const TAG_MEMSET: u8 = 3;
const TAG_ORDER: u8 = 4;

impl ShapeHasher {
    fn put(&mut self, tag: u8, fields: &[u64]) {
        self.bytes.push(tag);
        for field in fields {
            self.bytes.extend_from_slice(&field.to_le_bytes());
        }
    }

    pub fn launch(&mut self, stream: usize, kernel: u64, launch: &Launch, layout: u64) {
        self.put(
            TAG_LAUNCH,
            &[
                stream as u64,
                kernel,
                layout,
                u64::from(launch.grid.x),
                u64::from(launch.grid.y),
                u64::from(launch.grid.z),
                u64::from(launch.block.x),
                u64::from(launch.block.y),
                u64::from(launch.block.z),
                u64::from(launch.shared_bytes),
            ],
        );
    }

    pub fn copy(&mut self, stream: usize, kind: CopyKind, dst: u64, src: u64, bytes: u64) {
        self.put(TAG_COPY, &[stream as u64, kind as u64, dst, src, bytes]);
    }

    pub fn memset(&mut self, stream: usize, dst: u64, value: u8, bytes: u64) {
        self.put(TAG_MEMSET, &[stream as u64, dst, u64::from(value), bytes]);
    }

    /// Later work on `before` waits for the work submitted so far on `after`.
    pub fn order(&mut self, after: usize, before: usize) {
        self.put(TAG_ORDER, &[after as u64, before as u64]);
    }

    /// Forgets the operations and keeps the storage: a hasher reused for each update allocates only once.
    pub fn clear(&mut self) {
        self.bytes.clear();
    }

    pub fn finish(&self) -> Fingerprint {
        Fingerprint(
            hash_bytes_with_seed(&self.bytes, 0x6770_7573_6861_7065),
            hash_bytes_with_seed(&self.bytes, 0x6670_7269_6e74_0001),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Dim3, Launch};

    fn launch(x: u32) -> Launch {
        Launch {
            grid: Dim3::x(x),
            block: Dim3::x(128),
            shared_bytes: 0,
        }
    }

    fn base() -> ShapeHasher {
        let mut h = ShapeHasher::default();
        h.launch(0, 11, &launch(4), 5);
        h.order(0, 1);
        h.copy(1, CopyKind::ToHost, 0x10, 0x20, 64);
        h.memset(0, 0x30, 7, 8);
        h
    }

    #[test]
    fn fingerprints_differ_by_any_shape_field_and_not_by_nothing() {
        assert_eq!(base().finish(), base().finish());
        let variants: Vec<ShapeHasher> = vec![
            {
                let mut h = ShapeHasher::default();
                h.launch(0, 11, &launch(5), 5);
                h.order(0, 1);
                h.copy(1, CopyKind::ToHost, 0x10, 0x20, 64);
                h.memset(0, 0x30, 7, 8);
                h
            },
            {
                let mut h = ShapeHasher::default();
                h.launch(0, 11, &launch(4), 6);
                h.order(0, 1);
                h.copy(1, CopyKind::ToHost, 0x10, 0x20, 64);
                h.memset(0, 0x30, 7, 8);
                h
            },
            {
                let mut h = ShapeHasher::default();
                h.launch(0, 12, &launch(4), 5);
                h.order(0, 1);
                h.copy(1, CopyKind::ToHost, 0x10, 0x20, 64);
                h.memset(0, 0x30, 7, 8);
                h
            },
            {
                let mut h = ShapeHasher::default();
                h.launch(1, 11, &launch(4), 5);
                h.order(0, 1);
                h.copy(1, CopyKind::ToHost, 0x10, 0x20, 64);
                h.memset(0, 0x30, 7, 8);
                h
            },
            {
                let mut h = ShapeHasher::default();
                h.launch(0, 11, &launch(4), 5);
                h.order(0, 1);
                h.copy(1, CopyKind::ToHost, 0x18, 0x20, 64);
                h.memset(0, 0x30, 7, 8);
                h
            },
            {
                let mut h = base();
                h.order(1, 0);
                h
            },
        ];
        let reference = base().finish();
        for (i, v) in variants.iter().enumerate() {
            assert_ne!(v.finish(), reference, "variant {i}");
        }
    }
}
