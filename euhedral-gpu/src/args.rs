//! Typed kernel arguments and launch geometry.

use std::ffi::c_void;

use euhedral_hashing::hasher_api::hash_bytes;

/// A value that can be a kernel argument.
///
/// # Safety
/// Plain data: `Copy`, no pointers into host memory the device cannot see, every byte initialized (no padding), and
/// the device's layout of the type equals Rust's (`#[repr(C)]` for structs).
pub unsafe trait KernelArg: Copy + Send + Sync + 'static {}

macro_rules! kernel_args {
    ($($t:ty),*) => { $(
        // SAFETY: a primitive number is plain data with the device's layout.
        unsafe impl KernelArg for $t {}
    )* };
}
kernel_args!(u8, u16, u32, u64, i8, i16, i32, i64, f32, f64, usize, isize);

// SAFETY: an array of plain data is plain data with the elements' layout.
unsafe impl<T: KernelArg, const N: usize> KernelArg for [T; N] {}

/// The arguments of one kernel launch: values packed in order, each at its own alignment.
///
/// Build with [`Args::new`] and [`arg`](Args::arg); a device address from `euhedral-memory`'s `Block::addr` goes in
/// through [`device`](Args::device).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Args {
    /// 16-byte aligned storage; slot offsets are byte offsets into it.
    storage: Vec<u128>,
    /// (byte offset, size, alignment) per argument.
    slots: Vec<(u32, u16, u16)>,
    /// Bytes used.
    used: usize,
}

impl Args {
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends `value`.
    ///
    /// # Panics
    /// If the type is aligned beyond 16 bytes or larger than 64 KiB: a misuse found when the arguments are built.
    #[must_use]
    pub fn arg<T: KernelArg>(mut self, value: T) -> Self {
        let (size, align) = (size_of::<T>(), align_of::<T>());
        assert!(
            align <= 16,
            "a kernel argument is aligned to at most 16 bytes"
        );
        assert!(
            size <= usize::from(u16::MAX),
            "a kernel argument is under 64 KiB"
        );
        let offset = self.used.next_multiple_of(align);
        let end = offset + size;
        self.storage.resize(end.div_ceil(16), 0);
        // SAFETY: `storage` holds at least `end` bytes; the source is a live `T` of `size` bytes and the ranges
        // cannot overlap (the source is a local).
        unsafe {
            std::ptr::copy_nonoverlapping(
                std::ptr::from_ref(&value).cast::<u8>(),
                self.storage.as_mut_ptr().cast::<u8>().add(offset),
                size,
            );
        }
        self.slots.push((offset as u32, size as u16, align as u16));
        self.used = end;
        self
    }

    /// Appends a device address (from `euhedral_memory::Block::addr`).
    #[must_use]
    pub fn device(self, addr: u64) -> Self {
        self.arg(addr)
    }

    pub fn len(&self) -> usize {
        self.slots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// The size in bytes of each argument, in order.
    pub fn sizes(&self) -> impl Iterator<Item = usize> + '_ {
        self.slots.iter().map(|&(_, size, _)| usize::from(size))
    }

    /// A hash of each argument's size and alignment: the same for different values, different for different types.
    pub fn layout(&self) -> u64 {
        let mut bytes = Vec::with_capacity(self.slots.len() * 4);
        for &(_, size, align) in &self.slots {
            bytes.extend_from_slice(&size.to_le_bytes());
            bytes.extend_from_slice(&align.to_le_bytes());
        }
        hash_bytes(&bytes)
    }

    /// Argument `index` as a `T`; `None` when it is absent or its size is not `T`'s.
    pub fn read<T: KernelArg>(&self, index: usize) -> Option<T> {
        let &(offset, size, _) = self.slots.get(index)?;
        if usize::from(size) != size_of::<T>() {
            return None;
        }
        // SAFETY: the slot holds `size_of::<T>()` initialized bytes inside `storage`; any bit pattern is a valid
        // `KernelArg` value of that size by the trait's contract, and the read is unaligned-safe.
        Some(unsafe {
            std::ptr::read_unaligned(
                self.storage
                    .as_ptr()
                    .cast::<u8>()
                    .add(offset as usize)
                    .cast::<T>(),
            )
        })
    }

    /// One pointer per argument, into this value's storage: valid while it is neither moved nor changed. The driver
    /// reads through them and never writes.
    #[cfg_attr(
        all(),
        allow(dead_code, reason = "used by the backends, which come next")
    )]
    pub(crate) fn pointers(&self) -> Vec<*mut c_void> {
        let base = self.storage.as_ptr().cast::<u8>();
        self.slots
            .iter()
            // SAFETY: each offset is inside `storage`.
            .map(|&(offset, _, _)| unsafe { base.add(offset as usize) }.cast_mut().cast())
            .collect()
    }
}

/// A grid or block extent.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Dim3 {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl Dim3 {
    pub const fn x(x: u32) -> Self {
        Self { x, y: 1, z: 1 }
    }

    pub const fn new(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }
}

/// The geometry of one launch.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Launch {
    pub grid: Dim3,
    pub block: Dim3,
    /// Dynamic shared memory per block, in bytes.
    pub shared_bytes: u32,
}

impl Launch {
    /// `ceil(n / block)` blocks (at least one) of `block` threads in x.
    pub fn linear(n: u32, block: u32) -> Self {
        Self {
            grid: Dim3::x(n.div_ceil(block.max(1)).max(1)),
            block: Dim3::x(block),
            shared_bytes: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arguments_read_back_by_index_and_type() {
        let a = Args::new()
            .arg(3u32)
            .arg(2.5f32)
            .device(0x1000)
            .arg([1u16; 3]);
        assert_eq!(a.read::<u32>(0), Some(3));
        assert_eq!(a.read::<f32>(1), Some(2.5));
        assert_eq!(a.read::<u64>(2), Some(0x1000));
        assert_eq!(a.read::<u64>(0), None);
        assert_eq!(a.read::<[u16; 3]>(3), Some([1; 3]));
        assert_eq!(a.len(), 4);
        assert_eq!(a.sizes().collect::<Vec<_>>(), [4, 4, 8, 6]);
    }

    #[test]
    fn pointers_are_aligned_to_each_argument() {
        let a = Args::new().arg(1u8).arg(2u64).arg(3u16).arg(u128_like());
        let p = a.pointers();
        assert_eq!(p.len(), 4);
        assert_eq!(p[1] as usize % 8, 0);
        assert_eq!(p[2] as usize % 2, 0);
    }

    fn u128_like() -> [u64; 2] {
        [7, 9]
    }

    #[test]
    fn layout_ignores_values_and_sees_types() {
        assert_eq!(
            Args::new().arg(1u32).layout(),
            Args::new().arg(2u32).layout()
        );
        assert_ne!(
            Args::new().arg(1u32).layout(),
            Args::new().arg(1u64).layout()
        );
    }

    #[test]
    fn linear_launch_rounds_up() {
        assert_eq!(Launch::linear(1000, 256).grid, Dim3::x(4));
        assert_eq!(Launch::linear(1024, 256).grid, Dim3::x(4));
        assert_eq!(Launch::linear(0, 256).grid, Dim3::x(1));
    }
}
