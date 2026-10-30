//! Typed kernel arguments and launch geometry.

#[cfg(feature = "cuda")]
use std::ffi::c_void;

use euhedral_hashing::hasher_api::hash_bytes_with_seed;

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

/// The most arguments one launch takes: a launch lists their addresses on the stack, so it allocates nothing.
pub const MAX_ARGS: usize = 64;

/// The arguments of one kernel launch: values packed in order, each at its own alignment.
///
/// Build with [`Args::new`] and [`arg`](Args::arg); a device address from `euhedral-memory`'s `Block::addr` goes in
/// through [`device`](Args::device).
///
/// **Reuse.** A lane keeps an `Args` for each launch it repeats and rewrites it in place: [`clear`](Args::clear) and
/// [`push`](Args::push) refill it (its storage is kept, so after the first launch nothing allocates), and
/// [`set`](Args::set) and [`set_device`](Args::set_device) change one argument of a launch whose shape stays. An
/// [`assign`](Args::assign) copies another value's arguments into this one's storage.
#[derive(Clone, Debug, Default)]
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

    /// Appends `value` in place: the storage grows only past what it held before.
    ///
    /// # Panics
    /// As [`arg`](Self::arg).
    pub fn push<T: KernelArg>(&mut self, value: T) -> &mut Self {
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
        if end.div_ceil(16) > self.storage.len() {
            self.storage.resize(end.div_ceil(16), 0);
        }
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

    /// Appends a device address in place.
    pub fn push_device(&mut self, addr: u64) -> &mut Self {
        self.push(addr)
    }

    /// Forgets the arguments and keeps the storage, ready to be filled again.
    pub fn clear(&mut self) {
        self.slots.clear();
        self.used = 0;
    }

    /// Replaces argument `index` with `value`, which must be of the type's size the slot was made with.
    ///
    /// `Err` (nothing changed) when the argument is absent or of another size.
    pub fn set<T: KernelArg>(&mut self, index: usize, value: T) -> Result<(), ArgMismatch> {
        let &(offset, size, _) = self.slots.get(index).ok_or(ArgMismatch)?;
        if usize::from(size) != size_of::<T>() {
            return Err(ArgMismatch);
        }
        // SAFETY: the slot holds `size_of::<T>()` bytes inside `storage`; the source is a live local.
        unsafe {
            std::ptr::copy_nonoverlapping(
                std::ptr::from_ref(&value).cast::<u8>(),
                self.storage.as_mut_ptr().cast::<u8>().add(offset as usize),
                size_of::<T>(),
            );
        }
        Ok(())
    }

    /// Replaces a device address argument.
    pub fn set_device(&mut self, index: usize, addr: u64) -> Result<(), ArgMismatch> {
        self.set(index, addr)
    }

    /// Makes this value equal to `other`, in this value's storage: nothing allocates once the storage has held as much.
    pub fn assign(&mut self, other: &Args) {
        self.storage.clear();
        self.storage.extend_from_slice(&other.storage);
        self.slots.clear();
        self.slots.extend_from_slice(&other.slots);
        self.used = other.used;
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

    /// A hash of each argument's size and alignment: the same for different values, different for different types. It
    /// allocates nothing (up to [`MAX_ARGS`] arguments are listed on the stack).
    pub fn layout(&self) -> u64 {
        let mut bytes = [0u8; MAX_ARGS * 4];
        let mut used = 0;
        for &(_, size, align) in self.slots.iter().take(MAX_ARGS) {
            bytes[used..used + 2].copy_from_slice(&size.to_le_bytes());
            bytes[used + 2..used + 4].copy_from_slice(&align.to_le_bytes());
            used += 4;
        }
        let tail = self.slots.len().saturating_sub(MAX_ARGS) as u64;
        hash_bytes_with_seed(&bytes[..used], tail)
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

    /// One pointer per argument, into this value's storage, written to `out`: valid while this value is neither moved nor
    /// changed. The driver reads through them and never writes. Returns how many, or `None` for more than
    /// [`MAX_ARGS`].
    #[cfg(feature = "cuda")]
    pub(crate) fn pointers_into(&self, out: &mut [*mut c_void; MAX_ARGS]) -> Option<usize> {
        if self.slots.len() > MAX_ARGS {
            return None;
        }
        let base = self.storage.as_ptr().cast::<u8>();
        for (to, &(offset, _, _)) in out.iter_mut().zip(&self.slots) {
            // SAFETY: each offset is inside `storage`.
            *to = unsafe { base.add(offset as usize) }.cast_mut().cast();
        }
        Some(self.slots.len())
    }
}

impl PartialEq for Args {
    /// Equal when the arguments are: their layout and their bytes (a reused value's stale storage past its arguments does not count).
    fn eq(&self, other: &Self) -> bool {
        if self.slots != other.slots || self.used != other.used {
            return false;
        }
        let bytes = |a: &Args| -> &[u8] {
            // SAFETY: `storage` holds at least `used` initialized bytes.
            unsafe { std::slice::from_raw_parts(a.storage.as_ptr().cast::<u8>(), a.used) }
        };
        bytes(self) == bytes(other)
    }
}

/// A [`set`](Args::set) of an argument that is not there or is of another size.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArgMismatch;

impl std::fmt::Display for ArgMismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("the argument is absent or of another size")
    }
}

impl std::error::Error for ArgMismatch {}

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

    #[cfg(feature = "cuda")]
    #[test]
    fn pointers_are_aligned_to_each_argument() {
        let a = Args::new().arg(1u8).arg(2u64).arg(3u16).arg(u128_like());
        let mut p = [std::ptr::null_mut(); MAX_ARGS];
        assert_eq!(a.pointers_into(&mut p), Some(4));
        assert_eq!(p[1] as usize % 8, 0);
        assert_eq!(p[2] as usize % 2, 0);
    }

    #[cfg(feature = "cuda")]
    fn u128_like() -> [u64; 2] {
        [7, 9]
    }

    #[test]
    fn a_reused_value_is_refilled_in_place_without_growing() {
        let mut a = Args::new();
        a.push(1u32)
            .push(2.5f32)
            .push_device(0x1000)
            .push([1u16; 3]);
        let capacity = a.storage.capacity();
        for round in 0..100u32 {
            a.clear();
            a.push(round)
                .push(2.5f32)
                .push_device(0x2000 + u64::from(round))
                .push([1u16; 3]);
            assert_eq!(a.read::<u32>(0), Some(round));
            assert_eq!(a.read::<u64>(2), Some(0x2000 + u64::from(round)));
            assert_eq!(
                a.storage.capacity(),
                capacity,
                "round {round}: the storage was kept"
            );
        }
        assert_eq!(a.len(), 4);
    }

    #[test]
    fn set_changes_one_argument_of_its_own_size_only() {
        let mut a = Args::new().arg(1u32).device(0x10);
        assert!(a.set(0, 7u32).is_ok());
        assert!(a.set_device(1, 0x99).is_ok());
        assert_eq!((a.read::<u32>(0), a.read::<u64>(1)), (Some(7), Some(0x99)));
        assert_eq!(a.set(0, 7u64), Err(ArgMismatch), "another size");
        assert_eq!(a.set(2, 7u32), Err(ArgMismatch), "absent");
        assert_eq!(a.read::<u32>(0), Some(7), "nothing changed by the refusals");
    }

    #[test]
    fn assign_copies_the_arguments_into_this_storage_and_equality_ignores_stale_bytes() {
        let source = Args::new().arg(5u32).device(0x30);
        let mut target = Args::new().arg(1u64).arg(2u64).arg(3u64).arg(4u64);
        target.assign(&source);
        assert_eq!(target, source);
        let mut stale = Args::new().arg(5u32).device(0x30).arg(9u64);
        stale.clear();
        stale.push(5u32).push_device(0x30);
        assert_eq!(stale, source, "bytes past the arguments do not count");
        let mut other = source.clone();
        other.set(0, 6u32).unwrap();
        assert_ne!(other, source);
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
