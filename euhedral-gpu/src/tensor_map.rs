//! Tensor maps: the driver-encoded descriptor of a global tensor that a kernel hands to the Tensor Memory Accelerator
//! (compute capability 9.0 and newer), as a 128-byte kernel argument passed by value (`CUtensorMap`).
//!
//! A [`TiledMap`] describes the tensor (address, element type, extents, byte strides) and the box one bulk copy moves (box
//! extents, traversal strides, interleave, swizzle, L2 promotion, out-of-bounds fill). On CUDA
//! `cuda::TensorMap::tiled` (this type, built by the driver) encodes it with `cuTensorMapEncodeTiled`: a driver call that belongs
//! to a prepare, never to a unit of work. [`TiledMap::check`] refuses what the driver's documented limits refuse (rank, zero
//! extents, box and stride limits, alignments) before the driver is asked; it covers the general limits only, and the driver
//! still refuses the rest (32-byte address and strides for 32-byte interleave and the 16-byte-aligned packed types, the packed
//! types' extent and box rules, a box row that is not a multiple of 16 bytes or wider than the swizzle span, interleave and
//! swizzle combinations) as an error naming the call. In a kernel the argument is
//! `const __grid_constant__ CUtensorMap map` and a copy is `cp.async.bulk.tensor` with its address. The map does not own the
//! memory: whoever does keeps it allocated until the last kernel that reads it retired, and a value is a copy of bytes, so it
//! needs no drop and no driver call to pass again.

use crate::GpuError;
use crate::args::KernelArg;

/// The most dimensions a tensor map describes (`cuTensorMapEncodeTiled`: "the maximum supported dimensionality of 5").
pub const MAX_RANK: usize = 5;

/// `CU_TENSOR_MAP_NUM_QWORDS`.
const WORDS: usize = 16;

/// The element type of the tensor (`CUtensorMapDataType`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MapDataType {
    U8,
    U16,
    U32,
    I32,
    U64,
    I64,
    F16,
    F32,
    F64,
    Bf16,
    F32Ftz,
    Tf32,
    Tf32Ftz,
    /// 16 packed 4-bit values, copied to memory aligned to 8 bytes.
    U4x16Align8,
    /// 16 packed 4-bit values, copied to memory aligned to 16 bytes.
    U4x16Align16,
    /// 16 packed 6-bit values, copied to memory aligned to 16 bytes.
    U6x16Align16,
}

impl MapDataType {
    /// `CUtensorMapDataType`'s value.
    pub(crate) fn code(self) -> u32 {
        match self {
            MapDataType::U8 => 0,
            MapDataType::U16 => 1,
            MapDataType::U32 => 2,
            MapDataType::I32 => 3,
            MapDataType::U64 => 4,
            MapDataType::I64 => 5,
            MapDataType::F16 => 6,
            MapDataType::F32 => 7,
            MapDataType::F64 => 8,
            MapDataType::Bf16 => 9,
            MapDataType::F32Ftz => 10,
            MapDataType::Tf32 => 11,
            MapDataType::Tf32Ftz => 12,
            MapDataType::U4x16Align8 => 13,
            MapDataType::U4x16Align16 => 14,
            MapDataType::U6x16Align16 => 15,
        }
    }

    fn is_float(self) -> bool {
        matches!(
            self,
            MapDataType::F16
                | MapDataType::F32
                | MapDataType::F64
                | MapDataType::Bf16
                | MapDataType::F32Ftz
                | MapDataType::Tf32
                | MapDataType::Tf32Ftz
        )
    }

    fn is_packed(self) -> bool {
        matches!(
            self,
            MapDataType::U4x16Align8 | MapDataType::U4x16Align16 | MapDataType::U6x16Align16
        )
    }
}

/// The interleaved layout of the tensor (`CUtensorMapInterleave`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Interleave {
    #[default]
    None,
    /// 16-byte channel groups (`NC/8HWC8` of 2-byte channels).
    B16,
    /// 32-byte channel groups.
    B32,
}

impl Interleave {
    pub(crate) fn code(self) -> u32 {
        match self {
            Interleave::None => 0,
            Interleave::B16 => 1,
            Interleave::B32 => 2,
        }
    }
}

/// The shared-memory bank swizzling of a copied box (`CUtensorMapSwizzle`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Swizzle {
    #[default]
    None,
    /// 16-byte chunks within 32 bytes.
    B32,
    /// 16-byte chunks within 64 bytes.
    B64,
    /// 16-byte chunks within 128 bytes.
    B128,
    /// 32-byte chunks within 128 bytes.
    B128Atom32,
    /// 32-byte chunks within 128 bytes, the lower 8 bytes swapped with the upper 8 of every 16 on alternate rows.
    B128Atom32Flip8,
    /// 64-byte chunks within 128 bytes.
    B128Atom64,
}

impl Swizzle {
    pub(crate) fn code(self) -> u32 {
        match self {
            Swizzle::None => 0,
            Swizzle::B32 => 1,
            Swizzle::B64 => 2,
            Swizzle::B128 => 3,
            Swizzle::B128Atom32 => 4,
            Swizzle::B128Atom32Flip8 => 5,
            Swizzle::B128Atom64 => 6,
        }
    }
}

/// How much the L2 cache fetches around a request (`CUtensorMapL2promotion`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum L2Promotion {
    #[default]
    None,
    B64,
    B128,
    B256,
}

impl L2Promotion {
    pub(crate) fn code(self) -> u32 {
        match self {
            L2Promotion::None => 0,
            L2Promotion::B64 => 1,
            L2Promotion::B128 => 2,
            L2Promotion::B256 => 3,
        }
    }
}

/// What an element outside the tensor reads (`CUtensorMapFloatOOBfill`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum OobFill {
    /// Zeros.
    #[default]
    Zero,
    /// A NaN that a fused multiply-add treats as zero. Only for floating-point element types.
    NanRequestZeroFma,
}

impl OobFill {
    pub(crate) fn code(self) -> u32 {
        match self {
            OobFill::Zero => 0,
            OobFill::NanRequestZeroFma => 1,
        }
    }
}

/// The description of a tiled tensor map. Arrays are read up to `rank` (`strides` up to `rank - 1`); the rest is ignored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TiledMap {
    /// The device address of the tensor's first element; on 16 bytes.
    pub address: u64,
    pub data_type: MapDataType,
    /// The dimensions, 1 to [`MAX_RANK`].
    pub rank: u32,
    /// The elements along each dimension, the innermost (contiguous) first; each 1 to 2^32.
    pub dims: [u64; MAX_RANK],
    /// The bytes between consecutive elements of each dimension but the innermost; each a multiple of 16, under 2^40.
    pub strides: [u64; MAX_RANK - 1],
    /// The elements one copy moves along each dimension; each 1 to 256.
    pub box_dims: [u32; MAX_RANK],
    /// The traversal step along each dimension; each 1 to 8 (the first is ignored without interleave).
    pub element_strides: [u32; MAX_RANK],
    pub interleave: Interleave,
    pub swizzle: Swizzle,
    pub l2_promotion: L2Promotion,
    pub oob_fill: OobFill,
}

impl TiledMap {
    /// Refuses the general limits of `cuTensorMapEncodeTiled`, as `GpuError::Invalid` naming the field, before the driver is
    /// asked. It is not the driver's whole list: the driver still refuses, as a call error, what depends on the element type
    /// or the layout (a 32-byte address and strides for 32-byte interleave and the 16-byte-aligned packed types; the packed
    /// types' `dims[0]` multiples and `box_dims[0] == 128`; a `box_dims[0]` times the element size that is not a multiple of
    /// 16 bytes, or wider than the swizzle span; interleave and swizzle combinations).
    pub fn check(&self) -> Result<(), GpuError> {
        let bad = |what: String| Err(GpuError::Invalid(format!("tensor map: {what}")));
        let rank = self.rank as usize;
        if rank == 0 || rank > MAX_RANK {
            return bad(format!("rank {} is not 1 to {MAX_RANK}", self.rank));
        }
        if !self.address.is_multiple_of(16) {
            return bad(format!("address {:#x} is not on 16 bytes", self.address));
        }
        if self.interleave != Interleave::None && rank < 3 {
            return bad(format!("interleave needs rank 3 or more, not {rank}"));
        }
        if self.oob_fill == OobFill::NanRequestZeroFma
            && (!self.data_type.is_float() || self.data_type.is_packed())
        {
            return bad("the NaN fill is for floating-point element types".into());
        }
        for i in 0..rank {
            if self.dims[i] == 0 || self.dims[i] > 1 << 32 {
                return bad(format!("dims[{i}] = {} is not 1 to 2^32", self.dims[i]));
            }
            if self.box_dims[i] == 0 || self.box_dims[i] > 256 {
                return bad(format!(
                    "box_dims[{i}] = {} is not 1 to 256",
                    self.box_dims[i]
                ));
            }
            if self.element_strides[i] == 0 || self.element_strides[i] > 8 {
                return bad(format!(
                    "element_strides[{i}] = {} is not 1 to 8",
                    self.element_strides[i]
                ));
            }
        }
        for i in 0..rank - 1 {
            let s = self.strides[i];
            if s == 0 || !s.is_multiple_of(16) || s >= 1 << 40 {
                return bad(format!(
                    "strides[{i}] = {s} is not a nonzero multiple of 16 under 2^40"
                ));
            }
        }
        Ok(())
    }
}

/// A tensor map: the 128 bytes of a `CUtensorMap`, an opaque value that only the driver and the device read, aligned to 64
/// bytes. Pass it by value as a kernel argument (`const __grid_constant__ CUtensorMap`), eagerly or recorded; it is plain data,
/// so copying it, keeping it in a recorded launch and updating a replay with another one are as for any value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C, align(64))]
pub struct TensorMap {
    pub(crate) opaque: [u64; WORDS],
}

// SAFETY: 128 bytes of integers with no padding (size 128 at alignment 64), the layout of `CUtensorMap`.
unsafe impl KernelArg for TensorMap {}

const _: () = assert!(size_of::<TensorMap>() == 128 && align_of::<TensorMap>() == 64);

impl TensorMap {
    /// A map of zeros, for the driver to fill.
    #[cfg(feature = "cuda")]
    pub(crate) fn zeroed() -> Self {
        Self { opaque: [0; WORDS] }
    }

    /// The bytes of the map.
    pub fn words(&self) -> &[u64; WORDS] {
        &self.opaque
    }

    /// **Fake**: a map that is a deterministic function of `description` (checked as the real one is), for tests of code built
    /// on this crate that cannot reach a driver: equal descriptions give equal maps and any differing field gives a different
    /// one. It is no descriptor a device could use.
    #[cfg(feature = "fake")]
    pub fn fake(description: &TiledMap) -> Result<Self, GpuError> {
        description.check()?;
        let mut words = [0u64; WORDS];
        let mut put = |i: usize, v: u64| {
            words[i % WORDS] = words[i % WORDS].rotate_left(7)
                ^ v.wrapping_mul(0x9E37_79B9_7F4A_7C15)
                ^ (i as u64);
        };
        put(0, description.address);
        put(1, u64::from(description.data_type.code()));
        put(2, u64::from(description.rank));
        let rank = description.rank as usize;
        for i in 0..rank {
            put(3 + i, description.dims[i]);
            put(8 + i, u64::from(description.box_dims[i]));
            put(13 + i, u64::from(description.element_strides[i]));
        }
        for i in 0..rank - 1 {
            put(18 + i, description.strides[i]);
        }
        put(
            22,
            u64::from(description.interleave.code())
                | u64::from(description.swizzle.code()) << 8
                | u64::from(description.l2_promotion.code()) << 16
                | u64::from(description.oob_fill.code()) << 24,
        );
        Ok(Self { opaque: words })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Args;

    fn tiled() -> TiledMap {
        TiledMap {
            address: 0x7000_0000,
            data_type: MapDataType::F32,
            rank: 2,
            dims: [64, 32, 1, 1, 1],
            strides: [256, 0, 0, 0],
            box_dims: [16, 8, 1, 1, 1],
            element_strides: [1; MAX_RANK],
            interleave: Interleave::None,
            swizzle: Swizzle::None,
            l2_promotion: L2Promotion::None,
            oob_fill: OobFill::Zero,
        }
    }

    fn refused(map: &TiledMap, field: &str) {
        match map.check() {
            Err(GpuError::Invalid(text)) => assert!(text.contains(field), "{text}"),
            other => panic!("expected a refusal naming {field}, got {other:?}"),
        }
    }

    #[test]
    fn a_tensor_map_is_128_bytes_on_64_and_a_kernel_argument() {
        fn is_arg<T: KernelArg>() {}
        is_arg::<TensorMap>();
        assert_eq!((size_of::<TensorMap>(), align_of::<TensorMap>()), (128, 64));
    }

    #[test]
    fn a_description_is_checked_before_the_driver_is_asked() {
        assert!(tiled().check().is_ok());
        refused(&TiledMap { rank: 0, ..tiled() }, "rank 0");
        refused(&TiledMap { rank: 6, ..tiled() }, "rank 6");
        refused(
            &TiledMap {
                address: 0x7000_0008,
                ..tiled()
            },
            "address",
        );
        refused(
            &TiledMap {
                dims: [64, 0, 1, 1, 1],
                ..tiled()
            },
            "dims[1]",
        );
        refused(
            &TiledMap {
                dims: [(1 << 32) + 1, 4, 1, 1, 1],
                ..tiled()
            },
            "dims[0]",
        );
        refused(
            &TiledMap {
                box_dims: [16, 257, 1, 1, 1],
                ..tiled()
            },
            "box_dims[1]",
        );
        refused(
            &TiledMap {
                element_strides: [1, 9, 1, 1, 1],
                ..tiled()
            },
            "element_strides[1]",
        );
        refused(
            &TiledMap {
                strides: [250, 0, 0, 0],
                ..tiled()
            },
            "strides[0]",
        );
        refused(
            &TiledMap {
                interleave: Interleave::B16,
                ..tiled()
            },
            "interleave",
        );
        refused(
            &TiledMap {
                data_type: MapDataType::U8,
                oob_fill: OobFill::NanRequestZeroFma,
                ..tiled()
            },
            "NaN",
        );
        // Entries past the rank are not read.
        assert!(
            TiledMap {
                dims: [64, 32, 0, 0, 0],
                box_dims: [16, 8, 0, 0, 0],
                ..tiled()
            }
            .check()
            .is_ok()
        );
    }

    #[test]
    fn the_documented_limits_are_accepted_at_the_edge_and_refused_past_it() {
        // Accepted: 2^32 elements, a 256 box, a traversal stride of 8, rank 5, the largest byte stride under 2^40.
        let edge = TiledMap {
            rank: 5,
            dims: [1 << 32; 5],
            strides: [(1 << 40) - 16; 4],
            box_dims: [256; 5],
            element_strides: [8; 5],
            ..tiled()
        };
        assert!(edge.check().is_ok());
        refused(&TiledMap { rank: 6, ..edge }, "rank 6");
        refused(
            &TiledMap {
                dims: [(1 << 32) + 1; 5],
                ..edge
            },
            "dims[0]",
        );
        refused(
            &TiledMap {
                box_dims: [257; 5],
                ..edge
            },
            "box_dims[0]",
        );
        refused(
            &TiledMap {
                element_strides: [9; 5],
                ..edge
            },
            "element_strides[0]",
        );
        refused(
            &TiledMap {
                strides: [1 << 40; 4],
                ..edge
            },
            "strides[0]",
        );
        refused(
            &TiledMap {
                strides: [(1 << 40) - 16, (1 << 40) - 16, (1 << 40) - 8, 16],
                ..edge
            },
            "strides[2]",
        );
        refused(
            &TiledMap {
                strides: [0; 4],
                ..edge
            },
            "strides[0]",
        );
        // Rank 3 with interleave is the smallest that is accepted.
        assert!(
            TiledMap {
                rank: 3,
                interleave: Interleave::B32,
                ..edge
            }
            .check()
            .is_ok()
        );
    }

    #[test]
    fn args_hold_a_tensor_map_on_its_own_alignment_and_give_its_bytes_back() {
        let mut words = [0u64; WORDS];
        for (i, w) in words.iter_mut().enumerate() {
            *w = 0x0102_0304_0506_0708 ^ (i as u64) << 56;
        }
        let map = TensorMap { opaque: words };
        let args = Args::new().arg(7u8).arg(map).arg(9u32);
        assert_eq!(args.len(), 3);
        assert_eq!(args.sizes().collect::<Vec<_>>(), [1, 128, 4]);
        assert_eq!(args.read::<TensorMap>(1), Some(map));
        assert_eq!(args.read::<u8>(0), Some(7));
        assert_eq!(args.read::<u32>(2), Some(9));
        // Set and assign move the bytes whole.
        let other = TensorMap {
            opaque: [0xAB; WORDS],
        };
        let mut b = args.clone();
        b.set(1, other).unwrap();
        assert_eq!(b.read::<TensorMap>(1), Some(other));
        assert_ne!(b, args);
        let mut c = Args::new();
        c.assign(&b);
        assert_eq!(c, b);
        // The layout sees the 128 bytes at 64, not a smaller or less aligned value.
        assert_ne!(
            Args::new().arg(map).layout(),
            Args::new().arg([0u64; 16]).layout()
        );
    }

    #[test]
    #[should_panic(expected = "at most 64 bytes")]
    fn an_argument_aligned_beyond_64_bytes_is_still_refused() {
        #[derive(Clone, Copy)]
        #[repr(C, align(128))]
        struct Wide([u8; 128]);
        // SAFETY: plain bytes, no padding, for the refusal under test.
        unsafe impl KernelArg for Wide {}
        let _ = Args::new().arg(Wide([0; 128]));
    }

    #[cfg(feature = "fake")]
    #[test]
    fn a_fake_map_is_a_function_of_its_description() {
        let a = TensorMap::fake(&tiled()).unwrap();
        assert_eq!(a, TensorMap::fake(&tiled()).unwrap());
        for changed in [
            TiledMap {
                address: 0x7000_0010,
                ..tiled()
            },
            TiledMap {
                data_type: MapDataType::Bf16,
                ..tiled()
            },
            TiledMap {
                dims: [64, 33, 1, 1, 1],
                ..tiled()
            },
            TiledMap {
                swizzle: Swizzle::B64,
                ..tiled()
            },
        ] {
            assert_ne!(a, TensorMap::fake(&changed).unwrap());
        }
        assert!(TensorMap::fake(&TiledMap { rank: 0, ..tiled() }).is_err());
        // Entries past the rank are not part of it.
        assert_eq!(
            a,
            TensorMap::fake(&TiledMap {
                dims: [64, 32, 7, 8, 9],
                strides: [256, 99, 98, 97],
                box_dims: [16, 8, 5, 5, 5],
                element_strides: [1, 1, 4, 4, 4],
                ..tiled()
            })
            .unwrap()
        );
    }
}
