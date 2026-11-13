//! `TensorMap::tiled`: the driver's encoding of a [`TiledMap`] (`cuTensorMapEncodeTiled`, compute capability 9.0 and newer). The
//! value and its description are in the crate root's `tensor_map` module, which every backend shares.

use std::ffi::c_void;

use crate::cuda::{CudaGpu, from_cuda};
use crate::{GpuError, TensorMap, TiledMap};

/// `cuTensorMapEncodeTiled`: the map, element type, rank, address, extents, byte strides, box extents, traversal strides,
/// interleave, swizzle, L2 promotion and fill.
pub(crate) type EncodeTiled = unsafe extern "C" fn(
    *mut TensorMap,
    u32,
    u32,
    *mut c_void,
    *const u64,
    *const u64,
    *const u32,
    *const u32,
    u32,
    u32,
    u32,
    u32,
) -> i32;

impl TensorMap {
    /// **Build**: the map for `description`, encoded by the driver (a host-side computation with the context current; it launches nothing). `Err`
    /// naming the field when the description is invalid ([`TiledMap::check`]), naming `cuTensorMapEncodeTiled` when the
    /// driver has no such call (older than CUDA 12.0), or the driver's error, which names the call, when it refuses the
    /// description (for example a box wider than the swizzle span).
    pub fn tiled(gpu: &CudaGpu, description: &TiledMap) -> Result<TensorMap, GpuError> {
        description.check()?;
        let encode = gpu.fns.tensor_map_encode_tiled.ok_or_else(|| {
            GpuError::Invalid("tensor map: libcuda has no cuTensorMapEncodeTiled".into())
        })?;
        let _current = gpu.device().enter().map_err(from_cuda)?;
        let mut map = TensorMap::zeroed();
        // SAFETY: `map` is a live 128-byte value aligned to 64, which the call fills; the arrays hold at least `rank` and
        // `rank - 1` values (checked) and live for the call; the address is only recorded, never dereferenced by the host.
        let code = unsafe {
            encode(
                &mut map,
                description.data_type.code(),
                description.rank,
                description.address as *mut c_void,
                description.dims.as_ptr(),
                description.strides.as_ptr(),
                description.box_dims.as_ptr(),
                description.element_strides.as_ptr(),
                description.interleave.code(),
                description.swizzle.code(),
                description.l2_promotion.code(),
                description.oob_fill.code(),
            )
        };
        gpu.check("cuTensorMapEncodeTiled", code)?;
        Ok(map)
    }
}
