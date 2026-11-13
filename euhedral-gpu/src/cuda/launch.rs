//! The driver launch of one kernel: `cuLaunchKernel`, or `cuLaunchKernelEx` for a programmatic dependent launch.

use std::ffi::{c_uint, c_void};

use euhedral_memory::cuda::CuStream;

use crate::{GpuError, Launch};

use super::fns::{CuFunction, GpuFns};

/// `CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION`.
const PROGRAMMATIC_STREAM_SERIALIZATION: c_uint = 6;

/// `CUlaunchAttribute`: the id, then the 64-byte value union (an int for this attribute).
#[repr(C)]
struct LaunchAttribute {
    id: c_uint,
    pad: [u8; 4],
    value: [u64; 8],
}

/// `CUlaunchConfig`.
#[repr(C)]
struct LaunchConfig {
    grid: [c_uint; 3],
    block: [c_uint; 3],
    shared_bytes: c_uint,
    stream: CuStream,
    attributes: *const LaunchAttribute,
    attribute_count: c_uint,
}

/// Whether the driver can make this launch: a programmatic one needs `cuLaunchKernelEx`.
pub(crate) fn supported(fns: &GpuFns, launch: &Launch) -> Result<(), GpuError> {
    if launch.programmatic && fns.launch_kernel_ex.is_none() {
        return Err(GpuError::Invalid(
            "a programmatic launch needs cuLaunchKernelEx, which the loaded driver does not have"
                .into(),
        ));
    }
    Ok(())
}

/// Launches `func` on `stream` and returns the driver's result code. A programmatic launch the driver cannot make
/// (see `supported`) returns `CUDA_ERROR_NOT_SUPPORTED`.
///
/// # Safety
/// The arguments of `cuLaunchKernel`: `params` match the kernel's parameters and their addresses are valid for the
/// call; the context is current.
pub(crate) unsafe fn issue(
    fns: &GpuFns,
    func: CuFunction,
    launch: &Launch,
    stream: CuStream,
    params: *mut *mut c_void,
) -> i32 {
    if !launch.programmatic {
        // SAFETY: the caller's contract.
        return unsafe {
            (fns.launch_kernel)(
                func,
                launch.grid.x,
                launch.grid.y,
                launch.grid.z,
                launch.block.x,
                launch.block.y,
                launch.block.z,
                launch.shared_bytes,
                stream,
                params,
                std::ptr::null_mut(),
            )
        };
    }
    let Some(launch_ex) = fns.launch_kernel_ex else {
        return CUDA_ERROR_NOT_SUPPORTED;
    };
    let mut attribute = LaunchAttribute {
        id: PROGRAMMATIC_STREAM_SERIALIZATION,
        pad: [0; 4],
        value: [0; 8],
    };
    attribute.value[0] = 1;
    let config = LaunchConfig {
        grid: [launch.grid.x, launch.grid.y, launch.grid.z],
        block: [launch.block.x, launch.block.y, launch.block.z],
        shared_bytes: launch.shared_bytes,
        stream,
        attributes: &attribute,
        attribute_count: 1,
    };
    // SAFETY: the config and its attribute live across the call (the driver copies them); the rest is the caller's
    // contract.
    unsafe {
        launch_ex(
            std::ptr::from_ref(&config).cast(),
            func,
            params,
            std::ptr::null_mut(),
        )
    }
}

const CUDA_ERROR_NOT_SUPPORTED: i32 = 801;
