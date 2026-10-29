//! The driver entry points beyond the ones `euhedral-memory` loads, looked up in the same loaded library.

use std::ffi::{CStr, c_char, c_int, c_uint, c_void};
use std::fmt;

use euhedral_memory::cuda::{CuContext, CuEvent, CuResult, CuStream, Driver};

use crate::GpuError;

pub type CuModule = *mut c_void;
pub type CuFunction = *mut c_void;
pub type CuGraph = *mut c_void;
pub type CuGraphExec = *mut c_void;
pub type CuGraphNode = *mut c_void;

/// `CUstreamCallback`.
pub type StreamCallback = unsafe extern "C" fn(CuStream, CuResult, *mut c_void);

/// `CUDA_KERNEL_NODE_PARAMS_v2`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct KernelNodeParams {
    pub func: CuFunction,
    pub grid_x: c_uint,
    pub grid_y: c_uint,
    pub grid_z: c_uint,
    pub block_x: c_uint,
    pub block_y: c_uint,
    pub block_z: c_uint,
    pub shared_bytes: c_uint,
    pub kernel_params: *mut *mut c_void,
    pub extra: *mut *mut c_void,
    pub kern: *mut c_void,
    pub ctx: CuContext,
}

/// `CUgraphExecUpdateResultInfo`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ExecUpdateResultInfo {
    pub result: c_int,
    pub error_node: CuGraphNode,
    pub error_from_node: CuGraphNode,
}

/// `CU_STREAM_CAPTURE_MODE_RELAXED`.
pub const CAPTURE_MODE_RELAXED: c_int = 2;
/// `CU_STREAM_CAPTURE_STATUS_ACTIVE`.
pub const CAPTURE_STATUS_ACTIVE: c_int = 1;
pub const ATTR_CC_MAJOR: c_int = 75;
pub const ATTR_CC_MINOR: c_int = 76;

/// `cuStreamGetCaptureInfo`, in the newest form the driver has.
#[derive(Clone, Copy)]
pub enum CaptureInfo {
    /// `cuStreamGetCaptureInfo_v3`: status, id, graph, dependencies, edge data, count.
    V3(
        unsafe extern "C" fn(
            CuStream,
            *mut c_int,
            *mut u64,
            *mut CuGraph,
            *mut *const CuGraphNode,
            *mut *const c_void,
            *mut usize,
        ) -> CuResult,
    ),
    /// `cuStreamGetCaptureInfo_v2`: status, id, graph, dependencies, count.
    V2(
        unsafe extern "C" fn(
            CuStream,
            *mut c_int,
            *mut u64,
            *mut CuGraph,
            *mut *const CuGraphNode,
            *mut usize,
        ) -> CuResult,
    ),
}

impl fmt::Debug for CaptureInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            CaptureInfo::V3(_) => "CaptureInfo::V3",
            CaptureInfo::V2(_) => "CaptureInfo::V2",
        })
    }
}

/// The entry points of streams' waits, callbacks, modules, launches, capture and graphs.
#[derive(Clone, Copy)]
pub struct GpuFns {
    pub stream_wait_event: unsafe extern "C" fn(CuStream, CuEvent, c_uint) -> CuResult,
    /// `cuStreamCreateWithPriority`: the stream's flags and its priority.
    pub stream_create_with_priority: unsafe extern "C" fn(*mut CuStream, c_uint, c_int) -> CuResult,
    /// `cuCtxGetStreamPriorityRange`: the least and the greatest priority (the greatest is the numerically lowest).
    pub ctx_get_stream_priority_range: unsafe extern "C" fn(*mut c_int, *mut c_int) -> CuResult,
    /// `cuStreamGetPriority`.
    pub stream_get_priority: unsafe extern "C" fn(CuStream, *mut c_int) -> CuResult,
    /// `cuStreamAddCallback`: required. Its callback receives the status of the work before it, exactly once, even
    /// after a device fault, which is what retirement relies on.
    pub stream_add_callback:
        unsafe extern "C" fn(CuStream, StreamCallback, *mut c_void, c_uint) -> CuResult,
    pub module_load_data_ex: unsafe extern "C" fn(
        *mut CuModule,
        *const c_void,
        c_uint,
        *mut c_int,
        *mut *mut c_void,
    ) -> CuResult,
    pub module_get_function:
        unsafe extern "C" fn(*mut CuFunction, CuModule, *const c_char) -> CuResult,
    pub module_unload: unsafe extern "C" fn(CuModule) -> CuResult,
    pub func_get_param_count: Option<unsafe extern "C" fn(CuFunction, *mut usize) -> CuResult>,
    pub func_get_param_info:
        Option<unsafe extern "C" fn(CuFunction, usize, *mut usize, *mut usize) -> CuResult>,
    pub launch_kernel: unsafe extern "C" fn(
        CuFunction,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        CuStream,
        *mut *mut c_void,
        *mut *mut c_void,
    ) -> CuResult,
    pub device_get_attribute: unsafe extern "C" fn(*mut c_int, c_int, c_int) -> CuResult,
    pub driver_get_version: unsafe extern "C" fn(*mut c_int) -> CuResult,
    pub stream_begin_capture: unsafe extern "C" fn(CuStream, c_int) -> CuResult,
    pub stream_end_capture: unsafe extern "C" fn(CuStream, *mut CuGraph) -> CuResult,
    pub capture_info: CaptureInfo,
    pub graph_instantiate: unsafe extern "C" fn(*mut CuGraphExec, CuGraph, u64) -> CuResult,
    pub graph_launch: unsafe extern "C" fn(CuGraphExec, CuStream) -> CuResult,
    pub graph_exec_update:
        unsafe extern "C" fn(CuGraphExec, CuGraph, *mut ExecUpdateResultInfo) -> CuResult,
    pub graph_exec_kernel_node_set_params:
        unsafe extern "C" fn(CuGraphExec, CuGraphNode, *const KernelNodeParams) -> CuResult,
    pub graph_destroy: unsafe extern "C" fn(CuGraph) -> CuResult,
    pub graph_exec_destroy: unsafe extern "C" fn(CuGraphExec) -> CuResult,
}

impl fmt::Debug for GpuFns {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GpuFns")
            .field("func_get_param_info", &self.func_get_param_info.is_some())
            .finish_non_exhaustive()
    }
}

/// A required entry point, or `Load("libcuda: no <name>")`.
fn required(driver: &Driver, name: &CStr) -> Result<*mut c_void, GpuError> {
    let p = driver.symbol(name);
    if p.is_null() {
        Err(GpuError::Load(format!(
            "libcuda: no {}",
            name.to_string_lossy()
        )))
    } else {
        Ok(p)
    }
}

/// An entry point the driver may lack.
fn optional(driver: &Driver, name: &CStr) -> Option<*mut c_void> {
    let p = driver.symbol(name);
    (!p.is_null()).then_some(p)
}

/// A symbol address as a function pointer of type `F`.
///
/// # Safety
/// `F` is the `unsafe extern "C" fn` type of the C signature the driver exports under the name `p` was found by.
pub(super) unsafe fn cast<F: Copy>(p: *mut c_void) -> F {
    assert_eq!(size_of::<F>(), size_of::<*mut c_void>());
    // SAFETY: `F` is a function pointer, the same size as `p`; the caller vouches for the signature.
    unsafe { std::mem::transmute_copy::<*mut c_void, F>(&p) }
}

macro_rules! cast {
    ($p:expr) => {
        // SAFETY: the driver exports the symbol with the C signature of the field it fills (cuda.h).
        unsafe { cast($p) }
    };
}

impl GpuFns {
    /// Looks every entry point up in the library `driver` loaded.
    pub fn load(driver: &Driver) -> Result<Self, GpuError> {
        let capture_info = if let Some(p) = optional(driver, c"cuStreamGetCaptureInfo_v3") {
            CaptureInfo::V3(cast!(p))
        } else {
            CaptureInfo::V2(cast!(required(driver, c"cuStreamGetCaptureInfo_v2")?))
        };
        Ok(Self {
            stream_wait_event: cast!(required(driver, c"cuStreamWaitEvent")?),
            stream_create_with_priority: cast!(required(driver, c"cuStreamCreateWithPriority")?),
            ctx_get_stream_priority_range: cast!(required(driver, c"cuCtxGetStreamPriorityRange")?),
            stream_get_priority: cast!(required(driver, c"cuStreamGetPriority")?),
            stream_add_callback: cast!(required(driver, c"cuStreamAddCallback")?),
            module_load_data_ex: cast!(required(driver, c"cuModuleLoadDataEx")?),
            module_get_function: cast!(required(driver, c"cuModuleGetFunction")?),
            module_unload: cast!(required(driver, c"cuModuleUnload")?),
            func_get_param_count: optional(driver, c"cuFuncGetParamCount").map(|p| cast!(p)),
            func_get_param_info: optional(driver, c"cuFuncGetParamInfo").map(|p| cast!(p)),
            launch_kernel: cast!(required(driver, c"cuLaunchKernel")?),
            device_get_attribute: cast!(required(driver, c"cuDeviceGetAttribute")?),
            driver_get_version: cast!(required(driver, c"cuDriverGetVersion")?),
            stream_begin_capture: cast!(required(driver, c"cuStreamBeginCapture_v2")?),
            stream_end_capture: cast!(required(driver, c"cuStreamEndCapture")?),
            capture_info,
            graph_instantiate: cast!(required(driver, c"cuGraphInstantiateWithFlags")?),
            graph_launch: cast!(required(driver, c"cuGraphLaunch")?),
            graph_exec_update: cast!(required(driver, c"cuGraphExecUpdate_v2")?),
            graph_exec_kernel_node_set_params: cast!(required(
                driver,
                c"cuGraphExecKernelNodeSetParams_v2"
            )?),
            graph_destroy: cast!(required(driver, c"cuGraphDestroy")?),
            graph_exec_destroy: cast!(required(driver, c"cuGraphExecDestroy")?),
        })
    }
}
