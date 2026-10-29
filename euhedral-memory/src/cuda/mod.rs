//! The CUDA driver backend (feature `cuda`): `libcuda.so.1` loaded with `dlopen` at run time, so building needs no
//! CUDA toolkit and a machine without a driver only loses this module.
//!
//! - [`Driver`]: the entry points this crate uses, and [`Device`]: a device's primary context, made current around
//!   each call.
//! - Backends ([`backends`]): [`DeviceMemory`] (`cuMemAlloc`), [`StreamPool`] (`cuMemAllocAsync` from the device's
//!   default pool, kept by a release threshold), [`Vmm`] (`cuMemCreate` + `cuMemMap`, exportable as a POSIX fd that
//!   Vulkan imports with `VK_KHR_external_memory_fd`) and [`PinnedHost`] (`cuMemHostAlloc`, released buffers kept by
//!   power-of-two size class up to a limit: a pinned free synchronizes the device).
//! - [`StreamCopier`]: the [`Copier`](crate::Copier) between a device and host memory for the tiers' moves:
//!   `cuMemcpy*Async` on one stream, completion by a ring of events created once.

mod backends;
mod copier;

pub use backends::{DeviceMemory, PinnedHost, StreamPool, Vmm, VmmBlock};
pub use copier::StreamCopier;

use euhedral_data_structures::sync::LifecycleMutex;
use std::ffi::{CStr, c_char, c_int, c_uint, c_void};
use std::fmt;
use std::sync::Arc;

pub type CuResult = c_int;
pub type CuDevicePtr = u64;
pub type CuContext = *mut c_void;
pub type CuStream = *mut c_void;
pub type CuEvent = *mut c_void;
pub type CuMemPool = *mut c_void;
pub type CuMemHandle = u64;

/// `CUmemLocation`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct MemLocation {
    pub kind: c_int,
    pub id: c_int,
}

/// `CUmemAllocationProp` (CUDA 12/13 layout).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct AllocationProp {
    pub kind: c_int,
    pub requested_handle_types: c_int,
    pub location: MemLocation,
    pub win32_handle_meta_data: *mut c_void,
    pub compression_type: u8,
    pub gpu_direct_rdma_capable: u8,
    pub usage: u16,
    pub reserved: [u8; 4],
}

/// `CUmemAccessDesc`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct AccessDesc {
    pub location: MemLocation,
    pub flags: c_int,
}

/// The fd or Win32 handle of `CUDA_EXTERNAL_MEMORY_HANDLE_DESC`.
#[repr(C)]
#[derive(Clone, Copy)]
pub union ExternalHandle {
    pub fd: c_int,
    pub win32: [*mut c_void; 2],
}

impl fmt::Debug for ExternalHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("ExternalHandle")
    }
}

/// `CUDA_EXTERNAL_MEMORY_HANDLE_DESC`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ExternalMemoryHandleDesc {
    pub kind: c_uint,
    pub handle: ExternalHandle,
    pub size: u64,
    pub flags: c_uint,
    pub reserved: [c_uint; 16],
}

/// `CUDA_EXTERNAL_MEMORY_BUFFER_DESC`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ExternalMemoryBufferDesc {
    pub offset: u64,
    pub size: u64,
    pub flags: c_uint,
    pub reserved: [c_uint; 16],
}

pub type CuExternalMemory = *mut c_void;
pub const EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD: c_uint = 1;
/// `CUDA_EXTERNAL_MEMORY_DEDICATED`: the memory is a Vulkan dedicated allocation.
pub const EXTERNAL_MEMORY_DEDICATED: c_uint = 1;

pub const MEM_ALLOCATION_TYPE_PINNED: c_int = 1;
pub const MEM_HANDLE_TYPE_NONE: c_int = 0;
pub const MEM_HANDLE_TYPE_POSIX_FD: c_int = 1;
pub const MEM_LOCATION_TYPE_DEVICE: c_int = 1;
pub const MEM_ACCESS_FLAGS_READWRITE: c_int = 3;
pub const MEM_ALLOC_GRANULARITY_MINIMUM: c_int = 0;
pub const MEMPOOL_ATTR_RELEASE_THRESHOLD: c_int = 4;
pub const STREAM_NON_BLOCKING: c_uint = 1;

macro_rules! driver_fns {
    ($( $field:ident : $sym:literal => fn($($arg:ty),*) ; )*) => {
        /// The driver entry points this crate calls.
        #[allow(missing_debug_implementations)]
        pub struct Fns {
            $( pub $field: unsafe extern "C" fn($($arg),*) -> CuResult, )*
        }

        impl Fns {
            /// # Safety
            /// `lib` is a live `dlopen` handle of the CUDA driver library.
            unsafe fn load(lib: *mut c_void) -> Result<Self, String> {
                Ok(Fns {
                    $( $field: {
                        // SAFETY: a NUL-terminated name; the handle is live.
                        let p = unsafe { libc::dlsym(lib, concat!($sym, "\0").as_ptr() as *const c_char) };
                        if p.is_null() {
                            return Err(format!("libcuda: no {}", $sym));
                        }
                        // SAFETY: the driver exports this symbol with this C signature (cuda.h).
                        unsafe { std::mem::transmute::<*mut c_void, unsafe extern "C" fn($($arg),*) -> CuResult>(p) }
                    }, )*
                })
            }
        }
    };
}

driver_fns! {
    init: "cuInit" => fn(c_uint);
    device_get: "cuDeviceGet" => fn(*mut c_int, c_int);
    primary_ctx_retain: "cuDevicePrimaryCtxRetain" => fn(*mut CuContext, c_int);
    primary_ctx_release: "cuDevicePrimaryCtxRelease_v2" => fn(c_int);
    ctx_push: "cuCtxPushCurrent_v2" => fn(CuContext);
    ctx_pop: "cuCtxPopCurrent_v2" => fn(*mut CuContext);
    ctx_synchronize: "cuCtxSynchronize" => fn();
    mem_get_info: "cuMemGetInfo_v2" => fn(*mut usize, *mut usize);
    mem_alloc: "cuMemAlloc_v2" => fn(*mut CuDevicePtr, usize);
    mem_free: "cuMemFree_v2" => fn(CuDevicePtr);
    mem_host_alloc: "cuMemHostAlloc" => fn(*mut *mut c_void, usize, c_uint);
    mem_free_host: "cuMemFreeHost" => fn(*mut c_void);
    mem_host_register: "cuMemHostRegister_v2" => fn(*mut c_void, usize, c_uint);
    mem_host_unregister: "cuMemHostUnregister" => fn(*mut c_void);
    stream_create: "cuStreamCreate" => fn(*mut CuStream, c_uint);
    stream_destroy: "cuStreamDestroy_v2" => fn(CuStream);
    stream_synchronize: "cuStreamSynchronize" => fn(CuStream);
    mem_alloc_async: "cuMemAllocAsync" => fn(*mut CuDevicePtr, usize, CuStream);
    mem_free_async: "cuMemFreeAsync" => fn(CuDevicePtr, CuStream);
    device_get_default_mem_pool: "cuDeviceGetDefaultMemPool" => fn(*mut CuMemPool, c_int);
    mem_pool_set_attribute: "cuMemPoolSetAttribute" => fn(CuMemPool, c_int, *mut c_void);
    mem_pool_trim_to: "cuMemPoolTrimTo" => fn(CuMemPool, usize);
    mem_create: "cuMemCreate" => fn(*mut CuMemHandle, usize, *const AllocationProp, u64);
    mem_release: "cuMemRelease" => fn(CuMemHandle);
    mem_address_reserve: "cuMemAddressReserve" => fn(*mut CuDevicePtr, usize, usize, CuDevicePtr, u64);
    mem_address_free: "cuMemAddressFree" => fn(CuDevicePtr, usize);
    mem_map: "cuMemMap" => fn(CuDevicePtr, usize, usize, CuMemHandle, u64);
    mem_unmap: "cuMemUnmap" => fn(CuDevicePtr, usize);
    mem_set_access: "cuMemSetAccess" => fn(CuDevicePtr, usize, *const AccessDesc, usize);
    mem_get_allocation_granularity: "cuMemGetAllocationGranularity" => fn(*mut usize, *const AllocationProp, c_int);
    mem_export_to_shareable_handle: "cuMemExportToShareableHandle" => fn(*mut c_void, CuMemHandle, c_int, u64);
    memcpy_htod_async: "cuMemcpyHtoDAsync_v2" => fn(CuDevicePtr, *const c_void, usize, CuStream);
    memcpy_dtoh_async: "cuMemcpyDtoHAsync_v2" => fn(*mut c_void, CuDevicePtr, usize, CuStream);
    memcpy_dtod_async: "cuMemcpyDtoDAsync_v2" => fn(CuDevicePtr, CuDevicePtr, usize, CuStream);
    memset_d8_async: "cuMemsetD8Async" => fn(CuDevicePtr, u8, usize, CuStream);
    event_create: "cuEventCreate" => fn(*mut CuEvent, c_uint);
    event_record: "cuEventRecord" => fn(CuEvent, CuStream);
    event_query: "cuEventQuery" => fn(CuEvent);
    event_synchronize: "cuEventSynchronize" => fn(CuEvent);
    event_elapsed_time: "cuEventElapsedTime" => fn(*mut f32, CuEvent, CuEvent);
    event_destroy: "cuEventDestroy_v2" => fn(CuEvent);
    import_external_memory: "cuImportExternalMemory" => fn(*mut CuExternalMemory, *const ExternalMemoryHandleDesc);
    external_memory_get_mapped_buffer: "cuExternalMemoryGetMappedBuffer" => fn(*mut CuDevicePtr, CuExternalMemory, *const ExternalMemoryBufferDesc);
    destroy_external_memory: "cuDestroyExternalMemory" => fn(CuExternalMemory);
}

type ErrorNameFn = unsafe extern "C" fn(CuResult, *mut *const c_char) -> CuResult;

/// The loaded CUDA driver.
pub struct Driver {
    pub fns: Fns,
    error_name: ErrorNameFn,
    /// The `dlopen` handle, kept as an integer so `Driver` stays `Send + Sync`. The library is never closed.
    lib: usize,
}

impl fmt::Debug for Driver {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Driver")
    }
}

/// A CUDA error: the call and the driver's name for the code.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CudaError {
    pub call: &'static str,
    pub code: CuResult,
    pub name: String,
}

impl fmt::Display for CudaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {} ({})", self.call, self.name, self.code)
    }
}

impl std::error::Error for CudaError {}

impl Driver {
    /// Loads `libcuda.so.1` and runs `cuInit`.
    pub fn load() -> Result<Arc<Self>, String> {
        // SAFETY: a NUL-terminated library name; the handle is never closed (the driver stays for the process).
        let lib =
            unsafe { libc::dlopen(c"libcuda.so.1".as_ptr(), libc::RTLD_NOW | libc::RTLD_GLOBAL) };
        if lib.is_null() {
            return Err("libcuda.so.1 not found".into());
        }
        // SAFETY: the handle is live.
        let fns = unsafe { Fns::load(lib)? };
        // SAFETY: as above.
        let p = unsafe { libc::dlsym(lib, c"cuGetErrorName".as_ptr()) };
        if p.is_null() {
            return Err("libcuda: no cuGetErrorName".into());
        }
        // SAFETY: the documented signature.
        let error_name = unsafe { std::mem::transmute::<*mut c_void, ErrorNameFn>(p) };
        let d = Driver {
            fns,
            error_name,
            lib: lib as usize,
        };
        // SAFETY: flags must be 0.
        d.check("cuInit", unsafe { (d.fns.init)(0) })
            .map_err(|e| e.to_string())?;
        Ok(Arc::new(d))
    }

    /// The process's driver: loaded and initialized on first use; every call returns the same `Arc`. Code that
    /// shares the device with this crate (another crate's streams and kernels) takes the driver from here, so there
    /// is one loaded library and, through [`Device::open`], one primary context. A failed load is not remembered: a
    /// later call tries again.
    pub fn shared() -> Result<Arc<Self>, String> {
        static SHARED: LifecycleMutex<Option<Arc<Driver>>> = LifecycleMutex::new(None);
        let mut shared = SHARED.lock();
        if let Some(driver) = &*shared {
            return Ok(driver.clone());
        }
        let driver = Driver::load()?;
        *shared = Some(driver.clone());
        Ok(driver)
    }

    /// An entry point beyond [`Fns`], looked up in the loaded library; null when the driver lacks it.
    pub fn symbol(&self, name: &CStr) -> *mut c_void {
        // SAFETY: the handle is live for the process (never closed) and `name` is NUL-terminated.
        unsafe { libc::dlsym(self.lib as *mut c_void, name.as_ptr()) }
    }

    /// `Ok` for `CUDA_SUCCESS`, else the error with its name.
    pub fn check(&self, call: &'static str, code: CuResult) -> Result<(), CudaError> {
        if code == 0 {
            return Ok(());
        }
        let mut name: *const c_char = std::ptr::null();
        // SAFETY: an out pointer; the driver returns a static string.
        let name = if unsafe { (self.error_name)(code, &mut name) } == 0 && !name.is_null() {
            // SAFETY: a NUL-terminated static string from the driver.
            unsafe { CStr::from_ptr(name) }
                .to_string_lossy()
                .into_owned()
        } else {
            "unknown".into()
        };
        Err(CudaError { call, code, name })
    }
}

/// A device's primary context, retained for this value's life.
#[derive(Debug)]
pub struct Device {
    pub driver: Arc<Driver>,
    pub ordinal: c_int,
    pub ctx: CuContext,
}

// SAFETY: a primary context is process-wide and usable from any thread once pushed.
unsafe impl Send for Device {}
// SAFETY: as above; the handle is only read.
unsafe impl Sync for Device {}

/// The context is current on this thread while this lives.
#[derive(Debug)]
pub struct Current<'a> {
    device: &'a Device,
}

impl Drop for Current<'_> {
    fn drop(&mut self) {
        let mut popped = std::ptr::null_mut();
        // SAFETY: pops the context pushed by `Device::enter`.
        unsafe { (self.device.driver.fns.ctx_pop)(&mut popped) };
    }
}

impl Device {
    pub fn open(driver: Arc<Driver>, ordinal: c_int) -> Result<Arc<Self>, CudaError> {
        let mut dev = 0;
        // SAFETY: out pointer.
        driver.check("cuDeviceGet", unsafe {
            (driver.fns.device_get)(&mut dev, ordinal)
        })?;
        let mut ctx = std::ptr::null_mut();
        // SAFETY: out pointer; released in Drop.
        driver.check("cuDevicePrimaryCtxRetain", unsafe {
            (driver.fns.primary_ctx_retain)(&mut ctx, dev)
        })?;
        Ok(Arc::new(Device {
            driver,
            ordinal: dev,
            ctx,
        }))
    }

    /// Makes the context current on this thread until the guard drops.
    pub fn enter(&self) -> Result<Current<'_>, CudaError> {
        // SAFETY: a retained context.
        self.driver.check("cuCtxPushCurrent", unsafe {
            (self.driver.fns.ctx_push)(self.ctx)
        })?;
        Ok(Current { device: self })
    }

    /// (free, total) device memory: the whole device's, every process included.
    pub fn mem_info(&self) -> Result<(u64, u64), CudaError> {
        let _c = self.enter()?;
        let (mut free, mut total) = (0usize, 0usize);
        // SAFETY: out pointers; the context is current.
        self.driver.check("cuMemGetInfo", unsafe {
            (self.driver.fns.mem_get_info)(&mut free, &mut total)
        })?;
        Ok((free as u64, total as u64))
    }

    pub fn check(&self, call: &'static str, code: CuResult) -> Result<(), CudaError> {
        self.driver.check(call, code)
    }

    /// Imports memory another API exported as an opaque fd (a Vulkan allocation) and maps `size` bytes of it.
    /// CUDA takes the fd on success.
    pub fn import_fd(
        &self,
        fd: std::os::fd::OwnedFd,
        size: u64,
        dedicated: bool,
    ) -> Result<Imported<'_>, CudaError> {
        use std::os::fd::IntoRawFd;
        let f = &self.driver.fns;
        let _c = self.enter()?;
        let raw = fd.into_raw_fd();
        let desc = ExternalMemoryHandleDesc {
            kind: EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD,
            handle: ExternalHandle {
                win32: [std::ptr::null_mut(); 2],
            },
            size,
            flags: if dedicated {
                EXTERNAL_MEMORY_DEDICATED
            } else {
                0
            },
            reserved: [0; 16],
        };
        let mut desc = desc;
        desc.handle.fd = raw;
        let mut ext = std::ptr::null_mut();
        // SAFETY: a valid descriptor; on success CUDA owns the fd.
        if let Err(e) = self.check("cuImportExternalMemory", unsafe {
            (f.import_external_memory)(&mut ext, &desc)
        }) {
            // SAFETY: the import failed: the fd is still ours.
            unsafe { libc::close(raw) };
            return Err(e);
        }
        let buf = ExternalMemoryBufferDesc {
            offset: 0,
            size,
            flags: 0,
            reserved: [0; 16],
        };
        let mut ptr = 0;
        // SAFETY: the imported memory; a range inside it.
        if let Err(e) = self.check("cuExternalMemoryGetMappedBuffer", unsafe {
            (f.external_memory_get_mapped_buffer)(&mut ptr, ext, &buf)
        }) {
            // SAFETY: the import just made, unmapped.
            unsafe { (f.destroy_external_memory)(ext) };
            return Err(e);
        }
        Ok(Imported {
            device: self,
            ext,
            ptr,
        })
    }
}

/// Memory imported from another API, mapped at `ptr`. Dropping it unmaps (`cuMemFree`) and destroys the import:
/// both may synchronize the device, so drop it where blocking is allowed.
#[derive(Debug)]
pub struct Imported<'a> {
    device: &'a Device,
    ext: CuExternalMemory,
    pub ptr: CuDevicePtr,
}

impl Drop for Imported<'_> {
    fn drop(&mut self) {
        let f = &self.device.driver.fns;
        if let Ok(_c) = self.device.enter() {
            // SAFETY: the mapping and the import made in `import_fd`, unused.
            unsafe {
                (f.mem_free)(self.ptr);
                (f.destroy_external_memory)(self.ext);
            }
        }
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        // SAFETY: the retain made in `open`.
        unsafe { (self.driver.fns.primary_ctx_release)(self.ordinal) };
    }
}
