//! The CUDA driver backend (feature `cuda`): `libcuda.so.1` (Linux) or `nvcuda.dll` (Windows) loaded at run time
//! through [`loader`], so building needs no CUDA toolkit and a machine without a driver only loses this module.
//!
//! - [`Driver`]: the entry points this crate uses, and [`Device`]: a device's primary context, made current around
//!   each call.
//! - Backends ([`backends`]): [`DeviceMemory`] (`cuMemAlloc`), [`StreamPool`] (`cuMemAllocAsync` from the device's
//!   default pool, kept by a release threshold), [`Vmm`] (`cuMemCreate` + `cuMemMap`, exportable as a
//!   [`ShareableHandle`]: a POSIX fd that Vulkan imports with `VK_KHR_external_memory_fd`, or a Win32 handle for
//!   `VK_KHR_external_memory_win32`) and [`PinnedHost`] (`cuMemHostAlloc`, released buffers kept by
//!   power-of-two size class up to a limit: a pinned free synchronizes the device).
//! - [`StreamCopier`]: the [`Copier`](crate::Copier) between a device and host memory for the tiers' moves:
//!   `cuMemcpy*Async` on one stream, completion by a ring of events created once.

mod backends;
mod copier;
pub mod loader;

pub use backends::{DeviceMemory, PinnedHost, StreamPool, Vmm, VmmBlock};
pub use copier::StreamCopier;

use euhedral_data_structures::sync::{HotScope, LifecycleMutex};
use loader::Library;
use std::ffi::{CStr, c_char, c_int, c_uint, c_void};
use std::fmt;
use std::sync::{Arc, OnceLock};

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

/// `CUDA_MEMCPY2D`: a copy of `height` rows of `width_in_bytes` bytes between host and device layouts with their own
/// pitches. Arrays are not used here; their fields stay null.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Memcpy2D {
    pub src_x_in_bytes: usize,
    pub src_y: usize,
    pub src_memory_type: c_int,
    pub src_host: *const c_void,
    pub src_device: CuDevicePtr,
    pub src_array: *mut c_void,
    pub src_pitch: usize,
    pub dst_x_in_bytes: usize,
    pub dst_y: usize,
    pub dst_memory_type: c_int,
    pub dst_host: *mut c_void,
    pub dst_device: CuDevicePtr,
    pub dst_array: *mut c_void,
    pub dst_pitch: usize,
    pub width_in_bytes: usize,
    pub height: usize,
}

/// `CU_MEMORYTYPE_HOST` and `CU_MEMORYTYPE_DEVICE`.
pub const MEMORYTYPE_HOST: c_int = 1;
pub const MEMORYTYPE_DEVICE: c_int = 2;

impl Memcpy2D {
    /// A copy that reads host memory when `from_device` is false and device memory when it is true, and writes the
    /// other kind according to `to_device`.
    pub fn pitched(from_device: bool, to_device: bool, copy: &crate::PitchedCopy) -> Self {
        let kind = |device| {
            if device {
                MEMORYTYPE_DEVICE
            } else {
                MEMORYTYPE_HOST
            }
        };
        Memcpy2D {
            src_x_in_bytes: 0,
            src_y: 0,
            src_memory_type: kind(from_device),
            src_host: if from_device {
                std::ptr::null()
            } else {
                copy.src as *const c_void
            },
            src_device: if from_device { copy.src } else { 0 },
            src_array: std::ptr::null_mut(),
            src_pitch: copy.src_pitch as usize,
            dst_x_in_bytes: 0,
            dst_y: 0,
            dst_memory_type: kind(to_device),
            dst_host: if to_device {
                std::ptr::null_mut()
            } else {
                copy.dst as *mut c_void
            },
            dst_device: if to_device { copy.dst } else { 0 },
            dst_array: std::ptr::null_mut(),
            dst_pitch: copy.dst_pitch as usize,
            width_in_bytes: copy.row_bytes as usize,
            height: copy.rows as usize,
        }
    }
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
/// `CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32`: an NT handle (not a KMT handle).
pub const EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32: c_uint = 2;
/// The handle type the platform imports: an fd on Unix, an NT handle on Windows.
pub const EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE: c_uint = if cfg!(windows) {
    EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32
} else {
    EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD
};
/// `CUDA_EXTERNAL_MEMORY_DEDICATED`: the memory is a Vulkan dedicated allocation.
pub const EXTERNAL_MEMORY_DEDICATED: c_uint = 1;

pub const MEM_ALLOCATION_TYPE_PINNED: c_int = 1;
pub const MEM_HANDLE_TYPE_NONE: c_int = 0;
pub const MEM_HANDLE_TYPE_POSIX_FD: c_int = 1;
pub const MEM_HANDLE_TYPE_WIN32: c_int = 2;
/// The handle type the platform exports allocations as: a POSIX fd on Unix, an NT handle on Windows.
pub const MEM_HANDLE_TYPE_SHAREABLE: c_int = if cfg!(windows) {
    MEM_HANDLE_TYPE_WIN32
} else {
    MEM_HANDLE_TYPE_POSIX_FD
};
/// The owned form of a shareable allocation handle: an `OwnedFd` on Unix, an `OwnedHandle` on Windows. It closes
/// itself on drop. [`Vmm::export_fd`] returns one and [`Device::import_fd`] takes one; the names say "fd" on both
/// platforms.
#[cfg(unix)]
pub type ShareableHandle = std::os::fd::OwnedFd;
/// The owned form of a shareable allocation handle: an `OwnedFd` on Unix, an `OwnedHandle` on Windows. It closes
/// itself on drop. [`Vmm::export_fd`] returns one and [`Device::import_fd`] takes one; the names say "fd" on both
/// platforms.
#[cfg(windows)]
pub type ShareableHandle = std::os::windows::io::OwnedHandle;
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
            /// `lib` is the CUDA driver library, and stays loaded for the process.
            unsafe fn load(lib: &Library) -> Result<Self, String> {
                Ok(Fns {
                    $( $field: {
                        let name = CStr::from_bytes_with_nul(concat!($sym, "\0").as_bytes()).map_err(|e| e.to_string())?;
                        let p = lib.symbol(name);
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
    memcpy_2d_async: "cuMemcpy2DAsync_v2" => fn(*const Memcpy2D, CuStream);
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
    /// The driver library, never closed.
    lib: Library,
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
    /// Loads the driver library ([`loader::DRIVER_LIBRARY`]: `libcuda.so.1`, or `nvcuda.dll` on Windows) and runs
    /// `cuInit`. A machine without the library gets `Err("<library> not found")`.
    pub fn load() -> Result<Arc<Self>, String> {
        // The driver stays for the process: the library is never closed.
        let lib = Library::open(loader::DRIVER_LIBRARY)
            .map_err(|_| format!("{} not found", loader::DRIVER_LIBRARY))?;
        // SAFETY: the library is loaded and stays so.
        let fns = unsafe { Fns::load(&lib)? };
        let p = lib.symbol(c"cuGetErrorName");
        if p.is_null() {
            return Err("libcuda: no cuGetErrorName".into());
        }
        // SAFETY: the documented signature.
        let error_name = unsafe { std::mem::transmute::<*mut c_void, ErrorNameFn>(p) };
        let d = Driver {
            fns,
            error_name,
            lib,
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
        static LOADED: OnceLock<Arc<Driver>> = OnceLock::new();
        static LOADING: LifecycleMutex<()> = LifecycleMutex::new(());
        // After the first load every call is one acquire load.
        if let Some(driver) = LOADED.get() {
            return Ok(driver.clone());
        }
        // The first load is lifecycle work, also when a frame makes it.
        let _cold = HotScope::leave();
        let _loading = LOADING.lock();
        if let Some(driver) = LOADED.get() {
            return Ok(driver.clone());
        }
        let driver = Driver::load()?;
        // Under `LOADING` nobody else sets the cell, so the set succeeds.
        let _ = LOADED.set(driver.clone());
        Ok(driver)
    }

    /// An entry point beyond [`Fns`], looked up in the loaded library; null when the driver lacks it.
    pub fn symbol(&self, name: &CStr) -> *mut c_void {
        self.lib.symbol(name)
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
    /// This device's own `Arc`, so a deferred call can keep the context alive until it runs.
    me: std::sync::Weak<Device>,
    /// Where the device's blocking frees and unloads wait for a moment with nothing in flight, once attached.
    deferral: std::sync::OnceLock<crate::deferred::WeakDeferral>,
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
        Ok(Arc::new_cyclic(|me| Device {
            driver,
            ordinal: dev,
            ctx,
            me: me.clone(),
            deferral: std::sync::OnceLock::new(),
        }))
    }

    /// This device's `Arc` (`None` only while it is being dropped).
    pub fn arc(&self) -> Option<Arc<Device>> {
        self.me.upgrade()
    }

    /// Makes `deferral` the place this device's blocking calls wait (once; a second is refused and returned). From then on
    /// `DeviceMemory::free`, `PinnedHost::trim`, dropping an [`Imported`] and euhedral-gpu's module unload queue their driver
    /// call on it instead of making it where the memory is dropped, an allocation that runs out of memory asks it to make
    /// the calls queued so far and tries once more (`DeviceMemory`, `PinnedHost`, `Vmm::alloc_vmm`, `StreamPool`), and
    /// `free_blocks` of the backends reports that a free no longer waits.
    ///
    /// The device holds the deferral **weakly**: the owner keeps the `DeferredCalls` (a queued call keeps the device alive, and
    /// the device must not keep the source alive). When the owner drops it, the source makes what is queued and the device makes
    /// its calls where they are dropped again.
    pub fn attach_deferral(&self, deferral: crate::Deferral) -> Result<(), crate::Deferral> {
        self.deferral
            .set(deferral.downgrade())
            .map_err(|_| deferral)
    }

    /// The device's deferral, while its owner holds the source.
    pub fn deferral(&self) -> Option<crate::Deferral> {
        self.deferral
            .get()
            .and_then(crate::deferred::WeakDeferral::upgrade)
    }

    /// Makes `call` where the memory is dropped, or queues it on the device's deferral with the `bytes` it gives back at
    /// `place`.
    pub fn defer_or_run(
        &self,
        place: crate::Place,
        bytes: u64,
        call: impl FnOnce(&Device) + Send + 'static,
    ) {
        match (self.deferral(), self.arc()) {
            (Some(deferral), Some(device)) => {
                deferral.defer(place, bytes, move || call(&device));
            }
            _ => call(self),
        }
    }

    /// For an allocation that failed for lack of memory: makes the deferred calls queued so far and says whether any were
    /// waiting (so the allocation is worth one more try). A second failure is the caller's error.
    pub(crate) fn reclaim_deferred(&self) -> bool {
        self.deferral().is_some_and(|d| d.reclaim())
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

    /// Imports memory another API exported as an opaque handle (a Vulkan allocation) and maps `size` bytes of it.
    /// On Unix the handle is a POSIX fd and CUDA takes it on success. On Windows it is an NT handle, which CUDA
    /// does not take: it is closed when this call returns, the import keeping its own reference.
    pub fn import_fd(
        &self,
        handle: ShareableHandle,
        size: u64,
        dedicated: bool,
    ) -> Result<Imported<'_>, CudaError> {
        let f = &self.driver.fns;
        let _c = self.enter()?;
        let mut desc = ExternalMemoryHandleDesc {
            kind: EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE,
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
        #[cfg(unix)]
        let raw = {
            use std::os::fd::IntoRawFd;
            let raw = handle.into_raw_fd();
            desc.handle.fd = raw;
            raw
        };
        #[cfg(windows)]
        {
            use std::os::windows::io::AsRawHandle;
            desc.handle.win32 = [handle.as_raw_handle(), std::ptr::null_mut()];
        }
        let mut ext = std::ptr::null_mut();
        // SAFETY: a valid descriptor; on success (Unix) CUDA owns the fd, and on Windows the handle stays ours and
        // is closed when `handle` drops.
        let imported = self.check("cuImportExternalMemory", unsafe {
            (f.import_external_memory)(&mut ext, &desc)
        });
        #[cfg(unix)]
        if imported.is_err() {
            use std::os::fd::FromRawFd;
            // SAFETY: the import failed: the fd is still ours.
            drop(unsafe { ShareableHandle::from_raw_fd(raw) });
        }
        imported?;
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
            size,
            ptr,
        })
    }
}

/// Memory imported from another API, mapped at `ptr`. Dropping it unmaps (`cuMemFree`) and destroys the import:
/// both wait for the device and make every stream's enqueue wait with them. With a deferral attached to the device
/// ([`Device::attach_deferral`]) they are queued with the rest and made at a moment with nothing in flight.
#[derive(Debug)]
pub struct Imported<'a> {
    device: &'a Device,
    ext: CuExternalMemory,
    size: u64,
    pub ptr: CuDevicePtr,
}

/// A raw driver handle that moves to the thread that makes a deferred call.
struct Handle(*mut c_void);

// SAFETY: a handle of the primary context, usable from any thread with that context current.
unsafe impl Send for Handle {}

impl Handle {
    /// The handle. A method, so a closure captures the whole `Handle` (which is `Send`), not its raw pointer field.
    fn get(&self) -> *mut c_void {
        self.0
    }
}

impl Drop for Imported<'_> {
    fn drop(&mut self) {
        let (ptr, ext) = (self.ptr, Handle(self.ext));
        self.device
            .defer_or_run(crate::Place::Device, self.size, move |device| {
                let f = &device.driver.fns;
                if let Ok(_c) = device.enter() {
                    // SAFETY: the mapping and the import made in `import_fd`, unused.
                    unsafe {
                        (f.mem_free)(ptr);
                        (f.destroy_external_memory)(ext.get());
                    }
                }
            });
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        // SAFETY: the retain made in `open`.
        unsafe { (self.driver.fns.primary_ctx_release)(self.ordinal) };
    }
}
