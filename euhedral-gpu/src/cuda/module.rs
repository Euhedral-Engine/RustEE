//! CUDA modules and kernels: source through NVRTC (cached), or a prebuilt image.

use std::collections::HashMap;
use std::ffi::{CString, c_int, c_void};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use euhedral_data_structures::sync::{HotScope, LifecycleMutex};
use euhedral_hashing::hasher_api::{hash_bytes, hash_bytes_with_seed};
use euhedral_memory::cuda::Device;

use crate::api::over_shared_limit;
use crate::{
    CompileOptions, Gpu, GpuError, Kernel, KernelOptions, KeyMaterial, Launch, Module, ModuleCache,
    cache_key,
};

use super::fns::{CuFunction, CuModule};
use super::nvrtc::{Nvrtc, Target};
use super::{CudaGpu, from_cuda};

/// `CU_JIT_ERROR_LOG_BUFFER` and `CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES`.
const JIT_ERROR_LOG_BUFFER: c_int = 5;
const JIT_ERROR_LOG_BUFFER_SIZE_BYTES: c_int = 6;
/// `CUDA_ERROR_NOT_FOUND`.
const NOT_FOUND: i32 = 500;
/// `CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES`.
const MAX_DYNAMIC_SHARED_SIZE_BYTES: c_int = 8;
/// The first bytes of a cubin, and of a fatbin (little-endian `0xBA55ED50`).
const ELF_MAGIC: &[u8] = b"\x7fELF";
const FATBIN_MAGIC: &[u8] = &[0x50, 0xED, 0x55, 0xBA];

struct ModuleInner {
    gpu: Arc<CudaGpu>,
    raw: CuModule,
    /// What identifies the module in kernel ids: the cache key's first half, or the hash of the image.
    key: u64,
    /// The dynamic shared limit of each kernel looked up so far, shared by every handle of it. The driver keeps it
    /// per function, and the function is the module's.
    limits: LifecycleMutex<HashMap<String, Arc<Limit>>>,
}

// SAFETY: a module of the primary context, usable from any thread with that context current.
unsafe impl Send for ModuleInner {}
// SAFETY: as above; the handle is only read.
unsafe impl Sync for ModuleInner {}

impl std::fmt::Debug for ModuleInner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModuleInner")
            .field("key", &self.key)
            .finish_non_exhaustive()
    }
}

/// A module handle that moves to the thread that makes the deferred unload.
struct RawModule(CuModule);

// SAFETY: a module of the primary context, usable from any thread with that context current.
unsafe impl Send for RawModule {}

impl RawModule {
    /// The handle. A method, so a closure captures the whole `RawModule` (which is `Send`), not its raw pointer field.
    fn get(&self) -> CuModule {
        self.0
    }
}

impl Drop for ModuleInner {
    /// `cuModuleUnload` makes its caller wait for the device's running kernels (measured: it does not hold up another
    /// stream's marker, unlike `cuModuleLoadDataEx`). With a deferral attached to the device it is queued, so the caller of
    /// the drop, which may be a lane's worker, does not wait; without one it is made here.
    fn drop(&mut self) {
        let (gpu, raw) = (Arc::clone(&self.gpu), RawModule(self.raw));
        let device: &Device = self.gpu.device();
        device.defer_or_run(euhedral_memory::Place::Device, 0, move |device| {
            if let Ok(_c) = device.enter() {
                // SAFETY: the module loaded in `load_image`, unloaded once; the kernels hold an `Arc` of this.
                unsafe { (gpu.fns.module_unload)(raw.get()) };
            }
        });
    }
}

/// A loaded module. Loading blocks (compile, file reads, the driver's load): it is a build-time call.
#[derive(Clone, Debug)]
pub struct CudaModule {
    inner: Arc<ModuleInner>,
}

#[derive(Clone, Copy, Debug)]
struct Func(CuFunction);

// SAFETY: a function of a loaded module, usable from any thread with the context current.
unsafe impl Send for Func {}
// SAFETY: as above; the handle is only read.
unsafe impl Sync for Func {}

/// A kernel of a module. It keeps its module loaded.
#[derive(Clone, Debug)]
pub struct CudaKernel {
    func: Func,
    #[allow(dead_code, reason = "keeps the module loaded while the kernel lives")]
    module: Arc<ModuleInner>,
    name: Arc<str>,
    id: u64,
    params: Option<Arc<[u32]>>,
    limit: Arc<Limit>,
}

/// The dynamic shared memory a kernel of a module allows, shared by every handle of it.
#[derive(Debug, Default)]
struct Limit {
    bytes: AtomicU32,
    /// Whether `bytes` is the driver's limit. It is not when the driver would not say what its default is and no
    /// opt-in has been made: the launch is then left to the driver's own check.
    known: AtomicBool,
}

impl Limit {
    /// The limit, if known.
    fn get(&self) -> Option<u32> {
        self.known
            .load(Ordering::Acquire)
            .then(|| self.bytes.load(Ordering::Acquire))
    }

    fn set(&self, bytes: u32) {
        self.bytes.store(bytes, Ordering::Release);
        self.known.store(true, Ordering::Release);
    }
}

impl CudaKernel {
    pub(crate) fn raw(&self) -> CuFunction {
        self.func.0
    }

    /// The most dynamic shared memory, in bytes, a launch of this kernel may ask for: what the driver allows by
    /// default, or the largest value any lookup of the kernel in its module opted in to.
    ///
    /// `None` when the driver would not say what its default is and the kernel has not been opted in.
    pub fn max_dynamic_shared_bytes(&self) -> Option<u32> {
        self.limit.get()
    }

    /// `Invalid` if `launch` asks for more dynamic shared memory than the kernel allows.
    pub(crate) fn check_shared(&self, launch: &Launch) -> Result<(), GpuError> {
        match self.max_dynamic_shared_bytes() {
            Some(limit) if launch.shared_bytes > limit => {
                Err(over_shared_limit(&self.name, launch.shared_bytes, limit))
            }
            _ => Ok(()),
        }
    }

    /// The size in bytes of each parameter, when the driver reports them.
    pub fn parameter_sizes(&self) -> Option<&[u32]> {
        self.params.as_deref()
    }
}

impl Kernel for CudaKernel {
    fn name(&self) -> &str {
        &self.name
    }

    fn id(&self) -> u64 {
        self.id
    }
}

/// Loads `image` (cubin, fatbin, or NUL-terminated PTX text) into the context.
fn load_image(gpu: &Arc<CudaGpu>, image: &[u8], key: u64) -> Result<CudaModule, GpuError> {
    let mut text;
    let binary = image.starts_with(ELF_MAGIC) || image.starts_with(FATBIN_MAGIC);
    let image = if binary || image.last() == Some(&0) {
        image
    } else {
        text = image.to_vec();
        text.push(0);
        &text
    };
    let _c = gpu.device().enter().map_err(from_cuda)?;
    let mut log = vec![0u8; 8192];
    let mut options = [JIT_ERROR_LOG_BUFFER, JIT_ERROR_LOG_BUFFER_SIZE_BYTES];
    let mut values: [*mut c_void; 2] = [log.as_mut_ptr().cast(), log.len() as *mut c_void];
    let mut raw: CuModule = std::ptr::null_mut();
    // SAFETY: an image the driver reads during the call; option and value arrays of equal length, the log buffer
    // as long as its size says; an out pointer; the context is current.
    let code = unsafe {
        (gpu.fns.module_load_data_ex)(
            &mut raw,
            image.as_ptr().cast(),
            2,
            options.as_mut_ptr(),
            values.as_mut_ptr(),
        )
    };
    if code != 0 {
        let end = log.iter().position(|&b| b == 0).unwrap_or(log.len());
        let log = String::from_utf8_lossy(&log[..end]).trim().to_owned();
        // A device fault poisons the device (and stays a fault); anything else is a module that will not load.
        let error = gpu
            .check("cuModuleLoadDataEx", code)
            .err()
            .unwrap_or_else(|| GpuError::Load(format!("cuModuleLoadDataEx: code {code}")));
        return Err(match error {
            fault @ GpuError::Fault { .. } => fault,
            other if log.is_empty() => GpuError::Load(other.to_string()),
            other => GpuError::Load(format!("{other}: {log}")),
        });
    }
    Ok(CudaModule {
        inner: Arc::new(ModuleInner {
            gpu: gpu.clone(),
            raw,
            key,
            limits: LifecycleMutex::default(),
        }),
    })
}

impl Module for CudaModule {
    type Gpu = CudaGpu;

    fn from_source(
        gpu: &Arc<CudaGpu>,
        source: &str,
        options: &CompileOptions,
    ) -> Result<Self, GpuError> {
        let nvrtc = Nvrtc::shared(options.compiler.as_deref())?;
        let target = nvrtc.target_for(gpu.arch())?;
        let (target_name, ext) = match target {
            Target::Cubin(sm) => (format!("sm_{sm}"), "cubin"),
            Target::Ptx(cc) => (format!("compute_{cc}"), "ptx"),
        };
        let key = cache_key(&KeyMaterial {
            source,
            options,
            target: &target_name,
            compiler_version: nvrtc.version(),
            compiler_build: nvrtc.build(),
        });
        let cache = options.cache_dir.as_ref().map(ModuleCache::new);
        if let Some(cache) = &cache
            && let Some(bytes) = cache.load(key, ext)
            && let Ok(module) = load_image(gpu, &bytes, key.0)
        {
            return Ok(module);
        }
        // No entry, or one that will not load: compile, and overwrite the entry.
        let image = nvrtc.compile(source, options, target)?;
        let module = load_image(gpu, &image, key.0)?;
        if let Some(cache) = &cache {
            // A cache that cannot be written costs the next start-up a compile, not this one.
            let _ = cache.store(key, ext, &image);
        }
        Ok(module)
    }

    fn from_image(gpu: &Arc<CudaGpu>, image: &[u8]) -> Result<Self, GpuError> {
        load_image(gpu, image, hash_bytes(image))
    }

    fn kernel(&self, name: &str) -> Result<<CudaGpu as Gpu>::Kernel, GpuError> {
        self.kernel_with(name, KernelOptions::default())
    }

    fn kernel_with(
        &self,
        name: &str,
        options: KernelOptions,
    ) -> Result<<CudaGpu as Gpu>::Kernel, GpuError> {
        let gpu = &self.inner.gpu;
        let c_name = CString::new(name)
            .map_err(|_| GpuError::Invalid(format!("a NUL byte in the kernel name {name:?}")))?;
        let _c = gpu.device().enter().map_err(from_cuda)?;
        let mut func: CuFunction = std::ptr::null_mut();
        // SAFETY: an out pointer, a loaded module and a NUL-terminated name; the context is current.
        let code =
            unsafe { (gpu.fns.module_get_function)(&mut func, self.inner.raw, c_name.as_ptr()) };
        if code == NOT_FOUND {
            return Err(GpuError::NotFound(format!(
                "{name} (declare kernels extern \"C\" to look them up by their source name)"
            )));
        }
        gpu.check("cuModuleGetFunction", code)?;
        let limit = self.opt_in(gpu, func, name, options.max_dynamic_shared_bytes)?;
        Ok(CudaKernel {
            func: Func(func),
            module: self.inner.clone(),
            name: name.into(),
            id: hash_bytes_with_seed(name.as_bytes(), self.inner.key),
            params: parameter_sizes(gpu, func),
            limit,
        })
    }
}

impl CudaModule {
    /// The shared limit of kernel `name` (its function `func`), raised to `wanted` bytes when that is more. The
    /// driver's attribute is per function and the last set wins there, so a lookup only ever raises it: the limit
    /// is the largest value asked, whichever lookup came last.
    fn opt_in(
        &self,
        gpu: &CudaGpu,
        func: CuFunction,
        name: &str,
        wanted: u32,
    ) -> Result<Arc<Limit>, GpuError> {
        // A lookup is build-time work, also when a frame makes it.
        let _cold = HotScope::leave();
        let mut limits = self.inner.limits.lock();
        let limit = match limits.get(name) {
            Some(limit) => Arc::clone(limit),
            None => {
                let mut current: c_int = 0;
                // SAFETY: an out pointer and a function of a loaded module; the context is current.
                let code = unsafe {
                    (gpu.fns.func_get_attribute)(&mut current, MAX_DYNAMIC_SHARED_SIZE_BYTES, func)
                };
                let limit = Arc::new(Limit::default());
                // A driver that will not say, or says a value that is no size, leaves the limit unknown.
                if code == 0
                    && let Ok(current) = u32::try_from(current)
                {
                    limit.set(current);
                }
                limits.insert(name.to_owned(), Arc::clone(&limit));
                limit
            }
        };
        // An unknown limit is at least 0, so any opt-in is made.
        let current = limit.get().unwrap_or(0);
        if wanted > current {
            let value = c_int::try_from(wanted).map_err(|_| {
                GpuError::Invalid(format!(
                    "{name}: {wanted} bytes of dynamic shared memory is more than the driver can be asked for"
                ))
            })?;
            // SAFETY: a function of a loaded module; the context is current.
            let code =
                unsafe { (gpu.fns.func_set_attribute)(func, MAX_DYNAMIC_SHARED_SIZE_BYTES, value) };
            if let Err(error) = gpu.check("cuFuncSetAttribute", code) {
                return Err(match error {
                    fault @ GpuError::Fault { .. } => fault,
                    other => GpuError::Invalid(format!(
                        "{name}: the device refused {wanted} bytes of dynamic shared memory: {other}"
                    )),
                });
            }
            limit.set(wanted);
        }
        Ok(limit)
    }
}

/// The size of each parameter of `func`, when the driver can say.
fn parameter_sizes(gpu: &CudaGpu, func: CuFunction) -> Option<Arc<[u32]>> {
    let (count, info) = (gpu.fns.func_get_param_count?, gpu.fns.func_get_param_info?);
    let mut n = 0usize;
    // SAFETY: a function of a loaded module and an out pointer; the context is current.
    if unsafe { count(func, &mut n) } != 0 {
        return None;
    }
    let mut sizes = Vec::with_capacity(n);
    for i in 0..n {
        let (mut offset, mut size) = (0usize, 0usize);
        // SAFETY: as above, with out pointers.
        if unsafe { info(func, i, &mut offset, &mut size) } != 0 {
            return None;
        }
        sizes.push(u32::try_from(size).ok()?);
    }
    Some(sizes.into())
}
