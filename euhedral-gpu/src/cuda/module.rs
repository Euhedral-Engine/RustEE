//! CUDA modules and kernels: source through NVRTC (cached), or a prebuilt image.

use std::ffi::{CString, c_int, c_void};
use std::sync::Arc;

use euhedral_hashing::hasher_api::{hash_bytes, hash_bytes_with_seed};
use euhedral_memory::cuda::Device;

use crate::{CompileOptions, Gpu, GpuError, Kernel, KeyMaterial, Module, ModuleCache, cache_key};

use super::fns::{CuFunction, CuModule};
use super::nvrtc::{Nvrtc, Target};
use super::{CudaGpu, from_cuda};

/// `CU_JIT_ERROR_LOG_BUFFER` and `CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES`.
const JIT_ERROR_LOG_BUFFER: c_int = 5;
const JIT_ERROR_LOG_BUFFER_SIZE_BYTES: c_int = 6;
/// `CUDA_ERROR_NOT_FOUND`.
const NOT_FOUND: i32 = 500;
/// The first bytes of a cubin, and of a fatbin (little-endian `0xBA55ED50`).
const ELF_MAGIC: &[u8] = b"\x7fELF";
const FATBIN_MAGIC: &[u8] = &[0x50, 0xED, 0x55, 0xBA];

struct ModuleInner {
    gpu: Arc<CudaGpu>,
    raw: CuModule,
    /// What identifies the module in kernel ids: the cache key's first half, or the hash of the image.
    key: u64,
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

impl Drop for ModuleInner {
    /// A shutdown call: unloading may wait for the device.
    fn drop(&mut self) {
        let device: &Device = self.gpu.device();
        if let Ok(_c) = device.enter() {
            // SAFETY: the module loaded in `load_image`, unloaded once; the kernels hold an `Arc` of this.
            unsafe { (self.gpu.fns.module_unload)(self.raw) };
        }
    }
}

/// A loaded module. Loading blocks (compile, file reads, the driver's load): it is a build-time call.
#[derive(Clone, Debug)]
pub struct CudaModule {
    inner: Arc<ModuleInner>,
}

#[derive(Clone, Copy, Debug)]
#[allow(dead_code, reason = "read by the launch, which comes next")]
struct Func(CuFunction);

// SAFETY: a function of a loaded module, usable from any thread with the context current.
unsafe impl Send for Func {}
// SAFETY: as above; the handle is only read.
unsafe impl Sync for Func {}

/// A kernel of a module. It keeps its module loaded.
#[derive(Clone, Debug)]
pub struct CudaKernel {
    #[allow(dead_code, reason = "read by the launch, which comes next")]
    func: Func,
    #[allow(dead_code, reason = "keeps the module loaded while the kernel lives")]
    module: Arc<ModuleInner>,
    name: Arc<str>,
    id: u64,
    params: Option<Arc<[u32]>>,
}

impl CudaKernel {
    #[allow(dead_code, reason = "used by the launch, which comes next")]
    pub(crate) fn raw(&self) -> CuFunction {
        self.func.0
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
        let call = gpu
            .device()
            .check("cuModuleLoadDataEx", code)
            .err()
            .map_or_else(|| format!("code {code}"), |e| e.to_string());
        return Err(GpuError::Load(if log.is_empty() {
            call
        } else {
            format!("{call}: {log}")
        }));
    }
    Ok(CudaModule {
        inner: Arc::new(ModuleInner {
            gpu: gpu.clone(),
            raw,
            key,
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
        let target = nvrtc.target_for(gpu.arch());
        let (target_name, ext) = match target {
            Target::Cubin(sm) => (format!("sm_{sm}"), "cubin"),
            Target::Ptx(cc) => (format!("compute_{cc}"), "ptx"),
        };
        let key = cache_key(&KeyMaterial {
            source,
            options,
            target: &target_name,
            compiler_version: nvrtc.version(),
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
        Ok(CudaKernel {
            func: Func(func),
            module: self.inner.clone(),
            name: name.into(),
            id: hash_bytes_with_seed(name.as_bytes(), self.inner.key),
            params: parameter_sizes(gpu, func),
        })
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
