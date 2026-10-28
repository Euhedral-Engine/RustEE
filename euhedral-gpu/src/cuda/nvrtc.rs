//! NVRTC, the run-time compiler for kernel source, loaded with `dlopen` (no link-time dependency).
//!
//! The library is found through a search order ([`candidates`]); the first load serves the process. A machine without
//! NVRTC can still load prebuilt images.

use std::ffi::{CStr, CString, OsString, c_char, c_int, c_void};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, PoisonError};

use crate::{CompileOptions, GpuError};

use super::fns::cast;

type NvrtcResult = c_int;
type Program = *mut c_void;

/// What to compile for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Target {
    /// Machine code (cubin) for `sm_XY`; the number is `X * 10 + Y`.
    Cubin(u32),
    /// PTX for `compute_XY`, which the driver compiles forward to the device.
    Ptx(u32),
}

/// A place to try loading the library from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Candidate {
    /// A name for the dynamic loader.
    Soname(&'static str),
    File(PathBuf),
}

/// The library paths to try, in order, for `explicit` (only it, when given), then `EUHEDRAL_NVRTC`, the loader's
/// own search, and the toolkit directories (`CUDA_HOME`, `CUDA_PATH`, `/usr/local/cuda`, then
/// `/usr/local/cuda-*` newest first). A directory contributes its `libnvrtc.so.<N>` files, newest first, then
/// `libnvrtc.so`. `env` reads the environment and `list` lists a directory's entries as full paths.
pub(crate) fn candidates(
    explicit: Option<&Path>,
    env: &dyn Fn(&str) -> Option<OsString>,
    list: &dyn Fn(&Path) -> Vec<PathBuf>,
) -> Vec<Candidate> {
    if let Some(path) = explicit {
        return vec![Candidate::File(path.to_owned())];
    }
    let mut out: Vec<Candidate> = Vec::new();
    let mut push = |c: Candidate| {
        if !out.contains(&c) {
            out.push(c);
        }
    };
    if let Some(path) = env("EUHEDRAL_NVRTC") {
        push(Candidate::File(PathBuf::from(path)));
    }
    for soname in ["libnvrtc.so.13", "libnvrtc.so.12", "libnvrtc.so"] {
        push(Candidate::Soname(soname));
    }
    let mut roots: Vec<PathBuf> = Vec::new();
    for var in ["CUDA_HOME", "CUDA_PATH"] {
        if let Some(home) = env(var) {
            roots.push(PathBuf::from(home));
        }
    }
    roots.push(PathBuf::from("/usr/local/cuda"));
    let mut versioned: Vec<PathBuf> = list(Path::new("/usr/local"))
        .into_iter()
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("cuda-"))
        })
        .collect();
    versioned.sort_by(|a, b| b.cmp(a));
    roots.extend(versioned);
    for root in roots {
        let dir = root.join("lib64");
        let mut libs: Vec<(Vec<u64>, PathBuf)> = list(&dir)
            .into_iter()
            .filter_map(|p| {
                let name = p.file_name()?.to_str()?;
                let rest = name.strip_prefix("libnvrtc.so")?;
                // `libnvrtc.so` sorts last; `libnvrtc.so.12.4` by its numbers, newest first.
                let version: Vec<u64> = match rest.strip_prefix('.') {
                    Some(v) => v.split('.').filter_map(|n| n.parse().ok()).collect(),
                    None if rest.is_empty() => Vec::new(),
                    None => return None,
                };
                Some((version, p))
            })
            .collect();
        libs.sort_by(|a, b| b.0.cmp(&a.0));
        for (_, path) in libs {
            push(Candidate::File(path));
        }
    }
    out
}

struct Fns {
    version: unsafe extern "C" fn(*mut c_int, *mut c_int) -> NvrtcResult,
    get_num_supported_archs: unsafe extern "C" fn(*mut c_int) -> NvrtcResult,
    get_supported_archs: unsafe extern "C" fn(*mut c_int) -> NvrtcResult,
    create_program: unsafe extern "C" fn(
        *mut Program,
        *const c_char,
        *const c_char,
        c_int,
        *const *const c_char,
        *const *const c_char,
    ) -> NvrtcResult,
    compile_program: unsafe extern "C" fn(Program, c_int, *const *const c_char) -> NvrtcResult,
    get_program_log_size: unsafe extern "C" fn(Program, *mut usize) -> NvrtcResult,
    get_program_log: unsafe extern "C" fn(Program, *mut c_char) -> NvrtcResult,
    get_cubin_size: unsafe extern "C" fn(Program, *mut usize) -> NvrtcResult,
    get_cubin: unsafe extern "C" fn(Program, *mut c_char) -> NvrtcResult,
    get_ptx_size: unsafe extern "C" fn(Program, *mut usize) -> NvrtcResult,
    get_ptx: unsafe extern "C" fn(Program, *mut c_char) -> NvrtcResult,
    destroy_program: unsafe extern "C" fn(*mut Program) -> NvrtcResult,
    get_error_string: unsafe extern "C" fn(NvrtcResult) -> *const c_char,
}

/// The loaded compiler.
pub struct Nvrtc {
    fns: Fns,
    version: (i32, i32),
    supported: Vec<u32>,
}

impl std::fmt::Debug for Nvrtc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Nvrtc")
            .field("version", &self.version)
            .finish_non_exhaustive()
    }
}

fn dl_error() -> String {
    // SAFETY: `dlerror` returns a NUL-terminated string owned by the loader, or null.
    let p = unsafe { libc::dlerror() };
    if p.is_null() {
        "unknown error".into()
    } else {
        // SAFETY: as above; copied before the next loader call.
        unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
    }
}

fn open(name: &CStr) -> Result<*mut c_void, String> {
    // SAFETY: a NUL-terminated name; the handle is never closed (the compiler stays for the process).
    let lib = unsafe { libc::dlopen(name.as_ptr(), libc::RTLD_NOW | libc::RTLD_GLOBAL) };
    if lib.is_null() {
        Err(dl_error())
    } else {
        Ok(lib)
    }
}

fn symbol(lib: *mut c_void, name: &CStr) -> Result<*mut c_void, GpuError> {
    // SAFETY: a live handle and a NUL-terminated name.
    let p = unsafe { libc::dlsym(lib, name.as_ptr()) };
    if p.is_null() {
        Err(GpuError::Load(format!(
            "nvrtc: no {}",
            name.to_string_lossy()
        )))
    } else {
        Ok(p)
    }
}

macro_rules! sym {
    ($lib:expr, $name:literal) => {
        // SAFETY: NVRTC exports this symbol with the C signature of the field it fills (nvrtc.h).
        unsafe { cast(symbol($lib, $name)?) }
    };
}

static SHARED: Mutex<Option<Arc<Nvrtc>>> = Mutex::new(None);

impl Nvrtc {
    /// The process's NVRTC: the first successful load, by `explicit` or the search order. Loading blocks (file
    /// reads); it is a build-time call.
    pub fn shared(explicit: Option<&Path>) -> Result<Arc<Self>, GpuError> {
        let mut shared = SHARED.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(nvrtc) = &*shared {
            return Ok(nvrtc.clone());
        }
        let nvrtc = Arc::new(Self::find(explicit)?);
        *shared = Some(nvrtc.clone());
        Ok(nvrtc)
    }

    #[allow(
        clippy::disallowed_methods,
        reason = "module loading is a build-time call; listing a directory blocks by nature"
    )]
    fn find(explicit: Option<&Path>) -> Result<Self, GpuError> {
        let list = |dir: &Path| -> Vec<PathBuf> {
            std::fs::read_dir(dir)
                .map(|entries| entries.filter_map(|e| e.ok().map(|e| e.path())).collect())
                .unwrap_or_default()
        };
        let mut tried = Vec::new();
        for candidate in candidates(explicit, &|k| std::env::var_os(k), &list) {
            let (name, preload) = match &candidate {
                Candidate::Soname(s) => ((*s).to_owned(), None),
                Candidate::File(p) => (
                    p.to_string_lossy().into_owned(),
                    p.parent().map(Path::to_owned),
                ),
            };
            // NVRTC loads its builtins library by soname, and toolkit copies carry no run path: load the one beside
            // the compiler first, globally, so that load finds it.
            if let Some(dir) = preload {
                let mut builtins: Vec<PathBuf> = list(&dir)
                    .into_iter()
                    .filter(|p| {
                        p.file_name()
                            .and_then(|n| n.to_str())
                            .is_some_and(|n| n.starts_with("libnvrtc-builtins.so"))
                    })
                    .collect();
                builtins.sort();
                for path in builtins {
                    if let Ok(c) = CString::new(path.to_string_lossy().as_bytes()) {
                        let _ = open(&c);
                    }
                }
            }
            let Ok(cname) = CString::new(name.as_bytes()) else {
                continue;
            };
            match open(&cname) {
                Ok(lib) => return Self::bind(lib),
                Err(e) => tried.push(format!("{name}: {e}")),
            }
        }
        Err(GpuError::Load(format!(
            "no NVRTC found (set EUHEDRAL_NVRTC, or CUDA_HOME to a toolkit): {}",
            tried.join("; ")
        )))
    }

    fn bind(lib: *mut c_void) -> Result<Self, GpuError> {
        let fns = Fns {
            version: sym!(lib, c"nvrtcVersion"),
            get_num_supported_archs: sym!(lib, c"nvrtcGetNumSupportedArchs"),
            get_supported_archs: sym!(lib, c"nvrtcGetSupportedArchs"),
            create_program: sym!(lib, c"nvrtcCreateProgram"),
            compile_program: sym!(lib, c"nvrtcCompileProgram"),
            get_program_log_size: sym!(lib, c"nvrtcGetProgramLogSize"),
            get_program_log: sym!(lib, c"nvrtcGetProgramLog"),
            get_cubin_size: sym!(lib, c"nvrtcGetCUBINSize"),
            get_cubin: sym!(lib, c"nvrtcGetCUBIN"),
            get_ptx_size: sym!(lib, c"nvrtcGetPTXSize"),
            get_ptx: sym!(lib, c"nvrtcGetPTX"),
            destroy_program: sym!(lib, c"nvrtcDestroyProgram"),
            get_error_string: sym!(lib, c"nvrtcGetErrorString"),
        };
        let mut nvrtc = Self {
            fns,
            version: (0, 0),
            supported: Vec::new(),
        };
        let (mut major, mut minor) = (0, 0);
        // SAFETY: out pointers.
        nvrtc.check("nvrtcVersion", unsafe {
            (nvrtc.fns.version)(&mut major, &mut minor)
        })?;
        nvrtc.version = (major, minor);
        let mut count = 0;
        // SAFETY: an out pointer.
        nvrtc.check("nvrtcGetNumSupportedArchs", unsafe {
            (nvrtc.fns.get_num_supported_archs)(&mut count)
        })?;
        let mut archs = vec![0 as c_int; usize::try_from(count).unwrap_or(0)];
        // SAFETY: `archs` has room for the `count` values NVRTC reported.
        nvrtc.check("nvrtcGetSupportedArchs", unsafe {
            (nvrtc.fns.get_supported_archs)(archs.as_mut_ptr())
        })?;
        nvrtc.supported = archs
            .into_iter()
            .filter_map(|a| u32::try_from(a).ok())
            .collect();
        Ok(nvrtc)
    }

    fn check(&self, call: &'static str, code: NvrtcResult) -> Result<(), GpuError> {
        if code == 0 {
            return Ok(());
        }
        // SAFETY: NVRTC returns a static NUL-terminated string.
        let name = unsafe {
            let p = (self.fns.get_error_string)(code);
            if p.is_null() {
                "unknown".into()
            } else {
                CStr::from_ptr(p).to_string_lossy().into_owned()
            }
        };
        Err(GpuError::Call { call, code, name })
    }

    pub fn version(&self) -> (i32, i32) {
        self.version
    }

    /// `sm_XY` machine code when NVRTC supports the architecture, else PTX for the newest `compute_XY` it supports.
    pub fn target_for(&self, arch: (i32, i32)) -> Target {
        let sm = u32::try_from(arch.0 * 10 + arch.1).unwrap_or(0);
        if self.supported.contains(&sm) {
            Target::Cubin(sm)
        } else {
            Target::Ptx(self.supported.iter().copied().max().unwrap_or(sm))
        }
    }

    /// Compiles `source` for `target`: a cubin, or PTX with a trailing NUL. `options.options` are passed as given,
    /// then the architecture and each include path.
    pub fn compile(
        &self,
        source: &str,
        options: &CompileOptions,
        target: Target,
    ) -> Result<Vec<u8>, GpuError> {
        let cstr = |s: &str| {
            CString::new(s).map_err(|_| GpuError::Invalid(format!("a NUL byte in {s:?}")))
        };
        let name = if options.name.is_empty() {
            "module.cu"
        } else {
            options.name.as_str()
        };
        let (c_source, c_name) = (cstr(source)?, cstr(name)?);
        let header_names = options
            .headers
            .iter()
            .map(|(n, _)| cstr(n))
            .collect::<Result<Vec<_>, _>>()?;
        let header_bodies = options
            .headers
            .iter()
            .map(|(_, b)| cstr(b))
            .collect::<Result<Vec<_>, _>>()?;
        let mut flags: Vec<String> = options.options.clone();
        flags.push(match target {
            Target::Cubin(sm) => format!("--gpu-architecture=sm_{sm}"),
            Target::Ptx(cc) => format!("--gpu-architecture=compute_{cc}"),
        });
        for path in &options.include_paths {
            flags.push(format!("--include-path={}", path.display()));
        }
        let c_flags = flags
            .iter()
            .map(|f| cstr(f))
            .collect::<Result<Vec<_>, _>>()?;
        let flag_ptrs: Vec<*const c_char> = c_flags.iter().map(|f| f.as_ptr()).collect();
        let body_ptrs: Vec<*const c_char> = header_bodies.iter().map(|h| h.as_ptr()).collect();
        let name_ptrs: Vec<*const c_char> = header_names.iter().map(|h| h.as_ptr()).collect();

        let mut program: Program = std::ptr::null_mut();
        // SAFETY: NUL-terminated strings that outlive the call, with matching header arrays; an out pointer.
        self.check("nvrtcCreateProgram", unsafe {
            (self.fns.create_program)(
                &mut program,
                c_source.as_ptr(),
                c_name.as_ptr(),
                c_int::try_from(body_ptrs.len()).unwrap_or(0),
                body_ptrs.as_ptr(),
                name_ptrs.as_ptr(),
            )
        })?;
        let result = self.compile_created(program, name, &flag_ptrs, target);
        // SAFETY: the program made above, destroyed once.
        unsafe { (self.fns.destroy_program)(&mut program) };
        result
    }

    fn compile_created(
        &self,
        program: Program,
        name: &str,
        flags: &[*const c_char],
        target: Target,
    ) -> Result<Vec<u8>, GpuError> {
        // SAFETY: a live program and NUL-terminated option strings.
        let code = unsafe {
            (self.fns.compile_program)(
                program,
                c_int::try_from(flags.len()).unwrap_or(0),
                flags.as_ptr(),
            )
        };
        if code != 0 {
            let log = self.log(program);
            return Err(GpuError::Compile {
                log: format!("{name}: {}", log.trim_end()),
            });
        }
        let (size_fn, get_fn, call) = match target {
            Target::Cubin(_) => (self.fns.get_cubin_size, self.fns.get_cubin, "nvrtcGetCUBIN"),
            Target::Ptx(_) => (self.fns.get_ptx_size, self.fns.get_ptx, "nvrtcGetPTX"),
        };
        let mut size = 0usize;
        // SAFETY: a compiled program and an out pointer.
        self.check(call, unsafe { size_fn(program, &mut size) })?;
        let mut image = vec![0u8; size];
        // SAFETY: `image` has room for the `size` bytes NVRTC reported.
        self.check(call, unsafe { get_fn(program, image.as_mut_ptr().cast()) })?;
        if matches!(target, Target::Ptx(_)) && image.last() != Some(&0) {
            image.push(0);
        }
        Ok(image)
    }

    fn log(&self, program: Program) -> String {
        let mut size = 0usize;
        // SAFETY: a live program and an out pointer.
        if unsafe { (self.fns.get_program_log_size)(program, &mut size) } != 0 || size == 0 {
            return String::new();
        }
        let mut log = vec![0u8; size];
        // SAFETY: `log` has room for the `size` bytes NVRTC reported.
        if unsafe { (self.fns.get_program_log)(program, log.as_mut_ptr().cast()) } != 0 {
            return String::new();
        }
        let end = log.iter().position(|&b| b == 0).unwrap_or(log.len());
        String::from_utf8_lossy(&log[..end]).into_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_search_order_is_explicit_env_sonames_then_toolkits() {
        let env = |k: &str| match k {
            "CUDA_HOME" => Some(OsString::from("/x/cuda")),
            _ => None,
        };
        let list = |dir: &Path| -> Vec<PathBuf> {
            let names: &[&str] = match dir.to_str().unwrap() {
                "/x/cuda/lib64" => &[
                    "libnvrtc.so.12",
                    "libnvrtc-builtins.so.13.1",
                    "libnvrtc.so.13",
                    "libnvrtc.so",
                ],
                "/usr/local" => &["cuda-12.4", "cuda-11.8", "src"],
                "/usr/local/cuda-12.4/lib64" => &["libnvrtc.so.12"],
                _ => &[],
            };
            names.iter().map(|n| dir.join(n)).collect()
        };
        let got = candidates(None, &env, &list);
        let file = |p: &str| Candidate::File(PathBuf::from(p));
        assert_eq!(
            got,
            [
                Candidate::Soname("libnvrtc.so.13"),
                Candidate::Soname("libnvrtc.so.12"),
                Candidate::Soname("libnvrtc.so"),
                file("/x/cuda/lib64/libnvrtc.so.13"),
                file("/x/cuda/lib64/libnvrtc.so.12"),
                file("/x/cuda/lib64/libnvrtc.so"),
                file("/usr/local/cuda-12.4/lib64/libnvrtc.so.12"),
            ]
        );
        let with_env = |k: &str| match k {
            "EUHEDRAL_NVRTC" => Some(OsString::from("/opt/libnvrtc.so.13")),
            other => env(other),
        };
        assert_eq!(
            candidates(None, &with_env, &list)[0],
            file("/opt/libnvrtc.so.13"),
            "the environment comes first"
        );
        assert_eq!(
            candidates(Some(Path::new("/elsewhere/nvrtc.so")), &with_env, &list),
            [file("/elsewhere/nvrtc.so")],
            "an explicit path is the only candidate"
        );
    }
}
