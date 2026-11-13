//! NVRTC, the run-time compiler for kernel source, loaded at run time through the memory crate's loader (`dlopen`
//! on Linux, `LoadLibrary` on Windows; no link-time dependency).
//!
//! The library is found through a search order ([`candidates`]); the first load serves the process. A machine without
//! NVRTC can still load prebuilt images.

use std::ffi::{CStr, CString, OsString, c_char, c_int, c_void};
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

use euhedral_data_structures::sync::{HotScope, LifecycleMutex};
use euhedral_memory::cuda::loader::Library;

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

/// The target for a device of `sm` (`X * 10 + Y`) among the architectures NVRTC `supported`: its exact machine code
/// when supported, else PTX for the newest supported architecture that is not newer than the device (the driver
/// compiles PTX forward, never backward).
pub(crate) fn pick_target(supported: &[u32], sm: u32) -> Option<Target> {
    if supported.contains(&sm) {
        Some(Target::Cubin(sm))
    } else {
        supported
            .iter()
            .copied()
            .filter(|&a| a <= sm)
            .max()
            .map(Target::Ptx)
    }
}

/// A place to try loading the library from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Candidate {
    /// A name for the dynamic loader (a soname on Linux, a file name on Windows).
    Soname(&'static str),
    File(PathBuf),
}

/// The library paths to try, in order, for `explicit` (only it, when given), then `EUHEDRAL_NVRTC`, the loader's
/// own search, and the toolkit directories. `env` reads the environment and `list` lists a directory's entries as
/// full paths. See [`candidates_unix`] and [`candidates_windows`] for each platform's names and directories.
pub(crate) fn candidates(
    explicit: Option<&Path>,
    env: &dyn Fn(&str) -> Option<OsString>,
    list: &dyn Fn(&Path) -> Vec<PathBuf>,
) -> Vec<Candidate> {
    if cfg!(windows) {
        candidates_windows(explicit, env, list)
    } else {
        candidates_unix(explicit, env, list)
    }
}

/// The Linux search order: `explicit` (only it, when given), then `EUHEDRAL_NVRTC`, the loader's own search, and
/// the toolkit directories (`CUDA_HOME`, `CUDA_PATH`, `/usr/local/cuda`, then `/usr/local/cuda-*` newest first). A
/// directory contributes its `libnvrtc.so.<N>` files, newest first, then `libnvrtc.so`.
fn candidates_unix(
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

/// The Windows search order: `explicit` (only it, when given), then `EUHEDRAL_NVRTC`, the loader's own search by
/// file name (the application directory, the system directories and `PATH`, where a toolkit's installer puts its
/// `bin`), and the toolkit directories `CUDA_HOME` and `CUDA_PATH`, each as `bin\x64` (CUDA 13) then `bin` (CUDA 12
/// and before). A directory contributes its `nvrtc64_<N>_0.dll` files, newest first.
fn candidates_windows(
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
    for name in ["nvrtc64_130_0.dll", "nvrtc64_120_0.dll"] {
        push(Candidate::Soname(name));
    }
    for var in ["CUDA_HOME", "CUDA_PATH"] {
        let Some(home) = env(var) else { continue };
        let home = PathBuf::from(home);
        for dir in [home.join("bin").join("x64"), home.join("bin")] {
            let mut libs: Vec<(u64, PathBuf)> = list(&dir)
                .into_iter()
                .filter_map(|p| {
                    let name = p.file_name()?.to_str()?.to_ascii_lowercase();
                    let number = name.strip_prefix("nvrtc64_")?.strip_suffix("_0.dll")?;
                    Some((number.parse().ok()?, p))
                })
                .collect();
            libs.sort_by_key(|lib| std::cmp::Reverse(lib.0));
            for (_, path) in libs {
                push(Candidate::File(path));
            }
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
    /// The library file the compiler was loaded from, when the loader says.
    library: Option<PathBuf>,
    /// Names the build of the compiler beyond its version: the library file's name (a toolkit's carries the patch
    /// level), size and modification time. Part of the module cache key.
    build: String,
}

impl std::fmt::Debug for Nvrtc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Nvrtc")
            .field("version", &self.version)
            .finish_non_exhaustive()
    }
}

fn symbol(lib: &Library, name: &CStr) -> Result<*mut c_void, GpuError> {
    lib.require(name)
        .map_err(|e| GpuError::Load(format!("nvrtc: {e}")))
}

macro_rules! sym {
    ($lib:expr, $name:literal) => {
        // SAFETY: NVRTC exports this symbol with the C signature of the field it fills (nvrtc.h).
        unsafe { cast(symbol(&$lib, $name)?) }
    };
}

/// Whether a file name is NVRTC's builtins library (`libnvrtc-builtins.so*`, `nvrtc-builtins64_*.dll`).
fn is_builtins(name: &str) -> bool {
    if cfg!(windows) {
        let name = name.to_ascii_lowercase();
        name.starts_with("nvrtc-builtins64_") && name.ends_with(".dll")
    } else {
        name.starts_with("libnvrtc-builtins.so")
    }
}

/// Whether two paths name one file.
#[allow(
    clippy::disallowed_methods,
    reason = "LIFECYCLE(load): module loading is a build-time call; resolving a path blocks by nature"
)]
fn same_file(a: &Path, b: &Path) -> bool {
    match (std::fs::canonicalize(a), std::fs::canonicalize(b)) {
        (Ok(a), Ok(b)) => a == b,
        _ => a == b,
    }
}

/// The library's real file name, size and modification time.
#[allow(
    clippy::disallowed_methods,
    reason = "LIFECYCLE(load): module loading is a build-time call; reading a file's metadata blocks by nature"
)]
fn build_name(path: &Path) -> String {
    let real = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_owned());
    let name = real
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let (len, modified) = std::fs::metadata(&real)
        .map(|m| {
            let secs = m
                .modified()
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map_or(0, |d| d.as_secs());
            (m.len(), secs)
        })
        .unwrap_or((0, 0));
    format!("{name}:{len}:{modified}")
}

/// The process's compiler once loaded; a read is one acquire load.
static LOADED: OnceLock<Arc<Nvrtc>> = OnceLock::new();
/// Serializes the first load.
static LOADING: LifecycleMutex<()> = LifecycleMutex::new(());

impl Nvrtc {
    /// The process's NVRTC: the first successful load, by `explicit` or the search order. Loading blocks (file
    /// reads); it is a build-time call. A failed load is not remembered: a later call tries again.
    ///
    /// One compiler serves the process. Asking for a different library file once one is loaded is
    /// `Err(Invalid)`, not a silent use of the loaded one.
    pub fn shared(explicit: Option<&Path>) -> Result<Arc<Self>, GpuError> {
        // After the first load every call is one acquire load, so a stage that compiles a kernel takes no lock.
        if let Some(loaded) = LOADED.get() {
            return Self::agree(loaded, explicit);
        }
        // The first load is lifecycle work, also when a frame makes it.
        let _cold = HotScope::leave();
        let _loading = LOADING.lock();
        if let Some(loaded) = LOADED.get() {
            return Self::agree(loaded, explicit);
        }
        let nvrtc = Arc::new(Self::find(explicit)?);
        // Under `LOADING` nobody else sets the cell, so the set succeeds.
        let _ = LOADED.set(nvrtc.clone());
        Ok(nvrtc)
    }

    /// The loaded compiler, unless `explicit` names another library file.
    fn agree(loaded: &Arc<Self>, explicit: Option<&Path>) -> Result<Arc<Self>, GpuError> {
        if let (Some(want), Some(have)) = (explicit, &loaded.library)
            && !same_file(want, have)
        {
            return Err(GpuError::Invalid(format!(
                "NVRTC is already loaded from {}; this process cannot also use {}",
                have.display(),
                want.display()
            )));
        }
        Ok(loaded.clone())
    }

    #[allow(
        clippy::disallowed_methods,
        reason = "LIFECYCLE(load): module loading is a build-time call; listing a directory blocks by nature"
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
            // NVRTC loads its builtins library by name, and toolkit copies carry no run path: load the one beside
            // the compiler first, globally, so that load finds it.
            if let Some(dir) = preload {
                let mut builtins: Vec<PathBuf> = list(&dir)
                    .into_iter()
                    .filter(|p| {
                        p.file_name()
                            .and_then(|n| n.to_str())
                            .is_some_and(is_builtins)
                    })
                    .collect();
                builtins.sort();
                for path in builtins {
                    let _ = Library::open(&path.to_string_lossy());
                }
            }
            match Library::open(&name) {
                Ok(lib) => return Self::bind(lib),
                Err(e) => tried.push(e),
            }
        }
        Err(GpuError::Load(format!(
            "no NVRTC found (set EUHEDRAL_NVRTC, or CUDA_HOME to a toolkit): {}",
            tried.join("; ")
        )))
    }

    fn bind(lib: Library) -> Result<Self, GpuError> {
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
        let library = Library::file_of(symbol(&lib, c"nvrtcVersion")?);
        let build = library.as_deref().map(build_name).unwrap_or_default();
        let mut nvrtc = Self {
            fns,
            version: (0, 0),
            supported: Vec::new(),
            library,
            build,
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

    /// Names this build of the compiler beyond its version (see the module cache key).
    pub fn build(&self) -> &str {
        &self.build
    }

    /// `sm_XY` machine code when NVRTC supports the architecture, else PTX for the newest `compute_XY` it supports
    /// that is not newer than the device. An error when NVRTC supports nothing that old.
    pub fn target_for(&self, arch: (i32, i32)) -> Result<Target, GpuError> {
        let sm = u32::try_from(arch.0 * 10 + arch.1).unwrap_or(0);
        pick_target(&self.supported, sm).ok_or_else(|| {
            GpuError::Load(format!(
                "NVRTC {}.{} supports no architecture up to sm_{sm}",
                self.version.0, self.version.1
            ))
        })
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
    fn the_unix_search_order_is_explicit_env_sonames_then_toolkits() {
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
        let got = candidates_unix(None, &env, &list);
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
            candidates_unix(None, &with_env, &list)[0],
            file("/opt/libnvrtc.so.13"),
            "the environment comes first"
        );
        assert_eq!(
            candidates_unix(Some(Path::new("/elsewhere/nvrtc.so")), &with_env, &list),
            [file("/elsewhere/nvrtc.so")],
            "an explicit path is the only candidate"
        );
    }

    #[test]
    fn the_windows_search_order_is_explicit_env_names_then_toolkit_bins() {
        let root = Path::new("root").join("cuda");
        let x64 = root.join("bin").join("x64");
        let bin = root.join("bin");
        let env = |k: &str| match k {
            "CUDA_PATH" => Some(root.clone().into_os_string()),
            _ => None,
        };
        let list = |dir: &Path| -> Vec<PathBuf> {
            let names: &[&str] = if dir == x64 {
                &[
                    "nvrtc64_130_0.dll",
                    "nvrtc-builtins64_130.dll",
                    "cudart64_13.dll",
                ]
            } else if dir == bin {
                &[
                    "NVRTC64_120_0.DLL",
                    "nvrtc64_112_0.dll",
                    "nvrtc64_120_0.alt.dll",
                ]
            } else {
                &[]
            };
            names.iter().map(|n| dir.join(n)).collect()
        };
        let got = candidates_windows(None, &env, &list);
        assert_eq!(
            got,
            [
                Candidate::Soname("nvrtc64_130_0.dll"),
                Candidate::Soname("nvrtc64_120_0.dll"),
                Candidate::File(x64.join("nvrtc64_130_0.dll")),
                Candidate::File(bin.join("NVRTC64_120_0.DLL")),
                Candidate::File(bin.join("nvrtc64_112_0.dll")),
            ]
        );
        let with_env = |k: &str| match k {
            "EUHEDRAL_NVRTC" => Some(OsString::from("nvrtc.dll")),
            other => env(other),
        };
        assert_eq!(
            candidates_windows(None, &with_env, &list)[0],
            Candidate::File(PathBuf::from("nvrtc.dll")),
            "the environment comes first"
        );
        assert_eq!(
            candidates_windows(Some(Path::new("other.dll")), &with_env, &list),
            [Candidate::File(PathBuf::from("other.dll"))],
            "an explicit path is the only candidate"
        );
    }

    #[test]
    fn the_builtins_library_is_recognised_by_platform_name() {
        if cfg!(windows) {
            assert!(is_builtins("nvrtc-builtins64_130.dll"));
            assert!(!is_builtins("nvrtc64_130_0.dll"));
        } else {
            assert!(is_builtins("libnvrtc-builtins.so.13.1"));
            assert!(!is_builtins("libnvrtc.so.13"));
        }
    }

    #[test]
    fn the_target_is_never_newer_than_the_device() {
        let supported = [75, 80, 86, 89, 90, 100, 120];
        assert_eq!(pick_target(&supported, 120), Some(Target::Cubin(120)));
        // A device NVRTC supports exactly gets machine code.
        assert_eq!(pick_target(&supported, 89), Some(Target::Cubin(89)));
        // A device between two supported architectures gets the older one's PTX, never a newer one.
        assert_eq!(pick_target(&supported, 87), Some(Target::Ptx(86)));
        assert_eq!(pick_target(&supported, 121), Some(Target::Ptx(120)));
        // A device older than everything NVRTC knows has no target.
        assert_eq!(pick_target(&supported, 61), None);
        assert_eq!(pick_target(&[], 120), None);
    }
}
