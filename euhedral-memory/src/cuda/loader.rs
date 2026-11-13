//! The dynamic loader behind the CUDA backend: open a library by name, look up a symbol, find the file a loaded
//! address came from. `dlopen`, `dlsym` and `dladdr` on Unix; `LoadLibraryExW`, `GetProcAddress` and
//! `GetModuleHandleExW` with `GetModuleFileNameW` on Windows. The driver ([`Driver`](super::Driver)) and the
//! run-time compiler of `euhedral-gpu` use this one type, so the two platforms differ only here.

use std::ffi::{CStr, c_void};
use std::path::PathBuf;

/// The CUDA driver library's name for the loader: `libcuda.so.1` on Unix, `nvcuda.dll` on Windows.
pub const DRIVER_LIBRARY: &str = if cfg!(windows) {
    "nvcuda.dll"
} else {
    "libcuda.so.1"
};

/// A library the loader opened. It is never closed: a driver or a compiler stays for the process, and the symbols
/// looked up in it stay valid.
#[derive(Clone, Copy, Debug)]
pub struct Library {
    /// The loader's handle, kept as an integer so a `Library` is `Send + Sync`.
    handle: usize,
}

impl Library {
    /// Opens the library `name`: a bare name goes through the loader's own search (the dynamic linker's on Unix;
    /// the application directory, the system directories and `PATH` on Windows); a path is loaded from there, and
    /// on Windows the libraries it depends on are looked for beside it first. The error names the library and
    /// carries the loader's reason, e.g. `"nvcuda.dll: ..."`.
    pub fn open(name: &str) -> Result<Library, String> {
        imp::open(name)
            .map(|handle| Library { handle })
            .map_err(|reason| format!("{name}: {reason}"))
    }

    /// The address of the exported `name`, or null when the library has no such symbol.
    pub fn symbol(&self, name: &CStr) -> *mut c_void {
        imp::symbol(self.handle, name)
    }

    /// Like [`symbol`](Self::symbol), but a missing symbol is the error `"no <name>"`.
    pub fn require(&self, name: &CStr) -> Result<*mut c_void, String> {
        let p = self.symbol(name);
        if p.is_null() {
            Err(format!("no {}", name.to_string_lossy()))
        } else {
            Ok(p)
        }
    }

    /// The file the loader mapped the code at `address` from, when it says. The library's own directory is its
    /// parent.
    pub fn file_of(address: *mut c_void) -> Option<PathBuf> {
        imp::file_of(address)
    }
}

#[cfg(unix)]
mod imp {
    use std::ffi::{CStr, CString, c_void};
    use std::path::PathBuf;

    pub(super) fn open(name: &str) -> Result<usize, String> {
        let c = CString::new(name).map_err(|_| "the name contains a NUL".to_owned())?;
        // SAFETY: a NUL-terminated name; the handle is never closed.
        let lib = unsafe { libc::dlopen(c.as_ptr(), libc::RTLD_NOW | libc::RTLD_GLOBAL) };
        if lib.is_null() {
            // SAFETY: `dlerror` returns a NUL-terminated string owned by the loader, or null.
            let p = unsafe { libc::dlerror() };
            Err(if p.is_null() {
                "unknown error".into()
            } else {
                // SAFETY: as above; copied before the next loader call.
                unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
            })
        } else {
            Ok(lib as usize)
        }
    }

    pub(super) fn symbol(handle: usize, name: &CStr) -> *mut c_void {
        // SAFETY: a handle `open` returned (never closed) and a NUL-terminated name.
        unsafe { libc::dlsym(handle as *mut c_void, name.as_ptr()) }
    }

    pub(super) fn file_of(address: *mut c_void) -> Option<PathBuf> {
        let mut info: libc::Dl_info = libc::Dl_info {
            dli_fname: std::ptr::null(),
            dli_fbase: std::ptr::null_mut(),
            dli_sname: std::ptr::null(),
            dli_saddr: std::ptr::null_mut(),
        };
        // SAFETY: an address inside a loaded library and an out pointer to a zeroed `Dl_info`.
        if unsafe { libc::dladdr(address, &mut info) } == 0 || info.dli_fname.is_null() {
            return None;
        }
        // SAFETY: `dli_fname` is a NUL-terminated path owned by the loader.
        let name = unsafe { CStr::from_ptr(info.dli_fname) };
        Some(PathBuf::from(name.to_string_lossy().into_owned()))
    }
}

#[cfg(windows)]
mod imp {
    use std::ffi::{CStr, OsString, c_void};
    use std::os::windows::ffi::{OsStrExt, OsStringExt};
    use std::path::{Path, PathBuf};

    use windows_sys::Win32::Foundation::{HMODULE, MAX_PATH};
    use windows_sys::Win32::System::LibraryLoader::{
        GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        GetModuleFileNameW, GetModuleHandleExW, GetProcAddress, LOAD_WITH_ALTERED_SEARCH_PATH,
        LoadLibraryExW,
    };

    pub(super) fn open(name: &str) -> Result<usize, String> {
        let wide: Vec<u16> = std::ffi::OsStr::new(name)
            .encode_wide()
            .chain(Some(0))
            .collect();
        if wide[..wide.len() - 1].contains(&0) {
            return Err("the name contains a NUL".into());
        }
        // A path (absolute) searches its own directory for the libraries it depends on; a bare name uses the
        // standard search order.
        let flags = if Path::new(name).is_absolute() {
            LOAD_WITH_ALTERED_SEARCH_PATH
        } else {
            0
        };
        // SAFETY: a NUL-terminated wide name and no file handle (reserved, null); the module is never freed.
        let module = unsafe { LoadLibraryExW(wide.as_ptr(), std::ptr::null_mut(), flags) };
        if module.is_null() {
            Err(std::io::Error::last_os_error().to_string())
        } else {
            Ok(module as usize)
        }
    }

    pub(super) fn symbol(handle: usize, name: &CStr) -> *mut c_void {
        // SAFETY: a module `open` returned (never freed) and a NUL-terminated name.
        match unsafe { GetProcAddress(handle as HMODULE, name.as_ptr().cast()) } {
            Some(f) => f as usize as *mut c_void,
            None => std::ptr::null_mut(),
        }
    }

    pub(super) fn file_of(address: *mut c_void) -> Option<PathBuf> {
        let mut module: HMODULE = std::ptr::null_mut();
        // SAFETY: the flags ask for the module containing the address, without touching its reference count; an
        // out pointer.
        let ok = unsafe {
            GetModuleHandleExW(
                GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS
                    | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                address.cast(),
                &mut module,
            )
        };
        if ok == 0 {
            return None;
        }
        // Long paths exceed `MAX_PATH`: grow until the name fits.
        let mut buf = vec![0u16; MAX_PATH as usize];
        loop {
            // SAFETY: `buf` has `buf.len()` elements.
            let n = unsafe { GetModuleFileNameW(module, buf.as_mut_ptr(), buf.len() as u32) };
            if n == 0 {
                return None;
            }
            if (n as usize) < buf.len() {
                buf.truncate(n as usize);
                return Some(PathBuf::from(OsString::from_wide(&buf)));
            }
            let doubled = buf.len() * 2;
            buf.resize(doubled, 0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A library every process of the platform can load, a function it exports, and a substring of its file name.
    #[cfg(windows)]
    const SYSTEM: (&str, &CStr, &str) = ("kernel32.dll", c"GetCurrentProcessId", "kernel32");
    #[cfg(all(unix, target_os = "macos"))]
    const SYSTEM: (&str, &CStr, &str) = ("libSystem.B.dylib", c"getpid", "libsystem");
    #[cfg(all(unix, not(target_os = "macos")))]
    const SYSTEM: (&str, &CStr, &str) = ("libc.so.6", c"getpid", "libc");

    #[test]
    fn a_missing_library_is_an_error_naming_it() {
        let err = Library::open("euhedral-no-such-library-4f2a").unwrap_err();
        assert!(err.starts_with("euhedral-no-such-library-4f2a: "), "{err}");
        assert!(
            err.len() > "euhedral-no-such-library-4f2a: ".len(),
            "carries a reason: {err}"
        );
    }

    #[test]
    fn a_name_with_a_nul_is_an_error() {
        assert!(Library::open("a\0b").is_err());
    }

    #[test]
    fn the_driver_library_is_found_or_reported_by_name() {
        // With no driver installed this is the skip path of every device test: an error, never a panic.
        match Library::open(DRIVER_LIBRARY) {
            Ok(lib) => assert!(!lib.symbol(c"cuInit").is_null()),
            Err(e) => assert!(e.starts_with(DRIVER_LIBRARY), "{e}"),
        }
    }

    #[test]
    fn a_missing_symbol_is_named() {
        let lib = Library::open(SYSTEM.0).unwrap();
        assert!(lib.symbol(c"euhedral_no_such_symbol").is_null());
        let err = lib.require(c"euhedral_no_such_symbol").unwrap_err();
        assert_eq!(err, "no euhedral_no_such_symbol");
    }

    #[test]
    fn a_symbol_is_found_and_its_file_is_the_library() {
        let lib = Library::open(SYSTEM.0).unwrap();
        let p = lib.require(SYSTEM.1).unwrap();
        let file = Library::file_of(p).expect("the loader names the file");
        let name = file.file_name().unwrap().to_string_lossy().to_lowercase();
        assert!(name.contains(SYSTEM.2), "{file:?}");
        assert!(file.parent().is_some());
    }

    #[test]
    fn an_address_outside_any_library_has_no_file() {
        assert_eq!(Library::file_of(std::ptr::null_mut()), None);
    }
}
