//! The macOS system calls behind [`Sysctl`] and [`Probe`].

use super::Sysctl;
use super::resources::{HostMemory, Probe, ProcessUsage, ticks_to_ns};
use std::ffi::{CStr, CString, c_char, c_void};
use std::mem::MaybeUninit;
use std::ptr;

unsafe extern "C" {
    static mach_task_self_: libc::mach_port_t;
    fn mach_host_self() -> libc::mach_port_t;
    fn mach_port_deallocate(
        task: libc::mach_port_t,
        name: libc::mach_port_t,
    ) -> libc::kern_return_t;
    fn mach_timebase_info(info: *mut Timebase) -> libc::kern_return_t;
}

#[repr(C)]
#[derive(Default)]
struct Timebase {
    numer: u32,
    denom: u32,
}

/// `sysctlbyname`.
pub(crate) struct SystemSysctl;

impl Sysctl for SystemSysctl {
    fn integer(&self, name: &str) -> Option<u64> {
        let name = CString::new(name).ok()?;
        let mut buffer = [0u8; 8];
        let mut size: libc::size_t = buffer.len();
        // SAFETY: the output pointer and size describe `buffer`; nothing is written.
        let result = unsafe {
            libc::sysctlbyname(
                name.as_ptr(),
                buffer.as_mut_ptr().cast(),
                &mut size,
                ptr::null_mut(),
                0,
            )
        };
        if result != 0 {
            return None;
        }
        match size {
            4 => u64::try_from(i32::from_ne_bytes(buffer[..4].try_into().ok()?)).ok(),
            8 => Some(u64::from_ne_bytes(buffer)),
            _ => None,
        }
    }
}

type MsgSend = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
type MsgSendInteger = unsafe extern "C" fn(*mut c_void, *mut c_void) -> isize;
/// `BOOL` is one byte on every Apple ABI, whether `bool` or `signed char`.
type MsgSendBool = unsafe extern "C" fn(*mut c_void, *mut c_void) -> i8;
type MsgSendBoolSelector = unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void) -> i8;

/// `[NSProcessInfo processInfo]` and the selectors read from it, looked up through Foundation at
/// run time so the crate does not link it.
struct ProcessInfo {
    msg_send: *const c_void,
    object: *mut c_void,
    thermal_state: Option<*mut c_void>,
    low_power_mode: Option<*mut c_void>,
}

// SAFETY: the process-info object is an immutable process-wide singleton whose properties may be read
// from any thread, and selectors and function pointers are plain addresses.
unsafe impl Send for ProcessInfo {}

impl ProcessInfo {
    fn load() -> Option<Self> {
        const FOUNDATION: &CStr = c"/System/Library/Frameworks/Foundation.framework/Foundation";
        // SAFETY: a valid path and flags; the handle is kept for the life of the process.
        let handle =
            unsafe { libc::dlopen(FOUNDATION.as_ptr(), libc::RTLD_LAZY | libc::RTLD_LOCAL) };
        if handle.is_null() {
            return None;
        }
        let symbol = |name: &CStr| {
            // SAFETY: `handle` is open and `name` is NUL-terminated.
            Some(unsafe { libc::dlsym(handle, name.as_ptr()) }).filter(|symbol| !symbol.is_null())
        };
        let get_class = symbol(c"objc_getClass")?;
        let register_name = symbol(c"sel_registerName")?;
        let msg_send = symbol(c"objc_msgSend")?;
        // SAFETY: these are the runtime's `objc_getClass` and `sel_registerName`.
        let (get_class, register_name) = unsafe {
            (
                std::mem::transmute::<
                    *mut c_void,
                    unsafe extern "C" fn(*const c_char) -> *mut c_void,
                >(get_class),
                std::mem::transmute::<
                    *mut c_void,
                    unsafe extern "C" fn(*const c_char) -> *mut c_void,
                >(register_name),
            )
        };
        // SAFETY: NUL-terminated names.
        let selector = |name: &CStr| {
            Some(unsafe { register_name(name.as_ptr()) }).filter(|sel| !sel.is_null())
        };
        // SAFETY: as above.
        let class = Some(unsafe { get_class(c"NSProcessInfo".as_ptr()) })
            .filter(|class| !class.is_null())?;
        // SAFETY: `objc_msgSend` cast to the signature of `+[NSProcessInfo processInfo]`.
        let object = unsafe {
            std::mem::transmute::<*mut c_void, MsgSend>(msg_send)(class, selector(c"processInfo")?)
        };
        if object.is_null() {
            return None;
        }
        let responds = selector(c"respondsToSelector:")?;
        let supported = |sel: *mut c_void| {
            // SAFETY: `objc_msgSend` cast to the signature of `-respondsToSelector:`.
            unsafe {
                std::mem::transmute::<*mut c_void, MsgSendBoolSelector>(msg_send)(
                    object, responds, sel,
                ) != 0
            }
        };
        Some(Self {
            msg_send,
            object,
            thermal_state: selector(c"thermalState").filter(|&sel| supported(sel)),
            low_power_mode: selector(c"isLowPowerModeEnabled").filter(|&sel| supported(sel)),
        })
    }

    fn thermal_state(&self) -> Option<i64> {
        let sel = self.thermal_state?;
        // SAFETY: the object responds to `thermalState`, which returns an `NSInteger`.
        let state = unsafe {
            std::mem::transmute::<*const c_void, MsgSendInteger>(self.msg_send)(self.object, sel)
        };
        i64::try_from(state).ok()
    }

    fn low_power_mode(&self) -> Option<bool> {
        let sel = self.low_power_mode?;
        // SAFETY: the object responds to `isLowPowerModeEnabled`, which returns a `BOOL`.
        Some(
            unsafe {
                std::mem::transmute::<*const c_void, MsgSendBool>(self.msg_send)(self.object, sel)
            } != 0,
        )
    }
}

/// The live system.
pub(crate) struct SystemProbe {
    timebase: (u32, u32),
    host: libc::mach_port_t,
    process_info: Option<Option<ProcessInfo>>,
}

impl SystemProbe {
    pub(crate) fn new() -> Self {
        let mut timebase = Timebase::default();
        // SAFETY: `timebase` is a valid output.
        let timebase = match unsafe { mach_timebase_info(&mut timebase) } {
            libc::KERN_SUCCESS if timebase.denom != 0 => (timebase.numer, timebase.denom),
            _ => (1, 1),
        };
        Self {
            timebase,
            // SAFETY: no preconditions; the send right is released on drop.
            host: unsafe { mach_host_self() },
            process_info: None,
        }
    }

    fn process_info(&mut self) -> Option<&ProcessInfo> {
        self.process_info
            .get_or_insert_with(ProcessInfo::load)
            .as_ref()
    }

    fn getrusage_ns() -> Option<u64> {
        let mut usage = MaybeUninit::<libc::rusage>::zeroed();
        // SAFETY: `usage` is a valid output.
        if unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) } != 0 {
            return None;
        }
        // SAFETY: getrusage filled it.
        let usage = unsafe { usage.assume_init() };
        let ns = |time: libc::timeval| {
            u64::try_from(time.tv_sec)
                .ok()?
                .checked_mul(1_000_000_000)?
                .checked_add(u64::try_from(time.tv_usec).ok()? * 1_000)
        };
        ns(usage.ru_utime)?.checked_add(ns(usage.ru_stime)?)
    }
}

impl Drop for SystemProbe {
    fn drop(&mut self) {
        // SAFETY: `host` is a send right this probe owns.
        unsafe { mach_port_deallocate(mach_task_self_, self.host) };
    }
}

impl Probe for SystemProbe {
    /// `proc_pid_rusage` reports Mach ticks; when it fails, `getrusage` still gives the CPU time.
    fn process_usage(&mut self) -> Option<ProcessUsage> {
        let mut info = MaybeUninit::<libc::rusage_info_v3>::zeroed();
        // SAFETY: `info` is a valid output of the requested flavor.
        let result = unsafe {
            libc::proc_pid_rusage(
                libc::getpid(),
                libc::RUSAGE_INFO_V3,
                info.as_mut_ptr().cast(),
            )
        };
        if result != 0 {
            return Self::getrusage_ns().map(|cpu_ns| ProcessUsage {
                cpu_ns,
                io_bytes: None,
            });
        }
        // SAFETY: proc_pid_rusage filled it.
        let info = unsafe { info.assume_init() };
        let (numer, denom) = self.timebase;
        Some(ProcessUsage {
            cpu_ns: ticks_to_ns(
                info.ri_user_time.saturating_add(info.ri_system_time),
                numer,
                denom,
            ),
            io_bytes: Some(
                info.ri_diskio_bytesread
                    .saturating_add(info.ri_diskio_byteswritten),
            ),
        })
    }

    fn host_memory(&mut self) -> Option<HostMemory> {
        let total = SystemSysctl.integer("hw.memsize")?;
        // SAFETY: no preconditions.
        let page = u64::try_from(unsafe { libc::sysconf(libc::_SC_PAGESIZE) })
            .ok()
            .filter(|&page| page > 0)?;
        let mut stats = MaybeUninit::<libc::vm_statistics64>::zeroed();
        let mut count = libc::HOST_VM_INFO64_COUNT;
        // SAFETY: `stats` holds `count` integers.
        let result = unsafe {
            libc::host_statistics64(
                self.host,
                libc::HOST_VM_INFO64,
                stats.as_mut_ptr().cast(),
                &mut count,
            )
        };
        if result != libc::KERN_SUCCESS {
            return None;
        }
        // SAFETY: zero-initialized and partly or wholly filled by the kernel.
        let stats = unsafe { stats.assume_init() };
        Some(HostMemory::from_pages(
            total,
            page,
            stats.free_count.into(),
            stats.inactive_count.into(),
            stats.speculative_count.into(),
        ))
    }

    fn thermal_state(&mut self) -> Option<i64> {
        self.process_info()?.thermal_state()
    }

    fn low_power_mode(&mut self) -> Option<bool> {
        self.process_info()?.low_power_mode()
    }
}
