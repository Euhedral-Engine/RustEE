//! Windows thread affinity through processor-group affinity, and the process timer resolution
//! (upstream `WindowsAffinity` and its native library).
//!
//! A thread runs in one processor group at a time, so a mask spanning two groups is rejected. The
//! timer resolution is a request of the whole process that Windows drops when the process exits.

use crate::affinity::{AffinityCapability, AffinityProvider, masks::nonzero_words};
use crate::util::CpuSet;
use windows_sys::Win32::System::Kernel::PROCESSOR_NUMBER;
use windows_sys::Win32::System::LibraryLoader::{GetModuleHandleW, GetProcAddress};
use windows_sys::Win32::System::SystemInformation::GROUP_AFFINITY;
use windows_sys::Win32::System::Threading::{
    GetCurrentProcessorNumberEx, GetCurrentThread, GetThreadGroupAffinity, SetThreadGroupAffinity,
};

type NtSetTimerResolution =
    unsafe extern "system" fn(desired: u32, set: u8, current: *mut u32) -> i32;

#[derive(Debug)]
pub(crate) struct WindowsAffinity {
    set_timer_resolution: Option<NtSetTimerResolution>,
}

impl WindowsAffinity {
    pub(crate) fn new() -> Self {
        let ntdll: Vec<u16> = "ntdll.dll\0".encode_utf16().collect();
        // SAFETY: a NUL-terminated module name; ntdll is mapped into every process.
        let module = unsafe { GetModuleHandleW(ntdll.as_ptr()) };
        let set_timer_resolution = if module.is_null() {
            None
        } else {
            // SAFETY: a NUL-terminated export name of a loaded module.
            unsafe { GetProcAddress(module, c"NtSetTimerResolution".as_ptr().cast()) }.map(
                |function| {
                    // SAFETY: the export has this signature.
                    unsafe {
                        std::mem::transmute::<
                            unsafe extern "system" fn() -> isize,
                            NtSetTimerResolution,
                        >(function)
                    }
                },
            )
        };
        Self {
            set_timer_resolution,
        }
    }

    fn set(mask: &[u64]) -> bool {
        if nonzero_words(mask) != 1 {
            return false;
        }
        let Some((group, &word)) = mask.iter().enumerate().find(|(_, word)| **word != 0) else {
            return false;
        };
        let (Ok(group), Ok(mask)) = (u16::try_from(group), usize::try_from(word)) else {
            return false;
        };
        let affinity = GROUP_AFFINITY {
            Mask: mask,
            Group: group,
            Reserved: [0; 3],
        };
        // SAFETY: a valid affinity for the calling thread's pseudo-handle; the previous value is not
        // requested.
        unsafe { SetThreadGroupAffinity(GetCurrentThread(), &affinity, std::ptr::null_mut()) != 0 }
    }
}

impl AffinityProvider for WindowsAffinity {
    fn capability(&self) -> AffinityCapability {
        AffinityCapability::Exact
    }

    fn current_cpu(&self) -> Option<usize> {
        let mut number = PROCESSOR_NUMBER {
            Group: 0,
            Number: 0,
            Reserved: 0,
        };
        // SAFETY: `number` is a valid output.
        unsafe { GetCurrentProcessorNumberEx(&mut number) };
        Some(usize::from(number.Group) * 64 + usize::from(number.Number))
    }

    fn capture_affinity(&self) -> Option<Vec<u64>> {
        let mut affinity = GROUP_AFFINITY {
            Mask: 0,
            Group: 0,
            Reserved: [0; 3],
        };
        // SAFETY: `affinity` is a valid output for the calling thread's pseudo-handle.
        if unsafe { GetThreadGroupAffinity(GetCurrentThread(), &mut affinity) } == 0
            || affinity.Mask == 0
        {
            return None;
        }
        let mut words = vec![0u64; usize::from(affinity.Group) + 1];
        words[usize::from(affinity.Group)] = affinity.Mask as u64;
        Some(CpuSet::from_words(&words).words().to_vec())
    }

    fn apply_exact(&self, mask: &[u64]) -> bool {
        Self::set(mask)
    }

    fn restore_exact(&self, mask: &[u64]) -> bool {
        Self::set(mask)
    }

    /// Requests the resolution in 100 ns units; Windows rounds it to a supported value.
    fn set_timer_resolution(&self, nanos: u64) -> bool {
        let Some(set_timer_resolution) = self.set_timer_resolution else {
            return false;
        };
        let units = u32::try_from((nanos / 100).max(1)).unwrap_or(u32::MAX);
        let mut current = 0;
        // SAFETY: a valid output; NtSetTimerResolution has no other preconditions.
        unsafe { set_timer_resolution(units, 1, &mut current) >= 0 }
    }
}
