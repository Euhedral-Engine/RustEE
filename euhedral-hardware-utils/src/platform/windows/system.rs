//! The Win32 calls behind [`Probe`] and the topology buffer.

use super::resources::{Memory, Probe, Reading};
use crate::util::CpuSet;
use std::ffi::c_void;
use std::mem::{size_of, zeroed};
use std::ptr;
use windows_sys::Win32::Foundation::{ERROR_INSUFFICIENT_BUFFER, FILETIME, GetLastError};
use windows_sys::Win32::System::JobObjects::{
    IsProcessInJob, JOB_OBJECT_CPU_RATE_CONTROL_ENABLE, JOB_OBJECT_CPU_RATE_CONTROL_HARD_CAP,
    JOB_OBJECT_CPU_RATE_CONTROL_MIN_MAX_RATE, JOB_OBJECT_CPU_RATE_CONTROL_WEIGHT_BASED,
    JOB_OBJECT_LIMIT_JOB_MEMORY, JOB_OBJECT_LIMIT_PROCESS_MEMORY,
    JOBOBJECT_BASIC_AND_IO_ACCOUNTING_INFORMATION, JOBOBJECT_CPU_RATE_CONTROL_INFORMATION,
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JOBOBJECT_LIMIT_VIOLATION_INFORMATION,
    JOBOBJECTINFOCLASS, JobObjectBasicAndIoAccountingInformation,
    JobObjectCpuRateControlInformation, JobObjectExtendedLimitInformation,
    JobObjectLimitViolationInformation, QueryInformationJobObject,
};
use windows_sys::Win32::System::Power::{GetSystemPowerStatus, SYSTEM_POWER_STATUS};
use windows_sys::Win32::System::ProcessStatus::{
    K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS_EX,
};
use windows_sys::Win32::System::SystemInformation::{
    GetLogicalProcessorInformationEx, GlobalMemoryStatusEx, MEMORYSTATUSEX, RelationAll,
    SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX,
};
use windows_sys::Win32::System::Threading::{
    GetCurrentProcess, GetProcessAffinityMask, GetProcessGroupAffinity, GetProcessIoCounters,
    GetProcessTimes, IO_COUNTERS,
};

/// The raw `GetLogicalProcessorInformationEx(RelationAll)` buffer.
pub(crate) fn logical_processor_information() -> Option<Vec<u8>> {
    let mut length = 0u32;
    // SAFETY: a null buffer with a zero length only reports the size needed.
    unsafe { GetLogicalProcessorInformationEx(RelationAll, ptr::null_mut(), &mut length) };
    loop {
        // SAFETY: no preconditions.
        if unsafe { GetLastError() } != ERROR_INSUFFICIENT_BUFFER || length == 0 {
            return None;
        }
        // u64 storage keeps the records aligned.
        let mut storage = vec![0u64; (length as usize).div_ceil(8)];
        let mut written = length;
        // SAFETY: `storage` holds at least `written` bytes.
        let ok = unsafe {
            GetLogicalProcessorInformationEx(
                RelationAll,
                storage
                    .as_mut_ptr()
                    .cast::<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(),
                &mut written,
            )
        };
        if ok != 0 {
            let bytes: Vec<u8> = storage.iter().flat_map(|word| word.to_ne_bytes()).collect();
            return Some(bytes[..written as usize].to_vec());
        }
        // The topology grew between the calls; retry with the new size.
        length = written;
    }
}

/// A job query into `T`, or `None` when it fails.
fn query_job<T>(class: JOBOBJECTINFOCLASS) -> Option<T> {
    // SAFETY: every queried type is a plain-data Win32 struct for which all-zero is valid.
    let mut info: T = unsafe { zeroed() };
    // SAFETY: the pointer and length describe `info`; a null job is the process's own job.
    let ok = unsafe {
        QueryInformationJobObject(
            ptr::null_mut(),
            class,
            (&raw mut info).cast::<c_void>(),
            u32::try_from(size_of::<T>()).ok()?,
            ptr::null_mut(),
        )
    };
    (ok != 0).then_some(info)
}

fn filetime_ns(time: FILETIME) -> u64 {
    ((u64::from(time.dwHighDateTime) << 32) | u64::from(time.dwLowDateTime)).saturating_mul(100)
}

fn ticks_ns(ticks: i64) -> Option<u64> {
    u64::try_from(ticks)
        .ok()
        .map(|ticks| ticks.saturating_mul(100))
}

/// The live system.
#[derive(Debug, Default)]
pub(crate) struct SystemProbe;

impl SystemProbe {
    pub(crate) fn new() -> Self {
        Self
    }

    /// The process may be placed in a job at any time, so this is asked on every read.
    fn in_job() -> Option<bool> {
        let mut result = 0;
        // SAFETY: the pseudo-handle needs no closing; a null job asks about any job.
        let ok = unsafe { IsProcessInJob(GetCurrentProcess(), ptr::null_mut(), &mut result) };
        (ok != 0).then_some(result != 0)
    }

    fn job_accounting() -> Option<JOBOBJECT_BASIC_AND_IO_ACCOUNTING_INFORMATION> {
        query_job(JobObjectBasicAndIoAccountingInformation)
    }

    fn host_memory() -> Option<Memory> {
        // SAFETY: all-zero is a valid MEMORYSTATUSEX.
        let mut status: MEMORYSTATUSEX = unsafe { zeroed() };
        status.dwLength = size_of::<MEMORYSTATUSEX>() as u32;
        // SAFETY: `status` is a valid output with its length set.
        if unsafe { GlobalMemoryStatusEx(&mut status) } == 0 {
            return None;
        }
        Some(Memory {
            limit_bytes: status.ullTotalPhys,
            usage_bytes: status.ullTotalPhys.saturating_sub(status.ullAvailPhys),
        })
    }

    fn private_bytes() -> Option<u64> {
        // SAFETY: all-zero is a valid PROCESS_MEMORY_COUNTERS_EX.
        let mut counters: PROCESS_MEMORY_COUNTERS_EX = unsafe { zeroed() };
        let size = size_of::<PROCESS_MEMORY_COUNTERS_EX>() as u32;
        // SAFETY: the pointer and size describe `counters`.
        let ok = unsafe {
            K32GetProcessMemoryInfo(GetCurrentProcess(), (&raw mut counters).cast(), size)
        };
        (ok != 0).then_some(counters.PrivateUsage as u64)
    }
}

impl Probe for SystemProbe {
    /// A weight-based or soft rate is not a capacity limit, so only hard caps and maximum rates
    /// count.
    fn cpu_rate_cap(&mut self) -> Reading<f64> {
        match Self::in_job() {
            Some(true) => {}
            Some(false) => return Reading::Unsupported,
            None => return Reading::Failed,
        }
        let Some(info) =
            query_job::<JOBOBJECT_CPU_RATE_CONTROL_INFORMATION>(JobObjectCpuRateControlInformation)
        else {
            return Reading::Failed;
        };
        let flags = info.ControlFlags;
        // SAFETY: the flags say which member of the union the job set.
        let rate = unsafe {
            if flags & JOB_OBJECT_CPU_RATE_CONTROL_ENABLE == 0
                || flags & JOB_OBJECT_CPU_RATE_CONTROL_WEIGHT_BASED != 0
            {
                None
            } else if flags & JOB_OBJECT_CPU_RATE_CONTROL_MIN_MAX_RATE != 0 {
                Some(u32::from(info.Anonymous.Anonymous.MaxRate))
            } else if flags & JOB_OBJECT_CPU_RATE_CONTROL_HARD_CAP != 0 {
                Some(info.Anonymous.CpuRate)
            } else {
                None
            }
        };
        match rate.filter(|&rate| rate > 0) {
            Some(rate) => Reading::Value(f64::from(rate.min(10_000)) / 10_000.0),
            None => Reading::Unsupported,
        }
    }

    fn cpu_time_ns(&mut self) -> Option<u64> {
        if Self::in_job()? {
            let basic = Self::job_accounting()?.BasicInfo;
            return ticks_ns(basic.TotalUserTime)?.checked_add(ticks_ns(basic.TotalKernelTime)?);
        }
        // SAFETY: all-zero FILETIMEs are valid outputs.
        let mut times: [FILETIME; 4] = unsafe { zeroed() };
        let [creation, exit, kernel, user] = &mut times;
        // SAFETY: four valid outputs and the process pseudo-handle.
        if unsafe { GetProcessTimes(GetCurrentProcess(), creation, exit, kernel, user) } == 0 {
            return None;
        }
        filetime_ns(*kernel).checked_add(filetime_ns(*user))
    }

    /// The job's committed memory against its limit, the process's private bytes against a
    /// per-process limit, or the host's physical memory.
    fn memory(&mut self) -> Option<Memory> {
        if Self::in_job()? {
            let limits = query_job::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>(
                JobObjectExtendedLimitInformation,
            )?;
            let flags = limits.BasicLimitInformation.LimitFlags;
            if flags & JOB_OBJECT_LIMIT_JOB_MEMORY != 0 {
                let usage = query_job::<JOBOBJECT_LIMIT_VIOLATION_INFORMATION>(
                    JobObjectLimitViolationInformation,
                )?;
                return Some(Memory {
                    limit_bytes: limits.JobMemoryLimit as u64,
                    usage_bytes: usage.JobMemory,
                });
            }
            if flags & JOB_OBJECT_LIMIT_PROCESS_MEMORY != 0 {
                return Some(Memory {
                    limit_bytes: limits.ProcessMemoryLimit as u64,
                    usage_bytes: Self::private_bytes()?,
                });
            }
        }
        Self::host_memory()
    }

    fn io_bytes(&mut self) -> Option<u64> {
        let io = if Self::in_job()? {
            Self::job_accounting()?.IoInfo
        } else {
            // SAFETY: all-zero is a valid IO_COUNTERS.
            let mut io: IO_COUNTERS = unsafe { zeroed() };
            // SAFETY: `io` is a valid output.
            if unsafe { GetProcessIoCounters(GetCurrentProcess(), &mut io) } == 0 {
                return None;
            }
            io
        };
        Some(io.ReadTransferCount.saturating_add(io.WriteTransferCount))
    }

    /// A process confined to one group has a mask within it; one spanning several groups may run
    /// anywhere in them.
    fn process_cpus(&mut self) -> Option<CpuSet> {
        // SAFETY: no preconditions; the pseudo-handle needs no closing.
        let process = unsafe { GetCurrentProcess() };
        let mut count = 0u16;
        // SAFETY: a zero count with a null array only reports the count.
        unsafe { GetProcessGroupAffinity(process, &mut count, ptr::null_mut()) };
        let mut groups = vec![0u16; usize::from(count.max(1))];
        count = groups.len() as u16;
        // SAFETY: `groups` holds `count` entries.
        if unsafe { GetProcessGroupAffinity(process, &mut count, groups.as_mut_ptr()) } == 0 {
            return None;
        }
        groups.truncate(usize::from(count));
        if let [group] = groups[..] {
            let (mut process_mask, mut system_mask) = (0usize, 0usize);
            // SAFETY: two valid outputs and the process pseudo-handle.
            if unsafe { GetProcessAffinityMask(process, &mut process_mask, &mut system_mask) } == 0
            {
                return None;
            }
            let mut words = vec![0u64; usize::from(group) + 1];
            words[usize::from(group)] = process_mask as u64;
            return Some(CpuSet::from_words(&words));
        }
        let mut cpus = CpuSet::new();
        for group in groups {
            let first = usize::from(group) * 64;
            cpus = cpus.union(&CpuSet::range(first, first + 64));
        }
        Some(cpus)
    }

    fn battery_saver(&mut self) -> Reading<bool> {
        // SAFETY: all-zero is a valid SYSTEM_POWER_STATUS.
        let mut status: SYSTEM_POWER_STATUS = unsafe { zeroed() };
        // SAFETY: `status` is a valid output.
        if unsafe { GetSystemPowerStatus(&mut status) } == 0 {
            return Reading::Failed;
        }
        match status.SystemStatusFlag {
            0 => Reading::Value(false),
            1 => Reading::Value(true),
            _ => Reading::Unsupported,
        }
    }
}
