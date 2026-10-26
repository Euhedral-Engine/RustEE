//! The real Windows calls, with `windows-sys`: what [`Sys`] means on Windows.
//!
//! - **IoRing is resolved at run time, not linked.** `windows-sys` declares its functions against
//!   `api-ms-win-core-ioring-l1-1-0.dll` and `kernel32.dll`, so a binary that linked them would
//!   not start on a Windows that lacks them (Windows 10, and Windows 11 before 22H2). Only the
//!   types and constants are taken from `windows-sys`; the functions come from `LoadLibraryExW`
//!   and `GetProcAddress`.
//! - **`AcceptEx` and `ConnectEx`** come from `WSAIoctl`, as Winsock extensions are meant to.
//! - Every `unsafe` block is one call, with the reason it is sound.

use std::ffi::c_void;
use std::io;
use std::os::windows::ffi::OsStrExt;
use std::path::Path;
use std::ptr::{null, null_mut};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicI32, Ordering};

use windows_sys::Wdk::Storage::FileSystem::{
    FILE_MODE_INFORMATION, FILE_SYNCHRONOUS_IO_ALERT, FILE_SYNCHRONOUS_IO_NONALERT,
    FileIoCompletionNotificationInformation, FileModeInformation, NtQueryInformationFile,
};
use windows_sys::Win32::Foundation::{
    CloseHandle, FARPROC, GENERIC_READ, GENERIC_WRITE, HANDLE, INVALID_HANDLE_VALUE,
    IORING_E_SUBMISSION_QUEUE_FULL, RtlNtStatusToDosError, WAIT_TIMEOUT,
};
use windows_sys::Win32::Networking::WinSock::{
    AF_INET, AF_INET6, INVALID_SOCKET, IPPROTO_TCP, LPFN_ACCEPTEX, LPFN_CONNECTEX,
    SIO_GET_EXTENSION_FUNCTION_POINTER, SO_UPDATE_ACCEPT_CONTEXT, SO_UPDATE_CONNECT_CONTEXT,
    SOCK_STREAM, SOCKADDR, SOCKADDR_STORAGE, SOCKET, SOL_SOCKET, WSA_FLAG_NO_HANDLE_INHERIT,
    WSA_FLAG_OVERLAPPED, WSABUF, WSAID_ACCEPTEX, WSAID_CONNECTEX, WSAIoctl, WSARecv, WSASend,
    WSASendTo, WSASocketW, bind, closesocket, getsockname, setsockopt,
};
use windows_sys::Win32::Storage::FileSystem::{
    CREATE_ALWAYS, CREATE_NEW, CreateFileW, FILE_ATTRIBUTE_NORMAL, FILE_FLAG_OVERLAPPED,
    FILE_SHARE_DELETE, FILE_SHARE_READ, FILE_SHARE_WRITE, FILE_WRITE_FLAGS, FILE_WRITE_FLAGS_NONE,
    HIORING, IORING_BUFFER_REF, IORING_BUFFER_REF_0, IORING_CAPABILITIES, IORING_CQE,
    IORING_CREATE_ADVISORY_FLAGS_NONE, IORING_CREATE_FLAGS, IORING_CREATE_REQUIRED_FLAGS_NONE,
    IORING_FEATURE_UM_EMULATION, IORING_HANDLE_REF, IORING_HANDLE_REF_0, IORING_OP_CODE,
    IORING_OP_READ, IORING_OP_WRITE, IORING_REF_RAW, IORING_SQE_FLAGS, IORING_VERSION,
    IORING_VERSION_1, IORING_VERSION_3 as WIN_IORING_VERSION_3, IOSQE_FLAGS_NONE,
    MOVEFILE_REPLACE_EXISTING, MOVEFILE_WRITE_THROUGH, MoveFileExW, OPEN_ALWAYS, OPEN_EXISTING,
    ReadFile, WriteFile,
};
use windows_sys::Win32::System::IO::{
    CreateIoCompletionPort, GetQueuedCompletionStatusEx, IO_STATUS_BLOCK, OVERLAPPED,
    OVERLAPPED_ENTRY, PostQueuedCompletionStatus,
};
use windows_sys::Win32::System::LibraryLoader::{
    GetProcAddress, LOAD_LIBRARY_SEARCH_SYSTEM32, LoadLibraryExW,
};
use windows_sys::Win32::System::Threading::{PTP_CALLBACK_INSTANCE, TrySubmitThreadpoolCallback};

use super::super::{Create, OpenSpec};
use super::sys::{
    ACCEPT_BUF, BuildError, Cqe, Handle, HandleMode, Overlapped, Packet, RingProbe, SockAddr, Sys,
    hresult_error, is_nt_warning, socket_status_error, verbatim_path,
};

const _: () = assert!(size_of::<Overlapped>() == size_of::<OVERLAPPED>());
const _: () = assert!(align_of::<Overlapped>() == align_of::<OVERLAPPED>());

const ERROR_IO_PENDING: i32 = 997;
const WSAEINVAL: i32 = 10022;

/// A `Result` from the thread's last system error, for a call that reported failure.
fn last_error() -> io::Error {
    io::Error::last_os_error()
}

/// The result of starting an overlapped call that returned `ok`: `Ok` if a packet will follow.
///
/// Pending is success. A call that reports failure still queues a packet when its status is a
/// warning (`ERROR_MORE_DATA` of a message-mode pipe, `WSAEMSGSIZE` of a datagram): the I/O
/// manager posts for `!NT_ERROR(status) || pending returned`. Only the `OVERLAPPED`'s own status
/// tells which; `kernel32` pre-sets `STATUS_PENDING` even when a call fails without an IRP, so a
/// bare "not an error" test would be wrong, and only a warning counts.
///
/// The engine does not complete the operation until this returns, so reading `ov` is safe.
fn started(ok: bool, ov: *const Overlapped) -> io::Result<()> {
    if ok {
        return Ok(());
    }
    let error = last_error();
    if error.raw_os_error() == Some(ERROR_IO_PENDING) {
        return Ok(());
    }
    // SAFETY: `ov` is the caller's valid `OVERLAPPED`; the kernel wrote its status during the call
    // that just returned, and nothing else writes it until the engine completes the operation.
    let internal = unsafe { Overlapped::load_internal(ov) };
    if is_nt_warning(internal as u32 as i32) {
        Ok(())
    } else {
        Err(error)
    }
}

fn wide(path: &Path) -> io::Result<Vec<u16>> {
    let mut wide: Vec<u16> = path.as_os_str().encode_wide().collect();
    if wide.contains(&0) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "a path with a NUL character",
        ));
    }
    // A path near the ordinary limit goes as a verbatim path (`\\?\`), which Win32 takes at its
    // full length; it must be absolute and normalized first.
    if wide.len() >= 240
        && let Ok(absolute) = std::path::absolute(path)
        && let Some(verbatim) = absolute.to_str().and_then(verbatim_path)
    {
        wide = verbatim.encode_utf16().collect();
    }
    wide.push(0);
    Ok(wide)
}

fn wide_str(text: &str) -> Vec<u16> {
    text.encode_utf16().chain(Some(0)).collect()
}

/// The IoRing entry points, found at run time.
struct RingApi {
    query: unsafe extern "system" fn(*mut IORING_CAPABILITIES) -> i32,
    create: unsafe extern "system" fn(
        IORING_VERSION,
        IORING_CREATE_FLAGS,
        u32,
        u32,
        *mut HIORING,
    ) -> i32,
    close: unsafe extern "system" fn(HIORING) -> i32,
    submit: unsafe extern "system" fn(HIORING, u32, u32, *mut u32) -> i32,
    pop: unsafe extern "system" fn(HIORING, *mut IORING_CQE) -> i32,
    supported: unsafe extern "system" fn(HIORING, IORING_OP_CODE) -> i32,
    build_read: unsafe extern "system" fn(
        HIORING,
        IORING_HANDLE_REF,
        IORING_BUFFER_REF,
        u32,
        u64,
        usize,
        IORING_SQE_FLAGS,
    ) -> i32,
    /// Absent before Windows 11 22H2.
    build_write: Option<
        unsafe extern "system" fn(
            HIORING,
            IORING_HANDLE_REF,
            IORING_BUFFER_REF,
            u32,
            u64,
            FILE_WRITE_FLAGS,
            usize,
            IORING_SQE_FLAGS,
        ) -> i32,
    >,
}

/// An entry point by name, from the API-set DLL or else from kernel32 (a system DLL, found only
/// in the system directory).
fn proc_address(name: &[u8]) -> Option<usize> {
    for module in ["api-ms-win-core-ioring-l1-1-0.dll", "kernel32.dll"] {
        let module = wide_str(module);
        // SAFETY: a NUL-terminated wide string, no file handle, and a flag that restricts the
        // search to the system directory.
        let handle =
            unsafe { LoadLibraryExW(module.as_ptr(), null_mut(), LOAD_LIBRARY_SEARCH_SYSTEM32) };
        if handle.is_null() {
            continue;
        }
        // SAFETY: a loaded module (never freed) and a NUL-terminated name.
        let address: FARPROC = unsafe { GetProcAddress(handle, name.as_ptr()) };
        if let Some(address) = address {
            return Some(address as usize);
        }
    }
    None
}

/// Loads an entry point as the function type `$ty`.
macro_rules! entry {
    ($name:literal, $ty:ty) => {
        proc_address(concat!($name, "\0").as_bytes()).map(|address| {
            // SAFETY: the address is the named export, whose signature is `$ty` (ioringapi.h).
            unsafe { std::mem::transmute::<usize, $ty>(address) }
        })
    };
}

impl RingApi {
    fn load() -> Option<Self> {
        Some(Self {
            query: entry!(
                "QueryIoRingCapabilities",
                unsafe extern "system" fn(*mut IORING_CAPABILITIES) -> i32
            )?,
            create: entry!(
                "CreateIoRing",
                unsafe extern "system" fn(
                    IORING_VERSION,
                    IORING_CREATE_FLAGS,
                    u32,
                    u32,
                    *mut HIORING,
                ) -> i32
            )?,
            close: entry!("CloseIoRing", unsafe extern "system" fn(HIORING) -> i32)?,
            submit: entry!(
                "SubmitIoRing",
                unsafe extern "system" fn(HIORING, u32, u32, *mut u32) -> i32
            )?,
            pop: entry!(
                "PopIoRingCompletion",
                unsafe extern "system" fn(HIORING, *mut IORING_CQE) -> i32
            )?,
            supported: entry!(
                "IsIoRingOpSupported",
                unsafe extern "system" fn(HIORING, IORING_OP_CODE) -> i32
            )?,
            build_read: entry!(
                "BuildIoRingReadFile",
                unsafe extern "system" fn(
                    HIORING,
                    IORING_HANDLE_REF,
                    IORING_BUFFER_REF,
                    u32,
                    u64,
                    usize,
                    IORING_SQE_FLAGS,
                ) -> i32
            )?,
            build_write: entry!(
                "BuildIoRingWriteFile",
                unsafe extern "system" fn(
                    HIORING,
                    IORING_HANDLE_REF,
                    IORING_BUFFER_REF,
                    u32,
                    u64,
                    FILE_WRITE_FLAGS,
                    usize,
                    IORING_SQE_FLAGS,
                ) -> i32
            ),
        })
    }
}

/// The Winsock extension functions, taken from the first socket that needs them.
struct Extensions {
    accept: unsafe extern "system" fn(
        SOCKET,
        SOCKET,
        *mut c_void,
        u32,
        u32,
        u32,
        *mut u32,
        *mut OVERLAPPED,
    ) -> i32,
    connect: unsafe extern "system" fn(
        SOCKET,
        *const SOCKADDR,
        i32,
        *const c_void,
        u32,
        *mut u32,
        *mut OVERLAPPED,
    ) -> i32,
}

/// The IoRing entry points, found once for the process.
static RING: OnceLock<Option<RingApi>> = OnceLock::new();
/// The ring version the probe created, for the rings the engine creates.
static RING_VERSION: AtomicI32 = AtomicI32::new(IORING_VERSION_1);
/// The Winsock extension functions, found once for the process.
static EXTENSIONS: OnceLock<Extensions> = OnceLock::new();

pub(in crate::io) struct RealSys;

impl RealSys {
    pub(in crate::io) fn new() -> Self {
        Self
    }

    fn ring_api(&self) -> Option<&'static RingApi> {
        RING.get_or_init(RingApi::load).as_ref()
    }

    fn extensions(&self, sock: Handle) -> io::Result<&'static Extensions> {
        if let Some(extensions) = EXTENSIONS.get() {
            return Ok(extensions);
        }
        let accept = extension_pointer(sock, &WSAID_ACCEPTEX)?;
        let connect = extension_pointer(sock, &WSAID_CONNECTEX)?;
        // SAFETY: Winsock returns these as `LPFN_ACCEPTEX` and `LPFN_CONNECTEX`, pointers of the
        // size of `usize`; they were checked for null.
        let (accept, connect) = unsafe {
            (
                std::mem::transmute::<usize, LPFN_ACCEPTEX>(accept),
                std::mem::transmute::<usize, LPFN_CONNECTEX>(connect),
            )
        };
        let (Some(accept), Some(connect)) = (accept, connect) else {
            return Err(io::Error::other("Winsock has no AcceptEx or ConnectEx"));
        };
        Ok(EXTENSIONS.get_or_init(|| Extensions { accept, connect }))
    }
}

/// A Winsock extension function's address for `sock`.
fn extension_pointer(sock: Handle, guid: &windows_sys::core::GUID) -> io::Result<usize> {
    let mut address = 0usize;
    let mut returned = 0u32;
    // SAFETY: the input is the GUID, the output is a `usize` the call fills, both for the sizes
    // given; no overlapped structure, so the call finishes before it returns.
    let status = unsafe {
        WSAIoctl(
            sock as SOCKET,
            SIO_GET_EXTENSION_FUNCTION_POINTER,
            std::ptr::from_ref(guid).cast(),
            size_of::<windows_sys::core::GUID>() as u32,
            std::ptr::from_mut(&mut address).cast(),
            size_of::<usize>() as u32,
            &mut returned,
            null_mut(),
            None,
        )
    };
    if status != 0 {
        return Err(last_error());
    }
    if address == 0 {
        return Err(io::Error::other(
            "Winsock returned no address for an extension function",
        ));
    }
    Ok(address)
}

fn io_status(status: i32) -> io::Result<()> {
    if status == 0 {
        Ok(())
    } else {
        Err(last_error())
    }
}

impl Sys for RealSys {
    fn create_port(&self) -> io::Result<Handle> {
        // SAFETY: a new port: no file, no existing port.
        let port = unsafe { CreateIoCompletionPort(INVALID_HANDLE_VALUE, null_mut(), 0, 0) };
        if port.is_null() {
            return Err(last_error());
        }
        Ok(port as Handle)
    }

    fn close_port(&self, port: Handle) {
        // SAFETY: the port was made by `create_port` and is closed once, when its last user is
        // gone.
        unsafe { CloseHandle(port as HANDLE) };
    }

    fn associate(&self, port: Handle, handle: Handle) -> io::Result<()> {
        // SAFETY: joins an open handle to an open port; both stay open for the call.
        let joined = unsafe { CreateIoCompletionPort(handle as HANDLE, port as HANDLE, 0, 0) };
        if joined.is_null() {
            // A handle that is already on a port is refused with `ERROR_INVALID_PARAMETER`; the
            // engine decides whether that port is its own.
            Err(last_error())
        } else {
            Ok(())
        }
    }

    fn dequeue(&self, port: Handle, out: &mut Vec<Packet>, max: usize) -> io::Result<()> {
        const N: usize = 64;
        let mut entries = [OVERLAPPED_ENTRY::default(); N];
        let mut removed = 0u32;
        // SAFETY: `entries` has `N >= count` slots; a zero timeout, so the call never waits.
        let ok = unsafe {
            GetQueuedCompletionStatusEx(
                port as HANDLE,
                entries.as_mut_ptr(),
                max.min(N) as u32,
                &mut removed,
                0,
                0,
            )
        };
        if ok == 0 {
            let error = last_error();
            return if error.raw_os_error() == Some(WAIT_TIMEOUT as i32) {
                Ok(())
            } else {
                Err(error)
            };
        }
        out.extend(entries[..removed as usize].iter().map(|entry| Packet {
            overlapped: entry.lpOverlapped as usize,
            bytes: entry.dwNumberOfBytesTransferred,
        }));
        Ok(())
    }

    fn post(&self, port: Handle, overlapped: usize) -> io::Result<()> {
        // SAFETY: queues a packet carrying a pointer value the engine only compares.
        let ok = unsafe {
            PostQueuedCompletionStatus(port as HANDLE, 0, 0, overlapped as *const OVERLAPPED)
        };
        if ok == 0 { Err(last_error()) } else { Ok(()) }
    }

    fn handle_mode(&self, handle: Handle, socket: bool) -> io::Result<HandleMode> {
        let mut status = IO_STATUS_BLOCK::default();
        let mut mode = FILE_MODE_INFORMATION { Mode: 0 };
        // SAFETY: an out structure of the size given, for the information class that fills it.
        let nt = unsafe {
            NtQueryInformationFile(
                handle as HANDLE,
                &mut status,
                std::ptr::from_mut(&mut mode).cast(),
                size_of::<FILE_MODE_INFORMATION>() as u32,
                FileModeInformation,
            )
        };
        if nt < 0 {
            // A socket is a file object too, but not every kind answers: read it as the default.
            if socket {
                return Ok(HandleMode::default());
            }
            return Err(io::Error::other(format!(
                "NtQueryInformationFile failed (NTSTATUS {nt:#010x})"
            )));
        }
        // The completion-notification flags (`FILE_SKIP_COMPLETION_PORT_ON_SUCCESS` is bit 0). A
        // failed query reads as "not set".
        let mut flags = 0u32;
        // SAFETY: an out `u32`, the size of `FILE_IO_COMPLETION_NOTIFICATION_INFORMATION`.
        let nt = unsafe {
            NtQueryInformationFile(
                handle as HANDLE,
                &mut status,
                std::ptr::from_mut(&mut flags).cast(),
                size_of::<u32>() as u32,
                FileIoCompletionNotificationInformation,
            )
        };
        Ok(HandleMode {
            synchronous: mode.Mode & (FILE_SYNCHRONOUS_IO_ALERT | FILE_SYNCHRONOUS_IO_NONALERT)
                != 0,
            skips_port_on_success: nt >= 0 && flags & 1 != 0,
        })
    }

    fn status_error(&self, status: i32, socket: bool) -> io::Error {
        if socket && let Some(code) = socket_status_error(status) {
            return io::Error::from_raw_os_error(code);
        }
        // SAFETY: a pure translation of a status code.
        let code = unsafe { RtlNtStatusToDosError(status) };
        io::Error::from_raw_os_error(code as i32)
    }

    unsafe fn read_file(
        &self,
        file: Handle,
        buf: *mut u8,
        len: u32,
        ov: *mut Overlapped,
    ) -> io::Result<()> {
        // SAFETY: by this method's contract.
        let ok = unsafe { ReadFile(file as HANDLE, buf, len, null_mut(), ov.cast()) };
        started(ok != 0, ov)
    }

    unsafe fn write_file(
        &self,
        file: Handle,
        buf: *const u8,
        len: u32,
        ov: *mut Overlapped,
    ) -> io::Result<()> {
        // SAFETY: by this method's contract.
        let ok = unsafe { WriteFile(file as HANDLE, buf, len, null_mut(), ov.cast()) };
        started(ok != 0, ov)
    }

    unsafe fn send(
        &self,
        sock: Handle,
        buf: *const u8,
        len: u32,
        ov: *mut Overlapped,
    ) -> io::Result<()> {
        let wsabuf = WSABUF {
            len,
            buf: buf.cast_mut(),
        };
        // SAFETY: by this method's contract; the call copies `wsabuf`.
        let status = unsafe { WSASend(sock as SOCKET, &wsabuf, 1, null_mut(), 0, ov.cast(), None) };
        started(status == 0, ov)
    }

    unsafe fn recv(
        &self,
        sock: Handle,
        buf: *mut u8,
        len: u32,
        ov: *mut Overlapped,
    ) -> io::Result<()> {
        let wsabuf = WSABUF { len, buf };
        let mut flags = 0u32;
        // SAFETY: by this method's contract; the call copies `wsabuf` and reads `flags` at once.
        let status = unsafe {
            WSARecv(
                sock as SOCKET,
                &wsabuf,
                1,
                null_mut(),
                &mut flags,
                ov.cast(),
                None,
            )
        };
        started(status == 0, ov)
    }

    unsafe fn send_to(
        &self,
        sock: Handle,
        buf: *const u8,
        len: u32,
        to: *const SockAddr,
        ov: *mut Overlapped,
    ) -> io::Result<()> {
        let wsabuf = WSABUF {
            len,
            buf: buf.cast_mut(),
        };
        // SAFETY: `to` is valid by this method's contract.
        let (address, address_len) = unsafe { ((*to).as_ptr(), (*to).len()) };
        // SAFETY: by this method's contract; the call copies `wsabuf`.
        let status = unsafe {
            WSASendTo(
                sock as SOCKET,
                &wsabuf,
                1,
                null_mut(),
                0,
                address.cast(),
                address_len,
                ov.cast(),
                None,
            )
        };
        started(status == 0, ov)
    }

    unsafe fn accept(
        &self,
        listener: Handle,
        accepted: Handle,
        addr_buf: *mut u8,
        ov: *mut Overlapped,
    ) -> io::Result<()> {
        let extensions = self.extensions(listener)?;
        let address_len = (ACCEPT_BUF / 2) as u32;
        let mut received = 0u32;
        // SAFETY: by this method's contract; no data is received into the buffer, which holds the
        // two addresses.
        let ok = unsafe {
            (extensions.accept)(
                listener as SOCKET,
                accepted as SOCKET,
                addr_buf.cast(),
                0,
                address_len,
                address_len,
                &mut received,
                ov.cast(),
            )
        };
        started(ok != 0, ov)
    }

    unsafe fn connect(
        &self,
        sock: Handle,
        to: *const SockAddr,
        ov: *mut Overlapped,
    ) -> io::Result<()> {
        let extensions = self.extensions(sock)?;
        // SAFETY: `to` is valid by this method's contract.
        let (address, address_len) = unsafe { ((*to).as_ptr(), (*to).len()) };
        // SAFETY: by this method's contract; no data is sent with the connect.
        let ok = unsafe {
            (extensions.connect)(
                sock as SOCKET,
                address.cast(),
                address_len,
                null(),
                0,
                null_mut(),
                ov.cast(),
            )
        };
        started(ok != 0, ov)
    }

    fn accept_socket(&self, listener: Handle) -> io::Result<Handle> {
        let mut storage = SOCKADDR_STORAGE::default();
        let mut len = size_of::<SOCKADDR_STORAGE>() as i32;
        // SAFETY: an out structure of the size given.
        let status = unsafe {
            getsockname(
                listener as SOCKET,
                std::ptr::from_mut(&mut storage).cast(),
                &mut len,
            )
        };
        io_status(status)?;
        let family = i32::from(storage.ss_family);
        debug_assert!(family == i32::from(AF_INET) || family == i32::from(AF_INET6));
        // SAFETY: a new socket: no protocol info, no group.
        let sock = unsafe {
            WSASocketW(
                family,
                SOCK_STREAM,
                IPPROTO_TCP,
                null(),
                0,
                WSA_FLAG_OVERLAPPED | WSA_FLAG_NO_HANDLE_INHERIT,
            )
        };
        if sock == INVALID_SOCKET {
            return Err(last_error());
        }
        Ok(sock)
    }

    fn close_socket(&self, sock: Handle) {
        // SAFETY: a socket this engine made and still owns.
        unsafe { closesocket(sock as SOCKET) };
    }

    fn bind_any(&self, sock: Handle, to: &SockAddr) -> io::Result<()> {
        // SAFETY: the address is valid for its length.
        let status = unsafe { bind(sock as SOCKET, to.as_ptr().cast(), to.len()) };
        if status == 0 {
            return Ok(());
        }
        let error = last_error();
        // Already bound.
        if error.raw_os_error() == Some(WSAEINVAL) {
            Ok(())
        } else {
            Err(error)
        }
    }

    fn update_accept(&self, accepted: Handle, listener: Handle) -> io::Result<()> {
        let listener = listener as SOCKET;
        // SAFETY: the option's value is the listening socket, of the size given.
        let status = unsafe {
            setsockopt(
                accepted as SOCKET,
                SOL_SOCKET,
                SO_UPDATE_ACCEPT_CONTEXT,
                std::ptr::from_ref(&listener).cast(),
                size_of::<SOCKET>() as i32,
            )
        };
        io_status(status)
    }

    fn update_connect(&self, sock: Handle) -> io::Result<()> {
        // SAFETY: this option takes no value.
        let status = unsafe {
            setsockopt(
                sock as SOCKET,
                SOL_SOCKET,
                SO_UPDATE_CONNECT_CONTEXT,
                null(),
                0,
            )
        };
        io_status(status)
    }

    fn open_file(&self, path: &Path, spec: OpenSpec) -> io::Result<Handle> {
        let path = wide(path)?;
        let access =
            if spec.read { GENERIC_READ } else { 0 } | if spec.write { GENERIC_WRITE } else { 0 };
        let disposition = match spec.create {
            Create::Never => OPEN_EXISTING,
            Create::IfMissing => OPEN_ALWAYS,
            Create::New => CREATE_NEW,
            Create::Truncate => CREATE_ALWAYS,
        };
        // SAFETY: a NUL-terminated wide path; default security; no template file.
        let handle = unsafe {
            CreateFileW(
                path.as_ptr(),
                access,
                FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                null(),
                disposition,
                FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED,
                null_mut(),
            )
        };
        if handle == INVALID_HANDLE_VALUE {
            return Err(last_error());
        }
        Ok(handle as Handle)
    }

    fn rename(&self, from: &Path, to: &Path) -> io::Result<()> {
        let (from, to) = (wide(from)?, wide(to)?);
        // SAFETY: NUL-terminated wide paths.
        let ok = unsafe {
            MoveFileExW(
                from.as_ptr(),
                to.as_ptr(),
                MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH,
            )
        };
        if ok == 0 { Err(last_error()) } else { Ok(()) }
    }

    fn run_on_pool(&self, job: Box<dyn FnOnce() + Send + 'static>) -> io::Result<()> {
        type Job = Box<dyn FnOnce() + Send + 'static>;

        unsafe extern "system" fn run(_instance: PTP_CALLBACK_INSTANCE, context: *mut c_void) {
            // SAFETY: `context` is the `Box<Job>` that `run_on_pool` leaked for this one call.
            let job: Job = *unsafe { Box::from_raw(context.cast::<Job>()) };
            // A panic must not unwind into the system's thread.
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(job));
        }

        let context = Box::into_raw(Box::new(job));
        // SAFETY: the callback takes the context back exactly once, if the call succeeds.
        let ok = unsafe { TrySubmitThreadpoolCallback(Some(run), context.cast(), null()) };
        if ok == 0 {
            let error = last_error();
            // SAFETY: the pool did not take the context: it is still ours.
            drop(unsafe { Box::from_raw(context) });
            return Err(error);
        }
        Ok(())
    }

    fn ring_probe(&self) -> Option<RingProbe> {
        let api = self.ring_api()?;
        let mut caps = IORING_CAPABILITIES::default();
        // SAFETY: an out structure.
        if unsafe { (api.query)(&mut caps) } < 0 {
            return None;
        }
        let emulated = caps.FeatureFlags & IORING_FEATURE_UM_EMULATION != 0;
        let version = caps.MaxVersion.min(WIN_IORING_VERSION_3);
        let flags = IORING_CREATE_FLAGS {
            Required: IORING_CREATE_REQUIRED_FLAGS_NONE,
            Advisory: IORING_CREATE_ADVISORY_FLAGS_NONE,
        };
        let mut ring: HIORING = null_mut();
        // SAFETY: an out handle, for a probe ring that is closed below.
        if unsafe { (api.create)(version, flags, 8, 16, &mut ring) } < 0 {
            return None;
        }
        // SAFETY: the probe ring just made.
        let read = unsafe { (api.supported)(ring, IORING_OP_READ) } != 0;
        // SAFETY: as above.
        let write = version >= WIN_IORING_VERSION_3
            && api.build_write.is_some()
            && unsafe { (api.supported)(ring, IORING_OP_WRITE) } != 0;
        // SAFETY: closing the probe ring, once.
        unsafe { (api.close)(ring) };
        RING_VERSION.store(version, Ordering::Release);
        Some(RingProbe {
            version: version as u32,
            emulated,
            read,
            write,
        })
    }

    fn ring_create(&self, entries: u32) -> io::Result<Handle> {
        let api = self
            .ring_api()
            .ok_or_else(|| io::Error::other("IoRing is not available"))?;
        let flags = IORING_CREATE_FLAGS {
            Required: IORING_CREATE_REQUIRED_FLAGS_NONE,
            Advisory: IORING_CREATE_ADVISORY_FLAGS_NONE,
        };
        let mut ring: HIORING = null_mut();
        // SAFETY: an out handle.
        let hr = unsafe {
            (api.create)(
                RING_VERSION.load(Ordering::Acquire),
                flags,
                entries,
                entries * 2,
                &mut ring,
            )
        };
        if hr < 0 {
            return Err(hresult_error(hr));
        }
        Ok(ring as Handle)
    }

    fn ring_close(&self, ring: Handle) {
        if let Some(api) = self.ring_api() {
            // SAFETY: a ring made by `ring_create`, closed once.
            unsafe { (api.close)(ring as HIORING) };
        }
    }

    unsafe fn ring_read(
        &self,
        ring: Handle,
        file: Handle,
        buf: *mut u8,
        len: u32,
        offset: u64,
        user_data: usize,
    ) -> Result<(), BuildError> {
        let api = self.ring_api().expect("a ring implies the API");
        // SAFETY: by this method's contract (the entry references the file and buffer raw).
        let hr = unsafe {
            (api.build_read)(
                ring as HIORING,
                handle_ref(file),
                buffer_ref(buf.cast()),
                len,
                offset,
                user_data,
                IOSQE_FLAGS_NONE,
            )
        };
        built(hr)
    }

    unsafe fn ring_write(
        &self,
        ring: Handle,
        file: Handle,
        buf: *const u8,
        len: u32,
        offset: u64,
        user_data: usize,
    ) -> Result<(), BuildError> {
        let api = self.ring_api().expect("a ring implies the API");
        let build = api
            .build_write
            .expect("a ring write implies the entry point");
        // SAFETY: by this method's contract.
        let hr = unsafe {
            build(
                ring as HIORING,
                handle_ref(file),
                buffer_ref(buf.cast_mut().cast()),
                len,
                offset,
                FILE_WRITE_FLAGS_NONE,
                user_data,
                IOSQE_FLAGS_NONE,
            )
        };
        built(hr)
    }

    fn ring_submit(&self, ring: Handle) -> io::Result<()> {
        let api = self.ring_api().expect("a ring implies the API");
        let mut submitted = 0u32;
        // SAFETY: a ring of this engine; no waiting (zero operations, zero milliseconds).
        let hr = unsafe { (api.submit)(ring as HIORING, 0, 0, &mut submitted) };
        if hr < 0 {
            Err(hresult_error(hr))
        } else {
            Ok(())
        }
    }

    fn ring_pop(&self, ring: Handle) -> io::Result<Option<Cqe>> {
        let api = self.ring_api().expect("a ring implies the API");
        let mut cqe = IORING_CQE::default();
        // SAFETY: an out structure; never waits.
        let hr = unsafe { (api.pop)(ring as HIORING, &mut cqe) };
        match hr {
            0 => Ok(Some(Cqe {
                user_data: cqe.UserData,
                result: cqe.ResultCode,
                info: cqe.Information,
            })),
            // S_FALSE: the completion queue is empty.
            1 => Ok(None),
            hr => Err(hresult_error(hr)),
        }
    }
}

fn handle_ref(file: Handle) -> IORING_HANDLE_REF {
    IORING_HANDLE_REF {
        Kind: IORING_REF_RAW,
        Handle: IORING_HANDLE_REF_0 {
            Handle: file as HANDLE,
        },
    }
}

fn buffer_ref(address: *mut c_void) -> IORING_BUFFER_REF {
    IORING_BUFFER_REF {
        Kind: IORING_REF_RAW,
        Buffer: IORING_BUFFER_REF_0 { Address: address },
    }
}

fn built(hr: i32) -> Result<(), BuildError> {
    if hr >= 0 {
        Ok(())
    } else if hr == IORING_E_SUBMISSION_QUEUE_FULL {
        Err(BuildError::Full)
    } else {
        Err(BuildError::Other(hresult_error(hr)))
    }
}
