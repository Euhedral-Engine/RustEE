//! The seam between the Windows engine and the Windows calls.
//!
//! The engine never calls Windows itself: every system call goes through [`Sys`]. [`RealSys`]
//! (Windows only) makes the calls with `windows-sys`; the tests on every platform use a simulated
//! `Sys` that scripts a completion port and a ring. Handles are plain `usize` values, so the
//! engine and its simulated tests build on any platform.

use std::io;
use std::net::SocketAddr;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::super::OpenSpec;

/// A Windows `HANDLE` or `SOCKET`, as its value.
pub(crate) type Handle = usize;

/// `IORING_VERSION_3`: the version that adds file writes.
pub(crate) const IORING_VERSION_3: u32 = 300;

/// The layout of `OVERLAPPED`: the system writes the result into it, and a dequeued packet leads
/// back to it.
#[repr(C)]
#[derive(Debug, Default)]
pub(crate) struct Overlapped {
    pub internal: usize,
    pub internal_high: usize,
    pub offset: u32,
    pub offset_high: u32,
    pub event: usize,
}

impl Overlapped {
    /// `Internal` (the call's `NTSTATUS`, `STATUS_PENDING` until it completes), loaded as an
    /// atomic: the kernel writes it.
    ///
    /// # Safety
    /// `ov` points at a valid `OVERLAPPED`.
    pub(crate) unsafe fn load_internal(ov: *const Self) -> usize {
        // SAFETY: by this function's contract; `usize` and `AtomicUsize` have the same layout.
        unsafe {
            AtomicUsize::from_ptr((&raw const (*ov).internal).cast_mut()).load(Ordering::Acquire)
        }
    }

    /// `InternalHigh` (the bytes the call moved), loaded as an atomic.
    ///
    /// # Safety
    /// As [`load_internal`](Self::load_internal).
    pub(crate) unsafe fn load_internal_high(ov: *const Self) -> usize {
        // SAFETY: by this function's contract.
        unsafe {
            AtomicUsize::from_ptr((&raw const (*ov).internal_high).cast_mut())
                .load(Ordering::Acquire)
        }
    }

    /// The kernel's writes of a completed call (for the simulated `Sys`).
    ///
    /// # Safety
    /// As [`load_internal`](Self::load_internal), and no other thread reads it at once.
    #[cfg(test)]
    pub(crate) unsafe fn store_result(ov: *mut Self, status: usize, bytes: usize) {
        // SAFETY: by this function's contract.
        unsafe {
            AtomicUsize::from_ptr(&raw mut (*ov).internal_high).store(bytes, Ordering::Release);
            AtomicUsize::from_ptr(&raw mut (*ov).internal).store(status, Ordering::Release);
        }
    }

    /// Cleared for a new operation at `offset` (for handles that have a file position).
    pub(crate) fn reset(&mut self, offset: u64) {
        *self = Self {
            offset: offset as u32,
            offset_high: (offset >> 32) as u32,
            ..Self::default()
        };
    }
}

/// A socket address in the layout of `SOCKADDR_STORAGE` (128 bytes, 8-aligned).
#[repr(C, align(8))]
pub(crate) struct SockAddr {
    bytes: [u8; 128],
    len: i32,
}

impl SockAddr {
    pub(crate) fn new(to: SocketAddr) -> Self {
        let mut bytes = [0u8; 128];
        let len = match to {
            SocketAddr::V4(v4) => {
                // sockaddr_in: family, port (network order), address, 8 bytes of zeros.
                bytes[0..2].copy_from_slice(&2u16.to_ne_bytes());
                bytes[2..4].copy_from_slice(&v4.port().to_be_bytes());
                bytes[4..8].copy_from_slice(&v4.ip().octets());
                16
            }
            SocketAddr::V6(v6) => {
                // sockaddr_in6: family, port, flow info, address, scope id.
                bytes[0..2].copy_from_slice(&23u16.to_ne_bytes());
                bytes[2..4].copy_from_slice(&v6.port().to_be_bytes());
                bytes[4..8].copy_from_slice(&v6.flowinfo().to_ne_bytes());
                bytes[8..24].copy_from_slice(&v6.ip().octets());
                bytes[24..28].copy_from_slice(&v6.scope_id().to_ne_bytes());
                28
            }
        };
        Self { bytes, len }
    }

    /// The unspecified address (port 0) of `like`'s family, to bind a socket before `ConnectEx`.
    pub(crate) fn unspecified(like: SocketAddr) -> Self {
        let any = match like {
            SocketAddr::V4(_) => SocketAddr::from(([0u8; 4], 0)),
            SocketAddr::V6(_) => SocketAddr::from(([0u8; 16], 0)),
        };
        Self::new(any)
    }

    pub(crate) fn as_ptr(&self) -> *const u8 {
        self.bytes.as_ptr()
    }

    pub(crate) fn len(&self) -> i32 {
        self.len
    }

    #[cfg(test)]
    pub(crate) fn family(&self) -> u16 {
        u16::from_ne_bytes([self.bytes[0], self.bytes[1]])
    }
}

/// `STATUS_PENDING`: the value an `OVERLAPPED`'s `Internal` holds until its call completes.
pub(crate) const STATUS_PENDING: i32 = 0x103;
/// `STATUS_END_OF_FILE`.
pub(crate) const STATUS_END_OF_FILE: i32 = 0xC000_0011_u32 as i32;

/// Whether an `NTSTATUS` is a warning (severity 2): the I/O manager still queues a completion
/// packet for a call that returns one, although the call reports failure.
pub(crate) fn is_nt_warning(status: i32) -> bool {
    (status as u32) >> 30 == 2
}

/// A packet dequeued from a completion port: the `OVERLAPPED` it carries and the byte count it
/// reports. The call's status and bytes are read from the `OVERLAPPED` (`Internal` and
/// `InternalHigh`, which are documented), and the packet's count is only checked against them.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Packet {
    /// The address of the `OVERLAPPED` the packet carries.
    pub overlapped: usize,
    pub bytes: u32,
}

/// How a handle makes calls behave.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct HandleMode {
    /// Opened without overlapped I/O: its calls block the caller.
    pub synchronous: bool,
    /// "Skip the port on success" is set: a call that completes at once queues no packet.
    pub skips_port_on_success: bool,
}

/// What the OS says about its kernel ring (IoRing).
#[derive(Clone, Copy, Debug)]
pub(crate) struct RingProbe {
    /// The highest ring version it offers (`IORING_VERSION_3` is 300).
    pub version: u32,
    /// The "ring" is emulated in user mode: it has no kernel queue behind it.
    pub emulated: bool,
    /// File reads are supported.
    pub read: bool,
    /// File writes are supported.
    pub write: bool,
}

/// A popped ring completion.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Cqe {
    pub user_data: usize,
    /// An `HRESULT`.
    pub result: i32,
    /// The bytes moved.
    pub info: usize,
}

/// Why a ring entry was not built.
#[derive(Debug)]
pub(crate) enum BuildError {
    /// The submission queue is full: try again after a submit.
    Full,
    Other(io::Error),
}

/// Every Windows call of the engine. A method that takes a pointer is `unsafe`: the caller
/// guarantees what its safety comment says.
///
/// An asynchronous call returns `Ok(())` when a completion packet (or ring completion) will
/// follow, and `Err` when the call failed at once and none will. A call that reports failure with
/// a warning status (`ERROR_MORE_DATA`, `WSAEMSGSIZE`) still queues a packet, so it returns `Ok`.
/// The engine does not complete an operation until its call has returned, so the implementation
/// may read the `OVERLAPPED` after the call to tell the two apart.
pub(crate) trait Sys: Send + Sync + 'static {
    // ---- the completion port
    fn create_port(&self) -> io::Result<Handle>;
    fn close_port(&self, port: Handle);
    /// Associates `handle` with `port`; a handle that is already associated is fine.
    fn associate(&self, port: Handle, handle: Handle) -> io::Result<()>;
    /// Appends up to `max` packets to `out`, taking none and never waiting if there are none.
    fn dequeue(&self, port: Handle, out: &mut Vec<Packet>, max: usize) -> io::Result<()>;
    /// Queues a packet that carries `overlapped` (tests).
    #[cfg_attr(not(test), allow(dead_code))]
    fn post(&self, port: Handle, overlapped: usize) -> io::Result<()>;
    /// How the handle (a file, or with `socket` a socket) makes calls behave. A failed query of a
    /// socket is not an error: it reads as the default mode.
    fn handle_mode(&self, handle: Handle, socket: bool) -> io::Result<HandleMode>;
    /// An `NTSTATUS` of a finished call as an error (Win32 codes; Winsock codes for `socket`).
    fn status_error(&self, status: i32, socket: bool) -> io::Error;

    // ---- overlapped calls
    /// # Safety
    /// `file` is open for overlapped I/O and associated with the port; `buf` is valid for `len`
    /// bytes and `ov` is valid, both until the packet is dequeued (or the call failed).
    unsafe fn read_file(
        &self,
        file: Handle,
        buf: *mut u8,
        len: u32,
        ov: *mut Overlapped,
    ) -> io::Result<()>;
    /// # Safety
    /// As [`read_file`](Self::read_file).
    unsafe fn write_file(
        &self,
        file: Handle,
        buf: *const u8,
        len: u32,
        ov: *mut Overlapped,
    ) -> io::Result<()>;
    /// # Safety
    /// As [`read_file`](Self::read_file), for a connected socket.
    unsafe fn send(
        &self,
        sock: Handle,
        buf: *const u8,
        len: u32,
        ov: *mut Overlapped,
    ) -> io::Result<()>;
    /// # Safety
    /// As [`read_file`](Self::read_file), for a connected socket.
    unsafe fn recv(
        &self,
        sock: Handle,
        buf: *mut u8,
        len: u32,
        ov: *mut Overlapped,
    ) -> io::Result<()>;
    /// # Safety
    /// As [`read_file`](Self::read_file), and `to` is valid until the packet is dequeued.
    unsafe fn send_to(
        &self,
        sock: Handle,
        buf: *const u8,
        len: u32,
        to: *const SockAddr,
        ov: *mut Overlapped,
    ) -> io::Result<()>;
    /// `AcceptEx` into `accepted`, with `addr_buf` ([`ACCEPT_BUF`] bytes) for the two addresses.
    ///
    /// # Safety
    /// As [`read_file`](Self::read_file), for a listening socket; `accepted` is a fresh socket
    /// and `addr_buf` is valid for [`ACCEPT_BUF`] bytes until the packet is dequeued.
    unsafe fn accept(
        &self,
        listener: Handle,
        accepted: Handle,
        addr_buf: *mut u8,
        ov: *mut Overlapped,
    ) -> io::Result<()>;
    /// `ConnectEx` on a bound socket.
    ///
    /// # Safety
    /// As [`read_file`](Self::read_file), and `to` is valid until the packet is dequeued.
    unsafe fn connect(
        &self,
        sock: Handle,
        to: *const SockAddr,
        ov: *mut Overlapped,
    ) -> io::Result<()>;
    // ---- sockets
    /// A new overlapped, non-inheritable stream socket of `listener`'s family.
    fn accept_socket(&self, listener: Handle) -> io::Result<Handle>;
    fn close_socket(&self, sock: Handle);
    /// Binds `sock` to the unspecified address of `to`'s family, if it is not bound.
    fn bind_any(&self, sock: Handle, to: &SockAddr) -> io::Result<()>;
    fn update_accept(&self, accepted: Handle, listener: Handle) -> io::Result<()>;
    fn update_connect(&self, sock: Handle) -> io::Result<()>;

    // ---- files, on the OS thread pool
    /// `CreateFileW`, for overlapped I/O. Blocks: the engine calls it on the pool.
    fn open_file(&self, path: &Path, spec: OpenSpec) -> io::Result<Handle>;
    /// `MoveFileExW`, replacing. Blocks: the engine calls it on the pool.
    fn rename(&self, from: &Path, to: &Path) -> io::Result<()>;
    /// Runs `job` once on the OS thread pool.
    fn run_on_pool(&self, job: Box<dyn FnOnce() + Send + 'static>) -> io::Result<()>;

    // ---- the kernel ring
    /// What the OS offers, or `None` where there is no usable IoRing.
    fn ring_probe(&self) -> Option<RingProbe>;
    /// A ring with `entries` submission entries (and twice as many completions).
    fn ring_create(&self, entries: u32) -> io::Result<Handle>;
    fn ring_close(&self, ring: Handle);
    /// Builds a read entry.
    ///
    /// # Safety
    /// `file` is open for overlapped I/O; `buf` is valid for `len` bytes until the completion is
    /// popped.
    unsafe fn ring_read(
        &self,
        ring: Handle,
        file: Handle,
        buf: *mut u8,
        len: u32,
        offset: u64,
        user_data: usize,
    ) -> Result<(), BuildError>;
    /// Builds a write entry.
    ///
    /// # Safety
    /// As [`ring_read`](Self::ring_read).
    unsafe fn ring_write(
        &self,
        ring: Handle,
        file: Handle,
        buf: *const u8,
        len: u32,
        offset: u64,
        user_data: usize,
    ) -> Result<(), BuildError>;
    /// Hands the built entries to the kernel; never waits.
    fn ring_submit(&self, ring: Handle) -> io::Result<()>;
    /// The next completion, if there is one.
    fn ring_pop(&self, ring: Handle) -> io::Result<Option<Cqe>>;
}

/// The bytes `AcceptEx` needs for the local and remote addresses: two of (the largest address
/// plus 16).
pub(crate) const ACCEPT_BUF: usize = 2 * (128 + 16);

/// The `HRESULT` of an IoRing completion as an error: a Win32 failure keeps its code.
pub(crate) fn hresult_error(hr: i32) -> io::Error {
    const FACILITY_WIN32: u32 = 7;
    let bits = hr as u32;
    if (bits >> 16) & 0x1fff == FACILITY_WIN32 {
        io::Error::from_raw_os_error((bits & 0xffff) as i32)
    } else {
        io::Error::other(format!("an IoRing operation failed (HRESULT {bits:#010x})"))
    }
}

/// A path as a verbatim path (`\\?\`), for a path too long for the ordinary Win32 limit. `path`
/// is absolute and normalized (no `.` or `..`, backslashes only); anything else is `None`.
pub(crate) fn verbatim_path(path: &str) -> Option<String> {
    if path.starts_with(r"\\?\") || path.starts_with(r"\\.\") {
        return None;
    }
    if let Some(unc) = path.strip_prefix(r"\\") {
        return Some(format!(r"\\?\UNC\{unc}"));
    }
    let bytes = path.as_bytes();
    if bytes.len() >= 3 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':' && bytes[2] == b'\\' {
        return Some(format!(r"\\?\{path}"));
    }
    None
}

/// The Winsock error of a socket call's `NTSTATUS`, for the statuses whose Win32 translation
/// differs from it (the table `libuv` keeps for the same reason). `None`: use the Win32 error.
pub(crate) fn socket_status_error(status: i32) -> Option<i32> {
    Some(match status as u32 {
        0xC000_0236 => 10061, // STATUS_CONNECTION_REFUSED: WSAECONNREFUSED
        0xC000_020D | 0xC000_013C | 0xC000_023F => 10054, // reset, remote disconnect, port unreachable: WSAECONNRESET
        0xC000_0241 => 10053, // STATUS_CONNECTION_ABORTED: WSAECONNABORTED
        0xC000_023D => 10065, // STATUS_HOST_UNREACHABLE: WSAEHOSTUNREACH
        0xC000_023C => 10051, // STATUS_NETWORK_UNREACHABLE: WSAENETUNREACH
        0xC000_00B5 => 10060, // STATUS_IO_TIMEOUT: WSAETIMEDOUT
        0x8000_0005 => 10040, // STATUS_BUFFER_OVERFLOW: WSAEMSGSIZE
        _ => return None,
    })
}

/// `ERROR_HANDLE_EOF` and `ERROR_BROKEN_PIPE`: a read at the end of the data.
pub(crate) fn is_end_of_file(error: &io::Error) -> bool {
    matches!(error.raw_os_error(), Some(38 | 109))
}
