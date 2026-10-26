//! The seam between the Windows engine and the Windows calls.
//!
//! The engine never calls Windows itself: every system call goes through [`Sys`]. [`RealSys`]
//! (Windows only) makes the calls with `windows-sys`; the tests on every platform use a simulated
//! `Sys` that scripts a completion port and a ring. Handles are plain `usize` values, so the
//! engine and its simulated tests build on any platform.

use std::io;
use std::net::SocketAddr;
use std::path::Path;

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

/// A packet dequeued from a completion port.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Packet {
    /// The address of the `OVERLAPPED` the packet carries.
    pub overlapped: usize,
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
/// follow, and `Err` when the call failed at once and none will.
pub(crate) trait Sys: Send + Sync + 'static {
    // ---- the completion port
    fn create_port(&self) -> io::Result<Handle>;
    fn close_port(&self, port: Handle);
    /// Associates `handle` with `port`; a handle that is already associated is fine.
    fn associate(&self, port: Handle, handle: Handle) -> io::Result<()>;
    /// Appends up to `max` packets to `out`, taking none and never waiting if there are none.
    fn dequeue(&self, port: Handle, out: &mut Vec<Packet>, max: usize) -> io::Result<()>;
    /// Queues a packet that carries `overlapped`.
    fn post(&self, port: Handle, overlapped: usize) -> io::Result<()>;
    /// Whether the file handle makes calls block (opened without overlapped I/O).
    fn is_synchronous(&self, file: Handle) -> io::Result<bool>;

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
    /// What a finished overlapped call moved, or why it failed.
    ///
    /// # Safety
    /// `ov` is the `OVERLAPPED` of a call on `handle` whose packet was dequeued.
    unsafe fn result(
        &self,
        handle: Handle,
        ov: *const Overlapped,
        socket: bool,
    ) -> io::Result<usize>;

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

/// `ERROR_HANDLE_EOF` and `ERROR_BROKEN_PIPE`: a read at the end of the data.
pub(crate) fn is_end_of_file(error: &io::Error) -> bool {
    matches!(error.raw_os_error(), Some(38 | 109))
}
