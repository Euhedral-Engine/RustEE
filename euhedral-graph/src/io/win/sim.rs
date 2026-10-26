//! A simulated [`Sys`]: a scripted completion port, ring and file system, so that the engine's
//! state machines run on any platform.
//!
//! Every call does its work at once (copying into the caller's buffer, appending to a socket's
//! sent bytes), records its result under the `OVERLAPPED`'s address, and queues the packet (or
//! ring completion). `hold` keeps completions back until `release`, so a test chooses when they
//! arrive. Scripts: a cap on the bytes a transfer moves, errors for the next call of a kind (at
//! the call, or at its completion), the ring's capabilities and submission-queue size.

use std::collections::{HashMap, HashSet, VecDeque};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, PoisonError};

use super::super::{Create, OpenSpec};
use super::sys::{BuildError, Cqe, Handle, Overlapped, Packet, RingProbe, SockAddr, Sys};

/// The calls a script can make fail.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Op {
    Read,
    Write,
    Send,
    Recv,
    SendTo,
    Accept,
    Connect,
}

struct SimFile {
    data: Vec<u8>,
    synchronous: bool,
}

#[derive(Default)]
struct SimSock {
    sent: Vec<u8>,
    incoming: VecDeque<u8>,
}

#[derive(Clone, Copy)]
struct Built {
    write: bool,
    file: Handle,
    buf: usize,
    len: u32,
    offset: u64,
    user_data: usize,
}

#[derive(Default)]
struct SimRing {
    sq: Vec<Built>,
    cq: VecDeque<Cqe>,
}

#[derive(Default)]
struct State {
    next: usize,
    files: HashMap<Handle, SimFile>,
    socks: HashMap<Handle, SimSock>,
    fs: HashMap<PathBuf, Vec<u8>>,
    port: VecDeque<Packet>,
    results: HashMap<usize, io::Result<usize>>,
    hold: bool,
    held_packets: Vec<Packet>,
    held_cqes: Vec<(Handle, Cqe)>,
    cap: usize,
    fail_issue: VecDeque<(Op, io::Error)>,
    fail_result: VecDeque<(Op, io::Error)>,
    log: Vec<String>,
    probe: Option<RingProbe>,
    sq_limit: usize,
    rings: HashMap<Handle, SimRing>,
    associated: HashSet<Handle>,
    pool_fail: bool,
    closed_sockets: Vec<Handle>,
    fail_update: bool,
    fail_submits: u32,
}

pub(super) struct Sim {
    state: Mutex<State>,
}

fn hresult_for(error: &io::Error) -> i32 {
    match error.raw_os_error() {
        Some(code) => (0x8007_0000u32 | (code as u32 & 0xffff)) as i32,
        None => 0x8000_4005u32 as i32,
    }
}

impl Sim {
    pub(super) fn new() -> Self {
        Self {
            state: Mutex::new(State {
                next: 0x100,
                sq_limit: usize::MAX,
                ..State::default()
            }),
        }
    }

    fn s(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    // ---- scripting

    /// A file of `data`, opened for overlapped I/O.
    pub(super) fn file(&self, data: &[u8]) -> Handle {
        self.add_file(data, false)
    }

    /// A file of `data` opened without overlapped I/O.
    pub(super) fn synchronous_file(&self, data: &[u8]) -> Handle {
        self.add_file(data, true)
    }

    fn add_file(&self, data: &[u8], synchronous: bool) -> Handle {
        let mut s = self.s();
        s.next += 1;
        let handle = s.next;
        s.files.insert(
            handle,
            SimFile {
                data: data.to_vec(),
                synchronous,
            },
        );
        handle
    }

    pub(super) fn file_data(&self, file: Handle) -> Vec<u8> {
        self.s().files[&file].data.clone()
    }

    pub(super) fn sock(&self) -> Handle {
        let mut s = self.s();
        s.next += 1;
        let handle = s.next;
        s.socks.insert(handle, SimSock::default());
        handle
    }

    /// Bytes the peer of `sock` has sent, for a receive to find.
    pub(super) fn feed(&self, sock: Handle, bytes: &[u8]) {
        self.s()
            .socks
            .get_mut(&sock)
            .unwrap()
            .incoming
            .extend(bytes);
    }

    pub(super) fn sent(&self, sock: Handle) -> Vec<u8> {
        self.s().socks[&sock].sent.clone()
    }

    pub(super) fn put_file(&self, path: &Path, data: &[u8]) {
        self.s().fs.insert(path.to_owned(), data.to_vec());
    }

    pub(super) fn path_exists(&self, path: &Path) -> bool {
        self.s().fs.contains_key(path)
    }

    /// At most `bytes` move per read, write, send or receive (0: no limit).
    pub(super) fn cap_transfers(&self, bytes: usize) {
        self.s().cap = bytes;
    }

    pub(super) fn fail_next_call(&self, op: Op, error: io::Error) {
        self.s().fail_issue.push_back((op, error));
    }

    pub(super) fn fail_next_completion(&self, op: Op, error: io::Error) {
        self.s().fail_result.push_back((op, error));
    }

    /// Completions wait until [`release`](Self::release).
    pub(super) fn hold(&self, hold: bool) {
        self.s().hold = hold;
    }

    pub(super) fn release(&self) {
        let mut s = self.s();
        s.hold = false;
        let packets = std::mem::take(&mut s.held_packets);
        s.port.extend(packets);
        for (ring, cqe) in std::mem::take(&mut s.held_cqes) {
            s.rings.get_mut(&ring).unwrap().cq.push_back(cqe);
        }
    }

    pub(super) fn set_probe(&self, probe: Option<RingProbe>) {
        self.s().probe = probe;
    }

    /// A ring's submission queue holds at most `entries` built entries.
    pub(super) fn set_sq_limit(&self, entries: usize) {
        self.s().sq_limit = entries;
    }

    /// The next `count` ring submits fail (entries stay built, not taken).
    pub(super) fn fail_submits(&self, count: u32) {
        self.s().fail_submits = count;
    }

    pub(super) fn fail_pool(&self) {
        self.s().pool_fail = true;
    }

    pub(super) fn fail_context_updates(&self) {
        self.s().fail_update = true;
    }

    pub(super) fn log(&self) -> Vec<String> {
        self.s().log.clone()
    }

    pub(super) fn closed_sockets(&self) -> Vec<Handle> {
        self.s().closed_sockets.clone()
    }

    pub(super) fn rings_made(&self) -> usize {
        self.s().rings.len()
    }

    // ---- the work of a call

    /// Moves bytes for a read or write at `offset`: how many, or why not.
    fn file_op(
        s: &mut State,
        write: bool,
        file: Handle,
        buf: usize,
        len: u32,
        offset: u64,
    ) -> io::Result<usize> {
        let op = if write { Op::Write } else { Op::Read };
        let result = if let Some(at) = s.fail_result.iter().position(|(o, _)| *o == op) {
            Err(s.fail_result.remove(at).unwrap().1)
        } else {
            let cap = if s.cap == 0 { usize::MAX } else { s.cap };
            let len = (len as usize).min(cap);
            let offset = offset as usize;
            let Some(f) = s.files.get_mut(&file) else {
                return Err(io::Error::from_raw_os_error(6));
            };
            if write {
                if f.data.len() < offset + len {
                    f.data.resize(offset + len, 0);
                }
                // SAFETY: the engine's contract: `buf` is valid for `len` bytes of the record's
                // buffer until the completion is handled.
                let src = unsafe { std::slice::from_raw_parts(buf as *const u8, len) };
                f.data[offset..offset + len].copy_from_slice(src);
                Ok(len)
            } else if offset >= f.data.len() {
                // ERROR_HANDLE_EOF
                Err(io::Error::from_raw_os_error(38))
            } else {
                let n = len.min(f.data.len() - offset);
                // SAFETY: as above; `buf` is writable for `len >= n` bytes.
                unsafe {
                    std::ptr::copy_nonoverlapping(f.data[offset..].as_ptr(), buf as *mut u8, n)
                };
                Ok(n)
            }
        };
        s.log.push(format!(
            "{} {file:#x} at {offset} len {len}",
            if write { "write" } else { "read" }
        ));
        result
    }

    fn sock_op(s: &mut State, op: Op, sock: Handle, buf: usize, len: u32) -> io::Result<usize> {
        if let Some(at) = s.fail_result.iter().position(|(o, _)| *o == op) {
            return Err(s.fail_result.remove(at).unwrap().1);
        }
        let cap = if s.cap == 0 { usize::MAX } else { s.cap };
        let len = (len as usize).min(cap);
        let sock_state = s.socks.get_mut(&sock).expect("a simulated socket");
        let n = match op {
            Op::Send | Op::SendTo => {
                // SAFETY: the engine's contract: valid for `len` bytes until the completion.
                let src = unsafe { std::slice::from_raw_parts(buf as *const u8, len) };
                sock_state.sent.extend_from_slice(src);
                len
            }
            _ => {
                let n = len.min(sock_state.incoming.len());
                for i in 0..n {
                    // SAFETY: `buf` is writable for `len >= n` bytes.
                    unsafe { *(buf as *mut u8).add(i) = sock_state.incoming.pop_front().unwrap() };
                }
                n
            }
        };
        s.log.push(format!("{op:?} {sock:#x} len {len} -> {n}"));
        Ok(n)
    }

    fn finish(s: &mut State, ov: *mut Overlapped, result: io::Result<usize>) {
        let packet = Packet {
            overlapped: ov as usize,
        };
        s.results.insert(ov as usize, result);
        if s.hold {
            s.held_packets.push(packet);
        } else {
            s.port.push_back(packet);
        }
    }

    /// The failure a script asks of the next call of `op`.
    fn scripted(s: &mut State, op: Op) -> io::Result<()> {
        match s.fail_issue.iter().position(|(o, _)| *o == op) {
            Some(at) => Err(s.fail_issue.remove(at).unwrap().1),
            None => Ok(()),
        }
    }

    fn new_handle(s: &mut State) -> Handle {
        s.next += 1;
        s.next
    }
}

impl Sys for Sim {
    fn create_port(&self) -> io::Result<Handle> {
        let mut s = self.s();
        let port = Self::new_handle(&mut s);
        s.log.push(format!("create port {port:#x}"));
        Ok(port)
    }

    fn close_port(&self, port: Handle) {
        self.s().log.push(format!("close port {port:#x}"));
    }

    fn associate(&self, _port: Handle, handle: Handle) -> io::Result<()> {
        let mut s = self.s();
        if s.associated.insert(handle) {
            s.log.push(format!("associate {handle:#x}"));
        }
        Ok(())
    }

    fn dequeue(&self, _port: Handle, out: &mut Vec<Packet>, max: usize) -> io::Result<()> {
        let mut s = self.s();
        for _ in 0..max {
            let Some(packet) = s.port.pop_front() else {
                break;
            };
            out.push(packet);
        }
        Ok(())
    }

    fn post(&self, _port: Handle, overlapped: usize) -> io::Result<()> {
        self.s().port.push_back(Packet { overlapped });
        Ok(())
    }

    fn is_synchronous(&self, file: Handle) -> io::Result<bool> {
        match self.s().files.get(&file) {
            Some(f) => Ok(f.synchronous),
            None => Err(io::Error::from_raw_os_error(6)),
        }
    }

    unsafe fn read_file(
        &self,
        file: Handle,
        buf: *mut u8,
        len: u32,
        ov: *mut Overlapped,
    ) -> io::Result<()> {
        let mut s = self.s();
        Self::scripted(&mut s, Op::Read)?;
        // SAFETY: `ov` is valid by the trait's contract.
        let offset = unsafe { u64::from((*ov).offset) | (u64::from((*ov).offset_high) << 32) };
        let result = Self::file_op(&mut s, false, file, buf as usize, len, offset);
        Self::finish(&mut s, ov, result);
        Ok(())
    }

    unsafe fn write_file(
        &self,
        file: Handle,
        buf: *const u8,
        len: u32,
        ov: *mut Overlapped,
    ) -> io::Result<()> {
        let mut s = self.s();
        Self::scripted(&mut s, Op::Write)?;
        // SAFETY: `ov` is valid by the trait's contract.
        let offset = unsafe { u64::from((*ov).offset) | (u64::from((*ov).offset_high) << 32) };
        let result = Self::file_op(&mut s, true, file, buf as usize, len, offset);
        Self::finish(&mut s, ov, result);
        Ok(())
    }

    unsafe fn send(
        &self,
        sock: Handle,
        buf: *const u8,
        len: u32,
        ov: *mut Overlapped,
    ) -> io::Result<()> {
        let mut s = self.s();
        Self::scripted(&mut s, Op::Send)?;
        let result = Self::sock_op(&mut s, Op::Send, sock, buf as usize, len);
        Self::finish(&mut s, ov, result);
        Ok(())
    }

    unsafe fn recv(
        &self,
        sock: Handle,
        buf: *mut u8,
        len: u32,
        ov: *mut Overlapped,
    ) -> io::Result<()> {
        let mut s = self.s();
        Self::scripted(&mut s, Op::Recv)?;
        let result = Self::sock_op(&mut s, Op::Recv, sock, buf as usize, len);
        Self::finish(&mut s, ov, result);
        Ok(())
    }

    unsafe fn send_to(
        &self,
        sock: Handle,
        buf: *const u8,
        len: u32,
        to: *const SockAddr,
        ov: *mut Overlapped,
    ) -> io::Result<()> {
        let mut s = self.s();
        Self::scripted(&mut s, Op::SendTo)?;
        // SAFETY: `to` is valid by the trait's contract.
        let family = unsafe { (*to).family() };
        s.log.push(format!("send_to family {family}"));
        let result = Self::sock_op(&mut s, Op::SendTo, sock, buf as usize, len);
        Self::finish(&mut s, ov, result);
        Ok(())
    }

    unsafe fn accept(
        &self,
        listener: Handle,
        accepted: Handle,
        _addr_buf: *mut u8,
        ov: *mut Overlapped,
    ) -> io::Result<()> {
        let mut s = self.s();
        Self::scripted(&mut s, Op::Accept)?;
        s.log
            .push(format!("accept on {listener:#x} into {accepted:#x}"));
        let result = match s.fail_result.iter().position(|(o, _)| *o == Op::Accept) {
            Some(at) => Err(s.fail_result.remove(at).unwrap().1),
            None => Ok(0),
        };
        Self::finish(&mut s, ov, result);
        Ok(())
    }

    unsafe fn connect(
        &self,
        sock: Handle,
        _to: *const SockAddr,
        ov: *mut Overlapped,
    ) -> io::Result<()> {
        let mut s = self.s();
        Self::scripted(&mut s, Op::Connect)?;
        s.log.push(format!("connect {sock:#x}"));
        let result = match s.fail_result.iter().position(|(o, _)| *o == Op::Connect) {
            Some(at) => Err(s.fail_result.remove(at).unwrap().1),
            None => Ok(0),
        };
        Self::finish(&mut s, ov, result);
        Ok(())
    }

    unsafe fn result(
        &self,
        _handle: Handle,
        ov: *const Overlapped,
        _socket: bool,
    ) -> io::Result<usize> {
        self.s()
            .results
            .remove(&(ov as usize))
            .expect("a result for the finished call")
    }

    fn accept_socket(&self, listener: Handle) -> io::Result<Handle> {
        let mut s = self.s();
        let sock = Self::new_handle(&mut s);
        s.socks.insert(sock, SimSock::default());
        s.log
            .push(format!("accept socket {sock:#x} for {listener:#x}"));
        Ok(sock)
    }

    fn close_socket(&self, sock: Handle) {
        let mut s = self.s();
        s.closed_sockets.push(sock);
        s.log.push(format!("close socket {sock:#x}"));
    }

    fn bind_any(&self, sock: Handle, to: &SockAddr) -> io::Result<()> {
        self.s()
            .log
            .push(format!("bind {sock:#x} family {}", to.family()));
        Ok(())
    }

    fn update_accept(&self, accepted: Handle, listener: Handle) -> io::Result<()> {
        let mut s = self.s();
        s.log
            .push(format!("update accept {accepted:#x} from {listener:#x}"));
        if s.fail_update {
            return Err(io::Error::from_raw_os_error(10038));
        }
        Ok(())
    }

    fn update_connect(&self, sock: Handle) -> io::Result<()> {
        let mut s = self.s();
        s.log.push(format!("update connect {sock:#x}"));
        if s.fail_update {
            return Err(io::Error::from_raw_os_error(10038));
        }
        Ok(())
    }

    fn open_file(&self, path: &Path, spec: OpenSpec) -> io::Result<Handle> {
        let mut s = self.s();
        s.log.push(format!("open {}", path.display()));
        let exists = s.fs.contains_key(path);
        match (spec.create, exists) {
            (Create::Never, false) => return Err(io::Error::from_raw_os_error(2)),
            (Create::New, true) => return Err(io::Error::from_raw_os_error(80)),
            (Create::IfMissing | Create::New, false) | (Create::Truncate, _) => {
                s.fs.insert(path.to_owned(), Vec::new());
            }
            _ => {}
        }
        let data = s.fs[path].clone();
        let handle = Self::new_handle(&mut s);
        s.files.insert(
            handle,
            SimFile {
                data,
                synchronous: false,
            },
        );
        Ok(handle)
    }

    fn rename(&self, from: &Path, to: &Path) -> io::Result<()> {
        let mut s = self.s();
        s.log
            .push(format!("rename {} {}", from.display(), to.display()));
        match s.fs.remove(from) {
            Some(data) => {
                s.fs.insert(to.to_owned(), data);
                Ok(())
            }
            None => Err(io::Error::from_raw_os_error(2)),
        }
    }

    fn run_on_pool(&self, job: Box<dyn FnOnce() + Send + 'static>) -> io::Result<()> {
        if self.s().pool_fail {
            return Err(io::Error::from_raw_os_error(8));
        }
        // The pool's thread, here and now (the lock is not held while the job calls back).
        job();
        Ok(())
    }

    fn ring_probe(&self) -> Option<RingProbe> {
        self.s().probe
    }

    fn ring_create(&self, entries: u32) -> io::Result<Handle> {
        let mut s = self.s();
        let ring = Self::new_handle(&mut s);
        s.rings.insert(ring, SimRing::default());
        s.log.push(format!("ring {ring:#x} entries {entries}"));
        Ok(ring)
    }

    fn ring_close(&self, ring: Handle) {
        self.s().log.push(format!("close ring {ring:#x}"));
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
        self.build(ring, false, file, buf as usize, len, offset, user_data)
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
        self.build(ring, true, file, buf as usize, len, offset, user_data)
    }

    fn ring_submit(&self, ring: Handle) -> io::Result<()> {
        let mut s = self.s();
        if s.fail_submits > 0 {
            s.fail_submits -= 1;
            return Err(io::Error::from_raw_os_error(5));
        }
        let built = std::mem::take(&mut s.rings.get_mut(&ring).unwrap().sq);
        for entry in built {
            let result = Self::file_op(
                &mut s,
                entry.write,
                entry.file,
                entry.buf,
                entry.len,
                entry.offset,
            );
            let cqe = match result {
                Ok(n) => Cqe {
                    user_data: entry.user_data,
                    result: 0,
                    info: n,
                },
                Err(error) => Cqe {
                    user_data: entry.user_data,
                    result: hresult_for(&error),
                    info: 0,
                },
            };
            if s.hold {
                s.held_cqes.push((ring, cqe));
            } else {
                s.rings.get_mut(&ring).unwrap().cq.push_back(cqe);
            }
        }
        Ok(())
    }

    fn ring_pop(&self, ring: Handle) -> io::Result<Option<Cqe>> {
        Ok(self.s().rings.get_mut(&ring).unwrap().cq.pop_front())
    }
}

impl Sim {
    #[allow(clippy::too_many_arguments)]
    fn build(
        &self,
        ring: Handle,
        write: bool,
        file: Handle,
        buf: usize,
        len: u32,
        offset: u64,
        user_data: usize,
    ) -> Result<(), BuildError> {
        let mut s = self.s();
        let limit = s.sq_limit;
        let ring = s.rings.get_mut(&ring).unwrap();
        if ring.sq.len() >= limit {
            return Err(BuildError::Full);
        }
        ring.sq.push(Built {
            write,
            file,
            buf,
            len,
            offset,
            user_data,
        });
        Ok(())
    }
}
