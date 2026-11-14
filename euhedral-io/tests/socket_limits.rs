//! Time limits and cancel for the socket operations of `AsyncIo`, through the real engine (io_uring),
//! over loopback.
//!
//! The tests drive the source by hand, as a lattice worker's polls do. A limit is the kernel's own
//! (a linked timeout), so a test that waits for one waits for a real clock, for a few tens of
//! milliseconds; nothing else in them depends on time. "Still in flight" is shown by reaping while
//! the peer is silent: no completion can come.
#![allow(
    clippy::disallowed_types,
    reason = "test fixtures: continuations record into a list the test thread reads"
)]
#![cfg(all(feature = "async-io", any(target_os = "linux", target_os = "android")))]
// Fixtures open sockets and run a peer thread outside any lattice.
#![allow(clippy::disallowed_methods)]

use std::io::{self, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::os::fd::{AsRawFd, OwnedFd};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use euhedral_core::frames::{Frame, FrameHeader};
use euhedral_core::generics::LatticeSource;
use euhedral_io::{AsyncIo, Cancelled, IoBuf, OpOptions};

/// The frame a continuation emits: it has no work, the outcome went to the list.
struct Ended(FrameHeader);

impl Frame for Ended {
    fn header(&self) -> &FrameHeader {
        &self.0
    }

    fn header_mut(&mut self) -> &mut FrameHeader {
        &mut self.0
    }
}

fn ended() -> Box<dyn Frame> {
    Box::new(Ended(FrameHeader::new(1)))
}

/// What the continuations saw, in order.
struct Outcomes<T>(Arc<Mutex<Vec<T>>>);

impl<T> Clone for Outcomes<T> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<T> Default for Outcomes<T> {
    fn default() -> Self {
        Self(Arc::default())
    }
}

impl<T> Outcomes<T> {
    fn len(&self) -> usize {
        self.0.lock().unwrap().len()
    }

    fn take(&self) -> Vec<T> {
        std::mem::take(&mut *self.0.lock().unwrap())
    }
}

/// What a send or receive ended with: the bytes it moved or its error, and the buffer's length (the
/// buffer came back), and how many bytes of it are filled.
type Io = (io::Result<usize>, usize, usize);

fn io_then(out: &Outcomes<Io>) -> euhedral_io::IoThen {
    let out = out.clone();
    Box::new(move |done| {
        out.0
            .lock()
            .unwrap()
            .push((done.result, done.buf.bytes.len(), done.buf.filled));
        ended()
    })
}

/// Reaps and hands out until `done` holds (10 s at most).
fn drive(io: &AsyncIo, mut done: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while !done() {
        assert!(Instant::now() < deadline, "the operations did not end");
        step(io);
    }
}

fn step(io: &AsyncIo) {
    io.reap();
    io.pull(&mut |_| {}, &mut |_| false, u64::MAX);
}

/// Reaps many times: nothing that is still waiting for a peer can end meanwhile.
fn stay_pending(io: &AsyncIo, out_len: impl Fn() -> usize) {
    for _ in 0..500 {
        step(io);
    }
    assert_eq!(out_len(), 0, "an operation ended that should still wait");
}

fn buf(len: usize) -> IoBuf {
    IoBuf {
        bytes: vec![0; len],
        filled: 0,
    }
}

fn pair() -> (TcpStream, TcpStream) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
    let (server, _) = listener.accept().unwrap();
    (client, server)
}

fn limit(ms: u64) -> OpOptions {
    OpOptions {
        limit: Some(Duration::from_millis(ms)),
    }
}

const NEVER: OpOptions = OpOptions {
    limit: Some(Duration::from_secs(3600)),
};

fn only(out: &Outcomes<Io>) -> Io {
    let mut all = out.take();
    assert_eq!(all.len(), 1, "an operation must end exactly once");
    all.remove(0)
}

fn recv_ok(io: &AsyncIo, socket: &TcpStream, expect: &[u8]) {
    let out = Outcomes::default();
    io.recv(socket.as_raw_fd(), buf(64), io_then(&out)).unwrap();
    drive(io, || out.len() == 1);
    let (result, _, filled) = only(&out);
    assert_eq!(result.unwrap(), expect.len());
    assert_eq!(filled, expect.len());
}

#[test]
fn a_receive_with_a_limit_on_a_silent_peer_times_out_and_the_socket_receives_afterwards() {
    let io = AsyncIo::open().unwrap();
    let (mut peer, ours) = pair();
    let out = Outcomes::default();
    io.recv_with(ours.as_raw_fd(), buf(64), limit(50), io_then(&out))
        .unwrap();
    assert_eq!(io.pending(), 1);
    drive(&io, || out.len() == 1);
    let (result, len, filled) = only(&out);
    assert_eq!(result.unwrap_err().kind(), io::ErrorKind::TimedOut);
    assert_eq!((len, filled), (64, 0));
    assert_eq!(io.pending(), 0);
    peer.write_all(b"hello").unwrap();
    recv_ok(&io, &ours, b"hello");
}

#[test]
fn a_limited_operation_that_finishes_in_time_ends_once_and_leaves_no_trace() {
    let io = AsyncIo::open().unwrap();
    let (mut peer, ours) = pair();
    // A receive with data waiting.
    peer.write_all(b"abc").unwrap();
    let out = Outcomes::default();
    io.recv_with(ours.as_raw_fd(), buf(64), NEVER, io_then(&out))
        .unwrap();
    drive(&io, || out.len() == 1);
    assert_eq!(only(&out).0.unwrap(), 3);
    // A send, in several pieces (the limit binds the whole), to a reader.
    let total = 3 * 1024 * 1024 + 17;
    let reader = std::thread::spawn(move || {
        let mut got = 0;
        let mut chunk = vec![0u8; 64 * 1024];
        while got < total {
            let n = peer.read(&mut chunk).unwrap();
            assert!(n > 0);
            got += n;
        }
        peer
    });
    let out = Outcomes::default();
    io.send_with(
        ours.as_raw_fd(),
        IoBuf {
            bytes: vec![7; total],
            filled: total,
        },
        NEVER,
        io_then(&out),
    )
    .unwrap();
    drive(&io, || out.len() == 1);
    assert_eq!(only(&out).0.unwrap(), total);
    let mut peer = reader.join().unwrap();
    // An accept with a peer connecting, and a connect to a listener.
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let to = listener.local_addr().unwrap();
    let accepted: Outcomes<io::Result<OwnedFd>> = Outcomes::default();
    let sink = accepted.clone();
    io.accept_with(
        listener.as_raw_fd(),
        NEVER,
        Box::new(move |result| {
            sink.0.lock().unwrap().push(result);
            ended()
        }),
    )
    .unwrap();
    let client = TcpStream::connect(to).unwrap();
    drive(&io, || accepted.len() == 1);
    assert!(accepted.take().remove(0).is_ok());
    let connected: Outcomes<io::Result<()>> = Outcomes::default();
    let sink = connected.clone();
    let socket = socket_for(to);
    io.connect_with(
        socket.as_raw_fd(),
        to,
        NEVER,
        Box::new(move |result| {
            sink.0.lock().unwrap().push(result);
            ended()
        }),
    )
    .unwrap();
    drive(&io, || connected.len() == 1);
    assert!(connected.take().remove(0).is_ok());
    // Nothing else ends later, and the ring still works after the entries of the limits ended.
    for _ in 0..200 {
        step(&io);
    }
    assert_eq!(
        (io.pending(), out.len(), accepted.len(), connected.len()),
        (0, 0, 0, 0)
    );
    peer.write_all(b"z").unwrap();
    recv_ok(&io, &ours, b"z");
    drop(client);
}

/// A new unconnected stream socket of the address's family.
fn socket_for(to: SocketAddr) -> OwnedFd {
    use std::os::fd::FromRawFd;
    let family = if to.is_ipv4() {
        libc::AF_INET
    } else {
        libc::AF_INET6
    };
    // SAFETY: a plain `socket(2)` call; the descriptor is new and owned by the result.
    let fd = unsafe { libc::socket(family, libc::SOCK_STREAM | libc::SOCK_CLOEXEC, 0) };
    assert!(fd >= 0);
    // SAFETY: `fd` was just opened and nothing else owns it.
    unsafe { OwnedFd::from_raw_fd(fd) }
}

#[test]
fn a_cancelled_receive_ends_once_with_the_cancelled_error_and_the_socket_receives_afterwards() {
    let io = AsyncIo::open().unwrap();
    let (mut peer, ours) = pair();
    let out = Outcomes::default();
    let id = io
        .recv_with(
            ours.as_raw_fd(),
            buf(64),
            OpOptions::default(),
            io_then(&out),
        )
        .unwrap();
    stay_pending(&io, || out.len());
    io.cancel(id);
    io.cancel(id);
    drive(&io, || out.len() == 1);
    let (result, len, filled) = only(&out);
    let error = result.unwrap_err();
    assert!(Cancelled::is(&error));
    assert_ne!(error.kind(), io::ErrorKind::TimedOut);
    assert_eq!((len, filled), (64, 0));
    assert_eq!(io.pending(), 0);
    for _ in 0..100 {
        step(&io);
    }
    assert_eq!(out.len(), 0, "a second completion came");
    peer.write_all(b"after").unwrap();
    recv_ok(&io, &ours, b"after");
}

#[test]
fn a_cancelled_send_to_a_peer_with_full_buffers_ends_once_and_a_limited_one_times_out() {
    let io = AsyncIo::open().unwrap();
    // The peer never reads: the socket's buffers fill and the send waits.
    let (_peer, ours) = pair();
    let total = 256 * 1024 * 1024;
    let big = || IoBuf {
        bytes: vec![1; total],
        filled: total,
    };
    let out = Outcomes::default();
    let id = io
        .send_with(ours.as_raw_fd(), big(), OpOptions::default(), io_then(&out))
        .unwrap();
    stay_pending(&io, || out.len());
    io.cancel(id);
    drive(&io, || out.len() == 1);
    let (result, len, _) = only(&out);
    assert!(Cancelled::is(&result.unwrap_err()));
    assert_eq!(len, total);
    assert_eq!(io.pending(), 0);
    for _ in 0..100 {
        step(&io);
    }
    assert_eq!(out.len(), 0);
    // The buffers are still full: a limited send times out, and gets its buffer back.
    io.send_with(ours.as_raw_fd(), big(), limit(50), io_then(&out))
        .unwrap();
    drive(&io, || out.len() == 1);
    let (result, len, _) = only(&out);
    assert_eq!(result.unwrap_err().kind(), io::ErrorKind::TimedOut);
    assert_eq!(len, total);
}

#[test]
fn a_cancel_that_comes_after_the_completion_delivers_the_real_result_once() {
    let io = AsyncIo::open().unwrap();
    let (mut peer, ours) = pair();
    peer.write_all(b"data").unwrap();
    let out = Outcomes::default();
    // The data is there: the kernel completes the receive during the submission, before this
    // thread can ask to cancel it, and before it was reaped.
    let id = io
        .recv_with(
            ours.as_raw_fd(),
            buf(64),
            OpOptions::default(),
            io_then(&out),
        )
        .unwrap();
    io.cancel(id);
    drive(&io, || out.len() == 1);
    assert_eq!(only(&out).0.unwrap(), 4);
    for _ in 0..100 {
        step(&io);
    }
    assert_eq!(out.len(), 0);
    assert_eq!(io.pending(), 0);
}

#[test]
fn cancelling_an_operation_that_ended_does_nothing_even_after_its_slot_was_reused() {
    let io = AsyncIo::open().unwrap();
    let (mut peer, ours) = pair();
    peer.write_all(b"first").unwrap();
    let first = Outcomes::default();
    let old = io
        .recv_with(
            ours.as_raw_fd(),
            buf(64),
            OpOptions::default(),
            io_then(&first),
        )
        .unwrap();
    drive(&io, || first.len() == 1);
    assert_eq!(only(&first).0.unwrap(), 5);
    // The next operation of this thread takes the freed slot.
    let second = Outcomes::default();
    let new = io
        .recv_with(
            ours.as_raw_fd(),
            buf(64),
            OpOptions::default(),
            io_then(&second),
        )
        .unwrap();
    assert_ne!(old, new);
    io.cancel(old);
    stay_pending(&io, || second.len());
    assert_eq!(io.pending(), 1);
    peer.write_all(b"second").unwrap();
    drive(&io, || second.len() == 1);
    assert_eq!(only(&second).0.unwrap(), 6);
}

#[test]
fn a_limited_accept_with_no_peer_times_out_and_the_listener_accepts_afterwards() {
    let io = AsyncIo::open().unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let to = listener.local_addr().unwrap();
    let out: Outcomes<io::Result<OwnedFd>> = Outcomes::default();
    let sink = out.clone();
    io.accept_with(
        listener.as_raw_fd(),
        limit(50),
        Box::new(move |result| {
            sink.0.lock().unwrap().push(result);
            ended()
        }),
    )
    .unwrap();
    drive(&io, || out.len() == 1);
    assert_eq!(
        out.take().remove(0).unwrap_err().kind(),
        io::ErrorKind::TimedOut
    );
    let sink = out.clone();
    io.accept(
        listener.as_raw_fd(),
        Box::new(move |result| {
            sink.0.lock().unwrap().push(result);
            ended()
        }),
    )
    .unwrap();
    let _client = TcpStream::connect(to).unwrap();
    drive(&io, || out.len() == 1);
    assert!(out.take().remove(0).is_ok());
}

#[test]
fn a_limited_connect_to_a_peer_that_does_not_answer_times_out() {
    let io = AsyncIo::open().unwrap();
    // A listener whose queue holds one connection and is full: further handshakes are dropped
    // and never answered.
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let to = listener.local_addr().unwrap();
    // SAFETY: `listen(2)` on a listening socket of ours only changes its backlog.
    assert_eq!(unsafe { libc::listen(listener.as_raw_fd(), 0) }, 0);
    let _queued = TcpStream::connect(to).unwrap();
    let _more = TcpStream::connect_timeout(&to, Duration::from_millis(100));
    let out: Outcomes<io::Result<()>> = Outcomes::default();
    let sink = out.clone();
    let socket = socket_for(to);
    io.connect_with(
        socket.as_raw_fd(),
        to,
        limit(100),
        Box::new(move |result| {
            sink.0.lock().unwrap().push(result);
            ended()
        }),
    )
    .unwrap();
    drive(&io, || out.len() == 1);
    assert_eq!(
        out.take().remove(0).unwrap_err().kind(),
        io::ErrorKind::TimedOut
    );
    assert_eq!(io.pending(), 0);
}
