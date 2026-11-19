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

mod common;

use std::io::{self, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::os::fd::{AsRawFd, OwnedFd};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use euhedral_core::frames::{Frame, FrameHeader};
use euhedral_core::generics::LatticeSource;
use euhedral_io::{AsyncIo, Cancelled, IoBuf, LimitPassed, OpOptions};

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

fn a_receive_with_a_limit_on_a_silent_peer_times_out_and_the_socket_receives_afterwards_on(
    open: fn() -> Arc<AsyncIo>,
) {
    let io = open();
    let (mut peer, ours) = pair();
    let out = Outcomes::default();
    io.recv_with(ours.as_raw_fd(), buf(64), limit(50), io_then(&out))
        .unwrap();
    assert_eq!(io.pending(), 1);
    drive(&io, || out.len() == 1);
    let (result, len, filled) = only(&out);
    let error = result.unwrap_err();
    assert_eq!(error.kind(), io::ErrorKind::TimedOut);
    // Its own marker, not an `ETIMEDOUT` of the system, and not a cancel.
    assert!(LimitPassed::is(&error) && !Cancelled::is(&error));
    assert!(error.raw_os_error().is_none());
    assert_eq!(LimitPassed::moved_of(&error), Some(0));
    assert_eq!((len, filled), (64, 0));
    assert_eq!(io.pending(), 0);
    peer.write_all(b"hello").unwrap();
    recv_ok(&io, &ours, b"hello");
}

#[test]
fn a_receive_with_a_limit_on_a_silent_peer_times_out_and_the_socket_receives_afterwards() {
    for open in common::engines() {
        a_receive_with_a_limit_on_a_silent_peer_times_out_and_the_socket_receives_afterwards_on(
            open,
        );
    }
}

fn a_limited_operation_that_finishes_in_time_ends_once_and_leaves_no_trace_on(
    open: fn() -> Arc<AsyncIo>,
) {
    let io = open();
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
    let socket = io.socket_for(to).unwrap();
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

#[test]
fn a_limited_operation_that_finishes_in_time_ends_once_and_leaves_no_trace() {
    for open in common::engines() {
        a_limited_operation_that_finishes_in_time_ends_once_and_leaves_no_trace_on(open);
    }
}

fn a_cancelled_receive_ends_once_with_the_cancelled_error_and_the_socket_receives_afterwards_on(
    open: fn() -> Arc<AsyncIo>,
) {
    let io = open();
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
    // Not `Interrupted` (a retry loop would reissue it), not a timeout, and no OS code.
    assert_eq!(error.kind(), io::ErrorKind::Other);
    assert!(error.raw_os_error().is_none() && !LimitPassed::is(&error));
    assert_eq!(Cancelled::moved_of(&error), Some(0));
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
fn a_cancelled_receive_ends_once_with_the_cancelled_error_and_the_socket_receives_afterwards() {
    for open in common::engines() {
        a_cancelled_receive_ends_once_with_the_cancelled_error_and_the_socket_receives_afterwards_on(open);
    }
}

/// Sets a socket option of integer type.
fn set_int(socket: &impl AsRawFd, level: i32, name: i32, value: i32) {
    // SAFETY: `setsockopt(2)` on a socket we own, with a pointer to an int that outlives the call.
    let set = unsafe {
        libc::setsockopt(
            socket.as_raw_fd(),
            level,
            name,
            std::ptr::from_ref(&value).cast(),
            size_of::<i32>() as libc::socklen_t,
        )
    };
    assert_eq!(set, 0);
}

/// A connected pair whose buffers are small (a few tens of KiB), so that a megabyte sent to a peer
/// that never reads cannot fit.
fn small_pair() -> (TcpStream, TcpStream) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    // The window of the accepted socket comes from the listener's receive buffer at the handshake.
    set_int(&listener, libc::SOL_SOCKET, libc::SO_RCVBUF, 4096);
    let client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
    set_int(&client, libc::SOL_SOCKET, libc::SO_SNDBUF, 4096);
    let (server, _) = listener.accept().unwrap();
    (client, server)
}

fn a_cancelled_send_to_a_peer_with_full_buffers_ends_once_and_a_limited_one_times_out_on(
    open: fn() -> Arc<AsyncIo>,
) {
    let io = open();
    // The peer never reads: the socket's buffers fill and the send waits.
    let (_peer, ours) = small_pair();
    let total = 4 * 1024 * 1024;
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
    let error = result.unwrap_err();
    assert!(Cancelled::is(&error));
    let sent = Cancelled::moved_of(&error).unwrap();
    assert!(sent < total, "{sent} of {total} bytes");
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
    let error = result.unwrap_err();
    assert_eq!(error.kind(), io::ErrorKind::TimedOut);
    assert!(LimitPassed::moved_of(&error).is_some_and(|moved| moved < total));
    assert_eq!(len, total);
}

#[test]
fn a_cancelled_send_to_a_peer_with_full_buffers_ends_once_and_a_limited_one_times_out() {
    for open in common::engines() {
        a_cancelled_send_to_a_peer_with_full_buffers_ends_once_and_a_limited_one_times_out_on(open);
    }
}

fn a_cancel_that_comes_after_the_completion_delivers_the_real_result_once_on(
    open: fn() -> Arc<AsyncIo>,
) {
    let io = open();
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
    // On io_uring the kernel completed the receive during the submission, so this cancel comes after it. The
    // fallback ends the receive at its first visit: reap first, then cancel.
    #[cfg(any(target_os = "linux", target_os = "android"))]
    if io.backend() == euhedral_io::IoBackend::Positional {
        drive(&io, || out.len() == 1);
    }
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
fn a_cancel_that_comes_after_the_completion_delivers_the_real_result_once() {
    for open in common::engines() {
        a_cancel_that_comes_after_the_completion_delivers_the_real_result_once_on(open);
    }
}

fn cancelling_an_operation_that_ended_does_nothing_even_after_its_slot_was_reused_on(
    open: fn() -> Arc<AsyncIo>,
) {
    let io = open();
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
fn cancelling_an_operation_that_ended_does_nothing_even_after_its_slot_was_reused() {
    for open in common::engines() {
        cancelling_an_operation_that_ended_does_nothing_even_after_its_slot_was_reused_on(open);
    }
}

fn a_limited_accept_with_no_peer_times_out_and_the_listener_accepts_afterwards_on(
    open: fn() -> Arc<AsyncIo>,
) {
    let io = open();
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
fn a_limited_accept_with_no_peer_times_out_and_the_listener_accepts_afterwards() {
    for open in common::engines() {
        a_limited_accept_with_no_peer_times_out_and_the_listener_accepts_afterwards_on(open);
    }
}

/// No address that never answers exists on every machine, so the test makes one on loopback: a
/// listener whose accept queue is full has the kernel drop the SYNs that come after it, and answer
/// none (whatever `tcp_abort_on_overflow` says: that setting only governs the final ACK of a
/// handshake that already began).
fn a_limited_connect_to_a_peer_that_does_not_answer_times_out_on(open: fn() -> Arc<AsyncIo>) {
    let io = open();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let SocketAddr::V4(to) = listener.local_addr().unwrap() else {
        unreachable!("bound to an IPv4 address")
    };
    // SAFETY: `listen(2)` on a listening socket of ours only changes its backlog.
    assert_eq!(unsafe { libc::listen(listener.as_raw_fd(), 0) }, 0);
    // One connection fills the queue (a backlog of 0 holds one); the next handshake is dropped.
    let _queued = TcpStream::connect(to).unwrap();
    let out: Outcomes<io::Result<()>> = Outcomes::default();
    let sink = out.clone();
    let socket = io.socket_for(SocketAddr::V4(to)).unwrap();
    io.connect_with(
        socket.as_raw_fd(),
        SocketAddr::V4(to),
        limit(100),
        Box::new(move |result| {
            sink.0.lock().unwrap().push(result);
            ended()
        }),
    )
    .unwrap();
    drive(&io, || out.len() == 1);
    let error = out.take().remove(0).unwrap_err();
    assert_eq!(
        error.kind(),
        io::ErrorKind::TimedOut,
        "the kernel answered the handshake: {error}"
    );
    assert!(LimitPassed::is(&error));
    assert_eq!(io.pending(), 0);
}

#[test]
fn a_limited_connect_to_a_peer_that_does_not_answer_times_out() {
    for open in common::engines() {
        a_limited_connect_to_a_peer_that_does_not_answer_times_out_on(open);
    }
}

fn a_cancel_from_another_worker_ends_the_operation_once_on(open: fn() -> Arc<AsyncIo>) {
    let io = open();
    let (_peer, ours) = pair();
    let out = Outcomes::default();
    let id = io
        .recv_with(ours.as_raw_fd(), buf(64), NEVER, io_then(&out))
        .unwrap();
    stay_pending(&io, || out.len());
    // A cancel is a request any thread may make: here, one that did not submit.
    let canceller = {
        let io = io.clone();
        std::thread::spawn(move || io.cancel(id))
    };
    canceller.join().unwrap();
    drive(&io, || out.len() == 1);
    let (result, ..) = only(&out);
    assert!(Cancelled::is(&result.unwrap_err()));
    for _ in 0..100 {
        step(&io);
    }
    assert_eq!(out.len(), 0);
    assert_eq!(io.pending(), 0);
}

#[test]
fn a_cancel_from_another_worker_ends_the_operation_once() {
    for open in common::engines() {
        a_cancel_from_another_worker_ends_the_operation_once_on(open);
    }
}

fn cancelling_a_limited_operation_ends_it_with_the_cancel_and_no_timeout_follows_on(
    open: fn() -> Arc<AsyncIo>,
) {
    let io = open();
    let (mut peer, ours) = pair();
    let out = Outcomes::default();
    let id = io
        .recv_with(ours.as_raw_fd(), buf(64), limit(200), io_then(&out))
        .unwrap();
    stay_pending(&io, || out.len());
    io.cancel(id);
    drive(&io, || out.len() == 1);
    let (result, ..) = only(&out);
    let error = result.unwrap_err();
    assert!(Cancelled::is(&error) && !LimitPassed::is(&error));
    // Past the limit nothing more comes, and the socket works.
    let past = Instant::now() + Duration::from_millis(300);
    while Instant::now() < past {
        step(&io);
    }
    assert_eq!(out.len(), 0);
    assert_eq!(io.pending(), 0);
    peer.write_all(b"ok").unwrap();
    recv_ok(&io, &ours, b"ok");
}

#[test]
fn cancelling_a_limited_operation_ends_it_with_the_cancel_and_no_timeout_follows() {
    for open in common::engines() {
        cancelling_a_limited_operation_ends_it_with_the_cancel_and_no_timeout_follows_on(open);
    }
}

fn the_id_of_another_asyncio_cancels_nothing_on(open: fn() -> Arc<AsyncIo>) {
    let first = open();
    let second = open();
    let (mut peer, ours) = pair();
    let out = Outcomes::default();
    let id = first
        .recv_with(ours.as_raw_fd(), buf(64), NEVER, io_then(&out))
        .unwrap();
    // The same ring index and slot can exist in both engines: only the engine tells them apart.
    let (_peer2, ours2) = pair();
    let out2 = Outcomes::default();
    let id2 = second
        .recv_with(ours2.as_raw_fd(), buf(64), NEVER, io_then(&out2))
        .unwrap();
    assert_ne!(id, id2);
    second.cancel(id);
    first.cancel(id2);
    stay_pending(&first, || out.len());
    stay_pending(&second, || out2.len());
    assert_eq!((first.pending(), second.pending()), (1, 1));
    peer.write_all(b"fine").unwrap();
    drive(&first, || out.len() == 1);
    assert_eq!(only(&out).0.unwrap(), 4);
    second.cancel(id2);
    drive(&second, || out2.len() == 1);
    assert!(Cancelled::is(&only(&out2).0.unwrap_err()));
}

#[test]
fn the_id_of_another_asyncio_cancels_nothing() {
    for open in common::engines() {
        the_id_of_another_asyncio_cancels_nothing_on(open);
    }
}
