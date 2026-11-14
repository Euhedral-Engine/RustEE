//! `AsyncIo::socket_for`: a new unconnected socket that `connect` connects and that carries data, for IPv4 and
//! IPv6, and that closes when dropped, through the real engine (io_uring), over loopback.
#![allow(
    clippy::disallowed_types,
    reason = "test fixtures: continuations record into a list the test thread reads"
)]
#![cfg(all(feature = "async-io", any(target_os = "linux", target_os = "android")))]
// Fixtures open sockets outside any lattice.
#![allow(clippy::disallowed_methods)]

use std::io::{self, Read, Write};
use std::net::{SocketAddr, TcpListener};
use std::os::fd::AsRawFd;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use euhedral_core::frames::{Frame, FrameHeader};
use euhedral_core::generics::LatticeSource;
use euhedral_io::{AsyncIo, IoBuf};

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

fn drive(io: &AsyncIo, mut done: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while !done() {
        assert!(Instant::now() < deadline, "the operations did not end");
        io.reap();
        io.pull(&mut |_| {}, &mut |_| false, u64::MAX);
    }
}

fn open_descriptors() -> usize {
    std::fs::read_dir("/proc/self/fd").unwrap().count()
}

/// Connects a new socket to `to`, sends `ping`, and reads `pong` back from the listener's side.
fn round_trip(io: &AsyncIo, listener: &TcpListener) {
    let to = listener.local_addr().unwrap();
    let socket = io.socket_for(to).unwrap();
    let connected: Arc<Mutex<Vec<io::Result<()>>>> = Arc::default();
    let sink = connected.clone();
    io.connect(
        socket.as_raw_fd(),
        to,
        Box::new(move |result| {
            sink.lock().unwrap().push(result);
            ended()
        }),
    )
    .unwrap();
    drive(io, || connected.lock().unwrap().len() == 1);
    connected.lock().unwrap().remove(0).unwrap();
    let (mut peer, _) = listener.accept().unwrap();

    let sent: Arc<Mutex<Vec<io::Result<usize>>>> = Arc::default();
    let sink = sent.clone();
    io.send(
        socket.as_raw_fd(),
        IoBuf {
            bytes: b"ping".to_vec(),
            filled: 4,
        },
        Box::new(move |done| {
            sink.lock().unwrap().push(done.result);
            ended()
        }),
    )
    .unwrap();
    drive(io, || sent.lock().unwrap().len() == 1);
    assert_eq!(sent.lock().unwrap().remove(0).unwrap(), 4);
    let mut ping = [0u8; 4];
    peer.read_exact(&mut ping).unwrap();
    assert_eq!(&ping, b"ping");

    peer.write_all(b"pong").unwrap();
    let got: Arc<Mutex<Vec<Vec<u8>>>> = Arc::default();
    let sink = got.clone();
    io.recv(
        socket.as_raw_fd(),
        IoBuf {
            bytes: vec![0; 16],
            filled: 0,
        },
        Box::new(move |done| {
            let n = done.result.unwrap();
            sink.lock().unwrap().push(done.buf.bytes[..n].to_vec());
            ended()
        }),
    )
    .unwrap();
    drive(io, || got.lock().unwrap().len() == 1);
    assert_eq!(got.lock().unwrap().remove(0), b"pong");
    assert_eq!(io.pending(), 0);
}

#[test]
fn a_new_socket_connects_and_carries_data_and_is_closed_when_dropped() {
    let io = AsyncIo::open().unwrap();
    // The first operation may open what the engine keeps; the count is taken after it.
    round_trip(&io, &TcpListener::bind("127.0.0.1:0").unwrap());
    let before = open_descriptors();

    // Dropped unconnected: nothing stays open, and nothing was in flight.
    let socket = io
        .socket_for(SocketAddr::from(([127, 0, 0, 1], 9)))
        .unwrap();
    assert_eq!(open_descriptors(), before + 1);
    assert_eq!(io.pending(), 0);
    drop(socket);
    assert_eq!(open_descriptors(), before);

    // IPv6 loopback may be absent on the machine.
    if let Ok(listener) = TcpListener::bind("[::1]:0") {
        round_trip(&io, &listener);
    }
    assert_eq!(open_descriptors(), before);
}
