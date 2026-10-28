//! The engine on a real Windows: files opened for overlapped I/O, loopback sockets, open and
//! rename on the pool. Every test runs twice: with IoRing where the OS has it, and forced onto
//! the completion port. Compiled and run on Windows only.

// Test fixtures: blocking files and sockets (the peers), outside any lattice.
#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream, UdpSocket};
use std::os::windows::fs::OpenOptionsExt;
use std::os::windows::io::{AsRawHandle, AsRawSocket};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use euhedral_core::frames::{Frame, FrameHeader};
use euhedral_core::generics::LatticeSource;

use super::real::RealSys;
use super::sys::Sys;
use crate::Continuation;
use crate::{
    AsyncIo, AsyncIoOptions, Create, IoBuf, IoDone, IoDoneOwned, IoThen, OpenSpec, OwnedFile,
    OwnedSock,
};

const FILE_FLAG_OVERLAPPED: u32 = 0x4000_0000;

/// A completion frame: carries what the operation ended with.
struct Done {
    header: FrameHeader,
    done: Option<IoDone>,
    raw: Option<std::io::Result<()>>,
}

impl Frame for Done {
    fn header(&self) -> &FrameHeader {
        &self.header
    }

    fn header_mut(&mut self) -> &mut FrameHeader {
        &mut self.header
    }
}

fn then(tag: u64) -> IoThen {
    Box::new(move |done| {
        Box::new(Done {
            header: FrameHeader::new(tag),
            done: Some(done),
            raw: None,
        })
    })
}

fn unit_then(tag: u64) -> Continuation<std::io::Result<()>> {
    Box::new(move |result| {
        Box::new(Done {
            header: FrameHeader::new(tag),
            done: None,
            raw: Some(result),
        })
    })
}

/// What the engine should use on this machine: file reads and writes through a ring.
fn expected_ring_use(windows_ioring: bool) -> (bool, bool) {
    let Some(probe) = RealSys::new()
        .ring_probe()
        .filter(|p| windows_ioring && !p.emulated)
    else {
        return (false, false);
    };
    (
        probe.read,
        probe.write && probe.version >= super::sys::IORING_VERSION_3,
    )
}

/// Runs a test body with IoRing (where the OS has it) and then forced onto the port.
fn both(body: impl Fn(&Arc<AsyncIo>)) {
    for windows_ioring in [true, false] {
        let io = AsyncIo::open_with_options(AsyncIoOptions { windows_ioring }).unwrap();
        assert_eq!(io.engine.ring_use(), expected_ring_use(windows_ioring));
        body(&io);
        assert_eq!(io.pending(), 0, "windows_ioring = {windows_ioring}");
    }
}

/// Polls as a worker does until `count` completions came out (10 s at most).
fn poll(io: &AsyncIo, count: usize) -> Vec<Done> {
    let mut out = Vec::new();
    let deadline = Instant::now() + Duration::from_secs(10);
    while out.len() < count {
        assert!(Instant::now() < deadline, "the operations did not complete");
        io.pull(
            &mut |frame| {
                out.push(
                    *(frame as Box<dyn std::any::Any>)
                        .downcast::<Done>()
                        .expect("a Done frame"),
                )
            },
            &mut |_| false,
            u64::MAX,
        );
    }
    out.sort_by_key(|done| done.header.id_hash());
    out
}

fn one(io: &AsyncIo) -> Done {
    poll(io, 1).pop().unwrap()
}

fn content(len: usize, seed: u64) -> Vec<u8> {
    let mut state = seed | 1;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state as u8
        })
        .collect()
}

fn buf(len: usize) -> IoBuf {
    IoBuf {
        bytes: vec![0; len],
        filled: 0,
    }
}

/// A scratch path beside the test binary, under `target/`.
fn scratch(name: &str) -> PathBuf {
    let exe = std::env::current_exe().expect("the test binary's path");
    let dir = exe
        .parent()
        .and_then(std::path::Path::parent)
        .expect("a target directory")
        .join("euhedral-graph-scratch");
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(format!(
        "{name}-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ))
}

/// A file of `content` opened for overlapped I/O.
fn overlapped_file(name: &str, content: &[u8]) -> (PathBuf, std::fs::File) {
    let path = scratch(name);
    std::fs::write(&path, content).unwrap();
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .custom_flags(FILE_FLAG_OVERLAPPED)
        .open(&path)
        .unwrap();
    (path, file)
}

/// Prints what this machine offers, so a CI log says which engine the default run exercised.
#[test]
fn the_machines_ring_is_reported() {
    let probe = RealSys::new().ring_probe();
    eprintln!("IoRing on this machine: {probe:?}");
}

#[test]
fn a_file_read_in_several_pieces_and_a_tail_that_ends_inside() {
    both(|io| {
        let data = content(3 * 256 * 1024 + 4321, 3);
        let (path, file) = overlapped_file("pieces", &data);
        io.read(file.as_raw_handle(), 0, buf(data.len()), then(1))
            .unwrap();
        io.read(file.as_raw_handle(), 100, buf(data.len()), then(2))
            .unwrap();
        let done = poll(io, 2);
        let a = done[0].done.as_ref().unwrap();
        assert_eq!(a.result.as_ref().unwrap(), &data.len());
        assert_eq!(a.buf.bytes, data);
        let b = done[1].done.as_ref().unwrap();
        assert_eq!(b.result.as_ref().unwrap(), &(data.len() - 100));
        assert_eq!(&b.buf.bytes[..b.buf.filled], &data[100..]);
        drop(file);
        std::fs::remove_file(path).unwrap();
    });
}

/// Reads into memory the caller owns: cut into pieces, short at the end of the file, and
/// returned on failure.
#[test]
fn a_read_into_owned_memory_lands_in_place_and_returns_the_memory() {
    both(|io| {
        let data = content(3 * 256 * 1024 + 4321, 4);
        let (path, file) = overlapped_file("owned", &data);
        let landed = Arc::new(Mutex::new(Vec::<(u64, IoDoneOwned<Vec<u8>>)>::new()));
        let read = |offset: u64, len: usize, tag: u64| {
            let landed = landed.clone();
            io.read_into(
                file.as_raw_handle(),
                offset,
                vec![0xAA; len],
                Box::new(move |done| {
                    landed.lock().unwrap().push((tag, done));
                    Box::new(Done {
                        header: FrameHeader::new(tag),
                        done: None,
                        raw: None,
                    })
                }),
            )
            .unwrap();
        };
        read(0, data.len(), 1);
        read(100, data.len(), 2);
        read(1 << 30, 64, 3);
        poll(io, 3);
        let mut landed = std::mem::take(&mut *landed.lock().unwrap());
        landed.sort_by_key(|(tag, _)| *tag);
        let [(_, whole), (_, tail), (_, past)] = &mut landed[..] else {
            panic!("three reads ended");
        };
        assert_eq!(whole.result.as_ref().unwrap(), &data.len());
        assert_eq!(whole.buf.as_deref(), Some(&data[..]));
        assert_eq!(tail.result.as_ref().unwrap(), &(data.len() - 100));
        assert_eq!(
            &tail.buf.as_ref().unwrap()[..data.len() - 100],
            &data[100..]
        );
        assert_eq!(
            past.result.as_ref().unwrap_err().kind(),
            std::io::ErrorKind::UnexpectedEof
        );
        assert_eq!(past.buf.as_deref(), Some(&[0xAA; 64][..]));
        drop(file);
        std::fs::remove_file(path).unwrap();
    });
}

#[test]
fn a_read_past_the_end_of_the_file_fails_with_unexpected_eof() {
    both(|io| {
        let (path, file) = overlapped_file("eof", &[0; 100]);
        io.read(file.as_raw_handle(), 8192, buf(4096), then(1))
            .unwrap();
        let error = one(io).done.unwrap().result.unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::UnexpectedEof, "{error}");
        drop(file);
        std::fs::remove_file(path).unwrap();
    });
}

#[test]
fn a_write_lands_at_its_offset_in_pieces() {
    both(|io| {
        let (path, file) = overlapped_file("write", &[7; 16]);
        let data = content(2 * 256 * 1024 + 11, 5);
        let write = IoBuf {
            bytes: data.clone(),
            filled: data.len(),
        };
        io.write(file.as_raw_handle(), 16, write, then(1)).unwrap();
        assert_eq!(one(io).done.unwrap().result.unwrap(), data.len());
        drop(file);
        let on_disk = std::fs::read(&path).unwrap();
        assert_eq!(&on_disk[..16], &[7; 16]);
        assert_eq!(&on_disk[16..], &data[..]);
        std::fs::remove_file(path).unwrap();
    });
}

#[test]
fn a_handle_opened_without_overlapped_io_is_refused() {
    both(|io| {
        let path = scratch("sync");
        std::fs::write(&path, [1; 10]).unwrap();
        let file = std::fs::File::open(&path).unwrap();
        let error = io
            .read(file.as_raw_handle(), 0, buf(10), then(1))
            .unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
        drop(file);
        std::fs::remove_file(path).unwrap();
    });
}

#[test]
fn open_write_and_rename_make_an_atomic_save() {
    both(|io| {
        let target = scratch("saved");
        let temp = scratch("saved.tmp");
        std::fs::write(&target, b"old").unwrap();
        let opened: Arc<Mutex<Option<std::io::Result<OwnedFile>>>> = Arc::default();
        let slot = opened.clone();
        let spec = OpenSpec {
            read: false,
            write: true,
            create: Create::Truncate,
            ..OpenSpec::default()
        };
        io.open_path(
            &temp,
            spec,
            Box::new(move |result| {
                *slot.lock().unwrap() = Some(result);
                Box::new(Done {
                    header: FrameHeader::new(1),
                    done: None,
                    raw: Some(Ok(())),
                })
            }),
        )
        .unwrap();
        assert!(one(io).raw.unwrap().is_ok());
        let file = opened.lock().unwrap().take().unwrap().unwrap();
        let write = IoBuf {
            bytes: b"new contents".to_vec(),
            filled: 12,
        };
        io.write(file.as_raw_handle(), 0, write, then(2)).unwrap();
        assert_eq!(one(io).done.unwrap().result.unwrap(), 12);
        drop(file);
        io.rename(&temp, &target, unit_then(3)).unwrap();
        assert!(one(io).raw.unwrap().is_ok());
        assert_eq!(std::fs::read(&target).unwrap(), b"new contents");
        assert!(!temp.exists());
        std::fs::remove_file(target).unwrap();

        // A missing file is an error, not a hang.
        let missing = scratch("missing");
        io.open_path(
            &missing,
            OpenSpec::default(),
            Box::new(|result: std::io::Result<OwnedFile>| {
                Box::new(Done {
                    header: FrameHeader::new(4),
                    done: None,
                    raw: Some(result.map(|_| ())),
                })
            }),
        )
        .unwrap();
        let error = one(io).raw.unwrap().unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::NotFound);
    });
}

fn loopback_pair() -> (TcpStream, TcpStream) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
    let (server, _) = listener.accept().unwrap();
    (client, server)
}

#[test]
fn a_send_and_a_receive_over_loopback() {
    both(|io| {
        let (client, mut server) = loopback_pair();
        let data = content(300_000, 9);
        let send = IoBuf {
            bytes: data.clone(),
            filled: data.len(),
        };
        io.send(client.as_raw_socket(), send, then(1)).unwrap();
        // Drain the peer as the send proceeds.
        let mut got = Vec::new();
        let mut chunk = vec![0u8; 64 * 1024];
        let mut sent = false;
        let deadline = Instant::now() + Duration::from_secs(10);
        while got.len() < data.len() {
            assert!(Instant::now() < deadline, "the send did not finish");
            io.pull(
                &mut |frame| {
                    let done = (frame as Box<dyn std::any::Any>)
                        .downcast::<Done>()
                        .unwrap();
                    assert_eq!(done.done.unwrap().result.unwrap(), data.len());
                    sent = true;
                },
                &mut |_| false,
                u64::MAX,
            );
            server.set_nonblocking(true).unwrap();
            match server.read(&mut chunk) {
                Ok(n) => got.extend_from_slice(&chunk[..n]),
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {}
                Err(error) => panic!("{error}"),
            }
        }
        assert_eq!(got, data);
        while !sent {
            poll_until_sent(io, &mut sent);
        }

        server.set_nonblocking(false).unwrap();
        server.write_all(b"hello").unwrap();
        io.recv(client.as_raw_socket(), buf(64), then(2)).unwrap();
        let done = one(io).done.unwrap();
        assert_eq!(done.result.unwrap(), 5);
        assert_eq!(&done.buf.bytes[..done.buf.filled], b"hello");

        drop(server);
        io.recv(client.as_raw_socket(), buf(64), then(3)).unwrap();
        assert_eq!(one(io).done.unwrap().result.unwrap(), 0, "the peer's end");
    });
}

fn poll_until_sent(io: &AsyncIo, sent: &mut bool) {
    io.pull(
        &mut |frame| {
            let done = (frame as Box<dyn std::any::Any>)
                .downcast::<Done>()
                .unwrap();
            assert!(done.done.unwrap().result.is_ok());
            *sent = true;
        },
        &mut |_| false,
        u64::MAX,
    );
}

/// An unconnected overlapped stream socket of `listener`'s family (std has no such socket).
fn unconnected_socket(listener: &TcpListener) -> usize {
    RealSys::new()
        .accept_socket(listener.as_raw_socket() as usize)
        .unwrap()
}

#[test]
fn an_accepted_socket_carries_bytes_both_ways() {
    both(|io| {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let accepted: Arc<Mutex<Option<std::io::Result<OwnedSock>>>> = Arc::default();
        let slot = accepted.clone();
        io.accept(
            listener.as_raw_socket(),
            Box::new(move |result| {
                *slot.lock().unwrap() = Some(result);
                Box::new(Done {
                    header: FrameHeader::new(1),
                    done: None,
                    raw: Some(Ok(())),
                })
            }),
        )
        .unwrap();
        let mut client = TcpStream::connect(addr).unwrap();
        assert!(one(io).raw.unwrap().is_ok());
        let server = accepted.lock().unwrap().take().unwrap().unwrap();

        let send = IoBuf {
            bytes: b"ping".to_vec(),
            filled: 4,
        };
        io.send(server.as_raw_socket(), send, then(2)).unwrap();
        assert_eq!(one(io).done.unwrap().result.unwrap(), 4);
        let mut got = [0u8; 4];
        client.read_exact(&mut got).unwrap();
        assert_eq!(&got, b"ping");

        client.write_all(b"pong").unwrap();
        io.recv(server.as_raw_socket(), buf(16), then(3)).unwrap();
        let done = one(io).done.unwrap();
        assert_eq!(&done.buf.bytes[..done.buf.filled], b"pong");
    });
}

#[test]
fn a_connect_through_the_engine_reaches_the_listener() {
    both(|io| {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let socket = unconnected_socket(&listener);
        io.connect(socket as _, addr, unit_then(1)).unwrap();
        assert!(one(io).raw.unwrap().is_ok());
        let (mut peer, _) = listener.accept().unwrap();

        let send = IoBuf {
            bytes: b"hello".to_vec(),
            filled: 5,
        };
        io.send(socket as _, send, then(2)).unwrap();
        assert_eq!(one(io).done.unwrap().result.unwrap(), 5);
        let mut got = [0u8; 5];
        peer.read_exact(&mut got).unwrap();
        assert_eq!(&got, b"hello");
        RealSys::new().close_socket(socket);
    });
}

#[test]
fn a_connect_to_a_closed_port_is_refused() {
    both(|io| {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let closed = {
            let other = TcpListener::bind("127.0.0.1:0").unwrap();
            other.local_addr().unwrap()
        };
        let socket = unconnected_socket(&listener);
        io.connect(socket as _, closed, unit_then(1)).unwrap();
        let error = one(io).raw.unwrap().unwrap_err();
        assert_eq!(
            error.kind(),
            std::io::ErrorKind::ConnectionRefused,
            "{error}"
        );
        RealSys::new().close_socket(socket);
    });
}

#[test]
fn a_datagram_reaches_its_address() {
    both(|io| {
        let receiver = UdpSocket::bind("127.0.0.1:0").unwrap();
        receiver
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        let sender = UdpSocket::bind("127.0.0.1:0").unwrap();
        let datagram = b"a datagram".to_vec();
        let send = IoBuf {
            filled: datagram.len(),
            bytes: datagram.clone(),
        };
        io.send_to(
            sender.as_raw_socket(),
            receiver.local_addr().unwrap(),
            send,
            then(1),
        )
        .unwrap();
        assert_eq!(one(io).done.unwrap().result.unwrap(), datagram.len());
        let mut got = [0u8; 64];
        let n = receiver.recv(&mut got).unwrap();
        assert_eq!(&got[..n], &datagram[..]);
    });
}

#[test]
fn threads_submit_to_their_own_rings_and_every_read_ends_once() {
    both(|io| {
        let data = content(64 * 4096, 11);
        let (path, file) = overlapped_file("threads", &data);
        let raw = file.as_raw_handle() as usize;
        let io_ref = io.clone();
        let handles: Vec<_> = (0..8u64)
            .map(|i| {
                let io = io_ref.clone();
                #[allow(clippy::disallowed_methods)] // submitting threads, as lattice workers are
                std::thread::spawn(move || {
                    io.read(raw as _, i * 4096, buf(4096), then(i)).unwrap();
                })
            })
            .collect();
        for handle in handles {
            handle.join().unwrap();
        }
        let rings = io.engine.rings.lock().unwrap().len();
        assert_eq!(
            rings,
            if io.engine.ring_use().0 { 8 } else { 0 },
            "a ring per submitting thread"
        );
        for (i, done) in poll(io, 8).into_iter().enumerate() {
            let done = done.done.unwrap();
            assert_eq!(done.result.unwrap(), 4096);
            assert_eq!(done.buf.bytes, data[i * 4096..(i + 1) * 4096]);
        }
        drop(file);
        std::fs::remove_file(path).unwrap();
    });
}

#[test]
fn a_datagram_larger_than_the_buffer_ends_once_with_wsaemsgsize() {
    // The call reports failure and the I/O manager queues a packet for it anyway (a warning
    // status): the operation must still end once, and its slot must stay sound.
    both(|io| {
        let receiver = UdpSocket::bind("127.0.0.1:0").unwrap();
        let sender = UdpSocket::bind("127.0.0.1:0").unwrap();
        let to = receiver.local_addr().unwrap();
        for round in 0..20u64 {
            sender.send_to(&[7u8; 100], to).unwrap();
            std::thread::sleep(Duration::from_millis(5));
            io.recv(receiver.as_raw_socket(), buf(10), then(round))
                .unwrap();
            let error = one(io).done.unwrap().result.unwrap_err();
            assert_eq!(
                error.raw_os_error(),
                Some(10040),
                "WSAEMSGSIZE, round {round}"
            );
            // Nothing else ends, however long the port is polled.
            let mut extra = Vec::new();
            io.pull(&mut |frame| extra.push(frame), &mut |_| false, u64::MAX);
            assert!(extra.is_empty());
        }
        // The slots still work: a datagram that fits.
        sender.send_to(b"fits", to).unwrap();
        std::thread::sleep(Duration::from_millis(5));
        io.recv(receiver.as_raw_socket(), buf(10), then(99))
            .unwrap();
        let done = one(io).done.unwrap();
        assert_eq!(&done.buf.bytes[..done.buf.filled], b"fits");
        assert_eq!(
            io.engine.strays.load(std::sync::atomic::Ordering::Relaxed),
            0
        );
    });
}

#[test]
fn a_file_joined_to_a_port_is_served_by_the_port_not_a_ring() {
    // IoRing refuses a handle that belongs to a completion port (ERROR_INVALID_PARAMETER), so the
    // engine keeps such a handle on the port.
    let io = AsyncIo::open().unwrap();
    let data = content(8 * 4096, 21);
    let (path, file) = overlapped_file("ring-vs-port", &data);
    let raw = file.as_raw_handle() as usize;
    io.engine.sys.associate(io.engine.port.handle, raw).unwrap();
    io.engine.associated.lock().unwrap().insert(raw);
    for i in 0..4u64 {
        io.read(file.as_raw_handle(), i * 4096, buf(4096), then(i))
            .unwrap();
    }
    for (i, done) in poll(&io, 4).into_iter().enumerate() {
        let done = done.done.unwrap();
        assert!(done.result.is_ok(), "read {i}: {:?}", done.result);
        assert_eq!(done.buf.bytes, data[i * 4096..(i + 1) * 4096]);
    }
    assert_eq!(
        io.engine.rings.lock().unwrap().len(),
        0,
        "no ring was made for it"
    );
    drop(file);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn a_handle_that_skips_the_port_on_success_is_refused() {
    both(|io| {
        let (path, file) = overlapped_file("skip", &[0; 100]);
        // SAFETY: a file handle this test owns.
        let set = unsafe {
            windows_sys::Win32::Storage::FileSystem::SetFileCompletionNotificationModes(
                file.as_raw_handle(),
                1,
            )
        };
        assert_ne!(set, 0);
        let error = io
            .read(file.as_raw_handle(), 0, buf(10), then(1))
            .unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput, "{error}");
        drop(file);
        std::fs::remove_file(path).unwrap();
    });
}

#[test]
fn a_path_past_the_ordinary_limit_opens_as_a_verbatim_path() {
    both(|io| {
        let mut dir = scratch("long");
        for _ in 0..5 {
            dir.push("d".repeat(60));
        }
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("file.bin");
        assert!(path.as_os_str().len() > 260);
        std::fs::write(&path, b"long path").unwrap();
        let opened: Arc<Mutex<Option<std::io::Result<OwnedFile>>>> = Arc::default();
        let slot = opened.clone();
        io.open_path(
            &path,
            OpenSpec::default(),
            Box::new(move |result| {
                *slot.lock().unwrap() = Some(result);
                Box::new(Done {
                    header: FrameHeader::new(1),
                    done: None,
                    raw: Some(Ok(())),
                })
            }),
        )
        .unwrap();
        assert!(one(io).raw.unwrap().is_ok());
        let file = opened.lock().unwrap().take().unwrap().unwrap();
        io.read(file.as_raw_handle(), 0, buf(9), then(2)).unwrap();
        assert_eq!(one(io).done.unwrap().buf.bytes, b"long path");
    });
}
