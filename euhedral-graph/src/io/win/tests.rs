//! The engine's state machines against the simulated `Sys`: every operation's success, error and
//! short transfers, the ring and its fallbacks, a full submission queue, held-back frames.

use std::io;
use std::net::SocketAddr;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use euhedral_core::frames::{Frame, FrameHeader};

use super::sim::{Op, Sim};
use super::sys::{Handle, IORING_VERSION_3, RingProbe, SockAddr, hresult_error};
use super::{Engine, MAX_PIECE};
use crate::io::{
    AsyncIoOptions, Create, Engine as EngineTrait, Front, IoBuf, IoDone, OpenSpec, Request, Then,
};

/// A completion frame: carries what the operation ended with.
struct Done {
    header: FrameHeader,
    done: Option<IoDone>,
    raw: Option<io::Result<usize>>,
}

impl Frame for Done {
    fn header(&self) -> &FrameHeader {
        &self.header
    }

    fn header_mut(&mut self) -> &mut FrameHeader {
        &mut self.header
    }
}

fn io_then(tag: u64) -> Then {
    Then::Io(Box::new(move |done| {
        Box::new(Done {
            header: FrameHeader::new(tag),
            done: Some(done),
            raw: None,
        })
    }))
}

fn raw_then(tag: u64) -> Then {
    Then::Raw(Box::new(move |result| {
        Box::new(Done {
            header: FrameHeader::new(tag),
            done: None,
            raw: Some(result),
        })
    }))
}

fn full_probe() -> RingProbe {
    RingProbe {
        version: IORING_VERSION_3,
        emulated: false,
        read: true,
        write: true,
    }
}

/// An engine over a simulated `Sys`, driven as `AsyncIo` drives its engine.
struct Harness {
    sim: Arc<Sim>,
    engine: Engine,
    front: Front,
}

impl Harness {
    fn new() -> Self {
        Self::with(None, true, MAX_PIECE)
    }

    fn with(probe: Option<RingProbe>, ioring: bool, max_piece: usize) -> Self {
        let sim = Arc::new(Sim::new());
        sim.set_probe(probe);
        let engine = Engine::new(
            sim.clone(),
            AsyncIoOptions {
                windows_ioring: ioring,
            },
            max_piece,
        )
        .unwrap();
        Self {
            sim,
            engine,
            front: Front::new(),
        }
    }

    fn submit(&self, request: Request, buf: IoBuf, then: Then) -> io::Result<()> {
        self.front.pending.fetch_add(1, Ordering::AcqRel);
        let started = self.engine.submit(request, buf, then);
        if started.is_err() {
            self.front.pending.fetch_sub(1, Ordering::AcqRel);
        }
        started
    }

    fn read(&self, file: Handle, offset: u64, len: usize, tag: u64) -> io::Result<()> {
        let buf = IoBuf {
            bytes: vec![0; len],
            filled: 0,
        };
        self.submit(Request::Read { file, offset }, buf, io_then(tag))
    }

    fn write(&self, file: Handle, offset: u64, bytes: &[u8], tag: u64) -> io::Result<()> {
        let buf = IoBuf {
            bytes: bytes.to_vec(),
            filled: bytes.len(),
        };
        self.submit(Request::Write { file, offset }, buf, io_then(tag))
    }

    /// One poll with `stop`.
    fn pull_with(&self, stop: &mut dyn FnMut(&dyn Frame) -> bool) -> Vec<Done> {
        let mut out = Vec::new();
        self.front.reap(
            &self.engine,
            &mut |frame| {
                out.push(
                    *(frame as Box<dyn std::any::Any>)
                        .downcast::<Done>()
                        .expect("a Done frame"),
                )
            },
            stop,
            u64::MAX,
        );
        out
    }

    /// Polls until `count` completions came out (and checks nothing hangs).
    fn drain(&self, count: usize) -> Vec<Done> {
        let mut out = Vec::new();
        for _ in 0..1000 {
            out.extend(self.pull_with(&mut |_| false));
            if out.len() >= count {
                break;
            }
        }
        assert_eq!(out.len(), count, "the operations completed");
        out.sort_by_key(|done| done.header.id_hash());
        out
    }

    fn one(&self) -> Done {
        self.drain(1).pop().unwrap()
    }

    fn pending(&self) -> usize {
        self.front.pending()
    }
}

fn content(len: usize) -> Vec<u8> {
    (0..len).map(|i| (i * 7 + i / 251) as u8).collect()
}

fn ended(done: Done) -> IoDone {
    done.done.expect("an I/O completion")
}

#[test]
fn a_read_is_cut_into_pieces_and_ends_once_with_its_bytes() {
    let h = Harness::with(None, true, 4096);
    let data = content(3 * 4096 + 123);
    let file = h.sim.file(&data);
    h.read(file, 0, data.len(), 1).unwrap();
    let done = ended(h.one());
    assert_eq!(done.result.unwrap(), data.len());
    assert_eq!(done.buf.filled, data.len());
    assert_eq!(done.buf.bytes, data);
    let reads = h.sim.log().iter().filter(|l| l.starts_with("read")).count();
    assert_eq!(reads, 4, "pieces of at most 4096 bytes");
    assert_eq!(h.pending(), 0);
}

#[test]
fn a_short_transfer_is_continued_with_the_rest() {
    let h = Harness::new();
    h.sim.cap_transfers(1000);
    let data = content(5000);
    let file = h.sim.file(&data);
    h.read(file, 0, 5000, 1).unwrap();
    let done = ended(h.one());
    assert_eq!(done.result.unwrap(), 5000);
    assert_eq!(done.buf.bytes, data);
    let reads = h.sim.log().iter().filter(|l| l.starts_with("read")).count();
    assert_eq!(reads, 5);
}

#[test]
fn a_read_at_the_end_fails_and_one_that_ends_inside_returns_the_bytes() {
    let h = Harness::new();
    let data = content(100);
    let file = h.sim.file(&data);
    h.read(file, 8192, 4096, 1).unwrap();
    let error = ended(h.one()).result.unwrap_err();
    assert_eq!(error.kind(), io::ErrorKind::UnexpectedEof);
    h.read(file, 40, 4096, 2).unwrap();
    let done = ended(h.one());
    assert_eq!(done.result.unwrap(), 60);
    assert_eq!(&done.buf.bytes[..done.buf.filled], &data[40..]);
}

#[test]
fn a_call_that_fails_at_once_ends_with_its_error_and_the_rest_still_complete() {
    let h = Harness::new();
    let file = h.sim.file(&content(4096));
    h.sim
        .fail_next_call(Op::Read, io::Error::from_raw_os_error(1117));
    h.read(file, 0, 100, 1).unwrap();
    h.read(file, 0, 100, 2).unwrap();
    let mut done = h.drain(2);
    let second = ended(done.pop().unwrap());
    let first = ended(done.pop().unwrap());
    assert_eq!(first.result.unwrap_err().raw_os_error(), Some(1117));
    assert_eq!(second.result.unwrap(), 100);
    assert_eq!(h.pending(), 0);
}

#[test]
fn a_call_that_fails_in_its_completion_ends_with_that_error() {
    let h = Harness::new();
    let file = h.sim.file(&content(4096));
    h.sim
        .fail_next_completion(Op::Read, io::Error::from_raw_os_error(995));
    h.read(file, 0, 100, 1).unwrap();
    let error = ended(h.one()).result.unwrap_err();
    assert_eq!(error.raw_os_error(), Some(995));
}

#[test]
fn a_write_moves_every_byte_in_pieces_at_its_offset() {
    let h = Harness::with(None, true, 1000);
    let file = h.sim.file(&[9; 10]);
    let data = content(2500);
    h.write(file, 10, &data, 1).unwrap();
    let done = ended(h.one());
    assert_eq!(done.result.unwrap(), 2500);
    let mut expected = vec![9; 10];
    expected.extend_from_slice(&data);
    assert_eq!(h.sim.file_data(file), expected);
}

#[test]
fn a_handle_opened_without_overlapped_io_is_refused() {
    let h = Harness::new();
    let file = h.sim.synchronous_file(&content(100));
    let error = h.read(file, 0, 10, 1).unwrap_err();
    assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
    let error = h.write(file, 0, b"x", 1).unwrap_err();
    assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
    assert_eq!(h.pending(), 0, "nothing was submitted");
}

#[test]
fn every_handle_joins_the_port_once_before_its_first_operation() {
    let h = Harness::new();
    let file = h.sim.file(&content(100));
    h.read(file, 0, 10, 1).unwrap();
    h.read(file, 10, 10, 2).unwrap();
    h.drain(2);
    let associated = h
        .sim
        .log()
        .iter()
        .filter(|l| l.starts_with("associate"))
        .count();
    assert_eq!(associated, 1);
}

#[test]
fn sends_and_receives_move_bytes_in_pieces() {
    let h = Harness::with(None, true, 100);
    let sock = h.sim.sock();
    let data = content(250);
    let buf = IoBuf {
        bytes: data.clone(),
        filled: data.len(),
    };
    h.submit(Request::Send { sock }, buf, io_then(1)).unwrap();
    let done = ended(h.one());
    assert_eq!(done.result.unwrap(), 250);
    assert_eq!(h.sim.sent(sock), data);

    h.sim.feed(sock, b"hello");
    let buf = IoBuf {
        bytes: vec![0; 64],
        filled: 0,
    };
    h.submit(Request::Recv { sock }, buf, io_then(2)).unwrap();
    let done = ended(h.one());
    assert_eq!(done.result.unwrap(), 5);
    assert_eq!(&done.buf.bytes[..done.buf.filled], b"hello");

    // The peer's end of stream: a receive of nothing.
    let buf = IoBuf {
        bytes: vec![0; 64],
        filled: 0,
    };
    h.submit(Request::Recv { sock }, buf, io_then(3)).unwrap();
    let done = ended(h.one());
    assert_eq!(done.result.unwrap(), 0);
}

#[test]
fn a_datagram_is_one_operation_to_its_address() {
    let h = Harness::new();
    let sock = h.sim.sock();
    let to: SocketAddr = "127.0.0.1:9000".parse().unwrap();
    let buf = IoBuf {
        bytes: vec![5; 30],
        filled: 30,
    };
    h.submit(Request::SendTo { sock, to }, buf, io_then(1))
        .unwrap();
    let done = ended(h.one());
    // One operation: the call's result is the result, whatever it moved.
    assert_eq!(done.result.unwrap(), 30);
    assert!(h.sim.log().iter().any(|l| l == "send_to family 2"));
}

#[test]
fn an_accept_updates_the_context_and_yields_the_new_socket() {
    let h = Harness::new();
    let listener = h.sim.sock();
    h.submit(Request::Accept { listener }, IoBuf::default(), raw_then(1))
        .unwrap();
    let accepted = h.one().raw.unwrap().unwrap();
    let log = h.sim.log();
    assert!(log.iter().any(|l| l.starts_with("accept socket")));
    assert!(log.iter().any(|l| l.starts_with("update accept")));
    assert!(
        !h.sim.closed_sockets().contains(&accepted),
        "the caller owns it"
    );
}

#[test]
fn a_failed_accept_closes_the_socket_it_made() {
    for fail_update in [false, true] {
        let h = Harness::new();
        let listener = h.sim.sock();
        if fail_update {
            h.sim.fail_context_updates();
        } else {
            h.sim
                .fail_next_completion(Op::Accept, io::Error::from_raw_os_error(10054));
        }
        h.submit(Request::Accept { listener }, IoBuf::default(), raw_then(1))
            .unwrap();
        assert!(h.one().raw.unwrap().is_err());
        assert_eq!(
            h.sim.closed_sockets().len(),
            1,
            "fail_update = {fail_update}"
        );
    }
}

#[test]
fn a_connect_binds_connects_and_updates_the_context() {
    let h = Harness::new();
    let sock = h.sim.sock();
    let to: SocketAddr = "[::1]:80".parse().unwrap();
    h.submit(Request::Connect { sock, to }, IoBuf::default(), raw_then(1))
        .unwrap();
    assert_eq!(h.one().raw.unwrap().unwrap(), 0);
    let log = h.sim.log();
    let position = |what: &str| log.iter().position(|l| l.starts_with(what)).unwrap();
    assert!(position("bind") < position("connect"));
    assert!(position("connect") < position("update connect"));
    assert!(
        log.iter().any(|l| l.ends_with("family 23")),
        "bound to the IPv6 any"
    );

    h.sim
        .fail_next_completion(Op::Connect, io::Error::from_raw_os_error(10061));
    h.submit(Request::Connect { sock, to }, IoBuf::default(), raw_then(2))
        .unwrap();
    let error = h.one().raw.unwrap().unwrap_err();
    assert_eq!(
        error.raw_os_error(),
        Some(10061),
        "WSAECONNREFUSED, as Winsock says it"
    );
}

#[test]
fn open_and_rename_run_on_the_pool_and_come_out_of_the_poll() {
    let h = Harness::new();
    let dir = Path::new("scratch");
    h.sim.put_file(&dir.join("a"), b"abc");
    let spec = OpenSpec::default();
    h.submit(
        Request::Open {
            path: dir.join("a"),
            spec,
        },
        IoBuf::default(),
        raw_then(1),
    )
    .unwrap();
    assert_eq!(h.pending(), 1);
    let handle = h.one().raw.unwrap().unwrap();
    // The new handle is a file the engine can read.
    h.read(handle, 0, 3, 2).unwrap();
    assert_eq!(ended(h.one()).buf.bytes, b"abc");

    // A missing file is an error; creating one is not.
    h.submit(
        Request::Open {
            path: dir.join("b"),
            spec,
        },
        IoBuf::default(),
        raw_then(3),
    )
    .unwrap();
    let error = h.one().raw.unwrap().unwrap_err();
    assert_eq!(error.kind(), io::ErrorKind::NotFound);
    let create = OpenSpec {
        create: Create::New,
        write: true,
        ..spec
    };
    h.submit(
        Request::Open {
            path: dir.join("b"),
            spec: create,
        },
        IoBuf::default(),
        raw_then(4),
    )
    .unwrap();
    assert!(h.one().raw.unwrap().is_ok());

    h.submit(
        Request::Rename {
            from: dir.join("a"),
            to: dir.join("c"),
        },
        IoBuf::default(),
        raw_then(5),
    )
    .unwrap();
    assert!(h.one().raw.unwrap().is_ok());
    assert!(h.sim.path_exists(&dir.join("c")) && !h.sim.path_exists(&dir.join("a")));
    assert_eq!(h.pending(), 0);
}

#[test]
fn a_pool_that_refuses_the_job_ends_the_operation_with_an_error() {
    let h = Harness::new();
    h.sim.fail_pool();
    h.submit(
        Request::Rename {
            from: "x".into(),
            to: "y".into(),
        },
        IoBuf::default(),
        raw_then(1),
    )
    .unwrap();
    assert!(h.one().raw.unwrap().is_err());
    assert_eq!(h.pending(), 0);
}

#[test]
fn a_frame_the_pull_stops_before_is_held_for_a_later_poll_in_order() {
    let h = Harness::new();
    let file = h.sim.file(&content(4096));
    for tag in 1..=3 {
        h.read(file, 0, 10, tag).unwrap();
    }
    // Stop before tag 2: tag 1 goes out, 2 and what was reaped after it wait.
    let first = h.pull_with(&mut |frame| frame.header().id_hash() == 2);
    assert_eq!(
        first.iter().map(|d| d.header.id_hash()).collect::<Vec<_>>(),
        [1]
    );
    assert_eq!(h.pending(), 2);
    let rest = h.pull_with(&mut |_| false);
    assert_eq!(
        rest.iter().map(|d| d.header.id_hash()).collect::<Vec<_>>(),
        [2, 3]
    );
    assert_eq!(h.pending(), 0);
}

#[test]
fn a_packet_of_no_operation_is_ignored() {
    let h = Harness::new();
    let file = h.sim.file(&content(100));
    h.read(file, 0, 10, 1).unwrap();
    h.engine
        .sys
        .post(h.engine.port.handle, 0xdead_beef)
        .unwrap();
    let done = h.one();
    assert_eq!(ended(done).result.unwrap(), 10);
    assert!(h.pull_with(&mut |_| false).is_empty());
}

#[test]
fn a_piece_is_a_u32_within_the_largest_the_kernel_takes() {
    let h = Harness::new();
    assert_eq!(h.engine.piece(5usize << 30), MAX_PIECE as u32);
    assert_eq!(h.engine.piece(usize::MAX), MAX_PIECE as u32);
    assert_eq!(h.engine.piece(100), 100);
    let small = Harness::with(None, true, 4096);
    assert_eq!(small.engine.piece(1 << 20), 4096);
}

// ---- the ring

#[test]
fn with_a_kernel_ring_reads_and_writes_go_through_it_and_end_once() {
    let h = Harness::with(Some(full_probe()), true, 4096);
    assert_eq!(h.engine.ring_use(), (true, true));
    let data = content(2 * 4096 + 5);
    let file = h.sim.file(&data);
    h.read(file, 0, data.len(), 1).unwrap();
    let done = ended(h.one());
    assert_eq!(done.result.unwrap(), data.len());
    assert_eq!(done.buf.bytes, data);
    h.write(file, 0, b"ring", 2).unwrap();
    assert_eq!(ended(h.one()).result.unwrap(), 4);
    assert_eq!(&h.sim.file_data(file)[..4], b"ring");
    assert_eq!(h.sim.rings_made(), 1, "a ring per submitting thread");
    assert!(
        !h.sim.log().iter().any(|l| l.starts_with("associate")),
        "a ring needs no port"
    );
    assert_eq!(h.pending(), 0);
}

#[test]
fn a_read_at_the_end_through_the_ring_is_the_same_eof() {
    let h = Harness::with(Some(full_probe()), true, MAX_PIECE);
    let file = h.sim.file(&content(100));
    h.read(file, 500, 10, 1).unwrap();
    let error = ended(h.one()).result.unwrap_err();
    assert_eq!(error.kind(), io::ErrorKind::UnexpectedEof);
}

#[test]
fn the_ring_is_used_only_where_the_os_has_one_for_the_operation() {
    let probe = |version, emulated, read, write| {
        Some(RingProbe {
            version,
            emulated,
            read,
            write,
        })
    };
    let uses = |probe, ioring| Harness::with(probe, ioring, MAX_PIECE).engine.ring_use();
    assert_eq!(uses(None, true), (false, false), "no IoRing");
    assert_eq!(
        uses(probe(300, true, true, true), true),
        (false, false),
        "emulated"
    );
    assert_eq!(
        uses(probe(300, false, true, true), false),
        (false, false),
        "forced off"
    );
    assert_eq!(uses(probe(300, false, true, true), true), (true, true));
    assert_eq!(
        uses(probe(2, false, true, true), true),
        (true, false),
        "reads only before v3"
    );
    assert_eq!(
        uses(probe(300, false, true, false), true),
        (true, false),
        "no write op"
    );
    assert_eq!(uses(probe(300, false, false, true), true), (false, true));
}

#[test]
fn a_write_without_ring_support_goes_through_the_port_while_reads_use_the_ring() {
    let probe = RingProbe {
        version: 2,
        ..full_probe()
    };
    let h = Harness::with(Some(probe), true, MAX_PIECE);
    let file = h.sim.file(&content(100));
    h.read(file, 0, 10, 1).unwrap();
    h.write(file, 50, b"zz", 2).unwrap();
    let done = h.drain(2);
    assert!(done.iter().all(|d| d.done.as_ref().unwrap().result.is_ok()));
    let log = h.sim.log();
    assert_eq!(
        log.iter().filter(|l| l.starts_with("associate")).count(),
        1,
        "the write's"
    );
}

#[test]
fn a_full_submission_queue_waits_in_the_backlog_and_every_read_still_ends_once() {
    let h = Harness::with(Some(full_probe()), true, MAX_PIECE);
    h.sim.set_sq_limit(2);
    // The kernel refuses the first submits: built entries pile up, and the queue fills.
    h.sim.fail_submits(3);
    let data = content(4096);
    let file = h.sim.file(&data);
    for tag in 0..6 {
        h.read(file, tag * 100, 100, tag + 1).unwrap();
    }
    assert_eq!(h.pending(), 6);
    let done = h.drain(6);
    for (i, done) in done.into_iter().enumerate() {
        let done = ended(done);
        assert_eq!(done.result.unwrap(), 100);
        assert_eq!(done.buf.bytes, data[i * 100..(i + 1) * 100]);
    }
    assert_eq!(h.pending(), 0);
}

#[test]
fn completions_of_the_ring_and_the_port_each_come_out_once() {
    let h = Harness::with(Some(full_probe()), true, MAX_PIECE);
    let file = h.sim.file(&content(100));
    let sock = h.sim.sock();
    h.sim.hold(true);
    h.read(file, 0, 10, 1).unwrap();
    h.submit(
        Request::Send { sock },
        IoBuf {
            bytes: vec![1; 8],
            filled: 8,
        },
        io_then(2),
    )
    .unwrap();
    h.read(file, 10, 10, 3).unwrap();
    assert!(
        h.pull_with(&mut |_| false).is_empty(),
        "nothing completed yet"
    );
    assert_eq!(h.pending(), 3);
    h.sim.release();
    let done = h.drain(3);
    assert_eq!(
        done.iter().map(|d| d.header.id_hash()).collect::<Vec<_>>(),
        [1, 2, 3]
    );
    assert!(h.pull_with(&mut |_| false).is_empty());
    assert_eq!(h.pending(), 0);
}

#[test]
fn a_ring_completion_with_an_error_ends_that_operation_with_it() {
    let h = Harness::with(Some(full_probe()), true, MAX_PIECE);
    let file = h.sim.file(&content(100));
    h.sim
        .fail_next_completion(Op::Read, io::Error::from_raw_os_error(1450));
    h.read(file, 0, 10, 1).unwrap();
    let error = ended(h.one()).result.unwrap_err();
    assert_eq!(error.raw_os_error(), Some(1450));
}

// ---- the portable pieces

#[test]
fn socket_addresses_have_the_windows_layout() {
    let v4 = SockAddr::new("10.1.2.3:4660".parse().unwrap());
    assert_eq!(v4.len(), 16);
    assert_eq!(v4.family(), 2);
    // SAFETY: the address is 128 bytes.
    let bytes = unsafe { std::slice::from_raw_parts(v4.as_ptr(), 8) };
    assert_eq!(
        &bytes[2..8],
        &[0x12, 0x34, 10, 1, 2, 3],
        "port in network order, then the address"
    );

    let v6 = SockAddr::new("[::1]:80".parse().unwrap());
    assert_eq!(v6.len(), 28);
    assert_eq!(v6.family(), 23);
    let any = SockAddr::unspecified("[::1]:80".parse().unwrap());
    assert_eq!((any.len(), any.family()), (28, 23));
    let any = SockAddr::unspecified("1.2.3.4:80".parse().unwrap());
    assert_eq!((any.len(), any.family()), (16, 2));
}

#[test]
fn an_ioring_hresult_keeps_a_win32_code_and_otherwise_says_what_it_was() {
    // HRESULT_FROM_WIN32(ERROR_HANDLE_EOF)
    assert_eq!(
        hresult_error(0x8007_0026u32 as i32).raw_os_error(),
        Some(38)
    );
    assert_eq!(hresult_error(0x8007_0005u32 as i32).raw_os_error(), Some(5));
    let other = hresult_error(0x8000_4005u32 as i32);
    assert_eq!(other.raw_os_error(), None);
    assert!(other.to_string().contains("0x80004005"));
}
