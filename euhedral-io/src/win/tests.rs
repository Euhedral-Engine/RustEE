//! The engine's state machines against the simulated `Sys`: every operation's success, error and
//! short transfers, the ring and its fallbacks, a full submission queue, held-back frames.

use std::io;
use std::net::SocketAddr;
use std::path::Path;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

use euhedral_core::frames::{Frame, FrameHeader};

use super::sim::{Op, Sim};
use super::sys::{
    Handle, IORING_VERSION_3, RingProbe, SockAddr, hresult_error, socket_status_error,
    verbatim_path,
};
use super::{Engine, MAX_PIECE};
use crate::{
    AsyncIoOptions, Buf, Create, Engine as EngineTrait, Front, IoBuf, IoDone, IoDoneOwned,
    OpenSpec, Region, Request, Then,
};

/// A completion frame: carries what the operation ended with.
struct Done {
    header: FrameHeader,
    done: Option<IoDone>,
    raw: Option<io::Result<usize>>,
    owned: Option<IoDoneOwned<Vec<u8>>>,
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
            owned: None,
        })
    }))
}

/// A read into caller-owned memory: the region of `memory`, and the continuation that holds it.
fn owned_read(mut memory: Vec<u8>, tag: u64) -> (Buf, Then) {
    let region = Region::of(&mut memory);
    let then = Then::Owned(Box::new(move |result, filled, leaked| {
        let buf = if leaked {
            std::mem::forget(memory);
            None
        } else {
            Some(memory)
        };
        Box::new(Done {
            header: FrameHeader::new(tag),
            done: None,
            raw: None,
            owned: Some(IoDoneOwned {
                buf,
                result,
                filled,
            }),
        })
    }));
    (Buf::Region(region), then)
}

fn raw_then(tag: u64) -> Then {
    Then::Raw(Box::new(move |result| {
        Box::new(Done {
            header: FrameHeader::new(tag),
            done: None,
            raw: Some(result),
            owned: None,
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
    engine: Arc<Engine>,
    front: Arc<Front>,
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
            engine: Arc::new(engine),
            front: Arc::new(Front::new()),
        }
    }

    fn submit(&self, request: Request, buf: IoBuf, then: Then) -> io::Result<()> {
        self.front.pending.fetch_add(1, Ordering::AcqRel);
        let started = self.engine.submit(request, buf.into(), then);
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

    fn read_owned(&self, file: Handle, offset: u64, memory: Vec<u8>, tag: u64) -> io::Result<()> {
        let (buf, then) = owned_read(memory, tag);
        self.front.pending.fetch_add(1, Ordering::AcqRel);
        let started = self
            .engine
            .submit(Request::Read { file, offset }, buf, then);
        if started.is_err() {
            self.front.pending.fetch_sub(1, Ordering::AcqRel);
        }
        started
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
            &*self.engine,
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

    /// One poll that takes at most `demand` completions.
    fn pull_n(&self, demand: u64) -> Vec<Done> {
        let mut out = Vec::new();
        self.front.reap(
            &*self.engine,
            &mut |frame| {
                out.push(
                    *(frame as Box<dyn std::any::Any>)
                        .downcast::<Done>()
                        .expect("a Done frame"),
                )
            },
            &mut |_| false,
            demand,
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
fn the_ring_is_used_only_where_the_os_has_one_for_reads_and_writes() {
    let probe = |version, emulated, read, write| {
        Some(RingProbe {
            version,
            emulated,
            read,
            write,
        })
    };
    let uses = |probe, ioring| Harness::with(probe, ioring, MAX_PIECE).engine.ring_use().0;
    assert!(!uses(None, true), "no IoRing");
    assert!(!uses(probe(300, true, true, true), true), "emulated");
    assert!(!uses(probe(300, false, true, true), false), "forced off");
    assert!(uses(probe(300, false, true, true), true));
    assert!(
        !uses(probe(2, false, true, true), true),
        "no writes before v3"
    );
    assert!(!uses(probe(300, false, true, false), true), "no write op");
    assert!(!uses(probe(300, false, false, true), true), "no read op");
}

#[test]
fn without_ring_writes_files_go_through_the_port_for_reads_and_writes_alike() {
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
    assert_eq!(h.sim.rings_made(), 0);
    let log = h.sim.log();
    assert_eq!(
        log.iter().filter(|l| l.starts_with("associate")).count(),
        1,
        "joined once"
    );
}

#[test]
fn a_file_keeps_its_route() {
    let h = Harness::with(Some(full_probe()), true, MAX_PIECE);
    // A file that joined the port stays on it, ring or not.
    let joined = h.sim.file(&content(100));
    h.engine
        .sys
        .associate(h.engine.port.handle, joined)
        .unwrap();
    h.engine.associated.lock().unwrap().insert(joined);
    h.read(joined, 0, 10, 1).unwrap();
    assert_eq!(ended(h.one()).result.unwrap(), 10);
    assert_eq!(h.sim.rings_made(), 0, "no ring for a file on the port");
    // A file a ring serves is never joined to the port.
    let ringed = h.sim.file(&content(100));
    h.read(ringed, 0, 10, 2).unwrap();
    h.write(ringed, 0, b"x", 3).unwrap();
    h.drain(2);
    assert_eq!(h.sim.rings_made(), 1);
    assert!(
        !h.sim
            .log()
            .iter()
            .any(|l| l == &format!("associate {ringed:#x}")),
        "a ring-served file never joins the port"
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

// ---- identity, late packets and the review's cases

/// The address of the `OVERLAPPED` of slot 0.
fn first_overlapped(h: &Harness) -> usize {
    lock!(&h.engine.table).slots[0].rec.0.as_ptr() as usize
}

#[test]
fn a_call_that_fails_with_a_warning_status_and_queues_a_packet_ends_exactly_once() {
    let h = Harness::new();
    let file = h.sim.file(&content(100));
    // The call reports failure (ERROR_MORE_DATA) and a packet follows anyway: the case the real
    // layer reads as "a packet follows"; here the sim misbehaves to check that, even then, the
    // operation ends once and the late packet is ignored.
    h.sim.fail_with_packet(Op::Read, 234);
    h.read(file, 0, 10, 1).unwrap();
    assert_eq!(h.sim.packets_waiting(), 1);
    let done = h.drain(1);
    assert_eq!(ended_clone_error(&done[0]), Some(234));
    for _ in 0..5 {
        assert!(h.pull_with(&mut |_| false).is_empty(), "nothing ends twice");
    }
    assert_eq!(h.pending(), 0);
    assert_eq!(
        h.engine.strays.load(Ordering::Relaxed),
        1,
        "the late packet was ignored"
    );
    // The slot is usable again.
    h.read(file, 0, 10, 2).unwrap();
    assert_eq!(ended(h.one()).result.unwrap(), 10);
}

fn ended_clone_error(done: &Done) -> Option<i32> {
    done.done
        .as_ref()
        .unwrap()
        .result
        .as_ref()
        .err()
        .and_then(io::Error::raw_os_error)
}

#[test]
fn a_late_packet_never_ends_the_operation_that_reused_its_slot() {
    let h = Harness::new();
    let file = h.sim.file(&content(100));
    h.sim.fail_with_packet(Op::Read, 234);
    h.read(file, 0, 10, 1).unwrap();
    // End the first operation, but leave its packet undelivered (the poll has no room for it).
    let first = h.pull_n(1);
    assert_eq!(ended_clone_error(&first[0]), Some(234));
    assert_eq!(
        h.sim.packets_waiting() + lock!(&h.engine.inbox).len(),
        1,
        "the late packet waits"
    );
    // The second operation takes the same slot; its call is pending (its completion is held).
    h.sim.hold(true);
    h.read(file, 20, 10, 2).unwrap();
    assert!(
        h.pull_with(&mut |_| false).is_empty(),
        "the late packet met a pending call"
    );
    assert_eq!(h.engine.strays.load(Ordering::Relaxed), 1);
    h.sim.release();
    let done = ended(h.one());
    assert_eq!(done.result.unwrap(), 10);
    assert_eq!(done.buf.bytes, content(100)[20..30]);
    assert!(h.pull_with(&mut |_| false).is_empty());
    assert_eq!(h.pending(), 0);
}

#[test]
fn a_packet_that_arrives_while_its_call_is_returning_waits_for_the_next_poll() {
    let h = Harness::new();
    let file = h.sim.file(&content(100));
    let seen: Arc<Mutex<Vec<usize>>> = Arc::default();
    let (engine, front, log) = (h.engine.clone(), h.front.clone(), seen.clone());
    // A poll on another thread, in the middle of the read call.
    h.sim.on_read(Arc::new(move || {
        let mut got = 0;
        front.reap(&*engine, &mut |_| got += 1, &mut |_| false, u64::MAX);
        log.lock().unwrap().push(got);
    }));
    h.read(file, 0, 10, 1).unwrap();
    assert_eq!(
        *seen.lock().unwrap(),
        [0],
        "the operation was not ended under its call"
    );
    h.sim.on_read(Arc::new(|| {}));
    assert_eq!(ended(h.one()).result.unwrap(), 10);
    assert_eq!(h.pending(), 0);
}

#[test]
fn a_pool_job_that_panics_or_is_dropped_still_ends_its_operation() {
    let h = Harness::new();
    h.sim.panic_next_open();
    h.submit(
        Request::Open {
            path: "x".into(),
            spec: OpenSpec::default(),
        },
        IoBuf::default(),
        raw_then(1),
    )
    .unwrap();
    let error = h.one().raw.unwrap().unwrap_err();
    assert!(error.to_string().contains("did not finish"), "{error}");
    assert_eq!(h.pending(), 0);
}

#[test]
fn a_handle_that_belongs_to_another_port_is_refused() {
    let h = Harness::new();
    let file = h.sim.file(&content(100));
    h.sim.joined_elsewhere(file);
    let error = h.read(file, 0, 10, 1).unwrap_err();
    assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
    assert!(error.to_string().contains("another completion port"));
    assert_eq!(h.pending(), 0);
}

#[test]
fn handles_that_would_block_or_skip_the_port_are_refused() {
    let h = Harness::new();
    let skipping = h.sim.skipping_file(&content(10));
    assert_eq!(
        h.read(skipping, 0, 10, 1).unwrap_err().kind(),
        io::ErrorKind::InvalidInput
    );
    let sock = h.sim.synchronous_sock();
    let buf = IoBuf {
        bytes: vec![0; 4],
        filled: 0,
    };
    let error = h
        .submit(Request::Recv { sock }, buf, io_then(2))
        .unwrap_err();
    assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
    assert_eq!(h.pending(), 0);
}

#[test]
fn a_ring_completion_or_a_packet_ends_only_a_record_on_its_own_route() {
    let h = Harness::with(Some(full_probe()), true, MAX_PIECE);
    let file = h.sim.file(&content(100));
    h.sim.hold(true);
    h.read(file, 0, 10, 1).unwrap();
    // A packet naming the ring-routed record's OVERLAPPED does not end it.
    h.engine
        .sys
        .post(h.engine.port.handle, first_overlapped(&h))
        .unwrap();
    assert!(h.pull_with(&mut |_| false).is_empty());
    assert_eq!(h.engine.strays.load(Ordering::Relaxed), 1);
    // A ring completion with the wrong generation, or on a record not routed through the ring,
    // ends nothing either.
    let ring = lock!(&h.engine.rings)[0].clone();
    assert!(
        h.engine.take_token(0, 1 + 7, Some(&ring)).is_none(),
        "wrong generation"
    );
    assert!(
        h.engine.take_token(0, 1, None).is_none(),
        "a pool result for a ring record"
    );
    h.sim.release();
    assert_eq!(ended(h.one()).result.unwrap(), 10);
    assert_eq!(h.pending(), 0);

    // And a port record is not ended by a ring completion naming it.
    let sock = h.sim.sock();
    h.sim.hold(true);
    h.submit(
        Request::Send { sock },
        IoBuf {
            bytes: vec![1; 4],
            filled: 4,
        },
        io_then(2),
    )
    .unwrap();
    assert!(
        h.engine.take_token(0, 2, Some(&ring)).is_none(),
        "a port record"
    );
    h.sim.release();
    assert_eq!(ended(h.one()).result.unwrap(), 4);
}

#[test]
fn a_write_goes_in_pieces_of_at_most_a_read_piece() {
    let h = Harness::new();
    let file = h.sim.file(&[]);
    let data = content(600 * 1024);
    h.write(file, 0, &data, 1).unwrap();
    assert_eq!(ended(h.one()).result.unwrap(), data.len());
    let writes: Vec<_> = h
        .sim
        .log()
        .into_iter()
        .filter(|l| l.starts_with("write"))
        .collect();
    assert_eq!(writes.len(), 3, "{writes:?}");
    assert_eq!(h.sim.file_data(file), data);
}

#[test]
fn a_flush_builds_a_bounded_number_of_entries_and_a_submission_never_waits_for_one() {
    let h = Harness::with(Some(full_probe()), true, MAX_PIECE);
    let data = content(4096);
    let file = h.sim.file(&data);
    h.read(file, 0, 10, 99).unwrap(); // makes this thread's ring
    h.drain(1);
    let ring = lock!(&h.engine.rings)[0].clone();
    let busy = lock!(&ring.inner);
    // Another thread is submitting: this one only queues its entries (it must not wait).
    for tag in 0..40u64 {
        h.read(file, tag * 10, 10, tag).unwrap();
    }
    drop(busy);
    let done = h.drain(40);
    for (i, done) in done.into_iter().enumerate() {
        let done = ended(done);
        assert_eq!(done.buf.bytes, data[i * 10..(i + 1) * 10]);
    }
    let sizes = h.sim.submit_sizes();
    assert!(sizes.iter().all(|n| *n <= 16), "{sizes:?}");
    assert_eq!(sizes.iter().sum::<usize>(), 41);
}

#[test]
fn a_ring_that_cannot_be_read_ends_its_operations_with_the_error() {
    let h = Harness::with(Some(full_probe()), true, MAX_PIECE);
    let file = h.sim.file(&content(100));
    h.sim.hold(true);
    h.read(file, 0, 10, 1).unwrap();
    h.read(file, 10, 10, 2).unwrap();
    h.sim.fail_pops(5);
    let done = h.drain(2);
    for done in done {
        let error = ended(done).result.unwrap_err();
        assert!(error.to_string().contains("IoRing failed"), "{error}");
    }
    assert_eq!(h.pending(), 0);
    // The thread gets a new ring for what follows.
    h.sim.fail_pops(0);
    h.sim.hold(false);
    h.sim.release();
    h.read(file, 0, 10, 3).unwrap();
    assert_eq!(ended(h.one()).result.unwrap(), 10);
    assert_eq!(h.sim.rings_made(), 2);
}

#[test]
fn a_truncated_datagram_ends_once_with_its_winsock_error() {
    let h = Harness::new();
    let sock = h.sim.sock();
    h.sim.mark_warning(10040);
    h.sim
        .fail_next_completion(Op::Recv, io::Error::from_raw_os_error(10040));
    let buf = IoBuf {
        bytes: vec![0; 4],
        filled: 0,
    };
    h.submit(Request::Recv { sock }, buf, io_then(1)).unwrap();
    let error = ended(h.one()).result.unwrap_err();
    assert_eq!(error.raw_os_error(), Some(10040));
    assert!(h.pull_with(&mut |_| false).is_empty());
}

#[test]
fn socket_statuses_map_to_their_winsock_errors_and_other_statuses_are_left_alone() {
    assert_eq!(socket_status_error(0xC000_0236_u32 as i32), Some(10061));
    assert_eq!(socket_status_error(0xC000_020D_u32 as i32), Some(10054));
    assert_eq!(socket_status_error(0x8000_0005_u32 as i32), Some(10040));
    assert_eq!(
        socket_status_error(0xC000_0120_u32 as i32),
        None,
        "cancelled is the Win32 995"
    );
}

#[test]
fn long_paths_become_verbatim_paths_and_others_are_left_alone() {
    assert_eq!(verbatim_path(r"C:\a\b").as_deref(), Some(r"\\?\C:\a\b"));
    assert_eq!(
        verbatim_path(r"\\server\share\f").as_deref(),
        Some(r"\\?\UNC\server\share\f")
    );
    assert_eq!(verbatim_path(r"\\?\C:\a"), None);
    assert_eq!(verbatim_path(r"relative\path"), None);
}

// ---- the second review's cases

#[test]
fn a_ring_that_breaks_while_an_issuer_is_mid_call_cannot_touch_its_record() {
    let h = Harness::with(Some(full_probe()), true, MAX_PIECE);
    let file = h.sim.file(&content(100));
    h.read(file, 0, 10, 1).unwrap(); // makes this thread's ring
    h.drain(1);
    let ring = h.engine.ring().unwrap();
    // An issuer that has begun (its slot is `Issuing`) and has not yet pushed its entry.
    let bytes = vec![0u8; 10];
    let buf = IoBuf { bytes, filled: 0 };
    let op = Engine::op(
        super::Kind::Read { file, offset: 0 },
        super::Route::Ring(ring.clone()),
        buf.into(),
        io_then(2),
    );
    let held = h.engine.begin(op, super::State::Issuing).unwrap();
    h.front.pending.fetch_add(1, Ordering::AcqRel);
    // The ring breaks now: the issuer's operation and its buffer are none of its business.
    h.engine.break_ring(&ring, &io::Error::from_raw_os_error(5));
    // SAFETY: the slot is `Issuing`: this test thread has the record.
    let len = unsafe { (*held.rec.as_ptr()).op.as_ref().unwrap().buf.len() };
    assert_eq!(len, 10, "the buffer was left alone");
    assert!(
        h.pull_with(&mut |_| false).is_empty(),
        "and the operation was not ended"
    );
    // The issuer goes on and hands its entry to the broken ring: the operation ends with an
    // error instead of hanging.
    h.engine.launch(held);
    let error = ended(h.one()).result.unwrap_err();
    assert!(error.to_string().contains("IoRing failed"), "{error}");
    assert_eq!(h.pending(), 0);
}

#[test]
fn an_inbox_of_packets_for_returning_calls_does_not_stall_the_other_completions() {
    let h = Harness::new();
    let file = h.sim.file(&content(100));
    // B is pending until released.
    h.sim.hold(true);
    h.read(file, 50, 10, 2).unwrap();
    h.sim.hold(false);
    let seen: Arc<Mutex<Vec<usize>>> = Arc::default();
    let (engine, front, sim, log) = (
        h.engine.clone(),
        h.front.clone(),
        h.sim.clone(),
        seen.clone(),
    );
    // While A's call returns: a poll sees A's packet (and sets it aside), B completes, and a
    // second poll must still end B.
    h.sim.on_read(Arc::new(move || {
        let count = |front: &Front| {
            let mut got = 0;
            front.reap(&*engine, &mut |_| got += 1, &mut |_| false, u64::MAX);
            got
        };
        let first = count(&front);
        sim.release();
        let second = count(&front);
        log.lock().unwrap().extend([first, second]);
    }));
    h.read(file, 0, 10, 1).unwrap();
    assert_eq!(
        *seen.lock().unwrap(),
        [0, 1],
        "B ended while A's packet waited"
    );
    h.sim.on_read(Arc::new(|| {}));
    let last = h.one();
    assert_eq!(last.header.id_hash(), 1);
    assert_eq!(ended(last).result.unwrap(), 10);
}

#[test]
fn a_packet_whose_count_disagrees_with_the_call_is_ignored() {
    let h = Harness::new();
    let file = h.sim.file(&content(100));
    h.read(file, 0, 10, 1).unwrap();
    h.one();
    let slot0 = first_overlapped(&h);
    // A packet naming the slot with a count that is not the next call's.
    h.engine.sys.post(h.engine.port.handle, slot0).unwrap();
    h.read(file, 10, 10, 2).unwrap();
    let done = ended(h.one());
    assert_eq!(done.result.unwrap(), 10);
    assert_eq!(done.buf.bytes, content(100)[10..20]);
    assert_eq!(h.engine.strays.load(Ordering::Relaxed), 1);
}

#[test]
fn a_receive_that_fails_with_a_warning_keeps_the_bytes_that_arrived() {
    let h = Harness::new();
    let sock = h.sim.sock();
    h.sim.mark_warning(10040);
    h.sim.feed(sock, &[9; 100]);
    h.sim
        .fail_next_completion(Op::Recv, io::Error::from_raw_os_error(10040));
    let buf = IoBuf {
        bytes: vec![0; 10],
        filled: 0,
    };
    h.submit(Request::Recv { sock }, buf, io_then(1)).unwrap();
    let done = ended(h.one());
    assert_eq!(done.result.unwrap_err().raw_os_error(), Some(10040));
    assert_eq!(done.buf.filled, 10, "the datagram's first bytes arrived");
    assert_eq!(done.buf.bytes, [9; 10]);
}

#[test]
fn an_operation_ended_by_the_issuers_recheck_of_a_broken_ring_leaks_its_buffer() {
    let h = Harness::with(Some(full_probe()), true, MAX_PIECE);
    let file = h.sim.file(&content(100));
    h.read(file, 0, 10, 1).unwrap(); // makes this thread's ring
    h.drain(1);
    let ring = h.engine.ring().unwrap();
    let op = Engine::op(
        super::Kind::Read { file, offset: 0 },
        super::Route::Ring(ring.clone()),
        IoBuf {
            bytes: vec![0u8; 10],
            filled: 0,
        }
        .into(),
        io_then(2),
    );
    let held = h.engine.begin(op, super::State::Issuing).unwrap();
    h.front.pending.fetch_add(1, Ordering::AcqRel);
    // The issuer's half of `launch`: its slot goes in flight and its entry is pushed ...
    h.engine.set_state(held, super::State::InFlight);
    let entry = super::Entry {
        write: false,
        file,
        // SAFETY: the slot is `InFlight` with nothing else touching the record in this test.
        buf: unsafe {
            (*held.rec.as_ptr())
                .op
                .as_mut()
                .unwrap()
                .buf
                .bytes_mut()
                .as_mut_ptr() as usize
        },
        len: 10,
        offset: 0,
        user_data: super::token(held.slot, held.generation),
    };
    lock!(&ring.backlog).push_back(entry);
    // ... and a concurrent flush (which saw the ring healthy) builds it: the kernel has it.
    h.engine.flush_ring(&ring);
    assert_eq!(
        h.sim.submit_sizes().last(),
        Some(&1),
        "the entry reached the kernel"
    );
    // Then the ring breaks, and the issuer's recheck reaches the slot before `break_ring`'s scan.
    ring.broken.store(true, Ordering::Release);
    h.engine.end_if_broken(&ring, entry.user_data);
    let done = ended(h.one());
    assert!(done.result.is_err());
    assert!(
        done.buf.bytes.is_empty(),
        "the buffer the kernel may still write was not handed back"
    );
    assert_eq!(h.pending(), 0);
}

// ---- reads into caller-owned memory

fn owned(done: Done) -> IoDoneOwned<Vec<u8>> {
    done.owned.expect("a read into owned memory")
}

#[test]
fn an_owned_read_is_cut_into_pieces_that_land_at_their_offsets() {
    for ioring in [false, true] {
        let h = Harness::with(Some(full_probe()), ioring, 4096);
        let data = content(3 * 4096 + 123);
        let file = h.sim.file(&data);
        h.read_owned(file, 0, vec![0; data.len()], 1).unwrap();
        let done = owned(h.one());
        assert_eq!(done.result.unwrap(), data.len());
        assert_eq!(done.buf.unwrap(), data);
        let reads = h.sim.log().iter().filter(|l| l.starts_with("read")).count();
        assert!(reads >= 4, "pieces of at most 4096 bytes: {reads}");
        assert_eq!(h.pending(), 0);
    }
}

#[test]
fn an_owned_read_that_ends_inside_the_file_reports_its_count_and_one_past_it_fails_with_the_memory_back()
 {
    let h = Harness::new();
    let data = content(100);
    let file = h.sim.file(&data);
    h.read_owned(file, 40, vec![0; 4096], 1).unwrap();
    let done = owned(h.one());
    assert_eq!(done.result.unwrap(), 60);
    assert_eq!(done.filled, 60);
    let buf = done.buf.unwrap();
    assert_eq!(buf.len(), 4096, "the whole memory comes back");
    assert_eq!(&buf[..60], &data[40..]);
    h.read_owned(file, 8192, vec![7; 64], 2).unwrap();
    let done = owned(h.one());
    assert_eq!(
        done.result.unwrap_err().kind(),
        io::ErrorKind::UnexpectedEof
    );
    assert_eq!(
        done.buf.unwrap(),
        [7; 64],
        "and so does the memory of a failed read"
    );
}

#[test]
fn an_owned_read_whose_ring_broke_leaks_its_memory_and_its_continuation_says_so() {
    use std::sync::atomic::AtomicBool;

    /// Tells when it is dropped.
    struct Watched(Vec<u8>, Arc<AtomicBool>);
    impl Drop for Watched {
        fn drop(&mut self) {
            self.1.store(true, Ordering::Release);
        }
    }

    let h = Harness::with(Some(full_probe()), true, MAX_PIECE);
    let file = h.sim.file(&content(100));
    h.read(file, 0, 10, 1).unwrap(); // makes this thread's ring
    h.drain(1);
    let ring = h.engine.ring().unwrap();
    let dropped = Arc::new(AtomicBool::new(false));
    let ran: Arc<Mutex<Option<(bool, bool)>>> = Arc::default();
    let mut memory = Watched(vec![0u8; 10], dropped.clone());
    let region = Region::of(&mut memory.0);
    let ran_with = ran.clone();
    let then = Then::Owned(Box::new(move |result, _, leaked| {
        let kept = if leaked {
            std::mem::forget(memory);
            None
        } else {
            Some(memory)
        };
        *ran_with.lock().unwrap() = Some((result.is_err(), kept.is_none()));
        Box::new(Done {
            header: FrameHeader::new(2),
            done: None,
            raw: None,
            owned: None,
        })
    }));
    let op = Engine::op(
        super::Kind::Read { file, offset: 0 },
        super::Route::Ring(ring.clone()),
        Buf::Region(region),
        then,
    );
    let held = h.engine.begin(op, super::State::Issuing).unwrap();
    h.front.pending.fetch_add(1, Ordering::AcqRel);
    h.engine.set_state(held, super::State::InFlight);
    let entry = super::Entry {
        write: false,
        file,
        // SAFETY: the slot is `InFlight` with nothing else touching the record in this test.
        buf: unsafe {
            (*held.rec.as_ptr())
                .op
                .as_mut()
                .unwrap()
                .buf
                .bytes_mut()
                .as_mut_ptr() as usize
        },
        len: 10,
        offset: 0,
        user_data: super::token(held.slot, held.generation),
    };
    lock!(&ring.backlog).push_back(entry);
    h.engine.flush_ring(&ring);
    ring.broken.store(true, Ordering::Release);
    h.engine.end_if_broken(&ring, entry.user_data);
    h.drain(1);
    assert_eq!(
        *ran.lock().unwrap(),
        Some((true, true)),
        "the continuation ran with the ring's failure and no memory"
    );
    assert!(
        !dropped.load(Ordering::Acquire),
        "the memory the kernel may still write was not dropped"
    );
    assert_eq!(h.pending(), 0);
}
