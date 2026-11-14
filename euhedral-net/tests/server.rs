//! The server over loopback, on a real lattice and a real `AsyncIo`.
//!
//! Each test starts its own lattice, `AsyncIo` and sink, serves on a loopback listener and talks to it with plain
//! blocking sockets. Order is driven by what the peer does (a blocking read returns when the server has answered)
//! and by channels the handlers and the stop notification send on; nothing sleeps to synchronise. A time-limit test
//! asserts a lower bound (the connection did not end before its limit) and the outcome, never an upper bound. The
//! read timeouts on the peers are a failsafe against a hang, not a synchronisation.
//!
//! Every test ends with `Fixture::finish`: the server is stopped, and when its notification arrives no operation is
//! pending and no connection is counted.
#![cfg(any(target_os = "linux", target_os = "android", windows))]
#![allow(
    clippy::disallowed_types,
    clippy::disallowed_methods,
    reason = "test fixtures: the test thread is not a lattice worker; it connects, reads and waits on channels"
)]

use std::io::{ErrorKind, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use euhedral_core::config::{FragmentConfig, IdlePolicy, LatticeConfig};
use euhedral_core::control_plane::ControlPlaneLattice;
use euhedral_core::frames::{Frame, FrameError, FrameHeader};
use euhedral_core::impls::{BaseCloneableObject, DefaultExecutor};
use euhedral_core::ingest::{IngestSink, QueueIngestSink};
use euhedral_hardware_utils::affinity;
use euhedral_io::AsyncIo;
use euhedral_net::{
    Counters, Handler, Request, Responder, Response, ResponseParser, ServerConfig, ServerHandle,
    no_work, serve,
};
use euhedral_net::{Limits, Progress};

const PATIENCE: Duration = Duration::from_secs(20);

fn lattice() -> Arc<ControlPlaneLattice> {
    let allowed = affinity::base_mask();
    let cpus: Vec<usize> = allowed.iter().take(4).collect();
    assert!(!cpus.is_empty(), "this process may run on some CPU");
    let park = Duration::from_micros(100);
    let idle = IdlePolicy::fixed(park, IdlePolicy::DEFAULT_CONTENTION_HALF_LIFE)
        .unwrap()
        .with_unproductive_park(park);
    let fragment = FragmentConfig::default().with_idle_policy(idle);
    let mut config = LatticeConfig::new(Arc::new(BaseCloneableObject::new(
        fragment,
        DefaultExecutor::new(),
    )));
    config.name = "net-server-test".into();
    config.shard_name = "net-server-test".into();
    config.allowed_cpus = cpus.into_iter().collect();
    config.shutdown_timeout = Duration::from_secs(5);
    config.startup_timeout = Duration::from_secs(10);
    let lattice = ControlPlaneLattice::new(config).unwrap();
    lattice.start().unwrap();
    lattice
}

/// The stop notification: a frame that tells the test thread.
struct Notify {
    header: FrameHeader,
    tx: Sender<()>,
}

impl Frame for Notify {
    fn header(&self) -> &FrameHeader {
        &self.header
    }

    fn header_mut(&mut self) -> &mut FrameHeader {
        &mut self.header
    }

    fn execute(&mut self) -> Result<(), FrameError> {
        let _ = self.tx.send(());
        Ok(())
    }
}

struct Fixture {
    lattice: Arc<ControlPlaneLattice>,
    io: Arc<AsyncIo>,
    sink: Arc<QueueIngestSink>,
    handle: ServerHandle,
    addr: SocketAddr,
    stopped: Receiver<()>,
    notify: Sender<()>,
}

impl Fixture {
    fn start(config: ServerConfig, handler: impl Handler) -> Fixture {
        let lattice = lattice();
        let io = AsyncIo::open().unwrap();
        let sink = Arc::new(QueueIngestSink::new());
        lattice.add_upstream(io.clone()).unwrap();
        lattice.add_upstream(sink.delegate()).unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        assert_eq!(io.pending(), 0);
        let handle = serve(
            io.clone(),
            sink.clone(),
            listener,
            config,
            Arc::new(handler),
        )
        .unwrap();
        let (notify, stopped) = channel();
        Fixture {
            lattice,
            io,
            sink,
            handle,
            addr,
            stopped,
            notify,
        }
    }

    fn connect(&self) -> Peer {
        let stream = TcpStream::connect(self.addr).unwrap();
        stream.set_read_timeout(Some(PATIENCE)).unwrap();
        Peer {
            stream,
            buf: Vec::new(),
        }
    }

    /// Asks the server to stop and returns when it says it has.
    fn stop(&self) {
        self.handle.stop(Box::new(Notify {
            header: FrameHeader::new(7),
            tx: self.notify.clone(),
        }));
        self.stopped
            .recv_timeout(PATIENCE)
            .expect("the server did not report that it stopped");
    }

    /// Stops the server, checks that nothing is left, and returns the final counters.
    fn finish(self) -> Counters {
        self.stop();
        let counters = self.handle.counters();
        assert_eq!(counters.open, 0, "a connection is still counted");
        assert_eq!(self.io.pending(), 0, "an operation is still pending");
        assert!(
            self.stopped.try_recv().is_err(),
            "the stop was reported twice"
        );
        self.sink.complete_gracefully();
        self.lattice.close();
        counters
    }
}

/// A read that goes on after a signal interrupted it.
fn read_on(stream: &mut TcpStream, buf: &mut [u8]) -> std::io::Result<usize> {
    loop {
        match stream.read(buf) {
            Err(e) if e.kind() == ErrorKind::Interrupted => {}
            other => return other,
        }
    }
}

/// A blocking client.
struct Peer {
    stream: TcpStream,
    buf: Vec<u8>,
}

impl Peer {
    fn send(&mut self, bytes: &[u8]) {
        self.stream.write_all(bytes).unwrap();
    }

    /// Reads one response (to a HEAD request when `head`): head, and a body by its `Content-Length`.
    fn read_response(&mut self, head: bool) -> Response {
        let mut parser = ResponseParser::new(Limits {
            max_body: 64 << 20,
            ..Limits::default()
        });
        parser.set_response_to_head(head);
        let mut chunk = vec![0u8; 64 * 1024];
        loop {
            if !self.buf.is_empty() {
                let (used, progress) = parser.feed(&self.buf).expect("a response");
                self.buf.drain(..used);
                match progress {
                    Progress::Complete(response) => return response,
                    Progress::HeadComplete { .. } => continue,
                    Progress::NeedMore => {}
                }
            }
            let n = read_on(&mut self.stream, &mut chunk).expect("the server's answer");
            assert!(n > 0, "the server closed the connection before it answered");
            self.buf.extend_from_slice(&chunk[..n]);
        }
    }

    fn get(&mut self, target: &str) -> Response {
        self.send(format!("GET {target} HTTP/1.1\r\nHost: t\r\n\r\n").as_bytes());
        self.read_response(false)
    }

    /// The server closed the connection and sent nothing more.
    fn expect_closed(&mut self) {
        assert!(self.buf.is_empty(), "bytes the test did not expect");
        let mut chunk = [0u8; 64];
        match read_on(&mut self.stream, &mut chunk) {
            Ok(0) => {}
            Ok(n) => panic!("{n} unexpected bytes before the end"),
            // A reset is the connection gone too.
            Err(e) if e.kind() == ErrorKind::ConnectionReset => {}
            Err(e) => panic!("no end of stream: {e}"),
        }
    }
}

fn config() -> ServerConfig {
    ServerConfig {
        max_connections: 16,
        read_limit: None,
        idle_limit: None,
        write_limit: None,
        ..ServerConfig::default()
    }
}

/// Answers every request at once with its target as the body.
fn echo_target(request: Request, reply: Responder) -> Box<dyn Frame> {
    reply.send(Response::with_status(200).with_body(request.target.into_bytes()));
    no_work()
}

fn text(response: &Response) -> &str {
    std::str::from_utf8(&response.body).unwrap()
}

#[test]
fn one_request_gets_one_reply() {
    let server = Fixture::start(config(), echo_target);
    let mut peer = server.connect();
    let response = peer.get("/hello");
    assert_eq!(response.status, 200);
    assert_eq!(text(&response), "/hello");
    assert!(response.keep_alive());
    drop(peer);
    let counters = server.finish();
    assert_eq!(
        (
            counters.accepted,
            counters.requests,
            counters.replies_2xx,
            counters.parse_errors
        ),
        (1, 1, 1, 0)
    );
}

#[test]
fn a_keep_alive_connection_serves_request_after_request() {
    let server = Fixture::start(config(), echo_target);
    let mut peer = server.connect();
    for i in 0..5 {
        let target = format!("/{i}");
        assert_eq!(text(&peer.get(&target)), target);
    }
    drop(peer);
    let counters = server.finish();
    assert_eq!((counters.accepted, counters.requests), (1, 5));
}

#[test]
fn a_pipelined_pair_is_answered_in_order() {
    // The first answer comes from a later frame, the second at once: the order on the wire is still the
    // order of the requests.
    let handler = |request: Request, reply: Responder| -> Box<dyn Frame> {
        if request.target == "/first" {
            Box::new(Later {
                header: FrameHeader::new(99),
                reply: Some(reply),
                target: request.target,
            })
        } else {
            reply.send(Response::with_status(200).with_body(request.target.into_bytes()));
            no_work()
        }
    };
    let server = Fixture::start(config(), handler);
    let mut peer = server.connect();
    peer.send(b"GET /first HTTP/1.1\r\nHost: t\r\n\r\nGET /second HTTP/1.1\r\nHost: t\r\n\r\n");
    assert_eq!(text(&peer.read_response(false)), "/first");
    assert_eq!(text(&peer.read_response(false)), "/second");
    drop(peer);
    assert_eq!(server.finish().requests, 2);
}

/// A frame that answers from another frame than the handler's own.
struct Later {
    header: FrameHeader,
    reply: Option<Responder>,
    target: String,
}

impl Frame for Later {
    fn header(&self) -> &FrameHeader {
        &self.header
    }

    fn header_mut(&mut self) -> &mut FrameHeader {
        &mut self.header
    }

    fn execute(&mut self) -> Result<(), FrameError> {
        if let Some(reply) = self.reply.take() {
            reply.send(
                Response::with_status(200).with_body(std::mem::take(&mut self.target).into_bytes()),
            );
        }
        Ok(())
    }
}

#[test]
fn http_1_0_without_keep_alive_closes_and_with_it_stays() {
    let server = Fixture::start(config(), echo_target);
    let mut peer = server.connect();
    peer.send(b"GET /a HTTP/1.0\r\n\r\n");
    let response = peer.read_response(false);
    assert_eq!(text(&response), "/a");
    assert!(!response.keep_alive());
    peer.expect_closed();

    let mut peer = server.connect();
    peer.send(b"GET /b HTTP/1.0\r\nConnection: keep-alive\r\n\r\n");
    let response = peer.read_response(false);
    assert_eq!(response.headers.get("Connection"), Some("keep-alive"));
    peer.send(b"GET /c HTTP/1.0\r\n\r\n");
    assert_eq!(text(&peer.read_response(false)), "/c");
    peer.expect_closed();
    server.finish();
}

#[test]
fn connection_close_closes_after_the_reply() {
    let server = Fixture::start(config(), echo_target);
    let mut peer = server.connect();
    peer.send(b"GET /x HTTP/1.1\r\nHost: t\r\nConnection: close\r\n\r\n");
    let response = peer.read_response(false);
    assert_eq!(text(&response), "/x");
    assert_eq!(response.headers.get("Connection"), Some("close"));
    peer.expect_closed();
    server.finish();
}

#[test]
fn a_reply_may_be_sent_by_a_later_frame_that_hops_between_workers() {
    /// Hops through the sink a few times, each hop unordered, then answers.
    struct Hop {
        header: FrameHeader,
        sink: Arc<QueueIngestSink>,
        hops: usize,
        reply: Option<Responder>,
        seen: Arc<AtomicUsize>,
    }
    impl Frame for Hop {
        fn header(&self) -> &FrameHeader {
            &self.header
        }
        fn header_mut(&mut self) -> &mut FrameHeader {
            &mut self.header
        }
        fn execute(&mut self) -> Result<(), FrameError> {
            self.seen.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
        fn do_finally(mut self: Box<Self>) {
            if self.hops == 0 {
                if let Some(reply) = self.reply.take() {
                    reply.send(Response::with_status(201).with_body("late"));
                }
                return;
            }
            self.hops -= 1;
            self.header.randomize_hash(self.hops as u64 + 1);
            let sink = self.sink.clone();
            let _ = sink.offer(self);
        }
    }
    let sink_slot: Arc<OnceLock<Arc<QueueIngestSink>>> = Arc::new(OnceLock::new());
    let seen = Arc::new(AtomicUsize::new(0));
    let (slot, count) = (sink_slot.clone(), seen.clone());
    let handler = move |_request: Request, reply: Responder| -> Box<dyn Frame> {
        Box::new(Hop {
            header: FrameHeader::new(5),
            sink: slot
                .get()
                .expect("the sink is set before a request")
                .clone(),
            hops: 6,
            reply: Some(reply),
            seen: count.clone(),
        })
    };
    let server = Fixture::start(config(), handler);
    sink_slot.set(server.sink.clone()).ok().unwrap();
    let mut peer = server.connect();
    let response = peer.get("/");
    assert_eq!((response.status, text(&response)), (201, "late"));
    assert_eq!(seen.load(Ordering::SeqCst), 7);
    drop(peer);
    server.finish();
}

#[test]
fn a_dropped_responder_answers_500_exactly_once() {
    let handler = |request: Request, reply: Responder| -> Box<dyn Frame> {
        if request.target == "/drop" {
            drop(reply);
        } else {
            reply.send(Response::with_status(200));
        }
        no_work()
    };
    let server = Fixture::start(config(), handler);
    let mut peer = server.connect();
    let response = peer.get("/drop");
    assert_eq!(response.status, 500);
    // Nothing else follows on the connection: the next request's answer is the next thing it carries.
    assert_eq!(peer.get("/ok").status, 200);
    drop(peer);
    let counters = server.finish();
    assert_eq!((counters.replies_5xx, counters.replies_2xx), (1, 1));
}

#[test]
fn a_response_that_cannot_be_written_becomes_a_500() {
    let handler = |_request: Request, reply: Responder| -> Box<dyn Frame> {
        reply.send(Response::with_status(200).header("X", "a\r\nSet-Cookie: b"));
        no_work()
    };
    let server = Fixture::start(config(), handler);
    let mut peer = server.connect();
    assert_eq!(peer.get("/").status, 500);
    drop(peer);
    assert_eq!(server.finish().replies_5xx, 1);
}

#[test]
fn the_100_continue_is_sent_before_the_body_and_without_the_handler() {
    let server = Fixture::start(config(), |request: Request, reply: Responder| {
        reply.send(Response::with_status(200).with_body(request.body));
        no_work()
    });
    let mut peer = server.connect();
    peer.send(b"POST /up HTTP/1.1\r\nHost: t\r\nContent-Length: 5\r\nExpect: 100-continue\r\n\r\n");
    // The interim answer arrives while the body has not been sent: the handler cannot have been asked.
    let interim = peer.read_response(false);
    assert_eq!(interim.status, 100);
    assert_eq!(server.handle.counters().requests, 0);
    peer.send(b"he");
    peer.send(b"llo");
    let response = peer.read_response(false);
    assert_eq!((response.status, text(&response)), (200, "hello"));
    drop(peer);
    server.finish();
}

#[test]
fn a_head_request_gets_the_head_and_no_body() {
    let server = Fixture::start(config(), |_request: Request, reply: Responder| {
        reply.send(Response::with_status(200).with_body("twelve bytes"));
        no_work()
    });
    let mut peer = server.connect();
    peer.send(b"HEAD /h HTTP/1.1\r\nHost: t\r\n\r\n");
    let response = peer.read_response(true);
    assert_eq!(response.status, 200);
    assert_eq!(response.headers.get("Content-Length"), Some("12"));
    assert!(response.body.is_empty());
    // No body bytes follow: the next answer is the next thing on the stream.
    let next = peer.get("/g");
    assert_eq!(text(&next), "twelve bytes");
    drop(peer);
    server.finish();
}

#[test]
fn a_large_reply_goes_out_whole() {
    let body: Vec<u8> = (0..6 * 1024 * 1024).map(|i| (i % 251) as u8).collect();
    let expected = body.clone();
    let server = Fixture::start(config(), move |_request: Request, reply: Responder| {
        reply.send(Response::with_status(200).with_body(body.clone()));
        no_work()
    });
    let mut peer = server.connect();
    let response = peer.get("/");
    assert_eq!(response.body, expected);
    drop(peer);
    server.finish();
}

#[test]
fn each_parse_error_answers_its_status_and_closes() {
    let long_target = format!("GET /{} HTTP/1.1\r\nHost: t\r\n\r\n", "a".repeat(9000));
    let many_headers = {
        let mut request = String::from("GET / HTTP/1.1\r\nHost: t\r\n");
        for i in 0..2000 {
            request.push_str(&format!("X-{i}: v\r\n"));
        }
        request.push_str("\r\n");
        request
    };
    let cases: Vec<(&str, Vec<u8>, u16)> = vec![
        ("no host", b"GET / HTTP/1.1\r\n\r\n".to_vec(), 400),
        ("bare lf", b"GET / HTTP/1.1\nHost: t\n\n".to_vec(), 400),
        (
            "version",
            b"GET / HTTP/1.2\r\nHost: t\r\n\r\n".to_vec(),
            505,
        ),
        (
            "chunked",
            b"POST / HTTP/1.1\r\nHost: t\r\nTransfer-Encoding: chunked\r\n\r\n".to_vec(),
            411,
        ),
        ("long target", long_target.into_bytes(), 414),
        ("many headers", many_headers.into_bytes(), 431),
        (
            "long body",
            b"POST / HTTP/1.1\r\nHost: t\r\nContent-Length: 2000\r\n\r\n".to_vec(),
            413,
        ),
    ];
    let server = Fixture::start(
        ServerConfig {
            max_body_bytes: 1000,
            ..config()
        },
        echo_target,
    );
    for (name, bytes, status) in &cases {
        let mut peer = server.connect();
        peer.send(bytes);
        let response = peer.read_response(false);
        assert_eq!(response.status, *status, "{name}");
        assert_eq!(response.headers.get("Connection"), Some("close"), "{name}");
        peer.expect_closed();
    }
    let counters = server.finish();
    assert_eq!(counters.parse_errors, cases.len() as u64);
    assert_eq!(counters.requests, 0, "the handler is never asked");
    assert_eq!(counters.replies_4xx, 6);
    assert_eq!(counters.replies_5xx, 1);
}

#[test]
fn a_body_over_the_limit_is_refused_before_it_is_read() {
    let server = Fixture::start(
        ServerConfig {
            max_body_bytes: 10,
            ..config()
        },
        echo_target,
    );
    let mut peer = server.connect();
    // The head only, with an expectation: no `100 Continue`, the answer is the 413, and the body was never sent.
    peer.send(
        b"POST / HTTP/1.1\r\nHost: t\r\nContent-Length: 1000000\r\nExpect: 100-continue\r\n\r\n",
    );
    assert_eq!(peer.read_response(false).status, 413);
    peer.expect_closed();
    server.finish();
}

#[test]
fn at_the_cap_a_connection_waits_in_the_backlog_until_one_closes() {
    let server = Fixture::start(
        ServerConfig {
            max_connections: 2,
            ..config()
        },
        echo_target,
    );
    let mut first = server.connect();
    let mut second = server.connect();
    assert_eq!(text(&first.get("/1")), "/1");
    assert_eq!(text(&second.get("/2")), "/2");
    // The kernel completes the handshake of a third connection, but the server has no accept pending.
    let mut third = server.connect();
    third.send(b"GET /3 HTTP/1.1\r\nHost: t\r\n\r\n");
    third
        .stream
        .set_read_timeout(Some(Duration::from_millis(200)))
        .unwrap();
    let mut byte = [0u8; 1];
    let err = read_on(&mut third.stream, &mut byte).unwrap_err();
    assert!(matches!(
        err.kind(),
        ErrorKind::WouldBlock | ErrorKind::TimedOut
    ));
    let counters = server.handle.counters();
    assert_eq!((counters.accepted, counters.open), (2, 2));
    // One ends, and the third is taken in and answered.
    drop(first);
    third.stream.set_read_timeout(Some(PATIENCE)).unwrap();
    assert_eq!(text(&third.read_response(false)), "/3");
    assert_eq!(server.handle.counters().accepted, 3);
    drop((second, third));
    server.finish();
}

#[test]
fn a_request_that_is_too_slow_to_arrive_is_answered_408_and_closed() {
    let limit = Duration::from_millis(150);
    let server = Fixture::start(
        ServerConfig {
            read_limit: Some(limit),
            ..config()
        },
        echo_target,
    );
    let mut peer = server.connect();
    let started = Instant::now();
    peer.send(b"GET / HT");
    let response = peer.read_response(false);
    assert_eq!(response.status, 408);
    assert!(started.elapsed() >= limit, "the limit did not pass");
    peer.expect_closed();
    let counters = server.finish();
    assert_eq!((counters.limits_passed, counters.requests), (1, 0));
}

#[test]
fn an_idle_connection_is_closed_when_the_idle_limit_passes_and_a_busy_one_is_not() {
    let limit = Duration::from_millis(150);
    let server = Fixture::start(
        ServerConfig {
            idle_limit: Some(limit),
            ..config()
        },
        echo_target,
    );
    let mut peer = server.connect();
    let started = Instant::now();
    assert_eq!(text(&peer.get("/1")), "/1");
    peer.expect_closed();
    assert!(started.elapsed() >= limit, "the limit did not pass");
    let counters = server.finish();
    assert_eq!((counters.limits_passed, counters.requests), (1, 1));
}

#[test]
fn a_peer_that_stops_reading_passes_the_write_limit_and_the_connection_closes() {
    let limit = Duration::from_millis(150);
    let size = 64 * 1024 * 1024;
    let (called_tx, called) = channel::<()>();
    let handler = move |_request: Request, reply: Responder| -> Box<dyn Frame> {
        reply.send(Response::with_status(200).with_body(vec![b'x'; size]));
        let _ = called_tx.send(());
        no_work()
    };
    let server = Fixture::start(
        ServerConfig {
            write_limit: Some(limit),
            ..config()
        },
        handler,
    );
    let mut peer = server.connect();
    let started = Instant::now();
    peer.send(b"GET / HTTP/1.1\r\nHost: t\r\n\r\n");
    called.recv_timeout(PATIENCE).unwrap();
    // The peer reads nothing. The stop waits for the reply, which can only end at its limit.
    server.stop();
    assert!(started.elapsed() >= limit, "the limit did not pass");
    let mut got = 0usize;
    let mut chunk = vec![0u8; 256 * 1024];
    loop {
        match read_on(&mut peer.stream, &mut chunk) {
            Ok(0) => break,
            Ok(n) => got += n,
            Err(e) if e.kind() == ErrorKind::ConnectionReset => break,
            Err(e) => panic!("{e}"),
        }
    }
    assert!(got < size, "the whole reply went out");
    let counters = server.finish();
    assert_eq!(counters.limits_passed, 1);
}

/// Loops through the sink until the cancel flag is set, then answers (late, to nobody).
struct WatchCancel {
    header: FrameHeader,
    sink: Arc<QueueIngestSink>,
    reply: Option<Responder>,
    seen: Sender<bool>,
}

impl Frame for WatchCancel {
    fn header(&self) -> &FrameHeader {
        &self.header
    }

    fn header_mut(&mut self) -> &mut FrameHeader {
        &mut self.header
    }

    fn do_finally(mut self: Box<Self>) {
        let cancelled = self
            .reply
            .as_ref()
            .is_some_and(|r| r.cancelled().load(Ordering::Acquire));
        if cancelled {
            let reply = self.reply.take().unwrap();
            reply.send(Response::with_status(200).with_body("too late"));
            let _ = self.seen.send(true);
            return;
        }
        let sink = self.sink.clone();
        let _ = sink.offer(self);
    }
}

#[test]
fn a_peer_that_closes_mid_request_sets_the_cancel_flag_and_the_late_reply_is_discarded() {
    let sink_slot: Arc<OnceLock<Arc<QueueIngestSink>>> = Arc::new(OnceLock::new());
    let (called_tx, called) = channel::<()>();
    let (seen_tx, seen) = channel::<bool>();
    let slot = sink_slot.clone();
    let handler = move |_request: Request, reply: Responder| -> Box<dyn Frame> {
        let _ = called_tx.send(());
        let mut header = FrameHeader::new(1);
        header.randomize_hash(3);
        Box::new(WatchCancel {
            header,
            sink: slot.get().unwrap().clone(),
            reply: Some(reply),
            seen: seen_tx.clone(),
        })
    };
    let server = Fixture::start(config(), handler);
    sink_slot.set(server.sink.clone()).ok().unwrap();
    let mut peer = server.connect();
    peer.send(b"GET /slow HTTP/1.1\r\nHost: t\r\n\r\n");
    called.recv_timeout(PATIENCE).unwrap();
    drop(peer);
    assert!(seen.recv_timeout(PATIENCE).unwrap());
    let counters = server.finish();
    assert_eq!(counters.cancelled, 1);
    assert_eq!(counters.discarded, 1);
    assert_eq!(
        counters.replies_2xx + counters.replies_4xx + counters.replies_5xx,
        0
    );
}

#[test]
fn stop_closes_idle_connections_and_reports_once() {
    let server = Fixture::start(config(), echo_target);
    let mut peers = Vec::new();
    for i in 0..3 {
        let mut peer = server.connect();
        assert_eq!(text(&peer.get(&format!("/{i}"))), format!("/{i}"));
        peers.push(peer);
    }
    // A connection that has sent half a request is closed too.
    let mut partial = server.connect();
    partial.send(b"GET /par");
    server.stop();
    for peer in &mut peers {
        peer.expect_closed();
    }
    partial.expect_closed();
    // A second stop on a stopped server reports again, at once, as its own frame.
    server.stop();
    let counters = server.handle.counters();
    assert_eq!(counters.open, 0);
    server.sink.complete_gracefully();
    server.lattice.close();
}

#[test]
fn stop_lets_a_request_in_flight_finish_and_then_closes_its_connection() {
    let sink_slot: Arc<OnceLock<Arc<QueueIngestSink>>> = Arc::new(OnceLock::new());
    let (called_tx, called) = channel::<()>();
    let gate = Arc::new(AtomicBool::new(false));
    struct Gated {
        header: FrameHeader,
        sink: Arc<QueueIngestSink>,
        gate: Arc<AtomicBool>,
        reply: Option<Responder>,
    }
    impl Frame for Gated {
        fn header(&self) -> &FrameHeader {
            &self.header
        }
        fn header_mut(&mut self) -> &mut FrameHeader {
            &mut self.header
        }
        fn do_finally(mut self: Box<Self>) {
            if self.gate.load(Ordering::Acquire) {
                self.reply
                    .take()
                    .unwrap()
                    .send(Response::with_status(200).with_body("finished"));
                return;
            }
            let sink = self.sink.clone();
            let _ = sink.offer(self);
        }
    }
    let (slot, g) = (sink_slot.clone(), gate.clone());
    let handler = move |_request: Request, reply: Responder| -> Box<dyn Frame> {
        let _ = called_tx.send(());
        let mut header = FrameHeader::new(1);
        header.randomize_hash(4);
        Box::new(Gated {
            header,
            sink: slot.get().unwrap().clone(),
            gate: g.clone(),
            reply: Some(reply),
        })
    };
    let server = Fixture::start(config(), handler);
    sink_slot.set(server.sink.clone()).ok().unwrap();
    let mut busy = server.connect();
    busy.send(b"GET /busy HTTP/1.1\r\nHost: t\r\n\r\n");
    called.recv_timeout(PATIENCE).unwrap();
    // The stop is asked for while the request is in the handler's hands; the reply is released afterwards.
    server.handle.stop(Box::new(Notify {
        header: FrameHeader::new(7),
        tx: server.notify.clone(),
    }));
    gate.store(true, Ordering::Release);
    let response = busy.read_response(false);
    assert_eq!((response.status, text(&response)), (200, "finished"));
    busy.expect_closed();
    // The server lingers until the peer closes too, as a client does on the end of the stream.
    drop(busy);
    server
        .stopped
        .recv_timeout(PATIENCE)
        .expect("the server did not report that it stopped");
    let counters = server.handle.counters();
    assert_eq!((counters.open, counters.replies_2xx), (0, 1));
    assert_eq!(server.io.pending(), 0);
    assert!(server.stopped.try_recv().is_err());
    server.sink.complete_gracefully();
    server.lattice.close();
}
