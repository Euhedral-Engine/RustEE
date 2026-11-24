//! The server over loopback, on a real lattice and a real `AsyncIo`.
//!
//! Each test starts its own lattice, `AsyncIo` and sink, serves on a loopback listener and talks to it with plain
//! blocking sockets. Order is driven by what the peer does (a blocking read returns when the server has answered)
//! and by channels the handlers and the stop notification send on; nothing sleeps to synchronise. A handler that
//! answers later parks its `Responder` in a channel, and the test releases it by offering a frame that sends. A
//! time-limit test asserts a lower bound (the connection did not end before its limit) and the outcome, never an
//! upper bound. The read timeouts on the peers are a failsafe against a hang, not a synchronisation. The one
//! thing a test waits for by looking is a counter of the failing-accept test (`until`), which has no event to wait on.
//! A handler learns that its peer went by the frame it gave `Responder::on_cancel`, which tells the test thread. A
//! thing that must not happen is shown by a state read at a gate that the server's own order puts after it, never by
//! a timed wait.
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
use std::net::{Shutdown, SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, OnceLock};
use std::thread::sleep;
use std::thread::yield_now;
use std::time::{Duration, Instant};

use euhedral_core::config::{FragmentConfig, IdlePolicy, LatticeConfig};
use euhedral_core::control_plane::ControlPlaneLattice;
use euhedral_core::frames::{Frame, FrameError, FrameHeader};
use euhedral_core::impls::{BaseCloneableObject, DefaultExecutor};
use euhedral_core::ingest::{IngestSink, QueueIngestSink};
use euhedral_graph::Completions;
use euhedral_hardware_utils::affinity;
use euhedral_io::AsyncIo;
use euhedral_net::{
    Counters, Handler, PieceSent, ReplyStream, Request, Responder, Response, ResponseParser,
    ServerConfig, ServerHandle, StreamError, no_work, serve,
};
use euhedral_net::{Limits, Progress};

/// What a parking handler holds: the request's target and the right to answer it.
type Parked = (String, Responder);

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

/// A frame that tells the test thread it ran.
fn signal() -> (Box<dyn Frame>, Receiver<()>) {
    let (tx, rx) = channel();
    let frame = Notify {
        header: FrameHeader::new(9),
        tx,
    };
    (Box::new(frame), rx)
}

/// A frame that says whether it ran or was dropped without running.
struct Spy {
    header: FrameHeader,
    tx: Sender<&'static str>,
    ran: bool,
}

impl Spy {
    fn pair() -> (Box<dyn Frame>, Receiver<&'static str>) {
        let (tx, rx) = channel();
        let spy = Spy {
            header: FrameHeader::new(11),
            tx,
            ran: false,
        };
        (Box::new(spy), rx)
    }
}

impl Frame for Spy {
    fn header(&self) -> &FrameHeader {
        &self.header
    }

    fn header_mut(&mut self) -> &mut FrameHeader {
        &mut self.header
    }

    fn execute(&mut self) -> Result<(), FrameError> {
        self.ran = true;
        let _ = self.tx.send("ran");
        Ok(())
    }
}

impl Drop for Spy {
    fn drop(&mut self) {
        if !self.ran {
            let _ = self.tx.send("dropped");
        }
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
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        Fixture::start_on(listener, config, handler)
    }

    fn start_on(listener: TcpListener, config: ServerConfig, handler: impl Handler) -> Fixture {
        let lattice = lattice();
        let io = AsyncIo::open().unwrap();
        let sink = Arc::new(QueueIngestSink::new());
        let timers = Completions::without_lanes();
        lattice.add_upstream(io.clone()).unwrap();
        lattice.add_upstream(sink.delegate()).unwrap();
        lattice.add_upstream(timers.clone()).unwrap();
        let addr = listener.local_addr().unwrap();
        assert_eq!(io.pending(), 0);
        let handle = serve(
            lattice.clone(),
            io.clone(),
            sink.clone(),
            listener,
            config,
            Arc::new(handler),
            timers,
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

    /// Asks the server to stop and returns at once: when it returns, the request is recorded, so a reply sent
    /// after it is sent after the stop.
    fn begin_stop(&self) {
        self.handle.stop(Box::new(Notify {
            header: FrameHeader::new(7),
            tx: self.notify.clone(),
        }));
    }

    /// Asks the server to stop and returns when it says it has.
    fn stop(&self) {
        self.begin_stop();
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

/// A read that goes on after it was interrupted. The cause of the interruptions seen on the test threads of this
/// machine was not found (nothing in this workspace installs a signal handler, and the server does no I/O on the
/// test thread), so the one retry stays; `std` retries `Interrupted` for `read_exact` but not for `read`. A likely
/// cause, not confirmed: a read timeout set on the test socket (`SO_RCVTIMEO`) makes the kernel return `EINTR` from
/// a read that a signal reaches, instead of restarting it.
fn read_on(stream: &mut TcpStream, buf: &mut [u8]) -> std::io::Result<usize> {
    loop {
        match stream.read(buf) {
            Err(e) if e.kind() == ErrorKind::Interrupted => {}
            other => return other,
        }
    }
}

/// A peek that goes on after it was interrupted (see `read_on`).
fn peek_on(stream: &mut TcpStream, buf: &mut [u8]) {
    loop {
        match stream.peek(buf) {
            Err(e) if e.kind() == ErrorKind::Interrupted => {}
            other => {
                other.expect("the data to peek at");
                return;
            }
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

    /// The next bytes on the connection are exactly `expected`. A byte that differs fails at once, without waiting
    /// for the rest.
    fn expect_wire(&mut self, expected: &[u8]) {
        let mut chunk = vec![0u8; 64 * 1024];
        while self.buf.len() < expected.len() {
            let seen = self.buf.len().min(expected.len());
            assert_eq!(
                String::from_utf8_lossy(&self.buf[..seen]),
                String::from_utf8_lossy(&expected[..seen]),
                "the wire differs from what was expected"
            );
            let n = read_on(&mut self.stream, &mut chunk).expect("the server's bytes");
            assert!(
                n > 0,
                "the server closed the connection after {:?}",
                String::from_utf8_lossy(&self.buf)
            );
            self.buf.extend_from_slice(&chunk[..n]);
        }
        assert_eq!(
            String::from_utf8_lossy(&self.buf[..expected.len()]),
            String::from_utf8_lossy(expected),
            "the wire differs from what was expected"
        );
        self.buf.drain(..expected.len());
    }

    /// Everything up to the end of the stream (or a reset).
    fn read_to_end(&mut self) -> Vec<u8> {
        let mut all = std::mem::take(&mut self.buf);
        let mut chunk = vec![0u8; 64 * 1024];
        loop {
            match read_on(&mut self.stream, &mut chunk) {
                Ok(0) => return all,
                Ok(n) => all.extend_from_slice(&chunk[..n]),
                Err(e) if e.kind() == ErrorKind::ConnectionReset => return all,
                Err(e) => panic!("no end of stream: {e}"),
            }
        }
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

/// A handler that answers `/ok` at once and parks everything else: the test thread gets the `Responder` through
/// the channel and answers (or drops it) when it chooses. `/boom` panics.
fn parking() -> (impl Handler, Receiver<Parked>) {
    let (tx, rx) = channel::<Parked>();
    let handler = move |request: Request, reply: Responder| -> Box<dyn Frame> {
        match request.target.as_str() {
            "/ok" => reply.send(Response::with_status(200).with_body("ok")),
            "/boom" => panic!("the handler failed"),
            _ => {
                let _ = tx.send((request.target, reply));
            }
        }
        no_work()
    };
    (handler, rx)
}

/// Waits for a condition that the server alone makes true and that has no event to wait on.
fn until(what: &str, mut condition: impl FnMut() -> bool) {
    let limit = Instant::now() + PATIENCE;
    while !condition() {
        assert!(Instant::now() < limit, "{what}: not seen");
        yield_now();
    }
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
    let (handler, parked) = parking();
    let server = Fixture::start(
        ServerConfig {
            max_connections: 2,
            ..config()
        },
        handler,
    );
    // Two connections hold a request each: each has one `recv` in flight, and the server has no accept pending.
    let mut first = server.connect();
    let mut second = server.connect();
    first.send(b"GET /1 HTTP/1.1\r\nHost: t\r\nConnection: close\r\n\r\n");
    second.send(b"GET /2 HTTP/1.1\r\nHost: t\r\n\r\n");
    let mut replies = Vec::new();
    for _ in 0..2 {
        replies.push(parked.recv_timeout(PATIENCE).unwrap());
    }
    // The kernel completes the handshake of a third connection, but the cap holds: the state is read at a gate
    // that both requests being handled provide (an accept would be one more operation in flight).
    let mut third = server.connect();
    third.send(b"GET /ok HTTP/1.1\r\nHost: t\r\n\r\n");
    let counters = server.handle.counters();
    assert_eq!((counters.accepted, counters.open), (2, 2));
    assert_eq!(server.io.pending(), 2, "an accept is pending at the cap");
    // One ends, and the next accept follows: the third is taken in and answered.
    // The two handlers run in parallel and in no fixed order: the reply is picked by its target.
    let at = replies
        .iter()
        .position(|(target, _)| target == "/1")
        .unwrap();
    let (_, reply) = replies.remove(at);
    reply.send(Response::with_status(200));
    assert_eq!(first.read_response(false).status, 200);
    first.expect_closed();
    drop(first);
    assert_eq!(text(&third.read_response(false)), "ok");
    assert_eq!(server.handle.counters().accepted, 3);
    let (_, reply) = replies.remove(0);
    reply.send(Response::with_status(200));
    assert_eq!(second.read_response(false).status, 200);
    drop((second, third));
    server.finish();
}

/// The most connects a stop test has open that the server may not have taken: far below the listen backlog (128
/// for a `TcpListener` of the standard library), so the backlog never fills.
const STOP_BATCH: usize = 32;

#[test]
fn a_stopped_server_refuses_new_connections_while_a_request_drains() {
    let (handler, parked) = parking();
    let server = Fixture::start(config(), handler);
    let mut held = server.connect();
    held.send(b"GET /held HTTP/1.1\r\nHost: t\r\n\r\n");
    let (_, reply) = parked.recv_timeout(PATIENCE).unwrap();
    server.begin_stop();
    // The listener is closed once the cancelled accept has ended: a connect is refused from then on. A connect
    // that lands before that is taken in and closed by the server, or waits in the backlog and is reset by the
    // listener's close. The connects go in batches far below the listen backlog, and each batch is read to its end
    // before the next starts, so the backlog never fills: at a full backlog the kernel may drop the last step of a
    // handshake that the peer already saw complete, and then neither the server (which never had the connection)
    // nor the close of the listener sends that peer anything.
    let limit = Instant::now() + PATIENCE;
    let never =
        "the listening socket was never closed: new connections are still accepted after the stop";
    let mut refused = false;
    while !refused {
        let mut batch = Vec::new();
        while batch.len() < STOP_BATCH {
            let remaining = limit.saturating_duration_since(Instant::now());
            assert!(!remaining.is_zero(), "{never}");
            match TcpStream::connect_timeout(&server.addr, remaining) {
                Ok(stream) => batch.push(stream),
                Err(error) if error.kind() == ErrorKind::TimedOut => panic!("{never}"),
                // A connect still in flight when the socket closes is reset; the next one is refused.
                Err(error) if error.kind() == ErrorKind::ConnectionReset => {}
                Err(error) => {
                    assert_eq!(error.kind(), ErrorKind::ConnectionRefused);
                    refused = true;
                    break;
                }
            }
            yield_now();
        }
        // A refusal alone could come from a full backlog: every connect that got in was closed.
        for stream in &mut batch {
            let remaining = limit.saturating_duration_since(Instant::now());
            assert!(
                !remaining.is_zero(),
                "a connection that got in before the stop was never closed"
            );
            stream.set_read_timeout(Some(remaining)).unwrap();
            let mut byte = [0u8; 1];
            match read_on(stream, &mut byte) {
                Ok(0) => {}
                Err(error) if error.kind() == ErrorKind::ConnectionReset => {}
                other => {
                    let counters = server.handle.counters();
                    panic!(
                        "a connection that got in before the stop was neither closed by the server nor reset \
                         by the listener's close: {other:?} (refused yet: {refused}; server accepted {}, open {}; \
                         {:?} to {:?})",
                        counters.accepted,
                        counters.open,
                        stream.local_addr(),
                        stream.peer_addr(),
                    );
                }
            }
        }
    }
    // The request in flight is answered, and its connection closes.
    reply.send(Response::with_status(200));
    assert_eq!(held.read_response(false).status, 200);
    held.expect_closed();
    drop(held);
    server
        .stopped
        .recv_timeout(PATIENCE)
        .expect("the server did not finish after its last connection closed");
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

#[test]
fn a_peer_that_breaks_mid_request_sets_the_cancel_flag_and_the_late_reply_is_discarded() {
    let (handler, parked) = parking();
    let server = Fixture::start(config(), handler);
    let mut peer = server.connect();
    assert_eq!(text(&peer.get("/ok")), "ok");
    // The answer to the second request is never read, and the first answer is left unread too: closing a socket
    // that holds unread data resets the connection.
    peer.send(b"GET /ok HTTP/1.1\r\nHost: t\r\n\r\nGET /slow HTTP/1.1\r\nHost: t\r\n\r\n");
    let (_, reply) = parked.recv_timeout(PATIENCE).unwrap();
    let (woken, cancelled) = signal();
    reply.on_cancel(woken);
    let mut probe = [0u8; 1];
    peek_on(&mut peer.stream, &mut probe);
    drop(peer);
    cancelled
        .recv_timeout(PATIENCE)
        .expect("the frame for the cancel did not run");
    assert!(reply.cancelled().load(Ordering::Acquire));
    reply.send(Response::with_status(200).with_body("too late"));
    let counters = server.finish();
    assert_eq!((counters.cancelled, counters.discarded), (1, 1));
    assert_eq!(counters.replies_2xx, 2, "the two answers that were sent");
    assert_eq!(counters.replies_4xx + counters.replies_5xx, 0);
}

#[test]
fn a_half_closed_peer_still_gets_its_reply_and_then_the_connection_closes() {
    for (name, request) in [
        ("1.0", &b"GET /h HTTP/1.0\r\n\r\n"[..]),
        ("1.1", &b"GET /h HTTP/1.1\r\nHost: t\r\n\r\n"[..]),
    ] {
        let (handler, parked) = parking();
        let server = Fixture::start(config(), handler);
        let mut peer = server.connect();
        peer.send(request);
        let (_, reply) = parked.recv_timeout(PATIENCE).unwrap();
        let (woken, cancelled) = signal();
        reply.on_cancel(woken);
        peer.stream.shutdown(Shutdown::Write).unwrap();
        // The flag is a hint that the peer's end of the stream was seen; the answer is still owed.
        cancelled
            .recv_timeout(PATIENCE)
            .expect("the frame for the cancel did not run");
        assert!(reply.cancelled().load(Ordering::Acquire));
        // A frame asked for after the cancel was seen runs at once (the request is still being handled).
        let (late, late_ran) = signal();
        reply.on_cancel(late);
        late_ran
            .recv_timeout(PATIENCE)
            .expect("the late frame for the cancel did not run");
        reply.send(Response::with_status(200).with_body("still here"));
        let response = peer.read_response(false);
        assert_eq!(
            (response.status, text(&response)),
            (200, "still here"),
            "{name}"
        );
        assert_eq!(response.headers.get("Connection"), Some("close"), "{name}");
        peer.expect_closed();
        let counters = server.finish();
        assert_eq!(
            (counters.replies_2xx, counters.cancelled, counters.discarded),
            (1, 1, 0),
            "{name}"
        );
    }
}

#[test]
fn stop_lets_a_request_in_flight_finish_and_then_closes_its_connection() {
    let (handler, parked) = parking();
    let server = Fixture::start(config(), handler);
    let mut busy = server.connect();
    busy.send(b"GET /busy HTTP/1.1\r\nHost: t\r\n\r\n");
    let (_, reply) = parked.recv_timeout(PATIENCE).unwrap();
    // The stop is recorded before the reply is sent, so the order is forced: the request began as a keep-alive
    // one, and its answer ends the connection.
    server.begin_stop();
    reply.send(Response::with_status(200).with_body("finished"));
    let response = busy.read_response(false);
    assert_eq!((response.status, text(&response)), (200, "finished"));
    assert_eq!(response.headers.get("Connection"), Some("close"));
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

#[test]
fn stop_is_not_reported_while_a_reply_is_still_owed_even_to_a_peer_that_left() {
    let (handler, parked) = parking();
    let server = Fixture::start(config(), handler);
    let mut peer = server.connect();
    assert_eq!(text(&peer.get("/ok")), "ok");
    peer.send(b"GET /ok HTTP/1.1\r\nHost: t\r\n\r\nGET /slow HTTP/1.1\r\nHost: t\r\n\r\n");
    let (_, reply) = parked.recv_timeout(PATIENCE).unwrap();
    let (woken, cancelled) = signal();
    reply.on_cancel(woken);
    // Unread data makes the close a reset: the connection ends while the handler still owes its answer.
    let mut probe = [0u8; 1];
    peek_on(&mut peer.stream, &mut probe);
    drop(peer);
    // The gate: the server has seen the peer go, so its connection has closed its socket and holds only what the
    // owed answer needs.
    cancelled
        .recv_timeout(PATIENCE)
        .expect("the frame for the cancel did not run");
    server.begin_stop();
    // State read at the gate: the connection is still counted, so the stop has nothing to report yet. (It is the
    // reply, not a wait, that lets the server finish: the next lines send it.)
    assert_eq!(server.handle.counters().open, 1);
    assert!(server.stopped.try_recv().is_err());
    reply.send(Response::with_status(200));
    server
        .stopped
        .recv_timeout(PATIENCE)
        .expect("the server did not report that it stopped");
    // The late reply was counted before the notification: nothing runs after it.
    assert_eq!(server.handle.counters().discarded, 1);
    assert_eq!(server.handle.counters().open, 0);
    server.sink.complete_gracefully();
    server.lattice.close();
}

#[test]
fn stop_after_the_100_continue_ends_the_connection_and_the_body_never_reaches_the_handler() {
    let (handler, _parked) = parking();
    let server = Fixture::start(config(), handler);
    let mut peer = server.connect();
    peer.send(b"POST /up HTTP/1.1\r\nHost: t\r\nContent-Length: 5\r\nExpect: 100-continue\r\n\r\n");
    // The interim answer is out: the connection waits for the body.
    let mut interim = [0u8; 25];
    let mut got = 0;
    while got < interim.len() {
        let n = read_on(&mut peer.stream, &mut interim[got..]).expect("the interim answer");
        assert!(n > 0, "closed before the interim answer");
        got += n;
        if interim[..got].ends_with(b"\r\n\r\n") {
            break;
        }
    }
    assert!(interim[..got].starts_with(b"HTTP/1.1 100"));
    // The stop is recorded before the body is sent, so the order is forced: the body arrives after the stop.
    server.begin_stop();
    let _ = peer.stream.write_all(b"hello");
    peer.expect_closed();
    drop(peer);
    server
        .stopped
        .recv_timeout(PATIENCE)
        .expect("the server did not report that it stopped");
    assert_eq!(server.handle.counters().requests, 0);
    server.sink.complete_gracefully();
    server.lattice.close();
}

#[test]
fn a_cancel_frame_asked_for_after_the_peer_broke_the_connection_still_runs_once() {
    let (handler, parked) = parking();
    let server = Fixture::start(config(), handler);
    let mut peer = server.connect();
    assert_eq!(text(&peer.get("/ok")), "ok");
    peer.send(b"GET /ok HTTP/1.1\r\nHost: t\r\n\r\nGET /slow HTTP/1.1\r\nHost: t\r\n\r\n");
    let (_, reply) = parked.recv_timeout(PATIENCE).unwrap();
    // The first frame is the gate: when it ran, the server has seen the peer go, and its connection is closed and
    // waits for the answer only. A frame asked for now must still run.
    let (seen, gone) = signal();
    reply.on_cancel(seen);
    let mut probe = [0u8; 1];
    peek_on(&mut peer.stream, &mut probe);
    drop(peer);
    gone.recv_timeout(PATIENCE)
        .expect("the cancel was not seen");
    let (spy, report) = Spy::pair();
    reply.on_cancel(spy);
    assert_eq!(report.recv_timeout(PATIENCE), Ok("ran"));
    reply.send(Response::with_status(200));
    server.finish();
    assert!(
        report.try_recv().is_err(),
        "the frame was dropped or ran twice"
    );
}

#[test]
fn a_cancel_frame_is_dropped_without_running_when_the_reply_comes_first() {
    let (handler, parked) = parking();
    let server = Fixture::start(config(), handler);
    let mut peer = server.connect();
    peer.send(b"GET /slow HTTP/1.1\r\nHost: t\r\n\r\n");
    let (_, reply) = parked.recv_timeout(PATIENCE).unwrap();
    let (spy, report) = Spy::pair();
    reply.on_cancel(spy);
    reply.send(Response::with_status(200));
    assert_eq!(peer.read_response(false).status, 200);
    // The registration was ahead of the reply in the connection's order, so the frame was dropped by it.
    assert_eq!(report.recv_timeout(PATIENCE), Ok("dropped"));
    drop(peer);
    server.finish();
    assert!(report.try_recv().is_err(), "the frame ran after all");
}

#[test]
fn a_peer_that_drips_after_the_reply_is_closed_at_the_linger_deadline() {
    let limit = Duration::from_millis(400);
    let server = Fixture::start(
        ServerConfig {
            linger_limit: limit,
            // `write_limit` is `None` in `config()`: the linger is bounded by its own setting.
            ..config()
        },
        echo_target,
    );
    let mut peer = server.connect();
    let started = Instant::now();
    peer.send(b"GET /a HTTP/1.0\r\n\r\n");
    assert_eq!(text(&peer.read_response(false)), "/a");
    // The server has closed its side and reads what the peer still sends. A byte every few milliseconds renews
    // every single read, so only the overall deadline can end this. The sleep paces the peer, it synchronises
    // nothing.
    peer.stream
        .set_read_timeout(Some(Duration::from_millis(5)))
        .unwrap();
    let mut chunk = [0u8; 16];
    let ended = loop {
        assert!(
            started.elapsed() < PATIENCE,
            "the connection was never closed"
        );
        // The end of the stream (the server shut its side down) is not the close: a write or read that fails is.
        if peer.stream.write_all(b"x").is_err() {
            break Instant::now();
        }
        match peer.stream.read(&mut chunk) {
            Err(e) if e.kind() == ErrorKind::ConnectionReset => break Instant::now(),
            // After the shutdown the read returns at once: pace the drip (a flood would spend the linger's byte
            // budget, which is another way to end it).
            Ok(0) => sleep(Duration::from_millis(5)),
            _ => {}
        }
    };
    assert!(
        ended - started >= limit,
        "closed before the deadline: {:?}",
        ended - started
    );
    drop(peer);
    server.finish();
}

#[test]
fn a_stop_does_not_wait_for_a_peer_that_never_closes() {
    let (handler, parked) = parking();
    let server = Fixture::start(
        ServerConfig {
            // A stop must shorten these: the notification below would otherwise come after `PATIENCE`.
            linger_limit: Duration::from_secs(600),
            stop_linger_limit: Duration::from_millis(200),
            ..config()
        },
        handler,
    );
    let mut busy = server.connect();
    busy.send(b"GET /busy HTTP/1.1\r\nHost: t\r\n\r\n");
    let (_, reply) = parked.recv_timeout(PATIENCE).unwrap();
    server.begin_stop();
    reply.send(Response::with_status(200));
    assert_eq!(busy.read_response(false).status, 200);
    // The peer keeps its end open and says nothing.
    server
        .stopped
        .recv_timeout(PATIENCE)
        .expect("the server did not report that it stopped");
    assert_eq!(server.handle.counters().open, 0);
    busy.expect_closed();
    server.sink.complete_gracefully();
    server.lattice.close();
}

#[test]
fn a_request_dripped_in_slowly_passes_the_read_limit_as_a_whole() {
    let limit = Duration::from_millis(300);
    let server = Fixture::start(
        ServerConfig {
            read_limit: Some(limit),
            ..config()
        },
        echo_target,
    );
    let mut peer = server.connect();
    peer.stream
        .set_read_timeout(Some(Duration::from_millis(10)))
        .unwrap();
    let started = Instant::now();
    peer.send(b"GET / HTTP/1.1\r\nHost: t\r\nX-Slow: ");
    let mut seen = Vec::new();
    let mut chunk = [0u8; 256];
    while !seen.ends_with(b"\r\n\r\n") && !seen.windows(4).any(|w| w == b"\r\n\r\n") {
        assert!(started.elapsed() < PATIENCE, "no answer");
        // A byte now and then: every read of the server's gets data in time, the request as a whole is late.
        let _ = peer.stream.write_all(b"a");
        match peer.stream.read(&mut chunk) {
            Ok(0) => break,
            Ok(n) => seen.extend_from_slice(&chunk[..n]),
            Err(_) => {}
        }
    }
    assert!(seen.starts_with(b"HTTP/1.1 408"), "the answer is a 408");
    assert!(started.elapsed() >= limit, "the limit did not pass");
    drop(peer);
    let counters = server.finish();
    assert_eq!((counters.limits_passed, counters.requests), (1, 0));
}

#[test]
fn a_handler_that_panics_answers_500_once_and_the_server_goes_on() {
    let (handler, _parked) = parking();
    let server = Fixture::start(config(), handler);
    let mut peer = server.connect();
    assert_eq!(peer.get("/boom").status, 500);
    assert_eq!(peer.get("/ok").status, 200);
    drop(peer);
    let counters = server.finish();
    assert_eq!((counters.replies_5xx, counters.replies_2xx), (1, 1));
}

#[cfg(unix)]
#[test]
fn a_failing_accept_is_counted_and_retried_after_the_back_off_and_stop_still_completes() {
    use std::net::UdpSocket;
    use std::os::fd::OwnedFd;
    // A datagram socket is not listening: every accept on it fails.
    let socket = UdpSocket::bind("127.0.0.1:0").unwrap();
    let listener = TcpListener::from(OwnedFd::from(socket));
    let backoff = Duration::from_millis(60);
    let started = Instant::now();
    let server = Fixture::start_on(
        listener,
        ServerConfig {
            accept_backoff: backoff,
            ..config()
        },
        echo_target,
    );
    until("three failed accepts", || {
        server.handle.counters().accept_errors >= 3
    });
    // Three failures with a back-off before each retry: at least two back-offs have passed.
    assert!(
        started.elapsed() >= 2 * backoff,
        "no back-off between the accepts"
    );
    let counters = server.finish();
    assert!(counters.accept_errors >= 3);
    assert_eq!(counters.accepted, 0);
}

// ---- streamed replies ---------------------------------------------------------------------------------------

type Started = Result<ReplyStream, StreamError>;
type Sent = Result<PieceSent, StreamError>;

/// A handler that answers `/ok` at once and answers everything else with a streamed reply whose head `head` makes
/// from the request: the test thread gets what `then` was given through the channel.
fn streaming(
    head: impl Fn(&Request) -> Response + Send + Sync + 'static,
) -> (impl Handler, Receiver<Started>) {
    let (tx, rx) = channel::<Started>();
    let handler = move |request: Request, reply: Responder| -> Box<dyn Frame> {
        if request.target == "/ok" {
            reply.send(Response::with_status(200).with_body("ok"));
            return no_work();
        }
        let response = head(&request);
        let tx = tx.clone();
        reply.stream(
            response,
            Box::new(move |started| {
                let _ = tx.send(started);
                no_work()
            }),
        );
        no_work()
    };
    (handler, rx)
}

fn ok_head(_request: &Request) -> Response {
    Response::with_status(200).header("X-Kind", "s")
}

fn started(rx: &Receiver<Started>) -> ReplyStream {
    rx.recv_timeout(PATIENCE)
        .expect("the head was not reported")
        .expect("the stream did not start")
}

/// Sends a piece and returns what the continuation was given.
fn try_piece(stream: ReplyStream, bytes: &[u8]) -> Sent {
    let (tx, rx) = channel::<Sent>();
    stream.send(
        bytes.to_vec(),
        Box::new(move |sent| {
            let _ = tx.send(sent);
            no_work()
        }),
    );
    rx.recv_timeout(PATIENCE)
        .expect("the piece was not reported")
}

/// Sends a piece that must be written and returns the stream.
fn piece(stream: ReplyStream, bytes: &[u8]) -> ReplyStream {
    let sent = try_piece(stream, bytes).expect("the piece was not written");
    assert!(sent.piece.is_empty(), "the buffer came back with bytes");
    assert!(
        sent.piece.capacity() >= bytes.len(),
        "the buffer's capacity was lost"
    );
    sent.stream
}

const STREAM_GET: &[u8] = b"GET /s HTTP/1.1\r\nHost: t\r\n\r\n";
const STREAM_HEAD: &[u8] = b"HTTP/1.1 200 OK\r\nX-Kind: s\r\nTransfer-Encoding: chunked\r\n\r\n";

#[test]
fn a_streamed_reply_goes_out_piece_by_piece_and_the_connection_is_kept() {
    let (handler, rx) = streaming(ok_head);
    let server = Fixture::start(config(), handler);
    let mut peer = server.connect();
    peer.send(STREAM_GET);
    let stream = started(&rx);
    // The head is out before any piece was given.
    peer.expect_wire(STREAM_HEAD);
    let stream = piece(stream, b"hello");
    peer.expect_wire(b"5\r\nhello\r\n");
    let stream = piece(stream, &[b'a'; 255]);
    let mut expected = b"ff\r\n".to_vec();
    expected.extend_from_slice(&[b'a'; 255]);
    expected.extend_from_slice(b"\r\n");
    peer.expect_wire(&expected);
    stream.end();
    peer.expect_wire(b"0\r\n\r\n");
    // The connection serves its next request.
    assert_eq!(text(&peer.get("/ok")), "ok");
    drop(peer);
    let counters = server.finish();
    assert_eq!(
        (
            counters.requests,
            counters.streams,
            counters.streams_cut,
            counters.replies_2xx
        ),
        (2, 1, 0, 2)
    );
}

#[test]
fn an_empty_piece_writes_nothing_and_does_not_end_the_stream() {
    let (handler, rx) = streaming(ok_head);
    let server = Fixture::start(config(), handler);
    let mut peer = server.connect();
    peer.send(STREAM_GET);
    let stream = started(&rx);
    peer.expect_wire(STREAM_HEAD);
    // The empty piece is reported as written, and the next bytes on the wire are those of the piece after it.
    let stream = piece(stream, b"");
    let stream = piece(stream, b"a");
    peer.expect_wire(b"1\r\na\r\n");
    stream.end();
    peer.expect_wire(b"0\r\n\r\n");
    assert_eq!(text(&peer.get("/ok")), "ok");
    drop(peer);
    server.finish();
}

#[test]
fn a_head_with_a_body_sends_it_as_the_first_piece() {
    let (handler, rx) = streaming(|_| Response::with_status(200).with_body("first"));
    let server = Fixture::start(config(), handler);
    let mut peer = server.connect();
    peer.send(STREAM_GET);
    let stream = started(&rx);
    peer.expect_wire(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nfirst\r\n");
    let stream = piece(stream, b"second");
    peer.expect_wire(b"6\r\nsecond\r\n");
    stream.end();
    peer.expect_wire(b"0\r\n\r\n");
    drop(peer);
    server.finish();
}

#[test]
fn a_dropped_stream_closes_the_connection_without_the_last_chunk() {
    let (handler, rx) = streaming(ok_head);
    let server = Fixture::start(config(), handler);
    let mut peer = server.connect();
    peer.send(STREAM_GET);
    let stream = started(&rx);
    peer.expect_wire(STREAM_HEAD);
    let stream = piece(stream, b"x");
    peer.expect_wire(b"1\r\nx\r\n");
    drop(stream);
    // The body ends here with no last chunk: what follows is the end of the connection.
    peer.expect_closed();
    drop(peer);
    let counters = server.finish();
    assert_eq!((counters.streams, counters.streams_cut), (1, 1));
}

#[test]
fn a_stream_to_http_1_0_is_ended_by_the_close() {
    let (handler, rx) = streaming(ok_head);
    let server = Fixture::start(config(), handler);
    let mut peer = server.connect();
    peer.send(b"GET /s HTTP/1.0\r\nConnection: keep-alive\r\n\r\n");
    let stream = started(&rx);
    // No chunked coding and no length: the close ends the body, so `keep-alive` is not honoured.
    peer.expect_wire(b"HTTP/1.1 200 OK\r\nX-Kind: s\r\nConnection: close\r\n\r\n");
    let stream = piece(stream, b"ab");
    peer.expect_wire(b"ab");
    stream.end();
    assert_eq!(peer.read_to_end(), b"");
    drop(peer);
    let counters = server.finish();
    assert_eq!((counters.streams, counters.streams_cut), (1, 0));
}

#[test]
fn a_stream_to_head_sends_the_head_only_and_keeps_the_connection() {
    let (handler, rx) = streaming(|_| {
        Response::with_status(200)
            .header("X-Kind", "s")
            .with_body("dropped")
    });
    let server = Fixture::start(config(), handler);
    let mut peer = server.connect();
    peer.send(b"HEAD /s HTTP/1.1\r\nHost: t\r\n\r\n");
    let stream = started(&rx);
    peer.expect_wire(b"HTTP/1.1 200 OK\r\nX-Kind: s\r\n\r\n");
    // The pieces are reported written and nothing goes out.
    let stream = piece(stream, b"abc");
    stream.end();
    // The next thing on the wire is the answer to the next request.
    assert_eq!(text(&peer.get("/ok")), "ok");
    drop(peer);
    let counters = server.finish();
    assert_eq!((counters.streams, counters.streams_cut), (1, 0));
}

#[test]
fn a_connection_close_request_gets_its_stream_and_then_closes() {
    let (handler, rx) = streaming(ok_head);
    let server = Fixture::start(config(), handler);
    let mut peer = server.connect();
    peer.send(b"GET /s HTTP/1.1\r\nHost: t\r\nConnection: close\r\n\r\n");
    let stream = started(&rx);
    peer.expect_wire(
        b"HTTP/1.1 200 OK\r\nX-Kind: s\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n",
    );
    let stream = piece(stream, b"abc");
    peer.expect_wire(b"3\r\nabc\r\n");
    stream.end();
    peer.expect_wire(b"0\r\n\r\n");
    peer.expect_closed();
    drop(peer);
    server.finish();
}

#[test]
fn a_stream_head_that_cannot_be_written_answers_500_and_reports_it() {
    let (handler, rx) = streaming(|request| match request.target.as_str() {
        "/204" => Response::with_status(204),
        "/line" => Response::with_status(200).header("X", "a\r\nSet-Cookie: b"),
        _ => Response::with_status(200).header("Content-Length", "3"),
    });
    let server = Fixture::start(config(), handler);
    let mut peer = server.connect();
    for (target, expected) in [
        ("/204", "BodyNotAllowed"),
        ("/line", "BadHeaderValue"),
        ("/length", "ReservedHeader"),
    ] {
        let mut request = Vec::new();
        request.extend_from_slice(format!("GET {target} HTTP/1.1\r\nHost: t\r\n\r\n").as_bytes());
        peer.send(&request);
        match rx
            .recv_timeout(PATIENCE)
            .expect("the head was not reported")
        {
            Err(StreamError::Head(error)) => assert_eq!(format!("{error:?}"), expected),
            other => panic!("{target}: {other:?}"),
        }
        assert_eq!(peer.read_response(false).status, 500, "{target}");
    }
    // The connection is still open.
    assert_eq!(text(&peer.get("/ok")), "ok");
    drop(peer);
    let counters = server.finish();
    assert_eq!(
        (counters.streams, counters.replies_5xx, counters.replies_2xx),
        (0, 3, 1)
    );
}

#[test]
fn a_peer_that_stops_reading_passes_the_piece_limit_and_the_piece_reports_it() {
    let limit = Duration::from_millis(150);
    let size = 64 * 1024 * 1024;
    let (handler, rx) = streaming(ok_head);
    let server = Fixture::start(
        ServerConfig {
            piece_limit: Some(limit),
            ..config()
        },
        handler,
    );
    let mut peer = server.connect();
    peer.send(STREAM_GET);
    let stream = started(&rx);
    // The peer reads nothing of the piece, which is larger than the connection can hold.
    let began = Instant::now();
    let result = try_piece(stream, &vec![b'x'; size]);
    assert!(began.elapsed() >= limit, "the limit did not pass");
    assert!(
        matches!(result, Err(StreamError::TimedOut)),
        "the piece ended with {result:?}"
    );
    let got = peer.read_to_end().len();
    assert!(got < size, "the whole piece went out");
    drop(peer);
    let counters = server.finish();
    assert_eq!(
        (counters.limits_passed, counters.streams_cut),
        (1, 1),
        "the cut is counted once"
    );
}

#[test]
fn a_peer_that_leaves_mid_stream_runs_the_cancel_frame_and_the_next_piece_reports_closed() {
    let (handler, rx) = streaming(ok_head);
    let server = Fixture::start(config(), handler);
    let mut peer = server.connect();
    peer.send(STREAM_GET);
    let stream = started(&rx);
    let (woken, cancelled) = signal();
    stream.on_cancel(woken);
    // The head is left unread: closing a socket that holds unread data resets the connection.
    let mut probe = [0u8; 1];
    peek_on(&mut peer.stream, &mut probe);
    drop(peer);
    cancelled
        .recv_timeout(PATIENCE)
        .expect("the frame for the cancel did not run");
    assert!(stream.cancelled().load(Ordering::Acquire));
    let result = try_piece(stream, b"late");
    assert!(
        matches!(result, Err(StreamError::Closed)),
        "the piece ended with {result:?}"
    );
    let counters = server.finish();
    assert_eq!(
        (counters.cancelled, counters.streams_cut, counters.discarded),
        (1, 1, 0)
    );
}

#[test]
fn a_half_closed_peer_gets_the_whole_stream_and_then_the_connection_closes() {
    let (handler, rx) = streaming(ok_head);
    let server = Fixture::start(config(), handler);
    let mut peer = server.connect();
    peer.send(STREAM_GET);
    let stream = started(&rx);
    let (woken, cancelled) = signal();
    stream.on_cancel(woken);
    peer.stream.shutdown(Shutdown::Write).unwrap();
    // The flag is a hint that the peer's end of the stream was seen; the stream goes on.
    cancelled
        .recv_timeout(PATIENCE)
        .expect("the frame for the cancel did not run");
    assert!(stream.cancelled().load(Ordering::Acquire));
    peer.expect_wire(STREAM_HEAD);
    let stream = piece(stream, b"abc");
    peer.expect_wire(b"3\r\nabc\r\n");
    stream.end();
    peer.expect_wire(b"0\r\n\r\n");
    peer.expect_closed();
    drop(peer);
    let counters = server.finish();
    assert_eq!(
        (counters.cancelled, counters.streams, counters.streams_cut),
        (1, 1, 0)
    );
}

#[test]
fn stop_lets_a_stream_end_and_is_not_reported_while_the_stream_is_held() {
    let (handler, rx) = streaming(ok_head);
    let server = Fixture::start(config(), handler);
    let mut peer = server.connect();
    peer.send(STREAM_GET);
    let stream = started(&rx);
    peer.expect_wire(STREAM_HEAD);
    server.begin_stop();
    // A piece goes out and comes back after the stop was recorded: the stream is still ours, and the server has not
    // reported. (It is the end of the stream, not a wait, that lets the server finish: the next lines end it.)
    let stream = piece(stream, b"abc");
    peer.expect_wire(b"3\r\nabc\r\n");
    assert!(server.stopped.try_recv().is_err());
    assert_eq!(server.handle.counters().open, 1);
    stream.end();
    peer.expect_wire(b"0\r\n\r\n");
    // The stop closes the connection after the stream, which the peer sees.
    peer.expect_closed();
    drop(peer);
    server
        .stopped
        .recv_timeout(PATIENCE)
        .expect("the server did not report that it stopped");
    assert_eq!(server.handle.counters().streams_cut, 0);
    server.finish();
}

#[test]
fn a_pipelined_request_behind_a_stream_is_answered_after_its_end() {
    let (handler, rx) = streaming(ok_head);
    let server = Fixture::start(config(), handler);
    let mut peer = server.connect();
    peer.send(b"GET /s HTTP/1.1\r\nHost: t\r\n\r\nGET /ok HTTP/1.1\r\nHost: t\r\n\r\n");
    let stream = started(&rx);
    peer.expect_wire(STREAM_HEAD);
    let stream = piece(stream, b"abc");
    peer.expect_wire(b"3\r\nabc\r\n");
    // The second request is held back while the stream goes on.
    assert_eq!(server.handle.counters().requests, 1);
    stream.end();
    peer.expect_wire(b"0\r\n\r\n");
    let response = peer.read_response(false);
    assert_eq!((response.status, text(&response)), (200, "ok"));
    drop(peer);
    let counters = server.finish();
    assert_eq!((counters.requests, counters.streams_cut), (2, 0));
}

#[test]
fn a_stream_head_given_after_the_peer_left_reports_closed_and_is_discarded() {
    let (handler, parked) = parking();
    let server = Fixture::start(config(), handler);
    let mut peer = server.connect();
    assert_eq!(text(&peer.get("/ok")), "ok");
    peer.send(b"GET /ok HTTP/1.1\r\nHost: t\r\n\r\nGET /slow HTTP/1.1\r\nHost: t\r\n\r\n");
    let (_, reply) = parked.recv_timeout(PATIENCE).unwrap();
    let (woken, cancelled) = signal();
    reply.on_cancel(woken);
    let mut probe = [0u8; 1];
    peek_on(&mut peer.stream, &mut probe);
    drop(peer);
    cancelled
        .recv_timeout(PATIENCE)
        .expect("the frame for the cancel did not run");
    let (tx, rx) = channel::<Started>();
    reply.stream(
        Response::with_status(200),
        Box::new(move |started| {
            let _ = tx.send(started);
            no_work()
        }),
    );
    let result = rx
        .recv_timeout(PATIENCE)
        .expect("the head was not reported");
    assert!(
        matches!(result, Err(StreamError::Closed)),
        "the head ended with {result:?}"
    );
    let counters = server.finish();
    assert_eq!((counters.discarded, counters.streams), (1, 0));
}

#[test]
fn a_continuation_that_panics_drops_the_stream_and_closes_the_connection() {
    let handler = |request: Request, reply: Responder| -> Box<dyn Frame> {
        if request.target == "/ok" {
            reply.send(Response::with_status(200).with_body("ok"));
        } else {
            reply.stream(
                Response::with_status(200).header("X-Kind", "s"),
                Box::new(|started| {
                    let _stream = started;
                    panic!("the continuation failed");
                }),
            );
        }
        no_work()
    };
    let server = Fixture::start(config(), handler);
    let mut peer = server.connect();
    peer.send(STREAM_GET);
    // The head is out, the stream was dropped with the continuation that held it, and the connection closed with
    // no last chunk. The server goes on.
    peer.expect_wire(STREAM_HEAD);
    peer.expect_closed();
    let mut other = server.connect();
    assert_eq!(text(&other.get("/ok")), "ok");
    drop((peer, other));
    let counters = server.finish();
    assert_eq!((counters.streams, counters.streams_cut), (1, 1));
}

// ---- streamed replies: the ends ------------------------------------------------------------------------------

/// A body larger than the connection can hold when the peer reads nothing, so a write of it stays in flight.
const BIG: usize = 64 * 1024 * 1024;

/// Reads and drops `count` bytes of the wire.
fn discard(peer: &mut Peer, count: usize) {
    let mut left = count;
    let held = peer.buf.len().min(left);
    peer.buf.drain(..held);
    left -= held;
    let mut chunk = vec![0u8; 256 * 1024];
    while left > 0 {
        let want = left.min(chunk.len());
        let n = read_on(&mut peer.stream, &mut chunk[..want]).expect("the server's bytes");
        assert!(
            n > 0,
            "the server closed the connection with {left} bytes to go"
        );
        left -= n;
    }
}

fn big_head(_request: &Request) -> Response {
    Response::with_status(200)
        .header("X-Kind", "s")
        .with_body(vec![b'x'; BIG])
}

#[test]
fn a_peer_that_leaves_while_the_head_is_being_written_ends_the_head_with_closed_once() {
    let (handler, rx) = streaming(big_head);
    let server = Fixture::start(config(), handler);
    let mut peer = server.connect();
    peer.send(STREAM_GET);
    // The peer reads nothing, so the head and its body stay in flight; it leaves with the bytes unread, which
    // resets the connection.
    let mut probe = [0u8; 1];
    peek_on(&mut peer.stream, &mut probe);
    drop(peer);
    let result = rx
        .recv_timeout(PATIENCE)
        .expect("the head was not reported");
    assert!(
        matches!(result, Err(StreamError::Closed)),
        "the head ended with {result:?}"
    );
    // The stop is not held by the stream that never came.
    let counters = server.finish();
    assert_eq!((counters.streams, counters.streams_cut), (1, 1));
}

#[test]
fn a_peer_that_leaves_while_a_piece_is_being_written_ends_the_piece_with_closed_once() {
    let (handler, rx) = streaming(ok_head);
    let server = Fixture::start(config(), handler);
    let mut peer = server.connect();
    peer.send(STREAM_GET);
    let stream = started(&rx);
    let (tx, done) = channel::<Sent>();
    stream.send(
        vec![b'x'; BIG],
        Box::new(move |sent| {
            let _ = tx.send(sent);
            no_work()
        }),
    );
    let mut probe = [0u8; 1];
    peek_on(&mut peer.stream, &mut probe);
    drop(peer);
    let result = done
        .recv_timeout(PATIENCE)
        .expect("the piece was not reported");
    assert!(
        matches!(result, Err(StreamError::Closed)),
        "the piece ended with {result:?}"
    );
    let counters = server.finish();
    assert_eq!((counters.streams, counters.streams_cut), (1, 1));
    assert!(done.try_recv().is_err(), "the piece was reported twice");
}

#[test]
fn a_stop_that_comes_while_the_head_is_being_written_waits_for_the_stream_and_its_end() {
    let (handler, rx) = streaming(big_head);
    let server = Fixture::start(config(), handler);
    let mut peer = server.connect();
    peer.send(STREAM_GET);
    // The head is counted when its write starts: a stop asked before would find the connection waiting for a
    // request and close it.
    until("the head was taken", || {
        server.handle.counters().streams == 1
    });
    server.begin_stop();
    // The peer has read nothing, so the head cannot be out, and the server cannot have reported.
    assert!(rx.try_recv().is_err());
    assert!(server.stopped.try_recv().is_err());
    // The head, then the body as one chunk: `4000000` and its CRLF, the bytes and a CRLF.
    discard(&mut peer, STREAM_HEAD.len() + 9 + BIG + 2);
    let stream = started(&rx);
    assert!(server.stopped.try_recv().is_err());
    stream.end();
    assert_eq!(peer.read_to_end(), b"0\r\n\r\n");
    drop(peer);
    server
        .stopped
        .recv_timeout(PATIENCE)
        .expect("the server did not report that it stopped");
    assert_eq!(server.handle.counters().streams_cut, 0);
    server.finish();
}

#[test]
fn a_stream_to_http_1_0_that_is_dropped_is_cut_and_counted() {
    let (handler, rx) = streaming(ok_head);
    let server = Fixture::start(config(), handler);
    let mut peer = server.connect();
    peer.send(b"GET /s HTTP/1.0\r\n\r\n");
    let stream = started(&rx);
    peer.expect_wire(b"HTTP/1.1 200 OK\r\nX-Kind: s\r\nConnection: close\r\n\r\n");
    let stream = piece(stream, b"ab");
    peer.expect_wire(b"ab");
    drop(stream);
    peer.expect_closed();
    drop(peer);
    let counters = server.finish();
    assert_eq!((counters.streams, counters.streams_cut), (1, 1));
}

#[test]
fn a_stream_to_head_that_is_dropped_is_cut_and_counted() {
    let (handler, rx) = streaming(ok_head);
    let server = Fixture::start(config(), handler);
    let mut peer = server.connect();
    peer.send(b"HEAD /s HTTP/1.1\r\nHost: t\r\n\r\n");
    let stream = started(&rx);
    peer.expect_wire(b"HTTP/1.1 200 OK\r\nX-Kind: s\r\n\r\n");
    drop(stream);
    peer.expect_closed();
    drop(peer);
    let counters = server.finish();
    assert_eq!((counters.streams, counters.streams_cut), (1, 1));
}

#[test]
fn a_cancel_frame_asked_for_on_a_stream_after_the_peer_left_still_runs_once() {
    let (handler, rx) = streaming(ok_head);
    let server = Fixture::start(config(), handler);
    let mut peer = server.connect();
    peer.send(STREAM_GET);
    let stream = started(&rx);
    // The first frame is the gate: when it ran, the server has seen the peer go and closed the connection.
    let (seen, gone) = signal();
    stream.on_cancel(seen);
    let mut probe = [0u8; 1];
    peek_on(&mut peer.stream, &mut probe);
    drop(peer);
    gone.recv_timeout(PATIENCE)
        .expect("the cancel was not seen");
    let (spy, report) = Spy::pair();
    stream.on_cancel(spy);
    assert_eq!(report.recv_timeout(PATIENCE), Ok("ran"));
    drop(stream);
    server.finish();
    assert!(
        report.try_recv().is_err(),
        "the frame was dropped or ran twice"
    );
}

#[test]
fn a_stream_head_with_a_status_that_has_no_body_answers_500_and_reports_it() {
    let (handler, rx) = streaming(|request| match request.target.as_str() {
        "/304" => Response::with_status(304),
        _ => Response::with_status(100),
    });
    let server = Fixture::start(config(), handler);
    let mut peer = server.connect();
    for target in ["/304", "/100"] {
        peer.send(format!("GET {target} HTTP/1.1\r\nHost: t\r\n\r\n").as_bytes());
        match rx
            .recv_timeout(PATIENCE)
            .expect("the head was not reported")
        {
            Err(StreamError::Head(error)) => assert_eq!(format!("{error:?}"), "BodyNotAllowed"),
            other => panic!("{target}: {other:?}"),
        }
        assert_eq!(peer.read_response(false).status, 500, "{target}");
    }
    drop(peer);
    let counters = server.finish();
    assert_eq!((counters.streams, counters.replies_5xx), (0, 2));
}
