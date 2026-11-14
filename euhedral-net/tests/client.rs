//! The client over loopback, on a real lattice and a real `AsyncIo`.
//!
//! A well-behaved peer is the crate's own server; a misbehaving one is a plain blocking socket on the test thread
//! (the test thread is not a lattice worker). Order is driven by what the peers do and by channels that `then`
//! frames and handlers send on; nothing sleeps to synchronise. A time-limit test asserts a lower bound (the call
//! did not end before its limit) and the error, never an upper bound; the receive timeouts are a failsafe against
//! a hang. Every test ends with `Fixture::finish`: no operation is pending, and `then` ran exactly once per call.
#![cfg(any(target_os = "linux", target_os = "android", windows))]
#![allow(
    clippy::disallowed_types,
    clippy::disallowed_methods,
    reason = "test fixtures: the test thread is not a lattice worker; it accepts, reads and waits on channels"
)]

use std::io::{ErrorKind, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, channel};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use euhedral_core::config::{FragmentConfig, IdlePolicy, LatticeConfig};
use euhedral_core::control_plane::ControlPlaneLattice;
use euhedral_core::frames::{Frame, FrameError, FrameHeader};
use euhedral_core::generics::LatticeSource;
use euhedral_core::impls::{BaseCloneableObject, DefaultExecutor};
use euhedral_core::ingest::{IngestSink, QueueIngestSink};
use euhedral_graph::Completions;
use euhedral_hardware_utils::affinity;
use euhedral_io::AsyncIo;
use euhedral_net::{
    CallStage, ClientConfig, Limits, Method, NetError, Request, Responder, Response, ServerConfig,
    ServerHandle, WriteError, no_work, request, serve,
};

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
    config.name = "net-client-test".into();
    config.shard_name = "net-client-test".into();
    config.allowed_cpus = cpus.into_iter().collect();
    config.shutdown_timeout = Duration::from_secs(5);
    config.startup_timeout = Duration::from_secs(10);
    let lattice = ControlPlaneLattice::new(config).unwrap();
    lattice.start().unwrap();
    lattice
}

/// A frame that runs a closure once: what a test offers to do something on a worker, and what `then` returns.
struct Run {
    header: FrameHeader,
    work: Option<Box<dyn FnOnce() + Send>>,
}

impl Run {
    fn frame(work: impl FnOnce() + Send + 'static) -> Box<dyn Frame> {
        Box::new(Run {
            header: FrameHeader::new(9),
            work: Some(Box::new(work)),
        })
    }
}

impl Frame for Run {
    fn header(&self) -> &FrameHeader {
        &self.header
    }

    fn header_mut(&mut self) -> &mut FrameHeader {
        &mut self.header
    }

    fn execute(&mut self) -> Result<(), FrameError> {
        if let Some(work) = self.work.take() {
            work();
        }
        Ok(())
    }
}

/// What a case expects of its error.
type Expect = fn(&NetError) -> bool;

type Outcome = Result<Response, NetError>;

struct Fixture {
    lattice: Arc<ControlPlaneLattice>,
    io: Arc<AsyncIo>,
    sink: Arc<QueueIngestSink>,
    /// How many times a `then` ran.
    thens: Arc<AtomicUsize>,
    /// How many calls were made.
    calls: usize,
}

impl Fixture {
    fn new() -> Fixture {
        let lattice = lattice();
        let io = AsyncIo::open().unwrap();
        let sink = Arc::new(QueueIngestSink::new());
        lattice.add_upstream(io.clone()).unwrap();
        lattice.add_upstream(sink.delegate()).unwrap();
        Fixture {
            lattice,
            io,
            sink,
            thens: Arc::new(AtomicUsize::new(0)),
            calls: 0,
        }
    }

    /// Makes a call; its outcome arrives on the channel.
    fn call(&mut self, to: SocketAddr, req: &Request, config: ClientConfig) -> Receiver<Outcome> {
        self.calls += 1;
        let (tx, rx) = channel();
        let thens = self.thens.clone();
        request(
            &self.io,
            &self.sink,
            to,
            req,
            config,
            Box::new(move |outcome| {
                thens.fetch_add(1, Ordering::SeqCst);
                Run::frame(move || {
                    let _ = tx.send(outcome);
                })
            }),
        );
        rx
    }

    fn get(&mut self, to: SocketAddr, config: ClientConfig) -> Receiver<Outcome> {
        self.call(to, &get("/x"), config)
    }

    /// Nothing is left: the outcome of every call came, once, and no operation is pending.
    fn finish(self) {
        assert_eq!(
            self.thens.load(Ordering::SeqCst),
            self.calls,
            "`then` runs once per call"
        );
        assert_eq!(self.io.pending(), 0, "an operation is still pending");
        self.sink.complete_gracefully();
        self.lattice.close();
    }
}

fn get(target: &str) -> Request {
    Request::new(Method::Get, target).header("Host", "t")
}

fn config() -> ClientConfig {
    ClientConfig {
        time_limit: Some(PATIENCE),
        ..ClientConfig::default()
    }
}

fn outcome(rx: &Receiver<Outcome>) -> Outcome {
    rx.recv_timeout(PATIENCE).expect("the call did not end")
}

/// A raw peer: a listener whose accepted stream the test thread drives.
struct Peer {
    listener: TcpListener,
}

impl Peer {
    fn new() -> Peer {
        Peer {
            listener: TcpListener::bind("127.0.0.1:0").unwrap(),
        }
    }

    fn addr(&self) -> SocketAddr {
        self.listener.local_addr().unwrap()
    }

    /// Accepts the call's connection and reads its request head.
    fn accept(&self) -> (TcpStream, String) {
        let (mut stream, _) = self.listener.accept().unwrap();
        stream.set_read_timeout(Some(PATIENCE)).unwrap();
        let mut head = Vec::new();
        let mut byte = [0u8; 1];
        while !head.ends_with(b"\r\n\r\n") {
            // A read that goes on after it was interrupted (see the server tests).
            let n = match stream.read(&mut byte) {
                Err(e) if e.kind() == ErrorKind::Interrupted => continue,
                other => other.unwrap(),
            };
            assert_eq!(n, 1, "the client closed before its request was whole");
            head.push(byte[0]);
        }
        (stream, String::from_utf8(head).unwrap())
    }
}

/// A server that answers every request at once: the target as the body; HEAD gets its head.
fn echo(request: Request, reply: Responder) -> Box<dyn Frame> {
    let body = request.target.clone().into_bytes();
    reply.send(Response::with_status(200).with_body(body));
    no_work()
}

struct Served {
    lattice: Arc<ControlPlaneLattice>,
    io: Arc<AsyncIo>,
    handle: ServerHandle,
    addr: SocketAddr,
}

fn serve_on(
    fixture: &Fixture,
    handler: impl Fn(Request, Responder) -> Box<dyn Frame> + Send + Sync + 'static,
) -> Served {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let timers = Completions::without_lanes();
    fixture.lattice.add_upstream(timers.clone()).unwrap();
    let handle = serve(
        fixture.lattice.clone(),
        fixture.io.clone(),
        fixture.sink.clone(),
        listener,
        ServerConfig {
            read_limit: None,
            idle_limit: None,
            write_limit: None,
            ..ServerConfig::default()
        },
        Arc::new(handler),
        timers,
    )
    .unwrap();
    Served {
        lattice: fixture.lattice.clone(),
        io: fixture.io.clone(),
        handle,
        addr,
    }
}

impl Served {
    fn stop(self) {
        let (tx, rx) = channel();
        self.handle.stop(Run::frame(move || {
            let _ = tx.send(());
        }));
        rx.recv_timeout(PATIENCE).expect("the server did not stop");
        let _ = (&self.lattice, &self.io);
    }
}

#[test]
fn a_response_with_a_body_arrives() {
    let mut f = Fixture::new();
    let server = serve_on(&f, echo);
    let rx = f.call(server.addr, &get("/hello"), config());
    let response = outcome(&rx).unwrap();
    assert_eq!(response.status, 200);
    assert_eq!(response.body, b"/hello");
    assert_eq!(response.headers.get("content-length"), Some("6"));
    server.stop();
    f.finish();
}

#[test]
fn an_empty_body_arrives_as_an_empty_body() {
    let mut f = Fixture::new();
    let server = serve_on(&f, |_: Request, reply: Responder| {
        reply.send(Response::with_status(200));
        no_work()
    });
    let response = outcome(&f.get(server.addr, config())).unwrap();
    assert_eq!((response.status, response.body.len()), (200, 0));
    server.stop();
    f.finish();
}

#[test]
fn a_head_request_reads_no_body() {
    let mut f = Fixture::new();
    let server = serve_on(&f, echo);
    let head = Request::new(Method::Head, "/hello").header("Host", "t");
    let response = outcome(&f.call(server.addr, &head, config())).unwrap();
    assert_eq!(response.status, 200);
    // The length the body would have, and no body.
    assert_eq!(response.headers.get("content-length"), Some("6"));
    assert!(response.body.is_empty());
    server.stop();
    f.finish();
}

#[test]
fn the_request_is_sent_whole_with_connection_close() {
    let mut f = Fixture::new();
    let peer = Peer::new();
    let post = Request::new(Method::Post, "/p")
        .header("Host", "t")
        .with_body("abc");
    let rx = f.call(peer.addr(), &post, config());
    let (mut stream, head) = peer.accept();
    assert!(head.starts_with("POST /p HTTP/1.1\r\n"));
    assert!(head.contains("Connection: close\r\n") && head.contains("Content-Length: 3\r\n"));
    let mut body = [0u8; 3];
    stream.read_exact(&mut body).unwrap();
    assert_eq!(&body, b"abc");
    stream
        .write_all(b"HTTP/1.1 201 Created\r\nContent-Length: 2\r\n\r\nok")
        .unwrap();
    let response = outcome(&rx).unwrap();
    assert_eq!(
        (response.status, response.body.as_slice()),
        (201, &b"ok"[..])
    );
    // The client closed its end.
    let mut rest = [0u8; 8];
    assert_eq!(stream.read(&mut rest).unwrap_or(0), 0);
    f.finish();
}

#[test]
fn a_refused_connection_keeps_the_io_error() {
    let mut f = Fixture::new();
    let addr = Peer::new().addr();
    // The listener is gone: nothing listens on the port.
    let error = outcome(&f.get(addr, config())).unwrap_err();
    match error {
        NetError::Connect(e) => assert_eq!(e.kind(), ErrorKind::ConnectionRefused),
        other => panic!("{other}"),
    }
    f.finish();
}

#[test]
fn a_request_the_writer_refuses_is_reported_and_nothing_is_sent() {
    let mut f = Fixture::new();
    let peer = Peer::new();
    peer.listener.set_nonblocking(true).unwrap();
    let no_host = Request::new(Method::Get, "/");
    let error = outcome(&f.call(peer.addr(), &no_host, config())).unwrap_err();
    assert!(
        matches!(error, NetError::Write(WriteError::BadHost)),
        "{error}"
    );
    // No connection was made.
    assert_eq!(
        peer.listener.accept().unwrap_err().kind(),
        ErrorKind::WouldBlock
    );
    f.finish();
}

#[test]
fn a_server_that_never_answers_ends_the_call_at_its_limit_in_the_head() {
    let mut f = Fixture::new();
    let peer = Peer::new();
    let limit = Duration::from_millis(200);
    let started = Instant::now();
    let rx = f.get(
        peer.addr(),
        ClientConfig {
            time_limit: Some(limit),
            ..config()
        },
    );
    let (_stream, _head) = peer.accept();
    let error = outcome(&rx).unwrap_err();
    assert!(started.elapsed() >= limit);
    assert!(
        matches!(error, NetError::TimedOut(CallStage::ReceiveHead)),
        "{error}"
    );
    f.finish();
}

#[test]
fn a_server_that_stalls_in_the_body_ends_the_call_at_its_limit_in_the_body() {
    let mut f = Fixture::new();
    let peer = Peer::new();
    let limit = Duration::from_millis(200);
    let started = Instant::now();
    let rx = f.get(
        peer.addr(),
        ClientConfig {
            time_limit: Some(limit),
            ..config()
        },
    );
    let (mut stream, _head) = peer.accept();
    stream
        .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 10\r\n\r\nabc")
        .unwrap();
    let error = outcome(&rx).unwrap_err();
    assert!(started.elapsed() >= limit);
    assert!(
        matches!(error, NetError::TimedOut(CallStage::ReceiveBody)),
        "{error}"
    );
    f.finish();
}

#[test]
fn a_time_limit_already_passed_ends_the_call_before_it_connects() {
    let mut f = Fixture::new();
    let peer = Peer::new();
    peer.listener.set_nonblocking(true).unwrap();
    let rx = f.get(
        peer.addr(),
        ClientConfig {
            time_limit: Some(Duration::ZERO),
            ..config()
        },
    );
    let error = outcome(&rx).unwrap_err();
    assert!(
        matches!(error, NetError::TimedOut(CallStage::Connect)),
        "{error}"
    );
    f.finish();
}

fn peer_sends_then_closes(bytes: &[u8]) -> NetError {
    let mut f = Fixture::new();
    let peer = Peer::new();
    let rx = f.get(peer.addr(), config());
    let (mut stream, _head) = peer.accept();
    stream.write_all(bytes).unwrap();
    drop(stream);
    let error = outcome(&rx).unwrap_err();
    f.finish();
    error
}

#[test]
fn a_peer_that_closes_in_the_head_is_reported_in_the_head() {
    let error = peer_sends_then_closes(b"HTTP/1.1 200 OK\r\nContent-Le");
    assert!(
        matches!(error, NetError::PeerClosed(CallStage::ReceiveHead)),
        "{error}"
    );
    let error = peer_sends_then_closes(b"");
    assert!(
        matches!(error, NetError::PeerClosed(CallStage::ReceiveHead)),
        "{error}"
    );
}

#[test]
fn a_peer_that_closes_in_the_body_is_reported_in_the_body() {
    let error = peer_sends_then_closes(b"HTTP/1.1 200 OK\r\nContent-Length: 10\r\n\r\nabc");
    assert!(
        matches!(error, NetError::PeerClosed(CallStage::ReceiveBody)),
        "{error}"
    );
}

#[test]
fn a_response_over_the_limits_is_reported_as_over_the_limits() {
    let mut f = Fixture::new();
    let peer = Peer::new();
    let small = ClientConfig {
        limits: Limits {
            max_header_bytes: 64,
            max_body: 10,
            ..Limits::default()
        },
        ..config()
    };
    // A head over its limit.
    let rx = f.get(peer.addr(), small);
    let (mut stream, _) = peer.accept();
    let long = "x".repeat(200);
    stream
        .write_all(format!("HTTP/1.1 200 OK\r\nX: {long}\r\nContent-Length: 0\r\n\r\n").as_bytes())
        .unwrap();
    let error = outcome(&rx).unwrap_err();
    assert!(matches!(error, NetError::OverLimit(_)), "{error}");
    // A body over its limit, refused at the head.
    let rx = f.get(peer.addr(), small);
    let (mut stream, _) = peer.accept();
    stream
        .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\n")
        .unwrap();
    let error = outcome(&rx).unwrap_err();
    assert!(matches!(error, NetError::OverLimit(_)), "{error}");
    f.finish();
}

#[test]
fn a_malformed_or_unsupported_response_is_reported_as_such() {
    let mut f = Fixture::new();
    let peer = Peer::new();
    let cases: [(&[u8], Expect); 3] = [
        (b"this is not http\r\n\r\n", |e| {
            matches!(e, NetError::Malformed(_))
        }),
        (b"HTTP/1.1 200 OK\r\n\r\nbody until close", |e| {
            matches!(e, NetError::Unsupported(_))
        }),
        (
            b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n",
            |e| matches!(e, NetError::Unsupported(_)),
        ),
    ];
    for (bytes, expect) in cases {
        let rx = f.get(peer.addr(), config());
        let (mut stream, _) = peer.accept();
        stream.write_all(bytes).unwrap();
        let error = outcome(&rx).unwrap_err();
        assert!(expect(&error), "{error}");
    }
    f.finish();
}

#[test]
fn an_interim_response_is_skipped_and_a_switch_of_protocol_is_refused() {
    let mut f = Fixture::new();
    let peer = Peer::new();
    let rx = f.get(peer.addr(), config());
    let (mut stream, _) = peer.accept();
    // The interim and the final response in two writes, the head split too.
    stream
        .write_all(b"HTTP/1.1 100 Continue\r\n\r\nHTTP/1.1 200 O")
        .unwrap();
    stream
        .write_all(b"K\r\nContent-Length: 2\r\n\r\nhi")
        .unwrap();
    let response = outcome(&rx).unwrap();
    assert_eq!(
        (response.status, response.body.as_slice()),
        (200, &b"hi"[..])
    );

    let rx = f.get(peer.addr(), config());
    let (mut stream, _) = peer.accept();
    stream
        .write_all(b"HTTP/1.1 101 Switching Protocols\r\n\r\n")
        .unwrap();
    let error = outcome(&rx).unwrap_err();
    assert!(matches!(error, NetError::UnexpectedInterim(101)), "{error}");

    let rx = f.get(peer.addr(), config());
    let (mut stream, _) = peer.accept();
    stream
        .write_all("HTTP/1.1 102 Processing\r\n\r\n".repeat(9).as_bytes())
        .unwrap();
    let error = outcome(&rx).unwrap_err();
    assert!(matches!(error, NetError::UnexpectedInterim(102)), "{error}");
    f.finish();
}

#[test]
fn a_head_split_into_single_bytes_is_read() {
    let mut f = Fixture::new();
    let peer = Peer::new();
    let rx = f.get(
        peer.addr(),
        ClientConfig {
            read_chunk: 1,
            ..config()
        },
    );
    let (mut stream, _) = peer.accept();
    stream
        .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 3\r\n\r\nabc")
        .unwrap();
    let response = outcome(&rx).unwrap();
    assert_eq!(response.body, b"abc");
    f.finish();
}

#[test]
fn calls_in_flight_together_complete_independently() {
    const CALLS: usize = 8;
    let mut f = Fixture::new();
    // The handler parks every request until all of them have arrived, then the test answers them in reverse.
    let (tx, parked) = channel::<(String, Responder)>();
    let tx = Mutex::new(tx);
    let server = serve_on(&f, move |request: Request, reply: Responder| {
        let _ = tx.lock().unwrap().send((request.target, reply));
        no_work()
    });
    let calls: Vec<(String, Receiver<Outcome>)> = (0..CALLS)
        .map(|i| {
            let target = format!("/call-{i}");
            let rx = f.call(server.addr, &get(&target), config());
            (target, rx)
        })
        .collect();
    let mut held: Vec<(String, Responder)> = (0..CALLS)
        .map(|_| parked.recv_timeout(PATIENCE).expect("a request arrived"))
        .collect();
    // All are in flight: none has an outcome yet.
    assert!(calls.iter().all(|(_, rx)| rx.try_recv().is_err()));
    held.reverse();
    for (target, reply) in held {
        let (done_tx, done) = channel();
        let body = target.clone().into_bytes();
        f.sink
            .offer(Run::frame(move || {
                reply.send(Response::with_status(200).with_body(body));
                let _ = done_tx.send(());
            }))
            .ok()
            .unwrap();
        done.recv_timeout(PATIENCE).unwrap();
        // The call whose request was answered ends, with its own response.
        let (_, rx) = calls.iter().find(|(t, _)| *t == target).unwrap();
        let response = outcome(rx).unwrap();
        assert_eq!(response.body, target.as_bytes());
    }
    server.stop();
    f.finish();
}

#[test]
fn an_operation_the_engine_refuses_ends_the_call_as_abandoned() {
    let mut f = Fixture::new();
    let peer = Peer::new();
    peer.listener.set_nonblocking(true).unwrap();
    // A completed `AsyncIo` starts nothing: the continuation is dropped, and the call with it.
    f.io.complete();
    let error = outcome(&f.get(peer.addr(), config())).unwrap_err();
    assert!(
        matches!(error, NetError::Abandoned(CallStage::Connect)),
        "{error}"
    );
    f.finish();
}
