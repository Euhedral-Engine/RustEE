//! The HTTP/1.1 server: [`serve`] takes a listener, limits and a [`Handler`] and answers every request it
//! accepts exactly once.
//!
//! The server owns no thread and no timer, and nothing in it waits. Sockets go through `AsyncIo` (whose
//! operations carry the time limits and can be cancelled); work runs as frames offered to a `QueueIngestSink`.
//! The listener's own state (the pending accept, the registry of connections, the stop) is plain state that only
//! frames ordered by the listener's hash touch; each connection's state is the same for the connection's hash (see
//! the [`connection`](crate::connection) module notes).
//!
//! # Allocation
//!
//! Per connection: its `Conn` (an `Arc`), the parser (a buffer for one head within `max_header_bytes`, and a body
//! within `max_body_bytes`), one receive buffer of `read_chunk` bytes, a slot of the registry and the socket. Per
//! request: the `Request` the parser builds, one `Responder` with its cancel flag, the bytes of the reply, the
//! handler's own frame, and one frame for each event of the connection (a `recv`, a `send`, a reply) plus the
//! buffers and boxes `AsyncIo` keeps for each operation.

use std::io;
use std::net::TcpListener;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use euhedral_core::frames::{Frame, FrameError, FrameHeader};
use euhedral_core::ingest::QueueIngestSink;
use euhedral_data_structures::sync::ClaimCell;
use euhedral_io::{AsyncIo, Cancelled, OpId, OpOptions, OwnedSock, RawSock};

use crate::connection::{Conn, Event, Responder};
use crate::message::Request;
use crate::parse::Limits;

/// The limits and sizes of a server. Every field is a setting: none is tied to a machine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ServerConfig {
    /// The most connections open at once. At the cap no accept is pending and the kernel's backlog holds the
    /// rest; an accept is armed again when a connection ends. Zero accepts nothing.
    pub max_connections: usize,
    /// The most bytes of a request's header lines (more is 431).
    pub max_header_bytes: usize,
    /// The longest request body (a `Content-Length` over it is 413 as soon as the head is read, before the
    /// body is read and before any `100 Continue`).
    pub max_body_bytes: usize,
    /// How long a whole request may take to arrive, from its first byte seen. `None` is no limit. A request
    /// still incomplete at the limit is answered `408` with `Connection: close`.
    pub read_limit: Option<Duration>,
    /// How long a connection may wait for the first byte of a request (also the wait of a new connection). The
    /// connection is closed without an answer when it passes. `None` is no limit.
    pub idle_limit: Option<Duration>,
    /// How long one reply (or the `100 Continue`) may take to go out, and how long a closing connection waits for
    /// its peer to close. The connection is closed when it passes. `None` is no limit.
    pub write_limit: Option<Duration>,
    /// The size of the buffer of each `recv`. It also bounds the bytes kept for a pipelined request while the
    /// one before it is handled.
    pub read_chunk: usize,
}

impl Default for ServerConfig {
    fn default() -> ServerConfig {
        ServerConfig {
            max_connections: 256,
            max_header_bytes: 16 * 1024,
            max_body_bytes: 1024 * 1024,
            read_limit: Some(Duration::from_secs(30)),
            idle_limit: Some(Duration::from_secs(60)),
            write_limit: Some(Duration::from_secs(30)),
            read_chunk: 16 * 1024,
        }
    }
}

/// What answers the requests of a server.
///
/// `handle` runs inside the frame of the connection's order, so it must return promptly and must not block: it
/// does the work that decides what to do, and returns the frame that does the rest, which the server offers to
/// the sink. The reply may be sent from `handle` itself, from the returned frame or from any later frame on any
/// worker; `reply` is not tied to the call. A closure of the same shape is a `Handler`.
pub trait Handler: Send + Sync + 'static {
    /// Takes a request and the right to answer it.
    fn handle(&self, request: Request, reply: Responder) -> Box<dyn Frame>;
}

impl<F> Handler for F
where
    F: Fn(Request, Responder) -> Box<dyn Frame> + Send + Sync + 'static,
{
    fn handle(&self, request: Request, reply: Responder) -> Box<dyn Frame> {
        self(request, reply)
    }
}

/// A frame that does nothing: what a handler returns when it has answered at once.
pub fn no_work() -> Box<dyn Frame> {
    Box::new(NoWork(FrameHeader::new(0)))
}

struct NoWork(FrameHeader);

impl Frame for NoWork {
    fn header(&self) -> &FrameHeader {
        &self.0
    }

    fn header_mut(&mut self) -> &mut FrameHeader {
        &mut self.0
    }
}

/// What a program can read about a running server, each a count since it started (`open` is the number now).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Counters {
    /// Connections accepted.
    pub accepted: u64,
    /// Connections open now: accepted and not yet ended.
    pub open: u64,
    /// Requests handed to the handler.
    pub requests: u64,
    /// Final replies by class (sent or attempted; a reply that could not be written counts as the `500` it became).
    pub replies_2xx: u64,
    pub replies_3xx: u64,
    pub replies_4xx: u64,
    pub replies_5xx: u64,
    /// Requests refused by the parser (each answered with its status, including `413`).
    pub parse_errors: u64,
    /// Operations that ended at their limit.
    pub limits_passed: u64,
    /// Requests whose peer went while they were handled (their cancel flag was set).
    pub cancelled: u64,
    /// Replies discarded because their request's connection had ended.
    pub discarded: u64,
    /// Accepts that failed (other than the server's own cancel on stop).
    pub accept_errors: u64,
}

/// The cells behind [`Counters`], shared by every frame of the server.
#[derive(Debug, Default)]
pub(crate) struct CounterCells {
    pub accepted: AtomicU64,
    pub open: AtomicU64,
    pub requests: AtomicU64,
    pub replies: [AtomicU64; 4],
    pub parse_errors: AtomicU64,
    pub limits_passed: AtomicU64,
    pub cancelled: AtomicU64,
    pub discarded: AtomicU64,
    pub accept_errors: AtomicU64,
}

impl CounterCells {
    fn snapshot(&self) -> Counters {
        let get = |cell: &AtomicU64| cell.load(Ordering::Relaxed);
        Counters {
            accepted: get(&self.accepted),
            open: get(&self.open),
            requests: get(&self.requests),
            replies_2xx: get(&self.replies[0]),
            replies_3xx: get(&self.replies[1]),
            replies_4xx: get(&self.replies[2]),
            replies_5xx: get(&self.replies[3]),
            parse_errors: get(&self.parse_errors),
            limits_passed: get(&self.limits_passed),
            cancelled: get(&self.cancelled),
            discarded: get(&self.discarded),
            accept_errors: get(&self.accept_errors),
        }
    }
}

/// What reaches the listener's state.
pub(crate) enum ListenerEvent {
    /// The server started: arm the accept.
    Start,
    /// An accept ended.
    Accepted(io::Result<OwnedSock>),
    /// The connection in registry slot `usize` ended.
    Closed(usize),
    /// Stop; the frame is offered once the last connection is gone.
    Stop(Box<dyn Frame>),
}

/// The plain state of the listener.
struct ListenerState {
    listener: Option<TcpListener>,
    /// The accept in flight.
    accepting: Option<OpId>,
    /// The registry of open connections, by slot.
    conns: Vec<Option<Arc<Conn>>>,
    free: Vec<usize>,
    open: usize,
    stopping: bool,
    finished: bool,
    /// The frames to offer when the server is stopped.
    stopped: Vec<Box<dyn Frame>>,
}

/// What a server's frames share.
pub(crate) struct Shared {
    pub io: Arc<AsyncIo>,
    pub sink: Arc<QueueIngestSink>,
    pub config: ServerConfig,
    pub limits: Limits,
    pub handler: Arc<dyn Handler>,
    pub counters: CounterCells,
    listener_hash: u64,
    salt: u64,
    next_conn: AtomicU64,
    listener: ClaimCell<ListenerState>,
}

/// A frame that carries an event to the listener's order.
struct ListenerFrame {
    header: FrameHeader,
    shared: Arc<Shared>,
    event: Option<ListenerEvent>,
}

impl Frame for ListenerFrame {
    fn header(&self) -> &FrameHeader {
        &self.header
    }

    fn header_mut(&mut self) -> &mut FrameHeader {
        &mut self.header
    }

    fn execute(&mut self) -> Result<(), FrameError> {
        if let Some(event) = self.event.take() {
            self.shared.run(event);
        }
        Ok(())
    }
}

/// A well-spread hash of `x` (the finaliser of SplitMix64), so that the ids of connections land on all cores.
fn mix(mut x: u64) -> u64 {
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

#[cfg(unix)]
fn raw_listener(listener: &TcpListener) -> RawSock {
    std::os::fd::AsRawFd::as_raw_fd(listener)
}

#[cfg(windows)]
fn raw_listener(listener: &TcpListener) -> RawSock {
    std::os::windows::io::AsRawSocket::as_raw_socket(listener)
}

impl Shared {
    /// Counts a final reply of `status`.
    pub(crate) fn count_reply(&self, status: u16) {
        let class = usize::from(status / 100).clamp(2, 5) - 2;
        self.counters.replies[class].fetch_add(1, Ordering::Relaxed);
    }

    /// The frame that carries `event` to the listener's order.
    fn listener_frame(self: &Arc<Self>, event: ListenerEvent) -> Box<dyn Frame> {
        Box::new(ListenerFrame {
            header: FrameHeader::new(self.listener_hash),
            shared: self.clone(),
            event: Some(event),
        })
    }

    /// Offers `event` to the listener's order. Hands the event's frame back when the sink has completed.
    pub(crate) fn post_listener(self: &Arc<Self>, event: ListenerEvent) {
        let frame = self.listener_frame(event);
        let _ = self.sink.offer_seeded(self.listener_hash, frame);
    }

    fn run(self: &Arc<Self>, event: ListenerEvent) {
        let Some(mut guard) = self.listener.try_claim() else {
            // Frames of one hash run one at a time: an ordering that was not kept. Try again, never wait.
            self.post_listener(event);
            return;
        };
        let ls: &mut ListenerState = &mut guard;
        match event {
            ListenerEvent::Start => self.arm_accept(ls),
            ListenerEvent::Accepted(result) => {
                ls.accepting = None;
                match result {
                    Ok(socket) => self.accepted(ls, socket),
                    Err(error) => {
                        if !Cancelled::is(&error) {
                            self.counters.accept_errors.fetch_add(1, Ordering::Relaxed);
                            // With connections open, one ending arms the accept again; with none, nothing
                            // would, so it is armed now (an error that persists with none open repeats).
                            if ls.open == 0 {
                                self.arm_accept(ls);
                            }
                        }
                    }
                }
                self.check_finished(ls);
            }
            ListenerEvent::Closed(slot) => {
                if let Some(entry) = ls.conns.get_mut(slot)
                    && entry.take().is_some()
                {
                    ls.free.push(slot);
                    ls.open -= 1;
                    self.counters.open.store(ls.open as u64, Ordering::Relaxed);
                }
                self.arm_accept(ls);
                self.check_finished(ls);
            }
            ListenerEvent::Stop(frame) => {
                if ls.finished {
                    // Already stopped: reported at once.
                    let _ = self.sink.offer(frame);
                    return;
                }
                ls.stopped.push(frame);
                if !ls.stopping {
                    ls.stopping = true;
                    if let Some(id) = ls.accepting {
                        self.io.cancel(id);
                    }
                    for conn in ls.conns.iter().flatten() {
                        conn.post(Event::Stop);
                    }
                }
                self.check_finished(ls);
            }
        }
    }

    /// Arms the accept unless one is pending, the server is stopping or the cap is reached.
    fn arm_accept(self: &Arc<Self>, ls: &mut ListenerState) {
        if ls.stopping || ls.accepting.is_some() || ls.open >= self.config.max_connections {
            return;
        }
        let Some(listener) = &ls.listener else {
            return;
        };
        let shared = self.clone();
        let started = self.io.accept_with(
            raw_listener(listener),
            OpOptions::default(),
            Box::new(move |result| shared.listener_frame(ListenerEvent::Accepted(result))),
        );
        match started {
            Ok(id) => ls.accepting = Some(id),
            Err(_) => {
                self.counters.accept_errors.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Takes in an accepted socket: registers its connection and has it start reading.
    fn accepted(self: &Arc<Self>, ls: &mut ListenerState, socket: OwnedSock) {
        if ls.stopping {
            // Dropped: the peer sees the connection close.
            return;
        }
        let stream = std::net::TcpStream::from(socket);
        let id = self.next_conn.fetch_add(1, Ordering::Relaxed);
        let slot = match ls.free.pop() {
            Some(slot) => slot,
            None => {
                ls.conns.push(None);
                ls.conns.len() - 1
            }
        };
        let conn = Arc::new(Conn::new(
            self.clone(),
            mix(self.salt ^ id.wrapping_mul(0x9E37_79B9_7F4A_7C15)),
            stream,
            slot,
        ));
        ls.conns[slot] = Some(conn.clone());
        ls.open += 1;
        self.counters.accepted.fetch_add(1, Ordering::Relaxed);
        self.counters.open.store(ls.open as u64, Ordering::Relaxed);
        conn.post(Event::Start);
        self.arm_accept(ls);
    }

    /// When the server is stopping and nothing is left, closes the listener and offers the stop frames.
    fn check_finished(self: &Arc<Self>, ls: &mut ListenerState) {
        if ls.stopping && !ls.finished && ls.open == 0 && ls.accepting.is_none() {
            ls.finished = true;
            ls.listener = None;
            for frame in ls.stopped.drain(..) {
                let _ = self.sink.offer(frame);
            }
        }
    }
}

/// A running server: reads its counters and stops it. Cloning gives another handle to the same server;
/// dropping a handle does not stop the server.
#[derive(Clone)]
pub struct ServerHandle {
    shared: Arc<Shared>,
}

impl std::fmt::Debug for ServerHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ServerHandle")
            .field("counters", &self.counters())
            .finish_non_exhaustive()
    }
}

impl ServerHandle {
    /// The counters now. Each is read on its own, so a snapshot taken while the server works need not be one
    /// instant.
    pub fn counters(&self) -> Counters {
        self.shared.counters.snapshot()
    }

    /// Stops the server and returns at once. The accept in flight is cancelled and no more connections are
    /// taken; connections that wait for a request (idle, or part-way through reading one, or lingering after
    /// their last reply) are closed; a request being handled is answered, with `Connection: close`, and its
    /// connection closes after the reply. When the last connection is gone and the accept has ended, `stopped`
    /// is offered to the sink, once. A second `stop` adds another frame, offered at the same time (or at
    /// once if the server is already stopped).
    ///
    /// A `Responder` still alive after the server stopped is not special: its reply goes through the
    /// connection's order as always, and is sent if its connection is still open and discarded otherwise. A
    /// handler that never answers keeps the server from finishing, until its peer leaves. The sink must live
    /// until the stop is done: a sink that completed refuses the events the server needs.
    pub fn stop(&self, stopped: Box<dyn Frame>) {
        self.shared.post_listener(ListenerEvent::Stop(stopped));
    }
}

/// Starts serving: takes `listener` (a bound, listening socket), accepts connections on `io` and answers
/// their requests with `handler`. Work runs as frames offered to `sink`, which a lattice drains; `io` must be
/// attached to a lattice too (its continuations are frames).
///
/// Returns an error if the sink refuses the first frame (it has completed). The server then does nothing.
///
/// A connection that the server refuses to read on (`max_connections`) waits in the kernel's backlog.
pub fn serve(
    io: Arc<AsyncIo>,
    sink: Arc<QueueIngestSink>,
    listener: TcpListener,
    config: ServerConfig,
    handler: Arc<dyn Handler>,
) -> io::Result<ServerHandle> {
    static SERVERS: AtomicU64 = AtomicU64::new(1);
    let server = SERVERS.fetch_add(1, Ordering::Relaxed);
    let limits = Limits {
        max_header_bytes: config.max_header_bytes,
        max_body: config.max_body_bytes,
        ..Limits::default()
    };
    let shared = Arc::new(Shared {
        io,
        sink,
        config,
        limits,
        handler,
        counters: CounterCells::default(),
        listener_hash: mix(server),
        salt: mix(server ^ 0xA5A5_A5A5_A5A5_A5A5),
        next_conn: AtomicU64::new(0),
        listener: ClaimCell::new(ListenerState {
            listener: Some(listener),
            accepting: None,
            conns: Vec::new(),
            free: Vec::new(),
            open: 0,
            stopping: false,
            finished: false,
            stopped: Vec::new(),
        }),
    });
    let start = shared.listener_frame(ListenerEvent::Start);
    if shared
        .sink
        .offer_seeded(shared.listener_hash, start)
        .is_err()
    {
        return Err(io::Error::other("the sink has completed"));
    }
    Ok(ServerHandle { shared })
}
