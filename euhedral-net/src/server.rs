//! The HTTP/1.1 server: [`serve`] takes a listener, limits and a [`Handler`] and answers every request it
//! accepts exactly once.
//!
//! The server owns no thread and no timer, and nothing in it waits. Sockets go through `AsyncIo` (whose
//! operations carry the time limits and can be cancelled); the handler's frames run as frames offered to a
//! `QueueIngestSink`. The listener and every connection are each a source of the lattice with an inbox (see
//! [`station`](crate::station)): the listener is attached by [`serve`], a connection when it is accepted, and a
//! connection is detached when it ends. The listener's own state (the pending accept, the registry of
//! connections, the stop) is plain state that only the listener's key frame touches, and a connection's state is
//! the same for its key frame, so different connections are served at once and nothing waits on a cell.
//!
//! A failed accept is tried again after `accept_backoff`, through a timer of the lattice's own
//! (`euhedral_graph::Completions`), so the server keeps no timer and no thread of its own.
//!
//! # Allocation
//!
//! Per connection: its `Conn` (an `Arc`), the parser (a buffer for one head within `max_header_bytes`, and a body
//! within `max_body_bytes`), one receive buffer of `read_chunk` bytes, a slot of the registry and the socket. Per
//! request: the `Request` the parser builds, one `Responder` with its cancel flag, the bytes of the reply, the
//! handler's own frame, and, for each operation of the connection (a `recv`, a `send`), the frame that does nothing
//! which its continuation returns, plus the buffers and boxes `AsyncIo` keeps for each operation.

use std::io;
use std::net::{TcpListener, TcpStream};
#[cfg(unix)]
use std::os::fd::AsRawFd;
#[cfg(windows)]
use std::os::windows::io::AsRawSocket;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use euhedral_core::control_plane::ControlPlaneLattice;
use euhedral_core::frames::{Frame, FrameHeader};
use euhedral_core::generics::{LatticeReceiver, LatticeSource, SourceCell};
use euhedral_core::ingest::QueueIngestSink;
use euhedral_data_structures::atomics::CachePadded;
use euhedral_data_structures::sync::{HotScope, LifecycleMutex};
use euhedral_graph::Completions;
use euhedral_io::{AsyncIo, Cancelled, OpId, OpOptions, OwnedSock, RawSock};

use crate::connection::{Conn, Event, Responder};
use crate::message::Request;
use crate::parse::Limits;
use crate::station::{Drive, Post, Station};

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
    /// How long one reply (or the `100 Continue`) may take to go out. The connection is closed when it passes.
    /// `None` is no limit.
    pub write_limit: Option<Duration>,
    /// How long, from its start, the whole close of a connection after its last reply may take: the server shuts
    /// its writing side and reads and drops what the peer still sends until the peer closes or this passes. It is
    /// always a bound (a peer that sends a little at a time, or never closes, cannot hold the connection).
    pub linger_limit: Duration,
    /// The bound of that close when the server is stopping, if it is shorter than `linger_limit`; a close
    /// already in progress is cut to it when the stop comes.
    pub stop_linger_limit: Duration,
    /// How long the server waits after a failed accept before it accepts again.
    pub accept_backoff: Duration,
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
            linger_limit: Duration::from_secs(5),
            stop_linger_limit: Duration::from_secs(1),
            accept_backoff: Duration::from_millis(100),
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

/// The counters of every connection's request, written by every frame that handles one.
#[derive(Debug, Default)]
pub(crate) struct RequestCells {
    pub requests: AtomicU64,
    pub replies: [AtomicU64; 4],
}

/// The counters of the connections themselves, written by the listener's order.
#[derive(Debug, Default)]
pub(crate) struct ConnectionCells {
    pub accepted: AtomicU64,
    pub open: AtomicU64,
    pub accept_errors: AtomicU64,
}

/// The counters of what rarely happens.
#[derive(Debug, Default)]
pub(crate) struct RareCells {
    pub parse_errors: AtomicU64,
    pub limits_passed: AtomicU64,
    pub cancelled: AtomicU64,
    pub discarded: AtomicU64,
}

/// The cells behind [`Counters`], shared by every frame of the server and read from outside. They are grouped by how
/// often they are written and each group has its own cache line, so the counters every request writes do not share
/// a line with those of the listener or with those that rarely change.
#[derive(Debug, Default)]
pub(crate) struct CounterCells {
    pub request: CachePadded<RequestCells>,
    pub connection: CachePadded<ConnectionCells>,
    pub rare: CachePadded<RareCells>,
}

impl CounterCells {
    fn snapshot(&self) -> Counters {
        let get = |cell: &AtomicU64| cell.load(Ordering::Relaxed);
        Counters {
            accepted: get(&self.connection.accepted),
            open: get(&self.connection.open),
            requests: get(&self.request.requests),
            replies_2xx: get(&self.request.replies[0]),
            replies_3xx: get(&self.request.replies[1]),
            replies_4xx: get(&self.request.replies[2]),
            replies_5xx: get(&self.request.replies[3]),
            parse_errors: get(&self.rare.parse_errors),
            limits_passed: get(&self.rare.limits_passed),
            cancelled: get(&self.rare.cancelled),
            discarded: get(&self.rare.discarded),
            accept_errors: get(&self.connection.accept_errors),
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
    /// Stop (the frames to offer when it is done are in `Shared::stops`).
    Stop,
    /// The back-off after a failed accept is over.
    Retry,
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
    /// A retry of the accept is waiting on the timer.
    retrying: bool,
}

/// The frames to offer when the server is stopped, and whether it is.
struct Stops {
    done: bool,
    frames: Vec<Box<dyn Frame>>,
}

/// What a server's frames share.
pub(crate) struct Shared {
    pub io: Arc<AsyncIo>,
    /// The lattice the connections attach their sources to.
    lattice: Arc<ControlPlaneLattice>,
    pub sink: Arc<QueueIngestSink>,
    pub config: ServerConfig,
    pub limits: Limits,
    pub handler: Arc<dyn Handler>,
    pub counters: CounterCells,
    timers: Arc<Completions>,
    /// A stop was asked for; set by `ServerHandle::stop` before its event is posted, so a reply given after the
    /// call returns already finds it.
    pub stop_requested: AtomicBool,
    /// The listener's state, entered by its key frame only.
    listener: SourceCell<ListenerState>,
    station: Station<Shared>,
    /// Stop is lifecycle work: its frames wait in a list that the listener's finish and `ServerHandle::stop` share.
    stops: LifecycleMutex<Stops>,
}

#[cfg(unix)]
fn raw_listener(listener: &TcpListener) -> RawSock {
    listener.as_raw_fd()
}

#[cfg(windows)]
fn raw_listener(listener: &TcpListener) -> RawSock {
    listener.as_raw_socket()
}

impl Shared {
    /// Counts a final reply of `status`.
    pub(crate) fn count_reply(&self, status: u16) {
        let class = usize::from(status / 100).clamp(2, 5) - 2;
        self.counters.request.replies[class].fetch_add(1, Ordering::Relaxed);
    }

    /// Pushes `event` into the listener's inbox.
    pub(crate) fn post_listener(&self, event: ListenerEvent) {
        self.station.post(event);
    }

    fn run_event(self: &Arc<Self>, event: ListenerEvent) {
        // Only the listener's key frame runs events, one at a time: the entry cannot find the cell taken.
        let mut guard = self.listener.enter();
        let ls: &mut ListenerState = &mut guard;
        match event {
            ListenerEvent::Start => self.arm_accept(ls),
            ListenerEvent::Accepted(result) => {
                ls.accepting = None;
                match result {
                    Ok(socket) => self.accepted(ls, socket),
                    Err(error) => {
                        if !Cancelled::is(&error) {
                            self.accept_failed(ls);
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
                    self.counters
                        .connection
                        .open
                        .store(ls.open as u64, Ordering::Relaxed);
                }
                self.arm_accept(ls);
                self.check_finished(ls);
            }
            ListenerEvent::Retry => {
                ls.retrying = false;
                self.arm_accept(ls);
                self.check_finished(ls);
            }
            ListenerEvent::Stop => {
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
        if ls.stopping
            || ls.accepting.is_some()
            || ls.retrying
            || ls.open >= self.config.max_connections
        {
            return;
        }
        let Some(listener) = &ls.listener else {
            return;
        };
        let shared = self.clone();
        let started = self.io.accept_with(
            raw_listener(listener),
            OpOptions::default(),
            Box::new(move |result| {
                shared.post_listener(ListenerEvent::Accepted(result));
                no_work()
            }),
        );
        match started {
            Ok(id) => ls.accepting = Some(id),
            Err(_) => self.accept_failed(ls),
        }
    }

    /// An accept failed (it ended in an error, or could not start): counted, and tried again after the back-off, so
    /// that neither a persistent error spins nor the server is left without an accept.
    fn accept_failed(self: &Arc<Self>, ls: &mut ListenerState) {
        self.counters
            .connection
            .accept_errors
            .fetch_add(1, Ordering::Relaxed);
        if ls.stopping || ls.retrying {
            return;
        }
        ls.retrying = true;
        let frame = Post::new(self.clone(), ListenerEvent::Retry);
        self.timers
            .at(Instant::now() + self.config.accept_backoff, frame);
    }

    /// Takes in an accepted socket: registers its connection and has it start reading.
    fn accepted(self: &Arc<Self>, ls: &mut ListenerState, socket: OwnedSock) {
        if ls.stopping {
            // Dropped: the peer sees the connection close.
            return;
        }
        let stream = TcpStream::from(socket);
        let slot = match ls.free.pop() {
            Some(slot) => slot,
            None => {
                ls.conns.push(None);
                ls.conns.len() - 1
            }
        };
        let conn = Conn::new(self.clone(), stream, slot);
        // Attaching a source is cold work, and the lattice says so itself; it refuses once it is closed.
        if self.lattice.add_upstream(conn.clone()).is_err() {
            ls.free.push(slot);
            return;
        }
        ls.conns[slot] = Some(conn.clone());
        ls.open += 1;
        self.counters
            .connection
            .accepted
            .fetch_add(1, Ordering::Relaxed);
        self.counters
            .connection
            .open
            .store(ls.open as u64, Ordering::Relaxed);
        conn.post(Event::Start);
        self.arm_accept(ls);
    }

    /// When the server is stopping and nothing is left, closes the listener and offers the stop frames.
    fn check_finished(self: &Arc<Self>, ls: &mut ListenerState) {
        if ls.stopping && !ls.finished && ls.open == 0 && ls.accepting.is_none() && !ls.retrying {
            ls.finished = true;
            ls.listener = None;
            let frames = {
                let _cold = HotScope::leave();
                let mut stops = self.stops.lock();
                stops.done = true;
                std::mem::take(&mut stops.frames)
            };
            for frame in frames {
                let _ = self.sink.offer(frame);
            }
            // The listener's source ends: the lattice lets go of it. Its key is running this, and goes home.
            self.station.complete();
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
    /// The notification means that no frame of the server will run after it: every connection is gone, no accept
    /// or retry is pending, and every `Responder` handed to a handler has been used or dropped and its reply has
    /// been through its connection's order (sent, or discarded and counted). So a handler that never answers (and
    /// keeps its `Responder`) keeps the server from finishing, even when its peer is gone. A reply after the stop
    /// started is sent if its connection is open (the request is answered with `Connection: close`) and
    /// discarded otherwise. A lingering close is cut to `stop_linger_limit`. The sink must live until the stop is
    /// done: a sink that completed refuses the events the server needs.
    pub fn stop(&self, stopped: Box<dyn Frame>) {
        let shared = &self.shared;
        shared.stop_requested.store(true, Ordering::Release);
        let _cold = HotScope::leave();
        let mut stops = shared.stops.lock();
        if stops.done {
            // Already stopped, and the listener's source is gone: reported at once.
            drop(stops);
            let _ = shared.sink.offer(stopped);
            return;
        }
        stops.frames.push(stopped);
        drop(stops);
        shared.post_listener(ListenerEvent::Stop);
    }
}

/// Starts serving: takes `listener` (a bound, listening socket), accepts connections on `io` and answers
/// their requests with `handler`. Work runs as frames offered to `sink`, which a lattice drains; `io` must be
/// attached to a lattice too (its continuations are frames), and so must `timers`, a [`Completions`] the server
/// uses for the back-off after a failed accept (the lattice's own timer: the server keeps none).
///
/// Returns an error if the lattice refuses the listener's source (it is closed). The server then does nothing.
///
/// A connection that the server refuses to read on (`max_connections`) waits in the kernel's backlog.
pub fn serve(
    lattice: Arc<ControlPlaneLattice>,
    io: Arc<AsyncIo>,
    sink: Arc<QueueIngestSink>,
    listener: TcpListener,
    config: ServerConfig,
    handler: Arc<dyn Handler>,
    timers: Arc<Completions>,
) -> io::Result<ServerHandle> {
    let limits = Limits {
        max_header_bytes: config.max_header_bytes,
        max_body: config.max_body_bytes,
        ..Limits::default()
    };
    let shared = Arc::new_cyclic(|me| Shared {
        io,
        lattice,
        sink,
        config,
        limits,
        handler,
        counters: CounterCells::default(),
        timers,
        stop_requested: AtomicBool::new(false),
        listener: SourceCell::new(ListenerState {
            listener: Some(listener),
            accepting: None,
            conns: Vec::new(),
            free: Vec::new(),
            open: 0,
            stopping: false,
            finished: false,
            retrying: false,
        }),
        station: Station::new(me.clone()),
        stops: LifecycleMutex::new(Stops {
            done: false,
            frames: Vec::new(),
        }),
    });
    shared.post_listener(ListenerEvent::Start);
    shared
        .lattice
        .add_upstream(shared.clone())
        .map_err(|error| io::Error::other(error.to_string()))?;
    Ok(ServerHandle { shared })
}

impl Drive for Shared {
    type Event = ListenerEvent;

    fn station(&self) -> &Station<Shared> {
        &self.station
    }

    fn run(self: &Arc<Self>, event: ListenerEvent) {
        self.run_event(event);
    }
}

impl LatticeSource for Shared {
    fn add_downstream(self: Arc<Self>, downstream: Arc<dyn LatticeReceiver>) {
        let me: Arc<dyn LatticeSource> = self.clone();
        self.station.add_downstream(me, downstream);
    }

    /// Hands out nothing: the listener's work is its key frame, which only `request` pushes.
    fn pull(
        &self,
        _consumer: &mut dyn FnMut(Box<dyn Frame>),
        _stop: &mut dyn FnMut(&dyn Frame) -> bool,
        _demand: u64,
    ) -> u64 {
        0
    }

    fn request(&self, demand: u64) {
        self.station.request(demand);
    }

    fn complete(&self) {
        self.station.complete();
    }

    fn is_complete(&self) -> bool {
        self.station.is_complete()
    }
}
