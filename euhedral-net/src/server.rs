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

use std::fmt;
use std::io;
use std::mem::take;
use std::net::{TcpListener, TcpStream};
#[cfg(unix)]
use std::os::fd::AsRawFd;
#[cfg(windows)]
use std::os::windows::io::AsRawSocket;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Weak};
use std::time::{Duration, Instant};

use euhedral_core::control_plane::ControlPlaneLattice;
use euhedral_core::frames::{Frame, FrameHeader};
use euhedral_core::generics::{LatticeReceiver, LatticeSource, SourceCell};
use euhedral_core::ingest::QueueIngestSink;
use euhedral_data_structures::atomics::CachePadded;
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
    /// Stop; the frame is offered when the server is stopped.
    Stop(Box<dyn Frame>),
    /// The back-off after a failed accept is over.
    Retry,
    /// The last [`ServerHandle`] is gone: nothing can ask for a stop any more.
    Released,
}

/// What the caller owns and the server uses: the sockets' engine, the sink and the handler. Each connection and the
/// listener hold one `Arc` of it and let go of it when they end, so that when the stop is reported nothing of the
/// caller's is held by the server any more (but the sink, which reports it; see [`ServerHandle::stop`]).
pub(crate) struct Ctx {
    pub io: Arc<AsyncIo>,
    pub sink: Arc<QueueIngestSink>,
    pub handler: Arc<dyn Handler>,
}

/// The plain state of the listener: only its key frame touches it.
struct ListenerState {
    listener: Option<TcpListener>,
    /// The caller's things, until the server is finished.
    ctx: Option<Arc<Ctx>>,
    /// The sink, kept to report a stop (also one asked for after the server finished) until the listener ends.
    sink: Option<Arc<QueueIngestSink>>,
    /// The lattice's own timers, for the back-off after a failed accept, until the server is finished.
    timers: Option<Arc<Completions>>,
    /// The lattice the connections attach to. Weak: the lattice holds this source, so a strong one would be a
    /// cycle, and the server must not keep the caller's lattice alive.
    lattice: Weak<ControlPlaneLattice>,
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
    /// The last handle is gone.
    released: bool,
    /// The listener's source has ended.
    completed: bool,
    /// The frames to offer when the server is stopped.
    stops: Vec<Box<dyn Frame>>,
}

/// What a server's frames share. Nothing in it belongs to the caller.
pub(crate) struct Shared {
    pub config: ServerConfig,
    pub limits: Limits,
    pub counters: CounterCells,
    /// A stop was asked for; set by `ServerHandle::stop` before its event is posted, so a reply given after the
    /// call returns already finds it. Read by every connection's frames, which run in parallel with the listener.
    pub stop_requested: AtomicBool,
    /// The listener's state, entered by its key frame only.
    listener: SourceCell<ListenerState>,
    station: Station<Shared>,
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
            }
            ListenerEvent::Retry => {
                ls.retrying = false;
                self.arm_accept(ls);
            }
            ListenerEvent::Stop(frame) => {
                if let (true, Some(sink)) = (ls.finished, &ls.sink) {
                    // Already stopped: reported at once.
                    let _ = sink.offer(frame);
                } else {
                    ls.stops.push(frame);
                    self.begin_stop(ls);
                }
            }
            ListenerEvent::Released => ls.released = true,
        }
        self.check_finished(ls);
    }

    /// Stops accepting and tells every connection.
    fn begin_stop(self: &Arc<Self>, ls: &mut ListenerState) {
        if ls.stopping {
            return;
        }
        ls.stopping = true;
        if let (Some(id), Some(ctx)) = (ls.accepting, &ls.ctx) {
            ctx.io.cancel(id);
        }
        for conn in ls.conns.iter().flatten() {
            conn.post(Event::Stop);
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
        let (Some(listener), Some(ctx)) = (&ls.listener, &ls.ctx) else {
            return;
        };
        let shared = self.clone();
        let started = ctx.io.accept_with(
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
        let Some(timers) = &ls.timers else {
            return;
        };
        ls.retrying = true;
        let frame = Post::new(self.clone(), ListenerEvent::Retry);
        timers.at(Instant::now() + self.config.accept_backoff, frame);
    }

    /// Takes in an accepted socket: registers its connection and has it start reading. A lattice that is gone or
    /// closed refuses the connection (the socket is dropped) and ends the server as a stop does: nothing would
    /// run its frames.
    fn accepted(self: &Arc<Self>, ls: &mut ListenerState, socket: OwnedSock) {
        let Some(ctx) = ls.ctx.clone().filter(|_| !ls.stopping) else {
            // Dropped: the peer sees the connection close.
            return;
        };
        let stream = TcpStream::from(socket);
        let slot = match ls.free.pop() {
            Some(slot) => slot,
            None => {
                ls.conns.push(None);
                ls.conns.len() - 1
            }
        };
        let conn = Conn::new(self.clone(), ctx, stream, slot);
        // Attaching a source is cold work, and the lattice says so itself. The strong reference lives for this call
        // only.
        let attached = ls
            .lattice
            .upgrade()
            .is_some_and(|lattice| lattice.add_upstream(conn.clone()).is_ok());
        if !attached {
            ls.free.push(slot);
            self.begin_stop(ls);
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

    /// When the server is stopping and nothing is left, lets go of everything of the caller's but the sink, and
    /// then offers the stop frames: the last thing the server does. The listener's source ends when the last
    /// handle is gone too.
    fn check_finished(self: &Arc<Self>, ls: &mut ListenerState) {
        if ls.stopping && !ls.finished && ls.open == 0 && ls.accepting.is_none() && !ls.retrying {
            ls.finished = true;
            ls.listener = None;
            ls.timers = None;
            ls.ctx = None;
            if let Some(sink) = &ls.sink {
                for frame in take(&mut ls.stops) {
                    let _ = sink.offer(frame);
                }
            }
        }
        if ls.finished && ls.released && !ls.completed {
            ls.completed = true;
            ls.sink = None;
            // The listener's source ends: the lattice lets go of it. Its key is running this, and goes home.
            self.station.complete();
        }
    }
}

/// What the handles of a server share: the last of them going away tells the listener.
struct HandleCore {
    shared: Arc<Shared>,
}

impl Drop for HandleCore {
    fn drop(&mut self) {
        self.shared.post_listener(ListenerEvent::Released);
    }
}

/// A running server: reads its counters and stops it. Cloning gives another handle to the same server;
/// dropping a handle does not stop the server. The server's listener stays attached to the lattice until its last
/// handle is dropped after a stop, so that a stop asked for at any time is heard.
#[derive(Clone)]
pub struct ServerHandle {
    core: Arc<HandleCore>,
}

impl fmt::Debug for ServerHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ServerHandle")
            .field("counters", &self.counters())
            .finish_non_exhaustive()
    }
}

impl ServerHandle {
    /// The counters now. Each is read on its own, so a snapshot taken while the server works need not be one
    /// instant.
    pub fn counters(&self) -> Counters {
        self.core.shared.counters.snapshot()
    }

    /// Stops the server and returns at once. The accept in flight is cancelled and no more connections are
    /// taken; connections that wait for a request (idle, or part-way through reading one, or lingering after
    /// their last reply) are closed; a request being handled is answered, with `Connection: close`, and its
    /// connection closes after the reply. When the last connection is gone and the accept has ended, `stopped`
    /// is offered to the sink, once. A second `stop` adds another frame, offered at the same time (or at
    /// once if the server is already stopped).
    ///
    /// The notification is the last thing the server does. Before it: every connection is gone and its source
    /// has ended, no accept or retry is pending, every `Responder` handed to a handler has been used or dropped
    /// and its reply has been through its connection's order (sent, or discarded and counted), and the server
    /// has let go of the listener socket, the handler, the `AsyncIo`, the timers and the lattice. So a handler
    /// that never answers (and keeps its `Responder`) keeps the server from finishing, even when its peer is
    /// gone. What can still run afterwards is only the server's own bookkeeping, which touches nothing of the
    /// caller's, and the frames the handler returned (which are the handler's). A reply after the stop started
    /// is sent if its connection is open (the request is answered with `Connection: close`) and discarded
    /// otherwise. A lingering close is cut to `stop_linger_limit`. The sink is the one thing the server keeps: it
    /// reports the stop, and one asked for later, until the last handle is dropped; it must live that long.
    pub fn stop(&self, stopped: Box<dyn Frame>) {
        let shared = &self.core.shared;
        shared.stop_requested.store(true, Ordering::Release);
        shared.post_listener(ListenerEvent::Stop(stopped));
    }
}

/// Starts serving: takes `listener` (a bound, listening socket), accepts connections on `io` and answers
/// their requests with `handler`. Work runs as frames offered to `sink`, which `lattice` drains; `io` must be
/// attached to the lattice too (its continuations are frames), and so must `timers`, a [`Completions`] the server
/// uses for the back-off after a failed accept (the lattice's own timer: the server keeps none).
///
/// The listener, and each connection when it is accepted, becomes a source of `lattice`. The server holds the
/// lattice weakly: the caller keeps it alive while the server runs. If the lattice is gone or closed when a
/// connection is accepted, the connection is refused and the server ends as if stopped.
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
    let state = ListenerState {
        listener: Some(listener),
        ctx: Some(Arc::new(Ctx {
            io,
            sink: sink.clone(),
            handler,
        })),
        sink: Some(sink),
        timers: Some(timers),
        lattice: Arc::downgrade(&lattice),
        accepting: None,
        conns: Vec::new(),
        free: Vec::new(),
        open: 0,
        stopping: false,
        finished: false,
        retrying: false,
        released: false,
        completed: false,
        stops: Vec::new(),
    };
    let shared = Arc::new_cyclic(|me| Shared {
        config,
        limits,
        counters: CounterCells::default(),
        stop_requested: AtomicBool::new(false),
        listener: SourceCell::new(state),
        station: Station::new(me.clone()),
    });
    lattice
        .add_upstream(shared.clone())
        .map_err(|error| io::Error::other(error.to_string()))?;
    shared.post_listener(ListenerEvent::Start);
    Ok(ServerHandle {
        core: Arc::new(HandleCore { shared }),
    })
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
