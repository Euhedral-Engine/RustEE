//! One connection of the server: its state, the events that change it, and the `Responder` a handler answers
//! through.
//!
//! # Who owns what
//!
//! A [`Conn`] is a source of the lattice of its own, attached when the connection is accepted and detached when it
//! ends (see [`station`](crate::station)). Everything that changes lives in one [`ConnState`], which only the
//! connection's *key frame* touches: producers (the continuation of a `recv` or a `send`, a `Responder`, the stop)
//! push an [`Event`] into the connection's inbox and return; the key, which the lattice runs on some worker when
//! the inbox holds something and which is never in two places, drains the inbox and runs the events one at a time.
//! The state is plain: the parser, the bytes read and not yet parsed, the socket, the ids of the operations in
//! flight and the phase need no lock and no atomic of their own. Two connections are two sources, so they run in
//! parallel. The `SourceCell` that holds the state is *entered* by the key and by nothing else, and an entry that
//! found it taken would break that and panic ([`SourceCell::enter`]).
//!
//! Operations of one connection that end in an order the code did not expect are still handled: the continuation
//! of an `AsyncIo` operation and a `Responder` push from different places, so each event is handled in whatever
//! phase the connection is in.
//!
//! What the continuation of an operation does is push the event and return a frame that does nothing (the engine
//! wants a frame back): that frame is the cost of one completion on top of the key's own run.
//!
//! # Phases
//!
//! - `Reading`: no request is being handled. One `recv` is in flight, with the idle limit until the first byte
//!   of a request, then with what is left of the read limit.
//! - `Continuing`: the `100 Continue` is being sent; nothing is read until it is out.
//! - `Handling`: the handler has the request. One `recv` is in flight (without a limit) to see the peer go; bytes
//!   it brings are kept for the next request and no further `recv` is issued. At most one `recv` and at most one
//!   `send` are ever in flight.
//! - `Streaming`: the handler answered with [`Responder::stream`]: the head (and any first piece) is sent, and the
//!   body follows piece by piece, one send in flight at a time, each started by an event the handler's
//!   [`ReplyStream`] posts. The watch `recv` of `Handling` stays pending and means what it did there (the cancel flag,
//!   a half-close that lets the stream finish, a broken peer that ends it). The stream ends by its `end` (the last
//!   chunk is sent, then the phase is `Replying`) or is cut (dropped without `end`, a failed or late send, a peer that
//!   left): the connection then closes, with no last chunk.
//! - `Replying`: the reply is being sent; the `recv` of `Handling`, if still in flight, was cancelled.
//! - `Closing`: the reply that ended the connection is out and the write side is shut down; what the peer still
//!   sends is read and dropped (so that closing does not reset the connection and lose the reply) until it closes,
//!   the budget of bytes is spent or the linger deadline passes (one deadline for the whole linger, from its start:
//!   `linger_limit`, or `stop_linger_limit` if it is shorter and the server is stopping).
//! - `Closed`: the socket is dropped as soon as no operation is in flight on it (a cancelled operation ends
//!   first). The server is told the connection ended once the socket is dropped *and* the handler's `Responder`
//!   for the request being handled, if any, has been used or dropped and its reply has been through the
//!   connection's order (a discarded one); so a stop is complete only when no frame of the server is left.

use std::error::Error as StdError;
use std::fmt::{self, Debug, Display, Formatter};
use std::io::{Error, ErrorKind};
use std::mem::take;
use std::net::{Shutdown, TcpStream};
#[cfg(unix)]
use std::os::fd::AsRawFd;
#[cfg(windows)]
use std::os::windows::io::AsRawSocket;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use euhedral_core::frames::Frame;
use euhedral_core::generics::{LatticeReceiver, LatticeSource, SourceCell};
use euhedral_core::ingest::QueueIngestSink;
use euhedral_io::{Cancelled, Continuation, IoBuf, IoDone, LimitPassed, OpId, OpOptions, RawSock};

use crate::message::{
    ConnectionMode, Method, ParseError, Request, Response, Version, WriteError, status_has_no_body,
};
use crate::parse::{Progress, RequestParser};
use crate::server::{Ctx, ListenerEvent, Shared};
use crate::station::{Drive, Post, Station};
use crate::write::{
    write_chunk, write_chunked_head, write_continue, write_last_chunk, write_response,
    write_response_head,
};

/// The handle of a connected stream, as `AsyncIo` takes it.
#[cfg(unix)]
pub(crate) fn raw_stream(stream: &TcpStream) -> RawSock {
    stream.as_raw_fd()
}

/// The handle of a connected stream, as `AsyncIo` takes it.
#[cfg(windows)]
pub(crate) fn raw_stream(stream: &TcpStream) -> RawSock {
    stream.as_raw_socket()
}

/// What a connection is doing.
#[derive(Debug)]
enum Phase {
    Reading,
    Continuing,
    Handling {
        seq: u64,
        head: bool,
        keep: bool,
        version: Version,
        cancel: Arc<AtomicBool>,
    },
    /// A streamed reply is going out: `framing` says how its body is delimited, `keep` whether the connection may
    /// stay open after it.
    Streaming {
        seq: u64,
        keep: bool,
        framing: Framing,
        cancel: Arc<AtomicBool>,
    },
    Replying {
        keep: bool,
    },
    Closing,
    Closed,
}

/// How the body of a streamed reply is delimited.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Framing {
    /// HTTP/1.1: the chunked coding, ended by the last chunk.
    Chunked,
    /// HTTP/1.0: no length; the body ends when the connection does.
    Close,
    /// The answer to HEAD: the head only; pieces are dropped.
    None,
}

type HeadThen = Continuation<Result<ReplyStream, StreamError>>;
type PieceThen = Continuation<Result<PieceSent, StreamError>>;

/// The continuation waiting for the send in flight to end.
enum Awaited {
    Head(HeadThen),
    Piece(PieceThen, Vec<u8>),
}

impl Awaited {
    /// Runs the continuation with `error`.
    fn fail(self, sink: &QueueIngestSink, error: StreamError) {
        match self {
            Awaited::Head(then) => run_then(sink, then, Err(error)),
            Awaited::Piece(then, _) => run_then(sink, then, Err(error)),
        }
    }
}

/// An [`Awaited`] that is never lost: dropped without having been taken, it runs with `StreamError::Closed`.
struct Waiting {
    sink: Arc<QueueIngestSink>,
    awaited: Option<Awaited>,
}

impl Waiting {
    fn new(sink: Arc<QueueIngestSink>, awaited: Awaited) -> Waiting {
        Waiting {
            sink,
            awaited: Some(awaited),
        }
    }

    /// Takes the continuation out; the guard then has nothing left to deliver.
    fn take(mut self) -> (Arc<QueueIngestSink>, Awaited) {
        let awaited = self
            .awaited
            .take()
            .expect("a waiting continuation is taken once");
        (self.sink.clone(), awaited)
    }
}

impl Drop for Waiting {
    fn drop(&mut self) {
        if let Some(awaited) = self.awaited.take() {
            awaited.fail(&self.sink, StreamError::Closed);
        }
    }
}

/// Runs `then` with `value` and offers the frame it returns to `sink`. A `then` that panics is dropped with what it
/// held (a `ReplyStream` posts its abort); the panic ends here.
fn run_then<T>(sink: &QueueIngestSink, then: Continuation<T>, value: T) {
    if let Ok(frame) = catch_unwind(AssertUnwindSafe(|| then(value))) {
        let _ = sink.offer(frame);
    }
}

/// What reaches a connection's state.
pub(crate) enum Event {
    /// The connection was accepted: start reading.
    Start,
    /// A `recv` ended.
    Recv(IoDone),
    /// A `send` ended.
    Sent(IoDone),
    /// The handler's answer to request `seq`.
    Reply { seq: u64, response: Response },
    /// The server is stopping.
    Stop,
    /// The handler asked to be told if request `seq` is cancelled.
    OnCancel { seq: u64, frame: Box<dyn Frame> },
    /// The handler answers request `seq` with a streamed reply: `head` (and its body, if any, as the first piece).
    StreamHead {
        seq: u64,
        head: Response,
        then: HeadThen,
    },
    /// The next piece of the stream of request `seq`.
    Piece {
        seq: u64,
        bytes: Vec<u8>,
        then: PieceThen,
    },
    /// The stream of request `seq` ends: cleanly (`end`), or because its `ReplyStream` was dropped.
    StreamEnd { seq: u64, clean: bool },
}

/// The plain state of a connection.
struct ConnState {
    /// The socket, until the connection is finished.
    stream: Option<TcpStream>,
    /// What the caller owns (`AsyncIo`, sink, handler), until the server is told that the connection ended.
    ctx: Option<Arc<Ctx>>,
    /// The frame to offer if the request being handled is cancelled.
    on_cancel: Option<Box<dyn Frame>>,
    /// The number of the request whose cancel flag was set, whatever the phase became after.
    cancelled_seq: Option<u64>,
    parser: RequestParser,
    /// Bytes received and not yet parsed.
    buf: Vec<u8>,
    /// The buffer of the `recv`, while none is in flight.
    spare: Option<Vec<u8>>,
    phase: Phase,
    /// The `recv` in flight.
    recv: Option<OpId>,
    /// The `send` in flight.
    send: Option<OpId>,
    /// When the first byte of the request being read was seen.
    started: Option<Instant>,
    /// The number of the request being handled, or the last one.
    seq: u64,
    stopping: bool,
    /// The peer broke the stream (or an operation of it hit its limit) while a reply was going out.
    peer_gone: bool,
    /// The peer ended its stream (a half-close) while the request was handled: its reply is still sent, and the
    /// connection ends after it.
    half_closed: bool,
    /// A `Responder` of this connection has not yet been through the connection's order, or a `ReplyStream` of it
    /// is held or a piece of it is in flight.
    owed: bool,
    /// The continuation to run when the send of a stream's head or piece ends.
    waiting: Option<Waiting>,
    /// The send buffer of a stream, kept between its pieces and dropped when the stream ends.
    out_spare: Option<Vec<u8>>,
    /// The last chunk of a stream is in flight.
    finishing: bool,
    /// Bytes the closing phase may still read and drop.
    linger_budget: usize,
    /// When the closing phase ends whatever the peer does.
    linger_deadline: Option<Instant>,
    /// Where the connection sits in the server's registry.
    slot: usize,
    /// The server has been told that this connection ended.
    reported: bool,
}

/// A connection: the immutable part, the state that only its key frame enters, and its station.
pub(crate) struct Conn {
    shared: Arc<Shared>,
    /// The id hash of every frame of this connection.
    state: SourceCell<ConnState>,
    station: Station<Conn>,
}

impl Debug for Conn {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("Conn").finish_non_exhaustive()
    }
}

/// What a `recv` came to.
enum Got {
    Data(usize),
    Eof,
    Limit,
    Cancelled,
    Failed,
}

impl Conn {
    pub(crate) fn new(
        shared: Arc<Shared>,
        ctx: Arc<Ctx>,
        stream: TcpStream,
        slot: usize,
    ) -> Arc<Conn> {
        let parser = RequestParser::new(shared.limits);
        let linger_budget = shared
            .config
            .max_header_bytes
            .saturating_add(shared.config.max_body_bytes);
        Arc::new_cyclic(|me| Conn {
            shared,
            station: Station::new(me.clone()),
            state: SourceCell::new(ConnState {
                stream: Some(stream),
                ctx: Some(ctx),
                on_cancel: None,
                cancelled_seq: None,
                parser,
                buf: Vec::new(),
                spare: None,
                phase: Phase::Reading,
                recv: None,
                send: None,
                started: None,
                seq: 0,
                stopping: false,
                peer_gone: false,
                half_closed: false,
                owed: false,
                waiting: None,
                out_spare: None,
                finishing: false,
                linger_budget,
                linger_deadline: None,
                slot,
                reported: false,
            }),
        })
    }

    /// The server is stopping: this connection was told, or the stop was asked for and the news is on its way (it
    /// is one atomic, read by the frames that decide whether a connection stays open).
    fn is_stopping(&self, st: &ConnState) -> bool {
        st.stopping || self.shared.stop_requested.load(Ordering::Acquire)
    }

    /// Pushes `event` into this connection's inbox; its key frame runs it.
    pub(crate) fn post(&self, event: Event) {
        self.station.post(event);
    }

    /// Runs one event: the only way into the state.
    fn run_event(self: &Arc<Self>, event: Event) {
        // Only the key frame runs events, one at a time (see the module notes): the entry cannot find the cell
        // taken.
        let mut guard = self.state.enter();
        let st: &mut ConnState = &mut guard;
        match event {
            Event::Start => {
                if self.is_stopping(st) {
                    self.close_now(st);
                } else {
                    self.begin_request(st);
                }
            }
            Event::Recv(done) => self.on_recv(st, done),
            Event::Sent(done) => self.on_sent(st, done),
            Event::Reply { seq, response } => self.on_reply(st, seq, response),
            Event::Stop => self.on_stop(st),
            Event::OnCancel { seq, frame } => self.on_cancel_asked(st, seq, frame),
            Event::StreamHead { seq, head, then } => self.on_stream_head(st, seq, head, then),
            Event::Piece { seq, bytes, then } => self.on_piece(st, seq, bytes, then),
            Event::StreamEnd { seq, clean } => self.on_stream_end(st, seq, clean),
        }
    }

    // ---- operations ----------------------------------------------------------------------------------------

    /// Cancels an operation of this connection.
    fn cancel_op(&self, st: &ConnState, id: OpId) {
        if let Some(io) = st.ctx.as_ref().and_then(|ctx| ctx.io.upgrade()) {
            io.cancel(id);
        }
    }

    /// Starts the one `recv` of the connection, bounded by `limit`.
    fn issue_recv(self: &Arc<Self>, st: &mut ConnState, limit: Option<Duration>) {
        debug_assert!(st.recv.is_none());
        let (Some(stream), Some(ctx)) = (&st.stream, &st.ctx) else {
            return;
        };
        let bytes = st
            .spare
            .take()
            .unwrap_or_else(|| vec![0; self.shared.config.read_chunk.max(1)]);
        let conn = self.clone();
        let started = match ctx.io.upgrade() {
            Some(io) => io.recv_with(
                raw_stream(stream),
                IoBuf { bytes, filled: 0 },
                OpOptions { limit },
                Box::new(move |done| Post::new(conn, Event::Recv(done))),
            ),
            None => Err(Error::from(ErrorKind::NotConnected)),
        };
        match started {
            Ok(id) => st.recv = Some(id),
            Err(_) => self.close_now(st),
        }
    }

    /// The limit of the next `recv` of a request being read: the idle limit before its first byte, then what is
    /// left of the read limit, which covers the whole request.
    fn reading_limit(&self, st: &ConnState) -> Option<Duration> {
        match st.started {
            None => self.shared.config.idle_limit,
            Some(first) => self
                .shared
                .config
                .read_limit
                .map(|limit| limit.saturating_sub(first.elapsed())),
        }
    }

    /// Starts the one `send` of the connection, bounded by `write_limit`. The engine moves every byte or ends with
    /// an error.
    fn issue_send(self: &Arc<Self>, st: &mut ConnState, bytes: Vec<u8>) {
        let limit = self.shared.config.write_limit;
        self.issue_send_limited(st, bytes, limit);
    }

    /// Starts the one `send` of the connection, bounded by `limit`.
    fn issue_send_limited(
        self: &Arc<Self>,
        st: &mut ConnState,
        bytes: Vec<u8>,
        limit: Option<Duration>,
    ) {
        debug_assert!(st.send.is_none());
        let (Some(stream), Some(ctx)) = (&st.stream, &st.ctx) else {
            return;
        };
        let filled = bytes.len();
        let conn = self.clone();
        let started = match ctx.io.upgrade() {
            Some(io) => io.send_with(
                raw_stream(stream),
                IoBuf { bytes, filled },
                OpOptions { limit },
                Box::new(move |done| Post::new(conn, Event::Sent(done))),
            ),
            None => Err(Error::from(ErrorKind::NotConnected)),
        };
        match started {
            Ok(id) => st.send = Some(id),
            Err(_) => self.close_now(st),
        }
    }

    // ---- reading -------------------------------------------------------------------------------------------

    /// Starts on the next request: from the bytes already held, or by waiting for the first one.
    fn begin_request(self: &Arc<Self>, st: &mut ConnState) {
        if self.is_stopping(st) {
            self.close_now(st);
            return;
        }
        st.phase = Phase::Reading;
        st.started = None;
        if st.buf.is_empty() {
            self.issue_recv(st, self.shared.config.idle_limit);
        } else {
            st.started = Some(Instant::now());
            self.parse_loop(st);
        }
    }

    /// Feeds the parser what is held, until a request is complete, an error ends the connection or more bytes
    /// are needed.
    fn parse_loop(self: &Arc<Self>, st: &mut ConnState) {
        loop {
            let fed = st.parser.feed(&st.buf);
            let (used, progress) = match fed {
                Ok(fed) => fed,
                Err(error) => {
                    self.parse_failed(st, error);
                    return;
                }
            };
            st.buf.drain(..used);
            match progress {
                Progress::NeedMore => {
                    let limit = self.reading_limit(st);
                    self.issue_recv(st, limit);
                    return;
                }
                Progress::HeadComplete {
                    expect_continue, ..
                } => {
                    if expect_continue {
                        if self.is_stopping(st) {
                            self.close_now(st);
                            return;
                        }
                        // The handler is not asked: the head passed the limits (a body over `max_body_bytes`
                        // was refused with 413 before this), so the body is invited.
                        st.phase = Phase::Continuing;
                        let mut out = Vec::new();
                        write_continue(&mut out);
                        self.issue_send(st, out);
                        return;
                    }
                }
                Progress::Complete(request) => {
                    // A request that is complete when the stop is known is not handed to the handler: the
                    // connection ends, as an idle one does.
                    if self.is_stopping(st) {
                        self.close_now(st);
                    } else {
                        self.dispatch(st, request);
                    }
                    return;
                }
            }
        }
    }

    /// A parse error: the status of the error, then the end of the connection.
    fn parse_failed(self: &Arc<Self>, st: &mut ConnState, error: ParseError) {
        self.shared
            .counters
            .rare
            .parse_errors
            .fetch_add(1, Ordering::Relaxed);
        self.final_status(st, error.status().unwrap_or(400));
    }

    /// Answers `status` with `Connection: close` and ends the connection after it.
    fn final_status(self: &Arc<Self>, st: &mut ConnState, status: u16) {
        let mut out = Vec::new();
        let response = Response::with_status(status);
        // A status line and a close: the writer cannot refuse it.
        let _ = write_response(&response, ConnectionMode::Close, &mut out);
        self.shared.count_reply(status);
        st.phase = Phase::Replying { keep: false };
        self.issue_send(st, out);
    }

    fn on_recv(self: &Arc<Self>, st: &mut ConnState, done: IoDone) {
        st.recv = None;
        let IoDone { buf, result } = done;
        let got = match result {
            Ok(0) => Got::Eof,
            Ok(n) => Got::Data(n),
            Err(e) if LimitPassed::is(&e) => Got::Limit,
            Err(e) if Cancelled::is(&e) => Got::Cancelled,
            Err(_) => Got::Failed,
        };
        if let Got::Data(n) = got {
            // Kept whatever the phase: the next request, or the bytes of a lingering peer.
            if !matches!(st.phase, Phase::Closing | Phase::Closed) {
                st.buf.extend_from_slice(&buf.bytes[..n]);
            }
        }
        st.spare = Some(buf.bytes);
        // The end of a lingering close is the server's own deadline, not a limit a peer passed.
        if matches!(got, Got::Limit) && !matches!(st.phase, Phase::Closing | Phase::Closed) {
            self.shared
                .counters
                .rare
                .limits_passed
                .fetch_add(1, Ordering::Relaxed);
        }
        match st.phase {
            Phase::Closed => self.finalize(st),
            Phase::Closing => {
                if let Got::Data(n) = got {
                    st.linger_budget = st.linger_budget.saturating_sub(n);
                }
                // Data and a cancel (the stop shortening the deadline) go on, within the one deadline and the
                // budget; anything else ends the linger.
                let goes_on = matches!(got, Got::Data(_) | Got::Cancelled) && st.linger_budget > 0;
                let left = st.linger_deadline.map_or(Duration::ZERO, |d| {
                    d.saturating_duration_since(Instant::now())
                });
                if goes_on && !left.is_zero() {
                    self.issue_recv(st, Some(left));
                } else {
                    self.close_now(st);
                }
            }
            Phase::Reading => match got {
                Got::Data(_) => {
                    st.started.get_or_insert_with(Instant::now);
                    self.parse_loop(st);
                }
                Got::Limit if st.started.is_some() => self.final_status(st, 408),
                _ => self.close_now(st),
            },
            Phase::Handling { .. } | Phase::Streaming { .. } => match got {
                // Kept for the next request; no further `recv` until the reply is out.
                Got::Data(_) | Got::Cancelled => {}
                // A half-close: the peer may still read. The flag is a hint; the reply is sent.
                Got::Eof => self.peer_half_closed(st),
                Got::Limit | Got::Failed => self.peer_left(st),
            },
            Phase::Replying { .. } => {
                match got {
                    Got::Eof => st.half_closed = true,
                    Got::Limit | Got::Failed => st.peer_gone = true,
                    Got::Data(_) | Got::Cancelled => {}
                }
                self.settle(st);
            }
            Phase::Continuing => {}
        }
    }

    /// The peer ended its stream while the handler has its request: the cancel flag is set as a hint that the
    /// work may stop, but the peer may still be reading, so the reply is sent when it comes, and the connection
    /// ends after it.
    fn peer_half_closed(self: &Arc<Self>, st: &mut ConnState) {
        st.half_closed = true;
        self.raise_cancel(st);
    }

    fn raise_cancel(&self, st: &mut ConnState) {
        if let Phase::Handling { cancel, seq, .. } | Phase::Streaming { cancel, seq, .. } =
            &st.phase
        {
            st.cancelled_seq = Some(*seq);
            cancel.store(true, Ordering::Release);
            self.shared
                .counters
                .rare
                .cancelled
                .fetch_add(1, Ordering::Relaxed);
        } else {
            return;
        }
        // Told once: the frame the handler asked to be woken by.
        if let (Some(frame), Some(ctx)) = (st.on_cancel.take(), &st.ctx) {
            let _ = ctx.sink.offer(frame);
        }
    }

    /// The handler asked for a frame to be offered when the request is cancelled. If the request was already
    /// cancelled (and its reply is still owed, whatever the connection has done since) it is offered at once; while
    /// the request is being handled it waits, a later one replacing it; otherwise (the reply went through) it is
    /// dropped.
    fn on_cancel_asked(self: &Arc<Self>, st: &mut ConnState, seq: u64, frame: Box<dyn Frame>) {
        if st.owed && seq == st.seq && st.cancelled_seq == Some(seq) {
            if let Some(ctx) = &st.ctx {
                let _ = ctx.sink.offer(frame);
            }
            return;
        }
        if let Phase::Handling { seq: current, .. } | Phase::Streaming { seq: current, .. } =
            &st.phase
            && *current == seq
        {
            st.on_cancel = Some(frame);
        }
    }

    /// The peer broke the connection while the handler has its request: its cancel flag is set, and the
    /// connection ends; the handler's reply, when it comes, is discarded.
    fn peer_left(self: &Arc<Self>, st: &mut ConnState) {
        self.raise_cancel(st);
        self.close_now(st);
    }

    // ---- handling and replying -----------------------------------------------------------------------------

    fn dispatch(self: &Arc<Self>, st: &mut ConnState, request: Request) {
        st.seq += 1;
        st.started = None;
        let seq = st.seq;
        let cancel = Arc::new(AtomicBool::new(false));
        st.phase = Phase::Handling {
            seq,
            head: request.method == Method::Head,
            keep: request.keep_alive() && !self.is_stopping(st),
            version: request.version,
            cancel: cancel.clone(),
        };
        self.shared
            .counters
            .request
            .requests
            .fetch_add(1, Ordering::Relaxed);
        // The `recv` that sees the peer go. Bytes of a pipelined request already held mean there is something
        // to answer after this one; the peer's departure is then seen when the reply is sent.
        if st.buf.is_empty() {
            self.issue_recv(st, None);
            if matches!(st.phase, Phase::Closed) {
                return;
            }
        }
        st.owed = true;
        let responder = Responder {
            conn: self.clone(),
            seq,
            cancel,
            sent: false,
        };
        // A handler that panics drops its `Responder` while it unwinds, which answers `500`; the panic ends there
        // and does not take the connection's frame (or its worker) with it.
        let Some(ctx) = st.ctx.clone() else {
            return;
        };
        let Ok(frame) = catch_unwind(AssertUnwindSafe(|| ctx.handler.handle(request, responder)))
        else {
            return;
        };
        // A sink that has completed refuses: the frame (and the `Responder` in it, which answers 500 to nobody)
        // is dropped.
        let _ = ctx.sink.offer(frame);
    }

    fn on_reply(self: &Arc<Self>, st: &mut ConnState, seq: u64, response: Response) {
        // The one `Responder` of this connection has been through its order, whatever becomes of its reply, and
        // the frame that waited for a cancel will not run.
        st.on_cancel = None;
        if seq == st.seq {
            st.owed = false;
        }
        let Phase::Handling {
            seq: current,
            head,
            keep,
            version,
            ..
        } = &st.phase
        else {
            self.discard(st);
            return;
        };
        if *current != seq {
            self.discard(st);
            return;
        }
        let (head, version) = (*head, *version);
        let keep = *keep && !self.is_stopping(st) && !st.peer_gone && !st.half_closed;
        let mode = match (keep, version) {
            (false, _) => ConnectionMode::Close,
            (true, Version::Http10) => ConnectionMode::KeepAlive,
            (true, Version::Http11) => ConnectionMode::Unspecified,
        };
        let mut out = Vec::new();
        let mut status = response.status;
        if write_final(response, head, mode, &mut out).is_err() {
            out.clear();
            status = 500;
            let _ = write_response(&Response::with_status(500), mode, &mut out);
        }
        self.shared.count_reply(status);
        st.phase = Phase::Replying { keep };
        // The `recv` that watched the peer is no longer wanted: it ends cancelled, or with what it found.
        if let Some(id) = st.recv {
            self.cancel_op(st, id);
        }
        self.issue_send(st, out);
    }

    /// A reply nobody can be sent: counted, and the end of the connection may now be reported.
    fn discard(self: &Arc<Self>, st: &mut ConnState) {
        self.shared
            .counters
            .rare
            .discarded
            .fetch_add(1, Ordering::Relaxed);
        self.finalize(st);
    }

    fn on_sent(self: &Arc<Self>, st: &mut ConnState, done: IoDone) {
        st.send = None;
        let IoDone { buf, result } = done;
        if matches!(st.phase, Phase::Streaming { .. }) {
            st.out_spare = Some(buf.bytes);
        }
        let finishing = take(&mut st.finishing);
        if let Err(error) = result {
            if LimitPassed::is(&error) {
                self.shared
                    .counters
                    .rare
                    .limits_passed
                    .fetch_add(1, Ordering::Relaxed);
            }
            if let Some(waiting) = st.waiting.take() {
                let (sink, awaited) = waiting.take();
                awaited.fail(&sink, stream_error(error));
                st.owed = false;
            } else if finishing {
                self.count_cut();
            }
            // A stream a send ended early on is at a position only the count tells: it is closed.
            self.close_now(st);
            return;
        }
        if let Some(waiting) = st.waiting.take() {
            self.stream_sent(st, waiting);
            return;
        }
        match st.phase {
            Phase::Continuing => {
                if self.is_stopping(st) {
                    self.close_now(st);
                } else {
                    st.phase = Phase::Reading;
                    self.parse_loop(st);
                }
            }
            Phase::Replying { .. } => self.settle(st),
            Phase::Closed => self.finalize(st),
            _ => {}
        }
    }

    // ---- streaming -----------------------------------------------------------------------------------------

    /// A stream is cut: counted once.
    fn count_cut(&self) {
        self.shared
            .counters
            .rare
            .streams_cut
            .fetch_add(1, Ordering::Relaxed);
    }

    /// The right to send the rest of the stream of the phase the connection is in, if it is streaming.
    fn new_stream(self: &Arc<Self>, st: &ConnState) -> Option<ReplyStream> {
        match &st.phase {
            Phase::Streaming { seq, cancel, .. } => Some(ReplyStream {
                conn: self.clone(),
                seq: *seq,
                cancel: cancel.clone(),
                live: true,
            }),
            _ => None,
        }
    }

    /// The send of a stream's head or piece is out: the stream goes back to the handler, or, if the connection was
    /// closed meanwhile, the handler is told so.
    fn stream_sent(self: &Arc<Self>, st: &mut ConnState, waiting: Waiting) {
        let (sink, awaited) = waiting.take();
        let Some(stream) = self.new_stream(st) else {
            awaited.fail(&sink, StreamError::Closed);
            st.owed = false;
            self.finalize(st);
            return;
        };
        match awaited {
            Awaited::Head(then) => run_then(&sink, then, Ok(stream)),
            Awaited::Piece(then, mut piece) => {
                piece.clear();
                run_then(&sink, then, Ok(PieceSent { stream, piece }));
            }
        }
    }

    /// The handler answers with a streamed reply: the head (and its body, as the first piece) is written, and the
    /// handler gets the stream when it is out.
    fn on_stream_head(
        self: &Arc<Self>,
        st: &mut ConnState,
        seq: u64,
        mut head: Response,
        then: HeadThen,
    ) {
        let Some(sink) = st.ctx.as_ref().map(|ctx| ctx.sink.clone()) else {
            return;
        };
        let stream_phase = match &st.phase {
            Phase::Handling {
                seq: current,
                head,
                keep,
                version,
                cancel,
            } if *current == seq => Some((*head, *keep, *version, cancel.clone())),
            _ => None,
        };
        let Some((is_head, keep, version, cancel)) = stream_phase else {
            st.on_cancel = None;
            if seq == st.seq {
                st.owed = false;
            }
            run_then(&sink, then, Err(StreamError::Closed));
            self.discard(st);
            return;
        };
        let framing = match (is_head, version) {
            (true, _) => Framing::None,
            (false, Version::Http11) => Framing::Chunked,
            (false, Version::Http10) => Framing::Close,
        };
        // A body that ends with the connection leaves nothing to keep.
        let keep = keep
            && framing != Framing::Close
            && !self.is_stopping(st)
            && !st.peer_gone
            && !st.half_closed;
        let mode = match (keep, version) {
            (false, _) => ConnectionMode::Close,
            (true, Version::Http10) => ConnectionMode::KeepAlive,
            (true, Version::Http11) => ConnectionMode::Unspecified,
        };
        let status = head.status;
        let first = take(&mut head.body);
        let mut out = st.out_spare.take().unwrap_or_default();
        out.clear();
        let written = if status_has_no_body(status) {
            Err(WriteError::BodyNotAllowed)
        } else if framing == Framing::Chunked {
            write_chunked_head(&head, mode, &mut out)
        } else {
            write_response_head(&head, None, mode, &mut out)
        };
        if let Err(error) = written {
            // The head cannot be written: the request is answered 500, and the handler is told why.
            run_then(&sink, then, Err(StreamError::Head(error)));
            self.on_reply(st, seq, Response::with_status(500));
            return;
        }
        match framing {
            Framing::Chunked => write_chunk(&first, &mut out),
            Framing::Close => out.extend_from_slice(&first),
            Framing::None => {}
        }
        self.shared.count_reply(status);
        self.shared
            .counters
            .request
            .streams
            .fetch_add(1, Ordering::Relaxed);
        st.phase = Phase::Streaming {
            seq,
            keep,
            framing,
            cancel,
        };
        // The watch `recv` stays: the stream is answered to a peer that may leave.
        st.waiting = Some(Waiting::new(sink, Awaited::Head(then)));
        let limit = self.shared.config.piece_limit;
        self.issue_send_limited(st, out, limit);
    }

    /// The handler sends the next piece of its stream.
    fn on_piece(self: &Arc<Self>, st: &mut ConnState, seq: u64, bytes: Vec<u8>, then: PieceThen) {
        let Some(sink) = st.ctx.as_ref().map(|ctx| ctx.sink.clone()) else {
            return;
        };
        let framing = match &st.phase {
            Phase::Streaming {
                seq: current,
                framing,
                ..
            } if *current == seq && st.waiting.is_none() && st.send.is_none() => Some(*framing),
            _ => None,
        };
        let Some(framing) = framing else {
            // The connection ended (or the stream was cut) while the handler held the stream.
            run_then(&sink, then, Err(StreamError::Closed));
            if seq == st.seq {
                st.owed = false;
            }
            self.finalize(st);
            return;
        };
        if bytes.is_empty() || framing == Framing::None {
            self.piece_done(st, &sink, then, bytes);
            return;
        }
        let mut out = st.out_spare.take().unwrap_or_default();
        out.clear();
        match framing {
            Framing::Chunked => write_chunk(&bytes, &mut out),
            Framing::Close | Framing::None => out.extend_from_slice(&bytes),
        }
        st.waiting = Some(Waiting::new(sink, Awaited::Piece(then, bytes)));
        let limit = self.shared.config.piece_limit;
        self.issue_send_limited(st, out, limit);
    }

    /// A piece that needs no write is done: the stream goes back with the emptied buffer.
    fn piece_done(
        self: &Arc<Self>,
        st: &mut ConnState,
        sink: &QueueIngestSink,
        then: PieceThen,
        mut piece: Vec<u8>,
    ) {
        piece.clear();
        if let Some(stream) = self.new_stream(st) {
            run_then(sink, then, Ok(PieceSent { stream, piece }));
        }
    }

    /// The handler ends its stream, or drops it.
    fn on_stream_end(self: &Arc<Self>, st: &mut ConnState, seq: u64, clean: bool) {
        let framing = match &st.phase {
            Phase::Streaming {
                seq: current,
                framing,
                ..
            } if *current == seq => Some(*framing),
            _ => None,
        };
        if seq == st.seq {
            st.owed = false;
        }
        let Some(framing) = framing else {
            // Already closed: what is left is to report the end of the connection.
            self.finalize(st);
            return;
        };
        if !clean {
            // A stream that is dropped is cut: the body stays unfinished, and the connection closes.
            st.on_cancel = None;
            self.close_now(st);
            return;
        }
        let Phase::Streaming { keep, .. } = st.phase else {
            return;
        };
        st.on_cancel = None;
        let keep = keep && !self.is_stopping(st) && !st.half_closed && !st.peer_gone;
        st.phase = Phase::Replying { keep };
        // The `recv` that watched the peer is no longer wanted.
        if let Some(id) = st.recv {
            self.cancel_op(st, id);
        }
        if framing == Framing::Chunked {
            let mut out = st.out_spare.take().unwrap_or_default();
            out.clear();
            write_last_chunk(&mut out);
            st.finishing = true;
            let limit = self.shared.config.piece_limit;
            self.issue_send_limited(st, out, limit);
        } else {
            st.out_spare = None;
            self.settle(st);
        }
    }

    /// After a reply: the next request, or the end of the connection, once nothing is in flight.
    fn settle(self: &Arc<Self>, st: &mut ConnState) {
        let Phase::Replying { keep } = st.phase else {
            return;
        };
        if st.send.is_some() || st.recv.is_some() {
            return;
        }
        if st.peer_gone {
            self.close_now(st);
        } else if keep && !self.is_stopping(st) && !st.half_closed {
            self.begin_request(st);
        } else {
            self.linger(st);
        }
    }

    /// The last reply is out: shut the write side, so the peer reads to its end, and drop what it still sends.
    fn linger(self: &Arc<Self>, st: &mut ConnState) {
        if let Some(stream) = &st.stream {
            let _ = stream.shutdown(Shutdown::Write);
        }
        st.phase = Phase::Closing;
        let config = &self.shared.config;
        let bound = if self.is_stopping(st) {
            config.linger_limit.min(config.stop_linger_limit)
        } else {
            config.linger_limit
        };
        let deadline = Instant::now() + bound;
        st.linger_deadline = Some(deadline);
        self.issue_recv(st, Some(bound));
    }

    /// The server is stopping: what waits for a request ends, a request being handled is answered and ends, and a
    /// linger is cut to the stop's bound.
    fn on_stop(self: &Arc<Self>, st: &mut ConnState) {
        st.stopping = true;
        match st.phase {
            Phase::Reading | Phase::Continuing => self.close_now(st),
            Phase::Closing => {
                let cut = Instant::now() + self.shared.config.stop_linger_limit;
                st.linger_deadline = Some(st.linger_deadline.map_or(cut, |d| d.min(cut)));
                // The pending `recv` carries the old deadline: it ends cancelled and is issued again with the
                // new one.
                if let Some(id) = st.recv {
                    self.cancel_op(st, id);
                }
            }
            Phase::Handling { .. }
            | Phase::Streaming { .. }
            | Phase::Replying { .. }
            | Phase::Closed => {}
        }
    }

    // ---- ending --------------------------------------------------------------------------------------------

    /// Ends the connection: cancels the `recv` in flight, and drops the socket once nothing is in flight on it.
    fn close_now(self: &Arc<Self>, st: &mut ConnState) {
        // A stream is cut when it is closed while going, or while its last chunk is not yet out.
        let finishing = take(&mut st.finishing);
        if finishing || matches!(st.phase, Phase::Streaming { .. }) {
            self.count_cut();
        }
        st.phase = Phase::Closed;
        if let Some(id) = st.recv {
            self.cancel_op(st, id);
        }
        if let Some(id) = st.send {
            self.cancel_op(st, id);
        } else if let Some(waiting) = st.waiting.take() {
            // The send that would have delivered it never started.
            let (sink, awaited) = waiting.take();
            awaited.fail(&sink, StreamError::Closed);
            st.owed = false;
        }
        self.finalize(st);
    }

    fn finalize(self: &Arc<Self>, st: &mut ConnState) {
        if !matches!(st.phase, Phase::Closed) || st.recv.is_some() || st.send.is_some() {
            return;
        }
        if let (Some(stream), Some(io)) =
            (&st.stream, st.ctx.as_ref().and_then(|ctx| ctx.io.upgrade()))
        {
            io.forget_socket(raw_stream(stream));
        }
        st.stream = None;
        st.on_cancel = None;
        st.buf = Vec::new();
        st.spare = None;
        st.out_spare = None;
        // Not told to the server while a `Responder` is still to come through this order: the stop of the
        // server is complete only when no frame of it is left to run.
        if st.owed || st.reported {
            return;
        }
        st.reported = true;
        // Everything of the caller's is let go of before the server is told: when the listener reports a stop, no
        // connection holds it.
        st.ctx = None;
        // The source ends with the connection: the lattice lets go of it. This runs inside the key, whose last act
        // is to go home; nothing of the connection is touched after the listener hears of it but that.
        self.station.complete();
        self.shared.post_listener(ListenerEvent::Closed(st.slot));
    }
}

impl Drive for Conn {
    type Event = Event;

    fn station(&self) -> &Station<Conn> {
        &self.station
    }

    fn run(self: &Arc<Self>, event: Event) {
        self.run_event(event);
    }
}

impl LatticeSource for Conn {
    fn add_downstream(self: Arc<Self>, downstream: Arc<dyn LatticeReceiver>) {
        let me: Arc<dyn LatticeSource> = self.clone();
        self.station.add_downstream(me, downstream);
    }

    /// Hands out nothing: the connection's work is its key frame, which only `request` pushes.
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

/// Writes a final response (a HEAD gets its head with the length of the body it would have had).
fn write_final(
    mut response: Response,
    head: bool,
    mode: ConnectionMode,
    out: &mut Vec<u8>,
) -> Result<(), WriteError> {
    if response.status < 200 {
        return Err(WriteError::BodyNotAllowed);
    }
    if head {
        let length = response.body.len() as u64;
        response.body = Vec::new();
        let length = (response.status != 204).then_some(length);
        write_response_head(&response, length, mode, out)
    } else {
        write_response(&response, mode, out)
    }
}

/// The right to answer one request: exactly one reply is sent for it.
///
/// [`send`](Self::send) consumes it. A `Responder` dropped without `send` (a handler that failed, a frame that was
/// dropped) answers `500`. The reply is delivered through the connection's order, so `send` may be called from any
/// frame on any worker, and never waits. A reply for a request whose peer has gone, or whose connection ended, is
/// discarded without a word to the handler.
pub struct Responder {
    conn: Arc<Conn>,
    seq: u64,
    cancel: Arc<AtomicBool>,
    sent: bool,
}

impl Debug for Responder {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("Responder")
            .field("seq", &self.seq)
            .field("cancelled", &self.cancel.load(Ordering::Relaxed))
            .finish_non_exhaustive()
    }
}

impl Responder {
    /// Sends `response` as the answer to this request. A response that cannot be written (a header with a line
    /// break, a body on a `204`, a status below 200) is replaced by a `500`. The `Connection` line and the
    /// decision to keep the connection are the server's.
    pub fn send(mut self, response: Response) {
        self.sent = true;
        self.conn.post(Event::Reply {
            seq: self.seq,
            response,
        });
    }

    /// Answers with a reply whose body is sent piece by piece, and returns at once. `head` carries the status and the
    /// headers, and its body, if it has one, goes out as the first piece in the same write as the head; the writer
    /// owns the framing (`Transfer-Encoding: chunked` for an HTTP/1.1 peer; for HTTP/1.0 no length, `Connection:
    /// close` and the end of the body is the close; for HEAD the head alone, with no `Content-Length` and no
    /// `Transfer-Encoding`, and the pieces are dropped), so `head` must not set `Content-Length`,
    /// `Transfer-Encoding` or `Connection`.
    ///
    /// `then` runs once the head is written, with the [`ReplyStream`] to send the rest through, or with the reason
    /// the stream did not start: a head the writer refuses (and a 1xx, 204 or 304, which have no body) is answered
    /// `500` and `then` gets `Err(StreamError::Head(_))`; a connection that ended gets `Err(StreamError::Closed)`;
    /// the write that passed [`piece_limit`](crate::server::ServerConfig::piece_limit) gets `TimedOut`. `then` runs
    /// inside the connection's own frame and must return quickly; the frame it returns is offered to the sink.
    pub fn stream(mut self, head: Response, then: Continuation<Result<ReplyStream, StreamError>>) {
        self.sent = true;
        self.conn.post(Event::StreamHead {
            seq: self.seq,
            head,
            then,
        });
    }

    /// Asks for `frame` to be offered to the sink if this request is cancelled (the peer ended its stream or broke
    /// it while the request is handled): the way to be told without looking at the flag. It runs at most once, and
    /// never if the reply is sent before the cancel is seen: it is then dropped without running. The two race by
    /// nature (the reply and the peer's departure arrive in no fixed order), so the frame may also run after the
    /// reply was sent, or when the reply is still sent because the peer only half-closed. Each
    /// frame runs at most once: before a cancel is seen a later call replaces the earlier frame (which is dropped);
    /// once it is seen, every frame asked for while the reply is still owed runs at once, so a call before the
    /// cancel and one after it both run.
    pub fn on_cancel(&self, frame: Box<dyn Frame>) {
        self.conn.post(Event::OnCancel {
            seq: self.seq,
            frame,
        });
    }

    /// Set when the peer ended its stream or broke it while this request was being handled: the answer would be
    /// discarded, so work for it can stop. The flag is only ever set, never cleared.
    pub fn cancelled(&self) -> &Arc<AtomicBool> {
        &self.cancel
    }
}

impl Drop for Responder {
    fn drop(&mut self) {
        if !self.sent {
            self.conn.post(Event::Reply {
                seq: self.seq,
                response: Response::with_status(500),
            });
        }
    }
}

/// The right to send the rest of one streamed reply, from [`Responder::stream`]: whoever holds it owns it, and a
/// piece in flight takes it away, so there is never more than one piece in flight (a second [`send`](Self::send)
/// before the first one is out does not compile: `send` consumes the stream, and it comes back only in
/// [`PieceSent`]).
///
/// ```compile_fail
/// # use euhedral_io::Continuation;
/// # use euhedral_net::{PieceSent, ReplyStream, StreamError};
/// # type Then = Continuation<Result<PieceSent, StreamError>>;
/// # fn two(stream: ReplyStream, then: Then, again: Then) {
/// stream.send(b"a".to_vec(), then);
/// stream.send(b"b".to_vec(), again); // the stream was moved by the first send
/// # }
/// ```
///
/// A `ReplyStream` dropped without [`end`](Self::end) cuts the stream: the connection closes and the body has no
/// last chunk, so the peer sees it was cut (an HTTP/1.0 peer cannot: the close is the end of its body). A stream
/// held keeps the server from reporting its stop, as a `Responder` does.
pub struct ReplyStream {
    conn: Arc<Conn>,
    seq: u64,
    cancel: Arc<AtomicBool>,
    /// The stream has not ended or been given to a piece: its drop cuts the stream.
    live: bool,
}

impl Debug for ReplyStream {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("ReplyStream")
            .field("seq", &self.seq)
            .field("cancelled", &self.cancel.load(Ordering::Relaxed))
            .finish_non_exhaustive()
    }
}

impl ReplyStream {
    /// Sends `piece` as the next part of the body and returns at once. `then` runs once it is written, with the
    /// stream back and the piece's buffer emptied (its capacity kept, for the next piece), or with the error that
    /// ended the stream (the stream is then gone). An empty piece writes nothing and does not end the body. `then`
    /// runs inside the connection's own frame and must return quickly; the frame it returns is offered to the sink.
    pub fn send(mut self, piece: Vec<u8>, then: Continuation<Result<PieceSent, StreamError>>) {
        self.live = false;
        self.conn.post(Event::Piece {
            seq: self.seq,
            bytes: piece,
            then,
        });
    }

    /// Ends the body cleanly: the last chunk is written (on HTTP/1.1) and the connection then serves its next
    /// request, or closes if it was to.
    pub fn end(mut self) {
        self.live = false;
        self.conn.post(Event::StreamEnd {
            seq: self.seq,
            clean: true,
        });
    }

    /// Asks for `frame` to be offered to the sink if the request is cancelled while the stream is going (the peer
    /// ended its stream or broke it); the rules are those of [`Responder::on_cancel`], for the whole stream.
    pub fn on_cancel(&self, frame: Box<dyn Frame>) {
        self.conn.post(Event::OnCancel {
            seq: self.seq,
            frame,
        });
    }

    /// Set when the peer ended its stream or broke it while the stream is going; only ever set, never cleared. A
    /// peer that only half-closed may still read what is sent.
    pub fn cancelled(&self) -> &Arc<AtomicBool> {
        &self.cancel
    }
}

impl Drop for ReplyStream {
    fn drop(&mut self) {
        if self.live {
            self.conn.post(Event::StreamEnd {
                seq: self.seq,
                clean: false,
            });
        }
    }
}

/// A piece is written: the stream back, and the piece's buffer, emptied, with its capacity kept.
#[derive(Debug)]
pub struct PieceSent {
    pub stream: ReplyStream,
    pub piece: Vec<u8>,
}

/// Why a stream did not start or went no further. The stream is gone with it.
#[derive(Debug)]
pub enum StreamError {
    /// The head cannot be written (see [`WriteError`]); the request is answered `500`.
    Head(WriteError),
    /// The connection ended (the peer left, a stop cut the connection, or an earlier failure closed it).
    Closed,
    /// A write passed `piece_limit`: the peer does not read.
    TimedOut,
    /// A write failed.
    Io(Error),
}

impl Display for StreamError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            StreamError::Head(error) => {
                write!(f, "the head of the stream cannot be written: {error}")
            }
            StreamError::Closed => f.write_str("the connection ended"),
            StreamError::TimedOut => f.write_str("a write of the stream passed its time limit"),
            StreamError::Io(error) => write!(f, "a write of the stream failed: {error}"),
        }
    }
}

impl StdError for StreamError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            StreamError::Head(error) => Some(error),
            StreamError::Io(error) => Some(error),
            StreamError::Closed | StreamError::TimedOut => None,
        }
    }
}

/// What a failed send of a stream means to its handler: a peer that reset or closed the connection is `Closed`,
/// whether the connection saw it first or the send did.
fn stream_error(error: Error) -> StreamError {
    if LimitPassed::is(&error) {
        StreamError::TimedOut
    } else if Cancelled::is(&error)
        || matches!(
            error.kind(),
            ErrorKind::BrokenPipe
                | ErrorKind::ConnectionReset
                | ErrorKind::ConnectionAborted
                | ErrorKind::NotConnected
        )
    {
        StreamError::Closed
    } else {
        StreamError::Io(error)
    }
}
