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
//! - `Replying`: the reply is being sent; the `recv` of `Handling`, if still in flight, was cancelled.
//! - `Closing`: the reply that ended the connection is out and the write side is shut down; what the peer still
//!   sends is read and dropped (so that closing does not reset the connection and lose the reply) until it closes,
//!   the budget of bytes is spent or the linger deadline passes (one deadline for the whole linger, from its start:
//!   `linger_limit`, or `stop_linger_limit` if it is shorter and the server is stopping).
//! - `Closed`: the socket is dropped as soon as no operation is in flight on it (a cancelled operation ends
//!   first). The server is told the connection ended once the socket is dropped *and* the handler's `Responder`
//!   for the request being handled, if any, has been used or dropped and its reply has been through the
//!   connection's order (a discarded one); so a stop is complete only when no frame of the server is left.

use std::fmt::{self, Debug, Formatter};
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
use euhedral_io::{Cancelled, IoBuf, IoDone, LimitPassed, OpId, OpOptions, RawSock};

use crate::message::{ConnectionMode, Method, ParseError, Request, Response, Version, WriteError};
use crate::parse::{Progress, RequestParser};
use crate::server::{Ctx, ListenerEvent, Shared, no_work};
use crate::station::{Drive, Station};
use crate::write::{write_continue, write_response, write_response_head};

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
    Replying {
        keep: bool,
    },
    Closing,
    Closed,
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
}

/// The plain state of a connection.
struct ConnState {
    /// The socket, until the connection is finished.
    stream: Option<TcpStream>,
    /// What the caller owns (`AsyncIo`, sink, handler), until the server is told that the connection ended.
    ctx: Option<Arc<Ctx>>,
    /// The frame to offer if the request being handled is cancelled.
    on_cancel: Option<Box<dyn Frame>>,
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
    /// A `Responder` of this connection has not yet been through the connection's order.
    owed: bool,
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
        }
    }

    // ---- operations ----------------------------------------------------------------------------------------

    /// Cancels an operation of this connection.
    fn cancel_op(&self, st: &ConnState, id: OpId) {
        if let Some(ctx) = &st.ctx {
            ctx.io.cancel(id);
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
        let started = ctx.io.recv_with(
            raw_stream(stream),
            IoBuf { bytes, filled: 0 },
            OpOptions { limit },
            Box::new(move |done| {
                conn.post(Event::Recv(done));
                no_work()
            }),
        );
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

    /// Starts the one `send` of the connection. The engine moves every byte or ends with an error.
    fn issue_send(self: &Arc<Self>, st: &mut ConnState, bytes: Vec<u8>) {
        debug_assert!(st.send.is_none());
        let (Some(stream), Some(ctx)) = (&st.stream, &st.ctx) else {
            return;
        };
        let filled = bytes.len();
        let conn = self.clone();
        let started = ctx.io.send_with(
            raw_stream(stream),
            IoBuf { bytes, filled },
            OpOptions {
                limit: self.shared.config.write_limit,
            },
            Box::new(move |done| {
                conn.post(Event::Sent(done));
                no_work()
            }),
        );
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
            Phase::Handling { .. } => match got {
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
        if let Phase::Handling { cancel, .. } = &st.phase {
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

    /// The handler asked for a frame to be offered when the request is cancelled. A request that is already
    /// cancelled offers it at once; one that is over drops it.
    fn on_cancel_asked(self: &Arc<Self>, st: &mut ConnState, seq: u64, frame: Box<dyn Frame>) {
        let Phase::Handling {
            seq: current,
            cancel,
            ..
        } = &st.phase
        else {
            return;
        };
        if *current != seq {
            return;
        }
        if cancel.load(Ordering::Acquire) {
            if let Some(ctx) = &st.ctx {
                let _ = ctx.sink.offer(frame);
            }
        } else {
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
        if let Err(error) = &done.result {
            if LimitPassed::is(error) {
                self.shared
                    .counters
                    .rare
                    .limits_passed
                    .fetch_add(1, Ordering::Relaxed);
            }
            // A stream a send ended early on is at a position only the count tells: it is closed.
            self.close_now(st);
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
            Phase::Handling { .. } | Phase::Replying { .. } | Phase::Closed => {}
        }
    }

    // ---- ending --------------------------------------------------------------------------------------------

    /// Ends the connection: cancels the `recv` in flight, and drops the socket once nothing is in flight on it.
    fn close_now(self: &Arc<Self>, st: &mut ConnState) {
        st.phase = Phase::Closed;
        if let Some(id) = st.recv {
            self.cancel_op(st, id);
        }
        if let Some(id) = st.send {
            self.cancel_op(st, id);
        }
        self.finalize(st);
    }

    fn finalize(self: &Arc<Self>, st: &mut ConnState) {
        if !matches!(st.phase, Phase::Closed) || st.recv.is_some() || st.send.is_some() {
            return;
        }
        st.stream = None;
        st.on_cancel = None;
        st.buf = Vec::new();
        st.spare = None;
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

    /// Asks for `frame` to be offered to the sink if this request is cancelled (the peer ended its stream or broke
    /// it while the request is handled): the way to be told without looking at the flag. It runs at most once, and
    /// never if the reply is sent before the cancel is seen: it is then dropped without running. The two race by
    /// nature (the reply and the peer's departure arrive in no fixed order), so the frame may also run after the
    /// reply was sent, or when the reply is still sent because the peer only half-closed. A second call replaces
    /// the frame of the first.
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
