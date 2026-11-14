//! One connection of the server: its state, the events that change it, and the `Responder` a handler answers
//! through.
//!
//! # Who owns what
//!
//! A [`Conn`] is shared (by `Arc`) between the operations in flight, the registry of the server and the
//! `Responder`, but everything that changes lives in one [`ConnState`] inside a `ClaimCell`, and **only frames
//! ordered by the connection's id hash touch it**. Such frames run one at a time, so the claim never fails; a
//! claim that does fail (an ordering that was not kept) re-offers the event rather than waiting. The state is plain:
//! the parser, the bytes read and not yet parsed, the socket, the id of the `recv` in flight and the phase need
//! no lock and no atomic of their own.
//!
//! Everything reaches the state as an event: the continuation of a `recv` or `send` only wraps the result in a
//! frame (it runs inside the reaping frame, on any worker), `Responder::send` offers a frame, and the server's
//! stop offers one. The event frame carries the connection's hash, so it runs in the connection's order.
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
//!   the budget of bytes is spent or `write_limit` passes.
//! - `Closed`: the socket is dropped as soon as no operation is in flight on it (a cancelled `recv` ends first).

use std::net::{Shutdown, TcpStream};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use euhedral_core::frames::{Frame, FrameError, FrameHeader};
use euhedral_data_structures::sync::ClaimCell;
use euhedral_io::{Cancelled, IoBuf, IoDone, LimitPassed, OpId, OpOptions, RawSock};

use crate::message::{ConnectionMode, Method, ParseError, Request, Response, Version, WriteError};
use crate::parse::{Progress, RequestParser};
use crate::server::{ListenerEvent, Shared};
use crate::write::{write_continue, write_response, write_response_head};

/// The handle of a connected stream, as `AsyncIo` takes it.
#[cfg(unix)]
pub(crate) fn raw_stream(stream: &TcpStream) -> RawSock {
    std::os::fd::AsRawFd::as_raw_fd(stream)
}

/// The handle of a connected stream, as `AsyncIo` takes it.
#[cfg(windows)]
pub(crate) fn raw_stream(stream: &TcpStream) -> RawSock {
    std::os::windows::io::AsRawSocket::as_raw_socket(stream)
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
}

/// The plain state of a connection.
struct ConnState {
    /// The socket, until the connection is finished.
    stream: Option<TcpStream>,
    parser: RequestParser,
    /// Bytes received and not yet parsed.
    buf: Vec<u8>,
    /// The buffer of the `recv`, while none is in flight.
    spare: Option<Vec<u8>>,
    phase: Phase,
    /// The `recv` in flight.
    recv: Option<OpId>,
    /// A `send` is in flight.
    sending: bool,
    /// When the first byte of the request being read was seen.
    started: Option<Instant>,
    /// The number of the request being handled, or the last one.
    seq: u64,
    stopping: bool,
    /// The peer ended or broke the stream while a reply was going out.
    peer_gone: bool,
    /// Bytes the closing phase may still read and drop.
    linger_budget: usize,
    /// Where the connection sits in the server's registry.
    slot: usize,
    /// The server has been told that this connection ended.
    reported: bool,
}

/// A connection: the immutable part and the claim on its state.
pub(crate) struct Conn {
    shared: Arc<Shared>,
    /// The id hash of every frame of this connection.
    pub(crate) hash: u64,
    state: ClaimCell<ConnState>,
}

impl std::fmt::Debug for Conn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Conn")
            .field("hash", &self.hash)
            .finish_non_exhaustive()
    }
}

/// The frame that carries an event to a connection's order.
struct ConnFrame {
    header: FrameHeader,
    conn: Arc<Conn>,
    event: Option<Event>,
}

impl Frame for ConnFrame {
    fn header(&self) -> &FrameHeader {
        &self.header
    }

    fn header_mut(&mut self) -> &mut FrameHeader {
        &mut self.header
    }

    fn execute(&mut self) -> Result<(), FrameError> {
        if let Some(event) = self.event.take() {
            self.conn.run(event);
        }
        Ok(())
    }
}

/// The frame that carries `event` to `conn`'s order.
pub(crate) fn conn_frame(conn: Arc<Conn>, event: Event) -> Box<dyn Frame> {
    Box::new(ConnFrame {
        header: FrameHeader::new(conn.hash),
        conn,
        event: Some(event),
    })
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
    pub(crate) fn new(shared: Arc<Shared>, hash: u64, stream: TcpStream, slot: usize) -> Conn {
        let parser = RequestParser::new(shared.limits);
        let linger_budget = shared
            .config
            .max_header_bytes
            .saturating_add(shared.config.max_body_bytes);
        Conn {
            shared,
            hash,
            state: ClaimCell::new(ConnState {
                stream: Some(stream),
                parser,
                buf: Vec::new(),
                spare: None,
                phase: Phase::Reading,
                recv: None,
                sending: false,
                started: None,
                seq: 0,
                stopping: false,
                peer_gone: false,
                linger_budget,
                slot,
                reported: false,
            }),
        }
    }

    /// Offers `event` to this connection's order.
    pub(crate) fn post(self: &Arc<Self>, event: Event) {
        let frame = conn_frame(self.clone(), event);
        // A sink that has completed refuses; the event is dropped and the connection is left to its peer.
        let _ = self.shared.sink.offer_seeded(self.hash, frame);
    }

    /// Runs one event: the only way into the state.
    fn run(self: &Arc<Self>, event: Event) {
        let Some(mut guard) = self.state.try_claim() else {
            // Frames of one hash run one at a time, so this is an ordering that was not kept: try again
            // through the sink rather than wait.
            self.post(event);
            return;
        };
        let st: &mut ConnState = &mut guard;
        match event {
            Event::Start => {
                if st.stopping {
                    self.close_now(st);
                } else {
                    self.begin_request(st);
                }
            }
            Event::Recv(done) => self.on_recv(st, done),
            Event::Sent(done) => self.on_sent(st, done),
            Event::Reply { seq, response } => self.on_reply(st, seq, response),
            Event::Stop => {
                st.stopping = true;
                if matches!(st.phase, Phase::Reading | Phase::Closing) {
                    self.close_now(st);
                }
            }
        }
    }

    // ---- operations ----------------------------------------------------------------------------------------

    /// Starts the one `recv` of the connection, bounded by `limit`.
    fn issue_recv(self: &Arc<Self>, st: &mut ConnState, limit: Option<Duration>) {
        debug_assert!(st.recv.is_none());
        let Some(stream) = &st.stream else {
            return;
        };
        let bytes = st
            .spare
            .take()
            .unwrap_or_else(|| vec![0; self.shared.config.read_chunk.max(1)]);
        let conn = self.clone();
        let started = self.shared.io.recv_with(
            raw_stream(stream),
            IoBuf { bytes, filled: 0 },
            OpOptions { limit },
            Box::new(move |done| conn_frame(conn, Event::Recv(done))),
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
        debug_assert!(!st.sending);
        let Some(stream) = &st.stream else {
            return;
        };
        let filled = bytes.len();
        let conn = self.clone();
        let started = self.shared.io.send_with(
            raw_stream(stream),
            IoBuf { bytes, filled },
            OpOptions {
                limit: self.shared.config.write_limit,
            },
            Box::new(move |done| conn_frame(conn, Event::Sent(done))),
        );
        match started {
            Ok(_) => st.sending = true,
            Err(_) => self.close_now(st),
        }
    }

    // ---- reading -------------------------------------------------------------------------------------------

    /// Starts on the next request: from the bytes already held, or by waiting for the first one.
    fn begin_request(self: &Arc<Self>, st: &mut ConnState) {
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
                    self.dispatch(st, request);
                    return;
                }
            }
        }
    }

    /// A parse error: the status of the error, then the end of the connection.
    fn parse_failed(self: &Arc<Self>, st: &mut ConnState, error: ParseError) {
        self.shared
            .counters
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
            Ok(_) => Got::Data(buf.filled),
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
        if matches!(got, Got::Limit) {
            self.shared
                .counters
                .limits_passed
                .fetch_add(1, Ordering::Relaxed);
        }
        match st.phase {
            Phase::Closed => self.finalize(st),
            Phase::Closing => match got {
                Got::Data(n) => {
                    st.linger_budget = st.linger_budget.saturating_sub(n);
                    if st.linger_budget == 0 {
                        self.close_now(st);
                    } else {
                        self.issue_recv(st, self.shared.config.write_limit);
                    }
                }
                _ => self.close_now(st),
            },
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
                Got::Eof | Got::Limit | Got::Failed => self.peer_left(st),
            },
            Phase::Replying { .. } => {
                if matches!(got, Got::Eof | Got::Limit | Got::Failed) {
                    st.peer_gone = true;
                }
                self.settle(st);
            }
            Phase::Continuing => {}
        }
    }

    /// The peer ended its stream or broke it while the handler has its request: its cancel flag is set, and the
    /// connection ends; the handler's reply, when it comes, is discarded.
    fn peer_left(self: &Arc<Self>, st: &mut ConnState) {
        if let Phase::Handling { cancel, .. } = &st.phase {
            cancel.store(true, Ordering::Release);
            self.shared
                .counters
                .cancelled
                .fetch_add(1, Ordering::Relaxed);
        }
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
            keep: request.keep_alive() && !st.stopping,
            version: request.version,
            cancel: cancel.clone(),
        };
        self.shared
            .counters
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
        let responder = Responder {
            conn: self.clone(),
            seq,
            cancel,
            sent: false,
        };
        let frame = self.shared.handler.handle(request, responder);
        // A sink that has completed refuses: the frame (and the `Responder` in it, which answers 500 to nobody)
        // is dropped.
        let _ = self.shared.sink.offer(frame);
    }

    fn on_reply(self: &Arc<Self>, st: &mut ConnState, seq: u64, response: Response) {
        let Phase::Handling {
            seq: current,
            head,
            keep,
            version,
            ..
        } = &st.phase
        else {
            self.shared
                .counters
                .discarded
                .fetch_add(1, Ordering::Relaxed);
            return;
        };
        if *current != seq {
            self.shared
                .counters
                .discarded
                .fetch_add(1, Ordering::Relaxed);
            return;
        }
        let (head, version) = (*head, *version);
        let keep = *keep && !st.stopping && !st.peer_gone;
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
            self.shared.io.cancel(id);
        }
        self.issue_send(st, out);
    }

    fn on_sent(self: &Arc<Self>, st: &mut ConnState, done: IoDone) {
        st.sending = false;
        if let Err(error) = &done.result {
            if LimitPassed::is(error) {
                self.shared
                    .counters
                    .limits_passed
                    .fetch_add(1, Ordering::Relaxed);
            }
            // A stream a send ended early on is at a position only the count tells: it is closed.
            self.close_now(st);
            return;
        }
        match st.phase {
            Phase::Continuing => {
                st.phase = Phase::Reading;
                self.parse_loop(st);
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
        if st.sending || st.recv.is_some() {
            return;
        }
        if st.peer_gone {
            self.close_now(st);
        } else if keep && !st.stopping {
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
        self.issue_recv(st, self.shared.config.write_limit);
    }

    // ---- ending --------------------------------------------------------------------------------------------

    /// Ends the connection: cancels the `recv` in flight, and drops the socket once nothing is in flight on it.
    fn close_now(self: &Arc<Self>, st: &mut ConnState) {
        st.phase = Phase::Closed;
        if let Some(id) = st.recv {
            self.shared.io.cancel(id);
        }
        self.finalize(st);
    }

    fn finalize(self: &Arc<Self>, st: &mut ConnState) {
        if !matches!(st.phase, Phase::Closed) || st.recv.is_some() || st.sending || st.reported {
            return;
        }
        st.reported = true;
        st.stream = None;
        st.buf = Vec::new();
        st.spare = None;
        self.shared.post_listener(ListenerEvent::Closed(st.slot));
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

impl std::fmt::Debug for Responder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
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
