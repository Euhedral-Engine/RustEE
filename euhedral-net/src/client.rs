//! One HTTP/1.1 request to a socket address, answered through a continuation.
//!
//! # Shape: step frames, no source
//!
//! A call is a chain of frames. Its state, a [`Call`], is passed by ownership: the continuation of an `AsyncIo`
//! operation moves the call into the step frame it returns, the step frame's `execute` takes it out, handles what
//! the operation brought and starts the next operation with the call moved into that operation's continuation. At
//! any time exactly one place owns the call (the operation in flight, or the frame about to run), so the state is
//! plain data with no cell, queue, lock or atomic, and the only thing a call costs on the lattice is what each
//! completion costs anyway: the frame its continuation returns. A source per call (as the server has per
//! connection) would add an allocation, the locks and atomics of attaching and detaching it and an inbox, to
//! order events that cannot arrive together here: a call has one operation in flight at any moment, and nothing
//! else touches it.
//!
//! The stages are connect, send the whole request, then receive until the parser has one final response. A
//! stage that does not start (the overall limit has passed, or the engine refuses the operation) ends the call
//! the same way as one that fails.
//!
//! # Ends
//!
//! The continuation `then` of [`request`] runs exactly once, on whichever path ends the call, and the frame it
//! returns is offered to the sink. If the operation's continuation or the step frame is dropped without having run
//! (the engine refused to start the operation, or was dropped, or the lattice discarded the frame), the call's
//! drop runs `then` with [`NetError::Abandoned`]. The socket is closed before `then` runs.

use std::error::Error;
use std::fmt::{self, Formatter};
use std::io;
use std::mem::take;
use std::net::SocketAddr;
#[cfg(unix)]
use std::os::fd::AsRawFd;
#[cfg(windows)]
use std::os::windows::io::AsRawSocket;
use std::sync::{Arc, Weak};
use std::time::{Duration, Instant};

use euhedral_core::frames::{Frame, FrameError, FrameHeader};
use euhedral_core::ingest::QueueIngestSink;
use euhedral_io::{
    AsyncIo, Continuation, IoBuf, IoDone, LimitPassed, OpOptions, OwnedSock, RawSock,
};

use crate::message::{ConnectionMode, Method, ParseError, Request, Response, WriteError};
use crate::parse::{Limits, Progress, ResponseParser, State};
use crate::write::write_request;

/// The most interim (1xx) responses a call skips before the final one.
pub const MAX_INTERIM_RESPONSES: usize = 8;

/// The part of a call that was going on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CallStage {
    /// Connecting.
    Connect,
    /// Sending the request.
    Send,
    /// Reading the status line and headers (and any interim response).
    ReceiveHead,
    /// Reading the body.
    ReceiveBody,
}

/// Why a call did not end with a response.
#[derive(Debug)]
pub enum NetError {
    /// The connection could not be made: refused, unreachable, or the socket could not be created. The error of
    /// the operating system is kept.
    Connect(io::Error),
    /// The overall time limit passed, in this stage.
    TimedOut(CallStage),
    /// An operation failed in this stage for another reason (a reset, a broken pipe). The error is kept. A peer
    /// that answers early and closes while the request is still being sent shows up here, as the send's error, and
    /// its response is lost.
    Io { stage: CallStage, error: io::Error },
    /// The peer closed the connection in this stage, before a whole response (`ReceiveHead`: before the head
    /// was complete; `ReceiveBody`: before the body was).
    PeerClosed(CallStage),
    /// The response is not valid HTTP/1.1 (the parser's error is kept).
    Malformed(ParseError),
    /// A valid response that the client does not read: no `Content-Length` where the body needs one (it would
    /// end with the connection), or a `Transfer-Encoding`.
    Unsupported(ParseError),
    /// The response is over the call's [`Limits`] (status line, header bytes, header count or body).
    OverLimit(ParseError),
    /// An interim response that is not skipped: a `101` (the connection would change protocol), or more than
    /// [`MAX_INTERIM_RESPONSES`] of them. The status of the one that stopped the call.
    UnexpectedInterim(u16),
    /// The request cannot be written (the writer's error is kept); nothing was sent.
    Write(WriteError),
    /// The call was dropped without having ended: the `AsyncIo` refused to start an operation (it is complete;
    /// the engine drops the continuation, and the call with it, before it returns its error, so the real error
    /// is not available), or was dropped with the operation in flight, or the lattice discarded the call's frame.
    Abandoned(CallStage),
}

impl fmt::Display for NetError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            NetError::Connect(e) => write!(f, "cannot connect: {e}"),
            NetError::TimedOut(s) => write!(f, "the time limit passed ({s:?})"),
            NetError::Io { stage, error } => write!(f, "{stage:?} failed: {error}"),
            NetError::PeerClosed(s) => write!(f, "the peer closed the connection ({s:?})"),
            NetError::Malformed(e) => write!(f, "malformed response: {e}"),
            NetError::Unsupported(e) => write!(f, "response not supported: {e}"),
            NetError::OverLimit(e) => write!(f, "response over the limits: {e}"),
            NetError::UnexpectedInterim(s) => write!(f, "interim response {s} not accepted"),
            NetError::Write(e) => write!(f, "request cannot be written: {e}"),
            NetError::Abandoned(s) => write!(f, "the call was dropped ({s:?})"),
        }
    }
}

impl Error for NetError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            NetError::Connect(e) | NetError::Io { error: e, .. } => Some(e),
            NetError::Malformed(e) | NetError::Unsupported(e) | NetError::OverLimit(e) => Some(e),
            NetError::Write(e) => Some(e),
            _ => None,
        }
    }
}

/// The settings of one call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClientConfig {
    /// The response's size limits.
    pub limits: Limits,
    /// The one time limit of the whole call, from the call of [`request`] to the end of the response. Each
    /// operation is issued with what is left of it. `None` is no limit.
    pub time_limit: Option<Duration>,
    /// The size of the buffer of each `recv`.
    pub read_chunk: usize,
}

impl Default for ClientConfig {
    fn default() -> ClientConfig {
        ClientConfig {
            limits: Limits::default(),
            time_limit: Some(Duration::from_secs(30)),
            read_chunk: 16 * 1024,
        }
    }
}

/// What an operation brought to the next step.
enum Event {
    Connected(io::Result<()>),
    Sent(IoDone),
    Received(IoDone),
}

/// The state of one call: owned by whoever holds it (one place at a time).
struct Call {
    io: Weak<AsyncIo>,
    sink: Arc<QueueIngestSink>,
    socket: Option<OwnedSock>,
    to: SocketAddr,
    deadline: Option<Instant>,
    parser: ResponseParser,
    /// The serialised request, until it is sent.
    out: Vec<u8>,
    /// The buffer of the next `recv`.
    spare: Vec<u8>,
    interim: usize,
    /// The stage of the operation in flight, for a call dropped with it.
    stage: CallStage,
    then: Option<Continuation<Result<Response, NetError>>>,
}

#[cfg(unix)]
fn raw_of(socket: &OwnedSock) -> RawSock {
    socket.as_raw_fd()
}

#[cfg(windows)]
fn raw_of(socket: &OwnedSock) -> RawSock {
    socket.as_raw_socket()
}

impl Call {
    /// Ends the call: the socket closes, `then` runs and the frame it returns goes to the sink.
    fn deliver(&mut self, result: Result<Response, NetError>) {
        if let (Some(socket), Some(io)) = (&self.socket, self.io.upgrade()) {
            io.forget_socket(raw_of(socket));
        }
        self.socket = None;
        if let Some(then) = self.then.take() {
            let _ = self.sink.offer(then(result));
        }
    }

    fn finish(mut self: Box<Call>, result: Result<Response, NetError>) {
        self.deliver(result);
    }

    /// The time left, `Ok(None)` for no limit, `Err` when none is left.
    fn left(&self) -> Result<Option<Duration>, ()> {
        match self.deadline {
            None => Ok(None),
            Some(deadline) => {
                let left = deadline.saturating_duration_since(Instant::now());
                if left.is_zero() {
                    Err(())
                } else {
                    Ok(Some(left))
                }
            }
        }
    }

    fn receive_stage(&self) -> CallStage {
        match self.parser.state() {
            State::Body { .. } => CallStage::ReceiveBody,
            _ => CallStage::ReceiveHead,
        }
    }

    fn start(mut self: Box<Call>) {
        let Ok(limit) = self.left() else {
            return self.finish(Err(NetError::TimedOut(CallStage::Connect)));
        };
        let Some(io) = self.io.upgrade() else {
            return self.finish(Err(NetError::Abandoned(CallStage::Connect)));
        };
        let (Some(socket), to) = (&self.socket, self.to) else {
            return;
        };
        let raw = raw_of(socket);
        self.stage = CallStage::Connect;
        // A refused start drops the continuation, and with it the call, which reports `Abandoned`.
        let _ = io.connect_with(
            raw,
            to,
            OpOptions { limit },
            Box::new(move |result| Step::frame(self, Event::Connected(result))),
        );
    }

    fn on_connected(self: Box<Call>, result: io::Result<()>) {
        match result {
            Ok(()) => self.send(),
            Err(e) if LimitPassed::is(&e) => {
                self.finish(Err(NetError::TimedOut(CallStage::Connect)))
            }
            Err(e) => self.finish(Err(NetError::Connect(e))),
        }
    }

    fn send(mut self: Box<Call>) {
        let Ok(limit) = self.left() else {
            return self.finish(Err(NetError::TimedOut(CallStage::Send)));
        };
        let Some(io) = self.io.upgrade() else {
            return self.finish(Err(NetError::Abandoned(CallStage::Send)));
        };
        let Some(socket) = &self.socket else {
            return;
        };
        let raw = raw_of(socket);
        let bytes = take(&mut self.out);
        let filled = bytes.len();
        self.stage = CallStage::Send;
        let _ = io.send_with(
            raw,
            IoBuf { bytes, filled },
            OpOptions { limit },
            Box::new(move |done| Step::frame(self, Event::Sent(done))),
        );
    }

    fn on_sent(self: Box<Call>, done: IoDone) {
        match done.result {
            Ok(_) => self.receive(),
            Err(e) if LimitPassed::is(&e) => self.finish(Err(NetError::TimedOut(CallStage::Send))),
            Err(error) => self.finish(Err(NetError::Io {
                stage: CallStage::Send,
                error,
            })),
        }
    }

    fn receive(mut self: Box<Call>) {
        let stage = self.receive_stage();
        let Ok(limit) = self.left() else {
            return self.finish(Err(NetError::TimedOut(stage)));
        };
        let Some(io) = self.io.upgrade() else {
            return self.finish(Err(NetError::Abandoned(stage)));
        };
        let Some(socket) = &self.socket else {
            return;
        };
        let raw = raw_of(socket);
        let bytes = take(&mut self.spare);
        self.stage = stage;
        let _ = io.recv_with(
            raw,
            IoBuf { bytes, filled: 0 },
            OpOptions { limit },
            Box::new(move |done| Step::frame(self, Event::Received(done))),
        );
    }

    fn on_received(mut self: Box<Call>, done: IoDone) {
        let stage = self.receive_stage();
        let IoDone { buf, result } = done;
        let n = match result {
            Ok(0) => return self.finish(Err(NetError::PeerClosed(stage))),
            Ok(n) => n,
            Err(e) if LimitPassed::is(&e) => return self.finish(Err(NetError::TimedOut(stage))),
            Err(error) => return self.finish(Err(NetError::Io { stage, error })),
        };
        let mut rest = &buf.bytes[..n];
        while !rest.is_empty() {
            match self.parser.feed(rest) {
                Err(e) => return self.finish(Err(classify(e))),
                Ok((used, Progress::Complete(response))) => {
                    if response.status >= 200 {
                        return self.finish(Ok(response));
                    }
                    self.interim += 1;
                    if response.status == 101 || self.interim > MAX_INTERIM_RESPONSES {
                        return self.finish(Err(NetError::UnexpectedInterim(response.status)));
                    }
                    rest = &rest[used..];
                }
                Ok((used, _)) => rest = &rest[used..],
            }
        }
        self.spare = buf.bytes;
        self.receive();
    }
}

impl Drop for Call {
    fn drop(&mut self) {
        if self.then.is_some() {
            let stage = self.stage;
            self.deliver(Err(NetError::Abandoned(stage)));
        }
    }
}

/// Which error class a parser error is. Every variant is named, so a new one does not compile until it is placed.
fn classify(error: ParseError) -> NetError {
    match error {
        ParseError::MissingContentLength
        | ParseError::TransferEncodingNotSupported
        | ParseError::ChunkedNotSupported
        | ParseError::UnsupportedVersion => NetError::Unsupported(error),
        ParseError::StatusLineTooLong
        | ParseError::RequestLineTooLong
        | ParseError::HeadersTooLarge
        | ParseError::TooManyHeaders
        | ParseError::BodyTooLarge => NetError::OverLimit(error),
        ParseError::BadRequestLine
        | ParseError::BadStatusLine
        | ParseError::BadMethod
        | ParseError::BadTarget
        | ParseError::BadVersion
        | ParseError::BadLineEnding
        | ParseError::BadHeaderLine
        | ParseError::ObsoleteFolding
        | ParseError::SpaceBeforeColon
        | ParseError::BadHeaderName
        | ParseError::BadHeaderValue
        | ParseError::BadHost
        | ParseError::BadContentLength
        | ParseError::ConflictingContentLength
        | ParseError::TransferEncodingWithContentLength
        | ParseError::BadTransferEncoding => NetError::Malformed(error),
    }
}

/// The frame an operation's continuation returns: it carries the call and what the operation brought, and its
/// `execute` runs the next step.
struct Step {
    header: FrameHeader,
    work: Option<(Box<Call>, Event)>,
}

impl Step {
    fn frame(call: Box<Call>, event: Event) -> Box<dyn Frame> {
        Box::new(Step {
            header: FrameHeader::new(0),
            work: Some((call, event)),
        })
    }
}

impl Frame for Step {
    fn header(&self) -> &FrameHeader {
        &self.header
    }

    fn header_mut(&mut self) -> &mut FrameHeader {
        &mut self.header
    }

    fn execute(&mut self) -> Result<(), FrameError> {
        if let Some((call, event)) = self.work.take() {
            match event {
                Event::Connected(result) => call.on_connected(result),
                Event::Sent(done) => call.on_sent(done),
                Event::Received(done) => call.on_received(done),
            }
        }
        Ok(())
    }
}

/// Makes one request to `to` and calls `then` with the response or the error, exactly once.
///
/// One connection per call: connect, send `request` with `Connection: close`, read one response with a
/// `Content-Length` (or none, for a HEAD request or a status without a body), close. The frame `then` returns is
/// offered to `sink`; `io` is held weakly, so keep it alive while calls are in flight. Every operation is issued
/// with what is left of `config.time_limit`, so the whole call ends within it (the operations that cannot start
/// within it end the call with [`NetError::TimedOut`]).
///
/// Returns at once, without waiting for anything; `then` may run before this returns (when the request cannot be
/// written or the socket cannot be made).
///
/// Interim responses (1xx) are skipped, up to [`MAX_INTERIM_RESPONSES`]; a `101` is refused. A response with no
/// `Content-Length` that needs one, or with a `Transfer-Encoding`, is refused (the client cannot read to the end
/// of the connection or read chunks): see [`NetError::Unsupported`].
pub fn request(
    io: &Arc<AsyncIo>,
    sink: &Arc<QueueIngestSink>,
    to: SocketAddr,
    request: &Request,
    config: ClientConfig,
    then: Continuation<Result<Response, NetError>>,
) {
    let mut out = Vec::new();
    if let Err(e) = write_request(request, ConnectionMode::Close, &mut out) {
        let _ = sink.offer(then(Err(NetError::Write(e))));
        return;
    }
    let socket = match io.socket_for(to) {
        Ok(socket) => socket,
        Err(e) => {
            let _ = sink.offer(then(Err(NetError::Connect(e))));
            return;
        }
    };
    let mut parser = ResponseParser::new(config.limits);
    parser.set_response_to_head(request.method == Method::Head);
    let call = Box::new(Call {
        io: Arc::downgrade(io),
        sink: sink.clone(),
        socket: Some(socket),
        to,
        // A limit too long to add to the clock is no limit.
        deadline: config
            .time_limit
            .and_then(|limit| Instant::now().checked_add(limit)),
        parser,
        out,
        spare: vec![0; config.read_chunk.max(1)],
        interim: 0,
        stage: CallStage::Connect,
        then: Some(then),
    });
    call.start();
}
