# euhedral-net

HTTP/1.1 for programs built on RustEE. The crate is an HTTP/1.1 server and a minimal client over
[`euhedral-io`](../euhedral-io/README.md)'s `AsyncIo`: a request line and headers, bodies with `Content-Length`,
keep-alive, `Expect: 100-continue`, size limits, time limits and a connection cap. Out of scope for the crate:
TLS, HTTP/2, chunked request bodies (answered 411), compression, routing and JSON.

**This part.** The messages, a parser and a writer (pure: no I/O), the server and the client on `euhedral-io`. See the
[top-level glossary](../GLOSSARY.md) for the terms a user meets and [`GLOSSARY.md`](GLOSSARY.md) for this crate's.

| Item | What it is |
|---|---|
| [`Request`, `Response`, `Headers`, `Method`, `Version`](src/message.rs) | The messages. `Headers` keeps order and repeated names and finds a name without regard to case. |
| [`ParseError`, `WriteError`](src/message.rs) | Why input is refused. `ParseError::status()` is the status a server answers: 400, 411, 413, 414, 431 or 505, and `None` for an error that has no status to send (a response the client cannot read). |
| [`RequestParser`, `ResponseParser`, `Limits`, `Progress`, `State`, `PendingHead`](src/parse.rs) | The strict, incremental parsers and their size limits. |
| [`write_response`, `write_response_head`, `write_request`, `write_continue`, `ConnectionMode`](src/write.rs) | The writer. |
| [`write_chunked_head`, `write_chunk`, `write_last_chunk`](src/write.rs) | The writer for a body sent in pieces: the head with `Transfer-Encoding: chunked`, one chunk, the last chunk. |
| [`serve`, `ServerConfig`, `ServerHandle`, `Counters`, `Handler`, `no_work`](src/server.rs) | The server: accept loop, connection cap, counters (including `streams` and `streams_cut`), stop. Linux, Android and Windows. |
| [`request`, `ClientConfig`, `NetError`, `CallStage`](src/client.rs) | The client: one request to a socket address, answered through a continuation. Linux, Android and Windows. |
| [`Responder`](src/connection.rs) | The one reply of one request, and its cancel flag. `Responder::stream` answers with a reply whose body is sent piece by piece. |
| [`ReplyStream`, `PieceSent`, `StreamError`](src/connection.rs) | The right to send the rest of a streamed reply, one piece at a time; a piece written; why a stream did not start or went no further. |

## The parser

A parser is a state machine owned by the one consumer that reads a connection (a connection's frames are ordered,
so it needs no synchronisation). `feed(bytes)` consumes the bytes of the message being read, never more, and
returns how many it used with a `Progress`: `NeedMore`, `HeadComplete` (the headers are in and a body is pending;
the place to answer `100 Continue`) or `Complete(message)`. The caller keeps the rest of its slice for the next
call, so pipelined requests are never lost and the parser never buffers beyond its [`Limits`](src/parse.rs). A
slice may be split anywhere; the result does not change.

```rust
use euhedral_net::{Limits, Method, Progress, RequestParser};

let mut parser = RequestParser::new(Limits::default());
let wire = b"POST /echo HTTP/1.1\r\nHost: example\r\nContent-Length: 5\r\n\r\nhelloGET / HT";
let mut rest = &wire[..];
let mut seen = Vec::new();
while !rest.is_empty() {
    let (used, progress) = parser.feed(rest).expect("a valid request");
    rest = &rest[used..];
    match progress {
        Progress::HeadComplete { content_length, .. } => assert_eq!(content_length, 5),
        Progress::Complete(request) => seen.push(request),
        Progress::NeedMore => break,
    }
}
assert_eq!(seen.len(), 1);
assert_eq!(seen[0].method, Method::Post);
assert_eq!(seen[0].headers.get("host"), Some("example"));
assert_eq!(seen[0].body, b"hello");
assert!(seen[0].keep_alive());
// The start of the next request is held until the rest arrives.
assert_eq!(parser.buffered(), "GET / HT".len());
```

The parser is strict where the RFCs let a recipient be lenient, because leniency about where a message ends is
how a server and a proxy come to disagree about the next request. It refuses a bare LF, obsolete line folding,
whitespace before a colon, a control character in a value, differing `Content-Length` values, `Transfer-Encoding`
with `Content-Length`, more than one `Host` on any request (HTTP/1.1 needs exactly one), a `Host` that is not a host with an optional port (stricter than RFC 3986: a registered name is letters, digits and `-._~`, with no sub-delimiters and no percent escapes, since an escape can hide a byte such as `a%00b` or `a%2Fb`; an IPv6 literal must be a real IPv6 address and the IPvFuture form is refused; a port has at most five digits and is at most 65535), a target that does not fit its method (RFC 9112 section 3.2: CONNECT takes an authority, OPTIONS may take `*`, others an origin or absolute form), and every version but 1.0 and 1.1. HTTP/1.2 is answered 505, and a status line with no space before a missing reason is refused, both on purpose. Each
rule, its error and the RFC section it rests on are in the [`parse`](src/parse.rs) module notes and the
[changelog](../CHANGELOG.md).

**Routing.** A server routes on `Request::authority()` (also on `PendingHead`), not on the raw `Host` header: for an
absolute-form or authority-form target the authority of the target wins over `Host` (RFC 9112 section 3.2.2), else it
is the `Host` value, and it is `None` when there is neither. `PendingHead` also has `keep_alive()` and
`expects_continue()`, so a server can choose between `100 Continue` and an early final status before the body arrives.

**Limits of the crate.** A request with `Transfer-Encoding` is refused: 411 when its last non-empty coding is `chunked`
(the crate cannot read chunked bodies yet), and 400 when it is not (RFC 9112 section 6.3). Empty list elements are
ignored (`chunked,` and `,chunked` are chunked, RFC 9110 section 5.6.1). Any `Transfer-Encoding` on an HTTP/1.0
request is 400 (section 6.1). The client is minimal: a response must carry `Content-Length` unless it answers HEAD or has a status
without a body (1xx, 204, 304); a response with `Transfer-Encoding` (so a streamed reply of this crate's server), or one that ends with the
connection, is an error the client reports.

## The writer

```rust
use euhedral_net::{ConnectionMode, Response, write_response};

let mut out = Vec::new();
let response = Response::with_status(200).header("Content-Type", "text/plain").with_body("hi");
write_response(&response, ConnectionMode::Close, &mut out).expect("a valid response");
assert_eq!(
    out,
    b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: 2\r\nConnection: close\r\n\r\nhi"
);

// A header value that holds a line break is refused, and the output is left as it was.
let bad = Response::with_status(200).header("X", "a\r\nSet-Cookie: b");
assert!(write_response(&bad, ConnectionMode::Unspecified, &mut out).is_err());
```

The writer sets `Content-Length` from the body, adds `Connection: close` or `keep-alive` as the `ConnectionMode` says (never on a 1xx response: the mode is ignored there), and refuses a method, target, reason phrase, header name or header value that could put a CR, LF
or other control character into the output. A header named `Content-Length`, `Transfer-Encoding` or `Connection` is
the writer's: the caller cannot set it. `write_request` also needs exactly one valid `Host` and a target that fits the method, so what it writes parses back. `write_response_head` writes the head of a response to HEAD with the length the body would have. `write_continue` appends the interim `100 Continue`.

A body whose length is not known when the head goes out is sent in the chunked coding: `write_chunked_head` writes the
head with `Transfer-Encoding: chunked` and no `Content-Length` (and refuses a 1xx, 204 or 304, which have no body),
`write_chunk` frames one piece (its length in lowercase hexadecimal, CRLF, the bytes, CRLF; an empty piece appends
nothing, since a zero-length chunk would end the body) and `write_last_chunk` writes `0`, CRLF, CRLF. There are no
trailers.

```rust
use euhedral_net::{ConnectionMode, Response, write_chunk, write_chunked_head, write_last_chunk};

let mut out = Vec::new();
write_chunked_head(&Response::with_status(200), ConnectionMode::Unspecified, &mut out).expect("a valid head");
write_chunk(b"hello", &mut out);
write_last_chunk(&mut out);
assert_eq!(out, b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nhello\r\n0\r\n\r\n");
```

## The server

`serve(lattice, io, sink, listener, config, handler, timers)` takes the lattice, a bound listener, an `AsyncIo`, a
`QueueIngestSink` and a `Completions` (the last three attached to the lattice by the caller), and a `Handler`. It
attaches the listener as a source of the lattice and returns a `ServerHandle`; each accepted connection is a source of
its own, attached on accept and detached on close. The crate keeps no thread and no timer of its own: every step is a
frame, and every time limit is the limit of a socket operation (the back-off after a failed accept is one of
`timers`).

```no_run
use std::net::TcpListener;
use std::sync::Arc;
use euhedral_core::control_plane::ControlPlaneLattice;
use euhedral_core::frames::Frame;
use euhedral_core::ingest::QueueIngestSink;
use euhedral_graph::Completions;
use euhedral_io::AsyncIo;
use euhedral_net::{Request, Response, Responder, ServerConfig, no_work, serve};

fn start(
    lattice: Arc<ControlPlaneLattice>,
    io: Arc<AsyncIo>,
    sink: Arc<QueueIngestSink>,
    timers: Arc<Completions>,
) -> std::io::Result<()> {
    // `io`, `sink.delegate()` and `timers` are attached to the lattice as upstreams.
    let listener = TcpListener::bind("127.0.0.1:8080")?;
    let handler = |request: Request, reply: Responder| -> Box<dyn Frame> {
        // The reply may be sent here, or from the returned frame, or from any later frame on any worker.
        reply.send(Response::with_status(200).with_body(request.target.into_bytes()));
        no_work()
    };
    let server = serve(lattice, io, sink, listener, ServerConfig::default(), Arc::new(handler), timers)?;
    println!("{:?}", server.counters());
    // Later, from anywhere: stop accepting, let requests in flight finish, and be told when it is done.
    server.stop(no_work());
    Ok(())
}
```

- **One reply per request.** `Responder::send` is the reply; a `Responder` dropped without it answers 500. The server
  chooses `Connection`; a HEAD request gets the head of the response only. `Responder::cancelled()` is a flag the
  server sets when the peer ends its stream or breaks the connection while the request is being handled.
  `Responder::on_cancel(frame)` asks for a frame to be offered to the sink when that happens, so a handler need not
  look at the flag: it runs at most once, and is dropped without running if the reply is sent first (the two race, so
  it may also run just after a reply that was still sent, after a half-close). Before a cancel a later call replaces
  the earlier frame; after it, every frame asked for while the reply is still owed runs at once.
- **Streamed replies.** `Responder::stream(head, then)` answers with a reply whose body is sent piece by piece, for a
  producer that has the body a part at a time (generated text, a long export). `head` has the status, the headers and
  a body that, if non-empty, goes out as the first piece in the same write as the head. `then` runs once the head is
  written, with a `ReplyStream` (or a `StreamError` if the stream did not start); `ReplyStream::send(piece, then)`
  writes a piece and runs `then` with a `PieceSent`: the stream back, and the piece's `Vec`, emptied, with its capacity
  kept, to fill again; `ReplyStream::end()` ends the body. The framing is the server's, by the peer's version: HTTP/1.1
  gets the chunked coding (`Transfer-Encoding: chunked`, a last chunk at the end); HTTP/1.0 gets no length and
  `Connection: close`, and the close ends the body (so a 1.0 peer cannot tell a cut stream from a finished one); the
  answer to HEAD is the head alone, with neither `Content-Length` nor `Transfer-Encoding`, and its pieces are dropped
  (reported written at once). A head the writer refuses, or a 1xx, 204 or 304, is answered `500` and `then` gets
  `StreamError::Head`.
  - *One piece at a time.* `send` consumes the stream and it comes back only in `PieceSent`, so a second piece cannot
    be sent while one is in flight: this is checked by the compiler (a doctest of `ReplyStream` shows it). The pace
    is the producer's: the server holds no queue of pieces. An empty piece writes nothing and does not end the body.
    A producer that wants fewer, larger writes joins its pieces before sending.
  - *Time.* `piece_limit` (default 30 s) bounds each write of the head, a piece and the end; a write that passes it
    ends the stream with `StreamError::TimedOut` and closes the connection. There is no limit on the gap between two
    pieces.
  - *Cancel.* The watch `recv` stays pending during the stream, and `ReplyStream::cancelled()` and `on_cancel(frame)`
    behave as the `Responder`'s do, for the whole stream: a half-close sets the flag and the stream goes on (the
    connection closes after it); a broken peer sets the flag, closes the connection and ends the piece in flight (and
    every later piece) with `StreamError::Closed`. Once pipelined bytes are held no `recv` watches the peer, and a
    departure shows on a send.
  - *The end.* `end()` writes the last chunk and the connection then serves its next request (or closes, if the
    request said `Connection: close`, the peer half-closed or the server is stopping). A `ReplyStream` dropped
    without `end` cuts the stream: the connection closes with no last chunk, so an HTTP/1.1 peer sees a truncated
    body; it is counted in `streams_cut`, as is a write that failed or passed `piece_limit` and a peer that left. A
    pipelined request behind a stream is answered after its end.
  - *Stop.* A stream runs to its end, and the connection closes after it; the stop is not reported while a
    `ReplyStream` is held or a piece is in flight, so the program ends its streams when it stops (as it answers its
    requests). The stop does not set the cancel flag.
  - *Where `then` runs.* Inside the connection's own frame, under `catch_unwind`: it must return quickly. The frame
    it returns is offered to the sink; a `then` that panics drops the stream it was given, which cuts it.
- **One source per connection.** A connection's parser, buffers and socket are plain state that only its key frame
  touches (see the glossary: station). Requests on one connection are answered in the order they arrived, one at a
  time; pipelined bytes wait in the buffer. Different connections run in parallel.
  While pipelined bytes are buffered when a request is handed to the handler, no read watches the peer, so a cancel
  is seen only when the reply is sent (a reply to a peer that has gone is then discarded).
- **`Expect: 100-continue`** is answered by the server with `100 Continue` once the head is accepted (not too large,
  well formed); the handler is not consulted and cannot refuse it. A body over `max_body_bytes` is answered 413 at
  the head, before it is read.
- **Limits.** `read_limit` covers a request from its first byte, `idle_limit` the wait for the first byte of a
  request, `write_limit` each reply, `piece_limit` each write of a streamed reply (head, piece, end). A passed read limit after the first byte is answered 408 and closes; any
  other passed limit closes without a reply. `None` is no limit.
- **Half-close.** A peer that ends its stream while its request is handled sets the cancel flag as a hint, but the
  reply is still sent (with `Connection: close`) and the connection then closes; a reply is discarded only if its
  send fails or the read ended in an error or a reset.
- **Errors close.** A parse error answers its status with `Connection: close`, then the server shuts its writing
  side and reads (and discards, up to the size limits) until the peer closes, so that the answer is not lost to a
  reset. The linger ends at one deadline from its start, `linger_limit` (`stop_linger_limit` if shorter while the
  server stops), whatever the peer sends and whatever `write_limit` is.
- **Accept errors** are counted (`accept_errors`) and the accept is tried again after `accept_backoff`.
- **Cap.** At `max_connections` no accept is pending; the kernel's backlog holds the rest.
- **Stop.** Waiting connections close (also one that waits for a body after `100 Continue`, and a request that
  completes after the stop is not handed to the handler); a request in flight is answered, with `Connection:
  close`. The frame given to `stop` is offered to the sink as the last thing the server does: when every connection
  is gone and its source has ended, no accept is pending and every `Responder` has been used or dropped, and after
  the server let go of the listener socket, the handler, the `AsyncIo`, the timers and the lattice. Only the sink is
  kept, to report a stop, until the last `ServerHandle` is dropped. The listener's source stays attached to the
  lattice until then too, so that a stop asked for at any time is heard.
- **The lattice and the `AsyncIo`** are held weakly: keep them alive while the server runs, and stop the server
  before dropping either (a server abandoned without a stop leaks through no cycle, but nothing runs its frames).
  A lattice that is gone or closed when a connection arrives refuses the connection and ends the server as a stop
  does.
- **Cost of a source per connection.** Each connection is one allocation for its source and its state, plus the
  locks and atomics of attaching it to the lattice and detaching it. That matters only for very short-lived
  connections arriving at a very high rate; the server is designed for long-lived connections, where it is paid once
  and amortised over many requests. It is not free.
- **Allocated.** Per connection: the connection record (one `Arc`), a parser, a buffer for unparsed bytes and one
  read buffer of `read_chunk` bytes, kept between reads. Per request: the `Request` the parser builds, the
  `Responder`, the response and its serialised bytes, and the frame each I/O completion returns. A streamed reply
  adds, per piece, the continuation's box and the frame it returns, and the continuation and frame of the send; the
  piece's bytes are copied once into the stream's send buffer (kept for the whole stream, since `AsyncIo` has no
  vectored send), and the caller's `Vec` comes back for reuse.

## The client

`request(io, sink, to, &request, config, then)` makes one request to a socket address and calls `then` with the
`Response` or a `NetError`, exactly once, on every path. It returns at once and waits for nothing: the call is a
chain of frames. One connection per call: connect, write the request with `Connection: close`, read one response,
close (the socket is closed before `then` runs). The frame `then` returns is offered to `sink`. `io` is held
weakly: keep it alive while calls are in flight. Not offered: a pool, keep-alive, redirects, TLS, chunked bodies,
DNS (the address is a socket address), retries, and a cancel of a call in flight (it would need the call's
operation id shared with whoever cancels, which the plain state of a call does not have; the time limit is the
way to end a call).

```no_run
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use euhedral_core::frames::Frame;
use euhedral_core::ingest::QueueIngestSink;
use euhedral_io::AsyncIo;
use euhedral_net::{ClientConfig, Method, Request, request, no_work};

fn fetch(io: &Arc<AsyncIo>, sink: &Arc<QueueIngestSink>, to: SocketAddr) {
    let get = Request::new(Method::Get, "/status").header("Host", "service");
    let config = ClientConfig {
        time_limit: Some(Duration::from_secs(5)), // the whole call: connect, send and receive
        ..ClientConfig::default()
    };
    request(io, sink, to, &get, config, Box::new(|outcome| -> Box<dyn Frame> {
        match outcome {
            Ok(response) => println!("{} {} bytes", response.status, response.body.len()),
            Err(error) => println!("failed: {error}"),
        }
        no_work() // or the frame that does the next work
    }));
}
```

- **One deadline.** `time_limit` runs from the call of `request`. Each operation is issued with what is left of it
  (`recv_with`, `send_with`, `connect_with` limits); when none is left before an operation starts, the call ends
  without issuing it. The error names the stage: `NetError::TimedOut(CallStage)`, with `Connect`, `Send`,
  `ReceiveHead` or `ReceiveBody`. The crate has no timer of its own.
- **Errors.** `Connect(io::Error)` (refused, unreachable, no socket), `TimedOut(stage)`, `Io { stage, error }` (a reset,
  a broken pipe), `PeerClosed(stage)` (the peer closed before a whole response), `Malformed(ParseError)`,
  `Unsupported(ParseError)` (a response that needs `Content-Length` and has none, or has `Transfer-Encoding`:
  read-to-close and chunked bodies are refused, as the parser refuses them), `OverLimit(ParseError)` (status line,
  header bytes, header count or body over the `Limits` in the config), `UnexpectedInterim(status)`,
  `Write(WriteError)` (the request cannot be written; nothing was sent) and `Abandoned(stage)` (the engine refused to
  start an operation, was dropped with one in flight, or the call's frame was discarded; the engine drops the
  continuation before it returns its error, so the real error is not available).
  A peer that answers early and closes while the request is still being sent is reported as the send's error
  (`Io` in `Send`), and its response is lost. `NetError` and `CallStage` are not `#[non_exhaustive]`, as the
  crate's other public enums (`ParseError`, `WriteError`) are not; a `time_limit` too long to add to the clock
  is no limit.
- **HEAD** reads no body (`set_response_to_head`): the response has the headers, with the length the body would
  have, and an empty body. A response to another method that has a status without a body (204, 304) has none.
- **Interim responses.** A 1xx response is skipped, up to `MAX_INTERIM_RESPONSES` (8); a `101` is refused, since
  the connection would change protocol, and so is a ninth interim: `UnexpectedInterim(status)`. Bytes after the final
  response are ignored (the connection closes).
- **State.** A call's state (the socket, the parser, the buffers, the deadline, `then`) is owned by one place at a
  time: the operation in flight, or the step frame its continuation returned, which moves it into the next
  operation. No lock, atomic, cell or source is involved.
- **Cost per call.** One boxed state, one socket, the serialised request and a read buffer (`read_chunk`, 16 KiB
  by default, kept between reads), the parsed response, and for each of the (connect, send, one or more receives)
  operations a continuation and the step frame it returns. No source is attached to the lattice and no lock is
  taken by the crate: the cost is what the engine's operations cost, and the few allocations above. A call is
  one connection, so a program that makes many calls to one peer pays the handshake each time.
