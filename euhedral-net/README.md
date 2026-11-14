# euhedral-net

HTTP/1.1 for programs built on RustEE. The crate will be an HTTP/1.1 server and a minimal client over
[`euhedral-io`](../euhedral-io/README.md)'s `AsyncIo`: a request line and headers, bodies with `Content-Length`,
keep-alive, `Expect: 100-continue`, size limits, time limits and a connection cap. Out of scope for the crate:
TLS, HTTP/2, chunked request bodies (answered 411), compression, routing and JSON.

**This part.** The messages, a parser and a writer, with no I/O: no socket, no frame, no timer, no thread, no
dependency. The server and the client come in later changes and are built on exactly this. See the
[top-level glossary](../GLOSSARY.md) for the terms a user meets and [`GLOSSARY.md`](GLOSSARY.md) for this crate's.

| Item | What it is |
|---|---|
| [`Request`, `Response`, `Headers`, `Method`, `Version`](src/message.rs) | The messages. `Headers` keeps order and repeated names and finds a name without regard to case. |
| [`ParseError`, `WriteError`](src/message.rs) | Why input is refused. `ParseError::status()` is the status a server answers: 400, 411, 413, 414, 431 or 505, and `None` for an error that has no status to send (a response the client cannot read). |
| [`RequestParser`, `ResponseParser`, `Limits`, `Progress`, `State`, `PendingHead`](src/parse.rs) | The strict, incremental parsers and their size limits. |
| [`write_response`, `write_response_head`, `write_request`, `write_continue`, `ConnectionMode`](src/write.rs) | The writer. |

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
without a body (1xx, 204, 304); a response with `Transfer-Encoding`, or one that ends with the connection, is an
error the client reports.

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
