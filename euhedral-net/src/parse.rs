//! The incremental, strict parsers: [`RequestParser`] for a server and [`ResponseParser`] for a client.
//!
//! Both are pure state machines fed with bytes. They read no socket, own no thread and wait for nothing. A
//! parser's state belongs to one consumer, the owner of the connection, and needs no synchronisation.
//!
//! # Feeding
//!
//! `feed(input)` consumes the bytes of the message being read and no more, and returns how many it consumed with
//! what happened. The caller keeps the rest of its slice for the next call: that is how pipelined requests are
//! kept, and why a parser never buffers more than one message and never more than its limits allow. A slice may
//! be split anywhere (down to one byte per call) and the result is the same.
//!
//! ```text
//! let (used, progress) = parser.feed(bytes)?;
//! bytes = &bytes[used..];
//! match progress { NeedMore => .., HeadComplete { .. } => .., Complete(message) => .. }
//! ```
//!
//! After an error the parser stays failed: every later call returns the same error and consumes nothing. The
//! connection is closed after the answer to the error.
//!
//! # What is refused, and why
//!
//! A server that is lenient about where a message ends can be made to disagree with the proxy in front of it
//! about where the next request starts (request smuggling, RFC 9112 section 11.2). So the parsers are strict
//! where the RFC allows a recipient to be lenient, and refuse:
//!
//! - a bare LF, or a CR that is not followed by LF, as a line ending (RFC 9112 section 2.2 lets a recipient accept
//!   a bare LF; this crate does not). Empty lines (CRLF) before a request line are ignored, as section 2.2
//!   requires, up to [`MAX_LEADING_EMPTY_LINES`];
//! - obsolete line folding (section 5.2) and whitespace between a field name and its colon (section 5.1);
//! - a field value with a control character or a byte above 0x7e (the obsolete 0x80 to 0xff text is refused, so
//!   values are always ASCII);
//! - `Content-Length` values that differ, or that are not digits (section 6.3), and `Transfer-Encoding` together
//!   with `Content-Length`;
//! - a request line with more than single spaces between its parts, or a lowercase or malformed version, and any
//!   version but 1.0 and 1.1 (HTTP/1.2 is 505 too: this crate speaks the two it knows);
//! - a request with more than one `Host` line, on any version, an HTTP/1.1 request without one (section 3.2), and
//!   a `Host` that is not a host and an optional port (a registered name or IPv4 address, or a bracketed IP
//!   literal; stricter than RFC 3986, which also lets a name hold `,` `;` and the like). An empty value is allowed;
//! - a request target that is not one of the four forms (section 3.2): origin-form begins with `/`, asterisk-form
//!   `*` only with OPTIONS, authority-form `host:port` only with CONNECT, absolute-form a scheme, `://` and an
//!   authority;
//! - a status line with no space after the status code, even when the reason is empty (`HTTP/1.1 200` is refused;
//!   `HTTP/1.1 200 ` is the form with an empty reason).
//!
//! # Limits of the crate
//!
//! A request with `Transfer-Encoding` is refused. Where `chunked` is not the final coding (or is the only one
//! missing) the length of the body cannot be told: 400 (section 6.3). Where `chunked` is the final coding the
//! request is well formed but this crate does not decode a chunked request body: 411. A response must have a
//! `Content-Length`, unless it answers a HEAD request or has a status without a body (1xx, 204, 304): a response
//! with `Transfer-Encoding`, or one that ends with the connection, is an error. An `Expect` other than
//! `100-continue` is ignored. The client side is minimal by design.

use std::mem::{replace, take};

use crate::message::{
    Headers, Method, ParseError, Request, Response, Version, status_has_no_body, target_form,
    valid_host_value,
};

/// How many empty lines before a request line are skipped before the input is refused.
pub const MAX_LEADING_EMPTY_LINES: usize = 4;

/// The size limits of a parser. Each is enforced before the bytes past it are buffered.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Limits {
    /// The longest request line or status line, with its CRLF. Longer is 414.
    pub max_start_line: usize,
    /// The most bytes of all header lines together, each with its CRLF and the empty line that ends them. More
    /// is 431.
    pub max_header_bytes: usize,
    /// The most header lines. More is 431.
    pub max_headers: usize,
    /// The longest body. A `Content-Length` over it is 413 as soon as the headers are read.
    pub max_body: usize,
}

impl Default for Limits {
    fn default() -> Limits {
        Limits {
            max_start_line: 8 * 1024,
            max_header_bytes: 16 * 1024,
            max_headers: 100,
            max_body: 1024 * 1024,
        }
    }
}

/// What a call to `feed` found.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Progress<T> {
    /// All the input was consumed and the message is not complete.
    NeedMore,
    /// The header lines are complete and a body is pending: the caller may answer `100 Continue` now when
    /// `expect_continue` is set. Given once per message that has a body; a message without a body goes straight
    /// to `Complete`.
    HeadComplete {
        /// The request has `Expect: 100-continue` (HTTP/1.1 only) and a body to wait for.
        expect_continue: bool,
        /// The body's length.
        content_length: usize,
    },
    /// A whole message. The parser is ready for the next one.
    Complete(T),
}

/// Where a parser is, for a caller that wants to look between calls.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum State {
    /// Reading a start line or header lines.
    Head,
    /// The head is complete and `remaining` body bytes are pending.
    Body {
        expect_continue: bool,
        remaining: usize,
    },
    /// An error ended the parser.
    Failed(ParseError),
}

fn is_tchar(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b"!#$%&'*+-.^_`|~".contains(&b)
}

fn is_ows(b: u8) -> bool {
    b == b' ' || b == b'\t'
}

fn is_value_byte(b: u8) -> bool {
    b == b'\t' || (0x20..=0x7e).contains(&b)
}

fn trim_ows(mut s: &[u8]) -> &[u8] {
    while let [first, rest @ ..] = s
        && is_ows(*first)
    {
        s = rest;
    }
    while let [rest @ .., last] = s
        && is_ows(*last)
    {
        s = rest;
    }
    s
}

fn ascii(bytes: &[u8], error: ParseError) -> Result<String, ParseError> {
    std::str::from_utf8(bytes)
        .map(str::to_owned)
        .map_err(|_| error)
}

/// A version token: `Ok` for 1.0 and 1.1, 505 for another well-formed version, 400 for anything else.
fn parse_version(token: &[u8]) -> Result<Version, ParseError> {
    match token {
        b"HTTP/1.1" => Ok(Version::Http11),
        b"HTTP/1.0" => Ok(Version::Http10),
        [b'H', b'T', b'T', b'P', b'/', major, b'.', minor]
            if major.is_ascii_digit() && minor.is_ascii_digit() =>
        {
            Err(ParseError::UnsupportedVersion)
        }
        _ => Err(ParseError::BadVersion),
    }
}

fn parse_header_line(content: &[u8]) -> Result<(String, String), ParseError> {
    if content.first().copied().is_some_and(is_ows) {
        return Err(ParseError::ObsoleteFolding);
    }
    let colon = content
        .iter()
        .position(|&b| b == b':')
        .ok_or(ParseError::BadHeaderLine)?;
    let (name, rest) = content.split_at(colon);
    if name.is_empty() {
        return Err(ParseError::BadHeaderLine);
    }
    if name.last().copied().is_some_and(is_ows) {
        return Err(ParseError::SpaceBeforeColon);
    }
    if !name.iter().all(|&b| is_tchar(b)) {
        return Err(ParseError::BadHeaderName);
    }
    let value = trim_ows(rest.get(1..).unwrap_or_default());
    if !value.iter().all(|&b| is_value_byte(b)) {
        return Err(ParseError::BadHeaderValue);
    }
    Ok((
        ascii(name, ParseError::BadHeaderName)?,
        ascii(value, ParseError::BadHeaderValue)?,
    ))
}

/// Reads the start line and the header lines, line by line, within the limits.
#[derive(Debug)]
struct HeadReader {
    limits: Limits,
    /// Skip empty lines before the start line (requests only).
    skip_leading: bool,
    /// The error for a start line over its limit.
    start_too_long: ParseError,
    /// The error for a malformed start line without a better answer.
    bad_start: ParseError,
    /// The current partial line, with its CRLF when complete.
    line: Vec<u8>,
    start: Option<Vec<u8>>,
    headers: Headers,
    header_bytes: usize,
    leading: usize,
}

impl HeadReader {
    fn new(
        limits: Limits,
        skip_leading: bool,
        start_too_long: ParseError,
        bad_start: ParseError,
    ) -> HeadReader {
        HeadReader {
            limits,
            skip_leading,
            start_too_long,
            bad_start,
            line: Vec::new(),
            start: None,
            headers: Headers::new(),
            header_bytes: 0,
            leading: 0,
        }
    }

    /// The bytes of the head held: the partial line, the start line with its CRLF and the header lines.
    fn buffered(&self) -> usize {
        self.line.len() + self.start.as_ref().map_or(0, |s| s.len() + 2) + self.header_bytes
    }

    /// Consumes input up to the end of the head. Returns the bytes consumed and whether the head is complete.
    fn feed(&mut self, input: &[u8]) -> Result<(usize, bool), ParseError> {
        let mut used = 0;
        while let Some(rest) = input.get(used..).filter(|r| !r.is_empty()) {
            let (budget, too_long) = if self.start.is_none() {
                (self.limits.max_start_line, self.start_too_long)
            } else {
                (
                    self.limits
                        .max_header_bytes
                        .saturating_sub(self.header_bytes),
                    ParseError::HeadersTooLarge,
                )
            };
            let lf = rest.iter().position(|&b| b == b'\n');
            let chunk = lf.map_or(rest.len(), |p| p + 1);
            if self.line.len().saturating_add(chunk) > budget {
                return Err(too_long);
            }
            self.line.extend_from_slice(&rest[..chunk]);
            used += chunk;
            if lf.is_none() {
                break;
            }
            let line = take(&mut self.line);
            let Some(content) = line.strip_suffix(b"\r\n") else {
                return Err(ParseError::BadLineEnding);
            };
            if content.contains(&b'\r') {
                return Err(ParseError::BadLineEnding);
            }
            if self.start.is_none() {
                if content.is_empty() {
                    self.leading += 1;
                    if !self.skip_leading || self.leading > MAX_LEADING_EMPTY_LINES {
                        return Err(self.bad_start);
                    }
                } else {
                    self.start = Some(content.to_vec());
                }
            } else {
                self.header_bytes += line.len();
                if content.is_empty() {
                    return Ok((used, true));
                }
                if self.headers.len() >= self.limits.max_headers {
                    return Err(ParseError::TooManyHeaders);
                }
                let (name, value) = parse_header_line(content)?;
                self.headers.append(name, value);
            }
            // Keep the allocation for the next line.
            let mut line = line;
            line.clear();
            self.line = line;
        }
        Ok((used, false))
    }

    /// The start line, the headers and the bytes of the head, and the reader ready for the next one.
    fn take(&mut self) -> (Vec<u8>, Headers, usize) {
        let head_bytes = self.buffered();
        let start = self.start.take().unwrap_or_default();
        let headers = take(&mut self.headers);
        self.line.clear();
        self.header_bytes = 0;
        self.leading = 0;
        (start, headers, head_bytes)
    }

    fn reset(&mut self) {
        self.take();
        self.line = Vec::new();
    }
}

/// The `Content-Length` of a header block: `None` when there is none; every value of every line must be digits
/// that fit in 64 bits and all must be equal.
fn content_length(headers: &Headers) -> Result<Option<u64>, ParseError> {
    let mut found: Option<u64> = None;
    for value in headers.get_all("Content-Length") {
        for element in value.split(',') {
            let digits = trim_ows(element.as_bytes());
            if digits.is_empty() || !digits.iter().all(u8::is_ascii_digit) {
                return Err(ParseError::BadContentLength);
            }
            let mut n: u64 = 0;
            for &d in digits {
                n = n
                    .checked_mul(10)
                    .and_then(|n| n.checked_add(u64::from(d - b'0')))
                    .ok_or(ParseError::BadContentLength)?;
            }
            match found {
                Some(previous) if previous != n => {
                    return Err(ParseError::ConflictingContentLength);
                }
                _ => found = Some(n),
            }
        }
    }
    Ok(found)
}

fn body_length(n: u64, limits: &Limits) -> Result<usize, ParseError> {
    usize::try_from(n)
        .ok()
        .filter(|&n| n <= limits.max_body)
        .ok_or(ParseError::BodyTooLarge)
}

/// Reads body bytes into `body`, up to `remaining`. Returns the bytes consumed.
fn take_body(input: &[u8], remaining: &mut usize, body: &mut Vec<u8>) -> usize {
    let n = input.len().min(*remaining);
    body.extend_from_slice(&input[..n]);
    *remaining -= n;
    n
}

#[derive(Debug)]
enum Stage<T> {
    Head,
    Body {
        message: T,
        /// The bytes of the head, which the parser holds in `message`.
        head_bytes: usize,
        expect_continue: bool,
        remaining: usize,
    },
    Failed(ParseError),
}

/// The head of a request whose body has not all arrived.
#[derive(Clone, Copy, Debug)]
pub struct PendingHead<'a> {
    pub method: &'a Method,
    pub target: &'a str,
    pub version: Version,
    pub headers: &'a Headers,
}

/// Reads requests, for a server.
#[derive(Debug)]
pub struct RequestParser {
    reader: HeadReader,
    stage: Stage<Request>,
}

impl RequestParser {
    /// A parser at the start of a connection.
    pub fn new(limits: Limits) -> RequestParser {
        RequestParser {
            reader: HeadReader::new(
                limits,
                true,
                ParseError::RequestLineTooLong,
                ParseError::BadRequestLine,
            ),
            stage: Stage::Head,
        }
    }

    /// Where the parser is.
    pub fn state(&self) -> State {
        match &self.stage {
            Stage::Head => State::Head,
            Stage::Body {
                expect_continue,
                remaining,
                ..
            } => State::Body {
                expect_continue: *expect_continue,
                remaining: *remaining,
            },
            Stage::Failed(e) => State::Failed(*e),
        }
    }

    /// Every byte the parser holds for the message being read: the partial line, the start line and the header
    /// lines (with their CRLFs, also while a body is pending) and the body so far. Leading empty lines are not
    /// held. At most the limits' sum, and equal to the bytes of the message fed so far until it completes.
    pub fn buffered(&self) -> usize {
        self.reader.buffered()
            + match &self.stage {
                Stage::Body {
                    message,
                    head_bytes,
                    ..
                } => head_bytes + message.body.len(),
                _ => 0,
            }
    }

    /// The head of the request whose body is pending (after `HeadComplete`, before `Complete`), borrowed: a server
    /// looks at the target and headers to choose between `100 Continue` and an early final status.
    pub fn pending_head(&self) -> Option<PendingHead<'_>> {
        match &self.stage {
            Stage::Body { message, .. } => Some(PendingHead {
                method: &message.method,
                target: &message.target,
                version: message.version,
                headers: &message.headers,
            }),
            _ => None,
        }
    }

    /// Consumes bytes of the current request; see the module's notes on feeding.
    pub fn feed(&mut self, input: &[u8]) -> Result<(usize, Progress<Request>), ParseError> {
        let result = self.step(input);
        if let Err(e) = result {
            self.stage = Stage::Failed(e);
            self.reader.reset();
        }
        result
    }

    fn step(&mut self, input: &[u8]) -> Result<(usize, Progress<Request>), ParseError> {
        match &mut self.stage {
            Stage::Failed(e) => Err(*e),
            Stage::Head => {
                let (used, done) = self.reader.feed(input)?;
                if !done {
                    return Ok((used, Progress::NeedMore));
                }
                let (start, headers, head_bytes) = self.reader.take();
                let (request, length) = request_head(&start, headers, &self.reader.limits)?;
                if length == 0 {
                    return Ok((used, Progress::Complete(request)));
                }
                let expect_continue = request.expects_continue();
                self.stage = Stage::Body {
                    message: request,
                    head_bytes,
                    expect_continue,
                    remaining: length,
                };
                Ok((
                    used,
                    Progress::HeadComplete {
                        expect_continue,
                        content_length: length,
                    },
                ))
            }
            Stage::Body {
                message, remaining, ..
            } => {
                let used = take_body(input, remaining, &mut message.body);
                if *remaining > 0 {
                    return Ok((used, Progress::NeedMore));
                }
                let Stage::Body { message, .. } = replace(&mut self.stage, Stage::Head) else {
                    return Err(ParseError::BadRequestLine);
                };
                Ok((used, Progress::Complete(message)))
            }
        }
    }
}

/// Whether the last coding of the last `Transfer-Encoding` line is `chunked` (RFC 9112 section 6.3).
fn final_coding_is_chunked(headers: &Headers) -> bool {
    headers
        .get_all("Transfer-Encoding")
        .last()
        .and_then(|value| value.split(',').next_back())
        .is_some_and(|coding| coding.trim().eq_ignore_ascii_case("chunked"))
}

/// The request a start line and headers make, and the length of its body.
fn request_head(
    start: &[u8],
    headers: Headers,
    limits: &Limits,
) -> Result<(Request, usize), ParseError> {
    let mut parts = start.split(|&b| b == b' ');
    let (Some(method), Some(target), Some(version), None) =
        (parts.next(), parts.next(), parts.next(), parts.next())
    else {
        return Err(ParseError::BadRequestLine);
    };
    if method.is_empty() || target.is_empty() || version.is_empty() {
        return Err(ParseError::BadRequestLine);
    }
    if !method.iter().all(|&b| is_tchar(b)) {
        return Err(ParseError::BadMethod);
    }
    if !target
        .iter()
        .all(|&b| (0x21..=0x7e).contains(&b) && b != b'#')
    {
        return Err(ParseError::BadTarget);
    }
    let version = parse_version(version)?;
    let method = Method::from_token(&ascii(method, ParseError::BadMethod)?);
    let target = ascii(target, ParseError::BadTarget)?;
    if target_form(&method, &target).is_none() {
        return Err(ParseError::BadTarget);
    }
    {
        let mut hosts = headers.get_all("Host");
        match (hosts.next(), hosts.next()) {
            (None, _) if version == Version::Http10 => {}
            (Some(host), None) if valid_host_value(host) => {}
            _ => return Err(ParseError::BadHost),
        }
    }
    let length = content_length(&headers)?;
    if headers.contains("Transfer-Encoding") {
        return Err(if length.is_some() {
            ParseError::TransferEncodingWithContentLength
        } else if final_coding_is_chunked(&headers) {
            ParseError::ChunkedNotSupported
        } else {
            ParseError::BadTransferEncoding
        });
    }
    let length = body_length(length.unwrap_or(0), limits)?;
    let request = Request {
        method,
        target,
        version,
        headers,
        body: Vec::new(),
    };
    Ok((request, length))
}

/// Reads responses, for a client.
#[derive(Debug)]
pub struct ResponseParser {
    reader: HeadReader,
    stage: Stage<Response>,
    to_head: bool,
}

impl ResponseParser {
    /// A parser at the start of a connection.
    pub fn new(limits: Limits) -> ResponseParser {
        ResponseParser {
            reader: HeadReader::new(
                limits,
                false,
                ParseError::StatusLineTooLong,
                ParseError::BadStatusLine,
            ),
            stage: Stage::Head,
            to_head: false,
        }
    }

    /// Says whether the response being read answers a HEAD request, which has no body whatever its headers say.
    /// It is one-shot: it applies to the next final response (a 1xx before it keeps it) and is cleared when that
    /// one completes, so set it before each response that answers a HEAD request.
    pub fn set_response_to_head(&mut self, to_head: bool) {
        self.to_head = to_head;
    }

    /// Where the parser is.
    pub fn state(&self) -> State {
        match &self.stage {
            Stage::Head => State::Head,
            Stage::Body { remaining, .. } => State::Body {
                expect_continue: false,
                remaining: *remaining,
            },
            Stage::Failed(e) => State::Failed(*e),
        }
    }

    /// Every byte the parser holds for the message being read, as for [`RequestParser::buffered`].
    pub fn buffered(&self) -> usize {
        self.reader.buffered()
            + match &self.stage {
                Stage::Body {
                    message,
                    head_bytes,
                    ..
                } => head_bytes + message.body.len(),
                _ => 0,
            }
    }

    /// Consumes bytes of the current response; see the module's notes on feeding. A response with a 1xx status
    /// is returned as a response with no body: the caller reads on for the final one.
    pub fn feed(&mut self, input: &[u8]) -> Result<(usize, Progress<Response>), ParseError> {
        let result = self.step(input);
        if let Err(e) = result {
            self.stage = Stage::Failed(e);
            self.reader.reset();
        }
        result
    }

    fn step(&mut self, input: &[u8]) -> Result<(usize, Progress<Response>), ParseError> {
        match &mut self.stage {
            Stage::Failed(e) => Err(*e),
            Stage::Head => {
                let (used, done) = self.reader.feed(input)?;
                if !done {
                    return Ok((used, Progress::NeedMore));
                }
                let (start, headers, head_bytes) = self.reader.take();
                let (response, length) =
                    response_head(&start, headers, self.to_head, &self.reader.limits)?;
                if length == 0 {
                    if response.status >= 200 {
                        self.to_head = false;
                    }
                    return Ok((used, Progress::Complete(response)));
                }
                self.stage = Stage::Body {
                    message: response,
                    head_bytes,
                    expect_continue: false,
                    remaining: length,
                };
                Ok((
                    used,
                    Progress::HeadComplete {
                        expect_continue: false,
                        content_length: length,
                    },
                ))
            }
            Stage::Body {
                message, remaining, ..
            } => {
                let used = take_body(input, remaining, &mut message.body);
                if *remaining > 0 {
                    return Ok((used, Progress::NeedMore));
                }
                let Stage::Body { message, .. } = replace(&mut self.stage, Stage::Head) else {
                    return Err(ParseError::BadStatusLine);
                };
                Ok((used, Progress::Complete(message)))
            }
        }
    }
}

fn response_head(
    start: &[u8],
    headers: Headers,
    to_head: bool,
    limits: &Limits,
) -> Result<(Response, usize), ParseError> {
    let bad = ParseError::BadStatusLine;
    let space = start.iter().position(|&b| b == b' ').ok_or(bad)?;
    let (version, rest) = start.split_at(space);
    let version = parse_version(version).map_err(|e| match e {
        ParseError::BadVersion => bad,
        other => other,
    })?;
    // rest is ` ddd ` and the reason.
    let [b' ', d1, d2, d3, b' ', reason @ ..] = rest else {
        return Err(bad);
    };
    if ![d1, d2, d3].iter().all(|d| d.is_ascii_digit()) || !(b'1'..=b'5').contains(d1) {
        return Err(bad);
    }
    let status = u16::from(d1 - b'0') * 100 + u16::from(d2 - b'0') * 10 + u16::from(d3 - b'0');
    if !reason.iter().all(|&b| is_value_byte(b)) {
        return Err(bad);
    }
    let response = Response {
        version,
        status,
        reason: ascii(reason, bad)?,
        headers,
        body: Vec::new(),
    };
    if to_head || status_has_no_body(status) {
        return Ok((response, 0));
    }
    let length = content_length(&response.headers)?;
    if response.headers.contains("Transfer-Encoding") {
        return Err(if length.is_some() {
            ParseError::TransferEncodingWithContentLength
        } else {
            ParseError::TransferEncodingNotSupported
        });
    }
    let length = length.ok_or(ParseError::MissingContentLength)?;
    let length = body_length(length, limits)?;
    Ok((response, length))
}
