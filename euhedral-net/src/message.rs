//! The HTTP/1.1 message types: `Request`, `Response`, `Headers`, `Method`, `Version` and the errors.

use std::fmt;

/// The request method. A method the crate knows by name is one of the named variants; any other token is `Other`.
/// The parser only builds `Other` for a token that is not one of the named methods.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Method {
    Get,
    Head,
    Post,
    Put,
    Delete,
    Connect,
    Options,
    Trace,
    Patch,
    /// Any other method token (methods are case-sensitive).
    Other(String),
}

impl Method {
    /// The method for a token: a named variant when the token is that method's name, else `Other`.
    pub fn from_token(token: &str) -> Method {
        match token {
            "GET" => Method::Get,
            "HEAD" => Method::Head,
            "POST" => Method::Post,
            "PUT" => Method::Put,
            "DELETE" => Method::Delete,
            "CONNECT" => Method::Connect,
            "OPTIONS" => Method::Options,
            "TRACE" => Method::Trace,
            "PATCH" => Method::Patch,
            other => Method::Other(other.to_owned()),
        }
    }

    /// The method as it is written on the wire.
    pub fn as_str(&self) -> &str {
        match self {
            Method::Get => "GET",
            Method::Head => "HEAD",
            Method::Post => "POST",
            Method::Put => "PUT",
            Method::Delete => "DELETE",
            Method::Connect => "CONNECT",
            Method::Options => "OPTIONS",
            Method::Trace => "TRACE",
            Method::Patch => "PATCH",
            Method::Other(token) => token,
        }
    }
}

/// The protocol version. Only 1.0 and 1.1 are spoken.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Version {
    Http10,
    Http11,
}

impl Version {
    /// The version as it is written on the wire.
    pub fn as_str(self) -> &'static str {
        match self {
            Version::Http10 => "HTTP/1.0",
            Version::Http11 => "HTTP/1.1",
        }
    }
}

/// The header fields of a message, in the order they were added, with repeated names kept as separate lines.
/// A lookup ignores the case of the name; the name is kept as it was given.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Headers {
    entries: Vec<(String, String)>,
}

impl Headers {
    /// No fields.
    pub fn new() -> Headers {
        Headers::default()
    }

    /// Adds a field line after the others. Nothing is checked here: the parser only builds valid fields and the
    /// writer refuses an invalid one.
    pub fn append(&mut self, name: impl Into<String>, value: impl Into<String>) {
        self.entries.push((name.into(), value.into()));
    }

    /// The value of the first line with this name.
    pub fn get(&self, name: &str) -> Option<&str> {
        self.entries
            .iter()
            .find(|(n, _)| n.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }

    /// The values of every line with this name, in order.
    pub fn get_all<'a>(&'a self, name: &'a str) -> impl Iterator<Item = &'a str> + 'a {
        self.entries
            .iter()
            .filter(move |(n, _)| n.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }

    /// Whether a line with this name exists.
    pub fn contains(&self, name: &str) -> bool {
        self.get(name).is_some()
    }

    /// Whether the comma-separated list in any line with this name holds `token`, ignoring case (`Connection:
    /// keep-alive, Upgrade` holds `upgrade`).
    pub fn has_token(&self, name: &str, token: &str) -> bool {
        self.get_all(name).any(|value| {
            value
                .split(',')
                .any(|t| t.trim().eq_ignore_ascii_case(token))
        })
    }

    /// Every line as `(name, value)`, in order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &str)> {
        self.entries.iter().map(|(n, v)| (n.as_str(), v.as_str()))
    }

    /// The number of lines.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether there are no lines.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// A request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Request {
    pub method: Method,
    /// The request target as written (origin-form, absolute-form, authority-form or `*`).
    pub target: String,
    pub version: Version,
    pub headers: Headers,
    pub body: Vec<u8>,
}

impl Request {
    /// An HTTP/1.1 request with no headers and no body.
    pub fn new(method: Method, target: impl Into<String>) -> Request {
        Request {
            method,
            target: target.into(),
            version: Version::Http11,
            headers: Headers::new(),
            body: Vec::new(),
        }
    }

    /// Adds a header line.
    pub fn header(mut self, name: impl Into<String>, value: impl Into<String>) -> Request {
        self.headers.append(name, value);
        self
    }

    /// Sets the body.
    pub fn with_body(mut self, body: impl Into<Vec<u8>>) -> Request {
        self.body = body.into();
        self
    }

    /// Whether the connection stays open after this request's response (RFC 9112 section 9.3): HTTP/1.1 keeps
    /// it open unless `Connection` holds `close`; HTTP/1.0 closes it unless `Connection` holds `keep-alive`
    /// (and not `close`).
    pub fn keep_alive(&self) -> bool {
        keep_alive(self.version, &self.headers)
    }

    /// Whether the client asked for a `100 Continue` before it sends the body (`Expect: 100-continue`,
    /// RFC 9110 section 10.1.1). Always false for HTTP/1.0, which must ignore the expectation.
    pub fn expects_continue(&self) -> bool {
        self.version == Version::Http11 && self.headers.has_token("Expect", "100-continue")
    }
}

pub(crate) fn keep_alive(version: Version, headers: &Headers) -> bool {
    if headers.has_token("Connection", "close") {
        return false;
    }
    match version {
        Version::Http11 => true,
        Version::Http10 => headers.has_token("Connection", "keep-alive"),
    }
}

/// A response.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Response {
    pub version: Version,
    pub status: u16,
    pub reason: String,
    pub headers: Headers,
    pub body: Vec<u8>,
}

impl Response {
    /// An HTTP/1.1 response with no headers and no body.
    pub fn new(status: u16, reason: impl Into<String>) -> Response {
        Response {
            version: Version::Http11,
            status,
            reason: reason.into(),
            headers: Headers::new(),
            body: Vec::new(),
        }
    }

    /// A response with the usual reason phrase of the status (empty for a status without one).
    pub fn with_status(status: u16) -> Response {
        Response::new(status, reason_phrase(status))
    }

    /// Adds a header line.
    pub fn header(mut self, name: impl Into<String>, value: impl Into<String>) -> Response {
        self.headers.append(name, value);
        self
    }

    /// Sets the body.
    pub fn with_body(mut self, body: impl Into<Vec<u8>>) -> Response {
        self.body = body.into();
        self
    }

    /// Whether the connection stays open after this response, by the same rule as [`Request::keep_alive`].
    pub fn keep_alive(&self) -> bool {
        keep_alive(self.version, &self.headers)
    }
}

/// Whether a response with this status never has a body: 1xx, 204 and 304 (RFC 9110 section 6.4.1).
pub fn status_has_no_body(status: u16) -> bool {
    (100..200).contains(&status) || status == 204 || status == 304
}

/// The usual reason phrase of a status, or the empty string for one this table does not hold.
pub fn reason_phrase(status: u16) -> &'static str {
    match status {
        100 => "Continue",
        200 => "OK",
        201 => "Created",
        202 => "Accepted",
        204 => "No Content",
        304 => "Not Modified",
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        405 => "Method Not Allowed",
        408 => "Request Timeout",
        409 => "Conflict",
        411 => "Length Required",
        413 => "Content Too Large",
        414 => "URI Too Long",
        417 => "Expectation Failed",
        429 => "Too Many Requests",
        431 => "Request Header Fields Too Large",
        500 => "Internal Server Error",
        501 => "Not Implemented",
        503 => "Service Unavailable",
        505 => "HTTP Version Not Supported",
        _ => "",
    }
}

/// Why input is not a message this crate accepts. [`ParseError::status`] is the status a server answers.
/// The status is meaningful for the errors of a request; the response parser reports the same kinds for what
/// it reads, and the client decides what to do.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParseError {
    /// The request line is not `method SP target SP version` with single spaces (400).
    BadRequestLine,
    /// The status line is not `version SP three-digit status SP reason` (400).
    BadStatusLine,
    /// The method is not a token (400).
    BadMethod,
    /// The target is empty or holds a byte that is not visible ASCII, or a fragment (400).
    BadTarget,
    /// The version is not `HTTP/` digit `.` digit (400).
    BadVersion,
    /// A version other than 1.0 and 1.1 (505).
    UnsupportedVersion,
    /// A bare LF or a CR that is not followed by LF ends or interrupts a line (400).
    BadLineEnding,
    /// A header line with no colon, or a missing name (400).
    BadHeaderLine,
    /// A header line that starts with a space or tab: obsolete line folding (400).
    ObsoleteFolding,
    /// Whitespace between a header name and its colon (400).
    SpaceBeforeColon,
    /// A header name that is not a token (400).
    BadHeaderName,
    /// A header value with a control character or a byte above 0x7e (400).
    BadHeaderValue,
    /// An HTTP/1.1 request without exactly one `Host` line (400).
    BadHost,
    /// A `Content-Length` that is not a list of digits that fit in 64 bits (400).
    BadContentLength,
    /// `Content-Length` values that differ (400).
    ConflictingContentLength,
    /// `Transfer-Encoding` and `Content-Length` in one message (400).
    TransferEncodingWithContentLength,
    /// A request with a chunked body, which this crate does not read (411).
    ChunkedNotSupported,
    /// A `Transfer-Encoding` that is not chunked (501).
    TransferEncodingNotSupported,
    /// A response whose body has no `Content-Length` and which is not a status without a body (the client's
    /// limit: it cannot read to the end of the connection).
    MissingContentLength,
    /// The request line is longer than its limit (414).
    RequestLineTooLong,
    /// The status line is longer than its limit.
    StatusLineTooLong,
    /// The header lines are more bytes than their limit (431).
    HeadersTooLarge,
    /// More header lines than the limit (431).
    TooManyHeaders,
    /// A body longer than its limit (413).
    BodyTooLarge,
}

impl ParseError {
    /// The status a server answers a request with this error.
    pub fn status(self) -> u16 {
        use ParseError::*;
        match self {
            BadRequestLine
            | BadStatusLine
            | BadMethod
            | BadTarget
            | BadVersion
            | BadLineEnding
            | BadHeaderLine
            | ObsoleteFolding
            | SpaceBeforeColon
            | BadHeaderName
            | BadHeaderValue
            | BadHost
            | BadContentLength
            | ConflictingContentLength
            | TransferEncodingWithContentLength
            | MissingContentLength => 400,
            ChunkedNotSupported => 411,
            BodyTooLarge => 413,
            RequestLineTooLong | StatusLineTooLong => 414,
            HeadersTooLarge | TooManyHeaders => 431,
            TransferEncodingNotSupported => 501,
            UnsupportedVersion => 505,
        }
    }
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?} (status {})", self.status())
    }
}

impl std::error::Error for ParseError {}

/// Why a message cannot be written.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WriteError {
    /// The method is not a token.
    BadMethod,
    /// The target is empty or holds a byte that is not visible ASCII.
    BadTarget,
    /// The status is outside 100 to 599.
    BadStatus,
    /// The reason phrase holds a byte other than tab, space and visible ASCII (so no CR or LF).
    BadReason,
    /// A header name that is not a token.
    BadHeaderName,
    /// A header value with a control character (CR and LF included), a byte above 0x7e, or whitespace at either end.
    BadHeaderValue,
    /// The caller set `Content-Length`, `Transfer-Encoding` or `Connection`, which the writer owns.
    ReservedHeader,
    /// A body on a response whose status has none (1xx, 204, 304).
    BodyNotAllowed,
}

impl fmt::Display for WriteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for WriteError {}
