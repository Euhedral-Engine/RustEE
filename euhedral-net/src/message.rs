//! The HTTP/1.1 message types: `Request`, `Response`, `Headers`, `Method`, `Version` and the errors.

use std::error::Error;
use std::fmt;
use std::net::Ipv6Addr;

/// What a writer says about the connection: a `Connection: close`, a `Connection: keep-alive` (what an HTTP/1.0
/// peer needs to keep the connection open), or no `Connection` line.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConnectionMode {
    Close,
    KeepAlive,
    Unspecified,
}

/// The form of a request target (RFC 9112 section 3.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TargetForm {
    /// `/path?query`.
    Origin,
    /// `scheme://authority/path`.
    Absolute,
    /// `host:port`, for CONNECT.
    Authority,
    /// `*`, for OPTIONS.
    Asterisk,
}

/// The request method. A method the crate knows by name is one of the named variants; any other token is `Other`.
/// The parser only builds `Other` for a token that is not one of the named methods, but a hand-built
/// `Other("CONNECT")` is still the CONNECT method to the writer and the target checks, which go by [`Method::as_str`].
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

    /// The form of the target, or `None` when it is not valid for the method: origin-form begins with `/`;
    /// asterisk-form is `*` and only for OPTIONS; authority-form is `host:port` and only for CONNECT;
    /// absolute-form has a scheme and `://` and an authority. Every byte is visible ASCII and there is no `#`.
    pub fn target_form(&self) -> Option<TargetForm> {
        target_form(&self.method, &self.target)
    }

    /// The authority this request is for (RFC 9112 section 3.2.2): the authority of an absolute-form or
    /// authority-form target, which wins over the `Host` header, else the `Host` value. `None` when there is
    /// neither (an HTTP/1.0 request with no `Host`, or an empty `Host`). A server must route on this, not on the
    /// raw header, or a request can be sent to one place by its target and another by its `Host`.
    pub fn authority(&self) -> Option<&str> {
        authority_of(&self.method, &self.target, &self.headers)
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
        101 => "Switching Protocols",
        200 => "OK",
        201 => "Created",
        202 => "Accepted",
        203 => "Non-Authoritative Information",
        204 => "No Content",
        205 => "Reset Content",
        206 => "Partial Content",
        300 => "Multiple Choices",
        301 => "Moved Permanently",
        302 => "Found",
        303 => "See Other",
        304 => "Not Modified",
        307 => "Temporary Redirect",
        308 => "Permanent Redirect",
        400 => "Bad Request",
        401 => "Unauthorized",
        402 => "Payment Required",
        403 => "Forbidden",
        404 => "Not Found",
        405 => "Method Not Allowed",
        406 => "Not Acceptable",
        407 => "Proxy Authentication Required",
        408 => "Request Timeout",
        409 => "Conflict",
        410 => "Gone",
        411 => "Length Required",
        412 => "Precondition Failed",
        413 => "Content Too Large",
        414 => "URI Too Long",
        415 => "Unsupported Media Type",
        416 => "Range Not Satisfiable",
        417 => "Expectation Failed",
        421 => "Misdirected Request",
        422 => "Unprocessable Content",
        426 => "Upgrade Required",
        428 => "Precondition Required",
        429 => "Too Many Requests",
        431 => "Request Header Fields Too Large",
        500 => "Internal Server Error",
        501 => "Not Implemented",
        502 => "Bad Gateway",
        503 => "Service Unavailable",
        504 => "Gateway Timeout",
        505 => "HTTP Version Not Supported",
        511 => "Network Authentication Required",
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
    /// The target is not one of the four forms its method allows, or holds a byte that is not visible ASCII, or a
    /// fragment (400).
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
    /// A request with more than one `Host` line, an HTTP/1.1 request without one, or a `Host` that is not a host
    /// and an optional port (400).
    BadHost,
    /// A `Content-Length` that is not a list of digits that fit in 64 bits (400).
    BadContentLength,
    /// `Content-Length` values that differ (400).
    ConflictingContentLength,
    /// `Transfer-Encoding` and `Content-Length` in one message (400).
    TransferEncodingWithContentLength,
    /// A request with a chunked body, which this crate does not read (411).
    ChunkedNotSupported,
    /// A `Transfer-Encoding` on a request whose final coding is not `chunked`: the body's length cannot be told
    /// (400, RFC 9112 section 6.3).
    BadTransferEncoding,
    /// A response with a `Transfer-Encoding`, which the client does not read.
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
    /// The status a server answers a request with this error, or `None` for an error only the response parser
    /// reports (`BadStatusLine`, `StatusLineTooLong`, `MissingContentLength`, `TransferEncodingNotSupported`),
    /// which no server answers with a status.
    pub fn status(self) -> Option<u16> {
        use ParseError::*;
        Some(match self {
            BadRequestLine
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
            | BadTransferEncoding => 400,
            ChunkedNotSupported => 411,
            BodyTooLarge => 413,
            RequestLineTooLong => 414,
            HeadersTooLarge | TooManyHeaders => 431,
            UnsupportedVersion => 505,
            BadStatusLine
            | StatusLineTooLong
            | MissingContentLength
            | TransferEncodingNotSupported => {
                return None;
            }
        })
    }
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.status() {
            Some(status) => write!(f, "{self:?} (status {status})"),
            None => write!(f, "{self:?}"),
        }
    }
}

impl Error for ParseError {}

/// Why a message cannot be written.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WriteError {
    /// The method is not a token.
    BadMethod,
    /// The target is not valid for the method (see [`Request::target_form`]).
    BadTarget,
    /// The status is outside 100 to 599.
    BadStatus,
    /// The reason phrase holds a byte other than tab, space and visible ASCII (so no CR or LF).
    BadReason,
    /// A header name that is not a token.
    BadHeaderName,
    /// A request without exactly one `Host` line, or with a `Host` that is not a valid host and optional port.
    BadHost,
    /// A header value with a control character (CR and LF included), a byte above 0x7e, or whitespace at either end.
    BadHeaderValue,
    /// The caller set `Content-Length`, `Transfer-Encoding` or `Connection`, which the writer owns.
    ReservedHeader,
    /// A body, or a length, on a response whose status has none (1xx, 204, 304; a 304 may have a length), or a
    /// body given to the head-only writer.
    BodyNotAllowed,
}

impl fmt::Display for WriteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl Error for WriteError {}

/// See [`Request::authority`].
pub(crate) fn authority_of<'a>(
    method: &Method,
    target: &'a str,
    headers: &'a Headers,
) -> Option<&'a str> {
    let from_target = match target_form(method, target) {
        Some(TargetForm::Authority) => Some(target),
        Some(TargetForm::Absolute) => target
            .split_once("://")
            .and_then(|(_, rest)| rest.split(['/', '?']).next()),
        _ => None,
    };
    from_target
        .or_else(|| headers.get("Host"))
        .filter(|a| !a.is_empty())
}

/// The form of `target` for `method`; see [`Request::target_form`].
pub(crate) fn target_form(method: &Method, target: &str) -> Option<TargetForm> {
    if target.is_empty()
        || !target
            .bytes()
            .all(|b| (0x21..=0x7e).contains(&b) && b != b'#')
    {
        return None;
    }
    if method.as_str() == "CONNECT" {
        return valid_authority(target, true).then_some(TargetForm::Authority);
    }
    if target == "*" {
        return (method.as_str() == "OPTIONS").then_some(TargetForm::Asterisk);
    }
    if target.starts_with('/') {
        return Some(TargetForm::Origin);
    }
    let (scheme, rest) = target.split_once("://")?;
    let mut chars = scheme.chars();
    let scheme_ok = chars.next().is_some_and(|c| c.is_ascii_alphabetic())
        && chars.all(|c| c.is_ascii_alphanumeric() || "+-.".contains(c));
    let authority = rest.split(['/', '?']).next().unwrap_or_default();
    (scheme_ok && !authority.is_empty() && valid_authority(authority, false))
        .then_some(TargetForm::Absolute)
}

/// `uri-host [":" port]` with a non-empty host; the port is digits, and must be present when `port_required`.
fn valid_authority(value: &str, port_required: bool) -> bool {
    let (host, port) = if let Some(inner) = value.strip_prefix('[') {
        let Some((literal, rest)) = inner.split_once(']') else {
            return false;
        };
        if !valid_ip_literal(literal) {
            return false;
        }
        match rest.strip_prefix(':') {
            Some(port) => (literal, Some(port)),
            None if rest.is_empty() => (literal, None),
            None => return false,
        }
    } else {
        match value.split_once(':') {
            Some((host, port)) => (host, Some(port)),
            None => (value, None),
        }
    };
    let host_ok = if value.starts_with('[') {
        true
    } else {
        valid_reg_name(host)
    };
    let port_ok = match port {
        Some(port) => valid_port(port, port_required),
        None => !port_required,
    };
    host_ok && port_ok
}

/// A registered name or IPv4 address: letters, digits and `-._~` only. This is stricter than RFC 3986, which also
/// allows the sub-delimiters (`a,b`, `a;b`) and percent-encodings: a percent-encoding in a host is a way to hide a
/// byte (`a%00b`, `a%2Fb`, `a%40b`) from one reader and show it to another, and none belongs in a host name.
fn valid_reg_name(host: &str) -> bool {
    !host.is_empty()
        && host
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"-._~".contains(&b))
}

/// The inside of `[...]`: an IPv6 address, checked as one. The IPvFuture form is refused.
fn valid_ip_literal(literal: &str) -> bool {
    literal.parse::<Ipv6Addr>().is_ok()
}

/// A port: at most five digits and at most 65535. Empty is allowed unless `required`.
fn valid_port(port: &str, required: bool) -> bool {
    if port.is_empty() {
        return !required;
    }
    port.len() <= 5
        && port.bytes().all(|b| b.is_ascii_digit())
        && port.parse::<u32>().is_ok_and(|n| n <= 65535)
}

/// A `Host` field value: empty (an authority that is missing, RFC 9112 section 3.2) or `uri-host [":" port]`.
pub(crate) fn valid_host_value(value: &str) -> bool {
    value.is_empty() || valid_authority(value, false)
}
