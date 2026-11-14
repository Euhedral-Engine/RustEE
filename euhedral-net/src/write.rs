//! The writer: [`write_response`] for a server, [`write_request`] for a client, and [`write_continue`].
//!
//! Every message is written as HTTP/1.1. The writer owns the framing: it sets `Content-Length` from the body,
//! sets `Connection: close` when the caller says the connection ends with this message, and refuses a message
//! whose headers already hold `Content-Length`, `Transfer-Encoding` or `Connection`. It refuses, rather than
//! repairs, anything that could put a CR or LF (or another control character) into the output: a method, target,
//! reason phrase, header name or header value that is invalid is an error, and nothing is appended to the output.
//!
//! A response to a HEAD request has no body but a real `Content-Length`; this part of the crate does not write
//! one (the writer's `Content-Length` is the body's).

use crate::message::{Headers, Request, Response, WriteError, status_has_no_body};

fn valid_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"!#$%&'*+-.^_`|~".contains(&b))
}

fn valid_value(value: &str) -> bool {
    let bytes = value.as_bytes();
    let edge = |b: Option<&u8>| b.is_some_and(|b| *b == b' ' || *b == b'\t');
    bytes
        .iter()
        .all(|&b| b == b'\t' || (0x20..=0x7e).contains(&b))
        && !edge(bytes.first())
        && !edge(bytes.last())
}

fn check_headers(headers: &Headers) -> Result<(), WriteError> {
    for (name, value) in headers.iter() {
        if !valid_name(name) {
            return Err(WriteError::BadHeaderName);
        }
        if ["content-length", "transfer-encoding", "connection"]
            .iter()
            .any(|r| name.eq_ignore_ascii_case(r))
        {
            return Err(WriteError::ReservedHeader);
        }
        if !valid_value(value) {
            return Err(WriteError::BadHeaderValue);
        }
    }
    Ok(())
}

fn push_headers(out: &mut Vec<u8>, headers: &Headers, length: Option<usize>, close: bool) {
    for (name, value) in headers.iter() {
        out.extend_from_slice(name.as_bytes());
        out.extend_from_slice(b": ");
        out.extend_from_slice(value.as_bytes());
        out.extend_from_slice(b"\r\n");
    }
    if let Some(length) = length {
        out.extend_from_slice(format!("Content-Length: {length}\r\n").as_bytes());
    }
    if close {
        out.extend_from_slice(b"Connection: close\r\n");
    }
    out.extend_from_slice(b"\r\n");
}

/// Appends a response to `out`. `close` says the connection ends after this response and adds `Connection:
/// close`. A status without a body (1xx, 204, 304) is written without `Content-Length` and must have no body.
/// On an error `out` is unchanged.
pub fn write_response(
    response: &Response,
    close: bool,
    out: &mut Vec<u8>,
) -> Result<(), WriteError> {
    if !(100..600).contains(&response.status) {
        return Err(WriteError::BadStatus);
    }
    if !response
        .reason
        .bytes()
        .all(|b| b == b'\t' || (0x20..=0x7e).contains(&b))
    {
        return Err(WriteError::BadReason);
    }
    check_headers(&response.headers)?;
    let no_body = status_has_no_body(response.status);
    if no_body && !response.body.is_empty() {
        return Err(WriteError::BodyNotAllowed);
    }
    out.extend_from_slice(
        format!("HTTP/1.1 {} {}\r\n", response.status, response.reason).as_bytes(),
    );
    let length = (!no_body).then_some(response.body.len());
    push_headers(out, &response.headers, length, close);
    out.extend_from_slice(&response.body);
    Ok(())
}

/// Appends a request to `out`. `Content-Length` is written when there is a body, and for POST, PUT and PATCH
/// without one (as `0`). `close` adds `Connection: close`. On an error `out` is unchanged. The version of the
/// request is not used: the request is written as HTTP/1.1, which needs a `Host` header from the caller.
pub fn write_request(request: &Request, close: bool, out: &mut Vec<u8>) -> Result<(), WriteError> {
    let method = request.method.as_str();
    if !valid_name(method) {
        return Err(WriteError::BadMethod);
    }
    let target = request.target.as_bytes();
    if target.is_empty()
        || !target
            .iter()
            .all(|&b| (0x21..=0x7e).contains(&b) && b != b'#')
    {
        return Err(WriteError::BadTarget);
    }
    check_headers(&request.headers)?;
    out.extend_from_slice(format!("{method} {} HTTP/1.1\r\n", request.target).as_bytes());
    let needs_length = !request.body.is_empty() || matches!(method, "POST" | "PUT" | "PATCH");
    push_headers(
        out,
        &request.headers,
        needs_length.then_some(request.body.len()),
        close,
    );
    out.extend_from_slice(&request.body);
    Ok(())
}

/// Appends the interim `100 Continue` response.
pub fn write_continue(out: &mut Vec<u8>) {
    out.extend_from_slice(b"HTTP/1.1 100 Continue\r\n\r\n");
}
