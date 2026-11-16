//! The writer: [`write_response`] and [`write_response_head`] for a server, [`write_request`] for a client, and
//! [`write_continue`].
//!
//! Every message is written as HTTP/1.1. The writer owns the framing: it sets `Content-Length` from the body,
//! sets `Connection` from the [`ConnectionMode`] the caller passes, and refuses a message whose headers already
//! hold `Content-Length`, `Transfer-Encoding` or `Connection`. It refuses, rather than repairs, anything that
//! could put a CR or LF (or another control character) into the output: a method, target, reason phrase, header
//! name or header value that is invalid is an error, and nothing is appended to the output. A request is written
//! only if this crate's parser would accept it: it needs exactly one valid `Host` and a target that is valid for
//! its method.
//!
//! A response to a HEAD request has no body but the length the body would have: [`write_response_head`] writes
//! the head with that length given explicitly and no body.

use crate::message::{
    ConnectionMode, Headers, Request, Response, WriteError, status_has_no_body, target_form,
    valid_host_value,
};

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

fn push_headers(
    out: &mut Vec<u8>,
    headers: &Headers,
    length: Option<u64>,
    connection: ConnectionMode,
) {
    for (name, value) in headers.iter() {
        out.extend_from_slice(name.as_bytes());
        out.extend_from_slice(b": ");
        out.extend_from_slice(value.as_bytes());
        out.extend_from_slice(b"\r\n");
    }
    if let Some(length) = length {
        out.extend_from_slice(format!("Content-Length: {length}\r\n").as_bytes());
    }
    match connection {
        ConnectionMode::Close => out.extend_from_slice(b"Connection: close\r\n"),
        ConnectionMode::KeepAlive => out.extend_from_slice(b"Connection: keep-alive\r\n"),
        ConnectionMode::Unspecified => {}
    }
    out.extend_from_slice(b"\r\n");
}

fn check_response(response: &Response) -> Result<(), WriteError> {
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
    check_headers(&response.headers)
}

/// An interim (1xx) response never carries `Connection: close` or `keep-alive`: it does not end or keep anything,
/// the final response does. The mode is ignored for it.
fn interim_mode(response: &Response, connection: ConnectionMode) -> ConnectionMode {
    if response.status < 200 {
        ConnectionMode::Unspecified
    } else {
        connection
    }
}

fn push_status_line(out: &mut Vec<u8>, response: &Response) {
    out.extend_from_slice(
        format!("HTTP/1.1 {} {}\r\n", response.status, response.reason).as_bytes(),
    );
}

/// Appends a response to `out`. `connection` (ignored for a 1xx) adds `Connection: close` or `Connection: keep-alive`, or nothing. A
/// status without a body (1xx, 204, 304) is written without `Content-Length` and must have no body. On an error
/// `out` is unchanged.
pub fn write_response(
    response: &Response,
    connection: ConnectionMode,
    out: &mut Vec<u8>,
) -> Result<(), WriteError> {
    check_response(response)?;
    let no_body = status_has_no_body(response.status);
    if no_body && !response.body.is_empty() {
        return Err(WriteError::BodyNotAllowed);
    }
    push_status_line(out, response);
    let length = (!no_body).then_some(response.body.len() as u64);
    push_headers(
        out,
        &response.headers,
        length,
        interim_mode(response, connection),
    );
    out.extend_from_slice(&response.body);
    Ok(())
}

/// Appends the head of a response and no body, for the answer to a HEAD request (or any head that is sent before
/// its body): `content_length` is the length the body has, or `None` for no `Content-Length` line. The response's
/// own body must be empty. A 1xx or 204 takes no length; a 304 may have one. On an error `out` is unchanged.
pub fn write_response_head(
    response: &Response,
    content_length: Option<u64>,
    connection: ConnectionMode,
    out: &mut Vec<u8>,
) -> Result<(), WriteError> {
    check_response(response)?;
    let no_length = response.status < 200 || response.status == 204;
    if !response.body.is_empty() || (no_length && content_length.is_some()) {
        return Err(WriteError::BodyNotAllowed);
    }
    push_status_line(out, response);
    push_headers(
        out,
        &response.headers,
        content_length,
        interim_mode(response, connection),
    );
    Ok(())
}

/// Appends a request to `out`. `Content-Length` is written when there is a body, and for POST, PUT and PATCH
/// without one (as `0`). The request is written as HTTP/1.1 whatever its version, and needs exactly one valid
/// `Host` header (empty only for an authority that is missing) and a target valid for its method. On an error `out`
/// is unchanged.
pub fn write_request(
    request: &Request,
    connection: ConnectionMode,
    out: &mut Vec<u8>,
) -> Result<(), WriteError> {
    let method = request.method.as_str();
    if !valid_name(method) {
        return Err(WriteError::BadMethod);
    }
    if target_form(&request.method, &request.target).is_none() {
        return Err(WriteError::BadTarget);
    }
    check_headers(&request.headers)?;
    let mut hosts = request.headers.get_all("Host");
    match (hosts.next(), hosts.next()) {
        (Some(host), None) if valid_host_value(host) => {}
        _ => return Err(WriteError::BadHost),
    }
    out.extend_from_slice(format!("{method} {} HTTP/1.1\r\n", request.target).as_bytes());
    let needs_length = !request.body.is_empty() || matches!(method, "POST" | "PUT" | "PATCH");
    push_headers(
        out,
        &request.headers,
        needs_length.then_some(request.body.len() as u64),
        connection,
    );
    out.extend_from_slice(&request.body);
    Ok(())
}

/// Appends the interim `100 Continue` response.
pub fn write_continue(out: &mut Vec<u8>) {
    out.extend_from_slice(b"HTTP/1.1 100 Continue\r\n\r\n");
}
