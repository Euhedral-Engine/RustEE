//! Tests of the message types, the parsers and the writer.

use crate::message::{ParseError, ParseError as E};
use crate::*;

// ---- drivers ----------------------------------------------------------------------------------------------

#[derive(Debug, PartialEq)]
enum Event<T> {
    Head {
        expect_continue: bool,
        content_length: usize,
    },
    Message(T),
}

type Outcome<T> = (Vec<Event<T>>, Option<ParseError>);

/// Feeds the chunks one after another, looping inside a chunk while the parser stops at a message's end.
fn drive<T>(
    mut feed: impl FnMut(&[u8]) -> Result<(usize, Progress<T>), ParseError>,
    chunks: &[&[u8]],
) -> Outcome<T> {
    let mut events = Vec::new();
    for chunk in chunks {
        let mut rest = *chunk;
        while !rest.is_empty() {
            match feed(rest) {
                Err(e) => return (events, Some(e)),
                Ok((used, progress)) => {
                    assert!(used <= rest.len());
                    rest = &rest[used..];
                    match progress {
                        Progress::NeedMore => assert!(rest.is_empty(), "stopped with input left"),
                        Progress::HeadComplete {
                            expect_continue,
                            content_length,
                        } => events.push(Event::Head {
                            expect_continue,
                            content_length,
                        }),
                        Progress::Complete(m) => events.push(Event::Message(m)),
                    }
                }
            }
        }
    }
    (events, None)
}

fn cut<'a>(input: &'a [u8], cuts: &[usize]) -> Vec<&'a [u8]> {
    let mut out = Vec::new();
    let mut last = 0;
    for &c in cuts {
        out.push(&input[last..c]);
        last = c;
    }
    out.push(&input[last..]);
    out
}

fn requests(limits: Limits, input: &[u8], cuts: &[usize]) -> Outcome<Request> {
    let mut parser = RequestParser::new(limits);
    let chunks = cut(input, cuts);
    let outcome = drive(|b| parser.feed(b), &chunks);
    let held = limits.max_start_line + limits.max_header_bytes + limits.max_body;
    assert!(parser.buffered() <= held);
    outcome
}

fn responses(limits: Limits, to_head: bool, input: &[u8], cuts: &[usize]) -> Outcome<Response> {
    let mut parser = ResponseParser::new(limits);
    parser.set_response_to_head(to_head);
    let chunks = cut(input, cuts);
    drive(|b| parser.feed(b), &chunks)
}

/// Every split point natively; under Miri, every fifth (and byte by byte, always), to keep the shard short.
const SPLIT_STRIDE: usize = if cfg!(miri) { 5 } else { 1 };

fn every_split_request(limits: Limits, input: &[u8]) -> Outcome<Request> {
    let whole = requests(limits, input, &[]);
    for at in (0..=input.len()).step_by(SPLIT_STRIDE) {
        assert_eq!(requests(limits, input, &[at]), whole, "split at {at}");
    }
    let all: Vec<usize> = (0..=input.len()).collect();
    assert_eq!(requests(limits, input, &all), whole, "byte by byte");
    whole
}

fn every_split_response(limits: Limits, to_head: bool, input: &[u8]) -> Outcome<Response> {
    let whole = responses(limits, to_head, input, &[]);
    for at in 0..=input.len() {
        assert_eq!(
            responses(limits, to_head, input, &[at]),
            whole,
            "split at {at}"
        );
    }
    let all: Vec<usize> = (0..=input.len()).collect();
    assert_eq!(
        responses(limits, to_head, input, &all),
        whole,
        "byte by byte"
    );
    whole
}

fn one_request(input: &[u8]) -> Request {
    let (events, error) = every_split_request(Limits::default(), input);
    assert_eq!(error, None);
    let mut messages = events.into_iter().filter_map(|e| match e {
        Event::Message(m) => Some(m),
        Event::Head { .. } => None,
    });
    let first = messages.next().expect("a request");
    assert!(messages.next().is_none());
    first
}

fn req(method: Method, target: &str, headers: &[(&str, &str)], body: &[u8]) -> Request {
    let mut r = Request::new(method, target);
    for (n, v) in headers {
        r.headers.append(*n, *v);
    }
    r.body = body.to_vec();
    r
}

// ---- requests: good ---------------------------------------------------------------------------------------

#[test]
fn good_requests_parse_to_the_expected_message_at_every_split() {
    let table: Vec<(&str, Vec<u8>, Request)> = vec![
        (
            "a plain GET",
            b"GET /a/b?c=d HTTP/1.1\r\nHost: example\r\n\r\n".to_vec(),
            req(Method::Get, "/a/b?c=d", &[("Host", "example")], b""),
        ),
        (
            "a POST with a body",
            b"POST /x HTTP/1.1\r\nHost: h\r\nContent-Length: 5\r\n\r\nhello".to_vec(),
            req(
                Method::Post,
                "/x",
                &[("Host", "h"), ("Content-Length", "5")],
                b"hello",
            ),
        ),
        (
            "a body that holds a request",
            b"POST / HTTP/1.1\r\nHost: h\r\nContent-Length: 18\r\n\r\nGET / HTTP/1.1\r\n\r\n"
                .to_vec(),
            req(
                Method::Post,
                "/",
                &[("Host", "h"), ("Content-Length", "18")],
                b"GET / HTTP/1.1\r\n\r\n",
            ),
        ),
        (
            "empty lines before the request line",
            b"\r\n\r\nGET / HTTP/1.1\r\nHost: h\r\n\r\n".to_vec(),
            req(Method::Get, "/", &[("Host", "h")], b""),
        ),
        (
            "HTTP/1.0 without Host",
            b"GET / HTTP/1.0\r\n\r\n".to_vec(),
            Request {
                version: Version::Http10,
                ..req(Method::Get, "/", &[], b"")
            },
        ),
        (
            "repeated names, mixed case, trimmed values, an empty value",
            b"GET / HTTP/1.1\r\nhost:   h \t\r\nX-A: 1\r\nx-a: 2\r\nX-Empty:\r\n\r\n".to_vec(),
            req(
                Method::Get,
                "/",
                &[("host", "h"), ("X-A", "1"), ("x-a", "2"), ("X-Empty", "")],
                b"",
            ),
        ),
        (
            "identical Content-Length values in one line",
            b"PUT /p HTTP/1.1\r\nHost: h\r\nContent-Length: 2, 2\r\n\r\nab".to_vec(),
            req(
                Method::Put,
                "/p",
                &[("Host", "h"), ("Content-Length", "2, 2")],
                b"ab",
            ),
        ),
        (
            "identical Content-Length values in two lines, with a leading zero",
            b"PUT /p HTTP/1.1\r\nHost: h\r\nContent-Length: 2\r\nContent-Length: 02\r\n\r\nab"
                .to_vec(),
            req(
                Method::Put,
                "/p",
                &[
                    ("Host", "h"),
                    ("Content-Length", "2"),
                    ("Content-Length", "02"),
                ],
                b"ab",
            ),
        ),
        (
            "the asterisk target of OPTIONS",
            b"OPTIONS * HTTP/1.1\r\nHost: h\r\n\r\n".to_vec(),
            req(Method::Options, "*", &[("Host", "h")], b""),
        ),
        (
            "an absolute-form target",
            b"GET http://h/p HTTP/1.1\r\nHost: h\r\n\r\n".to_vec(),
            req(Method::Get, "http://h/p", &[("Host", "h")], b""),
        ),
        (
            "a method of the crate's own",
            b"PURGE /c HTTP/1.1\r\nHost: h\r\n\r\n".to_vec(),
            req(Method::Other("PURGE".into()), "/c", &[("Host", "h")], b""),
        ),
        (
            "a zero Content-Length",
            b"POST / HTTP/1.1\r\nHost: h\r\nContent-Length: 0\r\n\r\n".to_vec(),
            req(
                Method::Post,
                "/",
                &[("Host", "h"), ("Content-Length", "0")],
                b"",
            ),
        ),
    ];
    for (name, input, expected) in table {
        let (events, error) = every_split_request(Limits::default(), &input);
        assert_eq!(error, None, "{name}");
        let message = events.into_iter().find_map(|e| match e {
            Event::Message(m) => Some(m),
            Event::Head { .. } => None,
        });
        assert_eq!(message, Some(expected), "{name}");
    }
}

#[test]
fn a_pipelined_pair_comes_out_as_two_requests_at_every_split() {
    let pair: &[u8] = b"POST /1 HTTP/1.1\r\nHost: h\r\nContent-Length: 3\r\n\r\nabcGET /2 HTTP/1.1\r\nHost: h\r\n\r\n";
    let (events, error) = every_split_request(Limits::default(), pair);
    assert_eq!(error, None);
    let messages: Vec<_> = events
        .iter()
        .filter_map(|e| match e {
            Event::Message(m) => Some(m),
            Event::Head { .. } => None,
        })
        .collect();
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0].target, "/1");
    assert_eq!(messages[0].body, b"abc");
    assert_eq!(messages[1].target, "/2");
    // A third request cut in the middle stays unfinished, and the first two are not lost.
    let mut parser = RequestParser::new(Limits::default());
    let mut input = pair.to_vec();
    input.extend_from_slice(b"GET /3 HTTP/1.1\r\nHo");
    let (events, error) = drive(|b| parser.feed(b), &[&input]);
    assert_eq!((events.len(), error), (3, None));
    assert_eq!(parser.state(), State::Head);
    assert!(parser.buffered() > 0);
}

#[test]
fn the_parser_consumes_only_the_current_request() {
    let mut parser = RequestParser::new(Limits::default());
    let input = b"GET /1 HTTP/1.1\r\nHost: h\r\n\r\nGET /2 HTTP/1.1\r\nHost: h\r\n\r\n";
    let (used, progress) = parser.feed(input).unwrap();
    assert_eq!(used, "GET /1 HTTP/1.1\r\nHost: h\r\n\r\n".len());
    assert!(matches!(progress, Progress::Complete(r) if r.target == "/1"));
}

#[test]
fn a_head_with_a_body_pending_is_observable_with_the_expectation() {
    let head = b"POST /u HTTP/1.1\r\nHost: h\r\nExpect: 100-Continue\r\nContent-Length: 4\r\n\r\n";
    let mut parser = RequestParser::new(Limits::default());
    let (used, progress) = parser.feed(head).unwrap();
    assert_eq!(used, head.len());
    assert_eq!(
        progress,
        Progress::HeadComplete {
            expect_continue: true,
            content_length: 4
        }
    );
    assert_eq!(
        parser.state(),
        State::Body {
            expect_continue: true,
            remaining: 4
        }
    );
    let (used, progress) = parser.feed(b"ab").unwrap();
    assert_eq!((used, progress), (2, Progress::NeedMore));
    assert_eq!(
        parser.state(),
        State::Body {
            expect_continue: true,
            remaining: 2
        }
    );
    let (used, progress) = parser.feed(b"cdGET").unwrap();
    assert_eq!(used, 2);
    let Progress::Complete(request) = progress else {
        panic!("complete")
    };
    assert_eq!(request.body, b"abcd");
    assert!(request.expects_continue());
    assert_eq!(parser.state(), State::Head);
}

#[test]
fn the_expectation_is_set_only_where_it_means_something() {
    let cases: [(&str, bool); 4] = [
        // HTTP/1.1, a body: yes.
        (
            "POST / HTTP/1.1\r\nHost: h\r\nExpect: 100-continue\r\nContent-Length: 1\r\n\r\n",
            true,
        ),
        // No body to wait for.
        (
            "POST / HTTP/1.1\r\nHost: h\r\nExpect: 100-continue\r\n\r\n",
            false,
        ),
        // HTTP/1.0 ignores it.
        (
            "POST / HTTP/1.0\r\nExpect: 100-continue\r\nContent-Length: 1\r\n\r\n",
            false,
        ),
        // Another expectation is ignored.
        (
            "POST / HTTP/1.1\r\nHost: h\r\nExpect: other\r\nContent-Length: 1\r\n\r\n",
            false,
        ),
    ];
    for (head, expected) in cases {
        let (events, error) = requests(Limits::default(), head.as_bytes(), &[]);
        assert_eq!(error, None, "{head}");
        let seen = events.iter().any(|e| {
            matches!(
                e,
                Event::Head {
                    expect_continue: true,
                    ..
                }
            )
        });
        assert_eq!(seen, expected, "{head}");
    }
}

#[test]
fn keep_alive_follows_the_version_and_connection() {
    let table = [
        ("GET / HTTP/1.1\r\nHost: h\r\n\r\n", true),
        (
            "GET / HTTP/1.1\r\nHost: h\r\nConnection: close\r\n\r\n",
            false,
        ),
        (
            "GET / HTTP/1.1\r\nHost: h\r\nConnection: Keep-Alive, CLOSE\r\n\r\n",
            false,
        ),
        (
            "GET / HTTP/1.1\r\nHost: h\r\nConnection: keep-alive\r\n\r\n",
            true,
        ),
        ("GET / HTTP/1.0\r\n\r\n", false),
        ("GET / HTTP/1.0\r\nConnection: keep-alive\r\n\r\n", true),
        ("GET / HTTP/1.0\r\nConnection: Keep-Alive\r\n\r\n", true),
        (
            "GET / HTTP/1.0\r\nConnection: keep-alive, close\r\n\r\n",
            false,
        ),
        ("GET / HTTP/1.0\r\nConnection: upgrade\r\n\r\n", false),
    ];
    for (input, expected) in table {
        assert_eq!(
            one_request(input.as_bytes()).keep_alive(),
            expected,
            "{input}"
        );
    }
}

// ---- requests: bad ----------------------------------------------------------------------------------------

fn bad_requests() -> Vec<(&'static str, Vec<u8>, ParseError, u16)> {
    let h = "Host: h\r\n";
    let line = |s: &str| format!("GET / HTTP/1.1\r\n{h}{s}\r\n\r\n").into_bytes();
    let t: Vec<(&'static str, Vec<u8>, ParseError, u16)> = vec![
        (
            "a bare LF after the request line",
            b"GET / HTTP/1.1\nHost: h\r\n\r\n".to_vec(),
            E::BadLineEnding,
            400,
        ),
        (
            "a bare LF after a header",
            b"GET / HTTP/1.1\r\nHost: h\n\r\n".to_vec(),
            E::BadLineEnding,
            400,
        ),
        (
            "a bare LF for the empty line",
            b"GET / HTTP/1.1\r\nHost: h\r\n\n".to_vec(),
            E::BadLineEnding,
            400,
        ),
        (
            "a CR inside a line",
            b"GET / HTTP/1.1\r\nHost: h\rx\r\n\r\n".to_vec(),
            E::BadLineEnding,
            400,
        ),
        (
            "a CR in the request line",
            b"GET /\r HTTP/1.1\r\nHost: h\r\n\r\n".to_vec(),
            E::BadLineEnding,
            400,
        ),
        (
            "a bare LF before the request line",
            b"\nGET / HTTP/1.1\r\nHost: h\r\n\r\n".to_vec(),
            E::BadLineEnding,
            400,
        ),
        (
            "too many empty lines before the request line",
            b"\r\n\r\n\r\n\r\n\r\nGET / HTTP/1.1\r\nHost: h\r\n\r\n".to_vec(),
            E::BadRequestLine,
            400,
        ),
        (
            "two spaces in the request line",
            b"GET  / HTTP/1.1\r\nHost: h\r\n\r\n".to_vec(),
            E::BadRequestLine,
            400,
        ),
        (
            "a tab in the request line",
            b"GET\t/ HTTP/1.1\r\nHost: h\r\n\r\n".to_vec(),
            E::BadRequestLine,
            400,
        ),
        (
            "no version",
            b"GET /\r\nHost: h\r\n\r\n".to_vec(),
            E::BadRequestLine,
            400,
        ),
        (
            "an extra word",
            b"GET / HTTP/1.1 x\r\nHost: h\r\n\r\n".to_vec(),
            E::BadRequestLine,
            400,
        ),
        (
            "a trailing space",
            b"GET / HTTP/1.1 \r\nHost: h\r\n\r\n".to_vec(),
            E::BadRequestLine,
            400,
        ),
        (
            "a method that is not a token",
            b"G(T / HTTP/1.1\r\nHost: h\r\n\r\n".to_vec(),
            E::BadMethod,
            400,
        ),
        (
            "a target with a fragment",
            b"GET /a#b HTTP/1.1\r\nHost: h\r\n\r\n".to_vec(),
            E::BadTarget,
            400,
        ),
        (
            "a target with a byte above 0x7e",
            b"GET /\xc3\xa9 HTTP/1.1\r\nHost: h\r\n\r\n".to_vec(),
            E::BadTarget,
            400,
        ),
        (
            "a lowercase version",
            b"GET / http/1.1\r\nHost: h\r\n\r\n".to_vec(),
            E::BadVersion,
            400,
        ),
        (
            "a version of two digits",
            b"GET / HTTP/1.11\r\nHost: h\r\n\r\n".to_vec(),
            E::BadVersion,
            400,
        ),
        (
            "a version without a minor",
            b"GET / HTTP/1\r\nHost: h\r\n\r\n".to_vec(),
            E::BadVersion,
            400,
        ),
        (
            "HTTP/2.0",
            b"GET / HTTP/2.0\r\nHost: h\r\n\r\n".to_vec(),
            E::UnsupportedVersion,
            505,
        ),
        (
            "HTTP/0.9",
            b"GET / HTTP/0.9\r\n\r\n".to_vec(),
            E::UnsupportedVersion,
            505,
        ),
        (
            "HTTP/1.2",
            b"GET / HTTP/1.2\r\nHost: h\r\n\r\n".to_vec(),
            E::UnsupportedVersion,
            505,
        ),
        (
            "a folded line",
            b"GET / HTTP/1.1\r\nHost: h\r\nX: a\r\n b\r\n\r\n".to_vec(),
            E::ObsoleteFolding,
            400,
        ),
        (
            "a fold right after the request line",
            b"GET / HTTP/1.1\r\n Host: h\r\n\r\n".to_vec(),
            E::ObsoleteFolding,
            400,
        ),
        (
            "a tab fold",
            b"GET / HTTP/1.1\r\nHost: h\r\n\tb\r\n\r\n".to_vec(),
            E::ObsoleteFolding,
            400,
        ),
        (
            "a space before the colon",
            b"GET / HTTP/1.1\r\nHost : h\r\n\r\n".to_vec(),
            E::SpaceBeforeColon,
            400,
        ),
        (
            "a tab before the colon",
            b"GET / HTTP/1.1\r\nHost\t: h\r\n\r\n".to_vec(),
            E::SpaceBeforeColon,
            400,
        ),
        (
            "a space inside a name",
            b"GET / HTTP/1.1\r\nHo st: h\r\n\r\n".to_vec(),
            E::BadHeaderName,
            400,
        ),
        (
            "a name that is not a token",
            b"GET / HTTP/1.1\r\nH@st: h\r\n\r\n".to_vec(),
            E::BadHeaderName,
            400,
        ),
        (
            "a line with no colon",
            b"GET / HTTP/1.1\r\nHost: h\r\nNoColon\r\n\r\n".to_vec(),
            E::BadHeaderLine,
            400,
        ),
        (
            "an empty name",
            b"GET / HTTP/1.1\r\nHost: h\r\n: v\r\n\r\n".to_vec(),
            E::BadHeaderLine,
            400,
        ),
        ("a NUL in a value", line("X: a\0b"), E::BadHeaderValue, 400),
        (
            "a DEL in a value",
            line("X: a\x7fb"),
            E::BadHeaderValue,
            400,
        ),
        (
            "a vertical tab in a value",
            line("X: a\x0bb"),
            E::BadHeaderValue,
            400,
        ),
        (
            "a byte above 0x7e in a value",
            {
                let mut v = b"GET / HTTP/1.1\r\nHost: h\r\nX: a".to_vec();
                v.extend_from_slice(b"\xe9b\r\n\r\n");
                v
            },
            E::BadHeaderValue,
            400,
        ),
        (
            "no Host on HTTP/1.1",
            b"GET / HTTP/1.1\r\n\r\n".to_vec(),
            E::BadHost,
            400,
        ),
        (
            "two Host lines",
            b"GET / HTTP/1.1\r\nHost: a\r\nHost: b\r\n\r\n".to_vec(),
            E::BadHost,
            400,
        ),
        (
            "a Host with a space",
            b"GET / HTTP/1.1\r\nHost: a b\r\n\r\n".to_vec(),
            E::BadHost,
            400,
        ),
        (
            "a Content-Length of letters",
            line("Content-Length: abc"),
            E::BadContentLength,
            400,
        ),
        (
            "a signed Content-Length",
            line("Content-Length: +5"),
            E::BadContentLength,
            400,
        ),
        (
            "a negative Content-Length",
            line("Content-Length: -5"),
            E::BadContentLength,
            400,
        ),
        (
            "an empty Content-Length",
            line("Content-Length:"),
            E::BadContentLength,
            400,
        ),
        (
            "a Content-Length with a space inside",
            line("Content-Length: 5 5"),
            E::BadContentLength,
            400,
        ),
        (
            "a hexadecimal Content-Length",
            line("Content-Length: 0x10"),
            E::BadContentLength,
            400,
        ),
        (
            "a Content-Length list with an empty element",
            line("Content-Length: 5,"),
            E::BadContentLength,
            400,
        ),
        (
            "a Content-Length that overflows 64 bits",
            line("Content-Length: 18446744073709551616"),
            E::BadContentLength,
            400,
        ),
        (
            "a Content-Length that overflows with leading zeros",
            line("Content-Length: 99999999999999999999999"),
            E::BadContentLength,
            400,
        ),
        (
            "differing Content-Length lines",
            line("Content-Length: 5\r\nContent-Length: 6"),
            E::ConflictingContentLength,
            400,
        ),
        (
            "differing Content-Length in a list",
            line("Content-Length: 5, 6"),
            E::ConflictingContentLength,
            400,
        ),
        (
            "a Content-Length over the body limit",
            line("Content-Length: 2000000"),
            E::BodyTooLarge,
            413,
        ),
        (
            "a huge Content-Length",
            line("Content-Length: 18446744073709551615"),
            E::BodyTooLarge,
            413,
        ),
        (
            "Transfer-Encoding with Content-Length",
            line("Transfer-Encoding: chunked\r\nContent-Length: 5"),
            E::TransferEncodingWithContentLength,
            400,
        ),
        (
            "Content-Length before Transfer-Encoding",
            line("Content-Length: 5\r\nTransfer-Encoding: gzip"),
            E::TransferEncodingWithContentLength,
            400,
        ),
        (
            "a chunked request",
            line("Transfer-Encoding: chunked"),
            E::ChunkedNotSupported,
            411,
        ),
        (
            "a chunked request in capitals",
            line("transfer-encoding: ChUnKeD"),
            E::ChunkedNotSupported,
            411,
        ),
        (
            "a chunked request after another coding",
            line("Transfer-Encoding: gzip, chunked"),
            E::ChunkedNotSupported,
            411,
        ),
        (
            "another transfer coding",
            line("Transfer-Encoding: gzip"),
            E::TransferEncodingNotSupported,
            501,
        ),
        (
            "an empty Transfer-Encoding",
            line("Transfer-Encoding:"),
            E::TransferEncodingNotSupported,
            501,
        ),
    ];
    t
}

#[test]
fn bad_requests_fail_with_the_exact_error_and_status_at_every_split() {
    for (name, input, error, status) in bad_requests() {
        let (events, got) = every_split_request(Limits::default(), &input);
        assert_eq!(got, Some(error), "{name}");
        assert_eq!(error.status(), status, "{name}");
        assert!(
            events.iter().all(|e| !matches!(e, Event::Message(_))),
            "{name}: a request came out"
        );
    }
}

#[test]
fn a_failed_parser_stays_failed_and_holds_nothing() {
    let mut parser = RequestParser::new(Limits::default());
    assert_eq!(parser.feed(b"GET / HTTP/1.1\nxyz"), Err(E::BadLineEnding));
    assert_eq!(parser.state(), State::Failed(E::BadLineEnding));
    assert_eq!(parser.buffered(), 0);
    assert_eq!(
        parser.feed(b"GET / HTTP/1.1\r\nHost: h\r\n\r\n"),
        Err(E::BadLineEnding)
    );
}

#[test]
fn a_smuggling_pair_never_gives_a_request() {
    // The two framings a lenient parser might disagree about with a proxy.
    let inputs: [&[u8]; 3] = [
        b"POST / HTTP/1.1\r\nHost: h\r\nContent-Length: 4\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\nGET /evil HTTP/1.1\r\nHost: h\r\n\r\n",
        b"POST / HTTP/1.1\r\nHost: h\r\nContent-Length: 4\r\nContent-Length: 40\r\n\r\nabcd",
        b"POST / HTTP/1.1\r\nHost: h\r\nTransfer-Encoding : chunked\r\nContent-Length: 4\r\n\r\nabcd",
    ];
    for input in inputs {
        let (events, error) = every_split_request(Limits::default(), input);
        assert!(error.is_some() && events.is_empty());
    }
}

// ---- limits -----------------------------------------------------------------------------------------------

fn small() -> Limits {
    Limits {
        max_start_line: 40,
        max_header_bytes: 100,
        max_headers: 4,
        max_body: 16,
    }
}

/// Feeds byte by byte, returning the index of the byte that failed, and checking that the parser holds no more
/// than its limits allow after each byte.
fn first_failure(limits: Limits, input: &[u8]) -> Option<(usize, ParseError)> {
    let mut parser = RequestParser::new(limits);
    for (i, b) in input.iter().enumerate() {
        match parser.feed(&[*b]) {
            Err(e) => {
                assert_eq!(parser.buffered(), 0);
                return Some((i, e));
            }
            Ok((used, _)) => {
                assert_eq!(used, 1);
                assert!(
                    parser.buffered()
                        <= limits.max_start_line + limits.max_header_bytes + limits.max_body
                );
            }
        }
    }
    None
}

#[test]
fn the_request_line_limit_passes_at_the_limit_and_fails_one_byte_over() {
    let limits = small();
    let make = |total: usize| {
        // `GET /<fill> HTTP/1.1\r\n` is 16 bytes plus the fill.
        let fill = "a".repeat(total - 16);
        format!("GET /{fill} HTTP/1.1\r\nHost: h\r\n\r\n").into_bytes()
    };
    let at = make(limits.max_start_line);
    let (events, error) = every_split_request(limits, &at);
    assert_eq!((events.len(), error), (1, None));
    let over = make(limits.max_start_line + 1);
    let (_, error) = every_split_request(limits, &over);
    assert_eq!(error, Some(E::RequestLineTooLong));
    assert_eq!(E::RequestLineTooLong.status(), 414);
    // The byte that fails is the one past the limit: nothing past it was buffered.
    assert_eq!(
        first_failure(limits, &over),
        Some((limits.max_start_line, E::RequestLineTooLong))
    );
    // Without a line end the parser still stops at the limit.
    let endless = vec![b'a'; 1000];
    assert_eq!(
        first_failure(limits, &endless),
        Some((limits.max_start_line, E::RequestLineTooLong))
    );
    let mut parser = RequestParser::new(limits);
    assert_eq!(parser.feed(&endless), Err(E::RequestLineTooLong));
}

#[test]
fn the_header_bytes_limit_passes_at_the_limit_and_fails_one_byte_over() {
    let limits = small();
    let make = |total: usize| {
        // The header block is `Host: h\r\n` (9), `X: <fill>\r\n` and the final `\r\n` (2).
        let fill = "v".repeat(total - 9 - 2 - 5);
        format!("GET / HTTP/1.1\r\nHost: h\r\nX: {fill}\r\n\r\n").into_bytes()
    };
    let (events, error) = every_split_request(limits, &make(limits.max_header_bytes));
    assert_eq!((events.len(), error), (1, None));
    let over = make(limits.max_header_bytes + 1);
    let (events, error) = every_split_request(limits, &over);
    assert_eq!((events.len(), error), (0, Some(E::HeadersTooLarge)));
    assert_eq!(E::HeadersTooLarge.status(), 431);
    let start = "GET / HTTP/1.1\r\n".len();
    assert_eq!(
        first_failure(limits, &over),
        Some((start + limits.max_header_bytes, E::HeadersTooLarge))
    );
    // An endless header line with no end stops at the limit as well.
    let mut endless = b"GET / HTTP/1.1\r\nX: ".to_vec();
    endless.extend(std::iter::repeat_n(b'a', 1000));
    assert_eq!(
        first_failure(limits, &endless),
        Some((start + limits.max_header_bytes, E::HeadersTooLarge))
    );
}

#[test]
fn the_header_count_limit_passes_at_the_limit_and_fails_one_over() {
    let limits = small();
    let make = |n: usize| {
        let mut s = String::from("GET / HTTP/1.1\r\nHost: h\r\n");
        for i in 1..n {
            s.push_str(&format!("X{i}: v\r\n"));
        }
        s.push_str("\r\n");
        s.into_bytes()
    };
    let (events, error) = every_split_request(limits, &make(limits.max_headers));
    assert_eq!((events.len(), error), (1, None));
    let (events, error) = every_split_request(limits, &make(limits.max_headers + 1));
    assert_eq!((events.len(), error), (0, Some(E::TooManyHeaders)));
    assert_eq!(E::TooManyHeaders.status(), 431);
}

#[test]
fn the_body_limit_passes_at_the_limit_and_fails_one_over_before_a_byte_is_buffered() {
    let limits = small();
    let head = |n: usize| {
        format!("POST / HTTP/1.1\r\nHost: h\r\nContent-Length: {n}\r\n\r\n").into_bytes()
    };
    let mut at = head(limits.max_body);
    at.extend(std::iter::repeat_n(b'x', limits.max_body));
    let (events, error) = every_split_request(limits, &at);
    assert_eq!(error, None);
    assert!(matches!(&events[1], Event::Message(r) if r.body.len() == limits.max_body));
    let over = head(limits.max_body + 1);
    let mut parser = RequestParser::new(limits);
    assert_eq!(parser.feed(&over), Err(E::BodyTooLarge));
    assert_eq!(parser.buffered(), 0);
    assert_eq!(E::BodyTooLarge.status(), 413);
    // The failure is at the end of the head: the body was never asked for.
    assert_eq!(
        first_failure(limits, &over),
        Some((over.len() - 1, E::BodyTooLarge))
    );
    // The held body never exceeds the limit while it arrives.
    let mut parser = RequestParser::new(limits);
    let (used, _) = parser.feed(&at).unwrap();
    for b in &at[used..at.len() - 1] {
        parser.feed(&[*b]).unwrap();
        assert!(
            parser.buffered() <= limits.max_header_bytes + limits.max_start_line + limits.max_body
        );
    }
}

// ---- responses --------------------------------------------------------------------------------------------

fn resp(status: u16, reason: &str, headers: &[(&str, &str)], body: &[u8]) -> Response {
    let mut r = Response::new(status, reason);
    for (n, v) in headers {
        r.headers.append(*n, *v);
    }
    r.body = body.to_vec();
    r
}

#[test]
fn good_responses_parse_to_the_expected_message_at_every_split() {
    let table: Vec<(&str, Vec<u8>, bool, Response)> = vec![
        (
            "a 200 with a body",
            b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\nX-A: b\r\n\r\nhello".to_vec(),
            false,
            resp(
                200,
                "OK",
                &[("Content-Length", "5"), ("X-A", "b")],
                b"hello",
            ),
        ),
        (
            "an empty reason",
            b"HTTP/1.1 200 \r\nContent-Length: 0\r\n\r\n".to_vec(),
            false,
            resp(200, "", &[("Content-Length", "0")], b""),
        ),
        (
            "a reason with spaces",
            b"HTTP/1.1 404 Not Found Here\r\nContent-Length: 1\r\n\r\nx".to_vec(),
            false,
            resp(404, "Not Found Here", &[("Content-Length", "1")], b"x"),
        ),
        (
            "HTTP/1.0",
            b"HTTP/1.0 200 OK\r\nContent-Length: 0\r\n\r\n".to_vec(),
            false,
            Response {
                version: Version::Http10,
                ..resp(200, "OK", &[("Content-Length", "0")], b"")
            },
        ),
        (
            "a 204 with no length",
            b"HTTP/1.1 204 No Content\r\n\r\n".to_vec(),
            false,
            resp(204, "No Content", &[], b""),
        ),
        (
            "a 304 with a length and no body",
            b"HTTP/1.1 304 Not Modified\r\nContent-Length: 10\r\n\r\n".to_vec(),
            false,
            resp(304, "Not Modified", &[("Content-Length", "10")], b""),
        ),
        (
            "a 100",
            b"HTTP/1.1 100 Continue\r\n\r\n".to_vec(),
            false,
            resp(100, "Continue", &[], b""),
        ),
        (
            "an answer to HEAD with a length and no body",
            b"HTTP/1.1 200 OK\r\nContent-Length: 10\r\n\r\n".to_vec(),
            true,
            resp(200, "OK", &[("Content-Length", "10")], b""),
        ),
        (
            "an answer to HEAD with neither length nor coding",
            b"HTTP/1.1 200 OK\r\n\r\n".to_vec(),
            true,
            resp(200, "OK", &[], b""),
        ),
    ];
    for (name, input, to_head, expected) in table {
        let (events, error) = every_split_response(Limits::default(), to_head, &input);
        assert_eq!(error, None, "{name}");
        let message = events.into_iter().find_map(|e| match e {
            Event::Message(m) => Some(m),
            Event::Head { .. } => None,
        });
        assert_eq!(message, Some(expected), "{name}");
    }
}

#[test]
fn two_responses_in_a_row_come_out_whole_at_every_split() {
    let input = b"HTTP/1.1 100 Continue\r\n\r\nHTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nokHTTP/1.1 204 No Content\r\n\r\n";
    let (events, error) = every_split_response(Limits::default(), false, input);
    assert_eq!(error, None);
    let statuses: Vec<u16> = events
        .iter()
        .filter_map(|e| match e {
            Event::Message(m) => Some(m.status),
            Event::Head { .. } => None,
        })
        .collect();
    assert_eq!(statuses, [100, 200, 204]);
}

#[test]
fn bad_responses_fail_with_the_exact_error_at_every_split() {
    let table: Vec<(&str, &[u8], ParseError)> = vec![
        (
            "no length",
            b"HTTP/1.1 200 OK\r\n\r\n",
            E::MissingContentLength,
        ),
        (
            "a chunked body",
            b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n",
            E::TransferEncodingNotSupported,
        ),
        (
            "coding and length",
            b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nContent-Length: 1\r\n\r\n",
            E::TransferEncodingWithContentLength,
        ),
        (
            "differing lengths",
            b"HTTP/1.1 200 OK\r\nContent-Length: 1\r\nContent-Length: 2\r\n\r\n",
            E::ConflictingContentLength,
        ),
        (
            "a signed length",
            b"HTTP/1.1 200 OK\r\nContent-Length: +1\r\n\r\n",
            E::BadContentLength,
        ),
        (
            "a length over the limit",
            b"HTTP/1.1 200 OK\r\nContent-Length: 2000000\r\n\r\n",
            E::BodyTooLarge,
        ),
        (
            "a two-digit status",
            b"HTTP/1.1 20 OK\r\nContent-Length: 0\r\n\r\n",
            E::BadStatusLine,
        ),
        (
            "a four-digit status",
            b"HTTP/1.1 2000 OK\r\nContent-Length: 0\r\n\r\n",
            E::BadStatusLine,
        ),
        (
            "status 600",
            b"HTTP/1.1 600 X\r\nContent-Length: 0\r\n\r\n",
            E::BadStatusLine,
        ),
        (
            "status 099",
            b"HTTP/1.1 099 X\r\nContent-Length: 0\r\n\r\n",
            E::BadStatusLine,
        ),
        (
            "a signed status",
            b"HTTP/1.1 +20 X\r\nContent-Length: 0\r\n\r\n",
            E::BadStatusLine,
        ),
        (
            "no space after the status",
            b"HTTP/1.1 200\r\nContent-Length: 0\r\n\r\n",
            E::BadStatusLine,
        ),
        (
            "no status",
            b"HTTP/1.1\r\nContent-Length: 0\r\n\r\n",
            E::BadStatusLine,
        ),
        (
            "a bad version",
            b"http/1.1 200 OK\r\nContent-Length: 0\r\n\r\n",
            E::BadStatusLine,
        ),
        (
            "version 2",
            b"HTTP/2.0 200 OK\r\nContent-Length: 0\r\n\r\n",
            E::UnsupportedVersion,
        ),
        (
            "a control byte in the reason",
            b"HTTP/1.1 200 O\x01K\r\nContent-Length: 0\r\n\r\n",
            E::BadStatusLine,
        ),
        (
            "an empty line first",
            b"\r\nHTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n",
            E::BadStatusLine,
        ),
        (
            "a bare LF",
            b"HTTP/1.1 200 OK\nContent-Length: 0\r\n\r\n",
            E::BadLineEnding,
        ),
        (
            "a folded line",
            b"HTTP/1.1 200 OK\r\nX: a\r\n b\r\nContent-Length: 0\r\n\r\n",
            E::ObsoleteFolding,
        ),
        (
            "a space before the colon",
            b"HTTP/1.1 200 OK\r\nContent-Length : 0\r\n\r\n",
            E::SpaceBeforeColon,
        ),
    ];
    for (name, input, error) in table {
        let (events, got) = every_split_response(Limits::default(), false, input);
        assert_eq!(got, Some(error), "{name}");
        assert!(events.is_empty(), "{name}");
    }
}

#[test]
fn the_response_limits_pass_at_the_limit_and_fail_one_over() {
    let limits = small();
    let line = |total: usize| {
        let reason = "r".repeat(total - "HTTP/1.1 200 \r\n".len());
        format!("HTTP/1.1 200 {reason}\r\nContent-Length: 0\r\n\r\n").into_bytes()
    };
    let (events, error) = every_split_response(limits, false, &line(limits.max_start_line));
    assert_eq!((events.len(), error), (1, None));
    let (_, error) = every_split_response(limits, false, &line(limits.max_start_line + 1));
    assert_eq!(error, Some(E::StatusLineTooLong));
    let headers = |total: usize| {
        let fill = "v".repeat(total - "Content-Length: 0\r\n".len() - 2 - 5);
        format!("HTTP/1.1 200 OK\r\nContent-Length: 0\r\nX: {fill}\r\n\r\n").into_bytes()
    };
    let (events, error) = every_split_response(limits, false, &headers(limits.max_header_bytes));
    assert_eq!((events.len(), error), (1, None));
    let (_, error) = every_split_response(limits, false, &headers(limits.max_header_bytes + 1));
    assert_eq!(error, Some(E::HeadersTooLarge));
    let many = |n: usize| {
        let mut s = String::from("HTTP/1.1 200 OK\r\nContent-Length: 0\r\n");
        for i in 1..n {
            s.push_str(&format!("X{i}: v\r\n"));
        }
        s.push_str("\r\n");
        s.into_bytes()
    };
    let (_, error) = every_split_response(limits, false, &many(limits.max_headers));
    assert_eq!(error, None);
    let (_, error) = every_split_response(limits, false, &many(limits.max_headers + 1));
    assert_eq!(error, Some(E::TooManyHeaders));
    let body = |n: usize| {
        let mut v = format!("HTTP/1.1 200 OK\r\nContent-Length: {n}\r\n\r\n").into_bytes();
        v.extend(std::iter::repeat_n(b'b', n));
        v
    };
    let (events, error) = every_split_response(limits, false, &body(limits.max_body));
    assert_eq!((events.len(), error), (2, None));
    let (events, error) = every_split_response(limits, false, &body(limits.max_body + 1));
    assert_eq!((events.len(), error), (0, Some(E::BodyTooLarge)));
}

// ---- the writer -------------------------------------------------------------------------------------------

#[test]
fn a_written_response_parses_back_to_the_same_message() {
    let cases = [
        Response::with_status(200)
            .header("X-A", "b")
            .header("x-a", "c")
            .with_body("hello"),
        Response::with_status(404).with_body(Vec::new()),
        Response::new(200, "").with_body(vec![0, 1, 2, 255, b'\r', b'\n']),
        Response::with_status(204),
        Response::with_status(304).header("ETag", "\"x\""),
        Response::with_status(100),
    ];
    for original in cases {
        for close in [false, true] {
            let mut out = Vec::new();
            write_response(&original, close, &mut out).unwrap();
            let (events, error) = every_split_response(Limits::default(), false, &out);
            assert_eq!(error, None);
            let Some(Event::Message(parsed)) = events.into_iter().last() else {
                panic!("a response")
            };
            assert_eq!(parsed.keep_alive(), !close);
            let mut expected = original.clone();
            if !status_has_no_body(original.status) {
                expected
                    .headers
                    .append("Content-Length", original.body.len().to_string());
            }
            if close {
                expected.headers.append("Connection", "close");
            }
            assert_eq!(parsed, expected);
        }
    }
}

#[test]
fn a_written_request_parses_back_to_the_same_message() {
    let cases = [
        Request::new(Method::Get, "/a?b=c").header("Host", "h"),
        Request::new(Method::Post, "/p")
            .header("Host", "h")
            .with_body("data"),
        Request::new(Method::Post, "/p").header("Host", "h"),
        Request::new(Method::Other("PURGE".into()), "http://h/x")
            .header("Host", "h")
            .header("X", "1"),
        Request::new(Method::Put, "/p")
            .header("Host", "h")
            .with_body(vec![0, 13, 10, 200]),
    ];
    for original in cases {
        for close in [false, true] {
            let mut out = Vec::new();
            write_request(&original, close, &mut out).unwrap();
            let parsed = one_request(&out);
            assert_eq!(parsed.keep_alive(), !close);
            let mut expected = original.clone();
            let has_length =
                !original.body.is_empty() || matches!(original.method, Method::Post | Method::Put);
            if has_length {
                expected
                    .headers
                    .append("Content-Length", original.body.len().to_string());
            }
            if close {
                expected.headers.append("Connection", "close");
            }
            assert_eq!(parsed, expected);
        }
    }
}

#[test]
fn the_writer_refuses_what_could_split_a_message() {
    let ok = || Response::with_status(200);
    let bad_responses: Vec<(&str, Response, WriteError)> = vec![
        (
            "a CR in a value",
            ok().header("X", "a\rb"),
            WriteError::BadHeaderValue,
        ),
        (
            "a LF in a value",
            ok().header("X", "a\nSet-Cookie: x"),
            WriteError::BadHeaderValue,
        ),
        (
            "a NUL in a value",
            ok().header("X", "a\0"),
            WriteError::BadHeaderValue,
        ),
        (
            "a leading space in a value",
            ok().header("X", " a"),
            WriteError::BadHeaderValue,
        ),
        (
            "a trailing tab in a value",
            ok().header("X", "a\t"),
            WriteError::BadHeaderValue,
        ),
        (
            "a non-ASCII value",
            ok().header("X", "\u{e9}"),
            WriteError::BadHeaderValue,
        ),
        (
            "a LF in a name",
            ok().header("X\nY", "a"),
            WriteError::BadHeaderName,
        ),
        (
            "a colon in a name",
            ok().header("X:Y", "a"),
            WriteError::BadHeaderName,
        ),
        (
            "an empty name",
            ok().header("", "a"),
            WriteError::BadHeaderName,
        ),
        (
            "a CRLF in the reason",
            Response::new(200, "OK\r\nX: y"),
            WriteError::BadReason,
        ),
        (
            "a status of 99",
            Response::new(99, "x"),
            WriteError::BadStatus,
        ),
        (
            "a status of 600",
            Response::new(600, "x"),
            WriteError::BadStatus,
        ),
        (
            "a Content-Length of the caller",
            ok().header("content-length", "1"),
            WriteError::ReservedHeader,
        ),
        (
            "a Transfer-Encoding of the caller",
            ok().header("Transfer-Encoding", "chunked"),
            WriteError::ReservedHeader,
        ),
        (
            "a Connection of the caller",
            ok().header("Connection", "close"),
            WriteError::ReservedHeader,
        ),
        (
            "a body on a 204",
            Response::with_status(204).with_body("x"),
            WriteError::BodyNotAllowed,
        ),
        (
            "a body on a 100",
            Response::with_status(100).with_body("x"),
            WriteError::BodyNotAllowed,
        ),
    ];
    for (name, response, error) in bad_responses {
        let mut out = b"kept".to_vec();
        assert_eq!(
            write_response(&response, false, &mut out),
            Err(error),
            "{name}"
        );
        assert_eq!(out, b"kept", "{name}: the output changed");
    }
    let good = || Request::new(Method::Get, "/").header("Host", "h");
    let bad_requests: Vec<(&str, Request, WriteError)> = vec![
        (
            "a CRLF in the target",
            Request::new(Method::Get, "/a\r\nX: y"),
            WriteError::BadTarget,
        ),
        (
            "a space in the target",
            Request::new(Method::Get, "/a b"),
            WriteError::BadTarget,
        ),
        (
            "an empty target",
            Request::new(Method::Get, ""),
            WriteError::BadTarget,
        ),
        (
            "a fragment in the target",
            Request::new(Method::Get, "/a#b"),
            WriteError::BadTarget,
        ),
        (
            "a space in the method",
            Request::new(Method::Other("G T".into()), "/"),
            WriteError::BadMethod,
        ),
        (
            "an empty method",
            Request::new(Method::Other(String::new()), "/"),
            WriteError::BadMethod,
        ),
        (
            "a CR in a value",
            good().header("X", "a\r"),
            WriteError::BadHeaderValue,
        ),
        (
            "a LF in a name",
            good().header("X\n", "a"),
            WriteError::BadHeaderName,
        ),
        (
            "a Content-Length of the caller",
            good().header("Content-Length", "1"),
            WriteError::ReservedHeader,
        ),
    ];
    for (name, request, error) in bad_requests {
        let mut out = b"kept".to_vec();
        assert_eq!(
            write_request(&request, false, &mut out),
            Err(error),
            "{name}"
        );
        assert_eq!(out, b"kept", "{name}: the output changed");
    }
}

#[test]
fn the_writer_output_is_exact() {
    let mut out = Vec::new();
    write_response(
        &Response::with_status(200).header("X", "y").with_body("ab"),
        true,
        &mut out,
    )
    .unwrap();
    assert_eq!(
        out,
        b"HTTP/1.1 200 OK\r\nX: y\r\nContent-Length: 2\r\nConnection: close\r\n\r\nab"
    );
    let mut out = Vec::new();
    write_continue(&mut out);
    assert_eq!(out, b"HTTP/1.1 100 Continue\r\n\r\n");
    let mut out = Vec::new();
    write_request(
        &Request::new(Method::Post, "/p").header("Host", "h"),
        false,
        &mut out,
    )
    .unwrap();
    assert_eq!(
        out,
        b"POST /p HTTP/1.1\r\nHost: h\r\nContent-Length: 0\r\n\r\n"
    );
    // The interim response parses as a response with no body.
    let (events, error) =
        every_split_response(Limits::default(), false, b"HTTP/1.1 100 Continue\r\n\r\n");
    assert_eq!((events.len(), error), (1, None));
}

// ---- the message types ------------------------------------------------------------------------------------

#[test]
fn headers_keep_order_and_repeats_and_look_up_without_case() {
    let mut h = Headers::new();
    h.append("Accept", "a");
    h.append("X", "1");
    h.append("ACCEPT", "b, Upgrade");
    assert_eq!(h.get("accept"), Some("a"));
    assert_eq!(h.get_all("Accept").collect::<Vec<_>>(), ["a", "b, Upgrade"]);
    assert!(h.contains("x") && !h.contains("y"));
    assert!(h.has_token("accept", "upgrade") && !h.has_token("accept", "up"));
    assert_eq!(
        h.iter().map(|(n, _)| n).collect::<Vec<_>>(),
        ["Accept", "X", "ACCEPT"]
    );
    assert_eq!((h.len(), h.is_empty()), (3, false));
}

#[test]
fn methods_and_errors_report_their_names_and_statuses() {
    for m in [
        "GET", "HEAD", "POST", "PUT", "DELETE", "CONNECT", "OPTIONS", "TRACE", "PATCH", "X",
    ] {
        assert_eq!(Method::from_token(m).as_str(), m);
    }
    assert_eq!(Method::from_token("get"), Method::Other("get".into()));
    let statuses: Vec<u16> = [
        E::BadRequestLine,
        E::ChunkedNotSupported,
        E::BodyTooLarge,
        E::RequestLineTooLong,
        E::HeadersTooLarge,
        E::TransferEncodingNotSupported,
        E::UnsupportedVersion,
    ]
    .iter()
    .map(|e| e.status())
    .collect();
    assert_eq!(statuses, [400, 411, 413, 414, 431, 501, 505]);
    assert!(E::BodyTooLarge.to_string().contains("413"));
}

// ---- arbitrary and mutated bytes ----------------------------------------------------------------------------

/// xorshift64*: a fixed-seed generator, no dependency.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n.max(1) as u64) as usize
    }
}

const PIECES: [&[u8]; 14] = [
    b"\r\n",
    b"\n",
    b"\r",
    b" ",
    b"\t",
    b": ",
    b":",
    b"Content-Length: 5\r\n",
    b"Transfer-Encoding: chunked\r\n",
    b"Host: h\r\n",
    b"HTTP/1.1",
    b"\0",
    b"\xff",
    b"Expect: 100-continue\r\n",
];

fn mutate(rng: &mut Rng, seed: &[u8]) -> Vec<u8> {
    let mut v = seed.to_vec();
    for _ in 0..1 + rng.below(4) {
        let at = rng.below(v.len() + 1);
        match rng.below(6) {
            0 if !v.is_empty() => {
                let i = at.min(v.len() - 1);
                v[i] = rng.next() as u8;
            }
            1 => v.insert(at, rng.next() as u8),
            2 if !v.is_empty() => {
                v.remove(at.min(v.len() - 1));
            }
            3 => v.truncate(at),
            4 => {
                let piece = PIECES[rng.below(PIECES.len())];
                v.splice(at..at, piece.iter().copied());
            }
            _ => {
                let end = (at + rng.below(20)).min(v.len());
                let copy = v[at.min(end)..end].to_vec();
                v.splice(at..at, copy);
            }
        }
    }
    v
}

/// What every request the parser yields must satisfy, whatever the input was.
fn check_request(request: &Request, limits: &Limits) {
    assert!(!request.target.is_empty());
    assert!(
        request
            .target
            .bytes()
            .all(|b| (0x21..=0x7e).contains(&b) && b != b'#')
    );
    assert!(!request.method.as_str().is_empty());
    assert!(request.body.len() <= limits.max_body);
    let mut hosts = 0;
    let mut length: Option<usize> = None;
    for (name, value) in request.headers.iter() {
        assert!(!name.is_empty());
        assert!(
            name.bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"!#$%&'*+-.^_`|~".contains(&b))
        );
        assert!(
            value
                .bytes()
                .all(|b| b == b'\t' || (0x20..=0x7e).contains(&b))
        );
        assert!(!value.starts_with([' ', '\t']) && !value.ends_with([' ', '\t']));
        assert!(!name.eq_ignore_ascii_case("transfer-encoding"));
        if name.eq_ignore_ascii_case("host") {
            hosts += 1;
        }
        if name.eq_ignore_ascii_case("content-length") {
            for element in value.split(',') {
                let n: usize = element.trim().parse().expect("digits");
                assert!(element.trim().bytes().all(|b| b.is_ascii_digit()));
                assert_eq!(*length.get_or_insert(n), n);
            }
        }
    }
    assert_eq!(request.body.len(), length.unwrap_or(0));
    if request.version == Version::Http11 {
        assert_eq!(hosts, 1);
    }
}

#[test]
fn arbitrary_and_mutated_bytes_never_panic_and_never_pass_the_strict_tables() {
    let limits = Limits {
        max_start_line: 200,
        max_header_bytes: 400,
        max_headers: 20,
        max_body: 100,
    };
    let rounds = if cfg!(miri) { 12 } else { 3000 };
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut seeds: Vec<Vec<u8>> = vec![
        b"POST /a HTTP/1.1\r\nHost: h\r\nContent-Length: 5\r\nExpect: 100-continue\r\n\r\nhelloGET / HTTP/1.0\r\n\r\n".to_vec(),
        b"GET /a?b HTTP/1.1\r\nHost: h\r\nX-A: b\r\nConnection: close\r\n\r\n".to_vec(),
    ];
    seeds.extend(bad_requests().into_iter().map(|(_, input, _, _)| input));
    let mut yielded = 0;
    for round in 0..rounds {
        let input = if round % 5 == 0 {
            (0..rng.below(300)).map(|_| rng.next() as u8).collect()
        } else {
            let seed = &seeds[rng.below(seeds.len())];
            mutate(&mut rng, seed)
        };
        let whole = requests(limits, &input, &[]);
        let a = rng.below(input.len() + 1);
        let b = rng.below(input.len() + 1);
        let (lo, hi) = (a.min(b), a.max(b));
        assert_eq!(requests(limits, &input, &[lo, hi]), whole);
        if round % 4 == 0 {
            let all: Vec<usize> = (0..=input.len()).collect();
            assert_eq!(requests(limits, &input, &all), whole);
        }
        for event in &whole.0 {
            if let Event::Message(request) = event {
                check_request(request, &limits);
                yielded += 1;
            }
        }
        // The response side gets the same bytes.
        let whole = responses(limits, false, &input, &[]);
        assert_eq!(responses(limits, false, &input, &[lo, hi]), whole);
    }
    if !cfg!(miri) {
        assert!(yielded > 0, "the mutations never produced a valid request");
    }
}

#[test]
fn bad_table_inputs_never_give_a_request_at_random_splits() {
    let rounds = if cfg!(miri) { 2 } else { 50 };
    let mut rng = Rng(42);
    for (name, input, error, _) in bad_requests() {
        for _ in 0..rounds {
            let mut cuts: Vec<usize> = (0..rng.below(6))
                .map(|_| rng.below(input.len() + 1))
                .collect();
            cuts.sort_unstable();
            let (events, got) = requests(Limits::default(), &input, &cuts);
            assert_eq!(got, Some(error), "{name}");
            assert!(events.is_empty(), "{name}");
        }
    }
}
