//! HTTP/1.1 for programs built on RustEE: the message types, a strict incremental parser and a writer.
//!
//! This part of the crate does no I/O: see the module notes of [`parse`] for what is refused and why.

pub mod message;
pub mod parse;
pub mod write;

pub use message::{
    ConnectionMode, Headers, Method, ParseError, Request, Response, TargetForm, Version,
    WriteError, reason_phrase, status_has_no_body,
};
pub use parse::{Limits, PendingHead, Progress, RequestParser, ResponseParser, State};
pub use write::{write_continue, write_request, write_response, write_response_head};

/// Compiles and runs the examples in the crate README as doctests.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;

#[cfg(test)]
mod tests;
