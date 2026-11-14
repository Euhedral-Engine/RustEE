//! HTTP/1.1 for programs built on RustEE: the message types, a strict incremental parser, a writer and a server.
//!
//! The messages, the parsers and the writer do no I/O: see the module notes of [`parse`] for what is refused and
//! why. The server (`server`, on Linux, Android and Windows) runs on `AsyncIo` and a lattice.

#[cfg(any(target_os = "linux", target_os = "android", windows))]
mod connection;
pub mod message;
pub mod parse;
#[cfg(any(target_os = "linux", target_os = "android", windows))]
pub mod server;
#[cfg(any(target_os = "linux", target_os = "android", windows))]
mod station;
pub mod write;

#[cfg(any(target_os = "linux", target_os = "android", windows))]
pub use connection::Responder;
pub use message::{
    ConnectionMode, Headers, Method, ParseError, Request, Response, TargetForm, Version,
    WriteError, reason_phrase, status_has_no_body,
};
pub use parse::{Limits, PendingHead, Progress, RequestParser, ResponseParser, State};
#[cfg(any(target_os = "linux", target_os = "android", windows))]
pub use server::{Counters, Handler, ServerConfig, ServerHandle, no_work, serve};
pub use write::{write_continue, write_request, write_response, write_response_head};

/// Compiles and runs the examples in the crate README as doctests.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;

#[cfg(test)]
mod tests;
