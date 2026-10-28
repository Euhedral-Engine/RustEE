//! `AsyncIo::read_into`: reads into memory the caller owns, through the real engine of the
//! platform (io_uring, or IoRing / a completion port on Windows).
//!
//! The files are opened by `AsyncIo::open_path` (overlapped on Windows). The tests drive the source
//! by hand, as a lattice worker's polls do, and assume nothing about the number of cores.

#![cfg(all(
    feature = "async-io",
    any(target_os = "linux", target_os = "android", windows)
))]
// Fixtures write files synchronously, outside any lattice.
#![allow(clippy::disallowed_methods)]

use std::io;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use euhedral_core::frames::{Frame, FrameHeader};
use euhedral_core::generics::LatticeSource;
use euhedral_io::{AsyncIo, Create, IoDoneOwned, OpenSpec, OwnedFile, RawFile};

/// The most bytes the engine moves in one piece is 256 KiB; a read of more is cut.
const PIECE: usize = 256 * 1024;
const SENTINEL: u8 = 0xAA;

/// The frame a completion's continuation emits: the result itself goes to a slot the test reads,
/// since these frames have no work to do.
struct Ended {
    header: FrameHeader,
}

impl Ended {
    fn frame(tag: u64) -> Box<dyn Frame> {
        Box::new(Self {
            header: FrameHeader::new(tag),
        })
    }
}

impl Frame for Ended {
    fn header(&self) -> &FrameHeader {
        &self.header
    }

    fn header_mut(&mut self) -> &mut FrameHeader {
        &mut self.header
    }
}

/// Polls as a worker does (the frames have no work to run) until `pending()` is
/// zero; fails after 10 s.
fn drain(io: &AsyncIo) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while io.pending() > 0 {
        assert!(Instant::now() < deadline, "the operations did not complete");
        io.pull(&mut |frame| drop(frame), &mut |_| false, u64::MAX);
    }
}

struct Scratch(PathBuf);

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn content(len: usize) -> Vec<u8> {
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state as u8
        })
        .collect()
}

/// A file of `bytes` in cargo's scratch directory for these tests, opened through `io`.
fn open_file(io: &AsyncIo, name: &str, bytes: &[u8]) -> (Scratch, OwnedFile) {
    let path = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("io-owned-{name}-{}", std::process::id()));
    std::fs::write(&path, bytes).unwrap();
    let scratch = Scratch(path.clone());
    let slot = Arc::new(Mutex::new(None));
    let landing = slot.clone();
    io.open_path(
        &path,
        OpenSpec {
            create: Create::Never,
            ..OpenSpec::default()
        },
        Box::new(move |file| {
            *landing.lock().unwrap() = Some(file);
            Ended::frame(1)
        }),
    )
    .unwrap();
    drain(io);
    let file = slot.lock().unwrap().take().expect("the open ended");
    (scratch, file.unwrap())
}

#[cfg(unix)]
fn raw(file: &OwnedFile) -> RawFile {
    std::os::fd::AsRawFd::as_raw_fd(file)
}

#[cfg(windows)]
fn raw(file: &OwnedFile) -> RawFile {
    std::os::windows::io::AsRawHandle::as_raw_handle(file)
}

/// Reads `memory` from `file` at `offset` and returns how it ended.
fn read_into(io: &AsyncIo, file: &OwnedFile, offset: u64, memory: Vec<u8>) -> IoDoneOwned<Vec<u8>> {
    let slot = Arc::new(Mutex::new(None));
    let landing = slot.clone();
    io.read_into(
        raw(file),
        offset,
        memory,
        Box::new(move |done| {
            *landing.lock().unwrap() = Some(done);
            Ended::frame(2)
        }),
    )
    .unwrap();
    drain(io);
    slot.lock().unwrap().take().expect("the read ended")
}

#[test]
fn read_into_fills_owned_region() {
    let io = AsyncIo::open().unwrap();
    let data = content(64 * 1024);
    let (_scratch, file) = open_file(&io, "fills", &data);
    let memory = vec![SENTINEL; data.len()];
    let region = memory.as_ptr();
    let done = read_into(&io, &file, 0, memory);
    assert_eq!(done.result.unwrap(), data.len());
    let buf = done.buf.unwrap();
    assert_eq!(buf, data);
    assert_eq!(
        buf.as_ptr(),
        region,
        "the bytes landed in the caller's own region"
    );
    assert_eq!(io.pending(), 0);
}

#[test]
fn read_into_split_pieces_preserve_offsets() {
    let io = AsyncIo::open().unwrap();
    let data = content(3 * PIECE + 777);
    let (_scratch, file) = open_file(&io, "pieces", &data);
    // Larger than a piece, from the start and from an offset that is not aligned to one.
    let done = read_into(&io, &file, 0, vec![SENTINEL; data.len()]);
    assert_eq!(done.result.unwrap(), data.len());
    assert!(
        done.buf.unwrap() == data,
        "every piece landed at its own offset"
    );
    let from = 5_000;
    let done = read_into(&io, &file, from as u64, vec![SENTINEL; 2 * PIECE + 13]);
    assert_eq!(done.result.unwrap(), 2 * PIECE + 13);
    assert!(
        done.buf.unwrap() == data[from..from + 2 * PIECE + 13],
        "and so did the pieces of a read from an offset"
    );
}

#[test]
fn read_into_short_read_reports_count() {
    let io = AsyncIo::open().unwrap();
    let data = content(1000);
    let (_scratch, file) = open_file(&io, "short", &data);
    let done = read_into(&io, &file, 400, vec![SENTINEL; 4096]);
    assert_eq!(done.result.unwrap(), 600, "the file ended after 600 bytes");
    assert_eq!(done.filled, 600);
    let buf = done.buf.unwrap();
    assert_eq!(buf.len(), 4096, "the whole region comes back");
    assert_eq!(&buf[..600], &data[400..]);
    assert!(
        buf[600..].iter().all(|&b| b == SENTINEL),
        "past the end of the file the region is untouched"
    );
}

#[test]
fn owned_buffer_returned_on_error() {
    let io = AsyncIo::open().unwrap();
    let data = content(100);
    let (_scratch, file) = open_file(&io, "error", &data);
    // A read at the end of the file fails, and the region comes back unchanged.
    let done = read_into(&io, &file, 4096, vec![SENTINEL; 512]);
    assert_eq!(
        done.result.unwrap_err().kind(),
        io::ErrorKind::UnexpectedEof
    );
    assert_eq!(done.buf.unwrap(), vec![SENTINEL; 512]);
    assert_eq!(io.pending(), 0);
    // An empty region is refused at submission: nothing started, nothing pending.
    let refused = io.read_into(
        raw(&file),
        0,
        Vec::new(),
        Box::new(|_| unreachable!("nothing was started")),
    );
    assert_eq!(refused.unwrap_err().kind(), io::ErrorKind::InvalidInput);
    assert_eq!(io.pending(), 0);
}
