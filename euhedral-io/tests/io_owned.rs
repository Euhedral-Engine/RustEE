//! `AsyncIo::read_into`: reads into memory the caller owns, through the real engine of the
//! platform (io_uring, or IoRing / a completion port on Windows).
//!
//! The files are opened by `AsyncIo::open_path` (overlapped on Windows). The tests drive the source
//! by hand, as a lattice worker's polls do, and assume nothing about the number of cores.
#![allow(
    clippy::disallowed_types,
    reason = "test fixtures: the test thread locks its records"
)]
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

/// Takes each frame its upstream pushes and records its tag.
#[derive(Default)]
struct Collector {
    tags: Mutex<Vec<u64>>,
}

impl euhedral_core::generics::LatticeTerminal for Collector {
    fn add_upstream(&self, _upstream: Arc<dyn LatticeSource>) {}
}

impl euhedral_core::generics::LatticeReceiver for Collector {
    fn push(&self, frame: Box<dyn Frame>) {
        self.tags.lock().unwrap().push(frame.header().id_hash());
    }

    fn on_complete(&self) {}

    fn on_error(&self, _error: euhedral_core::frames::Failure) {}
}

/// A file handle shared with reader threads: the file outlives them, and the engine takes the raw
/// value only to submit.
#[derive(Clone, Copy)]
struct SharedFile(RawFile);

// SAFETY: the handle is a plain value; the test keeps the owning `OwnedFile` alive past every thread.
#[allow(unsafe_code)]
unsafe impl Send for SharedFile {}
// SAFETY: as above.
#[allow(unsafe_code)]
unsafe impl Sync for SharedFile {}

impl SharedFile {
    fn get(&self) -> RawFile {
        self.0
    }
}

#[test]
fn a_request_only_lattice_delivers_every_completion() {
    const THREADS: u64 = 4;
    const EACH: u64 = 125;
    let io = AsyncIo::open().unwrap();
    let data = content(4096);
    let (_scratch, file) = open_file(&io, "request-only", &data);
    let collector = Arc::new(Collector::default());
    io.clone().add_downstream(collector.clone());
    let shared = SharedFile(raw(&file));
    std::thread::scope(|scope| {
        for thread in 0..THREADS {
            let io = &io;
            scope.spawn(move || {
                for index in 0..EACH {
                    let tag = thread * EACH + index;
                    io.read_into(
                        shared.get(),
                        0,
                        vec![SENTINEL; 64],
                        Box::new(move |_done| Ended::frame(tag)),
                    )
                    .unwrap();
                }
            });
        }
        // The driver is a lattice that only requests: no pull is ever made.
        let deadline = Instant::now() + Duration::from_secs(30);
        while collector.tags.lock().unwrap().len() < (THREADS * EACH) as usize {
            assert!(
                Instant::now() < deadline,
                "the completions did not all arrive"
            );
            io.request(8);
            std::thread::yield_now();
        }
    });
    let mut tags = collector.tags.lock().unwrap().clone();
    tags.sort_unstable();
    assert_eq!(
        tags,
        (0..THREADS * EACH).collect::<Vec<_>>(),
        "each arrived once"
    );
    assert_eq!(io.pending(), 0);
}

#[test]
fn a_request_while_a_pull_holds_the_front_returns_at_once_and_is_served_after() {
    let io = AsyncIo::open().unwrap();
    let data = content(4096);
    let (_scratch, file) = open_file(&io, "refused-request", &data);
    let collector = Arc::new(Collector::default());
    io.clone().add_downstream(collector.clone());
    let raw = raw(&file);
    for tag in 0..2u64 {
        io.read_into(
            raw,
            0,
            vec![SENTINEL; 64],
            Box::new(move |_done| Ended::frame(tag)),
        )
        .unwrap();
    }
    let inside = std::sync::Barrier::new(2);
    let release = std::sync::Barrier::new(2);
    let pulled = Mutex::new(Vec::new());
    std::thread::scope(|scope| {
        scope.spawn(|| {
            // Poll until one completion is in hand; its consumer holds the front.
            let deadline = Instant::now() + Duration::from_secs(10);
            loop {
                assert!(Instant::now() < deadline, "no read completed");
                let got = io.pull(
                    &mut |frame| {
                        pulled.lock().unwrap().push(frame.header().id_hash());
                        inside.wait();
                        release.wait();
                    },
                    &mut |_| false,
                    1,
                );
                if got == 1 {
                    break;
                }
            }
        });
        inside.wait();
        io.request(5);
        assert!(
            collector.tags.lock().unwrap().is_empty(),
            "the request did not wait for the pull, and pushed nothing yet"
        );
        release.wait();
        let deadline = Instant::now() + Duration::from_secs(10);
        while collector.tags.lock().unwrap().is_empty() {
            assert!(
                Instant::now() < deadline,
                "the other completion did not arrive"
            );
            io.request(1);
            std::thread::yield_now();
        }
    });
    let mut tags = pulled.lock().unwrap().clone();
    tags.extend(collector.tags.lock().unwrap().iter().copied());
    tags.sort_unstable();
    assert_eq!(tags, [0, 1], "each completion went to exactly one of them");
    assert_eq!(io.pending(), 0);
}
