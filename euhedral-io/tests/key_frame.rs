//! `AsyncIo` hands out in `request` and `pull`; reaping is a frame's work.
//!
//! `request` and `pull` do no heavy or high-frequency work: a worker that executes a frame can be measured, and
//! the workers learn the shape of the workload from the frames they execute. Reaping asks the kernel (a system call
//! and a lock) at every visit, so `request` and `pull` only pass on what was already reaped, and push one key frame whose `execute` reaps
//! (looks at the kernel's completions and calls each operation's continuation). The tests play the
//! worker: they keep what the source pushes and run it.
#![allow(
    clippy::disallowed_types,
    reason = "test fixtures: the test thread locks its records"
)]
#![cfg(all(feature = "async-io", any(target_os = "linux", target_os = "android")))]
// Fixtures write files synchronously, outside any lattice.
#![allow(clippy::disallowed_methods)]

use std::cell::Cell;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use euhedral_core::frames::{Failure, Frame, FrameHeader};
use euhedral_core::generics::{LatticeReceiver, LatticeSource, LatticeTerminal};
use euhedral_io::{AsyncIo, IoBuf};

thread_local! {
    /// This thread is executing a frame: what a lattice worker is doing when it runs one.
    static IN_FRAME: Cell<bool> = const { Cell::new(false) };
}

/// A receiver that keeps what the source pushes, to be run by the test as a worker would.
#[derive(Default)]
struct Keep {
    frames: Mutex<Vec<Box<dyn Frame>>>,
}

impl Keep {
    /// Runs every frame kept, as a worker's executor does, and returns how many.
    fn run(&self) -> usize {
        let frames = std::mem::take(&mut *self.frames.lock().unwrap());
        let n = frames.len();
        for mut frame in frames {
            IN_FRAME.set(true);
            let ran = frame.execute();
            IN_FRAME.set(false);
            ran.unwrap();
            frame.do_finally();
        }
        n
    }

    fn kept(&self) -> usize {
        self.frames.lock().unwrap().len()
    }
}

impl LatticeTerminal for Keep {
    fn add_upstream(&self, _upstream: Arc<dyn LatticeSource>) {}
}

impl LatticeReceiver for Keep {
    fn push(&self, frame: Box<dyn Frame>) {
        self.frames.lock().unwrap().push(frame);
    }

    fn on_complete(&self) {}

    fn on_error(&self, _error: Failure) {}
}

/// The continuation's frame: it has no work, it only shows the continuation ran.
struct Ended(FrameHeader);

impl Frame for Ended {
    fn header(&self) -> &FrameHeader {
        &self.0
    }

    fn header_mut(&mut self) -> &mut FrameHeader {
        &mut self.0
    }
}

struct TempFile(PathBuf);

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// A file of `content` in a scratch directory under the target directory.
fn temp_file(name: &str, content: &[u8]) -> (TempFile, std::fs::File) {
    let exe = std::env::current_exe().expect("the test binary's path");
    // The test binary is `target/<profile>/deps/<name>`: scratch goes beside it, never /tmp or $HOME.
    let dir = exe
        .parent()
        .and_then(std::path::Path::parent)
        .expect("a target directory")
        .join("euhedral-io-scratch");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("{name}-{}", std::process::id()));
    std::fs::write(&path, content).unwrap();
    let file = std::fs::File::open(&path).unwrap();
    (TempFile(path), file)
}

#[test]
fn a_request_and_a_pull_reap_nothing_and_the_key_frame_does() {
    use std::os::fd::AsRawFd;
    let (_guard, file) = temp_file("key", &[7; 4096]);
    let io = AsyncIo::open().unwrap();
    let keep = Arc::new(Keep::default());
    io.clone().add_downstream(keep.clone());

    // The continuation is called by whoever reaps: it must be a frame's execution, never a request or a pull.
    let continued = Arc::new(AtomicUsize::new(0));
    let c = continued.clone();
    io.read(
        file.as_raw_fd(),
        0,
        IoBuf {
            bytes: vec![0; 64],
            filled: 0,
        },
        Box::new(move |done| {
            assert!(
                IN_FRAME.get(),
                "a continuation was called outside a frame's execution (in a request or a pull)"
            );
            assert_eq!(done.result.unwrap(), 64);
            c.fetch_add(1, Ordering::AcqRel);
            Box::new(Ended(FrameHeader::new(1)))
        }),
    )
    .unwrap();
    assert_eq!(io.pending(), 1);

    let deadline = Instant::now() + Duration::from_secs(10);
    let mut delivered = 0;
    while delivered == 0 {
        assert!(Instant::now() < deadline, "the read did not complete");
        // Neither call reaps: whatever the kernel has finished stays in its queue.
        io.request(8);
        let mut pulled = 0;
        io.pull(&mut |_| pulled += 1, &mut |_| false, 8);
        assert_eq!(
            continued.load(Ordering::Acquire) - delivered,
            0,
            "a request or a pull reaped"
        );
        // The frames the request pushed are run by the "worker": the key reaps, and what it reaped is
        // handed out by the requests that follow.
        if keep.run() > 0 {
            io.request(8);
            delivered += keep.run();
        }
        delivered += pulled;
    }
    assert_eq!(continued.load(Ordering::Acquire), 1);
    assert_eq!(io.pending(), 0);
}

#[test]
fn no_key_frame_is_pushed_while_nothing_is_in_flight() {
    let io = AsyncIo::open().unwrap();
    let keep = Arc::new(Keep::default());
    io.clone().add_downstream(keep.clone());
    for _ in 0..4 {
        io.request(8);
    }
    assert_eq!(keep.kept(), 0, "the absence of work is the signal");
}

#[test]
fn an_operation_that_never_ends_is_reaped_ever_less_often() {
    use std::io::Write;
    use std::os::fd::AsRawFd;
    let (ours, theirs) = std::os::unix::net::UnixStream::pair().unwrap();
    let io = AsyncIo::open().unwrap();
    let keep = Arc::new(Keep::default());
    io.clone().add_downstream(keep.clone());
    io.recv(
        ours.as_raw_fd(),
        IoBuf {
            bytes: vec![0; 16],
            filled: 0,
        },
        Box::new(|done| {
            assert_eq!(done.result.unwrap(), 4);
            Box::new(Ended(FrameHeader::new(3)))
        }),
    )
    .unwrap();
    // A worker that finds nothing parks only if the handle it visits reads unproductive: a visit that pushes a
    // frame does not. Between reaps a visit pushes nothing.
    let started = Instant::now();
    let mut visits = 0_u32;
    let mut pushes = 0;
    while started.elapsed() < Duration::from_millis(30) {
        if visits.is_multiple_of(2) {
            io.pull(&mut |_| panic!("nothing ended"), &mut |_| false, 8);
        } else {
            io.request(8);
        }
        visits += 1;
        pushes += keep.run();
    }
    let elapsed = started.elapsed().as_micros() as usize;
    assert!(pushes >= 4, "it is still reaped ({pushes})");
    // The window between reaps grows from 16 us to 100 us, whatever the visit rate: after the first four, one per
    // 100 us of the run at most.
    assert!(
        pushes <= 6 + elapsed / 100,
        "{pushes} reaps in {elapsed} us and {visits} visits: the back-off did not grow"
    );
    // The peer sends: a new operation resets the back-off, and the data is found within the window.
    (&theirs).write_all(b"ping").unwrap();
    let deadline = Instant::now() + Duration::from_secs(10);
    while io.pending() > 0 {
        assert!(Instant::now() < deadline, "the receive did not complete");
        io.pull(&mut |_| {}, &mut |_| false, 8);
        io.request(8);
        keep.run();
    }
}

#[test]
fn a_continuation_that_panics_loses_its_operation_and_not_the_others() {
    use std::os::fd::AsRawFd;
    let (_guard, file) = temp_file("panic", &[7; 4096]);
    let io = AsyncIo::open().unwrap();
    let buf = || IoBuf {
        bytes: vec![0; 64],
        filled: 0,
    };
    let ran = Arc::new(AtomicUsize::new(0));
    io.read(
        file.as_raw_fd(),
        0,
        buf(),
        Box::new(|_| panic!("a continuation failed")),
    )
    .unwrap();
    let r = ran.clone();
    io.read(
        file.as_raw_fd(),
        0,
        buf(),
        Box::new(move |_| {
            r.fetch_add(1, Ordering::AcqRel);
            Box::new(Ended(FrameHeader::new(2)))
        }),
    )
    .unwrap();
    let deadline = Instant::now() + Duration::from_secs(10);
    while io.pending() > 0 {
        assert!(Instant::now() < deadline, "the operations did not drain");
        // The reap does not unwind: the panicking operation's frame is one that does nothing, and it is counted.
        io.reap();
        io.pull(&mut |_| {}, &mut |_| false, 8);
    }
    assert_eq!(ran.load(Ordering::Acquire), 1);
    assert_eq!(io.panicked(), 1);
}

#[test]
fn a_pull_pushes_and_hands_out_no_key_a_request_pushes_it_and_a_key_out_is_never_out_twice() {
    use std::os::fd::AsRawFd;
    let (ours, _theirs) = std::os::unix::net::UnixStream::pair().unwrap();
    let io = AsyncIo::open().unwrap();
    let keep = Arc::new(Keep::default());
    io.clone().add_downstream(keep.clone());
    io.recv(
        ours.as_raw_fd(),
        IoBuf {
            bytes: vec![0; 16],
            filled: 0,
        },
        Box::new(|_| Box::new(Ended(FrameHeader::new(3)))),
    )
    .unwrap();
    // The reap is wanted (a submission) and the frame is home: a pull, whatever it is asked, hands out and pushes
    // nothing. The reap is long (a system call, a lock, the application's continuations), so only a request sends it.
    for demand in [1, 8, u64::MAX] {
        for stop in [false, true] {
            assert_eq!(
                io.pull(
                    &mut |_| panic!("a key is not handed out by a pull"),
                    &mut |_| stop,
                    demand
                ),
                0
            );
        }
    }
    assert_eq!(keep.kept(), 0, "no pull pushed it");
    io.request(1);
    assert_eq!(keep.kept(), 1, "a request pushes it");
    // It is out: neither another request nor a pull sends a second.
    io.request(1);
    io.pull(
        &mut |_| panic!("a key is not handed out by a pull"),
        &mut |_| false,
        8,
    );
    assert_eq!(keep.kept(), 1);
}

#[test]
fn a_reap_that_runs_while_a_pull_holds_the_hand_out_leaves_the_pushing_to_a_request() {
    use std::os::fd::AsRawFd;
    let (_guard, file) = temp_file("held", &[7; 4096]);
    let io = AsyncIo::open().unwrap();
    let keep = Arc::new(Keep::default());
    io.clone().add_downstream(keep.clone());
    let read = |id: u64| {
        io.read(
            file.as_raw_fd(),
            0,
            IoBuf {
                bytes: vec![0; 64],
                filled: 0,
            },
            Box::new(move |_| Box::new(Ended(FrameHeader::new(id)))),
        )
        .unwrap();
    };
    // Reaps until `n` more operations have been reaped, as the key frame's worker would.
    let reap = |n: usize| {
        let deadline = Instant::now() + Duration::from_secs(10);
        let mut reaped = 0;
        while reaped < n {
            assert!(Instant::now() < deadline, "the reads did not complete");
            reaped += io.reap();
        }
    };
    for id in [1, 2, 3] {
        read(id);
    }
    reap(3);
    let mut handed = 0;
    io.pull(
        &mut |_| {
            handed += 1;
            if handed > 1 {
                return;
            }
            // The worker that ran the reap frame while this pull holds the hand-out: a request banks demand, and a
            // reap finds one more. Neither can enter the hand-out, so what they leave is the next call's.
            io.request(8);
            read(4);
            reap(1);
        },
        &mut |_| false,
        1,
    );
    assert_eq!(handed, 1);
    assert_eq!(
        keep.kept(),
        0,
        "the pull's release pushed nothing: a pull never pushes"
    );
    // The next request serves what waited, within the demand it banked.
    io.request(1);
    assert_eq!(keep.kept(), 3);
}
