# euhedral-io

Asynchronous file and socket I/O on the RustEE lattice. `AsyncIo` is a `LatticeSource` of
continuations: each operation returns at once, a frame the source pushes (its reap frame) reaps the completions on a
worker, and the source hands out the continuation frames. See the [top-level glossary](../GLOSSARY.md) for the terms a user meets and
[`GLOSSARY.md`](GLOSSARY.md) for this crate's implementation-level terms.

Its operations are reads, writes, sends, receives, accepts and connects.

**Where it sits.** It depends on [`euhedral-core`](../euhedral-core) and
[`euhedral-data-structures`](../euhedral-data-structures) (`MpscQueue`), and, for the engines, on
`io-uring` and `libc` (Linux and Android) or `windows-sys` (Windows). Nothing in the workspace depends
on it; applications do. It does not re-export `euhedral-core`: import `Frame`, `LatticeSource` and the rest
from there. Like the rest of the control plane it owns no thread and nothing in it waits.

## Contents

| Item | What it is |
|---|---|
| [`AsyncIo`](src/lib.rs) | A `LatticeSource` of continuations behind file and socket operations, with `AsyncIoOptions`. |
| `IoBuf`, `IoDone`, `IoThen`, `IoMemory`, `IoDoneOwned`, `OpenSpec` (`Create`) | The buffer, the result, the continuation and the owned-memory contract of an operation. |
| `RawFile`, `RawSock`, `OwnedFile`, `OwnedSock` | The handle aliases of the platform. |
| `Continuation` | What turns a completed operation's result into the next frame. |
| [`uring`](src/uring.rs) | The io_uring engine, on Linux and Android. |
| [`win`](src/win/mod.rs) | The Windows engines: IoRing and completion ports. |

Feature `async-io` (default) builds all of it; without it, or on macOS (no engine yet), the crate is empty.
The crate's clippy bans (see [`clippy.toml`](clippy.toml)) are the control plane's: no threads, sleeps,
channels, `Condvar`, blocking `std::fs` or `std::net` calls.

## `AsyncIo`

`AsyncIo` does file and socket I/O that holds no worker: `read`, `write`, `send`, `recv`, `accept`,
`open_path` (and `open_file` with raw flags on Linux), `rename`, `send_to` and `connect` each
return at once, and the operation's `IoDone` (its buffer back, and the bytes moved or the error)
reaches the `IoThen` continuation, which builds the frame the source emits. Handles are the
platform's: descriptors on Linux and Android, `HANDLE`s and `SOCKET`s on Windows (the aliases
`RawFile`, `RawSock`, `OwnedFile` and `OwnedSock`). A read is cut into pieces so that no single
completion holds a worker for long. The `AsyncIo` type is the same everywhere; an engine under it
starts operations and finds their ends, and `src/lib.rs` does the rest.

The source side takes no lock: the continuations a pull's `stop` held back are plain state of the hand-out,
and the downstream is a `DownstreamSlot`. **Reaping is a frame's work** (it asks the kernel, takes a lock and calls the
continuations, at every visit): `request` and `pull` only hand out
what was reaped, and `request` pushes one reap frame (a `pull` pushes nothing and hands no reap frame out) while operations are in flight, whose `execute` is
`AsyncIo::reap` (a caller outside a lattice calls it before it pulls). A `request` that finds a hand-out running
banks its demand and leaves a wakeup, which the next `request` serves (a `pull` pushes nothing, so it does not serve it
as it lets go).

**Reads into memory you own.** `read_into(file, offset, buf, then)` reads straight into a region the
caller owns, such as a pinned host arena, so the bytes land in place with no copy. `buf` is any
`IoMemory`: an `unsafe` trait whose implementor promises that the region is initialized, stays put
when the value moves (and the move claims no unique ownership of it, which excludes `Box<[u8]>`), and
is exclusively the operation's from submission until `then` gets the value back in an
`IoDoneOwned { buf, result, filled }` (`Vec<u8>` implements it). The result is a `read`'s: `Ok(n)` with
`n` the region's length or less where the file ended, `UnexpectedEof` for a read that finds no bytes,
and `filled` is the bytes that arrived (`n`; after a Windows warning status, the bytes moved anyway).
The memory comes back whether the read succeeded or failed, and `then` always runs. The one case where
`buf` is `None` is a ring that broke while the kernel may still have been writing the region: the value
is leaked (never dropped) and `result` is the ring's failure. The region is cut into pieces
internally, each landing at its own offset. `read` stays as it was, for a `Vec`-backed `IoBuf`.

- **Linux and Android:** io_uring, with one ring per submitting thread. An operation belongs to the
  thread that submitted it, and lattice workers live as long as the lattice.
- **Windows:** one I/O completion port per `AsyncIo`, polled with a zero timeout, for every
  operation, and one IoRing per submitting thread for file reads and writes where the OS has a
  kernel ring for both (Windows 11 22H2 and later; not one emulated in user mode). The ring
  functions are looked up at run time, so the library still loads where they are missing. A file
  keeps its route: IoRing refuses a handle that belongs to a completion port. `AsyncIoOptions { windows_ioring: false }` forces the port.
  Windows has no asynchronous open or rename, so those run on the OS thread pool and post their
  result to the port: the one place the crate uses threads it did not create. Handles must be
  opened for overlapped I/O (`open_path` does, and so do Rust's sockets) and must not skip the port
  on success; a handle that would block or never complete is refused. A handle joins the engine's
  port and can serve one `AsyncIo`, ever. Writes go in pieces of at most 256 KiB, since a write
  that extends a file is synchronous on NTFS. Tested on a real Windows by
  [CI](../.github/workflows/windows.yml) on every push to the `graph` branch.
- **macOS:** no engine yet.

### Time limits and cancel

`send_with`, `recv_with`, `accept_with` and `connect_with` take an `OpOptions { limit: Option<Duration> }`
and return an `OpId`; `cancel(id)` asks for that operation to end. Either way the continuation runs
exactly once, the buffer comes back in it, and the socket stays usable.

```no_run
use std::time::Duration;
use euhedral_io::{AsyncIo, Cancelled, IoBuf, IoThen, LimitPassed, OpOptions, RawSock};

fn receive_for_a_second(io: &AsyncIo, socket: RawSock, then: IoThen) {
    let buf = IoBuf { bytes: vec![0; 4096], filled: 0 };
    let options = OpOptions { limit: Some(Duration::from_secs(1)) };
    let id = io.recv_with(socket, buf, options, then).unwrap();
    // Later, from any worker:
    io.cancel(id);
}

/// What a continuation makes of a result.
fn why(error: &std::io::Error) -> &'static str {
    if Cancelled::is(error) {
        "cancelled"
    } else if LimitPassed::is(error) {
        "limit passed"
    } else {
        "an error of the system"
    }
}
```

- **Three results, told apart.** A cancel ends with kind `Other` carrying `Cancelled` (not `Interrupted`,
  which means "retry": a retry loop would reissue a cancelled operation). A passed limit ends with kind
  `TimedOut` carrying `LimitPassed`, and no OS code: an `ETIMEDOUT` of the system has a code and no payload,
  and so has an operation Windows aborted (`ERROR_OPERATION_ABORTED`, which `std` also calls `TimedOut`).
  Test with `Cancelled::is`, `LimitPassed::is`, or `raw_os_error`. A kernel interruption (`EINTR`) of an
  operation that was cancelled or whose limit passed is reported as that cancel or limit.
- **The limit covers the whole operation**, every piece of a send included, counted from the call: no piece
  starts after it passed.
- **Bytes moved.** A send that ended early may have sent part of its bytes: `Cancelled::moved_of(&error)` and
  `LimitPassed::moved_of(&error)` say how many (a receive, an accept and a connect move none that they report).
  A stream on which a send ended early is no longer at a position the caller knows: close it, or resend from
  the count with the peer's agreement. A connect that ended early leaves its socket in whatever state the
  handshake had reached: close it.
- **`cancel` never blocks** and may be called from any worker. An id names its operation, the slot's
  generation and its `AsyncIo`: cancelling an operation that ended, even after its slot was reused, or
  with the id of another `AsyncIo`, does nothing. A cancel that loses a race with the completion leaves the
  real result.
- **Linux and Android:** a limit is a linked timeout (`IORING_OP_LINK_TIMEOUT`, absolute on the monotonic
  clock, so the kernel's own timer: no timer thread, no scan) and a cancel is `IORING_OP_ASYNC_CANCEL` by
  the operation's user data. It needs a kernel with linked timeouts (5.5) and absolute timeouts (5.6). A
  limited operation costs one boxed pair of entries per piece, and a ring keeps a part of its completion queue
  for cancels, so that a cancel never waits for an operation in the kernel. An operation that still waits for
  room in the queues (its piece is not in the kernel yet) ends on its cancel, or when its limit passes, without
  going in, and without any other operation completing.
- **Windows:** a cancel is `CancelIoEx` on the operation's own `OVERLAPPED`. The crate has no timer, so a
  limit is a deadline kept in the engine's table and checked at the start of each poll: it ends the operation
  at the first poll after it expires (a poll before the soonest deadline takes no lock). Limits and cancel
  apply to socket operations on the completion port.

## Build and test

```sh
mise exec -- cargo test -p euhedral-io
```

The Windows engine's state machines run on every platform against a simulated `Sys` (the seam over
the Windows calls, in `src/win/sim.rs`). Its real calls (`src/win/real.rs`) compile on Windows only;
`.github/workflows/ci.yml` runs formatting, Clippy and the tests of the whole workspace on Ubuntu and
Windows, and the workflow `.github/workflows/windows.yml` runs `cargo test -p euhedral-io` on a
`windows-2025` runner, where `src/win/live.rs` exercises files, loopback sockets, open and rename,
each twice: with IoRing and forced onto the completion port.

The tests are unit tests in each module (they drive the source by hand) and `tests/io_owned.rs`.
`AsyncIo`'s tests use a scratch directory beside the test binary, under `target/`.
