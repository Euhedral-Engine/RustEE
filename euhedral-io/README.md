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
| [`positional`](src/positional.rs) | The fallback engine, on Linux and Android, for a system that refuses io_uring. |
| [`win`](src/win/mod.rs) | The Windows engines: IoRing and completion ports. |

Feature `async-io` (default) builds all of it; without it, or on macOS (no engine yet), the crate is empty.
The crate's clippy bans (see [`clippy.toml`](clippy.toml)) are the control plane's: no threads, sleeps,
channels, `Condvar`, blocking `std::fs` or `std::net` calls.

## `AsyncIo`

`AsyncIo` does file and socket I/O that holds no worker: `read`, `write`, `send`, `recv`, `accept`,
`open_path` (and `open_file` with raw flags on Linux), `rename`, `file_len`, `send_to` and `connect` each
return at once, and the operation's `IoDone` (its buffer back, and the bytes moved or the error)
reaches the `IoThen` continuation, which builds the frame the source emits. Handles are the
platform's: descriptors on Linux and Android, `HANDLE`s and `SOCKET`s on Windows (the aliases
`RawFile`, `RawSock`, `OwnedFile` and `OwnedSock`). A read is cut into pieces, one
after another, each piece's completion submitting the next. The caller sets the size (see
*Piece size*, below). The `AsyncIo` type is the same everywhere; an engine under it
starts operations and finds their ends, and `src/lib.rs` does the rest.

**Piece size.** `AsyncIoOptions::read_piece`, `send_piece` and `write_piece` (each 1 to `MAX_PIECE`, 2 GiB less
4 KiB; `open_with_options` refuses anything else, naming the value) set how many bytes one piece of a read, a
send or a write moves, and `ReadOptions::with_piece(n)` on `read_with` and
`read_into_with` sets it for one read, so one `AsyncIo` can hold buffered and direct files. `write_piece` applies
to the positional engine and to Windows; io_uring does not cut writes below `MAX_PIECE`. The crate sets no
policy: all three default to `DEFAULT_PIECE` (256 KiB), a default the caller may change. What a piece costs the
worker depends on the engine:

- **io_uring** (and Windows): a read the page cache can serve is copied inside the submitting call, on the
  submitting worker, so for a buffered file keep the default small piece, which bounds how long
  one call holds a worker. A read of a file opened for direct I/O copies nothing and no worker waits for the
  device; each piece is a round trip, so for a direct file a large piece (up to `MAX_PIECE`) is recommended: the
  device works on the whole read.
- **Positional engine:** every piece is a synchronous `pread` or `pwrite` made by the polling worker, buffered or
  direct, and the worker waits for it. A visit makes up to `attempts_per_poll` of them (64 by default) for a
  lone operation, and at most `pieces_per_turn` (4) in a row for each when several are due, so the worker is
  held for about that many pieces' time: size the piece with that in mind (a piece of `MAX_PIECE` holds it for
  the time of a 2 GiB read).

For a direct file the piece must be a multiple of the device's logical block size (a multiple of 4096 is safe),
which the crate cannot check and the kernel enforces.

`file_len(file, then)` finds the length of an open file without blocking the caller, so a reader can size its
buffers or plan its reads from it: `statx(2)` of the descriptor through the ring on Linux and Android
(`STATX_SIZE`, empty path with `AT_EMPTY_PATH`), `GetFileSizeEx` on the OS thread pool on Windows (any handle works, as the
call does not wait on the file). `then` gets `io::Result<u64>`: the size at the moment the kernel looked, or the
system's error (a bad descriptor ends with `EBADF` on Linux and Android, `ERROR_INVALID_HANDLE` (6) on Windows). The file must stay open until `then` runs. Where the length
does not fit a `usize` (a 32-bit target) it ends with an overflow error.

`socket_for(addr)` makes the socket a client's `connect` needs: a new, unconnected stream socket of the address's
family (IPv4 or IPv6), owned by the result, so that dropping it closes it. The standard library cannot make one
(its connect blocks), and a caller should not need `unsafe` or a platform crate to get one. It is made as the engine
wants it (close-on-exec on Linux and Android; overlapped and not inheritable on Windows) and the engine does the rest
when it connects (the bind and the join to the completion port on Windows). It is no operation: nothing is in
flight on it and `pending` does not count it. A connect that ended early leaves it in an unknown state: drop it.

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

- **Linux and Android:** io_uring (or the positional fallback, below), with one ring per submitting thread. An operation belongs to the
  thread that submitted it, and lattice workers live as long as the lattice.
- **Windows:** one I/O completion port per `AsyncIo`, polled with a zero timeout, for every
  operation, and one IoRing per submitting thread for file reads and writes where the OS has a
  kernel ring for both (Windows 11 22H2 and later; not one emulated in user mode). The ring
  functions are looked up at run time, so the library still loads where they are missing. A file
  keeps its route: IoRing refuses a handle that belongs to a completion port. `AsyncIoOptions { windows_ioring: false, ..AsyncIoOptions::default() }` forces the port.
  Windows has no asynchronous open, rename or file length, so those run on the OS thread pool and leave their
  result in a queue the poll drains: the one place the crate uses threads it did not create. Handles must be
  opened for overlapped I/O (`open_path` does, and so do Rust's sockets) and must not skip the port
  on success; a handle that would block or never complete is refused. A handle joins the engine's
  port and can serve one `AsyncIo`, ever. Writes go in pieces of at most `write_piece` (256 KiB by default), since a write
  that extends a file is synchronous on NTFS. Tested on a real Windows by
  [CI](../.github/workflows/ci.yml) (its `workspace` job on `windows-2025`).
- **macOS:** no engine yet.

### The fallback where io_uring is refused

A container or sandbox may refuse io_uring: Docker's and Podman's default seccomp profile, Kubernetes `RuntimeDefault`,
`kernel.io_uring_disabled`, gVisor. `AsyncIo::open` then does not fail: when the first ring's setup is refused with
`EPERM`, `EACCES` or `ENOSYS` (and only those: any other error still fails the open), it runs on the
positional engine, and `backend()` says which engine it got. The contract is the same: continuations, short reads and
writes resubmitted for the rest, refusals delivered to the continuation, time limits and cancel, `pending`, and the
same counters.

```
# #[cfg(any(target_os = "linux", target_os = "android"))]
# {
use euhedral_io::{AsyncIo, AsyncIoOptions, IoBackend};

// Where io_uring works this is `IoUring`; where the system refuses it, `Positional`.
let io = AsyncIo::open().unwrap();
println!("asynchronous I/O runs on {:?}", io.backend());

// Choose the fallback on purpose (tests, or a deployment that wants it everywhere).
let io = AsyncIo::open_with_options(AsyncIoOptions {
    positional_io: true,
    ..AsyncIoOptions::default()
})
.unwrap();
assert_eq!(io.backend(), IoBackend::Positional);
# }
```

- **No thread, no wait.** The fallback is an engine under the same `AsyncIo` source. A submission queues the
  operation; the reap frame (the work a worker does for the source, see above) walks the operations in flight once
  per visit and hands the ended ones on. A visit makes one `poll(2)` call with a zero timeout over the descriptors
  that operations wait for (sockets, accept, pipes), and attempts only the ones it reports ready; a socket that is
  not ready costs that one shared call and no attempt of its own. An attempt is one piece of a read or write
  (`pread`, `pwrite`, at most the read, write or send piece, 256 KiB by default), one non-blocking socket call, or the whole of an open, a rename or a length
  (`fstat`). A visit makes at most `attempts_per_poll` attempts (64 by default), and every attempt counts against that. An operation that is the only
  one due in a visit (the others wait for a descriptor that is not ready, back off or were cancelled) goes on until
  the attempts are used, so a lone long read moves up to 64 pieces (16 MiB at the default size) a visit; with two or
  more due, each moves at most `pieces_per_turn` (4) pieces in a row.
- **Streams.** A pipe, tty or character device cannot be read at an offset (`ESPIPE`): such an operation goes on as
  a stream, waits for readiness and then uses `read` or `write`; the offset is ignored.
- **Opening a FIFO** never holds the worker: the open is made with `O_NONBLOCK` and the flag is cleared on the
  descriptor that comes back. A write open with no reader (`ENXIO`) stays in the line and is retried at later visits,
  so a limit or a cancel still ends it, with growing gaps (1, 2, 4 ... 64 visits) between attempts, since nothing
  signals that a reader appeared. `open_path` therefore never waits for the other end of a FIFO.
- **A leased file** (`F_SETLEASE` held by another process) makes the non-blocking open fail with `EWOULDBLOCK`. For a
  caller that did not ask for `O_NONBLOCK` the open is retried with the same gaps until the lease breaks (45 s by
  default), as a blocking open would wait; a limit shorter than that ends it with `LimitPassed`.
- **What it costs.** There is no kernel queue: the calls run on the worker that reaps, one reap at a time, so a
  file operation holds a worker for the length of its piece, and path calls (`open_path`, `rename`) for as long
  as the file system takes. It is a correct fallback, not a fast one.
- **Everything the io_uring engine offers is offered:** reads (into a `Vec` or caller memory), writes, sends,
  receives, accept, connect, datagrams, `open_path`, `open_file`, `rename`, `file_len`, and `send_with`, `recv_with`,
  `accept_with`, `connect_with` with limits and `cancel`. A limit or cancel is looked at before every step, so it ends
  an operation that waits for a socket and one between two pieces. A connect makes a blocking socket non-blocking for its
  handshake and puts its flags back when it ends (also when the `AsyncIo` is dropped first). An accept on a
  blocking listener sets `O_NONBLOCK` only around the `accept4` call and restores the flags on every path; a
  listener that another blocking accepter shares may see `EAGAIN` in that window.
- **The probe.** At open the engine also enters a ring with a no-op, without waiting. A system that allows
  `io_uring_setup` but refuses `io_uring_enter` with `EPERM`, `EACCES` or `ENOSYS`
  gets the fallback too. On kernels before 5.12 a ring is charged to the memlock limit, and a refusal with `ENOMEM`
  there still fails the open: it is not a permission refusal.
- **Dropping** an `AsyncIo` with operations in flight frees their buffers and runs no continuation: nothing
  is in a kernel that could still write them (the io_uring engine leaks them instead).
- `AsyncIoOptions::positional_io` forces the fallback where io_uring works; `windows_ioring` does not apply to it; the piece sizes do.

### Closing handles on Windows

The Windows engine remembers which handle values it joined to its completion port, and the system gives a closed
handle's value to the next handle made. Sockets and files the engine makes itself (`socket_for`, accepted sockets,
`open_path`) need nothing. For a socket or file you made, call `io.forget_socket(raw)` or `io.forget_file(raw)` before
you close it, with nothing in flight on it (both do nothing elsewhere). A handle forgotten while it is still open is
refused by the system at its next join and no longer taken for ours: every later operation on it is taken, as
any operation is, and ends in its completion with `InvalidInput` ("belongs to another completion port"); none
hangs and no call refuses it. The same holds for a handle of another port, for a file a ring serves used by a thread
with no ring, and for a handle the engine cannot hold: submit returns an error only for an operation it cannot take. A handle one `AsyncIo` made and a different
`AsyncIo` uses counts as caller-made for that other one: forget it there before closing it.

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
Windows; on its `windows-2025` runner `src/win/live.rs` exercises files, loopback sockets, open and rename,
each twice: with IoRing and forced onto the completion port.

The tests are unit tests in each module (they drive the source by hand) and `tests/io_owned.rs`.
`AsyncIo`'s tests use a scratch directory beside the test binary, under `target/`.
