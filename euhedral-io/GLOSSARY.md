# euhedral-io glossary

The implementation-level terms of [`euhedral-io`](README.md): the engines under `AsyncIo` and the memory
contract of an operation. What a user calls (`AsyncIo`, continuations) and how an operation proceeds are in
the [top-level glossary](../GLOSSARY.md). Paths are relative to this crate. Entries run from the portable
layer down to the platform engines.

**engine** ([`uring`](src/uring.rs), [`win`](src/win/mod.rs)): the platform part of `AsyncIo`. An engine
only starts operations and finds their ends; the public API, the buffers, the continuations and the
`LatticeSource` behaviour are portable and live in `src/lib.rs`. io_uring on Linux and Android; IoRing and
completion ports on Windows; none on macOS yet, where the crate is empty.

**pieces**: a read is cut into pieces so that no single completion holds a worker for long. On Windows a
write goes in pieces of at most 256 KiB, because a write that extends a file is synchronous on NTFS. A
region read with `read_into` is cut internally too, each piece landing at its own offset.

**IoMemory** ([`IoMemory`](src/lib.rs)): an `unsafe` trait for memory the caller owns, that
`AsyncIo::read_into` reads into. The implementor promises that the region is initialized, stays put when the
value moves, and is exclusively the operation's from submission until the continuation gets the value back
in an `IoDoneOwned`. The memory comes back whether the read succeeded or failed. The one exception is a ring
that broke while the kernel may still have been writing the region: the value is leaked and the result is
the ring's failure.

**ring** ([`uring`](src/uring.rs), [`win`](src/win/mod.rs)): an io_uring (or Windows IoRing) submission and
completion queue pair. Each submitting thread has its own, so submitting never contends; an operation
belongs to the ring of the thread that submitted it.

**completion port, IoRing** ([`win`](src/win/mod.rs)): the two engines of `AsyncIo` on Windows. A
completion port collects the packets of overlapped calls and is polled with a zero timeout; IoRing is
Windows' kernel ring for file reads and writes (Windows 11 22H2 and later), used where the OS has one that
is not emulated in user mode. `AsyncIoOptions { windows_ioring: false, ..AsyncIoOptions::default() }` forces the port. Open, rename and file length
run on the OS thread pool and leave their result in a queue the poll drains. A handle joins the engine's port and can serve one
`AsyncIo`, ever.

**limit, cancel, `OpId`** ([`lib.rs`](src/lib.rs)): `OpOptions { limit }` bounds a socket operation in time
(the whole operation, every piece of a send); `cancel(OpId)` ends one on request. The `OpId` names an
operation by its `AsyncIo`, slot and generation, so a cancel that arrives after the operation ended, after its
slot was reused, or on another `AsyncIo`, finds nothing. A limit ends with `TimedOut` carrying `LimitPassed`, a
cancel with `Other` carrying `Cancelled`; both carry the bytes moved. On Linux a limit is a linked timeout and
a cancel an async cancel, which may use a reserve of the completion queue that operations may not; on Windows
a cancel is `CancelIoEx` and a limit a deadline the poll checks.

**file length** ([`lib.rs`](src/lib.rs), [`uring.rs`](src/uring.rs), [`win`](src/win/mod.rs)): `file_len` is a `Request::FileLen`. On Linux it is a `statx` of the descriptor (empty path, `AT_EMPTY_PATH`, `STATX_SIZE`) into a boxed result block that stays where the kernel writes it; the engine reads `stx_size` when the completion arrives and refuses a reply whose mask lacks the size. On Windows it is a `Kind::FileLen` on the OS thread pool, behind `Sys::file_len` (`GetFileSizeEx`); the simulated `Sys` answers from its files.
