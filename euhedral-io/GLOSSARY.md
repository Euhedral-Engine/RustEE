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
is not emulated in user mode. `AsyncIoOptions { windows_ioring: false }` forces the port. Open and rename
run on the OS thread pool and post to the port. A handle joins the engine's port and can serve one
`AsyncIo`, ever.
