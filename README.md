# RustEE

RustEE is an asynchronous scheduler and runtime for Rust, a hybrid of an event-driven architecture
and a CUDA-like system. A **Frame** is the kernel: a unit of work. A graph is Frames chained through
their lifecycle events, each finished frame handing the next one on. A **LatticeSource** is a
programmable stream that the lattice's workers pull frames from or request them from. The lattice
runs one pinned worker per core, routes frames by their routing hash and routing policy, keeps
ordered frames in order on their owning CPU, and lets idle workers steal the rest. RustEE is a port
of the Java [Euhedral-Execution](https://github.com/Euhedral-Engine/Euhedral-Execution) runtime. It runs on x86 and
arm64 (Linux, macOS and Windows back ends) and is untested on mobile.

## Crates

| Crate | What it is |
|---|---|
| [`euhedral-core`](euhedral-core/README.md) | Frames, the lattice (shards, fragments, caches), ingest sinks, routing and execution. |
| [`euhedral-hardware-utils`](euhedral-hardware-utils/README.md) | CPU topology, thread affinity, pinned executors and resource-pressure monitoring. |
| [`euhedral-data-structures`](euhedral-data-structures/README.md) | Cache-line padded atomics and lock-free chunked queues. |
| [`euhedral-hashing`](euhedral-hashing/README.md) | xxHash64 for routing keys, ordered-chain ids and vectors. |
| [`euhedral-memory`](euhedral-memory/README.md) | A location-agnostic memory and residency planner: budgets, owners with lifetimes, claims with priorities, give-back as frames, admission, arenas and a leak check; CUDA allocation backends (feature `cuda`). |
| [`euhedral-graph`](euhedral-graph/README.md) | No-wait dataflow on the lattice: completions, async I/O (io_uring; IoRing and completion ports on Windows), a sequencer, joins, device lanes and stage graphs. |

`euhedral-core` depends on the other three; they depend on none of each other. `euhedral-graph` builds
on `euhedral-core`; nothing depends on it. `euhedral-memory` builds on `euhedral-core`, `euhedral-data-structures` and
`euhedral-hashing`; nothing depends on it.

- [`GLOSSARY.md`](GLOSSARY.md): every term a user of RustEE meets, with links to the code.
- [`CHANGELOG.md`](CHANGELOG.md): changes to the lattice's execution model, newest first.

## Build and test

The toolchain is pinned in [`mise.toml`](mise.toml) (Rust 1.99.1 with rustfmt and Clippy). With
[mise](https://mise.jdx.dev) installed:

```sh
mise install
mise exec -- cargo build --workspace
mise exec -- cargo test --workspace
mise exec -- cargo test -p euhedral-core   # one crate
```

Linux builds also need a C linker and libc headers (`sudo apt install build-essential` on
Ubuntu/Debian). Some `euhedral-core` tests need an SMT core or two whole cores the process may run
on, and pass without checking anything when the host lacks them; see the
[`euhedral-core` README](euhedral-core/README.md#build-and-test). Each crate README's examples run as
doctests.
