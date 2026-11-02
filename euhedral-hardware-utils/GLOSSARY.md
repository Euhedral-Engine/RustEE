# euhedral-hardware-utils glossary

The implementation-level terms of [`euhedral-hardware-utils`](README.md): how threads are bound to CPUs
and how the machine's load is sampled. The terms a user meets first (topology, effective topology, CPU set)
are in the [top-level glossary](../GLOSSARY.md). Paths are relative to this crate. Entries run from the
platform layer up to the monitor.

**platform back end** ([`platform`](src/platform/mod.rs)): the part of the crate that reads the machine and
binds threads. It is chosen at compile time: sysfs, cgroups, `/proc` and exact masks on Linux; sysctl, Mach
and affinity tags on macOS; processor groups and job objects on Windows; a flat fallback with no affinity
elsewhere. The parsers and the resource arithmetic build and are tested on every platform; only the system
calls differ.

**P-core, E-core** ([`CoreInfo::p_core`](src/topology/mod.rs)): performance and efficiency cores. A core is
a P-core unless the platform classified it as an efficiency core.

**TopologyMapper** ([`TopologyMapper`](src/topology/mapper.rs)): keeps the effective topology. It
publishes it as an immutable `Arc` snapshot with a version, only when membership changes; readers never
block.

**affinity, pinning** ([`affinity`](src/affinity/mod.rs)): binding a thread to CPUs. `AffinityCapability`
says what the platform can do: `Exact` masks (Linux, Windows), a `LocalityHint` (macOS), or `Unsupported`,
which changes nothing and returns `false`.

**affinity lease** ([`AffinityController`](src/affinity/controller.rs)): the free functions in `affinity`
act on the calling thread through one process-wide `AffinityController`. A thread's first successful
request takes a lease on its original binding, and `release_affinity` restores it, however many requests
were nested.

**PinnedThreadExecutor** ([`PinnedThreadExecutor`](src/affinity/executor.rs)): runs each command on a fresh
thread pinned to one CPU, with at most one live executor per CPU. Each fragment worker runs on one. Every
handle is one use of the CPU's shared executor: it closes when its last use ends (the last handle is
dropped or `release`d), so fragments of different lattices can share a CPU and each closes only its own
worker task.

**TaskInterrupt** ([`TaskInterrupt`](src/affinity/executor.rs)): interruption of a pinned task is
cooperative. `shutdown_now` and `close` set a flag on every task; `submit_interruptible` returns a
`TaskInterrupt` that sets it on one task alone; a task reads it with `is_interrupted()`. `close` closes the
executor for every user; a user that only wants to stop its own work interrupts its task and releases its
handle.

**monitor** ([`ResourceMonitor`](src/monitor/mod.rs)): a background sampler that polls CPU, memory, I/O and
power signals (every 200 ms by default), publishes a smoothed `HardwareUtilization`, and keeps a
`TopologyMapper` in step. The lattice follows one. Listeners get the latest value first: a slow listener
skips values rather than queueing them. Its lifecycle is `New`, `start`, `Running` and `Stopped`, then
`close`; dropping the monitor closes it.

**pressure** ([`HardwareUtilization`](src/monitor/utilization.rs),
[`monitor::pressure`](src/monitor/pressure.rs)): a per-CPU value in `[0, 1]` combining scheduler stall,
throttling, contention, memory, I/O and thermal signals, each smoothed with a fast attack and a slow
release, the smoothed cells combined by `max`. It lowers batch caps and cap factors. A failed or
unsupported read is never a zero: it carries its validity and resolves as unavailable.
