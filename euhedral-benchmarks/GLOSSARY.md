# euhedral-benchmarks glossary

The terms of [`euhedral-benchmarks`](README.md), the end-to-end benchmarks of the control plane. They are
implementation-level: nothing here is part of what a user builds on. The [top-level glossary](../GLOSSARY.md)
covers the layers it measures. Paths are relative to this crate. Entries run from what is measured to how it
is reported.

**high-contention throughput** ([`high_contention`](src/high_contention.rs),
[`bin/high_contention_throughput`](src/bin/high_contention_throughput.rs)): the benchmark of a lattice that
many sources feed at once, as a plain binary. It reports the average time and the throughput.

**repeating source**: a source that circulates a fixed set of no-op frames (2048 per source). Finishing a
frame bumps its CPU's cell of a striped completion counter and returns the frame to its source through a
queue, to be pushed again. The sources default to the logical CPU count (`--sources` or
`EUHEDRAL_HCT_SOURCES` overrides it).

**harness core**: the core the harness thread is pinned to (the highest efficiency core, else the highest
core). Its CPUs are left out of the lattice's workers. Where the platform will not pin, or has fewer than
two cores, the benchmark says so and runs unpinned.

**fork, iteration, invocation**: an invocation waits for the completion count to advance by 32,000,000
(1 minute timeout). An iteration repeats invocations for its time. A fork builds its own lattice and runs 3
warm-up iterations of 3 seconds and 5 measured ones of 5 seconds; the driver runs 3 forks as separate
processes.

**latency sample**: one frame in 64 is stamped when its source hands it out, and the worker that executes
it records the time since in a histogram of its own. The latency runs from the source handing a frame out
to a worker starting it; the way back is not in it. `--no-latency` turns the sampling off. Each iteration
reports the quantiles P0, P25, P50, P90, P99 and P100 in nanoseconds.

**push path** ([`bin/push_path`](src/bin/push_path.rs)): a microbenchmark of one push down the routing path:
a vertex that routes to shard vertices, each of which routes to worker edges that end in a receiver. One
thread pushes, and the receiver drops each frame, so the number is the cost of the hops plus one frame
allocation, which `--calibrate` measures alone. Each sample times `--ops` pushes (2,000,000 by default); the
summary reports nanoseconds per push over the samples (40 by default).
