# euhedral-benchmarks

End-to-end benchmarks of the control plane. Nothing depends on this crate.

## High-contention throughput

`high_contention_throughput` measures the average time and the throughput of a lattice that many sources feed at once
(the Java `HighContentionThroughput` benchmark, as a plain binary).

- One repeating source per source, each circulating 2048 no-op frames; finishing a frame bumps its CPU's cell of a striped completion
  counter (one writer per cell: a worker is pinned to its CPU) and hands the frame back to its source. The sources are attached to one lattice.
- The harness thread is pinned to a core of its own (the highest efficiency core, else the highest core), and that
  core's CPUs are left out of the lattice's workers. Where the platform will not pin, or has fewer than two cores, it
  says so and runs unpinned.
- An invocation waits for the completion count to advance by 32,000,000 (1 minute timeout), spinning and yielding. An
  iteration repeats invocations for its time. A fork builds its own lattice and runs 3 warm-up iterations of 3 seconds
  and 5 measured ones of 5 seconds; the driver runs 3 forks as separate processes.
- One frame in 64 is stamped when its source hands it out, and the worker that executes it records the time since in
  a histogram of its own (relaxed atomic increments, no lock; fixed log-linear buckets, 0.8% relative error). Each
  iteration reports the latency quantiles P0, P25, P50, P90, P99 and P100 in nanoseconds; the summary reports them over
  every measured iteration of every fork. `--no-latency` turns the sampling off. `--poll-spins N` sets the spins between two sums of the completion counter (upstream: 32, the default): each sum reads every worker's counter line, and 1024 or more measured 5% higher throughput. Per-iteration P0 and P100 are the
  edges of the lowest and highest occupied buckets.
- The output is one line per iteration, and a `summary` line with the mean and standard deviation of the measured
  iterations' throughput (operations per second) and time per operation (nanoseconds), the sources and the worker CPUs.

```text
cargo run --release -p euhedral-benchmarks --bin high_contention_throughput
cargo run --release -p euhedral-benchmarks --bin high_contention_throughput -- --forks 1 --iterations 2 --seconds 2
EUHEDRAL_HCT_SOURCES=4 cargo run --release -p euhedral-benchmarks --bin high_contention_throughput
```

Run it alone on the machine: it keeps every core busy. The sources default to the physical cores minus one.

## Where it differs from Java

- Java's `RepeatingSink` re-pushes a fixed array and takes nothing back. A Rust `Box<dyn Frame>` cannot be aliased, so
  each finished frame returns to its source through a queue (`recycle`) and is pushed again: a harness cost with no
  Java counterpart. The queue has many producers (the workers) and one consumer, held with the banked demand in a
  `SourceCell`.
- Java counts completions with `PaddedLongAdder.getAndAdd`. Here the counter has one cell per CPU id, written by the
  one worker pinned to that CPU with a load and a store (a deviation asked for, so that the counting takes no locked
  instruction). A thread that is not alone on its CPU uses a shared cell with the add.
- The latency measured runs from the source handing a frame out to a worker starting it; the way back is not in it.
- A claimed counter cell is never released: a tally lives for one fork.
