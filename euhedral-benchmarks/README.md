# euhedral-benchmarks

End-to-end benchmarks of the control plane. Nothing depends on this crate.

## High-contention throughput

`high_contention_throughput` measures the average time and the throughput of a lattice that many sources feed at once
(the Java `HighContentionThroughput` benchmark, as a plain binary).

- One repeating source per source, each circulating 2048 no-op frames; finishing a frame bumps a striped completion
  counter and hands the frame back to its source. The sources are attached to one lattice.
- The harness thread is pinned to a core of its own (the highest efficiency core, else the highest core), and that
  core's CPUs are left out of the lattice's workers. Where the platform will not pin, or has fewer than two cores, it
  says so and runs unpinned.
- An invocation waits for the completion count to advance by 32,000,000 (1 minute timeout), spinning and yielding. An
  iteration repeats invocations for its time. A fork builds its own lattice and runs 3 warm-up iterations of 3 seconds
  and 5 measured ones of 5 seconds; the driver runs 3 forks as separate processes.
- The output is one line per iteration, and a `summary` line with the mean and standard deviation of the measured
  iterations' throughput (operations per second) and time per operation (nanoseconds), the sources and the worker CPUs.

```text
cargo run --release -p euhedral-benchmarks --bin high_contention_throughput
cargo run --release -p euhedral-benchmarks --bin high_contention_throughput -- --forks 1 --iterations 2 --seconds 2
EUHEDRAL_HCT_SOURCES=4 cargo run --release -p euhedral-benchmarks --bin high_contention_throughput
```

Run it alone on the machine: it keeps every core busy. The sources default to the physical cores minus one.
