# euhedral-hardware-utils

CPU topology, thread affinity, pinned executors and resource-pressure monitoring for RustEE. It is
a port of the upstream Java `euhedral-hardware-utils`. The Java native library is replaced by direct
system calls, so there is nothing to extract or load at run time.

**Where it sits.** Depends only on `log`, plus `libc` on Unix and `windows-sys` on Windows.
`euhedral-core` builds on it: the lattice follows a `ResourceMonitor` and a `TopologyMapper`, each
fragment worker runs on a `PinnedThreadExecutor`, and frames carry a `CpuInfo` as their origin.

**Platforms.** The back end is chosen at compile time:

| Platform | Topology | Resources | Affinity |
|---|---|---|---|
| Linux | sysfs | cgroups and `/proc` | exact masks (`AffinityCapability::Exact`) |
| macOS | sysctl | Mach and `NSProcessInfo` | affinity tags, a locality hint only (`LocalityHint`) |
| Windows | processor groups | job objects and process counters | group affinity (`Exact`) |
| other | a flat fallback | none: supply a provider | unsupported |

RustEE runs on x86 and arm64. It is untested on mobile. The Linux, macOS and Windows parsers and
the resource arithmetic build and are tested on every platform; only the system calls are
platform-specific, so macOS and Windows behaviour at run time is less tested than Linux.

## Modules

| Module | Main types |
|---|---|
| [`topology`](src/topology/mod.rs) | `Topology` (immutable; `Topology::system()` probes the host once), `CpuInfo`, `CoreInfo`, `SocketInfo`, `CpuCacheLayout`; `TopologyMapper` and its `EffectiveSystemTopology` / `EffectiveSocketTopology` snapshots; `parse_hex_mask`, `to_hex_mask`. |
| [`topology::source`](src/topology/source.rs) | Raw observations (`TopologyInput`, `LogicalCpu`, `CacheDomain`) and `normalize` / `fallback`, to build a fixture topology or describe a machine this crate does not probe. |
| [`affinity`](src/affinity/mod.rs) | Free functions on the calling thread (`set_affinity`, `set_affinity_set`, `release_affinity`, `current_cpu`, `current_cpu_info`, `capability`, `base_mask`); `AffinityController`, `AffinityCapability`, `AffinityProvider`; `PinnedThreadExecutor` and `is_interrupted`. |
| [`affinity::masks`](src/affinity/masks.rs) | Little-endian CPU masks: word `i`, bit `b` is CPU `64 * i + b`. |
| [`monitor`](src/monitor/mod.rs) | `ResourceMonitor` and `MonitorState`; the published `HardwareUtilization` with its `SocketSnapshot`, `CoreSnapshot` and `CpuSnapshot`; `system_snapshot`. |
| [`monitor::sampling`](src/monitor/sampling.rs) | Typed signals (`Signal`, `CounterSignal`, gauges), their validity and resolution, and the provider traits a custom back end implements. |
| [`monitor::utilization`](src/monitor/utilization.rs) | The published records. Their constructors clamp ratios to `[0, 1]` and turn NaN and negatives into zero. |
| [`util`](src/util/mod.rs) | `CpuSet`, the immutable sparse set of logical CPU ids every module shares; `OsName`. |

## Key concepts

### Topology and CPU sets

A `Topology` lists the logical CPUs, cores, sockets and caches. Logical CPU ids are the OS ids and
may be sparse: `cpu_span()` is the highest id plus one (size id-indexed arrays by it), `cpu_count()`
is the number of CPUs, and you iterate `cpu_set()` rather than `0..span`. A lookup for an id that
is not active returns `None`. Cores carry a `p_core` flag, false only for a core the platform
classified as an efficiency core.

A `CpuSet` has no mutators: `union`, `intersection`, `difference`, `with` and `without` return new
sets, and equal sets always compare and hash equal.

```rust
use euhedral_hardware_utils::topology::Topology;
use euhedral_hardware_utils::util::CpuSet;

let topology = Topology::system();
assert!(topology.cpu_count() >= 1);
for cpu in topology.cpu_set().iter() {
    let info = topology.cpu_info(cpu).unwrap();
    assert_eq!(info.cpu, cpu);
    assert!(topology.core_info(info.core).unwrap().cpus.contains(cpu));
}

let set: CpuSet = [0, 2, 5].into_iter().collect();
assert_eq!(set.span(), 6);
assert_eq!(set.without(2), [0, 5].into_iter().collect());
```

### Effective topology

The **effective topology** is the part of the topology that is usable right now: the CPUs the
resource monitor reports as effective (on Linux, the cgroup's `cpuset`), intersected with the
topology's CPUs and the CPUs allowed when the `TopologyMapper` was built. The mapper publishes it
as an immutable `Arc` snapshot with a version, only when membership changes, and readers never
block. A socket or core with no effective CPU is inactive.

### Affinity and pinning

The free functions in `affinity` act on the calling thread through one process-wide
`AffinityController`. A thread's first successful request takes a **lease** on its original
binding, and `release_affinity` restores it, however many requests were nested. What a request
can do depends on `capability()`: `Exact` applies the whole mask, `LocalityHint` only a scheduler
preference, and `Unsupported` changes nothing and returns `false`.

A `PinnedThreadExecutor` runs each command on a fresh thread pinned to one CPU. There is at most
one live executor per CPU in the process; handles are cheap to clone, and dropping the last one
closes it. Interruption is cooperative: `shutdown_now` and `close` set a flag the task reads with
`is_interrupted()`.

```rust
use euhedral_hardware_utils::affinity::{self, AffinityCapability, PinnedThreadExecutor};

let cpu = affinity::base_mask().first().unwrap();
let executor = PinnedThreadExecutor::get_or_create(cpu, "example").unwrap();
let ran_on = executor.submit(affinity::current_cpu).unwrap().join().unwrap();
if affinity::capability() == AffinityCapability::Exact && ran_on.is_some() {
    assert_eq!(ran_on, Some(cpu));
}
executor.close();
```

### Monitor and pressure

A `ResourceMonitor` polls the platform's provider on a fixed cadence (200 ms by default), turns
each sample into a smoothed `HardwareUtilization`, keeps its `TopologyMapper` in step with the
effective CPUs, and hands each result to its listeners on a dedicated thread. Listeners get the
latest value first: a slow listener skips values rather than queueing them. Lifecycle: `New` →
`start` → `Running` ⇄ `Stopped` → `close` → `Closed`; dropping the monitor closes it.

**Pressure** is a per-CPU value in `[0, 1]` (`HardwareUtilization::per_quota_cpu_pressure`, and
`CpuSnapshot::pressure`). Each input (scheduler wait and stall, quota throttling, steal and other
external contention, run-queue depth, memory headroom and reclaim, I/O latency and queue depth,
thermal and low-power state) becomes a ratio in `[0, 1]`, is smoothed with a fast attack and a slow
release, and the smoothed cells combine by `max`. A failed or unsupported read is never a zero: it
carries its validity and resolves as unavailable, and a counter that fails for a moment repeats its
last interval within a TTL rather than dropping the pressure.
`euhedral-core` lowers a worker's batch cap and its cache's cap factor as its CPU's pressure rises.

```rust,no_run
use euhedral_hardware_utils::monitor::{MonitorError, ResourceMonitor};
use euhedral_hardware_utils::topology::TopologyMapper;
use std::sync::Arc;
use std::time::Duration;

let mapper = Arc::new(TopologyMapper::new());
let monitor = match ResourceMonitor::with_period(mapper.clone(), Duration::from_millis(50)) {
    Err(MonitorError::Unsupported) => return, // no provider on this platform
    other => other.unwrap(),
};
monitor.start().unwrap();
std::thread::sleep(Duration::from_millis(200));
if let Some(utilization) = monitor.utilization() {
    println!("pressure per CPU: {:?}", utilization.per_quota_cpu_pressure);
}
println!("effective CPUs: {}", mapper.effective_topology().effective_cpus);
monitor.close();
```

## Build and test

```sh
mise exec -- cargo test -p euhedral-hardware-utils
```

`tests/smoke.rs` runs the real back end of the machine running the tests: it checks the system
topology, starts a monitor, and pins a task. The pinning check passes with any affinity capability;
it only asserts the CPU when the capability is `Exact`. Linux builds need a C linker and libc
headers (see `mise.toml`). The first two examples above are compiled and run as doctests; the
monitor example is compiled only (`no_run`), because it starts a sampling thread.
