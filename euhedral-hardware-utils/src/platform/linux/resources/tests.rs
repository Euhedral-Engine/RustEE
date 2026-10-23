use super::*;
use crate::monitor::sampling::SignalValidity;
use crate::platform::testing::{TempDir, write};
use crate::topology::source;

const V2_MOUNTINFO: &str = "29 23 0:26 / /sys/fs/cgroup rw,nosuid shared:4 - cgroup2 cgroup2 rw\n";
const CG: &str = "sys/fs/cgroup/app";

fn v2_fixture() -> TempDir {
    let dir = TempDir::new("resources-v2");
    let root = dir.path();
    write(root, "proc/self/mountinfo", V2_MOUNTINFO);
    write(root, "proc/self/cgroup", "0::/app\n");
    for (file, content) in [
        ("cpu.max", "50000 100000\n"),
        (
            "cpu.stat",
            "usage_usec 1000\nuser_usec 600\nthrottled_usec 20\n",
        ),
        (
            "cpu.pressure",
            "some avg10=0.00 avg60=0.00 avg300=0.00 total=7\nfull avg10=0 avg60=0 avg300=0 total=1\n",
        ),
        ("cpuset.cpus.effective", "0-1\n"),
        ("memory.max", "1048576\n"),
        ("memory.high", "max\n"),
        ("memory.current", "524288\n"),
        (
            "memory.stat",
            "anon 1\ninactive_file 4096\npgsteal_kswapd 2\npgsteal_direct 3\n",
        ),
        (
            "memory.pressure",
            "some avg10=0 avg60=0 avg300=0 total=11\n",
        ),
        ("io.stat", "8:0 rbytes=100 wbytes=50 rios=1 wios=1\n"),
        ("io.pressure", "some avg10=0 avg60=0 avg300=0 total=13\n"),
    ] {
        write(root, &format!("{CG}/{file}"), content);
    }
    write(
        root,
        "proc/stat",
        "cpu  10 0 5 100 1 0 0 3 0 0\ncpu0 5 0 2 50 1 0 0 2 0 0\ncpu1 5 0 3 50 0 0 0 1 0 0\nintr 1 2\n",
    );
    write(
        root,
        "proc/schedstat",
        "version 15\ntimestamp 1\ncpu0 0 0 1 1 1 1 100 4000 5\ndomain0 3 1\ncpu1 0 0 1 1 1 1 100 6000 5\n",
    );
    write(
        root,
        "proc/diskstats",
        "   8       0 sda 10 0 100 7 20 0 200 3 2 0 0 0 0 0 0\n\
            8       1 sda1 10 0 100 7 20 0 200 3 9 0 0 0 0 0 0\n\
          253       0 dm-0 10 0 100 7 20 0 200 3 9 0 0 0 0 0 0\n\
            7       0 loop0 10 0 100 7 20 0 200 3 9 0 0 0 0 0 0\n",
    );
    write(root, "sys/block/sda/device/model", "disk\n");
    write(root, "sys/block/dm-0/dm/name", "root\n");
    write(root, "proc/meminfo", "MemTotal: 8 kB\nMemAvailable: 2 kB\n");
    dir
}

fn provider(root: &Path) -> LinuxResources {
    let mut resources = LinuxResources::with_root(root, &source::fallback(4));
    resources.ns_per_tick = 10_000_000;
    resources.page_size = 4096;
    resources
}

#[test]
fn reads_a_v2_cgroup() {
    let dir = v2_fixture();
    let mut resources = provider(dir.path());
    let sample = resources.sample_fast_at(5);
    assert_eq!(sample.check_shape(), Ok(()));
    assert_eq!(sample.effective_cpus, CpuSet::range(0, 2));
    assert_eq!(sample.quota_capacity_cpus.get(), Some(0.5));
    assert_eq!(sample.quota_period_ns.get(), Some(100_000_000));
    assert_eq!(sample.productive_cpu_ns.get(), Some(1_000_000));
    assert_eq!(sample.scope_quota_throttled_ns.get(), Some(20_000));
    assert_eq!(sample.scope_psi_stall_ns.get(), Some(7_000));
    assert_eq!(sample.cpu_signals[0].steal.get(), Some(20_000_000));
    assert_eq!(sample.cpu_signals[1].scheduler_wait.get(), Some(6_000));
    assert_eq!(
        sample.cpu_signals[2].steal.validity(),
        SignalValidity::Unsupported,
        "not effective"
    );

    let memory = &sample.memory_signals;
    assert_eq!(memory.hard_limit_bytes.get(), Some(1_048_576));
    assert_eq!(
        memory.high_limit_bytes.validity(),
        SignalValidity::Unsupported
    );
    assert_eq!(memory.usage_bytes.get(), Some(524_288));
    assert_eq!(memory.inactive_file_bytes.get(), Some(4096));
    assert_eq!(memory.cumulative_reclaim_bytes.get(), Some(5 * 4096));
    assert_eq!(memory.memory_stall_ns.get(), Some(11_000));

    let io = &sample.io_signals;
    assert_eq!(io.productive_bytes.get(), Some(150), "scoped io.stat");
    assert_eq!(io.stall_ns.get(), Some(13_000));
    // Only the whole disk sda counts: 7 + 3 ms over 30 operations, two in flight.
    assert_eq!(io.operation_latency_ns.get(), Some(10_000_000));
    assert_eq!(io.completed_operations.get(), Some(30));
    assert_eq!(io.maximum_queue_depth.get(), Some(2.0));
}

#[test]
fn unlimited_limits() {
    let dir = v2_fixture();
    write(dir.path(), &format!("{CG}/cpu.max"), "max 250000\n");
    write(dir.path(), &format!("{CG}/memory.max"), "max\n");
    write(dir.path(), &format!("{CG}/memory.high"), "4096\n");
    let sample = provider(dir.path()).sample_fast_at(5);
    assert_eq!(
        sample.quota_capacity_cpus.validity(),
        SignalValidity::Unsupported
    );
    assert_eq!(sample.quota_period_ns.get(), Some(250_000_000));
    assert_eq!(
        sample.memory_signals.hard_limit_bytes.get(),
        Some(8 * 1024),
        "bounded by the host"
    );
    assert_eq!(sample.memory_signals.high_limit_bytes.get(), Some(4096));
}

#[test]
fn malformed_and_vanished_sources_are_transient_and_absent_ones_unsupported() {
    let dir = v2_fixture();
    let mut resources = provider(dir.path());
    resources.sample_fast_at(1);
    write(dir.path(), &format!("{CG}/cpu.max"), "0 100000\n");
    write(dir.path(), &format!("{CG}/cpu.stat"), "usage_usec abc\n");
    std::fs::remove_file(dir.path().join(format!("{CG}/memory.current"))).unwrap();
    let sample = resources.sample_fast_at(2);
    assert_eq!(
        sample.quota_capacity_cpus.validity(),
        SignalValidity::TransientFailure
    );
    assert_eq!(
        sample.productive_cpu_ns.validity(),
        SignalValidity::TransientFailure
    );
    assert_eq!(
        sample.scope_quota_throttled_ns.validity(),
        SignalValidity::TransientFailure,
        "a key that vanished"
    );
    assert_eq!(
        sample.memory_signals.usage_bytes.validity(),
        SignalValidity::TransientFailure
    );

    let fresh = provider(dir.path()).sample_fast_at(3);
    assert_eq!(
        fresh.scope_quota_throttled_ns.validity(),
        SignalValidity::Unsupported,
        "a key never seen"
    );
    assert_eq!(
        fresh.scope_scheduler_wait_ns.validity(),
        SignalValidity::Unsupported
    );
}

#[test]
fn a_missing_throttle_key_stays_unsupported() {
    let dir = v2_fixture();
    write(dir.path(), &format!("{CG}/cpu.stat"), "usage_usec 10\n");
    let mut resources = provider(dir.path());
    for at in 1..4 {
        let sample = resources.sample_fast_at(at);
        assert_eq!(
            sample.scope_quota_throttled_ns.validity(),
            SignalValidity::Unsupported
        );
        assert_eq!(sample.productive_cpu_ns.get(), Some(10_000));
    }
}

#[test]
fn a_bare_host_reads_proc() {
    let dir = TempDir::new("resources-bare");
    let root = dir.path();
    write(
        root,
        "proc/stat",
        "cpu  10 2 5 100 1 1 1 3\ncpu0 10 2 5 100 1 1 1 3\n",
    );
    write(
        root,
        "proc/pressure/cpu",
        "some avg10=0 avg60=0 avg300=0 total=3\n",
    );
    write(
        root,
        "proc/pressure/io",
        "some avg10=0 avg60=0 avg300=0 total=4\n",
    );
    write(root, "proc/meminfo", "MemTotal: 8 kB\nMemAvailable: 2 kB\n");
    write(
        root,
        "proc/vmstat",
        "pgsteal_kswapd_normal 4\npgsteal_direct_normal 1\n",
    );
    write(root, "proc/diskstats", "8 0 sda 1 0 8 1 1 0 8 1 0 0 0\n");
    write(root, "sys/block/sda/device/model", "");
    let sample = provider(root).sample_fast_at(5);
    assert_eq!(
        sample.effective_cpus,
        CpuSet::range(0, 4),
        "the topology's CPUs"
    );
    assert_eq!(
        sample.quota_capacity_cpus.validity(),
        SignalValidity::Unsupported
    );
    assert_eq!(sample.productive_cpu_ns.get(), Some(19 * 10_000_000));
    assert_eq!(sample.scope_psi_stall_ns.get(), Some(3_000));
    assert_eq!(sample.memory_signals.hard_limit_bytes.get(), Some(8 * 1024));
    assert_eq!(sample.memory_signals.usage_bytes.get(), Some(6 * 1024));
    assert_eq!(
        sample.memory_signals.cumulative_reclaim_bytes.get(),
        Some(5 * 4096)
    );
    assert_eq!(
        sample.io_signals.productive_bytes.get(),
        Some(16 * 512),
        "host diskstats"
    );
    assert_eq!(sample.io_signals.stall_ns.get(), Some(4_000));
    assert_eq!(
        sample.cpu_signals[0].scheduler_wait.validity(),
        SignalValidity::Unsupported
    );
}

#[test]
fn reads_v1_controllers() {
    let dir = TempDir::new("resources-v1");
    let root = dir.path();
    write(
        root,
        "proc/self/mountinfo",
        "31 25 0:27 / /sys/fs/cgroup/cpu,cpuacct rw - cgroup cgroup rw,cpu,cpuacct\n\
         32 25 0:28 / /sys/fs/cgroup/memory rw - cgroup cgroup rw,memory\n\
         33 25 0:29 / /sys/fs/cgroup/cpuset rw - cgroup cgroup rw,cpuset\n",
    );
    write(
        root,
        "proc/self/cgroup",
        "4:cpu,cpuacct:/c\n3:memory:/c\n2:cpuset:/c\n",
    );
    write(
        root,
        "sys/fs/cgroup/cpu,cpuacct/c/cpu.cfs_quota_us",
        "150000\n",
    );
    write(
        root,
        "sys/fs/cgroup/cpu,cpuacct/c/cpu.cfs_period_us",
        "100000\n",
    );
    write(
        root,
        "sys/fs/cgroup/cpu,cpuacct/c/cpuacct.usage",
        "123456\n",
    );
    write(
        root,
        "sys/fs/cgroup/cpu,cpuacct/c/cpu.stat",
        "nr_periods 1\nnr_throttled 1\nthrottled_time 999\n",
    );
    write(
        root,
        "sys/fs/cgroup/memory/c/memory.limit_in_bytes",
        "9223372036854771712\n",
    );
    write(
        root,
        "sys/fs/cgroup/memory/c/memory.usage_in_bytes",
        "2048\n",
    );
    write(
        root,
        "sys/fs/cgroup/memory/c/memory.stat",
        "cache 1\ntotal_inactive_file 512\n",
    );
    write(root, "sys/fs/cgroup/cpuset/c/cpuset.cpus", "1,3\n");
    write(root, "proc/meminfo", "MemTotal: 4 kB\n");
    let sample = provider(root).sample_fast_at(5);
    assert_eq!(sample.effective_cpus, [1, 3].into_iter().collect());
    assert_eq!(sample.quota_capacity_cpus.get(), Some(1.5));
    assert_eq!(sample.quota_period_ns.get(), Some(100_000_000));
    assert_eq!(sample.productive_cpu_ns.get(), Some(123_456));
    assert_eq!(sample.scope_quota_throttled_ns.get(), Some(999));
    assert_eq!(
        sample.memory_signals.hard_limit_bytes.get(),
        Some(4096),
        "unlimited, bounded by the host"
    );
    assert_eq!(sample.memory_signals.usage_bytes.get(), Some(2048));
    assert_eq!(sample.memory_signals.inactive_file_bytes.get(), Some(512));
    assert_eq!(
        sample.memory_signals.cumulative_reclaim_bytes.validity(),
        SignalValidity::Unsupported
    );
    assert_eq!(
        sample.scope_psi_stall_ns.validity(),
        SignalValidity::Unsupported,
        "host PSI is not the scope's"
    );

    write(root, "sys/fs/cgroup/cpu,cpuacct/c/cpu.cfs_quota_us", "-1\n");
    let unlimited = provider(root).sample_fast_at(6);
    assert_eq!(
        unlimited.quota_capacity_cpus.validity(),
        SignalValidity::Unsupported
    );
}

#[test]
fn slow_samples_report_frequency_caps_thermal_trips_and_power_profile() {
    let dir = TempDir::new("resources-slow");
    let root = dir.path();
    write(
        root,
        "sys/devices/system/cpu/cpu1/cpufreq/scaling_max_freq",
        "2000000\n",
    );
    write(
        root,
        "sys/devices/system/cpu/cpu1/cpufreq/cpuinfo_max_freq",
        "4000000\n",
    );
    let zone = "sys/class/thermal/thermal_zone0";
    write(root, &format!("{zone}/temp"), "86000\n");
    write(root, &format!("{zone}/trip_point_0_type"), "passive\n");
    write(root, &format!("{zone}/trip_point_0_temp"), "85000\n");
    write(root, &format!("{zone}/trip_point_1_type"), "critical\n");
    write(root, &format!("{zone}/trip_point_1_temp"), "105000\n");
    write(root, "sys/class/thermal/thermal_zone1/temp", "99000\n");
    write(root, "sys/firmware/acpi/platform_profile", "low-power\n");
    let sample = provider(root).sample_slow_at(9);
    assert_eq!(sample.check_shape(), Ok(()));
    assert_eq!(
        sample.cpu_signals[1].constrained_frequency_hz.get(),
        Some(2_000_000_000)
    );
    assert_eq!(
        sample.cpu_signals[1].nominal_frequency_hz.get(),
        Some(4_000_000_000)
    );
    assert_eq!(
        sample.cpu_signals[0].nominal_frequency_hz.validity(),
        SignalValidity::Unsupported
    );
    assert_eq!(
        sample.system_signals.thermal_severity.get(),
        Some(ThermalSeverity::Serious)
    );
    assert_eq!(sample.system_signals.low_power_mode.get(), Some(true));

    write(root, &format!("{zone}/temp"), "81000\n");
    assert_eq!(
        provider(root)
            .sample_slow_at(10)
            .system_signals
            .thermal_severity
            .get(),
        Some(ThermalSeverity::Fair)
    );
    let empty = TempDir::new("resources-slow-empty");
    let sample = provider(empty.path()).sample_slow_at(11);
    assert_eq!(
        sample.system_signals.thermal_severity.validity(),
        SignalValidity::Unsupported
    );
    assert_eq!(
        sample.system_signals.low_power_mode.validity(),
        SignalValidity::Unsupported
    );
}

#[test]
fn legacy_snapshots_difference_their_own_run_delay() {
    let dir = v2_fixture();
    let mut snapshots = LinuxSnapshots::new(provider(dir.path()));
    let first = snapshots.snapshot().unwrap();
    assert_eq!(first.pressure_per_cpu.iter().sum::<f64>(), 0.0);
    assert_eq!(first.quota_cpus, 0.5);
    assert_eq!(first.period_ns, 100_000_000);
    assert_eq!(first.memory_limit_bytes, 1_048_576);
    write(
        dir.path(),
        "proc/schedstat",
        "cpu0 0 0 1 1 1 1 100 4000 5\ncpu1 0 0 1 1 1 1 100 900000000000 5\n",
    );
    let second = snapshots.snapshot().unwrap();
    assert_eq!(second.pressure_per_cpu[0], 0.0);
    assert!(second.pressure_per_cpu[1] > 0.0);
}
