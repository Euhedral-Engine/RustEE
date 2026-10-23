//! The platform back end on the machine running the tests.

use euhedral_hardware_utils::affinity::{self, AffinityCapability, PinnedThreadExecutor};
use euhedral_hardware_utils::monitor::{self, MonitorError, ResourceMonitor};
use euhedral_hardware_utils::topology::{Topology, TopologyMapper};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[test]
fn the_system_topology_is_consistent() {
    let topology = Topology::system();
    assert!(topology.cpu_count() >= 1);
    assert!(topology.cpu_span() > topology.cpu_set().last().unwrap());
    for cpu in topology.cpu_set().iter() {
        assert!(topology.cpu_info(cpu).is_some(), "CPU {cpu}");
        assert!(topology.cache_layout(cpu).is_some(), "CPU {cpu}");
    }
}

#[test]
fn a_monitor_publishes_bounded_pressure() {
    let mapper = Arc::new(TopologyMapper::new());
    let monitor = match ResourceMonitor::with_period(mapper, Duration::from_millis(20)) {
        Err(MonitorError::Unsupported) => return,
        other => other.unwrap(),
    };
    monitor.start().unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    let utilization = loop {
        if let Some(utilization) = monitor.utilization() {
            break utilization;
        }
        assert!(Instant::now() < deadline, "no utilization within 5 s");
        std::thread::sleep(Duration::from_millis(10));
    };
    let span = Topology::system().cpu_span();
    assert_eq!(utilization.per_quota_cpu_pressure.len(), span);
    assert!(
        utilization
            .per_quota_cpu_pressure
            .iter()
            .all(|pressure| (0.0..=1.0).contains(pressure))
    );
    assert!((0.0..=1.0).contains(&utilization.quota_cpu_usage));
    assert!(monitor.close_timeout(Duration::from_secs(5)));
    if cfg!(any(target_os = "linux", target_os = "macos", windows)) {
        assert!(monitor::system_snapshot().is_some());
    }
}

#[test]
fn a_pinned_task_runs_on_its_cpu() {
    let cpu = affinity::base_mask().first().unwrap();
    let executor = PinnedThreadExecutor::get_or_create(cpu, "smoke").unwrap();
    let observed = executor
        .submit(affinity::current_cpu)
        .unwrap()
        .join()
        .unwrap();
    if affinity::capability() == AffinityCapability::Exact && observed.is_some() {
        assert_eq!(observed, Some(cpu));
    }
    executor.close();
}
