use super::*;
use std::sync::Weak;

#[test]
fn names_are_upstreams_except_service_time() {
    assert_eq!(DEFAULT_PREFIX, "euhedral");
    assert_eq!(CORE_TAG, "core");
    let names = [
        (THROUGHPUT_SUMMARY_SUFFIX, "euhedral.execution.throughput"),
        (IN_PROGRESS_SUFFIX, "euhedral.execution.inProgress"),
        (CACHE_BACKLOG_SUFFIX, "euhedral.cache.backlog"),
        (CAP_FACTOR_SUFFIX, "euhedral.cache.capFactor"),
        (FRAMES_SUFFIX, "euhedral.execution.frames"),
        (SERVICE_TIME_SUFFIX, "euhedral.execution.serviceTime"),
        (
            SERVICE_TIME_FLOW_SUFFIX,
            "euhedral.execution.serviceTime.flow",
        ),
        (THROUGHPUT_FLOW_SUFFIX, "euhedral.execution.throughput.flow"),
    ];
    for (suffix, name) in names {
        assert_eq!(metric_name(DEFAULT_PREFIX, suffix), name);
    }
}

#[test]
fn a_prefix_loses_its_trailing_dot_and_a_blank_one_is_the_default() {
    assert_eq!(
        metric_name("app.", SERVICE_TIME_SUFFIX),
        "app.execution.serviceTime"
    );
    assert_eq!(
        metric_name("my.app", CAP_FACTOR_SUFFIX),
        "my.app.cache.capFactor"
    );
    assert_eq!(
        metric_name("", SERVICE_TIME_SUFFIX),
        "euhedral.execution.serviceTime"
    );
    assert_eq!(
        metric_name("  ", SERVICE_TIME_SUFFIX),
        "euhedral.execution.serviceTime"
    );
    assert_eq!(MetricsConfig::default().prefix(), "euhedral");
    assert_eq!(
        MetricsConfig::default().with_prefix("").prefix(),
        "euhedral"
    );
    assert_eq!(MetricsConfig::default().with_prefix("x").prefix(), "x");
}

#[test]
fn without_a_registry_nothing_is_built() {
    let config = MetricsConfig::default().with_prefix("app");
    assert!(ExecutionMetrics::new(&config, 0, 0).is_none());
    assert!(CacheMetrics::new(&config, 0, 0).is_none());
}

/// A source with a fixed set of meters.
struct Fixed(Arc<CachePadded<ExecutionMetrics>>);

impl MeterSource for Fixed {
    fn collect(&self, out: &mut Vec<Meter>) {
        self.0.collect(3, out);
    }
}

fn fixed(registry: &Arc<MetricsRegistry>, prefix: &str, core: usize, cpu: usize) -> Arc<Fixed> {
    let config = MetricsConfig::default()
        .with_prefix(prefix)
        .with_registry(Some(registry.clone()));
    let source = Arc::new(Fixed(ExecutionMetrics::new(&config, core, cpu).unwrap()));
    let weak: Weak<dyn MeterSource> = Arc::downgrade(&source) as Weak<dyn MeterSource>;
    registry.register(weak);
    source
}

#[test]
fn execution_meters_report_what_was_recorded() {
    let registry = MetricsRegistry::new();
    let source = fixed(&registry, "app", 4, 9);
    source.0.record_frames(5);
    source.0.record_frames(2);
    source
        .0
        .record_service_time(std::time::Duration::from_nanos(250));
    assert_eq!(source.0.service_time_totals(), (1, 250.0));
    source.0.report_throughput(0.002);

    let snapshot = registry.snapshot();
    let get = |name| snapshot.get(name, 9).unwrap();
    assert_eq!(get("app.execution.frames").counter(), Some(7));
    // The gauge says whether the worker runs frames: any count in progress is 1.
    assert_eq!(get("app.execution.inProgress").gauge(), Some(1.0));
    let service_time = get("app.execution.serviceTime").summary().unwrap();
    assert_eq!((service_time.count, service_time.sum), (1, 250.0));
    // 0.002 frames per nanosecond is 2,000,000 frames per second.
    let throughput = get("app.execution.throughput").summary().unwrap();
    assert_eq!(throughput.max, 2_000_000.0);
    let meter = get("app.execution.serviceTime");
    assert_eq!((meter.core, meter.cpu), (4, 9));
    assert_eq!(
        (meter.tag(CORE_TAG), meter.tag(CPU_TAG), meter.tag("x")),
        (Some(4), Some(9), None)
    );
    assert!(meter.gauge().is_none() && meter.counter().is_none() && meter.flow().is_none());
    assert!(get("app.execution.serviceTime.flow").flow().is_none());
    let stats = FlowStats {
        average: 4.0,
        standard_deviation: 1.0,
        trend: 0.5,
        min: 2.0,
        max: 6.0,
    };
    source.0.publish_service_time(&stats);
    source.0.publish_throughput(&stats);
    let snapshot = registry.snapshot();
    assert_eq!(
        snapshot
            .get("app.execution.serviceTime.flow", 9)
            .unwrap()
            .flow(),
        Some(&stats)
    );
    assert_eq!(
        snapshot
            .get("app.execution.throughput.flow", 9)
            .unwrap()
            .flow(),
        Some(&stats)
    );
}

#[test]
fn lookups_match_names_and_suffixes_like_the_aggregator() {
    let registry = MetricsRegistry::new();
    let _first = fixed(&registry, "a", 0, 0);
    let _second = fixed(&registry, "a", 0, 1);
    let _other = fixed(&registry, "b", 1, 2);
    let snapshot = registry.snapshot();
    assert_eq!(snapshot.meters().len(), 18);
    assert_eq!(snapshot.service_time_summaries().count(), 3);
    assert_eq!(snapshot.service_time_flows().count(), 3);
    assert_eq!(snapshot.throughput_flows().count(), 3);
    assert_eq!(snapshot.throughput_summaries().count(), 3);
    assert_eq!(snapshot.in_progress_gauges().count(), 3);
    assert_eq!(snapshot.frames_counters().count(), 3);
    assert_eq!(snapshot.cache_backlog_gauges().count(), 0);
    assert_eq!(snapshot.cap_factor_summaries().count(), 0);
    let prefixed = metric_name("a", SERVICE_TIME_SUFFIX);
    assert_eq!(snapshot.named(&prefixed).count(), 2);
    assert!(snapshot.get(&prefixed, 2).is_none());
    assert!(snapshot.get("b.execution.serviceTime", 2).is_some());
}

#[test]
fn sources_leave_when_removed_or_dropped() {
    let registry = MetricsRegistry::new();
    let kept = fixed(&registry, "a", 0, 0);
    let removed = fixed(&registry, "a", 0, 1);
    let dropped = fixed(&registry, "a", 0, 2);
    assert_eq!(registry.len(), 3);
    registry.remove(Arc::as_ptr(&removed).cast());
    drop(dropped);
    assert_eq!(registry.len(), 1);
    let snapshot = registry.snapshot();
    assert!(snapshot.meters().iter().all(|meter| meter.cpu == 0));
    drop(kept);
    assert!(registry.is_empty());
    assert_eq!(registry.snapshot(), MetricsSnapshot::default());
}

#[test]
fn registries_are_equal_only_to_themselves() {
    let registry = MetricsRegistry::new();
    assert_eq!(*registry, *registry.clone());
    assert_ne!(*registry, *MetricsRegistry::new());
    assert!(format!("{registry:?}").contains("sources: 0"));
}

#[test]
fn snapshots_read_while_writers_record() {
    let registry = MetricsRegistry::new();
    let sources: Vec<_> = (0..4).map(|cpu| fixed(&registry, "c", 0, cpu)).collect();
    let stop = std::sync::atomic::AtomicBool::new(false);
    std::thread::scope(|scope| {
        for source in &sources {
            let stop = &stop;
            scope.spawn(move || {
                let mut value = 0u64;
                while !stop.load(Ordering::Relaxed) || value < 1000 {
                    source.0.record_frames(1);
                    source
                        .0
                        .record_service_time(std::time::Duration::from_nanos(value % 1000));
                    value += 1;
                }
            });
        }
        let mut last = [0u64; 4];
        for _ in 0..200 {
            let snapshot = registry.snapshot();
            for meter in snapshot.frames_counters() {
                let frames = meter.counter().unwrap();
                assert!(frames >= last[meter.cpu]);
                last[meter.cpu] = frames;
            }
            for meter in snapshot.service_time_summaries() {
                let service_time = meter.summary().unwrap();
                assert!(service_time.buckets.iter().sum::<u64>() >= service_time.count);
            }
        }
        stop.store(true, Ordering::Relaxed);
    });
    let snapshot = registry.snapshot();
    for cpu in 0..4 {
        let frames = snapshot
            .get("c.execution.frames", cpu)
            .unwrap()
            .counter()
            .unwrap();
        let service_time = snapshot
            .get("c.execution.serviceTime", cpu)
            .unwrap()
            .summary()
            .unwrap();
        assert_eq!(service_time.count, frames);
        assert_eq!(service_time.buckets.iter().sum::<u64>(), frames);
    }
}
