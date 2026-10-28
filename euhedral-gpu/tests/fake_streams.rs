//! Ordering, markers, retirement and poisoning on the fake backend. No GPU, no wall clock.
#![cfg(feature = "fake")]
#![allow(
    clippy::disallowed_methods,
    reason = "peer threads stand in for the lattice's workers, each owning a stream"
)]

use std::sync::{Arc, Mutex};

use euhedral_gpu::fake::{FakeFault, FakeGpu, FakeOp, FakeStream};
use euhedral_gpu::{
    Args, CompileOptions, Enqueue, Gpu, GpuError, Launch, Module, RetireListener, RetireTicket,
    Stream, StreamOptions,
};
use euhedral_graph::{LaneId, Marker, QueueId};

struct Record(Arc<Mutex<Vec<RetireTicket>>>);

impl RetireListener for Record {
    fn retired(self: Box<Self>, ticket: RetireTicket) {
        self.0.lock().unwrap().push(ticket);
    }
}

fn gpu_with_kernels() -> Arc<FakeGpu> {
    let gpu = FakeGpu::new();
    gpu.define("add1", |mem, _launch, args| {
        let at = args.read::<u64>(0).unwrap();
        let word = mem.bytes_mut(at, 4)?;
        let v = u32::from_le_bytes(word.try_into().unwrap()) + 1;
        word.copy_from_slice(&v.to_le_bytes());
        Ok(())
    });
    gpu.define("a", |_, _, _| Ok(()));
    gpu.define("b", |_, _, _| Ok(()));
    gpu.define("illegal", |_, _, _| Err(FakeFault::IllegalAddress));
    gpu
}

fn module(gpu: &Arc<FakeGpu>) -> euhedral_gpu::fake::FakeModule {
    euhedral_gpu::fake::FakeModule::from_source(gpu, "// fake source", &CompileOptions::default())
        .unwrap()
}

fn lanes(gpu: &Arc<FakeGpu>, names: &[&str]) -> (Arc<euhedral_graph::LanePool>, Vec<FakeStream>) {
    gpu.open_lanes(names, StreamOptions::default()).unwrap()
}

fn add1(m: &euhedral_gpu::fake::FakeModule, at: u64) -> (euhedral_gpu::fake::FakeKernel, Args) {
    (m.kernel("add1").unwrap(), Args::new().device(at))
}

#[test]
fn ops_on_one_stream_run_in_submission_order() {
    let gpu = gpu_with_kernels();
    let m = module(&gpu);
    let (_pool, mut s) = lanes(&gpu, &["compute"]);
    let buf = gpu.alloc(4);
    let mut out = vec![9u8; 4];
    let (k, args) = add1(&m, buf);
    s[0].submit(|q| {
        // SAFETY: `buf` is a live fake allocation and `out` outlives the run below.
        unsafe {
            q.memset(buf, 0, 4)?;
            q.launch(&k, Launch::linear(1, 1), &args)?;
            q.copy_to_host(out.as_mut_ptr(), buf, 4)
        }
    })
    .unwrap();
    gpu.run_until_idle();
    assert_eq!(
        gpu.log(),
        [
            FakeOp::Memset { lane: 0 },
            FakeOp::Launch {
                lane: 0,
                kernel: "add1".into()
            },
            FakeOp::Copy { lane: 0 },
        ]
    );
    assert_eq!(u32::from_le_bytes(out.try_into().unwrap()), 1);
}

#[test]
fn an_await_orders_work_behind_another_streams_marker() {
    let gpu = gpu_with_kernels();
    let m = module(&gpu);
    let (pool, mut s) = lanes(&gpu, &["a", "b"]);
    let buf = gpu.alloc(4);
    let mut out = vec![0u8; 4];
    let (k, args) = add1(&m, buf);
    s[0].submit(|q| {
        // SAFETY: live fake allocation.
        unsafe { q.launch(&k, Launch::linear(1, 1), &args) }
    })
    .unwrap();
    let marker = s[0].mark().unwrap();
    s[1].await_marker(marker).unwrap();
    s[1].submit(|q| {
        // SAFETY: `out` outlives the run below.
        unsafe { q.copy_to_host(out.as_mut_ptr(), buf, 4) }
    })
    .unwrap();
    assert!(gpu.step(), "the launch");
    assert!(!pool.reached(marker));
    gpu.run_until_idle();
    assert!(pool.reached(marker));
    assert_eq!(u32::from_le_bytes(out.try_into().unwrap()), 1);
    let log = gpu.log();
    let mark = log
        .iter()
        .position(|o| *o == FakeOp::Mark { lane: 0, value: 1 })
        .unwrap();
    let copy = log
        .iter()
        .position(|o| *o == FakeOp::Copy { lane: 1 })
        .unwrap();
    assert!(mark < copy, "the copy waited for the marker");
}

#[test]
fn markers_complete_as_the_device_reaches_them() {
    let gpu = gpu_with_kernels();
    let (pool, mut s) = lanes(&gpu, &["a"]);
    let m = s[0].mark().unwrap();
    assert!(!pool.reached(m));
    gpu.run_until_idle();
    assert!(pool.reached(m));
}

#[test]
fn retirement_runs_each_listener_once_in_order() {
    let gpu = gpu_with_kernels();
    let (_pool, mut s) = lanes(&gpu, &["a"]);
    let seen = Arc::new(Mutex::new(Vec::new()));
    for _ in 0..3 {
        s[0].notify_retired(Box::new(Record(seen.clone()))).unwrap();
    }
    gpu.run_until_idle();
    assert!(
        seen.lock().unwrap().is_empty(),
        "the device only queued the outcomes; nothing is told until the owner polls"
    );
    assert!(s[0].poll_retired().is_none());
    assert!(s[0].poll_retired().is_none(), "each exactly once");
    let seen = seen.lock().unwrap();
    assert_eq!(seen.iter().map(|t| t.seq()).collect::<Vec<_>>(), [0, 1, 2]);
    assert!(seen.iter().all(RetireTicket::ok));
}

#[test]
fn kept_values_live_until_confirmed_or_the_stream_drops() {
    let gpu = gpu_with_kernels();
    let (pool, mut s) = lanes(&gpu, &["a"]);
    let mut s = s.remove(0);
    let (confirmed, forgotten) = (Arc::new(()), Arc::new(()));
    let seen = Arc::new(Mutex::new(Vec::new()));
    s.submit(|q| {
        q.keep(Box::new(confirmed.clone()));
        Ok(())
    })
    .unwrap();
    s.notify_retired(Box::new(Record(seen.clone()))).unwrap();
    s.submit(|q| {
        q.keep(Box::new(forgotten.clone()));
        Ok(())
    })
    .unwrap();
    gpu.run_until_idle();
    s.poll_retired();
    assert_eq!(
        Arc::strong_count(&confirmed),
        2,
        "retired, not yet confirmed"
    );
    let ticket = seen.lock().unwrap()[0];
    assert_eq!(s.confirm_retired(ticket), None);
    assert_eq!(Arc::strong_count(&confirmed), 1);
    assert_eq!(
        Arc::strong_count(&forgotten),
        2,
        "kept for a later retirement"
    );
    drop((s, pool));
    assert_eq!(Arc::strong_count(&forgotten), 1, "released with the stream");
}

#[test]
fn submitters_move_to_their_own_threads_and_stay_contiguous() {
    let gpu = gpu_with_kernels();
    let m = module(&gpu);
    let (_pool, streams) = lanes(&gpu, &["a", "b"]);
    // Each lane's submitter is owned by one thread, as a lattice worker owns it. Nothing is locked: a submission is
    // contiguous on its stream because nothing else can enqueue on it.
    std::thread::scope(|scope| {
        for (mut stream, name) in streams.into_iter().zip(["a", "b"]) {
            let k = m.kernel(name).unwrap();
            scope.spawn(move || {
                for _ in 0..100 {
                    stream
                        .submit(|q| {
                            for _ in 0..3 {
                                // SAFETY: the fake kernels touch no memory.
                                unsafe { q.launch(&k, Launch::linear(1, 1), &Args::new())? };
                            }
                            Ok(())
                        })
                        .unwrap();
                }
            });
        }
    });
    gpu.run_until_idle();
    let log = gpu.log();
    assert_eq!(log.len(), 600);
    for lane in [0u16, 1] {
        let names: Vec<&str> = log
            .iter()
            .filter_map(|o| match o {
                FakeOp::Launch { lane: l, kernel } if *l == lane => Some(kernel.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(names.len(), 300);
        assert!(names.iter().all(|n| *n == names[0]), "one kernel per lane");
    }
}

#[test]
fn unknown_kernels_are_not_found() {
    let gpu = gpu_with_kernels();
    let m = module(&gpu);
    assert!(matches!(m.kernel("nope"), Err(GpuError::NotFound(_))));
}

fn faulting_kernel(gpu: &Arc<FakeGpu>) -> euhedral_gpu::fake::FakeKernel {
    module(gpu).kernel("illegal").unwrap()
}

#[test]
fn a_fault_surfaces_through_confirm_retired_and_poisons_the_device() {
    let gpu = gpu_with_kernels();
    let (_pool, mut s) = lanes(&gpu, &["a", "b"]);
    let k = faulting_kernel(&gpu);
    let seen = Arc::new(Mutex::new(Vec::new()));
    s[0].submit(|q| {
        // SAFETY: the fake kernel touches no memory.
        unsafe { q.launch(&k, Launch::linear(1, 1), &Args::new()) }
    })
    .unwrap();
    s[0].notify_retired(Box::new(Record(seen.clone()))).unwrap();
    gpu.run_until_idle();
    s[0].poll_retired();
    let ticket = seen.lock().unwrap()[0];
    assert!(!ticket.ok());
    assert_eq!(ticket.status(), 700);
    assert!(matches!(
        s[0].confirm_retired(ticket),
        Some(GpuError::Fault { code: 700, .. })
    ));
    assert!(matches!(gpu.poisoned(), Some(GpuError::Fault { .. })));
    assert!(matches!(
        s[1].submit(|_| Ok(())),
        Err(GpuError::Poisoned(_))
    ));
    assert!(matches!(s[1].mark(), Err(GpuError::Poisoned(_))));
    assert!(matches!(
        s[0].notify_retired(Box::new(Record(seen.clone()))),
        Err(GpuError::Poisoned(_))
    ));
}

#[test]
fn work_queued_behind_a_fault_never_runs_but_its_markers_are_reached() {
    let gpu = gpu_with_kernels();
    let m = module(&gpu);
    let (pool, mut s) = lanes(&gpu, &["a"]);
    let bad = faulting_kernel(&gpu);
    let after = m.kernel("a").unwrap();
    let before_fault = s[0].mark().unwrap();
    s[0].submit(|q| {
        // SAFETY: the fake kernels touch no memory.
        unsafe {
            q.launch(&bad, Launch::linear(1, 1), &Args::new())?;
            q.launch(&after, Launch::linear(1, 1), &Args::new())
        }
    })
    .unwrap();
    let behind = s[0].mark().unwrap();
    gpu.run_until_idle();
    assert!(
        !gpu.log()
            .iter()
            .any(|o| matches!(o, FakeOp::Launch { kernel, .. } if kernel == "a")),
        "nothing runs behind the fault"
    );
    assert!(pool.reached(before_fault));
    assert!(pool.reached(behind), "no continuation is stranded");
}

#[test]
fn a_poisoned_device_still_emits_retire_into_continuations() {
    use euhedral_core::frames::RunnableFrame;
    use euhedral_core::generics::LatticeSource;
    let gpu = gpu_with_kernels();
    let (pool, mut s) = lanes(&gpu, &["a", "b"]);
    let bad = faulting_kernel(&gpu);
    s[0].submit(|q| {
        // SAFETY: the fake kernel touches no memory.
        unsafe { q.launch(&bad, Launch::linear(1, 1), &Args::new()) }
    })
    .unwrap();
    let after_fault = s[0].mark().unwrap();
    // Lane b waits for lane a's marker, so its listener cannot run before the fault is the only thing that happens.
    s[1].await_marker(after_fault).unwrap();
    let completions = euhedral_graph::Completions::new(pool);
    let confirmed = Arc::new(Mutex::new(None));
    // Lane b's submitter travels with its continuation.
    let b = s.remove(1);
    assert!(
        euhedral_gpu::retire_into(b, &completions, {
            let confirmed = confirmed.clone();
            move |mut stream, ticket| {
                Box::new(RunnableFrame::new(
                    1,
                    Arc::new({
                        let confirmed = confirmed.clone();
                        let outcome = stream.confirm_retired(ticket);
                        move || *confirmed.lock().unwrap() = Some(outcome.clone())
                    }),
                ))
            }
        })
        .is_ok()
    );
    assert!(gpu.step(), "only the faulting launch runs");
    assert!(gpu.poisoned().is_some());
    let mut emitted = 0;
    for _ in 0..1000 {
        if completions.pending() == 0 {
            break;
        }
        emitted += completions.pull(
            &mut |mut frame| {
                frame.execute().unwrap();
                frame.do_finally();
            },
            &mut |_| false,
            u64::MAX,
        );
    }
    assert_eq!(emitted, 1);
    assert!(matches!(
        confirmed.lock().unwrap().take(),
        Some(Some(GpuError::Poisoned(_)))
    ));
}

#[test]
fn a_ticket_settles_only_on_the_stream_that_armed_it() {
    let gpu = gpu_with_kernels();
    let (_pool, mut s) = lanes(&gpu, &["a", "b"]);
    let kept = Arc::new(());
    let (from_a, from_b) = (
        Arc::new(Mutex::new(Vec::new())),
        Arc::new(Mutex::new(Vec::new())),
    );
    s[0].submit(|q| {
        q.keep(Box::new(kept.clone()));
        Ok(())
    })
    .unwrap();
    s[0].notify_retired(Box::new(Record(from_a.clone())))
        .unwrap();
    s[1].notify_retired(Box::new(Record(from_b.clone())))
        .unwrap();
    gpu.run_until_idle();
    s[0].poll_retired();
    s[1].poll_retired();
    let (ticket_a, ticket_b) = (from_a.lock().unwrap()[0], from_b.lock().unwrap()[0]);
    assert_ne!(s[0].id(), s[1].id());
    assert_eq!(
        (ticket_a.stream(), ticket_b.stream()),
        (s[0].id(), s[1].id())
    );
    // Settling stream b's ticket on stream a is refused and releases nothing of a's.
    let refused = s[0].confirm_retired(ticket_b);
    assert!(matches!(refused, Some(GpuError::Invalid(_))), "{refused:?}");
    assert_eq!(Arc::strong_count(&kept), 2, "a's kept value is still held");
    assert_eq!(s[0].confirm_retired(ticket_a), None);
    assert_eq!(Arc::strong_count(&kept), 1);
    assert_eq!(s[1].confirm_retired(ticket_b), None);
}

#[test]
fn a_marker_of_another_pool_is_refused_not_treated_as_done() {
    let gpu = gpu_with_kernels();
    let (pool_a, mut a) = lanes(&gpu, &["a0", "a1"]);
    let (pool_b, mut b) = lanes(&gpu, &["b0", "b1"]);
    // A marker of pool b that has long completed: its lane exists in pool a, and its value is below a's own.
    let from_b = b[0].mark().unwrap();
    let first = a[0].mark().unwrap();
    a[0].mark().unwrap();
    gpu.run_until_idle();
    assert!(pool_b.reached(from_b) && pool_a.reached(first));
    let refused = a[1].await_marker(from_b);
    assert!(matches!(refused, Err(GpuError::Invalid(_))), "{refused:?}");
    // Not even its own lane's number saves it: pool a does not reach it either.
    assert!(!pool_a.reached(from_b));
    // The same lane and value stamped with a's queue is a's own marker, already done.
    let own = Marker::new(a[0].id(), from_b.lane, from_b.value);
    a[1].await_marker(own).unwrap();
    // Neither pool's queue: refused, and never reached.
    let stranger = Marker::new(QueueId::next(), LaneId(0), 1);
    assert!(matches!(
        a[0].await_marker(stranger),
        Err(GpuError::Invalid(_))
    ));
    assert!(!pool_a.reached(stranger));
    // A lane beyond the pool, with the right queue: invalid.
    let beyond = Marker::new(a[0].id(), LaneId(7), 1);
    assert!(matches!(
        a[0].await_marker(beyond),
        Err(GpuError::Invalid(_))
    ));
}

#[test]
fn a_poisoned_device_reports_the_retirements_still_outstanding() {
    let gpu = gpu_with_kernels();
    let (_pool, mut s) = lanes(&gpu, &["a", "b"]);
    let bad = faulting_kernel(&gpu);
    s[0].submit(|q| {
        // SAFETY: the fake kernel touches no memory.
        unsafe { q.launch(&bad, Launch::linear(1, 1), &Args::new()) }
    })
    .unwrap();
    let behind = s[0].mark().unwrap();
    // Lane b waits for lane a, so its retirements are still outstanding when the fault happens.
    s[1].await_marker(behind).unwrap();
    let seen = Arc::new(Mutex::new(Vec::new()));
    for _ in 0..2 {
        s[1].notify_retired(Box::new(Record(seen.clone()))).unwrap();
    }
    assert!(gpu.step(), "only the faulting launch runs");
    assert!(seen.lock().unwrap().is_empty());
    s[1].poll_retired();
    let seen = seen.lock().unwrap();
    assert_eq!(seen.len(), 2, "each reported once, none stranded");
    assert!(
        seen.iter()
            .all(|t| t.status() == euhedral_gpu::POISONED_STATUS)
    );
    assert_eq!(seen.iter().map(|t| t.seq()).collect::<Vec<_>>(), [0, 1]);
}
