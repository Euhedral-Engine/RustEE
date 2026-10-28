//! A deliberate device fault. A sticky fault poisons the context for the whole process, so this file holds one test
//! and runs as its own process. Skips, passing, when no driver or device is present.
#![cfg(feature = "cuda")]
#![allow(
    clippy::disallowed_methods,
    reason = "the test thread is not a lattice worker: it polls and sleeps"
)]

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use euhedral_core::frames::RunnableFrame;
use euhedral_core::generics::LatticeSource;
use euhedral_gpu::cuda::{CudaGpu, CudaModule};
use euhedral_gpu::{
    Args, CompileOptions, Enqueue, Gpu, GpuError, Launch, Module, RetireListener, RetireTicket,
    Stream, StreamOptions, retire_into,
};
use euhedral_graph::{Completions, DeviceQueue};

const OPS: &str = include_str!("kernels/ops.cu");
const OPS_PTX: &[u8] = include_bytes!("kernels/ops.ptx");

struct Record(Arc<Mutex<Vec<RetireTicket>>>);

impl RetireListener for Record {
    fn retired(self: Box<Self>, ticket: RetireTicket) {
        self.0.lock().unwrap().push(ticket);
    }
}

fn wait_until(limit: Duration, condition: impl Fn() -> bool) -> bool {
    let start = Instant::now();
    while !condition() {
        if start.elapsed() > limit {
            return false;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    true
}

#[test]
fn a_device_fault_surfaces_through_confirm_retired_and_poisons_the_device() {
    let Ok(gpu) = CudaGpu::open(0) else {
        eprintln!("no CUDA device: skipped");
        return;
    };
    let module = if gpu.driver_version() >= 13010 {
        CudaModule::from_image(&gpu, OPS_PTX)
    } else {
        CudaModule::from_source(&gpu, OPS, &CompileOptions::default())
    };
    let Ok(module) = module else {
        eprintln!("no way to load the test kernels: skipped");
        return;
    };
    let (spin, poke) = (
        module.kernel("spin").unwrap(),
        module.kernel("poke").unwrap(),
    );
    let (pool, s) = gpu
        .open_lanes(&["a", "b"], StreamOptions::default())
        .unwrap();
    let seen = Arc::new(Mutex::new(Vec::new()));
    // Everything is submitted while the spin runs, before the fault happens.
    s[0].submit(|q| {
        // SAFETY: `spin` touches no memory. `poke` writes through an address that is not mapped: the deliberate
        // fault of this test.
        unsafe {
            q.launch(
                &spin,
                Launch::linear(1, 1),
                &Args::new().arg(125_000_000u64),
            )?;
            q.launch(&poke, Launch::linear(1, 1), &Args::new().device(0x10))
        }
    })
    .unwrap();
    let after_poke = s[0].mark().unwrap();
    s[0].notify_retired(Box::new(Record(seen.clone()))).unwrap();
    s[1].await_marker(after_poke).unwrap();
    let completions = Completions::new(pool.clone());
    let confirmed = Arc::new(Mutex::new(Vec::new()));
    retire_into(&s[1], &completions, {
        let (stream, confirmed) = (s[1].clone(), confirmed.clone());
        move |ticket| {
            Box::new(RunnableFrame::new(
                1,
                Arc::new(move || {
                    confirmed
                        .lock()
                        .unwrap()
                        .push(stream.confirm_retired(ticket))
                }),
            ))
        }
    })
    .unwrap();

    // The listener runs once, with the device's error.
    assert!(wait_until(Duration::from_secs(10), || !seen
        .lock()
        .unwrap()
        .is_empty()));
    let ticket = seen.lock().unwrap()[0];
    assert_eq!(seen.lock().unwrap().len(), 1);
    assert!(!ticket.ok());
    assert_eq!(ticket.status(), 700);
    assert!(matches!(
        s[0].confirm_retired(ticket),
        Some(GpuError::Fault { code: 700, .. })
    ));
    assert!(matches!(
        gpu.poisoned(),
        Some(GpuError::Fault { code: 700, .. })
    ));

    // Later calls fail fast with the cause.
    assert!(matches!(
        s[1].submit(|_| Ok(())),
        Err(GpuError::Poisoned(_))
    ));
    assert!(matches!(s[1].mark(), Err(GpuError::Poisoned(_))));
    assert!(matches!(
        s[0].notify_retired(Box::new(Record(seen.clone()))),
        Err(GpuError::Poisoned(_))
    ));
    // Nothing is stranded: every marker reads as reached.
    assert!(pool.reached(after_poke));
    assert_eq!(s[0].completed(), after_poke.value);

    // The continuation behind the fault is still emitted by the lattice's own poll.
    assert!(wait_until(Duration::from_secs(10), || {
        completions.pull(
            &mut |mut frame| {
                frame.execute().unwrap();
                frame.do_finally();
            },
            &mut |_| false,
            u64::MAX,
        );
        completions.pending() == 0
    }));
    let confirmed = confirmed.lock().unwrap();
    assert_eq!(confirmed.len(), 1);
    assert!(
        confirmed[0].is_some(),
        "the fault is known to the continuation"
    );
}
