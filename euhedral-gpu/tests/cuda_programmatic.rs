//! Programmatic dependent launches, eagerly and from a recorded graph. Every test skips, passing, when no driver,
//! device or run-time compiler is present, or the device is older than compute capability 9.0.
#![cfg(feature = "cuda")]
#![allow(
    clippy::disallowed_methods,
    reason = "the test thread is not a lattice worker: it polls and sleeps"
)]

use std::time::{Duration, Instant};

use euhedral_gpu::cuda::{CudaGpu, CudaKernel, CudaModule, CudaStream};
use euhedral_gpu::{
    Args, CompileOptions, Gpu, GpuError, Graph, LaneEnqueue, Launch, Module, Sequence, Stream,
    StreamOptions,
};
use euhedral_memory::cuda::{DeviceMemory, PinnedHost};
use euhedral_memory::{Backend, LocationId};

const N: usize = 4096;
const BYTES: usize = N * 4;

/// `consume` waits for the grid before it, then reads what that grid wrote.
const SOURCE: &str = r#"
extern "C" __global__ void produce(unsigned* y, unsigned base, unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = base + i * 7u;
}
extern "C" __global__ void consume(unsigned* z, const unsigned* y, unsigned c, unsigned n) {
    asm volatile("griddepcontrol.wait;");
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) z[i] = y[i] + c;
}
"#;

struct Rig {
    produce: CudaKernel,
    consume: CudaKernel,
    y: u64,
    z: u64,
    out: *mut u32,
}

impl Rig {
    /// The chain: `y[i] = base + 7 i`, then, programmatically, `z[i] = y[i] + c`, then a copy of z to the host.
    fn chain<S: Sequence<Kernel = CudaKernel>>(
        &self,
        q: &mut S,
        base: u32,
        c: u32,
        programmatic: bool,
    ) -> Result<(), GpuError> {
        let first = Args::new().device(self.y).arg(base).arg(N as u32);
        let second = Args::new()
            .device(self.z)
            .device(self.y)
            .arg(c)
            .arg(N as u32);
        q.submit(0, |e| {
            // SAFETY: matching arguments over live blocks that the test keeps until the work ends.
            unsafe {
                e.launch(&self.produce, Launch::linear(N as u32, 256), &first)?;
                e.launch(
                    &self.consume,
                    Launch::linear(N as u32, 256).programmatic(programmatic),
                    &second,
                )?;
                e.copy_to_host(self.out.cast(), self.z, BYTES)
            }
        })
    }

    fn result(&self) -> Vec<u32> {
        // SAFETY: a pinned block of N words that the test reads only between settled runs.
        unsafe { std::slice::from_raw_parts(self.out, N) }.to_vec()
    }

    fn clear(&self) {
        // SAFETY: as in `result`.
        unsafe { std::slice::from_raw_parts_mut(self.out, N) }.fill(0);
    }
}

fn expected(base: u32, c: u32) -> Vec<u32> {
    (0..N as u32).map(|i| base + i * 7 + c).collect()
}

fn settle(s: &mut CudaStream, pool: &euhedral_graph::LanePool) {
    let m = s.mark().unwrap();
    let start = Instant::now();
    while !pool.reached(m) {
        assert!(start.elapsed() < Duration::from_secs(10), "the work ends");
        std::thread::sleep(Duration::from_millis(1));
    }
}

#[test]
fn a_programmatic_chain_gives_the_same_result_eagerly_and_from_a_recorded_updated_graph() {
    let gpu = match CudaGpu::open(0) {
        Ok(gpu) => gpu,
        Err(e) => {
            eprintln!("no CUDA device: skipped ({e})");
            return;
        }
    };
    if gpu.arch().0 < 9 {
        eprintln!("compute capability below 9.0: skipped");
        return;
    }
    let module = match CudaModule::from_source(&gpu, SOURCE, &CompileOptions::default()) {
        Ok(module) => module,
        Err(e) => {
            eprintln!("no way to compile the test kernels: skipped ({e})");
            return;
        }
    };
    let (pool, mut s) = gpu.open_lanes(&["a"], StreamOptions::default()).unwrap();
    let mem = DeviceMemory {
        device: gpu.device().clone(),
        location: LocationId(0),
    };
    let pinned = PinnedHost::new(gpu.device().clone(), LocationId(1), 0);
    let (y, z) = (
        mem.alloc(BYTES as u64).unwrap(),
        mem.alloc(BYTES as u64).unwrap(),
    );
    let out = pinned.alloc(BYTES as u64).unwrap();
    let rig = Rig {
        produce: module.kernel("produce").unwrap(),
        consume: module.kernel("consume").unwrap(),
        y: y.addr,
        z: z.addr,
        out: out.addr as *mut u32,
    };

    // Eagerly.
    rig.clear();
    s[0].submit(|e| {
        // SAFETY: as in `chain`.
        unsafe {
            e.launch(
                &rig.produce,
                Launch::linear(N as u32, 256),
                &Args::new().device(rig.y).arg(10u32).arg(N as u32),
            )?;
            e.launch(
                &rig.consume,
                Launch::linear(N as u32, 256).programmatic(true),
                &Args::new()
                    .device(rig.z)
                    .device(rig.y)
                    .arg(3u32)
                    .arg(N as u32),
            )?;
            e.copy_to_host(rig.out.cast(), rig.z, BYTES)
        }
    })
    .unwrap();
    settle(&mut s[0], &pool);
    assert_eq!(rig.result(), expected(10, 3), "eagerly");

    // Recorded (the recording runs the work for real), then replayed.
    rig.clear();
    let mut graph = gpu
        .record(&mut [&mut s[0]], |q| rig.chain(q, 20, 4, true))
        .unwrap();
    settle(&mut s[0], &pool);
    assert_eq!(rig.result(), expected(20, 4), "the recording's own run");
    rig.clear();
    graph.replay(&mut s[0]).unwrap();
    settle(&mut s[0], &pool);
    assert_eq!(rig.result(), expected(20, 4), "the replay");

    // The capture holds a programmatic edge into the second kernel (and a plain one without the flag).
    match graph.edge_types_into(1) {
        Some(types) => {
            assert_eq!(types, [1], "a programmatic edge into the consumer");
            let plain = gpu
                .record(&mut [&mut s[0]], |q| rig.chain(q, 1, 1, false))
                .unwrap();
            settle(&mut s[0], &pool);
            assert_eq!(plain.edge_types_into(1), Some(vec![0]), "a plain edge");
        }
        None => eprintln!("the driver cannot report edge types: the edge assertion is skipped"),
    }

    // After an update pass with new arguments.
    rig.clear();
    gpu.update(&mut graph, |q| rig.chain(q, 30, 5, true))
        .unwrap();
    graph.replay(&mut s[0]).unwrap();
    settle(&mut s[0], &pool);
    assert_eq!(rig.result(), expected(30, 5), "the replay after an update");

    // A body without the flag is another shape.
    let r = gpu.update(&mut graph, |q| {
        q.submit(0, |e| {
            // SAFETY: nothing is enqueued on an update.
            unsafe {
                e.launch(
                    &rig.produce,
                    Launch::linear(N as u32, 256),
                    &Args::new().device(rig.y).arg(1u32).arg(N as u32),
                )?;
                e.launch(
                    &rig.consume,
                    Launch::linear(N as u32, 256),
                    &Args::new()
                        .device(rig.z)
                        .device(rig.y)
                        .arg(1u32)
                        .arg(N as u32),
                )?;
                e.copy_to_host(rig.out.cast(), rig.z, BYTES)
            }
        })
    });
    assert!(matches!(r, Err(GpuError::ShapeChanged { .. })), "{r:?}");

    drop(graph);
    // SAFETY: nothing runs on the device any more; the blocks are unused.
    unsafe {
        mem.free(y);
        mem.free(z);
        pinned.free(out);
    }
}
