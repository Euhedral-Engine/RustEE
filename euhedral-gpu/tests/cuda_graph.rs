//! CUDA graph record, replay, update and re-record. Every test skips, passing, when no driver or device is present.
#![cfg(feature = "cuda")]
#![allow(
    clippy::disallowed_methods,
    reason = "the test thread is not a lattice worker: it polls, sleeps and runs peer threads"
)]

use std::sync::Arc;
use std::time::{Duration, Instant};

use euhedral_gpu::cuda::{CudaGpu, CudaKernel, CudaModule, CudaStream};
use euhedral_gpu::{
    Args, CompileOptions, Enqueue, Gpu, GpuError, Graph, Launch, Module, Sequence, Stream,
    StreamOptions,
};
use euhedral_graph::{LanePool, Marker};
use euhedral_memory::cuda::{DeviceMemory, PinnedHost};
use euhedral_memory::{Backend, Block, LocationId};

const OPS: &str = include_str!("kernels/ops.cu");
const OPS_PTX: &[u8] = include_bytes!("kernels/ops.ptx");
const N: usize = 4096;
const BYTES: usize = N * 4;

fn ops(gpu: &Arc<CudaGpu>) -> Option<CudaModule> {
    if gpu.driver_version() >= 13010 {
        CudaModule::from_image(gpu, OPS_PTX).ok()
    } else {
        CudaModule::from_source(gpu, OPS, &CompileOptions::default()).ok()
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

/// A device, two streams, the test kernels and buffers: x (floats), y, z and a pinned output.
struct Fixture {
    gpu: Arc<CudaGpu>,
    pool: Arc<LanePool>,
    s: Vec<Arc<CudaStream>>,
    axpb: CudaKernel,
    add: CudaKernel,
    spin: CudaKernel,
    mem: DeviceMemory,
    pinned: PinnedHost,
    x: Block,
    y: Block,
    z: Block,
    /// Pinned host buffers: `out[0]` and `out[1]`.
    out: [Block; 2],
}

impl Fixture {
    fn new() -> Option<Self> {
        let gpu = match CudaGpu::open(0) {
            Ok(gpu) => gpu,
            Err(e) => {
                eprintln!("no CUDA device: skipped ({e})");
                return None;
            }
        };
        let module = ops(&gpu)?;
        let (pool, s) = gpu
            .open_lanes(&["a", "b"], StreamOptions::default())
            .unwrap();
        let mem = DeviceMemory {
            device: gpu.device().clone(),
            location: LocationId(0),
        };
        let pinned = PinnedHost::new(gpu.device().clone(), LocationId(1), 0);
        let (x, y, z) = (
            mem.alloc(BYTES as u64).unwrap(),
            mem.alloc(BYTES as u64).unwrap(),
            mem.alloc(BYTES as u64).unwrap(),
        );
        let out = [
            pinned.alloc(BYTES as u64).unwrap(),
            pinned.alloc(BYTES as u64).unwrap(),
        ];
        let fixture = Self {
            axpb: module.kernel("axpb").unwrap(),
            add: module.kernel("add_u32").unwrap(),
            spin: module.kernel("spin").unwrap(),
            gpu,
            pool,
            s,
            mem,
            pinned,
            x,
            y,
            z,
            out,
        };
        // x[i] = i / 4 (exact in f32), uploaded through the output buffer.
        let init = fixture.out_words_mut(0);
        for (i, v) in init.iter_mut().enumerate() {
            *v = (i as f32 * 0.25).to_bits();
        }
        fixture.s[0]
            .submit(|q| {
                // SAFETY: valid blocks; the marker below is waited for before the buffers are used again.
                unsafe { q.copy_to_device(fixture.x.addr, fixture.out[0].addr as *const u8, BYTES) }
            })
            .unwrap();
        fixture.settle(0);
        Some(fixture)
    }

    #[allow(
        clippy::mut_from_ref,
        reason = "pinned host memory that the device writes; tests touch it between settled runs"
    )]
    fn out_words_mut(&self, which: usize) -> &mut [u32] {
        // SAFETY: a pinned block of BYTES bytes that this fixture owns; tests touch it only between settled runs.
        unsafe { std::slice::from_raw_parts_mut(self.out[which].addr as *mut u32, N) }
    }

    fn out_words(&self, which: usize) -> Vec<u32> {
        self.out_words_mut(which).to_vec()
    }

    /// Waits (from the test thread) for everything submitted so far on stream `which`, and the streams it joined.
    fn settle(&self, which: usize) {
        let m: Marker = self.s[which].mark().unwrap();
        assert!(
            wait_until(Duration::from_secs(10), || self.pool.reached(m)),
            "the work ends"
        );
    }

    /// The CPU's answer for `axpb(a, b)` followed by `add_u32(+3)` over the bits.
    fn reference(&self, a: f32, b: f32) -> Vec<u32> {
        (0..N)
            .map(|i| a.mul_add(i as f32 * 0.25, b).to_bits().wrapping_add(3))
            .collect()
    }

    /// The sequence: stream 0 computes y = a x + b, stream 1 adds 3 to y's bits into z behind it, and stream 0
    /// copies z to `out[which]` behind that.
    fn sequence<S: Sequence<Kernel = CudaKernel>>(
        &self,
        q: &mut S,
        a: f32,
        b: f32,
        which: usize,
    ) -> Result<(), GpuError> {
        let first = Args::new()
            .device(self.y.addr)
            .device(self.x.addr)
            .arg(a)
            .arg(b)
            .arg(N as u32);
        let second = Args::new()
            .device(self.z.addr)
            .device(self.y.addr)
            .arg(3u32)
            .arg(N as u32);
        let geometry = Launch::linear(N as u32, 256);
        q.submit(0, |e| {
            // SAFETY: matching arguments over live blocks, which the fixture keeps until the work retires.
            unsafe { e.launch(&self.axpb, geometry, &first) }
        })?;
        q.order(0, 1)?;
        q.submit(1, |e| {
            // SAFETY: as above.
            unsafe { e.launch(&self.add, geometry, &second) }
        })?;
        q.order(1, 0)?;
        q.submit(0, |e| {
            // SAFETY: as above; the pinned output outlives the work.
            unsafe { e.copy_to_host(self.out[which].addr as *mut u8, self.z.addr, BYTES) }
        })
    }

    /// The same work submitted directly, with markers between the streams.
    fn direct(&self, a: f32, b: f32, which: usize) {
        let first = Args::new()
            .device(self.y.addr)
            .device(self.x.addr)
            .arg(a)
            .arg(b)
            .arg(N as u32);
        let second = Args::new()
            .device(self.z.addr)
            .device(self.y.addr)
            .arg(3u32)
            .arg(N as u32);
        let geometry = Launch::linear(N as u32, 256);
        self.s[0]
            .submit(|e| {
                // SAFETY: matching arguments over live blocks.
                unsafe { e.launch(&self.axpb, geometry, &first) }
            })
            .unwrap();
        let m = self.s[0].mark().unwrap();
        self.s[1].await_marker(m).unwrap();
        self.s[1]
            .submit(|e| {
                // SAFETY: as above.
                unsafe { e.launch(&self.add, geometry, &second) }
            })
            .unwrap();
        let m = self.s[1].mark().unwrap();
        self.s[0].await_marker(m).unwrap();
        self.s[0]
            .submit(|e| {
                // SAFETY: as above.
                unsafe { e.copy_to_host(self.out[which].addr as *mut u8, self.z.addr, BYTES) }
            })
            .unwrap();
    }

    fn streams(&self) -> [&Arc<CudaStream>; 2] {
        [&self.s[0], &self.s[1]]
    }

    fn clear(&self) {
        self.settle(0);
        self.settle(1);
        self.out_words_mut(0).fill(0);
        self.out_words_mut(1).fill(0);
        self.s[0]
            .submit(|q| {
                // SAFETY: live blocks; settled before the next use.
                unsafe {
                    q.memset(self.y.addr, 0, BYTES)?;
                    q.memset(self.z.addr, 0, BYTES)
                }
            })
            .unwrap();
        self.settle(0);
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        // The streams end their work as they drop (a shutdown call).
        self.s.clear();
        // SAFETY: nothing runs on the device any more; the blocks are unused.
        unsafe {
            for b in [self.x, self.y, self.z] {
                self.mem.free(b);
            }
            for b in self.out {
                self.pinned.free(b);
            }
        }
    }
}

#[test]
fn record_runs_the_sequence_once_and_replay_repeats_it() {
    let Some(f) = Fixture::new() else { return };
    f.clear();
    let graph = f
        .gpu
        .record(&f.streams(), |q| f.sequence(q, 1.5, -2.0, 0))
        .unwrap();
    assert_eq!(graph.launches(), 2);
    f.settle(0);
    let reference = f.reference(1.5, -2.0);
    assert!(
        f.out_words(0) == reference,
        "the recording ran the sequence for real"
    );
    f.clear();
    graph.replay(&f.s[0]).unwrap();
    f.settle(0);
    assert!(f.out_words(0) == reference, "the replay repeats it");
    drop(graph);
}

#[test]
fn replay_matches_direct_submission_bit_for_bit() {
    let Some(f) = Fixture::new() else { return };
    f.clear();
    let graph = f
        .gpu
        .record(&f.streams(), |q| f.sequence(q, 0.5, 1.0, 0))
        .unwrap();
    f.settle(0);
    f.clear();
    graph.replay(&f.s[0]).unwrap();
    f.settle(0);
    let replayed = f.out_words(0);
    f.clear();
    f.direct(0.5, 1.0, 1);
    f.settle(0);
    assert!(
        replayed == f.out_words(1),
        "replay equals direct submission"
    );
    assert!(replayed == f.reference(0.5, 1.0));
}

#[test]
fn a_replay_runs_on_any_stream() {
    let Some(f) = Fixture::new() else { return };
    f.clear();
    let graph = f
        .gpu
        .record(&f.streams(), |q| f.sequence(q, 2.0, 0.0, 0))
        .unwrap();
    f.settle(0);
    f.clear();
    graph.replay(&f.s[1]).unwrap();
    f.settle(1);
    assert!(f.out_words(0) == f.reference(2.0, 0.0));
}

#[test]
fn submissions_from_other_threads_wait_out_a_recording() {
    use std::sync::atomic::{AtomicBool, Ordering};
    let Some(f) = Fixture::new() else { return };
    f.clear();
    let inside = AtomicBool::new(false);
    let seen_inside = AtomicBool::new(true);
    let graph = std::thread::scope(|scope| {
        let peer = scope.spawn(|| {
            assert!(wait_until(Duration::from_secs(10), || inside.load(Ordering::Acquire)));
            f.s[0]
                .submit(|q| {
                    seen_inside.store(inside.load(Ordering::Acquire), Ordering::Release);
                    // SAFETY: `spin` takes one 64-bit argument and touches no memory.
                    unsafe { q.launch(&f.spin, Launch::linear(1, 1), &Args::new().arg(1000u64)) }
                })
                .unwrap();
        });
        let graph = f
            .gpu
            .record(&f.streams(), |q| {
                inside.store(true, Ordering::Release);
                f.sequence(q, 1.0, 1.0, 0)?;
                std::thread::sleep(Duration::from_millis(20));
                inside.store(false, Ordering::Release);
                Ok(())
            })
            .unwrap();
        peer.join().unwrap();
        graph
    });
    assert!(
        !seen_inside.load(Ordering::Acquire),
        "the peer's submission ran after the recording ended"
    );
    assert_eq!(
        graph.launches(),
        2,
        "the peer's launch is not in the recording"
    );
    f.settle(0);
}
