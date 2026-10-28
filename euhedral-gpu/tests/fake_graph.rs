//! Record, fingerprint, update, re-record and replay on the fake backend. No GPU.
#![cfg(feature = "fake")]

use std::sync::Arc;

use euhedral_gpu::fake::{FakeGpu, FakeKernel, FakeModule, FakeStream};
use euhedral_gpu::{
    Args, CompileOptions, Enqueue, Gpu, GpuError, Graph, Launch, Module, Rerecorded, Sequence,
    Stream, StreamOptions,
};

const WORDS: usize = 8;
const BYTES: usize = WORDS * 4;

fn words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn gpu() -> Arc<FakeGpu> {
    let gpu = FakeGpu::new();
    // scale(dst, src, a): dst[i] = src[i] * a
    gpu.define("scale", |mem, _l, args| {
        let (dst, src, a) = (
            args.read::<u64>(0).unwrap(),
            args.read::<u64>(1).unwrap(),
            args.read::<u32>(2).unwrap(),
        );
        let input = words(mem.bytes(src, BYTES)?);
        let out = mem.bytes_mut(dst, BYTES)?;
        for (i, v) in input.iter().enumerate() {
            out[i * 4..i * 4 + 4].copy_from_slice(&(v * a).to_le_bytes());
        }
        Ok(())
    });
    // add1(dst): dst[i] += 1
    gpu.define("add1", |mem, _l, args| {
        let dst = args.read::<u64>(0).unwrap();
        let out = mem.bytes_mut(dst, BYTES)?;
        for chunk in out.chunks_mut(4) {
            let v = u32::from_le_bytes((&*chunk).try_into().unwrap()) + 1;
            chunk.copy_from_slice(&v.to_le_bytes());
        }
        Ok(())
    });
    gpu
}

struct Fixture {
    gpu: Arc<FakeGpu>,
    pool: Arc<euhedral_graph::LanePool>,
    s: Vec<Arc<FakeStream>>,
    scale: FakeKernel,
    add1: FakeKernel,
    x: u64,
    y: u64,
    /// Host outputs, kept alive for the whole test.
    out: Vec<[u8; BYTES]>,
}

impl Fixture {
    fn new() -> Self {
        let gpu = gpu();
        let module =
            FakeModule::from_source(&gpu, "// kernels", &CompileOptions::default()).unwrap();
        let (pool, s) = gpu
            .open_lanes(&["a", "b"], StreamOptions::default())
            .unwrap();
        let x = gpu.alloc(BYTES);
        let y = gpu.alloc(BYTES);
        let input: Vec<u8> = (0..WORDS as u32)
            .flat_map(|i| (i + 1).to_le_bytes())
            .collect();
        gpu.write(x, &input);
        Self {
            scale: module.kernel("scale").unwrap(),
            add1: module.kernel("add1").unwrap(),
            gpu,
            pool,
            s,
            x,
            y,
            out: vec![[0; BYTES]; 2],
        }
    }

    fn out_ptr(&mut self, which: usize) -> *mut u8 {
        self.out[which].as_mut_ptr()
    }

    fn out_words(&self, which: usize) -> Vec<u32> {
        words(&self.out[which])
    }

    fn reference(&self, a: u32) -> Vec<u32> {
        (0..WORDS as u32).map(|i| (i + 1) * a + 1).collect()
    }

    /// The sequence: stream 0 scales, stream 1 adds one behind it, stream 0 copies out behind that.
    fn sequence<S: Sequence<Kernel = FakeKernel>>(
        &self,
        q: &mut S,
        a: u32,
        out: *mut u8,
    ) -> Result<(), GpuError> {
        let (x, y) = (self.x, self.y);
        let scale_args = Args::new().device(y).device(x).arg(a);
        q.submit(0, |e| {
            // SAFETY: fake allocations; the kernels read and write inside them.
            unsafe { e.launch(&self.scale, Launch::linear(WORDS as u32, 8), &scale_args) }
        })?;
        q.order(0, 1)?;
        q.submit(1, |e| {
            // SAFETY: as above.
            unsafe {
                e.launch(
                    &self.add1,
                    Launch::linear(WORDS as u32, 8),
                    &Args::new().device(y),
                )
            }
        })?;
        q.order(1, 0)?;
        q.submit(0, |e| {
            // SAFETY: `out` is a live host buffer of BYTES bytes.
            unsafe { e.copy_to_host(out, y, BYTES) }
        })
    }

    /// The same work submitted directly on the streams.
    fn direct(&self, a: u32, out: *mut u8) {
        let (x, y) = (self.x, self.y);
        let scale_args = Args::new().device(y).device(x).arg(a);
        self.s[0]
            .submit(|e| {
                // SAFETY: fake allocations.
                unsafe { e.launch(&self.scale, Launch::linear(WORDS as u32, 8), &scale_args) }
            })
            .unwrap();
        let m = self.s[0].mark().unwrap();
        self.s[1].await_marker(m).unwrap();
        self.s[1]
            .submit(|e| {
                // SAFETY: fake allocations.
                unsafe {
                    e.launch(
                        &self.add1,
                        Launch::linear(WORDS as u32, 8),
                        &Args::new().device(y),
                    )
                }
            })
            .unwrap();
        let m = self.s[1].mark().unwrap();
        self.s[0].await_marker(m).unwrap();
        self.s[0]
            .submit(|e| {
                // SAFETY: `out` is a live host buffer.
                unsafe { e.copy_to_host(out, y, BYTES) }
            })
            .unwrap();
    }

    fn streams(&self) -> [&Arc<FakeStream>; 2] {
        [&self.s[0], &self.s[1]]
    }
}

#[test]
fn record_runs_the_sequence_once_for_real() {
    let mut f = Fixture::new();
    let out = f.out_ptr(0);
    let _graph = f
        .gpu
        .record(&f.streams(), |q| f.sequence(q, 2, out))
        .unwrap();
    f.gpu.run_until_idle();
    assert_eq!(f.out_words(0), f.reference(2));
}

#[test]
fn replay_with_updated_arguments_matches_direct_submission() {
    let mut f = Fixture::new();
    let out = f.out_ptr(0);
    let mut graph = f
        .gpu
        .record(&f.streams(), |q| f.sequence(q, 2, out))
        .unwrap();
    f.gpu.run_until_idle();
    assert_eq!(graph.launches(), 2);
    f.gpu.update(&mut graph, |q| f.sequence(q, 3, out)).unwrap();
    graph.replay(&f.s[0]).unwrap();
    f.gpu.run_until_idle();
    let replayed = f.out_words(0);
    let direct_out = f.out_ptr(1);
    f.direct(3, direct_out);
    f.gpu.run_until_idle();
    assert_eq!(replayed, f.out_words(1), "replay equals direct submission");
    assert_eq!(replayed, f.reference(3));
}

#[test]
fn the_fingerprint_matches_a_probe_of_the_same_sequence() {
    let mut f = Fixture::new();
    let out = f.out_ptr(0);
    let graph = f
        .gpu
        .record(&f.streams(), |q| f.sequence(q, 2, out))
        .unwrap();
    let probed = f.gpu.fingerprint(2, |q| f.sequence(q, 5, out)).unwrap();
    assert_eq!(
        probed,
        graph.fingerprint(),
        "values are not part of the shape"
    );
    let longer = f
        .gpu
        .fingerprint(2, |q| {
            f.sequence(q, 5, out)?;
            q.submit(0, |e| {
                // SAFETY: nothing is enqueued on a probe.
                unsafe { e.launch(&f.add1, Launch::linear(1, 1), &Args::new().device(f.y)) }
            })
        })
        .unwrap();
    assert_ne!(longer, graph.fingerprint());
}

#[test]
fn an_update_of_another_shape_applies_nothing() {
    let mut f = Fixture::new();
    let out = f.out_ptr(0);
    let mut graph = f
        .gpu
        .record(&f.streams(), |q| f.sequence(q, 2, out))
        .unwrap();
    f.gpu.run_until_idle();
    let r = f.gpu.update(&mut graph, |q| {
        f.sequence(q, 9, out)?;
        q.submit(0, |e| {
            // SAFETY: nothing is enqueued on an update.
            unsafe { e.launch(&f.add1, Launch::linear(1, 1), &Args::new().device(f.y)) }
        })
    });
    assert!(matches!(r, Err(GpuError::ShapeChanged { .. })), "{r:?}");
    f.out[0].fill(0);
    graph.replay(&f.s[0]).unwrap();
    f.gpu.run_until_idle();
    assert_eq!(
        f.out_words(0),
        f.reference(2),
        "the old arguments still apply"
    );
}

#[test]
fn rerecord_reports_update_or_instantiate() {
    let mut f = Fixture::new();
    let out = f.out_ptr(0);
    let mut graph = f
        .gpu
        .record(&f.streams(), |q| f.sequence(q, 2, out))
        .unwrap();
    f.gpu.run_until_idle();
    // The same kernels, copying to another host buffer.
    let out2 = f.out_ptr(1);
    let how = f
        .gpu
        .rerecord(&mut graph, &f.streams(), |q| f.sequence(q, 4, out2))
        .unwrap();
    assert_eq!(how, Rerecorded::Updated);
    f.gpu.run_until_idle();
    f.out[1].fill(0);
    graph.replay(&f.s[0]).unwrap();
    f.gpu.run_until_idle();
    assert_eq!(f.out_words(1), f.reference(4));
    // Another kernel order: add1 first, then scale, all on stream 0.
    let (x, y) = (f.x, f.y);
    let flipped = |q: &mut <FakeGpu as Gpu>::Sequence<'_>| -> Result<(), GpuError> {
        q.submit(0, |e| {
            // SAFETY: fake allocations and a live host buffer.
            unsafe {
                e.launch(&f.add1, Launch::linear(1, 1), &Args::new().device(y))?;
                e.launch(
                    &f.scale,
                    Launch::linear(WORDS as u32, 8),
                    &Args::new().device(y).device(x).arg(3u32),
                )?;
                e.copy_to_host(out2, y, BYTES)
            }
        })
    };
    let how = f.gpu.rerecord(&mut graph, &f.streams(), flipped).unwrap();
    assert_eq!(how, Rerecorded::Instantiated);
    f.gpu.run_until_idle();
    f.out[1].fill(0);
    graph.replay(&f.s[1]).unwrap();
    f.gpu.run_until_idle();
    let expected: Vec<u32> = (0..WORDS as u32).map(|i| (i + 1) * 3).collect();
    assert_eq!(f.out_words(1), expected);
}

#[test]
fn a_sequence_names_only_its_own_streams() {
    let f = Fixture::new();
    f.gpu
        .fingerprint(2, |q| {
            assert!(matches!(q.submit(2, |_| Ok(())), Err(GpuError::Invalid(_))));
            assert!(matches!(q.order(0, 2), Err(GpuError::Invalid(_))));
            assert!(q.order(1, 1).is_ok());
            Ok(())
        })
        .unwrap();
    let _ = &f.pool;
}
