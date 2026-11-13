//! A tile copied through the Tensor Memory Accelerator from a tensor map passed by value. The test skips, passing, when no
//! driver, device or run-time compiler is present, the device is older than compute capability 9.0, or the driver has no
//! `cuTensorMapEncodeTiled`.
#![cfg(feature = "cuda")]
#![allow(
    clippy::disallowed_methods,
    reason = "the test thread is not a lattice worker: it polls and sleeps"
)]

use std::time::{Duration, Instant};

use euhedral_gpu::cuda::{CudaGpu, CudaKernel, CudaModule, CudaStream};
use euhedral_gpu::{
    Args, CompileOptions, Gpu, GpuError, Graph, Interleave, L2Promotion, LaneEnqueue, Launch,
    MapDataType, Module, OobFill, Sequence, Stream, StreamOptions, Swizzle, TensorMap, TiledMap,
};
use euhedral_memory::cuda::{DeviceMemory, PinnedHost};
use euhedral_memory::{Backend, LocationId};

/// The source is `ROWS` rows of `COLS` 32-bit words; one copy moves a `BOX_COLS` by `BOX_ROWS` tile.
const COLS: usize = 64;
const ROWS: usize = 32;
const BOX_COLS: usize = 16;
const BOX_ROWS: usize = 8;
const TILE: usize = BOX_COLS * BOX_ROWS;
/// The tile's origin in the source: column, row.
const AT: (usize, usize) = (32, 16);

/// One block of `TILE` threads: thread 0 asks the TMA for the tile at (c0, c1) and waits on an mbarrier for its bytes; every
/// thread then writes one word of the shared tile out. The map is a by-value kernel parameter, so it travels in the launch
/// arguments' storage and, in a graph, in the kernel node's.
const SOURCE: &str = r#"
struct alignas(64) TMap { unsigned long long opaque[16]; };

__device__ __forceinline__ unsigned smem_address(const void* p) {
    return static_cast<unsigned>(__cvta_generic_to_shared(p));
}

extern "C" __global__ void tile_copy(unsigned* out, int c0, int c1, const __grid_constant__ TMap map) {
    __shared__ __align__(128) unsigned tile[128];
    __shared__ unsigned long long bar;
    if (threadIdx.x == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_address(&bar)), "r"(1));
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                     ::"r"(smem_address(&bar)), "r"(128u * 4u) : "memory");
        asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
                     ::"r"(smem_address(tile)), "l"(&map), "r"(c0), "r"(c1), "r"(smem_address(&bar)) : "memory");
    }
    asm volatile(
        "{\n.reg .pred p;\nWAIT_%=:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
        "@!p bra WAIT_%=;\n}" ::"r"(smem_address(&bar)), "r"(0u) : "memory");
    out[threadIdx.x] = tile[threadIdx.x];
}
"#;

fn source_word(row: usize, col: usize) -> u32 {
    ((row * COLS + col) as u32).wrapping_mul(2_654_435_761) ^ 0x5bd1_e995
}

/// The tile a map over the source shifted down by `shift_rows` rows reads at `AT`.
fn expected(shift_rows: usize) -> Vec<u32> {
    (0..BOX_ROWS)
        .flat_map(|r| (0..BOX_COLS).map(move |c| source_word(AT.1 + shift_rows + r, AT.0 + c)))
        .collect()
}

fn description(address: u64) -> TiledMap {
    TiledMap {
        address,
        data_type: MapDataType::U32,
        rank: 2,
        dims: [COLS as u64, ROWS as u64, 1, 1, 1],
        strides: [(COLS * 4) as u64, 0, 0, 0],
        box_dims: [BOX_COLS as u32, BOX_ROWS as u32, 1, 1, 1],
        element_strides: [1; 5],
        interleave: Interleave::None,
        swizzle: Swizzle::None,
        l2_promotion: L2Promotion::None,
        oob_fill: OobFill::Zero,
    }
}

struct Rig {
    kernel: CudaKernel,
    out_dev: u64,
    out: *mut u32,
}

impl Rig {
    /// The launch: the map follows 16 bytes of other arguments, so it is the argument that needs the 64-byte alignment.
    fn work<S: Sequence<Kernel = CudaKernel>>(
        &self,
        q: &mut S,
        map: TensorMap,
    ) -> Result<(), GpuError> {
        let args = Args::new()
            .device(self.out_dev)
            .arg(AT.0 as i32)
            .arg(AT.1 as i32)
            .arg(map);
        q.submit(0, |e| {
            // SAFETY: matching arguments over live blocks that the test keeps until the work ends.
            unsafe {
                e.launch(
                    &self.kernel,
                    Launch::linear(TILE as u32, TILE as u32),
                    &args,
                )?;
                e.copy_to_host(self.out.cast(), self.out_dev, TILE * 4)
            }
        })
    }

    fn result(&self) -> Vec<u32> {
        // SAFETY: a pinned block of TILE words that the test reads only between settled runs.
        unsafe { std::slice::from_raw_parts(self.out, TILE) }.to_vec()
    }

    fn clear(&self) {
        // SAFETY: as in `result`.
        unsafe { std::slice::from_raw_parts_mut(self.out, TILE) }.fill(0xDEAD_BEEF);
    }
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
fn a_tile_copied_through_a_tensor_map_equals_its_source() {
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
    let src = mem.alloc((ROWS * COLS * 4) as u64).unwrap();
    let out_dev = mem.alloc((TILE * 4) as u64).unwrap();
    let stage = pinned.alloc((ROWS * COLS * 4) as u64).unwrap();
    let out = pinned.alloc((TILE * 4) as u64).unwrap();
    // SAFETY: a pinned block of ROWS * COLS words that nothing else touches yet.
    unsafe {
        let words = std::slice::from_raw_parts_mut(stage.addr as *mut u32, ROWS * COLS);
        for (i, w) in words.iter_mut().enumerate() {
            *w = source_word(i / COLS, i % COLS);
        }
    }
    s[0].submit(|e| {
        // SAFETY: live blocks of the same size.
        unsafe { e.copy_to_device(src.addr, stage.addr as *const u8, ROWS * COLS * 4) }
    })
    .unwrap();
    settle(&mut s[0], &pool);

    let map = |address: u64| match TensorMap::tiled(&gpu, &description(address)) {
        Ok(map) => Some(map),
        Err(GpuError::Invalid(text)) if text.contains("libcuda has no cuTensorMapEncodeTiled") => {
            None
        }
        Err(e) => panic!("the driver refused a valid description: {e}"),
    };
    let Some(first) = map(src.addr) else {
        eprintln!("the driver has no cuTensorMapEncodeTiled: skipped");
        return;
    };
    // The same description encodes to the same bytes; another address does not.
    let shifted = map(src.addr + (COLS * 4) as u64).unwrap();
    assert_eq!(first, map(src.addr).unwrap());
    assert_ne!(first, shifted);
    // An invalid description never reaches the driver.
    assert!(matches!(
        TensorMap::tiled(
            &gpu,
            &TiledMap {
                rank: 0,
                ..description(src.addr)
            }
        ),
        Err(GpuError::Invalid(_))
    ));

    let rig = Rig {
        kernel: module.kernel("tile_copy").unwrap(),
        out_dev: out_dev.addr,
        out: out.addr as *mut u32,
    };

    // Eagerly: the tile equals the source's, and a map one row down reads the tile one row down, so a wrong or shifted
    // descriptor arriving at the kernel would fail here.
    rig.clear();
    s[0].submit(|e| {
        // SAFETY: as in `work`.
        unsafe {
            e.launch(
                &rig.kernel,
                Launch::linear(TILE as u32, TILE as u32),
                &Args::new()
                    .device(rig.out_dev)
                    .arg(AT.0 as i32)
                    .arg(AT.1 as i32)
                    .arg(first),
            )?;
            e.copy_to_host(rig.out.cast(), rig.out_dev, TILE * 4)
        }
    })
    .unwrap();
    settle(&mut s[0], &pool);
    assert_eq!(rig.result(), expected(0), "eagerly");
    rig.clear();
    s[0].submit(|e| {
        // SAFETY: as in `work`.
        unsafe {
            e.launch(
                &rig.kernel,
                Launch::linear(TILE as u32, TILE as u32),
                &Args::new()
                    .device(rig.out_dev)
                    .arg(AT.0 as i32)
                    .arg(AT.1 as i32)
                    .arg(shifted),
            )?;
            e.copy_to_host(rig.out.cast(), rig.out_dev, TILE * 4)
        }
    })
    .unwrap();
    settle(&mut s[0], &pool);
    assert_eq!(rig.result(), expected(1), "eagerly, the map shifted a row");
    assert_ne!(expected(0), expected(1));

    // Recorded (the recording runs the work for real), then replayed, then updated with the other map and replayed.
    rig.clear();
    let mut graph = gpu
        .record(&mut [&mut s[0]], |q| rig.work(q, first))
        .unwrap();
    settle(&mut s[0], &pool);
    assert_eq!(rig.result(), expected(0), "the recording's own run");
    rig.clear();
    graph.replay(&mut s[0]).unwrap();
    settle(&mut s[0], &pool);
    assert_eq!(rig.result(), expected(0), "the replay");
    rig.clear();
    gpu.update(&mut graph, |q| rig.work(q, shifted)).unwrap();
    graph.replay(&mut s[0]).unwrap();
    settle(&mut s[0], &pool);
    assert_eq!(rig.result(), expected(1), "the replay after an update");
}
