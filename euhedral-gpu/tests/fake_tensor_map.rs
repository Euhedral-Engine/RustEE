//! A kernel that takes a tensor map by value, recorded, replayed and updated on the fake backend. No GPU.
#![cfg(feature = "fake")]

use euhedral_gpu::fake::{FakeGpu, FakeKernel, FakeModule};
use euhedral_gpu::{
    Args, CompileOptions, Gpu, GpuError, Graph, Interleave, L2Promotion, LaneEnqueue, Launch,
    MapDataType, Module, OobFill, Sequence, StreamOptions, Swizzle, TensorMap, TiledMap,
};

fn description(address: u64) -> TiledMap {
    TiledMap {
        address,
        data_type: MapDataType::F32,
        rank: 2,
        dims: [64, 32, 1, 1, 1],
        strides: [256, 0, 0, 0],
        box_dims: [16, 8, 1, 1, 1],
        element_strides: [1; 5],
        interleave: Interleave::None,
        swizzle: Swizzle::B64,
        l2_promotion: L2Promotion::B128,
        oob_fill: OobFill::Zero,
    }
}

fn launch_with<S: Sequence<Kernel = FakeKernel>>(
    q: &mut S,
    kernel: &FakeKernel,
    dst: u64,
    map: TensorMap,
) -> Result<(), GpuError> {
    q.submit(0, |e| {
        // SAFETY: a fake allocation; the kernel writes inside it.
        unsafe {
            e.launch(
                kernel,
                Launch::linear(1, 1),
                &Args::new().device(dst).arg(map),
            )
        }
    })
}

#[test]
fn a_tensor_map_argument_is_recorded_replayed_and_updated_like_any_value() {
    let gpu = FakeGpu::new();
    // stamp(dst, map): dst[0..16] = the map's words, truncated to 32 bits.
    gpu.define("stamp", |mem, _l, args| {
        let dst = args.read::<u64>(0).unwrap();
        let map = args.read::<TensorMap>(1).unwrap();
        let out = mem.bytes_mut(dst, 64)?;
        for (i, w) in map.words().iter().enumerate() {
            out[i * 4..i * 4 + 4].copy_from_slice(&(*w as u32).to_le_bytes());
        }
        Ok(())
    });
    let module = FakeModule::from_source(&gpu, "// kernels", &CompileOptions::default()).unwrap();
    let kernel = module.kernel("stamp").unwrap();
    let (_pool, mut s) = gpu.open_lanes(&["a"], StreamOptions::default()).unwrap();
    let dst = gpu.alloc(64);
    let expect = |map: &TensorMap| -> Vec<u8> {
        map.words()
            .iter()
            .flat_map(|w| (*w as u32).to_le_bytes())
            .collect()
    };
    let (first, second) = (
        TensorMap::fake(&description(0x1000)).unwrap(),
        TensorMap::fake(&description(0x2000)).unwrap(),
    );
    assert_ne!(first, second);

    let mut graph = gpu
        .record(&mut [&mut s[0]], |q| launch_with(q, &kernel, dst, first))
        .unwrap();
    gpu.run_until_idle();
    assert_eq!(gpu.read(dst, 64), expect(&first), "the recording's own run");

    // The fingerprint treats the map as bytes of a fixed size: another value is the same shape, so it updates in place.
    assert_eq!(
        gpu.fingerprint(1, |q| launch_with(q, &kernel, dst, first))
            .unwrap(),
        gpu.fingerprint(1, |q| launch_with(q, &kernel, dst, second))
            .unwrap()
    );
    gpu.update(&mut graph, |q| launch_with(q, &kernel, dst, second))
        .unwrap();
    graph.replay(&mut s[0]).unwrap();
    gpu.run_until_idle();
    assert_eq!(
        gpu.read(dst, 64),
        expect(&second),
        "the replay after an update"
    );
}
