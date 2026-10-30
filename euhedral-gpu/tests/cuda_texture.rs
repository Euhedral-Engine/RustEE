//! Texture objects on a real device: made over pitched memory, refused when misaligned, and read in a kernel. Every test skips,
//! passing, when no driver, device or compiler is present.
#![cfg(feature = "cuda")]
#![allow(
    clippy::disallowed_methods,
    reason = "the test thread is not a lattice worker: it waits for the device"
)]

use std::sync::Arc;
use std::time::{Duration, Instant};

use euhedral_gpu::cuda::texture::{Address, Filter, Format, TexObject, View, alignments};
use euhedral_gpu::cuda::{CudaGpu, CudaModule};
use euhedral_gpu::{Args, CompileOptions, Enqueue, Gpu, Launch, Module, Stream, StreamOptions};
use euhedral_memory::cuda::DeviceMemory;
use euhedral_memory::{Backend, LocationId};

const SOURCE: &str = r#"
extern "C" __global__ void sample_f(unsigned long long tex, const float2* uv, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = tex2D<float>(tex, uv[i].x, uv[i].y);
}
extern "C" __global__ void sample_u(unsigned long long tex, const float2* uv, unsigned* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = tex2D<unsigned>(tex, uv[i].x, uv[i].y);
}
"#;

struct Rig {
    gpu: Arc<CudaGpu>,
    mem: DeviceMemory,
    module: CudaModule,
}

fn rig() -> Option<Rig> {
    let gpu = CudaGpu::open(0).ok()?;
    let module = match CudaModule::from_source(&gpu, SOURCE, &CompileOptions::default()) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("no kernel compiler: skipped ({e})");
            return None;
        }
    };
    let mem = DeviceMemory {
        device: gpu.device().clone(),
        location: LocationId(0),
    };
    Some(Rig { gpu, mem, module })
}

/// A 16-bit image of `w` x `h` with a deterministic pseudo-random content, uploaded with the pitch the device wants.
fn image(rig: &Rig, w: usize, h: usize) -> (euhedral_memory::Block, usize, Vec<u16>) {
    let (_, pitch_align) = alignments(rig.gpu.device()).expect("alignments");
    let pitch = (w * 2).next_multiple_of(pitch_align as usize);
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let texels: Vec<u16> = (0..w * h)
        .map(|_| {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            (state.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 40) as u16
        })
        .collect();
    let mut bytes = vec![0u8; pitch * h];
    for y in 0..h {
        for x in 0..w {
            bytes[y * pitch + 2 * x..y * pitch + 2 * x + 2]
                .copy_from_slice(&texels[y * w + x].to_le_bytes());
        }
    }
    let block = rig.mem.alloc((pitch * h) as u64).expect("device memory");
    let device = rig.gpu.device();
    let _c = device.enter().expect("context");
    // SAFETY: the host range is valid for the call and the device range is the block; synchronous below.
    let code = unsafe {
        (device.driver.fns.memcpy_htod_async)(
            block.addr,
            bytes.as_ptr().cast(),
            bytes.len(),
            std::ptr::null_mut(),
        )
    };
    device.check("cuMemcpyHtoDAsync", code).expect("upload");
    (block, pitch, texels)
}

fn view(base: u64, w: usize, h: usize, pitch: usize) -> View {
    View {
        base,
        format: Format::U16,
        channels: 1,
        width: w,
        height: h,
        pitch,
        filter: Filter::Linear,
        address: Address::Clamp,
        normalized: true,
        read_as_integer: false,
    }
}

/// Runs `kernel` over `uv` with `tex` and reads `n` 32-bit results back.
fn sample(rig: &Rig, kernel: &str, tex: &TexObject, uv: &[(f32, f32)]) -> Vec<u32> {
    let n = uv.len();
    let (_pool, mut streams) = rig
        .gpu
        .open_lanes(&["tex"], StreamOptions::default())
        .expect("a lane");
    let mut stream = streams.remove(0);
    let kernel = rig.module.kernel(kernel).expect("kernel");
    let uv_bytes: Vec<u8> = uv
        .iter()
        .flat_map(|(u, v)| u.to_le_bytes().into_iter().chain(v.to_le_bytes()))
        .collect();
    let (uv_buf, out_buf) = (
        rig.mem.alloc(uv_bytes.len() as u64).unwrap(),
        rig.mem.alloc(n as u64 * 4).unwrap(),
    );
    let device = rig.gpu.device();
    let mut out = vec![0u8; n * 4];
    {
        let _c = device.enter().expect("context");
        // SAFETY: valid ranges; synchronous below.
        let code = unsafe {
            (device.driver.fns.memcpy_htod_async)(
                uv_buf.addr,
                uv_bytes.as_ptr().cast(),
                uv_bytes.len(),
                std::ptr::null_mut(),
            )
        };
        device.check("cuMemcpyHtoDAsync", code).expect("upload");
    }
    let args = Args::new()
        .arg(tex.handle())
        .device(uv_buf.addr)
        .device(out_buf.addr)
        .arg(n as i32);
    // SAFETY: matching arguments over live buffers; the texture object outlives the launch (waited for below).
    stream
        .submit(|q| unsafe { q.launch(&kernel, Launch::linear(n as u32, 128), &args) })
        .expect("launch");
    let marker = stream.mark().unwrap();
    let start = Instant::now();
    while !_pool.reached(marker) {
        assert!(start.elapsed() < Duration::from_secs(20), "the kernel ends");
        std::thread::sleep(Duration::from_millis(1));
    }
    let _c = device.enter().expect("context");
    // SAFETY: valid ranges; the kernel ended.
    let code = unsafe {
        (device.driver.fns.memcpy_dtoh_async)(
            out.as_mut_ptr().cast(),
            out_buf.addr,
            out.len(),
            std::ptr::null_mut(),
        )
    };
    device.check("cuMemcpyDtoHAsync", code).expect("download");
    // SAFETY: the context is current.
    device
        .check("cuCtxSynchronize", unsafe {
            (device.driver.fns.ctx_synchronize)()
        })
        .expect("synchronise");
    out.as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect()
}

#[test]
fn a_view_that_misses_the_devices_alignment_is_refused_before_the_driver_is_asked() {
    let Some(rig) = rig() else { return };
    let (base_align, pitch_align) = alignments(rig.gpu.device()).expect("alignments");
    assert!(base_align.is_power_of_two() && pitch_align.is_power_of_two());
    let (block, pitch, _) = image(&rig, 100, 32);
    let device = rig.gpu.device();
    let ok = view(block.addr, 100, 32, pitch);
    assert!(TexObject::create(device, &ok).is_ok());
    assert!(
        TexObject::create(
            device,
            &View {
                base: block.addr + 2,
                ..ok
            }
        )
        .unwrap_err()
        .contains("base")
    );
    assert!(
        TexObject::create(
            device,
            &View {
                pitch: pitch + 2,
                ..ok
            }
        )
        .unwrap_err()
        .contains("pitch")
    );
    assert!(
        TexObject::create(
            device,
            &View {
                pitch: pitch_align as usize * 2,
                ..ok
            }
        )
        .unwrap_err()
        .contains("less than a row")
    );
}

#[test]
fn point_sampling_in_texels_reads_the_texels_exactly_and_clamps_outside() {
    let Some(rig) = rig() else { return };
    let (w, h) = (96usize, 40usize);
    let (block, pitch, texels) = image(&rig, w, h);
    let tex = TexObject::create(
        rig.gpu.device(),
        &View {
            filter: Filter::Point,
            normalized: false,
            read_as_integer: true,
            ..view(block.addr, w, h, pitch)
        },
    )
    .expect("texture");
    // Texel centres at (x + 0.5, y + 0.5), and points outside, which clamp to the edge.
    let mut uv = Vec::new();
    let mut want = Vec::new();
    for y in (0..h).step_by(3) {
        for x in (0..w).step_by(5) {
            uv.push((x as f32 + 0.5, y as f32 + 0.5));
            want.push(u32::from(texels[y * w + x]));
        }
    }
    uv.push((-10.0, 3.5));
    want.push(u32::from(texels[3 * w]));
    uv.push((w as f32 + 20.0, h as f32 + 20.0));
    want.push(u32::from(texels[w * h - 1]));
    assert_eq!(sample(&rig, "sample_u", &tex, &uv), want);
}

#[test]
fn linear_normalised_reads_match_the_exact_bilinear_value_within_the_weights_fixed_point_step() {
    let Some(rig) = rig() else { return };
    let (w, h) = (256usize, 128usize);
    let (block, pitch, texels) = image(&rig, w, h);
    let tex = TexObject::create(rig.gpu.device(), &view(block.addr, w, h, pitch)).expect("texture");
    let mut state = 11u64;
    let mut next = || {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        state.wrapping_mul(0x2545_f491_4f6c_dd1d)
    };
    let uv: Vec<(f32, f32)> = (0..4096)
        .map(|_| {
            (
                (next() % 12000) as f32 / 10000.0 - 0.1,
                (next() % 12000) as f32 / 10000.0 - 0.1,
            )
        })
        .collect();
    let got = sample(&rig, "sample_f", &tex, &uv);
    let at = |x: isize, y: isize| {
        f32::from(
            texels[(y.clamp(0, h as isize - 1) as usize) * w + x.clamp(0, w as isize - 1) as usize],
        ) / 65535.0
    };
    let mut worst = 0.0f32;
    for (i, (u, v)) in uv.iter().enumerate() {
        let (x, y) = (u * w as f32 - 0.5, v * h as f32 - 0.5);
        let (x0, y0) = (x.floor(), y.floor());
        let (fx, fy) = (x - x0, y - y0);
        let (xi, yi) = (x0 as isize, y0 as isize);
        let top = at(xi, yi) * (1.0 - fx) + at(xi + 1, yi) * fx;
        let bottom = at(xi, yi + 1) * (1.0 - fx) + at(xi + 1, yi + 1) * fx;
        worst = worst.max((f32::from_bits(got[i]) - (top * (1.0 - fy) + bottom * fy)).abs());
    }
    // The weights are fixed point with 8 fractional bits: at most a step of a full-range texel difference on each axis.
    assert!(worst < 2.0 / 256.0, "worst {worst}");
}
