//! The CUDA backend on a real device. Every test skips, passing, when no driver or device is present.
#![cfg(feature = "cuda")]
#![allow(
    clippy::disallowed_methods,
    reason = "the test thread is not a lattice worker: it polls and sleeps, and test fixtures use files"
)]

use std::sync::Arc;

use euhedral_gpu::cuda::CudaGpu;
use euhedral_memory::cuda::Driver;

fn gpu() -> Option<Arc<CudaGpu>> {
    match CudaGpu::open(0) {
        Ok(gpu) => Some(gpu),
        Err(e) => {
            eprintln!("no CUDA device: skipped ({e})");
            None
        }
    }
}

#[test]
fn the_gpu_shares_memorys_driver_and_reports_its_architecture() {
    let Some(gpu) = gpu() else { return };
    let shared = Driver::shared().unwrap();
    assert!(Arc::ptr_eq(&gpu.device().driver, &shared));
    assert!(gpu.arch().0 >= 7, "{:?}", gpu.arch());
    assert!(gpu.driver_version() >= 12000);
}

mod streams {
    use super::*;
    use std::time::{Duration, Instant};

    use euhedral_gpu::{Enqueue, Gpu, GpuError, Stream, StreamOptions};
    use euhedral_graph::{LaneId, LanePool, Marker};
    use euhedral_memory::cuda::{DeviceMemory, PinnedHost};
    use euhedral_memory::{Backend, LocationId};

    /// Polls `pool` for `marker` from the test thread (which is not the lattice).
    #[allow(
        clippy::disallowed_methods,
        reason = "the test thread is not a lattice worker"
    )]
    pub fn reached_within(pool: &LanePool, marker: Marker, limit: Duration) -> bool {
        let start = Instant::now();
        while !pool.reached(marker) {
            if start.elapsed() > limit {
                return false;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        true
    }

    pub fn device_memory(gpu: &Arc<CudaGpu>) -> DeviceMemory {
        DeviceMemory {
            device: gpu.device().clone(),
            location: LocationId(0),
        }
    }

    #[test]
    fn a_stream_is_a_lane_whose_markers_complete() {
        let Some(gpu) = gpu() else { return };
        let (pool, mut s) = gpu
            .open_lanes(&["compute"], StreamOptions::default())
            .unwrap();
        let mem = device_memory(&gpu);
        let buf = mem.alloc(4 << 20).unwrap();
        s[0].submit(|q| {
            // SAFETY: a live device allocation, used by nothing else.
            unsafe { q.memset(buf.addr, 0xAB, 4 << 20) }
        })
        .unwrap();
        let m = s[0].mark().unwrap();
        assert_eq!(m, Marker::new(s[0].id(), LaneId(0), 1));
        assert!(reached_within(&pool, m, Duration::from_secs(5)));
        assert_eq!(s[0].queue().name(), "compute");
        drop((pool, s));
        // SAFETY: the stream ended its work before it dropped; nothing uses the block.
        unsafe { mem.free(buf) };
    }

    #[test]
    fn copies_round_trip_through_the_device() {
        let Some(gpu) = gpu() else { return };
        let (pool, mut s) = gpu.open_lanes(&["copy"], StreamOptions::default()).unwrap();
        let mem = device_memory(&gpu);
        let pinned = PinnedHost::new(gpu.device().clone(), LocationId(1), 0);
        let len = 1usize << 20;
        let (src, dst) = (
            pinned.alloc(len as u64).unwrap(),
            pinned.alloc(len as u64).unwrap(),
        );
        let dev = mem.alloc(len as u64).unwrap();
        // SAFETY: pinned host blocks of `len` bytes, owned by this test.
        let (src_bytes, dst_bytes) = unsafe {
            (
                std::slice::from_raw_parts_mut(src.addr as *mut u8, len),
                std::slice::from_raw_parts_mut(dst.addr as *mut u8, len),
            )
        };
        for (i, b) in src_bytes.iter_mut().enumerate() {
            *b = (i * 31 % 251) as u8;
        }
        dst_bytes.fill(0);
        s[0].submit(|q| {
            // SAFETY: valid blocks, untouched until the marker below is reached.
            unsafe {
                q.copy_to_device(dev.addr, src.addr as *const u8, len)?;
                q.copy_to_host(dst.addr as *mut u8, dev.addr, len)
            }
        })
        .unwrap();
        let m = s[0].mark().unwrap();
        assert!(reached_within(&pool, m, Duration::from_secs(5)));
        assert_eq!(src_bytes, dst_bytes);
        assert_eq!(s[0].queue().completed(), 1);
        drop((pool, s));
        // SAFETY: the stream is gone and nothing uses the blocks.
        unsafe {
            mem.free(dev);
            pinned.free(src);
            pinned.free(dst);
        }
    }

    #[test]
    fn a_marker_from_an_unknown_lane_is_invalid() {
        let Some(gpu) = gpu() else { return };
        let (_pool, mut s) = gpu.open_lanes(&["a"], StreamOptions::default()).unwrap();
        let own = s[0].id();
        let r = s[0].await_marker(Marker::new(own, LaneId(9), 1));
        assert!(matches!(r, Err(GpuError::Invalid(_))), "{r:?}");
    }

    #[test]
    fn a_marker_of_another_pool_is_refused_not_treated_as_done() {
        let Some(gpu) = gpu() else { return };
        let (pool_a, mut a) = gpu
            .open_lanes(&["a0", "a1"], StreamOptions::default())
            .unwrap();
        let (pool_b, mut b) = gpu
            .open_lanes(&["b0", "b1"], StreamOptions::default())
            .unwrap();
        // A real marker of pool b, made and long done: its lane exists in pool a and its value is below a's own.
        let from_b = b[0].mark().unwrap();
        assert!(reached_within(&pool_b, from_b, Duration::from_secs(5)));
        a[0].mark().unwrap();
        let first = a[0].mark().unwrap();
        assert!(reached_within(&pool_a, first, Duration::from_secs(5)));
        let refused = a[1].await_marker(from_b);
        assert!(matches!(refused, Err(GpuError::Invalid(_))), "{refused:?}");
        // The same lane and value, stamped with pool a's queue, is accepted (and needs no device wait).
        let own = Marker::new(a[0].id(), from_b.lane, from_b.value);
        a[1].await_marker(own).unwrap();
        // Another pool's pool does not reach it either.
        assert!(!pool_a.reached(from_b));
        // A marker with an in-range lane and a done value, but a queue that is neither pool's, is refused too.
        let stranger = Marker::new(euhedral_graph::QueueId::next(), LaneId(0), 1);
        assert!(matches!(
            a[0].await_marker(stranger),
            Err(GpuError::Invalid(_))
        ));
        assert!(!pool_a.reached(stranger));
    }

    /// Polls `condition` from the test thread until it holds or `limit` passes.
    #[allow(
        clippy::disallowed_methods,
        reason = "the test thread is not a lattice worker"
    )]
    pub fn wait_until(limit: Duration, condition: impl Fn() -> bool) -> bool {
        let start = Instant::now();
        while !condition() {
            if start.elapsed() > limit {
                return false;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        true
    }

    struct Record(Arc<std::sync::Mutex<Vec<euhedral_gpu::RetireTicket>>>);

    impl euhedral_gpu::RetireListener for Record {
        // Enqueue only: this runs on the driver's callback thread.
        fn retired(self: Box<Self>, ticket: euhedral_gpu::RetireTicket) {
            self.0.lock().unwrap().push(ticket);
        }
    }

    #[test]
    fn retirement_listeners_run_once_in_order() {
        let Some(gpu) = gpu() else { return };
        let (pool, mut s) = gpu.open_lanes(&["a"], StreamOptions::default()).unwrap();
        let mem = device_memory(&gpu);
        let buf = mem.alloc(1 << 20).unwrap();
        let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
        for i in 0..5u8 {
            s[0].submit(|q| {
                // SAFETY: a live device allocation, used by nothing else.
                unsafe { q.memset(buf.addr, i, 1 << 20) }
            })
            .unwrap();
            s[0].notify_retired(Box::new(Record(seen.clone()))).unwrap();
        }
        assert!(wait_until(Duration::from_secs(5), || seen
            .lock()
            .unwrap()
            .len()
            >= 5));
        let tickets = seen.lock().unwrap().clone();
        assert_eq!(
            tickets.iter().map(|t| t.seq()).collect::<Vec<_>>(),
            [0, 1, 2, 3, 4]
        );
        for t in tickets {
            assert!(t.ok());
            assert_eq!(s[0].confirm_retired(t), None);
        }
        drop((pool, s));
        // SAFETY: the stream ended its work; nothing uses the block.
        unsafe { mem.free(buf) };
    }

    #[test]
    fn retire_into_emits_a_continuation_frame() {
        use euhedral_core::frames::RunnableFrame;
        use euhedral_core::generics::LatticeSource;
        let Some(gpu) = gpu() else { return };
        let (pool, mut s) = gpu.open_lanes(&["a"], StreamOptions::default()).unwrap();
        let mem = device_memory(&gpu);
        let buf = mem.alloc(1 << 20).unwrap();
        let completions = euhedral_graph::Completions::new(pool.clone());
        let ran = Arc::new(std::sync::Mutex::new(Vec::new()));
        s[0].submit(|q| {
            // SAFETY: a live device allocation, used by nothing else.
            unsafe { q.memset(buf.addr, 1, 1 << 20) }
        })
        .unwrap();
        // The submitter travels with its continuation: the frame owns it, confirms the ticket, and drops it.
        let stream = s.remove(0);
        assert!(
            euhedral_gpu::retire_into(stream, &completions, {
                let ran = ran.clone();
                move |mut stream, ticket| {
                    let outcome = stream.confirm_retired(ticket);
                    Box::new(RunnableFrame::new(
                        1,
                        Arc::new(move || ran.lock().unwrap().push(outcome.clone())),
                    ))
                }
            })
            .is_ok()
        );
        // The test thread is the worker: it polls until the continuation is emitted.
        assert!(wait_until(Duration::from_secs(5), || {
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
        assert_eq!(*ran.lock().unwrap(), [None]);
        drop((completions, pool, s));
        // SAFETY: the stream ended its work; nothing uses the block.
        unsafe { mem.free(buf) };
    }

    #[test]
    fn a_ticket_of_another_stream_is_refused_and_frees_nothing() {
        let Some(gpu) = gpu() else { return };
        let (pool, mut s) = gpu
            .open_lanes(&["a", "b"], StreamOptions::default())
            .unwrap();
        let mem = device_memory(&gpu);
        let pinned = PinnedHost::new(gpu.device().clone(), LocationId(1), 0);
        let host = Arc::new(pinned.alloc(1 << 20).unwrap());
        let dev = mem.alloc(1 << 20).unwrap();
        let (from_a, from_b) = (
            Arc::new(std::sync::Mutex::new(Vec::new())),
            Arc::new(std::sync::Mutex::new(Vec::new())),
        );
        s[0].submit(|q| {
            q.keep(Box::new(host.clone()));
            // SAFETY: valid blocks; the host block is kept until a's retirement is confirmed.
            unsafe { q.copy_to_device(dev.addr, host.addr as *const u8, 1 << 20) }
        })
        .unwrap();
        s[0].notify_retired(Box::new(Record(from_a.clone())))
            .unwrap();
        s[1].notify_retired(Box::new(Record(from_b.clone())))
            .unwrap();
        assert!(wait_until(Duration::from_secs(5), || {
            !from_a.lock().unwrap().is_empty() && !from_b.lock().unwrap().is_empty()
        }));
        let (ticket_a, ticket_b) = (from_a.lock().unwrap()[0], from_b.lock().unwrap()[0]);
        let refused = s[0].confirm_retired(ticket_b);
        assert!(matches!(refused, Some(GpuError::Invalid(_))), "{refused:?}");
        assert_eq!(Arc::strong_count(&host), 2, "a's buffer is still kept");
        assert_eq!(s[0].confirm_retired(ticket_a), None);
        assert_eq!(Arc::strong_count(&host), 1);
        drop((pool, s));
        // SAFETY: the streams ended their work; nothing uses the blocks.
        unsafe {
            mem.free(dev);
            pinned.free(*host);
        }
    }

    #[test]
    fn kept_host_buffers_outlive_the_copy_until_confirmed() {
        let Some(gpu) = gpu() else { return };
        let (pool, mut s) = gpu.open_lanes(&["a"], StreamOptions::default()).unwrap();
        let mem = device_memory(&gpu);
        let pinned = PinnedHost::new(gpu.device().clone(), LocationId(1), 0);
        let host = Arc::new(pinned.alloc(1 << 20).unwrap());
        let dev = mem.alloc(1 << 20).unwrap();
        let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
        s[0].submit(|q| {
            q.keep(Box::new(host.clone()));
            // SAFETY: valid blocks; the host block is kept until the retirement is confirmed.
            unsafe { q.copy_to_device(dev.addr, host.addr as *const u8, 1 << 20) }
        })
        .unwrap();
        s[0].notify_retired(Box::new(Record(seen.clone()))).unwrap();
        assert!(wait_until(Duration::from_secs(5), || !seen
            .lock()
            .unwrap()
            .is_empty()));
        assert_eq!(Arc::strong_count(&host), 2, "kept until confirmed");
        let ticket = seen.lock().unwrap()[0];
        assert_eq!(s[0].confirm_retired(ticket), None);
        assert_eq!(Arc::strong_count(&host), 1, "released by the confirmation");
        drop((pool, s));
        // SAFETY: the stream ended its work; nothing uses the blocks.
        unsafe {
            mem.free(dev);
            pinned.free(*host);
        }
    }
}

mod compiler {
    use super::*;
    use euhedral_gpu::cuda::{Nvrtc, Target};
    use euhedral_gpu::{CompileOptions, GpuError};

    const OPS: &str = include_str!("kernels/ops.cu");

    #[test]
    fn nvrtc_compiles_for_this_device() {
        let Some(gpu) = gpu() else { return };
        let Ok(nvrtc) = Nvrtc::shared(None) else {
            eprintln!("no NVRTC: skipped");
            return;
        };
        let (major, minor) = gpu.arch();
        let target = nvrtc.target_for(gpu.arch()).unwrap();
        if nvrtc.version().0 >= 13 {
            assert_eq!(target, Target::Cubin((major * 10 + minor) as u32));
        }
        let image = nvrtc
            .compile(OPS, &CompileOptions::default(), target)
            .unwrap();
        match target {
            Target::Cubin(_) => assert_eq!(&image[..4], b"\x7fELF"),
            Target::Ptx(_) => assert_eq!(image.last(), Some(&0)),
        }
    }

    #[test]
    fn another_compiler_library_is_refused_once_one_is_loaded() {
        let Ok(loaded) = Nvrtc::shared(None) else {
            eprintln!("no NVRTC: skipped");
            return;
        };
        let again = Nvrtc::shared(None).unwrap();
        assert!(Arc::ptr_eq(&loaded, &again));
        let other = Nvrtc::shared(Some(std::path::Path::new("/nonexistent/other/libnvrtc.so")));
        assert!(matches!(other, Err(GpuError::Invalid(_))), "{other:?}");
        assert!(
            !loaded.build().is_empty(),
            "the build is named for the cache key"
        );
    }

    #[test]
    fn a_compile_error_returns_the_log() {
        let Some(gpu) = gpu() else { return };
        let Ok(nvrtc) = Nvrtc::shared(None) else {
            eprintln!("no NVRTC: skipped");
            return;
        };
        let target = nvrtc.target_for(gpu.arch()).unwrap();
        let r = nvrtc.compile(
            "extern \"C\" __global__ void k() { nope; }",
            &CompileOptions::default(),
            target,
        );
        match r {
            Err(GpuError::Compile { log }) => assert!(log.contains("nope"), "{log}"),
            other => panic!("{other:?}"),
        }
    }
}

mod modules {
    use super::*;
    use std::path::PathBuf;

    use euhedral_gpu::cuda::{CudaModule, Nvrtc};
    use euhedral_gpu::{CompileOptions, GpuError, Module};

    const OPS: &str = include_str!("kernels/ops.cu");
    const OPS_PTX: &[u8] = include_bytes!("kernels/ops.ptx");

    fn nvrtc_available() -> bool {
        let ok = Nvrtc::shared(None).is_ok();
        if !ok {
            eprintln!("no NVRTC: the source half is skipped");
        }
        ok
    }

    /// A fresh cache directory under the build's temporary directory.
    fn cache_dir(name: &str) -> PathBuf {
        let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
            .join(format!("{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn modules_load_from_source_and_from_an_image() {
        let Some(gpu) = gpu() else { return };
        if nvrtc_available() {
            let module = CudaModule::from_source(&gpu, OPS, &CompileOptions::default()).unwrap();
            let axpb = module.kernel("axpb").unwrap();
            assert_eq!(axpb.name(), "axpb");
            assert_eq!(axpb.parameter_sizes(), Some(&[8, 8, 4, 4, 4][..]));
        }
        if gpu.driver_version() >= 13010 {
            let module = CudaModule::from_image(&gpu, OPS_PTX).unwrap();
            let axpb = module.kernel("axpb").unwrap();
            assert_eq!(axpb.parameter_sizes(), Some(&[8, 8, 4, 4, 4][..]));
            assert_ne!(
                axpb.id(),
                module.kernel("add_u32").unwrap().id(),
                "kernels differ in identity"
            );
        } else {
            eprintln!("driver older than 13.1: the image half is skipped");
        }
    }

    use euhedral_gpu::Kernel;

    #[test]
    fn missing_kernels_are_not_found() {
        let Some(gpu) = gpu() else { return };
        if gpu.driver_version() < 13010 {
            return;
        }
        let module = CudaModule::from_image(&gpu, OPS_PTX).unwrap();
        let r = module.kernel("axpbx");
        assert!(matches!(r, Err(GpuError::NotFound(_))), "{r:?}");
    }

    #[test]
    fn a_corrupt_cache_entry_is_rebuilt() {
        let Some(gpu) = gpu() else { return };
        if !nvrtc_available() {
            return;
        }
        let dir = cache_dir("corrupt");
        let options = CompileOptions {
            cache_dir: Some(dir.clone()),
            ..CompileOptions::default()
        };
        CudaModule::from_source(&gpu, OPS, &options).unwrap();
        let entry = std::fs::read_dir(&dir)
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();
        std::fs::write(&entry, [0x13u8; 16]).unwrap();
        CudaModule::from_source(&gpu, OPS, &options).unwrap();
        assert_ne!(std::fs::metadata(&entry).unwrap().len(), 16, "rewritten");
    }

    #[test]
    fn a_cache_hit_skips_the_compiler() {
        let Some(gpu) = gpu() else { return };
        if !nvrtc_available() {
            return;
        }
        let dir = cache_dir("hit");
        let options = CompileOptions {
            cache_dir: Some(dir.clone()),
            ..CompileOptions::default()
        };
        CudaModule::from_source(&gpu, OPS, &options).unwrap();
        let entry = std::fs::read_dir(&dir)
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();
        let first = std::fs::metadata(&entry).unwrap().modified().unwrap();
        CudaModule::from_source(&gpu, OPS, &options).unwrap();
        let second = std::fs::metadata(&entry).unwrap().modified().unwrap();
        assert_eq!(first, second, "the cached file was used, not rewritten");
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 1);
    }
}

mod launches {
    use super::streams::{device_memory, reached_within};
    use super::*;
    use std::time::Duration;

    use euhedral_gpu::cuda::{CudaModule, Nvrtc};
    use euhedral_gpu::{
        Args, CompileOptions, Enqueue, Gpu, GpuError, Launch, Module, Stream, StreamOptions,
    };
    use euhedral_memory::cuda::PinnedHost;
    use euhedral_memory::{Backend, LocationId};

    const OPS: &str = include_str!("kernels/ops.cu");
    const OPS_PTX: &[u8] = include_bytes!("kernels/ops.ptx");

    #[test]
    fn axpb_from_source_and_from_ptx_match_the_cpu_bit_for_bit() {
        let Some(gpu) = gpu() else { return };
        let mut modules = Vec::new();
        if Nvrtc::shared(None).is_ok() {
            modules.push(CudaModule::from_source(&gpu, OPS, &CompileOptions::default()).unwrap());
        }
        if gpu.driver_version() >= 13010 {
            modules.push(CudaModule::from_image(&gpu, OPS_PTX).unwrap());
        }
        if modules.is_empty() {
            eprintln!("neither NVRTC nor a 13.1 driver: skipped");
            return;
        }
        let (pool, mut s) = gpu
            .open_lanes(&["compute"], StreamOptions::default())
            .unwrap();
        let mem = device_memory(&gpu);
        let pinned = PinnedHost::new(gpu.device().clone(), LocationId(1), 0);
        let n = 1usize << 20;
        let bytes = n * 4;
        let (a, b) = (1.5f32, -2.0f32);
        let x_host = pinned.alloc(bytes as u64).unwrap();
        // SAFETY: a pinned block of `bytes` bytes owned by this test.
        let x_slice = unsafe { std::slice::from_raw_parts_mut(x_host.addr as *mut f32, n) };
        for (i, v) in x_slice.iter_mut().enumerate() {
            *v = i as f32 * 0.25;
        }
        let x_dev = mem.alloc(bytes as u64).unwrap();
        s[0].submit(|q| {
            // SAFETY: valid blocks; the host block is untouched until the marker below is reached.
            unsafe { q.copy_to_device(x_dev.addr, x_host.addr as *const u8, bytes) }
        })
        .unwrap();
        let mut outs = Vec::new();
        for module in &modules {
            let axpb = module.kernel("axpb").unwrap();
            let y_dev = mem.alloc(bytes as u64).unwrap();
            let y_host = pinned.alloc(bytes as u64).unwrap();
            let args = Args::new()
                .device(y_dev.addr)
                .device(x_dev.addr)
                .arg(a)
                .arg(b)
                .arg(n as u32);
            s[0].submit(|q| {
                // SAFETY: the arguments match `axpb`'s parameters; the buffers are live until the marker.
                unsafe {
                    q.launch(&axpb, Launch::linear(n as u32, 256), &args)?;
                    q.copy_to_host(y_host.addr as *mut u8, y_dev.addr, bytes)
                }
            })
            .unwrap();
            outs.push((y_dev, y_host));
        }
        let m = s[0].mark().unwrap();
        assert!(reached_within(&pool, m, Duration::from_secs(10)));
        let reference: Vec<u32> = x_slice.iter().map(|&x| a.mul_add(x, b).to_bits()).collect();
        for (_, y_host) in &outs {
            // SAFETY: the pinned block was written by the copy that retired.
            let y = unsafe { std::slice::from_raw_parts(y_host.addr as *const u32, n) };
            assert!(y == reference.as_slice(), "bit-for-bit with the CPU");
        }
        drop((pool, s));
        // SAFETY: the stream ended its work; nothing uses the blocks.
        unsafe {
            for (d, h) in outs {
                mem.free(d);
                pinned.free(h);
            }
            mem.free(x_dev);
            pinned.free(x_host);
        }
    }

    #[test]
    fn arguments_that_do_not_fit_are_refused_before_launch() {
        let Some(gpu) = gpu() else { return };
        if gpu.driver_version() < 13010 {
            return;
        }
        let module = CudaModule::from_image(&gpu, OPS_PTX).unwrap();
        let axpb = module.kernel("axpb").unwrap();
        let (_pool, mut s) = gpu
            .open_lanes(&["compute"], StreamOptions::default())
            .unwrap();
        let args = Args::new()
            .device(0x1000)
            .device(0x2000)
            .arg(1.0f32)
            .arg(2.0f32);
        let r = s[0].submit(|q| {
            // SAFETY: refused before the driver sees it.
            unsafe { q.launch(&axpb, Launch::linear(1, 1), &args) }
        });
        assert!(matches!(r, Err(GpuError::Args(_))), "{r:?}");
        assert_eq!(gpu.poisoned(), None);
    }
}

mod markers {
    use super::streams::{device_memory, reached_within, wait_until};
    use super::*;
    use std::sync::Mutex;
    use std::time::Duration;

    use euhedral_gpu::cuda::CudaModule;
    use euhedral_gpu::{
        Args, CompileOptions, Enqueue, Gpu, GpuError, Launch, Module, RetireListener, RetireTicket,
        Stream, StreamOptions,
    };
    use euhedral_memory::cuda::PinnedHost;
    use euhedral_memory::{Backend, LocationId};

    const OPS: &str = include_str!("kernels/ops.cu");
    const OPS_PTX: &[u8] = include_bytes!("kernels/ops.ptx");

    /// The test kernels, from the image when the driver takes it, else from source.
    pub fn ops(gpu: &Arc<CudaGpu>) -> Option<CudaModule> {
        if gpu.driver_version() >= 13010 {
            CudaModule::from_image(gpu, OPS_PTX).ok()
        } else {
            CudaModule::from_source(gpu, OPS, &CompileOptions::default()).ok()
        }
    }

    /// Cycles of the `spin` kernel for about `ms` milliseconds.
    pub fn cycles(ms: u64) -> u64 {
        ms * 2_500_000
    }

    #[test]
    fn a_marker_orders_another_streams_work_on_the_device() {
        let Some(gpu) = gpu() else { return };
        let Some(ops) = ops(&gpu) else { return };
        let (spin, add) = (ops.kernel("spin").unwrap(), ops.kernel("add_u32").unwrap());
        let (pool, mut s) = gpu
            .open_lanes(&["a", "b"], StreamOptions::default())
            .unwrap();
        let mem = device_memory(&gpu);
        let pinned = PinnedHost::new(gpu.device().clone(), LocationId(1), 0);
        let n = 1usize << 16;
        let bytes = n * 4;
        let x_host = pinned.alloc(bytes as u64).unwrap();
        let out_host = pinned.alloc(bytes as u64).unwrap();
        // SAFETY: pinned blocks of `bytes` bytes owned by this test.
        let (x_slice, out_slice) = unsafe {
            (
                std::slice::from_raw_parts_mut(x_host.addr as *mut u32, n),
                std::slice::from_raw_parts_mut(out_host.addr as *mut u32, n),
            )
        };
        for (i, v) in x_slice.iter_mut().enumerate() {
            *v = i as u32;
        }
        out_slice.fill(0xFFFF_FFFF);
        let (x, z) = (
            mem.alloc(bytes as u64).unwrap(),
            mem.alloc(bytes as u64).unwrap(),
        );
        let add_args = Args::new()
            .device(z.addr)
            .device(x.addr)
            .arg(7u32)
            .arg(n as u32);
        s[0].submit(|q| {
            // SAFETY: valid blocks and matching arguments; the buffers outlive the markers waited for below.
            unsafe {
                q.copy_to_device(x.addr, x_host.addr as *const u8, bytes)?;
                q.memset(z.addr, 0, bytes)?;
                q.launch(&spin, Launch::linear(1, 1), &Args::new().arg(cycles(50)))?;
                q.launch(&add, Launch::linear(n as u32, 256), &add_args)
            }
        })
        .unwrap();
        let m = s[0].mark().unwrap();
        s[1].await_marker(m).unwrap();
        s[1].submit(|q| {
            // SAFETY: as above; this copy is ordered behind the marker on the device.
            unsafe { q.copy_to_host(out_host.addr as *mut u8, z.addr, bytes) }
        })
        .unwrap();
        let m1 = s[1].mark().unwrap();
        assert!(reached_within(&pool, m1, Duration::from_secs(10)));
        assert!(
            out_slice
                .iter()
                .enumerate()
                .all(|(i, &v)| v == i as u32 + 7),
            "the copy ran after the kernel"
        );
        drop((pool, s));
        // SAFETY: the streams ended their work; nothing uses the blocks.
        unsafe {
            mem.free(x);
            mem.free(z);
            pinned.free(x_host);
            pinned.free(out_host);
        }
    }

    struct Ordered(Arc<Mutex<Vec<u16>>>);

    impl RetireListener for Ordered {
        fn retired(self: Box<Self>, ticket: RetireTicket) {
            self.0.lock().unwrap().push(ticket.lane().0);
        }
    }

    #[test]
    fn retirements_on_two_streams_follow_their_markers() {
        let Some(gpu) = gpu() else { return };
        let Some(ops) = ops(&gpu) else { return };
        let spin = ops.kernel("spin").unwrap();
        let (pool, mut s) = gpu
            .open_lanes(&["a", "b"], StreamOptions::default())
            .unwrap();
        let order = Arc::new(Mutex::new(Vec::new()));
        s[0].submit(|q| {
            // SAFETY: `spin` takes one 64-bit argument and touches no memory.
            unsafe { q.launch(&spin, Launch::linear(1, 1), &Args::new().arg(cycles(20))) }
        })
        .unwrap();
        // The callback precedes the marker on stream a, and a callback finishes before later work on its stream.
        s[0].notify_retired(Box::new(Ordered(order.clone())))
            .unwrap();
        let m = s[0].mark().unwrap();
        s[1].await_marker(m).unwrap();
        s[1].notify_retired(Box::new(Ordered(order.clone())))
            .unwrap();
        assert!(wait_until(Duration::from_secs(10), || order
            .lock()
            .unwrap()
            .len()
            == 2));
        assert_eq!(*order.lock().unwrap(), [0, 1]);
        drop((pool, s));
    }

    #[test]
    fn a_full_marker_ring_is_busy_and_old_markers_need_no_wait() {
        let Some(gpu) = gpu() else { return };
        let Some(ops) = ops(&gpu) else { return };
        let spin = ops.kernel("spin").unwrap();
        let (pool, mut s) = gpu
            .open_lanes(&["a", "b"], StreamOptions { markers: 4 })
            .unwrap();
        let spin_for = |stream: &mut euhedral_gpu::cuda::CudaStream, ms: u64| {
            stream
                .submit(|q| {
                    // SAFETY: `spin` takes one 64-bit argument and touches no memory.
                    unsafe { q.launch(&spin, Launch::linear(1, 1), &Args::new().arg(cycles(ms))) }
                })
                .unwrap();
        };
        spin_for(&mut s[0], 300);
        let first = s[0].mark().unwrap();
        let mut last = first;
        for _ in 1..4 {
            last = s[0].mark().unwrap();
        }
        assert_eq!(
            s[0].mark(),
            Err(GpuError::Busy),
            "four markers are outstanding"
        );
        assert!(reached_within(&pool, last, Duration::from_secs(10)));
        // A fifth marker reuses the first slot, behind a long spin.
        spin_for(&mut s[0], 1500);
        let fifth = s[0].mark().unwrap();
        assert_eq!(fifth.value, 5);
        // The first marker completed long ago: waiting for it adds no device wait, though its slot now holds the
        // fifth marker, which is behind the spin.
        s[1].await_marker(first).unwrap();
        let other = s[1].mark().unwrap();
        assert!(reached_within(&pool, other, Duration::from_secs(10)));
        assert!(!pool.reached(fifth), "stream a is still spinning");
        assert!(reached_within(&pool, fifth, Duration::from_secs(30)));
        let _ = s[0].queue().completed();
        drop((pool, s));
    }
}
