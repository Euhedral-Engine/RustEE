//! The CUDA backend on a real device. Every test skips, passing, when no driver or device is present.
#![cfg(feature = "cuda")]

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
        let (pool, s) = gpu
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
        assert_eq!(
            m,
            Marker {
                lane: LaneId(0),
                value: 1
            }
        );
        assert!(reached_within(&pool, m, Duration::from_secs(5)));
        assert_eq!(s[0].name(), "compute");
        drop((pool, s));
        // SAFETY: the stream ended its work before it dropped; nothing uses the block.
        unsafe { mem.free(buf) };
    }

    use euhedral_graph::DeviceQueue;

    #[test]
    fn copies_round_trip_through_the_device() {
        let Some(gpu) = gpu() else { return };
        let (pool, s) = gpu.open_lanes(&["copy"], StreamOptions::default()).unwrap();
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
        assert_eq!(s[0].completed(), 1);
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
        let (_pool, s) = gpu.open_lanes(&["a"], StreamOptions::default()).unwrap();
        let r = s[0].await_marker(Marker {
            lane: LaneId(9),
            value: 1,
        });
        assert!(matches!(r, Err(GpuError::Invalid(_))), "{r:?}");
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
        let (pool, s) = gpu.open_lanes(&["a"], StreamOptions::default()).unwrap();
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
            tickets.iter().map(|t| t.seq).collect::<Vec<_>>(),
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
        let (pool, s) = gpu.open_lanes(&["a"], StreamOptions::default()).unwrap();
        let mem = device_memory(&gpu);
        let buf = mem.alloc(1 << 20).unwrap();
        let completions = euhedral_graph::Completions::new(pool.clone());
        let ran = Arc::new(std::sync::Mutex::new(Vec::new()));
        s[0].submit(|q| {
            // SAFETY: a live device allocation, used by nothing else.
            unsafe { q.memset(buf.addr, 1, 1 << 20) }
        })
        .unwrap();
        euhedral_gpu::retire_into(&s[0], &completions, {
            let (stream, ran) = (s[0].clone(), ran.clone());
            move |ticket| {
                Box::new(RunnableFrame::new(
                    1,
                    Arc::new(move || ran.lock().unwrap().push(stream.confirm_retired(ticket))),
                ))
            }
        })
        .unwrap();
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
    fn kept_host_buffers_outlive_the_copy_until_confirmed() {
        let Some(gpu) = gpu() else { return };
        let (pool, s) = gpu.open_lanes(&["a"], StreamOptions::default()).unwrap();
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
