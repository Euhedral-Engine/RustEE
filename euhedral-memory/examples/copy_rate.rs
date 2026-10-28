//! Copy rate between pinned host memory and the device: how close each way of pinning, copy size and stream count
//! comes to the link's rate.
//!
//! `cargo run --release -p euhedral-memory --features cuda --example copy_rate [-- --quick]`
//!
//! Host memory three ways: the driver's (`cuMemHostAlloc`), and an anonymous mapping registered with the driver
//! (`cuMemHostRegister`), on huge pages and on small pages. For each: host to device, device to host, and both at
//! once (one direction per stream group), copies of 2 to 128 MiB on 1, 2 or 4 streams, 1 GiB moved per direction and
//! measurement; GB/s is the bytes over the wall time from the first issue to the last stream's completion (p50 of 5).

use std::ffi::c_void;
use std::time::Instant;

use euhedral_memory::cuda::{CuStream, Device, Driver};

const MIB: usize = 1 << 20;
const HUGE: usize = 2 * MIB;
const TOTAL: usize = 1024 * MIB;

#[derive(Clone, Copy, Debug, PartialEq)]
enum Kind {
    Driver,
    RegisteredHuge,
    RegisteredSmall,
}

struct Host {
    ptr: *mut u8,
    len: usize,
    kind: Kind,
}

fn check(driver: &Driver, call: &'static str, code: euhedral_memory::cuda::CuResult) {
    if let Err(e) = driver.check(call, code) {
        panic!("{e}");
    }
}

fn host(driver: &Driver, kind: Kind, len: usize) -> Host {
    match kind {
        Kind::Driver => {
            let mut p = std::ptr::null_mut();
            // SAFETY: out pointer; flags 0; freed in `free_host`.
            check(driver, "cuMemHostAlloc", unsafe {
                (driver.fns.mem_host_alloc)(&mut p, len, 0)
            });
            Host {
                ptr: p.cast(),
                len,
                kind,
            }
        }
        Kind::RegisteredHuge | Kind::RegisteredSmall => {
            // SAFETY: an anonymous private mapping, unmapped in `free_host`.
            let p = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    len,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                    -1,
                    0,
                )
            };
            assert_ne!(p, libc::MAP_FAILED, "mmap");
            // SAFETY: the mapping made above; populate and touch its pages before registering them.
            unsafe {
                if kind == Kind::RegisteredHuge {
                    libc::madvise(p, len, libc::MADV_HUGEPAGE);
                }
                for off in (0..len).step_by(HUGE) {
                    let chunk = p.cast::<u8>().add(off);
                    if libc::madvise(chunk.cast(), HUGE.min(len - off), libc::MADV_POPULATE_WRITE)
                        != 0
                    {
                        for page in (0..HUGE.min(len - off)).step_by(4096) {
                            chunk.add(page).write_volatile(1);
                        }
                    }
                }
            }
            // SAFETY: `len` bytes of mapped, populated memory, not registered yet.
            check(driver, "cuMemHostRegister", unsafe {
                (driver.fns.mem_host_register)(p, len, 0)
            });
            Host {
                ptr: p.cast(),
                len,
                kind,
            }
        }
    }
}

fn free_host(driver: &Driver, h: Host) {
    // SAFETY: the memory `host` made, no copy in flight.
    unsafe {
        match h.kind {
            Kind::Driver => {
                (driver.fns.mem_free_host)(h.ptr.cast());
            }
            _ => {
                (driver.fns.mem_host_unregister)(h.ptr.cast());
                libc::munmap(h.ptr.cast(), h.len);
            }
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Dir {
    HtoD,
    DtoH,
    Both,
}

/// GB/s of moving `TOTAL` bytes per direction in copies of `size` over `streams` streams per direction.
fn run(driver: &Driver, h: &Host, d: u64, dir: Dir, size: usize, streams: &[CuStream]) -> f64 {
    let per_dir = if dir == Dir::Both {
        streams.len() / 2
    } else {
        streams.len()
    };
    let copies = TOTAL / size;
    let start = Instant::now();
    for i in 0..copies {
        let off = (i * size) % (h.len - size + 1);
        let dev = d + ((i * size) % (TOTAL - size + 1)) as u64;
        let hp = h.ptr.wrapping_add(off);
        let issue = |htod: bool, s: CuStream| {
            // SAFETY: both ranges lie inside their allocations; the host range is pinned; the stream is live.
            let code = unsafe {
                if htod {
                    (driver.fns.memcpy_htod_async)(dev, hp.cast::<c_void>().cast_const(), size, s)
                } else {
                    (driver.fns.memcpy_dtoh_async)(hp.cast(), dev, size, s)
                }
            };
            check(driver, "cuMemcpyAsync", code);
        };
        match dir {
            Dir::HtoD => issue(true, streams[i % per_dir]),
            Dir::DtoH => issue(false, streams[i % per_dir]),
            Dir::Both => {
                issue(true, streams[i % per_dir]);
                issue(false, streams[per_dir + i % per_dir]);
            }
        }
    }
    for &s in streams {
        // SAFETY: a live stream.
        check(driver, "cuStreamSynchronize", unsafe {
            (driver.fns.stream_synchronize)(s)
        });
    }
    let bytes = if dir == Dir::Both { 2 * TOTAL } else { TOTAL } as f64;
    bytes / start.elapsed().as_secs_f64() / 1e9
}

fn main() {
    let quick = std::env::args().any(|a| a == "--quick");
    let driver = Driver::load().expect("libcuda");
    let device = Device::open(driver.clone(), 0).expect("device 0");
    let _current = device.enter().expect("context");
    let mut d = 0;
    // SAFETY: out pointer; freed at the end.
    check(&driver, "cuMemAlloc", unsafe {
        (driver.fns.mem_alloc)(&mut d, TOTAL)
    });
    let mut streams = Vec::new();
    for _ in 0..8 {
        let mut s = std::ptr::null_mut();
        // SAFETY: out pointer; destroyed at the end.
        check(&driver, "cuStreamCreate", unsafe {
            (driver.fns.stream_create)(&mut s, euhedral_memory::cuda::STREAM_NON_BLOCKING)
        });
        streams.push(s);
    }
    let sizes: &[usize] = if quick {
        &[8 * MIB, 64 * MIB]
    } else {
        &[2 * MIB, 8 * MIB, 32 * MIB, 128 * MIB]
    };
    let counts: &[usize] = if quick { &[1, 2] } else { &[1, 2, 4] };
    println!(
        "# copy rate, GB/s (1 GiB per direction, p50 of 5); both = the sum of the two directions"
    );
    println!("host,direction,copy_mib,streams,gbs");
    for kind in [Kind::Driver, Kind::RegisteredHuge, Kind::RegisteredSmall] {
        let made = Instant::now();
        let h = host(&driver, kind, TOTAL);
        eprintln!(
            "{kind:?}: 1 GiB pinned in {:.0} ms",
            made.elapsed().as_secs_f64() * 1e3
        );
        for dir in [Dir::HtoD, Dir::DtoH, Dir::Both] {
            for &size in sizes {
                for &n in counts {
                    let used: Vec<CuStream> =
                        streams[..if dir == Dir::Both { 2 * n } else { n }].to_vec();
                    let _ = run(&driver, &h, d, dir, size, &used);
                    let mut r: Vec<f64> = (0..5)
                        .map(|_| run(&driver, &h, d, dir, size, &used))
                        .collect();
                    r.sort_by(f64::total_cmp);
                    let name = match dir {
                        Dir::HtoD => "htod",
                        Dir::DtoH => "dtoh",
                        Dir::Both => "both",
                    };
                    println!("{kind:?},{name},{},{n},{:.1}", size / MIB, r[2]);
                }
            }
        }
        free_host(&driver, h);
    }
    // SAFETY: everything above is synchronised.
    unsafe {
        for s in streams {
            (driver.fns.stream_destroy)(s);
        }
        (driver.fns.mem_free)(d);
    }
}
