//! The pitched device-to-device copy on a card, directly and through a recorded graph: every byte of the destination
//! is compared, the gaps between rows included. Skips, passing, when no driver or device is present.
#![cfg(feature = "cuda")]
#![allow(
    clippy::disallowed_methods,
    reason = "the test thread is not a lattice worker: it waits for the device with a bounded poll"
)]

mod common;

use std::sync::Arc;
use std::time::Duration;

use common::cuda_support::{gpu, wait_until};
use euhedral_gpu::cuda::CudaStream;
use euhedral_gpu::{Gpu, Graph, LaneEnqueue, Sequence, Stream, StreamOptions};
use euhedral_graph::LanePool;
use euhedral_memory::cuda::{DeviceMemory, PinnedHost};
use euhedral_memory::{Backend, LocationId};

const FILL: u8 = 0xEE;

/// One copy: the pitches, the row length and the number of rows.
#[derive(Clone, Copy)]
struct Case {
    dst_pitch: usize,
    src_pitch: usize,
    row_bytes: usize,
    rows: usize,
}

const fn case(dst_pitch: usize, src_pitch: usize, row_bytes: usize, rows: usize) -> Case {
    Case {
        dst_pitch,
        src_pitch,
        row_bytes,
        rows,
    }
}

const CASES: [Case; 6] = [
    // The destination pitch is wider than the source's.
    case(256, 100, 77, 9),
    // And narrower: the rows pack.
    case(77, 512, 77, 9),
    // Odd sizes, neither pitch aligned.
    case(131, 97, 61, 33),
    // One row: a plain copy whatever the pitches.
    case(300, 200, 150, 1),
    // Both pitches equal to the row: a plain copy.
    case(64, 64, 64, 16),
    // A larger tile.
    case(4096 + 64, 2048, 1500, 300),
];

fn span(pitch: usize, case: Case) -> usize {
    pitch * (case.rows - 1) + case.row_bytes
}

fn pattern(i: usize) -> u8 {
    (i * 31 % 251) as u8
}

/// The destination after the copy, as the model says: the source's bytes in the rows and `FILL` everywhere else.
fn expected(case: Case, len: usize) -> Vec<u8> {
    let mut want = vec![FILL; len];
    for r in 0..case.rows {
        for b in 0..case.row_bytes {
            want[r * case.dst_pitch + b] = pattern(r * case.src_pitch + b);
        }
    }
    want
}

fn settle(pool: &Arc<LanePool>, s: &mut CudaStream) {
    let m = s.mark().unwrap();
    assert!(
        wait_until(Duration::from_secs(10), || pool.reached(m)),
        "the work ends"
    );
}

#[test]
fn a_pitched_copy_moves_the_rows_and_leaves_the_gaps_alone_directly_and_replayed() {
    let Some(gpu) = gpu() else { return };
    let (pool, mut s) = gpu.open_lanes(&["copy"], StreamOptions::default()).unwrap();
    let s = &mut s[0];
    let mem = DeviceMemory {
        device: gpu.device().clone(),
        location: LocationId(0),
    };
    let pinned = PinnedHost::new(gpu.device().clone(), LocationId(1), 0);
    for (n, case) in CASES.into_iter().enumerate() {
        // The destination's gap after the last row is part of its allocation and must stay untouched too.
        let (src_len, dst_len) = (
            span(case.src_pitch, case) + 64,
            span(case.dst_pitch, case) + 64,
        );
        let len = src_len.max(dst_len);
        let (src, dst) = (
            mem.alloc(src_len as u64).unwrap(),
            mem.alloc(dst_len as u64).unwrap(),
        );
        let host = pinned.alloc(len as u64).unwrap();
        // SAFETY: a pinned block of `len` bytes, owned by this test; the device is idle whenever it is touched.
        let bytes = unsafe { std::slice::from_raw_parts_mut(host.addr as *mut u8, len) };
        let want = expected(case, dst_len);

        let upload = |s: &mut CudaStream, to: u64, len: usize| {
            s.submit(|q| {
                // SAFETY: valid blocks, untouched until the settle that follows.
                unsafe { q.copy_to_device(to, host.addr as *const u8, len) }
            })
            .unwrap();
            settle(&pool, s);
        };
        let download = |s: &mut CudaStream, from: u64, len: usize| {
            s.submit(|q| {
                // SAFETY: as above.
                unsafe { q.copy_to_host(host.addr as *mut u8, from, len) }
            })
            .unwrap();
            settle(&pool, s);
        };

        for (i, b) in bytes[..src_len].iter_mut().enumerate() {
            *b = pattern(i);
        }
        upload(s, src.addr, src_len);

        // Directly.
        bytes[..dst_len].fill(FILL);
        upload(s, dst.addr, dst_len);
        s.submit(|q| {
            // SAFETY: live device blocks, which nothing else uses until the settle that follows.
            unsafe {
                q.copy_pitched_on_device(
                    dst.addr,
                    case.dst_pitch,
                    src.addr,
                    case.src_pitch,
                    case.row_bytes,
                    case.rows,
                )
            }
        })
        .unwrap();
        settle(&pool, s);
        bytes.fill(0);
        download(s, dst.addr, dst_len);
        assert!(bytes[..dst_len] == want[..], "case {n}: the direct copy");

        // Recorded into a graph (the recording runs it once), then replayed on a refilled destination.
        bytes[..dst_len].fill(FILL);
        upload(s, dst.addr, dst_len);
        let mut graph = gpu
            .record(&mut [&mut *s], |q| {
                q.submit(0, |e| {
                    // SAFETY: as above; the blocks outlive the graph's runs, which end before they are freed.
                    unsafe {
                        e.copy_pitched_on_device(
                            dst.addr,
                            case.dst_pitch,
                            src.addr,
                            case.src_pitch,
                            case.row_bytes,
                            case.rows,
                        )
                    }
                })
            })
            .unwrap();
        settle(&pool, s);
        bytes.fill(0);
        download(s, dst.addr, dst_len);
        assert!(
            bytes[..dst_len] == want[..],
            "case {n}: the recording's run"
        );
        bytes[..dst_len].fill(FILL);
        upload(s, dst.addr, dst_len);
        graph.replay(s).unwrap();
        settle(&pool, s);
        bytes.fill(0);
        download(s, dst.addr, dst_len);
        assert!(bytes[..dst_len] == want[..], "case {n}: the replay");
        drop(graph);
        // SAFETY: the work on these blocks has ended.
        unsafe {
            mem.free(src);
            mem.free(dst);
            pinned.free(host);
        }
    }
}
