//! The pitched device-to-device copy on the fake backend: where the bytes land, the refusals, and the recorded shape.
//! No GPU.
#![cfg(feature = "fake")]

use std::sync::Arc;

use euhedral_gpu::fake::{FakeGpu, FakeOp, FakeStream};
use euhedral_gpu::{Gpu, GpuError, Graph, LaneEnqueue, Sequence, Stream, StreamOptions};

const FILL: u8 = 0xEE;

fn one_stream() -> (Arc<FakeGpu>, FakeStream) {
    let gpu = FakeGpu::new();
    let (_pool, mut s) = gpu.open_lanes(&["a"], StreamOptions::default()).unwrap();
    (gpu, s.remove(0))
}

/// A source of `len` bytes holding `i + 1` at byte `i`, and a destination of `len` bytes holding `FILL`.
fn buffers(gpu: &FakeGpu, src_len: usize, dst_len: usize) -> (u64, u64) {
    let src = gpu.alloc(src_len);
    let dst = gpu.alloc(dst_len);
    let pattern: Vec<u8> = (0..src_len).map(|i| (i % 250) as u8 + 1).collect();
    gpu.write(src, &pattern);
    gpu.write(dst, &vec![FILL; dst_len]);
    (dst, src)
}

fn copy(
    s: &mut FakeStream,
    dst: (u64, usize),
    src: (u64, usize),
    row_bytes: usize,
    rows: usize,
) -> Result<(), GpuError> {
    s.submit(|q| {
        // SAFETY: fake allocations that outlive the run.
        unsafe { q.copy_pitched_on_device(dst.0, dst.1, src.0, src.1, row_bytes, rows) }
    })
}

/// What the destination holds after the copy, row by row, as the model says.
fn expected(
    dst_len: usize,
    dst_pitch: usize,
    src_pitch: usize,
    row_bytes: usize,
    rows: usize,
) -> Vec<u8> {
    let mut want = vec![FILL; dst_len];
    for r in 0..rows {
        for b in 0..row_bytes {
            want[r * dst_pitch + b] = ((r * src_pitch + b) % 250) as u8 + 1;
        }
    }
    want
}

#[test]
fn rows_land_at_the_destination_pitch_and_the_gaps_stay_untouched() {
    let (gpu, mut s) = one_stream();
    let (dst, src) = buffers(&gpu, 4 * 12, 4 * 20);
    copy(&mut s, (dst, 20), (src, 12), 7, 4).unwrap();
    gpu.run_until_idle();
    assert_eq!(gpu.read(dst, 80), expected(80, 20, 12, 7, 4));
    assert_eq!(gpu.log(), [FakeOp::Copy { lane: 0 }]);
}

#[test]
fn a_narrower_destination_pitch_packs_the_rows() {
    let (gpu, mut s) = one_stream();
    let (dst, src) = buffers(&gpu, 5 * 32, 5 * 9);
    copy(&mut s, (dst, 9), (src, 32), 9, 5).unwrap();
    gpu.run_until_idle();
    assert_eq!(gpu.read(dst, 45), expected(45, 9, 32, 9, 5));
}

#[test]
fn the_last_rows_gap_is_not_part_of_either_range() {
    let (gpu, mut s) = one_stream();
    // Exactly `pitch * (rows - 1) + row_bytes` bytes on each side: a larger range would be out of the allocation.
    let (dst, src) = buffers(&gpu, 2 * 16 + 5, 2 * 11 + 5);
    copy(&mut s, (dst, 11), (src, 16), 5, 3).unwrap();
    gpu.run_until_idle();
    assert_eq!(gpu.read(dst, 27), expected(27, 11, 16, 5, 3));
    assert_eq!(gpu.log().len(), 1);
}

#[test]
fn one_row_is_a_plain_copy_whatever_the_pitches() {
    let (gpu, mut s) = one_stream();
    let (dst, src) = buffers(&gpu, 16, 16);
    copy(&mut s, (dst, 99), (src, 77), 10, 1).unwrap();
    gpu.run_until_idle();
    assert_eq!(gpu.read(dst, 16), expected(16, 99, 77, 10, 1));
}

#[test]
fn both_pitches_equal_to_the_row_is_one_plain_copy() {
    let (gpu, mut s) = one_stream();
    let (dst, src) = buffers(&gpu, 24, 30);
    copy(&mut s, (dst, 6), (src, 6), 6, 4).unwrap();
    gpu.run_until_idle();
    let mut want = vec![FILL; 30];
    for (i, b) in want.iter_mut().take(24).enumerate() {
        *b = i as u8 + 1;
    }
    assert_eq!(gpu.read(dst, 30), want);
}

#[test]
fn nothing_is_copied_for_zero_rows_or_zero_row_bytes() {
    let (gpu, mut s) = one_stream();
    let (dst, src) = buffers(&gpu, 16, 16);
    copy(&mut s, (dst, 8), (src, 8), 4, 0).unwrap();
    copy(&mut s, (dst, 8), (src, 8), 0, 3).unwrap();
    // Even the pitches' bound and the overflow check apply to nothing when there is nothing to copy.
    gpu.run_until_idle();
    assert_eq!(gpu.read(dst, 16), vec![FILL; 16]);
}

#[test]
fn a_pitch_shorter_than_the_row_is_refused_and_nothing_is_enqueued() {
    let (gpu, mut s) = one_stream();
    let (dst, src) = buffers(&gpu, 64, 64);
    for (dst_pitch, src_pitch) in [(7, 8), (8, 7)] {
        let refused = copy(&mut s, (dst, dst_pitch), (src, src_pitch), 8, 2);
        assert!(matches!(refused, Err(GpuError::Invalid(_))), "{refused:?}");
    }
    gpu.run_until_idle();
    assert!(gpu.log().is_empty());
    assert_eq!(gpu.read(dst, 64), vec![FILL; 64]);
}

#[test]
fn a_range_that_overflows_is_refused() {
    let (gpu, mut s) = one_stream();
    let (dst, src) = buffers(&gpu, 8, 8);
    let huge = usize::MAX / 2 + 1;
    for (dst_pitch, src_pitch) in [(huge, 8), (8, huge)] {
        let refused = copy(&mut s, (dst, dst_pitch), (src, src_pitch), 8, 3);
        assert!(matches!(refused, Err(GpuError::Invalid(_))), "{refused:?}");
    }
    // The last byte of the range is what overflows, not the product.
    let refused = copy(&mut s, (dst, usize::MAX), (src, usize::MAX), usize::MAX, 2);
    assert!(matches!(refused, Err(GpuError::Invalid(_))), "{refused:?}");
    gpu.run_until_idle();
    assert!(gpu.log().is_empty());
}

fn shape(
    gpu: &FakeGpu,
    dst: (u64, usize),
    src: (u64, usize),
    row_bytes: usize,
    rows: usize,
) -> euhedral_gpu::Fingerprint {
    gpu.fingerprint(1, |q| {
        q.submit(0, |e| {
            // SAFETY: nothing is enqueued on a probe.
            unsafe { e.copy_pitched_on_device(dst.0, dst.1, src.0, src.1, row_bytes, rows) }
        })
    })
    .unwrap()
}

#[test]
fn the_shape_tells_the_pitches_the_row_the_rows_and_the_addresses_apart() {
    let (gpu, _s) = one_stream();
    let base = shape(&gpu, (0x1000, 64), (0x2000, 48), 40, 8);
    assert_eq!(base, shape(&gpu, (0x1000, 64), (0x2000, 48), 40, 8));
    for other in [
        shape(&gpu, (0x1000, 72), (0x2000, 48), 40, 8),
        shape(&gpu, (0x1000, 64), (0x2000, 56), 40, 8),
        shape(&gpu, (0x1000, 64), (0x2000, 48), 32, 8),
        shape(&gpu, (0x1000, 64), (0x2000, 48), 40, 7),
        shape(&gpu, (0x1008, 64), (0x2000, 48), 40, 8),
        shape(&gpu, (0x1000, 64), (0x2008, 48), 40, 8),
        // Swapping the two pitches is another copy.
        shape(&gpu, (0x1000, 48), (0x2000, 64), 40, 8),
    ] {
        assert_ne!(base, other);
    }
}

#[test]
fn the_shape_is_not_that_of_a_plain_copy_of_the_same_bytes() {
    let (gpu, _s) = one_stream();
    let plain = gpu
        .fingerprint(1, |q| {
            q.submit(0, |e| {
                // SAFETY: nothing is enqueued on a probe.
                unsafe { e.copy_on_device(0x1000, 0x2000, 32) }
            })
        })
        .unwrap();
    assert_ne!(plain, shape(&gpu, (0x1000, 8), (0x2000, 8), 8, 4));
}

#[test]
fn a_recorded_pitched_copy_replays_as_recorded_and_its_shape_is_checked_on_update() {
    let (gpu, mut s) = one_stream();
    let (dst, src) = buffers(&gpu, 4 * 12, 4 * 20);
    let mut graph = gpu
        .record(&mut [&mut s], |q| {
            q.submit(0, |e| {
                // SAFETY: fake allocations that outlive the runs.
                unsafe { e.copy_pitched_on_device(dst, 20, src, 12, 7, 4) }
            })
        })
        .unwrap();
    gpu.run_until_idle();
    assert_eq!(gpu.read(dst, 80), expected(80, 20, 12, 7, 4));
    gpu.write(dst, &[FILL; 80]);
    graph.replay(&mut s).unwrap();
    gpu.run_until_idle();
    assert_eq!(gpu.read(dst, 80), expected(80, 20, 12, 7, 4));
    // Another pitch is another shape: the update applies nothing.
    let refused = gpu.update(&mut graph, |q| {
        q.submit(0, |e| {
            // SAFETY: nothing is enqueued on an update.
            unsafe { e.copy_pitched_on_device(dst, 21, src, 12, 7, 4) }
        })
    });
    assert!(refused.is_err());
    // The same shape is accepted.
    gpu.update(&mut graph, |q| {
        q.submit(0, |e| {
            // SAFETY: as above.
            unsafe { e.copy_pitched_on_device(dst, 20, src, 12, 7, 4) }
        })
    })
    .unwrap();
}
