//! How a device test says it did not run. CI reads the test output for these lines (`scripts/ci/device_check.py`):
//! a skip for want of a driver, a card or a kernel compiler fails the device job, because the test proved nothing;
//! a skip for a feature this card lacks is a hardware skip, which the job lists and accepts.

#![allow(dead_code, reason = "each test binary uses part of the fixtures")]

use std::fmt::Display;

/// The test did not run for want of a driver, a card, NVRTC or a compiled kernel.
///
/// With `EUHEDRAL_DEVICE_REQUIRED` set (`bazel test --config=gpu` sets it) this panics, so the test fails where it
/// would have skipped.
pub fn skip(reason: impl Display) {
    eprintln!("DEVICE-TEST-SKIPPED: {reason}");
    assert!(
        !device_required(),
        "EUHEDRAL_DEVICE_REQUIRED is set and the test would have skipped: {reason}"
    );
}

/// Whether a skip for want of a driver, a card or NVRTC must fail the test.
fn device_required() -> bool {
    std::env::var_os("EUHEDRAL_DEVICE_REQUIRED").is_some()
}

/// The test did not run because this card or driver lacks the feature it tests (listed, not a failure).
pub fn skip_hardware(reason: impl Display) {
    eprintln!("DEVICE-TEST-SKIPPED-HARDWARE: {reason}");
}
