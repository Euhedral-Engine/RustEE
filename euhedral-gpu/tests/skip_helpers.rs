//! The device-test skip helper fails the test where it would have skipped when `EUHEDRAL_DEVICE_REQUIRED` is set, and
//! prints its line and returns when it is not. The probe runs in a child process, so the parent can read how it ended.

#[path = "common/skip.rs"]
mod skip;

use std::process::{Command, Output};

const PROBE_MARKER: &str = "EUHEDRAL_SKIP_PROBE";

/// Runs only as the child of the tests below: calls `skip` and, if it returns, says so.
#[test]
fn skip_probe() {
    if std::env::var_os(PROBE_MARKER).is_none() {
        return;
    }
    skip::skip("probe");
    eprintln!("PROBE-RETURNED");
}

#[allow(
    clippy::disallowed_methods,
    reason = "the test reads how a child process ended: it runs one test that returns at once"
)]
fn run_probe(required: bool) -> Output {
    let mut child = Command::new(std::env::current_exe().unwrap());
    child
        .args(["skip_probe", "--exact", "--nocapture", "--test-threads=1"])
        .env(PROBE_MARKER, "1")
        .env_remove("EUHEDRAL_DEVICE_REQUIRED");
    if required {
        child.env("EUHEDRAL_DEVICE_REQUIRED", "1");
    }
    child.output().unwrap()
}

#[test]
fn a_skip_prints_its_line_and_returns_when_a_device_is_not_required() {
    let out = run_probe(false);
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "{err}");
    assert!(err.contains("DEVICE-TEST-SKIPPED: probe"), "{err}");
    assert!(err.contains("PROBE-RETURNED"), "{err}");
}

#[test]
fn a_skip_fails_the_test_when_a_device_is_required() {
    let out = run_probe(true);
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "{err}");
    assert!(err.contains("DEVICE-TEST-SKIPPED: probe"), "{err}");
    assert!(!err.contains("PROBE-RETURNED"), "{err}");
}
