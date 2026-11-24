//! Where tests write their scratch files: `TEST_TMPDIR` when it is set (Bazel sets it, and its
//! sandbox makes other paths read-only), otherwise the directory the test names, so a Cargo run is
//! unchanged.

#![allow(dead_code, reason = "each test binary uses part of the fixtures")]

use std::path::PathBuf;

/// The directory to put scratch files in, created if missing: `TEST_TMPDIR`, or `fallback()`.
pub fn test_scratch_dir(fallback: impl FnOnce() -> PathBuf) -> PathBuf {
    let dir = std::env::var_os("TEST_TMPDIR").map_or_else(fallback, PathBuf::from);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}
