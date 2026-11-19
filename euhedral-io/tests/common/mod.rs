//! What the integration tests share: the engines an `AsyncIo` can run on.

use std::sync::Arc;

use euhedral_io::AsyncIo;
#[cfg(any(target_os = "linux", target_os = "android"))]
use euhedral_io::AsyncIoOptions;

/// Opens an `AsyncIo` on the platform's default engine.
pub fn open_default() -> Arc<AsyncIo> {
    AsyncIo::open().unwrap()
}

/// Opens an `AsyncIo` on the positional engine (Linux and Android).
#[cfg(any(target_os = "linux", target_os = "android"))]
pub fn open_positional() -> Arc<AsyncIo> {
    AsyncIo::open_with_options(AsyncIoOptions {
        positional_io: true,
        ..AsyncIoOptions::default()
    })
    .unwrap()
}

/// The engines a test runs on: the default one, and where there is one, the positional fallback.
pub fn engines() -> Vec<fn() -> Arc<AsyncIo>> {
    #[cfg(any(target_os = "linux", target_os = "android"))]
    return vec![open_default, open_positional];
    #[cfg(not(any(target_os = "linux", target_os = "android")))]
    return vec![open_default];
}
