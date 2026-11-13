//! Copy rate between pinned host memory and the device: how close each way of pinning, copy size and stream count
//! comes to the link's rate.
//!
//! `cargo run --release -p euhedral-memory --features cuda --example copy_rate [-- --quick]`
//!
//! Host memory three ways: the driver's (`cuMemHostAlloc`), and an anonymous mapping registered with the driver
//! (`cuMemHostRegister`), on huge pages and on small pages. For each: host to device, device to host, and both at
//! once (one direction per stream group), copies of 2 to 128 MiB on 1, 2 or 4 streams, 1 GiB moved per direction and
//! measurement; GB/s is the bytes over the wall time from the first issue to the last stream's completion (p50 of 5).
//!
//! This is a Unix program (POSIX file descriptors, `mmap`); on Windows it prints a note and exits.

#[cfg(unix)]
#[path = "copy_rate/unix.rs"]
mod unix;

#[cfg(unix)]
fn main() {
    unix::main();
}

#[cfg(not(unix))]
fn main() {
    eprintln!("copy_rate: Unix only");
}
