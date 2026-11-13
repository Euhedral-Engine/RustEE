//! Allocation sweep: CUDA against Vulkan on this GPU, weighted to small sizes.
//!
//! `cargo run --release -p euhedral-memory --features cuda --example alloc_sweep [-- --quick]`
//!
//! Prints CSV tables: per method and size, alloc and free latency p50/p99 (µs) for pairs (alloc then free at
//! once: churn) and batches (K live, then all freed); the import of a CUDA VMM allocation into Vulkan; copy
//! throughput; and whether a free or an allocation waits for unrelated busy device work.
//!
//! This is a Unix program (POSIX file descriptors, `mmap`); on Windows it prints a note and exits.

#[cfg(unix)]
#[path = "alloc_sweep/unix.rs"]
mod unix;

#[cfg(unix)]
fn main() {
    unix::main();
}

#[cfg(not(unix))]
fn main() {
    eprintln!("alloc_sweep: Unix only");
}
