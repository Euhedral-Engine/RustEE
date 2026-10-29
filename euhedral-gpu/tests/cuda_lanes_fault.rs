//! A deliberate device fault under lane sources, with the faulting item and the item behind it both asking for a
//! retirement. A sticky fault poisons the context for the whole process, so this file holds one test and runs as its
//! own process. Skips, passing, when no driver or device is present.
#![cfg(feature = "cuda")]

mod common;

#[test]
fn a_poisoned_device_fails_armed_items() {
    common::cuda_support::fault_run(true);
}
