//! A deliberate device fault under lane sources where only the item behind the fault asks for a retirement: its
//! callback never comes, and the lane's look at the device must find the fault and report it. A sticky fault poisons the
//! context for the whole process, so this file holds one test and runs as its own process. Skips, passing, when no
//! driver or device is present.
#![cfg(feature = "cuda")]

mod common;

#[test]
fn a_retirement_behind_a_faulted_lane_is_reported_not_stranded() {
    common::cuda_support::fault_run(false);
}
