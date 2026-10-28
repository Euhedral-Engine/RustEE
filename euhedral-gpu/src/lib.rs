//! Asynchronous, backend-neutral GPU access on the Euhedral control plane.
//!
//! Streams are device lanes, markers order work across streams on the device, retirement is a completion that
//! continues as a frame, kernels are loaded at run time, and a recurring sequence of submissions is recorded once and
//! replayed. Nothing here waits on the device from a lattice worker.
