//! The CUDA backend (feature `cuda`).
//!
//! It takes `euhedral-memory`'s shared [`Driver`] and [`Device`], so there is one loaded driver and one primary
//! context, and looks up its extra entry points ([`fns::GpuFns`]) in the same library.

mod enqueue;
pub mod fns;
mod graph;
mod module;
pub mod nvrtc;
mod retire;
mod stream;

pub use enqueue::CudaEnqueue;
pub use graph::{CudaGraph, CudaSequence};
pub use module::{CudaKernel, CudaModule};
pub use nvrtc::{Nvrtc, Target};
pub use stream::{CudaQueue, CudaStream};

use std::sync::Arc;

use euhedral_memory::cuda::{CudaError, Device, Driver};

use euhedral_graph::LanePool;

use enqueue::Mode;

use crate::{Fingerprint, Gpu, GpuError, Poison, Rerecorded, StreamOptions};

use fns::{ATTR_CC_MAJOR, ATTR_CC_MINOR, GpuFns};

/// Device faults: the context is lost and the device is poisoned. These are the codes for which the driver says
/// further work returns the same error or the process must be relaunched:
/// - 214 uncorrectable ECC error, 220 uncorrectable NVLink error, 226 contained error (an error contained to the
///   context);
/// - 700 illegal address, 702 launch timeout, 709 destroyed context, 710 assert;
/// - 714 hardware stack error, 715 illegal instruction, 716 misaligned address, 717 invalid address space, 718
///   invalid program counter, 719 launch failed;
/// - 721 tensor memory leak, and 999 unknown.
pub(crate) fn is_fault(code: i32) -> bool {
    matches!(
        code,
        214 | 220 | 226 | 700 | 702 | 709 | 710 | 714..=719 | 721 | 999
    )
}

/// Records `err` as the device's cause when it is a device fault.
pub(crate) fn note_fault(poison: &Poison, err: &GpuError) {
    if matches!(err, GpuError::Fault { .. }) {
        poison.set(err.clone());
    }
}

/// Records `err` as the device's cause whatever it is: for a query that fails in a way nobody can recover from, where
/// carrying on would strand the continuations behind it.
pub(crate) fn note_unrecoverable(poison: &Poison, err: &GpuError) {
    poison.set(err.clone());
}

/// The error for driver `code` of `call`: a [`GpuError::Fault`] for a device fault, else a [`GpuError::Call`].
pub(crate) fn classify(call: &'static str, code: i32, name: String) -> GpuError {
    if is_fault(code) {
        GpuError::Fault { call, code, name }
    } else {
        GpuError::Call { call, code, name }
    }
}

pub(crate) fn from_cuda(e: CudaError) -> GpuError {
    classify(e.call, e.code, e.name)
}

/// A CUDA device: `euhedral-memory`'s device, plus the entry points this crate adds.
#[derive(Debug)]
pub struct CudaGpu {
    device: Arc<Device>,
    pub(crate) fns: GpuFns,
    pub(crate) poison: Poison,
    arch: (i32, i32),
    driver_version: i32,
}

impl CudaGpu {
    /// Opens device `ordinal` on the process's shared driver.
    pub fn open(ordinal: i32) -> Result<Arc<Self>, GpuError> {
        let driver = Driver::shared().map_err(GpuError::Load)?;
        let device = Device::open(driver, ordinal).map_err(from_cuda)?;
        Self::from_device(device)
    }

    /// A GPU over the exact [`Device`] (and so the same primary context) that `euhedral-memory`'s backends use.
    pub fn from_device(device: Arc<Device>) -> Result<Arc<Self>, GpuError> {
        let fns = GpuFns::load(&device.driver)?;
        let attribute = |attr| -> Result<i32, GpuError> {
            let mut value = 0;
            // SAFETY: an out pointer; the attribute and the device handle are plain integers.
            let code = unsafe { (fns.device_get_attribute)(&mut value, attr, device.ordinal) };
            device
                .driver
                .check("cuDeviceGetAttribute", code)
                .map_err(from_cuda)?;
            Ok(value)
        };
        let arch = (attribute(ATTR_CC_MAJOR)?, attribute(ATTR_CC_MINOR)?);
        let mut driver_version = 0;
        // SAFETY: an out pointer.
        let code = unsafe { (fns.driver_get_version)(&mut driver_version) };
        device
            .driver
            .check("cuDriverGetVersion", code)
            .map_err(from_cuda)?;
        Ok(Arc::new(Self {
            device,
            fns,
            poison: Poison::new(),
            arch,
            driver_version,
        }))
    }

    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }

    /// The device's compute capability.
    pub fn arch(&self) -> (i32, i32) {
        self.arch
    }

    /// The driver's CUDA version, for example 13040.
    pub fn driver_version(&self) -> i32 {
        self.driver_version
    }

    /// `Ok` for `CUDA_SUCCESS`; else the error, and a device fault poisons the device.
    pub(crate) fn check(&self, call: &'static str, code: i32) -> Result<(), GpuError> {
        match self.device.driver.check(call, code) {
            Ok(()) => Ok(()),
            Err(e) => {
                let err = from_cuda(e);
                note_fault(&self.poison, &err);
                Err(err)
            }
        }
    }

    /// As [`check`](Self::check), for a call whose failure leaves the device unusable whatever the code: the device
    /// is poisoned with it.
    pub(crate) fn fail(&self, call: &'static str, code: i32) -> GpuError {
        let err = match self.device.driver.check(call, code) {
            Ok(()) => classify(call, code, "unknown".into()),
            Err(e) => from_cuda(e),
        };
        note_unrecoverable(&self.poison, &err);
        err
    }
}

impl Gpu for CudaGpu {
    type Stream = CudaStream;
    type Module = CudaModule;
    type Kernel = CudaKernel;
    type Graph = CudaGraph;
    type Sequence<'a> = CudaSequence<'a>;

    fn open_lanes(
        self: &Arc<Self>,
        names: &[&str],
        options: StreamOptions,
    ) -> Result<crate::Lanes<Self>, GpuError> {
        let streams = CudaStream::open(self, names, options)?;
        // The pool holds the shared query sides; the submitters go to their owners.
        let queues = streams
            .iter()
            .map(|s| s.queue_arc() as Arc<dyn euhedral_graph::DeviceQueue>)
            .collect();
        Ok((Arc::new(LanePool::new(queues)), streams))
    }

    fn poisoned(&self) -> Option<GpuError> {
        self.poison.get()
    }

    fn record<F>(&self, streams: &mut [&mut CudaStream], f: F) -> Result<CudaGraph, GpuError>
    where
        F: FnOnce(&mut CudaSequence<'_>) -> Result<(), GpuError>,
    {
        // The graph keeps its device alive: the streams hold the `Arc`.
        let gpu = streams
            .first()
            .ok_or_else(|| GpuError::Invalid("a recording needs a stream".into()))?
            .gpu
            .clone();
        let count = streams.len();
        let (rec, graph) = gpu.capture(streams, f)?;
        gpu.instantiate(rec, graph, count)
    }

    fn rerecord<F>(
        &self,
        graph: &mut CudaGraph,
        streams: &mut [&mut CudaStream],
        f: F,
    ) -> Result<Rerecorded, GpuError>
    where
        F: FnOnce(&mut CudaSequence<'_>) -> Result<(), GpuError>,
    {
        let gpu = graph.gpu().clone();
        let count = streams.len();
        let (rec, new) = gpu.capture(streams, f)?;
        graph.replace(rec, new, count)
    }

    fn fingerprint<F>(&self, streams: usize, f: F) -> Result<Fingerprint, GpuError>
    where
        F: FnOnce(&mut CudaSequence<'_>) -> Result<(), GpuError>,
    {
        Ok(self.probe(Mode::Probe, streams, f)?.hasher.finish())
    }

    fn update<F>(&self, graph: &mut CudaGraph, f: F) -> Result<(), GpuError>
    where
        F: FnOnce(&mut CudaSequence<'_>) -> Result<(), GpuError>,
    {
        graph.update(f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fault_codes_are_classified() {
        assert!(is_fault(700) && is_fault(719) && is_fault(710));
        for sticky in [214, 220, 226, 702, 709, 714, 715, 716, 717, 718, 721, 999] {
            assert!(is_fault(sticky), "{sticky}");
        }
        assert!(!is_fault(1) && !is_fault(600) && !is_fault(0) && !is_fault(101) && !is_fault(222));
        assert!(matches!(
            classify("k", 700, "X".into()),
            GpuError::Fault { code: 700, .. }
        ));
        assert!(matches!(
            classify("k", 1, "X".into()),
            GpuError::Call { code: 1, .. }
        ));
    }

    #[test]
    fn only_faults_poison_by_default_but_an_unrecoverable_failure_always_does() {
        let poison = Poison::new();
        note_fault(&poison, &classify("k", 101, "INVALID_DEVICE".into()));
        assert_eq!(
            poison.get(),
            None,
            "an ordinary error leaves the device usable"
        );
        note_fault(&poison, &classify("k", 700, "ILLEGAL_ADDRESS".into()));
        assert!(matches!(
            poison.get(),
            Some(GpuError::Fault { code: 700, .. })
        ));
        let other = Poison::new();
        note_unrecoverable(
            &other,
            &classify("cuEventQuery", 101, "INVALID_DEVICE".into()),
        );
        assert!(matches!(
            other.get(),
            Some(GpuError::Call { code: 101, .. })
        ));
    }
}
