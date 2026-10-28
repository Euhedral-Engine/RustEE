//! The CUDA backend (feature `cuda`).
//!
//! It takes `euhedral-memory`'s shared [`Driver`] and [`Device`], so there is one loaded driver and one primary
//! context, and looks up its extra entry points ([`fns::GpuFns`]) in the same library.

pub mod fns;
mod graph;
mod module;
mod stream;

pub use graph::{CudaGraph, CudaSequence};
pub use module::{CudaKernel, CudaModule};
pub use stream::{CudaEnqueue, CudaStream};

use std::sync::Arc;

use euhedral_memory::cuda::{CudaError, Device, Driver};

use euhedral_graph::LanePool;

use crate::{Fingerprint, Gpu, GpuError, Poison, Rerecorded, StreamOptions};

use fns::{ATTR_CC_MAJOR, ATTR_CC_MINOR, GpuFns};

/// Device faults: the context is lost and the device is poisoned. (Illegal address, launch failure, misaligned and
/// other device-side exceptions, hardware stack and instruction errors.)
pub(crate) fn is_fault(code: i32) -> bool {
    matches!(code, 700 | 702 | 710 | 714..=719)
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
                if matches!(err, GpuError::Fault { .. }) {
                    self.poison.set(err.clone());
                }
                Err(err)
            }
        }
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
        let queues = streams
            .iter()
            .map(|s| s.clone() as Arc<dyn euhedral_graph::DeviceQueue>)
            .collect();
        Ok((Arc::new(LanePool::new(queues)), streams))
    }

    fn poisoned(&self) -> Option<GpuError> {
        self.poison.get()
    }

    fn record<F>(&self, _streams: &[&Arc<CudaStream>], _f: F) -> Result<CudaGraph, GpuError>
    where
        F: FnOnce(&mut CudaSequence<'_>) -> Result<(), GpuError>,
    {
        Err(GpuError::Invalid("not yet".into()))
    }

    fn rerecord<F>(
        &self,
        _graph: &mut CudaGraph,
        _streams: &[&Arc<CudaStream>],
        _f: F,
    ) -> Result<Rerecorded, GpuError>
    where
        F: FnOnce(&mut CudaSequence<'_>) -> Result<(), GpuError>,
    {
        Err(GpuError::Invalid("not yet".into()))
    }

    fn fingerprint<F>(&self, _streams: usize, _f: F) -> Result<Fingerprint, GpuError>
    where
        F: FnOnce(&mut CudaSequence<'_>) -> Result<(), GpuError>,
    {
        Err(GpuError::Invalid("not yet".into()))
    }

    fn update<F>(&self, _graph: &mut CudaGraph, _f: F) -> Result<(), GpuError>
    where
        F: FnOnce(&mut CudaSequence<'_>) -> Result<(), GpuError>,
    {
        Err(GpuError::Invalid("not yet".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fault_codes_are_classified() {
        assert!(is_fault(700) && is_fault(719) && is_fault(710));
        assert!(!is_fault(1) && !is_fault(600) && !is_fault(0));
        assert!(matches!(
            classify("k", 700, "X".into()),
            GpuError::Fault { code: 700, .. }
        ));
        assert!(matches!(
            classify("k", 1, "X".into()),
            GpuError::Call { code: 1, .. }
        ));
    }
}
