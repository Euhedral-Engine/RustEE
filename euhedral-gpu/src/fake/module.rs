//! Fake modules and kernels.

use std::fmt;
use std::sync::Arc;

use euhedral_hashing::hasher_api::{hash_bytes, hash_bytes_with_seed};

use crate::{CompileOptions, Gpu, GpuError, Kernel, Module};

use super::{FakeGpu, FakeKernelFn};

/// A fake module: its kernels resolve by name in the device's registry.
#[derive(Debug)]
pub struct FakeModule {
    gpu: Arc<FakeGpu>,
    id: u64,
}

/// A fake kernel.
#[derive(Clone)]
pub struct FakeKernel {
    name: Arc<str>,
    id: u64,
    pub(crate) f: FakeKernelFn,
}

impl fmt::Debug for FakeKernel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FakeKernel")
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

impl Kernel for FakeKernel {
    fn name(&self) -> &str {
        &self.name
    }

    fn id(&self) -> u64 {
        self.id
    }
}

impl Module for FakeModule {
    type Gpu = FakeGpu;

    fn from_source(
        gpu: &Arc<FakeGpu>,
        source: &str,
        _options: &CompileOptions,
    ) -> Result<Self, GpuError> {
        Ok(Self {
            gpu: gpu.clone(),
            id: hash_bytes(source.as_bytes()),
        })
    }

    fn from_image(gpu: &Arc<FakeGpu>, image: &[u8]) -> Result<Self, GpuError> {
        Ok(Self {
            gpu: gpu.clone(),
            id: hash_bytes(image),
        })
    }

    fn kernel(&self, name: &str) -> Result<<FakeGpu as Gpu>::Kernel, GpuError> {
        let f = self
            .gpu
            .kernel_fn(name)
            .ok_or_else(|| GpuError::NotFound(name.to_owned()))?;
        Ok(FakeKernel {
            name: name.into(),
            id: hash_bytes_with_seed(name.as_bytes(), self.id),
            f,
        })
    }
}
