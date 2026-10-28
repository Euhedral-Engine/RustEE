//! CUDA modules and kernels (stubs until the module task).

use std::sync::Arc;

use crate::{CompileOptions, Gpu, GpuError, Kernel, Module};

use super::CudaGpu;

/// A loaded module.
#[derive(Debug)]
pub struct CudaModule {
    _gpu: Arc<CudaGpu>,
}

/// A kernel of a module.
#[derive(Clone, Debug)]
pub struct CudaKernel {
    name: Arc<str>,
    id: u64,
}

impl Kernel for CudaKernel {
    fn name(&self) -> &str {
        &self.name
    }

    fn id(&self) -> u64 {
        self.id
    }
}

impl Module for CudaModule {
    type Gpu = CudaGpu;

    fn from_source(
        _gpu: &Arc<CudaGpu>,
        _source: &str,
        _options: &CompileOptions,
    ) -> Result<Self, GpuError> {
        Err(GpuError::Invalid("not yet".into()))
    }

    fn from_image(_gpu: &Arc<CudaGpu>, _image: &[u8]) -> Result<Self, GpuError> {
        Err(GpuError::Invalid("not yet".into()))
    }

    fn kernel(&self, _name: &str) -> Result<<CudaGpu as Gpu>::Kernel, GpuError> {
        Err(GpuError::Invalid("not yet".into()))
    }
}
