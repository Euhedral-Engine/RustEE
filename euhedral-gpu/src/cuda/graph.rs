//! CUDA graphs (stubs until the graph task).

use std::marker::PhantomData;

use crate::{Fingerprint, GpuError, Graph, Sequence};

use super::stream::{CudaEnqueue, CudaStream};
use super::{CudaGpu, CudaKernel};

/// A recording.
#[derive(Debug)]
pub struct CudaGraph {
    fingerprint: Fingerprint,
}

impl Graph for CudaGraph {
    type Gpu = CudaGpu;

    fn fingerprint(&self) -> Fingerprint {
        self.fingerprint
    }

    fn launches(&self) -> usize {
        0
    }

    fn replay(&self, _stream: &CudaStream) -> Result<(), GpuError> {
        Err(GpuError::Invalid("not yet".into()))
    }
}

/// A pass over a recurring sequence.
#[derive(Debug)]
pub struct CudaSequence<'a> {
    _marker: PhantomData<&'a ()>,
}

impl Sequence for CudaSequence<'_> {
    type Kernel = CudaKernel;
    type Enqueue<'a> = CudaEnqueue<'a>;

    fn submit(
        &mut self,
        _stream: usize,
        _f: impl FnOnce(&mut CudaEnqueue<'_>) -> Result<(), GpuError>,
    ) -> Result<(), GpuError> {
        Err(GpuError::Invalid("not yet".into()))
    }

    fn order(&mut self, _after: usize, _before: usize) -> Result<(), GpuError> {
        Err(GpuError::Invalid("not yet".into()))
    }
}
