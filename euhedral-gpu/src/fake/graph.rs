//! Fake graphs (Task 7).

use std::marker::PhantomData;

use crate::{Fingerprint, GpuError, Graph, Sequence};

use super::{FakeEnqueue, FakeGpu, FakeKernel, FakeStream};

/// A fake recording.
#[derive(Debug)]
pub struct FakeGraph {
    fingerprint: Fingerprint,
}

impl Graph for FakeGraph {
    type Gpu = FakeGpu;

    fn fingerprint(&self) -> Fingerprint {
        self.fingerprint
    }

    fn launches(&self) -> usize {
        0
    }

    fn replay(&self, _stream: &FakeStream) -> Result<(), GpuError> {
        Err(GpuError::Invalid("not yet".into()))
    }
}

/// A pass over a recurring sequence.
#[derive(Debug)]
pub struct FakeSequence<'a> {
    _marker: PhantomData<&'a ()>,
}

impl Sequence for FakeSequence<'_> {
    type Kernel = FakeKernel;
    type Enqueue<'a>
        = FakeEnqueue<'a>
    where
        Self: 'a;

    fn submit(
        &mut self,
        _stream: usize,
        _f: impl FnOnce(&mut FakeEnqueue<'_>) -> Result<(), GpuError>,
    ) -> Result<(), GpuError> {
        Err(GpuError::Invalid("not yet".into()))
    }

    fn order(&mut self, _after: usize, _before: usize) -> Result<(), GpuError> {
        Err(GpuError::Invalid("not yet".into()))
    }
}
