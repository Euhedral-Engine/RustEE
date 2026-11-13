//! Fake modules and kernels.

use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use euhedral_hashing::hasher_api::{hash_bytes, hash_bytes_with_seed};

use crate::api::over_shared_limit;
use crate::{CompileOptions, Gpu, GpuError, Kernel, KernelOptions, Launch, Module};

use super::{FakeGpu, FakeKernelFn};

/// The dynamic shared memory, in bytes, a fake kernel allows a launch without an opt-in: what the fake device
/// models, not a fact about any real one.
pub const DEFAULT_DYNAMIC_SHARED_BYTES: u32 = 48 * 1024;

/// A fake module: its kernels resolve by name in the device's registry.
#[derive(Debug)]
pub struct FakeModule {
    gpu: Arc<FakeGpu>,
    id: u64,
    /// The dynamic shared limit of each kernel looked up so far, shared by every handle of it.
    limits: Mutex<HashMap<String, Arc<AtomicU32>>>,
}

/// A fake kernel.
#[derive(Clone)]
pub struct FakeKernel {
    name: Arc<str>,
    id: u64,
    limit: Arc<AtomicU32>,
    pub(crate) f: FakeKernelFn,
}

impl FakeKernel {
    /// The most dynamic shared memory, in bytes, a launch of this kernel may ask for: the default, or the largest
    /// value any lookup of the kernel in its module opted in to.
    pub fn max_dynamic_shared_bytes(&self) -> u32 {
        self.limit.load(Ordering::Relaxed)
    }

    /// `Invalid` if `launch` asks for more dynamic shared memory than the kernel allows.
    pub(crate) fn check_shared(&self, launch: &Launch) -> Result<(), GpuError> {
        let limit = self.max_dynamic_shared_bytes();
        if launch.shared_bytes > limit {
            return Err(over_shared_limit(&self.name, launch.shared_bytes, limit));
        }
        Ok(())
    }
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
            limits: Mutex::default(),
        })
    }

    fn from_image(gpu: &Arc<FakeGpu>, image: &[u8]) -> Result<Self, GpuError> {
        Ok(Self {
            gpu: gpu.clone(),
            id: hash_bytes(image),
            limits: Mutex::default(),
        })
    }

    fn kernel_with(
        &self,
        name: &str,
        options: KernelOptions,
    ) -> Result<<FakeGpu as Gpu>::Kernel, GpuError> {
        let f = self
            .gpu
            .kernel_fn(name)
            .ok_or_else(|| GpuError::NotFound(name.to_owned()))?;
        let limit = Arc::clone(
            self.limits
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .entry(name.to_owned())
                .or_insert_with(|| Arc::new(AtomicU32::new(DEFAULT_DYNAMIC_SHARED_BYTES))),
        );
        limit.fetch_max(options.max_dynamic_shared_bytes, Ordering::Relaxed);
        Ok(FakeKernel {
            name: name.into(),
            id: hash_bytes_with_seed(name.as_bytes(), self.id),
            limit,
            f,
        })
    }
}
