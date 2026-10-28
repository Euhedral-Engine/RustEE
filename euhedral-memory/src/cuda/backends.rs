//! The CUDA allocation backends.

use std::ffi::c_void;
use std::os::fd::{FromRawFd, OwnedFd};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use euhedral_data_structures::queues::{Consumer, MpmcQueue, Multi, Producer};

use super::{
    AccessDesc, AllocationProp, CuDevicePtr, CuMemHandle, CuMemPool, CuStream, CudaError, Device,
    MEM_ACCESS_FLAGS_READWRITE, MEM_ALLOC_GRANULARITY_MINIMUM, MEM_ALLOCATION_TYPE_PINNED,
    MEM_HANDLE_TYPE_NONE, MEM_HANDLE_TYPE_POSIX_FD, MEM_LOCATION_TYPE_DEVICE,
    MEMPOOL_ATTR_RELEASE_THRESHOLD, MemLocation, STREAM_NON_BLOCKING,
};
use crate::backend::{AllocError, Backend, Block};
use crate::planner::LocationId;

fn err(e: CudaError) -> AllocError {
    AllocError {
        what: e.to_string(),
    }
}

/// `cuMemAlloc` / `cuMemFree`. The free synchronizes the device: run it on a reaper.
#[derive(Debug)]
pub struct DeviceMemory {
    pub device: Arc<Device>,
    pub location: LocationId,
}

impl Backend for DeviceMemory {
    fn location(&self) -> LocationId {
        self.location
    }

    fn granted(&self, bytes: u64) -> u64 {
        // cuMemAlloc rounds to 2 MiB pages for large requests and sub-allocates small ones; account what was asked
        // rounded to 512 bytes (its alignment).
        bytes.next_multiple_of(512)
    }

    fn alloc(&self, bytes: u64) -> Result<Block, AllocError> {
        let d = &self.device;
        let _c = d.enter().map_err(err)?;
        let mut ptr = 0;
        // SAFETY: out pointer; the context is current.
        d.check("cuMemAlloc", unsafe {
            (d.driver.fns.mem_alloc)(&mut ptr, bytes as usize)
        })
        .map_err(err)?;
        Ok(Block {
            addr: ptr,
            bytes,
            handle: 0,
        })
    }

    unsafe fn free(&self, block: Block) {
        let d = &self.device;
        if let Ok(_c) = d.enter() {
            // SAFETY: the caller's contract (a block of this backend, unused).
            unsafe { (d.driver.fns.mem_free)(block.addr) };
        }
    }

    fn free_blocks(&self) -> bool {
        true
    }
}

/// Stream-ordered allocation from the device's default pool. Frees are queued on the stream (they never wait);
/// the pool keeps freed memory up to its release threshold, so a steady working set is reused without the driver.
#[derive(Debug)]
pub struct StreamPool {
    pub device: Arc<Device>,
    pub location: LocationId,
    pub stream: CuStream,
    pool: CuMemPool,
}

// SAFETY: the stream and pool handles are process-wide; calls on them are thread-safe per the driver.
unsafe impl Send for StreamPool {}
// SAFETY: as above.
unsafe impl Sync for StreamPool {}

impl StreamPool {
    /// A pool on its own non-blocking stream, keeping up to `keep` bytes of freed memory.
    pub fn new(device: Arc<Device>, location: LocationId, keep: u64) -> Result<Self, CudaError> {
        let d = &device;
        let _c = d.enter()?;
        let mut stream = std::ptr::null_mut();
        // SAFETY: out pointer; the context is current.
        d.check("cuStreamCreate", unsafe {
            (d.driver.fns.stream_create)(&mut stream, STREAM_NON_BLOCKING)
        })?;
        let mut pool = std::ptr::null_mut();
        // SAFETY: out pointer.
        d.check("cuDeviceGetDefaultMemPool", unsafe {
            (d.driver.fns.device_get_default_mem_pool)(&mut pool, d.ordinal)
        })?;
        let mut threshold = keep;
        // SAFETY: the attribute takes a cuuint64_t.
        d.check("cuMemPoolSetAttribute", unsafe {
            (d.driver.fns.mem_pool_set_attribute)(
                pool,
                MEMPOOL_ATTR_RELEASE_THRESHOLD,
                (&raw mut threshold).cast(),
            )
        })?;
        drop(_c);
        Ok(StreamPool {
            device,
            location,
            stream,
            pool,
        })
    }

    /// Gives pool memory above `keep` bytes back to the device.
    pub fn trim(&self, keep: u64) -> Result<(), CudaError> {
        let d = &self.device;
        let _c = d.enter()?;
        // SAFETY: the device's pool.
        d.check("cuMemPoolTrimTo", unsafe {
            (d.driver.fns.mem_pool_trim_to)(self.pool, keep as usize)
        })
    }
}

impl Backend for StreamPool {
    fn location(&self) -> LocationId {
        self.location
    }

    fn granted(&self, bytes: u64) -> u64 {
        bytes.next_multiple_of(512)
    }

    fn alloc(&self, bytes: u64) -> Result<Block, AllocError> {
        let d = &self.device;
        let _c = d.enter().map_err(err)?;
        let mut ptr = 0;
        // SAFETY: out pointer; our stream.
        d.check("cuMemAllocAsync", unsafe {
            (d.driver.fns.mem_alloc_async)(&mut ptr, bytes as usize, self.stream)
        })
        .map_err(err)?;
        Ok(Block {
            addr: ptr,
            bytes,
            handle: 0,
        })
    }

    unsafe fn free(&self, block: Block) {
        let d = &self.device;
        if let Ok(_c) = d.enter() {
            // SAFETY: the caller's contract; the free is ordered on the stream.
            unsafe { (d.driver.fns.mem_free_async)(block.addr, self.stream) };
        }
    }

    fn free_blocks(&self) -> bool {
        false
    }
}

impl Drop for StreamPool {
    fn drop(&mut self) {
        let d = &self.device;
        if let Ok(_c) = d.enter() {
            // SAFETY: our stream; synchronized so its queued frees ran.
            unsafe {
                (d.driver.fns.stream_synchronize)(self.stream);
                (d.driver.fns.stream_destroy)(self.stream);
            }
        }
    }
}

/// A VMM allocation: physical memory (`cuMemCreate`) mapped at a reserved address, optionally exportable as a POSIX
/// fd for Vulkan (`VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VmmBlock {
    pub ptr: CuDevicePtr,
    /// Bytes mapped: the request rounded up to the granularity.
    pub size: u64,
    pub handle: CuMemHandle,
}

/// `cuMemCreate` + `cuMemAddressReserve` + `cuMemMap` + `cuMemSetAccess`; the free is the reverse.
#[derive(Debug)]
pub struct Vmm {
    pub device: Arc<Device>,
    pub location: LocationId,
    pub exportable: bool,
    pub granularity: u64,
}

impl Vmm {
    pub fn new(
        device: Arc<Device>,
        location: LocationId,
        exportable: bool,
    ) -> Result<Self, CudaError> {
        let prop = Self::prop(&device, exportable);
        let _c = device.enter()?;
        let mut g = 0usize;
        // SAFETY: out pointer, a valid property.
        device.check("cuMemGetAllocationGranularity", unsafe {
            (device.driver.fns.mem_get_allocation_granularity)(
                &mut g,
                &prop,
                MEM_ALLOC_GRANULARITY_MINIMUM,
            )
        })?;
        drop(_c);
        Ok(Vmm {
            device,
            location,
            exportable,
            granularity: g as u64,
        })
    }

    fn prop(device: &Device, exportable: bool) -> AllocationProp {
        AllocationProp {
            kind: MEM_ALLOCATION_TYPE_PINNED,
            requested_handle_types: if exportable {
                MEM_HANDLE_TYPE_POSIX_FD
            } else {
                MEM_HANDLE_TYPE_NONE
            },
            location: MemLocation {
                kind: MEM_LOCATION_TYPE_DEVICE,
                id: device.ordinal,
            },
            win32_handle_meta_data: std::ptr::null_mut(),
            compression_type: 0,
            gpu_direct_rdma_capable: 0,
            usage: 0,
            reserved: [0; 4],
        }
    }

    pub fn alloc_vmm(&self, bytes: u64) -> Result<VmmBlock, CudaError> {
        let d = &self.device;
        let f = &d.driver.fns;
        let size = bytes.max(1).next_multiple_of(self.granularity) as usize;
        let prop = Self::prop(d, self.exportable);
        let _c = d.enter()?;
        let mut handle = 0;
        // SAFETY: out pointer, a valid property, flags 0.
        d.check("cuMemCreate", unsafe {
            (f.mem_create)(&mut handle, size, &prop, 0)
        })?;
        let mut ptr = 0;
        // SAFETY: out pointer; a reservation of `size` at the granularity.
        if let Err(e) = d.check("cuMemAddressReserve", unsafe {
            (f.mem_address_reserve)(&mut ptr, size, self.granularity as usize, 0, 0)
        }) {
            // SAFETY: the handle just created.
            unsafe { (f.mem_release)(handle) };
            return Err(e);
        }
        // SAFETY: the reserved range and the handle, same size.
        let mapped = d
            .check("cuMemMap", unsafe { (f.mem_map)(ptr, size, 0, handle, 0) })
            .and_then(|()| {
                let access = AccessDesc {
                    location: MemLocation {
                        kind: MEM_LOCATION_TYPE_DEVICE,
                        id: d.ordinal,
                    },
                    flags: MEM_ACCESS_FLAGS_READWRITE,
                };
                // SAFETY: the mapped range; one descriptor.
                d.check("cuMemSetAccess", unsafe {
                    (f.mem_set_access)(ptr, size, &access, 1)
                })
            });
        if let Err(e) = mapped {
            // SAFETY: undo the reservation and the handle (an unmap of an unmapped range fails harmlessly).
            unsafe {
                (f.mem_unmap)(ptr, size);
                (f.mem_address_free)(ptr, size);
                (f.mem_release)(handle);
            }
            return Err(e);
        }
        Ok(VmmBlock {
            ptr,
            size: size as u64,
            handle,
        })
    }

    /// A new POSIX fd for the allocation (the caller owns it; Vulkan's import takes it).
    pub fn export_fd(&self, block: &VmmBlock) -> Result<OwnedFd, CudaError> {
        let d = &self.device;
        let _c = d.enter()?;
        let mut fd: i32 = -1;
        // SAFETY: an int out pointer for a POSIX fd handle.
        d.check("cuMemExportToShareableHandle", unsafe {
            (d.driver.fns.mem_export_to_shareable_handle)(
                (&raw mut fd).cast::<c_void>(),
                block.handle,
                MEM_HANDLE_TYPE_POSIX_FD,
                0,
            )
        })?;
        // SAFETY: a new fd from the driver, owned by the caller.
        Ok(unsafe { OwnedFd::from_raw_fd(fd) })
    }

    /// # Safety
    /// A block of this backend, freed once, unused by host and device.
    pub unsafe fn free_vmm(&self, block: VmmBlock) {
        let d = &self.device;
        let f = &d.driver.fns;
        if let Ok(_c) = d.enter() {
            // SAFETY: the caller's contract; the reverse of `alloc_vmm`.
            unsafe {
                (f.mem_unmap)(block.ptr, block.size as usize);
                (f.mem_address_free)(block.ptr, block.size as usize);
                (f.mem_release)(block.handle);
            }
        }
    }
}

impl Backend for Vmm {
    fn location(&self) -> LocationId {
        self.location
    }

    fn granted(&self, bytes: u64) -> u64 {
        bytes.max(1).next_multiple_of(self.granularity)
    }

    fn alloc(&self, bytes: u64) -> Result<Block, AllocError> {
        let b = self.alloc_vmm(bytes).map_err(err)?;
        Ok(Block {
            addr: b.ptr,
            bytes: b.size,
            handle: b.handle,
        })
    }

    unsafe fn free(&self, block: Block) {
        // SAFETY: the caller's contract.
        unsafe {
            self.free_vmm(VmmBlock {
                ptr: block.addr,
                size: block.bytes,
                handle: block.handle,
            })
        };
    }

    fn free_blocks(&self) -> bool {
        true
    }
}

/// Pinned host memory (`cuMemHostAlloc`), with released buffers kept by power-of-two size class up to `keep` bytes:
/// a pinned free synchronizes the device (3 ms per 64 MiB idle; it waits for all queued CUDA work), so it happens
/// only on [`PinnedHost::trim`]. Lock-free: one queue per size class.
pub struct PinnedHost {
    pub device: Arc<Device>,
    pub location: LocationId,
    keep: u64,
    classes: Vec<(Producer<u64, Multi>, Consumer<u64, Multi>)>,
    cached: AtomicU64,
    overflow: Ends<(u64, u64)>,
}

impl std::fmt::Debug for PinnedHost {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PinnedHost")
            .field("keep", &self.keep)
            .field("cached", &self.cached())
            .finish()
    }
}

/// A lock-free queue's two ends.
type Ends<T> = (Producer<T, Multi>, Consumer<T, Multi>);

/// The smallest pinned size class (64 KiB).
pub const PINNED_MIN_CLASS: u64 = 64 << 10;
const CLASSES: usize = 48;

impl PinnedHost {
    pub fn new(device: Arc<Device>, location: LocationId, keep: u64) -> Self {
        let classes = (0..CLASSES)
            .map(|_| {
                let q = MpmcQueue::<u64>::new(16);
                (q.producer(), q.consumer())
            })
            .collect();
        let o = MpmcQueue::new(16);
        PinnedHost {
            device,
            location,
            keep,
            classes,
            cached: AtomicU64::new(0),
            overflow: (o.producer(), o.consumer()),
        }
    }

    pub fn class_of(bytes: u64) -> u64 {
        bytes.max(PINNED_MIN_CLASS).next_power_of_two()
    }

    fn index(class: u64) -> usize {
        (class.trailing_zeros() - PINNED_MIN_CLASS.trailing_zeros()) as usize
    }

    /// Bytes of released buffers kept for reuse.
    pub fn cached(&self) -> u64 {
        self.cached.load(Ordering::Acquire)
    }

    /// Frees the released buffers beyond the limit (and all of them with `everything`). Synchronizes the device.
    pub fn trim(&self, everything: bool) {
        let mut gone = Vec::new();
        self.overflow.1.drain_all(|(a, _)| gone.push(a));
        if everything {
            for (i, (_, c)) in self.classes.iter().enumerate() {
                let class = PINNED_MIN_CLASS << i;
                c.drain_all(|a| {
                    self.cached.fetch_sub(class, Ordering::AcqRel);
                    gone.push(a);
                });
            }
        }
        let d = &self.device;
        if let Ok(_c) = d.enter() {
            for a in gone {
                // SAFETY: released buffers of this backend, unused.
                unsafe { (d.driver.fns.mem_free_host)(a as *mut c_void) };
            }
        }
    }
}

impl Backend for PinnedHost {
    fn location(&self) -> LocationId {
        self.location
    }

    fn granted(&self, bytes: u64) -> u64 {
        Self::class_of(bytes)
    }

    fn alloc(&self, bytes: u64) -> Result<Block, AllocError> {
        let class = Self::class_of(bytes);
        if let Some(addr) = self
            .classes
            .get(Self::index(class))
            .and_then(|(_, c)| c.poll())
        {
            self.cached.fetch_sub(class, Ordering::AcqRel);
            return Ok(Block {
                addr,
                bytes: class,
                handle: 0,
            });
        }
        let d = &self.device;
        let _c = d.enter().map_err(err)?;
        let mut p = std::ptr::null_mut();
        // SAFETY: out pointer; flags 0 (portable to this context).
        d.check("cuMemHostAlloc", unsafe {
            (d.driver.fns.mem_host_alloc)(&mut p, class as usize, 0)
        })
        .map_err(err)?;
        Ok(Block {
            addr: p as u64,
            bytes: class,
            handle: 0,
        })
    }

    unsafe fn free(&self, block: Block) {
        let kept = self
            .cached
            .try_update(Ordering::AcqRel, Ordering::Acquire, |c| {
                (c + block.bytes <= self.keep).then_some(c + block.bytes)
            })
            .is_ok();
        let i = Self::index(block.bytes);
        if kept && i < CLASSES && self.classes[i].0.offer(block.addr).is_ok() {
            return;
        }
        if kept {
            self.cached.fetch_sub(block.bytes, Ordering::AcqRel);
        }
        let _ = self.overflow.0.offer((block.addr, block.bytes));
    }

    fn free_blocks(&self) -> bool {
        // A release only caches; `trim` is the blocking part.
        false
    }
}

impl Drop for PinnedHost {
    fn drop(&mut self) {
        self.trim(true);
    }
}
