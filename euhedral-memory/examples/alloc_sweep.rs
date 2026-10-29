//! Allocation sweep: CUDA against Vulkan on this GPU, weighted to small sizes.
//!
//! `cargo run --release -p euhedral-memory --features cuda --example alloc_sweep [-- --quick]`
//!
//! Prints CSV tables: per method and size, alloc and free latency p50/p99 (µs) for pairs (alloc then free at
//! once: churn) and batches (K live, then all freed); the import of a CUDA VMM allocation into Vulkan; copy
//! throughput; and whether a free or an allocation waits for unrelated busy device work.
#![allow(
    clippy::disallowed_methods,
    reason = "test fixtures: the test thread runs peer threads and waits on them"
)]

use std::ffi::CStr;
use std::os::fd::IntoRawFd;
use std::time::Instant;

use ash::vk;
use euhedral_memory::cuda::{CuStream, Device, Driver, PinnedHost, StreamPool, Vmm};
use euhedral_memory::{Backend, LocationId};

const KIB: u64 = 1 << 10;
const MIB: u64 = 1 << 20;

fn sizes() -> Vec<(String, u64)> {
    vec![
        ("256B".into(), 256),
        ("4KiB".into(), 4 * KIB),
        ("64KiB".into(), 64 * KIB),
        ("1MiB".into(), MIB),
        ("16MiB".into(), 16 * MIB),
        ("68MiB".into(), 68 * MIB),
        ("74MiB".into(), 74 * MIB),
        ("96MiB".into(), 96 * MIB),
    ]
}

fn iters(bytes: u64, quick: bool) -> (usize, usize) {
    // (pairs, batch size)
    let (p, b) = match bytes {
        b if b <= 64 * KIB => (2000, 256),
        b if b <= MIB => (500, 128),
        b if b <= 16 * MIB => (100, 32),
        _ => (30, 12),
    };
    if quick { (p / 5, b / 2) } else { (p, b) }
}

fn pct(v: &mut [f64], p: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[((v.len() - 1) as f64 * p).round() as usize]
}

struct Stats {
    alloc: Vec<f64>,
    free: Vec<f64>,
}

impl Stats {
    fn new() -> Self {
        Stats {
            alloc: Vec::new(),
            free: Vec::new(),
        }
    }
    fn row(&mut self, method: &str, size: &str, mode: &str) {
        println!(
            "{method},{size},{mode},{:.1},{:.1},{:.1},{:.1}",
            pct(&mut self.alloc, 0.5),
            pct(&mut self.alloc, 0.99),
            pct(&mut self.free, 0.5),
            pct(&mut self.free, 0.99)
        );
    }
}

fn us(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e6
}

/// Runs pairs and a batch of an (alloc, free) method.
fn sweep<T>(
    method: &str,
    quick: bool,
    mut alloc: impl FnMut(u64) -> Option<T>,
    mut free: impl FnMut(T),
) {
    for (name, bytes) in sizes() {
        let (pairs, batch) = iters(bytes, quick);
        for _ in 0..3 {
            if let Some(x) = alloc(bytes) {
                free(x);
            }
        }
        let mut s = Stats::new();
        for _ in 0..pairs {
            let t = Instant::now();
            let Some(x) = alloc(bytes) else { break };
            s.alloc.push(us(t));
            let t = Instant::now();
            free(x);
            s.free.push(us(t));
        }
        s.row(method, &name, "pair");
        let mut s = Stats::new();
        let mut live = Vec::new();
        for _ in 0..batch {
            let t = Instant::now();
            let Some(x) = alloc(bytes) else { break };
            s.alloc.push(us(t));
            live.push(x);
        }
        for x in live {
            let t = Instant::now();
            free(x);
            s.free.push(us(t));
        }
        s.row(method, &name, &format!("batch{batch}"));
    }
}

// ----------------------------------------------------------------------------------------------------------------
// Vulkan

struct Vk {
    _entry: ash::Entry,
    instance: ash::Instance,
    device: ash::Device,
    fd: ash::khr::external_memory_fd::Device,
    queue: vk::Queue,
    family: u32,
    props: vk::PhysicalDeviceMemoryProperties,
}

impl Vk {
    fn new() -> Vk {
        // SAFETY: loading the system Vulkan loader.
        let entry = unsafe { ash::Entry::load() }.expect("vulkan loader");
        let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_3);
        let info = vk::InstanceCreateInfo::default().application_info(&app);
        // SAFETY: valid create info.
        let instance = unsafe { entry.create_instance(&info, None) }.expect("instance");
        // SAFETY: a live instance.
        let pdevs = unsafe { instance.enumerate_physical_devices() }.unwrap();
        let pdev = *pdevs
            .iter()
            .find(|&&p| {
                // SAFETY: a device of this instance.
                let pr = unsafe { instance.get_physical_device_properties(p) };
                pr.vendor_id == 0x10de
            })
            .expect("an NVIDIA device");
        // SAFETY: as above.
        let families = unsafe { instance.get_physical_device_queue_family_properties(pdev) };
        let family = families
            .iter()
            .position(|f| f.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .unwrap() as u32;
        let prio = [1.0f32];
        let qinfo = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(family)
            .queue_priorities(&prio)];
        let exts = [ash::khr::external_memory_fd::NAME.as_ptr()];
        let dinfo = vk::DeviceCreateInfo::default()
            .queue_create_infos(&qinfo)
            .enabled_extension_names(&exts);
        // SAFETY: valid create info.
        let device = unsafe { instance.create_device(pdev, &dinfo, None) }.expect("device");
        // SAFETY: the queue we asked for.
        let queue = unsafe { device.get_device_queue(family, 0) };
        let fd = ash::khr::external_memory_fd::Device::new(&instance, &device);
        // SAFETY: as above.
        let props = unsafe { instance.get_physical_device_memory_properties(pdev) };
        // SAFETY: as above.
        let p = unsafe { instance.get_physical_device_properties(pdev) };
        // SAFETY: a NUL-terminated name from the driver.
        let name = unsafe { CStr::from_ptr(p.device_name.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        eprintln!("vulkan: {name}");
        Vk {
            _entry: entry,
            instance,
            device,
            fd,
            queue,
            family,
            props,
        }
    }

    fn mem_type(&self, bits: u32, flags: vk::MemoryPropertyFlags) -> u32 {
        (0..self.props.memory_type_count)
            .find(|&i| {
                bits & (1 << i) != 0
                    && self.props.memory_types[i as usize]
                        .property_flags
                        .contains(flags)
            })
            .expect("memory type")
    }

    fn device_type_bits(&self) -> u32 {
        let info = vk::BufferCreateInfo::default()
            .size(4096)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER);
        // SAFETY: valid.
        let b = unsafe { self.device.create_buffer(&info, None) }.unwrap();
        // SAFETY: our buffer.
        let r = unsafe { self.device.get_buffer_memory_requirements(b) };
        // SAFETY: as above.
        unsafe { self.device.destroy_buffer(b, None) };
        r.memory_type_bits
    }
}

struct VkBuf {
    buf: vk::Buffer,
    mem: vk::DeviceMemory,
}

fn vk_buffer(v: &Vk, bytes: u64, flags: vk::MemoryPropertyFlags, export: bool) -> Option<VkBuf> {
    let mut ext = vk::ExternalMemoryBufferCreateInfo::default()
        .handle_types(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD);
    let mut info = vk::BufferCreateInfo::default().size(bytes).usage(
        vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST,
    );
    if export {
        info = info.push_next(&mut ext);
    }
    // SAFETY: valid create info.
    let buf = unsafe { v.device.create_buffer(&info, None) }.ok()?;
    // SAFETY: our buffer.
    let req = unsafe { v.device.get_buffer_memory_requirements(buf) };
    let mut exp = vk::ExportMemoryAllocateInfo::default()
        .handle_types(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD);
    let mut ded = vk::MemoryDedicatedAllocateInfo::default().buffer(buf);
    let mut alloc = vk::MemoryAllocateInfo::default()
        .allocation_size(req.size)
        .memory_type_index(v.mem_type(req.memory_type_bits, flags));
    if export {
        alloc = alloc.push_next(&mut exp).push_next(&mut ded);
    }
    // SAFETY: valid allocation info.
    let mem = unsafe { v.device.allocate_memory(&alloc, None) }.ok()?;
    // SAFETY: fresh memory, bound once.
    unsafe { v.device.bind_buffer_memory(buf, mem, 0) }.ok()?;
    if flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE) {
        // SAFETY: host-visible memory, mapped for its life (unmapped by the free).
        unsafe {
            v.device
                .map_memory(mem, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
        }
        .ok()?;
    }
    Some(VkBuf { buf, mem })
}

fn vk_free(v: &Vk, b: VkBuf) {
    // SAFETY: unused by the device (nothing was submitted on it, or the submission was waited for).
    unsafe {
        v.device.destroy_buffer(b.buf, None);
        v.device.free_memory(b.mem, None);
    }
}

type VkImage = (vk::Image, vk::DeviceMemory, bool);

/// An optimal-tiling R16 image of about `bytes` (a square); on `mem` if given.
fn vk_image(
    v: &Vk,
    bytes: u64,
    external: bool,
    mem: Option<(vk::DeviceMemory, u64)>,
) -> Option<VkImage> {
    let side = ((bytes / 2) as f64).sqrt().max(16.0) as u32;
    let mut ext = vk::ExternalMemoryImageCreateInfo::default()
        .handle_types(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD);
    let mut info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(vk::Format::R16_UNORM)
        .extent(vk::Extent3D {
            width: side,
            height: side,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(
            vk::ImageUsageFlags::STORAGE
                | vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::TRANSFER_DST,
        )
        .initial_layout(vk::ImageLayout::UNDEFINED);
    if external {
        info = info.push_next(&mut ext);
    }
    // SAFETY: valid.
    let img = unsafe { v.device.create_image(&info, None) }.ok()?;
    let mut ded = vk::MemoryDedicatedRequirements::default();
    let mut req2 = vk::MemoryRequirements2::default().push_next(&mut ded);
    let rinfo = vk::ImageMemoryRequirementsInfo2::default().image(img);
    // SAFETY: our image.
    unsafe { v.device.get_image_memory_requirements2(&rinfo, &mut req2) };
    let req = req2.memory_requirements;
    let dedicated = ded.requires_dedicated_allocation == vk::TRUE
        || ded.prefers_dedicated_allocation == vk::TRUE;
    let memory = match mem {
        Some((m, size)) => {
            if size < req.size {
                eprintln!("imported {size} < image needs {}", req.size);
                // SAFETY: unbound.
                unsafe { v.device.destroy_image(img, None) };
                return None;
            }
            m
        }
        None => {
            let alloc = vk::MemoryAllocateInfo::default()
                .allocation_size(req.size)
                .memory_type_index(
                    v.mem_type(req.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL),
                );
            // SAFETY: valid.
            unsafe { v.device.allocate_memory(&alloc, None) }.ok()?
        }
    };
    // SAFETY: memory of at least the image's size.
    unsafe { v.device.bind_image_memory(img, memory, 0) }.ok()?;
    Some((img, memory, dedicated))
}

/// Imports a CUDA VMM allocation (POSIX fd) as Vulkan memory.
fn vk_import(
    v: &Vk,
    fd: std::os::fd::OwnedFd,
    size: u64,
    type_bits: u32,
) -> Option<vk::DeviceMemory> {
    let mut fdp = vk::MemoryFdPropertiesKHR::default();
    let raw = fd.into_raw_fd();
    // SAFETY: a valid OPAQUE_FD handle.
    let _ = unsafe {
        v.fd.get_memory_fd_properties(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD, raw, &mut fdp)
    };
    let bits = if fdp.memory_type_bits != 0 {
        fdp.memory_type_bits & type_bits
    } else {
        type_bits
    };
    let mut imp = vk::ImportMemoryFdInfoKHR::default()
        .handle_type(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD)
        .fd(raw);
    let alloc = vk::MemoryAllocateInfo::default()
        .allocation_size(size)
        .memory_type_index(v.mem_type(bits, vk::MemoryPropertyFlags::DEVICE_LOCAL))
        .push_next(&mut imp);
    // SAFETY: valid import info.
    match unsafe { v.device.allocate_memory(&alloc, None) } {
        Ok(m) => Some(m),
        Err(e) => {
            eprintln!("import failed: {e}");
            // SAFETY: the import failed, so we still own the fd.
            unsafe { libc::close(raw) };
            None
        }
    }
}

fn record_copies(
    v: &Vk,
    src: vk::Buffer,
    dst: vk::Buffer,
    bytes: u64,
    n: u32,
) -> (vk::CommandPool, vk::CommandBuffer) {
    let pool_info = vk::CommandPoolCreateInfo::default().queue_family_index(v.family);
    // SAFETY: valid.
    let pool = unsafe { v.device.create_command_pool(&pool_info, None) }.unwrap();
    let ai = vk::CommandBufferAllocateInfo::default()
        .command_pool(pool)
        .command_buffer_count(1);
    // SAFETY: valid.
    let cb = unsafe { v.device.allocate_command_buffers(&ai) }.unwrap()[0];
    // SAFETY: valid recording.
    unsafe {
        v.device
            .begin_command_buffer(cb, &vk::CommandBufferBeginInfo::default())
            .unwrap();
        for _ in 0..n {
            v.device.cmd_copy_buffer(
                cb,
                src,
                dst,
                &[vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size: bytes,
                }],
            );
        }
        v.device.end_command_buffer(cb).unwrap();
    }
    (pool, cb)
}

fn vk_copy_gbs(v: &Vk, src: vk::Buffer, dst: vk::Buffer, bytes: u64) -> f64 {
    let (pool, cb) = record_copies(v, src, dst, bytes, 1);
    // SAFETY: valid.
    let fence = unsafe { v.device.create_fence(&vk::FenceCreateInfo::default(), None) }.unwrap();
    let mut ms = Vec::new();
    for _ in 0..10 {
        let cbs = [cb];
        let submit = [vk::SubmitInfo::default().command_buffers(&cbs)];
        let t = Instant::now();
        // SAFETY: the recorded buffer; waited for here.
        unsafe {
            v.device.queue_submit(v.queue, &submit, fence).unwrap();
            v.device.wait_for_fences(&[fence], true, u64::MAX).unwrap();
            v.device.reset_fences(&[fence]).unwrap();
        }
        ms.push(us(t) / 1e3);
    }
    // SAFETY: idle.
    unsafe {
        v.device.destroy_fence(fence, None);
        v.device.destroy_command_pool(pool, None);
    }
    bytes as f64 / (pct(&mut ms, 0.5) * 1e-3) / 1e9
}

// ----------------------------------------------------------------------------------------------------------------

fn main() {
    let quick = std::env::args().any(|a| a == "--quick");
    let driver = Driver::load().expect("cuda");
    let dev = Device::open(driver.clone(), 0).expect("device 0");
    let f = &driver.fns;
    let v = Vk::new();
    let loc = LocationId(0);

    println!("# alloc/free latency, µs");
    println!("method,size,mode,alloc_p50,alloc_p99,free_p50,free_p99");
    {
        let _c = dev.enter().unwrap();
        sweep(
            "cu.memalloc",
            quick,
            |b| {
                let mut p = 0;
                // SAFETY: out pointer; context current.
                (unsafe { (f.mem_alloc)(&mut p, b as usize) } == 0).then_some(p)
            },
            |p| {
                // SAFETY: our allocation.
                unsafe { (f.mem_free)(p) };
            },
        );
    }
    {
        let pool = StreamPool::new(dev.clone(), loc, u64::MAX).unwrap();
        let _c = dev.enter().unwrap();
        sweep(
            "cu.async",
            quick,
            |b| {
                let mut p = 0;
                // SAFETY: out pointer; the pool's stream.
                (unsafe { (f.mem_alloc_async)(&mut p, b as usize, pool.stream) } == 0).then_some(p)
            },
            |p| {
                // SAFETY: our allocation; ordered on the stream.
                unsafe { (f.mem_free_async)(p, pool.stream) };
            },
        );
        // SAFETY: our stream.
        unsafe { (f.stream_synchronize)(pool.stream) };
        pool.trim(0).unwrap();
    }
    let vmm = Vmm::new(dev.clone(), loc, false).unwrap();
    println!("# vmm granularity {} KiB", vmm.granularity / KIB);
    sweep(
        "cu.vmm",
        quick,
        |b| vmm.alloc_vmm(b).ok(),
        |x| {
            // SAFETY: unused.
            unsafe { vmm.free_vmm(x) }
        },
    );
    let vmm_fd = Vmm::new(dev.clone(), loc, true).unwrap();
    sweep(
        "cu.vmm+fd",
        quick,
        |b| {
            let x = vmm_fd.alloc_vmm(b).ok()?;
            drop(vmm_fd.export_fd(&x).ok()?);
            Some(x)
        },
        |x| {
            // SAFETY: unused.
            unsafe { vmm_fd.free_vmm(x) }
        },
    );
    {
        let _c = dev.enter().unwrap();
        sweep(
            "cu.hostalloc",
            quick,
            |b| {
                let mut p = std::ptr::null_mut();
                // SAFETY: out pointer.
                (unsafe { (f.mem_host_alloc)(&mut p, b as usize, 0) } == 0).then_some(p as usize)
            },
            |p| {
                // SAFETY: ours.
                unsafe { (f.mem_free_host)(p as *mut _) };
            },
        );
    }
    {
        let pinned = PinnedHost::new(dev.clone(), loc, 4 << 30);
        sweep(
            "cu.pinned.cached",
            quick,
            |b| pinned.alloc(b).ok(),
            |x| {
                // SAFETY: unused.
                unsafe { pinned.free(x) }
            },
        );
    }
    let dl = vk::MemoryPropertyFlags::DEVICE_LOCAL;
    let hv = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
    sweep(
        "vk.buffer.device",
        quick,
        |b| vk_buffer(&v, b, dl, false),
        |x| vk_free(&v, x),
    );
    {
        let t = v.mem_type(v.device_type_bits(), dl);
        sweep(
            "vk.allocmem.device",
            quick,
            |b| {
                let a = vk::MemoryAllocateInfo::default()
                    .allocation_size(b)
                    .memory_type_index(t);
                // SAFETY: valid.
                unsafe { v.device.allocate_memory(&a, None) }.ok()
            },
            |m| {
                // SAFETY: unused.
                unsafe { v.device.free_memory(m, None) }
            },
        );
    }
    sweep(
        "vk.buffer.export.dedicated",
        quick,
        |b| vk_buffer(&v, b, dl, true),
        |x| vk_free(&v, x),
    );
    sweep(
        "vk.buffer.hostmapped",
        quick,
        |b| vk_buffer(&v, b, hv, false),
        |x| vk_free(&v, x),
    );
    sweep(
        "vk.image.optimal",
        quick,
        |b| vk_image(&v, b, false, None),
        |(i, m, _)| {
            // SAFETY: unused.
            unsafe {
                v.device.destroy_image(i, None);
                v.device.free_memory(m, None);
            }
        },
    );
    {
        // Sub-allocation from one block: a size-class free list (host bookkeeping only).
        let lists: std::cell::RefCell<std::collections::HashMap<u64, Vec<u64>>> =
            Default::default();
        let mut top = 0u64;
        sweep(
            "suballoc.freelist",
            quick,
            |b| {
                let class = b.next_power_of_two().max(256);
                let off = lists
                    .borrow_mut()
                    .get_mut(&class)
                    .and_then(Vec::pop)
                    .unwrap_or_else(|| {
                        let o = top;
                        top += class;
                        o
                    });
                Some((class, off))
            },
            |(class, off)| lists.borrow_mut().entry(class).or_default().push(off),
        );
    }

    // ------------------------------------------------------------------------------------------------------------
    println!("# import: CUDA VMM (POSIX fd) into Vulkan, µs p50; optimal-tiling image bound to it");
    println!(
        "size,vmm_alloc+export,vk_import,buffer_bind,all_bound,ext_image_prefers_dedicated,native_image_prefers_dedicated"
    );
    let type_bits = v.device_type_bits();
    for (name, bytes) in sizes() {
        let mut rows = Vec::new();
        let n = if bytes <= MIB { 50 } else { 10 };
        for _ in 0..n {
            let t = Instant::now();
            let x = vmm_fd.alloc_vmm(bytes).unwrap();
            let fd = vmm_fd.export_fd(&x).unwrap();
            let a = us(t);
            let t = Instant::now();
            let Some(mem) = vk_import(&v, fd, x.size, type_bits) else {
                break;
            };
            let i = us(t);
            let t = Instant::now();
            let mut ext = vk::ExternalMemoryBufferCreateInfo::default()
                .handle_types(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD);
            let info = vk::BufferCreateInfo::default()
                .size(bytes)
                .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
                .push_next(&mut ext);
            // SAFETY: valid.
            let buf = unsafe { v.device.create_buffer(&info, None) }.unwrap();
            // SAFETY: imported memory of at least `bytes`.
            let bound = unsafe { v.device.bind_buffer_memory(buf, mem, 0) }.is_ok();
            let bb = us(t);
            // SAFETY: unused.
            unsafe { v.device.destroy_buffer(buf, None) };
            let (img_ok, img_ded) =
                match vk_image(&v, bytes.min(x.size) * 9 / 10, true, Some((mem, x.size))) {
                    Some((i, _, d)) => {
                        // SAFETY: unused.
                        unsafe { v.device.destroy_image(i, None) };
                        (true, d)
                    }
                    None => (false, false),
                };
            let nat_ded = vk_image(&v, bytes, false, None).map(|(i, m, d)| {
                // SAFETY: unused.
                unsafe {
                    v.device.destroy_image(i, None);
                    v.device.free_memory(m, None);
                }
                d
            });
            // SAFETY: unused; then the CUDA side.
            unsafe {
                v.device.free_memory(mem, None);
                vmm_fd.free_vmm(x);
            }
            rows.push((a, i, bb, bound && img_ok, img_ded, nat_ded.unwrap_or(false)));
        }
        let mut av: Vec<f64> = rows.iter().map(|r| r.0).collect();
        let mut iv: Vec<f64> = rows.iter().map(|r| r.1).collect();
        let mut bv: Vec<f64> = rows.iter().map(|r| r.2).collect();
        println!(
            "{name},{:.1},{:.1},{:.1},{},{},{}",
            pct(&mut av, 0.5),
            pct(&mut iv, 0.5),
            pct(&mut bv, 0.5),
            !rows.is_empty() && rows.iter().all(|r| r.3),
            rows.first().is_some_and(|r| r.4),
            rows.first().is_some_and(|r| r.5)
        );
    }

    println!(
        "# the reverse: a Vulkan dedicated exportable buffer, its fd imported into CUDA, µs p50"
    );
    println!("size,vk_alloc_export,vk_get_fd,cu_import_map,cu_unmap_destroy,vk_free");
    for (name, bytes) in sizes() {
        let n = if bytes <= MIB { 50 } else { 10 };
        let mut t5: [Vec<f64>; 5] = Default::default();
        for _ in 0..n {
            let t = Instant::now();
            let b = vk_buffer(&v, bytes, dl, true).unwrap();
            t5[0].push(us(t));
            let t = Instant::now();
            let info = vk::MemoryGetFdInfoKHR::default()
                .memory(b.mem)
                .handle_type(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD);
            // SAFETY: exportable memory.
            let fd = unsafe { v.fd.get_memory_fd(&info) }.unwrap();
            // SAFETY: a new fd we own.
            let fd = unsafe { <std::os::fd::OwnedFd as std::os::fd::FromRawFd>::from_raw_fd(fd) };
            t5[1].push(us(t));
            let t = Instant::now();
            // SAFETY: the buffer's requirements size is what was allocated.
            let req = unsafe { v.device.get_buffer_memory_requirements(b.buf) };
            let imp = dev.import_fd(fd, req.size, true).unwrap();
            t5[2].push(us(t));
            let t = Instant::now();
            drop(imp);
            t5[3].push(us(t));
            let t = Instant::now();
            vk_free(&v, b);
            t5[4].push(us(t));
        }
        let [mut a, mut b, mut c, mut d, mut e] = t5;
        println!(
            "{name},{:.1},{:.1},{:.1},{:.1},{:.1}",
            pct(&mut a, 0.5),
            pct(&mut b, 0.5),
            pct(&mut c, 0.5),
            pct(&mut d, 0.5),
            pct(&mut e, 0.5)
        );
    }

    // ------------------------------------------------------------------------------------------------------------
    println!(
        "# copy throughput, GB/s (100 MiB, 10 runs p50: CUDA events / Vulkan fence wall time)"
    );
    {
        let bytes = 100 * MIB as usize;
        let _c = dev.enter().unwrap();
        let mut stream = std::ptr::null_mut();
        let (mut d1, mut d2) = (0, 0);
        let mut h: *mut std::ffi::c_void = std::ptr::null_mut();
        let (mut e0, mut e1) = (std::ptr::null_mut(), std::ptr::null_mut());
        // SAFETY: out pointers.
        unsafe {
            (f.stream_create)(&mut stream, 1);
            (f.mem_alloc)(&mut d1, bytes);
            (f.mem_alloc)(&mut d2, bytes);
            (f.mem_host_alloc)(&mut h, bytes, 0);
            (f.event_create)(&mut e0, 0);
            (f.event_create)(&mut e1, 0);
        }
        let time = |what: &str, op: &dyn Fn()| {
            let mut ms = Vec::new();
            for _ in 0..10 {
                // SAFETY: events and stream are ours.
                unsafe {
                    (f.event_record)(e0, stream);
                    op();
                    (f.event_record)(e1, stream);
                    (f.event_synchronize)(e1);
                }
                let mut t = 0f32;
                // SAFETY: two recorded events.
                unsafe { (f.event_elapsed_time)(&mut t, e0, e1) };
                ms.push(f64::from(t));
            }
            println!(
                "{what},{:.1}",
                bytes as f64 / (pct(&mut ms, 0.5) * 1e-3) / 1e9
            );
        };
        // SAFETY: valid ranges of `bytes`.
        time("cuda.h2d.pinned", &|| unsafe {
            (f.memcpy_htod_async)(d1, h, bytes, stream);
        });
        // SAFETY: as above.
        time("cuda.d2h.pinned", &|| unsafe {
            (f.memcpy_dtoh_async)(h, d1, bytes, stream);
        });
        // SAFETY: as above.
        time("cuda.d2d", &|| unsafe {
            (f.memcpy_dtod_async)(d2, d1, bytes, stream);
        });
        // SAFETY: ours, idle.
        unsafe {
            (f.mem_free)(d1);
            (f.mem_free)(d2);
            (f.mem_free_host)(h);
            (f.event_destroy)(e0);
            (f.event_destroy)(e1);
            (f.stream_destroy)(stream);
        }
    }
    {
        let bytes = 100 * MIB;
        let a = vk_buffer(&v, bytes, dl, false).unwrap();
        let b = vk_buffer(&v, bytes, dl, false).unwrap();
        println!(
            "vulkan.d2d.native,{:.1}",
            vk_copy_gbs(&v, a.buf, b.buf, bytes)
        );
        vk_free(&v, a);
        vk_free(&v, b);
        let mut imported = Vec::new();
        for _ in 0..2 {
            let x = vmm_fd.alloc_vmm(bytes).unwrap();
            let fd = vmm_fd.export_fd(&x).unwrap();
            let mem = vk_import(&v, fd, x.size, type_bits).unwrap();
            let mut ext = vk::ExternalMemoryBufferCreateInfo::default()
                .handle_types(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD);
            let info = vk::BufferCreateInfo::default()
                .size(bytes)
                .usage(vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST)
                .push_next(&mut ext);
            // SAFETY: valid.
            let buf = unsafe { v.device.create_buffer(&info, None) }.unwrap();
            // SAFETY: imported memory big enough.
            unsafe { v.device.bind_buffer_memory(buf, mem, 0) }.unwrap();
            imported.push((x, mem, buf));
        }
        println!(
            "vulkan.d2d.imported_cuda_vmm,{:.1}",
            vk_copy_gbs(&v, imported[0].2, imported[1].2, bytes)
        );
        for (x, mem, buf) in imported {
            // SAFETY: idle.
            unsafe {
                v.device.destroy_buffer(buf, None);
                v.device.free_memory(mem, None);
                vmm_fd.free_vmm(x);
            }
        }
    }

    // ------------------------------------------------------------------------------------------------------------
    println!(
        "# blocking: a call's latency (µs p50) idle, and while unrelated work runs on another CUDA stream / the Vulkan queue"
    );
    println!("call,idle,busy_cuda,busy_vulkan");
    let busy_cuda = || -> (CuStream, u64) {
        let _c = dev.enter().unwrap();
        let mut s = std::ptr::null_mut();
        let mut d = 0;
        // SAFETY: out pointers; a 1 GiB scratch memset many times on a non-blocking stream.
        unsafe {
            (f.stream_create)(&mut s, 1);
            (f.mem_alloc)(&mut d, 1 << 30);
            for _ in 0..300 {
                (f.memset_d8_async)(d, 1, 1 << 30, s);
            }
        }
        (s, d)
    };
    let end_cuda = |(s, d): (CuStream, u64)| {
        let _c = dev.enter().unwrap();
        // SAFETY: ours.
        unsafe {
            (f.stream_synchronize)(s);
            (f.mem_free)(d);
            (f.stream_destroy)(s);
        }
    };
    let busy_vk = || {
        let bytes = 512 * MIB;
        let a = vk_buffer(&v, bytes, dl, false).unwrap();
        let b = vk_buffer(&v, bytes, dl, false).unwrap();
        let (pool, cb) = record_copies(&v, a.buf, b.buf, bytes, 120);
        // SAFETY: valid.
        let fence =
            unsafe { v.device.create_fence(&vk::FenceCreateInfo::default(), None) }.unwrap();
        let cbs = [cb];
        let submit = [vk::SubmitInfo::default().command_buffers(&cbs)];
        // SAFETY: recorded; waited for in `end_vk`.
        unsafe { v.device.queue_submit(v.queue, &submit, fence) }.unwrap();
        (a, b, pool, fence)
    };
    let end_vk = |(a, b, pool, fence): (VkBuf, VkBuf, vk::CommandPool, vk::Fence)| {
        // SAFETY: wait, then everything is idle.
        unsafe {
            v.device.wait_for_fences(&[fence], true, u64::MAX).unwrap();
            v.device.destroy_fence(fence, None);
            v.device.destroy_command_pool(pool, None);
        }
        vk_free(&v, a);
        vk_free(&v, b);
    };
    type Probe<'a> = (&'a str, Box<dyn Fn() -> f64 + 'a>);
    let probes: Vec<Probe> = vec![
        (
            "cuMemAlloc+cuMemFree 1MiB",
            Box::new(|| {
                let _c = dev.enter().unwrap();
                let mut p = 0;
                let t = Instant::now();
                // SAFETY: ours.
                unsafe {
                    (f.mem_alloc)(&mut p, MIB as usize);
                    (f.mem_free)(p);
                }
                us(t)
            }),
        ),
        (
            "cuMemFree 64MiB",
            Box::new(|| {
                let _c = dev.enter().unwrap();
                let mut p = 0;
                // SAFETY: ours.
                unsafe { (f.mem_alloc)(&mut p, 64 << 20) };
                let t = Instant::now();
                // SAFETY: ours.
                unsafe { (f.mem_free)(p) };
                us(t)
            }),
        ),
        (
            "cuMemHostAlloc 64MiB",
            Box::new(|| {
                let _c = dev.enter().unwrap();
                let mut p = std::ptr::null_mut();
                let t = Instant::now();
                // SAFETY: ours.
                unsafe { (f.mem_host_alloc)(&mut p, 64 << 20, 0) };
                let r = us(t);
                // SAFETY: ours.
                unsafe { (f.mem_free_host)(p) };
                r
            }),
        ),
        (
            "cuMemFreeHost 64MiB",
            Box::new(|| {
                let _c = dev.enter().unwrap();
                let mut p = std::ptr::null_mut();
                // SAFETY: ours.
                unsafe { (f.mem_host_alloc)(&mut p, 64 << 20, 0) };
                let t = Instant::now();
                // SAFETY: ours.
                unsafe { (f.mem_free_host)(p) };
                us(t)
            }),
        ),
        (
            "vmm alloc 64MiB",
            Box::new(|| {
                let t = Instant::now();
                let x = vmm.alloc_vmm(64 << 20).unwrap();
                let r = us(t);
                // SAFETY: unused.
                unsafe { vmm.free_vmm(x) };
                r
            }),
        ),
        (
            "vmm free 64MiB",
            Box::new(|| {
                let x = vmm.alloc_vmm(64 << 20).unwrap();
                let t = Instant::now();
                // SAFETY: unused.
                unsafe { vmm.free_vmm(x) };
                us(t)
            }),
        ),
        (
            "cuda import drop (vk-exported) 64MiB",
            Box::new(|| {
                let b = vk_buffer(&v, 64 << 20, dl, true).unwrap();
                let info = vk::MemoryGetFdInfoKHR::default()
                    .memory(b.mem)
                    .handle_type(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD);
                // SAFETY: exportable memory.
                let fd = unsafe { v.fd.get_memory_fd(&info) }.unwrap();
                // SAFETY: a new fd we own.
                let fd =
                    unsafe { <std::os::fd::OwnedFd as std::os::fd::FromRawFd>::from_raw_fd(fd) };
                // SAFETY: as above.
                let req = unsafe { v.device.get_buffer_memory_requirements(b.buf) };
                let imp = dev.import_fd(fd, req.size, true).unwrap();
                let t = Instant::now();
                drop(imp);
                let r = us(t);
                vk_free(&v, b);
                r
            }),
        ),
        (
            "cuMemAllocAsync+cuMemFreeAsync 64MiB",
            Box::new(|| {
                let _c = dev.enter().unwrap();
                let mut p = 0;
                let t = Instant::now();
                // SAFETY: ours; the null stream is not used: a pool stream of its own would be; the legacy stream
                // here shows only the call's host cost.
                unsafe {
                    (f.mem_alloc_async)(&mut p, 64 << 20, std::ptr::null_mut());
                    (f.mem_free_async)(p, std::ptr::null_mut());
                }
                us(t)
            }),
        ),
        (
            "vkAllocateMemory 64MiB",
            Box::new(|| {
                let t = Instant::now();
                let b = vk_buffer(&v, 64 << 20, dl, false).unwrap();
                let r = us(t);
                vk_free(&v, b);
                r
            }),
        ),
        (
            "vkFreeMemory 64MiB",
            Box::new(|| {
                let b = vk_buffer(&v, 64 << 20, dl, false).unwrap();
                let t = Instant::now();
                vk_free(&v, b);
                us(t)
            }),
        ),
    ];
    let t = Instant::now();
    end_cuda(busy_cuda());
    println!("busy_cuda_total_us,{:.0}", us(t));
    let t = Instant::now();
    end_vk(busy_vk());
    println!("busy_vulkan_total_us,{:.0}", us(t));
    for (name, probe) in &probes {
        let mut idle: Vec<f64> = (0..5).map(|_| probe()).collect();
        let mut bc = Vec::new();
        for _ in 0..3 {
            let busy = busy_cuda();
            std::thread::sleep(std::time::Duration::from_millis(20));
            bc.push(probe());
            end_cuda(busy);
        }
        let mut bv = Vec::new();
        for _ in 0..3 {
            let busy = busy_vk();
            std::thread::sleep(std::time::Duration::from_millis(20));
            bv.push(probe());
            end_vk(busy);
        }
        println!(
            "{name},{:.0},{:.0},{:.0}",
            pct(&mut idle, 0.5),
            pct(&mut bc, 0.5),
            pct(&mut bv, 0.5)
        );
    }

    // SAFETY: everything is freed and idle.
    unsafe {
        v.device.destroy_device(None);
        v.instance.destroy_instance(None);
    }
}
