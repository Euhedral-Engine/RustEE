//! Texture objects over pitched device memory: the texture units' filtered, clamped, normalised reads of 2D memory, as a 64-bit kernel
//! argument (`cudaTextureObject_t`).
//!
//! A [`View`] describes the memory (base, format, channels, size, pitch) and the sampling (filter, address and coordinate modes).
//! [`TexObject::create`] makes the object with `cuTexObjectCreate`: a driver call that belongs to a prepare, never to a unit of
//! work. The memory's base must be on the device's texture alignment and its pitch on the texture pitch alignment
//! ([`alignments`] says which; [`check_alignment`] refuses a view that does not meet them before the driver is asked, so a bad
//! arena fails the prepare). In a kernel a read is `tex2D<float>(object, u, v)` (normalised formats read as floats), or
//! `tex2D<unsigned>` with [`View::read_as_integer`]. The object does not own the memory: whoever does keeps it allocated until the
//! last kernel that reads it retired.

use std::ffi::{c_int, c_uint, c_void};
use std::sync::Arc;

use euhedral_memory::cuda::{Device, Driver};

/// `CU_RESOURCE_TYPE_PITCH2D`.
const RESOURCE_PITCH2D: c_int = 3;
/// `CU_TRSF_NORMALIZED_COORDINATES`: coordinates in 0..1.
const FLAG_NORMALIZED: c_uint = 2;
/// `CU_TRSF_READ_AS_INTEGER`: integer reads, no conversion to float.
const FLAG_READ_AS_INTEGER: c_uint = 1;

/// A texel format (`CUarray_format`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    U8,
    U16,
    F16,
    F32,
}

impl Format {
    fn code(self) -> c_int {
        match self {
            Format::U8 => 0x01,
            Format::U16 => 0x02,
            Format::F16 => 0x10,
            Format::F32 => 0x20,
        }
    }

    /// Bytes of one channel.
    pub fn bytes(self) -> u32 {
        match self {
            Format::U8 => 1,
            Format::U16 | Format::F16 => 2,
            Format::F32 => 4,
        }
    }
}

/// How a sample between texels is made.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Filter {
    Point,
    Linear,
}

/// What a coordinate outside the texture reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Address {
    Wrap,
    Clamp,
    Mirror,
    Border,
}

impl Address {
    fn code(self) -> c_int {
        match self {
            Address::Wrap => 0,
            Address::Clamp => 1,
            Address::Mirror => 2,
            Address::Border => 3,
        }
    }
}

/// A 2D view of pitched device memory.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct View {
    /// The device address of the first row; on the device's texture alignment.
    pub base: u64,
    pub format: Format,
    /// Channels per texel: 1, 2 or 4.
    pub channels: u32,
    pub width: usize,
    pub height: usize,
    /// Bytes per row; on the device's texture pitch alignment.
    pub pitch: usize,
    pub filter: Filter,
    pub address: Address,
    /// Coordinates in 0..1 over the texture (GLSL's normalised `sampler2D`), else in texels.
    pub normalized: bool,
    /// Integer formats read as normalised floats (`UNORM`, `tex2D<float>`), or as the integers (`tex2D<unsigned>`).
    pub read_as_integer: bool,
}

/// `CUDA_RESOURCE_DESC` (cuda.h): the type, the union of the resource kinds (128 bytes) and flags.
#[repr(C)]
struct ResourceDesc {
    res_type: c_int,
    pitch2d: Pitch2D,
    _union_tail: [u8; 128 - std::mem::size_of::<Pitch2D>()],
    flags: c_uint,
}

#[repr(C)]
struct Pitch2D {
    dev_ptr: u64,
    format: c_int,
    num_channels: c_uint,
    width: usize,
    height: usize,
    pitch_in_bytes: usize,
}

/// `CUDA_TEXTURE_DESC` (cuda.h).
#[repr(C)]
struct TextureDesc {
    address_mode: [c_int; 3],
    filter_mode: c_int,
    flags: c_uint,
    max_anisotropy: c_uint,
    mipmap_filter_mode: c_int,
    mipmap_level_bias: f32,
    min_mipmap_level_clamp: f32,
    max_mipmap_level_clamp: f32,
    border_color: [f32; 4],
    reserved: [c_int; 12],
}

const _: () = assert!(
    std::mem::size_of::<ResourceDesc>() == 144,
    "CUDA_RESOURCE_DESC is 144 bytes"
);
const _: () = assert!(
    std::mem::size_of::<TextureDesc>() == 104,
    "CUDA_TEXTURE_DESC is 104 bytes"
);

type FnCreate =
    unsafe extern "C" fn(*mut u64, *const ResourceDesc, *const TextureDesc, *const c_void) -> c_int;
type FnDestroy = unsafe extern "C" fn(u64) -> c_int;

/// The device attributes a texture needs (`cuDeviceGetAttribute`).
const ATTR_TEXTURE_ALIGNMENT: c_int = 14;
const ATTR_TEXTURE_PITCH_ALIGNMENT: c_int = 51;

type FnDeviceAttribute = unsafe extern "C" fn(*mut c_int, c_int, c_int) -> c_int;

/// The device's alignments: `(base, pitch)` in bytes.
pub fn alignments(device: &Device) -> Result<(u64, u64), String> {
    let driver = &device.driver;
    let get = symbol(driver, c"cuDeviceGetAttribute")?;
    // SAFETY: the driver exports this with cuda.h's signature.
    let get: FnDeviceAttribute =
        unsafe { std::mem::transmute::<*mut c_void, FnDeviceAttribute>(get) };
    let attribute = |attr| -> Result<u64, String> {
        let mut value = 0;
        // SAFETY: an out pointer; the attribute and the device ordinal are plain integers.
        let code = unsafe { get(&mut value, attr, device.ordinal) };
        driver
            .check("cuDeviceGetAttribute", code)
            .map_err(|e| e.to_string())?;
        Ok(u64::try_from(value).unwrap_or(0).max(1))
    };
    Ok((
        attribute(ATTR_TEXTURE_ALIGNMENT)?,
        attribute(ATTR_TEXTURE_PITCH_ALIGNMENT)?,
    ))
}

/// Whether a view's base and pitch meet the device's alignments (a prepare checks, so a bad arena fails the prepare).
pub fn check_alignment(view: &View, alignments: (u64, u64)) -> Result<(), String> {
    if !view.base.is_multiple_of(alignments.0) {
        return Err(format!(
            "texture base {:#x} is not on {} bytes",
            view.base, alignments.0
        ));
    }
    if !(view.pitch as u64).is_multiple_of(alignments.1) {
        return Err(format!(
            "texture pitch {} is not on {} bytes",
            view.pitch, alignments.1
        ));
    }
    if view.pitch < view.width * view.channels as usize * view.format.bytes() as usize {
        return Err(format!("texture pitch {} is less than a row", view.pitch));
    }
    Ok(())
}

/// A texture object. Its drop destroys it, which is a driver call: a prepare releases it, never a unit of work.
#[derive(Debug)]
pub struct TexObject {
    device: Arc<Device>,
    handle: u64,
    destroy: usize,
}

impl TexObject {
    /// **Build**: the object for `view`, made with the context current.
    pub fn create(device: &Arc<Device>, view: &View) -> Result<TexObject, String> {
        check_alignment(view, alignments(device)?)?;
        let driver: &Driver = &device.driver;
        let create = symbol(driver, c"cuTexObjectCreate")?;
        let destroy = symbol(driver, c"cuTexObjectDestroy")?;
        // SAFETY: the driver exports these with cuda.h's signatures.
        let create: FnCreate = unsafe { std::mem::transmute::<*mut c_void, FnCreate>(create) };
        let res = ResourceDesc {
            res_type: RESOURCE_PITCH2D,
            pitch2d: Pitch2D {
                dev_ptr: view.base,
                format: view.format.code(),
                num_channels: view.channels,
                width: view.width,
                height: view.height,
                pitch_in_bytes: view.pitch,
            },
            _union_tail: [0; 128 - std::mem::size_of::<Pitch2D>()],
            flags: 0,
        };
        let mut flags = 0;
        if view.normalized {
            flags |= FLAG_NORMALIZED;
        }
        if view.read_as_integer {
            flags |= FLAG_READ_AS_INTEGER;
        }
        let tex = TextureDesc {
            address_mode: [
                view.address.code(),
                view.address.code(),
                view.address.code(),
            ],
            filter_mode: match view.filter {
                Filter::Point => 0,
                Filter::Linear => 1,
            },
            flags,
            max_anisotropy: 1,
            mipmap_filter_mode: 0,
            mipmap_level_bias: 0.0,
            min_mipmap_level_clamp: 0.0,
            max_mipmap_level_clamp: 0.0,
            border_color: [0.0; 4],
            reserved: [0; 12],
        };
        let _current = device.enter().map_err(|e| e.to_string())?;
        let mut handle = 0u64;
        // SAFETY: valid descriptors for the call's duration, an out pointer, a null view descriptor; the context is current.
        let code = unsafe { create(&mut handle, &res, &tex, std::ptr::null()) };
        driver
            .check("cuTexObjectCreate", code)
            .map_err(|e| e.to_string())?;
        Ok(TexObject {
            device: device.clone(),
            handle,
            destroy: destroy as usize,
        })
    }

    /// The object as a kernel argument (`cudaTextureObject_t`).
    pub fn handle(&self) -> u64 {
        self.handle
    }
}

impl Drop for TexObject {
    fn drop(&mut self) {
        if let Ok(_current) = self.device.enter() {
            // SAFETY: the address came from `cuTexObjectDestroy`'s symbol; the object was made by `create` and is destroyed once.
            let destroy: FnDestroy =
                unsafe { std::mem::transmute::<usize, FnDestroy>(self.destroy) };
            // SAFETY: a live texture object of this context; destroying it does not wait for kernels that read it (the owner drops it after its last reader retired).
            unsafe { destroy(self.handle) };
        }
    }
}

fn symbol(driver: &Driver, name: &std::ffi::CStr) -> Result<*mut c_void, String> {
    let address = driver.symbol(name);
    if address.is_null() {
        Err(format!("libcuda: no {}", name.to_string_lossy()))
    } else {
        Ok(address)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn view() -> View {
        View {
            base: 512 * 7,
            format: Format::U16,
            channels: 1,
            width: 100,
            height: 10,
            pitch: 224,
            filter: Filter::Linear,
            address: Address::Clamp,
            normalized: true,
            read_as_integer: false,
        }
    }

    #[test]
    fn alignment_is_checked_before_the_driver_is_asked() {
        let a = (512, 32);
        assert!(check_alignment(&view(), a).is_ok());
        assert!(
            check_alignment(
                &View {
                    base: 513,
                    ..view()
                },
                a
            )
            .unwrap_err()
            .contains("base")
        );
        assert!(
            check_alignment(
                &View {
                    pitch: 210,
                    ..view()
                },
                a
            )
            .unwrap_err()
            .contains("pitch")
        );
        assert!(
            check_alignment(
                &View {
                    pitch: 160,
                    ..view()
                },
                a
            )
            .unwrap_err()
            .contains("less than a row")
        );
    }

    #[test]
    fn the_descriptors_have_cuda_hs_sizes() {
        assert_eq!(std::mem::size_of::<ResourceDesc>(), 144);
        assert_eq!(std::mem::size_of::<TextureDesc>(), 104);
    }
}
