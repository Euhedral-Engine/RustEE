//! Linux and Android: which engine an [`AsyncIo`] runs on, io_uring or the positional fallback.

use std::io;
use std::sync::Arc;
use std::time::Duration;

use euhedral_core::frames::Frame;

use super::positional::Positional;
use super::uring::Uring;
use super::{AsyncIo, AsyncIoOptions, Buf, Engine, OpId, Request, Then};

/// The engine an [`AsyncIo`] runs on (see [`AsyncIo::backend`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum IoBackend {
    /// io_uring: the kernel does the work.
    IoUring = 0,
    /// Positional calls (`pread`, `pwrite`, ...) made by the reap frame on a worker: the fallback for a
    /// system that refuses io_uring (a container's default seccomp profile, `kernel.io_uring_disabled`,
    /// gVisor), or the choice of [`AsyncIoOptions::positional_io`].
    Positional = 1,
}

/// The engine of an [`AsyncIo`].
pub(crate) enum Platform {
    Uring(Uring),
    Positional(Positional),
}

/// Whether `error` is how a system says it refuses io_uring: a seccomp filter or a sysctl (`EPERM`,
/// `EACCES`) or a kernel without it (`ENOSYS`). Any other error is a real failure.
fn refused(error: &io::Error) -> bool {
    matches!(
        error.raw_os_error(),
        Some(libc::EPERM | libc::EACCES | libc::ENOSYS)
    )
}

impl Platform {
    /// Chooses the engine: the positional one if `options` force it, else io_uring from `uring` (the setup
    /// of the first ring), and the positional one if that setup was refused.
    pub(crate) fn choose(
        options: AsyncIoOptions,
        uring: impl FnOnce() -> io::Result<Uring>,
    ) -> io::Result<Self> {
        if options.positional_io {
            return Ok(Self::Positional(Positional::new()));
        }
        match uring() {
            Ok(uring) => Ok(Self::Uring(uring)),
            Err(error) if refused(&error) => Ok(Self::Positional(Positional::new())),
            Err(error) => Err(error),
        }
    }

    pub(crate) fn backend(&self) -> IoBackend {
        match self {
            Self::Uring(_) => IoBackend::IoUring,
            Self::Positional(_) => IoBackend::Positional,
        }
    }
}

impl Engine for Platform {
    fn submit(
        &self,
        request: Request,
        buf: Buf,
        then: Then,
        limit: Option<Duration>,
    ) -> io::Result<OpId> {
        match self {
            Self::Uring(engine) => engine.submit(request, buf, then, limit),
            Self::Positional(engine) => engine.submit(request, buf, then, limit),
        }
    }

    fn cancel(&self, id: OpId) {
        match self {
            Self::Uring(engine) => engine.cancel(id),
            Self::Positional(engine) => engine.cancel(id),
        }
    }

    fn poll(&self, room: &mut dyn FnMut() -> bool, offer: &mut dyn FnMut(Box<dyn Frame>) -> bool) {
        match self {
            Self::Uring(engine) => engine.poll(room, offer),
            Self::Positional(engine) => engine.poll(room, offer),
        }
    }
}

impl AsyncIo {
    /// Asynchronous I/O on this machine: io_uring, or where the system refuses it the positional fallback
    /// (see [`backend`](Self::backend)). It fails only for another reason (no memory, no descriptors).
    pub fn open() -> io::Result<Arc<Self>> {
        Self::open_with_options(AsyncIoOptions::default())
    }

    /// As [`open`](Self::open), with options: [`AsyncIoOptions::positional_io`] forces the fallback.
    pub fn open_with_options(options: AsyncIoOptions) -> io::Result<Arc<Self>> {
        Ok(Self::from_engine(Platform::choose(options, || {
            Uring::native()
        })?))
    }

    /// The engine this `AsyncIo` runs on.
    pub fn backend(&self) -> IoBackend {
        self.engine.backend()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn with_setup_error(code: i32) -> io::Result<Arc<AsyncIo>> {
        Ok(AsyncIo::from_engine(Platform::choose(
            AsyncIoOptions::default(),
            || Err(io::Error::from_raw_os_error(code)),
        )?))
    }

    #[test]
    fn a_setup_the_system_refuses_selects_the_positional_engine() {
        for code in [libc::EPERM, libc::EACCES, libc::ENOSYS] {
            let io = with_setup_error(code).unwrap();
            assert_eq!(io.backend(), IoBackend::Positional, "errno {code}");
        }
    }

    #[test]
    fn any_other_setup_error_fails_the_open() {
        for code in [libc::ENOMEM, libc::EMFILE, libc::EINVAL, libc::ENFILE] {
            let Err(error) = with_setup_error(code) else {
                panic!("errno {code} did not fail the open");
            };
            assert_eq!(error.raw_os_error(), Some(code));
        }
    }

    #[test]
    fn the_option_forces_the_positional_engine_without_trying_io_uring() {
        let options = AsyncIoOptions {
            positional_io: true,
            ..AsyncIoOptions::default()
        };
        let engine = Platform::choose(options, || panic!("io_uring was tried")).unwrap();
        assert_eq!(engine.backend(), IoBackend::Positional);
        let io = AsyncIo::open_with_options(options).unwrap();
        assert_eq!(io.backend(), IoBackend::Positional);
    }

    #[test]
    fn by_default_io_uring_is_used_where_it_works() {
        let io = AsyncIo::open().unwrap();
        // A machine that refuses io_uring runs the fallback; the open must not fail either way.
        assert!(matches!(
            io.backend(),
            IoBackend::IoUring | IoBackend::Positional
        ));
    }

    #[test]
    fn the_backends_have_fixed_discriminants() {
        assert_eq!(IoBackend::IoUring as u8, 0);
        assert_eq!(IoBackend::Positional as u8, 1);
    }
}
