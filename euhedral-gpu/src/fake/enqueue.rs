//! What a fake submission enqueues, on a stream or into a recurring sequence's passes.

use std::any::Any;

use crate::kept::{Frozen, freeze};
use crate::shape::{CopyKind, ShapeHasher};
use crate::{Args, Enqueue, GpuError, Kernel, Launch};

use super::{FakeKernel, FakeStream, QOp};

/// An operation of a recording, replayed in recording order.
#[derive(Clone, Debug)]
pub(crate) enum RecOp {
    /// `arg` indexes the graph's launch arguments.
    Launch {
        kernel: FakeKernel,
        launch: Launch,
        arg: usize,
    },
    ToDevice {
        dst: u64,
        src: usize,
        bytes: usize,
    },
    ToHost {
        dst: usize,
        src: u64,
        bytes: usize,
    },
    OnDevice {
        dst: u64,
        src: u64,
        bytes: usize,
    },
    Memset {
        dst: u64,
        value: u8,
        bytes: usize,
    },
}

/// Which pass a sequence is running.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Mode {
    /// Run for real and keep the operations.
    Record,
    /// Fingerprint only.
    Probe,
    /// Fingerprint, and stage the arguments of each launch.
    Update,
}

/// What a pass collects.
#[derive(Debug)]
pub(crate) struct Recording {
    pub(crate) mode: Mode,
    pub(crate) ops: Vec<RecOp>,
    /// The arguments of each launch, in launch order.
    pub(crate) args: Vec<Args>,
    pub(crate) hasher: ShapeHasher,
    pub(crate) launches: usize,
    /// The values the closure kept while recording.
    pub(crate) kept: Vec<Box<dyn Any + Send>>,
    /// The same, frozen once the run ended, to be shared by the streams, the graph and every replay.
    pub(crate) frozen: Frozen,
    pub(crate) staged: Vec<Args>,
}

impl Recording {
    pub(crate) fn new(mode: Mode) -> Self {
        Self {
            mode,
            ops: Vec::new(),
            args: Vec::new(),
            hasher: ShapeHasher::default(),
            launches: 0,
            kept: Vec::new(),
            frozen: freeze(Vec::new()),
            staged: Vec::new(),
        }
    }
}

/// What a fake submission enqueues: onto a stream, into a recording, or both.
#[derive(Debug)]
pub struct FakeEnqueue<'a> {
    /// The stream the operations go to; none on a probe or an update.
    pub(crate) target: Option<&'a mut FakeStream>,
    /// The pass collecting them, and the sequence's index of the stream.
    pub(crate) rec: Option<(&'a mut Recording, usize)>,
}

impl FakeEnqueue<'_> {
    fn push(&self, op: QOp) {
        if let Some(stream) = &self.target {
            stream.push(op);
        }
    }
}

impl Enqueue for FakeEnqueue<'_> {
    type Kernel = FakeKernel;

    unsafe fn launch(
        &mut self,
        kernel: &FakeKernel,
        launch: Launch,
        args: &Args,
    ) -> Result<(), GpuError> {
        if let Some((rec, stream)) = &mut self.rec {
            rec.hasher
                .launch(*stream, kernel.id(), &launch, args.layout());
            match rec.mode {
                Mode::Record => {
                    rec.ops.push(RecOp::Launch {
                        kernel: kernel.clone(),
                        launch,
                        arg: rec.args.len(),
                    });
                    rec.args.push(args.clone());
                }
                Mode::Update => rec.staged.push(args.clone()),
                Mode::Probe => {}
            }
            rec.launches += 1;
        }
        self.push(QOp::Launch {
            kernel: kernel.clone(),
            launch,
            args: args.clone(),
        });
        Ok(())
    }

    unsafe fn copy_to_device(
        &mut self,
        dst: u64,
        src: *const u8,
        bytes: usize,
    ) -> Result<(), GpuError> {
        if let Some((rec, stream)) = &mut self.rec {
            rec.hasher
                .copy(*stream, CopyKind::ToDevice, dst, src as u64, bytes as u64);
            if rec.mode == Mode::Record {
                rec.ops.push(RecOp::ToDevice {
                    dst,
                    src: src as usize,
                    bytes,
                });
            }
        }
        self.push(QOp::ToDevice {
            dst,
            src: src as usize,
            bytes,
        });
        Ok(())
    }

    unsafe fn copy_to_host(
        &mut self,
        dst: *mut u8,
        src: u64,
        bytes: usize,
    ) -> Result<(), GpuError> {
        if let Some((rec, stream)) = &mut self.rec {
            rec.hasher
                .copy(*stream, CopyKind::ToHost, dst as u64, src, bytes as u64);
            if rec.mode == Mode::Record {
                rec.ops.push(RecOp::ToHost {
                    dst: dst as usize,
                    src,
                    bytes,
                });
            }
        }
        self.push(QOp::ToHost {
            dst: dst as usize,
            src,
            bytes,
        });
        Ok(())
    }

    unsafe fn copy_on_device(&mut self, dst: u64, src: u64, bytes: usize) -> Result<(), GpuError> {
        if let Some((rec, stream)) = &mut self.rec {
            rec.hasher
                .copy(*stream, CopyKind::OnDevice, dst, src, bytes as u64);
            if rec.mode == Mode::Record {
                rec.ops.push(RecOp::OnDevice { dst, src, bytes });
            }
        }
        self.push(QOp::OnDevice { dst, src, bytes });
        Ok(())
    }

    unsafe fn memset(&mut self, dst: u64, value: u8, bytes: usize) -> Result<(), GpuError> {
        if let Some((rec, stream)) = &mut self.rec {
            rec.hasher.memset(*stream, dst, value, bytes as u64);
            if rec.mode == Mode::Record {
                rec.ops.push(RecOp::Memset { dst, value, bytes });
            }
        }
        self.push(QOp::Memset { dst, value, bytes });
        Ok(())
    }

    fn keep(&mut self, value: Box<dyn Any + Send>) {
        match (&mut self.rec, &mut self.target) {
            // In a recording the value belongs to the graph.
            (Some((rec, _)), _) if rec.mode == Mode::Record => rec.kept.push(value),
            (Some(_), _) => drop(value),
            (None, Some(stream)) => {
                let seq = stream.next_retire;
                stream.kept.push(seq, value);
            }
            (None, None) => drop(value),
        }
    }
}
