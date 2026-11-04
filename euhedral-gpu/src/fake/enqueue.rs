//! What a fake submission enqueues, on a stream or into a recurring sequence's passes.

use std::any::Any;

use crate::hold::{Frozen, freeze};
use crate::shape::{CopyKind, ShapeHasher};
use euhedral_graph::{LaneId, Marker};

use crate::{
    Args, Enqueue, GpuError, GraphHold, Kernel, Lane, LaneEnqueue, LaneStream, Launch, RawStream,
};

use super::{FakeKernel, FakeQueue, FakeStream, QOp};

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

/// Where an operation goes: an owned stream, or a shared lane.
#[derive(Debug)]
pub(crate) enum Real<'a> {
    Stream(&'a mut FakeStream),
    Lane(&'a FakeQueue),
}

impl Real<'_> {
    pub(crate) fn reborrow(&mut self) -> Real<'_> {
        match self {
            Real::Stream(s) => Real::Stream(s),
            Real::Lane(l) => Real::Lane(l),
        }
    }

    /// The lane\'s id and global index.
    pub(crate) fn identity(&self) -> (LaneId, usize) {
        match self {
            Real::Stream(s) => (LaneId(s.lane_id()), s.global()),
            Real::Lane(l) => (l.lane, l.global()),
        }
    }

    pub(crate) fn push(&self, op: QOp) {
        match self {
            Real::Stream(s) => s.push(op),
            Real::Lane(l) => l.push_op(op),
        }
    }

    /// A marker after the work enqueued so far.
    pub(crate) fn mark(&self) -> Result<Marker, GpuError> {
        match self {
            Real::Stream(s) => s.mark_now(),
            Real::Lane(l) => Lane::mark(*l),
        }
    }

    /// Later work waits for `marker`.
    pub(crate) fn await_marker(&self, marker: Marker) -> Result<(), GpuError> {
        match self {
            Real::Stream(s) => s.await_now(marker),
            Real::Lane(l) => l.await_now(marker),
        }
    }
}

/// What a fake submission enqueues: onto a stream or a shared lane, into a recording, or both.
#[derive(Debug)]
pub struct FakeEnqueue<'a> {
    /// Where the operations go; none on a probe or an update.
    pub(crate) target: Option<Real<'a>>,
    /// The pass collecting them, and the sequence's index of the stream.
    pub(crate) rec: Option<(&'a mut Recording, usize)>,
    /// Where the values a body keeps go when it runs directly on a shared lane, which keeps none.
    pub(crate) hold: Option<&'a mut GraphHold>,
}

impl FakeEnqueue<'_> {
    fn push(&self, op: QOp) {
        if let Some(target) = &self.target {
            target.push(op);
        }
    }
}

impl LaneEnqueue for FakeEnqueue<'_> {
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
}

impl Enqueue for FakeEnqueue<'_> {
    fn keep_with<T: Any + Send>(&mut self, make: impl FnOnce() -> T) {
        if matches!(&self.rec, Some((rec, _)) if rec.mode != Mode::Record) {
            return;
        }
        self.keep(Box::new(make()));
    }

    fn keep(&mut self, value: Box<dyn Any + Send>) {
        match (&mut self.rec, &mut self.target) {
            // In a recording the value belongs to the graph.
            (Some((rec, _)), _) if rec.mode == Mode::Record => rec.kept.push(value),
            (Some(_), _) => drop(value),
            (None, Some(Real::Stream(stream))) => {
                let seq = stream.retire.next_seq();
                stream.kept.push(seq, value);
            }
            (None, Some(Real::Lane(_))) => match &mut self.hold {
                Some(hold) => hold.add_value(value),
                None => drop(value),
            },
            (None, None) => drop(value),
        }
    }
}

impl LaneStream for FakeEnqueue<'_> {
    type Raw = super::FakeRawStream;

    fn raw_stream(&self) -> Option<RawStream<'_, super::FakeRawStream>> {
        if self.rec.is_some() {
            return None;
        }
        self.target.as_ref().map(|target| {
            let (lane, global) = target.identity();
            RawStream::new(super::FakeRawStream { lane, global })
        })
    }
}
