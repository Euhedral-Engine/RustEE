//! Fake recurring sequences: record, probe, update, re-record and replay.

use euhedral_graph::Marker;

use crate::hold::{Frozen, freeze};
use crate::{Fingerprint, GpuError, Graph, GraphHold, Kernel, Rerecorded, Sequence, Stream};

use super::enqueue::{Mode, Real, RecOp, Recording};
use super::{FakeEnqueue, FakeGpu, FakeKernel, FakeQueue, FakeStream, QOp};

/// A fake recording.
#[derive(Debug)]
pub struct FakeGraph {
    ops: Vec<RecOp>,
    /// The arguments of each launch, as of the last update.
    args: Vec<crate::Args>,
    fingerprint: Fingerprint,
    streams: usize,
    /// The buffers the recording reads, frozen and shared with every replay (see `Graph::replay`).
    kept: Frozen,
}

impl FakeGraph {
    fn kernels(&self) -> Vec<u64> {
        self.ops
            .iter()
            .filter_map(|op| match op {
                RecOp::Launch { kernel, .. } => Some(kernel.id()),
                _ => None,
            })
            .collect()
    }
}

impl Graph for FakeGraph {
    type Gpu = FakeGpu;

    fn fingerprint(&self) -> Fingerprint {
        self.fingerprint
    }

    fn launches(&self) -> usize {
        self.args.len()
    }

    fn replay(&mut self, stream: &mut FakeStream) -> Result<(), GpuError> {
        stream.gpu().alive()?;
        self.push_ops(|op| stream.push(op));
        // A replay runs after this graph may have dropped: the stream keeps the recording's buffers until the next
        // retirement armed after it is confirmed.
        if let Some(kept) = &self.kept {
            let seq = stream.retire.next_seq();
            stream.kept.push(seq, Box::new(kept.clone()));
        }
        Ok(())
    }

    fn replay_on(
        &mut self,
        lane: &FakeQueue,
        waits: &[Marker],
        hold: &mut GraphHold,
    ) -> Result<(), GpuError> {
        lane.gpu.alive()?;
        for marker in waits {
            lane.await_now(*marker)?;
        }
        self.push_ops(|op| lane.push_op(op));
        // The lane keeps nothing: the caller holds the recording's buffers until the work ended.
        hold.add_frozen(&self.kept);
        Ok(())
    }
}

impl FakeGraph {
    /// The recording's operations, in order, with the arguments of the last update.
    fn push_ops(&self, mut push: impl FnMut(QOp)) {
        for op in &self.ops {
            push(match op {
                RecOp::Launch {
                    kernel,
                    launch,
                    arg,
                } => QOp::Launch {
                    kernel: kernel.clone(),
                    launch: *launch,
                    args: self.args[*arg].clone(),
                },
                RecOp::ToDevice { dst, src, bytes } => QOp::ToDevice {
                    dst: *dst,
                    src: *src,
                    bytes: *bytes,
                },
                RecOp::ToHost { dst, src, bytes } => QOp::ToHost {
                    dst: *dst,
                    src: *src,
                    bytes: *bytes,
                },
                RecOp::OnDevice { dst, src, bytes } => QOp::OnDevice {
                    dst: *dst,
                    src: *src,
                    bytes: *bytes,
                },
                RecOp::Memset { dst, value, bytes } => QOp::Memset {
                    dst: *dst,
                    value: *value,
                    bytes: *bytes,
                },
            });
        }
    }
}

/// A pass over a recurring sequence.
#[derive(Debug)]
pub struct FakeSequence<'a> {
    rec: Recording,
    streams: usize,
    /// The real streams or lanes, borrowed for the whole closure, in the sequence's order (a record pass only).
    real: Vec<Real<'a>>,
}

impl Sequence for FakeSequence<'_> {
    type Kernel = FakeKernel;
    type Enqueue<'a> = FakeEnqueue<'a>;

    fn submit(
        &mut self,
        stream: usize,
        f: impl FnOnce(&mut FakeEnqueue<'_>) -> Result<(), GpuError>,
    ) -> Result<(), GpuError> {
        if stream >= self.streams {
            return Err(GpuError::Invalid(format!(
                "stream {stream} is not one of the sequence's {}",
                self.streams
            )));
        }
        let target = self.real.get_mut(stream).map(Real::reborrow);
        let mut q = FakeEnqueue {
            target,
            rec: Some((&mut self.rec, stream)),
            hold: None,
        };
        f(&mut q)
    }

    fn order(&mut self, after: usize, before: usize) -> Result<(), GpuError> {
        if after >= self.streams || before >= self.streams {
            return Err(GpuError::Invalid(format!(
                "order({after}, {before}) names a stream outside the sequence's {}",
                self.streams
            )));
        }
        if after == before {
            return Ok(());
        }
        self.rec.hasher.order(after, before);
        if let (Some(a), Some(b)) = (self.real.get(after), self.real.get(before)) {
            let marker: Marker = a.mark()?;
            b.await_marker(marker)?;
        }
        Ok(())
    }
}

impl FakeGpu {
    /// Runs `f` for real over `real`, forked from and joined to `real[0]`, collecting the recording. The streams are
    /// borrowed mutably for the whole run (a lane is shared: the caller holds what the run reads).
    pub(crate) fn record_pass<F>(
        &self,
        real: Vec<Real<'_>>,
        mut hold: Option<&mut GraphHold>,
        f: F,
    ) -> Result<(Recording, Fingerprint), GpuError>
    where
        F: FnOnce(&mut FakeSequence<'_>) -> Result<(), GpuError>,
    {
        self.alive()?;
        if real.is_empty() {
            return Err(GpuError::Invalid("a recording needs a stream".into()));
        }
        // Fork: the other streams start behind real[0], as a replay on real[0] would.
        if real.len() > 1 {
            let marker = real[0].mark()?;
            for s in &real[1..] {
                s.await_marker(marker)?;
            }
        }
        let mut seq = FakeSequence {
            rec: Recording::new(Mode::Record),
            streams: real.len(),
            real,
        };
        let result = f(&mut seq);
        // The record pass runs the work for real: freeze what it kept and hand it to every real stream, to be held
        // until that stream's next retirement is confirmed (or to the caller of a shared lane), whatever becomes of the
        // recording. The graph shares the same values.
        seq.rec.frozen = freeze(std::mem::take(&mut seq.rec.kept));
        if let Some(frozen) = seq.rec.frozen.clone() {
            for stream in &mut seq.real {
                if let Real::Stream(stream) = stream {
                    let at = stream.retire.next_seq();
                    stream.kept.push(at, Box::new(frozen.clone()));
                }
            }
        }
        if let Some(hold) = hold.as_mut() {
            hold.add_frozen(&seq.rec.frozen);
        }
        let rec = std::mem::replace(&mut seq.rec, Recording::new(Mode::Probe));
        // Join: real[0] ends behind every other stream. Done even when `f` failed, so the streams stay consistent.
        let mut joined = Ok(());
        for s in seq.real.iter().skip(1) {
            joined = joined.and_then(|()| seq.real[0].await_marker(s.mark()?));
        }
        result?;
        joined?;
        let fingerprint = rec.hasher.finish();
        Ok((rec, fingerprint))
    }

    pub(crate) fn probe_pass<F>(
        &self,
        mode: Mode,
        streams: usize,
        f: F,
    ) -> Result<Recording, GpuError>
    where
        F: FnOnce(&mut FakeSequence<'_>) -> Result<(), GpuError>,
    {
        self.alive()?;
        let mut seq = FakeSequence {
            rec: Recording::new(mode),
            streams,
            real: Vec::new(),
        };
        f(&mut seq)?;
        Ok(seq.rec)
    }

    pub(crate) fn record_graph<F>(
        &self,
        real: Vec<Real<'_>>,
        hold: Option<&mut GraphHold>,
        f: F,
    ) -> Result<FakeGraph, GpuError>
    where
        F: FnOnce(&mut FakeSequence<'_>) -> Result<(), GpuError>,
    {
        let count = real.len();
        let hook = self
            .fail_recording
            .swap(0, std::sync::atomic::Ordering::AcqRel);
        if hook == 2 {
            return Err(GpuError::Capture(
                "failed by the test hook before the body ran".into(),
            ));
        }
        let (rec, fingerprint) = self.record_pass(real, hold, f)?;
        if hook == 1 {
            return Err(GpuError::Capture("failed by the test hook".into()));
        }
        Ok(FakeGraph {
            ops: rec.ops,
            args: rec.args,
            fingerprint,
            streams: count,
            kept: rec.frozen,
        })
    }

    pub(crate) fn rerecord_graph<F>(
        &self,
        graph: &mut FakeGraph,
        real: Vec<Real<'_>>,
        hold: Option<&mut GraphHold>,
        f: F,
    ) -> Result<Rerecorded, GpuError>
    where
        F: FnOnce(&mut FakeSequence<'_>) -> Result<(), GpuError>,
    {
        let new = self.record_graph(real, hold, f)?;
        let how = if new.kernels() == graph.kernels() {
            Rerecorded::Updated
        } else {
            Rerecorded::Instantiated
        };
        // The old recording's buffers go with it; a replay still in flight holds its own handle.
        *graph = new;
        Ok(how)
    }

    pub(crate) fn update_graph<F>(&self, graph: &mut FakeGraph, f: F) -> Result<(), GpuError>
    where
        F: FnOnce(&mut FakeSequence<'_>) -> Result<(), GpuError>,
    {
        let rec = self.probe_pass(Mode::Update, graph.streams, f)?;
        let now = rec.hasher.finish();
        if now != graph.fingerprint {
            return Err(GpuError::ShapeChanged {
                recorded: graph.fingerprint,
                now,
            });
        }
        graph.args = rec.staged;
        Ok(())
    }
}
