//! Fake recurring sequences: record, probe, update, re-record and replay.

use euhedral_graph::Marker;

use crate::kept::{Frozen, freeze};
use crate::{Fingerprint, GpuError, Graph, Kernel, Rerecorded, Sequence, Stream};

use super::enqueue::{Mode, RecOp, Recording};
use super::{FakeEnqueue, FakeGpu, FakeKernel, FakeStream, QOp};

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
        stream.gpu().poison.check()?;
        for op in &self.ops {
            stream.push(match op {
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
        // A replay runs after this graph may have dropped: the stream keeps the recording's buffers until the next
        // retirement armed after it is confirmed.
        if !self.kept.is_empty() {
            let seq = stream.next_retire;
            stream.kept.push(seq, Box::new(self.kept.clone()));
        }
        Ok(())
    }
}

/// A pass over a recurring sequence.
#[derive(Debug)]
pub struct FakeSequence<'a> {
    rec: Recording,
    streams: usize,
    /// The real streams, borrowed for the whole closure, in the sequence's order (a record pass only).
    real: Vec<&'a mut FakeStream>,
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
        let target = self.real.get_mut(stream).map(|s| &mut **s);
        let mut q = FakeEnqueue {
            target,
            rec: Some((&mut self.rec, stream)),
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
            let marker: Marker = a.mark_now()?;
            b.await_now(marker)?;
        }
        Ok(())
    }
}

impl FakeGpu {
    /// Runs `f` for real over `streams`, forked from and joined to `streams[0]`, collecting the recording. The
    /// streams are borrowed mutably for the whole run: nothing else can enqueue on them meanwhile.
    pub(crate) fn record_pass<F>(
        &self,
        streams: &mut [&mut FakeStream],
        f: F,
    ) -> Result<(Recording, Fingerprint), GpuError>
    where
        F: FnOnce(&mut FakeSequence<'_>) -> Result<(), GpuError>,
    {
        self.poison.check()?;
        if streams.is_empty() {
            return Err(GpuError::Invalid("a recording needs a stream".into()));
        }
        let real: Vec<&mut FakeStream> = streams.iter_mut().map(|s| &mut **s).collect();
        // Fork: the other streams start behind streams[0], as a replay on streams[0] would.
        if real.len() > 1 {
            let marker = real[0].mark_now()?;
            for s in &real[1..] {
                s.await_now(marker)?;
            }
        }
        let mut seq = FakeSequence {
            rec: Recording::new(Mode::Record),
            streams: real.len(),
            real,
        };
        let result = f(&mut seq);
        // The record pass runs the work for real: freeze what it kept and hand it to every real stream, to be held
        // until that stream's next retirement is confirmed, whatever becomes of the recording. The graph shares the
        // same values.
        seq.rec.frozen = freeze(std::mem::take(&mut seq.rec.kept));
        if !seq.rec.frozen.is_empty() {
            let frozen = seq.rec.frozen.clone();
            for stream in &mut seq.real {
                let at = stream.next_retire;
                stream.kept.push(at, Box::new(frozen.clone()));
            }
        }
        let rec = std::mem::replace(&mut seq.rec, Recording::new(Mode::Probe));
        // Join: streams[0] ends behind every other stream. Done even when `f` failed, so the streams stay consistent.
        let mut joined = Ok(());
        for s in seq.real.iter().skip(1) {
            joined = joined.and_then(|()| seq.real[0].await_now(s.mark_now()?));
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
        self.poison.check()?;
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
        streams: &mut [&mut FakeStream],
        f: F,
    ) -> Result<FakeGraph, GpuError>
    where
        F: FnOnce(&mut FakeSequence<'_>) -> Result<(), GpuError>,
    {
        let count = streams.len();
        let (rec, fingerprint) = self.record_pass(streams, f)?;
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
        streams: &mut [&mut FakeStream],
        f: F,
    ) -> Result<Rerecorded, GpuError>
    where
        F: FnOnce(&mut FakeSequence<'_>) -> Result<(), GpuError>,
    {
        let new = self.record_graph(streams, f)?;
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
