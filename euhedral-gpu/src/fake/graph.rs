//! Fake recurring sequences: record, probe, update, re-record and replay.

use std::any::Any;
use std::sync::{Arc, Mutex, PoisonError};

use euhedral_graph::Marker;

use crate::stream_lock::StreamGuard;
use crate::{Fingerprint, GpuError, Graph, Kernel, Rerecorded, Sequence, Stream};

use super::enqueue::{Mode, RecOp, Recording};
use super::stream::StreamState;
use super::{FakeEnqueue, FakeGpu, FakeKernel, FakeStream, QOp};

/// A fake recording.
#[derive(Debug)]
pub struct FakeGraph {
    ops: Vec<RecOp>,
    /// The arguments of each launch, as of the last update.
    args: Vec<crate::Args>,
    fingerprint: Fingerprint,
    streams: usize,
    #[allow(
        dead_code,
        reason = "held for the graph's life, like the buffers a recording reads"
    )]
    kept: Mutex<Vec<Box<dyn Any + Send>>>,
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

    fn replay(&self, stream: &FakeStream) -> Result<(), GpuError> {
        let _guard = stream.lock()?;
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
        Ok(())
    }
}

/// A pass over a recurring sequence.
#[derive(Debug)]
pub struct FakeSequence<'a> {
    rec: Recording,
    streams: usize,
    /// The real streams and their held locks, in the sequence's order (a record pass only).
    real: Vec<(&'a FakeStream, StreamGuard<'a, StreamState>)>,
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
        let target = self
            .real
            .get_mut(stream)
            .map(|(s, guard)| (*s, &mut **guard));
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
        if let (Some(&(a, _)), Some(&(b, _))) = (self.real.get(after), self.real.get(before)) {
            let marker: Marker = a.mark_held()?;
            b.await_held(marker)?;
        }
        Ok(())
    }
}

impl FakeGpu {
    /// Runs `f` for real over `streams`, forked from and joined to `streams[0]`, collecting the recording.
    pub(crate) fn record_pass<F>(
        &self,
        streams: &[&Arc<FakeStream>],
        f: F,
    ) -> Result<(Recording, Fingerprint), GpuError>
    where
        F: FnOnce(&mut FakeSequence<'_>) -> Result<(), GpuError>,
    {
        self.poison.check()?;
        if streams.is_empty() {
            return Err(GpuError::Invalid("a recording needs a stream".into()));
        }
        // Lock in ascending lane order, so two recordings never wait on each other crosswise.
        let mut order: Vec<usize> = (0..streams.len()).collect();
        order.sort_by_key(|&i| streams[i].global_index());
        let mut guards: Vec<Option<StreamGuard<'_, StreamState>>> =
            (0..streams.len()).map(|_| None).collect();
        for i in order {
            guards[i] = Some(streams[i].lock()?);
        }
        let real: Vec<(&FakeStream, StreamGuard<'_, StreamState>)> = streams
            .iter()
            .zip(guards)
            .filter_map(|(s, g)| g.map(|g| (&***s, g)))
            .collect();
        // Fork: the other streams start behind streams[0], as a replay on streams[0] would.
        if real.len() > 1 {
            let marker = real[0].0.mark_held()?;
            for (s, _) in &real[1..] {
                s.await_held(marker)?;
            }
        }
        let mut seq = FakeSequence {
            rec: Recording::new(Mode::Record),
            streams: real.len(),
            real,
        };
        let result = f(&mut seq);
        let FakeSequence { rec, real, .. } = seq;
        // Join: streams[0] ends behind every other stream. Done even when `f` failed, so the streams stay consistent.
        let mut joined = Ok(());
        for (s, _) in real.iter().skip(1) {
            joined = joined.and_then(|()| real[0].0.await_held(s.mark_held()?));
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
        streams: &[&Arc<FakeStream>],
        f: F,
    ) -> Result<FakeGraph, GpuError>
    where
        F: FnOnce(&mut FakeSequence<'_>) -> Result<(), GpuError>,
    {
        let (rec, fingerprint) = self.record_pass(streams, f)?;
        Ok(FakeGraph {
            ops: rec.ops,
            args: rec.args,
            fingerprint,
            streams: streams.len(),
            kept: Mutex::new(rec.kept),
        })
    }

    pub(crate) fn rerecord_graph<F>(
        &self,
        graph: &mut FakeGraph,
        streams: &[&Arc<FakeStream>],
        f: F,
    ) -> Result<Rerecorded, GpuError>
    where
        F: FnOnce(&mut FakeSequence<'_>) -> Result<(), GpuError>,
    {
        let mut new = self.record_graph(streams, f)?;
        let how = if new.kernels() == graph.kernels() {
            Rerecorded::Updated
        } else {
            Rerecorded::Instantiated
        };
        // The old recording's kept buffers stay alive with the new one: replays of it may be in flight.
        new.kept
            .get_mut()
            .unwrap_or_else(PoisonError::into_inner)
            .append(graph.kept.get_mut().unwrap_or_else(PoisonError::into_inner));
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
