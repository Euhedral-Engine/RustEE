//! Streams as lane sources: [`LaneSubmit`] for the backends' streams, and [`open_lane_sources`], which opens a device's
//! lanes as a pool with its intakes and one [`LaneSource`] per lane.
//!
//! A lane's submitter has one owner, the lane's source; the lattice serializes the source's `request` and `pull`, so
//! the stream is only ever used by one worker at a time, and nothing is locked. A pool of several backends is built
//! with `LanePoolBuilder`, [`Gpu::open_lanes_at`], `LaneIntakes::new` and `LaneSource::new` instead.

use std::sync::Arc;

use euhedral_graph::{
    LaneId, LaneIntakes, LanePool, LanePoolBuilder, LaneSource, LaneState, LaneSubmit, LaneWork,
    Retired,
};

use crate::{Gpu, GpuError, Stream, StreamOptions};

/// One lane to open: its name, how its stream is set up, and the state its source keeps.
#[derive(Clone, Debug)]
pub struct LaneSpec<X> {
    pub name: String,
    pub options: StreamOptions,
    pub state: X,
}

/// The lanes [`open_lane_sources`] opened.
pub struct OpenedLanes<S: LaneSubmit, X: LaneState, W> {
    pub pool: Arc<LanePool>,
    pub intakes: Arc<LaneIntakes<W>>,
    /// Lane `i`'s source is `sources[i]`.
    pub sources: Vec<Arc<LaneSource<S, X, W>>>,
}

impl<S: LaneSubmit, X: LaneState, W> std::fmt::Debug for OpenedLanes<S, X, W> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenedLanes")
            .field("lanes", &self.sources.len())
            .finish()
    }
}

/// One pool of this device's lanes, its intakes and a source per lane (lane `i` is `lanes[i]`). `groups` lists the
/// lanes of each group that shares an intake. Lanes whose states differ in type share an enum `X`.
///
/// `Err(Invalid)` for no lanes, more than 64, a group lane that is not one of the lanes, or a lane in two groups.
pub fn open_lane_sources<G, X, W>(
    gpu: &Arc<G>,
    lanes: Vec<LaneSpec<X>>,
    groups: &[&[LaneId]],
) -> Result<OpenedLanes<G::Stream, X, W>, GpuError>
where
    G: Gpu,
    G::Stream: LaneSubmit<Error = GpuError>,
    X: LaneState,
    W: LaneWork<G::Stream, X>,
{
    if lanes.is_empty() || lanes.len() > 64 {
        return Err(GpuError::Invalid("a pool has 1 to 64 lanes".into()));
    }
    let mut member = vec![false; lanes.len()];
    for lane in groups.iter().flat_map(|group| group.iter()) {
        match member.get_mut(usize::from(lane.0)) {
            None => {
                return Err(GpuError::Invalid(format!(
                    "group lane {} is not one of the {} lanes",
                    lane.0,
                    lanes.len()
                )));
            }
            Some(taken) if *taken => {
                return Err(GpuError::Invalid(format!(
                    "lane {} is in two groups",
                    lane.0
                )));
            }
            Some(taken) => *taken = true,
        }
    }
    let names: Vec<&str> = lanes.iter().map(|l| l.name.as_str()).collect();
    let options: Vec<StreamOptions> = lanes.iter().map(|l| l.options).collect();
    let mut builder = LanePoolBuilder::new();
    let first = builder.reserve(lanes.len());
    let streams = gpu.open_lanes_at(first, &names, &options)?;
    for stream in &streams {
        builder.set(Stream::lane(stream), Stream::queue(stream));
    }
    let pool = builder.build();
    let (intakes, inputs) = LaneIntakes::new(pool.clone(), groups);
    let sources = streams
        .into_iter()
        .zip(lanes)
        .zip(inputs)
        .map(|((stream, spec), input)| LaneSource::new(stream, spec.state, input, intakes.clone()))
        .collect();
    Ok(OpenedLanes {
        pool,
        intakes,
        sources,
    })
}

/// `LaneSubmit` for a backend's stream: the methods delegate to the stream's own, with fully qualified calls.
macro_rules! lane_submit {
    ($stream:ty, $raw:ty, |$s:ident| $raw_of:expr) => {
        impl LaneSubmit for $stream {
            type Error = GpuError;
            type Raw = $raw;

            fn lane(&self) -> LaneId {
                Stream::lane(self)
            }

            fn id(&self) -> euhedral_graph::QueueId {
                Stream::id(self)
            }

            fn queue(&self) -> Arc<dyn euhedral_graph::DeviceQueue> {
                Stream::queue(self)
            }

            fn accepts(&self, marker: euhedral_graph::Marker) -> bool {
                self.owns_marker_lane(marker)
            }

            fn await_marker(&mut self, marker: euhedral_graph::Marker) -> Result<(), GpuError> {
                Stream::await_marker(self, marker)
            }

            /// A full marker ring (or a slot an awaiter holds) is `Ok(None)`: the lane source stalls the lane and
            /// marks the item at its next entry.
            fn mark(&mut self) -> Result<Option<euhedral_graph::Marker>, GpuError> {
                match Stream::mark(self) {
                    Ok(marker) => Ok(Some(marker)),
                    Err(GpuError::Busy) => Ok(None),
                    Err(error) => Err(error),
                }
            }

            fn arm_retirement(&mut self) -> Result<u64, GpuError> {
                // The retirement armed next gets the number the stream's retirement queue counts next.
                let seq = self.retire.next_seq();
                Stream::arm_retirement(self)?;
                Ok(seq)
            }

            /// When no callback has arrived and retirements are outstanding, the stream looks at the device once: a
            /// retirement behind a lane that faulted never gets its callback, and the look is what poisons the
            /// device so the rest are reported. A retirement that arrived with status 0 is confirmed with the same
            /// look, as Java's `confirmRetired` queries the completion event.
            fn poll_retired(&mut self) -> Option<Retired<GpuError>> {
                let ticket = match Stream::poll_retired(self) {
                    Some(ticket) => ticket,
                    None => {
                        self.probe_fault(true);
                        Stream::poll_retired(self)?
                    }
                };
                if ticket.ok() {
                    self.probe_fault(false);
                }
                let failure = Stream::confirm_retired(self, ticket);
                Some(Retired {
                    seq: ticket.seq(),
                    failure,
                })
            }

            /// # Safety
            ///
            /// As [`LaneSubmit::raw`]: the caller enqueues only from inside an item of this lane, a mark afterwards
            /// covers the work, and the caller makes the backend's context current (for CUDA, `Device::enter`)
            /// before it calls driver functions with the handle.
            unsafe fn raw(&mut self) -> $raw {
                let $s = self;
                $raw_of
            }
        }
    };
}

#[cfg(feature = "cuda")]
lane_submit!(
    crate::cuda::CudaStream,
    euhedral_memory::cuda::CuStream,
    |s| s.raw
);

#[cfg(feature = "fake")]
lane_submit!(crate::fake::FakeStream, u64, |s| Stream::id(s).get());
