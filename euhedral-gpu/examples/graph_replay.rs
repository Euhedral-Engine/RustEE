//! The acceptance program: kernels that belong to this program compiled at run time, two streams joined by a
//! marker, the sequence recorded as a graph and replayed 100 times with changing arguments, and a lattice frame
//! that follows each replay's completion. The results are checked against a CPU reference.
//!
//! - The kernels are in `kernels/pipeline.cu`, outside the crate's `src`, and are compiled with
//!   `CudaModule::from_source`. Compiled modules are cached in `$EUHEDRAL_GPU_CACHE` when it is set, else in a
//!   `gpu-module-cache` directory beside this executable.
//! - Nothing on the lattice waits for the device, and nothing is locked: a replay's completion arrives as a frame,
//!   which checks the result, updates the arguments, submits the next replay and arms its retirement. Lane a's
//!   submitter and the graph have one owner at a time: each continuation frame owns them, uses them, and hands them to
//!   the next (`retire_into` carries the stream with its continuation). The main thread waits on a latch for the last
//!   frame, which hands them back so that dropping them (which waits for the device) happens there.
//!
//! Run with `cargo run -p euhedral-gpu --features cuda --example graph_replay`; NVRTC is found through
//! `EUHEDRAL_NVRTC` or `CUDA_HOME`.

use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;

use euhedral_core::config::LatticeConfig;
use euhedral_core::control_plane::ControlPlaneLattice;
use euhedral_core::frames::Frame;
use euhedral_core::generics::LatticeSource;
use euhedral_core::impls::DefaultExecutor;
use euhedral_gpu::cuda::{CudaGpu, CudaGraph, CudaKernel, CudaModule, CudaStream};
use euhedral_gpu::{
    Args, CompileOptions, Enqueue, Gpu, GpuError, Graph, Launch, Module, OnceFrame, Sequence,
    Stream, StreamOptions, retire_into, retire_into_with,
};
use euhedral_graph::{Completions, Latch, LatchSetter};
use euhedral_hardware_utils::affinity;
use euhedral_hardware_utils::util::CpuSet;
use euhedral_memory::cuda::{DeviceMemory, PinnedHost};
use euhedral_memory::{Backend, Block, LocationId};

/// Elements per buffer.
const N: usize = 1 << 18;
const BYTES: usize = N * 4;
/// Replays after the recording's own run.
const REPLAYS: u32 = 100;
/// The routing hash of the continuation frames: one ordered lane.
const LANE: u64 = 0x6772_6170;

/// The arguments of iteration `i`.
fn multiplier(i: u32) -> u32 {
    3 + i
}

fn mask(i: u32) -> u32 {
    0x9e37_79b9 ^ i
}

/// The answer the device must give for iteration `i`, computed on the CPU.
fn reference(x: &[u32], i: u32) -> Vec<u32> {
    x.iter()
        .enumerate()
        .map(|(k, &x)| {
            let y = x.wrapping_mul(multiplier(i)).wrapping_add(k as u32);
            (y ^ mask(i)).wrapping_add(y >> 3)
        })
        .collect()
}

/// What the continuation frames share, read-only.
struct Shared {
    mix: CudaKernel,
    fold: CudaKernel,
    x: Block,
    y: Block,
    z: Block,
    /// Pinned host memory the sequence copies its result to.
    out: Block,
    x_host: Vec<u32>,
    gpu: Arc<CudaGpu>,
    completions: Arc<Completions>,
}

/// What the chain hands back to the main thread when it ends: its result, and the pieces it owned, so that dropping
/// them (which waits for the device) happens on the main thread.
struct Outcome {
    result: Result<(), String>,
    parts: Option<(CudaStream, CudaGraph)>,
}

/// The way out of the chain. If the chain is dropped without finishing (a retirement that could not be armed, a
/// lattice that closed early), the guard still ends the main thread's wait.
struct Finish(Option<LatchSetter<Outcome>>);

impl Finish {
    fn with(mut self, outcome: Outcome) {
        if let Some(setter) = self.0.take() {
            setter.set(outcome);
        }
    }
}

impl Drop for Finish {
    fn drop(&mut self) {
        if let Some(setter) = self.0.take() {
            setter.set(Outcome {
                result: Err("the chain ended before it finished".into()),
                parts: None,
            });
        }
    }
}

/// Lane a's state, owned by one frame at a time: each continuation receives it, uses it, and hands it to the next.
/// Nothing in it is shared or locked.
struct Chain {
    shared: Arc<Shared>,
    stream: CudaStream,
    graph: CudaGraph,
    finish: Finish,
}

impl Shared {
    /// The recurring sequence for iteration `i`: stream 0 mixes, stream 1 folds behind it, and stream 0 copies the
    /// result out behind that.
    fn sequence<S: Sequence<Kernel = CudaKernel>>(
        &self,
        q: &mut S,
        i: u32,
    ) -> Result<(), GpuError> {
        let mix = Args::new()
            .device(self.y.addr)
            .device(self.x.addr)
            .arg(multiplier(i))
            .arg(N as u32);
        let fold = Args::new()
            .device(self.z.addr)
            .device(self.y.addr)
            .arg(mask(i))
            .arg(N as u32);
        let geometry = Launch::linear(N as u32, 256);
        q.submit(0, |e| {
            // SAFETY: the arguments match `mix`; the buffers live until the program ends, and each replay is
            // submitted after the previous one retired.
            unsafe { e.launch(&self.mix, geometry, &mix) }
        })?;
        q.order(0, 1)?;
        q.submit(1, |e| {
            // SAFETY: as above, for `fold`.
            unsafe { e.launch(&self.fold, geometry, &fold) }
        })?;
        q.order(1, 0)?;
        q.submit(0, |e| {
            // SAFETY: `out` is pinned host memory of BYTES bytes, read only after the retirement.
            unsafe { e.copy_to_host(self.out.addr as *mut u8, self.z.addr, BYTES) }
        })
    }
}

/// Arms the retirement of iteration `i`. The chain travels with its continuation: the frame the lattice's own poll
/// emits owns the stream and the graph, so no lock is taken anywhere on the path. Dropping a stream waits for the
/// device, so the lattice never drops one: a finished or failed chain hands its pieces to the main thread.
fn arm(chain: Chain, i: u32) {
    let Chain {
        shared,
        stream,
        graph,
        finish,
    } = chain;
    let completions = shared.completions.clone();
    let armed = retire_into_with(
        stream,
        (shared, graph, finish),
        &completions,
        move |stream, (shared, graph, finish), ticket| {
            let chain = Chain {
                shared,
                stream,
                graph,
                finish,
            };
            let frame: Box<dyn Frame> =
                Box::new(OnceFrame::new(LANE, move || step(chain, i, ticket)));
            frame
        },
    );
    if let Err(e) = armed {
        // Arming failed: the stream and everything that travelled with it come back, and go home to the main thread.
        let (_shared, graph, finish) = e.state;
        let result = Err(format!("arming a retirement failed: {}", e.error));
        finish.with(Outcome {
            result,
            parts: Some((e.stream, graph)),
        });
    }
}

/// Iteration `i` retired: check it, then submit the next replay with its arguments, or finish.
fn step(chain: Chain, i: u32, ticket: euhedral_gpu::RetireTicket) {
    let Chain {
        shared,
        mut stream,
        mut graph,
        finish,
    } = chain;
    let end = |finish: Finish, stream, graph, result| {
        finish.with(Outcome {
            result,
            parts: Some((stream, graph)),
        });
    };
    if let Some(error) = stream.confirm_retired(ticket) {
        return end(
            finish,
            stream,
            graph,
            Err(format!("iteration {i}: {error}")),
        );
    }
    // SAFETY: `out` holds N words the retired copy wrote; nothing writes it until the next replay, which is
    // submitted below.
    let got = unsafe { std::slice::from_raw_parts(shared.out.addr as *const u32, N) };
    if got != reference(&shared.x_host, i) {
        let message = format!("iteration {i} differs from the CPU reference");
        return end(finish, stream, graph, Err(message));
    }
    if i == REPLAYS {
        return end(finish, stream, graph, Ok(()));
    }
    let next = i + 1;
    let submitted = shared
        .gpu
        .update(&mut graph, |q| shared.sequence(q, next))
        .and_then(|()| graph.replay(&mut stream));
    match submitted {
        Ok(()) => arm(
            Chain {
                shared,
                stream,
                graph,
                finish,
            },
            next,
        ),
        Err(e) => end(finish, stream, graph, Err(e.to_string())),
    }
}

fn cache_dir() -> PathBuf {
    std::env::var_os("EUHEDRAL_GPU_CACHE")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            std::env::current_exe()
                .ok()
                .and_then(|exe| exe.parent().map(|dir| dir.join("gpu-module-cache")))
                .unwrap_or_else(|| PathBuf::from("gpu-module-cache"))
        })
}

fn run() -> Result<(), String> {
    let gpu = CudaGpu::open(0).map_err(|e| format!("no CUDA device: {e}"))?;
    let started = std::time::Instant::now();
    let module = CudaModule::from_source(
        &gpu,
        include_str!("kernels/pipeline.cu"),
        &CompileOptions {
            name: "pipeline.cu".into(),
            cache_dir: Some(cache_dir()),
            ..CompileOptions::default()
        },
    )
    .map_err(|e| e.to_string())?;
    println!("kernels ready in {:?}", started.elapsed());
    let (mix, fold) = (
        module.kernel("mix").map_err(|e| e.to_string())?,
        module.kernel("fold").map_err(|e| e.to_string())?,
    );

    // The lattice: one CPU of this process, and the completions it polls.
    let cpu = affinity::base_mask()
        .first()
        .ok_or("the process may run on no CPU")?;
    let mut config = LatticeConfig::with_executor(DefaultExecutor::new());
    config.name = "graph-replay".into();
    config.shard_name = "graph-replay".into();
    config.allowed_cpus = CpuSet::range(cpu, cpu + 1);
    config.startup_timeout = Duration::from_secs(10);
    config.shutdown_timeout = Duration::from_secs(5);
    let lattice = ControlPlaneLattice::new(config).map_err(|e| format!("{e:?}"))?;
    let (pool, streams) = gpu
        .open_lanes(&["compute-a", "compute-b"], StreamOptions::default())
        .map_err(|e| e.to_string())?;
    let mut streams = streams.into_iter();
    let (mut lane_a, lane_b) = (
        streams.next().ok_or("no lane a")?,
        streams.next().ok_or("no lane b")?,
    );
    let completions = Completions::new(pool);
    lattice
        .add_upstream(completions.clone())
        .map_err(|e| format!("{e:?}"))?;

    // Buffers: device x, y, z, and pinned host memory for the input and the output.
    let mem = DeviceMemory {
        device: gpu.device().clone(),
        location: LocationId(0),
    };
    let pinned = PinnedHost::new(gpu.device().clone(), LocationId(1), 0);
    let alloc = |b: &dyn Backend| b.alloc(BYTES as u64).map_err(|e| e.to_string());
    let (x, y, z) = (alloc(&mem)?, alloc(&mem)?, alloc(&mem)?);
    let (staging, out) = (alloc(&pinned)?, alloc(&pinned)?);
    let x_host: Vec<u32> = (0..N as u32)
        .map(|k| k.wrapping_mul(2_654_435_761))
        .collect();
    // SAFETY: a pinned block of BYTES bytes that only this thread touches until the upload retires.
    unsafe { std::slice::from_raw_parts_mut(staging.addr as *mut u32, N) }.copy_from_slice(&x_host);

    let shared = Arc::new(Shared {
        mix,
        fold,
        x,
        y,
        z,
        out,
        x_host,
        gpu: gpu.clone(),
        completions: completions.clone(),
    });

    // The input upload. Its completion is a frame that carries lane a back to the main thread, so the main thread
    // polls nothing and lane a is never shared.
    lane_a
        .submit(|q| {
            // SAFETY: valid blocks; the staging block is not touched until the upload retires.
            unsafe { q.copy_to_device(x.addr, staging.addr as *const u8, BYTES) }
        })
        .map_err(|e| e.to_string())?;
    let (uploaded_set, uploaded) = Latch::new();
    retire_into(lane_a, &completions, move |mut stream, ticket| {
        let failure = stream.confirm_retired(ticket);
        Box::new(OnceFrame::new(LANE, move || {
            uploaded_set.set((stream, failure));
        }))
    })
    .map_err(|e| e.error.to_string())?;
    let (mut lane_a, failure) = uploaded.wait();
    if let Some(error) = failure {
        return Err(format!("the upload: {error}"));
    }

    // Iteration 0 is the recording's own run, on both lanes. Lane b is needed only to record.
    let mut lane_b = lane_b;
    let graph = gpu
        .record(&mut [&mut lane_a, &mut lane_b], |q| shared.sequence(q, 0))
        .map_err(|e| e.to_string())?;
    drop(lane_b);
    let (setter, waiter) = Latch::new();
    arm(
        Chain {
            shared: shared.clone(),
            stream: lane_a,
            graph,
            finish: Finish(Some(setter)),
        },
        0,
    );

    let outcome = waiter.wait();

    // Shut down: nothing is pending once the last frame ran. The streams and the graph come home to be dropped here.
    let (p50, p99, max) = completions.lateness_us();
    LatticeSource::complete(&*completions);
    lattice.close();
    drop(outcome.parts);
    drop(shared);
    // SAFETY: the streams ended their work as they dropped; nothing uses the blocks.
    unsafe {
        for b in [x, y, z] {
            mem.free(b);
        }
        pinned.free(staging);
        pinned.free(out);
    }
    outcome.result?;
    println!(
        "{REPLAYS} replays matched the CPU reference; lateness p50/p99/max µs: {p50}/{p99}/{max}"
    );
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("graph_replay failed: {message}");
            ExitCode::FAILURE
        }
    }
}
