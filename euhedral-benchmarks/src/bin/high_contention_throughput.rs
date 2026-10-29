//! The high-contention throughput benchmark: average time and throughput of the control plane with many sources.
//!
//! ```text
//! high_contention_throughput [--forks N] [--sources N] [--warmup-iterations N] [--warmup-seconds S]
//!                            [--iterations N] [--seconds S] [--tasks N] [--no-pin] [--no-latency]
//! ```
//!
//! The defaults are upstream's: 3 forks (separate processes, each with its own lattice), 3 warm-up iterations of 3
//! seconds and 5 measured iterations of 5 seconds, each invocation waiting for 32,000,000 completions. The sources
//! default to the physical cores minus one; `EUHEDRAL_HCT_SOURCES` or `--sources` overrides them. Run it alone on the
//! machine: it keeps every core busy.

use euhedral_benchmarks::high_contention::{
    ForkError, Options, SOURCES_VARIABLE, run_fork, sources_from,
};
use euhedral_benchmarks::latency::Histogram;
use euhedral_benchmarks::report::{
    histogram_line, parse_histogram, parse_sample, sample_line, setup_line, summary_line,
};
use euhedral_hardware_utils::topology::Topology;
use std::io::{BufRead, BufReader};
use std::process::{Command, ExitCode, Stdio};
use std::time::Duration;

struct Arguments {
    forks: usize,
    child: bool,
    options: Options,
}

fn parse(arguments: impl Iterator<Item = String>) -> Result<Arguments, String> {
    let cores = Topology::system().core_count();
    let sources = sources_from(std::env::var(SOURCES_VARIABLE).ok().as_deref(), cores);
    let mut parsed = Arguments {
        forks: 3,
        child: false,
        options: Options::defaults(sources),
    };
    let mut arguments = arguments.peekable();
    while let Some(flag) = arguments.next() {
        let mut number = |name: &str| -> Result<f64, String> {
            arguments
                .next()
                .and_then(|value| value.parse::<f64>().ok())
                .filter(|value| value.is_finite() && *value >= 0.0)
                .ok_or_else(|| format!("{name} takes a number"))
        };
        match flag.as_str() {
            "--forks" => parsed.forks = number("--forks")? as usize,
            "--sources" => parsed.options.sources = (number("--sources")? as usize).max(1),
            "--warmup-iterations" => {
                parsed.options.warmup_iterations = number("--warmup-iterations")? as u32;
            }
            "--warmup-seconds" => {
                parsed.options.warmup_time = Duration::from_secs_f64(number("--warmup-seconds")?);
            }
            "--iterations" => parsed.options.iterations = number("--iterations")? as u32,
            "--seconds" => {
                parsed.options.iteration_time = Duration::from_secs_f64(number("--seconds")?);
            }
            "--tasks" => parsed.options.tasks = (number("--tasks")? as u64).max(1),
            "--no-pin" => parsed.options.pin_harness = false,
            "--no-latency" => parsed.options.latency = false,
            "--child" => parsed.child = true,
            "--help" | "-h" => return Err(String::new()),
            other => return Err(format!("unknown argument {other}")),
        }
    }
    Ok(parsed)
}

/// One fork, in this process: prints its setup and iterations.
fn child(options: &Options) -> Result<(), ForkError> {
    let outcome = run_fork(
        options,
        |setup| println!("{}", setup_line(setup.sources, &setup.isolation)),
        |sample| println!("{}", sample_line(sample)),
    )?;
    println!("{}", histogram_line(&outcome.measured_latency));
    Ok(())
}

/// The forks, each a process of this binary with `--child`.
fn drive(arguments: &Arguments) -> Result<(), String> {
    let executable = std::env::current_exe().map_err(|error| error.to_string())?;
    let mut samples = Vec::new();
    let mut setup = String::new();
    let mut latency = Histogram::new();
    for fork in 1..=arguments.forks {
        let options = &arguments.options;
        let mut process = Command::new(&executable)
            .arg("--child")
            .args(["--sources", &options.sources.to_string()])
            .args([
                "--warmup-iterations",
                &options.warmup_iterations.to_string(),
            ])
            .args([
                "--warmup-seconds",
                &options.warmup_time.as_secs_f64().to_string(),
            ])
            .args(["--iterations", &options.iterations.to_string()])
            .args([
                "--seconds",
                &options.iteration_time.as_secs_f64().to_string(),
            ])
            .args(["--tasks", &options.tasks.to_string()])
            .args(if options.pin_harness {
                vec![]
            } else {
                vec!["--no-pin"]
            })
            .stdout(Stdio::piped())
            .spawn()
            .map_err(|error| format!("cannot start fork {fork}: {error}"))?;
        let output = process.stdout.take().ok_or("no output from the fork")?;
        for line in BufReader::new(output).lines() {
            let line = line.map_err(|error| error.to_string())?;
            println!("fork {fork}: {line}");
            if let Some(rest) = line.strip_prefix("setup ") {
                rest.clone_into(&mut setup);
            } else if let Some(histogram) = parse_histogram(&line) {
                latency.merge(&histogram);
            } else if let Some(sample) = parse_sample(&line) {
                samples.push(sample);
            }
        }
        let status = process.wait().map_err(|error| error.to_string())?;
        if !status.success() {
            return Err(format!("fork {fork} failed: {status}"));
        }
    }
    println!(
        "{}",
        summary_line(arguments.forks, &setup, &samples, &latency)
    );
    Ok(())
}

fn main() -> ExitCode {
    let arguments = match parse(std::env::args().skip(1)) {
        Ok(arguments) => arguments,
        Err(message) => {
            if !message.is_empty() {
                eprintln!("{message}");
            }
            eprintln!(
                "usage: high_contention_throughput [--forks N] [--sources N] [--warmup-iterations N] \
                 [--warmup-seconds S] [--iterations N] [--seconds S] [--tasks N] [--no-pin] [--no-latency]"
            );
            return ExitCode::from(2);
        }
    };
    let outcome = if arguments.child {
        child(&arguments.options).map_err(|error| error.to_string())
    } else {
        drive(&arguments)
    };
    match outcome {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("{message}");
            ExitCode::FAILURE
        }
    }
}
