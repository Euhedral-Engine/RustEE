//! `bazel run //:gazelle` (or `cargo run -p euhedral-bazel-gen`): writes every BUILD file from the Cargo manifests.
//! `-- -mode=diff` writes nothing and fails when a file is stale.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use euhedral_bazel_gen::{emit, load_workspace, stale};

fn main() -> ExitCode {
    match run() {
        Ok(code) => code,
        Err(message) => {
            eprintln!("euhedral-bazel-gen: {message}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<ExitCode, String> {
    let mut diff = false;
    for arg in std::env::args().skip(1) {
        match arg.trim_start_matches('-') {
            "mode=diff" => diff = true,
            "mode=fix" => diff = false,
            _ => {
                return Err(format!(
                    "unknown argument `{arg}` (expected -mode=fix or -mode=diff)"
                ));
            }
        }
    }
    // `bazel run` starts in the runfiles tree and names the workspace in BUILD_WORKSPACE_DIRECTORY.
    let cwd = std::env::current_dir().map_err(|e| e.to_string())?;
    let root = std::env::var_os("BUILD_WORKSPACE_DIRECTORY")
        .map(PathBuf::from)
        .unwrap_or(cwd.clone());
    let cargo = match std::env::var_os("CARGO") {
        // Under `bazel run` the toolchain's cargo is a runfiles path, relative to where the binary starts.
        Some(path) if Path::new(&path).components().count() > 1 => {
            cwd.join(path).display().to_string()
        }
        Some(path) => path.to_string_lossy().into_owned(),
        None => "cargo".to_owned(),
    };
    let ws = load_workspace(&root, &cargo, None)?;
    let files = emit::generate(&ws)?;
    let read = |path: &str| std::fs::read_to_string(root.join(path)).ok();
    let stale = stale(&files, &read);
    if diff {
        if stale.is_empty() {
            return Ok(ExitCode::SUCCESS);
        }
        for path in &stale {
            eprintln!("stale: {path}");
        }
        eprintln!("regenerate with `bazel run //:gazelle` (or `cargo run -p euhedral-bazel-gen`)");
        return Ok(ExitCode::FAILURE);
    }
    for path in &stale {
        std::fs::write(root.join(path), &files[path])
            .map_err(|e| format!("writing {path}: {e}"))?;
        println!("wrote {path}");
    }
    Ok(ExitCode::SUCCESS)
}
