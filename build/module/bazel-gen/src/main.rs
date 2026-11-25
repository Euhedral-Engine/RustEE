//! `bazel run //:gazelle`: writes every BUILD file from the Cargo manifests. `-- -mode=diff` writes nothing and fails when a
//! file is stale; `bazel test //:bazel_gen_stale_test` runs that check (README, "Commands").
//!
//! Arguments, each also read from the environment (`BAZEL_GEN_MODE`, `BAZEL_GEN_CONFIG`, `BAZEL_GEN_LOCK`) so a test rule
//! can set them:
//! - `-mode=fix|diff`;
//! - `-config=PATH`: the configuration, instead of the root's `bazel_gen.toml`;
//! - `-lock=PATH`: also fail when the lock at PATH (root-relative) is stale: `MODULE.bazel.lock`'s record of the manifests,
//!   or, when PATH is the configuration's `crate_lock`, crate_universe's own lock against `Cargo.lock`.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use euhedral_bazel_gen::{
    build_files_on_disk, crate_lock_problems, emit, extra_build_files, load_workspace,
    lock_problems, locked_inputs, stale,
};

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
    let env = |name: &str| std::env::var(name).ok().filter(|v| !v.is_empty());
    let mut mode = env("BAZEL_GEN_MODE");
    let mut config = env("BAZEL_GEN_CONFIG");
    let mut lock = env("BAZEL_GEN_LOCK");
    for arg in std::env::args().skip(1) {
        let arg = arg.trim_start_matches('-');
        if let Some(value) = arg.strip_prefix("mode=") {
            mode = Some(value.to_owned());
        } else if let Some(value) = arg.strip_prefix("config=") {
            config = Some(value.to_owned());
        } else if let Some(value) = arg.strip_prefix("lock=") {
            lock = Some(value.to_owned());
        } else {
            return Err(format!(
                "unknown argument `{arg}` (expected -mode=fix|diff, -config=PATH or -lock=PATH)"
            ));
        }
    }
    let diff = match mode.as_deref() {
        None | Some("fix") => false,
        Some("diff") => true,
        Some(other) => return Err(format!("unknown mode `{other}` (fix or diff)")),
    };
    // `bazel run` starts in the runfiles tree and names the workspace in BUILD_WORKSPACE_DIRECTORY; `bazel test` starts at
    // the runfiles root of the main repository, which holds the workspace's sources.
    let cwd = std::env::current_dir().map_err(|e| e.to_string())?;
    let root = std::env::var_os("BUILD_WORKSPACE_DIRECTORY")
        .map(PathBuf::from)
        .unwrap_or(cwd.clone());
    // Under Bazel the toolchain's cargo is a runfiles path, relative to where the binary starts.
    let from_cwd = |path: &str| -> String {
        if Path::new(path).components().count() > 1 {
            cwd.join(path).display().to_string()
        } else {
            path.to_owned()
        }
    };
    let cargo = env("CARGO").map_or_else(|| "cargo".to_owned(), |c| from_cwd(&c));
    // A test's sandbox has no writable home: cargo gets its own under the test's scratch directory.
    let cargo_home = env("TEST_TMPDIR").map(|dir| PathBuf::from(dir).join("cargo-home"));
    let config = config.map(|c| cwd.join(c));
    let ws = load_workspace(&root, &cargo, cargo_home.as_deref(), config.as_deref())?;
    let files = emit::generate(&ws)?;
    let read = |path: &str| std::fs::read_to_string(root.join(path)).ok();
    let stale = stale(&files, &read);
    let extra = extra_build_files(&files, &build_files_on_disk(&root)?);
    for path in &extra {
        eprintln!("not generated: {path} (remove it: a removed member's, or a hand-added package)");
    }
    let mut lock_stale = Vec::new();
    if let Some(lock) = &lock {
        let text =
            std::fs::read_to_string(root.join(lock)).map_err(|e| format!("reading {lock}: {e}"))?;
        let crate_lock = ws.config.crate_lock.as_deref() == Some(lock.as_str());
        let repin = if crate_lock {
            // crate_universe's own lock: compare it with Cargo.lock.
            let cargo_lock = std::fs::read_to_string(root.join("Cargo.lock"))
                .map_err(|e| format!("reading Cargo.lock: {e}"))?;
            lock_stale =
                crate_lock_problems(&cargo_lock, &text).map_err(|e| format!("{lock}: {e}"))?;
            "CARGO_BAZEL_REPIN=1 bazel mod deps --lockfile_mode=update"
        } else {
            let required = locked_inputs(&ws);
            lock_stale = lock_problems(
                &text,
                &|path| std::fs::read(root.join(path)).ok(),
                &required,
            );
            if lock_stale.len() >= required.len()
                && lock_stale.iter().all(|p| p.ends_with("is not recorded"))
            {
                eprintln!(
                    "{lock} records none of the manifests: a crate hub with `lockfile =` is checked against that \
                     lock instead (set `[root] crate_lock` in bazel_gen.toml)"
                );
            }
            "bazel mod deps --lockfile_mode=update"
        };
        for problem in &lock_stale {
            eprintln!("{lock}: {problem} (update it with `{repin}`)");
        }
    }
    if diff {
        if stale.is_empty() && extra.is_empty() && lock_stale.is_empty() {
            return Ok(ExitCode::SUCCESS);
        }
        for path in &stale {
            eprintln!("stale: {path}");
        }
        if !stale.is_empty() {
            eprintln!("regenerate with `bazel run //:gazelle`");
        }
        return Ok(ExitCode::FAILURE);
    }
    for path in &stale {
        std::fs::write(root.join(path), &files[path])
            .map_err(|e| format!("writing {path}: {e}"))?;
        println!("wrote {path}");
    }
    Ok(if extra.is_empty() && lock_stale.is_empty() {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    })
}
