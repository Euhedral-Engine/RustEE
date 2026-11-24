//! Fails when a BUILD file or `MODULE.bazel.lock` is out of date with the Cargo manifests and target files: a manifest or a
//! target was added, removed or changed without regenerating. Runs under Cargo and Bazel alike.

use std::path::PathBuf;

use euhedral_bazel_gen::{emit, load_workspace, lock_problems, locked_inputs, stale};

/// The workspace root: under Bazel, the runfiles copy of `//:workspace_sources`.
fn root() -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    if dir.is_absolute() {
        dir
    } else {
        std::env::current_dir()
            .expect("the test's directory")
            .join(dir)
    }
}

/// Bazel's sandbox has no writable home: cargo gets its own under the test's scratch directory.
fn cargo_home() -> Option<PathBuf> {
    std::env::var_os("TEST_TMPDIR").map(|dir| PathBuf::from(dir).join("cargo-home"))
}

fn cargo() -> String {
    match std::env::var("CARGO") {
        Ok(path) if !PathBuf::from(&path).is_absolute() && path.contains('/') => {
            std::env::current_dir()
                .expect("the test's directory")
                .join(path)
                .display()
                .to_string()
        }
        Ok(path) => path,
        Err(_) => "cargo".to_owned(),
    }
}

#[test]
fn the_build_files_are_current() {
    let root = root();
    let ws = load_workspace(&root, &cargo(), cargo_home().as_deref()).expect("the workspace loads");
    let files = emit::generate(&ws).expect("the generator accepts the workspace");
    let stale = stale(&files, &|path| {
        std::fs::read_to_string(root.join(path)).ok()
    });
    assert!(
        stale.is_empty(),
        "stale BUILD files {stale:?}: regenerate with `bazel run //:gazelle` (or `cargo run -p euhedral-bazel-gen`)"
    );
}

#[test]
fn the_module_lock_records_the_current_manifests() {
    let root = root();
    let ws = load_workspace(&root, &cargo(), cargo_home().as_deref()).expect("the workspace loads");
    let lock = std::fs::read_to_string(root.join("MODULE.bazel.lock")).expect("MODULE.bazel.lock");
    let problems = lock_problems(
        &lock,
        &|path| std::fs::read(root.join(path)).ok(),
        &locked_inputs(&ws),
    );
    assert!(
        problems.is_empty(),
        "MODULE.bazel.lock is out of date ({problems:?}): run `bazel mod deps --lockfile_mode=update`"
    );
}
