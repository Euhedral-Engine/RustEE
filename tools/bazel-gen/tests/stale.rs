//! Fails when a BUILD file or `MODULE.bazel.lock` is out of date with the Cargo manifests and target files: a manifest or a
//! target was added, removed or changed without regenerating. Runs under Cargo and Bazel alike.

use std::path::PathBuf;

use euhedral_bazel_gen::{
    build_files_on_disk, emit, extra_build_files, load_workspace, lock_problems, locked_inputs,
    stale,
};

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

/// A directory the test may write: Bazel's TEST_TMPDIR, or Cargo's target scratch directory.
fn scratch() -> PathBuf {
    std::env::var_os("TEST_TMPDIR")
        .map(PathBuf::from)
        .or(option_env!("CARGO_TARGET_TMPDIR").map(PathBuf::from))
        .expect("Bazel or Cargo names a scratch directory")
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
fn no_build_file_is_left_that_the_generator_does_not_write() {
    let root = root();
    let ws = load_workspace(&root, &cargo(), cargo_home().as_deref()).expect("the workspace loads");
    let files = emit::generate(&ws).expect("the generator accepts the workspace");
    let extra = extra_build_files(
        &files,
        &build_files_on_disk(&root).expect("the workspace lists"),
    );
    assert!(
        extra.is_empty(),
        "BUILD files the generator does not write {extra:?}: remove them (a removed member's, or a hand-added package)"
    );
}

#[test]
fn the_disk_walk_skips_what_bazel_ignores() {
    let dir = scratch().join("build-walk");
    let _ = std::fs::remove_dir_all(&dir);
    for path in [
        "BUILD.bazel",
        "a/src/BUILD",
        "target/x/BUILD.bazel",
        ".git/BUILD",
        "a/README.md",
    ] {
        let file = dir.join(path);
        std::fs::create_dir_all(file.parent().expect("a parent")).expect("mkdir");
        std::fs::write(file, "").expect("write");
    }
    std::fs::write(dir.join(".bazelignore"), "# output\ntarget/\n").expect("write");
    assert_eq!(
        build_files_on_disk(&dir).expect("walks"),
        ["BUILD.bazel", "a/src/BUILD"]
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
