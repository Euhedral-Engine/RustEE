#![doc = include_str!("../README.md")]

pub mod cfg;
pub mod emit;
pub mod json;
pub mod model;
pub mod starlark;
pub mod worlds;

#[cfg(test)]
mod fixture_tests;

use std::collections::BTreeMap;
use std::path::Path;
use std::process::Command;

use crate::model::Workspace;

/// Reads the workspace at `root` with `cargo metadata --no-deps` (run with `cargo`, and `CARGO_HOME` when given) and the
/// members' manifests.
pub fn load_workspace(
    root: &Path,
    cargo: &str,
    cargo_home: Option<&Path>,
) -> Result<Workspace, String> {
    let mut command = Command::new(cargo);
    if let Some(home) = cargo_home {
        command.env("CARGO_HOME", home);
    }
    let output = command
        .args([
            "metadata",
            "--no-deps",
            "--format-version",
            "1",
            "--offline",
            "--manifest-path",
        ])
        .arg(root.join("Cargo.toml"))
        .output()
        .map_err(|e| format!("running `{cargo} metadata`: {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "`cargo metadata` failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    let text =
        String::from_utf8(output.stdout).map_err(|_| "cargo metadata printed invalid UTF-8")?;
    let metadata = json::parse(&text)?;
    let mut ws = model::load(&metadata, &|path| {
        std::fs::read_to_string(root.join(path)).map_err(|e| format!("reading {path}: {e}"))
    })?;
    for package in &mut ws.packages {
        let mut found = Vec::new();
        let mut files = Vec::new();
        rust_files(&root.join(&package.dir), "src", &mut files)?;
        for file in files {
            let text = std::fs::read_to_string(root.join(&package.dir).join(&file))
                .map_err(|e| format!("reading {}/{file}: {e}", package.dir))?;
            found.extend(
                model::path_attributes(&file, &text)?
                    .into_iter()
                    .filter(|p| !p.starts_with("src/")),
            );
        }
        found.sort();
        found.dedup();
        package.path_modules = found;
    }
    Ok(ws)
}

/// Every `.rs` file under `dir` (package-relative, `/`-separated), in `out`.
fn rust_files(package: &Path, dir: &str, out: &mut Vec<String>) -> Result<(), String> {
    let entries = match std::fs::read_dir(package.join(dir)) {
        Ok(entries) => entries,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(format!("listing {dir}: {e}")),
    };
    for entry in entries {
        let entry = entry.map_err(|e| e.to_string())?;
        let name = entry.file_name().to_string_lossy().into_owned();
        let path = format!("{dir}/{name}");
        // Runfiles trees hold symlinks: follow them.
        let meta = std::fs::metadata(entry.path()).map_err(|e| format!("{path}: {e}"))?;
        if meta.is_dir() {
            rust_files(package, &path, out)?;
        } else if name.ends_with(".rs") {
            out.push(path);
        }
    }
    Ok(())
}

/// The generated files that differ from what `read` returns for their path (a missing file differs), sorted.
pub fn stale(
    expected: &BTreeMap<String, String>,
    read: &dyn Fn(&str) -> Option<String>,
) -> Vec<String> {
    expected
        .iter()
        .filter(|(path, content)| read(path).as_deref() != Some(content.as_str()))
        .map(|(path, _)| path.clone())
        .collect()
}

/// The BUILD files on disk that the generator does not write: one left by a removed member, or a hand-added one, which
/// would cut its parent package's globs. Sorted.
pub fn extra_build_files(expected: &BTreeMap<String, String>, on_disk: &[String]) -> Vec<String> {
    let mut extra: Vec<String> = on_disk
        .iter()
        .filter(|path| !expected.contains_key(*path))
        .cloned()
        .collect();
    extra.sort();
    extra
}

/// Every `BUILD.bazel` or `BUILD` file under `root` (root-relative, `/`-separated) that Bazel would read as a package:
/// the paths `.bazelignore` names, `.git`, every `target` directory next to a `Cargo.toml` (Cargo's output; a
/// `target` source directory is a package) and symlinked directories (the `bazel-*` output links) are skipped. A
/// directory that cannot be listed is an error naming it: skipping it could hide a BUILD file the generator should own.
pub fn build_files_on_disk(root: &Path) -> Result<Vec<String>, String> {
    let ignored: Vec<String> = match std::fs::read_to_string(root.join(".bazelignore")) {
        Ok(text) => text
            .lines()
            .map(|line| line.trim().trim_end_matches('/').to_owned())
            .filter(|line| !line.is_empty() && !line.starts_with('#'))
            .collect(),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Vec::new(),
        Err(e) => return Err(format!("reading .bazelignore: {e}")),
    };
    let mut out = Vec::new();
    let mut dirs = vec![String::new()];
    while let Some(dir) = dirs.pop() {
        let entries =
            std::fs::read_dir(root.join(&dir)).map_err(|e| {
                let shown = if dir.is_empty() { "." } else { dir.as_str() };
                format!(
                    "cannot read directory `{shown}` ({e}): fix its permissions or list it in .bazelignore, since a BUILD file inside it could not be checked"
                )
            })?;
        for entry in entries {
            let entry = entry.map_err(|e| e.to_string())?;
            let name = entry.file_name().to_string_lossy().into_owned();
            let path = if dir.is_empty() {
                name.clone()
            } else {
                format!("{dir}/{name}")
            };
            let kind = entry.file_type().map_err(|e| format!("{path}: {e}"))?;
            if kind.is_dir() {
                let cargo_output = name == "target" && root.join(&dir).join("Cargo.toml").is_file();
                if name != ".git" && !cargo_output && !ignored.contains(&path) {
                    dirs.push(path);
                }
            } else if (name == "BUILD.bazel" || name == "BUILD")
                && (kind.is_file() || std::fs::metadata(entry.path()).is_ok_and(|m| m.is_file()))
            {
                out.push(path);
            }
        }
    }
    out.sort();
    Ok(out)
}

/// The `name version` of every package in `Cargo.lock` that comes from a registry.
pub fn registry_packages(cargo_lock: &str) -> Vec<String> {
    let mut found = Vec::new();
    let (mut name, mut version, mut registry) = (None, None, false);
    let mut close =
        |name: &mut Option<String>, version: &mut Option<String>, registry: &mut bool| {
            if let (Some(n), Some(v), true) = (name.take(), version.take(), *registry) {
                found.push(format!("{n} {v}"));
            }
            *registry = false;
        };
    for line in cargo_lock.lines() {
        if line.starts_with("[[package]]") {
            close(&mut name, &mut version, &mut registry);
        } else if let Some(v) = line.strip_prefix("name = ") {
            name = Some(v.trim_matches('"').to_owned());
        } else if let Some(v) = line.strip_prefix("version = ") {
            version = Some(v.trim_matches('"').to_owned());
        } else if line.starts_with("source = \"registry+") {
            registry = true;
        }
    }
    close(&mut name, &mut version, &mut registry);
    found.sort();
    found
}

/// What is wrong with `cargo-bazel-lock.json` next to `Cargo.lock`: a registry package of `Cargo.lock` the lock has no
/// crate for, or a crate of the lock (a workspace member aside) that `Cargo.lock` lacks. crate_universe's own digest does
/// not cover `Cargo.lock`, so a `cargo update` without a repin would otherwise build the old versions without a word.
/// Empty when they agree.
pub fn lock_problems(cargo_lock: &str, bazel_lock: &str) -> Result<Vec<String>, String> {
    let bazel = json::parse(bazel_lock)?;
    let keys = |field: &str| -> Result<Vec<String>, String> {
        let object = bazel
            .get(field)
            .and_then(json::Json::as_object)
            .ok_or_else(|| format!("cargo-bazel-lock.json has no `{field}` object"))?;
        Ok(object.keys().cloned().collect())
    };
    let members = keys("workspace_members")?;
    let locked: Vec<String> = keys("crates")?
        .into_iter()
        .filter(|key| !members.contains(key))
        .collect();
    let wanted = registry_packages(cargo_lock);
    let mut problems = Vec::new();
    for package in wanted.iter().filter(|p| !locked.contains(p)) {
        problems.push(format!(
            "{package} is in Cargo.lock but not in cargo-bazel-lock.json"
        ));
    }
    for package in locked.iter().filter(|p| !wanted.contains(p)) {
        problems.push(format!(
            "{package} is in cargo-bazel-lock.json but not in Cargo.lock"
        ));
    }
    Ok(problems)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_changed_or_missing_file_is_stale() {
        let expected: BTreeMap<String, String> = [
            ("a/BUILD.bazel", "x\n"),
            ("BUILD.bazel", "y\n"),
            ("b/BUILD.bazel", "z\n"),
        ]
        .into_iter()
        .map(|(p, c)| (p.to_owned(), c.to_owned()))
        .collect();
        let on_disk = |path: &str| match path {
            "a/BUILD.bazel" => Some("x\n".to_owned()),
            "BUILD.bazel" => Some("y changed\n".to_owned()),
            _ => None,
        };
        assert_eq!(stale(&expected, &on_disk), ["BUILD.bazel", "b/BUILD.bazel"]);
        assert!(stale(&expected, &|p| expected.get(p).cloned()).is_empty());
    }

    #[test]
    fn a_build_file_the_generator_does_not_write_is_extra() {
        let expected: BTreeMap<String, String> = [("BUILD.bazel", ""), ("a/BUILD.bazel", "")]
            .into_iter()
            .map(|(p, c)| (p.to_owned(), c.to_owned()))
            .collect();
        let on_disk = [
            "BUILD.bazel".to_owned(),
            "removed/BUILD.bazel".to_owned(),
            "a/src/BUILD".to_owned(),
            "a/BUILD.bazel".to_owned(),
        ];
        assert_eq!(
            extra_build_files(&expected, &on_disk),
            ["a/src/BUILD", "removed/BUILD.bazel"]
        );
    }

    #[test]
    fn the_lock_check_names_a_package_only_one_lock_has() {
        let cargo_lock = "[[package]]\nname = \"a\"\nversion = \"1.0.0\"\nsource = \"registry+x\"\n\n\
            [[package]]\nname = \"b\"\nversion = \"2.0.0\"\nsource = \"registry+x\"\n\n\
            [[package]]\nname = \"me\"\nversion = \"0.1.0\"\n";
        let current = r#"{"workspace_members":{"me 0.1.0":"."},"crates":{"a 1.0.0":{},"b 2.0.0":{},"me 0.1.0":{}}}"#;
        assert_eq!(lock_problems(cargo_lock, current), Ok(vec![]));
        let stale = r#"{"workspace_members":{"me 0.1.0":"."},"crates":{"a 1.0.0":{},"b 1.9.0":{},"me 0.1.0":{}}}"#;
        assert_eq!(
            lock_problems(cargo_lock, stale),
            Ok(vec![
                "b 2.0.0 is in Cargo.lock but not in cargo-bazel-lock.json".to_owned(),
                "b 1.9.0 is in cargo-bazel-lock.json but not in Cargo.lock".to_owned(),
            ])
        );
        assert!(lock_problems(cargo_lock, "{}").is_err());
    }
}
