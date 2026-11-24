#![doc = include_str!("../README.md")]

pub mod cfg;
pub mod emit;
pub mod json;
pub mod model;
pub mod sha256;
pub mod starlark;
pub mod worlds;

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
    model::load(&metadata, &|path| {
        std::fs::read_to_string(root.join(path)).map_err(|e| format!("reading {path}: {e}"))
    })
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

/// The manifest files `MODULE.bazel.lock` records a digest of: the root's `Cargo.toml` and `Cargo.lock` and every
/// member's `Cargo.toml`.
pub fn locked_inputs(ws: &Workspace) -> Vec<String> {
    let mut paths = vec!["Cargo.lock".to_owned(), "Cargo.toml".to_owned()];
    paths.extend(ws.packages.iter().map(|p| format!("{}/Cargo.toml", p.dir)));
    paths
}

/// What is wrong with `MODULE.bazel.lock`'s record of the workspace manifests: a recorded digest that differs from the file
/// `read` returns, or a path in `required` with no record. Empty when the lock is current.
pub fn lock_problems(
    lock: &str,
    read: &dyn Fn(&str) -> Option<Vec<u8>>,
    required: &[String],
) -> Vec<String> {
    let mut recorded = BTreeMap::new();
    for line in lock.lines() {
        let Some(rest) = line
            .trim()
            .trim_start_matches('"')
            .strip_prefix("FILE:@@//")
        else {
            continue;
        };
        let rest = rest.trim_end_matches(',').trim_end_matches('"');
        if let Some((path, digest)) = rest.split_once(' ') {
            recorded.insert(path.to_owned(), digest.to_owned());
        }
    }
    let mut problems = Vec::new();
    for (path, digest) in &recorded {
        match read(path) {
            None => problems.push(format!("{path} is recorded but missing")),
            Some(bytes) if sha256::hex_digest(&bytes) != *digest => {
                problems.push(format!("{path} changed since the lock was written"))
            }
            Some(_) => {}
        }
    }
    for path in required {
        if !recorded.contains_key(path) {
            problems.push(format!("{path} is not recorded"));
        }
    }
    problems
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
    fn the_lock_check_names_changed_and_unrecorded_manifests() {
        let digest = sha256::hex_digest(b"[package]\n");
        let lock = format!(
            "  \"FILE:@@//a/Cargo.toml {digest}\",\n  \"FILE:@@//b/Cargo.toml {digest}\"\n  \"FILE:@@rules_rust+//MODULE.bazel 00\"\n"
        );
        let files = |path: &str| match path {
            "a/Cargo.toml" => Some(b"[package]\n".to_vec()),
            "b/Cargo.toml" => Some(b"[package]\nname = \"b\"\n".to_vec()),
            _ => None,
        };
        let required = ["a/Cargo.toml".to_owned(), "c/Cargo.toml".to_owned()];
        assert_eq!(
            lock_problems(&lock, &files, &required),
            [
                "b/Cargo.toml changed since the lock was written",
                "c/Cargo.toml is not recorded"
            ]
        );
        assert!(
            lock_problems(&lock, &|_| Some(b"[package]\n".to_vec()), &required[..1]).is_empty()
        );
    }
}
